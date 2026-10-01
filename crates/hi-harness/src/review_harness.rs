//! `Harness` glue for the `/review` drive: the slash command, the facts fed
//! to [`ReviewDrive::next_step`] after each turn (verdict, changed files,
//! files read, files in scope), the fix checklist, the forced intent, the
//! seed run before audit turns, and persistence.

use hi_ai::Role;
use hi_tools::{PlanStatus, PlanStep, ToolOutcome, ToolStatus, TruncationState};

use crate::check_run::CheckRun;
use crate::completion::Intent;
use crate::review::{
    ChecklistItem, Finding, InputKind, ReviewAction, ReviewArgs, ReviewSettings, ReviewVerdict,
    VerifySeed, audit_prompt, check_citations, discover_inputs, parse_checklist,
    scope_files_section,
};
use crate::review_drive::{ReviewCommand, ReviewDrive, ReviewPhase, ReviewStep, ReviewTurnFacts};
use crate::review_reads::files_read;
use crate::review_scope::{chunk_files, dir_files};
use crate::ui::Ui;
use crate::{Harness, TurnCancellation, TurnOutcome};

fn plan_steps_for(findings: &[Finding]) -> Vec<PlanStep> {
    findings
        .iter()
        .map(|finding| PlanStep {
            title: finding.label(),
            status: PlanStatus::Pending,
        })
        .collect()
}

impl Harness {
    pub fn review_drive(&self) -> &ReviewDrive {
        &self.review_drive
    }

    pub fn review_settings(&self) -> &ReviewSettings {
        &self.review_settings
    }

    /// `[review]` knobs from the config file; the defaults otherwise.
    pub fn set_review_settings(&mut self, settings: ReviewSettings) {
        self.review_settings = settings;
    }

    /// `/review [audit|status|stop] [all] [path...]`.
    pub fn review_command(&mut self, arg: &str) -> ReviewCommand {
        let args = ReviewArgs::parse(arg);
        match args.action {
            ReviewAction::Status => ReviewCommand::Message(self.review_status_text()),
            ReviewAction::Stop => ReviewCommand::Message(self.stop_review()),
            ReviewAction::Run
                if args.paths.is_empty() && !args.all && self.review_drive.is_active() =>
            {
                self.resume_review()
            }
            ReviewAction::Run | ReviewAction::Audit => {
                let audit_only = args.action == ReviewAction::Audit;
                let passes = self.review_settings.passes;
                match self.begin_review(&args.paths, args.all, audit_only, passes) {
                    Ok(prompt) => ReviewCommand::Start {
                        prompt,
                        notes: self.review_start_notes(),
                    },
                    Err(message) => ReviewCommand::Message(message),
                }
            }
        }
    }

    fn review_start_notes(&self) -> Vec<String> {
        let mut notes = vec![self.review_drive.status_line()];
        if let Some(notice) = &self.review_drive.inputs.notice {
            notes.push(format!("spec review: {notice}"));
        }
        if let Some(command) = self.review_seed_command() {
            notes.push(format!(
                "spec review: `{command}` runs before each audit turn; its failures are seeded as candidate defects"
            ));
        }
        notes
    }

    /// Discover inputs, decide the scope ([`crate::review_scope`]), start a
    /// drive, and return the first audit prompt. `Err` carries a user-facing
    /// message: a missing explicit path, or a large workspace with nothing
    /// to audit against and no recent work.
    pub fn begin_review(
        &mut self,
        paths: &[String],
        all: bool,
        audit_only: bool,
        max_passes: u32,
    ) -> Result<String, String> {
        let mut inputs = discover_inputs(&self.workspace_root, paths);
        if !inputs.missing.is_empty() {
            return Err(format!(
                "spec review: {}",
                inputs.notice.unwrap_or_else(|| "missing input".into())
            ));
        }
        crate::review_scope::resolve_target(
            &self.workspace_root,
            &mut inputs,
            all,
            &self.review_settings,
        )?;
        inputs.root = Some(self.workspace_root.display().to_string());
        // A README's lists are install steps and tips, not a checklist of
        // claims; its feature claims become coverage rows via the prompt.
        let checklists: Vec<(String, Vec<ChecklistItem>)> = inputs
            .files
            .iter()
            .filter(|input| input.kind != InputKind::Readme)
            .map(|input| {
                let body = std::fs::read_to_string(self.workspace_root.join(&input.path))
                    .unwrap_or_default();
                (input.path.clone(), parse_checklist(&body))
            })
            .collect();
        let prompt = audit_prompt(&inputs, &checklists, 0, max_passes.max(1), None);
        self.review_drive = ReviewDrive::start(inputs, audit_only, max_passes);
        self.review_drive.checklist_items = checklists
            .into_iter()
            .flat_map(|(_, items)| items)
            .collect();
        self.forced_intent = Some(Intent::Review);
        self.persist_review_drive();
        Ok(self.with_scope_files(prompt))
    }

    fn resume_review(&mut self) -> ReviewCommand {
        let Some(prompt) = self.review_drive.resume_prompt() else {
            return ReviewCommand::Message(self.review_status_text());
        };
        let prompt = self.with_scope_files(prompt);
        self.review_drive.paused = false;
        if self.review_drive.phase == ReviewPhase::Fix
            && !crate::completion::plan_is_open(&self.plan)
        {
            self.seed_plan(plan_steps_for(&self.review_drive.findings_to_fix));
        }
        self.forced_intent = self.review_drive.intent_for_prompt(&prompt);
        self.persist_review_drive();
        ReviewCommand::Start {
            prompt,
            notes: vec![format!("resuming {}", self.review_drive.status_line())],
        }
    }

    /// Feed the turn that just ended into the drive. Returns what to do next;
    /// seeds the fix checklist and arms the forced intent when the answer is
    /// another turn. The phase status line is announced by that turn when it
    /// starts (`review_turn_status`), so it is not repeated here.
    pub fn review_next_step(&mut self, outcome: &TurnOutcome, ui: &mut dyn Ui) -> ReviewStep {
        if !self.review_drive.is_active() || self.review_drive.paused {
            return ReviewStep::Idle;
        }
        let facts = self.review_turn_facts(outcome);
        let step = match self.review_drive.next_step(&facts) {
            ReviewStep::RunPrompt(prompt) => ReviewStep::RunPrompt(self.with_scope_files(prompt)),
            step => step,
        };
        match &step {
            ReviewStep::RunPrompt(prompt) => {
                if self.review_drive.phase == ReviewPhase::Fix {
                    self.seed_plan(plan_steps_for(&self.review_drive.findings_to_fix));
                    ui.plan(&self.plan);
                }
                self.forced_intent = self.review_drive.intent_for_prompt(prompt);
            }
            ReviewStep::Paused(_) => ui.suggested_prompt("/review"),
            ReviewStep::Done(_) | ReviewStep::Stopped(_) => {
                if self.drop_review_checklist() {
                    ui.plan(&[]);
                }
            }
            ReviewStep::Idle => {}
        }
        self.persist_review_drive();
        step
    }

    /// The checklist seeded for the fix passes belongs to the review. Once
    /// the loop ends, the report carries what is still open; a leftover
    /// `[P0]` step would otherwise keep the plan drive nagging and make the
    /// completion policy demand edits on the user's next prompt.
    fn drop_review_checklist(&mut self) -> bool {
        if self.review_drive.pass == 0 || self.plan.is_empty() {
            return false;
        }
        self.seed_plan(Vec::new());
        true
    }

    fn review_turn_facts(&self, outcome: &TurnOutcome) -> ReviewTurnFacts {
        let text = self.last_turn_assistant_text();
        let mut verdict = ReviewVerdict::parse(&text);
        let unverified = verdict
            .as_mut()
            .map(|verdict| check_citations(&self.workspace_root, verdict))
            .unwrap_or_default();
        ReviewTurnFacts {
            stop_reason: outcome.stop_reason,
            error: outcome.error.clone(),
            changed_files: outcome.changed_files.clone(),
            verdict,
            unverified_citations: unverified,
            read_files: self.review_read_files.clone(),
            scope_files: self.review_scope_files(),
        }
    }

    /// The files under audit in the current turn: the git scope (or its
    /// current chunk), the current chunk's directories, the explicit scope
    /// directories, or, for a defects-only audit, the whole (small)
    /// workspace. `None` for a spec audit of the whole tree, whose
    /// deliverable is the coverage rows, not a read of every file.
    pub(crate) fn review_scope_files(&self) -> Option<Vec<String>> {
        let drive = &self.review_drive;
        if let Some(git) = &drive.inputs.git {
            return Some(git.chunk_files(drive.chunk));
        }
        let exclude = &drive.inputs.exclude;
        if let Some(chunk) = drive.inputs.chunks.get(drive.chunk) {
            return Some(chunk_files(&self.workspace_root, chunk, exclude));
        }
        if !drive.inputs.scope.is_empty() {
            let mut files = Vec::new();
            for dir in &drive.inputs.scope {
                for file in dir_files(&self.workspace_root, dir, exclude) {
                    if !files.contains(&file) {
                        files.push(file);
                    }
                }
            }
            return Some(files);
        }
        drive
            .inputs
            .defects_only()
            .then(|| dir_files(&self.workspace_root, ".", exclude))
    }

    /// An audit prompt for a non-git scope lists the files the harness
    /// resolved for it (the prompt builders do not walk the tree).
    fn with_scope_files(&self, mut prompt: String) -> String {
        if self.review_drive.phase != ReviewPhase::Audit
            || self.review_drive.inputs.git.is_some()
            || !prompt.starts_with(crate::review::REVIEW_PREFIX)
            || prompt.contains("Format re-ask")
            || prompt.contains("Unread files (read every one)")
        {
            return prompt;
        }
        if let Some(files) = self.review_scope_files() {
            prompt.push_str(&scope_files_section(&files));
        }
        prompt
    }

    /// The command the turn loop runs before an audit or re-audit turn, when
    /// the drive asks for one ([`ReviewDrive::pending_seed`]).
    pub(crate) fn review_seed_command(&self) -> Option<String> {
        self.review_settings
            .seed_command(&self.workspace_root, self.verify_command.as_deref())
    }

    /// When `input` starts a review audit or re-audit turn that asked for a
    /// seed run, run the seed command and return the prompt section that
    /// reports it. The drive records the run for the report and for the
    /// synthetic finding when the model ignores a failure.
    pub(crate) async fn review_seed_section(
        &mut self,
        input: &str,
        resume: bool,
        ui: &mut dyn Ui,
        cancel: &TurnCancellation,
    ) -> Option<String> {
        if resume
            || !self.review_drive.pending_seed
            || self.review_drive.intent_for_prompt(input) != Some(Intent::Review)
        {
            return None;
        }
        let command = self.review_seed_command()?;
        let CheckRun::Finished { passed, output } =
            self.run_check(&command, "seed", ui, cancel).await?
        else {
            return None;
        };
        self.record_review_seed(&command, passed, &output);
        Some(
            self.review_drive
                .verify
                .as_ref()
                .map(|seed| seed.prompt_section())
                .unwrap_or_default(),
        )
    }

    /// Record a seed run and clear the request for one.
    pub(crate) fn record_review_seed(&mut self, command: &str, passed: bool, output: &str) {
        let before = self.review_drive.seed_label();
        self.review_drive.verify = Some(VerifySeed::from_output(
            &self.workspace_root,
            command,
            passed,
            output,
            &before,
        ));
        self.review_drive.pending_seed = false;
        self.persist_review_drive();
    }

    /// Add what a tool call of an audit turn read to the turn's ledger
    /// ([`crate::review_reads`]). A denied or failed call (a `read` of too
    /// many paths, a missing file) read nothing.
    pub(crate) fn note_review_reads(&mut self, name: &str, arguments: &str, outcome: &ToolOutcome) {
        if outcome.status != ToolStatus::Succeeded {
            return;
        }
        let complete = matches!(outcome.truncation, TruncationState::Complete);
        for path in files_read(
            &self.workspace_root,
            name,
            arguments,
            &outcome.content,
            complete,
        ) {
            if !self.review_read_files.contains(&path) {
                self.review_read_files.push(path);
            }
        }
    }

    /// Assistant text since the last user line, joined. The verdict block is
    /// normally in the final reply, but a model may emit it a round early.
    fn last_turn_assistant_text(&self) -> String {
        let start = self
            .messages
            .iter()
            .rposition(|message| message.role == Role::User)
            .map(|index| index + 1)
            .unwrap_or(0);
        self.messages[start..]
            .iter()
            .filter(|message| message.role == Role::Assistant)
            .map(|message| message.text())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn review_status_text(&self) -> String {
        if self.review_drive.phase == ReviewPhase::Idle {
            return "spec review: not running · /review starts one (/review audit reports only; /review audit all covers the whole repo in chunks)"
                .into();
        }
        self.review_drive.status_report().join("\n")
    }

    /// End the loop and drop a checklist seeded for the current fix pass.
    pub fn stop_review(&mut self) -> String {
        if !self.review_drive.is_active() {
            return format!(
                "spec review: nothing to stop · {}",
                self.review_drive.status_line()
            );
        }
        self.drop_review_checklist();
        let step = self.review_drive.stop("stopped by /review stop");
        self.forced_intent = None;
        self.persist_review_drive();
        match step {
            ReviewStep::Stopped(text) => text,
            _ => "spec review stopped".into(),
        }
    }

    /// Status line for the turn that is about to run, when it is a review turn.
    pub(crate) fn review_turn_status(&self, prompt: &str) -> Option<String> {
        self.review_drive
            .intent_for_prompt(prompt)
            .map(|_| self.review_drive.status_line())
    }

    /// Replace the checklist (persisted). Empty clears it.
    pub(crate) fn seed_plan(&mut self, steps: Vec<PlanStep>) {
        self.plan = steps;
        if let Some(session) = &mut self.session {
            let _ = session.record_plan(&self.plan);
        }
    }

    pub(crate) fn persist_review_drive(&mut self) {
        if let Some(session) = &mut self.session {
            let _ = session.record_review_drive(&self.review_drive);
        }
    }

    /// Findings still open, for headless exit codes.
    pub fn review_open_blocking(&self) -> Vec<Finding> {
        self.review_drive.open_blocking()
    }
}
