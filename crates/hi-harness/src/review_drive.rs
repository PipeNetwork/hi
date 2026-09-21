//! `/review` drive: the audit -> fix -> re-audit state machine and its
//! `Harness` glue.
//!
//! The state machine itself ([`ReviewDrive::next_step`]) is pure so every
//! transition can be unit-tested without a model. The harness methods below
//! feed it facts from the turn that just ended, seed the fix checklist, arm
//! the forced intent for the next turn, and persist the drive.

use hi_ai::Role;
use hi_tools::{PlanStatus, PlanStep};
use serde::{Deserialize, Serialize};

use crate::completion::Intent;
use crate::review::{
    ChecklistItem, DEFAULT_REVIEW_PASSES, Finding, InputKind, REVIEW_PREFIX, ReviewAction,
    ReviewArgs, ReviewInputs, ReviewVerdict, audit_prompt, check_citations, chunk_label,
    discover_inputs, fingerprint, fix_prompt, format_reask_prompt, parse_checklist, reaudit_prompt,
};
use crate::ui::Ui;
use crate::{Harness, TurnOutcome, TurnStopReason};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewPhase {
    #[default]
    Idle,
    Audit,
    Fix,
    Reaudit,
    Done,
    Stopped,
}

impl ReviewPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Audit => "audit",
            Self::Fix => "fix",
            Self::Reaudit => "re-audit",
            Self::Done => "done",
            Self::Stopped => "stopped",
        }
    }
}

/// Persisted state of one `/review` loop.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewDrive {
    pub phase: ReviewPhase,
    /// Fix passes started so far.
    pub pass: u32,
    pub max_passes: u32,
    pub audit_only: bool,
    /// Set when a review turn was cancelled; `/review` resumes the phase.
    pub paused: bool,
    pub inputs: ReviewInputs,
    /// Checklist rows from the plan/spec files, in order. The re-audit and
    /// the format re-ask list them so the coverage rows have a fixed item
    /// column (the last verdict's rows take precedence once there is one);
    /// the unchecked rows decide which findings are feature gaps.
    #[serde(default)]
    pub checklist_items: Vec<ChecklistItem>,
    #[serde(default)]
    pub last_verdict: Option<ReviewVerdict>,
    /// Fingerprint of the P0/P1 set that started the current fix pass.
    #[serde(default)]
    pub last_fingerprint: Option<String>,
    #[serde(default)]
    pub format_retry_used: bool,
    /// Blocking findings handed to the current fix pass.
    #[serde(default)]
    pub findings_to_fix: Vec<Finding>,
    /// Files changed by the current fix pass (scopes the re-audit).
    #[serde(default)]
    pub changed_files: Vec<String>,
    /// Files changed by every fix pass so far.
    #[serde(default)]
    pub all_changed_files: Vec<String>,
    /// Files an audit or re-audit turn changed. Those turns are read-only
    /// and write tools are denied, but a shell redirection cannot be proven
    /// mutating up front; anything that slips through is reported here so
    /// the verdict is read knowing the tree moved under it.
    #[serde(default)]
    pub audit_changed_files: Vec<String>,
    #[serde(default)]
    pub unverified_citations: Vec<String>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    /// Completion-policy error that ended the last fix turn (plan stall,
    /// unverified stop, tool storm). The re-audit judges the tree it left.
    #[serde(default)]
    pub fix_turn_error: Option<String>,
    /// Chunked audit (`all`): index into `inputs.chunks` of the chunk being
    /// audited; equals the chunk count once the audit phase is over.
    #[serde(default)]
    pub chunk: usize,
    /// Chunked audit: the verdicts of the chunks done so far, merged. Becomes
    /// `last_verdict` when the last chunk is in.
    #[serde(default)]
    pub chunk_verdict: Option<ReviewVerdict>,
}

impl Default for ReviewDrive {
    fn default() -> Self {
        Self {
            phase: ReviewPhase::Idle,
            pass: 0,
            max_passes: DEFAULT_REVIEW_PASSES,
            audit_only: false,
            paused: false,
            inputs: ReviewInputs::default(),
            checklist_items: Vec::new(),
            last_verdict: None,
            last_fingerprint: None,
            format_retry_used: false,
            findings_to_fix: Vec::new(),
            changed_files: Vec::new(),
            all_changed_files: Vec::new(),
            audit_changed_files: Vec::new(),
            unverified_citations: Vec::new(),
            stop_reason: None,
            fix_turn_error: None,
            chunk: 0,
            chunk_verdict: None,
        }
    }
}

/// What the frontend does after [`Harness::review_next_step`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReviewStep {
    /// Run another turn with this prompt.
    RunPrompt(String),
    /// The loop ended with a verdict; text for the user.
    Done(String),
    /// The loop ended early; reason for the user.
    Stopped(String),
    /// A review turn was cancelled; `/review` resumes it.
    Paused(String),
    /// No review is running.
    Idle,
}

/// What `/review …` asks the frontend to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReviewCommand {
    /// Print `notes`, then run a turn with `prompt`.
    Start {
        prompt: String,
        notes: Vec<String>,
    },
    Message(String),
}

/// Facts about the turn that just ended, as the state machine sees them.
#[derive(Clone, Debug)]
pub struct ReviewTurnFacts {
    pub stop_reason: TurnStopReason,
    pub error: Option<String>,
    pub changed_files: Vec<String>,
    /// Parsed from the turn's assistant text, citations already checked.
    pub verdict: Option<ReviewVerdict>,
    pub unverified_citations: Vec<String>,
}

impl ReviewDrive {
    pub fn start(inputs: ReviewInputs, audit_only: bool, max_passes: u32) -> Self {
        Self {
            phase: ReviewPhase::Audit,
            max_passes: max_passes.max(1),
            audit_only,
            inputs,
            ..Self::default()
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.phase,
            ReviewPhase::Audit | ReviewPhase::Fix | ReviewPhase::Reaudit
        )
    }

    /// Item column for the coverage rows: the last verdict's items (the
    /// model's own naming) once there is one, else the checklist rows.
    pub fn coverage_items(&self) -> Vec<String> {
        let from_verdict: Vec<String> = self
            .last_verdict
            .as_ref()
            .map(|verdict| {
                verdict
                    .coverage
                    .iter()
                    .map(|row| row.item.clone())
                    .collect()
            })
            .unwrap_or_default();
        if from_verdict.is_empty() {
            self.checklist_items
                .iter()
                .map(|item| item.text.clone())
                .collect()
        } else {
            from_verdict
        }
    }

    /// `- [ ]` rows: items the plan itself says are not built yet.
    pub fn unchecked_items(&self) -> Vec<String> {
        self.checklist_items
            .iter()
            .filter(|item| item.checked == Some(false))
            .map(|item| item.text.clone())
            .collect()
    }

    /// Intent for a review-injected prompt while the drive is active. Audit
    /// prompts mention fixing, so the heuristic classifier would demand
    /// edits mid-audit; this pins Review/Fix by phase instead. Also covers
    /// resumed turns, where `forced_intent` was never re-armed.
    pub(crate) fn intent_for_prompt(&self, prompt: &str) -> Option<Intent> {
        if !self.is_active() || !prompt.trim_start().starts_with(REVIEW_PREFIX) {
            return None;
        }
        Some(match self.phase {
            ReviewPhase::Fix => Intent::Fix,
            _ => Intent::Review,
        })
    }

    /// `spec review · fix pass 1/3 · 2 P0/P1 · plan.md + docs/spec.md`; a
    /// narrowed audit adds its scope: `… · audit only · 47 uncommitted files
    /// (git status) · no plan/spec (defects only)`, `… · audit · chunk 2/7
    /// · crates/hi-cli · …`.
    pub fn status_line(&self) -> String {
        let mut parts = vec!["spec review".to_string()];
        let mut chunk_shown = false;
        match self.phase {
            ReviewPhase::Idle => parts.push("idle".into()),
            ReviewPhase::Audit => {
                parts.push(if self.audit_only {
                    "audit only".to_string()
                } else {
                    format!("audit · up to {} fix passes", self.max_passes)
                });
                if let Some(chunk) = self.inputs.chunks.get(self.chunk) {
                    parts.push(format!(
                        "chunk {}/{} · {}",
                        self.chunk + 1,
                        self.inputs.chunks.len(),
                        chunk_label(chunk)
                    ));
                    chunk_shown = true;
                }
            }
            ReviewPhase::Fix => parts.push(format!(
                "fix pass {}/{} · {} P0/P1",
                self.pass,
                self.max_passes,
                self.findings_to_fix.len()
            )),
            ReviewPhase::Reaudit => {
                parts.push(format!(
                    "re-audit after pass {}/{}",
                    self.pass, self.max_passes
                ));
                if let Some(error) = &self.fix_turn_error {
                    parts.push(format!("fix turn ended early: {error}"));
                }
            }
            ReviewPhase::Done => parts.push("done".into()),
            ReviewPhase::Stopped => parts.push(format!(
                "stopped: {}",
                self.stop_reason.as_deref().unwrap_or("stopped")
            )),
        }
        if self.paused {
            parts.push("paused".into());
        }
        if self.phase != ReviewPhase::Idle {
            if let Some(scope) = self.inputs.scope_summary()
                && !chunk_shown
            {
                parts.push(scope);
            }
            parts.push(self.inputs.summary());
        }
        parts.join(" · ")
    }

    /// Prompt that re-runs the current phase after a pause or a reload.
    pub fn resume_prompt(&self) -> Option<String> {
        match self.phase {
            ReviewPhase::Audit => {
                Some(audit_prompt(&self.inputs, &[], self.chunk, self.max_passes))
            }
            ReviewPhase::Fix => Some(fix_prompt(
                &self.findings_to_fix,
                self.pass,
                self.max_passes,
            )),
            ReviewPhase::Reaudit => Some(reaudit_prompt(
                &self.inputs,
                &self.coverage_items(),
                &self.changed_files,
                &self.findings_to_fix,
                self.pass,
                self.max_passes,
            )),
            _ => None,
        }
    }

    fn stop(&mut self, reason: impl Into<String>) -> ReviewStep {
        let reason = reason.into();
        self.phase = ReviewPhase::Stopped;
        self.paused = false;
        self.stop_reason = Some(reason.clone());
        ReviewStep::Stopped(format!("spec review stopped: {reason}"))
    }

    /// Advance after a turn. Pure: no I/O, no model.
    pub fn next_step(&mut self, facts: &ReviewTurnFacts) -> ReviewStep {
        if !self.is_active() {
            return ReviewStep::Idle;
        }
        if matches!(self.phase, ReviewPhase::Audit | ReviewPhase::Reaudit) {
            for file in &facts.changed_files {
                if !self.audit_changed_files.contains(file) {
                    self.audit_changed_files.push(file.clone());
                }
            }
        }
        match facts.stop_reason {
            TurnStopReason::Cancelled => {
                self.paused = true;
                return ReviewStep::Paused(format!(
                    "spec review paused during {} · Enter on /review resumes, /review stop ends it",
                    self.phase.label()
                ));
            }
            TurnStopReason::Error => {
                let error = facts
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown error".into());
                if self.phase != ReviewPhase::Fix {
                    return self.stop(format!("{} turn failed: {error}", self.phase.label()));
                }
                // A fix turn ending on a completion-policy error (plan
                // stall because the defect was already fixed, unverified
                // stop, tool storm) still leaves a tree worth judging. The
                // re-audit decides; its fingerprint and the pass cap bound
                // the loop.
                self.fix_turn_error = Some(error);
            }
            TurnStopReason::Completed => {}
        }
        match self.phase {
            ReviewPhase::Audit | ReviewPhase::Reaudit => self.after_audit(facts),
            ReviewPhase::Fix => {
                for file in &facts.changed_files {
                    if !self.changed_files.contains(file) {
                        self.changed_files.push(file.clone());
                    }
                    if !self.all_changed_files.contains(file) {
                        self.all_changed_files.push(file.clone());
                    }
                }
                self.phase = ReviewPhase::Reaudit;
                ReviewStep::RunPrompt(reaudit_prompt(
                    &self.inputs,
                    &self.coverage_items(),
                    &self.changed_files,
                    &self.findings_to_fix,
                    self.pass,
                    self.max_passes,
                ))
            }
            _ => ReviewStep::Idle,
        }
    }

    fn after_audit(&mut self, facts: &ReviewTurnFacts) -> ReviewStep {
        let Some(verdict) = facts.verdict.clone() else {
            if !self.format_retry_used {
                self.format_retry_used = true;
                let prior = if self.phase == ReviewPhase::Reaudit {
                    self.findings_to_fix.clone()
                } else {
                    Vec::new()
                };
                return ReviewStep::RunPrompt(format_reask_prompt(
                    &self.coverage_items(),
                    &prior,
                    self.inputs.defects_only(),
                ));
            }
            return self.stop("no parseable <review> block after one format re-ask");
        };
        self.format_retry_used = false;
        if self.phase == ReviewPhase::Audit && self.chunk > 0 {
            // Later chunks add to the first chunk's unverified citations.
            for path in &facts.unverified_citations {
                if !self.unverified_citations.contains(path) {
                    self.unverified_citations.push(path.clone());
                }
            }
        } else {
            self.unverified_citations = facts.unverified_citations.clone();
        }
        let mut verdict = match self.absorb_chunk(verdict) {
            Ok(verdict) => verdict,
            Err(next_chunk) => return ReviewStep::RunPrompt(next_chunk),
        };
        verdict.mark_feature_gaps(&self.unchecked_items());
        let blocking = verdict.blocking_findings();
        let was_reaudit = self.phase == ReviewPhase::Reaudit;
        self.last_verdict = Some(verdict);
        self.fix_turn_error = None;
        if blocking.is_empty() || self.audit_only {
            self.phase = ReviewPhase::Done;
            self.paused = false;
            return ReviewStep::Done(self.done_summary());
        }
        let print = fingerprint(&blocking);
        if was_reaudit && self.last_fingerprint.as_deref() == Some(print.as_str()) {
            return self.stop(format!(
                "no progress: the same {} P0/P1 finding(s) remain after fix pass {}",
                blocking.len(),
                self.pass
            ));
        }
        if self.pass >= self.max_passes {
            return self.stop(format!(
                "pass cap reached: {} P0/P1 finding(s) still open after {} fix pass(es)",
                blocking.len(),
                self.pass
            ));
        }
        self.last_fingerprint = Some(print);
        self.pass += 1;
        self.phase = ReviewPhase::Fix;
        self.findings_to_fix = blocking;
        self.changed_files.clear();
        ReviewStep::RunPrompt(fix_prompt(
            &self.findings_to_fix,
            self.pass,
            self.max_passes,
        ))
    }

    /// In a chunked audit, fold this chunk's verdict into the running one
    /// and move to the next chunk (`Err(prompt)`); once the last chunk is
    /// in, the merged verdict is the audit's (`Ok`). Not chunked: `Ok` as is.
    fn absorb_chunk(&mut self, verdict: ReviewVerdict) -> Result<ReviewVerdict, String> {
        if self.phase != ReviewPhase::Audit || self.inputs.chunks.is_empty() {
            return Ok(verdict);
        }
        let merged = match self.chunk_verdict.take() {
            Some(mut running) => {
                running.merge(verdict);
                running
            }
            None => verdict,
        };
        self.chunk += 1;
        if self.chunk < self.inputs.chunks.len() {
            self.chunk_verdict = Some(merged);
            return Err(audit_prompt(&self.inputs, &[], self.chunk, self.max_passes));
        }
        Ok(merged)
    }
}

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
                match self.begin_review(&args.paths, args.all, audit_only, DEFAULT_REVIEW_PASSES) {
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
        crate::review_scope::resolve_target(&self.workspace_root, &mut inputs, all)?;
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
        let prompt = audit_prompt(&inputs, &checklists, 0, max_passes.max(1));
        self.review_drive = ReviewDrive::start(inputs, audit_only, max_passes);
        self.review_drive.checklist_items = checklists
            .into_iter()
            .flat_map(|(_, items)| items)
            .collect();
        self.forced_intent = Some(Intent::Review);
        self.persist_review_drive();
        Ok(prompt)
    }

    fn resume_review(&mut self) -> ReviewCommand {
        let Some(prompt) = self.review_drive.resume_prompt() else {
            return ReviewCommand::Message(self.review_status_text());
        };
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
        let step = self.review_drive.next_step(&facts);
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

#[cfg(test)]
#[path = "review_drive_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "review_drive_scope_tests.rs"]
mod scope_tests;
