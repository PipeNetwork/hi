//! `Harness` glue for `/ladder`: the slash command, starting a ladder (the
//! check that judges it, Jev's start rung), applying each rung's model and
//! effort, feeding [`Ladder::after_turn`], and restoring the session's model
//! and verify command when the ladder ends.
//!
//! A rung's model goes to the live settings only. The session file keeps
//! the model the ladder started from, so a restart mid-ladder resumes on
//! that model with no ladder running.

use crate::command::EffortArg;
use crate::ladder::{
    JEV_SKIP_THRESHOLD, Ladder, LadderCommand, LadderPhase, LadderRung, LadderSettings, LadderStep,
    LadderTurnFacts, MAX_ATTEMPTS_PER_RUNG, MIN_RUNGS_FOR_JEV_SKIP, distinct_rungs,
};
use crate::typesafe::score_needs_high_effort;
use crate::ui::Ui;
use crate::{Harness, ReasoningEffort, TurnCancellation, TurnOutcome};

/// The managed Pipe workflow picks models and reviews work server-side; a
/// ladder on top of it would fight that routing.
const MANAGED_MODEL: &str = "pipe/auto";
/// Stop reason for `/ladder stop`.
const STOPPED_BY_USER: &str = "stopped by /ladder stop";

impl Harness {
    pub fn ladder(&self) -> &Ladder {
        &self.ladder
    }

    pub fn ladder_settings(&self) -> &LadderSettings {
        &self.ladder_settings
    }

    /// `[ladder]` knobs from the config file; the defaults otherwise.
    pub fn set_ladder_settings(&mut self, settings: LadderSettings) {
        self.ladder_settings = settings;
    }

    /// `/ladder [status|stop|<task>]`.
    pub async fn ladder_command(&mut self, arg: &str) -> LadderCommand {
        let arg = arg.trim();
        match arg.to_ascii_lowercase().as_str() {
            "" | "status" => LadderCommand::Message(self.ladder_status_text()),
            "stop" => LadderCommand::Message(self.stop_ladder(STOPPED_BY_USER)),
            _ => match self.begin_ladder(arg).await {
                Ok(notes) => LadderCommand::Start {
                    prompt: arg.to_string(),
                    notes,
                },
                Err(message) => LadderCommand::Message(message),
            },
        }
    }

    /// Start a ladder for `task` and apply its first rung. The caller runs
    /// `task` itself as the first turn. Returns the lines to show first.
    ///
    /// # Errors
    /// A user-facing message: an empty task, a ladder or spec review already
    /// running, `pipe/auto` as the session model or a rung, or no check to
    /// judge attempts with.
    pub async fn begin_ladder(&mut self, task: &str) -> Result<Vec<String>, String> {
        let task = task.trim();
        if task.is_empty() {
            return Err("ladder: give it a task · /ladder <task>".into());
        }
        if self.ladder.is_active() {
            return Err(format!(
                "{} · /ladder stop ends it",
                self.ladder.status_line()
            ));
        }
        if self.review_drive.is_active() {
            return Err("ladder: a spec review is running · /review stop ends it first".into());
        }
        if !self.ladder_settings.invalid_rungs.is_empty() {
            return Err(format!(
                "ladder: fix [ladder] rungs first · {}",
                self.ladder_settings.invalid_rungs.join("; ")
            ));
        }
        let base_model = self.model();
        let (rungs, merged) = self.ladder_rungs(&base_model);
        if base_model == MANAGED_MODEL
            || rungs
                .iter()
                .any(|rung| rung.model.as_deref() == Some(MANAGED_MODEL))
        {
            return Err(format!(
                "ladder: {MANAGED_MODEL} picks models and reviews work server-side · switch to a direct model with /model and keep {MANAGED_MODEL} out of [ladder] rungs"
            ));
        }
        let Some(verify) = self.ladder_verify_command() else {
            return Err(
                "ladder: needs a check to judge each attempt · set /verify <cmd> or [ladder] verify"
                    .into(),
            );
        };
        let (start_rung, start_note) = self.ladder_start_rung(task, rungs.len()).await;
        let mut ladder = Ladder::start(
            task,
            &base_model,
            rungs,
            self.ladder_settings.attempts_per_rung,
            &verify,
            start_rung,
        );
        ladder.start_note = start_note;
        ladder.temporary_verify = self.verify_command.is_none();
        if ladder.temporary_verify {
            self.verify_command = Some(ladder.verify_command.clone());
        }
        self.ladder = ladder;
        self.last_verify_output = None;
        self.apply_ladder_rung();
        let mut notes = vec![self.ladder.status_line()];
        if merged > 0 {
            notes.push(format!(
                "ladder: merged {merged} rung(s) that send the same request as the next one"
            ));
        }
        if let Some(note) = &self.ladder.start_note {
            notes.push(format!("ladder: {note}"));
        }
        Ok(notes)
    }

    /// Feed the turn that just ended into the ladder. Before another attempt
    /// the next rung is applied and its status announced; when the ladder
    /// ends, the session's model and verify command are restored.
    pub fn ladder_next_step(&mut self, outcome: &TurnOutcome, ui: &mut dyn Ui) -> LadderStep {
        let verify_output = self.last_verify_output.take();
        if !self.ladder.is_active() {
            return LadderStep::Idle;
        }
        let step = self.ladder.after_turn(&LadderTurnFacts {
            stop_reason: outcome.stop_reason,
            error: outcome.error.clone(),
            changed_files: outcome.changed_files.clone(),
            verification: outcome.verification.clone(),
            verify_output,
            usage: outcome.usage,
        });
        match &step {
            LadderStep::RunPrompt(_) => {
                self.apply_ladder_rung();
                ui.status(&self.ladder.status_line());
            }
            LadderStep::Verified(_) | LadderStep::Stopped(_) => self.finish_ladder(),
            LadderStep::Idle => {}
        }
        step
    }

    /// End a running ladder early and restore the session: `/ladder stop`,
    /// or a turn that failed before it produced an outcome.
    pub fn stop_ladder(&mut self, reason: &str) -> String {
        match self.ladder.stop(reason) {
            LadderStep::Stopped(summary) => {
                self.finish_ladder();
                summary
            }
            _ => "ladder: nothing to stop".into(),
        }
    }

    /// `/ladder` and `/ladder status`: the running or last ladder, then the
    /// configured rungs when none is running.
    pub fn ladder_status_text(&self) -> String {
        let mut lines = Vec::new();
        if self.ladder.phase == LadderPhase::Idle {
            lines.push(
                "ladder: not running · /ladder <task> runs the task on the cheapest rung and climbs only when the check fails"
                    .to_string(),
            );
        } else {
            lines.push(self.ladder.status_line());
            lines.extend(self.ladder.report_lines());
        }
        if self.ladder.is_active() {
            return lines.join("\n");
        }
        let base_model = self.model();
        let rungs: Vec<String> = self
            .ladder_rungs(&base_model)
            .0
            .iter()
            .map(|rung| rung.label(&base_model))
            .collect();
        lines.push(format!("  rungs: {}", rungs.join(" → ")));
        for invalid in &self.ladder_settings.invalid_rungs {
            lines.push(format!("  invalid: {invalid}"));
        }
        let verify = self
            .ladder_verify_command()
            .map(|command| format!("`{command}`"))
            .unwrap_or_else(|| "none (set /verify <cmd> or [ladder] verify)".into());
        lines.push(format!(
            "  attempts per rung: {} · verify: {verify}",
            self.ladder_settings
                .attempts_per_rung
                .clamp(1, MAX_ATTEMPTS_PER_RUNG)
        ));
        if self.ladder_settings.jev_start && self.typesafe.is_enabled() {
            lines
                .push("  Jev start: on · a task Jev scores as hard skips the cheapest rung".into());
        }
        lines.join("\n")
    }

    /// The current rung's reasoning effort while a ladder runs, ahead of a
    /// pinned `/effort` and Jev's per-turn effort. `Some(None)` is a rung set
    /// to `off`.
    pub(crate) fn ladder_reasoning_effort(&self) -> Option<Option<ReasoningEffort>> {
        self.ladder.rung_effort().map(|effort| match effort {
            EffortArg::Level(level) => Some(level),
            EffortArg::Off => None,
        })
    }

    /// A pinned `/effort` or a running rung fixes the effort, so Jev's
    /// turn-start hint is not asked for.
    pub(crate) fn effort_fixed(&self) -> bool {
        self.live.effort_pinned() || self.ladder.rung_effort().is_some()
    }

    /// A ladder attempt is judged by its check, not by how the model stopped:
    /// after a completion-policy stop with edits, run the check anyway.
    pub(crate) async fn ladder_check_on_stop(
        &mut self,
        mutated: bool,
        ui: &mut dyn Ui,
        cancel: &TurnCancellation,
    ) -> Option<String> {
        if mutated && self.ladder.is_active() {
            self.run_verify(ui, cancel).await
        } else {
            None
        }
    }

    /// The model the session file records: while a ladder runs, the model
    /// it started from rather than the current rung's.
    pub(crate) fn recorded_model(&self) -> String {
        if self.ladder.is_active() {
            self.ladder.base_model.clone()
        } else {
            self.model()
        }
    }

    /// The verify command the session file records: never one the ladder
    /// supplied for its own run.
    pub(crate) fn recorded_verify_command(&self) -> Option<String> {
        if self.ladder.is_active() && self.ladder.temporary_verify {
            None
        } else {
            self.verify_command.clone()
        }
    }

    /// The rungs a ladder from `base_model` would run, and how many
    /// configured rungs merged into their neighbour.
    fn ladder_rungs(&self, base_model: &str) -> (Vec<LadderRung>, usize) {
        distinct_rungs(self.ladder_settings.resolved_rungs(), base_model)
    }

    /// `/verify` wins; `[ladder] verify` (or the frontend's detected
    /// pipeline) otherwise.
    fn ladder_verify_command(&self) -> Option<String> {
        self.verify_command
            .clone()
            .or_else(|| self.ladder_settings.verify.clone())
            .map(|command| command.trim().to_string())
            .filter(|command| !command.is_empty())
    }

    fn apply_ladder_rung(&mut self) {
        if let Some(model) = self.ladder.rung_model().map(str::to_string) {
            self.switch_live_model(model);
        }
    }

    fn finish_ladder(&mut self) {
        let base_model = self.ladder.base_model.clone();
        if !base_model.is_empty() {
            self.switch_live_model(base_model);
        }
        if self.ladder.temporary_verify {
            self.verify_command = None;
            self.ladder.temporary_verify = false;
        }
    }

    /// Switch the live model without recording it in the session.
    fn switch_live_model(&mut self, model: String) {
        if self.model() != model {
            self.live.set_model(model);
            self.apply_cached_window();
        }
    }

    /// Rung to start on: one up when Jev scores the task as hard, else the
    /// cheapest. Fails open to the cheapest rung.
    async fn ladder_start_rung(&self, task: &str, rungs: usize) -> (usize, Option<String>) {
        if !self.ladder_settings.jev_start
            || rungs < MIN_RUNGS_FOR_JEV_SKIP
            || !self.typesafe.is_enabled()
        {
            return (0, None);
        }
        let Some(client) = self.typesafe.client() else {
            return (0, None);
        };
        match score_needs_high_effort(&client, task).await {
            Some(score) if score >= JEV_SKIP_THRESHOLD => (
                1,
                Some(format!(
                    "Jev scored the task {score:.2} for high effort, so it starts on rung 2"
                )),
            ),
            _ => (0, None),
        }
    }
}

#[cfg(test)]
#[path = "ladder_harness_tests.rs"]
mod tests;
