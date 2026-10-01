//! `/review` drive: the audit -> fix -> re-audit state machine.
//!
//! [`ReviewDrive::next_step`] is pure so every transition can be
//! unit-tested without a model. The `Harness` glue that feeds it facts from
//! the turn that just ended, seeds the fix checklist, arms the forced
//! intent, and persists the drive is in [`crate::review_harness`]; the
//! report it renders is in [`crate::review_report`].

use serde::{Deserialize, Serialize};

use crate::TurnStopReason;
use crate::completion::Intent;
use crate::review::{
    ChecklistItem, DEFAULT_REVIEW_PASSES, Finding, REVIEW_PREFIX, ReviewInputs, ReviewVerdict,
    VerifySeed, audit_prompt, chunk_label, continuation_prompt, fix_prompt, format_reask_prompt,
    reaudit_prompt, same_finding_set, uncited_changes,
};
use crate::review_reads::unread_files;

/// Continuation turns per chunk when a defects-only audit leaves files in
/// scope unread. What is still unread after them makes the audit partial.
pub const MAX_CONTINUATIONS: u32 = 2;

/// Unread files kept by name on the drive (and in the report); the count
/// is exact.
const UNREAD_LISTED: usize = 100;

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
    /// The P0/P1 set that started the current fix pass; a re-audit that
    /// reports the same set (same places, however worded) is no progress.
    #[serde(default)]
    pub last_blocking: Vec<Finding>,
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
    /// Files a fix pass changed that none of its findings cited (tests and
    /// lockfiles excepted). The re-audit is pointed at them; the report
    /// warns.
    #[serde(default)]
    pub uncited_changes: Vec<String>,
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
    /// Chunked audit: index into `inputs.chunks` of the chunk being
    /// audited; equals the chunk count once the audit phase is over.
    #[serde(default)]
    pub chunk: usize,
    /// The verdicts of this audit's turns so far (chunks and
    /// continuations), merged. Becomes `last_verdict` when the audit ends.
    #[serde(default)]
    pub chunk_verdict: Option<ReviewVerdict>,
    /// Continuation turns spent on the current chunk.
    #[serde(default)]
    pub continuations: u32,
    /// Files the audit turns read (workspace-relative), every turn.
    #[serde(default)]
    pub read_files: Vec<String>,
    /// Files in scope still unread when the audit gave up asking (first
    /// [`UNREAD_LISTED`] by name); a defects-only audit with any is partial.
    #[serde(default)]
    pub unread_files: Vec<String>,
    #[serde(default)]
    pub unread_count: usize,
    /// Files in scope, over the whole audit (for `read N/M`).
    #[serde(default)]
    pub scope_count: usize,
    /// The seed command's last run ([`crate::review_seed`]).
    #[serde(default)]
    pub verify: Option<VerifySeed>,
    /// True when the next audit or re-audit turn should run the seed
    /// command first; the turn loop clears it.
    #[serde(default)]
    pub pending_seed: bool,
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
            last_blocking: Vec::new(),
            format_retry_used: false,
            findings_to_fix: Vec::new(),
            changed_files: Vec::new(),
            all_changed_files: Vec::new(),
            uncited_changes: Vec::new(),
            audit_changed_files: Vec::new(),
            unverified_citations: Vec::new(),
            stop_reason: None,
            fix_turn_error: None,
            chunk: 0,
            chunk_verdict: None,
            continuations: 0,
            read_files: Vec::new(),
            unread_files: Vec::new(),
            unread_count: 0,
            scope_count: 0,
            verify: None,
            pending_seed: false,
        }
    }
}

/// What the frontend does after `Harness::review_next_step`.
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
#[derive(Clone, Debug, Default)]
pub struct ReviewTurnFacts {
    pub stop_reason: TurnStopReason,
    pub error: Option<String>,
    pub changed_files: Vec<String>,
    /// Parsed from the turn's assistant text, citations already checked.
    pub verdict: Option<ReviewVerdict>,
    pub unverified_citations: Vec<String>,
    /// Files an audit turn read ([`crate::review_reads`]).
    pub read_files: Vec<String>,
    /// Files under audit in this turn, when the scope is bounded (a
    /// defects-only audit); `None` leaves read coverage untracked.
    pub scope_files: Option<Vec<String>>,
}

impl ReviewDrive {
    pub fn start(inputs: ReviewInputs, audit_only: bool, max_passes: u32) -> Self {
        Self {
            phase: ReviewPhase::Audit,
            max_passes: max_passes.max(1),
            audit_only,
            inputs,
            pending_seed: true,
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
    /// · crates/hi-cli · continued 1/2 · …`.
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
                    chunk_shown = self.inputs.git.is_none();
                }
                if self.continuations > 0 {
                    parts.push(format!(
                        "continued {}/{MAX_CONTINUATIONS}",
                        self.continuations
                    ));
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
            ReviewPhase::Audit => Some(audit_prompt(
                &self.inputs,
                &[],
                self.chunk,
                self.max_passes,
                self.verify.as_ref(),
            )),
            ReviewPhase::Fix => Some(fix_prompt(
                &self.findings_to_fix,
                self.pass,
                self.max_passes,
            )),
            ReviewPhase::Reaudit => Some(self.reaudit_prompt()),
            _ => None,
        }
    }

    fn reaudit_prompt(&self) -> String {
        reaudit_prompt(
            &self.inputs,
            &self.coverage_items(),
            &self.changed_files,
            &self.uncited_changes,
            &self.findings_to_fix,
            self.pass,
            self.max_passes,
        )
    }

    /// `audit` / `re-audit 2`: what the next seed run precedes.
    pub fn seed_label(&self) -> String {
        match self.phase {
            ReviewPhase::Reaudit => format!("re-audit {}", self.pass),
            _ => "audit".to_string(),
        }
    }

    /// End the loop. An audit stopped between its turns (continuations,
    /// chunks) reports what its finished turns found.
    pub(crate) fn stop(&mut self, reason: impl Into<String>) -> ReviewStep {
        let reason = reason.into();
        if let Some(mut partial) = self.chunk_verdict.take() {
            partial.mark_feature_gaps(&self.unchecked_items());
            self.last_verdict = Some(partial);
        }
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
            push_unique(&mut self.audit_changed_files, &facts.changed_files);
            push_unique(&mut self.read_files, &facts.read_files);
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
                // re-audit decides; the blocking set and the pass cap bound
                // the loop.
                self.fix_turn_error = Some(error);
            }
            TurnStopReason::Completed => {}
        }
        match self.phase {
            ReviewPhase::Audit | ReviewPhase::Reaudit => self.after_audit(facts),
            ReviewPhase::Fix => {
                push_unique(&mut self.changed_files, &facts.changed_files);
                push_unique(&mut self.all_changed_files, &facts.changed_files);
                let uncited = uncited_changes(&facts.changed_files, &self.findings_to_fix);
                push_unique(&mut self.uncited_changes, &uncited);
                self.phase = ReviewPhase::Reaudit;
                self.pending_seed = true;
                ReviewStep::RunPrompt(self.reaudit_prompt())
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
        if self.phase == ReviewPhase::Audit {
            // Every turn of one audit adds to its unverified citations.
            push_unique(&mut self.unverified_citations, &facts.unverified_citations);
        } else {
            self.unverified_citations = facts.unverified_citations.clone();
        }
        let running = match self.chunk_verdict.take() {
            Some(mut running) => {
                running.merge(verdict);
                running
            }
            None => verdict,
        };
        if self.phase == ReviewPhase::Audit
            && let Some(prompt) = self.continue_audit(facts, &running)
        {
            self.chunk_verdict = Some(running);
            return ReviewStep::RunPrompt(prompt);
        }
        let mut verdict = running;
        verdict.mark_feature_gaps(&self.unchecked_items());
        self.seed_finding(&mut verdict);
        let blocking = verdict.blocking_findings();
        let was_reaudit = self.phase == ReviewPhase::Reaudit;
        self.last_verdict = Some(verdict);
        self.fix_turn_error = None;
        if blocking.is_empty() || self.audit_only {
            self.phase = ReviewPhase::Done;
            self.paused = false;
            return ReviewStep::Done(self.done_summary());
        }
        if was_reaudit && same_finding_set(&self.last_blocking, &blocking) {
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
        self.last_blocking = blocking.clone();
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

    /// In the audit phase, after a turn's verdict was merged into
    /// `running`: the prompt for another turn of the same audit, or `None`
    /// when the audit is complete. A defects-only turn that left files in
    /// scope unread gets a continuation (bounded per chunk; what is left
    /// after that makes the audit partial); then the next chunk, if any.
    fn continue_audit(
        &mut self,
        facts: &ReviewTurnFacts,
        running: &ReviewVerdict,
    ) -> Option<String> {
        if let Some(scope) = &facts.scope_files {
            self.scope_count = self.scope_count_with(scope);
            let unread = unread_files(scope, &self.read_files);
            if !unread.is_empty() {
                // A spec audit reads what grounds its items: its unread
                // files are reported, not re-asked.
                if self.inputs.defects_only() && self.continuations < MAX_CONTINUATIONS {
                    self.continuations += 1;
                    return Some(continuation_prompt(
                        &self.inputs,
                        self.chunk,
                        &unread,
                        &running.findings,
                        self.continuations,
                        MAX_CONTINUATIONS,
                        self.verify.as_ref(),
                    ));
                }
                self.unread_count += unread.len();
                for file in unread {
                    if self.unread_files.len() < UNREAD_LISTED {
                        self.unread_files.push(file);
                    }
                }
            }
        }
        self.continuations = 0;
        if self.inputs.chunks.is_empty() {
            return None;
        }
        self.chunk += 1;
        (self.chunk < self.inputs.chunks.len()).then(|| {
            audit_prompt(
                &self.inputs,
                &[],
                self.chunk,
                self.max_passes,
                self.verify.as_ref(),
            )
        })
    }

    /// Scope files seen so far plus this chunk's: chunks are disjoint, so a
    /// chunked audit sums them; a continuation of the same chunk does not.
    fn scope_count_with(&self, scope: &[String]) -> usize {
        if self.continuations > 0 {
            self.scope_count
        } else {
            self.scope_count + scope.len()
        }
    }

    /// The seed command failed and the model reported no P0/P1: the
    /// failure is the defect, unless plan items are missing (then the
    /// failing tests are likely theirs, and a fix pass must not build them)
    /// or its output names no workspace file (a missing toolchain, no
    /// network, a read-only sandbox: nothing a fix pass can reach).
    fn seed_finding(&self, verdict: &mut ReviewVerdict) {
        let Some(seed) = &self.verify else {
            return;
        };
        if seed.passed || seed.location.is_none() || verdict.has_blocking() {
            return;
        }
        let items_missing = verdict
            .coverage
            .iter()
            .any(|row| row.state != crate::review::CoverageState::Implemented);
        if items_missing {
            return;
        }
        verdict.findings.insert(0, seed.synthetic_finding());
    }
}

fn push_unique(into: &mut Vec<String>, items: &[String]) {
    for item in items {
        if !into.contains(item) {
            into.push(item.clone());
        }
    }
}

#[cfg(test)]
#[path = "review_drive_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "review_drive_scope_tests.rs"]
mod scope_tests;

#[cfg(test)]
#[path = "review_drive_reads_tests.rs"]
mod reads_tests;
