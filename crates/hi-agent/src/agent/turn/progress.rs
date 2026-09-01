//! Per-turn progress classification and no-progress stall tracking.

use std::collections::BTreeMap;

use hi_ai::Content;

use crate::ProgressEvent;
use crate::heuristics::{looks_like_unfinished_step, parse_text_tool_calls};
use crate::steering::{
    EvidenceTracker, ImplementationTracker, bash_no_progress_signature, classify_bash_command,
    evidence_kind_for_tool, implementation_tool_call_validates,
    implementation_tool_result_landed_mutation, implementation_tool_result_landed_substantive_edit,
    inspection_signature,
};

pub(super) const PROGRESS_EVENT_LIMIT: usize = 20;
pub(super) const NO_PROGRESS_FINAL_ANSWER_NUDGE_THRESHOLD: u32 = 2;
pub(super) const NO_PROGRESS_FINAL_ANSWER_NUDGE: &str = "You have not made new progress after repeated tool-use nudges. Stop using tools now and give the best final answer from the evidence already in the conversation. If the task cannot be completed from that evidence, say exactly what is missing.";
/// Sent when a turn reaches its configured step cap: one final tool-free round
/// so the model reports where it left the work instead of the turn dying
/// mid-flight with no answer. The uniform default cap or a deliberate override
/// (`--max-steps`, `/config steps <n>`, or an internal subagent budget) can
/// trigger it.
pub(super) const STEP_LIMIT_WRAP_UP_NUDGE: &str = "You have reached this turn's step limit. Stop using tools now. In a short final answer, report what you completed, what remains unfinished, and the exact state you are leaving the work in (files changed, checks not yet run). Do not claim the task is complete unless it actually is; the user can raise or remove the limit with /config steps.";
pub(super) const TOOL_LIMIT_WRAP_UP_NUDGE: &str = "You have reached this turn's tool-call limit. Stop using tools now. In a short final answer, report what you completed, what remains unfinished, and the exact state you are leaving the work in (files changed, checks not yet run). Do not claim the task is complete unless it actually is; the user can raise the limit with --max-tool-calls.";
/// Progress reason shared between the waiting-round recovery (Steer) and the
/// final-answer acceptance paths: it marks the turn as blocked only on live
/// background work, so a status answer is a valid terminal outcome.
pub(super) const AWAITING_BACKGROUND_REASON: &str = "background process is still running";
/// Consecutive waiting rounds (only polls/status probes of a still-running
/// background process) tolerated before the turn is steered to end with a
/// status report. Three rounds ≈ one launch check plus two follow-ups — enough
/// to catch a fast finish without funding an open-ended babysitting loop.
pub(super) const WAITING_ROUND_BUDGET: u32 = 3;
pub(super) const REPEATED_VALIDATION_DIAGNOSIS_NUDGE: &str = "The same deterministic validation failure survived another edit-and-test cycle. Stop applying variants of the previous patch. Re-read the failing code and trace the relevant state transition from the assertion backward; if an independent explore tool is available, use it for one focused root-cause diagnosis before editing again. Then make one bounded fix and rerun the narrowest failing validation.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProgressKind {
    Meaningful,
    Weak,
    None,
}

impl ProgressKind {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Meaningful => "meaningful",
            Self::Weak => "weak",
            Self::None => "none",
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ToolProgressLabel {
    pub(super) kind: ProgressKind,
    pub(super) reason: String,
    pub(super) signature: Option<String>,
    validation_scope: Option<String>,
    validation_failure: Option<String>,
}

impl ToolProgressLabel {
    pub(super) fn new(
        kind: ProgressKind,
        reason: impl Into<String>,
        signature: Option<String>,
    ) -> Self {
        Self {
            kind,
            reason: reason.into(),
            signature,
            validation_scope: None,
            validation_failure: None,
        }
    }

    fn validation(
        kind: ProgressKind,
        reason: impl Into<String>,
        scope: String,
        failure: Option<String>,
    ) -> Self {
        Self {
            kind,
            reason: reason.into(),
            signature: failure.clone(),
            validation_scope: Some(scope),
            validation_failure: failure,
        }
    }
}

#[derive(Clone, Debug)]
struct ValidationFailureProgress {
    signature: String,
    repeats: u32,
    diagnosis_nudged: bool,
    mutation_epoch: u32,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ProgressTracker {
    pub(super) no_progress_streak: u32,
    pub(super) no_progress_nudges: u32,
    pub(super) forced_final_answer_attempts: u32,
    pub(super) last_progress_reason: String,
    pub(super) last_stall_reason: String,
    /// Consecutive tool rounds that only watched still-running background
    /// work (see [`WAITING_ROUND_BUDGET`]). Reset by any non-waiting round.
    pub(super) waiting_rounds: u32,
    /// Sticky once the waiting budget is spent: the turn is blocked on live
    /// background work, so plan-continue nudges are suppressed and a status
    /// answer ends the turn. Cleared by a round that does real work.
    pub(super) awaiting_background: bool,
    /// Extra recoveries after a stall budget was spent (`max_keep_working`).
    pub(super) keep_working_rounds: u32,
    /// Signature of the stall that last consumed keep-working. A following
    /// round with the same signature is not another recovery.
    pub(super) keep_working_blocked_signature: Option<String>,
    /// Whether any tool ran since the last keep-working recovery. The
    /// blocked-signature guard only applies when this is true: a tool that
    /// re-issues the stalled action is a real repeat, but two consecutive
    /// text-only recap stalls share a stale tool signature without the model
    /// re-issuing anything.
    pub(super) saw_tool_since_keep_working: bool,
    /// Failed model-authored validation commands keyed by validator family.
    /// Entries survive unrelated green validators so `cargo check` cannot
    /// erase a still-failing `cargo test` repair trajectory.
    failed_validations: BTreeMap<String, ValidationFailureProgress>,
    pub(super) mutation_epoch: u32,
    pub(super) events: Vec<ProgressEvent>,
}

impl ProgressTracker {
    pub(super) fn push_event(
        &mut self,
        kind: ProgressKind,
        reason: impl Into<String>,
        signature: Option<String>,
    ) {
        self.events.push(ProgressEvent {
            kind: kind.as_str().to_string(),
            reason: reason.into(),
            signature,
        });
        if self.events.len() > PROGRESS_EVENT_LIMIT {
            let excess = self.events.len() - PROGRESS_EVENT_LIMIT;
            self.events.drain(0..excess);
        }
    }

    pub(super) fn record(
        &mut self,
        kind: ProgressKind,
        reason: impl Into<String>,
        signature: Option<String>,
    ) {
        let reason = reason.into();
        match kind {
            ProgressKind::Meaningful | ProgressKind::Weak => {
                self.no_progress_streak = 0;
                self.last_progress_reason = reason.clone();
            }
            ProgressKind::None => {
                self.no_progress_streak = self.no_progress_streak.saturating_add(1);
                self.last_stall_reason = reason.clone();
            }
        }
        self.push_event(kind, reason, signature);
    }

    /// Spend one keep-working recovery. Returns false when the budget is
    /// exhausted or disabled (`max == 0`).
    pub(super) fn try_keep_working(&mut self, max: u32) -> bool {
        if max == 0 || self.keep_working_rounds >= max {
            return false;
        }
        self.keep_working_rounds = self.keep_working_rounds.saturating_add(1);
        true
    }

    pub(super) fn last_event_signature(&self) -> Option<String> {
        self.events
            .iter()
            .rev()
            .find_map(|event| event.signature.clone())
    }

    pub(super) fn record_no_progress_nudge(
        &mut self,
        reason: impl Into<String>,
        signature: Option<String>,
    ) -> bool {
        // A signature here means the model issued a tool call this round (e.g.
        // a repeat caught by the repeat guard before execution). The
        // keep-working blocked-signature guard must treat that as a tool ran,
        // so a re-issued stalled action is still blocked even though the
        // repeat guard skipped `record_tool`.
        if signature.is_some() {
            self.saw_tool_since_keep_working = true;
        }
        self.no_progress_nudges = self.no_progress_nudges.saturating_add(1);
        self.record(ProgressKind::None, reason, signature);
        self.no_progress_nudges >= NO_PROGRESS_FINAL_ANSWER_NUDGE_THRESHOLD
            && self.forced_final_answer_attempts == 0
    }

    pub(super) fn record_tool(&mut self, label: &ToolProgressLabel) {
        // A tool ran this round, so the next keep-working guard can compare
        // against this round's signature (the model may be re-issuing the
        // stalled action). Without this flag, two consecutive text-only recap
        // stalls would both compare against the last *tool* signature (which
        // never changed between them) and the second recovery would be
        // wrongly blocked.
        self.saw_tool_since_keep_working = true;
        self.push_event(label.kind, label.reason.clone(), label.signature.clone());
    }

    pub(super) fn record_round_from_tools(&mut self, labels: &[ToolProgressLabel]) {
        self.observe_validation_round(labels);
        if let Some(label) = labels
            .iter()
            .find(|label| label.kind == ProgressKind::Meaningful)
        {
            self.record(
                ProgressKind::Meaningful,
                label.reason.clone(),
                label.signature.clone(),
            );
        } else if labels.iter().all(|label| label.kind == ProgressKind::None) {
            self.record(ProgressKind::None, "tool round made no progress", None);
        } else if let Some(label) = labels.first() {
            self.record(
                ProgressKind::Weak,
                label.reason.clone(),
                label.signature.clone(),
            );
        }
    }

    fn observe_validation_round(&mut self, labels: &[ToolProgressLabel]) {
        if labels.iter().any(|label| {
            matches!(
                label.reason.as_str(),
                "substantive edit" | "successful mutation" | "successful delegated mutation"
            )
        }) {
            self.mutation_epoch = self.mutation_epoch.saturating_add(1);
        }
        for scope in labels.iter().filter_map(|label| {
            (label.reason == "successful validation after mutation")
                .then(|| label.validation_scope.clone())
                .flatten()
        }) {
            self.failed_validations.remove(&scope);
        }
        for label in labels
            .iter()
            .filter(|label| label.reason == "validation command failed")
        {
            let (Some(scope), Some(signature)) = (
                label.validation_scope.as_ref(),
                label.validation_failure.as_ref(),
            ) else {
                continue;
            };
            match self.failed_validations.get_mut(scope) {
                Some(progress) if progress.signature == *signature => {
                    if progress.mutation_epoch != self.mutation_epoch {
                        progress.repeats = progress.repeats.saturating_add(1);
                        progress.mutation_epoch = self.mutation_epoch;
                    }
                }
                Some(progress) => {
                    *progress = ValidationFailureProgress {
                        signature: signature.clone(),
                        repeats: 1,
                        diagnosis_nudged: false,
                        mutation_epoch: self.mutation_epoch,
                    };
                }
                None => {
                    self.failed_validations.insert(
                        scope.clone(),
                        ValidationFailureProgress {
                            signature: signature.clone(),
                            repeats: 1,
                            diagnosis_nudged: false,
                            mutation_epoch: self.mutation_epoch,
                        },
                    );
                }
            }
        }
    }

    pub(super) fn take_repeated_validation_diagnosis(&mut self) -> bool {
        let Some(progress) = self
            .failed_validations
            .values_mut()
            .find(|progress| progress.repeats >= 2 && !progress.diagnosis_nudged)
        else {
            return false;
        };
        progress.diagnosis_nudged = true;
        true
    }

    pub(super) fn repeated_validation_repair_exhausted(&self) -> Option<String> {
        self.failed_validations
            .iter()
            .find(|(_, progress)| progress.diagnosis_nudged && progress.repeats >= 3)
            .map(|(scope, progress)| format!("{scope}\u{1f}{}", progress.signature))
    }

    pub(super) fn record_final_answer(&mut self) {
        self.record(ProgressKind::Meaningful, "accepted final answer", None);
    }

    pub(super) fn record_forced_final_answer_attempt(&mut self) {
        self.forced_final_answer_attempts = self.forced_final_answer_attempts.saturating_add(1);
    }
}

pub(super) fn no_progress_signature_for_calls(
    calls: &[(String, String, String)],
) -> Option<String> {
    calls.iter().find_map(|(_, name, args)| {
        inspection_signature(name, args)
            .or_else(|| bash_no_progress_signature(args).map(|sig| format!("bash:{sig}")))
    })
}

pub(super) fn forced_final_answer_is_unusable(text: &str, plan_incomplete: bool) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || plan_incomplete || looks_like_unfinished_step(trimmed) {
        return true;
    }
    parse_text_tool_calls(trimmed, 0)
        .iter()
        .any(|content| matches!(content, Content::ToolCall { .. }))
}

pub(super) fn signature_seen(evidence: &EvidenceTracker, signature: &Option<String>) -> bool {
    signature
        .as_ref()
        .is_some_and(|sig| evidence.seen_signatures.iter().any(|seen| seen == sig))
}

pub(super) fn background_handle_terminal(name: &str, output: &str) -> bool {
    match name {
        "bash_output" => output
            .lines()
            .next()
            .is_some_and(|status| status.contains(": exited") || status.contains(": killed")),
        "bash_kill" => {
            output.starts_with('[')
                && (output.contains("] killed")
                    || output.contains("] already exited")
                    || output.contains("] already killed"))
        }
        _ => false,
    }
}

/// Coarse validator identity used only for cross-edit convergence. Options and
/// test filters intentionally do not split a runner family, while combined
/// commands keep every family so an unrelated green command cannot clear it.
fn contains_command_phrase(command: &str, phrase: &str) -> bool {
    command.match_indices(phrase).any(|(start, matched)| {
        let before = command[..start].chars().next_back();
        let after = command[start + matched.len()..].chars().next();
        before.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
            && after.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
    })
}

fn validation_scope(arguments: &str) -> Option<String> {
    let command = crate::steering::bash_command(arguments)?;
    let compact = command
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let mut families = Vec::new();
    for (needle, family) in [
        ("cargo test", "cargo:test"),
        ("cargo check", "cargo:check"),
        ("cargo build", "cargo:build"),
        ("cargo clippy", "cargo:clippy"),
        ("npm run test", "npm:test"),
        ("npm test", "npm:test"),
        ("npm run build", "npm:build"),
        ("npm run check", "npm:check"),
        ("npm run lint", "npm:lint"),
        ("pnpm test", "pnpm:test"),
        ("pnpm build", "pnpm:build"),
        ("pnpm check", "pnpm:check"),
        ("pnpm lint", "pnpm:lint"),
        ("yarn test", "yarn:test"),
        ("yarn build", "yarn:build"),
        ("bun test", "bun:test"),
        ("bun run build", "bun:build"),
        ("pytest", "pytest"),
        ("go test", "go:test"),
        ("make test", "make:test"),
        ("make check", "make:check"),
        ("make build", "make:build"),
        ("just test", "just:test"),
        ("just check", "just:check"),
        ("just build", "just:build"),
        ("cargo run", "cargo:run"),
        ("true # validate", "fixture:validate"),
    ] {
        if contains_command_phrase(&compact, needle) && !families.contains(&family) {
            families.push(family);
        }
    }
    if families.is_empty() {
        Some(format!(
            "command:{}",
            hi_policy::normalize_command(&command)
        ))
    } else {
        Some(families.join("+"))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn classify_tool_progress(
    name: &str,
    arguments: &str,
    output: &str,
    error: bool,
    validation_succeeded: bool,
    signature: Option<String>,
    signature_was_seen: bool,
    repeated_idempotent_result: bool,
    tracker_before: &ImplementationTracker,
    plan_changed: bool,
    workspace_root: &std::path::Path,
) -> ToolProgressLabel {
    if plan_changed {
        return ToolProgressLabel::new(ProgressKind::Meaningful, "changed plan state", signature);
    }
    if repeated_idempotent_result {
        return ToolProgressLabel::new(
            ProgressKind::None,
            "repeated idempotent tool output",
            signature,
        );
    }
    if name == "bash" && bash_no_progress_signature(arguments).is_some() {
        return ToolProgressLabel::new(
            ProgressKind::None,
            "semantic no-op bash command",
            signature,
        );
    }
    if signature_was_seen {
        let reason = if matches!(name, "bash_output" | "bash_kill")
            && background_handle_terminal(name, output)
        {
            "stale background handle"
        } else {
            "repeated inspection signature"
        };
        return ToolProgressLabel::new(ProgressKind::None, reason, signature);
    }
    if error {
        if implementation_tool_call_validates(name, arguments)
            && let Some(scope) = validation_scope(arguments)
            && let Some(digest) = crate::verify_digest::digest_failure(workspace_root, output)
        {
            let signature = digest
                .signature
                .into_iter()
                .collect::<Vec<_>>()
                .join("\u{1f}");
            return ToolProgressLabel::validation(
                ProgressKind::Weak,
                "validation command failed",
                scope,
                Some(signature),
            );
        }
        return ToolProgressLabel::new(ProgressKind::Weak, "tool returned an error", signature);
    }
    if implementation_tool_result_landed_substantive_edit(name, arguments, output) {
        return ToolProgressLabel::new(ProgressKind::Meaningful, "substantive edit", signature);
    }
    if implementation_tool_result_landed_mutation(name, arguments, output) {
        return ToolProgressLabel::new(ProgressKind::Meaningful, "successful mutation", signature);
    }
    if tracker_before.mutation_seen
        && validation_succeeded
        && implementation_tool_call_validates(name, arguments)
        && let Some(scope) = validation_scope(arguments)
    {
        return ToolProgressLabel::validation(
            ProgressKind::Meaningful,
            "successful validation after mutation",
            scope,
            None,
        );
    }
    if let Some(kind) = evidence_kind_for_tool(name, arguments) {
        let (progress_kind, reason) = match kind {
            crate::steering::EvidenceKind::FileRead => {
                (ProgressKind::Meaningful, "new file evidence")
            }
            crate::steering::EvidenceKind::TargetedSearch => {
                (ProgressKind::Meaningful, "new targeted search evidence")
            }
            crate::steering::EvidenceKind::Listing => (ProgressKind::Weak, "new listing evidence"),
        };
        return ToolProgressLabel::new(progress_kind, reason, signature);
    }
    if name == "bash" {
        let Some(command) = crate::steering::bash_command(arguments) else {
            return ToolProgressLabel::new(ProgressKind::Weak, "bash command completed", signature);
        };
        let kind = classify_bash_command(&command);
        let reason = format!("bash {} command completed", kind.as_str());
        return ToolProgressLabel::new(ProgressKind::Weak, reason, signature);
    }
    ToolProgressLabel::new(ProgressKind::Weak, "tool completed", signature)
}

#[cfg(test)]
mod validation_progress_tests {
    use super::*;

    fn label(kind: ProgressKind, reason: &str, signature: Option<&str>) -> ToolProgressLabel {
        ToolProgressLabel::new(kind, reason, signature.map(str::to_string))
    }

    fn failed_validation(scope: &str, signature: &str) -> ToolProgressLabel {
        ToolProgressLabel::validation(
            ProgressKind::Weak,
            "validation command failed",
            scope.to_string(),
            Some(signature.to_string()),
        )
    }

    fn passed_validation(scope: &str) -> ToolProgressLabel {
        ToolProgressLabel::validation(
            ProgressKind::Meaningful,
            "successful validation after mutation",
            scope.to_string(),
            None,
        )
    }

    #[test]
    fn validation_scope_separates_check_from_test_and_ignores_test_filters() {
        assert_eq!(
            validation_scope(r#"{"command":"cargo test moves::checkmate_detected"}"#).as_deref(),
            Some("cargo:test")
        );
        assert_eq!(
            validation_scope(r#"{"command":"cargo check --workspace"}"#).as_deref(),
            Some("cargo:check")
        );
    }

    #[test]
    fn repeated_validation_failure_survives_intervening_edit_and_nudges_once() {
        let mut tracker = ProgressTracker::default();
        tracker.record_round_from_tools(&[failed_validation(
            "cargo:test",
            "test:moves::checkmate_detected:state-a",
        )]);
        assert!(!tracker.take_repeated_validation_diagnosis());
        tracker.record_round_from_tools(&[passed_validation("cargo:check")]);
        assert!(tracker.failed_validations.contains_key("cargo:test"));
        tracker.record_round_from_tools(&[failed_validation(
            "cargo:test",
            "test:moves::checkmate_detected:state-a",
        )]);
        assert!(
            !tracker.take_repeated_validation_diagnosis(),
            "rerunning the same failure without an intervening edit is not a second repair cycle"
        );

        tracker.record_round_from_tools(&[label(
            ProgressKind::Meaningful,
            "successful mutation",
            None,
        )]);
        tracker.record_round_from_tools(&[failed_validation(
            "cargo:test",
            "test:moves::checkmate_detected:state-a",
        )]);
        assert!(tracker.take_repeated_validation_diagnosis());
        assert!(!tracker.take_repeated_validation_diagnosis());

        tracker.record_round_from_tools(&[label(
            ProgressKind::Meaningful,
            "successful mutation",
            None,
        )]);
        tracker.record_round_from_tools(&[failed_validation(
            "cargo:test",
            "test:moves::checkmate_detected:state-a",
        )]);
        assert!(tracker.repeated_validation_repair_exhausted().is_some());

        tracker.record_round_from_tools(&[label(
            ProgressKind::Meaningful,
            "successful mutation",
            None,
        )]);
        tracker.record_round_from_tools(&[failed_validation(
            "cargo:test",
            "test:moves::checkmate_detected:state-b",
        )]);
        assert_eq!(tracker.failed_validations["cargo:test"].repeats, 1);
        assert!(!tracker.take_repeated_validation_diagnosis());
        assert!(tracker.repeated_validation_repair_exhausted().is_none());

        tracker.record_round_from_tools(&[passed_validation("cargo:test")]);
        assert!(!tracker.failed_validations.contains_key("cargo:test"));
        assert!(tracker.repeated_validation_repair_exhausted().is_none());
    }
}
