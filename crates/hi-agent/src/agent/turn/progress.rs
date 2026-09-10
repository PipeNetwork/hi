//! Per-turn progress classification and no-progress tracking.

use std::collections::HashSet;

use crate::ProgressEvent;
#[cfg(test)]
use crate::steering::GoalKind;
use crate::steering::{
    EvidenceTracker, ImplementationTracker, ToolLoopGuardrail, bash_no_progress_signature,
    classify_bash_command, evidence_kind_for_tool, implementation_tool_call_validates,
    inspection_signature,
};

use super::retention::ProgressEventLog;

pub(super) const NO_PROGRESS_FINAL_ANSWER_NUDGE_THRESHOLD: u32 = 2;
pub(super) const NO_PROGRESS_FINAL_ANSWER_NUDGE: &str = "You have not made new progress after repeated tool-use nudges. Stop using tools now and give the best final answer from the evidence already in the conversation. If the task cannot be completed from that evidence, say exactly what is missing.";
/// Sent when a turn reaches its configured step cap: one final tool-free round
/// so the model reports where it left the work instead of the turn dying
/// mid-flight with no answer. Only a deliberate override (`--max-steps`,
/// `/config steps <n>`, or an internal subagent budget) can trigger it.
pub(super) const STEP_LIMIT_WRAP_UP_NUDGE: &str = "You have reached this turn's step limit. Stop using tools now. In a short final answer, report what you completed, what remains unfinished, and the exact state you are leaving the work in (files changed, checks not yet run). Do not claim the task is complete unless it actually is; the user can raise or remove the limit with /config steps.";
pub(super) const TOOL_LIMIT_WRAP_UP_NUDGE: &str = "You have reached this turn's tool-call limit. Stop using tools now. In a short final answer, report what you completed, what remains unfinished, and the exact state you are leaving the work in (files changed, checks not yet run). Do not claim the task is complete unless it actually is; the user can raise the limit with --max-tool-calls.";
/// Progress reason shared between the waiting-round recovery (Steer) and the
/// final-answer acceptance paths: it marks the turn as blocked only on live
/// background work, so a status answer is a valid terminal outcome.
pub(super) const AWAITING_BACKGROUND_REASON: &str = "background process is still running";
/// Consecutive waiting rounds tolerated before the turn is steered to end with
/// a status report. This catches fast completions without allowing unbounded
/// model-driven polling.
pub(super) const WAITING_ROUND_BUDGET: u32 = 3;
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
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct ProgressTracker {
    pub(super) no_progress_streak: u32,
    pub(super) no_progress_nudges: u32,
    pub(super) forced_final_answer_attempts: u32,
    pub(super) last_progress_reason: String,
    pub(super) last_no_progress_reason: String,
    /// Consecutive tool rounds that only watched still-running background work.
    pub(super) waiting_rounds: u32,
    /// Sticky once the waiting budget is spent, until a round does real work.
    pub(super) awaiting_background: bool,
    /// Extra recoveries after a no-progress budget was spent (`max_keep_working`).
    pub(super) keep_working_rounds: u32,
    /// Signature of the no-progress event that last consumed keep-working.
    pub(super) keep_working_blocked_signature: Option<String>,
    /// Whether a tool ran since the last keep-working recovery.
    pub(super) saw_tool_since_keep_working: bool,
    /// `update_plan` (or equivalent) changed checklist state this turn.
    /// Grok-build's TodoGate: a live in-turn checklist continues even on
    /// analysis. An inherited stale plan does not.
    pub(super) plan_updated: bool,
    /// Cross-round repeat-loop state lives here so the newer owned turn bag can
    /// retain its construction shape while preserving the established guards.
    pub(super) repeat_sampling_rounds: u32,
    pub(super) force_no_progress_final_answer_next: bool,
    /// The provider repeated a semantically empty completion claim through
    /// the bounded answer-repair budget while durable plan work remained.
    /// Settlement owns this as a typed no-progress outcome; it is not a
    /// provider transport or verification-infrastructure failure.
    pub(super) bounded_plan_answer_recovery_exhausted: bool,
    pub(super) stationarity: crate::steering::IdenticalToolCallRun,
    pub(super) stationarity_ended: bool,
    pub(super) laziness_nudges: u32,
    /// Revision shared by the pre-call signature and post-call result guards.
    inspection_revision: Option<u64>,
    pub(super) prev_added_no_evidence: bool,
    pub(super) prev_call_sig: Option<Vec<(String, String)>>,
    pub(super) tool_guardrail: ToolLoopGuardrail,
    /// Bounded diagnostic trail. Correctness-relevant plan-drive evidence is
    /// pinned separately so middle compaction cannot turn productive work into
    /// a false stall.
    pub(super) events: ProgressEventLog,
    plan_drive_progress_event: Option<ProgressEvent>,
    /// Complete hashed read/search identities for cross-turn drive correctness.
    /// Unlike the diagnostic event trail, this set is never head/tail compacted.
    drive_evidence_hashes: HashSet<String>,
}

impl ProgressTracker {
    pub(super) fn observe_workspace_revision(
        &mut self,
        evidence: &mut EvidenceTracker,
        revision: u64,
        mutation_applied: bool,
    ) -> bool {
        let changed = self
            .inspection_revision
            .replace(revision)
            .is_some_and(|previous| previous != revision)
            || mutation_applied;
        self.tool_guardrail.observe_workspace_revision(revision);
        if changed {
            evidence.invalidate_workspace_inspections();
            // The just-completed mutation remains eligible for exact-call
            // deduplication (e.g. writing identical bytes again). External
            // changes invalidate even the immediately preceding inspection.
            if !mutation_applied {
                self.prev_call_sig = None;
            }
            self.prev_added_no_evidence = false;
            self.no_progress_nudges = 0;
            self.no_progress_streak = 0;
            self.last_no_progress_reason.clear();
            self.force_no_progress_final_answer_next = false;
        }
        changed
    }

    pub(super) fn push_event(
        &mut self,
        kind: ProgressKind,
        reason: impl Into<String>,
        signature: Option<String>,
    ) {
        let event = ProgressEvent {
            kind: kind.as_str().to_string(),
            reason: reason.into(),
            signature,
        };
        if event.reason == "changed plan state" {
            self.plan_updated = true;
        }
        if self.plan_drive_progress_event.is_none()
            && crate::plan_drive::progress_event_counts_as_plan_drive(&event)
        {
            self.plan_drive_progress_event = Some(event.clone());
        }
        if crate::plan_drive::progress_event_is_drive_evidence(&event)
            && let Some(signature) = event.signature.as_deref()
        {
            self.drive_evidence_hashes
                .insert(crate::plan_drive::hash_drive_evidence_signature(signature));
        }
        self.events.push(event);
    }

    /// Materialize the bounded trail for reports/settlement. If the one
    /// correctness-relevant plan-drive event fell in the compacted middle,
    /// reinsert that exact event and evict one non-prefix diagnostic instead.
    pub(super) fn retained_events(&self) -> Vec<ProgressEvent> {
        let mut events = self.events.to_vec();
        let Some(pinned) = self.plan_drive_progress_event.as_ref() else {
            return events;
        };
        if events.iter().any(|event| event == pinned) {
            return events;
        }
        if events.len() >= super::retention::PROGRESS_EVENT_LIMIT {
            events.remove(super::retention::PROGRESS_EVENT_HEAD);
        }
        events.insert(super::retention::PROGRESS_EVENT_HEAD, pinned.clone());
        events
    }

    pub(super) fn retained_events_dropped(&self) -> u64 {
        let retained = self.retained_events().len() as u64;
        self.events.total().saturating_sub(retained)
    }

    pub(super) fn drive_evidence_hashes(&self) -> Vec<String> {
        let mut hashes = self
            .drive_evidence_hashes
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        hashes.sort_unstable();
        hashes
    }

    pub(super) fn record(
        &mut self,
        kind: ProgressKind,
        reason: impl Into<String>,
        signature: Option<String>,
    ) {
        let reason = reason.into();
        if kind == ProgressKind::Meaningful {
            // Warnings describe a consecutive stall, not lifetime debt. New
            // edits/evidence must not inherit a forced closeout from earlier work.
            self.no_progress_nudges = 0;
            self.force_no_progress_final_answer_next = false;
            self.repeat_sampling_rounds = 0;
            self.keep_working_rounds = 0;
            self.keep_working_blocked_signature = None;
            self.last_no_progress_reason.clear();
        }
        match kind {
            ProgressKind::Meaningful | ProgressKind::Weak => {
                self.no_progress_streak = 0;
                self.last_progress_reason = reason.clone();
            }
            ProgressKind::None => {
                self.no_progress_streak = self.no_progress_streak.saturating_add(1);
                self.last_no_progress_reason = reason.clone();
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
        if signature.is_some() {
            self.saw_tool_since_keep_working = true;
        }
        self.no_progress_nudges = self.no_progress_nudges.saturating_add(1);
        self.record(ProgressKind::None, reason, signature);
        self.no_progress_nudges >= NO_PROGRESS_FINAL_ANSWER_NUDGE_THRESHOLD
            && self.forced_final_answer_attempts == 0
    }

    pub(super) fn record_tool(&mut self, label: &ToolProgressLabel) {
        self.saw_tool_since_keep_working = true;
        self.push_event(label.kind, label.reason.clone(), label.signature.clone());
    }

    pub(super) fn record_round_from_tools(&mut self, labels: &[ToolProgressLabel]) {
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

#[cfg(test)]
pub(super) fn forced_final_answer_is_unusable(
    text: &str,
    plan_incomplete: bool,
    kind: GoalKind,
) -> bool {
    crate::steering::forced_final_answer_is_unusable(text, plan_incomplete, kind)
}

pub(super) fn signature_seen(evidence: &EvidenceTracker, signature: &Option<String>) -> bool {
    signature
        .as_ref()
        .is_some_and(|signature| evidence.has_seen_signature(signature))
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

#[allow(clippy::too_many_arguments)]
pub(super) fn classify_tool_progress(
    name: &str,
    arguments: &str,
    output: &str,
    error: bool,
    validation_succeeded: bool,
    mutation_applied: bool,
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
    if mutation_applied {
        return ToolProgressLabel::new(ProgressKind::Meaningful, "successful mutation", signature);
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
            && let Some(digest) = crate::verify_digest::digest_failure(workspace_root, output)
        {
            let signature = digest
                .signature
                .into_iter()
                .collect::<Vec<_>>()
                .join("\u{1f}");
            return ToolProgressLabel::new(
                ProgressKind::Weak,
                "validation command failed",
                Some(signature),
            );
        }
        return ToolProgressLabel::new(ProgressKind::Weak, "tool returned an error", signature);
    }
    if tracker_before.mutation_seen
        && validation_succeeded
        && implementation_tool_call_validates(name, arguments)
    {
        return ToolProgressLabel::new(
            ProgressKind::Meaningful,
            "successful validation after mutation",
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
mod progress_retention_tests {
    use super::*;

    #[test]
    fn workspace_revision_reopens_inspection_without_erasing_diagnostics() {
        let mut tracker = ProgressTracker::default();
        let mut evidence = EvidenceTracker::default();
        let args = r#"{"path":"src/context.rs"}"#;
        let calls = vec![("read".into(), "read".into(), args.into())];
        assert!(!tracker.observe_workspace_revision(&mut evidence, 7, false));
        evidence.record_success("read", args, "complete source");
        tracker.prev_added_no_evidence = true;
        tracker.no_progress_nudges = 2;
        tracker.force_no_progress_final_answer_next = true;
        assert!(!evidence.round_adds_evidence(&calls));
        assert!(evidence.rereads_only_completed_files(&calls));
        assert!(!tracker.observe_workspace_revision(&mut evidence, 7, false));
        assert!(!evidence.round_adds_evidence(&calls));
        assert!(tracker.force_no_progress_final_answer_next);

        assert!(tracker.observe_workspace_revision(&mut evidence, 8, false));
        assert!(evidence.round_adds_evidence(&calls));
        assert!(!evidence.rereads_only_completed_files(&calls));
        assert!(!tracker.force_no_progress_final_answer_next);
        assert!(!tracker.prev_added_no_evidence);
        assert_eq!(tracker.no_progress_nudges, 0);
        assert_eq!(evidence.file_reads, 1);
        assert_eq!(
            evidence.inspected_paths.front().map(String::as_str),
            Some("src/context.rs")
        );

        evidence.record_success("read", args, "complete source");
        assert!(!evidence.round_adds_evidence(&calls));
        assert!(tracker.observe_workspace_revision(&mut evidence, 8, true));
        assert!(evidence.round_adds_evidence(&calls));
    }

    #[test]
    fn analysis_review_is_usable_even_if_it_offers_to_implement() {
        let review = "I have all the source in context now. Here's my review.\n\n\
## Major issues found\n\n\
1. PRIVMSG leaks whether a username exists.\n\
2. NICK rename leaks rate-limiter entries.\n\
3. KICK broadcasts KICKED to every subscriber.\n\n\
Those are the highest-value fixes.\n\n\
Let me implement fixes for #1, #2, and #3.";
        assert!(
            !forced_final_answer_is_unusable(review, false, GoalKind::Analysis),
            "grok-build analysis: the written review is the deliverable"
        );
        assert!(!forced_final_answer_is_unusable(
            "Let me implement the parser.",
            false,
            GoalKind::Analysis
        ));
        assert!(forced_final_answer_is_unusable(
            "",
            false,
            GoalKind::Analysis
        ));
        assert!(forced_final_answer_is_unusable(
            "The parser is fixed.",
            true,
            GoalKind::CodeChange
        ));
        assert!(!forced_final_answer_is_unusable(
            "The parser is fixed.",
            false,
            GoalKind::CodeChange
        ));
        assert!(forced_final_answer_is_unusable(
            "I can't proceed.",
            false,
            GoalKind::CodeChange
        ));
    }

    #[test]
    fn meaningful_work_resets_stall_warnings_but_tool_errors_do_not() {
        let mut tracker = ProgressTracker::default();
        assert!(!tracker.record_no_progress_nudge("repeated read", None));
        tracker.record(ProgressKind::Meaningful, "substantive edit", None);
        assert!(!tracker.record_no_progress_nudge("repeated read", None));
        tracker.record(ProgressKind::Weak, "tool returned an error", None);
        assert!(tracker.record_no_progress_nudge("repeated read", None));
        tracker.record_forced_final_answer_attempt();
        tracker.record_final_answer();
        assert_eq!(tracker.forced_final_answer_attempts, 1);
    }

    #[test]
    fn plan_drive_progress_is_pinned_across_bounded_middle_compaction() {
        let mut tracker = ProgressTracker::default();
        for index in 0..400 {
            let (kind, reason) = if index == 100 {
                (ProgressKind::Meaningful, "substantive edit".to_string())
            } else {
                (ProgressKind::None, format!("no-progress event {index}"))
            };
            tracker.record(kind, reason, Some(format!("event-{index:03}")));
        }

        let retained = tracker.retained_events();
        assert_eq!(
            retained.len(),
            super::super::retention::PROGRESS_EVENT_LIMIT
        );
        assert_eq!(tracker.retained_events_dropped(), 144);
        assert_eq!(retained[31].signature.as_deref(), Some("event-031"));
        assert_eq!(retained[32].signature.as_deref(), Some("event-100"));
        assert_eq!(retained[33].signature.as_deref(), Some("event-177"));
        assert_eq!(
            retained.last().unwrap().signature.as_deref(),
            Some("event-399")
        );
        assert!(crate::plan_drive_made_progress(
            Some("same step"),
            Some("same step"),
            &retained,
            &[] as &[String],
        ));
    }

    #[test]
    fn drive_evidence_hashes_are_complete_when_diagnostics_compact() {
        let mut tracker = ProgressTracker::default();
        for index in 0..400 {
            tracker.record(
                ProgressKind::Meaningful,
                "new file evidence",
                Some(format!("read:file-{index}:1:default")),
            );
        }

        assert_eq!(
            tracker.retained_events().len(),
            super::super::retention::PROGRESS_EVENT_LIMIT
        );
        assert_eq!(tracker.drive_evidence_hashes().len(), 400);
    }
}
