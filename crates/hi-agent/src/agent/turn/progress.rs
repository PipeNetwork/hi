//! Per-turn progress classification and no-progress stall tracking.

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
    pub(super) last_stall_reason: String,
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

    pub(super) fn record_no_progress_nudge(
        &mut self,
        reason: impl Into<String>,
        signature: Option<String>,
    ) -> bool {
        self.no_progress_nudges = self.no_progress_nudges.saturating_add(1);
        self.record(ProgressKind::None, reason, signature);
        self.no_progress_nudges >= NO_PROGRESS_FINAL_ANSWER_NUDGE_THRESHOLD
            && self.forced_final_answer_attempts == 0
    }

    pub(super) fn record_tool(&mut self, label: &ToolProgressLabel) {
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
    {
        return ToolProgressLabel::new(
            ProgressKind::Meaningful,
            "successful validation after mutation",
            signature,
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
