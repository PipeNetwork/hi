//! Post-turn finalize recap, usage/steer formatting, and text-tool cleanup.

use std::sync::Arc;

use hi_ai::{ChatRequest, Content, Message, RequestProfile, StreamEvent, ToolMode, ToolSpec};

use crate::heuristics::{
    humanize_count, looks_mutating, parse_text_tool_calls, textcall_id_offset,
};
use crate::transcript::{NudgeKind, repair_invalid_tool_call_arguments_in_messages};
use crate::{FINALIZE_PROMPT, Ui, partial_text_tool_call_start};

use super::helpers::rate_limit_summary;

impl crate::Agent {
    pub(super) async fn finalize_turn(&mut self, turn_start: usize, ui: &mut dyn Ui) {
        // Only send the current turn's messages (plus the system prompt for
        // context), not the entire session history. The recap only needs to
        // know what happened *this turn* — sending 40K tokens of old context
        // to produce a 200-token summary is pure waste.
        let turn = &self.messages.as_slice()[turn_start..];
        let mut messages = Vec::with_capacity(turn.len() + 2);
        messages.push(self.minimal_system_message());
        messages.extend_from_slice(turn);
        messages.push(Message::user(FINALIZE_PROMPT));
        repair_invalid_tool_call_arguments_in_messages(&mut messages);

        let request = ChatRequest {
            model: self.config.routing.model.clone(),
            user_turn: false,
            canonical_objective: None,
            messages: Arc::from(messages),
            tools: Arc::new([]), // recap only — no tool use
            max_tokens: 2048,    // throwaway call — recaps can be detailed
            temperature: self.config.routing.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                compat: self.config.routing.compat,
                tool_mode: ToolMode::ChatOnly,
                stream_usage: None,
            },
        };

        let mut recap = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => {
                recap.push_str(&text);
                ui.assistant_text(&text);
            }
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Reasoning(_) => {}
        };
        let completion = match self.provider.stream(request, &mut sink).await {
            Ok(completion) => completion,
            Err(err) => {
                // Finalize is a side call — book its error usage without resetting
                // the main conversation's `context_used` gauge.
                self.add_side_error_usage(&err);
                self.emit_usage(ui);
                // Flush any partially-streamed recap text before the status
                // line, so it isn't left dangling in the UI's pending buffer.
                ui.assistant_end();
                ui.status(&format!("(couldn't generate the final summary: {err})"));
                return;
            }
        };

        // Side call: spend counts, but its small request must not clobber the
        // main conversation's context gauge (see add_side_usage).
        self.add_side_usage(completion.usage);
        self.emit_usage(ui);

        // Fall back to the final content if the provider didn't stream text.
        // Emit it through the UI before assistant_end so the user actually sees
        // the recap — without this, a provider that returns text only in the
        // completion object (not via stream deltas) would have its summary
        // recorded in history but never displayed, so the turn appears to end
        // without its closing message.
        if recap.trim().is_empty() {
            for c in &completion.content {
                if let Content::Text(t) = c {
                    recap.push_str(t);
                    ui.assistant_text(t);
                }
            }
        }
        ui.assistant_end();

        if recap.trim().is_empty() {
            return; // nothing to record
        }
        // Record both the synthetic request and the recap so roles alternate.
        // The recap is a text-only assistant message (no tool calls).
        self.messages
            .push_nudge(NudgeKind::Finalize, FINALIZE_PROMPT);
        self.messages.push_assistant(vec![Content::Text(recap)]);
    }

    /// Format the completed-turn usage marker with explicitly scoped metrics.
    pub(crate) fn usage_summary(&self, usage: &hi_ai::Usage) -> String {
        // User-facing prompt size first. The full request can include system,
        // tool, and history context, so putting it first made a short question
        // like "what's your name?" appear to be a 1.5k-token user prompt.
        let mut summary = format!(
            "[user prompt estimate {} · output across all model calls {}{}",
            humanize_count(self.report.last_user_prompt_tokens),
            if self.report.last_turn_usage.estimated {
                "~"
            } else {
                ""
            },
            humanize_count(self.report.last_turn_usage.output_tokens),
        );
        if self.report.last_turn_usage.cache_read_tokens > 0 {
            summary.push_str(&format!(
                " ⟲{}",
                humanize_count(self.report.last_turn_usage.cache_read_tokens)
            ));
        }
        // The context gauge is the point-in-time full request size, which is
        // the number providers generally bill as input and the number that
        // drives context-window pressure.
        if let Some(window) = self.config.routing.context_window
            && window > 0
        {
            let pct = (self.report.context_used * 100 / u64::from(window)).min(100);
            summary.push_str(&format!(
                " · ctx {}{pct}% ({}/{})",
                if self.report.last_turn_usage.estimated {
                    "~"
                } else {
                    ""
                },
                humanize_count(self.report.context_used),
                humanize_count(u64::from(window)),
            ));
        } else if self.report.context_used > 0 {
            summary.push_str(&format!(
                " · ctx {}{}",
                if self.report.last_turn_usage.estimated {
                    "~"
                } else {
                    ""
                },
                humanize_count(self.report.context_used)
            ));
        }
        if let Some(limits) = usage.rate_limits.and_then(rate_limit_summary) {
            summary.push_str(&format!(" · {limits}"));
        }
        // Per-turn trajectory: a terse "steer" suffix when the turn needed
        // more than one shot, so a noisy success reads differently from a clean
        // one. Clean turns (no verify rounds, no recovery retries, no nudges,
        // no stalls) add nothing. See `TurnTelemetry`.
        if let Some(steer) = self.turn_steer() {
            summary.push_str(&format!(" · {steer}"));
        }
        summary.push(']');
        summary
    }

    /// A terse per-turn steering summary for the usage line, or `None` when the
    /// turn was clean (no extra rounds of any kind, no stall). Format:
    /// `steer: 2 verify · 1 retry · stalled` — components omitted when zero.
    pub(crate) fn turn_steer(&self) -> Option<String> {
        let t = &self.report.last_turn_telemetry;
        let mut parts: Vec<String> = Vec::new();
        if t.verify_rounds > 0 {
            parts.push(format!("{} verify", t.verify_rounds));
        }
        if t.recovery_retries > 0 {
            parts.push(format!("{} retry", t.recovery_retries));
        }
        if t.repeat_nudges > 0 {
            parts.push(format!("{} repeat", t.repeat_nudges));
        }
        if t.continue_nudges > 0 {
            parts.push(format!("{} continue", t.continue_nudges));
        }
        if t.quality_repair_nudges > 0 {
            parts.push(format!("{} review-repair", t.quality_repair_nudges));
        }
        if t.truncation_retries > 0 {
            parts.push(format!("{} trunc", t.truncation_retries));
        }
        if t.stalled_unfinished || t.stalled_repeating {
            parts.push("stalled".to_string());
        }
        if parts.is_empty() {
            None
        } else {
            Some(format!("steer: {}", parts.join(" · ")))
        }
    }

    pub(crate) fn request_tools_for(&self, mode: ToolMode) -> Arc<[ToolSpec]> {
        match mode {
            ToolMode::ChatOnly => Arc::new([]),
            // `explore` isn't classified read-only (that keeps a read-only *child*
            // from ever seeing it), but delegating a read-only investigation is
            // itself read-only — so a top-level agent keeps `explore` in a
            // read-only/review turn. A subagent never has it in `self.tools`.
            ToolMode::ReadOnly => self
                .tools
                .iter()
                .filter(|tool| {
                    hi_tools::is_read_only(&tool.name)
                        || (tool.name == "explore" && !self.config.subagents.is_subagent)
                })
                .cloned()
                .collect::<Vec<_>>()
                .into(),
            ToolMode::Auto | ToolMode::Required => self.tools.clone(),
        }
    }

    pub(super) fn tools_unavailable_for(&self, input: &str) -> bool {
        matches!(
            self.config.routing.tool_mode,
            ToolMode::ChatOnly | ToolMode::ReadOnly
        ) && looks_mutating(input)
    }

    /// Clean text-embedded tool-call JSON from `Content::Text` blocks in
    /// `content`. Used on the truncation path (before `parse_text_tool_calls`
    /// would normally run) so raw tool-call JSON doesn't leak into recorded
    /// history. Complete tool calls are extracted and stripped; partial JSON
    /// stays as text. `ToolCall` blocks are left in place — the caller
    /// (`push_assistant_text_only`) strips them.
    pub(super) fn clean_text_tool_calls_from_content(&self, content: &mut Vec<Content>) -> bool {
        let mut new_content = Vec::new();
        let mut saw_partial_tool_call = false;
        for c in content.drain(..) {
            match c {
                Content::Text(t) => {
                    let parsed = parse_text_tool_calls(&t, textcall_id_offset(&self.messages));
                    if parsed.iter().any(|p| matches!(p, Content::ToolCall { .. })) {
                        // Tool calls found — keep only the Text blocks (drop
                        // the extracted ToolCalls; they're partial/truncated
                        // and have no matching results).
                        new_content.extend(
                            parsed.into_iter().filter(|p| {
                                matches!(p, Content::Text(_) | Content::Thinking { .. })
                            }),
                        );
                    } else if let Some(index) = partial_text_tool_call_start(&t) {
                        let prose = t[..index].trim_end();
                        if !prose.is_empty() {
                            new_content.push(Content::Text(prose.to_string()));
                        }
                        saw_partial_tool_call = true;
                    } else {
                        new_content.push(Content::Text(t));
                    }
                }
                Content::ToolCall { .. } => saw_partial_tool_call = true,
                other => new_content.push(other),
            }
        }
        *content = new_content;
        saw_partial_tool_call
    }
}

/// Pure classification of the public turn outcome from settled turn state.
/// Extracted from `run_turn` so status/stop-reason rules can be unit-tested
/// without driving the full loop.
pub(super) fn classify_turn_outcome(
    verification_infrastructure_error: bool,
    verification_unstable: bool,
    last_verify: Option<bool>,
    changed_files: &[String],
    turn_had_mutation: bool,
    no_check_executed: bool,
    independent_review_status: crate::ReviewStatus,
    skeptic_last_status: Option<crate::SkepticStatus>,
    ended_at_cap: bool,
    stalled_unfinished: bool,
    stalled_repeating: bool,
    expected_mutation: bool,
    allow_unverified: bool,
) -> (crate::TurnStatus, crate::VerificationStatus, crate::ReviewStatus, crate::TurnStopReason) {
    use crate::{ReviewStatus, TurnStatus, TurnStopReason, VerificationStatus};
    use super::helpers::combined_review_status;
    use crate::verify::is_prose_only_path;

    let verification = if verification_infrastructure_error {
        VerificationStatus::InfrastructureError
    } else if last_verify == Some(true) {
        VerificationStatus::Passed
    } else if last_verify == Some(false) {
        VerificationStatus::Failed
    } else if (changed_files.is_empty() && !turn_had_mutation)
        || no_check_executed
        || (!changed_files.is_empty() && changed_files.iter().all(|path| is_prose_only_path(path)))
    {
        VerificationStatus::NotApplicable
    } else {
        VerificationStatus::Unverified
    };
    let skeptic_review = match skeptic_last_status {
        Some(crate::SkepticStatus::Approved) => ReviewStatus::Passed,
        Some(crate::SkepticStatus::Objected | crate::SkepticStatus::Escalated) => {
            ReviewStatus::Objected
        }
        Some(crate::SkepticStatus::Unavailable) => ReviewStatus::Unavailable,
        None => ReviewStatus::NotRequired,
    };
    let review = combined_review_status(independent_review_status, skeptic_review);
    let status = if verification_infrastructure_error {
        TurnStatus::Failed
    } else if ended_at_cap
        || stalled_unfinished
        || stalled_repeating
        || (expected_mutation && changed_files.is_empty())
        || verification == VerificationStatus::Failed
        || review == ReviewStatus::Objected
        || (verification == VerificationStatus::Unverified && !allow_unverified)
    {
        TurnStatus::Incomplete
    } else {
        TurnStatus::Completed
    };
    let stop_reason = if verification_infrastructure_error {
        TurnStopReason::InfrastructureFailure
    } else if verification_unstable {
        TurnStopReason::VerificationUnstable
    } else if ended_at_cap {
        TurnStopReason::StepLimit
    } else if review == ReviewStatus::Objected {
        TurnStopReason::ReviewObjected
    } else if verification == VerificationStatus::Failed {
        TurnStopReason::VerificationFailed
    } else if stalled_unfinished
        || stalled_repeating
        || (expected_mutation && changed_files.is_empty())
    {
        TurnStopReason::Stalled
    } else if verification == VerificationStatus::Unverified {
        TurnStopReason::VerificationUnavailable
    } else if verification == VerificationStatus::NotApplicable {
        TurnStopReason::NoApplicableVerification
    } else {
        TurnStopReason::Completed
    };
    (status, verification, review, stop_reason)
}

#[cfg(test)]
mod classify_tests {
    use super::classify_turn_outcome;
    use crate::{ReviewStatus, TurnStatus, TurnStopReason, VerificationStatus};

    #[test]
    fn completed_when_verify_passed() {
        let (status, verification, review, stop) = classify_turn_outcome(
            false,
            false,
            Some(true),
            &["src/lib.rs".into()],
            true,
            false,
            ReviewStatus::NotRequired,
            None,
            false,
            false,
            false,
            true,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(verification, VerificationStatus::Passed);
        assert_eq!(review, ReviewStatus::NotRequired);
        assert_eq!(stop, TurnStopReason::Completed);
    }

    #[test]
    fn incomplete_when_unverified_and_not_allowed() {
        let (status, verification, _, stop) = classify_turn_outcome(
            false,
            false,
            None,
            &["src/lib.rs".into()],
            true,
            false,
            ReviewStatus::NotRequired,
            None,
            false,
            false,
            false,
            true,
            false,
        );
        assert_eq!(status, TurnStatus::Incomplete);
        assert_eq!(verification, VerificationStatus::Unverified);
        assert_eq!(stop, TurnStopReason::VerificationUnavailable);
    }

    #[test]
    fn prose_only_is_not_applicable() {
        let (status, verification, _, stop) = classify_turn_outcome(
            false,
            false,
            None,
            &["README.md".into()],
            true,
            true,
            ReviewStatus::NotRequired,
            None,
            false,
            false,
            false,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(verification, VerificationStatus::NotApplicable);
        assert_eq!(stop, TurnStopReason::NoApplicableVerification);
    }
}
