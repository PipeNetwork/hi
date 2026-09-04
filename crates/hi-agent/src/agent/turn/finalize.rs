//! Post-turn finalize recap, usage/steer formatting, and text-tool cleanup.

use std::sync::Arc;

use hi_ai::{ChatRequest, Content, Message, RequestProfile, StreamEvent, ToolMode, ToolSpec};

use crate::heuristics::{
    humanize_count, looks_mutating, parse_text_tool_calls, textcall_id_offset,
};
use crate::transcript::{
    NudgeKind, PROVIDER_VISIBLE_ASSISTANT_PLACEHOLDER,
    repair_invalid_tool_call_arguments_in_messages,
};
use crate::{FINALIZE_PROMPT, Ui, partial_text_tool_call_start};

use super::helpers::rate_limit_summary;

fn text_is_user_visible_answer(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty()
        && trimmed != PROVIDER_VISIBLE_ASSISTANT_PLACEHOLDER
        && ![
            "[answer retry:",
            "[answer rejected:",
            "[review retry:",
            "[plain-text tool retry:",
        ]
        .iter()
        .any(|prefix| trimmed.starts_with(prefix))
}

pub(super) fn turn_has_visible_assistant_text(messages: &[Message], turn_start: usize) -> bool {
    messages
        .get(turn_start..)
        .into_iter()
        .flatten()
        .any(|message| {
            message.role == hi_ai::Role::Assistant
                && message
                    .content
                    .iter()
                    .any(|c| matches!(c, Content::Text(t) if text_is_user_visible_answer(t)))
        })
}

impl crate::Agent {
    /// Generate and display the user-facing closeout. Returns `true` only when
    /// the provider supplied non-empty recap text.
    pub(super) async fn finalize_turn(&mut self, turn_start: usize, ui: &mut dyn Ui) -> bool {
        // Only send the current turn's messages (plus the system prompt for
        // context), not the entire session history. The recap only needs to
        // know what happened *this turn* — sending 40K tokens of old context
        // to produce a 200-token summary is pure waste.
        let Some(turn) = self.messages.as_slice().get(turn_start..) else {
            // Transcript replacement paths should re-anchor `turn_start`, but
            // finalization is a best-effort side call and must never crash a
            // completed turn if a future compaction path forgets to do so.
            return false;
        };
        let mut messages = Vec::with_capacity(turn.len() + 2);
        messages.push(self.minimal_system_message());
        messages.extend_from_slice(turn);
        messages.push(Message::user(FINALIZE_PROMPT));
        repair_invalid_tool_call_arguments_in_messages(&mut messages);

        let model = self.config.routing.model.clone();
        let request_policy = self.seal_chat_only_auxiliary_request(&model, 2048).await;
        let request = ChatRequest {
            model,
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: Arc::from(messages),
            tools: request_policy.tools,
            tool_envelope: Some(request_policy.envelope),
            max_tokens: request_policy.max_tokens,
            temperature: self.config.routing.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                compat: self.config.routing.compat,
                tool_mode: request_policy.tool_mode,
                stream_usage: None,
                deepseek_compat: self.config.routing.deepseek_compat,
                deepseek_strict: None,
                deepseek_thinking: None,
                output_token_parameter: self.config.routing.output_token_parameter,
            },
        };

        let mut recap = String::new();
        // Buffer the side-call response until it has passed the same answer
        // checks as a normal model response. Streaming a canned completion
        // claim here would briefly show the exact text the main loop rejected.
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => recap.push_str(&text),
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Warning(text) => ui.top_status(&text),
            StreamEvent::Reasoning(_) => {}
            StreamEvent::WireAudit(_) => {}
            StreamEvent::ToolCallDelta { .. } => {}
        };
        let timeout = self.side_call_timeout();
        let completion =
            match super::await_side_call(timeout, self.provider.stream(request, &mut sink)).await {
                Err(timeout) => {
                    // A recap is optional and the main turn has already settled.
                    ui.status(&format!(
                        "(final summary timed out after {:.1}s; turn already completed)",
                        timeout.as_secs_f64()
                    ));
                    return false;
                }
                Ok(Ok(completion)) => completion,
                Ok(Err(err)) => {
                    // Finalize is a side call — book its error usage without resetting
                    // the main conversation's `context_used` gauge.
                    self.add_side_error_usage(&err);
                    self.emit_usage(ui);
                    ui.status(&format!("(couldn't generate the final summary: {err})"));
                    return false;
                }
            };

        // Side call: spend counts, but its small request must not clobber the
        // main conversation's context gauge (see add_side_usage).
        self.add_side_usage(completion.usage);
        self.emit_usage(ui);

        // Fall back to the final content if the provider didn't stream text.
        if recap.trim().is_empty() {
            for c in &completion.content {
                if let Content::Text(t) = c {
                    recap.push_str(t);
                }
            }
        }
        if !text_is_user_visible_answer(&recap) {
            if !recap.trim().is_empty() {
                ui.status("(final summary was unusable; using a deterministic closeout)");
            }
            return false;
        }

        ui.assistant_text(&recap);
        ui.assistant_end();
        // Record both the synthetic request and the recap so roles alternate.
        // The recap is a text-only assistant message (no tool calls).
        self.messages
            .push_nudge(NudgeKind::Finalize, FINALIZE_PROMPT);
        self.messages.push_assistant(vec![Content::Text(recap)]);
        true
    }

    /// Always leave a settled turn with a concrete terminal message, even when
    /// the model-side recap timed out or returned another canned placeholder.
    /// This path makes no completion claim that verification cannot support.
    pub(super) fn emit_deterministic_closeout(&mut self, ui: &mut dyn Ui) {
        let closeout = if self.report.verify.failed() {
            "I stopped after the latest verification failed. I left the current changes in place for inspection instead of repeating the same repair; the failing check is shown above."
        } else if self.report.verify.passed() {
            "The code changes are in place and verification passed. The model did not produce a usable summary, so I closed the turn with this verified status instead of leaving it running."
        } else if !self.workspace.last_changed_files.is_empty() {
            "I stopped before the current changes could be verified. I left them in place for inspection instead of continuing without evidence."
        } else {
            "I could not complete this request after repeated attempts made no progress. I stopped instead of continuing to repeat the same steps; the last diagnostic is shown above."
        };
        ui.assistant_text(closeout);
        ui.assistant_end();

        // Persist the terminal answer too. When the last assistant entry is a
        // private repair marker, replace it rather than appending an adjacent
        // assistant turn; otherwise use the transcript's provider-safe append.
        let replaced_private_marker = self
            .messages
            .mutate_slice()
            .last_mut()
            .filter(|message| message.role == hi_ai::Role::Assistant)
            .filter(|message| {
                message.content.iter().all(|content| match content {
                    Content::Text(text) => !text_is_user_visible_answer(text),
                    Content::Thinking { .. } => true,
                    _ => false,
                })
            })
            .map(|message| {
                message.content = vec![Content::Text(closeout.to_string())];
            })
            .is_some();
        if !replaced_private_marker {
            self.messages
                .push_assistant(vec![Content::Text(closeout.to_string())]);
        }
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
    /// `steer: 2 verify · 1 retry · tool-repair` — components omitted when zero.
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
            parts.push(format!("{} tool-repair", t.repeat_nudges));
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
                        || tool.name == "run_program"
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
#[allow(
    clippy::too_many_arguments,
    reason = "the classifier deliberately receives the complete settled turn state"
)]
pub(super) fn classify_turn_outcome(
    verification_infrastructure_error: bool,
    verification_unstable: bool,
    last_verify: Option<bool>,
    changed_files: &[String],
    turn_had_mutation: bool,
    no_applicable_check: bool,
    independent_review_status: crate::ReviewStatus,
    skeptic_last_status: Option<crate::SkepticStatus>,
    ended_at_cap: bool,
    ended_at_deadline: bool,
    allow_unverified: bool,
) -> (
    crate::TurnStatus,
    crate::VerificationStatus,
    crate::ReviewStatus,
    crate::TurnStopReason,
) {
    use super::helpers::combined_review_status;
    use crate::verify::is_prose_only_path;
    use crate::{ReviewStatus, TurnStatus, TurnStopReason, VerificationStatus};

    // `no_applicable_check` is computed at settle from the verification mode,
    // obligation state, and model-run validation evidence. `Unverified` is for
    // "checks should have settled but did not" (see call-site comment in loop_).
    //
    // Deliberately NOT escalated to `Unverified` for code no stage exercised.
    // The obligation nudge (see `obligation::NoExecutableCheck`) asks the model
    // to run such code itself, but a model-run bash command is not a
    // verification *stage* — `verification_executions` stays empty either way.
    // Classifying that as Unverified would be unsatisfiable: even a model that
    // complied and showed passing output would still fail in every repo without
    // a detected pipeline. Evidence the model produced is surfaced through the
    // nudge and transcript, not by reclassifying the turn.
    let verification = if verification_infrastructure_error {
        VerificationStatus::InfrastructureError
    } else if last_verify == Some(true) {
        VerificationStatus::Passed
    } else if last_verify == Some(false) {
        VerificationStatus::Failed
    } else if (changed_files.is_empty() && !turn_had_mutation)
        || no_applicable_check
        || (!changed_files.is_empty() && changed_files.iter().all(|path| is_prose_only_path(path)))
    {
        VerificationStatus::NotApplicable
    } else {
        VerificationStatus::Unverified
    };
    let skeptic_review = match skeptic_last_status {
        Some(crate::SkepticStatus::Approved) => ReviewStatus::Passed,
        Some(crate::SkepticStatus::Objected) => ReviewStatus::Objected,
        // Escalated = intentional skip/scar, not a defect objection.
        Some(crate::SkepticStatus::Escalated) => ReviewStatus::Escalated,
        Some(crate::SkepticStatus::Unavailable) => ReviewStatus::Unavailable,
        None => ReviewStatus::NotRequired,
    };
    let review = combined_review_status(independent_review_status, skeptic_review);
    // Review Escalated does not fail the turn: the goal path already skipped
    // the step with a scar and continues the run.
    // A deterministic pass is the only evidence strong enough to overrule a
    // review objection. NotApplicable supplies no counter-evidence, so an
    // objection there remains a real failure.
    // Observed: a reviewer hallucinated version "0" from "0.1.0" and objected
    // to a turn whose tests all passed; a reviewer also objected to a prose-only
    // turn where no checks could run at all.
    let objection_overrides =
        review == ReviewStatus::Objected && verification != VerificationStatus::Passed;
    // Heuristic repair exhaustion is intentionally absent here. A normal turn
    // either settles or reports a typed failure. An explicit productive-work
    // cap is a failure unless the caller deliberately excludes an accepted
    // read-only wrap-up from `ended_at_cap`.
    let status = if verification_infrastructure_error
        || verification == VerificationStatus::Failed
        || objection_overrides
        || (verification == VerificationStatus::Unverified && !allow_unverified)
        || ended_at_cap
    {
        TurnStatus::Failed
    } else {
        TurnStatus::Completed
    };
    let stop_reason = if verification_infrastructure_error {
        TurnStopReason::InfrastructureFailure
    } else if verification_unstable {
        TurnStopReason::VerificationUnstable
    } else if verification == VerificationStatus::Failed {
        TurnStopReason::VerificationFailed
    } else if review == ReviewStatus::Objected {
        TurnStopReason::ReviewObjected
    } else if verification == VerificationStatus::Unverified && !allow_unverified {
        // Preserve the reason for the actual failure. A coincident work limit
        // is diagnostic, but must not make a failed unverified turn look like
        // a normal bounded settlement to consumers.
        TurnStopReason::VerificationUnavailable
    } else if ended_at_cap {
        TurnStopReason::StepLimit
    } else if ended_at_deadline {
        TurnStopReason::TimeLimit
    } else if verification == VerificationStatus::Unverified {
        TurnStopReason::VerificationUnavailable
    } else if verification == VerificationStatus::NotApplicable {
        TurnStopReason::NoApplicableVerification
    } else if review == ReviewStatus::Escalated {
        TurnStopReason::ReviewEscalated
    } else {
        TurnStopReason::Completed
    };
    (status, verification, review, stop_reason)
}

#[cfg(test)]
mod classify_tests {
    use super::{classify_turn_outcome, turn_has_visible_assistant_text};
    use crate::{ReviewStatus, TurnStatus, TurnStopReason, VerificationStatus};
    use hi_ai::{Content, Message};

    #[test]
    fn private_repair_markers_are_not_user_visible_answers() {
        for marker in [
            "[answer retry: generic completion placeholder rejected; provide the actual result]",
            "[answer rejected: generic completion placeholder repeated]",
            "[review retry: reason=no_evidence; required_next=read]",
            "[plain-text tool retry: no executable tool call was emitted]",
            crate::transcript::PROVIDER_VISIBLE_ASSISTANT_PLACEHOLDER,
        ] {
            let messages = vec![Message::assistant(vec![Content::Text(marker.into())])];
            assert!(
                !turn_has_visible_assistant_text(&messages, 0),
                "private/canned marker was counted as a user answer: {marker}"
            );
        }
    }

    #[test]
    fn concrete_assistant_text_is_user_visible() {
        let messages = vec![Message::assistant(vec![Content::Text(
            "Implemented SQLite persistence; cargo test passes.".into(),
        )])];
        assert!(turn_has_visible_assistant_text(&messages, 0));
    }

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
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(verification, VerificationStatus::Passed);
        assert_eq!(review, ReviewStatus::NotRequired);
        assert_eq!(stop, TurnStopReason::Completed);
    }

    #[test]
    fn no_change_turn_completes() {
        // A finished answer with no file changes completes without inventing
        // a synthetic failure state.
        let (status, verification, _, stop) = classify_turn_outcome(
            false,
            false,
            None,
            &[],
            false,
            true,
            ReviewStatus::NotRequired,
            None,
            false,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(verification, VerificationStatus::NotApplicable);
        assert_eq!(stop, TurnStopReason::NoApplicableVerification);
    }

    #[test]
    fn exhausted_heuristic_challenge_does_not_change_the_public_outcome() {
        let (status, _, _, stop) = classify_turn_outcome(
            false,
            false,
            None,
            &[],
            false,
            true,
            ReviewStatus::NotRequired,
            None,
            false,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(stop, TurnStopReason::NoApplicableVerification);
    }

    #[test]
    fn failed_when_unverified_and_not_allowed() {
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
        );
        assert_eq!(status, TurnStatus::Failed);
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
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(verification, VerificationStatus::NotApplicable);
        assert_eq!(stop, TurnStopReason::NoApplicableVerification);
    }

    #[test]
    fn mutation_with_no_applicable_check_is_not_applicable() {
        // The caller found no applicable verification obligation.
        let (status, verification, _, stop) = classify_turn_outcome(
            false,
            false,
            None,
            &["src/lib.rs".into()],
            true,
            true, // no_applicable_check
            ReviewStatus::NotRequired,
            None,
            false,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(verification, VerificationStatus::NotApplicable);
        assert_eq!(stop, TurnStopReason::NoApplicableVerification);
    }

    /// Completion-review transport failure is soft: green verify + IR Unavailable
    /// still completes. Goal skeptic Unavailable is fail-closed at goal advance;
    /// when folded into the public outcome it remains visible but does not alone
    /// mark the turn Failed (only Objected does).
    #[test]
    fn independent_review_unavailable_still_completes() {
        let (status, verification, review, stop) = classify_turn_outcome(
            false,
            false,
            Some(true),
            &["src/lib.rs".into()],
            true,
            false,
            ReviewStatus::Unavailable,
            None,
            false,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(verification, VerificationStatus::Passed);
        assert_eq!(review, ReviewStatus::Unavailable);
        assert_eq!(stop, TurnStopReason::Completed);
    }

    #[test]
    fn deterministic_pass_is_authoritative_over_heuristic_repair_state() {
        // The final workspace verifier proves the settled tree is green;
        // heuristic repair bookkeeping cannot override that evidence.
        let (status, verification, review, stop) = classify_turn_outcome(
            false,
            false,
            Some(true),
            &["src/lib.rs".into()],
            true,
            false,
            ReviewStatus::Unavailable,
            None,
            false,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(verification, VerificationStatus::Passed);
        assert_eq!(review, ReviewStatus::Unavailable);
        assert_eq!(stop, TurnStopReason::Completed);
    }

    #[test]
    fn baseline_pass_completes_a_no_change_turn() {
        let (status, verification, _, stop) = classify_turn_outcome(
            false,
            false,
            Some(true),
            &[],
            true,
            false,
            ReviewStatus::Unavailable,
            None,
            false,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(verification, VerificationStatus::Passed);
        assert_eq!(stop, TurnStopReason::Completed);
    }

    #[test]
    fn goal_skeptic_unavailable_is_visible_without_incompleting() {
        let (status, _, review, stop) = classify_turn_outcome(
            false,
            false,
            Some(true),
            &["src/lib.rs".into()],
            true,
            false,
            ReviewStatus::NotRequired,
            Some(crate::SkepticStatus::Unavailable),
            false,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(review, ReviewStatus::Unavailable);
        assert_eq!(stop, TurnStopReason::Completed);
    }

    #[test]
    fn goal_skeptic_escalated_completes_with_scar() {
        // Goal Escalated skips the step with a scar; public turn Completes so
        // callers do not treat an intentional skip as a failed repair.
        let (status, _, review, stop) = classify_turn_outcome(
            false,
            false,
            Some(true),
            &["src/lib.rs".into()],
            true,
            false,
            ReviewStatus::NotRequired,
            Some(crate::SkepticStatus::Escalated),
            false,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(review, ReviewStatus::Escalated);
        assert_eq!(stop, TurnStopReason::ReviewEscalated);
    }

    #[test]
    fn verified_turn_completes_with_objection_scar() {
        // Deterministic verification passed; the reviewer still objects after
        // its repair budget. The verified evidence wins: the turn completes
        // and the objection rides along as a scar, instead of a reviewer
        // false-positive stalling a passing task (observed: "0.1.0" misread
        // as version "0" marked a reward-1 benchmark task incomplete).
        let (status, _, review, stop) = classify_turn_outcome(
            false,
            false,
            Some(true),
            &["src/lib.rs".into()],
            true,
            false,
            ReviewStatus::Objected,
            Some(crate::SkepticStatus::Approved),
            false,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(review, ReviewStatus::Objected);
        assert_eq!(stop, TurnStopReason::ReviewObjected);
    }

    #[test]
    fn deadline_reports_time_limit_without_inventing_incompleteness() {
        // Work finished and verified with the budget nearly spent: the turn is
        // Completed, and TimeLimit records only that it stopped starting new
        // work. A deadline must not manufacture a failure.
        let (status, verification, _, stop) = classify_turn_outcome(
            false,
            false,
            Some(true),
            &["src/lib.rs".into()],
            true,
            false,
            ReviewStatus::NotRequired,
            None,
            false,
            true,
            false,
        );
        assert_eq!(status, TurnStatus::Completed);
        assert_eq!(verification, VerificationStatus::Passed);
        assert_eq!(stop, TurnStopReason::TimeLimit);
    }

    #[test]
    fn deadline_with_unverified_mutation_reports_verification_failure() {
        // Ran out of time mid-work: the mutation never got green checks, so
        // the missing verification remains the public failure reason.
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
            true,
            false,
        );
        assert_eq!(status, TurnStatus::Failed);
        assert_eq!(verification, VerificationStatus::Unverified);
        assert_eq!(stop, TurnStopReason::VerificationUnavailable);
    }

    #[test]
    fn step_limit_with_unverified_mutation_reports_verification_failure() {
        let (status, verification, _, stop) = classify_turn_outcome(
            false,
            false,
            None,
            &["src/lib.rs".into()],
            true,
            false,
            ReviewStatus::NotRequired,
            None,
            true,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Failed);
        assert_eq!(verification, VerificationStatus::Unverified);
        assert_eq!(stop, TurnStopReason::VerificationUnavailable);
    }

    #[test]
    fn step_limit_outranks_deadline_when_both_fire() {
        let (status, verification, _, stop) = classify_turn_outcome(
            false,
            false,
            Some(true),
            &["src/lib.rs".into()],
            true,
            false,
            ReviewStatus::NotRequired,
            None,
            true,
            true,
            false,
        );
        assert_eq!(status, TurnStatus::Failed);
        assert_eq!(verification, VerificationStatus::Passed);
        assert_eq!(stop, TurnStopReason::StepLimit);
    }

    #[test]
    fn objection_without_green_verification_keeps_teeth() {
        // No deterministic pass to outrank the reviewer — an exhausted
        // objection still marks the turn Failed.
        let (status, _, review, stop) = classify_turn_outcome(
            false,
            false,
            None,
            &["src/lib.rs".into()],
            true,
            false,
            ReviewStatus::Objected,
            None,
            false,
            false,
            true,
        );
        assert_eq!(status, TurnStatus::Failed);
        assert_eq!(review, ReviewStatus::Objected);
        assert_eq!(stop, TurnStopReason::ReviewObjected);
    }

    #[test]
    fn objection_with_not_applicable_verification_fails() {
        // No deterministic checks ran, so there is no proof strong enough to
        // overrule the review objection.
        let (status, _, review, stop) = classify_turn_outcome(
            false,
            false,
            None,
            &["README.md".into()],
            true,
            true, // no_applicable_check → NotApplicable
            ReviewStatus::Objected,
            None,
            false,
            false,
            false,
        );
        assert_eq!(status, TurnStatus::Failed);
        assert_eq!(review, ReviewStatus::Objected);
        assert_eq!(stop, TurnStopReason::ReviewObjected);
    }
}
