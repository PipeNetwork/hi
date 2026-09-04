//! Compaction drivers: `compact`/`compact_with` entry points and the
//! summarize / hybrid / elide-then-summarize-tail strategies, plus the
//! in-turn context elision and the `summarize` helper used by both compaction
//! and memory distillation.

use std::sync::Arc;

use anyhow::Result;
use hi_ai::{
    ChatRequest, Content, Message, ProviderError, ProviderErrorKind, RequestProfile, Role,
    StreamEvent,
};

use crate::Ui;
use crate::compaction::{self, CompactionKind};
use crate::transcript::repair_invalid_tool_call_arguments_in_messages;
use crate::{COMPACTION_REFERENCE_PREFIX, COMPACTION_SUMMARY_END, SUMMARIZE_PROMPT};

pub(crate) struct ContextPreflight {
    pub(crate) max_tokens: u32,
    pub(crate) dropped_prior_context: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ContextWindowLimits {
    pub(crate) safety: Option<u32>,
    pub(crate) provider: Option<u32>,
}

/// Cap recovered recap text so a too-large retry cannot immediately overflow
/// again. ~8k chars is enough for a design recap without the tool-output bulk
/// that caused the overflow.
const DROPPED_CONTEXT_RECAP_MAX_CHARS: usize = 8_000;

/// Last non-empty assistant text in `messages`, truncated to
/// [`DROPPED_CONTEXT_RECAP_MAX_CHARS`]. Skips tool-call-only assistant
/// turns so a mid-loop "let me read X" does not hide an earlier recap when
/// the recap is the last text-bearing assistant message.
pub(crate) fn last_assistant_recap(messages: &[Message]) -> Option<String> {
    let recap = messages.iter().rev().find_map(|message| {
        if message.role != Role::Assistant {
            return None;
        }
        let text = message.text();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })?;
    Some(truncate_recap(&recap))
}

fn truncate_recap(recap: &str) -> String {
    if recap.chars().count() <= DROPPED_CONTEXT_RECAP_MAX_CHARS {
        return recap.to_string();
    }
    let mut truncated: String = recap
        .chars()
        .take(DROPPED_CONTEXT_RECAP_MAX_CHARS)
        .collect();
    if let Some(at) = truncated.rfind('\n')
        && at > DROPPED_CONTEXT_RECAP_MAX_CHARS / 2
    {
        truncated.truncate(at);
    }
    truncated.push_str("\n…[recap truncated]");
    truncated
}

fn dropped_context_retry_prompt(
    input: &str,
    recap: Option<&str>,
    provider_rejected: bool,
) -> String {
    let reason = if provider_rejected {
        "the provider rejected the request as too large"
    } else {
        "the next request would exceed the model context window"
    };
    let mut body = format!(
        "[Earlier conversation context was omitted because {reason}. Continue from this latest \
         user request; ask for missing details if the omitted context is required.]\n\n{input}"
    );
    if let Some(recap) = recap.map(str::trim).filter(|text| !text.is_empty()) {
        body.push_str(
            "\n\n[Last assistant recap before context was dropped — recovered session state, \
             not a new instruction]\n",
        );
        body.push_str(recap);
    }
    body
}

fn dropped_context_retry_status(provider_rejected: bool, kept_recap: bool) -> String {
    let trigger = if provider_rejected {
        "provider rejected the request as too large"
    } else {
        "request would exceed the model context window"
    };
    let recap = if kept_recap {
        "kept last assistant recap"
    } else {
        "no assistant recap to keep"
    };
    format!("{trigger}; dropped prior conversation context, {recap}, and retrying")
}

impl crate::Agent {
    /// The compaction strategy configured for this session.
    pub fn compaction_kind(&self) -> CompactionKind {
        self.config.memory.compaction.clone()
    }

    /// Provider byte/request caps can be lower than the model catalog's token
    /// window, so a request can be rejected before usage is reported and before
    /// the normal auto-compaction trigger fires.
    ///
    /// First recovery: hybrid compact — summarize older turns locally (no extra
    /// model call, which could 413 again), keep the recent tail including the
    /// current user message. Second recovery: drop prior history and keep the
    /// latest prompt + last recap.
    pub(crate) fn retry_after_request_too_large_compact(
        &mut self,
        ui: &mut dyn Ui,
    ) -> Result<bool> {
        let Some(split) = compaction::recent_split(self.messages.as_slice(), 1) else {
            return Ok(false);
        };
        if split <= 1 {
            return Ok(false);
        }
        let old = self.messages.as_slice()[1..split].to_vec();
        let summary = local_overflow_summary(&old);
        let system = self.system_message();
        let mut recent = self.messages.as_slice()[split..].to_vec();
        if recent.is_empty() {
            return Ok(false);
        }
        let recent_len = recent.len();
        compaction::elide_tool_outputs(&mut recent, recent_len);
        let head = recent[0].text();
        recent[0] = Message::user(fold_reference_summary_into_user(&summary, &head));
        let mut next = Vec::with_capacity(recent.len() + 1);
        next.push(system);
        next.extend(recent);
        self.replace_history_with_compaction(next)?;
        self.runtime.invalidate_context_after_compaction();
        self.report.context_used = 0;
        ui.status(
            "provider rejected the request as too large; summarized prior turns, kept the latest prompt, and retrying",
        );
        Ok(true)
    }

    /// Keep the latest user request and the last assistant recap, drop the
    /// remaining history once, and let the loop retry immediately.
    pub(crate) fn retry_after_request_too_large(
        &mut self,
        input: &str,
        _turn_start: usize,
        ui: &mut dyn Ui,
    ) -> Result<bool> {
        if self.messages.len() <= 1 {
            return Ok(false);
        }
        let live = self
            .messages
            .as_slice()
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .map(|message| message.text())
            .unwrap_or_default();
        if live.contains("Earlier conversation context was omitted") {
            return Ok(false);
        }
        let compacted = live.contains("[CONTEXT COMPACTION");
        let user_turns = compaction::user_turn_starts(self.messages.as_slice()).len();
        if user_turns <= 1 && !compacted {
            return Ok(false);
        }

        let recap = last_assistant_recap(self.messages.as_slice());
        self.replace_history_with_compaction(vec![self.system_message()])?;
        self.runtime.invalidate_context_after_compaction();
        self.messages
            .push_user(dropped_context_retry_prompt(input, recap.as_deref(), true));
        self.report.context_used = 0;
        ui.status(&dropped_context_retry_status(true, recap.is_some()));
        Ok(true)
    }

    /// Refuse or shrink a request that can be locally estimated to exceed the
    /// advertised context window. This keeps predictable overflow from reaching
    /// the API as a 400, while preserving the provider-reactive retry for byte
    /// caps or unexpectedly stricter providers.
    pub(crate) fn ensure_request_fits_context(
        &mut self,
        input: &str,
        turn_start: usize,
        requested_max_tokens: u32,
        request_tool_schema_tokens: u64,
        windows: ContextWindowLimits,
        ui: &mut dyn Ui,
    ) -> Result<ContextPreflight> {
        // The *soft* window is the min of the real model window and any
        // read-only safety window (12k). It only ever triggers non-destructive
        // tool-output elision — keeping a read-only review turn lean — and must
        // NOT gate the destructive steps below. The *hard* window is the real
        // model window alone; only exceeding it may drop prior history or
        // hard-fail. Otherwise an ordinary read-only question on a 200k-window
        // model would durably discard the whole session the instant its context
        // crossed the 12k safety preference.
        let hard_window =
            min_context_windows(self.effective_context_window(None), windows.provider);
        let soft_window =
            min_context_windows(self.effective_context_window(windows.safety), hard_window);

        // 1. Already within the soft preference → nothing to do.
        if let Some(soft) = soft_window
            && soft > 0
            && self.request_estimated_tokens(requested_max_tokens, request_tool_schema_tokens)
                <= u64::from(soft)
        {
            return Ok(ContextPreflight {
                max_tokens: requested_max_tokens,
                dropped_prior_context: false,
            });
        }

        // 2. Over the soft preference: try non-destructive elision (no model
        //    call — see note below), and if that brings us under, we're done.
        if self.config.memory.auto_compact
            && let Some(soft) = soft_window
            && soft > 0
        {
            let freed = compaction::elide_tool_outputs_except_recent(
                self.messages.mutate_slice(),
                self.config.memory.in_turn_keep_tool_results,
            );
            if freed > 0 {
                self.runtime.invalidate_context_after_compaction();
                self.report.context_used = 0;
                ui.status(&format!(
                    "elided ~{}k chars of old tool output before request to fit context",
                    freed / 1000
                ));
            }
            if self.request_estimated_tokens(requested_max_tokens, request_tool_schema_tokens)
                <= u64::from(soft)
            {
                return Ok(ContextPreflight {
                    max_tokens: requested_max_tokens,
                    dropped_prior_context: false,
                });
            }

            // Do not run model-based summarizing compaction here: the latest
            // user prompt has already been appended, and summarizing the whole
            // transcript can erase the exact task we are about to answer. The
            // pre-turn auto-compact path still performs summarization before a
            // new prompt is added; at send time we either elide deterministic
            // bulk or drop prior context while preserving the latest prompt.
        }

        // 3. Destructive recovery is gated on the REAL model window only. With
        //    no configured window we can't tell — so proceed rather than drop.
        let Some(window) = hard_window else {
            return Ok(ContextPreflight {
                max_tokens: requested_max_tokens,
                dropped_prior_context: false,
            });
        };
        if window == 0
            || self.request_estimated_tokens(requested_max_tokens, request_tool_schema_tokens)
                <= u64::from(window)
        {
            return Ok(ContextPreflight {
                max_tokens: requested_max_tokens,
                dropped_prior_context: false,
            });
        }

        let mut dropped_prior_context = false;
        if turn_start > 1 {
            let recap = last_assistant_recap(self.messages.as_slice());
            self.replace_history_with_compaction(vec![self.system_message()])?;
            self.runtime.invalidate_context_after_compaction();
            self.messages
                .push_user(dropped_context_retry_prompt(input, recap.as_deref(), false));
            self.report.context_used = 0;
            dropped_prior_context = true;
            ui.status(&dropped_context_retry_status(false, recap.is_some()));

            if self.request_estimated_tokens(requested_max_tokens, request_tool_schema_tokens)
                <= u64::from(window)
            {
                return Ok(ContextPreflight {
                    max_tokens: requested_max_tokens,
                    dropped_prior_context,
                });
            }
        }

        let prompt_estimate = self.request_estimated_tokens(0, request_tool_schema_tokens);
        if prompt_estimate < u64::from(window) {
            let available = (u64::from(window) - prompt_estimate).min(u64::from(u32::MAX)) as u32;
            if available > 0 && available < requested_max_tokens {
                ui.nudge(&format!(
                    "request would exceed the model context window; reducing max_tokens from {requested_max_tokens} to {available}"
                ));
                return Ok(ContextPreflight {
                    max_tokens: available,
                    dropped_prior_context,
                });
            }
        }

        let estimated =
            self.request_estimated_tokens(requested_max_tokens, request_tool_schema_tokens);
        ui.status(
            "request would exceed the model context window even after local context recovery; shorten the prompt or attached input, then retry",
        );
        Err(ProviderError::new(
            ProviderErrorKind::RequestTooLarge,
            format!(
                "estimated request context {estimated} tokens exceeds model context window of {window} tokens"
            ),
        )
        .into())
    }

    fn effective_context_window(&self, safety_window: Option<u32>) -> Option<u32> {
        match (self.config.routing.context_window, safety_window) {
            (Some(configured), Some(safety)) => Some(configured.min(safety)),
            (Some(configured), None) => Some(configured),
            (None, Some(safety)) => Some(safety),
            (None, None) => None,
        }
    }

    /// Window used to decide when old tool output is worth stubbing. Falls back
    /// to [`crate::FALLBACK_CONTEXT_WINDOW`] when `/models` omitted a window so
    /// a long mutation loop still elides. Never used as the hard drop bound.
    fn occupancy_context_window(&self, safety_window: Option<u32>) -> u32 {
        match (
            self.config
                .routing
                .context_window
                .filter(|window| *window > 0),
            safety_window.filter(|window| *window > 0),
        ) {
            (Some(configured), Some(safety)) => configured.min(safety),
            (Some(configured), None) => configured,
            (None, Some(safety)) => safety,
            (None, None) => crate::FALLBACK_CONTEXT_WINDOW,
        }
    }

    fn request_estimated_tokens(&self, max_tokens: u32, request_tool_schema_tokens: u64) -> u64 {
        compaction::estimate_tokens(self.messages.as_slice())
            .saturating_add(request_tool_schema_tokens)
            .saturating_add(u64::from(max_tokens))
    }

    pub(crate) fn elide_in_turn_context_if_needed(
        &mut self,
        _ui: &mut dyn Ui,
        safety_window: Option<u32>,
    ) {
        if !self.config.memory.auto_compact {
            return;
        }
        let window = self.occupancy_context_window(safety_window);
        if window == 0 {
            return;
        }

        let used = compaction::estimate_tokens(self.messages.as_slice())
            .saturating_add(hi_ai::estimate_tool_schema_tokens(&self.tools));
        if used * 100 < u64::from(window) * self.config.memory.in_turn_elide_percent {
            return;
        }

        let freed = compaction::elide_tool_outputs_except_recent(
            self.messages.mutate_slice(),
            self.config.memory.in_turn_keep_tool_results,
        );
        if freed == 0 {
            return;
        }

        self.report
            .last_turn_telemetry
            .record_compaction(crate::CompactionEvent {
                freed_chars: freed as u64,
                keep_recent: self.config.memory.in_turn_keep_tool_results,
            });
        self.runtime.invalidate_context_after_compaction();
        self.report.context_used = 0;
    }

    /// Run the summarization model call over `slice`, returning the summary text
    /// (trimmed), or `None` if the model produced nothing. Shared by the
    /// Summarize and Hybrid strategies.
    pub(super) async fn summarize(
        &mut self,
        slice: &[Message],
        ui: &mut dyn Ui,
    ) -> Result<Option<String>> {
        ui.status("compacting the conversation…");

        // Elide bulky tool outputs before sending to the model — the summary
        // doesn't need verbatim command output, just the conversation shape.
        // This can cut input tokens by 50-80% on tool-heavy sessions.
        let mut slice_owned: Vec<Message> = slice.to_vec();
        let len = slice_owned.len();
        compaction::elide_tool_outputs(&mut slice_owned, len);

        let mut messages = Vec::with_capacity(slice_owned.len() + 2);
        messages.push(self.minimal_system_message());
        messages.extend_from_slice(&slice_owned);
        messages.push(Message::user(SUMMARIZE_PROMPT));
        repair_invalid_tool_call_arguments_in_messages(&mut messages);

        let model = self.config.routing.model.clone();
        let request_policy = self.seal_chat_only_auxiliary_request(&model, 1024).await;
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

        let mut summary = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => {
                summary.push_str(&text);
                ui.assistant_text(&text);
            }
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Warning(text) => ui.top_status(&text),
            StreamEvent::Reasoning(_) => {}
            StreamEvent::WireAudit(_) => {}
            StreamEvent::ToolCallDelta { .. } => {}
        };
        let timeout = self.side_call_timeout();
        let completion = match crate::agent::turn::await_side_call(
            timeout,
            self.provider.stream(request, &mut sink),
        )
        .await
        {
            Err(timeout) => {
                ui.assistant_end();
                ui.status(&format!(
                    "compaction summary timed out after {:.1}s; keeping the existing history",
                    timeout.as_secs_f64()
                ));
                let _ = self.persist();
                return Ok(None);
            }
            Ok(Ok(completion)) => completion,
            Ok(Err(err)) => {
                // Summarize is a side call — don't let its request size clobber
                // the main conversation's `context_used` gauge.
                self.add_side_error_usage(&err);
                self.emit_usage(ui);
                // Flush any partially-streamed summary text before returning.
                ui.assistant_end();
                let _ = self.persist();
                return Err(err);
            }
        };
        self.add_side_usage(completion.usage);
        let _ = self.persist();
        self.emit_usage(ui);

        // Fall back to the final content if the provider didn't stream text.
        // Emit it through the UI before assistant_end so the user sees the
        // summary even when the provider returned text only in the completion
        // object (not via stream deltas).
        if summary.trim().is_empty() {
            for c in &completion.content {
                if let Content::Text(t) = c {
                    summary.push_str(t);
                    ui.assistant_text(t);
                }
            }
        }
        ui.assistant_end();
        let summary = summary.trim();
        Ok((!summary.is_empty()).then(|| summary.to_string()))
    }
}

fn min_context_windows(left: Option<u32>, right: Option<u32>) -> Option<u32> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

const MAX_COMPACTION_SUMMARY_CHARS: usize = 6_000;

fn clip_summary(summary: &str) -> String {
    if summary.chars().count() <= MAX_COMPACTION_SUMMARY_CHARS {
        return summary.to_string();
    }
    let clipped: String = summary
        .chars()
        .take(MAX_COMPACTION_SUMMARY_CHARS.saturating_sub(1))
        .collect();
    format!("{clipped}…")
}

pub(super) fn reference_summary_block(summary: &str) -> String {
    format!(
        "{COMPACTION_REFERENCE_PREFIX}\n\n{}\n\n{COMPACTION_SUMMARY_END}",
        clip_summary(summary)
    )
}

pub(super) fn fold_reference_summary_into_user(summary: &str, latest_user: &str) -> String {
    format!(
        "{}\n\n--- LATEST USER MESSAGE ---\n\n{}",
        reference_summary_block(summary),
        latest_user
    )
}

fn local_overflow_summary(old: &[Message]) -> String {
    let mut parts = Vec::new();
    for message in old {
        match message.role {
            Role::User => {
                let trimmed = message.text();
                let trimmed = trimmed.trim();
                if !trimmed.is_empty() {
                    parts.push(format!("User: {}", clip_overflow_part(trimmed, 400)));
                }
            }
            Role::Assistant => {
                let trimmed = message.text();
                let trimmed = trimmed.trim();
                if !trimmed.is_empty() {
                    parts.push(format!("Assistant: {}", clip_overflow_part(trimmed, 400)));
                }
            }
            Role::System | Role::Tool => {}
        }
    }
    if parts.is_empty() {
        "Prior conversation compacted after a context overflow.".to_string()
    } else {
        clip_overflow_part(&parts.join("\n"), 4_000)
    }
}

fn clip_overflow_part(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{clipped}…")
}

#[cfg(test)]
mod recap_tests {
    use super::*;
    use hi_ai::Content;

    #[test]
    fn last_assistant_recap_skips_tool_only_turns_and_keeps_text() {
        let messages = vec![
            Message::system("sys"),
            Message::user("review the tui"),
            Message::assistant(vec![Content::Text(
                "Gap #1: fold stream_area into the Run row.".into(),
            )]),
            Message::assistant(vec![Content::ToolCall {
                id: "read-1".into(),
                name: "read".into(),
                arguments: r#"{"path":"render.rs"}"#.into(),
            }]),
        ];
        assert_eq!(
            last_assistant_recap(&messages).as_deref(),
            Some("Gap #1: fold stream_area into the Run row.")
        );
    }

    #[test]
    fn last_assistant_recap_truncates_long_text() {
        let long = "recap line\n".repeat(2_000);
        let messages = vec![Message::assistant(vec![Content::Text(long.clone())])];
        let recap = last_assistant_recap(&messages).expect("recap");
        assert!(recap.chars().count() < long.chars().count());
        assert!(recap.contains("…[recap truncated]"));
        assert!(recap.contains("recap line"));
    }

    #[test]
    fn dropped_context_retry_prompt_includes_recap() {
        let prompt = dropped_context_retry_prompt(
            "start with gap #1",
            Some("Fold stream_area into live_run_tail_lines."),
            true,
        );
        assert!(prompt.contains("start with gap #1"));
        assert!(prompt.contains("Last assistant recap"));
        assert!(prompt.contains("Fold stream_area into live_run_tail_lines."));
        assert!(prompt.contains("provider rejected the request as too large"));
    }
}
