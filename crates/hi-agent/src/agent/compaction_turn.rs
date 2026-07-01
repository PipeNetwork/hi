//! Compaction drivers: `compact`/`compact_with` entry points and the
//! summarize / hybrid / elide-then-summarize-tail strategies, plus the
//! in-turn context elision and the `summarize` helper used by both compaction
//! and memory distillation.

use std::sync::Arc;

use anyhow::Result;
use hi_ai::{ChatRequest, Content, Message, RequestProfile, StreamEvent, ToolMode};

use crate::SUMMARIZE_PROMPT;
use crate::Ui;
use crate::compaction::{self, CompactionKind};

impl crate::Agent {
    /// The compaction strategy configured for this session.
    pub fn compaction_kind(&self) -> CompactionKind {
        self.config.compaction.clone()
    }

    /// Reclaim context using the session's configured strategy. Compaction is
    /// persisted as a replacement boundary, so resuming starts from the
    /// compacted transcript.
    pub async fn compact(&mut self, ui: &mut dyn Ui) -> Result<()> {
        self.compact_with(self.config.compaction.clone(), ui).await
    }

    /// Reclaim context using a specific strategy (e.g. `/compact <kind>`).
    pub async fn compact_with(&mut self, kind: CompactionKind, ui: &mut dyn Ui) -> Result<()> {
        match kind {
            CompactionKind::Summarize => self.compact_summarize(ui).await,
            CompactionKind::Hybrid { keep_recent } => self.compact_hybrid(keep_recent, ui).await,
            CompactionKind::ElideToolOutput { keep_recent } => self.compact_elide(keep_recent, ui),
            CompactionKind::ElideThenSummarizeTail { keep_recent } => {
                self.compact_elide_then_summarize_tail(keep_recent, ui)
                    .await
            }
        }
    }

    /// Provider byte/request caps can be lower than the model catalog's token
    /// window, so a request can be rejected before usage is reported and before
    /// the normal auto-compaction trigger fires. Keep the latest user request,
    /// drop earlier in-memory context once, and let the loop retry immediately.
    pub(crate) fn retry_after_request_too_large(
        &mut self,
        input: &str,
        turn_start: usize,
        ui: &mut dyn Ui,
    ) -> Result<bool> {
        if turn_start <= 1 {
            return Ok(false);
        }

        self.replace_history_with_compaction(vec![self.system_message()])?;
        self.messages.push_user(format!(
            "[Earlier conversation context was omitted because the provider rejected the request \
             as too large. Continue from this latest user request; ask for missing details if the \
             omitted context is required.]\n\n{input}"
        ));
        self.context_used = 0;
        ui.status(
            "provider rejected the request as too large; dropped prior conversation context and retrying",
        );
        Ok(true)
    }

    /// Summarize the whole conversation and reset to system + summary.
    async fn compact_summarize(&mut self, ui: &mut dyn Ui) -> Result<()> {
        // Need at least one exchange beyond the system prompt to summarize.
        if self.messages.len() <= 1 {
            ui.status("nothing to compact yet");
            return Ok(());
        }
        // Own the slice so it doesn't borrow `self` across the `&mut self` call.
        let slice = self.messages.as_slice()[1..].to_vec();
        let Some(summary) = self.summarize(&slice, ui).await? else {
            ui.status("compaction produced no summary; keeping history");
            return Ok(());
        };
        let system = self.system_message();
        let next = vec![
            system,
            Message::user(format!("[Summary of the conversation so far]\n\n{summary}")),
        ];
        self.replace_history_with_compaction(next)?;
        ui.status("✓ compacted — context reset to the summary");
        Ok(())
    }

    /// Keep the last `keep_recent` user turns verbatim; summarize everything
    /// older and fold the brief into the first kept turn. Folding (rather than
    /// inserting a separate summary message) avoids two consecutive user
    /// messages, which some providers reject.
    async fn compact_hybrid(&mut self, keep_recent: usize, ui: &mut dyn Ui) -> Result<()> {
        if keep_recent == 0 {
            return self.compact_summarize(ui).await;
        }
        let Some(split) = compaction::recent_split(self.messages.as_slice(), keep_recent) else {
            // Nothing older than the recent window — summarize everything so a
            // triggered compaction still makes progress.
            return self.compact_summarize(ui).await;
        };
        let old = self.messages.as_slice()[1..split].to_vec();
        let Some(summary) = self.summarize(&old, ui).await? else {
            ui.status("compaction produced no summary; keeping history");
            return Ok(());
        };

        let system = self.system_message();
        let mut recent = self.messages.as_slice()[split..].to_vec();
        let head = recent[0].text();
        recent[0] = Message::user(format!(
            "[Summary of earlier conversation]\n\n{summary}\n\n---\n\n{head}"
        ));
        let mut next = Vec::with_capacity(recent.len() + 1);
        next.push(system);
        next.extend(recent);
        self.replace_history_with_compaction(next)?;
        ui.status("✓ compacted — kept recent turns, summarized the rest");
        Ok(())
    }

    /// Elide-first, summarize-only-the-conversational-tail. Keep the recent
    /// `keep_recent` turns verbatim (their tool results elided, skeleton kept).
    /// For old turns: **keep** the tool-bearing ones in history with their bulky
    /// output elided (the call/result skeleton stays, so the model remembers
    /// "I read file X" — just without the verbatim output), and summarize only
    /// the tool-free Q&A turns into a brief folded into the first kept turn. A
    /// pure tool-heavy session with no old Q&A makes no model call at all — just
    /// the deterministic elision.
    async fn compact_elide_then_summarize_tail(
        &mut self,
        keep_recent: usize,
        ui: &mut dyn Ui,
    ) -> Result<()> {
        if keep_recent == 0 {
            return self.compact_summarize(ui).await;
        }
        let Some(split) = compaction::recent_split(self.messages.as_slice(), keep_recent) else {
            // Nothing older than the recent window — fall back to summarizing
            // everything so a triggered compaction still makes progress.
            return self.compact_summarize(ui).await;
        };
        // Elide bulky tool output in an owned copy. The live transcript is only
        // replaced after the durable boundary is recorded.
        let mut working = self.messages.as_slice().to_vec();
        compaction::elide_tool_outputs(&mut working, split);

        // Summarize only the conversational (tool-free) old tail. The tool-bearing
        // old turns are NOT summarized — they stay in history, elided.
        let convo = compaction::conversational_tail(&working, split);
        let summary = if convo.is_empty() {
            None
        } else {
            self.summarize(&convo, ui).await?
        };

        // Rebuild: system + old tool-bearing turns (elided, kept) + recent turns
        // (with the Q&A summary folded into the first recent turn). The old
        // Q&A-only messages are dropped (replaced by the summary).
        let system = self.system_message();
        let old = compaction::tool_bearing_turns(&working, split);
        let mut recent = working[split..].to_vec();
        let had_summary = summary.is_some();
        if let Some(summary) = summary {
            // Fold the brief into the first kept turn (avoids two consecutive
            // user messages, which some providers reject) — same shape as
            // `compact_hybrid`. If the old tool-bearing region is non-empty, the
            // summary sits between it and the recent turns as a user message.
            // A preserved tool-bearing turn ends with either a ToolResult or a
            // final Assistant answer, so the folded recent User turn alternates
            // correctly.
            let head = recent[0].text();
            recent[0] = Message::user(format!(
                "[Summary of earlier conversation]\n\n{summary}\n\n---\n\n{head}"
            ));
        }
        let mut next = Vec::with_capacity(1 + old.len() + recent.len());
        next.push(system);
        next.extend(old);
        next.extend(recent);
        self.replace_history_with_compaction(next)?;
        if had_summary {
            ui.status("✓ compacted — elided old tool output, summarized the Q&A tail");
        } else {
            ui.status("✓ compacted — elided old tool output (no Q&A tail to summarize)");
        }
        Ok(())
    }

    /// Deterministically shrink the bulky output of old tool calls. No model
    /// call. Persisted as a replacement boundary, like the summary strategies.
    fn compact_elide(&mut self, keep_recent: usize, ui: &mut dyn Ui) -> Result<()> {
        // Only turns older than the recent window are eligible; if everything is
        // recent there's nothing to elide.
        let Some(split) = compaction::recent_split(self.messages.as_slice(), keep_recent) else {
            ui.status("nothing old to elide");
            return Ok(());
        };
        let mut next = self.messages.as_slice().to_vec();
        let freed = compaction::elide_tool_outputs(&mut next, split);
        if freed > 0 {
            self.replace_history_with_compaction(next)?;
            ui.status(&format!(
                "✓ elided ~{}k chars of old tool output",
                freed / 1000
            ));
        } else {
            ui.status("nothing old to elide");
        }
        Ok(())
    }

    pub(crate) fn elide_in_turn_context_if_needed(
        &mut self,
        _ui: &mut dyn Ui,
        safety_window: Option<u32>,
    ) {
        if !self.config.auto_compact {
            return;
        }
        let window = match (self.config.context_window, safety_window) {
            (Some(configured), Some(safety)) => configured.min(safety),
            (Some(configured), None) => configured,
            (None, Some(safety)) => safety,
            (None, None) => {
                return;
            }
        };
        if window == 0 {
            return;
        };

        let used = compaction::estimate_tokens(self.messages.as_slice());
        if used * 100 < u64::from(window) * self.config.in_turn_elide_percent {
            return;
        }

        let freed = compaction::elide_tool_outputs_except_recent(
            self.messages.mutate_slice(),
            self.config.in_turn_keep_tool_results,
        );
        if freed == 0 {
            return;
        }

        self.context_used = 0;
    }

    /// Run the summarization model call over `slice`, returning the summary text
    /// (trimmed), or `None` if the model produced nothing. Shared by the
    /// Summarize and Hybrid strategies.
    async fn summarize(&mut self, slice: &[Message], ui: &mut dyn Ui) -> Result<Option<String>> {
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

        let request = ChatRequest {
            model: self.config.model.clone(),
            messages: Arc::from(messages),
            tools: Arc::new([]), // summarizing — no tool use
            max_tokens: 1024,    // throwaway call — summaries are short
            temperature: self.config.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            profile: RequestProfile {
                compat: self.config.compat,
                tool_mode: ToolMode::ChatOnly,
                stream_usage: None,
            },
        };

        let mut summary = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => {
                summary.push_str(&text);
                ui.assistant_text(&text);
            }
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Reasoning(_) => {}
        };
        let completion = match self.provider.stream(request, &mut sink).await {
            Ok(completion) => completion,
            Err(err) => {
                self.add_error_usage(&err);
                // Flush any partially-streamed summary text before returning.
                ui.assistant_end();
                let _ = self.persist();
                return Err(err);
            }
        };
        self.add_usage(completion.usage);
        let _ = self.persist();
        ui.usage(
            self.totals.input_tokens,
            self.totals.output_tokens,
            self.context_used,
            self.config.context_window,
        );

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
