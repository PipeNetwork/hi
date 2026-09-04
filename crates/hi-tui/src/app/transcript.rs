//! `App` methods: transcript.

mod outcome;
mod run_output;
mod selection;
mod steering;

use run_output::{bash_output_is_idle, bash_process_live, is_missing_background_process_result};
use steering::{
    ExploreChrome, absorb_explore_chrome, append_assistant_line, is_steering_assistant_line,
    is_steering_assistant_text, last_entry_is_blank, normalize_steering_text,
};

use std::time::Instant;

use hi_agent::ui::tool_label;
use ratatui::style::Style;
use ratatui::text::Line;

use crate::activity_feed::{self, ActivityBlock, ActivityKind, ExploreVerb, label_detail};
use crate::event::UiEvent;
use crate::render::markdown_line;
use crate::theme::theme;
use crate::{MAX_EVENT_LOG, MAX_TRANSCRIPT_LINES, TranscriptEntry, TurnEventKind};

impl crate::App {
    pub(crate) fn push(&mut self, line: Line<'static>) {
        // Anything pushed directly ends a streaming table, so emit it first and
        // keep the ordering correct.
        self.flush_table();
        self.transcript.push(TranscriptEntry::Line(line));
        self.bump_transcript();
        self.cap_transcript();
    }

    /// Push a live-updating progress line: while it remains the LAST
    /// transcript entry (tracked via `index`) and still looks like one of our
    /// progress lines (`marker` guard against index drift after capping),
    /// the line is replaced in place — one smoothly-updating bar instead of
    /// a line of spam per second.
    pub(crate) fn push_or_replace_progress(
        &mut self,
        index: &mut Option<usize>,
        marker: &str,
        line: Line<'static>,
    ) {
        if let Some(at) = *index
            && at + 1 == self.transcript.len()
            && matches!(
                self.transcript.get(at),
                Some(TranscriptEntry::Line(existing))
                    if existing.spans.iter().any(|span| span.content.contains(marker))
            )
        {
            self.transcript[at] = TranscriptEntry::Line(line);
            self.bump_transcript();
            return;
        }
        self.push(line);
        *index = Some(self.transcript.len().saturating_sub(1));
    }

    /// Push a distinct user-prompt echo so rendering can pin it as a sticky header.
    pub(crate) fn push_user_prompt(&mut self, line: Line<'static>) {
        self.record_projected_user_prompt(&line);
        self.freeze_verb_group();
        if self.following {
            self.page_flip_on_send = true;
        }
        self.transcript.push(TranscriptEntry::UserPrompt {
            line,
            at: std::time::SystemTime::now(),
        });
        self.bump_transcript();
        self.cap_transcript();
    }

    /// Bound the transcript so a very long session can't overflow the u16 scroll
    /// range, slow the per-frame render clone, or grow memory without limit. Older
    /// lines scroll off the top (the full session is still in the JSONL log). Only
    /// trims while pinned to the bottom, so a reader scrolled up isn't yanked by
    /// the offsets shifting underneath them. Sets `trimmed` so the render shows a
    /// "↑ N lines compacted" marker at the top of the transcript.
    pub(crate) fn cap_transcript(&mut self) {
        if self.following && self.transcript.len() > MAX_TRANSCRIPT_LINES {
            let excess = self.transcript.len() - MAX_TRANSCRIPT_LINES;
            self.transcript.drain(..excess);
            self.trimmed = self.trimmed.saturating_add(excess as u64);
            self.bump_transcript();
        }
    }

    /// Re-pin the view to the latest output. Called on explicit user actions (a
    /// new turn, a command's output) — not on streaming appends, so a reader who
    /// scrolled up stays put.
    pub(crate) fn follow(&mut self) {
        self.following = true;
        self.page_flip_on_send = false;
    }

    /// Show the inline `/btw` overlay (idempotent). Auto-called on first side activity.
    pub(crate) fn open_btw_pane(&mut self) {
        self.show_btw = true;
    }

    /// Grok-style overlay derived from the live thread (None when dismissed).
    pub(crate) fn btw_overlay(&self) -> Option<crate::btw::BtwOverlayState> {
        crate::btw::overlay_from_thread(self.show_btw, &self.btw_thread, self.btw_scroll)
    }

    /// Persist a finished overlay answer into the transcript as a collapsed
    /// `/btw <question>` block. Loading/error states are not written.
    fn persist_btw_done(&mut self) {
        if let Some(crate::btw::BtwOverlayState::Done {
            question, answer, ..
        }) = self.btw_overlay()
        {
            self.transcript.push(TranscriptEntry::Btw {
                question,
                answer,
                expanded: false,
            });
            self.bump_transcript();
            self.cap_transcript();
        }
    }

    /// Dismiss the overlay. A Done answer is flushed to scrollback first
    /// (grok-build). Returns whether an overlay was showing.
    pub(crate) fn dismiss_btw_overlay(&mut self) -> bool {
        if !self.show_btw {
            return false;
        }
        self.persist_btw_done();
        self.show_btw = false;
        self.btw_thread.clear();
        self.btw_scroll = 0;
        self.last_btw_area = ratatui::layout::Rect::default();
        self.last_btw_close = ratatui::layout::Rect::default();
        true
    }

    /// Scroll the Done overlay when the pointer is over it. Returns true if
    /// the event was consumed (even when the answer already fits).
    fn scroll_btw_overlay_at(&mut self, x: u16, y: u16, up: bool) -> bool {
        if !crate::btw::cell_in(self.last_btw_area, x, y) {
            return false;
        }
        let Some(state) = self.btw_overlay() else {
            return true;
        };
        let cw = self.last_btw_area.width.saturating_sub(4);
        let max_body = self.last_btw_area.height.saturating_sub(2) as usize;
        let max = state.max_scroll_offset(cw, max_body.max(1));
        if up {
            self.btw_scroll = self.btw_scroll.saturating_sub(1);
        } else {
            self.btw_scroll = (self.btw_scroll + 1).min(max);
        }
        true
    }

    /// Push a user `/btw` question into the overlay immediately (before the
    /// agent drains the inbox) so the panel isn't empty while waiting. A
    /// previous Done overlay is persisted first. Idempotent if the same
    /// question is already the latest.
    pub(crate) fn btw_note_question(&mut self, question: &str) {
        let q = question.trim();
        if q.is_empty() {
            return;
        }
        let already = self
            .btw_thread
            .iter()
            .rev()
            .any(|e| matches!(e, crate::BtwEntry::Question(prev) if prev == q));
        if !already {
            if matches!(self.btw_thread.last(), Some(crate::BtwEntry::Answer(_))) {
                self.persist_btw_done();
            }
            self.btw_thread.clear();
            self.btw_scroll = 0;
            self.btw_thread
                .push(crate::BtwEntry::Question(q.to_string()));
        }
        self.open_btw_pane();
        match self.btw_thread.last() {
            Some(crate::BtwEntry::Thinking(_)) => {}
            _ => {
                self.btw_thread
                    .push(crate::BtwEntry::Thinking("answering…".into()));
            }
        }
    }

    /// Replace the trailing thinking marker (if any) with a fresher status.
    fn btw_set_thinking(&mut self, msg: &str) {
        if let Some(crate::BtwEntry::Thinking(t)) = self.btw_thread.last_mut() {
            *t = msg.to_string();
        } else {
            self.btw_thread
                .push(crate::BtwEntry::Thinking(msg.to_string()));
        }
    }

    fn btw_clear_thinking(&mut self) {
        if matches!(self.btw_thread.last(), Some(crate::BtwEntry::Thinking(_))) {
            self.btw_thread.pop();
        }
    }

    /// Commit the in-progress streamed line, if any.
    pub(crate) fn flush_pending(&mut self) {
        if let Some((style, markdown, text)) = self.pending.take()
            && !text.is_empty()
        {
            if markdown {
                self.commit_md_line(text);
            } else {
                self.transcript
                    .push(TranscriptEntry::Line(Line::styled(text, style)));
                self.bump_transcript();
            }
        }
        // A table may have ended exactly on a newline (no following line to
        // trigger the flush), so always emit any buffered table here.
        self.flush_table();
        self.cap_transcript();
    }

    /// Commit one line of streamed markdown. Consecutive pipe-table rows are held
    /// in `table_buf` and rendered together (aligned) once the table ends; every
    /// other line flushes any pending table, then renders normally.
    fn commit_md_line(&mut self, text: String) {
        if self.code_lang.is_none() && crate::render::is_table_line(&text) {
            self.table_buf.push(text);
            return;
        }
        self.flush_table();
        // Track fenced code blocks so Ctrl-Y can copy the most recent one. A
        // fence-open line (```lang) starts a new block buffer; interior lines
        // accumulate; the closing fence finalizes `last_code_block`.
        let in_fence = self.code_lang.is_some();
        let trimmed = text.trim_start();
        if trimmed.starts_with("```") {
            if self.code_lang.is_none() {
                // Opening a fence: start capturing a fresh block.
                self.last_code_block = Some(String::new());
            } else {
                // Closing the fence: the block is complete — keep it as the
                // last code block. (No-op; accumulation already happened.)
            }
        } else if self.code_lang.is_some() {
            // Interior code line: append to the in-progress block.
            if let Some(block) = self.last_code_block.as_mut() {
                if !block.is_empty() {
                    block.push('\n');
                }
                block.push_str(&text);
            }
        }
        // Some providers stream a short narration after the tool-call event.
        // Fold it into the open explore row immediately, instead of showing a
        // transient assistant line until the next tool call arrives. Only
        // short steering qualifies; substantive final answers remain visible.
        if !in_fence && is_steering_assistant_text(&text) && !text.trim().is_empty() {
            let key = normalize_steering_text(&text);
            if !self.turn_steering_seen.insert(key) {
                return;
            }
            if self.absorb_open_explore_steering(&text) {
                return;
            }
        }
        if !in_fence
            && text.trim().is_empty()
            && last_entry_is_blank(&self.transcript)
            && !self.transcript.is_empty()
        {
            return;
        }
        let _ = markdown_line(&text, &mut self.code_lang);
        self.append_streamed_assistant_line(&text);
    }

    fn append_streamed_assistant_line(&mut self, text: &str) {
        append_assistant_line(&mut self.transcript, text, self.assistant_message_open);
        self.assistant_message_open = true;
        self.bump_transcript();
    }

    fn absorb_open_explore_steering(&mut self, text: &str) -> bool {
        if text.trim().is_empty() || !is_steering_assistant_text(text) {
            return false;
        }
        let Some(group) = self.open_verb_group_mut() else {
            return false;
        };
        if !group
            .steering
            .iter()
            .any(|existing| existing.trim() == text.trim())
        {
            group.steering.push(text.to_string());
            self.bump_transcript();
        }
        true
    }

    /// Emit the accumulated pipe table as aligned rows, clearing the buffer.
    fn flush_table(&mut self) {
        if self.table_buf.is_empty() {
            return;
        }
        let rows = std::mem::take(&mut self.table_buf);
        for row in &rows {
            self.append_streamed_assistant_line(row);
        }
    }

    /// Commit any buffered reasoning as a single collapsible entry, then clear
    /// the buffer. Called when the reasoning phase ends (first text arrives, or
    /// the message ends) so the reasoning isn't flooded inline.
    pub(crate) fn flush_reasoning(&mut self) {
        if self.reasoning_buffer.is_empty() {
            self.reasoning_started = None;
            return;
        }
        let elapsed = self
            .reasoning_started
            .map(|t| t.elapsed())
            .unwrap_or_default();
        let text = std::mem::take(&mut self.reasoning_buffer);
        self.transcript
            .push(TranscriptEntry::Reasoning { text, elapsed });
        self.bump_transcript();
        self.reasoning_started = None;
        self.cap_transcript();
    }

    /// Append streamed text under `style`, committing complete lines. When
    /// `markdown` is set, committed lines are rendered with light markdown
    /// styling (headings, bullets, code fences, inline emphasis).
    pub(crate) fn stream(&mut self, style: Style, markdown: bool, chunk: &str) {
        // A style/kind change ends the current line.
        if let Some((prev, prev_md, _)) = &self.pending
            && (*prev != style || *prev_md != markdown)
        {
            self.flush_pending();
        }
        let (_, _, buf) = self
            .pending
            .get_or_insert_with(|| (style, markdown, String::new()));
        buf.push_str(chunk);
        // Collect the complete lines first, then commit — `commit_md_line` borrows
        // `self`, which can't overlap the `buf` borrow above.
        let mut committed: Vec<String> = Vec::new();
        while let Some(idx) = buf.find('\n') {
            committed.push(buf[..idx].to_string());
            buf.drain(..=idx);
        }
        for line in committed {
            if markdown {
                self.commit_md_line(line);
            } else {
                self.transcript
                    .push(TranscriptEntry::Line(Line::styled(line, style)));
                self.bump_transcript();
            }
        }
        self.cap_transcript();
        // No follow() here: streaming must not yank a reader who scrolled up.
        // While following, the view already tracks the growing bottom.
    }

    pub(crate) fn apply_legacy(&mut self, event: UiEvent) {
        // Bound the debug event log (each arm below pushes one entry). Drop the
        // oldest quarter in a batch when over the cap, so the front-drain is
        // amortized O(1) per event rather than shifting the whole vec each push.
        if self.event_log.len() > MAX_EVENT_LOG {
            let drop_to = MAX_EVENT_LOG * 3 / 4;
            let excess = self.event_log.len() - drop_to;
            self.event_log.drain(..excess);
        }
        match event {
            // Wire evidence is consumed by the structured event tap before
            // `App::apply`; it must never become transcript or debug content.
            UiEvent::ProviderRequest { .. } => {}
            UiEvent::Text { text } => {
                self.event_log
                    .push(format!("assistant_text {} chars", text.len()));
                self.last_turn_event = Some(TurnEventKind::Assistant);
                // If reasoning preceded this text, commit it as a collapsible
                // block before the answer starts.
                self.flush_reasoning();
                self.current_assistant.push_str(&text);
                if !should_buffer_generic_completion_prefix(&self.current_assistant) {
                    let unstreamed =
                        self.current_assistant[self.current_assistant_streamed_bytes..].to_string();
                    self.stream(Style::default(), true, &unstreamed);
                    self.current_assistant_streamed_bytes = self.current_assistant.len();
                }
            }
            UiEvent::BtwQuestion { question } => {
                self.event_log
                    .push(format!("btw_question {} chars", question.len()));
                self.btw_note_question(&question);
            }
            UiEvent::BtwAnswer { text } => {
                self.event_log
                    .push(format!("btw_answer {} chars", text.len()));
                self.last_turn_event = Some(TurnEventKind::Assistant);
                self.flush_reasoning();
                self.open_btw_pane();
                self.btw_clear_thinking();
                match self.btw_thread.last_mut() {
                    Some(crate::BtwEntry::Answer(buf)) => buf.push_str(&text),
                    _ => self.btw_thread.push(crate::BtwEntry::Answer(text.clone())),
                }
            }
            UiEvent::BtwToolStarted { name, arguments } => {
                self.event_log.push(format!("btw_tool_started {name}"));
                self.open_btw_pane();
                self.btw_clear_thinking();
                let detail = btw_tool_detail(&name, &arguments);
                self.btw_thread.push(crate::BtwEntry::Tool {
                    name: name.clone(),
                    detail,
                });
                self.btw_set_thinking(&format!("running {name}…"));
            }
            UiEvent::BtwToolResult { name, result } => {
                self.event_log
                    .push(format!("btw_tool_result {name} {} chars", result.len()));
                self.btw_clear_thinking();
                // Update the last matching tool crumb with a short result peek.
                let peek: String = result
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(48)
                    .collect();
                if let Some(crate::BtwEntry::Tool {
                    name: n, detail, ..
                }) =
                    self.btw_thread.iter_mut().rev().find(
                        |e| matches!(e, crate::BtwEntry::Tool { name: tn, .. } if tn == &name),
                    )
                {
                    if !peek.is_empty() {
                        *detail = peek;
                    }
                    let _ = n;
                }
                self.btw_set_thinking("answering…");
            }
            UiEvent::BtwEnd => {
                self.event_log.push("btw_end".into());
                self.btw_clear_thinking();
            }
            UiEvent::Reasoning { text } => {
                self.event_log
                    .push(format!("reasoning {} chars", text.len()));
                self.last_turn_event = Some(TurnEventKind::Reasoning);
                if self.reasoning_started.is_none() {
                    self.reasoning_started = Some(Instant::now());
                }
                // Grok-build folds thoughts among an open explore burst into
                // that row instead of a standalone thinking block.
                let elapsed = self
                    .reasoning_started
                    .map(|started| started.elapsed())
                    .unwrap_or_default();
                if let Some(group) = self.open_verb_group_mut() {
                    group.thinking.push_str(&text);
                    group.thinking_elapsed = elapsed;
                    self.bump_transcript();
                    return;
                }
                self.reasoning_buffer.push_str(&text);
            }
            UiEvent::AssistantEnd => {
                self.event_log.push("assistant_end".to_string());
                self.last_turn_event = Some(TurnEventKind::AssistantEnd);
                self.turn_rounds = self.turn_rounds.saturating_add(1);
                self.flush_reasoning();
                let generic = generic_completion_guards_enabled()
                    && hi_agent::answer_is_generic_completion_placeholder(&self.current_assistant);
                if !generic && self.current_assistant_streamed_bytes < self.current_assistant.len()
                {
                    let unstreamed =
                        self.current_assistant[self.current_assistant_streamed_bytes..].to_string();
                    self.stream(Style::default(), true, &unstreamed);
                }
                self.flush_pending();
                self.assistant_message_open = false;
                if !generic && !self.current_assistant.trim().is_empty() {
                    self.last_assistant = self.current_assistant.trim().to_string();
                }
                self.current_assistant.clear();
                self.current_assistant_streamed_bytes = 0;
                // Fences don't span messages; reset so a stray ``` can't bleed
                // code styling into the next response.
                self.code_lang = None;
                // Do not freeze explore grouping here. Grok-build keeps
                // consecutive reads/searches in one row across model rounds
                // ("Let me look at…") until a write/run or the turn ends.
            }
            UiEvent::ToolStarted { name, arguments } => {
                let label = tool_label(&name, &arguments);
                self.event_log.push(format!("tool_started {label}"));
                if activity_feed::is_parent_subagent_tool(&name) {
                    return;
                }
                // Track the in-flight tool for interrupt/watchdog, not chrome.
                // Command identity lives on transcript `Run` rows (emitted on
                // ToolCall for shell tools, otherwise with the result).
                self.current_tool = Some(label);
                self.current_tool_started = Some(Instant::now());
                self.run_streamed_this_call = false;
            }
            UiEvent::ToolCall { name, arguments } => {
                let label = tool_label(&name, &arguments);
                self.event_log.push(format!("tool_call {label}"));
                self.last_turn_event = Some(TurnEventKind::ToolCall);
                self.turn_tool_calls = self.turn_tool_calls.saturating_add(1);
                if activity_feed::is_parent_subagent_tool(&name) {
                    self.flush_reasoning();
                    self.flush_pending();
                    return;
                }
                if activity_feed::is_edit_tool(&name) {
                    self.last_turn_had_file_edits = true;
                }
                self.flush_reasoning();
                self.flush_pending();
                // A tool call also closes the preceding assistant response if
                // a provider omitted its explicit AssistantEnd event.
                self.assistant_message_open = false;
                self.current_tool = Some(label.clone());
                self.current_tool_started = Some(Instant::now());
                self.run_streamed_this_call = false;
                if let Some(verb) = ExploreVerb::from_tool(&name) {
                    self.note_explore_call(verb, label_detail(&label));
                } else {
                    self.freeze_verb_group();
                    if activity_feed::is_shell_run_tool(&name) {
                        self.ensure_run_placeholder(&activity_feed::run_command(&name, &label));
                    }
                }
            }
            UiEvent::ToolResult { name, result } => {
                self.event_log
                    .push(format!("tool_result {} chars", result.len()));
                self.last_turn_event = Some(TurnEventKind::ToolResult);
                if activity_feed::is_parent_subagent_tool(&name) {
                    self.current_tool = None;
                    self.current_tool_started = None;
                    return;
                }
                let label = self.current_tool.take().unwrap_or_else(|| name.clone());
                self.current_tool_started = None;
                self.flush_pending();
                self.push_result(&name, &result, &label);
            }
            UiEvent::ToolStream { name, line } => {
                if self.append_live_run_output(&line) {
                    self.run_streamed_this_call = true;
                } else {
                    // No idle Run row yet: fold the streamed line into a Run
                    // row body so live_run_tail_lines is the single source of
                    // truth for running-command output (previously a separate
                    // stream_area block buffering tool_stream_tail).
                    let command = self
                        .current_tool
                        .as_deref()
                        .map(|l| activity_feed::run_command(&name, l))
                        .unwrap_or(name);
                    self.append_run_output_for(&command, &line);
                }
            }
            UiEvent::Status { text } => {
                // Status is not a tool. Humanize it, drop skeptic/btw chatter,
                // leftover ↳ subagent notes (typed Subagent rows own that),
                // and paint the rest as an unguttered dim line so it cannot be
                // mistaken for a Read/Edit/Run row.
                let Some(text) = hi_agent::ui::user_facing_status(&text) else {
                    return;
                };
                if hi_agent::ui::is_live_progress_status(&text) {
                    self.event_log.push(format!("live_status {text}"));
                    self.last_turn_event = Some(TurnEventKind::Status);
                    self.working_status = Some(text);
                    return;
                }
                if is_legacy_subagent_status(&text) {
                    self.event_log
                        .push(format!("status(suppressed subagent) {text}"));
                    return;
                }
                if text.contains("❓ btw")
                    || text.contains("side question")
                    || text.starts_with("btw ·")
                {
                    self.event_log
                        .push(format!("status(suppressed btw) {text}"));
                } else if text.to_ascii_lowercase().contains("skeptic") {
                    self.event_log.push(format!("status {text}"));
                    self.last_turn_event = Some(TurnEventKind::Status);
                } else {
                    self.event_log.push(format!("status {text}"));
                    self.last_turn_event = Some(TurnEventKind::Status);
                    if !self.turn_status_seen.insert(normalize_steering_text(&text)) {
                        return;
                    }
                    self.flush_pending();
                    self.push(Line::styled(text, Style::default().fg(theme().status)));
                }
            }
            UiEvent::TopStatus { text } => {
                let Some(text) = hi_agent::ui::user_facing_status(&text) else {
                    return;
                };
                self.event_log.push(format!("top_status {text}"));
                self.top_notice = Some(text);
            }
            UiEvent::CheckpointWarning { text } => {
                self.event_log.push("checkpoint integrity warning".into());
                // Keep the warning pinned in the top chrome as well as the
                // turn-local composer notice, but do not add a third copy to
                // the transcript.
                self.top_notice = Some(text.clone());
                self.checkpoint_warning = Some(text.clone());
            }
            // Plan updates replace the pinned checklist in place — no transcript
            // line, so progress reads as one updating block rather than a scroll.
            UiEvent::Plan { steps } => {
                self.event_log.push(format!("plan {} steps", steps.len()));
                self.plan = steps;
            }
            // Live counters only — no transcript line; the working/title bars read them.
            UiEvent::Usage {
                prompt,
                generated,
                ctx_used,
                ctx_window,
                estimated,
            } => {
                self.event_log
                    .push(format!("usage {prompt} prompt {generated} generated"));
                self.last_turn_event = Some(TurnEventKind::Usage);
                self.usage = (prompt, generated);
                self.context_used = ctx_used;
                self.context_window = ctx_window;
                self.usage_estimated = estimated;
            }
            UiEvent::SessionUsage { usage } => {
                self.session_totals = usage;
            }
            UiEvent::RateLimits { rate_limits } => {
                self.event_log.push("rate_limits".to_string());
                self.rate_limits = rate_limits;
            }
            UiEvent::TurnEnd { summary } => {
                self.event_log.push(format!("turn_end {summary}"));
                self.last_turn_event = Some(TurnEventKind::TurnEnd);
                self.working_status = None;
                self.flush_pending();
                self.freeze_verb_group();
                // Token/ctx accounting lives in the header chip, not here.
            }
            UiEvent::TurnError {
                error_kind,
                message,
                guidance,
            } => {
                self.event_log
                    .push(format!("turn_error {error_kind} {message}"));
                self.last_turn_event = Some(TurnEventKind::TurnEnd);
                self.working_status = None;
                self.flush_pending();
                self.note_turn_failed(&message, &error_kind, &guidance);
            }
            UiEvent::ChangedFiles { files } => {
                self.event_log
                    .push(format!("changed_files {}", files.len()));
                self.last_changed_files = files;
                self.accumulate_session_files();
                self.changed_files_rect = ratatui::layout::Rect::default();
                self.follow();
            }
            UiEvent::SuggestedPrompt { text } => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    self.suggested_prompt = None;
                } else if self.input.is_empty() && self.queue.is_empty() {
                    // Allow apply while `working` is still true: suggest runs at
                    // the end of `run_turn`, before the TUI clears the working
                    // flag. `set_working(true)` clears any stale ghost text at
                    // the start of the next turn. Skip when the prompt queue is
                    // non-empty — the next turn starts immediately.
                    self.suggested_prompt = Some(trimmed.to_string());
                    self.suggested_prompt_dismissed = false;
                }
            }
            UiEvent::WorkflowUpdated { snapshot } => {
                let terminal = snapshot.status.is_terminal();
                if let Some((revision, tombstone)) = self.workflow_revisions.get(&snapshot.run_id)
                    && (snapshot.revision <= *revision || *tombstone)
                {
                    return;
                }
                self.flush_pending();
                self.event_log.push(format!(
                    "workflow_updated {} {} {:?}",
                    snapshot.run_id, snapshot.revision, snapshot.status
                ));
                self.workflow_revisions
                    .insert(snapshot.run_id.clone(), (snapshot.revision, terminal));
                if snapshot.status.is_completion_reportable()
                    && self
                        .workflow_completion_handoffs
                        .get(&snapshot.run_id)
                        .is_none_or(|revision| *revision < snapshot.revision)
                {
                    self.workflow_completion_handoffs
                        .insert(snapshot.run_id.clone(), snapshot.revision);
                    let summary = snapshot
                        .result_summary
                        .as_deref()
                        .or(snapshot.pause_message.as_deref())
                        .unwrap_or("no result summary was provided");
                    let _ = self.enqueue_prompt(format!(
                        "Review workflow '{}' ({}) after status {:?}. Summarize its result for the user and recommend the next action. Result: {}",
                        snapshot.workflow_name, snapshot.run_id, snapshot.status, summary
                    ));
                }
                if let Some(entry) = self.transcript.iter_mut().find(|entry| {
                    matches!(
                        entry,
                        TranscriptEntry::Workflow { snapshot: existing }
                            if existing.run_id == snapshot.run_id
                    )
                }) {
                    *entry = TranscriptEntry::Workflow { snapshot };
                } else {
                    self.transcript.push(TranscriptEntry::Workflow { snapshot });
                }
                self.bump_transcript();
                self.cap_transcript();
            }
            UiEvent::DiffRunUpdated { snapshot } => {
                if let Some(overlay) = self.diff_lab.as_mut()
                    && overlay.snapshot.run_id == snapshot.run_id
                {
                    overlay.snapshot = snapshot;
                }
            }
            UiEvent::SubagentSpawned {
                id,
                subagent_kind,
                description,
                background,
            } => {
                self.apply_subagent_spawned(id, subagent_kind, description, background);
            }
            UiEvent::SubagentProgress { id, activity, line } => {
                self.apply_subagent_progress(id, activity, line);
            }
            UiEvent::SubagentFinished {
                id,
                status,
                elapsed_ms,
                summary,
            } => {
                self.apply_subagent_finished(id, status, elapsed_ms, summary);
            }
        }
    }

    fn apply_subagent_spawned(
        &mut self,
        id: String,
        kind: String,
        description: String,
        background: bool,
    ) {
        self.event_log.push(format!("subagent_spawned {kind} {id}"));
        let started_at = Instant::now();
        self.subagents.insert(
            id.clone(),
            crate::subagent_overlay::SubagentInfo {
                id: id.clone(),
                kind: kind.clone(),
                description: description.clone(),
                background,
                activity: if background {
                    String::new()
                } else {
                    "running".into()
                },
                started_at,
                finished: None,
                summary: String::new(),
                lines: Vec::new(),
            },
        );
        self.freeze_verb_group();
        self.flush_pending();
        self.push_activity(ActivityKind::Subagent {
            id,
            kind,
            description,
            background,
            activity: if background {
                String::new()
            } else {
                "running".into()
            },
            status: None,
            started_at,
            elapsed_ms: 0,
        });
    }

    fn apply_subagent_progress(&mut self, id: String, activity: String, line: Option<String>) {
        if let Some(info) = self.subagents.get_mut(&id) {
            if !activity.is_empty() {
                info.activity = activity.clone();
            }
            if let Some(line) = line {
                let line = line.trim();
                if !line.is_empty() && info.lines.len() < 200 {
                    info.lines.push(line.to_string());
                }
            }
        }
        if activity.is_empty() {
            return;
        }
        let background = self.subagents.get(&id).is_some_and(|info| info.background);
        if background {
            return;
        }
        if let Some(block) = self.subagent_block_mut(&id)
            && let ActivityKind::Subagent {
                activity: current,
                status,
                ..
            } = &mut block.kind
            && status.is_none()
        {
            *current = activity;
            self.bump_transcript();
        }
    }

    fn apply_subagent_finished(
        &mut self,
        id: String,
        status: String,
        elapsed_ms: u64,
        summary: String,
    ) {
        self.event_log
            .push(format!("subagent_finished {id} {status}"));
        let background = self.subagents.get(&id).is_some_and(|info| info.background);
        let (kind, description, started_at) = self
            .subagents
            .get(&id)
            .map(|info| (info.kind.clone(), info.description.clone(), info.started_at))
            .unwrap_or_else(|| ("task".into(), id.clone(), Instant::now()));
        if let Some(info) = self.subagents.get_mut(&id) {
            info.finished = Some((status.clone(), elapsed_ms));
            info.summary = summary.clone();
            if !summary.is_empty() && info.lines.len() < 200 {
                info.lines.push(summary.clone());
            }
        }
        if background {
            self.freeze_verb_group();
            self.flush_pending();
            self.push_activity(ActivityKind::Subagent {
                id,
                kind,
                description,
                background: true,
                activity: String::new(),
                status: Some(status),
                started_at,
                elapsed_ms,
            });
            return;
        }
        if let Some(block) = self.subagent_block_mut(&id)
            && let ActivityKind::Subagent {
                status: row_status,
                elapsed_ms: row_elapsed,
                activity,
                ..
            } = &mut block.kind
        {
            *row_status = Some(status);
            *row_elapsed = elapsed_ms;
            *activity = String::new();
            self.bump_transcript();
        }
    }

    fn subagent_block_mut(&mut self, id: &str) -> Option<&mut ActivityBlock> {
        self.transcript
            .iter_mut()
            .rev()
            .find_map(|entry| match entry {
                TranscriptEntry::Activity(block) if block.subagent_id() == Some(id) => Some(block),
                _ => None,
            })
    }

    /// Render a tool result as a typed activity row.
    pub(crate) fn push_result(&mut self, name: &str, result: &str, label: &str) {
        if activity_feed::is_parent_subagent_tool(name) {
            return;
        }
        let display_result = hi_agent::ui::user_visible_tool_result(result);
        if let Some(verb) = ExploreVerb::from_tool(name) {
            self.note_explore_result(verb, label_detail(label), &display_result);
            return;
        }
        if name == "bash_output" {
            let id = label
                .strip_prefix("bash_output ")
                .unwrap_or(label)
                .to_string();
            if is_missing_background_process_result(result) {
                self.freeze_verb_group();
                self.pop_idle_run(&id);
                self.push_activity(ActivityKind::Other {
                    verb: "background".into(),
                    detail: format!("process {id} unavailable"),
                    body: String::new(),
                });
                return;
            }
            if bash_output_is_idle(result) {
                self.note_idle_bash_poll(&id);
                return;
            }
            self.apply_run_result(&id, &display_result, bash_process_live(result));
            return;
        }
        self.freeze_verb_group();
        if matches!(name, "memory_update" | "memory_forget") {
            let line = display_result
                .lines()
                .next()
                .unwrap_or(display_result.as_str())
                .to_string();
            self.push_activity(ActivityKind::Other {
                verb: "memory".into(),
                detail: line,
                body: String::new(),
            });
            return;
        }
        if activity_feed::is_edit_tool(name) {
            let path = label_detail(label).unwrap_or_else(|| name.to_string());
            let (additions, deletions) = activity_feed::parse_diff_stats(&display_result);
            let diff = if display_result.contains('\u{1b}')
                || crate::render::looks_like_diff(&display_result)
                || additions > 0
                || deletions > 0
            {
                display_result
            } else {
                String::new()
            };
            if self.try_coalesce_edit(&path, additions, deletions, &diff) {
                return;
            }
            self.push_activity(ActivityKind::Edit {
                path,
                additions,
                deletions,
                diff,
            });
            return;
        }
        if activity_feed::is_run_tool(name) {
            let command = activity_feed::run_command(name, label);
            self.apply_run_result(&command, &display_result, bash_process_live(result));
            return;
        }
        let (verb, detail) = match label.split_once(' ') {
            Some((v, rest)) => (v.to_string(), rest.to_string()),
            None => (name.to_string(), String::new()),
        };
        self.push_activity(ActivityKind::Other {
            verb,
            detail,
            body: display_result,
        });
    }

    fn push_activity(&mut self, kind: ActivityKind) {
        self.flush_table();
        self.transcript
            .push(TranscriptEntry::Activity(ActivityBlock {
                kind,
                expanded: false,
            }));
        self.bump_transcript();
        self.cap_transcript();
    }

    fn try_coalesce_edit(
        &mut self,
        path: &str,
        additions: u32,
        deletions: u32,
        diff: &str,
    ) -> bool {
        let Some(TranscriptEntry::Activity(block)) = self.transcript.last_mut() else {
            return false;
        };
        let ActivityKind::Edit {
            path: existing,
            additions: add,
            deletions: del,
            diff: existing_diff,
        } = &mut block.kind
        else {
            return false;
        };
        if existing != path {
            return false;
        }
        *add = add.saturating_add(additions);
        *del = del.saturating_add(deletions);
        if !diff.is_empty() {
            if !existing_diff.is_empty() {
                existing_diff.push('\n');
            }
            existing_diff.push_str(diff);
        }
        self.bump_transcript();
        true
    }

    pub(crate) fn freeze_verb_group(&mut self) {
        if let Some(group) = self.open_verb_group_mut() {
            group.live = false;
            group.open = false;
            self.bump_transcript();
        }
    }

    /// Grok-build keeps consecutive reads/searches in one row even when the
    /// model narrates between them. Skip assistant/reasoning/status lines and
    /// resume the still-open explore group from this turn. A substantial
    /// assistant answer (heading, list, long prose) starts a new burst.
    fn open_verb_group_mut(&mut self) -> Option<&mut crate::activity_feed::VerbGroup> {
        for entry in self.transcript.iter_mut().rev() {
            match entry {
                TranscriptEntry::Activity(block) => {
                    return block.as_verb_group_mut().filter(|group| group.open);
                }
                TranscriptEntry::Assistant(line) if is_steering_assistant_line(line) => continue,
                TranscriptEntry::Assistant(_) => return None,
                TranscriptEntry::AssistantMessage { text } if is_steering_assistant_text(text) => {
                    continue;
                }
                TranscriptEntry::AssistantMessage { .. } => return None,
                TranscriptEntry::Reasoning { .. } | TranscriptEntry::Line(_) => continue,
                TranscriptEntry::UserPrompt { .. }
                | TranscriptEntry::Btw { .. }
                | TranscriptEntry::Workflow { .. }
                | TranscriptEntry::ToolOutput { .. } => return None,
            }
        }
        None
    }

    fn note_explore_call(&mut self, verb: ExploreVerb, detail: Option<String>) {
        let chrome = self.take_explore_chrome();
        if let Some(group) = self.open_verb_group_mut() {
            absorb_explore_chrome(group, chrome);
            group.add(verb, detail);
            self.bump_transcript();
            return;
        }
        self.freeze_verb_group();
        self.flush_table();
        let mut block = ActivityBlock::verb_group(verb, detail);
        if let Some(group) = block.as_verb_group_mut() {
            absorb_explore_chrome(group, chrome);
        }
        self.transcript.push(TranscriptEntry::Activity(block));
        self.bump_transcript();
        self.cap_transcript();
    }

    /// Steal trailing CoT and short steering lines so they live inside the
    /// explore row instead of sitting above it.
    fn take_explore_chrome(&mut self) -> ExploreChrome {
        let mut steal: Vec<usize> = Vec::new();
        for i in (0..self.transcript.len()).rev() {
            match &self.transcript[i] {
                TranscriptEntry::Reasoning { .. } => steal.push(i),
                TranscriptEntry::Line(_) => continue,
                TranscriptEntry::Assistant(line) if is_steering_assistant_line(line) => {
                    steal.push(i);
                }
                TranscriptEntry::AssistantMessage { text } if is_steering_assistant_text(text) => {
                    steal.push(i);
                }
                TranscriptEntry::Assistant(_) | TranscriptEntry::AssistantMessage { .. } => {
                    if let Some(&reason_at) = steal
                        .iter()
                        .find(|&&j| matches!(self.transcript[j], TranscriptEntry::Reasoning { .. }))
                    {
                        steal.retain(|&j| j >= reason_at);
                    } else {
                        steal.clear();
                    }
                    break;
                }
                TranscriptEntry::Activity(_)
                | TranscriptEntry::UserPrompt { .. }
                | TranscriptEntry::Btw { .. }
                | TranscriptEntry::Workflow { .. }
                | TranscriptEntry::ToolOutput { .. } => break,
            }
        }
        steal.sort_unstable();
        let mut chrome = ExploreChrome::default();
        for i in steal.into_iter().rev() {
            match self.transcript.remove(i) {
                TranscriptEntry::Reasoning { text, elapsed } => {
                    if chrome.thinking.is_empty() {
                        chrome.thinking = text;
                    } else {
                        chrome.thinking = format!("{text}\n{}", chrome.thinking);
                    }
                    chrome.thinking_elapsed = chrome.thinking_elapsed.saturating_add(elapsed);
                }
                TranscriptEntry::Assistant(line) => {
                    let text = crate::render::line_text(&line);
                    if !text.trim().is_empty() {
                        chrome.steering.insert(0, text);
                    }
                }
                TranscriptEntry::AssistantMessage { text } => {
                    for line in text.lines().rev() {
                        if !line.trim().is_empty() {
                            chrome.steering.insert(0, line.to_string());
                        }
                    }
                }
                _ => {}
            }
        }
        chrome
    }

    fn note_explore_result(&mut self, verb: ExploreVerb, detail: Option<String>, result: &str) {
        let n = if result.trim().is_empty() {
            0
        } else {
            result.lines().count() as u32
        };
        if let Some(group) = self.open_verb_group_mut() {
            group.lines = group.lines.saturating_add(n);
            if n > 0 {
                group.all_empty = false;
            } else if group.total() == 1 {
                group.all_empty = true;
            }
            group.live = false;
            self.bump_transcript();
            return;
        }
        // Replay/result without a preceding ToolCall: start a finished row.
        let mut block = ActivityBlock::verb_group(verb, detail);
        if let Some(group) = block.as_verb_group_mut() {
            group.lines = n;
            group.all_empty = n == 0;
            group.live = false;
            group.open = false;
        }
        self.flush_table();
        self.transcript.push(TranscriptEntry::Activity(block));
        self.bump_transcript();
        self.cap_transcript();
    }
}

const fn generic_completion_guards_enabled() -> bool {
    !cfg!(feature = "smoke-negative-control-disable-generic-completion-guards")
}

fn should_buffer_generic_completion_prefix(content: &str) -> bool {
    generic_completion_guards_enabled() && could_be_generic_completion_prefix(content)
}

fn could_be_generic_completion_prefix(content: &str) -> bool {
    let normalized = content
        .to_ascii_lowercase()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    [
        "completed the requested action",
        "the requested action is complete",
        "the requested action has been completed",
        "the requested task is complete",
        "the requested task has been completed",
    ]
    .iter()
    .any(|candidate| candidate.starts_with(&normalized))
}

/// Compact tool-arg detail for the BTW pane timeline (path/pattern/command).
fn btw_tool_detail(name: &str, arguments: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();
    let pick = |keys: &[&str]| {
        keys.iter()
            .find_map(|k| v.get(*k).and_then(|x| x.as_str()))
            .unwrap_or("")
            .to_string()
    };
    match name {
        "read" | "list" | "glob" | "diff" => pick(&["path", "target", "directory"]),
        "grep" => {
            let pat = pick(&["pattern", "query"]);
            let path = pick(&["path", "glob"]);
            if path.is_empty() {
                pat
            } else {
                format!("{pat} in {path}")
            }
        }
        "repo_map" | "find_symbol" => pick(&["task", "symbol", "query", "name"]),
        "web_search" | "web_fetch" => pick(&["query", "url"]),
        _ => pick(&["path", "command", "query", "task"]),
    }
}

fn is_legacy_subagent_status(text: &str) -> bool {
    let text = text.trim();
    text.starts_with('↳')
        && (text.contains("subagent")
            || text.starts_with("↳ explore:")
            || text.starts_with("↳ delegate:")
            || text.starts_with("↳ task:")
            || text.starts_with("↳ plan:")
            || text.starts_with("↳ general-purpose:"))
}

#[cfg(test)]
mod generic_completion_negative_control_tests {
    use super::*;

    #[test]
    fn feature_controls_only_the_transcript_buffer_guard() {
        let placeholder = "Completed the requested action.";
        assert!(could_be_generic_completion_prefix(placeholder));
        assert_eq!(
            should_buffer_generic_completion_prefix(placeholder),
            generic_completion_guards_enabled()
        );
    }
}
