//! `App` methods: transcript.

use std::time::{Duration, Instant};

use ansi_to_tui::IntoText;
use hi_agent::ui::tool_label;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Text};

use crate::event::UiEvent;
use crate::render::{diff_lines, dim, looks_like_diff, markdown_line};
use crate::util::fmt_rate_limits;
use crate::{
    ExploreRun, MAX_EVENT_LOG, MAX_TRANSCRIPT_LINES, TranscriptEntry, TurnEventKind, TurnState,
};

impl crate::App {
    pub(crate) fn push(&mut self, line: Line<'static>) {
        self.transcript.push(TranscriptEntry::Line(line));
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
        }
    }

    pub(crate) fn note_turn_completed_without_summary(&mut self) {
        match self.last_turn_event {
            Some(TurnEventKind::ToolCall | TurnEventKind::ToolResult) => {
                self.status = "stopped after tool output".to_string();
                self.last_turn_state = TurnState::Warning("stopped after tool output".to_string());
                self.last_error = Some("turn stopped after tool output".to_string());
                self.push(Line::styled(
                    "⚠ turn stopped after tool output without a final assistant response; try /retry",
                    Style::default().fg(Color::Yellow),
                ));
                self.record_model_issue();
            }
            _ => {
                self.status = "done · no usage reported".to_string();
                self.last_turn_state = TurnState::Done("no usage reported".to_string());
                self.push(Line::styled("✓ done · no usage reported", dim()));
            }
        }
        self.follow();
    }

    pub(crate) fn note_turn_failed(&mut self, error: &str, kind: &str, guidance: &str) {
        self.status = format!("failed · {kind}").to_string();
        self.last_turn_state = TurnState::Failed(error.to_string());
        self.last_error = Some(error.to_string());
        let guidance_line = if guidance.is_empty() {
            String::new()
        } else {
            format!("\n  💡 {guidance}")
        };
        let limits = fmt_rate_limits(self.rate_limits)
            .map(|limits| format!("\n  {limits}"))
            .unwrap_or_default();
        self.push(Line::styled(
            format!("✗ failed · {kind}: {error}{guidance_line}{limits}"),
            Style::default().fg(Color::Red),
        ));
        self.follow();
    }

    pub(crate) fn note_backend_waiting(&mut self, idle: Duration, threshold: Duration) {
        let _ = (idle, threshold);
        self.push(Line::styled(
            "⚠ Still thinking. Ctrl-C cancels; keep waiting to continue.",
            Style::default().fg(Color::Yellow),
        ));
        self.follow();
    }

    /// Re-pin the view to the latest output. Called on explicit user actions (a
    /// new turn, a command's output) — not on streaming appends, so a reader who
    /// scrolled up stays put.
    pub(crate) fn follow(&mut self) {
        self.following = true;
    }

    pub(crate) fn transcript_text(&self) -> String {
        self.transcript
            .iter()
            .map(TranscriptEntry::text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub(crate) fn scroll_up(&mut self, n: u16) {
        self.scroll_by(-(n as i32));
    }

    pub(crate) fn scroll_down(&mut self, n: u16) {
        self.scroll_by(n as i32);
    }

    pub(crate) fn handle_mouse(&mut self, kind: crossterm::event::MouseEventKind) {
        match kind {
            crossterm::event::MouseEventKind::ScrollUp => {
                if let Some(picker) = self.picker.as_mut() {
                    picker.up();
                } else if self.completion.is_some() {
                    self.completion_move(-1);
                } else {
                    self.scroll_up(3);
                }
            }
            crossterm::event::MouseEventKind::ScrollDown => {
                if let Some(picker) = self.picker.as_mut() {
                    picker.down();
                } else if self.completion.is_some() {
                    self.completion_move(1);
                } else {
                    self.scroll_down(3);
                }
            }
            _ => {}
        }
    }

    /// Move the viewport by `delta` wrapped lines (negative = toward older
    /// output). Re-pins to the bottom when scrolled all the way down; snapshots
    /// the line count when first leaving the bottom (for the "↓ N new" hint).
    /// Uses the metrics cached by the last render.
    pub(crate) fn scroll_by(&mut self, delta: i32) {
        let max = self.view_max_scroll as i32;
        let cur = if self.following {
            max
        } else {
            (self.scroll as i32).min(max)
        };
        let next = (cur + delta).clamp(0, max);
        if next >= max {
            self.following = true;
        } else {
            if self.following {
                self.total_when_unpinned = self.view_total;
            }
            self.following = false;
            self.scroll = next as u16;
        }
    }

    /// Commit the in-progress streamed line, if any.
    pub(crate) fn flush_pending(&mut self) {
        if let Some((style, markdown, text)) = self.pending.take() {
            let line = if markdown {
                markdown_line(&text, &mut self.code_lang)
            } else {
                Line::styled(text, style)
            };
            self.transcript.push(TranscriptEntry::Line(line));
            self.cap_transcript();
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
        while let Some(idx) = buf.find('\n') {
            let committed: String = buf[..idx].to_string();
            buf.drain(..=idx);
            let line = if markdown {
                markdown_line(&committed, &mut self.code_lang)
            } else {
                Line::styled(committed, style)
            };
            self.transcript.push(TranscriptEntry::Line(line));
        }
        self.cap_transcript();
        // No follow() here: streaming must not yank a reader who scrolled up.
        // While following, the view already tracks the growing bottom.
    }

    pub(crate) fn apply(&mut self, event: UiEvent) {
        // Bound the debug event log (each arm below pushes one entry). Drop the
        // oldest quarter in a batch when over the cap, so the front-drain is
        // amortized O(1) per event rather than shifting the whole vec each push.
        if self.event_log.len() > MAX_EVENT_LOG {
            let drop_to = MAX_EVENT_LOG * 3 / 4;
            let excess = self.event_log.len() - drop_to;
            self.event_log.drain(..excess);
        }
        match event {
            UiEvent::Text { text } => {
                self.event_log
                    .push(format!("assistant_text {} chars", text.len()));
                self.last_turn_event = Some(TurnEventKind::Assistant);
                // If reasoning preceded this text, commit it as a collapsible
                // block before the answer starts.
                self.flush_reasoning();
                self.current_assistant.push_str(&text);
                self.stream(Style::default(), true, &text);
            }
            UiEvent::Reasoning { text } => {
                self.event_log
                    .push(format!("reasoning {} chars", text.len()));
                self.last_turn_event = Some(TurnEventKind::Reasoning);
                // Buffer reasoning instead of streaming it inline — it's
                // committed as a single collapsible "thought for Ns" entry when
                // the reasoning phase ends (first text or assistant_end).
                if self.reasoning_started.is_none() {
                    self.reasoning_started = Some(Instant::now());
                }
                self.reasoning_buffer.push_str(&text);
            }
            UiEvent::AssistantEnd => {
                self.event_log.push("assistant_end".to_string());
                self.last_turn_event = Some(TurnEventKind::AssistantEnd);
                self.turn_rounds = self.turn_rounds.saturating_add(1);
                self.flush_reasoning();
                self.flush_pending();
                if !self.current_assistant.trim().is_empty() {
                    self.last_assistant = self.current_assistant.trim().to_string();
                }
                self.current_assistant.clear();
                // Fences don't span messages; reset so a stray ``` can't bleed
                // code styling into the next response.
                self.code_lang = None;
            }
            UiEvent::ToolStarted { name, arguments } => {
                let label = tool_label(&name, &arguments);
                self.event_log.push(format!("tool_started {label}"));
                // Mark this tool as the active party so the working line can
                // name it with its own timer until the result lands. No
                // transcript line — the header is emitted with the result.
                self.current_tool = Some(label);
                self.current_tool_started = Some(Instant::now());
                // Clear any previous stream tail when a new tool starts.
                self.tool_stream_tail.clear();
            }
            UiEvent::ToolCall { name, arguments } => {
                let label = tool_label(&name, &arguments);
                self.event_log.push(format!("tool_call {label}"));
                self.last_turn_event = Some(TurnEventKind::ToolCall);
                self.turn_tool_calls = self.turn_tool_calls.saturating_add(1);
                if matches!(name.as_str(), "write" | "edit") {
                    self.last_turn_had_file_edits = true;
                }
                self.flush_reasoning();
                self.flush_pending();
                // Exploration tools (read/list/grep) defer their header until the
                // result lands, so the file name and line count share one line
                // instead of printing a header followed by a bare "N lines".
                if matches!(name.as_str(), "read" | "list" | "grep") {
                    self.pending_explore_label = Some(label);
                } else {
                    // A non-explore tool breaks any active explore run.
                    self.explore_run = None;
                    self.push(Line::styled(
                        format!("⏺ {label}"),
                        Style::default().fg(Color::Cyan),
                    ));
                }
            }
            UiEvent::ToolResult { name, result } => {
                self.event_log
                    .push(format!("tool_result {} chars", result.len()));
                self.last_turn_event = Some(TurnEventKind::ToolResult);
                // The tool finished — back to the model being the active party.
                self.current_tool = None;
                self.current_tool_started = None;
                self.tool_stream_tail.clear();
                self.flush_pending();
                self.push_result(&name, &result);
            }
            UiEvent::ToolStream { line, .. } => {
                // Accumulate streamed lines for the live working-area display.
                // Keep only the last few so the panel stays compact.
                self.tool_stream_tail.push(line.to_string());
                if self.tool_stream_tail.len() > 4 {
                    self.tool_stream_tail.remove(0);
                }
            }
            UiEvent::Status { text } => {
                self.event_log.push(format!("status {text}"));
                self.last_turn_event = Some(TurnEventKind::Status);
                self.flush_pending();
                self.push(Line::styled(text, Style::default().fg(Color::Blue)));
            }
            UiEvent::CheckpointWarning { text } => {
                self.event_log.push("checkpoint unavailable".into());
                self.checkpoint_warning = Some(text.clone());
                self.flush_pending();
                self.push(Line::styled(text, Style::default().fg(Color::Yellow)));
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
            UiEvent::RateLimits { rate_limits } => {
                self.event_log.push("rate_limits".to_string());
                self.rate_limits = rate_limits;
            }
            UiEvent::TurnEnd { summary } => {
                self.event_log.push(format!("turn_end {summary}"));
                self.last_turn_event = Some(TurnEventKind::TurnEnd);
                self.flush_pending();
                // Keep detailed usage in exactly one historical location. The
                // persistent status remains token-free for subsequent turns.
                let summary = summary.trim_matches(['[', ']']);
                if summary.contains("stalled") {
                    self.status = "incomplete · stalled".to_string();
                    self.last_turn_state = TurnState::Warning("incomplete".to_string());
                    self.last_error = Some("turn ended incomplete".to_string());
                    self.push(Line::styled(
                        format!("⚠ incomplete · {summary}"),
                        Style::default().fg(Color::Yellow),
                    ));
                    self.record_model_issue();
                } else {
                    self.status = "done".to_string();
                    self.last_turn_state = TurnState::Done(self.status.clone());
                    self.push(Line::styled(format!("✓ done · {summary}"), dim()));
                }
                // No follow(): respect a reader who scrolled up — the "↓ N new"
                // hint tells them the summary landed below.
            }
            UiEvent::TurnError {
                error_kind,
                message,
                guidance,
            } => {
                self.event_log
                    .push(format!("turn_error {error_kind} {message}"));
                self.last_turn_event = Some(TurnEventKind::TurnEnd);
                self.flush_pending();
                self.note_turn_failed(&message, &error_kind, &guidance);
            }
            UiEvent::ChangedFiles { files } => {
                self.event_log
                    .push(format!("changed_files {}", files.len()));
                self.flush_pending();
                let label = if files.len() == 1 { "file" } else { "files" };
                let list = files.join(", ");
                let clipped = hi_agent::ui::clip(&list, 200);
                self.push(Line::styled(
                    format!("✎ {} {} changed: {}", files.len(), label, clipped),
                    Style::default().fg(Color::Green),
                ));
                self.follow();
            }
        }
    }

    /// Render a tool result, clipped to a handful of lines and indented.
    /// Preserves any ANSI colors (e.g. edit/write diffs); for *plain* unified
    /// diff output from a shell command (`git diff`, `diff -u`) — which CLIs
    /// emit without color when piped — adds diff coloring so it's readable.
    ///
    /// Read-only exploration tools (`read`/`list`/`grep`) already named the
    /// file or pattern in their `tool_call` header line — dumping their full
    /// output into the transcript is noise during a codebase review. Show a
    /// compact line count instead, so the transcript reads as a list of files
    /// consulted rather than a wall of their contents.
    pub(crate) fn push_result(&mut self, name: &str, result: &str) {
        if matches!(name, "read" | "list" | "grep") {
            let n = result.lines().count() as u32;
            // Collapse the header and the line count into one transcript line:
            // `⏺ read path/to/file · 113 lines`. Falls back to the bare header
            // if we never saw the ToolCall (e.g. replay from a transcript).
            let label = self.pending_explore_label.take();
            let header = match &label {
                Some(l) => l.clone(),
                None => name.to_string(),
            };
            // Merge consecutive same-tool explore results into one line, so a
            // burst of reads renders as `⏺ read 6 files · 743 lines` instead of
            // six separate lines. A run continues only while the tool name is
            // the same AND the run's summary line is still the last transcript
            // entry — events that commit lines without resetting the run
            // (assistant text, status) would otherwise get overwritten by the
            // in-place update below.
            let last_pos = (self.trimmed + self.transcript.len() as u64).checked_sub(1);
            let merge = self
                .explore_run
                .as_ref()
                .is_some_and(|r| r.tool == name && Some(r.line_pos) == last_pos);
            if merge {
                let run = self.explore_run.as_mut().unwrap();
                run.count += 1;
                run.lines += n;
                if n > 0 {
                    run.all_empty = false;
                }
                let line = self.render_explore_run(&header);
                self.replace_last_line(line);
                return;
            }
            // Start a new run; its summary line is about to be pushed at the
            // current end of the transcript.
            self.explore_run = Some(ExploreRun {
                tool: name.to_string(),
                count: 1,
                lines: n,
                all_empty: n == 0,
                line_pos: self.trimmed + self.transcript.len() as u64,
            });
            let line = self.render_explore_run(&header);
            self.push(Line::styled(line, Style::default().fg(Color::Cyan)));
            return;
        }
        // A non-explore result breaks any active explore run.
        self.explore_run = None;
        // Enough to show a small edit's diff with its context inline; larger
        // results truncate with a footer (use `/diff` for the full diff).
        const MAX: usize = 16;
        if result.trim().is_empty() {
            self.push(Line::styled("  (no output)", dim()));
            return;
        }
        let body: String = result.lines().take(MAX).collect::<Vec<_>>().join("\n");
        let lines: Vec<Line<'static>> = if !body.contains('\u{1b}') && looks_like_diff(result) {
            diff_lines(&body)
        } else {
            // ANSI (already-colored) or non-diff text: parse escapes as before.
            body.into_text()
                .unwrap_or_else(|_| Text::from(body.clone()))
                .lines
        };
        for mut line in lines {
            line.spans.insert(0, "  ".into());
            self.transcript.push(TranscriptEntry::Line(line));
        }
        let extra = result.lines().count().saturating_sub(MAX);
        if extra > 0 {
            self.push(Line::styled(format!("  … {extra} more lines"), dim()));
        }
    }

    /// Render the current explore run as a single transcript line. A run of one
    /// shows the per-call label and line count (`⏺ read src/a.rs · 113 lines`);
    /// a run of many collapses to a summary (`⏺ read 6 files · 743 lines`).
    fn render_explore_run(&self, header: &str) -> String {
        let run = match &self.explore_run {
            Some(r) => r,
            None => return format!("⏺ {header}"),
        };
        if run.count <= 1 {
            if run.all_empty {
                format!("⏺ {header} · (no output)")
            } else {
                let s = if run.lines == 1 { "" } else { "s" };
                format!("⏺ {header} · {} line{}", run.lines, s)
            }
        } else {
            // Multi-call summary: drop the per-file label, show counts.
            let noun = match run.tool.as_str() {
                "read" => "files",
                _ => "calls",
            };
            if run.all_empty {
                format!("⏺ {} {} {} · (no output)", run.tool, run.count, noun)
            } else {
                let s = if run.lines == 1 { "" } else { "s" };
                format!(
                    "⏺ {} {} {} · {} line{}",
                    run.tool, run.count, noun, run.lines, s
                )
            }
        }
    }

    /// Replace the last transcript line in place (used to update a merged
    /// explore-run line as more results fold in). No-op if the transcript is
    /// empty or the last entry isn't a plain line.
    fn replace_last_line(&mut self, text: String) {
        if let Some(TranscriptEntry::Line(line)) = self.transcript.last_mut() {
            *line = Line::styled(text, Style::default().fg(Color::Cyan));
        }
    }
}
