//! `App` methods: transcript.

use std::time::{Duration, Instant};

use ansi_to_tui::IntoText;
use hi_agent::ui::tool_label;
use hi_agent::{ReviewStatus, TurnOutcome, TurnStatus, TurnStopReason, VerificationStatus};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};

use crate::event::UiEvent;
use crate::render::{accent_line, diff_lines, dim, gutter, looks_like_diff, markdown_line};
use crate::theme::theme;
use crate::util::fmt_rate_limits;
use crate::{
    ExploreRun, MAX_EVENT_LOG, MAX_TRANSCRIPT_LINES, TranscriptEntry, TurnEventKind, TurnState,
};

/// Build a tool-call header line: `┃ ◆ verb rest` — the accent gutter and `◆`
/// bullet in the tool color, the leading verb bold, the rest in secondary text.
/// This is the block signature that marks agent machinery at a glance.
fn tool_header(label: &str) -> Line<'static> {
    let t = theme();
    let (verb, rest) = match label.split_once(' ') {
        Some((v, r)) => (v, r),
        None => (label, ""),
    };
    let mut spans = vec![
        gutter(t.accent_tool),
        Span::styled("◆ ", Style::default().fg(t.accent_tool)),
        Span::styled(
            verb.to_string(),
            Style::default()
                .fg(t.text_secondary)
                .add_modifier(Modifier::BOLD),
        ),
    ];
    if !rest.is_empty() {
        spans.push(Span::styled(
            format!(" {rest}"),
            Style::default().fg(t.text_secondary),
        ));
    }
    Line::from(spans)
}

impl crate::App {
    pub(crate) fn push(&mut self, line: Line<'static>) {
        // Anything pushed directly ends a streaming table, so emit it first and
        // keep the ordering correct.
        self.flush_table();
        self.transcript.push(TranscriptEntry::Line(line));
        self.cap_transcript();
    }

    /// Push a user-prompt echo as a structurally-distinct entry so the render
    /// pass can pin it as a sticky header when scrolled past.
    pub(crate) fn push_user_prompt(&mut self, line: Line<'static>) {
        self.transcript.push(TranscriptEntry::UserPrompt(line));
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

    /// Apply the authoritative typed result returned by `Agent::run_turn`.
    ///
    /// `Ui::turn_end` carries token accounting only and can arrive before final
    /// workspace reconciliation. It must therefore never decide whether a turn
    /// succeeded. This is the sole success-state transition for a normal turn.
    pub(crate) fn note_turn_outcome(&mut self, outcome: &TurnOutcome) {
        let detail = outcome_detail(outcome);
        match outcome_state(outcome) {
            OutcomeState::Done => {
                self.status = format!("done · {detail}");
                self.last_turn_state = TurnState::Done(detail.clone());
                self.last_error = None;
                self.push(accent_line(
                    theme().accent_success,
                    format!("✓ done · {detail}"),
                    dim(),
                ));
            }
            OutcomeState::Warning => {
                let label = match outcome.status {
                    TurnStatus::Blocked => format!("blocked · {detail}"),
                    TurnStatus::Incomplete => format!("incomplete · {detail}"),
                    _ => detail,
                };
                self.status = format!("warning · {label}");
                self.last_turn_state = TurnState::Warning(label.clone());
                self.last_error = Some(label.clone());
                self.push(accent_line(
                    theme().warning,
                    format!("⚠ {label}"),
                    Style::default().fg(theme().warning),
                ));
            }
            OutcomeState::Failed => {
                self.status = format!("failed · {detail}");
                self.last_turn_state = TurnState::Failed(detail.clone());
                self.last_error = Some(detail.clone());
                self.push(accent_line(
                    theme().accent_error,
                    format!("✗ failed · {detail}"),
                    Style::default().fg(theme().accent_error),
                ));
            }
            OutcomeState::Cancelled => {
                self.status = "cancelled".to_string();
                self.last_turn_state = TurnState::Cancelled;
                self.last_error = None;
                self.push(accent_line(
                    theme().warning,
                    "⚠ cancelled",
                    Style::default().fg(theme().warning),
                ));
            }
        }
        // No follow(): preserve a reader's scroll position at turn end.
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
        self.push(accent_line(
            theme().accent_error,
            format!("✗ failed · {kind}: {error}{guidance_line}{limits}"),
            Style::default().fg(theme().accent_error),
        ));
        self.follow();
    }

    pub(crate) fn note_backend_waiting(&mut self, idle: Duration, threshold: Duration) {
        let _ = (idle, threshold);
        self.push(accent_line(
            theme().warning,
            "⚠ Still thinking. Ctrl-C cancels; keep waiting to continue.",
            Style::default().fg(theme().warning),
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

    pub(crate) fn handle_mouse(&mut self, mouse: crossterm::event::MouseEvent) {
        use crossterm::event::{MouseButton, MouseEventKind};
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                if let Some(picker) = self.picker.as_mut() {
                    picker.up();
                } else if self.completion.is_some() {
                    self.completion_move(-1);
                } else {
                    self.scroll_up(3);
                }
            }
            MouseEventKind::ScrollDown => {
                if let Some(picker) = self.picker.as_mut() {
                    picker.down();
                } else if self.completion.is_some() {
                    self.completion_move(1);
                } else {
                    self.scroll_down(3);
                }
            }
            // Left press/drag/release drive text selection; a press with no drag
            // falls through to a fold on release.
            MouseEventKind::Down(MouseButton::Left) => self.mouse_down(mouse.column, mouse.row),
            MouseEventKind::Drag(MouseButton::Left) => self.mouse_drag(mouse.column, mouse.row),
            MouseEventKind::Up(MouseButton::Left) => self.mouse_up(mouse.column, mouse.row),
            _ => {}
        }
    }

    /// Left-button press: drop a selection anchor on the line under the cursor.
    /// Folding is deferred to release so a click and a drag can be told apart.
    fn mouse_down(&mut self, col: u16, row: u16) {
        self.clear_selection();
        if let Some(point) = self.point_at(col, row) {
            self.select_anchor = Some(point);
            self.select_cursor = Some(point);
            self.select_dragged = false;
        }
    }

    /// Left-button drag: extend the selection to the point under the cursor,
    /// clamping the row into the transcript so a drag past an edge selects to it.
    fn mouse_drag(&mut self, col: u16, row: u16) {
        if self.select_anchor.is_none() {
            return;
        }
        if let Some(point) = self.point_at_clamped(col, row) {
            self.select_cursor = Some(point);
            self.select_dragged = true;
        }
    }

    /// Left-button release: a real drag copies the selection; a plain click (no
    /// motion) folds the tool-output block under it.
    fn mouse_up(&mut self, col: u16, row: u16) {
        if self.select_dragged {
            self.copy_selection();
        } else {
            self.clear_selection();
            self.handle_click(col, row);
        }
    }

    /// The selected flattened-line range `(lo, hi)` inclusive, if a selection is
    /// active.
    pub(crate) fn selection_range(&self) -> Option<(usize, usize)> {
        match (self.select_anchor, self.select_cursor) {
            (Some(a), Some(b)) => Some((a.0.min(b.0), a.0.max(b.0))),
            _ => None,
        }
    }

    /// A character-precise selection `(line, col_lo, col_hi)` when both ends sit
    /// on the same non-wrapped line — so dragging within one line copies just
    /// those characters. `None` when the selection spans lines or a wrapped line
    /// (where a screen column can't be mapped to a character unambiguously), in
    /// which case whole-line selection applies.
    pub(crate) fn char_span(&self) -> Option<(usize, usize, usize)> {
        let (a, b) = (self.select_anchor?, self.select_cursor?);
        if a.0 != b.0 {
            return None;
        }
        let line = a.0;
        // Single display row only (prefix rows for this line == 1).
        let rows = self
            .view_prefix
            .get(line + 1)?
            .checked_sub(*self.view_prefix.get(line)?)?;
        if rows != 1 {
            return None;
        }
        let len = self
            .view_line_texts
            .get(line)
            .map(|t| t.chars().count())
            .unwrap_or(0);
        let lo = a.1.min(b.1).min(len);
        let hi = a.1.max(b.1).min(len);
        (lo < hi).then_some((line, lo, hi))
    }

    pub(crate) fn clear_selection(&mut self) {
        self.select_anchor = None;
        self.select_cursor = None;
        self.select_dragged = false;
    }

    /// The `(line, column)` under terminal `(col, row)`, or `None` if the point is
    /// outside the transcript's inner area. The column is the character offset
    /// from the line's left edge (meaningful for non-wrapped lines).
    fn point_at(&self, col: u16, row: u16) -> Option<(usize, usize)> {
        let a = self.view_inner;
        if a.width == 0
            || a.height == 0
            || col < a.x
            || col >= a.x + a.width
            || row < a.y
            || row >= a.y + a.height
        {
            return None;
        }
        let line = self.line_at_row(self.view_scroll as u32 + (row - a.y) as u32)?;
        Some((line, (col - a.x) as usize))
    }

    /// Like [`Self::point_at`] but clamps both axes into the transcript, so a drag
    /// past an edge keeps extending to that corner.
    fn point_at_clamped(&self, col: u16, row: u16) -> Option<(usize, usize)> {
        let a = self.view_inner;
        if a.width == 0 || a.height == 0 {
            return None;
        }
        let rel_row = row.clamp(a.y, a.y + a.height - 1) - a.y;
        let rel_col = col.clamp(a.x, a.x + a.width - 1) - a.x;
        let line = self.line_at_row(self.view_scroll as u32 + rel_row as u32)?;
        Some((line, rel_col as usize))
    }

    /// Map an absolute wrapped-row to the flattened line index it falls in, using
    /// the prefix sums cached by the last render.
    fn line_at_row(&self, abs_row: u32) -> Option<usize> {
        let p = &self.view_prefix;
        if p.len() < 2 {
            return None;
        }
        let i = match p.binary_search(&abs_row) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        Some(i.min(p.len() - 2))
    }

    /// The selected text (trimmed of trailing blank), or `None` if there's no
    /// selection or it's empty. A single-line character selection yields just
    /// those characters; anything else yields the whole selected lines. Pure — no
    /// clipboard side effect.
    pub(crate) fn selected_text(&self) -> Option<String> {
        if let Some((line, lo, hi)) = self.char_span() {
            let chars: Vec<char> = self.view_line_texts.get(line)?.chars().collect();
            let text: String = chars[lo.min(chars.len())..hi.min(chars.len())]
                .iter()
                .collect();
            let text = text.trim_end();
            return (!text.is_empty()).then(|| text.to_string());
        }
        let (lo, hi) = self.selection_range()?;
        if self.view_line_texts.is_empty() {
            return None;
        }
        let hi = hi.min(self.view_line_texts.len() - 1);
        let lo = lo.min(hi);
        let text = self.view_line_texts[lo..=hi].join("\n");
        let text = text.trim_end();
        (!text.is_empty()).then(|| text.to_string())
    }

    /// Copy the selected line range to the clipboard. Success shows a brief toast;
    /// the highlight also stays put as in-place feedback. Failures print a line.
    fn copy_selection(&mut self) {
        let Some(text) = self.selected_text() else {
            return;
        };
        match crate::util::copy_to_clipboard(&text) {
            Ok(()) => self.copy_toast = Some((text.chars().count(), Instant::now())),
            Err(err) => {
                self.push(Line::styled(
                    format!("copy failed: {err}"),
                    Style::default().fg(theme().warning),
                ));
                self.follow();
            }
        }
    }

    /// Map a click at terminal `(col, row)` to the tool-output block under it (if
    /// any) using the geometry cached by the last render, and toggle its fold.
    pub(crate) fn handle_click(&mut self, col: u16, row: u16) {
        let a = self.view_inner;
        if a.width == 0
            || a.height == 0
            || col < a.x
            || col >= a.x + a.width
            || row < a.y
            || row >= a.y + a.height
        {
            return;
        }
        let abs = self.view_scroll as u32 + (row - a.y) as u32;
        // Deep-link: if the click lands on a `✎ files changed` line, open the
        // full-screen diff review filtered to those files.
        if let Some(files) = self.changed_files_at_flat_line(abs as usize) {
            self.open_review(Some(&files));
            return;
        }
        if let Some(&(_, _, ord)) = self
            .block_row_spans
            .iter()
            .find(|&&(start, end, _)| abs >= start && abs < end)
        {
            self.block_cursor = ord;
            self.toggle_block_ord(ord);
        }
    }

    /// If flattened line `abs` falls on a `ChangedFiles` transcript entry,
    /// return its file list — so a click can deep-link to the diff review.
    /// Walks the transcript accumulating each entry's flattened line count
    /// (matching the render pass's `flatten` output length).
    pub(crate) fn changed_files_at_flat_line(&self, abs: usize) -> Option<Vec<String>> {
        let mut line_idx = 0usize;
        for entry in &self.transcript {
            let count = entry
                .flatten(self.show_reasoning, self.show_tool_output)
                .len();
            if abs >= line_idx && abs < line_idx + count {
                if let crate::TranscriptEntry::ChangedFiles { files, .. } = entry {
                    return Some(files.clone());
                }
                return None;
            }
            line_idx += count;
        }
        None
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
            if markdown {
                self.commit_md_line(text);
            } else {
                self.transcript
                    .push(TranscriptEntry::Line(Line::styled(text, style)));
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
        let line = markdown_line(&text, &mut self.code_lang);
        self.transcript.push(TranscriptEntry::Line(line));
    }

    /// Emit the accumulated pipe table as aligned rows, clearing the buffer.
    fn flush_table(&mut self) {
        if self.table_buf.is_empty() {
            return;
        }
        let rows = std::mem::take(&mut self.table_buf);
        for line in crate::render::render_table(&rows) {
            self.transcript.push(TranscriptEntry::Line(line));
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
            }
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
                    self.push(tool_header(&label));
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
                // The status stream is informational — a muted gutter + muted
                // text so it reads as agent chatter, not as the user's own words
                // (which historically shared this color).
                self.push(accent_line(
                    theme().gray_dim,
                    text,
                    Style::default().fg(theme().status),
                ));
            }
            UiEvent::CheckpointWarning { text } => {
                self.event_log.push("checkpoint integrity warning".into());
                self.checkpoint_warning = Some(text.clone());
                self.flush_pending();
                self.push(accent_line(
                    theme().warning,
                    text,
                    Style::default().fg(theme().warning),
                ));
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
                // This callback is a usage summary, not a completion result.
                // The typed `TurnOutcome` returned after final workspace
                // reconciliation decides Done/Warning/Failed/Cancelled.
                self.push(accent_line(
                    theme().gray_dim,
                    format!("usage · {summary}"),
                    dim(),
                ));
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
                let line = accent_line(
                    theme().accent_success,
                    format!("✎ {} {} changed: {}", files.len(), label, clipped),
                    Style::default().fg(theme().accent_success),
                );
                self.transcript
                    .push(TranscriptEntry::ChangedFiles { line, files });
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
                let line = tool_header(&self.render_explore_run(&header));
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
            let line = tool_header(&self.render_explore_run(&header));
            self.push(line);
            return;
        }
        // A non-explore result breaks any active explore run.
        self.explore_run = None;
        if result.trim().is_empty() {
            self.push(accent_line(theme().gray_dim, "(no output)", dim()));
            return;
        }
        // Keep the *entire* output — it becomes a foldable ToolOutput block that
        // shows a preview by default and expands on Ctrl-O. (The old path hard-
        // truncated at 16 lines and discarded the rest.)
        let lines: Vec<Line<'static>> = if !result.contains('\u{1b}') && looks_like_diff(result) {
            diff_lines(result)
        } else {
            // ANSI (already-colored) or non-diff text: parse escapes as before.
            result
                .into_text()
                .unwrap_or_else(|_| Text::from(result.to_string()))
                .lines
        };
        // Sit tool output under a dim continuation gutter so it reads as the
        // body of the tool block above it, not free-floating text.
        let body: Vec<Line<'static>> = lines
            .into_iter()
            .map(|mut line| {
                line.spans.insert(0, gutter(theme().gray_dim));
                line
            })
            .collect();
        self.transcript.push(TranscriptEntry::ToolOutput {
            body,
            expanded: false,
        });
        self.cap_transcript();
    }

    /// Render the current explore run as a single transcript label (no bullet —
    /// the caller wraps it with [`tool_header`]). A run of one shows the per-call
    /// label and line count (`read src/a.rs · 113 lines`); a run of many collapses
    /// to a summary (`read 6 files · 743 lines`).
    fn render_explore_run(&self, header: &str) -> String {
        let run = match &self.explore_run {
            Some(r) => r,
            None => return header.to_string(),
        };
        if run.count <= 1 {
            if run.all_empty {
                format!("{header} · (no output)")
            } else {
                let s = if run.lines == 1 { "" } else { "s" };
                format!("{header} · {} line{}", run.lines, s)
            }
        } else {
            // Multi-call summary: drop the per-file label, show counts.
            let noun = match run.tool.as_str() {
                "read" => "files",
                _ => "calls",
            };
            if run.all_empty {
                format!("{} {} {} · (no output)", run.tool, run.count, noun)
            } else {
                let s = if run.lines == 1 { "" } else { "s" };
                format!(
                    "{} {} {} · {} line{}",
                    run.tool, run.count, noun, run.lines, s
                )
            }
        }
    }

    /// Replace the last transcript line in place (used to update a merged
    /// explore-run line as more results fold in). No-op if the transcript is
    /// empty or the last entry isn't a plain line.
    fn replace_last_line(&mut self, line: Line<'static>) {
        if let Some(TranscriptEntry::Line(slot)) = self.transcript.last_mut() {
            *slot = line;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutcomeState {
    Done,
    Warning,
    Failed,
    Cancelled,
}

fn outcome_state(outcome: &TurnOutcome) -> OutcomeState {
    if outcome.status == TurnStatus::Cancelled {
        OutcomeState::Cancelled
    } else if outcome.status == TurnStatus::Failed
        || outcome.verification == VerificationStatus::InfrastructureError
    {
        OutcomeState::Failed
    } else if outcome.status == TurnStatus::Completed
        && matches!(
            outcome.verification,
            VerificationStatus::Passed | VerificationStatus::NotApplicable
        )
        && outcome.review != ReviewStatus::Objected
    {
        OutcomeState::Done
    } else {
        OutcomeState::Warning
    }
}

fn outcome_detail(outcome: &TurnOutcome) -> String {
    let base = match outcome.stop_reason {
        TurnStopReason::Completed => match outcome.verification {
            VerificationStatus::Passed => "verified",
            VerificationStatus::NotApplicable => "no applicable checks",
            VerificationStatus::Unverified => "unverified changes",
            VerificationStatus::Failed => "verification failed",
            VerificationStatus::InfrastructureError => "verification infrastructure failure",
        },
        TurnStopReason::NoApplicableVerification => "no applicable checks",
        TurnStopReason::VerificationUnavailable => "unverified changes",
        TurnStopReason::VerificationFailed => "verification failed",
        TurnStopReason::VerificationUnstable => "verification was unstable",
        TurnStopReason::ReviewObjected => "review objected",
        TurnStopReason::ToolModeDenied => "required tool was denied",
        TurnStopReason::StepLimit => "step limit reached",
        TurnStopReason::Stalled => "stalled",
        TurnStopReason::Cancelled => "cancelled",
        TurnStopReason::InfrastructureFailure => "infrastructure failure",
    };
    match outcome.review {
        ReviewStatus::Passed if outcome.verification == VerificationStatus::Passed => {
            format!("{base} · reviewed")
        }
        ReviewStatus::Unavailable if outcome.verification == VerificationStatus::Passed => {
            format!("{base} · review unavailable")
        }
        _ => base.to_string(),
    }
}
