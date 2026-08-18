//! `App` methods: transcript.

use std::time::{Duration, Instant};

/// Drop the decorative left gutter baked into painted transcript lines so
/// whole-line selection copies content, not chrome. Matches the prefixes used
/// by [`crate::render::gutter`] (`┃ `) and fenced-code rendering (`▏ `).
fn strip_display_gutter(line: &str) -> &str {
    line.strip_prefix("┃ ")
        .or_else(|| line.strip_prefix("▏ "))
        .or_else(|| line.strip_prefix("• "))
        .or_else(|| line.strip_prefix("◆ "))
        .unwrap_or(line)
}

use hi_agent::ui::tool_label;
use hi_agent::{ReviewStatus, TurnOutcome, TurnStatus, TurnStopReason, VerificationStatus};
use ratatui::style::Style;
use ratatui::text::Line;

use crate::activity_feed::{self, ActivityBlock, ActivityKind, ExploreVerb, label_detail};
use crate::event::UiEvent;
use crate::render::{accent_line, dim, markdown_line};
use crate::theme::theme;
use crate::util::fmt_rate_limits;
use crate::{MAX_EVENT_LOG, MAX_TRANSCRIPT_LINES, TranscriptEntry, TurnEventKind, TurnState};

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

    /// Push a user-prompt echo as a structurally-distinct entry so the render
    /// pass can pin it as a sticky header when scrolled past.
    pub(crate) fn push_user_prompt(&mut self, line: Line<'static>) {
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

    /// Apply the authoritative typed result returned by `Agent::run_turn`.
    ///
    /// `Ui::turn_end` carries token accounting only and can arrive before final
    /// workspace reconciliation. It must therefore never decide whether a turn
    /// succeeded. This is the sole success-state transition for a normal turn.
    pub(crate) fn note_turn_outcome(&mut self, outcome: &TurnOutcome) {
        self.last_stop_reason = Some(outcome.stop_reason);
        let detail = outcome_detail(outcome);
        match outcome_state(outcome) {
            OutcomeState::Done => {
                self.status = format!("done · {detail}");
                self.last_turn_state = TurnState::Done(detail.clone());
                self.last_error = None;
                // “No applicable checks” is a non-event. Keep the typed state
                // for /status, but don't paint a green receipt into the pane.
                if outcome.verification == VerificationStatus::Passed {
                    self.push(accent_line(
                        theme().accent_success,
                        format!("✓ done · {detail}"),
                        dim(),
                    ));
                }
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
                // Infrastructure failures are internal (provider/runner/session).
                // Keep typed state for reports/eval, but don't dump the jargon
                // banner into the user transcript.
                if !is_infrastructure_failure_detail(&detail) {
                    self.push(accent_line(
                        theme().accent_error,
                        format!("✗ failed · {detail}"),
                        Style::default().fg(theme().accent_error),
                    ));
                }
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

    /// Scroll to the top of the transcript (line 0).
    pub(crate) fn scroll_to_top(&mut self) {
        self.following = false;
        self.page_flip_on_send = false;
        self.scroll = 0;
    }

    /// Scroll to the bottom of the transcript (follow the latest content).
    pub(crate) fn scroll_to_bottom(&mut self) {
        self.following = true;
        self.page_flip_on_send = false;
    }

    /// Scroll to an absolute line index. Clamped to the valid scroll range.
    pub(crate) fn scroll_to(&mut self, line: u16) {
        let max = self.view_max_scroll;
        self.scroll = line.min(max);
        self.following = line >= max;
    }

    pub(crate) fn handle_mouse(&mut self, mouse: crossterm::event::MouseEvent) {
        use crossterm::event::{MouseButton, MouseEventKind};
        self.mouse_col = mouse.column;
        self.mouse_row = mouse.row;
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                if !self.scroll_btw_overlay_at(mouse.column, mouse.row, true) {
                    if let Some(picker) = self.picker.as_mut() {
                        picker.up();
                    } else if self.completion.is_some() {
                        self.completion_move(-1);
                    } else {
                        self.scroll_up(3);
                    }
                }
            }
            MouseEventKind::ScrollDown => {
                if !self.scroll_btw_overlay_at(mouse.column, mouse.row, false) {
                    if let Some(picker) = self.picker.as_mut() {
                        picker.down();
                    } else if self.completion.is_some() {
                        self.completion_move(1);
                    } else {
                        self.scroll_down(3);
                    }
                }
            }
            MouseEventKind::Down(MouseButton::Left)
                if crate::btw::cell_in(self.last_btw_close, mouse.column, mouse.row) =>
            {
                let _ = self.dismiss_btw_overlay();
            }
            // Left press/drag/release drive text selection; a press with no drag
            // falls through to a fold on release.
            MouseEventKind::Down(MouseButton::Left) => {
                if self.plan_approval.as_ref().is_some_and(|c| c.parked)
                    && crate::btw::cell_in(self.turn_status_rect, mouse.column, mouse.row)
                {
                    self.unpark_plan_approval();
                    return;
                }
                if self.apply_timeline_click(mouse.column, mouse.row) {
                    return;
                }
                self.mouse_down(mouse.column, mouse.row);
            }
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

    fn timeline_contains(&self, col: u16, row: u16) -> bool {
        let r = self.timeline_rect;
        r.width > 0
            && col >= r.x
            && col < r.x.saturating_add(r.width)
            && row >= r.y
            && row < r.y.saturating_add(r.height)
    }

    fn apply_timeline_click(&mut self, col: u16, row: u16) -> bool {
        if !self.timeline_contains(col, row) {
            return false;
        }
        if let Some(hit) = crate::timeline::hit_at(&self.timeline_hits, row) {
            match hit {
                crate::timeline::TimelineHit::Tick(i) => {
                    let _ = self.scroll_to_user_prompt(i);
                }
                crate::timeline::TimelineHit::Up => {
                    self.jump_transcript_marker(crate::dispatch::TranscriptMarker::UserPrompt, -1);
                }
                crate::timeline::TimelineHit::Down => {
                    self.jump_transcript_marker(crate::dispatch::TranscriptMarker::UserPrompt, 1);
                }
            }
        }
        true
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
    ///
    /// Some terminals omit intermediate `Drag` events and only deliver
    /// Down + Up at different cells. Apply the release point here so that
    /// path still extends the selection and auto-copies.
    fn mouse_up(&mut self, col: u16, row: u16) {
        if self.timeline_contains(col, row) {
            self.clear_selection();
            return;
        }
        if self.select_anchor.is_none() {
            self.handle_click(col, row);
            return;
        }
        // Finalize the cursor from the release cell even when Drag was dropped.
        if let Some(point) = self.point_at_clamped(col, row)
            && self.select_cursor != Some(point)
        {
            self.select_cursor = Some(point);
            self.select_dragged = true;
        }
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
    ///
    /// Whole-line copy strips decorative display gutters (`┃ ` tool/status,
    /// `▏ ` fenced code) so the clipboard matches the underlying content.
    /// Character spans keep the raw slice — columns already index into the
    /// painted line, including any gutter the user dragged across.
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
        let text = self.view_line_texts[lo..=hi]
            .iter()
            .map(|line| strip_display_gutter(line))
            .collect::<Vec<_>>()
            .join("\n");
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
                .flatten(self.show_reasoning, self.show_tool_output, self.density)
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
            self.page_flip_on_send = false;
        } else {
            if self.following {
                self.total_when_unpinned = self.view_total;
            }
            self.following = false;
            self.page_flip_on_send = false;
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
        if !in_fence
            && text.trim().is_empty()
            && last_entry_is_blank(&self.transcript)
            && !self.transcript.is_empty()
        {
            return;
        }
        let _ = markdown_line(&text, &mut self.code_lang);
        append_assistant_line(&mut self.transcript, &text);
        self.bump_transcript();
    }

    /// Emit the accumulated pipe table as aligned rows, clearing the buffer.
    fn flush_table(&mut self) {
        if self.table_buf.is_empty() {
            return;
        }
        let rows = std::mem::take(&mut self.table_buf);
        for row in &rows {
            append_assistant_line(&mut self.transcript, row);
        }
        self.bump_transcript();
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
                self.flush_pending();
                if !self.current_assistant.trim().is_empty() {
                    self.last_assistant = self.current_assistant.trim().to_string();
                }
                self.current_assistant.clear();
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
                    self.flush_pending();
                    self.push(Line::styled(text, Style::default().fg(theme().status)));
                }
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
                self.bump_transcript();
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
                | TranscriptEntry::ChangedFiles { .. }
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
                | TranscriptEntry::ChangedFiles { .. }
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

    fn apply_run_result(&mut self, command: &str, display: &str, live: bool) {
        self.freeze_verb_group();
        let chunk = strip_bg_status_lines(display);
        if live {
            if !self.run_streamed_this_call && !chunk.trim().is_empty() {
                self.append_run_output_for(command, &chunk);
            }
            self.touch_idle_run(command);
            return;
        }
        if let Some(TranscriptEntry::Activity(block)) = self.transcript.last_mut()
            && let Some((existing, dest, idle, _)) = block.as_run_mut()
            && (existing == command || *idle)
        {
            if !self.run_streamed_this_call {
                dest.clear();
                dest.push_str(&chunk);
            }
            *idle = false;
            self.bump_transcript();
            return;
        }
        self.push_activity(ActivityKind::Run {
            command: command.to_string(),
            body: chunk,
            idle: false,
            poll_count: 0,
        });
    }

    fn append_live_run_output(&mut self, line: &str) -> bool {
        for entry in self.transcript.iter_mut().rev() {
            if let TranscriptEntry::Activity(block) = entry
                && let Some((_, body, idle, _)) = block.as_run_mut()
                && *idle
            {
                append_capped_run_body(body, line);
                self.bump_transcript();
                return true;
            }
        }
        false
    }

    fn append_run_output_for(&mut self, command: &str, chunk: &str) {
        for entry in self.transcript.iter_mut().rev() {
            if let TranscriptEntry::Activity(block) = entry
                && let Some((existing, body, idle, _)) = block.as_run_mut()
                && (existing == command || *idle)
            {
                append_capped_run_body(body, chunk);
                *idle = true;
                self.bump_transcript();
                return;
            }
        }
        let mut body = String::new();
        append_capped_run_body(&mut body, chunk);
        self.push_activity(ActivityKind::Run {
            command: command.to_string(),
            body,
            idle: true,
            poll_count: 0,
        });
    }

    fn idle_run_count_and_match(&self, command: &str) -> (usize, bool) {
        let mut count = 0;
        let mut matching = false;
        for entry in &self.transcript {
            if let TranscriptEntry::Activity(block) = entry
                && let ActivityKind::Run {
                    command: existing,
                    idle: true,
                    ..
                } = &block.kind
            {
                count += 1;
                if existing == command {
                    matching = true;
                }
            }
        }
        (count, matching)
    }

    fn touch_idle_run(&mut self, command: &str) {
        let (count, matching) = self.idle_run_count_and_match(command);
        if matching || count == 1 {
            for entry in self.transcript.iter_mut().rev() {
                if let TranscriptEntry::Activity(block) = entry
                    && let Some((existing, _, idle, poll_count)) = block.as_run_mut()
                    && *idle
                    && (existing == command || count == 1)
                {
                    *poll_count = poll_count.saturating_add(1);
                    self.bump_transcript();
                    return;
                }
            }
        }
        self.push_activity(ActivityKind::Run {
            command: command.to_string(),
            body: String::new(),
            idle: true,
            poll_count: 1,
        });
    }

    fn note_idle_bash_poll(&mut self, id: &str) {
        self.freeze_verb_group();
        self.touch_idle_run(id);
    }

    fn pop_idle_run(&mut self, command: &str) {
        let drop_last = matches!(
            self.transcript.last(),
            Some(TranscriptEntry::Activity(block))
                if matches!(
                    &block.kind,
                    ActivityKind::Run {
                        command: existing,
                        idle: true,
                        ..
                    } if existing == command
                )
        );
        if drop_last {
            self.transcript.pop();
            self.bump_transcript();
        }
    }

    fn ensure_run_placeholder(&mut self, command: &str) {
        let (count, matching) = self.idle_run_count_and_match(command);
        if matching || count == 1 {
            return;
        }
        self.push_activity(ActivityKind::Run {
            command: command.to_string(),
            body: String::new(),
            idle: true,
            poll_count: 0,
        });
    }
}

fn append_assistant_line(transcript: &mut Vec<TranscriptEntry>, line: &str) {
    if let Some(TranscriptEntry::AssistantMessage { text }) = transcript.last_mut() {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(line);
        return;
    }
    transcript.push(TranscriptEntry::AssistantMessage {
        text: line.to_string(),
    });
}

fn last_entry_is_blank(transcript: &[TranscriptEntry]) -> bool {
    match transcript.last() {
        None => true,
        Some(TranscriptEntry::Assistant(line) | TranscriptEntry::Line(line)) => {
            crate::render::line_text(line).trim().is_empty()
        }
        Some(TranscriptEntry::AssistantMessage { text }) => {
            text.lines().last().is_none_or(|l| l.trim().is_empty())
        }
        _ => false,
    }
}

#[allow(dead_code)]
fn last_entry_is_list(transcript: &[TranscriptEntry]) -> bool {
    match transcript.last() {
        Some(TranscriptEntry::Assistant(line)) => {
            crate::render::is_markdown_list_line(&crate::render::line_text(line))
        }
        Some(TranscriptEntry::AssistantMessage { text }) => text
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .is_some_and(crate::render::is_markdown_list_line),
        _ => false,
    }
}

const STEERING_MAX_CHARS: usize = 140;

/// Short “let me look…” chrome, not a real answer (heading / list / document).
fn is_steering_assistant_text(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.chars().count() > STEERING_MAX_CHARS {
        return false;
    }
    if crate::render::markdown_heading(trimmed).is_some() {
        return false;
    }
    if trimmed.starts_with('▏') || trimmed.starts_with('─') || trimmed.contains('│') {
        return false;
    }
    if crate::render::is_markdown_list_line(text) {
        return false;
    }
    true
}

fn is_steering_assistant_line(line: &Line<'_>) -> bool {
    let text = crate::render::line_text(line);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.chars().count() > STEERING_MAX_CHARS {
        return false;
    }
    if crate::render::line_looks_like_heading(line) {
        return false;
    }
    if trimmed.starts_with('▏') || trimmed.starts_with('─') || trimmed.contains('│') {
        return false;
    }
    if crate::render::is_markdown_list_line(&text) {
        return false;
    }
    true
}

#[derive(Default)]
struct ExploreChrome {
    thinking: String,
    thinking_elapsed: Duration,
    steering: Vec<String>,
}

fn absorb_explore_chrome(group: &mut crate::activity_feed::VerbGroup, chrome: ExploreChrome) {
    if !chrome.thinking.trim().is_empty() {
        if !group.thinking.is_empty() {
            group.thinking.push('\n');
        }
        group.thinking.push_str(&chrome.thinking);
        group.thinking_elapsed = group
            .thinking_elapsed
            .saturating_add(chrome.thinking_elapsed);
    }
    group.steering.extend(chrome.steering);
}

fn bash_process_live(result: &str) -> bool {
    result.lines().next().is_some_and(|status| {
        status.contains("still running")
            || status.contains("continued as")
            || (status.starts_with("Started ") && status.contains('('))
    })
}

fn strip_bg_status_lines(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim();
            !(trimmed.starts_with('[')
                && trimmed.ends_with(']')
                && (trimmed.contains("still running")
                    || trimmed.contains("exited")
                    || trimmed.contains("stopped")
                    || trimmed.contains(": failed")))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

const LIVE_RUN_BODY_MAX: usize = 64 * 1024;

fn append_capped_run_body(body: &mut String, chunk: &str) {
    if chunk.is_empty() {
        return;
    }
    if !body.is_empty() && !body.ends_with('\n') && !chunk.starts_with('\n') {
        body.push('\n');
    }
    body.push_str(chunk);
    if !body.ends_with('\n') {
        body.push('\n');
    }
    if body.len() <= LIVE_RUN_BODY_MAX {
        return;
    }
    let overflow = body.len() - LIVE_RUN_BODY_MAX;
    let cut = body[overflow..]
        .find('\n')
        .map(|i| overflow + i + 1)
        .unwrap_or(overflow)
        .min(body.len());
    body.replace_range(..cut, "");
}

fn bash_output_is_idle(result: &str) -> bool {
    result.lines().next().is_some_and(|status| {
        status.contains("still running — no new output")
            || status.contains("running — no new output")
    })
}

fn is_missing_background_process_result(result: &str) -> bool {
    result
        .trim_start()
        .starts_with("Error: no background process")
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
    } else if (outcome.status == TurnStatus::Completed
        && matches!(
            outcome.verification,
            VerificationStatus::Passed | VerificationStatus::NotApplicable
        )
        || outcome.status == TurnStatus::Incomplete
            && outcome.verification == VerificationStatus::Passed)
        && outcome.review != ReviewStatus::Objected
    {
        // Escalated is a completed scar, not a defect objection.
        OutcomeState::Done
    } else {
        OutcomeState::Warning
    }
}

fn is_infrastructure_failure_detail(detail: &str) -> bool {
    detail == "infrastructure failure"
        || detail == "verification infrastructure failure"
        || detail.starts_with("infrastructure failure")
        || detail.starts_with("verification infrastructure failure")
}

fn outcome_detail(outcome: &TurnOutcome) -> String {
    // Keep the user-facing line single-axis: a settled deterministic pass is
    // a successful outcome, even if a steering guard also fired. The guard is
    // retained in telemetry/stop_reason for diagnostics, but concatenating it
    // with "incomplete" and "verified" produced the contradictory banner
    // users reported.
    let green_settled = matches!(
        outcome.status,
        TurnStatus::Completed | TurnStatus::Incomplete
    ) && outcome.verification == VerificationStatus::Passed
        && outcome.review != ReviewStatus::Objected;
    let base = if green_settled {
        "verified".to_string()
    } else if outcome.status == TurnStatus::Incomplete {
        if let Some(leftover) = outcome
            .leftover
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            leftover.to_string()
        } else {
            match outcome.stop_reason {
                TurnStopReason::Completed => match outcome.verification {
                    VerificationStatus::Passed => "verified",
                    VerificationStatus::NotApplicable => "no applicable checks",
                    VerificationStatus::Unverified => "checks did not settle",
                    VerificationStatus::Failed => "verification failed",
                    VerificationStatus::InfrastructureError => {
                        "verification infrastructure failure"
                    }
                },
                TurnStopReason::NoApplicableVerification => "no applicable checks",
                TurnStopReason::VerificationUnavailable => "checks did not settle",
                TurnStopReason::VerificationFailed => "verification failed",
                TurnStopReason::VerificationUnstable => "verification was unstable",
                TurnStopReason::ReviewObjected => "review objected",
                TurnStopReason::ReviewEscalated => "review escalated",
                TurnStopReason::ToolModeDenied => "required tool was denied",
                TurnStopReason::StepLimit => "step limit reached",
                TurnStopReason::TimeLimit => "time budget reached",
                TurnStopReason::TurnLimit => "turn limit reached",
                TurnStopReason::Stalled => "stalled",
                TurnStopReason::Cancelled => "cancelled",
                TurnStopReason::InfrastructureFailure => "infrastructure failure",
            }
            .to_string()
        }
    } else {
        match outcome.stop_reason {
            TurnStopReason::Completed => match outcome.verification {
                VerificationStatus::Passed => "verified",
                VerificationStatus::NotApplicable => "no applicable checks",
                VerificationStatus::Unverified => "checks did not settle",
                VerificationStatus::Failed => "verification failed",
                VerificationStatus::InfrastructureError => "verification infrastructure failure",
            },
            TurnStopReason::NoApplicableVerification => "no applicable checks",
            TurnStopReason::VerificationUnavailable => "checks did not settle",
            TurnStopReason::VerificationFailed => "verification failed",
            TurnStopReason::VerificationUnstable => "verification was unstable",
            TurnStopReason::ReviewObjected => "review objected",
            TurnStopReason::ReviewEscalated => "review escalated",
            TurnStopReason::ToolModeDenied => "required tool was denied",
            TurnStopReason::StepLimit => "step limit reached",
            TurnStopReason::TimeLimit => "time budget reached",
            TurnStopReason::TurnLimit => "turn limit reached",
            TurnStopReason::Stalled => "stalled",
            TurnStopReason::Cancelled => "cancelled",
            TurnStopReason::InfrastructureFailure => "infrastructure failure",
        }
        .to_string()
    };
    match outcome.review {
        ReviewStatus::Passed if outcome.verification == VerificationStatus::Passed => {
            format!("{base} · reviewed")
        }
        // A review transport failure is non-blocking after deterministic
        // verification passes. Keep it in the report/debug telemetry rather
        // than turning a green result into a noisy warning banner.
        ReviewStatus::Unavailable if outcome.verification == VerificationStatus::Passed => base,
        ReviewStatus::Objected if base == "review objected" => base,
        ReviewStatus::Objected => format!("{base} · review objected"),
        ReviewStatus::Escalated => format!("{base} · review escalated"),
        _ => base,
    }
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
