//! Transcript navigation, mouse interaction, selection, and clipboard copy.

use std::time::Instant;

use ratatui::style::Style;
use ratatui::text::Line;

use crate::TranscriptEntry;
use crate::theme::theme;

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

impl crate::App {
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
        if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left))
            && crate::btw::cell_in(self.changed_files_rect, mouse.column, mouse.row)
        {
            let files = self.last_changed_files.clone();
            if !files.is_empty() {
                self.open_review(Some(&files));
            }
            return;
        }
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
        if let Some(&(_, _, ord)) = self
            .block_row_spans
            .iter()
            .find(|&&(start, end, _)| abs >= start && abs < end)
        {
            self.block_cursor = ord;
            self.toggle_block_ord(ord);
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
}
