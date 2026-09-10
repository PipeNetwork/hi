//! Docked / overlay working-tree diff review (Ctrl-G).
//!
//! Wide terminals dock a right-hand pane so the composer stays usable. Narrow
//! terminals keep the exclusive full-screen overlay. Selecting a hunk (click or
//! `n/p` while focused) writes an `@path:N-M` chip; send attaches a hunk quote.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, BorderType, Paragraph};

use crate::chrome::ShortcutHint;
use crate::inline_diff::is_hunk_gap;
use crate::layout::UiLayout;
use crate::mode::UiMode;
use crate::render::{diff_lines, dim, hunk_start_indices, line_text};
use crate::theme::{UiTone, theme};
use crate::{diff_for_files_sync, working_tree_diff_sync};

const HUNK_QUOTE_CAP: usize = 30;
const REVIEW_CHIP_TAG: &str = "@";
const MIN_TRANSCRIPT: u16 = 24;
const MIN_PANE: u16 = 36;
const SPLIT_GAP: u16 = 1;

/// Session state for the git-diff review surface.
#[derive(Clone, Debug, Default)]
pub(crate) struct ReviewState {
    pub open: bool,
    pub focused: bool,
    pub scroll: usize,
    pub selected_hunk: Option<usize>,
    pub diff_text: Option<String>,
    pub filter: Option<Vec<String>>,
    pub rect: Rect,
    pub width: Option<u16>,
    pub gap_rect: Rect,
    pub resizing: bool,
    resize_anchor_col: u16,
    resize_anchor_width: u16,
}

/// One unified-diff hunk, aligned to painted review rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReviewHunk {
    pub path: String,
    pub start: Option<u32>,
    pub end: Option<u32>,
    pub painted_start: usize,
    pub painted_end: usize,
}

impl ReviewHunk {
    pub(crate) fn mention(&self) -> Option<String> {
        if self.path.is_empty() {
            return None;
        }
        Some(match (self.start, self.end) {
            (Some(a), Some(b)) if a != b => format!("@{}:{a}-{b}", self.path),
            (Some(a), _) => format!("@{}:{a}", self.path),
            _ => format!("@{}", self.path),
        })
    }

    pub(crate) fn label(&self) -> String {
        match (self.start, self.end) {
            (Some(a), Some(b)) if a != b => format!("{}:{a}-{b}", self.path),
            (Some(a), _) => format!("{}:{a}", self.path),
            _ if self.path.is_empty() => "diff".to_string(),
            _ => self.path.clone(),
        }
    }
}

pub(crate) fn can_dock(frame_width: u16) -> bool {
    UiLayout::from_width(frame_width) == UiLayout::Wide
}

fn hint(key: &'static str, label: &'static str) -> ShortcutHint {
    ShortcutHint { key, label }
}

pub(crate) fn docked_session_hints(focused: bool) -> Vec<ShortcutHint> {
    if focused {
        vec![
            hint("n/p", "hunk"),
            hint("j/k", "scroll"),
            hint("space", "prompt"),
            hint("esc", "unfocus"),
            hint("q", "close"),
        ]
    } else {
        vec![
            hint("tab", "focus diff"),
            hint("ctrl+g", "close"),
            hint("?", "help"),
        ]
    }
}

pub(crate) fn default_pane_width(area_width: u16) -> u16 {
    clamp_pane_width(area_width, area_width.saturating_mul(42) / 100)
}

pub(crate) fn clamp_pane_width(area_width: u16, want: u16) -> u16 {
    let max_w = area_width.saturating_sub(MIN_TRANSCRIPT.saturating_add(SPLIT_GAP));
    if max_w == 0 {
        return 0;
    }
    want.clamp(MIN_PANE.min(max_w), max_w)
}

pub(crate) fn split_body(
    area: Rect,
    dock: bool,
    width: Option<u16>,
) -> (Rect, Option<(Rect, Rect)>) {
    if !dock || area.width < MIN_TRANSCRIPT.saturating_add(SPLIT_GAP).saturating_add(16) {
        return (area, None);
    }
    let diff_w = clamp_pane_width(
        area.width,
        width.unwrap_or_else(|| default_pane_width(area.width)),
    );
    if diff_w == 0 {
        return (area, None);
    }
    let chunks = Layout::horizontal([
        Constraint::Min(MIN_TRANSCRIPT),
        Constraint::Length(SPLIT_GAP),
        Constraint::Length(diff_w),
    ])
    .split(area);
    (chunks[0], Some((chunks[1], chunks[2])))
}

pub(crate) fn parse_review_hunks(diff: &str) -> Vec<ReviewHunk> {
    let painted = diff_lines(diff);
    let starts = hunk_start_indices(&painted);
    let mut path = String::new();
    let mut old_path = String::new();
    let mut raw = Vec::new();
    for line in diff.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("diff --git ") {
            path.clear();
            old_path.clear();
            if let Some(b) = rest.split_whitespace().nth(1) {
                path = strip_git_prefix(b);
            }
            continue;
        }
        if let Some(rest) = t.strip_prefix("--- ") {
            old_path = strip_git_prefix(rest);
            continue;
        }
        if let Some(rest) = t.strip_prefix("+++ ") {
            let newp = strip_git_prefix(rest);
            if newp != "/dev/null" {
                path = newp;
            } else if old_path != "/dev/null" {
                path.clone_from(&old_path);
            }
            continue;
        }
        if t.starts_with("@@") {
            let (start, count) = new_file_span(t);
            let end = match (start, count) {
                (Some(s), Some(c)) if c > 0 => Some(s.saturating_add(c.saturating_sub(1))),
                (Some(s), _) => Some(s),
                _ => None,
            };
            raw.push(ReviewHunk {
                path: path.clone(),
                start,
                end,
                painted_start: 0,
                painted_end: 0,
            });
        }
    }
    if raw.is_empty() && !starts.is_empty() {
        return starts
            .iter()
            .enumerate()
            .map(|(i, &s)| ReviewHunk {
                path: path.clone(),
                start: None,
                end: None,
                painted_start: s,
                painted_end: starts.get(i + 1).copied().unwrap_or(painted.len()),
            })
            .collect();
    }
    for (i, hunk) in raw.iter_mut().enumerate() {
        hunk.painted_start = starts.get(i).copied().unwrap_or(0);
        hunk.painted_end = starts.get(i + 1).copied().unwrap_or(painted.len());
    }
    raw
}

fn strip_git_prefix(s: &str) -> String {
    let s = s.trim().trim_matches('"');
    s.strip_prefix("a/")
        .or_else(|| s.strip_prefix("b/"))
        .unwrap_or(s)
        .to_string()
}

fn new_file_span(header: &str) -> (Option<u32>, Option<u32>) {
    let Some(plus) = header.split_whitespace().find(|p| p.starts_with('+')) else {
        return (None, None);
    };
    let body = plus.trim_start_matches('+');
    if let Some((a, b)) = body.split_once(',') {
        (a.parse().ok(), b.parse().ok())
    } else {
        let start = body
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .ok();
        (start, Some(1))
    }
}

pub(crate) fn replace_or_prepend_chip(text: &str, chip: &str) -> String {
    let trimmed = text.trim_start();
    if let Some(rest) = trimmed.strip_prefix(REVIEW_CHIP_TAG) {
        let (token, remainder) = match rest.split_once(char::is_whitespace) {
            Some((tok, rem)) => (tok, rem.trim_start()),
            None => (rest, ""),
        };
        if is_review_chip_token(token) {
            if remainder.is_empty() {
                return format!("{chip} ");
            }
            return format!("{chip} {remainder}");
        }
    }
    if text.trim().is_empty() {
        format!("{chip} ")
    } else {
        format!("{chip} {text}")
    }
}

fn is_review_chip_token(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    !token.contains("://") && (token.contains('/') || token.contains('.') || token.contains(':'))
}

pub(crate) fn prompt_still_has_mention(prompt: &str, mention: &str) -> bool {
    prompt.split_whitespace().any(|tok| tok == mention)
}

pub(crate) fn attach_hunk_quote(prompt: &str, hunk: &ReviewHunk, diff: &str) -> String {
    let Some(mention) = hunk.mention() else {
        return prompt.to_string();
    };
    if !prompt_still_has_mention(prompt, &mention) || prompt.contains("<review hunk>") {
        return prompt.to_string();
    }
    let painted = diff_lines(diff);
    let mut lines: Vec<String> = painted
        .iter()
        .skip(hunk.painted_start)
        .take(hunk.painted_end.saturating_sub(hunk.painted_start))
        .filter(|line| !is_hunk_gap(line))
        .map(line_text)
        .take(HUNK_QUOTE_CAP)
        .collect();
    if lines.is_empty() {
        return prompt.to_string();
    }
    if hunk.painted_end.saturating_sub(hunk.painted_start) > HUNK_QUOTE_CAP {
        lines.push("…".to_string());
    }
    format!(
        "{prompt}\n\n<review hunk>\n{}\n{}\n</review hunk>",
        hunk.label(),
        lines.join("\n")
    )
}

fn hunk_at_row(hunks: &[ReviewHunk], painted_row: usize) -> Option<usize> {
    hunks.iter().rposition(|h| h.painted_start <= painted_row)
}

impl crate::App {
    pub(crate) fn can_dock_review(&self) -> bool {
        can_dock(self.frame_width)
    }

    pub(crate) fn review_is_overlay(&self) -> bool {
        if self.can_dock_review() && self.review.open {
            return false;
        }
        self.mode.is_review() || self.review.open
    }

    pub(crate) fn sync_review_mode(&mut self) {
        if !self.review.open {
            return;
        }
        if self.can_dock_review() {
            if self.mode.is_review() {
                self.mode.to_insert();
            }
        } else if !self.mode.is_block_nav() && !self.mode.is_history_search() {
            self.mode = UiMode::Review;
            self.review.focused = true;
        }
    }

    pub(crate) fn toggle_review(&mut self) {
        if self.review.open {
            self.close_review();
        } else {
            self.open_review(None);
        }
    }

    pub(crate) fn close_review(&mut self) {
        self.review.open = false;
        self.review.focused = false;
        self.review.selected_hunk = None;
        self.review.diff_text = None;
        self.review.filter = None;
        self.review.scroll = 0;
        self.review.rect = Rect::default();
        self.review.gap_rect = Rect::default();
        self.review.resizing = false;
        if self.mode.is_review() {
            self.mode.to_insert();
        }
    }

    pub(crate) fn unfocus_or_close_review(&mut self) {
        if self.review.open && self.can_dock_review() && self.review.focused {
            self.review.focused = false;
        } else {
            self.close_review();
        }
    }

    pub(crate) fn toggle_review_focus(&mut self) {
        if !self.review.open || !self.can_dock_review() {
            return;
        }
        self.review.focused = !self.review.focused;
    }

    /// Open review. `files = None` is the whole tree; `Some` filters paths.
    pub(crate) fn open_review(&mut self, files: Option<&[String]>) {
        let diff = match files {
            None => working_tree_diff_sync(&self.workspace_root),
            Some(paths) => diff_for_files_sync(&self.workspace_root, paths),
        };
        self.review.diff_text = Some(diff);
        self.review.filter = files.map(|p| p.to_vec());
        self.review.scroll = 0;
        self.review.selected_hunk = None;
        self.review.open = true;
        self.review.focused = false;
        if self.can_dock_review() {
            if self.mode.is_review() {
                self.mode.to_insert();
            }
        } else {
            self.mode = UiMode::Review;
            self.review.focused = true;
        }
    }

    pub(crate) fn refresh_review_if_open(&mut self) {
        if !self.review.open {
            self.review.diff_text = None;
            return;
        }
        let filter = self.review.filter.clone();
        let prev_sel = self.review.selected_hunk;
        let prev_scroll = self.review.scroll;
        let diff = match filter.as_deref() {
            None => working_tree_diff_sync(&self.workspace_root),
            Some(paths) => diff_for_files_sync(&self.workspace_root, paths),
        };
        self.review.diff_text = Some(diff);
        self.review.scroll = prev_scroll;
        let n = self.review_hunks().len();
        self.review.selected_hunk = prev_sel.filter(|&i| i < n);
    }

    pub(crate) fn review_hunks(&self) -> Vec<ReviewHunk> {
        parse_review_hunks(self.review.diff_text.as_deref().unwrap_or(""))
    }

    pub(crate) fn review_scroll_by(&mut self, delta: i32) {
        if delta == crate::action::Action::REVIEW_SCROLL_END {
            let total = diff_lines(self.review.diff_text.as_deref().unwrap_or("")).len();
            self.review.scroll = total;
            return;
        }
        if delta > 0 {
            self.review.scroll = self.review.scroll.saturating_add(delta as usize);
        } else {
            self.review.scroll = self
                .review
                .scroll
                .saturating_sub(delta.unsigned_abs() as usize);
        }
    }

    pub(crate) fn review_jump_hunk(&mut self, dir: i32) {
        self.review.scroll =
            crate::app::review_next_hunk(self.review.diff_text.as_deref(), self.review.scroll, dir);
        self.select_hunk_at_painted(self.review.scroll);
    }

    pub(crate) fn select_hunk_at_painted(&mut self, painted_row: usize) {
        let hunks = self.review_hunks();
        let Some(idx) = hunk_at_row(&hunks, painted_row) else {
            return;
        };
        self.review.selected_hunk = Some(idx);
        self.review.scroll = hunks[idx].painted_start;
        if let Some(chip) = hunks[idx].mention() {
            let next = replace_or_prepend_chip(&self.input.text(), &chip);
            let cursor_at_end = self.input.cursor() >= self.input.chars.len();
            self.input.set(&next);
            if !cursor_at_end {
                let pos = chip.len().saturating_add(1).min(self.input.chars.len());
                self.input.cursor = pos;
            }
        }
    }

    pub(crate) fn attach_review_quote(&self, prompt: String) -> String {
        let Some(idx) = self.review.selected_hunk else {
            return prompt;
        };
        let hunks = self.review_hunks();
        let Some(hunk) = hunks.get(idx) else {
            return prompt;
        };
        attach_hunk_quote(
            &prompt,
            hunk,
            self.review.diff_text.as_deref().unwrap_or(""),
        )
    }

    pub(crate) fn handle_review_mouse(&mut self, mouse: &crossterm::event::MouseEvent) -> bool {
        use crossterm::event::{MouseButton, MouseEventKind};
        if !self.review.open && !self.mode.is_review() {
            return false;
        }
        let overlay = self.review_is_overlay();
        let on_splitter = !overlay && self.review_splitter_contains(mouse.column, mouse.row);
        let in_pane = overlay || crate::btw::cell_in(self.review.rect, mouse.column, mouse.row);
        match mouse.kind {
            MouseEventKind::Drag(MouseButton::Left) | MouseEventKind::Moved
                if self.review.resizing =>
            {
                self.apply_review_resize(mouse.column);
                true
            }
            MouseEventKind::Up(MouseButton::Left) if self.review.resizing => {
                self.review.resizing = false;
                true
            }
            MouseEventKind::Down(MouseButton::Left) if on_splitter => {
                self.begin_review_resize(mouse.column);
                true
            }
            MouseEventKind::ScrollUp if overlay || in_pane => {
                self.review_scroll_by(-3);
                true
            }
            MouseEventKind::ScrollDown if overlay || in_pane => {
                self.review_scroll_by(3);
                true
            }
            MouseEventKind::Down(MouseButton::Left) if in_pane => {
                if self.can_dock_review() {
                    self.review.focused = true;
                }
                if let Some(row) = self.painted_row_at(mouse.row) {
                    self.select_hunk_at_painted(row);
                }
                true
            }
            MouseEventKind::Down(MouseButton::Left)
                if self.review.open && self.can_dock_review() && self.review.focused =>
            {
                if crate::btw::cell_in(self.composer_rect, mouse.column, mouse.row)
                    || crate::btw::cell_in(self.view_inner, mouse.column, mouse.row)
                {
                    self.review.focused = false;
                    return true;
                }
                false
            }
            _ => false,
        }
    }

    fn review_splitter_contains(&self, col: u16, row: u16) -> bool {
        if crate::btw::cell_in(self.review.gap_rect, col, row) {
            return true;
        }
        let r = self.review.rect;
        r.width > 0
            && r.height > 0
            && col == r.x
            && row >= r.y
            && row < r.y.saturating_add(r.height)
    }

    fn begin_review_resize(&mut self, col: u16) {
        self.review.resizing = true;
        self.review.resize_anchor_col = col;
        self.review.resize_anchor_width = self.review.width.unwrap_or(self.review.rect.width);
    }

    fn apply_review_resize(&mut self, col: u16) {
        let dock_w = self
            .view_inner
            .width
            .saturating_add(SPLIT_GAP)
            .saturating_add(self.review.rect.width);
        if dock_w == 0 {
            return;
        }
        let delta = self.review.resize_anchor_col as i32 - col as i32;
        let want = (self.review.resize_anchor_width as i32 + delta).max(0) as u16;
        let w = clamp_pane_width(dock_w, want);
        if w > 0 {
            self.review.width = Some(w);
        }
    }

    pub(crate) fn paint_splitter(&self, frame: &mut ratatui::Frame, gap: Rect) {
        if gap.width == 0 || gap.height == 0 {
            return;
        }
        let hover = self.review_splitter_contains(self.mouse_col, self.mouse_row);
        let th = theme();
        let style = if self.review.resizing || hover {
            th.chrome(UiTone::Info).border.add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(th.gray_dim)
        };
        let lines: Vec<Line<'static>> = (0..gap.height)
            .map(|_| Line::from(ratatui::text::Span::styled("│", style)))
            .collect();
        frame.render_widget(Paragraph::new(lines), gap);
    }

    fn painted_row_at(&self, screen_row: u16) -> Option<usize> {
        let area = self.review.rect;
        if area.height <= 2 {
            return None;
        }
        let inner_y = screen_row.saturating_sub(area.y.saturating_add(1));
        let visible = area.height.saturating_sub(3) as usize;
        if inner_y as usize >= visible {
            return None;
        }
        Some(self.review.scroll.saturating_add(inner_y as usize))
    }

    pub(crate) fn render_review_overlay(
        &mut self,
        frame: &mut ratatui::Frame,
        area: ratatui::layout::Rect,
    ) {
        self.review.rect = area;
        self.paint_review(frame, area, true);
    }

    pub(crate) fn render_review_pane(
        &mut self,
        frame: &mut ratatui::Frame,
        area: ratatui::layout::Rect,
    ) {
        self.review.rect = area;
        self.paint_review(frame, area, false);
    }

    fn paint_review(&self, frame: &mut ratatui::Frame, area: ratatui::layout::Rect, overlay: bool) {
        let text = self.review.diff_text.as_deref().unwrap_or("").trim();
        let mut rendered = if text.is_empty() {
            vec![Line::styled("(no changes in the working tree)", dim())]
        } else {
            diff_lines(text)
        };
        let hunks = parse_review_hunks(text);
        if let Some(idx) = self.review.selected_hunk
            && let Some(hunk) = hunks.get(idx)
        {
            let bg = theme().selection_bg;
            for line in rendered
                .iter_mut()
                .skip(hunk.painted_start)
                .take(hunk.painted_end.saturating_sub(hunk.painted_start))
            {
                if is_hunk_gap(line) {
                    continue;
                }
                line.style = line.style.bg(bg);
                for span in &mut line.spans {
                    span.style = span.style.bg(bg);
                }
            }
        }
        let total = rendered.len();
        let visible = area.height.saturating_sub(3) as usize;
        let max_scroll = total.saturating_sub(visible);
        let scroll = self.review.scroll.min(max_scroll);
        let mut body: Vec<Line<'static>> = rendered
            .iter()
            .skip(scroll)
            .take(visible)
            .cloned()
            .collect();
        while body.len() < visible {
            body.push(Line::raw(""));
        }
        let at = hunk_at_row(&hunks, scroll)
            .and_then(|i| hunks.get(i))
            .map(ReviewHunk::label)
            .unwrap_or_else(|| "working tree".to_string());
        let footer = if overlay {
            format!(
                " j/k scroll · n/p hunks · PgUp/PgDn · G end · q/Esc close   {at}  [{}/{}]",
                scroll.saturating_add(1),
                total.max(1)
            )
        } else {
            format!(
                " drag │ to resize · click to select · {at}  [{}/{}]",
                scroll.saturating_add(1),
                total.max(1)
            )
        };
        body.push(Line::styled(footer, dim()));
        let title = if overlay {
            " Diff review (Ctrl-G) "
        } else {
            " Diff review "
        };
        let mut border = theme().chrome(UiTone::Info).border;
        if !overlay && self.review.focused {
            border = border.add_modifier(Modifier::BOLD);
        }
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(border)
            .title(title);
        frame.render_widget(Paragraph::new(body).block(block), area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_FILE: &str = "\
diff --git a/a.rs b/a.rs
--- a/a.rs
+++ b/a.rs
@@ -1,1 +1,1 @@
-old a
+new a
diff --git a/b.rs b/b.rs
--- a/b.rs
+++ b/b.rs
@@ -10,2 +10,3 @@
 ctx
-old b
+new b
";

    #[test]
    fn parse_two_file_unified_diff() {
        let hunks = parse_review_hunks(TWO_FILE);
        assert_eq!(hunks.len(), 2, "{hunks:?}");
        assert_eq!(hunks[0].path, "a.rs");
        assert_eq!(hunks[0].start, Some(1));
        assert_eq!(hunks[1].path, "b.rs");
        assert_eq!(hunks[1].start, Some(10));
        assert_eq!(hunks[1].end, Some(12));
        assert!(hunks[1].painted_start > hunks[0].painted_start);
    }

    #[test]
    fn deleted_file_uses_old_path() {
        let diff = "\
--- a/gone.rs
+++ /dev/null
@@ -1,1 +0,0 @@
-bye
";
        let hunks = parse_review_hunks(diff);
        assert_eq!(hunks[0].path, "gone.rs");
    }

    #[test]
    fn chip_replaces_previous_review_mention() {
        let next = replace_or_prepend_chip("@a.rs:1-2 please undo", "@b.rs:10-12");
        assert_eq!(next, "@b.rs:10-12 please undo");
        assert_eq!(replace_or_prepend_chip("", "@a.rs:1"), "@a.rs:1 ");
        assert_eq!(replace_or_prepend_chip("hello", "@a.rs:1"), "@a.rs:1 hello");
    }

    #[test]
    fn quote_requires_mention_and_is_capped() {
        let hunks = parse_review_hunks(TWO_FILE);
        let h = &hunks[1];
        let mention = h.mention().unwrap();
        let quoted = attach_hunk_quote(&format!("{mention} why?"), h, TWO_FILE);
        assert!(quoted.contains("<review hunk>"), "{quoted}");
        assert!(quoted.contains("b.rs:10-12"), "{quoted}");
        assert!(quoted.contains("new b"), "{quoted}");
        let dropped = attach_hunk_quote("no chip here", h, TWO_FILE);
        assert_eq!(dropped, "no chip here");
    }

    #[test]
    fn dock_only_on_wide_layout() {
        assert!(!can_dock(80));
        assert!(!can_dock(99));
        assert!(can_dock(100));
        assert!(can_dock(120));
    }

    #[test]
    fn split_body_honors_width_and_clamps() {
        let area = Rect {
            x: 0,
            y: 0,
            width: 100,
            height: 20,
        };
        let (left, Some((gap, right))) = split_body(area, true, Some(40)) else {
            panic!("expected docked split");
        };
        assert_eq!(right.width, 40);
        assert_eq!(gap.width, 1);
        assert_eq!(left.width + gap.width + right.width, 100);

        let (_, Some((_, narrow))) = split_body(area, true, Some(10)) else {
            panic!("clamp min");
        };
        assert_eq!(narrow.width, 36);

        let (_, Some((_, wide))) = split_body(area, true, Some(90)) else {
            panic!("clamp max");
        };
        assert_eq!(wide.width, 100 - 24 - 1);
    }

    #[test]
    fn docked_hints_depend_on_focus() {
        let idle = docked_session_hints(false);
        assert!(
            idle.iter()
                .any(|h| h.key == "tab" && h.label == "focus diff")
        );
        assert!(idle.iter().any(|h| h.key == "ctrl+g" && h.label == "close"));
        let focused = docked_session_hints(true);
        assert!(focused.iter().any(|h| h.key == "n/p" && h.label == "hunk"));
        assert!(
            focused
                .iter()
                .any(|h| h.key == "esc" && h.label == "unfocus")
        );
        assert!(!focused.iter().any(|h| h.key == "tab"));
    }
}
