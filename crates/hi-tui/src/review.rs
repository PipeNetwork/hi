//! Docked / overlay Changes pane (Ctrl-G): a running diff of the session.
//!
//! Claude Code's layout: a `N files changed +A -D` summary, one row per file
//! with its own counts, then every file's numbered hunks under a bold path
//! header. The pane follows the session — it refreshes as each edit lands
//! and again at turn end — so it is a live view rather than a snapshot.
//! Until the session has edited anything it shows the whole working tree.
//!
//! Wide terminals dock a right-hand pane so the composer stays usable. Narrow
//! terminals keep the exclusive full-screen overlay. Selecting a hunk (click or
//! `n/p` while focused) writes an `@path:N-M` chip; send attaches a hunk quote
//! for the model. The transcript hides that quote until thinking is expanded.

use std::hash::{Hash, Hasher};

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};

use crate::activity_feed::parse_diff_stats;
use crate::chrome::ShortcutHint;
use crate::inline_diff::{is_git_meta_line, is_hunk_gap};
use crate::layout::{UiLayout, display_width};
use crate::mode::UiMode;
use crate::render::{diff_lines, dim, hunk_start_indices, line_text};
use crate::theme::{UiTone, theme};
use crate::{session_diff_sync, working_tree_diff_sync};

const HUNK_QUOTE_CAP: usize = 30;
const REVIEW_CHIP_TAG: &str = "@";
const MIN_TRANSCRIPT: u16 = 24;
const MIN_PANE: u16 = 36;
const SPLIT_GAP: u16 = 1;
/// File rows listed under the summary before the list folds to `… N more`.
const MAX_LISTED_FILES: usize = 12;

/// Which paths the pane diffs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum ReviewScope {
    /// Everything this session edited; the working tree until the first edit.
    #[default]
    Session,
    /// An explicit path set (the last-turn `changed:` chip).
    Files(Vec<String>),
}

/// What the loaded diff actually covers, for the summary label.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ReviewSource {
    #[default]
    WorkingTree,
    Session,
    Turn,
}

impl ReviewSource {
    fn label(self) -> &'static str {
        match self {
            Self::WorkingTree => "working tree",
            Self::Session => "this session",
            Self::Turn => "last turn",
        }
    }

    fn empty_label(self) -> &'static str {
        match self {
            Self::WorkingTree => "(working tree clean)",
            Self::Session => "(no changes this session yet)",
            Self::Turn => "(no changes in the last turn)",
        }
    }
}

/// Session state for the Changes pane.
#[derive(Clone, Debug, Default)]
pub(crate) struct ReviewState {
    pub open: bool,
    pub focused: bool,
    pub scroll: usize,
    pub selected_hunk: Option<usize>,
    pub diff_text: Option<String>,
    pub scope: ReviewScope,
    pub source: ReviewSource,
    pub rect: Rect,
    pub width: Option<u16>,
    pub gap_rect: Rect,
    pub resizing: bool,
    resize_anchor_col: u16,
    resize_anchor_width: u16,
    /// Painted document for the current `diff_text` / width; rebuilt lazily.
    doc: Option<ReviewDoc>,
    doc_key: Option<(u64, u16, ReviewSource)>,
}

/// One unified-diff hunk, aligned to painted pane rows.
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum FileStatus {
    Added,
    #[default]
    Modified,
    Deleted,
}

/// One file in the pane: its counts plus where it sits in the painted rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReviewFile {
    pub path: String,
    pub status: FileStatus,
    pub additions: u32,
    pub deletions: u32,
    /// Row of this file's entry in the list under the summary, when listed.
    pub list_row: Option<usize>,
    /// Row of this file's bold path header above its hunks.
    pub section_row: usize,
}

/// The painted pane: summary, file list, then per-file sections. Hunk and
/// file positions index into `rows`.
#[derive(Clone, Debug, Default)]
pub(crate) struct ReviewDoc {
    pub rows: Vec<Line<'static>>,
    pub files: Vec<ReviewFile>,
    pub hunks: Vec<ReviewHunk>,
    pub additions: u32,
    pub deletions: u32,
}

/// One file's slice of a unified diff (or of a tool preview).
#[derive(Debug, Default)]
struct FileChunk {
    path: String,
    old_path: String,
    status: FileStatus,
    /// Whether the `---`/`+++` pair for this chunk has been consumed.
    saw_file_header: bool,
    body: String,
}

impl ReviewDoc {
    /// Paint `diff` for a pane `width` columns wide (0 when unknown; the file
    /// list then left-aligns its counts).
    pub(crate) fn build(diff: &str, source: ReviewSource, width: u16) -> Self {
        let th = theme();
        let chunks = split_file_chunks(diff);
        let mut doc = ReviewDoc::default();
        if chunks.is_empty() {
            doc.rows.push(Line::styled(source.empty_label(), dim()));
            return doc;
        }

        struct Painted {
            path: String,
            status: FileStatus,
            additions: u32,
            deletions: u32,
            rows: Vec<Line<'static>>,
            hunks: Vec<ReviewHunk>,
        }
        let mut painted: Vec<Painted> = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let rows = diff_lines(&chunk.body);
            let hunks = chunk_hunks(chunk, &rows);
            let (additions, deletions) = parse_diff_stats(&chunk.body);
            doc.additions = doc.additions.saturating_add(additions);
            doc.deletions = doc.deletions.saturating_add(deletions);
            painted.push(Painted {
                path: chunk.path.clone(),
                status: chunk.status,
                additions,
                deletions,
                rows,
                hunks,
            });
        }

        // Summary: `8 files changed  +394 -46 · this session`.
        let n = painted.len();
        let mut summary = vec![Span::styled(
            format!("{n} file{} changed", if n == 1 { "" } else { "s" }),
            Style::default()
                .fg(th.text_primary)
                .add_modifier(Modifier::BOLD),
        )];
        summary.push(Span::raw("  "));
        summary.extend(stat_spans(doc.additions, doc.deletions, true));
        summary.push(Span::styled(
            format!(" · {}", source.label()),
            Style::default().fg(th.gray_dim),
        ));
        doc.rows.push(Line::from(summary));

        // File list, counts right-aligned to the pane when the width is known.
        let listed = if n > MAX_LISTED_FILES {
            MAX_LISTED_FILES - 1
        } else {
            n
        };
        let mut list_rows: Vec<Option<usize>> = Vec::with_capacity(n);
        for file in painted.iter().take(listed) {
            list_rows.push(Some(doc.rows.len()));
            doc.rows.push(file_list_row(
                &file.path,
                file.status,
                file.additions,
                file.deletions,
                width,
            ));
        }
        if listed < n {
            doc.rows.push(Line::styled(
                format!("  … {} more files", n - listed),
                Style::default().fg(th.gray_dim),
            ));
            list_rows.resize(n, None);
        }
        doc.rows.push(Line::raw(""));

        // Per-file sections: bold path, its counts, then the numbered hunks.
        for (i, mut file) in painted.into_iter().enumerate() {
            if i > 0 {
                doc.rows.push(Line::raw(""));
            }
            let section_row = doc.rows.len();
            doc.rows.push(file_section_header(
                &file.path,
                file.status,
                file.additions,
                file.deletions,
            ));
            let offset = doc.rows.len();
            for hunk in &mut file.hunks {
                hunk.painted_start += offset;
                hunk.painted_end += offset;
            }
            doc.hunks.append(&mut file.hunks);
            if file.rows.is_empty() {
                doc.rows.push(Line::styled(
                    match file.status {
                        FileStatus::Added => "  (empty file)",
                        FileStatus::Deleted => "  (file removed)",
                        FileStatus::Modified => "  (no textual changes)",
                    },
                    dim(),
                ));
            } else {
                doc.rows.append(&mut file.rows);
            }
            doc.files.push(ReviewFile {
                path: file.path,
                status: file.status,
                additions: file.additions,
                deletions: file.deletions,
                list_row: list_rows.get(i).copied().flatten(),
                section_row,
            });
        }
        doc
    }

    /// Painted row indexes where each hunk starts.
    pub(crate) fn hunk_starts(&self) -> Vec<usize> {
        self.hunks.iter().map(|h| h.painted_start).collect()
    }

    /// Index of the listed file whose list row is `row`, if any.
    pub(crate) fn file_at_list_row(&self, row: usize) -> Option<usize> {
        self.files.iter().position(|f| f.list_row == Some(row))
    }

    /// Index of the file whose section contains `row`.
    pub(crate) fn file_at_row(&self, row: usize) -> Option<usize> {
        self.files.iter().rposition(|f| f.section_row <= row)
    }
}

fn stat_spans(additions: u32, deletions: u32, bold: bool) -> Vec<Span<'static>> {
    let th = theme();
    let mut add = Style::default().fg(th.diff_add);
    let mut del = Style::default().fg(th.diff_del);
    if bold {
        add = add.add_modifier(Modifier::BOLD);
        del = del.add_modifier(Modifier::BOLD);
    }
    vec![
        Span::styled(format!("+{additions}"), add),
        Span::raw(" "),
        Span::styled(format!("-{deletions}"), del),
    ]
}

fn status_suffix(status: FileStatus) -> &'static str {
    match status {
        FileStatus::Added => " new",
        FileStatus::Deleted => " deleted",
        FileStatus::Modified => "",
    }
}

fn file_list_row(
    path: &str,
    status: FileStatus,
    additions: u32,
    deletions: u32,
    width: u16,
) -> Line<'static> {
    let th = theme();
    let suffix = status_suffix(status);
    let stat_w = format!("+{additions} -{deletions}").len();
    let left_w = 2 + display_width(path) + display_width(suffix);
    let pad = if width == 0 {
        2
    } else {
        (width as usize).saturating_sub(left_w + stat_w).max(2)
    };
    let mut spans = vec![
        Span::raw("  "),
        Span::styled(path.to_string(), Style::default().fg(th.path)),
    ];
    if !suffix.is_empty() {
        spans.push(Span::styled(
            suffix.to_string(),
            Style::default().fg(th.gray_dim),
        ));
    }
    spans.push(Span::raw(" ".repeat(pad)));
    spans.extend(stat_spans(additions, deletions, false));
    Line::from(spans)
}

fn file_section_header(
    path: &str,
    status: FileStatus,
    additions: u32,
    deletions: u32,
) -> Line<'static> {
    let th = theme();
    let mut spans = vec![Span::styled(
        path.to_string(),
        Style::default()
            .fg(th.text_primary)
            .add_modifier(Modifier::BOLD),
    )];
    let suffix = status_suffix(status);
    if !suffix.is_empty() {
        spans.push(Span::styled(
            suffix.to_string(),
            Style::default().fg(th.gray_dim),
        ));
    }
    spans.push(Span::raw("  "));
    spans.extend(stat_spans(additions, deletions, false));
    Line::from(spans)
}

/// Split a unified diff (git or the tools' `--- p / +++ p` previews) into one
/// chunk per file. Content that arrives before any header (bare `@@` hunks or
/// compact numbered rows) becomes a single unnamed chunk.
fn split_file_chunks(diff: &str) -> Vec<FileChunk> {
    let mut chunks: Vec<FileChunk> = Vec::new();
    let mut current: Option<FileChunk> = None;
    let lines: Vec<&str> = diff.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let t = line.trim_start();
        // Never indented in git output; an indented copy is a context line.
        if let Some(rest) = line.strip_prefix("diff --git ") {
            if let Some(chunk) = current.take() {
                chunks.push(chunk);
            }
            let mut chunk = FileChunk::default();
            if let Some(b) = rest.split_whitespace().nth(1) {
                chunk.path = strip_git_prefix(b);
            }
            current = Some(chunk);
            i += 1;
            continue;
        }
        if let Some(rest) = t.strip_prefix("--- ")
            && lines
                .get(i + 1)
                .is_some_and(|next| next.trim_start().starts_with("+++ "))
        {
            let new_rest = lines[i + 1].trim_start().strip_prefix("+++ ").unwrap_or("");
            let old_path = strip_git_prefix(rest);
            let new_path = strip_git_prefix(new_rest);
            let needs_new = current.as_ref().is_none_or(|c| c.saw_file_header);
            if needs_new {
                if let Some(chunk) = current.take() {
                    chunks.push(chunk);
                }
                current = Some(FileChunk::default());
            }
            let chunk = current.as_mut().expect("chunk just ensured");
            chunk.saw_file_header = true;
            chunk.old_path = old_path.clone();
            if new_path == "/dev/null" {
                chunk.status = FileStatus::Deleted;
                if old_path != "/dev/null" {
                    chunk.path = old_path;
                }
            } else {
                if old_path == "/dev/null" {
                    chunk.status = FileStatus::Added;
                }
                chunk.path = new_path;
            }
            i += 2;
            continue;
        }
        if line.starts_with("new file mode ") {
            if let Some(chunk) = current.as_mut() {
                chunk.status = FileStatus::Added;
            }
            i += 1;
            continue;
        }
        if line.starts_with("deleted file mode ") {
            if let Some(chunk) = current.as_mut() {
                chunk.status = FileStatus::Deleted;
            }
            i += 1;
            continue;
        }
        if is_git_meta_line(line) || line.starts_with("index ") {
            i += 1;
            continue;
        }
        if current.is_none() {
            if t.is_empty() {
                i += 1;
                continue;
            }
            current = Some(FileChunk::default());
        }
        if let Some(chunk) = current.as_mut() {
            chunk.body.push_str(line);
            chunk.body.push('\n');
        }
        i += 1;
    }
    if let Some(chunk) = current.take() {
        chunks.push(chunk);
    }
    chunks
}

/// Hunks of one chunk aligned to its painted rows (relative to the chunk).
fn chunk_hunks(chunk: &FileChunk, painted: &[Line<'static>]) -> Vec<ReviewHunk> {
    let starts = hunk_start_indices(painted);
    let mut raw: Vec<ReviewHunk> = chunk
        .body
        .lines()
        .map(str::trim_start)
        .filter(|t| t.starts_with("@@"))
        .map(|t| {
            let (start, count) = new_file_span(t);
            let end = match (start, count) {
                (Some(s), Some(c)) if c > 0 => Some(s.saturating_add(c.saturating_sub(1))),
                (Some(s), _) => Some(s),
                _ => None,
            };
            ReviewHunk {
                path: chunk.path.clone(),
                start,
                end,
                painted_start: 0,
                painted_end: 0,
            }
        })
        .collect();
    if raw.is_empty() {
        return starts
            .iter()
            .enumerate()
            .map(|(i, &s)| ReviewHunk {
                path: chunk.path.clone(),
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

/// Hunks of `diff` aligned to the pane's painted rows.
pub(crate) fn parse_review_hunks(diff: &str) -> Vec<ReviewHunk> {
    ReviewDoc::build(diff, ReviewSource::WorkingTree, 0).hunks
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

/// Painted row of the next (dir=1) or previous (dir=-1) hunk start relative
/// to `from`, clamped to the painted bounds.
pub(crate) fn next_hunk_row(starts: &[usize], total: usize, from: usize, dir: i32) -> usize {
    if total == 0 {
        return from;
    }
    if starts.is_empty() {
        return from.min(total.saturating_sub(1));
    }
    if dir > 0 {
        starts
            .iter()
            .copied()
            .find(|&i| i > from)
            .unwrap_or(total.saturating_sub(1))
    } else {
        starts
            .iter()
            .copied()
            .rev()
            .find(|&i| i < from)
            .unwrap_or(0)
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

/// Split a submitted review prompt into the visible question and the attached
/// hunk. The agent still receives the full quote; the transcript hides the
/// code until thinking is expanded (Ctrl-E / Ctrl-T).
pub(crate) fn split_review_hunk_echo(prompt: &str) -> (&str, Option<&str>) {
    const OPEN: &str = "<review hunk>";
    const CLOSE: &str = "</review hunk>";
    let Some(open_at) = prompt.find(OPEN) else {
        return (prompt, None);
    };
    let visible = prompt[..open_at].trim_end();
    let mut body = &prompt[open_at + OPEN.len()..];
    if let Some(stripped) = body.strip_prefix('\n') {
        body = stripped;
    }
    if let Some(close_at) = body.find(CLOSE) {
        body = body[..close_at].trim_end();
    } else {
        body = body.trim_end();
    }
    (visible, (!body.is_empty()).then_some(body))
}

pub(crate) fn review_hunk_thinking_lines(hunk: Option<&str>) -> Vec<Line<'static>> {
    let Some(hunk) = hunk.filter(|h| !h.trim().is_empty()) else {
        return Vec::new();
    };
    let th = theme();
    std::iter::once(Line::styled(
        "  review hunk",
        Style::default().fg(th.accent_thinking),
    ))
    .chain(
        hunk.lines()
            .map(|raw| Line::styled(format!("  {raw}"), Style::default().fg(th.gray_dim))),
    )
    .collect()
}

pub(crate) fn user_prompt_entry(
    line: Line<'static>,
    at: std::time::SystemTime,
) -> crate::TranscriptEntry {
    let text = line_text(&line);
    let (visible, hunk) = split_review_hunk_echo(&text);
    let line = if hunk.is_some() && visible != text {
        let style = line
            .spans
            .first()
            .map(|span| span.style)
            .unwrap_or(line.style);
        Line::styled(visible.to_string(), style)
    } else {
        line
    };
    crate::TranscriptEntry::UserPrompt {
        line,
        at,
        review_hunk: hunk.map(str::to_string),
    }
}

/// Append the selected hunk's painted rows to `prompt` when its chip is still
/// present. `hunk` must come from the same `diff` (see [`parse_review_hunks`]).
pub(crate) fn attach_hunk_quote(prompt: &str, hunk: &ReviewHunk, diff: &str) -> String {
    let Some(mention) = hunk.mention() else {
        return prompt.to_string();
    };
    if !prompt_still_has_mention(prompt, &mention) || prompt.contains("<review hunk>") {
        return prompt.to_string();
    }
    let doc = ReviewDoc::build(diff, ReviewSource::WorkingTree, 0);
    let mut lines: Vec<String> = doc
        .rows
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

fn text_hash(text: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
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
        self.review.scope = ReviewScope::Session;
        self.review.scroll = 0;
        self.review.rect = Rect::default();
        self.review.gap_rect = Rect::default();
        self.review.resizing = false;
        self.review.doc = None;
        self.review.doc_key = None;
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

    /// Open the pane. `files = None` follows the session (working tree until
    /// the first edit); `Some` pins an explicit path set.
    pub(crate) fn open_review(&mut self, files: Option<&[String]>) {
        self.review.scope = match files {
            None => ReviewScope::Session,
            Some(paths) => ReviewScope::Files(paths.to_vec()),
        };
        self.load_review_diff();
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

    /// Reload the diff for the current scope. Session scope shows the
    /// working tree until the session has edited something.
    fn load_review_diff(&mut self) {
        let root = self.workspace_root.clone();
        let (diff, source) = match &self.review.scope {
            ReviewScope::Files(paths) => (session_diff_sync(&root, paths), ReviewSource::Turn),
            ReviewScope::Session if self.session_changed_files.is_empty() => {
                (working_tree_diff_sync(&root), ReviewSource::WorkingTree)
            }
            ReviewScope::Session => (
                session_diff_sync(&root, &self.session_changed_files),
                ReviewSource::Session,
            ),
        };
        self.review.diff_text = Some(diff);
        self.review.source = source;
    }

    /// `/diff`: open the live pane, or re-diff and focus it when it is already
    /// up. Never dumps the raw diff into the transcript.
    pub(crate) fn show_changes(&mut self) {
        if self.review.open {
            self.refresh_review_if_open();
            if self.can_dock_review() {
                self.review.focused = true;
            }
        } else {
            self.open_review(None);
        }
    }

    /// `/files`: the pane's summary and per-file counts as transcript lines,
    /// for a coder who wants the list without opening the pane.
    pub(crate) fn session_files_lines(&self) -> Vec<Line<'static>> {
        let root = self.workspace_root.clone();
        let diff = session_diff_sync(&root, &self.session_changed_files);
        let width = self.view_inner.width;
        let doc = ReviewDoc::build(diff.trim(), ReviewSource::Session, width);
        let mut lines: Vec<Line<'static>> = doc
            .rows
            .iter()
            .take_while(|row| !line_text(row).is_empty())
            .cloned()
            .collect();
        // Files edited then reverted (or committed) are still part of the
        // session story; list them so the count matches what the model did.
        let mut extra = self
            .session_changed_files
            .iter()
            .filter(|path| !doc.files.iter().any(|f| &f.path == *path))
            .peekable();
        if extra.peek().is_some() {
            lines.push(Line::styled("  no pending changes:", dim()));
            for path in extra {
                lines.push(Line::styled(format!("    {path}"), dim()));
            }
        }
        lines
    }

    /// Re-diff after an edit landed or a turn ended, keeping the reader's
    /// scroll position and hunk selection where they still exist.
    pub(crate) fn refresh_review_if_open(&mut self) {
        if !self.review.open {
            self.review.diff_text = None;
            self.review.doc = None;
            self.review.doc_key = None;
            return;
        }
        let prev_sel = self.review.selected_hunk;
        let prev_scroll = self.review.scroll;
        self.load_review_diff();
        self.review.scroll = prev_scroll;
        let n = self.review_hunks().len();
        self.review.selected_hunk = prev_sel.filter(|&i| i < n);
    }

    /// Inner width the pane last painted at (0 before the first frame).
    fn review_doc_width(&self) -> u16 {
        self.review.rect.width.saturating_sub(2)
    }

    /// The painted document for the current diff, rebuilt when the text,
    /// source, or pane width changed.
    pub(crate) fn review_doc(&mut self) -> &ReviewDoc {
        let width = self.review_doc_width();
        let text = self.review.diff_text.as_deref().unwrap_or("");
        let key = (text_hash(text), width, self.review.source);
        if self.review.doc.is_none() || self.review.doc_key != Some(key) {
            self.review.doc = Some(ReviewDoc::build(text.trim(), self.review.source, width));
            self.review.doc_key = Some(key);
        }
        self.review.doc.as_ref().expect("doc just built")
    }

    pub(crate) fn review_hunks(&mut self) -> Vec<ReviewHunk> {
        self.review_doc().hunks.clone()
    }

    pub(crate) fn review_files(&mut self) -> Vec<ReviewFile> {
        self.review_doc().files.clone()
    }

    pub(crate) fn review_scroll_by(&mut self, delta: i32) {
        if delta == crate::action::Action::REVIEW_SCROLL_END {
            let total = self.review_doc().rows.len();
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
        let (starts, total) = {
            let doc = self.review_doc();
            (doc.hunk_starts(), doc.rows.len())
        };
        self.review.scroll = next_hunk_row(&starts, total, self.review.scroll, dir);
        self.select_hunk_at_painted(self.review.scroll);
    }

    /// Scroll to a file's section (a click on its row in the list) and
    /// select its first hunk.
    pub(crate) fn review_jump_to_file(&mut self, index: usize) {
        let Some(file) = self.review_files().get(index).cloned() else {
            return;
        };
        self.review.scroll = file.section_row;
        let hunks = self.review_hunks();
        if let Some(idx) = hunks
            .iter()
            .position(|h| h.painted_start > file.section_row && h.path == file.path)
        {
            self.select_hunk_index(idx, &hunks);
            self.review.scroll = file.section_row;
        }
    }

    pub(crate) fn select_hunk_at_painted(&mut self, painted_row: usize) {
        let hunks = self.review_hunks();
        let Some(idx) = hunk_at_row(&hunks, painted_row) else {
            return;
        };
        self.select_hunk_index(idx, &hunks);
    }

    fn select_hunk_index(&mut self, idx: usize, hunks: &[ReviewHunk]) {
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

    pub(crate) fn attach_review_quote(&mut self, prompt: String) -> String {
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
            self.review.diff_text.as_deref().unwrap_or("").trim(),
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
                    if let Some(file) = self.review_doc().file_at_list_row(row) {
                        self.review_jump_to_file(file);
                    } else {
                        self.select_hunk_at_painted(row);
                    }
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
            .map(|_| Line::from(Span::styled("│", style)))
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

    fn paint_review(
        &mut self,
        frame: &mut ratatui::Frame,
        area: ratatui::layout::Rect,
        overlay: bool,
    ) {
        let selected = self.review.selected_hunk;
        let scroll_want = self.review.scroll;
        let focused = self.review.focused;
        let source = self.review.source;
        let doc = self.review_doc();
        let mut rendered = doc.rows.clone();
        let hunks = &doc.hunks;
        if let Some(idx) = selected
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
        let scroll = scroll_want.min(max_scroll);
        let mut body: Vec<Line<'static>> = rendered
            .iter()
            .skip(scroll)
            .take(visible)
            .cloned()
            .collect();
        while body.len() < visible {
            body.push(Line::raw(""));
        }
        let at = hunk_at_row(hunks, scroll)
            .and_then(|i| hunks.get(i))
            .map(ReviewHunk::label)
            .or_else(|| {
                doc.file_at_row(scroll)
                    .and_then(|i| doc.files.get(i))
                    .map(|f| f.path.clone())
            })
            .unwrap_or_else(|| source.label().to_string());
        let footer = if overlay {
            format!(
                " j/k scroll · n/p hunks · PgUp/PgDn · G end · q/Esc close   {at}  [{}/{}]",
                scroll.saturating_add(1),
                total.max(1)
            )
        } else {
            format!(
                " click a file or hunk · drag │ to resize · {at}  [{}/{}]",
                scroll.saturating_add(1),
                total.max(1)
            )
        };
        body.push(Line::styled(footer, dim()));
        let title = if overlay {
            " Changes (Ctrl-G) "
        } else {
            " Changes "
        };
        let mut border = theme().chrome(UiTone::Info).border;
        if !overlay && focused {
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
+more b
";

    fn texts(doc: &ReviewDoc) -> Vec<String> {
        doc.rows.iter().map(line_text).collect()
    }

    #[test]
    fn doc_has_summary_file_list_and_sections() {
        let doc = ReviewDoc::build(TWO_FILE, ReviewSource::Session, 40);
        let rows = texts(&doc);
        assert_eq!(rows[0], "2 files changed  +3 -2 · this session");
        assert_eq!(doc.additions, 3);
        assert_eq!(doc.deletions, 2);
        assert!(rows[1].starts_with("  a.rs"), "{rows:?}");
        assert!(rows[1].ends_with("+1 -1"), "counts right-aligned: {rows:?}");
        assert_eq!(rows[1].len(), 40, "list rows fill the pane width");
        assert!(rows[2].starts_with("  b.rs") && rows[2].ends_with("+2 -1"));
        assert_eq!(rows[3], "");
        assert_eq!(doc.files.len(), 2);
        assert_eq!(doc.files[0].list_row, Some(1));
        assert_eq!(doc.files[1].list_row, Some(2));
        assert_eq!(rows[doc.files[0].section_row], "a.rs  +1 -1");
        assert_eq!(rows[doc.files[1].section_row], "b.rs  +2 -1");
        assert!(
            rows[doc.files[1].section_row - 1].is_empty(),
            "blank row between files"
        );
        assert!(
            rows.iter().any(|r| r.contains("new a")) && rows.iter().any(|r| r.contains("more b"))
        );
        assert!(
            !rows
                .iter()
                .any(|r| r.contains("@@") || r.contains("diff --git")),
            "{rows:?}"
        );
    }

    #[test]
    fn doc_hunks_point_into_their_sections() {
        let doc = ReviewDoc::build(TWO_FILE, ReviewSource::WorkingTree, 0);
        let rows = texts(&doc);
        assert_eq!(doc.hunks.len(), 2);
        assert_eq!(doc.hunks[0].path, "a.rs");
        assert_eq!(doc.hunks[0].start, Some(1));
        assert!(rows[doc.hunks[0].painted_start].contains("old a"));
        assert_eq!(doc.hunks[1].path, "b.rs");
        assert_eq!((doc.hunks[1].start, doc.hunks[1].end), (Some(10), Some(12)));
        assert!(rows[doc.hunks[1].painted_start].contains("ctx"));
        assert!(doc.hunks[0].painted_end <= doc.files[1].section_row);
        assert_eq!(doc.file_at_list_row(1), Some(0));
        assert_eq!(doc.file_at_list_row(0), None);
        assert_eq!(doc.file_at_row(doc.hunks[1].painted_start), Some(1));
    }

    #[test]
    fn new_and_deleted_files_are_flagged() {
        let diff = "\
diff --git a/n.rs b/n.rs
new file mode 100644
--- /dev/null
+++ b/n.rs
@@ -0,0 +1,1 @@
+hello
diff --git a/gone.rs b/gone.rs
deleted file mode 100644
--- a/gone.rs
+++ /dev/null
@@ -1,1 +0,0 @@
-bye
";
        let doc = ReviewDoc::build(diff, ReviewSource::Session, 0);
        assert_eq!(doc.files[0].path, "n.rs");
        assert_eq!(doc.files[0].status, FileStatus::Added);
        assert_eq!(doc.files[1].path, "gone.rs");
        assert_eq!(doc.files[1].status, FileStatus::Deleted);
        let rows = texts(&doc);
        assert!(rows[1].contains("n.rs new"), "{rows:?}");
        assert!(rows[2].contains("gone.rs deleted"), "{rows:?}");
        assert!(
            !rows.iter().any(|r| r.contains("file mode")),
            "git metadata stays hidden: {rows:?}"
        );
    }

    #[test]
    fn tool_preview_headers_split_into_files() {
        let preview = "--- /dev/null\n+++ src/a.rs\n1 addition, 0 deletions\n   1 + a\n--- src/b.rs\n+++ src/b.rs\n1 addition, 1 deletion\n   2 - x\n   2 + y\n";
        let doc = ReviewDoc::build(preview, ReviewSource::Turn, 0);
        assert_eq!(doc.files.len(), 2);
        assert_eq!(doc.files[0].path, "src/a.rs");
        assert_eq!(doc.files[0].status, FileStatus::Added);
        assert_eq!(doc.files[1].path, "src/b.rs");
        assert_eq!((doc.additions, doc.deletions), (2, 1));
        assert_eq!(doc.hunks.len(), 2, "one synthetic hunk per compact chunk");
    }

    #[test]
    fn empty_diff_paints_the_scope_hint() {
        let doc = ReviewDoc::build("", ReviewSource::Session, 30);
        assert_eq!(texts(&doc), vec!["(no changes this session yet)"]);
        assert!(doc.files.is_empty() && doc.hunks.is_empty());
        let tree = ReviewDoc::build("   \n", ReviewSource::WorkingTree, 30);
        assert_eq!(texts(&tree), vec!["(working tree clean)"]);
    }

    #[test]
    fn long_file_lists_fold() {
        let mut diff = String::new();
        for i in 0..20 {
            diff.push_str(&format!(
                "diff --git a/f{i}.rs b/f{i}.rs\n--- a/f{i}.rs\n+++ b/f{i}.rs\n@@ -1,1 +1,1 @@\n-a\n+b\n"
            ));
        }
        let doc = ReviewDoc::build(&diff, ReviewSource::Session, 0);
        let rows = texts(&doc);
        assert_eq!(rows[0], "20 files changed  +20 -20 · this session");
        assert_eq!(rows[MAX_LISTED_FILES], "  … 9 more files");
        assert_eq!(rows[MAX_LISTED_FILES + 1], "");
        assert!(doc.files[0].list_row.is_some());
        assert!(doc.files[19].list_row.is_none());
        assert_eq!(doc.files.len(), 20, "every file still has a section");
        assert_eq!(doc.hunks.len(), 20);
    }

    #[test]
    fn next_hunk_row_walks_starts() {
        let starts = [4, 9, 15];
        assert_eq!(next_hunk_row(&starts, 20, 0, 1), 4);
        assert_eq!(next_hunk_row(&starts, 20, 4, 1), 9);
        assert_eq!(next_hunk_row(&starts, 20, 15, 1), 19);
        assert_eq!(next_hunk_row(&starts, 20, 9, -1), 4);
        assert_eq!(next_hunk_row(&starts, 20, 4, -1), 0);
        assert_eq!(next_hunk_row(&[], 20, 30, 1), 19);
        assert_eq!(next_hunk_row(&starts, 0, 3, 1), 3);
    }
}
