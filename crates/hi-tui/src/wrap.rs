//! Soft-wrapping of styled transcript lines.
//!
//! Assistant prose is wrapped before it reaches the paint pass for two
//! reasons ratatui's own word wrapper cannot cover: a 200-column terminal
//! should not stretch paragraphs into unreadable measures, and list items
//! should keep a hanging indent, which needs knowledge of markdown markers.
//! Rows produced here always fit the width the paint pass wraps at, so they
//! are never split a second time.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

use crate::layout::display_width;
use crate::render::numbered_text;

/// Columns assistant prose wraps at on a wide terminal, so a 200-col window
/// does not stretch paragraphs into unreadable measures. Narrower panels wrap
/// at their own width instead; see [`prose_wrap_cols`].
pub(crate) const PROSE_WRAP_COLS: u16 = 120;

/// Fewest text columns a row keeps beside a hanging indent. Below this the
/// indent is dropped rather than wrapping one word per row.
const MIN_HANGING_TEXT_COLS: usize = 16;

/// The gutter `markdown_line` puts before fenced code and blockquote lines.
const GUTTER: &str = "▏ ";

/// The wrap width for a transcript panel `width` columns wide: the panel's
/// own width, capped at [`PROSE_WRAP_COLS`]. An unknown width (0) is the cap.
///
/// Wrapping at the panel's own width matters because the paint pass wraps at
/// that width too: rows pre-wrapped wider than the panel would be split a
/// second time, leaving alternating full and stub rows and hanging indents
/// broken across rows.
pub(crate) fn prose_wrap_cols(width: u16) -> u16 {
    if width == 0 {
        PROSE_WRAP_COLS
    } else {
        width.min(PROSE_WRAP_COLS)
    }
}

/// How a transcript row joins the row above it when rows are copied as text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RowJoin {
    /// A line of its own.
    #[default]
    NewLine,
    /// The next row of a wrapped line, broken where whitespace was dropped.
    Space,
    /// The next row of a wrapped line, inside a word too wide for a row.
    Glued,
}

/// Soft-wrap `line` to `max_cols` display columns. Rows break at whitespace,
/// which is dropped at the break; only a word wider than a whole row is split
/// mid-word. Continuation rows of list items are indented under the item
/// text and continuation rows of quoted or fenced lines repeat the `▏ `
/// gutter, so wrapped markdown keeps its shape. Rows keep the line's own
/// style and alignment. Width 0 returns the line unchanged.
pub(crate) fn wrap_line_to_width(line: &Line<'static>, max_cols: u16) -> Vec<Line<'static>> {
    wrap_line_rows(line, max_cols)
        .into_iter()
        .map(|(row, _)| row)
        .collect()
}

/// Push the rows of `line` wrapped to `max_cols` onto `rows`, and how each
/// joins the row above onto `joins`, so a copy can rejoin them.
pub(crate) fn push_wrapped(
    rows: &mut Vec<Line<'static>>,
    joins: &mut Vec<RowJoin>,
    line: &Line<'static>,
    max_cols: u16,
) {
    for (row, join) in wrap_line_rows(line, max_cols) {
        rows.push(row);
        joins.push(join);
    }
}

/// [`wrap_line_to_width`] with each row's [`RowJoin`]: the first row is a
/// new line and the rest continue it.
fn wrap_line_rows(line: &Line<'static>, max_cols: u16) -> Vec<(Line<'static>, RowJoin)> {
    if max_cols == 0 {
        return vec![(line.clone(), RowJoin::NewLine)];
    }
    let max_cols = max_cols as usize;
    if line_width(line) <= max_cols {
        return vec![(line.clone(), RowJoin::NewLine)];
    }
    let hang = continuation_prefix(line)
        .filter(|prefix| display_width(&prefix.content) + MIN_HANGING_TEXT_COLS <= max_cols);
    let cells: Vec<Cell> = line
        .spans
        .iter()
        .enumerate()
        .flat_map(|(source, span)| {
            span.content.chars().map(move |ch| Cell {
                ch,
                style: span.style,
                source,
            })
        })
        .collect();
    let mut rows = Rows::new(line, hang, max_cols);
    let mut start = 0;
    while start < cells.len() {
        let blank = cells[start].ch.is_whitespace();
        let end = cells[start..]
            .iter()
            .position(|cell| cell.ch.is_whitespace() != blank)
            .map_or(cells.len(), |len| start + len);
        if blank {
            rows.push_blanks(&cells[start..end]);
        } else {
            rows.push_word(&cells[start..end]);
        }
        start = end;
    }
    rows.finish()
}

/// Display columns of `line`.
pub(crate) fn line_width(line: &Line) -> usize {
    line.spans
        .iter()
        .map(|span| display_width(&span.content))
        .sum()
}

/// What continuation rows of `line` start with, if anything: blanks as wide
/// as the leading indent plus any list marker (`• `, `1. `, `☐ `), so wrapped
/// item text stays aligned under the first row's text; or the same `▏ `
/// gutter for quoted and fenced lines. `markdown_line` emits indent and
/// marker together as the first span, so that span is what is inspected.
fn continuation_prefix(line: &Line<'static>) -> Option<Span<'static>> {
    let first = line.spans.first()?;
    let content = first.content.as_ref();
    let marker = content.trim_start_matches(' ');
    if marker == GUTTER {
        return Some(first.clone());
    }
    let is_marker = matches!(marker, "• " | "☐ " | "☑ ")
        || numbered_text(marker).is_some_and(|(_, rest)| rest.is_empty());
    let indent = if is_marker {
        display_width(content)
    } else {
        content.len() - marker.len()
    };
    (indent > 0).then(|| Span::raw(" ".repeat(indent)))
}

/// One character of the source line and the span it came from.
struct Cell {
    ch: char,
    style: Style,
    source: usize,
}

/// The rows [`wrap_line_to_width`] is filling.
struct Rows {
    done: Vec<(Line<'static>, RowJoin)>,
    current: Vec<Span<'static>>,
    /// How `current` joins the row above it.
    join: RowJoin,
    /// Source span of the last character in `current`. Characters extend the
    /// current span only when they share a source span, so a row never merges
    /// two source spans (a `▏ ` gutter stays its own span even beside text of
    /// the same style, which `with_gutter` relies on) and never extends a
    /// hanging prefix.
    last_source: Option<usize>,
    /// Display columns `current` occupies, hanging prefix included.
    width: usize,
    /// Whether `current` holds a word yet. A row breaks only after one, so a
    /// leading indent or hanging prefix is never left alone on a row.
    has_word: bool,
    hang: Option<Span<'static>>,
    max_cols: usize,
    /// The source line without its spans: every row inherits its style and
    /// alignment (a fenced code line's panel background, for one).
    template: Line<'static>,
}

impl Rows {
    fn new(line: &Line<'static>, hang: Option<Span<'static>>, max_cols: usize) -> Self {
        let mut template = line.clone();
        template.spans.clear();
        Self {
            done: Vec::new(),
            current: Vec::new(),
            join: RowJoin::NewLine,
            last_source: None,
            width: 0,
            has_word: false,
            hang,
            max_cols,
            template,
        }
    }

    /// Whitespace where a row broke is neither carried over nor shown.
    fn push_blanks(&mut self, cells: &[Cell]) {
        if !self.done.is_empty() && !self.has_word {
            return;
        }
        let width = cells.iter().map(|cell| char_width(cell.ch)).sum();
        self.push_cells(cells, width);
    }

    fn push_word(&mut self, cells: &[Cell]) {
        let text: String = cells.iter().map(|cell| cell.ch).collect();
        let width = display_width(&text);
        if self.width + width <= self.max_cols {
            self.push_cells(cells, width);
            self.has_word = true;
            return;
        }
        let hang_width = self.hang.as_ref().map_or(0, |p| display_width(&p.content));
        if hang_width + width <= self.max_cols && self.has_word {
            self.break_row(RowJoin::Space);
            self.push_cells(cells, width);
            self.has_word = true;
            return;
        }
        // Wider than any row: fill this one and split by character.
        for (index, cell) in cells.iter().enumerate() {
            let width = char_width(cell.ch);
            if self.width + width > self.max_cols && self.has_word {
                self.break_row(if index == 0 {
                    RowJoin::Space
                } else {
                    RowJoin::Glued
                });
            }
            self.push_cells(std::slice::from_ref(cell), width);
            self.has_word = true;
        }
    }

    fn push_cells(&mut self, cells: &[Cell], width: usize) {
        for cell in cells {
            match self.current.last_mut() {
                Some(last) if self.last_source == Some(cell.source) => {
                    last.content.to_mut().push(cell.ch);
                }
                _ => {
                    self.current
                        .push(Span::styled(cell.ch.to_string(), cell.style));
                    self.last_source = Some(cell.source);
                }
            }
        }
        self.width += width;
    }

    fn break_row(&mut self, join: RowJoin) {
        self.close_row();
        self.join = join;
        self.last_source = None;
        self.width = 0;
        self.has_word = false;
        if let Some(prefix) = &self.hang {
            self.width = display_width(&prefix.content);
            self.current.push(prefix.clone());
        }
    }

    fn close_row(&mut self) {
        trim_trailing_blanks(&mut self.current);
        let mut row = self.template.clone();
        row.spans = std::mem::take(&mut self.current);
        self.done.push((row, self.join));
    }

    fn finish(mut self) -> Vec<(Line<'static>, RowJoin)> {
        // A row holding only a hanging prefix (the line ended in whitespace
        // after a break) is not a row.
        if self.has_word || self.done.is_empty() {
            self.close_row();
        }
        self.done
    }
}

fn char_width(ch: char) -> usize {
    UnicodeWidthChar::width(ch).unwrap_or(0)
}

fn trim_trailing_blanks(spans: &mut Vec<Span<'static>>) {
    while let Some(last) = spans.last_mut() {
        let trimmed = last.content.trim_end();
        if trimmed.len() == last.content.len() {
            return;
        }
        if trimmed.is_empty() {
            spans.pop();
        } else {
            last.content = trimmed.to_string().into();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{line_text, markdown_body_lines, markdown_line, wrapped_line_height};
    use ratatui::style::Modifier;

    fn md(text: &str) -> Line<'static> {
        markdown_line(text, &mut None)
    }

    fn texts(rows: &[Line<'static>]) -> Vec<String> {
        rows.iter().map(line_text).collect()
    }

    /// Every row fits, and the paint pass (ratatui's word wrapper at the
    /// same width) would leave each row as the single row it is.
    fn assert_rows_fit(rows: &[Line<'static>], max_cols: u16) {
        for row in rows {
            assert!(
                line_width(row) <= max_cols as usize,
                "{:?} is {} cols, wider than {max_cols}",
                line_text(row),
                line_width(row)
            );
            assert_eq!(
                wrapped_line_height(row, max_cols),
                1,
                "the paint pass would split {:?} again",
                line_text(row)
            );
        }
    }

    /// Words of `rows` joined by single spaces, ignoring the hanging prefix.
    fn words(rows: &[Line<'static>]) -> Vec<String> {
        texts(rows)
            .iter()
            .flat_map(|row| {
                row.split_whitespace()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn rows_break_between_words_never_inside_them() {
        let line = md(
            "- `Harness::persist_snapshot` returns `bool` but callers in `turn.rs` treat a false \
             as non-fatal; there is no surfaced error path for a failed session rewrite.",
        );
        let rows = wrap_line_to_width(&line, 60);
        assert!(rows.len() >= 3, "{:?}", texts(&rows));
        assert_rows_fit(&rows, 60);
        assert_eq!(
            words(&rows),
            words(std::slice::from_ref(&line)),
            "no word is split or lost: {:?}",
            texts(&rows)
        );
        for row in &rows {
            let text = line_text(row);
            assert!(!text.ends_with(' '), "trailing blank kept on {text:?}");
        }
    }

    #[test]
    fn a_style_that_straddles_a_break_survives_on_both_rows() {
        let line =
            md("- plain words then `a_rather_long_inline_code_span_here` and more words after");
        let rows = wrap_line_to_width(&line, 40);
        let code_rows: Vec<&Line<'static>> = rows
            .iter()
            .filter(|row| {
                row.spans.iter().any(|span| {
                    span.content.contains("inline_code") || span.content.contains("a_rather")
                })
            })
            .collect();
        assert!(!code_rows.is_empty(), "{:?}", texts(&rows));
        let code_style = line
            .spans
            .iter()
            .find(|span| span.content.contains("a_rather"))
            .map(|span| span.style)
            .unwrap();
        for row in &rows {
            for span in &row.spans {
                if span.content.contains("a_rather") || span.content.contains("code_span") {
                    assert_eq!(span.style, code_style, "{:?}", texts(&rows));
                }
            }
        }
    }

    #[test]
    fn list_continuation_rows_hang_under_the_item_text() {
        let bullet = md("- alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu");
        let rows = wrap_line_to_width(&bullet, 30);
        assert!(rows.len() >= 2, "{:?}", texts(&rows));
        assert!(line_text(&rows[0]).starts_with("• alpha"));
        for row in &rows[1..] {
            let text = line_text(row);
            assert!(
                text.starts_with("  ") && !text.starts_with("   "),
                "bullet continuation hangs two columns: {text:?}"
            );
        }
        assert_rows_fit(&rows, 30);

        let numbered = md("12. alpha beta gamma delta epsilon zeta eta theta iota kappa lambda");
        let rows = wrap_line_to_width(&numbered, 30);
        assert!(line_text(&rows[0]).starts_with("12. alpha"));
        assert!(
            line_text(&rows[1]).starts_with("    ") && !line_text(&rows[1]).starts_with("     "),
            "numbered continuation hangs under the text: {:?}",
            texts(&rows)
        );

        let nested = md("  - alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu");
        let rows = wrap_line_to_width(&nested, 30);
        assert!(line_text(&rows[0]).starts_with("  • alpha"));
        assert!(
            line_text(&rows[1]).starts_with("    ") && !line_text(&rows[1]).starts_with("     "),
            "nested bullet keeps its indent plus the marker: {:?}",
            texts(&rows)
        );

        let task = md("- [ ] alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu");
        let rows = wrap_line_to_width(&task, 30);
        assert!(line_text(&rows[0]).starts_with("☐ alpha"));
        assert!(
            line_text(&rows[1]).starts_with("  ") && !line_text(&rows[1]).starts_with("   "),
            "{:?}",
            texts(&rows)
        );
    }

    #[test]
    fn indented_prose_keeps_its_indent_on_continuation_rows() {
        let line = md("   continuation prose under a list item that runs long enough to wrap");
        let rows = wrap_line_to_width(&line, 30);
        assert!(rows.len() >= 2, "{:?}", texts(&rows));
        for row in &rows {
            let text = line_text(row);
            assert!(
                text.starts_with("   ") && !text.starts_with("    "),
                "{text:?}"
            );
        }
    }

    #[test]
    fn quotes_and_fenced_code_repeat_their_gutter() {
        let quote = md("> a quoted sentence that is long enough to need a second row here");
        let rows = wrap_line_to_width(&quote, 30);
        assert!(rows.len() >= 2, "{:?}", texts(&rows));
        for row in &rows {
            assert_eq!(row.spans[0].content.as_ref(), "▏ ", "{:?}", texts(&rows));
            assert_eq!(row.spans[0].style, quote.spans[0].style);
        }

        let mut lang = None;
        markdown_line("```rust", &mut lang);
        let code = markdown_line(
            "let value = some_function(first_argument, second_argument, third_argument);",
            &mut lang,
        );
        let rows = wrap_line_to_width(&code, 40);
        assert!(rows.len() >= 2, "{:?}", texts(&rows));
        for row in &rows {
            assert_eq!(row.spans[0].content.as_ref(), "▏ ", "{:?}", texts(&rows));
            assert_eq!(
                row.style, code.style,
                "rows keep the code block's line style"
            );
        }
        let ungutter = |lines: &[Line<'static>]| -> Vec<String> {
            lines
                .iter()
                .map(|line| line_text(line).trim_start_matches("▏ ").to_string())
                .collect::<Vec<_>>()
                .join(" ")
                .split_whitespace()
                .map(str::to_string)
                .collect()
        };
        assert_eq!(ungutter(&rows), ungutter(std::slice::from_ref(&code)));
    }

    #[test]
    fn a_word_wider_than_a_row_is_split_by_character_under_the_hang() {
        let url = format!("https://example.invalid/{}", "segment/".repeat(12));
        let line = md(&format!("- see {url} for details"));
        let rows = wrap_line_to_width(&line, 30);
        assert_rows_fit(&rows, 30);
        assert!(line_text(&rows[0]).starts_with("• see https://"));
        for row in &rows[1..] {
            assert!(line_text(row).starts_with("  "), "{:?}", texts(&rows));
        }
        let joined: String = texts(&rows)
            .iter()
            .map(|row| row.trim_start().to_string())
            .collect::<Vec<_>>()
            .join("");
        assert!(
            joined.contains(&url.replace(' ', "")),
            "the URL is intact: {joined}"
        );
    }

    #[test]
    fn rows_say_how_they_join_the_row_above() {
        let joins = |line: &Line<'static>, cols| -> Vec<RowJoin> {
            wrap_line_rows(line, cols)
                .into_iter()
                .map(|(_, join)| join)
                .collect()
        };
        assert_eq!(
            joins(&md("- alpha beta gamma delta epsilon zeta eta"), 20),
            vec![RowJoin::NewLine, RowJoin::Space, RowJoin::Space]
        );
        // The long word starts a row of its own (the space before it was
        // dropped), then is split inside itself.
        let line = Line::from(format!("{} {}", "a".repeat(19), "x".repeat(50)));
        assert_eq!(
            joins(&line, 20),
            vec![
                RowJoin::NewLine,
                RowJoin::Space,
                RowJoin::Glued,
                RowJoin::Glued
            ]
        );
        assert_eq!(joins(&md("- short"), 20), vec![RowJoin::NewLine]);
    }

    #[test]
    fn narrow_rows_drop_the_hanging_indent_rather_than_starving_the_text() {
        let line = md("1. alpha beta gamma delta epsilon zeta eta theta iota");
        let rows = wrap_line_to_width(&line, 12);
        assert_rows_fit(&rows, 12);
        assert!(
            rows[1..].iter().all(|row| !line_text(row).starts_with(' ')),
            "{:?}",
            texts(&rows)
        );
    }

    #[test]
    fn short_lines_and_width_zero_pass_through_unchanged() {
        let line = md("- short");
        assert_eq!(wrap_line_to_width(&line, 40), vec![line.clone()]);
        assert_eq!(wrap_line_to_width(&line, 0), vec![line.clone()]);
        let mut styled = Line::from("x ".repeat(50));
        styled.style = Style::default().add_modifier(Modifier::BOLD);
        assert_eq!(wrap_line_to_width(&styled, 0), vec![styled.clone()]);
    }

    #[test]
    fn a_line_ending_in_blanks_after_a_break_leaves_no_empty_row() {
        let line = Line::from("alpha beta gamma delta epsilon zeta      ");
        let rows = wrap_line_to_width(&line, 20);
        assert!(
            rows.iter().all(|row| !line_text(row).trim().is_empty()),
            "{:?}",
            texts(&rows)
        );
        assert_eq!(words(&rows), words(std::slice::from_ref(&line)));
    }

    #[test]
    fn wide_characters_count_by_display_width() {
        let line = Line::from("日本語 のテキスト は 二列 幅です さらに 続きます");
        let rows = wrap_line_to_width(&line, 12);
        assert_rows_fit(&rows, 12);
        assert_eq!(words(&rows), words(std::slice::from_ref(&line)));
    }

    #[test]
    fn prose_wrap_cols_is_the_panel_width_up_to_the_cap() {
        assert_eq!(prose_wrap_cols(80), 80);
        assert_eq!(prose_wrap_cols(200), PROSE_WRAP_COLS);
        assert_eq!(prose_wrap_cols(0), PROSE_WRAP_COLS);
    }

    #[test]
    fn assistant_messages_flatten_to_rows_that_fit_the_panel() {
        let text = "Build Next\n\n1. Add `pub async fn run_turn(&mut self, prompt: &str, ui: &mut dyn Ui)` \
                    in `crates/hi-harness/src/lib.rs` that delegates to the existing turn loop in \
                    `turn.rs`, wiring `TurnCancellation` and `interrupt`.\n\
                    - Route the existing `turn.rs` loop rather than duplicating it; `run_turn` \
                    becomes a thin wrapper that sets `turn_oneshot`, calls the internal `drive_turn`, \
                    and returns `TurnOutcome`.";
        let entry = crate::TranscriptEntry::AssistantMessage {
            text: text.to_string(),
        };
        for width in [48u16, 80, 100] {
            let rows = entry.flatten_at(width, false, false, crate::Density::Comfortable);
            assert_rows_fit(&rows, width);
            let first_numbered = rows
                .iter()
                .position(|row| line_text(row).starts_with("1. "))
                .expect("numbered item");
            assert!(
                line_text(&rows[first_numbered + 1]).starts_with("   "),
                "continuation of `1.` hangs at width {width}: {:?}",
                texts(&rows)
            );
        }
        let wide = entry.flatten_at(300, false, false, crate::Density::Comfortable);
        assert_rows_fit(&wide, PROSE_WRAP_COLS);
        assert_eq!(
            entry.flatten(false, false, crate::Density::Comfortable),
            wide,
            "the width-less flatten is the capped one"
        );
        let body = markdown_body_lines(text);
        assert_eq!(
            words(&wide),
            words(&body),
            "wrapping keeps every word: {:?}",
            texts(&wide)
        );
    }
}
