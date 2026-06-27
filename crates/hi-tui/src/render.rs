//! Rendering helpers for the transcript: markdown → styled [`Line`]s, plain
//! unified-diff colorization, minimal code highlighting, and text wrapping.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

pub(crate) fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

pub(crate) fn line_text(line: &Line) -> String {
    line.spans.iter().map(|s| s.content.as_ref()).collect()
}

/// The style for code — inline spans and fenced blocks.
fn code_style() -> Style {
    Style::default().fg(Color::Cyan)
}

/// Whether `s` looks like unified-diff output (a hunk header, a git header, or
/// a `---`/`+++` file-header pair) — so we can colorize plain `git diff` /
/// `diff -u` output the model runs via the shell.
pub(crate) fn looks_like_diff(s: &str) -> bool {
    let (mut minus, mut plus) = (false, false);
    for line in s.lines() {
        if line.starts_with("@@") || line.starts_with("diff --git ") {
            return true;
        }
        minus |= line.starts_with("--- ");
        plus |= line.starts_with("+++ ");
    }
    minus && plus
}

/// Render a unified diff with coloring and a new-file line-number gutter:
/// additions green, removals red, hunk headers cyan, file headers bold, context
/// muted. The line number (tracked from each `@@` header) is shown for context
/// and added lines; removed lines and headers get a blank gutter.
pub(crate) fn diff_lines(body: &str) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut new_line: Option<u32> = None;
    for line in body.lines() {
        let (style, gutter, advance) = if line.starts_with("+++") || line.starts_with("---") {
            (Style::default().add_modifier(Modifier::BOLD), None, false)
        } else if line.starts_with("@@") {
            new_line = parse_hunk_new_start(line);
            (Style::default().fg(Color::Cyan), None, false)
        } else if line.starts_with('+') {
            (Style::default().fg(Color::Green), new_line, true)
        } else if line.starts_with('-') {
            (Style::default().fg(Color::Red), None, false)
        } else {
            (dim(), new_line, true)
        };
        let num = match gutter {
            Some(n) => format!("{n:>4} "),
            None => "     ".to_string(),
        };
        out.push(Line::from(vec![
            Span::styled(num, dim()),
            Span::styled(line.to_string(), style),
        ]));
        if advance && let Some(n) = new_line.as_mut() {
            *n += 1;
        }
    }
    out
}

/// Parse the new-file start line from a unified-diff hunk header
/// `@@ -old,n +new,m @@` → `new`.
fn parse_hunk_new_start(header: &str) -> Option<u32> {
    let plus = header.split('+').nth(1)?;
    let num: String = plus.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

/// Light syntax highlighting for one line of fenced code: whole-line comments
/// (by the fence language) are dimmed and string literals are greened; the rest
/// stays in the default color. Deliberately minimal — no keyword tables — so it
/// reads as intentional on every language and never mis-colors an unknown one.
fn highlight_code(line: &str, lang: &str) -> Vec<Span<'static>> {
    if let Some(marker) = line_comment_marker(lang)
        && line.trim_start().starts_with(marker)
    {
        return vec![Span::styled(line.to_string(), dim())];
    }
    highlight_strings(line)
}

/// The line-comment marker for a fence language, if we know it. Unknown
/// languages return `None` (no comment dimming) rather than guess.
fn line_comment_marker(lang: &str) -> Option<&'static str> {
    match lang.to_lowercase().as_str() {
        "rust" | "rs" | "c" | "cpp" | "c++" | "h" | "hpp" | "js" | "javascript" | "jsx" | "ts"
        | "typescript" | "tsx" | "go" | "java" | "kotlin" | "kt" | "swift" | "scala" | "zig"
        | "dart" | "php" => Some("//"),
        "python" | "py" | "sh" | "bash" | "shell" | "zsh" | "fish" | "ruby" | "rb" | "yaml"
        | "yml" | "toml" | "ini" | "conf" | "r" | "perl" | "pl" | "makefile" | "make"
        | "dockerfile" | "elixir" | "ex" => Some("#"),
        "sql" | "lua" | "haskell" | "hs" => Some("--"),
        _ => None,
    }
}

/// Split a code line into spans, greening `"…"` / `'…'` string literals (honoring
/// `\` escapes) and leaving everything else in the default style.
fn highlight_strings(line: &str) -> Vec<Span<'static>> {
    let chars: Vec<char> = line.chars().collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut plain = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '"' || c == '\'' {
            // Find the matching close on this line, skipping escaped quotes.
            let mut j = i + 1;
            let mut closed = false;
            while j < chars.len() {
                if chars[j] == '\\' {
                    j += 2;
                    continue;
                }
                if chars[j] == c {
                    closed = true;
                    break;
                }
                j += 1;
            }
            if closed {
                if !plain.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut plain)));
                }
                let s: String = chars[i..=j].iter().collect();
                spans.push(Span::styled(s, Style::default().fg(Color::Green)));
                i = j + 1;
                continue;
            }
        }
        plain.push(c);
        i += 1;
    }
    if !plain.is_empty() {
        spans.push(Span::raw(plain));
    }
    if spans.is_empty() {
        spans.push(Span::raw(String::new()));
    }
    spans
}

/// Render one committed line of assistant markdown into a styled [`Line`].
/// Block-level constructs (headings, lists, fences, rules, quotes) are detected
/// per line; `code_lang` carries the ``` fence state across calls (`Some(lang)`
/// while inside a fence) so code interiors are highlighted for that language.
/// Anything else gets inline emphasis/code styling.
pub(crate) fn markdown_line(text: &str, code_lang: &mut Option<String>) -> Line<'static> {
    let trimmed = text.trim_start();

    // Fenced code: ``` toggles the block; the fence line becomes a dim gutter
    // (with the language as a caption when opening).
    if trimmed.starts_with("```") {
        let lang = trimmed.trim_start_matches('`').trim();
        let caption = if code_lang.is_none() { lang } else { "" };
        *code_lang = if code_lang.is_none() {
            Some(lang.to_string())
        } else {
            None
        };
        return Line::from(vec![
            Span::styled("▏ ", dim()),
            Span::styled(caption.to_string(), dim().add_modifier(Modifier::ITALIC)),
        ]);
    }
    if let Some(lang) = code_lang.as_deref() {
        let mut spans = vec![Span::styled("▏ ", dim())];
        spans.extend(highlight_code(text, lang));
        return Line::from(spans);
    }

    // Horizontal rule.
    if is_hr(trimmed) {
        return Line::styled("─".repeat(40), dim());
    }

    // Headings: # … ###### → bold, markers stripped.
    if let Some(rest) = heading_text(trimmed) {
        return Line::from(inline_spans(
            rest,
            Style::default().add_modifier(Modifier::BOLD),
        ));
    }

    // Blockquote.
    if let Some(rest) = trimmed
        .strip_prefix("> ")
        .or_else(|| trimmed.strip_prefix('>'))
    {
        let mut spans = vec![Span::styled("▏ ", dim())];
        spans.extend(inline_spans(rest, dim()));
        return Line::from(spans);
    }

    // List items keep their original indentation.
    let indent = &text[..text.len() - trimmed.len()];
    if let Some(rest) = bullet_text(trimmed) {
        let mut spans = vec![Span::raw(format!("{indent}• "))];
        spans.extend(inline_spans(rest, Style::default()));
        return Line::from(spans);
    }
    if let Some((num, rest)) = numbered_text(trimmed) {
        let mut spans = vec![Span::styled(
            format!("{indent}{num}. "),
            Style::default().add_modifier(Modifier::BOLD),
        )];
        spans.extend(inline_spans(rest, Style::default()));
        return Line::from(spans);
    }

    // Plain paragraph (keep leading whitespace) with inline formatting.
    Line::from(inline_spans(text, Style::default()))
}

/// `---`, `***`, or `___` (3+ of one char) — a horizontal rule.
fn is_hr(s: &str) -> bool {
    let s = s.trim_end();
    s.len() >= 3 && ['-', '*', '_'].iter().any(|&m| s.chars().all(|c| c == m))
}

/// Strip a leading `#`..`###### `, returning the heading text.
fn heading_text(s: &str) -> Option<&str> {
    let hashes = s.len() - s.trim_start_matches('#').len();
    if (1..=6).contains(&hashes) {
        return s[hashes..].strip_prefix(' ').map(str::trim_end);
    }
    None
}

/// Strip a leading `- `, `* `, or `+ ` bullet marker.
fn bullet_text(s: &str) -> Option<&str> {
    ['-', '*', '+']
        .iter()
        .find_map(|&m| s.strip_prefix(m)?.strip_prefix(' '))
}

/// Split a leading `N. ` / `N) ` ordered-list marker into (number, rest).
fn numbered_text(s: &str) -> Option<(&str, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit())?;
    if end == 0 {
        return None;
    }
    let rest = s[end..]
        .strip_prefix(". ")
        .or_else(|| s[end..].strip_prefix(") "))?;
    Some((&s[..end], rest))
}

/// Parse inline `**bold**`, `*italic*`/`_italic_`, and `` `code` `` into styled
/// spans over `base`. Unmatched markers fall through as literal text.
fn inline_spans(text: &str, base: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut plain = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // `code`
        if c == '`'
            && let Some(close) = find_char(&chars, i + 1, '`')
        {
            flush_plain(&mut spans, &mut plain, base);
            spans.push(Span::styled(slice(&chars, i + 1, close), code_style()));
            i = close + 1;
            continue;
        }
        // **bold**
        if c == '*'
            && chars.get(i + 1) == Some(&'*')
            && let Some(close) = find_double_star(&chars, i + 2)
        {
            flush_plain(&mut spans, &mut plain, base);
            spans.push(Span::styled(
                slice(&chars, i + 2, close),
                base.add_modifier(Modifier::BOLD),
            ));
            i = close + 2;
            continue;
        }
        // *italic* (not ** and not an empty/space-led run)
        if c == '*'
            && chars.get(i + 1) != Some(&'*')
            && chars.get(i + 1) != Some(&' ')
            && let Some(close) = find_char(&chars, i + 1, '*')
            && close > i + 1
        {
            flush_plain(&mut spans, &mut plain, base);
            spans.push(Span::styled(
                slice(&chars, i + 1, close),
                base.add_modifier(Modifier::ITALIC),
            ));
            i = close + 1;
            continue;
        }
        // _italic_ — word-boundary guarded so snake_case is left alone.
        if c == '_'
            && (i == 0 || !chars[i - 1].is_alphanumeric())
            && let Some(close) = find_char(&chars, i + 1, '_')
            && close > i + 1
            && chars.get(close + 1).is_none_or(|c| !c.is_alphanumeric())
        {
            flush_plain(&mut spans, &mut plain, base);
            spans.push(Span::styled(
                slice(&chars, i + 1, close),
                base.add_modifier(Modifier::ITALIC),
            ));
            i = close + 1;
            continue;
        }
        plain.push(c);
        i += 1;
    }
    flush_plain(&mut spans, &mut plain, base);
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }
    spans
}

fn slice(chars: &[char], from: usize, to: usize) -> String {
    chars[from..to].iter().collect()
}

fn flush_plain(spans: &mut Vec<Span<'static>>, plain: &mut String, base: Style) {
    if !plain.is_empty() {
        spans.push(Span::styled(std::mem::take(plain), base));
    }
}

fn find_char(chars: &[char], from: usize, target: char) -> Option<usize> {
    (from..chars.len()).find(|&j| chars[j] == target)
}

fn find_double_star(chars: &[char], from: usize) -> Option<usize> {
    (from..chars.len().saturating_sub(1)).find(|&j| chars[j] == '*' && chars[j + 1] == '*')
}

/// Approximate the number of terminal rows `lines` occupy when wrapped to
/// `width` — used to keep the transcript scrolled to the bottom.
///
/// Uses ratatui's own `Paragraph::line_count` (the same `WordWrapper` the
/// render path uses) so the height estimate exactly matches what the user
/// sees. A previous version counted characters (`ceil(len/width)`), which
/// undercounted whenever word-boundary wrapping produced extra rows — the
/// accumulated shortfall made `max_scroll` too small and the bottom of a
/// long message was clipped off-screen.
pub(crate) fn wrapped_height(lines: &[Line], width: u16) -> u16 {
    // Sum in usize and saturate to u16. A long transcript can exceed u16 rows, and
    // a u16 sum (or `as u16` per line) would wrap to a tiny value — zeroing
    // max_scroll and freezing scrolling. u16::MAX is also ratatui's scroll ceiling.
    let total: usize = if width == 0 {
        lines.len()
    } else {
        // `line_count` includes the block's vertical space (borders). We pass
        // the *inner* width and no block, so it returns the pure text height.
        // Each call constructs a small Paragraph — cheap relative to rendering.
        let mut sum = 0usize;
        for line in lines {
            let para = ratatui::widgets::Paragraph::new(vec![line.clone()])
                .wrap(ratatui::widgets::Wrap { trim: false });
            sum = sum.saturating_add(para.line_count(width));
        }
        sum
    };
    total.min(u16::MAX as usize) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concatenated text of a rendered line.
    fn line_text(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    /// True if any span carrying `needle` has the given modifier.
    fn span_has(line: &Line, needle: &str, m: Modifier) -> bool {
        line.spans
            .iter()
            .any(|s| s.content.contains(needle) && s.style.add_modifier.contains(m))
    }

    #[test]
    fn looks_like_diff_detects_unified_and_ignores_lists() {
        assert!(looks_like_diff("@@ -1,2 +1,2 @@\n-a\n+b"));
        assert!(looks_like_diff("--- a/x\n+++ b/x\n context"));
        assert!(looks_like_diff("diff --git a/x b/x\n..."));
        // A bullet list or a flag line must not be mistaken for a diff.
        assert!(!looks_like_diff("- one\n- two\n+ three"));
        assert!(!looks_like_diff("plain output\nno diff here"));
    }

    #[test]
    fn diff_lines_number_the_new_file() {
        let body = "--- a/x\n+++ b/x\n@@ -10,3 +10,4 @@\n ctx\n-old\n+new\n+more\n";
        let lines = diff_lines(body);
        let text: Vec<String> = lines.iter().map(line_text).collect();
        // Context line is numbered from the hunk's new-file start (10).
        assert!(
            text.iter().any(|t| t.contains("10") && t.contains("ctx")),
            "{text:?}"
        );
        // Additions continue the new-file numbering (11, 12); removals don't advance it.
        assert!(
            text.iter().any(|t| t.contains("11") && t.contains("+new")),
            "{text:?}"
        );
        assert!(
            text.iter().any(|t| t.contains("12") && t.contains("+more")),
            "{text:?}"
        );
        // The removed line carries no number (blank gutter before the '-').
        let removed = text.iter().find(|t| t.contains("-old")).unwrap();
        assert!(
            !removed.chars().any(|c| c.is_ascii_digit()),
            "removed line has no number: {removed:?}"
        );
    }

    #[test]
    fn markdown_headings_bullets_and_rules() {
        let mut code: Option<String> = None;
        let h = markdown_line("#### 5. visited reset", &mut code);
        assert_eq!(
            line_text(&h),
            "5. visited reset",
            "heading markers stripped"
        );
        assert!(span_has(&h, "visited", Modifier::BOLD), "heading is bold");

        let b = markdown_line("- Threefold repetition", &mut code);
        assert_eq!(line_text(&b), "• Threefold repetition", "bullet rewritten");

        let n = markdown_line("7. parse_move accepts", &mut code);
        assert_eq!(line_text(&n), "7. parse_move accepts", "numbered list kept");

        assert_eq!(line_text(&markdown_line("---", &mut code)), "─".repeat(40));
    }

    #[test]
    fn markdown_code_fence_renders_interior_verbatim() {
        let mut code: Option<String> = None;
        let open = markdown_line("```rust", &mut code);
        assert!(code.is_some(), "fence opens a code block");
        assert!(line_text(&open).contains("rust"), "lang caption shown");

        // Markdown markers inside a fence are NOT interpreted.
        let inner = markdown_line("visited[tr][tc] = **true**;", &mut code);
        assert!(
            line_text(&inner).contains("**true**"),
            "code interior is verbatim: {:?}",
            line_text(&inner)
        );

        markdown_line("```", &mut code);
        assert!(code.is_none(), "closing fence ends the block");
    }

    #[test]
    fn markdown_inline_emphasis_and_code() {
        let mut code: Option<String> = None;
        let line = markdown_line("Use **mut** and `Vec` not _that_", &mut code);
        assert_eq!(
            line_text(&line),
            "Use mut and Vec not that",
            "markers consumed"
        );
        assert!(span_has(&line, "mut", Modifier::BOLD), "**bold**");
        assert!(span_has(&line, "that", Modifier::ITALIC), "_italic_");
        assert!(
            line.spans
                .iter()
                .any(|s| s.content == "Vec" && s.style.fg == Some(Color::Cyan)),
            "`code` styled"
        );
        // A bare underscore in an identifier must not start italics.
        let id = markdown_line("call is_empty here", &mut code);
        assert_eq!(line_text(&id), "call is_empty here");
        assert!(
            !span_has(&id, "is_empty", Modifier::ITALIC),
            "snake_case spared"
        );
    }

    #[test]
    fn code_block_highlights_strings_and_comments() {
        let mut code: Option<String> = None;
        markdown_line("```rust", &mut code);
        // A whole-line comment is dimmed.
        let c = markdown_line("    // a note", &mut code);
        assert!(
            c.spans
                .iter()
                .any(|s| s.content.contains("// a note")
                    && s.style.add_modifier.contains(Modifier::DIM)),
            "comment dimmed"
        );
        // A string literal is greened; the rest is not.
        let s = markdown_line("let x = \"hi\";", &mut code);
        assert!(
            s.spans
                .iter()
                .any(|sp| sp.content == "\"hi\"" && sp.style.fg == Some(Color::Green)),
            "string greened: {:?}",
            s.spans
                .iter()
                .map(|sp| (sp.content.as_ref(), sp.style.fg))
                .collect::<Vec<_>>()
        );
        // Unknown language → no comment marker, so a `#`-line isn't dimmed away.
        let mut code2 = Some(String::new());
        let u = markdown_line("# this is a heading-ish line", &mut code2);
        assert!(
            !u.spans
                .iter()
                .skip(1)
                .all(|s| s.style.add_modifier.contains(Modifier::DIM)),
            "unknown lang doesn't treat # as a comment"
        );
    }

    #[test]
    fn wrapped_height_does_not_overflow_on_a_tall_transcript() {
        // A long session can exceed u16 rows. The height must saturate, not wrap
        // around to a tiny value — which would zero out max_scroll and freeze
        // scrolling (the "scrolling is broken" report on a huge session).
        let lines: Vec<Line> = (0..70_000).map(|_| Line::raw("x")).collect();
        let h = wrapped_height(&lines, 80);
        assert!(
            h >= 60_000,
            "tall transcript reports a large height, got {h}"
        );
    }
}
