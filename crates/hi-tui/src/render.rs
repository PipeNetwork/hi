//! Rendering helpers for the transcript: markdown → styled [`Line`]s, plain
//! unified-diff colorization, minimal code highlighting, and text wrapping.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme::theme;

pub(crate) fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

pub(crate) fn line_text(line: &Line) -> String {
    line.spans.iter().map(|s| s.content.as_ref()).collect()
}

/// A left accent-gutter span (`┃ `) in a role color — the block-accent bar that
/// marks agent "machinery" lines (tool calls, status, errors) as distinct from
/// the user's prompts and the assistant's prose (which stay flush-left).
pub(crate) fn gutter(color: Color) -> Span<'static> {
    Span::styled("┃ ", Style::default().fg(color))
}

/// Build an accent-gutter line: the role-colored `┃ ` bar followed by `content`
/// styled with `content_style`.
pub(crate) fn accent_line(
    color: Color,
    content: impl Into<String>,
    content_style: Style,
) -> Line<'static> {
    Line::from(vec![
        gutter(color),
        Span::styled(content.into(), content_style),
    ])
}

// --- Live-activity animation: the running accent wave + waiting pulse. ---
//
// While a turn runs, the accent bars of the live status stream ripple with a
// bright crest that travels down the rows (grok-build's "the block is alive"
// signal), and the lead glyph breathes dim→bright while the model is thinking.
// Both are driven by the per-redraw `spinner` tick, so they animate at the
// event loop's cadence with no extra timers.

/// Ticks for the wave crest to sweep once from the top row to the bottom and
/// wrap back to the top. At the ~120ms tick cadence this is a ~1.7s sweep.
const WAVE_PERIOD: usize = 14;
/// The crest's half-width in rows — how many rows around the crest still glow.
const WAVE_SPREAD: f32 = 2.5;
/// Ticks for one dim→bright→dim breath of the waiting pulse (~1.4s).
const PULSE_PERIOD: usize = 12;

/// Brightness weight in `0.0..=1.0` for `row` of a `rows`-tall active region at
/// animation `phase`, forming a crest that travels down and wraps. The falloff
/// is squared so the crest reads as a soft band rather than a hard line.
pub(crate) fn wave_weight(phase: usize, row: usize, rows: usize) -> f32 {
    if rows == 0 {
        return 0.0;
    }
    let period = WAVE_PERIOD.max(1);
    let rows_f = rows as f32;
    // Crest head sweeps 0..rows as phase advances through one period.
    let head = (phase % period) as f32 / period as f32 * rows_f;
    let raw = (row as f32 - head).abs();
    // Circular distance so the crest re-enters at the top as it leaves the bottom.
    let dist = raw.min(rows_f - raw);
    let w = 1.0 - dist / WAVE_SPREAD;
    if w <= 0.0 { 0.0 } else { w * w }
}

/// Brightness weight in `0.0..=1.0` for the waiting pulse: a smooth
/// dim→bright→dim breath (`sin²`) that repeats every [`PULSE_PERIOD`] ticks.
pub(crate) fn pulse_weight(phase: usize) -> f32 {
    let t = (phase % PULSE_PERIOD.max(1)) as f32 / PULSE_PERIOD.max(1) as f32;
    let s = (std::f32::consts::PI * t).sin();
    s * s
}

/// Blend `base`→`crest` by `t` in `0.0..=1.0`. Truecolor endpoints interpolate
/// channel-wise; named/ANSI colors (no channels to mix) snap to `crest` past the
/// halfway point so 16-color terminals still show the motion as an on/off crest.
pub(crate) fn lerp_color(base: Color, crest: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    match (base, crest) {
        (Color::Rgb(r0, g0, b0), Color::Rgb(r1, g1, b1)) => {
            Color::Rgb(lerp_u8(r0, r1, t), lerp_u8(g0, g1, t), lerp_u8(b0, b1, t))
        }
        _ => {
            if t >= 0.5 {
                crest
            } else {
                base
            }
        }
    }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// The accent color for `row` of a `rows`-tall live-activity region at
/// `phase` — [`base`] with a [`crest`]-colored wave rippling through it.
pub(crate) fn wave_color(
    base: Color,
    crest: Color,
    phase: usize,
    row: usize,
    rows: usize,
) -> Color {
    lerp_color(base, crest, wave_weight(phase, row, rows))
}

/// The waiting-pulse color at `phase`: [`base`] breathing toward [`crest`] and
/// back once per [`PULSE_PERIOD`]. Used for the lead glyph while the model is
/// thinking (no tool is running to drive the wave).
pub(crate) fn pulse_color(base: Color, crest: Color, phase: usize) -> Color {
    lerp_color(base, crest, pulse_weight(phase))
}

/// How long the finish flash takes to fade, in milliseconds.
pub(crate) const FLASH_MS: u128 = 450;

/// Brightness weight in `0.0..=1.0` for the finish flash `elapsed_ms` after a
/// turn completes: full at the moment of completion, fading linearly to zero at
/// [`FLASH_MS`] and staying there — so a settled status line shows no flash.
pub(crate) fn flash_weight(elapsed_ms: u128) -> f32 {
    if elapsed_ms >= FLASH_MS {
        return 0.0;
    }
    1.0 - elapsed_ms as f32 / FLASH_MS as f32
}

/// The style for code — inline spans and fenced blocks.
fn code_style() -> Style {
    Style::default().fg(theme().code)
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
            (Style::default().fg(theme().diff_hunk), None, false)
        } else if line.starts_with('+') {
            (Style::default().fg(theme().diff_add), new_line, true)
        } else if line.starts_with('-') {
            (Style::default().fg(theme().diff_del), None, false)
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

/// The broad language family a fence belongs to, for keyword/type tables and
/// comment syntax. `Other` gets string+number+comment highlighting but no
/// keyword table, so an unknown language never mis-colors ordinary words.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lang {
    Rust,
    Python,
    JsTs,
    Go,
    C,
    Other,
}

fn lang_category(lang: &str) -> Lang {
    match lang.to_lowercase().as_str() {
        "rust" | "rs" => Lang::Rust,
        "python" | "py" => Lang::Python,
        "js" | "javascript" | "jsx" | "ts" | "typescript" | "tsx" => Lang::JsTs,
        "go" | "golang" => Lang::Go,
        "c" | "cpp" | "c++" | "h" | "hpp" | "java" | "kotlin" | "kt" | "swift" | "scala"
        | "zig" | "dart" | "php" | "cs" | "csharp" => Lang::C,
        _ => Lang::Other,
    }
}

/// The line-comment marker for a fence language, if we know it.
fn line_comment_marker(lang: &str) -> Option<&'static str> {
    match lang.to_lowercase().as_str() {
        "rust" | "rs" | "c" | "cpp" | "c++" | "h" | "hpp" | "js" | "javascript" | "jsx" | "ts"
        | "typescript" | "tsx" | "go" | "golang" | "java" | "kotlin" | "kt" | "swift" | "scala"
        | "zig" | "dart" | "php" | "cs" | "csharp" => Some("//"),
        "python" | "py" | "sh" | "bash" | "shell" | "zsh" | "fish" | "ruby" | "rb" | "yaml"
        | "yml" | "toml" | "ini" | "conf" | "r" | "perl" | "pl" | "makefile" | "make"
        | "dockerfile" | "elixir" | "ex" => Some("#"),
        "sql" | "lua" | "haskell" | "hs" => Some("--"),
        _ => None,
    }
}

/// Control/declaration keywords per language family.
fn keywords(lang: Lang) -> &'static [&'static str] {
    match lang {
        Lang::Rust => &[
            "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
            "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod",
            "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super",
            "trait", "true", "type", "unsafe", "use", "where", "while",
        ],
        Lang::Python => &[
            "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
            "elif", "else", "except", "False", "finally", "for", "from", "global", "if", "import",
            "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return",
            "True", "try", "while", "with", "yield",
        ],
        Lang::JsTs => &[
            "as",
            "async",
            "await",
            "break",
            "case",
            "catch",
            "class",
            "const",
            "continue",
            "default",
            "delete",
            "do",
            "else",
            "enum",
            "export",
            "extends",
            "false",
            "finally",
            "for",
            "from",
            "function",
            "if",
            "import",
            "in",
            "instanceof",
            "interface",
            "let",
            "new",
            "null",
            "of",
            "return",
            "super",
            "switch",
            "this",
            "throw",
            "true",
            "try",
            "type",
            "typeof",
            "undefined",
            "var",
            "void",
            "while",
            "yield",
        ],
        Lang::Go => &[
            "break",
            "case",
            "chan",
            "const",
            "continue",
            "default",
            "defer",
            "else",
            "fallthrough",
            "false",
            "for",
            "func",
            "go",
            "goto",
            "if",
            "import",
            "interface",
            "map",
            "nil",
            "package",
            "range",
            "return",
            "select",
            "struct",
            "switch",
            "true",
            "type",
            "var",
        ],
        Lang::C => &[
            "auto",
            "break",
            "case",
            "char",
            "class",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "false",
            "float",
            "for",
            "if",
            "import",
            "int",
            "long",
            "new",
            "null",
            "public",
            "private",
            "protected",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "struct",
            "switch",
            "this",
            "true",
            "typedef",
            "union",
            "unsigned",
            "void",
            "while",
        ],
        Lang::Other => &[],
    }
}

/// Known primitive/builtin type names, colored as types even when lowercase.
fn primitives(lang: Lang) -> &'static [&'static str] {
    match lang {
        Lang::Rust => &[
            "bool", "char", "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "str",
            "String", "u8", "u16", "u32", "u64", "u128", "usize", "Vec", "Option", "Result", "Box",
        ],
        Lang::Python => &[
            "bool", "bytes", "dict", "float", "int", "list", "set", "str", "tuple",
        ],
        Lang::JsTs => &[
            "any", "bigint", "boolean", "never", "number", "object", "string", "symbol", "unknown",
            "void",
        ],
        Lang::Go => &[
            "bool",
            "byte",
            "complex64",
            "complex128",
            "error",
            "float32",
            "float64",
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            "rune",
            "string",
            "uint",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "uintptr",
        ],
        Lang::C => &[
            "bool", "char", "double", "float", "int", "long", "short", "size_t", "void",
        ],
        Lang::Other => &[],
    }
}

/// Syntax-highlight one line of fenced code into styled spans: keywords, types,
/// function calls, string/char literals (with escapes), numbers, and comments
/// take theme syntax colors; everything else stays in the default text color.
/// Line-at-a-time — a block comment `/* … */` is colored to its close on the
/// same line (or to end-of-line), but doesn't carry across lines.
fn highlight_code(line: &str, lang: &str) -> Vec<Span<'static>> {
    let th = theme();
    let cat = lang_category(lang);
    let line_comment = line_comment_marker(lang);
    let block_comment = matches!(cat, Lang::Rust | Lang::JsTs | Lang::Go | Lang::C);
    let chars: Vec<char> = line.chars().collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut plain = String::new();
    let flush = |plain: &mut String, spans: &mut Vec<Span<'static>>| {
        if !plain.is_empty() {
            spans.push(Span::styled(
                std::mem::take(plain),
                Style::default().fg(th.text_primary),
            ));
        }
    };
    let starts_with = |i: usize, m: &str| chars[i..].iter().collect::<String>().starts_with(m);

    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // Line comment → rest of line.
        if let Some(marker) = line_comment
            && starts_with(i, marker)
        {
            flush(&mut plain, &mut spans);
            spans.push(Span::styled(
                chars[i..].iter().collect::<String>(),
                Style::default().fg(th.syn_comment),
            ));
            break;
        }
        // Block comment `/* … */` (single line): color to close or end.
        if block_comment && starts_with(i, "/*") {
            flush(&mut plain, &mut spans);
            let mut j = i + 2;
            while j + 1 < chars.len() && !(chars[j] == '*' && chars[j + 1] == '/') {
                j += 1;
            }
            let end = if j + 1 < chars.len() {
                j + 2
            } else {
                chars.len()
            };
            spans.push(Span::styled(
                chars[i..end].iter().collect::<String>(),
                Style::default().fg(th.syn_comment),
            ));
            i = end;
            continue;
        }
        // String / char literal, honoring `\` escapes.
        if c == '"' || c == '\'' || (c == '`' && cat == Lang::JsTs) {
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
                flush(&mut plain, &mut spans);
                spans.push(Span::styled(
                    chars[i..=j].iter().collect::<String>(),
                    Style::default().fg(th.syn_string),
                ));
                i = j + 1;
                continue;
            }
        }
        // Number literal (not an identifier suffix like the 2 in `x2`).
        let prev_ident = i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '_');
        if c.is_ascii_digit() && !prev_ident {
            flush(&mut plain, &mut spans);
            let mut j = i;
            while j < chars.len()
                && (chars[j].is_ascii_alphanumeric() || chars[j] == '.' || chars[j] == '_')
            {
                j += 1;
            }
            spans.push(Span::styled(
                chars[i..j].iter().collect::<String>(),
                Style::default().fg(th.syn_number),
            ));
            i = j;
            continue;
        }
        // Identifier → keyword / type / function / plain.
        if c.is_alphabetic() || c == '_' {
            let mut j = i;
            while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '_') {
                j += 1;
            }
            let ident: String = chars[i..j].iter().collect();
            let next = chars.get(j).copied();
            let style = if keywords(cat).contains(&ident.as_str()) {
                Some(th.syn_keyword)
            } else if primitives(cat).contains(&ident.as_str()) || is_pascal_case(&ident) {
                Some(th.syn_type)
            } else if next == Some('(') {
                Some(th.syn_function)
            } else {
                None
            };
            if let Some(color) = style {
                flush(&mut plain, &mut spans);
                spans.push(Span::styled(ident, Style::default().fg(color)));
            } else {
                plain.push_str(&ident);
            }
            i = j;
            continue;
        }
        plain.push(c);
        i += 1;
    }
    flush(&mut plain, &mut spans);
    if spans.is_empty() {
        spans.push(Span::styled(
            String::new(),
            Style::default().fg(th.text_primary),
        ));
    }
    spans
}

/// A PascalCase identifier (leading uppercase + at least one lowercase) — the
/// naming convention for types across most languages. All-caps constants are
/// deliberately excluded (they're not types).
fn is_pascal_case(ident: &str) -> bool {
    let mut chars = ident.chars();
    chars.next().is_some_and(|c| c.is_uppercase()) && ident.chars().any(|c| c.is_lowercase())
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
    // Task-list checkbox — must be tried before the plain bullet, which would
    // otherwise consume the `- ` and leave `[ ] text`.
    if let Some((checked, rest)) = task_item(trimmed) {
        let (glyph, gstyle) = if checked {
            ("☑ ", Style::default().fg(theme().accent_success))
        } else {
            ("☐ ", dim())
        };
        let mut spans = vec![Span::styled(format!("{indent}{glyph}"), gstyle)];
        spans.extend(inline_spans(rest, Style::default()));
        return Line::from(spans);
    }
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

    // Pipe tables: the `|---|:--:|` separator becomes a ruled divider; data and
    // header rows get muted `│` cell separators. Rendering is line-local, so
    // columns align only as well as the source padded them.
    if is_table_separator(trimmed) {
        return render_table_separator(trimmed);
    }
    if is_table_row(trimmed) {
        return render_table_row(trimmed);
    }

    // Plain paragraph (keep leading whitespace) with inline formatting.
    Line::from(inline_spans(text, Style::default()))
}

/// `- [ ] rest` / `- [x] rest` (also `*`/`+` bullets) → (checked, rest).
fn task_item(s: &str) -> Option<(bool, &str)> {
    let after_bullet = ['-', '*', '+']
        .iter()
        .find_map(|&m| s.strip_prefix(m)?.strip_prefix(' '))?;
    let inner = after_bullet.strip_prefix('[')?;
    let mark = inner.chars().next()?;
    let rest = inner.strip_prefix(mark)?.strip_prefix("] ")?;
    match mark {
        ' ' => Some((false, rest)),
        'x' | 'X' => Some((true, rest)),
        _ => None,
    }
}

/// The interior cells of a `| a | b |` row (outer pipes stripped).
fn table_cells(s: &str) -> Vec<&str> {
    s.trim()
        .trim_start_matches('|')
        .trim_end_matches('|')
        .split('|')
        .collect()
}

/// A `| … | … |` row — starts and ends with `|` and has at least two pipes.
fn is_table_row(s: &str) -> bool {
    let s = s.trim();
    s.starts_with('|') && s.ends_with('|') && s.matches('|').count() >= 2
}

/// A table's `|---|:--:|---:|` separator: every cell is dashes with optional
/// alignment colons.
fn is_table_separator(s: &str) -> bool {
    is_table_row(s)
        && table_cells(s).iter().all(|c| {
            let c = c.trim();
            c.contains('-') && c.chars().all(|ch| matches!(ch, '-' | ':'))
        })
}

fn render_table_row(s: &str) -> Line<'static> {
    let cells = table_cells(s);
    let mut spans = vec![Span::styled("│ ", dim())];
    for (k, cell) in cells.iter().enumerate() {
        if k > 0 {
            spans.push(Span::styled(" │ ", dim()));
        }
        spans.extend(inline_spans(cell.trim(), Style::default()));
    }
    spans.push(Span::styled(" │", dim()));
    Line::from(spans)
}

fn render_table_separator(s: &str) -> Line<'static> {
    let cells = table_cells(s);
    let mut out = String::from("├");
    for (k, cell) in cells.iter().enumerate() {
        if k > 0 {
            out.push('┼');
        }
        let w = cell.trim().len().max(3) + 2;
        out.push_str(&"─".repeat(w));
    }
    out.push('┤');
    Line::styled(out, dim())
}

/// Whether `s` is a pipe-table row (data, header, or the `|---|` separator) —
/// used by the streaming committer to accumulate a whole table before rendering
/// it aligned.
pub(crate) fn is_table_line(s: &str) -> bool {
    is_table_row(s)
}

/// A column's text alignment, from the separator's `:` markers.
#[derive(Clone, Copy)]
enum Align {
    Left,
    Center,
    Right,
}

fn parse_align(cell: &str) -> Align {
    let c = cell.trim();
    match (c.starts_with(':'), c.ends_with(':')) {
        (true, true) => Align::Center,
        (false, true) => Align::Right,
        _ => Align::Left,
    }
}

fn pad_cell(s: &str, width: usize, align: Align) -> String {
    let len = s.chars().count();
    if len >= width {
        return s.to_string();
    }
    let pad = width - len;
    match align {
        Align::Left => format!("{s}{}", " ".repeat(pad)),
        Align::Right => format!("{}{s}", " ".repeat(pad)),
        Align::Center => {
            let left = pad / 2;
            format!("{}{s}{}", " ".repeat(left), " ".repeat(pad - left))
        }
    }
}

/// Render a whole pipe table (its source rows) with columns aligned to the widest
/// cell in each column: header rows bold, the `|---|` row a ruled divider, cells
/// padded per the separator's alignment markers. Called once the streaming
/// committer has the full table, so widths span every row.
pub(crate) fn render_table(rows: &[String]) -> Vec<Line<'static>> {
    let mut grid: Vec<Vec<String>> = Vec::new();
    let mut sep_at: Option<usize> = None;
    let mut aligns: Vec<Align> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        if is_table_separator(r) {
            sep_at = Some(i);
            aligns = table_cells(r).iter().map(|c| parse_align(c)).collect();
        }
        grid.push(
            table_cells(r)
                .iter()
                .map(|c| c.trim().to_string())
                .collect(),
        );
    }
    let ncols = grid
        .iter()
        .map(|r| r.len())
        .max()
        .unwrap_or(0)
        .max(aligns.len());
    if ncols == 0 {
        return rows.iter().map(|r| Line::raw(r.clone())).collect();
    }
    while aligns.len() < ncols {
        aligns.push(Align::Left);
    }
    let mut widths = vec![0usize; ncols];
    for (i, row) in grid.iter().enumerate() {
        if Some(i) == sep_at {
            continue;
        }
        for (c, cell) in row.iter().enumerate() {
            widths[c] = widths[c].max(cell.chars().count());
        }
    }
    let mut out = Vec::with_capacity(rows.len());
    for (i, row) in grid.iter().enumerate() {
        if Some(i) == sep_at {
            let mut s = String::from("├");
            for (c, w) in widths.iter().enumerate() {
                if c > 0 {
                    s.push('┼');
                }
                s.push_str(&"─".repeat(w + 2));
            }
            s.push('┤');
            out.push(Line::styled(s, dim()));
        } else {
            let header = sep_at.is_some_and(|s| i < s);
            let base = if header {
                Style::default().add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let mut spans = vec![Span::styled("│", dim())];
            for (c, w) in widths.iter().enumerate() {
                let cell = row.get(c).map(String::as_str).unwrap_or("");
                spans.push(Span::styled(
                    format!(" {} ", pad_cell(cell, *w, aligns[c])),
                    base,
                ));
                spans.push(Span::styled("│", dim()));
            }
            out.push(Line::from(spans));
        }
    }
    out
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
    let link_style = base.fg(theme().link).add_modifier(Modifier::UNDERLINED);
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
        // [label](url): the label is the link text; the URL trails dimmed unless
        // it's the same string the label already shows.
        if c == '['
            && let Some(rbrack) = find_char(&chars, i + 1, ']')
            && chars.get(rbrack + 1) == Some(&'(')
            && let Some(rparen) = find_char(&chars, rbrack + 2, ')')
        {
            flush_plain(&mut spans, &mut plain, base);
            let label = slice(&chars, i + 1, rbrack);
            let url = slice(&chars, rbrack + 2, rparen);
            spans.push(Span::styled(label.clone(), link_style));
            if url != label {
                spans.push(Span::styled(format!(" ({url})"), dim()));
            }
            i = rparen + 1;
            continue;
        }
        // Bare autolink: an `http(s)://…` run, underlined. Trailing sentence
        // punctuation is left outside the link.
        if (c == 'h') && (matches_at(&chars, i, "http://") || matches_at(&chars, i, "https://")) {
            let mut j = i;
            while j < chars.len()
                && !chars[j].is_whitespace()
                && !matches!(chars[j], ')' | ']' | '"' | '<' | '>' | '`')
            {
                j += 1;
            }
            while j > i && matches!(chars[j - 1], '.' | ',' | ';' | ':' | '!' | '?') {
                j -= 1;
            }
            flush_plain(&mut spans, &mut plain, base);
            spans.push(Span::styled(slice(&chars, i, j), link_style));
            i = j;
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

/// Whether `chars[i..]` begins with `pat` (char-wise, no allocation).
fn matches_at(chars: &[char], i: usize, pat: &str) -> bool {
    pat.chars()
        .enumerate()
        .all(|(k, pc)| chars.get(i + k) == Some(&pc))
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
/// Wrapped row count for a single line at `width` (the same `WordWrapper` the
/// render path uses). `width == 0` → 1 row. Used to locate sticky-header
/// positions without re-measuring the whole transcript.
pub(crate) fn wrapped_line_height(line: &Line, width: u16) -> u16 {
    if width == 0 {
        return 1;
    }
    ratatui::widgets::Paragraph::new(vec![line.clone()])
        .wrap(ratatui::widgets::Wrap { trim: false })
        .line_count(width)
        .min(u16::MAX as usize) as u16
}

/// Total wrapped height of `lines` at `width`, saturating at `u16::MAX`. The
/// render path now sums [`wrapped_line_height`] inline (it also needs each
/// prefix offset for sticky headers); this stays for the overflow-saturation
/// regression test, which pins that a very tall transcript can't wrap the sum
/// back to a tiny value and freeze scrolling.
#[cfg(test)]
pub(crate) fn wrapped_height(lines: &[Line], width: u16) -> u16 {
    let mut sum = 0u32;
    for line in lines {
        sum = sum.saturating_add(wrapped_line_height(line, width) as u32);
    }
    sum.min(u16::MAX as u32) as u16
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
                .any(|s| s.content == "Vec" && s.style.fg == Some(crate::theme::theme().code)),
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
    fn markdown_links_are_underlined_and_show_url() {
        let th = crate::theme::theme();
        let mut code: Option<String> = None;
        let line = markdown_line("see [the docs](https://ex.com/x) now", &mut code);
        // Label carries the link color + underline; the URL trails, dimmed.
        assert!(
            line.spans.iter().any(|s| s.content == "the docs"
                && s.style.fg == Some(th.link)
                && s.style.add_modifier.contains(Modifier::UNDERLINED)),
            "label is an underlined link: {:?}",
            line.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<Vec<_>>()
        );
        assert!(
            line.spans
                .iter()
                .any(|s| s.content.contains("https://ex.com/x")),
            "url shown alongside label"
        );

        // A bare URL autolinks (trailing sentence punctuation stays outside it).
        let bare = markdown_line("go to https://ex.com/y.", &mut code);
        assert!(
            bare.spans.iter().any(|s| s.content == "https://ex.com/y"
                && s.style.add_modifier.contains(Modifier::UNDERLINED)),
            "bare url underlined without the trailing period: {:?}",
            bare.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn markdown_task_list_checkboxes() {
        let th = crate::theme::theme();
        let mut code: Option<String> = None;
        let done = markdown_line("- [x] ship it", &mut code);
        assert_eq!(line_text(&done), "☑ ship it", "checked box glyph");
        assert!(
            done.spans
                .iter()
                .any(|s| s.content.contains('☑') && s.style.fg == Some(th.accent_success)),
            "checked box is success-colored"
        );
        let todo = markdown_line("- [ ] later", &mut code);
        assert_eq!(line_text(&todo), "☐ later", "unchecked box glyph");
        // A normal bullet is untouched by the task-list path.
        let plain = markdown_line("- not a task", &mut code);
        assert_eq!(line_text(&plain), "• not a task");
    }

    #[test]
    fn markdown_pipe_table_rows_and_separator() {
        let mut code: Option<String> = None;
        let header = markdown_line("| Name | Score |", &mut code);
        assert_eq!(
            line_text(&header),
            "│ Name │ Score │",
            "cells split by bars"
        );
        // The separator row becomes a ruled divider, not a data row.
        let sep = markdown_line("|------|:-----:|", &mut code);
        let sep_text = line_text(&sep);
        assert!(
            sep_text.starts_with('├') && sep_text.contains('┼') && sep_text.ends_with('┤'),
            "separator ruled: {sep_text:?}"
        );
        assert!(!sep_text.contains('-'), "separator has no leftover dashes");
        // Prose containing a pipe is not mistaken for a table.
        let prose = markdown_line("run a | b to pipe", &mut code);
        assert_eq!(line_text(&prose), "run a | b to pipe");
    }

    #[test]
    fn render_table_aligns_columns() {
        let rows = vec![
            "| Name | Score |".to_string(),
            "|------|------:|".to_string(),
            "| Alice | 10 |".to_string(),
            "| Bob | 5 |".to_string(),
        ];
        let lines = render_table(&rows);
        let texts: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(texts.len(), 4, "header, rule, two data rows: {texts:?}");
        // Every non-separator row is padded to the same width — the whole point.
        let widths: Vec<usize> = texts
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 1)
            .map(|(_, t)| t.chars().count())
            .collect();
        assert!(
            widths.windows(2).all(|w| w[0] == w[1]),
            "rows aligned to equal width: {texts:?}"
        );
        // The header row is bold.
        assert!(
            lines[0].spans.iter().any(
                |s| s.content.contains("Name") && s.style.add_modifier.contains(Modifier::BOLD)
            ),
            "header bold"
        );
        // Row two is a ruled divider spanning the table.
        assert!(
            texts[1].starts_with('├') && texts[1].contains('┼') && texts[1].ends_with('┤'),
            "ruled separator: {:?}",
            texts[1]
        );
        // The right-aligned Score column pads on the left, so "5" sits under "10".
        assert!(
            texts[3].contains("    5 "),
            "right-aligned cell: {:?}",
            texts[3]
        );
    }

    #[test]
    fn code_block_highlights_keywords_strings_comments_and_calls() {
        let th = crate::theme::theme();
        let fg = |line: &Line, needle: &str| -> Option<ratatui::style::Color> {
            line.spans
                .iter()
                .find(|s| s.content == needle)
                .and_then(|s| s.style.fg)
        };
        let mut code: Option<String> = None;
        markdown_line("```rust", &mut code);

        // A comment takes the comment color (to end of line).
        let c = markdown_line("    // a note", &mut code);
        assert!(
            c.spans
                .iter()
                .any(|s| s.content.contains("// a note") && s.style.fg == Some(th.syn_comment)),
            "comment colored: {:?}",
            c.spans
                .iter()
                .map(|s| (s.content.as_ref(), s.style.fg))
                .collect::<Vec<_>>()
        );

        // Keyword, type, string, number, and a function call each get their slot.
        let s = markdown_line("let x: Vec<u8> = parse(\"hi\", 42);", &mut code);
        let got: Vec<(&str, Option<ratatui::style::Color>)> = s
            .spans
            .iter()
            .map(|sp| (sp.content.as_ref(), sp.style.fg))
            .collect();
        assert_eq!(fg(&s, "let"), Some(th.syn_keyword), "keyword: {got:?}");
        assert_eq!(fg(&s, "Vec"), Some(th.syn_type), "type: {got:?}");
        assert_eq!(fg(&s, "u8"), Some(th.syn_type), "primitive type: {got:?}");
        assert_eq!(fg(&s, "parse"), Some(th.syn_function), "call: {got:?}");
        assert_eq!(fg(&s, "\"hi\""), Some(th.syn_string), "string: {got:?}");
        assert_eq!(fg(&s, "42"), Some(th.syn_number), "number: {got:?}");

        // A snake_case identifier that's not a keyword/type/call stays plain
        // (coalesced with the surrounding plain run).
        let p = markdown_line("total_count + 1", &mut code);
        let plain = p
            .spans
            .iter()
            .find(|s| s.content.contains("total_count"))
            .expect("plain run present");
        assert_eq!(
            plain.style.fg,
            Some(th.text_primary),
            "plain ident colored text_primary"
        );

        // Unknown language → no comment marker or keyword table; a `#`-line isn't
        // swallowed as a comment.
        let mut code2 = Some(String::new());
        let u = markdown_line("# this is a heading-ish line", &mut code2);
        assert!(
            !u.spans
                .iter()
                .skip(1)
                .any(|s| s.style.fg == Some(th.syn_comment)),
            "unknown lang doesn't treat # as a comment"
        );
    }

    #[test]
    fn code_block_python_and_block_comments() {
        let th = crate::theme::theme();
        let fg = |line: &Line, needle: &str| -> Option<ratatui::style::Color> {
            line.spans
                .iter()
                .find(|s| s.content == needle)
                .and_then(|s| s.style.fg)
        };

        // Python uses `#` for line comments and `def`/`return` keywords.
        let mut py: Option<String> = None;
        markdown_line("```python", &mut py);
        let d = markdown_line("def load(path):  # read it", &mut py);
        assert_eq!(fg(&d, "def"), Some(th.syn_keyword), "python keyword");
        assert_eq!(fg(&d, "load"), Some(th.syn_function), "python call");
        assert!(
            d.spans
                .iter()
                .any(|s| s.content.contains("# read it") && s.style.fg == Some(th.syn_comment)),
            "python line comment"
        );

        // A `/* … */` block comment on one line is colored to its close, and code
        // after it resumes normal highlighting.
        let mut rs: Option<String> = None;
        markdown_line("```rust", &mut rs);
        let b = markdown_line("let a = /* note */ 3;", &mut rs);
        assert!(
            b.spans
                .iter()
                .any(|s| s.content == "/* note */" && s.style.fg == Some(th.syn_comment)),
            "block comment colored: {:?}",
            b.spans
                .iter()
                .map(|s| (s.content.as_ref(), s.style.fg))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            fg(&b, "let"),
            Some(th.syn_keyword),
            "keyword before block comment"
        );
        assert_eq!(
            fg(&b, "3"),
            Some(th.syn_number),
            "number after block comment"
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

    #[test]
    fn wave_weight_is_bounded_and_crest_travels_down() {
        let rows = 8;
        // Bounded to [0, 1] for every row and a spread of phases.
        for phase in 0..40 {
            for row in 0..rows {
                let w = wave_weight(phase, row, rows);
                assert!((0.0..=1.0).contains(&w), "w={w} phase={phase} row={row}");
            }
        }
        // The brightest row advances (crest moves down) as phase increases across
        // the first half of a period.
        let brightest = |phase: usize| -> usize {
            (0..rows)
                .max_by(|&a, &b| {
                    wave_weight(phase, a, rows)
                        .partial_cmp(&wave_weight(phase, b, rows))
                        .unwrap()
                })
                .unwrap()
        };
        assert!(brightest(0) <= brightest(3), "crest should move downward");
        assert!(brightest(3) < brightest(6), "crest keeps descending");
        // An empty region is a no-op (no divide-by-zero, no crest).
        assert_eq!(wave_weight(5, 0, 0), 0.0);
    }

    #[test]
    fn pulse_weight_breathes_between_zero_and_one() {
        // Starts dim, is bounded, and peaks near the middle of its period.
        assert!(pulse_weight(0).abs() < 1e-6, "starts dim");
        for phase in 0..30 {
            let w = pulse_weight(phase);
            assert!((0.0..=1.0).contains(&w), "w={w} phase={phase}");
        }
        // The mid-period sample is the brightest of the period.
        let peak = (0..PULSE_PERIOD).map(pulse_weight).fold(0.0f32, f32::max);
        assert!(peak > 0.9, "reaches near-full brightness, peak={peak}");
    }

    #[test]
    fn flash_weight_fades_from_full_to_zero() {
        assert!((flash_weight(0) - 1.0).abs() < 1e-6, "full at completion");
        let mid = flash_weight(FLASH_MS / 2);
        assert!(
            (mid - 0.5).abs() < 0.05,
            "roughly half-faded at the midpoint: {mid}"
        );
        assert_eq!(flash_weight(FLASH_MS), 0.0, "gone at the window edge");
        assert_eq!(flash_weight(FLASH_MS + 5_000), 0.0, "stays gone afterward");
    }

    #[test]
    fn lerp_color_interpolates_rgb_and_thresholds_named() {
        // Truecolor endpoints mix channel-wise; the midpoint is the average.
        let mid = lerp_color(Color::Rgb(0, 0, 0), Color::Rgb(100, 200, 40), 0.5);
        assert_eq!(mid, Color::Rgb(50, 100, 20));
        assert_eq!(
            lerp_color(Color::Rgb(10, 20, 30), Color::Rgb(90, 90, 90), 0.0),
            Color::Rgb(10, 20, 30)
        );
        // Out-of-range t clamps.
        assert_eq!(
            lerp_color(Color::Rgb(0, 0, 0), Color::Rgb(80, 80, 80), 5.0),
            Color::Rgb(80, 80, 80)
        );
        // Named/ANSI colors have no channels → snap to crest past halfway.
        assert_eq!(lerp_color(Color::Gray, Color::Cyan, 0.2), Color::Gray);
        assert_eq!(lerp_color(Color::Gray, Color::Cyan, 0.8), Color::Cyan);
    }
}
