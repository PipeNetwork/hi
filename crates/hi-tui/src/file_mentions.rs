//! `@path` and `@path:N-M` mentions: parse, inject, and style as prompt chips.

use ratatui::style::{Modifier, Style};
use ratatui::text::Span;

use crate::theme::theme;

/// One `@path` token, optionally narrowed to a 1-indexed inclusive line range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileMention {
    pub path: String,
    pub start: Option<usize>,
    pub end: Option<usize>,
}

impl FileMention {
    pub(crate) fn range_label(&self) -> Option<String> {
        match (self.start, self.end) {
            (Some(a), Some(b)) if a != b => Some(format!("{a}-{b}")),
            (Some(a), _) => Some(a.to_string()),
            _ => None,
        }
    }
}

/// Split a mention token (no leading `@`) into path + optional `:N` / `:N-M`.
pub(crate) fn split_path_range(token: &str) -> (&str, Option<&str>) {
    let Some((path, rest)) = token.rsplit_once(':') else {
        return (token, None);
    };
    if path.is_empty() {
        return (token, None);
    }
    let range_ok = rest.is_empty() || rest.chars().all(|c| c.is_ascii_digit() || c == '-');
    if range_ok {
        (path, Some(rest))
    } else {
        (token, None)
    }
}

fn parse_range(rest: &str) -> (Option<usize>, Option<usize>) {
    if rest.is_empty() {
        return (None, None);
    }
    if let Some((a, b)) = rest.split_once('-') {
        return (a.parse().ok(), b.parse().ok());
    }
    (rest.parse().ok(), None)
}

/// Walk `prompt` for `@path` tokens. `@@` is a literal `@`.
pub(crate) fn parse_mentions(prompt: &str) -> Vec<FileMention> {
    let mut out = Vec::new();
    let mut chars = prompt.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '@' {
            continue;
        }
        if chars.peek() == Some(&'@') {
            chars.next();
            continue;
        }
        let mut token = String::new();
        while let Some(&pc) = chars.peek() {
            if pc.is_whitespace() {
                break;
            }
            token.push(pc);
            chars.next();
        }
        if token.is_empty() {
            continue;
        }
        let (path, range) = split_path_range(&token);
        let (start, end) = range.map(parse_range).unwrap_or((None, None));
        out.push(FileMention {
            path: path.to_string(),
            start,
            end,
        });
    }
    out
}

/// Expand `@file` mentions: append each existing workspace file (or line range)
/// under a labeled block. Original tokens stay in the user-visible text.
pub(crate) fn expand_file_mentions(prompt: &str, root: &std::path::Path) -> String {
    const MAX_FILE_BYTES: usize = 8 * 1024;
    let mut additions: Vec<String> = Vec::new();
    for mention in parse_mentions(prompt) {
        let label = match mention.range_label() {
            Some(range) => format!("{}:{range}", mention.path),
            None => mention.path.clone(),
        };
        let candidate = std::path::Path::new(&mention.path);
        let full = root.join(candidate);
        if candidate.is_absolute()
            || candidate
                .components()
                .any(|component| component == std::path::Component::ParentDir)
        {
            additions.push(format!(
                "\n\n<file mention=\"{label}\">\noutside workspace\n</file>"
            ));
            continue;
        }
        if !full.is_file() {
            additions.push(format!(
                "\n\n<file mention=\"{label}\">\nnot found\n</file>"
            ));
            continue;
        }
        let contained = root
            .canonicalize()
            .ok()
            .zip(full.canonicalize().ok())
            .is_some_and(|(root, full)| full.starts_with(root));
        if !contained {
            additions.push(format!(
                "\n\n<file mention=\"{label}\">\noutside workspace\n</file>"
            ));
            continue;
        }
        match std::fs::metadata(&full) {
            Ok(meta) if (meta.len() as usize) > MAX_FILE_BYTES => {
                additions.push(format!(
                    "\n\n<file mention=\"{label}\">\ntoo large ({} bytes; limit {})\n</file>",
                    meta.len(),
                    MAX_FILE_BYTES
                ));
            }
            Ok(_) => match std::fs::read_to_string(&full) {
                Ok(contents) => {
                    let body = slice_lines(&contents, mention.start, mention.end);
                    additions.push(format!("\n\n<file mention=\"{label}\">\n{body}\n</file>"));
                }
                Err(err) => {
                    additions.push(format!(
                        "\n\n<file mention=\"{label}\">\nread error: {err}\n</file>"
                    ));
                }
            },
            Err(err) => {
                additions.push(format!(
                    "\n\n<file mention=\"{label}\">\nread error: {err}\n</file>"
                ));
            }
        }
    }
    if additions.is_empty() {
        prompt.to_string()
    } else {
        format!("{prompt}{}", additions.join(""))
    }
}

fn slice_lines(contents: &str, start: Option<usize>, end: Option<usize>) -> String {
    let Some(from_line) = start else {
        return contents.to_string();
    };
    let lines: Vec<&str> = contents.lines().collect();
    if lines.is_empty() {
        return contents.to_string();
    }
    let from = from_line.saturating_sub(1).min(lines.len());
    let to = end.unwrap_or(from_line).max(from_line).min(lines.len());
    lines[from..to].join("\n")
}

/// Style a prompt chunk so `@path` / `@path:N-M` read as chips, not raw text.
pub(crate) fn mention_spans(chunk: &str) -> Vec<Span<'static>> {
    let th = theme();
    let dim = Style::default().fg(th.gray);
    let path_style = Style::default().fg(th.path);
    let num_style = Style::default().fg(th.gray).add_modifier(Modifier::BOLD);
    let text_style = Style::default().fg(th.text_primary);
    let mut spans = Vec::new();
    let mut rest = chunk;
    while !rest.is_empty() {
        let Some(at) = rest.find('@') else {
            spans.push(Span::styled(rest.to_string(), text_style));
            break;
        };
        if at > 0 {
            spans.push(Span::styled(rest[..at].to_string(), text_style));
        }
        let after = &rest[at + 1..];
        if after.starts_with('@') {
            spans.push(Span::styled("@@".to_string(), text_style));
            rest = &after[1..];
            continue;
        }
        let token_len = after
            .find(|c: char| c.is_whitespace())
            .unwrap_or(after.len());
        let token = &after[..token_len];
        if token.is_empty() || (at > 0 && !rest[..at].ends_with(|c: char| c.is_whitespace())) {
            spans.push(Span::styled("@".to_string(), text_style));
            rest = after;
            continue;
        }
        let (path, range) = split_path_range(token);
        spans.push(Span::styled("@".to_string(), dim));
        spans.push(Span::styled(path.to_string(), path_style));
        if let Some(range) = range.filter(|r| !r.is_empty()) {
            spans.push(Span::styled(":".to_string(), dim));
            spans.push(Span::styled(range.to_string(), num_style));
        }
        rest = &after[token_len..];
    }
    if spans.is_empty() {
        vec![Span::styled(chunk.to_string(), text_style)]
    } else {
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_path_range_reads_line_span() {
        assert_eq!(split_path_range("src/a.rs"), ("src/a.rs", None));
        assert_eq!(split_path_range("src/a.rs:40"), ("src/a.rs", Some("40")));
        assert_eq!(
            split_path_range("src/a.rs:40-80"),
            ("src/a.rs", Some("40-80"))
        );
        assert_eq!(split_path_range("src/a.rs:"), ("src/a.rs", Some("")));
    }

    #[test]
    fn parse_mentions_skips_double_at() {
        let found = parse_mentions("see @@user and @lib.rs:2");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].path, "lib.rs");
        assert_eq!(found[0].start, Some(2));
    }

    #[test]
    fn expand_file_mentions_slices_a_line_range() {
        let dir = std::env::temp_dir().join(format!("hi-tui-range-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("n.rs"), "a\nb\nc\nd\n").unwrap();
        let out = expand_file_mentions("look @n.rs:2-3", &dir);
        assert!(out.contains("<file mention=\"n.rs:2-3\">"));
        assert!(out.contains("b\nc"));
        assert!(!out.contains("\na\n") && !out.contains("d"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mention_spans_chip_two_tokens_and_skip_email() {
        let spans = mention_spans("see @a.rs and @b.rs:2-4");
        let text: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(text, "see @a.rs and @b.rs:2-4");
        assert!(
            spans.iter().any(|s| s.content == "a.rs"),
            "first path is a chip: {spans:?}"
        );
        assert!(
            spans.iter().any(|s| s.content == "b.rs"),
            "second path is a chip: {spans:?}"
        );
        assert!(
            spans.iter().any(|s| s.content == "2-4"),
            "range is a chip: {spans:?}"
        );
        let email = mention_spans("mail user@host.tld please");
        assert_eq!(
            email.iter().map(|s| s.content.as_ref()).collect::<String>(),
            "mail user@host.tld please"
        );
        assert!(
            !email.iter().any(|s| s.content == "host.tld"),
            "email must not chip: {email:?}"
        );
    }
}
