//! Peel in-band CoT wrappers out of assistant `content`.
//!
//! Some models emit `<thinking>…</thinking>` or `<think>…</think>` as ordinary
//! text instead of `reasoning_content`. hi also used to serialize prior
//! thinking that way for OpenAI-compatible endpoints, and models copy the
//! format. Either way the tags must not reach the transcript as visible text:
//! the TUI already has a collapsed "thought for Ns" row for reasoning.

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Piece {
    Text(String),
    Reasoning(String),
}

#[derive(Debug, Clone, Copy)]
enum Scan {
    /// Tag at `at`, `len` bytes long.
    Found { at: usize, len: usize },
    /// Keep `buf[hold..]` — it may be an incomplete tag.
    Hold { hold: usize },
}

pub(crate) struct InlineThinkingSplitter {
    buf: String,
    in_thinking: bool,
}

impl InlineThinkingSplitter {
    pub(crate) fn new() -> Self {
        Self {
            buf: String::new(),
            in_thinking: false,
        }
    }

    pub(crate) fn push(&mut self, chunk: &str) -> Vec<Piece> {
        if chunk.is_empty() {
            return Vec::new();
        }
        self.buf.push_str(chunk);
        let mut out = Vec::new();
        while self.drain_one(&mut out) {}
        out
    }

    pub(crate) fn finish(&mut self) -> Vec<Piece> {
        let rest = std::mem::take(&mut self.buf);
        let was_thinking = self.in_thinking;
        self.in_thinking = false;
        if rest.is_empty() {
            return Vec::new();
        }
        if was_thinking {
            vec![Piece::Reasoning(trim_block(&rest).to_string())]
                .into_iter()
                .filter(|p| !piece_empty(p))
                .collect()
        } else {
            vec![Piece::Text(rest)]
        }
    }

    fn drain_one(&mut self, out: &mut Vec<Piece>) -> bool {
        if self.in_thinking {
            match find_close(&self.buf) {
                Scan::Found { at, len } => {
                    let inner = self.buf[..at].to_string();
                    self.buf = self.buf[at + len..].to_string();
                    self.in_thinking = false;
                    push_piece(out, Piece::Reasoning(trim_block(&inner).to_string()));
                    skip_leading_newline(&mut self.buf);
                    true
                }
                Scan::Hold { hold } => {
                    if hold > 0 {
                        let inner = self.buf[..hold].to_string();
                        self.buf = self.buf[hold..].to_string();
                        push_piece(out, Piece::Reasoning(inner));
                    }
                    false
                }
            }
        } else {
            match find_open(&self.buf) {
                Scan::Found { at, len } => {
                    let text = self.buf[..at].to_string();
                    self.buf = self.buf[at + len..].to_string();
                    self.in_thinking = true;
                    push_piece(out, Piece::Text(text));
                    skip_leading_newline(&mut self.buf);
                    true
                }
                Scan::Hold { hold } => {
                    if hold > 0 {
                        let text = self.buf[..hold].to_string();
                        self.buf = self.buf[hold..].to_string();
                        push_piece(out, Piece::Text(text));
                    }
                    false
                }
            }
        }
    }
}

/// Split a complete assistant string into (reasoning, visible text).
pub(crate) fn split_inline_thinking(text: &str) -> (String, String) {
    let mut splitter = InlineThinkingSplitter::new();
    let mut reasoning = String::new();
    let mut visible = String::new();
    for piece in splitter.push(text).into_iter().chain(splitter.finish()) {
        match piece {
            Piece::Reasoning(t) => {
                if !reasoning.is_empty() && !t.is_empty() && !reasoning.ends_with('\n') {
                    reasoning.push('\n');
                }
                reasoning.push_str(&t);
            }
            Piece::Text(t) => visible.push_str(&t),
        }
    }
    (
        reasoning.trim().to_string(),
        visible.trim_start_matches(['\n', '\r']).to_string(),
    )
}

fn push_piece(out: &mut Vec<Piece>, piece: Piece) {
    if !piece_empty(&piece) {
        out.push(piece);
    }
}

fn piece_empty(piece: &Piece) -> bool {
    match piece {
        Piece::Text(t) | Piece::Reasoning(t) => t.is_empty(),
    }
}

fn trim_block(s: &str) -> &str {
    s.trim_matches(['\n', '\r'])
}

fn skip_leading_newline(s: &mut String) {
    if let Some(rest) = s.strip_prefix("\r\n") {
        *s = rest.to_string();
    } else if let Some(rest) = s.strip_prefix('\n') {
        *s = rest.to_string();
    }
}

fn find_open(s: &str) -> Scan {
    find_tag(s, "<thinking>", "<think>")
}

fn find_close(s: &str) -> Scan {
    find_tag(s, "</thinking>", "</think>")
}

/// Prefer the long tag. Incomplete prefixes of the long tag are held so
/// `"<thin"` + `"king>"` still becomes `<thinking>`, and `"<think"` can still
/// become either form. A complete short tag (`<think>`) cannot become the long
/// form — they diverge at the character after `k`.
fn find_tag(s: &str, long: &str, short: &str) -> Scan {
    if let Some(at) = s.find(long) {
        return Scan::Found {
            at,
            len: long.len(),
        };
    }
    if let Some(at) = s.find(short) {
        return Scan::Found {
            at,
            len: short.len(),
        };
    }
    Scan::Hold {
        hold: hold_prefix(s, long),
    }
}

fn hold_prefix(s: &str, needle: &str) -> usize {
    let max = needle.len().saturating_sub(1);
    let start = s.len().saturating_sub(max);
    for (offset, _) in s[start..].char_indices() {
        let idx = start + offset;
        let suffix = &s[idx..];
        if !suffix.is_empty() && needle.starts_with(suffix) {
            return idx;
        }
    }
    s.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_all(chunks: &[&str]) -> (String, String) {
        let mut s = InlineThinkingSplitter::new();
        let mut reasoning = String::new();
        let mut text = String::new();
        let mut take = |pieces: Vec<Piece>| {
            for piece in pieces {
                match piece {
                    Piece::Reasoning(t) => reasoning.push_str(&t),
                    Piece::Text(t) => text.push_str(&t),
                }
            }
        };
        for chunk in chunks {
            take(s.push(chunk));
        }
        take(s.finish());
        (reasoning.trim().to_string(), text.trim().to_string())
    }

    #[test]
    fn thinking_block_is_reasoning_and_answer_is_text() {
        let (r, t) = split_inline_thinking(
            "<thinking>\nLet me read the architecture doc.\n</thinking>\nLet me look at the turn loop.",
        );
        assert_eq!(r, "Let me read the architecture doc.");
        assert_eq!(t, "Let me look at the turn loop.");
        assert!(!t.contains("<thinking>"));
        assert!(!t.contains("</thinking>"));
    }

    #[test]
    fn short_think_tags_are_stripped() {
        let (r, t) = split_inline_thinking("<think>plan</think>\nanswer");
        assert_eq!(r, "plan");
        assert_eq!(t, "answer");
    }

    #[test]
    fn fragmented_long_open_tag_does_not_leak() {
        let (r, t) = split_all(&["<thin", "king>\nplan\n</think", "ing>\nvisible"]);
        assert_eq!(r, "plan");
        assert_eq!(t, "visible");
    }

    #[test]
    fn think_is_not_confused_with_thinking() {
        let (r, t) = split_all(&["<think>\nshort\n</think>\nok"]);
        assert_eq!(r, "short");
        assert_eq!(t, "ok");
    }

    #[test]
    fn unclosed_thinking_is_reasoning_not_visible_text() {
        let (r, t) = split_all(&["<thinking>\nstill going"]);
        assert_eq!(r, "still going");
        assert!(t.is_empty());
    }

    #[test]
    fn prose_without_tags_is_unchanged() {
        let (r, t) = split_inline_thinking("just an answer");
        assert!(r.is_empty());
        assert_eq!(t, "just an answer");
    }

    #[test]
    fn less_than_in_prose_is_not_eaten() {
        let (r, t) = split_inline_thinking("use i < n as the bound");
        assert!(r.is_empty());
        assert_eq!(t, "use i < n as the bound");
    }
}
