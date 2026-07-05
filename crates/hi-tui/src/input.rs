//! Terminal-free input line: text + cursor + history.

/// Terminal-free input line: text + cursor + history. Unit-tested below.
#[derive(Default)]
pub(crate) struct InputLine {
    pub chars: Vec<char>,
    pub cursor: usize,
    pub history: Vec<String>,
    pub history_pos: Option<usize>,
}

/// A Ctrl-R reverse-incremental search over input history. Filters the history
/// (most-recent-first) by case-insensitive substring; the match under the cursor
/// is loaded into the input line so Enter submits it immediately.
#[derive(Default)]
pub(crate) struct HistorySearch {
    pub query: String,
    /// Indices into `InputLine::history` (most-recent-first) matching the query.
    pub matches: Vec<usize>,
    /// Index into `matches` of the highlighted result.
    pub selected: usize,
}

impl HistorySearch {
    /// Recompute matches from `query` against `history`, newest-first.
    pub fn refilter(&mut self, history: &[String]) {
        let needle = self.query.to_lowercase();
        self.matches = history
            .iter()
            .rev()
            .enumerate()
            .filter(|(_, h)| needle.is_empty() || h.to_lowercase().contains(&needle))
            .map(|(i, _)| history.len() - 1 - i)
            .collect();
        self.selected = 0;
    }

    pub fn insert(&mut self, c: char, history: &[String]) {
        self.query.push(c);
        self.refilter(history);
    }
    pub fn backspace(&mut self, history: &[String]) {
        self.query.pop();
        self.refilter(history);
    }
    pub fn next(&mut self) {
        if self.selected + 1 < self.matches.len() {
            self.selected += 1;
        }
    }
    pub fn prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }
    /// The history index of the currently highlighted match, if any.
    pub fn current(&self) -> Option<usize> {
        self.matches.get(self.selected).copied()
    }
}

impl InputLine {
    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }
    pub fn cursor(&self) -> usize {
        self.cursor
    }
    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }
    pub fn insert(&mut self, c: char) {
        self.chars.insert(self.cursor, c);
        self.cursor += 1;
    }
    /// If the character just before the cursor is a backslash, replace it with a
    /// newline and report `true` — so a line ending in `\` continues instead of
    /// submitting (a universal fallback for terminals without Alt+Enter).
    pub fn continue_line(&mut self) -> bool {
        if self.cursor > 0 && self.chars[self.cursor - 1] == '\\' {
            self.chars[self.cursor - 1] = '\n';
            true
        } else {
            false
        }
    }
    /// Insert a (possibly multi-line) string at the cursor — used for pastes.
    /// Line endings are normalized to `\n` so the text submits as one prompt.
    pub fn insert_str(&mut self, s: &str) {
        let normalized = s.replace("\r\n", "\n").replace('\r', "\n");
        let chars: Vec<char> = normalized.chars().collect();
        let n = chars.len();
        self.chars.splice(self.cursor..self.cursor, chars);
        self.cursor += n;
        self.history_pos = None;
    }
    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.chars.remove(self.cursor - 1);
            self.cursor -= 1;
        }
    }
    pub fn kill_to_start(&mut self) {
        self.chars.drain(..self.cursor);
        self.cursor = 0;
    }
    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }
    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.chars.len());
    }
    pub fn home(&mut self) {
        self.cursor = 0;
    }
    pub fn end(&mut self) {
        self.cursor = self.chars.len();
    }
    pub fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
        self.history_pos = None;
    }
    pub fn submit(&mut self) -> String {
        let line = self.text();
        self.clear();
        if !line.trim().is_empty() && self.history.last() != Some(&line) {
            self.history.push(line.clone());
        }
        line
    }
    pub fn set(&mut self, text: &str) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
    }
    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let pos = match self.history_pos {
            Some(0) => 0,
            Some(p) => p - 1,
            None => self.history.len() - 1,
        };
        self.history_pos = Some(pos);
        self.set(&self.history[pos].clone());
    }
    pub fn history_next(&mut self) {
        match self.history_pos {
            Some(p) if p + 1 < self.history.len() => {
                self.history_pos = Some(p + 1);
                self.set(&self.history[p + 1].clone());
            }
            Some(_) => {
                self.history_pos = None;
                self.set("");
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_editing_and_history() {
        let mut input = InputLine::default();
        for c in "helo".chars() {
            input.insert(c);
        }
        input.left();
        input.insert('l');
        assert_eq!(input.text(), "hello");
        input.submit();
        for c in "two".chars() {
            input.insert(c);
        }
        input.submit();
        input.history_prev();
        assert_eq!(input.text(), "two");
        input.history_prev();
        assert_eq!(input.text(), "hello");
    }

    #[test]
    fn paste_inserts_multiline_as_one_prompt() {
        // The bug: a pasted block used to submit each line. It must instead
        // become one multi-line input that submits whole on Enter.
        let mut input = InputLine::default();
        input.insert_str("line one\nline two\nline three");
        assert_eq!(input.text(), "line one\nline two\nline three");
        assert_eq!(input.submit(), "line one\nline two\nline three");
    }

    #[test]
    fn paste_normalizes_crlf() {
        let mut input = InputLine::default();
        input.insert_str("a\r\nb\rc");
        assert_eq!(input.text(), "a\nb\nc");
    }

    #[test]
    fn slash_commands_are_cached_in_history() {
        let mut input = InputLine::default();

        // A real prompt is cached.
        input.set("fix the bug");
        input.submit();
        assert_eq!(input.history, vec!["fix the bug"]);

        // Slash commands — bare, with leading whitespace, and with args — are
        // cached like any other input.
        input.set("/help");
        input.submit();
        input.set("   /model gpt-4o");
        input.submit();
        input.set("/provider");
        input.submit();

        assert_eq!(
            input.history,
            vec!["fix the bug", "/help", "   /model gpt-4o", "/provider"],
            "slash commands should be available through input history"
        );

        // The next real prompt still appends after them.
        input.set("next real prompt");
        input.submit();
        assert_eq!(
            input.history,
            vec![
                "fix the bug",
                "/help",
                "   /model gpt-4o",
                "/provider",
                "next real prompt"
            ]
        );
    }
}
