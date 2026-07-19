//! Terminal-free input line: text + cursor + history.

/// Terminal-free input line: text + cursor + history. Unit-tested below.
#[derive(Default, Clone)]
pub(crate) struct InputLine {
    pub chars: Vec<char>,
    pub cursor: usize,
    pub history: Vec<String>,
    pub history_pos: Option<usize>,
}

/// A Ctrl-R reverse-incremental search over input history. Filters the history
/// (most-recent-first) by case-insensitive substring; the match under the cursor
/// is loaded into the input line so Enter submits it immediately.
#[derive(Clone, Debug, Default)]
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
    /// Kill from the cursor to the end of the line (Ctrl-K).
    pub fn kill_to_end(&mut self) {
        self.chars.truncate(self.cursor);
    }
    /// Move the cursor left one word (Alt-B): skip whitespace going back, then
    /// skip the non-whitespace word, landing at the start of the word.
    pub fn word_left(&mut self) {
        let mut i = self.cursor;
        // Skip trailing whitespace.
        while i > 0 && self.chars[i - 1].is_whitespace() {
            i -= 1;
        }
        // Skip the word.
        while i > 0 && !self.chars[i - 1].is_whitespace() {
            i -= 1;
        }
        self.cursor = i;
    }
    /// Move the cursor right one word (Alt-F): skip the current word, then skip
    /// whitespace, landing at the start of the next word (or end of line).
    pub fn word_right(&mut self) {
        let mut i = self.cursor;
        // Skip the current word.
        while i < self.chars.len() && !self.chars[i].is_whitespace() {
            i += 1;
        }
        // Skip whitespace.
        while i < self.chars.len() && self.chars[i].is_whitespace() {
            i += 1;
        }
        self.cursor = i;
    }
    /// Delete the word before the cursor (Ctrl-W): remove the whitespace and
    /// the preceding non-whitespace run back to the previous word start.
    /// Matches readline behavior (Ctrl-W on "foo bar |" deletes "bar ").
    pub fn delete_word_back(&mut self) {
        let mut i = self.cursor;
        // Skip trailing whitespace — it gets deleted along with the word.
        while i > 0 && self.chars[i - 1].is_whitespace() {
            i -= 1;
        }
        // Skip the word itself.
        while i > 0 && !self.chars[i - 1].is_whitespace() {
            i -= 1;
        }
        self.chars.drain(i..self.cursor);
        self.cursor = i;
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
    /// Load persistent input history from `.hi/history` under `root`, merging
    /// with any in-memory entries. One line per entry, newest-last. Capped at
    /// 1000 entries (oldest dropped). Used on startup so Ctrl-R searches across
    /// sessions, not just the current one.
    pub fn load_history(&mut self, root: &std::path::Path) {
        let path = root.join(".hi").join("history");
        let Ok(contents) = std::fs::read_to_string(&path) else {
            return;
        };
        let file_entries: Vec<String> = contents
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.to_string())
            .collect();
        if file_entries.is_empty() {
            return;
        }
        // Merge: file entries first (oldest), then in-memory (this session,
        // newest), deduplicating while preserving order.
        let mut merged: Vec<String> = file_entries;
        for entry in self.history.drain(..) {
            if !merged.contains(&entry) {
                merged.push(entry);
            }
        }
        // Cap at 1000, dropping the oldest.
        let start = merged.len().saturating_sub(1000);
        self.history = merged[start..].to_vec();
    }
    /// Save persistent input history to `.hi/history` under `root`. Creates the
    /// `.hi/` directory if needed. Writes the full history (newest-last), capped
    /// at 1000 entries. Called on submit and on shutdown.
    pub fn save_history(&self, root: &std::path::Path) {
        let dir = root.join(".hi");
        let path = dir.join("history");
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let start = self.history.len().saturating_sub(1000);
        let body = self.history[start..].join("\n");
        let _ = std::fs::write(&path, body);
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
    fn word_motions_move_by_word() {
        let mut input = InputLine::default();
        input.set("foo bar baz");
        input.cursor = 11; // end of line
        // Alt-B: back one word → cursor at "bar baz" (index 8).
        input.word_left();
        assert_eq!(input.cursor, 8, "word_left from end lands at 'baz' start");
        // Alt-B again → cursor at "foo bar baz" (index 4).
        input.word_left();
        assert_eq!(input.cursor, 4, "word_left again lands at 'bar' start");
        // Alt-F: forward one word → skips "bar " to "baz" (index 8).
        input.word_right();
        assert_eq!(input.cursor, 8, "word_right skips whitespace to next word");
    }

    #[test]
    fn delete_word_back_removes_preceding_word() {
        let mut input = InputLine::default();
        input.set("foo bar baz");
        input.cursor = 11; // end of line
        input.delete_word_back();
        assert_eq!(input.text(), "foo bar ", "Ctrl-W deletes 'baz'");
        assert_eq!(input.cursor, 8, "cursor at the deleted word's start");
        // Ctrl-W again deletes "bar " (word + trailing whitespace).
        input.delete_word_back();
        assert_eq!(input.text(), "foo ", "Ctrl-W deletes 'bar ' (word + space)");
        assert_eq!(input.cursor, 4, "cursor at 'foo ' end");
    }

    #[test]
    fn kill_to_end_truncates_at_cursor() {
        let mut input = InputLine::default();
        input.set("hello world");
        input.cursor = 5;
        input.kill_to_end();
        assert_eq!(input.text(), "hello", "Ctrl-K kills from cursor to end");
        assert_eq!(input.cursor, 5, "cursor stays put");
    }

    #[test]
    fn persistent_history_round_trips_and_merges() {
        let dir = tempfile_dir();
        let history_path = dir.join(".hi").join("history");

        // Save some history.
        let mut input = InputLine::default();
        input.set("git status");
        input.submit();
        input.set("cargo build");
        input.submit();
        input.save_history(&dir);
        assert!(history_path.exists(), "history file written");
        let contents = std::fs::read_to_string(&history_path).unwrap();
        assert!(
            contents.contains("git status") && contents.contains("cargo build"),
            "saved history contains both entries: {contents}"
        );

        // Load into a fresh InputLine — should pick up the file entries.
        let mut loaded = InputLine::default();
        loaded.load_history(&dir);
        assert_eq!(
            loaded.history,
            vec!["git status".to_string(), "cargo build".to_string()],
            "loaded history matches saved order"
        );

        // Merge: in-memory entries (this session) append after file entries,
        // deduplicated.
        loaded.set("cargo build"); // duplicate — should not appear twice
        loaded.submit();
        loaded.set("cargo test"); // new
        loaded.submit();
        let mut merged = loaded.clone();
        merged.load_history(&dir);
        assert_eq!(
            merged.history,
            vec![
                "git status".to_string(),
                "cargo build".to_string(),
                "cargo test".to_string(),
            ],
            "merge dedupes and appends new in-memory entries"
        );
    }

    /// Create a unique temp dir for history tests. Cleaned up when dropped.
    fn tempfile_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hi-tui-history-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
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
