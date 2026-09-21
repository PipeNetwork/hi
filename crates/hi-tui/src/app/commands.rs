//! `App` methods: commands.

#[cfg(test)]
use ansi_to_tui::IntoText;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::Style;
use ratatui::text::Line;
#[cfg(test)]
use ratatui::text::Text;
use std::time::Instant;

use crate::action::Action;
use crate::render::dim;
use crate::util::{copy_to_clipboard, read_clipboard, read_primary};

impl crate::App {
    /// Apply a pure editing/navigation key to the input line, shared by the
    /// idle input phase and the in-turn queue-entry path. Returns the submitted
    /// text on Enter (when non-empty); the caller decides whether to run it now
    /// or queue it. Phase-specific control keys (Ctrl-C/Esc) are handled by the
    /// caller, not here.
    pub(crate) fn edit_key(&mut self, key: &KeyEvent) -> Option<String> {
        let history_search_was_active = self.mode.is_history_search();
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        // Grok Ctrl+E is global. Handle it before history-search / block-nav
        // swallow the chord, and accept every encoding terminals actually send.
        if crate::keys::is_toggle_reasoning_key(key) {
            self.apply_action(Action::ToggleReasoning);
            return None;
        }
        if crate::keys::is_paste_key(key) {
            self.paste_from_clipboard();
            return None;
        }
        // --- Ctrl-R reverse history search mode ---
        // When active, keystrokes go to the search filter, not the input line.
        if self.mode.is_history_search() {
            let mut search = match std::mem::replace(&mut self.mode, crate::mode::UiMode::Insert) {
                crate::mode::UiMode::HistorySearch(s) => s,
                other => {
                    self.mode = other;
                    return None;
                }
            };
            let restore = |app: &mut Self, search: crate::input::HistorySearch| {
                app.mode = crate::mode::UiMode::HistorySearch(search);
            };
            match key.code {
                KeyCode::Enter => {
                    let idx = search.current();
                    if let Some(i) = idx
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                        let line = self.input.submit();
                        if !line.trim().is_empty() {
                            self.input.save_history_file(&self.input_history_path);
                            return Some(line);
                        }
                    }
                    return None;
                }
                KeyCode::Esc => {
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                KeyCode::Char('r') if ctrl => {
                    search.next();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    restore(self, search);
                    return None;
                }
                KeyCode::Char('s') if ctrl => {
                    search.prev();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    restore(self, search);
                    return None;
                }
                KeyCode::Backspace => {
                    search.backspace(&self.input.history);
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    restore(self, search);
                    return None;
                }
                KeyCode::Up => {
                    search.prev();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    restore(self, search);
                    return None;
                }
                KeyCode::Down => {
                    search.next();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    restore(self, search);
                    return None;
                }
                KeyCode::Char(c) if !ctrl => {
                    search.insert(c, &self.input.history);
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    restore(self, search);
                    return None;
                }
                _ => {
                    restore(self, search);
                    return None;
                }
            }
        }
        // --- Block-navigation mode (Ctrl-B) ---
        // A cursor over tool-output blocks; keys drive the cursor and folding
        // rather than the input line. Any block count change is handled by the
        // clamp in `selected_block_ord`.
        if self.mode.is_block_nav() {
            match key.code {
                KeyCode::Esc => self.mode.to_insert(),
                KeyCode::Char('b') if ctrl => self.mode.to_insert(),
                KeyCode::Up | KeyCode::Char('k') => {
                    self.block_cursor = self.selected_block_ord().saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    let n = self.tool_block_count();
                    if n > 0 {
                        self.block_cursor = (self.selected_block_ord() + 1).min(n - 1);
                    }
                }
                KeyCode::Enter => self.toggle_selected_block(),
                KeyCode::Char(' ') => self.focus_prompt(),
                _ => {}
            }
            return None;
        }
        // Queue edit chords (also available via Action dispatch).
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        if alt && !self.queue.is_empty() {
            match key.code {
                KeyCode::Up if shift => {
                    self.queue_move_selected(-1);
                    return None;
                }
                KeyCode::Down if shift => {
                    self.queue_move_selected(1);
                    return None;
                }
                KeyCode::Up => {
                    self.queue_select_prev();
                    return None;
                }
                KeyCode::Down => {
                    self.queue_select_next();
                    return None;
                }
                KeyCode::Backspace => {
                    let _ = self.queue_remove_selected();
                    return None;
                }
                _ => {}
            }
        }
        if self.completion.is_some() {
            match key.code {
                KeyCode::Up => {
                    self.completion_move(-1);
                    return None;
                }
                KeyCode::Down => {
                    self.completion_move(1);
                    return None;
                }
                KeyCode::Esc => {
                    self.completion = None;
                    return None;
                }
                KeyCode::Tab => {
                    let _ = self.accept_completion(false);
                    self.sync_completion();
                    return None;
                }
                KeyCode::Enter => {
                    if let Some(line) = self.accept_completion(true)
                        && !line.trim().is_empty()
                    {
                        return Some(self.attach_review_quote(line));
                    }
                    self.sync_completion();
                    return None;
                }
                _ => {}
            }
        }
        match key.code {
            // Shift+Enter and Alt+Enter insert a newline (matching common prompt
            // editors); a trailing backslash is the fallback for terminals that
            // cannot distinguish modified Enter.
            KeyCode::Enter if alt || shift => self.input.insert('\n'),
            KeyCode::Enter if self.input.continue_line() => {}
            KeyCode::Enter => {
                let line = self.input.submit();
                if !line.trim().is_empty() {
                    self.input.save_history_file(&self.input_history_path);
                    return Some(self.attach_review_quote(line));
                }
                if let Some(hint) = self
                    .suggested_prompt
                    .as_ref()
                    .filter(|_| !self.suggested_prompt_dismissed && self.input.is_empty())
                    .cloned()
                {
                    self.suggested_prompt = None;
                    return Some(hint);
                }
            }
            KeyCode::Char('u') if ctrl => self.input.kill_to_start(),
            KeyCode::Char('a') if ctrl => self.input.home(),
            // Readline word motions: Alt-B/F move by word, Ctrl-W deletes the
            // word before the cursor, Ctrl-K kills to end of line.
            KeyCode::Char('b') if alt => self.input.word_left(),
            KeyCode::Char('f') if alt => self.input.word_right(),
            KeyCode::Char('w') if ctrl => self.input.delete_word_back(),
            KeyCode::Char('k') if ctrl => self.input.kill_to_end(),
            // Diff review (Ctrl-D aliases Ctrl-G): docked pane when wide,
            // exclusive overlay when narrow.
            KeyCode::Char('d') if ctrl => {
                self.toggle_review();
            }
            KeyCode::Char('g') if ctrl => {
                self.toggle_review();
            }
            // Toggle the agent-observability panel (Ctrl-? = Ctrl-Shift-/).
            // Shows the last turn's trajectory telemetry, tool-call count, and
            // context composition — read-only diagnostics for the agent's own
            // behavior.
            KeyCode::Char('?') if ctrl => {
                self.show_debug = !self.show_debug;
            }
            // Toggle full tool-output expansion: long blocks fold to a preview
            // by default; Ctrl-O reveals every block's full body (and back).
            KeyCode::Char('o') if ctrl => {
                self.show_tool_output = !self.show_tool_output;
                self.bump_transcript();
            }
            // Copy the assistant's most recent fenced code block to the
            // clipboard — the most-copied artifact in a coding session, now
            // one keystroke instead of a mouse drag.
            KeyCode::Char('y') if ctrl => {
                self.copy_last_code_block();
            }
            // Enter block-navigation mode: a cursor over tool-output blocks so a
            // single block can be folded/unfolded (Enter) while the rest stay as
            // they were. Starts on the most recent block; no-op if there are none.
            KeyCode::Char('b') if ctrl => {
                let n = self.tool_block_count();
                if n > 0 {
                    self.mode = crate::mode::UiMode::BlockNav;
                    self.block_cursor = n - 1;
                }
            }
            // External editor hand-off (Ctrl-X): dump the current draft into
            // `$VISUAL`/`$EDITOR` (fallback `vi`), suspend the TUI, and read
            // the result back on save. Useful for long, structured prompts even
            // though the built-in composer supports wrapped multiline editing.
            KeyCode::Char('x') if ctrl => {
                self.edit_in_external_editor();
            }
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            // `?` on an empty input line toggles a keybindings help overlay;
            // when there's text, it's a normal character.
            KeyCode::Char('?') if !ctrl && self.input.is_empty() => {
                self.show_help = !self.show_help;
            }
            KeyCode::Char(c) if !ctrl => {
                // Shrink-on-type: keep the suggestion so a matching prefix still
                // shows the remaining ghost; divergent text just hides it.
                self.input.insert(c);
            }
            KeyCode::Backspace => {
                self.input.backspace();
            }
            KeyCode::Left => self.input.left(),
            KeyCode::Right => {
                if self.input.cursor() == self.input.chars.len() && self.ghost_suffix().is_some() {
                    let _ = self.accept_suggested_prompt();
                } else {
                    self.input.right();
                }
            }
            KeyCode::Up => {
                self.clear_suggested_prompt();
                self.input.history_prev();
            }
            KeyCode::Down => {
                self.clear_suggested_prompt();
                self.input.history_next();
            }
            KeyCode::PageUp => self.scroll_up(5),
            KeyCode::PageDown => self.scroll_down(5),
            _ => {}
        }
        self.sync_completion_after_edit_key(key, history_search_was_active);
        None
    }

    /// Insert clipboard/bracketed-paste text into the composer (grok Ctrl+V).
    pub(crate) fn paste_into_prompt(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.input.insert_str(text);
        self.sync_completion();
    }

    pub(crate) fn paste_from_clipboard(&mut self) {
        if let Ok(text) = read_clipboard() {
            self.paste_into_prompt(&text);
        }
    }

    pub(crate) fn paste_from_primary(&mut self) {
        if let Ok(text) = read_primary() {
            self.paste_into_prompt(&text);
        }
    }

    /// Number of tool-output blocks in the transcript (the foldable blocks that
    /// block-nav steps over).
    pub(crate) fn tool_block_count(&self) -> usize {
        self.transcript.iter().filter(|e| e.is_foldable()).count()
    }

    /// The block cursor clamped to the current block count (blocks can be
    /// capped away between keypresses). Zero when there are no blocks.
    pub(crate) fn selected_block_ord(&self) -> usize {
        self.block_cursor
            .min(self.tool_block_count().saturating_sub(1))
    }

    /// Flip the expand state of the block the cursor is on.
    pub(crate) fn toggle_selected_block(&mut self) {
        self.toggle_block_ord(self.selected_block_ord());
    }

    /// Flip the expand state of the `target`-th foldable block.
    pub(crate) fn toggle_block_ord(&mut self, target: usize) {
        let mut ord = 0;
        for entry in self.transcript.iter_mut() {
            if !entry.is_foldable() {
                continue;
            }
            if ord == target {
                if let Some(expanded) = entry.expanded_mut() {
                    *expanded = !*expanded;
                    self.bump_transcript();
                }
                break;
            }
            ord += 1;
        }
    }

    /// Run a `!cmd` shell-escape: execute `command` read-only in the workspace
    /// root and push its combined stdout/stderr into the transcript as a
    /// foldable tool-output block. This is a quick local command (e.g. `!git
    /// status`, `!ls -la`) that never involves the model — it saves a whole
    /// agent turn for trivial state checks. Output is capped so a runaway
    /// command can't flood the transcript. (The live TUI uses the async
    /// `run_shell_escape_async` in run.rs; this sync version is kept for tests.)
    #[cfg(test)]
    pub(crate) fn run_shell_escape(&mut self, command: &str) {
        use ratatui::text::Line as RLine;
        let command = command.trim();
        if command.is_empty() {
            return;
        }
        // Header line: `⏺ $ <command>` so it reads like a shell invocation.
        self.push(crate::render::accent_line(
            crate::theme::theme().accent_goal,
            format!("$ {command}"),
            Style::default().fg(crate::theme::theme().accent_goal),
        ));
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&self.workspace_root)
            .output();
        let body = match output {
            Ok(o) => {
                let mut combined = String::from_utf8_lossy(&o.stdout).into_owned();
                if !o.stderr.is_empty() {
                    let err = String::from_utf8_lossy(&o.stderr);
                    if !combined.is_empty() {
                        combined.push('\n');
                    }
                    combined.push_str(&err);
                }
                // Cap the output so a verbose command can't bury the transcript.
                const MAX_LINES: usize = 200;
                let lines: Vec<&str> = combined.lines().collect();
                if lines.len() > MAX_LINES {
                    let mut capped = lines[..MAX_LINES].join("\n");
                    capped.push_str(&format!("\n… ({} more lines)", lines.len() - MAX_LINES));
                    capped
                } else {
                    combined
                }
            }
            Err(err) => format!("failed to run: {err}"),
        };
        // Render the body with ANSI parsing (so colored tool output stays
        // colored) under a dim gutter, matching how tool results are shown.
        let text = body
            .into_text()
            .unwrap_or_else(|_| Text::from(body.clone()));
        // Keep the semantic gutter display-only so copied shell output stays raw.
        let lines: Vec<RLine<'static>> = text.lines;
        for line in lines {
            self.transcript.push(crate::TranscriptEntry::ToolOutput {
                body: vec![line],
                expanded: false,
            });
        }
        self.bump_transcript();
        self.cap_transcript();
        self.follow();
    }

    /// Open the current input draft in an external editor (Ctrl-X). Writes the
    /// draft to a temp file, suspends the TUI (leaves raw mode + alternate
    /// screen), spawns `$VISUAL` or `$EDITOR` (fallback `vi`), waits for it to
    /// exit, then reads the file back and replaces the input. Makes multi-line
    /// prompts practical. Errors are noted in the transcript rather than
    /// propagated so the TUI never crashes on a misconfigured editor.
    pub(crate) fn edit_in_external_editor(&mut self) {
        use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
        use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
        use std::io::Write as _;

        let draft = self.input.text();
        // Pick the editor: `$VISUAL` then `$EDITOR` then `vi`. An empty string
        // is treated as unset so `VISUAL=""` falls through to `EDITOR` (a common
        // misconfiguration that would otherwise launch `vi` with no args).
        let editor = std::env::var("VISUAL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                std::env::var("EDITOR")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
            })
            .unwrap_or_else(|| "vi".to_string());
        // Write the draft to a temp file.
        let tmp = std::env::temp_dir().join(format!(".hi-prompt-{}.md", std::process::id()));
        let tmp_path = match tmp.to_str() {
            Some(s) => s.to_string(),
            None => {
                self.push(Line::styled(
                    "edit failed: couldn't build temp path",
                    Style::default().fg(crate::theme::theme().warning),
                ));
                self.follow();
                return;
            }
        };
        let write = std::fs::write(&tmp_path, &draft);
        if let Err(err) = write {
            self.push(Line::styled(
                format!("edit failed: {err}"),
                Style::default().fg(crate::theme::theme().warning),
            ));
            self.follow();
            return;
        }
        // Suspend the TUI: leave alternate screen + raw mode so the editor
        // gets a normal terminal. Skipped when `HI_TUI_NO_TERMINAL` is set (used
        // by tests so the crossterm calls don't block without a real terminal).
        if std::env::var("HI_TUI_NO_TERMINAL").is_err() {
            let _ = disable_raw_mode();
            let _ = crossterm::execute!(std::io::stdout(), LeaveAlternateScreen);
            let _ = std::io::stdout().flush();
        }

        // Run the editor. Split on whitespace so `$EDITOR="code --wait"` works;
        // the temp file is appended as the last argument. This handles the
        // common case (a program name + optional flags). Block until it exits.
        let mut parts = editor.split_whitespace();
        let prog = parts.next().unwrap_or("vi");
        let args: Vec<&str> = parts.collect();
        let status = std::process::Command::new(prog)
            .args(&args)
            .arg(&tmp_path)
            .status();

        // Resume the TUI: re-enter alternate screen + raw mode. Skipped in
        // tests (see `HI_TUI_NO_TERMINAL` above).
        if std::env::var("HI_TUI_NO_TERMINAL").is_err() {
            let _ = crossterm::execute!(std::io::stdout(), EnterAlternateScreen);
            let _ = enable_raw_mode();
            let _ = std::io::stdout().flush();
        }

        match status {
            Ok(s) if s.success() => {
                match std::fs::read_to_string(&tmp) {
                    Ok(contents) => {
                        // Normalize CRLF and set the input to the edited text.
                        let normalized = contents.replace("\r\n", "\n").replace('\r', "\n");
                        self.input.set(&normalized);
                        self.push(Line::styled(
                            format!("edited in {prog} ({} chars)", normalized.chars().count()),
                            dim(),
                        ));
                    }
                    Err(err) => {
                        self.push(Line::styled(
                            format!("edit: editor exited but couldn't read back: {err}"),
                            Style::default().fg(crate::theme::theme().warning),
                        ));
                    }
                }
            }
            Ok(s) => {
                self.push(Line::styled(
                    format!("edit: {prog} exited with {s}"),
                    Style::default().fg(crate::theme::theme().warning),
                ));
            }
            Err(err) => {
                self.push(Line::styled(
                    format!("edit: couldn't run {prog}: {err}"),
                    Style::default().fg(crate::theme::theme().warning),
                ));
            }
        }
        // Clean up the temp file.
        let _ = std::fs::remove_file(&tmp_path);
        self.follow();
    }

    /// Accumulate `last_changed_files` into `session_changed_files` (the
    /// session-cumulative set), deduplicating while preserving first-seen order.
    /// Called after each turn so `/files` can show everything the session
    /// touched, even while a turn is running (when the per-turn line is hidden).
    pub(crate) fn accumulate_session_files(&mut self) {
        for f in &self.last_changed_files {
            if !self.session_changed_files.iter().any(|s| s == f) {
                self.session_changed_files.push(f.clone());
            }
        }
    }

    /// Show all files touched this session (`/files`): the Changes pane's
    /// `N files changed +A -D` summary and per-file counts, as transcript
    /// lines. If nothing has changed yet, says so.
    pub(crate) fn show_session_files(&mut self) {
        for line in self.session_files_lines() {
            self.push(line);
        }
        self.follow();
    }

    /// Copy the assistant's most recent fenced code block to the clipboard
    /// (Ctrl-Y). The block is captured during streaming in `last_code_block`;
    /// when that's empty (e.g. a resumed session whose transcript was replayed
    /// from JSONL, so `commit_md_line` never ran), fall back to scanning the
    /// transcript backward for the last fenced code block.
    pub(crate) fn copy_last_code_block(&mut self) {
        let text = self
            .last_code_block
            .as_deref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| self.scan_transcript_for_last_code_block())
            .unwrap_or_default();
        if text.is_empty() {
            self.push(Line::styled("no code block to copy yet", dim()));
        } else {
            match copy_to_clipboard(&text) {
                Ok(()) => {
                    self.copy_toast = Some((text.chars().count(), Instant::now()));
                    self.push(Line::styled(
                        format!("copied code block ({} chars)", text.chars().count()),
                        dim(),
                    ));
                }
                Err(err) => self.push(Line::styled(
                    format!("copy failed: {err}"),
                    Style::default().fg(crate::theme::theme().warning),
                )),
            }
        }
        self.follow();
    }

    /// Fallback for resumed sessions: scan the transcript backward for the last
    /// fenced code block. Code lines render with a `▏ ` gutter prefix (from
    /// `markdown_line`); a contiguous run of gutter-prefixed lines is one code
    /// block. We take the last such run, strip the gutter, and return the body
    /// (dropping the fence-open line, which carries only the language tag).
    pub(crate) fn scan_transcript_for_last_code_block(&self) -> Option<String> {
        let mut lines: Vec<String> = Vec::new();
        let mut found = false;
        // Walk backward; collect gutter-prefixed lines until the run breaks.
        for entry in self.transcript.iter().rev() {
            // `text()` intentionally removes display-only gutters for copy and
            // export. Code-block recovery needs the raw rendered line so it can
            // still distinguish markdown's `▏ ` guide from ordinary text.
            let text = match entry {
                crate::TranscriptEntry::Line(line) => crate::render::line_text(line),
                crate::TranscriptEntry::AssistantMessage { text } => {
                    if let Some(block) = Self::last_fenced_block(text) {
                        return Some(block);
                    }
                    continue;
                }
                _ => entry.text(),
            };
            if let Some(body) = text.strip_prefix("▏ ").or_else(|| text.strip_prefix('▏')) {
                // A code line (interior or fence). Keep collecting.
                lines.push(body.to_string());
                found = true;
            } else if found {
                // We were inside a code run and hit a non-code line — stop.
                break;
            }
        }
        if !found {
            return None;
        }
        // `lines` is in reverse order; reverse to get top-to-bottom.
        lines.reverse();
        // Drop the fence-open line (first line, carries the language tag) and
        // the fence-close line (last line, empty after gutter). Interior lines
        // are the actual code.
        if lines.len() >= 2 {
            // The first line is the ```lang fence; the last is the ``` close.
            let interior = &lines[1..lines.len() - 1];
            let body = interior.join("\n");
            let body = body.trim();
            if body.is_empty() {
                None
            } else {
                Some(body.to_string())
            }
        } else {
            None
        }
    }

    fn last_fenced_block(text: &str) -> Option<String> {
        let mut last: Option<String> = None;
        let mut current: Option<String> = None;
        for line in text.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("```") {
                if current.is_some() {
                    last = current.take();
                } else {
                    current = Some(String::new());
                }
            } else if let Some(buf) = current.as_mut() {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(line);
            }
        }
        last.filter(|s| !s.trim().is_empty())
    }

    pub(crate) fn copy(&mut self, arg: &str) {
        let text = match arg.trim() {
            "all" | "transcript" => self.transcript_text(),
            _ => self.last_assistant.trim().to_string(),
        };
        if text.is_empty() {
            self.push(Line::styled("nothing to copy yet", dim()));
        } else {
            match copy_to_clipboard(&text) {
                Ok(()) => self.push(Line::styled(format!("copied {} chars", text.len()), dim())),
                Err(err) => self.push(Line::styled(
                    format!("copy failed: {err}"),
                    Style::default().fg(crate::theme::theme().warning),
                )),
            }
        }
        self.follow();
    }

    pub(crate) fn handle_pending_resume_key(&mut self, key: &KeyEvent) -> bool {
        let Some(card) = self.pending_resume.as_mut() else {
            return false;
        };
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                card.selected = 0;
                true
            }
            KeyCode::Down | KeyCode::Char('j') => {
                card.selected = 1;
                true
            }
            KeyCode::Char('y') => {
                self.pending_resume = None;
                self.resume_incomplete_requested = true;
                true
            }
            KeyCode::Esc | KeyCode::Char('n') => {
                self.pending_resume = None;
                true
            }
            KeyCode::Enter => {
                let continue_turn = card.selected == 0;
                self.pending_resume = None;
                if continue_turn {
                    self.resume_incomplete_requested = true;
                }
                true
            }
            _ => true,
        }
    }
}
