//! `App` methods: commands.

use ansi_to_tui::IntoText;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_agent::{Agent, Command, command};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use std::time::Instant;

use crate::model_picker::ModelPicker;
use crate::render::dim;
use crate::util::{copy_to_clipboard, goal_feedback};
use crate::{TurnState, diff_for_files_sync, working_tree_diff_sync};

impl crate::App {
    /// Apply a pure editing/navigation key to the input line, shared by the
    /// idle input phase and the in-turn queue-entry path. Returns the submitted
    /// text on Enter (when non-empty); the caller decides whether to run it now
    /// or queue it. Phase-specific control keys (Ctrl-C/Esc) are handled by the
    /// caller, not here.
    pub(crate) fn edit_key(&mut self, key: &KeyEvent) -> Option<String> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        // --- Ctrl-R reverse history search mode ---
        // When active, keystrokes go to the search filter, not the input line.
        if let Some(search) = &mut self.history_search {
            match key.code {
                KeyCode::Enter => {
                    // Load the highlighted match into the input and submit it.
                    let idx = search.current();
                    self.history_search = None;
                    if let Some(i) = idx
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                        let line = self.input.submit();
                        if !line.trim().is_empty() {
                            self.input.save_history(&self.workspace_root);
                            return Some(line);
                        }
                    }
                    return None;
                }
                KeyCode::Esc => {
                    // On Esc, load the highlighted match for editing (don't submit).
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    self.history_search = None;
                    return None;
                }
                KeyCode::Char('r') if ctrl => {
                    // Cycle to the next match (like bash Ctrl-R).
                    search.next();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                KeyCode::Char('s') if ctrl => {
                    search.prev();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                KeyCode::Backspace => {
                    search.backspace(&self.input.history);
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                KeyCode::Up => {
                    search.prev();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                KeyCode::Down => {
                    search.next();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                KeyCode::Char(c) if !ctrl => {
                    search.insert(c, &self.input.history);
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                _ => return None,
            }
        }
        // --- Block-navigation mode (Ctrl-B) ---
        // A cursor over tool-output blocks; keys drive the cursor and folding
        // rather than the input line. Any block count change is handled by the
        // clamp in `selected_block_ord`.
        if self.nav_mode {
            match key.code {
                KeyCode::Esc => self.nav_mode = false,
                KeyCode::Char('b') if ctrl => self.nav_mode = false,
                KeyCode::Up | KeyCode::Char('k') => {
                    self.block_cursor = self.selected_block_ord().saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    let n = self.tool_block_count();
                    if n > 0 {
                        self.block_cursor = (self.selected_block_ord() + 1).min(n - 1);
                    }
                }
                KeyCode::Enter | KeyCode::Char(' ') => self.toggle_selected_block(),
                _ => {}
            }
            return None;
        }
        match key.code {
            // Alt+Enter inserts a newline (multi-line prompt without pasting); so
            // does a trailing backslash, for terminals that can't send Alt+Enter.
            KeyCode::Enter if alt => self.input.insert('\n'),
            KeyCode::Enter if self.input.continue_line() => {}
            KeyCode::Enter => {
                let line = self.input.submit();
                if !line.trim().is_empty() {
                    self.input.save_history(&self.workspace_root);
                    return Some(line);
                }
            }
            KeyCode::Char('u') if ctrl => self.input.kill_to_start(),
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
            // Readline word motions: Alt-B/F move by word, Ctrl-W deletes the
            // word before the cursor, Ctrl-K kills to end of line.
            KeyCode::Char('b') if alt => self.input.word_left(),
            KeyCode::Char('f') if alt => self.input.word_right(),
            KeyCode::Char('w') if ctrl => self.input.delete_word_back(),
            KeyCode::Char('k') if ctrl => self.input.kill_to_end(),
            // Toggle the working-tree diff panel. Refreshed when opened so it
            // reflects the current tree, not a stale snapshot. Fetched
            // synchronously (a `git diff` is fast and user-initiated) since the
            // key handler isn't async.
            KeyCode::Char('d') if ctrl => {
                self.show_diff = !self.show_diff;
                if self.show_diff {
                    self.diff_text = Some(working_tree_diff_sync(&self.workspace_root));
                } else {
                    self.diff_text = None;
                }
            }
            // Full-screen diff review overlay (Ctrl-G): a scrollable,
            // syntax-colored view of the entire working-tree diff with
            // hunk-to-hunk navigation (n/p). Takes over the screen until
            // closed with q/Esc/Ctrl-G.
            KeyCode::Char('g') if ctrl => {
                if self.show_review {
                    self.show_review = false;
                } else {
                    self.open_review(None);
                }
            }
            // Toggle the agent-observability panel (Ctrl-? = Ctrl-Shift-/).
            // Shows the last turn's trajectory telemetry, tool-call count, and
            // context composition — read-only diagnostics for the agent's own
            // behavior.
            KeyCode::Char('?') if ctrl => {
                self.show_debug = !self.show_debug;
            }
            // Toggle reasoning (CoT) expansion: collapsed "thought for Ns"
            // summaries vs. the full thinking text. Off by default so reasoning
            // doesn't flood the transcript; Ctrl-T shows/hides all blocks.
            KeyCode::Char('t') if ctrl => {
                self.show_reasoning = !self.show_reasoning;
            }
            // Toggle full tool-output expansion: long blocks fold to a preview
            // by default; Ctrl-O reveals every block's full body (and back).
            KeyCode::Char('o') if ctrl => {
                self.show_tool_output = !self.show_tool_output;
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
                    self.nav_mode = true;
                    self.block_cursor = n - 1;
                }
            }
            // External editor hand-off (Ctrl-X): dump the current draft into
            // `$VISUAL`/`$EDITOR` (fallback `vi`), suspend the TUI, and read
            // the result back on save. Makes multi-line prompts practical —
            // anything past ~5 lines is painful in the single-line editor.
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
            KeyCode::Char(c) if !ctrl => self.input.insert(c),
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Up => self.input.history_prev(),
            KeyCode::Down => self.input.history_next(),
            KeyCode::PageUp => self.scroll_up(5),
            KeyCode::PageDown => self.scroll_down(5),
            _ => {}
        }
        None
    }

    /// Number of tool-output blocks in the transcript (the foldable blocks that
    /// block-nav steps over).
    pub(crate) fn tool_block_count(&self) -> usize {
        self.transcript
            .iter()
            .filter(|e| matches!(e, crate::TranscriptEntry::ToolOutput { .. }))
            .count()
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

    /// Flip the expand state of the `target`-th tool-output block.
    pub(crate) fn toggle_block_ord(&mut self, target: usize) {
        let mut ord = 0;
        for entry in self.transcript.iter_mut() {
            if let crate::TranscriptEntry::ToolOutput { expanded, .. } = entry {
                if ord == target {
                    *expanded = !*expanded;
                    return;
                }
                ord += 1;
            }
        }
    }

    pub(crate) fn write_debug_log(&mut self) {
        let path = std::path::Path::new(".hi-debug.log");
        let mut body = String::new();
        body.push_str("# hi debug log (redacted; best-effort secret detection)\n\n");
        body.push_str("## status\n");
        body.push_str(&format!(
            "provider: {}\nmodel: {}\n",
            self.provider, self.model
        ));
        body.push_str(&format!("status: {}\n\n", self.status));
        body.push_str(&format!(
            "last_error: {}\nwaiting_for: {:?}\nstartup_notice: {}\nlast_turn_file_edits: {}\n\n",
            self.last_error.as_deref().unwrap_or("none"),
            self.waiting_for,
            self.startup_notice.as_deref().unwrap_or("none"),
            self.last_turn_had_file_edits
        ));
        body.push_str("## events\n");
        for event in &self.event_log {
            body.push_str(event);
            body.push('\n');
        }
        body.push_str("\n## transcript\n");
        for entry in &self.transcript {
            match entry {
                crate::TranscriptEntry::Line(_)
                | crate::TranscriptEntry::UserPrompt(_)
                | crate::TranscriptEntry::ChangedFiles { .. }
                | crate::TranscriptEntry::ToolOutput { .. } => {
                    body.push_str(&entry.text());
                    body.push('\n');
                }
                crate::TranscriptEntry::Reasoning { .. } => {
                    body.push_str("[reasoning omitted]\n");
                }
            }
        }
        let body = hi_agent::ui::redact_debug_text(&body, &[self.api_key.as_str()]);
        match hi_agent::ui::write_private_debug_log(path, &body) {
            Ok(()) => self.push(Line::styled(
                "wrote redacted debug log: .hi-debug.log",
                dim(),
            )),
            Err(err) => self.push(Line::styled(
                format!("log failed: {err}"),
                Style::default().fg(Color::Yellow),
            )),
        }
        self.follow();
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
        let gutter = crate::render::gutter(crate::theme::theme().gray_dim);
        let lines: Vec<RLine<'static>> = text
            .lines
            .into_iter()
            .map(|mut line| {
                line.spans.insert(0, gutter.clone());
                line
            })
            .collect();
        for line in lines {
            self.transcript.push(crate::TranscriptEntry::ToolOutput {
                body: vec![line],
                expanded: false,
            });
        }
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
                    Style::default().fg(Color::Yellow),
                ));
                self.follow();
                return;
            }
        };
        let write = std::fs::write(&tmp_path, &draft);
        if let Err(err) = write {
            self.push(Line::styled(
                format!("edit failed: {err}"),
                Style::default().fg(Color::Yellow),
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
                            Style::default().fg(Color::Yellow),
                        ));
                    }
                }
            }
            Ok(s) => {
                self.push(Line::styled(
                    format!("edit: {prog} exited with {s}"),
                    Style::default().fg(Color::Yellow),
                ));
            }
            Err(err) => {
                self.push(Line::styled(
                    format!("edit: couldn't run {prog}: {err}"),
                    Style::default().fg(Color::Yellow),
                ));
            }
        }
        // Clean up the temp file.
        let _ = std::fs::remove_file(&tmp_path);
        self.follow();
    }

    /// Open the full-screen diff review overlay (Ctrl-G). When `files` is
    /// `None`, shows the entire working-tree diff; when `Some`, shows only the
    /// diff for those paths — used by the deep-link from a `✎ files changed`
    /// transcript line (click or `/review <file>`).
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

    /// Show all files touched this session (`/files`): a header with the count,
    /// then one line per file. If nothing has changed yet, says so.
    pub(crate) fn show_session_files(&mut self) {
        if self.session_changed_files.is_empty() {
            self.push(Line::styled("no files changed this session yet", dim()));
            return;
        }
        let count = self.session_changed_files.len();
        let files: Vec<String> = self.session_changed_files.clone();
        self.push(Line::styled(
            format!(
                "── {} file{} changed this session ──",
                count,
                if count == 1 { "" } else { "s" }
            ),
            Style::default()
                .fg(crate::theme::theme().accent_goal)
                .add_modifier(Modifier::BOLD),
        ));
        for f in &files {
            self.push(Line::styled(format!("  {f}"), dim()));
        }
        self.follow();
    }

    pub(crate) fn open_review(&mut self, files: Option<&[String]>) {
        let diff = match files {
            None => working_tree_diff_sync(&self.workspace_root),
            Some(paths) => diff_for_files_sync(&self.workspace_root, paths),
        };
        self.diff_text = Some(diff);
        self.review_scroll = 0;
        self.show_review = true;
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
                    Style::default().fg(Color::Yellow),
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
            let text = entry.text();
            if let Some(body) = text.strip_prefix("▏ ") {
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
                    Style::default().fg(Color::Yellow),
                )),
            }
        }
        self.follow();
    }

    /// `/goal` (read), `/goal clear`, and `/goal <objective>` when no planner
    /// decomposition applies (non-pipenetwork, or the planner is unavailable). The
    /// planner-decomposed path is driven from the run loop (it's an async call that
    /// needs the spinner) and lands in [`set_planned_goal`](Self::set_planned_goal).
    pub(crate) fn handle_goal(&mut self, agent: &mut Agent, arg: &str) {
        match arg.trim() {
            // `/goal limit <n>` / `limit off` — cap or uncap plan growth.
            s if command::parse_goal_limit(s).is_some() => {
                if let Some(limit) = command::parse_goal_limit(s) {
                    self.handle_goal_limit(agent, limit);
                }
            }
            // `/goal team on|off` — toggle the skeptic review gate.
            s if command::parse_goal_team(s).is_some() => {
                if let Some(team) = command::parse_goal_team(s) {
                    self.handle_goal_team(agent, team);
                }
            }
            // Pause/resume: hold progress, stop/restart steering. Own messaging,
            // not the goal-set echo.
            "pause" | "resume" => {
                let pause = arg.trim() == "pause";
                let (msg, style) = if agent.set_goal_paused(pause) {
                    let text = if pause {
                        "✓ goal paused — resume with /goal resume"
                    } else {
                        "✓ goal resumed — steering turns again"
                    };
                    (text.to_string(), Style::default().fg(Color::Green))
                } else {
                    (format!("no goal to {}", arg.trim()), dim())
                };
                self.refresh_goal(agent);
                self.push(Line::styled(msg, style));
                self.follow();
            }
            "clear" | "off" | "none" => {
                let error = agent
                    .set_transient_goal(None)
                    .err()
                    .map(|err| format!("goal clear failed: {err:#}"));
                self.refresh_goal(agent);
                self.report_goal_result(agent, arg, error);
            }
            "" => self.report_goal_result(agent, arg, None), // report current
            // A single sub-goal equal to the objective (no decomposition).
            goal => {
                let error = Self::apply_goal(agent, goal, vec![goal.to_string()]);
                self.refresh_goal(agent);
                self.report_goal_result(agent, arg, error);
            }
        }
    }

    /// `/goal limit …`: set/clear/report the plan-growth ceiling.
    fn handle_goal_limit(&mut self, agent: &mut Agent, limit: command::GoalLimitArg) {
        use command::GoalLimitArg;
        let (msg, style) = match limit {
            GoalLimitArg::Show => match agent.structured_goal().and_then(|g| g.step_limit) {
                Some(n) => (format!("goal limit: {n} sub-goals"), dim()),
                None => (
                    "goal limit: none — the plan grows freely".to_string(),
                    dim(),
                ),
            },
            GoalLimitArg::Set(n) => {
                if agent.set_goal_step_limit(Some(n)) {
                    (
                        format!("✓ goal limit set to {n} sub-goals"),
                        Style::default().fg(Color::Green),
                    )
                } else {
                    ("no goal to limit".to_string(), dim())
                }
            }
            GoalLimitArg::Unlimited => {
                if agent.set_goal_step_limit(None) {
                    (
                        "✓ goal limit removed — the plan grows freely".to_string(),
                        Style::default().fg(Color::Green),
                    )
                } else {
                    ("no goal to limit".to_string(), dim())
                }
            }
            GoalLimitArg::Invalid(value) => (
                format!(
                    "goal limit: '{value}' isn't a number — use /goal limit <n> or 'limit off'"
                ),
                Style::default().fg(Color::Yellow),
            ),
        };
        self.refresh_goal(agent);
        self.push(Line::styled(msg, style));
        self.follow();
    }

    /// `/goal team on|off`: toggle the skeptic review gate for the active goal.
    fn handle_goal_team(&mut self, agent: &mut Agent, team: command::GoalTeamArg) {
        use command::GoalTeamArg;
        let (msg, style) = match team {
            GoalTeamArg::Show => match agent.structured_goal() {
                Some(g) if g.team => (
                    format!(
                        "goal team: on — skeptic reviews each advance ({} objection(s), {} unavailable; last: {})",
                        g.skeptic_objections,
                        g.skeptic_unavailable,
                        g.last_skeptic_status
                            .map(|s| format!("{s:?}"))
                            .unwrap_or_else(|| "not run".into())
                    ),
                    dim(),
                ),
                Some(_) => (
                    "goal team: off — enable with /goal team on".to_string(),
                    dim(),
                ),
                None => (
                    "no active goal — set one with /goal <text> first".to_string(),
                    dim(),
                ),
            },
            GoalTeamArg::On => {
                if agent.set_goal_team(true) {
                    (
                        format!(
                            "✓ goal team on — {} reviews each turn before advancing a sub-goal",
                            agent.effective_skeptic_model()
                        ),
                        Style::default().fg(Color::Green),
                    )
                } else {
                    (
                        "no active goal — set one with /goal <text> first".to_string(),
                        dim(),
                    )
                }
            }
            GoalTeamArg::Off => {
                if agent.set_goal_team(false) {
                    (
                        "✓ goal team off — single-agent driving".to_string(),
                        Style::default().fg(Color::Green),
                    )
                } else {
                    ("no active goal".to_string(), dim())
                }
            }
            GoalTeamArg::Invalid(value) => (
                format!("goal team: '{value}' — use /goal team on|off"),
                Style::default().fg(Color::Yellow),
            ),
        };
        self.refresh_goal(agent);
        self.push(Line::styled(msg, style));
        self.follow();
    }

    /// Install a goal whose sub-goals a planner already decomposed (from the run
    /// loop, after [`Agent::decompose_goal`]), then echo the resulting checklist.
    pub(crate) fn set_planned_goal(
        &mut self,
        agent: &mut Agent,
        objective: &str,
        sub_goals: Vec<String>,
    ) {
        let error = Self::apply_goal(agent, objective, sub_goals);
        self.refresh_goal(agent);
        self.report_goal_result(agent, objective, error);
    }

    /// Set a structured `Goal` from a decomposed sub-goal list; fall back to a
    /// transient goal string when the long-horizon path is off. Returns an error
    /// message on failure. When long-horizon is on, the executor's own
    /// `update_plan` calls report progress onto these sub-goals.
    fn apply_goal(agent: &mut Agent, objective: &str, sub_goals: Vec<String>) -> Option<String> {
        if agent.long_horizon() {
            match agent
                .set_structured_goal(Some(hi_agent::Goal::new(objective.to_string(), sub_goals)))
            {
                Ok(true) => None,
                Ok(false) => agent
                    .set_transient_goal(Some(objective.to_string()))
                    .err()
                    .map(|err| format!("goal set failed: {err:#}")),
                Err(err) => Some(format!("goal set failed: {err:#}")),
            }
        } else {
            agent
                .set_transient_goal(Some(objective.to_string()))
                .err()
                .map(|err| format!("goal set failed: {err:#}"))
        }
    }

    /// Mirror the agent's active structured goal into the `App` so the pinned plan
    /// block and header can render sub-goal progress.
    pub(crate) fn refresh_goal(&mut self, agent: &Agent) {
        self.goal = agent.structured_goal().cloned();
    }

    /// Queue the synthetic drive prompt when an active, unpaused goal should keep
    /// moving: the run loop pops it like user input, so the agent works the next
    /// sub-goal without the user re-prompting. Queued user input always takes
    /// priority (only queues into an empty queue), and a stall stop holds until a
    /// user turn resets it.
    pub(crate) fn maybe_queue_goal_drive(&mut self, agent: &Agent) {
        if !self.queue.is_empty() || self.goal_drive_stall >= hi_agent::GOAL_DRIVE_STALL_LIMIT {
            return;
        }
        if agent
            .structured_goal()
            .is_some_and(hi_agent::Goal::should_auto_drive)
        {
            self.queue
                .push_back(hi_agent::GOAL_CONTINUE_PROMPT.to_string());
        }
    }

    /// Handle `/theme`: set a named mode (`dark`/`light`/`ansi`/`auto`), or
    /// cycle to the next when the arg is empty. Applies immediately (the whole
    /// TUI re-reads the theme each frame) and echoes the new mode.
    fn handle_theme(&mut self, arg: &str) {
        let arg = arg.trim();
        let mode = if arg.is_empty() {
            crate::theme::cycle_mode()
        } else if let Some(mode) = crate::theme::ThemeMode::parse(arg) {
            crate::theme::set_mode(mode);
            mode
        } else {
            self.push(Line::styled(
                format!("unknown theme '{arg}' — try dark, light, ansi, or auto"),
                Style::default().fg(crate::theme::theme().warning),
            ));
            self.follow();
            return;
        };
        let note = if mode == crate::theme::ThemeMode::Auto {
            format!("theme: {} (following OS light/dark)", mode.label())
        } else {
            format!("theme: {}", mode.label())
        };
        self.push(Line::styled(
            note,
            Style::default().fg(crate::theme::theme().accent_success),
        ));
        self.follow();
    }

    /// Toggle terminal mouse capture. Off releases the mouse to the terminal's
    /// own text selection (at the cost of the scroll wheel and click/drag block
    /// folding + copy); on restores app control.
    fn handle_mouse_command(&mut self, arg: &str) {
        use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
        let want = match arg.trim() {
            "" => !self.mouse_capture,
            "on" | "enable" => true,
            "off" | "disable" => false,
            other => {
                self.push(Line::styled(
                    format!("unknown '{other}' — try /mouse on or /mouse off"),
                    Style::default().fg(crate::theme::theme().warning),
                ));
                self.follow();
                return;
            }
        };
        if want == self.mouse_capture {
            self.push(Line::styled(
                format!("mouse capture already {}", if want { "on" } else { "off" }),
                crate::render::dim(),
            ));
            self.follow();
            return;
        }
        let res = if want {
            crossterm::execute!(std::io::stdout(), EnableMouseCapture)
        } else {
            crossterm::execute!(std::io::stdout(), DisableMouseCapture)
        };
        match res {
            Ok(()) => {
                self.mouse_capture = want;
                if !want {
                    self.clear_selection();
                }
                let note = if want {
                    "mouse capture on — scroll wheel, click-to-fold, and drag-to-copy active"
                } else {
                    "mouse capture off — drag selects text natively; scroll wheel / click-fold / drag-copy off"
                };
                self.push(Line::styled(
                    note,
                    Style::default().fg(crate::theme::theme().accent_success),
                ));
            }
            Err(err) => self.push(Line::styled(
                format!("could not change mouse capture: {err}"),
                Style::default().fg(crate::theme::theme().warning),
            )),
        }
        self.follow();
    }

    /// Echo the current goal state: the structured checklist summary (prominent),
    /// or the transient set/clear/read feedback.
    fn report_goal_result(&mut self, agent: &Agent, arg: &str, error: Option<String>) {
        if let Some(msg) = error {
            self.push(Line::styled(msg, Style::default().fg(Color::Yellow)));
            self.follow();
            return;
        }
        let (msg, prominent) = if let Some(g) = agent.structured_goal() {
            let done = g
                .sub_goals
                .iter()
                .filter(|s| s.status == hi_agent::GoalStatus::Done)
                .count();
            (
                format!(
                    "goal: {} — {}/{} sub-goals done",
                    g.objective,
                    done,
                    g.sub_goals.len()
                ),
                true,
            )
        } else {
            goal_feedback(arg, agent.goal())
        };
        // A set/clear is an applied change — show it plainly (green), not dim, so
        // it's obvious it took effect. A bare `/goal` is just a read-out.
        let style = if prominent {
            Style::default().fg(Color::Green)
        } else {
            dim()
        };
        self.push(Line::styled(msg, style));
        self.follow();
    }

    pub(crate) async fn handle_command(&mut self, agent: &mut Agent, command: Command) {
        match command {
            Command::Quit => {}
            // Handled inline by the run loop (needs terminal/input/ticker).
            Command::Dashboard(_) => {}
            // Handled inline by the run loop (needs the loops manager handle).
            Command::Loop(_) => {}
            // Handled inline by the run loop (needs terminal/input/ticker).
            Command::Watch => {}
            // Handled inline by the run loop (needs the loops manager handle).
            Command::Digest => {}
            Command::Rsi(arg) => {
                let message = match agent.rsi_command(&arg).await {
                    Ok(output) => output,
                    Err(error) => format!("RSI command error: {error:#}"),
                };
                for line in message.lines() {
                    self.push(Line::styled(line.to_string(), dim()));
                }
            }
            Command::Theme(arg) => self.handle_theme(&arg),
            Command::Mouse(arg) => self.handle_mouse_command(&arg),
            Command::Help => {
                for line in command::help_text().lines() {
                    self.push(Line::styled(line.to_string(), dim()));
                }
            }
            Command::Status => self.report_status(agent),
            Command::Log => self.write_debug_log(),
            Command::Model(id) => {
                if id.is_empty() {
                    // Open the interactive picker (filter + arrow-select) on the
                    // live served list — no static catalog fallback.
                    let current = self.model.clone();
                    let tags = self.served_tags();
                    let mut ids: Vec<String> = self.served.keys().cloned().collect();
                    ids.sort();
                    if ids.is_empty() {
                        self.push(Line::styled(
                            "no live model list available yet".to_string(),
                            dim(),
                        ));
                    } else {
                        self.picker = Some(ModelPicker::new(ids, &current, tags, &self.served));
                    }
                } else {
                    self.select_model(agent, &id);
                }
            }
            Command::Clear => {
                let count = agent
                    .messages()
                    .iter()
                    .filter(|m| m.role != hi_ai::Role::System)
                    .count();
                match agent.clear_history() {
                    Ok(()) => {
                        self.transcript.clear();
                        self.event_log.clear();
                        self.pending = None;
                        self.code_lang = None;
                        self.current_assistant.clear();
                        self.last_assistant.clear();
                        self.status.clear();
                        self.last_turn_state = TurnState::Idle;
                        self.push(Line::styled(
                            format!("cleared {count} messages — starting fresh"),
                            dim(),
                        ));
                    }
                    Err(err) => {
                        self.push(Line::styled(
                            format!("clear failed: {err}"),
                            Style::default().fg(Color::Yellow),
                        ));
                    }
                }
            }
            Command::Verify(arg) => {
                let msg = match arg.trim() {
                    "" if agent.verify_is_on() => format!("verify: {}", agent.verify_summary()),
                    "" => "verify: off (set one with /verify <cmd>)".to_string(),
                    "off" | "none" | "clear" | "disable" => match agent.set_verify_command(None) {
                        Ok(()) => "verification disabled".to_string(),
                        Err(error) => format!("verification config error: {error}"),
                    },
                    cmd => match agent.set_verify_command(Some(cmd.to_string())) {
                        Ok(()) => format!(
                            "verification on: `{cmd}` — runs after each turn, iterates on failure"
                        ),
                        Err(error) => format!("verification config error: {error}"),
                    },
                };
                self.push(Line::styled(msg, dim()));
            }
            Command::Config(arg) => {
                use hi_agent::command::{ConfigArg, parse_config_arg};
                match parse_config_arg(&arg) {
                    ConfigArg::Show => {
                        let s = agent.config_snapshot();
                        // Box border + field labels stay dim; values render at
                        // normal intensity so the actual settings are readable.
                        let row = |label: &str, value: String| {
                            Line::from(vec![
                                Span::styled(format!("│ {label}"), dim()),
                                Span::raw(format!(" {value}")),
                            ])
                        };
                        self.push(Line::styled(
                            "╭─ config ───────────────────────────────────────────╮".to_string(),
                            dim(),
                        ));
                        self.push(row("model:          ", s.model));
                        if !s.provider_route.is_empty() {
                            self.push(row("provider:       ", s.provider_route));
                        }
                        self.push(row("max-tokens:     ", s.max_tokens));
                        self.push(row("thinking-budget:", s.thinking_budget));
                        self.push(row("reasoning:      ", s.reasoning_effort));
                        self.push(row("temperature:    ", s.temperature));
                        self.push(row("steps:          ", s.max_steps));
                        self.push(row("tool-mode:      ", s.tool_mode));
                        self.push(row("compat:         ", s.compat));
                        self.push(row("verify:         ", s.verify));
                        self.push(row("review:         ", s.review));
                        self.push(row("lsp:            ", s.lsp));
                        self.push(row("tool-set:       ", s.tool_set));
                        self.push(row("auto-compact:   ", s.auto_compact));
                        self.push(row("proactive-verify:", s.proactive_verify.to_string()));
                        self.push(row(
                            "read-only-preflight:",
                            s.read_only_preflight.to_string(),
                        ));
                        self.push(row("long-horizon:   ", s.long_horizon.to_string()));
                        self.push(row("confirm-edits:  ", s.confirm_edits.to_string()));
                        self.push(row("curate-skills:  ", s.curate_skills.to_string()));
                        self.push(row("explore-subagents:", s.explore_subagents.to_string()));
                        self.push(row("write-subagents:", s.write_subagents.to_string()));
                        self.push(row("planner-model:  ", s.planner_model));
                        self.push(row("skeptic-model:  ", s.skeptic_model));
                        self.push(row("moe-streaming:  ", s.moe_streaming));
                        let (rsi_requested, rsi_mode, rsi_latest) = agent.rsi_status();
                        let rsi_latest =
                            rsi_latest.map_or("none", |value| if value { "yes" } else { "no" });
                        self.push(row("RSI requested:  ", rsi_requested.to_string()));
                        self.push(row("RSI active mode:", rsi_mode.to_string()));
                        self.push(row("RSI channel:    ", agent.rsi_channel().to_string()));
                        let rsi_spend = agent
                            .rsi_maximum_cost_microusd()
                            .map(hi_agent::command::format_usd_micros)
                            .unwrap_or_else(|| "unavailable".to_string());
                        self.push(row("RSI spend limit:", format!("{rsi_spend} per run")));
                        self.push(row("RSI observed:   ", rsi_latest.to_string()));
                        self.push(Line::styled(
                            "╰────────────────────────────────────────────────────╯".to_string(),
                            dim(),
                        ));
                        self.push(Line::styled(
                            "set: /config reasoning <minimal|low|medium|high|xhigh|off> · \
                             /config temp <0.0-2.0|off> · /config steps <1+|auto|off> · \
                             /config moe-streaming <on|off|auto> · /config rsi [on|off|spend-limit <USD>|channel stable|beta]"
                                .to_string(),
                            dim(),
                        ));
                    }
                    ConfigArg::Reasoning(effort) => {
                        agent.set_reasoning_effort(effort);
                        let msg = match effort {
                            Some(e) => format!(
                                "reasoning effort → {} (applies next turn; OpenAI-compatible endpoints only)",
                                e.as_str()
                            ),
                            None => "reasoning effort → off (no reasoning_effort sent; endpoint default)"
                                .to_string(),
                        };
                        self.push(Line::styled(msg, dim()));
                    }
                    ConfigArg::Temperature(temp) => {
                        agent.set_temperature(temp);
                        let msg = match temp {
                            Some(t) => format!("temperature → {t}"),
                            None => "temperature → provider default (cleared)".to_string(),
                        };
                        self.push(Line::styled(msg, dim()));
                    }
                    ConfigArg::MaxSteps(limit) => {
                        agent.set_max_steps_limit(limit);
                        let msg = match limit {
                            Some(limit) => format!("step limit → {limit} (applies next turn)"),
                            None => "step limit → off (applies next turn)".to_string(),
                        };
                        self.push(Line::styled(msg, dim()));
                    }
                    ConfigArg::MaxStepsAuto => {
                        agent.set_max_steps_auto();
                        self.push(Line::styled(
                            "step limit → auto (intent-aware; applies next turn)".to_string(),
                            dim(),
                        ));
                    }
                    ConfigArg::MoeStreaming(mode) => {
                        let env = "HI_MLX_EXPERT_STREAMING";
                        let msg = match mode {
                            hi_agent::command::MoeStreamingMode::On => {
                                // SAFETY: TUI runs single-threaded for command handling.
                                unsafe { std::env::set_var(env, "1") };
                                "MoE streaming → on (applies next model load; MLX backend)"
                                    .to_string()
                            }
                            hi_agent::command::MoeStreamingMode::Off => {
                                // SAFETY: TUI runs single-threaded for command handling.
                                unsafe { std::env::set_var(env, "0") };
                                "MoE streaming → off / resident (applies next model load; MLX backend)"
                                    .to_string()
                            }
                            hi_agent::command::MoeStreamingMode::Auto => {
                                // SAFETY: TUI runs single-threaded for command handling.
                                unsafe { std::env::remove_var(env) };
                                "MoE streaming → auto (applies next model load; streams when model exceeds memory budget)"
                                    .to_string()
                            }
                        };
                        self.push(Line::styled(msg, dim()));
                    }
                    ConfigArg::SkepticLocal(on) => {
                        if on {
                            self.push(Line::styled(
                                "local skeptic: detecting backend…".to_string(),
                                dim(),
                            ));
                            // The TUI owns an alternate screen, so it can't run
                            // the progress-to-terminal model download inline —
                            // it reports NeedsDownload instead of corrupting it.
                            let msg = match agent.enable_local_skeptic(false).await {
                                Ok(hi_agent::LocalSkepticOutcome::Ready { endpoint, model_id }) => {
                                    format!(
                                        "local skeptic on → {model_id} at {endpoint} (used for /goal team reviews)"
                                    )
                                }
                                Ok(hi_agent::LocalSkepticOutcome::NoBackend) => {
                                    "no local backend (needs Apple-Silicon MLX or an NVIDIA GPU) — skeptic stays on the main model".to_string()
                                }
                                Ok(hi_agent::LocalSkepticOutcome::NeedsDownload { repo, dir }) => {
                                    format!(
                                        "model {repo} isn't cached — run `hi` in a plain terminal with `/config skeptic-local on` once to fetch it into {}, then retry here",
                                        dir.display()
                                    )
                                }
                                Err(err) => {
                                    format!(
                                        "couldn't start local skeptic: {err:#} — skeptic stays on the main model"
                                    )
                                }
                            };
                            self.push(Line::styled(msg, dim()));
                        } else {
                            let msg = if agent.disable_local_skeptic() {
                                "local skeptic off — review back on the main model"
                            } else {
                                "local skeptic was not on"
                            };
                            self.push(Line::styled(msg.to_string(), dim()));
                        }
                    }
                    ConfigArg::RsiShow => {
                        match agent.rsi_public_status().await {
                            Ok(status) => {
                                for line in status.lines() {
                                    self.push(Line::styled(line.to_string(), dim()));
                                }
                            }
                            Err(error) => self.push(Line::styled(
                                format!("RSI status unavailable: {error:#}"),
                                dim(),
                            )),
                        }
                        self.push(Line::styled(
                            "set with /config rsi on|off, /config rsi spend-limit <USD>, or /config rsi channel stable|beta"
                                .to_string(),
                            dim(),
                        ));
                    }
                    ConfigArg::Rsi(enabled) => {
                        let message = match agent.set_rsi_enabled_validated(enabled).await {
                            Ok(()) if enabled => "RSI candidate channel → on (saved). You confirmed repository/context upload, 30-day operational evidence retention, and training off without separate consent.".to_string(),
                            Ok(()) => "RSI candidate channel → off (saved)".to_string(),
                            Err(error) => format!("RSI config error: {error}"),
                        };
                        self.push(Line::styled(message, dim()));
                    }
                    ConfigArg::RsiSpendLimit(value) => {
                        let message = match agent.set_rsi_maximum_cost_microusd(value) {
                            Ok(()) => format!(
                                "RSI spend limit → {} per run (saved)",
                                hi_agent::command::format_usd_micros(value)
                            ),
                            Err(error) => format!("RSI config error: {error}"),
                        };
                        self.push(Line::styled(message, dim()));
                    }
                    ConfigArg::RsiChannel(channel) => {
                        let message = match agent.set_rsi_channel(channel) {
                            Ok(()) => format!("RSI channel → {} (saved)", channel.as_str()),
                            Err(error) => format!("RSI config error: {error}"),
                        };
                        self.push(Line::styled(message, dim()));
                    }
                    ConfigArg::Invalid(m) => {
                        self.push(Line::styled(m, Style::default().fg(Color::Yellow)));
                    }
                }
            }
            Command::Diff => {
                let out = hi_tools::working_tree_diff_in(agent.workspace_root()).await;
                let text = out.into_text().unwrap_or_else(|_| Text::from(out.clone()));
                for line in text.lines {
                    self.push(line);
                }
            }
            Command::Files => self.show_session_files(),
            Command::Review(_arg) => {
                // `/review` opens the full-screen diff review overlay (like
                // Ctrl-G). File-filtered review is via clicking a `✎ files
                // changed` transcript line.
                self.open_review(None);
            }
            Command::Commit => {
                let out = hi_tools::commit_in(agent.workspace_root()).await;
                for line in out.lines() {
                    self.push(Line::styled(format!("── {line} ──"), dim()));
                }
            }
            Command::Copy(arg) => self.copy(&arg),
            Command::Goal(arg) => self.handle_goal(agent, &arg),
            Command::Context => {
                let breakdown = agent.context_breakdown();
                for line in breakdown.lines() {
                    self.push(Line::styled(line.to_string(), dim()));
                }
            }
            Command::Skills => {
                let skills = hi_agent::list_skills();
                if skills.is_empty() {
                    self.push(Line::styled("no learned skills found".to_string(), dim()));
                } else {
                    self.push(Line::styled("learned skills:".to_string(), dim()));
                    for skill in skills {
                        self.push(Line::styled(
                            format!("  {}  [{}]  {}", skill.name, skill.scope, skill.description),
                            dim(),
                        ));
                    }
                }
            }
            // Handled in the event loop (async / runs a turn / needs config); never reach here.
            Command::Prompt(_)
            | Command::Moa(_)
            | Command::Compact(_)
            | Command::Retry
            | Command::Edit
            | Command::Undo
            | Command::Init
            | Command::Learn(_)
            | Command::Skill(_)
            | Command::Hf(_)
            | Command::Provider(_) => {}
            Command::Version => {
                self.push(Line::styled(format!("hi {}", hi_agent::VERSION), dim()));
            }
            Command::Mcp => {
                let Some(url) = self.mcp_url.clone() else {
                    self.push(Line::styled(
                        "no MCP URL configured for this provider".to_string(),
                        Style::default().fg(Color::Yellow),
                    ));
                    return;
                };
                self.push(Line::styled("contacting MCP endpoint…".to_string(), dim()));
                let result: Result<_, anyhow::Error> = async {
                    let client = hi_ai::PipeMcpClient::new(url, self.api_key.clone());
                    let (server, protocol) = client.server_info().await?;
                    let tools = client.tools_list().await?;
                    let models = client.list_models().await?;
                    Ok((server, protocol, tools, models))
                }
                .await;
                match result {
                    Ok((server, protocol, tools, models)) => {
                        let url = self.mcp_url.as_deref().unwrap_or("");
                        self.push(Line::styled(format!("mcp_url:  {url}"), dim()));
                        self.push(Line::styled(format!("server:   {server}"), dim()));
                        self.push(Line::styled(format!("protocol: {protocol}"), dim()));
                        self.push(Line::styled("tools:", dim()));
                        for tool in &tools {
                            let title = tool.title.as_deref().unwrap_or("");
                            if title.is_empty() {
                                self.push(Line::styled(format!("  {}", tool.name), dim()));
                            } else {
                                self.push(Line::styled(
                                    format!("  {}  - {}", tool.name, title),
                                    dim(),
                                ));
                            }
                        }
                        self.push(Line::styled(format!("models:   {}", models.len()), dim()));
                        if let Some(model) = models.iter().find(|m| m.id == self.model) {
                            let provider = model.provider_label.as_deref().unwrap_or("Pipe");
                            self.push(Line::styled(
                                format!("current:  {} · {}", model.id, provider),
                                dim(),
                            ));
                        }
                    }
                    Err(err) => {
                        self.push(Line::styled(
                            format!("mcp inspection failed: {err:#}"),
                            Style::default().fg(Color::Yellow),
                        ));
                    }
                }
            }
            Command::Lsp(arg) => {
                let arg = arg.trim();
                match arg {
                    "on" => {
                        agent.set_lsp_enabled(true);
                        self.push(Line::styled(
                            "LSP enabled — servers will warm up on first query.".to_string(),
                            dim(),
                        ));
                    }
                    "off" => {
                        agent.set_lsp_enabled(false);
                        self.push(Line::styled("LSP disabled.".to_string(), dim()));
                    }
                    _ => {
                        // `/lsp` or `/lsp status` — show enabled state plus
                        // per-language server availability and running state.
                        let report = agent.lsp_status_report();
                        for line in report.lines() {
                            self.push(Line::styled(line.to_string(), dim()));
                        }
                    }
                }
            }
            Command::Delegate(arg) => {
                let msg = match arg.trim() {
                    "on" => {
                        agent.set_write_subagents(true);
                        "delegate enabled — the model can hand a self-contained subtask to a \
                         worktree-isolated subagent whose changes are kept only if they verify."
                            .to_string()
                    }
                    "off" => {
                        agent.set_write_subagents(false);
                        "delegate disabled.".to_string()
                    }
                    _ => format!(
                        "delegate is {} (off by default; `/delegate on` to enable).",
                        if agent.write_subagents_enabled() {
                            "on"
                        } else {
                            "off"
                        }
                    ),
                };
                self.push(Line::styled(msg, dim()));
            }
            Command::Export(arg) => {
                let path = if arg.trim().is_empty() {
                    "transcript.md"
                } else {
                    arg.trim()
                };
                let content = agent.export_markdown();
                let count = agent
                    .messages()
                    .iter()
                    .filter(|m| m.role != hi_ai::Role::System)
                    .count();
                match std::fs::write(path, &content) {
                    Ok(()) => self.push(Line::styled(
                        format!("exported {count} messages to {path}"),
                        dim(),
                    )),
                    Err(err) => self.push(Line::styled(
                        format!("export failed: {err}"),
                        Style::default().fg(Color::Yellow),
                    )),
                }
            }
            Command::Sync(arg) => self.handle_sync_command(&arg).await,
            Command::Sessions(arg) => self.handle_sessions_command(agent, &arg).await,
            Command::Attach(arg) => self.handle_attach_command(&arg).await,
            Command::Daemon(arg) => self.handle_daemon_command(&arg).await,
            Command::Unknown(name) => {
                self.push(Line::styled(
                    format!("unknown command /{name}; try /help"),
                    dim(),
                ));
            }
            Command::Removed(msg) => {
                self.push(Line::styled(format!("/{msg}"), dim()));
            }
        }
        self.follow();
    }
}
