//! Input helpers for the TUI run loop: @mentions, normal mode, chords, shell escape.

use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use ansi_to_tui::IntoText;
use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use hi_agent::{Command, command};
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::*;
use ratatui::Terminal;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use tokio::sync::mpsc;

use crate::dispatch;
use crate::event::UiEvent;
use crate::render::dim;
use crate::{App, action};

/// Expand `@file` mentions in `prompt`: for each `@path` token (a path
/// relative to `root` that exists and is a file), append the file's contents
/// to the prompt under a labeled fenced block. This injects the file into
/// context without a separate `read` tool call. The original `@path` tokens
/// remain in the user-visible text. Files over 8 KiB are noted as "too large"
/// rather than dumped, and missing files are noted as "not found". `@@` is
/// treated as a literal `@`, not a mention.
pub(super) fn expand_file_mentions(prompt: &str, root: &std::path::Path) -> String {
    const MAX_FILE_BYTES: usize = 8 * 1024;
    let mut additions: Vec<String> = Vec::new();
    let mut chars = prompt.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '@' {
            continue;
        }
        // `@@` is a literal `@`, not a mention.
        if chars.peek() == Some(&'@') {
            chars.next();
            continue;
        }
        // Collect the path token: chars until whitespace or end.
        let mut path = String::new();
        while let Some(&pc) = chars.peek() {
            if pc.is_whitespace() {
                break;
            }
            path.push(pc);
            chars.next();
        }
        if path.is_empty() {
            continue;
        }
        let full = root.join(&path);
        if !full.is_file() {
            additions.push(format!("\n\n<file mention=\"{path}\">\nnot found\n</file>"));
            continue;
        }
        match std::fs::metadata(&full) {
            Ok(meta) if (meta.len() as usize) > MAX_FILE_BYTES => {
                additions.push(format!(
                    "\n\n<file mention=\"{path}\">\ntoo large ({} bytes; limit {})\n</file>",
                    meta.len(),
                    MAX_FILE_BYTES
                ));
            }
            Ok(_) => match std::fs::read_to_string(&full) {
                Ok(contents) => {
                    additions.push(format!(
                        "\n\n<file mention=\"{path}\">\n{contents}\n</file>"
                    ));
                }
                Err(err) => {
                    additions.push(format!(
                        "\n\n<file mention=\"{path}\">\nread error: {err}\n</file>"
                    ));
                }
            },
            Err(err) => {
                additions.push(format!(
                    "\n\n<file mention=\"{path}\">\nread error: {err}\n</file>"
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

/// Handle a key in vim-style normal mode (Esc on empty input). Modal
/// scroll/search/copy without leaving the keyboard. `i`, `q`, or Esc returns
/// to insert mode; `j`/`k` scroll; `u`/`d` half-page; `g`/`G` top/bottom; `/`
/// starts a transcript search; `n`/`N` jump to next/previous match; `y` copies
/// the last code block (mirroring Ctrl-Y).
pub(super) fn handle_normal_mode(app: &mut App, key: &KeyEvent) {
    // If we're collecting a search query, handle search-mode keys.
    if let Some(search_slot) = app.mode.normal_search_mut() {
        match key.code {
            KeyCode::Enter => {
                let query = search_slot.take().unwrap_or_default();
                if !query.is_empty() {
                    app.last_search = Some(query.clone());
                    search_transcript(app, &query, 1);
                }
                // Stay in Normal without an active search buffer.
                app.mode = crate::mode::UiMode::Normal { search: None };
            }
            KeyCode::Esc => {
                app.mode = crate::mode::UiMode::Normal { search: None };
            }
            KeyCode::Backspace => {
                if let Some(q) = search_slot {
                    if q.is_empty() {
                        app.mode = crate::mode::UiMode::Normal { search: None };
                    } else {
                        q.pop();
                    }
                }
            }
            KeyCode::Char(c) => {
                if let Some(q) = search_slot {
                    q.push(c);
                }
            }
            _ => {}
        }
        return;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        // Exit to insert mode.
        KeyCode::Char('i') | KeyCode::Char('q') | KeyCode::Esc => {
            app.mode.to_insert();
        }
        // Scroll one line.
        KeyCode::Char('j') | KeyCode::Down => app.scroll_down(1),
        KeyCode::Char('k') | KeyCode::Up => app.scroll_up(1),
        // Half-page scroll.
        KeyCode::Char('d') | KeyCode::PageDown => app.scroll_down(10),
        KeyCode::Char('u') | KeyCode::PageUp => app.scroll_up(10),
        // Top / bottom.
        KeyCode::Char('g') => app.scroll_to_top(),
        KeyCode::Char('G') => app.scroll_to_bottom(),
        // Search the transcript.
        KeyCode::Char('/') => {
            app.mode = crate::mode::UiMode::Normal {
                search: Some(String::new()),
            };
        }
        // Next / previous search match.
        KeyCode::Char('n') => {
            if let Some(q) = app.last_search.clone() {
                search_transcript(app, &q, 1);
            }
        }
        KeyCode::Char('N') => {
            if let Some(q) = app.last_search.clone() {
                search_transcript(app, &q, -1);
            }
        }
        // Copy last code block (mirrors Ctrl-Y).
        KeyCode::Char('y') => app.copy_last_code_block(),
        // Ctrl-C still works to interrupt.
        KeyCode::Char('c') if ctrl => {
            app.mode.to_insert();
        }
        _ => {}
    }
}

/// Shared palette + action-dispatch pipeline for idle and in-turn key loops.
///
/// Returns `Some(outcome)` when the key was fully consumed by palette or
/// dispatch; `None` when the caller should continue with edit/submit handling.
pub(super) enum ChordPipeline {
    /// Redraw and wait for the next key.
    Continue,
    /// Open the command palette (caller sets `app.palette`).
    OpenPalette,
    /// Palette accepted a command; idle loop submits/edits, drive loop queues.
    PaletteAccept(String),
}

pub(super) fn run_chord_pipeline(app: &mut App, key: &KeyEvent) -> Option<ChordPipeline> {
    use crate::domain::OverlayDomain;
    use crate::dispatch::DispatchResult;

    if OverlayDomain::palette_open(app) {
        let outcome = app.palette.as_mut().unwrap().handle_key(key);
        return Some(match outcome {
            crate::palette::PaletteOutcome::Continue => ChordPipeline::Continue,
            crate::palette::PaletteOutcome::Closed => {
                app.palette = None;
                ChordPipeline::Continue
            }
            crate::palette::PaletteOutcome::Accept(cmd) => {
                app.palette = None;
                ChordPipeline::PaletteAccept(cmd)
            }
        });
    }

    // Normal mode (including `/` search typing): actions first, then specialized handler.
    if app.mode.is_normal() {
        match app.dispatch_key(key) {
            DispatchResult::Handled => return Some(ChordPipeline::Continue),
            DispatchResult::OpenPalette => return Some(ChordPipeline::OpenPalette),
            DispatchResult::Fallthrough => {
                handle_normal_mode(app, key);
                return Some(ChordPipeline::Continue);
            }
        }
    }

    match app.dispatch_key(key) {
        DispatchResult::Handled => Some(ChordPipeline::Continue),
        DispatchResult::OpenPalette => Some(ChordPipeline::OpenPalette),
        DispatchResult::Fallthrough => None,
    }
}

/// Search the transcript for `query` and scroll to the next (dir=1) or
/// previous (dir=-1) match relative to the current scroll position. Case-
/// insensitive. If no match is found in the given direction, stays put.
pub(crate) fn search_transcript(app: &mut App, query: &str, dir: i32) {
    let text = app.transcript_text();
    if text.is_empty() || query.is_empty() {
        return;
    }
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len() as i32;
    // Use `scroll` (the live scroll offset) rather than `view_scroll` (a
    // cached copy updated only during render), so search works without a
    // render between calls.
    let cur = app.scroll as i32;
    let query_lower = query.to_lowercase();
    let search_from = if dir > 0 { cur + 1 } else { cur - 1 };
    let found = if dir > 0 {
        (search_from..total).find(|&i| {
            lines
                .get(i as usize)
                .map(|l| l.to_lowercase().contains(&query_lower))
                .unwrap_or(false)
        })
    } else {
        (0..=search_from.max(0)).rev().find(|&i| {
            lines
                .get(i as usize)
                .map(|l| l.to_lowercase().contains(&query_lower))
                .unwrap_or(false)
        })
    };
    if let Some(line_idx) = found {
        app.scroll_to(line_idx as u16);
    }
}

/// Find the line index of the next (dir=1) or previous (dir=-1) `@@` hunk
/// header in `diff`, starting the search from `from`. Used by the full-screen
/// diff review overlay's n/p navigation. Clamps to the diff bounds; returns
/// `from` unchanged if there's no hunk in the requested direction.
pub(crate) fn review_next_hunk(diff: Option<&str>, from: usize, dir: i32) -> usize {
    let Some(diff) = diff else { return from };
    let lines: Vec<&str> = diff.lines().collect();
    if lines.is_empty() {
        return from;
    }
    if dir > 0 {
        // Next hunk: first `@@` line strictly after `from`.
        (from + 1..lines.len())
            .find(|&i| lines[i].starts_with("@@"))
            .unwrap_or(lines.len().saturating_sub(1))
    } else {
        // Previous hunk: last `@@` line strictly before `from`.
        (0..from.min(lines.len()))
            .rev()
            .find(|&i| lines[i].starts_with("@@"))
            .unwrap_or(0)
    }
}

/// Run a `!cmd` shell-escape asynchronously so a slow command doesn't freeze
/// the TUI. Pushes a `$ command` header, then races the command's output
/// against input events so Esc/Ctrl-C cancels. The result (or cancellation
/// notice) is pushed to the transcript. Output is capped at 200 lines.
pub(super) async fn run_shell_escape_async(
    app: &mut App,
    command: &str,
    input: &mut mpsc::UnboundedReceiver<Event>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    let command = command.trim();
    if command.is_empty() {
        return Ok(());
    }
    // Header line: `⏺ $ <command>` so it reads like a shell invocation.
    app.push(crate::render::accent_line(
        crate::theme::theme().accent_goal,
        format!("$ {command}"),
        Style::default().fg(crate::theme::theme().accent_goal),
    ));
    app.push(Line::styled("running… (Esc to cancel)".to_string(), dim()));
    app.follow();

    // Spawn the command asynchronously. We keep the `Child` handle so we can
    // `kill()` it on cancellation — `wait_with_output` would consume the child
    // and leave it running if the user hits Esc. Instead we read stdout/stderr
    // concurrently and wait for exit ourselves.
    let spawn = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(&app.workspace_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();
    let cancelled = match spawn {
        Ok(mut child) => {
            let mut stdout = child.stdout.take().expect("piped stdout");
            let mut stderr = child.stderr.take().expect("piped stderr");
            // Read stdout and stderr concurrently into buffers.
            let collect = tokio::spawn(async move {
                use tokio::io::AsyncReadExt;
                let mut out = Vec::new();
                let mut err = Vec::new();
                let _ = stdout.read_to_end(&mut out).await;
                let _ = stderr.read_to_end(&mut err).await;
                (out, err)
            });
            tokio::pin!(collect);
            // Race the command's exit against Esc/Ctrl-C, redrawing while we wait.
            let mut cancelled = false;
            let mut ticker = tokio::time::interval(std::time::Duration::from_millis(80));
            loop {
                terminal.draw(|f| app.render(f))?;
                tokio::select! {
                    // `child.wait()` resolves when the process exits; we race it
                    // alongside the output read so both are ready by the time
                    // we render.
                    status = child.wait() => {
                        // Drain any remaining output the read task captured.
                        let (out, err) = (&mut collect).await.unwrap_or_default();
                        // Remove the "running…" line before pushing the real output.
                        if app.transcript.last().map(|e| e.text()).as_deref() == Some("running… (Esc to cancel)") {
                            app.transcript.pop();
                        }
                        match status {
                            Ok(_) => {
                                let mut combined = String::from_utf8_lossy(&out).into_owned();
                                if !err.is_empty() {
                                    let e = String::from_utf8_lossy(&err);
                                    if !combined.is_empty() {
                                        combined.push('\n');
                                    }
                                    combined.push_str(&e);
                                }
                                push_shell_output(app, &combined);
                            }
                            Err(err) => {
                                app.push(Line::styled(
                                    format!("failed to run: {err}"),
                                    Style::default().fg(crate::theme::theme().warning),
                                ));
                            }
                        }
                        break;
                    }
                    _ = ticker.tick() => {
                        app.spinner = app.spinner.wrapping_add(1);
                    }
                    maybe = input.recv() => {
                        match maybe {
                            Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                                if matches!(key.code, KeyCode::Esc)
                                    || (ctrl && matches!(key.code, KeyCode::Char('c')))
                                {
                                    cancelled = true;
                                    // Kill the child so a cancelled `!cargo build`
                                    // actually stops, instead of running in the
                                    // background. Dropping the collect task detaches
                                    // it; the kill ensures the process is gone.
                                    let _ = child.kill().await;
                                    break;
                                }
                            }
                            Some(_) => {}
                            None => return Ok(()),
                        }
                    }
                }
            }
            cancelled
        }
        Err(err) => {
            if app.transcript.last().map(|e| e.text()).as_deref()
                == Some("running… (Esc to cancel)")
            {
                app.transcript.pop();
            }
            app.push(Line::styled(
                format!("failed to run: {err}"),
                Style::default().fg(crate::theme::theme().warning),
            ));
            false
        }
    };
    if cancelled {
        // Remove the "running…" line and note cancellation.
        if app.transcript.last().map(|e| e.text()).as_deref() == Some("running… (Esc to cancel)")
        {
            app.transcript.pop();
        }
        app.push(Line::styled("(cancelled)", dim()));
    }
    app.follow();
    Ok(())
}

/// Push shell-escape output into the transcript as foldable tool-output lines,
/// capped at 200 lines (with a "… (N more lines)" notice when truncated).
pub(super) fn push_shell_output(app: &mut App, body: &str) {
    const MAX_LINES: usize = 200;
    let lines: Vec<&str> = body.lines().collect();
    let display = if lines.len() > MAX_LINES {
        let mut capped = lines[..MAX_LINES].join("\n");
        capped.push_str(&format!("\n… ({} more lines)", lines.len() - MAX_LINES));
        capped
    } else {
        body.to_string()
    };
    let text = display
        .into_text()
        .unwrap_or_else(|_| Text::from(body.to_string()));
    let gutter = crate::render::gutter(crate::theme::theme().gray_dim);
    let lines: Vec<Line<'static>> = text
        .lines
        .into_iter()
        .map(|mut line| {
            line.spans.insert(0, gutter.clone());
            line
        })
        .collect();
    for line in lines {
        app.transcript.push(crate::TranscriptEntry::ToolOutput {
            body: vec![line],
            expanded: false,
        });
    }
    app.bump_transcript();
    app.cap_transcript();
}


