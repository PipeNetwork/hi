//! The TUI event loop: `run` (entry point that sets up the terminal, spawns
//! the agent turn behind a channel, and drives the render loop) and `drive`
//! (the per-event state machine that routes crossterm events to `App`).

use std::io;
use std::io::IsTerminal;
use std::time::{Duration, Instant};

use ansi_to_tui::IntoText;
use anyhow::{Context, Result};
use crossterm::event::{
    EnableBracketedPaste, EnableFocusChange, EnableMouseCapture, Event, EventStream, KeyCode,
    KeyEvent, KeyEventKind, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, enable_raw_mode};
use futures_util::StreamExt;
use hi_agent::{Agent, Command, CompactionKind, command};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Style};
use ratatui::text::{Line, Text};
use tokio::sync::mpsc;

use crate::event::{ChannelUi, ConfirmationControl, Restore, UiEvent};
use crate::input::HistorySearch;
use crate::model_picker::ModelPicker;
use crate::provider_form;
use crate::provider_picker;
use crate::render::dim;
use crate::{
    App, MlxProfileSwitcher, ProfileInfo, ProfileLoader, ProfileRemover, ProfileResolver,
    ProfileSaver, TICK, TurnState, apply_metadata, splash_lines, watchdog_stuck_timeout,
};

/// Expand `@file` mentions in `prompt`: for each `@path` token (a path
/// relative to `root` that exists and is a file), append the file's contents
/// to the prompt under a labeled fenced block. This injects the file into
/// context without a separate `read` tool call. The original `@path` tokens
/// remain in the user-visible text. Files over 8 KiB are noted as "too large"
/// rather than dumped, and missing files are noted as "not found". `@@` is
/// treated as a literal `@`, not a mention.
fn expand_file_mentions(prompt: &str, root: &std::path::Path) -> String {
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
fn handle_normal_mode(app: &mut App, key: &KeyEvent) {
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
async fn run_shell_escape_async(
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
fn push_shell_output(app: &mut App, body: &str) {
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

/// Run the full-screen TUI until the user quits. `history_path`, if given, is
/// the file used to persist input history across sessions (shared with the
/// plain REPL). `profiles` is the list of configured profiles (for `/provider`
/// with no arg); `resolver` resolves a name to a built provider at runtime.
/// Drop guard that stops any auto-managed local skeptic server when the TUI
/// session ends, covering every `return`/`break` exit path in [`run`]. The
/// server registry only holds skeptic servers, so a blanket kill is correct.
struct LocalServerGuard;

impl Drop for LocalServerGuard {
    fn drop(&mut self) {
        hi_tools::stop_all_local_servers();
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    agent: &mut Agent,
    provider: &str,
    base_url: &str,
    model: &str,
    history_path: Option<std::path::PathBuf>,
    auto_memory: bool,
    profiles: Vec<ProfileInfo>,
    active_profile: Option<String>,
    resolver: ProfileResolver,
    saver: ProfileSaver,
    loader: ProfileLoader,
    remover: ProfileRemover,
    mlx_switcher: MlxProfileSwitcher,
    session_remember: Option<crate::SessionRemember>,
    resume_summary: Option<String>,
    mcp_url: Option<String>,
    api_key: String,
    fleet_launcher: crate::FleetLauncher,
    remote_event_tap: Option<crate::RemoteEventTap>,
    remote_flush_callback: Option<crate::RemoteFlushCallback>,
    sync_config: Option<crate::SyncConfig>,
    sync_session_id: Option<String>,
    session_lister: Option<crate::SessionLister>,
    session_switcher: Option<crate::SessionSwitcher>,
    session_renamer: Option<crate::SessionRenamer>,
    sync_control: Option<crate::SyncControl>,
) -> Result<()> {
    if !io::stdin().is_terminal() {
        anyhow::bail!("TUI requires an interactive stdin");
    }

    enable_raw_mode().context("entering raw mode")?;
    // Install immediately after raw mode so any later startup error restores
    // the terminal before main falls back to plain mode.
    let _restore = Restore;
    // Tear down any auto-managed `/goal` skeptic server on every exit path.
    let _local_servers = LocalServerGuard;
    execute!(io::stdout(), EnterAlternateScreen).context("entering alternate screen")?;
    // Bracketed paste: the terminal wraps a paste so it arrives as one
    // Event::Paste instead of per-line Enter keys (which would submit each line).
    let _ = execute!(io::stdout(), EnableBracketedPaste);
    // Focus reporting: lets us tell when you've switched away, so a finished turn
    // can ping you only when you're not looking. Harmless if unsupported.
    let _ = execute!(io::stdout(), EnableFocusChange);
    // Mouse capture enables wheel scrolling inside the transcript. Most
    // terminals retain native text selection while Shift is held.
    let _ = execute!(io::stdout(), EnableMouseCapture);
    let mut terminal =
        Terminal::new(CrosstermBackend::new(io::stdout())).context("creating terminal")?;

    let mut app = App::new(
        provider,
        model,
        profiles,
        active_profile,
        resolver,
        saver,
        loader,
        remover,
        mlx_switcher,
        mcp_url,
        api_key,
    );
    app.session_remember = session_remember;
    app.workspace_root = agent.workspace_root().to_path_buf();
    // Load persistent input history (`.hi/history`) now that the workspace root
    // is known, so Ctrl-R searches across sessions, not just the current one.
    app.input.load_history(&app.workspace_root);
    app.plan = agent.current_plan().to_vec();
    app.resume_goal_drive(agent);
    app.sync_active = sync_config.is_some();
    app.sync_config = sync_config;
    app.sync_session_id = sync_session_id;
    app.session_lister = session_lister;
    app.session_switcher = session_switcher;
    app.session_renamer = session_renamer;
    app.sync_control = sync_control;
    app.remote_event_tap = remote_event_tap;
    app.remote_flush_callback = remote_flush_callback;
    if app.sync_config.is_some() {
        app.sync_http = Some(
            reqwest::Client::builder()
                // Session listing and renaming run on the TUI command loop;
                // bound outages so the interface cannot appear frozen for
                // half a minute when portal sync is unreachable.
                .connect_timeout(std::time::Duration::from_secs(3))
                .timeout(std::time::Duration::from_secs(8))
                .http1_only()
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        );
    }
    // Seed the context-fill gauge with the model's window so it reads 0% before
    // the first turn (it refreshes from real usage after each round).
    app.context_window = None;
    // Load the on-disk /models cache so model metadata (window/price)
    // applies instantly at startup, without blocking on the network. The live
    // fetch still runs in the background and refreshes this; the cache just
    // covers the cold-start gap so the UI never looks stalled.
    let models_cache_key = hi_ai::cache_key(provider, base_url);
    if let Some(cached) = hi_ai::load_cache(&models_cache_key).await {
        app.model_ids = cached.iter().map(|m| m.id.clone()).collect();
        app.model_ids.sort();
        app.served = cached.into_iter().map(|m| (m.id.clone(), m)).collect();
        let model_id = app.model.clone();
        app.apply_model(agent, &model_id);
    }
    if let Some(path) = &history_path
        && let Ok(text) = std::fs::read_to_string(path)
    {
        app.input.history = text
            .lines()
            .map(str::to_string)
            .filter(|l| !l.trim().is_empty())
            .collect();
    }
    {
        // The Pipenetwork.ai landing banner as the first transcript lines —
        // it sits at the top of the transcript and scrolls up naturally as the
        // session grows, like Claude's landing. Pushed before the usage hint.
        for line in splash_lines(provider, model, app.context_window) {
            app.push(line);
        }
        // A one-line usage hint as the next transcript line. The provider and
        // model already appear in the border title (top of the box), so we don't
        // repeat them here — that would render as a duplicate header line.
        let ctx = app
            .context_window
            .map(|w| format!(" · {w} token window"))
            .unwrap_or_default();
        // When resuming, show what we're walking back into before the hint.
        if let Some(summary) = &resume_summary {
            app.push(Line::styled(summary.clone(), dim()));
        }
        app.push(Line::styled(
            format!(
                "Enter to send · Alt-Enter for a newline · Ctrl-C interrupts/double exits · Ctrl-T shows reasoning · Ctrl-O expands tool output · Ctrl-D toggles diff · /theme to restyle · /help for all commands{ctx}.",
            ),
            dim(),
        ));
    }
    // Read terminal events in a dedicated task and forward them over a channel.
    // A channel receiver is fully cancel-safe, so the per-tick redraws in the
    // loops below can't drop or delay a keystroke — which repeatedly cancelling
    // an `EventStream::next()` future inside `select!` can.
    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<Event>();
    tokio::spawn(async move {
        let mut events = EventStream::new();
        loop {
            match events.next().await {
                Some(Ok(event)) => {
                    if input_tx.send(event).is_err() {
                        break; // main loop gone — stop reading
                    }
                }
                Some(Err(_)) => continue, // skip a malformed event, keep reading
                None => break,            // stdin closed
            }
        }
    });
    let mut ticker = tokio::time::interval(TICK);
    // The /loop manager: timers + firings in a background task (it never
    // touches the Agent); persisted loops re-arm now. Results drain on ticks.
    // Only start it if we can take the per-project fire-lock — otherwise a
    // background daemon (or another TUI) already owns firing, and starting a
    // second manager would double-fire every loop. Held for the session.
    let fleet_launcher = std::sync::Arc::new(fleet_launcher);
    let _fire_lock;
    match &fleet_launcher.loops_file {
        Some(lf) => {
            let lp = crate::lock::lock_path(lf);
            match crate::lock::try_acquire(&lp) {
                Some(guard) => {
                    _fire_lock = Some(guard);
                    app.loops = Some(crate::loops::start(
                        fleet_launcher.clone(),
                        fleet_launcher.loops_file.clone(),
                    ));
                }
                None => {
                    _fire_lock = None;
                    let who = crate::lock::live_holder(&lp)
                        .map(|p| format!(" (pid {p})"))
                        .unwrap_or_default();
                    app.push(Line::styled(
                        format!(
                            "⟳ loops are firing in a daemon{who} — /digest shows results; stop it to manage loops here"
                        ),
                        Style::default().fg(crate::theme::theme().accent_system),
                    ));
                }
            }
        }
        None => {
            // No persisted loops location: run the manager unlocked.
            _fire_lock = None;
            app.loops = Some(crate::loops::start(
                fleet_launcher.clone(),
                fleet_launcher.loops_file.clone(),
            ));
        }
    }
    // "While you were away": if loops noticed changes since you last looked,
    // nudge you toward /digest (which shows and then clears the marker).
    if let Some(lf) = &fleet_launcher.loops_file {
        let entries = crate::activity::load(&crate::activity::activity_path(lf));
        let seen = crate::activity::load_seen(&crate::activity::seen_path(lf));
        let fresh = entries.iter().filter(|e| e.at_ms > seen).count();
        if fresh > 0 {
            app.push(Line::styled(
                format!("⟳ {fresh} loop change(s) since you last looked — /digest to review"),
                Style::default().fg(crate::theme::theme().accent_system),
            ));
        }
    }
    // Startup metadata fetch: race the live `/models` fetch against the first
    // keystroke, with a spinner ticking and the screen redrawing each tick so
    // the UI never looks stalled. The on-disk cache already applied instantly
    // above; this just refreshes it. The fetch future is pinned locally (not
    // spawned — `Agent` isn't `Send`) and dropped before the main loop so its
    // borrow of `agent` doesn't block mutable uses during turns. A first input
    // event that wins the race is buffered for the main loop to process.
    let mut first_event: Option<Event> = None;
    let mut meta_result: Option<Result<Vec<hi_ai::ServedModel>>> = None;
    if app.context_window.is_none() {
        let meta_fut = agent.list_models();
        tokio::pin!(meta_fut);
        loop {
            terminal.draw(|f| app.render(f))?;
            tokio::select! {
                maybe = input_rx.recv() => {
                    let Some(event) = maybe else {
                        return Ok(());
                    };
                    first_event = Some(event);
                    break;
                }
                _ = ticker.tick() => {
                    app.spinner = app.spinner.wrapping_add(1);
                }
                result = &mut meta_fut => {
                    meta_result = Some(result);
                    break;
                }
            }
        }
        // `meta_fut` (and its borrow of `agent`) is dropped at the end of this
        // block, so `apply_metadata` can take `&mut agent` below.
    }
    if let Some(result) = meta_result {
        apply_metadata(&mut app, agent, &result, &models_cache_key);
    }

    let mut hf_state = hi_tools::HfCommandState::default();

    'session: loop {
        // Run a queued command first (typed while the previous turn ran);
        // otherwise edit the input line until the user submits.
        let line = match app.queue.pop_front() {
            Some(queued) => queued,
            None => 'input: loop {
                terminal.draw(|f| app.render(f))?;
                // The startup metadata fetch already completed (or was skipped)
                // before the main loop, so this is a plain input wait. The
                // spinner still ticks during turns (see the working branch).
                let event = match first_event.take() {
                    Some(e) => e,
                    None => {
                        // Race input against the quit-notice deadline (if armed)
                        // so the "Press Ctrl-C again to exit" notice auto-clears
                        // after 1.8s even with no further input.
                        let next = app.quit_notice;
                        let event = if let Some(deadline) = next {
                            let remaining = deadline.saturating_duration_since(Instant::now());
                            tokio::select! {
                                maybe = input_rx.recv() => maybe,
                                _ = tokio::time::sleep(remaining) => {
                                    app.quit_notice = None;
                                    continue 'input; // redraw without the notice
                                }
                            }
                        } else {
                            tokio::select! {
                                maybe = input_rx.recv() => maybe,
                                _ = ticker.tick() => {
                                    // Loop firings land while you're idle too.
                                    app.spinner = app.spinner.wrapping_add(1);
                                    app.drain_loops();
                                    // Follow OS light/dark when theme = auto.
                                    // ~5s cadence (40 × 120ms tick); a no-op for
                                    // fixed modes, so it only queries the OS on
                                    // auto. The next redraw picks up any change.
                                    if app.spinner.is_multiple_of(40) {
                                        crate::theme::poll_auto_appearance();
                                    }
                                    continue 'input;
                                }
                            }
                        };
                        let Some(event) = event else { break 'session };
                        event
                    }
                };
                match event {
                    Event::Mouse(mouse) => app.handle_mouse(mouse),
                    // A paste arrives as one event. Route it to whichever input
                    // surface is active: the provider form (its current field),
                    // or the main input line. Without this, a paste while the
                    // form is open silently went into the hidden main input.
                    Event::Paste(text) => {
                        if let Some(form) = app.provider_form.as_mut() {
                            form.insert_str(&text);
                        } else {
                            app.input.insert_str(&text);
                        }
                    }
                    // While the model picker is open, keys drive it.
                    // `/provider` selector. Enter queues `/provider <name>` so
                    // the switch runs through exactly the same path as typing
                    // it — one implementation, not two that can drift.
                    Event::Key(key)
                        if key.kind == KeyEventKind::Press && app.provider_picker.is_some() =>
                    {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        match key.code {
                            KeyCode::Esc => app.provider_picker = None,
                            KeyCode::Char('c') if ctrl => app.provider_picker = None,
                            KeyCode::Up => app.provider_picker.as_mut().unwrap().up(),
                            KeyCode::Down => app.provider_picker.as_mut().unwrap().down(),
                            KeyCode::PageUp => app.provider_picker.as_mut().unwrap().page_up(),
                            KeyCode::PageDown => app.provider_picker.as_mut().unwrap().page_down(),
                            KeyCode::Backspace => app.provider_picker.as_mut().unwrap().backspace(),
                            KeyCode::Enter => {
                                let chosen = app
                                    .provider_picker
                                    .as_ref()
                                    .and_then(|p| p.current_name())
                                    .map(str::to_string);
                                app.provider_picker = None;
                                if let Some(name) = chosen {
                                    app.queue.push_back(format!("/provider {name}"));
                                }
                            }
                            KeyCode::Char(c) if !ctrl => {
                                app.provider_picker.as_mut().unwrap().insert(c)
                            }
                            _ => {}
                        }
                        continue;
                    }
                    Event::Key(key) if key.kind == KeyEventKind::Press && app.picker.is_some() => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        if app.session_picker {
                            let selected = app.picker.as_ref().and_then(|picker| {
                                picker
                                    .matches
                                    .get(picker.selected)
                                    .and_then(|index| picker.all.get(*index))
                                    .cloned()
                            });
                            if app.session_picker_searching {
                                match key.code {
                                    KeyCode::Esc => app.session_picker_searching = false,
                                    KeyCode::Up => app.picker.as_mut().unwrap().up(),
                                    KeyCode::Down => app.picker.as_mut().unwrap().down(),
                                    KeyCode::PageUp => app.picker.as_mut().unwrap().page_up(),
                                    KeyCode::PageDown => app.picker.as_mut().unwrap().page_down(),
                                    KeyCode::Backspace => app.picker.as_mut().unwrap().backspace(),
                                    KeyCode::Enter => app.session_picker_searching = false,
                                    KeyCode::Char(c) if !ctrl => {
                                        app.picker.as_mut().unwrap().insert(c)
                                    }
                                    _ => {}
                                }
                            } else {
                                match key.code {
                                    KeyCode::Enter => {
                                        app.picker = None;
                                        app.session_picker = false;
                                        if let Some(id) = selected {
                                            app.switch_session(agent, &id).await;
                                        }
                                    }
                                    KeyCode::Esc => {
                                        app.picker = None;
                                        app.session_picker = false;
                                    }
                                    KeyCode::Char('c') if ctrl => {
                                        app.picker = None;
                                        app.session_picker = false;
                                    }
                                    KeyCode::Char('/') => {
                                        app.session_picker_searching = true;
                                    }
                                    KeyCode::Char('r') => {
                                        if let Some(id) = selected {
                                            app.input.set(&format!("/sessions rename {id} "));
                                            app.picker = None;
                                            app.session_picker = false;
                                        }
                                    }
                                    KeyCode::Char('f') => {
                                        if let Some(id) = selected {
                                            let flags = app
                                                .session_catalog_flags
                                                .get(&id)
                                                .copied()
                                                .unwrap_or_default();
                                            let next = !flags.0;
                                            app.patch_session(
                                                &id,
                                                serde_json::json!({"favorite": next}),
                                            )
                                            .await;
                                            app.session_catalog_flags.insert(id, (next, flags.1));
                                        }
                                    }
                                    KeyCode::Char('a') => {
                                        if let Some(id) = selected {
                                            let flags = app
                                                .session_catalog_flags
                                                .get(&id)
                                                .copied()
                                                .unwrap_or_default();
                                            let next = !flags.1;
                                            app.patch_session(
                                                &id,
                                                serde_json::json!({"archived": next}),
                                            )
                                            .await;
                                            app.session_catalog_flags.insert(id, (flags.0, next));
                                        }
                                    }
                                    KeyCode::Char('d') => {
                                        if let Some(id) = selected {
                                            if app.session_delete_pending.as_deref() == Some(&id) {
                                                app.picker = None;
                                                app.session_picker = false;
                                                app.session_delete_pending = None;
                                                app.delete_session(&id).await;
                                            } else {
                                                app.session_delete_pending = Some(id.clone());
                                                app.push(Line::styled(
                                                    format!("press d again to permanently delete session {id}"),
                                                    Style::default().fg(crate::theme::theme().warning),
                                                ));
                                            }
                                        }
                                    }
                                    code => {
                                        let picker = app.picker.as_mut().unwrap();
                                        match code {
                                            KeyCode::Up => picker.up(),
                                            KeyCode::Down => picker.down(),
                                            KeyCode::PageUp => picker.page_up(),
                                            KeyCode::PageDown => picker.page_down(),
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        } else {
                            match key.code {
                                KeyCode::Enter => app.pick_model(agent),
                                KeyCode::Esc => app.picker = None,
                                KeyCode::Char('c') if ctrl => app.picker = None,
                                code => {
                                    let picker = app.picker.as_mut().unwrap();
                                    match code {
                                        KeyCode::Up => picker.up(),
                                        KeyCode::Down => picker.down(),
                                        KeyCode::PageUp => picker.page_up(),
                                        KeyCode::PageDown => picker.page_down(),
                                        KeyCode::Backspace => picker.backspace(),
                                        KeyCode::Char(c) if !ctrl => picker.insert(c),
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    // Provider form: keystrokes go to the form, not the input.
                    Event::Key(key)
                        if key.kind == KeyEventKind::Press && app.provider_form.is_some() =>
                    {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
                        match key.code {
                            KeyCode::Esc => app.provider_form = None,
                            KeyCode::Char('c') if ctrl => app.provider_form = None,
                            KeyCode::Enter => {
                                // Submit the form.
                                let form = app.provider_form.as_ref().unwrap();
                                if let Some(data) = form.data() {
                                    // When adding (not editing), reject a name that
                                    // already exists to prevent silent clobbering.
                                    if !form.editing
                                        && app.profiles.iter().any(|p| p.name == data.name)
                                    {
                                        app.push(Line::styled(
                                            format!(
                                                "a profile '{}' already exists — use /provider edit {} to modify it",
                                                data.name, data.name
                                            ),
                                            Style::default().fg(crate::theme::theme().warning),
                                        ));
                                    } else {
                                        match (app.saver)(&data) {
                                            Ok(updated) => {
                                                app.profiles = updated;
                                                app.push(Line::styled(
                                                    format!("saved profile '{}'", data.name),
                                                    dim(),
                                                ));
                                                app.provider_form = None;
                                            }
                                            Err(err) => {
                                                app.push(Line::styled(
                                                    format!("save failed: {err:#}"),
                                                    Style::default().fg(crate::theme::theme().warning),
                                                ));
                                            }
                                        }
                                    }
                                } else {
                                    app.push(Line::styled(
                                        "name is required".to_string(),
                                        Style::default().fg(crate::theme::theme().warning),
                                    ));
                                }
                            }
                            KeyCode::Tab => {
                                let form = app.provider_form.as_mut().unwrap();
                                form.next_field();
                            }
                            KeyCode::BackTab => {
                                let form = app.provider_form.as_mut().unwrap();
                                form.prev_field();
                            }
                            // Up/Down cycle the provider picker. They are the keys
                            // people reach for on a list, and unlike Left/Right
                            // they don't collide with editing the text fields.
                            KeyCode::Up if !shift => {
                                app.provider_form.as_mut().unwrap().cycle_provider_prev();
                            }
                            KeyCode::Down if !shift => {
                                app.provider_form.as_mut().unwrap().cycle_provider();
                            }
                            // Left/Right keep their ordinary meaning: move the
                            // cursor inside the field being typed into.
                            KeyCode::Left if !shift => {
                                app.provider_form.as_mut().unwrap().cursor_left();
                            }
                            KeyCode::Right if !shift => {
                                app.provider_form.as_mut().unwrap().cursor_right();
                            }
                            KeyCode::Backspace => {
                                app.provider_form.as_mut().unwrap().backspace();
                            }
                            KeyCode::Char('u') if ctrl => {
                                app.provider_form.as_mut().unwrap().clear_field();
                            }
                            KeyCode::Char(c) if !ctrl => {
                                app.provider_form.as_mut().unwrap().insert(c);
                            }
                            _ => {}
                        }
                    }
                    // When the `/`-command menu is open, navigation/accept keys
                    // drive it; anything else edits the input and re-syncs it.
                    Event::Key(key)
                        if key.kind == KeyEventKind::Press && app.completion.is_some() =>
                    {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        let history_search_was_active = app.mode.is_history_search();
                        match key.code {
                            KeyCode::Char('c') if ctrl => app.input.clear(),
                            KeyCode::Esc => app.completion = None,
                            KeyCode::Up => app.completion_move(-1),
                            KeyCode::Down => app.completion_move(1),
                            KeyCode::Tab => {
                                // Completing a command that takes arguments fills
                                // `/name ` — re-sync so its value menu opens next.
                                app.accept_completion(false);
                                app.sync_completion();
                            }
                            KeyCode::Enter => {
                                if let Some(line) = app.accept_completion(true) {
                                    break 'input line;
                                }
                            }
                            _ => {
                                if let Some(line) = app.edit_key(&key) {
                                    break 'input line;
                                }
                                app.sync_completion_after_edit_key(&key, history_search_was_active);
                            }
                        }
                    }
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

                        // Command palette (Ctrl-K) owns keys while open.
                        if app.palette.is_some() {
                            let outcome = app.palette.as_mut().unwrap().handle_key(&key);
                            match outcome {
                                crate::palette::PaletteOutcome::Continue => {}
                                crate::palette::PaletteOutcome::Closed => {
                                    app.palette = None;
                                }
                                crate::palette::PaletteOutcome::Accept(cmd) => {
                                    app.palette = None;
                                    // Load into input and submit via the normal path.
                                    app.input.set(&cmd);
                                    if cmd.ends_with(' ') {
                                        // leaves trailing space for args — don't submit
                                        app.sync_completion();
                                    } else {
                                        let line = app.input.submit();
                                        if !line.trim().is_empty() {
                                            break 'input line;
                                        }
                                    }
                                }
                            }
                            continue 'input;
                        }

                        // Mode/action dispatch: review, block-nav, global chords.
                        // Specialized paths (normal-mode search typing, history
                        // search, text edit) fall through.
                        if app.mode.is_normal()
                            && !matches!(app.mode, crate::mode::UiMode::Normal { search: Some(_) })
                        {
                            // Non-search normal mode: try actions then normal handler.
                            match app.dispatch_key(&key) {
                                crate::dispatch::DispatchResult::Handled => continue 'input,
                                crate::dispatch::DispatchResult::OpenPalette => {
                                    app.palette = Some(crate::palette::CommandPalette::open());
                                    continue 'input;
                                }
                                crate::dispatch::DispatchResult::Fallthrough => {
                                    handle_normal_mode(&mut app, &key);
                                    continue 'input;
                                }
                            }
                        }
                        if app.mode.is_normal() {
                            // Search substate typing.
                            handle_normal_mode(&mut app, &key);
                            continue 'input;
                        }

                        match app.dispatch_key(&key) {
                            crate::dispatch::DispatchResult::Handled => continue 'input,
                            crate::dispatch::DispatchResult::OpenPalette => {
                                app.palette = Some(crate::palette::CommandPalette::open());
                                continue 'input;
                            }
                            crate::dispatch::DispatchResult::Fallthrough => {}
                        }

                        let history_search_was_active = app.mode.is_history_search();
                        // Ctrl-R opens reverse history search.
                        if ctrl
                            && key.code == KeyCode::Char('r')
                            && !app.mode.is_history_search()
                            && !app.input.history.is_empty()
                        {
                            let mut search = HistorySearch::default();
                            search.refilter(&app.input.history);
                            if let Some(i) = search.current()
                                && i < app.input.history.len()
                            {
                                app.input.set(&app.input.history[i].clone());
                            }
                            app.mode = crate::mode::UiMode::HistorySearch(search);
                            continue 'input;
                        }
                        match key.code {
                            KeyCode::Char('c')
                                if ctrl && app.input.is_empty() && app.quit_notice.is_some() =>
                            {
                                break 'session;
                            }
                            KeyCode::Char('c') if ctrl && app.input.is_empty() => {
                                app.quit_notice =
                                    Some(Instant::now() + Duration::from_millis(1800));
                            }
                            KeyCode::Char('c') if ctrl => {
                                app.quit_notice = None;
                                app.input.clear();
                            }
                            KeyCode::Esc => {
                                app.quit_notice = None;
                                if app.show_help {
                                    app.show_help = false;
                                } else if app.show_diff {
                                    app.show_diff = false;
                                    app.diff_text = None;
                                } else if app.input.is_empty() && !app.working {
                                    if app.mode.is_normal() {
                                        app.mode.to_insert();
                                    } else {
                                        app.mode =
                                            crate::mode::UiMode::Normal { search: None };
                                    }
                                } else {
                                    app.input.clear();
                                }
                            }
                            _ => {
                                app.quit_notice = None;
                                if let Some(line) = app.edit_key(&key) {
                                    break 'input line;
                                }
                                app.sync_completion_after_edit_key(
                                    &key,
                                    history_search_was_active,
                                );
                            }
                        }
                    }
                    Event::FocusGained => app.set_focus(true),
                    Event::FocusLost => app.set_focus(false),
                    _ => {}
                }
            },
        };
        // A line is committed — drop any lingering completion menu state.
        app.completion = None;

        // `!cmd` shell-escape: run a read-only command locally and show its
        // output in the transcript, without involving the model at all. Saves
        // a whole agent turn for trivial checks like `!git status`. Runs
        // asynchronously so a slow command (`!cargo build`) doesn't freeze the
        // TUI — Esc or Ctrl-C cancels it.
        if let Some(shell_cmd) = line.strip_prefix('!').filter(|s| !s.trim().is_empty()) {
            run_shell_escape_async(&mut app, shell_cmd, &mut input_rx, &mut terminal).await?;
            continue;
        }

        // Slash commands. Most are handled inline; `/compact` runs a model call
        // (driven like a turn so the spinner shows); `/retry` yields the prompt
        // to re-run in the turn phase below.
        let mut restore_model_state: Option<hi_agent::AgentModelState> = None;
        let mut restore_app_model: Option<(String, Option<u32>)> = None;
        let run_line = if let Some(cmd) = command::parse(&line) {
            match cmd {
                Command::Quit => break,
                Command::Prompt(prompt) => prompt,
                Command::Moa(prompt) => {
                    let prompt = prompt.trim().to_string();
                    if prompt.is_empty() {
                        app.push(Line::styled("usage: /moa <prompt>".to_string(), dim()));
                        continue;
                    }
                    restore_model_state = Some(agent.model_state());
                    restore_app_model = Some((app.model.clone(), app.context_window));
                    agent.set_model(hi_ai::MOA_MODEL_CONSERVATIVE.to_string(), None, None);
                    app.model = hi_ai::MOA_MODEL_CONSERVATIVE.to_string();
                    app.context_window = None;
                    prompt
                }
                Command::Compact(arg) => {
                    let kind =
                        CompactionKind::from_arg(&arg).unwrap_or_else(|| agent.compaction_kind());
                    app.set_working(true);
                    app.follow();
                    let (tx, rx) = mpsc::unbounded_channel();
                    let (confirm_tx, confirm_rx) = mpsc::unbounded_channel();
                    let mut sink = ChannelUi {
                        tx,
                        confirmations: confirm_tx,
                    };
                    {
                        let fut = agent.compact_with(kind, &mut sink);
                        drive(
                            &mut terminal,
                            &mut input_rx,
                            &mut ticker,
                            &mut app,
                            rx,
                            confirm_rx,
                            fut,
                            false,
                            None,
                        )
                        .await?;
                    }
                    app.set_working(false);
                    // Flush live events after compact too (background, non-blocking).
                    if let Some(rui) = &app.sync_remote_ui {
                        let rui = rui.clone();
                        tokio::spawn(async move {
                            let _ = rui.flush().await;
                        });
                    }
                    if let Some(cb) = &app.remote_flush_callback {
                        cb();
                    }
                    app.follow();
                    continue;
                }
                Command::Retry => {
                    match (app.last_prompt.clone(), app.last_turn_snapshot.as_ref()) {
                        (Some(prompt), Some(snapshot)) => {
                            if let Err(err) =
                                agent.rewind_to_snapshot_durable(app.last_turn_start, snapshot)
                            {
                                app.push(Line::styled(
                                    format!("retry failed: {err:#}"),
                                    Style::default().fg(crate::theme::theme().warning),
                                ));
                                app.follow();
                                continue;
                            }
                            let note = match app.last_turn_state {
                                TurnState::Warning(_) => {
                                    if app.last_turn_had_file_edits {
                                        "retrying from the last safe message checkpoint; file edits already made stay in the working tree and may be replayed if the model repeats them"
                                    } else {
                                        "retrying from the last safe message checkpoint; no file edits were recorded in the last turn"
                                    }
                                }
                                TurnState::Failed(_) => {
                                    "retrying after failure from the last safe message checkpoint"
                                }
                                _ => "retrying from the last safe message checkpoint",
                            };
                            app.push(Line::styled(note.to_string(), dim()));
                            app.push(Line::styled(format!("retrying: {prompt}"), dim()));
                            prompt
                        }
                        _ => {
                            app.push(Line::styled("nothing to retry yet".to_string(), dim()));
                            continue;
                        }
                    }
                }
                Command::Edit => {
                    // Load the last user prompt into the input line for editing.
                    // Unlike /retry, this doesn't submit — the user edits and
                    // presses Enter to send.
                    match agent.last_user_message() {
                        Some(prev) => {
                            app.input.set(&prev);
                            app.sync_completion();
                            continue;
                        }
                        None => {
                            app.push(Line::styled("nothing to edit yet".to_string(), dim()));
                            continue;
                        }
                    }
                }
                Command::Init => {
                    app.push(Line::styled(
                        "scanning the project to write HI.md…".to_string(),
                        dim(),
                    ));
                    command::INIT_PROMPT.to_string()
                }
                Command::Learn(request) => {
                    app.push(Line::styled(
                        "learning a reusable skill…".to_string(),
                        dim(),
                    ));
                    hi_agent::build_learn_prompt(&request)
                }
                Command::Skill(name) => {
                    let name = name.trim();
                    if name.is_empty() {
                        app.push(Line::styled("usage: /skill <name>".to_string(), dim()));
                        app.follow();
                        continue;
                    }
                    match hi_agent::read_skill(name) {
                        Ok(skill) => {
                            hi_agent::build_skill_use_prompt(&skill.skill.name, &skill.content)
                        }
                        Err(err) => {
                            app.push(Line::styled(
                                format!("{err}"),
                                Style::default().fg(crate::theme::theme().warning),
                            ));
                            app.follow();
                            continue;
                        }
                    }
                }
                Command::Hf(arg) => {
                    match hi_tools::handle_hf_command_result(&arg, &mut hf_state).await {
                        Ok(hi_tools::HfCommandResult::Text(text)) => {
                            for line in text.lines() {
                                app.push(Line::styled(line.to_string(), dim()));
                            }
                        }
                        Ok(hi_tools::HfCommandResult::MlxReady(run)) => {
                            for line in run.message.lines() {
                                app.push(Line::styled(line.to_string(), dim()));
                            }
                            match (app.mlx_switcher)(&run) {
                                Ok(switched) => {
                                    let label = switched.switched.label.clone();
                                    let model = switched.switched.model.clone();
                                    agent.set_provider(
                                        switched.switched.provider.into(),
                                        model.clone(),
                                        None,
                                        switched.switched.max_tokens,
                                        switched.switched.max_tokens_explicit,
                                        None,
                                    );
                                    if let Ok(models) = agent.list_models().await {
                                        app.served = models
                                            .into_iter()
                                            .map(|model| (model.id.clone(), model))
                                            .collect();
                                    }
                                    app.provider = label.clone();
                                    app.model = model.clone();
                                    app.active_profile = Some(run.profile_name.clone());
                                    app.profiles = switched.profiles;
                                    app.apply_model(agent, &model);
                                    app.remember_session_routing();
                                    app.push(Line::styled(
                                        format!(
                                            "using local MLX profile '{}' — model: {model}",
                                            run.profile_name
                                        ),
                                        dim(),
                                    ));
                                }
                                Err(err) => {
                                    app.push(Line::styled(
                                        format!("/hf run --mlx profile switch failed: {err:#}"),
                                        Style::default().fg(crate::theme::theme().warning),
                                    ));
                                }
                            }
                        }
                        Err(err) => {
                            app.push(Line::styled(
                                format!("/hf failed: {err:#}"),
                                Style::default().fg(crate::theme::theme().warning),
                            ));
                        }
                    }
                    app.follow();
                    continue;
                }
                Command::Undo => {
                    let checkpoints = agent.checkpoint_count();
                    if checkpoints > 0 {
                        app.push(Line::styled(
                            format!(
                                "undo: restoring latest checkpoint ({checkpoints} available); non-file side effects cannot be reverted"
                            ),
                            dim(),
                        ));
                    }
                    let msg = match agent.undo().await {
                        Ok(Some(0)) => "nothing changed to undo".to_string(),
                        Ok(Some(n)) => format!("↩ undid the last turn — restored {n} file(s)"),
                        Ok(None) => "nothing to undo".to_string(),
                        Err(err) => format!("undo failed: {err:#}"),
                    };
                    app.push(Line::styled(msg, dim()));
                    app.follow();
                    continue;
                }
                // Open the picker on the live model list.
                // The fetch runs behind a spinner so the UI stays responsive and
                // Esc/Ctrl-C can cancel the request.
                Command::Model(id) if id.is_empty() => {
                    app.fetching = Some(Instant::now());
                    let mut fetched: Option<Result<Vec<hi_ai::ServedModel>>> = None;
                    let mut cancelled = false;
                    {
                        let fut = agent.list_models();
                        tokio::pin!(fut);
                        loop {
                            terminal.draw(|f| app.render(f))?;
                            tokio::select! {
                                result = &mut fut => { fetched = Some(result); break; }
                                _ = ticker.tick() => app.spinner = app.spinner.wrapping_add(1),
                                maybe = input_rx.recv() => {
                                    match maybe {
                                        Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                                            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                                            if matches!(key.code, KeyCode::Esc)
                                                || (ctrl && matches!(key.code, KeyCode::Char('c')))
                                            {
                                                cancelled = true;
                                                break;
                                            }
                                        }
                                        Some(_) => {}
                                        None => return Ok(()),
                                    }
                                }
                            }
                        }
                    }
                    app.fetching = None;
                    if cancelled {
                        continue;
                    }
                    // Resolve the model list to show. The live `/models` fetch
                    // is the only source — no static catalog fallback (it dumps
                    // an irrelevant mess of cloud models). A failure or empty
                    // list surfaces the error and skips the picker.
                    let ids = match fetched {
                        Some(Ok(served)) if !served.is_empty() => {
                            // Remember the live metadata (window/price) so
                            // selecting a model can apply it.
                            app.served = served.into_iter().map(|m| (m.id.clone(), m)).collect();
                            let mut ids: Vec<String> = app.served.keys().cloned().collect();
                            ids.sort();
                            app.model_ids = ids.clone();
                            ids
                        }
                        _ => {
                            let note = match &fetched {
                                Some(Ok(_)) => "live model list is empty".to_string(),
                                Some(Err(err)) => format!("live model list not loaded: {err:#}"),
                                None => "live model list not loaded".to_string(),
                            };
                            app.push(Line::styled(note.to_string(), dim()));
                            continue;
                        }
                    };
                    let current = app.model.clone();
                    let tags = app.served_tags();
                    app.picker = Some(ModelPicker::new(ids, &current, tags, &app.served));
                    continue;
                }
                // `/provider` with no arg: list configured profiles.
                // `/provider <name>`: use that profile, fetch the live model
                // metadata, and open the model selector.
                // `/login <provider>`: request a device code (fast), show it,
                // then poll in the background. Awaiting the poll here would
                // freeze the event loop for as long as the user takes in their
                // browser, with no way to cancel.
                Command::Login(arg) => {
                    let arg = arg.trim().to_string();
                    if !matches!(arg.as_str(), "xai" | "grok") {
                        app.push(Line::styled(
                            if arg.is_empty() {
                                "usage: /login xai — sign in with a grok.com subscription"
                                    .to_string()
                            } else {
                                format!("'{arg}' has no subscription sign-in; only xai does")
                            },
                            dim(),
                        ));
                        app.follow();
                        continue;
                    }
                    match hi_ai::xai_auth::request_device_code().await {
                        Ok(device) => {
                            app.push(Line::styled(
                                format!("open  {}", device.url()),
                                ratatui::style::Style::default()
                                    .add_modifier(ratatui::style::Modifier::BOLD),
                            ));
                            app.push(Line::styled(
                                format!("code  {}", device.user_code),
                                ratatui::style::Style::default()
                                    .add_modifier(ratatui::style::Modifier::BOLD),
                            ));
                            app.push(Line::styled(
                                "approve in your browser, then run /provider xai to use it"
                                    .to_string(),
                                dim(),
                            ));
                            app.follow();
                            tokio::spawn(async move {
                                if let Ok(token) = hi_ai::xai_auth::poll_for_token(&device).await {
                                    let _ = hi_ai::auth_store::save(
                                        hi_ai::xai_auth::PROVIDER_ID,
                                        &token,
                                    );
                                }
                            });
                        }
                        Err(error) => {
                            app.push(Line::styled(format!("/login failed: {error:#}"), dim()));
                            app.follow();
                        }
                    }
                    continue;
                }
                Command::Logout(arg) => {
                    let arg = arg.trim().to_string();
                    let message = if matches!(arg.as_str(), "xai" | "grok") {
                        match hi_ai::xai_auth::logout_quiet() {
                            Ok(true) => "signed out of xAI".to_string(),
                            Ok(false) => "not signed in to xAI".to_string(),
                            Err(error) => format!("/logout failed: {error:#}"),
                        }
                    } else {
                        "usage: /logout xai".to_string()
                    };
                    app.push(Line::styled(message, dim()));
                    app.follow();
                    continue;
                }
                Command::Provider(arg) => {
                    let arg = arg.trim().to_string();
                    // --- Subcommands ---
                    if arg == "add" {
                        app.provider_form = Some(provider_form::ProviderForm::new_add());
                        continue;
                    }
                    if let Some(edit_name) = arg.strip_prefix("edit") {
                        let edit_name = edit_name.trim();
                        // If no name is given, use the first profile (or show a hint).
                        let target = if edit_name.is_empty() {
                            if app.profiles.is_empty() {
                                app.push(Line::styled(
                                    "no profiles to edit — use /provider add".to_string(),
                                    dim(),
                                ));
                                continue;
                            }
                            app.profiles[0].name.clone()
                        } else {
                            edit_name.to_string()
                        };
                        // Load the profile's current values via the loader callback.
                        match (app.loader)(&target) {
                            Ok(data) => {
                                app.provider_form = Some(provider_form::ProviderForm::new_edit(
                                    &data.name,
                                    &data.provider,
                                    &data.api_key,
                                    &data.model,
                                    &data.base_url,
                                ));
                            }
                            Err(err) => {
                                app.push(Line::styled(
                                    format!("/provider edit failed: {err:#}"),
                                    Style::default().fg(crate::theme::theme().warning),
                                ));
                            }
                        }
                        continue;
                    }
                    if let Some(rm_name) = arg
                        .strip_prefix("remove")
                        .or_else(|| arg.strip_prefix("rm"))
                    {
                        let rm_name = rm_name.trim();
                        // If no name is given, use the first profile (or show a hint).
                        let target = if rm_name.is_empty() {
                            if app.profiles.is_empty() {
                                app.push(Line::styled("no profiles to remove".to_string(), dim()));
                                continue;
                            }
                            app.profiles[0].name.clone()
                        } else {
                            rm_name.to_string()
                        };
                        // Don't remove the active profile — the agent is using it.
                        if app.active_profile.as_deref() == Some(&target) {
                            app.push(Line::styled(
                                format!("can't remove '{target}' — make a different profile active first"),
                                Style::default().fg(crate::theme::theme().warning),
                            ));
                            continue;
                        }
                        match (app.remover)(&target) {
                            Ok(updated) => {
                                app.profiles = updated;
                                app.push(Line::styled(
                                    format!("removed profile '{target}'"),
                                    dim(),
                                ));
                            }
                            Err(err) => {
                                app.push(Line::styled(
                                    format!("/provider remove failed: {err:#}"),
                                    Style::default().fg(crate::theme::theme().warning),
                                ));
                            }
                        }
                        continue;
                    }
                    // --- Use / list ---
                    if arg.is_empty() {
                        // Open the selector, mirroring `/model` with no arg.
                        let rows: Vec<(String, String)> = app
                            .profiles
                            .iter()
                            .map(|p| {
                                let model = p.model.as_deref().unwrap_or("(no model set)");
                                (p.name.clone(), format!("{} · {model}", p.provider))
                            })
                            .collect();
                        let current = app
                            .active_profile
                            .clone()
                            .unwrap_or_else(|| app.provider.clone());
                        app.provider_picker =
                            Some(provider_picker::ProviderPicker::new(rows, &current));
                        continue;
                    }
                    // Resolve the profile and update the provider.
                    match (app.resolver)(&arg) {
                        Ok(switched) => {
                            let label = switched.label.clone();
                            let model = switched.model.clone();
                            let needs_model = model == "__model_not_configured__";
                            agent.set_provider(
                                switched.provider.into(),
                                model.clone(),
                                None,
                                switched.max_tokens,
                                switched.max_tokens_explicit,
                                None,
                            );
                            app.provider = label.clone();
                            app.model = model.clone();
                            app.active_profile = Some(arg.clone());
                            app.context_window = None;
                            app.served.clear();
                            app.remember_session_routing();
                            // Say "profile" only when it is one: `/provider xai`
                            // selects a provider preset, and calling that a
                            // profile sends people looking for config that
                            // isn't there.
                            let is_profile = app.profiles.iter().any(|p| p.name == arg);
                            app.push(Line::styled(
                                if is_profile {
                                    format!("using {label} (profile: {arg}) — model: {model}")
                                } else {
                                    format!(
                                        "using {label} — model: {model}  \
                                         (no profile; /provider add to save these settings)"
                                    )
                                },
                                dim(),
                            ));
                            if needs_model {
                                app.push(Line::styled(
                                    "no model configured — choose from the available models"
                                        .to_string(),
                                    dim(),
                                ));
                            }
                            // Fetch served models and open the selector, just like
                            // `/model` with no arg.
                            app.fetching = Some(Instant::now());
                            let mut fetched: Option<Result<Vec<hi_ai::ServedModel>>> = None;
                            let mut cancelled = false;
                            {
                                let fut = agent.list_models();
                                tokio::pin!(fut);
                                loop {
                                    terminal.draw(|f| app.render(f))?;
                                    tokio::select! {
                                        result = &mut fut => { fetched = Some(result); break; }
                                        _ = ticker.tick() => app.spinner = app.spinner.wrapping_add(1),
                                        maybe = input_rx.recv() => {
                                            match maybe {
                                                Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                                                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                                                    if matches!(key.code, KeyCode::Esc)
                                                        || (ctrl && matches!(key.code, KeyCode::Char('c')))
                                                    {
                                                        cancelled = true;
                                                        break;
                                                    }
                                                }
                                                Some(_) => {}
                                                None => return Ok(()),
                                            }
                                        }
                                    }
                                }
                            }
                            app.fetching = None;
                            if cancelled {
                                continue;
                            }
                            let ids = match fetched {
                                Some(Ok(served)) if !served.is_empty() => {
                                    let count = served.len();
                                    app.served =
                                        served.into_iter().map(|m| (m.id.clone(), m)).collect();
                                    let mut ids: Vec<String> = app.served.keys().cloned().collect();
                                    ids.sort();
                                    app.model_ids = ids.clone();
                                    app.push(Line::styled(
                                        format!("{count} models available — select one"),
                                        dim(),
                                    ));
                                    ids
                                }
                                _ => {
                                    let note = match &fetched {
                                        Some(Ok(_)) => "live model list is empty".to_string(),
                                        Some(Err(err)) => {
                                            format!("live model list not loaded: {err:#}")
                                        }
                                        None => "live model list not loaded".to_string(),
                                    };
                                    app.push(Line::styled(note.to_string(), dim()));
                                    continue;
                                }
                            };
                            let current = app.model.clone();
                            let tags = app.served_tags();
                            app.picker = Some(ModelPicker::new(ids, &current, tags, &app.served));
                        }
                        Err(err) => {
                            app.push(Line::styled(
                                format!("/provider failed: {err:#}"),
                                Style::default().fg(crate::theme::theme().warning),
                            ));
                        }
                    }
                    continue;
                }
                // `/loop`: recurring agent turns on a cadence (manager task).
                Command::Loop(arg) => {
                    if app.loops.is_none() {
                        app.push(Line::styled(
                            "loops are managed by a background daemon — stop it to manage them here, or use /digest to see what they've noticed".to_string(),
                            dim(),
                        ));
                        app.follow();
                        continue;
                    }
                    match command::parse_loop_arg(&arg) {
                        command::LoopArg::Create { secs, prompt } => {
                            if let Some(loops) = &app.loops {
                                let (tx, rx) = tokio::sync::oneshot::channel();
                                let _ = loops.ctl.send(crate::loops::LoopCtl::Create {
                                    secs,
                                    prompt: prompt.clone(),
                                    reply: tx,
                                });
                                match rx.await {
                                    Ok(Ok(spec)) => {
                                        app.push(Line::styled(
                                            format!(
                                                "✓ loop#{} armed — every {}, expires in 7d, firing now: {}",
                                                spec.id,
                                                crate::loops::humanize_secs(spec.interval_secs),
                                                spec.name(),
                                            ),
                                            Style::default().fg(crate::theme::theme().accent_success),
                                        ));
                                    }
                                    Ok(Err(err)) => {
                                        app.push(Line::styled(
                                            err,
                                            Style::default().fg(crate::theme::theme().warning),
                                        ));
                                    }
                                    Err(_) => {}
                                }
                            }
                        }
                        command::LoopArg::Cancel(id) => {
                            if let Some(loops) = &app.loops {
                                let (tx, rx) = tokio::sync::oneshot::channel();
                                let _ = loops
                                    .ctl
                                    .send(crate::loops::LoopCtl::Cancel { id, reply: tx });
                                let msg = match rx.await {
                                    Ok(true) => (format!("✓ loop#{id} cancelled"), crate::theme::theme().accent_success),
                                    _ => (
                                        format!("no loop#{id} — /loop list shows ids"),
                                        crate::theme::theme().warning,
                                    ),
                                };
                                app.push(Line::styled(msg.0, Style::default().fg(msg.1)));
                            }
                        }
                        command::LoopArg::List => {
                            if let Some(loops) = &app.loops {
                                let (tx, rx) = tokio::sync::oneshot::channel();
                                let _ = loops.ctl.send(crate::loops::LoopCtl::List { reply: tx });
                                if let Ok(specs) = rx.await {
                                    if specs.is_empty() {
                                        app.push(Line::styled(
                                            "no active loops — /loop <interval> <prompt> to arm one"
                                                .to_string(),
                                            dim(),
                                        ));
                                    } else {
                                        app.push(Line::styled(
                                            format!("active loops ({}):", specs.len()),
                                            Style::default()
                                                .fg(crate::theme::theme().accent_system)
                                                .add_modifier(ratatui::style::Modifier::BOLD),
                                        ));
                                        let now = std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .map(|d| d.as_millis() as u64)
                                            .unwrap_or(0);
                                        for l in specs {
                                            let due_in = l.next_ms.saturating_sub(now) / 1000;
                                            let expires_h =
                                                l.expires_ms.saturating_sub(now) / 3_600_000;
                                            let next = if l.paused {
                                                "paused".to_string()
                                            } else {
                                                format!("next in {due_in}s")
                                            };
                                            let cost = match l.token_budget {
                                                Some(b) => format!(
                                                    " · {}/{}",
                                                    crate::loops::fmt_tokens(l.spent_tokens),
                                                    crate::loops::fmt_tokens(b)
                                                ),
                                                None if l.spent_tokens > 0 => format!(
                                                    " · {} spent",
                                                    crate::loops::fmt_tokens(l.spent_tokens)
                                                ),
                                                None => String::new(),
                                            };
                                            let mut marks = String::new();
                                            if l.trigger.is_some() {
                                                marks.push_str(" · ⚡");
                                            }
                                            if l.autofix {
                                                marks.push_str(if l.fix_pr {
                                                    " · ⚒pr"
                                                } else {
                                                    " · ⚒"
                                                });
                                            }
                                            if let Some(s) = &l.schedule {
                                                marks.push_str(&format!(" · ⌚{}", s.label()));
                                            }
                                            app.push(Line::styled(
                                                format!(
                                                    "  #{} every {} · {} · {} firing(s){}{} · expires {}h · {}",
                                                    l.id,
                                                    crate::loops::humanize_secs(l.interval_secs),
                                                    next,
                                                    l.firings,
                                                    cost,
                                                    marks,
                                                    expires_h,
                                                    l.name(),
                                                ),
                                                dim(),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        command::LoopArg::Pause(id) | command::LoopArg::Resume(id) => {
                            let on =
                                matches!(command::parse_loop_arg(&arg), command::LoopArg::Pause(_));
                            if let Some(loops) = &app.loops {
                                let (tx, rx) = tokio::sync::oneshot::channel();
                                let _ = loops.ctl.send(crate::loops::LoopCtl::Pause {
                                    id,
                                    on,
                                    reply: tx,
                                });
                                let verb = if on { "paused" } else { "resumed" };
                                let msg = match rx.await {
                                    Ok(true) => (format!("✓ loop#{id} {verb}"), crate::theme::theme().accent_success),
                                    _ => (
                                        format!("no loop#{id} — /loop list shows ids"),
                                        crate::theme::theme().warning,
                                    ),
                                };
                                app.push(Line::styled(msg.0, Style::default().fg(msg.1)));
                            }
                        }
                        command::LoopArg::Budget { id, tokens } => {
                            if let Some(loops) = &app.loops {
                                let (tx, rx) = tokio::sync::oneshot::channel();
                                let _ = loops.ctl.send(crate::loops::LoopCtl::Budget {
                                    id,
                                    tokens,
                                    reply: tx,
                                });
                                let msg = match (rx.await, tokens) {
                                    (Ok(true), Some(t)) => (
                                        format!(
                                            "✓ loop#{id} budget set to {}",
                                            crate::loops::fmt_tokens(t)
                                        ),
                                        crate::theme::theme().accent_success,
                                    ),
                                    (Ok(true), None) => {
                                        (format!("✓ loop#{id} budget cleared"), crate::theme::theme().accent_success)
                                    }
                                    _ => (
                                        format!("no loop#{id} — /loop list shows ids"),
                                        crate::theme::theme().warning,
                                    ),
                                };
                                app.push(Line::styled(msg.0, Style::default().fg(msg.1)));
                            }
                        }
                        command::LoopArg::Trigger { id, cmd } => {
                            if let Some(loops) = &app.loops {
                                let (tx, rx) = tokio::sync::oneshot::channel();
                                let set = cmd.is_some();
                                let _ = loops.ctl.send(crate::loops::LoopCtl::Trigger {
                                    id,
                                    cmd,
                                    reply: tx,
                                });
                                let msg = match (rx.await, set) {
                                    (Ok(true), true) => (
                                        format!("✓ loop#{id} will run its command on each change"),
                                        crate::theme::theme().accent_success,
                                    ),
                                    (Ok(true), false) => {
                                        (format!("✓ loop#{id} trigger cleared"), crate::theme::theme().accent_success)
                                    }
                                    _ => (
                                        format!("no loop#{id} — /loop list shows ids"),
                                        crate::theme::theme().warning,
                                    ),
                                };
                                app.push(Line::styled(msg.0, Style::default().fg(msg.1)));
                            }
                        }
                        command::LoopArg::Fix { id, on, pr } => {
                            if let Some(loops) = &app.loops {
                                let (tx, rx) = tokio::sync::oneshot::channel();
                                let _ = loops.ctl.send(crate::loops::LoopCtl::Fix {
                                    id,
                                    on,
                                    pr,
                                    reply: tx,
                                });
                                let no_verify = on && fleet_launcher.verify.is_none();
                                let msg = match rx.await {
                                    Ok(true) if on && pr => (
                                        format!(
                                            "✓ loop#{id} auto-fix on (PR mode) — a loud change opens a verified fix as a PR"
                                        ),
                                        crate::theme::theme().accent_success,
                                    ),
                                    Ok(true) if on => (
                                        format!(
                                            "✓ loop#{id} auto-fix on — a loud change merges a verified fix into your tree"
                                        ),
                                        crate::theme::theme().accent_success,
                                    ),
                                    Ok(true) => (format!("✓ loop#{id} auto-fix off"), crate::theme::theme().accent_success),
                                    _ => (
                                        format!("no loop#{id} — /loop list shows ids"),
                                        crate::theme::theme().warning,
                                    ),
                                };
                                app.push(Line::styled(msg.0, Style::default().fg(msg.1)));
                                if no_verify {
                                    app.push(Line::styled(
                                        "  note: no verify command set — fixes won't land until you /verify <cmd>"
                                            .to_string(),
                                        dim(),
                                    ));
                                }
                            }
                        }
                        command::LoopArg::Window { id, window } => {
                            if let Some(loops) = &app.loops {
                                let (tx, rx) = tokio::sync::oneshot::channel();
                                let _ = loops.ctl.send(crate::loops::LoopCtl::Window {
                                    id,
                                    window,
                                    reply: tx,
                                });
                                let msg = match (rx.await, window) {
                                    (Ok(true), Some((s, e, wd))) => (
                                        format!(
                                            "✓ loop#{id} fires only {s:02}-{e:02}{} (local time)",
                                            if wd { " weekdays" } else { "" }
                                        ),
                                        crate::theme::theme().accent_success,
                                    ),
                                    (Ok(true), None) => (
                                        format!("✓ loop#{id} window cleared — fires anytime"),
                                        crate::theme::theme().accent_success,
                                    ),
                                    _ => (
                                        format!("no loop#{id} — /loop list shows ids"),
                                        crate::theme::theme().warning,
                                    ),
                                };
                                app.push(Line::styled(msg.0, Style::default().fg(msg.1)));
                            }
                        }
                        command::LoopArg::Cost => {
                            if let Some(loops) = &app.loops {
                                let (tx, rx) = tokio::sync::oneshot::channel();
                                let _ = loops.ctl.send(crate::loops::LoopCtl::List { reply: tx });
                                if let Ok(mut specs) = rx.await {
                                    let total: u64 = specs.iter().map(|l| l.spent_tokens).sum();
                                    if specs.is_empty() {
                                        app.push(Line::styled(
                                            "no loops — nothing spent yet".to_string(),
                                            dim(),
                                        ));
                                    } else {
                                        app.push(Line::styled(
                                            format!(
                                                "loop spend — {} total across {} loop(s):",
                                                crate::loops::fmt_tokens(total),
                                                specs.len()
                                            ),
                                            Style::default()
                                                .fg(crate::theme::theme().accent_system)
                                                .add_modifier(ratatui::style::Modifier::BOLD),
                                        ));
                                        specs.sort_by_key(|l| std::cmp::Reverse(l.spent_tokens));
                                        for l in specs {
                                            let budget = l
                                                .token_budget
                                                .map(|b| {
                                                    format!(" / {}", crate::loops::fmt_tokens(b))
                                                })
                                                .unwrap_or_default();
                                            app.push(Line::styled(
                                                format!(
                                                    "  #{}  {:>8}{}  · {} firing(s) · {}",
                                                    l.id,
                                                    crate::loops::fmt_tokens(l.spent_tokens),
                                                    budget,
                                                    l.firings,
                                                    l.name(),
                                                ),
                                                dim(),
                                            ));
                                        }
                                        app.push(Line::styled(
                                            "  (loops only — fleet/goal spend is per-session)"
                                                .to_string(),
                                            dim(),
                                        ));
                                    }
                                }
                            }
                        }
                        command::LoopArg::Trio { prompt, max_rounds } => {
                            // ── Plan phase ───────────────────────────────────
                            app.push(Line::styled(
                                format!("trio: planning — {prompt}"),
                                Style::default().fg(crate::theme::theme().accent_system),
                            ));
                            app.follow();
                            app.planning = Some(Instant::now());
                            let mut plan_result: Option<Result<String>> = None;
                            let mut cancelled = false;
                            {
                                let fut = agent.trio_plan(&prompt);
                                tokio::pin!(fut);
                                loop {
                                    terminal.draw(|f| app.render(f))?;
                                    tokio::select! {
                                        result = &mut fut => { plan_result = Some(result); break; }
                                        _ = ticker.tick() => app.spinner = app.spinner.wrapping_add(1),
                                        maybe = input_rx.recv() => {
                                            match maybe {
                                                Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                                                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                                                    if matches!(key.code, KeyCode::Esc)
                                                        || (ctrl && matches!(key.code, KeyCode::Char('c')))
                                                    {
                                                        cancelled = true;
                                                        break;
                                                    }
                                                }
                                                Some(_) => {}
                                                None => return Ok(()),
                                            }
                                        }
                                    }
                                }
                            }
                            app.planning = None;
                            if cancelled {
                                app.push(Line::styled(
                                    "trio: planning cancelled".to_string(),
                                    dim(),
                                ));
                                app.follow();
                                continue;
                            }
                            let plan = plan_result
                                .unwrap_or_else(|| Ok(prompt.clone()))
                                .unwrap_or_else(|_| prompt.clone());
                            app.push(Line::styled(
                                format!("trio: plan ready, executing (max {max_rounds} rounds)"),
                                Style::default().fg(crate::theme::theme().accent_system),
                            ));
                            app.follow();

                            // ── Execute → Review loop ────────────────────────
                            let mut round: u8 = 0;
                            let mut last_objections: Vec<String> = Vec::new();
                            let mut approved = false;
                            let mut loop_cancelled = false;
                            while round < max_rounds {
                                round += 1;
                                // Build the execute input: plan + prompt + any
                                // objections from the previous round.
                                let run_line = if round == 1 {
                                    format!(
                                        "Implement this task using the following plan.\n\n\
                                         Task: {prompt}\n\n\
                                         Plan:\n{plan}"
                                    )
                                } else {
                                    format!(
                                        "The reviewer found issues with the previous attempt. \
                                         Fix them and re-implement.\n\n\
                                         Task: {prompt}\n\n\
                                         Plan:\n{plan}\n\n\
                                         Reviewer objections to address:\n{}",
                                        last_objections
                                            .iter()
                                            .map(|o| format!("• {o}"))
                                            .collect::<Vec<_>>()
                                            .join("\n")
                                    )
                                };

                                // ── Execute phase: run a normal turn ────────
                                // Mirrors the main turn path so cancellation,
                                // background-process cleanup, and session-state
                                // rewind are handled identically.
                                app.push_user_prompt(ratatui::text::Line::styled(
                                    format!("❯ {run_line}"),
                                    ratatui::style::Style::default()
                                        .fg(crate::theme::theme().accent_user),
                                ));
                                app.set_working(true);
                                app.follow();
                                let checkpoint = agent.messages().len();
                                let checkpoint_count = agent.checkpoint_count();
                                app.last_turn_start = checkpoint;
                                app.last_prompt = Some(run_line.clone());
                                let turn_snapshot = agent.state_snapshot();
                                app.last_turn_snapshot = Some(turn_snapshot.clone());
                                app.turn_tool_calls = 0;
                                app.turn_rounds = 0;
                                app.interrupt = Some(agent.interrupt_handle());
                                let (tx, rx) = mpsc::unbounded_channel();
                                let (confirm_tx, confirm_rx) = mpsc::unbounded_channel();
                                let mut sink = ChannelUi {
                                    tx,
                                    confirmations: confirm_tx,
                                };
                                let background_before = agent.background_process_ids();
                                let interject = agent.interjection_inbox();
                                let driven = {
                                    let fut = agent.run_turn(&run_line, &mut sink);
                                    drive(
                                        &mut terminal,
                                        &mut input_rx,
                                        &mut ticker,
                                        &mut app,
                                        rx,
                                        confirm_rx,
                                        fut,
                                        true,
                                        Some(interject),
                                    )
                                    .await?
                                };
                                let cancelled = driven.cancelled;
                                if let Some(outcome) = &driven.value {
                                    app.note_turn_outcome(outcome);
                                } else if !cancelled {
                                    let outcome = agent.finalize_failed_turn();
                                    app.note_turn_outcome(&outcome);
                                }
                                app.set_working(false);
                                app.interrupt = None;

                                if cancelled {
                                    // Full cancellation cleanup — same as the
                                    // main turn path: kill bg processes, rewind
                                    // session state, finalize the cancellation.
                                    let killed = agent.kill_background_processes_started_after(
                                        &background_before,
                                    );
                                    if agent.checkpoint_count() > checkpoint_count
                                        && let Err(err) = agent.undo().await
                                    {
                                        app.push(Line::styled(
                                            format!("couldn't roll back interrupted workspace edits: {err:#}"),
                                            Style::default().fg(crate::theme::theme().warning),
                                        ));
                                    }
                                    if let Err(err) =
                                        agent.rewind_to_snapshot_durable(checkpoint, &turn_snapshot)
                                    {
                                        app.push(Line::styled(
                                            format!(
                                                "couldn't persist interrupted turn discard: {err:#}"
                                            ),
                                            Style::default().fg(crate::theme::theme().warning),
                                        ));
                                        agent.truncate_messages(checkpoint);
                                        agent.restore_state_snapshot(&turn_snapshot);
                                    }
                                    match agent.finalize_cancelled_turn() {
                                        Ok(outcome) => app.note_turn_outcome(&outcome),
                                        Err(err) => {
                                            app.last_turn_state = TurnState::Cancelled;
                                            app.status = "cancelled".to_string();
                                            app.push(Line::styled(
                                                format!("couldn't finalize typed cancellation outcome: {err:#}"),
                                                Style::default().fg(crate::theme::theme().warning),
                                            ));
                                        }
                                    }
                                    let msg = if killed > 0 {
                                        format!(
                                            "trio: cancelled; killed {killed} background process(es)"
                                        )
                                    } else {
                                        "trio: cancelled".to_string()
                                    };
                                    app.push(Line::styled(msg, dim()));
                                    loop_cancelled = true;
                                    break;
                                }

                                // Turn finished normally — capture post-turn state.
                                app.maybe_notify_done();
                                app.last_changed_files = agent.last_changed_files().to_vec();
                                app.accumulate_session_files();
                                app.last_telemetry = Some(agent.last_turn_telemetry().clone());
                                app.diff_text = None;
                                app.refresh_goal(agent);

                                // ── Review phase: side-call to reviewer ──────
                                // Cancellable via Esc/Ctrl-C (fail-open on cancel
                                // — treat as approved so the loop exits cleanly).
                                app.push(Line::styled(
                                    format!("trio: reviewing round {round}/{max_rounds}…"),
                                    Style::default().fg(crate::theme::theme().accent_system),
                                ));
                                app.follow();
                                let mut verdict_result: Option<hi_agent::SkepticVerdict> = None;
                                let mut review_cancelled = false;
                                {
                                    let fut = agent.trio_review(&prompt, &plan);
                                    tokio::pin!(fut);
                                    loop {
                                        terminal.draw(|f| app.render(f))?;
                                        tokio::select! {
                                            result = &mut fut => { verdict_result = Some(result); break; }
                                            _ = ticker.tick() => app.spinner = app.spinner.wrapping_add(1),
                                            maybe = input_rx.recv() => {
                                                match maybe {
                                                    Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                                                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                                                        if matches!(key.code, KeyCode::Esc)
                                                            || (ctrl && matches!(key.code, KeyCode::Char('c')))
                                                        {
                                                            review_cancelled = true;
                                                            break;
                                                        }
                                                    }
                                                    Some(_) => {}
                                                    None => return Ok(()),
                                                }
                                            }
                                        }
                                    }
                                }
                                if review_cancelled {
                                    app.push(Line::styled(
                                        "trio: review cancelled — approving (fail-open)"
                                            .to_string(),
                                        Style::default().fg(crate::theme::theme().gray_dim),
                                    ));
                                    approved = true;
                                    break;
                                }
                                let verdict = verdict_result.unwrap_or(
                                    hi_agent::SkepticVerdict::Unavailable(
                                        "reviewer returned no result".into(),
                                    ),
                                );
                                match &verdict {
                                    hi_agent::SkepticVerdict::Approve => {
                                        approved = true;
                                        app.push(Line::styled(
                                            format!(
                                                "✓ trio: approved in round {round}/{max_rounds}"
                                            ),
                                            Style::default().fg(crate::theme::theme().accent_success),
                                        ));
                                        break;
                                    }
                                    hi_agent::SkepticVerdict::Object(objs) => {
                                        last_objections = objs.clone();
                                        app.push(Line::styled(
                                            format!(
                                                "trio: round {round} objected — {} issue(s), revising",
                                                objs.len()
                                            ),
                                            Style::default().fg(crate::theme::theme().warning),
                                        ));
                                        for o in objs {
                                            app.push(Line::styled(
                                                format!("  • {o}"),
                                                Style::default().fg(crate::theme::theme().warning),
                                            ));
                                        }
                                        app.follow();
                                    }
                                    hi_agent::SkepticVerdict::Escalate(objs) => {
                                        // Retrying can't fix it — surface and stop
                                        // the revision loop instead of burning rounds.
                                        last_objections = objs.clone();
                                        app.push(Line::styled(
                                            format!(
                                                "trio: round {round} escalated — needs your judgment, stopping revisions"
                                            ),
                                            Style::default().fg(crate::theme::theme().accent_error),
                                        ));
                                        for o in objs {
                                            app.push(Line::styled(
                                                format!("  • {o}"),
                                                Style::default().fg(crate::theme::theme().accent_error),
                                            ));
                                        }
                                        app.follow();
                                        break;
                                    }
                                    hi_agent::SkepticVerdict::Unavailable(msg) => {
                                        // Fail-open: treat as approved (can't wedge the loop).
                                        approved = true;
                                        app.push(Line::styled(
                                            format!("trio: reviewer unavailable ({msg}) — approving (fail-open)"),
                                            Style::default().fg(crate::theme::theme().gray_dim),
                                        ));
                                        break;
                                    }
                                }
                            }
                            if !approved && !loop_cancelled {
                                app.push(Line::styled(
                                    format!("trio: hit round cap ({max_rounds}) without approval"),
                                    Style::default().fg(crate::theme::theme().warning),
                                ));
                            }
                            if loop_cancelled {
                                app.push(Line::styled("trio: cancelled".to_string(), dim()));
                            }
                            app.follow();
                            continue;
                        }
                        command::LoopArg::Invalid(msg) => {
                            app.push(Line::styled(msg, Style::default().fg(crate::theme::theme().warning)));
                        }
                    }
                    app.follow();
                    continue;
                }
                // `/dashboard`: the fleet screen — dispatch, monitor, and steer
                // multiple concurrent agent sessions. Runs its own select! loop
                // over the same terminal/input/ticker; rows persist on `app.fleet`.
                // `/fleet status` lists this project's resumable fleet sessions.
                Command::Dashboard(arg) => {
                    match arg.trim() {
                        "" => {
                            crate::dashboard::run_dashboard(
                                &mut terminal,
                                &mut input_rx,
                                &mut ticker,
                                &mut app,
                                &fleet_launcher,
                                None,
                            )
                            .await?;
                        }
                        // `/fleet resume [id]`: re-adopt a past fleet session as
                        // a live row (most recent when no id) and open the fleet.
                        resume if resume == "resume" || resume.starts_with("resume ") => {
                            let id = resume.strip_prefix("resume").unwrap_or("").trim();
                            match (fleet_launcher.resume_info)(id) {
                                Some(info) => {
                                    crate::dashboard::run_dashboard(
                                        &mut terminal,
                                        &mut input_rx,
                                        &mut ticker,
                                        &mut app,
                                        &fleet_launcher,
                                        Some(info),
                                    )
                                    .await?;
                                }
                                None => {
                                    app.push(Line::styled(
                                        if id.is_empty() {
                                            "no fleet sessions to resume — /dashboard to dispatch some"
                                                .to_string()
                                        } else {
                                            format!("no fleet session '{id}' — see /fleet status")
                                        },
                                        dim(),
                                    ));
                                    app.follow();
                                }
                            }
                        }
                        "status" | "sessions" | "ls" => {
                            let sessions = (fleet_launcher.sessions)();
                            if sessions.is_empty() {
                                app.push(Line::styled(
                                    "no fleet sessions in this project yet — /dashboard to dispatch some"
                                        .to_string(),
                                    dim(),
                                ));
                            } else {
                                app.push(Line::styled(
                                    format!("fleet sessions ({}):", sessions.len()),
                                    Style::default()
                                        .fg(crate::theme::theme().accent_assistant)
                                        .add_modifier(ratatui::style::Modifier::BOLD),
                                ));
                                for s in sessions.iter().take(20) {
                                    app.push(Line::styled(
                                        format!(
                                            "  {}  {:>8} · {:>4} lines · {}",
                                            s.id,
                                            s.age,
                                            s.lines,
                                            crate::dashboard::truncate_title(&s.title, 56),
                                        ),
                                        dim(),
                                    ));
                                }
                                if sessions.len() > 20 {
                                    app.push(Line::styled(
                                        format!("  … +{} more", sessions.len() - 20),
                                        dim(),
                                    ));
                                }
                                app.push(Line::styled(
                                    "resume one with: hi --resume <id>".to_string(),
                                    dim(),
                                ));
                            }
                            app.follow();
                        }
                        other => {
                            app.push(Line::styled(
                                format!("unknown /fleet subcommand '{other}' — try /fleet status"),
                                dim(),
                            ));
                            app.follow();
                        }
                    }
                    continue;
                }
                // `/watch`: full-screen live dashboard of all active loops. Runs
                // over the same terminal/input/ticker; the loop manager keeps
                // firing throughout, and closing it returns to the chat.
                Command::Watch => {
                    if app.loops.is_none() {
                        app.push(Line::styled(
                            "loops are managed by a background daemon — /digest shows what they've noticed; stop the daemon to watch them live here".to_string(),
                            dim(),
                        ));
                        app.follow();
                        continue;
                    }
                    crate::watch::run_watch(&mut terminal, &mut input_rx, &mut ticker, &mut app)
                        .await?;
                    // Surface anything the loops reported while we were watching.
                    app.drain_loops();
                    continue;
                }
                // `/digest`: the loud things loops have noticed, grouped by loop,
                // with what's new since you last looked (then mark all as seen).
                Command::Digest => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);
                    if let Some(lf) = &fleet_launcher.loops_file {
                        let entries = crate::activity::load(&crate::activity::activity_path(lf));
                        let seen_path = crate::activity::seen_path(lf);
                        let seen = crate::activity::load_seen(&seen_path);
                        let (groups, total, fresh) = crate::activity::digest(&entries, 0, seen, 3);
                        if total == 0 {
                            app.push(Line::styled(
                                "no loop activity yet — loops record changes here as they notice them"
                                    .to_string(),
                                dim(),
                            ));
                        } else {
                            let header = if fresh > 0 {
                                format!(
                                    "activity digest — {total} change(s) across {} loop(s) · {fresh} new since you last looked",
                                    groups.len()
                                )
                            } else {
                                format!(
                                    "activity digest — {total} change(s) across {} loop(s)",
                                    groups.len()
                                )
                            };
                            app.push(Line::styled(
                                header,
                                Style::default()
                                    .fg(crate::theme::theme().accent_system)
                                    .add_modifier(ratatui::style::Modifier::BOLD),
                            ));
                            let ago = |ms: u64| -> String {
                                let s = now.saturating_sub(ms) / 1000;
                                if s < 60 {
                                    format!("{s}s")
                                } else if s < 3600 {
                                    format!("{}m", s / 60)
                                } else if s < 86_400 {
                                    format!("{}h", s / 3600)
                                } else {
                                    format!("{}d", s / 86_400)
                                }
                            };
                            for g in &groups {
                                let fresh_note = if g.fresh > 0 {
                                    format!(" · {} new", g.fresh)
                                } else {
                                    String::new()
                                };
                                app.push(Line::styled(
                                    format!("  {} — {} change(s){}", g.source, g.count, fresh_note),
                                    Style::default().add_modifier(ratatui::style::Modifier::BOLD),
                                ));
                                for (at, text, is_fresh) in &g.recent {
                                    let mark = if *is_fresh { "• " } else { "  " };
                                    let style = if *is_fresh {
                                        Style::default().fg(crate::theme::theme().accent_system)
                                    } else {
                                        dim()
                                    };
                                    app.push(Line::styled(
                                        format!(
                                            "    {mark}{:>4} ago  {}",
                                            ago(*at),
                                            crate::dashboard::truncate_title(text, 72)
                                        ),
                                        style,
                                    ));
                                }
                            }
                        }
                        crate::activity::save_seen(&seen_path, now);
                    } else {
                        app.push(Line::styled(
                            "activity digest unavailable (no project loops file)".to_string(),
                            dim(),
                        ));
                    }
                    app.follow();
                    continue;
                }
                // `/goal <objective>`: decompose with the planner behind a spinner
                // (Esc cancels), then install the structured goal. Control
                // subcommands (clear/pause/resume/limit) and the no-planner case
                // stay on the sync handler.
                Command::Goal(arg)
                    if agent.has_planner() && hi_agent::command::goal_arg_is_objective(&arg) =>
                {
                    let objective = arg.trim().to_string();
                    app.planning = Some(Instant::now());
                    let mut decomposed: Option<Result<Vec<String>>> = None;
                    let mut cancelled = false;
                    {
                        let fut = agent.decompose_goal(&objective);
                        tokio::pin!(fut);
                        loop {
                            terminal.draw(|f| app.render(f))?;
                            tokio::select! {
                                result = &mut fut => { decomposed = Some(result); break; }
                                _ = ticker.tick() => app.spinner = app.spinner.wrapping_add(1),
                                maybe = input_rx.recv() => {
                                    match maybe {
                                        Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                                            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                                            if matches!(key.code, KeyCode::Esc)
                                                || (ctrl && matches!(key.code, KeyCode::Char('c')))
                                            {
                                                cancelled = true;
                                                break;
                                            }
                                        }
                                        Some(_) => {}
                                        None => return Ok(()),
                                    }
                                }
                            }
                        }
                    }
                    app.planning = None;
                    if cancelled {
                        app.push(Line::styled("goal planning cancelled".to_string(), dim()));
                        app.follow();
                        continue;
                    }
                    // Fall back to a single sub-goal if the planner errored or
                    // returned nothing usable.
                    let sub_goals = match decomposed {
                        Some(Ok(steps)) if !steps.is_empty() => steps,
                        other => {
                            if let Some(Err(err)) = other {
                                app.push(Line::styled(
                                    format!(
                                        "planner unavailable ({err:#}); using the objective as one step"
                                    ),
                                    dim(),
                                ));
                            }
                            vec![objective.clone()]
                        }
                    };
                    app.set_planned_goal(agent, &objective, sub_goals);
                    // A goal is a contract: start pulling toward it immediately.
                    // The user monitors and steers — pause/Esc stops the drive.
                    app.goal_drive_stall = 0;
                    app.maybe_queue_goal_drive(agent);
                    continue;
                }
                // Other `/goal` forms (read/pause/resume/limit/clear, or an
                // objective with no planner): the sync handler — then start the
                // drive if an active goal came out of it (objective or resume).
                Command::Goal(arg) => {
                    let could_drive =
                        hi_agent::command::goal_arg_is_objective(&arg) || arg.trim() == "resume";
                    app.handle_goal(agent, &arg);
                    if could_drive {
                        app.goal_drive_stall = 0;
                        app.maybe_queue_goal_drive(agent);
                    }
                    continue;
                }
                other => {
                    app.handle_command(agent, other).await;
                    continue;
                }
            }
        } else {
            line
        };

        // Expand `@file` mentions: read each referenced file and append its
        // contents to the prompt so the model sees the file without a separate
        // `read` tool call. The `@path` tokens stay in the user-visible text
        // (so the transcript reads naturally); the contents are appended below
        // a clear separator. Missing/oversize files are noted inline.
        let run_line = expand_file_mentions(&run_line, &app.workspace_root);

        // --- Turn phase: run the agent behind a channel, staying responsive. ---
        app.push_user_prompt(ratatui::text::Line::styled(
            format!("❯ {run_line}"),
            ratatui::style::Style::default().fg(crate::theme::theme().accent_user),
        ));
        app.set_working(true);
        app.follow();
        let checkpoint = agent.messages().len();
        let checkpoint_count = agent.checkpoint_count();
        app.last_turn_start = checkpoint;
        app.last_prompt = Some(run_line.clone());
        // Long-horizon auto-drive bookkeeping: whether this is a synthetic drive
        // turn, and the goal state going in — any change by turn end (advance,
        // retry note, plan growth) counts as progress; no change is a stall.
        let goal_drive_turn = run_line == hi_agent::GOAL_CONTINUE_PROMPT;
        let goal_before = agent.structured_goal().cloned();
        let turn_snapshot = agent.state_snapshot();
        app.last_turn_snapshot = Some(turn_snapshot.clone());
        // Reset the per-turn tool-call counter for the observability panel.
        app.turn_tool_calls = 0;
        app.turn_rounds = 0;
        // Grab the interrupt handle so Esc during a tool call can signal it.
        app.interrupt = Some(agent.interrupt_handle());
        let (tx, rx) = mpsc::unbounded_channel();
        let (confirm_tx, confirm_rx) = mpsc::unbounded_channel();
        let mut sink = ChannelUi {
            tx,
            confirmations: confirm_tx,
        };
        let background_before = agent.background_process_ids();
        let interject = agent.interjection_inbox();
        let driven = {
            let fut = agent.run_turn(&run_line, &mut sink);
            drive(
                &mut terminal,
                &mut input_rx,
                &mut ticker,
                &mut app,
                rx,
                confirm_rx,
                fut,
                true,
                Some(interject),
            )
            .await?
        };
        let cancelled = driven.cancelled;
        if let Some(outcome) = &driven.value {
            app.note_turn_outcome(outcome);
        } else if !cancelled {
            // `run_turn` can return early on provider/runner/session failures
            // before its normal finalizer. Reconcile the surviving workspace
            // effects and retain the same typed infrastructure outcome used by
            // one-shot reports.
            let outcome = agent.finalize_failed_turn();
            app.note_turn_outcome(&outcome);
        }

        if cancelled {
            let killed = agent.kill_background_processes_started_after(&background_before);
            if agent.checkpoint_count() > checkpoint_count
                && let Err(err) = agent.undo().await
            {
                app.push(Line::styled(
                    format!("couldn't roll back interrupted workspace edits: {err:#}"),
                    Style::default().fg(crate::theme::theme().warning),
                ));
            }
            if let Err(err) = agent.rewind_to_snapshot_durable(checkpoint, &turn_snapshot) {
                app.push(Line::styled(
                    format!("couldn't persist interrupted turn discard: {err:#}"),
                    Style::default().fg(crate::theme::theme().warning),
                ));
                agent.truncate_messages(checkpoint);
                agent.restore_state_snapshot(&turn_snapshot);
            }
            match agent.finalize_cancelled_turn() {
                Ok(outcome) => app.note_turn_outcome(&outcome),
                Err(err) => {
                    app.last_turn_state = TurnState::Cancelled;
                    app.status = "cancelled".to_string();
                    app.push(Line::styled(
                        format!("couldn't finalize typed cancellation outcome: {err:#}"),
                        Style::default().fg(crate::theme::theme().warning),
                    ));
                }
            }
            let dropped = app.queue.len();
            app.queue.clear();
            let msg = if dropped > 0 {
                format!("^C interrupted; turn discarded ({dropped} queued command(s) dropped)")
            } else {
                "^C interrupted; turn discarded".to_string()
            };
            let msg = if killed > 0 {
                format!("{msg}; killed {killed} background process(es) started by it")
            } else {
                msg
            };
            app.push(Line::styled(msg, Style::default().fg(crate::theme::theme().warning)));
            // Interrupting a drive turn is an explicit "stop": pause the goal so
            // the drive doesn't restart on the next message. Progress is held;
            // `/goal resume` continues.
            if goal_drive_turn && agent.set_goal_paused(true) {
                app.push(Line::styled(
                    "goal drive interrupted — paused; /goal resume to continue".to_string(),
                    Style::default().fg(crate::theme::theme().warning),
                ));
            }
        } else {
            // Turn finished on its own — ping if you've likely stepped away.
            app.maybe_notify_done();
            // Capture which files this turn changed, so the "changed: …" line
            // above the input reflects the latest turn. The agent already
            // computed this for verify gating; reuse it rather than re-walking.
            app.last_changed_files = agent.last_changed_files().to_vec();
            app.accumulate_session_files();
            // Capture the turn's trajectory telemetry for the observability
            // panel (verify rounds, recovery retries, nudges, stalls).
            app.last_telemetry = Some(agent.last_turn_telemetry().clone());
            // A new turn's edits supersede any open diff panel's snapshot.
            app.diff_text = None;
        }
        if let Some(state) = restore_model_state.take() {
            agent.restore_model_state(state);
        }
        if let Some((model, context_window)) = restore_app_model.take() {
            app.model = model;
            app.context_window = context_window;
        }
        // The goal driver (`goal_turn_end`) may have advanced/failed a sub-goal
        // this turn — mirror the new state so the pinned block + header reflect it.
        app.refresh_goal(agent);
        // Record a main /goal that just reached a terminal state to the activity
        // feed (→ /digest), so the interactive autonomous producer joins loops +
        // fleet there instead of being the one hole.
        if let Some(before) = &goal_before
            && before.status == hi_agent::GoalStatus::Active
            && let Some(after) = agent.structured_goal()
            && matches!(
                after.status,
                hi_agent::GoalStatus::Done | hi_agent::GoalStatus::Failed
            )
            && let Some(lf) = &fleet_launcher.loops_file
        {
            let verb = if after.status == hi_agent::GoalStatus::Done {
                "goal complete"
            } else {
                "goal failed"
            };
            let at_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            crate::activity::append(
                &crate::activity::activity_path(lf),
                &crate::activity::ActivityEntry {
                    at_ms,
                    loop_id: 0,
                    source: "goal".into(),
                    text: format!("{verb}: {}", after.objective),
                },
            );
        }
        // Long-horizon auto-drive: keep pulling toward an active goal between
        // turns. Drive turns that change nothing count toward a stall stop; any
        // user turn (steering) resets it. Queued user input always wins — the
        // drive prompt is only queued into an empty queue.
        if !cancelled {
            if goal_drive_turn {
                if agent.structured_goal().cloned() == goal_before {
                    app.goal_drive_stall += 1;
                    if app.goal_drive_stall == hi_agent::GOAL_DRIVE_STALL_LIMIT {
                        app.push(Line::styled(
                            "goal drive paused itself: no progress for 2 turns — send guidance \
                             (your next message resumes the drive), or /goal pause|clear"
                                .to_string(),
                            Style::default().fg(crate::theme::theme().warning),
                        ));
                    }
                } else {
                    app.goal_drive_stall = 0;
                }
            } else {
                app.goal_drive_stall = 0;
            }
            app.maybe_queue_goal_drive(agent);
        }
        app.set_working(false);
        // Flush any pending live events from the TUI's /sync on RemoteUi.
        // Spawn as a background task so a slow/unreachable ipop doesn't block
        // the TUI event loop (5s timeout). Errors are silent — events are
        // re-buffered on failure and retried on the next flush.
        if let Some(rui) = &app.sync_remote_ui {
            let rui = rui.clone();
            tokio::spawn(async move {
                let _ = rui.flush().await;
            });
        }
        // Flush the startup RemoteUi (created in main.rs) so live events are
        // actually streamed during the session, not just buffered until exit.
        if let Some(cb) = &app.remote_flush_callback {
            cb();
        }
        // No follow() at turn end: if the user scrolled up to read mid-turn, leave
        // them there (the "↓ N new" hint shows the summary is below). A new turn
        // re-pins to the bottom.
    }

    // Session ending: distill durable lessons into .hi/memory.md (loaded next
    // session), shown live so the user sees what's saved. Only if work happened.
    if hi_agent::should_distill_memory(auto_memory, agent.totals().output_tokens) {
        app.set_working(true);
        app.follow();
        let (tx, rx) = mpsc::unbounded_channel();
        let (confirm_tx, confirm_rx) = mpsc::unbounded_channel();
        let mut sink = ChannelUi {
            tx,
            confirmations: confirm_tx,
        };
        {
            let fut = async {
                agent.update_memory(&mut sink).await;
                Ok::<(), anyhow::Error>(())
            };
            let _ = drive(
                &mut terminal,
                &mut input_rx,
                &mut ticker,
                &mut app,
                rx,
                confirm_rx,
                fut,
                false,
                None,
            )
            .await;
        }
        // Flush any pending live events from the TUI's /sync on RemoteUi.
        // Spawn as a background task so a slow/unreachable ipop doesn't block
        // the TUI event loop.
        if let Some(rui) = &app.sync_remote_ui {
            let rui = rui.clone();
            tokio::spawn(async move {
                let _ = rui.flush().await;
            });
        }
        if let Some(cb) = &app.remote_flush_callback {
            cb();
        }
        app.set_working(false);
    }

    // Persist input history for next time.
    if let Some(path) = &history_path {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, app.input.history.join("\n"));
    }
    // Snapshot provider/model so the next bare `hi` in this workspace resumes
    // with the same routing (also written on /model and /provider changes).
    app.remember_session_routing();
    // Remove any remaining fleet worktrees (sessions stay on disk, resumable).
    crate::dashboard::cleanup_fleet(&mut app);

    Ok(())
}

/// Drive a model future (a turn or a compaction) to completion while keeping
/// the UI live: redraw + spin every tick, drain the agent's events, let the
/// user scroll/queue/cancel. Successful values are preserved so typed turn
/// outcomes, rather than UI prose, can drive final presentation.
struct DriveCompletion<T> {
    cancelled: bool,
    value: Option<T>,
}

#[allow(clippy::too_many_arguments)]
async fn drive<T>(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    input: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    mut rx: mpsc::UnboundedReceiver<UiEvent>,
    mut confirmations: mpsc::UnboundedReceiver<ConfirmationControl>,
    fut: impl std::future::Future<Output = Result<T>>,
    expect_turn_end: bool,
    // When set, plain-text lines submitted while the turn runs are injected
    // into the *current* turn (mid-turn steering) instead of queued for the
    // next one. Slash-commands always queue.
    interject: Option<hi_agent::InterjectionInbox>,
) -> Result<DriveCompletion<T>> {
    tokio::pin!(fut);
    let mut cancelled = false;
    let mut value = None;
    let mut last_activity = Instant::now();
    let mut watchdog_stuck = false;
    let watchdog_timeout = watchdog_stuck_timeout();
    let mut pending_confirmation: Option<ConfirmationControl> = None;
    let mut confirmations_open = true;
    loop {
        terminal.draw(|f| app.render(f))?;
        tokio::select! {
            result = &mut fut => {
                while let Ok(event) = rx.try_recv() {
                    if let Some(tap) = &app.remote_event_tap {
                        tap(&event);
                    }
                    app.apply(event);
                }
                match result {
                    Ok(result) => value = Some(result),
                    Err(err) => {
                        let (kind, guidance) = hi_agent::classify_error(&err);
                        if !matches!(app.last_turn_state, TurnState::Failed(_)) {
                            app.note_turn_failed(&format!("{err:#}"), kind, guidance);
                        }
                        if hi_agent::ui::error_counts_as_model_issue(&err) {
                            app.record_model_issue();
                        }
                    }
                }
                break;
            }
            Some(event) = rx.recv() => {
                last_activity = Instant::now();
                if let Some(tap) = &app.remote_event_tap {
                    tap(&event);
                }
                app.apply(event);
            }
            request = confirmations.recv(), if pending_confirmation.is_none() && confirmations_open => {
                match request {
                    Some(request) => {
                        // Session-wide `a` or path-scoped `p` auto-approve.
                        if app.should_auto_approve(&request.request) {
                            let _ = request.response.send(hi_agent::ConfirmationResult::Approved);
                        } else {
                            app.confirmation = Some(request.request.clone());
                            app.confirmation_scroll = 0;
                            pending_confirmation = Some(request);
                        }
                    }
                    None => confirmations_open = false,
                }
            }
            _ = ticker.tick() => {
                app.spinner = app.spinner.wrapping_add(1);
                app.drain_loops();
                let idle = last_activity.elapsed();
                app.waiting_for = Some(idle);
                // Only notify about a quiet backend while no tool is legitimately
                // running. This is only a soft wait notice.
                if expect_turn_end
                    && !watchdog_stuck
                    && app.current_tool.is_none()
                    && idle >= watchdog_timeout
                {
                    watchdog_stuck = true;
                    app.note_backend_waiting(idle, watchdog_timeout);
                }
            },
            maybe = input.recv() => {
                match maybe {
                    Some(Event::Mouse(mouse)) => app.handle_mouse(mouse),
                    Some(Event::Paste(text)) if pending_confirmation.is_none() => app.input.insert_str(&text),
                    Some(Event::Paste(_)) => {}
                    Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        if let Some(request) = pending_confirmation.take() {
                            match key.code {
                                KeyCode::Char('y') if !ctrl => {
                                    let _ = request.response.send(hi_agent::ConfirmationResult::Approved);
                                    app.confirmation = None;
                                }
                                KeyCode::Char('n') if !ctrl => {
                                    let _ = request.response.send(hi_agent::ConfirmationResult::Rejected);
                                    app.confirmation = None;
                                }
                                // "Always allow this session": approve this
                                // request AND auto-approve all subsequent ones
                                // without showing the modal. Removes the y-y-y-y
                                // fatigue during a heavy edit session.
                                KeyCode::Char('a') if !ctrl => {
                                    let _ = request.response.send(hi_agent::ConfirmationResult::Approved);
                                    app.auto_approve_session = true;
                                    app.confirmation = None;
                                    app.push(Line::styled(
                                        "auto-approve on for this session (approvals suppressed until quit)",
                                        Style::default().fg(crate::theme::theme().accent_success),
                                    ));
                                }
                                // Path-scoped auto-approve for file edits (`p`).
                                KeyCode::Char('p') if !ctrl => {
                                    if let hi_agent::ConfirmationRequest::FileEdit { path, .. } =
                                        &request.request
                                    {
                                        let prefix = App::auto_approve_prefix_for(path);
                                        app.add_auto_approve_path(path);
                                        let _ = request
                                            .response
                                            .send(hi_agent::ConfirmationResult::Approved);
                                        app.confirmation = None;
                                        app.push(Line::styled(
                                            format!(
                                                "auto-approve path '{prefix}/' for this session"
                                            ),
                                            Style::default()
                                                .fg(crate::theme::theme().accent_success),
                                        ));
                                    } else {
                                        // Not a file edit — keep the modal open.
                                        pending_confirmation = Some(request);
                                    }
                                }
                                KeyCode::Esc => {
                                    let _ = request.response.send(hi_agent::ConfirmationResult::Rejected);
                                    app.confirmation = None;
                                }
                                KeyCode::Char('c') if ctrl => {
                                    let _ = request.response.send(hi_agent::ConfirmationResult::Cancelled);
                                    app.confirmation = None;
                                    cancelled = true;
                                    break;
                                }
                                KeyCode::Up => {
                                    app.confirmation_scroll = app.confirmation_scroll.saturating_sub(1);
                                    pending_confirmation = Some(request);
                                }
                                KeyCode::Down => {
                                    app.confirmation_scroll = app.confirmation_scroll.saturating_add(1);
                                    pending_confirmation = Some(request);
                                }
                                KeyCode::PageUp => {
                                    app.confirmation_scroll = app.confirmation_scroll.saturating_sub(10);
                                    pending_confirmation = Some(request);
                                }
                                KeyCode::PageDown => {
                                    app.confirmation_scroll = app.confirmation_scroll.saturating_add(10);
                                    pending_confirmation = Some(request);
                                }
                                _ => pending_confirmation = Some(request),
                            }
                            continue;
                        }
                        match key.code {
                            KeyCode::Char('c') if ctrl => { cancelled = true; break; }
                            // Esc clears a half-typed queued command, or — when the
                            // input is empty — interrupts the current tool call
                            // (if one is running) or cancels the whole turn.
                            KeyCode::Esc if app.input.is_empty() => {
                                if app.current_tool.is_some() {
                                    // A tool is running: signal interrupt to skip
                                    // just this tool call, not the whole turn.
                                    if let Some(flag) = &app.interrupt {
                                        flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                    }
                                } else {
                                    cancelled = true;
                                    break;
                                }
                            }
                            KeyCode::Esc => app.input.clear(),
                            // A line submitted while a turn runs: `/copy` reads the
                            // selection synchronously; a plain-text line steers the
                            // *current* turn (interjection) when supported; any
                            // slash-command queues for the next turn.
                            _ => {
                            // Palette / global chords while a turn runs.
                            if app.palette.is_some() {
                                match app.palette.as_mut().unwrap().handle_key(&key) {
                                    crate::palette::PaletteOutcome::Continue => continue,
                                    crate::palette::PaletteOutcome::Closed => {
                                        app.palette = None;
                                        continue;
                                    }
                                    crate::palette::PaletteOutcome::Accept(cmd) => {
                                        app.palette = None;
                                        app.queue.push_back(cmd);
                                        continue;
                                    }
                                }
                            }
                            match app.dispatch_key(&key) {
                                crate::dispatch::DispatchResult::Handled => continue,
                                crate::dispatch::DispatchResult::OpenPalette => {
                                    app.palette = Some(crate::palette::CommandPalette::open());
                                    continue;
                                }
                                crate::dispatch::DispatchResult::Fallthrough => {}
                            }
                            if let Some(submitted) = app.edit_key(&key) {
                                match command::parse(&submitted) {
                                    Some(Command::Copy(arg)) => app.copy(&arg),
                                    None if interject.is_some() => {
                                        interject.as_ref().unwrap().push(submitted.clone());
                                        app.push(Line::styled(
                                            format!("✉ {submitted}  (steering this turn)"),
                                            Style::default().fg(crate::theme::theme().accent_system),
                                        ));
                                        app.follow();
                                    }
                                    _ => app.queue.push_back(submitted),
                                }
                            }
                            }
                        }
                    }
                    Some(Event::FocusGained) => app.set_focus(true),
                    Some(Event::FocusLost) => app.set_focus(false),
                    None => {
                        cancelled = true;
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
    app.waiting_for = None;
    app.confirmation = None;
    Ok(DriveCompletion { cancelled, value })
}

#[cfg(test)]
mod tests {
    use super::expand_file_mentions;

    #[test]
    fn expand_file_mentions_reads_existing_file() {
        let dir = std::env::temp_dir().join(format!("hi-tui-mention-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("foo.rs"), "fn main() {}").unwrap();
        let out = expand_file_mentions("look at @foo.rs", &dir);
        assert!(
            out.starts_with("look at @foo.rs"),
            "original text preserved"
        );
        assert!(
            out.contains("<file mention=\"foo.rs\">"),
            "file block added"
        );
        assert!(out.contains("fn main() {}"), "file contents injected");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expand_file_mentions_notes_missing_file() {
        let dir = std::env::temp_dir().join(format!("hi-tui-mention-miss-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = expand_file_mentions("fix @nope.rs", &dir);
        assert!(out.contains("not found"), "missing file noted");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expand_file_mentions_ignores_double_at() {
        let dir = std::env::temp_dir().join(format!("hi-tui-mention-at-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = expand_file_mentions("mention @@user", &dir);
        assert_eq!(out, "mention @@user", "@@ is literal, no expansion");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expand_file_mentions_no_at_returns_unchanged() {
        let dir = std::env::temp_dir().join("hi-tui-mention-none");
        let out = expand_file_mentions("just a plain prompt", &dir);
        assert_eq!(out, "just a plain prompt");
    }
}
