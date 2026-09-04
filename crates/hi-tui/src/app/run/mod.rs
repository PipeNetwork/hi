//! The TUI event loop: `run` (entry point that sets up the terminal, spawns
//! the agent turn behind a channel, and drives the render loop) and `drive`
//! (the per-event state machine that routes crossterm events to `App`).

mod auth;
mod drive;
mod helpers;
#[cfg(test)]
mod plan_approval_tests;
mod plan_input;
mod queue;
mod turn_execution;

use auth::{apply_tui_auth, parse_tui_auth_arg};
pub(crate) use drive::drive;
#[cfg(test)]
pub(crate) use helpers::search_transcript;
use helpers::{ChordPipeline, expand_file_mentions, run_chord_pipeline, run_shell_escape_async};
pub(crate) use helpers::{handle_normal_mode, review_next_hunk};
use plan_input::handle_idle_plan_approval_key;
use queue::reconcile_queue_with_interjections;
use turn_execution::run_agent_turn;

use std::io;
use std::io::IsTerminal;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{
    EnableBracketedPaste, EnableFocusChange, EnableMouseCapture, Event, EventStream, KeyCode,
    KeyEventKind, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, enable_raw_mode};
use futures_util::StreamExt;
use hi_agent::{Agent, Command, CompactionKind, command};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::style::Style;
use ratatui::text::Line;
use tokio::sync::mpsc;

use crate::event::{ChannelUi, Restore};
use crate::input::HistorySearch;
use crate::provider_form;
use crate::provider_picker;
use crate::render::dim;
use crate::{App, TICK, TurnState, apply_metadata};
/// Run the full-screen TUI until the user quits. `history_path`, if given, is
/// the file used to persist input history across sessions (shared with the
/// plain REPL). `profiles` is the list of configured profiles (for `/provider`
/// with no arg); `resolver` resolves a name to a built provider at runtime.
/// Drop guard that stops any hi-managed local model server when the TUI
/// session ends, covering every `return`/`break` exit path in [`run`]. The
/// registry contains only hi-owned skeptic and team-role servers, so a blanket
/// cleanup is correct.
struct LocalServerGuard;

impl Drop for LocalServerGuard {
    fn drop(&mut self) {
        hi_tools::stop_all_local_servers();
    }
}

fn rearm_owned_loop_manager(
    app: &mut App,
    launcher: &std::sync::Arc<crate::FleetLauncher>,
    event_sink: &Option<std::sync::Arc<dyn hi_events::EventSink>>,
    fire_lock: Option<std::sync::Arc<crate::lock::FireLock>>,
) -> bool {
    if app.loops.is_some() {
        return false;
    }
    app.loops = Some(crate::loops::start_with_fire_lock(
        launcher.clone(),
        launcher.loops_file.clone(),
        event_sink.clone(),
        fire_lock,
    ));
    true
}

fn ensure_owned_loop_fire_lock(
    loops_file: Option<&std::path::Path>,
    fire_lock: &mut Option<std::sync::Arc<crate::lock::FireLock>>,
) -> bool {
    let Some(loops_file) = loops_file else {
        return true;
    };
    if fire_lock.is_none() {
        *fire_lock =
            crate::lock::try_acquire(&crate::lock::lock_path(loops_file)).map(std::sync::Arc::new);
    }
    fire_lock.is_some()
}

/// An approval card owns the next action. Retain user work while it is visible,
/// and recheck synthetic continuations when dequeuing because a pause, mode
/// change, or plan replacement may have invalidated them after they were queued.
fn dequeue_ready_prompt(app: &mut App, agent: &Agent) -> Option<String> {
    if app.plan_approval_capturing() {
        return None;
    }
    while let Some(prompt) = app.queue.pop_front() {
        let kind = hi_agent::DriveKind::from_prompt(&prompt);
        if !kind.is_drive()
            || (app.plan_approval.is_none()
                && agent.explicit_goal_drive_decision() == hi_agent::DriveAction::Enqueue(kind))
        {
            return Some(prompt);
        }
    }
    None
}

pub async fn run(agent: &mut Agent, options: crate::RunOptions) -> Result<()> {
    agent.set_interactive_session(true);
    let crate::RunOptions {
        provider,
        base_url,
        model,
        history_path,
        auto_memory,
        profiles,
        active_profile,
        resolver,
        saver,
        loader,
        remover,
        reasoning_effort_saver,
        mlx_switcher,
        local_runtime_switcher,
        session_remember,
        resume_summary,
        mcp_url,
        api_key,
        diff_api_runner,
        race_runner,
        race_defaults,
        race_setup_saver,
        event_sink,
        approval_store,
        fleet_launcher,
        tui_event_trace,
        remote_event_tap,
        remote_flush_callback,
        sync_config,
        sync_session_id,
        session_lister,
        session_switcher,
        session_renamer,
        session_host,
        sync_control,
        pipefs_command,
        startup_local_runtime,
        startup_fallback_profile,
        x402_broker,
    } = options;

    if !io::stdin().is_terminal() {
        anyhow::bail!("TUI requires an interactive stdin");
    }

    enable_raw_mode().context("entering raw mode")?;
    // Install immediately after raw mode so any later startup error restores
    // the terminal before main falls back to plain mode.
    let _restore = Restore;
    // Tear down any auto-managed `/goal` or `/team` server on every exit path.
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
        &provider,
        &model,
        profiles,
        active_profile,
        resolver,
        saver,
        loader,
        remover,
        reasoning_effort_saver,
        mlx_switcher,
        local_runtime_switcher,
        mcp_url,
        api_key,
        diff_api_runner,
        race_runner,
        race_defaults,
        race_setup_saver,
    );
    app.configure_session_projection_v2(agent.harness_settings().features.session_projection_v2);
    // Install lifecycle tracing before startup restoration can enqueue a plan
    // or goal drive. Those real queue members must be traced before dequeue.
    app.tui_event_trace = tui_event_trace.clone();
    app.session_remember = session_remember;
    app.x402_broker = x402_broker;
    app.event_sink = event_sink.clone();
    app.approval_store = approval_store.clone();
    app.execution = agent.execution_mode();
    app.workspace_root = agent.workspace_root().to_path_buf();
    app.input_history_path = agent.state_root().join("input-history");
    app.local_startup_fallback_profile = startup_fallback_profile
        .filter(|name| app.active_profile.as_deref() != Some(name.as_str()));
    if let Some(runtime) = startup_local_runtime {
        app.local_startup_blocked = true;
        app.local_startup_spec = Some(runtime.clone());
        app.local_runtime = Some(crate::LocalRuntimeIdentity {
            backend: runtime.backend.serve_flag().to_ascii_uppercase(),
            model_id: runtime.model_id.clone(),
            quantization: runtime.quantization.clone(),
            source: match &runtime.source {
                hi_agent::local_skeptic::LocalModelSource::Hub { repo } => repo.clone(),
                hi_agent::local_skeptic::LocalModelSource::Directory { path } => {
                    path.display().to_string()
                }
            },
            endpoint: None,
            ready: false,
        });
        app.push(Line::styled(
            format!(
                "restoring local MLX · {} — the TUI is ready; use /local retry, /local fallback, or /quit",
                runtime.model_id
            ),
            dim(),
        ));
        app.start_local_runtime_provision(
            agent,
            format!("restoring {}", runtime.model_id),
            runtime,
        )
        .await;
    }
    // Keep prompt history in Hi's state root so entering a prompt cannot dirty
    // the project or trigger verification. Import the old workspace-local file
    // once when the new store has not been created yet.
    if app.input_history_path.exists() {
        app.input.load_history_file(&app.input_history_path);
    } else {
        app.input
            .load_history_file(&app.workspace_root.join(".hi").join("history"));
        app.input.save_history_file(&app.input_history_path);
    }
    app.plan = agent.current_plan().to_vec();
    app.resume_goal_drive(agent);
    app.sync_active = sync_config.is_some();
    app.sync_config = sync_config;
    app.sync_session_id = sync_session_id;
    app.session_lister = session_lister;
    if let Some(lister) = &app.session_lister {
        app.session_completion_cache = lister();
    }
    app.git_branch = crate::chrome::git_branch(&app.workspace_root);
    app.session_switcher = session_switcher;
    app.session_renamer = session_renamer;
    app.session_host = session_host;
    app.sync_control = sync_control;
    app.pipefs_command = pipefs_command;
    let remote_event_tap =
        crate::tui_event_trace::compose_remote_event_tap(remote_event_tap, tui_event_trace.clone());
    app.base_event_tap = remote_event_tap.clone();
    app.remote_event_tap = remote_event_tap;
    app.remote_flush_callback = remote_flush_callback;
    if app.sync_config.is_some() {
        app.sync_http = Some(
            reqwest::Client::builder()
                // Session listing and renaming run on the TUI command loop;
                // bound outages so the interface cannot appear frozen for
                // half a minute when portal sync is unreachable.
                .redirect(hi_ai::credential_redirect_policy())
                .connect_timeout(std::time::Duration::from_secs(3))
                .timeout(std::time::Duration::from_secs(8))
                .http1_only()
                .build()
                .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(3, 8)),
        );
        // Default: a synced TUI session is hosted (tmux-like). Other machines
        // with the same user API key can attach and steer over ipop without SSH.
        // User can `/sessions host off` to go portable-only. The enablement
        // registers with the portal over the network, so it runs as a
        // background task — an unreachable portal must never delay first
        // paint (observed: tens of seconds of startup hang while its
        // registration retried against a dead endpoint).
        if app.session_host.is_some() && app.sync_session_id.is_some() {
            app.start_host_enable_in_background();
        }
    }
    // Seed the context-fill gauge with the model's window so it reads 0% before
    // the first turn (it refreshes from real usage after each round).
    app.context_window = None;
    // Mirror the agent's reasoning effort so the title bar shows the
    // live level (it can be set before the TUI starts, e.g. via config).
    app.reasoning_effort = agent.reasoning_effort();
    // Load the on-disk /models cache so model metadata (window/price)
    // applies instantly at startup, without blocking on the network. The live
    // fetch still runs in the background and refreshes this; the cache just
    // covers the cold-start gap so the UI never looks idle.
    let models_cache_key = hi_ai::cache_key(&provider, &base_url);
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
        // Fresh sessions stay empty so the canvas can show the wordmark.
        // Chrome already has cwd, model, and shortcuts. Resume still gets a
        // one-line summary of what you're walking back into.
        if let Some(summary) = &resume_summary {
            app.push(Line::styled(summary.clone(), dim()));
        }
        if crate::tutorial::should_offer(
            resume_summary.is_none(),
            std::env::var_os("HI_SKIP_TUTORIAL").is_some(),
            crate::tutorial::already_offered(),
        ) {
            app.tutorial = Some(crate::tutorial::TutorialOverlay::fresh());
            crate::tutorial::mark_offered();
        }
    }
    // Session-start ghost text from git dirty files (cheap; no model call).
    // Post-turn suggestions replace this via UiEvent::SuggestedPrompt.
    if agent.config_snapshot().suggest_next_prompt
        && let Some(hint) = crate::startup_prompt_suggestion(&app.workspace_root)
    {
        app.suggested_prompt = Some(hint);
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
    let mut fleet_runtime = crate::dashboard::FleetRuntime::new();
    let mut fire_lock = None;
    let mut may_manage_loops =
        ensure_owned_loop_fire_lock(fleet_launcher.loops_file.as_deref(), &mut fire_lock);
    if agent.pipefs_workspace_active() {
        // Loop children capture the launch workspace and cannot join this
        // process's PipeFS lease/durability fence. In particular, persisted
        // autofix loops must never re-arm against the directory that PipeFS
        // promised to leave untouched. Keep any acquired fire lock while
        // suspended so `/pipefs off` can safely re-arm this TUI's manager.
        let (message, color) = if may_manage_loops {
            (
                "⟳ recurring loops are suspended while PipeFS is active".to_string(),
                crate::theme::theme().accent_system,
            )
        } else {
            let holder = fleet_launcher
                .loops_file
                .as_deref()
                .and_then(|path| crate::lock::live_holder(&crate::lock::lock_path(path)))
                .map(|pid| format!(" (pid {pid})"))
                .unwrap_or_default();
            (
                format!(
                    "⚠ recurring loops continue in an external daemon{holder} against the isolated launch workspace; stop that daemon before switching a local session into PipeFS"
                ),
                crate::theme::theme().warning,
            )
        };
        app.push(Line::styled(message, Style::default().fg(color)));
    } else if may_manage_loops {
        app.loops = Some(crate::loops::start_with_fire_lock(
            fleet_launcher.clone(),
            fleet_launcher.loops_file.clone(),
            event_sink.clone(),
            fire_lock.clone(),
        ));
    } else if let Some(loops_file) = &fleet_launcher.loops_file {
        let lock_path = crate::lock::lock_path(loops_file);
        let who = crate::lock::live_holder(&lock_path)
            .map(|pid| format!(" (pid {pid})"))
            .unwrap_or_default();
        app.push(Line::styled(
            format!(
                "⟳ loops are firing in a daemon{who} — /digest shows results; stop it to manage loops here"
            ),
            Style::default().fg(crate::theme::theme().accent_system),
        ));
    }
    // Startup timing checkpoint for `HI_STARTUP_TRACE=1`: everything above is
    // the blocking path to the first frame; everything network-shaped beyond
    // this point (models fetch, host enable, sync) races input or runs in
    // the background.
    if std::env::var_os("HI_STARTUP_TRACE").is_some() {
        eprintln!("[startup-tui] interactive (first frame ready)");
    }
    terminal.draw(|frame| app.render(frame))?;
    let first_frame = terminal.size()?;
    app.trace_ready(first_frame.width, first_frame.height)?;
    // Startup metadata fetch: race the live `/models` fetch against the first
    // keystroke, with a spinner ticking and the screen redrawing each tick so
    // the UI never looks idle. The on-disk cache already applied instantly
    // above; this just refreshes it. The fetch future is pinned locally (not
    // spawned — `Agent` isn't `Send`) and dropped before the main loop so its
    // borrow of `agent` doesn't block mutable uses during turns. A first input
    // event that wins the race is buffered for the main loop to process.
    let mut first_event: Option<Event> = None;
    let mut meta_result: Option<Result<Vec<hi_ai::ServedModel>>> = None;
    if app.context_window.is_none() && !app.local_startup_blocked {
        let meta_fut = agent.list_models();
        tokio::pin!(meta_fut);
        loop {
            terminal.draw(|f| app.render(f))?;
            tokio::select! {
                maybe = input_rx.recv() => {
                    let Some(event) = maybe else {
                        anyhow::bail!("terminal input reader stopped unexpectedly");
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
        app.check_tui_event_trace()?;
        // Run a queued command first (typed while the previous turn ran);
        // otherwise edit the input line until the user submits.
        let mut line_was_queued = false;
        let line = match dequeue_ready_prompt(&mut app, agent) {
            Some(queued) => {
                line_was_queued = true;
                app.trace_prompt_dequeued(&queued)?;
                // Hosted-steer mode: forward to the remote host over ipop.
                if app.maybe_forward_steered_prompt(&queued).await {
                    continue 'session;
                }
                queued
            }
            None => 'input: loop {
                let finished_diff_task = app.diff_lab.as_mut().and_then(|overlay| {
                    overlay
                        .task
                        .as_ref()
                        .is_some_and(tokio::task::JoinHandle::is_finished)
                        .then(|| overlay.task.take())
                        .flatten()
                });
                if let Some(task) = finished_diff_task {
                    let result = task.await;
                    if let Some(overlay) = app.diff_lab.as_mut() {
                        match result {
                            Ok(Ok(snapshot)) => {
                                overlay.snapshot = snapshot;
                                overlay.message = "run complete · n/r to run again".into();
                            }
                            Ok(Err(error)) => {
                                overlay.snapshot.status = hi_diff::RunStatus::Failed;
                                overlay.message = format!("run failed: {error:#}");
                            }
                            Err(error) => {
                                overlay.snapshot.status = hi_diff::RunStatus::Failed;
                                overlay.message = format!("run task failed: {error}");
                            }
                        }
                    }
                }
                let finished_race_task = app.race.as_mut().and_then(|overlay| {
                    overlay
                        .task
                        .as_ref()
                        .is_some_and(tokio::task::JoinHandle::is_finished)
                        .then(|| overlay.task.take())
                        .flatten()
                });
                if let Some(task) = finished_race_task {
                    let result = task.await;
                    if let Some(overlay) = app.race.as_mut() {
                        match result {
                            Ok(Ok(snapshot)) => {
                                overlay.snapshot = snapshot;
                                overlay.message = if overlay.snapshot.status
                                    == hi_race::RaceStatus::Ready
                                {
                                    "winner ready · inspect the candidates, then press a to apply"
                                        .into()
                                } else {
                                    "race complete · inspect the candidate results".into()
                                };
                            }
                            Ok(Err(error)) => {
                                overlay.snapshot.status = hi_race::RaceStatus::Failed;
                                overlay.message = format!("race failed: {error:#}");
                            }
                            Err(error) => {
                                overlay.snapshot.status = hi_race::RaceStatus::Failed;
                                overlay.message = format!("race task failed: {error}");
                            }
                        }
                    }
                }
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
                                                crate::dashboard::pump_fleet(
                                                    &mut app,
                                                    &fleet_launcher,
                                                    &mut fleet_runtime,
                                                ).await;
                                                app.drain_loops();
                            app.drain_voice();
                                                // Startup host-enable runs in the background;
                                                // apply its outcome once it lands.
                                                app.poll_pending_host_enable().await;
                                                // `/team` local-model provisioning finishes in
                                                // the background; wire the role when ready.
                                                app.poll_pending_team_provision(agent).await;
                                                app.poll_pending_local_provider(agent).await;
                                                app.poll_pending_local_catalog().await;
                                                if let Some(cmd) = app.poll_pending_login().await {
                                                    let _ = app.enqueue_prompt_front(cmd);
                                                    continue 'session;
                                                }
                                                // Host mode: pull any attach prompts into the
                                                // turn queue without a separate daemon process.
                                                if app.drain_remote_input() {
                                                    continue 'session;
                                                }
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
                        let Some(event) = event else {
                            anyhow::bail!("terminal input reader stopped unexpectedly");
                        };
                        event
                    }
                };
                if let Event::Key(key) = &event
                    && key.kind == KeyEventKind::Press
                    && app.plan_approval_visible()
                {
                    if let Some(prompt) = handle_idle_plan_approval_key(&mut app, agent, key) {
                        break 'input prompt;
                    }
                    continue 'input;
                }
                match event {
                    Event::Key(key) if key.kind == KeyEventKind::Press && app.race.is_some() => {
                        let close = app.race.as_mut().unwrap().handle_key(key);
                        if close {
                            app.race = None;
                        }
                        continue;
                    }
                    Event::Key(key)
                        if key.kind == KeyEventKind::Press && app.diff_lab.is_some() =>
                    {
                        let close = app.diff_lab.as_mut().unwrap().handle_key(key);
                        if close {
                            app.diff_lab = None;
                        }
                        continue;
                    }
                    Event::Paste(text) if app.diff_lab.is_some() => {
                        app.diff_lab.as_mut().unwrap().handle_paste(&text);
                        continue;
                    }
                    Event::Resize(width, height) => {
                        // Acknowledge SIGWINCH before accepting the harness's
                        // next action. The next input-loop iteration redraws at
                        // the new size; no user input is consumed or discarded.
                        app.trace_resized(width, height)?;
                        continue 'input;
                    }
                    Event::Mouse(mouse) => {
                        app.handle_mouse(mouse);
                        app.push_session_face(agent);
                    }
                    // A paste arrives as one event. Route it to whichever input
                    // surface is active: the provider form (its current field),
                    // or the main input line. Without this, a paste while the
                    // form is open silently went into the hidden main input.
                    Event::Paste(text) => {
                        if app.paste_plan_comment(&text) {
                            continue 'input;
                        } else if let Some(form) = app.provider_form.as_mut() {
                            form.insert_str(&text);
                        } else if let Some(path) = app.local_directory_prompt.as_mut() {
                            path.push_str(&text);
                        } else {
                            app.input.insert_str(&text);
                        }
                    }
                    Event::Key(key)
                        if key.kind == KeyEventKind::Press
                            && app.local_download_confirmation.is_some() =>
                    {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        match key.code {
                            KeyCode::Esc | KeyCode::Char('n') => {
                                app.local_download_confirmation = None;
                            }
                            KeyCode::Enter | KeyCode::Char('y') => {
                                if let Some(option) = app.local_download_confirmation.take() {
                                    let ram = hi_agent::local_skeptic::system_ram_gb();
                                    let Some(backend) =
                                        hi_agent::local_skeptic::detect_backend_cached()
                                    else {
                                        app.push(Line::styled(
                                            "no local MLX backend detected",
                                            dim(),
                                        ));
                                        continue;
                                    };
                                    match app.local_runtime_from_option(&option, ram, backend) {
                                        Ok(runtime) => {
                                            app.start_local_runtime_provision(
                                                agent,
                                                option.display_name.clone(),
                                                runtime,
                                            )
                                            .await;
                                        }
                                        Err(error) => app.push(Line::styled(
                                            format!("local model rejected: {error:#}"),
                                            Style::default().fg(crate::theme::theme().warning),
                                        )),
                                    }
                                }
                            }
                            KeyCode::Char('c') if ctrl => {
                                app.local_download_confirmation = None;
                            }
                            _ => {}
                        }
                        continue;
                    }
                    Event::Key(key)
                        if key.kind == KeyEventKind::Press
                            && app.local_directory_prompt.is_some() =>
                    {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        match key.code {
                            KeyCode::Esc => app.local_directory_prompt = None,
                            KeyCode::Enter => app.submit_local_directory_prompt(agent).await,
                            KeyCode::Backspace => {
                                if let Some(path) = app.local_directory_prompt.as_mut() {
                                    path.pop();
                                }
                            }
                            KeyCode::Char('c') if ctrl => app.local_directory_prompt = None,
                            KeyCode::Char(c) if !ctrl => {
                                if let Some(path) = app.local_directory_prompt.as_mut() {
                                    path.push(c);
                                }
                            }
                            _ => {}
                        }
                        continue;
                    }
                    Event::Key(key)
                        if key.kind == KeyEventKind::Press && app.local_picker.is_some() =>
                    {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        match key.code {
                            KeyCode::Esc => app.local_picker = None,
                            KeyCode::Char('c') if ctrl => app.local_picker = None,
                            KeyCode::Up => app.local_picker.as_mut().unwrap().up(),
                            KeyCode::Down => app.local_picker.as_mut().unwrap().down(),
                            KeyCode::PageUp => app.local_picker.as_mut().unwrap().page_up(),
                            KeyCode::PageDown => app.local_picker.as_mut().unwrap().page_down(),
                            KeyCode::Char('d') if !ctrl => app.begin_local_directory_prompt(),
                            KeyCode::Backspace => app.local_picker.as_mut().unwrap().backspace(),
                            KeyCode::Enter => {
                                let choice = app
                                    .local_picker
                                    .as_ref()
                                    .and_then(|picker| picker.current_choice());
                                match choice {
                                    Some(crate::local_picker::LocalChoice::ExistingDirectory) => {
                                        app.begin_local_directory_prompt();
                                    }
                                    Some(crate::local_picker::LocalChoice::Model(option)) => {
                                        app.local_picker = None;
                                        if !option.installed
                                            && option.download_bytes.is_some_and(|bytes| {
                                                bytes >= 2 * 1024 * 1024 * 1024
                                            })
                                        {
                                            app.local_download_confirmation = Some(option);
                                        } else {
                                            let ram = hi_agent::local_skeptic::system_ram_gb();
                                            let Some(backend) =
                                                hi_agent::local_skeptic::detect_backend_cached()
                                            else {
                                                app.push(Line::styled(
                                                    "no local MLX backend detected",
                                                    dim(),
                                                ));
                                                continue;
                                            };
                                            match app
                                                .local_runtime_from_option(&option, ram, backend)
                                            {
                                                Ok(runtime) => {
                                                    app.start_local_runtime_provision(
                                                        agent,
                                                        option.display_name.clone(),
                                                        runtime,
                                                    )
                                                    .await;
                                                }
                                                Err(error) => app.push(Line::styled(
                                                    format!("local model rejected: {error:#}"),
                                                    Style::default()
                                                        .fg(crate::theme::theme().warning),
                                                )),
                                            }
                                        }
                                    }
                                    None => {}
                                }
                            }
                            KeyCode::Char(c) if !ctrl => {
                                app.local_picker.as_mut().unwrap().insert(c)
                            }
                            _ => {}
                        }
                        continue;
                    }
                    Event::Key(key)
                        if key.kind == KeyEventKind::Press
                            && app.local_startup_blocked
                            && app.input.is_empty()
                            && app.provider_picker.is_none() =>
                    {
                        match key.code {
                            KeyCode::Char('r') => {
                                if let Some(runtime) = app.local_startup_spec.clone() {
                                    app.local_startup_error = None;
                                    app.start_local_runtime_provision(
                                        agent,
                                        format!("restoring {}", runtime.model_id),
                                        runtime,
                                    )
                                    .await;
                                }
                            }
                            KeyCode::Char('f') => {
                                if let Some(profile) = app.local_startup_fallback_profile.clone() {
                                    let _ = app.enqueue_prompt(format!("/provider {profile}"));
                                } else {
                                    app.push(Line::styled(
                                        "no fallback provider is configured — choose one with /provider",
                                        Style::default().fg(crate::theme::theme().warning),
                                    ));
                                }
                            }
                            _ => {
                                if let Some(line) = app.edit_key(&key) {
                                    break 'input line;
                                }
                                app.sync_completion_after_edit_key(&key, false);
                            }
                        }
                        continue;
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
                                let choice = app
                                    .provider_picker
                                    .as_ref()
                                    .and_then(|p| p.current_choice());
                                app.provider_picker = None;
                                match choice {
                                    Some(provider_picker::ProviderChoice::Named(name)) => {
                                        app.cancel_pending_local_provider_if_active();
                                        let _ = app.enqueue_prompt(format!("/provider {name}"));
                                    }
                                    Some(provider_picker::ProviderChoice::LocalModel(model)) => {
                                        app.start_local_provider_provision(agent, &model).await;
                                    }
                                    None => {}
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
                                // Cancel must clear the team routing state too,
                                // or the NEXT /model pick would assign a role.
                                KeyCode::Esc => app.close_picker(),
                                KeyCode::Char('c') if ctrl => app.close_picker(),
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
                                                    Style::default()
                                                        .fg(crate::theme::theme().warning),
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

                        // Shared palette + action/mode dispatch (idle path).
                        match run_chord_pipeline(&mut app, &key) {
                            Some(ChordPipeline::Continue) => continue 'input,
                            Some(ChordPipeline::OpenPalette) => {
                                app.palette = Some(crate::palette::CommandPalette::open());
                                continue 'input;
                            }
                            Some(ChordPipeline::PaletteAccept(cmd)) => {
                                // Load into input and submit via the normal path.
                                app.input.set(&cmd);
                                if cmd.ends_with(' ') {
                                    app.sync_completion();
                                } else {
                                    let line = app.input.submit();
                                    if !line.trim().is_empty() {
                                        break 'input line;
                                    }
                                }
                                continue 'input;
                            }
                            Some(ChordPipeline::KillTask(id)) => {
                                let message = agent.kill_background_task(&id).await;
                                if message.contains("cancelled") {
                                    crate::subagent_overlay::mark_cancelled(&mut app, &id);
                                }
                                app.push(Line::styled(message, dim()));
                                continue 'input;
                            }
                            Some(ChordPipeline::CycleSessionMode) => {
                                app.cycle_session_face();
                                app.push_session_face(agent);
                                continue 'input;
                            }
                            Some(ChordPipeline::PlanApprove) => {
                                if app.apply_plan_approve(agent)
                                    && let Some(prompt) =
                                        agent.explicit_goal_drive_decision().prompt()
                                {
                                    break 'input prompt.to_string();
                                }
                                continue 'input;
                            }
                            Some(ChordPipeline::PlanPark) => {
                                app.park_plan_approval(agent);
                                continue 'input;
                            }
                            Some(ChordPipeline::PlanRequestChanges) => {
                                app.apply_plan_request_changes(agent);
                                continue 'input;
                            }
                            Some(ChordPipeline::PlanQuit) => {
                                app.apply_plan_quit(agent);
                                continue 'input;
                            }
                            None => {}
                        }

                        let history_search_was_active = app.mode.is_history_search();
                        // Tab / Right on empty input accepts Claude-style ghost text
                        // when the `/` completion menu is not open.
                        if app.completion.is_none()
                            && app.ghost_suffix().is_some()
                            && matches!(key.code, KeyCode::Tab)
                        {
                            let _ = app.accept_suggested_prompt();
                            continue 'input;
                        }
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
                            KeyCode::Char('c') if ctrl && app.pending_local_provider.is_some() => {
                                app.cancel_pending_local_provider_if_active();
                            }
                            KeyCode::Esc if app.pending_local_provider.is_some() => {
                                app.cancel_pending_local_provider_if_active();
                            }
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
                                app.clear_suggested_prompt();
                                app.input.clear();
                            }
                            KeyCode::Esc => {
                                app.quit_notice = None;
                                if app.pending_auth.take().is_some() {
                                    app.input.secret = false;
                                    app.input.clear();
                                    app.push(Line::styled("/auth cancelled".to_string(), dim()));
                                    app.follow();
                                } else if app.ghost_suffix().is_some() && app.input.is_empty() {
                                    app.dismiss_suggested_prompt();
                                } else if app.show_help {
                                    app.show_help = false;
                                } else if app.mode.is_review() {
                                    app.mode.to_insert();
                                } else if app.dismiss_btw_overlay() {
                                } else if app.input.is_empty() && !app.working {
                                    if app.mode.is_normal() {
                                        app.mode.to_insert();
                                    } else {
                                        app.mode = crate::mode::UiMode::Normal { search: None };
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
                                app.sync_completion_after_edit_key(&key, history_search_was_active);
                            }
                        }
                    }
                    Event::FocusGained => app.set_focus(true),
                    Event::FocusLost => app.set_focus(false),
                    _ => {}
                }
            },
        };
        if !line_was_queued {
            app.trace_immediate_prompt(&line)?;
        }
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

        // TUI-local command: opt-in, fresh every time, and never persisted.
        if matches!(line.trim(), "/tutorial" | "/tour" | "/onboarding") {
            app.tutorial = Some(crate::tutorial::TutorialOverlay::fresh());
            continue;
        }

        if let Some(provider) = app.pending_auth.take() {
            apply_tui_auth(&mut app, &provider, line.trim()).await;
            continue;
        }

        // Slash commands. Most are handled inline; `/compact` runs a model call
        // (driven like a turn so the spinner shows); `/retry` yields the prompt
        // to re-run in the turn phase below.
        let mut restore_model_state: Option<hi_agent::AgentModelState> = None;
        let mut restore_app_model: Option<(String, Option<u32>)> = None;
        let run_line = if let Some(cmd) = command::parse(&line).map(command::resolve_command) {
            match cmd {
                Command::Quit => break,
                Command::Prompt(prompt) => {
                    let prompt = prompt.trim().to_string();
                    if prompt.is_empty() {
                        continue;
                    }
                    prompt
                }
                // `/btw` is mid-turn only. Idle, there is no side channel to
                // answer against — don't silently promote it to a full task turn.
                Command::Btw(question) => {
                    let question = question.trim();
                    if question.is_empty() {
                        app.push(Line::styled("usage: /btw <question>".to_string(), dim()));
                    } else {
                        app.push(Line::styled(
                            "/btw is mid-turn only — start a task, then ask aside".to_string(),
                            dim(),
                        ));
                    }
                    continue;
                }
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
                        tx: tx.clone(),
                        confirmations: confirm_tx,
                        event_sink: event_sink.clone(),
                        approval_store: approval_store.clone(),
                    };
                    {
                        let bg_tasks = agent.background_task_registry();
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
                            None,
                            tx,
                            None,
                            bg_tasks,
                        )
                        .await?;
                    }
                    app.set_working(false);
                    app.push_session_face(agent);
                    app.refresh_goal(agent);
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
                                    agent.register_driver_local_server(
                                        run.base_url.clone(),
                                        run.model_id.clone(),
                                        run.process_id.clone(),
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
                                    app.remember_session_routing(agent);
                                    app.push(Line::styled(
                                        format!(
                                            "using local MLX profile '{}' — model: {model}",
                                            run.profile_name
                                        ),
                                        dim(),
                                    ));
                                }
                                Err(err) => {
                                    hi_tools::stop_local_server(&run.process_id);
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
                                        None => anyhow::bail!(
                                            "terminal input reader stopped unexpectedly"
                                        ),
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
                    app.picker = Some(app.model_picker(ids, &current));
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
                    match arg.as_str() {
                        "xai" | "grok" => {
                            if hi_ai::xai_auth::has_credential() {
                                app.push(Line::styled(
                                    "already signed in to xAI — switching to it \
                                     (/logout xai first to use a different account)"
                                        .to_string(),
                                    dim(),
                                ));
                                app.follow();
                                let _ = app.enqueue_prompt_front("/provider xai");
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
                                    "approve in your browser — hi will switch to xAI when that lands"
                                        .to_string(),
                                    dim(),
                                ));
                                    app.follow();
                                    let task = tokio::spawn(async move {
                                        let token =
                                            hi_ai::xai_auth::poll_for_token(&device).await?;
                                        hi_ai::auth_store::save(
                                            hi_ai::xai_auth::PROVIDER_ID,
                                            &token,
                                        )?;
                                        Ok(())
                                    });
                                    if let Some((_, previous)) =
                                        app.pending_login.replace(("xai".into(), task))
                                    {
                                        previous.abort();
                                    }
                                }
                                Err(error) => {
                                    app.push(Line::styled(
                                        format!("/login failed: {error:#}"),
                                        dim(),
                                    ));
                                    app.follow();
                                }
                            }
                        }
                        "pipenetwork" | "pipe" => {
                            if hi_ai::pipenetwork_auth::has_credential() {
                                app.push(Line::styled(
                                    "already signed in to pipenetwork — switching to it \
                                     (/logout pipenetwork first to pair a different account)"
                                        .to_string(),
                                    dim(),
                                ));
                                app.follow();
                                let _ = app.enqueue_prompt_front("/provider pipenetwork");
                                continue;
                            }
                            match hi_ai::pipenetwork_auth::request_pairing().await {
                                Ok(issue) => {
                                    app.push(Line::styled(
                                        format!("open  {}", issue.url()),
                                        ratatui::style::Style::default()
                                            .add_modifier(ratatui::style::Modifier::BOLD),
                                    ));
                                    app.push(Line::styled(
                                        format!("code  {}", issue.user_code),
                                        ratatui::style::Style::default()
                                            .add_modifier(ratatui::style::Modifier::BOLD),
                                    ));
                                    app.push(Line::styled(
                                        "approve in your browser — hi will switch to pipenetwork when that lands"
                                            .to_string(),
                                        dim(),
                                    ));
                                    app.follow();
                                    let task = tokio::spawn(async move {
                                        let token =
                                            hi_ai::pipenetwork_auth::poll_for_key(&issue).await?;
                                        hi_ai::auth_store::save(
                                            hi_ai::pipenetwork_auth::PROVIDER_ID,
                                            &token,
                                        )?;
                                        Ok(())
                                    });
                                    if let Some((_, previous)) =
                                        app.pending_login.replace(("pipenetwork".into(), task))
                                    {
                                        previous.abort();
                                    }
                                }
                                Err(error) => {
                                    app.push(Line::styled(
                                        format!("/login failed: {error:#}"),
                                        dim(),
                                    ));
                                    app.follow();
                                }
                            }
                        }
                        "x402" => {
                            let keypair = std::env::var("HI_X402_KEYPAIR")
                                .ok()
                                .map(|value| value.trim().to_string())
                                .filter(|value| !value.is_empty());
                            if let Some(path) = keypair {
                                match hi_ai::validate_keypair_file(std::path::Path::new(&path)) {
                                    Ok(()) => {
                                        app.push(Line::styled(
                                            format!(
                                                "x402 ready (keypair {path}) — first turn quotes USDC"
                                            ),
                                            dim(),
                                        ));
                                        app.follow();
                                        let _ = app.enqueue_prompt_front("/provider pipenetwork");
                                    }
                                    Err(error) => {
                                        app.push(Line::styled(
                                            format!("/login x402 failed: {error:#}"),
                                            dim(),
                                        ));
                                        app.follow();
                                    }
                                }
                            } else {
                                app.push(Line::styled(
                                    "set HI_X402_KEYPAIR to a Solana keypair JSON, or add \
                                     [x402] keypair in config.toml. Paste-sig works in `hi --plain`."
                                        .to_string(),
                                    dim(),
                                ));
                                app.follow();
                            }
                        }
                        "" => {
                            app.push(Line::styled(
                                "usage: /login xai | /login pipenetwork | /login x402".to_string(),
                                dim(),
                            ));
                            app.follow();
                        }
                        other => {
                            app.push(Line::styled(
                                format!("'{other}' has no sign-in; try xai, pipenetwork, or x402"),
                                dim(),
                            ));
                            app.follow();
                        }
                    }
                    continue;
                }
                Command::Logout(arg) => {
                    let arg = arg.trim().to_string();
                    let message = match arg.as_str() {
                        "xai" | "grok" => {
                            if app
                                .pending_login
                                .as_ref()
                                .is_some_and(|(provider, _)| provider == "xai")
                                && let Some((_, task)) = app.pending_login.take()
                            {
                                task.abort();
                            }
                            match hi_ai::xai_auth::logout_quiet() {
                                Ok(true) => "signed out of xAI".to_string(),
                                Ok(false) => "not signed in to xAI".to_string(),
                                Err(error) => format!("/logout failed: {error:#}"),
                            }
                        }
                        "pipenetwork" | "pipe" => {
                            if app
                                .pending_login
                                .as_ref()
                                .is_some_and(|(provider, _)| provider == "pipenetwork")
                                && let Some((_, task)) = app.pending_login.take()
                            {
                                task.abort();
                            }
                            match hi_ai::pipenetwork_auth::logout_quiet() {
                                Ok(true) => {
                                    if hi_ai::has_credit_token() {
                                        "signed out of pipenetwork pairing; x402 credit token still stored — /logout x402"
                                            .to_string()
                                    } else {
                                        "signed out of pipenetwork".to_string()
                                    }
                                }
                                Ok(false) => "not signed in to pipenetwork".to_string(),
                                Err(error) => format!("/logout failed: {error:#}"),
                            }
                        }
                        "x402" => match hi_ai::x402_logout_quiet() {
                            Ok(true) => "cleared pipenetwork x402 credit token".to_string(),
                            Ok(false) => "no pipenetwork x402 credit token stored".to_string(),
                            Err(error) => format!("/logout failed: {error:#}"),
                        },
                        _ => "usage: /logout xai | /logout pipenetwork | /logout x402".to_string(),
                    };
                    app.push(Line::styled(message, dim()));
                    app.follow();
                    continue;
                }
                Command::Auth(arg) => {
                    match parse_tui_auth_arg(&arg) {
                        Ok((provider, Some(key))) => {
                            apply_tui_auth(&mut app, &provider, &key).await;
                        }
                        Ok((provider, None)) => {
                            app.pending_auth = Some(provider.clone());
                            app.input.secret = true;
                            app.push(Line::styled(
                                format!("paste the {provider} API key (not saved to history)"),
                                dim(),
                            ));
                            app.follow();
                        }
                        Err(message) => {
                            app.push(Line::styled(message, dim()));
                            app.follow();
                        }
                    }
                    continue;
                }
                Command::Provider(arg) => {
                    let arg = arg.trim().to_string();
                    if arg == "cancel" {
                        app.cancel_pending_local_provider();
                        continue;
                    }
                    if !arg.is_empty() {
                        app.cancel_pending_local_provider_if_active();
                    }
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
                        let current = app
                            .active_profile
                            .clone()
                            .unwrap_or_else(|| app.provider.clone());
                        app.provider_picker =
                            Some(provider_picker::ProviderPicker::new_with_profile_infos(
                                app.profiles.clone(),
                                provider_picker::local_model_rows(),
                                &current,
                            ));
                        app.start_local_catalog_refresh();
                        app.push(Line::styled(
                            "local models are sized to this machine; refreshing Pipe Network choices in the background…",
                            dim(),
                        ));
                        continue;
                    }
                    // Resolve the profile and update the provider.
                    if let Some(source) = app
                        .profiles
                        .iter()
                        .find(|profile| profile.name == arg)
                        .and_then(|profile| {
                            profile
                                .managed_local_path
                                .as_ref()
                                .map(|path| path.to_string_lossy().into_owned())
                                .or_else(|| profile.managed_local_repo.clone())
                        })
                    {
                        app.start_local_provider_provision(agent, &source).await;
                        continue;
                    }
                    match (app.resolver)(&arg) {
                        Ok(switched) => {
                            let label = switched.label.clone();
                            let model = switched.model.clone();
                            let needs_model = model == "__model_not_configured__";
                            // A local driver server is owned by the agent, not
                            // by the profile endpoint. Release it when the
                            // driver switches away; shared team-role routes
                            // keep it alive through Agent's reference checks.
                            agent.clear_driver_local_server();
                            agent.set_provider(
                                switched.provider.into(),
                                model.clone(),
                                None,
                                switched.max_tokens,
                                switched.max_tokens_explicit,
                                None,
                            );
                            agent.set_tool_mode(switched.tool_mode);
                            app.provider = label.clone();
                            app.model = model.clone();
                            app.active_profile = Some(arg.clone());
                            app.local_runtime = switched.local_runtime.clone();
                            app.local_startup_blocked = false;
                            app.local_startup_error = None;
                            app.local_startup_spec = None;
                            app.context_window = None;
                            app.served.clear();
                            app.remember_session_routing(agent);
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
                                                None => anyhow::bail!(
                                                    "terminal input reader stopped unexpectedly"
                                                ),
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
                            app.picker = Some(app.model_picker(ids, &current));
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
                Command::Local(arg) => {
                    let arg = arg.trim().to_string();
                    if arg.is_empty() {
                        app.open_local_picker();
                    } else if arg.eq_ignore_ascii_case("cancel") {
                        app.cancel_pending_local_provider();
                    } else if arg.eq_ignore_ascii_case("retry") {
                        if let Some(runtime) = app.local_startup_spec.clone() {
                            app.local_startup_blocked = true;
                            app.local_startup_error = None;
                            app.start_local_runtime_provision(
                                agent,
                                format!("restoring {}", runtime.model_id),
                                runtime,
                            )
                            .await;
                        } else {
                            app.push(Line::styled(
                                "no failed local startup to retry — use /local to choose a model",
                                dim(),
                            ));
                        }
                    } else if arg.eq_ignore_ascii_case("fallback") {
                        if let Some(profile) = app.local_startup_fallback_profile.clone() {
                            let _ = app.enqueue_prompt(format!("/provider {profile}"));
                        } else {
                            app.push(Line::styled(
                                "no fallback provider is configured — choose one with /provider",
                                Style::default().fg(crate::theme::theme().warning),
                            ));
                        }
                    } else if arg.eq_ignore_ascii_case("quit") {
                        break;
                    } else if arg.starts_with('~')
                        || arg.starts_with('.')
                        || arg.contains(std::path::MAIN_SEPARATOR)
                    {
                        app.local_directory_prompt = Some(arg);
                        app.submit_local_directory_prompt(agent).await;
                    } else {
                        app.cancel_pending_local_provider_if_active();
                        app.start_local_provider_provision(agent, &arg).await;
                    }
                    continue;
                }
                Command::Pipefs(arg) => {
                    let operation = arg
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    let may_activate = matches!(operation.as_str(), "on" | "enable" | "retry");
                    let turning_off = matches!(operation.as_str(), "off" | "disable");

                    if may_activate && !agent.pipefs_workspace_active() {
                        may_manage_loops = ensure_owned_loop_fire_lock(
                            fleet_launcher.loops_file.as_deref(),
                            &mut fire_lock,
                        );
                        if !may_manage_loops {
                            let holder = fleet_launcher
                                .loops_file
                                .as_deref()
                                .and_then(|path| {
                                    crate::lock::live_holder(&crate::lock::lock_path(path))
                                })
                                .map(|pid| format!(" (pid {pid})"))
                                .unwrap_or_default();
                            app.push(Line::styled(
                                format!(
                                    "PipeFS activation blocked: recurring loops are running in another HI process{holder}; stop it and retry"
                                ),
                                Style::default().fg(crate::theme::theme().warning),
                            ));
                            continue;
                        }
                    }

                    if may_activate
                        && (app.race.as_ref().is_some_and(|race| race.task.is_some())
                            || app.plan_workflow_child.is_some()
                            || !fleet_runtime.is_idle(&app))
                    {
                        app.push(Line::styled(
                            "finish or cancel active race, fleet, and workflow children before enabling or recovering PipeFS; their launchers are bound to the launch workspace",
                            Style::default().fg(crate::theme::theme().warning),
                        ));
                        continue;
                    }

                    let mut manager_stopped = false;
                    if may_activate && let Some(loops) = app.loops.take() {
                        app.push(Line::styled(
                            "⟳ stopping recurring-loop children before switching workspaces…",
                            Style::default().fg(crate::theme::theme().accent_system),
                        ));
                        terminal.draw(|frame| app.render(frame))?;
                        // Drain completed notices before consuming the handle;
                        // shutdown then cancels and joins every manager-owned
                        // firing, trigger, and auto-fix child.
                        for (line, _) in loops.drain() {
                            app.push(Line::styled(line, dim()));
                        }
                        match loops.shutdown().await {
                            Ok(()) => manager_stopped = true,
                            Err(error) => {
                                app.push(Line::styled(
                                    format!(
                                        "PipeFS activation blocked: recurring-loop shutdown was not acknowledged ({error})"
                                    ),
                                    Style::default().fg(crate::theme::theme().warning),
                                ));
                                continue;
                            }
                        }
                    }

                    app.handle_command(agent, Command::Pipefs(arg)).await;
                    let pipefs_active = agent.pipefs_workspace_active();
                    let launch_workspace_active =
                        agent.workspace_root() == fleet_launcher.workspace_root;
                    if turning_off && !pipefs_active && launch_workspace_active {
                        may_manage_loops = ensure_owned_loop_fire_lock(
                            fleet_launcher.loops_file.as_deref(),
                            &mut fire_lock,
                        );
                    }
                    if may_activate && manager_stopped {
                        if pipefs_active {
                            app.push(Line::styled(
                                "⟳ recurring loops suspended while PipeFS is active; /pipefs off will re-arm them",
                                Style::default().fg(crate::theme::theme().accent_system),
                            ));
                        } else if launch_workspace_active
                            && may_manage_loops
                            && rearm_owned_loop_manager(
                                &mut app,
                                &fleet_launcher,
                                &event_sink,
                                fire_lock.clone(),
                            )
                        {
                            app.push(Line::styled(
                                "⟳ PipeFS was not activated; recurring loops re-armed",
                                Style::default().fg(crate::theme::theme().accent_system),
                            ));
                        }
                    } else if turning_off
                        && !pipefs_active
                        && launch_workspace_active
                        && may_manage_loops
                        && rearm_owned_loop_manager(
                            &mut app,
                            &fleet_launcher,
                            &event_sink,
                            fire_lock.clone(),
                        )
                    {
                        app.push(Line::styled(
                            "⟳ PipeFS is off; recurring loops re-armed in the original workspace",
                            Style::default().fg(crate::theme::theme().accent_system),
                        ));
                    }
                    continue;
                }
                // `/loop`: recurring agent turns on a cadence (manager task).
                Command::Loop(arg) => {
                    if app.loops.is_none() {
                        let message = if agent.pipefs_workspace_active() {
                            "loops are suspended while PipeFS is active; /pipefs off re-arms them when this TUI owns the fire lock"
                        } else {
                            "loops are managed by a background daemon — stop it to manage them here, or use /digest to see what they've noticed"
                        };
                        app.push(Line::styled(message.to_string(), dim()));
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
                                                "✓ loop#{} armed — every {}, runs until cancelled, firing now: {}",
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
                                    Ok(true) => (
                                        format!("✓ loop#{id} cancelled"),
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
                                            let lifetime = l
                                                .expires_ms
                                                .map(|expires| {
                                                    format!(
                                                        "expires {}h",
                                                        expires.saturating_sub(now) / 3_600_000
                                                    )
                                                })
                                                .unwrap_or_else(|| "no expiry".to_string());
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
                                                    "  #{} every {} · {} · {} firing(s){}{} · {} · {}",
                                                    l.id,
                                                    crate::loops::humanize_secs(l.interval_secs),
                                                    next,
                                                    l.firings,
                                                    cost,
                                                    marks,
                                                    lifetime,
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
                                    Ok(true) => (
                                        format!("✓ loop#{id} {verb}"),
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
                                    (Ok(true), None) => (
                                        format!("✓ loop#{id} budget cleared"),
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
                                    (Ok(true), false) => (
                                        format!("✓ loop#{id} trigger cleared"),
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
                                    Ok(true) => (
                                        format!("✓ loop#{id} auto-fix off"),
                                        crate::theme::theme().accent_success,
                                    ),
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
                                                None => anyhow::bail!(
                                                    "terminal input reader stopped unexpectedly"
                                                ),
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
                            let round_limit = max_rounds.map_or_else(
                                || "unlimited rounds".to_string(),
                                |max| format!("max {max} rounds"),
                            );
                            app.push(Line::styled(
                                format!("trio: plan ready, executing ({round_limit})"),
                                Style::default().fg(crate::theme::theme().accent_system),
                            ));
                            app.follow();

                            // ── Execute → Review loop ────────────────────────
                            let mut round: u64 = 0;
                            let mut last_objections: Vec<String> = Vec::new();
                            let mut approved = false;
                            let mut loop_stopped = false;
                            while !drive::trio_round_cap_reached(round, max_rounds) {
                                round = round.saturating_add(1);
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
                                app.push_session_face(agent);
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
                                let turn_cancel = hi_agent::TurnCancellation::new();
                                let (tx, rx) = mpsc::unbounded_channel();
                                let (confirm_tx, confirm_rx) = mpsc::unbounded_channel();
                                let mut sink = ChannelUi {
                                    tx: tx.clone(),
                                    confirmations: confirm_tx,
                                    event_sink: event_sink.clone(),
                                    approval_store: approval_store.clone(),
                                };
                                let _background_before = agent.background_process_ids();
                                let interject = agent.interjection_inbox();
                                let btw = agent.btw_dispatcher();
                                let driven = {
                                    let bg_tasks = agent.background_task_registry();
                                    let fut = agent.run_turn_cancellable(
                                        &run_line,
                                        &mut sink,
                                        turn_cancel.clone(),
                                    );
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
                                        Some(btw),
                                        tx,
                                        Some(turn_cancel.clone()),
                                        bg_tasks,
                                    )
                                    .await?
                                };
                                // A cancel key can race a turn that has already
                                // committed. `run_turn_cancellable` returns a
                                // typed Cancelled outcome when cancellation won;
                                // a Completed value means the body won first and
                                // must not be rewound by the frontend.
                                let shared_token_cancelled = turn_cancel.is_cancelled();
                                let stop_requested = driven.cancelled || shared_token_cancelled;
                                let settled_status = driven
                                    .value
                                    .as_ref()
                                    .map(|outcome| outcome.status)
                                    .or_else(|| {
                                        if shared_token_cancelled {
                                            agent.last_turn_outcome().map(|outcome| outcome.status)
                                        } else {
                                            None
                                        }
                                    });
                                let cancellation = drive::settle_turn_cancellation(
                                    driven.cancelled,
                                    shared_token_cancelled,
                                    settled_status,
                                );
                                let cancelled = cancellation.cancelled;
                                let agent_already_cleaned = cancellation.agent_already_cleaned;
                                if let Some(outcome) = &driven.value {
                                    app.note_turn_outcome(outcome);
                                } else if agent_already_cleaned {
                                    if let Some(outcome) = agent.last_turn_outcome() {
                                        app.note_turn_outcome(outcome);
                                    }
                                } else if !cancelled {
                                    let outcome = agent
                                        .cleanup_turn(hi_agent::TurnCleanupKind::Fail)
                                        .await
                                        .map(|r| r.outcome)
                                        .unwrap_or_else(|_| {
                                            agent.finalize_failed_turn_snapshot_only()
                                        });
                                    app.note_turn_outcome(&outcome);
                                }
                                app.set_working(false);
                                app.interrupt = None;

                                if cancelled {
                                    // Full cancellation cleanup — same as the
                                    // main turn path: kill bg processes, rewind
                                    // session state, finalize the cancellation.
                                    // When cooperative cancel already returned a
                                    // Cancelled outcome, the agent undid its own
                                    // checkpoints — skip a second undo.
                                    if !agent_already_cleaned
                                        && agent.checkpoint_count() > checkpoint_count
                                        && let Err(err) = agent.undo().await
                                    {
                                        app.push(Line::styled(
                                            format!("couldn't roll back interrupted workspace edits: {err:#}"),
                                            Style::default().fg(crate::theme::theme().warning),
                                        ));
                                    }
                                    if !agent_already_cleaned
                                        && let Err(err) = agent
                                            .rewind_to_snapshot_durable(checkpoint, &turn_snapshot)
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
                                    let killed = if agent_already_cleaned {
                                        0
                                    } else {
                                        match agent
                                            .cleanup_turn(hi_agent::TurnCleanupKind::Cancel {
                                                session: hi_agent::SessionRollback::AlreadyApplied,
                                            })
                                            .await
                                        {
                                            Ok(r) => {
                                                app.note_turn_outcome(&r.outcome);
                                                r.killed_backgrounds
                                            }
                                            Err(err) => {
                                                app.last_turn_state = TurnState::Cancelled;
                                                app.status = "cancelled".to_string();
                                                app.push(Line::styled(
                                                    format!("couldn't finalize typed cancellation outcome: {err:#}"),
                                                    Style::default()
                                                        .fg(crate::theme::theme().warning),
                                                ));
                                                0
                                            }
                                        }
                                    };
                                    let msg = if killed > 0 {
                                        format!(
                                            "trio: cancelled; killed {killed} background process(es)"
                                        )
                                    } else {
                                        "trio: cancelled".to_string()
                                    };
                                    app.push(Line::styled(msg, dim()));
                                    loop_stopped = true;
                                    break;
                                }

                                // Capture the settled turn before deciding
                                // whether a late stop permits the review phase.
                                // This keeps changed-file and telemetry state
                                // accurate when the body committed just before
                                // Ctrl-C/deadline.
                                app.maybe_notify_done();
                                app.last_changed_files = agent.last_changed_files().to_vec();
                                app.accumulate_session_files();
                                app.last_telemetry = Some(agent.last_turn_telemetry().clone());
                                app.last_turn_phase = Some(agent.turn_phase().label());
                                app.diff_text = None;
                                app.push_session_face(agent);
                                app.refresh_goal(agent);

                                if driven.value.is_none() {
                                    let message = if stop_requested {
                                        "trio: deadline reached while execution failed; stopping before review"
                                    } else {
                                        "trio: execution failed; stopping before review"
                                    };
                                    app.push(Line::styled(
                                        message.to_string(),
                                        Style::default().fg(crate::theme::theme().warning),
                                    ));
                                    loop_stopped = true;
                                    break;
                                }

                                if stop_requested {
                                    // Cancellation/deadline arrived after the
                                    // turn committed. Preserve that result, but
                                    // do not launch the review/repair round the
                                    // user (or deadline) just stopped.
                                    app.push(Line::styled(
                                        "trio: stop arrived after the round committed".to_string(),
                                        dim(),
                                    ));
                                    loop_stopped = true;
                                    break;
                                }

                                let non_reviewable_status =
                                    driven.value.as_ref().and_then(|outcome| {
                                        drive::trio_non_reviewable_status(outcome.status)
                                    });
                                if let Some(status) = non_reviewable_status {
                                    app.push(Line::styled(
                                        format!("trio: execution {status}; stopping before review"),
                                        Style::default().fg(crate::theme::theme().warning),
                                    ));
                                    loop_stopped = true;
                                    break;
                                }

                                // ── Review phase: side-call to reviewer ──────
                                // Cancellable via Esc/Ctrl-C (fail-open on cancel
                                // — treat as approved so the loop exits cleanly).
                                app.push(Line::styled(
                                    format!(
                                        "trio: reviewing round {}…",
                                        drive::trio_round_label(round, max_rounds)
                                    ),
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
                                                    None => anyhow::bail!(
                                                        "terminal input reader stopped unexpectedly"
                                                    ),
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
                                                "✓ trio: approved in round {}",
                                                drive::trio_round_label(round, max_rounds)
                                            ),
                                            Style::default()
                                                .fg(crate::theme::theme().accent_success),
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
                                                Style::default()
                                                    .fg(crate::theme::theme().accent_error),
                                            ));
                                        }
                                        app.follow();
                                        loop_stopped = true;
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
                            if !approved
                                && !loop_stopped
                                && let Some(max_rounds) = max_rounds
                            {
                                debug_assert!(drive::trio_round_cap_reached(
                                    round,
                                    Some(max_rounds)
                                ));
                                app.push(Line::styled(
                                    format!("trio: hit round cap ({max_rounds}) without approval"),
                                    Style::default().fg(crate::theme::theme().warning),
                                ));
                            }
                            app.follow();
                            continue;
                        }
                        command::LoopArg::Invalid(msg) => {
                            app.push(Line::styled(
                                msg,
                                Style::default().fg(crate::theme::theme().warning),
                            ));
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
                    if agent.pipefs_workspace_active() {
                        app.push(Line::styled(
                            "/fleet is unavailable while PipeFS is active because its launcher is bound to the launch workspace",
                            Style::default().fg(crate::theme::theme().warning),
                        ));
                        app.follow();
                        continue;
                    }
                    match arg.trim() {
                        "" => {
                            crate::dashboard::run_dashboard(
                                &mut terminal,
                                &mut input_rx,
                                &mut ticker,
                                &mut app,
                                &fleet_launcher,
                                &mut fleet_runtime,
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
                                        &mut fleet_runtime,
                                        Some(info),
                                    )
                                    .await?;
                                }
                                None => {
                                    app.push(Line::styled(
                                        if id.is_empty() {
                                            "no fleet sessions to resume — /fleet to dispatch some"
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
                                    "no fleet sessions in this project yet — /fleet to dispatch some"
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
                // `/workflow`: scripted multi-phase agent orchestration.
                // list/show/validate print to the transcript; run launches the
                // engine with a live host bridge that spawns real FleetRows.
                Command::Workflow(arg) => {
                    if agent.pipefs_workspace_active() {
                        app.push(Line::styled(
                            "/workflow is unavailable while PipeFS is active because its child launcher is bound to the launch workspace",
                            Style::default().fg(crate::theme::theme().warning),
                        ));
                        app.follow();
                        continue;
                    }
                    let arg = arg.trim();
                    // `/workflow plan …` drives the local plan-objectives
                    // engine as a detached `hi workflow run` child.
                    if let Some(rest) = arg.strip_prefix("plan")
                        && (rest.is_empty() || rest.starts_with(char::is_whitespace))
                    {
                        crate::workflow_tui::handle_plan_workflow(
                            &mut app,
                            rest,
                            &fleet_launcher.exe,
                            fleet_launcher.model_step_limit(),
                            fleet_launcher.model_tool_call_limit(),
                            fleet_launcher.model_verify_repair_limit(),
                        );
                        continue;
                    }
                    let is_run = !matches!(
                        arg.split_whitespace().next(),
                        Some(
                            "runs"
                                | "list"
                                | "ls"
                                | "show"
                                | "validate"
                                | "status"
                                | "stop"
                                | "pause"
                                | "resume"
                                | "delete"
                        )
                    ) && !arg.is_empty();
                    if is_run {
                        // Start the workflow run and open the dashboard so the
                        // host bridge can spawn real FleetRows.
                        match crate::workflow_tui::start_workflow_run(&mut app, arg).await {
                            Ok(()) => {
                                crate::dashboard::run_dashboard(
                                    &mut terminal,
                                    &mut input_rx,
                                    &mut ticker,
                                    &mut app,
                                    &fleet_launcher,
                                    &mut fleet_runtime,
                                    None,
                                )
                                .await?;
                            }
                            Err(err) => {
                                app.push(Line::styled(
                                    format!("✗ workflow failed to start: {err:#}"),
                                    ratatui::style::Style::default()
                                        .fg(crate::theme::theme().accent_error),
                                ));
                                app.follow();
                            }
                        }
                    } else {
                        crate::workflow_tui::handle_workflow_tui(&mut app, arg);
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
                // `/goal --workflow <plan.md>`: detach the existing plan runner.
                // Do not also install an in-session structured goal.
                Command::Goal(arg)
                    if hi_agent::command::parse_goal_objective_flags(&arg).workflow =>
                {
                    if agent.pipefs_workspace_active() {
                        app.push(Line::styled(
                            "/goal --workflow is unavailable while PipeFS is active because its child launcher is bound to the launch workspace",
                            Style::default().fg(crate::theme::theme().warning),
                        ));
                        app.follow();
                        continue;
                    }
                    let flags = hi_agent::command::parse_goal_objective_flags(&arg);
                    match hi_agent::goal_workflow_plan_path(
                        false,
                        agent.workspace_root(),
                        &flags.text,
                    ) {
                        Ok(path) => {
                            let plan = agent.workspace_root().join(&path);
                            crate::workflow_tui::handle_plan_workflow(
                                &mut app,
                                &plan.to_string_lossy(),
                                &fleet_launcher.exe,
                                fleet_launcher.model_step_limit(),
                                fleet_launcher.model_tool_call_limit(),
                                fleet_launcher.model_verify_repair_limit(),
                            );
                        }
                        Err(err) => {
                            app.push(Line::styled(
                                err,
                                Style::default().fg(crate::theme::theme().warning),
                            ));
                            app.follow();
                        }
                    }
                    continue;
                }
                // `/goal <objective>`: decompose with the planner behind a spinner
                // (Esc cancels), then install the structured goal. Control
                // subcommands (clear/pause/resume/limit) and the no-planner case
                // stay on the sync handler.
                Command::Goal(arg)
                    if agent.has_planner() && hi_agent::command::goal_arg_is_objective(&arg) =>
                {
                    // Strip control flags before calling the planner. In
                    // particular, `/goal --review <objective>` should pause
                    // the installed plan for review; the planner must receive
                    // only the objective text, not the CLI flag itself.
                    let flags = hi_agent::command::parse_goal_objective_flags(&arg);
                    let objective = if flags.text.is_empty() {
                        arg.trim().to_string()
                    } else {
                        flags.text
                    };
                    let mut goal_argument = objective.clone();
                    if flags.unattended {
                        goal_argument = format!("--unattended {goal_argument}");
                    }
                    if flags.review {
                        goal_argument = format!("--review {goal_argument}");
                    }
                    if let Some(goal) = agent.try_ingest_goal(&objective) {
                        app.set_ingested_goal(agent, &goal_argument, goal);
                        agent.reset_goal_drive_stall();
                        app.maybe_queue_explicit_goal_drive(agent);
                        continue;
                    }
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
                                        None => anyhow::bail!(
                                            "terminal input reader stopped unexpectedly"
                                        ),
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
                    app.set_planned_goal(agent, &goal_argument, sub_goals);
                    // A goal is a contract: start pulling toward it immediately.
                    // The user monitors and steers — pause/Esc stops the drive.
                    agent.reset_goal_drive_stall();
                    app.maybe_queue_explicit_goal_drive(agent);
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
                        agent.reset_goal_drive_stall();
                        app.maybe_queue_explicit_goal_drive(agent);
                    }
                    continue;
                }
                other => {
                    app.handle_command(agent, other).await;
                    // `/config steps` mutates the in-process Agent. Keep the
                    // shared child launcher aligned so future loop/fleet turns
                    // do not retain a startup cap that the user turned off (or
                    // miss a cap that the user just opted into).
                    fleet_launcher.set_model_step_limit(agent.max_steps_limit());
                    continue;
                }
            }
        } else {
            line
        };

        if app.local_startup_blocked {
            app.push(Line::styled(
                "local MLX is still starting — use r to retry, f to continue with fallback, /local, /provider, or /quit",
                dim(),
            ));
            continue;
        }

        // Expand `@file` mentions: read each referenced file and append its
        // contents to the prompt so the model sees the file without a separate
        // `read` tool call. The `@path` tokens stay in the user-visible text
        // (so the transcript reads naturally); the contents are appended below
        // a clear separator. Missing/oversize files are noted inline.
        let run_line = expand_file_mentions(&run_line, &app.workspace_root);

        run_agent_turn(
            &mut terminal,
            &mut input_rx,
            &mut ticker,
            &mut app,
            agent,
            &run_line,
            restore_model_state,
            restore_app_model,
            fleet_launcher.loops_file.as_deref(),
        )
        .await?;
    }

    // Session ending: distill durable lessons into .hi/memory.md (loaded next
    // session), shown live so the user sees what's saved. Only if work happened.
    if hi_agent::should_distill_memory(auto_memory, agent.totals().output_tokens) {
        app.set_working(true);
        app.follow();
        let (tx, rx) = mpsc::unbounded_channel();
        let (confirm_tx, confirm_rx) = mpsc::unbounded_channel();
        let mut sink = ChannelUi {
            tx: tx.clone(),
            confirmations: confirm_tx,
            event_sink: event_sink.clone(),
            approval_store: approval_store.clone(),
        };
        {
            let bg_tasks = agent.background_task_registry();
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
                None,
                tx,
                None,
                bg_tasks,
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

    // Do not release the per-project fire lock while a detached loop child can
    // still be writing the launch workspace. The same acknowledged shutdown
    // used by `/pipefs on` also closes this ordinary TUI-exit race.
    if let Some(loops) = app.loops.take() {
        loops
            .shutdown()
            .await
            .map_err(|error| anyhow::anyhow!(error))
            .context("stopping recurring-loop children before TUI exit")?;
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
    app.remember_session_routing(agent);
    // Fleet rows own launch-workspace worktrees.  A PipeFS session never
    // starts them, and must not clean a stale launch-root callback after its
    // portable cache has been acknowledged and removed.
    if !agent.pipefs_workspace_active() {
        crate::dashboard::cleanup_fleet(&mut app);
    }

    app.trace_session_ended(agent)?;

    Ok(())
}
#[cfg(test)]
mod tests {
    use super::{
        ensure_owned_loop_fire_lock, expand_file_mentions, reconcile_queue_with_interjections,
    };
    use crate::tests::test_app;

    #[test]
    fn pipefs_loop_fence_requires_exclusive_fire_lock_ownership() {
        let dir = tempfile::tempdir().unwrap();
        let loops_file = dir.path().join("loops.json");
        let external_lock = crate::lock::try_acquire(&crate::lock::lock_path(&loops_file)).unwrap();
        let mut owned_lock = None;

        assert!(!ensure_owned_loop_fire_lock(
            Some(&loops_file),
            &mut owned_lock
        ));
        drop(external_lock);
        assert!(ensure_owned_loop_fire_lock(
            Some(&loops_file),
            &mut owned_lock
        ));
    }

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
            out.contains("foo.rs") && out.contains("<file mentions>"),
            "pointer note added"
        );
        assert!(
            !out.contains("fn main() {}"),
            "file body must not be inlined"
        );
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
    fn expand_file_mentions_rejects_paths_outside_workspace() {
        let base =
            std::env::temp_dir().join(format!("hi-tui-mention-escape-{}", std::process::id()));
        let root = base.join("root");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(base.join("secret.txt"), "secret").unwrap();
        let out = expand_file_mentions("read @../secret.txt", &root);
        assert!(out.contains("outside workspace"));
        assert!(!out.contains("\nsecret\n"));
        std::fs::remove_dir_all(&base).ok();
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

    #[test]
    fn reconcile_drops_consumed_mid_turn_lines_from_queue() {
        let mut app = test_app("p", "m");
        let inbox = hi_agent::InterjectionInbox::default();
        app.queue.push_back("first".into());
        app.queue.push_back("second".into());
        app.queue.push_back("/status".into());
        app.mid_turn_offered.push_back("first".into());
        app.mid_turn_offered.push_back("second".into());
        // Agent applied both mid-turn (inbox empty).
        reconcile_queue_with_interjections(&mut app, &inbox, true);
        assert_eq!(
            app.queue.iter().cloned().collect::<Vec<_>>(),
            vec!["/status".to_string()],
            "consumed steering lines leave the next-turn queue"
        );
        assert!(app.mid_turn_offered.is_empty());
    }

    #[test]
    fn reconcile_keeps_undrained_interjections_for_next_turn() {
        let mut app = test_app("p", "m");
        let inbox = hi_agent::InterjectionInbox::default();
        app.queue.push_back("keep me".into());
        app.mid_turn_offered.push_back("keep me".into());
        inbox.push("keep me");
        // Turn ended before Model drained the inbox.
        reconcile_queue_with_interjections(&mut app, &inbox, true);
        assert_eq!(
            app.queue.iter().cloned().collect::<Vec<_>>(),
            vec!["keep me".to_string()],
            "undrained steering still runs as the next prompt"
        );
        assert!(!inbox.has_pending(), "inbox drained into queue ownership");
    }

    #[test]
    fn reconcile_partial_consume_keeps_suffix() {
        let mut app = test_app("p", "m");
        let inbox = hi_agent::InterjectionInbox::default();
        app.queue.push_back("applied".into());
        app.queue.push_back("pending".into());
        app.mid_turn_offered.push_back("applied".into());
        app.mid_turn_offered.push_back("pending".into());
        inbox.push("pending");
        reconcile_queue_with_interjections(&mut app, &inbox, true);
        assert_eq!(
            app.queue.iter().cloned().collect::<Vec<_>>(),
            vec!["pending".to_string()]
        );
    }

    #[test]
    fn reconcile_keeps_consumed_interjection_when_drive_failed() {
        let mut app = test_app("p", "m");
        let inbox = hi_agent::InterjectionInbox::default();
        app.queue.push_back("retry after failure".into());
        app.mid_turn_offered.push_back("retry after failure".into());
        // The agent drained the inbox before its provider request failed.
        reconcile_queue_with_interjections(&mut app, &inbox, false);
        assert_eq!(
            app.queue.iter().cloned().collect::<Vec<_>>(),
            vec!["retry after failure".to_string()],
            "a failed drive must not discard queued user work"
        );
        assert!(app.mid_turn_offered.is_empty());
    }
}
