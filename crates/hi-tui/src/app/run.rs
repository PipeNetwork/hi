//! The TUI event loop: `run` (entry point that sets up the terminal, spawns
//! the agent turn behind a channel, and drives the render loop) and `drive`
//! (the per-event state machine that routes crossterm events to `App`).

use std::io;
use std::io::IsTerminal;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{
    EnableBracketedPaste, EnableFocusChange, Event, EventStream, KeyCode, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, enable_raw_mode};
use futures_util::StreamExt;
use hi_agent::{Agent, Command, CompactionKind, command};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use tokio::sync::mpsc;

use crate::event::{ChannelUi, Restore, UiEvent};
use crate::input::HistorySearch;
use crate::model_picker::ModelPicker;
use crate::provider_form;
use crate::render::dim;
use crate::{
    App, MlxProfileSwitcher, ProfileInfo, ProfileLoader, ProfileRemover, ProfileResolver,
    ProfileSaver, TICK, TurnState, apply_metadata, splash_lines, watchdog_stuck_timeout,
};

/// Run the full-screen TUI until the user quits. `history_path`, if given, is
/// the file used to persist input history across sessions (shared with the
/// plain REPL). `profiles` is the list of configured profiles (for `/provider`
/// with no arg); `resolver` resolves a name to a built provider at runtime.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    agent: &mut Agent,
    provider: &str,
    base_url: &str,
    model: &str,
    registry: &hi_ai::Registry,
    history_path: Option<std::path::PathBuf>,
    auto_memory: bool,
    profiles: Vec<ProfileInfo>,
    active_profile: Option<String>,
    resolver: ProfileResolver,
    saver: ProfileSaver,
    loader: ProfileLoader,
    remover: ProfileRemover,
    mlx_switcher: MlxProfileSwitcher,
    resume_summary: Option<String>,
    mcp_url: Option<String>,
    api_key: String,
    fleet_launcher: crate::FleetLauncher,
) -> Result<()> {
    if !io::stdin().is_terminal() {
        anyhow::bail!("TUI requires an interactive stdin");
    }

    enable_raw_mode().context("entering raw mode")?;
    // Install immediately after raw mode so any later startup error restores
    // the terminal before main falls back to plain mode.
    let _restore = Restore;
    execute!(io::stdout(), EnterAlternateScreen).context("entering alternate screen")?;
    // Bracketed paste: the terminal wraps a paste so it arrives as one
    // Event::Paste instead of per-line Enter keys (which would submit each line).
    let _ = execute!(io::stdout(), EnableBracketedPaste);
    // Focus reporting: lets us tell when you've switched away, so a finished turn
    // can ping you only when you're not looking. Harmless if unsupported.
    let _ = execute!(io::stdout(), EnableFocusChange);
    // Deliberately NOT capturing the mouse: capture would route wheel events to
    // us (enabling in-app scroll) but the terminal would then stop doing native
    // click-drag selection, breaking copy/paste. Native selection wins; scroll
    // with PageUp/PageDown.
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
    // Seed the context-fill gauge with the model's window so it reads 0% before
    // the first turn (it refreshes from real usage after each round).
    app.context_window = registry.metadata(model).1;
    // The catalog, for inline `/model <id>` completion (the picker fetches the
    // live list on demand; this is the synchronous type-ahead source).
    app.model_ids = registry.model_ids();
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
        app.apply_model(agent, registry, &model_id);
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
        let ctx = registry
            .metadata(model)
            .1
            .map(|w| format!(" · {w} token window"))
            .unwrap_or_default();
        // When resuming, show what we're walking back into before the hint.
        if let Some(summary) = &resume_summary {
            app.push(Line::styled(summary.clone(), dim()));
        }
        app.push(Line::styled(
            format!(
                "Enter to send · Alt-Enter for a newline · Ctrl-C interrupts/double exits · Ctrl-T shows reasoning · Ctrl-D toggles diff · /help for all commands{ctx}.",
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
                        Style::default().fg(Color::Cyan),
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
                Style::default().fg(Color::Cyan),
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
        apply_metadata(&mut app, agent, registry, &result, &models_cache_key);
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
                                    continue 'input;
                                }
                            }
                        };
                        let Some(event) = event else { break 'session };
                        event
                    }
                };
                match event {
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
                    Event::Key(key) if key.kind == KeyEventKind::Press && app.picker.is_some() => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        match key.code {
                            KeyCode::Enter => app.pick_model(agent, registry),
                            KeyCode::Esc => app.picker = None,
                            KeyCode::Char('c') if ctrl => app.picker = None,
                            // Navigation/filter: a fresh borrow, no app-level action.
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
                                            Style::default().fg(Color::Yellow),
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
                                                    Style::default().fg(Color::Yellow),
                                                ));
                                            }
                                        }
                                    }
                                } else {
                                    app.push(Line::styled(
                                        "name is required".to_string(),
                                        Style::default().fg(Color::Yellow),
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
                            KeyCode::Left
                                if app.provider_form.as_ref().unwrap().active() == 0 && !shift =>
                            {
                                // Left arrow on the provider picker row cycles.
                                // (Only when on the "provider" pseudo-field, which
                                // we represent as active==0 in edit mode or the
                                // name field in add mode — but we always show the
                                // provider picker at the top, so cycle on Left/Right
                                // when the form's provider row is focused.)
                                // For simplicity, Left/Right always cycle the provider.
                                app.provider_form.as_mut().unwrap().cycle_provider_prev();
                            }
                            KeyCode::Right if !shift => {
                                app.provider_form.as_mut().unwrap().cycle_provider();
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
                        let history_search_was_active = app.history_search.is_some();
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
                        let history_search_was_active = app.history_search.is_some();
                        // Ctrl-R opens reverse history search (when not already
                        // in it and there's history to search).
                        if ctrl
                            && key.code == KeyCode::Char('r')
                            && app.history_search.is_none()
                            && !app.input.history.is_empty()
                        {
                            let mut search = HistorySearch::default();
                            search.refilter(&app.input.history);
                            app.history_search = Some(search);
                        }
                        match key.code {
                            // Double Ctrl-C to exit: the first press (when idle
                            // with empty input) arms a transient notice; the
                            // second while the notice is active quits. With
                            // non-empty input, Ctrl-C clears the line as usual.
                            KeyCode::Char('c')
                                if ctrl && app.input.is_empty() && app.quit_notice.is_some() =>
                            {
                                break 'session;
                            }
                            KeyCode::Char('c') if ctrl && app.input.is_empty() => {
                                app.quit_notice =
                                    Some(Instant::now() + Duration::from_millis(1800));
                            }
                            KeyCode::Char('c') if ctrl => app.input.clear(),
                            KeyCode::Esc => {
                                app.quit_notice = None;
                                if app.show_help {
                                    app.show_help = false;
                                } else if app.show_diff {
                                    app.show_diff = false;
                                    app.diff_text = None;
                                } else {
                                    app.input.clear();
                                }
                            }
                            _ => {
                                // Any other key dismisses a pending quit notice.
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
        // A line is committed — drop any lingering completion menu state.
        app.completion = None;

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
                    let mut sink = ChannelUi { tx };
                    {
                        let fut = agent.compact_with(kind, &mut sink);
                        drive(
                            &mut terminal,
                            &mut input_rx,
                            &mut ticker,
                            &mut app,
                            rx,
                            fut,
                            false,
                        )
                        .await?;
                    }
                    app.set_working(false);
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
                                    Style::default().fg(Color::Yellow),
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
                                Style::default().fg(Color::Yellow),
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
                                    let (_price, window) = registry.metadata(&model);
                                    agent.set_provider(
                                        switched.switched.provider.into(),
                                        model.clone(),
                                        window,
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
                                    app.apply_model(agent, registry, &model);
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
                                        Style::default().fg(Color::Yellow),
                                    ));
                                }
                            }
                        }
                        Err(err) => {
                            app.push(Line::styled(
                                format!("/hf failed: {err:#}"),
                                Style::default().fg(Color::Yellow),
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
                    let caps = App::capabilities_map(registry, &ids);
                    app.picker = Some(ModelPicker::new(ids, &current, tags, &app.served, &caps));
                    continue;
                }
                // `/provider` with no arg: list configured profiles.
                // `/provider <name>`: use that profile, fetch the live model
                // metadata, and open the model selector.
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
                                    Style::default().fg(Color::Yellow),
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
                                Style::default().fg(Color::Yellow),
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
                                    Style::default().fg(Color::Yellow),
                                ));
                            }
                        }
                        continue;
                    }
                    // --- Use / list ---
                    if arg.is_empty() {
                        if app.profiles.is_empty() {
                            app.push(Line::styled(
                                "no profiles configured — use /provider add, or add [profiles.<name>] to hi.toml"
                                    .to_string(),
                                dim(),
                            ));
                        } else {
                            app.push(Line::styled("configured profiles:".to_string(), dim()));
                            let active = app.active_profile.clone();
                            let rows: Vec<(String, Style)> = app
                                .profiles
                                .iter()
                                .map(|p| {
                                    let is_active = active.as_deref() == Some(&p.name);
                                    let mark = if is_active { "▶" } else { " " };
                                    let model = p.model.as_deref().unwrap_or("(not configured)");
                                    let mut row =
                                        format!("  {mark} {} — {} · {}", p.name, p.provider, model);
                                    if let Some(url) = &p.base_url {
                                        row.push_str(&format!("  ·  {url}"));
                                    }
                                    let style = if is_active {
                                        Style::default().fg(Color::Cyan)
                                    } else {
                                        dim()
                                    };
                                    (row, style)
                                })
                                .collect();
                            for (row, style) in rows {
                                app.push(Line::styled(row, style));
                            }
                            app.push(Line::styled(
                                "/provider <name> to use a profile · /provider add · /provider edit [name] · /provider remove [name]"
                                    .to_string(),
                                dim(),
                            ));
                        }
                        continue;
                    }
                    // Resolve the profile and update the provider.
                    match (app.resolver)(&arg) {
                        Ok(switched) => {
                            let label = switched.label.clone();
                            let model = switched.model.clone();
                            let needs_model = model == "__model_not_configured__";
                            // Refresh metadata from the registry for this model.
                            let (_price, window) = registry.metadata(&model);
                            agent.set_provider(
                                switched.provider.into(),
                                model.clone(),
                                window,
                                switched.max_tokens,
                                switched.max_tokens_explicit,
                                None,
                            );
                            app.provider = label.clone();
                            app.model = model.clone();
                            app.active_profile = Some(arg.clone());
                            app.context_window = window;
                            app.served.clear();
                            app.push(Line::styled(
                                format!("using {label} (profile: {arg}) — model: {model}"),
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
                            let caps = App::capabilities_map(registry, &ids);
                            app.picker =
                                Some(ModelPicker::new(ids, &current, tags, &app.served, &caps));
                        }
                        Err(err) => {
                            app.push(Line::styled(
                                format!("/provider failed: {err:#}"),
                                Style::default().fg(Color::Yellow),
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
                                            Style::default().fg(Color::Green),
                                        ));
                                    }
                                    Ok(Err(err)) => {
                                        app.push(Line::styled(
                                            err,
                                            Style::default().fg(Color::Yellow),
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
                                    Ok(true) => (format!("✓ loop#{id} cancelled"), Color::Green),
                                    _ => (
                                        format!("no loop#{id} — /loop list shows ids"),
                                        Color::Yellow,
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
                                                .fg(Color::Cyan)
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
                                                marks.push_str(" · ⚒");
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
                                    Ok(true) => (format!("✓ loop#{id} {verb}"), Color::Green),
                                    _ => (
                                        format!("no loop#{id} — /loop list shows ids"),
                                        Color::Yellow,
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
                                        Color::Green,
                                    ),
                                    (Ok(true), None) => {
                                        (format!("✓ loop#{id} budget cleared"), Color::Green)
                                    }
                                    _ => (
                                        format!("no loop#{id} — /loop list shows ids"),
                                        Color::Yellow,
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
                                        Color::Green,
                                    ),
                                    (Ok(true), false) => {
                                        (format!("✓ loop#{id} trigger cleared"), Color::Green)
                                    }
                                    _ => (
                                        format!("no loop#{id} — /loop list shows ids"),
                                        Color::Yellow,
                                    ),
                                };
                                app.push(Line::styled(msg.0, Style::default().fg(msg.1)));
                            }
                        }
                        command::LoopArg::Fix { id, on } => {
                            if let Some(loops) = &app.loops {
                                let (tx, rx) = tokio::sync::oneshot::channel();
                                let _ = loops.ctl.send(crate::loops::LoopCtl::Fix {
                                    id,
                                    on,
                                    reply: tx,
                                });
                                let no_verify = on && fleet_launcher.verify.is_none();
                                let msg = match rx.await {
                                    Ok(true) if on => (
                                        format!(
                                            "✓ loop#{id} auto-fix on — a loud change dispatches a verified fix"
                                        ),
                                        Color::Green,
                                    ),
                                    Ok(true) => (format!("✓ loop#{id} auto-fix off"), Color::Green),
                                    _ => (
                                        format!("no loop#{id} — /loop list shows ids"),
                                        Color::Yellow,
                                    ),
                                };
                                app.push(Line::styled(msg.0, Style::default().fg(msg.1)));
                                if no_verify {
                                    app.push(Line::styled(
                                        "  note: no verify command set — fixes won't auto-merge until you /verify <cmd>"
                                            .to_string(),
                                        dim(),
                                    ));
                                }
                            }
                        }
                        command::LoopArg::Invalid(msg) => {
                            app.push(Line::styled(msg, Style::default().fg(Color::Yellow)));
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
                                        .fg(Color::Magenta)
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
                                    .fg(Color::Cyan)
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
                                        Style::default().fg(Color::Cyan)
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
                    app.handle_command(agent, other, registry).await;
                    continue;
                }
            }
        } else {
            line
        };

        // --- Turn phase: run the agent behind a channel, staying responsive. ---
        app.push(Line::styled(
            format!("› {run_line}"),
            Style::default().fg(Color::Blue),
        ));
        app.set_working(true);
        app.follow();
        let checkpoint = agent.messages().len();
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
        let mut sink = ChannelUi { tx };
        let background_before = hi_tools::background_process_ids();
        let cancelled = {
            let fut = agent.run_turn(&run_line, &mut sink);
            drive(
                &mut terminal,
                &mut input_rx,
                &mut ticker,
                &mut app,
                rx,
                fut,
                true,
            )
            .await?
        };

        if cancelled {
            if let Err(err) = agent.rewind_to_snapshot_durable(checkpoint, &turn_snapshot) {
                app.push(Line::styled(
                    format!("couldn't persist interrupted turn discard: {err:#}"),
                    Style::default().fg(Color::Yellow),
                ));
                agent.truncate_messages(checkpoint);
                agent.restore_state_snapshot(&turn_snapshot);
            }
            let killed = hi_tools::kill_background_processes_started_after(&background_before);
            app.last_turn_state = TurnState::Cancelled;
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
            app.push(Line::styled(msg, Style::default().fg(Color::Yellow)));
            // Interrupting a drive turn is an explicit "stop": pause the goal so
            // the drive doesn't restart on the next message. Progress is held;
            // `/goal resume` continues.
            if goal_drive_turn && agent.set_goal_paused(true) {
                app.push(Line::styled(
                    "goal drive interrupted — paused; /goal resume to continue".to_string(),
                    Style::default().fg(Color::Yellow),
                ));
            }
        } else {
            // Turn finished on its own — ping if you've likely stepped away.
            app.maybe_notify_done();
            // Capture which files this turn changed, so the "changed: …" line
            // above the input reflects the latest turn. The agent already
            // computed this for verify gating; reuse it rather than re-walking.
            app.last_changed_files = agent.last_changed_files().to_vec();
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
                            Style::default().fg(Color::Yellow),
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
        let mut sink = ChannelUi { tx };
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
                fut,
                false,
            )
            .await;
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
    // Remove any remaining fleet worktrees (sessions stay on disk, resumable).
    crate::dashboard::cleanup_fleet(&mut app);

    Ok(())
}

/// Drive a model future (a turn or a compaction) to completion while keeping
/// the UI live: redraw + spin every tick, drain the agent's events, let the
/// user scroll/queue/cancel. Returns whether the user cancelled with Ctrl-C.
async fn drive(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    input: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    mut rx: mpsc::UnboundedReceiver<UiEvent>,
    fut: impl std::future::Future<Output = Result<()>>,
    expect_turn_end: bool,
) -> Result<bool> {
    tokio::pin!(fut);
    let mut cancelled = false;
    let mut saw_turn_end = false;
    let mut last_activity = Instant::now();
    let mut watchdog_stuck = false;
    let watchdog_timeout = watchdog_stuck_timeout();
    loop {
        terminal.draw(|f| app.render(f))?;
        tokio::select! {
            result = &mut fut => {
                while let Ok(event) = rx.try_recv() {
                    if matches!(event, UiEvent::TurnEnd(_) | UiEvent::TurnError(..)) {
                        saw_turn_end = true;
                    }
                    app.apply(event);
                }
                if let Err(err) = result {
                    let (kind, guidance) = hi_agent::classify_error(&err);
                    if !matches!(app.last_turn_state, TurnState::Failed(_)) {
                        app.note_turn_failed(&format!("{err:#}"), kind, guidance);
                    }
                    if hi_agent::ui::error_counts_as_model_issue(&err) {
                        app.record_model_issue();
                    }
                } else if expect_turn_end && !cancelled && !saw_turn_end {
                    app.note_turn_completed_without_summary();
                }
                break;
            }
            Some(event) = rx.recv() => {
                if matches!(event, UiEvent::TurnEnd(_) | UiEvent::TurnError(..)) {
                    saw_turn_end = true;
                }
                last_activity = Instant::now();
                app.apply(event);
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
                    Some(Event::Paste(text)) => app.input.insert_str(&text),
                    Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
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
                            // Typing while a turn runs queues the next command — except `/copy`,
                            // which reads the selection synchronously.
                            _ => if let Some(submitted) = app.edit_key(&key) {
                                match command::parse(&submitted) {
                                    Some(Command::Copy(arg)) => app.copy(&arg),
                                    _ => app.queue.push_back(submitted),
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
    Ok(cancelled)
}
