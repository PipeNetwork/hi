//! The TUI event loop: `run` (entry point that sets up the terminal, spawns
//! the agent turn behind a channel, and drives the render loop) and `drive`
//! (the per-event state machine that routes crossterm events to `App`).

use std::io;
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
    App, TICK, ProfileInfo, ProfileLoader, ProfileRemover,
    ProfileResolver, ProfileSaver, TurnState, apply_metadata,
    splash_lines, watchdog_stuck_timeout,
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
    resume_summary: Option<String>,
    mcp_url: Option<String>,
    api_key: String,
) -> Result<()> {
    enable_raw_mode().context("entering raw mode")?;
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
    let _restore = Restore;
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
        mcp_url,
        api_key,
    );
    // Seed the context-fill gauge with the model's window so it reads 0% before
    // the first turn (it refreshes from real usage after each round).
    app.context_window = registry.metadata(model).1;
    // The catalog, for inline `/model <id>` completion (the picker fetches the
    // live list on demand; this is the synchronous type-ahead source).
    app.model_ids = registry.model_ids();
    // Load the on-disk /models cache so model metadata (window/price/health)
    // applies instantly at startup, without blocking on the network. The live
    // fetch still runs in the background and refreshes this; the cache just
    // covers the cold-start gap so the UI never looks stalled.
    let models_cache_key = hi_ai::cache_key(provider, base_url);
    if let Some(cached) = hi_ai::load_cache(&models_cache_key).await {
        app.served = cached.into_iter().map(|m| (m.id.clone(), m)).collect();
        let model_id = app.model.clone();
        if let Some(health) = app.apply_model(agent, registry, &model_id) {
            app.warn_degraded(&model_id, &health);
        }
    }
    if let Some(path) = &history_path
        && let Ok(text) = std::fs::read_to_string(path)
    {
        app.input.history = text
            .lines()
            .map(str::to_string)
            .filter(|l| !l.trim().is_empty())
            // Slash commands are never cached on submit; drop any that an older
            // version persisted, for the same Up-arrow stall reason.
            .filter(|l| !l.trim_start().starts_with('/'))
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
                "Enter to send · Alt-Enter for a newline · Ctrl-C interrupts · Ctrl-T shows reasoning · Ctrl-D toggles diff · /help for all commands{ctx}.",
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
                    first_event = maybe;
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
                            input_rx.recv().await
                        };
                        let Some(event) = event else {
                            break 'session; // input channel closed (stdin gone)
                        };
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
                                app.sync_completion();
                            }
                        }
                    }
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
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
                            KeyCode::Char('d') if ctrl && app.input.is_empty() => break 'session,
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
                            KeyCode::Esc if app.input.is_empty() => break 'session,
                            KeyCode::Esc => app.input.clear(),
                            _ => {
                                // Any other key dismisses a pending quit notice.
                                app.quit_notice = None;
                                if let Some(line) = app.edit_key(&key) {
                                    break 'input line;
                                }
                                app.sync_completion();
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
        let run_line = if let Some(cmd) = command::parse(&line) {
            match cmd {
                Command::Quit => break,
                Command::Prompt(prompt) => prompt,
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
                Command::Retry => match app.last_prompt.clone() {
                    Some(prompt) => {
                        agent.truncate_messages(app.last_turn_start);
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
                    None => {
                        app.push(Line::styled("nothing to retry yet".to_string(), dim()));
                        continue;
                    }
                },
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
                // Open the picker on the provider's *live* model list (what this
                // endpoint actually serves), falling back to the static catalog.
                // The fetch runs behind a spinner so the UI stays responsive and
                // Esc/Ctrl-C can cancel a slow or hung endpoint.
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
                                    if let Some(Event::Key(key)) = maybe
                                        && key.kind == KeyEventKind::Press
                                    {
                                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                                        if matches!(key.code, KeyCode::Esc)
                                            || (ctrl && matches!(key.code, KeyCode::Char('c')))
                                        {
                                            cancelled = true;
                                            break;
                                        }
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
                            // Remember the live metadata (window/price/health) so
                            // selecting a model can apply it and tag its health.
                            app.served = served.into_iter().map(|m| (m.id.clone(), m)).collect();
                            let mut ids: Vec<String> = app.served.keys().cloned().collect();
                            ids.sort();
                            ids
                        }
                        _ => {
                            let note = match &fetched {
                                Some(Ok(_)) => "provider listed no models — check the endpoint",
                                Some(Err(err)) => &format!(
                                    "couldn't fetch models ({err:#}) — check the endpoint and API key"
                                ),
                                None => "couldn't fetch models — check the endpoint",
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
                // `/provider <name>`: switch to that profile, fetch the new
                // endpoint's served models, and open the model picker so the
                // user can pick a model immediately.
                Command::Provider(arg) => {
                    let arg = arg.trim().to_string();
                    // --- Subcommands ---
                    if arg == "add" {
                        app.provider_form = Some(provider_form::ProviderForm::new_add());
                        continue;
                    }
                    if let Some(edit_name) = arg.strip_prefix("edit") {
                        let edit_name = edit_name.trim();
                        // If no name given, pick the first profile (or show a hint).
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
                        // If no name given, pick the first profile (or show a hint).
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
                                format!("can't remove '{target}' — it's the active profile; switch first"),
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
                    // --- Switch / list ---
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
                                    let model = p.model.as_deref().unwrap_or("(pick via /model)");
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
                                "/provider <name> to switch · /provider add · /provider edit [name] · /provider remove [name]"
                                    .to_string(),
                                dim(),
                            ));
                        }
                        continue;
                    }
                    // Resolve the profile and swap the provider.
                    match (app.resolver)(&arg) {
                        Ok(switched) => {
                            let label = switched.label.clone();
                            let model = switched.model.clone();
                            let needs_pick = model == "__pick_via_model__";
                            // Refresh metadata from the registry for this model.
                            let (price, window) = registry.metadata(&model);
                            agent.set_provider(switched.provider, model.clone(), price, window);
                            app.provider = label.clone();
                            app.model = model.clone();
                            app.active_profile = Some(arg.clone());
                            app.context_window = window;
                            app.served.clear();
                            app.push(Line::styled(
                                format!("switched to {label} (profile: {arg}) — model: {model}"),
                                dim(),
                            ));
                            if needs_pick {
                                app.push(Line::styled(
                                    "no model configured — pick from what this endpoint serves"
                                        .to_string(),
                                    dim(),
                                ));
                            }
                            // Fetch served models and open the picker, just like
                            // `/model` with no arg — so the user can immediately
                            // pick a model on the new endpoint.
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
                                            if let Some(Event::Key(key)) = maybe
                                                && key.kind == KeyEventKind::Press
                                            {
                                                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                                                if matches!(key.code, KeyCode::Esc)
                                                    || (ctrl && matches!(key.code, KeyCode::Char('c')))
                                                {
                                                    cancelled = true;
                                                    break;
                                                }
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
                                    app.push(Line::styled(
                                        format!("{count} models available — pick one"),
                                        dim(),
                                    ));
                                    ids
                                }
                                _ => {
                                    let note = match &fetched {
                                        Some(Ok(_)) => {
                                            "provider listed no models — check the endpoint"
                                        }
                                        Some(Err(err)) => &format!(
                                            "couldn't fetch models ({err:#}) — check the endpoint and API key"
                                        ),
                                        None => "couldn't fetch models — check the endpoint",
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
        // Reset the per-turn tool-call counter for the observability panel.
        app.turn_tool_calls = 0;
        app.turn_rounds = 0;
        // Grab the interrupt handle so Esc during a tool call can signal it.
        app.interrupt = Some(agent.interrupt_handle());
        let (tx, rx) = mpsc::unbounded_channel();
        let mut sink = ChannelUi { tx };
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
            agent.truncate_messages(checkpoint);
            app.last_turn_state = TurnState::Cancelled;
            let dropped = app.queue.len();
            app.queue.clear();
            let msg = if dropped > 0 {
                format!("^C interrupted; turn discarded ({dropped} queued command(s) dropped)")
            } else {
                "^C interrupted; turn discarded".to_string()
            };
            app.push(Line::styled(msg, Style::default().fg(Color::Yellow)));
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
            // Sync cumulative cost for the title bar's persistent display.
            app.cost_usd = agent.cost_usd();
            // A new turn's edits supersede any open diff panel's snapshot.
            app.diff_text = None;
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
                let idle = last_activity.elapsed();
                app.waiting_for = Some(idle);
                // Only notify about a quiet backend while no tool is legitimately
                // running. This is a soft notice, not a model health signal.
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
                            // Typing while a turn runs queues the next command —
                            // except `/tokens`, which reads the live counter and
                            // runs in sync so you can watch usage climb mid-turn.
                            _ => if let Some(submitted) = app.edit_key(&key) {
                                match command::parse(&submitted) {
                                    Some(Command::Tokens) => app.report_tokens(),
                                    Some(Command::Copy(arg)) => app.copy(&arg),
                                    _ => app.queue.push_back(submitted),
                                }
                            }
                        }
                    }
                    Some(Event::FocusGained) => app.set_focus(true),
                    Some(Event::FocusLost) => app.set_focus(false),
                    _ => {}
                }
            }
        }
    }
    app.waiting_for = None;
    Ok(cancelled)
}

