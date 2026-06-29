//! Full-screen terminal UI for `hi`.
//!
//! A ratatui application on the alternate screen: a bordered, scrollable
//! conversation transcript with a title/status bar, and an input box with a
//! "working" spinner. The agent runs behind an mpsc channel ([`ChannelUi`]) so
//! the event loop can keep redrawing — spinner, streaming output, scrolling —
//! while a turn is in flight, and can cancel it with Ctrl-C.

mod completion;
mod event;
mod input;
mod model_picker;
mod provider_form;
mod render;
mod util;

use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ansi_to_tui::IntoText;
use anyhow::{Context, Result};
use crossterm::event::{
    EnableBracketedPaste, EnableFocusChange, Event, EventStream, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, enable_raw_mode};
use futures_util::StreamExt;
use hi_agent::ui::tool_label;
use hi_agent::{Agent, Command, CompactionKind, PlanStatus, command};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};
use tokio::sync::mpsc;

/// Info about a configured profile, for the `/provider` list and picker.
#[derive(Clone, Debug)]
pub struct ProfileInfo {
    pub name: String,
    /// Display label for the provider (e.g. "anthropic", "ollama").
    pub provider: String,
    /// The model configured on this profile, if any.
    pub model: Option<String>,
    /// The base URL configured on this profile, if any (non-default only).
    pub base_url: Option<String>,
}

/// The result of resolving a profile name at runtime: a built provider, the
/// model id to use, and the provider's display label. The caller swaps these
/// into the agent via [`Agent::set_provider`].
pub struct SwitchedProvider {
    pub provider: Box<dyn hi_ai::Provider>,
    pub model: String,
    pub label: String,
}

/// A callback that resolves a named profile into a built provider + model +
/// label, for `/provider` mid-session. `hi-cli` supplies this; the TUI calls
/// it without needing to know about `Config`/`Settings` (which live in
/// `hi-cli`).
pub type ProfileResolver = Box<dyn Fn(&str) -> Result<SwitchedProvider> + Send + Sync>;

/// Form data for creating or editing a profile, exchanged between the TUI
/// (which collects it via a form) and `hi-cli` (which writes it to the config
/// file). Mirrors `hi_cli::config::ProfileForm` but without the dependency.
#[derive(Clone, Debug)]
pub struct ProfileFormData {
    pub name: String,
    /// "ollama", "pipenetwork", "anthropic", or "openai".
    pub provider: String,
    pub api_key: String,
    /// If true, `api_key` is an env var name (stored as `api_key_env`).
    pub store_as_env: bool,
    pub model: String,
    pub base_url: String,
}

/// A callback that saves a profile (add or edit) to the config file and
/// returns the updated profile list. `hi-cli` supplies this; the TUI calls it
/// when the user submits the provider form.
pub type ProfileSaver = Box<dyn Fn(&ProfileFormData) -> Result<Vec<ProfileInfo>> + Send + Sync>;

/// A callback that loads an existing profile's form data for editing.
pub type ProfileLoader = Box<dyn Fn(&str) -> Result<ProfileFormData> + Send + Sync>;

/// A callback that removes a profile from the config file and returns the
/// updated profile list. `hi-cli` supplies this; the TUI calls it for
/// `/provider remove <name>`.
pub type ProfileRemover = Box<dyn Fn(&str) -> Result<Vec<ProfileInfo>> + Send + Sync>;

use completion::{
    CompletionContext, CompletionItem, CompletionState, MODEL_CMD, MODEL_COMPLETION_MAX,
    PROVIDER_CMD, completion_context, completion_items_for,
};
use event::{ChannelUi, Restore, UiEvent};
use input::{HistorySearch, InputLine};
use model_picker::{
    ModelPicker, display_capabilities, display_health, display_price, display_window,
};
use render::{diff_lines, dim, line_text, looks_like_diff, markdown_line, wrapped_height};
use util::{clip_reason, copy_to_clipboard, fmt_count, fmt_elapsed, goal_feedback, notify_done};

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// How many model rows the `/model` picker shows at once.
pub(crate) const PICKER_ROWS: usize = 12;

/// A synchronous, plain (uncolored) `git diff` of the working tree, for the
/// `Ctrl-D` diff panel. The TUI applies its own highlighting via `diff_lines`,
/// so we want the raw diff without ANSI codes. Returns empty when not a git
/// repo or there are no changes. Synchronous because the key handler isn't
/// async and `git diff` is fast/user-initiated.
fn working_tree_diff_sync() -> String {
    let out = std::process::Command::new("git")
        .args(["--no-pager", "diff", "--no-color", "HEAD"])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).into_owned(),
        // Not a git repo / no HEAD: fall back to an untracked+unstaged diff.
        Ok(_) => {
            let untracked = std::process::Command::new("git")
                .args(["--no-pager", "diff", "--no-color"])
                .output();
            untracked
                .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                .unwrap_or_default()
        }
        Err(_) => String::new(),
    }
}
const TICK: Duration = Duration::from_millis(120);
/// Only show an informational notice after a long, genuinely silent wait. This
/// is deliberately not a model-health signal: hosted APIs may buffer and retry
/// on the backend without streaming visible tokens to the TUI.
const DEFAULT_WATCHDOG_STUCK_SECS: u64 = 180;
const MIN_WATCHDOG_STUCK_SECS: u64 = 30;
const MAX_WATCHDOG_STUCK_SECS: u64 = 1_800;
/// On terminals that don't report focus, notify after a turn at least this long
/// (a proxy for "you probably stepped away").
const NOTIFY_THRESHOLD: Duration = Duration::from_secs(30);

fn watchdog_stuck_timeout() -> Duration {
    let configured = std::env::var("HI_TUI_WATCHDOG_SECS").ok();
    watchdog_stuck_timeout_from_value(configured.as_deref())
}

fn watchdog_stuck_timeout_from_value(value: Option<&str>) -> Duration {
    let seconds = value
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .unwrap_or(DEFAULT_WATCHDOG_STUCK_SECS)
        .clamp(MIN_WATCHDOG_STUCK_SECS, MAX_WATCHDOG_STUCK_SECS);
    Duration::from_secs(seconds)
}

/// The "PipeNetwork.AI" wordmark rendered as figlet-style 5-row block letters.
/// All orange, ~2x the height of a normal line — the splash centerpiece.
/// Generated from `figlet -f small`, then hand-trimmed of trailing whitespace.
const BANNER: [&str; 5] = [
    " ___ _           _  _     _                  _       _   ___ ",
    "| _ (_)_ __  ___| \\| |___| |___ __ _____ _ _| |__   /_\\ |_ _|",
    "|  _/ | '_ \\/ -_) .` / -_)  _\\ V  V / _ \\ '_| / /_ / _ \\ | | ",
    "|_| |_| .__/\\___|_|\\_\\___|\\__|\\_/\\_/\\___/_| |_\\_(_)_/ \\_\\___|",
    "      |_|                                                    ",
];

/// Build the PipeNetwork.AI landing banner as styled lines, pushed as the
/// first transcript entries on startup. The full "PipeNetwork.AI" wordmark is
/// rendered ~2x size as a 5-row block-letter banner (all orange), followed by
/// the dim model line and the working directory. A blank line of breathing
/// room follows before the usage hint. Sits at the top of the transcript and
/// scrolls up naturally as you work.
fn splash_lines(provider: &str, model: &str, context_window: Option<u32>) -> Vec<Line<'static>> {
    let orange = Style::default().fg(Color::Rgb(255, 140, 0));
    let bold_orange = orange.add_modifier(Modifier::BOLD);
    let dim = dim();

    let ctx = context_window
        .map(|w| format!(" ({}K context)", w / 1000))
        .unwrap_or_default();
    let cwd = std::env::current_dir()
        .map(|d| d.display().to_string())
        .unwrap_or_else(|_| "?".into());

    // The 5-row block-letter banner, all orange + bold.
    let mut lines: Vec<Line<'static>> = BANNER
        .iter()
        .map(|row| Line::from(vec![Span::styled((*row).to_string(), bold_orange)]))
        .collect();

    // Dim model line + working directory below the banner.
    lines.push(Line::from(vec![Span::styled(
        format!("{model}{ctx} · {provider}"),
        dim,
    )]));
    lines.push(Line::from(vec![Span::styled(cwd, dim)]));

    // A blank line for breathing room before the usage hint.
    lines.push(Line::raw(""));
    lines
}

/// Apply a freshly fetched `/models` result: update the served-metadata map,
/// re-apply the current model (so its window/price/health refresh), warn on
/// degraded health, and persist the result to the on-disk cache for next
/// startup. A failure or empty list sets a startup notice instead of panicking.
fn apply_metadata(
    app: &mut App,
    agent: &mut Agent,
    registry: &hi_ai::Registry,
    result: &Result<Vec<hi_ai::ServedModel>>,
    cache_key: &str,
) {
    match result {
        Ok(served) if !served.is_empty() => {
            app.served = served.iter().cloned().map(|m| (m.id.clone(), m)).collect();
            let model_id = app.model.clone();
            if let Some(health) = app.apply_model(agent, registry, &model_id) {
                app.warn_degraded(&model_id, &health);
            }
            // Persist for next startup (best-effort, fire-and-forget).
            let models = served.clone();
            let key = cache_key.to_string();
            tokio::spawn(async move {
                hi_ai::save_cache(&key, &models).await;
            });
        }
        Ok(_) => {
            app.startup_notice = Some("provider metadata unavailable; using catalog".into());
        }
        Err(err) => {
            app.startup_notice = Some(format!("provider metadata check failed: {err:#}"));
        }
    }
}

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
                    app.handle_command(agent, other, registry);
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

/// One entry in the display transcript. Most content is a plain styled line;
/// reasoning (CoT) is stored specially so it can be collapsed by default and
/// expanded on demand via Ctrl-T, rather than flooding the transcript inline.
#[derive(Clone)]
enum TranscriptEntry {
    Line(Line<'static>),
    /// Assistant reasoning/thinking, buffered until the reasoning phase ends.
    /// Shown collapsed ("thought for Ns") unless `show_reasoning` is on.
    Reasoning {
        text: String,
        elapsed: Duration,
    },
}

impl TranscriptEntry {
    /// Flatten this entry into display lines under the current `show_reasoning`
    /// setting. A collapsed reasoning block is one dim summary line; expanded,
    /// it's the full text indented and dimmed.
    fn flatten(&self, show_reasoning: bool) -> Vec<Line<'static>> {
        match self {
            TranscriptEntry::Line(line) => vec![line.clone()],
            TranscriptEntry::Reasoning { text, elapsed } => {
                let secs = elapsed.as_secs();
                let label = if secs >= 60 {
                    format!("{}m {:02}s", secs / 60, secs % 60)
                } else {
                    format!("{secs}s")
                };
                if show_reasoning {
                    let mut lines = vec![Line::styled(
                        format!("⏺ thought for {label} (Ctrl-T to collapse)"),
                        Style::default().fg(Color::DarkGray),
                    )];
                    for line in text.lines() {
                        lines.push(Line::styled(
                            format!("  {line}"),
                            Style::default().fg(Color::DarkGray),
                        ));
                    }
                    lines
                } else {
                    vec![Line::styled(
                        format!("⏺ thought for {label}  (Ctrl-T to expand)",),
                        Style::default().fg(Color::DarkGray),
                    )]
                }
            }
        }
    }

    /// The plain text of this entry, for /copy and /export (always includes
    /// reasoning text regardless of collapse state).
    fn text(&self) -> String {
        match self {
            TranscriptEntry::Line(line) => line_text(line),
            TranscriptEntry::Reasoning { text, .. } => text.clone(),
        }
    }
}

struct App {
    provider: String,
    model: String,
    /// A shared interrupt handle for the running turn. When the user presses
    /// Esc during a tool call, this is set so the agent skips the current tool
    /// and feeds "interrupted by user" back to the model.
    interrupt: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// The name of the currently-active profile, if any (for marking it in the
    /// `/provider` list). Updated when the user switches via `/provider <name>`.
    active_profile: Option<String>,
    /// Configured profiles (for `/provider` with no arg).
    profiles: Vec<ProfileInfo>,
    /// Resolves a profile name to a built provider at runtime (for `/provider`).
    resolver: ProfileResolver,
    /// Saves a profile form to the config file (for `/provider add/edit`).
    saver: ProfileSaver,
    /// Loads an existing profile's form data (for `/provider edit`).
    loader: ProfileLoader,
    /// Removes a profile from the config file (for `/provider remove`).
    remover: ProfileRemover,
    transcript: Vec<TranscriptEntry>,
    /// The in-progress streamed line: (style, markdown?, text). Committed on
    /// newline/end. `markdown` is set for assistant prose so it's rendered with
    /// light markdown styling; reasoning and other streams stay literal.
    pending: Option<(Style, bool, String)>,
    /// Buffer for assistant reasoning (CoT) chunks: accumulated until the
    /// reasoning phase ends, then committed as a single collapsible
    /// `TranscriptEntry::Reasoning` so it doesn't flood the transcript inline.
    reasoning_buffer: String,
    /// When the current reasoning phase started (for the "thought for Ns" label).
    reasoning_started: Option<Instant>,
    /// Whether reasoning (CoT) blocks are expanded inline. Off by default —
    /// reasoning is collapsed to a one-line "thought for Ns" summary; Ctrl-T
    /// toggles this to show/hide the full thinking text.
    show_reasoning: bool,
    /// The language of the ``` fence the streamed assistant text is currently
    /// inside (empty string if the fence gave none); `None` when not in a fence.
    /// Carries across streamed lines so code interiors highlight consistently.
    code_lang: Option<String>,
    input: InputLine,
    /// Transcript scroll state. `following` pins the view to the latest output
    /// (the default); scrolling up unpins it and `scroll` holds the absolute
    /// offset (wrapped lines hidden above the viewport). It re-pins once scrolled
    /// back to the bottom, so streaming output never yanks a reader downward.
    following: bool,
    scroll: u16,
    /// Cached each render so scroll events (which fire outside render and don't
    /// know the wrapped height) can clamp and detect the bottom.
    view_max_scroll: u16,
    view_total: u16,
    /// Wrapped-line total at the moment the view last left the bottom — drives
    /// the "↓ N new" indicator while scrolled up.
    total_when_unpinned: u16,
    working: bool,
    spinner: usize,
    /// When the current turn started, for the elapsed-time readout.
    started: Option<Instant>,
    /// The tool currently executing (its display label) and when it started, so
    /// the working line can name the in-flight action with its own timer. `None`
    /// while the model — not a tool — is the active party.
    current_tool: Option<String>,
    current_tool_started: Option<Instant>,
    /// Lines typed while a turn was running, to run once it finishes (FIFO).
    queue: VecDeque<String>,
    /// The last message actually sent to the model, for `/retry`.
    last_prompt: Option<String>,
    /// Message-history length just before the last turn started, so `/retry`
    /// can drop that turn before re-running.
    last_turn_start: usize,
    /// Active model picker (`/model` with no argument), if any.
    picker: Option<ModelPicker>,
    /// Active provider form (`/provider add` or `/provider edit`), if any.
    provider_form: Option<provider_form::ProviderForm>,
    /// Ctrl-R reverse-search over input history. When active, keystrokes go to
    /// the search filter instead of the input line.
    history_search: Option<HistorySearch>,
    /// When set, a model-list fetch is in flight (start time, for the spinner).
    fetching: Option<Instant>,
    status: String,
    /// The latest task plan from the `update_plan` tool, pinned above the input
    /// as a live checklist. Empty until the model posts a plan; replaced wholesale
    /// on each update so it never drifts.
    plan: Vec<hi_agent::PlanStep>,
    /// Cumulative session token usage (input, output), mirrored from the agent
    /// so the working line and `/tokens` can show it live while a turn runs.
    usage: (u64, u64),
    /// Current context occupancy (tokens of the last request) and the model's
    /// window, for the live context-fill gauge.
    context_used: u64,
    context_window: Option<u32>,
    /// Live per-model metadata (window/price/health) learned from the endpoint's
    /// `/models`, keyed by id — used to apply a model's settings and flag health.
    served: HashMap<String, hi_ai::ServedModel>,
    /// The model catalog (ids), for inline `/model <id>` type-ahead completion.
    model_ids: Vec<String>,
    /// Cumulative session USD cost, mirrored from the agent for the title bar.
    cost_usd: Option<f64>,
    /// How many transcript lines have been trimmed from the top by
    /// [`cap_transcript`]. When > 0, a "↑ N lines compacted" marker shows at
    /// the top of the transcript so it's obvious older content scrolled off.
    trimmed: u64,
    /// Assistant prose currently streaming. Tool output is intentionally not
    /// included; `/copy` copies the assistant's answer, not command logs.
    current_assistant: String,
    /// Last completed assistant prose, copied by `/copy`.
    last_assistant: String,
    /// Last event type applied during the active turn, for better fallback
    /// diagnostics when the provider stops without a final turn-end event.
    last_turn_event: Option<TurnEventKind>,
    /// Whether the current/last turn invoked file-editing tools.
    last_turn_had_file_edits: bool,
    /// Files the last turn changed (from `agent.last_changed_files()`), shown
    /// as a compact "changed: …" line above the input so the user always sees
    /// what a turn touched without scrolling the transcript.
    last_changed_files: Vec<String>,
    /// Whether the `Ctrl-D` diff panel is open (a full working-tree diff pinned
    /// above the input, rendered with the same highlighting as tool-output diffs).
    show_diff: bool,
    /// Cached working-tree diff text for the open diff panel, refreshed when the
    /// panel is toggled on so it reflects the current tree, not a stale snapshot.
    diff_text: Option<String>,
    /// Whether the `Ctrl-?` agent-observability panel is open: telemetry
    /// counters, per-turn tool-call count, and context composition.
    show_debug: bool,
    /// Whether the keybindings help overlay is open (toggled by `?`).
    show_help: bool,
    /// Telemetry from the last turn (verify rounds, recovery retries, nudges,
    /// stalls), captured post-turn from `agent.last_turn_telemetry()` for the
    /// observability panel.
    last_telemetry: Option<hi_agent::TurnTelemetry>,
    /// Tool calls seen this turn (incremented on each `UiEvent::ToolCall`),
    /// for the observability panel's "tool calls this turn" line.
    turn_tool_calls: u32,
    /// Model rounds seen this turn (incremented on each `UiEvent::AssistantEnd`),
    /// so the activity line can show "round 3 · 5 tool calls" for multi-step turns.
    turn_rounds: u32,
    /// A tail of recent streamed tool output lines (e.g. bash stdout), shown
    /// live in the working area while a tool runs. Cleared when the tool
    /// finishes and its final result is pushed to the transcript.
    tool_stream_tail: Vec<String>,
    waiting_for: Option<Duration>,
    last_turn_state: TurnState,
    last_error: Option<String>,
    event_log: Vec<String>,
    model_issues: HashMap<String, u32>,
    startup_notice: Option<String>,
    /// A transient "Press Ctrl-C again to exit" notice, shown after the first
    /// Ctrl-C when idle. Cleared after ~1.8s (see the deadline race in the idle
    /// input loop) or when any other key is pressed. A second Ctrl-C while this
    /// is active quits the session.
    quit_notice: Option<Instant>,
    /// Active `/`-command completion menu: the query it's synced to and the
    /// highlighted row. `None` when the input isn't a slash-command prefix.
    completion: Option<CompletionState>,
    /// Whether the terminal currently has focus (best-effort, via focus-change
    /// reporting). Stays `true` on terminals that don't report it.
    focused: bool,
    /// Set once we've seen any focus event — i.e. the terminal reports focus, so
    /// `focused` is trustworthy.
    focus_known: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TurnEventKind {
    Assistant,
    Reasoning,
    AssistantEnd,
    ToolCall,
    ToolResult,
    Status,
    Usage,
    TurnEnd,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum TurnState {
    Idle,
    Running,
    Done(String),
    Warning(String),
    Failed(String),
    Cancelled,
}

impl App {
    #[allow(clippy::too_many_arguments)]
    fn new(
        provider: &str,
        model: &str,
        profiles: Vec<ProfileInfo>,
        active_profile: Option<String>,
        resolver: ProfileResolver,
        saver: ProfileSaver,
        loader: ProfileLoader,
        remover: ProfileRemover,
    ) -> Self {
        Self {
            provider: provider.to_string(),
            model: model.to_string(),
            interrupt: None,
            active_profile,
            profiles,
            resolver,
            saver,
            loader,
            remover,
            transcript: Vec::new(),
            pending: None,
            reasoning_buffer: String::new(),
            reasoning_started: None,
            show_reasoning: false,
            code_lang: None,
            input: InputLine::default(),
            following: true,
            scroll: 0,
            view_max_scroll: 0,
            view_total: 0,
            total_when_unpinned: 0,
            working: false,
            spinner: 0,
            started: None,
            current_tool: None,
            current_tool_started: None,
            queue: VecDeque::new(),
            last_prompt: None,
            last_turn_start: 0,
            picker: None,
            provider_form: None,
            history_search: None,
            fetching: None,
            status: String::new(),
            plan: Vec::new(),
            usage: (0, 0),
            context_used: 0,
            context_window: None,
            served: HashMap::new(),
            model_ids: Vec::new(),
            cost_usd: None,
            trimmed: 0,
            current_assistant: String::new(),
            last_assistant: String::new(),
            last_turn_event: None,
            last_turn_had_file_edits: false,
            last_changed_files: Vec::new(),
            show_diff: false,
            diff_text: None,
            show_debug: false,
            show_help: false,
            last_telemetry: None,
            turn_tool_calls: 0,
            turn_rounds: 0,
            tool_stream_tail: Vec::new(),
            waiting_for: None,
            last_turn_state: TurnState::Idle,
            last_error: None,
            event_log: Vec::new(),
            model_issues: HashMap::new(),
            startup_notice: None,
            quit_notice: None,
            completion: None,
            focused: true,
            focus_known: false,
        }
    }

    /// Record a focus-change report from the terminal (and that it reports them).
    fn set_focus(&mut self, focused: bool) {
        self.focused = focused;
        self.focus_known = true;
    }

    /// Ping the terminal when a turn finishes and you're likely away: when the
    /// terminal reports it's unfocused, or — on terminals that don't report
    /// focus — when the turn ran long enough that you probably stepped away.
    fn maybe_notify_done(&self) {
        let elapsed = self.started.map(|t| t.elapsed()).unwrap_or_default();
        let away = if self.focus_known {
            !self.focused
        } else {
            elapsed >= NOTIFY_THRESHOLD
        };
        if away {
            notify_done();
        }
    }

    /// Rows for a completion context — model ids from the live catalog, profile
    /// names from the config, every other command's values from the static
    /// table.
    fn items_for_ctx(&self, ctx: &CompletionContext) -> Vec<CompletionItem> {
        if let CompletionContext::Arg { cmd, prefix } = ctx
            && *cmd == MODEL_CMD
        {
            return self.model_completion_items(prefix);
        }
        if let CompletionContext::Arg { cmd, prefix } = ctx
            && *cmd == PROVIDER_CMD
        {
            return self.provider_completion_items(prefix);
        }
        completion_items_for(ctx)
    }

    /// Up to [`MODEL_COMPLETION_MAX`] catalog ids starting with `prefix` (already
    /// lowercased), as `/model <id>` rows — inline type-ahead for `/model`.
    fn model_completion_items(&self, prefix: &str) -> Vec<CompletionItem> {
        self.model_ids
            .iter()
            .filter(|id| id.to_lowercase().starts_with(prefix))
            .take(MODEL_COMPLETION_MAX)
            .map(|id| CompletionItem {
                label: id.clone(),
                help: String::new(),
                insert: format!("/{MODEL_CMD} {id}"),
                submit_on_enter: true,
            })
            .collect()
    }

    /// Profile names + `add`/`edit`/`remove` subcommands matching `prefix`, as
    /// `/provider <name>` rows — inline type-ahead for `/provider`.
    fn provider_completion_items(&self, prefix: &str) -> Vec<CompletionItem> {
        let mut items: Vec<CompletionItem> = Vec::new();
        // Subcommands first.
        for sub in ["add", "edit", "remove"] {
            if sub.starts_with(prefix) {
                items.push(CompletionItem {
                    label: sub.to_string(),
                    help: match sub {
                        "add" => "create a new profile",
                        "edit" => "edit an existing profile",
                        "remove" => "remove a profile",
                        _ => "",
                    }
                    .to_string(),
                    insert: format!("/{PROVIDER_CMD} {sub}"),
                    submit_on_enter: true,
                });
            }
        }
        // Profile names.
        for p in &self.profiles {
            if p.name.starts_with(prefix) {
                let help = format!(
                    "{} · {}",
                    p.provider,
                    p.model.as_deref().unwrap_or("pick via /model")
                );
                items.push(CompletionItem {
                    label: p.name.clone(),
                    help,
                    insert: format!("/{PROVIDER_CMD} {}", p.name),
                    submit_on_enter: true,
                });
            }
        }
        items
    }

    /// The rows the completion menu currently offers (empty when closed).
    fn completion_items(&self) -> Vec<CompletionItem> {
        match &self.completion {
            Some(c) => self.items_for_ctx(&c.ctx),
            None => Vec::new(),
        }
    }

    /// Re-sync the completion menu to the current input: open/refresh it when the
    /// input is a slash-command name being typed (`/`, `/mo`, …) or the argument
    /// of a command with enumerable values (`/compact `, `/model gp`), with
    /// matches; otherwise close it. Called after every edit to the input line.
    fn sync_completion(&mut self) {
        match completion_context(&self.input.text()) {
            Some(ctx) if !self.items_for_ctx(&ctx).is_empty() => {
                // Reset the highlight only when the context actually changed, so
                // navigation survives unrelated redraws.
                if self.completion.as_ref().map(|c| &c.ctx) != Some(&ctx) {
                    self.completion = Some(CompletionState { ctx, selected: 0 });
                }
            }
            _ => self.completion = None,
        }
    }

    /// Move the completion highlight by `delta`, clamped to the match list.
    fn completion_move(&mut self, delta: isize) {
        let len = self.completion_items().len();
        if let Some(c) = &mut self.completion
            && len > 0
        {
            let last = len - 1;
            c.selected = match delta {
                d if d < 0 => c.selected.saturating_sub(1),
                _ => (c.selected + 1).min(last),
            };
        }
    }

    /// Accept the highlighted completion: replace the input with the row's
    /// insertion (`/name`, `/name ` for an arg-taking command, or `/cmd value`)
    /// and close the menu. When `submit` is set and the row is a complete line,
    /// return it to run immediately; otherwise leave it in the input.
    fn accept_completion(&mut self, submit: bool) -> Option<String> {
        let items = self.completion_items();
        let c = self.completion.as_ref()?;
        let item = items.get(c.selected)?;
        let submit_on_enter = item.submit_on_enter;
        self.input.set(&item.insert);
        self.completion = None;
        if submit && submit_on_enter {
            Some(self.input.submit())
        } else {
            None
        }
    }

    /// Health tags (id → label) for the models we have live metadata on, for the
    /// `/model` picker. Healthy models are omitted.
    /// Build a per-model-id capabilities map from the registry, for the model
    /// picker's capability-tag column.
    fn capabilities_map(
        registry: &hi_ai::Registry,
        ids: &[String],
    ) -> HashMap<String, Vec<&'static str>> {
        ids.iter()
            .map(|id| (id.clone(), registry.capabilities(id)))
            .collect()
    }

    fn served_tags(&self) -> HashMap<String, String> {
        self.served
            .iter()
            .map(|(id, m)| {
                let tag = match (
                    m.health().map(str::to_string),
                    self.model_issues.get(id).copied().unwrap_or(0),
                ) {
                    (Some(endpoint), issues) if issues > 0 => {
                        format!("{endpoint}; degraded in-session")
                    }
                    (Some(endpoint), _) => endpoint,
                    (None, issues) if issues > 0 => "degraded in-session".to_string(),
                    (None, _) => String::new(),
                };
                (id.clone(), tag)
            })
            .filter_map(|(id, tag)| (!tag.is_empty()).then_some((id, tag)))
            .collect()
    }

    /// Apply `id` as the model: prefer live endpoint metadata (window/price) when
    /// we have it, else the catalog. Updates the agent and the gauge. Returns the
    /// model's health label if the endpoint flags it as not fully available.
    fn apply_model(
        &mut self,
        agent: &mut Agent,
        registry: &hi_ai::Registry,
        id: &str,
    ) -> Option<String> {
        let (cat_price, cat_window) = registry.metadata(id);
        let served = self.served.get(id);
        let price = served.and_then(|m| m.price).or(cat_price);
        let window = served.and_then(|m| m.context_window).or(cat_window);
        agent.set_model(id.to_string(), price, window);
        self.model = id.to_string();
        self.context_window = window;
        served.and_then(|m| m.health()).map(str::to_string)
    }

    /// Push a yellow line warning that `id` is in a non-healthy state.
    fn warn_degraded(&mut self, id: &str, health: &str) {
        self.push(Line::styled(
            format!(
                "⚠ {id} is reported {health} on this endpoint — responses may be slow or flaky; \
                 /model to pick another"
            ),
            Style::default().fg(Color::Yellow),
        ));
    }

    /// Percent of the context window currently occupied, when the window is known.
    fn context_pct(&self) -> Option<u64> {
        let window = u64::from(self.context_window?);
        (window > 0).then(|| (self.context_used * 100 / window).min(100))
    }

    /// Apply the picker's current selection as the model, then close it.
    fn pick_model(&mut self, agent: &mut Agent, registry: &hi_ai::Registry) {
        let id = self
            .picker
            .as_ref()
            .and_then(|p| p.current())
            .map(str::to_string);
        if let Some(id) = id {
            let health = self.apply_model(agent, registry, &id);
            self.push(Line::styled(format!("model set to {id}"), dim()));
            if let Some(h) = health {
                self.warn_degraded(&id, &h);
            }
        }
        self.picker = None;
        self.follow();
    }

    /// Mark the turn as running (or done), stamping the start time so the
    /// prompt bar can show elapsed seconds.
    fn set_working(&mut self, working: bool) {
        self.working = working;
        self.started = working.then(Instant::now);
        self.current_tool = None;
        self.current_tool_started = None;
        if working {
            self.last_turn_event = None;
            self.last_turn_had_file_edits = false;
            self.waiting_for = Some(Duration::ZERO);
            self.last_turn_state = TurnState::Running;
        } else if matches!(self.last_turn_state, TurnState::Running) {
            self.last_turn_state = TurnState::Idle;
            self.waiting_for = None;
        }
    }

    /// The live "what's happening now" lead for the working line: the in-flight
    /// tool named with its own elapsed timer, otherwise the model phase —
    /// `thinking…` (reasoning), `responding…` (streaming text), or `waiting for
    /// the model…` (the round's model call is in flight but nothing's streamed
    /// yet). Lets you tell a slow tool from a slow model at a glance.
    fn activity_line(&self) -> String {
        // A compact progress suffix for multi-step turns: "round 3 · 5 calls".
        // Suppressed on the first round with no tool calls (the common single-shot case).
        let progress = if self.turn_rounds > 1 || (self.turn_rounds > 0 && self.turn_tool_calls > 0)
        {
            format!(
                " · round {} · {} call{}",
                self.turn_rounds,
                self.turn_tool_calls,
                if self.turn_tool_calls == 1 { "" } else { "s" }
            )
        } else {
            String::new()
        };
        if let (Some(tool), Some(started)) = (&self.current_tool, self.current_tool_started) {
            return format!(
                "running {tool} · {}{progress}",
                fmt_elapsed(started.elapsed().as_secs())
            );
        }
        let secs = self.started.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        let verb = match self.last_turn_event {
            Some(TurnEventKind::Reasoning) => "thinking",
            Some(TurnEventKind::Assistant) => "responding",
            _ => "waiting for the model",
        };
        format!("{verb}… {}{progress}", fmt_elapsed(secs))
    }

    /// Apply a pure editing/navigation key to the input line, shared by the
    /// idle input phase and the in-turn queue-entry path. Returns the submitted
    /// text on Enter (when non-empty); the caller decides whether to run it now
    /// or queue it. Phase-specific control keys (Ctrl-C/Ctrl-D/Esc) are handled
    /// by the caller, not here.
    fn edit_key(&mut self, key: &KeyEvent) -> Option<String> {
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
        match key.code {
            // Alt+Enter inserts a newline (multi-line prompt without pasting); so
            // does a trailing backslash, for terminals that can't send Alt+Enter.
            KeyCode::Enter if alt => self.input.insert('\n'),
            KeyCode::Enter if self.input.continue_line() => {}
            KeyCode::Enter => {
                let line = self.input.submit();
                if !line.trim().is_empty() {
                    return Some(line);
                }
            }
            KeyCode::Char('u') if ctrl => self.input.kill_to_start(),
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
            // Toggle the working-tree diff panel. Refreshed when opened so it
            // reflects the current tree, not a stale snapshot. Fetched
            // synchronously (a `git diff` is fast and user-initiated) since the
            // key handler isn't async.
            KeyCode::Char('d') if ctrl => {
                self.show_diff = !self.show_diff;
                if self.show_diff {
                    self.diff_text = Some(working_tree_diff_sync());
                } else {
                    self.diff_text = None;
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

    fn push(&mut self, line: Line<'static>) {
        self.transcript.push(TranscriptEntry::Line(line));
        self.cap_transcript();
    }

    /// Bound the transcript so a very long session can't overflow the u16 scroll
    /// range, slow the per-frame render clone, or grow memory without limit. Older
    /// lines scroll off the top (the full session is still in the JSONL log). Only
    /// trims while pinned to the bottom, so a reader scrolled up isn't yanked by
    /// the offsets shifting underneath them. Sets `trimmed` so the render shows a
    /// "↑ N lines compacted" marker at the top of the transcript.
    fn cap_transcript(&mut self) {
        if self.following && self.transcript.len() > MAX_TRANSCRIPT_LINES {
            let excess = self.transcript.len() - MAX_TRANSCRIPT_LINES;
            self.transcript.drain(..excess);
            self.trimmed = self.trimmed.saturating_add(excess as u64);
        }
    }

    fn note_turn_completed_without_summary(&mut self) {
        match self.last_turn_event {
            Some(TurnEventKind::ToolCall | TurnEventKind::ToolResult) => {
                self.status = "stopped after tool output".to_string();
                self.last_turn_state = TurnState::Warning("stopped after tool output".to_string());
                self.last_error = Some("turn stopped after tool output".to_string());
                self.push(Line::styled(
                    "⚠ turn stopped after tool output without a final assistant response; try /retry",
                    Style::default().fg(Color::Yellow),
                ));
                self.record_model_issue();
            }
            _ => {
                self.status = "done · no usage reported".to_string();
                self.last_turn_state = TurnState::Done("no usage reported".to_string());
                self.push(Line::styled("✓ done · no usage reported", dim()));
            }
        }
        self.follow();
    }

    fn note_turn_failed(&mut self, error: &str, kind: &str, guidance: &str) {
        self.status = format!("failed · {kind}").to_string();
        self.last_turn_state = TurnState::Failed(error.to_string());
        self.last_error = Some(error.to_string());
        let guidance_line = if guidance.is_empty() {
            String::new()
        } else {
            format!("\n  💡 {guidance}")
        };
        self.push(Line::styled(
            format!("✗ failed · {kind}: {error}{guidance_line}"),
            Style::default().fg(Color::Red),
        ));
        self.follow();
    }

    fn note_backend_waiting(&mut self, idle: Duration, threshold: Duration) {
        let _ = (idle, threshold);
        self.push(Line::styled(
            "⚠ Still thinking. Ctrl-C cancels; keep waiting to continue.",
            Style::default().fg(Color::Yellow),
        ));
        self.follow();
    }

    fn record_model_issue(&mut self) {
        let count = {
            let entry = self.model_issues.entry(self.model.clone()).or_insert(0);
            *entry += 1;
            *entry
        };
        // Note: don't touch `last_error` here — it holds the actual failure
        // reason set by the caller; the per-model count lives in `model_issues`
        // and surfaces via `/status` model health.
        if count == 1 {
            self.push(Line::styled(
                format!(
                    "⚠ {} returned an incomplete turn; it is now marked degraded in-session. Consider /model",
                    self.model
                ),
                Style::default().fg(Color::Yellow),
            ));
        } else if count >= 2 {
            self.push(Line::styled(
                format!(
                    "⚠ {} has had {count} reliability issue(s) this session and is degraded; consider /model",
                    self.model
                ),
                Style::default().fg(Color::Yellow),
            ));
        }
    }

    /// Re-pin the view to the latest output. Called on explicit user actions (a
    /// new turn, a command's output) — not on streaming appends, so a reader who
    /// scrolled up stays put.
    fn follow(&mut self) {
        self.following = true;
    }

    /// Push the cumulative-usage line from the live counters. Works mid-turn —
    /// when the agent itself is borrowed by the running turn — because it reads
    /// the mirrored `usage` rather than the agent.
    fn report_tokens(&mut self) {
        let (input, output) = self.usage;
        let mut line = format!(
            "cumulative: {input} in · {output} out · {} total",
            input + output
        );
        if let Some(pct) = self.context_pct() {
            line.push_str(&format!("  ·  context {pct}% full"));
        }
        self.push(Line::styled(line, dim()));
        self.follow();
    }

    fn report_status(&mut self, agent: &Agent) {
        let (input, output) = self.usage;
        let state = match &self.last_turn_state {
            TurnState::Idle => "idle".to_string(),
            TurnState::Running => "running".to_string(),
            TurnState::Done(s) => format!("done ({s})"),
            TurnState::Warning(s) => format!("warning ({s})"),
            TurnState::Failed(s) => format!("failed ({s})"),
            TurnState::Cancelled => "cancelled".to_string(),
        };
        let ctx = self
            .context_pct()
            .map(|p| format!("{p}%"))
            .unwrap_or_else(|| "unknown".to_string());
        let goal = agent.goal().unwrap_or("off");
        let verify = agent.verify_summary();
        let tel = agent.last_turn_telemetry();
        let error = self.last_error.as_deref().unwrap_or("none");
        let model_issues = self.model_issues.get(&self.model).copied().unwrap_or(0);
        let model_health = if model_issues >= 2 {
            format!("degraded ({model_issues} issue(s))")
        } else if model_issues == 1 {
            "degraded (1 issue)".to_string()
        } else {
            "ok".to_string()
        };
        for line in [
            format!("status: {state}"),
            format!("provider/model: {} · {}", self.provider, self.model),
            format!(
                "context: {ctx}; usage: {input} in · {output} out · {} total",
                input + output
            ),
            format!(
                "cost: {}",
                agent
                    .cost_usd()
                    .map(|c| format!("${c:.4}"))
                    .unwrap_or_else(|| "unknown".into())
            ),
            format!("model health: {model_health}"),
            format!("goal: {goal}"),
            format!("verify: {verify}"),
            format!(
                "evidence: {} (reads {}, searches {}, listing_only {}, repair nudges {})",
                tel.discovery_depth,
                tel.file_reads,
                tel.targeted_searches,
                tel.listing_only,
                tel.quality_repair_nudges
            ),
            format!("last error: {error}"),
            format!(
                "startup notice: {}",
                self.startup_notice.as_deref().unwrap_or("none")
            ),
            format!(
                "queued: {}; checkpoints: {}",
                self.queue.len(),
                agent.checkpoint_count()
            ),
        ] {
            self.push(Line::styled(line, dim()));
        }
        self.follow();
    }

    fn write_debug_log(&mut self) {
        let path = std::path::Path::new(".hi-debug.log");
        let mut body = String::new();
        body.push_str("# hi debug log\n\n");
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
        body.push_str(&self.transcript_text());
        match std::fs::write(path, body) {
            Ok(()) => self.push(Line::styled("wrote debug log: .hi-debug.log", dim())),
            Err(err) => self.push(Line::styled(
                format!("log failed: {err}"),
                Style::default().fg(Color::Yellow),
            )),
        }
        self.follow();
    }

    fn copy(&mut self, arg: &str) {
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

    fn handle_goal(&mut self, agent: &mut Agent, arg: &str) {
        // Apply the change first, then describe the resulting state. When
        // long-horizon agency is on, setting a goal creates a structured `Goal`
        // (a single sub-goal equal to the objective, which the model decomposes
        // as it works via `update_plan`); clearing drops both views.
        match arg.trim() {
            "" => {} // no argument — just report the current goal
            "clear" | "off" | "none" => {
                agent.set_goal(None);
                agent.set_structured_goal(None);
            }
            goal => {
                if agent.long_horizon() {
                    let accepted = agent.set_structured_goal(Some(hi_agent::Goal::new(
                        goal.to_string(),
                        vec![goal.to_string()],
                    )));
                    if !accepted {
                        agent.set_goal(Some(goal.to_string()));
                    }
                } else {
                    agent.set_goal(Some(goal.to_string()));
                }
            }
        }
        // Report whichever view is active.
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
        // A set/clear is an applied change — show it plainly (green ✓), not dim,
        // so it's obvious it took effect. A bare `/goal` is just a read-out.
        let style = if prominent {
            Style::default().fg(Color::Green)
        } else {
            dim()
        };
        self.push(Line::styled(msg, style));
        self.follow();
    }

    fn transcript_text(&self) -> String {
        self.transcript
            .iter()
            .map(TranscriptEntry::text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn scroll_up(&mut self, n: u16) {
        self.scroll_by(-(n as i32));
    }

    fn scroll_down(&mut self, n: u16) {
        self.scroll_by(n as i32);
    }

    /// Move the viewport by `delta` wrapped lines (negative = toward older
    /// output). Re-pins to the bottom when scrolled all the way down; snapshots
    /// the line count when first leaving the bottom (for the "↓ N new" hint).
    /// Uses the metrics cached by the last render.
    fn scroll_by(&mut self, delta: i32) {
        let max = self.view_max_scroll as i32;
        let cur = if self.following {
            max
        } else {
            (self.scroll as i32).min(max)
        };
        let next = (cur + delta).clamp(0, max);
        if next >= max {
            self.following = true;
        } else {
            if self.following {
                self.total_when_unpinned = self.view_total;
            }
            self.following = false;
            self.scroll = next as u16;
        }
    }

    /// Commit the in-progress streamed line, if any.
    fn flush_pending(&mut self) {
        if let Some((style, markdown, text)) = self.pending.take() {
            let line = if markdown {
                markdown_line(&text, &mut self.code_lang)
            } else {
                Line::styled(text, style)
            };
            self.transcript.push(TranscriptEntry::Line(line));
            self.cap_transcript();
        }
    }

    /// Commit any buffered reasoning as a single collapsible entry, then clear
    /// the buffer. Called when the reasoning phase ends (first text arrives, or
    /// the message ends) so the reasoning isn't flooded inline.
    fn flush_reasoning(&mut self) {
        if self.reasoning_buffer.is_empty() {
            self.reasoning_started = None;
            return;
        }
        let elapsed = self
            .reasoning_started
            .map(|t| t.elapsed())
            .unwrap_or_default();
        let text = std::mem::take(&mut self.reasoning_buffer);
        self.transcript
            .push(TranscriptEntry::Reasoning { text, elapsed });
        self.reasoning_started = None;
        self.cap_transcript();
    }

    /// Append streamed text under `style`, committing complete lines. When
    /// `markdown` is set, committed lines are rendered with light markdown
    /// styling (headings, bullets, code fences, inline emphasis).
    fn stream(&mut self, style: Style, markdown: bool, chunk: &str) {
        // A style/kind change ends the current line.
        if let Some((prev, prev_md, _)) = &self.pending
            && (*prev != style || *prev_md != markdown)
        {
            self.flush_pending();
        }
        let (_, _, buf) = self
            .pending
            .get_or_insert_with(|| (style, markdown, String::new()));
        buf.push_str(chunk);
        while let Some(idx) = buf.find('\n') {
            let committed: String = buf[..idx].to_string();
            buf.drain(..=idx);
            let line = if markdown {
                markdown_line(&committed, &mut self.code_lang)
            } else {
                Line::styled(committed, style)
            };
            self.transcript.push(TranscriptEntry::Line(line));
        }
        self.cap_transcript();
        // No follow() here: streaming must not yank a reader who scrolled up.
        // While following, the view already tracks the growing bottom.
    }

    fn apply(&mut self, event: UiEvent) {
        match event {
            UiEvent::Text(t) => {
                self.event_log
                    .push(format!("assistant_text {} chars", t.len()));
                self.last_turn_event = Some(TurnEventKind::Assistant);
                // If reasoning preceded this text, commit it as a collapsible
                // block before the answer starts.
                self.flush_reasoning();
                self.current_assistant.push_str(&t);
                self.stream(Style::default(), true, &t);
            }
            UiEvent::Reasoning(t) => {
                self.event_log.push(format!("reasoning {} chars", t.len()));
                self.last_turn_event = Some(TurnEventKind::Reasoning);
                // Buffer reasoning instead of streaming it inline — it's
                // committed as a single collapsible "thought for Ns" entry when
                // the reasoning phase ends (first text or assistant_end).
                if self.reasoning_started.is_none() {
                    self.reasoning_started = Some(Instant::now());
                }
                self.reasoning_buffer.push_str(&t);
            }
            UiEvent::AssistantEnd => {
                self.event_log.push("assistant_end".to_string());
                self.last_turn_event = Some(TurnEventKind::AssistantEnd);
                self.turn_rounds = self.turn_rounds.saturating_add(1);
                self.flush_reasoning();
                self.flush_pending();
                if !self.current_assistant.trim().is_empty() {
                    self.last_assistant = self.current_assistant.trim().to_string();
                }
                self.current_assistant.clear();
                // Fences don't span messages; reset so a stray ``` can't bleed
                // code styling into the next response.
                self.code_lang = None;
            }
            UiEvent::ToolStarted(name, args) => {
                let label = tool_label(&name, &args);
                self.event_log.push(format!("tool_started {label}"));
                // Mark this tool as the active party so the working line can
                // name it with its own timer until the result lands. No
                // transcript line — the header is emitted with the result.
                self.current_tool = Some(label);
                self.current_tool_started = Some(Instant::now());
                // Clear any previous stream tail when a new tool starts.
                self.tool_stream_tail.clear();
            }
            UiEvent::ToolCall(name, args) => {
                let label = tool_label(&name, &args);
                self.event_log.push(format!("tool_call {label}"));
                self.last_turn_event = Some(TurnEventKind::ToolCall);
                self.turn_tool_calls = self.turn_tool_calls.saturating_add(1);
                if matches!(name.as_str(), "write" | "edit") {
                    self.last_turn_had_file_edits = true;
                }
                self.flush_reasoning();
                self.flush_pending();
                self.push(Line::styled(
                    format!("⏺ {label}"),
                    Style::default().fg(Color::Cyan),
                ));
            }
            UiEvent::ToolResult(name, result) => {
                self.event_log
                    .push(format!("tool_result {} chars", result.len()));
                self.last_turn_event = Some(TurnEventKind::ToolResult);
                // The tool finished — back to the model being the active party.
                self.current_tool = None;
                self.current_tool_started = None;
                self.tool_stream_tail.clear();
                self.flush_pending();
                self.push_result(&name, &result);
            }
            UiEvent::ToolStream(_name, line) => {
                // Accumulate streamed lines for the live working-area display.
                // Keep only the last few so the panel stays compact.
                self.tool_stream_tail.push(line.to_string());
                if self.tool_stream_tail.len() > 4 {
                    self.tool_stream_tail.remove(0);
                }
            }
            UiEvent::Status(s) => {
                self.event_log.push(format!("status {s}"));
                self.last_turn_event = Some(TurnEventKind::Status);
                self.flush_pending();
                self.push(Line::styled(s, Style::default().fg(Color::Blue)));
            }
            // Plan updates replace the pinned checklist in place — no transcript
            // line, so progress reads as one updating block rather than a scroll.
            UiEvent::Plan(steps) => {
                self.event_log.push(format!("plan {} steps", steps.len()));
                self.plan = steps;
            }
            // Live counters only — no transcript line; the working/title bars read them.
            UiEvent::Usage {
                input,
                output,
                ctx_used,
                ctx_window,
            } => {
                self.event_log
                    .push(format!("usage {input} in {output} out"));
                self.last_turn_event = Some(TurnEventKind::Usage);
                self.usage = (input, output);
                self.context_used = ctx_used;
                self.context_window = ctx_window;
            }
            UiEvent::TurnEnd(summary) => {
                self.event_log.push(format!("turn_end {summary}"));
                self.last_turn_event = Some(TurnEventKind::TurnEnd);
                self.flush_pending();
                // Tokens/cost go to the status bar; a dim marker in the transcript
                // makes the end of a turn unmistakable (so a long run doesn't just
                // trail off with no clear "done").
                self.status = summary.trim_matches(['[', ']']).to_string();
                self.last_turn_state = TurnState::Done(self.status.clone());
                self.push(Line::styled(format!("✓ done · {}", self.status), dim()));
                // No follow(): respect a reader who scrolled up — the "↓ N new"
                // hint tells them the summary landed below.
            }
            UiEvent::TurnError(kind, message, guidance) => {
                self.event_log.push(format!("turn_error {kind} {message}"));
                self.last_turn_event = Some(TurnEventKind::TurnEnd);
                self.flush_pending();
                self.note_turn_failed(&message, &kind, &guidance);
            }
            UiEvent::ChangedFiles(files) => {
                self.event_log
                    .push(format!("changed_files {}", files.len()));
                self.flush_pending();
                let label = if files.len() == 1 { "file" } else { "files" };
                let list = files.join(", ");
                let clipped = hi_agent::ui::clip(&list, 200);
                self.push(Line::styled(
                    format!("✎ {} {} changed: {}", files.len(), label, clipped),
                    Style::default().fg(Color::Green),
                ));
                self.follow();
            }
        }
    }

    /// Render a tool result, clipped to a handful of lines and indented.
    /// Preserves any ANSI colors (e.g. edit/write diffs); for *plain* unified
    /// diff output from a shell command (`git diff`, `diff -u`) — which CLIs
    /// emit without color when piped — adds diff coloring so it's readable.
    ///
    /// Read-only exploration tools (`read`/`list`/`grep`) already named the
    /// file or pattern in their `tool_call` header line — dumping their full
    /// output into the transcript is noise during a codebase review. Show a
    /// compact line count instead, so the transcript reads as a list of files
    /// consulted rather than a wall of their contents.
    fn push_result(&mut self, name: &str, result: &str) {
        if matches!(name, "read" | "list" | "grep") {
            let n = result.lines().count();
            if n == 0 {
                self.push(Line::styled("  (no output)", dim()));
            } else {
                self.push(Line::styled(
                    format!("  {n} line{}", if n == 1 { "" } else { "s" }),
                    dim(),
                ));
            }
            return;
        }
        // Enough to show a small edit's diff with its context inline; larger
        // results truncate with a footer (use `/diff` for the full diff).
        const MAX: usize = 16;
        if result.trim().is_empty() {
            self.push(Line::styled("  (no output)", dim()));
            return;
        }
        let body: String = result.lines().take(MAX).collect::<Vec<_>>().join("\n");
        let lines: Vec<Line<'static>> = if !body.contains('\u{1b}') && looks_like_diff(result) {
            diff_lines(&body)
        } else {
            // ANSI (already-colored) or non-diff text: parse escapes as before.
            body.into_text()
                .unwrap_or_else(|_| Text::from(body.clone()))
                .lines
        };
        for mut line in lines {
            line.spans.insert(0, "  ".into());
            self.transcript.push(TranscriptEntry::Line(line));
        }
        let extra = result.lines().count().saturating_sub(MAX);
        if extra > 0 {
            self.push(Line::styled(format!("  … {extra} more lines"), dim()));
        }
    }

    fn handle_command(&mut self, agent: &mut Agent, command: Command, registry: &hi_ai::Registry) {
        match command {
            Command::Quit => {}
            Command::Help => {
                for line in command::help_text().lines() {
                    self.push(Line::styled(line.to_string(), dim()));
                }
            }
            Command::Tokens => {
                let t = agent.totals();
                self.usage = (t.input_tokens, t.output_tokens);
                let cost = agent
                    .cost_usd()
                    .map(|c| format!(" · ${c:.4}"))
                    .unwrap_or_default();
                self.push(Line::styled(
                    format!(
                        "cumulative: {} in · {} out · {} total{}",
                        t.input_tokens,
                        t.output_tokens,
                        t.total(),
                        cost,
                    ),
                    dim(),
                ));
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
                    let caps = App::capabilities_map(registry, &ids);
                    if ids.is_empty() {
                        self.push(Line::styled(
                            "no models fetched yet — the endpoint didn't respond".to_string(),
                            dim(),
                        ));
                    } else {
                        self.picker =
                            Some(ModelPicker::new(ids, &current, tags, &self.served, &caps));
                    }
                } else {
                    let health = self.apply_model(agent, registry, &id);
                    self.push(Line::styled(format!("model set to {id}"), dim()));
                    if let Some(h) = health {
                        self.warn_degraded(&id, &h);
                    }
                }
            }
            Command::Clear => {
                let count = agent
                    .messages()
                    .iter()
                    .filter(|m| m.role != hi_ai::Role::System)
                    .count();
                agent.clear_history();
                self.transcript.clear();
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
            Command::Verify(arg) => {
                let msg = match arg.trim() {
                    "" if agent.verify_is_on() => format!("verify: {}", agent.verify_summary()),
                    "" => "verify: off (set one with /verify <cmd>)".to_string(),
                    "off" | "none" | "clear" | "disable" => {
                        agent.set_verify_command(None);
                        "verification disabled".to_string()
                    }
                    cmd => {
                        agent.set_verify_command(Some(cmd.to_string()));
                        format!(
                            "verification on: `{cmd}` — runs after each turn, iterates on failure"
                        )
                    }
                };
                self.push(Line::styled(msg, dim()));
            }
            Command::Diff => {
                let handle = tokio::runtime::Handle::current();
                let out = handle.block_on(hi_tools::working_tree_diff());
                let text = out.into_text().unwrap_or_else(|_| Text::from(out.clone()));
                for line in text.lines {
                    self.push(line);
                }
            }
            Command::Commit => {
                let handle = tokio::runtime::Handle::current();
                let out = handle.block_on(hi_tools::commit());
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
            // Handled in the event loop (async / runs a turn / needs config); never reach here.
            Command::Prompt(_)
            | Command::Compact(_)
            | Command::Retry
            | Command::Edit
            | Command::Undo
            | Command::Init
            | Command::Provider(_) => {}
            Command::Version => {
                self.push(Line::styled(format!("hi {}", hi_agent::VERSION), dim()));
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
            Command::Unknown(name) => {
                self.push(Line::styled(
                    format!("unknown command /{name}; try /help"),
                    dim(),
                ));
            }
        }
        self.follow();
    }

    /// The editable input rendered as one or more lines (the prompt may hold a
    /// pasted multi-line block), plus the cursor's (row, col) within them. Long
    /// inputs show only their last [`MAX_INPUT_ROWS`] lines with a "… more above"
    /// note so they can't swallow the screen.
    fn input_view(&self) -> (Vec<Line<'static>>, u16, u16) {
        const MAX_INPUT_ROWS: usize = 10;
        let text = self.input.text();
        let before: String = text.chars().take(self.input.cursor()).collect();
        let cursor_row_full = before.matches('\n').count();
        let cursor_col = before.chars().rev().take_while(|&c| c != '\n').count() as u16;

        let all: Vec<&str> = text.split('\n').collect();
        let truncated = all.len() > MAX_INPUT_ROWS;
        let start = if truncated {
            all.len() - MAX_INPUT_ROWS
        } else {
            0
        };

        let mut lines: Vec<Line<'static>> = Vec::new();
        if truncated {
            lines.push(Line::styled(
                format!("  ⋮ {start} more line(s) above"),
                dim(),
            ));
        }
        for (i, seg) in all[start..].iter().enumerate() {
            let prefix = if i == 0 && !truncated { "› " } else { "  " };
            lines.push(Line::from(format!("{prefix}{seg}")));
        }
        let cursor_row = u16::from(truncated) + cursor_row_full.saturating_sub(start) as u16;
        (lines, cursor_row, 2 + cursor_col)
    }

    /// The pinned plan checklist shown just above the input, or empty when no
    /// plan has been posted. Done steps dim out; the active step is bold cyan.
    /// `max_steps` caps how many step lines are rendered (on top of the header)
    /// so a long plan can't swallow the input area or overflow the screen.
    fn plan_lines(&self, max_steps: usize) -> Vec<Line<'static>> {
        if self.plan.is_empty() {
            return Vec::new();
        }
        const HARD_CAP: usize = 8;
        let max_steps = max_steps.min(HARD_CAP);
        let total = self.plan.len();
        let done = self
            .plan
            .iter()
            .filter(|s| s.status == PlanStatus::Done)
            .count();
        let mut out = vec![Line::styled(
            format!("plan · {done}/{total}"),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )];
        for s in self.plan.iter().take(max_steps) {
            let (glyph, glyph_style, title_style) = match s.status {
                PlanStatus::Done => ('✓', Style::default().fg(Color::Green), dim()),
                PlanStatus::Active => (
                    '▸',
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                PlanStatus::Pending => ('☐', dim(), Style::default()),
            };
            out.push(Line::from(vec![
                Span::styled(format!("  {glyph} "), glyph_style),
                Span::styled(s.title.clone(), title_style),
            ]));
        }
        if total > max_steps {
            out.push(Line::styled(
                format!("  … +{} more", total - max_steps),
                dim(),
            ));
        }
        out
    }

    fn render(&mut self, frame: &mut ratatui::Frame) {
        let area = frame.area();
        // The input box grows to fit a spinner status line (while working), the
        // (possibly multi-line) input, and up to three queued commands.
        let status_lines = 1usize;
        let queued_shown = self.queue.len().min(3);
        let queue_extra = usize::from(self.queue.len() > 3);
        let (input_lines, cursor_row, cursor_col) = self.input_view();
        let completion_rows = self.completion_items().len();
        // The optional Ctrl-D diff panel height (header + up to 20 diff lines +
        // optional "more" line) and the compact changed-files summary line.
        let diff_h = if self.show_diff && self.diff_text.is_some() {
            let n = self
                .diff_text
                .as_deref()
                .map(|t| t.trim().lines().count())
                .unwrap_or(0);
            1 + n.min(20) + usize::from(n > 20)
        } else {
            0
        };
        let changed_h = usize::from(!self.last_changed_files.is_empty() && !self.working);
        // The Ctrl-? observability panel: header + 3 diagnostic lines.
        let debug_h = if self.show_debug { 5 } else { 0 };
        // The `?` keybindings help overlay: header + 10 lines.
        let help_h = if self.show_help { 12 } else { 0 };
        // Live streamed tool output tail (e.g. bash stdout), shown while a tool runs.
        let stream_h = if self.working && !self.tool_stream_tail.is_empty() {
            self.tool_stream_tail.len()
        } else {
            0
        };
        // Height of the input box excluding the plan checklist and the 2 border
        // rows. Used to figure out how many plan steps fit on screen.
        let base_h = diff_h
            + changed_h
            + debug_h
            + help_h
            + stream_h
            + usize::from(self.startup_notice.is_some())
            + usize::from(self.quit_notice.is_some())
            + status_lines
            + completion_rows
            + input_lines.len()
            + queued_shown
            + queue_extra;
        // The live plan checklist, pinned just above the input (input-bar state
        // only). The step count is capped to what fits on screen so a long plan
        // can't make the box taller than the terminal — ratatui's Layout would
        // otherwise clamp the rect and the Paragraph content would spill past
        // the bottom border. Reserve one row for the transcript (Min(1) below).
        let cap = area.height.saturating_sub(1).max(1) as usize;
        let avail_inner = cap.saturating_sub(base_h + 2);
        // plan_h = 1 (header) + steps_shown + (1 if total > steps_shown else 0).
        // Pick the largest step count (up to total and HARD_CAP) whose plan_h
        // fits avail_inner.
        let max_steps = if self.plan.is_empty() {
            0
        } else {
            const HARD_CAP: usize = 8;
            let total = self.plan.len();
            let upper = total.min(HARD_CAP);
            // plan_h for a candidate `n` (n <= upper): 1 + n + (total > n) as usize.
            // Try showing all `upper` first; if it doesn't fit, shrink.
            let mut n = upper;
            while n > 0 && 1 + n + usize::from(total > n) > avail_inner {
                n -= 1;
            }
            // If even n=0 (header only, +maybe more) doesn't fit, show 1 step
            // so the plan is still visible rather than entirely hidden.
            if 1 + n + usize::from(total > n) > avail_inner {
                1
            } else {
                n
            }
        };
        let plan_block = self.plan_lines(max_steps);
        let plan_h = plan_block.len();
        let input_h = if self.fetching.is_some() {
            3
        } else if let Some(p) = &self.picker {
            // filter line + visible model rows + borders, bounded by the screen.
            let rows = p.matches.len().clamp(1, PICKER_ROWS) as u16;
            (rows + 3).min(area.height.saturating_sub(3))
        } else if let Some(form) = &self.provider_form {
            // Provider form: provider picker row + hint row + text fields +
            // borders. The API-key field is hidden for Ollama, so subtract one.
            let fields = if form.api_key_unneeded() { 3 } else { 4 };
            (fields + 4) as u16
        } else {
            (base_h + plan_h + 2).min(cap) as u16
        };
        let rows = Layout::vertical([Constraint::Min(1), Constraint::Length(input_h)]).split(area);

        // --- Transcript ---
        let title = format!(" hi · {} · {} ", self.provider, self.model);
        // Right-aligned: persistent cost + token totals, a context-fill gauge,
        // then the last status — so spend is always visible without a command.
        let mut info_parts: Vec<String> = Vec::new();
        let (input, output) = self.usage;
        if input + output > 0 {
            info_parts.push(format!("↑{} ↓{}", fmt_count(input), fmt_count(output)));
        }
        if let Some(cost) = self.cost_usd {
            info_parts.push(format!("${cost:.4}"));
        }
        if let Some(pct) = self.context_pct() {
            info_parts.push(format!("{pct}% ctx"));
        }
        if !self.status.is_empty() {
            info_parts.push(self.status.clone());
        }
        let info = if info_parts.is_empty() {
            String::new()
        } else {
            format!(" {} ", info_parts.join(" · "))
        };
        let mut lines: Vec<Line<'static>> = Vec::new();
        // If older transcript lines have been trimmed, show a marker at the top
        // so the user knows earlier content scrolled off (it's still in the
        // JSONL session log).
        if self.trimmed > 0 {
            lines.push(Line::styled(
                format!("↑ {} lines compacted (see session log)", self.trimmed),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ));
        }
        lines.extend(
            self.transcript
                .iter()
                .flat_map(|e| e.flatten(self.show_reasoning)),
        );
        if let Some((style, markdown, text)) = &self.pending {
            // Style the in-progress line live (headings, bold, code, …) so prose
            // doesn't snap into formatting only when its newline lands. The line
            // isn't committed yet, so apply markdown against a CLONE of the fence
            // state — the real `code_lang` must only advance on a committed line.
            let line = if *markdown {
                markdown_line(text, &mut self.code_lang.clone())
            } else {
                Line::styled(text.clone(), *style)
            };
            lines.push(line);
        }
        let inner_w = rows[0].width.saturating_sub(2);
        let inner_h = rows[0].height.saturating_sub(2);
        let total = wrapped_height(&lines, inner_w);
        let max_scroll = total.saturating_sub(inner_h);
        // Cache the geometry so scroll events (which fire outside render) can clamp
        // and detect the bottom.
        self.view_max_scroll = max_scroll;
        self.view_total = total;
        // Pinned to the bottom while following; otherwise hold the user's absolute
        // offset, re-pinning if the content shrank back to within one screen.
        let scroll = if self.following || self.scroll >= max_scroll {
            self.following = true;
            max_scroll
        } else {
            self.scroll
        };

        let mut block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(dim())
            .title(title)
            .title_top(Line::from(info).right_aligned());
        // While scrolled up, a bottom-right hint shows how much is below — new
        // lines that arrived since you left the bottom, else how far down it is.
        if !self.following {
            let new = total.saturating_sub(self.total_when_unpinned);
            let label = if new > 0 {
                format!(" ↓ {new} new ")
            } else {
                format!(" ↓ {} below ", max_scroll.saturating_sub(scroll))
            };
            block = block.title_bottom(
                Line::from(Span::styled(
                    label,
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .right_aligned(),
            );
        }
        let para = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block)
            .scroll((scroll, 0));
        frame.render_widget(para, rows[0]);

        // --- Bottom region: a fetch spinner, the model picker, or the input bar. ---
        if let Some(started) = self.fetching {
            let frame_ch = SPINNER[self.spinner % SPINNER.len()];
            let elapsed = fmt_elapsed(started.elapsed().as_secs());
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan));
            let body = Line::from(vec![
                Span::styled(
                    format!(
                        "{frame_ch} fetching models from {}… {elapsed}",
                        self.provider
                    ),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   Esc to cancel", dim()),
            ]);
            frame.render_widget(Paragraph::new(body).block(block), rows[1]);
        } else if let Some(p) = &self.picker {
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" select a model ")
                .title_top(
                    Line::from(format!(" {}/{} ", p.selected + 1, p.matches.len().max(1)))
                        .right_aligned(),
                );
            let mut plines: Vec<Line> = vec![Line::from(vec![
                Span::raw(format!("filter: {}", p.filter)),
                Span::styled(
                    "   ↑↓ move · type to filter · Enter select · Esc cancel",
                    dim(),
                ),
            ])];
            let (_, visible) = p.visible();
            if visible.is_empty() {
                plines.push(Line::styled("  (no matches)".to_string(), dim()));
            }
            for row in visible {
                let mut tag = String::new();
                if row.id == p.current {
                    tag.push_str(" (current)");
                }
                let health = display_health(row.meta);
                if !health.is_empty() {
                    tag.push_str(&format!(" [{health}]"));
                }
                let caps = display_capabilities(row.meta);
                if !caps.is_empty() {
                    tag.push_str(&format!(" {{{caps}}}"));
                }
                // Price + window columns, right-aligned after the id.
                let price = display_price(row.meta);
                let window = display_window(row.meta);
                let meta_col = if price.is_empty() && window.is_empty() {
                    String::new()
                } else {
                    format!("  {price:>8}  {window:>5}")
                };
                if row.selected {
                    plines.push(Line::from(vec![
                        Span::styled(
                            format!("▶ {}", row.id),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(meta_col, Style::default().fg(Color::Yellow)),
                        Span::styled(tag, dim()),
                    ]));
                } else {
                    plines.push(Line::from(vec![
                        Span::raw(format!("  {}", row.id)),
                        Span::styled(meta_col, Style::default().fg(Color::DarkGray)),
                        Span::styled(tag, dim()),
                    ]));
                }
            }
            frame.render_widget(Paragraph::new(plines).block(block), rows[1]);
            // Cursor on the filter line, just after "filter: <text>".
            let cx = rows[1].x + 1 + 8 + p.filter.chars().count() as u16;
            frame.set_cursor_position((cx.min(rows[1].right().saturating_sub(2)), rows[1].y + 1));
        } else if let Some(form) = &self.provider_form {
            let title = if form.editing {
                " edit provider "
            } else {
                " add provider "
            };
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan))
                .title(title);
            let choices = form.provider_choices();
            let pidx = form.provider_idx();
            let mut lines: Vec<Line> = Vec::new();

            // Provider picker row.
            let mut prov_spans = vec![Span::raw("Provider: ")];
            for (i, (_id, label)) in choices.iter().enumerate() {
                if i == pidx {
                    prov_spans.push(Span::styled(
                        format!("▶ {label} "),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ));
                } else {
                    prov_spans.push(Span::styled(format!("  {label} "), dim()));
                }
            }
            lines.push(Line::from(prov_spans));
            lines.push(Line::styled(
                "  ←→ cycle · Tab next field · Enter save · Esc cancel".to_string(),
                dim(),
            ));

            // Text fields.
            let unneeded = form.api_key_unneeded();
            for (i, (label, placeholder, value, is_active)) in
                form.field_labels().into_iter().enumerate()
            {
                // Skip rendering the API-key field entirely for Ollama — it
                // would be a confusing, unusable field the user might try to fill.
                if i == 1 && unneeded {
                    continue;
                }
                let display = if value.is_empty() && !placeholder.is_empty() {
                    placeholder.clone()
                } else {
                    value.clone()
                };
                let prefix = if is_active { "▶ " } else { "  " };
                let val_span = if value.is_empty() && !placeholder.is_empty() {
                    Span::styled(display, Style::default().fg(Color::DarkGray))
                } else if is_active {
                    Span::styled(display, Style::default().fg(Color::Cyan))
                } else {
                    Span::raw(display)
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{prefix}{label}: "),
                        Style::default().fg(Color::Yellow),
                    ),
                    val_span,
                ]));
            }

            frame.render_widget(Paragraph::new(lines).block(block), rows[1]);

            // Cursor on the active text field.
            let form_fields = form.field_labels();
            let active_idx = form.active();
            // Account for the hidden API-key field (index 1) when computing the
            // display row: fields after it shift up by one.
            let hidden_before = if form.api_key_unneeded() && active_idx > 1 {
                1
            } else {
                0
            };
            // +3 for border + provider row + hint row.
            let cy = rows[1].y + 1 + 2 + (active_idx - hidden_before) as u16;
            let label = form_fields[active_idx].0;
            let prefix_len = 2 + label.len() + 2; // "▶ " + label + ": "
            let cx = rows[1].x + 1 + prefix_len as u16 + form.active_cursor() as u16;
            frame.set_cursor_position((cx.min(rows[1].right().saturating_sub(2)), cy));
        } else {
            // The border turns cyan and the top inner line becomes a bold
            // spinner + elapsed seconds while a turn runs; the prompt stays
            // editable so you can type the next command (it queues below).
            let input_block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(if self.working {
                    Style::default().fg(Color::Cyan)
                } else {
                    dim()
                });

            let mut ilines: Vec<Line> = Vec::new();
            // Pinned plan checklist at the very top of the input box.
            ilines.extend(plan_block);
            // Ctrl-R reverse history search overlay: shows the query, the match
            // count, and a few recent matches above the input line.
            if let Some(search) = &self.history_search {
                let count = search.matches.len();
                let preview = search
                    .current()
                    .and_then(|i| self.input.history.get(i))
                    .map(|s| s.replace('\n', " "))
                    .unwrap_or_default();
                let preview = if preview.len() > 60 {
                    format!("{}…", &preview[..60])
                } else {
                    preview
                };
                ilines.push(Line::from(vec![
                    Span::styled("reverse-i-search: ", Style::default().fg(Color::Green)),
                    Span::styled(
                        search.query.clone(),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  ({count} match{})", if count == 1 { "" } else { "es" }),
                        dim(),
                    ),
                ]));
                ilines.push(Line::styled(format!("  → {preview}"), dim()));
            }
            // The `Ctrl-D` working-tree diff panel: a compact view of what's
            // changed in the tree, rendered with the same highlighting as
            // tool-output diffs. Sits above the changed-files summary line.
            if self.show_diff
                && let Some(text) = &self.diff_text
            {
                ilines.push(Line::styled(
                    "diff (Ctrl-D to close)".to_string(),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ));
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    ilines.push(Line::styled("(no changes in the working tree)", dim()));
                } else {
                    // `diff_lines` parses the whole body (tracking `@@` line
                    // numbers) into highlighted lines; cap the result so a huge
                    // diff can't swallow the input box. The full diff is one
                    // `git diff` away.
                    let rendered = diff_lines(trimmed);
                    let total = rendered.len();
                    for line in rendered.into_iter().take(20) {
                        ilines.push(line);
                    }
                    if total > 20 {
                        ilines.push(Line::styled(
                            format!("  … +{} more (see `git diff`)", total - 20),
                            dim(),
                        ));
                    }
                }
            }
            // A compact "changed: …" line so the user always sees what the last
            // turn touched, without opening the diff panel or scrolling.
            if !self.last_changed_files.is_empty() && !self.working {
                let summary = self
                    .last_changed_files
                    .iter()
                    .map(|p| p.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                ilines.push(Line::styled(
                    format!("changed: {summary}  (Ctrl-D for diff)"),
                    dim(),
                ));
            }
            // The Ctrl-? agent-observability panel: trajectory telemetry, tool
            // calls this turn, and context composition. Read-only diagnostics.
            if self.show_debug {
                ilines.push(Line::styled(
                    "agent (Ctrl-? to close)".to_string(),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ));
                let t = self.last_telemetry.as_ref();
                let tel = if let Some(t) = t {
                    format!(
                        "telemetry: {} verify · {} retry · {} repeat · {} continue · {} trunc{}",
                        t.verify_rounds,
                        t.recovery_retries,
                        t.repeat_nudges,
                        t.continue_nudges,
                        t.truncation_retries,
                        if t.stalled_unfinished || t.stalled_repeating {
                            " · stalled"
                        } else {
                            ""
                        }
                    )
                } else {
                    "telemetry: (no turn yet)".to_string()
                };
                ilines.push(Line::styled(tel, dim()));
                if let Some(t) = self.last_telemetry.as_ref() {
                    ilines.push(Line::styled(
                        format!(
                            "evidence: {} · reads {} · searches {} · listing_only {} · repair {}",
                            t.discovery_depth,
                            t.file_reads,
                            t.targeted_searches,
                            t.listing_only,
                            t.quality_repair_nudges
                        ),
                        dim(),
                    ));
                }
                // Scheduler parallelism: max concurrent batch and serial share.
                let sched = if let Some(t) = self.last_telemetry.as_ref() {
                    if t.tool_calls > 0 {
                        format!(
                            "scheduler: {} calls · max batch {} · {} serial",
                            t.tool_calls, t.max_concurrent_batch, t.serial_runs,
                        )
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                if !sched.is_empty() {
                    ilines.push(Line::styled(sched, dim()));
                }
                ilines.push(Line::styled(
                    format!("tool calls this turn: {}", self.turn_tool_calls),
                    dim(),
                ));
                // Context composition: occupancy vs. window, and cumulative
                // session tokens (the same numbers the usage line shows, but
                // gathered here for a single diagnostics view).
                let (input, output) = self.usage;
                let ctx = if let Some(pct) = self.context_pct() {
                    format!(" · ctx {pct}%")
                } else {
                    String::new()
                };
                ilines.push(Line::styled(
                    format!("session: ↑{} ↓{}{ctx}", fmt_count(input), fmt_count(output)),
                    dim(),
                ));
            }
            // The `?` keybindings help overlay: a compact, contextual cheat
            // sheet. Toggled by pressing `?` on an empty input line.
            if self.show_help {
                ilines.push(Line::styled(
                    "keybindings (? to close)".to_string(),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ));
                let bindings = [
                    ("Enter", "send the prompt"),
                    ("Alt-Enter / \\", "insert a newline (multi-line prompt)"),
                    (
                        "Ctrl-C",
                        "interrupt the running turn, or double-press to quit",
                    ),
                    ("Ctrl-D (idle)", "quit"),
                    ("Ctrl-D (typing)", "toggle the working-tree diff panel"),
                    ("Ctrl-T", "toggle reasoning (thinking) display"),
                    ("Ctrl-?", "toggle agent observability panel"),
                    ("Ctrl-R", "fuzzy-search input history"),
                    ("PageUp/PageDown", "scroll the transcript"),
                    ("Esc", "clear input, or quit when empty"),
                    ("/help", "show all slash commands"),
                ];
                for (key, desc) in bindings {
                    ilines.push(Line::from(vec![
                        Span::styled(format!("  {key:<18}"), dim()),
                        Span::raw(desc),
                    ]));
                }
            }
            if let Some(notice) = &self.startup_notice {
                ilines.push(Line::styled(
                    notice.clone(),
                    Style::default().fg(Color::Yellow),
                ));
            }
            if self.quit_notice.is_some() {
                ilines.push(Line::styled(
                    "Press Ctrl-C again to exit",
                    Style::default().fg(Color::Yellow),
                ));
            }
            if self.working {
                let frame_ch = SPINNER[self.spinner % SPINNER.len()];
                let (input, output) = self.usage;
                // Show the running token total once the first round reports it.
                let mut stats = String::new();
                if input + output > 0 {
                    stats.push_str(&format!(" · ↑{} ↓{}", fmt_count(input), fmt_count(output)));
                }
                if let Some(pct) = self.context_pct() {
                    stats.push_str(&format!(" · {pct}% ctx"));
                }
                // The activity lead (named tool + timer, or thinking/responding)
                // replaces the old coarse "working… · last: <event>"; its own timer
                // and the watchdog notices cover the "is it stalled?" signal.
                ilines.push(Line::from(vec![
                    Span::styled(
                        format!("{frame_ch} {}{stats}", self.activity_line()),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("   Ctrl-C to interrupt", dim()),
                ]));
                // Show a tail of recent streamed tool output (e.g. bash stdout)
                // so the user sees live progress during long-running commands.
                for line in &self.tool_stream_tail {
                    ilines.push(Line::styled(
                        format!("  │ {}", clip_reason(line)),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
            } else {
                let line = match &self.last_turn_state {
                    TurnState::Idle => "ready".to_string(),
                    TurnState::Running => "working".to_string(),
                    TurnState::Done(s) => format!("ready · last: done ({s})"),
                    TurnState::Warning(s) => format!("ready · last: warning ({s})"),
                    // Show the failure reason inline so you don't have to scroll
                    // the transcript to learn what went wrong.
                    TurnState::Failed(s) => {
                        format!(
                            "ready · last: failed — {} · /retry to rerun",
                            clip_reason(s)
                        )
                    }
                    TurnState::Cancelled => "ready · last: cancelled".to_string(),
                };
                ilines.push(Line::styled(line, dim()));
            }
            // The `/`-command completion menu sits just above the input line. Rows
            // are command names (`/compact`) or, past the name, argument values
            // (`hybrid`, `full`, `elide`).
            let items = self.completion_items();
            let selected = self.completion.as_ref().map(|c| c.selected).unwrap_or(0);
            let label_w = items.iter().map(|i| i.label.len()).max().unwrap_or(0);
            for (i, item) in items.iter().enumerate() {
                let label = format!("{:<width$}", item.label, width = label_w);
                if i == selected {
                    ilines.push(Line::from(vec![
                        Span::styled(
                            format!("▶ {label}"),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("  {}", item.help), dim()),
                    ]));
                } else {
                    ilines.push(Line::from(vec![
                        Span::raw(format!("  {label}")),
                        Span::styled(format!("  {}", item.help), dim()),
                    ]));
                }
            }
            ilines.extend(input_lines);
            for q in self.queue.iter().take(3) {
                ilines.push(Line::styled(format!("⏳ {q}"), dim()));
            }
            if self.queue.len() > 3 {
                ilines.push(Line::styled(
                    format!("   … +{} more queued", self.queue.len() - 3),
                    dim(),
                ));
            }
            frame.render_widget(Paragraph::new(ilines).block(input_block), rows[1]);

            // Cursor sits within the editable input — below the optional startup
            // notice, the status line, and the completion menu.
            let above = plan_h
                + diff_h
                + changed_h
                + debug_h
                + help_h
                + stream_h
                + usize::from(self.startup_notice.is_some())
                + usize::from(self.quit_notice.is_some())
                + status_lines
                + self.completion_items().len();
            let cx = rows[1].x + 1 + cursor_col;
            let cy = rows[1].y + 1 + above as u16 + cursor_row;
            frame.set_cursor_position((
                cx.min(rows[1].right().saturating_sub(2)),
                cy.min(rows[1].bottom().saturating_sub(2)),
            ));
        }
    }
}

/// Max transcript lines kept for display and scrolling. Older lines scroll off
/// the top (the full session is still in the JSONL log). Bounds the u16 scroll
/// range, the per-frame render clone, and memory on very long sessions.
const MAX_TRANSCRIPT_LINES: usize = 10_000;

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn dump(term: &Terminal<TestBackend>) -> String {
        let buf = term.backend().buffer();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    /// A no-op resolver for tests — `/provider` isn't exercised in unit tests.
    fn test_resolver() -> ProfileResolver {
        Box::new(|_name| anyhow::bail!("no profiles in tests"))
    }

    fn test_saver() -> ProfileSaver {
        Box::new(|_form| anyhow::bail!("no profiles in tests"))
    }

    fn test_loader() -> ProfileLoader {
        Box::new(|_name| anyhow::bail!("no profiles in tests"))
    }

    fn test_remover() -> ProfileRemover {
        Box::new(|_name| anyhow::bail!("no profiles in tests"))
    }

    /// `App::new` with empty profiles and dummy callbacks, for tests.
    fn test_app(provider: &str, model: &str) -> App {
        App::new(
            provider,
            model,
            Vec::new(),
            None,
            test_resolver(),
            test_saver(),
            test_loader(),
            test_remover(),
        )
    }

    #[test]
    fn sticky_scroll_unpins_on_scroll_up_and_repins_at_bottom() {
        let mut app = test_app("openai", "gpt-4o");
        // Simulate what render() caches for a transcript taller than the viewport.
        app.view_max_scroll = 100;
        app.view_total = 120;
        assert!(app.following, "starts pinned to the bottom");

        // Scrolling up unpins, holds an absolute offset, and snapshots the count.
        app.scroll_up(10);
        assert!(!app.following, "scroll up unpins");
        assert_eq!(app.scroll, 90, "offset = max_scroll - 10");
        assert_eq!(app.total_when_unpinned, 120);

        // Streaming output below must NOT yank a scrolled-up reader back down.
        app.apply(UiEvent::Text("a fresh streamed line\n".into()));
        assert!(
            !app.following,
            "new output leaves the scrolled-up reader put"
        );

        // Scrolling back past the bottom re-pins so output follows again.
        app.scroll_down(1000);
        assert!(app.following, "reaching the bottom re-pins");
    }

    #[test]
    fn transcript_is_capped_while_following_but_not_while_scrolled_up() {
        let mut app = test_app("openai", "gpt-4o");
        // Following (the default): pushing far past the cap keeps it bounded, and
        // keeps the newest lines (the oldest scroll off the top).
        for i in 0..(MAX_TRANSCRIPT_LINES + 5_000) {
            app.push(Line::raw(format!("l{i}")));
        }
        assert_eq!(
            app.transcript.len(),
            MAX_TRANSCRIPT_LINES,
            "bounded while following"
        );
        assert_eq!(
            app.transcript.last().unwrap().text(),
            format!("l{}", MAX_TRANSCRIPT_LINES + 5_000 - 1),
            "newest line kept"
        );

        // Scrolled up: pushes are NOT trimmed, or the offsets would shift under a
        // reader. (render caches the geometry scroll_up needs.)
        app.view_max_scroll = 50;
        app.view_total = 60;
        app.scroll_up(5);
        assert!(!app.following, "scrolled up");
        let before = app.transcript.len();
        for i in 0..1_000 {
            app.push(Line::raw(format!("m{i}")));
        }
        assert_eq!(
            app.transcript.len(),
            before + 1_000,
            "grows while scrolled up, no trim"
        );
    }

    #[test]
    fn scrolling_moves_the_viewport_through_render_and_repins() {
        let mut app = test_app("openai", "gpt-4o");
        for i in 0..100 {
            app.push(Line::raw(format!("line {i:03}")));
        }
        let mut term = Terminal::new(TestBackend::new(40, 12)).unwrap();
        // Following: the bottom is visible, the top is not.
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("line 099"),
            "bottom visible when following:\n{screen}"
        );
        assert!(
            !screen.contains("line 000"),
            "top hidden when following:\n{screen}"
        );

        // Scroll up: earlier lines appear, the bottom leaves the viewport.
        app.scroll_up(40);
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(!app.following, "scroll up unpins");
        assert!(
            !screen.contains("line 099"),
            "bottom gone after scroll up:\n{screen}"
        );
        assert!(
            screen.contains("line 0"),
            "older lines now visible:\n{screen}"
        );

        // Scroll back down past the end: re-pins and shows the bottom again.
        app.scroll_down(1000);
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(app.following, "re-pinned at the bottom");
        assert!(
            screen.contains("line 099"),
            "bottom visible again:\n{screen}"
        );
    }

    #[test]
    fn following_shows_last_line_when_word_wrapping_creates_extra_rows() {
        // Regression: `wrapped_height` counted characters (ceil(len/width)) but
        // ratatui's `WordWrapper` wraps at word boundaries — a word that doesn't
        // fit the remaining space moves to the next line, and a word wider than
        // the line is broken across rows. That makes the real wrapped height
        // LARGER than the char-based estimate, so `max_scroll` was too small
        // and the bottom of a long message was clipped off-screen.
        //
        // Each line below has a 45-char word at width 38: ratatui wraps it to
        // 3 rows, but the old char-based estimate said 2. With 20 such lines
        // the ~20-row undercount pushed the last line entirely off-screen.
        let mut app = test_app("openai", "gpt-4o");
        for i in 0..20 {
            app.push(Line::raw(format!(
                "word{i:02} supercalifragilisticexpialidocious_extras"
            )));
        }
        app.push(Line::raw("LAST_LINE_MARKER_42"));

        let mut term = Terminal::new(TestBackend::new(40, 12)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("LAST_LINE_MARKER_42"),
            "last line must be visible when following (word-wrap clip bug):\n{screen}"
        );
    }

    #[test]
    fn following_shows_last_line_with_realistic_prose_word_wrapping() {
        // A second regression check with normal prose (no artificially long
        // words). At a narrow width, word-boundary wrapping still produces more
        // rows than char-based `ceil(len/width)` because words that don't fit
        // the remaining space leave the current line short. This is the case
        // that clipped the end of a long assistant message in practice.
        let mut app = test_app("openai", "gpt-4o");
        // 30 lines of prose, each ~70 chars. At width 36 (inner of a 38-wide
        // terminal), char-based says ceil(70/36) = 2 rows per line, but
        // word-wrap often produces 3 because words straddle the boundary.
        for i in 0..30 {
            app.push(Line::raw(format!(
                "The quick brown fox jumps over the lazy dog and then runs {i:02}"
            )));
        }
        app.push(Line::raw("FINAL_ANSWER_99"));

        let mut term = Terminal::new(TestBackend::new(38, 14)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("FINAL_ANSWER_99"),
            "last line must be visible with prose word-wrapping:\n{screen}"
        );
    }

    #[test]
    fn working_line_names_the_inflight_tool_and_model_phase() {
        let mut app = test_app("openai", "gpt-4o");
        app.set_working(true);
        // Model phase: reasoning then text stream distinctly.
        app.apply(UiEvent::Reasoning("hmm".into()));
        assert!(
            app.activity_line().starts_with("thinking…"),
            "{}",
            app.activity_line()
        );
        app.apply(UiEvent::Text("here".into()));
        assert!(
            app.activity_line().starts_with("responding…"),
            "{}",
            app.activity_line()
        );
        // A tool starts → the line names it (with its own timer)…
        app.apply(UiEvent::ToolStarted(
            "bash".into(),
            "{\"command\":\"cargo test\"}".into(),
        ));
        assert!(
            app.activity_line().starts_with("running bash cargo test"),
            "{}",
            app.activity_line()
        );
        // …and clears back to the model once the result lands.
        app.apply(UiEvent::ToolResult("bash".into(), "ok".into()));
        assert!(
            app.activity_line().starts_with("waiting for the model"),
            "{}",
            app.activity_line()
        );
    }

    #[test]
    fn renders_tool_call_diff_and_spinner() {
        let mut app = test_app("openai", "gpt-4o");
        app.apply(UiEvent::ToolCall(
            "edit".into(),
            "{\"path\":\"src/cli.rs\",\"old_string\":\"a\",\"new_string\":\"b\"}".into(),
        ));
        // ANSI-colored diff line (from the edit tool) must render as text.
        app.apply(UiEvent::ToolResult(
            "edit".into(),
            "\u{1b}[32m+ pub json: bool\u{1b}[0m".into(),
        ));
        app.apply(UiEvent::TurnEnd("[1234 in · 56 out · 1290 total]".into()));
        app.working = true;
        app.spinner = 2;

        let mut term = Terminal::new(TestBackend::new(56, 13)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);

        // The header reads as "edit <path>", not a raw JSON dump.
        assert!(screen.contains("⏺ edit src/cli.rs"), "readable tool header");
        assert!(
            !screen.contains("old_string"),
            "header must not dump JSON args"
        );
        assert!(
            screen.contains("pub json: bool"),
            "ANSI diff rendered as text"
        );
        assert!(screen.contains("1290 total"), "status bar shows usage");
        assert!(
            screen.contains(SPINNER[2]) && screen.contains("0s"),
            "prompt bar shows the spinner + an elapsed timer while working: {screen}"
        );
        assert!(
            screen.contains("Ctrl-C to interrupt"),
            "prompt bar shows the interrupt hint while working"
        );
    }

    #[test]
    fn colorizes_plain_diff_tool_output() {
        let mut app = test_app("openai", "gpt-4o");
        let diff = "--- a/x.rs\n+++ b/x.rs\n@@ -1,2 +1,2 @@\n-old\n+new\n ctx\n";
        app.apply(UiEvent::ToolResult("bash".into(), diff.into()));
        // The content span (after the "  " indent) carries the diff color.
        let colored: Vec<(String, Option<Color>)> = app
            .transcript
            .iter()
            .flat_map(|e| e.flatten(false))
            .map(|l| {
                let text: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
                (text, l.spans.last().map(|s| s.style.fg).unwrap_or(None))
            })
            .collect();
        assert!(
            colored
                .iter()
                .any(|(t, fg)| t.contains("+new") && *fg == Some(Color::Green)),
            "added line is green: {colored:?}"
        );
        assert!(
            colored
                .iter()
                .any(|(t, fg)| t.contains("-old") && *fg == Some(Color::Red)),
            "removed line is red"
        );
        assert!(
            colored
                .iter()
                .any(|(t, fg)| t.contains("@@") && *fg == Some(Color::Cyan)),
            "hunk header is cyan"
        );
    }

    #[test]
    fn non_diff_tool_output_is_not_colorized() {
        let mut app = test_app("openai", "gpt-4o");
        app.apply(UiEvent::ToolResult(
            "bash".into(),
            "- item one\n- item two\n".into(),
        ));
        let any_red = app
            .transcript
            .iter()
            .flat_map(|e| e.flatten(false))
            .any(|l| l.spans.last().map(|s| s.style.fg) == Some(Some(Color::Red)));
        assert!(!any_red, "a plain list must not be colorized as a diff");
    }

    #[test]
    fn usage_event_updates_live_counter_and_working_line() {
        let mut app = test_app("openai", "gpt-4o");
        app.set_working(true);
        app.apply(UiEvent::Usage {
            input: 1234,
            output: 340,
            ctx_used: 64_000,
            ctx_window: Some(128_000),
        });
        assert_eq!(app.usage, (1234, 340));
        assert_eq!(app.context_pct(), Some(50));

        let mut term = Terminal::new(TestBackend::new(72, 8)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(screen.contains(SPINNER[0]), "spinner shown: {screen}");
        assert!(screen.contains("↑1.2k"), "live input tokens: {screen}");
        assert!(screen.contains("↓340"), "live output tokens: {screen}");
        assert!(screen.contains("50% ctx"), "live context fill: {screen}");
    }

    #[test]
    fn report_tokens_pushes_cumulative_line() {
        // `/tokens` mid-turn reads the mirrored counter (the agent is borrowed).
        let mut app = test_app("openai", "gpt-4o");
        app.apply(UiEvent::Usage {
            input: 1000,
            output: 250,
            ctx_used: 0,
            ctx_window: None,
        });
        app.report_tokens();
        let line = app.transcript.last().unwrap().text();
        assert_eq!(line, "cumulative: 1000 in · 250 out · 1250 total");
    }

    #[test]
    fn renders_queued_commands_while_working() {
        let mut app = test_app("openai", "gpt-4o");
        app.set_working(true);
        app.queue.push_back("run the tests".into());
        app.queue.push_back("then commit".into());
        app.input.set("typing a third");

        let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);

        assert!(screen.contains(SPINNER[0]), "spinner shown while working");
        assert!(
            screen.contains("run the tests"),
            "first queued command shown"
        );
        assert!(
            screen.contains("then commit"),
            "second queued command shown"
        );
        assert!(
            screen.contains("typing a third"),
            "input stays editable while working"
        );
    }

    #[test]
    fn renders_pinned_plan_checklist() {
        use hi_agent::PlanStep;
        let mut app = test_app("openai", "gpt-4o");
        app.apply(UiEvent::Plan(vec![
            PlanStep {
                title: "find leak".into(),
                status: PlanStatus::Done,
            },
            PlanStep {
                title: "fix walkers".into(),
                status: PlanStatus::Active,
            },
            PlanStep {
                title: "add tests".into(),
                status: PlanStatus::Pending,
            },
        ]));

        let mut term = Terminal::new(TestBackend::new(60, 20)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);

        assert!(
            screen.contains("plan · 1/3"),
            "plan header w/ progress:\n{screen}"
        );
        assert!(screen.contains("find leak"), "step titles shown:\n{screen}");
        assert!(screen.contains("fix walkers"));
        assert!(screen.contains("add tests"));
        assert!(screen.contains('✓'), "done glyph:\n{screen}");
        assert!(screen.contains('▸'), "active glyph:\n{screen}");

        // A later update replaces the plan in place — progress advances and the
        // checklist isn't duplicated into the transcript.
        app.apply(UiEvent::Plan(vec![
            PlanStep {
                title: "find leak".into(),
                status: PlanStatus::Done,
            },
            PlanStep {
                title: "fix walkers".into(),
                status: PlanStatus::Done,
            },
            PlanStep {
                title: "add tests".into(),
                status: PlanStatus::Active,
            },
        ]));
        term.draw(|f| app.render(f)).unwrap();
        let screen2 = dump(&term);
        assert!(
            screen2.contains("plan · 2/3"),
            "progress advanced:\n{screen2}"
        );
        assert!(
            app.transcript.is_empty(),
            "plan must not echo into the transcript"
        );
    }

    #[test]
    fn long_plan_does_not_break_input_box_border() {
        // Regression: when the plan + status + input is taller than the screen,
        // the input box height used to exceed the terminal height. ratatui's
        // Layout clamps the fixed-Length rect, so the Paragraph content spilled
        // past the bottom border — the `╰` landed mid-content and later steps
        // rendered outside the box. The box must stay closed and fit on screen.
        use hi_agent::PlanStep;
        let mut app = test_app("openai", "gpt-4o");
        app.apply(UiEvent::Plan(
            (0..8)
                .map(|i| PlanStep {
                    title: format!("step {i} with a fairly long title to be realistic"),
                    status: if i < 3 {
                        PlanStatus::Done
                    } else if i == 3 {
                        PlanStatus::Active
                    } else {
                        PlanStatus::Pending
                    },
                })
                .collect(),
        ));
        app.working = true;
        // Tiny height: the full plan (9 lines) + status + input + borders can't fit.
        let mut term = Terminal::new(TestBackend::new(80, 12)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);

        // The bottom border must close on its own row, not overlap a plan step.
        let bottom_rows: Vec<&str> = screen.lines().filter(|l| l.contains('╰')).collect();
        assert_eq!(
            bottom_rows.len(),
            1,
            "exactly one bottom-left corner:\n{screen}"
        );
        // The corner must be the first non-space glyph on its row (a closed
        // border), not sitting on top of a `✓`/`▸`/`☐` step glyph.
        let row = bottom_rows[0];
        assert!(
            row.trim_start().starts_with('╰'),
            "bottom border row must start with `╰`, got: {row:?}\n{screen}"
        );
        // The plan is truncated to fit, with a "… +N more" line, rather than
        // overflowing.
        assert!(
            screen.contains("… +3 more"),
            "plan truncated to fit:\n{screen}"
        );
        // The box never exceeds the terminal height.
        assert!(
            screen.lines().filter(|l| !l.trim().is_empty()).count() <= 12,
            "box fits on screen:\n{screen}"
        );

        // A taller terminal shows the whole plan with no truncation.
        let mut term2 = Terminal::new(TestBackend::new(175, 14)).unwrap();
        term2.draw(|f| app.render(f)).unwrap();
        let screen2 = dump(&term2);
        assert!(
            screen2.contains("step 7 with a fairly long title to be realistic"),
            "full plan shown when it fits:\n{screen2}"
        );
        assert!(!screen2.contains("… +"), "no truncation when it fits");

        // Extreme case: a plan so large the box would fill the whole screen.
        // The transcript must still get its Min(1) row and the border must stay
        // closed — the cap reserves a row for the transcript so Layout never
        // clamps the box rect.
        let mut app2 = test_app("openai", "gpt-4o");
        app2.apply(UiEvent::Plan(
            (0..20)
                .map(|i| PlanStep {
                    title: format!("step {i}"),
                    status: PlanStatus::Pending,
                })
                .collect(),
        ));
        app2.working = true;
        let mut term3 = Terminal::new(TestBackend::new(60, 10)).unwrap();
        term3.draw(|f| app2.render(f)).unwrap();
        let screen3 = dump(&term3);
        let bottom3: Vec<&str> = screen3.lines().filter(|l| l.contains('╰')).collect();
        assert_eq!(bottom3.len(), 1, "one bottom corner:\n{screen3}");
        assert!(
            bottom3[0].trim_start().starts_with('╰'),
            "border closed, not overlapping content:\n{screen3}"
        );
        // The transcript title row survives (top border with `hi ·`).
        assert!(
            screen3.contains("hi · openai · gpt-4o"),
            "transcript keeps its row:\n{screen3}"
        );

        // Degenerate tiny terminal: must not panic, and the box border stays closed.
        let mut term4 = Terminal::new(TestBackend::new(60, 3)).unwrap();
        term4.draw(|f| app2.render(f)).unwrap();
        let screen4 = dump(&term4);
        let bottom4: Vec<&str> = screen4.lines().filter(|l| l.contains('╰')).collect();
        assert_eq!(
            bottom4.len(),
            1,
            "one bottom corner on tiny term:\n{screen4}"
        );
    }

    #[test]
    fn startup_notice_does_not_clip_input_line() {
        // On first load, a startup notice (e.g. "provider metadata check
        // failed: …") is pinned above the status line. The box height must
        // account for it, or the input line gets clipped and the cursor lands
        // on the wrong row.
        let mut app = test_app("openai", "gpt-4o");
        app.startup_notice = Some("provider metadata check failed: connection refused".into());
        let mut term = Terminal::new(TestBackend::new(70, 10)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("provider metadata check failed"),
            "notice shown:\n{screen}"
        );
        // The input prompt must still be visible inside the box (not clipped).
        assert!(screen.contains('›'), "input prompt visible:\n{screen}");
        // The input box's bottom border closes cleanly (the transcript block
        // also has a `╰`, so check the last one — the input box is at the bottom).
        let bottom: Vec<&str> = screen.lines().filter(|l| l.contains('╰')).collect();
        let input_box_border = bottom.last().expect("input box bottom border");
        assert!(
            input_box_border.trim_start().starts_with('╰'),
            "input box border closed:\n{screen}"
        );
        // The notice, status, and prompt all render inside the input box —
        // i.e. above the input box's bottom border row (the last `╰…─` row).
        let rows: Vec<&str> = screen.lines().collect();
        let border_row_idx = rows
            .iter()
            .rposition(|l| l.trim_start().starts_with('╰') && l.contains('─'))
            .unwrap();
        let above_border: String = rows[..border_row_idx].join("\n");
        assert!(
            above_border.contains("provider metadata check failed") && above_border.contains('›'),
            "notice + prompt above the border:\n{screen}"
        );
    }

    #[test]
    fn quit_notice_renders_and_does_not_clip_input() {
        // After the first Ctrl-C (idle, empty input), a "Press Ctrl-C again to
        // exit" notice is pinned above the status line. The box height must
        // account for it or the input line clips and the cursor lands wrong.
        let mut app = test_app("openai", "gpt-4o");
        app.quit_notice = Some(Instant::now() + Duration::from_millis(1800));
        let mut term = Terminal::new(TestBackend::new(70, 10)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("Press Ctrl-C again to exit"),
            "quit notice shown:\n{screen}"
        );
        assert!(screen.contains('›'), "input prompt visible:\n{screen}");
        // The input box bottom border closes cleanly (not overlapping content).
        let bottom: Vec<&str> = screen.lines().filter(|l| l.contains('╰')).collect();
        let input_box_border = bottom.last().expect("input box bottom border");
        assert!(
            input_box_border.trim_start().starts_with('╰'),
            "input box border closed:\n{screen}"
        );
    }

    #[test]
    fn changed_files_line_shows_what_last_turn_touched() {
        // After a turn that changed files, a compact "changed: …" line sits
        // above the input so the user sees what was touched without scrolling.
        let mut app = test_app("openai", "gpt-4o");
        app.last_changed_files = vec!["src/a.rs".into(), "src/b.rs".into()];
        let mut term = Terminal::new(TestBackend::new(60, 12)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("changed: src/a.rs, src/b.rs"),
            "changed-files line: {screen}"
        );
        assert!(
            screen.contains("Ctrl-D for diff"),
            "diff toggle hint: {screen}"
        );
    }

    #[test]
    fn ctrl_d_toggles_the_diff_panel() {
        // Toggling Ctrl-D opens the panel with the cached diff text and a
        // header; toggling again closes it. We set diff_text directly to avoid
        // a real git call in the unit test.
        let mut app = test_app("openai", "gpt-4o");
        app.show_diff = true;
        app.diff_text = Some("--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-old\n+new\n".into());
        let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("diff (Ctrl-D to close)"),
            "panel header: {screen}"
        );
        assert!(screen.contains("+new"), "diff content rendered: {screen}");

        // Closing drops the panel.
        app.show_diff = false;
        app.diff_text = None;
        term.draw(|f| app.render(f)).unwrap();
        let screen2 = dump(&term);
        assert!(
            !screen2.contains("diff (Ctrl-D to close)"),
            "panel closed: {screen2}"
        );
    }

    #[test]
    fn ctrl_question_toggles_the_observability_panel() {
        // The Ctrl-? agent-observability panel renders the last turn's telemetry
        // counters, the per-turn tool-call count, and session/context numbers.
        let mut app = test_app("openai", "gpt-4o");
        app.show_debug = true;
        app.last_telemetry = Some(hi_agent::TurnTelemetry {
            verify_rounds: 2,
            recovery_retries: 1,
            repeat_nudges: 0,
            continue_nudges: 1,
            truncation_retries: 0,
            hit_step_cap: false,
            stalled_unfinished: false,
            stalled_repeating: false,
            verify_attributions: Vec::new(),
            tool_calls: 7,
            max_concurrent_batch: 3,
            serial_runs: 2,
            tool_timeline: Vec::new(),
            file_reads: 2,
            targeted_searches: 1,
            listing_only: false,
            first_tool_kind: "read".to_string(),
            discovery_depth: "mixed".to_string(),
            quality_repair_nudges: 0,
        });
        app.turn_tool_calls = 7;
        let mut term = Terminal::new(TestBackend::new(60, 16)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("agent (Ctrl-? to close)"),
            "panel header: {screen}"
        );
        assert!(
            screen.contains("2 verify")
                && screen.contains("1 retry")
                && screen.contains("1 continue"),
            "telemetry counters: {screen}"
        );
        assert!(
            screen.contains("tool calls this turn: 7"),
            "tool-call count: {screen}"
        );

        // Closing drops the panel.
        app.show_debug = false;
        term.draw(|f| app.render(f)).unwrap();
        let screen2 = dump(&term);
        assert!(
            !screen2.contains("agent (Ctrl-? to close)"),
            "panel closed: {screen2}"
        );
    }

    #[test]
    fn in_progress_line_is_styled_live() {
        // A heading still streaming (no trailing newline yet) renders styled with
        // its markers stripped — not literally as "## …" until the line commits.
        let mut app = test_app("openai", "gpt-4o");
        app.apply(UiEvent::Text("## Hello world".into()));
        let mut term = Terminal::new(TestBackend::new(60, 12)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("Hello world"),
            "heading text shown:\n{screen}"
        );
        assert!(
            !screen.contains("## Hello"),
            "marker stripped live:\n{screen}"
        );

        // Styling the preview must NOT advance the real fence state: a partial
        // opening fence leaves code_lang untouched until its line commits.
        let mut app2 = test_app("openai", "gpt-4o");
        app2.apply(UiEvent::Text("```rust".into()));
        term.draw(|f| app2.render(f)).unwrap();
        assert!(
            app2.code_lang.is_none(),
            "live preview must not mutate the committed fence state"
        );
    }

    #[test]
    fn edit_key_submits_on_enter_and_clears() {
        let mut app = test_app("openai", "gpt-4o");
        app.input.set("queue me");
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(app.edit_key(&enter).as_deref(), Some("queue me"));
        assert!(app.input.is_empty(), "input cleared after submit");
        // An empty Enter submits nothing.
        assert_eq!(app.edit_key(&enter), None);
    }

    #[test]
    fn renders_title_transcript_and_input() {
        let mut app = test_app("openai", "gpt-4o");
        app.push(Line::raw("› hello"));
        app.apply(UiEvent::Text("hi there\n".into()));
        app.apply(UiEvent::AssistantEnd);
        app.input.set("next question");

        let mut term = Terminal::new(TestBackend::new(50, 12)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);

        assert!(screen.contains("gpt-4o"), "title shows model");
        assert!(screen.contains("hello"), "user line");
        assert!(screen.contains("hi there"), "assistant line");
        assert!(screen.contains("next question"), "input box");
    }

    #[test]
    fn turn_end_sets_status_and_marks_transcript_done() {
        let mut app = test_app("openai", "gpt-4o");
        app.apply(UiEvent::TurnEnd("[10 in · 2 out · 12 total]".into()));
        // Usage in the title bar...
        assert!(app.status.contains("12 total"));
        // ...and a clear "done" marker in the transcript so the turn's end shows.
        assert_eq!(app.transcript.len(), 1);
        let line = app.transcript[0].text();
        assert!(line.contains("✓ done"), "got: {line}");
    }

    #[test]
    fn turn_end_renders_the_steer_suffix_from_the_summary() {
        // The agent appends a "steer" suffix to the usage summary for noisy
        // turns; the TUI renders that string verbatim, so the suffix surfaces
        // in both the status bar and the done marker with no TUI-specific code.
        let mut app = test_app("openai", "gpt-4o");
        let noisy = "[↑10 ↓2 · ctx 5% (500/10k) · steer: 2 verify · 1 retry]";
        app.apply(UiEvent::TurnEnd(noisy.into()));
        assert!(
            app.status.contains("steer: 2 verify"),
            "steer in status bar: {}",
            app.status
        );
        let line = app.transcript[0].text();
        assert!(line.contains("steer"), "steer in done marker: {line}");
    }

    #[test]
    fn assistant_text_becomes_copy_target() {
        let mut app = test_app("openai", "gpt-4o");
        app.apply(UiEvent::Text("first ".into()));
        app.apply(UiEvent::Text("answer\n".into()));
        app.apply(UiEvent::AssistantEnd);
        assert_eq!(app.last_assistant, "first answer");

        app.apply(UiEvent::ToolCall(
            "bash".into(),
            r#"{"command":"echo noisy"}"#.into(),
        ));
        app.apply(UiEvent::ToolResult("bash".into(), "noisy output".into()));
        assert_eq!(
            app.last_assistant, "first answer",
            "tool logs are not copied as the assistant response"
        );
    }

    #[test]
    fn transcript_text_serializes_lines() {
        let mut app = test_app("openai", "gpt-4o");
        app.push(Line::raw("one"));
        app.push(Line::from(vec![Span::raw("t"), Span::raw("wo")]));
        assert_eq!(app.transcript_text(), "one\ntwo");
    }

    #[test]
    fn completed_turn_without_summary_is_visible() {
        let mut app = test_app("openai", "gpt-4o");
        app.note_turn_completed_without_summary();
        let line = app.transcript.last().unwrap().text();
        assert!(line.contains("✓ done"), "got: {line}");
        assert!(line.contains("no usage reported"), "got: {line}");
        assert_eq!(app.status, "done · no usage reported");
    }

    #[test]
    fn stopped_after_tool_output_without_turn_end_is_visible() {
        let mut app = test_app("openai", "gpt-4o");
        app.apply(UiEvent::ToolCall(
            "edit".into(),
            r#"{"path":"src/main.rs"}"#.into(),
        ));
        app.apply(UiEvent::ToolResult(
            "edit".into(),
            "19 additions, 3 deletions".into(),
        ));
        app.note_turn_completed_without_summary();

        let lines: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
        assert!(
            lines
                .iter()
                .any(|line| line.contains("stopped after tool output")),
            "transcript: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("degraded in-session")),
            "transcript: {lines:?}"
        );
        assert_eq!(app.status, "stopped after tool output");
    }

    #[test]
    fn failed_turn_is_visible() {
        let mut app = test_app("openai", "gpt-4o");
        app.note_turn_failed(
            "provider disconnected",
            "outage",
            "check the provider's status page",
        );
        let line = app.transcript.last().unwrap().text();
        assert!(line.contains("✗ failed"), "got: {line}");
        assert!(line.contains("provider disconnected"), "got: {line}");
        assert!(line.contains("outage"), "got: {line}");
        assert!(line.contains("💡"), "shows guidance: {line}");
        assert!(
            app.status.contains("outage"),
            "status has kind: {}",
            app.status
        );
    }

    #[test]
    fn empty_tool_result_is_visible() {
        let mut app = test_app("openai", "gpt-4o");
        app.apply(UiEvent::ToolCall(
            "bash".into(),
            r#"{"command":"true"}"#.into(),
        ));
        app.apply(UiEvent::ToolResult("bash".into(), String::new()));
        let rendered: Vec<String> = app.transcript.iter().map(TranscriptEntry::text).collect();
        assert!(
            rendered.iter().any(|line| line.contains("(no output)")),
            "transcript: {rendered:?}"
        );
    }

    #[test]
    fn renders_fetching_spinner() {
        let mut app = test_app("pipenetwork", "ipop/coder-balanced");
        app.fetching = Some(Instant::now());
        let mut term = Terminal::new(TestBackend::new(60, 10)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("fetching models from pipenetwork"),
            "fetch spinner: {screen}"
        );
        assert!(screen.contains("Esc to cancel"), "cancel hint: {screen}");
    }

    #[test]
    fn renders_model_picker() {
        let mut app = test_app("openai", "openai/gpt-4o");
        app.picker = Some(ModelPicker::new(
            vec!["anthropic/claude-sonnet-4".into(), "openai/gpt-4o".into()],
            "openai/gpt-4o",
            HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        ));
        let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(screen.contains("select a model"), "title: {screen}");
        assert!(screen.contains("filter:"), "filter line: {screen}");
        assert!(screen.contains("claude-sonnet-4"), "lists models: {screen}");
        assert!(screen.contains("▶"), "highlights a selection: {screen}");
        // The active model is marked and pre-selected.
        assert!(
            screen.contains("(current)"),
            "marks current model: {screen}"
        );
    }

    #[test]
    fn picker_shows_health_tag() {
        let mut app = test_app("pipenetwork", "ipop/coder-balanced");
        let tags = HashMap::from([("claude-sonnet-4.6".to_string(), "degraded".to_string())]);
        app.picker = Some(ModelPicker::new(
            vec!["claude-sonnet-4.6".into(), "ipop/coder-balanced".into()],
            "ipop/coder-balanced",
            tags,
            &HashMap::new(),
            &HashMap::new(),
        ));
        let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("[degraded]"),
            "degraded tag shown: {screen}"
        );
    }

    #[test]
    fn renders_multiline_input() {
        let mut app = test_app("openai", "gpt-4o");
        app.input.insert_str("first\nsecond\nthird");
        let mut term = Terminal::new(TestBackend::new(40, 14)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("› first"),
            "first line with prompt: {screen}"
        );
        assert!(screen.contains("second"), "second line: {screen}");
        assert!(screen.contains("third"), "third line: {screen}");
    }

    #[test]
    fn alt_enter_and_backslash_insert_newline_instead_of_submitting() {
        let mut app = test_app("openai", "gpt-4o");
        app.input.set("line one");
        let alt_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT);
        assert_eq!(app.edit_key(&alt_enter), None, "alt+enter does not submit");
        assert_eq!(app.input.text(), "line one\n");

        // Trailing backslash + Enter continues the line (universal fallback).
        app.input.set("a\\");
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(app.edit_key(&enter), None, "backslash continues");
        assert_eq!(app.input.text(), "a\n");

        // A normal Enter still submits.
        app.input.set("go");
        assert_eq!(app.edit_key(&enter).as_deref(), Some("go"));
    }

    #[test]
    fn failed_turn_shows_reason_and_keeps_error() {
        let mut app = test_app("openai", "gpt-4o");
        app.note_turn_failed(
            "API error 401: invalid or expired session",
            "auth",
            "check your API key",
        );
        // record_model_issue runs next in the real flow; it must NOT clobber the
        // real error with a reliability-count message.
        app.record_model_issue();
        assert_eq!(
            app.last_error.as_deref(),
            Some("API error 401: invalid or expired session"),
            "the real error is preserved for /status and /log"
        );
        // The bottom bar shows the reason inline, not a bare "failed".
        let mut term = Terminal::new(TestBackend::new(80, 8)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("last: failed — API error 401"),
            "reason inline: {screen}"
        );
        assert!(screen.contains("/retry"), "recovery hint: {screen}");
    }

    #[test]
    fn backend_wait_notice_does_not_mark_model_degraded() {
        let mut app = test_app("pipenetwork", "ipop/coder-balanced");
        app.note_backend_waiting(Duration::from_secs(181), Duration::from_secs(180));

        assert_eq!(app.model_issues.get("ipop/coder-balanced"), None);
        assert_eq!(app.last_error, None);
        let mut term = Terminal::new(TestBackend::new(100, 8)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("Still thinking. Ctrl-C cancels; keep waiting to continue."),
            "soft wait notice shown: {screen}"
        );
        assert!(
            !screen.contains("degraded in-session"),
            "soft wait notice is not model health: {screen}"
        );
    }

    #[test]
    fn watchdog_timeout_default_is_longer_than_client_warning_window() {
        assert_eq!(
            watchdog_stuck_timeout_from_value(None),
            Duration::from_secs(180)
        );
        assert_eq!(
            watchdog_stuck_timeout_from_value(Some("5")),
            Duration::from_secs(30)
        );
        assert_eq!(
            watchdog_stuck_timeout_from_value(Some("9999")),
            Duration::from_secs(1_800)
        );
    }

    #[test]
    fn completion_opens_filters_and_closes() {
        let mut app = test_app("openai", "gpt-4o");
        app.input.set("/");
        app.sync_completion();
        assert_eq!(
            app.completion_items().len(),
            hi_agent::command::COMMANDS.len(),
            "bare slash lists every command"
        );
        app.input.set("/co");
        app.sync_completion();
        let labels: Vec<String> = app
            .completion_items()
            .iter()
            .map(|i| i.label.clone())
            .collect();
        assert!(
            labels.contains(&"/copy".to_string()) && labels.contains(&"/compact".to_string()),
            "got {labels:?}"
        );
        assert!(labels.iter().all(|n| n.starts_with("/co")));
        // A space after a command that takes no argument closes the menu.
        app.input.set("/diff ");
        app.sync_completion();
        assert!(app.completion.is_none());
    }

    #[test]
    fn completion_offers_verify_and_goal_keywords() {
        let mut app = test_app("openai", "gpt-4o");
        app.input.set("/verify ");
        app.sync_completion();
        let labels: Vec<String> = app
            .completion_items()
            .iter()
            .map(|i| i.label.clone())
            .collect();
        assert_eq!(labels, vec!["off"], "verify offers its disable keyword");
        app.input.set("/goal cl");
        app.sync_completion();
        let labels: Vec<String> = app
            .completion_items()
            .iter()
            .map(|i| i.label.clone())
            .collect();
        assert_eq!(labels, vec!["clear"], "goal offers its clear keyword");
        assert_eq!(app.accept_completion(true).as_deref(), Some("/goal clear"));
    }

    #[test]
    fn completion_offers_live_model_ids() {
        let mut app = test_app("openai", "gpt-4o");
        app.model_ids = vec!["gpt-4o".into(), "gpt-4o-mini".into(), "claude-opus".into()];
        app.input.set("/model gp");
        app.sync_completion();
        let labels: Vec<String> = app
            .completion_items()
            .iter()
            .map(|i| i.label.clone())
            .collect();
        assert_eq!(
            labels,
            vec!["gpt-4o", "gpt-4o-mini"],
            "filters the catalog by prefix"
        );
        // Accepting a row runs the full command.
        app.completion.as_mut().unwrap().selected = 1;
        assert_eq!(
            app.accept_completion(true).as_deref(),
            Some("/model gpt-4o-mini")
        );

        // With no catalog loaded, there's no inline menu — the picker still
        // handles `/model` (so the feature degrades, it doesn't break).
        let mut bare = test_app("openai", "gpt-4o");
        bare.input.set("/model gp");
        bare.sync_completion();
        assert!(bare.completion.is_none());
    }

    #[test]
    fn completion_offers_then_fills_compact_kinds() {
        let mut app = test_app("openai", "gpt-4o");
        // The space that used to kill the menu now offers the kinds.
        app.input.set("/compact ");
        app.sync_completion();
        let labels: Vec<String> = app
            .completion_items()
            .iter()
            .map(|i| i.label.clone())
            .collect();
        assert_eq!(labels, vec!["hybrid", "full", "elide"], "offers every kind");
        // Typing narrows by prefix.
        app.input.set("/compact e");
        app.sync_completion();
        let labels: Vec<String> = app
            .completion_items()
            .iter()
            .map(|i| i.label.clone())
            .collect();
        assert_eq!(labels, vec!["elide"]);
        // Accepting a kind fills the whole command and runs it on Enter.
        assert_eq!(
            app.accept_completion(true).as_deref(),
            Some("/compact elide")
        );
        assert!(app.completion.is_none(), "menu closes after accept");
    }

    #[test]
    fn completing_compact_name_opens_its_kind_menu() {
        let mut app = test_app("openai", "gpt-4o");
        app.input.set("/compact");
        app.sync_completion();
        // Tab accepts the command name, leaving `/compact `…
        app.accept_completion(false);
        assert_eq!(app.input.text(), "/compact ");
        // …and the re-sync the Tab handler performs opens the kind menu.
        app.sync_completion();
        let labels: Vec<String> = app
            .completion_items()
            .iter()
            .map(|i| i.label.clone())
            .collect();
        assert!(labels.contains(&"hybrid".to_string()), "got {labels:?}");
    }

    #[test]
    fn completion_navigation_and_accept() {
        let mut app = test_app("openai", "gpt-4o");
        // No-arg command: Enter accepts and submits immediately.
        app.input.set("/hel");
        app.sync_completion();
        let line = app.accept_completion(true);
        assert_eq!(line.as_deref(), Some("/help"));
        assert!(app.completion.is_none(), "menu closes after accept");

        // Arg-taking command: accept leaves a trailing space, does not submit.
        app.input.set("/mod");
        app.sync_completion();
        assert_eq!(
            app.accept_completion(true),
            None,
            "arg command waits for input"
        );
        assert_eq!(app.input.text(), "/model ");

        // Tab never submits, even for a no-arg command.
        app.input.set("/dif");
        app.sync_completion();
        assert_eq!(app.accept_completion(false), None);
        assert_eq!(app.input.text(), "/diff");
    }

    #[test]
    fn completion_move_clamps() {
        let mut app = test_app("openai", "gpt-4o");
        app.input.set("/co"); // [commit, compact, context, copy]
        app.sync_completion();
        let last = app.completion_items().len().saturating_sub(1);
        app.completion_move(-1); // already at 0, stays
        assert_eq!(app.completion.as_ref().unwrap().selected, 0);
        app.completion_move(1);
        assert_eq!(app.completion.as_ref().unwrap().selected, 1);
        // Move past the end to verify clamping.
        for _ in 0..last + 1 {
            app.completion_move(1);
        }
        assert_eq!(app.completion.as_ref().unwrap().selected, last);
    }

    #[test]
    fn renders_completion_menu() {
        let mut app = test_app("openai", "gpt-4o");
        app.input.set("/");
        app.sync_completion();
        let mut term = Terminal::new(TestBackend::new(72, 20)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(screen.contains("/help"), "lists help: {screen}");
        assert!(screen.contains("/model"), "lists model: {screen}");
        assert!(screen.contains("▶"), "highlights a row: {screen}");
    }

    /// `splash_lines` now leads with a ~2x block-letter "PipeNetwork.AI"
    /// banner (5 figlet rows, all orange bold), then the dim model line, the
    /// cwd, and a trailing blank. Verifies banner shape, orange+bold styling
    /// on every banner row, the model/cwd content, and the line count.
    #[test]
    fn splash_shows_full_pipenetwork_wordmark_in_orange() {
        let lines = splash_lines("openai", "gpt-4o", Some(128_000));

        // 5 banner rows + model line + cwd line + trailing blank = 8.
        assert_eq!(
            lines.len(),
            8,
            "expected 8 lines (5 banner + model + cwd + blank)"
        );

        // Banner: rows 0..5, each one span styled orange + bold, carrying
        // figlet strokes (pipes / underscores).
        for (i, line) in lines[0..5].iter().enumerate() {
            assert_eq!(
                line.spans.len(),
                1,
                "banner row {i} should be a single span, got {} spans",
                line.spans.len()
            );
            let span = &line.spans[0];
            assert_eq!(
                span.style.fg,
                Some(Color::Rgb(255, 140, 0)),
                "banner row {i} should be orange"
            );
            assert!(
                span.style.add_modifier.contains(Modifier::BOLD),
                "banner row {i} should be bold"
            );
            let text = span.content.to_string();
            assert!(
                text.contains('|') || text.contains('_'),
                "banner row {i} should carry figlet strokes, got: {text:?}"
            );
        }

        // Row 5: dim model line (model + provider + context window).
        let model_line: String = lines[5]
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect();
        assert!(
            model_line.contains("gpt-4o"),
            "model line missing model: {model_line:?}"
        );
        assert!(
            model_line.contains("openai"),
            "model line missing provider: {model_line:?}"
        );
        assert!(
            model_line.contains("128K context"),
            "model line missing context window: {model_line:?}"
        );

        // Row 6: cwd — non-empty.
        let cwd_line: String = lines[6]
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect();
        assert!(
            !cwd_line.is_empty(),
            "cwd line should be non-empty, got: {cwd_line:?}"
        );

        // Row 7: the blank breathing-room line.
        assert!(
            lines[7].spans.is_empty(),
            "last line should be the blank breathing-room line"
        );
    }
}
