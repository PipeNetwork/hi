//! Interactive TUI loop backed by `hi-harness` (Pipe Network).

use anyhow::{Context, Result};
use crossterm::event::{
    DisableMouseCapture, EnableBracketedPaste, EnableFocusChange, EnableMouseCapture, Event,
    EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, enable_raw_mode};
use futures_util::StreamExt;
use hi_harness::{
    Command, ConfirmationResult, EffortArg, Harness, LiveSettings, PermissionMode,
    TurnCancellation, UsageTab, parse_command, parse_effort_arg, parse_model_args,
    resolve_model_query,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use std::collections::HashMap;
use std::io;
use std::io::IsTerminal;
use std::path::PathBuf;
use tokio::sync::mpsc;

use crate::confirm_overlay::ConfirmDecision;
use crate::event::{ChannelUi, ConfirmationControl, Restore};
use crate::model_picker::ModelPicker;
use crate::render::dim;
use crate::{App, TICK};

pub type PipenetworkLoginHook = Box<dyn FnMut() -> anyhow::Result<()> + Send + 'static>;

pub struct SessionOptions {
    pub provider: String,
    pub model: String,
    pub history_path: Option<PathBuf>,
    pub startup_prompt: Option<String>,
    /// Retry an unmatched PendingTurn through the Ask overlay (not StdoutUi).
    pub resume_incomplete: bool,
    /// After `/login pipenetwork`, persist `[profiles.pipenetwork]`.
    pub on_pipenetwork_login: Option<PipenetworkLoginHook>,
    /// Saved sessions for `/sessions`.
    pub list_sessions: Option<crate::SessionLister>,
    /// Configured OpenAI profile for dashboard `openai/…` rows.
    pub openai_api_key: Option<String>,
    pub openai_base_url: Option<String>,
    /// Current session JSONL, if this run is saved.
    pub session_path: Option<PathBuf>,
    /// `--no-save`: persist Sentinel enablement but do not exec.
    pub no_save: bool,
    /// RSI/managed mode must not wrap this process.
    pub sentinel_blocked: Option<String>,
}

pub async fn run_session(harness: &mut Harness, options: SessionOptions) -> Result<()> {
    if !io::stdin().is_terminal() {
        anyhow::bail!("TUI requires an interactive stdin");
    }
    let termios = crate::autoharnessfix::snapshot_termios();
    let ahf = crate::autoharnessfix::AutoharnessfixOpts {
        session_path: options.session_path,
        no_save: options.no_save,
        sentinel_blocked: options.sentinel_blocked,
    };
    enable_raw_mode().context("entering raw mode")?;
    let mut restore = Some(Restore);
    execute!(io::stdout(), EnterAlternateScreen).context("entering alternate screen")?;
    let _ = execute!(io::stdout(), EnableBracketedPaste);
    let _ = execute!(io::stdout(), EnableFocusChange);
    // Grok reports the mouse so click-to-expand (`›` rows) and wheel scroll
    // work. `/mouse off` turns this back into native terminal selection.
    let _ = execute!(io::stdout(), EnableMouseCapture);
    let mut terminal =
        Terminal::new(CrosstermBackend::new(io::stdout())).context("creating terminal")?;

    let mut on_pipenetwork_login = options.on_pipenetwork_login;
    let mut app = App::new(&options.provider, &options.model);
    app.session_lister = options.list_sessions;
    app.workspace_root = harness.workspace_root().to_path_buf();
    app.api_key = harness.api_key().to_string();
    app.pipe_base_url = harness.base_url().to_string();
    app.openai_api_key = options.openai_api_key;
    app.openai_base_url = options.openai_base_url;
    app.input_history_path = harness.workspace_root().join(".hi").join("input-history");
    app.permission_mode = harness.permission_mode();
    app.reasoning_effort = harness.reasoning_effort();
    if app.input_history_path.exists() {
        app.input.load_history_file(&app.input_history_path);
    }
    if let Some(path) = &options.history_path
        && let Ok(text) = std::fs::read_to_string(path)
    {
        app.input.history = text
            .lines()
            .map(str::to_string)
            .filter(|line| !line.trim().is_empty())
            .collect();
    }
    if harness.messages().is_empty() {
        app.push(Line::styled(
            format!(
                "hi · pipenetwork · {} — /login to connect, /help for commands",
                harness.model()
            ),
            dim(),
        ));
        if crate::tutorial::should_offer(
            options.startup_prompt.is_none(),
            std::env::var_os("HI_SKIP_TUTORIAL").is_some(),
            crate::tutorial::already_offered(),
        ) {
            app.tutorial = Some(crate::tutorial::TutorialOverlay::fresh());
            crate::tutorial::mark_offered();
        }
    } else {
        super::hydrate::hydrate_transcript(&mut app, harness.messages());
        app.push(Line::styled(
            format!(
                "resumed {} message(s) · {} — /help for commands",
                harness.messages().len(),
                harness.model()
            ),
            dim(),
        ));
    }
    let mut startup_prompt = options.startup_prompt;
    let mut resume_incomplete = options.resume_incomplete;

    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<Event>();
    tokio::spawn(async move {
        let mut events = EventStream::new();
        while let Some(Ok(event)) = events.next().await {
            if input_tx.send(event).is_err() {
                break;
            }
        }
    });
    let mut ticker = tokio::time::interval(TICK);

    loop {
        terminal.draw(|frame| app.render(frame))?;
        if app.exit_requested {
            break;
        }
        if resume_incomplete {
            resume_incomplete = false;
            run_turn(
                &mut terminal,
                &mut input_rx,
                &mut ticker,
                &mut app,
                harness,
                "",
                true,
            )
            .await?;
            continue;
        }
        if let Some(line) = startup_prompt.take() {
            run_prompt(
                &mut terminal,
                &mut input_rx,
                &mut ticker,
                &mut app,
                harness,
                &mut on_pipenetwork_login,
                &line,
                &ahf,
                &mut restore,
                termios.as_ref(),
            )
            .await?;
            continue;
        }
        tokio::select! {
            _ = ticker.tick() => {
                app.spinner = app.spinner.wrapping_add(1);
                drain_pending_pipenetwork_login(&mut app, harness, &mut on_pipenetwork_login).await;
            }
            maybe = input_rx.recv() => {
                let Some(event) = maybe else { break };
                if let Some(line) = handle_idle_event(&mut app, harness, event, &mut on_pipenetwork_login, &ahf, &mut restore, termios.as_ref()).await? {
                    if line == "/quit" {
                        break;
                    }
                    run_prompt(
                        &mut terminal,
                        &mut input_rx,
                        &mut ticker,
                        &mut app,
                        harness,
                        &mut on_pipenetwork_login,
                        &line,
                        &ahf,
                        &mut restore,
                        termios.as_ref(),
                    )
                    .await?;
                }
            }
        }
    }
    Ok(())
}

async fn handle_idle_event(
    app: &mut App,
    harness: &mut Harness,
    event: Event,
    on_login: &mut Option<PipenetworkLoginHook>,
    ahf: &crate::autoharnessfix::AutoharnessfixOpts,
    restore: &mut Option<Restore>,
    termios: Option<&libc::termios>,
) -> Result<Option<String>> {
    match event {
        Event::Resize(..) => Ok(None),
        Event::Mouse(mouse) if app.mouse_capture => {
            if crate::dashboard::is_open(app) {
                return Ok(None);
            }
            if let Some(overlay) = app.usage_overlay.as_mut() {
                crate::usage::handle_click(overlay, mouse.column, mouse.row);
            } else {
                app.handle_mouse(mouse);
            }
            Ok(None)
        }
        Event::Paste(text) => {
            if crate::dashboard::is_open(app)
                && let Some(overlay) = app.dashboard.as_mut()
            {
                overlay.draft.insert_str(&text);
                return Ok(None);
            }
            app.paste_into_prompt(&text);
            Ok(None)
        }
        Event::Key(key) if key.kind == KeyEventKind::Press => {
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            // Grok Ctrl+E is global: expand/collapse thinking even while the
            // tutorial, a picker, or the prompt overlay is up.
            if crate::keys::is_toggle_reasoning_key(&key) {
                app.apply_action(crate::action::Action::ToggleReasoning);
                return Ok(None);
            }
            if super::idle::handle_dashboard_event(app, harness, &key) {
                return Ok(None);
            }
            if matches!(key.code, KeyCode::Char('c')) && ctrl {
                app.exit_requested = true;
                return Ok(None);
            }
            if matches!(key.code, KeyCode::Char('d')) && ctrl && app.input.is_empty() {
                app.exit_requested = true;
                return Ok(None);
            }
            if let Some(overlay) = app.usage_overlay.as_mut() {
                if let Some(text) = crate::usage::copy_request(overlay, &key) {
                    crate::usage::apply_copy(app, &text);
                    return Ok(None);
                }
                if crate::usage::handle_key(overlay, &key) == crate::usage::UsageOutcome::Close {
                    app.usage_overlay = None;
                }
                return Ok(None);
            }
            if let Some(overlay) = app.tutorial.as_mut() {
                if crate::tutorial::handle_key(overlay, &key)
                    == crate::tutorial::TutorialOutcome::Close
                {
                    app.tutorial = None;
                }
                return Ok(None);
            }
            if super::idle::handle_picker_key(app, harness, &key) {
                return Ok(None);
            }
            if matches!(key.code, KeyCode::Char('m')) && ctrl && app.input.is_empty() {
                open_model_picker(app, harness).await?;
                return Ok(None);
            }
            if key.code == KeyCode::BackTab {
                cycle_permissions(app, &harness.live());
                harness.persist_live_knobs();
                return Ok(None);
            }
            if let Some(line) = app.edit_key(&key) {
                let line = line.trim().to_string();
                if line.is_empty() {
                    return Ok(None);
                }
                if let Some(command) = parse_command(&line) {
                    if let Some(prompt) =
                        handle_command(app, harness, command, on_login, ahf, restore, termios)
                            .await?
                    {
                        return Ok(Some(prompt));
                    }
                    return Ok(None);
                }
                return Ok(Some(line));
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

async fn handle_command(
    app: &mut App,
    harness: &mut Harness,
    command: Command,
    on_login: &mut Option<PipenetworkLoginHook>,
    ahf: &crate::autoharnessfix::AutoharnessfixOpts,
    restore: &mut Option<Restore>,
    termios: Option<&libc::termios>,
) -> Result<Option<String>> {
    match command {
        Command::Quit => app.exit_requested = true,
        Command::Help(_) => {
            app.push(Line::styled(
                "/login /auth /sessions /doctor /autoharnessfix /model /effort /permissions /undo /diff /retry /verify /compact /copy /usage /dashboard /status /exit  (type / for the menu)",
                dim(),
            ));
        }
        Command::AutoHarnessFix(arg) => {
            return crate::autoharnessfix::handle(app, harness, &arg, ahf, restore, termios);
        }
        Command::Login(arg) => start_pipenetwork_login(app, harness, &arg, on_login).await?,
        Command::Logout(arg) => logout_pipenetwork(app, harness, &arg),
        Command::Auth(arg) => paste_pipenetwork_key(app, harness, &arg, on_login).await?,
        Command::Clear => {
            harness.clear_history();
            app.transcript.clear();
            app.push(Line::styled("conversation cleared", dim()));
        }
        Command::Undo => {
            let msg = match harness.undo().await {
                Ok(Some(n)) => format!("restored {n} path(s)"),
                Ok(None) => "nothing to undo".into(),
                Err(err) => format!("undo failed: {err:#}"),
            };
            app.push(Line::styled(msg, dim()));
        }
        Command::Diff => {
            let diff = hi_tools::working_tree_diff_in(harness.workspace_root()).await;
            if diff.trim().is_empty() {
                app.push(Line::styled("working tree clean", dim()));
            } else {
                for line in diff.lines() {
                    app.push(Line::from(line.to_string()));
                }
            }
        }
        Command::Retry => match app.last_prompt.clone() {
            Some(prompt) => {
                harness.truncate_messages(app.last_turn_start);
                return Ok(Some(prompt));
            }
            None => app.push(Line::styled("nothing to retry", dim())),
        },
        Command::Density(arg) => apply_density_command(app, &arg),
        Command::Theme(arg) => apply_theme_command(app, &arg),
        Command::Sessions(_) => list_sessions_command(app),
        Command::Rewind(arg) => match arg.trim().parse::<usize>() {
            Ok(n) => match harness.rewind_to_user_turn(n) {
                Ok(()) => {
                    app.transcript.clear();
                    super::hydrate::hydrate_transcript(app, harness.messages());
                    app.push(Line::styled(format!("rewound to user turn {n}"), dim()));
                }
                Err(err) => app.push(Line::styled(format!("{err:#}"), dim())),
            },
            Err(_) => app.push(Line::styled("use /rewind <n>", dim())),
        },
        Command::Doctor => {
            for line in harness.doctor_report().await.lines() {
                app.push(Line::styled(line.to_string(), dim()));
            }
        }
        Command::Version => {
            app.push(Line::styled(
                format!("hi {}", hi_harness::Harness::version()),
                dim(),
            ));
        }
        Command::Model(arg) => {
            let parsed = parse_model_args(&arg);
            if parsed.model.is_none() && parsed.effort.is_none() {
                open_model_picker(app, harness).await?;
            } else if let Some(model) = parsed.model {
                apply_model(app, &harness.live(), &model, parsed.effort);
                harness.set_model(app.model.clone());
                harness.persist_live_knobs();
            } else if let Some(effort) = parsed.effort {
                apply_effort(app, &harness.live(), effort);
                harness.persist_live_knobs();
                app.push(Line::styled(model_status(&harness.live()), dim()));
            }
        }
        Command::Effort(arg) => {
            set_effort_command(app, &harness.live(), &arg);
            harness.persist_live_knobs();
        }
        Command::Config(arg) => {
            let mut parts = arg.splitn(2, char::is_whitespace);
            match parts.next() {
                None | Some("") => app.push(Line::styled(model_status(&harness.live()), dim())),
                Some("reasoning" | "effort") => {
                    set_effort_command(app, &harness.live(), parts.next().unwrap_or(""));
                    harness.persist_live_knobs();
                }
                _ => app.push(Line::styled(
                    "use /config reasoning <low|medium|high|xhigh|off>",
                    dim(),
                )),
            }
        }
        Command::Verify(arg) => {
            if arg.is_empty() {
                let msg = harness
                    .verify_command()
                    .map(|cmd| format!("verify: {cmd}"))
                    .unwrap_or_else(|| "verify: off".into());
                app.push(Line::styled(msg, dim()));
            } else {
                harness.set_verify_command(Some(arg.clone()));
                app.push(Line::styled(
                    format!("verify: {}", harness.verify_command().unwrap_or("off")),
                    dim(),
                ));
            }
        }
        Command::Permissions(arg) => {
            apply_permissions_command(app, &harness.live(), &arg);
            harness.persist_live_knobs();
        }
        Command::Compact(arg) => {
            app.push(Line::styled("compacting conversation…", dim()));
            let mut ui = hi_harness::TestUi::default();
            let note = arg.trim();
            match harness
                .compact(
                    (!note.is_empty()).then_some(note),
                    &mut ui,
                    &hi_harness::TurnCancellation::new(),
                )
                .await
            {
                Ok(true) => {
                    app.transcript.clear();
                    super::hydrate::hydrate_transcript(app, harness.messages());
                    app.push(Line::styled("compacted conversation", dim()));
                }
                Ok(false) => app.push(Line::styled("nothing to compact", dim())),
                Err(err) => app.push(Line::styled(format!("compact failed: {err:#}"), dim())),
            }
        }
        Command::Files => {
            let files = harness.last_changed_files();
            if files.is_empty() {
                app.push(Line::styled("no files changed this session", dim()));
            } else {
                for file in files {
                    app.push(Line::from(file.clone()));
                }
            }
        }
        Command::Status => {
            let usage = harness.session_usage();
            let occupancy = usage.context_occupancy.max(usage.input_tokens);
            let sandbox = if harness.sandbox_enforced() {
                harness.sandbox_backend_name()
            } else {
                "off"
            };
            app.push(Line::styled(
                format!(
                    "{} · {} · reasoning {} · ctx {}/{} · in {} / out {} · undo {} · sandbox {sandbox}",
                    harness.model(),
                    harness.permission_mode().describe(),
                    harness
                        .reasoning_effort()
                        .map(|effort| effort.as_str())
                        .unwrap_or("off"),
                    occupancy,
                    harness.context_window(),
                    usage.input_tokens,
                    usage.output_tokens,
                    harness.checkpoint_count()
                ),
                dim(),
            ));
        }
        Command::Usage(arg) => apply_usage_command(app, Some(harness), &arg),
        Command::Dashboard => {
            if let Err(err) = crate::dashboard::open_from_harness(app, harness) {
                app.push(Line::styled(err, dim()));
            }
        }
        Command::Copy(arg) => app.copy(&arg),
        Command::Mouse(arg) => apply_mouse_command(app, &arg),
        Command::Tutorial => {
            app.tutorial = Some(crate::tutorial::TutorialOverlay::fresh());
        }
        Command::Removed(msg) | Command::Unknown(msg) => {
            app.push(Line::styled(msg, dim()));
        }
    }
    Ok(None)
}

async fn open_model_picker(app: &mut App, harness: &Harness) -> Result<()> {
    match harness.list_models().await {
        Ok(models) => {
            let ids: Vec<String> = models.iter().map(|model| model.id.clone()).collect();
            if ids.is_empty() {
                app.push(Line::styled(
                    format!(
                        "no models from {} — type /model <id> to switch",
                        harness.base_url()
                    ),
                    dim(),
                ));
                return Ok(());
            }
            let served: HashMap<_, _> = models
                .into_iter()
                .map(|model| (model.id.clone(), model))
                .collect();
            app.model_ids = ids.clone();
            app.served = served.clone();
            app.session_picker = false;
            app.picker = Some(ModelPicker::new(
                ids,
                &harness.model(),
                HashMap::new(),
                &served,
            ));
        }
        Err(err) => {
            app.push(Line::styled(
                format!("could not list models: {err:#}"),
                dim(),
            ));
            app.push(Line::styled(
                format!(
                    "current: {} — type /model <id> [low|medium|high|xhigh] to switch",
                    harness.model()
                ),
                dim(),
            ));
        }
    }
    Ok(())
}

fn set_effort_command(app: &mut App, live: &LiveSettings, arg: &str) {
    if arg.trim().is_empty() {
        app.push(Line::styled(model_status(live), dim()));
        return;
    }
    match parse_effort_arg(arg) {
        Ok(effort) => {
            apply_effort(app, live, effort);
            app.push(Line::styled(model_status(live), dim()));
        }
        Err(message) => app.push(Line::styled(message, dim())),
    }
}

pub(super) fn apply_model(
    app: &mut App,
    live: &LiveSettings,
    query: &str,
    effort: Option<EffortArg>,
) {
    let id = resolve_model_query(query, &app.model_ids);
    live.set_model(id);
    app.model = live.model();
    if let Some(served) = app.served.get(&app.model) {
        app.usage_pricing = served.price;
        app.context_window = served.context_window;
    }
    if let Some(effort) = effort {
        apply_effort(app, live, effort);
    }
    app.push(Line::styled(model_status(live), dim()));
}

fn apply_effort(app: &mut App, live: &LiveSettings, effort: EffortArg) {
    live.apply_effort_arg(effort);
    app.reasoning_effort = live.reasoning_effort();
}

fn apply_permissions_command(app: &mut App, live: &LiveSettings, arg: &str) {
    let current = live.permission_mode();
    let next = match arg.trim().to_ascii_lowercase().as_str() {
        "" => {
            app.push(Line::styled(permissions_status(live), dim()));
            return;
        }
        "toggle-always" => current.toggle_always(),
        "toggle-auto" => current.toggle_auto(),
        other => match PermissionMode::from_arg(other) {
            Some(mode) => mode,
            None => {
                app.push(Line::styled(
                    "use /permissions ask|auto|always  (or /auto, /yolo)",
                    dim(),
                ));
                return;
            }
        },
    };
    live.set_permission_mode(next);
    app.permission_mode = next;
    app.push(Line::styled(permissions_status(live), dim()));
}

fn cycle_permissions(app: &mut App, live: &LiveSettings) {
    let next = live.permission_mode().cycle();
    live.set_permission_mode(next);
    app.permission_mode = next;
    app.push(Line::styled(permissions_status(live), dim()));
}

fn permissions_status(live: &LiveSettings) -> String {
    format!("permissions: {}", live.permission_mode().describe())
}

fn model_status(live: &LiveSettings) -> String {
    match live.reasoning_effort() {
        Some(effort) => format!("model: {} · reasoning: {}", live.model(), effort.as_str()),
        None => format!("model: {} · reasoning: off", live.model()),
    }
}

fn list_sessions_command(app: &mut App) {
    let Some(lister) = app.session_lister.as_ref() else {
        app.push(Line::styled(
            "resume with `hi resume` or `hi --resume <id>`",
            dim(),
        ));
        return;
    };
    let sessions = lister();
    if sessions.is_empty() {
        app.push(Line::styled("no saved sessions in this project", dim()));
        return;
    }
    for session in sessions.into_iter().take(20) {
        app.push(Line::styled(
            format!("{}  {} ago", session.id, session.age),
            dim(),
        ));
    }
    app.push(Line::styled(
        "resume with `hi --resume <id>` (or `hi resume` for latest)",
        dim(),
    ));
}

async fn paste_pipenetwork_key(
    app: &mut App,
    harness: &mut Harness,
    arg: &str,
    on_login: &mut Option<PipenetworkLoginHook>,
) -> Result<()> {
    let _awaiting = harness.awaiting_user();
    let key = match parse_tui_auth_key(arg) {
        Ok(key) => key,
        Err(message) => {
            app.push(Line::styled(message, dim()));
            return Ok(());
        }
    };
    harness.set_api_key(&key);
    match harness.list_models().await {
        Ok(_) => {
            let _ = hi_ai::auth_store::save(
                hi_ai::pipenetwork_auth::PROVIDER_ID,
                &hi_ai::auth_store::StoredToken::static_access(key),
            );
            if let Some(hook) = on_login.as_mut()
                && let Err(err) = hook()
            {
                app.push(Line::styled(
                    format!("key works but config write failed: {err:#}"),
                    dim(),
                ));
            } else {
                app.push(Line::styled(
                    "api key stored — /login also pairs a browser session",
                    dim(),
                ));
            }
        }
        Err(err) => {
            app.push(Line::styled(format!("key rejected: {err:#}"), dim()));
        }
    }
    Ok(())
}

fn parse_tui_auth_key(arg: &str) -> std::result::Result<String, String> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Err("usage: /auth pipenetwork <api-key>  (or `hi auth pipenetwork`)".into());
    }
    let (name, rest) = match arg.split_once(char::is_whitespace) {
        Some((name, rest)) => (name, rest.trim()),
        None => ("pipenetwork", arg),
    };
    match name {
        "pipenetwork" | "pipe" if !rest.is_empty() => Ok(rest.to_string()),
        "pipenetwork" | "pipe" => Err("usage: /auth pipenetwork <api-key>".into()),
        key if rest.is_empty() => Ok(key.to_string()),
        other => Err(format!("use /auth pipenetwork <key> (got {other})")),
    }
}

fn apply_density_command(app: &mut App, arg: &str) {
    let next = if arg.trim().is_empty() {
        app.density.next()
    } else if let Some(parsed) = crate::Density::parse(arg) {
        parsed
    } else {
        app.push(Line::styled(
            "use /density compact|comfortable|verbose",
            dim(),
        ));
        return;
    };
    app.density = next;
    app.bump_transcript();
    app.push(Line::styled(format!("density: {}", next.label()), dim()));
}

fn apply_usage_command(app: &mut App, harness: Option<&Harness>, arg: &str) {
    let arg = arg.trim();
    if arg.eq_ignore_ascii_case("manage") {
        match hi_harness::open_billing_url() {
            Ok(()) => app.push(Line::styled(
                format!("opened {}", hi_harness::BILLING_URL),
                dim(),
            )),
            Err(err) => app.push(Line::styled(
                format!("{err} — {}", hi_harness::BILLING_URL),
                dim(),
            )),
        }
        return;
    }
    let Some(tab) = UsageTab::from_arg(arg) else {
        app.push(Line::styled(
            "use /usage  or  /usage show|manage  (alias /cost; /context for the context tab)",
            dim(),
        ));
        return;
    };
    let snapshot = match harness {
        Some(harness) => harness.usage_snapshot(),
        None => crate::usage::snapshot_from_app(app),
    };
    app.usage_overlay = Some(crate::usage::UsageOverlay::new(snapshot, tab));
}

fn apply_mouse_command(app: &mut App, arg: &str) {
    let on = match arg.trim().to_ascii_lowercase().as_str() {
        "" | "toggle" => !app.mouse_capture,
        "on" | "enable" | "capture" => true,
        "off" | "disable" | "native" => false,
        _ => {
            app.push(Line::styled(
                "use /mouse on (click › to expand) or /mouse off (terminal highlight-to-copy)",
                dim(),
            ));
            return;
        }
    };
    app.mouse_capture = on;
    if on {
        let _ = execute!(io::stdout(), EnableMouseCapture);
        app.push(Line::styled(
            "mouse: click › to expand, wheel scrolls, drag copies. /mouse off for terminal selection",
            dim(),
        ));
    } else {
        let _ = execute!(io::stdout(), DisableMouseCapture);
        app.push(Line::styled(
            "mouse: native selection — highlight to copy. /mouse on for wheel/click",
            dim(),
        ));
    }
}

fn apply_theme_command(app: &mut App, arg: &str) {
    let mode = if arg.trim().is_empty() {
        crate::theme::cycle_mode()
    } else if let Some(parsed) = crate::theme::ThemeMode::parse(arg) {
        crate::theme::set_mode(parsed);
        parsed
    } else {
        app.push(Line::styled(
            "use /theme dark|light|tokyonight|oscura|rosepine|ansi|auto",
            dim(),
        ));
        return;
    };
    app.push(Line::styled(format!("theme: {}", mode.label()), dim()));
}

fn handle_running_key(
    app: &mut App,
    live: &LiveSettings,
    steer: &hi_harness::SteerQueue,
    key: &KeyEvent,
    cancel: &TurnCancellation,
    pending_confirm: &mut Option<ConfirmationControl>,
) {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    if crate::keys::is_toggle_reasoning_key(key) {
        app.apply_action(crate::action::Action::ToggleReasoning);
        return;
    }
    if crate::dashboard::is_open(app) {
        let action = app.dashboard.as_mut().expect("dashboard").handle_key(key);
        if action == crate::dashboard::DashAction::Close {
            crate::dashboard::hide(app);
            return;
        }
        if let Some(overlay) = app.dashboard.as_mut() {
            crate::dashboard::apply_action(overlay, action);
        }
        return;
    }
    if crate::dashboard::is_dashboard_toggle(key) {
        if let Err(err) = crate::dashboard::open_from_app(app) {
            app.push(Line::styled(err, dim()));
        }
        return;
    }
    if let Some(overlay) = app.usage_overlay.as_mut() {
        if let Some(text) = crate::usage::copy_request(overlay, key) {
            crate::usage::apply_copy(app, &text);
            return;
        }
        if crate::usage::handle_key(overlay, key) == crate::usage::UsageOutcome::Close {
            app.usage_overlay = None;
        }
        return;
    }
    if key.code == KeyCode::BackTab {
        cycle_permissions(app, live);
        resolve_confirm_if_allowed(app, live, pending_confirm);
        return;
    }
    if let Some(pending) = pending_confirm.take() {
        let start_slash = matches!(key.code, KeyCode::Char('/')) && !ctrl;
        let in_command = start_slash || !app.input.is_empty() || app.completion.is_some();
        if !in_command {
            match crate::confirm_overlay::handle_key(app, key, &pending.request) {
                ConfirmDecision::Approve => {
                    app.confirmation = None;
                    let _ = pending.response.send(ConfirmationResult::Approved);
                }
                ConfirmDecision::AlwaysSession => {
                    live.set_permission_mode(PermissionMode::Always);
                    app.permission_mode = PermissionMode::Always;
                    app.push(Line::styled(permissions_status(live), dim()));
                    app.confirmation = None;
                    let _ = pending.response.send(ConfirmationResult::Approved);
                }
                ConfirmDecision::AlwaysPath => {
                    if let hi_harness::ConfirmationRequest::FileEdit { path, .. } = &pending.request
                    {
                        app.add_auto_approve_path(path);
                    }
                    app.confirmation = None;
                    let _ = pending.response.send(ConfirmationResult::Approved);
                }
                ConfirmDecision::Reject => {
                    app.confirmation = None;
                    let _ = pending.response.send(ConfirmationResult::Rejected);
                }
                ConfirmDecision::Cancel => {
                    app.confirmation = None;
                    let _ = pending.response.send(ConfirmationResult::Cancelled);
                }
                ConfirmDecision::RejectFollowup(text) => {
                    app.confirmation = None;
                    let _ = pending.response.send(ConfirmationResult::Rejected);
                    if !text.trim().is_empty() {
                        app.queue.push_back(text);
                        app.clamp_queue_selection();
                    }
                }
                ConfirmDecision::Redraw => {
                    *pending_confirm = Some(pending);
                }
                ConfirmDecision::Unhandled | ConfirmDecision::Ask(_) => {
                    *pending_confirm = Some(pending);
                    if let Some(line) = app.edit_key(key) {
                        apply_running_line(app, live, steer, &line, pending_confirm);
                    }
                }
            }
            return;
        }
        *pending_confirm = Some(pending);
    }
    if matches!(key.code, KeyCode::Char('c')) && ctrl {
        cancel.cancel();
        if let Some(flag) = app.interrupt.as_ref() {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
        return;
    }
    if key.code == KeyCode::Esc && app.input.is_empty() && app.completion.is_none() {
        cancel.cancel();
        if let Some(flag) = app.interrupt.as_ref() {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
        return;
    }
    if let Some(line) = app.edit_key(key) {
        apply_running_line(app, live, steer, &line, pending_confirm);
    }
}

fn apply_running_line(
    app: &mut App,
    live: &LiveSettings,
    steer: &hi_harness::SteerQueue,
    line: &str,
    pending_confirm: &mut Option<ConfirmationControl>,
) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    if let Some(command) = parse_command(line) {
        match command {
            Command::Permissions(arg) => {
                apply_permissions_command(app, live, &arg);
                resolve_confirm_if_allowed(app, live, pending_confirm);
            }
            Command::Effort(arg) => {
                let changing = !arg.trim().is_empty();
                set_effort_command(app, live, &arg);
                if changing {
                    app.push(Line::styled("applies on the next model call", dim()));
                }
            }
            Command::Model(arg) => apply_running_model(app, live, &arg),
            Command::Config(arg) => {
                let mut parts = arg.splitn(2, char::is_whitespace);
                match parts.next() {
                    None | Some("") => app.push(Line::styled(model_status(live), dim())),
                    Some("reasoning" | "effort") => {
                        let rest = parts.next().unwrap_or("");
                        let changing = !rest.trim().is_empty();
                        set_effort_command(app, live, rest);
                        if changing {
                            app.push(Line::styled("applies on the next model call", dim()));
                        }
                    }
                    _ => app.push(Line::styled(
                        "use /config reasoning <low|medium|high|xhigh|off>",
                        dim(),
                    )),
                }
            }
            Command::Help(_) => {
                app.push(Line::styled(
                    "/yolo /auto /permissions /effort /model still work while a turn runs",
                    dim(),
                ));
            }
            Command::Status => {
                app.push(Line::styled(
                    format!("{} · {}", permissions_status(live), model_status(live)),
                    dim(),
                ));
            }
            Command::Usage(arg) => apply_usage_command(app, None, &arg),
            Command::Dashboard => {
                if let Err(err) = crate::dashboard::open_from_app(app) {
                    app.push(Line::styled(err, dim()));
                }
            }
            Command::Quit => {
                app.push(Line::styled(
                    "a turn is running — Ctrl-C or Esc to stop, then /exit",
                    dim(),
                ));
            }
            Command::Removed(msg) | Command::Unknown(msg) => {
                app.push(Line::styled(msg, dim()));
            }
            _ => app.push(Line::styled(
                "command not available while a turn is running",
                dim(),
            )),
        }
        return;
    }
    steer.push(line.to_string());
    app.push(Line::styled(
        "steering · applies on the next model call",
        dim(),
    ));
}

fn apply_running_model(app: &mut App, live: &LiveSettings, arg: &str) {
    let parsed = parse_model_args(arg);
    if parsed.model.is_none() && parsed.effort.is_none() {
        app.push(Line::styled(
            format!(
                "{} — type /model <id> [effort] while a turn is running",
                model_status(live)
            ),
            dim(),
        ));
        return;
    }
    if let Some(model) = parsed.model {
        apply_model(app, live, &model, parsed.effort);
    } else if let Some(effort) = parsed.effort {
        apply_effort(app, live, effort);
        app.push(Line::styled(model_status(live), dim()));
    }
    app.push(Line::styled("applies on the next model call", dim()));
}

fn resolve_confirm_if_allowed(
    app: &mut App,
    live: &LiveSettings,
    pending: &mut Option<ConfirmationControl>,
) {
    let Some(control) = pending.take() else {
        return;
    };
    let mode = live.permission_mode();
    let path_ok = match &control.request {
        hi_harness::ConfirmationRequest::FileEdit { path, .. } => app.path_auto_approved(path),
        _ => false,
    };
    let allow = mode == PermissionMode::Always
        || (mode == PermissionMode::Auto && control.request.safe_for_auto())
        || path_ok;
    if allow {
        app.confirmation = None;
        let _ = control.response.send(ConfirmationResult::Approved);
    } else {
        *pending = Some(control);
    }
}

fn pipenetwork_login_arg(arg: &str) -> std::result::Result<(), String> {
    match arg.trim() {
        "" | "pipenetwork" | "pipe" => Ok(()),
        other => Err(format!(
            "'{other}' has no sign-in in this session. Use /login pipenetwork."
        )),
    }
}

fn apply_pipenetwork_session_key(
    app: &mut App,
    harness: &mut Harness,
    on_login: &mut Option<PipenetworkLoginHook>,
) {
    if let Some(token) = hi_ai::auth_store::load(hi_ai::pipenetwork_auth::PROVIDER_ID) {
        harness.set_api_key(token.access);
    }
    if let Some(hook) = on_login.as_mut() {
        match hook() {
            Ok(()) => {
                app.push(Line::styled(
                    "wrote [profiles.pipenetwork] to ~/.config/hi/config.toml",
                    dim(),
                ));
                app.follow();
            }
            Err(err) => {
                app.push(Line::styled(
                    format!("signed in, but couldn't update config.toml: {err:#}"),
                    dim(),
                ));
                app.follow();
            }
        }
    }
}

async fn drain_pending_pipenetwork_login(
    app: &mut App,
    harness: &mut Harness,
    on_login: &mut Option<PipenetworkLoginHook>,
) {
    if app.poll_pending_login().await.is_some() {
        apply_pipenetwork_session_key(app, harness, on_login);
    }
}

async fn start_pipenetwork_login(
    app: &mut App,
    harness: &mut Harness,
    arg: &str,
    on_login: &mut Option<PipenetworkLoginHook>,
) -> Result<()> {
    let _awaiting = harness.awaiting_user();
    if let Err(message) = pipenetwork_login_arg(arg) {
        app.push(Line::styled(message, dim()));
        app.follow();
        return Ok(());
    }
    if hi_ai::pipenetwork_auth::has_credential() {
        apply_pipenetwork_session_key(app, harness, on_login);
        app.push(Line::styled(
            "already signed in to pipenetwork — API key applied \
             (/logout pipenetwork first to pair a different account)",
            dim(),
        ));
        app.follow();
        return Ok(());
    }
    match hi_ai::pipenetwork_auth::request_pairing().await {
        Ok(issue) => {
            app.push(Line::styled(
                format!("open  {}", issue.url()),
                Style::default().add_modifier(Modifier::BOLD),
            ));
            app.push(Line::styled(
                format!("code  {}", issue.user_code),
                Style::default().add_modifier(Modifier::BOLD),
            ));
            app.push(Line::styled(
                "approve in your browser — hi will configure the API key when that lands",
                dim(),
            ));
            app.follow();
            let task = tokio::spawn(async move {
                let token = hi_ai::pipenetwork_auth::poll_for_key(&issue).await?;
                hi_ai::auth_store::save(hi_ai::pipenetwork_auth::PROVIDER_ID, &token)?;
                Ok(())
            });
            if let Some((_, previous)) = app.pending_login.replace(("pipenetwork".into(), task)) {
                previous.abort();
            }
        }
        Err(error) => {
            app.push(Line::styled(format!("/login failed: {error:#}"), dim()));
            app.follow();
        }
    }
    Ok(())
}

fn logout_pipenetwork(app: &mut App, harness: &mut Harness, arg: &str) {
    if let Err(message) = pipenetwork_login_arg(arg) {
        app.push(Line::styled(message, dim()));
        app.follow();
        return;
    }
    if app
        .pending_login
        .as_ref()
        .is_some_and(|(provider, _)| provider == "pipenetwork")
        && let Some((_, task)) = app.pending_login.take()
    {
        task.abort();
    }
    let message = match hi_ai::pipenetwork_auth::logout_quiet() {
        Ok(true) => {
            harness.set_api_key("");
            "signed out of pipenetwork".to_string()
        }
        Ok(false) => "not signed in to pipenetwork".to_string(),
        Err(error) => format!("/logout failed: {error:#}"),
    };
    app.push(Line::styled(message, dim()));
    app.follow();
}

#[allow(clippy::too_many_arguments)] // slash handler needs the live Restore so exec can drop it
async fn run_prompt(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    input_rx: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    harness: &mut Harness,
    on_login: &mut Option<PipenetworkLoginHook>,
    prompt: &str,
    ahf: &crate::autoharnessfix::AutoharnessfixOpts,
    restore: &mut Option<Restore>,
    termios: Option<&libc::termios>,
) -> Result<()> {
    run_turn(terminal, input_rx, ticker, app, harness, prompt, false).await?;
    while let Some(next) = app.queue.pop_front() {
        app.clamp_queue_selection();
        if let Some(command) = parse_command(&next) {
            if let Some(prompt) =
                handle_command(app, harness, command, on_login, ahf, restore, termios).await?
            {
                run_turn(terminal, input_rx, ticker, app, harness, &prompt, false).await?;
            }
        } else {
            run_turn(terminal, input_rx, ticker, app, harness, &next, false).await?;
        }
    }
    Ok(())
}

async fn run_turn(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    input_rx: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    harness: &mut Harness,
    prompt: &str,
    resume: bool,
) -> Result<()> {
    if resume {
        app.last_prompt = harness
            .messages()
            .last()
            .filter(|message| message.role == hi_ai::Role::User)
            .map(|message| message.text());
        app.last_turn_start = harness.messages().len();
    } else {
        app.push_user_prompt(Line::styled(
            format!("❯ {prompt}"),
            Style::default().fg(crate::theme::theme().accent_user),
        ));
        app.last_prompt = Some(prompt.to_string());
        app.last_turn_start = harness.messages().len();
    }
    app.set_working(true);
    let live = harness.live();
    let steer = harness.steer();
    app.interrupt = Some(harness.interrupt_handle());
    let cancel = TurnCancellation::new();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let (confirm_tx, mut confirm_rx) = mpsc::unbounded_channel();
    let mut sink = ChannelUi {
        tx: tx.clone(),
        confirmations: confirm_tx,
        event_sink: app.event_sink.clone(),
        approval_store: app.approval_store.clone(),
    };
    {
        let turn = async {
            if resume {
                harness
                    .resume_incomplete_turn(&mut sink, cancel.clone())
                    .await
                    .map(|_| ())
            } else {
                harness
                    .run_turn_cancellable(prompt, &mut sink, cancel.clone())
                    .await
                    .map(|_| ())
            }
        };
        let mut fut = std::pin::pin!(turn);
        let mut pending_confirm: Option<ConfirmationControl> = None;
        loop {
            terminal.draw(|frame| app.render(frame))?;
            tokio::select! {
                result = &mut fut => {
                    while let Ok(event) = rx.try_recv() {
                        app.apply(event);
                    }
                    if let Err(err) = result {
                        app.push(Line::styled(format!("{err:#}"), dim()));
                    }
                    break;
                }
                Some(event) = rx.recv() => {
                    app.apply(event);
                }
                Some(request) = confirm_rx.recv() => {
                    app.confirmation = Some(request.request.clone());
                    app.confirmation_scroll = 0;
                    app.confirmation_selected = 0;
                    pending_confirm = Some(request);
                    resolve_confirm_if_allowed(app, &live, &mut pending_confirm);
                }
                _ = ticker.tick() => {
                    app.spinner = app.spinner.wrapping_add(1);
                }
                maybe = input_rx.recv() => {
                    match maybe {
                        Some(Event::Paste(text)) => {
                            if crate::dashboard::is_open(app)
                                && let Some(overlay) = app.dashboard.as_mut()
                            {
                                overlay.draft.insert_str(&text);
                            } else {
                                app.paste_into_prompt(&text);
                            }
                        }
                        Some(Event::Mouse(mouse)) if app.mouse_capture => {
                            if let Some(overlay) = app.usage_overlay.as_mut() {
                                crate::usage::handle_click(overlay, mouse.column, mouse.row);
                            } else {
                                app.handle_mouse(mouse);
                            }
                        }
                        Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                            handle_running_key(
                                app,
                                &live,
                                &steer,
                                &key,
                                &cancel,
                                &mut pending_confirm,
                            );
                        }
                        Some(Event::Resize(..)) | None => {}
                        _ => {}
                    }
                }
            }
        }
    }
    app.set_working(false);
    app.interrupt = None;
    harness.persist_live_knobs();
    app.plan = harness.current_plan().to_vec();
    app.last_changed_files = harness.last_changed_files().to_vec();
    Ok(())
}
