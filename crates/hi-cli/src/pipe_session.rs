//! Default interactive/one-shot path: Pipe Network harness + TUI.

use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use hi_harness::{
    Command, ConfirmationRequest, ConfirmationResult, Harness, HarnessConfig, JsonlSession,
    LoadedSession, TurnCancellation, TurnOutcome, TurnStopReason, Ui, parse_command,
    parse_effort_arg, parse_model_args, resolve_model_query,
};

use crate::config::{
    Cli, Config, Profile, ProviderName, Settings, default_config_path, is_official_provider_url,
    read_config_file, resolve_api_key_for_endpoint,
};
use crate::paths;
use crate::session_files::resolve_session_path;

const PIPE_LOGIN_HINT: &str = "run `hi login pipenetwork` to sign in and configure the API key, \
     `hi auth pipenetwork` to paste a key, or set PIPENETWORK_API_KEY";

pub async fn run(
    cli: &Cli,
    file: &Config,
    settings: &Settings,
    workspace_root: PathBuf,
    state_root: PathBuf,
    prompt: Option<String>,
) -> Result<()> {
    let stdout_tty = std::io::stdout().is_terminal();
    let stdin_tty = std::io::stdin().is_terminal();
    let use_tui = !cli.plain && stdout_tty && stdin_tty;
    let resume = crate::sentinel_resume::plan(cli, prompt.as_ref());
    let prompt = if resume.ignore_prompt { None } else { prompt };
    let interactive = prompt.is_none() && stdin_tty;
    let route = match resolve_pipe_route(cli, file, settings) {
        Ok(route) => route,
        Err(err) if interactive => {
            eprintln!("{err:#}");
            eprintln!("sign in with /login pipenetwork (or run `hi login pipenetwork`)");
            PipeRoute {
                api_key: String::new(),
                base_url: hi_harness::DEFAULT_BASE_URL.to_string(),
                model: hi_harness::DEFAULT_MODEL.to_string(),
            }
        }
        Err(err) => return Err(err),
    };
    let mut config = HarnessConfig::pipe(workspace_root.clone(), route.api_key);
    config.state_root = state_root;
    config.model = route.model;
    config.base_url = route.base_url;
    if settings.provider == ProviderName::Pipenetwork && settings.max_tokens > 0 {
        config.max_tokens = settings.max_tokens;
    }
    config.reasoning_effort = settings.reasoning_effort;
    let session_path = resolve_session_path(cli)?;
    config.session_path = session_path.clone();
    let mut harness = Harness::new(config)?;
    if let Some(path) = &session_path
        && path.is_file()
        && let Ok(loaded) = JsonlSession::load(path)
    {
        apply_loaded(&mut harness, loaded);
    }
    if cli.no_verify {
        harness.set_verify_command(None);
    } else if !cli.verify.is_empty() {
        harness.set_verify_command(Some(cli.verify.join(" && ")));
    }
    // TUI keeps Ask; one-shot/`--plain` defaults to Always unless `--confirm-edits`.
    if !use_tui {
        if cli.confirm_edits {
            harness.set_permission_mode(hi_harness::PermissionMode::Ask);
        } else {
            harness.set_permission_mode(hi_harness::PermissionMode::Always);
        }
    }

    let login_config_path = cli.config.clone();
    harness.set_turn_intent_mode(!use_tui && prompt.is_some(), cli.plain);

    if resume.run_incomplete {
        let mut ui = StdoutUi {
            quiet: cli.quiet,
            confirm_edits: cli.confirm_edits,
            ..StdoutUi::default()
        };
        if let Some(outcome) = harness
            .resume_incomplete_turn(&mut ui, TurnCancellation::new())
            .await?
        {
            if let Some(path) = &cli.report {
                write_turn_report(path, &outcome, &ui, &harness)?;
            }
            if let Some(error) = outcome.error {
                bail!(error);
            }
            if resume.exit_after_resume {
                return Ok(());
            }
        } else if resume.exit_after_resume {
            return Ok(());
        }
    }

    if use_tui {
        let model = harness.model().to_string();
        return hi_tui::run_session(
            &mut harness,
            hi_tui::SessionOptions {
                provider: "pipenetwork".into(),
                model,
                history_path: paths::history_path(),
                startup_prompt: resume.startup_prompt,
                on_pipenetwork_login: Some(Box::new(move || {
                    write_login_profile(login_config_path.as_deref())
                })),
                list_sessions: Some(Box::new(|| {
                    paths::session_summaries()
                        .into_iter()
                        .map(|summary| hi_tui::LocalSessionInfo {
                            id: summary.id,
                            title: String::new(),
                            age: summary.age,
                            lines: 0,
                        })
                        .collect()
                })),
                openai_api_key: optional_openai_key(file),
                openai_base_url: optional_openai_base(file),
                session_path: session_path.clone(),
                no_save: cli.no_save,
                sentinel_blocked: crate::sentinel_exec::rsi_off_limits(cli)
                    .then_some("Sentinel cannot wrap RSI".into()),
            },
        )
        .await;
    }

    if let Some(prompt) = prompt {
        let mut ui = StdoutUi {
            quiet: cli.quiet,
            confirm_edits: cli.confirm_edits,
            ..StdoutUi::default()
        };
        let outcome = harness
            .run_turn_cancellable(&prompt, &mut ui, TurnCancellation::new())
            .await?;
        if let Some(path) = &cli.report {
            write_turn_report(path, &outcome, &ui, &harness)?;
        }
        if let Some(error) = outcome.error {
            bail!(error);
        }
        return Ok(());
    }

    use rustyline::Editor;
    use rustyline::history::DefaultHistory;
    let mut editor = Editor::<(), DefaultHistory>::new()?;
    let mut last_prompt: Option<String> = None;
    let mut last_turn_start = 0usize;
    println!(
        "hi · pipenetwork · {} — /login, /help, Ctrl-D to quit.",
        harness.model()
    );
    while let Ok(line) = editor.readline("❯ ") {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        if matches!(line.as_str(), "/quit" | "/exit" | "/q") {
            break;
        }
        editor.add_history_entry(&line)?;
        let turn_line;
        if let Some(command) = parse_command(&line) {
            match command {
                Command::Quit => break,
                Command::Help(_) => {
                    println!(
                        "/login /logout /auth /model /effort /permissions /undo /diff /retry /verify /compact /files /status /usage /dashboard /doctor /autoharnessfix /sessions /rewind /exit"
                    );
                    continue;
                }
                Command::Login(arg) => {
                    match pipenetwork_login_arg(&arg) {
                        Ok(()) => match run_repl_login(&mut harness, login_config_path.as_deref())
                            .await
                        {
                            Ok(path) => {
                                println!("signed in — configured pipenetwork in {}", path.display())
                            }
                            Err(err) => eprintln!("/login failed: {err:#}"),
                        },
                        Err(message) => eprintln!("{message}"),
                    }
                    continue;
                }
                Command::Logout(arg) => {
                    match pipenetwork_login_arg(&arg) {
                        Ok(()) => match hi_ai::pipenetwork_auth::logout() {
                            Ok(()) => harness.set_api_key(""),
                            Err(err) => eprintln!("/logout failed: {err:#}"),
                        },
                        Err(message) => eprintln!("{message}"),
                    }
                    continue;
                }
                Command::Model(arg) => {
                    handle_repl_model(&mut harness, &arg).await;
                    continue;
                }
                Command::Effort(arg) => {
                    handle_repl_effort(&mut harness, &arg);
                    continue;
                }
                Command::Permissions(arg) => {
                    handle_repl_permissions(&mut harness, &arg);
                    continue;
                }
                Command::Undo => {
                    match harness.undo().await {
                        Ok(Some(n)) => println!("restored {n} path(s)"),
                        Ok(None) => println!("nothing to undo"),
                        Err(err) => eprintln!("undo failed: {err:#}"),
                    }
                    continue;
                }
                Command::Diff => {
                    let diff = hi_tools::working_tree_diff_in(harness.workspace_root()).await;
                    if diff.trim().is_empty() {
                        println!("working tree clean");
                    } else {
                        println!("{diff}");
                    }
                    continue;
                }
                Command::Retry => {
                    let Some(prompt) = last_prompt.clone() else {
                        eprintln!("nothing to retry");
                        continue;
                    };
                    harness.truncate_messages(last_turn_start);
                    turn_line = Some(prompt);
                }
                Command::Verify(arg) => {
                    if arg.is_empty() {
                        println!("verify: {}", harness.verify_command().unwrap_or("off"));
                    } else {
                        harness.set_verify_command(Some(arg.clone()));
                        println!("verify: {}", harness.verify_command().unwrap_or("off"));
                    }
                    continue;
                }
                Command::Compact(arg) => {
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
                        Ok(true) => println!("compacted conversation"),
                        Ok(false) => println!("nothing to compact"),
                        Err(err) => eprintln!("compact failed: {err:#}"),
                    }
                    continue;
                }
                Command::Files => {
                    let files = harness.last_changed_files();
                    if files.is_empty() {
                        println!("no files changed this session");
                    } else {
                        for file in files {
                            println!("{file}");
                        }
                    }
                    continue;
                }
                Command::Status => {
                    println!("{}", repl_status(&harness));
                    continue;
                }
                Command::Usage(arg) => {
                    if arg.trim().eq_ignore_ascii_case("manage") {
                        match hi_harness::open_billing_url() {
                            Ok(()) => println!("opened {}", hi_harness::BILLING_URL),
                            Err(err) => eprintln!("{err} — {}", hi_harness::BILLING_URL),
                        }
                        continue;
                    }
                    let Some(tab) = hi_harness::UsageTab::from_arg(&arg) else {
                        eprintln!(
                            "use /usage  or  /usage show|manage  (alias /cost; /context for the context tab)"
                        );
                        continue;
                    };
                    let snap = harness.usage_snapshot();
                    println!("{}", tab.title());
                    println!("{}", snap.tab_text(tab));
                    continue;
                }
                Command::Doctor => {
                    println!("{}", harness.doctor_report().await);
                    continue;
                }
                Command::AutoHarnessFix(arg) => {
                    let blocked = crate::sentinel_exec::rsi_off_limits(cli)
                        .then_some("Sentinel cannot wrap RSI");
                    match hi_sentinel::dispatch_slash(
                        &arg,
                        session_path.as_deref(),
                        cli.no_save,
                        blocked,
                        harness.workspace_root(),
                    ) {
                        Ok(hi_sentinel::SlashOutcome::Message(text)) => println!("{text}"),
                        Ok(hi_sentinel::SlashOutcome::ExecSupervisor { session_file }) => {
                            println!("{}", hi_sentinel::restart_line());
                            let err = hi_sentinel::exec_supervisor(&session_file);
                            return Err(err.into());
                        }
                        Ok(hi_sentinel::SlashOutcome::ExecRepair { incident }) => {
                            println!("{}", hi_sentinel::restart_line());
                            let err = hi_sentinel::exec_repair(&incident);
                            return Err(err.into());
                        }
                        Err(err) => eprintln!("{err:#}"),
                    }
                    continue;
                }
                Command::Version => {
                    println!("hi {}", hi_harness::Harness::version());
                    continue;
                }
                Command::Clear => {
                    harness.clear_history();
                    println!("conversation cleared");
                    continue;
                }
                Command::Rewind(arg) => {
                    match arg.trim().parse::<usize>() {
                        Ok(n) => match harness.rewind_to_user_turn(n) {
                            Ok(()) => println!("rewound to user turn {n}"),
                            Err(err) => eprintln!("{err:#}"),
                        },
                        Err(_) => eprintln!("use /rewind <n>"),
                    }
                    continue;
                }
                Command::Sessions(_) => {
                    let summaries = paths::session_summaries();
                    if summaries.is_empty() {
                        println!("no sessions — `hi resume` starts from the latest next time");
                    } else {
                        for summary in summaries.iter().take(20) {
                            println!("{}  {:>6} ago", summary.id, summary.age);
                        }
                        println!("resume with `hi --resume <id>`");
                    }
                    continue;
                }
                Command::Auth(arg) => {
                    match parse_repl_auth_key(&arg) {
                        Ok(key) => {
                            match apply_repl_auth(&mut harness, login_config_path.as_deref(), &key)
                                .await
                            {
                                Ok(()) => println!("api key stored"),
                                Err(err) => eprintln!("/auth failed: {err:#}"),
                            }
                        }
                        Err(message) => eprintln!("{message}"),
                    }
                    continue;
                }
                Command::Unknown(msg) | Command::Removed(msg) => {
                    eprintln!("{msg}");
                    continue;
                }
                Command::Tutorial => {
                    eprintln!("/tutorial is available in the full-screen TUI");
                    continue;
                }
                Command::Dashboard => {
                    eprintln!(
                        "open the TUI (`hi`) and type /dashboard (aliases /fleet, /agents-dashboard)"
                    );
                    continue;
                }
                _ => {
                    eprintln!("command not available in this frontend");
                    continue;
                }
            }
        } else {
            turn_line = Some(line);
        }
        let Some(prompt) = turn_line else {
            continue;
        };
        last_turn_start = harness.messages().len();
        last_prompt = Some(prompt.clone());
        let mut ui = StdoutUi {
            quiet: cli.quiet,
            confirm_edits: cli.confirm_edits,
            ..StdoutUi::default()
        };
        if let Err(err) = harness
            .run_turn_cancellable(&prompt, &mut ui, TurnCancellation::new())
            .await
        {
            eprintln!("{err:#}");
        }
    }
    Ok(())
}

fn apply_loaded(harness: &mut Harness, loaded: LoadedSession) {
    harness.apply_loaded_session(loaded);
}

/// Optional OpenAI profile from config. Absence is not an error: dashboard
/// gpt-6 rows stay on Pipe Network.
fn optional_openai_key(file: &Config) -> Option<String> {
    for profile in file.profiles.values() {
        if profile.provider != Some(ProviderName::Openai) {
            continue;
        }
        if let Some(key) = profile
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return Some(key.to_string());
        }
        if let Some(name) = profile.api_key_env.as_deref()
            && let Ok(key) = std::env::var(name)
        {
            let key = key.trim().to_string();
            if !key.is_empty() {
                return Some(key);
            }
        }
    }
    None
}

fn optional_openai_base(file: &Config) -> Option<String> {
    file.profiles.values().find_map(|profile| {
        (profile.provider == Some(ProviderName::Openai))
            .then(|| profile.base_url.clone())
            .flatten()
            .filter(|s| !s.trim().is_empty())
    })
}

fn repl_status(harness: &Harness) -> String {
    let usage = harness.session_usage();
    let occupancy = usage.context_occupancy.max(usage.input_tokens);
    let sandbox = if harness.sandbox_enforced() {
        harness.sandbox_backend_name()
    } else {
        "off"
    };
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
    )
}

fn parse_repl_auth_key(arg: &str) -> std::result::Result<String, String> {
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
        "pipenetwork" | "pipe" => {
            Err("usage: /auth pipenetwork <api-key>  (or `hi auth pipenetwork`)".into())
        }
        key if !key.is_empty() && rest.is_empty() => Ok(key.to_string()),
        other => Err(format!(
            "'{other}' has no pasted-key flow here. Use /auth pipenetwork <key>."
        )),
    }
}

async fn apply_repl_auth(
    harness: &mut Harness,
    explicit: Option<&std::path::Path>,
    key: &str,
) -> Result<()> {
    let _awaiting = harness.awaiting_user();
    let path = match explicit {
        Some(path) => path.to_path_buf(),
        None => default_config_path().context("could not determine config directory")?,
    };
    let mut config = if path.exists() {
        read_config_file(&path)?
    } else {
        Config::default()
    };
    match crate::auth::apply_pasted_key(&mut config, ProviderName::Pipenetwork, key, None, &path)
        .await?
    {
        hi_ai::KeyCheck::Accepted | hi_ai::KeyCheck::Unverified(_) => {
            harness.set_api_key(key);
            Ok(())
        }
        hi_ai::KeyCheck::Rejected(msg) => bail!("key rejected: {msg}"),
    }
}

fn repl_model_status(harness: &Harness) -> String {
    match harness.reasoning_effort() {
        Some(effort) => format!(
            "model: {} · reasoning: {}",
            harness.model(),
            effort.as_str()
        ),
        None => format!("model: {} · reasoning: off", harness.model()),
    }
}

async fn handle_repl_model(harness: &mut Harness, arg: &str) {
    let parsed = parse_model_args(arg);
    if parsed.model.is_none() && parsed.effort.is_none() {
        match harness.list_models().await {
            Ok(models) => {
                for model in &models {
                    let mark = if model.id == harness.model() {
                        "*"
                    } else {
                        " "
                    };
                    println!("{mark} {}", model.id);
                }
                if models.is_empty() {
                    println!("(no models from {})", harness.base_url());
                }
                println!("{}", repl_model_status(harness));
            }
            Err(err) => {
                eprintln!("could not list models: {err:#}");
                println!("{}", repl_model_status(harness));
            }
        }
        return;
    }
    if let Some(model) = parsed.model {
        let ids = match harness.list_models().await {
            Ok(models) => models.into_iter().map(|model| model.id).collect::<Vec<_>>(),
            Err(_) => Vec::new(),
        };
        harness.set_model(resolve_model_query(&model, &ids));
    }
    if let Some(effort) = parsed.effort {
        harness.apply_effort_arg(effort);
    }
    println!("{}", repl_model_status(harness));
}

fn handle_repl_permissions(harness: &mut Harness, arg: &str) {
    let current = harness.permission_mode();
    let next = match arg.trim().to_ascii_lowercase().as_str() {
        "" => {
            println!("permissions: {}", current.describe());
            return;
        }
        "toggle-always" => current.toggle_always(),
        "toggle-auto" => current.toggle_auto(),
        other => match hi_harness::PermissionMode::from_arg(other) {
            Some(mode) => mode,
            None => {
                eprintln!("use /permissions ask|auto|always  (or /auto, /yolo)");
                return;
            }
        },
    };
    harness.set_permission_mode(next);
    println!("permissions: {}", next.describe());
}

fn handle_repl_effort(harness: &mut Harness, arg: &str) {
    if arg.trim().is_empty() {
        println!("{}", repl_model_status(harness));
        return;
    }
    match parse_effort_arg(arg) {
        Ok(effort) => {
            harness.apply_effort_arg(effort);
            println!("{}", repl_model_status(harness));
        }
        Err(message) => eprintln!("{message}"),
    }
}

#[derive(Debug)]
struct PipeRoute {
    api_key: String,
    base_url: String,
    model: String,
}

fn resolve_pipe_route(cli: &Cli, config: &Config, settings: &Settings) -> Result<PipeRoute> {
    let profile = pipenetwork_profile(config);
    let mut model = if let Some(model) = cli
        .model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        model.to_string()
    } else if settings.provider == ProviderName::Pipenetwork
        && !settings.model.trim().is_empty()
        && settings.model != "__model_not_configured__"
    {
        settings.model.clone()
    } else if let Some(model) = profile
        .and_then(|profile| profile.model.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        model.to_string()
    } else {
        hi_harness::DEFAULT_MODEL.to_string()
    };
    hi_provider_config::apply_stale_pipenetwork_default_model(
        hi_provider_config::ProviderName::Pipenetwork,
        &mut model,
    );

    let base_url = if let Some(url) = cli
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        url.trim_end_matches('/').to_string()
    } else if settings.provider == ProviderName::Pipenetwork
        && is_official_provider_url(ProviderName::Pipenetwork, &settings.base_url)
    {
        settings.base_url.trim_end_matches('/').to_string()
    } else if let Some(url) = profile
        .and_then(|profile| profile.base_url.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        url.trim_end_matches('/').to_string()
    } else {
        hi_harness::DEFAULT_BASE_URL.to_string()
    };

    let api_key = if let Some(key) = cli
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        key.to_string()
    } else if settings.provider == ProviderName::Pipenetwork && !settings.api_key.trim().is_empty()
    {
        settings.api_key.clone()
    } else {
        resolve_api_key_for_endpoint(
            profile,
            ProviderName::Pipenetwork,
            &base_url,
            true,
            profile.is_some_and(|profile| profile.project_local),
        )
        .map_err(|err| {
            let text = format!("{err:#}");
            if text.contains("auth-store://pipenetwork")
                || text.contains("no pipenetwork credential")
            {
                anyhow!("{text}\n{PIPE_LOGIN_HINT}")
            } else {
                err
            }
        })?
    };
    if api_key.trim().is_empty() {
        bail!("{PIPE_LOGIN_HINT}");
    }

    Ok(PipeRoute {
        api_key,
        base_url,
        model,
    })
}

fn pipenetwork_profile(config: &Config) -> Option<&Profile> {
    for name in ["pipenetwork", "pipe"] {
        if let Some(profile) = config.profiles.get(name)
            && profile.provider.unwrap_or(ProviderName::Pipenetwork) == ProviderName::Pipenetwork
        {
            return Some(profile);
        }
    }
    if let Some(name) = config.default_profile.as_deref()
        && let Some(profile) = config.profiles.get(name)
        && profile.provider == Some(ProviderName::Pipenetwork)
    {
        return Some(profile);
    }
    let mut names: Vec<&str> = config
        .profiles
        .iter()
        .filter(|(_, profile)| profile.provider == Some(ProviderName::Pipenetwork))
        .map(|(name, _)| name.as_str())
        .collect();
    names.sort_unstable();
    names
        .first()
        .copied()
        .and_then(|name| config.profiles.get(name))
}

fn write_login_profile(explicit: Option<&std::path::Path>) -> Result<()> {
    let path = match explicit {
        Some(path) => path.to_path_buf(),
        None => default_config_path().context("could not determine config directory")?,
    };
    let mut config = if path.exists() {
        read_config_file(&path)?
    } else {
        Config::default()
    };
    crate::auth::install_pipenetwork_login_profile_at(&mut config, &path)
}

fn pipenetwork_login_arg(arg: &str) -> std::result::Result<(), String> {
    match arg.trim() {
        "" | "pipenetwork" | "pipe" => Ok(()),
        other => Err(format!(
            "'{other}' has no sign-in in this session. Use /login pipenetwork."
        )),
    }
}

async fn run_repl_login(
    harness: &mut Harness,
    explicit: Option<&std::path::Path>,
) -> Result<PathBuf> {
    let _awaiting = harness.awaiting_user();
    hi_ai::pipenetwork_auth::login().await?;
    let token = hi_ai::auth_store::load(hi_ai::pipenetwork_auth::PROVIDER_ID)
        .context("sign-in reported success but stored no credential")?;
    harness.set_api_key(token.access);
    let path = match explicit {
        Some(path) => path.to_path_buf(),
        None => default_config_path().context("could not determine config directory")?,
    };
    let mut config = if path.exists() {
        read_config_file(&path)?
    } else {
        Config::default()
    };
    crate::auth::install_pipenetwork_login_profile_at(&mut config, &path)?;
    Ok(path)
}

#[derive(Default)]
struct StdoutUi {
    assistant: String,
    tools: Vec<String>,
    tool_calls: Vec<serde_json::Value>,
    quiet: bool,
    confirm_edits: bool,
}

fn clip_report_text(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect::<String>() + "…"
}

impl Ui for StdoutUi {
    fn assistant_text(&mut self, text: &str) {
        print!("{text}");
        self.assistant.push_str(text);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    fn assistant_reasoning(&mut self, _text: &str) {}
    fn assistant_end(&mut self) {
        println!();
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.tools.push(name.to_string());
        self.tool_calls.push(serde_json::json!({
            "name": name,
            "arguments": clip_report_text(arguments, 2_000),
            "output": serde_json::Value::Null,
        }));
        if !self.quiet {
            println!("→ {name} {arguments}");
        }
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        if let Some(last) = self
            .tool_calls
            .iter_mut()
            .rev()
            .find(|call| call.get("name").and_then(|value| value.as_str()) == Some(name))
        {
            last["output"] = serde_json::Value::String(clip_report_text(result, 4_000));
        }
        if self.quiet {
            return;
        }
        let preview: String = result.chars().take(400).collect();
        println!("← {name}\n{preview}");
    }
    fn status(&mut self, text: &str) {
        if !self.quiet {
            eprintln!("{text}");
        }
    }
    fn turn_end(&mut self, _summary: &str) {}
    fn turn_error(&mut self, error_kind: &str, message: &str, guidance: &str) {
        eprintln!("{error_kind}: {message}\n{guidance}");
    }
    fn confirm(&mut self, request: ConfirmationRequest) -> hi_harness::ConfirmationFuture<'_> {
        if !self.confirm_edits {
            return Box::pin(async { ConfirmationResult::Approved });
        }
        let title = request.title();
        let details = request.details();
        Box::pin(async move {
            eprintln!("{title}\n{details}\napply? [y/N] ");
            let mut line = String::new();
            match std::io::stdin().read_line(&mut line) {
                Ok(_) if matches!(line.trim(), "y" | "Y" | "yes") => ConfirmationResult::Approved,
                _ => ConfirmationResult::Rejected,
            }
        })
    }
}

fn write_turn_report(
    path: &std::path::Path,
    outcome: &TurnOutcome,
    ui: &StdoutUi,
    harness: &Harness,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let status = match outcome.stop_reason {
        TurnStopReason::Completed => "completed",
        TurnStopReason::Cancelled => "cancelled",
        TurnStopReason::Error => "failed",
    };
    let model_requests = harness
        .messages()
        .iter()
        .filter(|message| message.role == hi_ai::Role::Assistant)
        .count() as u64;
    let body = serde_json::json!({
        "assistant_response": ui.assistant,
        "tools": if ui.tool_calls.is_empty() {
            ui.tools
                .iter()
                .map(|name| serde_json::json!({ "name": name }))
                .collect::<Vec<_>>()
        } else {
            ui.tool_calls.clone()
        },
        "outcome": {
            "status": status,
            "stop_reason": status,
            "changed_files": outcome.changed_files,
            "verification": outcome
                .verification
                .clone()
                .unwrap_or_else(|| "not_applicable".into()),
        },
        "model_outcome": {
            "model_requests": model_requests,
        }
    });
    std::fs::write(path, serde_json::to_vec_pretty(&body)?)
        .with_context(|| format!("writing report {}", path.display()))
}

#[cfg(test)]
#[path = "pipe_session_tests.rs"]
mod tests;
