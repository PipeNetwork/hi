mod bestof;
mod commands;
mod complete;
mod config;
mod delegate;
mod feedback;
mod repl;
mod session;
mod setup;
mod ui;

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::Parser;

use hi_agent::{Agent, AgentConfig, CompactionKind, VerifyStage};
use hi_ai::{
    AnthropicProvider, Backend, FallbackProvider, McpDiscoveryProvider, Message, MoaProvider,
    OpenAiProvider, PipeMcpClient, Provider, Usage,
};

use commands::tool_mode_label;
use config::{Cli, ProviderName, Settings};
use repl::repl;
use session::JsonlSession;
use ui::PlainUi;

#[tokio::main]
async fn main() -> Result<()> {
    let raw_args = std::env::args().collect::<Vec<_>>();
    if raw_args.get(1).map(String::as_str) == Some("hf") {
        return run_hf_cli(&raw_args[2..]).await;
    }

    let cli = Cli::parse();

    if cli.show_config {
        let file = config::load_config(cli.config.as_deref())?;
        match config::resolve(&cli, &file) {
            Ok(settings) => {
                let live = if settings.provider == ProviderName::Pipenetwork {
                    let provider = build_provider(&settings);
                    resolve_live_model_metadata(provider.as_ref(), &settings.model).await
                } else {
                    LiveModelMetadata {
                        context_window: None,
                        max_output_tokens: None,
                    }
                };
                let effective_max_tokens =
                    effective_max_tokens_for_model(&settings, live.max_output_tokens);
                println!("provider:   {}", provider_label(settings.provider));
                println!("model:      {}", settings.model);
                println!("base_url:   {}", settings.base_url);
                if let Some(mcp_url) = &settings.mcp_url {
                    println!("mcp_url:    {mcp_url}");
                }
                println!("max_tokens: {}", effective_max_tokens);
                if let Some(limit) = live.max_output_tokens {
                    println!("model_max_output_tokens: {limit}");
                }
                println!(
                    "thinking:   {}",
                    settings
                        .thinking_budget
                        .map(|b| b.to_string())
                        .unwrap_or_else(|| "off".into())
                );
                println!("tool_mode:  {:?}", settings.tool_mode);
                println!("compat:     {:?}", settings.compat);
                let api_key_display = if settings.api_key.len() > 8 {
                    format!(
                        "{}...{}",
                        &settings.api_key[..4],
                        &settings.api_key[settings.api_key.len() - 4..]
                    )
                } else {
                    "***".to_string()
                };
                println!("api_key:    {}", api_key_display);
                return Ok(());
            }
            Err(err) => {
                eprintln!("{err}");
                std::process::exit(2);
            }
        }
    }

    if cli.list_sessions {
        return session::list_sessions();
    }

    let mut file = config::load_config(cli.config.as_deref())?;

    // First run on a real terminal with nothing configured: walk the user
    // through an interactive setup instead of erroring.
    let settings = if cli.prompt.is_none()
        && config::needs_setup(&cli, &file)
        && std::io::stdin().is_terminal()
    {
        setup::run().await?
    } else {
        // Otherwise print config/onboarding guidance plainly (no "Error:" prefix).
        match config::resolve(&cli, &file) {
            Ok(settings) => settings,
            Err(err) => {
                eprintln!("{err}");
                std::process::exit(2);
            }
        }
    };

    if cli.prompt.as_deref() == Some("mcp") {
        return run_mcp_command(&settings).await;
    }

    // Fold piped stdin into the one-shot prompt as context.
    let prompt_input = effective_prompt(&cli)?;
    let report_path = cli
        .report
        .as_ref()
        .map(|path| absolutize_path(path.as_path()))
        .transpose()?;
    if let Some(prompt) = prompt_input.as_deref() {
        maybe_chdir_to_prompt_review_target(prompt)?;
    }

    if cli.best_of > 1 {
        let prompt = prompt_input
            .as_deref()
            .ok_or_else(|| anyhow!("--best-of requires a one-shot prompt"))?;
        let verify = pipeline_command(&resolve_verify(&cli))
            .ok_or_else(|| anyhow!("--best-of requires --verify or --auto-verify"))?;
        let exe = std::env::current_exe().context("locating the hi executable")?;
        return bestof::run(&bestof::BestOf {
            exe: &exe,
            provider: provider_label(settings.provider),
            model: &settings.model,
            base_url: &settings.base_url,
            api_key: &settings.api_key,
            verify: &verify,
            prompt,
            candidates: cli.best_of,
            max_steps: cli.max_steps,
            max_verify: cli.max_verify,
        });
    }

    // Resolve which session file to use and any history to resume.
    let (session_path, loaded) = resolve_session(&cli)?;
    let feedback_session_id = feedback::session_id_from_path(&session_path);

    let fallbacks = config::resolve_fallbacks(&cli, &file);
    // Arc so the agent can share it with read-only `explore` subagents.
    let provider: std::sync::Arc<dyn Provider> = build_chain(&settings, fallbacks).into();
    let live_metadata = if settings.provider == ProviderName::Pipenetwork {
        resolve_live_model_metadata(provider.as_ref(), &settings.model).await
    } else {
        LiveModelMetadata {
            context_window: None,
            max_output_tokens: None,
        }
    };
    let max_tokens = effective_max_tokens_for_model(&settings, live_metadata.max_output_tokens);
    // The goal planner (glm-5.2 on pipenetwork by default). `HI_PLANNER_MODEL`
    // overrides the profile; long-horizon goals turn on wherever a planner exists
    // (never for a subagent — it gets a direct task, not a goal).
    let planner_model = std::env::var("HI_PLANNER_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| settings.planner_model.clone());
    let agent_config = AgentConfig {
        model: settings.model.clone(),
        requested_max_tokens: settings.max_tokens,
        max_tokens,
        max_tokens_explicit: settings.max_tokens_explicit,
        temperature: cli.temperature,
        thinking_budget: settings.thinking_budget,
        tool_mode: settings.tool_mode,
        compat: settings.compat,
        minimal_tools: settings.minimal_tools,
        // Env override lets you flip on skill auto-curation without editing a profile.
        curate_skills: settings.curate_skills || std::env::var_os("HI_CURATE_SKILLS").is_some(),
        explore_subagents: settings.explore_subagents
            || std::env::var_os("HI_EXPLORE_SUBAGENTS").is_some(),
        write_subagents: settings.write_subagents
            || std::env::var_os("HI_WRITE_SUBAGENTS").is_some(),
        // `--subagent` marks a delegate child: no explore/delegate offered (depth ≤ 1).
        is_subagent: cli.subagent,
        context_window: live_metadata.context_window,
        project_context: load_project_context(),
        verify: resolve_verify(&cli),
        max_verify_iterations: cli.max_verify,
        max_steps: cli
            .max_steps
            .unwrap_or_else(|| AgentConfig::default().max_steps),
        max_steps_explicit: cli.max_steps.is_some(),
        auto_compact: !cli.no_auto_compact,
        compaction: cli
            .compaction
            .as_deref()
            .and_then(CompactionKind::from_arg)
            .unwrap_or(CompactionKind::Hybrid {
                keep_recent: hi_agent::DEFAULT_KEEP_RECENT,
            }),
        finalize: !cli.no_finalize,
        confirm_edits: cli.confirm_edits,
        planner_model: planner_model.clone(),
        // `--goal` always steers, even off-pipenetwork (single sub-goal fallback).
        long_horizon: !cli.subagent
            && (planner_model.is_some()
                || std::env::var_os("HI_LONG_HORIZON").is_some()
                || cli.goal.is_some()),
        ..AgentConfig::default()
    };
    let resume_summary = loaded.as_ref().and_then(|l| l.resume_summary.clone());
    let mut agent = match loaded {
        Some(loaded) => Agent::resume(
            provider,
            agent_config,
            loaded.messages,
            loaded.usage,
            loaded.checkpoint_refs,
            loaded.structured_goal,
            loaded.decisions,
        ),
        None => Agent::new(provider, agent_config),
    };
    // Attach the write-`delegate` subagent runner for any top-level agent (a
    // subagent can't delegate), regardless of whether write subagents start on —
    // so `/delegate on` can enable it at runtime. The tool stays gated by the
    // `write_subagents` advertisement; the runner just needs to be ready. It spawns
    // a `hi --subagent` child in an isolated worktree and applies only verified diffs.
    let delegate_runner: Option<std::sync::Arc<dyn hi_agent::DelegateRunner>> = if !cli.subagent
        && let Ok(exe) = std::env::current_exe()
    {
        let runner = delegate::CliDelegateRunner::new(
            exe,
            provider_label(settings.provider).to_string(),
            settings.model.clone(),
            settings.base_url.clone(),
            settings.api_key.clone(),
            pipeline_command(&resolve_verify(&cli)),
            cli.max_steps.unwrap_or(60),
            cli.max_verify,
        );
        Some(std::sync::Arc::new(runner))
    } else {
        None
    };
    if let Some(runner) = &delegate_runner {
        agent.set_delegate_runner(runner.clone());
    }
    if !cli.no_save && !cli.subagent {
        agent.set_session(Box::new(JsonlSession::new(session_path)));
    }
    // The fleet launcher: how `/dashboard` spawns worktree-isolated child `hi`
    // runs (one per row turn), each appending to a parent-owned session file.
    let fleet_launcher = hi_tui::FleetLauncher {
        exe: std::env::current_exe().unwrap_or_else(|_| PathBuf::from("hi")),
        provider: provider_label(settings.provider).to_string(),
        model: settings.model.clone(),
        base_url: settings.base_url.clone(),
        api_key: settings.api_key.clone(),
        verify: pipeline_command(&resolve_verify(&cli)),
        max_verify: cli.max_verify,
        max_steps: cli.max_steps.unwrap_or(60),
        session_path: Box::new(session::new_fleet_session_path),
        sessions: Box::new(|| {
            session::fleet_sessions()
                .into_iter()
                .map(|s| hi_tui::FleetSessionInfo {
                    id: s.id,
                    title: s.title,
                    age: s.age,
                    lines: s.lines,
                })
                .collect()
        }),
        resume_info: Box::new(|id| {
            let id = if id.is_empty() {
                // No id: the most recent fleet session in this project.
                session::fleet_sessions().into_iter().next()?.id
            } else {
                id.to_string()
            };
            let path = session::session_path(&id).ok().filter(|p| p.is_file())?;
            let title = session::fleet_sessions()
                .into_iter()
                .find(|s| s.id == id)
                .map(|s| s.title)
                .unwrap_or_else(|| id.clone());
            let goal = session::session_goal_summary(&path);
            Some(hi_tui::FleetResumeInfo {
                id,
                path,
                title,
                goal_active: goal.as_ref().is_some_and(|g| g.active),
                goal_done: goal.as_ref().map(|g| g.done).unwrap_or(0),
                goal_total: goal.as_ref().map(|g| g.total).unwrap_or(0),
            })
        }),
        loop_session_path: Box::new(session::new_loop_session_path),
        loops_file: session::loops_file(),
    };

    if let Some(mut prompt) = prompt_input {
        let mut restore_model_state: Option<hi_agent::AgentModelState> = None;
        let mut report_model = settings.model.clone();
        if let Some(hi_agent::Command::Moa(arg)) = hi_agent::command::parse(&prompt) {
            let arg = arg.trim().to_string();
            if arg.is_empty() {
                return Err(anyhow!("usage: /moa <prompt>"));
            }
            restore_model_state = Some(agent.model_state());
            agent.set_model(hi_ai::MOA_MODEL_CONSERVATIVE.to_string(), None, None);
            report_model = hi_ai::MOA_MODEL_CONSERVATIVE.to_string();
            prompt = arg;
        }
        // `--goal <objective>` (fleet rows): install a planner-decomposed goal
        // before the turn — but never re-plan when the resumed session already
        // carries one (later fleet turns drive the existing goal).
        if let Some(objective) = cli.goal.as_deref().map(str::trim).filter(|s| !s.is_empty())
            && agent.structured_goal().is_none()
        {
            if !cli.quiet {
                println!("\x1b[2mplanning goal with the planner model…\x1b[0m");
            }
            let steps = match agent.decompose_goal(objective).await {
                Ok(steps) if !steps.is_empty() => steps,
                _ => vec![objective.to_string()],
            };
            let goal = hi_agent::Goal::new(objective.to_string(), steps);
            if agent.set_structured_goal(Some(goal)).unwrap_or(false)
                && !cli.quiet
                && let Some(g) = agent.structured_goal()
            {
                println!(
                    "\x1b[2m✓ goal set — {} sub-goal(s)\x1b[0m",
                    g.sub_goals.len()
                );
            }
        }
        let mut plain = PlainUi::new();
        let mut quiet = ui::QuietUi;
        let view: &mut dyn hi_agent::Ui = if cli.quiet { &mut quiet } else { &mut plain };
        let result = agent.run_turn(&prompt, view).await;
        if let Some(state) = restore_model_state {
            agent.restore_model_state(state);
        }
        let report_result = if let Some(path) = &report_path {
            write_report(
                path,
                &agent,
                &report_model,
                result.as_ref().err(),
            )
        } else {
            Ok(())
        };
        if let Err(err) = &result {
            let (kind, guidance) = hi_agent::classify_error(err);
            let suffix = if guidance.is_empty() {
                String::new()
            } else {
                format!(" — {guidance}")
            };
            eprintln!("\x1b[31m{kind}: {err:#}{suffix}\x1b[0m");
        }
        if let Err(err) = &report_result {
            eprintln!("\x1b[33mreport error: {err:#}\x1b[0m");
        }
        // A one-shot turn may have started background processes; don't leak them.
        hi_tools::kill_background_processes();
        if result.is_ok() {
            report_result?;
        }
        return result;
    }

    // Auto-memory at the end of an interactive session (TUI or REPL), unless
    // disabled or the session isn't being saved (memory is a form of persistence).
    // One-shot prompts return above, so scripted/piped/eval runs never write it.
    let auto_memory = auto_memory_enabled(cli.no_memory, cli.no_save);

    let stdout_is_tty = std::io::stdout().is_terminal();
    let stdin_is_tty = std::io::stdin().is_terminal();
    let use_tui = !cli.plain && stdout_is_tty && stdin_is_tty;
    let active_profile = cli.profile.clone().or_else(|| file.default_profile.clone());

    // The full-screen TUI is the default interactive experience; fall back to
    // the plain REPL when not on a TTY, when --plain is set, or if it errors.
    if use_tui {
        // Build the profile list and resolver for `/provider` in the TUI.
        let profiles: Vec<hi_tui::ProfileInfo> = profile_infos(&file);
        let resolver: hi_tui::ProfileResolver = Box::new({
            let file = file.clone();
            move |name: &str| {
                let settings = config::resolve_named_profile(&file, name)?;
                let label = provider_label(settings.provider).to_string();
                let model = settings.model.clone();
                let provider = build_chain(&settings, Vec::new());
                Ok(hi_tui::SwitchedProvider {
                    provider,
                    model,
                    label,
                    max_tokens: settings.max_tokens,
                    max_tokens_explicit: settings.max_tokens_explicit,
                })
            }
        });
        let saver: hi_tui::ProfileSaver = Box::new({
            let file = std::sync::Mutex::new(file.clone());
            let config_path = cli.config.clone();
            move |data: &hi_tui::ProfileFormData| {
                let provider = data
                    .provider
                    .parse::<ProviderName>()
                    .map_err(|e| anyhow::anyhow!("invalid provider '{}': {e}", data.provider))?;
                let form = config::ProfileForm {
                    name: data.name.clone(),
                    provider,
                    api_key: data.api_key.clone(),
                    store_as_env: data.store_as_env,
                    model: data.model.clone(),
                    base_url: data.base_url.clone(),
                };
                let path = config::writable_config_path(config_path.as_deref())
                    .context("could not determine config path")?;
                let mut file = file.lock().unwrap();
                let mut profile = form.to_profile();
                if profile.mcp_url.is_none()
                    && let Some(existing) = file.profiles.get(&data.name)
                {
                    profile.mcp_url = existing.mcp_url.clone();
                }
                config::upsert_profile(&mut file, &data.name, profile, &path)?;
                // Return the updated profile list.
                Ok(profile_infos(&file))
            }
        });
        let loader: hi_tui::ProfileLoader = Box::new({
            let file = file.clone();
            move |name: &str| {
                let p = file
                    .profiles
                    .get(name)
                    .ok_or_else(|| anyhow::anyhow!("no profile named '{name}'"))?;
                let form = config::ProfileForm::from_profile(name, p);
                Ok(hi_tui::ProfileFormData {
                    name: form.name,
                    provider: form.provider.as_str().to_string(),
                    api_key: form.api_key,
                    store_as_env: form.store_as_env,
                    model: form.model,
                    base_url: form.base_url,
                })
            }
        });
        let remover: hi_tui::ProfileRemover = Box::new({
            let file = std::sync::Mutex::new(file.clone());
            let config_path = cli.config.clone();
            move |name: &str| {
                let path = config::writable_config_path(config_path.as_deref())
                    .context("could not determine config path")?;
                let mut file = file.lock().unwrap();
                let existed = config::remove_profile(&mut file, name, &path)?;
                if !existed {
                    anyhow::bail!("no profile named '{name}'");
                }
                Ok(profile_infos(&file))
            }
        });
        let mlx_switcher: hi_tui::MlxProfileSwitcher = Box::new({
            let file = std::sync::Mutex::new(file.clone());
            let config_path = cli.config.clone();
            move |run: &hi_tools::HfMlxRun| {
                let path = config::writable_config_path(config_path.as_deref())
                    .context("could not determine config path")?;
                let mut file = file.lock().unwrap();
                let profile = config::Profile {
                    provider: Some(ProviderName::Openai),
                    model: Some(run.model_id.clone()),
                    base_url: Some(run.base_url.clone()),
                    api_key: Some("local".to_string()),
                    max_tokens: Some(2048),
                    ..Default::default()
                };
                config::upsert_profile_as_default(&mut file, &run.profile_name, profile, &path)?;
                let settings = config::resolve_named_profile(&file, &run.profile_name)?;
                let label = provider_label(settings.provider).to_string();
                let model = settings.model.clone();
                let provider = build_chain(&settings, Vec::new());
                Ok(hi_tui::MlxProfileSwitch {
                    switched: hi_tui::SwitchedProvider {
                        provider,
                        model,
                        label,
                        max_tokens: settings.max_tokens,
                        max_tokens_explicit: settings.max_tokens_explicit,
                    },
                    profiles: profile_infos(&file),
                })
            }
        });
        match hi_tui::run(
            &mut agent,
            provider_label(settings.provider),
            &settings.base_url,
            &settings.model,
            session::history_path(),
            auto_memory,
            profiles,
            active_profile.clone(),
            resolver,
            saver,
            loader,
            remover,
            mlx_switcher,
            resume_summary.clone(),
            settings.mcp_url.clone(),
            settings.api_key.clone(),
            fleet_launcher,
        )
        .await
        {
            Ok(()) => {
                feedback::maybe_prompt_and_submit(&settings, &feedback_session_id).await;
                hi_tools::kill_background_processes();
                return Ok(());
            }
            Err(err) => eprintln!("\x1b[33mTUI error ({err:#}); falling back to plain mode\x1b[0m"),
        }
    }

    // Plain REPL startup (including TUI fallback): print normal-screen context
    // here, not before TUI launch. The TUI renders its own splash/resume summary
    // inside the alternate screen; printing them before entering TUI leaves a
    // stale banner in scrollback and makes a normal exit look like a crash.
    if let Some(summary) = &resume_summary {
        println!("\x1b[2m{summary}\x1b[0m");
    }
    if stdout_is_tty {
        print_landing(&settings, live_metadata.context_window);
    }

    let repl_result = repl(
        &mut agent,
        &settings,
        &mut file,
        auto_memory,
        active_profile,
        cli.config.clone(),
    )
    .await;
    if repl_result.is_ok() {
        feedback::maybe_prompt_and_submit(&settings, &feedback_session_id).await;
    }
    repl_result
}

pub(crate) fn provider_label(provider: ProviderName) -> &'static str {
    match provider {
        ProviderName::Openai => "openai",
        ProviderName::Anthropic => "anthropic",
        ProviderName::Pipenetwork => "pipenetwork",
        ProviderName::Ollama => "ollama",
    }
}

async fn run_mcp_command(settings: &Settings) -> Result<()> {
    let Some(url) = settings.mcp_url.as_deref() else {
        return Err(anyhow!("no MCP URL configured for this provider"));
    };
    let report = mcp_inspect(url, &settings.api_key, &settings.model).await?;
    print!("{report}");
    Ok(())
}

async fn run_hf_cli(args: &[String]) -> Result<()> {
    if args.is_empty() {
        print!(
            "{}",
            hi_tools::handle_hf_command("help", &mut hi_tools::HfCommandState::default()).await?
        );
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("download")
        && args
            .get(2)
            .map(String::as_str)
            .is_some_and(|arg| matches!(arg, "--keep" | "keep"))
    {
        let repo = args
            .get(1)
            .ok_or_else(|| anyhow!("usage: hi hf download <repo[@revision]> --keep <dir>"))?;
        let dir = args
            .get(3)
            .ok_or_else(|| anyhow!("usage: hi hf download <repo[@revision]> --keep <dir>"))?;
        print!(
            "{}",
            hi_tools::download_repo_keep_foreground(repo, dir).await?
        );
        return Ok(());
    }

    let mut state = hi_tools::HfCommandState::default();
    match hi_tools::handle_hf_command_result(&args.join(" "), &mut state).await? {
        hi_tools::HfCommandResult::Text(text) => print!("{text}"),
        hi_tools::HfCommandResult::MlxReady(run) => print!("{}", run.message),
    }
    Ok(())
}

/// Build the MCP inspection report (server, tools, model count, current model)
/// as a plain-text block. Shared by the `hi mcp` one-shot and the REPL `/mcp`
/// command so their output can't drift.
async fn mcp_inspect(url: &str, api_key: &str, current_model: &str) -> Result<String> {
    let client = PipeMcpClient::new(url, api_key);
    let (server, protocol) = client.server_info().await?;
    let tools = client.tools_list().await?;
    let models = client.list_models().await?;
    let mut out = String::new();
    out.push_str(&format!("mcp_url:  {url}\n"));
    out.push_str(&format!("server:   {server}\n"));
    out.push_str(&format!("protocol: {protocol}\n"));
    out.push_str("tools:\n");
    for tool in tools {
        let title = tool.title.as_deref().unwrap_or("");
        if title.is_empty() {
            out.push_str(&format!("  {}\n", tool.name));
        } else {
            out.push_str(&format!("  {}  - {}\n", tool.name, title));
        }
    }
    out.push_str(&format!("models:   {}\n", models.len()));
    if let Some(model) = models.iter().find(|m| m.id == current_model) {
        let provider = model.provider_label.as_deref().unwrap_or("Pipe");
        out.push_str(&format!("current:  {} · {}\n", model.id, provider));
    }
    Ok(out)
}

/// The "PipeNetwork.AI" wordmark as figlet-style 5-row block letters — the
/// splash centerpiece, ~2x the height of a normal line. Generated from
/// `figlet -f small`, then hand-trimmed of trailing whitespace.
const BANNER: [&str; 5] = [
    " ___ _           _  _     _                  _       _   ___ ",
    "| _ (_)_ __  ___| \\| |___| |___ __ _____ _ _| |__   /_\\ |_ _|",
    "|  _/ | '_ \\/ -_) .` / -_)  _\\ V  V / _ \\ '_| / /_ / _ \\ | | ",
    "|_| |_| .__/\\___|_|\\_\\___|\\__|\\_/\\_/\\___/_| |_\\_(_)_/ \\_\\___|",
    "      |_|                                                    ",
];

/// Print the PipeNetwork.AI landing banner on startup, with the wordmark
/// rendered ~2x size as a 5-row block-letter banner (all orange). Only called
/// on a real terminal in interactive mode (one-shot prompts and piped runs are
/// excluded upstream).
fn print_landing(settings: &Settings, context_window: Option<u32>) {
    // Formatting goes through `write_landing`, which is unit-tested; this is
    // just the stdout sink.
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    let _ = write_landing(&mut out, settings, context_window);
    let _ = out.flush();
}

/// Render the landing banner into `w`. Separated from `print_landing` so the
/// exact text (ANSI escapes, banner, model, cwd) can be asserted in tests
/// without touching real file descriptors.
fn write_landing<W: std::io::Write>(
    w: &mut W,
    settings: &Settings,
    context_window: Option<u32>,
) -> std::io::Result<()> {
    let orange = "\x1b[38;2;255;140;0m";
    let bold = "\x1b[1m";
    let dim = "\x1b[2m";
    let reset = "\x1b[0m";

    // The 5-row block-letter banner, all orange + bold.
    for row in BANNER {
        writeln!(w, "{bold}{orange}{row}{reset}")?;
    }

    // Model + context window + provider.
    let ctx = context_window
        .map(|win| format!("({}K context)", win / 1000))
        .unwrap_or_default();
    let provider = provider_label(settings.provider);
    let model_line = if ctx.is_empty() {
        format!("{} · {}", settings.model, provider)
    } else {
        format!("{} {} · {}", settings.model, ctx, provider)
    };
    writeln!(w, "{dim}{model_line}{reset}")?;

    // Current working directory.
    let cwd = std::env::current_dir()
        .map(|d| d.display().to_string())
        .unwrap_or_else(|_| "?".into());
    writeln!(w, "{dim}{cwd}{reset}")?;
    Ok(())
}

/// Build the TUI profile list from a config. Shared by the initial list, the
/// saver callback, and the remover callback so they all stay in sync. Only
/// non-default base URLs are included (to keep the `/provider` list concise).
fn profile_infos(config: &config::Config) -> Vec<hi_tui::ProfileInfo> {
    config::profile_names(config)
        .into_iter()
        .map(|name| {
            let p = config.profiles.get(&name);
            let provider = p
                .and_then(|p| p.provider)
                .map(provider_label)
                .unwrap_or("openai")
                .to_string();
            let model = p.and_then(|p| p.model.clone());
            // Only show the base URL when it differs from the provider default.
            let base_url = p.and_then(|p| {
                p.base_url.clone().filter(|url| {
                    let default = p.provider.map(|prov| prov.default_base_url()).unwrap_or("");
                    url.trim_end_matches('/') != default.trim_end_matches('/')
                })
            });
            hi_tui::ProfileInfo {
                name,
                provider,
                model,
                base_url,
            }
        })
        .collect()
}

/// Decide the session file and whether to preload history.
struct LoadedAgentSession {
    messages: Vec<Message>,
    usage: Usage,
    checkpoint_refs: Vec<String>,
    structured_goal: Option<hi_agent::Goal>,
    decisions: hi_agent::DecisionLog,
    /// A one-line summary of the resumed session, shown to the user on startup.
    resume_summary: Option<String>,
}

fn resolve_session(cli: &Cli) -> Result<(std::path::PathBuf, Option<LoadedAgentSession>)> {
    // An exact session file (fleet child): create it fresh, or resume it if it
    // already has history — the dashboard reuses one file across a row's turns.
    if let Some(path) = &cli.session_file {
        if path.is_file() {
            let loaded = session::load_history(path)?;
            return Ok((
                path.clone(),
                Some(LoadedAgentSession {
                    messages: loaded.messages,
                    usage: loaded.usage,
                    checkpoint_refs: loaded.checkpoint_refs,
                    structured_goal: loaded.goal,
                    decisions: loaded.decisions,
                    resume_summary: None,
                }),
            ));
        }
        return Ok((path.clone(), None));
    }
    if let Some(id) = &cli.resume {
        let path = session::session_path(id)?;
        let loaded = session::load_history(&path)?;
        let summary = session::resume_summary(&loaded);
        return Ok((
            path,
            Some(LoadedAgentSession {
                messages: loaded.messages,
                usage: loaded.usage,
                checkpoint_refs: loaded.checkpoint_refs,
                structured_goal: loaded.goal,
                decisions: loaded.decisions,
                resume_summary: Some(summary),
            }),
        ));
    }
    if cli.cont {
        if let Some(path) = session::latest_session() {
            let loaded = session::load_history(&path)?;
            let summary = session::resume_summary(&loaded);
            return Ok((
                path,
                Some(LoadedAgentSession {
                    messages: loaded.messages,
                    usage: loaded.usage,
                    checkpoint_refs: loaded.checkpoint_refs,
                    structured_goal: loaded.goal,
                    decisions: loaded.decisions,
                    resume_summary: Some(summary),
                }),
            ));
        }
        eprintln!("\x1b[33mno previous session; starting a new one\x1b[0m");
    }
    Ok((session::new_session_path()?, None))
}

/// The one-shot prompt, with piped stdin folded in as context when present
/// (e.g. `cargo test 2>&1 | hi "fix the failures"`). Interactive mode (no
/// prompt) leaves stdin alone for the REPL.
fn effective_prompt(cli: &Cli) -> Result<Option<String>> {
    use std::io::IsTerminal;
    let Some(prompt) = cli.prompt.clone() else {
        return Ok(None);
    };
    if std::io::stdin().is_terminal() {
        return Ok(Some(prompt));
    }
    let piped = std::io::read_to_string(std::io::stdin()).context("reading stdin")?;
    let piped = piped.trim();
    if piped.is_empty() {
        return Ok(Some(prompt));
    }
    Ok(Some(format!("{prompt}\n\nstdin:\n```\n{piped}\n```")))
}

fn absolutize_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(std::env::current_dir()
        .context("determining current directory")?
        .join(path))
}

fn maybe_chdir_to_prompt_review_target(prompt: &str) -> Result<Option<PathBuf>> {
    let Some(target) = review_target_dir_from_prompt(prompt) else {
        return Ok(None);
    };
    let current = std::env::current_dir().context("determining current directory")?;
    let current = current.canonicalize().unwrap_or(current);
    if target == current {
        return Ok(Some(target));
    }
    std::env::set_current_dir(&target)
        .with_context(|| format!("changing to review target {}", target.display()))?;
    Ok(Some(target))
}

fn review_target_dir_from_prompt(prompt: &str) -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let home = std::env::var_os("HOME").map(PathBuf::from);
    review_target_dir_from_prompt_at(prompt, &cwd, home.as_deref())
}

fn review_target_dir_from_prompt_at(
    prompt: &str,
    cwd: &Path,
    home: Option<&Path>,
) -> Option<PathBuf> {
    let prompt = prompt
        .split("\n\nstdin:\n```")
        .next()
        .unwrap_or(prompt)
        .trim();
    if !prompt_looks_like_review_request(prompt) {
        return None;
    }
    prompt
        .split_whitespace()
        .filter_map(trim_prompt_path_token)
        .filter_map(|token| expand_review_target_token(token, cwd, home))
        .next()
}

fn prompt_looks_like_review_request(prompt: &str) -> bool {
    let normalized = prompt
        .split_whitespace()
        .filter(|raw| match trim_prompt_path_token(raw) {
            Some(token) => !token_looks_pathish(token),
            None => true,
        })
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>();
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    words.iter().any(|word| {
        matches!(
            *word,
            "review" | "audit" | "status" | "roadmap" | "gap" | "gaps" | "security"
        )
    })
}

fn trim_prompt_path_token(raw: &str) -> Option<&str> {
    let mut token = raw.trim_matches(|ch: char| {
        matches!(
            ch,
            '"' | '\'' | '`' | '<' | '>' | '(' | ')' | '[' | ']' | '{' | '}' | ','
        )
    });
    while token.len() > 1
        && token
            .chars()
            .last()
            .is_some_and(|ch| matches!(ch, '.' | ',' | ';' | ':' | '?' | '!'))
    {
        token = &token[..token.len() - 1];
    }
    (!token.is_empty()).then_some(token)
}

fn token_looks_pathish(token: &str) -> bool {
    token == "~"
        || token == "."
        || token == ".."
        || token.starts_with("~/")
        || token.starts_with("./")
        || token.starts_with("../")
        || token.starts_with('/')
        || token.contains('/')
}

fn expand_review_target_token(token: &str, cwd: &Path, home: Option<&Path>) -> Option<PathBuf> {
    if token.contains("://") {
        return None;
    }
    let expanded = if token == "~" {
        home?.to_path_buf()
    } else if let Some(rest) = token.strip_prefix("~/") {
        home?.join(rest)
    } else {
        PathBuf::from(token)
    };
    let path = if expanded.is_absolute() {
        expanded
    } else if token_looks_pathish(token) {
        cwd.join(expanded)
    } else {
        return None;
    };
    if !path.is_dir() {
        return None;
    }
    Some(path.canonicalize().unwrap_or(path))
}

/// The verification pipeline: an explicit `--verify` wins (one stage); otherwise
/// `--auto-verify` detects a layered pipeline from the working directory. Empty
/// = verification off.
fn resolve_verify(cli: &Cli) -> Vec<VerifyStage> {
    if let Some(cmd) = &cli.verify {
        return vec![VerifyStage::new("verify", cmd.clone())];
    }
    if cli.auto_verify {
        let detected = config::detect_verify_pipeline(std::path::Path::new("."));
        if detected.is_empty() {
            eprintln!("\x1b[33mauto-verify: no test command detected\x1b[0m");
        } else {
            let summary = detected
                .iter()
                .map(|s| s.command.as_str())
                .collect::<Vec<_>>()
                .join(" → ");
            eprintln!("\x1b[2mauto-verify: {summary}\x1b[0m");
        }
        return detected;
    }
    Vec::new()
}

/// Flatten a verify pipeline to a single `&&`-chained shell command, for callers
/// that need one pass/fail command (e.g. best-of selection). `None` when empty.
fn pipeline_command(stages: &[VerifyStage]) -> Option<String> {
    if stages.is_empty() {
        return None;
    }
    Some(
        stages
            .iter()
            .map(|s| s.command.as_str())
            .collect::<Vec<_>>()
            .join(" && "),
    )
}

/// Write a machine-readable run report (tokens, verify outcome) for the
/// eval harness and other automation.
fn write_report(
    path: &std::path::Path,
    agent: &Agent,
    model: &str,
    error: Option<&anyhow::Error>,
) -> Result<()> {
    let totals = agent.totals();
    let tel = agent.last_turn_telemetry();
    let report = serde_json::json!({
        "model": model,
        "input_tokens": totals.input_tokens,
        "output_tokens": totals.output_tokens,
        "total_tokens": totals.total(),
        "verify_passed": agent.last_verify(),
        "provider_error_kind": error.and_then(hi_ai::provider_error_kind).map(|k| k.as_str()),
        "compat_fallbacks_used": agent.last_compat_fallbacks(),
        "tool_mode_effective": tool_mode_label(agent.tool_mode()),
        "changed_files": agent.last_changed_files(),
        // The long-horizon goal state, when one is active — fleet rows read
        // this to decide auto-continue. Null when no structured goal is set.
        "goal": agent.structured_goal().map(|g| {
            serde_json::json!({
                "objective": g.objective,
                "done": g.sub_goals.iter().filter(|s| s.status == hi_agent::GoalStatus::Done).count(),
                "total": g.sub_goals.len(),
                "status": format!("{:?}", g.status),
                "paused": g.paused,
            })
        }),
        "telemetry": {
            "effective_max_steps": tel.effective_max_steps,
            "verify_rounds": tel.verify_rounds,
            "recovery_retries": tel.recovery_retries,
            "repeat_nudges": tel.repeat_nudges,
            "continue_nudges": tel.continue_nudges,
            "truncation_retries": tel.truncation_retries,
            "no_progress_streak": tel.no_progress_streak,
            "forced_final_answer_attempts": tel.forced_final_answer_attempts,
            "last_progress_reason": tel.last_progress_reason,
            "last_stall_reason": tel.last_stall_reason,
            "hit_step_cap": tel.hit_step_cap,
            "stalled_unfinished": tel.stalled_unfinished,
            "stalled_repeating": tel.stalled_repeating,
            "verify_attributions": tel.verify_attributions,
            "tool_calls": tel.tool_calls,
            "max_concurrent_batch": tel.max_concurrent_batch,
            "serial_runs": tel.serial_runs,
            "tool_timeline": tel.tool_timeline,
            "progress_events": tel.progress_events,
            "file_reads": tel.file_reads,
            "targeted_searches": tel.targeted_searches,
            "listing_only": tel.listing_only,
            "first_tool_kind": tel.first_tool_kind,
            "discovery_depth": tel.discovery_depth,
            "quality_repair_nudges": tel.quality_repair_nudges,
            "review_repair_exhaustion_reason": tel.review_repair_exhaustion_reason,
            "review_repair_counts": tel.review_repair_counts,
            "review_repair_stopped_by_exhaustion": tel.review_repair_stopped_by_exhaustion,
            "stopped_by_step_cap": tel.hit_step_cap,
        },
    });
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating report directory {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("writing report {}", path.display()))?;
    Ok(())
}

/// Load project context files from the working directory (pi-style). Combines
/// any of HI.md / AGENTS.md that exist.
fn load_project_context() -> Option<String> {
    const FILES: &[&str] = &["HI.md", "AGENTS.md"];
    let mut parts = Vec::new();
    for name in FILES {
        if let Ok(text) = std::fs::read_to_string(name) {
            let text = text.trim();
            if !text.is_empty() {
                parts.push(format!("# Project context (from {name})\n{text}"));
            }
        }
    }
    // Memory distilled from past sessions (auto-maintained at session end).
    // Hierarchical: project memory (annotated for stale paths/commands) + a
    // global user-level layer for cross-project preferences.
    let project = hi_agent::read_project_annotated();
    let global = hi_agent::read_global_memory();
    let mem = render_memory_layers(&project, &global);
    if let Some(section) = memory_context(&mem) {
        parts.push(section);
    }
    // A heuristic repo map so the model can navigate without reading everything.
    if let Some(map) = config::build_repo_map(std::path::Path::new(".")) {
        parts.push(map);
    }
    if let Some(section) = hi_agent::learned_skills_context() {
        parts.push(section);
    }
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// Whether auto-memory is active for this session: on unless `--no-memory`, and
/// off when the session isn't saved (`--no-save`) since memory is persistence.
fn auto_memory_enabled(no_memory: bool, no_save: bool) -> bool {
    !no_memory && !no_save
}

/// Build the `# Memory` context section from the saved memory file's contents,
/// or `None` when it's empty/whitespace (so a blank file adds nothing).
fn memory_context(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| format!("# Memory (from past sessions)\n{text}"))
}

/// Render the hierarchical memory layers into a single context block.
///
/// Project bullets are emitted first (annotated with stale-path warnings on
/// render), then global user-level bullets under a sub-heading. Either layer
/// may be empty.
fn render_memory_layers(project: &[hi_agent::AnnotatedBullet], global: &str) -> String {
    let mut out = String::new();
    for b in project {
        out.push_str(&b.render());
        out.push('\n');
    }
    let global = global.trim();
    if !global.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("## User-level (global)\n");
        out.push_str(global);
        out.push('\n');
    }
    out
}

pub(crate) fn build_provider(settings: &Settings) -> Box<dyn Provider> {
    let base_url = settings.base_url.clone();
    let api_key = settings.api_key.clone();
    if settings.provider.is_anthropic() {
        Box::new(AnthropicProvider::new(base_url, api_key))
    } else {
        let inner: Box<dyn Provider> = if settings.provider == ProviderName::Pipenetwork {
            Box::new(OpenAiProvider::new_pipenetwork(base_url, api_key.clone()))
        } else {
            Box::new(OpenAiProvider::new(base_url, api_key.clone()))
        };
        if settings.provider == ProviderName::Pipenetwork
            && let Some(mcp_url) = settings.mcp_url.clone()
        {
            Box::new(McpDiscoveryProvider::new(
                inner,
                PipeMcpClient::new(mcp_url, api_key),
            ))
        } else {
            inner
        }
    }
}

fn build_backend(settings: &Settings) -> Backend {
    Backend {
        provider: build_provider(settings),
        model: settings.model.clone(),
        label: format!("{}/{}", provider_label(settings.provider), settings.model),
    }
}

/// The primary backend, plus any fallbacks, as a single [`Provider`]. With no
/// fallbacks it's just the primary provider (no wrapper overhead).
pub(crate) fn build_chain(primary: &Settings, fallbacks: Vec<Settings>) -> Box<dyn Provider> {
    let passthrough: Box<dyn Provider> = if fallbacks.is_empty() {
        build_provider(primary)
    } else {
        let mut chain = vec![build_backend(primary)];
        chain.extend(fallbacks.iter().map(build_backend));
        Box::new(FallbackProvider::new(chain))
    };

    if !primary.moa.enabled {
        return passthrough;
    }

    Box::new(
        MoaProvider::new(passthrough, build_provider(primary), primary.moa.clone())
            .expect("MoA config should be validated before provider construction"),
    )
}

#[derive(Clone, Copy, Debug, Default)]
struct LiveModelMetadata {
    context_window: Option<u32>,
    max_output_tokens: Option<u32>,
}

fn effective_max_tokens_for_model(
    settings: &Settings,
    advertised_max_output_tokens: Option<u32>,
) -> u32 {
    hi_ai::effective_coding_agent_max_tokens(
        &settings.model,
        settings.max_tokens,
        settings.max_tokens_explicit,
        advertised_max_output_tokens,
    )
}

async fn resolve_live_model_metadata(
    provider: &dyn Provider,
    model: &str,
) -> LiveModelMetadata {
    match provider.list_models().await {
        Ok(served) => served
            .into_iter()
            .find(|m| m.id == model)
            .map(|m| LiveModelMetadata {
                context_window: m.context_window,
                max_output_tokens: m.max_output_tokens,
            })
            .unwrap_or(LiveModelMetadata {
                context_window: None,
                max_output_tokens: None,
            }),
        Err(_) => LiveModelMetadata {
            context_window: None,
            max_output_tokens: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        auto_memory_enabled, effective_max_tokens_for_model, memory_context,
        review_target_dir_from_prompt_at, write_landing,
    };
    use crate::config::{ProviderName, Settings};
    use hi_ai::{CompatMode, ToolMode};
    use std::path::PathBuf;

    #[test]
    fn auto_memory_off_when_disabled_or_unsaved() {
        assert!(auto_memory_enabled(false, false), "default on");
        assert!(!auto_memory_enabled(true, false), "--no-memory disables");
        assert!(!auto_memory_enabled(false, true), "--no-save disables");
    }

    #[test]
    fn memory_context_wraps_nonempty_and_skips_blank() {
        let section = memory_context("- run cargo fmt before commits").unwrap();
        assert!(section.starts_with("# Memory (from past sessions)"));
        assert!(section.contains("- run cargo fmt before commits"));
        assert!(memory_context("   \n  ").is_none(), "blank → no section");
    }

    fn test_settings() -> Settings {
        Settings {
            provider: ProviderName::Openai,
            model: "gpt-4o".into(),
            base_url: String::new(),
            mcp_url: None,
            api_key: String::new(),
            max_tokens: 4096,
            max_tokens_explicit: true,
            thinking_budget: None,
            tool_mode: ToolMode::default(),
            compat: CompatMode::default(),
            minimal_tools: false,
            curate_skills: false,
            explore_subagents: false,
            write_subagents: false,
            planner_model: None,
            moa: hi_ai::MoaConfig::default(),
        }
    }

    fn pipenetwork_settings(model: &str, max_tokens: u32, explicit: bool) -> Settings {
        Settings {
            provider: ProviderName::Pipenetwork,
            model: model.into(),
            base_url: String::new(),
            mcp_url: None,
            api_key: String::new(),
            max_tokens,
            max_tokens_explicit: explicit,
            thinking_budget: None,
            tool_mode: ToolMode::default(),
            compat: CompatMode::default(),
            minimal_tools: false,
            curate_skills: false,
            explore_subagents: false,
            write_subagents: false,
            planner_model: None,
            moa: hi_ai::MoaConfig::default(),
        }
    }

    fn temp_review_dir(name: &str) -> PathBuf {
        let unique = format!(
            "hi-target-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).unwrap();
        dir.canonicalize().unwrap()
    }

    #[test]
    fn review_target_detects_absolute_directory() {
        let dir = temp_review_dir("absolute");
        let cwd = std::env::current_dir().unwrap();

        let found = review_target_dir_from_prompt_at(
            &format!("review {} and discuss only", dir.display()),
            &cwd,
            None,
        )
        .unwrap();

        assert_eq!(found, dir);
        let _ = std::fs::remove_dir_all(found);
    }

    #[test]
    fn review_target_expands_home_directory() {
        let home = temp_review_dir("home");
        let repo = home.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let cwd = std::env::current_dir().unwrap();

        let found =
            review_target_dir_from_prompt_at("security review ~/repo read only", &cwd, Some(&home))
                .unwrap();

        assert_eq!(found, repo.canonicalize().unwrap());
        let _ = std::fs::remove_dir_all(home);
    }

    #[test]
    fn review_target_ignores_non_review_prompt() {
        let dir = temp_review_dir("non-review");
        let cwd = std::env::current_dir().unwrap();

        let found = review_target_dir_from_prompt_at(&format!("fix {}", dir.display()), &cwd, None);

        assert!(found.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn review_target_ignores_paths_only_in_folded_stdin() {
        let dir = temp_review_dir("stdin");
        let cwd = std::env::current_dir().unwrap();
        let prompt = format!("review codebase\n\nstdin:\n```\n{}\n```", dir.display());

        let found = review_target_dir_from_prompt_at(&prompt, &cwd, None);

        assert!(found.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn pipenetwork_coding_routes_apply_live_output_limits() {
        let balanced = pipenetwork_settings("ipop/coder-balanced", 8192, false);
        assert_eq!(
            effective_max_tokens_for_model(&balanced, Some(131_072)),
            131_072
        );

        let auto_code = pipenetwork_settings("pipe/auto-coder", 8192, false);
        assert_eq!(
            effective_max_tokens_for_model(&auto_code, Some(16_384)),
            16_384
        );
    }

    #[test]
    fn explicit_max_tokens_survive_live_metadata_but_clamp_down() {
        let lower = pipenetwork_settings("ipop/coder-balanced", 4096, true);
        assert_eq!(effective_max_tokens_for_model(&lower, Some(131_072)), 4096);

        let too_high = pipenetwork_settings("pipe/auto-coder", 65_536, true);
        assert_eq!(
            effective_max_tokens_for_model(&too_high, Some(16_384)),
            16_384
        );
    }

    /// `write_landing` renders the ~2x block-letter "PipeNetwork.AI" banner.
    /// We render into a `Vec<u8>`, strip ANSI escapes, and assert the banner
    /// shape (5 figlet rows), the trailing model/cwd lines, and that the raw
    /// output carries the orange SGR escape — no real file descriptors touched.
    #[test]
    fn write_landing_shows_full_pipenetwork_wordmark() {
        let mut buf: Vec<u8> = Vec::new();
        write_landing(&mut buf, &test_settings(), Some(128_000)).expect("render landing");

        let raw = String::from_utf8(buf).expect("utf8");
        let stripped = strip_ansi(&raw);
        let lines: Vec<&str> = stripped.lines().collect();

        // 5 banner rows + model line + cwd line = 7 content rows.
        assert!(
            lines.len() >= 7,
            "expected ≥7 lines (5 banner + model + cwd), got {}: {lines:?}",
            lines.len()
        );

        // The banner rows are the figlet art — they contain block-letter
        // strokes (pipes, underscores, slashes) and span 5 consecutive rows.
        let banner = &lines[0..5];
        // Every banner row is non-empty and carries pipe/underscore strokes.
        for (i, row) in banner.iter().enumerate() {
            assert!(
                row.contains('|') || row.contains('_'),
                "banner row {i} should carry figlet strokes, got: {row:?}"
            );
        }

        // Row 6 (index 5): model + provider + context window.
        let model_line = lines[5];
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

        // Row 7 (index 6): cwd — at minimum, non-empty (a path).
        assert!(
            !lines[6].is_empty(),
            "cwd line should be non-empty, got: {:?}",
            lines[6]
        );

        // The raw output must carry the orange SGR escape on banner rows.
        let orange_count = raw.matches("\x1b[38;2;255;140;0m").count();
        assert!(
            orange_count >= 5,
            "expected ≥5 orange SGR escapes (one per banner row), got {orange_count}"
        );
    }

    fn strip_ansi(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
                // Skip until we pass a letter (the terminator of a CSI sequence).
                i += 2;
                while i < bytes.len() && !bytes[i].is_ascii_alphabetic() {
                    i += 1;
                }
                i += 1;
            } else {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
        out
    }
}
