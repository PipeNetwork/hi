mod bestof;
mod commands;
mod complete;
mod config;
mod repl;
mod session;
mod setup;
mod ui;

use std::io::IsTerminal;

use anyhow::{Context, Result, anyhow};
use clap::Parser;

use hi_agent::{Agent, AgentConfig, CompactionKind, VerifyStage};
use hi_ai::{
    AnthropicProvider, Backend, FallbackProvider, McpDiscoveryProvider, Message, OpenAiProvider,
    PipeMcpClient, Provider, Registry, Usage,
};

use commands::tool_mode_label;
use config::{Cli, ProviderName, Settings};
use repl::repl;
use session::JsonlSession;
use ui::PlainUi;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.show_config {
        let registry = Registry::load();
        let file = config::load_config(cli.config.as_deref())?;
        match config::resolve(&cli, &file, &registry) {
            Ok(settings) => {
                println!("provider:   {}", provider_label(settings.provider));
                println!("model:      {}", settings.model);
                println!("base_url:   {}", settings.base_url);
                if let Some(mcp_url) = &settings.mcp_url {
                    println!("mcp_url:    {mcp_url}");
                }
                println!("max_tokens: {}", settings.max_tokens);
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

    if cli.refresh_models {
        let count = hi_ai::registry::refresh().await?;
        let location = hi_ai::registry::cache_path()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        println!("Refreshed {count} models → {location}");
        return Ok(());
    }
    if cli.list_sessions {
        return session::list_sessions();
    }

    let registry = Registry::load();
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
        match config::resolve(&cli, &file, &registry) {
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

    let info = registry.lookup(&settings.model);
    if let Some(info) = info
        && !info.supports_tools
    {
        eprintln!(
            "\x1b[33mwarning: model '{}' is not known to support tool calling\x1b[0m",
            settings.model
        );
    }

    // Fold piped stdin into the one-shot prompt as context.
    let prompt_input = effective_prompt(&cli)?;

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

    let fallbacks = config::resolve_fallbacks(&cli, &file, &registry);
    let provider = build_chain(&settings, fallbacks);
    let context_window = if settings.provider == ProviderName::Pipenetwork {
        resolve_live_model_metadata(provider.as_ref(), &registry, &settings.model).await
    } else {
        registry.metadata(&settings.model).1
    };
    let agent_config = AgentConfig {
        model: settings.model.clone(),
        max_tokens: settings.max_tokens,
        temperature: cli.temperature,
        thinking_budget: settings.thinking_budget,
        tool_mode: settings.tool_mode,
        compat: settings.compat,
        context_window,
        project_context: load_project_context(),
        verify: resolve_verify(&cli),
        max_verify_iterations: cli.max_verify,
        max_steps: cli.max_steps,
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
    if !cli.no_save {
        agent.set_session(Box::new(JsonlSession::new(session_path)));
    }

    if let Some(prompt) = prompt_input {
        let mut plain = PlainUi::new();
        let mut quiet = ui::QuietUi;
        let view: &mut dyn hi_agent::Ui = if cli.quiet { &mut quiet } else { &mut plain };
        let result = agent.run_turn(&prompt, view).await;
        let report_result = if let Some(path) = &cli.report {
            write_report(
                path,
                &agent,
                &registry,
                &settings.model,
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

    // The full-screen TUI is the default interactive experience; fall back to
    // the plain REPL when not on a TTY, when --plain is set, or if it errors.
    if use_tui {
        // Build the profile list and resolver for `/provider` in the TUI.
        let profiles: Vec<hi_tui::ProfileInfo> = profile_infos(&file);
        let active_profile = cli.profile.clone().or_else(|| file.default_profile.clone());
        let resolver: hi_tui::ProfileResolver = Box::new({
            let file = file.clone();
            let registry = registry.clone();
            move |name: &str| {
                let settings = config::resolve_named_profile(&file, name, &registry)?;
                let label = provider_label(settings.provider).to_string();
                let model = settings.model.clone();
                let provider = build_provider(&settings);
                Ok(hi_tui::SwitchedProvider {
                    provider,
                    model,
                    label,
                })
            }
        });
        let saver: hi_tui::ProfileSaver = Box::new({
            let file = std::sync::Mutex::new(file.clone());
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
                let path = config::writable_config_path(None)
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
            move |name: &str| {
                let path = config::writable_config_path(None)
                    .context("could not determine config path")?;
                let mut file = file.lock().unwrap();
                let existed = config::remove_profile(&mut file, name, &path)?;
                if !existed {
                    anyhow::bail!("no profile named '{name}'");
                }
                Ok(profile_infos(&file))
            }
        });
        match hi_tui::run(
            &mut agent,
            provider_label(settings.provider),
            &settings.base_url,
            &settings.model,
            &registry,
            session::history_path(),
            auto_memory,
            profiles,
            active_profile,
            resolver,
            saver,
            loader,
            remover,
            resume_summary.clone(),
            settings.mcp_url.clone(),
            settings.api_key.clone(),
        )
        .await
        {
            Ok(()) => {
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
        print_landing(&settings, context_window);
    }

    repl(&mut agent, &settings, &mut file, &registry, auto_memory).await
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
        let health = model.health().unwrap_or("available");
        let provider = model.provider_label.as_deref().unwrap_or("Pipe");
        out.push_str(&format!(
            "current:  {} · {} · {}\n",
            model.id, provider, health
        ));
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
    _registry: &Registry,
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
        "telemetry": {
            "verify_rounds": tel.verify_rounds,
            "recovery_retries": tel.recovery_retries,
            "repeat_nudges": tel.repeat_nudges,
            "continue_nudges": tel.continue_nudges,
            "truncation_retries": tel.truncation_retries,
            "hit_step_cap": tel.hit_step_cap,
            "stalled_unfinished": tel.stalled_unfinished,
            "stalled_repeating": tel.stalled_repeating,
            "verify_attributions": tel.verify_attributions,
            "tool_calls": tel.tool_calls,
            "max_concurrent_batch": tel.max_concurrent_batch,
            "serial_runs": tel.serial_runs,
            "tool_timeline": tel.tool_timeline,
            "file_reads": tel.file_reads,
            "targeted_searches": tel.targeted_searches,
            "listing_only": tel.listing_only,
            "first_tool_kind": tel.first_tool_kind,
            "discovery_depth": tel.discovery_depth,
            "quality_repair_nudges": tel.quality_repair_nudges,
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
fn build_chain(primary: &Settings, fallbacks: Vec<Settings>) -> Box<dyn Provider> {
    if fallbacks.is_empty() {
        return build_provider(primary);
    }
    let mut chain = vec![build_backend(primary)];
    chain.extend(fallbacks.iter().map(build_backend));
    Box::new(FallbackProvider::new(chain))
}

async fn resolve_live_model_metadata(
    provider: &dyn Provider,
    registry: &Registry,
    model: &str,
) -> Option<u32> {
    let (_catalog_price, catalog_window) = registry.metadata(model);
    match provider.list_models().await {
        Ok(served) => served
            .into_iter()
            .find(|m| m.id == model)
            .map(|m| m.context_window.or(catalog_window))
            .unwrap_or(catalog_window),
        Err(_) => catalog_window,
    }
}

#[cfg(test)]
mod tests {
    use super::{auto_memory_enabled, memory_context, write_landing};
    use crate::config::{ProviderName, Settings};
    use hi_ai::{CompatMode, ToolMode};

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
            thinking_budget: None,
            tool_mode: ToolMode::default(),
            compat: CompatMode::default(),
        }
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
