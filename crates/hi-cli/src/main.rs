mod bestof;
mod config;
mod session;
mod setup;
mod ui;

use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use clap::Parser;

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
use hi_agent::{Agent, AgentConfig};
use hi_ai::{
    AnthropicProvider, Backend, FallbackProvider, Message, OpenAiProvider, Provider, Registry,
};

use config::{Cli, ProviderName, Settings};
use session::JsonlSession;
use ui::PlainUi;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

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
    let file = config::load_config(cli.config.as_deref())?;

    // First run on a real terminal with nothing configured: walk the user
    // through an interactive setup instead of erroring.
    let settings = if cli.prompt.is_none()
        && config::needs_setup(&cli, &file)
        && std::io::stdin().is_terminal()
    {
        setup::run()?
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
        let verify = resolve_verify(&cli)
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
    let (session_path, history) = resolve_session(&cli)?;

    let provider = build_chain(&settings, config::resolve_fallbacks(&cli, &file, &registry));
    let (price, context_window) = registry.metadata(&settings.model);
    let agent_config = AgentConfig {
        model: settings.model.clone(),
        max_tokens: settings.max_tokens,
        temperature: cli.temperature,
        thinking_budget: settings.thinking_budget,
        price,
        context_window,
        project_context: load_project_context(),
        verify_command: resolve_verify(&cli),
        max_verify_iterations: cli.max_verify,
        max_steps: cli.max_steps,
        auto_compact: !cli.no_auto_compact,
    };
    let mut agent = match history {
        Some(history) => Agent::resume(provider, agent_config, history),
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
        if let Some(path) = &cli.report {
            write_report(path, &agent, &registry, &settings.model)?;
        }
        return result;
    }

    // The full-screen TUI is the default interactive experience; fall back to
    // the plain REPL when not on a TTY, when --plain is set, or if it errors.
    if !cli.plain && std::io::stdout().is_terminal() {
        match hi_tui::run(
            &mut agent,
            provider_label(settings.provider),
            &settings.model,
            &registry,
            session::history_path(),
        )
        .await
        {
            Ok(()) => return Ok(()),
            Err(err) => eprintln!("\x1b[33mTUI error ({err:#}); falling back to plain mode\x1b[0m"),
        }
    }

    repl(&mut agent, &settings, &registry).await
}

fn provider_label(provider: ProviderName) -> &'static str {
    match provider {
        ProviderName::Openai => "openai",
        ProviderName::Anthropic => "anthropic",
        ProviderName::Terminaili => "terminaili",
        ProviderName::Ollama => "ollama",
    }
}

/// Decide the session file and whether to preload history.
fn resolve_session(cli: &Cli) -> Result<(std::path::PathBuf, Option<Vec<Message>>)> {
    if let Some(id) = &cli.resume {
        let path = session::session_path(id)?;
        let history = session::load_history(&path)?;
        return Ok((path, Some(history)));
    }
    if cli.cont {
        if let Some(path) = session::latest_session() {
            let history = session::load_history(&path)?;
            return Ok((path, Some(history)));
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

/// The verification command: explicit `--verify` wins; otherwise `--auto-verify`
/// detects the project's test command from the working directory.
fn resolve_verify(cli: &Cli) -> Option<String> {
    if cli.verify.is_some() {
        return cli.verify.clone();
    }
    if cli.auto_verify {
        let detected = config::detect_verify_command_in(std::path::Path::new("."));
        match &detected {
            Some(cmd) => eprintln!("\x1b[2mauto-verify: using `{cmd}`\x1b[0m"),
            None => eprintln!("\x1b[33mauto-verify: no test command detected\x1b[0m"),
        }
        return detected;
    }
    None
}

/// Write a machine-readable run report (tokens, cost, verify outcome) for the
/// eval harness and other automation.
fn write_report(
    path: &std::path::Path,
    agent: &Agent,
    registry: &Registry,
    model: &str,
) -> Result<()> {
    let totals = agent.totals();
    let (price, _) = registry.metadata(model);
    let cost = price.map(|(input, output)| {
        (totals.input_tokens as f64 * input + totals.output_tokens as f64 * output) / 1_000_000.0
    });
    let report = serde_json::json!({
        "model": model,
        "input_tokens": totals.input_tokens,
        "output_tokens": totals.output_tokens,
        "total_tokens": totals.total(),
        "cost_usd": cost,
        "verify_passed": agent.last_verify(),
    });
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
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

fn build_provider(settings: &Settings) -> Box<dyn Provider> {
    let base_url = settings.base_url.clone();
    let api_key = settings.api_key.clone();
    if settings.provider.is_anthropic() {
        Box::new(AnthropicProvider::new(base_url, api_key))
    } else {
        Box::new(OpenAiProvider::new(base_url, api_key))
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

async fn repl(agent: &mut Agent, settings: &Settings, registry: &Registry) -> Result<()> {
    use hi_agent::Command;
    use rustyline::DefaultEditor;
    use rustyline::error::ReadlineError;

    println!(
        "hi · {} · {} — /help for commands, Ctrl-D to quit.",
        provider_label(settings.provider),
        settings.model
    );

    let mut editor = DefaultEditor::new().context("initializing line editor")?;
    let history = session::history_path();
    if let Some(path) = &history {
        let _ = editor.load_history(path);
    }

    // For `/retry`: the last message sent, and the history length just before
    // that turn (so we can drop it before re-running).
    let mut last_prompt: Option<String> = None;
    let mut last_turn_start = 0usize;

    loop {
        match editor.readline("› ") {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                let _ = editor.add_history_entry(&line);

                // Resolve the line to a prompt to run. Commands either handle
                // themselves (and `continue`) or yield a prompt (`/retry`).
                let input = if let Some(command) = hi_agent::command::parse(&line) {
                    match command {
                        Command::Quit => break,
                        Command::Compact => {
                            let progress = Arc::new(AtomicBool::new(false));
                            let mut plain = PlainUi::with_progress(progress.clone());
                            let _ = drive_with_spinner(agent.compact(&mut plain), &progress).await;
                            continue;
                        }
                        Command::Retry => match last_prompt.clone() {
                            Some(prompt) => {
                                agent.truncate_messages(last_turn_start);
                                println!("\x1b[2mretrying: {prompt}\x1b[0m");
                                prompt
                            }
                            None => {
                                println!("\x1b[2mnothing to retry yet\x1b[0m");
                                continue;
                            }
                        },
                        other => {
                            handle_command(agent, other, registry);
                            continue;
                        }
                    }
                } else {
                    line
                };

                // Run the turn with an animated "working… Ns" spinner so it's
                // always clear something is happening. Ctrl-C cancels the turn.
                last_prompt = Some(input.clone());
                let checkpoint = agent.messages().len();
                last_turn_start = checkpoint;
                let progress = Arc::new(AtomicBool::new(false));
                let cancelled = {
                    let mut plain = PlainUi::with_progress(progress.clone());
                    drive_with_spinner(agent.run_turn(&input, &mut plain), &progress).await
                };
                if cancelled {
                    agent.truncate_messages(checkpoint);
                    println!("\x1b[33m^C — interrupted; turn discarded\x1b[0m");
                }
            }
            Err(ReadlineError::Interrupted) => continue, // Ctrl-C: discard the line
            Err(ReadlineError::Eof) => break,            // Ctrl-D: quit
            Err(err) => {
                eprintln!("input error: {err}");
                break;
            }
        }
    }

    if let Some(path) = &history {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = editor.save_history(path);
    }
    Ok(())
}

/// Drive a model future (a turn or a compaction) to completion, showing an
/// animated `working… Ns` spinner until the first output and letting Ctrl-C
/// cancel it. Returns whether it was cancelled.
async fn drive_with_spinner(
    fut: impl std::future::Future<Output = Result<()>>,
    progress: &AtomicBool,
) -> bool {
    tokio::pin!(fut);
    let started = std::time::Instant::now();
    let mut ticker = tokio::time::interval(Duration::from_millis(90));
    let mut frame = 0usize;
    let mut cancelled = false;
    loop {
        tokio::select! {
            result = &mut fut => {
                if let Err(err) = result {
                    eprintln!("\r\x1b[K\x1b[31merror: {err:#}\x1b[0m");
                }
                break;
            }
            _ = tokio::signal::ctrl_c() => { cancelled = true; break; }
            _ = ticker.tick() => {
                if !progress.load(Ordering::Relaxed) {
                    print!(
                        "\r\x1b[2m{} working… {}s\x1b[0m\x1b[K",
                        SPINNER[frame % SPINNER.len()],
                        started.elapsed().as_secs()
                    );
                    let _ = std::io::stdout().flush();
                    frame += 1;
                }
            }
        }
    }
    if !progress.load(Ordering::Relaxed) {
        print!("\r\x1b[K");
        let _ = std::io::stdout().flush();
    }
    cancelled
}

/// Act on a slash command. Returns true when the session should quit.
fn handle_command(agent: &mut Agent, command: hi_agent::Command, registry: &Registry) -> bool {
    use hi_agent::Command;
    match command {
        Command::Quit => return true,
        Command::Help => println!("{}", hi_agent::command::HELP),
        Command::Tokens => {
            let t = agent.totals();
            println!(
                "\x1b[2mcumulative: {} in · {} out · {} total\x1b[0m",
                t.input_tokens,
                t.output_tokens,
                t.total()
            );
        }
        Command::Model(id) => {
            if id.is_empty() {
                println!("model: {}", agent.model());
            } else {
                let (price, context_window) = registry.metadata(&id);
                agent.set_model(id.clone(), price, context_window);
                println!("model set to {id}");
            }
        }
        Command::Clear => {
            agent.clear_history();
            println!("\x1b[2mconversation cleared\x1b[0m");
        }
        Command::Verify(arg) => match arg.trim() {
            "" => match agent.verify_command() {
                Some(c) => println!("\x1b[2mverify: {c}\x1b[0m"),
                None => println!("\x1b[2mverify: off (set one with /verify <cmd>)\x1b[0m"),
            },
            "off" | "none" | "clear" | "disable" => {
                agent.set_verify_command(None);
                println!("\x1b[2mverification disabled\x1b[0m");
            }
            cmd => {
                agent.set_verify_command(Some(cmd.to_string()));
                println!(
                    "\x1b[2mverification on: {cmd} — runs after each turn, iterates on failure\x1b[0m"
                );
            }
        },
        Command::Diff => println!("{}", hi_tools::working_tree_diff()),
        // Handled in the repl loop (async / runs a turn); never reach here.
        Command::Compact | Command::Retry => {}
        Command::Unknown(name) => {
            eprintln!("\x1b[33munknown command /{name}; try /help\x1b[0m");
        }
    }
    false
}
