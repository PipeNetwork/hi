mod bestof;
mod config;
mod session;
mod ui;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use pi_agent::{Agent, AgentConfig};
use pi_ai::{AnthropicProvider, Message, OpenAiProvider, Provider, Registry};
use tokio::io::{AsyncBufReadExt, BufReader};

use config::{Cli, ProviderName, Settings};
use session::JsonlSession;
use ui::PlainUi;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.refresh_models {
        let count = pi_ai::registry::refresh().await?;
        let location = pi_ai::registry::cache_path()
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
    let settings = config::resolve(&cli, &file, &registry)?;

    let info = registry.lookup(&settings.model);
    if let Some(info) = info
        && !info.supports_tools
    {
        eprintln!(
            "\x1b[33mwarning: model '{}' is not known to support tool calling\x1b[0m",
            settings.model
        );
    }

    if cli.best_of > 1 {
        let prompt = cli
            .prompt
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

    let provider = build_provider(&settings);
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
    };
    let mut agent = match history {
        Some(history) => Agent::resume(provider, agent_config, history),
        None => Agent::new(provider, agent_config),
    };
    if !cli.no_save {
        agent.set_session(Box::new(JsonlSession::new(session_path)));
    }

    if let Some(prompt) = cli.prompt {
        let mut plain = PlainUi;
        let result = agent.run_turn(&prompt, &mut plain).await;
        if let Some(path) = &cli.report {
            write_report(path, &agent, &registry, &settings.model)?;
        }
        return result;
    }

    if cli.tui {
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() {
            return pi_tui::run(
                &mut agent,
                provider_label(settings.provider),
                &settings.model,
                &registry,
            )
            .await;
        }
        eprintln!("\x1b[33m--tui requires a terminal; falling back to plain mode\x1b[0m");
    }

    repl(&mut agent, &settings, &registry).await
}

fn provider_label(provider: ProviderName) -> &'static str {
    match provider {
        ProviderName::Openai => "openai",
        ProviderName::Anthropic => "anthropic",
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
fn write_report(path: &std::path::Path, agent: &Agent, registry: &Registry, model: &str) -> Result<()> {
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
    match settings.provider {
        ProviderName::Openai => Box::new(OpenAiProvider::new(base_url, api_key)),
        ProviderName::Anthropic => Box::new(AnthropicProvider::new(base_url, api_key)),
    }
}

async fn repl(agent: &mut Agent, settings: &Settings, registry: &Registry) -> Result<()> {
    use std::io::Write;

    println!(
        "hi · {} · {} — /help for commands, Ctrl-D to quit.",
        provider_label(settings.provider),
        settings.model
    );
    let mut plain = PlainUi;
    let mut lines = BufReader::new(tokio::io::stdin()).lines();

    loop {
        print!("\n› ");
        let _ = std::io::stdout().flush();

        let Some(line) = lines.next_line().await? else {
            break;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(command) = pi_agent::command::parse(line) {
            if handle_command(agent, command, registry) {
                break;
            }
            continue;
        }
        if let Err(err) = agent.run_turn(line, &mut plain).await {
            eprintln!("\x1b[31merror: {err:#}\x1b[0m");
        }
    }
    Ok(())
}

/// Act on a slash command. Returns true when the session should quit.
fn handle_command(agent: &mut Agent, command: pi_agent::Command, registry: &Registry) -> bool {
    use pi_agent::Command;
    match command {
        Command::Quit => return true,
        Command::Help => println!("{}", pi_agent::command::HELP),
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
        Command::Unknown(name) => {
            eprintln!("\x1b[33munknown command /{name}; try /help\x1b[0m");
        }
    }
    false
}
