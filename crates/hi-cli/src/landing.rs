//! Startup landing text and session resolution helpers.

use std::io::Write;

use anyhow::{Context, Result};
use hi_ai::{Message, Usage};

use crate::config::{Cli, Config, Settings};
use crate::provider::provider_label;
use crate::session;

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


pub(crate) fn print_landing(settings: &Settings, context_window: Option<u32>) {
    // Formatting goes through `write_landing`, which is unit-tested; this is
    // just the stdout sink.
        let mut out = std::io::stdout().lock();
    let _ = write_landing(&mut out, settings, context_window);
    let _ = out.flush();
}

/// Render the landing banner into `w`. Separated from `print_landing` so the
/// exact text (ANSI escapes, banner, model, cwd) can be asserted in tests
/// without touching real file descriptors.
pub(crate) fn write_landing<W: std::io::Write>(
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
pub(crate) fn profile_infos(config: &Config) -> Vec<hi_tui::ProfileInfo> {
    crate::config::profile_names(config)
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
pub(crate) struct LoadedAgentSession {
    pub(crate) messages: Vec<Message>,
    pub(crate) usage: Usage,
    pub(crate) checkpoint_refs: Vec<String>,
    pub(crate) structured_goal: Option<hi_agent::Goal>,
    pub(crate) decisions: hi_agent::DecisionLog,
    pub(crate) plan: Vec<hi_agent::PlanStep>,
    /// A one-line summary of the resumed session, shown to the user on startup.
    pub(crate) resume_summary: Option<String>,
}

pub(crate) fn resolve_session(cli: &Cli) -> Result<(std::path::PathBuf, Option<LoadedAgentSession>)> {
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
                    plan: loaded.plan,
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
                plan: loaded.plan,
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
                    plan: loaded.plan,
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
pub(crate) fn effective_prompt(cli: &Cli) -> Result<Option<String>> {
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

