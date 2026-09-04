//! Interactive first-run setup: pick a provider, paste a key, optionally save
//! it to the config file, and return ready-to-use [`Settings`].

use std::io::{self, Write};

use anyhow::{Context, Result, bail};

use crate::config::{
    Config, Profile, ProviderName, Settings, default_config_path, local_config_path,
    read_config_file, upsert_profile_as_default,
};

/// The profile name the wizard writes and selects as `default_profile`.
const WIZARD_PROFILE: &str = "default";

/// `config` is the session's merged config; the chosen profile is inserted into
/// it so the rest of the run (the `/provider` list, fallback resolution) sees
/// what was just saved without a reload.
pub async fn run(config: &mut Config) -> Result<Settings> {
    println!("Welcome to hi — let's set up a model provider.\n");
    println!("  1) pipenetwork.ai    hosted coding endpoint — browser sign-in or API key");
    println!("  2) OpenRouter        OpenAI-compatible cloud (OPENROUTER_API_KEY)");
    println!("  3) Anthropic         native Claude API (ANTHROPIC_API_KEY)");
    println!("  4) xAI (Grok)        subscription sign-in or API key from console.x.ai");
    println!("  5) Ollama (local)    models on this machine — free, private, no key");
    println!("                      needs `ollama serve` running (install: ollama.com)\n");
    println!(
        "  Cloud? 1–4. Local-first? 5. On Apple Silicon, /local can add a bundled MLX model later.\n"
    );

    let provider = loop {
        match prompt("Provider [1-5] (default 1): ")?.trim() {
            "" | "1" => break ProviderName::Pipenetwork,
            "2" => break ProviderName::Openai,
            "3" => break ProviderName::Anthropic,
            "4" => break ProviderName::Xai,
            "5" => break ProviderName::Ollama,
            other => println!("  '{other}' isn't a choice — pick 1-5."),
        }
    };

    print_sandbox_note();

    let model = match provider.default_model() {
        Some(model) => model.to_string(),
        None => {
            let hint = if matches!(provider, ProviderName::Ollama) {
                "qwen2.5-coder"
            } else {
                "anthropic/claude-sonnet-4"
            };
            let entered = prompt(&format!("Model id (default {hint}): "))?;
            let entered = entered.trim();
            if entered.is_empty() {
                hint.to_string()
            } else {
                entered.to_string()
            }
        }
    };

    let api_key = if matches!(provider, ProviderName::Ollama) {
        "ollama".to_string()
    } else if matches!(provider, ProviderName::Xai) {
        println!("\n  1) Sign in with a grok.com subscription (SuperGrok or X Premium)");
        println!("  2) Paste an API key from console.x.ai (billed per token)\n");
        let use_subscription = loop {
            match prompt("How would you like to authenticate? [1-2] (default 1): ")?.trim() {
                "" | "1" => break true,
                "2" => break false,
                other => println!("  '{other}' isn't a choice — pick 1-2."),
            }
        };
        if use_subscription {
            hi_ai::xai_auth::login().await?;
            hi_ai::auth_store::load(hi_ai::xai_auth::PROVIDER_ID)
                .map(|stored| stored.access)
                .context("sign-in reported success but stored no credential")?
        } else {
            let key = prompt("Paste your xAI API key: ")?.trim().to_string();
            if key.is_empty() {
                bail!("no API key entered");
            }
            key
        }
    } else if matches!(provider, ProviderName::Pipenetwork) {
        println!("\n  1) Sign in with your pipenetwork account (browser pairing)");
        println!("  2) Paste an existing API key\n");
        let use_login = loop {
            match prompt("How would you like to authenticate? [1-2] (default 1): ")?.trim() {
                "" | "1" => break true,
                "2" => break false,
                other => println!("  '{other}' isn't a choice — pick 1-2."),
            }
        };
        if use_login {
            hi_ai::pipenetwork_auth::login().await?;
            hi_ai::auth_store::load(hi_ai::pipenetwork_auth::PROVIDER_ID)
                .map(|stored| stored.access)
                .context("sign-in reported success but stored no credential")?
        } else {
            let key = prompt("Paste your pipenetwork API key: ")?
                .trim()
                .to_string();
            if key.is_empty() {
                bail!("no API key entered");
            }
            key
        }
    } else {
        let label = match provider {
            ProviderName::Anthropic => "Anthropic",
            ProviderName::Openai => "OpenRouter",
            other => other.as_str(),
        };
        let key = prompt(&format!("Paste your {label} API key: "))?;
        let key = key.trim().to_string();
        if key.is_empty() {
            bail!("no API key entered");
        }
        key
    };

    print!("\x1b[2m  testing connection…\x1b[0m\r");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let key_check =
        crate::auth::check_api_key(provider, &api_key, provider.default_base_url()).await;
    let skip_save = match &key_check {
        hi_ai::KeyCheck::Accepted => {
            println!("\x1b[2m  ✓ connection verified\x1b[0m");
            false
        }
        hi_ai::KeyCheck::Rejected(msg) => {
            println!("\x1b[31m  ✗ key rejected: {msg}\x1b[0m");
            println!("\x1b[2m  Not saving — a 401/403 key is never written to config.toml.\x1b[0m");
            true
        }
        hi_ai::KeyCheck::Unverified(msg) => {
            println!("\x1b[33m  ⚠ couldn't verify the connection: {msg}\x1b[0m");
            println!("\x1b[2m  You can continue — hi will retry on the first turn.\x1b[0m");
            false
        }
    };

    let credential_is_stored = (matches!(provider, ProviderName::Xai)
        && hi_ai::auth_store::load(hi_ai::xai_auth::PROVIDER_ID).is_some())
        || (matches!(provider, ProviderName::Pipenetwork)
            && hi_ai::auth_store::load(hi_ai::pipenetwork_auth::PROVIDER_ID).is_some());

    if !skip_save {
        let save = prompt("Save to ~/.config/hi/config.toml so you don't repeat this? [Y/n]: ")?;
        if !save.trim().eq_ignore_ascii_case("n") {
            let key_to_save = if credential_is_stored {
                None
            } else {
                Some(api_key.as_str())
            };
            match save_config(config, provider, &model, key_to_save) {
                Ok(path) => {
                    println!("Saved to {}", path.display());
                    warn_if_shadowed_by_local_config();
                }
                Err(err) => eprintln!("(couldn't save config: {err:#})"),
            }
        }
    }
    println!();

    Ok(Settings {
        execution: hi_agent::ExecutionMode::Ephemeral,
        provider,
        model,
        base_url: provider.default_base_url().to_string(),
        mcp_url: provider.default_mcp_url().map(String::from),
        api_key,
        max_tokens: 8192,
        max_tokens_explicit: false,
        top_p: None,
        output_token_parameter: hi_ai::OutputTokenParameter::Auto,
        thinking_budget: None,
        reasoning_effort: None,
        tool_mode: hi_ai::ToolMode::Auto,
        compat: hi_ai::CompatMode::Auto,
        deepseek_compat: hi_ai::DeepSeekCompat::Auto,
        curate_skills: false,
        explore_subagents: true,
        suggest_next_prompt: true,
        write_subagents: hi_agent::WriteSubagentPolicy::Risk,
        planner_model: None,
        skeptic_model: None,
        moa: hi_ai::MoaConfig::default(),
        api_unix_socket: None,
        runtime: None,
        x402: Default::default(),
        browser_enabled: true,
        browser_allow_private: false,
        mcp_pipe_enabled: true,
        mcp_pipe_allow: Vec::new(),
        session_harness: crate::session_harness::empty_layer(),
        harness: hi_workspace::ResolvedHarnessSettings::default(),
    })
}

fn print_sandbox_note() {
    let platform = if cfg!(target_os = "macos") {
        "macOS Seatbelt confines shell writes to this project"
    } else if cfg!(target_os = "linux") {
        "Linux confines shell writes when pipe-wrap is available; otherwise hi warns and continues"
    } else {
        "this OS does not confine shell writes — treat prompts as trusted"
    };
    println!(
        "\x1b[2m  Sandbox: {platform}. HI_SANDBOX=off disables it. /undo reverts the last turn.\x1b[0m\n"
    );
}

fn prompt(message: &str) -> Result<String> {
    print!("{message}");
    io::stdout().flush().ok();
    let mut line = String::new();
    if io::stdin().read_line(&mut line).context("reading input")? == 0 {
        bail!("setup cancelled");
    }
    Ok(line)
}

/// `api_key: None` writes a profile with no key — used when the credential
/// already lives in the private credential store (subscription login) or is
/// not needed (Ollama). `upsert_profile_as_default` seals any supplied static
/// key into that store and persists only its `api_key_ref`.
fn save_config(
    config: &mut Config,
    provider: ProviderName,
    model: &str,
    api_key: Option<&str>,
) -> Result<std::path::PathBuf> {
    let path = default_config_path().context("could not determine config directory")?;
    let profile = Profile {
        provider: Some(provider),
        model: Some(model.to_string()),
        api_key: api_key
            .filter(|_| !matches!(provider, ProviderName::Ollama))
            .map(str::to_string),
        ..Default::default()
    };
    upsert_profile_as_default(config, WIZARD_PROFILE, profile, Some(&path))?;
    Ok(path)
}

fn warn_if_shadowed_by_local_config() {
    let local = local_config_path();
    let Ok(file) = read_config_file(&local) else {
        return;
    };
    let shadows_profile = file.profiles.contains_key(WIZARD_PROFILE);
    let shadows_default = file.default_profile.is_some();
    if !shadows_profile && !shadows_default {
        return;
    }
    let what = if shadows_profile {
        format!("a '{WIZARD_PROFILE}' profile")
    } else {
        "default_profile".to_string()
    };
    println!(
        "\x1b[33m  ⚠ {} in this directory sets {what}, which overrides what was just saved.\x1b[0m",
        local.display()
    );
    println!(
        "\x1b[2m  Run hi from another directory, or edit that file, to use the new setup.\x1b[0m"
    );
}
