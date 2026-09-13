//! First-run and `hi setup`: sign in to Pipe Network, then return Settings.

use std::io::{self, Write};

use anyhow::{Context, Result, bail};

use crate::auth::{LOGIN_PROFILE, install_pipenetwork_login_profile};
use crate::config::{
    Config, ProviderName, Settings, default_config_path, local_config_path, read_config_file,
};

/// Interactive Pipe sign-in. Browser pairing is the default; pasting a key is
/// the fallback. Writes `[profiles.pipenetwork]` and selects it.
pub async fn run(config: &mut Config) -> Result<Settings> {
    println!("Welcome to hi — sign in to pipenetwork.ai to start.\n");
    println!("  1) Browser pairing   (default)  opens pipenetwork.ai and stores the API key");
    println!("  2) Paste an API key             `hi auth pipenetwork` does the same later\n");
    print_sandbox_note();

    let use_login = loop {
        match prompt("How would you like to authenticate? [1-2] (default 1): ")?.trim() {
            "" | "1" => break true,
            "2" => break false,
            other => println!("  '{other}' isn't a choice — pick 1 or 2."),
        }
    };

    let api_key = if use_login {
        hi_ai::pipenetwork_auth::login().await?;
        let token = hi_ai::auth_store::load(hi_ai::pipenetwork_auth::PROVIDER_ID)
            .context("sign-in reported success but stored no credential")?;
        match install_pipenetwork_login_profile(config) {
            Ok(path) => {
                println!("Configured pipenetwork in {}", path.display());
                warn_if_shadowed_by_local_config();
            }
            Err(err) => eprintln!("(couldn't save config: {err:#})"),
        }
        token.access
    } else {
        let key = prompt("Paste your pipenetwork API key: ")?
            .trim()
            .to_string();
        if key.is_empty() {
            bail!("no API key entered");
        }
        let path = default_config_path().context("could not determine config directory")?;
        print!("\x1b[2m  testing connection…\x1b[0m\r");
        let _ = io::Write::flush(&mut io::stdout());
        match crate::auth::apply_pasted_key(config, ProviderName::Pipenetwork, &key, None, &path)
            .await
        {
            Ok(hi_ai::KeyCheck::Accepted) => {
                println!("\x1b[2m  ✓ connection verified\x1b[0m");
                println!("Saved to {}", path.display());
                warn_if_shadowed_by_local_config();
            }
            Ok(hi_ai::KeyCheck::Rejected(msg)) => {
                println!("\x1b[31m  ✗ key rejected: {msg}\x1b[0m");
                bail!("not saving — a 401/403 key is never written to config.toml");
            }
            Ok(hi_ai::KeyCheck::Unverified(msg)) => {
                println!("\x1b[33m  ⚠ couldn't verify the connection: {msg}\x1b[0m");
                println!(
                    "Saved to {} — hi will retry on the first turn.",
                    path.display()
                );
                warn_if_shadowed_by_local_config();
            }
            Err(err) => return Err(err),
        }
        key
    };

    println!();
    Ok(pipe_settings(api_key))
}

pub(crate) fn pipe_settings(api_key: String) -> Settings {
    let provider = ProviderName::Pipenetwork;
    Settings {
        execution: crate::config::ExecutionMode::Ephemeral,
        provider,
        model: provider
            .default_model()
            .unwrap_or(hi_harness::DEFAULT_MODEL)
            .to_string(),
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
        write_subagents: crate::config::WriteSubagentPolicy::Risk,
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
        session_harness: crate::config::empty_session_harness(),
        harness: hi_workspace::ResolvedHarnessSettings::default(),
    }
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

fn warn_if_shadowed_by_local_config() {
    let local = local_config_path();
    let Ok(file) = read_config_file(&local) else {
        return;
    };
    let shadows_profile = file.profiles.contains_key(LOGIN_PROFILE);
    let shadows_default = file.default_profile.is_some();
    if !shadows_profile && !shadows_default {
        return;
    }
    let what = if shadows_profile {
        format!("a '{LOGIN_PROFILE}' profile")
    } else {
        "default_profile".to_string()
    };
    println!(
        "\x1b[33m  ⚠ {} in this directory sets {what}, which overrides what was just saved.\x1b[0m",
        local.display()
    );
    println!(
        "\x1b[2m  Run hi from another directory, or edit that file, to use the new sign-in.\x1b[0m"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipe_settings_are_pipenetwork() {
        let settings = pipe_settings("pk_test".into());
        assert_eq!(settings.provider, ProviderName::Pipenetwork);
        assert_eq!(settings.api_key, "pk_test");
        assert_eq!(settings.model, hi_harness::DEFAULT_MODEL);
        assert_eq!(settings.base_url, hi_harness::DEFAULT_BASE_URL);
    }

    #[test]
    fn setup_copy_does_not_offer_other_providers() {
        let code = include_str!("setup.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(
            !code.contains("OpenRouter")
                && !code.contains("Anthropic")
                && !code.contains("Ollama")
                && !code.contains("Provider [1-5]"),
            "setup must stay Pipe-only"
        );
        assert!(code.contains("pipenetwork.ai"));
        assert!(code.contains("Browser pairing"));
    }
}
