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
    println!("  1) pipenetwork.ai    hi's hosted endpoint — browser sign-in or API key");
    println!("  2) Ollama (local)    run models on your machine — free, private, no key");
    println!("                      needs `ollama serve` running (install: ollama.com)");
    println!("  3) xAI (Grok)        subscription sign-in or API key from console.x.ai\n");
    println!("  Want a cloud model? Pick 1 or 3. Local-first? Pick 2.\n");

    let provider = loop {
        match prompt("Provider [1-3] (default 1): ")?.trim() {
            "" | "1" => break ProviderName::Pipenetwork,
            "2" => break ProviderName::Ollama,
            "3" => break ProviderName::Xai,
            other => println!("  '{other}' isn't a choice — pick 1-3."),
        }
    };

    let model = match provider.default_model() {
        Some(model) => model.to_string(),
        None => {
            let hint = if matches!(provider, ProviderName::Ollama) {
                "qwen2.5-coder"
            } else {
                "anthropic/claude-opus-4-8"
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
        // xAI takes either a subscription sign-in or a metered API key.
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
            // The credential lives in auth.json, not the config file. Return the
            // access token so this session works immediately; later runs re-read
            // (and refresh) it from the store.
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
        // Browser pairing mints a project API key; paste remains for existing keys.
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
        let key = prompt(&format!("Paste your {} API key: ", provider.as_str()))?;
        let key = key.trim().to_string();
        if key.is_empty() {
            bail!("no API key entered");
        }
        key
    };

    // Test the connection before saving, so configuration issues surface during setup.
    print!("\x1b[2m  testing connection…\x1b[0m\r");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let test_result = test_connection(provider, &model, &api_key).await;
    match &test_result {
        Ok(()) => println!("\x1b[2m  ✓ connection verified\x1b[0m"),
        Err(err) => {
            println!("\x1b[33m  ⚠ couldn't verify the connection: {err:#}\x1b[0m");
            println!("\x1b[2m  You can continue — hi will retry on the first turn.\x1b[0m");
        }
    }

    // Browser/subscription login already persisted a credential in auth.json.
    // Writing it into config.toml would duplicate a secret the user may copy
    // into a project (and for xAI, bake in a value that expires in hours).
    let credential_is_stored = (matches!(provider, ProviderName::Xai)
        && hi_ai::auth_store::load(hi_ai::xai_auth::PROVIDER_ID).is_some())
        || (matches!(provider, ProviderName::Pipenetwork)
            && hi_ai::auth_store::load(hi_ai::pipenetwork_auth::PROVIDER_ID).is_some());

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
    println!();

    Ok(Settings {
        provider,
        model,
        base_url: provider.default_base_url().to_string(),
        mcp_url: provider.default_mcp_url().map(String::from),
        api_key,
        max_tokens: 8192,
        max_tokens_explicit: false,
        thinking_budget: None,
        reasoning_effort: None,
        tool_mode: hi_ai::ToolMode::Auto,
        compat: hi_ai::CompatMode::Auto,
        curate_skills: false,
        // Match production defaults: explore on; delegate risk-gated.
        explore_subagents: true,
        write_subagents: hi_agent::WriteSubagentPolicy::Risk,
        planner_model: None,
        skeptic_model: None,
        moa: hi_ai::MoaConfig::default(),
        api_unix_socket: None,
    })
}

/// Test the connection by listing models. Returns `Ok(())` for any successful
/// model-list response, including an empty list.
async fn test_connection(provider: ProviderName, _model: &str, api_key: &str) -> Result<()> {
    use hi_ai::{OpenAiProvider, Provider};
    let base_url = provider.default_base_url();
    let p = OpenAiProvider::new(base_url.to_string(), api_key.to_string());
    p.list_models()
        .await
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("{err:#}"))
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
/// lives in `auth.json` instead (subscription login) or is not needed (Ollama).
///
/// This is a read-modify-write of the `default` profile only. It used to
/// `fs::write` a whole hand-formatted file, which erased every other profile
/// (and its key) in the config — the reason `needs_setup` was once narrowed to
/// "no profiles at all", which in turn made the wizard unreachable in any
/// directory holding a `hi.toml`. Keep this non-destructive: the trigger
/// depends on it.
fn save_config(
    config: &mut Config,
    provider: ProviderName,
    model: &str,
    api_key: Option<&str>,
) -> Result<std::path::PathBuf> {
    // Always the global config, never a project-local `hi.toml`: that's what
    // the save prompt promises, and a key written into a project file is one
    // `git add` away from being published.
    let path = default_config_path().context("could not determine config directory")?;
    let profile = Profile {
        provider: Some(provider),
        model: Some(model.to_string()),
        // Store the literal key in the config file (which `save_config_to`
        // chmods to 0600). We used to store an env var reference
        // (api_key_env = "HI_API_KEY") and tell the user to `export HI_API_KEY=…`
        // in their shell profile, but if they didn't do that (or didn't restart
        // their shell), every run failed with "env var HI_API_KEY (from profile)
        // is not set". Storing the key directly is simpler and works immediately
        // — the config file is already protected.
        api_key: api_key
            .filter(|_| !matches!(provider, ProviderName::Ollama))
            .map(str::to_string),
        ..Default::default()
    };
    upsert_profile_as_default(config, WIZARD_PROFILE, profile, Some(&path))?;
    Ok(path)
}

/// The wizard writes to the global config, but a `hi.toml` in the working
/// directory wins the merge. Say so rather than letting the next run silently
/// use something other than what was just chosen.
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
