//! Interactive first-run setup: pick a provider, paste a key, optionally save
//! it to the config file, and return ready-to-use [`Settings`].

use std::io::{self, Write};

use anyhow::{Context, Result, bail};

use crate::config::{ProviderName, Settings, default_config_path};

pub async fn run() -> Result<Settings> {
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
        match save_config(provider, &model, key_to_save) {
            Ok(path) => println!("Saved to {}", path.display()),
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
fn save_config(
    provider: ProviderName,
    model: &str,
    api_key: Option<&str>,
) -> Result<std::path::PathBuf> {
    let path = default_config_path().context("could not determine config directory")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Store the literal key in the config file (chmod 600 below). We used to
    // store an env var reference (api_key_env = "HI_API_KEY") and tell the user
    // to `export HI_API_KEY=…` in their shell profile, but if they didn't do
    // that (or didn't restart their shell), every run failed with "env var
    // HI_API_KEY (from profile) is not set". Storing the key directly is simpler
    // and works immediately — the config file is already protected.
    let key_line = match api_key {
        Some(key) if !matches!(provider, ProviderName::Ollama) => {
            format!("api_key = \"{key}\"\n")
        }
        _ => String::new(),
    };
    let contents = format!(
        "default_profile = \"default\"\n\n\
         [profiles.default]\n\
         provider = \"{}\"\n\
         model = \"{}\"\n\
         {}",
        provider.as_str(),
        model,
        key_line,
    );
    std::fs::write(&path, contents).with_context(|| format!("writing {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(path)
}
