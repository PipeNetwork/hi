//! Interactive first-run setup: pick a provider, paste a key, optionally save
//! it to the config file, and return ready-to-use [`Settings`].

use std::io::{self, Write};

use anyhow::{Context, Result, bail};

use crate::config::{ProviderName, Settings, default_config_path};

pub async fn run() -> Result<Settings> {
    println!("Welcome to hi — let's set up a model provider.\n");
    println!("  1) pipenetwork.ai    hi's hosted endpoint — paste an API key");
    println!("  2) Ollama (local)    run models on your machine — free, private, no key");
    println!("                      needs `ollama serve` running (install: ollama.com)\n");
    println!("  Want a cloud model? Pick 1. Local-first? Pick 2.\n");

    let provider = loop {
        match prompt("Provider [1-2] (default 1): ")?.trim() {
            "" | "1" => break ProviderName::Pipenetwork,
            "2" => break ProviderName::Ollama,
            other => println!("  '{other}' isn't a choice — pick 1-2."),
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

    let save = prompt("Save to ~/.config/hi/config.toml so you don't repeat this? [Y/n]: ")?;
    if !save.trim().eq_ignore_ascii_case("n") {
        match save_config(provider, &model, &api_key) {
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
        tool_mode: hi_ai::ToolMode::Auto,
        compat: hi_ai::CompatMode::Auto,
        minimal_tools: false,
        moa: hi_ai::MoaConfig::default(),
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

fn save_config(provider: ProviderName, model: &str, api_key: &str) -> Result<std::path::PathBuf> {
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
    let contents = if matches!(provider, ProviderName::Ollama) {
        format!(
            "default_profile = \"default\"\n\n\
             [profiles.default]\n\
             provider = \"{}\"\n\
             model = \"{}\"\n",
            provider.as_str(),
            model,
        )
    } else {
        format!(
            "default_profile = \"default\"\n\n\
             [profiles.default]\n\
             provider = \"{}\"\n\
             model = \"{}\"\n\
             api_key = \"{}\"\n",
            provider.as_str(),
            model,
            api_key,
        )
    };
    std::fs::write(&path, contents).with_context(|| format!("writing {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(path)
}
