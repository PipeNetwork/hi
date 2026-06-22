//! Interactive first-run setup: pick a provider, paste a key, optionally save
//! it to the config file, and return ready-to-use [`Settings`].

use std::io::{self, Write};

use anyhow::{Context, Result, bail};

use crate::config::{ProviderName, Settings, default_config_path};

pub fn run() -> Result<Settings> {
    println!("Welcome to hi — let's set up a model provider.\n");
    println!("  1) terminaili.com   (paste API key)");
    println!("  2) OpenRouter       (paste API key)");
    println!("  3) Anthropic        (paste API key)");
    println!("  4) Ollama (local)   (no key needed)\n");

    let provider = loop {
        match prompt("Provider [1-4] (default 1): ")?.trim() {
            "" | "1" => break ProviderName::Terminaili,
            "2" => break ProviderName::Openai,
            "3" => break ProviderName::Anthropic,
            "4" => break ProviderName::Ollama,
            other => println!("  '{other}' isn't a choice — pick 1-4."),
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
        api_key,
        max_tokens: 4096,
        thinking_budget: None,
        tool_mode: hi_ai::ToolMode::Auto,
        compat: hi_ai::CompatMode::Auto,
    })
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
    let contents = format!(
        "default_profile = \"default\"\n\n\
         [profiles.default]\n\
         provider = \"{}\"\n\
         model = \"{}\"\n\
         api_key = \"{}\"\n",
        provider.as_str(),
        model,
        api_key,
    );
    std::fs::write(&path, contents).with_context(|| format!("writing {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(path)
}
