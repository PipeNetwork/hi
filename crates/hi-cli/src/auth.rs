//! Fail-closed API-key write: probe `/models` before touching `config.toml`.
//!
//! `hi auth <provider>` and `/auth` paste a key. `/login` stays pairing-only
//! (xAI / pipenetwork / x402). HTTP 401/403 never writes a profile; transport
//! failures are unverified and may still save with a warning.

use std::io::{self, IsTerminal, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use hi_ai::{AnthropicProvider, KeyCheck, OpenAiProvider, Provider};

use crate::config::{
    Config, Profile, ProviderName, default_config_path, read_config_file, upsert_profile_as_default,
};

/// Providers that accept a pasted API key (`hi auth` / `/auth`). Pairing stays `/login`.
pub fn parse_key_provider(name: &str) -> std::result::Result<ProviderName, String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "openai" | "openrouter" => Ok(ProviderName::Openai),
        "anthropic" => Ok(ProviderName::Anthropic),
        "xai" | "grok" => Ok(ProviderName::Xai),
        "pipenetwork" | "pipe" | "ollama" | "local" | "x402" => Err(format!(
            "'{name}' is not a pasted-key provider. Use openai, anthropic, or xai. \
             Pairing stays /login xai | /login pipenetwork | /login x402."
        )),
        "" => Err("usage: /auth openai|anthropic|xai [api-key]  (or `hi auth <provider>`)".into()),
        other => Err(format!(
            "'{other}' has no pasted-key flow. Supported: openai, anthropic, xai. \
             Pairing stays /login."
        )),
    }
}

/// Split `/auth openai sk-…` into `(provider, optional key)`.
pub fn split_auth_arg(arg: &str) -> std::result::Result<(ProviderName, Option<String>), String> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Err(parse_key_provider("").unwrap_err());
    }
    let (name, rest) = match arg.split_once(char::is_whitespace) {
        Some((name, rest)) => (name, rest.trim()),
        None => (arg, ""),
    };
    let provider = parse_key_provider(name)?;
    let key = if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    };
    Ok((provider, key))
}

pub async fn check_api_key(provider: ProviderName, api_key: &str, base_url: &str) -> KeyCheck {
    if matches!(provider, ProviderName::Ollama) {
        return KeyCheck::Accepted;
    }
    let base_url = base_url.trim_end_matches('/');
    match provider {
        ProviderName::Anthropic => {
            let p = AnthropicProvider::new(base_url.to_string(), api_key.to_string());
            KeyCheck::from_list_models(p.list_models().await)
        }
        _ => {
            let p = OpenAiProvider::new(base_url.to_string(), api_key.to_string());
            KeyCheck::from_list_models(p.list_models().await)
        }
    }
}

/// Probe then upsert. Rejected keys never touch `config_path`.
pub async fn apply_pasted_key(
    config: &mut Config,
    provider: ProviderName,
    api_key: &str,
    base_url: Option<&str>,
    config_path: &Path,
) -> Result<KeyCheck> {
    let base = base_url.unwrap_or_else(|| provider.default_base_url());
    let check = check_api_key(provider, api_key, base).await;
    if matches!(check, KeyCheck::Rejected(_)) {
        return Ok(check);
    }
    upsert_key_profile(config, provider, api_key, config_path)?;
    Ok(check)
}

fn upsert_key_profile(
    config: &mut Config,
    provider: ProviderName,
    api_key: &str,
    config_path: &Path,
) -> Result<()> {
    let name = provider.as_str().to_string();
    let profile = Profile {
        provider: Some(provider),
        model: provider.default_model().map(str::to_string),
        api_key: Some(api_key.to_string()),
        ..Default::default()
    };
    upsert_profile_as_default(config, &name, profile, Some(config_path))
}

/// `hi auth <provider>` — paste a key, probe it, write the matching profile.
pub async fn run_cli(args: &[String]) -> Result<()> {
    let provider = match args.first().map(String::as_str) {
        Some(name) => parse_key_provider(name).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => bail!("usage: hi auth openai|anthropic|xai"),
    };
    let key = if let Some(key) = args.get(1).filter(|s| !s.is_empty()) {
        key.clone()
    } else {
        read_secret_line(&format!("Paste your {} API key: ", provider.as_str()))?
    };
    if key.is_empty() {
        bail!("no API key entered");
    }
    let path = default_config_path().context("could not determine config directory")?;
    let mut config = if path.exists() {
        read_config_file(&path)?
    } else {
        Config::default()
    };
    match apply_pasted_key(&mut config, provider, &key, None, &path).await? {
        KeyCheck::Accepted => {
            println!("Saved {} profile to {}", provider.as_str(), path.display());
        }
        KeyCheck::Unverified(msg) => {
            println!(
                "Saved {} profile to {} (could not verify: {msg})",
                provider.as_str(),
                path.display()
            );
        }
        KeyCheck::Rejected(msg) => {
            bail!("refused to save: {msg}");
        }
    }
    Ok(())
}

pub fn read_secret_line(message: &str) -> Result<String> {
    eprint!("{message}");
    io::stderr().flush().ok();
    if !io::stdin().is_terminal() {
        let mut line = String::new();
        if io::stdin()
            .read_line(&mut line)
            .context("reading API key")?
            == 0
        {
            bail!("auth cancelled");
        }
        return Ok(line.trim().to_string());
    }
    #[cfg(unix)]
    {
        read_secret_line_unix()
    }
    #[cfg(not(unix))]
    {
        let mut line = String::new();
        if io::stdin()
            .read_line(&mut line)
            .context("reading API key")?
            == 0
        {
            bail!("auth cancelled");
        }
        Ok(line.trim().to_string())
    }
}

#[cfg(unix)]
fn read_secret_line_unix() -> Result<String> {
    let fd = libc::STDIN_FILENO;
    let mut orig = std::mem::MaybeUninit::<libc::termios>::uninit();
    let restored = unsafe {
        if libc::tcgetattr(fd, orig.as_mut_ptr()) == 0 {
            let orig = orig.assume_init();
            let mut silent = orig;
            silent.c_lflag &= !libc::ECHO;
            libc::tcsetattr(fd, libc::TCSANOW, &silent);
            Some(orig)
        } else {
            None
        }
    };
    let mut line = String::new();
    let n = io::stdin().read_line(&mut line);
    if let Some(orig) = restored {
        unsafe {
            libc::tcsetattr(fd, libc::TCSANOW, &orig);
        }
        eprintln!();
    }
    if n.context("reading API key")? == 0 {
        bail!("auth cancelled");
    }
    Ok(line.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::test_support::{FakeOpenAiServer, Response};

    #[tokio::test]
    async fn rejected_401_does_not_write_config() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(401, r#"{"error":"bad"}"#)])
        else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut config = Config::default();
        let check = apply_pasted_key(
            &mut config,
            ProviderName::Openai,
            "sk-bad",
            Some(server.url()),
            &path,
        )
        .await
        .unwrap();
        assert!(matches!(check, KeyCheck::Rejected(_)), "{check:?}");
        assert!(!path.exists(), "401 must not create a profile file");
        assert!(config.profiles.is_empty());
    }

    #[tokio::test]
    async fn accepted_200_upserts_profile() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(
            200,
            r#"{"data":[{"id":"test-model"}]}"#,
        )]) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut config = Config::default();
        let check = apply_pasted_key(
            &mut config,
            ProviderName::Openai,
            "sk-good",
            Some(server.url()),
            &path,
        )
        .await
        .unwrap();
        assert_eq!(check, KeyCheck::Accepted);
        let saved = read_config_file(&path).unwrap();
        let profile = saved.profiles.get("openai").expect("openai profile");
        assert_eq!(profile.api_key.as_deref(), Some("sk-good"));
        assert_eq!(saved.default_profile.as_deref(), Some("openai"));
    }
}
