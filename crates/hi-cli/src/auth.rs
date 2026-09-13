//! Fail-closed API-key write: probe `/models` before touching `config.toml`.
//!
//! `hi auth <provider>` and `/auth` paste an API key. `/login` remains the
//! distinct subscription-pairing flow (xAI / pipenetwork / x402). HTTP 401/403
//! never writes a profile; transport failures are unverified and may still
//! save with a warning.

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
        "pipenetwork" | "pipe" => Ok(ProviderName::Pipenetwork),
        "xai" | "grok" => Ok(ProviderName::Xai),
        "ollama" | "local" | "x402" => Err(format!(
            "'{name}' is not a pasted-key provider. Use openai, anthropic, pipenetwork, or xai. \
             Subscription pairing stays /login xai | /login pipenetwork | /login x402."
        )),
        "" => Err(
            "usage: /auth openai|anthropic|pipenetwork|xai [api-key]  (or `hi auth <provider>`)"
                .into(),
        ),
        other => Err(format!(
            "'{other}' has no pasted-key flow. Supported: openai, anthropic, pipenetwork, xai. \
             Subscription pairing stays /login."
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

/// Profile name written by browser pairing (`hi login pipenetwork`).
pub const LOGIN_PROFILE: &str = "pipenetwork";

fn pipenetwork_login_profile(existing: Option<&Profile>) -> Profile {
    let mut profile = existing.cloned().unwrap_or_default();
    profile.provider = Some(ProviderName::Pipenetwork);
    if profile
        .model
        .as_deref()
        .map(str::trim)
        .unwrap_or("")
        .is_empty()
    {
        profile.model = ProviderName::Pipenetwork
            .default_model()
            .map(str::to_string);
    }
    profile.api_key_ref = Some(format!(
        "auth-store://{}",
        hi_ai::pipenetwork_auth::PROVIDER_ID
    ));
    profile.api_key = None;
    profile.api_key_env = None;
    profile
}

/// Point `[profiles.pipenetwork]` at the pairing key in `auth.json` and select
/// it as the default profile.
pub fn install_pipenetwork_login_profile_at(config: &mut Config, config_path: &Path) -> Result<()> {
    let profile = pipenetwork_login_profile(config.profiles.get(LOGIN_PROFILE));
    upsert_profile_as_default(config, LOGIN_PROFILE, profile, Some(config_path))
}

/// Write the login profile to the user config path (`~/.config/hi/config.toml`).
pub fn install_pipenetwork_login_profile(config: &mut Config) -> Result<std::path::PathBuf> {
    let path = default_config_path().context("could not determine config directory")?;
    install_pipenetwork_login_profile_at(config, &path)?;
    Ok(path)
}

/// `hi login [pipenetwork]` — browser pairing, then write the minted key into
/// `config.toml` so the next session uses it without pasting.
pub async fn run_login_cli(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        None | Some("pipenetwork") | Some("pipe") => {}
        Some("-h" | "--help" | "help") => {
            println!("usage: hi login pipenetwork");
            println!(
                "  Opens a browser pairing flow, stores the API key, and writes\n  \
                 [profiles.pipenetwork] to ~/.config/hi/config.toml."
            );
            return Ok(());
        }
        Some(other) => bail!(
            "usage: hi login pipenetwork\n\
             '{other}' has no CLI login; this build signs in to pipenetwork.ai."
        ),
    }
    hi_ai::pipenetwork_auth::login().await?;
    if !hi_ai::pipenetwork_auth::has_credential() {
        bail!("sign-in reported success but stored no credential");
    }
    let path = default_config_path().context("could not determine config directory")?;
    let mut config = if path.exists() {
        read_config_file(&path)?
    } else {
        Config::default()
    };
    install_pipenetwork_login_profile_at(&mut config, &path)?;
    println!(
        "Configured pipenetwork profile in {} (api_key_ref = \"auth-store://pipenetwork\")",
        path.display()
    );
    Ok(())
}

/// `hi logout [pipenetwork]` — drop the stored pairing key. The profile stays
/// so the next `hi login pipenetwork` reuses it.
pub fn run_logout_cli(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        None | Some("pipenetwork") | Some("pipe") => hi_ai::pipenetwork_auth::logout(),
        Some("-h" | "--help" | "help") => {
            println!("usage: hi logout pipenetwork");
            Ok(())
        }
        Some(other) => bail!("usage: hi logout pipenetwork (got '{other}')"),
    }
}

/// `hi auth <provider>` — paste a key, probe it, write the matching profile.
pub async fn run_cli(args: &[String]) -> Result<()> {
    let provider = match args.first().map(String::as_str) {
        Some(name) => parse_key_provider(name).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => bail!("usage: hi auth openai|anthropic|pipenetwork|xai"),
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

    #[test]
    fn pipenetwork_api_keys_use_the_pasted_key_flow() {
        assert_eq!(
            parse_key_provider("pipenetwork"),
            Ok(ProviderName::Pipenetwork)
        );
        assert_eq!(parse_key_provider("pipe"), Ok(ProviderName::Pipenetwork));
        assert_eq!(
            split_auth_arg("pipe api_test"),
            Ok((ProviderName::Pipenetwork, Some("api_test".into())))
        );
    }

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
        assert!(
            profile.api_key.is_none(),
            "literal key must not be persisted"
        );
        assert!(profile.api_key_env.is_none());
        let reference = profile
            .api_key_ref
            .as_deref()
            .expect("credential-store reference");
        let key = reference
            .strip_prefix("auth-store://")
            .expect("private credential-store reference");
        assert_eq!(
            hi_ai::auth_store::load(key).map(|credential| credential.access),
            Some("sk-good".into())
        );
        assert!(!std::fs::read_to_string(&path).unwrap().contains("sk-good"));
        hi_ai::auth_store::delete(key).unwrap();
        assert_eq!(saved.default_profile.as_deref(), Some("openai"));
    }

    #[tokio::test]
    async fn accepted_pipenetwork_key_uses_the_flash_profile_and_private_store() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(
            200,
            r#"{"data":[{"id":"pipe/deepseek-v4-flash-0731"}]}"#,
        )]) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut config = Config::default();
        let check = apply_pasted_key(
            &mut config,
            ProviderName::Pipenetwork,
            "api_test",
            Some(server.url()),
            &path,
        )
        .await
        .unwrap();
        assert_eq!(check, KeyCheck::Accepted);

        let saved = read_config_file(&path).unwrap();
        let profile = saved
            .profiles
            .get("pipenetwork")
            .expect("pipenetwork profile");
        assert_eq!(
            profile.model.as_deref(),
            Some("pipe/deepseek-v4-flash-0731")
        );
        assert!(profile.api_key.is_none());
        let reference = profile
            .api_key_ref
            .as_deref()
            .expect("private credential-store reference");
        let key = reference
            .strip_prefix("auth-store://")
            .expect("auth-store reference");
        assert_eq!(
            hi_ai::auth_store::load(key).map(|credential| credential.access),
            Some("api_test".into())
        );
        assert!(!std::fs::read_to_string(&path).unwrap().contains("api_test"));
        hi_ai::auth_store::delete(key).unwrap();
        assert_eq!(saved.default_profile.as_deref(), Some("pipenetwork"));
    }

    #[test]
    fn login_profile_points_at_auth_store_pipenetwork() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut config = Config::default();
        install_pipenetwork_login_profile_at(&mut config, &path).unwrap();
        let saved = read_config_file(&path).unwrap();
        let profile = saved
            .profiles
            .get("pipenetwork")
            .expect("pipenetwork profile");
        assert_eq!(profile.provider, Some(ProviderName::Pipenetwork));
        assert_eq!(
            profile.model.as_deref(),
            Some("pipe/deepseek-v4-flash-0731")
        );
        assert_eq!(
            profile.api_key_ref.as_deref(),
            Some("auth-store://pipenetwork")
        );
        assert!(profile.api_key.is_none());
        assert!(profile.api_key_env.is_none());
        assert_eq!(saved.default_profile.as_deref(), Some("pipenetwork"));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            !text.contains("pk_live"),
            "pairing key must not land in config.toml: {text}"
        );
    }

    #[test]
    fn login_profile_keeps_an_existing_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut config = Config::default();
        config.profiles.insert(
            "pipenetwork".into(),
            Profile {
                provider: Some(ProviderName::Pipenetwork),
                model: Some("pipe/custom-model".into()),
                api_key: Some("pk_old".into()),
                ..Default::default()
            },
        );
        install_pipenetwork_login_profile_at(&mut config, &path).unwrap();
        let saved = read_config_file(&path).unwrap();
        let profile = saved.profiles.get("pipenetwork").unwrap();
        assert_eq!(profile.model.as_deref(), Some("pipe/custom-model"));
        assert_eq!(
            profile.api_key_ref.as_deref(),
            Some("auth-store://pipenetwork")
        );
        assert!(profile.api_key.is_none());
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("pk_old"), "old literal leaked: {text}");
    }

    fn run_with_isolated_credentials(test: &str) -> bool {
        const CHILD: &str = "HI_TEST_PIPE_LOGIN_PROFILE";
        if std::env::var(CHILD).as_deref() == Ok(test) {
            return false;
        }
        let config_home = tempfile::tempdir().unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture"])
            .env(CHILD, test)
            .env("XDG_CONFIG_HOME", config_home.path())
            .status()
            .expect("run login-profile resolution in an isolated process");
        assert!(
            status.success(),
            "isolated login-profile regression failed: {status}"
        );
        true
    }

    #[test]
    fn login_profile_resolves_the_pairing_key() {
        let test = std::thread::current().name().unwrap().to_string();
        if run_with_isolated_credentials(&test) {
            return;
        }
        hi_ai::auth_store::save(
            hi_ai::pipenetwork_auth::PROVIDER_ID,
            &hi_ai::StoredToken::static_access("pk_live_from_login".into()),
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut config = Config::default();
        install_pipenetwork_login_profile_at(&mut config, &path).unwrap();
        let saved = read_config_file(&path).unwrap();
        let key = crate::config::resolve_api_key_for_endpoint(
            saved.profiles.get("pipenetwork"),
            ProviderName::Pipenetwork,
            "https://api.pipenetwork.ai/v1",
            true,
            false,
        )
        .unwrap();
        assert_eq!(key, "pk_live_from_login");
        hi_ai::auth_store::delete(hi_ai::pipenetwork_auth::PROVIDER_ID).unwrap();
    }
}
