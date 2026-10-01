//! Startup recovery when a profile's saved API key is missing from the
//! private credential store.
//!
//! `config.toml` holds only an `auth-store://…` reference; the key itself
//! lives in `~/.config/hi/auth.json`. When the store no longer has that entry
//! (`hi logout`, a `config.toml` copied to a new machine, a store rewritten
//! from under it), the profile is intact but unusable, and failing on the bare
//! reference leaves the user to work out what it means. On a terminal hi asks
//! for the key again and seals it under the same reference, so the profile
//! keeps its name, model, and every other setting; a headless run is told the
//! commands that do the same.

use std::io::{self, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use hi_ai::KeyCheck;

use crate::auth::{check_api_key, read_secret_line};
use crate::config::{
    Config, MissingStoredCredential, ProviderName, repoint_profile_credential,
    reseal_profile_api_key,
};

/// The profile whose stored credential a startup error says is missing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingKey {
    pub profile: String,
    pub provider: ProviderName,
    pub reference: String,
}

impl MissingKey {
    /// Match `err` against the loaded config. The failing reference must be
    /// some profile's `api_key_ref`; when several profiles share it (login
    /// profiles all point at `auth-store://pipenetwork`), the one this run
    /// selected wins, then the first by name.
    pub fn from_error(
        cli_profile: Option<&str>,
        file: &Config,
        err: &anyhow::Error,
    ) -> Option<Self> {
        let missing = err.downcast_ref::<MissingStoredCredential>()?;
        let selected = cli_profile.or(file.default_profile.as_deref());
        let mut carrying: Vec<(&String, &crate::config::Profile)> = file
            .profiles
            .iter()
            .filter(|(_, profile)| {
                profile.api_key_ref.as_deref() == Some(missing.reference.as_str())
            })
            .collect();
        carrying.sort_by_key(|(name, _)| *name);
        let (name, profile) = carrying
            .iter()
            .find(|(name, _)| Some(name.as_str()) == selected)
            .or_else(|| carrying.first())?;
        Some(Self {
            profile: (*name).clone(),
            provider: profile.provider.unwrap_or(ProviderName::Openai),
            reference: missing.reference.clone(),
        })
    }

    /// What happened, in terms of the profile rather than the reference.
    pub fn explanation(&self) -> String {
        format!(
            "The saved API key for profile '{}' ({}) is missing from {}.\n\
             config.toml still points at it ({}), so the profile is intact but has no\n\
             credential. This happens after `hi logout`, when config.toml is copied to a new\n\
             machine, or when the credential store is rewritten. Re-entering the key restores\n\
             the profile as it was.",
            self.profile,
            self.provider.as_str(),
            store_display(),
            self.reference,
        )
    }

    /// The fix for a run that cannot ask.
    pub fn guidance(&self) -> String {
        let provider = self.provider.as_str();
        let mut lines = vec![format!(
            "The saved API key for profile '{}' ({provider}) is missing from {}.",
            self.profile,
            store_display()
        )];
        lines.push(format!(
            "Run `hi` on a terminal to re-enter it and keep the profile as it is, or \
             `hi auth {provider}` to paste a key into a new '{provider}' profile."
        ));
        if self.provider == ProviderName::Pipenetwork {
            lines.push("`hi login pipenetwork` (browser pairing) does the same.".into());
        }
        lines.join("\n")
    }
}

/// Explain the loss and collect the key again. Browser pairing is offered for
/// Pipe Network; pasting a key works for every provider with a key flow. The
/// profile is rewritten in place (`explicit_config` is `--config` when given).
pub async fn recover(
    file: &mut Config,
    missing: &MissingKey,
    explicit_config: Option<&Path>,
) -> Result<()> {
    eprintln!("{}\n", missing.explanation());
    let pairing = missing.provider == ProviderName::Pipenetwork && {
        eprintln!("  1) Browser pairing   (default)  opens pipenetwork.ai and stores a new key");
        eprintln!("  2) Paste an API key\n");
        loop {
            match prompt("How would you like to re-authenticate? [1-2] (default 1): ")?.trim() {
                "" | "1" => break true,
                "2" => break false,
                other => eprintln!("  '{other}' isn't a choice — pick 1 or 2."),
            }
        }
    };
    if pairing {
        hi_ai::pipenetwork_auth::login().await?;
        if !hi_ai::pipenetwork_auth::has_credential() {
            bail!("sign-in reported success but stored no credential");
        }
        let reference = format!("auth-store://{}", hi_ai::pipenetwork_auth::PROVIDER_ID);
        repoint_profile_credential(file, &missing.profile, &reference, explicit_config)?;
    } else {
        let key = read_secret_line(&format!(
            "Paste your {} API key: ",
            missing.provider.as_str()
        ))?;
        if key.is_empty() {
            bail!("no API key entered");
        }
        let base_url = file
            .profiles
            .get(&missing.profile)
            .and_then(|profile| profile.base_url.clone())
            .unwrap_or_else(|| missing.provider.default_base_url().to_string());
        eprint!("\x1b[2m  testing connection…\x1b[0m\r");
        io::stderr().flush().ok();
        match check_api_key(missing.provider, &key, &base_url).await {
            KeyCheck::Accepted => eprintln!("\x1b[2m  ✓ connection verified\x1b[0m"),
            KeyCheck::Rejected(msg) => {
                bail!("key rejected: {msg} — a 401/403 key is never saved")
            }
            KeyCheck::Unverified(msg) => eprintln!(
                "\x1b[33m  ⚠ couldn't verify the connection: {msg}\x1b[0m — saving anyway; \
                 hi retries on the first turn"
            ),
        }
        reseal_profile_api_key(file, &missing.profile, &key, explicit_config)?;
    }
    eprintln!("Restored the key for profile '{}'.\n", missing.profile);
    Ok(())
}

fn store_display() -> String {
    hi_ai::auth_store::auth_path()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "~/.config/hi/auth.json".to_string())
}

fn prompt(message: &str) -> Result<String> {
    eprint!("{message}");
    io::stderr().flush().ok();
    let mut line = String::new();
    if io::stdin().read_line(&mut line).context("reading input")? == 0 {
        bail!("re-authentication cancelled");
    }
    Ok(line)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Profile;

    const REFERENCE: &str = "auth-store://profile-api-key/pipenetwork/abc123";

    fn config_with(profiles: &[(&str, &str, Option<&str>)], default: Option<&str>) -> Config {
        let mut config = Config {
            default_profile: default.map(str::to_string),
            ..Default::default()
        };
        for (name, provider, reference) in profiles {
            config.profiles.insert(
                (*name).into(),
                Profile {
                    provider: Some(provider.parse().unwrap()),
                    model: Some("ipop/coder-balanced".into()),
                    api_key_ref: reference.map(str::to_string),
                    ..Default::default()
                },
            );
        }
        config
    }

    fn missing(reference: &str) -> anyhow::Error {
        anyhow::Error::new(MissingStoredCredential {
            reference: reference.into(),
        })
    }

    #[test]
    fn the_profile_carrying_the_failed_reference_is_found_through_context() {
        let config = config_with(
            &[
                ("default", "pipenetwork", Some(REFERENCE)),
                (
                    "work",
                    "openai",
                    Some("auth-store://profile-api-key/openai/def"),
                ),
            ],
            Some("default"),
        );
        let err = missing(REFERENCE).context("resolving the default profile");
        assert_eq!(
            MissingKey::from_error(None, &config, &err),
            Some(MissingKey {
                profile: "default".into(),
                provider: ProviderName::Pipenetwork,
                reference: REFERENCE.into(),
            })
        );
    }

    #[test]
    fn a_shared_reference_prefers_the_selected_profile_then_the_first_by_name() {
        let shared = "auth-store://pipenetwork";
        let config = config_with(
            &[
                ("zeta", "pipenetwork", Some(shared)),
                ("alpha", "pipenetwork", Some(shared)),
                ("other", "openai", None),
            ],
            Some("zeta"),
        );
        let err = missing(shared);
        assert_eq!(
            MissingKey::from_error(Some("zeta"), &config, &err)
                .unwrap()
                .profile,
            "zeta"
        );
        let no_default = Config {
            default_profile: None,
            ..config.clone()
        };
        assert_eq!(
            MissingKey::from_error(None, &no_default, &err)
                .unwrap()
                .profile,
            "alpha"
        );
    }

    #[test]
    fn other_errors_and_references_no_profile_carries_are_not_recoverable() {
        let config = config_with(&[("default", "pipenetwork", Some(REFERENCE))], None);
        assert_eq!(
            MissingKey::from_error(None, &config, &anyhow::anyhow!("profile 'x' not found")),
            None
        );
        assert_eq!(
            MissingKey::from_error(None, &config, &missing("auth-store://elsewhere")),
            None,
            "a reference nothing in the config uses cannot be re-keyed"
        );
    }

    #[test]
    fn guidance_names_the_profile_and_the_commands() {
        let key = MissingKey {
            profile: "default".into(),
            provider: ProviderName::Pipenetwork,
            reference: REFERENCE.into(),
        };
        let guidance = key.guidance();
        assert!(
            guidance.contains("profile 'default' (pipenetwork)"),
            "{guidance}"
        );
        assert!(guidance.contains("Run `hi` on a terminal"), "{guidance}");
        assert!(guidance.contains("`hi auth pipenetwork`"), "{guidance}");
        assert!(guidance.contains("`hi login pipenetwork`"), "{guidance}");
        assert!(key.explanation().contains(REFERENCE));

        let openai = MissingKey {
            provider: ProviderName::Openai,
            ..key
        };
        assert!(
            !openai.guidance().contains("hi login"),
            "pairing is Pipe-only"
        );
    }
}
