//! Credential-reference validation, storage, migration, and resolution.
//!
//! Legacy literal and environment fields remain readable, but production
//! persistence seals them into `auth-store://` or `env://` references.

use super::*;

/// Repair the one unambiguous legacy shape where `api_key_env` contains a
/// pasted literal rather than an environment-variable name.
pub(crate) fn migrate_api_key_env_to_literal(config: &mut Config, _path: &Path) {
    for profile in config.profiles.values_mut() {
        let Some(env_name) = profile.api_key_env.clone() else {
            continue;
        };
        if profile.api_key.is_some()
            || looks_like_env_var_name(&env_name)
            || std::env::vars_os()
                .any(|(name, _)| name.as_os_str() == std::ffi::OsStr::new(&env_name))
        {
            continue;
        }
        profile.api_key_env = None;
        profile.api_key = Some(env_name);
    }
}

pub(super) fn supported_credential_reference(reference: &str) -> bool {
    reference
        .strip_prefix("auth-store://")
        .or_else(|| reference.strip_prefix("env://"))
        .is_some_and(|target| !target.is_empty() && !target.chars().any(char::is_control))
}

fn validate_credential_reference(reference: &str) -> Result<()> {
    let layer = hi_workspace::SettingLayer {
        source: hi_workspace::SettingSource::OneShot,
        values: std::collections::BTreeMap::from([(
            hi_workspace::PROVIDER_API_KEY_CREDENTIAL.to_string(),
            hi_workspace::SettingValue::CredentialRef(reference.to_string()),
        )]),
    };
    hi_workspace::standard_harness_settings()
        .resolve(hi_workspace::PROVIDER_API_KEY_CREDENTIAL, &[layer], true)
        .context("validating typed credential setting")?;
    if !supported_credential_reference(reference) {
        bail!("credential reference must use auth-store:// or env:// and name a target");
    }
    Ok(())
}

pub(super) fn seal_credential_fields(
    namespace: &str,
    identity: &str,
    path: &Path,
    reference: Option<String>,
    literal: Option<String>,
    environment: Option<String>,
) -> Result<(Option<String>, Option<String>, Option<String>)> {
    if let Some(reference) = reference {
        validate_credential_reference(&reference)?;
        return Ok((Some(reference), None, None));
    }
    if let Some(secret) = literal {
        let binding = format!("{}\0{identity}", path.to_string_lossy());
        let key = format!("{namespace}/{}", blake3::hash(binding.as_bytes()).to_hex());
        hi_ai::auth_store::save(&key, &hi_ai::StoredToken::static_access(secret))
            .context("saving API key in the private credential store")?;
        return Ok((Some(format!("auth-store://{key}")), None, None));
    }
    if let Some(environment) = environment {
        let reference = format!("env://{environment}");
        validate_credential_reference(&reference)?;
        return Ok((Some(reference), None, None));
    }
    Ok((None, None, None))
}

pub(super) fn seal_profile_credential(
    name: &str,
    path: &Path,
    mut profile: Profile,
) -> Result<Profile> {
    let namespace = format!(
        "profile-api-key/{}",
        profile.provider.unwrap_or(ProviderName::Openai).as_str()
    );
    (profile.api_key_ref, profile.api_key, profile.api_key_env) = seal_credential_fields(
        &namespace,
        name,
        path,
        profile.api_key_ref.take(),
        profile.api_key.take(),
        profile.api_key_env.take(),
    )?;
    Ok(profile)
}

pub(super) fn rmw_profile_file(
    path: &Path,
    name: &str,
    fallback: Profile,
    mutate: impl FnOnce(&mut Profile),
) -> Result<Profile> {
    let mut file = if path.exists() {
        read_config_file(path)?
    } else {
        Config::default()
    };
    let mut profile = file.profiles.remove(name).unwrap_or(fallback);
    mutate(&mut profile);
    let profile = seal_profile_credential(name, path, profile)?;
    super::profile_edit::validate_profile(&profile)?;
    file.profiles.insert(name.to_string(), profile.clone());
    save_config_to(&file, path)?;
    Ok(profile)
}

pub(super) fn migrate_persisted_credentials(config: &Config, path: &Path) {
    let mut persisted = config.clone();
    let mut changed = false;
    for (name, profile) in &mut persisted.profiles {
        if profile.api_key.is_none() && profile.api_key_env.is_none() {
            continue;
        }
        if let Ok(sealed) = seal_profile_credential(name, path, profile.clone()) {
            *profile = sealed;
            changed = true;
        }
    }
    for (namespace, identity, section) in [
        (
            "config-api-key/sync",
            "sync",
            persisted.sync.as_mut().map(section_fields),
        ),
        (
            "config-api-key/rsi",
            "rsi",
            persisted.rsi.as_mut().map(section_fields),
        ),
        (
            "config-api-key/outcome",
            "outcome",
            persisted.outcome.as_mut().map(section_fields),
        ),
    ] {
        if let Some((reference, literal, environment)) = section {
            changed |= migrate_section_credential(
                namespace,
                identity,
                path,
                reference,
                literal,
                environment,
            );
        }
    }
    if changed {
        let _ = save_config_to(&persisted, path);
    }
}

trait CredentialSection {
    fn credential_fields(
        &mut self,
    ) -> (
        &mut Option<String>,
        &mut Option<String>,
        &mut Option<String>,
    );
}

macro_rules! credential_section {
    ($type:ty) => {
        impl CredentialSection for $type {
            fn credential_fields(
                &mut self,
            ) -> (
                &mut Option<String>,
                &mut Option<String>,
                &mut Option<String>,
            ) {
                (
                    &mut self.api_key_ref,
                    &mut self.api_key,
                    &mut self.api_key_env,
                )
            }
        }
    };
}

credential_section!(SyncSection);
credential_section!(RsiSection);
credential_section!(OutcomeSection);

fn section_fields<T: CredentialSection>(
    section: &mut T,
) -> (
    &mut Option<String>,
    &mut Option<String>,
    &mut Option<String>,
) {
    section.credential_fields()
}

fn migrate_section_credential(
    namespace: &str,
    identity: &str,
    path: &Path,
    reference: &mut Option<String>,
    literal: &mut Option<String>,
    environment: &mut Option<String>,
) -> bool {
    if literal.is_none() && environment.is_none() {
        return false;
    }
    let Ok((sealed_ref, sealed_literal, sealed_environment)) = seal_credential_fields(
        namespace,
        identity,
        path,
        reference.clone(),
        literal.clone(),
        environment.clone(),
    ) else {
        return false;
    };
    *reference = sealed_ref;
    *literal = sealed_literal;
    *environment = sealed_environment;
    true
}

pub(crate) fn resolve_credential_reference(
    reference: &str,
    project_local: bool,
    project_trusted: bool,
) -> Result<String> {
    validate_credential_reference(reference)?;
    if let Some(key) = reference.strip_prefix("auth-store://") {
        if project_local && !project_trusted {
            bail!("untrusted project configuration cannot resolve credential-store references");
        }
        return hi_ai::auth_store::load(key)
            .map(|credential| credential.access)
            .ok_or_else(|| anyhow!("credential reference {reference:?} was not found"));
    }
    let environment = reference
        .strip_prefix("env://")
        .expect("validated credential reference has a supported scheme");
    if project_local {
        bail!("project configuration cannot read credential environment variable '{environment}'");
    }
    std::env::var(environment)
        .map_err(|_| anyhow!("credential environment variable {environment} is not set"))
}
