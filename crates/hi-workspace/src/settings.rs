use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SETTING_SPEC_SCHEMA_VERSION: u16 = 1;
pub const PROVIDER_API_KEY_CREDENTIAL: &str = "credentials.provider_api_key";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingKind {
    Boolean,
    Integer,
    String,
    DurationMillis,
    CredentialRef,
    StringList,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum SettingValue {
    Boolean(bool),
    Integer(i64),
    String(String),
    DurationMillis(u64),
    CredentialRef(String),
    StringList(Vec<String>),
}

impl SettingValue {
    pub fn kind(&self) -> SettingKind {
        match self {
            Self::Boolean(_) => SettingKind::Boolean,
            Self::Integer(_) => SettingKind::Integer,
            Self::String(_) => SettingKind::String,
            Self::DurationMillis(_) => SettingKind::DurationMillis,
            Self::CredentialRef(_) => SettingKind::CredentialRef,
            Self::StringList(_) => SettingKind::StringList,
        }
    }

    fn numeric(&self) -> Option<u64> {
        match self {
            Self::Integer(value) => u64::try_from(*value).ok(),
            Self::DurationMillis(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettingSpec {
    pub schema_version: u16,
    pub key: String,
    pub kind: SettingKind,
    pub default: Option<SettingValue>,
    pub secret: bool,
    pub minimum: Option<u64>,
    pub maximum: Option<u64>,
    pub description: String,
}

impl SettingSpec {
    pub fn new(
        key: impl Into<String>,
        kind: SettingKind,
        default: Option<SettingValue>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: SETTING_SPEC_SCHEMA_VERSION,
            key: key.into(),
            kind,
            default,
            secret: false,
            minimum: None,
            maximum: None,
            description: description.into(),
        }
    }

    pub fn bounded(mut self, minimum: u64, maximum: u64) -> Self {
        self.minimum = Some(minimum);
        self.maximum = Some(maximum);
        self
    }

    pub fn secret(mut self) -> Self {
        self.secret = true;
        self.kind = SettingKind::CredentialRef;
        self.default = None;
        self
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingSource {
    BuiltIn,
    #[default]
    Profile,
    TrustedWorkspace,
    Session,
    OneShot,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettingLayer {
    pub source: SettingSource,
    pub values: BTreeMap<String, SettingValue>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedSetting {
    pub key: String,
    pub value: SettingValue,
    pub source: SettingSource,
}

#[derive(Clone, Debug, Default)]
pub struct SettingRegistry {
    specs: BTreeMap<String, SettingSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SettingError {
    #[error("setting key is empty or invalid: {0:?}")]
    InvalidKey(String),
    #[error("setting {0:?} is already registered")]
    Duplicate(String),
    #[error("unknown setting {0:?}")]
    Unknown(String),
    #[error("setting {key:?} requires {expected:?}, received {actual:?}")]
    WrongType {
        key: String,
        expected: SettingKind,
        actual: SettingKind,
    },
    #[error("secret setting {0:?} must contain a credential reference")]
    LiteralSecret(String),
    #[error("setting {key:?} value {value} is outside {minimum:?}..={maximum:?}")]
    OutOfRange {
        key: String,
        value: u64,
        minimum: Option<u64>,
        maximum: Option<u64>,
    },
    #[error("setting {0:?} has no configured or built-in value")]
    Missing(String),
}

impl SettingRegistry {
    pub fn register(&mut self, spec: SettingSpec) -> Result<(), SettingError> {
        validate_key(&spec.key)?;
        if spec.schema_version != SETTING_SPEC_SCHEMA_VERSION {
            return Err(SettingError::InvalidKey(spec.key));
        }
        if let Some(default) = &spec.default {
            validate_value(&spec, default)?;
        }
        if self.specs.contains_key(&spec.key) {
            return Err(SettingError::Duplicate(spec.key));
        }
        self.specs.insert(spec.key.clone(), spec);
        Ok(())
    }

    pub fn spec(&self, key: &str) -> Option<&SettingSpec> {
        self.specs.get(key)
    }

    /// Resolve with fixed precedence. Untrusted workspace layers are ignored,
    /// never partially accepted.
    pub fn resolve(
        &self,
        key: &str,
        layers: &[SettingLayer],
        workspace_trusted: bool,
    ) -> Result<ResolvedSetting, SettingError> {
        let spec = self
            .specs
            .get(key)
            .ok_or_else(|| SettingError::Unknown(key.to_owned()))?;
        let mut selected = spec
            .default
            .clone()
            .map(|value| (SettingSource::BuiltIn, value));
        for layer in layers {
            if layer.source == SettingSource::BuiltIn
                || (layer.source == SettingSource::TrustedWorkspace && !workspace_trusted)
            {
                continue;
            }
            if let Some(value) = layer.values.get(key) {
                validate_value(spec, value)?;
                if selected
                    .as_ref()
                    .is_none_or(|(source, _)| layer.source > *source)
                {
                    selected = Some((layer.source, value.clone()));
                }
            }
        }
        let (source, value) = selected.ok_or_else(|| SettingError::Missing(key.to_owned()))?;
        Ok(ResolvedSetting {
            key: key.to_owned(),
            value,
            source,
        })
    }

    pub fn specs(&self) -> impl Iterator<Item = &SettingSpec> {
        self.specs.values()
    }

    /// Validate every entry in a layer, including keys a particular consumer
    /// may not read yet. This prevents a persisted layer from smuggling an
    /// unknown or ill-typed value that would become active after an upgrade.
    pub fn validate_layer(&self, layer: &SettingLayer) -> Result<(), SettingError> {
        for (key, value) in &layer.values {
            let spec = self
                .specs
                .get(key)
                .ok_or_else(|| SettingError::Unknown(key.clone()))?;
            validate_value(spec, value)?;
        }
        Ok(())
    }
}

fn validate_key(key: &str) -> Result<(), SettingError> {
    let valid = !key.is_empty()
        && key.len() <= 128
        && key.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'.' || byte == b'_'
        });
    if valid {
        Ok(())
    } else {
        Err(SettingError::InvalidKey(key.to_owned()))
    }
}

fn validate_value(spec: &SettingSpec, value: &SettingValue) -> Result<(), SettingError> {
    if value.kind() != spec.kind {
        return Err(SettingError::WrongType {
            key: spec.key.clone(),
            expected: spec.kind,
            actual: value.kind(),
        });
    }
    if spec.secret
        && !matches!(value, SettingValue::CredentialRef(reference) if valid_credential_reference(reference))
    {
        return Err(SettingError::LiteralSecret(spec.key.clone()));
    }
    if let Some(number) = value.numeric()
        && (spec.minimum.is_some_and(|minimum| number < minimum)
            || spec.maximum.is_some_and(|maximum| number > maximum))
    {
        return Err(SettingError::OutOfRange {
            key: spec.key.clone(),
            value: number,
            minimum: spec.minimum,
            maximum: spec.maximum,
        });
    }
    Ok(())
}

fn valid_credential_reference(reference: &str) -> bool {
    let Some((scheme, target)) = reference.trim().split_once("://") else {
        return false;
    };
    !scheme.is_empty()
        && !target.is_empty()
        && scheme.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.')
        })
        && !target.chars().any(char::is_control)
}

pub fn standard_harness_settings() -> SettingRegistry {
    let mut registry = SettingRegistry::default();
    registry
        .register(
            SettingSpec::new(
                PROVIDER_API_KEY_CREDENTIAL,
                SettingKind::CredentialRef,
                None,
                "Credential reference for the selected provider API key",
            )
            .secret(),
        )
        .expect("built-in provider credential setting is valid");
    for (key, millis, description) in [
        (
            "jobs.queue_timeout",
            5 * 60_000,
            "Managed job queue deadline",
        ),
        (
            "jobs.candidate_timeout",
            15 * 60_000,
            "Candidate execution deadline",
        ),
        (
            "jobs.verifier_timeout",
            2 * 60_000,
            "Candidate verifier deadline",
        ),
        (
            "workspace.settlement_pending_after",
            60_000,
            "Settlement caller deadline",
        ),
    ] {
        registry
            .register(
                SettingSpec::new(
                    key,
                    SettingKind::DurationMillis,
                    Some(SettingValue::DurationMillis(millis)),
                    description,
                )
                .bounded(1, 24 * 60 * 60_000),
            )
            .expect("built-in harness duration setting is valid");
    }
    for (key, value, maximum, description) in [
        (
            "jobs.max_preparations",
            4,
            64,
            "Concurrent candidate preparations",
        ),
        ("jobs.max_active", 16, 256, "Concurrent managed jobs"),
    ] {
        registry
            .register(
                SettingSpec::new(
                    key,
                    SettingKind::Integer,
                    Some(SettingValue::Integer(value)),
                    description,
                )
                .bounded(1, maximum),
            )
            .expect("built-in harness integer setting is valid");
    }
    for (key, enabled, description) in [
        (
            "features.workspace_controller_v2",
            true,
            "Enforce workspace admission and settlement",
        ),
        (
            "features.session_reducer_v2",
            true,
            "Compare and publish the versioned session reducer",
        ),
        (
            "features.candidate_jobs_v2",
            false,
            "Enable unified detached candidate jobs",
        ),
        (
            "features.pipefs_causal_commit_v1",
            false,
            "Use negotiated atomic PipeFS operation commits",
        ),
        (
            "features.native_director_v2",
            false,
            "Promote the native director from shadow traces",
        ),
        (
            "features.session_projection_v2",
            false,
            "Drive presentation clients from session snapshots",
        ),
    ] {
        registry
            .register(SettingSpec::new(
                key,
                SettingKind::Boolean,
                Some(SettingValue::Boolean(enabled)),
                description,
            ))
            .expect("built-in harness feature setting is valid");
    }
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(source: SettingSource, key: &str, value: SettingValue) -> SettingLayer {
        SettingLayer {
            source,
            values: BTreeMap::from([(key.to_owned(), value)]),
        }
    }

    #[test]
    fn precedence_is_fixed_and_untrusted_workspace_is_ignored() {
        let registry = standard_harness_settings();
        let layers = [
            layer(
                SettingSource::Session,
                "jobs.max_active",
                SettingValue::Integer(20),
            ),
            layer(
                SettingSource::TrustedWorkspace,
                "jobs.max_active",
                SettingValue::Integer(30),
            ),
            layer(
                SettingSource::OneShot,
                "jobs.max_active",
                SettingValue::Integer(24),
            ),
        ];
        let resolved = registry.resolve("jobs.max_active", &layers, false).unwrap();
        assert_eq!(resolved.value, SettingValue::Integer(24));
        assert_eq!(resolved.source, SettingSource::OneShot);
        let trusted = registry
            .resolve("jobs.max_active", &layers[..2], true)
            .unwrap();
        assert_eq!(trusted.value, SettingValue::Integer(20));
    }

    #[test]
    fn secrets_only_accept_credential_references() {
        let mut registry = SettingRegistry::default();
        registry
            .register(SettingSpec::new("provider.key", SettingKind::String, None, "key").secret())
            .unwrap();
        let literal = layer(
            SettingSource::Session,
            "provider.key",
            SettingValue::String("secret".into()),
        );
        assert!(matches!(
            registry.resolve("provider.key", &[literal], true),
            Err(SettingError::WrongType { .. })
        ));
        let reference = layer(
            SettingSource::Profile,
            "provider.key",
            SettingValue::CredentialRef("keychain://provider/default".into()),
        );
        assert!(registry.resolve("provider.key", &[reference], true).is_ok());
        let disguised_literal = layer(
            SettingSource::Session,
            "provider.key",
            SettingValue::CredentialRef("literal-secret".into()),
        );
        assert!(matches!(
            registry.resolve("provider.key", &[disguised_literal], true),
            Err(SettingError::LiteralSecret(_))
        ));
    }

    #[test]
    fn whole_layer_validation_rejects_unknown_dormant_values() {
        let registry = standard_harness_settings();
        let layer = layer(
            SettingSource::Session,
            "future.unknown",
            SettingValue::Boolean(true),
        );
        assert!(matches!(
            registry.validate_layer(&layer),
            Err(SettingError::Unknown(key)) if key == "future.unknown"
        ));
    }

    #[test]
    fn managed_defaults_are_finite() {
        let registry = standard_harness_settings();
        for spec in registry.specs() {
            // Credentials deliberately have no material default: a missing
            // secret must stay missing rather than being persisted as a
            // placeholder value. Managed numeric controls remain finite.
            if spec.secret {
                assert!(spec.default.is_none());
                assert_eq!(spec.kind, SettingKind::CredentialRef);
                continue;
            }
            assert!(
                spec.default.is_some(),
                "{} needs a managed default",
                spec.key
            );
            if matches!(
                spec.kind,
                SettingKind::Integer | SettingKind::DurationMillis
            ) {
                assert!(spec.maximum.is_some());
            }
        }
    }
}
