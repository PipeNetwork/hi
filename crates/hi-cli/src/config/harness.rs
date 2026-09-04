//! CLI/config adapters for the typed harness setting registry.

use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail};
use hi_workspace::{
    ResolvedHarnessSettings, SettingKind, SettingLayer, SettingSource, SettingValue,
    standard_harness_settings,
};
use serde::{Deserialize, Serialize};

use super::{Cli, Config, Profile};

/// Scalar syntax accepted under `[harness]` and `[profiles.<name>.harness]`.
/// Dotted registry keys must be quoted in TOML, for example
/// `"jobs.max_active" = 8`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HarnessConfigValue {
    Boolean(bool),
    Integer(i64),
    String(String),
    StringList(Vec<String>),
}

pub type HarnessOverrides = BTreeMap<String, HarnessConfigValue>;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HarnessConfig {
    values: HarnessOverrides,
    #[serde(skip)]
    project: Option<ProjectHarnessLayer>,
}

impl HarnessConfig {
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub(crate) fn merge_project(&mut self, project: &mut Self, trusted: bool) {
        if !project.values.is_empty() {
            self.project = Some(ProjectHarnessLayer {
                values: std::mem::take(&mut project.values),
                trusted,
            });
        }
    }
}

impl From<HarnessOverrides> for HarnessConfig {
    fn from(values: HarnessOverrides) -> Self {
        Self {
            values,
            project: None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ProjectHarnessLayer {
    pub values: HarnessOverrides,
    pub trusted: bool,
}

/// Resolve profile, trusted-workspace, session, and one-shot sources through
/// the registry's fixed precedence.
pub(crate) fn resolve_harness(
    config: &Config,
    profile: Option<&Profile>,
    session: Option<SettingLayer>,
    one_shot: &[String],
) -> Result<ResolvedHarnessSettings> {
    let registry = standard_harness_settings();
    let mut layers = Vec::new();

    let mut profile_values = config.harness.values.clone();
    if let Some(profile) = profile.filter(|profile| !profile.project_local) {
        profile_values.extend(profile.harness.clone());
    }
    push_config_layer(
        &mut layers,
        &registry,
        SettingSource::Profile,
        &profile_values,
    )?;

    let mut workspace_values = config
        .harness
        .project
        .as_ref()
        .map(|layer| layer.values.clone())
        .unwrap_or_default();
    let mut workspace_trusted = config
        .harness
        .project
        .as_ref()
        .is_some_and(|layer| layer.trusted);
    if let Some(profile) = profile.filter(|profile| profile.project_local) {
        workspace_values.extend(profile.harness.clone());
        workspace_trusted |= profile.project_trusted;
    }
    if workspace_trusted {
        push_config_layer(
            &mut layers,
            &registry,
            SettingSource::TrustedWorkspace,
            &workspace_values,
        )?;
    }

    if let Some(session) = session {
        if session.source != SettingSource::Session {
            bail!("session harness layer has the wrong source");
        }
        registry
            .validate_layer(&session)
            .context("validating session harness layer")?;
        layers.push(session);
    }
    let one_shot = parse_layer(
        &registry,
        one_shot,
        SettingSource::OneShot,
        "--harness-setting",
    )?;
    if !one_shot.values.is_empty() {
        layers.push(one_shot);
    }

    ResolvedHarnessSettings::resolve(&layers, workspace_trusted)
        .context("resolving typed harness settings")
}

/// Load the selected session's durable layer, then apply persistent overrides
/// requested for this invocation. Callers append the returned complete layer
/// only after the final session path has been resolved.
pub(crate) fn resolve_session_harness(cli: &Cli) -> Result<SettingLayer> {
    let layer = if let Some(path) = &cli.session_file {
        crate::session_harness::load(path)?
    } else if let Some(id) = &cli.resume {
        crate::session_harness::load(&crate::session::session_path(id)?)?
    } else if cli.cont {
        crate::session::latest_session()
            .map(|path| crate::session_harness::load(&path))
            .transpose()?
            .unwrap_or_else(crate::session_harness::empty_layer)
    } else {
        crate::session_harness::empty_layer()
    };
    merge_session_harness(layer, &cli.session_harness_settings)
}

pub(crate) fn merge_session_harness(
    mut layer: SettingLayer,
    requested: &[String],
) -> Result<SettingLayer> {
    let requested = parse_layer(
        &standard_harness_settings(),
        requested,
        SettingSource::Session,
        "--session-harness-setting",
    )?;
    layer.values.extend(requested.values);
    standard_harness_settings()
        .validate_layer(&layer)
        .context("validating effective session harness layer")?;
    Ok(layer)
}

fn push_config_layer(
    layers: &mut Vec<SettingLayer>,
    registry: &hi_workspace::SettingRegistry,
    source: SettingSource,
    values: &HarnessOverrides,
) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }
    let values = values
        .iter()
        .map(|(key, value)| {
            configured_value(registry, key, value)
                .map(|value| (key.clone(), value))
                .with_context(|| format!("reading harness setting {key:?}"))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    layers.push(SettingLayer { source, values });
    Ok(())
}

fn configured_value(
    registry: &hi_workspace::SettingRegistry,
    key: &str,
    value: &HarnessConfigValue,
) -> Result<SettingValue> {
    let spec = registry
        .spec(key)
        .ok_or_else(|| anyhow!("unknown setting {key:?}"))?;
    let value = match (spec.kind, value) {
        (SettingKind::Boolean, HarnessConfigValue::Boolean(value)) => SettingValue::Boolean(*value),
        (SettingKind::Integer, HarnessConfigValue::Integer(value)) => SettingValue::Integer(*value),
        (SettingKind::DurationMillis, HarnessConfigValue::Integer(value)) => {
            SettingValue::DurationMillis(
                u64::try_from(*value).map_err(|_| anyhow!("duration cannot be negative"))?,
            )
        }
        (SettingKind::String, HarnessConfigValue::String(value)) => {
            SettingValue::String(value.clone())
        }
        (SettingKind::CredentialRef, HarnessConfigValue::String(value)) => {
            SettingValue::CredentialRef(value.clone())
        }
        (SettingKind::StringList, HarnessConfigValue::StringList(value)) => {
            SettingValue::StringList(value.clone())
        }
        (expected, actual) => bail!("expected {expected:?}, received {}", config_kind(actual)),
    };
    Ok(value)
}

fn config_kind(value: &HarnessConfigValue) -> &'static str {
    match value {
        HarnessConfigValue::Boolean(_) => "boolean",
        HarnessConfigValue::Integer(_) => "integer",
        HarnessConfigValue::String(_) => "string",
        HarnessConfigValue::StringList(_) => "string list",
    }
}

fn parse_layer(
    registry: &hi_workspace::SettingRegistry,
    values: &[String],
    source: SettingSource,
    flag: &str,
) -> Result<SettingLayer> {
    let mut layer = SettingLayer {
        source,
        values: BTreeMap::new(),
    };
    for entry in values {
        let (key, raw) = entry
            .split_once('=')
            .ok_or_else(|| anyhow!("{flag} expects KEY=VALUE, received {entry:?}"))?;
        let key = key.trim();
        let raw = raw.trim();
        let spec = registry
            .spec(key)
            .ok_or_else(|| anyhow!("unknown harness setting {key:?}"))?;
        let value = match spec.kind {
            SettingKind::Boolean => SettingValue::Boolean(parse_bool(raw)?),
            SettingKind::Integer => SettingValue::Integer(
                raw.parse()
                    .with_context(|| format!("parsing integer harness setting {key:?}"))?,
            ),
            SettingKind::DurationMillis => {
                SettingValue::DurationMillis(parse_duration_millis(raw)?)
            }
            SettingKind::String => SettingValue::String(raw.to_owned()),
            SettingKind::CredentialRef => SettingValue::CredentialRef(raw.to_owned()),
            SettingKind::StringList => SettingValue::StringList(
                raw.split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(str::to_owned)
                    .collect(),
            ),
        };
        layer.values.insert(key.to_owned(), value);
    }
    registry.validate_layer(&layer)?;
    Ok(layer)
}

fn parse_bool(raw: &str) -> Result<bool> {
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" => Ok(false),
        _ => bail!("expected true or false, received {raw:?}"),
    }
}

fn parse_duration_millis(raw: &str) -> Result<u64> {
    for (suffix, multiplier) in [("ms", 1), ("s", 1_000), ("m", 60_000), ("h", 3_600_000)] {
        if let Some(number) = raw.strip_suffix(suffix) {
            return number
                .parse::<u64>()
                .with_context(|| format!("parsing duration {raw:?}"))?
                .checked_mul(multiplier)
                .ok_or_else(|| anyhow!("duration {raw:?} is too large"));
        }
    }
    raw.parse::<u64>()
        .with_context(|| format!("parsing millisecond duration {raw:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overrides(
        values: impl IntoIterator<Item = (&'static str, HarnessConfigValue)>,
    ) -> HarnessOverrides {
        values
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect()
    }

    #[test]
    fn untrusted_workspace_cannot_override_profile_and_one_shot_wins() {
        let mut config = Config {
            harness: HarnessConfig::from(overrides([(
                hi_workspace::JOB_MAX_ACTIVE,
                HarnessConfigValue::Integer(9),
            )])),
            ..Config::default()
        };
        config.harness.project = Some(ProjectHarnessLayer {
            values: overrides([
                (hi_workspace::JOB_MAX_ACTIVE, HarnessConfigValue::Integer(2)),
                (
                    hi_workspace::CANDIDATE_JOBS_V2,
                    HarnessConfigValue::Boolean(true),
                ),
            ]),
            trusted: false,
        });
        let settings =
            resolve_harness(&config, None, None, &["jobs.max_active=3".to_owned()]).unwrap();
        assert_eq!(settings.jobs.max_active, 3);
        assert!(!settings.features.candidate_jobs_v2);
    }

    #[test]
    fn malformed_untrusted_workspace_layer_cannot_block_startup() {
        let mut config = Config::default();
        config.harness.project = Some(ProjectHarnessLayer {
            values: overrides([("unknown.future", HarnessConfigValue::Boolean(true))]),
            trusted: false,
        });
        let settings = resolve_harness(&config, None, None, &[]).unwrap();
        assert_eq!(settings.jobs.max_active, 16);
    }

    #[test]
    fn trusted_workspace_and_session_use_fixed_precedence() {
        let mut config = Config::default();
        config.harness.project = Some(ProjectHarnessLayer {
            values: overrides([(hi_workspace::JOB_MAX_ACTIVE, HarnessConfigValue::Integer(2))]),
            trusted: true,
        });
        let session = SettingLayer {
            source: SettingSource::Session,
            values: BTreeMap::from([(
                hi_workspace::JOB_MAX_ACTIVE.to_owned(),
                SettingValue::Integer(5),
            )]),
        };
        let settings = resolve_harness(&config, None, Some(session), &[]).unwrap();
        assert_eq!(settings.jobs.max_active, 5);
    }

    #[test]
    fn one_shot_still_beats_a_persisted_session_layer() {
        let session = SettingLayer {
            source: SettingSource::Session,
            values: BTreeMap::from([(
                hi_workspace::JOB_MAX_ACTIVE.to_owned(),
                SettingValue::Integer(5),
            )]),
        };
        let settings = resolve_harness(
            &Config::default(),
            None,
            Some(session),
            &["jobs.max_active=3".to_owned()],
        )
        .unwrap();
        assert_eq!(settings.jobs.max_active, 3);
    }

    #[test]
    fn cli_loads_and_updates_the_selected_session_layer() {
        use clap::Parser;

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("session.jsonl");
        crate::session_harness::append(
            &path,
            &SettingLayer {
                source: SettingSource::Session,
                values: BTreeMap::from([(
                    hi_workspace::JOB_MAX_ACTIVE.to_owned(),
                    SettingValue::Integer(7),
                )]),
            },
        )
        .unwrap();
        let cli = Cli::try_parse_from([
            "hi",
            "--session-file",
            path.to_str().unwrap(),
            "--session-harness-setting",
            "jobs.max_active=5",
        ])
        .unwrap();
        let layer = resolve_session_harness(&cli).unwrap();
        assert_eq!(
            layer.values.get(hi_workspace::JOB_MAX_ACTIVE),
            Some(&SettingValue::Integer(5))
        );
    }

    #[test]
    fn persistent_session_override_requires_a_saved_session() {
        use clap::Parser;

        assert!(
            Cli::try_parse_from([
                "hi",
                "--no-save",
                "--session-harness-setting",
                "jobs.max_active=5",
            ])
            .is_err()
        );
    }
}
