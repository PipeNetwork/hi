//! Concrete, validated harness settings consumed by runtime components.

use std::time::Duration;

use crate::{SettingError, SettingLayer, SettingRegistry, SettingValue, standard_harness_settings};

pub const JOB_QUEUE_TIMEOUT: &str = "jobs.queue_timeout";
pub const JOB_CANDIDATE_TIMEOUT: &str = "jobs.candidate_timeout";
pub const JOB_VERIFIER_TIMEOUT: &str = "jobs.verifier_timeout";
pub const JOB_MAX_PREPARATIONS: &str = "jobs.max_preparations";
pub const JOB_MAX_ACTIVE: &str = "jobs.max_active";
pub const SETTLEMENT_PENDING_AFTER: &str = "workspace.settlement_pending_after";
pub const WORKSPACE_CONTROLLER_V2: &str = "features.workspace_controller_v2";
pub const SESSION_REDUCER_V2: &str = "features.session_reducer_v2";
pub const CANDIDATE_JOBS_V2: &str = "features.candidate_jobs_v2";
pub const PIPEFS_CAUSAL_COMMIT_V1: &str = "features.pipefs_causal_commit_v1";
pub const NATIVE_DIRECTOR_V2: &str = "features.native_director_v2";
pub const SESSION_PROJECTION_V2: &str = "features.session_projection_v2";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarnessJobSettings {
    pub queue_timeout: Duration,
    pub candidate_timeout: Duration,
    pub verifier_timeout: Duration,
    pub max_preparations: usize,
    pub max_active: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarnessFeatureGates {
    pub workspace_controller_v2: bool,
    pub session_reducer_v2: bool,
    pub candidate_jobs_v2: bool,
    pub pipefs_causal_commit_v1: bool,
    pub native_director_v2: bool,
    pub session_projection_v2: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedHarnessSettings {
    pub jobs: HarnessJobSettings,
    pub settlement_pending_after: Duration,
    pub features: HarnessFeatureGates,
}

impl ResolvedHarnessSettings {
    /// Resolve every standard setting through the typed registry. Callers may
    /// include profile, trusted-workspace, session, and one-shot layers in any
    /// order; `SettingRegistry` applies the fixed source precedence.
    pub fn resolve(layers: &[SettingLayer], workspace_trusted: bool) -> Result<Self, SettingError> {
        let registry = standard_harness_settings();
        Self::resolve_with(&registry, layers, workspace_trusted)
    }

    fn resolve_with(
        registry: &SettingRegistry,
        layers: &[SettingLayer],
        workspace_trusted: bool,
    ) -> Result<Self, SettingError> {
        Ok(Self {
            jobs: HarnessJobSettings {
                queue_timeout: duration(registry, JOB_QUEUE_TIMEOUT, layers, workspace_trusted)?,
                candidate_timeout: duration(
                    registry,
                    JOB_CANDIDATE_TIMEOUT,
                    layers,
                    workspace_trusted,
                )?,
                verifier_timeout: duration(
                    registry,
                    JOB_VERIFIER_TIMEOUT,
                    layers,
                    workspace_trusted,
                )?,
                max_preparations: integer(
                    registry,
                    JOB_MAX_PREPARATIONS,
                    layers,
                    workspace_trusted,
                )?,
                max_active: integer(registry, JOB_MAX_ACTIVE, layers, workspace_trusted)?,
            },
            settlement_pending_after: duration(
                registry,
                SETTLEMENT_PENDING_AFTER,
                layers,
                workspace_trusted,
            )?,
            features: HarnessFeatureGates {
                workspace_controller_v2: boolean(
                    registry,
                    WORKSPACE_CONTROLLER_V2,
                    layers,
                    workspace_trusted,
                )?,
                session_reducer_v2: boolean(
                    registry,
                    SESSION_REDUCER_V2,
                    layers,
                    workspace_trusted,
                )?,
                candidate_jobs_v2: boolean(registry, CANDIDATE_JOBS_V2, layers, workspace_trusted)?,
                pipefs_causal_commit_v1: boolean(
                    registry,
                    PIPEFS_CAUSAL_COMMIT_V1,
                    layers,
                    workspace_trusted,
                )?,
                native_director_v2: boolean(
                    registry,
                    NATIVE_DIRECTOR_V2,
                    layers,
                    workspace_trusted,
                )?,
                session_projection_v2: boolean(
                    registry,
                    SESSION_PROJECTION_V2,
                    layers,
                    workspace_trusted,
                )?,
            },
        })
    }
}

impl Default for ResolvedHarnessSettings {
    fn default() -> Self {
        Self::resolve(&[], false).expect("built-in harness settings are valid")
    }
}

fn value(
    registry: &SettingRegistry,
    key: &str,
    layers: &[SettingLayer],
    workspace_trusted: bool,
) -> Result<SettingValue, SettingError> {
    Ok(registry.resolve(key, layers, workspace_trusted)?.value)
}

fn duration(
    registry: &SettingRegistry,
    key: &str,
    layers: &[SettingLayer],
    workspace_trusted: bool,
) -> Result<Duration, SettingError> {
    match value(registry, key, layers, workspace_trusted)? {
        SettingValue::DurationMillis(value) => Ok(Duration::from_millis(value)),
        _ => unreachable!("registry validated the standard setting kind"),
    }
}

fn integer(
    registry: &SettingRegistry,
    key: &str,
    layers: &[SettingLayer],
    workspace_trusted: bool,
) -> Result<usize, SettingError> {
    match value(registry, key, layers, workspace_trusted)? {
        SettingValue::Integer(value) => {
            Ok(usize::try_from(value).expect("registry bounded integer to a positive value"))
        }
        _ => unreachable!("registry validated the standard setting kind"),
    }
}

fn boolean(
    registry: &SettingRegistry,
    key: &str,
    layers: &[SettingLayer],
    workspace_trusted: bool,
) -> Result<bool, SettingError> {
    match value(registry, key, layers, workspace_trusted)? {
        SettingValue::Boolean(value) => Ok(value),
        _ => unreachable!("registry validated the standard setting kind"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::SettingSource;

    fn layer(
        source: SettingSource,
        values: impl IntoIterator<Item = (&'static str, SettingValue)>,
    ) -> SettingLayer {
        SettingLayer {
            source,
            values: values
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value))
                .collect::<BTreeMap<_, _>>(),
        }
    }

    #[test]
    fn defaults_are_the_locked_managed_limits_and_rollout_gates() {
        let settings = ResolvedHarnessSettings::default();
        assert_eq!(settings.jobs.queue_timeout, Duration::from_secs(5 * 60));
        assert_eq!(
            settings.jobs.candidate_timeout,
            Duration::from_secs(15 * 60)
        );
        assert_eq!(settings.jobs.verifier_timeout, Duration::from_secs(2 * 60));
        assert_eq!(settings.jobs.max_preparations, 4);
        assert_eq!(settings.jobs.max_active, 16);
        assert!(settings.features.workspace_controller_v2);
        assert!(settings.features.session_reducer_v2);
        assert!(!settings.features.candidate_jobs_v2);
        assert!(!settings.features.pipefs_causal_commit_v1);
        assert!(!settings.features.native_director_v2);
        assert!(!settings.features.session_projection_v2);
    }

    #[test]
    fn untrusted_workspace_is_ignored_and_one_shot_wins() {
        let layers = [
            layer(
                SettingSource::Profile,
                [(JOB_MAX_ACTIVE, SettingValue::Integer(8))],
            ),
            layer(
                SettingSource::TrustedWorkspace,
                [
                    (JOB_MAX_ACTIVE, SettingValue::Integer(2)),
                    (CANDIDATE_JOBS_V2, SettingValue::Boolean(true)),
                ],
            ),
            layer(
                SettingSource::Session,
                [(JOB_MAX_ACTIVE, SettingValue::Integer(6))],
            ),
            layer(
                SettingSource::OneShot,
                [(JOB_MAX_ACTIVE, SettingValue::Integer(3))],
            ),
        ];
        let untrusted = ResolvedHarnessSettings::resolve(&layers[..2], false).unwrap();
        assert_eq!(untrusted.jobs.max_active, 8);
        assert!(!untrusted.features.candidate_jobs_v2);

        let trusted = ResolvedHarnessSettings::resolve(&layers[..2], true).unwrap();
        assert_eq!(trusted.jobs.max_active, 2);
        assert!(trusted.features.candidate_jobs_v2);

        let one_shot = ResolvedHarnessSettings::resolve(&layers, true).unwrap();
        assert_eq!(one_shot.jobs.max_active, 3);
    }
}
