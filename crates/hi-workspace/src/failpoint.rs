//! Deterministic harness failpoints for crash-window characterization.

use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const HARNESS_FAILPOINT_ENV: &str = "HI_HARNESS_FAILPOINTS";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarnessFailpoint {
    AdmissionBeforeJournal,
    AdmissionAfterJournal,
    ToolBeforeStart,
    ExecutionAfterEffect,
    ArchiveAfterFsync,
    CommitAfterServerBeforeAck,
    TranscriptBeforeFlush,
    CandidateBeforeApply,
    CandidateAfterApply,
    RollbackBeforeRestore,
    RebindAfterDrain,
    LeaseGenerationChanged,
    JobAfterSpawn,
    JobAfterNaturalExit,
    JobAfterCancelRequest,
    ImportBeforePublish,
    ExportBeforeRename,
    CompactionBeforeCas,
    SchemaBeforeVersionUpdate,
}

impl HarnessFailpoint {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AdmissionBeforeJournal => "admission_before_journal",
            Self::AdmissionAfterJournal => "admission_after_journal",
            Self::ToolBeforeStart => "tool_before_start",
            Self::ExecutionAfterEffect => "execution_after_effect",
            Self::ArchiveAfterFsync => "archive_after_fsync",
            Self::CommitAfterServerBeforeAck => "commit_after_server_before_ack",
            Self::TranscriptBeforeFlush => "transcript_before_flush",
            Self::CandidateBeforeApply => "candidate_before_apply",
            Self::CandidateAfterApply => "candidate_after_apply",
            Self::RollbackBeforeRestore => "rollback_before_restore",
            Self::RebindAfterDrain => "rebind_after_drain",
            Self::LeaseGenerationChanged => "lease_generation_changed",
            Self::JobAfterSpawn => "job_after_spawn",
            Self::JobAfterNaturalExit => "job_after_natural_exit",
            Self::JobAfterCancelRequest => "job_after_cancel_request",
            Self::ImportBeforePublish => "import_before_publish",
            Self::ExportBeforeRename => "export_before_rename",
            Self::CompactionBeforeCas => "compaction_before_cas",
            Self::SchemaBeforeVersionUpdate => "schema_before_version_update",
        }
    }
}

impl FromStr for HarnessFailpoint {
    type Err = FailpointError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let point = match value.trim() {
            "admission_before_journal" => Self::AdmissionBeforeJournal,
            "admission_after_journal" => Self::AdmissionAfterJournal,
            "tool_before_start" => Self::ToolBeforeStart,
            "execution_after_effect" => Self::ExecutionAfterEffect,
            "archive_after_fsync" => Self::ArchiveAfterFsync,
            "commit_after_server_before_ack" => Self::CommitAfterServerBeforeAck,
            "transcript_before_flush" => Self::TranscriptBeforeFlush,
            "candidate_before_apply" => Self::CandidateBeforeApply,
            "candidate_after_apply" => Self::CandidateAfterApply,
            "rollback_before_restore" => Self::RollbackBeforeRestore,
            "rebind_after_drain" => Self::RebindAfterDrain,
            "lease_generation_changed" => Self::LeaseGenerationChanged,
            "job_after_spawn" => Self::JobAfterSpawn,
            "job_after_natural_exit" => Self::JobAfterNaturalExit,
            "job_after_cancel_request" => Self::JobAfterCancelRequest,
            "import_before_publish" => Self::ImportBeforePublish,
            "export_before_rename" => Self::ExportBeforeRename,
            "compaction_before_cas" => Self::CompactionBeforeCas,
            "schema_before_version_update" => Self::SchemaBeforeVersionUpdate,
            other => return Err(FailpointError::Unknown(other.to_owned())),
        };
        Ok(point)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Trigger {
    OnceAt(u64),
    Always,
}

#[derive(Clone, Debug)]
pub struct FailpointPlan {
    state: Arc<Mutex<FailpointState>>,
}

#[derive(Debug, Default)]
struct FailpointState {
    triggers: BTreeMap<HarnessFailpoint, Trigger>,
    observations: BTreeMap<HarnessFailpoint, u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum FailpointError {
    #[error("unknown harness failpoint {0:?}")]
    Unknown(String),
    #[error("invalid trigger for harness failpoint {point:?}: {trigger:?}")]
    InvalidTrigger { point: String, trigger: String },
    #[error("harness failpoint {0:?} was configured more than once")]
    Duplicate(String),
    #[error("injected harness failure at {point} (observation {observation})")]
    Injected {
        point: &'static str,
        observation: u64,
    },
    #[error("invalid {HARNESS_FAILPOINT_ENV}: {0}")]
    InvalidEnvironment(String),
}

impl FailpointPlan {
    /// Parse comma-separated `point=N` or `point=always` entries. `N` is
    /// one-based, making a boundary's first observation easy to target.
    pub fn parse(specification: &str) -> Result<Self, FailpointError> {
        let mut triggers = BTreeMap::new();
        for entry in specification
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
        {
            let (name, raw_trigger) =
                entry
                    .split_once('=')
                    .ok_or_else(|| FailpointError::InvalidTrigger {
                        point: entry.to_owned(),
                        trigger: String::new(),
                    })?;
            let point = HarnessFailpoint::from_str(name)?;
            let trigger = if raw_trigger == "always" {
                Trigger::Always
            } else {
                let occurrence = raw_trigger.parse::<u64>().ok().filter(|value| *value > 0);
                Trigger::OnceAt(occurrence.ok_or_else(|| FailpointError::InvalidTrigger {
                    point: name.to_owned(),
                    trigger: raw_trigger.to_owned(),
                })?)
            };
            if triggers.insert(point, trigger).is_some() {
                return Err(FailpointError::Duplicate(name.to_owned()));
            }
        }
        Ok(Self {
            state: Arc::new(Mutex::new(FailpointState {
                triggers,
                observations: BTreeMap::new(),
            })),
        })
    }

    pub fn hit(&self, point: HarnessFailpoint) -> Result<(), FailpointError> {
        let mut state = lock(&self.state);
        let observation = state
            .observations
            .get(&point)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        state.observations.insert(point, observation);
        let fires = match state.triggers.get(&point) {
            Some(Trigger::Always) => true,
            Some(Trigger::OnceAt(target)) => observation == *target,
            None => false,
        };
        if fires {
            Err(FailpointError::Injected {
                point: point.as_str(),
                observation,
            })
        } else {
            Ok(())
        }
    }

    pub fn observations(&self, point: HarnessFailpoint) -> u64 {
        lock(&self.state)
            .observations
            .get(&point)
            .copied()
            .unwrap_or_default()
    }
}

/// Hit the process-wide deterministic plan loaded once from
/// `HI_HARNESS_FAILPOINTS`. An unset or empty value has no effect. Invalid
/// configuration fails the boundary instead of silently disabling the test.
pub fn hit_harness_failpoint(point: HarnessFailpoint) -> Result<(), FailpointError> {
    static PLAN: OnceLock<Result<FailpointPlan, String>> = OnceLock::new();
    let plan = PLAN.get_or_init(|| {
        let specification = std::env::var(HARNESS_FAILPOINT_ENV).unwrap_or_default();
        FailpointPlan::parse(&specification).map_err(|error| error.to_string())
    });
    match plan {
        Ok(plan) => plan.hit(point),
        Err(error) => Err(FailpointError::InvalidEnvironment(error.clone())),
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nth_observation_fires_exactly_once() {
        let plan = FailpointPlan::parse("candidate_after_apply=2").unwrap();
        assert!(
            plan.hit(HarnessFailpoint::CandidateAfterApply).is_ok(),
            "first observation must pass"
        );
        assert!(matches!(
            plan.hit(HarnessFailpoint::CandidateAfterApply),
            Err(FailpointError::Injected { observation: 2, .. })
        ));
        assert!(plan.hit(HarnessFailpoint::CandidateAfterApply).is_ok());
        assert_eq!(plan.observations(HarnessFailpoint::CandidateAfterApply), 3);
    }

    #[test]
    fn always_and_multiple_points_are_independent() {
        let plan =
            FailpointPlan::parse("archive_after_fsync=always,transcript_before_flush=2").unwrap();
        assert!(plan.hit(HarnessFailpoint::ArchiveAfterFsync).is_err());
        assert!(plan.hit(HarnessFailpoint::ArchiveAfterFsync).is_err());
        assert!(plan.hit(HarnessFailpoint::TranscriptBeforeFlush).is_ok());
        assert!(plan.hit(HarnessFailpoint::TranscriptBeforeFlush).is_err());
    }

    #[test]
    fn invalid_or_duplicate_configuration_is_rejected() {
        assert!(matches!(
            FailpointPlan::parse("not_a_boundary=1"),
            Err(FailpointError::Unknown(_))
        ));
        assert!(matches!(
            FailpointPlan::parse("job_after_spawn=1,job_after_spawn=2"),
            Err(FailpointError::Duplicate(_))
        ));
        assert!(matches!(
            FailpointPlan::parse("job_after_spawn=0"),
            Err(FailpointError::InvalidTrigger { .. })
        ));
    }
}
