//! Typed public outcome of one agent turn.

use serde::{Deserialize, Serialize};

/// Whether the agent satisfied the turn's completion contract.
///
/// Normal completion is distinct from a blocked/cancelled turn and from a
/// failure. Legacy `incomplete` records deserialize as [`Self::Failed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    Completed,
    Blocked,
    Cancelled,
    #[serde(alias = "Incomplete", alias = "incomplete", alias = "Failed")]
    Failed,
}

/// Deterministic verification state for the final workspace revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    Passed,
    NotApplicable,
    Unverified,
    Failed,
    InfrastructureError,
}

/// **Completion-review / goal-skeptic** state for the turn.
///
/// Combined from independent/large-diff review and long-horizon skeptic via
/// `combined_review_status`. Steer-phase answer repair does **not** set this
/// field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewStatus {
    NotRequired,
    Passed,
    Objected,
    /// Goal skeptic skipped the step (unfixable); not a defect objection.
    Escalated,
    Unavailable,
}

/// Machine-readable reason the turn stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStopReason {
    Completed,
    NoApplicableVerification,
    VerificationUnavailable,
    VerificationFailed,
    VerificationUnstable,
    ReviewObjected,
    /// Goal skeptic escalated/skipped; turn may still Complete with a scar.
    ReviewEscalated,
    ToolModeDenied,
    /// Workspace admission was denied by a transient or concurrently owned
    /// controller state. The effect was not started.
    WorkspaceNotReady,
    /// Workspace admission was denied by a durable recovery fence. The effect
    /// was not started and recovery evidence remains available for inspection.
    WorkspaceRecoveryRequired,
    /// A productive/fault-recovery loop stopped because it produced no new
    /// evidence. Kept distinct from user-configured execution limits.
    #[serde(alias = "Stalled", alias = "stalled")]
    NoProgress,
    #[serde(alias = "StepLimit")]
    StepLimit,
    /// An explicitly configured finite per-turn tool-execution ceiling was
    /// reached. The ordinary default is unlimited.
    ToolLimit,
    /// The turn's soft wall-clock budget expired, so it stopped starting new
    /// work and settled early. Distinct from [`Self::StepLimit`] (model-call
    /// ceiling) and from a hard `turn_timeout`, which settles nothing.
    TimeLimit,
    /// Per-session turn limit (`/turns <n>`) reached before this turn started.
    /// Distinct from [`Self::StepLimit`], which is the per-turn model-call cap.
    TurnLimit,
    Cancelled,
    InfrastructureFailure,
}

impl TurnStopReason {
    /// Classify an error that escaped before normal turn finalization.
    ///
    /// Workspace admission is an expected fail-closed boundary, not an
    /// infrastructure outage. All other escaped errors retain the historical
    /// infrastructure classification.
    pub fn for_error(err: &anyhow::Error) -> Self {
        if let Some(failure) = crate::TurnFailure::from_error(err) {
            return failure.outcome.stop_reason;
        }

        let Some(denied) = err.downcast_ref::<hi_workspace::AdmissionDenied>() else {
            return Self::InfrastructureFailure;
        };
        match denied.state {
            hi_workspace::WorkspaceState::PendingRemote
            | hi_workspace::WorkspaceState::LeaseUncertain
            | hi_workspace::WorkspaceState::LeaseLost
            | hi_workspace::WorkspaceState::Conflict
            | hi_workspace::WorkspaceState::TranscriptPending
            | hi_workspace::WorkspaceState::CleanupPending
            | hi_workspace::WorkspaceState::RecoveryRequired
            | hi_workspace::WorkspaceState::JournalCorrupt
            | hi_workspace::WorkspaceState::Incompatible => Self::WorkspaceRecoveryRequired,
            _ => Self::WorkspaceNotReady,
        }
    }

    pub const fn is_workspace_admission(self) -> bool {
        matches!(
            self,
            Self::WorkspaceNotReady | Self::WorkspaceRecoveryRequired
        )
    }
}

/// Provider/model route that was effective for the turn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveModelRoute {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    pub model: String,
}

/// Complete typed result of [`crate::Agent::run_turn`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnOutcome {
    pub status: TurnStatus,
    pub verification: VerificationStatus,
    pub review: ReviewStatus,
    pub stop_reason: TurnStopReason,
    pub changed_files: Vec<String>,
    /// Stable fingerprint of the exact workspace state that passed verification.
    /// It is absent for unverified, failed, and not-applicable checks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified_workspace_revision: Option<String>,
    pub effective_route: EffectiveModelRoute,
    /// True when the effective skeptic/completion-review model is the session
    /// model (unconfigured `skeptic_model`, or explicitly set to the same id).
    /// Observability only — does not change gate policy. Defaults false on
    /// older deserialized records so they do not claim same-model review.
    #[serde(default)]
    pub review_same_model: bool,
    /// Leftover work the next drive would actually run (goal if auto-driving,
    /// else plan). Does not change the exit code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leftover: Option<String>,
    /// Checklist leftover even when a structured goal would shadow `leftover`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_leftover: Option<String>,
}

impl TurnOutcome {
    /// Construct the typed failure included in reports when `run_turn` returns
    /// an infrastructure/provider error instead of a normal turn outcome.
    ///
    /// This says nothing about deterministic verification. Callers that still
    /// hold current-revision verification evidence may attach it separately;
    /// an ordinary provider outage must never be reported as a verification
    /// infrastructure failure.
    pub fn infrastructure_failure(
        model: impl Into<String>,
        provider: Option<String>,
        changed_files: Vec<String>,
    ) -> Self {
        Self {
            status: TurnStatus::Failed,
            verification: VerificationStatus::Unverified,
            review: ReviewStatus::Unavailable,
            stop_reason: TurnStopReason::InfrastructureFailure,
            changed_files,
            verified_workspace_revision: None,
            effective_route: EffectiveModelRoute {
                provider,
                model: model.into(),
            },
            // Unknown outside Agent; callers with config should overwrite.
            review_same_model: false,
            leftover: None,
            plan_leftover: None,
        }
    }

    /// Construct the typed blocked result for a fail-closed workspace
    /// admission. No operation was admitted, so this is intentionally neither
    /// a provider failure nor a successful turn.
    pub fn workspace_admission_blocked(
        model: impl Into<String>,
        provider: Option<String>,
        changed_files: Vec<String>,
        stop_reason: TurnStopReason,
    ) -> Self {
        debug_assert!(stop_reason.is_workspace_admission());
        Self {
            status: TurnStatus::Blocked,
            verification: VerificationStatus::Unverified,
            review: ReviewStatus::NotRequired,
            stop_reason,
            changed_files,
            verified_workspace_revision: None,
            effective_route: EffectiveModelRoute {
                provider,
                model: model.into(),
            },
            review_same_model: false,
            leftover: None,
            plan_leftover: None,
        }
    }

    /// Process exit code for one-shot CLI runs.
    ///
    /// - `0` completed + passed / N/A (or unverified when allowed)
    /// - `1` blocked / non-infrastructure failure / verify failed / unverified
    /// - `3` infrastructure error
    /// - `130` cancelled
    pub fn exit_code(&self, allow_unverified: bool) -> i32 {
        match self.status {
            TurnStatus::Cancelled => 130,
            TurnStatus::Failed => {
                if self.verification == VerificationStatus::InfrastructureError
                    || self.stop_reason == TurnStopReason::InfrastructureFailure
                {
                    3
                } else {
                    1
                }
            }
            TurnStatus::Blocked => 1,
            TurnStatus::Completed => match self.verification {
                VerificationStatus::Passed | VerificationStatus::NotApplicable => 0,
                VerificationStatus::Unverified if allow_unverified => 0,
                VerificationStatus::Unverified | VerificationStatus::Failed => 1,
                VerificationStatus::InfrastructureError => 3,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EffectiveModelRoute, ReviewStatus, TurnOutcome, TurnStatus, TurnStopReason,
        VerificationStatus,
    };

    fn outcome(
        status: TurnStatus,
        verification: VerificationStatus,
        stop_reason: TurnStopReason,
    ) -> TurnOutcome {
        TurnOutcome {
            status,
            verification,
            review: ReviewStatus::NotRequired,
            stop_reason,
            changed_files: Vec::new(),
            verified_workspace_revision: None,
            effective_route: EffectiveModelRoute {
                provider: None,
                model: "test".into(),
            },
            review_same_model: false,
            leftover: None,
            plan_leftover: None,
        }
    }

    #[test]
    fn legacy_incomplete_status_deserializes_as_failed_without_re_emitting_it() {
        for legacy in [r#""incomplete""#, r#""Incomplete""#] {
            let status: TurnStatus = serde_json::from_str(legacy).unwrap();
            assert_eq!(status, TurnStatus::Failed);
            assert_eq!(serde_json::to_string(&status).unwrap(), r#""failed""#);
        }
    }

    #[test]
    fn legacy_stalled_reason_deserializes_as_no_progress_without_becoming_a_limit() {
        for legacy in [r#""stalled""#, r#""Stalled""#] {
            let reason: TurnStopReason = serde_json::from_str(legacy).unwrap();
            assert_eq!(reason, TurnStopReason::NoProgress);
            assert_eq!(serde_json::to_string(&reason).unwrap(), r#""no_progress""#);
        }
    }

    #[test]
    fn tool_limit_has_a_distinct_stable_wire_value() {
        let encoded = serde_json::to_string(&TurnStopReason::ToolLimit).unwrap();
        assert_eq!(encoded, r#""tool_limit""#);
        assert_eq!(
            serde_json::from_str::<TurnStopReason>(&encoded).unwrap(),
            TurnStopReason::ToolLimit
        );

        // Historical records remain unambiguous and readable.
        assert_eq!(
            serde_json::from_str::<TurnStopReason>(r#""step_limit""#).unwrap(),
            TurnStopReason::StepLimit
        );
    }

    #[test]
    fn workspace_stop_reasons_have_stable_wire_values_and_exit_one() {
        for (reason, wire) in [
            (
                TurnStopReason::WorkspaceNotReady,
                r#""workspace_not_ready""#,
            ),
            (
                TurnStopReason::WorkspaceRecoveryRequired,
                r#""workspace_recovery_required""#,
            ),
        ] {
            assert_eq!(serde_json::to_string(&reason).unwrap(), wire);
            assert_eq!(
                serde_json::from_str::<TurnStopReason>(wire).unwrap(),
                reason
            );
            assert_eq!(
                outcome(TurnStatus::Blocked, VerificationStatus::Unverified, reason)
                    .exit_code(false),
                1
            );
        }
    }

    #[test]
    fn failed_exit_codes_distinguish_contract_failure_from_infrastructure() {
        assert_eq!(
            outcome(
                TurnStatus::Failed,
                VerificationStatus::Passed,
                TurnStopReason::StepLimit,
            )
            .exit_code(false),
            1
        );
        assert_eq!(
            outcome(
                TurnStatus::Failed,
                VerificationStatus::InfrastructureError,
                TurnStopReason::InfrastructureFailure,
            )
            .exit_code(false),
            3
        );
    }
}

/// How session state was handled before [`crate::Agent::cleanup_turn`] on cancel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionRollback {
    /// Frontend already rewound transcript/goals/plan; agent must not truncate again.
    AlreadyApplied,
    /// Agent should undo a checkpoint created by this turn (if any), restore the
    /// exact bounded checkpoint stack from before the turn, and truncate to the
    /// turn message start. Comparing the stack identity rather than only its
    /// length matters once retention is full: pushing a new checkpoint evicts
    /// the oldest entry and leaves the length unchanged.
    AgentOwned { checkpoint_refs_before: Vec<String> },
}

/// Abnormal turn teardown requested by a frontend (not used on successful `run_turn`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TurnCleanupKind {
    /// User interrupt / dropped turn future.
    Cancel { session: SessionRollback },
    /// `run_turn` returned `Err` or escaped before the normal finalizer.
    Fail,
    /// `run_turn` returned `Err`, but the frontend retained a narrow typed
    /// non-infrastructure stop reason for terminal settlement.
    FailWithStopReason(TurnStopReason),
}

impl TurnCleanupKind {
    pub fn for_error(err: &anyhow::Error) -> Self {
        let stop_reason = TurnStopReason::for_error(err);
        if stop_reason == TurnStopReason::InfrastructureFailure {
            Self::Fail
        } else {
            Self::FailWithStopReason(stop_reason)
        }
    }
}

/// Result of [`crate::Agent::cleanup_turn`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnCleanupResult {
    pub outcome: TurnOutcome,
    /// Background processes killed via the turn-scoped baseline (for UI copy).
    pub killed_backgrounds: usize,
}

/// Coarse classification for top-level CLI errors that escape outside a typed
/// [`TurnOutcome`] (setup/config/parse vs infrastructure).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopLevelErrorKind {
    /// Usage, config, or JSON parse errors → exit 2.
    Usage,
    /// Unrecovered setup/provider/runner failure → exit 3.
    Infra,
}

impl TopLevelErrorKind {
    pub fn exit_code(self) -> i32 {
        match self {
            Self::Usage => 2,
            Self::Infra => 3,
        }
    }

    /// Classify an escaped `anyhow` error from message content.
    pub fn from_anyhow(error: &anyhow::Error) -> Self {
        let message = format!("{error:#}").to_ascii_lowercase();
        if message.contains("usage:")
            || message.contains("parsing skeptic-review json")
            || message.contains("invalid configuration")
        {
            Self::Usage
        } else {
            Self::Infra
        }
    }
}
