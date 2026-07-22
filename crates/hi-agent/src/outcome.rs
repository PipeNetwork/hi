//! Typed public outcome of one agent turn.

use serde::{Deserialize, Serialize};

/// Whether the agent satisfied the turn's completion contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    Completed,
    Incomplete,
    Blocked,
    Cancelled,
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

/// Independent-review state for the turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewStatus {
    NotRequired,
    Passed,
    Objected,
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
    ToolModeDenied,
    StepLimit,
    /// Per-session turn limit (`/turns <n>`) reached before this turn started.
    /// Distinct from [`Self::StepLimit`], which is the per-turn model-call cap.
    TurnLimit,
    Stalled,
    Cancelled,
    InfrastructureFailure,
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
}

impl TurnOutcome {
    /// Construct the typed failure included in reports when `run_turn` returns
    /// an infrastructure/provider error instead of a normal turn outcome.
    pub fn infrastructure_failure(
        model: impl Into<String>,
        provider: Option<String>,
        changed_files: Vec<String>,
    ) -> Self {
        Self {
            status: TurnStatus::Failed,
            verification: VerificationStatus::InfrastructureError,
            review: ReviewStatus::Unavailable,
            stop_reason: TurnStopReason::InfrastructureFailure,
            changed_files,
            verified_workspace_revision: None,
            effective_route: EffectiveModelRoute {
                provider,
                model: model.into(),
            },
        }
    }

    /// Process exit code for one-shot CLI runs.
    ///
    /// - `0` completed + passed / N/A (or unverified when allowed)
    /// - `1` incomplete / blocked / verify failed / unverified
    /// - `3` failed / infrastructure error
    /// - `130` cancelled
    pub fn exit_code(&self, allow_unverified: bool) -> i32 {
        match self.status {
            TurnStatus::Cancelled => 130,
            TurnStatus::Failed => 3,
            TurnStatus::Incomplete | TurnStatus::Blocked => 1,
            TurnStatus::Completed => match self.verification {
                VerificationStatus::Passed | VerificationStatus::NotApplicable => 0,
                VerificationStatus::Unverified if allow_unverified => 0,
                VerificationStatus::Unverified | VerificationStatus::Failed => 1,
                VerificationStatus::InfrastructureError => 3,
            },
        }
    }
}

/// How session state was handled before [`crate::Agent::cleanup_turn`] on cancel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionRollback {
    /// Frontend already rewound transcript/goals/plan; agent must not truncate again.
    AlreadyApplied,
    /// Agent should undo new checkpoints (if any) and truncate to the turn message start.
    AgentOwned { checkpoint_count_before: usize },
}

/// Abnormal turn teardown requested by a frontend (not used on successful `run_turn`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnCleanupKind {
    /// User interrupt / dropped turn future.
    Cancel { session: SessionRollback },
    /// `run_turn` returned `Err` or escaped before the normal finalizer.
    Fail,
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
