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
}
