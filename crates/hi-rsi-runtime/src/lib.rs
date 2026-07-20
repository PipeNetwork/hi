//! Candidate-side contracts for an RSI-managed `hi` execution.
//!
//! The trusted worker verifies the signed control-plane manifest and derives a
//! smaller [`ManagedRuntimeDescriptor`]. The candidate harness consumes this
//! descriptor, validates that its effective command-line policy matches it,
//! and binds every local evidence record to the supplied run and candidate.
//! This crate intentionally contains no backend credentials or promotion API.

//!
//! Ownership: RSI workflow path only — not `hi-agent::run_turn`. See `docs/architecture.md`.
mod budget;
mod contract;
mod replay;
mod state;
mod workflow;

pub use budget::{BudgetKind, BudgetLedger, BudgetReservation, BudgetUsage, SharedBudgetLedger};
pub use contract::{
    CandidateIdentity, EffectiveRuntime, IsolationProfile, ManagedRuntimeDescriptor, MutationLevel,
    RuntimeBudgets, RuntimePackage, RuntimePolicy,
};
pub use replay::{ExactReplay, RecordedExchange, ReplayError, ReplayKind};
pub use state::{
    ArtifactRef, Checkpoint, ContextItem, ContextManifest, EngineeringPlan, FailureDomain,
    FailureEvidence, MemoryClass, MemoryEntry, ModelRole, RepositoryState, RunState,
    VerificationCheck, VerificationReport, VerificationStatus,
};
pub use workflow::{
    StageDefinition, StageId, StageKind, TransitionCondition, TransitionRule, WorkflowGraph,
    WorkflowLimits,
};

pub const RUNTIME_DESCRIPTOR_SCHEMA_VERSION: u16 = 1;
pub const LOCAL_PROTOCOL_MAJOR_VERSION: u16 = 1;
