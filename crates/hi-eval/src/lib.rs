//! Library surface for hi-eval harness helpers (agent-path report validation).

pub mod agent_path;
pub mod backends;
pub mod config;
pub mod differential;
pub mod platform;

pub use platform::{
    ArtifactSpec, AttemptRecord, AttemptStatus, CasePlan, ClaimLevel, DatasetPlan, DatasetSource,
    DifferentialArm, DifferentialArmConfig, DifferentialComparison, EnvironmentSpec, EvalAttempt,
    EvalBackend, EvalEvidence, EvalInput, EvalManifest, EvalOutput, EvalProfile, EvalScore,
    EvalStateStore, EvidencePolicy, IdentityDetails, ImportStore, ImportedDataset, ImportedTask,
    McpServerSpec, NetworkPolicy, PLATFORM_SCHEMA_VERSION, PreparationReceipt, ProgressEvent,
    ResourceSpec, RunIdentity, RunRecord, RunStatus, ScoringPolicy, SourceIdentity, TaskPackage,
    TimedOutput, TranscriptMessage, VerifierSpec, command_output_with_timeout,
};
