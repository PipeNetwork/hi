use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::WORKSPACE_CONTRACT_SCHEMA_VERSION;

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

string_id!(ControllerId);
string_id!(BindingId);
string_id!(WorkspaceId);
string_id!(OperationId);
string_id!(IdempotencyKey);
string_id!(JobId);
string_id!(CandidateId);
string_id!(RecoveryId);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkspaceAuthority {
    Local,
    PipeFs {
        session_id: String,
        writer_protocol: u16,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkspaceVersion {
    #[default]
    Unknown,
    Local {
        generation: u64,
        content_digest: Option<String>,
    },
    PipeFs {
        lease_generation: u64,
        head: Option<String>,
        manifest_digest: Option<String>,
        transcript_cursor: Option<u64>,
    },
}

impl WorkspaceVersion {
    pub fn next_local(&self, content_digest: Option<String>) -> Self {
        let generation = match self {
            Self::Local { generation, .. } => generation.saturating_add(1),
            Self::Unknown | Self::PipeFs { .. } => 1,
        };
        Self::Local {
            generation,
            content_digest,
        }
    }

    /// Advance the version without changing the workspace authority variant.
    /// A compatibility PipeFS settlement does not have the typed remote head
    /// receipt yet, so its observed content revision is retained as the
    /// manifest digest while the last acknowledged head and cursor remain.
    pub fn advance_after_settlement(&self, content_digest: Option<String>) -> Self {
        match self {
            Self::PipeFs {
                lease_generation,
                head,
                manifest_digest,
                transcript_cursor,
            } => Self::PipeFs {
                lease_generation: *lease_generation,
                head: head.clone(),
                manifest_digest: content_digest.or_else(|| manifest_digest.clone()),
                transcript_cursor: *transcript_cursor,
            },
            Self::Local { .. } | Self::Unknown => self.next_local(content_digest),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceBinding {
    pub schema_version: u16,
    pub controller_id: ControllerId,
    pub binding_id: BindingId,
    pub workspace_id: WorkspaceId,
    pub epoch: u64,
    pub workspace_root: PathBuf,
    pub state_root: PathBuf,
    pub authority: WorkspaceAuthority,
    pub version: WorkspaceVersion,
}

impl WorkspaceBinding {
    pub fn new_local(
        controller_id: ControllerId,
        workspace_id: WorkspaceId,
        workspace_root: PathBuf,
        state_root: PathBuf,
    ) -> Self {
        Self {
            schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
            controller_id,
            binding_id: BindingId::new(uuid::Uuid::new_v4().to_string()),
            workspace_id,
            epoch: 0,
            workspace_root,
            state_root,
            authority: WorkspaceAuthority::Local,
            version: WorkspaceVersion::Local {
                generation: 0,
                content_digest: None,
            },
        }
    }

    pub fn new_pipefs(
        controller_id: ControllerId,
        workspace_id: WorkspaceId,
        session_id: String,
        writer_protocol: u16,
        workspace_root: PathBuf,
        state_root: PathBuf,
    ) -> Self {
        Self {
            schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
            controller_id,
            binding_id: BindingId::new(uuid::Uuid::new_v4().to_string()),
            workspace_id,
            epoch: 0,
            workspace_root,
            state_root,
            authority: WorkspaceAuthority::PipeFs {
                session_id,
                writer_protocol,
            },
            version: WorkspaceVersion::PipeFs {
                lease_generation: 0,
                head: None,
                manifest_digest: None,
                transcript_cursor: None,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrashRecoveryCapability {
    None,
    LocalJournal,
    RemoteWorkspace,
    CausalWorkspaceAndTranscript,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceCapabilities {
    pub crash_recovery: CrashRecoveryCapability,
    pub candidate_apply: bool,
    pub background_writers: bool,
    pub causal_commit: bool,
    pub project_hooks: bool,
    pub repository_mcp_imports: bool,
    pub keep_background: bool,
    pub import_export: bool,
}

impl WorkspaceCapabilities {
    pub const fn in_memory() -> Self {
        Self {
            crash_recovery: CrashRecoveryCapability::None,
            candidate_apply: true,
            background_writers: true,
            causal_commit: false,
            project_hooks: false,
            repository_mcp_imports: false,
            keep_background: false,
            import_export: false,
        }
    }

    pub const fn local() -> Self {
        Self {
            crash_recovery: CrashRecoveryCapability::LocalJournal,
            candidate_apply: true,
            background_writers: true,
            causal_commit: false,
            project_hooks: true,
            repository_mcp_imports: true,
            keep_background: true,
            import_export: false,
        }
    }

    pub const fn pipefs(causal_commit: bool) -> Self {
        Self {
            crash_recovery: if causal_commit {
                CrashRecoveryCapability::CausalWorkspaceAndTranscript
            } else {
                CrashRecoveryCapability::RemoteWorkspace
            },
            candidate_apply: true,
            // Causal publication is necessary but not sufficient for a live
            // process writer. Until the host can pause the whole process
            // group, take a stable checkpoint, and resume only under the same
            // lease fence, PipeFS exposes detached candidates only.
            background_writers: false,
            causal_commit,
            project_hooks: false,
            repository_mcp_imports: false,
            keep_background: false,
            import_export: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceState {
    #[default]
    Ready,
    Mutating,
    Settling,
    PendingRemote,
    LeaseUncertain,
    LeaseLost,
    Conflict,
    TranscriptPending,
    CleanupPending,
    RecoveryRequired,
    JournalCorrupt,
    Incompatible,
    LocalAuditDegraded,
}

impl WorkspaceState {
    pub const fn admits_mutation(self) -> bool {
        matches!(self, Self::Ready)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceStatus {
    pub schema_version: u16,
    pub controller_id: ControllerId,
    pub binding_id: BindingId,
    pub epoch: u64,
    pub state: WorkspaceState,
    pub sequence: u64,
    pub active_operation: Option<OperationId>,
    pub active_jobs: Vec<JobId>,
    pub recovery_id: Option<RecoveryId>,
    pub detail: Option<String>,
}

impl WorkspaceStatus {
    pub fn ready(binding: &WorkspaceBinding) -> Self {
        Self {
            schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
            controller_id: binding.controller_id.clone(),
            binding_id: binding.binding_id.clone(),
            epoch: binding.epoch,
            state: WorkspaceState::Ready,
            sequence: 0,
            active_operation: None,
            active_jobs: Vec::new(),
            recovery_id: None,
            detail: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectScope {
    ReadOnly,
    CandidateOnly,
    LiveWriter,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReplayClass {
    PureWorkspace,
    IdempotentExternal { key: IdempotencyKey },
    NonReplayableExternal,
}

impl ReplayClass {
    /// Use the caller-supplied key for idempotent effects; other operations
    /// receive a fresh publication key owned by the controller.
    pub fn operation_idempotency_key(&self) -> IdempotencyKey {
        match self {
            Self::IdempotentExternal { key } => key.clone(),
            Self::PureWorkspace | Self::NonReplayableExternal => {
                IdempotencyKey::new(uuid::Uuid::new_v4().to_string())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationIntent {
    pub effect_scope: EffectScope,
    pub replay_class: ReplayClass,
    pub dirty_paths: Option<Vec<PathBuf>>,
    pub description: Option<String>,
}

impl MutationIntent {
    pub fn workspace(description: impl Into<String>) -> Self {
        Self {
            effect_scope: EffectScope::LiveWriter,
            replay_class: ReplayClass::PureWorkspace,
            dirty_paths: None,
            description: Some(description.into()),
        }
    }

    /// Admit a controller-owned scan/receipt boundary for effects produced by
    /// an already-registered live writer. This is the only operation allowed
    /// to overlap such a job; it cannot authorize a new workspace writer.
    pub fn reconciliation() -> Self {
        Self {
            effect_scope: EffectScope::ReadOnly,
            replay_class: ReplayClass::PureWorkspace,
            dirty_paths: None,
            description: Some("workspace reconciliation".into()),
        }
    }

    pub fn is_reconciliation(&self) -> bool {
        matches!(self.effect_scope, EffectScope::ReadOnly)
            && matches!(self.replay_class, ReplayClass::PureWorkspace)
            && self.dirty_paths.is_none()
            && matches!(
                self.description.as_deref(),
                Some("workspace reconciliation")
            )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationPermitRecord {
    pub schema_version: u16,
    pub controller_id: ControllerId,
    pub operation_id: OperationId,
    pub idempotency_key: IdempotencyKey,
    pub binding_id: BindingId,
    pub epoch: u64,
    pub base_version: WorkspaceVersion,
    pub intent: MutationIntent,
    pub issued_at_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDisposition {
    Succeeded,
    Failed,
    Cancelled,
    Indeterminate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub uri: String,
    pub digest: Option<String>,
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub disposition: ExecutionDisposition,
    pub workspace_may_have_changed: bool,
    pub external_effect_may_have_occurred: bool,
    pub content_digest: Option<String>,
    pub changed_paths: Vec<PathBuf>,
    pub artifacts: Vec<ArtifactRef>,
    pub detail: Option<String>,
}

impl ExecutionReport {
    pub fn succeeded(content_digest: Option<String>) -> Self {
        Self {
            disposition: ExecutionDisposition::Succeeded,
            workspace_may_have_changed: content_digest.is_some(),
            external_effect_may_have_occurred: false,
            content_digest,
            changed_paths: Vec::new(),
            artifacts: Vec::new(),
            detail: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettlementStatus {
    Durable,
    NoChange,
    Pending,
    Indeterminate,
    LeaseLost,
    Conflict,
    TranscriptPending,
    RecoveryRequired,
    LocalAuditDegraded,
    Incompatible,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettlementReceipt {
    pub receipt_id: String,
    pub operation_id: OperationId,
    pub binding_id: BindingId,
    pub epoch: u64,
    pub version: WorkspaceVersion,
    pub transcript_cursor: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettlementOutcome {
    pub status: SettlementStatus,
    pub operation_id: OperationId,
    pub receipt: Option<SettlementReceipt>,
    pub recovery_id: Option<RecoveryId>,
    pub detail: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionDeniedReason {
    NotReady,
    ActiveMutation,
    ActiveWriter,
    StaleBinding,
    Incompatible,
    CapabilityUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("workspace admission denied ({reason:?}): {detail}")]
pub struct AdmissionDenied {
    pub reason: AdmissionDeniedReason,
    pub state: WorkspaceState,
    pub detail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Process,
    ReadAgent,
    WriteCandidate,
    Hook,
    Compaction,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobLimits {
    pub queue_ms: Option<u64>,
    pub execution_ms: Option<u64>,
    pub verification_ms: Option<u64>,
    pub output_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobSpec {
    pub kind: JobKind,
    pub effect_scope: EffectScope,
    pub name: String,
    pub limits: JobLimits,
    pub parent_operation: Option<OperationId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Starting,
    Running,
    ReadyToMerge,
    Merging,
    Settling,
    CancelRequested,
    Succeeded,
    Failed,
    Cancelled,
    DurabilityPending,
    RecoveryRequired,
    Orphaned,
    Stale,
}

impl JobState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::Orphaned | Self::Stale
        )
    }

    pub const fn can_transition_to(self, next: Self) -> bool {
        use JobState::*;
        match self {
            Queued => matches!(next, Starting | Cancelled | Failed | Orphaned),
            Starting => matches!(
                next,
                Running | CancelRequested | Cancelled | Failed | Orphaned
            ),
            Running => matches!(
                next,
                ReadyToMerge
                    | Settling
                    | CancelRequested
                    | Succeeded
                    | Failed
                    | Cancelled
                    | DurabilityPending
                    | RecoveryRequired
                    | Orphaned
            ),
            ReadyToMerge => matches!(next, Merging | Cancelled | Failed | Stale),
            Merging => matches!(next, Settling | Failed | RecoveryRequired | Stale),
            Settling => matches!(
                next,
                Succeeded | Failed | Cancelled | DurabilityPending | RecoveryRequired
            ),
            CancelRequested => matches!(
                next,
                Settling | Cancelled | Failed | DurabilityPending | RecoveryRequired | Orphaned
            ),
            DurabilityPending => matches!(next, Settling | Succeeded | Failed | RecoveryRequired),
            RecoveryRequired => matches!(next, Settling | Succeeded | Failed | Cancelled | Stale),
            Succeeded | Failed | Cancelled | Orphaned | Stale => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobPermit {
    pub schema_version: u16,
    pub controller_id: ControllerId,
    pub job_id: JobId,
    pub binding_id: BindingId,
    pub epoch: u64,
    pub spec: JobSpec,
    pub issued_at_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobCompletion {
    Succeeded,
    ReadyToMerge,
    Merging,
    Settling,
    Failed,
    Cancelled,
    DurabilityPending,
    RecoveryRequired,
    Stale,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobTerminal {
    pub completion: JobCompletion,
    pub detail: Option<String>,
    pub artifacts: Vec<ArtifactRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobSealStatus {
    Sealed,
    AlreadySealed,
    Rejected,
    NotFound,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobSealOutcome {
    pub job_id: JobId,
    pub status: JobSealStatus,
    pub state: Option<JobState>,
    pub recovery_id: Option<RecoveryId>,
    pub detail: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BarrierKind {
    Rebind,
    ModeSwitch,
    Exit,
    Checkpoint,
    Publish,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BarrierStatus {
    Passed,
    Blocked,
    TimedOut,
    RecoveryRequired,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BarrierReceipt {
    pub kind: BarrierKind,
    pub status: BarrierStatus,
    pub binding_id: BindingId,
    pub epoch: u64,
    pub active_operation: Option<OperationId>,
    pub pending_jobs: Vec<JobId>,
    pub recovery_id: Option<RecoveryId>,
    pub detail: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryKind {
    AbandonedMutation,
    UnsettledMutation,
    CrashedWriterJob,
    LeaseLost,
    Conflict,
    InterruptedRestore,
    TranscriptPending,
    CleanupPending,
    JournalFailure,
    IncompatibleState,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryRecord {
    pub schema_version: u16,
    pub recovery_id: RecoveryId,
    pub kind: RecoveryKind,
    pub binding_id: BindingId,
    pub epoch: u64,
    pub operation_id: Option<OperationId>,
    pub job_id: Option<JobId>,
    pub detail: String,
    pub created_at_ms: u64,
    pub resolved: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryStatus {
    Recovered,
    Pending,
    Conflict,
    NotFound,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryOutcome {
    pub recovery_id: RecoveryId,
    pub status: RecoveryStatus,
    pub binding: WorkspaceBinding,
    pub detail: Option<String>,
}
