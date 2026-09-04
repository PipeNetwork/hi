//! Portable, versioned PipeFS archives plus the small IPOP durability client.
//!
//! PipeFS deliberately materializes ordinary directories. Native file tools,
//! processes, and Git continue to use the operating system filesystem; this
//! crate owns only portable snapshots, safe restoration, and remote durability.

mod archive;
mod causal;
mod client;
mod controller;
mod export;
mod lease;
mod migration;
mod workspace;

pub use archive::{
    ARCHIVE_VERSION, ArchiveArtifact, ArchiveEntry, ArchiveEntryKind, ArchiveManifest,
    RevisionKind, Snapshot, SnapshotEntry, StagedArchiveArtifact, apply_archive,
    apply_archive_file, build_revision, build_revision_from_snapshot_to_file,
    build_revision_from_snapshot_to_file_bounded, build_revision_to_file,
    build_revision_to_file_bounded, revision_archive_size_upper_bound, scan_workspace,
};
pub use causal::{
    CAUSAL_COMMIT_CAPABILITY, CAUSAL_WRITER_PROTOCOL, CausalCommitReceipt, CausalCommitRequest,
    CausalIntentReceipt, CausalIntentRequest, CausalOperationReceipt, CausalTranscriptRecord,
};
pub use client::{
    ArtifactDescriptor, PipeFsCacheScope, PipeFsCapabilities, PipeFsClient, PipeFsClientConfig,
    PipeFsError, PipeFsLease, PipeFsRemoteState, RestoreRevision, UploadedRevision,
};
pub use controller::{
    CausalTranscriptBatch, PipeFsControllerConfig, PipeFsLeaseStatus, PipeFsSessionBridge,
    PipeFsWorkspaceController, PipeFsWriterMode,
};
pub use export::{RemoteExportReceipt, export_remote_workspace};
pub use lease::LeaseReceipt;
pub use migration::{
    ImportPreview, ImportReceipt, PipeFsRecoveryReceipt, detach_if_clean, import_workspace,
    preview_import, retry_recovery_cache,
};
pub use workspace::{
    Activation, PipeFsRecoveryCache, PipeFsStatus, PipeFsWorkspace, PipeFsWorkspaceConfig,
    WorkspacePhase, discard_recovery_cache, export_recovery_cache, inspect_recovery_cache,
    list_recovery_caches, local_recovery_required, local_state_requires_remote_probe,
    record_local_mode_hint, recovery_cache_operation_evidence,
};
