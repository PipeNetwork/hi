//! Portable, versioned PipeFS archives plus the small IPOP durability client.
//!
//! PipeFS deliberately materializes ordinary directories. Native file tools,
//! processes, and Git continue to use the operating system filesystem; this
//! crate owns only portable snapshots, safe restoration, and remote durability.

mod archive;
mod client;
mod workspace;

pub use archive::{
    ARCHIVE_VERSION, ArchiveArtifact, ArchiveEntry, ArchiveEntryKind, ArchiveManifest,
    RevisionKind, Snapshot, SnapshotEntry, StagedArchiveArtifact, apply_archive,
    apply_archive_file, build_revision, build_revision_from_snapshot_to_file,
    build_revision_from_snapshot_to_file_bounded, build_revision_to_file,
    build_revision_to_file_bounded, revision_archive_size_upper_bound, scan_workspace,
};
pub use client::{
    ArtifactDescriptor, PipeFsCacheScope, PipeFsCapabilities, PipeFsClient, PipeFsClientConfig,
    PipeFsError, PipeFsLease, PipeFsRemoteState, RestoreRevision,
};
pub use workspace::{
    Activation, PipeFsRecoveryCache, PipeFsStatus, PipeFsWorkspace, PipeFsWorkspaceConfig,
    WorkspacePhase, discard_recovery_cache, export_recovery_cache, inspect_recovery_cache,
    list_recovery_caches, local_recovery_required, local_state_requires_remote_probe,
    record_local_mode_hint,
};
