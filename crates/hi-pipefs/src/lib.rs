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
    RevisionKind, Snapshot, SnapshotEntry, apply_archive, build_revision, scan_workspace,
};
pub use client::{
    ArtifactDescriptor, PipeFsCapabilities, PipeFsClient, PipeFsClientConfig, PipeFsError,
    PipeFsLease, PipeFsRemoteState, RestoreRevision,
};
pub use workspace::{
    Activation, PipeFsStatus, PipeFsWorkspace, PipeFsWorkspaceConfig, WorkspacePhase,
    local_recovery_required, local_state_requires_remote_probe, record_local_mode_hint,
};
