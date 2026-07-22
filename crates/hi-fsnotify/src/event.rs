//! Semantic filesystem event types.
//!
//! These are the wire types emitted by [`crate::FsEventSource`]. They represent
//! a single causal stream of semantic events derived from raw OS file
//! notifications.

use std::path::PathBuf;

/// One semantic event from the local workspace. Causal order on the source's
/// broadcast channel. `FilesChanged` paths share a single `kind`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
#[non_exhaustive]
pub enum FsEvent {
    /// Workspace file changes; all paths share `kind`. Paths under `.git/` are
    /// excluded (metadata surfaces as `GitMetaChanged`); `.lock` files dropped.
    FilesChanged {
        paths: Vec<PathBuf>,
        kind: FsEventKind,
    },

    /// A git metadata file changed (HEAD, index, refs/, FETCH_HEAD).
    GitMetaChanged { kind: GitMetaKind },

    /// VCS lock activity observed: `index.lock`/`gc.pid` present, or an event
    /// for one arrived with the file already gone. State in flux until the
    /// matching `GitOperationCompleted`.
    GitOperationStarted,

    /// Lock gone for [`SETTLE_MS`]: rapid lock cycles merge into one operation.
    /// `head_changed` reports whether `.git/HEAD` differs from its value when
    /// the operation's *first* lock appeared.
    GitOperationCompleted { head_changed: bool },
}

/// The kind of file change observed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FsEventKind {
    Created,
    #[default]
    Modified,
    Removed,
    Renamed,
}

/// Which git metadata file changed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GitMetaKind {
    /// `.git/HEAD` (branch switch, commit, rebase step).
    HeadChanged,
    /// `.git/index` (`git add`, `git reset`, `git commit`).
    IndexChanged,
    /// `.git/refs/*` or `.git/packed-refs`.
    RefsChanged,
    /// `.git/FETCH_HEAD` (fetch / pull).
    FetchHeadChanged,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fs_event_serde_roundtrip() {
        let e = FsEvent::FilesChanged {
            paths: vec![PathBuf::from("/tmp/a.rs")],
            kind: FsEventKind::Modified,
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: FsEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn git_meta_serde_roundtrip() {
        let e = FsEvent::GitMetaChanged {
            kind: GitMetaKind::HeadChanged,
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("head_changed"));
        let back: FsEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn git_op_events_serde() {
        let started = FsEvent::GitOperationStarted;
        let completed = FsEvent::GitOperationCompleted { head_changed: true };
        let j1 = serde_json::to_string(&started).unwrap();
        let j2 = serde_json::to_string(&completed).unwrap();
        assert!(j1.contains("git_operation_started"));
        assert!(j2.contains("git_operation_completed"));
        assert_eq!(started, serde_json::from_str(&j1).unwrap());
        assert_eq!(completed, serde_json::from_str(&j2).unwrap());
    }

    #[test]
    fn fs_event_kind_default() {
        assert_eq!(FsEventKind::default(), FsEventKind::Modified);
    }
}
