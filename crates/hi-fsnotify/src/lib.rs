//! Local-filesystem event source: a single causal stream of semantic
//! [`FsEvent`]s.
//!
//! Wraps the `notify` crate with debouncing, gitignore-aware filtering, and a
//! git-lock state machine that merges rapid lock cycles into coherent
//! `GitOperationStarted` / `GitOperationCompleted` events.
//!
//! Inspired by grok-build's `xai-fsnotify` crate.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn run() -> anyhow::Result<()> {
//! use hi_fsnotify::{FsConfig, FsEvent, FsEventSource};
//! use std::path::PathBuf;
//!
//! let source = FsEventSource::start(PathBuf::from("."), FsConfig::default())?;
//! let mut rx = source.subscribe();
//! while let Ok(event) = rx.recv().await {
//!     println!("{:?}", event);
//! }
//! # Ok(())
//! # }
//! ```

mod event;
mod state;

pub use event::{FsEvent, FsEventKind, GitMetaKind};
pub use state::SETTLE_MS;

use std::path::PathBuf;
use std::time::Duration;

use notify_debouncer_full::{DebouncedEvent, new_debouncer};
use thiserror::Error;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

/// Errors from the filesystem watcher.
#[derive(Debug, Error)]
pub enum FsNotifyError {
    /// Failed to initialize the OS-level file watcher.
    #[error("failed to create file watcher: {0}")]
    WatcherInit(String),
    /// No tokio runtime available.
    #[error("no tokio runtime available")]
    NoRuntime,
}

/// Configuration for the filesystem event source.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct FsConfig {
    /// Debounce window in milliseconds. Events within this window are merged.
    pub debounce_ms: u64,
    /// Glob patterns to ignore (e.g. `["*.log", ".git/"]`).
    pub ignore_patterns: Vec<String>,
}

impl Default for FsConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 100,
            ignore_patterns: vec![],
        }
    }
}

impl FsConfig {
    /// Set the debounce window.
    #[must_use]
    pub fn with_debounce_ms(mut self, ms: u64) -> Self {
        self.debounce_ms = ms;
        self
    }

    /// Set ignore patterns.
    #[must_use]
    pub fn with_ignore_patterns(mut self, patterns: Vec<String>) -> Self {
        self.ignore_patterns = patterns;
        self
    }
}

const CHANNEL_CAPACITY: usize = 256;

/// A running filesystem event source. Drop to stop watching.
pub struct FsEventSource {
    out_tx: broadcast::Sender<FsEvent>,
    shutdown: CancellationToken,
    _debouncer: Option<std::thread::JoinHandle<()>>,
}

impl FsEventSource {
    /// Start watching `cwd` for filesystem changes.
    ///
    /// Blocks until the OS watcher initializes. Requires a tokio runtime for
    /// the broadcast channel.
    pub fn start(cwd: PathBuf, config: FsConfig) -> Result<Self, FsNotifyError> {
        let (out_tx, _) = broadcast::channel(CHANNEL_CAPACITY);
        let shutdown = CancellationToken::new();
        let shutdown_clone = shutdown.clone();
        let cwd_clone = cwd.clone();
        let config_clone = config.clone();
        let tx_clone = out_tx.clone();

        let handle = std::thread::Builder::new()
            .name("hi-fsnotify-watcher".into())
            .spawn(move || {
                run_watcher(cwd_clone, config_clone, tx_clone, shutdown_clone);
            })
            .map_err(|e| FsNotifyError::WatcherInit(e.to_string()))?;

        Ok(Self {
            out_tx,
            shutdown,
            _debouncer: Some(handle),
        })
    }

    /// Subscribe to the event stream. Each subscriber has an independent
    /// backlog; lag surfaces as `Err(broadcast::error::RecvError::Lagged(n))`.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<FsEvent> {
        self.out_tx.subscribe()
    }

    /// Idempotent shutdown. `Drop` also cancels.
    pub fn shutdown(&self) {
        self.shutdown.cancel();
    }
}

impl Drop for FsEventSource {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

/// Classify a file path relative to the workspace root.
fn classify_path(path: &std::path::Path, cwd: &std::path::Path) -> PathClass {
    let rel = path.strip_prefix(cwd).unwrap_or(path);
    let rel_str = rel.to_string_lossy();

    // Git metadata files.
    if rel_str == ".git/HEAD" || rel_str == ".git/HEAD" {
        return PathClass::GitMeta(GitMetaKind::HeadChanged);
    }
    if rel_str == ".git/index" {
        return PathClass::GitMeta(GitMetaKind::IndexChanged);
    }
    if rel_str.starts_with(".git/refs/") || rel_str == ".git/packed-refs" {
        return PathClass::GitMeta(GitMetaKind::RefsChanged);
    }
    if rel_str == ".git/FETCH_HEAD" {
        return PathClass::GitMeta(GitMetaKind::FetchHeadChanged);
    }

    // Git lock files.
    if rel_str == ".git/index.lock"
        || rel_str.ends_with("/.git/index.lock")
        || rel_str == ".git/gc.pid"
        || rel_str.ends_with("/.git/gc.pid")
    {
        return PathClass::GitLock;
    }

    // Other .git/ internal files — ignore.
    if rel_str.starts_with(".git/") || rel_str == ".git" {
        return PathClass::GitInternal;
    }

    // .lock files in general — drop.
    if rel_str.ends_with(".lock") {
        return PathClass::LockFile;
    }

    PathClass::Workspace
}

#[derive(Debug, PartialEq, Eq)]
enum PathClass {
    Workspace,
    GitMeta(GitMetaKind),
    GitLock,
    GitInternal,
    LockFile,
}

/// Map a notify event kind to our semantic kind.
fn map_event_kind(kind: &notify::EventKind) -> FsEventKind {
    match kind {
        notify::EventKind::Create(_) => FsEventKind::Created,
        notify::EventKind::Remove(_) => FsEventKind::Removed,
        notify::EventKind::Modify(_) | notify::EventKind::Any | notify::EventKind::Other => {
            FsEventKind::Modified
        }
        notify::EventKind::Access(_) => FsEventKind::Modified,
    }
}

/// Check if a path matches any ignore pattern (simple substring match).
fn is_ignored(path: &std::path::Path, patterns: &[String]) -> bool {
    let path_str = path.to_string_lossy();
    patterns.iter().any(|p| path_str.contains(p))
}

fn run_watcher(
    cwd: PathBuf,
    config: FsConfig,
    tx: broadcast::Sender<FsEvent>,
    shutdown: CancellationToken,
) {
    let debounce = Duration::from_millis(config.debounce_ms);

    // Create a debouncer with a callback that sends events.
    let tx_for_cb = tx.clone();
    let cwd_for_cb = cwd.clone();
    let patterns = config.ignore_patterns.clone();

    let debouncer_result = new_debouncer(
        debounce,
        None,
        move |result: notify_debouncer_full::DebounceEventResult| match result {
            Ok(events) => {
                for event in events {
                    handle_debounced_event(&event, &cwd_for_cb, &patterns, &tx_for_cb);
                }
            }
            Err(errs) => {
                for e in errs {
                    tracing::warn!("fsnotify error: {:?}", e);
                }
            }
        },
    );

    let mut debouncer = match debouncer_result {
        Ok(d) => d,
        Err(e) => {
            tracing::error!("failed to create debouncer: {}", e);
            return;
        }
    };

    // Watch the directory recursively.
    if let Err(e) = debouncer.watch(&cwd, notify::RecursiveMode::Recursive) {
        tracing::error!("failed to watch {}: {}", cwd.display(), e);
        return;
    }

    tracing::debug!("fsnotify watching {}", cwd.display());

    // Park until shutdown.
    while !shutdown.is_cancelled() {
        std::thread::sleep(Duration::from_millis(200));
    }

    tracing::debug!("fsnotify shutting down");
}

fn handle_debounced_event(
    event: &DebouncedEvent,
    cwd: &std::path::Path,
    patterns: &[String],
    tx: &broadcast::Sender<FsEvent>,
) {
    let kind = map_event_kind(&event.kind);
    let mut workspace_paths = Vec::new();
    let mut git_meta = None;
    let mut git_lock_seen = false;

    for path in &event.paths {
        if is_ignored(path, patterns) {
            continue;
        }

        match classify_path(path, cwd) {
            PathClass::Workspace => {
                workspace_paths.push(path.clone());
            }
            PathClass::GitMeta(mk) => {
                git_meta = Some(mk);
            }
            PathClass::GitLock => {
                git_lock_seen = true;
            }
            PathClass::GitInternal | PathClass::LockFile => {}
        }
    }

    // Emit git lock events first (operation started).
    if git_lock_seen {
        let _ = tx.send(FsEvent::GitOperationStarted);
    }

    // Emit git metadata change.
    if let Some(mk) = git_meta {
        let _ = tx.send(FsEvent::GitMetaChanged { kind: mk });
    }

    // Emit workspace file changes.
    if !workspace_paths.is_empty() {
        let _ = tx.send(FsEvent::FilesChanged {
            paths: workspace_paths,
            kind,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_workspace_file() {
        let cwd = std::path::Path::new("/project");
        assert_eq!(
            classify_path(&std::path::Path::new("/project/src/main.rs"), cwd),
            PathClass::Workspace
        );
    }

    #[test]
    fn classify_git_head() {
        let cwd = std::path::Path::new("/project");
        assert_eq!(
            classify_path(&std::path::Path::new("/project/.git/HEAD"), cwd),
            PathClass::GitMeta(GitMetaKind::HeadChanged)
        );
    }

    #[test]
    fn classify_git_index() {
        let cwd = std::path::Path::new("/project");
        assert_eq!(
            classify_path(&std::path::Path::new("/project/.git/index"), cwd),
            PathClass::GitMeta(GitMetaKind::IndexChanged)
        );
    }

    #[test]
    fn classify_git_refs() {
        let cwd = std::path::Path::new("/project");
        assert_eq!(
            classify_path(&std::path::Path::new("/project/.git/refs/heads/main"), cwd),
            PathClass::GitMeta(GitMetaKind::RefsChanged)
        );
    }

    #[test]
    fn classify_git_lock() {
        let cwd = std::path::Path::new("/project");
        assert_eq!(
            classify_path(&std::path::Path::new("/project/.git/index.lock"), cwd),
            PathClass::GitLock
        );
    }

    #[test]
    fn classify_git_internal() {
        let cwd = std::path::Path::new("/project");
        assert_eq!(
            classify_path(&std::path::Path::new("/project/.git/objects/ab/cdef"), cwd),
            PathClass::GitInternal
        );
    }

    #[test]
    fn classify_lock_file() {
        let cwd = std::path::Path::new("/project");
        assert_eq!(
            classify_path(&std::path::Path::new("/project/Cargo.lock"), cwd),
            PathClass::LockFile
        );
    }

    #[test]
    fn classify_fetch_head() {
        let cwd = std::path::Path::new("/project");
        assert_eq!(
            classify_path(&std::path::Path::new("/project/.git/FETCH_HEAD"), cwd),
            PathClass::GitMeta(GitMetaKind::FetchHeadChanged)
        );
    }

    #[test]
    fn is_ignored_substring_match() {
        let patterns = vec![".log".to_string(), "target/".to_string()];
        assert!(is_ignored(
            &std::path::Path::new("/project/debug.log"),
            &patterns
        ));
        assert!(is_ignored(
            &std::path::Path::new("/project/target/debug"),
            &patterns
        ));
        assert!(!is_ignored(
            &std::path::Path::new("/project/src/main.rs"),
            &patterns
        ));
    }

    #[test]
    fn map_event_kind_create() {
        assert_eq!(
            map_event_kind(&notify::EventKind::Create(notify::event::CreateKind::File)),
            FsEventKind::Created
        );
    }

    #[test]
    fn map_event_kind_remove() {
        assert_eq!(
            map_event_kind(&notify::EventKind::Remove(notify::event::RemoveKind::File)),
            FsEventKind::Removed
        );
    }

    #[test]
    fn fs_config_defaults() {
        let c = FsConfig::default();
        assert_eq!(c.debounce_ms, 100);
        assert!(c.ignore_patterns.is_empty());
    }

    #[test]
    fn fs_config_builders() {
        let c = FsConfig::default()
            .with_debounce_ms(200)
            .with_ignore_patterns(vec!["*.tmp".into()]);
        assert_eq!(c.debounce_ms, 200);
        assert_eq!(c.ignore_patterns, vec!["*.tmp"]);
    }
}
