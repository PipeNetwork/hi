//! Per-path file-operation lock — serializes concurrent reads/writes.
//!
//! Port of grok-build's `FileOperationLockManager`:
//! - Per-path lock: same path is exclusive; different paths run together.
//! - Exclusive lock: blocks every per-path lock (multi-file patches).
//! - FIFO queue with exclusive-waiter priority so writers are not starved.

use std::collections::{HashSet, VecDeque};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use tokio::sync::oneshot;

/// Shared lock manager stored on the workspace runtime.
#[derive(Clone)]
pub struct FileOperationLockManager {
    inner: Arc<Mutex<LockInner>>,
}

struct LockInner {
    locked_files: HashSet<String>,
    exclusive_lock_active: bool,
    wait_queue: VecDeque<QueuedWaiter>,
}

enum QueuedWaiter {
    File {
        path: String,
        tx: oneshot::Sender<()>,
    },
    Exclusive {
        tx: oneshot::Sender<()>,
    },
}

enum LockKind {
    File(String),
    Exclusive,
}

/// RAII guard that releases the lock when dropped.
pub struct FileOperationLockGuard {
    manager: FileOperationLockManager,
    kind: LockKind,
}

/// How a tool call should lock the working tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileLockNeed {
    None,
    Exclusive,
    Path(String),
}

impl FileOperationLockManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(LockInner {
                locked_files: HashSet::new(),
                exclusive_lock_active: false,
                wait_queue: VecDeque::new(),
            })),
        }
    }

    /// Acquire a per-path lock. Blocks while an exclusive lock is held, the
    /// same path is locked, or an exclusive waiter is ahead in the queue.
    pub async fn wait_for_lock(&self, path: &str) -> FileOperationLockGuard {
        loop {
            let rx = {
                let mut inner = self.inner.lock().unwrap();
                let needs_wait = inner.exclusive_lock_active
                    || inner.locked_files.contains(path)
                    || inner.has_exclusive_waiter_ahead();
                if needs_wait {
                    let (tx, rx) = oneshot::channel();
                    inner.wait_queue.push_back(QueuedWaiter::File {
                        path: path.to_string(),
                        tx,
                    });
                    Some(rx)
                } else {
                    inner.locked_files.insert(path.to_string());
                    None
                }
            };
            let Some(rx) = rx else {
                return FileOperationLockGuard {
                    manager: self.clone(),
                    kind: LockKind::File(path.to_string()),
                };
            };
            if rx.await.is_ok() {
                return FileOperationLockGuard {
                    manager: self.clone(),
                    kind: LockKind::File(path.to_string()),
                };
            }
        }
    }

    /// Acquire an exclusive lock. Blocks until every per-path lock is released.
    pub async fn wait_for_exclusive_lock(&self) -> FileOperationLockGuard {
        loop {
            let rx = {
                let mut inner = self.inner.lock().unwrap();
                let needs_wait = inner.exclusive_lock_active || !inner.locked_files.is_empty();
                if needs_wait {
                    let (tx, rx) = oneshot::channel();
                    inner.wait_queue.push_back(QueuedWaiter::Exclusive { tx });
                    Some(rx)
                } else {
                    inner.exclusive_lock_active = true;
                    None
                }
            };
            let Some(rx) = rx else {
                return FileOperationLockGuard {
                    manager: self.clone(),
                    kind: LockKind::Exclusive,
                };
            };
            if rx.await.is_ok() {
                return FileOperationLockGuard {
                    manager: self.clone(),
                    kind: LockKind::Exclusive,
                };
            }
        }
    }

    /// Lock whatever this tool call needs, then run `fut`.
    pub async fn with_tool_lock<F, Fut, T>(
        &self,
        root: &Path,
        name: &str,
        arguments: &str,
        fut: F,
    ) -> T
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        let _guard = self.acquire_for_tool(root, name, arguments).await;
        fut().await
    }

    /// Hold a per-path lock across a fused mutation and its follow-up command.
    pub async fn with_path_lock<F, Fut, T>(&self, root: &Path, path: &str, fut: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        let key = normalize_lock_path(root, path);
        let _guard = self.wait_for_lock(&key).await;
        fut().await
    }

    pub async fn acquire_for_tool(
        &self,
        root: &Path,
        name: &str,
        arguments: &str,
    ) -> Option<FileOperationLockGuard> {
        match file_lock_need(name, arguments) {
            FileLockNeed::None => None,
            FileLockNeed::Exclusive => Some(self.wait_for_exclusive_lock().await),
            FileLockNeed::Path(path) => {
                let key = normalize_lock_path(root, &path);
                Some(self.wait_for_lock(&key).await)
            }
        }
    }
}

impl Default for FileOperationLockManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FileOperationLockGuard {
    fn drop(&mut self) {
        let kind = std::mem::replace(&mut self.kind, LockKind::Exclusive);
        if let Ok(mut inner) = self.manager.inner.lock() {
            match kind {
                LockKind::File(path) => {
                    inner.locked_files.remove(&path);
                }
                LockKind::Exclusive => {
                    inner.exclusive_lock_active = false;
                }
            }
            inner.process_queue();
        }
    }
}

impl LockInner {
    fn has_exclusive_waiter_ahead(&self) -> bool {
        self.wait_queue
            .iter()
            .any(|w| matches!(w, QueuedWaiter::Exclusive { .. }))
    }

    fn process_queue(&mut self) {
        while let Some(front) = self.wait_queue.front() {
            match front {
                QueuedWaiter::Exclusive { .. } => {
                    if !self.locked_files.is_empty() || self.exclusive_lock_active {
                        break;
                    }
                    if let Some(QueuedWaiter::Exclusive { tx }) = self.wait_queue.pop_front() {
                        self.exclusive_lock_active = true;
                        if tx.send(()).is_err() {
                            self.exclusive_lock_active = false;
                            continue;
                        }
                    }
                    break;
                }
                QueuedWaiter::File { path, .. } => {
                    if self.exclusive_lock_active || self.locked_files.contains(path) {
                        break;
                    }
                    let path = path.clone();
                    if let Some(QueuedWaiter::File { tx, .. }) = self.wait_queue.pop_front() {
                        self.locked_files.insert(path.clone());
                        if tx.send(()).is_err() {
                            self.locked_files.remove(&path);
                            continue;
                        }
                    }
                }
            }
        }
    }
}

/// Which lock a catalog tool needs. File tools with one path take a per-path
/// lock; multi-file `apply_patch` takes the exclusive lock so overlapping
/// writes cannot interleave.
pub fn file_lock_need(name: &str, arguments: &str) -> FileLockNeed {
    match name {
        "read" | "write" | "edit" | "multi_edit" => {
            let paths = extract_lock_paths(name, arguments);
            match paths.len() {
                0 => FileLockNeed::Exclusive,
                1 => FileLockNeed::Path(paths.into_iter().next().unwrap()),
                _ => FileLockNeed::Exclusive,
            }
        }
        "apply_patch" => {
            let paths = apply_patch_paths(arguments);
            match paths.len() {
                0 => FileLockNeed::Exclusive,
                1 => FileLockNeed::Path(paths.into_iter().next().unwrap()),
                _ => FileLockNeed::Exclusive,
            }
        }
        _ => FileLockNeed::None,
    }
}

fn extract_lock_paths(name: &str, arguments: &str) -> Vec<String> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return Vec::new();
    };
    if name == "read"
        && let Some(items) = value.get("paths").and_then(|v| v.as_array())
    {
        return items
            .iter()
            .filter_map(|item| item.as_str())
            .filter(|path| !path.is_empty())
            .map(str::to_string)
            .collect();
    }
    value
        .get("path")
        .and_then(|v| v.as_str())
        .filter(|path| !path.is_empty())
        .map(str::to_string)
        .into_iter()
        .collect()
}

fn apply_patch_paths(arguments: &str) -> Vec<String> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return Vec::new();
    };
    let Some(patch) = value.get("patch").and_then(|v| v.as_str()) else {
        return Vec::new();
    };
    let mut paths: Vec<String> = patch
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("*** Update File: ")
                .or_else(|| line.trim().strip_prefix("*** Add File: "))
                .or_else(|| line.trim().strip_prefix("*** Delete File: "))
                .map(str::trim)
                .filter(|path| !path.is_empty())
                .map(str::to_string)
        })
        .collect();
    paths.sort();
    paths.dedup();
    paths
}

pub fn normalize_lock_path(root: &Path, path: &str) -> String {
    let raw = Path::new(path);
    let joined = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        root.join(raw)
    };
    lexical_normalize(&joined).to_string_lossy().to_string()
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::RootDir => out.push(Component::RootDir),
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::Normal(s) => out.push(s),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn serializes_same_path() {
        let mgr = FileOperationLockManager::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        let guard1 = mgr.wait_for_lock("a.ts").await;
        let order2 = order.clone();
        let mgr2 = mgr.clone();
        let handle = tokio::spawn(async move {
            order2.lock().await.push("2-waiting");
            let _guard2 = mgr2.wait_for_lock("a.ts").await;
            order2.lock().await.push("2-acquired");
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        order.lock().await.push("1-releasing");
        drop(guard1);
        handle.await.unwrap();
        let log = order.lock().await;
        assert_eq!(log.as_slice(), &["2-waiting", "1-releasing", "2-acquired"]);
    }

    #[tokio::test]
    async fn allows_different_paths_concurrently() {
        let mgr = FileOperationLockManager::new();
        let _guard_a = mgr.wait_for_lock("a.ts").await;
        let mgr2 = mgr.clone();
        let handle = tokio::spawn(async move {
            let _guard_b = mgr2.wait_for_lock("b.ts").await;
            true
        });
        let result = tokio::time::timeout(std::time::Duration::from_millis(100), handle)
            .await
            .expect("should not timeout")
            .expect("should not panic");
        assert!(result);
    }

    #[tokio::test]
    async fn exclusive_blocks_file_locks() {
        let mgr = FileOperationLockManager::new();
        let exclusive = mgr.wait_for_exclusive_lock().await;
        let acquired = Arc::new(Mutex::new(false));
        let mgr2 = mgr.clone();
        let acquired2 = acquired.clone();
        let handle = tokio::spawn(async move {
            let _guard = mgr2.wait_for_lock("a.ts").await;
            *acquired2.lock().await = true;
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!*acquired.lock().await);
        drop(exclusive);
        handle.await.unwrap();
        assert!(*acquired.lock().await);
    }

    #[test]
    fn lock_need_classifies_file_tools() {
        assert_eq!(
            file_lock_need("write", r#"{"path":"a.rs","content":"x"}"#),
            FileLockNeed::Path("a.rs".into())
        );
        assert_eq!(
            file_lock_need("read", r#"{"paths":["a.rs","b.rs"]}"#),
            FileLockNeed::Exclusive
        );
        assert_eq!(
            file_lock_need("bash", r#"{"command":"echo"}"#),
            FileLockNeed::None
        );
        let patch = r#"{"patch":"*** Update File: a.rs\n*** Update File: b.rs\n"}"#;
        assert_eq!(
            file_lock_need("apply_patch", patch),
            FileLockNeed::Exclusive
        );
    }
}
