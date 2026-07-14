//! Per-agent ownership boundary for all workspace-scoped state.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};

use crate::LspMode;
use crate::change_ledger::ChangeLedger;

/// Runtime state that must never leak between agents or workspace roots.
pub struct WorkspaceRuntime {
    root: PathBuf,
    state_root: PathBuf,
    process_runner: hi_tools::ProcessRunner,
    lsp: Arc<hi_lsp::LspManager>,
    lsp_enabled: std::sync::atomic::AtomicBool,
    background: hi_tools::BackgroundRegistry,
    read_cache: Mutex<hi_tools::ReadCache>,
    ledger: Mutex<ChangeLedger>,
    context_generation: std::sync::atomic::AtomicU64,
}

impl WorkspaceRuntime {
    pub fn new(
        root: impl AsRef<Path>,
        state_root: impl AsRef<Path>,
        lsp_mode: LspMode,
    ) -> Result<Self> {
        let root = root.as_ref().canonicalize().with_context(|| {
            format!("canonicalizing workspace root {}", root.as_ref().display())
        })?;
        ensure!(
            root.is_dir(),
            "workspace root is not a directory: {}",
            root.display()
        );
        let state_root = absolute_state_root(&root, state_root.as_ref());
        std::fs::create_dir_all(&state_root)
            .with_context(|| format!("creating workspace state root {}", state_root.display()))?;
        let state_root = state_root.canonicalize().with_context(|| {
            format!(
                "canonicalizing workspace state root {}",
                state_root.display()
            )
        })?;
        ensure!(
            state_root != root && !root.starts_with(&state_root),
            "workspace state root must be inside the workspace or disjoint from it, not equal to or an ancestor of {}",
            root.display()
        );
        hi_tools::recover_workspace_transactions(&root, &state_root)
            .context("recovering interrupted workspace transactions")?;
        let process_runner = hi_tools::ProcessRunner::new(&root)?;
        let ledger = ChangeLedger::new_with_state(&root, Some(&state_root))?;
        let lsp = Arc::new(hi_lsp::LspManager::new(&root));
        if !matches!(lsp_mode, LspMode::Off)
            && let Ok(handle) = tokio::runtime::Handle::try_current()
        {
            let manager = lsp.clone();
            handle.spawn(async move {
                manager.set_enabled(true).await;
            });
        }
        Ok(Self {
            root: root.clone(),
            state_root,
            process_runner,
            lsp,
            lsp_enabled: std::sync::atomic::AtomicBool::new(!matches!(lsp_mode, LspMode::Off)),
            background: hi_tools::BackgroundRegistry::default(),
            read_cache: Mutex::new(hi_tools::ReadCache::new()),
            ledger: Mutex::new(ledger),
            context_generation: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn state_root(&self) -> &Path {
        &self.state_root
    }

    pub fn process_runner(&self) -> &hi_tools::ProcessRunner {
        &self.process_runner
    }

    pub fn lsp(&self) -> Arc<hi_lsp::LspManager> {
        self.lsp.clone()
    }

    pub fn lsp_enabled(&self) -> bool {
        self.lsp_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set_lsp_enabled(&self, enabled: bool) {
        self.lsp_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
        let manager = self.lsp();
        tokio::spawn(async move {
            manager.set_enabled(enabled).await;
        });
    }

    pub fn background(&self) -> &hi_tools::BackgroundRegistry {
        &self.background
    }

    pub fn read_cache(&self) -> &Mutex<hi_tools::ReadCache> {
        &self.read_cache
    }

    pub fn clear_read_cache(&self) {
        if let Ok(mut cache) = self.read_cache.lock() {
            cache.clear();
        }
    }

    pub fn ledger(&self) -> std::sync::MutexGuard<'_, ChangeLedger> {
        self.ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn invalidate_context(&self) {
        self.context_generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Mark a transcript compaction boundary. The active turn consumes the
    /// same generation stream as workspace mutations before its next model
    /// request. Keeping one monotonic stream lets a burst of edits and
    /// compactions collapse into a single deterministic refresh.
    pub fn invalidate_context_after_compaction(&self) {
        self.invalidate_context();
    }

    pub fn context_generation(&self) -> u64 {
        self.context_generation
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn absolute_state_root(root: &Path, state_root: &Path) -> PathBuf {
    if state_root.is_absolute() {
        state_root.to_path_buf()
    } else {
        root.join(state_root)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    fn roots(label: &str) -> (PathBuf, PathBuf) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir().join(format!(
            "hi-runtime-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let root = base.join("workspace");
        let state = base.join("state");
        std::fs::create_dir_all(&root).unwrap();
        (root, state)
    }

    #[test]
    fn agents_in_different_roots_have_independent_state() {
        let (first_root, first_state) = roots("one");
        let (second_root, second_state) = roots("two");
        let first = WorkspaceRuntime::new(&first_root, &first_state, LspMode::Off).unwrap();
        let second = WorkspaceRuntime::new(&second_root, &second_state, LspMode::Off).unwrap();
        assert_ne!(first.root(), second.root());
        assert_ne!(first.state_root(), second.state_root());
        assert!(!Arc::ptr_eq(&first.lsp(), &second.lsp()));
        assert!(!std::ptr::eq(first.read_cache(), second.read_cache()));
        first.invalidate_context();
        assert_eq!(first.context_generation(), 1);
        assert_eq!(second.context_generation(), 0);
        first.invalidate_context_after_compaction();
        assert_eq!(first.context_generation(), 2);
        assert_eq!(second.context_generation(), 0);
        let _ = std::fs::remove_dir_all(first_root.parent().unwrap());
        let _ = std::fs::remove_dir_all(second_root.parent().unwrap());
    }

    #[tokio::test]
    async fn background_registries_are_workspace_local() {
        let (first_root, first_state) = roots("background-one");
        let (second_root, second_state) = roots("background-two");
        let first = WorkspaceRuntime::new(&first_root, &first_state, LspMode::Off).unwrap();
        let second = WorkspaceRuntime::new(&second_root, &second_state, LspMode::Off).unwrap();

        let id = first
            .background()
            .spawn(first.process_runner(), "sleep 600")
            .unwrap();
        assert_eq!(first.background().ids(), vec![id.clone()]);
        assert!(second.background().ids().is_empty());
        assert!(second.background().poll(&id).is_err());
        first.background().kill(&id).unwrap();

        let _ = std::fs::remove_dir_all(first_root.parent().unwrap());
        let _ = std::fs::remove_dir_all(second_root.parent().unwrap());
    }
}
