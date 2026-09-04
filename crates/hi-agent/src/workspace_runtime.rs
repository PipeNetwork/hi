//! Per-agent ownership boundary for all workspace-scoped state.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result, ensure};

use crate::LspMode;
use crate::change_ledger::ChangeLedger;

const RECONCILE_RUNNING: u8 = 0;
const RECONCILE_COMMITTING: u8 = 1;
const RECONCILE_CANCELLED: u8 = 2;
const RECONCILE_DONE: u8 = 3;

/// Arbitrates the only check/commit boundary in an async ledger reconcile.
/// Cancellation owns RUNNING work immediately; once the worker owns COMMITTING,
/// cancellation waits for its constant-size state swap to reach DONE.
struct ReconcileOwnership {
    state: AtomicU8,
    done_lock: Mutex<()>,
    done: Condvar,
}

impl ReconcileOwnership {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(RECONCILE_RUNNING),
            done_lock: Mutex::new(()),
            done: Condvar::new(),
        }
    }

    fn begin_commit(&self) -> bool {
        self.state
            .compare_exchange(
                RECONCILE_RUNNING,
                RECONCILE_COMMITTING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    fn finish(&self) {
        let _done = self
            .done_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.state.store(RECONCILE_DONE, Ordering::Release);
        self.done.notify_all();
    }

    fn cancel_or_wait_for_commit(&self, cancellation: &tokio_util::sync::CancellationToken) {
        loop {
            match self.state.load(Ordering::Acquire) {
                RECONCILE_RUNNING => {
                    if self
                        .state
                        .compare_exchange(
                            RECONCILE_RUNNING,
                            RECONCILE_CANCELLED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        cancellation.cancel();
                        return;
                    }
                }
                RECONCILE_COMMITTING => {
                    let mut done = self
                        .done_lock
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    while self.state.load(Ordering::Acquire) == RECONCILE_COMMITTING {
                        done = self
                            .done
                            .wait(done)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                    }
                    return;
                }
                RECONCILE_CANCELLED | RECONCILE_DONE => return,
                _ => unreachable!("invalid ledger reconcile ownership state"),
            }
        }
    }
}

/// Cancels the worker whenever its owning async future is dropped.
struct ReconcileFutureGuard {
    cancellation: tokio_util::sync::CancellationToken,
    ownership: Arc<ReconcileOwnership>,
    armed: bool,
}

impl ReconcileFutureGuard {
    fn new(
        cancellation: tokio_util::sync::CancellationToken,
        ownership: Arc<ReconcileOwnership>,
    ) -> Self {
        Self {
            cancellation,
            ownership,
            armed: true,
        }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for ReconcileFutureGuard {
    fn drop(&mut self) {
        if self.armed {
            self.ownership.cancel_or_wait_for_commit(&self.cancellation);
        }
    }
}

/// Guarantees that a panicking/early-returning worker cannot strand a waiter
/// in COMMITTING. Declare this before the ledger guard so unwinding unlocks the
/// ledger before publishing DONE.
struct ReconcileWorkerGuard {
    ownership: Arc<ReconcileOwnership>,
    finished: bool,
}

impl ReconcileWorkerGuard {
    fn new(ownership: Arc<ReconcileOwnership>) -> Self {
        Self {
            ownership,
            finished: false,
        }
    }

    fn finish(&mut self) {
        if !self.finished {
            self.ownership.finish();
            self.finished = true;
        }
    }
}

impl Drop for ReconcileWorkerGuard {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Runtime state that must never leak between agents or workspace roots.
pub struct WorkspaceRuntime {
    root: PathBuf,
    state_root: PathBuf,
    process_runner: hi_tools::ProcessRunner,
    lsp: Arc<hi_lsp::LspManager>,
    lsp_enabled: std::sync::atomic::AtomicBool,
    /// Auto mode only reuses a warm LSP server during fast feedback. Explicit
    /// `on` (or `/lsp on`) is allowed to pay the cold-start cost on purpose.
    lsp_fast_feedback_cold_start: std::sync::atomic::AtomicBool,
    /// Arc so a concurrent `/btw` side loop can hold a clone while the main
    /// turn keeps running (read-only inspect shares the same job registry).
    background: Arc<hi_tools::BackgroundRegistry>,
    read_cache: Arc<Mutex<hi_tools::ReadCache>>,
    repo_map: Arc<Mutex<hi_tools::RepoMapCache>>,
    ledger: Arc<Mutex<ChangeLedger>>,
    context_generation: std::sync::atomic::AtomicU64,
    // A deferred launch-root runtime is initially constructed without
    // repository hooks.  Keep this behind a lock so, once PipeFS authority
    // says that the ordinary local root is safe to use, we can populate the
    // trusted hook snapshot without rebuilding the agent and losing its
    // resumed task/checkpoint state.
    hooks: std::sync::RwLock<Option<Arc<hi_hooks::HookRegistry>>>,
}

impl WorkspaceRuntime {
    pub fn new(
        root: impl AsRef<Path>,
        state_root: impl AsRef<Path>,
        lsp_mode: LspMode,
    ) -> Result<Self> {
        Self::new_with_scan(root, state_root, lsp_mode, None)
    }

    /// Like [`Self::new`] but accepts a pre-started [`BackgroundScan`] so the
    /// initial workspace scan can overlap with all startup work before the
    /// runtime is constructed.
    pub fn new_with_scan(
        root: impl AsRef<Path>,
        state_root: impl AsRef<Path>,
        lsp_mode: LspMode,
        scan: Option<crate::change_ledger::BackgroundScan>,
    ) -> Result<Self> {
        Self::new_with_scan_and_sandbox(root, state_root, lsp_mode, scan, None)
    }

    /// Like [`Self::new_with_scan`] with an optional caller-owned sandbox
    /// policy. `None` preserves the normal `HI_SANDBOX` resolution used by
    /// standalone callers; an explicit policy avoids process-global
    /// environment reads for embedded agents and fixtures.
    pub fn new_with_scan_and_sandbox(
        root: impl AsRef<Path>,
        state_root: impl AsRef<Path>,
        lsp_mode: LspMode,
        scan: Option<crate::change_ledger::BackgroundScan>,
        sandbox_policy: Option<hi_tools::sandbox::SandboxPolicy>,
    ) -> Result<Self> {
        Self::new_with_scan_sandbox_and_project_hooks(
            root,
            state_root,
            lsp_mode,
            scan,
            sandbox_policy,
            true,
        )
    }

    /// Construct a runtime while optionally suppressing all repository-provided
    /// executable configuration. Portable workspaces restore untrusted bytes
    /// from another machine, so their `.hi/hooks` must not be admitted (or even
    /// trigger an stdin trust prompt) during a live root switch.
    pub fn new_with_scan_sandbox_and_project_hooks(
        root: impl AsRef<Path>,
        state_root: impl AsRef<Path>,
        lsp_mode: LspMode,
        scan: Option<crate::change_ledger::BackgroundScan>,
        sandbox_policy: Option<hi_tools::sandbox::SandboxPolicy>,
        allow_project_hooks: bool,
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
        let process_runner = match sandbox_policy {
            Some(policy) => hi_tools::ProcessRunner::new_with_policy(&root, policy)?,
            None => hi_tools::ProcessRunner::new(&root)?,
        };
        // In production, use a background scan (either a pre-started one passed
        // in by the caller, or one launched here). In tests, scan synchronously
        // so the initial snapshot is deterministic (tests write files
        // immediately after construction and expect `reconcile` to detect them
        // as external changes).
        let ledger = new_ledger(&root, &state_root, scan)?;
        let lsp = Arc::new(hi_lsp::LspManager::new(&root)?);
        if !matches!(lsp_mode, LspMode::Off)
            && let Ok(handle) = tokio::runtime::Handle::try_current()
        {
            let manager = lsp.clone();
            handle.spawn(async move {
                manager.set_enabled(true).await;
            });
        }
        let hooks = discover_hooks(&root, allow_project_hooks);
        Ok(Self {
            root: root.clone(),
            state_root,
            process_runner,
            lsp,
            lsp_enabled: std::sync::atomic::AtomicBool::new(!matches!(lsp_mode, LspMode::Off)),
            lsp_fast_feedback_cold_start: std::sync::atomic::AtomicBool::new(matches!(
                lsp_mode,
                LspMode::On
            )),
            background: Arc::new(hi_tools::BackgroundRegistry::default()),
            read_cache: Arc::new(Mutex::new(hi_tools::ReadCache::new())),
            repo_map: Arc::new(Mutex::new(hi_tools::RepoMapCache::new())),
            ledger: Arc::new(Mutex::new(ledger)),
            context_generation: std::sync::atomic::AtomicU64::new(0),
            hooks: std::sync::RwLock::new(hooks),
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

    /// The loaded hook registry, if any hooks were discovered.
    pub fn hooks(&self) -> Option<Arc<hi_hooks::HookRegistry>> {
        self.hooks
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Finish a deferred launch-root runtime after the PipeFS control plane
    /// has authoritatively selected the ordinary local workspace.  This does
    /// not replace workspace-scoped state, so a resumed transcript keeps its
    /// task context and undo/checkpoint references intact.
    pub fn activate_trusted_local_integrations(&self, lsp_mode: LspMode) {
        let hooks = discover_hooks(&self.root, true);
        *self
            .hooks
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = hooks;
        if !matches!(lsp_mode, LspMode::Off) {
            self.set_lsp_enabled(true);
        }
    }

    pub fn lsp(&self) -> Arc<hi_lsp::LspManager> {
        self.lsp.clone()
    }

    pub fn lsp_enabled(&self) -> bool {
        self.lsp_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn lsp_fast_feedback_cold_start_allowed(&self) -> bool {
        self.lsp_fast_feedback_cold_start
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set_lsp_enabled(&self, enabled: bool) {
        self.lsp_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
        self.lsp_fast_feedback_cold_start
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
        let manager = self.lsp();
        tokio::spawn(async move {
            manager.set_enabled(enabled).await;
        });
    }

    pub fn background(&self) -> &hi_tools::BackgroundRegistry {
        &self.background
    }

    /// Cloneable handle for concurrent side loops (`/btw`).
    pub fn background_arc(&self) -> Arc<hi_tools::BackgroundRegistry> {
        self.background.clone()
    }

    pub fn read_cache(&self) -> &Mutex<hi_tools::ReadCache> {
        &self.read_cache
    }

    pub fn read_cache_arc(&self) -> Arc<Mutex<hi_tools::ReadCache>> {
        self.read_cache.clone()
    }

    pub fn repo_map(&self) -> &Mutex<hi_tools::RepoMapCache> {
        &self.repo_map
    }

    pub fn repo_map_arc(&self) -> Arc<Mutex<hi_tools::RepoMapCache>> {
        self.repo_map.clone()
    }

    pub fn clear_read_cache(&self) {
        if let Ok(mut cache) = self.read_cache.lock() {
            cache.clear();
        }
    }

    pub fn clear_repo_map_cache(&self) {
        if let Ok(mut cache) = self.repo_map.lock() {
            cache.clear();
        }
    }

    pub fn ledger(&self) -> std::sync::MutexGuard<'_, ChangeLedger> {
        self.ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Try to inspect the ledger without ever parking the current async runtime
    /// thread. Abnormal-turn cleanup uses this after a bounded reconciliation:
    /// an OS-stalled filesystem worker must not defeat the turn's hard deadline.
    pub(crate) fn try_ledger(&self) -> Option<std::sync::MutexGuard<'_, ChangeLedger>> {
        match self.ledger.try_lock() {
            Ok(ledger) => Some(ledger),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => None,
        }
    }

    /// Wait briefly for a cooperatively-cancelled ledger worker to leave its
    /// critical section. The wait itself never takes the blocking mutex, so a
    /// stalled filesystem syscall cannot pin abnormal-turn cleanup.
    pub(crate) async fn wait_for_ledger_available(&self, timeout: std::time::Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if let Some(ledger) = self.try_ledger() {
                drop(ledger);
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }

    pub fn ledger_arc(&self) -> Arc<Mutex<ChangeLedger>> {
        self.ledger.clone()
    }

    /// Run [`ChangeLedger::reconcile`] on the blocking pool so a full workspace
    /// walk cannot freeze the TUI drive loop (which co-polls the agent future).
    pub async fn reconcile_ledger_async(&self) -> Result<Vec<hi_tools::FileChange>> {
        self.reconcile_ledger_paths_async(None).await
    }

    /// Wait for the initial ledger scan to finish. Verification uses this
    /// before reconciling the current workspace so a still-running startup scan
    /// cannot hide stage mutations from its before/after comparison.
    pub(crate) async fn ensure_ledger_scan_complete_async(&self) -> Result<()> {
        // The startup scan already owns its background thread. Poll its result
        // without joining while holding the ledger mutex: dropping this future
        // must immediately stop waiting, leaving the useful startup scan free to
        // finish for a later turn rather than detaching a mutex-owning waiter.
        loop {
            let complete = match self.ledger.try_lock() {
                Ok(mut ledger) => ledger.finish_background_scan_if_ready()?,
                Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                    poisoned.into_inner().finish_background_scan_if_ready()?
                }
                Err(std::sync::TryLockError::WouldBlock) => false,
            };
            if complete {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }

    /// Reconcile an exact set of known dirty paths without walking the entire
    /// workspace. Callers with opaque effects must use `reconcile_ledger_async`.
    pub async fn reconcile_dirty_paths_async(
        &self,
        paths: Vec<String>,
    ) -> Result<Vec<hi_tools::FileChange>> {
        self.reconcile_ledger_paths_async(Some(paths)).await
    }

    async fn reconcile_ledger_paths_async(
        &self,
        paths: Option<Vec<String>>,
    ) -> Result<Vec<hi_tools::FileChange>> {
        // `spawn_blocking` tasks cannot be force-aborted once running. Couple
        // the worker to this future with an operation token and explicit commit
        // ownership: dropping a cancelled/expired turn either prevents publish
        // or waits for the already-owned O(1) publish to finish.
        let cancellation = tokio_util::sync::CancellationToken::new();
        let ownership = Arc::new(ReconcileOwnership::new());
        let drop_guard = ReconcileFutureGuard::new(cancellation.clone(), ownership.clone());
        let worker_cancellation = cancellation.clone();
        let worker_ownership = ownership.clone();
        let ledger = self.ledger.clone();
        let joined = tokio::task::spawn_blocking(move || {
            // This guard is declared before the mutex guard so unwind releases
            // the ledger before waking a cancellation waiter.
            let mut worker_done = ReconcileWorkerGuard::new(worker_ownership.clone());
            let mut ledger = lock_ledger_cancellable(&ledger, &worker_cancellation)?;
            let prepared =
                ledger.prepare_reconcile_cancellable(paths.as_deref(), &worker_cancellation)?;
            if let Err(error) = ledger.await_reconcile_commit_ownership(&worker_cancellation) {
                drop(ledger);
                drop(prepared);
                return Err(error);
            }
            if !worker_ownership.begin_commit() {
                drop(ledger);
                drop(prepared);
                anyhow::bail!("workspace ledger reconcile cancelled before commit")
            }
            #[cfg(test)]
            ledger.await_owned_reconcile_commit_test_gate()?;
            let (changes, retired) = ledger.commit_prepared_reconcile(prepared);
            drop(ledger);
            // Cancellation waiting on COMMITTING may continue as soon as the
            // shared ledger is coherent and unlocked. Large retired maps and
            // history are destroyed afterward on this blocking worker.
            worker_done.finish();
            retired.discard();
            Ok(changes)
        })
        .await;
        drop_guard.disarm();
        joined.context("workspace ledger reconcile task panicked")?
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

/// Construct the change ledger with a background scan in production and a
/// synchronous scan in tests (where the initial snapshot must be deterministic).
/// If a pre-started `BackgroundScan` is provided, it is consumed instead of
/// launching a new one.
fn new_ledger(
    root: &Path,
    state_root: &Path,
    scan: Option<crate::change_ledger::BackgroundScan>,
) -> Result<ChangeLedger> {
    // An explicitly supplied production-style scan is honored in tests too.
    // Ordinary test runtimes still use a synchronous baseline below, while
    // focused cancellation tests can exercise the real startup path.
    if let Some(scan) = scan {
        return ChangeLedger::from_background_scan(root, Some(state_root), scan);
    }
    #[cfg(not(test))]
    {
        ChangeLedger::new_with_state_background(root, Some(state_root))
    }
    #[cfg(test)]
    {
        ChangeLedger::new_with_state(root, Some(state_root))
    }
}

/// Load global hooks and, when explicitly permitted, the local hook directory
/// only after folder trust has been resolved for this machine.  Keeping this in
/// one helper makes the initial and deferred runtime paths use identical trust
/// rules.
fn discover_hooks(root: &Path, allow_project_hooks: bool) -> Option<Arc<hi_hooks::HookRegistry>> {
    let home = std::env::var("HOME")
        .ok()
        .map(|h| std::path::Path::new(&h).join(".hi/hooks"));
    let project_hooks = root.join(".hi/hooks");
    let project_hooks_dir = if allow_project_hooks {
        match hi_tools::folder_trust::resolve_trust(root) {
            hi_tools::folder_trust::TrustOutcome::Trusted => Some(project_hooks.as_path()),
            hi_tools::folder_trust::TrustOutcome::Untrusted
            | hi_tools::folder_trust::TrustOutcome::Prompt => None,
        }
    } else {
        None
    };
    let (hooks, hook_errors) = hi_hooks::discover_hooks(home.as_deref(), project_hooks_dir);
    for err in &hook_errors {
        eprintln!("hook load warning: {err}");
    }
    (!hooks.is_empty()).then(|| Arc::new(hooks))
}

fn lock_ledger_cancellable<'a>(
    ledger: &'a Arc<Mutex<ChangeLedger>>,
    cancellation: &tokio_util::sync::CancellationToken,
) -> Result<std::sync::MutexGuard<'a, ChangeLedger>> {
    loop {
        match ledger.try_lock() {
            Ok(ledger) => return Ok(ledger),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                return Ok(poisoned.into_inner());
            }
            Err(std::sync::TryLockError::WouldBlock) if cancellation.is_cancelled() => {
                anyhow::bail!("workspace ledger reconcile cancelled while waiting for the ledger")
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
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
    use std::collections::BTreeSet;
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

    async fn wait_for_gate_count(
        gate: &crate::change_ledger::ScanTestGate,
        expected: usize,
        exited: bool,
    ) {
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let actual = if exited {
                    gate.exited()
                } else {
                    gate.entered()
                };
                if actual >= expected {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("ledger scan test gate was not reached");
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

    #[tokio::test]
    async fn dropping_full_reconcile_cancels_worker_and_releases_ledger() {
        let (root, state) = roots("cancel-full-reconcile");
        let runtime = Arc::new(WorkspaceRuntime::new(&root, &state, LspMode::Off).unwrap());
        std::fs::write(root.join("changed.txt"), "after baseline\n").unwrap();
        let gate = crate::change_ledger::install_scan_test_gate(&root);

        let worker_runtime = Arc::clone(&runtime);
        let reconcile = tokio::spawn(async move { worker_runtime.reconcile_ledger_async().await });
        wait_for_gate_count(&gate, 1, false).await;
        reconcile.abort();
        assert!(reconcile.await.unwrap_err().is_cancelled());
        wait_for_gate_count(&gate, 1, true).await;
        assert!(
            runtime
                .wait_for_ledger_available(std::time::Duration::from_millis(250))
                .await,
            "cancelled reconcile retained the ledger mutex"
        );

        gate.release();
        let changes = runtime.reconcile_ledger_async().await.unwrap();
        assert_eq!(
            changes
                .iter()
                .map(|change| change.path.as_str())
                .collect::<Vec<_>>(),
            vec!["changed.txt"]
        );
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    #[tokio::test]
    async fn cancellation_after_staging_preserves_the_change_for_retry() {
        let (root, state) = roots("cancel-reconcile-before-commit");
        let runtime = Arc::new(WorkspaceRuntime::new(&root, &state, LspMode::Off).unwrap());
        std::fs::write(root.join("changed.txt"), "after baseline\n").unwrap();
        let gate = crate::change_ledger::install_reconcile_commit_test_gate(&root);

        let worker_runtime = Arc::clone(&runtime);
        let reconcile = tokio::spawn(async move { worker_runtime.reconcile_ledger_async().await });
        wait_for_gate_count(&gate, 1, false).await;
        reconcile.abort();
        assert!(reconcile.await.unwrap_err().is_cancelled());
        wait_for_gate_count(&gate, 1, true).await;
        assert!(
            runtime
                .wait_for_ledger_available(std::time::Duration::from_millis(250))
                .await,
            "cancelled pre-commit reconcile retained the ledger mutex"
        );

        gate.release();
        let changes = runtime.reconcile_ledger_async().await.unwrap();
        assert_eq!(
            changes
                .iter()
                .map(|change| change.path.as_str())
                .collect::<Vec<_>>(),
            vec!["changed.txt"],
            "a cancelled worker must not consume the change before retry"
        );
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_waits_for_a_commit_that_already_owns_publication() {
        let (root, state) = roots("cancel-owned-reconcile-commit");
        let runtime = Arc::new(WorkspaceRuntime::new(&root, &state, LspMode::Off).unwrap());
        let baseline = runtime.ledger().revision();
        std::fs::write(root.join("changed.txt"), "after baseline\n").unwrap();
        let gate = crate::change_ledger::install_owned_reconcile_commit_test_gate(&root);

        let worker_runtime = Arc::clone(&runtime);
        let reconcile = tokio::spawn(async move { worker_runtime.reconcile_ledger_async().await });
        wait_for_gate_count(&gate, 1, false).await;
        reconcile.abort();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(
            !reconcile.is_finished(),
            "dropping the future returned while an owned commit was paused"
        );

        gate.release();
        let cancellation = tokio::time::timeout(std::time::Duration::from_secs(2), reconcile)
            .await
            .expect("owned commit did not release its cancellation waiter")
            .expect_err("aborted reconcile task unexpectedly returned normally");
        assert!(cancellation.is_cancelled());
        wait_for_gate_count(&gate, 1, true).await;
        assert!(runtime.try_ledger().is_some());
        assert_eq!(
            runtime
                .ledger()
                .changes_since(baseline)
                .into_iter()
                .map(|change| change.path)
                .collect::<Vec<_>>(),
            vec!["changed.txt"],
            "the owned commit must publish before cancellation settles"
        );
        assert!(
            runtime.reconcile_ledger_async().await.unwrap().is_empty(),
            "an owned commit must not leave its change for duplicate reconciliation"
        );
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    #[tokio::test]
    async fn dropping_exact_path_reconcile_cancels_hash_and_releases_ledger() {
        let (root, state) = roots("cancel-path-reconcile");
        std::fs::write(root.join("tracked.txt"), "baseline\n").unwrap();
        let runtime = Arc::new(WorkspaceRuntime::new(&root, &state, LspMode::Off).unwrap());
        std::fs::write(root.join("tracked.txt"), "changed\n").unwrap();
        let gate = crate::change_ledger::install_scan_test_gate(&root);

        let worker_runtime = Arc::clone(&runtime);
        let reconcile = tokio::spawn(async move {
            worker_runtime
                .reconcile_dirty_paths_async(vec!["tracked.txt".into()])
                .await
        });
        wait_for_gate_count(&gate, 1, false).await;
        reconcile.abort();
        assert!(reconcile.await.unwrap_err().is_cancelled());
        wait_for_gate_count(&gate, 1, true).await;
        assert!(
            runtime
                .wait_for_ledger_available(std::time::Duration::from_millis(250))
                .await,
            "cancelled exact-path reconcile retained the ledger mutex"
        );

        gate.release();
        let changes = runtime
            .reconcile_dirty_paths_async(vec!["tracked.txt".into()])
            .await
            .unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, "tracked.txt");
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    #[tokio::test]
    async fn cancelled_startup_scan_wait_never_owns_the_ledger_mutex() {
        let (root, state) = roots("cancel-startup-scan");
        std::fs::write(root.join("tracked.txt"), "baseline\n").unwrap();
        let gate = crate::change_ledger::install_scan_test_gate(&root);
        let scan = crate::BackgroundScan::start(&root, &[], &BTreeSet::new()).unwrap();
        wait_for_gate_count(&gate, 1, false).await;
        let runtime = Arc::new(
            WorkspaceRuntime::new_with_scan(&root, &state, LspMode::Off, Some(scan)).unwrap(),
        );

        let waiter_runtime = Arc::clone(&runtime);
        let waiter =
            tokio::spawn(async move { waiter_runtime.ensure_ledger_scan_complete_async().await });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        assert!(
            runtime.try_ledger().is_some(),
            "waiting for the startup scan held the ledger mutex"
        );

        drop(runtime);
        wait_for_gate_count(&gate, 1, true).await;
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }
}
