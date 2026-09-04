//! Workspace change ledger shared by completion, verification, reports, and undo.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use hi_tools::{FileChange, FileChangeKind, ToolEffects};

// Automatic reconciliation is for source/configuration state, not model
// weights, database images, or other multi-gigabyte artifacts. Tool-mediated
// edits remain exact through `explicit_paths`, regardless of their size.
pub(crate) const MAX_AUTOMATIC_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_REVISION_EVENTS: usize = 512;

#[derive(Clone, Debug)]
struct FileState {
    digest: String,
    len: u64,
    mode: u32,
    /// Used only to skip re-hashing unchanged files during reconcile. Not part
    /// of content equality — a touch that preserves bytes must not look like a
    /// mutation once the digest is reused.
    mtime_ns: u64,
}

impl PartialEq for FileState {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest && self.len == other.len && self.mode == other.mode
    }
}

impl Eq for FileState {}

#[derive(Clone, Default)]
struct LedgerWindow {
    baseline: u64,
    changes: BTreeMap<String, FileChange>,
    touched_paths: BTreeSet<String>,
    had_mutation: bool,
}

impl LedgerWindow {
    fn at(baseline: u64) -> Self {
        Self {
            baseline,
            ..Self::default()
        }
    }

    fn record(&mut self, revision: u64, changes: &[FileChange]) {
        if revision <= self.baseline {
            return;
        }
        self.had_mutation = true;
        for change in changes {
            self.touched_paths.insert(change.path.clone());
            merge_change(&mut self.changes, change.clone());
        }
    }
}

/// A monotonically versioned account of all relevant workspace mutations.
pub struct ChangeLedger {
    root: PathBuf,
    excluded_roots: Vec<PathBuf>,
    /// Paths changed through typed tools remain observable even when they live
    /// below a hard-pruned generated/dependency directory.
    explicit_paths: BTreeSet<String>,
    revision: u64,
    observed: BTreeMap<String, FileState>,
    events: VecDeque<(u64, Vec<FileChange>)>,
    compacted_through: u64,
    dropped_events: u64,
    /// Exact lifetime net change and monotonic touched-path evidence. Their
    /// size follows distinct workspace paths, not repeated tool rounds.
    origin_changes: BTreeMap<String, FileChange>,
    lifetime_touched_paths: BTreeSet<String>,
    lifetime_had_mutation: bool,
    /// Exact aggregates for the only baselines that may legitimately remain
    /// live longer than the raw event window.
    active_turn_window: Option<LedgerWindow>,
    verification_window: Option<LedgerWindow>,
    /// Background initial workspace scan, launched at construction so it runs
    /// concurrently with agent/system-prompt setup. Async callers poll it
    /// without holding the ledger mutex; synchronous snapshot helpers consume
    /// it only once it is ready. Once consumed this is `None` and the ledger
    /// behaves exactly as before.
    pending_scan: Option<BackgroundScan>,
}

/// Fully staged replacement for one reconciliation. Building this may be
/// expensive, but publishing it only swaps owned collections and scalars.
pub(crate) struct PreparedReconcile {
    observed: BTreeMap<String, FileState>,
    changes: Vec<FileChange>,
    event: Option<PreparedReconcileEvent>,
}

struct PreparedReconcileEvent {
    revision: u64,
    events: VecDeque<(u64, Vec<FileChange>)>,
    compacted_through: u64,
    dropped_events: u64,
    origin_changes: BTreeMap<String, FileChange>,
    lifetime_touched_paths: BTreeSet<String>,
    lifetime_had_mutation: bool,
    active_turn_window: Option<LedgerWindow>,
    verification_window: Option<LedgerWindow>,
    /// Cloned history evicted while staging must not be dropped under the
    /// ledger mutex after commit ownership has been acquired.
    discarded_events: Vec<Vec<FileChange>>,
}

/// State replaced by an O(1) reconciliation commit. The worker releases the
/// ledger mutex and commit ownership before calling [`Self::discard`].
pub(crate) struct RetiredReconcile {
    observed: BTreeMap<String, FileState>,
    event: Option<RetiredReconcileEvent>,
}

struct RetiredReconcileEvent {
    events: VecDeque<(u64, Vec<FileChange>)>,
    origin_changes: BTreeMap<String, FileChange>,
    lifetime_touched_paths: BTreeSet<String>,
    active_turn_window: Option<LedgerWindow>,
    verification_window: Option<LedgerWindow>,
    discarded_events: Vec<Vec<FileChange>>,
}

impl RetiredReconcile {
    /// Make the intentionally deferred destruction explicit and keep every
    /// retired field visibly used for dead-code linting.
    pub(crate) fn discard(self) {
        let Self { observed, event } = self;
        drop(observed);
        if let Some(event) = event {
            let RetiredReconcileEvent {
                events,
                origin_changes,
                lifetime_touched_paths,
                active_turn_window,
                verification_window,
                discarded_events,
            } = event;
            drop((
                events,
                origin_changes,
                lifetime_touched_paths,
                active_turn_window,
                verification_window,
                discarded_events,
            ));
        }
    }
}

/// A handle to a workspace scan running in a background thread. Launch it as
/// early as possible in startup (right after the workspace and state roots are
/// resolved) so the scan overlaps with config resolution, provider construction,
/// and project-context loading. Pass it to [`ChangeLedger::from_background_scan`]
/// to build the ledger.
/// The shared cell a background scan writes its result into (a snapshot of every
/// tracked file's state, or the error that stopped it).
type ScanResult = Arc<Mutex<Option<Result<BTreeMap<String, FileState>>>>>;

pub struct BackgroundScan {
    join: Option<std::thread::JoinHandle<()>>,
    result: ScanResult,
    cancellation: CancellationToken,
}

impl BackgroundScan {
    /// Start scanning `root` (excluding `excluded_roots`) in a background
    /// thread. The owning ledger consumes the result once the thread finishes.
    pub fn start(
        root: &Path,
        excluded_roots: &[PathBuf],
        explicit_paths: &BTreeSet<String>,
    ) -> Result<Self> {
        let scan_root = root.to_path_buf();
        let scan_excluded = excluded_roots.to_vec();
        let scan_explicit = explicit_paths.clone();
        let result = Arc::new(Mutex::new(None));
        let result_handle = result.clone();
        let cancellation = CancellationToken::new();
        let worker_cancellation = cancellation.clone();
        let join = std::thread::Builder::new()
            .name("hi-ledger-scan".into())
            .spawn(move || {
                let scanned = scan_workspace(
                    &scan_root,
                    &scan_excluded,
                    &scan_explicit,
                    None,
                    Some(&worker_cancellation),
                );
                // Swallow a poisoned mutex rather than panicking the scan
                // thread: if the lock is poisoned the result is already lost,
                // and a panic here would be silently dropped by JoinHandle
                // (the only signal we'd get is a None result on join). Leaving
                // the cell as `None` lets the ledger surface the missing scan
                // result once the thread is observed as finished.
                if let Ok(mut slot) = result_handle.lock() {
                    *slot = Some(scanned);
                }
            })
            .context("spawning ledger scan thread")?;
        Ok(Self {
            join: Some(join),
            result,
            cancellation,
        })
    }
}

impl Drop for BackgroundScan {
    fn drop(&mut self) {
        // Dropping a std JoinHandle detaches its thread. Signal the scan first
        // so a runtime/session dropped during startup does not leave an
        // unowned workspace walk running indefinitely.
        self.cancellation.cancel();
    }
}

impl ChangeLedger {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        Self::new_with_state(root, None)
    }

    pub fn new_with_state(root: impl AsRef<Path>, state_root: Option<&Path>) -> Result<Self> {
        let root = root.as_ref().canonicalize().with_context(|| {
            format!("canonicalizing workspace root {}", root.as_ref().display())
        })?;
        ensure!(root.is_dir(), "workspace root is not a directory");
        let excluded_roots = state_root
            .and_then(|path| path.canonicalize().ok())
            .filter(|path| path.starts_with(&root))
            .into_iter()
            .collect::<Vec<_>>();
        let explicit_paths = BTreeSet::new();
        let observed = scan_workspace(&root, &excluded_roots, &explicit_paths, None, None)?;
        Ok(Self {
            root,
            excluded_roots,
            explicit_paths,
            revision: 0,
            observed,
            events: VecDeque::new(),
            compacted_through: 0,
            dropped_events: 0,
            origin_changes: BTreeMap::new(),
            lifetime_touched_paths: BTreeSet::new(),
            lifetime_had_mutation: false,
            active_turn_window: None,
            verification_window: None,
            pending_scan: None,
        })
    }

    /// Like [`Self::new_with_state`] but launches the initial workspace scan in
    /// a background thread so it runs concurrently with the rest of agent
    /// startup (system-prompt build, project-context loading, provider
    /// construction). The scan typically completes before the first turn's
    /// reconciliation needs the result; if not, the async runtime polls it
    /// without blocking a worker or the ledger mutex. Tests use the synchronous
    /// [`Self::new_with_state`] so the initial snapshot is deterministic.
    pub fn new_with_state_background(
        root: impl AsRef<Path>,
        state_root: Option<&Path>,
    ) -> Result<Self> {
        let root = root.as_ref().canonicalize().with_context(|| {
            format!("canonicalizing workspace root {}", root.as_ref().display())
        })?;
        ensure!(root.is_dir(), "workspace root is not a directory");
        let excluded_roots = state_root
            .and_then(|path| path.canonicalize().ok())
            .filter(|path| path.starts_with(&root))
            .into_iter()
            .collect::<Vec<_>>();
        let explicit_paths = BTreeSet::new();
        let scan = BackgroundScan::start(&root, &excluded_roots, &explicit_paths)?;
        Ok(Self {
            root,
            excluded_roots,
            explicit_paths,
            revision: 0,
            observed: BTreeMap::new(),
            events: VecDeque::new(),
            compacted_through: 0,
            dropped_events: 0,
            origin_changes: BTreeMap::new(),
            lifetime_touched_paths: BTreeSet::new(),
            lifetime_had_mutation: false,
            active_turn_window: None,
            verification_window: None,
            pending_scan: Some(scan),
        })
    }

    /// Construct a ledger that consumes a previously-started [`BackgroundScan`].
    /// This lets the caller launch the scan as early as possible (right after
    /// the workspace and state roots are known) so it overlaps with all
    /// subsequent startup work — config resolution, provider construction,
    /// project-context loading, and system-prompt building.
    pub fn from_background_scan(
        root: impl AsRef<Path>,
        state_root: Option<&Path>,
        scan: BackgroundScan,
    ) -> Result<Self> {
        let root = root.as_ref().canonicalize().with_context(|| {
            format!("canonicalizing workspace root {}", root.as_ref().display())
        })?;
        ensure!(root.is_dir(), "workspace root is not a directory");
        let excluded_roots = state_root
            .and_then(|path| path.canonicalize().ok())
            .filter(|path| path.starts_with(&root))
            .into_iter()
            .collect::<Vec<_>>();
        Ok(Self {
            root,
            excluded_roots,
            explicit_paths: BTreeSet::new(),
            revision: 0,
            observed: BTreeMap::new(),
            events: VecDeque::new(),
            compacted_through: 0,
            dropped_events: 0,
            origin_changes: BTreeMap::new(),
            lifetime_touched_paths: BTreeSet::new(),
            lifetime_had_mutation: false,
            active_turn_window: None,
            verification_window: None,
            pending_scan: Some(scan),
        })
    }

    /// Consume the background initial scan when it is already finished and seed
    /// `observed` with its result. After the first successful consume this is a
    /// no-op. Never blocks on the scan thread — a still-running scan leaves
    /// `pending_scan` in place so callers (especially the TUI/agent hot path)
    /// cannot freeze waiting for a large workspace walk.
    ///
    /// Methods that read or diff `observed` must call this first;
    /// `record_tool_effects` does not because it only touches explicit paths via
    /// `refresh_paths` (which overwrites those entries regardless of the initial
    /// scan).
    fn ensure_scan_complete(&mut self) -> Result<()> {
        let _ = self.finish_background_scan_if_ready()?;
        Ok(())
    }

    /// Consume a completed startup scan without ever waiting for its thread.
    /// Returns `false` while a scan is still running and `true` once there is no
    /// pending scan. The async runtime polls this method without holding the
    /// ledger mutex across an await, which keeps turn cancellation responsive.
    pub(crate) fn finish_background_scan_if_ready(&mut self) -> Result<bool> {
        let Some(scan) = &self.pending_scan else {
            return Ok(true);
        };
        let ready = scan
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some();
        let finished = scan.join.as_ref().is_some_and(|join| join.is_finished());
        if !ready && !finished {
            return Ok(false);
        }
        self.consume_finished_background_scan()?;
        Ok(true)
    }

    fn consume_finished_background_scan(&mut self) -> Result<()> {
        let mut scan = self
            .pending_scan
            .take()
            .expect("finished background scan checked above");
        // The result is written before normal thread exit. `is_finished` also
        // lets us join and surface a panic instead of polling forever when a
        // scanner exits without writing the result cell.
        if let Some(join) = scan.join.take() {
            join.join()
                .map_err(|_| anyhow::anyhow!("ledger scan thread panicked"))?;
        }
        let result = scan
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .unwrap_or_else(|| {
                Err(anyhow::anyhow!(
                    "ledger scan thread did not produce a result"
                ))
            });
        let scanned = result?;
        // Merge: explicit-path updates recorded via `record_tool_effects` before
        // the scan completed take precedence over the (now slightly stale)
        // background snapshot.
        for (path, state) in scanned {
            self.observed.entry(path).or_insert(state);
        }
        Ok(())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Establish the long-lived baseline for a new productive turn. Repeated
    /// mutations after this point are compacted into an exact per-path
    /// aggregate, so the raw diagnostic event deque can remain bounded.
    pub(crate) fn begin_turn_retention_window(&mut self) -> u64 {
        let baseline = self.revision;
        self.active_turn_window = Some(LedgerWindow::at(baseline));
        self.verification_window = None;
        baseline
    }

    /// Preserve exact deltas from the workspace revision a verification pass
    /// attested. This window is replaced when a later pass supersedes it.
    pub(crate) fn retain_verification_baseline(&mut self, baseline: u64) {
        let mut window = LedgerWindow::at(baseline);
        for (revision, changes) in self
            .events
            .iter()
            .filter(|(revision, _)| *revision > baseline)
        {
            window.record(*revision, changes);
        }
        self.verification_window = Some(window);
    }

    /// Number of raw revision events compacted out of the diagnostic deque.
    /// Aggregate correctness state remains exact for origin, active-turn, and
    /// current verification baselines.
    pub fn dropped_event_count(&self) -> u64 {
        self.dropped_events
    }

    /// The last-reconciled workspace listing as `(relative path, byte length)`,
    /// in path order. Already excludes hard-pruned trees (`target/`,
    /// `node_modules/`, VCS metadata). This accessor never waits for the
    /// background startup scan: until an async reconciliation has consumed that
    /// scan, the listing contains only state already observed explicitly.
    /// Turn settlement and the goal auditor reconcile first, so their snapshots
    /// are complete without making this synchronous accessor a blocking path.
    pub fn observed_files(&mut self) -> Vec<(String, u64)> {
        self.ensure_scan_complete().ok();
        self.observed
            .iter()
            .map(|(path, state)| (path.clone(), state.len))
            .collect()
    }

    /// Stable digest of the last-reconciled state currently available. Like
    /// [`Self::observed_files`], this never waits for a pending startup scan;
    /// correctness-sensitive callers await reconciliation before reading it.
    pub fn workspace_revision(&mut self) -> String {
        self.ensure_scan_complete().ok();
        let mut hash = Sha256::new();
        for (path, state) in &self.observed {
            hash.update(path.as_bytes());
            hash.update([0]);
            hash.update(state.digest.as_bytes());
            hash.update([0]);
            hash.update(state.len.to_le_bytes());
            hash.update(state.mode.to_le_bytes());
        }
        format!("ledger:v1:{:x}", hash.finalize())
    }

    /// Record a transactional tool result, then update the observed states for
    /// its exact paths. Failed/denied attempted mutations do not advance the
    /// revision because they applied no workspace effect. Applied net-zero
    /// mutations still advance it: validation policy depends on whether a
    /// mutation occurred, independently of the final diff.
    pub fn record_tool_effects(&mut self, effects: &ToolEffects) -> Result<u64> {
        // `refresh_paths` overwrites explicit-path entries regardless of the
        // initial scan, so we don't need to wait for it here. But if the scan
        // is still running, the explicit-path entry will be merged correctly by
        // `ensure_scan_complete`'s `or_insert` (explicit paths take precedence).
        if !effects.mutation_applied {
            return Ok(self.revision);
        }
        let mut changes = effects.file_changes.clone();
        changes.sort_by(|left, right| left.path.cmp(&right.path));
        self.explicit_paths
            .extend(changes.iter().map(|change| normalize(&change.path)));
        if !changes.is_empty() {
            self.refresh_paths(changes.iter().map(|change| change.path.as_str()))?;
        }
        self.push_event(changes);
        Ok(self.revision)
    }

    /// Detect foreground/background shell, delegate, user, or other external
    /// edits by comparing content digests rather than timestamps.
    pub fn reconcile(&mut self) -> Result<Vec<FileChange>> {
        let prepared = self.prepare_reconcile_paths(None, None)?;
        let (changes, retired) = self.commit_prepared_reconcile(prepared);
        retired.discard();
        Ok(changes)
    }

    /// Reconcile only known dirty paths when the caller has exact mutation
    /// attribution. Unknown shell/editor activity passes `None` and retains the
    /// full-scan correctness fallback.
    pub fn reconcile_dirty_paths(&mut self, paths: &[String]) -> Result<Vec<FileChange>> {
        let prepared = self.prepare_reconcile_paths(Some(paths), None)?;
        let (changes, retired) = self.commit_prepared_reconcile(prepared);
        retired.discard();
        Ok(changes)
    }

    pub(crate) fn prepare_reconcile_cancellable(
        &mut self,
        paths: Option<&[String]>,
        cancellation: &CancellationToken,
    ) -> Result<PreparedReconcile> {
        self.prepare_reconcile_paths(paths, Some(cancellation))
    }

    fn prepare_reconcile_paths(
        &mut self,
        paths: Option<&[String]>,
        cancellation: Option<&CancellationToken>,
    ) -> Result<PreparedReconcile> {
        ensure_scan_active(cancellation)?;
        // If the background initial scan is still running, try to consume its
        // result without blocking. If it's not ready yet, skip the re-scan for
        // this call — the initial scan IS the current state, so there's nothing
        // to diff against anyway. The next `reconcile` (after the scan finishes)
        // will do a proper diff. This keeps startup from blocking on the scan.
        if !self.finish_background_scan_if_ready()? {
            return Ok(PreparedReconcile {
                observed: self.observed.clone(),
                changes: Vec::new(),
                event: None,
            });
        }
        let current = if let Some(paths) = paths {
            let mut current = self.observed.clone();
            for relative in paths.iter().map(|path| normalize(path)) {
                ensure_scan_active(cancellation)?;
                let absolute = self.root.join(&relative);
                match read_state_cancellable(&absolute, self.observed.get(&relative), cancellation)?
                {
                    Some(state) => {
                        current.insert(relative, state);
                    }
                    None => {
                        current.remove(&relative);
                    }
                }
            }
            current
        } else {
            scan_workspace(
                &self.root,
                &self.excluded_roots,
                &self.explicit_paths,
                Some(&self.observed),
                cancellation,
            )?
        };
        ensure_scan_active(cancellation)?;
        let changes = diff_states(&self.observed, &current, cancellation)?;
        // Clone and update every potentially large aggregate before commit
        // ownership is acquired. Cancellation that wins while this staging is
        // running prevents publication; cancellation that loses waits only for
        // the constant-size swaps in `commit_prepared_reconcile`.
        ensure_scan_active(cancellation)?;
        let event = (!changes.is_empty()).then(|| self.prepare_reconcile_event(&changes));
        Ok(PreparedReconcile {
            observed: current,
            changes,
            event,
        })
    }

    fn prepare_reconcile_event(&self, changes: &[FileChange]) -> PreparedReconcileEvent {
        let revision = self.revision.saturating_add(1);
        let mut events = self.events.clone();
        let mut compacted_through = self.compacted_through;
        let mut dropped_events = self.dropped_events;
        let mut origin_changes = self.origin_changes.clone();
        let mut lifetime_touched_paths = self.lifetime_touched_paths.clone();
        let mut active_turn_window = self.active_turn_window.clone();
        let mut verification_window = self.verification_window.clone();
        for change in changes {
            lifetime_touched_paths.insert(change.path.clone());
            merge_change(&mut origin_changes, change.clone());
        }
        if let Some(window) = active_turn_window.as_mut() {
            window.record(revision, changes);
        }
        if let Some(window) = verification_window.as_mut() {
            window.record(revision, changes);
        }
        events.push_back((revision, changes.to_vec()));
        let mut discarded_events = Vec::new();
        while events.len() > MAX_REVISION_EVENTS {
            if let Some((event_revision, event_changes)) = events.pop_front() {
                compacted_through = event_revision;
                dropped_events = dropped_events.saturating_add(1);
                discarded_events.push(event_changes);
            }
        }
        PreparedReconcileEvent {
            revision,
            events,
            compacted_through,
            dropped_events,
            origin_changes,
            lifetime_touched_paths,
            lifetime_had_mutation: true,
            active_turn_window,
            verification_window,
            discarded_events,
        }
    }

    /// Last cancellable boundary before the caller arbitrates RUNNING versus
    /// COMMITTING ownership.
    pub(crate) fn await_reconcile_commit_ownership(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        #[cfg(test)]
        wait_for_reconcile_commit_test_gate(&self.root, Some(cancellation))?;
        ensure_scan_active(Some(cancellation))
    }

    /// Test-only pause after COMMITTING ownership is acquired. Production has
    /// no work between ownership arbitration and the O(1) state swap.
    #[cfg(test)]
    pub(crate) fn await_owned_reconcile_commit_test_gate(&self) -> Result<()> {
        wait_for_test_gate(&self.root, None, &OWNED_RECONCILE_COMMIT_TEST_GATES)
    }

    /// Publish a fully staged reconciliation using only collection/scalar
    /// swaps. Destruction of the replaced state is returned to the caller.
    pub(crate) fn commit_prepared_reconcile(
        &mut self,
        prepared: PreparedReconcile,
    ) -> (Vec<FileChange>, RetiredReconcile) {
        let PreparedReconcile {
            observed,
            changes,
            event,
        } = prepared;
        let retired_observed = std::mem::replace(&mut self.observed, observed);
        let retired_event = event.map(|event| {
            let PreparedReconcileEvent {
                revision,
                events,
                compacted_through,
                dropped_events,
                origin_changes,
                lifetime_touched_paths,
                lifetime_had_mutation,
                active_turn_window,
                verification_window,
                discarded_events,
            } = event;
            self.revision = revision;
            self.compacted_through = compacted_through;
            self.dropped_events = dropped_events;
            self.lifetime_had_mutation = lifetime_had_mutation;
            RetiredReconcileEvent {
                events: std::mem::replace(&mut self.events, events),
                origin_changes: std::mem::replace(&mut self.origin_changes, origin_changes),
                lifetime_touched_paths: std::mem::replace(
                    &mut self.lifetime_touched_paths,
                    lifetime_touched_paths,
                ),
                active_turn_window: std::mem::replace(
                    &mut self.active_turn_window,
                    active_turn_window,
                ),
                verification_window: std::mem::replace(
                    &mut self.verification_window,
                    verification_window,
                ),
                discarded_events,
            }
        });
        (
            changes,
            RetiredReconcile {
                observed: retired_observed,
                event: retired_event,
            },
        )
    }

    pub fn changes_since(&self, revision: u64) -> Vec<FileChange> {
        if revision == 0 {
            return self.origin_changes.values().cloned().collect();
        }
        if let Some(window) = self.window_for(revision) {
            return window.changes.values().cloned().collect();
        }
        let mut merged: BTreeMap<String, FileChange> = BTreeMap::new();
        // An unregistered caller asking beyond the retained raw window gets a
        // conservative origin delta rather than silently missing mutations.
        // Live turn/verification callers register their baselines above and
        // therefore remain exact regardless of turn length.
        if revision < self.compacted_through {
            return self.origin_changes.values().cloned().collect();
        }
        for (_, changes) in self.events.iter().filter(|(event, _)| *event > revision) {
            for change in changes {
                merge_change(&mut merged, change.clone());
            }
        }
        merged.into_values().collect()
    }

    pub fn changed_paths_since(&self, revision: u64) -> Vec<String> {
        self.changes_since(revision)
            .into_iter()
            .map(|change| change.path)
            .collect()
    }

    /// Every path touched after `revision`, without cancelling a later restore
    /// or create-then-delete pair. Verification uses this monotonic view;
    /// reports and diffs continue to use [`Self::changes_since`].
    pub fn touched_paths_since(&self, revision: u64) -> Vec<String> {
        if revision == 0 {
            return self.lifetime_touched_paths.iter().cloned().collect();
        }
        if let Some(window) = self.window_for(revision) {
            return window.touched_paths.iter().cloned().collect();
        }
        if revision < self.compacted_through {
            return self.lifetime_touched_paths.iter().cloned().collect();
        }
        self.events
            .iter()
            .filter(|(event, _)| *event > revision)
            .flat_map(|(_, changes)| changes.iter().map(|change| change.path.clone()))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Whether any applied or externally observed mutation occurred after the
    /// supplied revision, including an applied mutation with no net file diff.
    pub fn had_mutation_since(&self, revision: u64) -> bool {
        if revision == 0 {
            return self.lifetime_had_mutation;
        }
        if let Some(window) = self.window_for(revision) {
            return window.had_mutation;
        }
        if revision < self.compacted_through {
            return self.lifetime_had_mutation;
        }
        self.events.iter().any(|(event, _)| *event > revision)
    }

    fn window_for(&self, revision: u64) -> Option<&LedgerWindow> {
        self.verification_window
            .as_ref()
            .filter(|window| window.baseline == revision)
            .or_else(|| {
                self.active_turn_window
                    .as_ref()
                    .filter(|window| window.baseline == revision)
            })
    }

    fn refresh_paths<'a>(&mut self, paths: impl Iterator<Item = &'a str>) -> Result<()> {
        for relative in paths {
            let path = self.root.join(relative);
            // Tool-mediated paths always re-hash: the typed mutation is the
            // correctness boundary and must not reuse a stale fingerprint.
            match read_state(&path, None)? {
                Some(state) => {
                    self.observed.insert(normalize(relative), state);
                }
                None => {
                    self.observed.remove(&normalize(relative));
                }
            }
        }
        Ok(())
    }

    fn push_event(&mut self, changes: Vec<FileChange>) {
        self.revision = self.revision.saturating_add(1);
        self.lifetime_had_mutation = true;
        for change in &changes {
            self.lifetime_touched_paths.insert(change.path.clone());
            merge_change(&mut self.origin_changes, change.clone());
        }
        if let Some(window) = self.active_turn_window.as_mut() {
            window.record(self.revision, &changes);
        }
        if let Some(window) = self.verification_window.as_mut() {
            window.record(self.revision, &changes);
        }
        self.events.push_back((self.revision, changes));
        while self.events.len() > MAX_REVISION_EVENTS {
            if let Some((revision, _)) = self.events.pop_front() {
                self.compacted_through = revision;
                self.dropped_events = self.dropped_events.saturating_add(1);
            }
        }
    }
}

fn merge_change(merged: &mut BTreeMap<String, FileChange>, latest: FileChange) {
    match merged.entry(latest.path.clone()) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(latest);
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            let first = entry.get().clone();
            let kind = match (&first.before_digest, &latest.after_digest) {
                (None, Some(_)) => FileChangeKind::Create,
                (Some(_), None) => FileChangeKind::Delete,
                (Some(_), Some(_)) => FileChangeKind::Modify,
                (None, None) => {
                    entry.remove();
                    return;
                }
            };
            if first.before_digest == latest.after_digest && first.before_mode == latest.after_mode
            {
                entry.remove();
                return;
            }
            entry.insert(FileChange {
                path: latest.path,
                kind,
                before_digest: first.before_digest,
                after_digest: latest.after_digest,
                before_len: first.before_len,
                after_len: latest.after_len,
                before_mode: first.before_mode,
                after_mode: latest.after_mode,
            });
        }
    }
}

fn diff_states(
    before: &BTreeMap<String, FileState>,
    after: &BTreeMap<String, FileState>,
    cancellation: Option<&CancellationToken>,
) -> Result<Vec<FileChange>> {
    let mut paths = BTreeSet::new();
    for path in before.keys().chain(after.keys()) {
        ensure_scan_active(cancellation)?;
        paths.insert(path);
    }
    let mut changes = Vec::new();
    for path in paths {
        ensure_scan_active(cancellation)?;
        let old = before.get(path);
        let new = after.get(path);
        if old != new {
            changes.push(FileChange {
                path: path.clone(),
                kind: match (old, new) {
                    (None, Some(_)) => FileChangeKind::Create,
                    (Some(_), None) => FileChangeKind::Delete,
                    (Some(_), Some(_)) => FileChangeKind::Modify,
                    (None, None) => continue,
                },
                before_digest: old.map(|state| state.digest.clone()),
                after_digest: new.map(|state| state.digest.clone()),
                before_len: old.map(|state| state.len),
                after_len: new.map(|state| state.len),
                before_mode: old.map(|state| state.mode),
                after_mode: new.map(|state| state.mode),
            });
        }
    }
    Ok(changes)
}

fn scan_workspace(
    root: &Path,
    excluded_roots: &[PathBuf],
    explicit_paths: &BTreeSet<String>,
    previous: Option<&BTreeMap<String, FileState>>,
    cancellation: Option<&CancellationToken>,
) -> Result<BTreeMap<String, FileState>> {
    ensure_scan_active(cancellation)?;
    #[cfg(test)]
    wait_for_scan_test_gate(root, cancellation)?;
    let mut states = BTreeMap::new();
    let filter_root = root.to_path_buf();
    let filter_excluded = excluded_roots.to_vec();
    for result in ignore::WalkBuilder::new(root)
        .hidden(false)
        // The ledger is a correctness boundary, not a context index. Ignored
        // files such as `.env` and `.hi/config.toml` can still affect a task and
        // must invalidate verification.
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .ignore(false)
        .parents(false)
        .filter_entry(move |entry| !hard_pruned(&filter_root, &filter_excluded, entry.path()))
        .build()
    {
        ensure_scan_active(cancellation)?;
        let entry = match result {
            Ok(entry) => entry,
            // A concurrent test/editor can remove a directory between the
            // walker's parent read and descent. Treat only that transient
            // disappearance as a reconciliation race; permission, loop, and
            // other traversal failures remain visible to the caller.
            Err(error)
                if error
                    .io_error()
                    .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
            {
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| format!("walking workspace {}", root.display()));
            }
        };
        let path = entry.path();
        if path == root {
            continue;
        }
        let metadata = match std::fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            // A concurrent editor/test may remove an entry after the walker
            // yielded it. That is a normal reconciliation race; the next scan
            // will observe the deletion. Other traversal errors remain fatal.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("reading workspace entry {}", path.display()));
            }
        };
        if !metadata.is_file() && !metadata.file_type().is_symlink() {
            continue;
        }
        let relative = normalize(
            &path
                .strip_prefix(root)
                .expect("workspace walker escaped root")
                .to_string_lossy(),
        );
        if metadata.is_file() && metadata.len() > MAX_AUTOMATIC_FILE_BYTES {
            continue;
        }
        let prior = previous.and_then(|map| map.get(&relative));
        if let Some(state) = read_state_cancellable(path, prior, cancellation)? {
            states.insert(relative, state);
        }
    }
    // A typed mutation supplies an exact path. Keep tracking it even below a
    // pruned build/dependency tree so a later full reconciliation cannot turn
    // the just-recorded create into a synthetic deletion.
    for relative in explicit_paths {
        ensure_scan_active(cancellation)?;
        if vcs_relative_path(relative) {
            continue;
        }
        let path = root.join(relative);
        let prior = previous.and_then(|map| map.get(relative));
        if let Some(state) = read_state_cancellable(&path, prior, cancellation)? {
            states.insert(relative.clone(), state);
        } else {
            states.remove(relative);
        }
    }
    Ok(states)
}

fn read_state(path: &Path, previous: Option<&FileState>) -> Result<Option<FileState>> {
    read_state_cancellable(path, previous, None)
}

fn read_state_cancellable(
    path: &Path,
    previous: Option<&FileState>,
    cancellation: Option<&CancellationToken>,
) -> Result<Option<FileState>> {
    ensure_scan_active(cancellation)?;
    #[cfg(test)]
    wait_for_scan_test_gate(path, cancellation)?;
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("reading metadata for {}", path.display()));
        }
    };
    let mode = file_mode(&metadata);
    let mtime_ns = mtime_as_ns(&metadata);
    if metadata.file_type().is_symlink() {
        let target = std::fs::read_link(path)
            .with_context(|| format!("reading symlink {}", path.display()))?;
        let bytes = target.as_os_str().as_encoded_bytes().to_vec();
        return Ok(Some(FileState {
            digest: format!("symlink:sha256:{:x}", Sha256::digest(&bytes)),
            len: bytes.len() as u64,
            mode,
            mtime_ns,
        }));
    }
    if !metadata.is_file() {
        return Ok(None);
    }
    let len = metadata.len();
    // Cheap fingerprint: reuse the prior digest when len/mode/mtime match so
    // reconcile does not re-read and re-hash every unchanged source file.
    if let Some(prev) = previous
        && prev.len == len
        && prev.mode == mode
        && prev.mtime_ns == mtime_ns
    {
        return Ok(Some(prev.clone()));
    }
    if len > MAX_AUTOMATIC_FILE_BYTES {
        return Ok(Some(FileState {
            digest: format!("oversized:{len}"),
            len,
            mode,
            mtime_ns,
        }));
    }
    let (digest, hashed_len) = hash_file_streaming(path, cancellation)?;
    Ok(Some(FileState {
        digest,
        len: hashed_len,
        mode,
        mtime_ns,
    }))
}

fn hash_file_streaming(
    path: &Path,
    cancellation: Option<&CancellationToken>,
) -> Result<(String, u64)> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut hashed_len = 0u64;
    loop {
        ensure_scan_active(cancellation)?;
        let n = file
            .read(&mut buf)
            .with_context(|| format!("reading {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        hashed_len += n as u64;
    }
    Ok((format!("sha256:{:x}", hasher.finalize()), hashed_len))
}

fn ensure_scan_active(cancellation: Option<&CancellationToken>) -> Result<()> {
    ensure!(
        cancellation.is_none_or(|token| !token.is_cancelled()),
        "workspace ledger scan cancelled"
    );
    Ok(())
}

#[cfg(test)]
pub(crate) struct ScanTestGate {
    root: PathBuf,
    entered: std::sync::atomic::AtomicUsize,
    exited: std::sync::atomic::AtomicUsize,
    released: std::sync::atomic::AtomicBool,
}

#[cfg(test)]
impl ScanTestGate {
    pub(crate) fn entered(&self) -> usize {
        self.entered.load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn exited(&self) -> usize {
        self.exited.load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn release(&self) {
        self.released
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(test)]
static SCAN_TEST_GATES: std::sync::LazyLock<Mutex<Vec<std::sync::Weak<ScanTestGate>>>> =
    std::sync::LazyLock::new(|| Mutex::new(Vec::new()));

#[cfg(test)]
static RECONCILE_COMMIT_TEST_GATES: std::sync::LazyLock<Mutex<Vec<std::sync::Weak<ScanTestGate>>>> =
    std::sync::LazyLock::new(|| Mutex::new(Vec::new()));

#[cfg(test)]
static OWNED_RECONCILE_COMMIT_TEST_GATES: std::sync::LazyLock<
    Mutex<Vec<std::sync::Weak<ScanTestGate>>>,
> = std::sync::LazyLock::new(|| Mutex::new(Vec::new()));

#[cfg(test)]
pub(crate) fn install_scan_test_gate(root: &Path) -> Arc<ScanTestGate> {
    install_test_gate(root, &SCAN_TEST_GATES)
}

#[cfg(test)]
pub(crate) fn install_reconcile_commit_test_gate(root: &Path) -> Arc<ScanTestGate> {
    install_test_gate(root, &RECONCILE_COMMIT_TEST_GATES)
}

#[cfg(test)]
pub(crate) fn install_owned_reconcile_commit_test_gate(root: &Path) -> Arc<ScanTestGate> {
    install_test_gate(root, &OWNED_RECONCILE_COMMIT_TEST_GATES)
}

#[cfg(test)]
fn install_test_gate(
    root: &Path,
    gates: &Mutex<Vec<std::sync::Weak<ScanTestGate>>>,
) -> Arc<ScanTestGate> {
    let gate = Arc::new(ScanTestGate {
        root: root.canonicalize().unwrap_or_else(|_| root.to_path_buf()),
        entered: std::sync::atomic::AtomicUsize::new(0),
        exited: std::sync::atomic::AtomicUsize::new(0),
        released: std::sync::atomic::AtomicBool::new(false),
    });
    let mut gates = gates
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    gates.retain(|gate| gate.strong_count() > 0);
    gates.push(Arc::downgrade(&gate));
    gate
}

#[cfg(test)]
fn wait_for_reconcile_commit_test_gate(
    path: &Path,
    cancellation: Option<&CancellationToken>,
) -> Result<()> {
    wait_for_test_gate(path, cancellation, &RECONCILE_COMMIT_TEST_GATES)
}

#[cfg(test)]
fn wait_for_scan_test_gate(path: &Path, cancellation: Option<&CancellationToken>) -> Result<()> {
    wait_for_test_gate(path, cancellation, &SCAN_TEST_GATES)
}

#[cfg(test)]
fn wait_for_test_gate(
    path: &Path,
    cancellation: Option<&CancellationToken>,
    gates: &Mutex<Vec<std::sync::Weak<ScanTestGate>>>,
) -> Result<()> {
    let comparable_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let gate = {
        let mut gates = gates
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        gates.retain(|gate| gate.strong_count() > 0);
        gates
            .iter()
            .filter_map(std::sync::Weak::upgrade)
            .find(|gate| comparable_path.starts_with(&gate.root))
    };
    let Some(gate) = gate else {
        return Ok(());
    };
    gate.entered
        .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    while !gate.released.load(std::sync::atomic::Ordering::Acquire)
        && cancellation.is_none_or(|token| !token.is_cancelled())
    {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    gate.exited
        .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    ensure_scan_active(cancellation)
}

fn mtime_as_ns(metadata: &std::fs::Metadata) -> u64 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(unix)]
fn file_mode(metadata: &std::fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o7777
}

#[cfg(not(unix))]
fn file_mode(metadata: &std::fs::Metadata) -> u32 {
    if metadata.permissions().readonly() {
        0o444
    } else {
        0o666
    }
}

fn hard_pruned(root: &Path, excluded_roots: &[PathBuf], path: &Path) -> bool {
    if path == root {
        return false;
    }
    if excluded_roots
        .iter()
        .any(|excluded| path == excluded || path.starts_with(excluded))
    {
        return true;
    }
    // Weight / model caches are multi-gigabyte trees of shards. Walking them
    // (even without hashing large files) stalls every reconcile. Only prune a
    // `models/` cache (or `.hi/models/`) sitting at a project root — the
    // workspace root itself, or a nested repository root (recognized by its
    // `.git` marker, e.g. a workspace that contains several checkouts) — not
    // arbitrary `src/models` source directories.
    if let Ok(relative) = path.strip_prefix(root) {
        if relative.starts_with(".hi/state") {
            return true;
        }
        let mut components = relative
            .components()
            .filter_map(|component| match component {
                std::path::Component::Normal(name) => name.to_str(),
                _ => None,
            });
        match (components.next(), components.next()) {
            (Some("models"), _) => return true,
            (Some(".hi"), Some("models")) => return true,
            _ => {}
        }
        // Terminal-bench stores timestamped transcripts and per-run outputs
        // under this path. They are ignored by Git and can contain thousands
        // of source-looking files, but are not workspace source that an agent
        // can safely use for change attribution.
        let terminal_bench_jobs = std::path::Path::new("bench/terminal-bench/jobs");
        if relative == terminal_bench_jobs || relative.starts_with(terminal_bench_jobs) {
            return true;
        }
    }
    let name = path.file_name().and_then(|name| name.to_str());
    if name == Some("models")
        && let Some(parent) = path.parent()
    {
        let project_root = if parent.file_name().and_then(|n| n.to_str()) == Some(".hi") {
            parent.parent()
        } else {
            Some(parent)
        };
        if project_root.is_some_and(|owner| owner != root && owner.join(".git").exists()) {
            return true;
        }
    }
    if name.is_some_and(|name| {
        name.starts_with(".venv-")
            || name.starts_with("venv-")
            || name.starts_with("node_modules-")
            // Benchmark and tool runners commonly put their isolated Cargo
            // trees in names such as `.build-arm64`. These are generated
            // artifacts, not source directories; leaving them visible to the
            // correctness scan makes every turn walk hundreds of megabytes
            // of dependency fingerprints even though they are gitignored.
            || name.starts_with(".build-")
    }) {
        return true;
    }
    matches!(
        name,
        Some(
            ".git"
                | ".hg"
                | ".svn"
                | ".jj"
                | ".hi-eval-oracle"
                | ".cargo-home"
                | "target"
                | "node_modules"
                | "vendor"
                | ".venv"
                | "venv"
                | "dist"
                | "build"
                | ".next"
                | ".turbo"
                | "coverage"
                | "__pycache__"
                | ".pytest_cache"
                | ".mypy_cache"
                | ".ruff_cache"
        )
    )
}

fn vcs_relative_path(path: &str) -> bool {
    path.split('/')
        .any(|component| matches!(component, ".git" | ".hg" | ".svn" | ".jj"))
}

fn normalize(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    fn root(label: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "hi-ledger-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn external_changes_advance_revision_and_merge() {
        let root = root("external");
        std::fs::write(root.join("a.txt"), "one").unwrap();
        let mut ledger = ChangeLedger::new(&root).unwrap();
        let baseline = ledger.revision();
        std::fs::write(root.join("a.txt"), "two").unwrap();
        ledger.reconcile().unwrap();
        std::fs::write(root.join("a.txt"), "three").unwrap();
        ledger.reconcile().unwrap();
        let changes = ledger.changes_since(baseline);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].before_len, Some(3));
        assert_eq!(changes[0].after_len, Some(5));
        assert_eq!(ledger.revision(), 2);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn startup_scan_skips_large_artifacts_and_named_virtualenvs() {
        let root = root("bounded-startup");
        let large = std::fs::File::create(root.join("model.safetensors")).unwrap();
        large
            .set_len(MAX_AUTOMATIC_FILE_BYTES.saturating_add(1))
            .unwrap();
        std::fs::create_dir_all(root.join(".venv-wan/lib/python")).unwrap();
        std::fs::write(
            root.join(".venv-wan/lib/python/generated.py"),
            "value = 1\n",
        )
        .unwrap();
        std::fs::write(root.join("main.py"), "value = 2\n").unwrap();

        let ledger = ChangeLedger::new(&root).unwrap();

        assert!(ledger.observed.contains_key("main.py"));
        assert!(!ledger.observed.contains_key("model.safetensors"));
        assert!(
            !ledger
                .observed
                .contains_key(".venv-wan/lib/python/generated.py")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn nested_project_model_caches_are_pruned_but_source_models_dirs_are_not() {
        // A workspace containing several checkouts (root/proj with its own
        // .git) must not walk proj/models or proj/.hi/models weight caches —
        // the root-relative prune alone left multi-hundred-GB model trees in
        // scan scope. Source directories merely named models stay tracked.
        let root = root("nested-model-caches");
        std::fs::create_dir_all(root.join("proj/.git")).unwrap();
        std::fs::create_dir_all(root.join("proj/models")).unwrap();
        std::fs::create_dir_all(root.join("proj/.hi/models")).unwrap();
        std::fs::create_dir_all(root.join("proj/src/models")).unwrap();
        std::fs::write(root.join("proj/models/weights.json"), "w\n").unwrap();
        std::fs::write(root.join("proj/.hi/models/cache.json"), "c\n").unwrap();
        std::fs::write(root.join("proj/src/models/user.rs"), "struct U;\n").unwrap();

        let ledger = ChangeLedger::new(&root).unwrap();

        assert!(!ledger.observed.contains_key("proj/models/weights.json"));
        assert!(!ledger.observed.contains_key("proj/.hi/models/cache.json"));
        assert!(ledger.observed.contains_key("proj/src/models/user.rs"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn generated_build_caches_are_pruned_even_when_nested() {
        let root = root("nested-build-cache");
        std::fs::create_dir_all(root.join(".cargo-home/registry/src/dep")).unwrap();
        std::fs::create_dir_all(root.join(".hi/state/cargo-home/registry/src/dep")).unwrap();
        std::fs::create_dir_all(root.join("bench/.build-arm64/release")).unwrap();
        std::fs::create_dir_all(root.join("bench/src/build-tools")).unwrap();
        std::fs::write(
            root.join("bench/.build-arm64/release/fingerprint"),
            "generated\n",
        )
        .unwrap();
        std::fs::create_dir_all(root.join("bench/terminal-bench/jobs/2026-01-01")).unwrap();
        std::fs::write(
            root.join("bench/terminal-bench/jobs/2026-01-01/output.py"),
            "generated = True\n",
        )
        .unwrap();
        std::fs::write(root.join("bench/src/build-tools/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(
            root.join(".cargo-home/registry/src/dep/lib.rs"),
            "pub fn dependency() {}\n",
        )
        .unwrap();
        std::fs::write(
            root.join(".hi/state/cargo-home/registry/src/dep/lib.rs"),
            "pub fn runtime_dependency() {}\n",
        )
        .unwrap();

        let ledger = ChangeLedger::new(&root).unwrap();

        assert!(
            !ledger
                .observed
                .contains_key("bench/.build-arm64/release/fingerprint")
        );
        assert!(
            !ledger
                .observed
                .contains_key("bench/terminal-bench/jobs/2026-01-01/output.py")
        );
        assert!(
            ledger
                .observed
                .contains_key("bench/src/build-tools/main.rs")
        );
        assert!(
            !ledger
                .observed
                .contains_key(".cargo-home/registry/src/dep/lib.rs")
        );
        assert!(
            !ledger
                .observed
                .contains_key(".hi/state/cargo-home/registry/src/dep/lib.rs")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn touched_paths_and_mutation_events_survive_net_zero_effects() {
        let root = root("net-zero");
        let mut ledger = ChangeLedger::new(&root).unwrap();
        let baseline = ledger.revision();
        std::fs::write(root.join("temporary.rs"), "x\n").unwrap();
        ledger.reconcile().unwrap();
        std::fs::remove_file(root.join("temporary.rs")).unwrap();
        ledger.reconcile().unwrap();

        assert!(ledger.changes_since(baseline).is_empty());
        assert_eq!(ledger.touched_paths_since(baseline), vec!["temporary.rs"]);
        assert!(ledger.had_mutation_since(baseline));

        let before_empty_effect = ledger.revision();
        ledger
            .record_tool_effects(&ToolEffects {
                mutation_attempted: true,
                mutation_applied: true,
                file_changes: Vec::new(),
            })
            .unwrap();
        assert!(ledger.had_mutation_since(before_empty_effect));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn revision_evidence_survives_more_than_512_distinct_mutations() {
        let root = root("long-running-revision-evidence");
        let mut ledger = ChangeLedger::new(&root).unwrap();
        let baseline = ledger.revision();

        for index in 0..513 {
            ledger.push_event(vec![FileChange {
                path: format!("generated/{index}.rs"),
                kind: FileChangeKind::Create,
                before_digest: None,
                after_digest: Some(format!("sha256:{index}")),
                before_len: None,
                after_len: Some(index),
                before_mode: None,
                after_mode: Some(0o644),
            }]);
        }

        let changes = ledger.changes_since(baseline);
        assert_eq!(changes.len(), 513);
        assert_eq!(
            changes.first().map(|change| change.path.as_str()),
            Some("generated/0.rs")
        );
        assert_eq!(
            changes.last().map(|change| change.path.as_str()),
            Some("generated/99.rs"),
            "BTreeMap ordering is lexical, but every mutation must remain represented"
        );
        assert!(
            changes
                .iter()
                .any(|change| change.path == "generated/512.rs"),
            "settlement must retain evidence beyond the former 512-event window"
        );
        assert_eq!(ledger.touched_paths_since(baseline).len(), 513);
        assert!(ledger.had_mutation_since(baseline));
        assert_eq!(ledger.events.len(), MAX_REVISION_EVENTS);
        assert_eq!(ledger.dropped_event_count(), 1);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn compacted_events_preserve_active_turn_and_verification_baselines() {
        let root = root("compacted-active-baselines");
        let mut ledger = ChangeLedger::new(&root).unwrap();
        ledger.push_event(vec![FileChange {
            path: "before-turn.rs".into(),
            kind: FileChangeKind::Create,
            before_digest: None,
            after_digest: Some("old".into()),
            before_len: None,
            after_len: Some(1),
            before_mode: None,
            after_mode: Some(0o644),
        }]);
        let turn_baseline = ledger.begin_turn_retention_window();

        for index in 0..700 {
            ledger.push_event(vec![FileChange {
                path: "repeated.rs".into(),
                kind: FileChangeKind::Modify,
                before_digest: Some(format!("digest-{index}")),
                after_digest: Some(format!("digest-{}", index + 1)),
                before_len: Some(index),
                after_len: Some(index + 1),
                before_mode: Some(0o644),
                after_mode: Some(0o644),
            }]);
        }
        let verification_baseline = ledger.revision();
        ledger.retain_verification_baseline(verification_baseline);
        for index in 700..1_400 {
            ledger.push_event(vec![FileChange {
                path: "repeated.rs".into(),
                kind: FileChangeKind::Modify,
                before_digest: Some(format!("digest-{index}")),
                after_digest: Some(format!("digest-{}", index + 1)),
                before_len: Some(index),
                after_len: Some(index + 1),
                before_mode: Some(0o644),
                after_mode: Some(0o644),
            }]);
        }
        ledger.push_event(vec![FileChange {
            path: "temporary.rs".into(),
            kind: FileChangeKind::Create,
            before_digest: None,
            after_digest: Some("temporary".into()),
            before_len: None,
            after_len: Some(1),
            before_mode: None,
            after_mode: Some(0o644),
        }]);
        ledger.push_event(vec![FileChange {
            path: "temporary.rs".into(),
            kind: FileChangeKind::Delete,
            before_digest: Some("temporary".into()),
            after_digest: None,
            before_len: Some(1),
            after_len: None,
            before_mode: Some(0o644),
            after_mode: None,
        }]);

        assert_eq!(ledger.events.len(), MAX_REVISION_EVENTS);
        assert!(ledger.dropped_event_count() > 512);
        let turn_changes = ledger.changes_since(turn_baseline);
        assert_eq!(turn_changes.len(), 1);
        assert_eq!(turn_changes[0].path, "repeated.rs");
        assert_eq!(turn_changes[0].before_digest.as_deref(), Some("digest-0"));
        assert_eq!(turn_changes[0].after_digest.as_deref(), Some("digest-1400"));
        assert_eq!(
            ledger.touched_paths_since(turn_baseline),
            vec!["repeated.rs", "temporary.rs"]
        );
        assert!(ledger.had_mutation_since(turn_baseline));

        let verification_changes = ledger.changes_since(verification_baseline);
        assert_eq!(verification_changes.len(), 1);
        assert_eq!(
            verification_changes[0].before_digest.as_deref(),
            Some("digest-700")
        );
        assert_eq!(
            verification_changes[0].after_digest.as_deref(),
            Some("digest-1400")
        );
        assert_eq!(
            ledger.touched_paths_since(verification_baseline),
            vec!["repeated.rs", "temporary.rs"]
        );
        assert!(ledger.had_mutation_since(verification_baseline));
        assert!(
            ledger
                .touched_paths_since(0)
                .contains(&"before-turn.rs".to_string())
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn independent_roots_never_share_state() {
        let first = root("first");
        let second = root("second");
        let mut left = ChangeLedger::new(&first).unwrap();
        let mut right = ChangeLedger::new(&second).unwrap();
        std::fs::write(first.join("only-left"), "x").unwrap();
        left.reconcile().unwrap();
        right.reconcile().unwrap();
        assert_eq!(left.changed_paths_since(0), vec!["only-left"]);
        assert!(right.changed_paths_since(0).is_empty());
        assert_ne!(left.workspace_revision(), right.workspace_revision());
        let _ = std::fs::remove_dir_all(first);
        let _ = std::fs::remove_dir_all(second);
    }

    #[test]
    fn ignored_config_and_explicit_pruned_paths_remain_authoritative() {
        let root = root("ignored-explicit");
        std::fs::write(root.join(".gitignore"), ".env\ntarget/\n").unwrap();
        let state_root = root.join(".hi/state");
        std::fs::create_dir_all(&state_root).unwrap();
        let mut ledger = ChangeLedger::new_with_state(&root, Some(&state_root)).unwrap();
        let baseline = ledger.revision();

        std::fs::write(root.join(".env"), "TOKEN=test\n").unwrap();
        std::fs::create_dir_all(root.join(".hi")).unwrap();
        std::fs::write(root.join(".hi/config.toml"), "[quality]\n").unwrap();
        ledger.reconcile().unwrap();

        let generated = root.join("target/generated.txt");
        std::fs::create_dir_all(generated.parent().unwrap()).unwrap();
        std::fs::write(&generated, "generated\n").unwrap();
        let after = read_state(&generated, None).unwrap().unwrap();
        ledger
            .record_tool_effects(&ToolEffects {
                mutation_attempted: true,
                mutation_applied: true,
                file_changes: vec![FileChange {
                    path: "target/generated.txt".into(),
                    kind: FileChangeKind::Create,
                    before_digest: None,
                    after_digest: Some(after.digest),
                    before_len: None,
                    after_len: Some(after.len),
                    before_mode: None,
                    after_mode: Some(after.mode),
                }],
            })
            .unwrap();

        // A full scan prunes target/, but the typed exact path must supplement
        // it rather than manufacturing a deletion that cancels the create.
        ledger.reconcile().unwrap();
        let paths = ledger.changed_paths_since(baseline);
        assert!(paths.contains(&".env".to_string()), "{paths:?}");
        assert!(paths.contains(&".hi/config.toml".to_string()), "{paths:?}");
        assert!(
            paths.contains(&"target/generated.txt".to_string()),
            "{paths:?}"
        );

        std::fs::write(state_root.join("journal"), "runtime-only").unwrap();
        assert!(ledger.reconcile().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn weight_cache_trees_are_pruned_but_src_models_are_not() {
        let root = root("weight-cache");
        std::fs::create_dir_all(root.join("models/shard")).unwrap();
        std::fs::write(root.join("models/shard/a.bin"), "weights").unwrap();
        std::fs::create_dir_all(root.join(".hi/models/shard")).unwrap();
        std::fs::write(root.join(".hi/models/shard/b.bin"), "weights").unwrap();
        std::fs::create_dir_all(root.join("src/models")).unwrap();
        std::fs::write(root.join("src/models/user.rs"), "struct User;\n").unwrap();

        let ledger = ChangeLedger::new(&root).unwrap();
        assert!(ledger.observed.contains_key("src/models/user.rs"));
        assert!(!ledger.observed.contains_key("models/shard/a.bin"));
        assert!(!ledger.observed.contains_key(".hi/models/shard/b.bin"));
        let _ = std::fs::remove_dir_all(root);
    }
}
