//! Workspace change ledger shared by completion, verification, reports, and undo.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};

use hi_tools::{FileChange, FileChangeKind, ToolEffects};

const MAX_REVISION_EVENTS: usize = 512;
// Automatic reconciliation is for source/configuration state, not model
// weights, database images, or other multi-gigabyte artifacts. Tool-mediated
// edits remain exact through `explicit_paths`, regardless of their size.
const MAX_AUTOMATIC_FILE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileState {
    digest: String,
    len: u64,
    mode: u32,
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
        let observed = scan_workspace(&root, &excluded_roots, &explicit_paths)?;
        Ok(Self {
            root,
            excluded_roots,
            explicit_paths,
            revision: 0,
            observed,
            events: VecDeque::new(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Stable digest of the last reconciled workspace state.
    pub fn workspace_revision(&self) -> String {
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
        let current = scan_workspace(&self.root, &self.excluded_roots, &self.explicit_paths)?;
        let changes = diff_states(&self.observed, &current);
        self.observed = current;
        if !changes.is_empty() {
            self.push_event(changes.clone());
        }
        Ok(changes)
    }

    pub fn changes_since(&self, revision: u64) -> Vec<FileChange> {
        let mut merged: BTreeMap<String, FileChange> = BTreeMap::new();
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
        self.events.iter().any(|(event, _)| *event > revision)
    }

    fn refresh_paths<'a>(&mut self, paths: impl Iterator<Item = &'a str>) -> Result<()> {
        for relative in paths {
            let path = self.root.join(relative);
            match read_state(&path)? {
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
        self.events.push_back((self.revision, changes));
        while self.events.len() > MAX_REVISION_EVENTS {
            self.events.pop_front();
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
) -> Vec<FileChange> {
    let paths: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    paths
        .into_iter()
        .filter_map(|path| {
            let old = before.get(path);
            let new = after.get(path);
            if old == new {
                return None;
            }
            Some(FileChange {
                path: path.clone(),
                kind: match (old, new) {
                    (None, Some(_)) => FileChangeKind::Create,
                    (Some(_), None) => FileChangeKind::Delete,
                    (Some(_), Some(_)) => FileChangeKind::Modify,
                    (None, None) => return None,
                },
                before_digest: old.map(|state| state.digest.clone()),
                after_digest: new.map(|state| state.digest.clone()),
                before_len: old.map(|state| state.len),
                after_len: new.map(|state| state.len),
                before_mode: old.map(|state| state.mode),
                after_mode: new.map(|state| state.mode),
            })
        })
        .collect()
}

fn scan_workspace(
    root: &Path,
    excluded_roots: &[PathBuf],
    explicit_paths: &BTreeSet<String>,
) -> Result<BTreeMap<String, FileState>> {
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
        let relative = path
            .strip_prefix(root)
            .expect("workspace walker escaped root")
            .to_string_lossy();
        if metadata.is_file() && metadata.len() > MAX_AUTOMATIC_FILE_BYTES {
            continue;
        }
        if let Some(state) = read_state(path)? {
            states.insert(normalize(&relative), state);
        }
    }
    // A typed mutation supplies an exact path. Keep tracking it even below a
    // pruned build/dependency tree so a later full reconciliation cannot turn
    // the just-recorded create into a synthetic deletion.
    for relative in explicit_paths {
        if vcs_relative_path(relative) {
            continue;
        }
        let path = root.join(relative);
        if let Some(state) = read_state(&path)? {
            states.insert(relative.clone(), state);
        } else {
            states.remove(relative);
        }
    }
    Ok(states)
}

fn read_state(path: &Path) -> Result<Option<FileState>> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("reading metadata for {}", path.display()));
        }
    };
    let (bytes, len) = if metadata.file_type().is_symlink() {
        let target = std::fs::read_link(path)
            .with_context(|| format!("reading symlink {}", path.display()))?;
        let bytes = target.as_os_str().as_encoded_bytes().to_vec();
        let len = bytes.len() as u64;
        (bytes, len)
    } else if metadata.is_file() {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let len = bytes.len() as u64;
        (bytes, len)
    } else {
        return Ok(None);
    };
    let prefix = if metadata.file_type().is_symlink() {
        "symlink:sha256:"
    } else {
        "sha256:"
    };
    Ok(Some(FileState {
        digest: format!("{prefix}{:x}", Sha256::digest(bytes)),
        len,
        mode: file_mode(&metadata),
    }))
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
    let name = path.file_name().and_then(|name| name.to_str());
    if name.is_some_and(|name| {
        name.starts_with(".venv-") || name.starts_with("venv-") || name.starts_with("node_modules-")
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
                | "hi-test-scratch"
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
        let after = read_state(&generated).unwrap().unwrap();
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
}
