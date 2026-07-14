use std::collections::{BTreeMap, HashSet};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{FileChange, FileChangeKind};

static TRANSACTION_ID: AtomicU64 = AtomicU64::new(1);

/// Whether a planned write may create or replace its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteDisposition {
    Upsert,
    CreateOnly,
    ExistingOnly,
}

/// One requested operation used to build a [`MutationPlan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlannedFileMutation {
    Write {
        path: PathBuf,
        content: Vec<u8>,
        disposition: WriteDisposition,
        /// Requested Unix permission bits. `None` preserves an existing mode
        /// and uses `0o644` for a newly-created file.
        mode: Option<u32>,
        /// When present, construction must observe exactly this source digest.
        /// Text editors and patch planners use it to close the read-to-plan
        /// race while computing a postimage.
        expected_digest: Option<String>,
    },
    Delete {
        path: PathBuf,
    },
}

/// A complete, no-follow filesystem postimage used by checkpoint restoration.
///
/// This is crate-private deliberately: normal tools operate on regular files,
/// while checkpoint backends need to restore symlinks and whole directory
/// subtrees without exposing a second mutation implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RestoreNode {
    File {
        bytes: Vec<u8>,
        mode: u32,
    },
    Symlink {
        target: PathBuf,
    },
    Directory {
        mode: u32,
        entries: BTreeMap<OsString, RestoreNode>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct RestoreMutation {
    pub path: PathBuf,
    pub postimage: Option<RestoreNode>,
}

impl PlannedFileMutation {
    pub fn write(path: impl Into<PathBuf>, content: impl Into<Vec<u8>>) -> Self {
        Self::Write {
            path: path.into(),
            content: content.into(),
            disposition: WriteDisposition::Upsert,
            mode: None,
            expected_digest: None,
        }
    }

    pub fn write_with_mode(
        path: impl Into<PathBuf>,
        content: impl Into<Vec<u8>>,
        mode: u32,
    ) -> Self {
        Self::Write {
            path: path.into(),
            content: content.into(),
            disposition: WriteDisposition::Upsert,
            mode: Some(mode),
            expected_digest: None,
        }
    }

    pub fn add(path: impl Into<PathBuf>, content: impl Into<Vec<u8>>) -> Self {
        Self::Write {
            path: path.into(),
            content: content.into(),
            disposition: WriteDisposition::CreateOnly,
            mode: None,
            expected_digest: None,
        }
    }

    pub fn add_with_mode(path: impl Into<PathBuf>, content: impl Into<Vec<u8>>, mode: u32) -> Self {
        Self::Write {
            path: path.into(),
            content: content.into(),
            disposition: WriteDisposition::CreateOnly,
            mode: Some(mode),
            expected_digest: None,
        }
    }

    pub fn update(path: impl Into<PathBuf>, content: impl Into<Vec<u8>>) -> Self {
        Self::Write {
            path: path.into(),
            content: content.into(),
            disposition: WriteDisposition::ExistingOnly,
            mode: None,
            expected_digest: None,
        }
    }

    pub(crate) fn update_from_preimage(
        path: impl Into<PathBuf>,
        preimage: &[u8],
        content: impl Into<Vec<u8>>,
    ) -> Self {
        Self::Write {
            path: path.into(),
            content: content.into(),
            disposition: WriteDisposition::ExistingOnly,
            mode: None,
            expected_digest: Some(digest_bytes(preimage)),
        }
    }

    pub fn update_with_mode(
        path: impl Into<PathBuf>,
        content: impl Into<Vec<u8>>,
        mode: u32,
    ) -> Self {
        Self::Write {
            path: path.into(),
            content: content.into(),
            disposition: WriteDisposition::ExistingOnly,
            mode: Some(mode),
            expected_digest: None,
        }
    }

    pub fn delete(path: impl Into<PathBuf>) -> Self {
        Self::Delete { path: path.into() }
    }
}

#[derive(Clone, Debug)]
struct PlannedChange {
    requested_path: PathBuf,
    target: PathBuf,
    before: Option<RestoreNode>,
    after: Option<RestoreNode>,
}

/// A fully-materialized, digest-sealed multi-file mutation.
///
/// Construction reads every preimage and computes every postimage before any
/// target is touched. Commit stages siblings, revalidates every preimage, and
/// uses backup renames so any error restores the whole batch.
#[derive(Debug)]
pub struct MutationPlan {
    root: PathBuf,
    journal_dir: PathBuf,
    changes: Vec<PlannedChange>,
}

impl MutationPlan {
    pub fn new(root: impl AsRef<Path>, mutations: Vec<PlannedFileMutation>) -> Result<Self> {
        Self::new_with_state(root, crate::checkpoint::default_state_root(), mutations)
    }

    pub fn new_with_state(
        root: impl AsRef<Path>,
        state_root: impl AsRef<Path>,
        mutations: Vec<PlannedFileMutation>,
    ) -> Result<Self> {
        ensure!(!mutations.is_empty(), "transaction has no file operations");
        let root = canonical_root(root.as_ref())?;
        let journal_dir = transaction_journal_dir(&root, state_root.as_ref());
        recover_pending(&journal_dir)?;

        let mut seen = HashSet::new();
        let mut changes = Vec::with_capacity(mutations.len());
        for mutation in mutations {
            let (requested_path, after, disposition, requested_mode, expected_digest) =
                match mutation {
                    PlannedFileMutation::Write {
                        path,
                        content,
                        disposition,
                        mode,
                        expected_digest,
                    } => (
                        path,
                        Some(content),
                        Some(disposition),
                        mode,
                        expected_digest,
                    ),
                    PlannedFileMutation::Delete { path } => (path, None, None, None, None),
                };
            if let Some(mode) = requested_mode {
                ensure!(mode <= 0o7777, "invalid file mode {mode:#o}");
            }
            let target = resolve_workspace_target(&root, &requested_path)?;
            ensure!(
                seen.insert(target.clone()),
                "duplicate operation for {}",
                requested_path.display()
            );
            let before = read_node(&target)?;
            ensure!(
                before.as_ref().is_none_or(RestoreNode::is_file),
                "target is not a regular file: {}",
                requested_path.display()
            );
            if let Some(expected_digest) = expected_digest {
                let observed_digest = before.as_ref().and_then(|node| match node {
                    RestoreNode::File { bytes, .. } => Some(digest_bytes(bytes)),
                    RestoreNode::Symlink { .. } | RestoreNode::Directory { .. } => None,
                });
                ensure!(
                    observed_digest.as_deref() == Some(expected_digest.as_str()),
                    "source changed while preparing {}",
                    requested_path.display()
                );
            }
            match disposition {
                Some(WriteDisposition::CreateOnly) if before.is_some() => {
                    bail!("cannot add existing path {}", requested_path.display())
                }
                Some(WriteDisposition::ExistingOnly) if before.is_none() => {
                    bail!("cannot update missing path {}", requested_path.display())
                }
                None if before.is_none() => {
                    bail!("cannot delete missing path {}", requested_path.display())
                }
                _ => {}
            }
            let after = after.map(|bytes| RestoreNode::File {
                mode: requested_mode.unwrap_or_else(|| {
                    before
                        .as_ref()
                        .and_then(RestoreNode::mode)
                        .unwrap_or(default_file_mode())
                }),
                bytes,
            });
            changes.push(PlannedChange {
                requested_path,
                target,
                before,
                after,
            });
        }
        Ok(Self {
            root,
            journal_dir,
            changes,
        })
    }

    /// Prepare a complete checkpoint restore through the normal recovery
    /// journal. Every current node is captured now and re-read immediately
    /// before commit; no target is touched during preparation.
    pub(crate) fn new_restore_with_state(
        root: impl AsRef<Path>,
        state_root: impl AsRef<Path>,
        mutations: Vec<RestoreMutation>,
    ) -> Result<Self> {
        ensure!(!mutations.is_empty(), "restore has no file operations");
        let root = canonical_root(root.as_ref())?;
        let journal_dir = transaction_journal_dir(&root, state_root.as_ref());
        recover_pending(&journal_dir)?;
        let mut seen = HashSet::new();
        let mut changes = Vec::with_capacity(mutations.len());
        for mutation in mutations {
            let target = resolve_workspace_target(&root, &mutation.path)?;
            ensure!(target != root, "refusing to replace the workspace root");
            ensure!(
                seen.insert(target.clone()),
                "duplicate restore operation for {}",
                mutation.path.display()
            );
            let before = read_node(&target)?;
            changes.push(PlannedChange {
                requested_path: mutation.path,
                target,
                before,
                after: mutation.postimage,
            });
        }
        // Restore frontiers must not overlap. Otherwise renaming an ancestor
        // would invalidate a descendant precondition midway through commit.
        for (index, change) in changes.iter().enumerate() {
            ensure!(
                !changes.iter().enumerate().any(|(other_index, other)| {
                    index != other_index && other.target.starts_with(&change.target)
                }),
                "overlapping restore operation for {}",
                change.requested_path.display()
            );
        }
        Ok(Self {
            root,
            journal_dir,
            changes,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Exact metadata that will be returned after a successful commit.
    pub fn file_changes(&self) -> Vec<FileChange> {
        self.changes
            .iter()
            .filter_map(|change| change.as_file_change(&self.root))
            .collect()
    }

    pub fn is_noop(&self) -> bool {
        self.file_changes().is_empty()
    }

    /// Render the exact in-memory plan without touching target files.
    pub fn preview(&self) -> String {
        let mut out = String::new();
        for change in &self.changes {
            if change.as_file_change(&self.root).is_none() {
                continue;
            }
            if !out.is_empty() {
                out.push('\n');
            }
            let path = report_path(&self.root, &change.target);
            match (&change.before, &change.after) {
                (
                    Some(RestoreNode::File { bytes: before, .. }),
                    Some(RestoreNode::File { bytes: after, .. }),
                ) => match (std::str::from_utf8(before), std::str::from_utf8(after)) {
                    (Ok(before), Ok(after)) => {
                        out.push_str(&format!("--- {path}\n+++ {path}\n"));
                        out.push_str(&crate::edit::diff(before, after));
                    }
                    _ => out.push_str(&format!("~ {path} (binary content changed)")),
                },
                (None, Some(RestoreNode::File { bytes: after, .. })) => {
                    match std::str::from_utf8(after) {
                        Ok(after) => {
                            out.push_str(&format!("--- /dev/null\n+++ {path}\n"));
                            out.push_str(&crate::edit::diff("", after));
                        }
                        Err(_) => out.push_str(&format!("+ {path} (binary file)")),
                    }
                }
                (Some(_), None) => out.push_str(&format!("- {path}")),
                (_, Some(_)) => out.push_str(&format!("~ {path} (filesystem node changed)")),
                (None, None) => {}
            }
        }
        if out.is_empty() {
            "(no changes)".to_string()
        } else {
            out
        }
    }

    pub fn commit(self) -> Result<Vec<FileChange>> {
        self.commit_inner(None)
    }

    fn commit_inner(self, fail_before_index: Option<usize>) -> Result<Vec<FileChange>> {
        // Detect a user/editor race before staging or renaming anything.
        for change in &self.changes {
            verify_preimage(change)?;
        }
        let actionable: Vec<&PlannedChange> = self
            .changes
            .iter()
            .filter(|change| change.as_file_change(&self.root).is_some())
            .collect();
        if actionable.is_empty() {
            return Ok(Vec::new());
        }

        let id = format!(
            "{}-{}",
            std::process::id(),
            TRANSACTION_ID.fetch_add(1, Ordering::Relaxed)
        );
        let mut entries = Vec::with_capacity(actionable.len());
        let mut created_dirs = Vec::new();
        let prepare_result = (|| -> Result<()> {
            for (index, change) in actionable.iter().enumerate() {
                let parent = change.target.parent().ok_or_else(|| {
                    anyhow::anyhow!("target has no parent: {}", change.target.display())
                })?;
                ensure_parent_dirs(&self.root, parent, &mut created_dirs)?;
                let stage = if let Some(after) = &change.after {
                    let stage = sibling_temp(&change.target, &id, index, "stage")?;
                    write_stage_node(&stage, after)?;
                    Some(stage)
                } else {
                    None
                };
                let backup = if change.before.is_some() {
                    Some(sibling_temp(&change.target, &id, index, "backup")?)
                } else {
                    None
                };
                entries.push(JournalEntry {
                    target: change.target.clone(),
                    stage,
                    backup,
                    before_exists: change.before.is_some(),
                });
            }
            Ok(())
        })();
        if let Err(error) = prepare_result {
            cleanup_prepared_entries(&entries);
            cleanup_empty_dirs(&created_dirs);
            return Err(error.context("transaction staging failed; no target changes were made"));
        }

        let journal_dir = self.journal_dir.clone();
        let journal_dir_existed = journal_dir.exists();
        if let Err(error) = fs::create_dir_all(&journal_dir) {
            cleanup_prepared_entries(&entries);
            cleanup_empty_dirs(&created_dirs);
            return Err(error).with_context(|| {
                format!(
                    "creating transaction journal directory {}",
                    journal_dir.display()
                )
            });
        }
        let journal_path = journal_dir.join(format!("{id}.json"));

        let mut journal = Journal {
            owner_pid: std::process::id(),
            phase: JournalPhase::Prepared,
            active_index: None,
            entries,
        };
        if let Err(error) = write_journal(&journal_path, &journal) {
            cleanup_prepared_entries(&journal.entries);
            cleanup_empty_dirs(&created_dirs);
            if !journal_dir_existed {
                let _ = fs::remove_dir(&journal_dir);
                let _ = fs::remove_dir(journal_dir.parent().unwrap_or(&journal_dir));
            }
            return Err(
                error.context("transaction journal setup failed; no target changes were made")
            );
        }
        journal.phase = JournalPhase::Committing;
        if let Err(error) = write_journal(&journal_path, &journal) {
            cleanup_prepared_entries(&journal.entries);
            let _ = fs::remove_file(&journal_path);
            cleanup_empty_dirs(&created_dirs);
            return Err(
                error.context("transaction journal setup failed; no target changes were made")
            );
        }

        let commit_result = (|| -> Result<()> {
            // Revalidate once more immediately before the first rename; staging
            // may have taken long enough for an external edit to arrive.
            for change in &actionable {
                verify_preimage(change)?;
            }
            for (index, entry) in journal.entries.iter().enumerate() {
                if fail_before_index == Some(index) {
                    bail!("injected transaction failure before operation {index}");
                }
                journal.active_index = Some(index);
                write_journal(&journal_path, &journal)?;
                if let Some(backup) = &entry.backup {
                    fs::rename(&entry.target, backup).with_context(|| {
                        format!(
                            "moving {} to transaction backup {}",
                            entry.target.display(),
                            backup.display()
                        )
                    })?;
                }
                if let Some(stage) = &entry.stage {
                    fs::rename(stage, &entry.target).with_context(|| {
                        format!("atomically replacing {}", entry.target.display())
                    })?;
                }
                sync_parent(&entry.target)?;
            }
            Ok(())
        })();

        if let Err(error) = commit_result {
            let rollback = rollback(&journal, &journal_path, &created_dirs);
            return match rollback {
                Ok(()) => Err(error.context("transaction rolled back; no target changes remain")),
                Err(rollback_error) => Err(error.context(format!(
                    "transaction failed and rollback also failed: {rollback_error:#}; recovery journal: {}",
                    journal_path.display()
                ))),
            };
        }

        journal.phase = JournalPhase::Committed;
        journal.active_index = None;
        if let Err(error) = write_journal(&journal_path, &journal) {
            // The durable state still says `committing`; restore immediately so
            // a caller never observes an error with committed target changes.
            let rollback = rollback(&journal, &journal_path, &created_dirs);
            return match rollback {
                Ok(()) => Err(error.context("could not seal transaction; changes were rolled back")),
                Err(rollback_error) => Err(error.context(format!(
                    "could not seal transaction and rollback failed: {rollback_error:#}; recovery journal: {}",
                    journal_path.display()
                ))),
            };
        }
        for entry in &journal.entries {
            if let Some(backup) = &entry.backup {
                let _ = remove_node_if_exists(backup);
            }
            if let Some(stage) = &entry.stage {
                let _ = remove_node_if_exists(stage);
            }
        }
        let _ = fs::remove_file(&journal_path);
        if !journal_dir_existed {
            let _ = fs::remove_dir(&journal_dir);
            if let Some(parent) = journal_dir.parent() {
                let _ = fs::remove_dir(parent);
            }
        }
        cleanup_empty_dirs(&created_dirs);
        Ok(self.file_changes())
    }
}

/// Recover interrupted file transactions for one explicit workspace.
///
/// This is called when a workspace runtime starts, before its ledger takes an
/// initial snapshot, so a previous process's half-commit cannot become the new
/// baseline merely because no mutation tool has been prepared yet.
pub fn recover_workspace_transactions(
    root: impl AsRef<Path>,
    state_root: impl AsRef<Path>,
) -> Result<()> {
    let root = canonical_root(root.as_ref())?;
    recover_pending(&transaction_journal_dir(&root, state_root.as_ref()))
}

fn transaction_journal_dir(root: &Path, state_root: &Path) -> PathBuf {
    let workspace_key = digest_bytes(root.to_string_lossy().as_bytes())
        .trim_start_matches("sha256:")
        .to_string();
    state_root.join("transactions").join(workspace_key)
}

impl PlannedChange {
    fn as_file_change(&self, root: &Path) -> Option<FileChange> {
        if self.before == self.after {
            return None;
        }
        let kind = match (&self.before, &self.after) {
            (None, Some(_)) => FileChangeKind::Create,
            (Some(_), None) => FileChangeKind::Delete,
            (Some(_), Some(_)) => FileChangeKind::Modify,
            (None, None) => return None,
        };
        Some(FileChange {
            path: report_path(root, &self.target),
            kind,
            before_digest: self.before.as_ref().map(RestoreNode::digest),
            after_digest: self.after.as_ref().map(RestoreNode::digest),
            before_len: self.before.as_ref().map(RestoreNode::len),
            after_len: self.after.as_ref().map(RestoreNode::len),
            before_mode: self.before.as_ref().and_then(RestoreNode::mode),
            after_mode: self.after.as_ref().and_then(RestoreNode::mode),
        })
    }
}

impl RestoreNode {
    fn is_file(&self) -> bool {
        matches!(self, Self::File { .. })
    }

    fn mode(&self) -> Option<u32> {
        match self {
            Self::File { mode, .. } | Self::Directory { mode, .. } => Some(*mode),
            Self::Symlink { .. } => None,
        }
    }

    fn len(&self) -> u64 {
        match self {
            Self::File { bytes, .. } => bytes.len() as u64,
            Self::Symlink { target } => os_path_bytes(target).len() as u64,
            Self::Directory { entries, .. } => entries.values().map(Self::len).sum(),
        }
    }

    fn digest(&self) -> String {
        let mut hasher = Sha256::new();
        self.update_digest(&mut hasher);
        format!("sha256:{:x}", hasher.finalize())
    }

    fn update_digest(&self, hasher: &mut Sha256) {
        match self {
            Self::File { bytes, mode } => {
                hasher.update(b"file\0");
                hasher.update(mode.to_le_bytes());
                hasher.update((bytes.len() as u64).to_le_bytes());
                hasher.update(bytes);
            }
            Self::Symlink { target } => {
                hasher.update(b"symlink\0");
                let target = os_path_bytes(target);
                hasher.update((target.len() as u64).to_le_bytes());
                hasher.update(target);
            }
            Self::Directory { mode, entries } => {
                hasher.update(b"directory\0");
                hasher.update(mode.to_le_bytes());
                for (name, child) in entries {
                    let name = os_string_bytes(name);
                    hasher.update((name.len() as u64).to_le_bytes());
                    hasher.update(name);
                    child.update_digest(hasher);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum JournalPhase {
    Prepared,
    Committing,
    Committed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct JournalEntry {
    target: PathBuf,
    stage: Option<PathBuf>,
    backup: Option<PathBuf>,
    before_exists: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Journal {
    owner_pid: u32,
    phase: JournalPhase,
    active_index: Option<usize>,
    entries: Vec<JournalEntry>,
}

fn canonical_root(root: &Path) -> Result<PathBuf> {
    let metadata =
        fs::metadata(root).with_context(|| format!("reading workspace root {}", root.display()))?;
    ensure!(
        metadata.is_dir(),
        "workspace root is not a directory: {}",
        root.display()
    );
    root.canonicalize()
        .with_context(|| format!("canonicalizing workspace root {}", root.display()))
}

pub(crate) fn resolve_workspace_target(root: &Path, requested: &Path) -> Result<PathBuf> {
    let joined = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        root.join(requested)
    };
    let lexical = lexical_normalize(&joined);
    let resolved = canonicalize_nearest(&lexical)?;
    ensure!(
        resolved.starts_with(root),
        "path '{}' is outside workspace {}",
        requested.display(),
        root.display()
    );
    Ok(lexical)
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::RootDir => out.push(Path::new("/")),
            Component::Normal(name) => out.push(name),
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
        }
    }
    out
}

fn canonicalize_nearest(path: &Path) -> Result<PathBuf> {
    let mut ancestor = path;
    let mut tail = Vec::new();
    loop {
        match ancestor.canonicalize() {
            Ok(mut canonical) => {
                for part in tail.iter().rev() {
                    canonical.push(part);
                }
                return Ok(canonical);
            }
            Err(_) => {
                if let Some(name) = ancestor.file_name() {
                    tail.push(name.to_os_string());
                }
                ancestor = ancestor
                    .parent()
                    .ok_or_else(|| anyhow::anyhow!("cannot resolve path {}", path.display()))?;
            }
        }
    }
}

fn read_node(path: &Path) -> Result<Option<RestoreNode>> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    let file_type = metadata.file_type();
    let node = if file_type.is_symlink() {
        RestoreNode::Symlink {
            target: fs::read_link(path)
                .with_context(|| format!("reading symlink {}", path.display()))?,
        }
    } else if file_type.is_file() {
        RestoreNode::File {
            bytes: fs::read(path).with_context(|| format!("reading {}", path.display()))?,
            mode: file_mode(&metadata),
        }
    } else if file_type.is_dir() {
        let mut entries = BTreeMap::new();
        for entry in fs::read_dir(path).with_context(|| format!("reading {}", path.display()))? {
            let entry = entry.with_context(|| format!("walking {}", path.display()))?;
            let child = read_node(&entry.path())?.with_context(|| {
                format!(
                    "filesystem entry disappeared while reading {}",
                    entry.path().display()
                )
            })?;
            entries.insert(entry.file_name(), child);
        }
        RestoreNode::Directory {
            mode: file_mode(&metadata),
            entries,
        }
    } else {
        bail!("unsupported filesystem entry {}", path.display());
    };
    Ok(Some(node))
}

fn verify_preimage(change: &PlannedChange) -> Result<()> {
    let current = read_node(&change.target)?;
    ensure!(
        current == change.before,
        "transaction precondition failed for {}: file changed after preview",
        change.requested_path.display()
    );
    Ok(())
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[cfg(unix)]
fn file_mode(metadata: &fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o7777
}

#[cfg(not(unix))]
fn file_mode(metadata: &fs::Metadata) -> u32 {
    if metadata.permissions().readonly() {
        0o444
    } else {
        0o666
    }
}

fn default_file_mode() -> u32 {
    0o644
}

fn write_stage(path: &Path, bytes: &[u8], mode: u32) -> Result<()> {
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("creating transaction stage {}", path.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("writing transaction stage {}", path.display()))?;
        file.sync_all()
            .with_context(|| format!("syncing transaction stage {}", path.display()))?;
        set_mode(path, mode)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(path);
    }
    result
}

fn write_stage_node(path: &Path, node: &RestoreNode) -> Result<()> {
    let result = match node {
        RestoreNode::File { bytes, mode } => write_stage(path, bytes, *mode),
        RestoreNode::Symlink { target } => create_restore_symlink(target, path)
            .with_context(|| format!("creating transaction symlink stage {}", path.display())),
        RestoreNode::Directory { mode, entries } => (|| -> Result<()> {
            fs::create_dir(path).with_context(|| {
                format!("creating transaction directory stage {}", path.display())
            })?;
            for (name, child) in entries {
                write_stage_node(&path.join(name), child)?;
            }
            set_mode(path, *mode)?;
            sync_parent(path)?;
            Ok(())
        })(),
    };
    if result.is_err() {
        let _ = remove_node_if_exists(path);
    }
    result
}

#[cfg(unix)]
fn create_restore_symlink(target: &Path, path: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, path)
}

#[cfg(not(unix))]
fn create_restore_symlink(_target: &Path, path: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!("restoring symlinks is unsupported: {}", path.display()),
    ))
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
        .with_context(|| format!("setting mode on {}", path.display()))
}

#[cfg(not(unix))]
fn set_mode(path: &Path, mode: u32) -> Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(mode & 0o222 == 0);
    fs::set_permissions(path, permissions)
        .with_context(|| format!("setting mode on {}", path.display()))
}

fn sibling_temp(target: &Path, id: &str, index: usize, kind: &str) -> Result<PathBuf> {
    let parent = target
        .parent()
        .ok_or_else(|| anyhow::anyhow!("target has no parent: {}", target.display()))?;
    let name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("file");
    Ok(parent.join(format!(".{name}.hi-{kind}-{id}-{index}")))
}

fn ensure_parent_dirs(root: &Path, parent: &Path, created: &mut Vec<PathBuf>) -> Result<()> {
    ensure!(parent.starts_with(root), "parent escaped workspace");
    let mut missing = Vec::new();
    let mut current = parent;
    while !current.exists() {
        missing.push(current.to_path_buf());
        current = current
            .parent()
            .ok_or_else(|| anyhow::anyhow!("no existing parent for {}", parent.display()))?;
    }
    let metadata = fs::symlink_metadata(current)?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "parent path traverses symlink {}",
        current.display()
    );
    ensure!(
        metadata.is_dir(),
        "parent is not a directory: {}",
        current.display()
    );
    for directory in missing.into_iter().rev() {
        fs::create_dir(&directory)
            .with_context(|| format!("creating directory {}", directory.display()))?;
        created.push(directory);
    }
    Ok(())
}

fn write_journal(path: &Path, journal: &Journal) -> Result<()> {
    let temp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec(journal).context("serializing transaction journal")?;
    let mut file = File::create(&temp)
        .with_context(|| format!("creating transaction journal {}", temp.display()))?;
    file.write_all(&bytes)
        .with_context(|| format!("writing transaction journal {}", temp.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing transaction journal {}", temp.display()))?;
    fs::rename(&temp, path)
        .with_context(|| format!("installing transaction journal {}", path.display()))?;
    sync_parent(path)?;
    Ok(())
}

fn rollback(journal: &Journal, journal_path: &Path, created_dirs: &[PathBuf]) -> Result<()> {
    let mut errors = Vec::new();
    for entry in journal.entries.iter().rev() {
        if let Some(backup) = &entry.backup {
            if path_exists_no_follow(backup) {
                if path_exists_no_follow(&entry.target)
                    && remove_node_if_exists(&entry.target).is_err()
                {
                    errors.push(format!("removing {}", entry.target.display()));
                }
                if let Err(error) = fs::rename(backup, &entry.target) {
                    errors.push(format!("restoring {}: {error}", entry.target.display()));
                }
            }
        } else if !entry.before_exists
            && entry.stage.as_ref().is_some_and(|stage| !stage.exists())
            && path_exists_no_follow(&entry.target)
            && remove_node_if_exists(&entry.target).is_err()
        {
            errors.push(format!("removing created {}", entry.target.display()));
        }
    }
    if errors.is_empty() {
        for entry in &journal.entries {
            if let Some(stage) = &entry.stage {
                let _ = remove_node_if_exists(stage);
            }
        }
        fs::remove_file(journal_path).with_context(|| {
            format!(
                "removing completed recovery journal {}",
                journal_path.display()
            )
        })?;
        cleanup_empty_dirs(created_dirs);
        Ok(())
    } else {
        // Keep the journal and every still-present stage/backup. A later
        // WorkspaceRuntime startup can retry restoration; deleting the journal
        // here would strand the only durable map from backups to targets.
        bail!(errors.join("; "))
    }
}

fn recover_pending(directory: &Path) -> Result<()> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| format!("reading {}", directory.display()));
        }
    };
    for entry in entries {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let journal: Journal = serde_json::from_slice(
            &fs::read(&path)
                .with_context(|| format!("reading recovery journal {}", path.display()))?,
        )
        .with_context(|| format!("parsing recovery journal {}", path.display()))?;
        if process_is_alive(journal.owner_pid) {
            continue;
        }
        match journal.phase {
            JournalPhase::Prepared | JournalPhase::Committed => {
                for item in &journal.entries {
                    if let Some(stage) = &item.stage {
                        let _ = remove_node_if_exists(stage);
                    }
                    if journal.phase == JournalPhase::Committed
                        && let Some(backup) = &item.backup
                    {
                        let _ = remove_node_if_exists(backup);
                    }
                }
                let _ = fs::remove_file(&path);
            }
            JournalPhase::Committing => rollback(&journal, &path, &[])?,
        }
    }
    Ok(())
}

#[cfg(unix)]
fn process_is_alive(pid: u32) -> bool {
    // SAFETY: signal 0 performs an existence/permission check only.
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(not(unix))]
fn process_is_alive(pid: u32) -> bool {
    pid == std::process::id()
}

fn cleanup_empty_dirs(directories: &[PathBuf]) {
    for directory in directories.iter().rev() {
        let _ = fs::remove_dir(directory);
    }
}

fn cleanup_prepared_entries(entries: &[JournalEntry]) {
    for entry in entries {
        if let Some(stage) = &entry.stage {
            let _ = remove_node_if_exists(stage);
        }
        if let Some(backup) = &entry.backup {
            let _ = remove_node_if_exists(backup);
        }
    }
}

fn path_exists_no_follow(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok()
}

fn remove_node_if_exists(path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        fs::remove_dir_all(path)?;
    } else {
        fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(unix)]
fn os_string_bytes(value: &std::ffi::OsStr) -> &[u8] {
    use std::os::unix::ffi::OsStrExt;
    value.as_bytes()
}

#[cfg(not(unix))]
fn os_string_bytes(value: &std::ffi::OsStr) -> &[u8] {
    // Restore node names originate from paths accepted by this process. On
    // Windows the digest is used only as typed change metadata, not as an
    // identity boundary.
    value.to_str().unwrap_or("").as_bytes()
}

#[cfg(unix)]
fn os_path_bytes(value: &Path) -> &[u8] {
    use std::os::unix::ffi::OsStrExt;
    value.as_os_str().as_bytes()
}

#[cfg(not(unix))]
fn os_path_bytes(value: &Path) -> &[u8] {
    value.to_str().unwrap_or("").as_bytes()
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    File::open(parent)
        .with_context(|| format!("opening directory {} for sync", parent.display()))?
        .sync_all()
        .with_context(|| format!("syncing directory {}", parent.display()))
}

#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> Result<()> {
    // Windows does not generally permit opening a directory with File::open;
    // rename durability follows the volume's replacement guarantees.
    Ok(())
}

fn report_path(root: &Path, target: &Path) -> String {
    target
        .strip_prefix(root)
        .unwrap_or(target)
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hi-transaction-{label}-{}-{}",
            std::process::id(),
            TRANSACTION_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn multi_file_commit_is_typed_and_preserves_mode() {
        let root = temp_root("commit");
        fs::write(root.join("a.txt"), b"before\r\n").unwrap();
        fs::write(root.join("delete.txt"), b"gone\n").unwrap();
        #[cfg(unix)]
        set_mode(&root.join("a.txt"), 0o755).unwrap();
        let plan = MutationPlan::new(
            &root,
            vec![
                PlannedFileMutation::update("a.txt", b"after\r\n".to_vec()),
                PlannedFileMutation::add("nested/new.txt", b"new".to_vec()),
                PlannedFileMutation::delete("delete.txt"),
            ],
        )
        .unwrap();
        let changes = plan.commit().unwrap();
        assert_eq!(changes.len(), 3);
        assert_eq!(fs::read(root.join("a.txt")).unwrap(), b"after\r\n");
        assert_eq!(fs::read(root.join("nested/new.txt")).unwrap(), b"new");
        assert!(!root.join("delete.txt").exists());
        #[cfg(unix)]
        assert_eq!(file_mode(&fs::metadata(root.join("a.txt")).unwrap()), 0o755);
        assert!(
            changes
                .iter()
                .all(|change| { change.before_digest.is_some() || change.after_digest.is_some() })
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn requested_modes_are_applied_transactionally() {
        let root = temp_root("requested-mode");
        fs::write(root.join("existing"), b"old").unwrap();
        let plan = MutationPlan::new(
            &root,
            vec![
                PlannedFileMutation::update_with_mode("existing", b"new".to_vec(), 0o600),
                PlannedFileMutation::add_with_mode("script", b"#!/bin/sh\n".to_vec(), 0o755),
            ],
        )
        .unwrap();

        let changes = plan.commit().unwrap();

        assert_eq!(changes[0].after_mode, Some(0o600));
        assert_eq!(changes[1].after_mode, Some(0o755));
        #[cfg(unix)]
        {
            assert_eq!(
                file_mode(&fs::metadata(root.join("existing")).unwrap()),
                0o600
            );
            assert_eq!(
                file_mode(&fs::metadata(root.join("script")).unwrap()),
                0o755
            );
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn invalid_requested_mode_is_rejected_before_writes() {
        let root = temp_root("invalid-mode");
        let result = MutationPlan::new(
            &root,
            vec![PlannedFileMutation::write_with_mode(
                "file",
                b"content".to_vec(),
                0o10000,
            )],
        );
        assert!(result.is_err());
        assert!(!root.join("file").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn mode_only_change_is_reported_and_applied() {
        let root = temp_root("mode-only");
        fs::write(root.join("script"), b"#!/bin/sh\n").unwrap();
        set_mode(&root.join("script"), 0o644).unwrap();
        let plan = MutationPlan::new(
            &root,
            vec![PlannedFileMutation::update_with_mode(
                "script",
                b"#!/bin/sh\n".to_vec(),
                0o755,
            )],
        )
        .unwrap();

        let changes = plan.commit().unwrap();

        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].kind, FileChangeKind::Modify);
        assert_eq!(changes[0].before_mode, Some(0o644));
        assert_eq!(changes[0].after_mode, Some(0o755));
        assert_eq!(
            file_mode(&fs::metadata(root.join("script")).unwrap()),
            0o755
        );
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn identical_write_is_a_true_noop() {
        use std::os::unix::fs::MetadataExt;

        let root = temp_root("noop");
        fs::write(root.join("file"), b"same").unwrap();
        let inode = fs::metadata(root.join("file")).unwrap().ino();
        let plan = MutationPlan::new(
            &root,
            vec![PlannedFileMutation::write("file", b"same".to_vec())],
        )
        .unwrap();
        assert!(plan.is_noop());

        assert!(plan.commit().unwrap().is_empty());
        assert_eq!(fs::metadata(root.join("file")).unwrap().ino(), inode);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn stale_precondition_leaves_every_file_unchanged() {
        let root = temp_root("stale");
        fs::write(root.join("a"), "one").unwrap();
        fs::write(root.join("b"), "two").unwrap();
        let plan = MutationPlan::new(
            &root,
            vec![
                PlannedFileMutation::update("a", b"ONE".to_vec()),
                PlannedFileMutation::update("b", b"TWO".to_vec()),
            ],
        )
        .unwrap();
        fs::write(root.join("b"), "external").unwrap();
        assert!(plan.commit().is_err());
        assert_eq!(fs::read_to_string(root.join("a")).unwrap(), "one");
        assert_eq!(fs::read_to_string(root.join("b")).unwrap(), "external");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn failure_on_second_restore_operation_rolls_back_first_node() {
        let root = temp_root("restore-second-failure");
        let state = root.parent().unwrap().join(format!(
            "hi-transaction-restore-state-{}",
            TRANSACTION_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&state).unwrap();
        fs::write(root.join("a"), "after-a").unwrap();
        fs::write(root.join("b"), "after-b").unwrap();
        let plan = MutationPlan::new_restore_with_state(
            &root,
            &state,
            vec![
                RestoreMutation {
                    path: "a".into(),
                    postimage: Some(RestoreNode::File {
                        bytes: b"before-a".to_vec(),
                        mode: 0o644,
                    }),
                },
                RestoreMutation {
                    path: "b".into(),
                    postimage: Some(RestoreNode::File {
                        bytes: b"before-b".to_vec(),
                        mode: 0o644,
                    }),
                },
            ],
        )
        .unwrap();

        let error = plan.commit_inner(Some(1)).unwrap_err();

        assert!(error.to_string().contains("rolled back"));
        assert_eq!(fs::read_to_string(root.join("a")).unwrap(), "after-a");
        assert_eq!(fs::read_to_string(root.join("b")).unwrap(), "after-b");
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(state);
    }

    #[test]
    fn partial_staging_failure_removes_stages_and_created_directories() {
        let root = temp_root("stage-cleanup");
        fs::write(root.join("a"), "one").unwrap();
        let plan = MutationPlan::new(
            &root,
            vec![
                PlannedFileMutation::update("a", b"ONE".to_vec()),
                PlannedFileMutation::add("blocked/new", b"new".to_vec()),
            ],
        )
        .unwrap();
        // This arrives after planning and makes the second target's parent
        // impossible to create. The first file has already been staged.
        fs::write(root.join("blocked"), "external").unwrap();
        assert!(plan.commit().is_err());
        assert_eq!(fs::read_to_string(root.join("a")).unwrap(), "one");
        assert_eq!(
            fs::read_to_string(root.join("blocked")).unwrap(),
            "external"
        );
        let names: Vec<String> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            !names.iter().any(|name| name.contains("hi-stage")),
            "{names:?}"
        );
        assert!(
            !root.join(".hi").exists(),
            "staging failure must not create workspace metadata"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn duplicate_and_add_existing_are_rejected_before_writes() {
        let root = temp_root("reject");
        fs::write(root.join("a"), "one").unwrap();
        assert!(
            MutationPlan::new(&root, vec![PlannedFileMutation::add("a", b"new".to_vec())]).is_err()
        );
        assert!(
            MutationPlan::new(
                &root,
                vec![
                    PlannedFileMutation::write("a", b"x".to_vec()),
                    PlannedFileMutation::delete("a"),
                ]
            )
            .is_err()
        );
        assert_eq!(fs::read_to_string(root.join("a")).unwrap(), "one");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn path_cannot_escape_root() {
        let root = temp_root("containment");
        let outside = root.parent().unwrap().join("outside-hi-transaction");
        let result = MutationPlan::new(
            &root,
            vec![PlannedFileMutation::write(&outside, b"no".to_vec())],
        );
        assert!(result.is_err());
        assert!(!outside.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn workspace_startup_recovery_restores_an_interrupted_commit() {
        let root = temp_root("startup-recovery");
        let state = root.parent().unwrap().join(format!(
            "hi-transaction-recovery-state-{}",
            TRANSACTION_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&state).unwrap();
        let target = root.join("value.txt");
        let backup = root.join(".value.txt.hi-backup-crashed");
        fs::write(&target, "after").unwrap();
        fs::write(&backup, "before").unwrap();
        let journal_dir = transaction_journal_dir(&root.canonicalize().unwrap(), &state);
        fs::create_dir_all(&journal_dir).unwrap();
        let journal_path = journal_dir.join("crashed.json");
        let owner_pid = (1_000_000..=4_000_000)
            .find(|pid| !process_is_alive(*pid))
            .expect("a non-running test pid");
        write_journal(
            &journal_path,
            &Journal {
                owner_pid,
                phase: JournalPhase::Committing,
                active_index: Some(0),
                entries: vec![JournalEntry {
                    target: target.clone(),
                    stage: None,
                    backup: Some(backup.clone()),
                    before_exists: true,
                }],
            },
        )
        .unwrap();

        recover_workspace_transactions(&root, &state).unwrap();

        assert_eq!(fs::read_to_string(&target).unwrap(), "before");
        assert!(!backup.exists());
        assert!(!journal_path.exists());
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(state);
    }

    #[test]
    fn failed_rollback_retains_journal_backup_and_stage_for_retry() {
        let root = temp_root("retain-failed-rollback");
        let target = root.join("missing-parent/value.txt");
        let backup = root.join(".value.txt.hi-backup");
        let stage = root.join(".value.txt.hi-stage");
        let journal_path = root.join("transaction.json");
        // A missing target parent makes the backup rename fail,
        // deterministically exercising the recovery-retention path.
        fs::write(&backup, "before").unwrap();
        fs::write(&stage, "after").unwrap();
        fs::write(&journal_path, "durable journal").unwrap();
        let journal = Journal {
            owner_pid: std::process::id(),
            phase: JournalPhase::Committing,
            active_index: Some(0),
            entries: vec![JournalEntry {
                target,
                stage: Some(stage.clone()),
                backup: Some(backup.clone()),
                before_exists: true,
            }],
        };

        assert!(rollback(&journal, &journal_path, &[]).is_err());

        assert!(journal_path.exists(), "journal must remain retryable");
        assert!(backup.exists(), "preimage backup must not be discarded");
        assert!(
            stage.exists(),
            "postimage stage must remain with the journal"
        );
        let _ = fs::remove_dir_all(root);
    }
}
