//! Content-addressed workspace snapshots used when Git checkpoints are absent.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::ffi::{OsStr, OsString};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const ID_PREFIX: &str = "internal:v1:";
const MAX_MANIFESTS: usize = 50;
const MAX_WORKSPACE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_CHECKPOINT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_CHECKPOINT_ENTRIES: usize = 200_000;
const MAX_DIFF_FILE_BYTES: usize = 64 * 1024;
static TEMP_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EntryKind {
    Directory,
    File,
    Symlink,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct SnapshotEntry {
    /// Hex-encoded raw relative-path bytes (not lossy UTF-8).
    path: String,
    kind: EntryKind,
    mode: u32,
    len: u64,
    object: Option<String>,
    target: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    workspace: String,
    created_ms: u64,
    entries: Vec<SnapshotEntry>,
}

#[derive(Clone, Debug)]
struct Store {
    workspace_dir: PathBuf,
    objects_dir: PathBuf,
    manifests_dir: PathBuf,
    workspace_id: String,
}

pub(crate) fn is_internal_id(id: &str) -> bool {
    id.starts_with(ID_PREFIX)
}

pub(crate) fn create(root: &Path, state_root: &Path) -> Result<String> {
    let root = canonical_root(root)?;
    let store = Store::open(&root, state_root)?;
    recover_temps(&store)?;
    let manifest = capture(&root, state_root, &store)?;
    let bytes = serde_json::to_vec(&manifest).context("serializing internal snapshot")?;
    let digest = digest(&bytes);
    let path = store.manifests_dir.join(format!("{digest}.json"));
    atomic_write_if_missing(&path, &bytes)?;
    enforce_limits(&store)?;
    Ok(format!("{ID_PREFIX}{}:{digest}", store.workspace_id))
}

pub(crate) fn restore(root: &Path, state_root: &Path, id: &str) -> Result<usize> {
    let (plan, changed) = prepare_restore(root, state_root, id, None)?;
    if let Some(plan) = plan {
        plan.commit()?;
    }
    Ok(changed)
}

/// Materialize `id` into a new, empty directory without changing the source
/// workspace. This is used for attribution checks: a verifier can execute the
/// same command against immutable pre-turn contents while the user's working
/// tree remains untouched.
pub(crate) fn materialize(
    root: &Path,
    state_root: &Path,
    id: &str,
    destination: &Path,
) -> Result<()> {
    let root = canonical_root(root)?;
    let (store, manifest, _) = load(&root, state_root, id)?;
    ensure!(
        !destination.exists(),
        "isolated snapshot destination already exists: {}",
        destination.display()
    );
    fs::create_dir(destination)
        .with_context(|| format!("creating isolated snapshot {}", destination.display()))?;

    let mut directories = Vec::new();
    for entry in manifest.entries {
        let relative = decode_path(&entry.path)?;
        let path = destination.join(&relative);
        match entry.kind {
            EntryKind::Directory => {
                fs::create_dir_all(&path)
                    .with_context(|| format!("creating snapshot directory {}", path.display()))?;
                directories.push((path, entry.mode));
            }
            EntryKind::File => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("creating {}", parent.display()))?;
                }
                let object = entry
                    .object
                    .as_deref()
                    .context("snapshot file has no object")?;
                let bytes = read_object(&store, object)?;
                ensure!(
                    digest(&bytes) == object,
                    "snapshot object digest mismatch for {}",
                    path.display()
                );
                atomic_replace(&path, &bytes, entry.mode)?;
            }
            EntryKind::Symlink => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("creating {}", parent.display()))?;
                }
                let target = entry
                    .target
                    .as_deref()
                    .context("snapshot symlink has no target")?;
                create_symlink(&decode_os(target)?, &path)?;
            }
        }
    }

    // Apply directory modes only after all children exist. A captured 0555
    // directory must not prevent the materializer from populating descendants.
    directories.sort_by_key(|(path, _)| std::cmp::Reverse(path.components().count()));
    for (path, entry_mode) in directories {
        set_mode(&path, entry_mode)?;
    }
    Ok(())
}

pub(crate) fn restore_sealed(
    root: &Path,
    state_root: &Path,
    target: &str,
    expected_current: &str,
) -> Result<usize> {
    let (plan, changed) = prepare_restore(root, state_root, target, Some(expected_current))?;
    if let Some(plan) = plan {
        plan.commit()?;
    }
    Ok(changed)
}

/// Construct every restore postimage and the transaction journal plan before
/// touching the workspace. Sealed restores compare the complete captured file
/// universe twice: once before planning and again immediately before commit.
fn prepare_restore(
    root: &Path,
    state_root: &Path,
    target_id: &str,
    expected_current: Option<&str>,
) -> Result<(Option<crate::transaction::MutationPlan>, usize)> {
    use crate::transaction::{MutationPlan, RestoreMutation};

    let root = canonical_root(root)?;
    let (store, target_manifest, _) = load(&root, state_root, target_id)?;
    let target = decoded_entries(target_manifest.entries)?;
    let current_encoded = scan(&root, state_root, None)?;
    let current = decoded_entries(current_encoded.into_values().collect())?;
    let expected = if let Some(expected_id) = expected_current {
        let (_, manifest, _) = load(&root, state_root, expected_id)?;
        let expected = decoded_entries(manifest.entries)?;
        ensure!(
            current == expected,
            "undo conflict: workspace changed externally after the turn"
        );
        Some(expected)
    } else {
        None
    };

    let protected = contained_relative(&root, state_root);
    let frontier = restore_frontier(&current, &target, protected.as_deref())?;
    let changed = current
        .keys()
        .chain(target.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|path| current.get(*path) != target.get(*path))
        .count();
    if frontier.is_empty() {
        return Ok((None, changed));
    }
    let mut mutations = Vec::with_capacity(frontier.len());
    for path in frontier {
        let postimage = if target.contains_key(&path) {
            Some(restore_node(&store, &target, &path)?)
        } else {
            None
        };
        mutations.push(RestoreMutation { path, postimage });
    }
    let plan = MutationPlan::new_restore_with_state(&root, state_root, mutations)?;

    if let Some(expected) = expected {
        let observed = decoded_entries(scan(&root, state_root, None)?.into_values().collect())?;
        ensure!(
            observed == expected,
            "undo conflict: workspace changed externally while preparing restore"
        );
    }
    Ok((Some(plan), changed))
}

fn decoded_entries(entries: Vec<SnapshotEntry>) -> Result<BTreeMap<PathBuf, SnapshotEntry>> {
    let mut decoded = BTreeMap::new();
    for entry in entries {
        let path = decode_path(&entry.path)?;
        ensure!(
            decoded.insert(path, entry).is_none(),
            "snapshot contains duplicate path"
        );
    }
    Ok(decoded)
}

fn contained_relative(root: &Path, candidate: &Path) -> Option<PathBuf> {
    let candidate = candidate.canonicalize().ok()?;
    candidate
        .strip_prefix(root)
        .ok()
        .filter(|path| !path.as_os_str().is_empty())
        .map(Path::to_path_buf)
}

fn restore_frontier(
    current: &BTreeMap<PathBuf, SnapshotEntry>,
    target: &BTreeMap<PathBuf, SnapshotEntry>,
    protected: Option<&Path>,
) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = current.keys().chain(target.keys()).cloned().collect();
    paths.sort_by_key(|path| (path.components().count(), path.clone()));
    paths.dedup();
    let mut frontier: Vec<PathBuf> = Vec::new();
    for path in paths {
        if frontier
            .iter()
            .any(|ancestor| path != *ancestor && path.starts_with(ancestor))
        {
            continue;
        }
        let before = current.get(&path);
        let after = target.get(&path);
        if before == after {
            continue;
        }
        let protects_runtime = protected.is_some_and(|runtime| runtime.starts_with(&path));
        if protects_runtime {
            ensure!(
                before.is_none_or(|entry| entry.kind == EntryKind::Directory)
                    && after.is_none_or(|entry| entry.kind == EntryKind::Directory),
                "cannot restore {} without replacing the runtime state directory",
                path.display()
            );
            continue;
        }
        // Existing directories are containers. Recurse into their children so
        // unrelated runtime/external contents are never replaced wholesale.
        // A mode change requires replacing the complete bounded subtree; this
        // keeps directory metadata in the same rollback journal as its files.
        if before.is_some_and(|entry| entry.kind == EntryKind::Directory)
            && after.is_some_and(|entry| entry.kind == EntryKind::Directory)
        {
            if before.map(|entry| entry.mode) != after.map(|entry| entry.mode) {
                frontier.push(path);
            }
            continue;
        }
        frontier.push(path);
    }
    Ok(frontier)
}

fn restore_node(
    store: &Store,
    entries: &BTreeMap<PathBuf, SnapshotEntry>,
    path: &Path,
) -> Result<crate::transaction::RestoreNode> {
    use crate::transaction::RestoreNode;
    let entry = entries
        .get(path)
        .with_context(|| format!("snapshot is missing {}", path.display()))?;
    match entry.kind {
        EntryKind::File => {
            let object = entry
                .object
                .as_deref()
                .context("snapshot file has no object")?;
            let bytes = read_object(store, object)?;
            ensure!(
                digest(&bytes) == object && bytes.len() as u64 == entry.len,
                "snapshot object digest/length mismatch for {}",
                path.display()
            );
            Ok(RestoreNode::File {
                bytes,
                mode: entry.mode,
            })
        }
        EntryKind::Symlink => {
            let target = entry
                .target
                .as_deref()
                .context("snapshot symlink has no target")?;
            Ok(RestoreNode::Symlink {
                target: PathBuf::from(decode_os(target)?),
            })
        }
        EntryKind::Directory => {
            let mut children = BTreeMap::new();
            for child in entries
                .keys()
                .filter(|candidate| candidate.parent() == Some(path))
            {
                let name = child
                    .file_name()
                    .context("snapshot child has no filename")?
                    .to_os_string();
                children.insert(name, restore_node(store, entries, child)?);
            }
            Ok(RestoreNode::Directory {
                mode: entry.mode,
                entries: children,
            })
        }
    }
}

pub(crate) fn diff(root: &Path, state_root: &Path, id: &str) -> Result<Option<String>> {
    let root = canonical_root(root)?;
    let (store, before_manifest, _) = load(&root, state_root, id)?;
    let current_manifest = capture(&root, state_root, &store)?;
    let before: BTreeMap<String, SnapshotEntry> = before_manifest
        .entries
        .into_iter()
        .map(|entry| (entry.path.clone(), entry))
        .collect();
    let after: BTreeMap<String, SnapshotEntry> = current_manifest
        .entries
        .into_iter()
        .map(|entry| (entry.path.clone(), entry))
        .collect();
    let keys: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    let mut out = String::new();
    for key in keys {
        let old = before.get(key);
        let new = after.get(key);
        if old == new {
            continue;
        }
        let path = decode_path(key)?.to_string_lossy().replace('\\', "/");
        match (old, new) {
            (None, Some(entry)) => {
                out.push_str(&format!("+ {path}"));
                append_created_content(&mut out, &store, entry)?;
                out.push('\n');
            }
            (Some(_), None) => out.push_str(&format!("- {path}\n")),
            (Some(old), Some(new)) => {
                if old.kind == EntryKind::File && new.kind == EntryKind::File {
                    let old_bytes = read_object(&store, old.object.as_deref().unwrap_or(""))?;
                    let new_bytes = read_object(&store, new.object.as_deref().unwrap_or(""))?;
                    if old_bytes.len() <= MAX_DIFF_FILE_BYTES
                        && new_bytes.len() <= MAX_DIFF_FILE_BYTES
                        && let (Ok(old_text), Ok(new_text)) = (
                            std::str::from_utf8(&old_bytes),
                            std::str::from_utf8(&new_bytes),
                        )
                    {
                        out.push_str(&format!("--- {path}\n+++ {path}\n"));
                        out.push_str(&crate::edit::diff(old_text, new_text));
                        out.push('\n');
                        continue;
                    }
                }
                out.push_str(&format!("~ {path} ({:?} -> {:?})\n", old.kind, new.kind));
            }
            (None, None) => {}
        }
    }
    let out = out.trim_end().to_string();
    Ok((!out.is_empty()).then_some(out))
}

fn capture(root: &Path, state_root: &Path, store: &Store) -> Result<Manifest> {
    let entries = match scan(root, state_root, Some(store)) {
        Ok(entries) => entries,
        Err(error) => {
            // Files are content-addressed and may have been installed before a
            // later file crosses the per-checkpoint ceiling. With no manifest
            // those objects are unreachable; collect them on every failed
            // capture so failed checkpoints cannot bypass store quotas.
            return match gc_objects(store) {
                Ok(()) => Err(error),
                Err(cleanup) => Err(error.context(format!(
                    "failed snapshot capture and could not collect orphan objects: {cleanup:#}"
                ))),
            };
        }
    };
    Ok(Manifest {
        version: 1,
        workspace: store.workspace_id.clone(),
        // Snapshot ids must identify workspace content, not wall-clock time.
        // LRU recency lives in the manifest file's mtime instead.
        created_ms: 0,
        entries: entries.into_values().collect(),
    })
}

fn scan(
    root: &Path,
    state_root: &Path,
    store: Option<&Store>,
) -> Result<BTreeMap<String, SnapshotEntry>> {
    let state = state_root
        .canonicalize()
        .unwrap_or_else(|_| state_root.to_path_buf());
    let mut entries = BTreeMap::new();
    let mut total = 0u64;
    let mut count = 0usize;
    scan_dir(
        root,
        Path::new(""),
        &state,
        store,
        &mut total,
        &mut count,
        &mut entries,
    )?;
    Ok(entries)
}

fn scan_dir(
    root: &Path,
    relative: &Path,
    state_root: &Path,
    store: Option<&Store>,
    total: &mut u64,
    count: &mut usize,
    entries: &mut BTreeMap<String, SnapshotEntry>,
) -> Result<()> {
    let directory = root.join(relative);
    let read_dir = fs::read_dir(&directory)
        .with_context(|| format!("reading snapshot directory {}", directory.display()))?;
    for item in read_dir {
        let item = item.with_context(|| format!("walking {}", directory.display()))?;
        let path = item.path();
        let rel = relative.join(item.file_name());
        if relative.as_os_str().is_empty()
            && matches!(
                item.file_name().to_str(),
                Some(".git" | ".hg" | ".svn" | ".jj")
            )
        {
            continue;
        }
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("reading snapshot metadata {}", path.display()))?;
        let key = encode_path(&rel);
        let file_type = metadata.file_type();
        // Do not recurse into the snapshot store when it is configured inside
        // the workspace. A symlink *to* that directory is still captured as a
        // symlink; canonicalizing it here would follow it and incorrectly omit
        // the link itself.
        if !file_type.is_symlink() {
            let canonical_guess = path.canonicalize().unwrap_or_else(|_| path.clone());
            if canonical_guess == state_root || canonical_guess.starts_with(state_root) {
                continue;
            }
        }
        *count = count.saturating_add(1);
        ensure!(
            *count <= MAX_CHECKPOINT_ENTRIES,
            "workspace checkpoint exceeds {MAX_CHECKPOINT_ENTRIES} entries"
        );
        let entry = if file_type.is_dir() {
            let entry = SnapshotEntry {
                path: key.clone(),
                kind: EntryKind::Directory,
                mode: mode(&metadata),
                len: 0,
                object: None,
                target: None,
            };
            entries.insert(key, entry);
            scan_dir(root, &rel, state_root, store, total, count, entries)?;
            continue;
        } else if file_type.is_file() {
            *total = total.saturating_add(metadata.len());
            ensure!(
                *total <= MAX_CHECKPOINT_BYTES,
                "workspace checkpoint exceeds {} MiB ceiling",
                MAX_CHECKPOINT_BYTES / 1024 / 1024
            );
            let bytes = fs::read(&path)
                .with_context(|| format!("reading snapshot file {}", path.display()))?;
            let object = digest(&bytes);
            if let Some(store) = store {
                write_object(store, &object, &bytes)?;
            }
            SnapshotEntry {
                path: key.clone(),
                kind: EntryKind::File,
                mode: mode(&metadata),
                len: bytes.len() as u64,
                object: Some(object),
                target: None,
            }
        } else if file_type.is_symlink() {
            let target = fs::read_link(&path)
                .with_context(|| format!("reading symlink {}", path.display()))?;
            SnapshotEntry {
                path: key.clone(),
                kind: EntryKind::Symlink,
                mode: mode(&metadata),
                len: os_bytes(target.as_os_str()).len() as u64,
                object: None,
                target: Some(encode_os(target.as_os_str())),
            }
        } else {
            bail!(
                "cannot checkpoint special filesystem entry {}",
                path.display()
            );
        };
        entries.insert(key, entry);
    }
    Ok(())
}

impl Store {
    fn open(root: &Path, state_root: &Path) -> Result<Self> {
        let workspace_id = digest(&os_bytes(root.as_os_str()));
        let workspace_dir = state_root.join("workspaces").join(&workspace_id);
        let objects_dir = workspace_dir.join("objects");
        let manifests_dir = workspace_dir.join("manifests");
        fs::create_dir_all(&objects_dir)
            .with_context(|| format!("creating snapshot store {}", objects_dir.display()))?;
        fs::create_dir_all(&manifests_dir)
            .with_context(|| format!("creating snapshot store {}", manifests_dir.display()))?;
        Ok(Self {
            workspace_dir,
            objects_dir,
            manifests_dir,
            workspace_id,
        })
    }
}

fn load(root: &Path, state_root: &Path, id: &str) -> Result<(Store, Manifest, PathBuf)> {
    let store = Store::open(root, state_root)?;
    let rest = id
        .strip_prefix(ID_PREFIX)
        .context("not an internal checkpoint id")?;
    let (workspace, manifest_id) = rest
        .split_once(':')
        .context("malformed internal checkpoint id")?;
    ensure!(
        workspace == store.workspace_id,
        "checkpoint belongs to a different workspace"
    );
    ensure!(
        manifest_id.len() == 64 && manifest_id.bytes().all(|b| b.is_ascii_hexdigit()),
        "malformed manifest digest"
    );
    let path = store.manifests_dir.join(format!("{manifest_id}.json"));
    let bytes =
        fs::read(&path).with_context(|| format!("reading snapshot manifest {}", path.display()))?;
    ensure!(
        digest(&bytes) == manifest_id,
        "snapshot manifest digest mismatch"
    );
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing snapshot manifest {}", path.display()))?;
    ensure!(
        manifest.version == 1 && manifest.workspace == store.workspace_id,
        "invalid snapshot manifest"
    );
    // Rewrite identical bytes to update the manifest's LRU timestamp.
    let _ = fs::write(&path, &bytes);
    Ok((store, manifest, path))
}

fn write_object(store: &Store, object: &str, bytes: &[u8]) -> Result<()> {
    let directory = store.objects_dir.join(&object[..2]);
    fs::create_dir_all(&directory)?;
    atomic_write_if_missing(&directory.join(&object[2..]), bytes)
}

fn read_object(store: &Store, object: &str) -> Result<Vec<u8>> {
    ensure!(
        object.len() == 64 && object.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid object digest"
    );
    fs::read(store.objects_dir.join(&object[..2]).join(&object[2..]))
        .with_context(|| format!("reading snapshot object {object}"))
}

fn atomic_write_if_missing(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    let temp = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        match fs::rename(&temp, path) {
            Ok(()) => Ok(()),
            Err(_) if path.exists() => Ok(()),
            Err(error) => Err(error.into()),
        }
    })();
    let _ = fs::remove_file(&temp);
    result
}

fn atomic_replace(path: &Path, bytes: &[u8], file_mode: u32) -> Result<()> {
    let parent = path.parent().context("snapshot target has no parent")?;
    let temp = parent.join(format!(
        ".hi-restore-{}-{}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        set_mode(&temp, file_mode)?;
        if path.exists() {
            remove_any(path)?;
        }
        fs::rename(&temp, path)?;
        Ok(())
    })();
    let _ = fs::remove_file(temp);
    result.with_context(|| format!("restoring {}", path.display()))
}

fn enforce_limits(store: &Store) -> Result<()> {
    let mut manifests = manifest_files(store)?;
    while manifests.len() > MAX_MANIFESTS {
        fs::remove_file(&manifests.remove(0).1)?;
    }
    gc_objects(store)?;
    manifests = manifest_files(store)?;
    while directory_size(&store.workspace_dir)? > MAX_WORKSPACE_BYTES && manifests.len() > 1 {
        fs::remove_file(&manifests.remove(0).1)?;
        gc_objects(store)?;
    }
    ensure!(
        directory_size(&store.workspace_dir)? <= MAX_WORKSPACE_BYTES,
        "snapshot store exceeds 2 GiB workspace limit"
    );
    Ok(())
}

fn manifest_files(store: &Store) -> Result<Vec<(SystemTime, PathBuf)>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(&store.manifests_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("json") {
            let modified = fs::metadata(&path)?.modified().unwrap_or(UNIX_EPOCH);
            files.push((modified, path));
        }
    }
    files.sort_by_key(|item| item.0);
    Ok(files)
}

fn gc_objects(store: &Store) -> Result<()> {
    let mut keep = HashSet::new();
    for (_, path) in manifest_files(store)? {
        let manifest: Manifest = serde_json::from_slice(&fs::read(path)?)?;
        keep.extend(
            manifest
                .entries
                .into_iter()
                .filter_map(|entry| entry.object),
        );
    }
    if !store.objects_dir.exists() {
        return Ok(());
    }
    for prefix in fs::read_dir(&store.objects_dir)? {
        let prefix = prefix?.path();
        if !prefix.is_dir() {
            continue;
        }
        for object in fs::read_dir(&prefix)? {
            let path = object?.path();
            let id = format!(
                "{}{}",
                prefix.file_name().unwrap_or_default().to_string_lossy(),
                path.file_name().unwrap_or_default().to_string_lossy()
            );
            if !keep.contains(&id) {
                let _ = fs::remove_file(path);
            }
        }
        let _ = fs::remove_dir(prefix);
    }
    Ok(())
}

fn recover_temps(store: &Store) -> Result<()> {
    fn clean(directory: &Path) -> Result<()> {
        if !directory.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path.is_dir() {
                clean(&path)?;
            } else if path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.starts_with("tmp-"))
            {
                let _ = fs::remove_file(path);
            }
        }
        Ok(())
    }
    clean(&store.workspace_dir)
}

fn append_created_content(out: &mut String, store: &Store, entry: &SnapshotEntry) -> Result<()> {
    if entry.kind != EntryKind::File || entry.len as usize > MAX_DIFF_FILE_BYTES {
        return Ok(());
    }
    let bytes = read_object(store, entry.object.as_deref().unwrap_or(""))?;
    if let Ok(text) = std::str::from_utf8(&bytes) {
        out.push('\n');
        for line in text.lines().take(200) {
            out.push_str("+ ");
            out.push_str(line);
            out.push('\n');
        }
    }
    Ok(())
}

fn remove_any(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        fs::remove_dir_all(path)?;
    } else {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn canonical_root(root: &Path) -> Result<PathBuf> {
    let root = root
        .canonicalize()
        .with_context(|| format!("canonicalizing {}", root.display()))?;
    ensure!(
        root.is_dir(),
        "workspace root is not a directory: {}",
        root.display()
    );
    Ok(root)
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn directory_size(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    if !path.exists() {
        return Ok(0);
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total = total.saturating_add(directory_size(&entry.path())?);
        } else {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

fn encode_path(path: &Path) -> String {
    encode_os(path.as_os_str())
}

fn decode_path(encoded: &str) -> Result<PathBuf> {
    let path = PathBuf::from(decode_os(encoded)?);
    ensure!(
        !path.is_absolute()
            && path
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_))),
        "snapshot contains unsafe path"
    );
    Ok(path)
}

fn encode_os(value: &OsStr) -> String {
    os_bytes(value)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn decode_os(encoded: &str) -> Result<OsString> {
    ensure!(
        encoded.len().is_multiple_of(2) && encoded.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid encoded path"
    );
    let bytes: Vec<u8> = (0..encoded.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&encoded[index..index + 2], 16))
        .collect::<std::result::Result<_, _>>()?;
    Ok(os_from_bytes(bytes))
}

#[cfg(unix)]
fn os_bytes(value: &OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    value.as_bytes().to_vec()
}

#[cfg(not(unix))]
fn os_bytes(value: &OsStr) -> Vec<u8> {
    value.to_string_lossy().as_bytes().to_vec()
}

#[cfg(unix)]
fn os_from_bytes(bytes: Vec<u8>) -> OsString {
    use std::os::unix::ffi::OsStringExt;
    OsString::from_vec(bytes)
}

#[cfg(not(unix))]
fn os_from_bytes(bytes: Vec<u8>) -> OsString {
    OsString::from(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(unix)]
fn mode(metadata: &fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o7777
}

#[cfg(not(unix))]
fn mode(metadata: &fs::Metadata) -> u32 {
    if metadata.permissions().readonly() {
        0o444
    } else {
        0o666
    }
}

#[cfg(unix)]
fn set_mode(path: &Path, value: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(value))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_mode(path: &Path, value: u32) -> Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(value & 0o222 == 0);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

#[cfg(unix)]
fn create_symlink(target: &OsStr, path: &Path) -> Result<()> {
    std::os::unix::fs::symlink(target, path)
        .with_context(|| format!("restoring symlink {}", path.display()))
}

#[cfg(not(unix))]
fn create_symlink(_target: &OsStr, path: &Path) -> Result<()> {
    bail!(
        "restoring symlinks is unsupported on this platform: {}",
        path.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roots(label: &str) -> (PathBuf, PathBuf) {
        let base = std::env::temp_dir().join(format!(
            "hi-internal-snapshot-{label}-{}-{}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let workspace = base.join("workspace");
        let state = base.join("state");
        fs::create_dir_all(&workspace).unwrap();
        fs::create_dir_all(&state).unwrap();
        (workspace, state)
    }

    #[test]
    fn restores_files_modes_directories_and_symlinks() {
        let (workspace, state) = roots("restore");
        fs::create_dir(workspace.join("empty")).unwrap();
        fs::create_dir(workspace.join("mode-dir")).unwrap();
        fs::write(workspace.join("keep"), b"before\r\n").unwrap();
        #[cfg(unix)]
        {
            set_mode(&workspace.join("keep"), 0o755).unwrap();
            set_mode(&workspace.join("mode-dir"), 0o700).unwrap();
            std::os::unix::fs::symlink("keep", workspace.join("link")).unwrap();
        }
        let id = create(&workspace, &state).unwrap();
        fs::write(workspace.join("keep"), b"after\n").unwrap();
        fs::write(workspace.join("new"), b"created").unwrap();
        fs::remove_dir(workspace.join("empty")).unwrap();
        #[cfg(unix)]
        {
            set_mode(&workspace.join("mode-dir"), 0o755).unwrap();
            fs::remove_file(workspace.join("link")).unwrap();
            std::os::unix::fs::symlink("elsewhere", workspace.join("link")).unwrap();
        }
        let changed = restore(&workspace, &state, &id).unwrap();
        assert!(changed >= 3);
        assert_eq!(fs::read(workspace.join("keep")).unwrap(), b"before\r\n");
        assert!(!workspace.join("new").exists());
        assert!(workspace.join("empty").is_dir());
        #[cfg(unix)]
        {
            assert_eq!(mode(&fs::metadata(workspace.join("keep")).unwrap()), 0o755);
            assert_eq!(
                mode(&fs::metadata(workspace.join("mode-dir")).unwrap()),
                0o700
            );
            assert_eq!(
                fs::read_link(workspace.join("link")).unwrap(),
                PathBuf::from("keep")
            );
        }
        let _ = fs::remove_dir_all(workspace.parent().unwrap());
    }

    #[test]
    fn sealed_restore_refuses_external_change() {
        let (workspace, state) = roots("sealed");
        fs::write(workspace.join("file"), "v1").unwrap();
        let before = create(&workspace, &state).unwrap();
        fs::write(workspace.join("file"), "v2").unwrap();
        let after = create(&workspace, &state).unwrap();
        fs::write(workspace.join("file"), "external").unwrap();
        assert!(restore_sealed(&workspace, &state, &before, &after).is_err());
        assert_eq!(
            fs::read_to_string(workspace.join("file")).unwrap(),
            "external"
        );
        let _ = fs::remove_dir_all(workspace.parent().unwrap());
    }

    #[test]
    fn missing_second_restore_object_leaves_all_targets_unchanged() {
        let (workspace, state) = roots("corrupt-object");
        fs::write(workspace.join("a"), "before-a").unwrap();
        fs::write(workspace.join("b"), "before-b").unwrap();
        let checkpoint = create(&workspace, &state).unwrap();
        let root = workspace.canonicalize().unwrap();
        let (store, manifest, _) = load(&root, &state, &checkpoint).unwrap();
        let b = manifest
            .entries
            .iter()
            .find(|entry| decode_path(&entry.path).unwrap() == Path::new("b"))
            .and_then(|entry| entry.object.as_deref())
            .unwrap();
        fs::remove_file(store.objects_dir.join(&b[..2]).join(&b[2..])).unwrap();
        fs::write(workspace.join("a"), "after-a").unwrap();
        fs::write(workspace.join("b"), "after-b").unwrap();

        assert!(restore(&workspace, &state, &checkpoint).is_err());

        assert_eq!(fs::read_to_string(workspace.join("a")).unwrap(), "after-a");
        assert_eq!(fs::read_to_string(workspace.join("b")).unwrap(), "after-b");
        let _ = fs::remove_dir_all(workspace.parent().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn failed_capture_collects_unreferenced_objects() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let (workspace, state) = roots("failed-gc");
        let store = Store::open(&workspace.canonicalize().unwrap(), &state).unwrap();
        let bytes = b"unreferenced";
        let object = digest(bytes);
        write_object(&store, &object, bytes).unwrap();
        let fifo = workspace.join("unsupported-fifo");
        let fifo_c = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        // SAFETY: fifo_c is a valid NUL-terminated path owned for the call.
        assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) }, 0);

        assert!(capture(&workspace, &state, &store).is_err());
        assert!(
            !store
                .objects_dir
                .join(&object[..2])
                .join(&object[2..])
                .exists(),
            "failed capture left an unreachable object"
        );
        let _ = fs::remove_dir_all(workspace.parent().unwrap());
    }
}
