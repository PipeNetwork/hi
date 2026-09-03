use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Component, Path};

use anyhow::{Context, Result, ensure};
use serde::Serialize;

pub(crate) const STAGED_CANDIDATE_DIR: &str = ".hi-smoke-candidate";

/// A content-addressed view of the harness-owned isolation tree. The active
/// workspace is deliberately omitted: it already has dedicated patch/listing
/// evidence, while this snapshot protects every writable sibling made
/// available to the full `hi` process and its descendants.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct IsolationSnapshot {
    entries: BTreeMap<String, IsolationEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct IsolationEntry {
    kind: IsolationEntryKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_assurance: Option<ContentAssurance>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ContentAssurance {
    Validated,
    Invalid,
    Opaque,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum IsolationEntryKind {
    Directory,
    File,
    Symlink,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum IsolationChange {
    Created,
    Modified,
    Removed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct IsolationMutation {
    path: String,
    change: IsolationChange,
    disposition: IsolationDisposition,
    #[serde(skip_serializing_if = "Option::is_none")]
    before: Option<IsolationEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    after: Option<IsolationEntry>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum IsolationDisposition {
    ExpectedRuntime,
    UnattributedAllowlisted,
    UnexpectedOutsideWorkspace,
}

/// Per-case identities that turn broad digest-shaped state paths into the
/// exact paths production derives for this isolated workspace.
///
/// The guarantee is deliberately precise: the final audit rejects unknown
/// paths, wrong per-case project/workspace identities, symlinks/special files,
/// malformed structured state, and sensitive opaque lifecycle mutations seen
/// in a case that executed the process tool. It does **not** cryptographically
/// identify the writer of a correctly shaped, correctly formatted runtime
/// file. That stronger guarantee requires a separately sandboxed execution
/// broker; PID/event/timing correlation cannot establish inode provenance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct IsolationPolicy {
    project_key: String,
    snapshot_workspace_key: String,
    transaction_workspace_key: String,
    process_tool_activity: bool,
}

impl IsolationPolicy {
    pub(crate) fn for_workspace(workspace: &Path, process_tool_activity: bool) -> Result<Self> {
        use sha2::{Digest, Sha256};

        let canonical = fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
        let bytes = canonical.as_os_str().as_encoded_bytes();
        let mut project = 0xcbf29ce484222325_u64;
        for byte in bytes {
            project ^= u64::from(*byte);
            project = project.wrapping_mul(0x100000001b3);
        }
        let sha256 = format!("{:x}", Sha256::digest(bytes));
        Ok(Self {
            project_key: format!("{project:016x}"),
            snapshot_workspace_key: sha256.clone(),
            transaction_workspace_key: sha256,
            process_tool_activity,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct IsolationEvidence {
    pub schema_version: u16,
    pub baseline_entry_count: usize,
    pub final_entry_count: usize,
    pub expected_runtime_mutation_count: usize,
    pub unexpected_mutation_count: usize,
    pub mutations: Vec<IsolationMutation>,
}

impl IsolationEvidence {
    pub(crate) fn unexpected_paths(&self) -> Vec<&str> {
        self.mutations
            .iter()
            .filter(|mutation| mutation.disposition != IsolationDisposition::ExpectedRuntime)
            .map(|mutation| mutation.path.as_str())
            .collect()
    }
}

pub(crate) fn capture(root: &Path) -> Result<IsolationSnapshot> {
    let metadata = fs::symlink_metadata(root)
        .with_context(|| format!("reading isolation root {}", root.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "isolation root is not a real directory: {}",
        root.display()
    );
    let mut snapshot = IsolationSnapshot::default();
    collect(root, root, &mut snapshot.entries)?;
    Ok(snapshot)
}

fn collect(
    root: &Path,
    directory: &Path,
    entries: &mut BTreeMap<String, IsolationEntry>,
) -> Result<()> {
    let mut children = fs::read_dir(directory)
        .with_context(|| format!("reading isolation directory {}", directory.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    children.sort_by_key(fs::DirEntry::file_name);
    for child in children {
        let path = child.path();
        let relative = path
            .strip_prefix(root)
            .with_context(|| format!("containing isolation path {}", path.display()))?;
        let portable = portable_relative_path(relative)?;

        // Workspace mutations have their own binary patch and final listing.
        // Omitting the subtree also keeps this hard-invariant scan bounded for
        // scenarios that build large fixtures.
        if portable == "workspace" || portable == STAGED_CANDIDATE_DIR {
            continue;
        }

        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("reading isolation entry {}", path.display()))?;
        let file_type = metadata.file_type();
        let entry = if file_type.is_dir() {
            IsolationEntry {
                kind: IsolationEntryKind::Directory,
                bytes: None,
                digest: None,
                mode: entry_mode(&metadata),
                content_assurance: None,
            }
        } else if file_type.is_file() {
            IsolationEntry {
                kind: IsolationEntryKind::File,
                bytes: Some(metadata.len()),
                digest: Some(hash_file(&path)?),
                mode: entry_mode(&metadata),
                content_assurance: Some(content_assurance(&portable, &path)),
            }
        } else if file_type.is_symlink() {
            let target = fs::read_link(&path)
                .with_context(|| format!("reading isolation symlink {}", path.display()))?;
            IsolationEntry {
                kind: IsolationEntryKind::Symlink,
                bytes: None,
                digest: Some(
                    blake3::hash(target.as_os_str().as_encoded_bytes())
                        .to_hex()
                        .to_string(),
                ),
                mode: entry_mode(&metadata),
                content_assurance: None,
            }
        } else {
            IsolationEntry {
                kind: IsolationEntryKind::Other,
                bytes: None,
                digest: None,
                mode: entry_mode(&metadata),
                content_assurance: None,
            }
        };
        entries.insert(portable, entry);
        if file_type.is_dir() {
            collect(root, &path, entries)?;
        }
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("opening isolation evidence file {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hashing isolation evidence file {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn content_assurance(portable: &str, path: &Path) -> ContentAssurance {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => return ContentAssurance::Invalid,
    };
    let parts = portable.split('/').collect::<Vec<_>>();
    match parts.as_slice() {
        ["events", "tui.jsonl"] | ["session", "session.jsonl"] => validate_jsonl(&bytes, true),
        ["home", ".hi", "crash", marker] if valid_crash_marker(marker) => {
            if bytes.is_empty() {
                ContentAssurance::Validated
            } else {
                ContentAssurance::Invalid
            }
        }
        ["xdg", "config", "hi", "models-cache.json"] => validate_models_cache(&bytes),
        ["xdg", "data", "hi", "machine-id"] => {
            if std::str::from_utf8(&bytes)
                .ok()
                .map(str::trim)
                .is_some_and(|value| is_lower_hex(value, 16))
            {
                ContentAssurance::Validated
            } else {
                ContentAssurance::Invalid
            }
        }
        ["xdg", "state", "hi", "trace-signing-key"] => {
            if bytes.len() == 32 {
                ContentAssurance::Validated
            } else {
                ContentAssurance::Invalid
            }
        }
        [.., "manifests", name]
            if name
                .strip_suffix(".json")
                .is_some_and(|digest| is_lower_hex(digest, 64)) =>
        {
            let digest = name.strip_suffix(".json").expect("guarded above");
            if validate_json(&bytes) == ContentAssurance::Validated {
                validate_sha256_name(&bytes, digest)
            } else {
                ContentAssurance::Invalid
            }
        }
        [.., name] if name.ends_with(".json") => validate_json(&bytes),
        [.., name] if name.ends_with(".jsonl") => validate_jsonl(&bytes, false),
        [.., name] if name.ends_with(".sqlite3") => validate_sqlite(&bytes),
        [.., "objects", shard, tail] if is_lower_hex(shard, 2) && is_lower_hex(tail, 62) => {
            validate_sha256_name(&bytes, &format!("{shard}{tail}"))
        }
        _ => ContentAssurance::Opaque,
    }
}

fn validate_json(bytes: &[u8]) -> ContentAssurance {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .map(|_| ContentAssurance::Validated)
        .unwrap_or(ContentAssurance::Invalid)
}

fn validate_jsonl(bytes: &[u8], require_nonempty: bool) -> ContentAssurance {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return ContentAssurance::Invalid;
    };
    let mut count = 0;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            return ContentAssurance::Invalid;
        };
        if !value.is_object() {
            return ContentAssurance::Invalid;
        }
        count += 1;
    }
    if require_nonempty && count == 0 {
        ContentAssurance::Invalid
    } else {
        ContentAssurance::Validated
    }
}

fn validate_models_cache(bytes: &[u8]) -> ContentAssurance {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return ContentAssurance::Invalid;
    };
    let Some(entries) = value.as_object() else {
        return ContentAssurance::Invalid;
    };
    if !entries.is_empty()
        && entries.values().all(|entry| {
            entry.get("ts").is_some_and(serde_json::Value::is_u64)
                && entry.get("models").is_some_and(serde_json::Value::is_array)
        })
    {
        ContentAssurance::Validated
    } else {
        ContentAssurance::Invalid
    }
}

fn validate_sqlite(bytes: &[u8]) -> ContentAssurance {
    if bytes.starts_with(b"SQLite format 3\0") {
        ContentAssurance::Validated
    } else {
        // WAL/SHM files have different mutable binary layouts and are not
        // treated as content-authenticated lifecycle evidence.
        ContentAssurance::Opaque
    }
}

fn validate_sha256_name(bytes: &[u8], expected: &str) -> ContentAssurance {
    use sha2::{Digest, Sha256};
    if format!("{:x}", Sha256::digest(bytes)) == expected {
        ContentAssurance::Validated
    } else {
        ContentAssurance::Invalid
    }
}

#[cfg(unix)]
fn entry_mode(metadata: &fs::Metadata) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    Some(metadata.mode())
}

#[cfg(not(unix))]
fn entry_mode(_metadata: &fs::Metadata) -> Option<u32> {
    None
}

fn portable_relative_path(path: &Path) -> Result<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                anyhow::bail!(
                    "isolation evidence path escaped its root: {}",
                    path.display()
                )
            }
        }
    }
    ensure!(!parts.is_empty(), "isolation evidence path was empty");
    Ok(parts.join("/"))
}

pub(crate) fn compare_with_policy(
    baseline: &IsolationSnapshot,
    final_snapshot: &IsolationSnapshot,
    policy: Option<&IsolationPolicy>,
) -> IsolationEvidence {
    let paths = baseline
        .entries
        .keys()
        .chain(final_snapshot.entries.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut mutations = Vec::new();
    for path in paths {
        let before = baseline.entries.get(&path);
        let after = final_snapshot.entries.get(&path);
        if before == after {
            continue;
        }
        let change = match (before, after) {
            (None, Some(_)) => IsolationChange::Created,
            (Some(_), None) => IsolationChange::Removed,
            (Some(_), Some(_)) => IsolationChange::Modified,
            (None, None) => unreachable!("path came from one of the snapshots"),
        };
        let disposition = runtime_mutation_disposition(&path, after, policy);
        mutations.push(IsolationMutation {
            path,
            change,
            disposition,
            before: before.cloned(),
            after: after.cloned(),
        });
    }
    let expected_runtime_mutation_count = mutations
        .iter()
        .filter(|mutation| mutation.disposition == IsolationDisposition::ExpectedRuntime)
        .count();
    let unexpected_mutation_count = mutations.len() - expected_runtime_mutation_count;
    IsolationEvidence {
        schema_version: 1,
        baseline_entry_count: baseline.entries.len(),
        final_entry_count: final_snapshot.entries.len(),
        expected_runtime_mutation_count,
        unexpected_mutation_count,
        mutations,
    }
}

/// This list is intentionally narrow and structural. It admits only paths
/// owned by normal `hi` lifecycle machinery; a tool writing `../home/pwn`, an
/// ad-hoc XDG file, a replacement config, or a second session/event stream is
/// still an invariant violation. Opaque runtime identifiers must retain their
/// production digest shape instead of becoming arbitrary path components.
fn runtime_mutation_disposition(
    path: &str,
    after: Option<&IsolationEntry>,
    policy: Option<&IsolationPolicy>,
) -> IsolationDisposition {
    let Some(after) = after else {
        // Harness/Hi state may be created or updated, but deleting a seeded
        // session or another pre-run sibling is never normal lifecycle state.
        return IsolationDisposition::UnexpectedOutsideWorkspace;
    };
    let expected_kind = if expected_runtime_directory(path) {
        IsolationEntryKind::Directory
    } else {
        IsolationEntryKind::File
    };
    if after.kind != expected_kind {
        // In particular, a symlink at a known state path must not inherit the
        // allowlist of the file or directory it impersonates.
        return IsolationDisposition::UnexpectedOutsideWorkspace;
    }
    let parts = path.split('/').collect::<Vec<_>>();
    let structurally_expected = match parts.as_slice() {
        ["events", "tui.jsonl"] | ["session", "session.jsonl"] => true,

        ["home", ".hi"] | ["home", ".hi", "crash"] => true,
        ["home", ".hi", "crash", marker] => valid_crash_marker(marker),
        // Cargo 1.90+ creates these exact lock/global-cache files even for an
        // offline metadata probe. Do not admit registry/, git/, config files,
        // or arbitrary HOME/.cargo descendants.
        ["home", ".cargo"]
        | ["home", ".cargo", ".global-cache"]
        | ["home", ".cargo", ".package-cache"]
        | ["home", ".cargo", ".package-cache-mutate"] => true,

        ["xdg", "config", "hi"] | ["xdg", "config", "hi", "models-cache.json"] => true,

        ["xdg", "data", "hi"]
        | ["xdg", "data", "hi", "projects"]
        | ["xdg", "data", "hi", "feedback"]
        | ["xdg", "data", "hi", "feedback", "pipe-session.json"]
        | ["xdg", "data", "hi", "machine-id"] => true,
        ["xdg", "data", "hi", database] if sqlite_file(database, "portal-sync.sqlite3") => true,
        ["xdg", "data", "hi", "projects", project, rest @ ..]
            if policy.is_some_and(|policy| project == &policy.project_key) =>
        {
            expected_project_state(rest, policy.expect("guarded above"))
        }

        ["xdg", "state", "hi"] => true,

        _ => false,
    };
    if !structurally_expected {
        return IsolationDisposition::UnexpectedOutsideWorkspace;
    }
    match after.content_assurance {
        Some(ContentAssurance::Invalid) => IsolationDisposition::UnattributedAllowlisted,
        Some(ContentAssurance::Opaque)
            if policy.is_some_and(|policy| policy.process_tool_activity)
                && ambiguous_during_process_tool(path) =>
        {
            IsolationDisposition::UnattributedAllowlisted
        }
        _ => IsolationDisposition::ExpectedRuntime,
    }
}

fn ambiguous_during_process_tool(path: &str) -> bool {
    matches!(
        path,
        "xdg/config/hi/models-cache.json"
            | "xdg/data/hi/machine-id"
            | "xdg/state/hi/trace-signing-key"
    ) || path.starts_with("home/.hi/crash/")
        || path.contains("/resource-leases/")
        || path.contains("/verification-flights/")
}

fn expected_runtime_directory(path: &str) -> bool {
    let parts = path.split('/').collect::<Vec<_>>();
    match parts.as_slice() {
        ["home", ".hi"] | ["home", ".hi", "crash"] | ["home", ".cargo"] => true,
        ["xdg", "config", "hi"] => true,
        ["xdg", "data", "hi"]
        | ["xdg", "data", "hi", "projects"]
        | ["xdg", "data", "hi", "feedback"] => true,
        ["xdg", "data", "hi", "projects", project, rest @ ..] if is_lower_hex(project, 16) => {
            expected_project_directory(rest)
        }
        ["xdg", "state", "hi"] | ["xdg", "state", "hi", "rsi"] => true,
        ["xdg", "state", "hi", "rsi", trace, rest @ ..] if is_lower_hex(trace, 32) => {
            matches!(rest, [] | ["blobs"])
        }
        _ => false,
    }
}

fn expected_project_directory(rest: &[&str]) -> bool {
    match rest {
        []
        | ["runtime"]
        | ["runtime", "workspaces"]
        | ["runtime", "transactions"]
        | ["runtime", "resource-leases"]
        | ["runtime", "verification-flights"] => true,
        ["runtime", "workspaces", workspace, rest @ ..] if is_lower_hex(workspace, 64) => {
            matches!(rest, [] | ["objects"] | ["manifests"])
                || matches!(rest, ["objects", shard] if is_lower_hex(shard, 2))
        }
        ["runtime", "transactions", workspace] if is_lower_hex(workspace, 64) => true,
        _ => false,
    }
}

fn valid_crash_marker(marker: &str) -> bool {
    if marker == "last-crash.bin" {
        return true;
    }
    let Some(identity) = marker
        .strip_prefix("last-crash-")
        .and_then(|value| value.strip_suffix(".bin"))
    else {
        return false;
    };
    let Some((pid, nonce)) = identity.split_once('-') else {
        return false;
    };
    !pid.is_empty() && pid.bytes().all(|byte| byte.is_ascii_digit()) && is_lower_hex(nonce, 32)
}

fn expected_project_state(rest: &[&str], policy: &IsolationPolicy) -> bool {
    match rest {
        [] | ["runtime"] => true,
        [file]
            if matches!(
                *file,
                "loops.json" | "loops.lock" | "history" | "activity.jsonl"
            ) =>
        {
            true
        }
        ["runtime", file]
            if sqlite_file(file, "events.sqlite3")
                || matches!(
                    *file,
                    "input-history" | "outcome-last.json" | "orchestration-metrics.csv"
                ) =>
        {
            true
        }
        ["runtime", "workspaces"] => true,
        ["runtime", "workspaces", workspace, rest @ ..]
            if workspace == &policy.snapshot_workspace_key =>
        {
            expected_snapshot_state(rest)
        }
        ["runtime", "transactions"] => true,
        ["runtime", "transactions", workspace, rest @ ..]
            if workspace == &policy.transaction_workspace_key =>
        {
            rest.is_empty() || matches!(rest, [journal] if valid_transaction_journal(journal))
        }
        ["runtime", directory]
            if matches!(*directory, "resource-leases" | "verification-flights") =>
        {
            true
        }
        ["runtime", "resource-leases", leaf] if valid_resource_penalty(leaf) => true,
        _ => false,
    }
}

fn expected_snapshot_state(rest: &[&str]) -> bool {
    match rest {
        [] | ["objects"] | ["manifests"] => true,
        ["manifests", manifest]
            if manifest
                .strip_suffix(".json")
                .is_some_and(|digest| is_lower_hex(digest, 64)) =>
        {
            true
        }
        ["objects", shard] if is_lower_hex(shard, 2) => true,
        ["objects", shard, tail] if is_lower_hex(shard, 2) && is_lower_hex(tail, 62) => true,
        _ => false,
    }
}

fn sqlite_file(actual: &str, base: &str) -> bool {
    actual == base || actual == format!("{base}-wal") || actual == format!("{base}-shm")
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_resource_penalty(value: &str) -> bool {
    ["setup", "model", "verifier", "merge"]
        .iter()
        .any(|class| value == format!("{class}-penalty"))
}

fn valid_transaction_journal(value: &str) -> bool {
    let Some(stem) = value.strip_suffix(".json") else {
        return false;
    };
    let Some((pid, sequence)) = stem.split_once('-') else {
        return false;
    };
    !pid.is_empty()
        && !sequence.is_empty()
        && pid.bytes().all(|byte| byte.is_ascii_digit())
        && sequence.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sibling_write_is_an_unexpected_hard_invariant_mutation() {
        let root = tempfile::tempdir().unwrap();
        for directory in ["workspace", "home", "xdg/config", "xdg/data", "xdg/state"] {
            fs::create_dir_all(root.path().join(directory)).unwrap();
        }
        let baseline = capture(root.path()).unwrap();
        fs::write(root.path().join("home/model-owned.txt"), "escaped\n").unwrap();
        let final_snapshot = capture(root.path()).unwrap();
        let evidence = compare_with_policy(&baseline, &final_snapshot, None);

        assert_eq!(evidence.unexpected_mutation_count, 1);
        assert_eq!(evidence.unexpected_paths(), vec!["home/model-owned.txt"]);
    }

    #[test]
    fn normal_hi_runtime_state_is_expected_but_ad_hoc_xdg_state_is_not() {
        let root = tempfile::tempdir().unwrap();
        for directory in [
            "workspace",
            "home",
            "events",
            "session",
            "xdg/config",
            "xdg/data",
            "xdg/state",
        ] {
            fs::create_dir_all(root.path().join(directory)).unwrap();
        }
        let baseline = capture(root.path()).unwrap();
        let policy = IsolationPolicy::for_workspace(&root.path().join("workspace"), false).unwrap();
        let manifest_digest = {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(b"{}"))
        };

        let files = [
            ("events/tui.jsonl".to_owned(), b"{}\n".to_vec()),
            ("session/session.jsonl".to_owned(), b"{}\n".to_vec()),
            (
                "home/.hi/crash/last-crash-42-0123456789abcdef0123456789abcdef.bin".to_owned(),
                Vec::new(),
            ),
            (
                "home/.cargo/.global-cache".to_owned(),
                b"runtime\n".to_vec(),
            ),
            (
                "home/.cargo/.package-cache".to_owned(),
                b"runtime\n".to_vec(),
            ),
            (
                "home/.cargo/.package-cache-mutate".to_owned(),
                b"runtime\n".to_vec(),
            ),
            (
                "xdg/config/hi/models-cache.json".to_owned(),
                br#"{"route":{"ts":1,"models":[]}}"#.to_vec(),
            ),
            (
                "xdg/data/hi/portal-sync.sqlite3".to_owned(),
                b"SQLite format 3\0runtime".to_vec(),
            ),
            (
                format!("xdg/data/hi/projects/{}/loops.json", policy.project_key),
                b"{}".to_vec(),
            ),
            (
                format!(
                    "xdg/data/hi/projects/{}/runtime/events.sqlite3",
                    policy.project_key
                ),
                b"SQLite format 3\0runtime".to_vec(),
            ),
            (
                format!(
                    "xdg/data/hi/projects/{}/runtime/input-history",
                    policy.project_key
                ),
                b"runtime\n".to_vec(),
            ),
            (
                format!(
                    "xdg/data/hi/projects/{}/runtime/workspaces/{}/manifests/{manifest_digest}.json",
                    policy.project_key, policy.snapshot_workspace_key,
                ),
                b"{}".to_vec(),
            ),
        ];
        for (relative, contents) in &files {
            let path = root.path().join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        }
        fs::write(root.path().join("xdg/state/model-owned.txt"), "escaped\n").unwrap();

        let evidence =
            compare_with_policy(&baseline, &capture(root.path()).unwrap(), Some(&policy));
        assert_eq!(
            evidence.unexpected_paths(),
            vec!["xdg/state/model-owned.txt"]
        );
        assert!(evidence.expected_runtime_mutation_count > files.len());
    }

    #[test]
    fn process_tool_cannot_hide_invalid_content_at_an_allowlisted_identity_path() {
        let root = tempfile::tempdir().unwrap();
        for directory in ["workspace", "xdg/config/hi"] {
            fs::create_dir_all(root.path().join(directory)).unwrap();
        }
        let baseline = capture(root.path()).unwrap();
        fs::write(
            root.path().join("xdg/config/hi/models-cache.json"),
            "model-controlled",
        )
        .unwrap();
        let policy = IsolationPolicy::for_workspace(&root.path().join("workspace"), true).unwrap();
        let evidence =
            compare_with_policy(&baseline, &capture(root.path()).unwrap(), Some(&policy));

        assert_eq!(
            evidence.unexpected_paths(),
            vec!["xdg/config/hi/models-cache.json"]
        );
        let mutation = evidence
            .mutations
            .iter()
            .find(|mutation| mutation.path == "xdg/config/hi/models-cache.json")
            .unwrap();
        assert_eq!(
            mutation.disposition,
            IsolationDisposition::UnattributedAllowlisted
        );
        assert_eq!(
            mutation.after.as_ref().unwrap().content_assurance,
            Some(ContentAssurance::Invalid)
        );
    }

    #[cfg(unix)]
    #[test]
    fn immutable_sibling_permission_changes_are_detected() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("workspace")).unwrap();
        fs::create_dir_all(root.path().join("config")).unwrap();
        let config = root.path().join("config/hi.toml");
        fs::write(&config, "[sync]\nmode='off'\n").unwrap();
        let baseline = capture(root.path()).unwrap();
        fs::set_permissions(&config, fs::Permissions::from_mode(0o600)).unwrap();

        let evidence = compare_with_policy(&baseline, &capture(root.path()).unwrap(), None);
        assert_eq!(evidence.unexpected_paths(), vec!["config/hi.toml"]);
    }
}
