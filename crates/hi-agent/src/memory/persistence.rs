//! Conflict-safe publication and undo for markdown memory.

use std::fs::{self, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};

use super::{MEMORY_HEADER, global_memory_file, memory_file_at, strip_header};

const MAX_MEMORY_PREIMAGE_BYTES: usize = 1024 * 1024;

/// Empty bodies are skipped so a blank distillation cannot wipe existing notes.
pub fn write_memory(path: &Path, body: &str) -> Result<usize, String> {
    if body.trim().is_empty() {
        return Ok(0);
    }
    write_memory_replace(path, body)
}

/// Write `body` even when empty (a header-only file).
fn write_memory_replace(path: &Path, body: &str) -> Result<usize, String> {
    let preimage = read_memory_preimage(path)?;
    let root = existing_transaction_root(path)?;
    write_memory_replace_if_unchanged_at(
        &root,
        &hi_tools::checkpoint::default_state_root(),
        path,
        body,
        preimage.as_deref(),
    )
}

/// Every durable filesystem entry changed by a memory publication.
pub(crate) fn memory_write_paths(path: &Path) -> Vec<PathBuf> {
    vec![path.to_path_buf(), undo_sidecar(path)]
}

/// Read exact source bytes without following a final symlink. Missing is
/// distinct from an unreadable file so an IO error never becomes a replacement.
pub(crate) fn read_memory_preimage(path: &Path) -> Result<Option<Vec<u8>>, String> {
    match read_regular_file_no_follow(path, MAX_MEMORY_PREIMAGE_BYTES) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("couldn't read {}: {error}", path.display())),
    }
}

fn read_regular_file_no_follow(path: &Path, max_bytes: usize) -> std::io::Result<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    #[cfg(not(unix))]
    if fs::symlink_metadata(path)?.file_type().is_symlink() {
        return Err(std::io::Error::other("refusing to follow memory symlink"));
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::other("memory path is not a regular file"));
    }
    let mut bytes = Vec::new();
    file.take(max_bytes.saturating_add(1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(std::io::Error::other(format!(
            "memory preimage exceeds the {max_bytes}-byte safety limit"
        )));
    }
    Ok(bytes)
}

pub(crate) fn memory_body_from_preimage(preimage: Option<&[u8]>) -> String {
    preimage
        .map(String::from_utf8_lossy)
        .map(|raw| strip_header(&raw))
        .unwrap_or_default()
}

/// Atomically publish a target and its undo sidecar against an exact source
/// revision inside the supplied authoritative workspace.
pub(crate) fn write_memory_replace_if_unchanged_at(
    root: &Path,
    state_root: &Path,
    path: &Path,
    body: &str,
    expected: Option<&[u8]>,
) -> Result<usize, String> {
    let body = body.trim();
    let notes = body.lines().filter(|line| !line.trim().is_empty()).count();
    let content = format!("{MEMORY_HEADER}\n{body}\n").into_bytes();
    let mutations = vec![
        hi_tools::PlannedFileMutation::write_from_preimage(path, expected, content),
        hi_tools::PlannedFileMutation::write(undo_sidecar(path), expected.unwrap_or_default()),
    ];
    hi_tools::MutationPlan::new_with_state(root, state_root, mutations)
        .and_then(hi_tools::MutationPlan::commit)
        .map_err(|error| format!("couldn't publish {}: {error:#}", path.display()))?;
    Ok(notes)
}

pub(crate) fn write_memory_replace_if_unchanged(
    path: &Path,
    body: &str,
    expected: Option<&[u8]>,
) -> Result<usize, String> {
    let root = existing_transaction_root(path)?;
    write_memory_replace_if_unchanged_at(
        &root,
        &hi_tools::checkpoint::default_state_root(),
        path,
        body,
        expected,
    )
}

fn existing_transaction_root(path: &Path) -> Result<PathBuf, String> {
    let mut cursor = path
        .parent()
        .ok_or_else(|| format!("no parent directory for {}", path.display()))?;
    while !cursor.exists() {
        cursor = cursor
            .parent()
            .ok_or_else(|| format!("no existing ancestor for {}", path.display()))?;
    }
    Ok(cursor.to_path_buf())
}

pub(super) fn undo_sidecar(path: &Path) -> PathBuf {
    let stem = path
        .file_stem()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "memory".into());
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .filter(|ext| !ext.is_empty())
        .unwrap_or("md");
    path.with_file_name(format!("{stem}.undo.{ext}"))
}

/// Restore and consume the newest project/global undo record transactionally.
pub fn undo_memory(workspace: &Path) -> Result<String, String> {
    let project = memory_file_at(workspace);
    let global = global_memory_file();
    let candidates = [undo_sidecar(&project), undo_sidecar(&global)];
    let Some(sidecar) = candidates
        .iter()
        .filter(|path| {
            fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_file())
        })
        .max_by_key(|path| {
            fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .ok()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        })
    else {
        return Err("nothing to undo — no recent memory write in this session".into());
    };
    let target = if sidecar == &undo_sidecar(&project) {
        project
    } else {
        global
    };
    let previous = read_regular_file_no_follow(sidecar, MAX_MEMORY_PREIMAGE_BYTES)
        .map_err(|error| format!("couldn't read undo snapshot: {error}"))?;
    let root = existing_transaction_root(&target)?;
    let mut mutations = Vec::with_capacity(2);
    if previous.is_empty() {
        match fs::symlink_metadata(&target) {
            Ok(_) => mutations.push(hi_tools::PlannedFileMutation::delete(&target)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("couldn't inspect {}: {error}", target.display())),
        }
    } else {
        mutations.push(hi_tools::PlannedFileMutation::write(
            &target,
            previous.clone(),
        ));
    }
    mutations.push(hi_tools::PlannedFileMutation::delete_from_preimage(
        sidecar, &previous,
    ));
    hi_tools::MutationPlan::new(&root, mutations)
        .and_then(hi_tools::MutationPlan::commit)
        .map_err(|error| format!("couldn't restore {}: {error:#}", target.display()))?;
    Ok(format!("restored {}", target.display()))
}
