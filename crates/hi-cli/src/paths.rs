//! Session and data-directory paths (no old-agent types).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

/// Constrain session IDs to one safe filename segment.
pub fn validate_session_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > 128
        || matches!(id, "." | "..")
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        anyhow::bail!("invalid session id: use 1-128 ASCII letters, digits, '.', '_' or '-'");
    }
    Ok(())
}

pub fn data_root() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
        .map(|p| p.join("hi"))
}

pub fn cwd_digest() -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    path_digest(&cwd)
}

pub fn path_digest(path: &Path) -> String {
    let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key.as_os_str().as_encoded_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

pub fn project_dir(digest: &str) -> Option<PathBuf> {
    data_root().map(|root| root.join("projects").join(digest))
}

pub fn write_workspace_sidecar(workspace: &Path) {
    let Some(root) = data_root() else {
        return;
    };
    let digest = path_digest(workspace);
    let path = root.join("projects").join(digest).join("workspace");
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let canonical = fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
    let _ = fs::write(path, format!("{}\n", canonical.display()));
}

pub fn read_workspace_sidecar(digest: &str) -> Option<PathBuf> {
    let path = project_dir(digest)?.join("workspace");
    let text = fs::read_to_string(path).ok()?;
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    Some(PathBuf::from(text))
}

pub fn project_digest_from_session(path: &Path) -> Option<String> {
    let mut dir = path.parent()?;
    if dir.file_name()?.to_str()? != "sessions" {
        return None;
    }
    dir = dir.parent()?;
    if dir.file_name()?.to_str()? == "dashboard" {
        dir = dir.parent()?;
    }
    let digest = dir.file_name()?.to_str()?.to_string();
    let projects = dir.parent()?;
    (projects.file_name()?.to_str()? == "projects").then_some(digest)
}

/// If this JSONL lives under another project digest, bind that workspace.
pub fn bind_workspace_for_session(
    session: Option<&Path>,
    workspace_root: PathBuf,
    state_root: PathBuf,
) -> (PathBuf, PathBuf) {
    write_workspace_sidecar(&workspace_root);
    let Some(session) = session else {
        return (workspace_root, state_root);
    };
    let Some(digest) = project_digest_from_session(session) else {
        return (workspace_root, state_root);
    };
    if digest == cwd_digest() {
        return (workspace_root, state_root);
    }
    match read_workspace_sidecar(&digest) {
        Some(path) if path.is_dir() => {
            eprintln!(
                "resuming session in {} (bound from workspace sidecar)",
                path.display()
            );
            let state = data_root()
                .map(|root| root.join("projects").join(&digest).join("runtime"))
                .unwrap_or_else(|| path.join(".hi/state"));
            let _ = fs::create_dir_all(&state);
            let state = fs::canonicalize(&state).unwrap_or(state);
            let path = fs::canonicalize(&path).unwrap_or(path);
            (path, state)
        }
        Some(path) => {
            eprintln!(
                "warning: workspace sidecar {} is missing; using current directory",
                path.display()
            );
            (workspace_root, state_root)
        }
        None => {
            eprintln!("warning: no workspace sidecar for digest {digest}; using current directory");
            (workspace_root, state_root)
        }
    }
}

pub fn latest_session() -> Option<PathBuf> {
    let dir = sessions_dir()?;
    fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
        .max_by_key(|p| {
            fs::metadata(p)
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH)
        })
}

pub fn sessions_dir() -> Option<PathBuf> {
    let base = data_root()?;
    let digest = cwd_digest();
    Some(base.join("projects").join(digest).join("sessions"))
}

pub fn machine_id() -> Option<String> {
    if let Some(id) = std::env::var_os("HI_SYNC_MACHINE_ID")
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.trim().is_empty())
    {
        return Some(id);
    }
    let root = data_root()?;
    let path = root.join("machine-id");
    if let Ok(id) = std::fs::read_to_string(&path) {
        let id = id.trim().to_string();
        if !id.is_empty() {
            return Some(id);
        }
    }
    let id = format!(
        "{:016x}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    );
    if std::fs::create_dir_all(&root).is_ok() && std::fs::write(&path, &id).is_ok() {
        Some(id)
    } else {
        None
    }
}

pub fn history_path() -> Option<PathBuf> {
    sessions_dir().and_then(|d| d.parent().map(|p| p.join("history")))
}

pub fn new_session_path() -> Result<PathBuf> {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = sessions_dir().context("could not determine session directory")?;
    if let Ok(cwd) = std::env::current_dir() {
        write_workspace_sidecar(&cwd);
    }
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let machine = machine_id().unwrap_or_else(|| format!("{:x}", std::process::id()));
    let mut machine_hash: u64 = 0xcbf29ce484222325;
    for byte in machine.bytes() {
        machine_hash ^= byte as u64;
        machine_hash = machine_hash.wrapping_mul(0x100000001b3);
    }
    let suffix = format!("{:08x}", machine_hash as u32);
    let process = std::process::id();
    let sequence = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(dir.join(format!(
        "{millis:013}-{suffix}-{process:x}-{sequence:x}.jsonl"
    )))
}

pub fn session_path(id: &str) -> Result<PathBuf> {
    validate_session_id(id)?;
    let name = if id.ends_with(".jsonl") {
        id.to_string()
    } else {
        format!("{id}.jsonl")
    };
    if let Some(dir) = sessions_dir() {
        let local = dir.join(&name);
        if local.exists() {
            return Ok(local);
        }
    }
    if let Some(root) = data_root() {
        let projects = root.join("projects");
        if let Ok(read) = fs::read_dir(&projects) {
            for entry in read.flatten() {
                let candidate = entry.path().join("sessions").join(&name);
                if candidate.exists() {
                    return Ok(candidate);
                }
                let dashboard = entry.path().join("dashboard").join("sessions").join(&name);
                if dashboard.exists() {
                    return Ok(dashboard);
                }
            }
        }
    }
    let dir = sessions_dir().context("could not determine session directory")?;
    Ok(dir.join(name))
}

#[derive(Clone, Debug)]
pub struct SessionSummary {
    pub id: String,
    pub age: String,
    pub title: String,
    pub flags: String,
    pub needs_attention: bool,
    pub dashboard: bool,
}

pub fn session_summaries() -> Vec<SessionSummary> {
    session_summaries_filtered(false)
}

fn session_summaries_filtered(needs_attention_only: bool) -> Vec<SessionSummary> {
    let Some(root) = data_root() else {
        return Vec::new();
    };
    crate::roster::scan_sessions(&root)
        .into_iter()
        .filter(|entry| !needs_attention_only || entry.needs_attention())
        .map(|entry| {
            let flags = entry.reason();
            let needs_attention = entry.needs_attention();
            SessionSummary {
                id: entry.id,
                age: entry.age,
                title: entry.title,
                flags,
                needs_attention,
                dashboard: entry.dashboard,
            }
        })
        .collect()
}

pub fn list_sessions() -> Result<()> {
    crate::roster::print_roster(false)
}

pub fn resolve_runtime_roots() -> Result<(PathBuf, PathBuf)> {
    let workspace_root = std::env::current_dir()
        .context("determining workspace root")?
        .canonicalize()
        .context("canonicalizing workspace root")?;
    anyhow::ensure!(
        workspace_root.is_dir(),
        "workspace root is not a directory: {}",
        workspace_root.display()
    );
    let state_root = data_root()
        .map(|root| root.join("projects").join(cwd_digest()).join("runtime"))
        .unwrap_or_else(|| workspace_root.join(".hi/state"));
    std::fs::create_dir_all(&state_root)
        .with_context(|| format!("creating workspace state root {}", state_root.display()))?;
    let state_root = state_root.canonicalize().with_context(|| {
        format!(
            "canonicalizing workspace state root {}",
            state_root.display()
        )
    })?;
    write_workspace_sidecar(&workspace_root);
    Ok((workspace_root, state_root))
}

pub fn absolutize_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(std::env::current_dir()
        .context("determining current directory")?
        .join(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_path_digest_walks_dashboard_and_plain() {
        let jsonl = PathBuf::from(
            "/Users/david/.local/share/hi/projects/abcdabcdabcdabcd/sessions/id.jsonl",
        );
        assert_eq!(
            project_digest_from_session(&jsonl).as_deref(),
            Some("abcdabcdabcdabcd")
        );
        let dash = PathBuf::from(
            "/Users/david/.local/share/hi/projects/abcdabcdabcdabcd/dashboard/sessions/id.jsonl",
        );
        assert_eq!(
            project_digest_from_session(&dash).as_deref(),
            Some("abcdabcdabcdabcd")
        );
    }

    #[test]
    fn bind_workspace_uses_sidecar_for_foreign_digest() {
        let _cwd = crate::CWD_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let prev_cwd = std::env::current_dir().unwrap();
        let prev_xdg = std::env::var_os("XDG_DATA_HOME");
        std::env::set_current_dir(tmp.path()).unwrap();
        let xdg = tmp.path().join("xdg");
        unsafe {
            std::env::set_var("XDG_DATA_HOME", &xdg);
        }
        let other = tmp.path().join("other-ws");
        std::fs::create_dir_all(&other).unwrap();
        let digest = "deadbeefdeadbeef";
        let project = xdg.join("hi").join("projects").join(digest);
        std::fs::create_dir_all(project.join("sessions")).unwrap();
        std::fs::write(project.join("workspace"), format!("{}\n", other.display())).unwrap();
        let session = project.join("sessions").join("s.jsonl");
        std::fs::write(&session, "{}\n").unwrap();
        let cwd_ws = tmp.path().to_path_buf();
        let cwd_state = tmp.path().join("state");
        std::fs::create_dir_all(&cwd_state).unwrap();
        let (bound, _) = bind_workspace_for_session(Some(&session), cwd_ws, cwd_state);
        assert_eq!(bound, other.canonicalize().unwrap());
        std::env::set_current_dir(prev_cwd).unwrap();
        unsafe {
            match prev_xdg {
                Some(value) => std::env::set_var("XDG_DATA_HOME", value),
                None => std::env::remove_var("XDG_DATA_HOME"),
            }
        }
    }
}
