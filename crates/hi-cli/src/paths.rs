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
    let key = std::fs::canonicalize(&cwd).unwrap_or(cwd);
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key.as_os_str().as_encoded_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
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
}

pub fn session_summaries() -> Vec<SessionSummary> {
    let Some(root) = data_root() else {
        return Vec::new();
    };
    let projects = root.join("projects");
    let mut entries: Vec<(PathBuf, SystemTime)> = Vec::new();
    if let Ok(buckets) = fs::read_dir(&projects) {
        for bucket in buckets.flatten() {
            let sess_dir = bucket.path().join("sessions");
            let Ok(read) = fs::read_dir(&sess_dir) else {
                continue;
            };
            for entry in read.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "jsonl") {
                    let modified = fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .unwrap_or(UNIX_EPOCH);
                    entries.push((path, modified));
                }
            }
        }
    }
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));
    let now = SystemTime::now();
    entries
        .into_iter()
        .map(|(path, modified)| {
            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_string();
            let age = now
                .duration_since(modified)
                .map(|d| {
                    let secs = d.as_secs();
                    if secs < 60 {
                        format!("{secs}s")
                    } else if secs < 3600 {
                        format!("{}m", secs / 60)
                    } else if secs < 86400 {
                        format!("{}h", secs / 3600)
                    } else {
                        format!("{}d", secs / 86400)
                    }
                })
                .unwrap_or_else(|_| "?".into());
            SessionSummary { id, age }
        })
        .collect()
}

pub fn list_sessions() -> Result<()> {
    let summaries = session_summaries();
    if summaries.is_empty() {
        match data_root() {
            Some(root) => println!("no sessions in {}", root.join("projects").display()),
            None => println!("no session directory"),
        }
        return Ok(());
    }
    for summary in summaries {
        println!("{}  {:>6} ago", summary.id, summary.age);
    }
    Ok(())
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
