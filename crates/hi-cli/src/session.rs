//! JSONL session persistence: one message per line, appended after each turn.
//!
//! Sessions live under `$XDG_DATA_HOME/hi/sessions` (or `~/.local/share/...`).
//! Resuming loads every line back as conversation history. Branching/tree
//! sessions (pi-style) are a future extension; this is a linear log.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use hi_agent::SessionSink;
use hi_ai::{Message, Role, Usage};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SessionMeta {
    /// User-defined display name. Last write wins; an empty name restores the
    /// automatic first-prompt title.
    Name {
        name: String,
    },
    Usage {
        input_tokens: u64,
        output_tokens: u64,
        /// Cache counters and the estimated marker ride along so a resumed
        /// session's totals keep full fidelity. `#[serde(default)]` so session
        /// files written before these fields load as zero/false.
        #[serde(default)]
        cache_read_tokens: u64,
        #[serde(default)]
        cache_creation_tokens: u64,
        #[serde(default)]
        estimated: bool,
    },
    Checkpoints {
        refs: Vec<String>,
    },
    /// A compaction boundary: all messages before this line are superseded by
    /// the compacted messages stored here. On resume, replace prior messages
    /// with these so the compaction survives across sessions.
    Compaction {
        messages: Vec<Message>,
    },
    /// A long-horizon goal's authoritative state, so a `/resume` picks up the
    /// in-progress goal at its active sub-goal. Last write wins (the goal is
    /// replaced wholesale, like `Compaction`).
    Goal {
        goal: hi_agent::Goal,
    },
    /// The long-horizon goal was explicitly cleared. Last write wins.
    GoalCleared,
    /// The intra-session decision log. Last write wins.
    Decisions {
        decisions: Vec<hi_agent::Decision>,
    },
    /// An explicit replacement of all retry-relevant state. This keeps
    /// transcript, structured goal, and decisions in sync when a turn is
    /// discarded by `/retry` or interrupt cleanup.
    StateReplacement {
        messages: Vec<Message>,
        #[serde(default)]
        goal: Option<hi_agent::Goal>,
        #[serde(default)]
        decisions: Vec<hi_agent::Decision>,
    },
}

/// Appends messages to a session's JSONL file.
pub struct JsonlSession {
    path: PathBuf,
}

impl JsonlSession {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Append a fully-formatted payload (one or more `\n`-terminated JSONL
    /// lines) with a single `write_all` on the `O_APPEND` fd. A buffered
    /// writer would split records larger than its buffer across multiple
    /// `write()` calls, letting a concurrent appender (a second `hi -c` in the
    /// same project, or a fleet child on `--session-file`) interleave mid-line
    /// — and `load_history` silently drops unparseable lines.
    fn append(&self, payload: &str) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        file.write_all(payload.as_bytes())
            .with_context(|| format!("appending to {}", self.path.display()))?;
        Ok(())
    }

    fn append_meta(&self, meta: &SessionMeta) -> Result<()> {
        let mut line = serde_json::to_string(meta)?;
        line.push('\n');
        self.append(&line)
    }

    /// Persist checkpoint refs so a resumed session knows where it branched.
    #[allow(dead_code)]
    pub fn record_checkpoints(&mut self, refs: &[String]) -> Result<()> {
        if refs.is_empty() {
            return Ok(());
        }
        self.append_meta(&SessionMeta::Checkpoints {
            refs: refs.to_vec(),
        })
    }
}

impl SessionSink for JsonlSession {
    fn record_checkpoints(&mut self, refs: &[String]) -> Result<()> {
        JsonlSession::record_checkpoints(self, refs)
    }

    fn record(&mut self, messages: &[Message], usage: Usage) -> Result<()> {
        if messages.is_empty() && usage.is_zero() {
            return Ok(());
        }
        let mut payload = String::new();
        for message in messages {
            payload.push_str(&serde_json::to_string(message)?);
            payload.push('\n');
        }
        payload.push_str(&serde_json::to_string(&SessionMeta::Usage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_read_tokens: usage.cache_read_tokens,
            cache_creation_tokens: usage.cache_creation_tokens,
            estimated: usage.estimated,
        })?);
        payload.push('\n');
        self.append(&payload)
    }

    fn record_compaction(&mut self, messages: &[Message]) -> Result<()> {
        self.append_meta(&SessionMeta::Compaction {
            messages: messages.to_vec(),
        })
    }

    fn record_goal(&mut self, goal: &hi_agent::Goal) -> Result<()> {
        self.append_meta(&SessionMeta::Goal { goal: goal.clone() })
    }

    fn clear_goal(&mut self) -> Result<()> {
        self.append_meta(&SessionMeta::GoalCleared)
    }

    fn record_decisions(&mut self, decisions: &hi_agent::DecisionLog) -> Result<()> {
        self.append_meta(&SessionMeta::Decisions {
            decisions: decisions.entries().to_vec(),
        })
    }

    fn record_state_replacement(
        &mut self,
        messages: &[Message],
        goal: Option<&hi_agent::Goal>,
        decisions: &hi_agent::DecisionLog,
    ) -> Result<()> {
        self.append_meta(&SessionMeta::StateReplacement {
            messages: messages.to_vec(),
            goal: goal.cloned(),
            decisions: decisions.entries().to_vec(),
        })
    }
}

#[allow(dead_code)]
pub struct LoadedSession {
    pub messages: Vec<Message>,
    pub usage: Usage,
    pub checkpoint_refs: Vec<String>,
    /// User-defined display name, if one has been assigned (last write wins).
    pub name: Option<String>,
    /// A long-horizon goal persisted across sessions, if any (last write wins).
    pub goal: Option<hi_agent::Goal>,
    /// Intra-session decisions persisted across resume (last write wins).
    pub decisions: hi_agent::DecisionLog,
}

/// Atomically cache reconstructed session state at `path`. A failed restore
/// never leaves a partial JSONL file that could later masquerade as a complete
/// session.
pub fn cache_loaded_session(path: &Path, loaded: &LoadedSession) -> Result<()> {
    static RESTORE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let restore_id = RESTORE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temp = path.with_extension(format!("restoring-{}-{restore_id}", std::process::id()));
    let result = (|| {
        let mut session = JsonlSession::new(temp.clone());
        session.record_state_replacement(
            &loaded.messages,
            loaded.goal.as_ref(),
            &loaded.decisions,
        )?;
        session.record(&[], loaded.usage)?;
        session.record_checkpoints(&loaded.checkpoint_refs)?;
        if let Some(name) = &loaded.name {
            session.append_meta(&SessionMeta::Name { name: name.clone() })?;
        }
        fs::rename(&temp, path)
            .with_context(|| format!("installing restored session {}", path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

/// One-line summary shown when a session is resumed: message count and
/// the last user instruction (clipped), so the user knows what they're walking
/// back into.
pub fn resume_summary(loaded: &LoadedSession) -> String {
    let n = loaded
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .count();
    let last = loaded
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| hi_agent::ui::clip(&m.text(), 60))
        .unwrap_or_default();
    format!("Resumed: {n} messages, last: '{last}'")
}

/// Directory holding all session files (may not exist yet).
///
/// Sessions are namespaced by the current working directory so that `hi -c`
/// and `--list-sessions` only see chats started in *this* project — the
/// history is no longer global. The namespace key is a short FNV-1a digest of
/// the canonical cwd; it lives under the same `$XDG_DATA_HOME/hi` (or
/// `~/.local/share/hi`) root, in a `projects/<digest>/sessions/` subtree.
pub fn sessions_dir() -> Option<PathBuf> {
    let base = data_root()?;
    let digest = cwd_digest();
    Some(base.join("projects").join(digest).join("sessions"))
}

/// The shared data root (`$XDG_DATA_HOME/hi` or `~/.local/share/hi`).
pub(crate) fn data_root() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
        .map(|p| p.join("hi"))
}

/// A persistent per-install machine identifier. Stored at
/// `$XDG_DATA_HOME/hi/machine-id` (generated on first run, reused thereafter).
/// Used as the `machine_id` in sync config so a remote viewer knows which
/// machine is hosting a session. Falls back to `HI_SYNC_MACHINE_ID` env var
/// if set (for explicit override), or `None` if the data dir isn't writable.
pub fn machine_id() -> Option<String> {
    // Explicit env override takes precedence.
    if let Some(id) = std::env::var_os("HI_SYNC_MACHINE_ID")
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.trim().is_empty())
    {
        return Some(id);
    }

    let root = data_root()?;
    let path = root.join("machine-id");

    // Try to read the existing ID.
    if let Ok(id) = std::fs::read_to_string(&path) {
        let id = id.trim().to_string();
        if !id.is_empty() {
            return Some(id);
        }
    }

    // Generate a new ID and persist it.
    let id = format!(
        "{:016x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    );
    if std::fs::create_dir_all(&root).is_ok() && std::fs::write(&path, &id).is_ok() {
        Some(id)
    } else {
        None
    }
}

/// A short, stable, filesystem-safe key for the current working directory.
/// Uses FNV-1a over the canonicalized path (resolves symlinks, so a project
/// reached via different paths still maps to one bucket). Falls back to the
/// raw cwd if canonicalization fails. Sixteen hex chars is enough to avoid
/// collisions across any realistic number of project dirs while keeping the
/// directory listing readable.
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

/// Account-wide opaque project identity. Remote URLs are normalized locally
/// and only their SHA-256 digest is sent to the portal. Repositories without a
/// remote deliberately include the machine id and remain machine-specific.
pub fn project_fingerprint() -> Option<String> {
    use sha2::{Digest, Sha256};
    let cwd = std::env::current_dir().ok()?;
    let top = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(&cwd)
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    let top = PathBuf::from(String::from_utf8(top.stdout).ok()?.trim());
    let relative = cwd
        .strip_prefix(&top)
        .unwrap_or(Path::new(""))
        .to_string_lossy()
        .replace('\\', "/");
    let remote = std::process::Command::new("git")
        .args(["config", "--get", "remote.origin.url"])
        .current_dir(&top)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let identity = if let Some(remote) = remote {
        normalize_git_remote(&remote).unwrap_or(remote)
    } else {
        format!(
            "local:{}/{}",
            machine_id().unwrap_or_else(|| "unknown".to_string()),
            std::fs::canonicalize(&top).unwrap_or(top).to_string_lossy()
        )
    };
    Some(format!(
        "{:x}",
        Sha256::digest(format!("{}\0{}", identity, relative).as_bytes())
    ))
}

fn normalize_git_remote(remote: &str) -> Option<String> {
    let value = remote.trim().trim_end_matches('/').trim_end_matches(".git");
    let host_path = if let Some(rest) = value.split_once("://").map(|(_, rest)| rest) {
        let rest = rest.rsplit_once('@').map(|(_, rest)| rest).unwrap_or(rest);
        rest.split_once('/').map(|(host, path)| (host, path))?
    } else if let Some((user_host, path)) = value.split_once(':') {
        let host = user_host
            .rsplit_once('@')
            .map(|(_, host)| host)
            .unwrap_or(user_host);
        (host, path)
    } else {
        return None;
    };
    Some(format!(
        "{}/{}",
        host_path.0.to_ascii_lowercase(),
        host_path.1.trim_matches('/').to_ascii_lowercase()
    ))
}

/// Path to the persistent REPL input-history file. Per-directory (lives inside
/// `sessions_dir()`) so Up-arrow history is scoped to the current project.
pub fn history_path() -> Option<PathBuf> {
    sessions_dir().and_then(|d| d.parent().map(|p| p.join("history")))
}

/// Path for a brand-new session. The millisecond prefix keeps listings
/// sortable; machine/process/counter suffixes prevent two concurrent clients
/// from sharing a file or merging distinct portal sessions under one ID.
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

/// A resumable fleet session (for the `/fleet status` view).
pub struct FleetSessionInfo {
    /// The resume id (file stem, e.g. `1783605123456-f0`).
    pub id: String,
    /// First user prompt, cleaned (the row's dispatch text).
    pub title: String,
    /// Humanized age ("3m ago", "2h ago").
    pub age: String,
    /// Session length in lines (rough size signal).
    pub lines: usize,
}

/// The current project's fleet sessions (dispatched from `/dashboard`), newest
/// first. Fleet sessions are recognizable by the `-f<n>` stem suffix.
pub fn fleet_sessions() -> Vec<FleetSessionInfo> {
    sessions_dir()
        .map(|dir| fleet_sessions_in(&dir))
        .unwrap_or_default()
}

/// List all sessions cached for the current project (not just fleet sessions).
/// The TUI merges these with the synced catalog for `/sessions`.
pub fn local_sessions() -> Vec<FleetSessionInfo> {
    let Some(dir) = sessions_dir() else {
        return Vec::new();
    };
    let Ok(read) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut entries: Vec<(PathBuf, SystemTime)> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
        .map(|p| {
            let modified = fs::metadata(&p)
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH);
            (p, modified)
        })
        .collect();
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
            let title = session_display_name(&path);
            let age = now
                .duration_since(modified)
                .map(|d| humanize(d.as_secs()))
                .unwrap_or_else(|_| "?".into());
            let lines = fs::read_to_string(&path)
                .map(|t| t.lines().count())
                .unwrap_or(0);
            FleetSessionInfo {
                id,
                title,
                age,
                lines,
            }
        })
        .collect()
}

fn fleet_sessions_in(dir: &Path) -> Vec<FleetSessionInfo> {
    let Ok(read) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut entries: Vec<(PathBuf, SystemTime)> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "jsonl")
                && p.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(is_fleet_stem)
        })
        .map(|p| {
            let modified = fs::metadata(&p)
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH);
            (p, modified)
        })
        .collect();
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
            let title = session_display_name(&path);
            let age = now
                .duration_since(modified)
                .map(|d| humanize(d.as_secs()))
                .unwrap_or_else(|_| "?".into());
            let lines = fs::read_to_string(&path)
                .map(|t| t.lines().count())
                .unwrap_or(0);
            FleetSessionInfo {
                id,
                title,
                age,
                lines,
            }
        })
        .collect()
}

/// Whether a session file stem names a fleet session: `<millis>-f<n>`.
fn is_fleet_stem(stem: &str) -> bool {
    stem.rsplit_once("-f")
        .is_some_and(|(_, n)| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
}

/// Whether a session file stem names a `/loop` session: `<millis>-loop<n>`.
fn is_loop_stem(stem: &str) -> bool {
    stem.rsplit_once("-loop")
        .is_some_and(|(_, n)| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
}

/// A session's persisted long-horizon goal state, summarized for the fleet:
/// whether it should still auto-drive, and its progress.
pub struct SessionGoalSummary {
    pub active: bool,
    pub done: usize,
    pub total: usize,
}

/// Read the last-written goal state from a session file (`goal` /
/// `goal_cleared` / `state_replacement` meta lines, last-wins — mirroring the
/// resume loader) without loading the whole conversation.
pub fn session_goal_summary(path: &Path) -> Option<SessionGoalSummary> {
    let text = fs::read_to_string(path).ok()?;
    let mut goal: Option<hi_agent::Goal> = None;
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match value.get("type").and_then(|t| t.as_str()) {
            Some("goal") => {
                if let Some(g) = value
                    .get("goal")
                    .and_then(|g| serde_json::from_value(g.clone()).ok())
                {
                    goal = Some(g);
                }
            }
            Some("goal_cleared") => goal = None,
            Some("state_replacement") => {
                goal = value
                    .get("goal")
                    .filter(|g| !g.is_null())
                    .and_then(|g| serde_json::from_value(g.clone()).ok());
            }
            _ => {}
        }
    }
    goal.map(|g| SessionGoalSummary {
        active: g.should_auto_drive(),
        done: g
            .sub_goals
            .iter()
            .filter(|s| s.status == hi_agent::GoalStatus::Done)
            .count(),
        total: g.sub_goals.len(),
    })
}

/// Path for a `/loop` session (each firing resumes it). `-loop<n>` stems keep
/// these out of `/fleet status` while staying resumable by id.
pub fn new_loop_session_path() -> Result<PathBuf> {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = sessions_dir().context("could not determine session directory")?;
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(dir.join(format!("{millis:013}-loop{n}.jsonl")))
}

/// The per-project `/loop` definitions file (sibling of the sessions dir).
pub fn loops_file() -> Option<PathBuf> {
    sessions_dir().and_then(|d| d.parent().map(|p| p.join("loops.json")))
}

/// Path for a fleet-dispatched session. Unlike [`new_session_path`] (millis
/// only), several fleet agents can be dispatched within the same millisecond,
/// so a per-process counter suffix keeps the paths (and resume ids) unique
/// while staying time-sortable.
pub fn new_fleet_session_path() -> Result<PathBuf> {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = sessions_dir().context("could not determine session directory")?;
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(dir.join(format!("{millis:013}-f{n}.jsonl")))
}

/// Path for an explicit session id (with or without the `.jsonl` suffix).
///
/// Looks in the current project's session dir first. If the id isn't found
/// there, falls back to a search across *all* project buckets under the data
/// root — so `--resume <id>` keeps working for a session started in a
/// different directory (e.g. an id copied from a `--list-sessions` run
/// elsewhere, or resuming after `cd`-ing to another project).
pub fn session_path(id: &str) -> Result<PathBuf> {
    let name = if id.ends_with(".jsonl") {
        id.to_string()
    } else {
        format!("{id}.jsonl")
    };
    // Current project bucket first.
    if let Some(dir) = sessions_dir() {
        let local = dir.join(&name);
        if local.exists() {
            return Ok(local);
        }
    }
    // Global fallback: scan every project bucket for a matching file name.
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
    // Nothing found — return the current-project path so the caller gets a
    // sensible "no such session" error rather than a panic.
    let dir = sessions_dir().context("could not determine session directory")?;
    Ok(dir.join(name))
}

/// Persist a user-defined display name for a session. Appending metadata keeps
/// the JSONL log backward-compatible and makes rename atomic with concurrent
/// turn appends; readers use the last name record.
pub fn rename_session(id: &str, name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        anyhow::bail!("session name cannot be empty");
    }
    if name.chars().count() > 120 {
        anyhow::bail!("session name must be at most 120 characters");
    }
    let path = session_path(id)?;
    if !path.is_file() {
        anyhow::bail!("no saved session '{id}'");
    }
    let session = JsonlSession::new(path);
    session.append_meta(&SessionMeta::Name {
        name: name.to_string(),
    })?;
    Ok(name.to_string())
}

fn session_display_name(path: &Path) -> String {
    custom_session_name(path)
        .or_else(|| first_user_message(path).map(|message| session_title(&message)))
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| "(no prompt yet)".to_string())
}

fn custom_session_name(path: &Path) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    let mut name = None;
    for line in text.lines() {
        if let Ok(SessionMeta::Name { name: next }) = serde_json::from_str::<SessionMeta>(line) {
            name = (!next.trim().is_empty()).then(|| next.trim().to_string());
        }
    }
    name
}

/// The most recently modified *user* session, if any. Fleet (`-f<n>`) and loop
/// (`-loop<n>`) sessions are excluded so `hi -c` resumes the user's own last
/// chat, not a background fleet child or a `/loop` firing — the latter rewrites
/// its session on every interval and would otherwise always win the mtime race,
/// making `-c` never reach the user's real session again.
pub fn latest_session() -> Option<PathBuf> {
    let dir = sessions_dir()?;
    fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
        .filter(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .is_none_or(|stem| !is_fleet_stem(stem) && !is_loop_stem(stem))
        })
        .max_by_key(|p| {
            fs::metadata(p)
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH)
        })
}

/// Load a session's messages back into conversation history.
pub fn load_history(path: &Path) -> Result<LoadedSession> {
    let text =
        fs::read_to_string(path).with_context(|| format!("reading session {}", path.display()))?;
    let mut messages = Vec::new();
    let mut usage = Usage::default();
    let mut checkpoint_refs = Vec::new();
    let mut loaded_goal: Option<hi_agent::Goal> = None;
    let mut loaded_decisions = hi_agent::DecisionLog::default();
    let mut loaded_name = None;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(meta) = serde_json::from_str::<SessionMeta>(line) {
            match meta {
                SessionMeta::Name { name } => {
                    loaded_name = (!name.trim().is_empty()).then(|| name.trim().to_string());
                }
                SessionMeta::Usage {
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                    estimated,
                } => {
                    usage = Usage {
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_creation_tokens,
                        input_includes_cache: false,
                        context_occupancy: input_tokens,
                        rate_limits: None,
                        estimated,
                    };
                }
                SessionMeta::Checkpoints { refs } => {
                    checkpoint_refs = refs;
                }
                SessionMeta::Compaction {
                    messages: compacted,
                } => {
                    // Replace all prior messages with the compacted set.
                    messages = compacted;
                }
                SessionMeta::Goal { goal } => {
                    loaded_goal = Some(goal);
                }
                SessionMeta::GoalCleared => {
                    loaded_goal = None;
                }
                SessionMeta::Decisions { decisions } => {
                    loaded_decisions = hi_agent::DecisionLog::from_entries(decisions);
                }
                SessionMeta::StateReplacement {
                    messages: replacement,
                    goal,
                    decisions,
                } => {
                    messages = replacement;
                    loaded_goal = goal;
                    loaded_decisions = hi_agent::DecisionLog::from_entries(decisions);
                }
            }
            continue;
        }
        let message: Message = match serde_json::from_str(line) {
            Ok(m) => m,
            Err(_) => {
                // Skip a corrupted/unparseable line (e.g. a partially-written
                // last line from a crash mid-flush) rather than failing the
                // entire resume. The session is still usable up to the last
                // complete line; a truncated final line carries no real content.
                continue;
            }
        };
        messages.push(message);
    }
    Ok(LoadedSession {
        messages,
        usage,
        checkpoint_refs,
        name: loaded_name,
        goal: loaded_goal,
        decisions: loaded_decisions,
    })
}

/// A remote session record: `(record_type, payload_json)`, as fetched from
/// ipop's `GET /v1/hi/sessions/{id}/records` endpoint.
pub struct RemoteRecord {
    pub record_type: String,
    pub payload_json: String,
}

/// Load a session from remote records (fetched from ipop) instead of a local
/// JSONL file. Applies the same parsing logic as [`load_history`]: bare
/// `message` records are conversation history; tagged metadata records
/// (`usage`, `compaction`, `goal`, etc.) update the session state.
///
/// This lets `hi --attach --resume-local` boot a local agent from the remote
/// session history when the daemon is down.
pub fn load_history_from_records(records: &[RemoteRecord]) -> Result<LoadedSession> {
    let mut messages = Vec::new();
    let mut usage = Usage::default();
    let mut checkpoint_refs = Vec::new();
    let mut loaded_goal: Option<hi_agent::Goal> = None;
    let mut loaded_decisions = hi_agent::DecisionLog::default();
    let mut loaded_name = None;

    for record in records {
        if record.record_type == "message" {
            if let Ok(message) = serde_json::from_str::<Message>(&record.payload_json) {
                messages.push(message);
            }
            continue;
        }
        // Metadata records: parse the payload as a SessionMeta.
        if let Ok(meta) = serde_json::from_str::<SessionMeta>(&record.payload_json) {
            match meta {
                SessionMeta::Name { name } => {
                    loaded_name = (!name.trim().is_empty()).then(|| name.trim().to_string());
                }
                SessionMeta::Usage {
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                    estimated,
                } => {
                    usage = Usage {
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_creation_tokens,
                        input_includes_cache: false,
                        context_occupancy: input_tokens,
                        rate_limits: None,
                        estimated,
                    };
                }
                SessionMeta::Checkpoints { refs } => {
                    checkpoint_refs = refs;
                }
                SessionMeta::Compaction {
                    messages: compacted,
                } => {
                    messages = compacted;
                }
                SessionMeta::Goal { goal } => {
                    loaded_goal = Some(goal);
                }
                SessionMeta::GoalCleared => {
                    loaded_goal = None;
                }
                SessionMeta::Decisions { decisions } => {
                    loaded_decisions = hi_agent::DecisionLog::from_entries(decisions);
                }
                SessionMeta::StateReplacement {
                    messages: replacement,
                    goal,
                    decisions,
                } => {
                    messages = replacement;
                    loaded_goal = goal;
                    loaded_decisions = hi_agent::DecisionLog::from_entries(decisions);
                }
            }
        }
    }

    Ok(LoadedSession {
        messages,
        usage,
        checkpoint_refs,
        name: loaded_name,
        goal: loaded_goal,
        decisions: loaded_decisions,
    })
}
///
/// Walks every project bucket under the data root (sessions are namespaced
/// per-directory) and lists them newest-first, annotating each with a short
/// project-digest prefix so you can tell which directory a session belongs to.
pub fn list_sessions() -> Result<()> {
    let Some(root) = data_root() else {
        println!("no session directory");
        return Ok(());
    };
    let projects = root.join("projects");

    // Collect (path, modified, project_digest) across all project buckets.
    let mut entries: Vec<(PathBuf, SystemTime, String)> = Vec::new();
    if let Ok(buckets) = fs::read_dir(&projects) {
        for bucket in buckets.flatten() {
            let digest = bucket.file_name().to_str().unwrap_or("?").to_string();
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
                    entries.push((path, modified, digest.clone()));
                }
            }
        }
    }

    if entries.is_empty() {
        println!("no sessions in {}", projects.display());
        return Ok(());
    }
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));

    let now = SystemTime::now();
    for (path, modified, digest) in entries {
        let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let age = now
            .duration_since(modified)
            .map(|d| humanize(d.as_secs()))
            .unwrap_or_else(|_| "?".into());
        let title = session_display_name(&path);
        // Short 8-char project prefix so the column stays narrow but remains
        // enough to disambiguate sessions from different directories.
        let proj = &digest[..digest.len().min(8)];
        println!(
            "{id}  {age:>6} ago  {proj}  {}",
            hi_agent::ui::clip(&title, 60)
        );
    }
    Ok(())
}

/// Derive a concise, single-line title from a session's first user message:
/// drop any folded stdin/code block (a piped-in `hi "fix this" < log` lands as a
/// fenced `stdin:` section) and collapse whitespace, so the listing shows the
/// human instruction rather than a wall of pasted output. Deterministic — no
/// model call, unlike minion's generated titles.
fn session_title(first_user: &str) -> String {
    let head = first_user
        .split("stdin:")
        .next()
        .unwrap_or(first_user)
        .split("```")
        .next()
        .unwrap_or(first_user);
    head.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn first_user_message(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    use std::io::BufRead;
    for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
        if let Ok(message) = serde_json::from_str::<Message>(&line)
            && message.role == Role::User
        {
            return Some(message.text());
        }
    }
    None
}

fn humanize(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m", s / 60),
        s if s < 86_400 => format!("{}h", s / 3600),
        s => format!("{}d", s / 86_400),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        JsonlSession, LoadedSession, SessionMeta, cache_loaded_session, cwd_digest, load_history,
        machine_id, session_display_name, session_title,
    };
    use hi_agent::SessionSink;
    use hi_ai::{Message, Usage};

    #[test]
    fn fleet_session_paths_are_unique_within_a_burst() {
        // Dispatching several fleet agents in one millisecond must still yield
        // distinct files (counter suffix).
        let paths: Vec<_> = (0..10)
            .map(|_| super::new_fleet_session_path().expect("path"))
            .collect();
        let unique: std::collections::HashSet<_> = paths.iter().collect();
        assert_eq!(unique.len(), paths.len(), "collision in {paths:?}");
    }

    #[test]
    fn user_session_paths_are_unique_and_safe_within_a_burst() {
        let paths = (0..10)
            .map(|_| super::new_session_path().expect("path"))
            .collect::<Vec<_>>();
        let ids = paths
            .iter()
            .map(|path| path.file_stem().unwrap().to_string_lossy().to_string())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(ids.len(), paths.len(), "collision in {paths:?}");
        assert!(
            ids.iter().all(|id| id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')
            }))
        );
    }

    #[test]
    fn restored_session_cache_round_trips_complete_state() {
        let path = std::env::temp_dir().join(format!(
            "hi-session-restore-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let expected = LoadedSession {
            messages: vec![Message::user("restored prompt")],
            usage: Usage {
                input_tokens: 12,
                output_tokens: 4,
                ..Usage::default()
            },
            checkpoint_refs: vec!["checkpoint-1".into()],
            name: Some("Restored session".into()),
            goal: None,
            decisions: hi_agent::DecisionLog::default(),
        };

        cache_loaded_session(&path, &expected).expect("cache restored session");
        let loaded = load_history(&path).expect("load restored cache");

        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].text(), "restored prompt");
        assert_eq!(loaded.usage.input_tokens, expected.usage.input_tokens);
        assert_eq!(loaded.usage.output_tokens, expected.usage.output_tokens);
        assert_eq!(loaded.checkpoint_refs, expected.checkpoint_refs);
        assert_eq!(loaded.name, expected.name);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn explicit_session_name_overrides_automatic_title_last_write_wins() {
        let path = std::env::temp_dir().join(format!(
            "hi-session-name-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        session
            .record(&[Message::user("automatic title")], Usage::default())
            .unwrap();
        session
            .append_meta(&SessionMeta::Name {
                name: "First name".into(),
            })
            .unwrap();
        session
            .append_meta(&SessionMeta::Name {
                name: "Renamed work".into(),
            })
            .unwrap();

        assert_eq!(session_display_name(&path), "Renamed work");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fleet_sessions_lists_only_fleet_stems_newest_first() {
        let dir = std::env::temp_dir().join(format!("hi-fleet-ls-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let user = |text: &str| serde_json::to_string(&Message::user(text)).unwrap();
        // A fleet session, an ordinary session, and junk.
        std::fs::write(
            dir.join("0000000000001-f0.jsonl"),
            user("fix the parser") + "\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("0000000000002.jsonl"),
            user("plain session") + "\n",
        )
        .unwrap();
        std::fs::write(dir.join("notes.txt"), "junk").unwrap();
        std::fs::write(
            dir.join("0000000000003-f11.jsonl"),
            user("port the cli") + "\n",
        )
        .unwrap();
        // Nudge mtimes so ordering is deterministic (f11 newer).
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        let f = std::fs::File::options()
            .append(true)
            .open(dir.join("0000000000001-f0.jsonl"))
            .unwrap();
        f.set_modified(old).unwrap();

        let list = super::fleet_sessions_in(&dir);
        let ids: Vec<&str> = list.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["0000000000003-f11", "0000000000001-f0"]);
        assert_eq!(list[0].title, "port the cli");
        assert_eq!(list[0].lines, 1);
        assert!(list[1].age.contains("ago") || !list[1].age.is_empty());
        // Stem filter specifics.
        assert!(super::is_fleet_stem("0000000000001-f0"));
        assert!(super::is_fleet_stem("0000000000001-f42"));
        assert!(!super::is_fleet_stem("0000000000002"));
        assert!(!super::is_fleet_stem("0000000000002-fx"));
        // Loop-stem filter (kept out of `hi -c`'s latest_session).
        assert!(super::is_loop_stem("0000000000001-loop0"));
        assert!(super::is_loop_stem("0000000000001-loop7"));
        assert!(!super::is_loop_stem("0000000000002"));
        assert!(!super::is_loop_stem("0000000000002-loopx"));
        assert!(!super::is_loop_stem("0000000000002-f3"));
        // A plain user-session stem is neither.
        assert!(!super::is_fleet_stem("0000000000002") && !super::is_loop_stem("0000000000002"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn title_strips_folded_stdin_and_collapses_whitespace() {
        assert_eq!(
            session_title("fix the   failing\n test"),
            "fix the failing test"
        );
        // Piped stdin is folded in as a fenced `stdin:` block — keep only the prose.
        assert_eq!(
            session_title("fix the failures\n\nstdin:\n```\nerror: boom\n```"),
            "fix the failures"
        );
        // A leading code fence is dropped too.
        assert_eq!(
            session_title("explain this\n```rust\nfn main() {}\n```"),
            "explain this"
        );
        assert_eq!(session_title("   "), "");
    }

    #[test]
    fn jsonl_session_round_trips_usage_metadata() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-session-usage-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        session
            .record(
                &[Message::system("sys"), Message::user("hello")],
                Usage {
                    input_tokens: 123,
                    output_tokens: 45,
                    context_occupancy: 123,
                    ..Usage::default()
                },
            )
            .unwrap();

        let loaded = load_history(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.usage.input_tokens, 123);
        assert_eq!(loaded.usage.output_tokens, 45);
    }

    #[test]
    fn jsonl_session_compaction_boundary_replaces_prior_messages_on_resume() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-session-clear-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        session
            .record(
                &[Message::system("sys-old"), Message::user("old context")],
                Usage::default(),
            )
            .unwrap();
        session
            .record_compaction(&[Message::system("sys-new")])
            .unwrap();
        session
            .record(&[Message::user("new context")], Usage::default())
            .unwrap();

        let loaded = load_history(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[0].text(), "sys-new");
        assert_eq!(loaded.messages[1].text(), "new context");
    }

    #[test]
    fn jsonl_session_round_trips_checkpoint_refs_last_write_wins() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-session-checkpoints-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());

        session
            .record_checkpoints(&["old".to_string(), "older".to_string()])
            .unwrap();
        session.record_checkpoints(&["new".to_string()]).unwrap();

        let loaded = load_history(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.checkpoint_refs, vec!["new".to_string()]);
    }

    #[test]
    fn jsonl_session_round_trips_decisions() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-session-decisions-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        let mut decisions = hi_agent::DecisionLog::default();
        decisions.record(hi_agent::Decision {
            summary: "use BTreeMap".into(),
            rationale: "ordered iteration".into(),
            files: vec!["src/m.rs".into()],
        });

        session.record_decisions(&decisions).unwrap();

        let loaded = load_history(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.decisions.entries().len(), 1);
        assert_eq!(loaded.decisions.entries()[0].summary, "use BTreeMap");
        assert_eq!(loaded.decisions.entries()[0].files, vec!["src/m.rs"]);
    }

    #[test]
    fn jsonl_state_replacement_overrides_prior_messages_goal_and_decisions() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-session-state-replacement-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        let old_goal = hi_agent::Goal::new("old goal", vec!["old step".into()]);
        let mut old_decisions = hi_agent::DecisionLog::default();
        old_decisions.record(hi_agent::Decision {
            summary: "discarded decision".into(),
            rationale: "old attempt".into(),
            files: Vec::new(),
        });
        session
            .record(
                &[Message::system("old sys"), Message::user("old attempt")],
                Usage::default(),
            )
            .unwrap();
        session.record_goal(&old_goal).unwrap();
        session.record_decisions(&old_decisions).unwrap();

        let mut kept_decisions = hi_agent::DecisionLog::default();
        kept_decisions.record(hi_agent::Decision {
            summary: "kept decision".into(),
            rationale: "pre-turn".into(),
            files: vec!["src/lib.rs".into()],
        });
        session
            .record_state_replacement(&[Message::system("new sys")], None, &kept_decisions)
            .unwrap();

        let loaded = load_history(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].text(), "new sys");
        assert!(loaded.goal.is_none());
        assert_eq!(loaded.decisions.entries().len(), 1);
        assert_eq!(loaded.decisions.entries()[0].summary, "kept decision");
    }

    #[test]
    fn jsonl_session_round_trips_a_structured_goal() {
        // A long-horizon goal persisted via record_goal survives a load so a
        // /resume picks it up at its active sub-goal.
        use hi_agent::{Goal, GoalStatus};
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-session-goal-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        session
            .record(
                &[Message::system("sys"), Message::user("go")],
                Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    context_occupancy: 1,
                    ..Usage::default()
                },
            )
            .unwrap();
        // A goal mid-progress: sub-goal 1 done, sub-goal 2 active.
        let mut goal = Goal::new(
            "refactor the parser",
            vec!["write tests".into(), "rewrite parser".into()],
        );
        goal.advance(); // mark step 1 done, step 2 active
        session.record_goal(&goal).unwrap();

        let loaded = load_history(&path).unwrap();

        let loaded_goal = loaded.goal.expect("goal persisted across load");
        assert_eq!(loaded_goal.objective, "refactor the parser");
        assert_eq!(loaded_goal.sub_goals.len(), 2);
        assert_eq!(loaded_goal.sub_goals[0].status, GoalStatus::Done);
        assert_eq!(
            loaded_goal.active_index(),
            Some(1),
            "resumes at the active sub-goal"
        );

        session.clear_goal().unwrap();
        let cleared = load_history(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(
            cleared.goal.is_none(),
            "goal_cleared metadata should override earlier persisted goals"
        );
    }

    #[test]
    fn usage_round_trips_cache_tokens_and_estimated_marker() {
        // Session totals must keep full fidelity across resume: cache counters
        // and the estimated marker used to be dropped (only input/output were
        // persisted), silently shrinking a resumed session's numbers.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-session-usage-fidelity-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        session
            .record(
                &[Message::user("go")],
                Usage {
                    input_tokens: 100,
                    output_tokens: 20,
                    cache_read_tokens: 60,
                    cache_creation_tokens: 7,
                    estimated: true,
                    ..Usage::default()
                },
            )
            .unwrap();

        let loaded = load_history(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.usage.input_tokens, 100);
        assert_eq!(loaded.usage.output_tokens, 20);
        assert_eq!(loaded.usage.cache_read_tokens, 60);
        assert_eq!(loaded.usage.cache_creation_tokens, 7);
        assert!(loaded.usage.estimated, "estimated marker survives resume");
    }

    #[test]
    fn load_history_skips_corrupted_lines() {
        // A partially-written last line (from a crash mid-flush) must not make
        // the entire session unresumable. The good lines before it should load.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-session-corrupt-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // Write a valid message line, a valid usage line, then a corrupted line
        // (truncated JSON — what a crash mid-write would leave).
        let valid_msg = serde_json::to_string(&Message::user("hello world")).unwrap();
        let valid_usage = r#"{"type":"usage","input_tokens":10,"output_tokens":5}"#;
        let corrupted = r#"{"role":"user","content":[{"type":"text","text":"trun"#;
        let content = format!("{valid_msg}\n{valid_usage}\n{corrupted}");
        std::fs::write(&path, &content).unwrap();

        let loaded = load_history(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        // The valid message loaded; the corrupted line was skipped.
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].text(), "hello world");
        // The valid usage line loaded too.
        assert_eq!(loaded.usage.input_tokens, 10);
    }

    /// `cwd_digest` is deterministic for a given cwd and stable across calls,
    /// so the same project maps to the same session bucket every run.
    #[test]
    fn cwd_digest_is_stable_and_distinct() {
        let a = cwd_digest();
        let b = cwd_digest();
        assert_eq!(a, b, "digest must be stable across calls");
        assert_eq!(a.len(), 16, "digest is 16 hex chars");
        assert!(
            a.chars().all(|c| c.is_ascii_hexdigit()),
            "digest is filesystem-safe hex: {a}"
        );
    }

    /// `machine_id` returns a non-empty string and is stable across calls
    /// (the same ID is persisted and reused).
    #[test]
    fn machine_id_is_stable() {
        // Don't use the env override (which might be set in CI).
        // Just verify the function returns something non-empty.
        let id = machine_id();
        // machine_id may return None if the data dir isn't writable, but in
        // practice it should always succeed in a test environment.
        if let Some(id) = id {
            assert!(!id.is_empty(), "machine_id must not be empty");
            // A second call should return the same ID (persisted).
            let id2 = machine_id();
            if let Some(id2) = id2 {
                assert_eq!(id, id2, "machine_id must be stable across calls");
            }
        }
    }
}
