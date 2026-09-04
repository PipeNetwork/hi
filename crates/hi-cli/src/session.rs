//! JSONL session persistence: one message per line, appended after each turn.
//!
//! Sessions live under `$XDG_DATA_HOME/hi/sessions` (or `~/.local/share/...`).
//! Resuming loads every line back as conversation history.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Take, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use hi_agent::SessionSink;
use hi_ai::{Message, Role, Usage};
use serde::{Deserialize, Serialize};

#[path = "session_shadow.rs"]
pub(crate) mod session_shadow;

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SessionMeta {
    /// Canonical IPOP identity for a locally cached continuation of a remote
    /// session. The local filename remains an implementation detail; all
    /// future sync and PipeFS operations must keep using this identifier.
    RemoteSessionIdentity {
        session_id: String,
    },
    /// Whether IPOP's PipeFS workspace is the authoritative root for this
    /// session. Last write wins; this survives cleanup of ephemeral caches.
    PipeFsMode {
        enabled: bool,
    },
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
    Plan {
        steps: Vec<hi_agent::PlanStep>,
    },
    PlanCleared,
    /// Plan auto-drive pause and stall. Last write wins. `default` so older
    /// session files load as running (paused=false, stall=0).
    PlanDrive {
        #[serde(default)]
        paused: bool,
        /// `Some(true)` identifies an interruption latch that a genuine user
        /// turn consumes. Missing legacy records are inferred as interruption
        /// pauses only when they follow a cancelled synthetic-drive rollback.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resume_on_user_input: Option<bool>,
        #[serde(default)]
        stall: u32,
        /// Delta-encoded exact novelty ledger. Raw tool signatures are hashed
        /// before reaching this record. A reset starts a new plan-step scope.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        evidence_reset: bool,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        evidence_add: Vec<String>,
    },
    /// The leftover-plan approval card was explicitly parked with Escape.
    /// Kept separate from `PlanDrive.paused` so `/view-plan` cannot silently
    /// consume an explicit `/plan pause`.
    PlanApproval {
        #[serde(default)]
        parked: bool,
    },
    /// Goal auto-drive stall. Last write wins. Pause stays on `Goal`.
    GoalDrive {
        #[serde(default)]
        stall: u32,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        evidence_reset: bool,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        evidence_add: Vec<String>,
    },
    /// One turn's final outcome, including why review produced no verdict
    /// when it didn't (that reason used to exist only as a transient status
    /// line, unrecoverable in post-mortems). Diagnostic; ignored on resume.
    TurnOutcome {
        /// Unix seconds when the turn settled — the denominator source for
        /// before/after intervention rates in `hi metrics`. `default` so
        /// pre-timestamp lines load (as 0, excluded from rate windows).
        #[serde(default)]
        ts: u64,
        status: hi_agent::TurnStatus,
        verification: hi_agent::VerificationStatus,
        review: hi_agent::ReviewStatus,
        stop_reason: hi_agent::TurnStopReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        review_unavailable_reason: Option<String>,
        /// True when skeptic/completion-review shared the session model.
        /// `default` keeps older JSONL lines loadable.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        review_same_model: bool,
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
        #[serde(default)]
        plan: Vec<hi_agent::PlanStep>,
    },
}

/// Open a stable, bounded snapshot of an append-only session. Limiting the
/// reader to the length observed from the opened file descriptor prevents a
/// busy writer from extending a status or resume scan indefinitely.
pub(crate) fn session_snapshot_reader(path: &Path) -> std::io::Result<BufReader<Take<File>>> {
    let file = File::open(path)?;
    let snapshot_len = file.metadata()?.len();
    Ok(BufReader::new(file.take(snapshot_len)))
}

/// Count JSONL records with fixed memory. A final unterminated record still
/// counts, matching `str::lines()` and making crash-truncated tails visible in
/// session listings without allocating their contents.
fn session_line_count(path: &Path) -> usize {
    let Ok(mut reader) = session_snapshot_reader(path) else {
        return 0;
    };
    let mut buffer = [0_u8; 64 * 1024];
    let mut lines = 0_usize;
    let mut saw_bytes = false;
    let mut ended_with_newline = false;
    loop {
        let Ok(read) = reader.read(&mut buffer) else {
            return 0;
        };
        if read == 0 {
            break;
        }
        saw_bytes = true;
        ended_with_newline = buffer[read - 1] == b'\n';
        lines = lines.saturating_add(buffer[..read].iter().filter(|byte| **byte == b'\n').count());
    }
    lines.saturating_add(usize::from(saw_bytes && !ended_with_newline))
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

    pub fn record_remote_session_identity(&mut self, session_id: &str) -> Result<()> {
        crate::sync::validate_session_id(session_id)?;
        self.append_meta(&SessionMeta::RemoteSessionIdentity {
            session_id: session_id.to_string(),
        })
    }

    pub fn record_pipefs_mode(&mut self, enabled: bool) -> Result<()> {
        self.append_meta(&SessionMeta::PipeFsMode { enabled })
    }

    /// Persist checkpoint refs so a resumed session knows where it branched.
    #[allow(dead_code)]
    pub fn record_checkpoints(&mut self, refs: &[String]) -> Result<()> {
        self.append_meta(&SessionMeta::Checkpoints {
            refs: refs.to_vec(),
        })
    }
}

impl SessionSink for JsonlSession {
    fn id(&self) -> Option<String> {
        self.path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
    }

    fn record_checkpoints(&mut self, refs: &[String]) -> Result<()> {
        JsonlSession::record_checkpoints(self, refs)
    }

    fn record_pipefs_mode(&mut self, enabled: bool) -> Result<()> {
        JsonlSession::record_pipefs_mode(self, enabled)
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

    fn record_plan(&mut self, plan: &[hi_agent::PlanStep]) -> Result<()> {
        self.append_meta(&SessionMeta::Plan {
            steps: plan.to_vec(),
        })
    }

    fn record_turn_outcome(
        &mut self,
        outcome: &hi_agent::TurnOutcome,
        review_unavailable_reason: Option<&str>,
    ) -> Result<()> {
        self.append_meta(&SessionMeta::TurnOutcome {
            ts: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            status: outcome.status,
            verification: outcome.verification,
            review: outcome.review,
            stop_reason: outcome.stop_reason,
            review_unavailable_reason: review_unavailable_reason.map(str::to_string),
            review_same_model: outcome.review_same_model,
        })
    }

    fn clear_plan(&mut self) -> Result<()> {
        self.append_meta(&SessionMeta::PlanCleared)
    }

    fn record_plan_drive(&mut self, paused: bool, stall: u32) -> Result<()> {
        self.record_plan_drive_state(paused, stall, false, &[])
    }

    fn record_plan_drive_state(
        &mut self,
        paused: bool,
        stall: u32,
        evidence_reset: bool,
        evidence_add: &[String],
    ) -> Result<()> {
        self.record_plan_drive_state_with_policy(paused, stall, false, evidence_reset, evidence_add)
    }

    fn record_plan_drive_state_with_policy(
        &mut self,
        paused: bool,
        stall: u32,
        resume_on_user_input: bool,
        evidence_reset: bool,
        evidence_add: &[String],
    ) -> Result<()> {
        self.append_meta(&SessionMeta::PlanDrive {
            paused,
            resume_on_user_input: Some(resume_on_user_input),
            stall,
            evidence_reset,
            evidence_add: evidence_add.to_vec(),
        })
    }

    fn record_plan_approval_parked(&mut self, parked: bool) -> Result<()> {
        self.append_meta(&SessionMeta::PlanApproval { parked })
    }

    fn record_goal_drive(&mut self, stall: u32) -> Result<()> {
        self.record_goal_drive_state(stall, false, &[])
    }

    fn record_goal_drive_state(
        &mut self,
        stall: u32,
        evidence_reset: bool,
        evidence_add: &[String],
    ) -> Result<()> {
        self.append_meta(&SessionMeta::GoalDrive {
            stall,
            evidence_reset,
            evidence_add: evidence_add.to_vec(),
        })
    }

    fn record_state_replacement(
        &mut self,
        messages: &[Message],
        goal: Option<&hi_agent::Goal>,
        decisions: &hi_agent::DecisionLog,
        plan: &[hi_agent::PlanStep],
    ) -> Result<()> {
        self.append_meta(&SessionMeta::StateReplacement {
            messages: messages.to_vec(),
            goal: goal.cloned(),
            decisions: decisions.entries().to_vec(),
            plan: plan.to_vec(),
        })
    }
}

#[allow(dead_code)]
pub struct LoadedSession {
    pub messages: Vec<Message>,
    pub usage: Usage,
    pub checkpoint_refs: Vec<String>,
    pub harness_settings: hi_workspace::SettingLayer,
    /// Canonical IPOP session identity when this file is a local continuation
    /// cache created by `--attach --resume-local`.
    pub remote_session_id: Option<String>,
    /// Last locally observed remote PipeFS authority bit. `None` identifies
    /// session files written before this metadata record existed.
    pub pipefs_enabled: Option<bool>,
    /// User-defined display name, if one has been assigned (last write wins).
    pub name: Option<String>,
    /// A long-horizon goal persisted across sessions, if any (last write wins).
    pub goal: Option<hi_agent::Goal>,
    /// Intra-session decisions persisted across resume (last write wins).
    pub decisions: hi_agent::DecisionLog,
    /// Unfinished task plan restored into the live plan box.
    pub plan: Vec<hi_agent::PlanStep>,
    /// Plan-drive pause restored from the last `PlanDrive` record.
    pub plan_drive_paused: bool,
    /// Whether the restored pause is an interruption latch consumed by the
    /// next genuine user turn instead of durable manual `/plan pause` intent.
    pub plan_drive_resume_on_user_input: bool,
    /// Whether leftover plan work is waiting on a parked approval card.
    pub plan_approval_parked: bool,
    /// Consecutive no-progress plan-drive turns restored with pause.
    pub plan_drive_stall: u32,
    /// Consecutive no-progress goal-drive turns. Pause stays on `Goal`.
    pub goal_drive_stall: u32,
    /// SHA-256 evidence identities already credited in the current plan scope.
    pub plan_drive_evidence: Vec<String>,
    /// SHA-256 evidence identities already credited in the current goal scope.
    pub goal_drive_evidence: Vec<String>,
}

pub(crate) use crate::session_harness::apply_loaded_session;

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
            &loaded.plan,
        )?;
        session.record(&[], loaded.usage)?;
        session.record_checkpoints(&loaded.checkpoint_refs)?;
        crate::session_harness::append(&temp, &loaded.harness_settings)?;
        if let Some(session_id) = &loaded.remote_session_id {
            session.record_remote_session_identity(session_id)?;
        }
        if let Some(enabled) = loaded.pipefs_enabled {
            session.record_pipefs_mode(enabled)?;
        }
        if loaded.plan_drive_paused
            || loaded.plan_drive_stall > 0
            || !loaded.plan_drive_evidence.is_empty()
        {
            session.record_plan_drive_state_with_policy(
                loaded.plan_drive_paused,
                loaded.plan_drive_stall,
                loaded.plan_drive_resume_on_user_input,
                true,
                &loaded.plan_drive_evidence,
            )?;
        }
        if loaded.plan_approval_parked {
            session.record_plan_approval_parked(true)?;
        }
        if loaded.goal_drive_stall > 0 || !loaded.goal_drive_evidence.is_empty() {
            session.record_goal_drive_state(
                loaded.goal_drive_stall,
                true,
                &loaded.goal_drive_evidence,
            )?;
        }
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
    // Intentionally blocking (`std::process`): session/discovery helpers are
    // sync and run off the async runtime (called during startup, not per-turn).
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
        hi_agent::normalize_git_remote(&remote).unwrap_or(remote)
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
            let lines = session_line_count(&path);
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
            let lines = session_line_count(&path);
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

/// Minimal metadata view for fleet goal status. In particular, parsing a
/// `state_replacement` ignores its potentially large message transcript while
/// retaining the authoritative replacement goal.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GoalSummaryRecord {
    Goal {
        goal: hi_agent::Goal,
    },
    GoalCleared,
    StateReplacement {
        #[serde(default)]
        goal: Option<hi_agent::Goal>,
    },
    #[serde(other)]
    Other,
}

/// Header-only view used by listings. Unknown fields (notably message content
/// and state-replacement transcripts) are skipped by serde instead of being
/// materialized merely to find a title or last-written name.
#[derive(Deserialize)]
struct SessionDisplayHeader {
    #[serde(default)]
    role: Option<Role>,
    #[serde(default, rename = "type")]
    record_type: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

/// Read the last-written goal state from a session file (`goal` /
/// `goal_cleared` / `state_replacement` meta lines, last-wins — mirroring the
/// resume loader) without loading the whole conversation.
pub fn session_goal_summary(path: &Path) -> Option<SessionGoalSummary> {
    let reader = session_snapshot_reader(path).ok()?;
    let mut goal: Option<hi_agent::Goal> = None;
    for line in reader.lines() {
        let Ok(line) = line else {
            continue;
        };
        match serde_json::from_str::<GoalSummaryRecord>(&line) {
            Ok(GoalSummaryRecord::Goal { goal: next }) => goal = Some(next),
            Ok(GoalSummaryRecord::GoalCleared) => goal = None,
            Ok(GoalSummaryRecord::StateReplacement { goal: next }) => goal = next,
            Ok(GoalSummaryRecord::Other) | Err(_) => {}
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
    // Session ids become local path components here, including on the global
    // fallback. Validate before touching the data root so `--resume ../...`
    // can never escape a sessions directory or target an arbitrary file.
    crate::sync::validate_session_id(id)?;
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
    session_display_name_impl(path)
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| "(no prompt yet)".to_string())
}

/// Read a session file once, extracting both the last `Name` meta line and the
/// first user message. Returns the custom name if set, otherwise the title
/// derived from the first user message. Streaming via `BufReader` avoids
/// loading the entire file into memory (some sessions are several MB).
fn session_display_name_impl(path: &Path) -> Option<String> {
    let reader = session_snapshot_reader(path).ok()?;
    let mut custom_name = None;
    let mut first_user = None;
    for line in reader.lines().map_while(Result::ok) {
        if let Ok(header) = serde_json::from_str::<SessionDisplayHeader>(&line) {
            if first_user.is_none()
                && header.role == Some(Role::User)
                && let Ok(message) = serde_json::from_str::<Message>(&line)
            {
                let title = session_title(&message.text());
                if !title.is_empty() {
                    first_user = Some(title);
                }
            }
            if header.record_type.as_deref() == Some("name")
                && let Some(next) = header.name
            {
                custom_name = (!next.trim().is_empty()).then(|| next.trim().to_string());
            }
        }
    }
    custom_name.or(first_user)
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

fn apply_drive_evidence_delta(evidence: &mut BTreeSet<String>, reset: bool, added: Vec<String>) {
    if reset {
        evidence.clear();
    }
    evidence.extend(added.into_iter().filter(|hash| {
        hash.len() == 64
            && hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }));
}

/// Recover the meaning of pre-policy `plan_drive` pause records without
/// treating every old manual `/plan pause` as interruption-resumable.
///
/// Old cancellation cleanup first replaced the transcript with a shorter
/// snapshot that discarded a synthetic plan-drive prompt, then wrote one or
/// more paused drive records (usage/checkpoint records could sit between).
/// That rollback signature is specific enough to migrate the interruption;
/// an otherwise bare legacy pause retains the historical manual semantics.
#[derive(Default)]
struct LegacyPlanPauseMigration {
    cancellation_candidate: bool,
    inferred_interruption_chain: bool,
    /// A missing-policy pause was identified as an interruption latch. Keep
    /// this separate from `inferred_interruption_chain`: ordinary turn records
    /// end the adjacent pause-record chain but must not forget that a later
    /// legacy user turn can consume the inferred latch.
    inferred_pause_active: bool,
    /// Whether the first turn-starting user message after the inferred pause
    /// was real user work rather than a synthetic plan/goal continuation.
    /// Later user-role nudge messages in the same turn must not overwrite it.
    pending_real_user_turn: Option<bool>,
}

impl LegacyPlanPauseMigration {
    fn note_state_replacement(&mut self, before: &[Message], replacement: &[Message]) {
        self.cancellation_candidate = replacement.len() < before.len()
            && before[replacement.len()..].iter().any(|message| {
                message.role == Role::User && message.text().contains(hi_agent::PLAN_DRIVE_PROMPT)
            });
        self.inferred_interruption_chain = false;
        // A replacement abandons the attempted turn. A subsequent successful
        // user turn may still consume the older inferred interruption latch.
        self.pending_real_user_turn = None;
    }

    fn clear_boundary(&mut self) {
        self.cancellation_candidate = false;
        self.inferred_interruption_chain = false;
    }

    fn invalidate(&mut self) {
        self.clear_boundary();
        self.inferred_pause_active = false;
        self.pending_real_user_turn = None;
    }

    fn resolve(&mut self, paused: bool, explicit: Option<bool>) -> bool {
        let inferred = explicit.is_none()
            && paused
            && (self.cancellation_candidate || self.inferred_interruption_chain);
        let resume_on_user_input = explicit.unwrap_or(inferred);
        self.cancellation_candidate = false;
        self.inferred_interruption_chain = paused && resume_on_user_input;
        // Every drive-state record is authoritative. Only an adjacent legacy
        // chain remains eligible for completed-user-turn migration; an
        // explicit-policy record or a later unrelated record supersedes it.
        self.inferred_pause_active = inferred;
        self.pending_real_user_turn = None;
        resume_on_user_input
    }

    fn note_message(&mut self, message: &Message) {
        self.clear_boundary();
        if self.inferred_pause_active
            && self.pending_real_user_turn.is_none()
            && message.role == Role::User
        {
            let text = message.text();
            // Persisted prompts may have session-context wrappers prepended,
            // so exact `DriveKind::from_prompt` classification is too narrow
            // for legacy logs. The synthetic sentinels themselves are unique.
            let synthetic = text.contains(hi_agent::PLAN_DRIVE_PROMPT)
                || text.contains(hi_agent::GOAL_CONTINUE_PROMPT);
            self.pending_real_user_turn = Some(!synthetic);
        }
    }

    fn completed_user_turn_consumes_pause(
        &mut self,
        status: hi_agent::TurnStatus,
        stop_reason: hi_agent::TurnStopReason,
    ) -> bool {
        let successful = status == hi_agent::TurnStatus::Completed
            && !matches!(
                stop_reason,
                hi_agent::TurnStopReason::Cancelled
                    | hi_agent::TurnStopReason::TurnLimit
                    | hi_agent::TurnStopReason::InfrastructureFailure
                    | hi_agent::TurnStopReason::NoProgress
            );
        let consume =
            self.inferred_pause_active && self.pending_real_user_turn == Some(true) && successful;
        self.pending_real_user_turn = None;
        if consume {
            self.inferred_pause_active = false;
            self.inferred_interruption_chain = false;
            self.cancellation_candidate = false;
        }
        consume
    }
}

/// Load a session's messages back into conversation history.
pub fn load_history(path: &Path) -> Result<LoadedSession> {
    let mut reader = session_snapshot_reader(path)
        .with_context(|| format!("opening session {}", path.display()))?;
    let mut reducer_shadow = session_shadow::SessionReducerShadow::new();
    let mut messages = Vec::new();
    let mut usage = Usage::default();
    let mut checkpoint_refs = Vec::new();
    let mut harness_settings = crate::session_harness::empty_layer();
    let mut remote_session_id = None;
    let mut pipefs_enabled = None;
    let mut loaded_goal: Option<hi_agent::Goal> = None;
    let mut loaded_decisions = hi_agent::DecisionLog::default();
    let mut loaded_plan = Vec::new();
    let mut loaded_name = None;
    let mut loaded_plan_drive_paused = false;
    let mut loaded_plan_drive_resume_on_user_input = false;
    let mut legacy_plan_pause = LegacyPlanPauseMigration::default();
    let mut loaded_plan_approval_parked = false;
    let mut loaded_plan_drive_stall = 0;
    let mut loaded_goal_drive_stall = 0;
    let mut loaded_plan_drive_evidence = BTreeSet::new();
    let mut loaded_goal_drive_evidence = BTreeSet::new();
    let mut record = Vec::new();
    loop {
        record.clear();
        if reader
            .read_until(b'\n', &mut record)
            .with_context(|| format!("reading session {}", path.display()))?
            == 0
        {
            break;
        }
        if record.last() == Some(&b'\n') {
            record.pop();
        }
        if record.last() == Some(&b'\r') {
            record.pop();
        }
        let Ok(line) = std::str::from_utf8(&record) else {
            reducer_shadow.observe_opaque_boundary();
            legacy_plan_pause.invalidate();
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        reducer_shadow.observe_legacy_json(line);
        if let Some(layer) = crate::session_harness::parse_record(line)? {
            legacy_plan_pause.clear_boundary();
            harness_settings = layer;
            continue;
        }
        if let Ok(meta) = serde_json::from_str::<SessionMeta>(line) {
            match meta {
                SessionMeta::RemoteSessionIdentity { session_id } => {
                    crate::sync::validate_session_id(&session_id).with_context(|| {
                        format!("invalid remote session identity in {}", path.display())
                    })?;
                    remote_session_id = Some(session_id);
                }
                SessionMeta::PipeFsMode { enabled } => {
                    pipefs_enabled = Some(enabled);
                }
                SessionMeta::Name { name } => {
                    legacy_plan_pause.clear_boundary();
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
                    legacy_plan_pause.clear_boundary();
                    // Replace all prior messages with the compacted set.
                    messages = compacted;
                }
                SessionMeta::Goal { goal } => {
                    legacy_plan_pause.clear_boundary();
                    loaded_goal = Some(goal);
                }
                SessionMeta::GoalCleared => {
                    legacy_plan_pause.clear_boundary();
                    loaded_goal = None;
                    loaded_goal_drive_evidence.clear();
                }
                SessionMeta::Decisions { decisions } => {
                    legacy_plan_pause.clear_boundary();
                    loaded_decisions = hi_agent::DecisionLog::from_entries(decisions);
                }
                SessionMeta::Plan { steps } => {
                    legacy_plan_pause.clear_boundary();
                    loaded_plan = steps;
                }
                SessionMeta::PlanCleared => {
                    legacy_plan_pause.clear_boundary();
                    loaded_plan.clear();
                    loaded_plan_drive_evidence.clear();
                }
                SessionMeta::PlanDrive {
                    paused,
                    resume_on_user_input,
                    stall,
                    evidence_reset,
                    evidence_add,
                } => {
                    loaded_plan_drive_paused = paused;
                    loaded_plan_drive_resume_on_user_input =
                        legacy_plan_pause.resolve(paused, resume_on_user_input);
                    loaded_plan_drive_stall = stall;
                    apply_drive_evidence_delta(
                        &mut loaded_plan_drive_evidence,
                        evidence_reset,
                        evidence_add,
                    );
                }
                SessionMeta::PlanApproval { parked } => {
                    legacy_plan_pause.clear_boundary();
                    loaded_plan_approval_parked = parked;
                }
                SessionMeta::GoalDrive {
                    stall,
                    evidence_reset,
                    evidence_add,
                } => {
                    legacy_plan_pause.clear_boundary();
                    loaded_goal_drive_stall = stall;
                    apply_drive_evidence_delta(
                        &mut loaded_goal_drive_evidence,
                        evidence_reset,
                        evidence_add,
                    );
                }
                // Cancellation outcomes may be written between rollback and
                // the legacy pause record. Other settlements break the
                // cancellation signature.
                SessionMeta::TurnOutcome {
                    status,
                    stop_reason,
                    ..
                } => {
                    if status != hi_agent::TurnStatus::Cancelled
                        && !matches!(
                            stop_reason,
                            hi_agent::TurnStopReason::Cancelled
                                | hi_agent::TurnStopReason::TurnLimit
                        )
                    {
                        legacy_plan_pause.clear_boundary();
                    }
                    if legacy_plan_pause.completed_user_turn_consumes_pause(status, stop_reason) {
                        loaded_plan_drive_paused = false;
                        loaded_plan_drive_resume_on_user_input = false;
                        loaded_plan_drive_stall = 0;
                        loaded_plan_drive_evidence.clear();
                    }
                }
                SessionMeta::StateReplacement {
                    messages: replacement,
                    goal,
                    decisions,
                    plan,
                } => {
                    legacy_plan_pause.note_state_replacement(&messages, &replacement);
                    messages = replacement;
                    loaded_goal = goal;
                    loaded_decisions = hi_agent::DecisionLog::from_entries(decisions);
                    loaded_plan = plan;
                    // A durable rewind replaces conversational/goal state, but
                    // it does not start a new drive-evidence scope in the live
                    // Agent. Preserve the ledger too so restart matches the
                    // uninterrupted process. Explicit evidence reset deltas
                    // remain the sole scope boundary.
                }
            }
            continue;
        }
        let message: Message = match serde_json::from_str(line) {
            Ok(m) => m,
            Err(_) => {
                legacy_plan_pause.invalidate();
                continue;
            }
        };
        legacy_plan_pause.note_message(&message);
        messages.push(message);
    }
    if loaded_plan
        .iter()
        .all(|step| step.status == hi_agent::PlanStatus::Done)
    {
        loaded_plan.clear();
    }
    let loaded = LoadedSession {
        messages,
        usage,
        checkpoint_refs,
        harness_settings,
        remote_session_id,
        pipefs_enabled,
        name: loaded_name,
        goal: loaded_goal,
        decisions: loaded_decisions,
        plan: loaded_plan,
        plan_drive_paused: loaded_plan_drive_paused,
        plan_drive_resume_on_user_input: loaded_plan_drive_resume_on_user_input,
        plan_approval_parked: loaded_plan_approval_parked,
        plan_drive_stall: loaded_plan_drive_stall,
        goal_drive_stall: loaded_goal_drive_stall,
        plan_drive_evidence: loaded_plan_drive_evidence.into_iter().collect(),
        goal_drive_evidence: loaded_goal_drive_evidence.into_iter().collect(),
    };
    reducer_shadow.finish(loaded, "local_jsonl")
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
    let mut reducer_shadow = session_shadow::SessionReducerShadow::new();
    let mut messages = Vec::new();
    let mut usage = Usage::default();
    let mut checkpoint_refs = Vec::new();
    let mut harness_settings = crate::session_harness::empty_layer();
    let mut remote_session_id = None;
    let mut pipefs_enabled = None;
    let mut loaded_goal: Option<hi_agent::Goal> = None;
    let mut loaded_decisions = hi_agent::DecisionLog::default();
    let mut loaded_plan = Vec::new();
    let mut loaded_name = None;
    let mut loaded_plan_drive_paused = false;
    let mut loaded_plan_drive_resume_on_user_input = false;
    let mut legacy_plan_pause = LegacyPlanPauseMigration::default();
    let mut loaded_plan_approval_parked = false;
    let mut loaded_plan_drive_stall = 0;
    let mut loaded_goal_drive_stall = 0;
    let mut loaded_plan_drive_evidence = BTreeSet::new();
    let mut loaded_goal_drive_evidence = BTreeSet::new();

    for record in records {
        reducer_shadow.observe_remote(&record.record_type, &record.payload_json);
        if record.record_type == crate::session_harness::RECORD_TYPE {
            harness_settings = crate::session_harness::parse_record(&record.payload_json)?
                .context("remote harness_settings record omitted its type tag")?;
            legacy_plan_pause.clear_boundary();
            continue;
        }
        if record.record_type == "message" {
            if let Ok(message) = serde_json::from_str::<Message>(&record.payload_json) {
                legacy_plan_pause.note_message(&message);
                messages.push(message);
            } else {
                legacy_plan_pause.invalidate();
            }
            continue;
        }
        if let Ok(meta) = serde_json::from_str::<SessionMeta>(&record.payload_json) {
            match meta {
                SessionMeta::RemoteSessionIdentity { session_id } => {
                    crate::sync::validate_session_id(&session_id)
                        .context("invalid canonical identity in remote session records")?;
                    remote_session_id = Some(session_id);
                }
                SessionMeta::PipeFsMode { enabled } => {
                    pipefs_enabled = Some(enabled);
                }
                SessionMeta::Name { name } => {
                    legacy_plan_pause.clear_boundary();
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
                    legacy_plan_pause.clear_boundary();
                    messages = compacted;
                }
                SessionMeta::Goal { goal } => {
                    legacy_plan_pause.clear_boundary();
                    loaded_goal = Some(goal);
                }
                SessionMeta::GoalCleared => {
                    legacy_plan_pause.clear_boundary();
                    loaded_goal = None;
                    loaded_goal_drive_evidence.clear();
                }
                SessionMeta::Decisions { decisions } => {
                    legacy_plan_pause.clear_boundary();
                    loaded_decisions = hi_agent::DecisionLog::from_entries(decisions);
                }
                SessionMeta::Plan { steps } => {
                    legacy_plan_pause.clear_boundary();
                    loaded_plan = steps;
                }
                SessionMeta::PlanCleared => {
                    legacy_plan_pause.clear_boundary();
                    loaded_plan.clear();
                    loaded_plan_drive_evidence.clear();
                }
                SessionMeta::PlanDrive {
                    paused,
                    resume_on_user_input,
                    stall,
                    evidence_reset,
                    evidence_add,
                } => {
                    loaded_plan_drive_paused = paused;
                    loaded_plan_drive_resume_on_user_input =
                        legacy_plan_pause.resolve(paused, resume_on_user_input);
                    loaded_plan_drive_stall = stall;
                    apply_drive_evidence_delta(
                        &mut loaded_plan_drive_evidence,
                        evidence_reset,
                        evidence_add,
                    );
                }
                SessionMeta::PlanApproval { parked } => {
                    legacy_plan_pause.clear_boundary();
                    loaded_plan_approval_parked = parked;
                }
                SessionMeta::GoalDrive {
                    stall,
                    evidence_reset,
                    evidence_add,
                } => {
                    legacy_plan_pause.clear_boundary();
                    loaded_goal_drive_stall = stall;
                    apply_drive_evidence_delta(
                        &mut loaded_goal_drive_evidence,
                        evidence_reset,
                        evidence_add,
                    );
                }
                // Cancellation outcomes may be written between rollback and
                // the legacy pause record. Other settlements break the
                // cancellation signature.
                SessionMeta::TurnOutcome {
                    status,
                    stop_reason,
                    ..
                } => {
                    if status != hi_agent::TurnStatus::Cancelled
                        && !matches!(
                            stop_reason,
                            hi_agent::TurnStopReason::Cancelled
                                | hi_agent::TurnStopReason::TurnLimit
                        )
                    {
                        legacy_plan_pause.clear_boundary();
                    }
                    if legacy_plan_pause.completed_user_turn_consumes_pause(status, stop_reason) {
                        loaded_plan_drive_paused = false;
                        loaded_plan_drive_resume_on_user_input = false;
                        loaded_plan_drive_stall = 0;
                        loaded_plan_drive_evidence.clear();
                    }
                }
                SessionMeta::StateReplacement {
                    messages: replacement,
                    goal,
                    decisions,
                    plan,
                } => {
                    legacy_plan_pause.note_state_replacement(&messages, &replacement);
                    messages = replacement;
                    loaded_goal = goal;
                    loaded_decisions = hi_agent::DecisionLog::from_entries(decisions);
                    loaded_plan = plan;
                    // Keep parity with the local JSONL loader above: state
                    // replacement is not itself an evidence-scope reset.
                }
            }
        } else {
            legacy_plan_pause.invalidate();
        }
    }

    if loaded_plan
        .iter()
        .all(|step| step.status == hi_agent::PlanStatus::Done)
    {
        loaded_plan.clear();
    }
    let loaded = LoadedSession {
        messages,
        usage,
        checkpoint_refs,
        harness_settings,
        remote_session_id,
        pipefs_enabled,
        name: loaded_name,
        goal: loaded_goal,
        decisions: loaded_decisions,
        plan: loaded_plan,
        plan_drive_paused: loaded_plan_drive_paused,
        plan_drive_resume_on_user_input: loaded_plan_drive_resume_on_user_input,
        plan_approval_parked: loaded_plan_approval_parked,
        plan_drive_stall: loaded_plan_drive_stall,
        goal_drive_stall: loaded_goal_drive_stall,
        plan_drive_evidence: loaded_plan_drive_evidence.into_iter().collect(),
        goal_drive_evidence: loaded_goal_drive_evidence.into_iter().collect(),
    };
    reducer_shadow.finish(loaded, "remote_records")
}
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

    // Resolve display names concurrently, but cap the number of simultaneous
    // record buffers. Spawning one reader per historical session can otherwise
    // amplify a handful of large message records into an avoidable memory spike.
    const TITLE_SCAN_CONCURRENCY: usize = 8;
    let mut titles = Vec::with_capacity(entries.len());
    for chunk in entries.chunks(TITLE_SCAN_CONCURRENCY) {
        std::thread::scope(|scope| {
            let handles = chunk
                .iter()
                .map(|(path, _, _)| scope.spawn(move || session_display_name(path)))
                .collect::<Vec<_>>();
            titles.extend(
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap_or_else(|_| "(no prompt yet)".into())),
            );
        });
    }

    let now = SystemTime::now();
    for ((path, modified, digest), title) in entries.iter().zip(titles.iter()) {
        let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let age = now
            .duration_since(*modified)
            .map(|d| humanize(d.as_secs()))
            .unwrap_or_else(|_| "?".into());
        // Short 8-char project prefix so the column stays narrow but remains
        // enough to disambiguate sessions from different directories.
        let proj = &digest[..digest.len().min(8)];
        println!(
            "{id}  {age:>6} ago  {proj}  {}",
            hi_agent::ui::clip(title, 60)
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
    hi_agent::ui::user_prompt_title(first_user, 72)
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
        JsonlSession, LoadedSession, RemoteRecord, SessionMeta, apply_loaded_session,
        cache_loaded_session, cwd_digest, load_history, load_history_from_records, machine_id,
        session_display_name, session_goal_summary, session_line_count, session_path,
        session_title,
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
    fn explicit_session_ids_are_single_safe_path_components() {
        for invalid in [
            "",
            ".",
            "..",
            "../escape",
            "nested/session",
            r"..\escape",
            "/absolute/session",
            "session\0suffix",
        ] {
            assert!(
                session_path(invalid).is_err(),
                "unsafe session id unexpectedly accepted: {invalid:?}"
            );
        }
    }

    #[test]
    fn line_count_uses_fixed_chunks_and_counts_an_unterminated_tail() {
        let path = std::env::temp_dir().join(format!(
            "hi-session-line-count-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let body = format!("{}\nsmall\nunterminated", "x".repeat(192 * 1024));
        std::fs::write(&path, body).unwrap();

        assert_eq!(session_line_count(&path), 3);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn large_many_record_history_streams_and_preserves_replacement_semantics() {
        use std::io::{BufWriter, Write};

        const STALE_RECORDS: usize = 8_192;
        let path = std::env::temp_dir().join(format!(
            "hi-session-streaming-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = BufWriter::new(file);
        let padding = "x".repeat(512);
        for index in 0..STALE_RECORDS {
            serde_json::to_writer(
                &mut writer,
                &Message::assistant(vec![hi_ai::Content::Text(format!(
                    "stale-{index}-{padding}"
                ))]),
            )
            .unwrap();
            writer.write_all(b"\n").unwrap();
        }
        writer.flush().unwrap();
        drop(writer);
        assert!(std::fs::metadata(&path).unwrap().len() > 4 * 1024 * 1024);

        let mut session = JsonlSession::new(path.clone());
        session
            .record_compaction(&[
                Message::system("compacted system"),
                Message::user("compacted prompt"),
            ])
            .unwrap();
        session
            .record(
                &[Message::assistant(vec![hi_ai::Content::Text(
                    "after compaction".into(),
                )])],
                Usage::default(),
            )
            .unwrap();
        let goal = hi_agent::Goal::new("authoritative goal", vec!["keep going".into()]);
        session
            .record_state_replacement(
                &[
                    Message::system("replacement system"),
                    Message::user("current prompt"),
                ],
                Some(&goal),
                &hi_agent::DecisionLog::default(),
                &[],
            )
            .unwrap();
        session
            .record(
                &[Message::assistant(vec![hi_ai::Content::Text(
                    "current answer".into(),
                )])],
                Usage {
                    input_tokens: 11,
                    output_tokens: 7,
                    ..Usage::default()
                },
            )
            .unwrap();

        let loaded = load_history(&path).unwrap();
        assert_eq!(
            loaded
                .messages
                .iter()
                .map(Message::text)
                .collect::<Vec<_>>(),
            vec!["replacement system", "current prompt", "current answer"]
        );
        assert_eq!(loaded.usage.input_tokens, 11);
        assert_eq!(
            loaded.goal.as_ref().map(|goal| goal.objective.as_str()),
            Some("authoritative goal")
        );
        let summary = session_goal_summary(&path).expect("replacement goal is authoritative");
        assert!(summary.active);
        assert_eq!((summary.done, summary.total), (0, 1));
        assert_eq!(session_line_count(&path), STALE_RECORDS + 6);

        std::fs::remove_file(path).unwrap();
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
            harness_settings: hi_workspace::SettingLayer {
                source: hi_workspace::SettingSource::Session,
                values: std::collections::BTreeMap::from([(
                    hi_workspace::JOB_MAX_ACTIVE.to_string(),
                    hi_workspace::SettingValue::Integer(7),
                )]),
            },
            remote_session_id: Some("canonical-remote-session".into()),
            pipefs_enabled: Some(true),
            name: Some("Restored session".into()),
            goal: None,
            decisions: hi_agent::DecisionLog::default(),
            plan: Vec::new(),
            plan_drive_paused: false,
            plan_drive_resume_on_user_input: false,
            plan_approval_parked: false,
            plan_drive_stall: 0,
            goal_drive_stall: 0,
            plan_drive_evidence: vec!["a".repeat(64)],
            goal_drive_evidence: vec!["b".repeat(64)],
        };

        cache_loaded_session(&path, &expected).expect("cache restored session");
        let loaded = load_history(&path).expect("load restored cache");

        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].text(), "restored prompt");
        assert_eq!(loaded.usage.input_tokens, expected.usage.input_tokens);
        assert_eq!(loaded.usage.output_tokens, expected.usage.output_tokens);
        assert_eq!(loaded.checkpoint_refs, expected.checkpoint_refs);
        assert_eq!(loaded.harness_settings, expected.harness_settings);
        assert_eq!(loaded.remote_session_id, expected.remote_session_id);
        assert_eq!(loaded.pipefs_enabled, expected.pipefs_enabled);
        assert_eq!(loaded.name, expected.name);
        assert_eq!(loaded.plan_drive_evidence, expected.plan_drive_evidence);
        assert_eq!(loaded.goal_drive_evidence, expected.goal_drive_evidence);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn apply_loaded_session_replaces_live_drive_state_and_policy() {
        let root = std::env::temp_dir().join(format!(
            "hi-session-apply-loaded-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let config = hi_agent::AgentConfig {
            paths: hi_agent::AgentPaths {
                workspace_root: root.clone(),
                state_root: root.join("state"),
            },
            ..hi_agent::AgentConfig::default()
        };
        let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
            "http://127.0.0.1:1/v1".into(),
            "unused".into(),
        ));
        let mut agent = hi_agent::Agent::new(provider, config).unwrap();
        agent.restore_plan_drive(true, 9, vec!["c".repeat(64)]);
        agent.restore_plan_approval_parked(true);
        agent.restore_goal_drive(8, vec!["d".repeat(64)]);

        let loaded = |paused, resume_on_user_input, plan_stall, goal_stall| LoadedSession {
            messages: vec![Message::system("restored")],
            usage: Usage::default(),
            checkpoint_refs: Vec::new(),
            harness_settings: crate::session_harness::empty_layer(),
            remote_session_id: None,
            pipefs_enabled: None,
            name: None,
            goal: None,
            decisions: hi_agent::DecisionLog::default(),
            plan: vec![hi_agent::PlanStep {
                title: "continue restored work".into(),
                status: hi_agent::PlanStatus::Pending,
            }],
            plan_drive_paused: paused,
            plan_drive_resume_on_user_input: resume_on_user_input,
            plan_approval_parked: false,
            plan_drive_stall: plan_stall,
            goal_drive_stall: goal_stall,
            plan_drive_evidence: vec!["a".repeat(64)],
            goal_drive_evidence: vec!["b".repeat(64)],
        };

        apply_loaded_session(&mut agent, loaded(false, false, 0, 0)).unwrap();
        assert!(!agent.plan_drive_paused());
        assert_eq!(agent.plan_drive_stall(), 0);
        assert!(!agent.plan_approval_parked());
        assert_eq!(agent.goal_drive_stall(), 0);

        apply_loaded_session(&mut agent, loaded(true, true, 2, 3)).unwrap();
        assert!(agent.plan_drive_paused());
        assert_eq!(agent.plan_drive_stall(), 2);
        assert_eq!(agent.goal_drive_stall(), 3);
        assert!(
            agent
                .prepare_plan_drive_for_turn(hi_agent::DriveKind::User)
                .unwrap()
        );
        assert!(!agent.plan_drive_paused());

        let _ = std::fs::remove_dir_all(root);
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
        let dumped = "[hi:context — session state, not instructions]\n\
# Memory (from past sessions; task-ranked)\n\
Prefer bullets.\n\
[/hi:context]\n\n\
fix the parser";
        assert_eq!(session_title(dumped), "fix the parser");
        assert_eq!(
            session_title("[hi:context — session state, not instructions] no closer"),
            ""
        );
    }

    #[test]
    fn display_name_uses_prompt_inside_context_wrapped_user_message() {
        let path = std::env::temp_dir().join(format!(
            "hi-session-ctx-title-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let dumped = "[hi:context — session state, not instructions]\n\
# Memory (from past sessions; task-ranked)\n\
[/hi:context]\n\n\
fix the parser";
        std::fs::write(
            &path,
            serde_json::to_string(&Message::user(dumped)).unwrap() + "\n",
        )
        .unwrap();
        assert_eq!(session_display_name(&path), "fix the parser");
        std::fs::remove_file(path).unwrap();
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
    fn jsonl_session_round_trips_checkpoint_refs_and_empty_boundary_last_write_wins() {
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
        session.record_checkpoints(&[]).unwrap();

        let loaded = load_history(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(loaded.checkpoint_refs.is_empty());
    }

    #[test]
    fn jsonl_session_persists_canonical_turn_outcome_and_resume_skips_it() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-session-turn-outcome-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        session
            .record(&[Message::user("do the thing")], Usage::default())
            .unwrap();
        let outcome = hi_agent::TurnOutcome {
            status: hi_agent::TurnStatus::Completed,
            verification: hi_agent::VerificationStatus::Passed,
            review: hi_agent::ReviewStatus::Unavailable,
            stop_reason: hi_agent::TurnStopReason::StepLimit,
            changed_files: vec!["src/lib.rs".into()],
            verified_workspace_revision: None,
            effective_route: hi_agent::EffectiveModelRoute {
                provider: None,
                model: "test-model".into(),
            },
            review_same_model: false,
            leftover: None,
            plan_leftover: None,
        };
        session
            .record_turn_outcome(&outcome, Some("provider timed out during review"))
            .unwrap();

        // The reason is durable in the raw JSONL for post-mortems…
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"type\":\"turn_outcome\""), "raw: {raw}");
        assert!(
            raw.contains("provider timed out during review"),
            "raw: {raw}"
        );
        assert!(raw.contains("\"status\":\"completed\""), "raw: {raw}");
        assert!(raw.contains("\"stop_reason\":\"step_limit\""), "raw: {raw}");
        assert!(!raw.contains("\"status\":\"incomplete\""), "raw: {raw}");
        assert!(!raw.contains("\"stop_reason\":\"stalled\""), "raw: {raw}");

        // …and resume skips the record without disturbing the transcript.
        let loaded = load_history(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].text(), "do the thing");
    }

    #[test]
    fn legacy_turn_outcome_names_remain_readable() {
        let legacy = r#"{"type":"turn_outcome","ts":1,"status":"incomplete","verification":"passed","review":"unavailable","stop_reason":"stalled"}"#;
        let parsed: SessionMeta = serde_json::from_str(legacy).unwrap();
        let normalized = serde_json::to_string(&parsed).unwrap();
        assert!(
            normalized.contains("\"stop_reason\":\"no_progress\""),
            "legacy stalled records must normalize as no-progress, not as an execution cap: {normalized}"
        );
        let SessionMeta::TurnOutcome {
            status,
            stop_reason,
            ..
        } = parsed
        else {
            panic!("legacy record parsed as the wrong metadata variant");
        };
        assert_eq!(status, hi_agent::TurnStatus::Failed);
        assert_eq!(stop_reason, hi_agent::TurnStopReason::NoProgress);
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
    fn jsonl_session_restores_unfinished_plan_and_clear_wins() {
        let path = std::env::temp_dir().join(format!(
            "hi-session-plan-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        let plan = vec![hi_agent::PlanStep {
            title: "implement".into(),
            status: hi_agent::PlanStatus::Active,
        }];
        session.record_plan(&plan).unwrap();
        assert_eq!(load_history(&path).unwrap().plan, plan);
        session.clear_plan().unwrap();
        assert!(load_history(&path).unwrap().plan.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn jsonl_session_restores_plan_drive_pause_and_stall() {
        let path = std::env::temp_dir().join(format!(
            "hi-session-plan-drive-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        session.record_plan_drive(true, 4).unwrap();
        let loaded = load_history(&path).unwrap();
        assert!(loaded.plan_drive_paused);
        assert!(
            !loaded.plan_drive_resume_on_user_input,
            "ordinary record_plan_drive is a manual pause"
        );
        assert_eq!(loaded.plan_drive_stall, 4);
        session.record_plan_drive(false, 0).unwrap();
        let loaded = load_history(&path).unwrap();
        assert!(!loaded.plan_drive_paused);
        assert!(!loaded.plan_drive_resume_on_user_input);
        assert_eq!(loaded.plan_drive_stall, 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bare_legacy_paused_plan_drive_remains_manual() {
        use std::io::Write as _;

        let path = std::env::temp_dir().join(format!(
            "hi-session-plan-drive-legacy-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(file, r#"{{"type":"plan_drive","paused":true,"stall":0}}"#).unwrap();
        file.flush().unwrap();

        let loaded = load_history(&path).unwrap();
        assert!(loaded.plan_drive_paused);
        assert!(
            !loaded.plan_drive_resume_on_user_input,
            "an ambiguous legacy pause must preserve manual pause intent"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn legacy_cancelled_plan_drive_rollback_migrates_to_interruption_pause() {
        use std::io::Write as _;

        let path = std::env::temp_dir().join(format!(
            "hi-session-plan-drive-legacy-cancel-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        let kept = vec![Message::system("system"), Message::user("earlier work")];
        let mut before_cancel = kept.clone();
        before_cancel.push(Message::user(format!(
            "context wrapper\n{}\nNext: keep working",
            hi_agent::PLAN_DRIVE_PROMPT
        )));
        session.record(&before_cancel, Usage::default()).unwrap();
        session
            .record_state_replacement(&kept, None, &hi_agent::DecisionLog::default(), &[])
            .unwrap();
        session
            .record(
                &[],
                Usage {
                    input_tokens: 1,
                    ..Usage::default()
                },
            )
            .unwrap();
        drop(session);

        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(file, r#"{{"type":"plan_drive","paused":true,"stall":0}}"#).unwrap();
        writeln!(
            file,
            r#"{{"type":"plan_drive","paused":true,"stall":0,"evidence_reset":true}}"#
        )
        .unwrap();
        file.flush().unwrap();

        let loaded = load_history(&path).unwrap();
        assert!(loaded.plan_drive_paused);
        assert!(
            loaded.plan_drive_resume_on_user_input,
            "rollback-discarded synthetic plan work must resume on user input"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn remote_legacy_pause_migration_matches_local_loader() {
        let kept = vec![Message::system("system"), Message::user("earlier work")];
        let mut before_cancel = kept.clone();
        before_cancel.push(Message::user(format!(
            "context wrapper\n{}\nNext: keep working",
            hi_agent::PLAN_DRIVE_PROMPT
        )));
        let records = vec![
            RemoteRecord {
                record_type: "message".into(),
                payload_json: serde_json::to_string(&before_cancel[0]).unwrap(),
            },
            RemoteRecord {
                record_type: "message".into(),
                payload_json: serde_json::to_string(&before_cancel[1]).unwrap(),
            },
            RemoteRecord {
                record_type: "message".into(),
                payload_json: serde_json::to_string(&before_cancel[2]).unwrap(),
            },
            RemoteRecord {
                record_type: "state_replacement".into(),
                payload_json: serde_json::to_string(&SessionMeta::StateReplacement {
                    messages: kept,
                    goal: None,
                    decisions: Vec::new(),
                    plan: Vec::new(),
                })
                .unwrap(),
            },
            RemoteRecord {
                record_type: "usage".into(),
                payload_json: r#"{"type":"usage","input_tokens":1,"output_tokens":0}"#.into(),
            },
            RemoteRecord {
                record_type: "plan_drive".into(),
                payload_json: r#"{"type":"plan_drive","paused":true,"stall":0}"#.into(),
            },
            RemoteRecord {
                record_type: "plan_drive".into(),
                payload_json:
                    r#"{"type":"plan_drive","paused":true,"stall":0,"evidence_reset":true}"#.into(),
            },
        ];

        let loaded = load_history_from_records(&records).unwrap();
        assert!(loaded.plan_drive_paused);
        assert!(loaded.plan_drive_resume_on_user_input);

        let mut broken_chain = records;
        broken_chain.insert(
            broken_chain.len() - 2,
            RemoteRecord {
                record_type: "message".into(),
                payload_json: serde_json::to_string(&Message::user("unrelated work")).unwrap(),
            },
        );
        let loaded = load_history_from_records(&broken_chain).unwrap();
        assert!(loaded.plan_drive_paused);
        assert!(!loaded.plan_drive_resume_on_user_input);
    }

    fn legacy_interruption_records() -> Vec<RemoteRecord> {
        let kept = vec![Message::system("system"), Message::user("earlier work")];
        let mut before_cancel = kept.clone();
        before_cancel.push(Message::user(format!(
            "context wrapper\n{}\nNext: keep working",
            hi_agent::PLAN_DRIVE_PROMPT
        )));
        let mut records = before_cancel
            .iter()
            .map(|message| RemoteRecord {
                record_type: "message".into(),
                payload_json: serde_json::to_string(message).unwrap(),
            })
            .collect::<Vec<_>>();
        records.push(RemoteRecord {
            record_type: "state_replacement".into(),
            payload_json: serde_json::to_string(&SessionMeta::StateReplacement {
                messages: kept,
                goal: None,
                decisions: Vec::new(),
                plan: vec![hi_agent::PlanStep {
                    title: "finish safely".into(),
                    status: hi_agent::PlanStatus::Pending,
                }],
            })
            .unwrap(),
        });
        // Deliberately omit `resume_on_user_input`: this is the old record the
        // rollback signature above identifies as an interruption pause.
        records.push(RemoteRecord {
            record_type: "plan_drive".into(),
            payload_json: r#"{"type":"plan_drive","paused":true,"stall":0}"#.into(),
        });
        records
    }

    fn turn_outcome_record(
        status: hi_agent::TurnStatus,
        stop_reason: hi_agent::TurnStopReason,
    ) -> RemoteRecord {
        RemoteRecord {
            record_type: "turn_outcome".into(),
            payload_json: serde_json::to_string(&SessionMeta::TurnOutcome {
                ts: 1,
                status,
                verification: hi_agent::VerificationStatus::NotApplicable,
                review: hi_agent::ReviewStatus::NotRequired,
                stop_reason,
                review_unavailable_reason: None,
                review_same_model: false,
            })
            .unwrap(),
        }
    }

    fn load_legacy_records_both(
        records: &[RemoteRecord],
        label: &str,
    ) -> (LoadedSession, LoadedSession) {
        use std::io::Write as _;

        let path = std::env::temp_dir().join(format!(
            "hi-session-legacy-user-resume-{label}-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        for record in records {
            writeln!(file, "{}", record.payload_json).unwrap();
        }
        file.flush().unwrap();
        drop(file);
        let local = load_history(&path).unwrap();
        let remote = load_history_from_records(records).unwrap();
        let _ = std::fs::remove_file(path);
        (local, remote)
    }

    #[test]
    fn completed_legacy_user_turn_consumes_inferred_interruption_pause_locally_and_remotely() {
        let mut records = legacy_interruption_records();
        records.push(RemoteRecord {
            record_type: "message".into(),
            payload_json: serde_json::to_string(&Message::user("fix all of that")).unwrap(),
        });
        // Real user work may update the plan before its terminal outcome. Such
        // metadata ends the adjacent legacy-record chain, not the active
        // inferred interruption latch being consumed by this turn.
        records.push(RemoteRecord {
            record_type: "plan".into(),
            payload_json: serde_json::to_string(&SessionMeta::Plan {
                steps: vec![hi_agent::PlanStep {
                    title: "finish safely".into(),
                    status: hi_agent::PlanStatus::Done,
                }],
            })
            .unwrap(),
        });
        records.push(turn_outcome_record(
            hi_agent::TurnStatus::Completed,
            hi_agent::TurnStopReason::NoApplicableVerification,
        ));

        let (local, remote) = load_legacy_records_both(&records, "completed");
        for loaded in [&local, &remote] {
            assert!(!loaded.plan_drive_paused);
            assert!(!loaded.plan_drive_resume_on_user_input);
            assert_eq!(loaded.plan_drive_stall, 0);
            assert!(loaded.plan_drive_evidence.is_empty());
        }
    }

    #[test]
    fn unsuccessful_legacy_user_turn_keeps_inferred_interruption_pause() {
        for (label, status, stop_reason) in [
            (
                "cancelled",
                hi_agent::TurnStatus::Cancelled,
                hi_agent::TurnStopReason::Cancelled,
            ),
            (
                "failed",
                hi_agent::TurnStatus::Failed,
                hi_agent::TurnStopReason::VerificationFailed,
            ),
        ] {
            let mut records = legacy_interruption_records();
            records.push(RemoteRecord {
                record_type: "message".into(),
                payload_json: serde_json::to_string(&Message::user("try the repair")).unwrap(),
            });
            records.push(turn_outcome_record(status, stop_reason));

            let (local, remote) = load_legacy_records_both(&records, label);
            for loaded in [&local, &remote] {
                assert!(loaded.plan_drive_paused, "{label}");
                assert!(loaded.plan_drive_resume_on_user_input, "{label}");
            }
        }
    }

    #[test]
    fn completed_user_turn_does_not_consume_explicit_or_bare_manual_pause() {
        let mut explicit = legacy_interruption_records();
        explicit.last_mut().unwrap().payload_json =
            serde_json::to_string(&SessionMeta::PlanDrive {
                paused: true,
                resume_on_user_input: Some(true),
                stall: 0,
                evidence_reset: false,
                evidence_add: Vec::new(),
            })
            .unwrap();
        explicit.push(RemoteRecord {
            record_type: "message".into(),
            payload_json: serde_json::to_string(&Message::user("continue explicitly")).unwrap(),
        });
        explicit.push(turn_outcome_record(
            hi_agent::TurnStatus::Completed,
            hi_agent::TurnStopReason::NoApplicableVerification,
        ));
        let (local, remote) = load_legacy_records_both(&explicit, "explicit");
        for loaded in [&local, &remote] {
            assert!(loaded.plan_drive_paused);
            assert!(loaded.plan_drive_resume_on_user_input);
        }

        let mut manual = vec![RemoteRecord {
            record_type: "plan_drive".into(),
            payload_json: r#"{"type":"plan_drive","paused":true,"stall":0}"#.into(),
        }];
        manual.push(RemoteRecord {
            record_type: "message".into(),
            payload_json: serde_json::to_string(&Message::user("ordinary conversation")).unwrap(),
        });
        manual.push(turn_outcome_record(
            hi_agent::TurnStatus::Completed,
            hi_agent::TurnStopReason::NoApplicableVerification,
        ));
        let (local, remote) = load_legacy_records_both(&manual, "manual");
        for loaded in [&local, &remote] {
            assert!(loaded.plan_drive_paused);
            assert!(!loaded.plan_drive_resume_on_user_input);
        }
    }

    #[test]
    fn synthetic_turn_and_user_role_nudges_do_not_consume_inferred_pause() {
        let mut records = legacy_interruption_records();
        records.push(RemoteRecord {
            record_type: "message".into(),
            payload_json: serde_json::to_string(&Message::user(format!(
                "[hi:context]\nwrapped state\n{}\nNext: finish safely",
                hi_agent::PLAN_DRIVE_PROMPT
            )))
            .unwrap(),
        });
        // Nudges are represented as user-role transcript messages. They belong
        // to the already-classified synthetic turn and cannot turn it into a
        // genuine user resume during legacy reconstruction.
        records.push(RemoteRecord {
            record_type: "message".into(),
            payload_json: serde_json::to_string(&Message::user("[hi:nudge:continue] keep working"))
                .unwrap(),
        });
        records.push(turn_outcome_record(
            hi_agent::TurnStatus::Completed,
            hi_agent::TurnStopReason::NoApplicableVerification,
        ));

        let (local, remote) = load_legacy_records_both(&records, "synthetic");
        for loaded in [&local, &remote] {
            assert!(loaded.plan_drive_paused);
            assert!(loaded.plan_drive_resume_on_user_input);
        }
    }

    #[test]
    fn jsonl_session_round_trips_interruption_resume_policy() {
        let path = std::env::temp_dir().join(format!(
            "hi-session-plan-drive-policy-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        session
            .record_plan_drive_state_with_policy(true, 0, true, false, &[])
            .unwrap();

        let loaded = load_history(&path).unwrap();
        assert!(loaded.plan_drive_paused);
        assert!(loaded.plan_drive_resume_on_user_input);
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains(r#""resume_on_user_input":true"#));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn jsonl_session_restores_plan_approval_separately_from_pause() {
        let path = std::env::temp_dir().join(format!(
            "hi-session-plan-approval-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        session.record_plan_drive(true, 0).unwrap();
        session.record_plan_approval_parked(true).unwrap();

        let loaded = load_history(&path).unwrap();
        assert!(loaded.plan_drive_paused);
        assert!(loaded.plan_approval_parked);

        session.record_plan_approval_parked(false).unwrap();
        let loaded = load_history(&path).unwrap();
        assert!(
            loaded.plan_drive_paused,
            "unpark must not consume /plan pause"
        );
        assert!(!loaded.plan_approval_parked);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn jsonl_session_restores_goal_drive_stall() {
        let path = std::env::temp_dir().join(format!(
            "hi-session-goal-drive-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        session.record_goal_drive(4).unwrap();
        let loaded = load_history(&path).unwrap();
        assert_eq!(loaded.goal_drive_stall, 4);
        session.record_goal_drive(0).unwrap();
        let loaded = load_history(&path).unwrap();
        assert_eq!(loaded.goal_drive_stall, 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn jsonl_session_replays_drive_evidence_deltas_and_scope_resets() {
        let path = std::env::temp_dir().join(format!(
            "hi-session-drive-evidence-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut session = JsonlSession::new(path.clone());
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let c = "c".repeat(64);

        session
            .record_plan_drive_state(false, 0, true, std::slice::from_ref(&a))
            .unwrap();
        session
            .record_plan_drive_state(false, 1, false, std::slice::from_ref(&b))
            .unwrap();
        session
            .record_goal_drive_state(2, true, std::slice::from_ref(&b))
            .unwrap();
        let loaded = load_history(&path).unwrap();
        assert_eq!(loaded.plan_drive_evidence, vec![a.clone(), b.clone()]);
        assert_eq!(loaded.goal_drive_evidence, vec![b.clone()]);
        assert_eq!(loaded.plan_drive_stall, 1);
        assert_eq!(loaded.goal_drive_stall, 2);

        // Cancellation/rollback persists a state replacement. It must not make
        // already-credited evidence novel again after process restart.
        session
            .record_state_replacement(
                &[Message::system("rewound")],
                None,
                &hi_agent::DecisionLog::default(),
                &[],
            )
            .unwrap();
        let loaded = load_history(&path).unwrap();
        assert_eq!(loaded.plan_drive_evidence, vec![a.clone(), b.clone()]);
        assert_eq!(loaded.goal_drive_evidence, vec![b.clone()]);

        session
            .record_plan_drive_state(false, 0, true, std::slice::from_ref(&c))
            .unwrap();
        session
            .record_goal_drive_state(0, true, std::slice::from_ref(&c))
            .unwrap();
        let loaded = load_history(&path).unwrap();
        assert_eq!(loaded.plan_drive_evidence, vec![c.clone()]);
        assert_eq!(loaded.goal_drive_evidence, vec![c]);

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"evidence_add\""));
        assert!(raw.contains("\"evidence_reset\":true"));
        let _ = std::fs::remove_file(path);
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
            .record_state_replacement(&[Message::system("new sys")], None, &kept_decisions, &[])
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
        // `cwd_digest` reads the process-wide cwd, so the two calls below are
        // only comparable while no other test is switching directories.
        let _cwd = crate::CWD_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
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
