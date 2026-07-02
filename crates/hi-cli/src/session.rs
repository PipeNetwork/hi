//! JSONL session persistence: one message per line, appended after each turn.
//!
//! Sessions live under `$XDG_DATA_HOME/hi/sessions` (or `~/.local/share/...`).
//! Resuming loads every line back as conversation history. Branching/tree
//! sessions (pi-style) are a future extension; this is a linear log.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use hi_agent::SessionSink;
use hi_ai::{Message, Role, Usage};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SessionMeta {
    Usage {
        input_tokens: u64,
        output_tokens: u64,
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

    /// Persist checkpoint refs so a resumed session knows where it branched.
    #[allow(dead_code)]
    pub fn record_checkpoints(&mut self, refs: &[String]) -> Result<()> {
        if refs.is_empty() {
            return Ok(());
        }
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        let mut writer = BufWriter::new(file);
        let line = serde_json::to_string(&SessionMeta::Checkpoints {
            refs: refs.to_vec(),
        })?;
        writeln!(writer, "{line}")?;
        writer.flush()?;
        Ok(())
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
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        let mut writer = BufWriter::new(file);
        for message in messages {
            let line = serde_json::to_string(message)?;
            writeln!(writer, "{line}")?;
        }
        let line = serde_json::to_string(&SessionMeta::Usage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        })?;
        writeln!(writer, "{line}")?;
        writer.flush()?;
        Ok(())
    }

    fn record_compaction(&mut self, messages: &[Message]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        let mut writer = BufWriter::new(file);
        let line = serde_json::to_string(&SessionMeta::Compaction {
            messages: messages.to_vec(),
        })?;
        writeln!(writer, "{line}")?;
        writer.flush()?;
        Ok(())
    }

    fn record_goal(&mut self, goal: &hi_agent::Goal) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        let mut writer = BufWriter::new(file);
        let line = serde_json::to_string(&SessionMeta::Goal { goal: goal.clone() })?;
        writeln!(writer, "{line}")?;
        writer.flush()?;
        Ok(())
    }

    fn clear_goal(&mut self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        let mut writer = BufWriter::new(file);
        let line = serde_json::to_string(&SessionMeta::GoalCleared)?;
        writeln!(writer, "{line}")?;
        writer.flush()?;
        Ok(())
    }

    fn record_decisions(&mut self, decisions: &hi_agent::DecisionLog) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        let mut writer = BufWriter::new(file);
        let line = serde_json::to_string(&SessionMeta::Decisions {
            decisions: decisions.entries().to_vec(),
        })?;
        writeln!(writer, "{line}")?;
        writer.flush()?;
        Ok(())
    }

    fn record_state_replacement(
        &mut self,
        messages: &[Message],
        goal: Option<&hi_agent::Goal>,
        decisions: &hi_agent::DecisionLog,
    ) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        let mut writer = BufWriter::new(file);
        let line = serde_json::to_string(&SessionMeta::StateReplacement {
            messages: messages.to_vec(),
            goal: goal.cloned(),
            decisions: decisions.entries().to_vec(),
        })?;
        writeln!(writer, "{line}")?;
        writer.flush()?;
        Ok(())
    }
}

#[allow(dead_code)]
pub struct LoadedSession {
    pub messages: Vec<Message>,
    pub usage: Usage,
    pub checkpoint_refs: Vec<String>,
    /// A long-horizon goal persisted across sessions, if any (last write wins).
    pub goal: Option<hi_agent::Goal>,
    /// Intra-session decisions persisted across resume (last write wins).
    pub decisions: hi_agent::DecisionLog,
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
fn data_root() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
        .map(|p| p.join("hi"))
}

/// A short, stable, filesystem-safe key for the current working directory.
/// Uses FNV-1a over the canonicalized path (resolves symlinks, so a project
/// reached via different paths still maps to one bucket). Falls back to the
/// raw cwd if canonicalization fails. Sixteen hex chars is enough to avoid
/// collisions across any realistic number of project dirs while keeping the
/// directory listing readable.
fn cwd_digest() -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let key = std::fs::canonicalize(&cwd).unwrap_or(cwd);
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in key.as_os_str().as_encoded_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Path to the persistent REPL input-history file. Per-directory (lives inside
/// `sessions_dir()`) so Up-arrow history is scoped to the current project.
pub fn history_path() -> Option<PathBuf> {
    sessions_dir().and_then(|d| d.parent().map(|p| p.join("history")))
}

/// Path for a brand-new session, named by creation time (sortable).
pub fn new_session_path() -> Result<PathBuf> {
    let dir = sessions_dir().context("could not determine session directory")?;
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    Ok(dir.join(format!("{millis:013}.jsonl")))
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

/// The most recently modified session, if any.
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

/// Load a session's messages back into conversation history.
pub fn load_history(path: &Path) -> Result<LoadedSession> {
    let text =
        fs::read_to_string(path).with_context(|| format!("reading session {}", path.display()))?;
    let mut messages = Vec::new();
    let mut usage = Usage::default();
    let mut checkpoint_refs = Vec::new();
    let mut loaded_goal: Option<hi_agent::Goal> = None;
    let mut loaded_decisions = hi_agent::DecisionLog::default();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(meta) = serde_json::from_str::<SessionMeta>(line) {
            match meta {
                SessionMeta::Usage {
                    input_tokens,
                    output_tokens,
                } => {
                    usage = Usage {
                        input_tokens,
                        output_tokens,
                        cache_read_tokens: 0,
                        cache_creation_tokens: 0,
                        input_includes_cache: false,
                        context_occupancy: input_tokens,
                        rate_limits: None,
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
        goal: loaded_goal,
        decisions: loaded_decisions,
    })
}

/// Print a summary of saved sessions (id, age, first user message).
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
        let title = first_user_message(&path)
            .map(|m| session_title(&m))
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| "(no prompt yet)".to_string());
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
    use super::{JsonlSession, cwd_digest, load_history, session_title};
    use hi_agent::SessionSink;
    use hi_ai::{Message, Usage};

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
                    cache_read_tokens: 0,
                    cache_creation_tokens: 0,
                    input_includes_cache: false,
                    context_occupancy: 123,
                    rate_limits: None,
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
                    cache_read_tokens: 0,
                    cache_creation_tokens: 0,
                    input_includes_cache: false,
                    context_occupancy: 1,
                    rate_limits: None,
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
}
