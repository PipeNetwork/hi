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
        #[serde(default)]
        cost_usd: Option<f64>,
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
    fn record(&mut self, messages: &[Message], usage: Usage, cost_usd: Option<f64>) -> Result<()> {
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
            cost_usd,
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
}

#[allow(dead_code)]
pub struct LoadedSession {
    pub messages: Vec<Message>,
    pub usage: Usage,
    pub cost_usd: Option<f64>,
    pub checkpoint_refs: Vec<String>,
    /// A long-horizon goal persisted across sessions, if any (last write wins).
    pub goal: Option<hi_agent::Goal>,
}

/// One-line summary shown when a session is resumed: message count, cost, and
/// the last user instruction (clipped), so the user knows what they're walking
/// back into.
pub fn resume_summary(loaded: &LoadedSession) -> String {
    let n = loaded
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .count();
    let cost = loaded
        .cost_usd
        .map(|c| format!("${c:.2}"))
        .unwrap_or_else(|| "unknown cost".into());
    let last = loaded
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| hi_agent::ui::clip(&m.text(), 60))
        .unwrap_or_default();
    format!("Resumed: {n} messages, {cost}, last: '{last}'")
}

/// Directory holding all session files (may not exist yet).
pub fn sessions_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
        })?;
    Some(base.join("hi").join("sessions"))
}

/// Path to the persistent REPL input-history file.
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
pub fn session_path(id: &str) -> Result<PathBuf> {
    let dir = sessions_dir().context("could not determine session directory")?;
    let name = if id.ends_with(".jsonl") {
        id.to_string()
    } else {
        format!("{id}.jsonl")
    };
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
    let mut cost_usd: Option<f64> = None;
    let mut checkpoint_refs = Vec::new();
    let mut loaded_goal: Option<hi_agent::Goal> = None;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(meta) = serde_json::from_str::<SessionMeta>(line) {
            match meta {
                SessionMeta::Usage {
                    input_tokens,
                    output_tokens,
                    cost_usd: saved_cost,
                } => {
                    usage = Usage {
                        input_tokens,
                        output_tokens,
                        cache_read_tokens: 0,
                        cache_creation_tokens: 0,
                        input_includes_cache: false,
                        context_occupancy: input_tokens,
                        billable: None,
                    };
                    cost_usd = saved_cost;
                }
                SessionMeta::Checkpoints { refs } => {
                    checkpoint_refs.extend(refs);
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
        cost_usd,
        checkpoint_refs,
        goal: loaded_goal,
    })
}

/// Print a summary of saved sessions (id, age, first user message).
pub fn list_sessions() -> Result<()> {
    let Some(dir) = sessions_dir() else {
        println!("no session directory");
        return Ok(());
    };
    let mut entries: Vec<(PathBuf, SystemTime)> = match fs::read_dir(&dir) {
        Ok(read) => read
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
            .map(|p| {
                let modified = fs::metadata(&p)
                    .and_then(|m| m.modified())
                    .unwrap_or(UNIX_EPOCH);
                (p, modified)
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    if entries.is_empty() {
        println!("no sessions in {}", dir.display());
        return Ok(());
    }
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));

    let now = SystemTime::now();
    for (path, modified) in entries {
        let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let age = now
            .duration_since(modified)
            .map(|d| humanize(d.as_secs()))
            .unwrap_or_else(|_| "?".into());
        let title = first_user_message(&path)
            .map(|m| session_title(&m))
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| "(no prompt yet)".to_string());
        println!("{id}  {age:>6} ago  {}", hi_agent::ui::clip(&title, 70));
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
    use super::{JsonlSession, load_history, session_title};
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
                    billable: None,
                },
                Some(0.1234),
            )
            .unwrap();

        let loaded = load_history(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.usage.input_tokens, 123);
        assert_eq!(loaded.usage.output_tokens, 45);
        assert_eq!(loaded.cost_usd, Some(0.1234));
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
                    billable: None,
                },
                None,
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
        let _ = std::fs::remove_file(&path);

        let loaded_goal = loaded.goal.expect("goal persisted across load");
        assert_eq!(loaded_goal.objective, "refactor the parser");
        assert_eq!(loaded_goal.sub_goals.len(), 2);
        assert_eq!(loaded_goal.sub_goals[0].status, GoalStatus::Done);
        assert_eq!(
            loaded_goal.active_index(),
            Some(1),
            "resumes at the active sub-goal"
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
}
