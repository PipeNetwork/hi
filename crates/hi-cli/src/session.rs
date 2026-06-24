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
}

/// Appends messages to a session's JSONL file.
pub struct JsonlSession {
    path: PathBuf,
}

impl JsonlSession {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
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
}

pub struct LoadedSession {
    pub messages: Vec<Message>,
    pub usage: Usage,
    pub cost_usd: Option<f64>,
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
    let mut cost_usd = Some(0.0);
    for (i, line) in text.lines().enumerate() {
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
                    };
                    cost_usd = saved_cost;
                }
            }
            continue;
        }
        let message: Message = serde_json::from_str(line)
            .with_context(|| format!("parsing {} line {}", path.display(), i + 1))?;
        messages.push(message);
    }
    Ok(LoadedSession {
        messages,
        usage,
        cost_usd,
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
}
