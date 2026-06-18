//! Built-in tools: `read`, `write`, `edit`, `bash`.
//!
//! Richer capabilities come from subprocess CLI tools the model invokes via
//! `bash` — not a plugin runtime — so this set stays intentionally small.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use pi_ai::ToolSpec;
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;

/// Cap tool output so a single read or noisy command can't blow the context.
const MAX_OUTPUT_CHARS: usize = 50_000;
const BASH_TIMEOUT: Duration = Duration::from_secs(120);
const CHECK_TIMEOUT: Duration = Duration::from_secs(300);

/// Run a verification command (e.g. a test suite) and report `(passed, output)`
/// based on its exit status. Used by the agent's verification loop.
pub async fn run_check(command: &str) -> (bool, String) {
    let future = Command::new("sh").arg("-c").arg(command).output();
    match tokio::time::timeout(CHECK_TIMEOUT, future).await {
        Ok(Ok(output)) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&output.stdout));
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.is_empty() {
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
                text.push_str(&stderr);
            }
            (output.status.success(), truncate(&text))
        }
        Ok(Err(err)) => (false, format!("failed to run verification: {err}")),
        Err(_) => (
            false,
            format!("verification timed out after {}s", CHECK_TIMEOUT.as_secs()),
        ),
    }
}

/// The tools advertised to the model each turn.
pub fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "read".into(),
            description: "Read a UTF-8 text file and return its contents.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to read." }
                },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "write".into(),
            description: "Create or overwrite a file with the given content. Parent directories are created as needed.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to write." },
                    "content": { "type": "string", "description": "Full content to write." }
                },
                "required": ["path", "content"]
            }),
        },
        ToolSpec {
            name: "edit".into(),
            description: "Replace an exact, unique substring in a file. old_string must occur exactly once.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to edit." },
                    "old_string": { "type": "string", "description": "Exact text to replace; must be unique in the file." },
                    "new_string": { "type": "string", "description": "Replacement text." }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
        ToolSpec {
            name: "bash".into(),
            description: "Run a shell command via `sh -c` in the current working directory and return combined stdout/stderr.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to run." }
                },
                "required": ["command"]
            }),
        },
    ]
}

/// The result of a tool call, split into `content` shown to the model and an
/// optional richer `display` for the UI (e.g. a colored diff). This keeps
/// edit/write feedback terse for the model while showing the user what changed.
pub struct ToolOutput {
    pub content: String,
    pub display: Option<String>,
}

impl ToolOutput {
    fn plain(content: String) -> Self {
        Self { content, display: None }
    }

    fn shown(content: String, display: String) -> Self {
        Self { content, display: Some(display) }
    }
}

/// Execute a tool by name. Tool failures are returned as content (not errors)
/// so the model sees them and can recover, rather than aborting the turn.
pub async fn execute(name: &str, arguments: &str) -> ToolOutput {
    match run(name, arguments).await {
        Ok(output) => output,
        Err(err) => ToolOutput::plain(format!("Error: {err:#}")),
    }
}

async fn run(name: &str, arguments: &str) -> Result<ToolOutput> {
    match name {
        "read" => {
            let args: ReadArgs = parse(arguments)?;
            let content = tokio::fs::read_to_string(&args.path)
                .await
                .with_context(|| format!("reading {}", args.path))?;
            Ok(ToolOutput::plain(truncate(&content)))
        }
        "write" => {
            let args: WriteArgs = parse(arguments)?;
            if let Some(parent) = Path::new(&args.path).parent()
                && !parent.as_os_str().is_empty()
            {
                tokio::fs::create_dir_all(parent)
                    .await
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            let before = tokio::fs::read_to_string(&args.path).await.unwrap_or_default();
            tokio::fs::write(&args.path, &args.content)
                .await
                .with_context(|| format!("writing {}", args.path))?;
            Ok(ToolOutput::shown(
                format!("Wrote {} bytes to {}", args.content.len(), args.path),
                diff(&before, &args.content),
            ))
        }
        "edit" => {
            let args: EditArgs = parse(arguments)?;
            let before = tokio::fs::read_to_string(&args.path)
                .await
                .with_context(|| format!("reading {}", args.path))?;
            match before.matches(&args.old_string).count() {
                0 => bail!("old_string not found in {}", args.path),
                1 => {}
                n => bail!(
                    "old_string is not unique in {} ({n} matches); include more surrounding context",
                    args.path
                ),
            }
            let after = before.replacen(&args.old_string, &args.new_string, 1);
            tokio::fs::write(&args.path, &after)
                .await
                .with_context(|| format!("writing {}", args.path))?;
            Ok(ToolOutput::shown(format!("Edited {}", args.path), diff(&before, &after)))
        }
        "bash" => {
            let args: BashArgs = parse(arguments)?;
            Ok(ToolOutput::plain(truncate(&run_bash(&args.command).await?)))
        }
        other => bail!("unknown tool: {other}"),
    }
}

/// A compact colored diff (changed lines only) for UI display.
fn diff(before: &str, after: &str) -> String {
    use similar::{ChangeTag, TextDiff};
    let mut out = String::new();
    for change in TextDiff::from_lines(before, after).iter_all_changes() {
        let (sign, color) = match change.tag() {
            ChangeTag::Delete => ("-", "\x1b[31m"),
            ChangeTag::Insert => ("+", "\x1b[32m"),
            ChangeTag::Equal => continue,
        };
        let line = change.value().trim_end_matches('\n');
        out.push_str(&format!("{color}{sign} {line}\x1b[0m\n"));
    }
    if out.is_empty() {
        out.push_str("(no changes)");
    }
    out
}

async fn run_bash(command: &str) -> Result<String> {
    let future = Command::new("sh").arg("-c").arg(command).output();
    let output = match tokio::time::timeout(BASH_TIMEOUT, future).await {
        Ok(result) => result.context("failed to spawn command")?,
        Err(_) => bail!("command timed out after {}s", BASH_TIMEOUT.as_secs()),
    };

    let mut out = String::new();
    out.push_str(&String::from_utf8_lossy(&output.stdout));
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&stderr);
    }
    if let Some(code) = output.status.code()
        && code != 0
    {
        out.push_str(&format!("\n[exit code {code}]"));
    }
    if out.is_empty() {
        out.push_str("[no output]");
    }
    Ok(out)
}

fn parse<T: for<'de> Deserialize<'de>>(arguments: &str) -> Result<T> {
    serde_json::from_str(arguments).context("invalid tool arguments")
}

/// Truncate output to a character budget, noting how much was dropped.
fn truncate(s: &str) -> String {
    if s.chars().count() <= MAX_OUTPUT_CHARS {
        return s.to_string();
    }
    let kept: String = s.chars().take(MAX_OUTPUT_CHARS).collect();
    let dropped = s.chars().count() - MAX_OUTPUT_CHARS;
    format!("{kept}\n… [truncated {dropped} characters]")
}

#[derive(Deserialize)]
struct ReadArgs {
    path: String,
}

#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}

#[derive(Deserialize)]
struct EditArgs {
    path: String,
    old_string: String,
    new_string: String,
}

#[derive(Deserialize)]
struct BashArgs {
    command: String,
}
