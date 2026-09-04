//! Startup landing text and session resolution helpers.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail};
use hi_ai::{Content, Message, Role, Usage};
use hi_eval::{EvalInput, TranscriptMessage};

use crate::config::{Cli, Config, Settings};
use crate::provider::provider_label;
use crate::session;

/// The "hi" wordmark as figlet-style 5-row block letters — the splash
/// centerpiece. Generated from `figlet -f small hi`, then padded.
const BANNER: [&str; 5] = [
    " _     _ ",
    "| |__ (_)",
    "| '_ \\| |",
    "| | | | |",
    "|_| |_|_|",
];

pub(crate) fn print_landing(settings: &Settings, context_window: Option<u32>) {
    // Formatting goes through `write_landing`, which is unit-tested; this is
    // just the stdout sink.
    let mut out = std::io::stdout().lock();
    let _ = write_landing(&mut out, settings, context_window);
    let _ = out.flush();
}

/// Render the landing banner into `w`. Separated from `print_landing` so the
/// exact text (ANSI escapes, banner, model, cwd) can be asserted in tests
/// without touching real file descriptors.
pub(crate) fn write_landing<W: std::io::Write>(
    w: &mut W,
    settings: &Settings,
    context_window: Option<u32>,
) -> std::io::Result<()> {
    let orange = "\x1b[38;2;255;140;0m";
    let bold = "\x1b[1m";
    let dim = "\x1b[2m";
    let reset = "\x1b[0m";

    // The 5-row block-letter banner, all orange + bold.
    for row in BANNER {
        writeln!(w, "{bold}{orange}{row}{reset}")?;
    }

    // Model + context window + provider.
    let ctx = context_window
        .map(|win| format!("({}K context)", win / 1000))
        .unwrap_or_default();
    let provider = provider_label(settings.provider);
    let model_line = if ctx.is_empty() {
        format!("{} · {}", settings.model, provider)
    } else {
        format!("{} {} · {}", settings.model, ctx, provider)
    };
    writeln!(w, "{dim}{model_line}{reset}")?;

    // Current working directory.
    let cwd = std::env::current_dir()
        .map(|d| d.display().to_string())
        .unwrap_or_else(|_| "?".into());
    writeln!(w, "{dim}{cwd}{reset}")?;
    Ok(())
}

/// Build the TUI profile list from a config. Shared by the initial list, the
/// saver callback, and the remover callback so they all stay in sync. Only
/// non-default base URLs are included (to keep the `/provider` list concise).
pub(crate) fn profile_infos(config: &Config) -> Vec<hi_tui::ProfileInfo> {
    crate::config::profile_names(config)
        .into_iter()
        .map(|name| {
            let p = config.profiles.get(&name);
            let provider = p
                .and_then(|p| p.provider)
                .map(provider_label)
                .unwrap_or("openai")
                .to_string();
            let model = p.and_then(|p| p.model.clone());
            // Only show the base URL when it differs from the provider default.
            let base_url = p.and_then(|p| {
                p.base_url.clone().filter(|url| {
                    let default = p.provider.map(|prov| prov.default_base_url()).unwrap_or("");
                    url.trim_end_matches('/') != default.trim_end_matches('/')
                })
            });
            hi_tui::ProfileInfo {
                name,
                provider,
                model,
                base_url,
                managed_local_repo: p
                    .and_then(|profile| profile.runtime.as_ref())
                    .filter(|runtime| runtime.kind == "mlx")
                    .map(|runtime| runtime.repo.clone()),
                managed_local_path: p
                    .and_then(|profile| profile.runtime.as_ref())
                    .filter(|runtime| runtime.kind == "mlx")
                    .and_then(|runtime| runtime.model_path.clone()),
            }
        })
        .collect()
}

/// Decide the session file and whether to preload history.
pub(crate) struct LoadedAgentSession {
    pub(crate) messages: Vec<Message>,
    pub(crate) usage: Usage,
    pub(crate) checkpoint_refs: Vec<String>,
    pub(crate) harness_settings: hi_workspace::SettingLayer,
    pub(crate) remote_session_id: Option<String>,
    pub(crate) pipefs_enabled: Option<bool>,
    pub(crate) structured_goal: Option<hi_agent::Goal>,
    pub(crate) decisions: hi_agent::DecisionLog,
    pub(crate) plan: Vec<hi_agent::PlanStep>,
    pub(crate) plan_drive_paused: bool,
    pub(crate) plan_drive_resume_on_user_input: bool,
    pub(crate) plan_approval_parked: bool,
    pub(crate) plan_drive_stall: u32,
    pub(crate) goal_drive_stall: u32,
    pub(crate) plan_drive_evidence: Vec<String>,
    pub(crate) goal_drive_evidence: Vec<String>,
    /// A one-line summary of the resumed session, shown to the user on startup.
    pub(crate) resume_summary: Option<String>,
}

/// Load the evaluator's role-preserving input document. The evaluator owns the
/// JSON contract; this layer only maps its provider-neutral content blocks to
/// the existing hi transcript type.
pub(crate) fn load_eval_input(path: &Path) -> Result<EvalInput> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading evaluation input {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing evaluation input {}", path.display()))
}

pub(crate) fn eval_prompt(input: &EvalInput) -> Result<String> {
    match input {
        EvalInput::Prompt { prompt } => non_empty_prompt(prompt),
        EvalInput::Transcript {
            messages,
            final_prompt,
        } => final_prompt
            .as_deref()
            .map(non_empty_prompt)
            .transpose()?
            .or_else(|| {
                messages
                    .iter()
                    .rev()
                    .find(|message| message.role.eq_ignore_ascii_case("user"))
                    .and_then(|message| transcript_text(&message.content))
            })
            .filter(|prompt| !prompt.trim().is_empty())
            .context("transcript evaluation input requires final_prompt or a text user message"),
    }
}

pub(crate) fn eval_loaded_session(input: &EvalInput) -> Result<Option<LoadedAgentSession>> {
    let EvalInput::Transcript {
        messages,
        final_prompt,
    } = input
    else {
        return Ok(None);
    };
    let retained = if final_prompt.is_none()
        && messages
            .last()
            .is_some_and(|message| message.role.eq_ignore_ascii_case("user"))
    {
        &messages[..messages.len().saturating_sub(1)]
    } else {
        messages.as_slice()
    };
    let messages = retained
        .iter()
        .map(transcript_message)
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(LoadedAgentSession {
        messages,
        usage: Usage::default(),
        checkpoint_refs: Vec::new(),
        harness_settings: crate::session_harness::empty_layer(),
        remote_session_id: None,
        pipefs_enabled: None,
        structured_goal: None,
        decisions: hi_agent::DecisionLog::default(),
        plan: Vec::new(),
        plan_drive_paused: false,
        plan_drive_resume_on_user_input: false,
        plan_approval_parked: false,
        plan_drive_stall: 0,
        goal_drive_stall: 0,
        plan_drive_evidence: Vec::new(),
        goal_drive_evidence: Vec::new(),
        resume_summary: None,
    }))
}

fn transcript_text(value: &serde_json::Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return Some(text.to_string());
    }
    if let Some(blocks) = value.as_array() {
        let text = blocks
            .iter()
            .filter_map(transcript_text)
            .collect::<Vec<_>>()
            .join("");
        return (!text.is_empty()).then_some(text);
    }
    value
        .as_object()
        .and_then(|object| object.get("text"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn non_empty_prompt(prompt: &str) -> Result<String> {
    if prompt.trim().is_empty() {
        bail!("evaluation input prompt must not be empty");
    }
    Ok(prompt.to_string())
}

fn transcript_message(message: &TranscriptMessage) -> Result<Message> {
    let role = match message.role.trim().to_ascii_lowercase().as_str() {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        other => bail!("unsupported transcript role {other:?}"),
    };
    Ok(Message {
        role,
        content: transcript_content(&message.content)?,
    })
}

fn transcript_content(value: &serde_json::Value) -> Result<Vec<Content>> {
    if let Some(text) = value.as_str() {
        return Ok(vec![Content::Text(text.to_string())]);
    }
    let blocks = value
        .as_array()
        .cloned()
        .unwrap_or_else(|| vec![value.clone()]);
    let mut output = Vec::with_capacity(blocks.len());
    for block in blocks {
        if let Some(text) = block.as_str() {
            output.push(Content::Text(text.to_string()));
            continue;
        }
        let object = block
            .as_object()
            .context("transcript content blocks must be strings or objects")?;
        if let Some(text) = object.get("text").and_then(serde_json::Value::as_str)
            && object.get("type").and_then(serde_json::Value::as_str) != Some("image")
        {
            output.push(Content::Text(text.to_string()));
            continue;
        }
        match object.get("type").and_then(serde_json::Value::as_str) {
            Some("image") | Some("image_url") => {
                let image = object.get("image_url").unwrap_or(&block);
                let data = image
                    .get("data")
                    .or_else(|| image.get("url"))
                    .and_then(serde_json::Value::as_str)
                    .context("image transcript block requires data or url")?;
                let media_type = image
                    .get("media_type")
                    .or_else(|| image.get("mime_type"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("image/png");
                output.push(Content::Image {
                    data: data.to_string(),
                    media_type: media_type.to_string(),
                });
            }
            Some("tool_call") | Some("tool_use") => {
                let id = object
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .context("tool call transcript block requires id")?;
                let name = object
                    .get("name")
                    .or_else(|| object.get("function").and_then(|v| v.get("name")))
                    .and_then(serde_json::Value::as_str)
                    .context("tool call transcript block requires name")?;
                let arguments = object
                    .get("arguments")
                    .or_else(|| object.get("input"))
                    .or_else(|| object.get("function").and_then(|v| v.get("arguments")))
                    .map(|value| {
                        value
                            .as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| value.to_string())
                    })
                    .unwrap_or_else(|| "{}".to_string());
                output.push(Content::ToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments,
                });
            }
            Some("tool_result") => {
                let call_id = object
                    .get("call_id")
                    .or_else(|| object.get("tool_call_id"))
                    .and_then(serde_json::Value::as_str)
                    .context("tool result transcript block requires call_id")?;
                let output_value = object.get("output").unwrap_or(&block);
                output.push(Content::ToolResult {
                    call_id: call_id.to_string(),
                    output: output_value
                        .as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| output_value.to_string()),
                });
            }
            Some(other) => bail!("unsupported transcript content block type {other:?}"),
            None => {
                // Accept the provider-neutral externally-tagged hi-ai shape
                // used by saved session records.
                if let Some(text) = object.get("Text").and_then(serde_json::Value::as_str) {
                    output.push(Content::Text(text.to_string()));
                } else {
                    bail!("unsupported untyped transcript content block")
                }
            }
        }
    }
    Ok(output)
}

pub(crate) fn resolve_session(
    cli: &Cli,
) -> Result<(std::path::PathBuf, Option<LoadedAgentSession>)> {
    if cli.eval_input.is_some() {
        return Ok((session::new_session_path()?, None));
    }
    // An exact session file (fleet child): create it fresh, or resume it if it
    // already has history — the dashboard reuses one file across a row's turns.
    if let Some(path) = &cli.session_file {
        if path.is_file() {
            let loaded = session::load_history(path)?;
            return Ok((
                path.clone(),
                Some(LoadedAgentSession {
                    messages: loaded.messages,
                    usage: loaded.usage,
                    checkpoint_refs: loaded.checkpoint_refs,
                    harness_settings: loaded.harness_settings,
                    remote_session_id: loaded.remote_session_id,
                    pipefs_enabled: loaded.pipefs_enabled,
                    structured_goal: loaded.goal,
                    decisions: loaded.decisions,
                    plan: loaded.plan,
                    plan_drive_paused: loaded.plan_drive_paused,
                    plan_drive_resume_on_user_input: loaded.plan_drive_resume_on_user_input,
                    plan_approval_parked: loaded.plan_approval_parked,
                    plan_drive_stall: loaded.plan_drive_stall,
                    goal_drive_stall: loaded.goal_drive_stall,
                    plan_drive_evidence: loaded.plan_drive_evidence,
                    goal_drive_evidence: loaded.goal_drive_evidence,
                    resume_summary: None,
                }),
            ));
        }
        return Ok((path.clone(), None));
    }
    if let Some(id) = &cli.resume {
        let path = session::session_path(id)?;
        let loaded = session::load_history(&path)?;
        let summary = session::resume_summary(&loaded);
        return Ok((
            path,
            Some(LoadedAgentSession {
                messages: loaded.messages,
                usage: loaded.usage,
                checkpoint_refs: loaded.checkpoint_refs,
                harness_settings: loaded.harness_settings,
                remote_session_id: loaded.remote_session_id,
                pipefs_enabled: loaded.pipefs_enabled,
                structured_goal: loaded.goal,
                decisions: loaded.decisions,
                plan: loaded.plan,
                plan_drive_paused: loaded.plan_drive_paused,
                plan_drive_resume_on_user_input: loaded.plan_drive_resume_on_user_input,
                plan_approval_parked: loaded.plan_approval_parked,
                plan_drive_stall: loaded.plan_drive_stall,
                goal_drive_stall: loaded.goal_drive_stall,
                plan_drive_evidence: loaded.plan_drive_evidence,
                goal_drive_evidence: loaded.goal_drive_evidence,
                resume_summary: Some(summary),
            }),
        ));
    }
    if cli.cont {
        if let Some(path) = session::latest_session() {
            let loaded = session::load_history(&path)?;
            let summary = session::resume_summary(&loaded);
            return Ok((
                path,
                Some(LoadedAgentSession {
                    messages: loaded.messages,
                    usage: loaded.usage,
                    checkpoint_refs: loaded.checkpoint_refs,
                    harness_settings: loaded.harness_settings,
                    remote_session_id: loaded.remote_session_id,
                    pipefs_enabled: loaded.pipefs_enabled,
                    structured_goal: loaded.goal,
                    decisions: loaded.decisions,
                    plan: loaded.plan,
                    plan_drive_paused: loaded.plan_drive_paused,
                    plan_drive_resume_on_user_input: loaded.plan_drive_resume_on_user_input,
                    plan_approval_parked: loaded.plan_approval_parked,
                    plan_drive_stall: loaded.plan_drive_stall,
                    goal_drive_stall: loaded.goal_drive_stall,
                    plan_drive_evidence: loaded.plan_drive_evidence,
                    goal_drive_evidence: loaded.goal_drive_evidence,
                    resume_summary: Some(summary),
                }),
            ));
        }
        eprintln!("\x1b[33mno previous session; starting a new one\x1b[0m");
    }
    Ok((session::new_session_path()?, None))
}

/// The one-shot prompt, with piped stdin folded in as context when present
/// (e.g. `cargo test 2>&1 | hi "fix the failures"`). Interactive mode (no
/// prompt) leaves stdin alone for the REPL.
pub(crate) fn effective_prompt(cli: &Cli) -> Result<Option<String>> {
    use std::io::IsTerminal;
    let Some(prompt) = cli.prompt.clone() else {
        return Ok(None);
    };
    if std::io::stdin().is_terminal() {
        return Ok(Some(prompt));
    }
    // Non-terminal stdin is usually a pipe or /dev/null — but it can also be
    // an inherited descriptor nothing ever writes to or closes (background
    // shells, some CI). Reading to EOF unconditionally hangs the whole
    // one-shot forever in that case. Wait briefly for the first byte: a real
    // pipe that has started flowing is then read to EOF (however long the
    // producer takes), while a silent open descriptor forfeits stdin folding.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::Read;
        let mut stdin = std::io::stdin().lock();
        let mut first = [0u8; 1];
        let piped = match stdin.read(&mut first) {
            Ok(0) | Err(_) => Vec::new(),
            Ok(n) => {
                let mut bytes = first[..n].to_vec();
                let _ = stdin.read_to_end(&mut bytes);
                bytes
            }
        };
        let _ = tx.send(piped);
    });
    let piped = match rx.recv_timeout(std::time::Duration::from_secs(2)) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => {
            eprintln!(
                "\x1b[2mstdin is open but silent — proceeding without piped input \
                 (pipe data promptly or redirect stdin from /dev/null)\x1b[0m"
            );
            return Ok(Some(prompt));
        }
    };
    let piped = piped.trim();
    if piped.is_empty() {
        return Ok(Some(prompt));
    }
    Ok(Some(format!("{prompt}\n\nstdin:\n```\n{piped}\n```")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn transcript_final_user_becomes_current_prompt_once() {
        let input = EvalInput::Transcript {
            messages: vec![
                TranscriptMessage {
                    role: "user".into(),
                    content: json!("old"),
                },
                TranscriptMessage {
                    role: "assistant".into(),
                    content: json!("answer"),
                },
                TranscriptMessage {
                    role: "user".into(),
                    content: json!([{ "type": "text", "text": "current" }]),
                },
            ],
            final_prompt: None,
        };
        assert_eq!(eval_prompt(&input).unwrap(), "current");
        let loaded = eval_loaded_session(&input).unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages.last().unwrap().text(), "answer");
    }

    #[test]
    fn explicit_final_prompt_retains_transcript_messages() {
        let input = EvalInput::Transcript {
            messages: vec![TranscriptMessage {
                role: "user".into(),
                content: json!("retained"),
            }],
            final_prompt: Some("new turn".into()),
        };
        assert_eq!(eval_prompt(&input).unwrap(), "new turn");
        assert_eq!(
            eval_loaded_session(&input).unwrap().unwrap().messages.len(),
            1
        );
    }
}
