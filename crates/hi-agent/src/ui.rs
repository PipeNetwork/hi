//! The output seam between the agent loop and whatever renders it.
//!
//! The agent emits raw events through [`Ui`]; each frontend (plain stdout, the
//! TUI) decides how to format them. This keeps the loop free of `print!` and
//! terminal concerns.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// A mutation that requires an explicit user decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfirmationRequest {
    FileEdit {
        path: String,
        diff: String,
    },
    ShellMutation {
        command: String,
        cwd: String,
    },
    DelegateApply {
        summary: String,
        diff: String,
    },
    /// Product/design question that must pause the turn. Not a write approval.
    AskUser {
        question: String,
        options: Vec<String>,
    },
    /// Network / MCP / browser action. `mcp_grant` is session standing-allow
    /// for that server+tool only — never for bash or `browser_exec`.
    External {
        tool: String,
        /// Complete canonical tool arguments used only to bind durable
        /// approvals to the exact operation. The bounded `summary` remains the
        /// user-facing preview and must not be used as authorization material.
        operation_arguments: serde_json::Value,
        summary: String,
        target: String,
        mcp_grant: Option<(String, String)>,
    },
}

impl ConfirmationRequest {
    pub fn title(&self) -> &'static str {
        match self {
            Self::FileEdit { .. } => "Confirm file edit",
            Self::ShellMutation { .. } => "Confirm shell mutation",
            Self::DelegateApply { .. } => "Confirm delegated changes",
            Self::AskUser { .. } => "Question for you",
            Self::External { tool, .. } if tool == "browser_exec" => "Confirm browser action",
            Self::External { tool, .. } if tool == "use_tool" => "Confirm MCP tool",
            Self::External { .. } => "Confirm network request",
        }
    }

    /// Conservative local classifier for `/permissions auto`.
    ///
    /// This is intentionally narrower than an LLM "looks safe" judgment: small
    /// source/doc edits are auto-approved, shell/delegate operations are not.
    pub fn safe_for_auto(&self) -> bool {
        match self {
            Self::FileEdit { path, diff } => {
                let lower = path.to_ascii_lowercase();
                let secretish = [".env", "credential", "secret", "token", "key.pem"]
                    .iter()
                    .any(|needle| lower.contains(needle));
                let destructive = diff.lines().filter(|line| line.starts_with('-')).count() > 80;
                !secretish && !destructive && diff.len() <= 32 * 1024
            }
            Self::ShellMutation { .. }
            | Self::DelegateApply { .. }
            | Self::AskUser { .. }
            | Self::External { .. } => false,
        }
    }

    pub fn details(&self) -> String {
        match self {
            Self::FileEdit { path, diff } => format!("file: {path}\n\n{diff}"),
            Self::ShellMutation { command, cwd } => format!(
                "working directory: {cwd}\nwarning: this command is likely to mutate the workspace\n\n$ {command}"
            ),
            Self::DelegateApply { summary, diff } => format!("{summary}\n\n{diff}"),
            Self::AskUser { question, options } => {
                if options.is_empty() {
                    question.clone()
                } else {
                    let listed = options
                        .iter()
                        .enumerate()
                        .map(|(i, option)| format!("{}. {option}", i + 1))
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!("{question}\n\n{listed}")
                }
            }
            Self::External {
                tool,
                summary,
                target,
                ..
            } => {
                let mut out = format!("tool: {tool}");
                if !target.is_empty() {
                    out.push_str(&format!("\ntarget: {target}"));
                }
                if !summary.is_empty() {
                    out.push_str("\n\n");
                    out.push_str(summary);
                }
                out
            }
        }
    }

    /// MCP server+tool this confirm may standing-grant for the session.
    pub fn mcp_standing_grant(&self) -> Option<(&str, &str)> {
        match self {
            Self::External {
                mcp_grant: Some((server, tool)),
                ..
            } => Some((server.as_str(), tool.as_str())),
            _ => None,
        }
    }
}

/// Status line (and loop-firing summary marker) when a confirm is parked.
pub const PARKED_FOR_APPROVAL_STATUS: &str = "⏸ parked for approval — /inbox";
/// Tool result returned to the model when a confirm is parked, not denied.
pub const PARKED_TOOL_RESULT: &str = "Action parked for approval (not run). Resolve with /inbox.";

/// Build a durable capability request for a confirm overlay (not `AskUser`).
pub fn confirmation_capability(
    request: &ConfirmationRequest,
) -> Option<(hi_policy::CapabilityRequest, hi_policy::OperationDigest)> {
    let workspace_id = std::env::current_dir()
        .ok()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "interactive".into());
    let (capability, scope, tool, arguments, detail) = match request {
        ConfirmationRequest::AskUser { .. } => return None,
        ConfirmationRequest::FileEdit { path, diff } => (
            hi_policy::CapabilityKind::WorkspaceWrite,
            hi_policy::ResourceScope::Paths {
                workspace_id: workspace_id.clone(),
                paths: vec![path.clone()],
            },
            "edit",
            serde_json::json!({ "path": path, "diff": diff }),
            format!("file edit requested for {path}"),
        ),
        ConfirmationRequest::ShellMutation { command, cwd } => (
            hi_policy::CapabilityKind::ProcessExecution,
            hi_policy::ResourceScope::Command {
                workspace_id: workspace_id.clone(),
                command: hi_policy::normalize_command(command),
                cwd: hi_policy::normalize_cwd(cwd),
            },
            "bash",
            serde_json::json!({
                "command": hi_policy::normalize_command(command),
                "cwd": hi_policy::normalize_cwd(cwd)
            }),
            format!("shell mutation requested in {cwd}"),
        ),
        ConfirmationRequest::DelegateApply { summary, diff } => (
            hi_policy::CapabilityKind::DelegateApplication,
            hi_policy::ResourceScope::Operation {
                workspace_id: workspace_id.clone(),
                label: summary.clone(),
            },
            "delegate",
            serde_json::json!({ "summary": summary, "diff": diff }),
            "verified delegated application requested".into(),
        ),
        ConfirmationRequest::External {
            tool,
            operation_arguments,
            target,
            ..
        } => (
            hi_policy::CapabilityKind::NetworkAccess,
            hi_policy::ResourceScope::Operation {
                workspace_id: workspace_id.clone(),
                label: if target.is_empty() {
                    tool.clone()
                } else {
                    format!("{tool}:{target}")
                },
            },
            tool.as_str(),
            serde_json::json!({
                "target": target,
                "arguments": operation_arguments,
            }),
            format!("external {tool} requested"),
        ),
    };
    let digest = hi_policy::OperationDigest::calculate(
        &capability,
        tool,
        &arguments,
        &workspace_id,
        &scope,
        match request {
            ConfirmationRequest::FileEdit { diff, .. }
            | ConfirmationRequest::DelegateApply { diff, .. } => Some(diff.as_str()),
            ConfirmationRequest::ShellMutation { .. }
            | ConfirmationRequest::AskUser { .. }
            | ConfirmationRequest::External { .. } => None,
        },
    );
    Some((
        hi_policy::approval_request(
            capability,
            scope,
            digest.clone(),
            tool,
            None,
            None,
            request.title(),
            detail,
        ),
        digest,
    ))
}

/// Persist a pending approval with a non-null `run_id` so `abandon_interactive`
/// on the next process start does not wipe it. Loop children set `HI_LOOP_ID`.
pub fn park_confirmation(
    store: &dyn hi_policy::ApprovalStore,
    request: &ConfirmationRequest,
) -> anyhow::Result<String> {
    let Some((mut cap, _)) = confirmation_capability(request) else {
        anyhow::bail!("not an approval-backed confirmation");
    };
    let run_id = std::env::var("HI_LOOP_ID")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(|id| format!("loop:{id}"))
        .unwrap_or_else(|| "parked".into());
    cap.run_id = Some(run_id);
    let record = store.create(cap)?;
    Ok(record.request.approval_id.0)
}

/// If `/inbox allow` already approved a matching digest, claim it and run.
pub fn try_claim_approved_confirmation(
    store: &dyn hi_policy::ApprovalStore,
    request: &ConfirmationRequest,
) -> bool {
    let Some((_, digest)) = confirmation_capability(request) else {
        return false;
    };
    let Ok(records) = store.pending() else {
        return false;
    };
    for rec in records {
        if rec.state != hi_policy::ApprovalState::Approved {
            continue;
        }
        if rec.request.operation_digest != digest {
            continue;
        }
        if store.claim(&rec.request.approval_id, &digest).is_ok() {
            return true;
        }
    }
    false
}

/// Ask/Auto pause before this egress tool. `Always` (confirm_edits off) never
/// pauses. `web_fetch` / research pause only in Ask; `browser_exec` / `use_tool`
/// pause in Ask and Auto (same bar as mutating shell).
pub fn egress_confirm_required(
    mode: crate::PermissionMode,
    confirm_edits: bool,
    tool: &str,
) -> bool {
    if !confirm_edits {
        return false;
    }
    match tool {
        "browser_exec" | "use_tool" => true,
        "web_fetch" | "research" | "research_read" => {
            matches!(mode, crate::PermissionMode::Ask)
        }
        _ => false,
    }
}

const EGRESS_PREVIEW_CHARS: usize = 1_200;

/// Build an [`ConfirmationRequest::External`] preview from a tool call.
pub fn confirmation_for_egress_tool(tool: &str, arguments: &str) -> ConfirmationRequest {
    // Tool dispatch parses JSON arguments too. Store that semantic value so
    // insignificant whitespace/object-key order cannot change an approval
    // digest, while every meaningful byte remains bound. `browser_exec` also
    // accepts a raw script for compatibility, so preserve invalid JSON exactly.
    let operation_arguments = serde_json::from_str(arguments)
        .unwrap_or_else(|_| serde_json::Value::String(arguments.to_string()));
    let truncated = clip(arguments, EGRESS_PREVIEW_CHARS);
    match tool {
        "use_tool" => {
            let parsed: Option<serde_json::Value> = serde_json::from_str(arguments).ok();
            let server = parsed
                .as_ref()
                .and_then(|v| v.get("server"))
                .and_then(|v| v.as_str())
                .unwrap_or("(unknown)")
                .to_string();
            let mcp_tool = parsed
                .as_ref()
                .and_then(|v| v.get("tool"))
                .and_then(|v| v.as_str())
                .unwrap_or("(unknown)")
                .to_string();
            let args_preview = parsed
                .as_ref()
                .and_then(|v| v.get("arguments"))
                .map(|v| clip(&v.to_string(), EGRESS_PREVIEW_CHARS))
                .unwrap_or_default();
            let target = format!("{server}.{mcp_tool}");
            ConfirmationRequest::External {
                tool: tool.into(),
                operation_arguments,
                summary: args_preview,
                target: target.clone(),
                mcp_grant: Some((server, mcp_tool)),
            }
        }
        "browser_exec" => {
            let script = serde_json::from_str::<serde_json::Value>(arguments)
                .ok()
                .and_then(|v| v.get("script").and_then(|s| s.as_str()).map(str::to_string))
                .unwrap_or_else(|| arguments.to_string());
            let has_eval = script.lines().any(|line| {
                line.split_whitespace()
                    .next()
                    .is_some_and(|op| op.eq_ignore_ascii_case("eval"))
            });
            let summary = if has_eval {
                format!("warning: script includes eval (arbitrary page JavaScript)\n\n{script}")
            } else {
                script
            };
            ConfirmationRequest::External {
                tool: tool.into(),
                operation_arguments,
                summary: clip(&summary, EGRESS_PREVIEW_CHARS),
                target: String::new(),
                mcp_grant: None,
            }
        }
        "web_fetch" | "research" | "research_read" => {
            let parsed: Option<serde_json::Value> = serde_json::from_str(arguments).ok();
            let target = parsed
                .as_ref()
                .and_then(|v| {
                    v.get("url")
                        .or_else(|| v.get("query"))
                        .or_else(|| v.get("research_id"))
                        .and_then(|v| v.as_str())
                })
                .unwrap_or("")
                .to_string();
            ConfirmationRequest::External {
                tool: tool.into(),
                operation_arguments,
                summary: truncated,
                target,
                mcp_grant: None,
            }
        }
        _ => ConfirmationRequest::External {
            tool: tool.into(),
            operation_arguments,
            summary: truncated,
            target: String::new(),
            mcp_grant: None,
        },
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfirmationResult {
    Approved,
    Rejected,
    Cancelled,
    /// The frontend cannot collect an interactive answer. Callers must fail closed.
    Unavailable,
    /// Confirm was persisted as pending; a human will decide later (`/inbox`).
    Parked,
    /// Free-form or numbered answer to [`ConfirmationRequest::AskUser`].
    Answer(String),
}

pub type ConfirmationFuture<'a> = Pin<Box<dyn Future<Output = ConfirmationResult> + Send + 'a>>;

/// Result of [`Ui::ask_user`]: a real decision pause, not a write approval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AskUserResult {
    Answer(String),
    Cancelled,
    Unavailable,
}

pub type AskUserFuture<'a> = Pin<Box<dyn Future<Output = AskUserResult> + Send + 'a>>;

/// Cloneable live-status sink for child explore/delegate/task jobs.
///
/// Parallel explore and background `task` workers cannot share `&mut dyn Ui`.
/// Frontends that want a live row return this from [`Ui::subagent_sink`].
pub trait SubagentSink: Send + Sync {
    fn spawned(&self, id: &str, kind: &str, description: &str, background: bool);
    fn progress(&self, id: &str, activity: &str, line: Option<&str>);
    fn finished(&self, id: &str, status: &str, elapsed_ms: u64, summary: &str);
}

/// Map a child tool call to a short activity label (`Reading lib.rs`).
pub fn subagent_activity_label(name: &str, arguments: &str) -> String {
    let name = name.rsplit(':').next().unwrap_or(name);
    let label = tool_label(name, arguments);
    let detail = label.split_once(' ').map(|(_, rest)| rest).unwrap_or("");
    match name {
        "read" => with_detail("Reading", detail),
        "grep" | "web_search" | "research" => with_detail("Searching", detail),
        "research_read" => with_detail("Reading", detail),
        "list" => with_detail("Listing", detail),
        "web_fetch" => with_detail("Fetching", detail),
        "bash" => with_detail("Run", detail),
        "write" | "edit" | "multi_edit" | "apply_patch" => with_detail("Edit", detail),
        _ if detail.is_empty() => title_case_word(name),
        _ => format!("{} {detail}", title_case_word(name)),
    }
}

fn with_detail(verb: &str, detail: &str) -> String {
    if detail.is_empty() {
        verb.to_string()
    } else {
        format!("{verb} {detail}")
    }
}

fn title_case_word(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => name.to_string(),
    }
}

/// Compact finish token for [`Ui::subagent_finished`].
pub fn subagent_finish_status(status: hi_tools::ToolStatus) -> &'static str {
    match status {
        hi_tools::ToolStatus::Succeeded => "completed",
        hi_tools::ToolStatus::Cancelled => "cancelled",
        hi_tools::ToolStatus::Denied => "denied",
        hi_tools::ToolStatus::Failed | hi_tools::ToolStatus::TimedOut => "failed",
    }
}

/// Clip a child task/description for the spawned callout and feed row.
pub fn clip_subagent_description(text: &str) -> String {
    let count = text.chars().count();
    let clipped: String = text.chars().take(72).collect();
    if count > 72 {
        format!("{clipped}…")
    } else {
        clipped
    }
}

/// Best-effort redaction for diagnostic text. It deliberately does not claim
/// perfect secret detection.
pub fn redact_debug_text(text: &str, known_secrets: &[&str]) -> String {
    let mut redacted = text.to_string();
    for secret in known_secrets.iter().copied().filter(|s| !s.is_empty()) {
        redacted = redacted.replace(secret, "[REDACTED]");
    }
    redacted
        .lines()
        .map(|line| {
            let lower = line.to_ascii_lowercase();
            if let Some(index) = lower.find("authorization:") {
                return format!("{}authorization: [REDACTED]", &line[..index]);
            }
            if let Some(index) = lower.find("bearer ") {
                let start = index + "bearer ".len();
                let end = line[start..]
                    .find(|c: char| c.is_whitespace() || matches!(c, ',' | '}' | ']'))
                    .map(|n| start + n)
                    .unwrap_or(line.len());
                let mut out = line.to_string();
                out.replace_range(start..end, "[REDACTED]");
                return out;
            }
            for separator in ['=', ':'] {
                if let Some(index) = line.find(separator) {
                    let name = line[..index]
                        .trim_end()
                        .split(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '{' | ','))
                        .next_back()
                        .unwrap_or("")
                        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                        .to_ascii_lowercase();
                    if ["key", "token", "secret", "password"]
                        .iter()
                        .any(|needle| name.contains(needle))
                    {
                        return format!("{}{} [REDACTED]", &line[..index], separator);
                    }
                }
            }
            // Backstop: redact bare provider-key-shaped tokens with no
            // `key=`/`Bearer` label (e.g. an API key printed on its own).
            redact_token_shapes(line)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Return a user-facing version of an agent status, or `None` for statuses that
/// are useful to telemetry/model steering but not to a person using hi.
///
/// The optional `explore: ` prefix is added when a read-only child agent's
/// buffered events are replayed into the parent UI, so classify after removing
/// that transport label as well.
pub fn user_facing_status(text: &str) -> Option<String> {
    let mut candidate = text.trim();
    while let Some(rest) = candidate.strip_prefix("explore: ") {
        candidate = rest.trim_start();
    }

    // Wire profiles, compatibility retries, and MoA routing are diagnostics.
    // They are deliberately kept in tracing/telemetry rather than printed on
    // every turn, where users reasonably interpret them as failures.
    if candidate.starts_with("compat:")
        || candidate.starts_with("MoA reference:")
        || candidate.starts_with("MoA aggregating:")
        || candidate.starts_with("status:MoA ")
    {
        return None;
    }

    // Capability/verification lifecycle is telemetry. A mutating tool batch
    // and every turn's workspace-repair pass would otherwise dump bookkeeping
    // into the transcript ("process_execution capability requested",
    // "verification started/finished", skip-when-unchanged).
    if candidate.ends_with(" capability requested")
        || candidate == "verification started"
        || candidate == "verification finished"
        || candidate == "verification skipped — no files changed this turn"
    {
        return None;
    }

    // Guardrails should explain the observable situation, not expose that the
    // model was being steered or how the scheduler detected the situation.
    if candidate.starts_with("background process handles were completed") {
        return Some("⚠ a background command became unavailable before completion".to_string());
    }
    if candidate.starts_with("⚠ the model kept emitting invalid tool turns") {
        return Some("⚠ tool calls were invalid, so the turn ended".to_string());
    }
    if candidate.starts_with("the model kept emitting invalid tool arguments") {
        return Some("⚠ a tool call did not match its schema, so the turn stopped".to_string());
    }
    if candidate.starts_with("DeepSeek tool arguments failed client validation") {
        return Some("retrying the tool call with a compatible schema".to_string());
    }
    if candidate.starts_with("structured tool calls kept failing") {
        return Some("retrying with a compatible tool-call format".to_string());
    }
    if candidate.starts_with("model states no file changes are needed") {
        return Some("no file changes were needed; accepting the text answer".to_string());
    }
    if candidate.starts_with("⚠ the model returned no response") {
        if candidate.contains("after retrying") {
            return Some("⚠ no response after retries".to_string());
        }
        return Some("⚠ no response yet; retrying".to_string());
    }
    if candidate.starts_with("⚠ tool call interrupted by user") {
        return Some("⚠ tool call interrupted".to_string());
    }
    if candidate.starts_with("⚠ tool scheduler could not make progress") {
        return Some(
            "⚠ a tool call could not be scheduled; continuing with the remaining work".to_string(),
        );
    }

    Some(text.to_string())
}

/// Whether a status is entirely internal and should be omitted from the UI.
pub fn is_internal_status(text: &str) -> bool {
    user_facing_status(text).is_none()
}

/// Whether a status is a replaceable live-progress update rather than a
/// durable transcript message. Frontends should render these in their
/// working chrome (or overwrite one plain-output line) instead of appending
/// one row per update.
pub fn is_live_progress_status(text: &str) -> bool {
    text.trim_start().starts_with("still working")
}

/// Remove leading warning glyphs so a renderer can add its own consistent
/// warning marker without producing `⚠ ⚠ …` for already-marked messages.
pub fn without_leading_warning_marker(mut text: &str) -> &str {
    text = text.trim_start();
    while let Some(stripped) = text.strip_prefix("⚠️").or_else(|| text.strip_prefix('⚠')) {
        text = stripped.trim_start();
    }
    text
}

/// Remove model-only process-control instructions from a tool result before it
/// is rendered. The unmodified result still goes into the model transcript.
///
/// Background tools historically returned lines such as `Use bash_output ...`.
/// Those are valid protocol instructions for the model, but showing them in a
/// user transcript makes the harness look like it is talking to itself.
pub fn user_visible_tool_result(result: &str) -> String {
    result
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("Error: no background process") {
                let id = trimmed
                    .split_once('`')
                    .and_then(|(_, rest)| rest.split_once('`').map(|(id, _)| id))
                    .unwrap_or("unknown");
                return Some(format!("background process {id} unavailable"));
            }
            let mut visible = line;
            for marker in [
                "Use bash_output with id",
                "Use bash_kill with id",
                "Do not call this again",
            ] {
                if let Some(index) = visible.find(marker) {
                    visible = visible[..index].trim_end();
                }
            }
            (!visible.trim().is_empty()).then_some(visible.to_string())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Recognizable prefixes of provider credentials. A token starting with one of
/// these (and long enough to be a real key, not just the prefix) is redacted
/// wherever it appears — even unlabeled — so a stray key in tool output or an
/// error message never lands in a transcript or log.
const CREDENTIAL_PREFIXES: &[&str] = &[
    "sk-",  // OpenAI / Anthropic-style
    "xai-", // xAI
    "pk-",  // various publishable-but-still-sensitive
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_", // GitHub tokens
    "glpat-",      // GitLab
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "xoxr-", // Slack
    "AKIA",
    "ASIA",  // AWS access key ids
    "AIza",  // Google API keys
    "ya29.", // Google OAuth
    "eyJ",   // JWT header (base64 `{"`)
];

/// Whether a character can appear inside a credential token (so we know where
/// the token ends when redacting it).
fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | '+' | '=')
}

/// Replace any provider-key-shaped token in `line` with `[REDACTED]`, preserving
/// the surrounding text and delimiters. A pure per-line pass with no allocation
/// when the line carries no such token.
fn redact_token_shapes(line: &str) -> String {
    // Fast path: no token in this line.
    if !CREDENTIAL_PREFIXES
        .iter()
        .any(|prefix| line.contains(prefix))
    {
        return line.to_string();
    }
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < line.len() {
        // Only consider a token boundary: start of line or after a non-token char.
        let at_boundary = i == 0 || !is_token_char(line[..i].chars().next_back().unwrap());
        let rest = &line[i..];
        let matched_prefix = at_boundary
            .then(|| CREDENTIAL_PREFIXES.iter().find(|p| rest.starts_with(**p)))
            .flatten();
        if let Some(prefix) = matched_prefix {
            // Extend to the full token; only redact if it's longer than the bare
            // prefix (otherwise it's just the literal word, not a key).
            let mut end = i;
            while end < line.len() && is_token_char(line[end..].chars().next().unwrap()) {
                end += line[end..].chars().next().unwrap().len_utf8();
            }
            if end - i > prefix.len() {
                out.push_str("[REDACTED]");
                i = end;
                continue;
            }
        }
        // Not a token start (or too short) — copy one char and advance.
        let ch = line[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Atomically replace a debug log with owner-only permissions.
pub fn write_private_debug_log(path: &std::path::Path, body: &str) -> std::io::Result<()> {
    use std::io::Write;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temp = path.with_extension(format!("debug-{}-{id}.tmp", std::process::id()));
    let mut options = std::fs::OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let result = (|| {
        let mut file = options.open(&temp)?;
        file.write_all(body.as_bytes())?;
        file.sync_all()?;
        std::fs::rename(&temp, path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp);
    }
    result
}

/// Receives streamed output and tool activity from a running turn.
///
/// `Send` is required because the streaming callback is handed to the provider
/// across `await` points.
pub trait Ui: Send {
    /// Receive a canonical semantic lifecycle event. Frontends that own a
    /// durable event sink may persist it; legacy frontends can ignore it.
    fn semantic_event(&mut self, _event: hi_events::RunEvent) {}
    /// Observe one concrete provider request attempt as soon as the provider
    /// reports it. This is diagnostic-only and must not be rendered as user
    /// content. [`hi_ai::WireAudit`] can carry a request body, so frontends
    /// must sanitize it before relaying or persisting the event.
    fn provider_request(&mut self, _audit: &hi_ai::WireAudit) {}
    /// A chunk of assistant text.
    fn assistant_text(&mut self, text: &str);
    /// A chunk of assistant text that answers a `/btw` side question. Distinct
    /// from [`assistant_text`](Ui::assistant_text) so a frontend can render the
    /// side-answer differently (dimmed, prefixed, or in a side pane) from main
    /// task output. Defaults to `assistant_text` so headless/frontends that
    /// don't distinguish still show it.
    fn btw_answer(&mut self, text: &str) {
        self.assistant_text(text);
    }
    /// The user asked a `/btw` side question — emitted before inspection/answer
    /// so a frontend can open a BTW pane / thread entry. Defaults to no-op so
    /// headless UIs don't dump side chrome into the main status stream.
    fn btw_question(&mut self, _question: &str) {}
    /// A read-only tool started inside a `/btw` side loop. Defaults to no-op so
    /// main-task tool chrome is not polluted; pane-aware frontends override.
    fn btw_tool_started(&mut self, _name: &str, _arguments: &str) {}
    /// A read-only tool finished inside a `/btw` side loop. Defaults to no-op.
    fn btw_tool_result(&mut self, _name: &str, _result: &str) {}
    /// The current `/btw` side answer stream finished (pane can close the entry
    /// / reset streaming state). Defaults to no-op — not the same as
    /// [`assistant_end`](Ui::assistant_end) which ends a main-task stream.
    fn btw_end(&mut self) {}
    /// A chunk of assistant reasoning/thinking.
    fn assistant_reasoning(&mut self, text: &str);
    /// The assistant's streamed message finished (before any tool calls run).
    fn assistant_end(&mut self);
    /// A tool has started running. Emitted as soon as a call is dispatched
    /// (even within a concurrent batch) so an interactive frontend can show a
    /// live "running {tool}" indicator with a timer. Unlike [`tool_call`], this
    /// is *not* a transcript line — the visible header is emitted later, paired
    /// with its result, so a reader can tell which result belongs to which call.
    /// Defaults to no-op; only the live TUI needs it.
    fn tool_started(&mut self, _name: &str, _arguments: &str) {}
    /// ID-aware tool-start event. The default preserves compatibility with
    /// frontends that only render by name; protocol adapters should override it.
    fn tool_started_id(&mut self, _id: &str, name: &str, arguments: &str) {
        self.tool_started(name, arguments);
    }
    /// A line of live output from a running tool (e.g. `bash` stdout/stderr
    /// streamed line-by-line). Emitted *during* execution, before the
    /// matching [`tool_call`]/[`tool_result`] pair. Unlike [`tool_result`],
    /// this is not a transcript line — it's a transient progress indicator
    /// that an interactive frontend can show in a live "running" panel and
    /// discard when the final result arrives. Defaults to no-op; only the
    /// live TUI needs it.
    fn tool_stream(&mut self, _name: &str, _line: &str) {}
    /// ID-aware live tool output.
    fn tool_stream_id(&mut self, _id: &str, name: &str, line: &str) {
        self.tool_stream(name, line);
    }
    /// Ask the frontend to authorize a mutation. The default fails closed so a
    /// headless frontend can never silently approve an opt-in confirmation.
    fn confirm(&mut self, _request: ConfirmationRequest) -> ConfirmationFuture<'_> {
        Box::pin(async { ConfirmationResult::Unavailable })
    }
    /// Pause for a product/design choice the tools cannot resolve.
    ///
    /// The default fails closed so headless/one-shot frontends never hang.
    /// Interactive frontends override this. Do not use this instead of
    /// keep-working when the next coding step is already known.
    fn ask_user(&mut self, _question: &str, _options: &[String]) -> AskUserFuture<'_> {
        Box::pin(async { AskUserResult::Unavailable })
    }
    /// Emit the transcript header for a tool call, immediately followed by the
    /// matching [`tool_result`]. In a concurrent batch these are emitted in
    /// completion order, as (header, result) pairs, so the two never drift
    /// apart.
    fn tool_call(&mut self, name: &str, arguments: &str);
    /// ID-aware transcript tool header.
    fn tool_call_id(&mut self, _id: &str, name: &str, arguments: &str) {
        self.tool_call(name, arguments);
    }
    /// The result of a tool call, with the tool's name so a frontend can
    /// tailor how much to show (e.g. suppress a `read`'s full file body,
    /// showing just the path that was already named in the `tool_call` line).
    fn tool_result(&mut self, name: &str, result: &str);
    /// ID-aware terminal tool result with machine-readable execution status.
    fn tool_result_id(
        &mut self,
        _id: &str,
        name: &str,
        result: &str,
        _status: hi_tools::ToolStatus,
    ) {
        self.tool_result(name, result);
    }
    /// Terminal result for a plan-producing tool. The default preserves the
    /// historical presentation: update the in-place plan tracker without
    /// adding a scrolling tool-result row. Frontends with a semantic event
    /// sink can override this to close the tool lifecycle before forwarding
    /// the plan update.
    fn plan_result_id(
        &mut self,
        _id: &str,
        _name: &str,
        _result: &str,
        _status: hi_tools::ToolStatus,
        steps: &[crate::PlanStep],
    ) {
        self.plan(steps);
    }
    /// A status note (e.g. verification progress).
    fn status(&mut self, text: &str);
    /// A prominent status warning that belongs in frontend chrome rather than
    /// the transcript or prompt. Frontends without a dedicated top-level
    /// surface fall back to their ordinary status presentation.
    fn top_status(&mut self, text: &str) {
        self.status(text);
    }
    /// Pin a strict-mode checkpoint integrity warning for the rest of a turn.
    fn checkpoint_warning(&mut self, text: &str) {
        self.status(text);
    }
    /// A prominent notice that the agent is delegating to (or finishing) a
    /// subagent — louder than an ordinary [`status`](Ui::status) so the user
    /// clearly sees a nested agent run. Defaults to a plain status; frontends
    /// override it to stand out.
    fn subagent_note(&mut self, text: &str) {
        self.status(text);
    }
    /// Live child-agent updates for frontends that can render a typed row.
    /// `None` (the default) keeps [`subagent_spawned`](Ui::subagent_spawned)
    /// / [`subagent_finished`](Ui::subagent_finished) on the note fallback.
    /// Return a cloneable sink so parallel explore jobs and background `task`
    /// workers can emit progress without sharing `&mut dyn Ui`.
    fn subagent_sink(&self) -> Option<Arc<dyn SubagentSink>> {
        None
    }
    /// A child explore/delegate/task has started. Defaults to a
    /// [`subagent_note`](Ui::subagent_note) so CLI/tests keep today's callout.
    fn subagent_spawned(&mut self, id: &str, kind: &str, description: &str, background: bool) {
        if let Some(sink) = self.subagent_sink() {
            sink.spawned(id, kind, description, background);
            return;
        }
        let _ = (id, background);
        self.subagent_note(&format!("↳ {kind} subagent: {description}"));
    }
    /// Live activity label for a running child (`Thinking`, `Reading lib.rs`).
    fn subagent_progress(&mut self, id: &str, activity: &str) {
        if let Some(sink) = self.subagent_sink() {
            sink.progress(id, activity, None);
        }
    }
    /// A child has finished. Defaults to a [`subagent_note`](Ui::subagent_note).
    fn subagent_finished(&mut self, id: &str, status: &str, elapsed_ms: u64, summary: &str) {
        if let Some(sink) = self.subagent_sink() {
            sink.finished(id, status, elapsed_ms, summary);
            return;
        }
        let _ = (id, elapsed_ms, summary);
        self.subagent_note(&format!("↳ subagent {status}"));
    }
    /// The task plan was created or updated (via the `update_plan` tool). The
    /// full step list is passed each time; a frontend shows it as a live,
    /// in-place checklist rather than a scrolling transcript echo. Defaults to
    /// ignoring it — only interactive frontends render a tracker.
    fn plan(&mut self, _steps: &[crate::PlanStep]) {}
    /// Token usage after a model round: raw user-prompt estimate and generated
    /// output for the current turn, plus the current context occupancy
    /// (`context_used` tokens against the model's `context_window`, when known)
    /// for a live fill gauge. Emitted each round so a frontend can show it climb
    /// while a turn runs. Defaults to ignoring it — only the live TUI needs it.
    fn usage(
        &mut self,
        _prompt_tokens: u64,
        _generated_tokens: u64,
        _context_used: u64,
        _context_window: Option<u32>,
        _usage_estimated: bool,
    ) {
    }
    /// Session-cumulative token usage after a model round. Used by the status
    /// chip to estimate `$` from catalog pricing. Defaults to ignoring it.
    fn session_usage(&mut self, _usage: &hi_ai::Usage) {}
    /// Latest provider rate-limit buckets observed on a model response. Emitted
    /// alongside usage when available so frontends can distinguish throttling
    /// from other request failures. Defaults to ignoring it.
    fn rate_limits(&mut self, _rate_limits: Option<hi_ai::RateLimitState>) {}
    /// End of the turn, with a prebuilt token summary line.
    fn turn_end(&mut self, summary: &str);
    /// The list of files changed during the turn (empty for a read-only or
    /// Q&A turn). Emitted just before [`turn_end`] so a frontend can show a
    /// compact "changed: a.rs, b.rs" line without needing `/diff`. Defaults
    /// to no-op — only interactive frontends render it.
    fn changed_files(&mut self, _files: &[String]) {}
    /// A predicted next user prompt for the idle input bar (Claude Code–style
    /// ghost text). Emitted after a successful turn when
    /// [`crate::AgentMemory::suggest_next_prompt`] is on. Defaults to no-op —
    /// only interactive frontends render it; accepting it is a UI concern.
    fn suggested_prompt(&mut self, _text: &str) {}
    /// The turn failed with a classified error. `kind` is a short slug
    /// (`auth`, `rate_limit`, `request`, ...) so a frontend can tailor its
    /// presentation; `message` is the raw error text; `guidance` is a
    /// user-facing remediation hint. Defaults to ignoring — frontends
    /// that already handle `turn_end` should override this for richer
    /// error UX.
    fn turn_error(&mut self, _kind: &str, _message: &str, _guidance: &str) {}
    /// An internal steering diagnostic — the agent detected a stall (re-reading
    /// already-inspected files, re-running the same command, polling a dead
    /// background handle, etc.) and injected a nudge. These are implementation
    /// details about *how* the agent steers the model, not user-facing status;
    /// real frontends ignore them (the default). Test/UI recorders capture them
    /// to assert on steering behavior.
    fn nudge(&mut self, _text: &str) {}
}

/// A no-op [`Ui`] for background subagents and other headless contexts.
///
/// Background subagents (spawned via the `task` tool) don't stream to any
/// frontend — their output is collected and returned on poll. This struct
/// implements `Ui` with all methods as no-ops.
pub struct NullUi;

impl Ui for NullUi {
    fn assistant_text(&mut self, _text: &str) {}
    fn assistant_reasoning(&mut self, _text: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _name: &str, _arguments: &str) {}
    fn tool_result(&mut self, _name: &str, _result: &str) {}
    fn status(&mut self, _text: &str) {}
    fn turn_end(&mut self, _summary: &str) {}
}

/// Blanket impl so `Box<dyn Ui>` can be used where `Ui` is expected — this
/// lets `MultiplexUi` hold a boxed primary UI (e.g. `PlainUi` or `QuietUi`)
/// alongside the `Arc<RemoteUi>`.
impl<U: Ui + ?Sized> Ui for Box<U> {
    fn semantic_event(&mut self, event: hi_events::RunEvent) {
        (**self).semantic_event(event);
    }
    fn provider_request(&mut self, audit: &hi_ai::WireAudit) {
        (**self).provider_request(audit);
    }
    fn assistant_text(&mut self, text: &str) {
        (**self).assistant_text(text);
    }
    fn btw_answer(&mut self, text: &str) {
        (**self).btw_answer(text);
    }
    fn btw_question(&mut self, question: &str) {
        (**self).btw_question(question);
    }
    fn btw_tool_started(&mut self, name: &str, arguments: &str) {
        (**self).btw_tool_started(name, arguments);
    }
    fn btw_tool_result(&mut self, name: &str, result: &str) {
        (**self).btw_tool_result(name, result);
    }
    fn btw_end(&mut self) {
        (**self).btw_end();
    }
    fn assistant_reasoning(&mut self, text: &str) {
        (**self).assistant_reasoning(text);
    }
    fn assistant_end(&mut self) {
        (**self).assistant_end();
    }
    fn tool_started(&mut self, name: &str, arguments: &str) {
        (**self).tool_started(name, arguments);
    }
    fn tool_stream(&mut self, name: &str, line: &str) {
        (**self).tool_stream(name, line);
    }
    fn confirm(&mut self, request: ConfirmationRequest) -> ConfirmationFuture<'_> {
        (**self).confirm(request)
    }
    fn ask_user(&mut self, question: &str, options: &[String]) -> AskUserFuture<'_> {
        (**self).ask_user(question, options)
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        (**self).tool_call(name, arguments);
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        (**self).tool_result(name, result);
    }
    fn plan_result_id(
        &mut self,
        id: &str,
        name: &str,
        result: &str,
        status: hi_tools::ToolStatus,
        steps: &[crate::PlanStep],
    ) {
        (**self).plan_result_id(id, name, result, status, steps);
    }
    fn status(&mut self, text: &str) {
        (**self).status(text);
    }
    fn checkpoint_warning(&mut self, text: &str) {
        (**self).checkpoint_warning(text);
    }
    fn subagent_note(&mut self, text: &str) {
        (**self).subagent_note(text);
    }
    fn plan(&mut self, steps: &[crate::PlanStep]) {
        (**self).plan(steps);
    }
    fn usage(
        &mut self,
        prompt_tokens: u64,
        generated_tokens: u64,
        context_used: u64,
        context_window: Option<u32>,
        usage_estimated: bool,
    ) {
        (**self).usage(
            prompt_tokens,
            generated_tokens,
            context_used,
            context_window,
            usage_estimated,
        );
    }
    fn session_usage(&mut self, usage: &hi_ai::Usage) {
        (**self).session_usage(usage);
    }
    fn rate_limits(&mut self, rate_limits: Option<hi_ai::RateLimitState>) {
        (**self).rate_limits(rate_limits);
    }
    fn turn_end(&mut self, summary: &str) {
        (**self).turn_end(summary);
    }
    fn top_status(&mut self, text: &str) {
        (**self).top_status(text);
    }
    fn changed_files(&mut self, files: &[String]) {
        (**self).changed_files(files);
    }
    fn turn_error(&mut self, kind: &str, message: &str, guidance: &str) {
        (**self).turn_error(kind, message, guidance);
    }
    fn nudge(&mut self, text: &str) {
        (**self).nudge(text);
    }
    fn subagent_sink(&self) -> Option<Arc<dyn SubagentSink>> {
        (**self).subagent_sink()
    }
    fn subagent_spawned(&mut self, id: &str, kind: &str, description: &str, background: bool) {
        (**self).subagent_spawned(id, kind, description, background);
    }
    fn subagent_progress(&mut self, id: &str, activity: &str) {
        (**self).subagent_progress(id, activity);
    }
    fn subagent_finished(&mut self, id: &str, status: &str, elapsed_ms: u64, summary: &str) {
        (**self).subagent_finished(id, status, elapsed_ms, summary);
    }
}

/// Classify a provider/agent error into a user-facing kind slug and
/// remediation guidance. Returns `(kind, guidance)` where `kind` is a
/// short lowercase slug and `guidance` is a one-line hint. Falls back to
/// `("error", "")` for unclassified errors.
pub fn classify_error(err: &anyhow::Error) -> (&'static str, &'static str) {
    use hi_ai::ProviderErrorKind as K;
    let external_processing_disabled_code = err
        .downcast_ref::<hi_ai::ProviderError>()
        .and_then(|error| error.code.as_deref())
        .is_some_and(|code| {
            matches!(
                code,
                "external_processing_disabled" | "external_processing_not_allowed"
            )
        });
    let external_processing_disabled_message = hi_ai::provider_error_kind(err)
        == Some(K::PolicyBlocked)
        && err
            .to_string()
            .to_ascii_lowercase()
            .contains("external processing is disabled");
    if external_processing_disabled_code || external_processing_disabled_message {
        return (
            "policy",
            "Pipe Network external processing is disabled for this credential — enable it in Pipe Network or switch providers; re-authentication will not change this",
        );
    }
    if hi_ai::provider_error_retryable(err) == Some(false)
        && matches!(
            hi_ai::provider_error_kind(err),
            Some(K::Outage | K::ModelUnavailable)
        )
    {
        return (
            "request",
            "the request was rejected and will not succeed unchanged — update the request or provider route before retrying",
        );
    }
    match hi_ai::provider_error_kind(err) {
        Some(K::Auth) if hi_ai::is_billing_or_quota_text(&err.to_string()) => (
            "auth",
            "this provider is out of credits — /login pipenetwork then /provider pipenetwork, /login x402 to pay USDC, or add credits and /retry",
        ),
        Some(K::PaymentRequired) => (
            "payment",
            "this turn needs a USDC x402 payment — /login x402 (local Solana keypair), confirm the quote (default cap $1.00 via HI_X402_MAX_USD), then /retry",
        ),
        Some(K::Auth) => (
            "auth",
            "your API key may be invalid or expired — try /provider to reconfigure, then /retry",
        ),
        Some(K::RateLimit) => (
            "rate_limit",
            "request limit reached — wait a moment, then /retry",
        ),
        Some(K::CapacityUnavailable) => (
            "capacity",
            "capacity is limited right now — wait a moment, then /retry",
        ),
        Some(K::ModelUnavailable) => (
            "request",
            "the request did not complete — wait a moment, then /retry",
        ),
        Some(K::Outage) => (
            "request",
            "the request did not complete — wait a moment, then /retry",
        ),
        Some(K::UnsupportedRequestShape) => (
            "compat",
            "the request shape was not accepted — try --compat auto, then /retry",
        ),
        Some(K::PolicyBlocked) => (
            "policy",
            "the provider blocked this request by policy — adjust the task or use an appropriate provider route",
        ),
        Some(K::UnsupportedTools) => (
            "tools",
            "tool use was not accepted — use --tool-mode chat-only for a Q&A turn",
        ),
        Some(K::RequestTooLarge) => (
            "context_full",
            "the request exceeded the model's context window — hi already retried after compacting, then dropping prior context; shorten the prompt or attached input",
        ),
        Some(K::QualityRejected) => (
            "quality",
            "the model did not gather enough evidence for this answer",
        ),
        Some(K::ToolProtocol) => (
            "tool_protocol",
            "the tool turn was invalid after automatic recovery",
        ),
        Some(K::MalformedStream) => (
            "malformed",
            "the response could not be parsed after automatic recovery",
        ),
        Some(K::EmptyCompletion) => (
            "empty",
            "the model returned an empty response after automatic recovery",
        ),
        Some(K::Other) | None => ("error", ""),
    }
}

pub fn error_counts_as_model_issue(err: &anyhow::Error) -> bool {
    !matches!(
        hi_ai::provider_error_kind(err),
        Some(
            hi_ai::ProviderErrorKind::CapacityUnavailable
                | hi_ai::ProviderErrorKind::ModelUnavailable
                | hi_ai::ProviderErrorKind::Outage
                | hi_ai::ProviderErrorKind::QualityRejected
                | hi_ai::ProviderErrorKind::ToolProtocol
                | hi_ai::ProviderErrorKind::PaymentRequired
        )
    )
}

/// A short, human-readable label for a tool call: the tool name followed by its
/// most salient argument — a path, command, or pattern — rather than a raw JSON
/// dump. `write checkers.rs` reads far better than `write({"content":"use std…})`.
/// Falls back to clipped JSON for tools we don't special-case (or unparsable args).
pub fn tool_label(name: &str, arguments: &str) -> String {
    match salient_arg(name, arguments) {
        Some(arg) => format!("{name} {arg}"),
        // Never dump raw JSON into the transcript header — unknown tools get
        // a bare name, unparsable args get a short plain note.
        None if arguments.trim().is_empty() || arguments.trim() == "{}" => name.to_string(),
        None if looks_like_json_object(arguments) => name.to_string(),
        None => format!("{name} {}", clip(&collapse_ws(arguments), 40)),
    }
}

fn looks_like_json_object(s: &str) -> bool {
    let t = s.trim();
    t.starts_with('{') || t.starts_with('[')
}

/// The one argument worth showing for a known tool, clipped to a sane width.
fn salient_arg(name: &str, arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    let str_field = |key: &str| value.get(key).and_then(|v| v.as_str());
    let label = match name {
        "read" => {
            // Multi-path reads: show "N files" instead of a single path.
            if let Some(paths) = value.get("paths").and_then(|v| v.as_array()) {
                if paths.len() == 1 {
                    paths[0].as_str()?.to_string()
                } else {
                    format!("{} files", paths.len())
                }
            } else {
                str_field("path")?.to_string()
            }
        }
        "write" | "edit" => str_field("path")?.to_string(),
        "list" => str_field("path").unwrap_or(".").to_string(),
        "grep" => {
            let pattern = clip(str_field("pattern")?, 50);
            match str_field("path") {
                Some(path) => format!("{pattern} in {}", clip(path, 40)),
                None => pattern,
            }
        }
        "bash" => {
            let command = collapse_ws(str_field("command")?);
            // Prefer the short auto-name so long commands don't flood the header.
            hi_tools::shell_title(&command)
        }
        "bash_output" | "bash_kill" => {
            // Show the shell handle as a plain name, never `{"id":"…"}`.
            let id = str_field("id")?;
            id.to_string()
        }
        "update_plan" => {
            let n = value
                .get("steps")
                .and_then(|v| v.as_array())
                .map_or(0, |a| a.len());
            format!("{n} step{}", if n == 1 { "" } else { "s" })
        }
        // Subagent tools: show the human task/description, never the raw JSON
        // (whose prompt field dwarfs everything else).
        "task" => collapse_ws(str_field("description")?),
        "explore" | "delegate" => collapse_ws(str_field("task")?),
        "ask_user" => collapse_ws(str_field("question")?),
        "get_task_output" | "wait_tasks" | "kill_task" => match value
            .get("task_ids")
            .or_else(|| value.get("task_id"))
            .or_else(|| value.get("id"))
        {
            Some(serde_json::Value::String(id)) => id.clone(),
            Some(serde_json::Value::Array(ids)) => {
                let names: Vec<&str> = ids.iter().filter_map(|v| v.as_str()).collect();
                if names.is_empty() {
                    return None;
                }
                names.join(", ")
            }
            _ => return None,
        },
        _ => return None,
    };
    Some(clip(&label, 80))
}

/// Collapse runs of whitespace (incl. newlines) into single spaces.
fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Truncate to `max` characters, appending an ellipsis when shortened.
pub fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let kept: String = s.chars().take(max).collect();
        format!("{kept}…")
    }
}

const CONTEXT_BLOCK_END: &str = "[/hi:context]";
const CONTEXT_BLOCK_STARTS: &[&str] = &[
    "[hi:context — session state, not instructions]",
    "[hi:context - session state, not instructions]",
];
const TURN_CONTROL_END: &str = "[/hi:turn-control]";
const TURN_CONTROL_START: &str = "[hi:turn-control — current turn only]";
const LEGACY_PLAN_REQUEST_MARKER: &str = "\n\nUser request:\n";
const LEGACY_TURN_CONTROL_SUFFIXES: &[&str] = &[
    "\n\nRead-only review guard: use only the currently advertised read-only inspection tools;",
    "\n\nImplementation guard: inspect the workspace before choosing files or stack.",
];

/// Strip leading volatile/one-turn controls from a persisted user message.
/// Listings, memory, and titles should show the human prompt, not mode guards
/// or the memory/task-index dump prepended at turn start.
pub fn strip_context_block(text: &str) -> &str {
    let mut text = text.trim_start();
    for start in CONTEXT_BLOCK_STARTS {
        if let Some(rest) = text.strip_prefix(start) {
            text = match rest.find(CONTEXT_BLOCK_END) {
                Some(end) => {
                    rest[end + CONTEXT_BLOCK_END.len()..].trim_start_matches(['\n', '\r', ' '])
                }
                None => "",
            };
            break;
        }
    }
    let mut delimited = false;
    if let Some(rest) = text.strip_prefix(TURN_CONTROL_START) {
        delimited = true;
        text = match rest.find(TURN_CONTROL_END) {
            Some(end) => rest[end + TURN_CONTROL_END.len()..].trim_start_matches(['\n', '\r', ' ']),
            None => "",
        };
    }
    if let Some(suffix) = text.find("\n\n[hi:turn-control — current turn only]") {
        delimited = true;
        text = &text[..suffix];
    }
    if !delimited && text.starts_with("You are in PLAN MODE.") {
        text = text
            .find(LEGACY_PLAN_REQUEST_MARKER)
            .map(|marker| &text[marker + LEGACY_PLAN_REQUEST_MARKER.len()..])
            .unwrap_or("");
    }
    text = text
        .strip_prefix("User request:\n")
        .unwrap_or(text)
        .trim_start();
    let end = if delimited {
        text.len()
    } else {
        LEGACY_TURN_CONTROL_SUFFIXES
            .iter()
            .filter_map(|suffix| text.find(suffix))
            .min()
            .unwrap_or(text.len())
    };
    text[..end].trim_end()
}

/// One-line title from a user message: drop the context block, folded stdin,
/// and code fences, then collapse whitespace.
pub fn user_prompt_title(text: &str, max: usize) -> String {
    let stripped = strip_context_block(text);
    let head = stripped
        .split("stdin:")
        .next()
        .unwrap_or(stripped)
        .split("```")
        .next()
        .unwrap_or(stripped);
    clip(&collapse_ws(head), max)
}

#[cfg(test)]
mod tests {
    use super::{
        ConfirmationRequest, classify_error, confirmation_capability, confirmation_for_egress_tool,
        egress_confirm_required, error_counts_as_model_issue, is_live_progress_status,
        redact_debug_text, subagent_activity_label, tool_label, user_facing_status,
        user_visible_tool_result, without_leading_warning_marker, write_private_debug_log,
    };
    use hi_ai::{ProviderError, ProviderErrorKind};

    #[test]
    fn auto_classifier_is_conservative() {
        assert!(
            ConfirmationRequest::FileEdit {
                path: "src/lib.rs".into(),
                diff: "+fn ok() {}\n".into(),
            }
            .safe_for_auto()
        );
        assert!(
            !ConfirmationRequest::FileEdit {
                path: ".env".into(),
                diff: "+TOKEN=x\n".into(),
            }
            .safe_for_auto()
        );
        assert!(
            !ConfirmationRequest::ShellMutation {
                command: "npm install".into(),
                cwd: ".".into(),
            }
            .safe_for_auto()
        );
        assert!(
            !ConfirmationRequest::AskUser {
                question: "which API?".into(),
                options: vec!["REST".into(), "gRPC".into()],
            }
            .safe_for_auto()
        );
        assert!(
            !ConfirmationRequest::External {
                tool: "browser_exec".into(),
                operation_arguments: serde_json::json!({
                    "script": "goto https://example.com"
                }),
                summary: "goto https://example.com".into(),
                target: String::new(),
                mcp_grant: None,
            }
            .safe_for_auto()
        );
    }

    #[test]
    fn egress_confirm_ladder_matches_ask_auto_always() {
        use crate::PermissionMode;
        assert!(!egress_confirm_required(
            PermissionMode::Always,
            false,
            "browser_exec"
        ));
        assert!(egress_confirm_required(
            PermissionMode::Auto,
            true,
            "browser_exec"
        ));
        assert!(egress_confirm_required(
            PermissionMode::Auto,
            true,
            "use_tool"
        ));
        assert!(!egress_confirm_required(
            PermissionMode::Auto,
            true,
            "web_fetch"
        ));
        assert!(egress_confirm_required(
            PermissionMode::Ask,
            true,
            "web_fetch"
        ));
        assert!(!egress_confirm_required(
            PermissionMode::Ask,
            true,
            "web_search"
        ));
    }

    #[test]
    fn browser_exec_preview_flags_eval() {
        let request = confirmation_for_egress_tool(
            "browser_exec",
            r#"{"script":"goto https://example.com\neval document.cookie"}"#,
        );
        let crate::ConfirmationRequest::External {
            summary, mcp_grant, ..
        } = request
        else {
            panic!("expected External");
        };
        assert!(summary.contains("eval"));
        assert!(summary.contains("warning"));
        assert!(mcp_grant.is_none());
    }

    #[test]
    fn use_tool_preview_offers_mcp_grant() {
        let request = confirmation_for_egress_tool(
            "use_tool",
            r#"{"server":"github","tool":"create_issue","arguments":{"title":"x"}}"#,
        );
        assert_eq!(
            request.mcp_standing_grant(),
            Some(("github", "create_issue"))
        );
    }

    #[test]
    fn external_approval_digest_binds_suffix_beyond_bounded_preview() {
        let shared = "a".repeat(1_300);
        let browser_a = serde_json::json!({
            "script": format!("# {shared}\ngoto https://safe-a.example")
        })
        .to_string();
        let browser_b = serde_json::json!({
            "script": format!("# {shared}\ngoto https://safe-b.example")
        })
        .to_string();
        assert_eq!(&browser_a[..1_200], &browser_b[..1_200]);
        let browser_a = confirmation_for_egress_tool("browser_exec", &browser_a);
        let browser_b = confirmation_for_egress_tool("browser_exec", &browser_b);
        let (
            ConfirmationRequest::External {
                summary: preview_a, ..
            },
            ConfirmationRequest::External {
                summary: preview_b, ..
            },
        ) = (&browser_a, &browser_b)
        else {
            panic!("expected External confirmations");
        };
        assert_eq!(preview_a, preview_b, "display previews stay bounded");
        let (_, digest_a) = confirmation_capability(&browser_a).unwrap();
        let (_, digest_b) = confirmation_capability(&browser_b).unwrap();
        assert_ne!(digest_a, digest_b, "hidden suffix must remain bound");

        let mcp_a = serde_json::json!({
            "server": "mail",
            "tool": "send",
            "arguments": {"body": format!("{shared}recipient-a")}
        })
        .to_string();
        let mcp_b = serde_json::json!({
            "server": "mail",
            "tool": "send",
            "arguments": {"body": format!("{shared}recipient-b")}
        })
        .to_string();
        assert_eq!(&mcp_a[..1_200], &mcp_b[..1_200]);
        let mcp_a = confirmation_for_egress_tool("use_tool", &mcp_a);
        let mcp_b = confirmation_for_egress_tool("use_tool", &mcp_b);
        let (_, digest_a) = confirmation_capability(&mcp_a).unwrap();
        let (_, digest_b) = confirmation_capability(&mcp_b).unwrap();
        assert_ne!(digest_a, digest_b, "MCP suffix must remain bound");
    }

    #[test]
    fn external_approval_digest_canonicalizes_json_arguments() {
        let first = confirmation_for_egress_tool(
            "use_tool",
            r#"{"server":"demo","tool":"echo","arguments":{"b":2,"a":1}}"#,
        );
        let reordered = confirmation_for_egress_tool(
            "use_tool",
            r#"{ "arguments": { "a": 1, "b": 2 }, "tool": "echo", "server": "demo" }"#,
        );
        let (_, first_digest) = confirmation_capability(&first).unwrap();
        let (_, reordered_digest) = confirmation_capability(&reordered).unwrap();
        assert_eq!(first_digest, reordered_digest);
    }

    #[test]
    fn labels_file_tools_by_path() {
        // The bug this fixes: write/edit/read used to dump their whole JSON
        // (content and all) into the header. Show just the path instead.
        assert_eq!(
            tool_label(
                "write",
                r#"{"path":"checkers.rs","content":"use std::fmt;\n…"}"#
            ),
            "write checkers.rs"
        );
        assert_eq!(
            tool_label(
                "edit",
                r#"{"path":"src/cli.rs","old_string":"a","new_string":"b"}"#
            ),
            "edit src/cli.rs"
        );
        assert_eq!(
            tool_label("read", r#"{"path":"Cargo.toml"}"#),
            "read Cargo.toml"
        );
        // Multi-path reads: a one-element array still names the file.
        assert_eq!(
            tool_label("read", r#"{"paths":["Cargo.toml"]}"#),
            "read Cargo.toml"
        );
        // A multi-element array collapses to "N files".
        assert_eq!(
            tool_label("read", r#"{"paths":["a.rs","b.rs","c.rs"]}"#),
            "read 3 files"
        );
    }

    #[test]
    fn labels_bash_by_command_and_grep_by_pattern() {
        assert_eq!(
            tool_label("bash", r#"{"command":"cargo  test\n  --all"}"#),
            "bash cargo test"
        );
        assert_eq!(
            tool_label("bash_output", r#"{"id":"sh_1"}"#),
            "bash_output sh_1"
        );
        assert_eq!(
            tool_label("grep", r#"{"pattern":"TODO","path":"src"}"#),
            "grep TODO in src"
        );
        assert_eq!(
            tool_label("grep", r#"{"pattern":"fn main"}"#),
            "grep fn main"
        );
        assert_eq!(tool_label("list", "{}"), "list .");
    }

    #[test]
    fn labels_subagent_tools_by_task_not_json() {
        // Subagent calls used to dump raw JSON (prompt and all) into the
        // header: task({"cost":"large","description":"Build…","prompt":"Impl…).
        assert_eq!(
            tool_label(
                "task",
                r#"{"cost":"large","description":"Build workflow run store","prompt":"Implement durable…"}"#
            ),
            "task Build workflow run store"
        );
        assert_eq!(
            tool_label("explore", r#"{"task":"Investigate workflow flags"}"#),
            "explore Investigate workflow flags"
        );
        assert_eq!(
            tool_label("delegate", r#"{"task":"Wire the scheduler"}"#),
            "delegate Wire the scheduler"
        );
        assert_eq!(
            tool_label("get_task_output", r#"{"task_ids":["task_1","task_2"]}"#),
            "get_task_output task_1, task_2"
        );
        assert_eq!(
            tool_label(
                "ask_user",
                r#"{"question":"Which transport should the public API use?"}"#
            ),
            "ask_user Which transport should the public API use?"
        );
    }

    #[test]
    fn internal_statuses_are_hidden_or_humanized() {
        assert!(
            user_facing_status("compat: deepseek profile=gateway protocol=auto strict=false")
                .is_none()
        );
        assert!(user_facing_status("MoA aggregating: coder").is_none());
        assert!(user_facing_status("process_execution capability requested").is_none());
        assert!(user_facing_status("verification started").is_none());
        assert!(user_facing_status("verification finished").is_none());
        assert!(user_facing_status("verification skipped — no files changed this turn").is_none());
        let leftover = user_facing_status("3/9 remaining — wire the scheduler").unwrap();
        assert_eq!(leftover, "3/9 remaining — wire the scheduler");
        assert_eq!(
            user_facing_status(
                "DeepSeek tool arguments failed client validation; retrying once without strict schemas"
            ),
            Some("retrying the tool call with a compatible schema".to_string())
        );
        assert_eq!(
            user_facing_status("⚠ the model returned no response after retrying — try /retry."),
            Some("⚠ no response after retries".to_string())
        );
    }

    #[test]
    fn live_progress_statuses_are_replaceable() {
        assert!(is_live_progress_status("still working — checking"));
        assert!(is_live_progress_status("  still working — retrying"));
        assert!(!is_live_progress_status("working status: complete"));
    }

    #[test]
    fn warning_marker_is_normalized_once() {
        assert_eq!(
            without_leading_warning_marker("⚠ already marked"),
            "already marked"
        );
        assert_eq!(
            without_leading_warning_marker("⚠️ already marked"),
            "already marked"
        );
        assert_eq!(
            without_leading_warning_marker("⚠️ ⚠ already marked"),
            "already marked"
        );
        assert_eq!(
            without_leading_warning_marker("plain notice"),
            "plain notice"
        );
    }

    #[test]
    fn model_only_background_instructions_are_removed_from_display_results() {
        let result = user_visible_tool_result(
            "Started cargo test (sh_1). Use bash_output with id sh_1 for progress; Use bash_kill with id sh_1 to stop.",
        );
        assert_eq!(result, "Started cargo test (sh_1).");

        let missing = user_visible_tool_result(
            "Error: no background process `git-status_1` — no background processes are running at all. Do not call this again; continue the task with other tools.",
        );
        assert_eq!(missing, "background process git-status_1 unavailable");
    }

    #[test]
    fn never_dumps_raw_json_into_labels() {
        // Unknown tools with JSON args: bare name only — no brace soup in the TUI.
        assert_eq!(tool_label("frobnicate", r#"{"x":  1}"#), "frobnicate");
        // Unparsable plain args still show a short plain note.
        assert_eq!(tool_label("write", "not json"), "write not json");
    }

    #[test]
    fn capacity_limit_is_not_a_model_quality_issue() {
        let err: anyhow::Error = ProviderError::new(
            ProviderErrorKind::CapacityUnavailable,
            "API error 409: capacity temporarily unavailable",
        )
        .into();

        let (kind, guidance) = classify_error(&err);

        assert_eq!(kind, "capacity");
        assert!(guidance.contains("capacity is limited"));
        assert!(!error_counts_as_model_issue(&err));
    }

    #[test]
    fn grok_credit_exhaustion_points_at_pipenetwork_not_a_dead_key() {
        let err: anyhow::Error = ProviderError::new(
            ProviderErrorKind::Auth,
            "API error 403 Forbidden: You have run out of credits or need a Grok subscription. Add credits at https://grok.com/?_s=usage",
        )
        .into();
        let (kind, guidance) = classify_error(&err);
        assert_eq!(kind, "auth");
        assert!(guidance.contains("/login pipenetwork"));
        assert!(guidance.contains("/provider pipenetwork"));
        assert!(!guidance.contains("API key may be invalid"));
    }

    #[test]
    fn x402_payment_required_points_at_usdc_login() {
        let err: anyhow::Error = ProviderError::new(
            ProviderErrorKind::PaymentRequired,
            "x402 quote $1.20 exceeds HI_X402_MAX_USD $1.00",
        )
        .into();
        let (kind, guidance) = classify_error(&err);
        assert_eq!(kind, "payment");
        assert!(guidance.contains("/login x402"));
        assert!(guidance.contains("HI_X402_MAX_USD"));
        assert!(!error_counts_as_model_issue(&err));
    }

    #[test]
    fn pipe_external_processing_disabled_explains_account_capability_not_key_failure() {
        let err: anyhow::Error = ProviderError::new(
            ProviderErrorKind::PolicyBlocked,
            "API error 403 Forbidden: external processing is disabled for this request",
        )
        .with_api_contract(
            Some("external_processing_disabled".into()),
            Some(false),
            None,
        )
        .into();

        let (kind, guidance) = classify_error(&err);

        assert_eq!(kind, "policy");
        assert!(guidance.contains("external processing is disabled"));
        assert!(guidance.contains("re-authentication will not change this"));
        assert!(!guidance.contains("API key may be invalid"));
    }

    #[test]
    fn route_rejection_is_not_reported_as_capacity_or_incomplete_turn() {
        let err: anyhow::Error = ProviderError::new(
            ProviderErrorKind::ModelUnavailable,
            "model temporarily unavailable",
        )
        .into();

        let (kind, guidance) = classify_error(&err);

        assert_eq!(kind, "request");
        assert!(!guidance.contains("/model"));
        assert!(!guidance.contains("switch"));
        assert!(!guidance.contains("capacity"));
        assert!(!error_counts_as_model_issue(&err));
    }

    #[test]
    fn explicitly_non_retryable_service_error_does_not_recommend_retrying() {
        let err: anyhow::Error = ProviderError::new(
            ProviderErrorKind::Outage,
            "API rejected the provider payload",
        )
        .with_api_contract(Some("service_unavailable".to_string()), Some(false), None)
        .into();
        let (kind, guidance) = classify_error(&err);
        assert_eq!(kind, "request");
        assert!(guidance.contains("will not succeed unchanged"));
    }

    #[test]
    fn external_processing_policy_code_has_actionable_guidance() {
        let err: anyhow::Error = ProviderError::new(
            ProviderErrorKind::PolicyBlocked,
            "request rejected by account policy",
        )
        .with_api_contract(
            Some("external_processing_disabled".to_string()),
            Some(false),
            None,
        )
        .with_http_status(Some(403))
        .into();

        let (kind, guidance) = classify_error(&err);
        assert_eq!(kind, "policy");
        assert!(guidance.contains("external processing is disabled"));
        assert!(guidance.contains("re-authentication will not change this"));
        assert!(!guidance.contains("API key may be invalid"));
    }

    #[test]
    fn soft_protocol_errors_are_not_model_quality_issues() {
        for (kind, expected_label) in [
            (ProviderErrorKind::QualityRejected, "quality"),
            (ProviderErrorKind::ToolProtocol, "tool_protocol"),
        ] {
            let err: anyhow::Error =
                ProviderError::new(kind, "model output did not satisfy the tool protocol").into();

            let (label, guidance) = classify_error(&err);

            assert_eq!(label, expected_label);
            assert!(!guidance.is_empty());
            assert!(
                !guidance.contains("/retry"),
                "model-glitch guidance must not ask the user to retry: {guidance}"
            );
            assert!(!error_counts_as_model_issue(&err));
        }
    }

    #[test]
    fn empty_and_malformed_guidance_does_not_ask_the_user_to_retry() {
        for kind in [
            ProviderErrorKind::MalformedStream,
            ProviderErrorKind::EmptyCompletion,
            ProviderErrorKind::QualityRejected,
        ] {
            let err: anyhow::Error = ProviderError::new(kind, "glitch").into();
            let (_, guidance) = classify_error(&err);
            assert!(
                !guidance.contains("/retry"),
                "{kind:?} guidance must not ask the user to retry: {guidance}"
            );
            assert!(!guidance.is_empty());
        }
    }

    #[test]
    fn debug_redaction_covers_known_and_structured_secrets() {
        let raw = "Authorization: Bearer abc\napi_key=abc\nlease_token: lease-123\npassword = hunter2\nplain ok";
        let clean = redact_debug_text(raw, &["abc", "lease-123"]);
        assert!(!clean.contains("abc"));
        assert!(!clean.contains("lease-123"));
        assert!(!clean.contains("hunter2"));
        assert!(clean.contains("plain ok"));
    }

    #[test]
    fn debug_redaction_catches_bare_provider_key_shapes() {
        // No `key=`/`Bearer` label — just a credential sitting in output.
        let raw = "loaded key sk-ABCDEF0123456789abcdef for the run\n\
                   token ghp_0123456789ABCDEFabcdef and AKIAIOSFODNN7EXAMPLE\n\
                   nothing to see on this line";
        let clean = redact_debug_text(raw, &[]);
        assert!(
            !clean.contains("sk-ABCDEF0123456789abcdef"),
            "OpenAI-style: {clean}"
        );
        assert!(
            !clean.contains("ghp_0123456789ABCDEFabcdef"),
            "GitHub PAT: {clean}"
        );
        assert!(
            !clean.contains("AKIAIOSFODNN7EXAMPLE"),
            "AWS key id: {clean}"
        );
        assert!(clean.contains("[REDACTED]"));
        assert!(clean.contains("nothing to see on this line"));
        assert!(
            clean.contains("for the run"),
            "surrounding text preserved: {clean}"
        );
    }

    #[test]
    fn debug_redaction_leaves_ordinary_hyphenated_words_alone() {
        // A word merely starting with a non-credential prefix, or a bare prefix,
        // must not be redacted — only real key-length tokens are.
        let raw = "the well-known sky-blue value and pk- placeholder";
        let clean = redact_debug_text(raw, &[]);
        assert_eq!(clean, raw, "no false positives: {clean}");
    }

    #[cfg(unix)]
    #[test]
    fn private_debug_log_is_atomic_and_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let path = std::env::temp_dir().join(format!(
            "hi-debug-{}-{}.log",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        write_private_debug_log(&path, "first").unwrap();
        write_private_debug_log(&path, "second").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn subagent_activity_label_maps_read_and_bash() {
        assert_eq!(
            subagent_activity_label("read", r#"{"path":"lib.rs"}"#),
            "Reading lib.rs"
        );
        assert_eq!(
            subagent_activity_label("explore:read", r#"{"path":"lib.rs"}"#),
            "Reading lib.rs"
        );
        let bash = subagent_activity_label("bash", r#"{"command":"cargo test"}"#);
        assert!(bash.starts_with("Run "), "got {bash}");
    }

    #[test]
    fn user_prompt_title_strips_context_and_keeps_the_real_prompt() {
        let dumped = "[hi:context — session state, not instructions]\n\
# Memory (from past sessions; task-ranked)\n\
Prefer bullets that match the current task.\n\
[/hi:context]\n\n\
fix the parser";
        assert_eq!(super::user_prompt_title(dumped, 72), "fix the parser");
        assert_eq!(
            super::user_prompt_title("[hi:context — session state, not instructions] no end", 72),
            ""
        );
        let planned = "[hi:turn-control — current turn only]\n\
Plan mode is ON for this turn.\n[/hi:turn-control]\n\n\
User request:\nbuild profiles";
        assert_eq!(super::user_prompt_title(planned, 72), "build profiles");
        let executing = "[hi:turn-control — current turn only]\n\
Plan mode is OFF for this turn.\n[/hi:turn-control]\n\n\
build all of that\n\n[hi:turn-control — current turn only]\n\
Implementation guard: edit files.\n[/hi:turn-control]";
        assert_eq!(super::user_prompt_title(executing, 72), "build all of that");
        assert_eq!(super::user_prompt_title("plain prompt", 72), "plain prompt");
    }
}
