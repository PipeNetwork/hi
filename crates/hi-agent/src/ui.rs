//! The output seam between the agent loop and whatever renders it.
//!
//! The agent emits raw events through [`Ui`]; each frontend (plain stdout, the
//! TUI) decides how to format them. This keeps the loop free of `print!` and
//! terminal concerns.

use std::future::Future;
use std::pin::Pin;

/// A mutation that requires an explicit user decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfirmationRequest {
    FileEdit { path: String, diff: String },
    ShellMutation { command: String, cwd: String },
    DelegateApply { summary: String, diff: String },
    MissingCheckpoint { reason: String },
}

impl ConfirmationRequest {
    pub fn title(&self) -> &'static str {
        match self {
            Self::FileEdit { .. } => "Confirm file edit",
            Self::ShellMutation { .. } => "Confirm shell mutation",
            Self::DelegateApply { .. } => "Confirm delegated changes",
            Self::MissingCheckpoint { .. } => "Continue without /undo?",
        }
    }

    pub fn details(&self) -> String {
        match self {
            Self::FileEdit { path, diff } => format!("file: {path}\n\n{diff}"),
            Self::ShellMutation { command, cwd } => format!(
                "working directory: {cwd}\nwarning: this command is likely to mutate the workspace\n\n$ {command}"
            ),
            Self::DelegateApply { summary, diff } => format!("{summary}\n\n{diff}"),
            Self::MissingCheckpoint { reason } => format!(
                "A checkpoint could not be created: {reason}\n\nChanges in this turn will not be available to /undo."
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfirmationResult {
    Approved,
    Rejected,
    Cancelled,
    /// The frontend cannot collect an interactive answer. Callers must fail closed.
    Unavailable,
}

pub type ConfirmationFuture<'a> = Pin<Box<dyn Future<Output = ConfirmationResult> + Send + 'a>>;

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
            line.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
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
    /// A chunk of assistant text.
    fn assistant_text(&mut self, text: &str);
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
    /// A line of live output from a running tool (e.g. `bash` stdout/stderr
    /// streamed line-by-line). Emitted *during* execution, before the
    /// matching [`tool_call`]/[`tool_result`] pair. Unlike [`tool_result`],
    /// this is not a transcript line — it's a transient progress indicator
    /// that an interactive frontend can show in a live "running" panel and
    /// discard when the final result arrives. Defaults to no-op; only the
    /// live TUI needs it.
    fn tool_stream(&mut self, _name: &str, _line: &str) {}
    /// Ask the frontend to authorize a mutation. The default fails closed so a
    /// headless frontend can never silently approve an opt-in confirmation.
    fn confirm(&mut self, _request: ConfirmationRequest) -> ConfirmationFuture<'_> {
        Box::pin(async { ConfirmationResult::Unavailable })
    }
    /// Emit the transcript header for a tool call, immediately followed by the
    /// matching [`tool_result`]. In a concurrent batch these are emitted in
    /// completion order, as (header, result) pairs, so the two never drift
    /// apart.
    fn tool_call(&mut self, name: &str, arguments: &str);
    /// The result of a tool call, with the tool's name so a frontend can
    /// tailor how much to show (e.g. suppress a `read`'s full file body,
    /// showing just the path that was already named in the `tool_call` line).
    fn tool_result(&mut self, name: &str, result: &str);
    /// A status note (e.g. verification progress).
    fn status(&mut self, text: &str);
    /// Pin a warning for the rest of a turn that is proceeding without /undo.
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

/// Blanket impl so `Box<dyn Ui>` can be used where `Ui` is expected — this
/// lets `MultiplexUi` hold a boxed primary UI (e.g. `PlainUi` or `QuietUi`)
/// alongside the `Arc<RemoteUi>`.
impl<U: Ui + ?Sized> Ui for Box<U> {
    fn assistant_text(&mut self, text: &str) {
        (**self).assistant_text(text);
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
    fn tool_call(&mut self, name: &str, arguments: &str) {
        (**self).tool_call(name, arguments);
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        (**self).tool_result(name, result);
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
    fn rate_limits(&mut self, rate_limits: Option<hi_ai::RateLimitState>) {
        (**self).rate_limits(rate_limits);
    }
    fn turn_end(&mut self, summary: &str) {
        (**self).turn_end(summary);
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
}

/// Classify a provider/agent error into a user-facing kind slug and
/// remediation guidance. Returns `(kind, guidance)` where `kind` is a
/// short lowercase slug and `guidance` is a one-line hint. Falls back to
/// `("error", "")` for unclassified errors.
pub fn classify_error(err: &anyhow::Error) -> (&'static str, &'static str) {
    use hi_ai::ProviderErrorKind as K;
    match hi_ai::provider_error_kind(err) {
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
        Some(K::UnsupportedTools) => (
            "tools",
            "tool use was not accepted — use --tool-mode chat-only for a Q&A turn",
        ),
        Some(K::RequestTooLarge) => (
            "context_full",
            "the request exceeded the model's context window — try /compact to reclaim room, then /retry",
        ),
        Some(K::QualityRejected) => (
            "quality",
            "the model did not gather enough evidence for this answer — /retry will ask it to inspect more concrete files",
        ),
        Some(K::ToolProtocol) => (
            "tool_protocol",
            "the tool turn was invalid — /retry usually fixes this",
        ),
        Some(K::MalformedStream) => (
            "malformed",
            "the response could not be parsed — /retry usually fixes this",
        ),
        Some(K::EmptyCompletion) => (
            "empty",
            "the model returned an empty response — /retry usually fixes this",
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
                | hi_ai::ProviderErrorKind::QualityRejected
                | hi_ai::ProviderErrorKind::ToolProtocol
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
        None => format!("{name}({})", clip(&collapse_ws(arguments), 80)),
    }
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
        "bash" => collapse_ws(str_field("command")?),
        "bash_output" | "bash_kill" => str_field("id")?.to_string(),
        "update_plan" => {
            let n = value
                .get("steps")
                .and_then(|v| v.as_array())
                .map_or(0, |a| a.len());
            format!("{n} step{}", if n == 1 { "" } else { "s" })
        }
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

#[cfg(test)]
mod tests {
    use super::{
        classify_error, error_counts_as_model_issue, redact_debug_text, tool_label,
        write_private_debug_log,
    };
    use hi_ai::{ProviderError, ProviderErrorKind};

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
            "bash cargo test --all"
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
    fn falls_back_to_clipped_json_for_unknown_or_unparsable() {
        assert_eq!(
            tool_label("frobnicate", r#"{"x":  1}"#),
            "frobnicate({\"x\": 1})"
        );
        // Unparsable args still produce something rather than panicking.
        assert_eq!(tool_label("write", "not json"), "write(not json)");
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
            assert!(!error_counts_as_model_issue(&err));
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
}
