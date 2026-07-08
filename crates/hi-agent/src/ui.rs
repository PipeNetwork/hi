//! The output seam between the agent loop and whatever renders it.
//!
//! The agent emits raw events through [`Ui`]; each frontend (plain stdout, the
//! TUI) decides how to format them. This keeps the loop free of `print!` and
//! terminal concerns.

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
    /// Ask the user to confirm a file edit. `path` is the file being changed;
    /// `diff` is a unified-diff preview. Returns `true` to approve, `false` to
    /// skip. Defaults to auto-approve (non-interactive). Only interactive
    /// frontends override this to actually prompt.
    fn confirm_edit(&mut self, _path: &str, _diff: &str) -> bool {
        true
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
    /// Token usage after a model round: cumulative session
    /// `input_tokens`/`output_tokens`, plus the current context occupancy
    /// (`context_used` tokens against the model's `context_window`, when known)
    /// for a live fill gauge. Emitted each round so a frontend can show it climb
    /// while a turn runs. Defaults to ignoring it — only the live TUI needs it.
    fn usage(
        &mut self,
        _input_tokens: u64,
        _output_tokens: u64,
        _context_used: u64,
        _context_window: Option<u32>,
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
    use super::{classify_error, error_counts_as_model_issue, tool_label};
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
}
