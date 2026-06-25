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
    /// A tool is about to run, with its raw JSON arguments.
    fn tool_call(&mut self, name: &str, arguments: &str);
    /// The result of a tool call.
    fn tool_result(&mut self, result: &str);
    /// A status note (e.g. verification progress).
    fn status(&mut self, text: &str);
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
    /// End of the turn, with a prebuilt token/cost summary line.
    fn turn_end(&mut self, summary: &str);
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
        "read" | "write" | "edit" => str_field("path")?.to_string(),
        "list" => str_field("path").unwrap_or(".").to_string(),
        "grep" => {
            let pattern = clip(str_field("pattern")?, 50);
            match str_field("path") {
                Some(path) => format!("{pattern} in {}", clip(path, 40)),
                None => pattern,
            }
        }
        "bash" => collapse_ws(str_field("command")?),
        "update_plan" => {
            let n = value.get("steps").and_then(|v| v.as_array()).map_or(0, |a| a.len());
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
    use super::tool_label;

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
}
