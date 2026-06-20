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
    /// End of the turn, with a prebuilt token/cost summary line.
    fn turn_end(&mut self, summary: &str);
}

/// A one-line, length-capped preview of a tool call's JSON arguments.
pub fn preview_args(arguments: &str) -> String {
    let collapsed = arguments.split_whitespace().collect::<Vec<_>>().join(" ");
    clip(&collapsed, 100)
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
