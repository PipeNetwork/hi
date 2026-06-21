//! Slash-command parsing, shared by every frontend.

/// A recognized in-session command. Frontends decide how to act on each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Help,
    /// Reset the conversation, keeping only the system prompt.
    Clear,
    /// Switch the model for subsequent turns (empty = report current).
    Model(String),
    /// Show cumulative token usage.
    Tokens,
    /// Show, set, or clear the verify command turns iterate against. Empty =
    /// show; `off`/`none`/`clear` = disable; anything else = set.
    Verify(String),
    /// Show what's changed in the working tree (git diff).
    Diff,
    /// Summarize the conversation so far and reset the live context to it.
    Compact,
    /// Re-run the last user message (after truncating its previous attempt).
    Retry,
    /// Revert the file changes the last turn made (from its git checkpoint).
    Undo,
    Quit,
    /// A `/word` that isn't recognized.
    Unknown(String),
}

/// Parse a line as a command. Returns `None` for ordinary input (anything not
/// starting with `/`).
pub fn parse(line: &str) -> Option<Command> {
    let line = line.trim();
    let rest = line.strip_prefix('/')?;
    let mut parts = rest.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("").trim().to_string();
    Some(match name {
        "help" | "h" | "?" => Command::Help,
        "clear" | "new" => Command::Clear,
        "model" | "m" => Command::Model(arg),
        "tokens" | "usage" | "cost" => Command::Tokens,
        "verify" | "test" => Command::Verify(arg),
        "diff" | "changes" => Command::Diff,
        "compact" => Command::Compact,
        "retry" | "redo" => Command::Retry,
        "undo" | "revert" => Command::Undo,
        "exit" | "quit" | "q" => Command::Quit,
        other => Command::Unknown(other.to_string()),
    })
}

/// Help text listing the available commands.
pub const HELP: &str = "\
commands:
  /help              show this help
  /model [id]        show or switch the model
  /verify [cmd|off]  show/set/clear the test command turns iterate against
  /diff              show what files have changed (git diff)
  /compact           summarize the conversation to reclaim context
  /retry             re-run your last message
  /undo              revert the file changes from the last turn
  /tokens            cumulative token usage this session
  /clear             start a fresh conversation
  /exit              quit";

#[cfg(test)]
mod tests {
    use super::{Command, parse};

    #[test]
    fn parses_commands_and_ignores_plain_input() {
        assert_eq!(parse("hello there"), None);
        assert_eq!(parse("/help"), Some(Command::Help));
        assert_eq!(parse("  /q "), Some(Command::Quit));
        assert_eq!(
            parse("/model gpt-4o"),
            Some(Command::Model("gpt-4o".into()))
        );
        assert_eq!(parse("/model"), Some(Command::Model(String::new())));
        assert_eq!(
            parse("/verify cargo test"),
            Some(Command::Verify("cargo test".into()))
        );
        assert_eq!(parse("/verify"), Some(Command::Verify(String::new())));
        assert_eq!(parse("/diff"), Some(Command::Diff));
        assert_eq!(parse("/compact"), Some(Command::Compact));
        assert_eq!(parse("/redo"), Some(Command::Retry));
        assert_eq!(parse("/undo"), Some(Command::Undo));
        assert_eq!(parse("/bogus"), Some(Command::Unknown("bogus".into())));
    }
}
