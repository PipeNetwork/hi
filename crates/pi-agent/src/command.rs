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
        "exit" | "quit" | "q" => Command::Quit,
        other => Command::Unknown(other.to_string()),
    })
}

/// Help text listing the available commands.
pub const HELP: &str = "\
commands:
  /help              show this help
  /model [id]        show or switch the model
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
        assert_eq!(parse("/model gpt-4o"), Some(Command::Model("gpt-4o".into())));
        assert_eq!(parse("/model"), Some(Command::Model(String::new())));
        assert_eq!(parse("/bogus"), Some(Command::Unknown("bogus".into())));
    }
}
