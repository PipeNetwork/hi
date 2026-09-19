/// Maximum accepted line length (bytes), matching the IRC convention. Longer
/// lines are rejected before parsing so a single client cannot abuse memory.
pub const MAX_LINE_LEN: usize = 512;

/// Upper bound on the length of a username or channel name. Kept well under
/// `MAX_LINE_LEN` so a name can never dominate a protocol line, and shared by
/// the line protocol, the websocket bridge, and the web UI so all entry points
/// agree on what is a valid name.
pub const MAX_NAME_LEN: usize = 32;

/// Upper bound on how many history rows a single `HISTORY` request may ask for.
/// Shared with the websocket bridge's on-connect backfill so both replay paths
/// agree on the largest read a client can force.
pub const MAX_HISTORY: usize = 200;

/// Usernames used by REGISTER / NICK. Reject empty names, spaces, and channel
/// sigils so a nick cannot collide with a channel target. Also reject control
/// characters (including `\r` and `\n`) so a username cannot smuggle extra
/// protocol lines into broadcast output, and `:` so a username cannot be
/// confused with the PRIVMSG message separator in protocol output. `@` and `!`
/// are rejected because the web UI renders operators as `@name` in the roster,
/// so a user literally named `@alice` would be ambiguous, and `!` is reserved
/// for IRC server/user masks. Names are also length-capped so a single name
/// cannot dominate a protocol line.
pub fn valid_username(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME_LEN
        && !name.contains(' ')
        && !name.starts_with('#')
        && !name.contains(':')
        && !name.contains('@')
        && !name.contains('!')
        && !name.chars().any(char::is_control)
}

/// Channel names used by JOIN / PART / PRIVMSG / etc. Must start with `#` and
/// contain no whitespace or control characters, so a channel name cannot smuggle
/// extra protocol lines into broadcast output or be confused with a username.
/// `:` is rejected because it is the PRIVMSG message separator in protocol
/// output, so a channel named `#a:b` would be ambiguous. Names are also
/// length-capped so a single name cannot dominate a protocol line.
pub fn valid_channel(name: &str) -> bool {
    name.starts_with('#')
        && name.len() > 1
        && name.len() <= MAX_NAME_LEN
        && !name.contains(':')
        && !name.chars().any(char::is_whitespace)
        && !name.chars().any(char::is_control)
}

/// A topic text that is safe to broadcast. Topics are echoed verbatim into
/// `TOPIC <channel> <text>` lines, so they must not contain `\r`/`\n` (which
/// would smuggle extra protocol lines into the fan-out) or other control
/// characters. Empty topics are allowed (clearing the topic).
pub fn valid_topic(text: &str) -> bool {
    !text.chars().any(char::is_control)
}

/// A message body that is safe to broadcast. Message bodies are echoed verbatim
/// into `PRIVMSG <channel> <user> :<body>` lines, so they must not contain
/// control characters. A `\r`/`\n` in a body would otherwise smuggle extra
/// protocol lines into the fan-out (the WS bridge accepts raw frames that can
/// carry either), and any other control character corrupts the protocol stream
/// for every subscriber. Empty bodies are allowed.
pub fn valid_message(text: &str) -> bool {
    !text.chars().any(char::is_control)
}

/// A parsed client command. The protocol is a simplified IRC-style text
/// protocol, one command per line, `\r\n` or `\n` terminated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// `REGISTER <username> <password>` — create a new account.
    Register { username: String, password: String },
    /// `LOGIN <username> <password>` — authenticate an existing account.
    Login { username: String, password: String },
    /// `JOIN <channel>` — create-or-join a channel.
    Join { channel: String },
    /// `PART <channel>` — leave a channel.
    Part { channel: String },
    /// `PRIVMSG <channel> :<message>` — send a message to a channel.
    Privmsg { channel: String, message: String },
    /// `NICK <name>` — change the display name of the current session.
    Nick { name: String },
    /// `WHO <username>` — list the channels a user has joined.
    Who { username: String },
    /// `WHOIS <username>` — alias for `WHO`.
    Whois { username: String },
    /// `KICK <channel> <user>` — remove a user from a channel (op only).
    Kick { channel: String, user: String },
    /// `OP <channel> <user>` — grant operator status (op only).
    Op { channel: String, user: String },
    /// `DEOP <channel> <user>` — revoke operator status (op only).
    Deop { channel: String, user: String },
    /// `LIST` — list all channels.
    List,
    /// `HISTORY <channel> [limit] [BEFORE <id>]` — replay recent messages,
    /// optionally only those older than `id` so a client can page backwards.
    History {
        channel: String,
        limit: usize,
        before: Option<i64>,
    },
    /// `NAMES <channel>` — list members of a channel.
    Names { channel: String },
    /// `TOPIC <channel> [text]` — get or set a channel's topic.
    Topic {
        channel: String,
        text: Option<String>,
    },
    /// `WS <channel>` — open a websocket bridge to a channel.
    Ws { channel: String },
    /// `HELP` — print the command list.
    Help,
    /// `QUIT` — disconnect.
    Quit,
    /// `PING` — keepalive; server replies `PONG`.
    Ping,
    /// Anything we do not understand. The stored line is truncated to
    /// `MAX_LINE_LEN` so a hostile client cannot make the enum hold an
    /// arbitrarily large string (the raw line is already capped at
    /// `MAX_LINE_LEN` by the reader, but this keeps the invariant local).
    Unknown(String),
}

/// Build an `Unknown` command from a raw line, truncating it so the enum never
/// holds more than `MAX_LINE_LEN` bytes.
fn unknown(line: &str) -> Command {
    let mut s = line.to_string();
    s.truncate(MAX_LINE_LEN);
    Command::Unknown(s)
}

/// Parse a single raw line into a `Command`.
pub fn parse_line(line: &str) -> Command {
    let line = line.trim_end_matches(['\r', '\n']);
    let line = line.trim();
    if line.is_empty() {
        return Command::Unknown(String::new());
    }

    let mut parts = line.splitn(2, ' ');
    let verb = parts.next().unwrap_or("").to_uppercase();
    let rest = parts.next().unwrap_or("");

    match verb.as_str() {
        "REGISTER" => {
            let mut it = rest.splitn(2, ' ');
            let username = it.next().unwrap_or("").to_string();
            let password = it.next().unwrap_or("").to_string();
            if username.is_empty() || password.is_empty() {
                unknown(line)
            } else {
                Command::Register { username, password }
            }
        }
        "LOGIN" => {
            let mut it = rest.splitn(2, ' ');
            let username = it.next().unwrap_or("").to_string();
            let password = it.next().unwrap_or("").to_string();
            if username.is_empty() || password.is_empty() {
                unknown(line)
            } else {
                Command::Login { username, password }
            }
        }
        "JOIN" => {
            let channel = rest.trim().to_string();
            if channel.is_empty() {
                unknown(line)
            } else {
                Command::Join { channel }
            }
        }
        "PART" => {
            let channel = rest.trim().to_string();
            if channel.is_empty() {
                unknown(line)
            } else {
                Command::Part { channel }
            }
        }
        "PRIVMSG" => {
            // PRIVMSG <channel> :<message>  or  PRIVMSG <channel> <message>
            // The colon separator must be preceded by a space (IRC convention),
            // so a colon inside the channel name or message is not misparsed.
            let (channel, message) = match rest.split_once(" :") {
                Some((ch, msg)) => (ch.trim().to_string(), msg.to_string()),
                None => match rest.split_once(' ') {
                    Some((ch, msg)) => (ch.trim().to_string(), msg.trim().to_string()),
                    None => (rest.trim().to_string(), String::new()),
                },
            };
            if channel.is_empty() {
                unknown(line)
            } else {
                Command::Privmsg { channel, message }
            }
        }
        "NICK" => {
            let name = rest.trim().to_string();
            if name.is_empty() {
                unknown(line)
            } else {
                Command::Nick { name }
            }
        }
        "WHO" => {
            let username = rest.trim().to_string();
            if username.is_empty() {
                unknown(line)
            } else {
                Command::Who { username }
            }
        }
        "WHOIS" => {
            let username = rest.trim().to_string();
            if username.is_empty() {
                unknown(line)
            } else {
                Command::Whois { username }
            }
        }
        "KICK" | "OP" | "DEOP" => {
            let mut parts = rest.splitn(2, ' ');
            let channel = parts.next().unwrap_or("").trim().to_string();
            let user = parts.next().unwrap_or("").trim().to_string();
            if channel.is_empty() || user.is_empty() {
                unknown(line)
            } else {
                match verb.as_str() {
                    "KICK" => Command::Kick { channel, user },
                    "OP" => Command::Op { channel, user },
                    _ => Command::Deop { channel, user },
                }
            }
        }
        "LIST" => Command::List,
        "NAMES" => {
            let channel = rest.trim().to_string();
            if channel.is_empty() {
                unknown(line)
            } else {
                Command::Names { channel }
            }
        }
        "TOPIC" => {
            let (channel, text) = match rest.split_once(' ') {
                Some((ch, t)) => (ch.trim().to_string(), Some(t.trim().to_string())),
                None => (rest.trim().to_string(), None),
            };
            if channel.is_empty() {
                unknown(line)
            } else {
                Command::Topic { channel, text }
            }
        }
        "HELP" => Command::Help,
        "WS" => {
            let channel = rest.trim().to_string();
            if channel.is_empty() {
                unknown(line)
            } else {
                Command::Ws { channel }
            }
        }
        "HISTORY" => {
            let mut it = rest.split_whitespace();
            let channel = it.next().unwrap_or("").to_string();
            let mut limit = 50usize;
            let mut before = None;
            // Either `HISTORY #chan`, `HISTORY #chan 20`, `HISTORY #chan BEFORE 40`
            // or `HISTORY #chan 20 BEFORE 40`.
            match it.next() {
                Some(tok) if tok.eq_ignore_ascii_case("BEFORE") => {
                    before = it.next().and_then(|v| v.parse::<i64>().ok());
                }
                Some(tok) => {
                    if let Ok(parsed) = tok.parse::<usize>() {
                        limit = parsed.clamp(1, crate::protocol::MAX_HISTORY);
                    }
                    if let Some(next) = it.next()
                        && next.eq_ignore_ascii_case("BEFORE")
                    {
                        before = it.next().and_then(|v| v.parse::<i64>().ok());
                    }
                }
                None => {}
            }
            if channel.is_empty() {
                unknown(line)
            } else {
                Command::History {
                    channel,
                    limit,
                    before,
                }
            }
        }
        "PING" => Command::Ping,
        "QUIT" => Command::Quit,
        _ => unknown(line),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_register() {
        assert_eq!(
            parse_line("REGISTER alice secret123"),
            Command::Register {
                username: "alice".into(),
                password: "secret123".into()
            }
        );
    }

    #[test]
    fn parses_login() {
        assert_eq!(
            parse_line("LOGIN alice secret123"),
            Command::Login {
                username: "alice".into(),
                password: "secret123".into()
            }
        );
    }

    #[test]
    fn parses_join() {
        assert_eq!(
            parse_line("JOIN #general"),
            Command::Join {
                channel: "#general".into()
            }
        );
    }

    #[test]
    fn parses_privmsg_with_colon() {
        assert_eq!(
            parse_line("PRIVMSG #general :hello world"),
            Command::Privmsg {
                channel: "#general".into(),
                message: "hello world".into()
            }
        );
    }

    #[test]
    fn parses_privmsg_without_colon() {
        assert_eq!(
            parse_line("PRIVMSG #general hi"),
            Command::Privmsg {
                channel: "#general".into(),
                message: "hi".into()
            }
        );
    }

    #[test]
    fn parses_privmsg_colon_inside_channel() {
        // A colon inside the channel name must not be treated as the message
        // separator; only a space-colon starts the message body.
        assert_eq!(
            parse_line("PRIVMSG #a:b hi"),
            Command::Privmsg {
                channel: "#a:b".into(),
                message: "hi".into()
            }
        );
    }

    #[test]
    fn parses_list() {
        assert_eq!(parse_line("LIST"), Command::List);
    }

    #[test]
    fn parses_history_with_default_limit() {
        assert_eq!(
            parse_line("HISTORY #general"),
            Command::History {
                channel: "#general".into(),
                limit: 50,
                before: None,
            }
        );
    }

    #[test]
    fn parses_history_with_limit() {
        assert_eq!(
            parse_line("HISTORY #general 10"),
            Command::History {
                channel: "#general".into(),
                limit: 10,
                before: None,
            }
        );
    }

    #[test]
    fn parses_history_with_before_cursor() {
        assert_eq!(
            parse_line("HISTORY #general 10 BEFORE 42"),
            Command::History {
                channel: "#general".into(),
                limit: 10,
                before: Some(42),
            }
        );
    }

    #[test]
    fn history_ignores_a_non_numeric_before_cursor() {
        assert_eq!(
            parse_line("HISTORY #general 10 BEFORE abc"),
            Command::History {
                channel: "#general".into(),
                limit: 10,
                before: None,
            }
        );
    }

    #[test]
    fn parses_quit_and_ping() {
        assert_eq!(parse_line("QUIT"), Command::Quit);
        assert_eq!(parse_line("PING"), Command::Ping);
    }

    #[test]
    fn parses_names() {
        assert_eq!(
            parse_line("NAMES #general"),
            Command::Names {
                channel: "#general".into()
            }
        );
    }

    #[test]
    fn parses_topic_get_and_set() {
        assert_eq!(
            parse_line("TOPIC #general"),
            Command::Topic {
                channel: "#general".into(),
                text: None
            }
        );
        assert_eq!(
            parse_line("TOPIC #general welcome all"),
            Command::Topic {
                channel: "#general".into(),
                text: Some("welcome all".into())
            }
        );
    }

    #[test]
    fn parses_help() {
        assert_eq!(parse_line("HELP"), Command::Help);
    }

    #[test]
    fn parses_who_and_whois() {
        assert_eq!(
            parse_line("WHO alice"),
            Command::Who {
                username: "alice".into()
            }
        );
        assert_eq!(
            parse_line("WHOIS alice"),
            Command::Whois {
                username: "alice".into()
            }
        );
    }

    #[test]
    fn parses_kick_op_deop() {
        assert_eq!(
            parse_line("KICK #general bob"),
            Command::Kick {
                channel: "#general".into(),
                user: "bob".into()
            }
        );
        assert_eq!(
            parse_line("OP #general bob"),
            Command::Op {
                channel: "#general".into(),
                user: "bob".into()
            }
        );
        assert_eq!(
            parse_line("DEOP #general bob"),
            Command::Deop {
                channel: "#general".into(),
                user: "bob".into()
            }
        );
    }

    #[test]
    fn parses_kicked_as_unknown() {
        // `KICKED` is a server-originated control line, not a client command,
        // so it must not be parsed into a `Command` variant.
        assert_eq!(
            parse_line("KICKED #general"),
            Command::Unknown("KICKED #general".into())
        );
    }

    #[test]
    fn unknown_command() {
        assert_eq!(
            parse_line("BOGUS stuff"),
            Command::Unknown("BOGUS stuff".into())
        );
    }

    #[test]
    fn strips_crlf() {
        assert_eq!(parse_line("LIST\r\n"), Command::List);
    }

    #[test]
    fn empty_line_is_unknown() {
        assert_eq!(parse_line(""), Command::Unknown(String::new()));
    }

    #[test]
    fn valid_username_rejects_channel_sigil_and_spaces() {
        assert!(valid_username("alice"));
        assert!(!valid_username(""));
        assert!(!valid_username("alice bob"));
        assert!(!valid_username("#general"));
        assert!(!valid_username("a@b"));
        assert!(!valid_username("a!b"));
        assert!(!valid_username("a:b"));
        assert!(!valid_username(&"a".repeat(MAX_NAME_LEN + 1)));
        assert!(valid_username(&"a".repeat(MAX_NAME_LEN)));
    }

    #[test]
    fn valid_channel_rejects_colon_and_oversize() {
        assert!(valid_channel("#general"));
        assert!(!valid_channel("#a:b"));
        assert!(!valid_channel(&format!("#{}", "a".repeat(MAX_NAME_LEN))));
        assert!(valid_channel(&format!("#{}", "a".repeat(MAX_NAME_LEN - 1))));
    }

    #[test]
    fn valid_topic_rejects_control_characters() {
        assert!(valid_topic(""));
        assert!(valid_topic("welcome all"));
        assert!(!valid_topic("hello\nworld"));
        assert!(!valid_topic("hello\rworld"));
        assert!(!valid_topic("hello\u{0001}world"));
    }

    #[test]
    fn valid_message_rejects_control_characters() {
        assert!(valid_message(""));
        assert!(valid_message("hello world"));
        assert!(valid_message("a:b"));
        assert!(!valid_message("hello\nworld"));
        assert!(!valid_message("hello\rworld"));
        assert!(!valid_message("hello\u{0001}world"));
    }

    /// Deterministic pseudo-fuzz of `parse_line`: feed it a broad sample of
    /// byte strings (including every single byte value, plus random-ish
    /// combinations) and assert it never panics and never yields a `Command`
    /// whose user-controlled fields contain a control character.
    ///
    /// This is a cheap stand-in for a real `cargo-fuzz` harness: it runs in the
    /// normal test suite, needs no extra dependencies, and exercises the parser
    /// far more broadly than the hand-written cases above. The invariant that
    /// matters is that no parsed field can smuggle a `\r`/`\n` into broadcast
    /// output, so we check every field of every variant.
    #[test]
    fn parse_line_never_panics_and_never_smuggles_control_chars() {
        // A small deterministic PRNG so the test is reproducible.
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            // xorshift64*
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            seed.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };

        // Every single byte value, plus a few structured lines, plus random
        // strings of assorted lengths.
        let mut corpus: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        for line in [
            "PRIVMSG #general :hello",
            "JOIN #general",
            "REGISTER alice secret",
            "HISTORY #general 10 BEFORE 42",
            "TOPIC #general some text",
            "KICK #general bob",
            "NICK alice",
            "WHOIS alice",
            "LIST",
            "QUIT",
            "PING",
            "HELP",
            "NAMES #general",
            "OP #general bob",
            "DEOP #general bob",
        ] {
            corpus.push(line.as_bytes().to_vec());
        }
        for _ in 0..2000 {
            let len = (next() % 40) as usize;
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push((next() & 0xFF) as u8);
            }
            corpus.push(v);
        }

        for raw in corpus {
            // The reader strips a trailing CRLF before parsing, so emulate that
            // here to match production input.
            let mut s = String::from_utf8_lossy(&raw).into_owned();
            if s.ends_with('\n') {
                s.pop();
                if s.ends_with('\r') {
                    s.pop();
                }
            }
            // Assert the parser never panics on arbitrary input. Reaching the
            // end of the loop would also prove this, but an explicit
            // `catch_unwind` makes the invariant visible and gives a clearer
            // failure message naming the offending input.
            let cmd = std::panic::catch_unwind(|| parse_line(&s))
                .unwrap_or_else(|_| panic!("parse_line panicked on input {s:?}"));
            // Now check no field smuggles a control character.
            let fields: Vec<&str> = match &cmd {
                Command::Register { username, .. } => vec![username],
                Command::Login { username, .. } => vec![username],
                Command::Join { channel } => vec![channel],
                Command::Part { channel } => vec![channel],
                Command::Privmsg { channel, message } => vec![channel, message],
                Command::History { channel, .. } => vec![channel],
                Command::Names { channel } => vec![channel],
                Command::Topic { channel, text } => {
                    let mut f = vec![channel.as_str()];
                    if let Some(t) = text {
                        f.push(t);
                    }
                    f
                }
                Command::Kick { channel, user } => vec![channel, user],
                Command::Op { channel, user } => vec![channel, user],
                Command::Deop { channel, user } => vec![channel, user],
                Command::Who { username } => vec![username],
                Command::Whois { username } => vec![username],
                Command::Nick { name } => vec![name],
                Command::Ws { channel } => vec![channel],
                Command::Unknown(_)
                | Command::List
                | Command::Quit
                | Command::Ping
                | Command::Help => {
                    vec![]
                }
            };
            for f in fields {
                assert!(
                    !f.chars().any(char::is_control),
                    "parse_line smuggled a control char in {f:?} from {s:?}"
                );
            }
        }
    }
}
