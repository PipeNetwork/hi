//! Slash-command parsing for the interactive session.

use hi_ai::ReasoningEffort;

/// A recognized in-session command. Frontends decide how to act on each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Help(String),
    Clear,
    Model(String),
    Effort(String),
    Config(String),
    Auth(String),
    Login(String),
    Logout(String),
    Status,
    Doctor,
    AutoHarnessFix(String),
    Permissions(String),
    Rewind(String),
    Verify(String),
    Trust(String),
    /// `/review [audit|status|stop] [all|path...]`: spec-coverage audit plus
    /// the bounded fix loop. The raw argument string is parsed by `ReviewArgs`.
    Review(String),
    Diff,
    Files,
    Copy(String),
    Compact(String),
    JevCompact(String),
    Retry,
    Undo,
    Sessions(String),
    Version,
    Theme(String),
    Density(String),
    Mouse(String),
    Tutorial,
    Usage(String),
    Dashboard,
    Quit,
    Unknown(String),
    Removed(String),
}

pub struct CommandSpec {
    pub name: &'static str,
    pub args: &'static str,
    pub help: &'static str,
}

pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "help",
        args: "",
        help: "show commands",
    },
    CommandSpec {
        name: "model",
        args: "[id] [effort]",
        help: "switch Pipe model, optionally with a reasoning level",
    },
    CommandSpec {
        name: "effort",
        args: "[low|medium|high|xhigh|off]",
        help: "set reasoning effort on the current model",
    },
    CommandSpec {
        name: "permissions",
        args: "[ask|auto|always]",
        help: "confirm ladder: ask, auto, or always (yolo)",
    },
    CommandSpec {
        name: "auto",
        args: "",
        help: "toggle auto-approve for safe file edits; Jev may auto-approve reversible shell",
    },
    CommandSpec {
        name: "yolo",
        args: "",
        help: "toggle always-approve (skip edit and shell confirms)",
    },
    CommandSpec {
        name: "always-approve",
        args: "",
        help: "same as /yolo",
    },
    CommandSpec {
        name: "undo",
        args: "",
        help: "restore files from the last turn checkpoint",
    },
    CommandSpec {
        name: "retry",
        args: "",
        help: "re-run the last prompt",
    },
    CommandSpec {
        name: "diff",
        args: "",
        help: "show changes since HEAD (the live Changes pane in the TUI)",
    },
    CommandSpec {
        name: "verify",
        args: "[cmd|off]",
        help: "set a post-turn check command",
    },
    CommandSpec {
        name: "trust",
        args: "[on|off]",
        help: "persist folder trust so hi.toml remote routes, hooks, and MCP load here",
    },
    CommandSpec {
        name: "review",
        args: "[audit|status|stop] [all|path...]",
        help: "audit the code against plan.md/spec.md (or recent git work), then fix P0/P1 defects in a bounded loop",
    },
    CommandSpec {
        name: "login",
        args: "[pipenetwork]",
        help: "sign in to pipenetwork.ai and configure the API key",
    },
    CommandSpec {
        name: "logout",
        args: "[pipenetwork]",
        help: "forget the stored Pipe Network credential",
    },
    CommandSpec {
        name: "status",
        args: "",
        help: "show session status",
    },
    CommandSpec {
        name: "usage",
        args: "[show|manage]",
        help: "credit/token usage modal (alias /cost); /usage manage opens billing",
    },
    CommandSpec {
        name: "cost",
        args: "",
        help: "same as /usage",
    },
    CommandSpec {
        name: "context",
        args: "",
        help: "context-window breakdown (opens /usage on Context usage)",
    },
    CommandSpec {
        name: "files",
        args: "",
        help: "list files changed this session",
    },
    CommandSpec {
        name: "compact",
        args: "[context]",
        help: "summarize the conversation to reclaim context",
    },
    CommandSpec {
        name: "jev-compact",
        args: "[on|off]",
        help: "Jev tool prune this session (needs a TypeSafe key)",
    },
    CommandSpec {
        name: "compact-jev",
        args: "[on|off]",
        help: "alias for /jev-compact",
    },
    CommandSpec {
        name: "copy",
        args: "",
        help: "copy the last assistant reply",
    },
    CommandSpec {
        name: "mouse",
        args: "[on|off]",
        help: "on (default): click › to expand; off: terminal highlight-to-copy",
    },
    CommandSpec {
        name: "clear",
        args: "",
        help: "reset the conversation",
    },
    CommandSpec {
        name: "config",
        args: "[reasoning <level>]",
        help: "show or set request settings",
    },
    CommandSpec {
        name: "auth",
        args: "pipenetwork [key]",
        help: "store a Pipe API key",
    },
    CommandSpec {
        name: "sessions",
        args: "",
        help: "list saved sessions",
    },
    CommandSpec {
        name: "rewind",
        args: "<n>",
        help: "drop back to user turn n",
    },
    CommandSpec {
        name: "doctor",
        args: "",
        help: "check key, Pipe /models, git, sandbox, and Sentinel",
    },
    CommandSpec {
        name: "autoharnessfix",
        args: "[on|off|status|diagnose|history|repair]",
        help: "Hi Sentinel: wrap this session, inspect, or repair the harness",
    },
    CommandSpec {
        name: "version",
        args: "",
        help: "show hi version",
    },
    CommandSpec {
        name: "tutorial",
        args: "",
        help: "interactive tour (starts with /login)",
    },
    CommandSpec {
        name: "quit",
        args: "",
        help: "exit",
    },
    CommandSpec {
        name: "exit",
        args: "",
        help: "same as /quit",
    },
    CommandSpec {
        name: "dashboard",
        args: "",
        help: "open the agent dashboard (concurrent sessions)",
    },
    CommandSpec {
        name: "fleet",
        args: "",
        help: "alias for /dashboard",
    },
    CommandSpec {
        name: "agents-dashboard",
        args: "",
        help: "alias for /dashboard",
    },
];

pub const CORE_COMMANDS: &[&str] = &[
    "login",
    "model",
    "effort",
    "permissions",
    "auto",
    "yolo",
    "verify",
    "review",
    "undo",
    "retry",
    "diff",
    "status",
    "usage",
    "config",
    "compact",
    "copy",
    "files",
    "clear",
    "help",
    "dashboard",
    "quit",
];

/// Parse a line as a command. Returns `None` for ordinary input.
pub fn parse(line: &str) -> Option<Command> {
    let line = line.trim();
    let rest = line.strip_prefix('/')?;
    let mut parts = rest.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("").trim().to_string();
    Some(match name {
        "help" | "h" | "?" => Command::Help(arg),
        "clear" | "new" => Command::Clear,
        "model" | "m" => Command::Model(arg),
        "effort" => Command::Effort(arg),
        "config" | "cfg" | "set" => Command::Config(arg),
        "auth" => Command::Auth(arg),
        "login" | "signin" => Command::Login(arg),
        "logout" | "signout" => Command::Logout(arg),
        "status" | "st" => Command::Status,
        "usage" | "cost" => Command::Usage(arg),
        "context" => Command::Usage(if arg.is_empty() {
            "context".into()
        } else {
            arg
        }),
        "doctor" => Command::Doctor,
        "autoharnessfix" => Command::AutoHarnessFix(arg),
        "permissions" | "permission" | "perms" => Command::Permissions(arg),
        "always-approve" | "alwaysapprove" | "yolo" => Command::Permissions(if arg.is_empty() {
            "toggle-always".into()
        } else {
            arg
        }),
        "auto" => Command::Permissions(if arg.is_empty() {
            "toggle-auto".into()
        } else {
            arg
        }),
        "rewind" => Command::Rewind(arg),
        "verify" | "test" => Command::Verify(arg),
        "trust" => Command::Trust(arg),
        "review" => Command::Review(arg),
        "diff" | "changes" => Command::Diff,
        "files" => Command::Files,
        "copy" | "cp" => Command::Copy(arg),
        "compact" => Command::Compact(arg),
        "jev-compact" | "compact-jev" => Command::JevCompact(arg),
        "retry" | "redo" => Command::Retry,
        "undo" | "revert" => Command::Undo,
        "sessions" | "resume" => Command::Sessions(arg),
        "version" | "ver" | "v" => Command::Version,
        "theme" => Command::Theme(arg),
        "density" => Command::Density(arg),
        "mouse" => Command::Mouse(arg),
        "tutorial" => Command::Tutorial,
        "dashboard" | "agents-dashboard" | "fleet" => Command::Dashboard,
        "quit" | "exit" | "q" => Command::Quit,
        "goal" | "race" | "rsi" | "workflow" | "pipefs" | "delegate" | "moa" | "team"
        | "diff-lab" | "loop" | "watch" | "mcp" | "local" | "inbox" | "btw" => {
            Command::Removed(format!("/{name} was removed with the old harness"))
        }
        other => Command::Unknown(format!(
            "unknown command /{other} — type /help for commands"
        )),
    })
}

/// `/trust [on|off]`: show or change the persisted folder-trust grant for
/// `workspace_root`. This is the only in-product way to grant trust now that
/// startup never prompts on stdin. Hooks and repo MCP read the store live;
/// project `hi.toml` provider routes are merged at config load, so those
/// apply on the next start.
pub fn trust_command(workspace_root: &std::path::Path, arg: &str) -> String {
    use hi_tools::folder_trust as trust;
    let key = trust::workspace_key(workspace_root);
    match arg.trim().to_ascii_lowercase().as_str() {
        "" | "status" => {
            let store = trust::trust_store_file()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "unavailable (no user profile)".into());
            format!(
                "folder trust: {} · {} · store {store} · /trust on|off",
                if trust::folder_trust_granted(workspace_root) {
                    "granted"
                } else {
                    "not granted"
                },
                key.display(),
            )
        }
        "on" | "grant" | "yes" | "allow" => match trust::grant_folder_trust(workspace_root) {
            Ok(()) => format!(
                "folder trust granted for {} · hi.toml provider routes load on the next start",
                key.display()
            ),
            Err(error) => format!("could not persist folder trust: {error}"),
        },
        "off" | "revoke" | "no" | "deny" => match trust::try_revoke_folder_trust(workspace_root) {
            Ok(true) => format!("folder trust revoked for {}", key.display()),
            Ok(false) => "folder trust was not granted here".into(),
            Err(error) => format!("could not update folder trust: {error}"),
        },
        _ => "use /trust, /trust on, or /trust off".into(),
    }
}

/// `/effort` / trailing `/model` token: a reasoning level, or off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffortArg {
    Off,
    Level(ReasoningEffort),
}

impl EffortArg {
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Level(level) => level.as_str(),
        }
    }
}

/// Split `/model` arguments into an optional model id/name and optional effort.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelArgs {
    pub model: Option<String>,
    pub effort: Option<EffortArg>,
}

/// Parse a user-supplied reasoning level (`low`, `medium`, `high`, `xhigh`, `off`).
pub fn parse_effort_arg(arg: &str) -> Result<EffortArg, String> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Err("missing reasoning level (low, medium, high, xhigh, off)".into());
    }
    match arg.to_ascii_lowercase().as_str() {
        "off" | "none" | "default" | "auto" => Ok(EffortArg::Off),
        "max" => Ok(EffortArg::Level(ReasoningEffort::Xhigh)),
        other => ReasoningEffort::from_arg(other)
            .map(EffortArg::Level)
            .ok_or_else(|| {
                format!("unknown reasoning level '{arg}' (low, medium, high, xhigh, off)")
            }),
    }
}

/// Grok-build shape: `/model grok-4.6`, `/model Grok 4.6`, `/model Reasoning X high`.
pub fn parse_model_args(arg: &str) -> ModelArgs {
    let arg = arg.trim();
    if arg.is_empty() {
        return ModelArgs {
            model: None,
            effort: None,
        };
    }
    if !arg.contains(char::is_whitespace) && parse_effort_arg(arg).is_ok() {
        return ModelArgs {
            model: None,
            effort: parse_effort_arg(arg).ok(),
        };
    }
    if let Some((rest, last)) = arg.rsplit_once(char::is_whitespace)
        && let Ok(effort) = parse_effort_arg(last)
    {
        let model = rest.trim();
        return ModelArgs {
            model: (!model.is_empty()).then(|| model.to_string()),
            effort: Some(effort),
        };
    }
    ModelArgs {
        model: Some(arg.to_string()),
        effort: None,
    }
}

/// Pick a catalog id for a typed `/model` query (exact, then unique substring).
pub fn resolve_model_query(query: &str, ids: &[String]) -> String {
    let query = query.trim();
    if query.is_empty() {
        return query.to_string();
    }
    if let Some(id) = ids.iter().find(|id| id.eq_ignore_ascii_case(query)) {
        return id.clone();
    }
    let needle = query.to_ascii_lowercase();
    let hits: Vec<&String> = ids
        .iter()
        .filter(|id| id.to_ascii_lowercase().contains(&needle))
        .collect();
    if hits.len() == 1 {
        return hits[0].clone();
    }
    query.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_core_commands() {
        assert_eq!(parse("/undo"), Some(Command::Undo));
        assert_eq!(parse("/diff"), Some(Command::Diff));
        assert_eq!(parse("hello"), None);
        assert_eq!(parse("/login"), Some(Command::Login(String::new())));
        assert_eq!(parse("/tutorial"), Some(Command::Tutorial));
        assert_eq!(parse("/copy"), Some(Command::Copy(String::new())));
        assert_eq!(parse("/mouse off"), Some(Command::Mouse("off".into())));
        assert_eq!(
            parse("/jev-compact"),
            Some(Command::JevCompact(String::new()))
        );
        assert_eq!(
            parse("/jev-compact on"),
            Some(Command::JevCompact("on".into()))
        );
        assert_eq!(
            parse("/jev-compact off"),
            Some(Command::JevCompact("off".into()))
        );
        assert_eq!(
            parse("/compact-jev on"),
            Some(Command::JevCompact("on".into()))
        );
        assert!(COMMANDS.iter().any(|s| s.name == "jev-compact"));
        assert!(COMMANDS.iter().any(|s| s.name == "compact-jev"));
        assert_eq!(parse("/usage"), Some(Command::Usage(String::new())));
        assert_eq!(parse("/cost"), Some(Command::Usage(String::new())));
        assert_eq!(
            parse("/usage manage"),
            Some(Command::Usage("manage".into()))
        );
        assert_eq!(parse("/context"), Some(Command::Usage("context".into())));
        assert_eq!(
            parse("/login pipenetwork"),
            Some(Command::Login("pipenetwork".into()))
        );
        assert_eq!(parse("/logout pipe"), Some(Command::Logout("pipe".into())));
        assert_eq!(parse("/effort high"), Some(Command::Effort("high".into())));
        assert_eq!(
            parse("/model grok-4.6 high"),
            Some(Command::Model("grok-4.6 high".into()))
        );
        assert_eq!(
            parse("/yolo"),
            Some(Command::Permissions("toggle-always".into()))
        );
        assert_eq!(
            parse("/always-approve"),
            Some(Command::Permissions("toggle-always".into()))
        );
        assert_eq!(
            parse("/auto"),
            Some(Command::Permissions("toggle-auto".into()))
        );
        assert_eq!(
            parse("/permissions always"),
            Some(Command::Permissions("always".into()))
        );
        assert_eq!(parse("/dashboard"), Some(Command::Dashboard));
        assert_eq!(parse("/fleet"), Some(Command::Dashboard));
        assert_eq!(parse("/agents-dashboard"), Some(Command::Dashboard));
        assert_eq!(
            parse("/autoharnessfix status"),
            Some(Command::AutoHarnessFix("status".into()))
        );
        assert_eq!(
            parse("/autoharnessfix"),
            Some(Command::AutoHarnessFix(String::new()))
        );
        assert_eq!(
            parse("/autoharnessfix on"),
            Some(Command::AutoHarnessFix("on".into()))
        );
        assert!(COMMANDS.iter().any(|s| s.name == "autoharnessfix"));
        assert!(COMMANDS.iter().any(|s| s.name == "dashboard"));
        assert!(COMMANDS.iter().any(|s| s.name == "fleet"));
        assert!(CORE_COMMANDS.contains(&"dashboard"));
        assert!(matches!(parse("/goal x"), Some(Command::Removed(_))));
        let unknown = parse("/nope").unwrap();
        assert!(
            matches!(unknown, Command::Unknown(ref msg) if msg.contains("unknown command /nope"))
        );
    }

    #[test]
    fn parses_review_command_and_subcommands() {
        assert_eq!(parse("/review"), Some(Command::Review(String::new())));
        assert_eq!(
            parse("/review audit"),
            Some(Command::Review("audit".into()))
        );
        assert_eq!(
            parse("/review status"),
            Some(Command::Review("status".into()))
        );
        assert_eq!(parse("/review stop"), Some(Command::Review("stop".into())));
        assert_eq!(
            parse("/review docs/spec.md plan.md"),
            Some(Command::Review("docs/spec.md plan.md".into()))
        );
        assert!(matches!(parse("/audit"), Some(Command::Unknown(_))));
        assert_eq!(
            parse("  /review   audit   src/  "),
            Some(Command::Review("audit   src/".into()))
        );
        assert!(COMMANDS.iter().any(|s| s.name == "review"));
        assert!(CORE_COMMANDS.contains(&"review"));
    }

    #[test]
    fn model_args_split_trailing_effort_like_grok_build() {
        assert_eq!(
            parse_model_args(""),
            ModelArgs {
                model: None,
                effort: None
            }
        );
        assert_eq!(
            parse_model_args("pipe/deepseek-v4-flash-0731"),
            ModelArgs {
                model: Some("pipe/deepseek-v4-flash-0731".into()),
                effort: None
            }
        );
        assert_eq!(
            parse_model_args("pipe/deepseek-v4-flash-0731 high"),
            ModelArgs {
                model: Some("pipe/deepseek-v4-flash-0731".into()),
                effort: Some(EffortArg::Level(ReasoningEffort::High))
            }
        );
        assert_eq!(
            parse_model_args("Reasoning X high"),
            ModelArgs {
                model: Some("Reasoning X".into()),
                effort: Some(EffortArg::Level(ReasoningEffort::High))
            }
        );
        assert_eq!(
            parse_model_args("xhigh"),
            ModelArgs {
                model: None,
                effort: Some(EffortArg::Level(ReasoningEffort::Xhigh))
            }
        );
        assert_eq!(
            parse_model_args("off"),
            ModelArgs {
                model: None,
                effort: Some(EffortArg::Off)
            }
        );
    }

    #[test]
    fn resolve_model_query_prefers_exact_then_unique_substring() {
        let ids = vec![
            "pipe/deepseek-v4-flash-0731".into(),
            "pipe/kimi-k2.5".into(),
        ];
        assert_eq!(
            resolve_model_query("PIPE/kimi-k2.5", &ids),
            "pipe/kimi-k2.5"
        );
        assert_eq!(resolve_model_query("kimi", &ids), "pipe/kimi-k2.5");
        assert_eq!(resolve_model_query("pipe/", &ids), "pipe/");
        assert_eq!(resolve_model_query("custom-id", &ids), "custom-id");
    }
}
