//! Slash-command parsing, shared by every frontend.

/// A recognized in-session command. Frontends decide how to act on each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Help,
    /// Reset the conversation, keeping only the system prompt.
    Clear,
    /// Switch the model for subsequent turns (empty = report current).
    Model(String),
    /// Switch the provider/profile for subsequent turns (empty = report current).
    /// Named profiles are resolved from the config; the model is then picked
    /// via `/model` from what the new endpoint serves.
    ///
    /// Subcommands: `add` (create a new profile interactively), `edit [name]`
    /// (edit an existing profile). The frontend parses these from the arg.
    Provider(String),
    /// Show cumulative token usage.
    Tokens,
    /// Show current session/runtime status.
    Status,
    /// Write a debug/event log for the current session.
    Log,
    /// Show, set, or clear the verify command turns iterate against. Empty =
    /// show; `off`/`none`/`clear` = disable; anything else = set.
    Verify(String),
    /// Show what's changed in the working tree (git diff).
    Diff,
    /// Copy the last assistant response, or `all` for the transcript.
    Copy(String),
    /// Show, set, or clear the current session goal.
    Goal(String),
    /// Show a context-occupancy breakdown: system prompt, per-turn token
    /// estimates, and what compaction would keep/elide.
    Context,
    /// Explore the repo and write a project-context file (runs as a turn).
    Init,
    /// Reclaim context. Empty arg = configured strategy; `full`/`hybrid`/`elide`
    /// pick one explicitly.
    Compact(String),
    /// Re-run the last user message (after truncating its previous attempt).
    Retry,
    /// Load the last user prompt into the input line for editing before
    /// resending. Handled by the frontend (it manipulates the input line).
    Edit,
    /// Revert the file changes the last turn made (from its git checkpoint).
    Undo,
    /// Stage all working-tree changes and commit them with an auto-generated
    /// message summarizing the changed files (the `/commit` command).
    Commit,
    /// Print the version and exit.
    Version,
    /// Export the conversation to a file.
    Export(String),
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
        "provider" | "prov" => Command::Provider(arg),
        "tokens" | "usage" | "cost" => Command::Tokens,
        "status" | "st" => Command::Status,
        "log" | "debug" => Command::Log,
        "verify" | "test" => Command::Verify(arg),
        "diff" | "changes" => Command::Diff,
        "copy" | "cp" => Command::Copy(arg),
        "goal" => Command::Goal(arg),
        "context" | "ctx" => Command::Context,
        "init" => Command::Init,
        "compact" => Command::Compact(arg),
        "retry" | "redo" => Command::Retry,
        "edit" => Command::Edit,
        "undo" | "revert" => Command::Undo,
        "commit" => Command::Commit,
        "version" | "ver" | "v" => Command::Version,
        "export" => Command::Export(arg),
        "exit" | "quit" | "q" => Command::Quit,
        other => Command::Unknown(other.to_string()),
    })
}

/// One user-facing slash command — the single source of truth for `/help` and
/// the interactive completion menu, so they can't drift from each other.
pub struct CommandSpec {
    /// Canonical name without the leading slash (what completion inserts).
    pub name: &'static str,
    /// Argument hint, e.g. `[id]`; empty when the command takes no arguments.
    pub args: &'static str,
    /// One-line description.
    pub help: &'static str,
    /// Enumerable values the argument can take, each with a one-line hint, for
    /// the completion menu (e.g. `/compact ` → hybrid/full/elide). Empty when the
    /// argument is freeform (`/model <id>`, `/goal <text>`) or absent.
    pub arg_values: &'static [(&'static str, &'static str)],
}

impl CommandSpec {
    /// Whether the command accepts arguments (so completion leaves a trailing
    /// space for the user to type them, rather than submitting immediately).
    pub fn takes_args(&self) -> bool {
        !self.args.is_empty()
    }
}

/// Every slash command, in display order. Each `name` must be parseable by
/// [`parse`] (guarded by a test).
pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "help",
        args: "",
        help: "show this help",
        arg_values: &[],
    },
    CommandSpec {
        name: "model",
        args: "[id]",
        help: "show or switch the model (no id opens a picker)",
        arg_values: &[],
    },
    CommandSpec {
        name: "provider",
        args: "[name|add|edit|remove]",
        help: "switch profile, or add/edit/remove a profile (no arg lists all)",
        arg_values: &[],
    },
    CommandSpec {
        name: "verify",
        args: "[cmd|off]",
        help: "show/set/clear the test command turns iterate against",
        arg_values: &[("off", "disable the verify command")],
    },
    CommandSpec {
        name: "diff",
        args: "",
        help: "show what files have changed (git diff)",
        arg_values: &[],
    },
    CommandSpec {
        name: "copy",
        args: "[all]",
        help: "copy the last response (or transcript) to the clipboard",
        arg_values: &[("all", "copy the whole transcript, not just the last reply")],
    },
    CommandSpec {
        name: "goal",
        args: "[text|clear]",
        help: "show, set, or clear the current session goal",
        arg_values: &[("clear", "clear the current goal")],
    },
    CommandSpec {
        name: "context",
        args: "",
        help: "show context-occupancy breakdown and compaction preview",
        arg_values: &[],
    },
    CommandSpec {
        name: "init",
        args: "",
        help: "scan the repo and write an HI.md project guide",
        arg_values: &[],
    },
    CommandSpec {
        name: "compact",
        args: "[kind]",
        help: "reclaim context (kind: hybrid, full, or elide)",
        arg_values: &[
            (
                "hybrid",
                "summarize old turns, keep the recent ones verbatim",
            ),
            ("full", "summarize the whole conversation into a brief"),
            ("elide", "drop old tool output, no model call"),
        ],
    },
    CommandSpec {
        name: "retry",
        args: "",
        help: "re-run your last message",
        arg_values: &[],
    },
    CommandSpec {
        name: "edit",
        args: "",
        help: "load your last message into the input line to edit and resend",
        arg_values: &[],
    },
    CommandSpec {
        name: "undo",
        args: "",
        help: "revert the file changes from the last turn",
        arg_values: &[],
    },
    CommandSpec {
        name: "commit",
        args: "",
        help: "stage all changes and commit them (git add -A && git commit)",
        arg_values: &[],
    },
    CommandSpec {
        name: "version",
        args: "",
        help: "show version",
        arg_values: &[],
    },
    CommandSpec {
        name: "export",
        args: "[path]",
        help: "export the conversation to a file (default: transcript.md)",
        arg_values: &[],
    },
    CommandSpec {
        name: "status",
        args: "",
        help: "show provider, model, queue, context, and last turn state",
        arg_values: &[],
    },
    CommandSpec {
        name: "log",
        args: "",
        help: "write a local debug log for this session",
        arg_values: &[],
    },
    CommandSpec {
        name: "tokens",
        args: "",
        help: "cumulative token usage this session",
        arg_values: &[],
    },
    CommandSpec {
        name: "clear",
        args: "",
        help: "start a fresh conversation",
        arg_values: &[],
    },
    CommandSpec {
        name: "exit",
        args: "",
        help: "quit",
        arg_values: &[],
    },
];

/// The message `/init` runs as a turn: explore the project and write a concise
/// `HI.md` guide that future sessions load as context.
pub const INIT_PROMPT: &str = "Explore this project (use the list and read tools) and write a \
file named HI.md at the repository root — a concise guide for a coding agent working here. \
Cover: what the project is and does; the main directories and files and their roles; the exact \
build, test, lint, and run commands; and any conventions or constraints worth knowing. Be \
factual and tight — this file is loaded as context for future sessions. Create HI.md with the \
write tool, then end with a one-line summary of what you captured.";

/// Commands whose canonical name starts with `prefix` (case-insensitive), in
/// display order — drives the `/`-completion menu. An empty prefix lists all.
pub fn matching(prefix: &str) -> Vec<&'static CommandSpec> {
    let needle = prefix.to_lowercase();
    COMMANDS
        .iter()
        .filter(|c| c.name.starts_with(&needle))
        .collect()
}

/// Enumerable argument values (value, hint) for command `name` whose value
/// starts with `prefix` (case-insensitive) — drives argument completion in the
/// `/`-menu (e.g. `/compact ` → hybrid/full/elide). Empty when the command is
/// unknown, takes a freeform argument, or nothing matches.
pub fn arg_matching(name: &str, prefix: &str) -> Vec<(&'static str, &'static str)> {
    let needle = prefix.to_lowercase();
    COMMANDS
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(name))
        .map(|c| {
            c.arg_values
                .iter()
                .filter(|(v, _)| v.starts_with(&needle))
                .copied()
                .collect()
        })
        .unwrap_or_default()
}

/// Help text, generated from [`COMMANDS`] so it always lists exactly what
/// exists. Includes a keybindings section so Ctrl- shortcuts aren't secret.
pub fn help_text() -> String {
    let mut out = String::from("commands:\n");
    for c in COMMANDS {
        let left = if c.args.is_empty() {
            format!("/{}", c.name)
        } else {
            format!("/{} {}", c.name, c.args)
        };
        out.push_str(&format!("  {left:<18} {}\n", c.help));
    }
    out.push_str("aliases: /m /st /cp /redo /revert /new /changes /usage /debug /h /?");
    out.push_str("\n\nkeybindings (TUI):\n");
    out.push_str("  Ctrl-T             toggle reasoning (thinking) collapse\n");
    out.push_str("  Ctrl-D (idle)      quit\n");
    out.push_str("  Ctrl-D (typing)    toggle the working-tree diff panel\n");
    out.push_str("  Ctrl-?             toggle the agent observability panel\n");
    out.push_str("  Ctrl-C             interrupt the running turn (or clear input)\n");
    out.push_str("  Ctrl-R             fuzzy-search input history\n");
    out.push_str("  Ctrl-A / Ctrl-E    move cursor to start / end of the line\n");
    out.push_str("  Ctrl-U             clear the input line\n");
    out.push_str("  Alt-Enter          insert a newline (multi-line prompt)\n");
    out.push_str("  PageUp / PageDown  scroll the transcript\n");
    out.push_str("  Esc                clear input, or quit when the line is empty\n");
    out
}

#[cfg(test)]
mod tests {
    use super::{COMMANDS, Command, matching, parse};

    #[test]
    fn every_listed_command_parses_to_a_real_command() {
        // Guards against the menu/help listing a command no frontend can run.
        for spec in COMMANDS {
            let line = format!("/{}", spec.name);
            match parse(&line) {
                Some(Command::Unknown(_)) | None => {
                    panic!("listed command {line} does not parse")
                }
                Some(_) => {}
            }
        }
    }

    #[test]
    fn matching_filters_by_prefix() {
        // Empty prefix → everything; a prefix narrows; no match → empty.
        assert_eq!(matching("").len(), COMMANDS.len());
        let m = matching("co");
        assert!(m.iter().any(|c| c.name == "compact"));
        assert!(m.iter().any(|c| c.name == "copy"));
        assert!(m.iter().all(|c| c.name.starts_with("co")));
        assert!(matching("zzz").is_empty());
    }

    #[test]
    fn arg_matching_filters_enumerable_values() {
        use super::arg_matching;
        fn names(v: Vec<(&'static str, &'static str)>) -> Vec<&'static str> {
            v.into_iter().map(|(n, _)| n).collect()
        }
        // Empty prefix → all of the command's values, in order.
        assert_eq!(
            names(arg_matching("compact", "")),
            ["hybrid", "full", "elide"]
        );
        // A prefix narrows; case-insensitive.
        assert_eq!(names(arg_matching("compact", "h")), ["hybrid"]);
        assert_eq!(names(arg_matching("compact", "E")), ["elide"]);
        // No match, freeform-arg command, and unknown command all → empty.
        assert!(arg_matching("compact", "z").is_empty());
        assert!(arg_matching("model", "").is_empty());
        assert!(arg_matching("nope", "").is_empty());
    }

    #[test]
    fn every_compact_kind_value_parses() {
        // The menu's compact values must stay in lockstep with the parser, or the
        // menu would offer a kind /compact can't actually run.
        let compact = COMMANDS.iter().find(|c| c.name == "compact").unwrap();
        for (value, _) in compact.arg_values {
            assert!(
                crate::CompactionKind::from_arg(value).is_some(),
                "compact kind {value:?} listed in the menu must parse"
            );
        }
    }

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
            parse("/provider sonnet"),
            Some(Command::Provider("sonnet".into()))
        );
        assert_eq!(parse("/provider"), Some(Command::Provider(String::new())));
        assert_eq!(
            parse("/prov local"),
            Some(Command::Provider("local".into()))
        );
        assert_eq!(
            parse("/provider add"),
            Some(Command::Provider("add".into()))
        );
        assert_eq!(
            parse("/provider edit sonnet"),
            Some(Command::Provider("edit sonnet".into()))
        );
        assert_eq!(
            parse("/provider remove local"),
            Some(Command::Provider("remove local".into()))
        );
        assert_eq!(
            parse("/provider rm local"),
            Some(Command::Provider("rm local".into()))
        );
        assert_eq!(
            parse("/verify cargo test"),
            Some(Command::Verify("cargo test".into()))
        );
        assert_eq!(parse("/verify"), Some(Command::Verify(String::new())));
        assert_eq!(parse("/status"), Some(Command::Status));
        assert_eq!(parse("/log"), Some(Command::Log));
        assert_eq!(parse("/diff"), Some(Command::Diff));
        assert_eq!(parse("/copy"), Some(Command::Copy(String::new())));
        assert_eq!(parse("/copy all"), Some(Command::Copy("all".into())));
        assert_eq!(parse("/goal"), Some(Command::Goal(String::new())));
        assert_eq!(
            parse("/goal ship it"),
            Some(Command::Goal("ship it".into()))
        );
        assert_eq!(parse("/compact"), Some(Command::Compact(String::new())));
        assert_eq!(
            parse("/compact hybrid"),
            Some(Command::Compact("hybrid".into()))
        );
        assert_eq!(parse("/redo"), Some(Command::Retry));
        assert_eq!(parse("/undo"), Some(Command::Undo));
        assert_eq!(parse("/bogus"), Some(Command::Unknown("bogus".into())));
    }
}
