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
    /// Expanded read-only slash prompt macro that should run as a model turn.
    Prompt(String),
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
    /// Inspect the configured MCP endpoint: server info, tools, model count.
    Mcp,
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
        "review" => Command::Prompt(read_only_macro_prompt("review", &arg)),
        "security" => Command::Prompt(read_only_macro_prompt("security", &arg)),
        "roadmap" => Command::Prompt(read_only_macro_prompt("roadmap", &arg)),
        "gaps" => Command::Prompt(read_only_macro_prompt("gaps", &arg)),
        "build" => Command::Prompt(build_macro_prompt(&arg)),
        "status" | "st" if arg.is_empty() => Command::Status,
        "status" | "st" => Command::Prompt(read_only_macro_prompt("status", &arg)),
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
        "mcp" => Command::Mcp,
        "exit" | "quit" | "q" => Command::Quit,
        other => Command::Unknown(other.to_string()),
    })
}

/// Expand a read-only prompt slash macro, or return `None` for non-macros.
pub fn expand_prompt_macro(line: &str) -> Option<String> {
    match parse(line)? {
        Command::Prompt(prompt) => Some(prompt),
        _ => None,
    }
}

fn read_only_macro_prompt(kind: &str, topic: &str) -> String {
    let topic = topic.trim();
    let topic = if topic.is_empty() {
        "the codebase"
    } else {
        topic
    };
    let recipe = match kind {
        "security" => {
            "Search for unsafe, unwrap, expect, panic!, command execution, filesystem/env access, and secret/token/auth patterns, then read the top matching files."
        }
        "status" => {
            "Inspect git status/diff summary, workspace manifests, README/docs when present, main crate or module entrypoints, and tests."
        }
        "roadmap" => {
            "Inspect workspace manifests, owning modules, tests, and TODO/FIXME or missing-coverage search results before naming build-next work."
        }
        "gaps" => {
            "Inspect workspace manifests, owning modules, tests, and TODO/FIXME or missing-coverage search results before naming gaps."
        }
        _ => "Inspect relevant files or targeted search results before giving findings.",
    };
    format!(
        "Read-only {kind} request for: {topic}\n\nDo not write, edit, apply patches, run mutating shell commands, or change files. Use read-only inspection before the final answer. {recipe}\n\nIf only a directory listing is available, keep inspecting or explicitly say the evidence is insufficient instead of making file-specific findings."
    )
}

fn build_macro_prompt(topic: &str) -> String {
    let topic = topic.trim();
    let topic = if topic.is_empty() {
        "the requested tool"
    } else {
        topic
    };
    format!(
        "Build {topic}.\n\nImplementation requirements:\n- Inspect the workspace before choosing files or stack.\n- Choose the local stack implied by existing manifests and entrypoints; if no stack is clear and this is a TUI, create a Rust binary in the current directory using Ratatui and Crossterm.\n- In an empty Rust target directory, prefer `cargo init --bin .` before editing so the manifest has a valid target from the start.\n- Edit or create the required files; do not stop at a plan, explanation, or scaffold.\n- Prefer a compact working vertical slice and small valid tool calls over one huge all-at-once source write.\n- Run an appropriate noninteractive validation command after the last file change.\n- Finish with a concise recap naming changed files and validation commands."
    )
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
        name: "review",
        args: "[topic]",
        help: "run a read-only code review with file inspection",
        arg_values: &[],
    },
    CommandSpec {
        name: "security",
        args: "[topic]",
        help: "run a read-only security review with targeted search",
        arg_values: &[],
    },
    CommandSpec {
        name: "roadmap",
        args: "[topic]",
        help: "discuss build-next roadmap after inspection",
        arg_values: &[],
    },
    CommandSpec {
        name: "gaps",
        args: "[topic]",
        help: "discuss missing gaps after inspection",
        arg_values: &[],
    },
    CommandSpec {
        name: "build",
        args: "[thing]",
        help: "build a tool/app end-to-end with edits and validation",
        arg_values: &[],
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
        name: "mcp",
        args: "",
        help: "inspect the MCP endpoint (server, tools, model count)",
        arg_values: &[],
    },
    CommandSpec {
        name: "status",
        args: "[topic]",
        help: "show runtime status, or discuss codebase status with a topic",
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
    out.push_str("  Ctrl-D             toggle the working-tree diff panel\n");
    out.push_str("  Ctrl-?             toggle the agent observability panel\n");
    out.push_str("  Ctrl-C             interrupt the running turn; double-press idle to quit\n");
    out.push_str("  Ctrl-R             fuzzy-search input history\n");
    out.push_str("  Ctrl-A / Ctrl-E    move cursor to start / end of the line\n");
    out.push_str("  Ctrl-U             clear the input line\n");
    out.push_str("  Alt-Enter          insert a newline (multi-line prompt)\n");
    out.push_str("  PageUp / PageDown  scroll the transcript\n");
    out.push_str("  Esc                clear input or dismiss panels\n");
    out.push_str("  /quit              quit\n");
    out
}

#[cfg(test)]
mod tests {
    use super::{COMMANDS, Command, expand_prompt_macro, help_text, matching, parse};

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
    fn help_text_matches_non_quitting_idle_keybindings() {
        let help = help_text();
        assert!(
            help.contains("Ctrl-D             toggle the working-tree diff panel"),
            "Ctrl-D should be documented as diff toggle:\n{help}"
        );
        assert!(
            help.contains("double-press idle to quit"),
            "Ctrl-C should document idle quit behavior:\n{help}"
        );
        assert!(
            help.contains("Esc                clear input or dismiss panels"),
            "Esc should not be documented as idle quit:\n{help}"
        );
        assert!(
            !help.contains("Ctrl-D (idle)") && !help.contains("quit when the line is empty"),
            "stale quit bindings should not be advertised:\n{help}"
        );
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
        assert!(matches!(
            parse("/status codebase state"),
            Some(Command::Prompt(_))
        ));
        assert!(matches!(
            parse("/review scheduler"),
            Some(Command::Prompt(_))
        ));
        assert!(matches!(
            parse("/security unsafe unwraps"),
            Some(Command::Prompt(_))
        ));
        assert!(matches!(
            parse("/roadmap next work"),
            Some(Command::Prompt(_))
        ));
        assert!(matches!(
            parse("/gaps missing pieces"),
            Some(Command::Prompt(_))
        ));
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

    #[test]
    fn prompt_macros_expand_to_read_only_inspection_prompts() {
        let review = expand_prompt_macro("/review parser").unwrap();
        assert!(review.contains("Read-only review request"));
        assert!(review.contains("parser"));
        assert!(review.contains("Do not write"));
        assert!(review.contains("Use read-only inspection"));

        let security = expand_prompt_macro("/security unsafe unwraps").unwrap();
        assert!(security.contains("unsafe unwraps"));
        assert!(security.contains("unsafe"));
        assert!(security.contains("unwrap"));
        assert!(security.contains("secret/token/auth"));

        let status = expand_prompt_macro("/status codebase state").unwrap();
        assert!(status.contains("codebase state"));
        assert!(status.contains("git status/diff"));

        let build = expand_prompt_macro("/build gpu training calculator").unwrap();
        assert!(build.contains("Build gpu training calculator."));
        assert!(build.contains("Inspect the workspace"));
        assert!(build.contains("Edit or create"));
        assert!(build.contains("validation command"));
        assert!(build.contains("changed files and validation commands"));

        assert!(expand_prompt_macro("/status").is_none());
    }
}
