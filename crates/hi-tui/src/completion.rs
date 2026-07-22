//! The `/`-command completion menu: derives what to offer from the input line
//! and resolves it to rows (command names or enumerable argument values).

use hi_agent::command;

/// State of the slash-command completion menu.
pub(crate) struct CompletionState {
    /// What the menu is completing — a command name, or the argument of a known
    /// command — and the prefix it's filtered to.
    pub ctx: CompletionContext,
    /// Index of the highlighted match.
    pub selected: usize,
}

/// What the completion menu is offering, derived from the input line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CompletionContext {
    /// Typing the command name itself (`/`, `/co`) — the lowercased prefix.
    Command(String),
    /// Typing the argument of a command that has enumerable values (`/compact `,
    /// `/compact hy`) — the canonical command name and the lowercased value prefix.
    Arg { cmd: &'static str, prefix: String },
    /// Typing an `@file` path mention (`@src/`, `@ren`) — the path prefix after
    /// the `@`, resolved against the workspace root at render time. Lets a
    /// coding user point the agent at a file without typing a full path.
    Path { prefix: String },
}

/// One row in the completion menu — a command name or an argument value, already
/// resolved to what shows and what gets inserted.
pub(crate) struct CompletionItem {
    /// Left column: `/compact` for a command, `hybrid` for an argument value.
    pub label: String,
    /// Right-column hint.
    pub help: String,
    /// Text the input becomes when this row is accepted.
    pub insert: String,
    /// Whether accepting with Enter submits the line. Command names that take
    /// arguments fill `/name ` and wait; everything else (no-arg commands, fully
    /// chosen argument values) is a complete line that runs.
    pub submit_on_enter: bool,
}

/// What the completion menu should offer for `input`, or `None` to close it:
/// the command name while it's being typed (`/`, `/co`), or — once past the name
/// — the argument of a command that has enumerable values (`/compact `,
/// `/compact hy`). A freeform-argument command (`/model <id>`) or a second arg
/// token closes the menu, as does any non-slash input.
/// The one command whose argument values come from live state (the model
/// catalog) rather than the static table.
pub(crate) const MODEL_CMD: &str = "model";
/// Cap on inline `/model` id completions, so a large catalog can't flood the menu.
pub(crate) const MODEL_COMPLETION_MAX: usize = 8;
/// The command whose argument values are profile names (live state from the
/// config, plus the `add`/`edit`/`remove` subcommands).
pub(crate) const PROVIDER_CMD: &str = "provider";
pub(crate) const SESSIONS_CMD: &str = "sessions";
pub(crate) const SESSIONS_SWITCH_CTX: &str = "sessions switch";
pub(crate) const SESSIONS_RENAME_CTX: &str = "sessions rename";
pub(crate) const SESSIONS_FAVORITE_CTX: &str = "sessions favorite";
pub(crate) const SESSIONS_ARCHIVE_CTX: &str = "sessions archive";
pub(crate) const SESSIONS_RESTORE_CTX: &str = "sessions restore";
pub(crate) const SESSIONS_DELETE_CTX: &str = "sessions delete";

pub(crate) fn completion_context(input: &str) -> Option<CompletionContext> {
    // `@file` path mention: the last whitespace-delimited token starts with
    // `@` and is still being typed (no trailing whitespace). Resolved against
    // the workspace root at render time. Only when not a slash command.
    if !input.starts_with('/')
        && let Some(ctx) = path_completion_context(input)
    {
        return Some(ctx);
    }
    let rest = input.strip_prefix('/')?;
    match rest.split_once(char::is_whitespace) {
        // No space yet → still choosing the command name.
        None => Some(CompletionContext::Command(rest.to_lowercase())),
        // Past the name, on the first argument token.
        Some((name, arg)) => {
            let spec = command::COMMANDS
                .iter()
                .find(|c| c.name.eq_ignore_ascii_case(name))?;
            if spec.name == SESSIONS_CMD
                && let Some((action, remainder)) = arg.split_once(char::is_whitespace)
            {
                if remainder.contains(char::is_whitespace) {
                    return None;
                }
                let cmd = match action {
                    "switch" => SESSIONS_SWITCH_CTX,
                    "rename" => SESSIONS_RENAME_CTX,
                    "favorite" => SESSIONS_FAVORITE_CTX,
                    "archive" => SESSIONS_ARCHIVE_CTX,
                    "restore" => SESSIONS_RESTORE_CTX,
                    "delete" => SESSIONS_DELETE_CTX,
                    _ => return None,
                };
                return Some(CompletionContext::Arg {
                    cmd,
                    prefix: remainder.to_lowercase(),
                });
            }
            // Nested `/config <key> …` keeps completing the key until a full
            // key is chosen; values after that are freeform or handled by the
            // rewritten bare command.
            if (spec.name == "config" || spec.name == "cfg" || spec.name == "set")
                && arg.contains(char::is_whitespace)
            {
                return None;
            }
            if arg.contains(char::is_whitespace) {
                return None; // a second token — past the single argument
            }
            let prefix = arg.to_lowercase();
            if !spec.arg_values.is_empty() {
                // A static enumerable set (compact, copy, verify, goal, config).
                // For `/config`, keep the menu open after a full key so the user
                // can still Tab-accept and type a value (`/config lsp `).
                let keep_open_after_key =
                    spec.name == "config" || spec.name == "cfg" || spec.name == "set";
                if !keep_open_after_key && spec.arg_values.iter().any(|(v, _)| *v == prefix) {
                    return None; // a full valid value is typed — nothing left to pick
                }
                if keep_open_after_key && spec.arg_values.iter().any(|(v, _)| *v == prefix) {
                    // Full key chosen — leave a trailing space via insert path.
                    return Some(CompletionContext::Arg {
                        cmd: "config",
                        prefix,
                    });
                }
                return Some(CompletionContext::Arg {
                    cmd: if spec.name == "cfg" || spec.name == "set" {
                        "config"
                    } else {
                        spec.name
                    },
                    prefix,
                });
            }
            if spec.name == MODEL_CMD {
                // Model ids are dynamic — the catalog is filtered at render time,
                // so emptiness is resolved there, not here.
                return Some(CompletionContext::Arg {
                    cmd: spec.name,
                    prefix,
                });
            }
            if spec.name == PROVIDER_CMD {
                // Profile names + subcommands are dynamic — resolved at render
                // time from the live profile list.
                return Some(CompletionContext::Arg {
                    cmd: spec.name,
                    prefix,
                });
            }
            None // freeform or no argument — nothing to enumerate
        }
    }
}

/// Detect an in-progress `@file` path mention in `input`: the last
/// whitespace-delimited token must start with `@`, with no trailing whitespace
/// (so the menu stays open only while the token is being typed). Returns the
/// path prefix (after the `@`). `@@` is treated as a literal `@`, not a
/// mention, so escaped/decorative uses don't trigger the menu.
fn path_completion_context(input: &str) -> Option<CompletionContext> {
    // The token currently being typed: everything after the last whitespace.
    let last_token = input.rsplit(char::is_whitespace).next()?;
    let after_at = last_token.strip_prefix('@')?;
    // `@@` is not a mention.
    if after_at.starts_with('@') {
        return None;
    }
    // A completed token (trailing whitespace) would have been split off, so
    // `after_at` is the live prefix. An empty prefix (`@` just typed) opens
    // the menu with everything.
    Some(CompletionContext::Path {
        prefix: after_at.to_string(),
    })
}

/// Resolve a completion context to the menu rows it offers.
pub(crate) fn completion_items_for(ctx: &CompletionContext) -> Vec<CompletionItem> {
    match ctx {
        CompletionContext::Command(prefix) => command::matching(prefix)
            .into_iter()
            .map(|spec| {
                let takes_args = spec.takes_args();
                CompletionItem {
                    label: format!("/{}", spec.name),
                    help: spec.help.to_string(),
                    insert: if takes_args {
                        format!("/{} ", spec.name)
                    } else {
                        format!("/{}", spec.name)
                    },
                    submit_on_enter: !takes_args,
                }
            })
            .collect(),
        CompletionContext::Arg { cmd, prefix } => command::arg_matching(cmd, prefix)
            .into_iter()
            .map(|(value, hint)| {
                let config_key_needs_value = *cmd == "config"
                    && matches!(
                        value,
                        "model"
                            | "provider"
                            | "auth"
                            | "reasoning"
                            | "temp"
                            | "steps"
                            | "verify"
                            | "lsp"
                            | "delegate"
                            | "moe-streaming"
                            | "skeptic-local"
                            | "rsi"
                            | "ui"
                            | "theme"
                            | "density"
                            | "mouse"
                    );
                let sessions_needs_id = *cmd == SESSIONS_CMD
                    && matches!(
                        value,
                        "switch" | "rename" | "favorite" | "archive" | "restore" | "delete"
                    );
                let needs_more = config_key_needs_value || sessions_needs_id;
                CompletionItem {
                    label: value.to_string(),
                    help: hint.to_string(),
                    insert: if needs_more {
                        format!("/{cmd} {value} ")
                    } else {
                        format!("/{cmd} {value}")
                    },
                    submit_on_enter: !needs_more,
                }
            })
            .collect(),
        // Path completion is resolved against the workspace root in
        // `App::items_for_ctx` (it needs `&self`); the free function offers
        // nothing so the menu closes when no App is available.
        CompletionContext::Path { .. } => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompletionContext::{Arg, Command, Path},
        completion_context,
    };

    #[test]
    fn completion_context_tracks_name_then_argument() {
        // The command name, until a space is typed.
        assert_eq!(completion_context("/"), Some(Command(String::new())));
        assert_eq!(completion_context("/mo"), Some(Command("mo".to_string())));
        assert_eq!(
            completion_context("/MODEL"),
            Some(Command("model".to_string()))
        );
        // Past the name, on the argument of a command with enumerable values.
        assert_eq!(
            completion_context("/compact "),
            Some(Arg {
                cmd: "compact",
                prefix: String::new()
            })
        );
        assert_eq!(
            completion_context("/compact hy"),
            Some(Arg {
                cmd: "compact",
                prefix: "hy".to_string()
            })
        );
        // Settings hub: first key is enumerable under /config.
        assert_eq!(
            completion_context("/config "),
            Some(Arg {
                cmd: "config",
                prefix: String::new()
            })
        );
        assert_eq!(
            completion_context("/config ls"),
            Some(Arg {
                cmd: "config",
                prefix: "ls".to_string()
            })
        );
        // Nested value after a full key closes the static menu.
        assert_eq!(completion_context("/config lsp on"), None);
        // The single-keyword commands and the dynamic model command, too.
        assert_eq!(
            completion_context("/verify "),
            Some(Arg {
                cmd: "verify",
                prefix: String::new()
            })
        );
        assert_eq!(
            completion_context("/model gp"),
            Some(Arg {
                cmd: "model",
                prefix: "gp".to_string()
            })
        );
        // A fully-typed valid static value has nothing left to complete → no menu.
        assert_eq!(completion_context("/compact hybrid"), None);
        assert_eq!(completion_context("/verify off"), None);
        // The dynamic provider command offers completions (profile names +
        // subcommands), resolved at render time.
        assert_eq!(
            completion_context("/provider "),
            Some(Arg {
                cmd: "provider",
                prefix: String::new()
            })
        );
        assert_eq!(
            completion_context("/provider lo"),
            Some(Arg {
                cmd: "provider",
                prefix: "lo".to_string()
            })
        );
        // A command that takes no argument, with a trailing space → no menu.
        assert_eq!(completion_context("/diff "), None);
        // A second argument token is past the single arg → no menu.
        assert_eq!(completion_context("/compact hybrid x"), None);
        assert_eq!(
            completion_context("/sessions switch 1783"),
            Some(Arg {
                cmd: "sessions switch",
                prefix: "1783".to_string()
            })
        );
        assert_eq!(
            completion_context("/sessions rename 1783"),
            Some(Arg {
                cmd: "sessions rename",
                prefix: "1783".to_string()
            })
        );
        assert_eq!(completion_context("/sessions rename 1783 new name"), None);
        // Not a slash command at all.
        assert_eq!(completion_context("hello"), None);
    }

    #[test]
    fn completion_context_detects_at_file_path_mentions() {
        // A bare `@` opens the path menu with an empty prefix.
        assert_eq!(
            completion_context("@"),
            Some(Path {
                prefix: String::new()
            })
        );
        // A prefix after `@` filters paths.
        assert_eq!(
            completion_context("@src/ren"),
            Some(Path {
                prefix: "src/ren".to_string()
            })
        );
        // `@` mid-prompt: the last token is the one being completed.
        assert_eq!(
            completion_context("fix the bug @crates/hi-t"),
            Some(Path {
                prefix: "crates/hi-t".to_string()
            })
        );
        // A completed `@path` token (trailing space) closes the menu.
        assert_eq!(completion_context("@src/main.rs "), None);
        // `@@` is not a mention (escaped/decorative).
        assert_eq!(completion_context("@@"), None);
        // A slash command is not treated as a path even if it has `@`.
        assert_eq!(
            completion_context("/model @gpt"),
            Some(Arg {
                cmd: "model",
                prefix: "@gpt".to_string()
            })
        );
        // Plain text with no `@` is no completion.
        assert_eq!(completion_context("fix the bug"), None);
    }
}
