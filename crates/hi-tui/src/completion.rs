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

pub(crate) fn completion_context(input: &str) -> Option<CompletionContext> {
    let rest = input.strip_prefix('/')?;
    match rest.split_once(char::is_whitespace) {
        // No space yet → still choosing the command name.
        None => Some(CompletionContext::Command(rest.to_lowercase())),
        // Past the name, on the first argument token.
        Some((name, arg)) => {
            if arg.contains(char::is_whitespace) {
                return None; // a second token — past the single argument
            }
            let spec = command::COMMANDS
                .iter()
                .find(|c| c.name.eq_ignore_ascii_case(name))?;
            let prefix = arg.to_lowercase();
            if !spec.arg_values.is_empty() {
                // A static enumerable set (compact, copy, verify, goal).
                if spec.arg_values.iter().any(|(v, _)| *v == prefix) {
                    return None; // a full valid value is typed — nothing left to pick
                }
                return Some(CompletionContext::Arg {
                    cmd: spec.name,
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
            .map(|(value, hint)| CompletionItem {
                label: value.to_string(),
                help: hint.to_string(),
                insert: format!("/{cmd} {value}"),
                submit_on_enter: true,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompletionContext::{Arg, Command},
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
        // Not a slash command at all.
        assert_eq!(completion_context("hello"), None);
    }
}
