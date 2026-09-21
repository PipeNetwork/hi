//! The `/`-command completion menu: derives what to offer from the input line
//! and resolves it to rows (command names or enumerable argument values).

use hi_harness::{COMMANDS, CommandSpec};

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
    /// Typing the argument of a command that has enumerable values (`/permissions `,
    /// `/permissions al`) — the canonical command name and the lowercased value prefix.
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
/// — the argument of a command that has enumerable values (`/permissions `,
/// `/permissions al`). A freeform-argument command (`/verify <cmd>`) or a second arg
/// token closes the menu, as does any non-slash input.
/// The one command whose argument values come from live state (the model
/// catalog) rather than the static table.
pub(crate) const MODEL_CMD: &str = "model";
/// Cap on inline `/model` id completions, so a large catalog can't flood the menu.
pub(crate) const MODEL_COMPLETION_MAX: usize = 8;
/// Visible rows in the `/` menu and Ctrl-K palette. Extra matches stay
/// reachable by moving the highlight — same compact window grok uses.
pub(crate) const COMPLETION_VISIBLE_ROWS: usize = 8;

/// Inclusive-start, exclusive-end window that keeps `selected` in view.
pub(crate) fn visible_range(selected: usize, len: usize, max_rows: usize) -> (usize, usize) {
    if len == 0 || max_rows == 0 {
        return (0, 0);
    }
    let window = max_rows.min(len);
    let selected = selected.min(len - 1);
    let start = selected.saturating_sub(window.saturating_sub(1));
    (start, start + window)
}

/// The command whose argument values are profile names, hosted provider
/// presets, and the `add`/`edit`/`remove` subcommands.
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
            let spec = spec_by_name(name)?;
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
            if spec.name == MODEL_CMD
                && let Some((_, effort_prefix)) = arg.split_once(char::is_whitespace)
            {
                if effort_prefix.contains(char::is_whitespace) {
                    return None;
                }
                return Some(CompletionContext::Arg {
                    cmd: "model effort",
                    prefix: effort_prefix.to_lowercase(),
                });
            }
            if arg.contains(char::is_whitespace) {
                return None;
            }
            let prefix = arg.to_lowercase();
            if spec.name == MODEL_CMD || spec.name == PROVIDER_CMD {
                return Some(CompletionContext::Arg {
                    cmd: spec.name,
                    prefix,
                });
            }
            if !arg_values(spec.name).is_empty() {
                return Some(CompletionContext::Arg {
                    cmd: spec.name,
                    prefix,
                });
            }
            None
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
    // Typing `:N-M` on an already-chosen path closes the menu so Tab/Right
    // can accept ghost text / stay in the composer.
    let (path, range) = crate::file_mentions::split_path_range(after_at);
    if range.is_some() {
        return None;
    }
    Some(CompletionContext::Path {
        prefix: path.to_string(),
    })
}

/// Bold/accent the subsequence of `prefix` inside a completion label.
pub(crate) fn highlight_label(
    label: &str,
    prefix: &str,
    selected: bool,
) -> Vec<ratatui::text::Span<'static>> {
    use ratatui::style::{Modifier, Style};
    use ratatui::text::Span;
    let th = crate::theme::theme();
    let base = if selected {
        Style::default()
            .fg(th.accent_system)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(th.text_primary)
    };
    let hit = base.fg(th.accent_plan);
    if prefix.is_empty() {
        return vec![Span::styled(label.to_string(), base)];
    }
    let mut needle = prefix.chars().peekable();
    let mut spans = Vec::new();
    let mut buf = String::new();
    let mut buf_hit = false;
    for ch in label.chars() {
        let is_hit = needle
            .peek()
            .copied()
            .is_some_and(|n| n.eq_ignore_ascii_case(&ch));
        if is_hit {
            needle.next();
        }
        if is_hit != buf_hit && !buf.is_empty() {
            spans.push(Span::styled(
                std::mem::take(&mut buf),
                if buf_hit { hit } else { base },
            ));
        }
        if buf.is_empty() {
            buf_hit = is_hit;
        }
        buf.push(ch);
    }
    if !buf.is_empty() {
        spans.push(Span::styled(buf, if buf_hit { hit } else { base }));
    }
    spans
}

fn spec_by_name(name: &str) -> Option<&'static CommandSpec> {
    COMMANDS
        .iter()
        .find(|spec| spec.name.eq_ignore_ascii_case(name))
}

fn matching(prefix: &str) -> Vec<&'static CommandSpec> {
    COMMANDS
        .iter()
        .filter(|spec| spec.name.starts_with(prefix))
        .collect()
}

fn arg_values(cmd: &str) -> &'static [(&'static str, &'static str)] {
    match cmd {
        "permissions" | "permission" | "perms" => &[
            ("ask", "confirm each mutation"),
            ("auto", "safe edits; Jev may auto-approve reversible shell"),
            ("always", "approve mutations this session"),
            ("yolo", "same as always"),
        ],
        "density" => &[
            ("compact", "headers only"),
            ("comfortable", "default"),
            ("verbose", "expand tool output"),
        ],
        "theme" => &[("dark", "dark theme"), ("light", "light theme")],
        "mouse" => &[("on", "capture mouse"), ("off", "native selection")],
        "jev-compact" | "compact-jev" => &[
            ("on", "prune stale tools with Jev this session"),
            ("off", "use cheap shrink and summary"),
        ],
        "verify" => &[("off", "disable post-turn check")],
        "trust" => &[
            ("on", "persist folder trust for this workspace"),
            ("off", "revoke folder trust for this workspace"),
        ],
        "review" => &[
            ("audit", "report coverage and defects only; no fix loop"),
            (
                "all",
                "audit the whole repo in chunks, one turn per top-level directory",
            ),
            ("status", "show the spec-review phase, pass, and findings"),
            ("stop", "end the spec-review loop and clear its plan"),
        ],
        "config" => &[("reasoning", "set reasoning effort")],
        "effort" | "model effort" => &[
            ("low", "less reasoning"),
            ("medium", "default reasoning"),
            ("high", "more reasoning"),
            ("xhigh", "maximum reasoning"),
            ("off", "endpoint default"),
        ],
        "usage" | "cost" => &[
            ("show", "open the usage modal"),
            ("manage", "open Pipe Network billing"),
            ("context", "context-window breakdown"),
        ],
        "autoharnessfix" => &[
            ("status", "supervised? generation? last incident"),
            ("on", "enable Sentinel and re-exec if the session is saved"),
            ("off", "disable Sentinel in machine config"),
            ("diagnose", "write a forensic snapshot"),
            ("history", "last 20 incidents"),
            ("repair", "manual harness repair (no crash required)"),
        ],
        _ => &[],
    }
}

fn takes_args(spec: &CommandSpec) -> bool {
    !spec.args.is_empty()
}

/// Resolve a completion context to the menu rows it offers.
pub(crate) fn completion_items_for(ctx: &CompletionContext) -> Vec<CompletionItem> {
    match ctx {
        CompletionContext::Command(prefix) => matching(prefix)
            .into_iter()
            .map(|spec| {
                let takes_args = takes_args(spec);
                // Optional `[show|manage]` must not block Enter — grok `/usage`
                // opens the modal immediately. `/permissions` still waits.
                let submit_immediately = !takes_args || matches!(spec.name, "usage" | "cost");
                CompletionItem {
                    label: format!("/{}", spec.name),
                    help: spec.help.to_string(),
                    insert: if takes_args && !submit_immediately {
                        format!("/{} ", spec.name)
                    } else {
                        format!("/{}", spec.name)
                    },
                    submit_on_enter: submit_immediately,
                }
            })
            .collect(),
        CompletionContext::Arg { cmd, prefix } => arg_values(cmd)
            .iter()
            .filter(|(value, _)| value.starts_with(prefix.as_str()))
            .map(|(value, hint)| CompletionItem {
                label: (*value).to_string(),
                help: (*hint).to_string(),
                insert: format!("/{cmd} {value}"),
                submit_on_enter: true,
            })
            .collect(),
        CompletionContext::Path { .. } => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompletionContext::{Arg, Command, Path},
        completion_context, completion_items_for, highlight_label, visible_range,
    };

    #[test]
    fn visible_range_keeps_the_highlight_in_a_small_window() {
        assert_eq!(visible_range(0, 0, 8), (0, 0));
        assert_eq!(visible_range(0, 3, 8), (0, 3));
        assert_eq!(visible_range(0, 20, 8), (0, 8));
        assert_eq!(visible_range(7, 20, 8), (0, 8));
        assert_eq!(visible_range(8, 20, 8), (1, 9));
        assert_eq!(visible_range(19, 20, 8), (12, 20));
        assert_eq!(visible_range(99, 20, 8), (12, 20));
    }

    #[test]
    fn slash_opens_the_full_command_menu() {
        assert_eq!(completion_context("/"), Some(Command(String::new())));
        let items = completion_items_for(&Command(String::new()));
        let labels: Vec<&str> = items.iter().map(|item| item.label.as_str()).collect();
        for name in [
            "/model",
            "/effort",
            "/permissions",
            "/auto",
            "/yolo",
            "/always-approve",
            "/undo",
            "/help",
            "/usage",
            "/dashboard",
        ] {
            assert!(
                labels.contains(&name),
                "missing {name} in / menu: {labels:?}"
            );
        }
        let yolo = items.iter().find(|item| item.label == "/yolo").unwrap();
        assert!(yolo.submit_on_enter);
        let permissions = items
            .iter()
            .find(|item| item.label == "/permissions")
            .unwrap();
        assert!(!permissions.submit_on_enter);
        assert_eq!(permissions.insert, "/permissions ");
        let usage = items.iter().find(|item| item.label == "/usage").unwrap();
        assert!(
            usage.submit_on_enter,
            "/usage must run on Enter, not wait for [show|manage]"
        );
        assert_eq!(usage.insert, "/usage");
    }

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
            completion_context("/permissions "),
            Some(Arg {
                cmd: "permissions",
                prefix: String::new()
            })
        );
        assert_eq!(
            completion_context("/usage "),
            Some(Arg {
                cmd: "usage",
                prefix: String::new()
            })
        );
        assert_eq!(
            completion_context("/permissions al"),
            Some(Arg {
                cmd: "permissions",
                prefix: "al".to_string()
            })
        );
        // Commands without enumerable argument values close the menu after the name.
        assert_eq!(completion_context("/compact "), None);
        assert_eq!(
            completion_context("/jev-compact "),
            Some(Arg {
                cmd: "jev-compact",
                prefix: String::new()
            })
        );
        assert_eq!(
            completion_context("/compact-jev on"),
            Some(Arg {
                cmd: "compact-jev",
                prefix: "on".to_string()
            })
        );
        assert_eq!(
            completion_context("/config "),
            Some(Arg {
                cmd: "config",
                prefix: String::new()
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
        assert_eq!(
            completion_context("/model pipe/deepseek-v4-flash-0731 hi"),
            Some(Arg {
                cmd: "model effort",
                prefix: "hi".to_string()
            })
        );
        assert_eq!(
            completion_context("/effort "),
            Some(Arg {
                cmd: "effort",
                prefix: String::new()
            })
        );
        assert_eq!(
            completion_context("/effort xh"),
            Some(Arg {
                cmd: "effort",
                prefix: "xh".to_string()
            })
        );
        assert_eq!(
            completion_context("/verify off"),
            Some(Arg {
                cmd: "verify",
                prefix: "off".to_string()
            })
        );
        // `/provider` is not in the Pipe command table.
        assert_eq!(completion_context("/provider "), None);
        assert_eq!(completion_context("/provider lo"), None);
        // A command that takes no argument, with a trailing space → no menu.
        assert_eq!(completion_context("/diff "), None);
        // A second argument token is past the single arg → no menu.
        assert_eq!(completion_context("/permissions always x"), None);
        assert_eq!(completion_context("/yo"), Some(Command("yo".to_string())));
        assert_eq!(
            completion_context("/permissions yo"),
            Some(Arg {
                cmd: "permissions",
                prefix: "yo".to_string()
            })
        );
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
        // Line-range suffix is not a path filter.
        assert_eq!(completion_context("@src/main.rs:40"), None);
        assert_eq!(completion_context("@src/main.rs:40-80"), None);
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

    #[test]
    fn highlight_label_marks_a_fuzzy_subsequence() {
        let spans = highlight_label("src/main.rs", "smn", false);
        let joined: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(joined, "src/main.rs");
        let hits: String = spans
            .iter()
            .filter(|s| s.style.fg == Some(crate::theme::theme().accent_plan))
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(hits, "smn");
    }
}
