//! Declarative keybinding table.
//!
//! The `?` help overlay and (where practical) dispatch documentation are
//! generated from this single source so they cannot drift.

/// Where a binding applies. Used to section the help overlay and to document
/// context; runtime dispatch still lives next to the mode handlers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BindContext {
    Input,
    Navigation,
    ReviewTools,
    Queue,
    Sessions,
    Normal,
    BlockNav,
    Confirm,
}

impl BindContext {
    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::Input => "Input",
            Self::Navigation => "Navigation",
            Self::ReviewTools => "Review & Tools",
            Self::Queue => "Queue",
            Self::Sessions => "Sessions",
            Self::Normal => "Normal mode",
            Self::BlockNav => "Block nav",
            Self::Confirm => "Confirm",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct KeyBinding {
    pub context: BindContext,
    pub keys: &'static str,
    pub help: &'static str,
    /// When true, shown in the `?` overlay (default chords). Internal/test-only
    /// bindings can set this false.
    pub in_help: bool,
}

/// Canonical binding table. Order within a context is the help-overlay order.
pub(crate) static KEY_BINDINGS: &[KeyBinding] = &[
    // --- Input ---
    KeyBinding {
        context: BindContext::Input,
        keys: "Enter",
        help: "send the prompt",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Alt-Enter / \\",
        help: "insert a newline (multi-line prompt)",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Ctrl-A/E/U/K",
        help: "line start/end / kill to start / kill to end",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Alt-B/F",
        help: "move cursor back/forward one word",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Ctrl-W",
        help: "delete the word before the cursor",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Ctrl-X",
        help: "edit the prompt in $EDITOR (multi-line)",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Ctrl-R",
        help: "fuzzy-search input history",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "@file",
        help: "Tab-complete a workspace path mention",
        in_help: true,
    },
    // --- Navigation ---
    KeyBinding {
        context: BindContext::Navigation,
        keys: "PgUp/PgDn",
        help: "scroll the transcript",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "Esc (empty)",
        help: "enter vim-style normal mode (j/k/g/G//)",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "Ctrl-B",
        help: "block nav: fold/unfold one tool-output block",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "?",
        help: "toggle this keybindings help",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "Ctrl-K",
        help: "command palette (fuzzy slash commands)",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "Ctrl-P/N",
        help: "jump to previous/next user prompt",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "Ctrl-[ / Ctrl-]",
        help: "jump to previous/next error line",
        in_help: true,
    },
    // --- Review & tools ---
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-D",
        help: "toggle the working-tree diff panel",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-G",
        help: "full-screen diff review (scrollable, n/p hunks)",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-Y",
        help: "copy the last code block to the clipboard",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-T",
        help: "toggle reasoning (thinking) display",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-O",
        help: "expand/collapse long tool output",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-?",
        help: "toggle agent observability panel",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "/density",
        help: "cycle compact / comfortable / verbose transcript density",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-C",
        help: "interrupt the running turn (twice when idle to quit)",
        in_help: true,
    },
    // --- Queue ---
    KeyBinding {
        context: BindContext::Queue,
        keys: "Alt-Up/Down",
        help: "select a queued prompt",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Queue,
        keys: "Alt-Backspace",
        help: "remove the selected queued prompt",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Queue,
        keys: "Alt-Shift-Up/Down",
        help: "reorder the selected queued prompt",
        in_help: true,
    },
    // --- Sessions ---
    KeyBinding {
        context: BindContext::Sessions,
        keys: "/help",
        help: "show all slash commands",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Sessions,
        keys: "/quit",
        help: "quit the session",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Sessions,
        keys: "/model · /provider",
        help: "switch model or provider profile",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Sessions,
        keys: "/theme",
        help: "cycle dark / light / ansi / auto",
        in_help: true,
    },
    // --- Normal mode ---
    KeyBinding {
        context: BindContext::Normal,
        keys: "j/k · u/d",
        help: "scroll line / half-page",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "g/G",
        help: "top / bottom",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "[ / ]",
        help: "previous/next user prompt",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "{ / }",
        help: "previous/next error line",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "/ · n/N",
        help: "search · next/prev match",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "y",
        help: "copy last code block",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "i/q/Esc",
        help: "return to insert mode",
        in_help: true,
    },
    // --- Block nav ---
    KeyBinding {
        context: BindContext::BlockNav,
        keys: "j/k · Enter",
        help: "move · fold/unfold selected block",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::BlockNav,
        keys: "Esc · Ctrl-B",
        help: "exit block nav",
        in_help: true,
    },
    // --- Confirm ---
    KeyBinding {
        context: BindContext::Confirm,
        keys: "y",
        help: "approve once",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Confirm,
        keys: "a",
        help: "always allow this session",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Confirm,
        keys: "p",
        help: "always allow this path prefix this session",
        in_help: true,
    },
    KeyBinding {
        context: BindContext::Confirm,
        keys: "n/Esc",
        help: "reject",
        in_help: true,
    },
];

/// Help overlay sections in display order (excludes mode-only contexts that are
/// already discoverable from their own banners, except we still list them).
const HELP_SECTIONS: &[BindContext] = &[
    BindContext::Input,
    BindContext::Navigation,
    BindContext::ReviewTools,
    BindContext::Queue,
    BindContext::Sessions,
];

/// Lines for the `?` help overlay: section headers + `key — help` rows.
/// Does not include the title line (`keybindings (? to close)`).
pub(crate) fn help_overlay_rows() -> Vec<(&'static str, Option<&'static str>)> {
    let mut rows = Vec::new();
    for ctx in HELP_SECTIONS {
        let bindings: Vec<_> = KEY_BINDINGS
            .iter()
            .filter(|b| b.in_help && b.context == *ctx)
            .collect();
        if bindings.is_empty() {
            continue;
        }
        rows.push((ctx.title(), None));
        for b in bindings {
            rows.push((b.keys, Some(b.help)));
        }
    }
    rows
}

/// How many rows the help overlay body occupies (sections + bindings).
pub(crate) fn help_overlay_height() -> usize {
    // title is rendered by the caller; body = section headers + bindings in HELP_SECTIONS
    help_overlay_rows().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_sections_are_non_empty() {
        let rows = help_overlay_rows();
        assert!(rows.len() > 10, "expected a full cheat sheet, got {}", rows.len());
        // Every help section header appears.
        for ctx in HELP_SECTIONS {
            assert!(
                rows.iter().any(|(k, h)| *k == ctx.title() && h.is_none()),
                "missing section {}",
                ctx.title()
            );
        }
    }

    #[test]
    fn every_in_help_binding_has_nonempty_keys() {
        for b in KEY_BINDINGS.iter().filter(|b| b.in_help) {
            assert!(!b.keys.is_empty());
            assert!(!b.help.is_empty());
        }
    }
}
