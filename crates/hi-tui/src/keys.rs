//! Declarative keybinding table.
//!
//! The `?` help overlay and runtime [`crate::action::resolve_key`] share this
//! table so help text and dispatch cannot drift. Bindings with
//! [`KeyBinding::action`] `Some` are executable chords; `None` rows are
//! help-only (text editing, slash commands, confirm modal owned by `run.rs`).

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::action::{Action, KeySurface};

/// Where a binding applies. Used to section the help overlay and to filter
/// which surfaces a runtime chord is active on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BindContext {
    Input,
    Navigation,
    ReviewTools,
    Queue,
    Sessions,
    Normal,
    BlockNav,
    Review,
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
            Self::Review => "Diff review",
            Self::Confirm => "Confirm",
        }
    }

    /// Surfaces on which a binding with this context may fire.
    fn surfaces(self) -> &'static [KeySurface] {
        match self {
            Self::Input | Self::Queue => &[KeySurface::Insert],
            // Navigation + review-tools globals work in insert and normal.
            Self::Navigation | Self::ReviewTools => &[KeySurface::Insert, KeySurface::Normal],
            Self::Normal => &[KeySurface::Normal],
            Self::BlockNav => &[KeySurface::BlockNav],
            Self::Review => &[KeySurface::Review],
            // Help-only / specialized handlers outside the action table.
            Self::Sessions | Self::Confirm => &[],
        }
    }
}

/// One physical key match for a binding (code + required modifiers).
#[derive(Clone, Copy, Debug)]
pub(crate) struct KeyMatch {
    pub code: KeyCode,
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    /// When set, the binding only fires on this surface (subset of context).
    pub only_surface: Option<KeySurface>,
}

impl KeyMatch {
    const fn plain(code: KeyCode) -> Self {
        Self {
            code,
            ctrl: false,
            alt: false,
            shift: false,
            only_surface: None,
        }
    }

    const fn ctrl(code: KeyCode) -> Self {
        Self {
            code,
            ctrl: true,
            alt: false,
            shift: false,
            only_surface: None,
        }
    }

    const fn alt(code: KeyCode) -> Self {
        Self {
            code,
            ctrl: false,
            alt: true,
            shift: false,
            only_surface: None,
        }
    }

    const fn alt_shift(code: KeyCode) -> Self {
        Self {
            code,
            ctrl: false,
            alt: true,
            shift: true,
            only_surface: None,
        }
    }

    const fn on_surface(mut self, surface: KeySurface) -> Self {
        self.only_surface = Some(surface);
        self
    }

    fn matches(&self, surface: KeySurface, key: &KeyEvent) -> bool {
        if let Some(only) = self.only_surface
            && only != surface
        {
            return false;
        }
        if key.code != self.code {
            return false;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        ctrl == self.ctrl && alt == self.alt && shift == self.shift
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct KeyBinding {
    pub context: BindContext,
    pub keys: &'static str,
    pub help: &'static str,
    /// When true, shown in the `?` overlay.
    pub in_help: bool,
    /// Runtime action when this binding fires. `None` = help-only.
    pub action: Option<Action>,
    /// Physical keys that produce `action`. Empty when help-only.
    pub matches: &'static [KeyMatch],
}

/// Canonical binding table. Order within a context is the help-overlay order.
pub(crate) static KEY_BINDINGS: &[KeyBinding] = &[
    // --- Input (help-only: edit_key owns these) ---
    KeyBinding {
        context: BindContext::Input,
        keys: "Enter",
        help: "send the prompt",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Alt-Enter / \\",
        help: "insert a newline (multi-line prompt)",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Ctrl-A/E/U/K",
        help: "line start/end / kill to start / kill to end",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Alt-B/F",
        help: "move cursor back/forward one word",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Ctrl-W",
        help: "delete the word before the cursor",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Ctrl-X",
        help: "edit the prompt in $EDITOR (multi-line)",
        in_help: true,
        // Chord is registered under Navigation so it also works in normal mode.
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "Ctrl-R",
        help: "fuzzy-search input history",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Input,
        keys: "@file",
        help: "Tab-complete a workspace path mention",
        in_help: true,
        action: None,
        matches: &[],
    },
    // --- Navigation ---
    KeyBinding {
        context: BindContext::Navigation,
        keys: "PgUp/PgDn",
        help: "scroll the transcript",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "Esc (empty)",
        help: "enter vim-style normal mode (j/k/g/G//)",
        in_help: true,
        action: Some(Action::EnterNormal),
        // Esc is handled in run.rs (empty-input + idle checks); listed for help.
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "Ctrl-X",
        help: "edit the prompt in $EDITOR (multi-line)",
        in_help: false, // listed under Input for help; runtime chord here for Insert+Normal
        action: Some(Action::ExternalEdit),
        matches: &[KeyMatch::ctrl(KeyCode::Char('x'))],
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "Ctrl-B",
        help: "block nav: fold/unfold one tool-output block",
        in_help: true,
        action: Some(Action::ToggleBlockNav),
        matches: &[KeyMatch::ctrl(KeyCode::Char('b'))],
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "?",
        help: "toggle this keybindings help",
        in_help: true,
        action: Some(Action::ToggleHelp),
        matches: &[KeyMatch::plain(KeyCode::Char('?')).on_surface(KeySurface::Insert)],
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "Ctrl-K",
        help: "command palette (fuzzy slash commands)",
        in_help: true,
        action: Some(Action::OpenPalette),
        matches: &[KeyMatch::ctrl(KeyCode::Char('k'))],
    },
    KeyBinding {
        context: BindContext::Navigation,
        keys: "Ctrl-P/N",
        help: "jump to previous/next user prompt",
        in_help: true,
        action: Some(Action::JumpPrompt { dir: -1 }),
        // Ctrl-P both surfaces; Ctrl-N insert-only (normal uses bare `n` for search).
        matches: &[
            KeyMatch::ctrl(KeyCode::Char('p')),
            KeyMatch::ctrl(KeyCode::Char('n')).on_surface(KeySurface::Insert),
        ],
    },
    // Separate row so help stays readable; action for Ctrl-N is in matches above
    // via the JumpPrompt dir fixup in resolve_from_table.
    KeyBinding {
        context: BindContext::Navigation,
        keys: "Ctrl-[ / Ctrl-]",
        help: "jump to previous/next error line",
        in_help: true,
        action: Some(Action::JumpError { dir: -1 }),
        matches: &[
            KeyMatch::ctrl(KeyCode::Char('[')),
            KeyMatch::ctrl(KeyCode::Char(']')),
        ],
    },
    // --- Review & tools ---
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-D",
        help: "toggle the working-tree diff panel",
        in_help: true,
        action: Some(Action::ToggleDiff),
        matches: &[KeyMatch::ctrl(KeyCode::Char('d'))],
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-G",
        help: "full-screen diff review (scrollable, n/p hunks)",
        in_help: true,
        action: Some(Action::ToggleReview),
        matches: &[KeyMatch::ctrl(KeyCode::Char('g'))],
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-Y",
        help: "copy the last code block to the clipboard",
        in_help: true,
        action: Some(Action::CopyLastCode),
        matches: &[KeyMatch::ctrl(KeyCode::Char('y'))],
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-T",
        help: "toggle reasoning (thinking) display",
        in_help: true,
        action: Some(Action::ToggleReasoning),
        matches: &[KeyMatch::ctrl(KeyCode::Char('t'))],
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-O",
        help: "expand/collapse long tool output",
        in_help: true,
        action: Some(Action::ToggleToolOutput),
        matches: &[KeyMatch::ctrl(KeyCode::Char('o'))],
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-?",
        help: "toggle agent observability panel",
        in_help: true,
        action: Some(Action::ToggleDebug),
        matches: &[KeyMatch::ctrl(KeyCode::Char('?'))],
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "/density",
        help: "cycle compact / comfortable / verbose transcript density",
        in_help: true,
        action: Some(Action::CycleDensity),
        // Slash command — not a key chord; action applied via /density handler.
        matches: &[],
    },
    KeyBinding {
        context: BindContext::ReviewTools,
        keys: "Ctrl-C",
        help: "interrupt the running turn (twice when idle to quit)",
        in_help: true,
        action: None,
        matches: &[],
    },
    // --- Queue ---
    KeyBinding {
        context: BindContext::Queue,
        keys: "Alt-Up/Down",
        help: "select a queued prompt",
        in_help: true,
        action: Some(Action::QueueSelectPrev),
        matches: &[
            KeyMatch::alt(KeyCode::Up),
            KeyMatch::alt(KeyCode::Down),
        ],
    },
    KeyBinding {
        context: BindContext::Queue,
        keys: "Alt-Backspace",
        help: "remove the selected queued prompt",
        in_help: true,
        action: Some(Action::QueueRemoveSelected),
        matches: &[KeyMatch::alt(KeyCode::Backspace)],
    },
    KeyBinding {
        context: BindContext::Queue,
        keys: "Alt-Shift-Up/Down",
        help: "reorder the selected queued prompt",
        in_help: true,
        action: Some(Action::QueueMoveSelected { delta: -1 }),
        matches: &[
            KeyMatch::alt_shift(KeyCode::Up),
            KeyMatch::alt_shift(KeyCode::Down),
        ],
    },
    // --- Sessions (slash commands — help only) ---
    KeyBinding {
        context: BindContext::Sessions,
        keys: "/help",
        help: "show all slash commands",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Sessions,
        keys: "/quit",
        help: "quit the session",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Sessions,
        keys: "/model · /provider",
        help: "switch model or provider profile",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Sessions,
        keys: "/theme",
        help: "cycle dark / light / ansi / auto",
        in_help: true,
        action: None,
        matches: &[],
    },
    // --- Normal mode ---
    KeyBinding {
        context: BindContext::Normal,
        keys: "j/k · u/d",
        help: "scroll line / half-page",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "g/G",
        help: "top / bottom",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "[ / ]",
        help: "previous/next user prompt",
        in_help: true,
        action: Some(Action::JumpPrompt { dir: -1 }),
        matches: &[
            KeyMatch::plain(KeyCode::Char('[')),
            KeyMatch::plain(KeyCode::Char(']')),
        ],
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "{ / }",
        help: "previous/next error line",
        in_help: true,
        action: Some(Action::JumpError { dir: -1 }),
        matches: &[
            KeyMatch::plain(KeyCode::Char('{')),
            KeyMatch::plain(KeyCode::Char('}')),
        ],
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "/ · n/N",
        help: "search · next/prev match",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "y",
        help: "copy last code block",
        in_help: true,
        action: Some(Action::CopyLastCode),
        matches: &[KeyMatch::plain(KeyCode::Char('y'))],
    },
    KeyBinding {
        context: BindContext::Normal,
        keys: "i/q/Esc",
        help: "return to insert mode",
        in_help: true,
        action: Some(Action::ExitToInsert),
        matches: &[
            KeyMatch::plain(KeyCode::Char('i')),
            KeyMatch::plain(KeyCode::Char('q')),
            KeyMatch::plain(KeyCode::Esc),
        ],
    },
    // --- Block nav ---
    KeyBinding {
        context: BindContext::BlockNav,
        keys: "j/k · Enter",
        help: "move · fold/unfold selected block",
        in_help: true,
        action: Some(Action::BlockNavUp),
        matches: &[
            KeyMatch::plain(KeyCode::Up),
            KeyMatch::plain(KeyCode::Char('k')),
            KeyMatch::plain(KeyCode::Down),
            KeyMatch::plain(KeyCode::Char('j')),
            KeyMatch::plain(KeyCode::Enter),
            KeyMatch::plain(KeyCode::Char(' ')),
        ],
    },
    KeyBinding {
        context: BindContext::BlockNav,
        keys: "Esc · Ctrl-B",
        help: "exit block nav",
        in_help: true,
        action: Some(Action::BlockNavExit),
        matches: &[
            KeyMatch::plain(KeyCode::Esc),
            KeyMatch::ctrl(KeyCode::Char('b')),
        ],
    },
    // --- Diff review (full-screen) ---
    KeyBinding {
        context: BindContext::Review,
        keys: "j/k · PgUp/PgDn",
        help: "scroll the diff",
        in_help: true,
        action: Some(Action::ReviewScroll { delta: 1 }),
        matches: &[
            KeyMatch::plain(KeyCode::Char('j')),
            KeyMatch::plain(KeyCode::Down),
            KeyMatch::plain(KeyCode::Char('k')),
            KeyMatch::plain(KeyCode::Up),
            KeyMatch::plain(KeyCode::PageDown),
            KeyMatch::plain(KeyCode::PageUp),
            KeyMatch::plain(KeyCode::Char('G')),
        ],
    },
    KeyBinding {
        context: BindContext::Review,
        keys: "n/p",
        help: "next / previous hunk",
        in_help: true,
        action: Some(Action::ReviewHunk { dir: 1 }),
        matches: &[
            KeyMatch::plain(KeyCode::Char('n')),
            KeyMatch::plain(KeyCode::Char('p')),
        ],
    },
    KeyBinding {
        context: BindContext::Review,
        keys: "q · Esc · Ctrl-G",
        help: "close review",
        in_help: true,
        action: Some(Action::ReviewClose),
        matches: &[
            KeyMatch::plain(KeyCode::Char('q')),
            KeyMatch::plain(KeyCode::Esc),
            KeyMatch::ctrl(KeyCode::Char('g')),
        ],
    },
    // --- Confirm ---
    KeyBinding {
        context: BindContext::Confirm,
        keys: "y",
        help: "approve once",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Confirm,
        keys: "a",
        help: "always allow this session",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Confirm,
        keys: "p",
        help: "always allow this path prefix this session",
        in_help: true,
        action: None,
        matches: &[],
    },
    KeyBinding {
        context: BindContext::Confirm,
        keys: "n/Esc",
        help: "reject",
        in_help: true,
        action: None,
        matches: &[],
    },
];

/// Help overlay sections in display order.
const HELP_SECTIONS: &[BindContext] = &[
    BindContext::Input,
    BindContext::Navigation,
    BindContext::ReviewTools,
    BindContext::Queue,
    BindContext::Sessions,
    BindContext::Normal,
    BindContext::BlockNav,
    BindContext::Review,
    BindContext::Confirm,
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
    help_overlay_rows().len()
}

/// Resolve a key against [`KEY_BINDINGS`] for `surface`.
///
/// Multi-key rows (e.g. Ctrl-P/N) store a representative `action` and fix up
/// direction from the matched physical key here.
pub(crate) fn resolve_from_table(surface: KeySurface, key: &KeyEvent) -> Action {
    // Overlay and history-search swallow table chords (specialized handlers).
    if matches!(surface, KeySurface::Overlay | KeySurface::HistorySearch) {
        return Action::None;
    }

    for binding in KEY_BINDINGS {
        let Some(base) = binding.action else {
            continue;
        };
        if binding.matches.is_empty() {
            continue;
        }
        if !binding.context.surfaces().contains(&surface) {
            continue;
        }
        for m in binding.matches {
            if !m.matches(surface, key) {
                continue;
            }
            return refine_action(base, m, key);
        }
    }
    Action::None
}

/// Adjust dir/delta for multi-key bindings that share one `action` prototype.
fn refine_action(base: Action, m: &KeyMatch, key: &KeyEvent) -> Action {
    match base {
        Action::JumpPrompt { .. } => {
            let dir = match key.code {
                KeyCode::Char('n') | KeyCode::Char(']') => 1,
                _ => -1,
            };
            Action::JumpPrompt { dir }
        }
        Action::JumpError { .. } => {
            let dir = match key.code {
                KeyCode::Char(']') | KeyCode::Char('}') => 1,
                _ => -1,
            };
            Action::JumpError { dir }
        }
        Action::QueueSelectPrev | Action::QueueSelectNext => match key.code {
            KeyCode::Down => Action::QueueSelectNext,
            _ => Action::QueueSelectPrev,
        },
        Action::QueueMoveSelected { .. } => {
            let delta = if key.code == KeyCode::Down { 1 } else { -1 };
            Action::QueueMoveSelected { delta }
        }
        Action::BlockNavUp | Action::BlockNavDown | Action::BlockNavToggle => match key.code {
            KeyCode::Enter | KeyCode::Char(' ') => Action::BlockNavToggle,
            KeyCode::Down | KeyCode::Char('j') => Action::BlockNavDown,
            _ => Action::BlockNavUp,
        },
        Action::ReviewScroll { .. } => {
            let delta = match key.code {
                KeyCode::Char('j') | KeyCode::Down => 1,
                KeyCode::Char('k') | KeyCode::Up => -1,
                KeyCode::PageDown => 10,
                KeyCode::PageUp => -10,
                KeyCode::Char('G') => Action::REVIEW_SCROLL_END,
                _ => 1,
            };
            Action::ReviewScroll { delta }
        }
        Action::ReviewHunk { .. } => {
            let dir = if key.code == KeyCode::Char('p') { -1 } else { 1 };
            Action::ReviewHunk { dir }
        }
        other => {
            let _ = m;
            other
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, mods)
    }

    #[test]
    fn help_sections_are_non_empty() {
        let rows = help_overlay_rows();
        assert!(
            rows.len() > 10,
            "expected a full cheat sheet, got {}",
            rows.len()
        );
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

    #[test]
    fn every_action_binding_has_matches_or_is_slash_command() {
        for b in KEY_BINDINGS {
            if b.action.is_none() {
                assert!(
                    b.matches.is_empty(),
                    "help-only binding {} should not list matches",
                    b.keys
                );
                continue;
            }
            // Slash commands and Esc-enter-normal are action-tagged for docs but
            // have no physical matches in the table.
            if b.matches.is_empty() {
                assert!(
                    b.keys.starts_with('/') || b.keys.starts_with("Esc"),
                    "action binding {} needs matches or a slash/Esc exception",
                    b.keys
                );
            }
        }
    }

    #[test]
    fn table_resolves_insert_globals() {
        assert_eq!(
            resolve_from_table(
                KeySurface::Insert,
                &key(KeyCode::Char('d'), KeyModifiers::CONTROL)
            ),
            Action::ToggleDiff
        );
        assert_eq!(
            resolve_from_table(
                KeySurface::Insert,
                &key(KeyCode::Char('k'), KeyModifiers::CONTROL)
            ),
            Action::OpenPalette
        );
        assert_eq!(
            resolve_from_table(
                KeySurface::Insert,
                &key(KeyCode::Char('n'), KeyModifiers::CONTROL)
            ),
            Action::JumpPrompt { dir: 1 }
        );
        assert_eq!(
            resolve_from_table(
                KeySurface::Normal,
                &key(KeyCode::Char('n'), KeyModifiers::CONTROL)
            ),
            Action::None,
            "Ctrl-N is insert-only"
        );
    }

    #[test]
    fn table_resolves_review_and_block_nav() {
        assert_eq!(
            resolve_from_table(KeySurface::Review, &key(KeyCode::Esc, KeyModifiers::NONE)),
            Action::ReviewClose
        );
        assert_eq!(
            resolve_from_table(
                KeySurface::Review,
                &key(KeyCode::Char('n'), KeyModifiers::NONE)
            ),
            Action::ReviewHunk { dir: 1 }
        );
        assert_eq!(
            resolve_from_table(
                KeySurface::BlockNav,
                &key(KeyCode::Enter, KeyModifiers::NONE)
            ),
            Action::BlockNavToggle
        );
        assert_eq!(
            resolve_from_table(
                KeySurface::BlockNav,
                &key(KeyCode::Char('k'), KeyModifiers::NONE)
            ),
            Action::BlockNavUp
        );
    }

    #[test]
    fn overlay_and_history_search_swallow() {
        assert_eq!(
            resolve_from_table(
                KeySurface::Overlay,
                &key(KeyCode::Char('d'), KeyModifiers::CONTROL)
            ),
            Action::None
        );
        assert_eq!(
            resolve_from_table(
                KeySurface::HistorySearch,
                &key(KeyCode::Char('d'), KeyModifiers::CONTROL)
            ),
            Action::None
        );
    }

    #[test]
    fn action_chords_are_reachable_from_table() {
        // Every physical match row must resolve on an allowed surface.
        for b in KEY_BINDINGS.iter().filter(|b| !b.matches.is_empty()) {
            let surfaces = b.context.surfaces();
            assert!(
                !surfaces.is_empty(),
                "binding {} has matches but no surfaces",
                b.keys
            );
            for m in b.matches {
                let surface = m.only_surface.unwrap_or(surfaces[0]);
                let mut mods = KeyModifiers::NONE;
                if m.ctrl {
                    mods |= KeyModifiers::CONTROL;
                }
                if m.alt {
                    mods |= KeyModifiers::ALT;
                }
                if m.shift {
                    mods |= KeyModifiers::SHIFT;
                }
                let got = resolve_from_table(surface, &key(m.code, mods));
                assert_ne!(
                    got,
                    Action::None,
                    "binding {} match {:?} did not resolve on {:?}",
                    b.keys,
                    m.code,
                    surface
                );
            }
        }
    }
}
