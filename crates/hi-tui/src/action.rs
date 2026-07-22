//! Key → [`Action`] resolution.
//!
//! Runtime chords resolve through [`resolve_key`], which reads the declarative
//! table in [`crate::keys`]. Complex editing still lives in `edit_key` / mode
//! handlers; this module owns the chords that map cleanly to a single action.
//!
//! [`Action`] and [`KeySurface`] are defined here and re-used by `keys` via the
//! binding table's action column — `keys` depends only on these types, while
//! resolution logic that needs the table lives in `keys::resolve_from_table`
//! and is called from here (no cycle: `keys` does not call back into `action`).

use crossterm::event::KeyEvent;

use crate::keys;
use crate::mode::UiMode;

/// A discrete user intent produced by key resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Action {
    /// No binding matched.
    None,
    ToggleHelp,
    ToggleDebug,
    ToggleDiff,
    ToggleReview,
    ToggleReasoning,
    ToggleToolOutput,
    ToggleBlockNav,
    /// Esc on empty idle input — applied by `run.rs`, not the table matcher.
    EnterNormal,
    ExitToInsert,
    CopyLastCode,
    ExternalEdit,
    /// Cycled via `/density` (table lists it for help; applicator supports it).
    CycleDensity,
    OpenPalette,
    /// Jump to next/prev user prompt in the transcript (`dir` = ±1).
    JumpPrompt {
        dir: i32,
    },
    /// Jump to next/prev error/failed line (`dir` = ±1).
    JumpError {
        dir: i32,
    },
    QueueSelectPrev,
    QueueSelectNext,
    QueueRemoveSelected,
    QueueMoveSelected {
        delta: i32,
    },
    ReviewClose,
    ReviewScroll {
        delta: i32,
    },
    ReviewHunk {
        dir: i32,
    },
    BlockNavUp,
    BlockNavDown,
    BlockNavToggle,
    BlockNavExit,
}

impl Action {
    /// Sentinel `ReviewScroll.delta` meaning "jump to end of diff".
    pub(crate) const REVIEW_SCROLL_END: i32 = i32::MAX / 4;
}

/// Which surface currently owns the keyboard for resolution purposes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KeySurface {
    Insert,
    Normal,
    BlockNav,
    HistorySearch,
    Review,
    /// Confirmation modal, pickers, forms, palette — specialized handlers own keys.
    Overlay,
}

impl KeySurface {
    pub(crate) fn from_app(mode: &UiMode, has_hard_overlay: bool) -> Self {
        if has_hard_overlay {
            return Self::Overlay;
        }
        match mode {
            UiMode::Insert => Self::Insert,
            UiMode::Normal { .. } => Self::Normal,
            UiMode::BlockNav => Self::BlockNav,
            UiMode::HistorySearch(_) => Self::HistorySearch,
            UiMode::Review => Self::Review,
        }
    }
}

/// Resolve a keypress on `surface` into an [`Action`] via [`keys::KEY_BINDINGS`].
pub(crate) fn resolve_key(surface: KeySurface, key: &KeyEvent) -> Action {
    keys::resolve_from_table(surface, key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyModifiers};

    fn key(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, mods)
    }

    #[test]
    fn insert_global_chords_resolve() {
        assert_eq!(
            resolve_key(
                KeySurface::Insert,
                &key(KeyCode::Char('d'), KeyModifiers::CONTROL)
            ),
            Action::ToggleDiff
        );
        assert_eq!(
            resolve_key(
                KeySurface::Insert,
                &key(KeyCode::Char('k'), KeyModifiers::CONTROL)
            ),
            Action::OpenPalette
        );
        assert_eq!(
            resolve_key(
                KeySurface::Insert,
                &key(KeyCode::Char('g'), KeyModifiers::CONTROL)
            ),
            Action::ToggleReview
        );
    }

    #[test]
    fn review_keys_resolve() {
        assert_eq!(
            resolve_key(KeySurface::Review, &key(KeyCode::Esc, KeyModifiers::NONE)),
            Action::ReviewClose
        );
        assert_eq!(
            resolve_key(
                KeySurface::Review,
                &key(KeyCode::Char('n'), KeyModifiers::NONE)
            ),
            Action::ReviewHunk { dir: 1 }
        );
        assert_eq!(
            resolve_key(
                KeySurface::Review,
                &key(KeyCode::Char('G'), KeyModifiers::NONE)
            ),
            Action::ReviewScroll {
                delta: Action::REVIEW_SCROLL_END
            }
        );
    }

    #[test]
    fn block_nav_keys_resolve() {
        assert_eq!(
            resolve_key(
                KeySurface::BlockNav,
                &key(KeyCode::Enter, KeyModifiers::NONE)
            ),
            Action::BlockNavToggle
        );
        assert_eq!(
            resolve_key(
                KeySurface::BlockNav,
                &key(KeyCode::Char('k'), KeyModifiers::NONE)
            ),
            Action::BlockNavUp
        );
    }

    #[test]
    fn overlay_swallows_globals() {
        assert_eq!(
            resolve_key(
                KeySurface::Overlay,
                &key(KeyCode::Char('d'), KeyModifiers::CONTROL)
            ),
            Action::None
        );
    }

    #[test]
    fn normal_exit_and_jumps() {
        assert_eq!(
            resolve_key(
                KeySurface::Normal,
                &key(KeyCode::Char('i'), KeyModifiers::NONE)
            ),
            Action::ExitToInsert
        );
        assert_eq!(
            resolve_key(
                KeySurface::Normal,
                &key(KeyCode::Char(']'), KeyModifiers::NONE)
            ),
            Action::JumpPrompt { dir: 1 }
        );
        assert_eq!(
            resolve_key(
                KeySurface::Normal,
                &key(KeyCode::Char('{'), KeyModifiers::NONE)
            ),
            Action::JumpError { dir: -1 }
        );
    }

    #[test]
    fn queue_chords_on_insert() {
        assert_eq!(
            resolve_key(KeySurface::Insert, &key(KeyCode::Up, KeyModifiers::ALT)),
            Action::QueueSelectPrev
        );
        assert_eq!(
            resolve_key(
                KeySurface::Insert,
                &key(KeyCode::Down, KeyModifiers::ALT | KeyModifiers::SHIFT)
            ),
            Action::QueueMoveSelected { delta: 1 }
        );
    }
}
