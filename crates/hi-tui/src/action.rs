//! Key → [`Action`] resolution and application.
//!
//! Global chords (and a growing set of mode-local keys) resolve through
//! [`resolve_key`] so help text in `keys.rs` and runtime behavior share one
//! vocabulary. Complex editing still lives in `edit_key` / mode handlers; this
//! module owns the chords that map cleanly to a single action.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::mode::UiMode;

/// A discrete user intent produced by key resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)] // some variants are resolved or reserved for keys-table parity
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
    EnterNormal,
    ExitToInsert,
    CopyLastCode,
    ExternalEdit,
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
    /// Submit is handled by the caller (needs the input string).
    // Mode-local review navigation.
    ReviewClose,
    ReviewScroll {
        delta: i32,
    },
    ReviewHunk {
        dir: i32,
    },
    // Block-nav.
    BlockNavUp,
    BlockNavDown,
    BlockNavToggle,
    BlockNavExit,
}

/// Which surface currently owns the keyboard for resolution purposes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KeySurface {
    Insert,
    Normal,
    BlockNav,
    HistorySearch,
    Review,
    /// Confirmation modal, pickers, forms — caller handles; we return None.
    Overlay,
}

impl KeySurface {
    pub(crate) fn from_app(
        mode: &UiMode,
        has_hard_overlay: bool, // confirmation / picker / form
    ) -> Self {
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

/// Resolve a keypress on `surface` into an [`Action`]. Returns [`Action::None`]
/// when the key should fall through to text editing or a specialized handler.
pub(crate) fn resolve_key(surface: KeySurface, key: &KeyEvent) -> Action {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);

    if surface == KeySurface::Overlay {
        return Action::None;
    }

    // Review owns almost all keys.
    if surface == KeySurface::Review {
        return match key.code {
            KeyCode::Char('q') | KeyCode::Esc => Action::ReviewClose,
            KeyCode::Char('g') if ctrl => Action::ReviewClose,
            KeyCode::Char('j') | KeyCode::Down => Action::ReviewScroll { delta: 1 },
            KeyCode::Char('k') | KeyCode::Up => Action::ReviewScroll { delta: -1 },
            KeyCode::PageDown => Action::ReviewScroll { delta: 10 },
            KeyCode::PageUp => Action::ReviewScroll { delta: -10 },
            KeyCode::Char('G') => Action::ReviewScroll { delta: i32::MAX / 4 },
            KeyCode::Char('n') => Action::ReviewHunk { dir: 1 },
            KeyCode::Char('p') => Action::ReviewHunk { dir: -1 },
            _ => Action::None,
        };
    }

    // Block-nav owns movement/fold.
    if surface == KeySurface::BlockNav {
        return match key.code {
            KeyCode::Esc => Action::BlockNavExit,
            KeyCode::Char('b') if ctrl => Action::BlockNavExit,
            KeyCode::Up | KeyCode::Char('k') => Action::BlockNavUp,
            KeyCode::Down | KeyCode::Char('j') => Action::BlockNavDown,
            KeyCode::Enter | KeyCode::Char(' ') => Action::BlockNavToggle,
            _ => Action::None,
        };
    }

    // History search is specialized — no global chords while filtering.
    if surface == KeySurface::HistorySearch {
        return Action::None;
    }

    // Queue chords (insert surface, alt held).
    if surface == KeySurface::Insert && alt {
        match key.code {
            KeyCode::Up if shift => return Action::QueueMoveSelected { delta: -1 },
            KeyCode::Down if shift => return Action::QueueMoveSelected { delta: 1 },
            KeyCode::Up => return Action::QueueSelectPrev,
            KeyCode::Down => return Action::QueueSelectNext,
            KeyCode::Backspace => return Action::QueueRemoveSelected,
            _ => {}
        }
    }

    // Global chords available in Insert and Normal (not while typing a normal-mode search).
    if matches!(surface, KeySurface::Insert | KeySurface::Normal) {
        match key.code {
            KeyCode::Char('d') if ctrl => return Action::ToggleDiff,
            KeyCode::Char('g') if ctrl => return Action::ToggleReview,
            KeyCode::Char('?') if ctrl => return Action::ToggleDebug,
            KeyCode::Char('t') if ctrl => return Action::ToggleReasoning,
            KeyCode::Char('o') if ctrl => return Action::ToggleToolOutput,
            KeyCode::Char('y') if ctrl => return Action::CopyLastCode,
            KeyCode::Char('b') if ctrl => return Action::ToggleBlockNav,
            KeyCode::Char('x') if ctrl => return Action::ExternalEdit,
            KeyCode::Char('k') if ctrl => return Action::OpenPalette,
            KeyCode::Char('p') if ctrl && !shift => return Action::JumpPrompt { dir: -1 },
            KeyCode::Char('n') if ctrl && !shift && surface == KeySurface::Insert => {
                // Ctrl-N next prompt — only in insert (normal mode uses `n` for search).
                return Action::JumpPrompt { dir: 1 };
            }
            KeyCode::Char('[') if ctrl => return Action::JumpError { dir: -1 },
            KeyCode::Char(']') if ctrl => return Action::JumpError { dir: 1 },
            KeyCode::Char('?') if !ctrl && surface == KeySurface::Insert => {
                // Caller checks empty input before treating as ToggleHelp.
                return Action::ToggleHelp;
            }
            _ => {}
        }
    }

    // Normal-mode exit chords also appear here for the action applicator.
    if surface == KeySurface::Normal {
        match key.code {
            KeyCode::Char('i') | KeyCode::Char('q') | KeyCode::Esc => {
                return Action::ExitToInsert;
            }
            KeyCode::Char('[') => return Action::JumpPrompt { dir: -1 },
            KeyCode::Char(']') => return Action::JumpPrompt { dir: 1 },
            KeyCode::Char('{') => return Action::JumpError { dir: -1 },
            KeyCode::Char('}') => return Action::JumpError { dir: 1 },
            _ => {}
        }
    }

    Action::None
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
