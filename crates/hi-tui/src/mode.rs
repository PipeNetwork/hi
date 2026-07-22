//! Exclusive interaction mode for the main session TUI.
//!
//! Panels (help, debug, diff) stay independent flags — they overlay chrome
//! without owning the keyboard. Modes here are mutually exclusive and own
//! key dispatch until exited.

use crate::input::HistorySearch;

/// Keyboard-owning interaction mode. Exactly one is active.
#[derive(Clone, Debug, Default)]
pub(crate) enum UiMode {
    /// Default: composer accepts typing; global chords still work.
    #[default]
    Insert,
    /// Vim-style normal mode (Esc on empty input). Optional in-progress `/` search.
    Normal { search: Option<String> },
    /// Cursor over foldable tool-output blocks (Ctrl-B).
    BlockNav,
    /// Ctrl-R reverse history search.
    HistorySearch(HistorySearch),
    /// Full-screen diff review (Ctrl-G).
    Review,
}

impl UiMode {
    pub(crate) fn is_normal(&self) -> bool {
        matches!(self, Self::Normal { .. })
    }

    pub(crate) fn is_block_nav(&self) -> bool {
        matches!(self, Self::BlockNav)
    }

    pub(crate) fn is_review(&self) -> bool {
        matches!(self, Self::Review)
    }

    pub(crate) fn is_history_search(&self) -> bool {
        matches!(self, Self::HistorySearch(_))
    }

    pub(crate) fn normal_search(&self) -> Option<&str> {
        match self {
            Self::Normal {
                search: Some(q), ..
            } => Some(q.as_str()),
            _ => None,
        }
    }

    pub(crate) fn normal_search_mut(&mut self) -> Option<&mut Option<String>> {
        match self {
            Self::Normal { search } => Some(search),
            _ => None,
        }
    }

    pub(crate) fn history_search(&self) -> Option<&HistorySearch> {
        match self {
            Self::HistorySearch(s) => Some(s),
            _ => None,
        }
    }

    /// Leave any exclusive mode and return to insert.
    pub(crate) fn to_insert(&mut self) {
        *self = Self::Insert;
    }
}
