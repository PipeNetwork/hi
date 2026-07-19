//! Logical domain groupings for [`crate::App`] state.
//!
//! Full structural extraction is incremental: these modules document the seams
//! and host helpers that operate on one concern. Field storage still lives on
//! `App` so call sites stay stable; new code should prefer these entry points.

#![allow(dead_code)] // seams are intentionally referenced as the ownership map grows

use crate::App;

/// Transcript view: entries, scroll, follow, cache.
pub(crate) struct TranscriptDomain;

impl TranscriptDomain {
    #[inline]
    pub(crate) fn following(app: &App) -> bool {
        app.following
    }

    #[inline]
    pub(crate) fn bump(app: &mut App) {
        app.bump_transcript();
    }
}

/// Composer: input line, queue, completion.
pub(crate) struct ComposerDomain;

impl ComposerDomain {
    #[inline]
    pub(crate) fn queue_len(app: &App) -> usize {
        app.queue.len()
    }

    #[inline]
    pub(crate) fn input_empty(app: &App) -> bool {
        app.input.is_empty()
    }
}

/// Turn chrome: working spinner, density labels.
pub(crate) struct TurnChromeDomain;

impl TurnChromeDomain {
    #[inline]
    pub(crate) fn is_working(app: &App) -> bool {
        app.working
    }

    #[inline]
    pub(crate) fn density_status(app: &App) -> String {
        format!(
            "{} · tool output: {}",
            app.density.label(),
            if app.density.show_tool_output(app.show_tool_output) {
                "expanded"
            } else {
                "folded"
            }
        )
    }
}

/// Overlays: review, help, debug, diff, confirm, palette, pickers.
pub(crate) struct OverlayDomain;

impl OverlayDomain {
    #[inline]
    pub(crate) fn any_hard(app: &App) -> bool {
        app.has_hard_overlay()
    }

    #[inline]
    pub(crate) fn palette_open(app: &App) -> bool {
        app.palette.is_some()
    }
}
