//! Logical domain groupings for [`crate::App`] state.
//!
//! Field storage still lives on `App` for call-site stability. Domain types own
//! the **rules** for one concern (what counts as a hard overlay, density status
//! text, etc.) so dispatch/render/commands don't re-encode them.

use crate::App;

/// Composer: input line, queue, completion.
pub(crate) struct ComposerDomain;

impl ComposerDomain {
    #[inline]
    pub(crate) fn queue_is_empty(app: &App) -> bool {
        app.queue.is_empty()
    }

    #[inline]
    pub(crate) fn input_empty(app: &App) -> bool {
        app.input.is_empty()
    }
}

/// Turn chrome: working spinner, density labels.
pub(crate) struct TurnChromeDomain;

impl TurnChromeDomain {
    /// Status line after `/density` or [`crate::action::Action::CycleDensity`].
    pub(crate) fn density_status(app: &App) -> String {
        format!(
            "density: {} · tool output: {}",
            app.density.label(),
            if app.density.show_tool_output(app.show_tool_output) {
                "expanded"
            } else {
                "folded"
            }
        )
    }
}

/// Overlays that fully own the keyboard (confirm, pickers, forms, palette).
///
/// Soft chrome (help / debug / diff panels) does **not** count — global chords
/// still apply underneath those.
pub(crate) struct OverlayDomain;

impl OverlayDomain {
    /// True when a hard keyboard-owning overlay is up.
    #[inline]
    pub(crate) fn any_hard(app: &App) -> bool {
        app.confirmation.is_some()
            || app.picker.is_some()
            || app.provider_picker.is_some()
            || app.provider_form.is_some()
            || app.palette.is_some()
    }

    #[inline]
    pub(crate) fn palette_open(app: &App) -> bool {
        app.palette.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_app;

    #[test]
    fn hard_overlay_tracks_palette_and_ignores_help() {
        let mut app = test_app("openai", "gpt-4o");
        assert!(!OverlayDomain::any_hard(&app));
        app.show_help = true;
        assert!(
            !OverlayDomain::any_hard(&app),
            "help is soft chrome, not a hard overlay"
        );
        app.palette = Some(crate::palette::CommandPalette::open());
        assert!(OverlayDomain::any_hard(&app));
        assert!(OverlayDomain::palette_open(&app));
    }

    #[test]
    fn density_status_mentions_label() {
        let mut app = test_app("openai", "gpt-4o");
        app.density = crate::Density::Compact;
        let s = TurnChromeDomain::density_status(&app);
        assert!(s.contains("compact"), "{s}");
        assert!(s.contains("tool output"), "{s}");
    }

    #[test]
    fn composer_empty_helpers() {
        let mut app = test_app("openai", "gpt-4o");
        assert!(ComposerDomain::input_empty(&app));
        assert!(ComposerDomain::queue_is_empty(&app));
        app.input.set("x");
        app.queue.push_back("y".into());
        assert!(!ComposerDomain::input_empty(&app));
        assert!(!ComposerDomain::queue_is_empty(&app));
    }
}
