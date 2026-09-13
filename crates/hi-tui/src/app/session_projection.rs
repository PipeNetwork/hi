//! Presentation projection kept as a no-op without the old agent reducer.

use ratatui::text::Line;

use crate::App;
use crate::event::UiEvent;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProjectedBlockIdentity {
    pub id: String,
    pub terminal: Option<&'static str>,
}

#[derive(Default)]
pub(crate) struct PresentationProjection {
    enabled: bool,
}

impl PresentationProjection {
    fn configure(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl App {
    pub(crate) fn configure_session_projection_v2(&mut self, enabled: bool) {
        self.session_projection.configure(enabled);
    }

    pub(crate) fn reset_session_projection_v2(&mut self) {
        let enabled = self.session_projection.enabled;
        self.session_projection = PresentationProjection { enabled };
        self.transcript.clear();
    }

    pub(crate) fn apply(&mut self, event: UiEvent) {
        self.apply_legacy(event);
    }

    pub(crate) fn record_projected_user_prompt(&mut self, _line: &Line<'_>) {}
}
