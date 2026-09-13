//! Full-screen inspection surfaces, suspended while a tool needs approval.

use crate::App;
use ratatui::{Frame, layout::Rect};

impl App {
    pub(super) fn render_fullscreen_overlay(&mut self, frame: &mut Frame, area: Rect) -> bool {
        if let Some(tutorial) = &self.tutorial {
            crate::tutorial::render(frame, area, tutorial);
            return true;
        }
        if crate::dashboard::is_open(self) {
            let spinner = self.spinner;
            if let Some(overlay) = self.dashboard.as_mut() {
                crate::dashboard::render(frame, area, overlay, spinner);
            }
            return true;
        }
        if self.usage_overlay.is_some() {
            let (th, _) = crate::theme::snapshot();
            crate::chrome::fill_background(frame, area, &th);
            if let Some(overlay) = self.usage_overlay.as_mut() {
                crate::usage::render(frame, area, overlay);
            }
            return true;
        }
        if self.block_viewer.is_some() {
            crate::block_viewer::render(frame, area, self);
            return true;
        }
        if self.review_is_overlay() {
            self.render_review_overlay(frame, area);
            return true;
        }
        false
    }
}
