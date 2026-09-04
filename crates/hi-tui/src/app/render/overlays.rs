//! Full-screen inspection surfaces, suspended while a tool needs approval.

use crate::App;
use ratatui::{Frame, layout::Rect};

impl App {
    pub(super) fn render_fullscreen_overlay(&mut self, frame: &mut Frame, area: Rect) -> bool {
        if let Some(tutorial) = &self.tutorial {
            crate::tutorial::render(frame, area, tutorial);
            return true;
        }
        if let Some(overlay) = &self.workflow_overlay {
            crate::workflow_tui::render_overlay(frame, area, overlay);
            return true;
        }
        if self.inspect_subagent.is_some() {
            crate::subagent_overlay::render_inspect(frame, area, self);
            return true;
        }
        if let Some(overlay) = &self.tasks_overlay {
            crate::subagent_overlay::render_tasks(frame, area, overlay);
            return true;
        }
        if self.block_viewer.is_some() {
            crate::block_viewer::render(frame, area, self);
            return true;
        }
        if let Some(picker) = &self.jump_picker {
            crate::session_pickers::render_jump(frame, area, picker);
            return true;
        }
        if let Some(picker) = &self.rewind_picker {
            crate::session_pickers::render_rewind(frame, area, picker);
            return true;
        }
        if let Some(browser) = &self.memory_browser {
            crate::memory_browser::render(frame, area, browser);
            return true;
        }
        if let Some(overlay) = &self.diff_lab {
            overlay.render(frame, area);
            return true;
        }
        if let Some(overlay) = &self.race {
            overlay.render(frame, area);
            return true;
        }
        // Full-screen diff review overlay (Ctrl-G): takes over the whole screen
        // with a scrollable, syntax-colored diff and hunk navigation. Rendered
        // before the normal layout and returned early so it's truly modal.
        if self.mode.is_review() {
            self.render_review(frame, area);
            return true;
        }
        false
    }
}
