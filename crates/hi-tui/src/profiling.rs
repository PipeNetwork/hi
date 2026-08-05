//! Opt-in renderer timing without touching the user-visible transcript.

use std::sync::OnceLock;
use std::time::Instant;

use ratatui::layout::Rect;

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("HI_TUI_PROFILE").as_deref() == Some(std::ffi::OsStr::new("1"))
    })
}

pub(crate) struct FrameTimer {
    view: &'static str,
    area: Rect,
    started: Option<Instant>,
}

impl FrameTimer {
    pub(crate) fn begin(view: &'static str, area: Rect) -> Self {
        Self {
            view,
            area,
            started: enabled().then(Instant::now),
        }
    }
}

impl Drop for FrameTimer {
    fn drop(&mut self) {
        let Some(started) = self.started else {
            return;
        };
        tracing::debug!(
            target: "hi_tui::render",
            view = self.view,
            width = self.area.width,
            height = self.area.height,
            elapsed_us = started.elapsed().as_micros() as u64,
            "tui frame"
        );
    }
}
