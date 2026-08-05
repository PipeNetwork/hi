//! Internal benchmark fixtures for the real TUI renderers.
//!
//! This module is public only so Cargo's external benchmark target can reach
//! the crate-private renderer seams. It is not a supported runtime API.

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::style::Style;
use ratatui::text::Line;

use crate::{App, ProfileInfo};

fn app() -> App {
    App::new(
        "openai",
        "gpt-4o",
        Vec::<ProfileInfo>::new(),
        None,
        Box::new(|_| anyhow::bail!("benchmark resolver")),
        Box::new(|_| anyhow::bail!("benchmark saver")),
        Box::new(|_| anyhow::bail!("benchmark loader")),
        Box::new(|_| anyhow::bail!("benchmark remover")),
        None,
        Box::new(|_| anyhow::bail!("benchmark mlx switcher")),
        None,
        String::new(),
    )
}

#[doc(hidden)]
pub struct SessionFixture {
    app: App,
    terminal: Terminal<TestBackend>,
    sequence: u64,
}

impl SessionFixture {
    #[doc(hidden)]
    pub fn new(width: u16, height: u16) -> Self {
        let mut app = app();
        let mut remaining = 10_000usize;
        let mut line = 0usize;
        while remaining > 0 {
            let text = format!("benchmark transcript line {line}: cached markdown output\n");
            remaining = remaining.saturating_sub(text.len());
            app.push(Line::raw(text));
            line += 1;
        }
        let mut fixture = Self {
            app,
            terminal: Terminal::new(TestBackend::new(width, height)).unwrap(),
            sequence: 0,
        };
        fixture.render_cache_hit();
        fixture
    }

    #[doc(hidden)]
    pub fn render_cache_hit(&mut self) {
        self.terminal.draw(|frame| self.app.render(frame)).unwrap();
    }

    #[doc(hidden)]
    pub fn render_rebuild(&mut self) {
        self.sequence = self.sequence.wrapping_add(1);
        self.app.pending = Some((
            Style::default(),
            false,
            format!("streamed benchmark chunk {}", self.sequence),
        ));
        self.app.transcript_gen = self.app.transcript_gen.wrapping_add(1);
        self.render_cache_hit();
    }

    #[doc(hidden)]
    pub fn render_full_rebuild(&mut self) {
        self.app.density = self.app.density.next();
        self.render_cache_hit();
    }
}

#[doc(hidden)]
pub struct DashboardFixture {
    app: App,
    terminal: Terminal<TestBackend>,
}

impl DashboardFixture {
    #[doc(hidden)]
    pub fn new(width: u16, height: u16, rows: usize) -> Self {
        let mut app = app();
        app.fleet = (1..=rows).map(crate::dashboard::benchmark_row).collect();
        Self {
            app,
            terminal: Terminal::new(TestBackend::new(width, height)).unwrap(),
        }
    }

    #[doc(hidden)]
    pub fn render(&mut self) {
        self.terminal
            .draw(|frame| crate::dashboard::render_benchmark_frame(frame, &self.app))
            .unwrap();
    }
}

#[doc(hidden)]
pub struct WatchFixture {
    rows: Vec<crate::loops::LoopWatchRow>,
    terminal: Terminal<TestBackend>,
    now: u64,
}

impl WatchFixture {
    #[doc(hidden)]
    pub fn new(width: u16, height: u16, rows: usize) -> Self {
        let now = 1_700_000_000_000;
        Self {
            rows: (1..=rows)
                .map(|id| crate::watch::benchmark_row(id as u64, now))
                .collect(),
            terminal: Terminal::new(TestBackend::new(width, height)).unwrap(),
            now,
        }
    }

    #[doc(hidden)]
    pub fn render(&mut self) {
        self.terminal
            .draw(|frame| crate::watch::render_benchmark_frame(frame, &self.rows, self.now))
            .unwrap();
    }
}
