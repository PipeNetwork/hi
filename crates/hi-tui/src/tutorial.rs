//! Opt-in, session-local tutorial modal. Offered once on a fresh TUI session.

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph, Wrap};

use crate::theme::{self, UiTone};

pub(crate) const LESSON_COUNT: usize = 8;

const LESSONS: [(&str, &str); LESSON_COUNT] = [
    (
        "Ask for outcomes",
        "Say what you want changed, what good looks like, and any constraints. Concrete prompts such as “fix the failing parser test without changing the public API” give hi a useful target and a finish line.",
    ),
    (
        "Tests decide",
        "hi’s distinguishing loop is verification. After edits it runs cargo test, pytest, go test, or a command you set with /verify. Failures go back to the model. /verify off turns that off; /status shows the current command.",
    ),
    (
        "Take it back",
        "hi does not nag for every edit. Before a mutating turn it checkpoints the tree. /undo restores the last turn’s files. Risky irreversible commands (sudo, force-push, curl|sh) are refused. Interrupt an in-flight turn with Ctrl-C.",
    ),
    (
        "Queue and steer",
        "Keep typing while hi works — the next message is queued and can steer the current turn. Ctrl-K opens a command palette grouped like /help: core first, type to find the rest.",
    ),
    (
        "Sessions and recovery",
        "Sessions preserve conversation and work. Use /sessions to inspect, rename, switch, or recover prior work; /retry returns to the last safe message checkpoint when a turn needs another attempt.",
    ),
    (
        "Goals",
        "Use /goal for long-horizon work. hi can keep a visible objective and sub-goal plan, pause when blocked, and continue across several focused turns instead of improvising one giant response.",
    ),
    (
        "Fleet",
        "Open /fleet to dispatch and monitor multiple agent sessions. Each row is its own git worktree; verified, non-overlapping diffs merge back. Best for independent tasks.",
    ),
    (
        "Workflows",
        "Use /workflow for scripted multi-phase work. Inspect available workflows, launch one with a clear objective, then follow its agents from the overlay or /fleet.",
    ),
];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TutorialOverlay {
    pub(crate) step: usize,
    pub(crate) scroll: u16,
}

impl TutorialOverlay {
    pub(crate) fn fresh() -> Self {
        Self::default()
    }

    fn previous(&mut self) {
        self.step = self.step.saturating_sub(1);
        self.scroll = 0;
    }

    fn next(&mut self) -> bool {
        if self.step + 1 == LESSON_COUNT {
            true
        } else {
            self.step += 1;
            self.scroll = 0;
            false
        }
    }

    fn scroll_by(&mut self, delta: i32) {
        self.scroll = if delta < 0 {
            self.scroll.saturating_sub(delta.unsigned_abs() as u16)
        } else {
            self.scroll.saturating_add(delta as u16)
        };
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TutorialOutcome {
    Continue,
    Close,
}

pub(crate) fn handle_key(overlay: &mut TutorialOverlay, key: &KeyEvent) -> TutorialOutcome {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => TutorialOutcome::Close,
        KeyCode::Left | KeyCode::Char('h') => {
            overlay.previous();
            TutorialOutcome::Continue
        }
        KeyCode::Right | KeyCode::Char('l') | KeyCode::Enter => {
            if overlay.next() {
                TutorialOutcome::Close
            } else {
                TutorialOutcome::Continue
            }
        }
        KeyCode::Up => {
            overlay.scroll_by(-1);
            TutorialOutcome::Continue
        }
        KeyCode::Down => {
            overlay.scroll_by(1);
            TutorialOutcome::Continue
        }
        KeyCode::PageUp => {
            overlay.scroll_by(-6);
            TutorialOutcome::Continue
        }
        KeyCode::PageDown => {
            overlay.scroll_by(6);
            TutorialOutcome::Continue
        }
        _ => TutorialOutcome::Continue,
    }
}

/// Offer the tutorial on a fresh TUI session that has never seen it.
pub(crate) fn should_offer(fresh_session: bool, skip_env_set: bool, already_offered: bool) -> bool {
    fresh_session && !skip_env_set && !already_offered
}

pub(crate) fn offered_marker_path() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
        .map(|p| p.join("hi").join("tutorial-offered"))
}

pub(crate) fn already_offered() -> bool {
    offered_marker_path().is_some_and(|p| p.is_file())
}

pub(crate) fn mark_offered() {
    let Some(path) = offered_marker_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, b"1\n");
}

fn centered(area: Rect) -> Rect {
    let width = area.width.saturating_sub(4).clamp(1, 76);
    let height = area.height.saturating_sub(2).clamp(1, 22);
    let vertical = Layout::vertical([Constraint::Length(height)])
        .flex(Flex::Center)
        .split(area)[0];
    Layout::horizontal([Constraint::Length(width)])
        .flex(Flex::Center)
        .split(vertical)[0]
}

pub(crate) fn render(frame: &mut ratatui::Frame, area: Rect, overlay: &TutorialOverlay) {
    let modal = centered(area);
    let th = crate::theme::theme();
    let (title, body) = LESSONS[overlay.step.min(LESSON_COUNT - 1)];
    let final_step = overlay.step + 1 == LESSON_COUNT;
    let lines = vec![
        Line::from(vec![
            Span::styled(
                format!("Lesson {} of {}  ", overlay.step + 1, LESSON_COUNT),
                Style::default()
                    .fg(th.accent_assistant)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(title, Style::default().add_modifier(Modifier::BOLD)),
        ]),
        Line::raw(""),
        Line::raw(body),
        Line::raw(""),
        Line::styled(
            if final_step {
                "←/h previous  ↑↓/PgUp/PgDn scroll  Enter finish  Esc/q close"
            } else {
                "←/h previous  ↑↓/PgUp/PgDn scroll  →/l/Enter next  Esc/q close"
            },
            Style::default().fg(th.text_secondary),
        ),
    ];
    frame.render_widget(Clear, modal);
    let block = theme::theme().panel_block(" hi tutorial ", UiTone::Info);
    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((overlay.scroll, 0)),
        modal,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_lessons_are_ask_verify_undo() {
        assert_eq!(LESSONS[0].0, "Ask for outcomes");
        assert_eq!(LESSONS[1].0, "Tests decide");
        assert_eq!(LESSONS[2].0, "Take it back");
        assert_eq!(LESSONS[6].0, "Fleet");
    }

    #[test]
    fn offers_only_once_on_a_fresh_session() {
        assert!(should_offer(true, false, false));
        assert!(!should_offer(false, false, false));
        assert!(!should_offer(true, true, false));
        assert!(!should_offer(true, false, true));
    }
}
