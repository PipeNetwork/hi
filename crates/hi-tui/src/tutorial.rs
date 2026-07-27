//! Opt-in, session-local tutorial modal.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph, Wrap};

pub(crate) const LESSON_COUNT: usize = 8;

const LESSONS: [(&str, &str); LESSON_COUNT] = [
    (
        "Ask for outcomes",
        "Say what you want changed, what good looks like, and any constraints. Concrete prompts such as “fix the failing parser test without changing the public API” give hi a useful target and a finish line.",
    ),
    (
        "Bring context",
        "Mention @path/to/file to attach focused file context. Use multiline input for detailed requests, or open the external editor when the prompt needs room for examples, logs, and acceptance criteria.",
    ),
    (
        "Tools and control",
        "hi reads, searches, edits, and runs checks as needed. Risky actions ask permission. Review the exact operation, approve only what you trust, and interrupt an in-flight turn with Ctrl-C whenever direction changes.",
    ),
    (
        "Queue and steer",
        "Submit another message while hi works to steer the current turn when possible; slash commands and remaining messages stay visibly queued for later. Reorder or remove queued work before it runs.",
    ),
    (
        "Sessions and recovery",
        "Sessions preserve conversation and work context. Use /sessions to inspect, rename, switch, or recover prior work; /retry returns to the last safe message checkpoint when a turn needs another attempt.",
    ),
    (
        "Goals and plans",
        "Use /goal for long-horizon work. hi can maintain a visible objective and sub-goal plan, report progress, pause when blocked, and continue across several focused turns instead of improvising one giant response.",
    ),
    (
        "Dashboard and fleet",
        "Open /dashboard to dispatch and monitor multiple agent sessions. Fleet work is best for independent tasks: keep scopes explicit, watch progress, and bring useful results back to the main session.",
    ),
    (
        "Workflows",
        "Use /workflow to browse scripted, repeatable multi-phase work. Inspect available workflows and runs, launch one with a clear objective, then follow its agents and phases from the workflow overlay or dashboard.",
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
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(th.accent_system))
        .title(" hi tutorial ");
    frame.render_widget(Clear, modal);
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
    use crossterm::event::KeyModifiers;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn navigation_resets_scroll_and_finishes() {
        let mut overlay = TutorialOverlay::fresh();
        handle_key(&mut overlay, &key(KeyCode::Down));
        assert_eq!(overlay.scroll, 1);
        assert_eq!(
            handle_key(&mut overlay, &key(KeyCode::Enter)),
            TutorialOutcome::Continue
        );
        assert_eq!((overlay.step, overlay.scroll), (1, 0));
        handle_key(&mut overlay, &key(KeyCode::Char('h')));
        assert_eq!(overlay.step, 0);
        overlay.step = LESSON_COUNT - 1;
        assert_eq!(
            handle_key(&mut overlay, &key(KeyCode::Char('l'))),
            TutorialOutcome::Close
        );
    }

    #[test]
    fn close_keys_close() {
        for code in [KeyCode::Esc, KeyCode::Char('q')] {
            assert_eq!(
                handle_key(&mut TutorialOverlay::fresh(), &key(code)),
                TutorialOutcome::Close
            );
        }
    }
}
