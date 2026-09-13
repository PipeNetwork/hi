//! Grok-build `/usage` modal: Usage limit, Context usage, Session info.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_harness::{UsageSnapshot, UsageTab};
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph, Wrap};

use crate::App;
use crate::theme::{UiTone, theme};
use crate::util::copy_to_clipboard;

#[derive(Clone, Debug)]
pub(crate) struct UsageOverlay {
    pub snapshot: UsageSnapshot,
    pub tab: UsageTab,
    pub scroll: u16,
    tab_hits: Vec<(Rect, UsageTab)>,
}

pub(crate) fn snapshot_from_app(app: &App) -> UsageSnapshot {
    let window = u64::from(
        app.context_window
            .unwrap_or(hi_harness::DEFAULT_CONTEXT_WINDOW)
            .max(1),
    );
    let occupancy = app.context_used.min(window);
    let free = window.saturating_sub(occupancy);
    UsageSnapshot {
        model: app.model.clone(),
        base_url: String::new(),
        permission: app.permission_mode.label().to_string(),
        reasoning: app
            .reasoning_effort
            .map(|effort| effort.as_str().to_string())
            .unwrap_or_else(|| "off".into()),
        sandbox: String::new(),
        signed_in: !app.api_key.is_empty(),
        workspace: app.workspace_root.clone(),
        session_id: None,
        user_turns: app
            .transcript
            .iter()
            .filter(|entry| matches!(entry, crate::TranscriptEntry::UserPrompt { .. }))
            .count() as u64,
        checkpoints: 0,
        window,
        occupancy,
        usage: app.session_totals,
        categories: vec![
            hi_harness::UsageCategory {
                name: "Context used",
                tokens: occupancy,
            },
            hi_harness::UsageCategory {
                name: "Free space",
                tokens: free,
            },
        ],
        info_rows: Vec::new(),
    }
}

impl UsageOverlay {
    pub fn new(snapshot: UsageSnapshot, tab: UsageTab) -> Self {
        Self {
            snapshot,
            tab,
            scroll: 0,
            tab_hits: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UsageOutcome {
    Continue,
    Close,
}

pub(crate) fn handle_key(overlay: &mut UsageOverlay, key: &KeyEvent) -> UsageOutcome {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => UsageOutcome::Close,
        KeyCode::Tab if !key.modifiers.contains(KeyModifiers::SHIFT) => {
            overlay.tab = overlay.tab.next();
            overlay.scroll = 0;
            UsageOutcome::Continue
        }
        KeyCode::BackTab | KeyCode::Left | KeyCode::Char('h') => {
            overlay.tab = overlay.tab.prev();
            overlay.scroll = 0;
            UsageOutcome::Continue
        }
        KeyCode::Right | KeyCode::Char('l') if !ctrl => {
            overlay.tab = overlay.tab.next();
            overlay.scroll = 0;
            UsageOutcome::Continue
        }
        KeyCode::Up | KeyCode::Char('k') => {
            overlay.scroll = overlay.scroll.saturating_sub(1);
            UsageOutcome::Continue
        }
        KeyCode::Down | KeyCode::Char('j') => {
            overlay.scroll = overlay.scroll.saturating_add(1);
            UsageOutcome::Continue
        }
        KeyCode::PageUp => {
            overlay.scroll = overlay.scroll.saturating_sub(6);
            UsageOutcome::Continue
        }
        KeyCode::PageDown => {
            overlay.scroll = overlay.scroll.saturating_add(6);
            UsageOutcome::Continue
        }
        KeyCode::Char('c') if !ctrl => UsageOutcome::Continue, // copy is applied by caller
        KeyCode::Char('y') if !ctrl => UsageOutcome::Continue,
        _ => UsageOutcome::Continue,
    }
}

pub(crate) fn copy_request(overlay: &UsageOverlay, key: &KeyEvent) -> Option<String> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Char('y') if !ctrl => Some(overlay.snapshot.copy_all()),
        KeyCode::Char('c') if !ctrl => Some(
            overlay
                .snapshot
                .session_id
                .clone()
                .unwrap_or_else(|| "(unsaved)".into()),
        ),
        _ => None,
    }
}

pub(crate) fn handle_click(overlay: &mut UsageOverlay, col: u16, row: u16) -> bool {
    if let Some((_, tab)) = overlay
        .tab_hits
        .iter()
        .find(|(rect, _)| crate::btw::cell_in(*rect, col, row))
    {
        overlay.tab = *tab;
        overlay.scroll = 0;
        return true;
    }
    false
}

pub(crate) fn render(frame: &mut ratatui::Frame, area: Rect, overlay: &mut UsageOverlay) {
    let modal = centered(area);
    let th = theme();
    let mut lines = vec![tab_line(overlay.tab), Line::raw("")];
    for line in overlay.snapshot.tab_text(overlay.tab).lines() {
        if line.chars().all(|c| c == '#' || c == '-') && line.len() >= 8 {
            lines.push(Line::styled(
                line.to_string(),
                Style::default().fg(th.accent_running),
            ));
        } else {
            lines.push(Line::raw(line.to_string()));
        }
    }
    lines.push(Line::raw(""));
    lines.push(Line::styled(
        "Tab switch · c copy session ID · y copy all · Esc close",
        Style::default().fg(th.text_secondary),
    ));
    lines.push(Line::styled(
        "click a tab or drag to copy",
        Style::default().fg(th.gray_dim),
    ));

    frame.render_widget(Clear, modal);
    let block = th.panel_block(" usage  ·  click or drag to copy ", UiTone::Info);
    let inner = block.inner(modal);
    frame.render_widget(block, modal);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((overlay.scroll, 0)),
        inner,
    );
    overlay.tab_hits = tab_hit_rects(inner);
}

fn tab_line(selected: UsageTab) -> Line<'static> {
    let th = theme();
    let mut spans = Vec::new();
    for (i, tab) in UsageTab::ALL.iter().enumerate() {
        if i > 0 {
            spans.push(Span::raw("   "));
        }
        let mut style = Style::default().fg(th.text_secondary);
        if *tab == selected {
            style = Style::default()
                .fg(th.text_primary)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
        }
        spans.push(Span::styled(tab.title().to_string(), style));
    }
    Line::from(spans)
}

fn tab_hit_rects(inner: Rect) -> Vec<(Rect, UsageTab)> {
    let mut x = inner.x;
    let y = inner.y;
    let mut hits = Vec::new();
    for (i, tab) in UsageTab::ALL.iter().enumerate() {
        if i > 0 {
            x = x.saturating_add(3);
        }
        let w = tab.title().chars().count() as u16;
        hits.push((
            Rect {
                x,
                y,
                width: w.max(1),
                height: 1,
            },
            *tab,
        ));
        x = x.saturating_add(w);
    }
    hits
}

fn centered(area: Rect) -> Rect {
    let width = area.width.saturating_sub(4).clamp(1, 78);
    let height = area.height.saturating_sub(2).clamp(1, 24);
    let vertical = Layout::vertical([Constraint::Length(height)])
        .flex(Flex::Center)
        .split(area)[0];
    Layout::horizontal([Constraint::Length(width)])
        .flex(Flex::Center)
        .split(vertical)[0]
}

pub(crate) fn apply_copy(app: &mut App, text: &str) {
    match copy_to_clipboard(text) {
        Ok(()) => {
            app.copy_toast = Some((text.chars().count(), std::time::Instant::now()));
        }
        Err(err) => {
            app.push(ratatui::text::Line::styled(
                format!("copy failed: {err}"),
                crate::render::dim(),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> UsageSnapshot {
        UsageSnapshot {
            model: "pipe/deepseek-v4-flash-0731".into(),
            base_url: "https://api.pipenetwork.ai/v1".into(),
            permission: "ask".into(),
            reasoning: "off".into(),
            sandbox: "off".into(),
            signed_in: true,
            workspace: "/tmp/hi".into(),
            session_id: Some("sess1".into()),
            user_turns: 2,
            checkpoints: 1,
            window: 128_000,
            occupancy: 4_000,
            usage: hi_ai::Usage {
                input_tokens: 3_000,
                output_tokens: 1_000,
                ..hi_ai::Usage::default()
            },
            categories: vec![
                hi_harness::UsageCategory {
                    name: "System prompt",
                    tokens: 200,
                },
                hi_harness::UsageCategory {
                    name: "Free space",
                    tokens: 124_000,
                },
            ],
            info_rows: vec![],
        }
    }

    #[test]
    fn tab_cycle_matches_grok_order() {
        let mut overlay = UsageOverlay::new(snapshot(), UsageTab::Limit);
        let tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
        handle_key(&mut overlay, &tab);
        assert_eq!(overlay.tab, UsageTab::Context);
        handle_key(&mut overlay, &tab);
        assert_eq!(overlay.tab, UsageTab::Session);
        handle_key(&mut overlay, &tab);
        assert_eq!(overlay.tab, UsageTab::Limit);
    }

    #[test]
    fn y_copies_every_tab() {
        let overlay = UsageOverlay::new(snapshot(), UsageTab::Limit);
        let y = KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE);
        let text = copy_request(&overlay, &y).expect("copy");
        assert!(text.contains("Usage limit"), "{text}");
        assert!(text.contains("Context usage"), "{text}");
        assert!(text.contains("Session info"), "{text}");
        assert!(text.contains("pipe/deepseek"), "{text}");
    }

    #[test]
    fn occupancy_bar_fills() {
        let bar = hi_harness::occupancy_bar(50, 10);
        assert_eq!(bar, "#####-----");
    }
}
