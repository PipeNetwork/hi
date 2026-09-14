//! Permission and ask-user overlay: j/k select, Enter activate, followup reject.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_harness::ConfirmationRequest;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;

use crate::App;
use crate::render::dim;
use crate::theme::theme;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ConfirmFocus {
    #[default]
    Options,
    Followup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PermAction {
    Approve,
    AlwaysSession,
    AlwaysPath,
    Reject,
    RejectFollowup,
}

#[derive(Debug)]
pub(crate) enum ConfirmDecision {
    Redraw,
    Approve,
    AlwaysSession,
    AlwaysPath,
    Reject,
    RejectFollowup(String),
    Cancel,
    Ask(String),
    Unhandled,
}

fn perm_actions(request: &ConfirmationRequest) -> Vec<PermAction> {
    match request {
        ConfirmationRequest::FileEdit { path, .. } => {
            let mut actions = vec![PermAction::Approve, PermAction::AlwaysSession];
            if App::can_scope_auto_approve_path(path) {
                actions.push(PermAction::AlwaysPath);
            }
            actions.extend([PermAction::Reject, PermAction::RejectFollowup]);
            actions
        }
        ConfirmationRequest::ShellMutation { .. } => {
            vec![
                PermAction::Approve,
                PermAction::Reject,
                PermAction::RejectFollowup,
            ]
        }
    }
}

fn perm_label(action: PermAction, _request: &ConfirmationRequest) -> &'static str {
    match action {
        PermAction::Approve => "Approve once",
        PermAction::AlwaysSession => "Always allow file edits this session",
        PermAction::AlwaysPath => "Always allow this path prefix this session",
        PermAction::Reject => "Reject",
        PermAction::RejectFollowup => "Reject and follow up",
    }
}

pub(crate) fn clamp_selected(app: &mut App, request: &ConfirmationRequest) {
    let n = perm_actions(request).len().max(1);
    if app.confirmation_selected >= n {
        app.confirmation_selected = n - 1;
    }
}

pub(crate) fn hint(request: &ConfirmationRequest, focus: ConfirmFocus, waiting: usize) -> String {
    let extra = if waiting > 0 {
        format!(" · {waiting} waiting ")
    } else {
        String::new()
    };
    let actions = perm_actions(request);
    let always = actions.contains(&PermAction::AlwaysSession);
    match focus {
        ConfirmFocus::Followup => " type a follow-up · Enter send · Esc back ".into(),
        ConfirmFocus::Options => {
            if always {
                format!(
                    " j/k · Enter · y approve · a always · /yolo · Shift-Tab · n/Esc reject · x follow up{extra}"
                )
            } else {
                format!(
                    " j/k · Enter · y approve · /yolo · Shift-Tab · n/Esc reject · x follow up{extra}"
                )
            }
        }
    }
}

pub(crate) fn option_lines(app: &App, request: &ConfirmationRequest) -> Vec<Line<'static>> {
    let th = theme();
    perm_actions(request)
        .into_iter()
        .enumerate()
        .map(|(i, action)| {
            let selected = app.confirmation_selected == i;
            let label = perm_label(action, request);
            if selected {
                Line::styled(
                    format!("▶ {label}"),
                    Style::default()
                        .fg(th.accent_plan)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                Line::styled(format!("  {label}"), dim())
            }
        })
        .collect()
}

pub(crate) fn handle_key(
    app: &mut App,
    key: &KeyEvent,
    request: &ConfirmationRequest,
) -> ConfirmDecision {
    clamp_selected(app, request);
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    handle_perm(app, key, request, ctrl)
}

fn handle_perm(
    app: &mut App,
    key: &KeyEvent,
    request: &ConfirmationRequest,
    ctrl: bool,
) -> ConfirmDecision {
    let actions = perm_actions(request);
    if app.confirm_focus == ConfirmFocus::Followup {
        match key.code {
            KeyCode::Esc => {
                app.confirm_focus = ConfirmFocus::Options;
                ConfirmDecision::Redraw
            }
            KeyCode::Enter => {
                let text = app.ask_user_draft.trim().to_string();
                app.ask_user_draft.clear();
                app.confirm_focus = ConfirmFocus::Options;
                ConfirmDecision::RejectFollowup(text)
            }
            KeyCode::Backspace => {
                app.ask_user_draft.pop();
                ConfirmDecision::Redraw
            }
            KeyCode::Char(c) if !ctrl => {
                app.ask_user_draft.push(c);
                ConfirmDecision::Redraw
            }
            KeyCode::Char('c') if ctrl => ConfirmDecision::Cancel,
            _ => ConfirmDecision::Redraw,
        }
    } else {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') if !ctrl => {
                app.confirmation_selected = app.confirmation_selected.saturating_sub(1);
                ConfirmDecision::Redraw
            }
            KeyCode::Down | KeyCode::Char('j') if !ctrl => {
                let max = actions.len().saturating_sub(1);
                app.confirmation_selected = (app.confirmation_selected + 1).min(max);
                ConfirmDecision::Redraw
            }
            KeyCode::Char('y') if !ctrl => ConfirmDecision::Approve,
            KeyCode::Char('a') if !ctrl => {
                if perm_actions(request).contains(&PermAction::AlwaysSession) {
                    ConfirmDecision::AlwaysSession
                } else {
                    ConfirmDecision::Redraw
                }
            }
            KeyCode::Char('p') if !ctrl => {
                if actions.contains(&PermAction::AlwaysPath) {
                    ConfirmDecision::AlwaysPath
                } else {
                    ConfirmDecision::Redraw
                }
            }
            KeyCode::Char('n') if !ctrl => ConfirmDecision::Reject,
            KeyCode::Char('x') if !ctrl => {
                app.confirm_focus = ConfirmFocus::Followup;
                ConfirmDecision::Redraw
            }
            KeyCode::Esc => ConfirmDecision::Reject,
            KeyCode::Char('c') if ctrl => ConfirmDecision::Cancel,
            KeyCode::Enter => match actions.get(app.confirmation_selected).copied() {
                Some(PermAction::Approve) => ConfirmDecision::Approve,
                Some(PermAction::AlwaysSession) => ConfirmDecision::AlwaysSession,
                Some(PermAction::AlwaysPath) => ConfirmDecision::AlwaysPath,
                Some(PermAction::Reject) => ConfirmDecision::Reject,
                Some(PermAction::RejectFollowup) => {
                    app.confirm_focus = ConfirmFocus::Followup;
                    ConfirmDecision::Redraw
                }
                None => ConfirmDecision::Redraw,
            },
            KeyCode::PageUp => {
                app.confirmation_scroll = app.confirmation_scroll.saturating_sub(10);
                ConfirmDecision::Redraw
            }
            KeyCode::PageDown => {
                app.confirmation_scroll = app.confirmation_scroll.saturating_add(10);
                ConfirmDecision::Redraw
            }
            _ => ConfirmDecision::Unhandled,
        }
    }
}
