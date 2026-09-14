//! Idle-loop dashboard and model-picker keys.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_harness::Harness;
use ratatui::text::Line;

use crate::App;
use crate::render::dim;

use super::session::apply_model;

pub(super) fn handle_dashboard_event(app: &mut App, harness: &Harness, key: &KeyEvent) -> bool {
    if crate::dashboard::is_open(app) {
        let action = app.dashboard.as_mut().expect("dashboard").handle_key(key);
        if action == crate::dashboard::DashAction::Close {
            crate::dashboard::hide(app);
            return true;
        }
        if let Some(overlay) = app.dashboard.as_mut() {
            crate::dashboard::apply_action(overlay, action);
        }
        return true;
    }
    if crate::dashboard::is_dashboard_toggle(key) {
        if let Err(err) = crate::dashboard::open_from_harness(app, harness) {
            app.push(Line::styled(err, dim()));
        }
        return true;
    }
    false
}

pub(super) fn handle_picker_key(app: &mut App, harness: &mut Harness, key: &KeyEvent) -> bool {
    if app.picker.is_none() {
        return false;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Esc => app.picker = None,
        KeyCode::Enter => {
            let id = app
                .picker
                .as_ref()
                .and_then(|picker| picker.current().map(str::to_string));
            app.picker = None;
            if let Some(id) = id {
                apply_model(app, &harness.live(), &id, None);
                harness.set_model(app.model.clone());
            }
        }
        KeyCode::Up => {
            if let Some(picker) = app.picker.as_mut() {
                picker.up();
            }
        }
        KeyCode::Down => {
            if let Some(picker) = app.picker.as_mut() {
                picker.down();
            }
        }
        KeyCode::PageUp => {
            if let Some(picker) = app.picker.as_mut() {
                picker.page_up();
            }
        }
        KeyCode::PageDown => {
            if let Some(picker) = app.picker.as_mut() {
                picker.page_down();
            }
        }
        KeyCode::Backspace => {
            if let Some(picker) = app.picker.as_mut() {
                picker.backspace();
            }
        }
        KeyCode::Char(c) if !ctrl => {
            if let Some(picker) = app.picker.as_mut() {
                picker.insert(c);
            }
        }
        _ => {}
    }
    true
}
