//! Docked review pane, overlay fallback, hunk citation, and mouse select.

use super::{dump, test_app};
use crate::action::Action;
use crate::dispatch::DispatchResult;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::Terminal;
use ratatui::backend::TestBackend;

const TWO_FILE: &str = "\
diff --git a/a.rs b/a.rs
--- a/a.rs
+++ b/a.rs
@@ -1,1 +1,1 @@
-old a
+new a
diff --git a/b.rs b/b.rs
--- a/b.rs
+++ b/b.rs
@@ -10,2 +10,3 @@
 ctx
-old b
+new b
";

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn open_docked(app: &mut crate::App) {
    app.frame_width = 120;
    app.open_review(None);
    app.review.diff_text = Some(TWO_FILE.into());
}

#[test]
fn wide_terminal_docks_review_beside_transcript() {
    let mut app = test_app("openai", "gpt-4o");
    app.push(ratatui::text::Line::raw("hello from the transcript"));
    open_docked(&mut app);
    assert!(app.review.open);
    assert!(
        !app.mode.is_review(),
        "docked review must not steal UiMode::Review"
    );

    let mut term = Terminal::new(TestBackend::new(120, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("hello from the transcript"),
        "transcript stays visible:\n{screen}"
    );
    assert!(
        screen.contains("Diff review"),
        "review pane title:\n{screen}"
    );
    assert!(
        !screen.contains("Diff review (Ctrl-G)"),
        "docked pane is not the overlay title:\n{screen}"
    );
    assert!(
        screen.contains("tab:focus diff") && screen.contains("ctrl+g:close"),
        "docked shortcuts live on the session bar:\n{screen}"
    );
}

#[test]
fn idle_shortcuts_advertise_ctrl_g_review() {
    let mut app = test_app("openai", "gpt-4o");
    let mut term = Terminal::new(TestBackend::new(120, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("ctrl+g:review"),
        "idle bar should advertise review:\n{screen}"
    );
}

#[test]
fn focused_docked_shortcuts_show_hunk_nav() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    app.review.focused = true;
    let mut term = Terminal::new(TestBackend::new(120, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("n/p:hunk") && screen.contains("esc:unfocus") && screen.contains("q:close"),
        "focused pane owns the session bar:\n{screen}"
    );
    assert!(
        !screen.contains("tab:focus diff"),
        "unfocus hint should not share the focused bar:\n{screen}"
    );
}

#[test]
fn narrow_terminal_keeps_exclusive_overlay() {
    let mut app = test_app("openai", "gpt-4o");
    app.frame_width = 80;
    app.open_review(None);
    app.review.diff_text = Some(TWO_FILE.into());
    assert!(app.mode.is_review());

    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Diff review (Ctrl-G)"),
        "overlay title:\n{screen}"
    );
    assert!(screen.contains("n/p hunks"), "overlay footer:\n{screen}");
}

#[test]
fn docked_unfocused_typing_goes_to_composer() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    assert!(!app.review.focused);

    let x = key(KeyCode::Char('x'));
    assert!(
        matches!(app.dispatch_key(&x), DispatchResult::Fallthrough),
        "unfocused pane must not swallow typing"
    );
    assert_eq!(app.edit_key(&x), None);
    assert_eq!(app.input.text(), "x");

    let j = key(KeyCode::Char('j'));
    assert!(matches!(app.dispatch_key(&j), DispatchResult::Fallthrough));
    assert_eq!(app.edit_key(&j), None);
    assert_eq!(app.input.text(), "xj");
}

#[test]
fn docked_focused_j_scrolls_instead_of_typing() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    app.review.focused = true;
    let j = key(KeyCode::Char('j'));
    assert!(matches!(app.dispatch_key(&j), DispatchResult::Handled));
    assert!(app.input.is_empty());
    assert!(app.review.scroll > 0);
}

#[test]
fn n_selects_hunk_and_writes_chip() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    app.review.focused = true;
    app.input.set("please undo");
    app.dispatch_key(&key(KeyCode::Char('n')));
    let text = app.input.text();
    assert!(
        text.starts_with("@b.rs:10-12") || text.starts_with("@a.rs:"),
        "chip prepended: {text}"
    );
    assert!(text.contains("please undo"), "{text}");
    assert!(app.review.selected_hunk.is_some());
}

#[test]
fn send_attaches_hunk_quote_when_chip_remains() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    app.review.focused = true;
    app.dispatch_key(&key(KeyCode::Char('n')));
    // Jump to the second hunk if n from 0 selected the first; n again.
    if app.review.selected_hunk == Some(0) {
        app.dispatch_key(&key(KeyCode::Char('n')));
    }
    let line = app.edit_key(&key(KeyCode::Enter)).expect("submit");
    assert!(line.contains("<review hunk>"), "{line}");
    assert!(
        prompt_still_mentions_a_file(&line),
        "mention survived submit: {line}"
    );
}

fn prompt_still_mentions_a_file(line: &str) -> bool {
    line.contains("@a.rs") || line.contains("@b.rs")
}

#[test]
fn deleting_chip_drops_the_quote() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    app.review.selected_hunk = Some(0);
    app.input.set("just a question, no mention");
    let line = app.edit_key(&key(KeyCode::Enter)).expect("submit");
    assert!(!line.contains("<review hunk>"), "{line}");
}

fn mouse_event(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind,
        column,
        row,
        modifiers: KeyModifiers::NONE,
    }
}

#[test]
fn mouse_click_selects_hunk_in_docked_pane() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    let mut term = Terminal::new(TestBackend::new(120, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let hit = app.review.rect;
    assert!(hit.width > 0 && hit.height > 2, "pane was painted: {hit:?}");
    app.handle_mouse(mouse_event(
        MouseEventKind::Down(MouseButton::Left),
        hit.x.saturating_add(2),
        hit.y.saturating_add(2),
    ));
    assert!(app.review.focused);
    assert!(app.review.selected_hunk.is_some());
    assert!(
        app.input.text().starts_with('@'),
        "click wrote a chip: {}",
        app.input.text()
    );
}

#[test]
fn dragging_the_pane_edge_resizes_the_review_box() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    let mut term = Terminal::new(TestBackend::new(120, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let hit = app.review.rect;
    let start_w = hit.width;
    assert!(start_w >= 36, "default pane width: {start_w}");
    let y = hit.y.saturating_add(2);
    app.handle_mouse(mouse_event(
        MouseEventKind::Down(MouseButton::Left),
        hit.x,
        y,
    ));
    assert!(app.review.resizing, "left edge starts a resize");
    assert!(
        app.review.selected_hunk.is_none(),
        "edge click must not select a hunk"
    );
    app.handle_mouse(mouse_event(
        MouseEventKind::Drag(MouseButton::Left),
        hit.x.saturating_sub(10),
        y,
    ));
    assert_eq!(app.review.width, Some(start_w.saturating_add(10)));
    term.draw(|f| app.render(f)).unwrap();
    assert_eq!(app.review.rect.width, start_w.saturating_add(10));
    app.handle_mouse(mouse_event(MouseEventKind::Up(MouseButton::Left), 0, y));
    assert!(!app.review.resizing);
}

#[test]
fn resize_drag_clamps_so_the_transcript_keeps_room() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    let mut term = Terminal::new(TestBackend::new(120, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let hit = app.review.rect;
    let y = hit.y.saturating_add(2);
    app.handle_mouse(mouse_event(
        MouseEventKind::Down(MouseButton::Left),
        hit.x,
        y,
    ));
    app.handle_mouse(mouse_event(MouseEventKind::Drag(MouseButton::Left), 0, y));
    let w = app.review.width.expect("width after drag");
    assert!(w >= 36, "min pane {w}");
    term.draw(|f| app.render(f)).unwrap();
    assert!(
        app.view_inner.width >= 24,
        "transcript kept a usable column: {}",
        app.view_inner.width
    );
}

#[test]
fn esc_unfocuses_docked_pane_without_closing() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    app.review.focused = true;
    app.apply_action(Action::ReviewUnfocus);
    assert!(app.review.open);
    assert!(!app.review.focused);
    app.apply_action(Action::ReviewClose);
    assert!(!app.review.open);
}

#[test]
fn ctrl_g_toggles_docked_pane() {
    let mut app = test_app("openai", "gpt-4o");
    app.frame_width = 120;
    app.apply_action(Action::ToggleReview);
    assert!(app.review.open);
    assert!(!app.mode.is_review());
    app.apply_action(Action::ToggleReview);
    assert!(!app.review.open);
}

#[test]
fn space_from_focused_review_returns_to_the_prompt() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    app.review.focused = true;
    app.following = false;
    assert!(matches!(
        app.dispatch_key(&key(KeyCode::Char(' '))),
        DispatchResult::Handled
    ));
    assert!(!app.review.focused);
    assert!(!app.mode.is_review());
    assert!(app.following);
    assert!(app.input.is_empty(), "space must not type a character");
}

#[test]
fn space_from_overlay_review_closes_and_lands_on_prompt() {
    let mut app = test_app("openai", "gpt-4o");
    app.frame_width = 80;
    app.open_review(None);
    assert!(app.mode.is_review());
    app.dispatch_key(&key(KeyCode::Char(' ')));
    assert!(!app.review.open);
    assert!(!app.mode.is_review());
    assert!(app.following);
}

#[test]
fn space_when_scrolled_away_follows_without_typing() {
    let mut app = test_app("openai", "gpt-4o");
    app.following = false;
    assert!(matches!(
        app.dispatch_key(&key(KeyCode::Char(' '))),
        DispatchResult::Handled
    ));
    assert!(app.following);
    assert!(app.input.is_empty());
}

#[test]
fn space_while_following_types_a_space() {
    let mut app = test_app("openai", "gpt-4o");
    assert!(app.following);
    assert!(matches!(
        app.dispatch_key(&key(KeyCode::Char(' '))),
        DispatchResult::Fallthrough
    ));
    assert_eq!(app.edit_key(&key(KeyCode::Char(' '))), None);
    assert_eq!(app.input.text(), " ");
}
