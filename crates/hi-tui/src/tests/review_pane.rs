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

#[test]
fn transcript_hides_hunk_quote_until_thinking_expands() {
    let mut app = test_app("openai", "gpt-4o");
    open_docked(&mut app);
    app.review.focused = true;
    app.dispatch_key(&key(KeyCode::Char('n')));
    if app.review.selected_hunk == Some(0) {
        app.dispatch_key(&key(KeyCode::Char('n')));
    }
    let line = app.edit_key(&key(KeyCode::Enter)).expect("submit");
    app.push_user_prompt(ratatui::text::Line::raw(format!("❯ {line}")));

    let texts = |show_reasoning: bool| -> Vec<String> {
        app.transcript
            .iter()
            .flat_map(|entry| entry.flatten(show_reasoning, false, crate::Density::Comfortable))
            .map(|row| crate::render::line_text(&row))
            .collect()
    };

    let collapsed = texts(false);
    assert!(
        collapsed
            .iter()
            .any(|row| row.contains("❯") && prompt_still_mentions_a_file(row)),
        "question stays visible: {collapsed:?}"
    );
    assert!(
        !collapsed
            .iter()
            .any(|row| row.contains("new a") || row.contains("new b")),
        "hunk is hidden until thinking expands: {collapsed:?}"
    );
    assert!(
        !collapsed.iter().any(|row| row.contains("<review hunk>")),
        "markup must not leak: {collapsed:?}"
    );

    let expanded = texts(true);
    assert!(
        expanded
            .iter()
            .any(|row| row.contains("new a") || row.contains("new b")),
        "Ctrl-E thinking shows the hunk: {expanded:?}"
    );
    assert!(
        expanded.iter().any(|row| row.contains("review hunk")),
        "expanded thinking labels the hunk: {expanded:?}"
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

#[test]
fn parse_two_file_unified_diff() {
    let hunks = crate::review::parse_review_hunks(TWO_FILE);
    assert_eq!(hunks.len(), 2, "{hunks:?}");
    assert_eq!(hunks[0].path, "a.rs");
    assert_eq!(hunks[0].start, Some(1));
    assert_eq!(hunks[1].path, "b.rs");
    assert_eq!(hunks[1].start, Some(10));
    assert_eq!(hunks[1].end, Some(12));
    assert!(hunks[1].painted_start > hunks[0].painted_start);
}

#[test]
fn deleted_file_uses_old_path() {
    let diff = "\
--- a/gone.rs
+++ /dev/null
@@ -1,1 +0,0 @@
-bye
";
    let hunks = crate::review::parse_review_hunks(diff);
    assert_eq!(hunks[0].path, "gone.rs");
}

#[test]
fn chip_replaces_previous_review_mention() {
    let next = crate::review::replace_or_prepend_chip("@a.rs:1-2 please undo", "@b.rs:10-12");
    assert_eq!(next, "@b.rs:10-12 please undo");
    assert_eq!(
        crate::review::replace_or_prepend_chip("", "@a.rs:1"),
        "@a.rs:1 "
    );
    assert_eq!(
        crate::review::replace_or_prepend_chip("hello", "@a.rs:1"),
        "@a.rs:1 hello"
    );
}

#[test]
fn echo_strips_the_hunk_from_the_visible_prompt() {
    let hunks = crate::review::parse_review_hunks(TWO_FILE);
    let h = &hunks[1];
    let mention = h.mention().unwrap();
    let quoted = crate::review::attach_hunk_quote(&format!("{mention} why?"), h, TWO_FILE);
    let (visible, body) = crate::review::split_review_hunk_echo(&quoted);
    assert_eq!(visible, format!("{mention} why?"));
    let body = body.expect("hunk body");
    assert!(body.contains("new b"), "{body}");
    assert!(!body.contains("<review hunk>"), "{body}");
    assert_eq!(
        crate::review::split_review_hunk_echo("plain question"),
        ("plain question", None)
    );
}

#[test]
fn quote_requires_mention_and_is_capped() {
    let hunks = crate::review::parse_review_hunks(TWO_FILE);
    let h = &hunks[1];
    let mention = h.mention().unwrap();
    let quoted = crate::review::attach_hunk_quote(&format!("{mention} why?"), h, TWO_FILE);
    assert!(quoted.contains("<review hunk>"), "{quoted}");
    assert!(quoted.contains("b.rs:10-12"), "{quoted}");
    assert!(quoted.contains("new b"), "{quoted}");
    let dropped = crate::review::attach_hunk_quote("no chip here", h, TWO_FILE);
    assert_eq!(dropped, "no chip here");
}

#[test]
fn dock_only_on_wide_layout() {
    assert!(!crate::review::can_dock(80));
    assert!(!crate::review::can_dock(99));
    assert!(crate::review::can_dock(100));
    assert!(crate::review::can_dock(120));
}

#[test]
fn split_body_honors_width_and_clamps() {
    let area = ratatui::layout::Rect {
        x: 0,
        y: 0,
        width: 100,
        height: 20,
    };
    let (left, Some((gap, right))) = crate::review::split_body(area, true, Some(40)) else {
        panic!("expected docked split");
    };
    assert_eq!(right.width, 40);
    assert_eq!(gap.width, 1);
    assert_eq!(left.width + gap.width + right.width, 100);

    let (_, Some((_, narrow))) = crate::review::split_body(area, true, Some(10)) else {
        panic!("clamp min");
    };
    assert_eq!(narrow.width, 36);

    let (_, Some((_, wide))) = crate::review::split_body(area, true, Some(90)) else {
        panic!("clamp max");
    };
    assert_eq!(wide.width, 100 - 24 - 1);
}

#[test]
fn docked_hints_depend_on_focus() {
    let idle = crate::review::docked_session_hints(false);
    assert!(
        idle.iter()
            .any(|h| h.key == "tab" && h.label == "focus diff")
    );
    assert!(idle.iter().any(|h| h.key == "ctrl+g" && h.label == "close"));
    let focused = crate::review::docked_session_hints(true);
    assert!(focused.iter().any(|h| h.key == "n/p" && h.label == "hunk"));
    assert!(
        focused
            .iter()
            .any(|h| h.key == "esc" && h.label == "unfocus")
    );
    assert!(!focused.iter().any(|h| h.key == "tab"));
}
