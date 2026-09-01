use super::*;
use crate::input::HistorySearch;
use crate::layout::display_width;
use ratatui::backend::TestBackend;
use std::time::Instant;

fn assert_composer_closed_with_prompt(screen: &str) {
    let rows: Vec<&str> = screen.lines().collect();
    let border_idx = rows
        .iter()
        .rposition(|line| line.trim_start().starts_with('╰'))
        .expect("composer bottom border");
    let border = rows[border_idx];
    assert!(
        border.trim_start().starts_with('╰'),
        "bottom border must close on its own row: {border:?}\n{screen}"
    );
    assert!(
        !border.contains('❯'),
        "prompt must not paint on the bottom border:\n{screen}"
    );
    let above = rows[..border_idx].join("\n");
    assert!(above.contains('❯'), "prompt above the border:\n{screen}");
    let prompt_idx = rows[..border_idx]
        .iter()
        .rposition(|line| line.contains('❯'))
        .unwrap();
    let prompt = rows[prompt_idx];
    assert!(
        prompt.trim_end().ends_with('│'),
        "prompt row must keep its right border: {prompt:?}\n{screen}"
    );
}

#[test]
fn long_ghost_does_not_fill_the_inner_width() {
    let mut app = test_app("openai", "gpt-4o");
    app.suggested_prompt = Some("x".repeat(80));
    let (lines, ..) = app.input_view(20);
    let text = lines[0].to_string();
    assert!(
        display_width(&text) < 20,
        "ghost must leave a gutter, got width {} for {text:?}",
        display_width(&text)
    );
    assert!(
        text.contains('…'),
        "truncated ghost should ellipsize: {text}"
    );
}

#[test]
fn long_suggested_prompt_keeps_composer_border_closed() {
    for (width, height) in [(80, 16), (48, 12), (40, 10)] {
        let mut app = test_app("openai", "gpt-4o");
        app.suggested_prompt =
            Some("Review uncommitted changes in crates/hi-tui/src/app/render.rs and more".into());
        let mut term = Terminal::new(TestBackend::new(width, height)).unwrap();
        term.draw(|frame| app.render(frame)).unwrap();
        let screen = dump(&term);
        assert_composer_closed_with_prompt(&screen);
        assert!(
            screen.contains("Review uncommitted") || screen.contains('…'),
            "ghost visible at {width}x{height}:\n{screen}"
        );
    }
}

#[test]
fn copy_toast_with_ghost_keeps_prompt_inside_the_box() {
    let mut app = test_app("openai", "gpt-4o");
    app.suggested_prompt = Some("Run the unit tests".into());
    app.copy_toast = Some((12, Instant::now()));
    let mut term = Terminal::new(TestBackend::new(60, 12)).unwrap();
    term.draw(|frame| app.render(frame)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("copied 12 chars"),
        "copy toast visible:\n{screen}"
    );
    assert_composer_closed_with_prompt(&screen);
    let rows: Vec<&str> = screen.lines().collect();
    let prompt_idx = rows.iter().rposition(|line| line.contains('❯')).unwrap();
    let toast_idx = rows
        .iter()
        .position(|line| line.contains("copied 12 chars"))
        .unwrap();
    assert!(
        toast_idx < prompt_idx,
        "toast should sit above the prompt:\n{screen}"
    );
}

#[test]
fn history_search_with_ghost_keeps_prompt_inside_the_box() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.history = vec!["hello".into()];
    let mut search = HistorySearch::default();
    search.refilter(&app.input.history);
    app.mode = crate::mode::UiMode::HistorySearch(search);
    app.suggested_prompt = Some("hello world this is a long follow-up".into());
    let mut term = Terminal::new(TestBackend::new(60, 12)).unwrap();
    term.draw(|frame| app.render(frame)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("reverse-i-search"),
        "history search overlay:\n{screen}"
    );
    assert_composer_closed_with_prompt(&screen);
}

#[test]
fn voice_indicator_with_ghost_keeps_prompt_inside_the_box() {
    let mut app = test_app("openai", "gpt-4o");
    let (_tx, rx) = tokio::sync::oneshot::channel();
    app.voice = crate::app::voice::VoiceState::Transcribing { rx };
    app.suggested_prompt = Some("Run the unit tests".into());
    let mut term = Terminal::new(TestBackend::new(60, 12)).unwrap();
    term.draw(|frame| app.render(frame)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("transcribing"),
        "voice indicator visible:\n{screen}"
    );
    assert_composer_closed_with_prompt(&screen);
}

#[test]
fn debug_panel_with_ghost_keeps_prompt_inside_the_box() {
    let mut app = test_app("openai", "gpt-4o");
    app.show_debug = true;
    app.last_turn_phase = Some("review");
    app.last_telemetry = Some(hi_agent::TurnTelemetry {
        tool_calls: 4,
        file_reads: 2,
        targeted_searches: 1,
        quality_repair_nudges: 1,
        ..hi_agent::TurnTelemetry::default()
    });
    app.suggested_prompt = Some("Run the unit tests".into());
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|frame| app.render(frame)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("agent (Ctrl-? to close)"),
        "debug panel visible:\n{screen}"
    );
    assert_composer_closed_with_prompt(&screen);
}

#[test]
fn changed_files_and_long_ghost_stay_above_the_box() {
    let mut app = test_app("openai", "gpt-4o");
    app.last_changed_files = vec![
        "crates/hi-tui/src/app/render.rs".into(),
        "crates/hi-tui/src/app/composer.rs".into(),
        "crates/hi-agent/src/agent/turn/loop_.rs".into(),
    ];
    app.suggested_prompt =
        Some("Review the remaining render overflow and add coverage for overlay rows".into());
    let mut term = Terminal::new(TestBackend::new(48, 12)).unwrap();
    term.draw(|frame| app.render(frame)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("changed:"), "changed-files line:\n{screen}");
    assert_composer_closed_with_prompt(&screen);

    let rows: Vec<&str> = screen.lines().collect();
    let bottom_idx = rows
        .iter()
        .rposition(|line| line.trim_start().starts_with('╰'))
        .expect("composer bottom border");
    let top_idx = rows[..=bottom_idx]
        .iter()
        .rposition(|line| line.trim_start().starts_with('╭'))
        .expect("composer top border");
    let changed_idx = rows
        .iter()
        .position(|line| line.contains("changed:"))
        .expect("changed-files row");
    assert_eq!(
        changed_idx + 1,
        top_idx,
        "changed-files row should sit directly above the prompt box:\n{screen}"
    );
}
