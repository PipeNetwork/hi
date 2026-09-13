//! Grok-build thinking fold: collapsed is header-only; Ctrl-E expands the body.

use super::*;
use std::time::Duration;

use crate::Density;
use crate::event::UiEvent;

#[test]
fn collapsed_thinking_is_header_only() {
    let entry = TranscriptEntry::Reasoning {
        text: "Check spacing and preserve the active input.".into(),
        elapsed: Duration::from_secs(3),
        expanded: false,
    };
    let collapsed = entry.flatten(false, false, Density::Comfortable);
    let collapsed_text: Vec<String> = collapsed.iter().map(crate::render::line_text).collect();
    assert_eq!(
        collapsed.len(),
        1,
        "collapsed thinking is header-only: {collapsed_text:?}"
    );
    assert!(
        collapsed_text[0].contains("Thought for"),
        "grok collapsed header: {collapsed_text:?}"
    );
    assert!(
        collapsed_text[0].contains('›'),
        "collapsed thought shows grok's expand chevron: {collapsed_text:?}"
    );
    assert!(
        !collapsed_text.iter().any(|l| l.contains("Check spacing")),
        "body must not leak while collapsed: {collapsed_text:?}"
    );

    let expanded = entry.flatten(true, false, Density::Comfortable);
    let expanded_text: Vec<String> = expanded.iter().map(crate::render::line_text).collect();
    assert!(
        expanded_text.iter().any(|l| l.contains("Check spacing")),
        "Ctrl-E shows the body: {expanded_text:?}"
    );
}

#[test]
fn collapsed_zero_second_thought_is_header_only() {
    let entry = TranscriptEntry::Reasoning {
        text: "instant".into(),
        elapsed: Duration::from_secs(0),
        expanded: false,
    };
    let collapsed = entry.flatten(false, false, Density::Comfortable);
    assert_eq!(collapsed.len(), 1, "instant thought still has a header");
    assert!(
        crate::render::line_text(&collapsed[0]).contains("Thought"),
        "{}",
        crate::render::line_text(&collapsed[0])
    );
    assert!(
        !collapsed
            .iter()
            .any(|l| crate::render::line_text(l).contains("instant")),
        "body stays folded"
    );
    let expanded = entry.flatten(true, false, Density::Comfortable);
    assert!(
        expanded
            .iter()
            .any(|l| crate::render::line_text(l).contains("instant")),
        "Ctrl-T still reveals 0s thought"
    );
}

#[test]
fn live_thinking_shows_header_until_ctrl_e() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Reasoning {
        text: "I will inspect the parser next.".into(),
    });
    assert!(
        app.transcript
            .iter()
            .all(|e| !matches!(e, TranscriptEntry::Reasoning { .. })),
        "live chunks stay in the buffer until flush"
    );
    let live_text: Vec<String> = app
        .live_thinking_lines()
        .iter()
        .map(crate::render::line_text)
        .collect();
    assert!(
        live_text.iter().any(|l| l.contains("Thinking")),
        "running thought uses grok Thinking... header: {live_text:?}"
    );
    assert!(
        live_text.iter().any(|l| l.contains("inspect the parser")),
        "live truncated tail shows the current thought: {live_text:?}"
    );

    app.apply(UiEvent::AssistantEnd);
    let finished = app.transcript.iter().find_map(|entry| match entry {
        TranscriptEntry::Reasoning { .. } => Some(
            entry
                .flatten(false, false, Density::Comfortable)
                .iter()
                .map(crate::render::line_text)
                .collect::<Vec<_>>(),
        ),
        _ => None,
    });
    let finished = finished.expect("finished thought");
    assert_eq!(finished.len(), 1, "finish auto-collapses: {finished:?}");
    assert!(
        !finished.iter().any(|l| l.contains("inspect the parser")),
        "finished thought is header-only: {finished:?}"
    );
}

#[test]
fn live_thinking_truncates_to_the_last_three_lines() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Reasoning {
        text: "one\ntwo\nthree\nfour\nfive\n".into(),
    });
    let live: Vec<String> = app
        .live_thinking_lines()
        .iter()
        .map(crate::render::line_text)
        .collect();
    assert!(live.iter().any(|l| l.contains("Thinking")), "{live:?}");
    assert!(live.iter().any(|l| l.contains("…")), "ellipsis: {live:?}");
    assert!(live.iter().any(|l| l.contains("five")), "{live:?}");
    assert!(
        !live.iter().any(|l| l.contains("one")),
        "early lines stay off-screen while running: {live:?}"
    );
    app.show_reasoning = true;
    let expanded: Vec<String> = app
        .live_thinking_lines()
        .iter()
        .map(crate::render::line_text)
        .collect();
    assert!(
        expanded.iter().any(|l| l.contains("one")),
        "Ctrl-E shows the full live thought: {expanded:?}"
    );
}

#[test]
fn expanded_thinking_renders_markdown_and_a_rail() {
    let entry = TranscriptEntry::Reasoning {
        text: "## Why\n\nUse `path:line` citations.".into(),
        elapsed: Duration::from_secs(2),
        expanded: false,
    };
    let expanded: Vec<String> = entry
        .flatten(true, false, Density::Comfortable)
        .iter()
        .map(crate::render::line_text)
        .collect();
    assert!(
        expanded
            .iter()
            .any(|l| l.contains("◆") && l.contains("Thought")),
        "header keeps the grok diamond: {expanded:?}"
    );
    assert!(
        expanded.iter().any(|l| l.contains("Why")),
        "markdown heading survives: {expanded:?}"
    );
    assert!(
        expanded.iter().any(|l| l.contains("path:line")),
        "markdown body survives: {expanded:?}"
    );
    assert!(
        expanded.iter().any(|l| l.contains("┃")),
        "open thinking uses the accent rail: {expanded:?}"
    );
    let collapsed: Vec<String> = entry
        .flatten(false, false, Density::Comfortable)
        .iter()
        .map(crate::render::line_text)
        .collect();
    assert!(
        !collapsed
            .iter()
            .any(|l| l.contains("Why") || l.contains("┃")),
        "collapsed stays header-only: {collapsed:?}"
    );
}

#[test]
fn click_or_block_nav_expands_one_thought() {
    let mut app = test_app("openai", "gpt-4o");
    app.transcript.push(TranscriptEntry::Reasoning {
        text: "secret plan".into(),
        elapsed: Duration::from_secs(4),
        expanded: false,
    });
    assert!(app.transcript[0].is_foldable());
    assert_eq!(app.tool_block_count(), 1);
    app.toggle_block_ord(0);
    let open: Vec<String> = app.transcript[0]
        .flatten(false, false, Density::Comfortable)
        .iter()
        .map(crate::render::line_text)
        .collect();
    assert!(
        open.iter().any(|l| l.contains("secret plan")),
        "one thought opens without Ctrl-E: {open:?}"
    );

    app.apply_action(crate::action::Action::ToggleReasoning);
    app.apply_action(crate::action::Action::ToggleReasoning);
    let closed: Vec<String> = app.transcript[0]
        .flatten(false, false, Density::Comfortable)
        .iter()
        .map(crate::render::line_text)
        .collect();
    assert_eq!(
        closed.len(),
        1,
        "Ctrl-E off collapses per-block thoughts: {closed:?}"
    );
}

#[test]
fn paste_into_prompt_keeps_newlines() {
    let mut app = test_app("openai", "gpt-4o");
    app.paste_into_prompt("line one\nline two");
    assert_eq!(app.input.text(), "line one\nline two");
    let paste = crossterm::event::KeyEvent::new(
        crossterm::event::KeyCode::Char('v'),
        crossterm::event::KeyModifiers::CONTROL,
    );
    assert!(app.edit_key(&paste).is_none(), "Ctrl+V must not submit");
}

#[test]
fn ctrl_e_expands_thinking_even_with_a_draft() {
    let mut app = test_app("openai", "gpt-4o");
    app.input.set("follow-up");
    let key = crossterm::event::KeyEvent::new(
        crossterm::event::KeyCode::Char('e'),
        crossterm::event::KeyModifiers::CONTROL,
    );
    assert!(app.edit_key(&key).is_none());
    assert!(
        app.show_reasoning,
        "Ctrl+E toggles thinking while the prompt has text"
    );
    assert_eq!(app.input.text(), "follow-up");
    assert!(app.edit_key(&key).is_none());
    assert!(!app.show_reasoning);
    assert_eq!(app.input.text(), "follow-up");
}

#[test]
fn ctrl_e_enq_byte_and_block_nav_still_toggle() {
    let mut app = test_app("openai", "gpt-4o");
    app.transcript.push(TranscriptEntry::Reasoning {
        text: "secret plan".into(),
        elapsed: Duration::from_secs(2),
        expanded: false,
    });
    app.mode = crate::mode::UiMode::BlockNav;
    let enq = crossterm::event::KeyEvent::new(
        crossterm::event::KeyCode::Char('\x05'),
        crossterm::event::KeyModifiers::NONE,
    );
    assert!(app.edit_key(&enq).is_none());
    assert!(
        app.show_reasoning,
        "ASCII ENQ (raw Ctrl+E) must expand thinking even in block-nav"
    );
    let capital = crossterm::event::KeyEvent::new(
        crossterm::event::KeyCode::Char('E'),
        crossterm::event::KeyModifiers::CONTROL,
    );
    assert!(app.edit_key(&capital).is_none());
    assert!(!app.show_reasoning);
}

#[test]
fn finished_thought_stays_visible_when_explore_tools_start() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::Reasoning {
        text: "I will inspect the parser next.".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: r#"{"path":"src/lib.rs"}"#.into(),
    });
    let lines: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|entry| entry.flatten(false, false, Density::Comfortable))
        .map(|line| crate::render::line_text(&line))
        .collect();
    assert!(
        lines
            .iter()
            .any(|l| l.contains("Thought for") || l.contains("Thought")),
        "grok keeps the collapsed thought header after tools start: {lines:?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains("inspect the parser")),
        "body stays folded until Ctrl+E: {lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("Reading") || l.contains("Read")),
        "explore tools still group: {lines:?}"
    );

    app.show_reasoning = true;
    let expanded: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|entry| entry.flatten(true, false, Density::Comfortable))
        .map(|line| crate::render::line_text(&line))
        .collect();
    assert!(
        expanded.iter().any(|l| l.contains("inspect the parser")),
        "Ctrl+E reveals the thought folded into the explore row: {expanded:?}"
    );
}

#[test]
fn live_thinking_after_a_tool_stays_a_thinking_row() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "read".into(),
        arguments: r#"{"path":"src/lib.rs"}"#.into(),
    });
    app.apply(UiEvent::Reasoning {
        text: "next I will check spacing".into(),
    });
    let live: Vec<String> = app
        .live_thinking_lines()
        .iter()
        .map(crate::render::line_text)
        .collect();
    assert!(
        live.iter().any(|l| l.contains("Thinking")),
        "in-flight thought stays a grok Thinking... row: {live:?}"
    );
    assert!(
        live.iter().any(|l| l.contains("check spacing")),
        "live tail shows the current thought: {live:?}"
    );
    let grouped: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|entry| entry.flatten(false, false, Density::Comfortable))
        .map(|line| crate::render::line_text(&line))
        .collect();
    assert!(
        !grouped.iter().any(|l| l.contains("check spacing")),
        "live thought is not swallowed into the tool row: {grouped:?}"
    );
}
