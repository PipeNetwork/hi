use super::*;
use hi_agent::ConfirmationRequest;
use ratatui::style::Color;

#[test]
fn compact_edit_preview_with_file_headers_stays_colored() {
    let mut app = test_app("openai", "gpt-4o");
    app.apply(UiEvent::ToolCall {
        name: "edit".into(),
        arguments: "{\"path\":\"src/cli.rs\"}".into(),
    });
    // Live `edit` display: unified file headers around hi's numbered compact
    // diff. The headers must not steal coloring from the `+`/`-` rows.
    app.apply(UiEvent::ToolResult {
        name: "edit".into(),
        result: "--- src/cli.rs\n+++ src/cli.rs\n\u{1b}[1m1 addition, 1 deletion\u{1b}[0m\n   1 - old line\n   2 + new line\n"
            .into(),
    });
    let collapsed_styles: Vec<(String, Option<Color>)> = app
        .transcript
        .iter()
        .flat_map(|e| e.flatten(false, false, crate::Density::Comfortable))
        .flat_map(|line| {
            line.spans
                .into_iter()
                .map(|span| (span.content.to_string(), span.style.fg))
        })
        .collect();
    assert!(
        collapsed_styles.iter().any(|(t, _)| t.contains("new line")),
        "added compact line is shown: {collapsed_styles:?}"
    );
    assert!(
        collapsed_styles.iter().any(|(t, _)| t.contains("old line")),
        "removed compact line is shown: {collapsed_styles:?}"
    );
    assert!(
        collapsed_styles
            .iter()
            .any(|(_, fg)| *fg == Some(crate::theme::theme().diff_add)),
        "additions color the gutter: {collapsed_styles:?}"
    );
    assert!(
        collapsed_styles
            .iter()
            .any(|(_, fg)| *fg == Some(crate::theme::theme().diff_del)),
        "deletions color the gutter: {collapsed_styles:?}"
    );
}

#[test]
fn confirmation_modal_colors_compact_edit_preview() {
    let mut app = test_app("openai", "gpt-4o");
    app.confirmation = Some(ConfirmationRequest::FileEdit {
        path: "src/main.rs".to_string(),
        diff: "--- src/main.rs\n+++ src/main.rs\n   1 - old\n   2 + new\n".to_string(),
    });
    let mut term = Terminal::new(TestBackend::new(80, 32)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("Confirm file edit"),
        "modal title shown: {screen}"
    );
    assert!(screen.contains("new"), "added compact line shown: {screen}");
    assert!(
        screen.contains("old"),
        "removed compact line shown: {screen}"
    );
}
