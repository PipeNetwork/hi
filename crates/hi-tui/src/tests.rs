use super::*;
use ratatui::Terminal;
use ratatui::backend::TestBackend;

mod review_pane;
mod thinking;

pub(crate) fn dump(term: &Terminal<TestBackend>) -> String {
    let buf = term.backend().buffer();
    let mut out = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            out.push_str(buf[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

pub(crate) fn test_app(provider: &str, model: &str) -> App {
    let mut app = App::new(provider, model);
    app.workspace_root = std::path::PathBuf::from("/workspace");
    app.timestamps_enabled = false;
    app
}

#[test]
fn write_result_stays_a_filename_row() {
    let mut app = test_app("pipe", "m");
    let display = concat!(
        "\u{1b}[1m1 addition, 0 deletions\u{1b}[0m\n",
        "--- /dev/null\n+++ note.txt\n",
        "\u{1b}[32m   1 + hello\u{1b}[0m\n",
    );
    app.push_result("write", display, "write note.txt");
    let Some(TranscriptEntry::Activity(block)) = app.transcript.last() else {
        panic!("expected an Edit activity after a write result with a compact diff");
    };
    let collapsed = block.flatten(false, false, Density::Comfortable);
    assert_eq!(collapsed.len(), 1);
    assert_eq!(crate::render::line_text(&collapsed[0]), "Edit note.txt");

    let expanded = block.flatten(false, false, Density::Verbose);
    assert!(expanded.len() > 1, "verbose density still paints the diff");
    let add = crate::theme::theme().diff_add;
    assert!(
        expanded
            .iter()
            .any(|line| line.spans.iter().any(|span| span.style.fg == Some(add))),
        "expected add-colored gutter in {expanded:?}"
    );
}

#[test]
fn grok_feed_is_a_verb_list() {
    use crate::event::UiEvent;

    let mut app = test_app("pipe", "m");
    for path in ["src/a.rs", "src/b.rs", "src/c.rs"] {
        app.apply(UiEvent::ToolCall {
            name: "read".into(),
            arguments: format!(r#"{{"path":"{path}"}}"#),
        });
        app.apply(UiEvent::ToolResult {
            name: "read".into(),
            result: "ok".into(),
        });
    }
    app.apply(UiEvent::ToolCall {
        name: "grep".into(),
        arguments: r#"{"pattern":"RateLimiter"}"#.into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "grep".into(),
        result: "src/ws.rs:88:pub struct RateLimiter {\nsrc/ws.rs:95:impl RateLimiter {\n".into(),
    });
    app.apply(UiEvent::ToolCall {
        name: "write".into(),
        arguments: r#"{"path":"src/ws.rs","content":"hi"}"#.into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "write".into(),
        result: "\u{1b}[1m1 addition, 0 deletions\u{1b}[0m\n--- /dev/null\n+++ src/ws.rs\n\u{1b}[32m   1 + hi\u{1b}[0m\n".into(),
    });

    let lines: Vec<String> = app
        .transcript
        .iter()
        .flat_map(|entry| entry.flatten(false, false, Density::Comfortable))
        .map(|line| crate::render::line_text(&line))
        .collect();
    assert!(
        lines.iter().any(|l| l == "Read 3 files ›"),
        "consecutive reads coalesce: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l == "Run grep"),
        "grep is a Run row: {lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("88:pub struct RateLimiter {")),
        "short grep shows hits: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l == "Edit ws.rs"),
        "edits are a filename: {lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.contains('◆') && !l.contains("Thought")),
        "activity rows have no diamond: {lines:?}"
    );
}

#[test]
fn resume_paints_hydrated_plan_progress() {
    let mut app = test_app("pipe", "m");
    app.plan = vec![
        hi_tools::PlanStep {
            title: "Forward HISTORY pagination".into(),
            status: hi_tools::PlanStatus::Done,
        },
        hi_tools::PlanStep {
            title: "Next".into(),
            status: hi_tools::PlanStatus::Pending,
        },
        hi_tools::PlanStep {
            title: "a".into(),
            status: hi_tools::PlanStatus::Pending,
        },
        hi_tools::PlanStep {
            title: "b".into(),
            status: hi_tools::PlanStatus::Pending,
        },
        hi_tools::PlanStep {
            title: "c".into(),
            status: hi_tools::PlanStatus::Pending,
        },
        hi_tools::PlanStep {
            title: "d".into(),
            status: hi_tools::PlanStatus::Pending,
        },
        hi_tools::PlanStep {
            title: "e".into(),
            status: hi_tools::PlanStatus::Pending,
        },
        hi_tools::PlanStep {
            title: "f".into(),
            status: hi_tools::PlanStatus::Pending,
        },
    ];
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(
        screen.contains("plan · 1/8"),
        "hydrated plan pane:\n{screen}"
    );
}

#[test]
fn plan_drive_paused_paints_on_the_plan_header() {
    let mut app = test_app("pipe", "m");
    app.plan = vec![hi_tools::PlanStep {
        title: "Durable pending smoke step".into(),
        status: hi_tools::PlanStatus::Pending,
    }];
    app.plan_drive_paused = true;
    let mut term = Terminal::new(TestBackend::new(80, 16)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("paused"), "paused drive:\n{screen}");
}

#[test]
fn empty_enter_with_suggestion_fires_one_turn() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let mut app = test_app("pipe", "m");
    app.suggested_prompt = Some(hi_harness::CONTINUE_PLAN_PROMPT.into());
    let key = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(
        app.edit_key(&key).as_deref(),
        Some(hi_harness::CONTINUE_PLAN_PROMPT)
    );
}

#[test]
fn pending_card_continue_vs_dismiss() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let mut app = test_app("pipe", "m");
    app.pending_resume = Some(crate::PendingResumeCard { selected: 0 });
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("Continue"), "{screen}");
    assert!(screen.contains("Dismiss"), "{screen}");

    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert!(app.handle_pending_resume_key(&enter));
    assert!(app.resume_incomplete_requested);
    assert!(app.pending_resume.is_none());

    app.resume_incomplete_requested = false;
    app.pending_resume = Some(crate::PendingResumeCard { selected: 1 });
    assert!(app.handle_pending_resume_key(&enter));
    assert!(!app.resume_incomplete_requested);
    assert!(app.pending_resume.is_none());
}
