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
fn write_result_is_a_claude_style_edit_row() {
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
    // Comfortable paints the header and the hunk inline, like Claude Code.
    let comfortable = block.flatten(false, false, Density::Comfortable);
    assert_eq!(
        crate::render::line_text(&comfortable[0]),
        "Created note.txt (+1 -0)"
    );
    assert_eq!(comfortable.len(), 2, "{comfortable:?}");
    assert!(crate::render::line_text(&comfortable[1]).contains("hello"));
    let add = crate::theme::theme().diff_add;
    assert!(
        comfortable[1]
            .spans
            .iter()
            .any(|span| span.style.fg == Some(add)),
        "expected add-colored gutter in {comfortable:?}"
    );

    // Compact keeps the verb list terse: header only.
    let compact = block.flatten(false, false, Density::Compact);
    assert_eq!(compact.len(), 1, "{compact:?}");
    assert_eq!(
        crate::render::line_text(&compact[0]),
        "Created note.txt (+1 -0)"
    );

    let verbose = block.flatten(false, false, Density::Verbose);
    assert!(verbose.len() > 1, "verbose density still paints the diff");
    assert_eq!(
        app.session_changed_files,
        vec!["note.txt".to_string()],
        "the edit joins the session's running change set"
    );
}

#[test]
fn edit_paths_are_workspace_relative_and_apply_patch_names_every_file() {
    let mut app = test_app("pipe", "m");
    app.push_result(
        "apply_patch",
        "--- /workspace/src/a.rs\n+++ /workspace/src/a.rs\n1 addition, 0 deletions\n   1 + a\n--- /dev/null\n+++ /workspace/src/b.rs\n1 addition, 0 deletions\n   1 + b\n",
        "apply_patch",
    );
    let Some(TranscriptEntry::Activity(block)) = app.transcript.last() else {
        panic!("expected an Edit activity");
    };
    let header = crate::render::line_text(&block.flatten(false, false, Density::Compact)[0]);
    assert_eq!(header, "Updated src/a.rs, src/b.rs (+2 -0)");
    assert_eq!(
        app.session_changed_files,
        vec!["src/a.rs".to_string(), "src/b.rs".to_string()]
    );
}

/// Live ~/drme: every failed or refused edit of `src/decoder/mod.rs` read
/// `Updated src/decoder/mod.rs (+0 -0)`, as if it had landed.
#[test]
fn failed_edits_say_so_instead_of_updated_plus_zero() {
    use crate::event::UiEvent;

    let mut app = test_app("pipe", "m");
    for (error, reason) in [
        (
            "Error: editing src/decoder/mod.rs: old_string not found, even allowing for \
             whitespace differences. (Do not include the line-number gutter.) The closest \
             matching lines in the file are:\n  1030\t    }\n",
            "old_string not found, even allowing for whitespace differences",
        ),
        (
            "Error: no change: your new_string is already in this file. It matches the text \
             it would replace except for whitespace or line endings.",
            "no change: your new_string is already in this file",
        ),
    ] {
        app.apply(UiEvent::ToolCall {
            name: "edit".into(),
            arguments:
                r#"{"path":"/workspace/src/decoder/mod.rs","old_string":"a","new_string":"b"}"#
                    .into(),
        });
        app.apply(UiEvent::ToolResult {
            name: "edit".into(),
            result: error.into(),
            failed: true,
        });
        let Some(TranscriptEntry::Activity(block)) = app.transcript.last() else {
            panic!("expected an activity row for the failed edit");
        };
        let header = &block.flatten(false, false, Density::Compact)[0];
        assert_eq!(
            crate::render::line_text(header),
            format!("Edit failed src/decoder/mod.rs · {reason} ›")
        );
        assert_eq!(
            header.spans[0].style.fg,
            Some(crate::theme::theme().accent_error),
            "the failure is coloured as one: {header:?}"
        );
        let expanded: Vec<String> = block
            .flatten(false, false, Density::Verbose)
            .iter()
            .map(crate::render::line_text)
            .collect();
        assert!(
            expanded
                .iter()
                .any(|line| line.contains(error.lines().next().unwrap())),
            "the full error is one fold away: {expanded:?}"
        );
    }
    assert!(
        app.session_changed_files.is_empty(),
        "a failed edit changed nothing: {:?}",
        app.session_changed_files
    );
}

#[test]
fn edit_rows_without_line_counts_do_not_claim_plus_zero() {
    let mut app = test_app("pipe", "m");
    app.push_result(
        "write",
        "~ assets/logo.png (binary content changed)",
        "write assets/logo.png",
    );
    let Some(TranscriptEntry::Activity(block)) = app.transcript.last() else {
        panic!("expected an Edit activity");
    };
    let header = crate::render::line_text(&block.flatten(false, false, Density::Compact)[0]);
    assert_eq!(header, "Updated assets/logo.png");
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
            failed: false,
        });
    }
    app.apply(UiEvent::ToolCall {
        name: "grep".into(),
        arguments: r#"{"pattern":"RateLimiter"}"#.into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "grep".into(),
        result: "src/ws.rs:88:pub struct RateLimiter {\nsrc/ws.rs:95:impl RateLimiter {\n".into(),
        failed: false,
    });
    app.apply(UiEvent::ToolCall {
        name: "write".into(),
        arguments: r#"{"path":"src/ws.rs","content":"hi"}"#.into(),
    });
    app.apply(UiEvent::ToolResult {
        name: "write".into(),
        result: "\u{1b}[1m1 addition, 0 deletions\u{1b}[0m\n--- /dev/null\n+++ src/ws.rs\n\u{1b}[32m   1 + hi\u{1b}[0m\n".into(),
        failed: false,
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
        lines.iter().any(|l| l == "Created src/ws.rs (+1 -0)"),
        "edits are Claude-style `Verb path (+N -M)` rows: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l.trim_start().starts_with("1  hi")),
        "the hunk paints inline under the edit header: {lines:?}"
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
fn plan_drive_header_drops_paused_once_the_drive_resumes() {
    let mut app = test_app("pipe", "m");
    app.plan = vec![hi_tools::PlanStep {
        title: "Durable pending smoke step".into(),
        status: hi_tools::PlanStatus::Pending,
    }];
    app.plan_drive_paused = true;
    app.working = true;
    app.plan_drive_paused = false;
    let mut term = Terminal::new(TestBackend::new(80, 16)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("plan · 0/1"), "resumed drive:\n{screen}");
    assert!(
        !screen.contains("paused"),
        "continue must clear the paused chip:\n{screen}"
    );
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

fn view_texts(cache: &crate::view_cache::TranscriptViewCache) -> Vec<String> {
    cache.lines.iter().map(crate::render::line_text).collect()
}

/// Update the view cache the way a frame does (incrementally when it can),
/// then check it against a rebuild from scratch.
fn assert_incremental_view_is_a_rebuild(app: &mut App, step: &str) {
    app.ensure_view_cache(40, None);
    let incremental = app.view_cache.clone();
    app.view_cache = Default::default();
    app.ensure_view_cache(40, None);
    let rebuilt = &app.view_cache;
    assert_eq!(
        view_texts(&incremental),
        view_texts(rebuilt),
        "lines after {step}"
    );
    assert_eq!(incremental.joins, rebuilt.joins, "joins after {step}");
    assert_eq!(incremental.prefix, rebuilt.prefix, "rows after {step}");
    assert_eq!(
        incremental.prompt_line_starts, rebuilt.prompt_line_starts,
        "prompts after {step}"
    );
    assert_eq!(
        incremental.block_line_ranges, rebuilt.block_line_ranges,
        "blocks after {step}"
    );
    assert_eq!(incremental.last_entry_start, rebuilt.last_entry_start);
}

#[test]
fn the_incremental_view_keeps_lines_streamed_into_the_open_message() {
    let mut app = test_app("pipe", "m");
    app.push_user_prompt(Line::raw("explain the parser"));
    assert_incremental_view_is_a_rebuild(&mut app, "the prompt");
    app.stream(
        Style::default(),
        true,
        "The parser reads the whole input before it builds a tree.\n",
    );
    assert_incremental_view_is_a_rebuild(&mut app, "a first line");
    app.stream(
        Style::default(),
        true,
        "It drops the last token at end of input.\nA fix",
    );
    app.ensure_view_cache(40, None);
    assert!(
        view_texts(&app.view_cache)
            .iter()
            .any(|line| line == "It drops the last token at end of input."),
        "{:?}",
        view_texts(&app.view_cache)
    );
    assert_incremental_view_is_a_rebuild(&mut app, "a line into the open message");
    app.stream(Style::default(), true, " is to flush the buffer.\n");
    assert_incremental_view_is_a_rebuild(&mut app, "the partial line committed");
    app.flush_pending();
    app.push_result("read", "fn main() {}\n", "read src/main.rs");
    assert_incremental_view_is_a_rebuild(&mut app, "a tool result");
    app.push_user_prompt(Line::raw("and the lexer?"));
    app.stream(Style::default(), true, "The lexer is fine.\nIt");
    assert_incremental_view_is_a_rebuild(&mut app, "a second turn");
}
