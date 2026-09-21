//! The read-only guard on `/review` audit turns, end to end against
//! `MockPipe`: write tools are denied with a result that says what to report
//! instead, the tree is untouched, the verdict still lands, and a turn that
//! keeps trying is stopped. Fix turns are not guarded.

use super::*;
use crate::pipe::test_support::{MockPipe, Scripted};
use crate::review_drive::{ReviewPhase, ReviewStep};
use crate::review_harness_tests::{
    P0_ROW, fix_turn_scripts, incomplete_with, run, start, test_harness, text, tool, write_inputs,
};
use std::fs;

const ORIGINAL_SERVER: &str = "pub fn nick(raw: &str) -> &str { raw }\n";

fn write_topic() -> Scripted {
    tool(
        "call_w",
        "write",
        &serde_json::json!({
            "path": "src/topic.rs",
            "content": "pub fn topic() {}\n"
        }),
    )
}

fn edit_server() -> Scripted {
    tool(
        "call_e",
        "edit",
        &serde_json::json!({
            "path": "src/server.rs",
            "old_string": "{ raw }",
            "new_string": "{ assert!(!raw.is_empty()); raw }"
        }),
    )
}

fn plan_topic() -> Scripted {
    tool(
        "call_p",
        "update_plan",
        &serde_json::json!({"steps": [
            {"title": "Add topics map to State", "status": "done"},
            {"title": "Implement TOPIC command", "status": "done"}
        ]}),
    )
}

/// Live run: the re-audit rewrote `src/main.rs` with the missing `/topic`
/// feature, added a test, closed a plan, and then reported `/topic` as
/// missing. Every one of those calls must be denied in an audit turn.
#[tokio::test]
async fn audit_turn_denies_write_tools_and_still_reports() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let Some(server) = MockPipe::new(vec![
        write_topic(),
        tool(
            "call_s",
            "bash",
            &serde_json::json!({"command": "sed -i 's/raw/nick/' src/server.rs"}),
        ),
        text(&incomplete_with(&[P0_ROW])),
        text("must not be requested"),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, _) = start(&mut harness, "audit");
    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::Completed,
        "{:?}",
        outcome.error
    );

    assert!(
        !dir.path().join("src/topic.rs").exists(),
        "the audit turn must not create files"
    );
    assert_eq!(
        fs::read_to_string(dir.path().join("src/server.rs")).unwrap(),
        ORIGINAL_SERVER,
        "the audit turn must not run mutating shell commands"
    );
    assert!(
        outcome.changed_files.is_empty(),
        "{:?}",
        outcome.changed_files
    );

    let denied: Vec<&(String, String)> = ui
        .tool_results
        .iter()
        .filter(|(_, result)| result.starts_with("[hi:review]"))
        .collect();
    assert_eq!(denied.len(), 2, "{:?}", ui.tool_results);
    assert!(
        denied[0]
            .1
            .contains("`write` was not run: this is a read-only audit turn")
            && denied[0]
                .1
                .contains("`missing` or `partial` `coverage:` row"),
        "{}",
        denied[0].1
    );
    assert!(
        denied[1]
            .1
            .contains("that shell command was not run (it modifies the tree)"),
        "{}",
        denied[1].1
    );
    let bodies = server.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 3, "denials are tool results, not turn ends");
    assert!(
        bodies[1].contains("read-only audit turn"),
        "the model sees the denial as the tool result: {}",
        bodies[1]
    );
    drop(bodies);

    match harness.review_next_step(&outcome, &mut ui) {
        ReviewStep::Done(summary) => {
            assert!(
                summary.contains("1 P0/P1 defect(s) still open"),
                "{summary}"
            );
        }
        other => panic!("the verdict still ends the audit, got {other:?}"),
    }
    assert!(harness.review_drive().audit_changed_files.is_empty());
    let report = harness.review_drive().report_lines().join("\n");
    assert!(
        !report.contains("an audit turn changed files"),
        "nothing changed, so no warning: {report}"
    );
}

/// The live sequence exactly: write the feature, edit the tests, close a
/// plan. The third denial stops the turn.
#[tokio::test]
async fn audit_turn_that_keeps_trying_to_write_is_stopped() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let Some(server) = MockPipe::new(vec![
        write_topic(),
        edit_server(),
        plan_topic(),
        text("must not be requested"),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, _) = start(&mut harness, "");
    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    assert_eq!(outcome.stop_reason, TurnStopReason::Error);
    assert!(
        ui.tool_results
            .iter()
            .any(|(_, result)| result.contains("`update_plan` was not run")),
        "{:?}",
        ui.tool_results
    );
    assert!(
        harness.current_plan().is_empty() && ui.plans.is_empty(),
        "update_plan is denied in an audit turn, got {:?} / {:?}",
        harness.current_plan(),
        ui.plans
    );
    assert_eq!(
        outcome.error.as_deref(),
        Some(crate::review_guard::AUDIT_WRITE_STORM_MSG)
    );
    assert!(
        ui.errors
            .iter()
            .any(|(kind, _)| kind == crate::review_guard::AUDIT_WRITE_STORM_KIND),
        "{:?}",
        ui.errors
    );
    assert_eq!(server.bodies.lock().unwrap().len(), 3);
    assert!(!dir.path().join("src/topic.rs").exists());
    assert_eq!(
        fs::read_to_string(dir.path().join("src/server.rs")).unwrap(),
        ORIGINAL_SERVER
    );

    match harness.review_next_step(&outcome, &mut ui) {
        ReviewStep::Stopped(text) => {
            assert!(
                text.contains("audit turn failed: audit turn kept calling write tools"),
                "{text}"
            );
        }
        other => panic!("an audit turn error stops the loop, got {other:?}"),
    }
    assert_eq!(harness.review_drive().phase, ReviewPhase::Stopped);
}

/// The guard is on audit turns only: the fix pass edits, runs the tests,
/// and closes the seeded plan exactly as before.
#[tokio::test]
async fn fix_turn_is_not_guarded() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let mut scripts = vec![text(&incomplete_with(&[P0_ROW]))];
    scripts.extend(fix_turn_scripts());
    scripts.push(text("must not be requested"));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, _) = start(&mut harness, "");
    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    let ReviewStep::RunPrompt(fix) = harness.review_next_step(&outcome, &mut ui) else {
        panic!("a P0 finding starts a fix pass");
    };
    let outcome = run(&mut harness, &fix, &mut ui).await;
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::Completed,
        "{:?}",
        outcome.error
    );
    assert!(
        ui.tool_results
            .iter()
            .all(|(_, result)| !result.starts_with("[hi:review]")),
        "{:?}",
        ui.tool_results
    );
    assert!(
        fs::read_to_string(dir.path().join("src/server.rs"))
            .unwrap()
            .contains("assert!(!raw.is_empty())")
    );
    assert!(hi_tools::PlanStep::all_complete(harness.current_plan()));
    assert_eq!(server.bodies.lock().unwrap().len(), 6);
}
