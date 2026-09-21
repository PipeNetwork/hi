//! `/review` end to end against `MockPipe`: audit verdicts, the format
//! re-ask, the fix loop's plan seeding and test demand, the stop conditions,
//! cancel/resume, and input discovery fallbacks.

use super::*;
use crate::pipe::test_support::{MockPipe, Scripted, text_chunk, tool_chunk, usage_chunk};
use crate::review::REVIEW_FORMAT_HINT;
use crate::review_drive::{ReviewCommand, ReviewPhase, ReviewStep};
use hi_tools::ProcessRunner;
use hi_tools::sandbox::SandboxPolicy;
use std::fs;
use std::path::Path;

pub(crate) fn test_harness(url: &str, workspace: PathBuf) -> Harness {
    let state = workspace.join(".hi");
    let runner = ProcessRunner::new_with_policy(&workspace, SandboxPolicy::Off).expect("runner");
    let tools = ToolHost::new_with_runner(workspace.clone(), state.clone(), runner).unwrap();
    let mut config = HarnessConfig::pipe(workspace, "pk_test");
    config.base_url = url.to_string();
    config.state_root = state;
    let mut harness = Harness::new_with_tools(config, tools).unwrap();
    harness.set_permission_mode(PermissionMode::Always);
    harness
}

pub(crate) fn write_inputs(root: &Path) {
    fs::create_dir_all(root.join("docs")).unwrap();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("plan.md"),
        "# Plan\n\n- [x] Welcome banner\n- [ ] KICK command\n- [ ] /topic command\n",
    )
    .unwrap();
    fs::write(
        root.join("docs/SPEC.md"),
        "# Spec\n\nWelcome on connect. KICK removes a user. /topic sets the topic.\n",
    )
    .unwrap();
    fs::write(
        root.join("src/server.rs"),
        "pub fn nick(raw: &str) -> &str { raw }\n",
    )
    .unwrap();
}

pub(crate) fn text(reply: &str) -> Scripted {
    Scripted::Sse(vec![text_chunk(reply), usage_chunk(8, 4)])
}

pub(crate) fn tool(id: &str, name: &str, args: &serde_json::Value) -> Scripted {
    Scripted::Sse(vec![
        tool_chunk(0, id, name, &args.to_string()),
        usage_chunk(8, 4),
    ])
}

fn verdict_block(rows: &[&str]) -> String {
    format!(
        "Coverage table for the user.\n\n<review>\n{}\n</review>",
        rows.join("\n")
    )
}

const COVERAGE_ROWS: [&str; 3] = [
    "coverage: implemented | Welcome banner | src/server.rs:1",
    "coverage: partial | KICK command | src/server.rs:1",
    "coverage: missing | /topic command | -",
];

pub(crate) fn incomplete_with(findings: &[&str]) -> String {
    let mut rows = vec!["verdict: INCOMPLETE"];
    rows.extend(COVERAGE_ROWS);
    if findings.is_empty() {
        rows.push("finding: none");
    } else {
        rows.extend(findings);
    }
    rows.push("residual: /topic is a missing feature, not a defect");
    verdict_block(&rows)
}

pub(crate) const P0_ROW: &str = "finding: P0 | Reject empty nicknames | src/server.rs:1";

/// Fix-turn script: edit, premature stop, tests, close the plan, final text.
pub(crate) fn fix_turn_scripts() -> Vec<Scripted> {
    vec![
        tool(
            "call_e",
            "edit",
            &serde_json::json!({
                "path": "src/server.rs",
                "old_string": "{ raw }",
                "new_string": "{ assert!(!raw.is_empty()); raw }"
            }),
        ),
        text("Patched nick()."),
        tool(
            "call_t",
            "bash",
            &serde_json::json!({"command": "cargo test --offline --quiet"}),
        ),
        tool(
            "call_pd",
            "update_plan",
            &serde_json::json!({"steps": [
                {"title": "[P0] Reject empty nicknames — src/server.rs:1", "status": "done"}
            ]}),
        ),
        text("Fixed the P0 and ran the tests."),
    ]
}

pub(crate) async fn run(harness: &mut Harness, prompt: &str, ui: &mut TestUi) -> TurnOutcome {
    harness
        .run_turn_cancellable(prompt, ui, TurnCancellation::new())
        .await
        .unwrap()
}

pub(crate) fn start(harness: &mut Harness, arg: &str) -> (String, Vec<String>) {
    match harness.review_command(arg) {
        ReviewCommand::Start { prompt, notes } => (prompt, notes),
        ReviewCommand::Message(text) => panic!("expected the review to start, got: {text}"),
    }
}

#[tokio::test]
async fn review_audit_reads_inputs_and_parses_verdict() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let Some(server) = MockPipe::new(vec![
        tool("call_r", "read", &serde_json::json!({"path": "plan.md"})),
        text(&incomplete_with(&[
            P0_ROW,
            "finding: P2 | Log unknown commands | src/server.rs:1",
        ])),
        text("must not be requested"),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    assert!(harness.review_status_text().contains("not running"));

    let (prompt, notes) = start(&mut harness, "audit");
    assert_eq!(
        notes,
        vec!["spec review · audit only · plan.md + docs/SPEC.md".to_string()],
        "case-insensitive SPEC.md discovery under docs/"
    );
    assert!(prompt.starts_with("[hi:review]"));
    assert!(
        prompt.contains(
            "plan: plan.md (3 checklist rows: 1 checked, claims to verify; 2 unchecked, known gaps"
        ),
        "{prompt}"
    );
    assert!(prompt.contains("spec: docs/SPEC.md"));
    assert!(prompt.contains("<review>"));

    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    let shown = ui.texts.join("");
    assert!(
        shown.contains("Coverage table for the user.") && !shown.contains("<review>"),
        "the prose streams, the verdict block is rendered once as the report, got {shown:?}"
    );
    assert!(
        ui.statuses
            .iter()
            .all(|status| !status.contains("asking for an edit")
                && !status.contains("asking for tests")),
        "the audit prompt mentions fixing but must stay read-only, got {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.starts_with("spec review · audit only")),
        "turn start announces the review phase, got {:?}",
        ui.statuses
    );
    assert!(
        server.bodies.lock().unwrap()[0].contains("[hi:review] Spec-coverage audit"),
        "the audit prompt reaches the model"
    );

    let step = harness.review_next_step(&outcome, &mut ui);
    match step {
        ReviewStep::Done(summary) => {
            assert!(
                summary.contains("1/3 plan/spec items implemented"),
                "{summary}"
            );
            assert!(
                summary.contains("1 P0/P1 defect(s) still open (audit only)"),
                "{summary}"
            );
            assert!(summary.contains("1 P2/P3 reported"), "{summary}");
        }
        other => panic!("audit-only ends after the verdict, got {other:?}"),
    }
    let drive = harness.review_drive();
    assert_eq!(drive.phase, ReviewPhase::Done);
    let verdict = drive.last_verdict.as_ref().expect("verdict stored");
    assert_eq!(verdict.coverage.len(), 3);
    assert_eq!(verdict.findings.len(), 2);
    assert!(!verdict.complete);
    let status = harness.review_status_text();
    assert!(status.contains("coverage (incomplete):"), "{status}");
    assert!(
        status.contains("next: `/review` (without `audit`) fixes the 1 open P0/P1 finding(s)"),
        "{status}"
    );
    assert!(status.contains("missing     /topic command"), "{status}");
    assert!(
        status.contains("[P0] Reject empty nicknames — src/server.rs:1"),
        "{status}"
    );
    assert!(
        status.contains("residual: /topic is a missing feature"),
        "{status}"
    );
    assert!(
        harness.current_plan().is_empty(),
        "audit-only never seeds a plan"
    );
    assert_eq!(server.bodies.lock().unwrap().len(), 2);
}

/// Live run: the audit reply said "Let me also check…" before its verdict
/// block, the promised-work nudge fired, and the model then edited files
/// mid-audit. Review-pinned turns must take the verdict as the answer.
#[tokio::test]
async fn review_audit_promise_wording_does_not_trigger_the_edit_nudge() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let reply = format!(
        "Let me also check: the reply closure never drains onto tx. I'll fix that next.\n\n{}",
        incomplete_with(&[P0_ROW])
    );
    let Some(server) = MockPipe::new(vec![text(&reply), text("must not be requested")]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, _) = start(&mut harness, "audit");
    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        ui.statuses
            .iter()
            .all(|status| !status.contains("described a fix") && !status.contains("claimed fixes")),
        "{:?}",
        ui.statuses
    );
    assert_eq!(
        server.bodies.lock().unwrap().len(),
        1,
        "the verdict ends the audit"
    );
    assert!(matches!(
        harness.review_next_step(&outcome, &mut ui),
        ReviewStep::Done(_)
    ));
    assert_eq!(harness.review_drive().open_blocking().len(), 1);
}

/// Live run: the fix turn found the defect already fixed, ran the tests,
/// could not close the seeded plan, and ended as `plan_stall`. The loop
/// must re-audit instead of giving up on a green tree.
#[tokio::test]
async fn review_fix_turn_plan_stall_still_reaudits() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let Some(server) = MockPipe::new(vec![
        text(&incomplete_with(&[P0_ROW])),
        tool(
            "call_t",
            "bash",
            &serde_json::json!({"command": "cargo test --offline --quiet"}),
        ),
        // Two edit demands (one continuation plus the plan last chance),
        // then `plan_stall`; the policy tells the model not to close a
        // plan step it made no edits for.
        text("Already fixed: nick() rejects empty input and the tests pass."),
        text("Nothing to edit; the defect is already fixed."),
        text("Still nothing to edit."),
        text(&incomplete_with(&[])),
        text("must not be requested"),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, _) = start(&mut harness, "");
    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    let ReviewStep::RunPrompt(fix) = harness.review_next_step(&outcome, &mut ui) else {
        panic!("fix pass expected");
    };
    let outcome = run(&mut harness, &fix, &mut ui).await;
    assert_eq!(outcome.stop_reason, TurnStopReason::Error, "{outcome:?}");
    assert!(
        ui.statuses.iter().any(|s| s.contains("asking for an edit")),
        "{:?}",
        ui.statuses
    );
    let ReviewStep::RunPrompt(reaudit) = harness.review_next_step(&outcome, &mut ui) else {
        panic!("a fix turn that ended on a policy error still re-audits");
    };
    assert!(reaudit.contains("Re-audit after fix pass 1/3"), "{reaudit}");
    let drive = harness.review_drive();
    assert_eq!(drive.phase, ReviewPhase::Reaudit);
    assert!(drive.fix_turn_error.is_some(), "{drive:?}");
    let before = ui.statuses.len();
    let outcome = run(&mut harness, &reaudit, &mut ui).await;
    assert!(
        ui.statuses[before..]
            .iter()
            .any(|s| s.starts_with("spec review · re-audit after pass 1/3 · fix turn ended early:")),
        "{:?}",
        &ui.statuses[before..]
    );
    match harness.review_next_step(&outcome, &mut ui) {
        ReviewStep::Done(summary) => {
            assert!(
                summary.contains("no P0/P1 defects after 1 fix pass(es)"),
                "{summary}"
            )
        }
        other => panic!("clean re-audit finishes, got {other:?}"),
    }
    assert!(
        harness.current_plan().is_empty(),
        "the stale checklist is dismissed"
    );
    assert_eq!(server.bodies.lock().unwrap().len(), 6);
}

#[tokio::test]
async fn review_audit_reasks_once_on_missing_block_then_stops() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let Some(server) = MockPipe::new(vec![
        text("Everything looks implemented to me."),
        text("Still prose, still no block."),
        text("must not be requested"),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, _) = start(&mut harness, "");
    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    let ReviewStep::RunPrompt(reask) = harness.review_next_step(&outcome, &mut ui) else {
        panic!("a reply without a block gets one format re-ask");
    };
    assert!(reask.starts_with("[hi:review] Format re-ask."), "{reask}");
    assert!(
        reask.contains("coverage: <state> | Welcome banner | <path:line or ->\n")
            && reask.contains("coverage: <state> | KICK command | <path:line or ->\n")
            && reask.contains("coverage: <state> | /topic command | <path:line or ->\n"),
        "the plan's checklist rows pre-fill the item column: {reask}"
    );
    assert_ne!(reask, REVIEW_FORMAT_HINT);
    assert_eq!(harness.review_drive().phase, ReviewPhase::Audit);

    let outcome = run(&mut harness, &reask, &mut ui).await;
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    let step = harness.review_next_step(&outcome, &mut ui);
    match step {
        ReviewStep::Stopped(text) => {
            assert!(text.contains("no parseable <review> block"), "{text}")
        }
        other => panic!("expected Stopped, got {other:?}"),
    }
    assert_eq!(harness.review_drive().phase, ReviewPhase::Stopped);
    assert!(
        harness
            .review_status_text()
            .contains("stopped: no parseable")
    );
    assert_eq!(server.bodies.lock().unwrap().len(), 2);
    assert!(
        server.bodies.lock().unwrap()[1].contains("Format re-ask"),
        "the second request carries the format hint"
    );
    assert!(harness.review_next_step(&outcome, &mut ui) == ReviewStep::Idle);
}

#[tokio::test]
async fn review_fix_turn_seeds_plan_and_requires_tests() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let mut scripts = vec![text(&incomplete_with(&[P0_ROW]))];
    scripts.extend(fix_turn_scripts());
    scripts.push(text(&incomplete_with(&[])));
    scripts.push(text("must not be requested"));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, notes) = start(&mut harness, "");
    assert_eq!(
        notes[0],
        "spec review · audit · up to 3 fix passes · plan.md + docs/SPEC.md"
    );

    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    let ReviewStep::RunPrompt(fix) = harness.review_next_step(&outcome, &mut ui) else {
        panic!("a P0 finding starts a fix pass");
    };
    assert!(fix.contains("Fix pass 1/3"), "{fix}");
    assert!(
        fix.contains("1. [P0] Reject empty nicknames — src/server.rs:1"),
        "{fix}"
    );
    let seeded = ui.plans.last().expect("fix pass posts the checklist");
    assert_eq!(seeded.len(), 1);
    assert_eq!(
        seeded[0].title,
        "[P0] Reject empty nicknames — src/server.rs:1"
    );
    assert_eq!(seeded[0].status, hi_tools::PlanStatus::Pending);
    assert_eq!(harness.current_plan(), seeded.as_slice());

    let before = ui.statuses.len();
    let outcome = run(&mut harness, &fix, &mut ui).await;
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::Completed,
        "{:?}",
        outcome.error
    );
    assert_eq!(
        ui.statuses[before..].first().map(String::as_str),
        Some("spec review · fix pass 1/3 · 1 P0/P1 · plan.md + docs/SPEC.md"),
        "the fix turn announces its phase once, at start: {:?}",
        &ui.statuses[before..]
    );
    assert_eq!(
        ui.statuses[before..]
            .iter()
            .filter(|s| s.starts_with("spec review ·"))
            .count(),
        1,
        "{:?}",
        &ui.statuses[before..]
    );
    assert!(
        ui.statuses[before..]
            .iter()
            .any(|status| status.contains("asking for tests")),
        "a fix turn that edits then stops must be asked for tests, got {:?}",
        &ui.statuses[before..]
    );
    assert!(
        ui.tool_calls
            .iter()
            .any(|(name, args)| name == "bash" && args.contains("cargo test")),
        "{:?}",
        ui.tool_calls
    );
    assert!(
        fs::read_to_string(dir.path().join("src/server.rs"))
            .unwrap()
            .contains("assert!(!raw.is_empty())")
    );
    assert_eq!(outcome.changed_files, vec!["src/server.rs".to_string()]);
    assert!(
        hi_tools::PlanStep::all_complete(harness.current_plan()),
        "the seeded step is closed by update_plan, got {:?}",
        harness.current_plan()
    );

    let ReviewStep::RunPrompt(reaudit) = harness.review_next_step(&outcome, &mut ui) else {
        panic!("a completed fix pass re-audits");
    };
    assert!(reaudit.contains("Re-audit after fix pass 1/3"), "{reaudit}");
    assert!(reaudit.contains("- src/server.rs\n"), "{reaudit}");
    assert!(reaudit.contains("[P0] Reject empty nicknames"), "{reaudit}");
    assert_eq!(harness.review_drive().phase, ReviewPhase::Reaudit);

    let outcome = run(&mut harness, &reaudit, &mut ui).await;
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    match harness.review_next_step(&outcome, &mut ui) {
        ReviewStep::Done(summary) => {
            assert!(
                summary.contains("no P0/P1 defects after 1 fix pass(es)"),
                "{summary}"
            );
            assert!(
                summary.contains("1/3 plan/spec items implemented"),
                "{summary}"
            );
        }
        other => panic!("clean re-audit finishes the loop, got {other:?}"),
    }
    assert!(
        harness.current_plan().is_empty(),
        "completed checklist is dismissed"
    );
    assert_eq!(
        harness.review_drive().all_changed_files,
        vec!["src/server.rs".to_string()]
    );
    assert_eq!(server.bodies.lock().unwrap().len(), 7);
}

#[tokio::test]
async fn review_loop_stops_on_identical_findings() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let mut scripts = vec![text(&incomplete_with(&[P0_ROW]))];
    scripts.extend(fix_turn_scripts());
    scripts.push(text(&incomplete_with(&[
        "finding: P0 | Reject empty nicknames | src/server.rs:7",
    ])));
    scripts.push(text("must not be requested"));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, _) = start(&mut harness, "");
    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    let ReviewStep::RunPrompt(fix) = harness.review_next_step(&outcome, &mut ui) else {
        panic!("fix pass expected");
    };
    let outcome = run(&mut harness, &fix, &mut ui).await;
    let ReviewStep::RunPrompt(reaudit) = harness.review_next_step(&outcome, &mut ui) else {
        panic!("re-audit expected");
    };
    let outcome = run(&mut harness, &reaudit, &mut ui).await;
    match harness.review_next_step(&outcome, &mut ui) {
        ReviewStep::Stopped(text) => {
            assert!(text.contains("no progress"), "{text}");
            assert!(
                text.contains("1 P0/P1 finding(s) remain after fix pass 1"),
                "{text}"
            );
        }
        other => panic!("same finding after a fix pass must stop, got {other:?}"),
    }
    assert_eq!(harness.review_drive().phase, ReviewPhase::Stopped);
    assert_eq!(harness.review_drive().pass, 1);
    assert_eq!(harness.review_open_blocking().len(), 1);
    assert!(
        harness.current_plan().is_empty(),
        "the review's checklist goes with the loop; the report lists what is open"
    );
    assert_eq!(ui.plans.last().map(Vec::len), Some(0), "{:?}", ui.plans);
    assert_eq!(server.bodies.lock().unwrap().len(), 7, "no second fix pass");
}

#[tokio::test]
async fn review_loop_respects_pass_cap() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let mut scripts = vec![text(&incomplete_with(&[P0_ROW]))];
    scripts.extend(fix_turn_scripts());
    scripts.push(text(&incomplete_with(&[
        "finding: P1 | KICK ignores operator status | src/server.rs:1",
    ])));
    scripts.push(text("must not be requested"));
    let Some(server) = MockPipe::new(scripts) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let prompt = harness.begin_review(&[], false, false, 1).unwrap();
    assert!(prompt.contains("up to 1 fix passes"), "{prompt}");
    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    let ReviewStep::RunPrompt(fix) = harness.review_next_step(&outcome, &mut ui) else {
        panic!("fix pass expected");
    };
    assert!(fix.contains("Fix pass 1/1"));
    let outcome = run(&mut harness, &fix, &mut ui).await;
    let ReviewStep::RunPrompt(reaudit) = harness.review_next_step(&outcome, &mut ui) else {
        panic!("re-audit expected");
    };
    let outcome = run(&mut harness, &reaudit, &mut ui).await;
    match harness.review_next_step(&outcome, &mut ui) {
        ReviewStep::Stopped(text) => {
            assert!(text.contains("pass cap reached"), "{text}");
            assert!(
                text.contains("1 P0/P1 finding(s) still open after 1 fix pass(es)"),
                "{text}"
            );
        }
        other => panic!("a new finding past the cap must stop, got {other:?}"),
    }
    assert_eq!(
        harness.review_open_blocking()[0].title,
        "KICK ignores operator status"
    );
    assert_eq!(server.bodies.lock().unwrap().len(), 7);
}

#[tokio::test]
async fn review_cancel_pauses_and_resume_continues_same_phase() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let Some(server) = MockPipe::new(vec![
        text(&verdict_block(&[
            "verdict: COMPLETE",
            "coverage: implemented | Welcome banner | src/server.rs:1",
            "coverage: implemented | KICK command | src/server.rs:1",
            "coverage: implemented | /topic command | src/server.rs:1",
            "finding: none",
        ])),
        text("must not be requested"),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, _) = start(&mut harness, "");
    let cancel = TurnCancellation::new();
    cancel.cancel();
    let mut ui = TestUi::default();
    let outcome = harness
        .run_turn_cancellable(&prompt, &mut ui, cancel)
        .await
        .unwrap();
    assert_eq!(outcome.stop_reason, TurnStopReason::Cancelled);
    match harness.review_next_step(&outcome, &mut ui) {
        ReviewStep::Paused(text) => assert!(text.contains("paused during audit"), "{text}"),
        other => panic!("cancel pauses, got {other:?}"),
    }
    assert_eq!(ui.suggested_prompts, vec!["/review".to_string()]);
    assert!(harness.review_drive().paused);
    assert!(harness.review_status_text().contains("paused"));
    assert_eq!(
        harness.review_next_step(&outcome, &mut ui),
        ReviewStep::Idle,
        "a paused drive does not advance on its own"
    );
    assert!(server.bodies.lock().unwrap().is_empty());

    // The drive survives a session reload.
    let snapshot = harness.snapshot_state();
    let mut reloaded = test_harness(&server.url, dir.path().to_path_buf());
    reloaded.apply_loaded_session(snapshot);
    assert_eq!(reloaded.review_drive().phase, ReviewPhase::Audit);
    assert!(reloaded.review_drive().paused);

    let (resumed, notes) = start(&mut reloaded, "");
    assert!(
        notes[0].starts_with("resuming spec review · audit"),
        "{notes:?}"
    );
    assert!(
        resumed.contains("Spec-coverage audit"),
        "same phase replays its prompt"
    );
    assert!(!reloaded.review_drive().paused);
    let outcome = run(&mut reloaded, &resumed, &mut ui).await;
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    match reloaded.review_next_step(&outcome, &mut ui) {
        ReviewStep::Done(summary) => {
            assert!(
                summary.contains("all 3 plan/spec items implemented"),
                "{summary}"
            );
            assert!(summary.contains("no P0/P1 defects"), "{summary}");
        }
        other => panic!("clean verdict finishes, got {other:?}"),
    }
    assert!(reloaded.review_drive().coverage_complete());
    assert_eq!(server.bodies.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn review_stop_clears_a_seeded_fix_checklist() {
    let dir = tempfile::tempdir().unwrap();
    write_inputs(dir.path());
    let Some(server) = MockPipe::new(vec![text(&incomplete_with(&[P0_ROW]))]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, _) = start(&mut harness, "");
    let mut ui = TestUi::default();
    let outcome = run(&mut harness, &prompt, &mut ui).await;
    assert!(matches!(
        harness.review_next_step(&outcome, &mut ui),
        ReviewStep::RunPrompt(_)
    ));
    assert_eq!(harness.current_plan().len(), 1);
    let status = harness.review_command("status");
    assert!(
        matches!(&status, ReviewCommand::Message(text) if text.starts_with("spec review · fix pass 1/3")),
        "{status:?}"
    );
    let ReviewCommand::Message(stopped) = harness.review_command("stop") else {
        panic!("stop replies with a message");
    };
    assert_eq!(stopped, "spec review stopped: stopped by /review stop");
    assert!(
        harness.current_plan().is_empty(),
        "the fix checklist is dropped"
    );
    assert_eq!(harness.review_drive().phase, ReviewPhase::Stopped);
    let ReviewCommand::Message(again) = harness.review_command("stop") else {
        panic!("stop replies with a message");
    };
    assert!(again.starts_with("spec review: nothing to stop"), "{again}");
}

#[tokio::test]
async fn review_without_plan_or_spec_falls_back_to_readme_with_notice() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("README.md"),
        "# Chat\n\nGreets with Welcome.\n\n1. Ask for an outcome.\n2. Run it.\n",
    )
    .unwrap();
    let Some(server) = MockPipe::new(vec![]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let (prompt, notes) = start(&mut harness, "audit");
    assert_eq!(notes[0], "spec review · audit only · README.md");
    assert_eq!(
        notes[1],
        "spec review: no plan.md or spec.md found; auditing against README.md"
    );
    assert!(
        prompt.contains("- readme: README.md (fallback: no plan.md or spec.md;"),
        "{prompt}"
    );
    assert!(
        harness.review_drive().checklist_items.is_empty(),
        "a README's numbered list is not a checklist"
    );
    assert!(
        harness
            .review_status_text()
            .contains("no plan.md or spec.md found")
    );

    // Large workspace, still no spec: the audit narrows to recent work
    // (`review_harness_scope_tests`); an explicit directory is the scope.
    fs::create_dir_all(dir.path().join("src")).unwrap();
    for index in 0..=crate::review_scope::LARGE_WORKSPACE_FILES {
        fs::write(dir.path().join(format!("src/f{index}.rs")), "").unwrap();
    }
    let mut large = test_harness(&server.url, dir.path().to_path_buf());
    let (scoped, notes) = start(&mut large, "audit src");
    assert!(scoped.contains("Scope: limit the code audit to src"));
    assert_eq!(notes[0], "spec review · audit only · in src · README.md");

    let empty = tempfile::tempdir().unwrap();
    let mut bare = test_harness(&server.url, empty.path().to_path_buf());
    let (prompt, notes) = start(&mut bare, "");
    assert_eq!(
        notes[0],
        "spec review · audit · up to 3 fix passes · no plan/spec (defects only)"
    );
    assert!(notes[1].contains("auditing for defects only"), "{notes:?}");
    assert!(
        prompt.contains("Inputs: none found (no plan.md or spec.md). This is a defects-only audit"),
        "{prompt}"
    );
    assert!(
        prompt.starts_with("[hi:review] Defect audit (up to 3 fix passes"),
        "{prompt}"
    );

    let ReviewCommand::Message(missing) = bare.review_command("docs/nope.md") else {
        panic!("missing explicit input refuses to start");
    };
    assert_eq!(missing, "spec review: no such file: docs/nope.md");
    assert!(server.bodies.lock().unwrap().is_empty());
}
