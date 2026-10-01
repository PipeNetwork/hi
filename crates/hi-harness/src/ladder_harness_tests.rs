//! `/ladder` end to end against `MockPipe`: each attempt goes out on its
//! rung's model and effort, a failing check climbs, a passing one stops, a
//! policy error after edits is judged by the check, and the session's model
//! and verify command come back when the ladder ends.

use super::*;
use crate::ladder::DEFAULT_EFFORT_RUNGS;
use crate::pipe::test_support::MockPipe;
use crate::review_harness_tests::{test_harness, text, tool};
use crate::{ReasoningEffort, TestUi, TurnCancellation, TurnStopReason};
use std::fs;

const CHECK: &str = "grep -qx 42 answer.txt";

fn settings(rungs: &[&str]) -> LadderSettings {
    LadderSettings {
        rungs: rungs
            .iter()
            .map(|spec| LadderRung::parse(spec).unwrap())
            .collect(),
        verify: Some(CHECK.into()),
        ..LadderSettings::default()
    }
}

fn offline_harness(dir: &std::path::Path) -> Harness {
    test_harness("http://127.0.0.1:9", dir.to_path_buf())
}

/// The model and effort of each request the mock received.
fn requests(server: &MockPipe) -> Vec<(String, Option<String>)> {
    server
        .bodies
        .lock()
        .unwrap()
        .iter()
        .map(|body| {
            let body: serde_json::Value = serde_json::from_str(body).unwrap();
            (
                body["model"].as_str().unwrap_or_default().to_string(),
                body["reasoning_effort"].as_str().map(str::to_string),
            )
        })
        .collect()
}

/// Run `task` through the ladder the way the frontends do.
async fn climb(harness: &mut Harness, task: &str, ui: &mut TestUi) -> LadderStep {
    harness.begin_ladder(task).await.unwrap();
    let mut prompt = task.to_string();
    loop {
        let outcome = harness
            .run_turn_cancellable(&prompt, ui, TurnCancellation::new())
            .await
            .unwrap();
        match harness.ladder_next_step(&outcome, ui) {
            LadderStep::RunPrompt(next) => prompt = next,
            step => return step,
        }
    }
}

#[tokio::test]
async fn a_failing_check_climbs_and_a_passing_one_stops() {
    let dir = tempfile::tempdir().unwrap();
    let write = serde_json::json!({"path": "answer.txt", "content": "42\n"});
    let check = serde_json::json!({"command": CHECK});
    let Some(server) = MockPipe::new(vec![
        text("Looks correct to me."),
        tool("call_w", "write", &write),
        tool("call_c", "bash", &check),
        text("answer.txt now holds 42."),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    let base_model = harness.model();
    harness.set_ladder_settings(settings(&["pipe/cheap low", "pipe/strong high"]));
    let mut ui = TestUi::default();

    let step = climb(&mut harness, "Make answer.txt hold 42.", &mut ui).await;

    let LadderStep::Verified(summary) = step else {
        panic!("expected a pass, got {step:?}");
    };
    assert!(
        summary.contains("rung 2/2 (pipe/strong · high)"),
        "{summary}"
    );
    let ladder = harness.ladder();
    assert_eq!(ladder.phase, LadderPhase::Verified);
    assert_eq!(ladder.attempts.len(), 2);
    assert_eq!(ladder.attempts[0].changed_files, 0);
    assert_eq!(
        fs::read_to_string(dir.path().join("answer.txt")).unwrap(),
        "42\n"
    );
    assert_eq!(
        requests(&server),
        vec![
            ("pipe/cheap".into(), Some("low".into())),
            ("pipe/strong".into(), Some("high".into())),
            ("pipe/strong".into(), Some("high".into())),
            ("pipe/strong".into(), Some("high".into())),
        ],
        "each attempt goes out on its rung"
    );
    let retry = server.bodies.lock().unwrap()[1].clone();
    assert!(retry.contains("[ladder] attempt 2"), "{retry}");
    assert!(retry.contains("changed no files"), "{retry}");
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.starts_with("ladder · rung 2/2 pipe/strong · high")),
        "the climb is announced: {:?}",
        ui.statuses
    );
    assert_eq!(harness.model(), base_model, "the session model comes back");
    assert_eq!(
        harness.verify_command(),
        None,
        "the ladder's check does not outlive it"
    );
    assert_eq!(harness.effective_reasoning_effort(), None);
}

#[tokio::test]
async fn a_policy_error_after_edits_is_judged_by_the_check() {
    let dir = tempfile::tempdir().unwrap();
    let write = serde_json::json!({"path": "answer.txt", "content": "42\n"});
    let Some(server) = MockPipe::new(vec![
        tool("call_w", "write", &write),
        text("Done."),
        text("Done."),
    ]) else {
        return;
    };
    let mut harness = test_harness(&server.url, dir.path().to_path_buf());
    harness.set_ladder_settings(settings(&["pipe/cheap low", "pipe/strong high"]));
    let mut ui = TestUi::default();

    harness
        .begin_ladder("Fix answer.txt so it holds 42.")
        .await
        .unwrap();
    let outcome = harness
        .run_turn_cancellable(
            "Fix answer.txt so it holds 42.",
            &mut ui,
            TurnCancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::Error,
        "the model never ran a test, so the completion policy ends the turn"
    );
    assert_eq!(outcome.verification.as_deref(), Some("passed"));

    let step = harness.ladder_next_step(&outcome, &mut ui);
    assert!(matches!(step, LadderStep::Verified(_)), "{step:?}");
    assert_eq!(
        harness.ladder().attempts.len(),
        1,
        "no climb for a fix that passes"
    );
    assert_eq!(requests(&server).len(), 3);
}

#[tokio::test]
async fn begin_needs_a_check_and_a_direct_model() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = offline_harness(dir.path());
    let no_check = harness.begin_ladder("fix it").await.unwrap_err();
    assert!(no_check.contains("needs a check"), "{no_check}");
    assert!(!harness.ladder().is_active());

    harness.set_ladder_settings(settings(&["pipe/auto high"]));
    let managed_rung = harness.begin_ladder("fix it").await.unwrap_err();
    assert!(managed_rung.contains("pipe/auto"), "{managed_rung}");

    harness.set_ladder_settings(settings(&["low"]));
    harness.set_model("pipe/auto".into());
    let managed = harness.begin_ladder("fix it").await.unwrap_err();
    assert!(managed.contains("pipe/auto"), "{managed}");

    let empty = harness.begin_ladder("   ").await.unwrap_err();
    assert!(empty.contains("give it a task"), "{empty}");

    harness.set_model("pipe/kimi-3".into());
    harness.set_ladder_settings(LadderSettings {
        invalid_rungs: vec!["ladder rung `pipe/kimi-3 hihg`: expected …".into()],
        ..settings(&["low"])
    });
    let typo = harness.begin_ladder("fix it").await.unwrap_err();
    assert!(typo.contains("fix [ladder] rungs first"), "{typo}");
    assert!(typo.contains("pipe/kimi-3 hihg"), "{typo}");
}

#[tokio::test]
async fn a_rung_is_live_only_and_stop_restores_the_session() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = offline_harness(dir.path());
    let base_model = harness.model();
    harness.set_reasoning_effort(Some(ReasoningEffort::Xhigh));
    harness.set_ladder_settings(settings(&["pipe/cheap low", "pipe/strong high"]));

    let notes = harness.begin_ladder("fix it").await.unwrap();
    assert!(
        notes[0].starts_with("ladder · rung 1/2 pipe/cheap · low"),
        "{notes:?}"
    );
    assert_eq!(harness.model(), "pipe/cheap");
    assert_eq!(
        harness.effective_reasoning_effort(),
        Some(ReasoningEffort::Low),
        "the rung outranks a pinned /effort"
    );
    assert_eq!(harness.verify_command(), Some(CHECK));
    let snapshot = harness.snapshot_state();
    assert_eq!(snapshot.model.as_deref(), Some(base_model.as_str()));
    assert_eq!(snapshot.verify_command, None);
    let busy = harness.begin_ladder("another").await.unwrap_err();
    assert!(busy.contains("/ladder stop"), "{busy}");

    let stopped = harness.stop_ladder(STOPPED_BY_USER);
    assert!(stopped.starts_with("ladder stopped"), "{stopped}");
    assert_eq!(harness.model(), base_model);
    assert_eq!(harness.verify_command(), None);
    assert_eq!(
        harness.effective_reasoning_effort(),
        Some(ReasoningEffort::Xhigh)
    );
    assert_eq!(
        harness.stop_ladder(STOPPED_BY_USER),
        "ladder: nothing to stop"
    );
}

#[tokio::test]
async fn a_session_check_is_kept_after_the_ladder() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = offline_harness(dir.path());
    harness.set_verify_command(Some("make check".into()));
    harness.set_ladder_settings(settings(&["low", "high"]));
    harness.begin_ladder("fix it").await.unwrap();
    assert_eq!(
        harness.ladder().verify_command,
        "make check",
        "/verify wins"
    );
    assert!(!harness.ladder().temporary_verify);
    harness.stop_ladder(STOPPED_BY_USER);
    assert_eq!(harness.verify_command(), Some("make check"));
}

#[tokio::test]
async fn the_default_ladder_on_deepseek_merges_identical_rungs() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = offline_harness(dir.path());
    harness.set_model("pipe/deepseek-v4-flash".into());
    harness.set_ladder_settings(LadderSettings {
        verify: Some(CHECK.into()),
        ..LadderSettings::default()
    });
    let notes = harness.begin_ladder("fix it").await.unwrap();
    let labels: Vec<String> = (0..harness.ladder().rungs.len())
        .map(|index| harness.ladder().rung_label(index))
        .collect();
    assert_eq!(
        labels,
        [
            "pipe/deepseek-v4-flash · low",
            "pipe/deepseek-v4-flash · high"
        ],
        "medium and high are the same request on DeepSeek"
    );
    assert!(
        notes.iter().any(|note| note.contains("merged 1 rung(s)")),
        "{notes:?}"
    );
    harness.stop_ladder(STOPPED_BY_USER);

    harness.set_model("pipe/gpt-6-astra".into());
    harness.begin_ladder("fix it").await.unwrap();
    assert_eq!(harness.ladder().rungs.len(), DEFAULT_EFFORT_RUNGS.len());
}

#[tokio::test]
async fn ladder_command_reports_status_and_config() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = offline_harness(dir.path());
    let LadderCommand::Message(idle) = harness.ladder_command("").await else {
        panic!("status is a message");
    };
    assert!(idle.starts_with("ladder: not running"), "{idle}");
    assert!(idle.contains("low → "), "{idle}");
    assert!(idle.contains("verify: none"), "{idle}");

    harness.set_ladder_settings(settings(&["pipe/cheap low", "pipe/strong high"]));
    let LadderCommand::Start { prompt, notes } = harness.ladder_command(" fix it ").await else {
        panic!("a task starts the ladder");
    };
    assert_eq!(prompt, "fix it");
    assert!(!notes.is_empty());
    let LadderCommand::Message(running) = harness.ladder_command("status").await else {
        panic!("status is a message");
    };
    assert!(running.starts_with("ladder · rung 1/2"), "{running}");
    assert!(
        !running.contains("rungs:"),
        "config is shown only when idle"
    );
    let LadderCommand::Message(stopped) = harness.ladder_command("stop").await else {
        panic!("stop is a message");
    };
    assert!(stopped.contains(STOPPED_BY_USER), "{stopped}");
}

#[test]
fn the_configured_check_counts_as_verification() {
    let dir = tempfile::tempdir().unwrap();
    let mut harness = offline_harness(dir.path());
    let bash = |command: &str| serde_json::json!({ "command": command }).to_string();
    assert!(!harness.runs_verify_command("bash", &bash("./check.sh")));

    harness.set_verify_command(Some("cargo check --quiet && ./check.sh".into()));
    assert!(
        harness.runs_verify_command("bash", &bash("cd . && cargo check  --quiet && ./check.sh"))
    );
    assert!(
        !harness.runs_verify_command("bash", &bash("./check.sh")),
        "every segment runs"
    );
    assert!(!harness.runs_verify_command("read", &bash("cargo check --quiet && ./check.sh")));
    assert!(!harness.runs_verify_command("bash", "not json"));

    harness.set_verify_command(Some("make".into()));
    assert!(!harness.runs_verify_command("bash", &bash("make install")));
    assert!(harness.runs_verify_command("bash", &bash("make")));
}
