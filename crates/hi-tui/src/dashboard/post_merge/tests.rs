use super::*;

fn git(root: &std::path::Path, args: &[&str]) -> String {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

struct MergedCandidate {
    root: tempfile::TempDir,
    _scratch: tempfile::TempDir,
    candidate: PathBuf,
    base: String,
}

impl MergedCandidate {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        git(root.path(), &["init", "-q"]);
        git(root.path(), &["config", "user.name", "test"]);
        git(
            root.path(),
            &["config", "user.email", "test@example.invalid"],
        );
        std::fs::write(root.path().join("source.rs"), "original\n").unwrap();
        git(root.path(), &["add", "source.rs"]);
        git(root.path(), &["commit", "-qm", "base"]);
        let base = git(root.path(), &["rev-parse", "HEAD"]);
        let scratch = tempfile::tempdir().unwrap();
        let candidate = scratch.path().join("candidate");
        worktree::add_worktree(root.path(), &candidate, &base).unwrap();
        std::fs::write(candidate.join("source.rs"), "candidate\n").unwrap();
        assert!(worktree::apply_changes_to(&candidate, &base, root.path()).unwrap());
        Self {
            root,
            _scratch: scratch,
            candidate,
            base,
        }
    }

    fn assert_preserved(&self) {
        assert_eq!(
            std::fs::read_to_string(self.candidate.join("source.rs")).unwrap(),
            "candidate\n"
        );
        assert_eq!(git(&self.candidate, &["rev-parse", "HEAD"]), self.base);
    }
}

impl Drop for MergedCandidate {
    fn drop(&mut self) {
        worktree::cleanup(self.root.path(), std::slice::from_ref(&self.candidate));
    }
}

#[tokio::test]
async fn failed_combined_tree_verification_keeps_the_original_candidate() {
    let fixture = MergedCandidate::new();
    let done = check(
        fixture.root.path().into(),
        fixture.candidate.clone(),
        Some("printf broken > source.rs".into()),
        None,
    )
    .await;
    assert!(matches!(
        done,
        RowDone::PostVerify {
            verify_ok: Some(false),
            new_base: Err(_)
        }
    ));
    fixture.assert_preserved();
    assert_eq!(
        std::fs::read_to_string(fixture.root.path().join("source.rs")).unwrap(),
        "broken"
    );
}

#[tokio::test]
async fn cancellation_during_verification_does_not_reset_the_candidate() {
    let fixture = MergedCandidate::new();
    let ready = fixture._scratch.path().join("verifier-started");
    let quoted = format!("'{}'", ready.to_str().unwrap().replace('\'', "'\\''"));
    let cancellation = CancellationToken::new();
    let worker = check(
        fixture.root.path().into(),
        fixture.candidate.clone(),
        Some(format!(
            "printf broken > source.rs; touch {quoted}; sleep 30"
        )),
        Some(cancellation.clone()),
    );
    tokio::pin!(worker);
    let started = async {
        while !ready.exists() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    };
    tokio::select! {
        _ = &mut worker => panic!("verification returned before cancellation"),
        result = tokio::time::timeout(std::time::Duration::from_secs(5), started) => result.unwrap(),
    }
    cancellation.cancel();
    let done = tokio::time::timeout(std::time::Duration::from_secs(3), worker)
        .await
        .unwrap();
    assert!(
        matches!(done, RowDone::PostVerify { new_base: Err(error), .. } if error.contains("cancelled"))
    );
    fixture.assert_preserved();
}

#[tokio::test]
async fn reset_failure_is_reported_and_preserves_the_original_base() {
    let fixture = MergedCandidate::new();
    let index = git(
        &fixture.candidate,
        &[
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "index.lock",
        ],
    );
    std::fs::write(&index, "held by another Git operation").unwrap();
    let done = check(
        fixture.root.path().into(),
        fixture.candidate.clone(),
        Some("true".into()),
        None,
    )
    .await;
    std::fs::remove_file(index).unwrap();
    assert!(matches!(
        done,
        RowDone::PostVerify { verify_ok: Some(true), new_base: Err(error) }
            if error.contains("could not refresh") && error.contains("index.lock")
    ));
    fixture.assert_preserved();
}

#[tokio::test]
async fn successful_post_merge_check_refreshes_the_candidate() {
    let fixture = MergedCandidate::new();
    for verify in [Some("true".into()), None] {
        let done = check(
            fixture.root.path().into(),
            fixture.candidate.clone(),
            verify,
            None,
        )
        .await;
        let RowDone::PostVerify {
            new_base: Ok(base), ..
        } = done
        else {
            panic!("successful check did not refresh candidate");
        };
        assert_eq!(git(&fixture.candidate, &["rev-parse", "HEAD"]), base);
        assert!(
            worktree::changed_files(&fixture.candidate, &base)
                .unwrap()
                .is_empty()
        );
    }
}

fn pending_row() -> FleetRow {
    let mut row = super::super::tests::row();
    row.merge = MergeState::Merged(1);
    row.changed = vec!["source.rs".into()];
    row.pending.push_back("follow up with the next fix".into());
    row.goal = Some(RowGoal {
        done: 0,
        total: 2,
        active: true,
        paused: false,
        drive: None,
        phases: Vec::new(),
    });
    row
}

#[tokio::test]
async fn failed_post_merge_gate_preserves_queued_input_and_parks_goal_drive() {
    for verify_ok in [Some(false), Some(true), None] {
        for queued in [true, false] {
            let mut app = crate::tests::test_app("test", "test");
            let mut row = pending_row();
            if !queued {
                row.pending.clear();
            }
            app.fleet.push(row);
            let launcher = super::super::tests::test_fleet_launcher();
            let (line_tx, _) = mpsc::unbounded_channel();
            let mut in_flight = FuturesUnordered::new();
            finish(
                &mut app,
                0,
                verify_ok,
                Err("refresh failed".into()),
                &launcher,
                &line_tx,
                &mut in_flight,
            );
            let row = &app.fleet[0];
            assert!(row.state == RowState::Failed);
            assert!(row.attention && row.stale);
            assert_eq!(row.pending.len(), usize::from(queued));
            assert_eq!(row.base, "abc");
            assert_eq!(row.changed, ["source.rs"]);
            assert!(
                in_flight.is_empty(),
                "failed post-merge gate started another turn"
            );
        }
    }
}

#[tokio::test]
async fn refresh_failure_cannot_report_workflow_success() {
    let mut app = crate::tests::test_app("test", "test");
    let mut row = pending_row();
    let (tx, rx) = oneshot::channel();
    row.workflow_reply = Some(tx);
    row.workflow_status = Some(WorkflowJobStatus::Running);
    app.fleet.push(row);
    let launcher = super::super::tests::test_fleet_launcher();
    let (line_tx, _) = mpsc::unbounded_channel();
    let mut in_flight = FuturesUnordered::new();
    finish(
        &mut app,
        0,
        Some(true),
        Err("reset failed".into()),
        &launcher,
        &line_tx,
        &mut in_flight,
    );
    assert!(!rx.await.unwrap().unwrap().success);
    assert_eq!(
        app.fleet[0].workflow_status,
        Some(WorkflowJobStatus::Failed)
    );
    assert_eq!(app.fleet[0].pending.len(), 1);
    assert!(in_flight.is_empty());
}

#[tokio::test]
async fn cancellation_racing_a_completed_reset_still_records_the_actual_base() {
    let mut app = crate::tests::test_app("test", "test");
    let mut row = pending_row();
    row.workflow_status = Some(WorkflowJobStatus::Cancelled);
    app.fleet.push(row);
    let launcher = super::super::tests::test_fleet_launcher();
    let (line_tx, _) = mpsc::unbounded_channel();
    let mut in_flight = FuturesUnordered::new();
    finish(
        &mut app,
        0,
        Some(true),
        Ok("new-base".into()),
        &launcher,
        &line_tx,
        &mut in_flight,
    );
    assert!(app.fleet[0].state == RowState::Failed);
    assert_eq!(app.fleet[0].base, "new-base");
    assert!(app.fleet[0].changed.is_empty());
    assert_eq!(app.fleet[0].pending.len(), 1);
    assert!(in_flight.is_empty());
}

#[tokio::test]
async fn confirmed_refresh_releases_the_next_queued_prompt() {
    let mut app = crate::tests::test_app("test", "test");
    let scratch = tempfile::tempdir().unwrap();
    let mut row = pending_row();
    row.worktree = scratch.path().into();
    row.session = scratch.path().join("session.jsonl");
    app.fleet.push(row);
    let mut launcher = super::super::tests::test_fleet_launcher();
    // The test executable exists on every supported host; the child exits on
    // the harness-only arguments, after exercising the actual spawn path.
    launcher.exe = std::env::current_exe().unwrap();
    let (line_tx, _) = mpsc::unbounded_channel();
    let mut in_flight = FuturesUnordered::new();
    finish(
        &mut app,
        0,
        Some(true),
        Ok("new-base".into()),
        &launcher,
        &line_tx,
        &mut in_flight,
    );
    assert_eq!(app.fleet[0].base, "new-base");
    assert!(app.fleet[0].changed.is_empty());
    assert!(app.fleet[0].pending.is_empty());
    assert!(app.fleet[0].state == RowState::Working);
    assert_eq!(in_flight.len(), 1);
    assert!(
        app.fleet[0]
            .tail
            .iter()
            .any(|line| line.contains("follow up with the next fix"))
    );
}

fn dashboard_fixture(fixture: &MergedCandidate) -> (App, FleetLauncher) {
    let mut app = crate::tests::test_app("test", "test");
    app.workspace_root = fixture.root.path().into();
    let mut row = pending_row();
    row.worktree = fixture.candidate.clone();
    row.base = fixture.base.clone();
    row.session = fixture._scratch.path().join("session.jsonl");
    app.fleet.push(row);
    let mut launcher = super::super::tests::test_fleet_launcher();
    launcher.workspace_root = fixture.root.path().into();
    (app, launcher)
}

async fn cancel_when_ready(
    app: &mut App,
    in_flight: &mut FuturesUnordered<RowFut>,
    ready: &std::path::Path,
) -> RowDone {
    let started = async {
        while !ready.exists() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    };
    let worker = in_flight.next();
    tokio::pin!(worker);
    tokio::select! {
        _ = &mut worker => panic!("operation finished before cancellation"),
        result = tokio::time::timeout(std::time::Duration::from_secs(5), started) => result.unwrap(),
    }
    assert!(
        app.fleet[0].kill.is_none(),
        "this must exercise cancellation after the child exits"
    );
    cancellation::request(app, 0);
    tokio::time::timeout(std::time::Duration::from_secs(3), worker)
        .await
        .unwrap()
        .unwrap()
        .1
}

#[tokio::test]
async fn ctrl_k_stops_an_ordinary_dashboard_merge_check_and_parks_queued_input() {
    let fixture = MergedCandidate::new();
    std::fs::write(fixture.root.path().join("source.rs"), "original\n").unwrap();
    let (mut app, mut launcher) = dashboard_fixture(&fixture);
    app.fleet[0].merge = MergeState::None;
    let ready = fixture._scratch.path().join("merge-check-started");
    launcher.verify = Some(format!("touch '{}'; sleep 30", ready.display()));
    let (line_tx, _) = mpsc::unbounded_channel();
    let mut in_flight = FuturesUnordered::new();
    finish_turn(
        &mut app,
        0,
        true,
        false,
        &launcher,
        &line_tx,
        &mut in_flight,
    );
    let done = cancel_when_ready(&mut app, &mut in_flight, &ready).await;
    let RowDone::MergeCheck { changed, verified } = done else {
        panic!("wrong merge-check completion");
    };
    finish_merge_check(
        &mut app,
        0,
        changed,
        verified,
        &launcher,
        &line_tx,
        &mut in_flight,
    );
    assert!(app.fleet[0].state == RowState::Failed);
    assert_eq!(app.fleet[0].pending.len(), 1);
    assert_eq!(app.fleet[0].changed, ["source.rs"]);
    assert!(in_flight.is_empty());
    fixture.assert_preserved();
    assert_eq!(
        std::fs::read_to_string(fixture.root.path().join("source.rs")).unwrap(),
        "original\n"
    );
}

#[tokio::test]
async fn ctrl_k_stops_ordinary_post_merge_verification_without_reset_or_dequeue() {
    let fixture = MergedCandidate::new();
    let (mut app, mut launcher) = dashboard_fixture(&fixture);
    let ready = fixture._scratch.path().join("post-check-started");
    launcher.verify = Some(format!("touch '{}'; sleep 30", ready.display()));
    let (line_tx, _) = mpsc::unbounded_channel();
    let mut in_flight = FuturesUnordered::new();
    queue(&mut app, 0, &launcher, &mut in_flight);
    let done = cancel_when_ready(&mut app, &mut in_flight, &ready).await;
    let RowDone::PostVerify {
        verify_ok,
        new_base,
    } = done
    else {
        panic!("wrong post-merge completion");
    };
    finish(
        &mut app,
        0,
        verify_ok,
        new_base,
        &launcher,
        &line_tx,
        &mut in_flight,
    );
    assert!(app.fleet[0].state == RowState::Failed);
    assert_eq!(app.fleet[0].pending.len(), 1);
    assert!(in_flight.is_empty());
    fixture.assert_preserved();
}

#[tokio::test]
async fn ctrl_k_stops_post_merge_checkpoint_filters_without_resetting_the_candidate() {
    let fixture = MergedCandidate::new();
    let (mut app, launcher) = dashboard_fixture(&fixture);
    let ready = fixture._scratch.path().join("checkpoint-started");
    std::fs::write(
        fixture.root.path().join(".gitattributes"),
        "source.rs filter=slow\n",
    )
    .unwrap();
    git(
        fixture.root.path(),
        &[
            "config",
            "filter.slow.clean",
            &format!("touch '{}'; sleep 30; cat", ready.display()),
        ],
    );
    git(
        fixture.root.path(),
        &["config", "filter.slow.required", "true"],
    );
    let (line_tx, _) = mpsc::unbounded_channel();
    let mut in_flight = FuturesUnordered::new();
    queue(&mut app, 0, &launcher, &mut in_flight);
    let done = cancel_when_ready(&mut app, &mut in_flight, &ready).await;
    let RowDone::PostVerify {
        verify_ok,
        new_base,
    } = done
    else {
        panic!("wrong checkpoint completion");
    };
    assert!(verify_ok.is_none());
    assert!(new_base.is_err());
    finish(
        &mut app,
        0,
        verify_ok,
        new_base,
        &launcher,
        &line_tx,
        &mut in_flight,
    );
    assert!(app.fleet[0].state == RowState::Failed);
    assert_eq!(app.fleet[0].pending.len(), 1);
    assert!(in_flight.is_empty());
    fixture.assert_preserved();
}

#[tokio::test]
async fn cancellation_racing_a_successful_merge_check_cannot_start_an_apply() {
    let fixture = MergedCandidate::new();
    let (mut app, launcher) = dashboard_fixture(&fixture);
    app.fleet[0].merge = MergeState::None;
    let (line_tx, _) = mpsc::unbounded_channel();
    let mut in_flight = FuturesUnordered::new();
    finish_turn(
        &mut app,
        0,
        true,
        false,
        &launcher,
        &line_tx,
        &mut in_flight,
    );
    let (_, RowDone::MergeCheck { changed, verified }) = in_flight.next().await.unwrap() else {
        panic!("wrong merge-check completion");
    };
    assert!(verified);
    cancellation::request(&mut app, 0);
    finish_merge_check(
        &mut app,
        0,
        changed,
        verified,
        &launcher,
        &line_tx,
        &mut in_flight,
    );
    assert!(app.fleet[0].state == RowState::Failed);
    assert_eq!(app.fleet[0].pending.len(), 1);
    assert!(in_flight.is_empty());
}

#[tokio::test]
async fn cancellation_racing_a_committed_apply_records_the_merge_and_stops_followups() {
    let fixture = MergedCandidate::new();
    let (mut app, launcher) = dashboard_fixture(&fixture);
    app.fleet[0].merge = MergeState::None;
    cancellation::reset(&mut app, 0);
    cancellation::request(&mut app, 0);
    let (line_tx, _) = mpsc::unbounded_channel();
    let mut in_flight = FuturesUnordered::new();
    finish_merge_apply(
        &mut app,
        0,
        vec!["source.rs".into()],
        Ok(()),
        &launcher,
        &line_tx,
        &mut in_flight,
    );
    assert!(app.fleet[0].state == RowState::Failed);
    assert!(app.fleet[0].merge == MergeState::Merged(1));
    assert!(app.fleet[0].stale);
    assert_eq!(app.fleet[0].pending.len(), 1);
    assert!(in_flight.is_empty());
}

#[tokio::test]
async fn late_turn_cancellation_keeps_reported_usage_before_notifying_the_workflow() {
    let fixture = MergedCandidate::new();
    let (mut app, launcher) = dashboard_fixture(&fixture);
    let (reply, result) = oneshot::channel();
    app.fleet[0].workflow_reply = Some(reply);
    app.fleet[0].workflow_status = Some(WorkflowJobStatus::Running);
    app.fleet[0].usage = 10;
    std::fs::write(
        report_path(&app.fleet[0]),
        r#"{
        "schema_version": 2,
        "usage": {"session": {"total_tokens": 4321}},
        "plan": {"done": 1, "total": 2, "pending": true, "drive": "running"}
    }"#,
    )
    .unwrap();
    cancellation::reset(&mut app, 0);
    cancellation::request(&mut app, 0);
    let (line_tx, _) = mpsc::unbounded_channel();
    let mut in_flight = FuturesUnordered::new();
    finish_turn(
        &mut app,
        0,
        true,
        false,
        &launcher,
        &line_tx,
        &mut in_flight,
    );
    let result = result.await.unwrap().unwrap();
    assert!(result.cancelled && !result.success);
    assert_eq!(result.tokens_used, 4321);
    assert_eq!(app.fleet[0].usage, 4321);
    assert_eq!(app.fleet[0].turns, 1);
    assert_eq!(app.fleet[0].plan.as_ref().unwrap().done, 1);
    assert!(app.fleet[0].state == RowState::Failed);
    assert_eq!(app.fleet[0].pending.len(), 1);
    assert!(in_flight.is_empty());
}
