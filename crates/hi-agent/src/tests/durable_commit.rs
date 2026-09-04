use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use hi_workspace::{ExecutionDisposition, InMemoryWorkspaceController, WorkspaceController};

use super::common::{agent, config};

struct StageSession {
    records: Arc<Mutex<Vec<crate::WorkspaceTranscriptExecution>>>,
    fail: bool,
}

impl crate::SessionSink for StageSession {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> Result<()> {
        Ok(())
    }

    fn stage_workspace_execution(
        &mut self,
        record: &crate::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        if self.fail {
            anyhow::bail!("commit stage unavailable");
        }
        self.records.lock().unwrap().push(record.clone());
        Ok(())
    }

    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> Result<()> {
        Ok(())
    }
}

struct Fixture {
    subject: crate::Agent,
    controller: Arc<InMemoryWorkspaceController>,
    records: Arc<Mutex<Vec<crate::WorkspaceTranscriptExecution>>>,
}

fn git(root: &Path, args: &[&str]) {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn git_stdout(root: &Path, args: &[&str]) -> String {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    assert!(output.status.success(), "git {args:?} failed");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn pipefs_subject(fail_stage: bool) -> Fixture {
    let cfg = config();
    let workspace_root = cfg.paths.workspace_root.clone();
    let state_root = cfg.paths.state_root.clone();
    git(&workspace_root, &["init", "-q"]);
    git(
        &workspace_root,
        &["config", "user.email", "test@example.test"],
    );
    git(&workspace_root, &["config", "user.name", "Harness Test"]);
    std::fs::write(workspace_root.join("tracked.txt"), "baseline\n").unwrap();
    git(&workspace_root, &["add", "tracked.txt"]);
    git(&workspace_root, &["commit", "-qm", "baseline"]);

    let mut subject = agent(Vec::new(), cfg);
    let controller = Arc::new(InMemoryWorkspaceController::new_pipefs(
        "commit-workspace",
        "commit-session",
        2,
        true,
        &workspace_root,
        &state_root,
    ));
    subject
        .install_workspace_controller(controller.clone())
        .unwrap();
    let records = Arc::new(Mutex::new(Vec::new()));
    subject.set_session(Box::new(StageSession {
        records: records.clone(),
        fail: fail_stage,
    }));
    Fixture {
        subject,
        controller,
        records,
    }
}

#[tokio::test]
async fn post_add_secret_refusal_is_staged_and_settled_as_failed() {
    let Fixture {
        mut subject,
        controller,
        records,
    } = pipefs_subject(false);
    std::fs::write(
        subject.workspace_root().join("tracked.txt"),
        "api_key=sk-abcdefghijklmnopqrstuvwxyz123456\n",
    )
    .unwrap();

    let error = subject
        .commit_session_changes(&["tracked.txt".into()])
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("looks like it contains secrets"));
    assert!(
        git_stdout(
            subject.workspace_root(),
            &["diff", "--cached", "--name-only"]
        )
        .trim()
        .is_empty(),
        "the partial staging effect must be reconciled and undone"
    );
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].execution.disposition,
        ExecutionDisposition::Failed
    );
    assert!(records[0].execution.workspace_may_have_changed);
    assert!(records[0].execution.external_effect_may_have_occurred);
    assert_eq!(records[0].calls[0].name, "session_git_commit");
    assert!(records[0].calls[0].result.contains("secrets"));
    assert!(controller.status().state.admits_mutation());
}

#[tokio::test]
async fn committed_bytes_are_not_reported_successful_when_exact_staging_fails() {
    let Fixture {
        mut subject,
        controller,
        records,
    } = pipefs_subject(true);
    std::fs::write(subject.workspace_root().join("tracked.txt"), "changed\n").unwrap();

    let error = subject
        .commit_session_changes(&["tracked.txt".into()])
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("commit stage unavailable"));
    assert!(records.lock().unwrap().is_empty());
    assert_eq!(
        git_stdout(subject.workspace_root(), &["log", "-1", "--pretty=%s"]).trim(),
        "update tracked.txt",
        "the test must exercise lost publication after a real Git commit"
    );
    assert_eq!(
        controller.status().state,
        hi_workspace::WorkspaceState::RecoveryRequired
    );
}

#[tokio::test]
async fn closed_admission_prevents_git_from_staging_paths() {
    let Fixture {
        mut subject,
        controller,
        records,
    } = pipefs_subject(false);
    std::fs::write(subject.workspace_root().join("tracked.txt"), "changed\n").unwrap();
    let permit = controller
        .begin(hi_workspace::MutationIntent::workspace("existing writer"))
        .await
        .unwrap();

    let error = subject
        .commit_session_changes(&["tracked.txt".into()])
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("workspace controller refused"));
    assert!(records.lock().unwrap().is_empty());
    assert!(
        git_stdout(
            subject.workspace_root(),
            &["diff", "--cached", "--name-only"]
        )
        .trim()
        .is_empty()
    );
    let settled = controller
        .settle(permit, hi_workspace::ExecutionReport::succeeded(None))
        .await;
    assert!(settled.receipt.is_some());
}
