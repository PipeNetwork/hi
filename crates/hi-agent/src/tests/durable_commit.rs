use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use hi_workspace::{ExecutionDisposition, InMemoryWorkspaceController, WorkspaceController};

use super::common::{RecordingUi, agent, completion, config, write_completion};

struct StageSession {
    records: Arc<Mutex<Vec<crate::WorkspaceTranscriptExecution>>>,
    fail: bool,
    fail_settlement_marker: bool,
    events: Option<Arc<Mutex<Vec<String>>>>,
}

impl crate::SessionSink for StageSession {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> Result<()> {
        Ok(())
    }

    fn requires_local_workspace_execution_stage(&self) -> bool {
        true
    }

    fn stage_workspace_execution(
        &mut self,
        record: &crate::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        if let Some(events) = &self.events {
            events.lock().unwrap().push("stage".into());
        }
        if self.fail {
            anyhow::bail!("commit stage unavailable");
        }
        self.records.lock().unwrap().push(record.clone());
        Ok(())
    }

    fn settle_local_workspace_execution(&mut self, _: &hi_workspace::OperationId) -> Result<()> {
        if self.fail_settlement_marker {
            anyhow::bail!("transcript settlement marker unavailable");
        }
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
        fail_settlement_marker: false,
        events: None,
    }));
    Fixture {
        subject,
        controller,
        records,
    }
}

fn local_subject(fail_stage: bool) -> Fixture {
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
    let controller = Arc::new(InMemoryWorkspaceController::new_local(
        "commit-workspace",
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
        fail_settlement_marker: false,
        events: None,
    }));
    Fixture {
        subject,
        controller,
        records,
    }
}

struct OrderedDurability {
    events: Arc<Mutex<Vec<String>>>,
    fail: bool,
}

struct PromptAnchoringSession {
    events: Arc<Mutex<Vec<String>>>,
    fail_record: bool,
}

struct FailPlanSession;

impl crate::SessionSink for FailPlanSession {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> Result<()> {
        Ok(())
    }

    fn record_plan(&mut self, _: &[crate::PlanStep]) -> Result<()> {
        anyhow::bail!("plan persistence unavailable")
    }
}

impl crate::SessionSink for PromptAnchoringSession {
    fn record(&mut self, messages: &[hi_ai::Message], _: hi_ai::Usage) -> Result<()> {
        if messages
            .iter()
            .any(|message| message.role == hi_ai::Role::User)
        {
            self.events.lock().unwrap().push("prompt".into());
            if self.fail_record {
                anyhow::bail!("prompt anchor unavailable");
            }
        }
        Ok(())
    }

    fn requires_local_workspace_execution_stage(&self) -> bool {
        true
    }

    fn stage_workspace_execution(&mut self, _: &crate::WorkspaceTranscriptExecution) -> Result<()> {
        self.events.lock().unwrap().push("stage".into());
        Ok(())
    }

    fn settle_local_workspace_execution(&mut self, _: &hi_workspace::OperationId) -> Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> Result<()> {
        Ok(())
    }
}

#[async_trait::async_trait]
impl crate::WorkspaceDurability for OrderedDurability {
    async fn mutation_started(&self, _: Option<Vec<String>>) -> Result<()> {
        Ok(())
    }

    async fn checkpoint(&self) -> Result<()> {
        self.events.lock().unwrap().push("settle".into());
        if self.fail {
            anyhow::bail!("settlement acknowledgement lost");
        }
        Ok(())
    }
}

fn ordinary_batch_subject(
    fail_stage: bool,
    fail_settlement: bool,
    fail_settlement_marker: bool,
) -> (crate::Agent, Arc<Mutex<Vec<String>>>) {
    let mut cfg = config();
    // Keep this boundary fixture to one model-authored tool: the deterministic
    // implementation preflight has its own terminal-settlement tests.
    cfg.loop_limits.max_tool_calls = 1;
    cfg.gates.allow_unverified = true;
    let mut subject = agent(
        vec![
            write_completion("published.txt"),
            completion(vec![hi_ai::Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    let events = Arc::new(Mutex::new(Vec::new()));
    subject.set_session(Box::new(StageSession {
        records: Arc::new(Mutex::new(Vec::new())),
        fail: fail_stage,
        fail_settlement_marker,
        events: Some(events.clone()),
    }));
    subject.set_workspace_durability(Some(Arc::new(OrderedDurability {
        events: events.clone(),
        fail: fail_settlement,
    })));
    (subject, events)
}

#[tokio::test]
async fn ordinary_mutation_terminal_is_published_after_stage_and_settlement() {
    let (mut subject, events) = ordinary_batch_subject(false, false, false);
    let mut ui = RecordingUi {
        terminal_events: Some(events.clone()),
        ..RecordingUi::default()
    };

    subject.run_turn("write the file", &mut ui).await.unwrap();

    assert_eq!(ui.tool_results.len(), 1);
    assert_eq!(ui.tool_results[0].0, "w");
    assert_eq!(ui.tool_results[0].3, hi_tools::ToolStatus::Succeeded);
    let events = events.lock().unwrap();
    assert_eq!(&events[..3], ["stage", "settle", "terminal:w:Succeeded"]);
}

#[cfg(unix)]
#[tokio::test]
async fn pre_execution_batch_failure_settles_admission_and_reopens_workspace() {
    let mut cfg = config();
    cfg.loop_limits.max_tool_calls = 1;
    cfg.gates.allow_unverified = true;
    let target = cfg.paths.workspace_root.join("must-not-run.txt");
    let socket = std::os::unix::net::UnixListener::bind(
        cfg.paths.workspace_root.join("snapshot-error.sock"),
    )
    .unwrap();
    let mut subject = agent(vec![write_completion("must-not-run.txt")], cfg);
    let records = Arc::new(Mutex::new(Vec::new()));
    let events = Arc::new(Mutex::new(Vec::new()));
    subject.set_session(Box::new(StageSession {
        records: records.clone(),
        fail: false,
        fail_settlement_marker: false,
        events: Some(events.clone()),
    }));
    subject.set_workspace_durability(Some(Arc::new(OrderedDurability {
        events: events.clone(),
        fail: false,
    })));

    let error = subject
        .run_turn("write the file", &mut RecordingUi::default())
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("special workspace entry"));
    assert!(
        format!("{error:#}").contains("workspace admission was settled"),
        "{error:#}"
    );
    assert!(
        !target.exists(),
        "the write crossed the failed snapshot gate"
    );
    let status = subject.workspace_controller_status();
    assert_eq!(status.state, hi_workspace::WorkspaceState::Ready);
    assert!(status.active_operation.is_none());
    {
        let records = records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0].calls.is_empty());
        assert_eq!(
            records[0].execution.disposition,
            ExecutionDisposition::Failed
        );
    }
    assert_eq!(&*events.lock().unwrap(), &["stage", "settle"]);

    drop(socket);
    subject
        .begin_durable_workspace_mutation(None)
        .await
        .unwrap();
    subject.checkpoint_durable_workspace().await.unwrap();
}

#[tokio::test]
async fn post_execution_batch_failure_is_settled_by_failed_turn_cleanup() {
    let mut cfg = config();
    cfg.loop_limits.max_tool_calls = 2;
    cfg.gates.allow_unverified = true;
    cfg.gates.proactive_verify = false;
    let target = cfg.paths.workspace_root.join("partial.txt");
    let response = completion(
        vec![
            hi_ai::Content::ToolCall {
                id: "write-before-error".into(),
                name: "write".into(),
                arguments: serde_json::json!({
                    "path": "partial.txt",
                    "content": "applied before persistence failed"
                })
                .to_string(),
            },
            hi_ai::Content::ToolCall {
                id: "plan-error".into(),
                name: "update_plan".into(),
                arguments: serde_json::json!({
                    "steps": [{"title": "finish", "status": "pending"}]
                })
                .to_string(),
            },
        ],
        1,
        1,
    );
    let mut subject = agent(vec![response], cfg);
    subject.set_session(Box::new(FailPlanSession));

    let error = subject
        .run_turn(
            "write and record the remaining plan",
            &mut RecordingUi::default(),
        )
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("plan persistence unavailable"));
    assert_eq!(
        std::fs::read_to_string(target).unwrap(),
        "applied before persistence failed"
    );
    let unsettled = subject.workspace_controller_status();
    assert_eq!(unsettled.state, hi_workspace::WorkspaceState::Mutating);
    assert!(unsettled.active_operation.is_some());

    let cleanup = subject
        .cleanup_turn(crate::TurnCleanupKind::Fail)
        .await
        .unwrap();
    assert_eq!(cleanup.outcome.status, crate::TurnStatus::Failed);
    let settled = subject.workspace_controller_status();
    assert_eq!(settled.state, hi_workspace::WorkspaceState::Ready);
    assert!(settled.active_operation.is_none());
}

#[tokio::test]
async fn batch_error_funnel_never_settles_a_preexisting_operation() {
    let mut cfg = config();
    cfg.loop_limits.max_tool_calls = 1;
    cfg.gates.allow_unverified = true;
    let mut subject = agent(vec![write_completion("must-not-run.txt")], cfg);
    subject
        .begin_durable_workspace_mutation(None)
        .await
        .unwrap();
    let before = subject.workspace_controller_status();

    let error = subject
        .run_turn("write the file", &mut RecordingUi::default())
        .await
        .unwrap_err();

    assert!(
        format!("{error:#}").contains("already admitted and awaiting settlement"),
        "{error:#}"
    );
    let after = subject.workspace_controller_status();
    assert_eq!(after.state, hi_workspace::WorkspaceState::Mutating);
    assert_eq!(after.active_operation, before.active_operation);

    subject
        .workspace_coordination
        .settle_active(
            None,
            hi_workspace::ExecutionReport {
                disposition: ExecutionDisposition::Failed,
                workspace_may_have_changed: false,
                external_effect_may_have_occurred: false,
                content_digest: None,
                changed_paths: Vec::new(),
                artifacts: Vec::new(),
                detail: Some("test cleanup".into()),
            },
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn ephemeral_local_stage_has_a_durable_prompt_anchor_before_execution() {
    let mut cfg = config();
    assert_eq!(cfg.execution, crate::ExecutionMode::Ephemeral);
    cfg.loop_limits.max_tool_calls = 1;
    cfg.gates.allow_unverified = true;
    let mut subject = agent(
        vec![
            write_completion("anchored.txt"),
            completion(vec![hi_ai::Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    let events = Arc::new(Mutex::new(Vec::new()));
    subject.set_session(Box::new(PromptAnchoringSession {
        events: events.clone(),
        fail_record: false,
    }));
    subject.set_workspace_durability(Some(Arc::new(OrderedDurability {
        events: events.clone(),
        fail: false,
    })));

    subject
        .run_turn("write the anchored file", &mut RecordingUi::default())
        .await
        .unwrap();

    let events = events.lock().unwrap();
    let prompt = events.iter().position(|event| event == "prompt").unwrap();
    let stage = events.iter().position(|event| event == "stage").unwrap();
    assert!(
        prompt < stage,
        "the exact workspace result was staged before its prompt: {events:?}"
    );
}

#[tokio::test]
async fn opted_in_custom_sink_fails_before_workspace_execution_without_an_anchor() {
    let mut cfg = config();
    cfg.loop_limits.max_tool_calls = 1;
    cfg.gates.allow_unverified = true;
    let changed = cfg.paths.workspace_root.join("must-not-run.txt");
    let mut subject = agent(vec![write_completion("must-not-run.txt")], cfg);
    let events = Arc::new(Mutex::new(Vec::new()));
    subject.set_session(Box::new(PromptAnchoringSession {
        events: events.clone(),
        fail_record: true,
    }));

    let error = subject
        .run_turn(
            "do not execute without an anchor",
            &mut RecordingUi::default(),
        )
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("prompt anchor unavailable"));
    assert!(!changed.exists());
    assert_eq!(*events.lock().unwrap(), vec!["prompt"]);
}

#[tokio::test]
async fn ordinary_mutation_stage_failure_closes_terminal_once_as_indeterminate() {
    let (mut subject, events) = ordinary_batch_subject(true, false, false);
    let mut ui = RecordingUi {
        terminal_events: Some(events.clone()),
        ..RecordingUi::default()
    };

    let error = subject
        .run_turn("write the file", &mut ui)
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("commit stage unavailable"));
    assert_eq!(ui.tool_results.len(), 1);
    assert_eq!(ui.tool_results[0].0, "w");
    assert_eq!(ui.tool_results[0].3, hi_tools::ToolStatus::Failed);
    assert!(
        ui.tool_results[0]
            .2
            .contains("publication is indeterminate")
    );
    assert_eq!(
        *events.lock().unwrap(),
        vec!["stage", "settle", "terminal:w:Failed"]
    );
    assert_eq!(
        subject.workspace_controller_status().state,
        hi_workspace::WorkspaceState::RecoveryRequired
    );
}

#[tokio::test]
async fn ordinary_mutation_settlement_failure_closes_terminal_once_as_indeterminate() {
    let (mut subject, events) = ordinary_batch_subject(false, true, false);
    let mut ui = RecordingUi {
        terminal_events: Some(events.clone()),
        ..RecordingUi::default()
    };

    let error = subject
        .run_turn("write the file", &mut ui)
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("settlement acknowledgement lost"));
    assert_eq!(ui.tool_results.len(), 1);
    assert_eq!(ui.tool_results[0].0, "w");
    assert_eq!(ui.tool_results[0].3, hi_tools::ToolStatus::Failed);
    assert!(
        ui.tool_results[0]
            .2
            .contains("publication is indeterminate")
    );
    assert_eq!(
        *events.lock().unwrap(),
        vec!["stage", "settle", "terminal:w:Failed"]
    );
    assert_eq!(
        subject.workspace_controller_status().state,
        hi_workspace::WorkspaceState::RecoveryRequired
    );
}

#[tokio::test]
async fn local_transcript_marker_failure_cannot_publish_a_success_terminal() {
    let (mut subject, events) = ordinary_batch_subject(false, false, true);
    let mut ui = RecordingUi {
        terminal_events: Some(events.clone()),
        ..RecordingUi::default()
    };

    let error = subject
        .run_turn("write the file", &mut ui)
        .await
        .unwrap_err();

    assert!(
        format!("{error:#}").contains("transcript settlement marker unavailable"),
        "{error:#}"
    );
    assert_eq!(ui.tool_results.len(), 1);
    assert_eq!(ui.tool_results[0].3, hi_tools::ToolStatus::Failed);
    assert!(
        ui.tool_results[0]
            .2
            .contains("publication is indeterminate")
    );
    assert_eq!(
        *events.lock().unwrap(),
        vec!["stage", "settle", "terminal:w:Failed"]
    );
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

    assert!(
        format!("{error:#}").contains("looks like it contains secrets"),
        "{error:#}"
    );
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

    assert!(
        format!("{error:#}").contains("commit stage unavailable"),
        "{error:#}"
    );
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
async fn local_commit_cannot_settle_successfully_before_exact_staging() {
    let Fixture {
        mut subject,
        controller,
        records,
    } = local_subject(true);
    std::fs::write(subject.workspace_root().join("tracked.txt"), "changed\n").unwrap();

    let error = subject
        .commit_session_changes(&["tracked.txt".into()])
        .await
        .unwrap_err();

    assert!(
        format!("{error:#}").contains("commit stage unavailable"),
        "{error:#}"
    );
    assert!(records.lock().unwrap().is_empty());
    assert_eq!(
        git_stdout(subject.workspace_root(), &["log", "-1", "--pretty=%s"]).trim(),
        "update tracked.txt",
        "the external effect occurred before the local transcript stage failed"
    );
    assert_eq!(
        controller.status().state,
        hi_workspace::WorkspaceState::RecoveryRequired,
        "local settlement must fail closed when its durable transcript cannot stage"
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
