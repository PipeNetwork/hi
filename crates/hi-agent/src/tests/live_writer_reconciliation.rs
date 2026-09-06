use std::sync::{Arc, Mutex};

use hi_ai::Content;
use hi_workspace::{
    EffectScope, JobCompletion, JobId, JobKind, JobLimits, JobSpec, JobTerminal,
    WorkspaceController, WorkspaceState,
};

use super::common::{Canned, IsolatedWorkspace, RecUi, completion, config};

struct PipeTranscriptSession;

struct PipeTestDurability;

#[async_trait::async_trait]
impl crate::WorkspaceDurability for PipeTestDurability {
    async fn mutation_started(&self, _: Option<Vec<String>>) -> anyhow::Result<()> {
        Ok(())
    }

    async fn checkpoint(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn stage_workspace_execution(
        &self,
        _: &crate::WorkspaceTranscriptExecution,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

impl crate::SessionSink for PipeTranscriptSession {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn stage_workspace_execution(
        &mut self,
        _: &crate::WorkspaceTranscriptExecution,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> anyhow::Result<()> {
        Ok(())
    }
}

fn call(id: &str, name: &str, arguments: serde_json::Value) -> hi_ai::Completion {
    completion(
        vec![Content::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.to_string(),
        }],
        1,
        1,
    )
}

struct FailingTerminalLifecycle {
    controller: Arc<dyn WorkspaceController>,
    job: Mutex<Option<JobId>>,
}

#[async_trait::async_trait]
impl hi_tools::BackgroundJobLifecycle for FailingTerminalLifecycle {
    async fn register(
        &self,
        registration: hi_tools::BackgroundJobRegistration,
    ) -> Result<(), String> {
        let parent_operation = self.controller.status().active_operation;
        let permit = self
            .controller
            .register_job(JobSpec {
                kind: JobKind::Process,
                effect_scope: EffectScope::LiveWriter,
                name: registration.name,
                limits: JobLimits::default(),
                parent_operation,
            })
            .await
            .map_err(|error| error.to_string())?;
        *self.job.lock().unwrap() = Some(permit.job_id);
        Ok(())
    }

    async fn observe_terminal(
        &self,
        _: &hi_tools::BackgroundJobId,
        _: hi_tools::BackgroundJobTerminal,
        _: Option<String>,
    ) -> Result<hi_tools::BackgroundJobPublication, String> {
        Err("injected terminal journal failure".into())
    }

    async fn pending(&self, _: &str) -> Vec<hi_tools::BackgroundJobId> {
        Vec::new()
    }

    async fn settle_after_workspace(&self, _: &[hi_tools::BackgroundJobId]) -> Result<(), String> {
        Ok(())
    }
}

#[tokio::test]
async fn terminal_bash_output_settles_live_writer_before_publication() {
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "spawn",
            "bash",
            serde_json::json!({
                "command": "sleep 600",
                "run_in_background": true,
            }),
        ),
        completion(
            vec![Content::Text("Background process started.".into())],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider.clone(), config()).unwrap();
    let mut ui = RecUi::default();

    subject.run_turn("start a service", &mut ui).await.unwrap();

    let ids = subject.runtime.background().ids();
    assert_eq!(ids.len(), 1);
    assert_eq!(subject.workspace_controller_status().active_jobs.len(), 1);
    subject
        .runtime
        .background()
        .kill_and_reap(&ids[0])
        .await
        .unwrap();
    assert_eq!(
        subject.workspace_controller_status().active_jobs.len(),
        1,
        "reaping alone must leave the writer durability-pending"
    );

    provider.0.lock().unwrap().extend([
        call("poll", "bash_output", serde_json::json!({ "id": ids[0] })),
        completion(vec![Content::Text("The process was stopped.".into())], 1, 1),
    ]);
    subject
        .run_turn("check the service", &mut ui)
        .await
        .unwrap();

    let status = subject.workspace_controller_status();
    assert_eq!(status.state, WorkspaceState::Ready);
    assert!(
        status.active_jobs.is_empty(),
        "the terminal poll must not publish before its workspace receipt seals the job"
    );
    assert!(
        ui.tool_results
            .iter()
            .any(|(name, result)| name == "bash_output"
                && result.to_ascii_lowercase().contains("stopped")),
        "tool results: {:?}",
        ui.tool_results
    );
}

#[tokio::test]
async fn same_turn_local_service_allows_other_tools_and_preserves_the_running_service() {
    let config = config();
    let allowed_path = config.paths.workspace_root.join("must-run.txt");
    let evidence_path = config.paths.workspace_root.join("service-access.txt");
    std::fs::write(&evidence_path, "service access marker").unwrap();
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "spawn",
            "bash",
            serde_json::json!({
                "command": "sleep 600",
                "run_in_background": true,
            }),
        ),
        completion(
            vec![
                Content::ToolCall {
                    id: "allowed-write".into(),
                    name: "write".into(),
                    arguments: serde_json::json!({
                        "path": allowed_path,
                        "content": "executed while service remained live",
                    })
                    .to_string(),
                },
                Content::ToolCall {
                    id: "safe-read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": evidence_path }).to_string(),
                },
                Content::ToolCall {
                    id: "safe-poll".into(),
                    name: "bash_output".into(),
                    arguments: serde_json::json!({
                        "id": "sleep_1",
                        "wait_secs": 0,
                    })
                    .to_string(),
                },
            ],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The service is running; use the URL reported above.".into(),
            )],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider, config).unwrap();
    let mut ui = RecUi::default();

    let outcome = subject
        .run_turn("start a service and report how to access it", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, crate::TurnStatus::Completed);
    assert!(!outcome.stop_reason.is_workspace_admission());
    assert_eq!(
        std::fs::read_to_string(&allowed_path).unwrap(),
        "executed while service remained live"
    );
    let mutation_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "write")
        .collect::<Vec<_>>();
    assert_eq!(
        mutation_results.len(),
        1,
        "the local mutation must receive exactly one terminal result: {:?}",
        ui.tool_results
    );
    assert!(!mutation_results[0].1.contains("workspace_admission_denied"));
    assert!(!mutation_results[0].1.contains("tool_protocol_error"));
    assert!(
        ui.tool_results
            .iter()
            .any(|(name, result)| { name == "read" && result.contains("service access marker") })
    );
    assert!(ui.tool_results.iter().any(|(name, result)| {
        name == "bash_output" && result.to_ascii_lowercase().contains("running")
    }));
    let protocol_repair_markers = [
        "sealed envelope",
        "schema-corrected",
        "invalid tool arguments",
        "strict schemas",
        "workspace changed after the request was sealed",
        "structured tool arguments kept failing validation",
    ];
    assert!(
        !ui.statuses.iter().any(|status| {
            let status = status.to_ascii_lowercase();
            protocol_repair_markers
                .iter()
                .any(|marker| status.contains(marker))
        }),
        "ordinary workspace admission denial must not enter protocol repair: {:?}",
        ui.statuses
    );
    let ids = subject.runtime.background().ids();
    assert_eq!(ids.len(), 1);
    assert_eq!(
        subject.runtime.background().outcome(&ids[0]).unwrap().state,
        hi_tools::BackgroundState::Running
    );
    let status = subject.workspace_controller_status();
    assert_eq!(status.state, WorkspaceState::Ready);
    assert_eq!(status.active_jobs.len(), 1);

    subject.settle_workspace_for_exit().await.unwrap();
}

#[tokio::test]
async fn foreground_overrun_hands_off_and_does_not_hold_workspace_admission() {
    let config = config();
    let allowed_path = config.paths.workspace_root.join("after-handoff.txt");
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "overrun",
            "bash",
            serde_json::json!({
                "command": "sleep 600",
            }),
        ),
        call(
            "write-after-handoff",
            "write",
            serde_json::json!({
                "path": allowed_path,
                "content": "workspace admission reopened",
            }),
        ),
        completion(
            vec![Content::Text(
                "The foreground overrun was handed off and work continued.".into(),
            )],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider, config).unwrap();
    subject
        .runtime
        .background()
        .set_foreground_handoff_budget(Some(std::time::Duration::from_secs(1)));
    let mut ui = RecUi::default();

    let outcome = subject
        .run_turn("run the command, then continue editing", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, crate::TurnStatus::Completed);
    assert_eq!(
        std::fs::read_to_string(&allowed_path).unwrap(),
        "workspace admission reopened"
    );
    assert!(ui.tool_results.iter().any(|(name, result)| {
        name == "bash"
            && (result.contains("continued as") || result.contains("still running after"))
    }));
    let status = subject.workspace_controller_status();
    assert_eq!(status.state, WorkspaceState::Ready);
    assert!(status.active_operation.is_none());
    subject.settle_workspace_for_exit().await.unwrap();
}

#[tokio::test]
async fn pipefs_without_background_writers_keeps_long_mutation_foreground() {
    let config = config();
    let workspace_root = config.paths.workspace_root.clone();
    let state_root = config.paths.state_root.clone();
    let output_path = workspace_root.join("pipefs-foreground-finished.txt");
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "pipefs-long-writer",
            "bash",
            serde_json::json!({
                "command": "sleep 0.08; printf finished > pipefs-foreground-finished.txt",
            }),
        ),
        completion(
            vec![Content::Text(
                "The PipeFS foreground command completed.".into(),
            )],
            1,
            1,
        ),
    ])));
    let controller = Arc::new(hi_workspace::InMemoryWorkspaceController::new_pipefs(
        "pipefs-foreground-workspace",
        "pipefs-foreground-session",
        1,
        false,
        &workspace_root,
        &state_root,
    ));
    assert!(!controller.capabilities().background_writers);
    let mut subject = crate::Agent::new(provider, config).unwrap();
    subject
        .install_workspace_controller(controller.clone())
        .unwrap();
    subject.set_workspace_durability(Some(Arc::new(PipeTestDurability)));
    subject.set_session(Box::new(PipeTranscriptSession));
    subject
        .runtime
        .background()
        .set_foreground_handoff_budget(Some(std::time::Duration::from_millis(10)));
    let mut ui = RecUi::default();
    let started = std::time::Instant::now();

    let outcome = subject
        .run_turn("run the PipeFS build to completion", &mut ui)
        .await
        .unwrap();

    assert!(started.elapsed() >= std::time::Duration::from_millis(60));
    assert_eq!(outcome.status, crate::TurnStatus::Completed);
    assert_eq!(std::fs::read_to_string(output_path).unwrap(), "finished");
    assert!(ui.tool_results.iter().any(|(name, result)| {
        name == "bash" && result.contains("[no output]") && !result.contains("continued as")
    }));
    assert!(subject.runtime.background().ids().is_empty());
    let status = controller.status();
    assert_eq!(status.state, WorkspaceState::Ready);
    assert!(status.active_operation.is_none());
    assert!(status.active_jobs.is_empty());
}

#[tokio::test]
async fn later_local_foreground_mutation_runs_while_service_is_live() {
    let config = config();
    let allowed_path = config.paths.workspace_root.join("must-run.txt");
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "spawn",
            "bash",
            serde_json::json!({
                "command": "sleep 600",
                "run_in_background": true,
            }),
        ),
        completion(
            vec![Content::Text("Background process started.".into())],
            1,
            1,
        ),
        call(
            "allowed-write",
            "write",
            serde_json::json!({
                "path": allowed_path,
                "content": "local edit completed",
            }),
        ),
        completion(
            vec![Content::Text(
                "The local edit completed while the service stayed up.".into(),
            )],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider, config).unwrap();
    let mut ui = RecUi::default();

    subject.run_turn("start a service", &mut ui).await.unwrap();
    let outcome = subject
        .run_turn("write another file", &mut ui)
        .await
        .unwrap();
    assert_eq!(outcome.status, crate::TurnStatus::Completed);
    assert!(!outcome.stop_reason.is_workspace_admission());
    assert_eq!(
        std::fs::read_to_string(&allowed_path).unwrap(),
        "local edit completed"
    );
    assert_eq!(subject.workspace_controller_status().active_jobs.len(), 1);
    subject.settle_workspace_for_exit().await.unwrap();
    assert_eq!(
        subject.workspace_controller_status().state,
        WorkspaceState::Ready
    );
    assert!(subject.workspace_controller_status().active_jobs.is_empty());
}

#[tokio::test]
async fn bash_kill_reaps_then_reconciles_the_existing_writer_job() {
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "spawn",
            "bash",
            serde_json::json!({
                "command": "sleep 600",
                "run_in_background": true,
            }),
        ),
        completion(
            vec![Content::Text("Background process started.".into())],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider.clone(), config()).unwrap();
    let mut ui = RecUi::default();
    subject.run_turn("start a service", &mut ui).await.unwrap();

    let ids = subject.runtime.background().ids();
    assert_eq!(ids.len(), 1);
    assert_eq!(subject.workspace_controller_status().active_jobs.len(), 1);
    provider.0.lock().unwrap().extend([
        call("kill", "bash_kill", serde_json::json!({ "id": ids[0] })),
        completion(vec![Content::Text("The process was stopped.".into())], 1, 1),
    ]);

    subject.run_turn("stop the service", &mut ui).await.unwrap();

    let status = subject.workspace_controller_status();
    assert_eq!(status.state, WorkspaceState::Ready);
    assert!(
        status.active_jobs.is_empty(),
        "kill must reap first, then reconcile and seal the existing writer job"
    );
    assert!(ui.tool_results.iter().any(|(name, result)| {
        name == "bash_kill" && result.to_ascii_lowercase().contains("stopped")
    }));
}

#[tokio::test]
async fn orderly_exit_reaps_processes_settles_tasks_and_passes_the_barrier() {
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "spawn",
            "bash",
            serde_json::json!({
                "command": "printf shutdown > shutdown-writer.txt; sleep 600",
                "run_in_background": true,
            }),
        ),
        completion(
            vec![Content::Text("Background process started.".into())],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider, config()).unwrap();
    let mut ui = RecUi::default();
    subject.run_turn("start a writer", &mut ui).await.unwrap();

    let tasks = subject.background_task_registry();
    let task_id = tasks
        .spawn(
            "pending reader",
            "explore",
            Box::new(|| Box::pin(std::future::pending::<hi_tools::BackgroundTaskOutcome>())),
        )
        .await
        .unwrap();
    assert_eq!(subject.workspace_controller_status().active_jobs.len(), 2);

    let receipt = subject.settle_workspace_for_exit().await.unwrap();

    assert_eq!(receipt.status, hi_workspace::BarrierStatus::Passed);
    assert!(subject.active_background_process_ids().is_empty());
    assert!(subject.active_background_task_ids().await.is_empty());
    assert!(subject.workspace_controller_status().active_jobs.is_empty());
    assert_eq!(
        tasks
            .poll(&task_id, std::time::Duration::ZERO)
            .await
            .unwrap()
            .state,
        hi_tools::BackgroundTaskState::Cancelled
    );
}

#[tokio::test]
async fn keep_background_settles_a_requested_writer_that_exits_before_handoff() {
    let workspace = IsolatedWorkspace::new("keep-background-exit-race");
    let config = workspace.config();
    let state_root = config.paths.state_root.clone();
    let workspace_root = config.paths.workspace_root.clone();
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "spawn",
            "bash",
            serde_json::json!({
                "command": "sleep 1; printf complete > requested-exit.txt",
                "run_in_background": true,
            }),
        ),
        completion(
            vec![Content::Text("Background process started.".into())],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider, config).unwrap();
    let mut ui = RecUi::default();
    subject
        .run_turn("start a short-lived writer", &mut ui)
        .await
        .unwrap();
    let id = subject.runtime.background().ids().pop().unwrap();

    tokio::time::timeout(std::time::Duration::from_secs(4), async {
        while subject.runtime.background().outcome(&id).unwrap().state
            == hi_tools::BackgroundState::Running
        {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        subject
            .runtime
            .background()
            .pending_job_settlements()
            .await
            .len(),
        1,
        "completion must win the race and remain durability-pending"
    );

    subject.release_background_services().await.unwrap();
    let status = subject.workspace_controller_status();
    assert_eq!(status.state, WorkspaceState::Ready);
    assert!(status.active_jobs.is_empty());
    drop(subject);

    let restarted = crate::workspace_coordination::WorkspaceCoordination::new_local(
        &workspace_root,
        &state_root,
    );
    assert_eq!(restarted.status().state, WorkspaceState::Ready);
    assert!(restarted.status().recovery_id.is_none());
    assert!(restarted.status().active_jobs.is_empty());
}

#[tokio::test]
async fn keep_background_rejects_a_reaped_writer_with_a_failed_terminal_callback() {
    let provider = Arc::new(Canned(Mutex::new(vec![
        call(
            "spawn",
            "bash",
            serde_json::json!({
                "command": "sleep 0.05",
                "run_in_background": true,
            }),
        ),
        completion(
            vec![Content::Text("Background process started.".into())],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider, config()).unwrap();
    let controller = subject.workspace_coordination.job_controller();
    let lifecycle = Arc::new(FailingTerminalLifecycle {
        controller: controller.clone(),
        job: Mutex::new(None),
    });
    subject
        .runtime
        .background()
        .set_job_lifecycle(lifecycle.clone());
    let mut ui = RecUi::default();

    subject
        .run_turn("start a short-lived writer", &mut ui)
        .await
        .unwrap();
    let id = subject.runtime.background().ids().pop().unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while subject.runtime.background().outcome(&id).unwrap().state
            == hi_tools::BackgroundState::Running
        {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        subject.runtime.background().outcome(&id).unwrap().state,
        hi_tools::BackgroundState::Failed
    );

    let job = lifecycle.job.lock().unwrap().clone().unwrap();
    assert_eq!(
        subject.workspace_controller_status().active_jobs.as_slice(),
        std::slice::from_ref(&job)
    );
    let error = subject.release_background_services().await.unwrap_err();
    assert!(format!("{error:#}").contains(job.as_str()), "{error:#}");

    controller
        .seal_job(
            job,
            JobTerminal {
                completion: JobCompletion::Failed,
                detail: Some("test cleanup after injected callback failure".into()),
                artifacts: Vec::new(),
            },
        )
        .await;
    subject.release_background_services().await.unwrap();
}
