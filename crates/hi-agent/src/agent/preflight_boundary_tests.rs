use std::sync::{Arc, Mutex};

use hi_workspace::{InMemoryWorkspaceController, WorkspaceController, WorkspaceState};

use super::*;
use crate::agent::turn::retention::ToolTimeline;
use crate::tests::common::{IsolatedWorkspace, NullUi, RecordingUi, agent};

struct FailingDurability;

#[async_trait::async_trait]
impl crate::WorkspaceDurability for FailingDurability {
    async fn mutation_started(&self, _: Option<Vec<String>>) -> Result<()> {
        Ok(())
    }

    async fn checkpoint(&self) -> Result<()> {
        anyhow::bail!("preflight settlement acknowledgement lost")
    }
}

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
            anyhow::bail!("preflight stage unavailable");
        }
        self.records.lock().unwrap().push(record.clone());
        Ok(())
    }

    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> Result<()> {
        Ok(())
    }
}

fn pipefs_subject(
    fail_stage: bool,
) -> (
    IsolatedWorkspace,
    crate::Agent,
    Arc<InMemoryWorkspaceController>,
    Arc<Mutex<Vec<crate::WorkspaceTranscriptExecution>>>,
) {
    let workspace = IsolatedWorkspace::new("implementation-preflight-boundary");
    std::fs::write(workspace.path("Cargo.toml"), "[workspace]\n").unwrap();
    let config = workspace.config();
    let workspace_root = config.paths.workspace_root.clone();
    let state_root = config.paths.state_root.clone();
    let mut subject = agent(Vec::new(), config);
    // Protocol 1 deliberately exercises the compatibility path. This
    // harness-owned PureWorkspace operation must not require a causal
    // non-replayable intent acknowledgement.
    let controller = Arc::new(InMemoryWorkspaceController::new_pipefs(
        "preflight-workspace",
        "preflight-session",
        1,
        false,
        workspace_root,
        state_root,
    ));
    subject
        .install_workspace_controller(controller.clone())
        .unwrap();
    let records = Arc::new(Mutex::new(Vec::new()));
    subject.set_session(Box::new(StageSession {
        records: records.clone(),
        fail: fail_stage,
    }));
    (workspace, subject, controller, records)
}

#[test]
fn fixed_preflight_disables_git_helpers_and_is_replayable() {
    let command = implementation_preflight_command();
    assert!(command.contains("--no-pager --no-optional-locks"));
    assert!(command.contains("-c core.fsmonitor=false"));
    assert!(command.contains("-c core.untrackedCache=false"));
    assert!(command.contains("diff --no-ext-diff --no-textconv --stat"));
    let intent = implementation_preflight_intent();
    assert_eq!(intent.effect_scope, hi_workspace::EffectScope::LiveWriter);
    assert_eq!(
        intent.replay_class,
        hi_workspace::ReplayClass::PureWorkspace
    );
}

#[tokio::test]
async fn protocol_one_pipefs_stages_and_settles_before_publishing_preflight() {
    let (_workspace, mut subject, controller, records) = pipefs_subject(false);
    let before_messages = subject.messages().len();
    let mut tracker = ImplementationTracker::default();
    let mut timeline = ToolTimeline::default();

    let summary = subject
        .run_implementation_preflight(&mut NullUi, &mut tracker, &mut timeline)
        .await
        .unwrap();

    assert_eq!(summary.executed, 1);
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].calls[0].name, "bash");
    assert_eq!(
        records[0].execution.disposition,
        hi_workspace::ExecutionDisposition::Succeeded
    );
    assert!(!records[0].execution.external_effect_may_have_occurred);
    assert!(subject.messages().len() > before_messages);
}

#[tokio::test]
async fn staging_failure_never_publishes_preflight_success() {
    let (_workspace, mut subject, controller, records) = pipefs_subject(true);
    let before_messages = subject.messages().len();
    let mut tracker = ImplementationTracker::default();
    let mut timeline = ToolTimeline::default();
    let mut ui = RecordingUi::default();

    let error = subject
        .run_implementation_preflight(&mut ui, &mut tracker, &mut timeline)
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("preflight stage unavailable"));
    assert!(records.lock().unwrap().is_empty());
    assert_eq!(subject.messages().len(), before_messages);
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
    assert_eq!(ui.tool_starts.len(), 1);
    assert_eq!(ui.tool_calls.len(), 1);
    assert_eq!(ui.tool_results.len(), 1);
    assert_eq!(ui.tool_starts[0].0, ui.tool_calls[0].0);
    assert_eq!(ui.tool_calls[0].0, ui.tool_results[0].0);
    assert_eq!(ui.tool_results[0].3, hi_tools::ToolStatus::Failed);
    assert!(ui.tool_results[0].2.contains("could not be staged"));
    assert!(ui.tool_results[0].2.contains("indeterminate"));
}

#[tokio::test]
async fn settlement_failure_closes_preflight_ui_lifecycle_exactly_once() {
    let workspace = IsolatedWorkspace::new("implementation-preflight-settlement-failure");
    std::fs::write(workspace.path("Cargo.toml"), "[workspace]\n").unwrap();
    let mut subject = agent(Vec::new(), workspace.config());
    subject.set_workspace_durability(Some(Arc::new(FailingDurability)));
    let before_messages = subject.messages().len();
    let mut tracker = ImplementationTracker::default();
    let mut timeline = ToolTimeline::default();
    let mut ui = RecordingUi::default();

    let error = subject
        .run_implementation_preflight(&mut ui, &mut tracker, &mut timeline)
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("preflight settlement acknowledgement lost"));
    assert_eq!(subject.messages().len(), before_messages);
    assert_eq!(
        subject.workspace_controller_status().state,
        WorkspaceState::RecoveryRequired
    );
    assert_eq!(ui.tool_starts.len(), 1);
    assert_eq!(ui.tool_calls.len(), 1);
    assert_eq!(ui.tool_results.len(), 1);
    assert_eq!(ui.tool_starts[0].0, ui.tool_calls[0].0);
    assert_eq!(ui.tool_calls[0].0, ui.tool_results[0].0);
    assert_eq!(ui.tool_results[0].3, hi_tools::ToolStatus::Failed);
    assert!(ui.tool_results[0].2.contains("settlement is indeterminate"));
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_flight_interrupt_kills_and_reaps_preflight_process_group() {
    let workspace = IsolatedWorkspace::new("implementation-preflight-reap");
    let subject = agent(Vec::new(), workspace.config());
    let runner = subject.runtime.process_runner().clone();
    let foreground = runner.foreground_registry();
    let interrupt = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let raise_interrupt = {
        let foreground = foreground.clone();
        let interrupt = interrupt.clone();
        tokio::spawn(async move {
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                while foreground.active_count() == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("preflight process never registered");
            interrupt.store(true, std::sync::atomic::Ordering::Release);
        })
    };

    let (process, interrupted) = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        execute_implementation_preflight_process(&runner, "sleep 600", interrupt),
    )
    .await
    .expect("preflight interruption did not reap promptly");
    raise_interrupt.await.unwrap();
    let outcome = implementation_preflight_outcome(process, interrupted);

    assert!(interrupted);
    assert_eq!(outcome.status, hi_tools::ToolStatus::Cancelled);
    assert_eq!(foreground.active_count(), 0);
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupt_between_prestart_check_and_first_poll_cannot_launch_an_unowned_process() {
    let workspace = IsolatedWorkspace::new("implementation-preflight-start-race");
    let subject = agent(Vec::new(), workspace.config());
    let runner = subject.runtime.process_runner().clone();
    let foreground = runner.foreground_registry();
    let interrupt = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let raise_interrupt = interrupt.clone();

    let (process, interrupted) = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        execute_implementation_preflight_process_after_check(
            &runner,
            "sleep 600",
            interrupt,
            move || raise_interrupt.store(true, std::sync::atomic::Ordering::Release),
        ),
    )
    .await
    .expect("racing preflight interruption left an unowned process");
    let outcome = implementation_preflight_outcome(process, interrupted);

    assert!(interrupted);
    assert_eq!(outcome.status, hi_tools::ToolStatus::Cancelled);
    assert_eq!(foreground.active_count(), 0);
}
