use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use hi_control::{ControlStore, OperationReplayClass, WorkspaceOperationStatus};
use hi_workspace::{ExecutionReport, MutationIntent, WorkspaceState};

use super::common::{IsolatedWorkspace, agent};

#[cfg(unix)]
struct ReapProofDurability {
    hook_pid: std::path::PathBuf,
    racing_write: std::path::PathBuf,
    observed_reaped_and_stable: Arc<AtomicBool>,
}

#[cfg(unix)]
#[async_trait::async_trait]
impl crate::WorkspaceDurability for ReapProofDurability {
    async fn mutation_started(&self, _: Option<Vec<String>>) -> anyhow::Result<()> {
        Ok(())
    }

    async fn checkpoint(&self) -> anyhow::Result<()> {
        let pid: libc::pid_t = std::fs::read_to_string(&self.hook_pid)?.trim().parse()?;
        anyhow::ensure!(
            !process_is_alive(pid),
            "workspace settlement began before hook child {pid} was reaped"
        );
        let before = std::fs::read(&self.racing_write)?;
        tokio::time::sleep(std::time::Duration::from_millis(75)).await;
        let after = std::fs::read(&self.racing_write)?;
        anyhow::ensure!(
            before == after,
            "hook descendant continued writing during workspace settlement"
        );
        self.observed_reaped_and_stable
            .store(true, Ordering::Release);
        Ok(())
    }
}

#[cfg(unix)]
fn process_is_alive(pid: libc::pid_t) -> bool {
    // SAFETY: signal 0 only probes the test-owned PID.
    unsafe { libc::kill(pid, 0) == 0 }
}

#[tokio::test]
async fn failed_hook_reconciles_partial_write_and_settles_failed_execution() {
    let workspace = IsolatedWorkspace::new("failed-hook-settlement");
    let cfg = workspace.config();
    let state_root = cfg.paths.state_root.clone();
    let mut subject = agent(Vec::new(), cfg);
    let binding = subject.workspace_controller_binding();
    let written = workspace.path("hook-ran.txt");
    let executor_path = written.clone();

    let error = subject
        .run_workspace_lifecycle_hook_for_test(
            "pre-turn",
            "test input",
            &crate::TurnCancellation::new(),
            async move {
                std::fs::write(executor_path, "hook-ran").unwrap();
                crate::session_ops::HookExecution::Completed(Err(anyhow::anyhow!(
                    "hook failed (17)"
                )))
            },
        )
        .await
        .unwrap_err();

    let message = format!("{error:#}");
    assert!(message.contains("hook failed (17)"), "{message}");
    assert_eq!(std::fs::read_to_string(written).unwrap(), "hook-ran");
    assert!(
        subject
            .last_file_changes()
            .iter()
            .any(|change| change.path == "hook-ran.txt"),
        "the hook's opaque write was not reconciled into the workspace ledger"
    );
    assert_eq!(
        subject.workspace_controller_status().state,
        WorkspaceState::Ready,
        "a known hook failure with a reconciled postimage must close admission cleanly"
    );

    let operations = ControlStore::open_for_state(&state_root)
        .unwrap()
        .operations_for_binding(binding.binding_id.as_str())
        .unwrap();
    let hook = operations.last().expect("hook operation was journaled");
    assert_eq!(hook.kind, "live_writer");
    assert_eq!(
        hook.replay_class,
        OperationReplayClass::NonReplayableExternal
    );
    assert_eq!(hook.status, WorkspaceOperationStatus::Durable);
    assert!(
        hook.error
            .as_deref()
            .is_some_and(|detail| detail.contains("lifecycle hook pre-turn failed")),
        "the durable operation did not retain the hook's failed execution detail: {hook:?}"
    );
}

#[tokio::test]
async fn closed_workspace_admission_prevents_hook_executor_start() {
    let workspace = IsolatedWorkspace::new("closed-hook-admission");
    let mut subject = agent(Vec::new(), workspace.config());
    subject
        .begin_classified_workspace_operation(MutationIntent::workspace("existing writer"))
        .await
        .unwrap();
    let starts = Arc::new(AtomicUsize::new(0));
    let executor_starts = starts.clone();
    let written = workspace.path("hook-ran.txt");
    let executor_path = written.clone();

    let error = subject
        .run_workspace_lifecycle_hook_for_test(
            "pre-turn",
            "test input",
            &crate::TurnCancellation::new(),
            async move {
                executor_starts.fetch_add(1, Ordering::SeqCst);
                std::fs::write(executor_path, "should-not-run").unwrap();
                crate::session_ops::HookExecution::Completed(Ok("hook pre-turn: ok".into()))
            },
        )
        .await
        .unwrap_err();

    let message = format!("{error:#}");
    assert!(
        message.contains("workspace controller refused"),
        "{message}"
    );
    assert_eq!(
        starts.load(Ordering::SeqCst),
        0,
        "the hook executor was polled despite closed admission"
    );
    assert!(
        !written.exists(),
        "the hook executor wrote despite closed controller admission"
    );
    assert_eq!(
        subject.workspace_controller_status().state,
        WorkspaceState::Mutating,
        "the rejected hook must not settle somebody else's active operation"
    );

    subject
        .checkpoint_durable_workspace_with_execution(ExecutionReport::succeeded(None))
        .await
        .unwrap();
    assert_eq!(
        subject.workspace_controller_status().state,
        WorkspaceState::Ready
    );
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_hook_is_reaped_and_quiescent_before_workspace_settlement() {
    use std::os::unix::fs::PermissionsExt as _;

    let workspace = IsolatedWorkspace::new("cancelled-hook-reap");
    let hooks = workspace.path(".hi/hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    let hook = hooks.join("pre-turn");
    std::fs::write(
        &hook,
        "#!/bin/sh\n\
         echo $$ > hook.pid\n\
         (n=0; while :; do printf '%s\\n' \"$n\" > racing.txt; n=$((n + 1)); done) &\n\
         echo $! > writer.pid\n\
         wait\n",
    )
    .unwrap();
    std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();

    let mut subject = agent(Vec::new(), workspace.config());
    let observed = Arc::new(AtomicBool::new(false));
    subject.set_workspace_durability(Some(Arc::new(ReapProofDurability {
        hook_pid: workspace.path("hook.pid"),
        racing_write: workspace.path("racing.txt"),
        observed_reaped_and_stable: observed.clone(),
    })));
    let cancellation = crate::TurnCancellation::new();
    let cancellation_trigger = {
        let cancellation = cancellation.clone();
        let hook_pid = workspace.path("hook.pid");
        let writer_pid = workspace.path("writer.pid");
        let racing = workspace.path("racing.txt");
        tokio::spawn(async move {
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                while !(hook_pid.is_file() && writer_pid.is_file() && racing.is_file()) {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
            })
            .await
            .expect("hook and descendant did not start");
            cancellation.cancel();
        })
    };
    let workspace_root = workspace.path("");
    let hook_execution = crate::session_ops::run_hook_process_cancellable_for_test(
        &workspace_root,
        "pre-turn",
        "input",
        &cancellation,
    );

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        subject.run_workspace_lifecycle_hook_for_test(
            "pre-turn",
            "input",
            &cancellation,
            hook_execution,
        ),
    )
    .await
    .expect("cancelled hook cleanup exceeded its bound")
    .expect("reaped cancellation should settle cleanly");
    cancellation_trigger.await.unwrap();

    assert!(output.is_none(), "cancelled hook returned success output");
    assert!(
        observed.load(Ordering::Acquire),
        "durability settlement did not observe the reaped, quiescent hook"
    );
    let hook_pid: libc::pid_t = std::fs::read_to_string(workspace.path("hook.pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(
        !process_is_alive(hook_pid),
        "direct hook child remained alive after cancelled settlement"
    );
    assert_eq!(
        subject.workspace_controller_status().state,
        WorkspaceState::Ready
    );
}
