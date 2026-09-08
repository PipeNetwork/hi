use super::*;
use std::time::Duration;

struct SettlementGate {
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
    observed: Mutex<Vec<crate::BackgroundJobTerminal>>,
}

#[derive(Default)]
struct SelectiveFailureLifecycle {
    fail_handle: Mutex<Option<String>>,
    observed: Mutex<Vec<(String, crate::BackgroundJobTerminal)>>,
}

#[derive(Default)]
struct PendingWriterLifecycle {
    observed: Mutex<Vec<crate::BackgroundJobTerminal>>,
    pending: Mutex<Vec<crate::BackgroundJobId>>,
    settled: Mutex<Vec<crate::BackgroundJobId>>,
}

#[async_trait::async_trait]
impl crate::BackgroundJobLifecycle for PendingWriterLifecycle {
    async fn register(&self, _: crate::BackgroundJobRegistration) -> Result<(), String> {
        Ok(())
    }

    async fn observe_terminal(
        &self,
        id: &crate::BackgroundJobId,
        terminal: crate::BackgroundJobTerminal,
        _: Option<String>,
    ) -> Result<crate::BackgroundJobPublication, String> {
        self.observed.lock().unwrap().push(terminal);
        self.pending.lock().unwrap().push(id.clone());
        Ok(crate::BackgroundJobPublication::DurabilityPending)
    }

    async fn pending(&self, source_id: &str) -> Vec<crate::BackgroundJobId> {
        self.pending
            .lock()
            .unwrap()
            .iter()
            .filter(|id| id.source_id == source_id)
            .cloned()
            .collect()
    }

    async fn settle_after_workspace(
        &self,
        settled: &[crate::BackgroundJobId],
    ) -> Result<(), String> {
        self.pending
            .lock()
            .unwrap()
            .retain(|id| !settled.contains(id));
        self.settled.lock().unwrap().extend_from_slice(settled);
        Ok(())
    }
}

#[async_trait::async_trait]
impl crate::BackgroundJobLifecycle for SelectiveFailureLifecycle {
    async fn register(&self, _: crate::BackgroundJobRegistration) -> Result<(), String> {
        Ok(())
    }

    async fn observe_terminal(
        &self,
        id: &crate::BackgroundJobId,
        terminal: crate::BackgroundJobTerminal,
        _: Option<String>,
    ) -> Result<crate::BackgroundJobPublication, String> {
        self.observed
            .lock()
            .unwrap()
            .push((id.handle.clone(), terminal));
        if self.fail_handle.lock().unwrap().as_deref() == Some(id.handle.as_str()) {
            return Err("injected orphan journal failure".into());
        }
        Ok(crate::BackgroundJobPublication::Published)
    }

    async fn pending(&self, _: &str) -> Vec<crate::BackgroundJobId> {
        Vec::new()
    }

    async fn settle_after_workspace(&self, _: &[crate::BackgroundJobId]) -> Result<(), String> {
        Ok(())
    }
}

#[async_trait::async_trait]
impl crate::BackgroundJobLifecycle for SettlementGate {
    async fn register(&self, _: crate::BackgroundJobRegistration) -> Result<(), String> {
        Ok(())
    }

    async fn observe_terminal(
        &self,
        _: &crate::BackgroundJobId,
        terminal: crate::BackgroundJobTerminal,
        _: Option<String>,
    ) -> Result<crate::BackgroundJobPublication, String> {
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
        self.observed.lock().unwrap().push(terminal);
        Ok(crate::BackgroundJobPublication::Published)
    }

    async fn pending(&self, _: &str) -> Vec<crate::BackgroundJobId> {
        Vec::new()
    }

    async fn settle_after_workspace(&self, _: &[crate::BackgroundJobId]) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(unix)]
#[tokio::test]
async fn preexisting_writer_and_new_read_only_process_do_not_prove_a_same_turn_writer() {
    let _guard = TEST_LOCK.lock().await;
    let root = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink("/dev/zero", root.path().join("zero")).unwrap();
    let state_root = root.path().join("state");
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    registry.set_job_lifecycle(Arc::new(SelectiveFailureLifecycle::default()));

    let writer = registry
        .spawn_managed_live_writer(&runner, "exec sleep 60")
        .await
        .unwrap();
    let turn_baseline = registry.ids();
    let snapshot = crate::effects::workspace_snapshot(root.path(), &state_root)
        .await
        .unwrap();
    let reader = registry
        .spawn_tracked(&runner, "cat zero", root.path(), &state_root, snapshot)
        .await
        .unwrap();

    assert_eq!(
        registry.outcome(&reader).unwrap().state,
        crate::BackgroundState::Running
    );
    assert_eq!(
        registry
            .processes
            .lock()
            .unwrap()
            .get(&reader)
            .unwrap()
            .managed_effect,
        Some(crate::BackgroundJobEffect::ReadOnly)
    );
    assert!(registry.has_running_managed_live_writer_started_after(&[]));
    assert!(
        !registry.has_running_managed_live_writer_started_after(&turn_baseline),
        "the newly started read-only handle must not soften the preexisting writer fence"
    );

    registry.kill_and_reap(&reader).await.unwrap();
    registry.kill_and_reap(&writer).await.unwrap();
}

#[tokio::test]
async fn cancel_during_exit_settlement_does_not_claim_to_kill_completed_work() {
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let gate = Arc::new(SettlementGate {
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
        observed: Mutex::new(Vec::new()),
    });
    registry.set_job_lifecycle(gate.clone());
    let id = registry
        .spawn_managed_live_writer(&runner, "printf done")
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), gate.entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();

    let message = registry.kill(&id).unwrap();
    let mut reap = std::pin::pin!(registry.kill_and_reap(&id));
    let waiting = tokio::time::timeout(Duration::from_millis(20), &mut reap).await;
    gate.release.add_permits(1);
    assert!(
        waiting.is_err(),
        "reap must await the blocked settlement callback"
    );
    tokio::time::timeout(Duration::from_secs(2), &mut reap)
        .await
        .unwrap()
        .unwrap();
    assert!(
        message.contains("already exited") && message.contains("settlement"),
        "{message}"
    );
    assert_eq!(
        registry.outcome(&id).unwrap().state,
        crate::BackgroundState::Exited
    );
    assert_eq!(registry.outcome(&id).unwrap().exit_code, Some(0));
    assert_eq!(
        gate.observed.lock().unwrap().as_slice(),
        &[crate::BackgroundJobTerminal::Succeeded]
    );
}

#[tokio::test]
async fn cancelling_quiescence_wait_reopens_background_admission() {
    let _guard = TEST_LOCK.lock().await;
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let gate = Arc::new(SettlementGate {
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
        observed: Mutex::new(Vec::new()),
    });
    registry.set_job_lifecycle(gate.clone());
    let completed = registry
        .spawn_managed_live_writer(&runner, "printf done")
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), gate.entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();

    {
        let mut quiescence = std::pin::pin!(registry.ensure_quiescent_and_reaped());
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut quiescence)
                .await
                .is_err(),
            "quiescence must wait for the terminal lifecycle callback"
        );
    }

    // Dropping the lifecycle barrier above is cancellation, not a permanent
    // workspace shutdown. A later launch must not inherit its admission latch.
    let next = registry
        .spawn(&runner, "exec sleep 60")
        .expect("cancelled quiescence must reopen background admission");

    gate.release.add_permits(1);
    registry.kill_and_reap(&completed).await.unwrap();
    registry.kill_and_reap(&next).await.unwrap();
}

#[cfg(unix)]
#[path = "release_tests.rs"]
mod release_tests;

#[cfg(unix)]
#[tokio::test]
async fn auto_backgrounded_writer_is_reaped_and_waits_for_workspace_settlement() {
    let _guard = TEST_LOCK.lock().await;
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let lifecycle = Arc::new(PendingWriterLifecycle::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let mut child = runner.spawn_shell("exec sleep 60").unwrap();
    let pgid = child.id().map(|pid| pid as i32);
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let state_root = root.path().join("state");
    std::fs::create_dir_all(&state_root).unwrap();
    let baseline = crate::effects::workspace_snapshot(root.path(), &state_root)
        .await
        .unwrap();
    let id = registry
        .adopt(
            "exec sleep 60",
            child,
            stdout,
            stderr,
            pgid,
            String::new(),
            (root.path().to_path_buf(), state_root, baseline),
        )
        .await
        .unwrap();

    let release = registry.release_requested_running().await.unwrap();
    assert!(release.released.is_empty());
    assert!(release.settlement_pending.is_empty());
    assert_eq!(registry.ids(), std::slice::from_ref(&id));
    assert!(lifecycle.observed.lock().unwrap().is_empty());

    assert_eq!(registry.kill_started_after_and_reap(&[]).await.unwrap(), 1);
    assert_eq!(
        lifecycle.observed.lock().unwrap().as_slice(),
        &[crate::BackgroundJobTerminal::Cancelled]
    );
    assert_eq!(
        registry.outcome(&id).unwrap().state,
        crate::BackgroundState::Killed
    );
    let pending = registry.pending_job_settlements().await;
    assert_eq!(pending.len(), 1);
    registry
        .settle_jobs_after_workspace(&pending)
        .await
        .unwrap();
    assert!(registry.pending_job_settlements().await.is_empty());
    assert_eq!(lifecycle.settled.lock().unwrap().as_slice(), pending);
}

#[tokio::test]
async fn exited_background_launcher_does_not_wait_for_inherited_pipe_eof() {
    let _guard = TEST_LOCK.lock().await;
    crate::preserve_detached_descendants(false);
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let id = registry
        .spawn(&runner, "printf diagnostic; sleep 60 & exit 7")
        .unwrap();
    let settled = tokio::time::timeout(Duration::from_secs(7), async {
        while registry.outcome(&id).unwrap().state == crate::BackgroundState::Running {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    let observed = registry.outcome(&id).unwrap();
    // Always clean up the failing-before case as well.
    registry.kill_and_reap(&id).await.unwrap();
    assert!(
        settled.is_ok(),
        "an exited launcher remained Running on an inherited pipe"
    );
    assert_eq!(observed.state, crate::BackgroundState::Exited);
    assert_eq!(observed.exit_code, Some(7));
    assert!(registry.poll(&id).unwrap().contains("diagnostic"));
    registry.ensure_quiescent_and_reaped().await.unwrap();
}

#[tokio::test]
async fn background_cancellation_keeps_unterminated_diagnostics() {
    let _guard = TEST_LOCK.lock().await;
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let id = registry
        .spawn(
            &runner,
            "printf partial; printf diagnostic >&2; touch ready; sleep 60",
        )
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while !root.path().join("ready").exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    registry.kill_and_reap(&id).await.unwrap();
    let output = registry.poll(&id).unwrap();
    assert!(
        output.contains("partial") && output.contains("diagnostic"),
        "{output}"
    );
    assert_eq!(
        registry.outcome(&id).unwrap().state,
        crate::BackgroundState::Killed
    );
}

#[tokio::test]
async fn overrun_wait_kills_hung_auto_backgrounded_processes() {
    let _guard = TEST_LOCK.lock().await;
    let directory = tempfile::tempdir().unwrap();
    let state = directory.path().join(".hi");
    std::fs::create_dir(&state).unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(directory.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let before = registry.ids();
    let download = registry.spawn(&runner, "sleep 600").unwrap();
    let snapshot = crate::effects::workspace_snapshot(directory.path(), &state)
        .await
        .unwrap();
    let command = "sleep 600";
    let mut child = runner.spawn_shell(command).unwrap();
    let pgid = child.id().map(|p| p as i32);
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let adopted = registry
        .adopt(
            command,
            child,
            stdout,
            stderr,
            pgid,
            String::new(),
            (directory.path().into(), state, snapshot),
        )
        .await
        .unwrap();
    tokio::time::timeout(
        Duration::from_secs(2),
        registry.wait_started_after_and_reap_before(
            &before,
            tokio::time::Instant::now() + Duration::from_millis(80),
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        registry.outcome(&adopted).unwrap().state,
        crate::BackgroundState::Killed
    );
    assert_eq!(
        registry.outcome(&download).unwrap().state,
        crate::BackgroundState::Running
    );
    registry.kill_and_reap(&download).await.unwrap();
}
