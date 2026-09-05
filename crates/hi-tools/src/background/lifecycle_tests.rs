use super::*;
use std::time::Duration;

struct SettlementGate {
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
    observed: Mutex<Vec<crate::BackgroundJobTerminal>>,
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

#[cfg(unix)]
struct DetachedService(i32);

#[cfg(unix)]
impl Drop for DetachedService {
    fn drop(&mut self) {
        unsafe { libc::kill(self.0, libc::SIGKILL) };
    }
}

struct ResetPreservation;

impl Drop for ResetPreservation {
    fn drop(&mut self) {
        crate::preserve_detached_descendants(false);
    }
}

#[cfg(unix)]
#[tokio::test]
async fn background_completion_honors_keep_background() {
    detached_launcher_obeys_preservation(true).await;
}

#[cfg(unix)]
#[tokio::test]
async fn background_completion_cleans_up_default_detached_descendants() {
    detached_launcher_obeys_preservation(false).await;
}

#[cfg(unix)]
#[tokio::test]
async fn cancellation_during_pipe_drain_overrides_keep_background() {
    let _guard = TEST_LOCK.lock().await;
    let _reset = ResetPreservation;
    crate::preserve_detached_descendants(true);
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let id = registry
        .spawn(
            &runner,
            "sleep 60 & echo $! > child.pid; printf diagnostic; exit 0",
        )
        .unwrap();
    let parent = registry.os_pid(&id).unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while unsafe { libc::kill(parent, 0) } == 0 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    let child = DetachedService(
        std::fs::read_to_string(root.path().join("child.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap(),
    );
    assert!(
        !lookup(&registry, &id)
            .unwrap()
            .inner
            .lock()
            .unwrap()
            .native_exited
    );
    let message = registry.kill_and_reap(&id).await.unwrap();
    assert!(message.contains("stopped"), "{message}");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_ne!(
        unsafe { libc::kill(child.0, 0) },
        0,
        "cancellation must kill the inherited-pipe owner"
    );
    assert_eq!(
        registry.outcome(&id).unwrap().state,
        crate::BackgroundState::Killed
    );
    assert!(registry.poll(&id).unwrap().contains("diagnostic"));
}

#[cfg(unix)]
async fn detached_launcher_obeys_preservation(preserve: bool) {
    let _guard = TEST_LOCK.lock().await;
    let _reset = ResetPreservation;
    crate::preserve_detached_descendants(preserve);
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let id = registry
        .spawn(&runner, "sleep 60 >/dev/null 2>&1 & echo $!")
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while registry.outcome(&id).unwrap().state == crate::BackgroundState::Running {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    let proc = lookup(&registry, &id).unwrap();
    let service = DetachedService(proc.inner.lock().unwrap().output.trim().parse().unwrap());
    tokio::time::sleep(Duration::from_millis(200)).await;
    let alive = unsafe { libc::kill(service.0, 0) } == 0;
    assert_eq!(
        alive, preserve,
        "background completion must honor keep-background"
    );
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
