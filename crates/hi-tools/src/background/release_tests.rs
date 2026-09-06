use super::*;

struct ExitHandoffRaceLifecycle {
    first_handle: Mutex<Option<String>>,
    second_handle: Mutex<Option<String>>,
    first_orphan_entered: tokio::sync::Semaphore,
    allow_first_orphan: tokio::sync::Semaphore,
    second_terminal_entered: tokio::sync::Semaphore,
    allow_second_terminal: tokio::sync::Semaphore,
    observed: Mutex<Vec<(String, crate::BackgroundJobTerminal)>>,
}

#[async_trait::async_trait]
impl crate::BackgroundJobLifecycle for ExitHandoffRaceLifecycle {
    async fn register(&self, _: crate::BackgroundJobRegistration) -> Result<(), String> {
        Ok(())
    }

    async fn observe_terminal(
        &self,
        id: &crate::BackgroundJobId,
        terminal: crate::BackgroundJobTerminal,
        _: Option<String>,
    ) -> Result<crate::BackgroundJobPublication, String> {
        let first = self.first_handle.lock().unwrap().as_deref() == Some(id.handle.as_str());
        let second = self.second_handle.lock().unwrap().as_deref() == Some(id.handle.as_str());
        if first && terminal == crate::BackgroundJobTerminal::Orphaned {
            self.first_orphan_entered.add_permits(1);
            self.allow_first_orphan.acquire().await.unwrap().forget();
        }
        if second && terminal != crate::BackgroundJobTerminal::Orphaned {
            self.second_terminal_entered.add_permits(1);
            self.allow_second_terminal.acquire().await.unwrap().forget();
        }
        self.observed
            .lock()
            .unwrap()
            .push((id.handle.clone(), terminal));
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
async fn release_waits_for_durable_orphan_and_preserves_the_native_process() {
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
    let id = registry
        .spawn_managed_live_writer(&runner, "exec sleep 60")
        .await
        .unwrap();
    let pid = registry.os_pid(&id).unwrap();
    let _cleanup = DetachedService(pid);

    let mut release = std::pin::pin!(registry.release_requested_running());
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut release)
            .await
            .is_err(),
        "registry release must wait for the durable lifecycle acknowledgement"
    );
    tokio::time::timeout(Duration::from_secs(2), gate.entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    assert_eq!(registry.ids(), std::slice::from_ref(&id));
    assert_eq!(unsafe { libc::kill(pid, 0) }, 0);

    gate.release.add_permits(1);
    tokio::time::timeout(Duration::from_secs(2), &mut release)
        .await
        .unwrap()
        .unwrap();
    assert!(registry.ids().is_empty());
    assert_eq!(
        gate.observed.lock().unwrap().as_slice(),
        &[crate::BackgroundJobTerminal::Orphaned]
    );
    assert_eq!(
        unsafe { libc::kill(pid, 0) },
        0,
        "keep-background handoff must not signal the native process"
    );
}

#[cfg(unix)]
#[test]
fn released_process_survives_registry_and_runtime_drop() {
    let root = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let pid = runtime.block_on(async {
        let runner =
            crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
                .unwrap();
        let registry = BackgroundRegistry::default();
        let id = registry.spawn(&runner, "exec sleep 60").unwrap();
        let pid = registry.os_pid(&id).unwrap();

        let released = registry.release_requested_running().await.unwrap();
        assert_eq!(released.released, [id]);
        assert!(registry.ids().is_empty());
        drop(registry);
        pid
    });

    // Dropping a Tokio runtime drops every outstanding driver future. The
    // released child must no longer carry Tokio's kill-on-drop ownership.
    drop(runtime);
    std::thread::sleep(Duration::from_millis(100));
    let mut status = 0;
    let observed = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
    let survived = observed == 0;
    if survived {
        unsafe {
            libc::kill(pid, libc::SIGKILL);
            libc::waitpid(pid, &mut status, 0);
        }
    }
    assert!(
        survived,
        "released process exited when its registry/runtime was dropped (waitpid={observed})"
    );
}

#[cfg(unix)]
#[test]
fn driver_drop_before_release_kills_the_owned_process() {
    let root = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (registry, process, pid) = runtime.block_on(async {
        let runner =
            crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
                .unwrap();
        let registry = BackgroundRegistry::default();
        let id = registry.spawn(&runner, "exec sleep 60").unwrap();
        let pid = registry.os_pid(&id).unwrap();
        let process = lookup(&registry, &id).unwrap();
        (registry, process, pid)
    });

    drop(runtime);
    let mut status = 0;
    let mut observed = 0;
    for _ in 0..50 {
        observed = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
        if observed != 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let still_running = unsafe { libc::kill(pid, 0) } == 0;
    if still_running {
        unsafe {
            libc::kill(pid, libc::SIGKILL);
            libc::waitpid(pid, &mut status, 0);
        }
    }
    // Avoid a redundant group signal after this test has itself reaped the
    // pid; the assertion above is specifically about the driver-drop guard.
    process.ownership_released.store(true, Ordering::Release);
    drop(process);
    drop(registry);
    assert!(
        observed == pid || (observed == -1 && !still_running),
        "dropping an owned driver did not terminate its process (waitpid={observed}, running={still_running})"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn released_managed_process_publishes_only_orphaned_after_native_exit() {
    let _guard = TEST_LOCK.lock().await;
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let lifecycle = Arc::new(SelectiveFailureLifecycle::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let id = registry
        .spawn_managed_live_writer(&runner, "exec sleep 60")
        .await
        .unwrap();
    let pid = registry.os_pid(&id).unwrap();
    let process = lookup(&registry, &id).unwrap();

    registry.release_requested_running().await.unwrap();
    unsafe { libc::kill(-pid, libc::SIGKILL) };
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let reaped = process.reaped.notified();
            tokio::pin!(reaped);
            reaped.as_mut().enable();
            if process.inner.lock().unwrap().reaped {
                break;
            }
            reaped.await;
        }
    })
    .await
    .unwrap();

    assert_eq!(
        lifecycle.observed.lock().unwrap().as_slice(),
        &[(id, crate::BackgroundJobTerminal::Orphaned)]
    );
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
    let launcher = registry.os_pid(&id).unwrap();
    if preserve {
        wait_for_process_to_disappear(launcher).await;
    } else {
        tokio::time::timeout(Duration::from_secs(3), async {
            while registry.outcome(&id).unwrap().state == crate::BackgroundState::Running {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
    }
    let proc = lookup(&registry, &id).unwrap();
    let service = DetachedService(proc.inner.lock().unwrap().output.trim().parse().unwrap());
    if preserve {
        assert_eq!(
            registry.outcome(&id).unwrap().state,
            crate::BackgroundState::Running
        );
        assert_eq!(
            registry.release_requested_running().await.unwrap().released,
            [id]
        );
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let alive = unsafe { libc::kill(service.0, 0) } == 0;
    assert_eq!(
        alive, preserve,
        "background completion must honor keep-background"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn requested_descendant_group_stays_live_until_durable_release() {
    let _guard = TEST_LOCK.lock().await;
    let _reset = ResetPreservation;
    crate::preserve_detached_descendants(true);
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let lifecycle = Arc::new(SelectiveFailureLifecycle::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let pid_file = root.path().join("descendant.pid");
    let command = format!(
        "sleep 60 >/dev/null 2>&1 & echo $! > '{}'",
        pid_file.display()
    );
    let id = registry
        .spawn_managed_live_writer(&runner, &command)
        .await
        .unwrap();
    let launcher_pid = registry.os_pid(&id).unwrap();
    let descendant_pid = wait_for_recorded_pid(&pid_file).await;
    let _cleanup = DetachedService(descendant_pid);
    wait_for_process_to_disappear(launcher_pid).await;

    assert_eq!(
        registry.outcome(&id).unwrap().state,
        crate::BackgroundState::Running,
        "the durable job must follow the live process group, not its exited launcher"
    );
    assert!(lifecycle.observed.lock().unwrap().is_empty());

    let release = registry.release_requested_running().await.unwrap();
    assert_eq!(release.released, std::slice::from_ref(&id));
    assert!(registry.ids().is_empty());
    assert_eq!(
        lifecycle.observed.lock().unwrap().as_slice(),
        &[(id, crate::BackgroundJobTerminal::Orphaned)]
    );
    assert_eq!(
        unsafe { libc::kill(descendant_pid, 0) },
        0,
        "durably released descendant must remain alive"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn requested_short_descendant_publishes_success_after_the_group_exits() {
    let _guard = TEST_LOCK.lock().await;
    let _reset = ResetPreservation;
    crate::preserve_detached_descendants(true);
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let lifecycle = Arc::new(SelectiveFailureLifecycle::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let id = registry
        .spawn_managed_live_writer(&runner, "sleep 0.2 >/dev/null 2>&1 &")
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(3), async {
        while registry.outcome(&id).unwrap().state == crate::BackgroundState::Running {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();

    let outcome = registry.outcome(&id).unwrap();
    assert_eq!(outcome.state, crate::BackgroundState::Exited);
    assert_eq!(outcome.exit_code, Some(0));
    assert_eq!(
        lifecycle.observed.lock().unwrap().as_slice(),
        &[(id, crate::BackgroundJobTerminal::Succeeded)]
    );
}

#[cfg(unix)]
#[tokio::test]
async fn killing_requested_descendant_group_publishes_cancelled() {
    let _guard = TEST_LOCK.lock().await;
    let _reset = ResetPreservation;
    crate::preserve_detached_descendants(true);
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let lifecycle = Arc::new(SelectiveFailureLifecycle::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let pid_file = root.path().join("cancel-descendant.pid");
    let command = format!(
        "sleep 60 >/dev/null 2>&1 & echo $! > '{}'",
        pid_file.display()
    );
    let id = registry
        .spawn_managed_live_writer(&runner, &command)
        .await
        .unwrap();
    let launcher_pid = registry.os_pid(&id).unwrap();
    let descendant_pid = wait_for_recorded_pid(&pid_file).await;
    let _cleanup = DetachedService(descendant_pid);
    wait_for_process_to_disappear(launcher_pid).await;

    registry.kill_and_reap(&id).await.unwrap();

    assert_eq!(
        registry.outcome(&id).unwrap().state,
        crate::BackgroundState::Killed
    );
    assert_eq!(
        lifecycle.observed.lock().unwrap().as_slice(),
        &[(id, crate::BackgroundJobTerminal::Cancelled)]
    );
}

#[cfg(unix)]
async fn wait_for_recorded_pid(path: &Path) -> i32 {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if let Ok(contents) = std::fs::read_to_string(path)
                && let Ok(pid) = contents.trim().parse()
            {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap()
}

#[cfg(unix)]
async fn wait_for_process_to_disappear(pid: i32) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while unsafe { libc::kill(pid, 0) } == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn partial_release_never_reclaims_an_acknowledged_orphan() {
    let _guard = TEST_LOCK.lock().await;
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let lifecycle = Arc::new(SelectiveFailureLifecycle::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let first = registry
        .spawn_managed_live_writer(&runner, "exec sleep 61")
        .await
        .unwrap();
    let second = registry
        .spawn_managed_live_writer(&runner, "exec sleep 62")
        .await
        .unwrap();
    let first_pid = registry.os_pid(&first).unwrap();
    let second_pid = registry.os_pid(&second).unwrap();
    let _first_cleanup = DetachedService(first_pid);
    let _second_cleanup = DetachedService(second_pid);
    *lifecycle.fail_handle.lock().unwrap() = Some(second.clone());

    let error = registry.release_requested_running().await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected orphan journal failure")
    );
    assert_eq!(registry.ids(), std::slice::from_ref(&second));
    assert_eq!(unsafe { libc::kill(first_pid, 0) }, 0);
    assert_eq!(unsafe { libc::kill(second_pid, 0) }, 0);
    assert_eq!(
        lifecycle.observed.lock().unwrap().as_slice(),
        &[
            (first, crate::BackgroundJobTerminal::Orphaned),
            (second, crate::BackgroundJobTerminal::Orphaned),
        ]
    );
}

#[cfg(unix)]
#[tokio::test]
async fn release_rechecks_native_exit_before_orphaning_snapshotted_processes() {
    let _guard = TEST_LOCK.lock().await;
    let root = tempfile::tempdir().unwrap();
    let runner =
        crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
            .unwrap();
    let registry = BackgroundRegistry::default();
    let lifecycle = Arc::new(ExitHandoffRaceLifecycle {
        first_handle: Mutex::new(None),
        second_handle: Mutex::new(None),
        first_orphan_entered: tokio::sync::Semaphore::new(0),
        allow_first_orphan: tokio::sync::Semaphore::new(0),
        second_terminal_entered: tokio::sync::Semaphore::new(0),
        allow_second_terminal: tokio::sync::Semaphore::new(0),
        observed: Mutex::new(Vec::new()),
    });
    registry.set_job_lifecycle(lifecycle.clone());

    let trigger = root.path().join("finish-backgrounds");
    let wait_for_trigger = format!(
        "while [ ! -e '{}' ]; do sleep 0.01; done",
        trigger.display()
    );
    let first = registry
        .spawn_managed_live_writer(&runner, "exec sleep 60")
        .await
        .unwrap();
    let second = registry
        .spawn_managed_live_writer(&runner, &wait_for_trigger)
        .await
        .unwrap();
    // `spawn` is intentionally unmanaged, but release must apply the same
    // guarded native-state re-check before forgetting it.
    let unmanaged = registry
        .spawn(&runner, &format!("{wait_for_trigger}; printf unmanaged"))
        .unwrap();
    *lifecycle.first_handle.lock().unwrap() = Some(first.clone());
    *lifecycle.second_handle.lock().unwrap() = Some(second.clone());
    let first_pid = registry.os_pid(&first).unwrap();
    let _first_cleanup = DetachedService(first_pid);

    let mut release = std::pin::pin!(registry.release_requested_running());
    tokio::select! {
        permit = lifecycle.first_orphan_entered.acquire() => permit.unwrap().forget(),
        result = &mut release => panic!("release completed before the first orphan gate: {result:?}"),
    }

    std::fs::write(&trigger, b"go").unwrap();
    tokio::time::timeout(
        Duration::from_secs(2),
        lifecycle.second_terminal_entered.acquire(),
    )
    .await
    .unwrap()
    .unwrap()
    .forget();
    tokio::time::timeout(Duration::from_secs(2), async {
        while registry.outcome(&unmanaged).unwrap().state == crate::BackgroundState::Running {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    lifecycle.allow_first_orphan.add_permits(1);
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut release)
            .await
            .is_err(),
        "release must wait for a native exit's real terminal publication"
    );
    assert_eq!(registry.ids(), [second.clone(), unmanaged.clone()]);
    assert_eq!(
        lifecycle.observed.lock().unwrap().as_slice(),
        &[(first.clone(), crate::BackgroundJobTerminal::Orphaned)]
    );

    lifecycle.allow_second_terminal.add_permits(1);
    let summary = tokio::time::timeout(Duration::from_secs(2), &mut release)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(summary.released, [first]);
    assert!(summary.settlement_pending.is_empty());
    assert_eq!(registry.ids(), [second.clone(), unmanaged]);
    assert_eq!(
        lifecycle.observed.lock().unwrap().as_slice(),
        &[
            (
                summary.released[0].clone(),
                crate::BackgroundJobTerminal::Orphaned,
            ),
            (second, crate::BackgroundJobTerminal::Succeeded),
        ]
    );
}
