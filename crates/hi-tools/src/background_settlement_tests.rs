use super::*;

#[tokio::test]
async fn normal_completion_wait_is_cancellation_safe_and_spares_deliberate_jobs() {
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
    let command = "while [ ! -f release ]; do sleep 0.01; done; printf done > completed";
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
    assert!(
        tokio::time::timeout(
            Duration::from_millis(50),
            registry.wait_started_after_and_reap(&before)
        )
        .await
        .is_err()
    );
    assert_eq!(
        registry.outcome(&adopted).unwrap().state,
        crate::BackgroundState::Running
    );
    std::fs::write(directory.path().join("release"), "").unwrap();
    tokio::time::timeout(
        Duration::from_secs(3),
        registry.wait_started_after_and_reap(&before),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(directory.path().join("completed")).unwrap(),
        "done"
    );
    assert_eq!(registry.outcome(&adopted).unwrap().exit_code, Some(0));
    assert_eq!(
        registry.outcome(&download).unwrap().state,
        crate::BackgroundState::Running
    );
    registry.kill_and_reap(&download).await.unwrap();
}
