use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::Notify;

#[tokio::test]
async fn poll_wait_returns_when_wait_abort_fires() {
    let registry = BackgroundRegistry::default();
    let abort = Arc::new(Notify::new());
    let pending = Arc::new(AtomicBool::new(false));
    registry.set_wait_abort(Some(abort.clone()), Some(pending.clone()));
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let id = registry.spawn(&runner, "sleep 30").unwrap();
    let wait = tokio::spawn(async move { registry.poll_wait(&id, Duration::from_secs(20)).await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    pending.store(true, Ordering::Release);
    abort.notify_waiters();
    let out = tokio::time::timeout(Duration::from_secs(2), wait)
        .await
        .expect("abort must not leave poll_wait parked")
        .expect("join")
        .expect("poll");
    assert!(
        !out.is_empty(),
        "aborted poll still returns a snapshot: {out}"
    );
}

#[tokio::test]
async fn monitor_wake_skips_own_turn_and_fires_for_later_turns() {
    let registry = BackgroundRegistry::default();
    registry.set_current_turn(1);
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let id = registry.spawn(&runner, "sleep 30").unwrap();
    registry.arm_monitor(&id, None, true);
    assert!(
        !registry.monitor_should_wake_turn(&id, 1),
        "own-turn monitor events stay in the running turn"
    );
    assert!(
        registry.monitor_should_wake_turn(&id, 2),
        "a later turn may auto-wake"
    );
    let _ = registry.kill(&id);
}

#[tokio::test]
async fn poll_wait_ignores_stale_abort_notify_without_pending_follow_up() {
    let registry = BackgroundRegistry::default();
    let abort = Arc::new(Notify::new());
    let pending = Arc::new(AtomicBool::new(false));
    registry.set_wait_abort(Some(abort.clone()), Some(pending));
    let runner = crate::ProcessRunner::from_current_dir().unwrap();
    let id = registry.spawn(&runner, "sleep 30").unwrap();
    abort.notify_waiters();
    abort.notify_one();
    let started = std::time::Instant::now();
    let wait = tokio::spawn({
        let id = id.clone();
        async move { registry.poll_wait(&id, Duration::from_millis(400)).await }
    });
    let out = tokio::time::timeout(Duration::from_secs(2), wait)
        .await
        .expect("join")
        .expect("task")
        .expect("poll");
    assert!(
        started.elapsed() >= Duration::from_millis(200),
        "stale notify must not treat a live process as idle: {:?}",
        started.elapsed()
    );
    assert!(!out.is_empty(), "{out}");
}
