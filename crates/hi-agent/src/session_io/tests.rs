use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

struct Sink;
impl SessionSink for Sink {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> Result<()> {
        Ok(())
    }
    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> Result<()> {
        Ok(())
    }
}

#[test]
fn compatibility_settlement_never_writes_credit_before_its_outcome() {
    #[derive(Default)]
    struct CompatibilitySink {
        fail_receipt: bool,
        calls: Vec<&'static str>,
    }
    impl SessionSink for CompatibilitySink {
        fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> Result<()> {
            Ok(())
        }
        fn record_compaction(&mut self, _: &[hi_ai::Message]) -> Result<()> {
            Ok(())
        }
        fn record_turn_outcome(&mut self, _: &crate::TurnOutcome, _: Option<&str>) -> Result<()> {
            self.calls.push("outcome");
            anyhow::ensure!(!self.fail_receipt, "receipt did not commit");
            Ok(())
        }
        fn record_task_recovery(&mut self, _: &crate::TaskRecoveryState) -> Result<()> {
            self.calls.push("credit");
            Ok(())
        }
        fn record_goal(&mut self, _: &crate::Goal) -> Result<()> {
            self.calls.push("goal");
            Ok(())
        }
    }
    let outcome = crate::TurnOutcome::infrastructure_failure("test", None, vec![]);
    let recovery = crate::TaskRecoveryState::default();
    let goal = crate::Goal::new("finish", vec!["implement".into()]);
    let mut sink = CompatibilitySink {
        fail_receipt: true,
        ..Default::default()
    };
    assert!(
        sink.record_turn_settlement(&outcome, None, Some(&recovery), Some(&goal))
            .is_err()
    );
    assert_eq!(sink.calls, ["outcome"]);
    sink.fail_receipt = false;
    sink.calls.clear();
    sink.record_turn_settlement(&outcome, None, Some(&recovery), Some(&goal))
        .unwrap();
    assert_eq!(sink.calls, ["outcome", "goal", "credit"]);
}

#[tokio::test(flavor = "current_thread")]
async fn accepted_write_survives_cancellation_and_barrier_waits_for_commit() {
    let owner = OwnedSessionSink::new(Box::new(Sink));
    let handle = owner.io_handle().unwrap();
    let (release, blocked) = mpsc::channel();
    let (entered, started) = oneshot::channel();
    let committed = Arc::new(AtomicBool::new(false));
    let written = committed.clone();
    let writer = handle.clone();
    let task = tokio::spawn(async move {
        writer
            .write(move |_| {
                let _ = entered.send(());
                blocked.recv().unwrap();
                written.store(true, Ordering::SeqCst);
                Ok(())
            })
            .await
    });
    started.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), handle.barrier())
            .await
            .is_err()
    );
    assert!(!committed.load(Ordering::SeqCst));
    release.send(()).unwrap();
    handle.barrier().await.unwrap();
    assert!(committed.load(Ordering::SeqCst));
}

#[tokio::test(flavor = "current_thread")]
async fn late_failure_remains_visible_when_its_original_waiter_disappears() {
    let owner = OwnedSessionSink::new(Box::new(Sink));
    let handle = owner.io_handle().unwrap();
    let (release, blocked) = mpsc::channel();
    let (entered, started) = oneshot::channel();
    let writer = handle.clone();
    let task = tokio::spawn(async move {
        writer
            .write(move |_| {
                let _ = entered.send(());
                blocked.recv().unwrap();
                anyhow::bail!("fsync failed after cancellation")
            })
            .await
    });
    started.await.unwrap();
    task.abort();
    let _ = task.await;
    release.send(()).unwrap();
    assert!(
        handle
            .barrier()
            .await
            .unwrap_err()
            .to_string()
            .contains("fsync failed")
    );
    handle.barrier().await.unwrap();
}

#[tokio::test]
async fn acknowledged_failure_does_not_poison_later_valid_writes() {
    let owner = OwnedSessionSink::new(Box::new(Sink));
    let handle = owner.io_handle().unwrap();
    assert!(
        handle
            .write(|_| anyhow::bail!("rejected write"))
            .await
            .is_err()
    );
    handle.write(|_| Ok(())).await.unwrap();
    handle.barrier().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn queue_bounds_accepted_work_and_cancels_unaccepted_waiters() {
    let owner = OwnedSessionSink::new(Box::new(Sink));
    let handle = owner.io_handle().unwrap();
    let (release, blocked) = mpsc::channel();
    let (entered, started) = oneshot::channel();
    let writer = handle.clone();
    let first = tokio::spawn(async move {
        writer
            .write(move |_| {
                let _ = entered.send(());
                blocked.recv().unwrap();
                Ok(())
            })
            .await
    });
    started.await.unwrap();
    let mut accepted = Vec::new();
    for _ in 1..CAPACITY {
        let writer = handle.clone();
        accepted.push(tokio::spawn(async move { writer.write(|_| Ok(())).await }));
    }
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while handle.0.capacity.available_permits() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let ran = Arc::new(AtomicBool::new(false));
    let cancelled = ran.clone();
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            handle.write(move |_| {
                cancelled.store(true, Ordering::SeqCst);
                Ok(())
            })
        )
        .await
        .is_err()
    );
    release.send(()).unwrap();
    first.await.unwrap().unwrap();
    for task in accepted {
        task.await.unwrap().unwrap();
    }
    handle.barrier().await.unwrap();
    assert!(
        !ran.load(Ordering::SeqCst),
        "cancelled admission must not run later"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn synchronous_compatibility_refuses_to_wait_behind_an_accepted_append() {
    let owner = OwnedSessionSink::new(Box::new(Sink));
    let handle = owner.io_handle().unwrap();
    let (release, blocked) = mpsc::channel();
    let (entered, started) = oneshot::channel();
    let writer = handle.clone();
    let task = tokio::spawn(async move {
        writer
            .write(move |_| {
                let _ = entered.send(());
                blocked.recv().unwrap();
                Ok(())
            })
            .await
    });
    started.await.unwrap();
    task.abort();
    let _ = task.await;
    let attempted = Arc::new(AtomicBool::new(false));
    let ran = attempted.clone();
    let error = handle
        .write_blocking(move |_| {
            ran.store(true, Ordering::SeqCst);
            Ok(())
        })
        .unwrap_err();
    assert!(error.to_string().contains("pending work"));
    assert!(!attempted.load(Ordering::SeqCst));
    release.send(()).unwrap();
    handle.barrier().await.unwrap();
    handle.write_blocking(|_| Ok(())).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn no_session_metadata_is_lazy_owned_and_ordered_before_later_saved_exports() {
    let mut agent = crate::tests::common::agent(Vec::new(), crate::tests::common::config());
    agent
        .write_session(|_| panic!("no-session transcript writes remain absent"))
        .await
        .unwrap();
    assert!(!agent.has_session_io());
    let (release, blocked) = mpsc::channel();
    let (entered, started) = oneshot::channel();
    let exports = Arc::new(Mutex::new(Vec::new()));
    let old_exports = exports.clone();
    let task = tokio::spawn(agent.write_session_metadata(move |_| {
        let _ = entered.send(());
        blocked.recv().unwrap();
        old_exports.lock().unwrap().push("old");
        Ok(())
    }));
    started.await.unwrap();
    assert!(agent.has_session_io());
    assert!(
        agent.session.is_none(),
        "metadata ownership must not create a saved session"
    );
    task.abort();
    let _ = task.await;
    agent.set_session(Box::new(Sink));
    let new_exports = exports.clone();
    let newer = tokio::spawn(agent.write_session_metadata(move |_| {
        new_exports.lock().unwrap().push("new");
        Ok(())
    }));
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            agent.session_barrier()
        )
        .await
        .is_err()
    );
    assert!(exports.lock().unwrap().is_empty());
    release.send(()).unwrap();
    newer.await.unwrap().unwrap();
    agent.session_barrier().await.unwrap();
    assert_eq!(*exports.lock().unwrap(), vec!["old", "new"]);
}

#[tokio::test(flavor = "current_thread")]
async fn session_barrier_drains_both_metadata_and_later_attached_saved_owner() {
    let mut agent = crate::tests::common::agent(Vec::new(), crate::tests::common::config());
    let (release_metadata, blocked_metadata) = mpsc::channel();
    let (entered_metadata, started_metadata) = oneshot::channel();
    let metadata = tokio::spawn(agent.write_session_metadata(move |_| {
        let _ = entered_metadata.send(());
        blocked_metadata.recv().unwrap();
        Ok(())
    }));
    started_metadata.await.unwrap();
    metadata.abort();
    let _ = metadata.await;
    agent.set_session(Box::new(Sink));
    let (release_saved, blocked_saved) = mpsc::channel();
    let (entered_saved, started_saved) = oneshot::channel();
    let saved = tokio::spawn(agent.write_session(move |_| {
        let _ = entered_saved.send(());
        blocked_saved.recv().unwrap();
        Ok(())
    }));
    started_saved.await.unwrap();
    saved.abort();
    let _ = saved.await;
    release_metadata.send(()).unwrap();
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            agent.session_barrier()
        )
        .await
        .is_err()
    );
    release_saved.send(()).unwrap();
    agent.session_barrier().await.unwrap();
}
