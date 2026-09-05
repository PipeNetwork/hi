use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::tests::test_app;

async fn completed_enable(app: &mut crate::App) -> tokio::task::JoinHandle<()> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tx.send("old session remote instruction".into()).unwrap();
    let poller = tokio::spawn(std::future::pending::<()>());
    let abort = poller.abort_handle();
    app.pending_host_enable = Some(tokio::spawn(async move { Ok(Some((rx, abort))) }));
    while !app.pending_host_enable.as_ref().unwrap().is_finished() {
        tokio::task::yield_now().await;
    }
    poller
}

async fn assert_poller_cancelled(poller: tokio::task::JoinHandle<()>) {
    assert!(
        tokio::time::timeout(Duration::from_secs(5), poller)
            .await
            .expect("old poller stopped")
            .unwrap_err()
            .is_cancelled()
    );
}

#[tokio::test]
async fn stopping_host_reaps_completed_enable_without_installing_stale_input() {
    let mut app = test_app("custom", "test-model");
    let poller = completed_enable(&mut app).await;
    app.stop_host_mode();
    app.sync_session_id = Some("destination-session".into());
    app.poll_pending_host_enable().await;
    assert!(app.pending_host_enable.is_none());
    assert!(app.remote_input_rx.is_none());
    assert!(!app.hosting_remote_input);
    assert!(!app.drain_remote_input());
    assert!(app.queue.is_empty());
    assert_poller_cancelled(poller).await;
}

#[tokio::test]
async fn stopping_host_cancels_enablement_still_waiting_for_portal() {
    let mut app = test_app("custom", "test-model");
    let (finished_tx, finished_rx) = tokio::sync::oneshot::channel::<()>();
    app.pending_host_enable = Some(tokio::spawn(async move {
        std::future::pending::<()>().await;
        let _ = finished_tx.send(());
        Ok(None)
    }));
    app.stop_host_mode();
    assert!(
        tokio::time::timeout(Duration::from_secs(5), finished_rx)
            .await
            .expect("pending enable future was cancelled")
            .is_err()
    );
    app.poll_pending_host_enable().await;
    assert!(!app.hosting_remote_input);
    assert!(app.pending_host_enable.is_none());
}

#[tokio::test]
async fn host_off_cancels_pending_startup_even_when_portal_disable_fails() {
    let mut app = test_app("custom", "test-model");
    let calls = Arc::new(Mutex::new(Vec::new()));
    let recorded = calls.clone();
    app.session_host = Some(Box::new(move |enable| {
        recorded.lock().unwrap().push(enable);
        Box::pin(async move { anyhow::bail!("portal unavailable") })
    }));
    let poller = completed_enable(&mut app).await;
    app.handle_daemon_command("off").await;
    app.poll_pending_host_enable().await;
    assert_eq!(*calls.lock().unwrap(), vec![false]);
    assert!(app.pending_host_enable.is_none());
    assert!(!app.hosting_remote_input);
    assert!(app.remote_input_rx.is_none());
    assert!(!app.drain_remote_input());
    assert_poller_cancelled(poller).await;
}

#[tokio::test]
async fn host_off_stops_installed_input_before_portal_failure() {
    let mut app = test_app("custom", "test-model");
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tx.send("do not execute after host off".into()).unwrap();
    let poller = tokio::spawn(std::future::pending::<()>());
    app.remote_input_rx = Some(rx);
    app.remote_input_poller = Some(poller.abort_handle());
    app.hosting_remote_input = true;
    app.session_host = Some(Box::new(|_| Box::pin(async { anyhow::bail!("offline") })));
    app.handle_daemon_command("off").await;
    assert!(!app.hosting_remote_input);
    assert!(!app.drain_remote_input());
    assert!(app.queue.is_empty());
    assert_poller_cancelled(poller).await;
}

#[tokio::test]
async fn repeated_host_on_does_not_duplicate_pending_enablement() {
    let mut app = test_app("custom", "test-model");
    app.session_host = Some(Box::new(|_| panic!("must reuse pending enablement")));
    app.pending_host_enable = Some(tokio::spawn(std::future::pending()));
    let task_id = app.pending_host_enable.as_ref().unwrap().id();
    app.handle_daemon_command("on").await;
    assert_eq!(app.pending_host_enable.as_ref().unwrap().id(), task_id);
    app.stop_host_mode();
}
