use super::test_app;
use crate::app::send_now_queued_follow_up;

#[test]
fn empty_enter_send_now_offers_plain_queue_head() {
    let mut app = test_app("p", "m");
    app.queue.push_back("also run tests".into());
    app.queue.push_back("/status".into());
    let inbox = hi_agent::InterjectionInbox::default();
    assert!(send_now_queued_follow_up(&mut app, Some(&inbox)));
    assert_eq!(inbox.pending(), vec!["also run tests".to_string()]);
    assert_eq!(
        app.queue.iter().cloned().collect::<Vec<_>>(),
        vec!["also run tests".to_string(), "/status".to_string()],
        "send-now keeps the visible queue until reconcile"
    );
}

#[test]
fn send_now_skips_slash_commands() {
    let mut app = test_app("p", "m");
    app.queue.push_back("/status".into());
    let inbox = hi_agent::InterjectionInbox::default();
    assert!(!send_now_queued_follow_up(&mut app, Some(&inbox)));
    assert!(!inbox.has_pending());
}
