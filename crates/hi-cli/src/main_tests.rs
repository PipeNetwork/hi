use super::{
    canonical_session_identity, completed_session_switch, pipefs_startup_authority_required,
    settle_before_post_session_work, validate_tui_event_trace_request,
};
use crate::config::Cli;
use clap::Parser;

#[tokio::test]
async fn interactive_shutdown_settles_before_optional_post_session_work() {
    let phase = std::sync::atomic::AtomicUsize::new(0);
    settle_before_post_session_work(
        async {
            assert_eq!(phase.load(std::sync::atomic::Ordering::SeqCst), 0);
            phase.store(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        },
        async {
            assert_eq!(
                phase.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "optional feedback/flush work ran before workspace settlement"
            );
            phase.store(2, std::sync::atomic::Ordering::SeqCst);
        },
    )
    .await
    .unwrap();
    assert_eq!(phase.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[tokio::test]
async fn failed_interactive_settlement_skips_optional_post_session_work() {
    let post_session_polled = std::sync::atomic::AtomicBool::new(false);
    let error =
        settle_before_post_session_work(async { anyhow::bail!("settlement failed") }, async {
            post_session_polled.store(true, std::sync::atomic::Ordering::SeqCst);
        })
        .await
        .unwrap_err();

    assert!(error.to_string().contains("settlement failed"));
    assert!(
        !post_session_polled.load(std::sync::atomic::Ordering::SeqCst),
        "post-session work must not run after failed authoritative settlement"
    );
}

#[test]
fn canonical_remote_session_identity_survives_a_random_local_cache_name() {
    let local = std::path::Path::new("/tmp/random-local-continuation.jsonl");
    assert_eq!(
        canonical_session_identity(None, Some("remote-session"), local),
        "remote-session"
    );
    assert_eq!(
        canonical_session_identity(Some("explicit-session"), Some("remote-session"), local),
        "explicit-session"
    );
    assert_eq!(
        canonical_session_identity(None, None, local),
        "random-local-continuation"
    );
    assert_eq!(
        completed_session_switch("remote-session".to_string(), "summary".to_string()).id,
        "remote-session"
    );
}

#[test]
fn resumed_remote_identity_always_requires_authoritative_pipefs_probe() {
    assert!(pipefs_startup_authority_required(
        true, false, false, true, false
    ));
    assert!(pipefs_startup_authority_required(
        true, false, false, false, true
    ));
    assert!(!pipefs_startup_authority_required(
        false, false, false, true, false
    ));
    assert!(!pipefs_startup_authority_required(
        true, false, false, false, false
    ));
}

#[test]
fn tui_event_trace_accepts_only_the_full_interactive_frontend() {
    let interactive =
        Cli::try_parse_from(["hi", "--tui-events-jsonl", "/tmp/hi-tui-events-test.jsonl"]).unwrap();
    validate_tui_event_trace_request(&interactive, true, true).unwrap();
    assert!(validate_tui_event_trace_request(&interactive, false, true).is_err());

    let plain = Cli::try_parse_from([
        "hi",
        "--plain",
        "--tui-events-jsonl",
        "/tmp/hi-tui-events-test.jsonl",
    ])
    .unwrap();
    assert!(validate_tui_event_trace_request(&plain, true, true).is_err());
}
