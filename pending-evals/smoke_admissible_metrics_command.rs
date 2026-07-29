//! target-crate: hi-agent
//! pre-fix: `/metrics` parsed as Command::Unknown, so learning-ledger reports were unreachable from a session

#[test]
fn metrics_is_a_recognized_session_command() {
    let parsed = hi_agent::command::parse("/metrics").expect("slash input always parses");
    assert!(
        !matches!(parsed, hi_agent::Command::Unknown(_)),
        "/metrics must be a recognized command, got {parsed:?}"
    );
}
