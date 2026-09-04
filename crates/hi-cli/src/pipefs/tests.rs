use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::{PipeFsHost, PipeFsMcpConfig, effective_startup_mode};
use crate::sync::SyncConfig;

#[test]
fn existing_remote_state_wins_startup_precedence() {
    assert!(!effective_startup_mode(true, Some(false), true));
    assert!(effective_startup_mode(true, Some(true), false));
    assert!(!effective_startup_mode(true, None, true));
}

#[test]
fn new_session_uses_explicit_or_configured_request_then_defaults_off() {
    assert!(effective_startup_mode(false, Some(false), true));
    assert!(!effective_startup_mode(false, Some(true), false));
    assert!(!effective_startup_mode(false, None, false));
}

#[test]
fn ordinary_local_host_does_not_require_pipefs_credentials() {
    let temp = tempfile::tempdir().expect("temporary directory");
    let host = PipeFsHost::new(
        SyncConfig {
            base_url: String::new(),
            api_key: String::new(),
            machine_id: None,
            cwd_digest: None,
        },
        "local-session".to_string(),
        temp.path().join("session.jsonl"),
        Arc::new(Mutex::new(None)),
        temp.path().join("workspace"),
        temp.path().join("state"),
        PipeFsMcpConfig {
            import_policy: hi_mcp::McpImportPolicy::default(),
            pipe_attach: None,
            server_policies: HashMap::new(),
        },
    )
    .expect("ordinary local session host");

    assert!(!host.local_state_requires_remote_probe());
}
