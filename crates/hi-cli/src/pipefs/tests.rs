use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::{PipeFsHost, PipeFsMcpConfig, effective_startup_mode, lease_monitor};
use crate::sync::{RemoteSessionSink, SyncConfig};
use hi_agent::WorkspaceDurability;

fn contains_file_named(root: &std::path::Path, name: &str) -> bool {
    let Ok(entries) = std::fs::read_dir(root) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|entry| {
        let path = entry.path();
        path.file_name().is_some_and(|file| file == name)
            || path.is_dir() && contains_file_named(&path, name)
    })
}

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

#[tokio::test]
async fn fenced_exit_quiesces_pipefs_workers_without_removing_recovery_cache() {
    let temp = tempfile::tempdir().unwrap();
    let original = temp.path().join("original");
    let state = temp.path().join("state");
    let cache = temp.path().join("cache");
    std::fs::create_dir_all(&original).unwrap();
    std::fs::create_dir_all(&state).unwrap();
    let sync_config = SyncConfig {
        base_url: "http://127.0.0.1:1".into(),
        api_key: "test-key".into(),
        machine_id: None,
        cwd_digest: None,
    };
    let client = hi_pipefs::PipeFsClient::new(hi_pipefs::PipeFsClientConfig::new(
        sync_config.base_url.clone(),
        sync_config.api_key.clone(),
    ))
    .unwrap();
    let cache_scope = client.cache_scope();
    let workspace = hi_pipefs::PipeFsWorkspace::new(
        client,
        hi_pipefs::PipeFsLease {
            token: "lease-token".into(),
            generation: 3,
        },
        hi_pipefs::PipeFsWorkspaceConfig {
            session_id: "fenced-exit-test".into(),
            cache_scope,
            original_workspace_root: original,
            original_state_root: state,
            cache_base: Some(cache.clone()),
        },
    )
    .unwrap();
    let sync = Arc::new(RemoteSessionSink::new_for_test(
        sync_config,
        "fenced-exit-test".into(),
    ));
    let durability = lease_monitor::build_durability(
        workspace,
        sync,
        Arc::new(hi_tools::BackgroundRegistry::default()),
        hi_tools::ForegroundProcessRegistry::default(),
    );
    durability
        .background_process_state("writer-1", true)
        .await
        .unwrap();
    assert!(contains_file_named(&cache, "recovery-required"));
    assert!(!durability._lease_loss_monitor.is_stopped());

    durability.quiesce_for_fenced_exit().await.unwrap();

    assert!(durability._lease_loss_monitor.is_stopped());
    assert!(
        durability
            .background_checkpoints
            .tasks
            .lock()
            .unwrap()
            .is_empty()
    );
    assert!(cache.exists());
    assert!(
        contains_file_named(&cache, "recovery-required"),
        "fenced exit must retain local recovery evidence"
    );
}
