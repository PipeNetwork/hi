use super::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::watch;

struct RecoveryTranscript {
    lease: PipeFsLease,
    lease_status: watch::Sender<crate::PipeFsLeaseStatus>,
    causal_acks: AtomicUsize,
    compatibility_flushes: AtomicUsize,
}

impl RecoveryTranscript {
    fn new(lease: PipeFsLease) -> Arc<Self> {
        let (lease_status, _) = watch::channel(crate::PipeFsLeaseStatus::Valid);
        Arc::new(Self {
            lease,
            lease_status,
            causal_acks: AtomicUsize::new(0),
            compatibility_flushes: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl PipeFsSessionBridge for RecoveryTranscript {
    fn subscribe_lease_status(&self) -> watch::Receiver<crate::PipeFsLeaseStatus> {
        self.lease_status.subscribe()
    }

    async fn refresh_lease(&self) -> Result<PipeFsLease> {
        Ok(self.lease.clone())
    }

    async fn prepare_causal_mutation(&self) -> Result<()> {
        bail!("recovery test must not admit mutations")
    }

    async fn causal_transcript_batch(&self) -> Result<crate::CausalTranscriptBatch> {
        bail!("recovery must use the exact batch persisted in the cache")
    }

    async fn acknowledge_causal_transcript(
        &self,
        _batch: &crate::CausalTranscriptBatch,
        _cursor: u64,
    ) -> Result<()> {
        self.causal_acks.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn flush_compatibility_transcript(
        &self,
        _operation: &CausalOperationReceipt,
    ) -> Result<Option<u64>> {
        self.compatibility_flushes.fetch_add(1, Ordering::SeqCst);
        Ok(Some(9))
    }
}

struct RecoveryServer {
    base_url: String,
    causal_calls: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for RecoveryServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl RecoveryServer {
    async fn start(causal: bool) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let causal_calls = Arc::new(AtomicUsize::new(0));
        let calls = causal_calls.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let calls = calls.clone();
                tokio::spawn(async move {
                    let request = read_recovery_request(&mut stream).await;
                    let first = request.lines().next().unwrap_or_default();
                    let body = if first.contains("/hi/pipefs/capabilities ") {
                        serde_json::json!({
                            "enabled": true,
                            "archive_version": crate::ARCHIVE_VERSION,
                            "transfer_modes": ["proxy"],
                            "maximum_revision_bytes": 1048576,
                            "maximum_workspace_bytes": 1048576,
                            "maximum_delta_chain": 20,
                            "transfer_expiry_seconds": 300,
                            "capabilities": if causal { serde_json::json!([crate::CAUSAL_COMMIT_CAPABILITY]) } else { serde_json::json!([]) },
                            "writer_protocols": if causal { serde_json::json!([2]) } else { serde_json::json!([1]) }
                        })
                    } else if first.contains("/operations/") && first.contains("/commit ") {
                        calls.fetch_add(1, Ordering::SeqCst);
                        let operation_id = request
                            .split("\r\n\r\n")
                            .nth(1)
                            .and_then(|body| serde_json::from_str::<serde_json::Value>(body).ok())
                            .and_then(|body| {
                                body.pointer("/operation/operation_id")?
                                    .as_str()
                                    .map(str::to_owned)
                            })
                            .unwrap_or_else(|| "missing".into());
                        serde_json::json!({
                            "head": null,
                            "manifest_digest": null,
                            "transcript_cursor": 9,
                            "operation_id": operation_id,
                            "replayed": calls.load(Ordering::SeqCst) > 1
                        })
                    } else {
                        serde_json::json!({
                            "session_id": "recovery-session",
                            "enabled": true,
                            "current_head": null,
                            "sequence": 0,
                            "manifest_digest": null,
                            "logical_size_bytes": 0,
                            "restore_chain": []
                        })
                    };
                    write_recovery_json(&mut stream, &body).await;
                });
            }
        });
        Self {
            base_url: format!("http://{address}"),
            causal_calls,
            task,
        }
    }
}

async fn read_recovery_request(stream: &mut tokio::net::TcpStream) -> String {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 2048];
    let mut total = None;
    loop {
        let read = stream.read(&mut buffer).await.unwrap_or(0);
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..read]);
        if total.is_none()
            && let Some(header_end) = bytes.windows(4).position(|part| part == b"\r\n\r\n")
        {
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .and_then(|value| value.parse::<usize>().ok())
                })
                .unwrap_or(0);
            total = Some(header_end + 4 + length);
        }
        if total.is_some_and(|total| bytes.len() >= total) {
            break;
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

async fn write_recovery_json(stream: &mut tokio::net::TcpStream, body: &serde_json::Value) {
    let body = body.to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).await.unwrap();
}

fn recovery_operation() -> CausalOperationReceipt {
    CausalOperationReceipt {
        operation_id: "operation-1".into(),
        idempotency_key: "idempotency-1".into(),
        binding_id: "binding-1".into(),
        binding_epoch: 3,
        replay_class: hi_workspace::ReplayClass::PureWorkspace,
        execution: hi_workspace::ExecutionReport::succeeded(None),
    }
}

async fn staged_operation_recovery(
    causal: bool,
) -> (
    tempfile::TempDir,
    PathBuf,
    PipeFsClient,
    PipeFsLease,
    crate::PipeFsRecoveryCache,
    RecoveryServer,
) {
    let server = RecoveryServer::start(causal).await;
    let temporary = tempfile::tempdir().unwrap();
    let original = temporary.path().join("original");
    let state = temporary.path().join("state");
    let cache_base = temporary.path().join("cache");
    fs::create_dir_all(&original).unwrap();
    fs::create_dir_all(&state).unwrap();
    let client =
        PipeFsClient::new(crate::PipeFsClientConfig::new(&server.base_url, "test-key")).unwrap();
    let lease = PipeFsLease {
        token: "lease-token".into(),
        generation: 7,
    };
    let workspace = PipeFsWorkspace::new(
        client.clone(),
        lease.clone(),
        PipeFsWorkspaceConfig {
            session_id: "recovery-session".into(),
            cache_scope: client.cache_scope(),
            original_workspace_root: original,
            original_state_root: state,
            cache_base: Some(cache_base.clone()),
        },
    )
    .unwrap();
    workspace.restore_existing().await.unwrap();
    let operation = recovery_operation();
    if causal {
        workspace
            .causal_checkpoint(
                operation,
                vec![crate::CausalTranscriptRecord {
                    record_id: 1,
                    client_record_id: "record-1".into(),
                    record_type: "workspace_execution".into(),
                    payload: serde_json::json!({"operation_id": "operation-1"}),
                }],
            )
            .await
            .unwrap();
    } else {
        workspace
            .checkpoint_for_compatibility_transcript(operation)
            .await
            .unwrap();
    }
    drop(workspace);
    let caches = crate::workspace::list_recovery_caches_at(
        &cache_base,
        &client.cache_scope(),
        "recovery-session",
    )
    .unwrap();
    assert_eq!(caches.len(), 1);
    (
        temporary,
        cache_base,
        client,
        lease,
        caches.into_iter().next().unwrap(),
        server,
    )
}

#[tokio::test]
async fn causal_recovery_replays_persisted_batch_and_clears_only_after_ack() {
    let (_temporary, cache_base, client, _old_lease, cache, server) =
        staged_operation_recovery(true).await;
    let lease = PipeFsLease {
        token: "recovery-lease".into(),
        generation: 8,
    };
    let transcript = RecoveryTranscript::new(lease.clone());
    let released = Cell::new(false);
    let cache_id = cache.id.clone();
    let receipt = retry_recovery_cache_from(
        &client,
        lease,
        "recovery-session",
        &cache_id,
        cache,
        Some(cache_base.clone()),
        transcript.clone(),
        |operation, revision, cursor| {
            assert_eq!(operation, &recovery_operation());
            assert_eq!(revision, None);
            assert_eq!(cursor, 9);
            released.set(true);
            Ok(())
        },
    )
    .await
    .unwrap();
    assert_eq!(receipt.revision_id, None);
    assert!(released.get());
    assert_eq!(transcript.causal_acks.load(Ordering::SeqCst), 1);
    assert_eq!(transcript.compatibility_flushes.load(Ordering::SeqCst), 0);
    assert_eq!(
        server.causal_calls.load(Ordering::SeqCst),
        1,
        "the retained server receipt must not blindly replay the operation"
    );
    assert!(
        crate::workspace::list_recovery_caches_at(
            &cache_base,
            &client.cache_scope(),
            "recovery-session"
        )
        .unwrap()
        .is_empty()
    );
}

#[tokio::test]
async fn compatibility_recovery_flushes_exact_operation_before_releasing_cache() {
    let (_temporary, cache_base, client, _old_lease, cache, _server) =
        staged_operation_recovery(false).await;
    let lease = PipeFsLease {
        token: "recovery-lease".into(),
        generation: 8,
    };
    let transcript = RecoveryTranscript::new(lease.clone());
    let cache_id = cache.id.clone();
    retry_recovery_cache_from(
        &client,
        lease,
        "recovery-session",
        &cache_id,
        cache,
        Some(cache_base.clone()),
        transcript.clone(),
        |operation, revision, cursor| {
            assert_eq!(operation, &recovery_operation());
            assert_eq!(revision, None);
            assert_eq!(cursor, 9);
            Ok(())
        },
    )
    .await
    .unwrap();
    assert_eq!(transcript.causal_acks.load(Ordering::SeqCst), 0);
    assert_eq!(transcript.compatibility_flushes.load(Ordering::SeqCst), 1);
    assert!(
        crate::workspace::list_recovery_caches_at(
            &cache_base,
            &client.cache_scope(),
            "recovery-session"
        )
        .unwrap()
        .is_empty()
    );
}

#[tokio::test]
async fn journal_release_failure_retains_complete_recovery_evidence() {
    let (_temporary, cache_base, client, _old_lease, cache, _server) =
        staged_operation_recovery(false).await;
    let lease = PipeFsLease {
        token: "recovery-lease".into(),
        generation: 8,
    };
    let transcript = RecoveryTranscript::new(lease.clone());
    let cache_id = cache.id.clone();
    let result = retry_recovery_cache_from(
        &client,
        lease,
        "recovery-session",
        &cache_id,
        cache,
        Some(cache_base.clone()),
        transcript,
        |_operation, _revision, _cursor| bail!("injected journal failure"),
    )
    .await;
    assert!(result.unwrap_err().to_string().contains("journal failure"));
    let caches = crate::workspace::list_recovery_caches_at(
        &cache_base,
        &client.cache_scope(),
        "recovery-session",
    )
    .unwrap();
    assert_eq!(caches.len(), 1);
    assert_eq!(caches[0].id, cache_id);
    let state = fs::read_to_string(caches[0].path.join("controller.json")).unwrap();
    assert!(state.contains("pending_compatibility"));
    assert!(state.contains("operation-1"));
    assert!(caches[0].path.join("recovery-required").is_file());
}

#[test]
fn preview_digest_changes_with_workspace_bytes() {
    let source = tempfile::tempdir().unwrap();
    fs::write(source.path().join("a.txt"), b"one").unwrap();
    let first = preview_import(source.path()).unwrap();
    fs::write(source.path().join("a.txt"), b"two").unwrap();
    let second = preview_import(source.path()).unwrap();
    assert_ne!(first.confirmation_digest, second.confirmation_digest);
    assert_eq!(second.entry_count, 1);
    assert_eq!(second.byte_count, 3);
}

#[test]
fn confirmation_is_checked_against_a_fresh_source_scan() {
    let source = tempfile::tempdir().unwrap();
    fs::write(source.path().join("a.txt"), b"before").unwrap();
    let preview = preview_import(source.path()).unwrap();
    fs::write(source.path().join("a.txt"), b"after").unwrap();

    let error = confirmed_import_preview(source.path(), &preview.confirmation_digest)
        .expect_err("changed source must invalidate the preview confirmation");
    assert!(error.to_string().contains("fresh preview"));
}

#[tokio::test]
async fn failure_before_import_publish_never_polls_remote_publish() {
    let published = Cell::new(false);
    let result = publish_import_after(
        || Err(anyhow::anyhow!("injected import boundary failure")),
        async {
            published.set(true);
            Ok(())
        },
    )
    .await;
    assert!(result.is_err());
    assert!(!published.get());
}
