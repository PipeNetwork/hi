use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Result, bail};
use async_trait::async_trait;
use hi_workspace::{
    AdmissionDeniedReason, EffectScope, ExecutionReport, InMemoryWorkspaceController,
    JobCompletion, JobKind, JobLimits, JobSealStatus, JobSpec, JobTerminal, MutationIntent,
    RecoveryStatus, SettlementStatus, WorkspaceController, WorkspaceState,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::watch;

use super::*;
use crate::{
    CausalTranscriptRecord, PipeFsClient, PipeFsClientConfig, PipeFsLease, PipeFsWorkspace,
    PipeFsWorkspaceConfig,
};

#[path = "controller_job_limit_tests.rs"]
mod job_limits;
#[path = "controller_job_recovery_tests.rs"]
mod job_recovery;
#[path = "controller_lease_loss_tests.rs"]
mod lease_loss;
#[path = "controller_restart_recovery_tests.rs"]
mod restart_recovery;
#[path = "controller_safety_tests.rs"]
mod safety;

struct FakeSession {
    lease: PipeFsLease,
    loss_tx: watch::Sender<PipeFsLeaseStatus>,
    fail_ack: AtomicBool,
    fail_preflight: AtomicBool,
    fail_compatibility_flush: AtomicBool,
    acknowledgements: AtomicUsize,
    preflights: AtomicUsize,
    compatibility_flushes: AtomicUsize,
}

impl FakeSession {
    fn new() -> Arc<Self> {
        let (loss_tx, _) = watch::channel(PipeFsLeaseStatus::Valid);
        Arc::new(Self {
            lease: PipeFsLease {
                token: "lease-token".into(),
                generation: 7,
            },
            loss_tx,
            fail_ack: AtomicBool::new(false),
            fail_preflight: AtomicBool::new(false),
            fail_compatibility_flush: AtomicBool::new(false),
            acknowledgements: AtomicUsize::new(0),
            preflights: AtomicUsize::new(0),
            compatibility_flushes: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl PipeFsSessionBridge for FakeSession {
    fn subscribe_lease_status(&self) -> watch::Receiver<PipeFsLeaseStatus> {
        self.loss_tx.subscribe()
    }

    async fn refresh_lease(&self) -> Result<PipeFsLease> {
        match *self.loss_tx.borrow() {
            PipeFsLeaseStatus::Valid => {}
            PipeFsLeaseStatus::Uncertain => bail!("test lease authority is uncertain"),
            PipeFsLeaseStatus::Lost => bail!("lease_lost: test lease was replaced"),
        }
        Ok(self.lease.clone())
    }

    async fn prepare_causal_mutation(&self) -> Result<()> {
        self.preflights.fetch_add(1, Ordering::SeqCst);
        if self.fail_preflight.load(Ordering::SeqCst) {
            bail!("stable transcript prefix is not acknowledged");
        }
        Ok(())
    }

    async fn causal_transcript_batch(&self) -> Result<CausalTranscriptBatch> {
        Ok(CausalTranscriptBatch {
            records: vec![CausalTranscriptRecord {
                record_id: 4,
                client_record_id: "record-4".into(),
                record_type: "tool".into(),
                payload: serde_json::json!({"result": "ok"}),
            }],
        })
    }

    async fn acknowledge_causal_transcript(
        &self,
        _batch: &CausalTranscriptBatch,
        _cursor: u64,
    ) -> Result<()> {
        if self.fail_ack.load(Ordering::SeqCst) {
            bail!("transcript outbox acknowledgement failed");
        }
        self.acknowledgements.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn flush_compatibility_transcript(
        &self,
        _operation: &CausalOperationReceipt,
    ) -> Result<Option<u64>> {
        if self.fail_compatibility_flush.load(Ordering::SeqCst) {
            bail!("compatibility transcript flush failed");
        }
        self.compatibility_flushes.fetch_add(1, Ordering::SeqCst);
        Ok(Some(9))
    }
}

struct FakeServer {
    base_url: String,
    causal_calls: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for FakeServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl FakeServer {
    async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let task_calls = calls.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let calls = task_calls.clone();
                tokio::spawn(async move {
                    let request = read_request(&mut stream).await;
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
                            "capabilities": [crate::CAUSAL_COMMIT_CAPABILITY],
                            "writer_protocols": [2]
                        })
                    } else if first.contains("/operations/") && first.contains("/commit ") {
                        calls.fetch_add(1, Ordering::SeqCst);
                        let operation_id = request
                            .split("\r\n\r\n")
                            .nth(1)
                            .and_then(|body| serde_json::from_str::<serde_json::Value>(body).ok())
                            .and_then(|body| {
                                body.pointer("/operation/operation_id")
                                    .and_then(|value| value.as_str())
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
                        remote_state()
                    };
                    write_json(&mut stream, &body).await;
                });
            }
        });
        Self {
            base_url: format!("http://{address}"),
            causal_calls: calls,
            task,
        }
    }
}

fn remote_state() -> serde_json::Value {
    serde_json::json!({
        "session_id": "session-1",
        "enabled": true,
        "current_head": null,
        "sequence": 0,
        "manifest_digest": null,
        "logical_size_bytes": 0,
        "restore_chain": []
    })
}

async fn read_request(stream: &mut tokio::net::TcpStream) -> String {
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

async fn write_json(stream: &mut tokio::net::TcpStream, body: &serde_json::Value) {
    let body = body.to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).await.unwrap();
}

async fn subject(
    allow_protocol_one_writes: bool,
) -> (
    tempfile::TempDir,
    PipeFsWorkspaceController,
    Arc<FakeSession>,
    FakeServer,
) {
    let server = FakeServer::start().await;
    let temporary = tempfile::tempdir().unwrap();
    let original = temporary.path().join("original");
    let state = temporary.path().join("state");
    std::fs::create_dir_all(&original).unwrap();
    std::fs::create_dir_all(&state).unwrap();
    let mut client_config = PipeFsClientConfig::new(&server.base_url, "test-key");
    client_config.request_timeout = Duration::from_secs(2);
    let client = PipeFsClient::new(client_config).unwrap();
    let workspace = PipeFsWorkspace::new(
        client.clone(),
        PipeFsLease {
            token: "lease-token".into(),
            generation: 7,
        },
        PipeFsWorkspaceConfig {
            session_id: "session-1".into(),
            cache_scope: client.cache_scope(),
            original_workspace_root: original,
            original_state_root: state,
            cache_base: Some(temporary.path().join("cache")),
        },
    )
    .unwrap();
    let activation = workspace.enable().await.unwrap();
    let session = FakeSession::new();
    let controller = PipeFsWorkspaceController::new(
        workspace,
        session.clone(),
        PipeFsControllerConfig {
            workspace_id: "workspace-1".into(),
            session_id: "session-1".into(),
            writer_protocol: activation.writer_protocol,
            causal_commit_available: activation.causal_commit_available,
            writes_available: activation.writes_available,
            workspace_root: activation.workspace_root,
            state_root: activation.state_root,
            epoch: 3,
            allow_protocol_one_writes,
        },
    )
    .await;
    (temporary, controller, session, server)
}

async fn compatibility_controller(
    source: &PipeFsWorkspaceController,
    session: Arc<FakeSession>,
) -> PipeFsWorkspaceController {
    let binding = source.binding();
    PipeFsWorkspaceController::new(
        source.inner.workspace.clone(),
        session,
        PipeFsControllerConfig {
            workspace_id: binding.workspace_id,
            session_id: "session-1".into(),
            writer_protocol: 1,
            causal_commit_available: false,
            writes_available: true,
            workspace_root: binding.workspace_root,
            state_root: binding.state_root,
            epoch: binding.epoch.saturating_add(1),
            allow_protocol_one_writes: true,
        },
    )
    .await
}

#[tokio::test]
async fn causal_settlement_uses_one_atomic_operation_and_returns_remote_cursor() {
    let (_temporary, controller, session, server) = subject(false).await;
    let permit = controller
        .begin(MutationIntent::workspace("tool"))
        .await
        .unwrap();
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;

    assert_eq!(outcome.status, SettlementStatus::NoChange);
    assert_eq!(server.causal_calls.load(Ordering::SeqCst), 1);
    assert_eq!(session.acknowledgements.load(Ordering::SeqCst), 1);
    assert_eq!(outcome.receipt.unwrap().transcript_cursor, Some(9));
    assert_eq!(controller.status().state, WorkspaceState::Ready);
}

#[tokio::test]
async fn durable_indeterminate_execution_remains_admission_closed_until_reconciled() {
    let (_temporary, controller, _session, server) = subject(false).await;
    let permit = controller
        .begin(MutationIntent::workspace("opaque interrupted tool"))
        .await
        .unwrap();
    let outcome = controller
        .settle(
            permit,
            ExecutionReport {
                disposition: hi_workspace::ExecutionDisposition::Indeterminate,
                workspace_may_have_changed: true,
                external_effect_may_have_occurred: true,
                content_digest: None,
                changed_paths: Vec::new(),
                artifacts: Vec::new(),
                detail: Some("tool response was lost after dispatch".into()),
            },
        )
        .await;

    assert_eq!(outcome.status, SettlementStatus::Indeterminate);
    assert!(outcome.receipt.is_some());
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
    assert!(
        controller
            .begin(MutationIntent::workspace("blocked while ambiguous"))
            .await
            .is_err()
    );

    let recovered = controller.reconcile(outcome.recovery_id.unwrap()).await;
    assert_eq!(recovered.status, RecoveryStatus::Recovered);
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    assert_eq!(server.causal_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn transcript_ack_failure_preserves_evidence_and_reconcile_finishes_once() {
    let (temporary, controller, session, server) = subject(false).await;
    session.fail_ack.store(true, Ordering::SeqCst);
    let permit = controller
        .begin(MutationIntent::workspace("tool"))
        .await
        .unwrap();
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;

    assert_eq!(outcome.status, SettlementStatus::TranscriptPending);
    assert_eq!(controller.status().state, WorkspaceState::TranscriptPending);
    assert!(outcome.recovery_id.is_some());
    assert_eq!(server.causal_calls.load(Ordering::SeqCst), 1);
    let controller_state = find_file(temporary.path(), "controller.json");
    let persisted = std::fs::read_to_string(controller_state).unwrap();
    assert!(persisted.contains("pending_causal"));
    assert!(find_file(temporary.path(), "recovery-required").is_file());

    session.fail_ack.store(false, Ordering::SeqCst);
    let recovered = controller.reconcile(outcome.recovery_id.unwrap()).await;
    assert_eq!(recovered.status, RecoveryStatus::Recovered);
    assert_eq!(server.causal_calls.load(Ordering::SeqCst), 1);
    assert_eq!(controller.status().state, WorkspaceState::Ready);
}

#[tokio::test]
async fn restart_restores_pending_causal_operation_as_typed_recovery() {
    let (temporary, controller, session, server) = subject(false).await;
    session.fail_ack.store(true, Ordering::SeqCst);
    let permit = controller
        .begin(MutationIntent::workspace("tool"))
        .await
        .unwrap();
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(outcome.status, SettlementStatus::TranscriptPending);
    let controller_state = find_file(temporary.path(), "controller.json");
    let cache_root = controller_state.parent().unwrap().to_path_buf();
    drop(controller);

    let mut client_config = PipeFsClientConfig::new(&server.base_url, "test-key");
    client_config.request_timeout = Duration::from_secs(2);
    let client = PipeFsClient::new(client_config).unwrap();
    let workspace = PipeFsWorkspace::new(
        client.clone(),
        session.lease.clone(),
        PipeFsWorkspaceConfig {
            session_id: "session-1".into(),
            cache_scope: client.cache_scope(),
            original_workspace_root: temporary.path().join("original"),
            original_state_root: temporary.path().join("state"),
            cache_base: Some(temporary.path().join("cache")),
        },
    )
    .unwrap();
    let activation = workspace.restore_existing().await.unwrap();
    assert!(activation.workspace_root.starts_with(&cache_root));
    let controller = PipeFsWorkspaceController::new(
        workspace,
        session.clone(),
        PipeFsControllerConfig {
            workspace_id: "workspace-1".into(),
            session_id: "session-1".into(),
            writer_protocol: activation.writer_protocol,
            causal_commit_available: activation.causal_commit_available,
            writes_available: activation.writes_available,
            workspace_root: activation.workspace_root,
            state_root: activation.state_root,
            epoch: 4,
            allow_protocol_one_writes: false,
        },
    )
    .await;

    let status = controller.status();
    assert_eq!(status.state, WorkspaceState::TranscriptPending);
    assert!(status.recovery_id.is_some());
    assert!(
        controller
            .begin(MutationIntent::workspace("blocked"))
            .await
            .is_err()
    );
    session.fail_ack.store(false, Ordering::SeqCst);
    let recovered = controller.reconcile(status.recovery_id.unwrap()).await;
    assert_eq!(recovered.status, RecoveryStatus::Recovered);
    assert_eq!(server.causal_calls.load(Ordering::SeqCst), 1);
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    assert!(!cache_root.join("recovery-required").exists());
    let persisted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(controller_state).unwrap()).unwrap();
    assert!(persisted["pending_causal"].is_null());
}

fn find_file(root: &std::path::Path, name: &str) -> std::path::PathBuf {
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory).unwrap().flatten() {
            let path = entry.path();
            if path.file_name().is_some_and(|value| value == name) {
                return path;
            }
            if path.is_dir() {
                pending.push(path);
            }
        }
    }
    panic!("{name} was not found under {}", root.display());
}

#[tokio::test]
async fn pushed_lease_loss_immediately_closes_admission() {
    let (_temporary, controller, session, _server) = subject(false).await;
    session.loss_tx.send_replace(PipeFsLeaseStatus::Lost);
    tokio::time::timeout(Duration::from_secs(2), async {
        let mut status = controller.subscribe();
        while status.borrow().state != WorkspaceState::LeaseLost {
            status.changed().await.unwrap();
        }
    })
    .await
    .unwrap();
    assert!(
        controller
            .begin(MutationIntent::workspace("blocked"))
            .await
            .is_err()
    );
    assert!(controller.status().recovery_id.is_none());
}

#[tokio::test]
async fn jobs_from_the_active_operation_are_admitted_during_mutation() {
    let (_temporary, controller, _session, _server) = subject(false).await;
    let permit = controller
        .begin(MutationIntent::workspace("mixed tool batch"))
        .await
        .unwrap();
    let parent = permit.record().operation_id.clone();
    for (kind, effect_scope) in [
        (JobKind::ReadAgent, EffectScope::ReadOnly),
        (JobKind::WriteCandidate, EffectScope::CandidateOnly),
    ] {
        let job = controller
            .register_job(JobSpec {
                kind,
                effect_scope,
                name: format!("{kind:?}"),
                limits: JobLimits::default(),
                parent_operation: Some(parent.clone()),
            })
            .await
            .unwrap();
        let sealed = controller
            .seal_job(
                job.job_id,
                JobTerminal {
                    completion: JobCompletion::Failed,
                    detail: None,
                    artifacts: Vec::new(),
                },
            )
            .await;
        assert_eq!(sealed.status, hi_workspace::JobSealStatus::Sealed);
    }
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(outcome.status, SettlementStatus::NoChange);
}

#[tokio::test]
async fn pipefs_disables_live_writers_but_keeps_isolated_candidates() {
    let (_temporary, controller, _session, _server) = subject(false).await;
    let capabilities = controller.capabilities();
    assert!(capabilities.candidate_apply);
    assert!(!capabilities.background_writers);

    let permit = controller
        .begin(MutationIntent::workspace("mixed tool batch"))
        .await
        .unwrap();
    let parent = permit.record().operation_id.clone();
    let live_writer = JobSpec {
        kind: JobKind::Process,
        effect_scope: EffectScope::LiveWriter,
        name: "unsafe live writer".into(),
        limits: JobLimits::default(),
        parent_operation: Some(parent.clone()),
    };
    for _ in 0..2 {
        let denied = controller
            .register_job(live_writer.clone())
            .await
            .unwrap_err();
        assert_eq!(denied.reason, AdmissionDeniedReason::CapabilityUnavailable);
    }
    let foreign = controller
        .register_job(JobSpec {
            kind: JobKind::WriteCandidate,
            effect_scope: EffectScope::CandidateOnly,
            name: "foreign candidate".into(),
            limits: JobLimits::default(),
            parent_operation: Some("foreign-operation".into()),
        })
        .await
        .unwrap_err();
    assert_eq!(foreign.reason, AdmissionDeniedReason::NotReady);
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(outcome.status, SettlementStatus::NoChange);
}

#[tokio::test]
async fn foreground_admission_rejects_an_unsettled_live_writer_projection() {
    let (_temporary, controller, _session, _server) = subject(false).await;
    let fence = controller.inner.jobs.fence();
    controller
        .inner
        .jobs
        .register(
            &fence,
            JobSpec {
                kind: JobKind::Process,
                effect_scope: EffectScope::LiveWriter,
                name: "restored live writer".into(),
                limits: JobLimits::default(),
                parent_operation: None,
            },
        )
        .unwrap();
    let denied = controller
        .begin(MutationIntent::workspace("foreground"))
        .await
        .unwrap_err();
    assert_eq!(denied.reason, AdmissionDeniedReason::ActiveWriter);
}

#[derive(Debug, PartialEq, Eq)]
struct NormalizedLifecycleTrace {
    checkpoints: Vec<(WorkspaceState, bool, usize)>,
    job_terminal: Option<hi_workspace::JobState>,
    settlement: SettlementStatus,
}

async fn no_change_trace(controller: &dyn WorkspaceController) -> NormalizedLifecycleTrace {
    let checkpoint = |controller: &dyn WorkspaceController| {
        let status = controller.status();
        (
            status.state,
            status.active_operation.is_some(),
            status.active_jobs.len(),
        )
    };
    let mut checkpoints = vec![checkpoint(controller)];
    let permit = controller
        .begin(MutationIntent::workspace("no-change script"))
        .await
        .unwrap();
    checkpoints.push(checkpoint(controller));
    let job = controller
        .register_job(JobSpec {
            kind: JobKind::ReadAgent,
            effect_scope: EffectScope::ReadOnly,
            name: "read-side verification".into(),
            limits: JobLimits::default(),
            parent_operation: Some(permit.record().operation_id.clone()),
        })
        .await
        .unwrap();
    checkpoints.push(checkpoint(controller));
    let sealed = controller
        .seal_job(
            job.job_id,
            JobTerminal {
                completion: JobCompletion::Succeeded,
                detail: None,
                artifacts: Vec::new(),
            },
        )
        .await;
    assert_eq!(sealed.status, JobSealStatus::Sealed);
    checkpoints.push(checkpoint(controller));
    let settlement = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await
        .status;
    checkpoints.push(checkpoint(controller));
    NormalizedLifecycleTrace {
        checkpoints,
        job_terminal: sealed.state,
        settlement,
    }
}

#[tokio::test]
async fn local_and_fake_pipefs_no_change_traces_are_equivalent() {
    let local_root = tempfile::tempdir().unwrap();
    let local = InMemoryWorkspaceController::new_local(
        "local-workspace",
        local_root.path(),
        local_root.path().join("state"),
    );
    let local_trace = no_change_trace(&local).await;
    let (_temporary, pipefs, _session, _server) = subject(false).await;
    let pipefs_trace = no_change_trace(&pipefs).await;
    assert_eq!(local_trace, pipefs_trace);
}

#[test]
fn protocol_one_requires_the_explicit_current_client_compatibility_adapter() {
    let config = PipeFsControllerConfig {
        workspace_id: "workspace".into(),
        session_id: "session".into(),
        writer_protocol: 1,
        causal_commit_available: false,
        writes_available: true,
        workspace_root: "/work".into(),
        state_root: "/state".into(),
        epoch: 0,
        allow_protocol_one_writes: false,
    };
    assert_eq!(config.writer_mode(), PipeFsWriterMode::ReadOnly);
    let mut server_read_only = config.clone();
    server_read_only.writer_protocol = 2;
    server_read_only.causal_commit_available = true;
    server_read_only.writes_available = false;
    assert_eq!(server_read_only.writer_mode(), PipeFsWriterMode::ReadOnly);
    let mut legacy = config;
    legacy.allow_protocol_one_writes = true;
    assert_eq!(legacy.writer_mode(), PipeFsWriterMode::Compatibility);
}

#[tokio::test]
async fn compatibility_flush_ambiguity_blocks_until_typed_recovery() {
    let (temporary, source, session, server) = subject(false).await;
    let controller = compatibility_controller(&source, session.clone()).await;
    let capabilities = controller.capabilities();
    assert!(!capabilities.causal_commit);
    assert!(capabilities.candidate_apply);
    assert!(!capabilities.background_writers);

    session
        .fail_compatibility_flush
        .store(true, Ordering::SeqCst);
    let permit = controller
        .begin(MutationIntent::workspace("legacy foreground"))
        .await
        .unwrap();
    let candidate = controller
        .register_job(JobSpec {
            kind: JobKind::WriteCandidate,
            effect_scope: EffectScope::CandidateOnly,
            name: "isolated compatibility candidate".into(),
            limits: JobLimits::default(),
            parent_operation: Some(permit.record().operation_id.clone()),
        })
        .await
        .unwrap();
    assert_eq!(
        controller
            .seal_job(
                candidate.job_id,
                JobTerminal {
                    completion: JobCompletion::Failed,
                    detail: None,
                    artifacts: Vec::new(),
                },
            )
            .await
            .status,
        JobSealStatus::Sealed
    );
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(outcome.status, SettlementStatus::TranscriptPending);
    assert_eq!(controller.status().state, WorkspaceState::TranscriptPending);
    let pending_status = source.inner.workspace.status().await;
    assert!(pending_status.transcript_pending);
    let controller_state = find_file(temporary.path(), "controller.json");
    let persisted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&controller_state).unwrap()).unwrap();
    assert!(persisted["pending_compatibility"].is_object());
    assert!(
        controller
            .begin(MutationIntent::workspace("blocked"))
            .await
            .is_err()
    );
    assert_eq!(server.causal_calls.load(Ordering::SeqCst), 0);

    // Simulate a process crash after workspace CAS but before transcript
    // acknowledgement. A fresh typed controller must reconstruct recovery
    // from the PipeFS cache instead of dead-ending on the old journal state.
    drop(controller);
    let restarted = compatibility_controller(&source, session.clone()).await;
    assert_eq!(restarted.status().state, WorkspaceState::TranscriptPending);

    session
        .fail_compatibility_flush
        .store(false, Ordering::SeqCst);
    let recovered = restarted
        .reconcile(restarted.status().recovery_id.unwrap())
        .await;
    assert_eq!(recovered.status, RecoveryStatus::Recovered);
    assert_eq!(session.compatibility_flushes.load(Ordering::SeqCst), 1);
    assert_eq!(restarted.status().state, WorkspaceState::Ready);
    let recovered_status = source.inner.workspace.status().await;
    assert!(!recovered_status.transcript_pending);
    let persisted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(controller_state).unwrap()).unwrap();
    assert!(persisted["pending_compatibility"].is_null());
}
