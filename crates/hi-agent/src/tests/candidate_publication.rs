use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use hi_workspace::{
    CandidateRoute, CandidateVerification, ExecutionDisposition, ExecutionReport,
    InMemoryWorkspaceController, JobId, JobState, WorkspaceController, WorkspaceState,
};

use super::common::{agent, config};

struct CandidateSession {
    attempts: Arc<Mutex<Vec<crate::WorkspaceTranscriptExecution>>>,
    fail_stage: bool,
}

impl crate::SessionSink for CandidateSession {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> Result<()> {
        Ok(())
    }

    fn stage_workspace_execution(
        &mut self,
        record: &crate::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        self.attempts.lock().unwrap().push(record.clone());
        if self.fail_stage {
            anyhow::bail!("candidate stage unavailable");
        }
        Ok(())
    }

    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> Result<()> {
        Ok(())
    }
}

struct CandidateFixture {
    subject: crate::Agent,
    controller: Arc<InMemoryWorkspaceController>,
    registry: Arc<hi_tools::BackgroundTaskRegistry>,
    task_id: String,
    root: PathBuf,
    artifact_path: PathBuf,
    artifact_uri: String,
    candidate_id: String,
    workspace_job_id: JobId,
    attempts: Arc<Mutex<Vec<crate::WorkspaceTranscriptExecution>>>,
}

async fn ready_candidate(fail_stage: bool, destination_verifier: &str) -> CandidateFixture {
    ready_candidate_with_contract(
        fail_stage,
        vec![hi_workspace::CandidateDestinationVerifier {
            name: "destination test".into(),
            command: destination_verifier.into(),
            timeout_ms: 5_000,
        }],
        5_000,
    )
    .await
}

async fn ready_candidate_with_contract(
    fail_stage: bool,
    destination_verification: Vec<hi_workspace::CandidateDestinationVerifier>,
    destination_verification_budget_ms: u64,
) -> CandidateFixture {
    let mut cfg = config();
    cfg.subagents.write_subagents = crate::WriteSubagentPolicy::On;
    cfg.harness.features.candidate_jobs_v2 = true;
    cfg.harness.jobs.verifier_timeout = Duration::from_millis(destination_verification_budget_ms);
    let root = cfg.paths.workspace_root.clone();
    let state_root = cfg.paths.state_root.clone();
    git(&root, &["init", "--quiet"]);
    std::fs::write(root.join("source.txt"), "source\n").unwrap();
    git(&root, &["add", "source.txt"]);
    git(
        &root,
        &[
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@invalid",
            "commit",
            "--quiet",
            "-m",
            "base",
        ],
    );

    let mut subject = agent(Vec::new(), cfg);
    let controller = Arc::new(InMemoryWorkspaceController::new_pipefs(
        "candidate-workspace",
        "candidate-session",
        2,
        true,
        &root,
        &state_root,
    ));
    subject
        .install_workspace_controller(controller.clone())
        .unwrap();
    let attempts = Arc::new(Mutex::new(Vec::new()));
    subject.set_session(Box::new(CandidateSession {
        attempts: attempts.clone(),
        fail_stage,
    }));

    let registry = subject.background_task_registry();
    let candidate = hi_tools::candidate_workspace::CandidateWorkspace::create(
        &root,
        &state_root,
        &state_root.join("candidate-owner"),
    )
    .unwrap();
    std::fs::write(candidate.root().join("candidate.txt"), "candidate\n").unwrap();
    let binding = subject.workspace_controller_binding();
    let worker_registry = registry.clone();
    let artifact_state = state_root.clone();
    let (id_tx, id_rx) = tokio::sync::oneshot::channel::<String>();
    let task_id = registry
        .spawn(
            "candidate publication",
            "general-purpose",
            Box::new(move || {
                Box::pin(async move {
                    let task_id = id_rx.await.unwrap();
                    let job_id = worker_registry
                        .candidate_workspace_job_id(&task_id)
                        .await
                        .unwrap();
                    let sealed = candidate
                        .seal_verified(hi_tools::candidate_workspace::CandidateSealContext {
                            job_id: hi_workspace::JobId::new(job_id),
                            binding,
                            route: CandidateRoute {
                                provider: "test".into(),
                                model: "test-model".into(),
                                actual_model_revision: Some("test-revision".into()),
                                capability_digest: "blake3:test-capabilities".into(),
                            },
                            verification: vec![CandidateVerification {
                                name: "test verifier".into(),
                                passed: true,
                                verifier_digest: "blake3:test-verifier".into(),
                                detail: None,
                                artifacts: Vec::new(),
                            }],
                            destination_verification,
                            destination_verification_budget_ms,
                        })
                        .unwrap();
                    let sealed =
                        hi_tools::candidate_workspace::PersistedDetachedCandidate::persist(
                            sealed,
                            &artifact_state,
                        )
                        .unwrap();
                    worker_registry.publish_candidate(&task_id, sealed).unwrap();
                    hi_tools::BackgroundTaskOutcome {
                        id: String::new(),
                        description: String::new(),
                        subagent_type: "general-purpose".into(),
                        state: hi_tools::BackgroundTaskState::Completed,
                        output: "candidate prepared".into(),
                        applied: false,
                        changed_files: vec!["candidate.txt".into()],
                    }
                })
            }),
        )
        .await
        .unwrap();
    id_tx.send(task_id.clone()).unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !registry.candidate_is_ready(&task_id) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let persisted =
        hi_tools::candidate_workspace::PersistedDetachedCandidate::discover(&state_root).unwrap();
    assert_eq!(persisted.len(), 1);
    let artifact_path = persisted[0].path().to_path_buf();
    let artifact_uri = persisted[0].artifact.uri.clone();
    let candidate_id = persisted[0].candidate.candidate_id.to_string();
    let workspace_job_id = persisted[0].candidate.job_id.clone();
    CandidateFixture {
        subject,
        controller,
        registry,
        task_id,
        root,
        artifact_path,
        artifact_uri,
        candidate_id,
        workspace_job_id,
        attempts,
    }
}

#[tokio::test]
async fn candidate_success_follows_exact_pipefs_stage_and_workspace_receipt() {
    let mut fixture = ready_candidate(false, "true").await;

    fixture
        .subject
        .settle_ready_candidates_at_boundary()
        .await
        .unwrap();
    let outcome = wait_terminal(&fixture.registry, &fixture.task_id).await;

    assert_eq!(outcome.state, hi_tools::BackgroundTaskState::Completed);
    assert!(outcome.applied);
    assert_eq!(fixture.controller.status().state, WorkspaceState::Ready);
    assert_eq!(
        std::fs::read_to_string(fixture.root.join("candidate.txt")).unwrap(),
        "candidate\n"
    );
    assert!(!fixture.artifact_path.exists());
    let attempts = fixture.attempts.lock().unwrap();
    assert_eq!(attempts.len(), 1);
    let record = &attempts[0];
    assert_eq!(
        record.execution.disposition,
        ExecutionDisposition::Succeeded
    );
    assert_eq!(
        record.execution.changed_paths,
        [PathBuf::from("candidate.txt")]
    );
    assert!(record.execution.content_digest.is_some());
    assert!(!record.execution.external_effect_may_have_occurred);
    assert!(
        record
            .execution
            .artifacts
            .iter()
            .any(|artifact| artifact.uri == fixture.artifact_uri)
    );
    let hi_ai::Content::ToolCall {
        name, arguments, ..
    } = &record.assistant_content[0]
    else {
        panic!("expected synthetic candidate apply call");
    };
    assert_eq!(name, "apply_background_candidate");
    let arguments: serde_json::Value = serde_json::from_str(arguments).unwrap();
    assert_eq!(arguments["task_id"], fixture.task_id);
    assert_eq!(arguments["candidate_id"], fixture.candidate_id);
    let result: ExecutionReport = serde_json::from_str(&record.calls[0].result).unwrap();
    assert_eq!(result, record.execution);
}

#[tokio::test]
async fn failed_live_destination_verifier_rolls_back_and_never_publishes_success() {
    let mut fixture = ready_candidate(
        false,
        "printf 'verifier-corruption\\n' > source.txt; exit 17",
    )
    .await;

    fixture
        .subject
        .settle_ready_candidates_at_boundary()
        .await
        .unwrap();
    let outcome = wait_terminal(&fixture.registry, &fixture.task_id).await;

    assert_eq!(outcome.state, hi_tools::BackgroundTaskState::Failed);
    assert!(!outcome.applied);
    assert_eq!(fixture.controller.status().state, WorkspaceState::Ready);
    assert_eq!(
        std::fs::read_to_string(fixture.root.join("source.txt")).unwrap(),
        "source\n",
        "verifier writes must be included in the sealed rollback"
    );
    assert!(
        !fixture.root.join("candidate.txt").exists(),
        "candidate postimages must be rolled back after destination rejection"
    );
    assert!(
        !fixture.artifact_path.exists(),
        "known-clean verifier rejection is terminal and needs no recovery artifact"
    );

    let attempts = fixture.attempts.lock().unwrap();
    assert_eq!(attempts.len(), 1);
    let execution = &attempts[0].execution;
    assert_eq!(execution.disposition, ExecutionDisposition::Failed);
    assert!(!execution.workspace_may_have_changed);
    assert!(execution.changed_paths.is_empty());
    assert!(
        execution
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("applied candidate changes were rolled back"))
    );
}

#[cfg(unix)]
#[tokio::test]
async fn destination_pipeline_uses_one_shared_total_deadline() {
    let mut fixture = ready_candidate_with_contract(
        false,
        vec![
            hi_workspace::CandidateDestinationVerifier {
                name: "fast stage".into(),
                command: "true".into(),
                timeout_ms: 5_000,
            },
            hi_workspace::CandidateDestinationVerifier {
                name: "slow stage".into(),
                command: "sleep 30".into(),
                timeout_ms: 5_000,
            },
        ],
        100,
    )
    .await;
    let started = std::time::Instant::now();

    fixture
        .subject
        .settle_ready_candidates_at_boundary()
        .await
        .unwrap();
    let elapsed = started.elapsed();
    let outcome = wait_terminal(&fixture.registry, &fixture.task_id).await;

    assert_eq!(outcome.state, hi_tools::BackgroundTaskState::Failed);
    assert!(!fixture.root.join("candidate.txt").exists());
    assert!(
        elapsed < Duration::from_secs(3),
        "later stages must receive only the shared pipeline remainder, not their full per-stage timeout: {elapsed:?}"
    );
    let attempts = fixture.attempts.lock().unwrap();
    assert_eq!(
        attempts[0].execution.disposition,
        ExecutionDisposition::Failed
    );
    assert!(
        attempts[0]
            .execution
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("TimedOut") && detail.contains("rolled back"))
    );
}

#[cfg(unix)]
#[tokio::test]
async fn cancelled_destination_verifier_is_reaped_rolled_back_and_terminally_fenced() {
    let mut fixture = ready_candidate(
        false,
        "printf started > verifier-started; printf '%s' $$ > verifier-pid; sleep 30",
    )
    .await;
    let cancellation = crate::TurnCancellation::new();
    fixture.subject.turn_cancellation = Some(cancellation.clone());
    let foreground = fixture.subject.foreground_process_registry();
    let marker = fixture.root.join("verifier-started");

    let mut publication = Box::pin(fixture.subject.settle_ready_candidates_at_boundary());
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            tokio::select! {
                result = &mut publication => {
                    panic!("candidate publication completed before cancellation: {result:?}");
                }
                _ = tokio::time::sleep(Duration::from_millis(5)) => {
                    if marker.exists() {
                        break;
                    }
                }
            }
        }
    })
    .await
    .expect("destination verifier should start");
    cancellation.cancel();
    drop(publication);
    fixture.subject.turn_cancellation = None;

    fixture
        .subject
        .quiesce_abnormal_turn_processes()
        .await
        .expect("cancellation must await publication through process reap and rollback");
    fixture
        .subject
        .workspace_coordination
        .abandon_active()
        .unwrap();
    let outcome = wait_terminal(&fixture.registry, &fixture.task_id).await;

    assert_eq!(outcome.state, hi_tools::BackgroundTaskState::Failed);
    assert!(!outcome.applied);
    assert_eq!(
        fixture.controller.job_state(&fixture.workspace_job_id),
        Some(JobState::RecoveryRequired),
        "a dropped publication caller must receive one fail-closed terminal job state"
    );
    assert_eq!(foreground.active_count(), 0, "verifier must be reaped");
    assert!(!fixture.root.join("candidate.txt").exists());
    assert!(!fixture.root.join("verifier-started").exists());
    assert!(!fixture.root.join("verifier-pid").exists());
    assert!(
        fixture.artifact_path.exists(),
        "cancellation recovery must retain the sealed candidate artifact"
    );
}

#[tokio::test]
async fn candidate_stage_failure_keeps_artifact_and_enters_recovery() {
    let mut fixture = ready_candidate(true, "true").await;

    let error = fixture
        .subject
        .settle_ready_candidates_at_boundary()
        .await
        .unwrap_err();
    let outcome = wait_terminal(&fixture.registry, &fixture.task_id).await;

    assert!(error.to_string().contains("candidate transcript staging"));
    assert_ne!(outcome.state, hi_tools::BackgroundTaskState::Completed);
    assert!(!outcome.applied);
    assert_eq!(
        fixture.controller.status().state,
        WorkspaceState::RecoveryRequired
    );
    assert!(fixture.root.join("candidate.txt").is_file());
    assert!(fixture.artifact_path.is_file());
    let attempts = fixture.attempts.lock().unwrap();
    assert_eq!(attempts.len(), 1);
    assert_eq!(
        attempts[0].execution.disposition,
        ExecutionDisposition::Succeeded,
        "the sink observes the actual apply result before its own failure makes settlement indeterminate"
    );
}

#[tokio::test]
async fn candidate_lifecycle_failure_is_staged_and_settled_as_failed() {
    let mut fixture = ready_candidate(false, "true").await;
    // Put the workspace job one step ahead of the candidate queue. The parent
    // still owns the candidate, but its normal Merging transition now fails
    // before any workspace byte can be applied.
    fixture
        .registry
        .transition_candidate(
            &fixture.task_id,
            hi_tools::BackgroundCandidateTransition::Merging,
            None,
        )
        .await
        .unwrap();

    let error = fixture
        .subject
        .settle_ready_candidates_at_boundary()
        .await
        .unwrap_err();
    let outcome = wait_terminal(&fixture.registry, &fixture.task_id).await;

    assert!(
        error
            .to_string()
            .contains("starting candidate merge lifecycle")
    );
    assert_eq!(outcome.state, hi_tools::BackgroundTaskState::Failed);
    assert!(!fixture.root.join("candidate.txt").exists());
    assert_eq!(fixture.controller.status().state, WorkspaceState::Ready);
    assert!(!fixture.artifact_path.exists());
    let attempts = fixture.attempts.lock().unwrap();
    assert_eq!(attempts.len(), 1);
    let execution = &attempts[0].execution;
    assert_eq!(execution.disposition, ExecutionDisposition::Failed);
    assert!(!execution.workspace_may_have_changed);
    assert!(execution.changed_paths.is_empty());
    assert!(execution.content_digest.is_none());
}

async fn wait_terminal(
    registry: &hi_tools::BackgroundTaskRegistry,
    task_id: &str,
) -> hi_tools::BackgroundTaskOutcome {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let outcome = registry.poll(task_id, Duration::ZERO).await.unwrap();
            if outcome.state != hi_tools::BackgroundTaskState::Running {
                break outcome;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap()
}

fn git(root: &Path, args: &[&str]) {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
}
