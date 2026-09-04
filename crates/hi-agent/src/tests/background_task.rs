//! Tests for the background subagent task system (`task`/`get_task_output`/
//! `wait_tasks`/`kill_task`).

use super::common::*;
use super::*;

fn bg_config() -> AgentConfig {
    let mut cfg = config();
    cfg.subagents.explore_subagents = true;
    cfg.harness.features.candidate_jobs_v2 = true;
    cfg
}

struct BackgroundDropFlag(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Drop for BackgroundDropFlag {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[tokio::test]
async fn dropping_agent_aborts_its_active_background_tasks() {
    let agent = agent(Vec::new(), bg_config());
    // Retain the same kind of observer handle used by frontends while a turn
    // borrows the Agent. Agent drop must cancel its tasks even though this Arc
    // keeps the registry allocation alive.
    let registry = agent.background_task_registry();
    let started = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let started_in_task = started.clone();
    let dropped_in_task = dropped.clone();
    let task_id = registry
        .spawn(
            "agent-owned",
            "delegate",
            Box::new(move || {
                Box::pin(async move {
                    let _drop_flag = BackgroundDropFlag(dropped_in_task);
                    started_in_task.store(true, std::sync::atomic::Ordering::SeqCst);
                    std::future::pending::<()>().await;
                    unreachable!("agent-owned background task survived forever")
                })
            }),
        )
        .await
        .unwrap();

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !started.load(std::sync::atomic::Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("background task should start");

    drop(agent);

    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !dropped.load(std::sync::atomic::Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping Agent must drop its active background task future");

    let outcome = registry
        .poll(&task_id, std::time::Duration::ZERO)
        .await
        .expect("retained observer should still see the terminal task");
    assert_eq!(outcome.state, hi_tools::BackgroundTaskState::Cancelled);

    let spawn_after_owner_drop = registry
        .spawn(
            "orphan",
            "delegate",
            Box::new(|| Box::pin(std::future::pending::<hi_tools::BackgroundTaskOutcome>())),
        )
        .await;
    assert!(
        spawn_after_owner_drop.is_err(),
        "an observer handle must not restart work after Agent drop"
    );
}

#[test]
fn task_tool_spec_exists_and_is_not_in_global_set() {
    assert!(!hi_tools::TOOL_SPECS.iter().any(|t| t.name == "task"));
    assert_eq!(hi_tools::task_tool_spec().name, "task");
    assert_eq!(
        hi_tools::get_task_output_tool_spec().name,
        "get_task_output"
    );
    assert_eq!(hi_tools::wait_tasks_tool_spec().name, "wait_tasks");
    assert_eq!(hi_tools::kill_task_tool_spec().name, "kill_task");
}

#[test]
fn task_tools_are_in_catalog() {
    assert!(hi_tools::is_known_tool("task"));
    assert!(hi_tools::is_known_tool("get_task_output"));
    assert!(hi_tools::is_known_tool("wait_tasks"));
    assert!(hi_tools::is_known_tool("kill_task"));
}

#[test]
fn task_tools_advertised_for_top_level_agent() {
    let agent = agent(Vec::new(), bg_config());
    let tools = agent.request_tools_for(hi_ai::ToolMode::Auto);
    assert!(
        tools.iter().any(|t| t.name == "task"),
        "task tool should be advertised for a top-level agent"
    );
    assert!(
        tools.iter().any(|t| t.name == "get_task_output"),
        "get_task_output should be advertised"
    );
    assert!(
        tools.iter().any(|t| t.name == "wait_tasks"),
        "wait_tasks should be advertised"
    );
    assert!(
        tools.iter().any(|t| t.name == "kill_task"),
        "kill_task should be advertised"
    );
}

#[test]
fn subagent_never_gets_task_tools() {
    let mut cfg = bg_config();
    cfg.subagents.is_subagent = true;
    let agent = agent(Vec::new(), cfg);
    for mode in [hi_ai::ToolMode::Auto, hi_ai::ToolMode::ReadOnly] {
        assert!(
            !agent
                .request_tools_for(mode)
                .iter()
                .any(|t| t.name == "task"),
            "a subagent must never see the task tool (depth cap)"
        );
    }
}

#[tokio::test]
async fn handle_task_missing_prompt_fails() {
    let mut agent = agent(Vec::new(), bg_config());
    let mut ui = NullUi;
    let outcome = agent
        .handle_task(r#"{"description": "test", "prompt": ""}"#, &mut ui)
        .await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
}

#[tokio::test]
async fn handle_task_missing_description_fails() {
    let mut agent = agent(Vec::new(), bg_config());
    let mut ui = NullUi;
    let outcome = agent
        .handle_task(r#"{"description": "", "prompt": "do something"}"#, &mut ui)
        .await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
}

#[tokio::test]
async fn handle_task_unknown_subagent_type_fails() {
    let mut agent = agent(Vec::new(), bg_config());
    let mut ui = NullUi;
    let outcome = agent
        .handle_task(
            r#"{"description": "x", "prompt": "do something", "subagent_type": "wizard"}"#,
            &mut ui,
        )
        .await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
    assert!(
        outcome.content.contains("unknown subagent_type"),
        "got: {}",
        outcome.content
    );
}

#[tokio::test]
async fn typed_limits_configure_registry_and_candidate_rollout_admission() {
    let mut cfg = config();
    cfg.subagents.write_subagents = WriteSubagentPolicy::On;
    cfg.harness.jobs.max_active = 3;
    cfg.harness.jobs.max_preparations = 2;
    cfg.harness.jobs.queue_timeout = std::time::Duration::from_millis(17);
    let mut agent = agent(Vec::new(), cfg);
    assert_eq!(
        agent.background_task_registry().limits(),
        hi_tools::BackgroundTaskLimits {
            max_tasks: 3,
            max_concurrent_preparations: 2,
            queue_timeout: std::time::Duration::from_millis(17),
        }
    );

    agent
        .activate_pipefs_workspace_controller("candidate-gate-disabled", 1, false)
        .unwrap();

    let outcome = agent
        .handle_task(
            r#"{"description":"writer","prompt":"write a file","subagent_type":"general-purpose"}"#,
            &mut NullUi,
        )
        .await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Denied);
    assert!(outcome.content.contains("features.candidate_jobs_v2"));
}

#[test]
fn causal_pipefs_capability_requires_the_resolved_rollout_gate() {
    let disabled = agent(Vec::new(), config());
    disabled
        .activate_pipefs_workspace_controller("causal-disabled", 2, true)
        .unwrap();
    assert!(!disabled.workspace_controller_capabilities().causal_commit);

    let mut cfg = config();
    cfg.harness.features.pipefs_causal_commit_v1 = true;
    let enabled = agent(Vec::new(), cfg);
    enabled
        .activate_pipefs_workspace_controller("causal-enabled", 2, true)
        .unwrap();
    assert!(enabled.workspace_controller_capabilities().causal_commit);
}

#[tokio::test]
async fn pipefs_without_background_writers_still_admits_detached_candidates() {
    let mut cfg = bg_config();
    cfg.subagents.write_subagents = WriteSubagentPolicy::On;
    let mut agent = agent(Vec::new(), cfg);
    agent.set_workspace_durability(Some(std::sync::Arc::new(TestWorkspaceDurability)));
    agent
        .activate_pipefs_workspace_controller("candidate-test", 2, true)
        .unwrap();
    let mut ui = NullUi;
    let outcome = agent
        .handle_task(
            r#"{"description":"writer","prompt":"write a file","subagent_type":"general-purpose"}"#,
            &mut ui,
        )
        .await;

    assert_eq!(outcome.status, hi_tools::ToolStatus::Succeeded);
    assert_eq!(agent.background_task_registry().list().await.len(), 1);
}

#[tokio::test]
async fn pipefs_protocol_one_rejects_background_write_candidates() {
    let mut cfg = bg_config();
    cfg.subagents.write_subagents = WriteSubagentPolicy::On;
    let mut agent = agent(Vec::new(), cfg);
    agent.set_workspace_durability(Some(std::sync::Arc::new(TestWorkspaceDurability)));
    agent
        .activate_pipefs_workspace_controller("candidate-protocol-one", 1, false)
        .unwrap();

    let outcome = agent
        .handle_task(
            r#"{"description":"writer","prompt":"write a file","subagent_type":"general-purpose"}"#,
            &mut NullUi,
        )
        .await;

    assert_eq!(outcome.status, hi_tools::ToolStatus::Denied);
    assert!(outcome.content.contains("writer protocol 2"));
    assert!(agent.background_task_registry().list().await.is_empty());
}

#[tokio::test]
async fn handle_task_reports_registry_capacity_as_actionable_denial() {
    let mut agent = agent(Vec::new(), bg_config());
    let registry = agent.background_task_registry();
    for index in 0..16 {
        registry
            .spawn(
                &format!("capacity-{index}"),
                "explore",
                Box::new(|| {
                    Box::pin(async {
                        std::future::pending::<()>().await;
                        unreachable!("capacity fixture must remain live")
                    })
                }),
            )
            .await
            .unwrap();
    }

    let mut ui = NullUi;
    let outcome = agent
        .handle_task(
            r#"{"description":"one more","prompt":"inspect one more thing","subagent_type":"explore"}"#,
            &mut ui,
        )
        .await;

    assert_eq!(outcome.status, hi_tools::ToolStatus::Denied);
    assert!(outcome.content.contains("background task capacity reached"));
    assert!(outcome.content.contains("get_task_output or wait_tasks"));
}

#[tokio::test]
async fn general_purpose_task_is_isolated_and_cannot_succeed_before_parent_apply() {
    let mut cfg = bg_config();
    cfg.memory.tool_set = ToolSet::Full;
    cfg.subagents.write_subagents = WriteSubagentPolicy::On;
    cfg.gates.review = ReviewPolicy::Off;
    // Candidate preparation must replace even an explicit ambient opt-out.
    cfg.sandbox_policy = Some(hi_tools::sandbox::SandboxPolicy::Off);
    let root = cfg.paths.workspace_root.clone();
    let state_root = cfg.paths.state_root.clone();
    git(&root, &["init", "--quiet"]);
    std::fs::write(root.join("source-sentinel.txt"), "source\n").unwrap();
    git(&root, &["add", "source-sentinel.txt"]);
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
    let git_before = directory_bytes(&root.join(".git"));
    let mut agent = agent(Vec::new(), cfg);
    agent.set_delegate_route(
        Some("delegate-model".into()),
        Some("http://delegate.invalid/v1".into()),
        None,
    );
    assert_eq!(
        agent.background_candidate_plan_identity(),
        (
            Some(hi_tools::sandbox::SandboxPolicy::Strict),
            true,
            vec![
                root.canonicalize().unwrap(),
                state_root.canonicalize().unwrap()
            ],
            "http://delegate.invalid/v1".into(),
            "delegate-model".into(),
            Some(std::time::Duration::from_secs(120)),
        )
    );
    let registry = agent.background_task_registry();
    let owner = state_root.join("candidate-owner");
    let candidate =
        hi_tools::candidate_workspace::CandidateWorkspace::create(&root, &state_root, &owner)
            .unwrap();
    std::fs::write(candidate.root().join("candidate.txt"), "detached\n").unwrap();
    assert!(!root.join("candidate.txt").exists());
    assert_eq!(directory_bytes(&root.join(".git")), git_before);
    let binding = agent.workspace_controller_binding();
    let (id_tx, id_rx) = tokio::sync::oneshot::channel::<String>();
    let worker_registry = registry.clone();
    let artifact_state = state_root.clone();
    let task_id = registry
        .spawn(
            "isolated write",
            "general-purpose",
            Box::new(move || {
                Box::pin(async move {
                    let task_id = id_rx.await.unwrap();
                    let job_id = worker_registry
                        .candidate_workspace_job_id(&task_id)
                        .await
                        .unwrap();
                    let verification_ms = worker_registry
                        .candidate_workspace_verification_ms(&task_id)
                        .await
                        .unwrap();
                    let sealed = candidate
                        .seal_verified(hi_tools::candidate_workspace::CandidateSealContext {
                            job_id: hi_workspace::JobId::new(job_id),
                            binding,
                            route: hi_workspace::CandidateRoute {
                                provider: "test".into(),
                                model: "test-model".into(),
                                actual_model_revision: None,
                                capability_digest: "blake3:test-capabilities".into(),
                            },
                            verification: vec![hi_workspace::CandidateVerification {
                                name: "test verification".into(),
                                passed: true,
                                verifier_digest: "blake3:test-verification".into(),
                                detail: None,
                                artifacts: Vec::new(),
                            }],
                            destination_verification: vec![
                                hi_workspace::CandidateDestinationVerifier {
                                    name: "test verification".into(),
                                    command: "true".into(),
                                    timeout_ms: verification_ms,
                                },
                            ],
                            destination_verification_budget_ms: verification_ms,
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
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while !registry.candidate_is_ready(&task_id) {
            let state = registry
                .poll(&task_id, std::time::Duration::ZERO)
                .await
                .unwrap();
            assert_eq!(
                state.state,
                hi_tools::BackgroundTaskState::Running,
                "candidate preparation failed: {}",
                state.output
            );
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("candidate should become ready for parent apply");

    let before_apply = registry
        .poll(&task_id, std::time::Duration::ZERO)
        .await
        .unwrap();
    assert_eq!(before_apply.state, hi_tools::BackgroundTaskState::Running);
    assert!(!before_apply.applied);
    assert!(!root.join("candidate.txt").exists());
    assert_eq!(directory_bytes(&root.join(".git")), git_before);
    let persisted =
        hi_tools::candidate_workspace::PersistedDetachedCandidate::discover(&state_root).unwrap();
    assert_eq!(persisted.len(), 1);
    let artifact_path = persisted[0].path().to_path_buf();
    let store = hi_control::ControlStore::open_for_state(&state_root).unwrap();
    let jobs = store
        .jobs_for_binding(agent.workspace_controller_binding().binding_id.as_str())
        .unwrap();
    assert_eq!(
        jobs[0].candidate_ref.as_deref(),
        Some(persisted[0].artifact.uri.as_str())
    );

    let binding_before_rebind = agent.workspace_controller_binding();
    let other_root = state_root.join("blocked-rebind-root");
    let other_state = state_root.join("blocked-rebind-state");
    std::fs::create_dir_all(&other_root).unwrap();
    let rebind_error = agent
        .rebind_workspace(&other_root, &other_state)
        .await
        .unwrap_err();
    assert!(rebind_error.to_string().contains("jobs remain unsettled"));
    assert_eq!(
        agent.workspace_root().canonicalize().unwrap(),
        root.canonicalize().unwrap()
    );
    assert_eq!(agent.workspace_controller_binding(), binding_before_rebind);

    // Claim and cancellation are one atomic ownership race: once the parent
    // owns merge, kill_task cannot publish Cancelled while these bytes remain
    // eligible to apply. Restore models a failed parent admission retry.
    let mut claimed = registry.claim_ready_candidates();
    assert_eq!(claimed.len(), 1);
    let (claimed_id, claimed_candidate) = claimed.pop().unwrap();
    assert_eq!(claimed_id, task_id);
    let kill_during_claim = registry.kill(&task_id).await.unwrap();
    assert_eq!(
        kill_during_claim.state,
        hi_tools::BackgroundTaskState::Running
    );
    assert!(kill_during_claim.output.contains("merge has started"));
    assert!(!root.join("candidate.txt").exists());
    registry.restore_ready_candidate(&task_id, claimed_candidate);

    agent.settle_ready_candidates_at_boundary().await.unwrap();
    let applied = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let outcome = registry
                .poll(&task_id, std::time::Duration::ZERO)
                .await
                .unwrap();
            if outcome.state != hi_tools::BackgroundTaskState::Running {
                break outcome;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("settled candidate should publish a terminal task result");
    assert_eq!(applied.state, hi_tools::BackgroundTaskState::Completed);
    assert!(applied.applied);
    assert_eq!(
        std::fs::read_to_string(root.join("candidate.txt")).unwrap(),
        "detached\n"
    );
    assert!(!artifact_path.exists());
}

#[tokio::test]
async fn killing_a_ready_candidate_cancels_its_workspace_job_without_applying() {
    let mut cfg = bg_config();
    cfg.subagents.write_subagents = WriteSubagentPolicy::On;
    let root = cfg.paths.workspace_root.clone();
    let state_root = cfg.paths.state_root.clone();
    git(&root, &["init", "--quiet"]);
    std::fs::write(root.join("source-sentinel.txt"), "source\n").unwrap();
    let git_before = directory_bytes(&root.join(".git"));
    let mut agent = agent(Vec::new(), cfg);
    let binding = agent.workspace_controller_binding();
    let registry = agent.background_task_registry();
    let candidate = hi_tools::candidate_workspace::CandidateWorkspace::create(
        &root,
        &state_root,
        &state_root.join("cancel-candidate-owner"),
    )
    .unwrap();
    std::fs::write(candidate.root().join("must-not-apply.txt"), "candidate\n").unwrap();
    let (id_tx, id_rx) = tokio::sync::oneshot::channel::<String>();
    let worker_registry = registry.clone();
    let artifact_state = state_root.clone();
    let task_id = registry
        .spawn(
            "cancel isolated write",
            "general-purpose",
            Box::new(move || {
                Box::pin(async move {
                    let task_id = id_rx.await.unwrap();
                    let job_id = worker_registry
                        .candidate_workspace_job_id(&task_id)
                        .await
                        .unwrap();
                    let verification_ms = worker_registry
                        .candidate_workspace_verification_ms(&task_id)
                        .await
                        .unwrap();
                    let sealed = candidate
                        .seal_verified(hi_tools::candidate_workspace::CandidateSealContext {
                            job_id: hi_workspace::JobId::new(job_id),
                            binding,
                            route: hi_workspace::CandidateRoute {
                                provider: "test".into(),
                                model: "test-model".into(),
                                actual_model_revision: None,
                                capability_digest: "blake3:test-capabilities".into(),
                            },
                            verification: vec![hi_workspace::CandidateVerification {
                                name: "test verification".into(),
                                passed: true,
                                verifier_digest: "blake3:test-verification".into(),
                                detail: None,
                                artifacts: Vec::new(),
                            }],
                            destination_verification: vec![
                                hi_workspace::CandidateDestinationVerifier {
                                    name: "test verification".into(),
                                    command: "true".into(),
                                    timeout_ms: verification_ms,
                                },
                            ],
                            destination_verification_budget_ms: verification_ms,
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
                        changed_files: vec!["must-not-apply.txt".into()],
                    }
                })
            }),
        )
        .await
        .unwrap();
    id_tx.send(task_id.clone()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while !registry.candidate_is_ready(&task_id) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    let store = hi_control::ControlStore::open_for_state(&state_root).unwrap();
    let ready = store
        .jobs_for_binding(agent.workspace_controller_binding().binding_id.as_str())
        .unwrap();
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].state, hi_control::ControlJobState::ReadyToMerge);
    let artifacts =
        hi_tools::candidate_workspace::PersistedDetachedCandidate::discover(&state_root).unwrap();
    assert_eq!(artifacts.len(), 1);
    let artifact_path = artifacts[0].path().to_path_buf();
    assert_eq!(
        ready[0].candidate_ref.as_deref(),
        Some(artifacts[0].artifact.uri.as_str())
    );
    let cancelled = registry.kill(&task_id).await.unwrap();
    assert_eq!(cancelled.state, hi_tools::BackgroundTaskState::Cancelled);
    assert!(!root.join("must-not-apply.txt").exists());
    assert_eq!(directory_bytes(&root.join(".git")), git_before);
    assert!(!artifact_path.exists());
    agent.settle_ready_candidates_at_boundary().await.unwrap();
    assert!(!root.join("must-not-apply.txt").exists());

    let terminal = store.get_job(&ready[0].job_id).unwrap().unwrap();
    assert_eq!(terminal.state, hi_control::ControlJobState::Cancelled);
    let revision = terminal.revision;
    assert_eq!(
        registry.kill(&task_id).await.unwrap().state,
        hi_tools::BackgroundTaskState::Cancelled
    );
    assert_eq!(
        store.get_job(&ready[0].job_id).unwrap().unwrap().revision,
        revision,
        "repeated kill must not write another terminal transition"
    );
}

#[test]
fn task_tool_spec_lists_grok_build_kinds() {
    let spec = hi_tools::task_tool_spec();
    let enum_vals = spec
        .parameters
        .pointer("/properties/subagent_type/enum")
        .and_then(|v| v.as_array())
        .expect("subagent_type enum");
    let names: Vec<&str> = enum_vals.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(names, vec!["explore", "plan", "general-purpose"]);
    assert!(spec.description.contains("explore"));
    assert!(spec.description.contains("plan"));
    assert!(spec.description.contains("general-purpose"));
    assert!(spec.parameters.pointer("/properties/scope").is_none());
    assert!(
        spec.description
            .contains("applies them transactionally at a safe turn boundary"),
        "general-purpose isolation contract missing from task tool description"
    );
    assert!(!spec.description.contains("live working tree"));
}

fn git(root: &std::path::Path, args: &[&str]) {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn directory_bytes(
    root: &std::path::Path,
) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    fn visit(
        root: &std::path::Path,
        at: &std::path::Path,
        files: &mut std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>,
    ) {
        for entry in std::fs::read_dir(at).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                visit(root, &path, files);
            } else {
                files.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(path).unwrap(),
                );
            }
        }
    }
    let mut files = std::collections::BTreeMap::new();
    visit(root, root, &mut files);
    files
}

#[tokio::test]
async fn handle_kill_task_unknown_id_fails() {
    let agent = agent(Vec::new(), bg_config());
    let outcome = agent
        .handle_kill_task(r#"{"task_id": "nonexistent"}"#)
        .await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
}

#[tokio::test]
async fn handle_get_task_output_invalid_json_fails() {
    let agent = agent(Vec::new(), bg_config());
    let outcome = agent.handle_get_task_output("not json").await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
}

#[tokio::test]
async fn handle_wait_tasks_empty_ids_fails() {
    let agent = agent(Vec::new(), bg_config());
    let outcome = agent.handle_wait_tasks(r#"{"task_ids": []}"#).await;
    assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
}

#[test]
fn mcp_memory_skill_tools_in_catalog() {
    assert!(hi_tools::is_known_tool("use_tool"));
    assert!(hi_tools::is_known_tool("search_tool"));
    assert!(hi_tools::is_known_tool("memory_search"));
    assert!(hi_tools::is_known_tool("memory_get"));
    assert!(hi_tools::is_known_tool("memory_update"));
    assert!(hi_tools::is_known_tool("memory_forget"));
    assert!(hi_tools::is_known_tool("skill"));
}

#[test]
fn mcp_memory_skill_tool_specs_exist() {
    assert_eq!(hi_tools::use_tool_tool_spec().name, "use_tool");
    assert_eq!(hi_tools::search_tool_tool_spec().name, "search_tool");
    assert_eq!(hi_tools::memory_search_tool_spec().name, "memory_search");
    assert_eq!(hi_tools::memory_get_tool_spec().name, "memory_get");
    assert_eq!(hi_tools::memory_update_tool_spec().name, "memory_update");
    assert_eq!(hi_tools::memory_forget_tool_spec().name, "memory_forget");
    assert_eq!(hi_tools::skill_tool_spec().name, "skill");
}
