//! Provider-backed [`StageModel`] plus the managed-path workflow composition.
//!
//! Model output is untrusted evidence: each role answers a strict JSON
//! contract, plans and patches are extracted and validated, patch bytes are
//! sealed into a content-addressed artifact store, and model-call/token
//! budgets are consumed against the shared ledger before and after every
//! invocation. Per docs/adr/001-rsi-runtime-boundary.md only the managed
//! (noninteractive, descriptor-bound) path may compose this with
//! `hi_agent_runtime::WorkflowExecutor` — never the interactive loop.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use hi_agent_runtime::{
    DenyGates, StageDriver, StageModel, StageOutcome, TerminalOutcome, WorkflowExecutor,
};
use hi_ai::{ChatRequest, Content, Message, Provider, RequestProfile};
use hi_rsi_runtime::{
    ArtifactRef, BudgetKind, Checkpoint, EngineeringPlan, RunState, RuntimeBudgets,
    SharedBudgetLedger, StageId, WorkflowGraph,
};
use hi_verifier::{AttestingVerifier, Attestor, CheckSpec};
use serde::Deserialize;

/// Output ceiling for one model stage answer — stage responses are structured
/// summaries and diffs, not transcripts.
const MODEL_STAGE_MAX_TOKENS: u32 = 8_192;
/// Recent failure evidence surfaced into a stage prompt.
const SURFACED_FAILURES: usize = 3;

pub(crate) struct ProviderStageModel {
    provider: Arc<dyn Provider>,
    model: String,
    ledger: SharedBudgetLedger,
    artifact_dir: PathBuf,
}

/// Strict per-stage response contract. `deny_unknown_fields` keeps a drifting
/// model from smuggling unmodeled claims into the run state.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelStageResponse {
    passed: bool,
    summary: String,
    #[serde(default)]
    plan: Option<ModelPlan>,
    #[serde(default)]
    patch: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelPlan {
    objective: String,
    #[serde(default)]
    assumptions: Vec<String>,
    #[serde(default)]
    affected_components: Vec<String>,
    proposed_changes: Vec<String>,
    #[serde(default)]
    tests: Vec<String>,
    #[serde(default)]
    risks: Vec<String>,
    rollback: String,
}

impl ProviderStageModel {
    pub(crate) fn new(
        provider: Arc<dyn Provider>,
        model: String,
        ledger: SharedBudgetLedger,
        artifact_dir: PathBuf,
    ) -> Self {
        Self {
            provider,
            model,
            ledger,
            artifact_dir,
        }
    }

    fn seal_patch(&self, patch: &str) -> Result<ArtifactRef> {
        let bytes = patch.as_bytes();
        let hash = blake3::hash(bytes).to_hex().to_string();
        std::fs::create_dir_all(&self.artifact_dir).with_context(|| {
            format!(
                "creating artifact directory {}",
                self.artifact_dir.display()
            )
        })?;
        let path = self.artifact_dir.join(format!("{hash}.patch"));
        if !path.exists() {
            let temp = self.artifact_dir.join(format!(".{hash}.tmp"));
            std::fs::write(&temp, bytes)
                .with_context(|| format!("writing patch artifact {}", temp.display()))?;
            std::fs::rename(&temp, &path)
                .with_context(|| format!("sealing patch artifact {}", path.display()))?;
        }
        Ok(ArtifactRef {
            hash,
            size_bytes: bytes.len() as u64,
            media_type: "text/x-diff".into(),
        })
    }
}

fn role_instructions(role: &str) -> String {
    let focus = match role {
        "requirement_normalizer" => "Restate the task as precise, testable requirements.",
        "repository_explorer" => "Summarize the repository areas relevant to the task.",
        "diagnostician" => "Identify the defect or gap the task addresses, with evidence.",
        "planner" => {
            "Produce an engineering plan. The `plan` field is REQUIRED: objective, \
             proposed_changes, tests, risks, and rollback."
        }
        "implementer" => {
            "Produce the change. The `patch` field is REQUIRED: one unified diff \
             against the worktree root implementing the plan."
        }
        "reviewer" => "Review the plan and patches for defects; fail the stage on concrete problems.",
        _ => "Complete this workflow stage.",
    };
    format!(
        "You are the `{role}` stage of a managed engineering workflow.\n{focus}\n\
         Respond with EXACTLY one JSON object and nothing else:\n\
         {{\"passed\": bool, \"summary\": string, \"plan\": {{...}} | null, \"patch\": string | null}}\n\
         `plan` fields: objective, assumptions[], affected_components[], proposed_changes[], \
         tests[], risks[], rollback. Omit `plan` and `patch` unless your role requires them."
    )
}

fn stage_context(stage: &StageId, attempt: u32, state: &RunState) -> String {
    let mut context = format!(
        "task: {}\nrun: {}\nstage: {} (attempt {attempt})\nworktree: {}\n",
        state.task_id, state.run_id, stage.0, state.repository.worktree_root
    );
    if let Some(plan) = &state.plan {
        context.push_str(&format!(
            "plan v{}: {}\nproposed changes: {}\n",
            plan.revision,
            plan.objective,
            plan.proposed_changes.join("; ")
        ));
    }
    context.push_str(&format!("patches so far: {}\n", state.patches.len()));
    for failure in state.failure_evidence.iter().rev().take(SURFACED_FAILURES) {
        context.push_str(&format!(
            "recent failure: {} at {}\n",
            failure.subcategory, failure.stage.0
        ));
    }
    context
}

/// Parse the strict stage contract, tolerating a fenced code block around the
/// JSON but nothing else.
fn parse_stage_response(text: &str) -> Result<ModelStageResponse> {
    let trimmed = text.trim();
    let body = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(|rest| rest.trim_start_matches(['\r', '\n']))
        .and_then(|rest| rest.strip_suffix("```"))
        .unwrap_or(trimmed);
    serde_json::from_str(body.trim()).with_context(|| {
        format!(
            "model stage response is not the required JSON contract: {}",
            &trimmed.chars().take(200).collect::<String>()
        )
    })
}

#[async_trait]
impl StageModel for ProviderStageModel {
    async fn invoke(
        &self,
        role: &str,
        stage: &StageId,
        attempt: u32,
        state: &RunState,
    ) -> Result<StageOutcome> {
        self.ledger.consume(BudgetKind::ModelCalls, 1)?;
        let request = ChatRequest {
            model: self.model.clone(),
            request_id: Some(format!("rsi-{}-{}-{attempt}", state.run_id, stage.0)),
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: Arc::new(vec![
                Message::system(role_instructions(role)),
                Message::user(stage_context(stage, attempt, state)),
            ]),
            tools: Vec::new().into(),
            max_tokens: MODEL_STAGE_MAX_TOKENS,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile::default(),
        };
        let completion = self.provider.stream(request, &mut |_| {}).await?;
        self.ledger
            .consume(BudgetKind::InputTokens, completion.usage.input_tokens)?;
        self.ledger
            .consume(BudgetKind::OutputTokens, completion.usage.output_tokens)?;
        let text: String = completion
            .content
            .iter()
            .filter_map(|content| match content {
                Content::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let response = parse_stage_response(&text)?;

        let plan = response
            .plan
            .map(|plan| -> Result<EngineeringPlan> {
                ensure!(
                    !plan.proposed_changes.is_empty(),
                    "stage {} returned a plan without proposed changes",
                    stage.0
                );
                let revision = state
                    .plan
                    .as_ref()
                    .map(|previous| previous.revision)
                    .unwrap_or(0)
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("plan revision overflow"))?;
                Ok(EngineeringPlan {
                    objective: plan.objective,
                    assumptions: plan.assumptions,
                    affected_components: plan.affected_components,
                    evidence: vec![],
                    proposed_changes: plan.proposed_changes,
                    tests: plan.tests,
                    risks: plan.risks,
                    rollback: plan.rollback,
                    revision,
                    revision_reason: (revision > 1)
                        .then(|| format!("stage {} attempt {attempt}", stage.0)),
                })
            })
            .transpose()?;
        let patches = response
            .patch
            .as_deref()
            .filter(|patch| !patch.trim().is_empty())
            .map(|patch| self.seal_patch(patch))
            .transpose()?
            .into_iter()
            .collect();

        Ok(StageOutcome {
            passed: response.passed,
            output: serde_json::json!({
                "role": role,
                "summary": response.summary,
            }),
            plan,
            patches,
            failures: vec![],
            verification: None,
        })
    }
}

/// The model roles [`ProviderStageModel`] carries stage instructions for.
/// Descriptor workflow graphs are validated against exactly this set — a
/// package graph naming any other role fails closed before the first stage.
pub(crate) const KNOWN_STAGE_ROLES: [&str; 6] = [
    "requirement_normalizer",
    "repository_explorer",
    "diagnostician",
    "planner",
    "implementer",
    "reviewer",
];

/// Launch the workflow a managed runtime descriptor declares: decode and
/// authorize its graph, derive the run identity and repository state from the
/// descriptor, hash the worktree, and drive [`run_managed_workflow`] under
/// the descriptor's budgets. Returns the terminal outcome with the final run
/// state for evidence upload.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_descriptor_workflow<A: Attestor>(
    descriptor: &hi_rsi_runtime::ManagedRuntimeDescriptor,
    provider: Arc<dyn Provider>,
    model: String,
    worktree: &Path,
    state_dir: &Path,
    attestor: A,
    environment_hash: String,
    verification_checks: Vec<CheckSpec>,
    tools: BTreeMap<String, CheckSpec>,
    resume_from: Option<&Checkpoint>,
) -> Result<(TerminalOutcome, RunState)> {
    let authorized_roles = KNOWN_STAGE_ROLES
        .into_iter()
        .map(str::to_owned)
        .collect::<std::collections::BTreeSet<_>>();
    // A graph tool stage is authorized exactly when this run can execute it:
    // the keys of the configured sandboxed-tool table.
    let authorized_tools = tools
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let graph = descriptor.workflow_graph(&authorized_roles, &authorized_tools)?;
    let mut state = RunState {
        task_id: descriptor.identity.task_id.clone(),
        run_id: descriptor.identity.run_id.clone(),
        candidate_id: descriptor.identity.candidate_id.clone(),
        repository: hi_rsi_runtime::RepositoryState {
            repository_snapshot_hash: descriptor.identity.repository_snapshot_hash.clone(),
            starting_commit: descriptor.identity.source_commit.clone(),
            source_tree_hash: hi_verifier::hash_tree(worktree)?,
            worktree_root: worktree.to_string_lossy().into_owned(),
            submodule_commits: BTreeMap::new(),
        },
        current_stages: Default::default(),
        attempts: BTreeMap::new(),
        working_memory: vec![],
        plan: None,
        patches: vec![],
        verification: vec![],
        budget: Default::default(),
        failure_evidence: vec![],
    };
    let outcome = run_managed_workflow(
        graph,
        provider,
        model,
        &descriptor.budgets,
        attestor,
        environment_hash,
        verification_checks,
        tools,
        state_dir,
        &mut state,
        resume_from,
    )
    .await?;
    Ok((outcome, state))
}

/// Compose the full managed workflow: provider-backed model stages, sandboxed
/// tools, trusted verification, durable checkpoints — freshly or resumed from
/// a sealed checkpoint. This is the managed path's entry; the interactive
/// loop must never call it.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_managed_workflow<A: Attestor>(
    graph: WorkflowGraph,
    provider: Arc<dyn Provider>,
    model: String,
    budgets: &RuntimeBudgets,
    attestor: A,
    environment_hash: String,
    verification_checks: Vec<CheckSpec>,
    tools: BTreeMap<String, CheckSpec>,
    state_dir: &Path,
    state: &mut RunState,
    resume_from: Option<&Checkpoint>,
) -> Result<TerminalOutcome> {
    let ledger = SharedBudgetLedger::new(budgets);
    let driver = StageDriver::new(
        ProviderStageModel::new(
            provider,
            model,
            ledger.clone(),
            state_dir.join("artifacts"),
        ),
        AttestingVerifier::new(attestor, environment_hash)?,
        verification_checks,
        tools,
        DenyGates,
        state_dir.join("checkpoints"),
    )?;
    let executor = WorkflowExecutor::new(graph, driver, ledger);
    match resume_from {
        Some(checkpoint) => executor.resume(checkpoint, state).await,
        None => executor.execute(state).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::{Completion, StreamEvent, Usage};
    use hi_rsi_runtime::RepositoryState;
    use std::collections::BTreeSet;
    use std::sync::Mutex;
    use std::time::Duration;

    /// Scripted provider: answers each stage by role with contract JSON.
    struct ScriptedProvider {
        requests: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        async fn stream(
            &self,
            request: ChatRequest,
            _sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> anyhow::Result<Completion> {
            let system = request
                .messages
                .first()
                .and_then(|message| match message.content.first() {
                    Some(Content::Text(text)) => Some(text.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            self.requests.lock().unwrap().push(system.clone());
            let body = if system.contains("`planner`") {
                r#"{"passed": true, "summary": "planned", "plan": {"objective": "fix", "proposed_changes": ["edit lib"], "rollback": "revert"}}"#.to_string()
            } else if system.contains("`implementer`") {
                r#"{"passed": true, "summary": "implemented", "patch": "--- a/lib.rs\n+++ b/lib.rs\n"}"#.to_string()
            } else {
                r#"{"passed": true, "summary": "ok"}"#.to_string()
            };
            Ok(Completion {
                content: vec![Content::Text(format!("```json\n{body}\n```"))],
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
                stop_reason: None,
            })
        }
    }

    struct TestAttestor;
    impl Attestor for TestAttestor {
        fn attest(&self, hash: &[u8; 32]) -> anyhow::Result<String> {
            Ok(hash.iter().map(|byte| format!("{byte:02x}")).collect())
        }
    }

    fn spec(name: &str, program: &str) -> CheckSpec {
        CheckSpec {
            name: name.into(),
            program: program.into(),
            arguments: vec![],
            timeout: Duration::from_secs(30),
            required: true,
            inherit_environment: false,
        }
    }

    fn budgets() -> RuntimeBudgets {
        RuntimeBudgets {
            wall_time_seconds: 600,
            cpu_time_seconds: 600,
            memory_bytes: 1,
            disk_bytes: 1,
            input_tokens: 10_000,
            output_tokens: 10_000,
            tool_calls: 100,
            cost_microusd: 1,
            model_calls: 100,
            repair_iterations: 1,
            trace_bytes: 1,
        }
    }

    fn state(worktree: &Path) -> RunState {
        RunState {
            task_id: "t".into(),
            run_id: "r".into(),
            candidate_id: "c".into(),
            repository: RepositoryState {
                repository_snapshot_hash: "a".repeat(64),
                starting_commit: "x".into(),
                source_tree_hash: "b".repeat(64),
                worktree_root: worktree.to_string_lossy().into_owned(),
                submodule_commits: BTreeMap::new(),
            },
            current_stages: BTreeSet::new(),
            attempts: BTreeMap::new(),
            working_memory: vec![],
            plan: None,
            patches: vec![],
            verification: vec![],
            budget: hi_rsi_runtime::BudgetUsage::default(),
            failure_evidence: vec![],
        }
    }

    #[test]
    fn strict_contract_rejects_unmodeled_claims() {
        assert!(parse_stage_response(r#"{"passed": true, "summary": "ok"}"#).is_ok());
        assert!(
            parse_stage_response(r#"```json
{"passed": false, "summary": "no"}
```"#)
            .is_ok(),
            "fenced contract JSON is tolerated"
        );
        assert!(
            parse_stage_response(r#"{"passed": true, "summary": "ok", "verification": {}}"#)
                .is_err(),
            "unknown fields must be rejected"
        );
        assert!(parse_stage_response("I did the thing!").is_err());
    }

    #[tokio::test]
    async fn descriptor_workflow_derives_identity_and_runs_to_success() {
        let base = tempfile::tempdir().unwrap();
        let worktree = base.path().join("worktree");
        std::fs::create_dir_all(&worktree).unwrap();
        std::fs::write(worktree.join("lib.rs"), "fn main() {}\n").unwrap();
        let descriptor = hi_rsi_runtime::ManagedRuntimeDescriptor {
            schema_version: 1,
            protocol_major: 1,
            identity: hi_rsi_runtime::CandidateIdentity {
                run_id: "run-1".into(),
                task_id: "task-1".into(),
                candidate_id: "candidate-1".into(),
                manifest_hash: "1".repeat(64),
                agent_artifact_hash: "2".repeat(64),
                repository_snapshot_hash: "3".repeat(64),
                source_repository: "pipe/hi".into(),
                source_commit: "abc123".into(),
            },
            budgets: budgets(),
            policy: hi_rsi_runtime::RuntimePolicy {
                task_policy_version: "task-v1".into(),
                mutation_level: hi_rsi_runtime::MutationLevel::Workflow,
                workflow_entrypoint: "intake".into(),
                model_role: "implementer".into(),
                tool_set: "minimal".into(),
                tool_mode: "auto".into(),
                filesystem_mode: "worktree-write".into(),
                allowed_tools: vec![],
                network_allowlist: vec![],
                isolation: hi_rsi_runtime::IsolationProfile::Namespace,
                trusted_launcher: true,
            },
            runtime_package: None,
            issued_at_unix_ms: 1_000,
            expires_at_unix_ms: 2_000,
        };
        let provider = Arc::new(ScriptedProvider {
            requests: Mutex::new(Vec::new()),
        });

        let (outcome, run_state) = run_descriptor_workflow(
            &descriptor,
            provider,
            "stage-model".into(),
            &worktree,
            &base.path().join("state"),
            TestAttestor,
            "a".repeat(64),
            vec![spec("verify_true", "true")],
            BTreeMap::from([
                ("cargo_check".to_string(), spec("cargo_check", "true")),
                ("cargo_test".to_string(), spec("cargo_test", "true")),
            ]),
            None,
        )
        .await
        .unwrap();

        assert_eq!(outcome, TerminalOutcome::Succeeded);
        assert_eq!(run_state.run_id, "run-1");
        assert_eq!(run_state.candidate_id, "candidate-1");
        assert_eq!(
            run_state.repository.source_tree_hash,
            hi_verifier::hash_tree(&worktree).unwrap(),
            "run state must bind the actual worktree content hash"
        );
        assert_eq!(run_state.verification.len(), 1);
    }

    #[tokio::test]
    async fn managed_workflow_runs_default_coding_graph_with_provider_model() {
        let base = tempfile::tempdir().unwrap();
        let worktree = base.path().join("worktree");
        std::fs::create_dir_all(&worktree).unwrap();
        std::fs::write(worktree.join("lib.rs"), "fn main() {}\n").unwrap();
        let state_dir = base.path().join("state");
        let provider = Arc::new(ScriptedProvider {
            requests: Mutex::new(Vec::new()),
        });
        let mut run_state = state(&worktree);

        let outcome = run_managed_workflow(
            WorkflowGraph::default_coding(),
            provider.clone(),
            "stage-model".into(),
            &budgets(),
            TestAttestor,
            "a".repeat(64),
            vec![spec("verify_true", "true")],
            BTreeMap::from([
                ("cargo_check".to_string(), spec("cargo_check", "true")),
                ("cargo_test".to_string(), spec("cargo_test", "true")),
            ]),
            &state_dir,
            &mut run_state,
            None,
        )
        .await
        .unwrap();

        assert_eq!(outcome, TerminalOutcome::Succeeded);
        assert_eq!(run_state.plan.as_ref().unwrap().revision, 1);
        assert_eq!(run_state.patches.len(), 1);
        assert_eq!(run_state.verification.len(), 1);
        // The patch artifact was sealed content-addressed on disk.
        let artifact = state_dir
            .join("artifacts")
            .join(format!("{}.patch", run_state.patches[0].hash));
        assert!(artifact.exists(), "sealed patch artifact must exist");
        // Model-call and token budgets were consumed for the six model stages.
        assert_eq!(
            run_state.budget.consumed.get(&BudgetKind::ModelCalls),
            Some(&6)
        );

        // A sealed checkpoint resumes through the same composition.
        let sealed = hi_agent_runtime::latest_checkpoint(&state_dir.join("checkpoints"))
            .unwrap()
            .expect("run seals checkpoints");
        let mut resumed_state = state(&worktree);
        let resumed = run_managed_workflow(
            WorkflowGraph::default_coding(),
            provider,
            "stage-model".into(),
            &budgets(),
            TestAttestor,
            "a".repeat(64),
            vec![spec("verify_true", "true")],
            BTreeMap::from([
                ("cargo_check".to_string(), spec("cargo_check", "true")),
                ("cargo_test".to_string(), spec("cargo_test", "true")),
            ]),
            &state_dir,
            &mut resumed_state,
            Some(&sealed),
        )
        .await
        .unwrap();
        assert_eq!(resumed, TerminalOutcome::Succeeded);
    }
}
