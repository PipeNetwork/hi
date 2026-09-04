//! Production [`TrustedStageDriver`]: binds workflow stages to model roles,
//! sandboxed tools, trusted verification, and durable checkpoints.
//!
//! The trust split follows the RSI boundary (docs/adr/001): model-side work is
//! injected through [`StageModel`] and treated as evidence — it can propose
//! plans, patches, and failures but can never attach verification or an
//! attestation. Tool commands run only from the driver's authorized table in
//! the hardened verifier environment, and `VerificationGate` stages call the
//! supervisor-owned [`AttestingVerifier`] directly. Checkpoints seal to disk
//! atomically and are reloadable for [`crate::WorkflowExecutor::resume`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use hi_rsi_runtime::{
    ArtifactRef, Checkpoint, FailureDomain, FailureEvidence, RunState, StageDefinition, StageId,
    StageKind, VerificationStatus,
};
use hi_verifier::{AttestingVerifier, Attestor, CheckSpec};
use serde_json::json;

use crate::{StageOutcome, TrustedStageDriver};

/// Bytes of tool output surfaced into a stage outcome; full output stays in
/// the tool's own logs, the outcome only needs enough to steer repair.
const TOOL_OUTPUT_SURFACE_BYTES: usize = 8 * 1024;

/// Untrusted model-side stage work, injected by the host (which owns provider
/// credentials and prompting). The driver constrains what its outcomes may
/// claim.
#[async_trait]
pub trait StageModel: Send + Sync {
    async fn invoke(
        &self,
        role: &str,
        stage: &StageId,
        attempt: u32,
        state: &RunState,
    ) -> Result<StageOutcome>;
}

/// Out-of-band judgment for policy and human-approval gates.
#[async_trait]
pub trait GateAuthority: Send + Sync {
    async fn policy_gate(&self, stage: &StageId, state: &RunState) -> Result<bool>;
    async fn human_approval(&self, stage: &StageId, state: &RunState) -> Result<bool>;
}

/// Fail-closed default: a graph with gates cannot run without an explicit
/// authority to decide them.
pub struct DenyGates;

#[async_trait]
impl GateAuthority for DenyGates {
    async fn policy_gate(&self, stage: &StageId, _: &RunState) -> Result<bool> {
        bail!("policy gate {} has no configured authority", stage.0)
    }
    async fn human_approval(&self, stage: &StageId, _: &RunState) -> Result<bool> {
        bail!(
            "human approval gate {} has no configured authority",
            stage.0
        )
    }
}

pub struct StageDriver<M, A, G = DenyGates> {
    models: M,
    verifier: AttestingVerifier<A>,
    verification_checks: Vec<CheckSpec>,
    tools: BTreeMap<String, CheckSpec>,
    gates: G,
    checkpoint_dir: PathBuf,
}

impl<M: StageModel, A: Attestor, G: GateAuthority> StageDriver<M, A, G> {
    pub fn new(
        models: M,
        verifier: AttestingVerifier<A>,
        verification_checks: Vec<CheckSpec>,
        tools: BTreeMap<String, CheckSpec>,
        gates: G,
        checkpoint_dir: PathBuf,
    ) -> Result<Self> {
        ensure!(
            !verification_checks.is_empty(),
            "trusted verification requires at least one check"
        );
        Ok(Self {
            models,
            verifier,
            verification_checks,
            tools,
            gates,
            checkpoint_dir,
        })
    }

    fn worktree<'a>(&self, state: &'a RunState) -> &'a Path {
        Path::new(&state.repository.worktree_root)
    }
}

#[async_trait]
impl<M: StageModel, A: Attestor, G: GateAuthority> TrustedStageDriver for StageDriver<M, A, G> {
    async fn stage(
        &self,
        definition: &StageDefinition,
        stage: &StageId,
        attempt: u32,
        state: &RunState,
    ) -> Result<StageOutcome> {
        match definition.kind {
            StageKind::DeterministicTransform
            | StageKind::ParallelFanOut
            | StageKind::Aggregation => Ok(StageOutcome {
                passed: true,
                output: json!({ "stage": stage.0, "kind": "structural" }),
                ..Default::default()
            }),
            StageKind::ModelInvocation => {
                let role = definition
                    .model_role
                    .as_deref()
                    .ok_or_else(|| anyhow!("model stage {} has no role", stage.0))?;
                let outcome = self.models.invoke(role, stage, attempt, state).await?;
                // Defense in depth alongside the executor's gate: model output
                // is evidence, never a trusted verdict.
                ensure!(
                    outcome.verification.is_none(),
                    "model stage {} may not attach verification",
                    stage.0
                );
                Ok(outcome)
            }
            StageKind::ToolInvocation => {
                let tool = definition
                    .tool
                    .as_deref()
                    .ok_or_else(|| anyhow!("tool stage {} has no tool", stage.0))?;
                let spec = self.tools.get(tool).ok_or_else(|| {
                    anyhow!("tool stage {} uses unauthorized tool {tool}", stage.0)
                })?;
                let run = hi_verifier::run_tool(self.worktree(state), spec).await;
                // Spawn/read failures are runtime faults, not candidate
                // quality signal — abort rather than mis-score the run.
                ensure!(
                    run.status != VerificationStatus::InfrastructureError,
                    "tool {tool} infrastructure failure: {}",
                    String::from_utf8_lossy(&run.output)
                );
                let passed = run.status == VerificationStatus::Passed;
                let surfaced: String = String::from_utf8_lossy(&run.output)
                    .chars()
                    .take(TOOL_OUTPUT_SURFACE_BYTES)
                    .collect();
                let failures = if passed {
                    vec![]
                } else {
                    vec![FailureEvidence {
                        domain: FailureDomain::Tool,
                        subcategory: format!("{tool}_failed"),
                        retryable: false,
                        causal_event_hash: None,
                        stage: stage.clone(),
                        artifacts: vec![],
                        counts_against_candidate: true,
                    }]
                };
                Ok(StageOutcome {
                    passed,
                    output: json!({
                        "tool": tool,
                        "exit_code": run.exit_code,
                        "output": surfaced,
                    }),
                    failures,
                    ..Default::default()
                })
            }
            StageKind::PolicyGate => {
                let passed = self.gates.policy_gate(stage, state).await?;
                Ok(StageOutcome {
                    passed,
                    output: json!({ "gate": stage.0, "approved": passed }),
                    ..Default::default()
                })
            }
            StageKind::HumanApprovalGate => {
                let passed = self.gates.human_approval(stage, state).await?;
                Ok(StageOutcome {
                    passed,
                    output: json!({ "gate": stage.0, "approved": passed }),
                    ..Default::default()
                })
            }
            StageKind::VerificationGate => {
                ensure!(
                    definition.trusted,
                    "verification gate {} is not marked trusted",
                    stage.0
                );
                let report = self
                    .verifier
                    .verify(
                        self.worktree(state),
                        &state.run_id,
                        &state.candidate_id,
                        &self.verification_checks,
                    )
                    .await?;
                Ok(StageOutcome {
                    passed: report.passed,
                    output: json!({
                        "checks": report.checks.len(),
                        "passed": report.passed,
                    }),
                    verification: Some(report),
                    ..Default::default()
                })
            }
            StageKind::TerminalSuccess | StageKind::TerminalFailure => {
                bail!("terminal stage {} is not executable", stage.0)
            }
        }
    }

    async fn checkpoint(&self, checkpoint: &Checkpoint, _reason: &str) -> Result<ArtifactRef> {
        let hash = checkpoint.canonical_hash()?;
        let bytes = serde_json::to_vec_pretty(checkpoint)?;
        std::fs::create_dir_all(&self.checkpoint_dir).with_context(|| {
            format!(
                "creating checkpoint directory {}",
                self.checkpoint_dir.display()
            )
        })?;
        let name = checkpoint_filename(checkpoint, &hash);
        let temp = self.checkpoint_dir.join(format!(".{name}.tmp"));
        let path = self.checkpoint_dir.join(&name);
        std::fs::write(&temp, &bytes)
            .with_context(|| format!("writing checkpoint {}", temp.display()))?;
        std::fs::rename(&temp, &path)
            .with_context(|| format!("sealing checkpoint {}", path.display()))?;
        Ok(ArtifactRef {
            hash,
            size_bytes: bytes.len() as u64,
            media_type: "application/json".into(),
        })
    }
}

fn checkpoint_filename(checkpoint: &Checkpoint, canonical_hash: &str) -> String {
    format!(
        "{:012}-{}.json",
        checkpoint.created_at_sequence,
        &canonical_hash[..16]
    )
}

/// Load and validate one sealed checkpoint for [`crate::WorkflowExecutor::resume`].
pub fn load_checkpoint(path: &Path) -> Result<Checkpoint> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading checkpoint {}", path.display()))?;
    let checkpoint: Checkpoint = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing checkpoint {}", path.display()))?;
    let hash = checkpoint.canonical_hash()?;
    let expected_name = checkpoint_filename(&checkpoint, &hash);
    let actual_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow!(
                "checkpoint path has no valid UTF-8 filename: {}",
                path.display()
            )
        })?;
    ensure!(
        actual_name == expected_name,
        "checkpoint filename seal mismatch: expected {expected_name}, found {actual_name}"
    );
    Ok(checkpoint)
}

/// The highest-sequence sealed checkpoint in a directory, if any.
pub fn latest_checkpoint(directory: &Path) -> Result<Option<Checkpoint>> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("reading checkpoint directory {}", directory.display()));
        }
    };
    let mut latest: Option<Checkpoint> = None;
    for entry in entries {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        let checkpoint = load_checkpoint(&path)?;
        if latest
            .as_ref()
            .is_none_or(|current| checkpoint.created_at_sequence > current.created_at_sequence)
        {
            latest = Some(checkpoint);
        }
    }
    Ok(latest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TerminalOutcome, WorkflowExecutor};
    use hi_rsi_runtime::{
        ArtifactRef, BudgetKind, BudgetUsage, EngineeringPlan, RepositoryState, RuntimeBudgets,
        SharedBudgetLedger, WorkflowGraph,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;
    use std::time::Duration;

    struct TestAttestor;
    impl Attestor for TestAttestor {
        fn attest(&self, hash: &[u8; 32]) -> Result<String> {
            Ok(format!("test:{}", blake3_hex(hash)))
        }
    }
    fn blake3_hex(hash: &[u8; 32]) -> String {
        hash.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[derive(Default)]
    struct StubModel {
        attach_verification: bool,
    }
    #[async_trait]
    impl StageModel for StubModel {
        async fn invoke(
            &self,
            role: &str,
            stage: &StageId,
            _attempt: u32,
            state: &RunState,
        ) -> Result<StageOutcome> {
            let mut outcome = StageOutcome {
                passed: true,
                output: json!({ "role": role, "stage": stage.0 }),
                ..Default::default()
            };
            if role == "planner" {
                outcome.plan = Some(EngineeringPlan {
                    objective: "task".into(),
                    assumptions: vec![],
                    affected_components: vec![],
                    evidence: vec![],
                    proposed_changes: vec!["edit".into()],
                    tests: vec![],
                    risks: vec![],
                    rollback: "revert".into(),
                    revision: 1,
                    revision_reason: None,
                });
            }
            if role == "implementer" {
                outcome.patches = vec![ArtifactRef {
                    hash: "e".repeat(64),
                    size_bytes: 1,
                    media_type: "text/x-diff".into(),
                }];
            }
            if self.attach_verification {
                outcome.verification = Some(hi_rsi_runtime::VerificationReport {
                    report_version: 1,
                    run_id: state.run_id.clone(),
                    candidate_id: state.candidate_id.clone(),
                    environment_hash: "a".repeat(64),
                    source_tree_hash: "b".repeat(64),
                    checks: vec![],
                    passed: true,
                    policy_violations: vec![],
                    artifacts: vec![],
                    supervisor_attestation: Some("forged".into()),
                });
            }
            Ok(outcome)
        }
    }

    fn spec(name: &str, program: &str) -> CheckSpec {
        CheckSpec {
            name: name.into(),
            program: program.into(),
            arguments: vec![],
            timeout: Some(Duration::from_secs(30)),
            required: true,
            inherit_environment: false,
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
            budget: BudgetUsage::default(),
            failure_evidence: vec![],
        }
    }

    fn budgets() -> RuntimeBudgets {
        RuntimeBudgets {
            wall_time_seconds: 600,
            cpu_time_seconds: 600,
            memory_bytes: 1,
            disk_bytes: 1,
            input_tokens: 1,
            output_tokens: 1,
            tool_calls: 100,
            cost_microusd: 1,
            model_calls: 10,
            repair_iterations: 1,
            trace_bytes: 1,
        }
    }

    fn driver(checkpoints: &Path, model: StubModel) -> StageDriver<StubModel, TestAttestor> {
        StageDriver::new(
            model,
            AttestingVerifier::new(TestAttestor, "a".repeat(64)).unwrap(),
            vec![spec("verify_true", "true")],
            BTreeMap::from([
                ("cargo_check".to_string(), spec("cargo_check", "true")),
                ("cargo_test".to_string(), spec("cargo_test", "true")),
            ]),
            DenyGates,
            checkpoints.to_path_buf(),
        )
        .unwrap()
    }

    #[async_trait]
    impl<M: StageModel, A: Attestor, G: GateAuthority> TrustedStageDriver
        for Arc<StageDriver<M, A, G>>
    {
        async fn stage(
            &self,
            definition: &StageDefinition,
            stage: &StageId,
            attempt: u32,
            state: &RunState,
        ) -> Result<StageOutcome> {
            (**self).stage(definition, stage, attempt, state).await
        }
        async fn checkpoint(&self, checkpoint: &Checkpoint, reason: &str) -> Result<ArtifactRef> {
            (**self).checkpoint(checkpoint, reason).await
        }
    }

    #[tokio::test]
    async fn model_stage_cannot_attach_verification() {
        let base = tempfile::tempdir().unwrap();
        std::fs::write(base.path().join("src.rs"), "fn main() {}\n").unwrap();
        let driver = driver(
            &base.path().join("checkpoints"),
            StubModel {
                attach_verification: true,
            },
        );
        let error = driver
            .stage(
                &StageDefinition {
                    kind: StageKind::ModelInvocation,
                    model_role: Some("planner".into()),
                    tool: None,
                    iteration_limit: None,
                    trusted: false,
                },
                &StageId::from("plan"),
                1,
                &state(base.path()),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("may not attach verification"));
    }

    #[tokio::test]
    async fn unauthorized_tool_is_rejected_and_failing_tool_yields_candidate_evidence() {
        let base = tempfile::tempdir().unwrap();
        std::fs::write(base.path().join("src.rs"), "fn main() {}\n").unwrap();
        let mut harness = driver(&base.path().join("checkpoints"), StubModel::default());
        let tool_stage = |tool: &str| StageDefinition {
            kind: StageKind::ToolInvocation,
            model_role: None,
            tool: Some(tool.into()),
            iteration_limit: None,
            trusted: false,
        };
        let error = harness
            .stage(
                &tool_stage("rm_rf"),
                &StageId::from("compile"),
                1,
                &state(base.path()),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("unauthorized tool"));

        harness
            .tools
            .insert("failing".into(), spec("failing", "false"));
        let outcome = harness
            .stage(
                &tool_stage("failing"),
                &StageId::from("compile"),
                1,
                &state(base.path()),
            )
            .await
            .unwrap();
        assert!(!outcome.passed);
        assert_eq!(outcome.failures.len(), 1);
        assert_eq!(outcome.failures[0].domain, FailureDomain::Tool);
        assert!(outcome.failures[0].counts_against_candidate);
    }

    #[tokio::test]
    async fn default_coding_graph_runs_end_to_end_and_resumes_from_disk() {
        let base = tempfile::tempdir().unwrap();
        let worktree = base.path().join("worktree");
        std::fs::create_dir_all(&worktree).unwrap();
        std::fs::write(worktree.join("src.rs"), "fn main() {}\n").unwrap();
        let checkpoint_dir = base.path().join("checkpoints");

        let harness = Arc::new(driver(&checkpoint_dir, StubModel::default()));
        let mut run_state = state(&worktree);
        let result = WorkflowExecutor::new(
            WorkflowGraph::default_coding(),
            harness.clone(),
            SharedBudgetLedger::new(&budgets()),
        )
        .execute(&mut run_state)
        .await
        .unwrap();
        assert_eq!(result, TerminalOutcome::Succeeded);
        assert!(run_state.plan.is_some());
        assert_eq!(run_state.patches.len(), 1);
        assert_eq!(run_state.verification.len(), 1);
        assert!(
            run_state.verification[0]
                .supervisor_attestation
                .as_deref()
                .is_some_and(|attestation| attestation.starts_with("test:")),
            "verification gate must return a supervisor-attested report"
        );

        // The sealed checkpoints are durable and resumable from disk.
        let sealed = latest_checkpoint(&checkpoint_dir).unwrap().unwrap();
        assert!(!sealed.workflow_position.is_empty());
        let resumed_driver = Arc::new(driver(&checkpoint_dir, StubModel::default()));
        let mut resumed_state = state(&worktree);
        let resumed = WorkflowExecutor::new(
            WorkflowGraph::default_coding(),
            resumed_driver,
            SharedBudgetLedger::new(&budgets()),
        )
        .resume(&sealed, &mut resumed_state)
        .await
        .unwrap();
        assert_eq!(resumed, TerminalOutcome::Succeeded);
    }

    #[tokio::test]
    async fn checkpoint_filename_seal_rejects_tampered_accounting_and_sequence() {
        let base = tempfile::tempdir().unwrap();
        let checkpoint_dir = base.path().join("checkpoints");
        let harness = driver(&checkpoint_dir, StubModel::default());
        let mut checkpoint_state = state(base.path());
        checkpoint_state.attempts.insert(StageId::from("plan"), 2);
        checkpoint_state
            .budget
            .consumed
            .insert(BudgetKind::ToolCalls, 3);
        let checkpoint = Checkpoint {
            schema_version: 1,
            run_id: checkpoint_state.run_id.clone(),
            candidate_id: checkpoint_state.candidate_id.clone(),
            workspace_tree_hash: checkpoint_state.repository.source_tree_hash.clone(),
            state: checkpoint_state,
            workflow_position: BTreeSet::from([StageId::from("implement")]),
            context_manifests: vec![],
            response_artifacts: vec![],
            created_at_sequence: 7,
        };
        harness.checkpoint(&checkpoint, "test").await.unwrap();
        let path = std::fs::read_dir(&checkpoint_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
            .expect("sealed checkpoint file");

        let mut tampered_budget = checkpoint.clone();
        tampered_budget
            .state
            .budget
            .consumed
            .insert(BudgetKind::ToolCalls, 0);
        std::fs::write(&path, serde_json::to_vec_pretty(&tampered_budget).unwrap()).unwrap();
        let error = latest_checkpoint(&checkpoint_dir)
            .expect_err("latest checkpoint selection must reject a tampered budget");
        assert!(error.to_string().contains("filename seal mismatch"));

        let mut tampered_attempts = checkpoint.clone();
        tampered_attempts
            .state
            .attempts
            .insert(StageId::from("plan"), 1);
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&tampered_attempts).unwrap(),
        )
        .unwrap();
        let error = load_checkpoint(&path).expect_err("tampered attempts must invalidate the seal");
        assert!(error.to_string().contains("filename seal mismatch"));

        let mut tampered_sequence = checkpoint.clone();
        tampered_sequence.created_at_sequence = 8;
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&tampered_sequence).unwrap(),
        )
        .unwrap();
        let error = load_checkpoint(&path).expect_err("tampered sequence must invalidate the seal");
        assert!(error.to_string().contains("filename seal mismatch"));

        std::fs::write(&path, serde_json::to_vec_pretty(&checkpoint).unwrap()).unwrap();
        assert_eq!(load_checkpoint(&path).unwrap(), checkpoint);
    }
}
