//! Evolvable workflow state machine. Trusted effects are delegated to a driver.

use std::collections::{BTreeSet, VecDeque};

use anyhow::{Result, anyhow, bail, ensure};
use async_trait::async_trait;
use hi_rsi_runtime::{
    ArtifactRef, BudgetKind, Checkpoint, EngineeringPlan, FailureEvidence, RunState,
    SharedBudgetLedger, StageDefinition, StageId, StageKind, TransitionCondition,
    VerificationReport, WorkflowGraph,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StageOutcome {
    pub passed: bool,
    pub output: Value,
    pub plan: Option<EngineeringPlan>,
    pub patches: Vec<ArtifactRef>,
    pub failures: Vec<FailureEvidence>,
    pub verification: Option<VerificationReport>,
}

#[async_trait]
pub trait TrustedStageDriver: Send {
    async fn stage(
        &mut self,
        definition: &StageDefinition,
        stage: &StageId,
        attempt: u32,
        state: &RunState,
    ) -> Result<StageOutcome>;

    async fn checkpoint(&mut self, checkpoint: &Checkpoint, reason: &str) -> Result<ArtifactRef>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TerminalOutcome {
    Succeeded,
    Failed,
}

pub struct WorkflowExecutor<D> {
    graph: WorkflowGraph,
    driver: D,
    ledger: SharedBudgetLedger,
    sequence: u64,
}

impl<D: TrustedStageDriver> WorkflowExecutor<D> {
    pub fn new(graph: WorkflowGraph, driver: D, ledger: SharedBudgetLedger) -> Self {
        Self {
            graph,
            driver,
            ledger,
            sequence: 0,
        }
    }

    pub async fn execute(mut self, state: &mut RunState) -> Result<TerminalOutcome> {
        let mut ready = VecDeque::from([self.graph.entry.clone()]);
        let mut transitions = 0_u32;
        while let Some(stage_id) = ready.pop_front() {
            let definition = self
                .graph
                .stages
                .get(&stage_id)
                .cloned()
                .ok_or_else(|| anyhow!("workflow selected missing stage {}", stage_id.0))?;
            if definition.kind == StageKind::TerminalSuccess {
                ensure!(
                    state.verification.last().is_some_and(
                        |report| report.passed && report.validate_supervisor_report().is_ok()
                    ),
                    "terminal success requires a passing trusted verification report"
                );
                state.current_stages.clear();
                return Ok(TerminalOutcome::Succeeded);
            }
            if definition.kind == StageKind::TerminalFailure {
                state.current_stages.clear();
                return Ok(TerminalOutcome::Failed);
            }

            transitions = transitions
                .checked_add(1)
                .ok_or_else(|| anyhow!("transition count overflow"))?;
            ensure!(
                transitions <= self.graph.limits.maximum_transitions,
                "workflow transition budget exhausted"
            );
            self.ledger.consume(BudgetKind::ToolCalls, 0)?;
            let attempt = state.attempts.entry(stage_id.clone()).or_default();
            *attempt = attempt
                .checked_add(1)
                .ok_or_else(|| anyhow!("stage attempt overflow"))?;
            if let Some(limit) = definition.iteration_limit {
                ensure!(
                    *attempt <= limit,
                    "stage {} iteration limit exhausted",
                    stage_id.0
                );
            }
            let attempt = *attempt;
            state.current_stages = BTreeSet::from([stage_id.clone()]);
            let outcome = self
                .driver
                .stage(&definition, &stage_id, attempt, state)
                .await?;
            self.apply_outcome(state, &stage_id, &definition, outcome.clone())?;
            self.sequence += 1;

            if requires_checkpoint(&stage_id, &definition, &outcome) {
                let checkpoint = Checkpoint {
                    schema_version: 1,
                    run_id: state.run_id.clone(),
                    candidate_id: state.candidate_id.clone(),
                    state: state.clone(),
                    workspace_tree_hash: state.repository.source_tree_hash.clone(),
                    workflow_position: BTreeSet::from([stage_id.clone()]),
                    context_manifests: vec![],
                    response_artifacts: outcome.patches.clone(),
                    created_at_sequence: self.sequence,
                };
                self.driver
                    .checkpoint(
                        &checkpoint,
                        checkpoint_reason(&stage_id, &definition, &outcome),
                    )
                    .await?;
            }

            let mut eligible: Vec<_> = self
                .graph
                .edges
                .iter()
                .filter(|edge| {
                    edge.from == stage_id && condition_matches(edge.condition, outcome.passed)
                })
                .collect();
            eligible.sort_by_key(|edge| edge.priority);
            ensure!(
                !eligible.is_empty(),
                "stage {} has no eligible transition",
                stage_id.0
            );
            if definition.kind == StageKind::ParallelFanOut {
                ensure!(
                    eligible.len() <= usize::from(self.graph.limits.maximum_parallelism),
                    "parallelism ceiling exceeded"
                );
                for edge in eligible {
                    ready.push_back(edge.to.clone());
                }
            } else {
                ready.push_back(eligible[0].to.clone());
            }
        }
        bail!("workflow ended without a terminal stage")
    }

    fn apply_outcome(
        &self,
        state: &mut RunState,
        stage: &StageId,
        definition: &StageDefinition,
        outcome: StageOutcome,
    ) -> Result<()> {
        for failure in &outcome.failures {
            failure.validate()?;
        }
        state.failure_evidence.extend(outcome.failures);
        if let Some(plan) = outcome.plan {
            if let Some(previous) = &state.plan {
                ensure!(
                    plan.revision > previous.revision,
                    "plan replacement requires a revision record"
                );
                ensure!(
                    plan.revision_reason
                        .as_ref()
                        .is_some_and(|v| !v.trim().is_empty()),
                    "plan revision requires a reason"
                );
            }
            state.plan = Some(plan);
        }
        if !outcome.patches.is_empty() {
            ensure!(
                state.plan.is_some(),
                "implementation cannot create patches before a typed plan"
            );
            state.patches.extend(outcome.patches);
        }
        if let Some(report) = outcome.verification {
            ensure!(
                definition.kind == StageKind::VerificationGate && definition.trusted,
                "only a trusted verification gate may return verification"
            );
            report.validate_supervisor_report()?;
            ensure!(
                report.run_id == state.run_id && report.candidate_id == state.candidate_id,
                "verification identity mismatch"
            );
            state.verification.push(report);
        } else if definition.kind == StageKind::VerificationGate {
            bail!("trusted verification gate {} omitted its report", stage.0);
        }
        state.budget = self.ledger.usage()?;
        Ok(())
    }
}

fn condition_matches(condition: TransitionCondition, passed: bool) -> bool {
    match condition {
        TransitionCondition::Always | TransitionCondition::BudgetRemaining => true,
        TransitionCondition::StagePassed | TransitionCondition::HumanApproved => passed,
        TransitionCondition::StageFailed => !passed,
    }
}

fn requires_checkpoint(
    stage: &StageId,
    definition: &StageDefinition,
    outcome: &StageOutcome,
) -> bool {
    matches!(
        stage.0.as_str(),
        "explore_repository" | "diagnose" | "plan" | "repair"
    ) || !outcome.patches.is_empty()
        || definition.kind == StageKind::VerificationGate
}

fn checkpoint_reason<'a>(
    stage: &'a StageId,
    definition: &StageDefinition,
    outcome: &StageOutcome,
) -> &'a str {
    if definition.kind == StageKind::VerificationGate {
        "verification_boundary"
    } else if !outcome.patches.is_empty() {
        "patch_batch"
    } else {
        stage.0.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_rsi_runtime::{
        BudgetUsage, RepositoryState, RuntimeBudgets, VerificationCheck, VerificationStatus,
    };
    use std::collections::{BTreeMap, BTreeSet};

    struct Driver {
        checkpoints: usize,
    }
    #[async_trait]
    impl TrustedStageDriver for Driver {
        async fn stage(
            &mut self,
            definition: &StageDefinition,
            stage: &StageId,
            _: u32,
            state: &RunState,
        ) -> Result<StageOutcome> {
            let mut outcome = StageOutcome {
                passed: true,
                ..Default::default()
            };
            if stage.0 == "plan" {
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
            if definition.kind == StageKind::VerificationGate {
                outcome.verification = Some(VerificationReport {
                    report_version: 1,
                    run_id: state.run_id.clone(),
                    candidate_id: state.candidate_id.clone(),
                    environment_hash: "a".repeat(64),
                    source_tree_hash: "b".repeat(64),
                    checks: vec![VerificationCheck {
                        name: "test".into(),
                        command_hash: "c".repeat(64),
                        status: VerificationStatus::Passed,
                        exit_code: Some(0),
                        duration_ms: 1,
                        output: None,
                    }],
                    passed: true,
                    policy_violations: vec![],
                    artifacts: vec![],
                    supervisor_attestation: Some("trusted".into()),
                });
            }
            Ok(outcome)
        }
        async fn checkpoint(&mut self, _: &Checkpoint, _: &str) -> Result<ArtifactRef> {
            self.checkpoints += 1;
            Ok(ArtifactRef {
                hash: "d".repeat(64),
                size_bytes: 1,
                media_type: "application/json".into(),
            })
        }
    }

    fn budgets() -> RuntimeBudgets {
        RuntimeBudgets {
            wall_time_seconds: 60,
            cpu_time_seconds: 60,
            memory_bytes: 1,
            disk_bytes: 1,
            input_tokens: 1,
            output_tokens: 1,
            tool_calls: 10,
            cost_microusd: 1,
            model_calls: 1,
            repair_iterations: 1,
            trace_bytes: 1,
        }
    }
    fn state() -> RunState {
        RunState {
            task_id: "t".into(),
            run_id: "r".into(),
            candidate_id: "c".into(),
            repository: RepositoryState {
                repository_snapshot_hash: "a".repeat(64),
                starting_commit: "x".into(),
                source_tree_hash: "b".repeat(64),
                worktree_root: "/tmp/work".into(),
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

    #[tokio::test]
    async fn executes_manifest_graph_through_trusted_verification() {
        let graph = WorkflowGraph::default_coding();
        let mut state = state();
        let result = WorkflowExecutor::new(
            graph,
            Driver { checkpoints: 0 },
            SharedBudgetLedger::new(&budgets()),
        )
        .execute(&mut state)
        .await
        .unwrap();
        assert_eq!(result, TerminalOutcome::Succeeded);
        assert!(state.plan.is_some());
        assert_eq!(state.verification.len(), 1);
    }
}
