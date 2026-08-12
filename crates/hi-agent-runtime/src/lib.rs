//! Evolvable workflow state machine. Trusted effects are delegated to a driver.
//!
//! Stages sharing a frontier (a `ParallelFanOut`'s successors) execute
//! concurrently in waves bounded by the graph's parallelism ceiling; outcomes
//! apply in deterministic stage-id order so parallel and serial runs converge
//! on identical state. Checkpoints seal the *next* frontier, and
//! [`WorkflowExecutor::resume`] continues a run from any sealed checkpoint.
//!
//! Ownership: RSI workflow path only — not `hi-agent::run_turn`. See `docs/architecture.md`.
use std::collections::BTreeSet;

pub mod driver;
pub use driver::{
    DenyGates, GateAuthority, StageDriver, StageModel, latest_checkpoint, load_checkpoint,
};

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
pub trait TrustedStageDriver: Send + Sync {
    /// Runs one stage attempt. Takes `&self` because stages on a parallel
    /// frontier execute concurrently against a shared driver; implementations
    /// use interior mutability for any per-stage bookkeeping.
    async fn stage(
        &self,
        definition: &StageDefinition,
        stage: &StageId,
        attempt: u32,
        state: &RunState,
    ) -> Result<StageOutcome>;

    async fn checkpoint(&self, checkpoint: &Checkpoint, reason: &str) -> Result<ArtifactRef>;
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

    /// Run the workflow from its entry stage.
    pub async fn execute(self, state: &mut RunState) -> Result<TerminalOutcome> {
        let entry = self.graph.entry.clone();
        self.run(state, BTreeSet::from([entry])).await
    }

    /// Resume a run from a checkpoint. The checkpoint's `workflow_position`
    /// is the frontier scheduled at the moment it was sealed; execution
    /// continues there against the checkpointed state, and the checkpoint
    /// sequence keeps advancing monotonically so resumed checkpoints never
    /// collide with pre-crash ones.
    pub async fn resume(
        mut self,
        checkpoint: &Checkpoint,
        state: &mut RunState,
    ) -> Result<TerminalOutcome> {
        ensure!(
            checkpoint.schema_version == 1,
            "unsupported checkpoint schema version {}",
            checkpoint.schema_version
        );
        // Validates checkpoint/state identity coherence as a side effect.
        checkpoint.canonical_hash()?;
        // Bind the checkpoint to *this* run before letting it replace state:
        // `canonical_hash` only proves the checkpoint is internally consistent
        // (its own run_id matches its embedded state's), not that it belongs to
        // the run the caller is resuming. Without this, any sealed checkpoint
        // from any run could be resumed into this executor, silently replacing
        // the run's identity and history.
        ensure!(
            checkpoint.run_id == state.run_id && checkpoint.candidate_id == state.candidate_id,
            "checkpoint identity ({}, {}) does not match the run being resumed ({}, {})",
            checkpoint.run_id,
            checkpoint.candidate_id,
            state.run_id,
            state.candidate_id
        );
        ensure!(
            !checkpoint.workflow_position.is_empty(),
            "checkpoint has no resumable workflow position"
        );
        for stage in &checkpoint.workflow_position {
            ensure!(
                self.graph.stages.contains_key(stage),
                "checkpoint position references missing stage {}",
                stage.0
            );
        }
        self.sequence = checkpoint.created_at_sequence;
        *state = checkpoint.state.clone();
        let frontier = checkpoint.workflow_position.clone();
        self.run(state, frontier).await
    }

    /// Drive the frontier to a terminal stage. Every stage on the frontier
    /// runs before the frontier advances; stages within it execute
    /// concurrently (in waves bounded by `maximum_parallelism`) against an
    /// immutable snapshot of the pre-frontier state, and their outcomes are
    /// applied in deterministic stage-id order so a parallel run reaches the
    /// same state as a serial one.
    async fn run(
        mut self,
        state: &mut RunState,
        mut frontier: BTreeSet<StageId>,
    ) -> Result<TerminalOutcome> {
        let mut transitions = 0_u32;
        while !frontier.is_empty() {
            let mut batch = Vec::with_capacity(frontier.len());
            let mut terminal_success = None;
            for stage_id in &frontier {
                let definition = self
                    .graph
                    .stages
                    .get(stage_id)
                    .cloned()
                    .ok_or_else(|| anyhow!("workflow selected missing stage {}", stage_id.0))?;
                match definition.kind {
                    // A failure anywhere on the frontier fails the run before
                    // any sibling work is spent.
                    StageKind::TerminalFailure => {
                        state.current_stages.clear();
                        return Ok(TerminalOutcome::Failed);
                    }
                    StageKind::TerminalSuccess => terminal_success = Some(stage_id.clone()),
                    _ => batch.push((stage_id.clone(), definition)),
                }
            }
            if let Some(stage_id) = terminal_success {
                ensure!(
                    batch.is_empty(),
                    "terminal success {} reached while other stages are still scheduled",
                    stage_id.0
                );
                ensure!(
                    state.verification.last().is_some_and(
                        |report| report.passed && report.validate_supervisor_report().is_ok()
                    ),
                    "terminal success requires a passing trusted verification report"
                );
                state.current_stages.clear();
                return Ok(TerminalOutcome::Succeeded);
            }

            // Pre-stage accounting for the whole frontier, in stage-id order.
            // Attempt counts are staged locally and only merged into `state`
            // after the whole frontier succeeds, so a failed wave or rejected
            // outcome never burns an iteration the run did not complete.
            let mut staged_attempts = state.attempts.clone();
            let mut attempts = Vec::with_capacity(batch.len());
            for (stage_id, definition) in &batch {
                transitions = transitions
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("transition count overflow"))?;
                ensure!(
                    transitions <= self.graph.limits.maximum_transitions,
                    "workflow transition budget exhausted"
                );
                self.ledger.consume(BudgetKind::ToolCalls, 0)?;
                let attempt = staged_attempts.entry(stage_id.clone()).or_default();
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
                attempts.push(*attempt);
            }
            state.current_stages = frontier.clone();

            // Execute concurrently against a shared snapshot, bounded by the
            // graph's parallelism ceiling.
            let snapshot = state.clone();
            let mut outcomes = Vec::with_capacity(batch.len());
            let wave_width = usize::from(self.graph.limits.effective_concurrency());
            for wave in batch.chunks(wave_width) {
                let offset = outcomes.len();
                let wave_outcomes =
                    futures_util::future::try_join_all(wave.iter().enumerate().map(
                        |(index, (stage_id, definition))| {
                            let attempt = attempts[offset + index];
                            self.driver.stage(definition, stage_id, attempt, &snapshot)
                        },
                    ))
                    .await?;
                outcomes.extend(wave_outcomes);
            }

            // Apply deterministically, compute the next frontier, and seal one
            // checkpoint for the whole frontier if any stage demands it. The
            // committed state only moves forward after a required checkpoint
            // seals, so a crash resumes from a consistent boundary. Staged
            // attempt counts merge here — only for a frontier that succeeded.
            let mut next_state = state.clone();
            next_state.attempts = staged_attempts;
            let mut next_frontier = BTreeSet::new();
            let mut checkpoint_stage: Option<(StageId, StageDefinition, StageOutcome)> = None;
            let mut batch_patches = Vec::new();
            for ((stage_id, definition), outcome) in batch.iter().zip(outcomes) {
                self.apply_outcome(&mut next_state, stage_id, definition, outcome.clone())?;
                batch_patches.extend(outcome.patches.clone());
                let mut eligible: Vec<_> = self
                    .graph
                    .edges
                    .iter()
                    .filter(|edge| {
                        &edge.from == stage_id && condition_matches(edge.condition, outcome.passed)
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
                    // A shared successor joins: the set deduplicates, and the
                    // joined stage runs once, only after this whole frontier.
                    next_frontier.extend(eligible.into_iter().map(|edge| edge.to.clone()));
                } else {
                    next_frontier.insert(eligible[0].to.clone());
                }
                if checkpoint_stage.is_none() && requires_checkpoint(stage_id, definition, &outcome)
                {
                    checkpoint_stage = Some((stage_id.clone(), definition.clone(), outcome));
                }
            }
            let next_sequence = self
                .sequence
                .checked_add(1)
                .ok_or_else(|| anyhow!("checkpoint sequence overflow"))?;
            if let Some((stage_id, definition, outcome)) = checkpoint_stage {
                let checkpoint = Checkpoint {
                    schema_version: 1,
                    run_id: next_state.run_id.clone(),
                    candidate_id: next_state.candidate_id.clone(),
                    state: next_state.clone(),
                    workspace_tree_hash: next_state.repository.source_tree_hash.clone(),
                    workflow_position: next_frontier.clone(),
                    context_manifests: vec![],
                    response_artifacts: batch_patches,
                    created_at_sequence: next_sequence,
                };
                self.driver
                    .checkpoint(
                        &checkpoint,
                        checkpoint_reason(&stage_id, &definition, &outcome),
                    )
                    .await?;
            }
            *state = next_state;
            self.sequence = next_sequence;
            frontier = next_frontier;
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
        let original = state.clone();
        if let Err(error) = self.apply_outcome_inner(state, stage, definition, outcome) {
            *state = original;
            return Err(error);
        }
        Ok(())
    }

    fn apply_outcome_inner(
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
        state.budget = self.ledger.usage().inspect_err(|_| {
            state.failure_evidence.push(FailureEvidence {
                domain: hi_rsi_runtime::FailureDomain::Budget,
                subcategory: "ledger_unavailable".into(),
                retryable: false,
                causal_event_hash: None,
                stage: stage.clone(),
                artifacts: vec![],
                counts_against_candidate: false,
            });
        })?;
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

    #[derive(Default)]
    struct Driver {
        invoked: std::sync::Mutex<Vec<String>>,
        checkpoints: std::sync::Mutex<Vec<Checkpoint>>,
        fail_checkpoint: bool,
        barrier: Option<std::sync::Arc<tokio::sync::Barrier>>,
    }
    #[async_trait]
    impl TrustedStageDriver for Driver {
        async fn stage(
            &self,
            definition: &StageDefinition,
            stage: &StageId,
            _: u32,
            state: &RunState,
        ) -> Result<StageOutcome> {
            self.invoked.lock().unwrap().push(stage.0.clone());
            if let Some(barrier) = &self.barrier
                && stage.0.starts_with("branch_")
            {
                // Proves overlap: both branches must be in flight at once, or
                // the serial fallback deadlocks and the test times out.
                tokio::time::timeout(std::time::Duration::from_secs(5), barrier.wait())
                    .await
                    .map_err(|_| anyhow!("parallel branches never overlapped"))?;
            }
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
        async fn checkpoint(&self, checkpoint: &Checkpoint, _: &str) -> Result<ArtifactRef> {
            self.checkpoints.lock().unwrap().push(checkpoint.clone());
            if self.fail_checkpoint {
                bail!("checkpoint failed");
            }
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

    #[async_trait]
    impl TrustedStageDriver for std::sync::Arc<Driver> {
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
    async fn executes_manifest_graph_through_trusted_verification() {
        let graph = WorkflowGraph::default_coding();
        let mut state = state();
        let result = WorkflowExecutor::new(
            graph,
            std::sync::Arc::new(Driver::default()),
            SharedBudgetLedger::new(&budgets()),
        )
        .execute(&mut state)
        .await
        .unwrap();
        assert_eq!(result, TerminalOutcome::Succeeded);
        assert!(state.plan.is_some());
        assert_eq!(state.verification.len(), 1);
    }

    fn fan_out_graph() -> WorkflowGraph {
        use StageKind::*;
        let stages: BTreeMap<StageId, StageDefinition> = [
            ("seed", DeterministicTransform, false),
            ("scatter", ParallelFanOut, false),
            ("branch_a", DeterministicTransform, false),
            ("branch_b", DeterministicTransform, false),
            ("join", Aggregation, false),
            ("verify", VerificationGate, true),
            ("complete", TerminalSuccess, false),
        ]
        .into_iter()
        .map(|(name, kind, trusted)| {
            (
                StageId::from(name),
                StageDefinition {
                    kind,
                    model_role: None,
                    tool: None,
                    iteration_limit: None,
                    trusted,
                },
            )
        })
        .collect();
        let edges = [
            ("seed", "scatter"),
            ("scatter", "branch_a"),
            ("scatter", "branch_b"),
            ("branch_a", "join"),
            ("branch_b", "join"),
            ("join", "verify"),
            ("verify", "complete"),
        ]
        .into_iter()
        .enumerate()
        .map(|(priority, (from, to))| hi_rsi_runtime::TransitionRule {
            from: StageId::from(from),
            to: StageId::from(to),
            condition: TransitionCondition::StagePassed,
            priority: priority as u16,
        })
        .collect();
        WorkflowGraph {
            entry: StageId::from("seed"),
            stages,
            edges,
            limits: hi_rsi_runtime::WorkflowLimits {
                maximum_transitions: 100,
                maximum_parallelism: 4,
                maximum_concurrency: None,
            },
        }
    }

    #[tokio::test]
    async fn fan_out_branches_run_concurrently_and_join_exactly_once() {
        let driver = std::sync::Arc::new(Driver {
            barrier: Some(std::sync::Arc::new(tokio::sync::Barrier::new(2))),
            ..Driver::default()
        });
        let mut state = state();
        let result = WorkflowExecutor::new(
            fan_out_graph(),
            driver.clone(),
            SharedBudgetLedger::new(&budgets()),
        )
        .execute(&mut state)
        .await
        .unwrap();
        assert_eq!(result, TerminalOutcome::Succeeded);
        let invoked = driver.invoked.lock().unwrap().clone();
        // The barrier only releases when both branches are in flight at the
        // same time — reaching here proves genuine concurrency.
        assert!(invoked.contains(&"branch_a".to_string()));
        assert!(invoked.contains(&"branch_b".to_string()));
        assert_eq!(
            invoked.iter().filter(|stage| *stage == "join").count(),
            1,
            "shared successor must join the fan-out exactly once: {invoked:?}"
        );
    }

    #[tokio::test]
    async fn failed_wave_does_not_burn_stage_attempts() {
        // A stage whose driver errors on the first try must not consume an
        // iteration: the failed wave never completed, so a retry resumes with
        // the original attempt budget intact.
        struct FlakyDriver {
            calls: std::sync::Arc<std::sync::Mutex<u32>>,
        }
        #[async_trait]
        impl TrustedStageDriver for FlakyDriver {
            async fn stage(
                &self,
                definition: &StageDefinition,
                stage: &StageId,
                attempt: u32,
                state: &RunState,
            ) -> Result<StageOutcome> {
                let call = {
                    let mut calls = self.calls.lock().unwrap();
                    *calls += 1;
                    *calls
                };
                if call == 1 {
                    bail!("transient first-call failure");
                }
                Driver::default()
                    .stage(definition, stage, attempt, state)
                    .await
            }
            async fn checkpoint(&self, _: &Checkpoint, _: &str) -> Result<ArtifactRef> {
                Ok(ArtifactRef {
                    hash: "d".repeat(64),
                    size_bytes: 1,
                    media_type: "application/json".into(),
                })
            }
        }

        let calls = std::sync::Arc::new(std::sync::Mutex::new(0));
        let mut state = state();

        let first = WorkflowExecutor::new(
            WorkflowGraph::default_coding(),
            FlakyDriver {
                calls: calls.clone(),
            },
            SharedBudgetLedger::new(&budgets()),
        )
        .execute(&mut state)
        .await;
        assert!(first.is_err(), "first wave fails on the transient error");
        assert!(
            state.attempts.values().all(|attempt| *attempt == 0),
            "a failed wave must not burn attempts: {:?}",
            state.attempts
        );

        let second = WorkflowExecutor::new(
            WorkflowGraph::default_coding(),
            FlakyDriver { calls },
            SharedBudgetLedger::new(&budgets()),
        )
        .execute(&mut state)
        .await
        .unwrap();
        assert_eq!(second, TerminalOutcome::Succeeded);
    }

    #[tokio::test]
    async fn resume_continues_from_a_sealed_checkpoint_frontier() {
        let first = std::sync::Arc::new(Driver::default());
        let mut state1 = state();
        WorkflowExecutor::new(
            WorkflowGraph::default_coding(),
            first.clone(),
            SharedBudgetLedger::new(&budgets()),
        )
        .execute(&mut state1)
        .await
        .unwrap();
        let plan_boundary = first
            .checkpoints
            .lock()
            .unwrap()
            .iter()
            .find(|checkpoint| {
                checkpoint.workflow_position == BTreeSet::from([StageId::from("implement")])
            })
            .cloned()
            .expect("plan stage seals a checkpoint whose frontier is implement");

        let resumed = std::sync::Arc::new(Driver::default());
        let mut state2 = state();
        let result = WorkflowExecutor::new(
            WorkflowGraph::default_coding(),
            resumed.clone(),
            SharedBudgetLedger::new(&budgets()),
        )
        .resume(&plan_boundary, &mut state2)
        .await
        .unwrap();

        assert_eq!(result, TerminalOutcome::Succeeded);
        let invoked = resumed.invoked.lock().unwrap().clone();
        assert_eq!(
            invoked,
            vec!["implement", "compile", "test", "review", "verify"],
            "resume must continue at the checkpointed frontier, not the entry"
        );
        assert!(
            state2.plan.is_some(),
            "checkpointed state carries the plan forward"
        );
        assert!(
            resumed
                .checkpoints
                .lock()
                .unwrap()
                .iter()
                .all(|checkpoint| {
                    checkpoint.created_at_sequence > plan_boundary.created_at_sequence
                }),
            "resumed checkpoint sequence continues monotonically"
        );
    }

    #[tokio::test]
    async fn resume_rejects_a_checkpoint_from_a_different_run() {
        let first = std::sync::Arc::new(Driver::default());
        let mut state1 = state();
        WorkflowExecutor::new(
            WorkflowGraph::default_coding(),
            first.clone(),
            SharedBudgetLedger::new(&budgets()),
        )
        .execute(&mut state1)
        .await
        .unwrap();
        let checkpoint = first
            .checkpoints
            .lock()
            .unwrap()
            .iter()
            .find(|checkpoint| {
                checkpoint.workflow_position == BTreeSet::from([StageId::from("implement")])
            })
            .cloned()
            .expect("plan stage seals a checkpoint whose frontier is implement");

        // The caller's state carries a *different* run identity than the
        // checkpoint — resume must refuse rather than let the foreign
        // checkpoint replace this run's state.
        let mut foreign = state();
        foreign.run_id = "someone-elses-run".into();
        let result = WorkflowExecutor::new(
            WorkflowGraph::default_coding(),
            std::sync::Arc::new(Driver::default()),
            SharedBudgetLedger::new(&budgets()),
        )
        .resume(&checkpoint, &mut foreign)
        .await;
        let err = result.expect_err("a checkpoint from another run must not resume");
        assert!(
            err.to_string().contains("does not match the run being resumed"),
            "unexpected error: {err}"
        );
        // State is untouched: the foreign run identity was not overwritten.
        assert_eq!(foreign.run_id, "someone-elses-run");
        assert!(foreign.plan.is_none());
    }
}
