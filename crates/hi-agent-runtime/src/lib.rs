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
    ArtifactRef, BudgetKind, BudgetReservation, BudgetUsage, Checkpoint, EngineeringPlan,
    FailureEvidence, RunState, SharedBudgetLedger, StageDefinition, StageId, StageKind,
    TransitionCondition, VerificationReport, WorkflowGraph,
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
    /// use interior mutability for any per-stage bookkeeping. The executor
    /// owns `ModelCalls` and `ToolCalls` admission; drivers only account for
    /// usage learned from the underlying operation, such as token counts.
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

/// Holds whole-frontier model/tool admission across bounded execution waves.
/// Each reservation moves into its stage future and commits on that future's
/// first poll. Unpolled siblings and later waves therefore release rather than
/// charge when an earlier stage errors or the workflow future is cancelled.
struct FrontierReservations {
    ledger: SharedBudgetLedger,
    waves: Vec<Option<Vec<StageReservation>>>,
}

impl FrontierReservations {
    fn new(ledger: SharedBudgetLedger, capacity: usize) -> Self {
        Self {
            ledger,
            waves: Vec::with_capacity(capacity),
        }
    }

    fn reserve_wave(&mut self, kinds: impl IntoIterator<Item = Option<BudgetKind>>) -> Result<()> {
        self.waves.push(Some(Vec::new()));
        let wave = self
            .waves
            .last_mut()
            .and_then(Option::as_mut)
            .expect("the wave was inserted immediately above");
        for kind in kinds {
            wave.push(StageReservation::new(self.ledger.clone(), kind)?);
        }
        Ok(())
    }

    fn take_wave(&mut self, index: usize) -> Vec<StageReservation> {
        self.waves[index]
            .take()
            .expect("each reserved wave is started at most once")
    }
}

struct StageReservation {
    ledger: SharedBudgetLedger,
    reservation: Option<BudgetReservation>,
}

impl StageReservation {
    fn new(ledger: SharedBudgetLedger, kind: Option<BudgetKind>) -> Result<Self> {
        let reservation = kind.map(|kind| ledger.reserve(kind, 1)).transpose()?;
        Ok(Self {
            ledger,
            reservation,
        })
    }

    fn commit(&mut self) -> Result<()> {
        if let Some(reservation) = self.reservation {
            self.ledger.commit(reservation, 1)?;
            self.reservation = None;
        }
        Ok(())
    }
}

impl Drop for StageReservation {
    fn drop(&mut self) {
        if let Some(reservation) = self.reservation.take() {
            let _ = self.ledger.release(reservation);
        }
    }
}

/// Keeps the caller-visible durable budget snapshot coherent when an async
/// operation is dropped. Stage reservations commit on first poll, and drivers
/// may also consume provider-reported usage before a later internal await, so
/// cancellation can strand irrevocable spend in the shared ledger unless it
/// is copied back during future destruction.
struct BudgetStateSync<'a> {
    budget: &'a mut BudgetUsage,
    ledger: SharedBudgetLedger,
}

impl<'a> BudgetStateSync<'a> {
    fn new(budget: &'a mut BudgetUsage, ledger: SharedBudgetLedger) -> Self {
        Self { budget, ledger }
    }
}

impl Drop for BudgetStateSync<'_> {
    fn drop(&mut self) {
        if let Ok(mut usage) = self.ledger.usage() {
            // Later tool waves are still only reserved and are released by
            // `FrontierReservations` after this guard drops. Reservations are
            // transient admission state and must never become durable state.
            usage.reserved.clear();
            *self.budget = usage;
        }
    }
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
        // A caller may retry after dropping an earlier workflow future. Its
        // RunState carries irrevocable calls/tokens published by cancellation
        // guards, so merge that durable floor before admitting entry work.
        state.budget = self.ledger.merge_consumption_floor(&state.budget)?;
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
        ensure!(
            state.budget.reserved.values().all(|amount| *amount == 0),
            "caller state contains unsettled budget reservations"
        );
        // The checkpoint can predate a cancelled post-checkpoint attempt whose
        // irrevocable spend was published into caller state. Merge both as a
        // monotonic floor; the ledger method also preserves any usage already
        // present when an in-process retry reuses the same shared ledger.
        let mut budget_floor = checkpoint.state.budget.clone();
        for (&kind, &amount) in &state.budget.consumed {
            let consumed = budget_floor.consumed.entry(kind).or_default();
            *consumed = (*consumed).max(amount);
        }
        let merged_budget = self.ledger.merge_consumption_floor(&budget_floor)?;
        self.sequence = checkpoint.created_at_sequence;
        *state = checkpoint.state.clone();
        state.budget = merged_budget;
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
        // Successful nonterminal stage attempts are the durable transition
        // counter. Reconstruct it from state so a checkpoint resume cannot
        // reset and repeatedly bypass `maximum_transitions`.
        let mut transitions = state.attempts.values().try_fold(0_u32, |total, attempts| {
            total
                .checked_add(*attempts)
                .ok_or_else(|| anyhow!("checkpoint transition count overflow"))
        })?;
        ensure!(
            transitions <= self.graph.limits.maximum_transitions,
            "checkpoint transition count exceeds workflow limit"
        );
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
            let wave_width = usize::from(self.graph.limits.effective_concurrency());
            let wave_count = batch.len().div_ceil(wave_width);
            let mut frontier_reservations =
                FrontierReservations::new(self.ledger.clone(), wave_count);
            // Admit every model and tool call in the complete frontier before
            // invoking any stage. Otherwise one request/tool can begin and a
            // sibling or later wave can discover insufficient budget,
            // discarding the first outcome and causing it to run again on
            // retry.
            for wave in batch.chunks(wave_width) {
                frontier_reservations.reserve_wave(wave.iter().map(|(_, definition)| {
                    match definition.kind {
                        StageKind::ToolInvocation => Some(BudgetKind::ToolCalls),
                        StageKind::ModelInvocation => Some(BudgetKind::ModelCalls),
                        _ => None,
                    }
                }))?;
            }
            state.current_stages = frontier.clone();

            // Execute concurrently against a shared snapshot, bounded by the
            // graph's parallelism ceiling.
            let snapshot = state.clone();
            let mut outcomes = Vec::with_capacity(batch.len());
            for (wave_index, wave) in batch.chunks(wave_width).enumerate() {
                let stage_reservations = frontier_reservations.take_wave(wave_index);
                // A stage commits its admitted model/tool call on first poll,
                // immediately before invoking the driver. Do not expose
                // still-pending sibling/later-wave reservations in durable
                // state: their guards release them if never polled.
                let mut settled_usage = self.ledger.usage()?;
                settled_usage.reserved.clear();
                state.budget = settled_usage;
                let offset = outcomes.len();
                let driver = &self.driver;
                let wave_snapshot = &snapshot;
                let wave_result = {
                    // This guard lives across the await so dropping the
                    // workflow future synchronously publishes any spend a
                    // driver incurred before it became pending.
                    let _budget_sync = BudgetStateSync::new(&mut state.budget, self.ledger.clone());
                    futures_util::future::try_join_all(
                        wave.iter().zip(stage_reservations).enumerate().map(
                            |(index, ((stage_id, definition), mut reservation))| {
                                let attempt = attempts[offset + index];
                                async move {
                                    // The async body does not run until this exact
                                    // stage future is polled.
                                    reservation.commit()?;
                                    driver
                                        .stage(definition, stage_id, attempt, wave_snapshot)
                                        .await
                                }
                            },
                        ),
                    )
                    .await
                };
                let wave_outcomes = match wave_result {
                    Ok(outcomes) => outcomes,
                    Err(error) => {
                        // Stage admission and provider adapters may have
                        // charged calls or tokens before returning an error.
                        // Budget spend is irrevocable even though frontier
                        // state is not.
                        drop(frontier_reservations);
                        state.budget = self.ledger.usage()?;
                        return Err(error);
                    }
                };
                state.budget = self.ledger.usage()?;
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
                if let Err(error) =
                    self.apply_outcome(&mut next_state, stage_id, definition, outcome.clone())
                {
                    state.budget = self.ledger.usage()?;
                    return Err(error);
                }
                batch_patches.extend(outcome.patches.clone());
                let repair_budget_remaining =
                    self.ledger.remaining(BudgetKind::RepairIterations)? > 0;
                let mut eligible: Vec<_> = self
                    .graph
                    .edges
                    .iter()
                    .filter(|edge| {
                        &edge.from == stage_id
                            && condition_matches(
                                edge.condition,
                                definition.kind,
                                outcome.passed,
                                repair_budget_remaining,
                            )
                    })
                    .collect();
                eligible.sort_by_key(|edge| edge.priority);
                ensure!(
                    !eligible.is_empty(),
                    "stage {} has no eligible transition",
                    stage_id.0
                );
                ensure!(
                    definition.kind != StageKind::ParallelFanOut
                        || eligible
                            .iter()
                            .all(|edge| { edge.condition != TransitionCondition::BudgetRemaining }),
                    "budget-remaining transitions cannot originate at parallel fan-out stages"
                );
                let selected = if definition.kind == StageKind::ParallelFanOut {
                    ensure!(
                        eligible.len() <= usize::from(self.graph.limits.maximum_parallelism),
                        "parallelism ceiling exceeded"
                    );
                    eligible
                } else {
                    eligible.into_iter().take(1).collect()
                };
                let repair_transitions = selected
                    .iter()
                    .filter(|edge| edge.condition == TransitionCondition::BudgetRemaining)
                    .count()
                    .try_into()
                    .map_err(|_| anyhow!("repair transition count overflow"))?;
                if let Err(error) = self
                    .ledger
                    .consume(BudgetKind::RepairIterations, repair_transitions)
                {
                    state.budget = self.ledger.usage()?;
                    return Err(error);
                }
                // A later sibling can still fail transition validation. Keep
                // each successful repair charge visible immediately rather
                // than waiting for the whole frontier or a checkpoint await.
                let usage = self.ledger.usage()?;
                state.budget = usage.clone();
                next_state.budget = usage;
                // A shared successor joins: the set deduplicates, and the
                // joined stage runs once, only after this whole frontier.
                next_frontier.extend(selected.into_iter().map(|edge| edge.to.clone()));
                if checkpoint_stage.is_none() && requires_checkpoint(stage_id, definition, &outcome)
                {
                    checkpoint_stage = Some((stage_id.clone(), definition.clone(), outcome));
                }
            }
            // Transition admission may have charged the repair budget after
            // individual stage outcomes were applied. Refresh the durable
            // snapshot before sealing a checkpoint.
            next_state.budget = self.ledger.usage()?;
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
                let checkpoint_result = {
                    // Repair transitions have already charged the ledger. If
                    // checkpoint persistence is cancelled, keep that
                    // irrevocable admission visible to the caller even though
                    // the frontier state itself is not committed.
                    let _budget_sync = BudgetStateSync::new(&mut state.budget, self.ledger.clone());
                    self.driver
                        .checkpoint(
                            &checkpoint,
                            checkpoint_reason(&stage_id, &definition, &outcome),
                        )
                        .await
                };
                if let Err(error) = checkpoint_result {
                    state.budget = self.ledger.usage()?;
                    return Err(error);
                }
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

fn condition_matches(
    condition: TransitionCondition,
    stage_kind: StageKind,
    passed: bool,
    repair_budget_remaining: bool,
) -> bool {
    match condition {
        TransitionCondition::Always => true,
        TransitionCondition::BudgetRemaining => repair_budget_remaining,
        TransitionCondition::StagePassed => passed,
        TransitionCondition::StageFailed => !passed,
        TransitionCondition::HumanApproved => stage_kind == StageKind::HumanApprovalGate && passed,
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
            model_calls: 10,
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
    async fn tool_stages_consume_the_authoritative_tool_call_budget() {
        let graph = tool_chain_graph();
        let mut limits = budgets();
        limits.tool_calls = 1;
        let ledger = SharedBudgetLedger::new(&limits);
        let driver = std::sync::Arc::new(Driver::default());
        let mut run_state = state();

        let error = WorkflowExecutor::new(graph, driver.clone(), ledger.clone())
            .execute(&mut run_state)
            .await
            .expect_err("the second tool stage must exceed the one-call budget");

        assert!(error.to_string().contains("ToolCalls budget exhausted"));
        assert_eq!(driver.invoked.lock().unwrap().as_slice(), ["tool_a"]);
        assert_eq!(
            ledger.usage().unwrap().consumed.get(&BudgetKind::ToolCalls),
            Some(&1)
        );
        assert_eq!(
            run_state.budget.consumed.get(&BudgetKind::ToolCalls),
            Some(&1)
        );
    }

    fn tool_chain_graph() -> WorkflowGraph {
        use StageKind::*;
        let stages = [
            ("tool_a", ToolInvocation, Some("a")),
            ("tool_b", ToolInvocation, Some("b")),
            ("verify", VerificationGate, None),
            ("complete", TerminalSuccess, None),
        ]
        .into_iter()
        .map(|(name, kind, tool)| {
            (
                StageId::from(name),
                StageDefinition {
                    kind,
                    model_role: None,
                    tool: tool.map(str::to_owned),
                    iteration_limit: None,
                    trusted: kind == VerificationGate,
                },
            )
        })
        .collect();
        let edges = [
            ("tool_a", "tool_b"),
            ("tool_b", "verify"),
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
            entry: StageId::from("tool_a"),
            stages,
            edges,
            limits: hi_rsi_runtime::WorkflowLimits {
                maximum_transitions: 10,
                maximum_parallelism: 1,
                maximum_concurrency: None,
            },
        }
    }

    #[tokio::test]
    async fn resume_restores_checkpointed_tool_budget_before_admission() {
        let mut limits = budgets();
        limits.tool_calls = 1;
        let ledger = SharedBudgetLedger::new(&limits);
        let driver = std::sync::Arc::new(Driver::default());
        let mut checkpoint_state = state();
        checkpoint_state
            .budget
            .consumed
            .insert(BudgetKind::ToolCalls, 1);
        let checkpoint = Checkpoint {
            schema_version: 1,
            run_id: checkpoint_state.run_id.clone(),
            candidate_id: checkpoint_state.candidate_id.clone(),
            state: checkpoint_state,
            workspace_tree_hash: "b".repeat(64),
            workflow_position: BTreeSet::from([StageId::from("tool_b")]),
            context_manifests: vec![],
            response_artifacts: vec![],
            created_at_sequence: 7,
        };
        let mut resumed_state = state();

        let error = WorkflowExecutor::new(tool_chain_graph(), driver.clone(), ledger.clone())
            .resume(&checkpoint, &mut resumed_state)
            .await
            .expect_err("resume must not grant a fresh tool-call budget");

        assert!(error.to_string().contains("ToolCalls budget exhausted"));
        assert!(driver.invoked.lock().unwrap().is_empty());
        assert_eq!(
            ledger.usage().unwrap().consumed.get(&BudgetKind::ToolCalls),
            Some(&1)
        );
        assert_eq!(
            resumed_state.budget.consumed.get(&BudgetKind::ToolCalls),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn resume_does_not_reset_the_workflow_transition_limit() {
        use StageKind::*;
        let stages = [
            ("repair", ModelInvocation, Some(10)),
            ("verify", VerificationGate, None),
            ("complete", TerminalSuccess, None),
        ]
        .into_iter()
        .map(|(name, kind, iteration_limit)| {
            (
                StageId::from(name),
                StageDefinition {
                    kind,
                    model_role: (kind == ModelInvocation).then(|| "repairer".into()),
                    tool: None,
                    iteration_limit,
                    trusted: kind == VerificationGate,
                },
            )
        })
        .collect();
        let graph = WorkflowGraph {
            entry: StageId::from("repair"),
            stages,
            edges: vec![
                hi_rsi_runtime::TransitionRule {
                    from: StageId::from("repair"),
                    to: StageId::from("repair"),
                    condition: TransitionCondition::BudgetRemaining,
                    priority: 0,
                },
                hi_rsi_runtime::TransitionRule {
                    from: StageId::from("repair"),
                    to: StageId::from("verify"),
                    condition: TransitionCondition::Always,
                    priority: 1,
                },
                hi_rsi_runtime::TransitionRule {
                    from: StageId::from("verify"),
                    to: StageId::from("complete"),
                    condition: TransitionCondition::StagePassed,
                    priority: 0,
                },
            ],
            limits: hi_rsi_runtime::WorkflowLimits {
                maximum_transitions: 2,
                maximum_parallelism: 1,
                maximum_concurrency: None,
            },
        };
        let mut limits = budgets();
        limits.repair_iterations = 10;
        let first = std::sync::Arc::new(Driver::default());
        let mut first_state = state();

        let first_error = WorkflowExecutor::new(
            graph.clone(),
            first.clone(),
            SharedBudgetLedger::new(&limits),
        )
        .execute(&mut first_state)
        .await
        .expect_err("the first process must stop at two transitions");
        assert!(
            first_error
                .to_string()
                .contains("transition budget exhausted")
        );
        let checkpoint = first
            .checkpoints
            .lock()
            .unwrap()
            .last()
            .cloned()
            .expect("repair stages seal resumable checkpoints");
        assert_eq!(checkpoint.state.attempts[&StageId::from("repair")], 2);

        let resumed = std::sync::Arc::new(Driver::default());
        let mut resumed_state = state();
        let resumed_error =
            WorkflowExecutor::new(graph, resumed.clone(), SharedBudgetLedger::new(&limits))
                .resume(&checkpoint, &mut resumed_state)
                .await
                .expect_err("resume must retain the already-spent transition count");

        assert!(
            resumed_error
                .to_string()
                .contains("transition budget exhausted")
        );
        assert!(resumed.invoked.lock().unwrap().is_empty());
    }

    fn two_wave_tool_graph() -> WorkflowGraph {
        use StageKind::*;
        let stages = [
            ("scatter", ParallelFanOut, None),
            ("tool_a", ToolInvocation, Some("a")),
            ("tool_b", ToolInvocation, Some("b")),
            ("failed", TerminalFailure, None),
        ]
        .into_iter()
        .map(|(name, kind, tool)| {
            (
                StageId::from(name),
                StageDefinition {
                    kind,
                    model_role: None,
                    tool: tool.map(str::to_owned),
                    iteration_limit: None,
                    trusted: false,
                },
            )
        })
        .collect();
        let edge = |from, to, priority| hi_rsi_runtime::TransitionRule {
            from: StageId::from(from),
            to: StageId::from(to),
            condition: TransitionCondition::StagePassed,
            priority,
        };
        WorkflowGraph {
            entry: StageId::from("scatter"),
            stages,
            edges: vec![
                edge("scatter", "tool_a", 0),
                edge("scatter", "tool_b", 1),
                edge("tool_a", "failed", 0),
                edge("tool_b", "failed", 0),
            ],
            limits: hi_rsi_runtime::WorkflowLimits {
                maximum_transitions: 10,
                maximum_parallelism: 2,
                maximum_concurrency: Some(1),
            },
        }
    }

    fn same_wave_tool_graph() -> WorkflowGraph {
        let mut graph = two_wave_tool_graph();
        graph.limits.maximum_concurrency = Some(2);
        graph
    }

    fn parallel_model_graph() -> WorkflowGraph {
        let mut graph = same_wave_tool_graph();
        for name in ["tool_a", "tool_b"] {
            let stage = graph.stages.get_mut(&StageId::from(name)).unwrap();
            stage.kind = StageKind::ModelInvocation;
            stage.model_role = Some("reviewer".into());
            stage.tool = None;
        }
        graph
    }

    #[tokio::test]
    async fn insufficient_budget_rejects_a_complete_tool_frontier_before_execution() {
        let mut limits = budgets();
        limits.tool_calls = 1;
        let ledger = SharedBudgetLedger::new(&limits);
        let driver = std::sync::Arc::new(Driver::default());
        let mut run_state = state();

        let error = WorkflowExecutor::new(two_wave_tool_graph(), driver.clone(), ledger.clone())
            .execute(&mut run_state)
            .await
            .expect_err("both waves must be admitted before either tool executes");

        assert!(error.to_string().contains("ToolCalls budget exhausted"));
        assert_eq!(driver.invoked.lock().unwrap().as_slice(), ["scatter"]);
        assert_eq!(
            ledger
                .usage()
                .unwrap()
                .consumed
                .get(&BudgetKind::ToolCalls)
                .copied()
                .unwrap_or(0),
            0
        );
        assert_eq!(ledger.remaining(BudgetKind::ToolCalls).unwrap(), 1);
    }

    #[tokio::test]
    async fn insufficient_budget_rejects_a_complete_model_frontier_before_execution() {
        let mut limits = budgets();
        limits.model_calls = 1;
        let ledger = SharedBudgetLedger::new(&limits);
        let driver = std::sync::Arc::new(Driver::default());
        let mut run_state = state();

        let error = WorkflowExecutor::new(parallel_model_graph(), driver.clone(), ledger.clone())
            .execute(&mut run_state)
            .await
            .expect_err("both model requests must be admitted before either starts");

        assert!(error.to_string().contains("ModelCalls budget exhausted"));
        assert_eq!(driver.invoked.lock().unwrap().as_slice(), ["scatter"]);
        assert_eq!(
            ledger
                .usage()
                .unwrap()
                .consumed
                .get(&BudgetKind::ModelCalls)
                .copied()
                .unwrap_or(0),
            0
        );
        assert_eq!(ledger.remaining(BudgetKind::ModelCalls).unwrap(), 1);
        assert!(
            ledger
                .usage()
                .unwrap()
                .reserved
                .values()
                .all(|amount| *amount == 0)
        );
    }

    #[tokio::test]
    async fn cancelled_tool_frontier_releases_unstarted_reservations_and_keeps_spend() {
        struct HangingFirstTool {
            started: std::sync::Arc<tokio::sync::Notify>,
        }
        #[async_trait]
        impl TrustedStageDriver for HangingFirstTool {
            async fn stage(
                &self,
                _: &StageDefinition,
                stage: &StageId,
                _: u32,
                _: &RunState,
            ) -> Result<StageOutcome> {
                if stage.0 == "tool_a" {
                    self.started.notify_one();
                    std::future::pending().await
                }
                Ok(StageOutcome {
                    passed: true,
                    ..Default::default()
                })
            }

            async fn checkpoint(&self, _: &Checkpoint, _: &str) -> Result<ArtifactRef> {
                unreachable!("the hanging frontier never reaches a checkpoint")
            }
        }

        let mut limits = budgets();
        limits.tool_calls = 2;
        let ledger = SharedBudgetLedger::new(&limits);
        let started = std::sync::Arc::new(tokio::sync::Notify::new());
        let mut run_state = state();
        {
            let run = WorkflowExecutor::new(
                two_wave_tool_graph(),
                HangingFirstTool {
                    started: started.clone(),
                },
                ledger.clone(),
            )
            .execute(&mut run_state);
            tokio::pin!(run);
            tokio::select! {
                _ = started.notified() => {}
                result = &mut run => panic!("workflow unexpectedly settled: {result:?}"),
                _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {
                    panic!("first tool wave never started")
                }
            }
        }

        assert_eq!(
            run_state.budget.consumed.get(&BudgetKind::ToolCalls),
            Some(&1)
        );
        assert!(
            run_state
                .budget
                .reserved
                .values()
                .all(|amount| *amount == 0)
        );
        assert_eq!(ledger.remaining(BudgetKind::ToolCalls).unwrap(), 1);
        assert!(
            ledger
                .usage()
                .unwrap()
                .reserved
                .values()
                .all(|amount| *amount == 0)
        );
    }

    #[tokio::test]
    async fn failed_early_wave_does_not_charge_later_uninvoked_tools() {
        struct FailFirstTool {
            invoked: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        }
        #[async_trait]
        impl TrustedStageDriver for FailFirstTool {
            async fn stage(
                &self,
                _: &StageDefinition,
                stage: &StageId,
                _: u32,
                _: &RunState,
            ) -> Result<StageOutcome> {
                self.invoked.lock().unwrap().push(stage.0.clone());
                if stage.0 == "tool_a" {
                    bail!("first tool failed before the next wave");
                }
                Ok(StageOutcome {
                    passed: true,
                    ..Default::default()
                })
            }

            async fn checkpoint(&self, _: &Checkpoint, _: &str) -> Result<ArtifactRef> {
                unreachable!("the failing frontier never reaches a checkpoint")
            }
        }

        let ledger = SharedBudgetLedger::new(&budgets());
        let invoked = std::sync::Arc::new(std::sync::Mutex::new(vec![]));
        let driver = FailFirstTool {
            invoked: invoked.clone(),
        };
        let mut run_state = state();

        let error = WorkflowExecutor::new(two_wave_tool_graph(), driver, ledger.clone())
            .execute(&mut run_state)
            .await
            .expect_err("the first tool wave is scripted to fail");

        assert!(error.to_string().contains("first tool failed"));
        assert_eq!(invoked.lock().unwrap().as_slice(), ["scatter", "tool_a"]);
        assert_eq!(
            ledger.usage().unwrap().consumed.get(&BudgetKind::ToolCalls),
            Some(&1),
            "tool_b never ran and must not be charged"
        );
        assert_eq!(
            run_state.budget.consumed.get(&BudgetKind::ToolCalls),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn immediate_same_wave_error_does_not_charge_an_unpolled_tool() {
        struct ImmediateFirstError {
            invoked: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        }
        #[async_trait]
        impl TrustedStageDriver for ImmediateFirstError {
            async fn stage(
                &self,
                _: &StageDefinition,
                stage: &StageId,
                _: u32,
                _: &RunState,
            ) -> Result<StageOutcome> {
                self.invoked.lock().unwrap().push(stage.0.clone());
                if stage.0 == "tool_a" {
                    bail!("first same-wave tool failed immediately");
                }
                Ok(StageOutcome {
                    passed: true,
                    ..Default::default()
                })
            }

            async fn checkpoint(&self, _: &Checkpoint, _: &str) -> Result<ArtifactRef> {
                unreachable!("the failing frontier never reaches a checkpoint")
            }
        }

        let ledger = SharedBudgetLedger::new(&budgets());
        let invoked = std::sync::Arc::new(std::sync::Mutex::new(vec![]));
        let mut run_state = state();

        let error = WorkflowExecutor::new(
            same_wave_tool_graph(),
            ImmediateFirstError {
                invoked: invoked.clone(),
            },
            ledger.clone(),
        )
        .execute(&mut run_state)
        .await
        .expect_err("the first same-wave tool is scripted to fail");

        assert!(error.to_string().contains("failed immediately"));
        assert_eq!(invoked.lock().unwrap().as_slice(), ["scatter", "tool_a"]);
        assert_eq!(
            ledger.usage().unwrap().consumed.get(&BudgetKind::ToolCalls),
            Some(&1),
            "the unpolled tool_b future must release rather than commit"
        );
        assert_eq!(ledger.remaining(BudgetKind::ToolCalls).unwrap(), 9);
        assert!(
            ledger
                .usage()
                .unwrap()
                .reserved
                .values()
                .all(|amount| *amount == 0)
        );
        assert_eq!(
            run_state.budget.consumed.get(&BudgetKind::ToolCalls),
            Some(&1)
        );
    }

    fn single_model_graph() -> WorkflowGraph {
        let stages = BTreeMap::from([
            (
                StageId::from("model"),
                StageDefinition {
                    kind: StageKind::ModelInvocation,
                    model_role: Some("reviewer".into()),
                    tool: None,
                    iteration_limit: None,
                    trusted: false,
                },
            ),
            (
                StageId::from("failed"),
                StageDefinition {
                    kind: StageKind::TerminalFailure,
                    model_role: None,
                    tool: None,
                    iteration_limit: None,
                    trusted: false,
                },
            ),
        ]);
        WorkflowGraph {
            entry: StageId::from("model"),
            stages,
            edges: vec![hi_rsi_runtime::TransitionRule {
                from: StageId::from("model"),
                to: StageId::from("failed"),
                condition: TransitionCondition::StageFailed,
                priority: 0,
            }],
            limits: hi_rsi_runtime::WorkflowLimits {
                maximum_transitions: 2,
                maximum_parallelism: 1,
                maximum_concurrency: None,
            },
        }
    }

    #[tokio::test]
    async fn driver_error_keeps_irrevocable_budget_spend_in_caller_state() {
        struct ChargingFailure;
        #[async_trait]
        impl TrustedStageDriver for ChargingFailure {
            async fn stage(
                &self,
                _: &StageDefinition,
                _: &StageId,
                _: u32,
                _: &RunState,
            ) -> Result<StageOutcome> {
                bail!("provider failed after accepting the request")
            }

            async fn checkpoint(&self, _: &Checkpoint, _: &str) -> Result<ArtifactRef> {
                unreachable!("the stage always fails")
            }
        }

        let ledger = SharedBudgetLedger::new(&budgets());
        let mut run_state = state();

        let error = WorkflowExecutor::new(single_model_graph(), ChargingFailure, ledger)
            .execute(&mut run_state)
            .await
            .expect_err("the driver is scripted to fail");

        assert!(error.to_string().contains("provider failed"));
        assert_eq!(
            run_state.budget.consumed.get(&BudgetKind::ModelCalls),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn cancelled_model_stage_keeps_admitted_call_without_unknown_tokens() {
        struct HangingModel {
            started: std::sync::Arc<tokio::sync::Notify>,
        }
        #[async_trait]
        impl TrustedStageDriver for HangingModel {
            async fn stage(
                &self,
                _: &StageDefinition,
                _: &StageId,
                _: u32,
                _: &RunState,
            ) -> Result<StageOutcome> {
                self.started.notify_one();
                std::future::pending().await
            }

            async fn checkpoint(&self, _: &Checkpoint, _: &str) -> Result<ArtifactRef> {
                unreachable!("the hanging model never reaches a checkpoint")
            }
        }

        let mut limits = budgets();
        limits.model_calls = 1;
        let ledger = SharedBudgetLedger::new(&limits);
        let started = std::sync::Arc::new(tokio::sync::Notify::new());
        let mut run_state = state();
        {
            let run = WorkflowExecutor::new(
                single_model_graph(),
                HangingModel {
                    started: started.clone(),
                },
                ledger.clone(),
            )
            .execute(&mut run_state);
            tokio::pin!(run);
            tokio::select! {
                _ = started.notified() => {}
                result = &mut run => panic!("workflow unexpectedly settled: {result:?}"),
                _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {
                    panic!("model stage never started")
                }
            }
        }

        let usage = ledger.usage().unwrap();
        assert_eq!(
            run_state.budget.consumed.get(&BudgetKind::ModelCalls),
            Some(&1)
        );
        assert_eq!(usage.consumed.get(&BudgetKind::ModelCalls), Some(&1));
        assert_eq!(
            run_state
                .budget
                .consumed
                .get(&BudgetKind::InputTokens)
                .copied()
                .unwrap_or(0),
            0
        );
        assert_eq!(
            run_state
                .budget
                .consumed
                .get(&BudgetKind::OutputTokens)
                .copied()
                .unwrap_or(0),
            0
        );
        assert!(run_state.budget.reserved.is_empty());
        assert!(usage.reserved.values().all(|amount| *amount == 0));

        let retry_driver = std::sync::Arc::new(Driver::default());
        let retry_error =
            WorkflowExecutor::new(single_model_graph(), retry_driver.clone(), ledger.clone())
                .execute(&mut run_state)
                .await
                .expect_err("an in-process retry must retain the cancelled model call");
        assert!(
            retry_error
                .to_string()
                .contains("ModelCalls budget exhausted")
        );
        assert!(retry_driver.invoked.lock().unwrap().is_empty());

        let checkpoint_state = state();
        let older_checkpoint = Checkpoint {
            schema_version: 1,
            run_id: checkpoint_state.run_id.clone(),
            candidate_id: checkpoint_state.candidate_id.clone(),
            state: checkpoint_state,
            workspace_tree_hash: "b".repeat(64),
            workflow_position: BTreeSet::from([StageId::from("model")]),
            context_manifests: vec![],
            response_artifacts: vec![],
            created_at_sequence: 1,
        };
        let resume_driver = std::sync::Arc::new(Driver::default());
        let resume_error = WorkflowExecutor::new(
            single_model_graph(),
            resume_driver.clone(),
            SharedBudgetLedger::new(&limits),
        )
        .resume(&older_checkpoint, &mut run_state)
        .await
        .expect_err("an older checkpoint must not erase post-checkpoint cancelled spend");
        assert!(
            resume_error
                .to_string()
                .contains("ModelCalls budget exhausted")
        );
        assert!(resume_driver.invoked.lock().unwrap().is_empty());
        assert_eq!(
            run_state.budget.consumed.get(&BudgetKind::ModelCalls),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn cancelled_checkpoint_keeps_repair_and_model_spend_in_caller_state() {
        struct HangingCheckpoint {
            started: std::sync::Arc<tokio::sync::Notify>,
        }
        #[async_trait]
        impl TrustedStageDriver for HangingCheckpoint {
            async fn stage(
                &self,
                _: &StageDefinition,
                _: &StageId,
                _: u32,
                _: &RunState,
            ) -> Result<StageOutcome> {
                Ok(StageOutcome::default())
            }

            async fn checkpoint(&self, _: &Checkpoint, _: &str) -> Result<ArtifactRef> {
                self.started.notify_one();
                std::future::pending().await
            }
        }

        let graph = WorkflowGraph {
            entry: StageId::from("repair"),
            stages: BTreeMap::from([
                (
                    StageId::from("repair"),
                    StageDefinition {
                        kind: StageKind::ModelInvocation,
                        model_role: Some("repairer".into()),
                        tool: None,
                        iteration_limit: Some(1),
                        trusted: false,
                    },
                ),
                (
                    StageId::from("failed"),
                    StageDefinition {
                        kind: StageKind::TerminalFailure,
                        model_role: None,
                        tool: None,
                        iteration_limit: None,
                        trusted: false,
                    },
                ),
            ]),
            edges: vec![hi_rsi_runtime::TransitionRule {
                from: StageId::from("repair"),
                to: StageId::from("failed"),
                condition: TransitionCondition::BudgetRemaining,
                priority: 0,
            }],
            limits: hi_rsi_runtime::WorkflowLimits {
                maximum_transitions: 2,
                maximum_parallelism: 1,
                maximum_concurrency: None,
            },
        };
        let ledger = SharedBudgetLedger::new(&budgets());
        let started = std::sync::Arc::new(tokio::sync::Notify::new());
        let mut run_state = state();
        {
            let run = WorkflowExecutor::new(
                graph,
                HangingCheckpoint {
                    started: started.clone(),
                },
                ledger.clone(),
            )
            .execute(&mut run_state);
            tokio::pin!(run);
            tokio::select! {
                _ = started.notified() => {}
                result = &mut run => panic!("workflow unexpectedly settled: {result:?}"),
                _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {
                    panic!("checkpoint never started")
                }
            }
        }

        let usage = ledger.usage().unwrap();
        for kind in [BudgetKind::ModelCalls, BudgetKind::RepairIterations] {
            assert_eq!(run_state.budget.consumed.get(&kind), Some(&1));
            assert_eq!(usage.consumed.get(&kind), Some(&1));
        }
        assert!(run_state.budget.reserved.is_empty());
        assert!(usage.reserved.values().all(|amount| *amount == 0));
    }

    fn budget_routed_graph() -> WorkflowGraph {
        use StageKind::*;
        let stages = [
            ("seed", DeterministicTransform),
            ("within_budget", DeterministicTransform),
            ("budget_exhausted", DeterministicTransform),
            ("verify", VerificationGate),
            ("complete", TerminalSuccess),
        ]
        .into_iter()
        .map(|(name, kind)| {
            (
                StageId::from(name),
                StageDefinition {
                    kind,
                    model_role: None,
                    tool: None,
                    iteration_limit: None,
                    trusted: kind == VerificationGate,
                },
            )
        })
        .collect();
        let edge = |from, to, condition, priority| hi_rsi_runtime::TransitionRule {
            from: StageId::from(from),
            to: StageId::from(to),
            condition,
            priority,
        };
        WorkflowGraph {
            entry: StageId::from("seed"),
            stages,
            edges: vec![
                edge(
                    "seed",
                    "within_budget",
                    TransitionCondition::BudgetRemaining,
                    0,
                ),
                edge("seed", "budget_exhausted", TransitionCondition::Always, 1),
                edge(
                    "within_budget",
                    "verify",
                    TransitionCondition::StagePassed,
                    0,
                ),
                edge(
                    "budget_exhausted",
                    "verify",
                    TransitionCondition::StagePassed,
                    0,
                ),
                edge("verify", "complete", TransitionCondition::StagePassed, 0),
            ],
            limits: hi_rsi_runtime::WorkflowLimits {
                maximum_transitions: 10,
                maximum_parallelism: 1,
                maximum_concurrency: None,
            },
        }
    }

    #[tokio::test]
    async fn budget_remaining_routes_and_charges_one_repair_iteration() {
        for (repair_iterations, expected_stage) in [(1, "within_budget"), (0, "budget_exhausted")] {
            let mut limits = budgets();
            limits.repair_iterations = repair_iterations;
            let ledger = SharedBudgetLedger::new(&limits);
            let driver = std::sync::Arc::new(Driver::default());
            let mut run_state = state();

            let outcome =
                WorkflowExecutor::new(budget_routed_graph(), driver.clone(), ledger.clone())
                    .execute(&mut run_state)
                    .await
                    .unwrap();

            assert_eq!(outcome, TerminalOutcome::Succeeded);
            let invoked = driver.invoked.lock().unwrap().clone();
            assert!(invoked.contains(&expected_stage.to_string()), "{invoked:?}");
            assert_eq!(
                ledger
                    .usage()
                    .unwrap()
                    .consumed
                    .get(&BudgetKind::RepairIterations)
                    .copied()
                    .unwrap_or(0),
                u64::from(repair_iterations)
            );
        }
    }

    #[tokio::test]
    async fn later_sibling_error_keeps_an_earlier_repair_charge_in_caller_state() {
        use StageKind::*;
        let stages = [
            ("scatter", ParallelFanOut),
            ("branch_a", DeterministicTransform),
            ("branch_b", DeterministicTransform),
            ("failed", TerminalFailure),
        ]
        .into_iter()
        .map(|(name, kind)| {
            (
                StageId::from(name),
                StageDefinition {
                    kind,
                    model_role: None,
                    tool: None,
                    iteration_limit: None,
                    trusted: false,
                },
            )
        })
        .collect();
        let edge = |from, to, condition, priority| hi_rsi_runtime::TransitionRule {
            from: StageId::from(from),
            to: StageId::from(to),
            condition,
            priority,
        };
        let graph = WorkflowGraph {
            entry: StageId::from("scatter"),
            stages,
            edges: vec![
                edge("scatter", "branch_a", TransitionCondition::StagePassed, 0),
                edge("scatter", "branch_b", TransitionCondition::StagePassed, 1),
                edge(
                    "branch_a",
                    "failed",
                    TransitionCondition::BudgetRemaining,
                    0,
                ),
            ],
            limits: hi_rsi_runtime::WorkflowLimits {
                maximum_transitions: 10,
                maximum_parallelism: 2,
                maximum_concurrency: Some(2),
            },
        };
        let ledger = SharedBudgetLedger::new(&budgets());
        let mut run_state = state();

        let error = WorkflowExecutor::new(graph, Driver::default(), ledger.clone())
            .execute(&mut run_state)
            .await
            .expect_err("branch_b deliberately has no eligible transition");

        assert!(
            error
                .to_string()
                .contains("branch_b has no eligible transition")
        );
        assert_eq!(
            ledger
                .usage()
                .unwrap()
                .consumed
                .get(&BudgetKind::RepairIterations),
            Some(&1)
        );
        assert_eq!(
            run_state.budget.consumed.get(&BudgetKind::RepairIterations),
            Some(&1)
        );
    }

    #[test]
    fn human_approval_condition_rejects_an_ordinary_passing_stage() {
        assert!(!condition_matches(
            TransitionCondition::HumanApproved,
            StageKind::PolicyGate,
            true,
            true,
        ));
        assert!(condition_matches(
            TransitionCondition::HumanApproved,
            StageKind::HumanApprovalGate,
            true,
            true,
        ));
        assert!(!condition_matches(
            TransitionCondition::HumanApproved,
            StageKind::HumanApprovalGate,
            false,
            true,
        ));
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
            err.to_string()
                .contains("does not match the run being resumed"),
            "unexpected error: {err}"
        );
        // State is untouched: the foreign run identity was not overwritten.
        assert_eq!(foreign.run_id, "someone-elses-run");
        assert!(foreign.plan.is_none());
    }
}
