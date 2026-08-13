//! `hi workflow run plan.md` — local, self-hosted workflow execution.
//!
//! ADR-001 carve-out: this entry drives `hi_agent_runtime::WorkflowExecutor`
//! locally, WITHOUT the managed attestation chain. Verification reports are
//! signed with the shared local ed25519 key (`local-signed:`) — tamper-evident
//! but not worker-attested, so they can never be mistaken for managed RSI
//! evidence. Each plan objective executes through the existing delegate
//! machinery — an isolated worktree child that must verify before its diff is
//! applied — so "passed" always means applied AND verified, never narrated.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use hi_agent::{
    DelegateRunner, actionability_issues, parse_objectives, plan_has_checked_objectives,
};
use hi_agent_runtime::{
    GateAuthority, StageDriver, StageModel, StageOutcome, TerminalOutcome, WorkflowExecutor,
    latest_checkpoint,
};
use hi_events::{
    ActivityObject, ActivityState, ActivityVerb, EventContext, EventKind, RunEvent,
    SemanticActivity,
};
use hi_rsi_runtime::{
    ArtifactRef, FailureDomain, FailureEvidence, RunState, RuntimeBudgets, SharedBudgetLedger,
    StageDefinition, StageId, StageKind, TransitionCondition, TransitionRule, VerificationReport,
    WorkflowGraph, WorkflowLimits,
};
use hi_verifier::{AttestingVerifier, Attestor, CheckSpec};

/// Objectives above this need a split plan — a single run of thousands of
/// delegate children is not a supervisable unit of work.
const MAX_OBJECTIVES: usize = hi_agent::MAX_PLAN_OBJECTIVES;
/// Default concurrent objective delegates per wave; the cross-process
/// resource governor additionally caps live children machine-wide.
const DEFAULT_WAVE_CONCURRENCY: u16 = 4;

/// Detach `hi workflow run <plan>` so the REPL/TUI stays interactive.
pub(crate) fn spawn_detached_workflow_run(exe: &Path, plan: &str) -> Result<(u32, PathBuf)> {
    let log = std::env::temp_dir().join(format!(
        "hi-workflow-plan-{}-{}.log",
        std::process::id(),
        plan.replace(['/', '.'], "_")
    ));
    let log_file = std::fs::File::create(&log)
        .with_context(|| format!("cannot create workflow log {}", log.display()))?;
    let stderr_file = log_file
        .try_clone()
        .context("cannot clone workflow log handle")?;
    let child = std::process::Command::new(exe)
        .args(["workflow", "run", plan])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(stderr_file))
        .spawn()
        .context("failed to start workflow child")?;
    let pid = child.id();
    drop(child);
    Ok((pid, log))
}

pub(crate) async fn run_workflow_cli(args: &[String]) -> Result<()> {
    let mut action = None;
    let mut plan_path = None;
    let mut verify_override = None;
    let mut parallel = DEFAULT_WAVE_CONCURRENCY;
    let mut retries = 0_u32;
    let mut bestof = 0_u32;
    let mut dry_run = false;
    let mut resume = false;
    let mut check_off = false;
    let mut iter = args.iter();
    while let Some(argument) = iter.next() {
        match argument.as_str() {
            "run" if action.is_none() => action = Some("run"),
            "resume" if action.is_none() => {
                action = Some("run");
                resume = true;
            }
            "verify" if action.is_none() => action = Some("verify"),
            "--verify" => {
                verify_override = Some(
                    iter.next()
                        .ok_or_else(|| anyhow!("--verify requires a command"))?
                        .clone(),
                );
            }
            "--parallel" => {
                parallel = iter
                    .next()
                    .ok_or_else(|| anyhow!("--parallel requires a number"))?
                    .parse::<u16>()
                    .context("--parallel requires a number between 1 and 16")?
                    .clamp(1, 16);
            }
            "--retries" => {
                retries = iter
                    .next()
                    .ok_or_else(|| anyhow!("--retries requires a number"))?
                    .parse::<u32>()
                    .context("--retries requires a number between 0 and 3")?
                    .min(3);
            }
            "--bestof" => {
                bestof = iter
                    .next()
                    .ok_or_else(|| anyhow!("--bestof requires a candidate count"))?
                    .parse::<u32>()
                    .context("--bestof requires a number between 2 and 4")?
                    .clamp(2, 4);
            }
            "--resume" => resume = true,
            "--dry-run" => dry_run = true,
            "--check-off" => check_off = true,
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other if plan_path.is_none() && action.is_some() => {
                plan_path = Some(PathBuf::from(other));
            }
            other => bail!("unexpected workflow argument {other:?} (see `hi workflow --help`)"),
        }
    }
    if action.is_none() && plan_path.is_none() && args.is_empty() {
        print_usage();
        return Ok(());
    }
    if action == Some("verify") {
        // With an explicit path, verify that file. Without one, resolve the
        // most recently written report under the state root (the path
        // `run`/`resume` persists to), mirroring `hi trace verify`.
        let path = match plan_path {
            Some(path) => path,
            None => latest_workflow_report()
                .ok_or_else(|| anyhow!("no workflow report found under the state root; run `hi workflow run` first, or pass a report path"))?,
        };
        return verify_report_cli(&path);
    }
    if action.is_none() || plan_path.is_none() {
        print_usage();
        bail!("usage: hi workflow run <plan.md>");
    }
    run(
        &plan_path.expect("checked above"),
        verify_override,
        parallel,
        retries,
        bestof,
        dry_run,
        resume,
        check_off,
    )
    .await
}

fn print_usage() {
    println!(
        "hi workflow — run a plan of objectives through the local workflow engine\n\n\
         USAGE:\n  hi workflow run <plan.md> [--verify CMD] [--parallel N] [--retries N] [--bestof N] [--check-off] [--dry-run]\n  \
         hi workflow resume <plan.md>\n  \
         hi workflow verify [report.json]\n\n\
         Objectives are unchecked markdown checkboxes (`- [ ] …`), else numbered\n\
         items, else bullets. Each objective runs as an isolated delegate child\n\
         that must pass verification before its diff is applied. A final trusted\n\
         verification gate runs the same pipeline across the whole workspace.\n\
         `--bestof N` (2-4): when an objective still fails after its retries,\n\
         run N diverse candidates in parallel worktrees and merge the one that\n\
         passes independent verification.\n\
         `--check-off` marks succeeded objectives `- [x]` in the plan after the\n\
         run, so a rerun only retries what failed.\n\
         Reports are signed `local-signed:` — this is the self-hosted mode,\n\
         not managed RSI evidence. Checkpoints live under the state root keyed\n\
         by plan content; `resume` continues the latest sealed checkpoint.\n\
         `verify` checks a report's local-signed attestation against the local\n\
         ed25519 key (latest persisted report, or an explicit report.json path),\n\
         failing on a forged or tampered signature. The signing key lives at\n\
         `$XDG_STATE_HOME/hi/trace-signing-key` (else `$HOME/.local/state/hi/`),\n\
         created owner-only on first signing run."
    );
}

fn objective_stage_id(index: usize) -> StageId {
    StageId(format!("objective_{index:04}"))
}

/// Build the plan graph:
/// intake → ingest_plan → scatter ⇉ objective_NNNN ⇉ join → objectives_gate
/// → verify → complete, with the gate and verify failing to a terminal
/// failure. Wave concurrency is bounded separately from fan-out width.
pub(crate) fn plan_graph(objective_count: usize, wave_concurrency: u16) -> Result<WorkflowGraph> {
    ensure!(objective_count >= 1, "the plan has no objectives");
    ensure!(
        objective_count <= MAX_OBJECTIVES,
        "the plan has {objective_count} objectives; split it below {MAX_OBJECTIVES}"
    );
    let stage = |kind: StageKind, role: Option<&str>, trusted: bool| StageDefinition {
        kind,
        model_role: role.map(str::to_owned),
        tool: None,
        iteration_limit: None,
        trusted,
    };
    let mut stages = BTreeMap::from([
        (
            StageId::from("intake"),
            stage(StageKind::DeterministicTransform, None, false),
        ),
        (
            StageId::from("ingest_plan"),
            stage(StageKind::ModelInvocation, Some("planner"), false),
        ),
        (
            StageId::from("scatter"),
            stage(StageKind::ParallelFanOut, None, false),
        ),
        (
            StageId::from("join"),
            stage(StageKind::Aggregation, None, false),
        ),
        (
            StageId::from("objectives_gate"),
            stage(StageKind::PolicyGate, None, false),
        ),
        (
            StageId::from("verify"),
            stage(StageKind::VerificationGate, None, true),
        ),
        (
            StageId::from("complete"),
            stage(StageKind::TerminalSuccess, None, false),
        ),
        (
            StageId::from("failed"),
            stage(StageKind::TerminalFailure, None, false),
        ),
    ]);
    // The typed plan must exist before any objective lands patches, so the
    // planner runs before the fan-out.
    let mut edges = vec![
        edge("intake", "ingest_plan", TransitionCondition::StagePassed, 0),
        edge(
            "ingest_plan",
            "scatter",
            TransitionCondition::StagePassed,
            0,
        ),
    ];
    for index in 1..=objective_count {
        let id = objective_stage_id(index);
        stages.insert(
            id.clone(),
            stage(StageKind::ModelInvocation, Some("implementer"), false),
        );
        edges.push(TransitionRule {
            from: StageId::from("scatter"),
            to: id.clone(),
            condition: TransitionCondition::StagePassed,
            priority: index as u16 - 1,
        });
        // Failed objectives still join; the objectives gate decides the run.
        edges.push(TransitionRule {
            from: id,
            to: StageId::from("join"),
            condition: TransitionCondition::Always,
            priority: 0,
        });
    }
    edges.extend([
        edge(
            "join",
            "objectives_gate",
            TransitionCondition::StagePassed,
            0,
        ),
        edge(
            "objectives_gate",
            "verify",
            TransitionCondition::StagePassed,
            0,
        ),
        edge(
            "objectives_gate",
            "failed",
            TransitionCondition::StageFailed,
            1,
        ),
        edge("verify", "complete", TransitionCondition::StagePassed, 0),
        edge("verify", "failed", TransitionCondition::StageFailed, 1),
    ]);
    let graph = WorkflowGraph {
        entry: StageId::from("intake"),
        stages,
        edges,
        limits: WorkflowLimits {
            maximum_transitions: (objective_count as u32 + 8) * 2,
            maximum_parallelism: objective_count.max(4) as u16,
            maximum_concurrency: Some(wave_concurrency.max(1)),
        },
    };
    graph.validate(
        &["planner".to_string(), "implementer".to_string()]
            .into_iter()
            .collect(),
        &BTreeSet::new(),
    )?;
    Ok(graph)
}

fn edge(from: &str, to: &str, condition: TransitionCondition, priority: u16) -> TransitionRule {
    TransitionRule {
        from: StageId::from(from),
        to: StageId::from(to),
        condition,
        priority,
    }
}

/// Self-hosted attestation: a real ed25519 signature over the report hash,
/// made with the shared local key (the same one `hi_trace::LocalAttestor`
/// uses). Emits `local-signed:<hex-signature>` over the hex report hash.
///
/// This is *not* worker attestation — the key lives on the same machine, so it
/// proves the report is unmodified since signing (tamper-evidence), not that a
/// trusted external worker anchored it. The `local-signed:` prefix keeps it
/// distinguishable from a worker scheme. Replaces the earlier
/// `local-unattested:` placeholder label.
pub(crate) struct LocalAttestor;

impl LocalAttestor {
    /// Sign a report hash with the shared local key, returning the
    /// `local-signed:` attestation string over the hash's hex form.
    fn sign(report_hash: &[u8; 32]) -> Result<String> {
        Self::sign_with_key(report_hash, &hi_trace::local_signing_key_path())
    }

    /// Key-path-injectable core so tests can point at a temp key without
    /// touching the shared default location or the environment.
    fn sign_with_key(report_hash: &[u8; 32], key_path: &Path) -> Result<String> {
        let key = hi_trace::load_or_create_signing_key(key_path)?;
        let hash_hex = blake3::Hash::from_bytes(*report_hash).to_hex();
        hi_trace::sign_root_hash(&key, hash_hex.as_str())
    }
}

impl Attestor for LocalAttestor {
    fn attest(&self, report_hash: &[u8; 32]) -> Result<String> {
        Self::sign(report_hash)
    }
}

/// Verify a workflow verification report's `local-signed:` attestation against
/// the local ed25519 key, mirroring `hi trace verify`. Recomputes the unsigned
/// report hash (the report with the attestation field cleared, exactly as
/// `AttestingVerifier` hashed it) and checks the signature. A signature
/// mismatch is a hard failure — the signature is the report's only integrity
/// mechanism, unlike a trace's hash chain.
fn verify_report(path: &Path) -> Result<Vec<String>> {
    verify_report_with_key(path, &hi_trace::local_signing_key_path())
}

/// The most recently written persisted workflow report
/// (`<state_root>/workflow/*/report.json`), newest first. This is the path
/// `run`/`resume` persists the final signed report to.
fn latest_workflow_report() -> Option<PathBuf> {
    let (_workspace, state_root) = crate::review_target::resolve_runtime_roots().ok()?;
    latest_workflow_report_under(&state_root)
}

/// Key-path-injectable core of [`latest_workflow_report`]: scan a state root
/// for the newest `workflow/*/report.json`.
fn latest_workflow_report_under(state_root: &Path) -> Option<PathBuf> {
    let workflow_root = state_root.join("workflow");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(workflow_root).ok()? {
        let entry = entry.ok()?;
        let report = entry.path().join("report.json");
        if !report.exists() {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if best.as_ref().is_none_or(|(t, _)| modified > *t) {
            best = Some((modified, report));
        }
    }
    best.map(|(_, path)| path)
}

/// Key-path-injectable core of [`verify_report`] so tests can point at a temp
/// key without touching the shared default location or the environment.
fn verify_report_with_key(path: &Path, key_path: &Path) -> Result<Vec<String>> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading report {}", path.display()))?;
    let report: VerificationReport = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing report {}", path.display()))?;
    let Some(attestation) = report.supervisor_attestation.clone() else {
        bail!("report {} has no attestation (unsigned)", path.display());
    };
    // Recompute the unsigned hash exactly as AttestingVerifier did: the report
    // with the attestation field cleared, serialized in field order.
    let mut unsigned = report.clone();
    unsigned.supervisor_attestation = None;
    let unsigned_bytes = serde_json::to_vec(&unsigned)?;
    let hash_hex = blake3::hash(&unsigned_bytes).to_hex().to_string();

    let mut lines = vec![
        format!("report:      {}", path.display()),
        format!("run_id:      {}", report.run_id),
        format!("passed:      {}", report.passed),
        format!("attestation: {attestation}"),
    ];
    if !attestation.starts_with(hi_trace::LOCAL_SIGNED_PREFIX) {
        lines.push(
            "signature:   not locally signed (worker or unknown scheme — not verifiable here)"
                .to_string(),
        );
        return Ok(lines);
    }
    if !key_path.exists() {
        lines.push("signature:   unverifiable (local signing key not found)".to_string());
        return Ok(lines);
    }
    match hi_trace::verify_local_signature(&attestation, &hash_hex, key_path) {
        Ok(true) => lines.push("signature:   ok (ed25519 signature matches local key)".to_string()),
        Ok(false) => bail!(
            "report {} signature does not match the local key (tampered or forged)",
            path.display()
        ),
        Err(e) => bail!("could not validate report signature: {e:#}"),
    }
    Ok(lines)
}

fn verify_report_cli(path: &Path) -> Result<()> {
    for line in verify_report(path)? {
        println!("{line}");
    }
    Ok(())
}

/// The objectives gate: passes only when no objective recorded failure
/// evidence. Human approval stays fail-closed (no gates in this graph).
pub(crate) struct ObjectiveGate;

#[async_trait]
impl GateAuthority for ObjectiveGate {
    async fn policy_gate(&self, _stage: &StageId, state: &RunState) -> Result<bool> {
        let failed: Vec<&str> = state
            .failure_evidence
            .iter()
            .filter(|failure| failure.subcategory == "objective_failed")
            .map(|failure| failure.stage.0.as_str())
            .collect();
        if !failed.is_empty() {
            eprintln!("objectives failed: {}", failed.join(", "));
        }
        Ok(failed.is_empty())
    }
    async fn human_approval(&self, stage: &StageId, _state: &RunState) -> Result<bool> {
        bail!(
            "human approval gate {} has no configured authority",
            stage.0
        )
    }
}

/// Local stage model: the planner is deterministic (the plan file IS the
/// plan), and each objective runs through the delegate runner — an isolated
/// worktree child whose diff applies only after its own verification passes.
pub(crate) struct LocalStageModel {
    objectives: BTreeMap<StageId, String>,
    plan_name: String,
    runner: Arc<dyn DelegateRunner>,
    verify: String,
    manifest_dir: PathBuf,
    /// Bounded per-objective repair attempts after a failed delegate run; the
    /// retry prompt carries the previous failure so the next child can
    /// address it instead of repeating it.
    retries: u32,
    /// `--bestof N`: when serial retries exhaust, run N diverse candidates in
    /// parallel worktrees and let the verification gate pick the winner —
    /// serial retries share the failed attempt's framing; diverse candidates
    /// don't. `None` = escalation off.
    escalation: Option<BestOfEscalation>,
}

/// Everything `bestof::run` needs to spawn candidate children, resolved once
/// at workflow start.
#[derive(Clone)]
pub(crate) struct BestOfEscalation {
    pub exe: PathBuf,
    pub provider: String,
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub workspace_root: PathBuf,
    pub state_root: PathBuf,
    pub candidates: u32,
    pub max_verify: u32,
}

impl LocalStageModel {
    fn seal_manifest(&self, stage: &StageId, body: &serde_json::Value) -> Result<ArtifactRef> {
        let bytes = serde_json::to_vec_pretty(body)?;
        let hash = blake3::hash(&bytes).to_hex().to_string();
        std::fs::create_dir_all(&self.manifest_dir)?;
        let path = self.manifest_dir.join(format!("{}-{hash}.json", stage.0));
        if !path.exists() {
            std::fs::write(&path, &bytes)
                .with_context(|| format!("writing objective manifest {}", path.display()))?;
        }
        Ok(ArtifactRef {
            hash,
            size_bytes: bytes.len() as u64,
            media_type: "application/json".into(),
        })
    }
}

#[async_trait]
impl StageModel for LocalStageModel {
    async fn invoke(
        &self,
        role: &str,
        stage: &StageId,
        _attempt: u32,
        state: &RunState,
    ) -> Result<StageOutcome> {
        match role {
            "planner" => {
                let revision = state
                    .plan
                    .as_ref()
                    .map(|plan| plan.revision)
                    .unwrap_or(0)
                    .saturating_add(1);
                Ok(StageOutcome {
                    passed: true,
                    output: serde_json::json!({
                        "plan": self.plan_name,
                        "objectives": self.objectives.len(),
                    }),
                    plan: Some(hi_rsi_runtime::EngineeringPlan {
                        objective: format!("execute {}", self.plan_name),
                        assumptions: vec![],
                        affected_components: vec![],
                        evidence: vec![],
                        proposed_changes: self.objectives.values().cloned().collect(),
                        tests: vec![format!("verification pipeline: {}", self.verify)],
                        risks: vec![],
                        rollback: "restore the sealed checkpoint (`/undo` semantics)".into(),
                        revision,
                        revision_reason: (revision > 1)
                            .then(|| "plan re-ingested on resume".to_string()),
                    }),
                    patches: vec![],
                    failures: vec![],
                    verification: None,
                })
            }
            "implementer" => {
                let objective = self
                    .objectives
                    .get(stage)
                    .ok_or_else(|| anyhow!("no objective text for stage {}", stage.0))?;
                let (index, total) = (
                    stage
                        .0
                        .trim_start_matches("objective_")
                        .trim_start_matches('0'),
                    self.objectives.len(),
                );
                let base_task = format!(
                    "Objective {index} of {total} from {}: {objective}\n\
                     Complete this objective fully. Keep the change scoped to this \
                     objective; other objectives run separately.",
                    self.plan_name
                );
                eprintln!("▶ {}: {objective}", stage.0);
                let mut outcome = self.runner.run(&base_task, Some(&self.verify)).await;
                let mut passed =
                    outcome.applied && outcome.status == hi_tools::ToolStatus::Succeeded;
                for retry in 1..=self.retries {
                    if passed {
                        break;
                    }
                    let failure_head: String = outcome.summary.chars().take(1_000).collect();
                    eprintln!("↻ {} retry {retry}/{}", stage.0, self.retries);
                    let task = format!(
                        "{base_task}\n\nA previous attempt at this objective failed:\n\
                         {failure_head}\nAddress that failure and complete the objective."
                    );
                    outcome = self.runner.run(&task, Some(&self.verify)).await;
                    passed = outcome.applied && outcome.status == hi_tools::ToolStatus::Succeeded;
                }
                if !passed && let Some(escalation) = &self.escalation {
                    // Serial retries carry the failed attempt's framing into
                    // the next one; diverse parallel candidates don't. The
                    // gate applies at most one verified winner.
                    eprintln!(
                        "⚡ {} escalating to best-of-{} diverse candidates",
                        stage.0, escalation.candidates
                    );
                    let failure_head: String = outcome.summary.chars().take(1_000).collect();
                    let prompt = format!(
                        "{base_task}\n\nSerial attempts at this objective failed:\n\
                         {failure_head}\nTake a fresh approach rather than repairing the \
                         previous attempt."
                    );
                    let escalation = escalation.clone();
                    let verify = self.verify.clone();
                    let merged = tokio::task::spawn_blocking(move || {
                        crate::bestof::run(&crate::bestof::BestOf {
                            exe: &escalation.exe,
                            provider: &escalation.provider,
                            model: &escalation.model,
                            base_url: &escalation.base_url,
                            api_key: &escalation.api_key,
                            verify: &verify,
                            prompt: &prompt,
                            candidates: escalation.candidates,
                            max_steps: None,
                            max_verify: escalation.max_verify,
                            workspace_root: &escalation.workspace_root,
                            state_root: &escalation.state_root,
                            report: None,
                            targets: None,
                            max_concurrency: escalation.candidates as usize,
                            apply: true,
                            fuzz: None,
                            expected_workspace_digest: None,
                        })
                    })
                    .await;
                    match merged {
                        Ok(Ok(true)) => {
                            passed = true;
                            outcome.applied = true;
                            outcome.status = hi_tools::ToolStatus::Succeeded;
                            outcome.summary = format!(
                                "best-of-{} escalation: a diverse candidate passed independent \
                                 verification and was merged",
                                self.escalation.as_ref().map(|e| e.candidates).unwrap_or(0)
                            );
                        }
                        Ok(Ok(false)) => eprintln!(
                            "✗ {} best-of escalation: no candidate survived the gate",
                            stage.0
                        ),
                        Ok(Err(error)) => {
                            eprintln!("✗ {} best-of escalation failed: {error:#}", stage.0);
                        }
                        Err(error) => {
                            eprintln!("✗ {} best-of escalation join error: {error}", stage.0);
                        }
                    }
                }
                let summary_head: String = outcome.summary.chars().take(2_000).collect();
                let manifest = serde_json::json!({
                    "stage": stage.0,
                    "objective": objective,
                    "applied": outcome.applied,
                    "changed_files": outcome.changed_files,
                    "summary": summary_head,
                });
                eprintln!(
                    "{} {}: {} file(s) changed",
                    if passed { "✓" } else { "✗" },
                    stage.0,
                    outcome.changed_files.len()
                );
                if !passed {
                    // The delegate summary carries the rejection reason —
                    // without it a failed objective is undiagnosable.
                    for line in summary_head.lines().take(4) {
                        eprintln!("    {line}");
                    }
                }
                let patches = if passed {
                    vec![self.seal_manifest(stage, &manifest)?]
                } else {
                    vec![]
                };
                let failures = if passed {
                    vec![]
                } else {
                    vec![FailureEvidence {
                        domain: FailureDomain::Candidate,
                        subcategory: "objective_failed".into(),
                        retryable: true,
                        causal_event_hash: None,
                        stage: stage.clone(),
                        artifacts: vec![],
                        counts_against_candidate: true,
                    }]
                };
                Ok(StageOutcome {
                    passed,
                    output: manifest,
                    plan: None,
                    patches,
                    failures,
                    verification: None,
                })
            }
            other => bail!("local workflow has no stage model for role {other}"),
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "this internal entry point mirrors the workflow CLI flags"
)]
async fn run(
    plan_path: &Path,
    verify_override: Option<String>,
    parallel: u16,
    retries: u32,
    bestof: u32,
    dry_run: bool,
    resume: bool,
    check_off: bool,
) -> Result<()> {
    let plan_text = std::fs::read_to_string(plan_path)
        .with_context(|| format!("reading plan {}", plan_path.display()))?;
    let plan_name = plan_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "plan.md".into());
    let objectives = parse_objectives(&plan_text);
    if objectives.is_empty() && plan_has_checked_objectives(&plan_text) {
        println!(
            "✓ {}: every objective is already checked off — nothing to build",
            plan_path.display()
        );
        return Ok(());
    }
    ensure!(
        !objectives.is_empty(),
        "no objectives found in {} — use unchecked `- [ ]` checkboxes, numbered items, or bullets",
        plan_path.display()
    );
    let rejected = actionability_issues(objectives.iter().map(String::as_str));
    let graph = plan_graph(objectives.len(), parallel)?;
    if dry_run {
        println!(
            "{}: {} objective(s), {} rejected, {} stages, wave concurrency {}",
            plan_name,
            objectives.len(),
            rejected.len(),
            graph.stages.len(),
            graph.limits.effective_concurrency()
        );
        for (index, objective) in objectives.iter().enumerate() {
            let mark = rejected
                .iter()
                .find(|(text, _)| text == objective)
                .map(|(_, reason)| format!("  REJECT {reason}"))
                .unwrap_or_default();
            println!("  {:>4}. {objective}{mark}", index + 1);
        }
        return Ok(());
    }
    if !rejected.is_empty() {
        let listing = rejected
            .iter()
            .map(|(text, reason)| format!("  - {text} ({reason})"))
            .collect::<Vec<_>>()
            .join("\n");
        bail!(
            "refusing to run {}: {} untestable/meta objective(s). Rewrite them as concrete deliverables, or use interactive /goal to drive a fuzzy list.\n{listing}",
            plan_path.display(),
            rejected.len()
        );
    }

    let (workspace_root, state_root) = crate::review_target::resolve_runtime_roots()?;
    ensure!(
        hi_tools::worktree::in_git_repo(&workspace_root),
        "hi workflow needs a git repository (objectives run in isolated worktrees)"
    );
    let cli = <crate::config::Cli as clap::Parser>::parse_from(["hi"]);
    let config = crate::config::load_config(None)?;
    let settings = crate::config::resolve(&cli, &config)?;
    let quality = crate::config::resolve_quality(&cli, &workspace_root)?;
    let verify_command = verify_override
        .or_else(|| {
            crate::report::pipeline_command(&quality.verification.resolved_stages(&workspace_root))
        })
        .filter(|command| !command.trim().is_empty())
        .ok_or_else(|| {
            anyhow!(
                "no verification pipeline resolved for this workspace; pass --verify \"<command>\""
            )
        })?;

    let plan_hash = blake3::hash(plan_text.as_bytes()).to_hex().to_string();
    let workflow_state = state_root.join("workflow").join(format!(
        "{}-{}",
        plan_name.replace('.', "_"),
        &plan_hash[..8]
    ));
    let checkpoint_dir = workflow_state.join("checkpoints");

    // `HI_IMPLEMENTER_MODEL` routes objective delegates to a different (often
    // faster) model than the session default — an explicit per-run choice, not
    // a guessed classification. The verification gate is model-agnostic, so a
    // cheaper implementer can't lower the bar for what merges.
    let implementer_model = std::env::var("HI_IMPLEMENTER_MODEL")
        .ok()
        .filter(|model| !model.trim().is_empty())
        .unwrap_or_else(|| settings.model.clone());
    let runner = crate::delegate::CliDelegateRunner::new(
        std::env::current_exe().context("resolving hi executable")?,
        crate::provider::provider_label(settings.provider).to_string(),
        implementer_model,
        settings.base_url.clone(),
        settings.api_key.clone(),
        Some(verify_command.clone()),
        cli.max_steps,
        quality.max_verify_repairs,
        workspace_root.clone(),
        state_root.clone(),
    )?;

    let objective_map: BTreeMap<StageId, String> = objectives
        .iter()
        .enumerate()
        .map(|(index, objective)| (objective_stage_id(index + 1), objective.clone()))
        .collect();
    let escalation = (bestof >= 2)
        .then(|| -> Result<BestOfEscalation> {
            Ok(BestOfEscalation {
                exe: std::env::current_exe().context("resolving hi executable")?,
                provider: crate::provider::provider_label(settings.provider).to_string(),
                model: settings.model.clone(),
                base_url: settings.base_url.clone(),
                api_key: settings.api_key.clone(),
                workspace_root: workspace_root.clone(),
                state_root: state_root.clone(),
                candidates: bestof,
                max_verify: quality.max_verify_repairs,
            })
        })
        .transpose()?;
    let model = LocalStageModel {
        objectives: objective_map.clone(),
        plan_name: plan_name.clone(),
        runner: Arc::new(runner),
        verify: verify_command.clone(),
        manifest_dir: workflow_state.join("artifacts"),
        retries,
        escalation,
    };
    let final_checks = vec![CheckSpec {
        name: "workspace_verification".into(),
        program: "/bin/sh".into(),
        arguments: vec!["-lc".into(), verify_command.clone()],
        timeout: std::time::Duration::from_secs(1_800),
        required: true,
        // Local self-hosted mode: the user's toolchain (rustup, nvm, …)
        // needs the user's environment. Reports stay `local-signed:`.
        inherit_environment: true,
    }];
    let driver = StageDriver::new(
        model,
        AttestingVerifier::new(
            LocalAttestor,
            blake3::hash(b"local-unattested-environment")
                .to_hex()
                .to_string(),
        )?,
        final_checks,
        BTreeMap::new(),
        ObjectiveGate,
        checkpoint_dir.clone(),
    )?;

    let budgets = RuntimeBudgets {
        wall_time_seconds: 24 * 60 * 60,
        cpu_time_seconds: 24 * 60 * 60,
        memory_bytes: 1,
        disk_bytes: 1,
        input_tokens: 1_000_000_000,
        output_tokens: 1_000_000_000,
        tool_calls: (objectives.len() as u64 + 16) * 8,
        cost_microusd: 1,
        model_calls: (objectives.len() as u32 + 16) * 2,
        repair_iterations: 4,
        trace_bytes: 1_000_000_000,
    };
    let starting_commit = git_head(&workspace_root).unwrap_or_else(|| "unknown".into());
    let mut state = RunState {
        task_id: plan_name.clone(),
        run_id: format!("local-{}", &plan_hash[..16]),
        candidate_id: "local".into(),
        repository: hi_rsi_runtime::RepositoryState {
            repository_snapshot_hash: plan_hash.clone(),
            starting_commit,
            source_tree_hash: blake3::hash(b"local-worktree").to_hex().to_string(),
            worktree_root: workspace_root.to_string_lossy().into_owned(),
            submodule_commits: BTreeMap::new(),
        },
        current_stages: BTreeSet::new(),
        attempts: BTreeMap::new(),
        working_memory: vec![],
        plan: None,
        patches: vec![],
        verification: vec![],
        budget: Default::default(),
        failure_evidence: vec![],
    };

    // Workflow execution is now represented in the shared local control
    // plane. The filesystem checkpoint remains the workflow engine's durable
    // replay artifact; the control record owns attempts, fencing, and audit.
    let control_store = hi_control::ControlStore::open_for_state(&state_root)?;
    control_store.recover_expired_attempts(hi_control::now_ms())?;
    if control_store.get_run(&state.run_id)?.is_none() {
        control_store.create_run(hi_control::NewRun {
            run_id: Some(state.run_id.clone()),
            kind: hi_control::RunKind::Workflow,
            workspace_id: Some(workspace_root.to_string_lossy().into_owned()),
            scope: None,
            session_id: None,
            parent_run_id: None,
            policy_snapshot: None,
            route_snapshot: Some(hi_control::RouteSnapshot {
                harness: Some("hi".into()),
                provider: Some(crate::provider::provider_label(settings.provider).into()),
                model: Some(settings.model.clone()),
                capability_digest: None,
            }),
            provenance: Some(hi_control::Provenance {
                principal: hi_control::Principal {
                    id: "local-process".into(),
                    kind: "local_cli".into(),
                },
                source: "workflow_cli".into(),
                run_id: Some(state.run_id.clone()),
                attempt_id: None,
                parent_ref: None,
                correlation_id: None,
                policy_version: None,
            }),
            desired_state: hi_control::DesiredState::Run,
        })?;
    } else {
        control_store.requeue_run(&state.run_id, hi_control::now_ms())?;
    }
    let lease = control_store.claim_attempt(
        &state.run_id,
        &format!("workflow-worker-{}", std::process::id()),
        hi_control::now_ms(),
        hi_control::DEFAULT_LEASE_TTL_MS,
    )?;
    publish_control_event(
        &control_store,
        EventKind::AttemptClaimed,
        &state.run_id,
        &lease.attempt.attempt_id,
        ActivityState::Running,
        ActivityVerb::Start,
        "workflow attempt claimed",
    )?;

    println!(
        "workflow {}: {} objective(s), waves of {}, verify: {}",
        plan_name,
        objectives.len(),
        graph.limits.effective_concurrency(),
        verify_command
    );
    println!("state: {}", workflow_state.display());

    let executor = WorkflowExecutor::new(graph, driver, SharedBudgetLedger::new(&budgets));
    let outcome = if resume {
        let checkpoint = latest_checkpoint(&checkpoint_dir)?.ok_or_else(|| {
            anyhow!(
                "no sealed checkpoint to resume under {}",
                checkpoint_dir.display()
            )
        })?;
        println!(
            "resuming from checkpoint sequence {} at {:?}",
            checkpoint.created_at_sequence,
            checkpoint
                .workflow_position
                .iter()
                .map(|stage| stage.0.as_str())
                .collect::<Vec<_>>()
        );
        executor.resume(&checkpoint, &mut state).await?
    } else {
        executor.execute(&mut state).await?
    };

    let attempt_status = match &outcome {
        TerminalOutcome::Succeeded => hi_control::AttemptStatus::Succeeded,
        TerminalOutcome::Failed => hi_control::AttemptStatus::Failed,
    };
    control_store.complete_attempt(
        &lease.attempt.attempt_id,
        lease.fencing_token,
        attempt_status,
        hi_control::now_ms(),
        (attempt_status == hi_control::AttemptStatus::Failed).then_some("workflow failed"),
    )?;
    publish_control_event(
        &control_store,
        match attempt_status {
            hi_control::AttemptStatus::Succeeded => EventKind::AttemptCompleted,
            _ => EventKind::AttemptFailed,
        },
        &state.run_id,
        &lease.attempt.attempt_id,
        match attempt_status {
            hi_control::AttemptStatus::Succeeded => ActivityState::Succeeded,
            _ => ActivityState::Failed,
        },
        match attempt_status {
            hi_control::AttemptStatus::Succeeded => ActivityVerb::Complete,
            _ => ActivityVerb::Fail,
        },
        "workflow attempt completed",
    )?;

    // Persist the final signed verification report to a known location so
    // `hi workflow verify` can resolve it without an explicit path. The last
    // trusted-gate report is the authoritative one; writing it under the
    // workflow state dir keeps it beside the checkpoints it summarizes.
    if let Some(report) = state.verification.last() {
        let report_path = workflow_state.join("report.json");
        match serde_json::to_vec_pretty(report) {
            Ok(bytes) => {
                if let Err(error) = std::fs::write(&report_path, bytes) {
                    eprintln!(
                        "could not persist verification report to {}: {error:#}",
                        report_path.display()
                    );
                }
            }
            Err(error) => eprintln!("could not serialize verification report: {error:#}"),
        }
    }

    let failed: Vec<&str> = state
        .failure_evidence
        .iter()
        .filter(|failure| failure.subcategory == "objective_failed")
        .map(|failure| failure.stage.0.as_str())
        .collect();
    if check_off {
        let succeeded: Vec<&str> = objective_map
            .iter()
            .filter(|(stage, _)| !failed.contains(&stage.0.as_str()))
            .map(|(_, objective)| objective.as_str())
            .collect();
        match check_off_objectives(plan_path, &succeeded) {
            Ok(0) => {}
            Ok(count) => println!(
                "checked off {count} objective(s) in {}",
                plan_path.display()
            ),
            Err(error) => eprintln!(
                "could not check off objectives in {}: {error:#}",
                plan_path.display()
            ),
        }
    }
    match outcome {
        TerminalOutcome::Succeeded => {
            println!(
                "✓ workflow complete: {} objective(s) applied, workspace verification passed",
                state.patches.len()
            );
            Ok(())
        }
        TerminalOutcome::Failed => {
            println!(
                "✗ workflow failed: {} of {} objective(s) did not verify{}",
                failed.len(),
                state.patches.len() + failed.len(),
                if failed.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", failed.join(", "))
                }
            );
            println!(
                "fix or reword those objectives in the plan, then `hi workflow run` again — \
                 completed objectives' changes are already applied to the worktree"
            );
            std::process::exit(1);
        }
    }
}

fn publish_control_event(
    store: &hi_control::ControlStore,
    kind: EventKind,
    run_id: &str,
    attempt_id: &str,
    state: ActivityState,
    verb: ActivityVerb,
    title: &str,
) -> Result<()> {
    store.append_event(
        RunEvent::new(
            kind,
            EventContext {
                run_id: Some(run_id.into()),
                attempt_id: Some(attempt_id.into()),
                ..EventContext::default()
            },
            SemanticActivity {
                verb,
                object: ActivityObject::Run,
                state,
                group_key: format!("run:{run_id}"),
                title: title.into(),
                detail: None,
                refs: vec![],
                progress: None,
            },
        )
        .required(),
    )?;
    Ok(())
}

/// Mark the given objective texts `- [x]` in a checkbox-format plan. Only
/// exact unchecked-checkbox lines are rewritten; numbered/bullet plans (which
/// have no checkbox state) are left untouched. Returns how many lines changed.
fn check_off_objectives(plan_path: &Path, succeeded: &[&str]) -> Result<usize> {
    let text = std::fs::read_to_string(plan_path)
        .with_context(|| format!("reading plan {}", plan_path.display()))?;
    let mut changed = 0usize;
    let rewritten: Vec<String> = text
        .lines()
        .map(|line| {
            let trimmed = line.trim_start();
            for marker in ["- [ ]", "* [ ]", "+ [ ]"] {
                if let Some(rest) = trimmed.strip_prefix(marker)
                    && succeeded.contains(&rest.trim())
                {
                    changed += 1;
                    let indent = &line[..line.len() - trimmed.len()];
                    let checked = marker.replace("[ ]", "[x]");
                    return format!("{indent}{checked}{rest}");
                }
            }
            line.to_string()
        })
        .collect();
    if changed > 0 {
        let mut output = rewritten.join("\n");
        if text.ends_with('\n') {
            output.push('\n');
        }
        std::fs::write(plan_path, output)
            .with_context(|| format!("updating plan {}", plan_path.display()))?;
    }
    Ok(changed)
}

fn git_head(root: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_attestor_signs_and_signature_verifies() {
        // The workflow LocalAttestor must emit a local-signed: ed25519
        // signature over the hex report hash, verifiable with the local key.
        let base = std::env::temp_dir().join(format!("hi-wf-attest-{}", std::process::id()));
        let key_path = base.join("hi").join(hi_trace::LOCAL_SIGNING_KEY_FILE);
        let report_hash = [7u8; 32];
        let attestation = LocalAttestor::sign_with_key(&report_hash, &key_path).unwrap();
        assert!(
            attestation.starts_with(hi_trace::LOCAL_SIGNED_PREFIX),
            "expected local-signed: attestation, got: {attestation}"
        );
        let hash_hex = blake3::Hash::from_bytes(report_hash).to_hex();
        assert!(
            hi_trace::verify_local_signature(&attestation, hash_hex.as_str(), &key_path).unwrap(),
            "signature must verify against the local key"
        );
        // A different report hash must not verify.
        assert!(
            !hi_trace::verify_local_signature(
                &attestation,
                blake3::Hash::from_bytes([9u8; 32]).to_hex().as_str(),
                &key_path
            )
            .unwrap(),
            "tampered report hash must not verify"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Write a VerificationReport whose attestation is a real local-signed:
    /// signature, produced the same way AttestingVerifier does (clear the
    /// field, hash the unsigned bytes, sign). Returns the report path.
    fn write_signed_report(dir: &Path, key_path: &Path, passed: bool) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let mut report = VerificationReport {
            report_version: 1,
            run_id: "run-1".into(),
            candidate_id: "cand-1".into(),
            environment_hash: "e".repeat(64),
            source_tree_hash: "s".repeat(64),
            checks: vec![],
            passed,
            policy_violations: vec![],
            artifacts: vec![],
            supervisor_attestation: None,
        };
        let unsigned = serde_json::to_vec(&report).unwrap();
        let hash = blake3::hash(&unsigned);
        report.supervisor_attestation =
            Some(LocalAttestor::sign_with_key(hash.as_bytes(), key_path).unwrap());
        let path = dir.join("report.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        path
    }

    #[test]
    fn workflow_verify_accepts_valid_signature() {
        let base = std::env::temp_dir().join(format!("hi-wf-verify-ok-{}", std::process::id()));
        let key_path = base.join("hi").join(hi_trace::LOCAL_SIGNING_KEY_FILE);
        let report_path = write_signed_report(&base, &key_path, true);
        let out = verify_report_with_key(&report_path, &key_path)
            .unwrap()
            .join("\n");
        assert!(out.contains("signature:   ok"), "expected ok: {out}");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn workflow_verify_rejects_tampered_report() {
        let base = std::env::temp_dir().join(format!("hi-wf-verify-bad-{}", std::process::id()));
        let key_path = base.join("hi").join(hi_trace::LOCAL_SIGNING_KEY_FILE);
        let report_path = write_signed_report(&base, &key_path, true);
        // Flip the `passed` flag after signing — the recomputed unsigned hash
        // no longer matches the signature.
        let mut report: VerificationReport =
            serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
        report.passed = false;
        std::fs::write(&report_path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        let result = verify_report_with_key(&report_path, &key_path);
        assert!(result.is_err(), "tampered report must fail verification");
        let err = result.unwrap_err();
        assert!(
            format!("{err:#}").contains("does not match"),
            "expected a mismatch error: {err:#}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn workflow_verify_handles_unsigned_and_foreign_scheme() {
        let base = std::env::temp_dir().join(format!("hi-wf-verify-none-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let key_path = base.join("hi").join(hi_trace::LOCAL_SIGNING_KEY_FILE);
        // Unsigned report (no attestation) -> hard error.
        let unsigned_report = VerificationReport {
            report_version: 1,
            run_id: "run-1".into(),
            candidate_id: "cand-1".into(),
            environment_hash: "e".repeat(64),
            source_tree_hash: "s".repeat(64),
            checks: vec![],
            passed: true,
            policy_violations: vec![],
            artifacts: vec![],
            supervisor_attestation: None,
        };
        let path = base.join("unsigned.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&unsigned_report).unwrap()).unwrap();
        assert!(verify_report_with_key(&path, &key_path).is_err());

        // A worker-scheme attestation is reported as not locally verifiable.
        let mut worker = unsigned_report.clone();
        worker.supervisor_attestation = Some("worker-v1:deadbeef".into());
        let wpath = base.join("worker.json");
        std::fs::write(&wpath, serde_json::to_vec_pretty(&worker).unwrap()).unwrap();
        let out = verify_report_with_key(&wpath, &key_path)
            .unwrap()
            .join("\n");
        assert!(
            out.contains("not locally signed"),
            "expected not-locally-signed note: {out}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn workflow_verify_reports_missing_key() {
        let base = std::env::temp_dir().join(format!("hi-wf-verify-key-{}", std::process::id()));
        let key_path = base.join("hi").join(hi_trace::LOCAL_SIGNING_KEY_FILE);
        let report_path = write_signed_report(&base, &key_path, true);
        // Point at a key path that does not exist -> unverifiable, not an error.
        let missing = base.join("absent").join("key");
        let out = verify_report_with_key(&report_path, &missing)
            .unwrap()
            .join("\n");
        assert!(
            out.contains("unverifiable"),
            "expected unverifiable note: {out}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn latest_workflow_report_resolves_newest_persisted_report() {
        // Two persisted reports under state_root/workflow/<plan>/report.json;
        // the resolver must return the most recently written one.
        let base = std::env::temp_dir().join(format!("hi-wf-latest-{}", std::process::id()));
        let key_path = base.join("hi").join(hi_trace::LOCAL_SIGNING_KEY_FILE);
        let state_root = base.join("state");
        let older = state_root.join("workflow").join("plan-aaa");
        let newer = state_root.join("workflow").join("plan-bbb");
        write_signed_report(&older, &key_path, true);
        // Ensure a measurable mtime gap so ordering is deterministic.
        std::thread::sleep(std::time::Duration::from_millis(10));
        let newer_report = write_signed_report(&newer, &key_path, false);

        let resolved = latest_workflow_report_under(&state_root).unwrap();
        assert_eq!(
            resolved,
            newer_report,
            "resolver must pick the newest report, got {}",
            resolved.display()
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn latest_workflow_report_none_when_no_reports() {
        let base = std::env::temp_dir().join(format!("hi-wf-latest-none-{}", std::process::id()));
        let state_root = base.join("state");
        std::fs::create_dir_all(state_root.join("workflow")).unwrap();
        assert!(latest_workflow_report_under(&state_root).is_none());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn objectives_prefer_unchecked_checkboxes_then_numbers_then_bullets() {
        let plan = "# Plan\n- [x] done already\n- [ ] first objective\n- [ ] second objective\n";
        assert_eq!(
            parse_objectives(plan),
            vec!["first objective", "second objective"]
        );
        let numbered = "notes\n1. build the loader\n2) train the model\n";
        assert_eq!(
            parse_objectives(numbered),
            vec!["build the loader", "train the model"]
        );
        let bullets = "* add tests\n* fix docs\n";
        assert_eq!(parse_objectives(bullets), vec!["add tests", "fix docs"]);
        assert!(parse_objectives("# just prose\n").is_empty());
    }

    #[test]
    fn actionability_issues_flag_meta_objectives() {
        let issues = actionability_issues(
            [
                "add a --seed flag",
                "investigate the parser",
                "Final workspace validation",
            ]
            .into_iter(),
        );
        assert_eq!(issues.len(), 2);
        assert!(issues.iter().any(|(t, _)| t.contains("investigate")));
        assert!(issues.iter().any(|(t, _)| t.contains("validation")));
    }

    #[test]
    fn plan_graph_validates_across_sizes_and_bounds_concurrency() {
        for count in [1, 2, 7, 200] {
            let graph = plan_graph(count, 4).unwrap();
            assert_eq!(graph.limits.effective_concurrency(), 4);
            assert_eq!(
                graph
                    .stages
                    .values()
                    .filter(|stage| stage.kind == StageKind::ModelInvocation
                        && stage.model_role.as_deref() == Some("implementer"))
                    .count(),
                count
            );
        }
        assert!(plan_graph(0, 4).is_err());
        assert!(plan_graph(MAX_OBJECTIVES + 1, 4).is_err());
    }

    struct FlakyRunner {
        calls: std::sync::Mutex<u32>,
        succeed_on: u32,
    }

    #[async_trait]
    impl DelegateRunner for FlakyRunner {
        async fn run(&self, task: &str, _verify: Option<&str>) -> hi_agent::DelegateOutcome {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            if *calls >= self.succeed_on {
                hi_agent::DelegateOutcome {
                    status: hi_tools::ToolStatus::Succeeded,
                    applied: true,
                    changed_files: vec!["src/lib.rs".into()],
                    summary: format!("applied on attempt {} for: {task}", *calls),
                }
            } else {
                hi_agent::DelegateOutcome {
                    status: hi_tools::ToolStatus::Failed,
                    applied: false,
                    changed_files: vec![],
                    summary: "verification failed: expected 5 models, found 4".into(),
                }
            }
        }
        async fn run_cancellable(
            &self,
            task: &str,
            verify: Option<&str>,
            _cancellation: hi_agent::TurnCancellation,
        ) -> hi_agent::DelegateOutcome {
            self.run(task, verify).await
        }
    }

    #[tokio::test]
    async fn objective_retries_carry_failure_context_and_are_bounded() {
        let base = tempfile::tempdir().unwrap();
        let model = |retries: u32, succeed_on: u32| LocalStageModel {
            objectives: BTreeMap::from([(
                StageId::from("objective_0001"),
                "add the fifth model".to_string(),
            )]),
            plan_name: "plan.md".into(),
            runner: Arc::new(FlakyRunner {
                calls: std::sync::Mutex::new(0),
                succeed_on,
            }),
            verify: "true".into(),
            manifest_dir: base.path().join("artifacts"),
            retries,
            escalation: None,
        };
        let state = gate_state();

        // One retry allowed, second attempt succeeds — and the retry prompt
        // carries the previous failure text.
        let outcome = model(1, 2)
            .invoke("implementer", &StageId::from("objective_0001"), 1, &state)
            .await
            .unwrap();
        assert!(outcome.passed);
        assert!(
            outcome.output["summary"]
                .as_str()
                .unwrap()
                .contains("attempt 2 for:")
        );
        assert!(
            outcome.output["summary"]
                .as_str()
                .unwrap()
                .contains("A previous attempt at this objective failed"),
            "retry prompt must carry the failure context"
        );

        // Zero retries: a single failure stands.
        let outcome = model(0, 2)
            .invoke("implementer", &StageId::from("objective_0001"), 1, &state)
            .await
            .unwrap();
        assert!(!outcome.passed);
        assert_eq!(outcome.failures.len(), 1);

        // Bounded: retries exhausted without success stays failed.
        let outcome = model(2, 9)
            .invoke("implementer", &StageId::from("objective_0001"), 1, &state)
            .await
            .unwrap();
        assert!(!outcome.passed);
    }

    pub(super) fn gate_state() -> RunState {
        RunState {
            task_id: "t".into(),
            run_id: "r".into(),
            candidate_id: "c".into(),
            repository: hi_rsi_runtime::RepositoryState {
                repository_snapshot_hash: "a".repeat(64),
                starting_commit: "x".into(),
                source_tree_hash: "b".repeat(64),
                worktree_root: "/tmp".into(),
                submodule_commits: BTreeMap::new(),
            },
            current_stages: BTreeSet::new(),
            attempts: BTreeMap::new(),
            working_memory: vec![],
            plan: None,
            patches: vec![],
            verification: vec![],
            budget: Default::default(),
            failure_evidence: vec![],
        }
    }

    #[test]
    fn check_off_marks_only_succeeded_checkbox_objectives() {
        let dir = tempfile::tempdir().unwrap();
        let plan = dir.path().join("plan.md");
        std::fs::write(
            &plan,
            "# Plan\n- [x] already done\n- [ ] first objective\n  - [ ] indented objective\n- [ ] failed objective\n",
        )
        .unwrap();
        let changed =
            check_off_objectives(&plan, &["first objective", "indented objective"]).unwrap();
        assert_eq!(changed, 2);
        let text = std::fs::read_to_string(&plan).unwrap();
        assert!(text.contains("- [x] first objective"));
        assert!(text.contains("  - [x] indented objective"), "{text}");
        assert!(text.contains("- [ ] failed objective"));
        // Idempotent: nothing left to change on a rerun.
        assert_eq!(
            check_off_objectives(&plan, &["first objective"]).unwrap(),
            0
        );
        // Numbered plans have no checkbox state to update.
        std::fs::write(&plan, "1. build it\n2. test it\n").unwrap();
        assert_eq!(check_off_objectives(&plan, &["build it"]).unwrap(), 0);
    }

    #[test]
    fn all_checked_plans_are_recognized() {
        assert!(plan_has_checked_objectives("- [x] done\n- [X] also done\n"));
        assert!(!plan_has_checked_objectives("# just prose\n"));
        assert!(parse_objectives("- [x] done\n").is_empty());
    }

    #[tokio::test]
    async fn objective_gate_fails_only_on_objective_failures() {
        let mut state = gate_state();
        let gate = ObjectiveGate;
        let stage = StageId::from("objectives_gate");
        assert!(gate.policy_gate(&stage, &state).await.unwrap());
        state.failure_evidence.push(FailureEvidence {
            domain: FailureDomain::Candidate,
            subcategory: "objective_failed".into(),
            retryable: true,
            causal_event_hash: None,
            stage: StageId::from("objective_0001"),
            artifacts: vec![],
            counts_against_candidate: true,
        });
        assert!(!gate.policy_gate(&stage, &state).await.unwrap());
    }
}
