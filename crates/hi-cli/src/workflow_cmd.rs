//! `hi workflow run plan.md` — local, self-hosted workflow execution.
//!
//! ADR-001 carve-out: this entry drives `hi_agent_runtime::WorkflowExecutor`
//! locally, WITHOUT the managed attestation chain. Verification reports are
//! attested with an explicit `local-unattested:` label so they can never be
//! mistaken for worker-attested RSI evidence. Each plan objective executes
//! through the existing delegate machinery — an isolated worktree child that
//! must verify before its diff is applied — so "passed" always means applied
//! AND verified, never narrated.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use hi_agent::DelegateRunner;
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
    StageDefinition, StageId, StageKind, TransitionCondition, TransitionRule, WorkflowGraph,
    WorkflowLimits,
};
use hi_verifier::{AttestingVerifier, Attestor, CheckSpec};

/// Objectives above this need a split plan — a single run of thousands of
/// delegate children is not a supervisable unit of work.
const MAX_OBJECTIVES: usize = 512;
/// Default concurrent objective delegates per wave; the cross-process
/// resource governor additionally caps live children machine-wide.
const DEFAULT_WAVE_CONCURRENCY: u16 = 4;

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
         hi workflow resume <plan.md>\n\n\
         Objectives are unchecked markdown checkboxes (`- [ ] …`), else numbered\n\
         items, else bullets. Each objective runs as an isolated delegate child\n\
         that must pass verification before its diff is applied. A final trusted\n\
         verification gate runs the same pipeline across the whole workspace.\n\
         `--bestof N` (2-4): when an objective still fails after its retries,\n\
         run N diverse candidates in parallel worktrees and merge the one that\n\
         passes independent verification.\n\
         `--check-off` marks succeeded objectives `- [x]` in the plan after the\n\
         run, so a rerun only retries what failed.\n\
         Reports are labeled `local-unattested:` — this is the self-hosted mode,\n\
         not managed RSI evidence. Checkpoints live under the state root keyed\n\
         by plan content; `resume` continues the latest sealed checkpoint."
    );
}

/// Extract objectives: unchecked checkboxes first, then numbered items, then
/// plain bullets. Checked boxes are respected as already done.
pub(crate) fn parse_objectives(markdown: &str) -> Vec<String> {
    let lines: Vec<&str> = markdown.lines().map(str::trim).collect();
    let checkbox = |line: &&str| -> Option<String> {
        ["- [ ]", "* [ ]", "+ [ ]"]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix))
            .map(|rest| rest.trim().to_string())
    };
    let unchecked: Vec<String> = lines.iter().filter_map(checkbox).collect();
    if !unchecked.is_empty() {
        return unchecked.into_iter().filter(|o| !o.is_empty()).collect();
    }
    let numbered: Vec<String> = lines
        .iter()
        .filter_map(|line| {
            let digits = line.chars().take_while(char::is_ascii_digit).count();
            if digits == 0 {
                return None;
            }
            let rest = &line[digits..];
            rest.strip_prefix('.')
                .or_else(|| rest.strip_prefix(')'))
                .map(|text| text.trim().to_string())
        })
        .filter(|objective| !objective.is_empty())
        .collect();
    if !numbered.is_empty() {
        return numbered;
    }
    lines
        .iter()
        .filter(|line| !checkbox_done(line))
        .filter_map(|line| {
            ["- ", "* ", "+ "]
                .iter()
                .find_map(|prefix| line.strip_prefix(prefix))
                .map(|text| text.trim().to_string())
        })
        .filter(|objective| !objective.is_empty() && !objective.starts_with("[x]"))
        .collect()
}

/// Whether the plan contains checked-off checkbox objectives — used to
/// distinguish "everything already done" (success) from "not a plan" (error).
fn plan_has_checked_objectives(markdown: &str) -> bool {
    markdown
        .lines()
        .map(str::trim)
        .any(|line| checkbox_done(&line))
}

fn checkbox_done(line: &&str) -> bool {
    ["- [x]", "* [x]", "+ [x]", "- [X]", "* [X]", "+ [X]"]
        .iter()
        .any(|prefix| line.starts_with(prefix))
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

/// Explicitly-labeled local attestation: honest about being self-hosted.
pub(crate) struct LocalAttestor;

impl Attestor for LocalAttestor {
    fn attest(&self, report_hash: &[u8; 32]) -> Result<String> {
        Ok(format!(
            "local-unattested:{}",
            blake3::Hash::from_bytes(*report_hash).to_hex()
        ))
    }
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
    let graph = plan_graph(objectives.len(), parallel)?;
    if dry_run {
        println!(
            "{}: {} objective(s), {} stages, wave concurrency {}",
            plan_name,
            objectives.len(),
            graph.stages.len(),
            graph.limits.effective_concurrency()
        );
        for (index, objective) in objectives.iter().enumerate() {
            println!("  {:>4}. {objective}", index + 1);
        }
        return Ok(());
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
        // needs the user's environment. Reports stay `local-unattested:`.
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
        attempt_status.clone(),
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
