//! Workspace repair verification — the interactive turn-loop subsystem.
//!
//! After the model stops calling tools, [`WorkspaceRepairVerifier`] runs the
//! configured pipeline stages in order (cheap compile/typecheck first, then
//! lint, then tests); the first to fail stops the turn and its output is fed
//! back to the model for another attempt, up to `max_rounds`. A passing
//! pipeline ends the turn. The "only verify turns that changed files" gating
//! lives here too — a turn that edited nothing can't have introduced a failure.
//!
//! **Not** review-answer repair ([`crate::steering::ReviewRepairMode`]) and
//! **not** RSI attestation ([`hi_verifier::AttestingVerifier`]). See
//! [`crate::agent::turn::phase::TurnPhase`] and `docs/architecture.md`.
//!
//! Extracted so the verify state machine (round counter, outcome) is owned by
//! one small type instead of entangled with the main loop's locals and the
//! `Agent`'s shared mutable fields.

use anyhow::Context;

use crate::config::VerifyStage;
use crate::snapshot::{
    FileFingerprint, SnapshotCache, changed_files_between, workspace_snapshot_meta,
};
use crate::ui::Ui;
use crate::workspace_coordination::WorkspaceCoordination;

const VERIFICATION_EXECUTION_LIMIT: usize = 256;
const VERIFICATION_EXECUTION_HEAD: usize = 32;
type VerificationExecutionLog = crate::diagnostic_retention::BoundedDiagnosticLog<
    VerificationExecution,
    VERIFICATION_EXECUTION_LIMIT,
    VERIFICATION_EXECUTION_HEAD,
>;

/// One verification-stage execution retained as report evidence.
///
/// LSP diagnostics do not launch a process, so `process` and `truncation` are
/// absent for those records. Shell stages preserve the exact structured
/// process result returned by `hi-tools`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VerificationExecution {
    /// One-based verification round.
    pub round: u32,
    pub name: String,
    pub command: String,
    pub status: hi_tools::ToolStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process: Option<hi_tools::ProcessOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<hi_tools::TruncationState>,
}

impl VerificationExecution {
    fn lsp(round: u32, status: hi_tools::ToolStatus) -> Self {
        Self {
            round,
            name: "lsp".to_string(),
            command: "diagnostics".to_string(),
            status,
            process: None,
            truncation: None,
        }
    }

    fn shell(round: u32, stage: &VerifyStage, execution: &hi_tools::ProcessExecution) -> Self {
        Self {
            round,
            name: stage.name.clone(),
            command: stage.command.clone(),
            status: execution.status,
            process: Some(execution.model_outcome()),
            truncation: Some(execution.truncation.clone()),
        }
    }

    fn infrastructure_failure(round: u32, stage: &VerifyStage) -> Self {
        Self {
            round,
            name: stage.name.clone(),
            command: stage.command.clone(),
            status: hi_tools::ToolStatus::Failed,
            process: None,
            truncation: None,
        }
    }
}

/// The snapshot type the verifier compares against.
pub(crate) type Snapshot = std::collections::BTreeMap<String, FileFingerprint>;

/// Workspace-local dependencies for one verifier check. Keeping these bound
/// together makes it difficult to accidentally pair a checkpoint or LSP
/// manager with a different workspace root.
pub(crate) struct VerifyWorkspace<'a> {
    root: &'a std::path::Path,
    state_root: &'a std::path::Path,
    /// The agent-owned runner. `None` is retained for small verifier unit
    /// tests that construct this value directly; live turns always provide it
    /// so verification uses the same sandbox policy as tool execution.
    process_runner: Option<&'a hi_tools::ProcessRunner>,
    pre_turn_checkpoint: Option<&'a str>,
    lsp: &'a hi_lsp::LspManager,
    known_changed_files: Option<&'a [String]>,
    mutation_seen: bool,
    /// Packages mid-turn `cargo check` already sealed green at the current
    /// ledger revision — skip matching `affected-check:` stages (Phase I).
    skip_affected_checks: Option<&'a std::collections::BTreeSet<String>>,
    /// Packages mid-turn `cargo test` already sealed green — skip `affected-test:`.
    skip_affected_tests: Option<&'a std::collections::BTreeSet<String>>,
    /// The always-present controller used by live agent turns. Small verifier
    /// unit tests may omit it; they never model a remotely authoritative
    /// workspace.
    coordination: Option<WorkspaceCoordination>,
    durability: Option<std::sync::Arc<dyn crate::WorkspaceDurability>>,
}

impl<'a> VerifyWorkspace<'a> {
    pub(crate) fn new(
        root: &'a std::path::Path,
        state_root: &'a std::path::Path,
        pre_turn_checkpoint: Option<&'a str>,
        lsp: &'a hi_lsp::LspManager,
    ) -> Self {
        Self {
            root,
            state_root,
            process_runner: None,
            pre_turn_checkpoint,
            lsp,
            known_changed_files: None,
            mutation_seen: false,
            skip_affected_checks: None,
            skip_affected_tests: None,
            coordination: None,
            durability: None,
        }
    }

    pub(crate) fn with_process_runner(
        mut self,
        process_runner: &'a hi_tools::ProcessRunner,
    ) -> Self {
        self.process_runner = Some(process_runner);
        self
    }

    /// Use the content ledger's complete turn-relative change universe for
    /// verification gating. Snapshot comparison remains the fallback in unit
    /// tests and for stage-mutation detection.
    pub(crate) fn with_changed_files(mut self, changed_files: &'a [String]) -> Self {
        self.known_changed_files = Some(changed_files);
        self
    }

    /// Require configured validation after an applied mutation even if later
    /// edits restored the original bytes and the net changed-file set is empty.
    pub(crate) fn with_mutation_seen(mut self, mutation_seen: bool) -> Self {
        self.mutation_seen = mutation_seen;
        self
    }

    /// Drop affected cargo check/test stages already proven green mid-turn at
    /// this ledger revision (see `FastFeedbackState` seals).
    pub(crate) fn with_skippable_affected(
        mut self,
        checks: &'a std::collections::BTreeSet<String>,
        tests: &'a std::collections::BTreeSet<String>,
    ) -> Self {
        self.skip_affected_checks = Some(checks);
        self.skip_affected_tests = Some(tests);
        self
    }

    /// Bind native verifier process execution to the same admission and
    /// settlement path used by model-authored tools.
    pub(crate) fn with_workspace_coordination(
        mut self,
        coordination: WorkspaceCoordination,
        durability: Option<std::sync::Arc<dyn crate::WorkspaceDurability>>,
    ) -> Self {
        self.coordination = Some(coordination);
        self.durability = durability;
        self
    }
}

async fn run_check_for_workspace(
    workspace: &VerifyWorkspace<'_>,
    command: &str,
    timeout: Option<std::time::Duration>,
) -> anyhow::Result<hi_tools::ProcessExecution> {
    match workspace.process_runner {
        Some(runner) => {
            hi_tools::run_check_in_with_runner_maybe_timeout(runner, command, timeout).await
        }
        None => {
            let runner = verification_runner(workspace.root, None)?;
            hi_tools::run_check_in_with_runner_maybe_timeout(&runner, command, timeout).await
        }
    }
}

async fn run_check_for_workspace_with_timeout(
    workspace: &VerifyWorkspace<'_>,
    command: &str,
    timeout: std::time::Duration,
) -> anyhow::Result<hi_tools::ProcessExecution> {
    match workspace.process_runner {
        Some(runner) => hi_tools::run_check_in_with_runner_timeout(runner, command, timeout).await,
        None => {
            let runner = verification_runner(workspace.root, None)?;
            hi_tools::run_check_in_with_runner_timeout(&runner, command, timeout).await
        }
    }
}

fn verification_runner(
    root: &std::path::Path,
    policy: Option<hi_tools::sandbox::SandboxPolicy>,
) -> anyhow::Result<hi_tools::ProcessRunner> {
    if let Some(policy) = policy {
        return hi_tools::ProcessRunner::new_with_policy(root, policy);
    }
    // Direct verifier tests construct `VerifyWorkspace` without the live
    // runtime. Keep those tests deterministic and independent of the host's
    // default sandbox; real turns always pass their configured runner above.
    #[cfg(test)]
    {
        hi_tools::ProcessRunner::new_with_policy(root, hi_tools::sandbox::SandboxPolicy::Off)
    }
    #[cfg(not(test))]
    {
        hi_tools::ProcessRunner::new(root)
    }
}

/// Commands proven to be pure workspace reads do not need mutation admission.
/// Every other native verifier command is opaque process execution: build and
/// test tools routinely write caches, run hooks, or reach external services.
fn native_verifier_intent(stage: &VerifyStage) -> Option<hi_workspace::MutationIntent> {
    let classification = hi_tools::protocol::classify_shell_command(&stage.command);
    if classification.is_proven_read_only() {
        return None;
    }
    Some(hi_workspace::MutationIntent {
        effect_scope: hi_workspace::EffectScope::LiveWriter,
        replay_class: hi_workspace::ReplayClass::NonReplayableExternal,
        dirty_paths: None,
        description: Some(format!(
            "native verifier `{}` ({:?})",
            stage.name, classification.basis
        )),
    })
}

/// RAII fence for native verifier admission. If the verifier future is
/// cancelled before it hands execution evidence to shielded settlement, the
/// live permit is dropped and the controller enters recovery-required instead
/// of remaining silently mutating forever.
struct NativeVerifierAdmission {
    coordination: WorkspaceCoordination,
    durability: Option<std::sync::Arc<dyn crate::WorkspaceDurability>>,
    operation_id: hi_workspace::OperationId,
    intent: hi_workspace::MutationIntent,
    armed: bool,
}

impl NativeVerifierAdmission {
    async fn begin(
        workspace: &VerifyWorkspace<'_>,
        stage: &VerifyStage,
    ) -> anyhow::Result<Option<Self>> {
        let Some(intent) = native_verifier_intent(stage) else {
            return Ok(None);
        };
        let Some(coordination) = workspace.coordination.clone() else {
            // Direct verifier unit tests deliberately omit the live Agent
            // harness. Production turns always install coordination in
            // `run_workspace_repair_verification`.
            return Ok(None);
        };
        coordination
            .begin_intent(workspace.durability.clone(), intent.clone())
            .await?;
        let operation_id = coordination
            .active_parent_operation()
            .ok_or_else(|| anyhow::anyhow!("native verifier admission produced no operation"))?;
        Ok(Some(Self {
            coordination,
            durability: workspace.durability.clone(),
            operation_id,
            intent,
            armed: true,
        }))
    }

    async fn settle(
        mut self,
        stage: &VerifyStage,
        round: u32,
        result: String,
        mut execution: hi_workspace::ExecutionReport,
    ) -> anyhow::Result<()> {
        if matches!(
            self.coordination.binding().authority,
            hi_workspace::WorkspaceAuthority::PipeFs { .. }
        ) {
            let durability = self.durability.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "PipeFS native verifier has no durability backend for transcript staging"
                )
            })?;
            let call_id = format!("native-verifier:{}", self.operation_id);
            let arguments = serde_json::json!({
                "round": round,
                "stage": stage.name,
                "command": stage.command,
            })
            .to_string();
            let record = crate::WorkspaceTranscriptExecution {
                schema_version: crate::WorkspaceTranscriptExecution::SCHEMA_VERSION,
                operation_id: self.operation_id.clone(),
                assistant_content: vec![hi_ai::Content::ToolCall {
                    id: call_id.clone(),
                    name: "native_verify".to_owned(),
                    arguments,
                }],
                calls: vec![crate::WorkspaceTranscriptCall {
                    call_id,
                    name: "native_verify".to_owned(),
                    result,
                }],
                execution: execution.clone(),
            };
            if let Err(stage_error) = durability.stage_workspace_execution(&record) {
                let execution_detail = execution.detail.take();
                execution.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
                execution.content_digest = None;
                execution.detail = Some(match execution_detail {
                    Some(detail) => format!(
                        "{detail}; native verifier transcript staging is ambiguous: {stage_error:#}"
                    ),
                    None => {
                        format!("native verifier transcript staging is ambiguous: {stage_error:#}")
                    }
                });
                let settlement = self
                    .coordination
                    .checkpoint(self.durability.clone(), execution)
                    .await;
                self.armed = false;
                return match settlement {
                    Ok(()) => Err(stage_error)
                        .context("staging native verifier execution for PipeFS settlement"),
                    Err(settlement_error) => Err(anyhow::anyhow!(
                        "native verifier transcript staging failed: {stage_error:#}; recovery settlement also failed: {settlement_error:#}"
                    )),
                };
            }
        }
        self.coordination
            .checkpoint(self.durability.clone(), execution)
            .await?;
        self.armed = false;
        Ok(())
    }
}

impl Drop for NativeVerifierAdmission {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.coordination.abandon_active();
        }
    }
}

fn native_verifier_execution_report(
    stage: &VerifyStage,
    intent: &hi_workspace::MutationIntent,
    execution: Option<&hi_tools::ProcessExecution>,
    changed_paths: Vec<String>,
    infrastructure_error: Option<&str>,
) -> hi_workspace::ExecutionReport {
    let mut disposition = match execution.map(|execution| execution.status) {
        Some(hi_tools::ToolStatus::Succeeded) => hi_workspace::ExecutionDisposition::Succeeded,
        Some(hi_tools::ToolStatus::Cancelled | hi_tools::ToolStatus::TimedOut) => {
            hi_workspace::ExecutionDisposition::Cancelled
        }
        Some(hi_tools::ToolStatus::Failed | hi_tools::ToolStatus::Denied) => {
            hi_workspace::ExecutionDisposition::Failed
        }
        None => hi_workspace::ExecutionDisposition::Indeterminate,
    };
    if disposition == hi_workspace::ExecutionDisposition::Succeeded && !changed_paths.is_empty() {
        disposition = hi_workspace::ExecutionDisposition::Failed;
    }
    let effect_could_have_started = !matches!(
        execution.map(|execution| execution.status),
        Some(hi_tools::ToolStatus::Denied)
    );
    let detail = infrastructure_error.map(str::to_owned).or_else(|| {
        (disposition != hi_workspace::ExecutionDisposition::Succeeded).then(|| {
            if execution.is_some_and(|execution| {
                execution.status == hi_tools::ToolStatus::Succeeded && !changed_paths.is_empty()
            }) {
                format!(
                    "native verifier `{}` (`{}`) modified {} relevant workspace path(s)",
                    stage.name,
                    stage.command,
                    changed_paths.len()
                )
            } else {
                format!(
                    "native verifier `{}` (`{}`) finished with {disposition:?}",
                    stage.name, stage.command
                )
            }
        })
    });
    hi_workspace::ExecutionReport {
        disposition,
        workspace_may_have_changed: effect_could_have_started
            && intent.effect_scope == hi_workspace::EffectScope::LiveWriter,
        external_effect_may_have_occurred: effect_could_have_started
            && intent.replay_class != hi_workspace::ReplayClass::PureWorkspace,
        content_digest: None,
        changed_paths: changed_paths.into_iter().map(Into::into).collect(),
        artifacts: Vec::new(),
        detail,
    }
}

async fn settle_native_verifier_execution(
    admission: Option<NativeVerifierAdmission>,
    stage: &VerifyStage,
    round: u32,
    result: String,
    execution: Option<&hi_tools::ProcessExecution>,
    changed_paths: Vec<String>,
    infrastructure_error: Option<&str>,
) -> anyhow::Result<()> {
    let Some(admission) = admission else {
        return Ok(());
    };
    let report = native_verifier_execution_report(
        stage,
        &admission.intent,
        execution,
        changed_paths,
        infrastructure_error,
    );
    admission.settle(stage, round, result, report).await
}

/// The outcome of one verify check.
#[derive(Debug)]
pub(crate) enum VerifyOutcome {
    /// All stages passed — the turn is done.
    Passed,
    /// No files changed since the turn baseline, so verification was skipped
    /// (a turn that edited nothing can't have introduced a failure). `first`
    /// is true only on the first round, so the caller can surface a one-time
    /// "skipped" status.
    SkippedNoChanges { first: bool },
    /// Only prose/documentation files changed. Running a compile/test pipeline
    /// would add noise but not verify the changed surface.
    SkippedProseOnly { first: bool },
    /// A stage failed; its output is fed back to the model. The caller records
    /// the nudge and loops. Carries the 1-based round number.
    Failed {
        stage: VerifyStage,
        output: String,
        round: u32,
    },
    /// The verifier itself could not run reliably (spawn/runner failure).
    InfrastructureError {
        stage: VerifyStage,
        output: String,
        round: u32,
    },
    /// A validation command rewrote relevant workspace inputs. A pass for that
    /// moving target is not evidence for a stable source revision.
    Unstable {
        stage: VerifyStage,
        changed_files: Vec<String>,
        round: u32,
    },
    /// Verification didn't run: no stages configured, or the round cap was
    /// already reached.
    NotRun,
}

/// Interactive **workspace** repair-loop verifier: cheap compile/typecheck →
/// lint → tests, feeding the first failure back to the model up to `max_rounds`.
///
/// Distinct from:
/// - [`hi_verifier::AttestingVerifier`] — RSI control-plane attestor
/// - [`crate::steering::ReviewRepairMode`] — answer-quality repair inside Steer
///
/// This type never attests; it only steers the agent turn after tools stop.
pub(crate) struct WorkspaceRepairVerifier {
    stages: Vec<VerifyStage>,
    include_affected_packages: bool,
    last_effective_stages: Vec<VerifyStage>,
    executions: VerificationExecutionLog,
    successful_test_stage: bool,
    stage_mutation_counts: std::collections::BTreeMap<String, u32>,
    /// Per-stage failure identity from the previous round — (distinct failure
    /// count, signature) — so repair feedback can say converging vs thrashing.
    previous_failures:
        std::collections::BTreeMap<String, (usize, std::collections::BTreeSet<String>)>,
    max_rounds: u32,
    round: u32,
    pub(crate) timeout_override: Option<std::time::Duration>,
}

/// Historical name — prefer [`WorkspaceRepairVerifier`].
#[allow(dead_code)]
pub(crate) type RepairVerifier = WorkspaceRepairVerifier;

impl WorkspaceRepairVerifier {
    /// Construct from the agent's config. `stages` empty means verification is
    /// off; `max_rounds` caps the retry rounds.
    pub(crate) fn new(stages: Vec<VerifyStage>, max_rounds: u32) -> Self {
        Self {
            stages,
            include_affected_packages: false,
            last_effective_stages: Vec::new(),
            executions: VerificationExecutionLog::default(),
            successful_test_stage: false,
            stage_mutation_counts: std::collections::BTreeMap::new(),
            previous_failures: std::collections::BTreeMap::new(),
            max_rounds,
            round: 0,
            timeout_override: None,
        }
    }

    /// Construct an automatically detected repair verifier. Unlike an explicit
    /// pipeline, automatic verification may prepend checks for changed nested
    /// package roots before the workspace-root stages.
    pub(crate) fn automatic(stages: Vec<VerifyStage>, max_rounds: u32) -> Self {
        let mut verifier = Self::new(stages, max_rounds);
        verifier.include_affected_packages = true;
        verifier
    }

    /// Whether any verification stage is configured.
    #[allow(dead_code)]
    pub(crate) fn is_on(&self) -> bool {
        !self.stages.is_empty() || self.include_affected_packages
    }

    /// The current round (0 before any verify run, 1-based after).
    #[allow(dead_code)]
    pub(crate) fn round(&self) -> u32 {
        self.round
    }

    /// Independent review gets one repair cycle even when deterministic
    /// verification's ordinary repair budget was zero. That repair must be
    /// followed by a fresh check of the resulting revision.
    pub(crate) fn allow_review_revalidation(&mut self) {
        if self.max_rounds != crate::UNLIMITED_REPAIR_CYCLES {
            self.max_rounds = self
                .max_rounds
                .saturating_add(1)
                .min(crate::UNLIMITED_REPAIR_CYCLES - 1);
        }
    }

    pub(crate) fn stages_summary(&self) -> Option<String> {
        let stages = if self.last_effective_stages.is_empty() {
            &self.stages
        } else {
            &self.last_effective_stages
        };
        (!stages.is_empty()).then(|| {
            stages
                .iter()
                .map(|stage| format!("{}: {}", stage.name, stage.command))
                .collect::<Vec<_>>()
                .join(" -> ")
        })
    }

    /// Executed stage evidence in chronological order across all repair
    /// rounds. Skipped checks do not create synthetic execution records.
    pub(crate) fn executions(&self) -> &[VerificationExecution] {
        self.executions.as_slice()
    }

    pub(crate) fn executions_dropped(&self) -> u64 {
        self.executions.dropped()
    }

    pub(crate) fn execution_count(&self) -> u64 {
        self.executions.total()
    }

    pub(crate) fn successful_test_stage(&self) -> bool {
        self.successful_test_stage
    }

    fn record_execution(&mut self, execution: VerificationExecution) {
        if execution.status == hi_tools::ToolStatus::Succeeded
            && (execution.name.contains("test")
                || execution.command.contains("test")
                || execution.command.contains("pytest"))
        {
            self.successful_test_stage = true;
        }
        self.executions.push(execution);
    }

    /// Run one verification check against the current workspace snapshot,
    /// compared to the turn baseline. Gates on file changes: if nothing
    /// changed, returns [`VerifyOutcome::SkippedNoChanges`] (and does NOT
    /// consume a round). Otherwise runs the stages in order and returns the
    /// first failure, or [`VerifyOutcome::Passed`].
    ///
    /// `snapshot_cache` is invalidated-on-mutation cache the verifier reads
    /// through; the caller passes the turn baseline separately.
    pub(crate) async fn check(
        &mut self,
        workspace: &VerifyWorkspace<'_>,
        turn_snapshot: &Snapshot,
        snapshot_cache: &mut SnapshotCache,
        ledger: Option<std::sync::Arc<std::sync::Mutex<crate::change_ledger::ChangeLedger>>>,
        ui: &mut dyn Ui,
    ) -> VerifyOutcome {
        if (self.stages.is_empty() && !self.include_affected_packages)
            || crate::config::repair_limit_reached(self.max_rounds, self.round)
        {
            return VerifyOutcome::NotRun;
        }
        let changed_files = if let Some(changed_files) = workspace.known_changed_files {
            changed_files.to_vec()
        } else {
            let current = match snapshot_cache.get(workspace.root).await {
                Ok(current) => current,
                Err(error) => {
                    self.round = self.round.saturating_add(1);
                    let round = self.round;
                    let stage =
                        self.stages.first().cloned().unwrap_or_else(|| {
                            VerifyStage::new("auto", "affected package discovery")
                        });
                    self.record_execution(VerificationExecution::infrastructure_failure(
                        round, &stage,
                    ));
                    return VerifyOutcome::InfrastructureError {
                        stage,
                        output: format!("workspace snapshot infrastructure failed: {error:#}"),
                        round,
                    };
                }
            };
            changed_files_between(turn_snapshot, &current)
        };
        if changed_files.is_empty() && !workspace.mutation_seen {
            let first = self.round == 0;
            return VerifyOutcome::SkippedNoChanges { first };
        }
        // Automatic code-oriented pipelines are not useful evidence for a
        // documentation-only change. An explicit pipeline is different: the
        // user may have supplied markdownlint, a docs builder, or any other
        // acceptance command, and `--verify` must run exactly as configured.
        if self.include_affected_packages
            && changed_files.iter().all(|path| is_prose_only_path(path))
        {
            let first = self.round == 0;
            return VerifyOutcome::SkippedProseOnly { first };
        }
        // Package discovery reads manifests and, for Python, may walk an
        // entire package to determine whether pytest would collect tests.
        // Keep that filesystem work off the async/UI executor.
        let discovery_root = workspace.root.to_path_buf();
        let discovery_changed_files = changed_files.clone();
        let discovery_configured = self.stages.clone();
        let discovery_include_affected = self.include_affected_packages;
        let mut stages = match tokio::task::spawn_blocking(move || {
            effective_stages(
                &discovery_root,
                &discovery_changed_files,
                &discovery_configured,
                discovery_include_affected,
            )
        })
        .await
        {
            Ok(stages) => stages,
            Err(error) => {
                self.round = self.round.saturating_add(1);
                let round = self.round;
                let stage = VerifyStage::new("auto", "affected package discovery");
                self.record_execution(VerificationExecution::infrastructure_failure(round, &stage));
                return VerifyOutcome::InfrastructureError {
                    stage,
                    output: format!("affected package discovery worker failed: {error}"),
                    round,
                };
            }
        };
        let empty_set = std::collections::BTreeSet::new();
        let skip_checks = workspace.skip_affected_checks.unwrap_or(&empty_set);
        let skip_tests = workspace.skip_affected_tests.unwrap_or(&empty_set);
        let before_filter = stages.len();
        stages.retain(|stage| !should_skip_affected_stage(stage, skip_checks, skip_tests));
        let skipped = before_filter.saturating_sub(stages.len());
        if skipped > 0 {
            ui.status(&format!(
                "verification · skipping {skipped} mid-turn-sealed affected stage(s)"
            ));
        }
        self.last_effective_stages = stages.clone();
        if stages.is_empty() {
            // Everything was either absent or already sealed mid-turn. If we
            // filtered at least one stage away, treat as Passed — the work was
            // already proven at this revision. Otherwise nothing to run.
            return if skipped > 0 {
                VerifyOutcome::Passed
            } else {
                VerifyOutcome::NotRun
            };
        }
        self.round = self.round.saturating_add(1);
        let round = self.round;
        let max_rounds = self.max_rounds;

        // LSP fast path: if enabled, check diagnostics on changed files before
        // running any shell stages. This catches type errors in ~1s instead of
        // a full `cargo test`/build, and gives line-level errors.
        if workspace.lsp.is_enabled().await {
            let mut lsp_errors = Vec::new();
            let mut lsp_failed = false;
            let mut lsp_checked = false;
            // Only files a server actually owns. A changed `Cargo.toml` or
            // `Makefile` has no language server; asking anyway makes the
            // manager fall back to the project language and report the whole
            // file as syntactically invalid.
            let paths = changed_files
                .iter()
                .map(std::path::PathBuf::from)
                .filter(|path| hi_lsp::detect_language(path).is_some())
                .collect::<Vec<_>>();
            for (path, state) in workspace.lsp.diagnostics_batch(&paths).await {
                match state {
                    hi_lsp::DiagnosticState::ConfirmedClean { .. } => lsp_checked = true,
                    hi_lsp::DiagnosticState::DiagnosticsPresent { diagnostics, .. } => {
                        lsp_checked = true;
                        for d in diagnostics {
                            if d.severity == "error" {
                                lsp_errors.push(format!(
                                    "{}:{}:{}: {}",
                                    path.display(),
                                    d.line + 1,
                                    d.col + 1,
                                    d.message
                                ));
                            }
                        }
                    }
                    hi_lsp::DiagnosticState::Failed { .. } => lsp_failed = true,
                    hi_lsp::DiagnosticState::Unavailable { .. } => {}
                }
            }
            if !lsp_errors.is_empty() {
                self.record_execution(VerificationExecution::lsp(
                    round,
                    hi_tools::ToolStatus::Failed,
                ));
                let output = format!(
                    "LSP diagnostics ({} error(s)):\n{}",
                    lsp_errors.len(),
                    lsp_errors.join("\n")
                );
                return VerifyOutcome::Failed {
                    stage: VerifyStage::new("lsp", "diagnostics"),
                    output,
                    round,
                };
            }
            if lsp_failed {
                self.record_execution(VerificationExecution::lsp(
                    round,
                    hi_tools::ToolStatus::Failed,
                ));
            } else if lsp_checked {
                self.record_execution(VerificationExecution::lsp(
                    round,
                    hi_tools::ToolStatus::Succeeded,
                ));
            }
        }

        for stage in &stages {
            ui.status(&format!(
                "verifying ({round}/{}) · {}: {}",
                crate::config::repair_limit_label(max_rounds),
                stage.name,
                stage.command
            ));
            // Stage-mutation detection: prefer the content ledger (already
            // reconciled before verify; the post-stage reconcile is cheap when
            // nothing changed). Fall back to snapshot walks in unit tests that
            // have no ledger.
            let stage_ledger_revision = ledger.as_ref().map(|ledger| {
                ledger
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .revision()
            });
            let before_stage = if ledger.is_none() {
                match workspace_snapshot_meta(workspace.root).await {
                    Ok(snapshot) => Some(snapshot),
                    Err(error) => {
                        self.record_execution(VerificationExecution::infrastructure_failure(
                            round, stage,
                        ));
                        return VerifyOutcome::InfrastructureError {
                            stage: stage.clone(),
                            output: format!("pre-stage workspace snapshot failed: {error:#}"),
                            round,
                        };
                    }
                }
            } else {
                None
            };
            let admission = match NativeVerifierAdmission::begin(workspace, stage).await {
                Ok(admission) => admission,
                Err(error) => {
                    self.record_execution(VerificationExecution::infrastructure_failure(
                        round, stage,
                    ));
                    return VerifyOutcome::InfrastructureError {
                        stage: stage.clone(),
                        output: format!(
                            "native verifier workspace admission failed before execution: {error:#}"
                        ),
                        round,
                    };
                }
            };
            let verification_timeout = self.timeout_override.or_else(hi_tools::check_timeout);
            let mut execution = match run_check_for_workspace(
                workspace,
                &stage.command,
                verification_timeout,
            )
            .await
            {
                Ok(execution) => execution,
                Err(error) => {
                    self.record_execution(VerificationExecution::infrastructure_failure(
                        round, stage,
                    ));
                    let output = format!("verification process infrastructure failed: {error:#}");
                    if let Err(settlement_error) = settle_native_verifier_execution(
                        admission,
                        stage,
                        round,
                        output.clone(),
                        None,
                        Vec::new(),
                        Some(&output),
                    )
                    .await
                    {
                        return VerifyOutcome::InfrastructureError {
                            stage: stage.clone(),
                            output: format!(
                                "{output}; workspace settlement also failed: {settlement_error:#}"
                            ),
                            round,
                        };
                    }
                    return VerifyOutcome::InfrastructureError {
                        stage: stage.clone(),
                        output,
                        round,
                    };
                }
            };
            self.record_execution(VerificationExecution::shell(round, stage, &execution));
            let mut exact_results = vec![execution.model_content()];
            if execution.status == hi_tools::ToolStatus::TimedOut && self.timeout_override.is_none()
            {
                // One doubled-budget retry distinguishes a cold first build
                // from a genuinely oversized stage before ending the turn as
                // unverified infrastructure.
                ui.status(&format!(
                    "verify stage `{}` timed out — one retry with a doubled budget (cold build?)",
                    stage.name
                ));
                execution = match run_check_for_workspace_with_timeout(
                    workspace,
                    &stage.command,
                    verification_timeout
                        .expect("TimedOut requires an explicitly configured verification timeout")
                        .saturating_mul(2),
                )
                .await
                {
                    Ok(execution) => execution,
                    Err(error) => {
                        self.record_execution(VerificationExecution::infrastructure_failure(
                            round, stage,
                        ));
                        let output =
                            format!("verification process infrastructure failed: {error:#}");
                        exact_results.push(output.clone());
                        if let Err(settlement_error) = settle_native_verifier_execution(
                            admission,
                            stage,
                            round,
                            exact_results.join("\n\n--- retry ---\n\n"),
                            None,
                            Vec::new(),
                            Some(&output),
                        )
                        .await
                        {
                            return VerifyOutcome::InfrastructureError {
                                stage: stage.clone(),
                                output: format!(
                                    "{output}; workspace settlement also failed: {settlement_error:#}"
                                ),
                                round,
                            };
                        }
                        return VerifyOutcome::InfrastructureError {
                            stage: stage.clone(),
                            output,
                            round,
                        };
                    }
                };
                self.record_execution(VerificationExecution::shell(round, stage, &execution));
                exact_results.push(execution.model_content());
            }
            let all_stage_changes = if let Some(ledger) = ledger.as_ref() {
                // Ledger path: reconcile (cheap via dir-stamp fast path when
                // nothing changed) and diff against the pre-stage revision.
                // The reconcile still walks the filesystem on a cold stamp, so
                // run it on the blocking pool instead of freezing the drive
                // loop while a verification stage is finishing.
                let before_revision = stage_ledger_revision.expect("set when ledger is Some");
                let ledger = std::sync::Arc::clone(ledger);
                let reconciled = tokio::task::spawn_blocking(move || {
                    let mut ledger = ledger
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let changes = ledger.reconcile()?;
                    let paths = if ledger.revision() == before_revision {
                        Vec::new()
                    } else {
                        changes
                            .into_iter()
                            .map(|change| change.path)
                            .collect::<Vec<_>>()
                    };
                    Ok::<Vec<String>, anyhow::Error>(paths)
                })
                .await;
                match reconciled {
                    Ok(Ok(paths)) => paths,
                    Ok(Err(error)) => {
                        self.record_execution(VerificationExecution::infrastructure_failure(
                            round, stage,
                        ));
                        let output = format!("post-stage ledger reconcile failed: {error:#}");
                        if let Err(settlement_error) = settle_native_verifier_execution(
                            admission,
                            stage,
                            round,
                            format!(
                                "{}\n\n--- reconcile ---\n\n{output}",
                                exact_results.join("\n\n--- retry ---\n\n")
                            ),
                            None,
                            Vec::new(),
                            Some(&output),
                        )
                        .await
                        {
                            return VerifyOutcome::InfrastructureError {
                                stage: stage.clone(),
                                output: format!(
                                    "{output}; workspace settlement also failed: {settlement_error:#}"
                                ),
                                round,
                            };
                        }
                        return VerifyOutcome::InfrastructureError {
                            stage: stage.clone(),
                            output,
                            round,
                        };
                    }
                    Err(error) => {
                        self.record_execution(VerificationExecution::infrastructure_failure(
                            round, stage,
                        ));
                        let output = format!("post-stage ledger worker failed: {error}");
                        if let Err(settlement_error) = settle_native_verifier_execution(
                            admission,
                            stage,
                            round,
                            format!(
                                "{}\n\n--- reconcile ---\n\n{output}",
                                exact_results.join("\n\n--- retry ---\n\n")
                            ),
                            None,
                            Vec::new(),
                            Some(&output),
                        )
                        .await
                        {
                            return VerifyOutcome::InfrastructureError {
                                stage: stage.clone(),
                                output: format!(
                                    "{output}; workspace settlement also failed: {settlement_error:#}"
                                ),
                                round,
                            };
                        }
                        return VerifyOutcome::InfrastructureError {
                            stage: stage.clone(),
                            output,
                            round,
                        };
                    }
                }
            } else {
                let after_stage = match workspace_snapshot_meta(workspace.root).await {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        self.record_execution(VerificationExecution::infrastructure_failure(
                            round, stage,
                        ));
                        let output = format!("post-stage workspace snapshot failed: {error:#}");
                        if let Err(settlement_error) = settle_native_verifier_execution(
                            admission,
                            stage,
                            round,
                            format!(
                                "{}\n\n--- reconcile ---\n\n{output}",
                                exact_results.join("\n\n--- retry ---\n\n")
                            ),
                            None,
                            Vec::new(),
                            Some(&output),
                        )
                        .await
                        {
                            return VerifyOutcome::InfrastructureError {
                                stage: stage.clone(),
                                output: format!(
                                    "{output}; workspace settlement also failed: {settlement_error:#}"
                                ),
                                round,
                            };
                        }
                        return VerifyOutcome::InfrastructureError {
                            stage: stage.clone(),
                            output,
                            round,
                        };
                    }
                };
                changed_files_between(
                    before_stage.as_ref().expect("set when ledger is None"),
                    &after_stage,
                )
            };
            let stage_changes = all_stage_changes
                .iter()
                .filter(|path| verification_relevant_path(path))
                .cloned()
                .collect::<Vec<_>>();
            if let Err(error) = settle_native_verifier_execution(
                admission,
                stage,
                round,
                exact_results.join("\n\n--- retry ---\n\n"),
                Some(&execution),
                stage_changes.clone(),
                None,
            )
            .await
            {
                self.record_execution(VerificationExecution::infrastructure_failure(round, stage));
                return VerifyOutcome::InfrastructureError {
                    stage: stage.clone(),
                    output: format!(
                        "native verifier completed but workspace settlement failed: {error:#}"
                    ),
                    round,
                };
            }
            if !stage_changes.is_empty() {
                snapshot_cache.invalidate();
                let mutation_count = self
                    .stage_mutation_counts
                    .entry(format!("{}\0{}", stage.name, stage.command))
                    .or_default();
                *mutation_count = mutation_count.saturating_add(1);
                if *mutation_count >= 2 {
                    return VerifyOutcome::Unstable {
                        stage: stage.clone(),
                        changed_files: stage_changes,
                        round,
                    };
                }
                return VerifyOutcome::Failed {
                    stage: stage.clone(),
                    output: format!(
                        "Verification stage modified relevant source files, so its result is invalid for a stable revision. Inspect or revert these changes before retrying:\n- {}\n\nStage output:\n{}",
                        stage_changes.join("\n- "),
                        execution.model_content(),
                    ),
                    round,
                };
            }
            // A stage that ran out of time reports nothing about correctness —
            // the command was killed mid-run, so its partial output is
            // meaningless as evidence either way. Treating it as a normal
            // failure is actively harmful: the model is handed a wall of
            // *passing* test results under a "stage failed" headline (the
            // timeout marker is the last line of a multi-KB blob, and
            // attribution happily nominates a passing assertion as the "likely
            // cause"), so it cannot tell a slow suite from a broken one. Worse,
            // the final round re-runs the very same command against an isolated
            // pre-turn checkpoint to attribute the failure, spending the whole
            // timeout budget a second time.
            //
            // Route it to the infrastructure path instead: verify becomes
            // "unknown" rather than "failed", the turn ends instead of burning
            // repair rounds re-running a command that cannot finish, and the
            // status line names the real problem.
            if execution.status == hi_tools::ToolStatus::TimedOut {
                self.record_execution(VerificationExecution::infrastructure_failure(round, stage));
                return VerifyOutcome::InfrastructureError {
                    stage: stage.clone(),
                    output: format!(
                        "stage `{}` (`{}`) exceeded its configured time budget and was killed, so this revision is unverified — this is not a code failure. Raise the verification timeout or narrow the stage to something that fits the budget (configured {}s).",
                        stage.name,
                        stage.command,
                        verification_timeout
                            .expect(
                                "TimedOut requires an explicitly configured verification timeout"
                            )
                            .as_secs(),
                    ),
                    round,
                };
            }
            if execution.status != hi_tools::ToolStatus::Succeeded {
                let mut output = execution.model_content();
                // Restructure the raw evidence: distinct root-cause diagnostics
                // with their source spans first, plus a converging-vs-thrashing
                // note against the previous round's failure set for this stage.
                let stage_key = format!("{}\0{}", stage.name, stage.command);
                if let Some(digest) = crate::verify_digest::digest_failure(workspace.root, &output)
                {
                    let note = crate::verify_digest::convergence_note(
                        self.previous_failures.get(&stage_key),
                        &digest,
                    );
                    self.previous_failures
                        .insert(stage_key, (digest.failure_count, digest.signature.clone()));
                    output = format!("{}{note}\nFull stage output:\n{output}", digest.text);
                }
                if crate::config::repair_limit_reached(max_rounds, round) {
                    ui.status(&format!(
                        "attributing final verification failure · {}",
                        stage.name
                    ));
                    let Some(checkpoint) = workspace.pre_turn_checkpoint else {
                        output.push_str(
                            "\n\nPre-turn attribution unavailable: this turn has no restorable pre-turn checkpoint.",
                        );
                        return VerifyOutcome::Failed {
                            stage: stage.clone(),
                            output,
                            round,
                        };
                    };
                    let command = stage.command.clone();
                    let sandbox_policy = workspace
                        .process_runner
                        .map(hi_tools::ProcessRunner::sandbox_policy);
                    let baseline = hi_tools::checkpoint::with_isolated_checkpoint(
                        workspace.root,
                        checkpoint,
                        workspace.state_root,
                        move |isolated| async move {
                            let runner = verification_runner(&isolated, sandbox_policy)?;
                            hi_tools::run_check_in_with_runner_maybe_timeout(
                                &runner,
                                &command,
                                verification_timeout,
                            )
                            .await
                        },
                    )
                    .await;
                    let baseline = match baseline {
                        Ok(baseline) => baseline,
                        Err(error) => {
                            return VerifyOutcome::InfrastructureError {
                                stage: stage.clone(),
                                output: format!(
                                    "verification failed, then isolated pre-turn attribution could not run: {error:#}"
                                ),
                                round,
                            };
                        }
                    };
                    match baseline.status {
                        hi_tools::ToolStatus::Succeeded => output.push_str(
                            "\n\nPre-turn attribution: this stage passed in an isolated pre-turn workspace, so the current failure was not present at the turn baseline.",
                        ),
                        hi_tools::ToolStatus::Failed
                            if baseline_failure_is_infrastructure(&baseline) =>
                        {
                            output.push_str(
                                "\n\nPre-turn attribution was inconclusive: the isolated baseline command was blocked by the execution environment, so this is not evidence that the project already failed before the turn. Baseline output:\n",
                            );
                            output.push_str(&bounded_baseline_output(&baseline.model_content()));
                        }
                        hi_tools::ToolStatus::Failed => {
                            output.push_str(
                                "\n\nPre-turn attribution: this stage also failed in an isolated pre-turn workspace; the project already failed this verification stage before the turn. Baseline output:\n",
                            );
                            output.push_str(&bounded_baseline_output(&baseline.model_content()));
                        }
                        hi_tools::ToolStatus::TimedOut => output.push_str(
                            "\n\nPre-turn attribution was inconclusive: the isolated baseline command timed out.",
                        ),
                        hi_tools::ToolStatus::Cancelled => output.push_str(
                            "\n\nPre-turn attribution was inconclusive: the isolated baseline command was cancelled.",
                        ),
                        hi_tools::ToolStatus::Denied => output.push_str(
                            "\n\nPre-turn attribution was inconclusive: the isolated baseline command was denied.",
                        ),
                    }
                }
                return VerifyOutcome::Failed {
                    stage: stage.clone(),
                    output,
                    round,
                };
            }
            // A pass resets the stage's failure history: a later re-failure is
            // a fresh problem, not a continuation of the old one.
            self.previous_failures
                .remove(&format!("{}\0{}", stage.name, stage.command));
        }
        VerifyOutcome::Passed
    }
}

fn effective_stages(
    root: &std::path::Path,
    changed_files: &[String],
    configured: &[VerifyStage],
    include_affected_packages: bool,
) -> Vec<VerifyStage> {
    let mut stages = if include_affected_packages {
        affected_package_stages(root, changed_files)
    } else {
        Vec::new()
    };
    // When the change already has package-local Cargo test coverage, the
    // detected whole-workspace `cargo test` *and* `cargo check` are redundant
    // on the end-of-turn path: package `cargo test` already compiles that
    // crate, and the workspace-wide stages' cost grows with the project rather
    // than the edit (measured: 24-crate `cargo test` 811s vs package-local
    // minutes). Cross-crate breakage in direct dependents is covered by the
    // `affected-dependent-check:` compile stages; deeper transitive breakage
    // is left to CI / an explicit `/verify` stage.
    //
    // This only applies to the auto-detected pipeline — explicitly configured
    // stages (include_affected_packages = false) are the user's choice and are
    // always run as written.
    let has_affected_cargo_tests = stages.iter().any(|stage| {
        stage.name.starts_with("affected-test:") && is_package_local_cargo_test(&stage.command)
    });
    let has_affected_tests = stages
        .iter()
        .any(|stage| stage.name.starts_with("affected-test:"));
    for stage in configured {
        if has_affected_cargo_tests
            && (is_whole_workspace_cargo_test(&stage.command)
                || is_whole_workspace_cargo_check(&stage.command))
        {
            continue;
        }
        // Non-Cargo ecosystems: still drop whole-workspace cargo test when any
        // package-local test exists (JS/Go/Python), matching prior behavior.
        if has_affected_tests && is_whole_workspace_cargo_test(&stage.command) {
            continue;
        }
        if !stages
            .iter()
            .any(|affected| affected.command == stage.command)
        {
            stages.push(stage.clone());
        }
    }
    drop_checks_superseded_by_package_tests(&mut stages);
    stages
}

/// Drop package-local compile/build stages when a package-local test for the
/// same label will run. `cargo test` / `go test` already compile the package, so
/// a preceding check/build is pure latency on the end-of-turn critical path.
/// Typecheck/lint stages are kept — tests do not replace them.
fn drop_checks_superseded_by_package_tests(stages: &mut Vec<VerifyStage>) {
    let test_labels = stages
        .iter()
        .filter_map(|stage| stage.name.strip_prefix("affected-test:"))
        .map(str::to_string)
        .collect::<std::collections::BTreeSet<_>>();
    if test_labels.is_empty() {
        return;
    }
    stages.retain(|stage| {
        let Some(label) = stage
            .name
            .strip_prefix("affected-check:")
            .or_else(|| stage.name.strip_prefix("affected-build:"))
        else {
            return true;
        };
        !test_labels.contains(label)
    });
}

/// Whether `command` is a `cargo test` run that is not narrowed to a package.
///
/// Deliberately conservative: any package selector (`-p`, `--package`,
/// `--manifest-path`) or an explicit `--workspace`-with-filter form means the
/// caller has already scoped it, and anything that isn't recognisably a plain
/// `cargo test` is left alone.
fn is_whole_workspace_cargo_test(command: &str) -> bool {
    let command = command.trim();
    let Some(rest) = command.strip_prefix("cargo test") else {
        return false;
    };
    // `cargo testfoo` is not `cargo test`.
    if !rest.is_empty() && !rest.starts_with([' ', '\t']) {
        return false;
    }
    // A shell chain (`cargo test && …`) is doing more than one thing; leave it.
    if rest.contains("&&") || rest.contains(';') || rest.contains('|') {
        return false;
    }
    !["-p ", "--package", "--manifest-path", "--bin ", "--test "]
        .iter()
        .any(|selector| rest.contains(selector))
}

/// Whether `command` is an unscoped `cargo check` (no package selector).
fn is_whole_workspace_cargo_check(command: &str) -> bool {
    let command = command.trim();
    let Some(rest) = command.strip_prefix("cargo check") else {
        return false;
    };
    if !rest.is_empty() && !rest.starts_with([' ', '\t']) {
        return false;
    }
    if rest.contains("&&") || rest.contains(';') || rest.contains('|') {
        return false;
    }
    !["-p ", "--package", "--manifest-path", "--bin "]
        .iter()
        .any(|selector| rest.contains(selector))
}

/// Package-local Cargo test stage produced by [`affected_cargo_stages`].
fn is_package_local_cargo_test(command: &str) -> bool {
    let command = command.trim();
    command.starts_with("cargo test") && command.contains("--manifest-path")
}

/// Mid-turn fast feedback already ran package checks/tests for these labels.
/// Matching auto-generated affected stages are redundant at the same ledger
/// revision. Root pipeline stages are never skipped.
///
/// Check-namespace seals cover: `affected-check:`, `affected-typecheck:`,
/// `affected-build:`, `affected-lint:`. Test-namespace: `affected-test:`.
fn should_skip_affected_stage(
    stage: &VerifyStage,
    skip_checks: &std::collections::BTreeSet<String>,
    skip_tests: &std::collections::BTreeSet<String>,
) -> bool {
    const CHECK_PREFIXES: &[&str] = &[
        "affected-check:",
        "affected-typecheck:",
        "affected-build:",
        "affected-lint:",
        "affected-dependent-check:",
    ];
    for prefix in CHECK_PREFIXES {
        if let Some(label) = stage.name.strip_prefix(prefix) {
            return skip_checks.contains(label);
        }
    }
    if let Some(label) = stage.name.strip_prefix("affected-test:") {
        return skip_tests.contains(label);
    }
    false
}

/// Package-local checks run before the automatically detected root pipeline.
/// Ecosystems and package paths have a fixed order so the same change set
/// always resolves to the same stage list.
fn affected_package_stages(root: &std::path::Path, changed_files: &[String]) -> Vec<VerifyStage> {
    let mut stages = affected_cargo_stages(root, changed_files);
    stages.extend(affected_javascript_stages(root, changed_files));
    stages.extend(affected_go_stages(root, changed_files));
    stages.extend(affected_python_stages(root, changed_files));
    stages
}

/// A manifest path selects the affected Cargo package even when a containing
/// workspace has a different `default-members` set. Root-package changes need
/// no extra stage because the root pipeline already covers them.
fn affected_cargo_stages(root: &std::path::Path, changed_files: &[String]) -> Vec<VerifyStage> {
    let packages = hi_tools::affected_cargo_package_dirs(root, changed_files);
    let mut stages: Vec<VerifyStage> = packages
        .iter()
        .flat_map(|label| {
            let manifest = shell_quote(&format!("{label}/Cargo.toml"));
            [
                VerifyStage::new(
                    format!("affected-check:{label}"),
                    format!("cargo check --quiet --manifest-path {manifest}"),
                ),
                VerifyStage::new(
                    format!("affected-test:{label}"),
                    format!("cargo test --quiet --manifest-path {manifest}"),
                ),
            ]
        })
        .collect();
    // Package-local tests cannot see an API break in the crates that consume
    // the changed crate. Direct dependents get a compile check — cost stays
    // proportional to the edit's blast radius, not the workspace size; past a
    // small fan-out one whole-workspace check is cheaper than many scoped ones.
    const MAX_DEPENDENT_CHECKS: usize = 4;
    // The root package is deliberately omitted from `affected_cargo_package_dirs`
    // because the configured root pipeline covers it. Use it as the dependency
    // graph seed nevertheless, or root-package API changes will skip all member
    // consumers that path-depend on the root package.
    let mut dependency_seeds = packages.clone();
    if dependency_seeds.is_empty()
        && root.join("Cargo.toml").is_file()
        && !hi_tools::rust_source_paths(changed_files.iter()).is_empty()
    {
        dependency_seeds.insert(".".into());
    }
    let dependents = hi_tools::cargo_dependent_package_dirs(root, &dependency_seeds);
    if dependents.len() > MAX_DEPENDENT_CHECKS {
        stages.push(VerifyStage::new(
            "affected-dependent-check:workspace",
            "cargo check --quiet --workspace",
        ));
    } else {
        for label in dependents {
            let manifest = shell_quote(&format!("{label}/Cargo.toml"));
            stages.push(VerifyStage::new(
                format!("affected-dependent-check:{label}"),
                format!("cargo check --quiet --manifest-path {manifest}"),
            ));
        }
    }
    stages
}

fn affected_javascript_stages(
    root: &std::path::Path,
    changed_files: &[String],
) -> Vec<VerifyStage> {
    hi_tools::affected_package_dirs(root, changed_files, |directory| {
        directory.join("package.json").is_file()
    })
    .into_iter()
    .flat_map(|label| {
        let package_root = root.join(&label);
        let package_json = package_root.join("package.json");
        let has_typecheck_script = std::fs::read_to_string(package_json)
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .and_then(|manifest| manifest.get("scripts").cloned())
            .and_then(|scripts| scripts.get("typecheck").cloned())
            .is_some();
        let quoted = shell_quote(&label);
        let mut stages = Vec::new();
        if has_typecheck_script {
            stages.push(VerifyStage::new(
                format!("affected-typecheck:{label}"),
                format!("npm --prefix {quoted} run typecheck --silent"),
            ));
        } else if package_root.join("tsconfig.json").is_file() {
            stages.push(VerifyStage::new(
                format!("affected-typecheck:{label}"),
                format!("npm --prefix {quoted} exec -- tsc --noEmit"),
            ));
        }
        // Match the root JavaScript pipeline's conservative behavior: a
        // missing or broken test script is a verification failure, not a pass.
        stages.push(VerifyStage::new(
            format!("affected-test:{label}"),
            format!("npm --prefix {quoted} test --silent"),
        ));
        stages
    })
    .collect()
}

fn affected_go_stages(root: &std::path::Path, changed_files: &[String]) -> Vec<VerifyStage> {
    hi_tools::affected_package_dirs(root, changed_files, |directory| {
        directory.join("go.mod").is_file()
    })
    .into_iter()
    .flat_map(|label| {
        let quoted = shell_quote(&label);
        [
            VerifyStage::new(
                format!("affected-build:{label}"),
                format!("go -C {quoted} build ./..."),
            ),
            VerifyStage::new(
                format!("affected-test:{label}"),
                format!("go -C {quoted} test ./..."),
            ),
        ]
    })
    .collect()
}

/// Whether a Python package directory tree contains any files pytest would
/// collect by default (`test_*.py` or `*_test.py`). A package with a
/// `pyproject.toml` but no tests would otherwise make pytest exit 5 ("no
/// tests collected"), which reads as a verification failure.
pub(crate) fn has_python_tests(package_root: &std::path::Path) -> bool {
    fn is_test_file(name: &str) -> bool {
        (name.starts_with("test_") || name.ends_with("_test.py")) && name.ends_with(".py")
    }
    fn walk(dir: &std::path::Path) -> bool {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };
            if file_type.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if matches!(
                    name,
                    "__pycache__"
                        | ".venv"
                        | ".cargo-home"
                        | "venv"
                        | "node_modules"
                        | "dist"
                        | "build"
                        | ".git"
                        | ".hg"
                        | ".svn"
                        | ".jj"
                        | ".tox"
                        | ".mypy_cache"
                        | ".pytest_cache"
                        | ".ruff_cache"
                ) {
                    continue;
                }
                if walk(&path) {
                    return true;
                }
            } else if file_type.is_file()
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
                && is_test_file(name)
            {
                return true;
            }
        }
        false
    }
    walk(package_root)
}

fn affected_python_stages(root: &std::path::Path, changed_files: &[String]) -> Vec<VerifyStage> {
    hi_tools::affected_package_dirs(root, changed_files, hi_tools::is_python_package_root)
        .into_iter()
        .flat_map(|label| {
            let package_root = root.join(&label);
            let pyproject_has_ruff = std::fs::read_to_string(package_root.join("pyproject.toml"))
                .ok()
                .is_some_and(|text| {
                    text.lines()
                        .any(|line| line.trim_start().starts_with("[tool.ruff"))
                });
            let quoted = shell_quote(&label);
            let mut stages = Vec::new();
            if package_root.join("ruff.toml").is_file()
                || package_root.join(".ruff.toml").is_file()
                || pyproject_has_ruff
            {
                stages.push(VerifyStage::new(
                    format!("affected-lint:{label}"),
                    format!("ruff check {quoted}"),
                ));
            }
            if has_python_tests(&package_root) {
                stages.push(VerifyStage::new(
                    format!("affected-test:{label}"),
                    format!("pytest -q {quoted}"),
                ));
            }
            stages
        })
        .collect()
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn bounded_baseline_output(output: &str) -> String {
    const MAX_CHARS: usize = 8_000;
    if output.chars().count() <= MAX_CHARS {
        return output.to_string();
    }
    let mut bounded = output.chars().take(MAX_CHARS).collect::<String>();
    bounded.push_str("\n… [baseline output truncated]");
    bounded
}

fn baseline_failure_is_infrastructure(outcome: &hi_tools::ProcessExecution) -> bool {
    let text = outcome.model_content().to_ascii_lowercase();
    outcome
        .outcome
        .exit_code
        .is_some_and(|code| matches!(code, 126 | 127))
        || [
            "operation not permitted",
            "permission denied",
            "sandbox denied",
            "failed to spawn",
            "could not execute process",
        ]
        .iter()
        .any(|marker| text.contains(marker))
}

fn verification_relevant_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/").to_ascii_lowercase();
    // Test/build caches are written BY the verification stages themselves, so
    // counting them as workspace mutation makes a stage look self-modifying:
    // two rounds of that is reported as unstable verification and can abort a
    // healthy turn with plan steps still pending (observed: `.pytest_cache`
    // churn ended a run 134s in, before its server step ever ran).
    !normalized.split('/').any(|part| {
        matches!(
            part,
            "__pycache__"
                | ".hi"
                | ".pytest_cache"
                | ".mypy_cache"
                | ".ruff_cache"
                | ".tox"
                | ".gradle"
                | ".cargo-home"
                | "node_modules"
        ) || part.ends_with(".egg-info")
    }) && !matches!(
        normalized.rsplit('.').next(),
        Some("pyc" | "pyo" | "class" | "o" | "obj")
    ) && !is_prose_only_path(&normalized)
}

/// Tailor the failure guidance to the stage kind: test failures imply a rule
/// to infer; compile/lint errors point at a root cause to fix first. Used by
/// the caller when building the verify nudge body.
pub(crate) fn stage_guidance(stage: &VerifyStage) -> &'static str {
    if stage.is_test() {
        "These checks define the exact required behavior. Compare the expected \
         and actual values to infer the precise rule — including edge cases and \
         tie-breaking — then make the smallest edit that satisfies every case."
    } else {
        "Read the error above and fix its root cause (a type, name, or syntax \
         problem) before anything else — the later stages can't run until this \
         passes."
    }
}

pub(crate) fn is_prose_only_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    let name = normalized
        .rsplit('/')
        .next()
        .unwrap_or(path)
        .to_ascii_lowercase();
    if matches!(
        name.as_str(),
        "readme"
            | "license"
            | "licence"
            | "copying"
            | "changelog"
            | "changes"
            | "authors"
            | "contributors"
            | "notice"
    ) {
        return true;
    }
    let Some(ext) = name.rsplit_once('.').map(|(_, ext)| ext) else {
        return false;
    };
    matches!(
        ext,
        "md" | "markdown" | "txt" | "rst" | "adoc" | "asciidoc" | "org"
    )
}

/// Hi-owned runtime files that may exist in legacy workspaces but are not
/// project changes. Keep this list intentionally narrow: `.hi/config.toml`,
/// hooks, skills, and project memory can be deliberate user-visible edits.
pub(crate) fn is_internal_runtime_artifact_path(path: &str) -> bool {
    matches!(
        path.replace('\\', "/").trim_start_matches("./"),
        ".hi/history" | ".hi/memory.undo.md"
    )
}

#[cfg(test)]
#[path = "verify_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "verify_tests_more.rs"]
mod tests_more;

#[cfg(test)]
#[path = "verify_test_support.rs"]
mod verify_test_support;

#[cfg(test)]
#[path = "verify_timeout_tests.rs"]
mod timeout_tests;
