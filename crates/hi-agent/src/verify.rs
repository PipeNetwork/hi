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

use hi_tools::run_check_in;

use crate::config::VerifyStage;
use crate::snapshot::{
    FileFingerprint, SnapshotCache, changed_files_between, workspace_snapshot,
    workspace_snapshot_meta,
};
use crate::ui::Ui;

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
            process: Some(execution.outcome.clone()),
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
    pre_turn_checkpoint: Option<&'a str>,
    lsp: &'a hi_lsp::LspManager,
    known_changed_files: Option<&'a [String]>,
    mutation_seen: bool,
    /// Packages mid-turn `cargo check` already sealed green at the current
    /// ledger revision — skip matching `affected-check:` stages (Phase I).
    skip_affected_checks: Option<&'a std::collections::BTreeSet<String>>,
    /// Packages mid-turn `cargo test` already sealed green — skip `affected-test:`.
    skip_affected_tests: Option<&'a std::collections::BTreeSet<String>>,
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
            pre_turn_checkpoint,
            lsp,
            known_changed_files: None,
            mutation_seen: false,
            skip_affected_checks: None,
            skip_affected_tests: None,
        }
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
    executions: Vec<VerificationExecution>,
    stage_mutation_counts: std::collections::BTreeMap<String, u32>,
    max_rounds: u32,
    round: u32,
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
            executions: Vec::new(),
            stage_mutation_counts: std::collections::BTreeMap::new(),
            max_rounds,
            round: 0,
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
        self.max_rounds = self.max_rounds.saturating_add(1);
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
        &self.executions
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
        ui: &mut dyn Ui,
    ) -> VerifyOutcome {
        if (self.stages.is_empty() && !self.include_affected_packages)
            || self.round >= self.max_rounds
        {
            return VerifyOutcome::NotRun;
        }
        let changed_files = if let Some(changed_files) = workspace.known_changed_files {
            changed_files.to_vec()
        } else {
            let current = match snapshot_cache.get(workspace.root).await {
                Ok(current) => current,
                Err(error) => {
                    self.round += 1;
                    let round = self.round;
                    let stage =
                        self.stages.first().cloned().unwrap_or_else(|| {
                            VerifyStage::new("auto", "affected package discovery")
                        });
                    self.executions
                        .push(VerificationExecution::infrastructure_failure(round, &stage));
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
        let mut stages = effective_stages(
            workspace.root,
            &changed_files,
            &self.stages,
            self.include_affected_packages,
        );
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
        self.round += 1;
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
                self.executions.push(VerificationExecution::lsp(
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
                self.executions.push(VerificationExecution::lsp(
                    round,
                    hi_tools::ToolStatus::Failed,
                ));
            } else if lsp_checked {
                self.executions.push(VerificationExecution::lsp(
                    round,
                    hi_tools::ToolStatus::Succeeded,
                ));
            }
        }

        for stage in &stages {
            ui.status(&format!(
                "verifying ({round}/{max_rounds}) · {}: {}",
                stage.name, stage.command
            ));
            let before_stage = match workspace_snapshot_meta(workspace.root).await {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    self.executions
                        .push(VerificationExecution::infrastructure_failure(round, stage));
                    return VerifyOutcome::InfrastructureError {
                        stage: stage.clone(),
                        output: format!("pre-stage workspace snapshot failed: {error:#}"),
                        round,
                    };
                }
            };
            let execution = match run_check_in(workspace.root, &stage.command).await {
                Ok(execution) => execution,
                Err(error) => {
                    self.executions
                        .push(VerificationExecution::infrastructure_failure(round, stage));
                    return VerifyOutcome::InfrastructureError {
                        stage: stage.clone(),
                        output: format!("verification process infrastructure failed: {error:#}"),
                        round,
                    };
                }
            };
            self.executions
                .push(VerificationExecution::shell(round, stage, &execution));
            let after_stage = match workspace_snapshot_meta(workspace.root).await {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    self.executions
                        .push(VerificationExecution::infrastructure_failure(round, stage));
                    return VerifyOutcome::InfrastructureError {
                        stage: stage.clone(),
                        output: format!("post-stage workspace snapshot failed: {error:#}"),
                        round,
                    };
                }
            };
            let stage_changes = changed_files_between(&before_stage, &after_stage)
                .into_iter()
                .filter(|path| verification_relevant_path(path))
                .collect::<Vec<_>>();
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
                self.executions
                    .push(VerificationExecution::infrastructure_failure(round, stage));
                return VerifyOutcome::InfrastructureError {
                    stage: stage.clone(),
                    output: format!(
                        "stage `{}` (`{}`) exceeded its time budget and was killed, so this revision is unverified — this is not a code failure. Raise HI_VERIFY_TIMEOUT_SECS (default {}s), or narrow the stage to something that fits the budget (for example a package-local check instead of a whole-workspace test run).",
                        stage.name,
                        stage.command,
                        hi_tools::check_timeout().as_secs(),
                    ),
                    round,
                };
            }
            if execution.status != hi_tools::ToolStatus::Succeeded {
                let mut output = execution.model_content();
                if round == max_rounds {
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
                    let baseline = hi_tools::checkpoint::with_isolated_checkpoint(
                        workspace.root,
                        checkpoint,
                        workspace.state_root,
                        move |isolated| async move { run_check_in(&isolated, &command).await },
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
    // minutes). Cross-crate breakage in untouched dependents is left to CI /
    // an explicit `/verify` stage.
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
    hi_tools::affected_cargo_package_dirs(root, changed_files)
        .into_iter()
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
        .collect()
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
            stages.push(VerifyStage::new(
                format!("affected-test:{label}"),
                format!("pytest -q {quoted}"),
            ));
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

fn verification_relevant_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/").to_ascii_lowercase();
    !normalized
        .split('/')
        .any(|part| part == "__pycache__" || part == ".hi")
        && !matches!(
            normalized.rsplit('.').next(),
            Some("pyc" | "pyo" | "class" | "o" | "obj")
        )
        && !is_prose_only_path(&normalized)
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct NullUi;

    impl Ui for NullUi {
        fn assistant_text(&mut self, _: &str) {}
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str, _: &str) {}
        fn status(&mut self, _: &str) {}
        fn turn_end(&mut self, _: &str) {}
    }

    fn roots(label: &str) -> (PathBuf, PathBuf, PathBuf) {
        static N: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir().join(format!(
            "hi-verifier-{label}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let root = base.join("workspace");
        let state = base.join("state");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        (base, root, state)
    }

    async fn checkpoint(root: &Path, state: &Path) -> String {
        match hi_tools::checkpoint::create_detailed_with_state(root, state).await {
            hi_tools::checkpoint::CreateResult::Created(id) => id,
            other => panic!("checkpoint failed: {other:?}"),
        }
    }

    #[test]
    fn verifier_is_off_when_no_stages() {
        let v = RepairVerifier::new(Vec::new(), 2);
        assert!(!v.is_on());
        assert_eq!(v.round(), 0);
    }

    #[test]
    fn verifier_is_on_with_stages() {
        let v = RepairVerifier::new(vec![VerifyStage::new("check", "true")], 2);
        assert!(v.is_on());
    }

    #[test]
    fn lsp_execution_evidence_does_not_invent_process_data() {
        let record = VerificationExecution::lsp(3, hi_tools::ToolStatus::Failed);
        let value = serde_json::to_value(&record).unwrap();

        assert_eq!(value["round"], 3);
        assert_eq!(value["name"], "lsp");
        assert_eq!(value["command"], "diagnostics");
        assert_eq!(value["status"], "failed");
        assert!(value.get("process").is_none());
        assert!(value.get("truncation").is_none());
    }

    #[test]
    fn stage_guidance_differs_tests_vs_compile() {
        let test_stage = VerifyStage::new("test", "pytest");
        let compile_stage = VerifyStage::new("check", "cargo check");
        assert_ne!(stage_guidance(&test_stage), stage_guidance(&compile_stage));
        assert!(stage_guidance(&test_stage).contains("required behavior"));
        assert!(stage_guidance(&compile_stage).contains("root cause"));
    }

    #[test]
    fn prose_only_path_detection_is_conservative() {
        assert!(is_prose_only_path("README.md"));
        assert!(is_prose_only_path("docs/guide.rst"));
        assert!(is_prose_only_path("LICENSE"));
        assert!(!is_prose_only_path("package.json"));
        assert!(!is_prose_only_path("docs/example.ts"));
        assert!(!is_prose_only_path(".github/workflows/test.yml"));
    }

    #[test]
    fn verification_mutation_filter_keeps_source_and_ignores_generated_noise() {
        assert!(verification_relevant_path("src/lib.rs"));
        assert!(verification_relevant_path("Cargo.lock"));
        assert!(!verification_relevant_path("pkg/__pycache__/mod.pyc"));
        assert!(!verification_relevant_path("README.md"));
        assert!(!verification_relevant_path(".hi/state.json"));
    }

    #[test]
    fn skip_affected_stage_matches_sealed_package_labels() {
        let mut checks = std::collections::BTreeSet::new();
        checks.insert("crates/library".into());
        checks.insert("web".into());
        checks.insert("svc".into());
        let tests = std::collections::BTreeSet::new();
        assert!(should_skip_affected_stage(
            &VerifyStage::new(
                "affected-check:crates/library",
                "cargo check --quiet --manifest-path 'crates/library/Cargo.toml'",
            ),
            &checks,
            &tests,
        ));
        // Phase O: polyglot check seals cover typecheck/build/lint stages.
        assert!(should_skip_affected_stage(
            &VerifyStage::new(
                "affected-typecheck:web",
                "npm --prefix 'web' exec -- tsc --noEmit",
            ),
            &checks,
            &tests,
        ));
        assert!(should_skip_affected_stage(
            &VerifyStage::new("affected-build:svc", "go -C 'svc' build ./..."),
            &checks,
            &tests,
        ));
        assert!(should_skip_affected_stage(
            &VerifyStage::new("affected-lint:pkg", "ruff check 'pkg'"),
            &{
                let mut c = checks.clone();
                c.insert("pkg".into());
                c
            },
            &tests,
        ));
        assert!(!should_skip_affected_stage(
            &VerifyStage::new(
                "affected-test:crates/library",
                "cargo test --quiet --manifest-path 'crates/library/Cargo.toml'",
            ),
            &checks,
            &tests,
        ));
        // Root pipeline is never skipped via this path.
        assert!(!should_skip_affected_stage(
            &VerifyStage::new("check", "cargo check --quiet"),
            &checks,
            &tests,
        ));
        let mut test_set = std::collections::BTreeSet::new();
        test_set.insert("crates/library".into());
        assert!(should_skip_affected_stage(
            &VerifyStage::new(
                "affected-test:crates/library",
                "cargo test --quiet --manifest-path 'crates/library/Cargo.toml'",
            ),
            &checks,
            &test_set,
        ));
    }

    #[test]
    fn affected_cargo_packages_precede_the_root_pipeline() {
        let (base, root, _) = roots("affected-cargo");
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/*\"]\ndefault-members = [\"crates/app\"]\n",
        )
        .unwrap();
        for package in ["app", "library"] {
            let package_root = root.join("crates").join(package);
            std::fs::create_dir_all(package_root.join("src")).unwrap();
            std::fs::write(
                package_root.join("Cargo.toml"),
                format!("[package]\nname = \"{package}\"\nversion = \"0.1.0\"\n"),
            )
            .unwrap();
        }
        let stages = effective_stages(
            &root,
            &[
                "crates/library/src/lib.rs".into(),
                "crates/library/src/other.rs".into(),
            ],
            &[
                VerifyStage::new("check", "cargo check --quiet"),
                VerifyStage::new("test", "cargo test --quiet"),
            ],
            true,
        );
        assert_eq!(
            stages
                .iter()
                .map(|stage| (stage.name.as_str(), stage.command.as_str()))
                .collect::<Vec<_>>(),
            vec![(
                "affected-test:crates/library",
                "cargo test --quiet --manifest-path 'crates/library/Cargo.toml'",
            ),]
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn package_local_tests_supersede_the_whole_workspace_test_run() {
        // Measured on a 24-crate workspace: `cargo test` 811s vs `cargo check`
        // 114s, against a 600s stage timeout. Every turn ended unjudged however
        // small the edit, because the gate's cost tracked the project rather
        // than the change.
        let (base, root, _) = roots("supersede-workspace-test");
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/*\"]\n",
        )
        .unwrap();
        let package_root = root.join("crates").join("library");
        std::fs::create_dir_all(package_root.join("src")).unwrap();
        std::fs::write(
            package_root.join("Cargo.toml"),
            "[package]\nname = \"library\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        let configured = vec![
            VerifyStage::new("check", "cargo check --quiet"),
            VerifyStage::new("test", "cargo test --quiet"),
        ];
        let changed = ["crates/library/src/lib.rs".to_string()];

        let auto = effective_stages(&root, &changed, &configured, true);
        assert!(
            !auto.iter().any(|s| s.command == "cargo test --quiet"),
            "the whole-workspace test run must be superseded: {auto:?}"
        );
        assert!(
            !auto.iter().any(|s| s.command == "cargo check --quiet"),
            "the whole-workspace check is also superseded when package tests cover the edit: {auto:?}"
        );
        assert!(
            auto.iter()
                .any(|s| s.name == "affected-test:crates/library"),
            "package-local coverage must actually be present: {auto:?}"
        );

        // Explicit configuration is the user's decision and is run as written —
        // this refinement applies only to the auto-detected pipeline.
        let explicit = effective_stages(&root, &changed, &configured, false);
        assert_eq!(explicit, configured, "explicit stages must be untouched");
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn only_unscoped_cargo_test_commands_are_superseded() {
        assert!(is_whole_workspace_cargo_test("cargo test"));
        assert!(is_whole_workspace_cargo_test("cargo test --quiet"));
        assert!(is_whole_workspace_cargo_test("  cargo test --workspace  "));
        // Already narrowed by the caller — leave it alone.
        assert!(!is_whole_workspace_cargo_test("cargo test -p library"));
        assert!(!is_whole_workspace_cargo_test("cargo test --package library"));
        assert!(!is_whole_workspace_cargo_test(
            "cargo test --manifest-path 'a/Cargo.toml'"
        ));
        assert!(!is_whole_workspace_cargo_test("cargo test --test integration"));
        // Not a plain `cargo test` at all.
        assert!(!is_whole_workspace_cargo_test("cargo testsuite"));
        assert!(!is_whole_workspace_cargo_test("cargo test && ./extra.sh"));
        assert!(!is_whole_workspace_cargo_test("cargo check --quiet"));
        assert!(!is_whole_workspace_cargo_test("make test"));

        assert!(is_whole_workspace_cargo_check("cargo check"));
        assert!(is_whole_workspace_cargo_check("cargo check --quiet"));
        assert!(!is_whole_workspace_cargo_check("cargo check -p library"));
        assert!(!is_whole_workspace_cargo_check(
            "cargo check --manifest-path 'a/Cargo.toml'"
        ));
        assert!(!is_whole_workspace_cargo_check("cargo test --quiet"));
        assert!(is_package_local_cargo_test(
            "cargo test --quiet --manifest-path 'crates/library/Cargo.toml'"
        ));
        assert!(!is_package_local_cargo_test("cargo test --quiet"));
    }

    #[test]
    fn root_cargo_changes_do_not_duplicate_the_root_pipeline() {
        let (base, root, _) = roots("root-cargo");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"single\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        let configured = vec![VerifyStage::new("test", "cargo test --quiet")];
        assert_eq!(
            effective_stages(&root, &["src/lib.rs".into()], &configured, true),
            configured
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn affected_javascript_package_precedes_root_pipeline_and_deduplicates_changes() {
        let (base, root, _) = roots("affected-javascript");
        std::fs::write(
            root.join("package.json"),
            r#"{"scripts":{"test":"root-test"}}"#,
        )
        .unwrap();
        let package = root.join("apps/web");
        std::fs::create_dir_all(package.join("src")).unwrap();
        std::fs::write(
            package.join("package.json"),
            r#"{"scripts":{"typecheck":"tsc --noEmit","test":"vitest"}}"#,
        )
        .unwrap();
        std::fs::write(package.join("tsconfig.json"), "{}\n").unwrap();

        let stages = effective_stages(
            &root,
            &[
                "apps/web/src/index.ts".into(),
                "apps/web/src/other.ts".into(),
            ],
            &[
                VerifyStage::new("typecheck", "npx --no-install tsc --noEmit"),
                VerifyStage::new("test", "npm test --silent"),
            ],
            true,
        );

        assert_eq!(
            stages
                .iter()
                .map(|stage| (stage.name.as_str(), stage.command.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (
                    "affected-typecheck:apps/web",
                    "npm --prefix 'apps/web' run typecheck --silent",
                ),
                (
                    "affected-test:apps/web",
                    "npm --prefix 'apps/web' test --silent",
                ),
                ("typecheck", "npx --no-install tsc --noEmit"),
                ("test", "npm test --silent"),
            ]
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn affected_go_modules_are_sorted_before_the_root_pipeline() {
        let (base, root, _) = roots("affected-go");
        std::fs::write(root.join("go.mod"), "module example.test/root\n").unwrap();
        for module in ["services/zeta", "services/alpha"] {
            let module_root = root.join(module);
            std::fs::create_dir_all(module_root.join("pkg")).unwrap();
            std::fs::write(
                module_root.join("go.mod"),
                format!("module example.test/{module}\n"),
            )
            .unwrap();
        }

        let stages = effective_stages(
            &root,
            &[
                "services/zeta/pkg/z.go".into(),
                "services/alpha/pkg/a.go".into(),
            ],
            &[VerifyStage::new("test", "go test ./...")],
            true,
        );

        assert_eq!(
            stages
                .iter()
                .map(|stage| stage.command.as_str())
                .collect::<Vec<_>>(),
            vec![
                // Package-local `go build` is dropped when `go test` for the
                // same module will run — test already compiles.
                "go -C 'services/alpha' test ./...",
                "go -C 'services/zeta' test ./...",
                "go test ./...",
            ]
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn affected_python_package_uses_pyproject_tools_before_root_pipeline() {
        let (base, root, _) = roots("affected-python");
        std::fs::write(root.join("pyproject.toml"), "[project]\nname='root'\n").unwrap();
        let package = root.join("packages/service");
        std::fs::create_dir_all(package.join("service")).unwrap();
        std::fs::write(
            package.join("pyproject.toml"),
            "[project]\nname='service'\n[tool.ruff]\nline-length=100\n",
        )
        .unwrap();

        let stages = effective_stages(
            &root,
            &["packages/service/service/api.py".into()],
            &[
                VerifyStage::new("lint", "ruff check ."),
                VerifyStage::new("test", "pytest -q"),
            ],
            true,
        );

        assert_eq!(
            stages
                .iter()
                .map(|stage| (stage.name.as_str(), stage.command.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (
                    "affected-lint:packages/service",
                    "ruff check 'packages/service'",
                ),
                (
                    "affected-test:packages/service",
                    "pytest -q 'packages/service'",
                ),
                ("lint", "ruff check ."),
                ("test", "pytest -q"),
            ]
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn python_setup_and_pytest_markers_define_nested_package_roots() {
        let (base, root, _) = roots("python-markers");
        for (package, marker) in [
            ("packages/legacy", "setup.py"),
            ("packages/tests-only", "pytest.ini"),
        ] {
            let package_root = root.join(package);
            std::fs::create_dir_all(package_root.join("src")).unwrap();
            std::fs::write(package_root.join(marker), "\n").unwrap();
        }

        let stages = effective_stages(
            &root,
            &[
                "packages/tests-only/src/test_api.py".into(),
                "packages/legacy/src/module.py".into(),
            ],
            &[],
            true,
        );

        assert_eq!(
            stages
                .iter()
                .map(|stage| stage.command.as_str())
                .collect::<Vec<_>>(),
            vec![
                "pytest -q 'packages/legacy'",
                "pytest -q 'packages/tests-only'",
            ]
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn root_javascript_go_and_python_changes_do_not_duplicate_root_stages() {
        let (base, root, _) = roots("root-polyglot");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("package.json"), "{}\n").unwrap();
        std::fs::write(root.join("go.mod"), "module example.test/root\n").unwrap();
        std::fs::write(root.join("pyproject.toml"), "[project]\nname='root'\n").unwrap();
        let configured = vec![VerifyStage::new("root", "./root-verify")];

        assert_eq!(
            effective_stages(
                &root,
                &[
                    "src/index.ts".into(),
                    "src/main.go".into(),
                    "src/main.py".into(),
                ],
                &configured,
                true,
            ),
            configured
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn automatic_mode_finds_nested_packages_without_a_root_manifest() {
        let (base, root, _) = roots("nested-only");
        let package = root.join("nested/app");
        std::fs::create_dir_all(package.join("src")).unwrap();
        std::fs::write(package.join("package.json"), "{}\n").unwrap();

        let stages = effective_stages(&root, &["nested/app/src/index.js".into()], &[], true);

        assert_eq!(
            stages,
            vec![VerifyStage::new(
                "affected-test:nested/app",
                "npm --prefix 'nested/app' test --silent",
            )]
        );
        assert!(RepairVerifier::automatic(Vec::new(), 1).is_on());
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn explicit_pipeline_is_exact_even_for_nested_package_changes() {
        let (base, root, _) = roots("explicit-exact");
        let package = root.join("apps/web");
        std::fs::create_dir_all(package.join("src")).unwrap();
        std::fs::write(package.join("package.json"), "{}\n").unwrap();
        let explicit = vec![VerifyStage::new("custom", "./verify-exactly")];

        assert_eq!(
            effective_stages(&root, &["apps/web/src/index.ts".into()], &explicit, false,),
            explicit
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn explicit_documentation_pipeline_is_not_skipped_as_prose_only() {
        let (base, root, state) = roots("explicit-docs");
        std::fs::write(root.join("README.md"), "before\n").unwrap();
        let turn_snapshot = workspace_snapshot(&root).await.unwrap();
        std::fs::write(root.join("README.md"), "after\n").unwrap();
        let mut verifier = RepairVerifier::new(vec![VerifyStage::new("docs", "false")], 1);
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let mut cache = SnapshotCache::default();
        let mut ui = NullUi;

        let outcome = verifier
            .check(
                &VerifyWorkspace::new(&root, &state, None, &lsp),
                &turn_snapshot,
                &mut cache,
                &mut ui,
            )
            .await;

        assert!(matches!(outcome, VerifyOutcome::Failed { .. }));
        assert_eq!(verifier.executions().len(), 1);
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn applied_net_zero_mutation_still_runs_explicit_verification() {
        let (base, root, state) = roots("net-zero-mutation");
        let turn_snapshot = workspace_snapshot(&root).await.unwrap();
        let mut verifier = RepairVerifier::new(vec![VerifyStage::new("test", "false")], 1);
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let mut cache = SnapshotCache::default();
        let mut ui = NullUi;

        let outcome = verifier
            .check(
                &VerifyWorkspace::new(&root, &state, None, &lsp)
                    .with_changed_files(&[])
                    .with_mutation_seen(true),
                &turn_snapshot,
                &mut cache,
                &mut ui,
            )
            .await;

        assert!(matches!(outcome, VerifyOutcome::Failed { .. }));
        assert_eq!(verifier.executions().len(), 1);
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn gitignored_inputs_still_trigger_verification() {
        let (base, root, state) = roots("ignored-input");
        std::fs::write(root.join(".gitignore"), ".env\n").unwrap();
        let turn_snapshot = workspace_snapshot(&root).await.unwrap();
        std::fs::write(root.join(".env"), "MODE=test\n").unwrap();
        let mut verifier = RepairVerifier::new(vec![VerifyStage::new("test", "test -f .env")], 1);
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let mut cache = SnapshotCache::default();
        let mut ui = NullUi;

        let outcome = verifier
            .check(
                &VerifyWorkspace::new(&root, &state, None, &lsp),
                &turn_snapshot,
                &mut cache,
                &mut ui,
            )
            .await;

        assert!(matches!(outcome, VerifyOutcome::Passed));
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn final_failure_is_classified_against_internal_pre_turn_checkpoint() {
        let (base, root, state) = roots("preexisting");
        std::fs::write(root.join("source.rs"), "before\n").unwrap();
        let turn_snapshot = workspace_snapshot(&root).await.unwrap();
        let checkpoint = checkpoint(&root, &state).await;
        std::fs::write(root.join("source.rs"), "current changed contents\n").unwrap();

        let mut verifier = RepairVerifier::new(
            vec![VerifyStage::new(
                "test",
                "printf 'baseline failure\\n' >&2; exit 7",
            )],
            1,
        );
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let mut cache = SnapshotCache::default();
        let mut ui = NullUi;
        let outcome = verifier
            .check(
                &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
                &turn_snapshot,
                &mut cache,
                &mut ui,
            )
            .await;
        let VerifyOutcome::Failed { output, round, .. } = outcome else {
            panic!("expected classified failure");
        };
        assert_eq!(round, 1);
        assert!(output.contains("already failed this verification stage before the turn"));
        assert!(output.contains("Baseline output:\nbaseline failure"));
        assert_eq!(
            std::fs::read_to_string(root.join("source.rs")).unwrap(),
            "current changed contents\n",
            "baseline attribution must never restore over the destination"
        );
        assert!(!state.join("verification-sandboxes").exists());
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn final_failure_absent_from_pre_turn_checkpoint_is_identified() {
        let (base, root, state) = roots("introduced");
        std::fs::write(root.join("state.toml"), "ok\n").unwrap();
        let turn_snapshot = workspace_snapshot(&root).await.unwrap();
        let checkpoint = checkpoint(&root, &state).await;
        std::fs::write(root.join("state.toml"), "broken now\n").unwrap();

        let mut verifier = RepairVerifier::new(
            vec![VerifyStage::new("test", "test \"$(cat state.toml)\" = ok")],
            1,
        );
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let mut cache = SnapshotCache::default();
        let mut ui = NullUi;
        let outcome = verifier
            .check(
                &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
                &turn_snapshot,
                &mut cache,
                &mut ui,
            )
            .await;
        let VerifyOutcome::Failed { output, .. } = outcome else {
            panic!("expected classified failure");
        };
        assert!(output.contains("current failure was not present at the turn baseline"));
        assert_eq!(
            std::fs::read_to_string(root.join("state.toml")).unwrap(),
            "broken now\n"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn baseline_attribution_runs_only_after_last_allowed_check() {
        let (base, root, state) = roots("final-only");
        std::fs::write(root.join("state.toml"), "ok\n").unwrap();
        let turn_snapshot = workspace_snapshot(&root).await.unwrap();
        let checkpoint = checkpoint(&root, &state).await;
        std::fs::write(root.join("state.toml"), "broken now\n").unwrap();

        let mut verifier = RepairVerifier::new(
            vec![VerifyStage::new("test", "test \"$(cat state.toml)\" = ok")],
            2,
        );
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let mut cache = SnapshotCache::default();
        let mut ui = NullUi;
        let first = verifier
            .check(
                &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
                &turn_snapshot,
                &mut cache,
                &mut ui,
            )
            .await;
        let VerifyOutcome::Failed { output, round, .. } = first else {
            panic!("expected first failure");
        };
        assert_eq!(round, 1);
        assert!(!output.contains("Pre-turn attribution"));

        let second = verifier
            .check(
                &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
                &turn_snapshot,
                &mut cache,
                &mut ui,
            )
            .await;
        let VerifyOutcome::Failed { output, round, .. } = second else {
            panic!("expected final failure");
        };
        assert_eq!(round, 2);
        assert!(output.contains("Pre-turn attribution"));
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn repair_budgets_zero_one_two_run_one_two_three_checks() {
        for repairs in 0..=2 {
            let (base, root, state) = roots(&format!("budget-{repairs}"));
            let counter = base.join("runs");
            std::fs::write(root.join("source.rs"), "before\n").unwrap();
            let turn_snapshot = workspace_snapshot(&root).await.unwrap();
            std::fs::write(root.join("source.rs"), "current changed contents\n").unwrap();
            let command = format!("printf x >> {}; exit 1", counter.display());
            let mut verifier =
                RepairVerifier::new(vec![VerifyStage::new("test", command)], repairs + 1);
            let lsp = hi_lsp::LspManager::new(&root).unwrap();
            let mut cache = SnapshotCache::default();
            let mut ui = NullUi;
            for expected_round in 1..=(repairs + 1) {
                let outcome = verifier
                    .check(
                        &VerifyWorkspace::new(&root, &state, None, &lsp),
                        &turn_snapshot,
                        &mut cache,
                        &mut ui,
                    )
                    .await;
                assert!(matches!(
                    outcome,
                    VerifyOutcome::Failed { round, .. } if round == expected_round
                ));
            }
            assert!(matches!(
                verifier
                    .check(
                        &VerifyWorkspace::new(&root, &state, None, &lsp),
                        &turn_snapshot,
                        &mut cache,
                        &mut ui,
                    )
                    .await,
                VerifyOutcome::NotRun
            ));
            assert_eq!(
                std::fs::read(&counter).unwrap().len(),
                (repairs + 1) as usize
            );
            assert_eq!(verifier.executions().len(), (repairs + 1) as usize);
            for (index, execution) in verifier.executions().iter().enumerate() {
                assert_eq!(execution.round, index as u32 + 1);
                assert_eq!(execution.name, "test");
                assert_eq!(execution.status, hi_tools::ToolStatus::Failed);
                assert_eq!(
                    execution
                        .process
                        .as_ref()
                        .and_then(|process| process.exit_code),
                    Some(1)
                );
                assert_eq!(
                    execution.truncation,
                    Some(hi_tools::TruncationState::Complete)
                );
            }
            let _ = std::fs::remove_dir_all(base);
        }
    }

    #[tokio::test]
    async fn late_mutation_requires_a_fresh_current_revision_pass() {
        let (base, root, state) = roots("late-mutation");
        std::fs::write(root.join("state.txt"), "ok\n").unwrap();
        std::fs::write(root.join("source.rs"), "before\n").unwrap();
        let turn_snapshot = workspace_snapshot(&root).await.unwrap();
        let checkpoint = checkpoint(&root, &state).await;
        std::fs::write(root.join("source.rs"), "current changed contents\n").unwrap();

        let mut verifier = RepairVerifier::new(
            vec![VerifyStage::new("test", "test \"$(cat state.txt)\" = ok")],
            1,
        );
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let mut cache = SnapshotCache::default();
        let mut ui = NullUi;
        assert!(matches!(
            verifier
                .check(
                    &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
                    &turn_snapshot,
                    &mut cache,
                    &mut ui,
                )
                .await,
            VerifyOutcome::Passed
        ));

        std::fs::write(root.join("state.txt"), "late mutation broke it\n").unwrap();
        cache.invalidate();
        verifier.allow_review_revalidation();
        let outcome = verifier
            .check(
                &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
                &turn_snapshot,
                &mut cache,
                &mut ui,
            )
            .await;
        assert!(matches!(outcome, VerifyOutcome::Failed { round: 2, .. }));
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn broken_attribution_checkpoint_is_infrastructure_error() {
        let (base, root, state) = roots("infra");
        std::fs::write(root.join("source.rs"), "before\n").unwrap();
        let turn_snapshot = workspace_snapshot(&root).await.unwrap();
        std::fs::write(root.join("source.rs"), "current changed contents\n").unwrap();
        let mut verifier = RepairVerifier::new(vec![VerifyStage::new("test", "exit 1")], 1);
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let mut cache = SnapshotCache::default();
        let mut ui = NullUi;
        let outcome = verifier
            .check(
                &VerifyWorkspace::new(
                    &root,
                    &state,
                    Some("internal:v1:not-this-workspace:missing"),
                    &lsp,
                ),
                &turn_snapshot,
                &mut cache,
                &mut ui,
            )
            .await;
        assert!(matches!(
            outcome,
            VerifyOutcome::InfrastructureError { round: 1, .. }
        ));
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn repeatedly_mutating_verification_stage_is_unstable_not_a_pass() {
        let (base, root, state) = roots("unstable");
        std::fs::write(root.join("source.rs"), "before\n").unwrap();
        let turn_snapshot = workspace_snapshot(&root).await.unwrap();
        std::fs::write(root.join("source.rs"), "current changed contents\n").unwrap();
        let mut verifier = RepairVerifier::new(
            vec![VerifyStage::new(
                "formatter",
                "printf mutation >> source.rs; exit 0",
            )],
            2,
        );
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let mut cache = SnapshotCache::default();
        let mut ui = NullUi;
        let first = verifier
            .check(
                &VerifyWorkspace::new(&root, &state, None, &lsp),
                &turn_snapshot,
                &mut cache,
                &mut ui,
            )
            .await;
        assert!(matches!(
            first,
            VerifyOutcome::Failed {
                round: 1,
                ref output,
                ..
            } if output.contains("modified relevant source files")
        ));
        let outcome = verifier
            .check(
                &VerifyWorkspace::new(&root, &state, None, &lsp),
                &turn_snapshot,
                &mut cache,
                &mut ui,
            )
            .await;
        assert!(matches!(
            outcome,
            VerifyOutcome::Unstable {
                round: 2,
                ref changed_files,
                ..
            } if changed_files == &["source.rs"]
        ));
        let _ = std::fs::remove_dir_all(base);
    }
}

