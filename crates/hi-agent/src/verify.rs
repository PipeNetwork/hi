//! The verification-in-the-loop subsystem, extracted from `run_turn`.
//!
//! After the model stops calling tools, [`Verifier`] runs the configured
//! pipeline stages in order (cheap compile/typecheck first, then lint, then
//! tests); the first to fail stops the turn and its output is fed back to the
//! model for another attempt, up to `max_rounds`. A passing pipeline ends the
//! turn. The "only verify turns that changed files" gating lives here too — a
//! turn that edited nothing can't have introduced a failure.
//!
//! Extracted so the verify state machine (round counter, outcome) is owned by
//! one small type instead of entangled with the main loop's locals and the
//! `Agent`'s shared mutable fields.

use hi_tools::run_check_in;

use crate::config::VerifyStage;
use crate::snapshot::{FileFingerprint, SnapshotCache, changed_files_between, workspace_snapshot};
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
    /// the nudge and loops. Carries the 1-based round number. `repeated` is
    /// true when this failure has the same signature as the previous round's —
    /// i.e. the repair attempt did not change the outcome.
    Failed {
        stage: VerifyStage,
        output: String,
        round: u32,
        repeated: bool,
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

/// Owns the verify state machine for one turn: the configured stages, the
/// round cap, and the current round counter.
pub(crate) struct Verifier {
    stages: Vec<VerifyStage>,
    include_affected_packages: bool,
    last_effective_stages: Vec<VerifyStage>,
    executions: Vec<VerificationExecution>,
    stage_mutation_counts: std::collections::BTreeMap<String, u32>,
    last_failure_signature: Option<u64>,
    repeated_failure_count: u32,
    max_rounds: u32,
    round: u32,
}

impl Verifier {
    /// Construct from the agent's config. `stages` empty means verification is
    /// off; `max_rounds` caps the retry rounds.
    pub(crate) fn new(stages: Vec<VerifyStage>, max_rounds: u32) -> Self {
        Self {
            stages,
            include_affected_packages: false,
            last_effective_stages: Vec::new(),
            executions: Vec::new(),
            stage_mutation_counts: std::collections::BTreeMap::new(),
            last_failure_signature: None,
            repeated_failure_count: 0,
            max_rounds,
            round: 0,
        }
    }

    /// Construct an automatically detected verifier. Unlike an explicit
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

    /// How many rounds re-failed with the same signature as their predecessor.
    pub(crate) fn repeated_failure_count(&self) -> u32 {
        self.repeated_failure_count
    }

    /// Record this round's failure signature and report whether it matches the
    /// previous round's — i.e. the repair attempt did not change the failure.
    /// The signature must be computed from output WITHOUT round-dependent
    /// suffixes (pre-turn attribution), or the final round never matches.
    fn note_failure(
        &mut self,
        name: &str,
        command: &str,
        exit_code: Option<i32>,
        output: &str,
    ) -> bool {
        let signature = failure_signature(name, command, exit_code, output);
        let repeated = self.last_failure_signature == Some(signature);
        self.last_failure_signature = Some(signature);
        if repeated {
            self.repeated_failure_count = self.repeated_failure_count.saturating_add(1);
        }
        repeated
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
        let stages = effective_stages(
            workspace.root,
            &changed_files,
            &self.stages,
            self.include_affected_packages,
        );
        self.last_effective_stages = stages.clone();
        if stages.is_empty() {
            return VerifyOutcome::NotRun;
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
            let paths = changed_files
                .iter()
                .map(std::path::PathBuf::from)
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
                let repeated = self.note_failure("lsp", "diagnostics", None, &output);
                return VerifyOutcome::Failed {
                    stage: VerifyStage::new("lsp", "diagnostics"),
                    output,
                    round,
                    repeated,
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
            let before_stage = match workspace_snapshot(workspace.root).await {
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
            let after_stage = match workspace_snapshot(workspace.root).await {
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
                    // Escalation for unstable stages is stage_mutation_counts'
                    // job — keep it out of the repeated-failure signal.
                    repeated: false,
                };
            }
            if execution.status != hi_tools::ToolStatus::Succeeded {
                let mut output = execution.model_content();
                let repeated = self.note_failure(
                    &stage.name,
                    &stage.command,
                    execution.outcome.exit_code,
                    &output,
                );
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
                            repeated,
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
                    repeated,
                };
            }
        }
        VerifyOutcome::Passed
    }
}

/// Hash of the stable identity of a stage failure: stage name/command, exit
/// code, and volatile-token-masked output. Used to detect "same failure,
/// different patch" across verify rounds within one turn.
fn failure_signature(name: &str, command: &str, exit_code: Option<i32>, output: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    command.hash(&mut hasher);
    exit_code.hash(&mut hasher);
    normalize_failure_output(output).hash(&mut hasher);
    hasher.finish()
}

/// Mask volatile tokens so a re-run of the same failure hashes identically:
/// digit runs become `#` (line/column numbers, durations, PIDs, counts) and
/// `0x…` hex runs become `0x#` (addresses). Input capped at 4 KB. A rare
/// false "repeated" only strengthens a nudge — it never blocks anything.
fn normalize_failure_output(output: &str) -> String {
    const CAP_BYTES: usize = 4096;
    let mut out = String::with_capacity(output.len().min(CAP_BYTES));
    let mut consumed = 0usize;
    let mut chars = output.chars().peekable();
    while let Some(c) = chars.next() {
        consumed += c.len_utf8();
        if consumed > CAP_BYTES {
            break;
        }
        if c == '0' && matches!(chars.peek(), Some('x' | 'X')) {
            consumed += chars.next().map(char::len_utf8).unwrap_or(0);
            while matches!(chars.peek(), Some(h) if h.is_ascii_hexdigit()) {
                consumed += chars.next().map(char::len_utf8).unwrap_or(0);
            }
            out.push_str("0x#");
        } else if c.is_ascii_digit() {
            while matches!(chars.peek(), Some(d) if d.is_ascii_digit()) {
                consumed += chars.next().map(char::len_utf8).unwrap_or(0);
            }
            out.push('#');
        } else {
            out.push(c);
        }
    }
    out
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
    for stage in configured {
        if !stages
            .iter()
            .any(|affected| affected.command == stage.command)
        {
            stages.push(stage.clone());
        }
    }
    stages
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
    affected_package_dirs(root, changed_files, |directory| {
        let manifest = directory.join("Cargo.toml");
        manifest.is_file()
            && std::fs::read_to_string(manifest)
                .ok()
                .is_some_and(|text| text.lines().any(|line| line.trim() == "[package]"))
    })
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
    affected_package_dirs(root, changed_files, |directory| {
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
    affected_package_dirs(root, changed_files, |directory| {
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
    affected_package_dirs(root, changed_files, is_python_package_root)
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

fn is_python_package_root(directory: &std::path::Path) -> bool {
    [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "pytest.ini",
        "tox.ini",
    ]
    .iter()
    .any(|marker| directory.join(marker).is_file())
}

/// Find the nearest nested package root for each changed path. The workspace
/// root itself is deliberately omitted because its configured stages already
/// cover it. Invalid or escaping ledger paths are ignored.
fn affected_package_dirs(
    root: &std::path::Path,
    changed_files: &[String],
    is_package_root: impl Fn(&std::path::Path) -> bool,
) -> std::collections::BTreeSet<String> {
    let mut packages = std::collections::BTreeSet::new();
    for changed in changed_files {
        let relative = std::path::Path::new(changed);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            continue;
        }
        let mut directory = root.join(relative);
        if !directory.is_dir() {
            directory.pop();
        }
        while directory.starts_with(root) && directory != root {
            if is_package_root(&directory) {
                if let Ok(relative_package) = directory.strip_prefix(root) {
                    packages.insert(relative_package.to_string_lossy().replace('\\', "/"));
                }
                break;
            }
            if !directory.pop() {
                break;
            }
        }
    }
    packages
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

    #[test]
    fn normalize_masks_volatile_tokens() {
        // Line/column drift must not change the signature: an edit that
        // shifts lines without fixing the error is still the same failure.
        assert_eq!(
            normalize_failure_output("error[E0308]: mismatched types\n --> src/lib.rs:42:18"),
            normalize_failure_output("error[E0308]: mismatched types\n --> src/lib.rs:43:18"),
        );
        // Durations and addresses are masked.
        assert_eq!(
            normalize_failure_output("test failed in 1.03s at 0xdeadbeef"),
            normalize_failure_output("test failed in 2.94s at 0x7ffe1234"),
        );
        // Different error text stays distinct even with digits masked:
        // E0308/E0499 both normalize to E#, but the message words differ.
        assert_ne!(
            normalize_failure_output("error[E0308]: mismatched types"),
            normalize_failure_output("error[E0499]: cannot borrow twice"),
        );
    }

    #[test]
    fn failure_signature_keys_on_stage_and_exit_code() {
        let sig = |name: &str, code, out: &str| failure_signature(name, "cmd", code, out);
        assert_eq!(sig("test", Some(1), "boom"), sig("test", Some(1), "boom"));
        assert_ne!(sig("test", Some(1), "boom"), sig("lint", Some(1), "boom"));
        assert_ne!(sig("test", Some(1), "boom"), sig("test", Some(2), "boom"));
    }

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
        let v = Verifier::new(Vec::new(), 2);
        assert!(!v.is_on());
        assert_eq!(v.round(), 0);
    }

    #[test]
    fn verifier_is_on_with_stages() {
        let v = Verifier::new(vec![VerifyStage::new("check", "true")], 2);
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
            vec![
                (
                    "affected-check:crates/library",
                    "cargo check --quiet --manifest-path 'crates/library/Cargo.toml'",
                ),
                (
                    "affected-test:crates/library",
                    "cargo test --quiet --manifest-path 'crates/library/Cargo.toml'",
                ),
                ("check", "cargo check --quiet"),
                ("test", "cargo test --quiet"),
            ]
        );
        let _ = std::fs::remove_dir_all(base);
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
                "go -C 'services/alpha' build ./...",
                "go -C 'services/alpha' test ./...",
                "go -C 'services/zeta' build ./...",
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
        assert!(Verifier::automatic(Vec::new(), 1).is_on());
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
        let mut verifier = Verifier::new(vec![VerifyStage::new("docs", "false")], 1);
        let lsp = hi_lsp::LspManager::new(&root);
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
        let mut verifier = Verifier::new(vec![VerifyStage::new("test", "false")], 1);
        let lsp = hi_lsp::LspManager::new(&root);
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
        let mut verifier = Verifier::new(vec![VerifyStage::new("test", "test -f .env")], 1);
        let lsp = hi_lsp::LspManager::new(&root);
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

        let mut verifier = Verifier::new(
            vec![VerifyStage::new(
                "test",
                "printf 'baseline failure\\n' >&2; exit 7",
            )],
            1,
        );
        let lsp = hi_lsp::LspManager::new(&root);
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

        let mut verifier = Verifier::new(
            vec![VerifyStage::new("test", "test \"$(cat state.toml)\" = ok")],
            1,
        );
        let lsp = hi_lsp::LspManager::new(&root);
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

        let mut verifier = Verifier::new(
            vec![VerifyStage::new("test", "test \"$(cat state.toml)\" = ok")],
            2,
        );
        let lsp = hi_lsp::LspManager::new(&root);
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
            let mut verifier = Verifier::new(vec![VerifyStage::new("test", command)], repairs + 1);
            let lsp = hi_lsp::LspManager::new(&root);
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

        let mut verifier = Verifier::new(
            vec![VerifyStage::new("test", "test \"$(cat state.txt)\" = ok")],
            1,
        );
        let lsp = hi_lsp::LspManager::new(&root);
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
        let mut verifier = Verifier::new(vec![VerifyStage::new("test", "exit 1")], 1);
        let lsp = hi_lsp::LspManager::new(&root);
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
        let mut verifier = Verifier::new(
            vec![VerifyStage::new(
                "formatter",
                "printf mutation >> source.rs; exit 0",
            )],
            2,
        );
        let lsp = hi_lsp::LspManager::new(&root);
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
