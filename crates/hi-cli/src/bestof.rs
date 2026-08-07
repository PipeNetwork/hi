//! Interactive best-of-N execution.
//!
//! Every candidate runs in an isolated worktree and must emit a successful
//! typed report, produce a non-empty exact diff, and pass an independent
//! parent-side verifier without changing that diff. The selected candidate is
//! applied transactionally and reverified in the destination. All candidate
//! reports, logs, patches, and gate decisions are retained in one aggregate
//! report, including when no candidate wins.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use serde_json::Value;

use crate::candidate_gate::{
    independently_verify_candidate, inspect_child_report, repository_root, same_paths,
    staged_candidate_diff,
};
use crate::candidate_merge::{MergeTimings, apply_candidate_and_reverify};

const CANDIDATE_TIMEOUT_SECS: u64 = 900;
const MAX_VERIFY_CONCURRENCY: usize = 8;

pub struct BestOf<'a> {
    pub exe: &'a Path,
    pub provider: &'a str,
    pub model: &'a str,
    pub base_url: &'a str,
    pub api_key: &'a str,
    pub verify: &'a str,
    pub prompt: &'a str,
    pub candidates: u32,
    pub max_steps: Option<u32>,
    pub max_verify: u32,
    pub workspace_root: &'a Path,
    pub state_root: &'a Path,
    /// User-requested aggregate report path. A private artifact copy is always
    /// retained as well.
    pub report: Option<&'a Path>,
    /// Optional configured target roster. When absent, legacy best-of uses the
    /// current provider/model with varied temperature.
    pub targets: Option<&'a [BestOfTarget]>,
    pub max_concurrency: usize,
    /// Review-only races leave the candidate patch in artifacts for the TUI
    /// instead of applying it immediately.
    pub apply: bool,
    pub fuzz: Option<&'a hi_race::FuzzConfig>,
    pub expected_workspace_digest: Option<&'a str>,
}

#[derive(Clone, Debug)]
pub(crate) struct BestOfTarget {
    pub name: String,
    pub provider: String,
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    pub priority: u32,
}

#[derive(Debug)]
struct CandidateExecution {
    index: u32,
    worktree: PathBuf,
    temperature: f32,
    report_path: PathBuf,
    log_path: PathBuf,
    process_succeeded: bool,
    process_status: String,
    typed_child_succeeded: bool,
    child_gate_reason: String,
    reported_changes: Vec<String>,
    child_review: Option<String>,
    child_report: Option<Value>,
    model_queue_ms: u128,
    wall_clock_ms: u128,
    base_revision: String,
    target_name: String,
    target_priority: u32,
}

#[derive(Debug, Serialize)]
struct CandidateAggregate {
    index: u32,
    temperature: f32,
    process_succeeded: bool,
    process_status: String,
    typed_child_succeeded: bool,
    child_gate_reason: String,
    child_review: Option<String>,
    reported_changes: Vec<String>,
    actual_changes: Vec<String>,
    diff_nonempty: bool,
    report_matches_diff: bool,
    parent_verification: String,
    verification_ms: u128,
    eligible: bool,
    selected: bool,
    application_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    application_changes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    application_timings: Option<MergeTimings>,
    report_path: String,
    log_path: String,
    patch_path: String,
    model_queue_ms: u128,
    wall_clock_ms: u128,
    changed_lines: u64,
    target_name: String,
    target_priority: u32,
    #[serde(default)]
    verify: Vec<hi_race::StageResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fuzz: Option<hi_race::StageResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    child_report: Option<Value>,
}

#[derive(Debug, Serialize)]
struct AggregateReport<'a> {
    schema_version: u32,
    report_kind: &'static str,
    status: &'a str,
    verifier: &'a str,
    provider: &'a str,
    requested_model: &'a str,
    base_revision: &'a str,
    candidate_count: u32,
    setup_wall_clock_ms: u128,
    parallel_wall_clock_ms: u128,
    latency_percentiles: LatencyPercentiles,
    selected_candidate: Option<u32>,
    candidates: &'a [CandidateAggregate],
}

#[derive(Debug, Serialize)]
struct LatencyPercentiles {
    samples: usize,
    p50_ms: u128,
    p95_ms: u128,
}

struct CandidateSlots {
    available: Mutex<usize>,
    released: Condvar,
}

struct CandidatePermit {
    slots: Arc<CandidateSlots>,
}

impl CandidateSlots {
    fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            available: Mutex::new(limit.max(1)),
            released: Condvar::new(),
        })
    }

    fn acquire(self: &Arc<Self>) -> CandidatePermit {
        let mut available = self.available.lock().unwrap_or_else(|p| p.into_inner());
        while *available == 0 {
            available = self
                .released
                .wait(available)
                .unwrap_or_else(|p| p.into_inner());
        }
        *available -= 1;
        CandidatePermit {
            slots: Arc::clone(self),
        }
    }
}

impl Drop for CandidatePermit {
    fn drop(&mut self) {
        let mut available = self
            .slots
            .available
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        *available += 1;
        self.slots.released.notify_one();
    }
}

fn latency_percentiles(samples: impl IntoIterator<Item = u128>) -> LatencyPercentiles {
    let mut samples = samples.into_iter().collect::<Vec<_>>();
    samples.sort_unstable();
    let nearest_rank = |percent: usize| {
        if samples.is_empty() {
            return 0;
        }
        let rank = (samples.len() * percent).div_ceil(100).saturating_sub(1);
        samples[rank.min(samples.len() - 1)]
    };
    LatencyPercentiles {
        samples: samples.len(),
        p50_ms: nearest_rank(50),
        p95_ms: nearest_rank(95),
    }
}

/// Returns `Ok(true)` when a candidate was applied, `Ok(false)` when all
/// candidates were validly evaluated but none won, and `Err` for setup/internal
/// infrastructure failures.
pub fn run(opts: &BestOf) -> Result<bool> {
    ensure!(
        !opts.verify.trim().is_empty(),
        "--best-of requires a resolved non-empty verification pipeline"
    );
    ensure!(
        opts.candidates > 0,
        "--best-of requires at least one candidate"
    );
    let workspace_root = canonical_directory(opts.workspace_root, "best-of workspace root")?;
    let state_root = canonical_directory(opts.state_root, "best-of state root")?;
    ensure!(
        state_root != workspace_root && !workspace_root.starts_with(&state_root),
        "best-of state root must not equal or contain the workspace root"
    );
    let repository = repository_root(&workspace_root)?
        .canonicalize()
        .context("canonicalizing best-of repository root")?;
    let workspace_relative = workspace_root
        .strip_prefix(&repository)
        .context("best-of workspace is outside its repository root")?
        .to_path_buf();
    if !hi_tools::worktree::in_git_repo(&workspace_root) {
        bail!("--best-of requires a git repository (candidates run in worktrees)");
    }
    let base_revision = resolve_revision(&repository, "HEAD")?;
    let workspace_snapshot = hi_race::capture_workspace_snapshot(&workspace_root)
        .context("capturing the race workspace snapshot")?;
    if working_tree_dirty(&workspace_root) {
        eprintln!(
            "\x1b[2mrace snapshot: preserving uncommitted changes for every candidate\x1b[0m"
        );
    }

    let art_dir = artifacts_dir(&state_root);
    std::fs::create_dir_all(&art_dir)
        .with_context(|| format!("creating best-of artifacts at {}", art_dir.display()))?;
    let aggregate_path = art_dir.join("aggregate.report.json");

    let setup_started = Instant::now();
    let worktrees = std::thread::scope(|scope| {
        (0..opts.candidates)
            .map(|index| {
                let repository = &repository;
                let base_revision = &base_revision;
                let state_root = &state_root;
                let workspace_relative = &workspace_relative;
                let snapshot = &workspace_snapshot;
                scope.spawn(move || -> Result<(u32, PathBuf, f32, String)> {
                    let _setup_lease = crate::resource_governor::acquire(
                        state_root,
                        crate::resource_governor::ResourceClass::Setup,
                        Duration::from_secs(120),
                    )?;
                    let temperature = temperature_for(index, opts.candidates);
                    let worktree = hi_tools::worktree::worktree_path("bestof", index);
                    hi_tools::worktree::add_worktree(repository, &worktree, base_revision)?;
                    let candidate_root = worktree.join(workspace_relative);
                    snapshot.materialize_into(&candidate_root)?;
                    let candidate_base =
                        materialize_snapshot_base(&worktree, base_revision, snapshot)?;
                    Ok((index, worktree, temperature, candidate_base))
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("best-of setup worker panicked"))
            .collect::<Result<Vec<_>>>()
    });
    let mut worktrees = worktrees?;
    worktrees.sort_by_key(|(index, _, _, _)| *index);
    let setup_wall_clock_ms = setup_started.elapsed().as_millis();

    println!(
        "\x1b[36m── running {} candidates in parallel ──────────────────\x1b[0m",
        opts.candidates
    );
    let cleanup_paths = worktrees
        .iter()
        .map(|(_, worktree, _, _)| worktree.clone())
        .collect::<Vec<_>>();

    let (completion_tx, completion_rx) = std::sync::mpsc::channel();
    let candidate_slots = CandidateSlots::new(opts.max_concurrency);
    let handles = worktrees
        .iter()
        .map(|(index, worktree, temperature, candidate_base)| {
            let index = *index;
            let worktree = worktree.join(&workspace_relative);
            let temperature = *temperature;
            let candidate_base = candidate_base.clone();
            let exe = opts.exe.to_path_buf();
            let verify = opts.verify.to_string();
            let prompt = opts.prompt.to_string();
            let max_steps = opts.max_steps;
            let max_verify = opts.max_verify;
            let candidate_state_root = state_root.clone();
            let configured_target = opts
                .targets
                .and_then(|targets| targets.get(index as usize % targets.len()))
                .cloned();
            let (provider, model, base_url, api_key, target_name, target_priority) =
                configured_target
                    .as_ref()
                    .map(|target| {
                        (
                            target.provider.clone(),
                            target.model.clone(),
                            target.base_url.clone(),
                            target.api_key.clone(),
                            target.name.clone(),
                            target.priority,
                        )
                    })
                    .unwrap_or_else(|| {
                        (
                            opts.provider.to_string(),
                            opts.model.to_string(),
                            opts.base_url.to_string(),
                            opts.api_key.to_string(),
                            format!("candidate-{index}"),
                            index,
                        )
                    });
            let report_path = art_dir.join(format!("candidate-{index}.report.json"));
            let log_path = art_dir.join(format!("candidate-{index}.log"));
            let completion_tx = completion_tx.clone();
            let candidate_slots = Arc::clone(&candidate_slots);
            (
                index,
                worktree.clone(),
                temperature,
                report_path.clone(),
                log_path.clone(),
                std::thread::spawn(move || {
                    let _candidate_permit = candidate_slots.acquire();
                    let thread_opts = BestOf {
                        exe: &exe,
                        provider: &provider,
                        model: &model,
                        base_url: &base_url,
                        api_key: &api_key,
                        verify: &verify,
                        prompt: &prompt,
                        candidates: 0,
                        max_steps,
                        max_verify,
                        workspace_root: &worktree,
                        state_root: &candidate_state_root,
                        report: None,
                        targets: None,
                        max_concurrency: 1,
                        apply: false,
                        fuzz: None,
                        expected_workspace_digest: None,
                    };
                    let model_queue_started = Instant::now();
                    let model_lease = crate::resource_governor::acquire(
                        &candidate_state_root,
                        crate::resource_governor::ResourceClass::Model,
                        Duration::from_secs(120),
                    );
                    let model_queue_ms = model_queue_started.elapsed().as_millis();
                    let mut result = match model_lease {
                        Ok(lease) => {
                            let mut result = run_candidate(
                                &thread_opts,
                                index,
                                &worktree,
                                temperature,
                                &report_path,
                                &log_path,
                            );
                            drop(lease);
                            result.model_queue_ms = model_queue_ms;
                            result
                        }
                        Err(error) => failed_execution(
                            index,
                            &worktree,
                            temperature,
                            &report_path,
                            &log_path,
                            "model_queue_failed",
                            format!("candidate could not acquire model capacity: {error:#}"),
                            model_queue_ms,
                        ),
                    };
                    // Keep roster identity and the exact snapshot base on all
                    // paths, including capacity/setup failures. This makes a
                    // failed candidate auditable and keeps ranking/reporting
                    // independent of which failure happened first.
                    result.base_revision = candidate_base;
                    result.target_name = target_name;
                    result.target_priority = target_priority;
                    println!(
                        "\x1b[36m── candidate {} (temp {temperature:.1}) finished ─────────────────\x1b[0m",
                        index + 1
                    );
                    let _ = completion_tx.send(result);
                }),
            )
        })
        .collect::<Vec<_>>();
    drop(completion_tx);

    let mut executions = completion_rx.into_iter().collect::<Vec<_>>();
    for (index, worktree, temperature, report_path, log_path, handle) in handles {
        if handle.join().is_err() {
            executions.push(CandidateExecution {
                index,
                worktree,
                temperature,
                report_path,
                log_path,
                process_succeeded: false,
                process_status: "thread_panicked".into(),
                typed_child_succeeded: false,
                child_gate_reason: "candidate thread panicked".into(),
                reported_changes: Vec::new(),
                child_review: None,
                child_report: None,
                model_queue_ms: 0,
                wall_clock_ms: 0,
                base_revision: base_revision.clone(),
                target_name: format!("candidate-{index}"),
                target_priority: index,
            });
        }
    }
    executions.sort_by_key(|execution| execution.index);

    // Prepare every candidate in index order. Parent verification is deferred
    // so expensive verifier processes can run with bounded parallelism.
    let mut aggregates = Vec::with_capacity(executions.len());
    let mut verification_positions = Vec::new();
    for execution in &executions {
        let patch_path = art_dir.join(format!("candidate-{}.patch", execution.index));
        let mut aggregate = CandidateAggregate {
            index: execution.index,
            temperature: execution.temperature,
            process_succeeded: execution.process_succeeded,
            process_status: execution.process_status.clone(),
            typed_child_succeeded: execution.typed_child_succeeded,
            child_gate_reason: execution.child_gate_reason.clone(),
            child_review: execution.child_review.clone(),
            reported_changes: execution.reported_changes.clone(),
            actual_changes: Vec::new(),
            diff_nonempty: false,
            report_matches_diff: false,
            parent_verification: "not_run".into(),
            verification_ms: 0,
            eligible: false,
            selected: false,
            application_status: "not_attempted".into(),
            application_changes: None,
            application_timings: None,
            report_path: execution.report_path.display().to_string(),
            log_path: execution.log_path.display().to_string(),
            patch_path: patch_path.display().to_string(),
            model_queue_ms: execution.model_queue_ms,
            wall_clock_ms: execution.wall_clock_ms,
            changed_lines: 0,
            target_name: execution.target_name.clone(),
            target_priority: execution.target_priority,
            fuzz: None,
            verify: Vec::new(),
            child_report: execution.child_report.clone(),
        };

        let diff = match staged_candidate_diff(&execution.worktree, &execution.base_revision) {
            Ok(diff) => diff,
            Err(error) => {
                aggregate.parent_verification = format!("diff_error: {error:#}");
                aggregates.push(aggregate);
                continue;
            }
        };
        let _ = std::fs::write(&patch_path, &diff.patch);
        aggregate.actual_changes = diff.display_paths.clone();
        aggregate.diff_nonempty = !diff.paths.is_empty();
        aggregate.report_matches_diff =
            same_paths(&aggregate.reported_changes, &aggregate.actual_changes);

        if !execution.typed_child_succeeded {
            aggregates.push(aggregate);
            continue;
        }
        if !aggregate.diff_nonempty {
            aggregate.parent_verification = "not_run: empty_diff".into();
            aggregates.push(aggregate);
            continue;
        }
        if !aggregate.report_matches_diff {
            aggregate.parent_verification = "not_run: report_diff_mismatch".into();
            aggregates.push(aggregate);
            continue;
        }

        verification_positions.push(aggregates.len());
        aggregates.push(aggregate);
    }

    let verification_results =
        bounded_ordered_map(&verification_positions, verify_concurrency(), |position| {
            let started = Instant::now();
            (
                independently_verify_candidate(
                    &executions[*position].worktree,
                    &executions[*position].base_revision,
                    opts.verify,
                ),
                started.elapsed().as_millis(),
            )
        });
    for (position, (verified, verification_ms)) in
        verification_positions.into_iter().zip(verification_results)
    {
        let aggregate = &mut aggregates[position];
        let patch_path = Path::new(&aggregate.patch_path);
        match verified {
            Ok(verified) => {
                if !same_paths(&aggregate.reported_changes, &verified.display_paths) {
                    aggregate.parent_verification = "failed: verified_diff_report_mismatch".into();
                } else {
                    // Persist the exact revision that passed the parent-side
                    // verifier (the helper rejects verifier-induced mutations).
                    if let Err(error) = std::fs::write(patch_path, &verified.patch) {
                        aggregate.parent_verification = format!("artifact_error: {error}");
                    } else {
                        aggregate.parent_verification = "passed".into();
                        aggregate.verify = vec![hi_race::StageResult {
                            name: "parent-verify".into(),
                            command: opts.verify.to_string(),
                            passed: true,
                            timed_out: false,
                            duration_ms: verification_ms,
                            detail: String::new(),
                        }];
                        aggregate.changed_lines =
                            hi_race::changed_lines(&executions[position].worktree);
                        if let Some(fuzz) = opts.fuzz {
                            let result = hi_race::run_fuzz(&executions[position].worktree, fuzz);
                            aggregate.fuzz = Some(result.clone());
                            aggregate.eligible = result.passed;
                            if !result.passed {
                                aggregate.parent_verification = if result.timed_out {
                                    "passed; fuzz timed out".into()
                                } else {
                                    "passed; fuzz failed".into()
                                };
                            }
                        } else {
                            aggregate.eligible = true;
                        }
                    }
                }
            }
            Err(error) => {
                aggregate.parent_verification = format!("failed: {error:#}");
                aggregate.verify = vec![hi_race::StageResult {
                    name: "parent-verify".into(),
                    command: opts.verify.to_string(),
                    passed: false,
                    timed_out: false,
                    duration_ms: verification_ms,
                    detail: error.to_string(),
                }];
            }
        }
        aggregate.verification_ms = verification_ms;
    }

    // Select the smallest verified candidate using the shared race ordering.
    // Application also performs destination verification; a failure is sealed-
    // rolled back and the overall run fails rather than choosing an unchecked
    // patch.
    let ranking_candidates = aggregates
        .iter()
        .map(|candidate| hi_race::CandidateReport {
            candidate_id: format!("candidate-{}", candidate.index),
            target: hi_race::RaceTarget {
                name: candidate.target_name.clone(),
                profile: String::new(),
                model: String::new(),
                priority: candidate.target_priority,
            },
            state: if candidate.eligible {
                hi_race::CandidateState::Passed
            } else {
                hi_race::CandidateState::Failed
            },
            process_succeeded: candidate.process_succeeded,
            report_matches_diff: candidate.report_matches_diff,
            actual_changes: candidate.actual_changes.clone(),
            changed_lines: candidate.changed_lines,
            verify: candidate.verify.clone(),
            fuzz: candidate.fuzz.clone(),
            wall_clock_ms: candidate.wall_clock_ms,
            cost_microusd: None,
            artifact_ref: Some(candidate.patch_path.clone()),
            failure_reason: Some(candidate.parent_verification.clone()),
        })
        .collect::<Vec<_>>();
    let selected_index = hi_race::select_winner(&ranking_candidates).and_then(|candidate_id| {
        aggregates
            .iter()
            .position(|candidate| format!("candidate-{}", candidate.index) == candidate_id)
    });
    let mut selected_candidate = None;
    let status;
    let mut terminal_error = None;
    if let Some(position) = selected_index {
        let execution = &executions[position];
        if !opts.apply {
            selected_candidate = Some(execution.index);
            aggregates[position].selected = true;
            aggregates[position].application_status = "awaiting_review".into();
            status = "ready";
        } else if let Some(expected) = opts.expected_workspace_digest
            && hi_race::capture_workspace_snapshot(&workspace_root)
                .map(|snapshot| snapshot.digest != expected)
                .unwrap_or(true)
        {
            status = "workspace_conflict";
            terminal_error =
                Some("workspace changed while the race was running; winner was not applied".into());
        } else {
            match apply_candidate_and_reverify(
                &execution.worktree,
                &execution.base_revision,
                &workspace_root,
                &state_root,
                opts.verify,
            ) {
                Ok(changes) => {
                    selected_candidate = Some(execution.index);
                    aggregates[position].selected = true;
                    aggregates[position].application_status =
                        "applied_and_destination_verified".into();
                    aggregates[position].application_changes = Some(changes.changes);
                    aggregates[position].application_timings = Some(changes.timings);
                    status = "completed";
                    println!(
                        "\x1b[32m✓ applied candidate {} after destination verification\x1b[0m",
                        execution.index + 1
                    );
                }
                Err(error) => {
                    aggregates[position].application_status = format!("failed: {error:#}");
                    status = "application_failed";
                    terminal_error = Some(format!(
                        "winning candidate failed transactional destination application: {error:#}"
                    ));
                }
            }
        }
    } else {
        status = "no_winner";
        terminal_error = Some(format!(
            "no candidate satisfied the typed outcome, non-empty diff, independent verification, and fuzz gates (tried {})",
            opts.candidates
        ));
    }

    let parallel_wall_clock_ms = aggregates
        .iter()
        .map(|candidate| candidate.wall_clock_ms)
        .max()
        .unwrap_or(0);
    let latency_percentiles = latency_percentiles(
        aggregates
            .iter()
            .map(|candidate| candidate.wall_clock_ms + candidate.verification_ms),
    );
    crate::orchestration_metrics::record(
        &state_root,
        "best_of",
        setup_wall_clock_ms + parallel_wall_clock_ms,
        terminal_error.is_none(),
    );
    let aggregate = AggregateReport {
        schema_version: 2,
        report_kind: "best_of",
        status,
        verifier: opts.verify,
        provider: opts.provider,
        requested_model: opts.model,
        base_revision: &base_revision,
        candidate_count: opts.candidates,
        setup_wall_clock_ms,
        parallel_wall_clock_ms,
        latency_percentiles,
        selected_candidate,
        candidates: &aggregates,
    };
    let report_result = write_aggregate_report(&aggregate_path, &aggregate).and_then(|_| {
        if let Some(requested) = opts.report
            && requested != aggregate_path
        {
            if let Some(parent) = requested.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("creating requested report directory {}", parent.display())
                })?;
            }
            write_aggregate_report(requested, &aggregate)?;
        }
        Ok(())
    });

    hi_tools::worktree::cleanup(&repository, &cleanup_paths);
    print_candidate_summary(&art_dir, &aggregates);
    report_result?;

    if let Some(error) = terminal_error {
        eprintln!("\x1b[31m✗ {error}\x1b[0m");
        return Ok(false);
    }
    Ok(true)
}

pub(crate) fn resolve_revision(root: &Path, revision: &str) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--verify", revision])
        .output()
        .with_context(|| format!("resolving best-of base revision {revision}"))?;
    ensure!(
        output.status.success(),
        "could not resolve best-of base revision {revision}: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let revision = String::from_utf8(output.stdout).context("base revision is not valid UTF-8")?;
    let revision = revision.trim();
    ensure!(!revision.is_empty(), "resolved best-of base is empty");
    Ok(revision.to_string())
}

pub(crate) fn materialize_snapshot_base(
    worktree: &Path,
    base_revision: &str,
    snapshot: &hi_race::WorkspaceSnapshot,
) -> Result<String> {
    if snapshot.tracked_patch.is_empty() && snapshot.untracked_files.is_empty() {
        return Ok(base_revision.to_string());
    }
    let output = Command::new("git")
        .current_dir(worktree)
        .args(["add", "-A", "--", "."])
        .output()
        .context("staging the race workspace snapshot")?;
    ensure!(
        output.status.success(),
        "could not stage the race workspace snapshot: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let output = Command::new("git")
        .current_dir(worktree)
        .args(["-c", "user.name=hi race", "-c", "user.email=hi@localhost"])
        .args(["commit", "--no-verify", "-m", "hi race workspace snapshot"])
        .output()
        .context("committing the race workspace snapshot")?;
    ensure!(
        output.status.success(),
        "could not commit the race workspace snapshot: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    resolve_revision(worktree, "HEAD")
}

fn run_candidate(
    opts: &BestOf,
    index: u32,
    worktree: &Path,
    temperature: f32,
    report_path: &Path,
    log_path: &Path,
) -> CandidateExecution {
    let started = Instant::now();
    let _ = std::fs::remove_file(report_path);
    let _ = std::fs::remove_file(log_path);
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut arguments = vec![
        OsString::from("--subagent"),
        OsString::from("--no-save"),
        OsString::from("--provider"),
        OsString::from(opts.provider),
        OsString::from("--model"),
        OsString::from(opts.model),
        OsString::from("--base-url"),
        OsString::from(opts.base_url),
        OsString::from("--verify"),
        OsString::from(opts.verify),
        OsString::from("--temperature"),
        OsString::from(temperature.to_string()),
        OsString::from("--max-verify-repairs"),
        OsString::from(opts.max_verify.to_string()),
        OsString::from("--review"),
        OsString::from("always"),
        OsString::from("--report"),
        report_path.as_os_str().to_os_string(),
    ];
    if let Some(max_steps) = opts.max_steps {
        arguments.push("--max-steps".into());
        arguments.push(max_steps.to_string().into());
    }
    arguments.push(opts.prompt.into());

    let process = match crate::child_process::run(
        worktree,
        opts.exe,
        arguments,
        vec![
            ("HI_FORCE_API_KEY".into(), opts.api_key.into()),
            ("HI_API_KEY".into(), opts.api_key.into()),
        ],
        Duration::from_secs(candidate_timeout_secs()),
        log_path,
    ) {
        Ok(process) => process,
        Err(error) => {
            let message = format!("failed to launch candidate hi: {error}");
            let _ = std::fs::write(log_path, &message);
            return failed_execution(
                index,
                worktree,
                temperature,
                report_path,
                log_path,
                "launch_failed",
                message,
                started.elapsed().as_millis(),
            );
        }
    };
    let process_succeeded = process.status == hi_tools::ToolStatus::Succeeded;
    let process_status = match process.status {
        hi_tools::ToolStatus::Succeeded | hi_tools::ToolStatus::Failed => process
            .outcome
            .exit_code
            .map(|code| format!("exit_{code}"))
            .unwrap_or_else(|| format!("{:?}", process.status).to_ascii_lowercase()),
        hi_tools::ToolStatus::TimedOut => "timed_out".into(),
        status => format!("{status:?}").to_ascii_lowercase(),
    };
    let raw_report = std::fs::read_to_string(report_path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok());

    let (typed_child_succeeded, child_gate_reason, reported_changes, child_review) =
        if !process_succeeded {
            (
                false,
                format!("candidate process failed ({process_status})"),
                Vec::new(),
                None,
            )
        } else {
            match inspect_child_report(report_path) {
                Ok(gate) => (
                    true,
                    "schema-v2 completed outcome with current-revision verification".into(),
                    gate.changed_files,
                    Some(gate.review_status),
                ),
                Err(error) => (
                    false,
                    format!("typed child gate failed: {error:#}"),
                    Vec::new(),
                    None,
                ),
            }
        };

    CandidateExecution {
        index,
        worktree: worktree.to_path_buf(),
        temperature,
        report_path: report_path.to_path_buf(),
        log_path: log_path.to_path_buf(),
        process_succeeded,
        process_status,
        typed_child_succeeded,
        child_gate_reason,
        reported_changes,
        child_review,
        child_report: raw_report,
        model_queue_ms: 0,
        wall_clock_ms: started.elapsed().as_millis(),
        base_revision: String::new(),
        target_name: format!("candidate-{index}"),
        target_priority: index,
    }
}

fn candidate_timeout_secs() -> u64 {
    std::env::var("HI_BEST_OF_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .unwrap_or(CANDIDATE_TIMEOUT_SECS)
}

fn configured_verify_concurrency(value: Option<&str>, default: usize) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(1, MAX_VERIFY_CONCURRENCY)
}

fn verify_concurrency() -> usize {
    let default = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().div_ceil(2))
        .unwrap_or(1);
    let configured = std::env::var("HI_BESTOF_VERIFY_CONCURRENCY").ok();
    configured_verify_concurrency(configured.as_deref(), default)
}

fn bounded_ordered_map<T, R, F>(items: &[T], concurrency: usize, operation: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    let mut results = Vec::with_capacity(items.len());
    for chunk in items.chunks(concurrency.max(1)) {
        std::thread::scope(|scope| {
            let handles = chunk
                .iter()
                .map(|item| scope.spawn(|| operation(item)))
                .collect::<Vec<_>>();
            for handle in handles {
                results.push(handle.join().expect("parent verification thread panicked"));
            }
        });
    }
    results
}

#[allow(clippy::too_many_arguments)]
fn failed_execution(
    index: u32,
    worktree: &Path,
    temperature: f32,
    report_path: &Path,
    log_path: &Path,
    process_status: &str,
    reason: String,
    wall_clock_ms: u128,
) -> CandidateExecution {
    CandidateExecution {
        index,
        worktree: worktree.to_path_buf(),
        temperature,
        report_path: report_path.to_path_buf(),
        log_path: log_path.to_path_buf(),
        process_succeeded: false,
        process_status: process_status.into(),
        typed_child_succeeded: false,
        child_gate_reason: reason,
        reported_changes: Vec::new(),
        child_review: None,
        child_report: None,
        model_queue_ms: 0,
        wall_clock_ms,
        base_revision: String::new(),
        target_name: format!("candidate-{index}"),
        target_priority: index,
    }
}

fn write_aggregate_report(path: &Path, report: &AggregateReport<'_>) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("writing aggregate best-of report {}", path.display()))
}

fn artifacts_dir(state_root: &Path) -> PathBuf {
    let pid = std::process::id();
    state_root.join("bestof-artifacts").join(pid.to_string())
}

fn print_candidate_summary(art_dir: &Path, candidates: &[CandidateAggregate]) {
    println!(
        "\x1b[36m── candidate artifacts: {} ──────────────────\x1b[0m",
        art_dir.display()
    );
    for candidate in candidates {
        println!(
            "   candidate {}/{}: child={} · parent={} · {} files · application={}",
            candidate.index + 1,
            candidates.len(),
            if candidate.typed_child_succeeded {
                "passed"
            } else {
                "failed"
            },
            candidate.parent_verification,
            candidate.actual_changes.len(),
            candidate.application_status,
        );
    }
}

/// Spread candidate temperatures across [0.2, 1.0] for diversity.
fn temperature_for(index: u32, count: u32) -> f32 {
    if count <= 1 {
        return 0.2;
    }
    0.2 + (index as f32) * (0.8 / (count - 1) as f32)
}

fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf> {
    let path = path
        .canonicalize()
        .with_context(|| format!("canonicalizing {label} {}", path.display()))?;
    ensure!(
        path.is_dir(),
        "{label} is not a directory: {}",
        path.display()
    );
    Ok(path)
}

fn working_tree_dirty(root: &Path) -> bool {
    Command::new("git")
        .current_dir(root)
        .args(["status", "--porcelain", "--", "."])
        .output()
        .map(|output| !output.stdout.is_empty())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    fn test_opts<'a>(exe: &'a Path, verify: &'a str) -> BestOf<'a> {
        BestOf {
            exe,
            provider: "openai",
            model: "test-model",
            base_url: "http://127.0.0.1:9/v1",
            api_key: "test-key",
            verify,
            prompt: "do the thing",
            candidates: 1,
            max_steps: Some(1),
            max_verify: 1,
            workspace_root: Path::new("/"),
            state_root: Path::new("/tmp"),
            report: None,
            targets: None,
            max_concurrency: 1,
            apply: true,
            fuzz: None,
            expected_workspace_digest: None,
        }
    }

    fn temp_file(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "hi-bestof-{label}-{}-candidate-0.report.json",
            std::process::id()
        ))
    }

    #[test]
    fn run_candidate_rejects_nonzero_exit_even_without_a_report() {
        let exe = Path::new("/bin/false");
        if !exe.exists() {
            return;
        }
        let opts = test_opts(exe, "true");
        let report = temp_file("failure");
        let log = report.with_extension("log");
        let workspace = std::env::current_dir().unwrap().canonicalize().unwrap();
        let execution = run_candidate(&opts, 0, &workspace, 0.2, &report, &log);
        assert!(!execution.process_succeeded);
        assert!(!execution.typed_child_succeeded);
        assert!(log.exists(), "candidate log must be persisted");
        let _ = std::fs::remove_file(report);
        let _ = std::fs::remove_file(log);
    }

    #[test]
    fn exit_zero_without_typed_report_is_not_eligible() {
        let exe = Path::new("/bin/true");
        if !exe.exists() {
            return;
        }
        let opts = test_opts(exe, "true");
        let report = temp_file("missing-report");
        let log = report.with_extension("log");
        let workspace = std::env::current_dir().unwrap().canonicalize().unwrap();
        let execution = run_candidate(&opts, 0, &workspace, 0.2, &report, &log);
        assert!(execution.process_succeeded);
        assert!(!execution.typed_child_succeeded);
        assert!(
            execution
                .child_gate_reason
                .contains("typed child gate failed")
        );
        let _ = std::fs::remove_file(report);
        let _ = std::fs::remove_file(log);
    }

    #[test]
    fn empty_verifier_is_rejected_before_candidate_setup() {
        let opts = test_opts(Path::new("/bin/true"), "  ");
        let error = run(&opts).expect_err("empty verifier must be a usage error");
        assert!(format!("{error:#}").contains("resolved non-empty"));
    }

    #[test]
    fn bounded_map_preserves_order_and_limits_parallelism() {
        let items = [0usize, 1, 2, 3];
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let first_wave = Arc::new(Barrier::new(2));
        let results = bounded_ordered_map(&items, 2, |item| {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(current, Ordering::SeqCst);
            if *item < 2 {
                first_wave.wait();
            }
            std::thread::sleep(Duration::from_millis((3 - item) as u64));
            active.fetch_sub(1, Ordering::SeqCst);
            item * 10
        });

        assert_eq!(results, [0, 10, 20, 30]);
        assert_eq!(peak.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let values = latency_percentiles([1, 2, 3, 4, 100]);
        assert_eq!(values.samples, 5);
        assert_eq!(values.p50_ms, 3);
        assert_eq!(values.p95_ms, 100);
    }

    #[test]
    fn verify_concurrency_clamps_and_falls_back() {
        assert_eq!(
            configured_verify_concurrency(Some("999"), 2),
            MAX_VERIFY_CONCURRENCY
        );
        assert_eq!(configured_verify_concurrency(Some("0"), 2), 1);
        assert_eq!(configured_verify_concurrency(Some("invalid"), 3), 3);
        assert_eq!(configured_verify_concurrency(None, 3), 3);
    }
}
