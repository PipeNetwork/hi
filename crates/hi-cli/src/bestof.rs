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
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use serde_json::Value;

use crate::candidate_gate::{
    independently_verify_candidate, inspect_child_report, repository_root, same_paths,
    staged_candidate_diff,
};
use crate::candidate_merge::apply_candidate_and_reverify;

const CANDIDATE_TIMEOUT_SECS: u64 = 900;

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
    wall_clock_ms: u128,
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
    eligible: bool,
    selected: bool,
    application_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    application_changes: Option<Vec<String>>,
    report_path: String,
    log_path: String,
    patch_path: String,
    wall_clock_ms: u128,
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
    parallel_wall_clock_ms: u128,
    selected_candidate: Option<u32>,
    candidates: &'a [CandidateAggregate],
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
    if working_tree_dirty(&workspace_root) {
        eprintln!(
            "\x1b[33mwarning: working tree has uncommitted changes; candidates run from HEAD, and transactional merge conflicts are rejected\x1b[0m"
        );
    }

    let art_dir = artifacts_dir(&state_root);
    std::fs::create_dir_all(&art_dir)
        .with_context(|| format!("creating best-of artifacts at {}", art_dir.display()))?;
    let aggregate_path = art_dir.join("aggregate.report.json");

    // Create every worktree before launching a provider call. A setup failure
    // therefore has no partially running candidate fleet.
    let mut worktrees: Vec<(u32, PathBuf, f32)> = Vec::new();
    for index in 0..opts.candidates {
        let temperature = temperature_for(index, opts.candidates);
        let worktree = hi_tools::worktree::worktree_path("bestof", index);
        if let Err(error) = hi_tools::worktree::add_worktree(&repository, &worktree, &base_revision)
        {
            cleanup_worktrees(&repository, &worktrees);
            return Err(error);
        }
        worktrees.push((index, worktree, temperature));
    }

    println!(
        "\x1b[36m── running {} candidates in parallel ──────────────────\x1b[0m",
        opts.candidates
    );
    let cleanup_paths = worktrees
        .iter()
        .map(|(_, worktree, _)| worktree.clone())
        .collect::<Vec<_>>();

    let handles = worktrees
        .iter()
        .map(|(index, worktree, temperature)| {
            let index = *index;
            let worktree = worktree.join(&workspace_relative);
            let temperature = *temperature;
            let exe = opts.exe.to_path_buf();
            let provider = opts.provider.to_string();
            let model = opts.model.to_string();
            let base_url = opts.base_url.to_string();
            let api_key = opts.api_key.to_string();
            let verify = opts.verify.to_string();
            let prompt = opts.prompt.to_string();
            let max_steps = opts.max_steps;
            let max_verify = opts.max_verify;
            let candidate_state_root = state_root.clone();
            let report_path = art_dir.join(format!("candidate-{index}.report.json"));
            let log_path = art_dir.join(format!("candidate-{index}.log"));
            (
                index,
                worktree.clone(),
                temperature,
                report_path.clone(),
                log_path.clone(),
                std::thread::spawn(move || {
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
                    };
                    let result = run_candidate(
                        &thread_opts,
                        index,
                        &worktree,
                        temperature,
                        &report_path,
                        &log_path,
                    );
                    println!(
                        "\x1b[36m── candidate {} (temp {temperature:.1}) finished ─────────────────\x1b[0m",
                        index + 1
                    );
                    result
                }),
            )
        })
        .collect::<Vec<_>>();

    let mut executions = Vec::with_capacity(handles.len());
    for (index, worktree, temperature, report_path, log_path, handle) in handles {
        match handle.join() {
            Ok(execution) => executions.push(execution),
            Err(_) => executions.push(CandidateExecution {
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
                wall_clock_ms: 0,
            }),
        }
    }
    executions.sort_by_key(|execution| execution.index);

    // Evaluate every candidate, even after finding an eligible one, so the
    // aggregate describes all N attempts rather than one representative.
    let mut aggregates = Vec::with_capacity(executions.len());
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
            eligible: false,
            selected: false,
            application_status: "not_attempted".into(),
            application_changes: None,
            report_path: execution.report_path.display().to_string(),
            log_path: execution.log_path.display().to_string(),
            patch_path: patch_path.display().to_string(),
            wall_clock_ms: execution.wall_clock_ms,
            child_report: execution.child_report.clone(),
        };

        let diff = match staged_candidate_diff(&execution.worktree, &base_revision) {
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

        match independently_verify_candidate(&execution.worktree, &base_revision, opts.verify) {
            Ok(verified) => {
                if !same_paths(&aggregate.reported_changes, &verified.display_paths) {
                    aggregate.parent_verification = "failed: verified_diff_report_mismatch".into();
                } else {
                    // Persist the exact revision that passed the parent-side
                    // verifier (the helper rejects verifier-induced mutations).
                    if let Err(error) = std::fs::write(&patch_path, &verified.patch) {
                        aggregate.parent_verification = format!("artifact_error: {error}");
                    } else {
                        aggregate.parent_verification = "passed".into();
                        aggregate.eligible = true;
                    }
                }
            }
            Err(error) => {
                aggregate.parent_verification = format!("failed: {error:#}");
            }
        }
        aggregates.push(aggregate);
    }

    // Deterministically select the first eligible candidate. Application also
    // performs destination verification; a failure is sealed-rolled back and
    // the overall best-of run fails rather than silently choosing an unchecked
    // patch.
    let selected_index = aggregates.iter().position(|candidate| candidate.eligible);
    let mut selected_candidate = None;
    let status;
    let mut terminal_error = None;
    if let Some(position) = selected_index {
        let execution = &executions[position];
        match apply_candidate_and_reverify(
            &execution.worktree,
            &base_revision,
            &workspace_root,
            &state_root,
            opts.verify,
        ) {
            Ok(changes) => {
                selected_candidate = Some(execution.index);
                aggregates[position].selected = true;
                aggregates[position].application_status = "applied_and_destination_verified".into();
                aggregates[position].application_changes = Some(changes);
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
    } else {
        status = "no_winner";
        terminal_error = Some(format!(
            "no candidate satisfied the typed outcome, non-empty diff, and independent verification gates (tried {})",
            opts.candidates
        ));
    }

    let parallel_wall_clock_ms = aggregates
        .iter()
        .map(|candidate| candidate.wall_clock_ms)
        .max()
        .unwrap_or(0);
    let aggregate = AggregateReport {
        schema_version: 2,
        report_kind: "best_of",
        status,
        verifier: opts.verify,
        provider: opts.provider,
        requested_model: opts.model,
        base_revision: &base_revision,
        candidate_count: opts.candidates,
        parallel_wall_clock_ms,
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

fn cleanup_worktrees(destination: &Path, worktrees: &[(u32, PathBuf, f32)]) {
    hi_tools::worktree::cleanup(
        destination,
        &worktrees
            .iter()
            .map(|(_, worktree, _)| worktree.clone())
            .collect::<Vec<_>>(),
    );
}

fn resolve_revision(root: &Path, revision: &str) -> Result<String> {
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
        wall_clock_ms: started.elapsed().as_millis(),
    }
}

fn candidate_timeout_secs() -> u64 {
    std::env::var("HI_BEST_OF_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .unwrap_or(CANDIDATE_TIMEOUT_SECS)
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
        wall_clock_ms,
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
}
