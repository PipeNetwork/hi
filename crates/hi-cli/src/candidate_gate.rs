//! Shared eligibility gates for delegate and best-of candidates.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde_json::Value;

const VERIFY_TIMEOUT_SECS: u64 = 120;
const PYCACHE_EXCLUDES: &[&str] = &[
    ":(exclude,glob)**/__pycache__/**",
    ":(exclude,glob)**/*.pyc",
    ":(exclude,glob)**/*.pyo",
];

#[derive(Clone, Debug)]
pub(crate) struct ChildReportGate {
    pub(crate) changed_files: Vec<String>,
    pub(crate) review_status: String,
}

/// Accept only the typed v2 success contract. The caller separately checks the
/// process status and compares these paths with the immutable worktree diff.
pub(crate) fn inspect_child_report(path: &Path) -> Result<ChildReportGate> {
    let raw_text = std::fs::read_to_string(path)
        .with_context(|| format!("reading child report {}", path.display()))?;
    let raw: Value = serde_json::from_str(&raw_text)
        .with_context(|| format!("parsing child report {}", path.display()))?;
    ensure!(
        raw.get("schema_version").and_then(Value::as_u64) == Some(2),
        "child report is not schema v2"
    );
    ensure!(
        raw.pointer("/outcome/status").and_then(Value::as_str) == Some("completed"),
        "child outcome was not completed"
    );
    ensure!(
        raw.pointer("/outcome/verification").and_then(Value::as_str) == Some("passed"),
        "child outcome was not deterministically verified"
    );
    ensure!(
        raw.pointer("/outcome/stop_reason").and_then(Value::as_str) == Some("completed"),
        "child stopped without satisfying its completion contract"
    );
    ensure!(
        raw.pointer("/outcome/verified_workspace_revision")
            .and_then(Value::as_str)
            .is_some_and(|revision| !revision.trim().is_empty()),
        "child pass was not tied to a workspace revision"
    );
    let review_status = raw
        .pointer("/outcome/review")
        .and_then(Value::as_str)
        .context("child outcome omitted independent-review status")?
        .to_string();
    ensure!(
        matches!(review_status.as_str(), "passed" | "unavailable"),
        "child independent review did not pass (status: {review_status})"
    );
    ensure!(
        raw.pointer("/review/status").and_then(Value::as_str) == Some(review_status.as_str()),
        "child report review fields disagree"
    );
    ensure!(
        raw.pointer("/verification/stages")
            .and_then(Value::as_array)
            .is_some_and(|stages| !stages.is_empty()),
        "child report has no resolved verifier"
    );
    ensure!(
        raw.pointer("/verification/status").and_then(Value::as_str) == Some("passed"),
        "child report verification fields disagree"
    );
    ensure!(
        raw.pointer("/outcome/effective_route/model")
            .and_then(Value::as_str)
            .is_some_and(|model| !model.trim().is_empty()),
        "child report omitted its effective model route"
    );
    ensure!(
        raw.get("changes_complete").and_then(Value::as_bool) == Some(true),
        "child report could not reconcile complete exact changes"
    );
    let changed_files = raw
        .pointer("/outcome/changed_files")
        .and_then(Value::as_array)
        .context("child outcome omitted exact changed files")?
        .iter()
        .map(|path| {
            path.as_str()
                .map(str::to_string)
                .context("child outcome contains a non-string changed path")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !changed_files.is_empty(),
        "child outcome reported no file changes"
    );
    let unique = changed_files.iter().collect::<BTreeSet<_>>();
    ensure!(
        unique.len() == changed_files.len(),
        "child outcome contains duplicate changed paths"
    );
    let exact_changes_value = raw
        .get("changes")
        .and_then(Value::as_array)
        .context("child report omitted exact change records")?;
    let exact_changes: Vec<hi_tools::FileChange> =
        serde_json::from_value(Value::Array(exact_changes_value.clone()))
            .context("child report contains incomplete exact change metadata")?;
    ensure!(
        !exact_changes.is_empty(),
        "child report has no exact change records"
    );
    for change in &exact_changes {
        ensure_safe_relative_path(Path::new(&change.path))?;
        let digest_present = |digest: &Option<String>| {
            digest
                .as_deref()
                .is_some_and(|digest| digest.starts_with("sha256:") && digest.len() > 7)
        };
        let valid = match change.kind {
            hi_tools::FileChangeKind::Create => {
                change.before_digest.is_none()
                    && change.before_len.is_none()
                    && change.before_mode.is_none()
                    && digest_present(&change.after_digest)
                    && change.after_len.is_some()
                    && change.after_mode.is_some()
            }
            hi_tools::FileChangeKind::Modify => {
                digest_present(&change.before_digest)
                    && digest_present(&change.after_digest)
                    && change.before_len.is_some()
                    && change.after_len.is_some()
                    && change.before_mode.is_some()
                    && change.after_mode.is_some()
            }
            hi_tools::FileChangeKind::Delete => {
                digest_present(&change.before_digest)
                    && change.before_len.is_some()
                    && change.before_mode.is_some()
                    && change.after_digest.is_none()
                    && change.after_len.is_none()
                    && change.after_mode.is_none()
            }
        };
        ensure!(
            valid,
            "child report has inconsistent exact metadata for {}",
            change.path
        );
    }
    let exact_paths = exact_changes
        .iter()
        .map(|change| change.path.clone())
        .collect::<Vec<_>>();
    ensure!(
        exact_paths.iter().collect::<BTreeSet<_>>().len() == exact_paths.len(),
        "child report contains duplicate exact change records"
    );
    ensure!(
        same_paths(&changed_files, &exact_paths),
        "child outcome paths disagree with exact change records"
    );
    ensure!(
        raw.get("route") == raw.pointer("/outcome/effective_route"),
        "child effective-route fields disagree"
    );
    Ok(ChildReportGate {
        changed_files,
        review_status,
    })
}

#[derive(Clone, Debug)]
pub(crate) struct CandidateDiff {
    pub(crate) paths: Vec<PathBuf>,
    pub(crate) display_paths: Vec<String>,
    pub(crate) patch: Vec<u8>,
}

/// Stage and materialize the exact binary patch against an immutable base.
pub(crate) fn staged_candidate_diff(worktree: &Path, base: &str) -> Result<CandidateDiff> {
    let add = Command::new("git")
        .current_dir(worktree)
        .args(["add", "-A", "--", "."])
        .args(PYCACHE_EXCLUDES)
        .output()
        .context("staging candidate diff")?;
    ensure_command(add, "git add in candidate worktree")?;

    let names = Command::new("git")
        .current_dir(worktree)
        .args([
            "diff",
            "--cached",
            "--relative",
            "--no-renames",
            "--name-status",
            "-z",
            base,
            "--",
            ".",
        ])
        .args(PYCACHE_EXCLUDES)
        .output()
        .context("listing candidate diff")?;
    let names = ensure_command(names, "git diff --name-status in candidate worktree")?;
    let paths = parse_name_status(&names.stdout)?;

    let patch = Command::new("git")
        .current_dir(worktree)
        .args([
            "diff",
            "--cached",
            "--relative",
            "--binary",
            "--no-renames",
            base,
            "--",
            ".",
        ])
        .args(PYCACHE_EXCLUDES)
        .output()
        .context("materializing candidate patch")?;
    let patch = ensure_command(patch, "git diff in candidate worktree")?.stdout;
    ensure!(
        paths.is_empty() == patch.is_empty(),
        "candidate path list and patch disagree"
    );
    let display_paths = paths.iter().map(|path| display_path(path)).collect();
    Ok(CandidateDiff {
        paths,
        display_paths,
        patch,
    })
}

pub(crate) fn ensure_command(
    output: std::process::Output,
    operation: &str,
) -> Result<std::process::Output> {
    if output.status.success() {
        Ok(output)
    } else {
        bail!(
            "{operation} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
    }
}

pub(crate) fn parse_name_status(bytes: &[u8]) -> Result<Vec<PathBuf>> {
    let fields = bytes
        .split(|byte| *byte == 0)
        .filter(|field| !field.is_empty())
        .collect::<Vec<_>>();
    let mut paths = Vec::new();
    let mut index = 0;
    while index < fields.len() {
        let field = fields[index];
        index += 1;
        let (status, path_bytes) = if let Some(tab) = field.iter().position(|byte| *byte == b'\t') {
            (&field[..tab], &field[tab + 1..])
        } else {
            ensure!(index < fields.len(), "truncated git name-status output");
            let path = fields[index];
            index += 1;
            (field, path)
        };
        ensure!(
            matches!(status.first().copied(), Some(b'A' | b'M' | b'D' | b'T')),
            "unsupported candidate change status '{}'",
            String::from_utf8_lossy(status)
        );
        let path = path_from_git_bytes(path_bytes)?;
        ensure_safe_relative_path(&path)?;
        ensure!(
            !paths.contains(&path),
            "duplicate candidate path {}",
            path.display()
        );
        paths.push(path);
    }
    Ok(paths)
}

#[cfg(unix)]
fn path_from_git_bytes(bytes: &[u8]) -> Result<PathBuf> {
    use std::os::unix::ffi::OsStringExt;
    Ok(PathBuf::from(std::ffi::OsString::from_vec(bytes.to_vec())))
}

#[cfg(not(unix))]
fn path_from_git_bytes(bytes: &[u8]) -> Result<PathBuf> {
    Ok(PathBuf::from(
        String::from_utf8(bytes.to_vec()).context("candidate path is not valid UTF-8")?,
    ))
}

fn ensure_safe_relative_path(path: &Path) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "candidate path is empty");
    ensure!(
        !path.is_absolute(),
        "candidate path is absolute: {}",
        path.display()
    );
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "candidate path escapes the workspace: {}",
        path.display()
    );
    Ok(())
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

pub(crate) fn same_paths(left: &[String], right: &[String]) -> bool {
    left.iter().collect::<BTreeSet<_>>() == right.iter().collect::<BTreeSet<_>>()
}

/// Rerun the verifier and prove that it did not mutate the passing patch.
pub(crate) fn independently_verify_candidate(
    worktree: &Path,
    base: &str,
    verify: &str,
) -> Result<CandidateDiff> {
    ensure!(!verify.trim().is_empty(), "candidate verifier is empty");
    let before = staged_candidate_diff(worktree, base)?;
    ensure!(!before.paths.is_empty(), "candidate diff is empty");
    run_verifier_sync(worktree, verify)
        .with_context(|| format!("configured verifier `{verify}` failed"))?;
    let after = staged_candidate_diff(worktree, base)?;
    ensure!(
        before.patch == after.patch,
        "configured verifier modified relevant candidate files (verification unstable)"
    );
    Ok(after)
}

pub(crate) fn repository_root(from: &Path) -> Result<PathBuf> {
    let output = Command::new("git")
        .arg("-C")
        .arg(from)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("resolving repository root")?;
    let output = ensure_command(output, "git rev-parse --show-toplevel")?;
    let root = String::from_utf8(output.stdout).context("repository root is not valid UTF-8")?;
    let root = root.trim();
    ensure!(!root.is_empty(), "repository root is empty");
    Ok(PathBuf::from(root))
}

pub(crate) fn run_verifier_sync(root: &Path, command: &str) -> Result<()> {
    let root = root.to_path_buf();
    let command = command.to_string();
    let timeout = std::env::var("HI_VERIFY_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .unwrap_or(VERIFY_TIMEOUT_SECS);
    hi_tools::prepare_verify_workdir(&root);
    run_async_thread(move || async move {
        let runner = hi_tools::ProcessRunner::new(&root)?;
        let execution = runner
            .run_shell(&command, Duration::from_secs(timeout))
            .await?;
        ensure!(
            execution.status == hi_tools::ToolStatus::Succeeded,
            "verifier status {:?} (exit {:?}): {}",
            execution.status,
            execution.outcome.exit_code,
            execution.model_content()
        );
        Ok(())
    })
}

pub(crate) fn run_async_thread<F, Fut, T>(operation: F) -> Result<T>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T>> + 'static,
    T: Send + 'static,
{
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("creating candidate-operation runtime")?;
        runtime.block_on(operation())
    })
    .join()
    .map_err(|_| anyhow!("candidate-operation worker panicked"))?
}
