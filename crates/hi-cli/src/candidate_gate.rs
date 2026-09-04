//! Shared eligibility gates for delegate and best-of candidates.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
#[cfg(test)]
use std::time::Instant;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, anyhow, bail, ensure};
use hi_workspace::ResolvedHarnessSettings as Harness;
use serde_json::Value;
use sha2::{Digest, Sha256};

const VERIFY_LOCK_POLL_MILLIS: u64 = 25;
/// Regenerable build/dependency artifacts excluded from candidate diffs and
/// merges. The child ledger prunes the same names; otherwise a build in a repo
/// without `.gitignore` fails report-vs-diff. Build output is never merged.
const PYCACHE_EXCLUDES: &[&str] = &[
    ":(exclude,glob)**/__pycache__/**",
    ":(exclude,glob)**/*.pyc",
    ":(exclude,glob)**/*.pyo",
    ":(exclude,glob)**/target/**",
    ":(exclude,glob)**/node_modules/**",
    ":(exclude,glob)**/.venv/**",
    ":(exclude,glob)**/venv/**",
    ":(exclude,glob)**/dist/**",
    ":(exclude,glob)**/build/**",
    ":(exclude,glob)**/.next/**",
    ":(exclude,glob)**/.turbo/**",
    ":(exclude,glob)**/coverage/**",
    ":(exclude,glob)**/.pytest_cache/**",
    ":(exclude,glob)**/.mypy_cache/**",
    ":(exclude,glob)**/.ruff_cache/**",
    ":(exclude,glob)**/.hi/**",
];
/// Rust-side mirror of [`PYCACHE_EXCLUDES`]: whether a child-reported path is
/// eligible for the exact report-vs-diff comparison and destination merge.
/// Must stay in lockstep with the pathspec list above.
pub(crate) fn merge_eligible_path(path: &str) -> bool {
    let excluded_component = path.split('/').any(|component| {
        matches!(
            component,
            "__pycache__"
                | "target"
                | "node_modules"
                | ".venv"
                | "venv"
                | "dist"
                | "build"
                | ".next"
                | ".turbo"
                | "coverage"
                | ".pytest_cache"
                | ".mypy_cache"
                | ".ruff_cache"
                | ".hi"
        )
    });
    !excluded_component && !path.ends_with(".pyc") && !path.ends_with(".pyo")
}

/// Distinct from a generic verifier failure so merge rollback vs cancel stay
/// correct, and so a cancelled run is never cached as a passing gate.
#[derive(Debug)]
pub(crate) struct VerifierCancelled;

impl std::fmt::Display for VerifierCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("verifier cancelled")
    }
}

impl std::error::Error for VerifierCancelled {}

/// Cancelled after the destination was mutated; the caller rolled it back.
#[derive(Debug)]
pub(crate) struct DestinationVerifyCancelled;

impl std::fmt::Display for DestinationVerifyCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("destination verifier cancelled")
    }
}

impl std::error::Error for DestinationVerifyCancelled {}

pub(crate) fn is_verifier_cancelled(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<VerifierCancelled>().is_some())
}

pub(crate) fn is_destination_verify_cancelled(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<DestinationVerifyCancelled>().is_some())
}

fn cancel_requested(cancellation: Option<&hi_agent::TurnCancellation>) -> bool {
    cancellation.is_some_and(hi_agent::TurnCancellation::is_cancelled)
}

static VERIFY_SINGLE_FLIGHT: OnceLock<Mutex<BTreeSet<String>>> = OnceLock::new();

struct VerifyFlight {
    key: String,
    lock_path: PathBuf,
    owner: String,
}

impl Drop for VerifyFlight {
    fn drop(&mut self) {
        if lock_owner(&self.lock_path).as_deref() == Some(self.owner.as_str()) {
            let _ = std::fs::remove_file(&self.lock_path);
        }
        if let Some(flights) = VERIFY_SINGLE_FLIGHT.get() {
            flights
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .remove(&self.key);
        }
    }
}

fn lock_owner(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| raw.lines().next().map(str::to_owned))
}

fn lock_pid(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path)
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("pid=")?
                .trim()
                .parse()
                .ok()
                .filter(|pid| *pid > 0)
        })
}

fn lock_birth(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("birth=").map(str::trim))
        .filter(|birth| !birth.is_empty())
        .map(str::to_owned)
}

fn verification_lock_stale_after(timeout: Option<Duration>) -> Option<Duration> {
    timeout.map(|timeout| {
        timeout
            .saturating_mul(2)
            .saturating_add(Duration::from_secs(60))
    })
}

fn lock_is_stale(path: &Path) -> bool {
    let birth = lock_birth(path);
    if crate::resource_governor::owner_record_is_stale(path, lock_pid(path), birth.as_deref()) {
        return true;
    }
    let Ok(modified) = std::fs::metadata(path).and_then(|metadata| metadata.modified()) else {
        return false;
    };
    let age = SystemTime::now().duration_since(modified).ok();
    verification_lock_stale_after(hi_tools::check_timeout())
        .is_some_and(|stale_after| age.is_some_and(|age| age >= stale_after))
}

fn create_verify_lock(path: &Path, owner: &str) -> std::io::Result<File> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let birth = crate::resource_governor::current_process_birth_identity()
        .map(|birth| format!("birth={birth}\n"))
        .unwrap_or_default();
    if let Err(error) = writeln!(file, "{owner}\npid={}\n{birth}", std::process::id())
        .and_then(|()| file.sync_all())
    {
        let _ = std::fs::remove_file(path);
        return Err(error);
    }
    Ok(file)
}

#[cfg(test)]
fn acquire_verify_flight(key: &str, cache_path: &Path) -> Result<Option<VerifyFlight>> {
    acquire_verify_flight_while(key, cache_path, &|| false)
}

fn acquire_verify_flight_while(
    key: &str,
    cache_path: &Path,
    stop: &dyn Fn() -> bool,
) -> Result<Option<VerifyFlight>> {
    let flights = VERIFY_SINGLE_FLIGHT.get_or_init(|| Mutex::new(BTreeSet::new()));
    let lock_path = cache_path.with_extension("lock");
    let owner = format!("{}-{key}", std::process::id());
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating verification cache {}", parent.display()))?;
    }
    loop {
        if stop() {
            return Err(VerifierCancelled.into());
        }
        if cache_path.is_file() {
            return Ok(None);
        }
        let owns_local = flights
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(key.to_string());
        if owns_local {
            match create_verify_lock(&lock_path, &owner) {
                Ok(_) => {
                    return Ok(Some(VerifyFlight {
                        key: key.to_string(),
                        lock_path,
                        owner,
                    }));
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    flights
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .remove(key);
                    if lock_is_stale(&lock_path) {
                        let _ = std::fs::remove_file(&lock_path);
                    }
                }
                Err(error) => {
                    flights
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .remove(key);
                    return Err(error).with_context(|| {
                        format!("creating verification lock {}", lock_path.display())
                    });
                }
            }
        }
        std::thread::sleep(Duration::from_millis(VERIFY_LOCK_POLL_MILLIS));
    }
}

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

fn verification_fingerprint() -> String {
    const ENVIRONMENT: &[&str] = &[
        "PATH",
        "RUSTUP_TOOLCHAIN",
        "RUSTFLAGS",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "PKG_CONFIG_PATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "NODE_ENV",
        "PYTHONPATH",
    ];
    let mut fingerprint = Sha256::new();
    fingerprint.update(std::env::consts::OS.as_bytes());
    fingerprint.update([0]);
    fingerprint.update(std::env::consts::ARCH.as_bytes());
    for name in ENVIRONMENT {
        fingerprint.update([0]);
        fingerprint.update(name.as_bytes());
        fingerprint.update(b"=");
        if let Some(value) = std::env::var_os(name) {
            fingerprint.update(value.to_string_lossy().as_bytes());
        }
    }
    format!("{:x}", fingerprint.finalize())
}

pub(crate) fn independently_verify_candidate_cached(
    worktree: &Path,
    base: &str,
    verify: &str,
    cache_root: &Path,
    cancellation: Option<hi_agent::TurnCancellation>,
) -> Result<CandidateDiff> {
    ensure!(!verify.trim().is_empty(), "candidate verifier is empty");
    let before = staged_candidate_diff(worktree, base)?;
    ensure!(!before.paths.is_empty(), "candidate diff is empty");
    let fingerprint = verification_fingerprint();
    let mut key = Sha256::new();
    key.update(b"delegate-verify-v2\0");
    key.update(base.as_bytes());
    key.update([0]);
    key.update(verify.as_bytes());
    key.update([0]);
    key.update(fingerprint.as_bytes());
    key.update([0]);
    key.update(&before.patch);
    let key_hex = format!("{:x}", key.finalize());
    let cache_path = cache_root.join(format!("{key_hex}.json"));
    if let Ok(raw) = std::fs::read(&cache_path)
        && let Ok(record) = serde_json::from_slice::<Value>(&raw)
        && record.get("schema_version").and_then(Value::as_u64) == Some(2)
        && record.get("key").and_then(Value::as_str) == Some(key_hex.as_str())
        && record.get("base").and_then(Value::as_str) == Some(base)
        && record.get("verify").and_then(Value::as_str) == Some(verify)
        && record.get("fingerprint").and_then(Value::as_str) == Some(fingerprint.as_str())
    {
        return Ok(before);
    }
    let _flight = acquire_verify_flight_while(&key_hex, &cache_path, &|| {
        cancel_requested(cancellation.as_ref())
    })?;
    if cache_path.is_file() {
        return Ok(before);
    }
    if cancel_requested(cancellation.as_ref()) {
        return Err(VerifierCancelled.into());
    }
    match run_verifier_sync_cancellable(worktree, verify, cancellation.clone()) {
        Ok(()) => {}
        Err(error) if is_verifier_cancelled(&error) => return Err(error),
        Err(error) => {
            return Err(error).context(format!("configured verifier `{verify}` failed"));
        }
    }
    let after = staged_candidate_diff(worktree, base)?;
    ensure!(
        before.patch == after.patch,
        "configured verifier modified relevant candidate files (verification unstable)"
    );
    std::fs::create_dir_all(cache_root)
        .with_context(|| format!("creating verification cache {}", cache_root.display()))?;
    let record = serde_json::json!({
        "schema_version": 2,
        "key": key_hex,
        "base": base,
        "verify": verify,
        "fingerprint": fingerprint,
    });
    let temporary = cache_path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec(&record)?)
        .with_context(|| format!("writing verification cache {}", temporary.display()))?;
    std::fs::rename(&temporary, &cache_path).with_context(|| {
        let _ = std::fs::remove_file(&temporary);
        format!("committing verification cache {}", cache_path.display())
    })?;
    Ok(after)
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
    run_verifier_sync_cancellable(root, command, None)
}

pub(crate) fn run_verifier_sync_cancellable(
    root: &Path,
    command: &str,
    cancellation: Option<hi_agent::TurnCancellation>,
) -> Result<()> {
    if cancel_requested(cancellation.as_ref()) {
        return Err(VerifierCancelled.into());
    }
    let root = root.to_path_buf();
    let command = command.to_string();
    let timeout = hi_tools::check_timeout().or(Some(Harness::default().jobs.verifier_timeout));
    hi_tools::prepare_verify_workdir(&root);
    run_async_thread(move || async move {
        let runner = hi_tools::ProcessRunner::new(&root)?;
        let run = runner.run_shell_maybe_timeout(&command, timeout);
        let execution = match cancellation {
            Some(cancel) => {
                tokio::select! {
                    result = run => result?,
                    _ = crate::child_process::wait_for_cancel(cancel) => {
                        return Err(VerifierCancelled.into());
                    }
                }
            }
            None => run.await?,
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlimited_verification_does_not_age_out_an_active_flight() {
        assert_eq!(verification_lock_stale_after(None), None);
        assert_eq!(
            verification_lock_stale_after(Some(Duration::from_secs(10))),
            Some(Duration::from_secs(80))
        );
    }

    #[test]
    fn verify_flight_removes_only_its_own_lock() {
        let root = std::env::temp_dir().join(format!(
            "hi-verify-flight-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = root.join("key.json");
        let flight = acquire_verify_flight("test-key", &cache).unwrap().unwrap();
        let lock = cache.with_extension("lock");
        assert!(lock.is_file());
        std::fs::write(&lock, "replacement-owner\n").unwrap();
        drop(flight);
        assert!(lock.is_file());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fingerprint_changes_with_relevant_environment() {
        let name = "RUSTFLAGS";
        let original = std::env::var_os(name);
        unsafe { std::env::set_var(name, "-Ctarget-cpu=one") };
        let first = verification_fingerprint();
        unsafe { std::env::set_var(name, "-Ctarget-cpu=two") };
        let second = verification_fingerprint();
        match original {
            Some(value) => unsafe { std::env::set_var(name, value) },
            None => unsafe { std::env::remove_var(name) },
        }
        assert_ne!(first, second);
    }

    #[test]
    fn run_verifier_sync_cancellable_stops_when_cancelled() {
        let original_sandbox = std::env::var_os("HI_SANDBOX");
        unsafe { std::env::set_var("HI_SANDBOX", "off") };
        let root = std::env::temp_dir().join(format!(
            "hi-verify-cancel-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cancel = hi_agent::TurnCancellation::new();
        let cancel_thread = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            cancel_thread.cancel();
        });
        let started = Instant::now();
        let error = run_verifier_sync_cancellable(&root, "sleep 30", Some(cancel))
            .expect_err("cancel must stop the verifier");
        let elapsed = started.elapsed();
        match original_sandbox {
            Some(value) => unsafe { std::env::set_var("HI_SANDBOX", value) },
            None => unsafe { std::env::remove_var("HI_SANDBOX") },
        }
        let _ = std::fs::remove_dir_all(root);
        assert!(
            is_verifier_cancelled(&error),
            "cancel must be distinct from a generic verify fail: {error:#}"
        );
        assert!(
            elapsed < Duration::from_secs(3),
            "cancel must kill the verifier instead of waiting out the command: {elapsed:?}"
        );
    }
}
