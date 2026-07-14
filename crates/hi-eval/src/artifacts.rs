use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::config::{EvalProfile, FinalOracle, Task};
use crate::results::{RunArtifact, RunResult};

const MAX_CAPTURED_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
pub const INJECTED_ORACLE_DIR: &str = ".hi-eval-oracle";
static JSONL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Debug)]
pub struct TimedOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub timed_out: bool,
    pub output_truncated: bool,
    pub duration: Duration,
}

impl TimedOutput {
    pub fn success(&self) -> bool {
        !self.timed_out && self.status.success()
    }
}

/// Run a child with closed stdin, bounded captured output, and a hard timeout.
/// On Unix the child gets its own process group so timeout cleanup also reaches
/// descendants that inherited its output pipes.
pub fn command_output_with_timeout(
    command: &mut Command,
    timeout: Duration,
) -> std::io::Result<TimedOutput> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let started = Instant::now();
    let mut child = command.spawn()?;
    #[cfg(unix)]
    let process_group = child.id() as i32;
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let stdout_reader = std::thread::spawn(move || read_bounded(stdout));
    let stderr_reader = std::thread::spawn(move || read_bounded(stderr));
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() >= timeout {
            timed_out = true;
            #[cfg(unix)]
            // SAFETY: the child was placed in a fresh process group whose id is
            // its pid. A negative pid targets that group only.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.kill();
            break child.wait()?;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    #[cfg(unix)]
    // The candidate/check process is complete; clean up any descendants it
    // left in the isolated process group before snapshotting or joining pipes.
    // SAFETY: `process_group` identifies the fresh group created above.
    unsafe {
        libc::kill(-process_group, libc::SIGKILL);
    }
    let (stdout, stdout_truncated) = stdout_reader
        .join()
        .unwrap_or_else(|_| (b"output reader panicked".to_vec(), true));
    let (stderr, stderr_truncated) = stderr_reader
        .join()
        .unwrap_or_else(|_| (b"output reader panicked".to_vec(), true));
    Ok(TimedOutput {
        status,
        stdout,
        stderr,
        timed_out,
        output_truncated: stdout_truncated || stderr_truncated,
        duration: started.elapsed(),
    })
}

fn read_bounded(mut reader: impl Read) -> (Vec<u8>, bool) {
    let mut out = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut truncated = false;
    while let Ok(n) = reader.read(&mut chunk) {
        if n == 0 {
            break;
        }
        let remaining = MAX_CAPTURED_OUTPUT_BYTES.saturating_sub(out.len());
        if remaining > 0 {
            out.extend_from_slice(&chunk[..n.min(remaining)]);
        }
        truncated |= n > remaining;
    }
    (out, truncated)
}

#[derive(Clone, Debug)]
pub struct CapturedOracle {
    command: String,
    entries: Vec<CapturedOracleEntry>,
}

#[derive(Clone, Debug)]
struct CapturedOracleEntry {
    path: PathBuf,
    kind: CapturedOracleEntryKind,
    permissions: std::fs::Permissions,
}

#[derive(Clone, Debug)]
enum CapturedOracleEntryKind {
    Directory,
    File(Vec<u8>),
    Symlink(PathBuf),
}

impl CapturedOracle {
    /// Capture both the command and every oracle file before a candidate can
    /// run. Contained relative symlinks are captured as links; links that could
    /// escape the bundle and special filesystem nodes are rejected.
    pub fn capture(task_dir: &Path, spec: &FinalOracle) -> Result<Self> {
        let mut entries = Vec::new();
        if let Some(bundle) = &spec.bundle {
            let root = task_dir.join(bundle);
            let metadata = std::fs::symlink_metadata(&root)
                .with_context(|| format!("reading oracle bundle metadata {}", root.display()))?;
            if !metadata.is_dir() || metadata.file_type().is_symlink() {
                bail!("oracle bundle is not a real directory: {}", root.display());
            }
            entries.push(CapturedOracleEntry {
                path: PathBuf::new(),
                kind: CapturedOracleEntryKind::Directory,
                permissions: metadata.permissions(),
            });
            capture_oracle_dir(&root, &root, &mut entries)?;
        }
        Ok(Self {
            command: spec.command.clone(),
            entries,
        })
    }

    fn inject(&self, dir: &Path) -> Result<()> {
        let root = dir.join(INJECTED_ORACLE_DIR);
        remove_path_if_present(&root)?;
        std::fs::create_dir_all(&root)?;
        for entry in &self.entries {
            let path = root.join(&entry.path);
            match &entry.kind {
                CapturedOracleEntryKind::Directory => {
                    std::fs::create_dir_all(&path)?;
                }
                CapturedOracleEntryKind::File(bytes) => {
                    let parent = path.parent().context("oracle file has no parent")?;
                    ensure_real_directory(parent)?;
                    std::fs::write(&path, bytes)?;
                    std::fs::set_permissions(&path, entry.permissions.clone())?;
                }
                CapturedOracleEntryKind::Symlink(target) => {
                    let parent = path.parent().context("oracle symlink has no parent")?;
                    ensure_real_directory(parent)?;
                    create_symlink(target, &path)?;
                }
            }
        }
        // Apply directory modes only after all children exist. This preserves
        // read-only/executable directory modes without blocking injection.
        for entry in self.entries.iter().rev() {
            if matches!(entry.kind, CapturedOracleEntryKind::Directory) {
                std::fs::set_permissions(root.join(&entry.path), entry.permissions.clone())?;
            }
        }
        Ok(())
    }
}

fn capture_oracle_dir(root: &Path, dir: &Path, out: &mut Vec<CapturedOracleEntry>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let metadata = std::fs::symlink_metadata(&path)?;
        let relative = path
            .strip_prefix(root)
            .expect("walked below root")
            .to_path_buf();
        if metadata.is_dir() {
            out.push(CapturedOracleEntry {
                path: relative,
                kind: CapturedOracleEntryKind::Directory,
                permissions: metadata.permissions(),
            });
            capture_oracle_dir(root, &path, out)?;
        } else if metadata.is_file() {
            out.push(CapturedOracleEntry {
                path: relative,
                kind: CapturedOracleEntryKind::File(std::fs::read(&path)?),
                permissions: metadata.permissions(),
            });
        } else if metadata.file_type().is_symlink() {
            let target = std::fs::read_link(&path)?;
            validate_contained_symlink(&relative, &target)
                .with_context(|| format!("unsafe oracle symlink {}", path.display()))?;
            out.push(CapturedOracleEntry {
                path: relative,
                kind: CapturedOracleEntryKind::Symlink(target),
                permissions: metadata.permissions(),
            });
        } else {
            bail!(
                "oracle bundle contains unsupported special filesystem node: {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn remove_path_if_present(path: &Path) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        std::fs::remove_dir_all(path)?;
    } else {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

pub fn write_artifact(
    dir: &Path,
    profile: EvalProfile,
    condense: bool,
    recovery: bool,
    write_subagents: bool,
    goal_mode: bool,
    result: &RunResult,
) -> Result<()> {
    let artifact = RunArtifact {
        schema_version: 2,
        task: result.task.clone(),
        config: result.config.to_string(),
        model: result.model.clone(),
        trial: result.trial,
        profile: profile.label().to_string(),
        condense,
        recovery,
        write_subagents,
        goal_mode,
        passed: result.passed,
        failure_bucket: result.fail.map(|f| f.label().to_string()),
        failure_confidence: result.failure_confidence,
        changed_files: result.changed_files.clone(),
        provider_error_kind: result.provider_error_kind.clone(),
        compat_fallbacks_used: result.compat_fallbacks_used.clone(),
        candidate_count: result.candidates.len(),
        candidate_pass_rate: result.candidate_pass_rate(),
        solve_at_n: result.passed,
        candidate_results: result.candidates.clone(),
        tokens: result.tokens,
        duration_seconds: result.seconds,
        mcp_model: result.mcp_model.clone(),
        verify_output_summary: result.verify_output_summary.clone(),
        trajectory: result.trajectory.clone(),
        growth: result.growth.clone(),
    };
    let name = format!(
        "trial-{:03}-{}-{}-{}.json",
        result.trial + 1,
        sanitize_name(&result.config),
        sanitize_name(&result.model),
        sanitize_name(&result.task)
    );
    let json = serde_json::to_string_pretty(&artifact)?;
    std::fs::write(dir.join(name), json)?;

    let _jsonl_guard = JSONL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut jsonl = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("runs.jsonl"))?;
    writeln!(jsonl, "{}", serde_json::to_string(&artifact)?)?;
    Ok(())
}

pub fn sanitize_name(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

pub fn default_artifacts_dir() -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from("target")
        .join("hi-eval")
        .join("runs")
        .join(format!("{stamp}-{}", std::process::id()))
}

/// Validate every integrity property that can be checked without a model:
/// fail-before, pass-after, nonempty/allowed reference changes, forbidden-file
/// rejection, and replacement of candidate-side oracle files.
pub fn validate_tasks(tasks: &[(PathBuf, Task)]) -> Result<()> {
    eprintln!("Validating {} task(s)...", tasks.len());
    let mut broken = 0;
    for (dir, task) in tasks {
        let label = task.name.clone().unwrap_or_else(|| dir_name(dir));
        match validate_task(dir, task) {
            Ok(()) => println!("  OK      {label}"),
            Err(reason) => {
                println!("  BROKEN  {label}: {reason}");
                broken += 1;
            }
        }
    }
    if broken > 0 {
        bail!("{broken} task(s) are not well-formed");
    }
    println!(
        "\nAll {} tasks well-formed (fail-before, pass-after, allowed changes, immutable oracle).",
        tasks.len()
    );
    Ok(())
}

pub fn validate_task(dir: &Path, task: &Task) -> std::result::Result<(), String> {
    task.validate().map_err(|err| err.to_string())?;
    let oracle = CapturedOracle::capture(dir, &task.final_oracle).map_err(|err| err.to_string())?;
    let work = make_workdir().map_err(|e| e.to_string())?;
    let fixture = dir.join("fixture");
    copy_dir(&fixture, &work).map_err(|e| e.to_string())?;
    crate::runner::initialize_workspace(&work, task).map_err(|err| err.to_string())?;

    let result = (|| {
        let before = directory_snapshot(&work).map_err(|err| err.to_string())?;
        let raw = run_final_oracle(&work, &oracle, task.timeouts.oracle_seconds)
            .map_err(|err| err.to_string())?;
        if raw.success() {
            return Err("final oracle already passes on the unmodified fixture".to_string());
        }

        // A candidate may create the eventual injection path, but scoring must
        // replace it with the captured bytes rather than trust those files.
        let fake_oracle = work.join(INJECTED_ORACLE_DIR);
        std::fs::create_dir_all(&fake_oracle).map_err(|err| err.to_string())?;
        std::fs::write(fake_oracle.join("check.py"), "raise SystemExit(0)\n")
            .map_err(|err| err.to_string())?;
        let tampered = run_final_oracle(&work, &oracle, task.timeouts.oracle_seconds)
            .map_err(|err| err.to_string())?;
        if tampered.success() {
            return Err("candidate-side oracle files can influence the final score".to_string());
        }
        remove_path_if_present(&fake_oracle).map_err(|err| err.to_string())?;

        let fixed = dir.join("fixed");
        copy_dir(&fixed, &work).map_err(|error| format!("invalid fixed/ reference: {error:#}"))?;
        let after = directory_snapshot(&work).map_err(|err| err.to_string())?;
        let changed = changed_paths(&before, &after);
        if changed.is_empty() {
            return Err("fixed/ reference makes no candidate-visible changes".to_string());
        }
        let forbidden =
            forbidden_changes(&changed, &task.allowed_changes).map_err(|err| err.to_string())?;
        if !forbidden.is_empty() {
            return Err(format!(
                "fixed/ changes paths outside allowed_changes: {}",
                forbidden.join(", ")
            ));
        }
        let fixed_output = run_final_oracle(&work, &oracle, task.timeouts.oracle_seconds)
            .map_err(|err| err.to_string())?;
        if !fixed_output.success() {
            return Err(format!(
                "final oracle still fails after applying fixed/: {}",
                output_summary(&fixed_output)
            ));
        }

        let mut forbidden_probe = changed.clone();
        forbidden_probe.push(".hi-eval-forbidden-probe".to_string());
        let rejected = forbidden_changes(&forbidden_probe, &task.allowed_changes)
            .map_err(|err| err.to_string())?;
        if !rejected
            .iter()
            .any(|path| path == ".hi-eval-forbidden-probe")
        {
            return Err("allowed_changes does not reject an unrelated file".to_string());
        }
        Ok(())
    })();

    let _ = std::fs::remove_dir_all(&work);
    result
}

#[cfg(test)]
pub fn verify_in(dir: &Path, cmd: &str) -> bool {
    verify_output_in(dir, cmd, crate::config::DEFAULT_CHECK_TIMEOUT_SECONDS)
        .map(|o| o.success())
        .unwrap_or(false)
}

pub fn verify_output_in(
    dir: &Path,
    cmd: &str,
    timeout_seconds: u64,
) -> std::io::Result<TimedOutput> {
    prepare_verify_workdir(dir)?;
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg(cmd)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .current_dir(dir);
    command_output_with_timeout(&mut command, Duration::from_secs(timeout_seconds))
}

/// Run the captured final oracle only after cloning the completed candidate.
/// Oracle files are never placed in the workspace used by the model.
pub fn run_final_oracle(
    candidate_dir: &Path,
    oracle: &CapturedOracle,
    timeout_seconds: u64,
) -> Result<TimedOutput> {
    run_final_oracle_without_artifacts(candidate_dir, oracle, timeout_seconds, &[])
}

/// Score a candidate after removing newly-created paths explicitly classified
/// as runtime artifacts. Ignored artifacts therefore cannot become a hidden
/// input to the final oracle.
pub fn run_final_oracle_without_artifacts(
    candidate_dir: &Path,
    oracle: &CapturedOracle,
    timeout_seconds: u64,
    runtime_artifacts: &[String],
) -> Result<TimedOutput> {
    let verification = make_workdir()?;
    let result = (|| {
        copy_dir(candidate_dir, &verification)?;
        prune_candidate_runtime_artifacts(&verification, runtime_artifacts)?;
        oracle.inject(&verification)?;
        verify_output_in(&verification, &oracle.command, timeout_seconds)
            .context("launching final oracle")
    })();
    let _ = std::fs::remove_dir_all(&verification);
    result
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotEntry {
    pub bytes: Vec<u8>,
    pub kind: SnapshotEntryKind,
    pub readonly: bool,
    #[cfg(unix)]
    pub mode: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotEntryKind {
    Directory,
    File,
    Symlink,
}

pub type DirectorySnapshot = std::collections::BTreeMap<String, SnapshotEntry>;

pub fn directory_snapshot(dir: &Path) -> Result<DirectorySnapshot> {
    fn walk(base: &Path, dir: &Path, out: &mut DirectorySnapshot) -> Result<()> {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("reading snapshot directory {}", dir.display()))?
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let relative = path.strip_prefix(base).expect("walked below snapshot root");
            if is_evaluator_runtime_file(relative) {
                continue;
            }
            let relative_text = relative
                .to_str()
                .with_context(|| format!("snapshot path is not UTF-8: {}", path.display()))?
                .replace('\\', "/");
            let metadata = std::fs::symlink_metadata(&path)
                .with_context(|| format!("reading snapshot metadata {}", path.display()))?;
            let file_type = metadata.file_type();
            let (kind, bytes) = if file_type.is_dir() {
                (SnapshotEntryKind::Directory, Vec::new())
            } else if file_type.is_file() {
                (
                    SnapshotEntryKind::File,
                    std::fs::read(&path)
                        .with_context(|| format!("reading snapshot file {}", path.display()))?,
                )
            } else if file_type.is_symlink() {
                let target = std::fs::read_link(&path)
                    .with_context(|| format!("reading snapshot symlink {}", path.display()))?;
                validate_contained_symlink(relative, &target)
                    .with_context(|| format!("unsafe candidate symlink {}", path.display()))?;
                (SnapshotEntryKind::Symlink, path_bytes(&target))
            } else {
                bail!(
                    "snapshot contains unsupported special filesystem node: {}",
                    path.display()
                );
            };
            out.insert(
                relative_text,
                SnapshotEntry {
                    bytes,
                    kind,
                    readonly: metadata.permissions().readonly(),
                    #[cfg(unix)]
                    mode: {
                        use std::os::unix::fs::PermissionsExt;
                        metadata.permissions().mode()
                    },
                },
            );
            if file_type.is_dir() {
                walk(base, &path, out)?;
            }
        }
        Ok(())
    }
    let mut out = DirectorySnapshot::new();
    ensure_real_directory(dir)?;
    walk(dir, dir, &mut out)?;
    Ok(out)
}

pub fn changed_paths(before: &DirectorySnapshot, after: &DirectorySnapshot) -> Vec<String> {
    all_changed_paths(before, after)
        .into_iter()
        .filter(|path| match (before.get(path), after.get(path)) {
            // Directory creation/deletion is implied by its file entries and
            // is not itself an allowed-change path (for example `tests/**`
            // does not match the parent `tests`). Mode/type changes to an
            // existing entry remain visible.
            (None, Some(entry)) | (Some(entry), None)
                if entry.kind == SnapshotEntryKind::Directory =>
            {
                false
            }
            _ => true,
        })
        .collect()
}

fn all_changed_paths(before: &DirectorySnapshot, after: &DirectorySnapshot) -> Vec<String> {
    let mut paths: std::collections::BTreeSet<&String> = before.keys().collect();
    paths.extend(after.keys());
    paths
        .into_iter()
        .filter(|path| before.get(*path) != after.get(*path))
        .cloned()
        .collect()
}

/// Candidate-visible changes, excluding only newly-created, explicitly-known
/// build/cache artifacts and the loose Git objects created by hi's immutable
/// checkpoints. Pre-existing entries in those trees are never ignored, and an
/// allowed-change glob always makes a matching path candidate-visible.
pub fn candidate_changed_paths(
    before: &DirectorySnapshot,
    after: &DirectorySnapshot,
    allowed: &[String],
) -> Result<Vec<String>> {
    let allowed = build_globset(allowed)?;
    Ok(changed_paths(before, after)
        .into_iter()
        .filter(|path| {
            allowed.is_match(path)
                || before.contains_key(path)
                || !is_new_runtime_artifact(path, after.get(path))
        })
        .collect())
}

pub fn candidate_runtime_artifact_paths(
    before: &DirectorySnapshot,
    after: &DirectorySnapshot,
    allowed: &[String],
) -> Result<Vec<String>> {
    let allowed = build_globset(allowed)?;
    Ok(all_changed_paths(before, after)
        .into_iter()
        .filter(|path| {
            let entry = after.get(path);
            let has_allowed_descendant = entry.is_some_and(|entry| {
                entry.kind == SnapshotEntryKind::Directory
                    && after.keys().any(|candidate| {
                        candidate
                            .strip_prefix(path)
                            .is_some_and(|suffix| suffix.starts_with('/'))
                            && allowed.is_match(candidate)
                    })
            });
            !has_allowed_descendant
                && !allowed.is_match(path)
                && !before.contains_key(path)
                && is_new_runtime_artifact(path, entry)
        })
        .collect())
}

pub fn forbidden_changes(changed: &[String], allowed: &[String]) -> Result<Vec<String>> {
    let set = build_globset(allowed)?;
    Ok(changed
        .iter()
        .filter(|path| !set.is_match(path))
        .cloned()
        .collect())
}

fn build_globset(allowed: &[String]) -> Result<globset::GlobSet> {
    let mut builder = globset::GlobSetBuilder::new();
    for pattern in allowed {
        builder.add(globset::Glob::new(pattern)?);
    }
    Ok(builder.build()?)
}

pub(crate) fn prune_candidate_runtime_artifacts(
    root: &Path,
    runtime_artifacts: &[String],
) -> Result<()> {
    for reserved in [
        INJECTED_ORACLE_DIR,
        ".hi-eval-report.json",
        ".hi-eval-session.jsonl",
        ".hi-debug.log",
    ] {
        remove_path_if_present(&root.join(reserved))?;
    }
    let mut paths: Vec<_> = runtime_artifacts.iter().map(Path::new).collect();
    paths.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for relative in paths {
        if relative.as_os_str().is_empty()
            || relative.is_absolute()
            || relative.components().any(|component| {
                !matches!(
                    component,
                    std::path::Component::Normal(_) | std::path::Component::CurDir
                )
            })
        {
            bail!("invalid runtime-artifact path: {}", relative.display());
        }
        remove_path_if_present(&root.join(relative))?;
    }
    Ok(())
}

fn is_evaluator_runtime_file(relative: &Path) -> bool {
    relative == Path::new(".hi-eval-report.json")
        || relative == Path::new(".hi-eval-session.jsonl")
        || relative == Path::new(".hi-debug.log")
}

fn is_new_runtime_artifact(path: &str, entry: Option<&SnapshotEntry>) -> bool {
    if is_loose_git_object(path) {
        return true;
    }
    let components: Vec<_> = path.split('/').collect();
    for (index, component) in components.iter().enumerate() {
        let suffix = &components[index + 1..];
        match *component {
            "target" if is_cargo_artifact(suffix, entry) => return true,
            ".next" | "dist" | "build" | ".pytest_cache" => return true,
            "__pycache__"
                if suffix.is_empty()
                    || suffix.last().is_some_and(|name| name.ends_with(".pyc")) =>
            {
                return true;
            }
            "node_modules"
                if suffix.is_empty()
                    || matches!(suffix.first(), Some(&".cache" | &".vite" | &".bin")) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

fn is_cargo_artifact(suffix: &[&str], entry: Option<&SnapshotEntry>) -> bool {
    if suffix.is_empty() {
        return true;
    }
    if matches!(
        suffix,
        [".rustc_info.json"] | ["CACHEDIR.TAG"] | ["package"] | ["doc"]
    ) || matches!(suffix.first(), Some(&"package" | &"doc"))
    {
        return true;
    }
    let Some(profile_index) = suffix
        .iter()
        .position(|part| matches!(*part, "debug" | "release"))
    else {
        return false;
    };
    let profile_suffix = &suffix[profile_index + 1..];
    if profile_suffix.is_empty() {
        return true;
    }
    if matches!(profile_suffix, [".cargo-lock"])
        || matches!(
            profile_suffix.first(),
            Some(&".fingerprint" | &"build" | &"deps" | &"examples" | &"incremental")
        )
    {
        return true;
    }
    // Cargo places final binaries directly below the profile directory.
    profile_suffix.len() == 1
        && entry.is_some_and(|entry| {
            entry.kind == SnapshotEntryKind::File
                && (entry.bytes.contains(&0) || {
                    #[cfg(unix)]
                    {
                        entry.mode & 0o111 != 0
                    }
                    #[cfg(not(unix))]
                    {
                        false
                    }
                })
        })
}

fn is_loose_git_object(path: &str) -> bool {
    let parts: Vec<_> = path.split('/').collect();
    matches!(parts.as_slice(), [".git", "objects", fanout] if is_lower_hex(fanout, 2))
        || matches!(
            parts.as_slice(),
            [".git", "objects", fanout, object]
                if is_lower_hex(fanout, 2) && is_lower_hex(object, 38)
        )
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(unix)]
fn path_bytes(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes().to_vec()
}

#[cfg(not(unix))]
fn path_bytes(path: &Path) -> Vec<u8> {
    path.to_string_lossy().into_owned().into_bytes()
}

pub fn output_summary(output: &TimedOutput) -> String {
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    let text = text.trim();
    if output.timed_out {
        format!(
            "timed out after {:.1}s: {text}",
            output.duration.as_secs_f64()
        )
    } else if text.is_empty() {
        format!("exit status {}", output.status)
    } else {
        text.to_string()
    }
}

pub fn prepare_verify_workdir(dir: &Path) -> std::io::Result<()> {
    fn walk(dir: &Path) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if entry.file_name() == "__pycache__" && file_type.is_dir() {
                std::fs::remove_dir_all(&path)?;
            } else if file_type.is_dir() {
                walk(&path)?;
            }
        }
        Ok(())
    }
    walk(dir)
}

pub fn discover_tasks(dir: &Path) -> Result<Vec<(PathBuf, Task)>> {
    let mut tasks = Vec::new();
    fn walk(dir: &Path, tasks: &mut Vec<(PathBuf, Task)>) -> Result<()> {
        let toml_path = dir.join("task.toml");
        let task_metadata = match std::fs::symlink_metadata(&toml_path) {
            Ok(metadata) => Some(metadata),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        if task_metadata
            .as_ref()
            .is_some_and(|metadata| metadata.file_type().is_symlink())
        {
            bail!(
                "task manifest may not be a symlink: {}",
                toml_path.display()
            );
        }
        if task_metadata
            .as_ref()
            .is_some_and(std::fs::Metadata::is_file)
        {
            let text = std::fs::read_to_string(&toml_path)
                .with_context(|| format!("reading {}", toml_path.display()))?;
            let task: Task = toml::from_str(&text)
                .with_context(|| format!("parsing {}", toml_path.display()))?;
            task.validate()
                .with_context(|| format!("validating {}", toml_path.display()))?;
            tasks.push((dir.to_path_buf(), task));
            return Ok(());
        }
        let entries = std::fs::read_dir(dir)
            .with_context(|| format!("reading tasks dir {}", dir.display()))?;
        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                paths.push(entry.path());
            }
        }
        paths.sort();
        for path in paths {
            walk(&path, tasks)?;
        }
        Ok(())
    }
    walk(dir, &mut tasks)?;
    Ok(tasks)
}

pub fn find_hi() -> Result<PathBuf> {
    // Must be absolute: each task runs with a different current_dir, so a
    // relative program path would resolve against the temp work dir.
    let candidate = if let Ok(bin) = std::env::var("HI_BIN") {
        PathBuf::from(bin)
    } else {
        ["target/debug/hi", "target/release/hi"]
            .into_iter()
            .map(PathBuf::from)
            .find(|p| p.is_file())
            .context("could not find the hi binary; build it or set HI_BIN")?
    };
    std::fs::canonicalize(&candidate)
        .with_context(|| format!("resolving hi binary path {}", candidate.display()))
}

pub fn make_workdir() -> Result<PathBuf> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    for _ in 0..10_000 {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("hi-eval-{}-{n}", std::process::id()));
        match std::fs::create_dir(&dir) {
            Ok(()) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
                }
                return Ok(dir);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("creating {}", dir.display()));
            }
        }
    }
    bail!("could not allocate a fresh hi-eval temporary directory")
}

pub fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    let source_metadata = std::fs::symlink_metadata(src)
        .with_context(|| format!("reading copy source metadata {}", src.display()))?;
    if !source_metadata.is_dir() || source_metadata.file_type().is_symlink() {
        bail!("copy source is not a real directory: {}", src.display());
    }
    ensure_real_directory(dst)?;
    copy_dir_contents(src, src, dst)
}

fn copy_dir_contents(root: &Path, src: &Path, dst: &Path) -> Result<()> {
    let mut entries = std::fs::read_dir(src)
        .with_context(|| format!("reading copy source directory {}", src.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let from = entry.path();
        let to = dst.join(entry.file_name());
        let relative = from
            .strip_prefix(root)
            .expect("copy walked below source root");
        let metadata = std::fs::symlink_metadata(&from)
            .with_context(|| format!("reading copy source metadata {}", from.display()))?;
        let file_type = metadata.file_type();
        if file_type.is_dir() {
            ensure_destination_directory(&to)?;
            copy_dir_contents(root, &from, &to)?;
            std::fs::set_permissions(&to, metadata.permissions())?;
        } else if file_type.is_file() {
            remove_path_if_present(&to)?;
            std::fs::copy(&from, &to).with_context(|| {
                format!(
                    "copying regular file {} to {}",
                    from.display(),
                    to.display()
                )
            })?;
            std::fs::set_permissions(&to, metadata.permissions())?;
        } else if file_type.is_symlink() {
            let target = std::fs::read_link(&from)
                .with_context(|| format!("reading copy symlink {}", from.display()))?;
            validate_contained_symlink(relative, &target)
                .with_context(|| format!("unsafe source symlink {}", from.display()))?;
            remove_path_if_present(&to)?;
            create_symlink(&target, &to)?;
        } else {
            bail!(
                "copy source contains unsupported special filesystem node: {}",
                from.display()
            );
        }
    }
    Ok(())
}

fn ensure_real_directory(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("reading directory metadata {}", path.display()))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        bail!("path is not a real directory: {}", path.display());
    }
    Ok(())
}

fn ensure_destination_directory(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            // An earlier overlay may have preserved a read-only source mode.
            // Temporarily make the directory writable, then restore the mode
            // from the source directory after its children are copied.
            let mut permissions = metadata.permissions();
            if permissions.readonly() {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    permissions.set_mode(permissions.mode() | 0o700);
                }
                #[cfg(not(unix))]
                permissions.set_readonly(false);
                std::fs::set_permissions(path, permissions)?;
            }
        }
        Ok(_) => {
            remove_path_if_present(path)?;
            std::fs::create_dir(path)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir(path)?;
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn validate_contained_symlink(link_path: &Path, target: &Path) -> Result<()> {
    use std::path::Component;

    if target.as_os_str().is_empty() || target.is_absolute() {
        bail!("symlink target must be a nonempty relative path");
    }
    let mut depth = link_path
        .parent()
        .into_iter()
        .flat_map(Path::components)
        .filter(|component| matches!(component, Component::Normal(_)))
        .count();
    for component in target.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(_) => depth += 1,
            Component::ParentDir if depth > 0 => depth -= 1,
            Component::ParentDir => bail!("symlink target escapes the copied tree"),
            Component::RootDir | Component::Prefix(_) => {
                bail!("symlink target must be relative")
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn create_symlink(target: &Path, link: &Path) -> Result<()> {
    std::os::unix::fs::symlink(target, link).with_context(|| {
        format!(
            "creating symlink {} -> {}",
            link.display(),
            target.display()
        )
    })
}

#[cfg(not(unix))]
fn create_symlink(_target: &Path, link: &Path) -> Result<()> {
    bail!(
        "symlink recreation is unsupported on this platform: {}",
        link.display()
    )
}

pub fn dir_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("task")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Task;

    #[test]
    fn verify_in_prunes_python_bytecode_cache_before_running() {
        let dir = make_workdir().expect("temp dir");
        std::fs::write(dir.join("solution.py"), "def value():\n    return 1\n").unwrap();
        let pycache = dir.join("__pycache__");
        std::fs::create_dir_all(&pycache).unwrap();
        std::fs::write(pycache.join("solution.cpython-311.pyc"), b"stale").unwrap();

        assert!(verify_in(
            &dir,
            "python3 -c \"import solution; assert solution.value() == 1\""
        ));
        assert!(
            !pycache.exists(),
            "verify should remove stale Python bytecode cache before import"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    fn oracle_test_task(root: &Path) -> Task {
        std::fs::create_dir_all(root.join("fixture")).unwrap();
        std::fs::create_dir_all(root.join("fixed")).unwrap();
        std::fs::create_dir_all(root.join("oracle")).unwrap();
        std::fs::write(root.join("fixture/solution.py"), "VALUE = 0\n").unwrap();
        std::fs::write(root.join("fixed/solution.py"), "VALUE = 42\n").unwrap();
        std::fs::write(
            root.join("oracle/check.py"),
            "from solution import VALUE\nassert VALUE == 42\n",
        )
        .unwrap();
        toml::from_str(
            r#"
schema_version = 2
prompt = "set VALUE"
allowed_changes = ["solution.py"]
[final_oracle]
bundle = "oracle"
command = "PYTHONPATH=. python3 .hi-eval-oracle/check.py"
"#,
        )
        .unwrap()
    }

    #[test]
    fn captured_oracle_ignores_candidate_and_source_side_tampering() {
        let root = make_workdir().unwrap();
        let task = oracle_test_task(&root);
        let oracle = CapturedOracle::capture(&root, &task.final_oracle).unwrap();

        // Even mutation of the task-side source after capture cannot change the
        // oracle bytes used for this candidate.
        std::fs::write(root.join("oracle/check.py"), "raise SystemExit(0)\n").unwrap();
        let candidate = make_workdir().unwrap();
        copy_dir(&root.join("fixture"), &candidate).unwrap();
        std::fs::create_dir_all(candidate.join(INJECTED_ORACLE_DIR)).unwrap();
        std::fs::write(
            candidate.join(INJECTED_ORACLE_DIR).join("check.py"),
            "raise SystemExit(0)\n",
        )
        .unwrap();

        assert!(
            !run_final_oracle(&candidate, &oracle, 10).unwrap().success(),
            "candidate-created oracle file must be replaced by captured bytes"
        );
        std::fs::write(candidate.join("solution.py"), "VALUE = 42\n").unwrap();
        assert!(run_final_oracle(&candidate, &oracle, 10).unwrap().success());

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(candidate);
    }

    #[test]
    fn task_validation_checks_fail_before_pass_after_and_allowed_paths() {
        let root = make_workdir().unwrap();
        let task = oracle_test_task(&root);
        validate_task(&root, &task).expect("well-formed immutable-oracle task");
        assert!(
            forbidden_changes(
                &["solution.py".into(), "tests/check.py".into()],
                &task.allowed_changes,
            )
            .unwrap()
            .contains(&"tests/check.py".to_string())
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(unix)]
    fn copy_preserves_modes_and_recreates_only_contained_symlinks() {
        use std::os::unix::fs::PermissionsExt;

        let source = make_workdir().unwrap();
        let destination = make_workdir().unwrap();
        std::fs::create_dir(source.join("bin")).unwrap();
        std::fs::write(source.join("bin/tool"), "#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(source.join("bin"), std::fs::Permissions::from_mode(0o750))
            .unwrap();
        std::fs::set_permissions(
            source.join("bin/tool"),
            std::fs::Permissions::from_mode(0o751),
        )
        .unwrap();
        std::os::unix::fs::symlink("bin/tool", source.join("tool-link")).unwrap();

        copy_dir(&source, &destination).unwrap();

        assert_eq!(
            std::fs::symlink_metadata(destination.join("bin"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o750
        );
        assert_eq!(
            std::fs::symlink_metadata(destination.join("bin/tool"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o751
        );
        assert!(
            std::fs::symlink_metadata(destination.join("tool-link"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            std::fs::read_link(destination.join("tool-link")).unwrap(),
            Path::new("bin/tool")
        );

        let unsafe_source = make_workdir().unwrap();
        std::os::unix::fs::symlink("../../outside", unsafe_source.join("escape")).unwrap();
        let unsafe_destination = make_workdir().unwrap();
        let error = copy_dir(&unsafe_source, &unsafe_destination).unwrap_err();
        assert!(format!("{error:#}").contains("escapes the copied tree"));
        assert!(!unsafe_destination.join("escape").exists());

        for path in [source, destination, unsafe_source, unsafe_destination] {
            let _ = std::fs::remove_dir_all(path);
        }
    }

    #[test]
    #[cfg(unix)]
    fn oracle_capture_preserves_modes_and_isolates_contained_symlinks() {
        use std::os::unix::fs::PermissionsExt;

        let root = make_workdir().unwrap();
        std::fs::create_dir(root.join("oracle")).unwrap();
        std::fs::write(root.join("oracle/real.py"), "raise SystemExit(7)\n").unwrap();
        std::fs::set_permissions(
            root.join("oracle/real.py"),
            std::fs::Permissions::from_mode(0o751),
        )
        .unwrap();
        std::os::unix::fs::symlink("real.py", root.join("oracle/check.py")).unwrap();
        let spec = FinalOracle {
            command: "python3 .hi-eval-oracle/check.py".to_string(),
            bundle: Some("oracle".to_string()),
        };
        let captured = CapturedOracle::capture(&root, &spec).unwrap();

        std::fs::write(root.join("oracle/real.py"), "raise SystemExit(0)\n").unwrap();
        let candidate = make_workdir().unwrap();
        let output = run_final_oracle(&candidate, &captured, 10).unwrap();
        assert_eq!(output.status.code(), Some(7));

        let injection = make_workdir().unwrap();
        captured.inject(&injection).unwrap();
        assert!(
            std::fs::symlink_metadata(injection.join(".hi-eval-oracle/check.py"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            std::fs::symlink_metadata(injection.join(".hi-eval-oracle/real.py"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o751
        );

        let unsafe_root = make_workdir().unwrap();
        std::fs::create_dir(unsafe_root.join("oracle")).unwrap();
        std::os::unix::fs::symlink("../../outside", unsafe_root.join("oracle/check.py")).unwrap();
        assert!(CapturedOracle::capture(&unsafe_root, &spec).is_err());

        for path in [root, candidate, injection, unsafe_root] {
            let _ = std::fs::remove_dir_all(path);
        }
    }

    #[test]
    #[cfg(unix)]
    fn candidate_and_oracle_special_nodes_are_rejected_without_reading_them() {
        use std::os::unix::net::UnixListener;

        let candidate = make_workdir().unwrap();
        let _candidate_socket = UnixListener::bind(candidate.join("socket")).unwrap();
        let destination = make_workdir().unwrap();
        assert!(copy_dir(&candidate, &destination).is_err());
        assert!(directory_snapshot(&candidate).is_err());

        let task = make_workdir().unwrap();
        std::fs::create_dir(task.join("oracle")).unwrap();
        let _oracle_socket = UnixListener::bind(task.join("oracle/socket")).unwrap();
        let spec = FinalOracle {
            command: "true".to_string(),
            bundle: Some("oracle".to_string()),
        };
        let error = CapturedOracle::capture(&task, &spec).unwrap_err();
        assert!(format!("{error:#}").contains("special filesystem node"));

        for path in [candidate, destination, task] {
            let _ = std::fs::remove_dir_all(path);
        }
    }

    #[test]
    fn snapshots_detect_vcs_and_preexisting_ignored_tree_tampering() {
        let root = make_workdir().unwrap();
        std::fs::create_dir_all(root.join(".git/objects")).unwrap();
        std::fs::write(root.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        std::fs::create_dir_all(root.join("target/debug/deps")).unwrap();
        std::fs::write(root.join("target/fixture.txt"), "original\n").unwrap();
        std::fs::create_dir_all(root.join("node_modules/package")).unwrap();
        std::fs::write(root.join("node_modules/package/index.js"), "original\n").unwrap();
        let before = directory_snapshot(&root).unwrap();

        std::fs::write(root.join(".git/HEAD"), "tampered\n").unwrap();
        std::fs::write(root.join("target/fixture.txt"), "tampered\n").unwrap();
        std::fs::write(root.join("node_modules/package/index.js"), "tampered\n").unwrap();
        std::fs::write(root.join("target/forbidden.txt"), "candidate source\n").unwrap();
        std::fs::write(root.join("target/debug/deps/generated.bin"), [0, 1, 2]).unwrap();
        let fanout = root.join(".git/objects/ab");
        std::fs::create_dir(&fanout).unwrap();
        std::fs::write(
            fanout.join("01234567890123456789012345678901234567"),
            [0, 1],
        )
        .unwrap();
        let after = directory_snapshot(&root).unwrap();

        let changed = candidate_changed_paths(&before, &after, &["src/**".to_string()]).unwrap();
        assert!(changed.contains(&".git/HEAD".to_string()));
        assert!(changed.contains(&"target/fixture.txt".to_string()));
        assert!(changed.contains(&"target/forbidden.txt".to_string()));
        assert!(changed.contains(&"node_modules/package/index.js".to_string()));
        assert!(!changed.iter().any(|path| path.contains("generated.bin")));
        assert!(
            !changed
                .iter()
                .any(|path| path.starts_with(".git/objects/ab"))
        );

        let forbidden = forbidden_changes(&changed, &["src/**".to_string()]).unwrap();
        assert!(forbidden.contains(&".git/HEAD".to_string()));
        assert!(forbidden.contains(&"target/fixture.txt".to_string()));
        assert!(forbidden.contains(&"target/forbidden.txt".to_string()));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn classified_runtime_artifacts_are_not_hidden_oracle_inputs() {
        let root = make_workdir().unwrap();
        std::fs::create_dir_all(root.join("target/debug/deps")).unwrap();
        let before = directory_snapshot(&root).unwrap();
        std::fs::write(root.join("target/debug/deps/cheat.bin"), [0, 42]).unwrap();
        let after = directory_snapshot(&root).unwrap();
        let runtime =
            candidate_runtime_artifact_paths(&before, &after, &["solution.py".to_string()])
                .unwrap();
        assert!(runtime.contains(&"target/debug/deps/cheat.bin".to_string()));

        let oracle = CapturedOracle::capture(
            &root,
            &FinalOracle {
                command: "test -e target/debug/deps/cheat.bin".to_string(),
                bundle: None,
            },
        )
        .unwrap();
        assert!(
            !run_final_oracle_without_artifacts(&root, &oracle, 10, &runtime)
                .unwrap()
                .success(),
            "a classified artifact must be removed before the oracle runs"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn check_timeout_is_typed_and_cannot_pass() {
        let root = make_workdir().unwrap();
        let output = verify_output_in(&root, "sleep 2", 1).unwrap();
        assert!(output.timed_out);
        assert!(!output.success());
        assert!(output.duration >= Duration::from_secs(1));
        let _ = std::fs::remove_dir_all(root);
    }
}
