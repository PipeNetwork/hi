//! Shared contracts for coding races.
//!
//! This crate intentionally does not know about providers, TUI widgets, or
//! approval storage. The CLI owns those boundaries; this crate owns the
//! redaction-safe run manifest, workspace snapshot, stage execution, and
//! deterministic candidate ranking used by every frontend.

use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_MAX_CANDIDATES: u32 = 2;
pub const MAX_CANDIDATES: u32 = 4;
pub const DEFAULT_FUZZ_TIMEOUT_SECS: u64 = 120;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RaceTarget {
    pub name: String,
    pub profile: String,
    pub model: String,
    #[serde(default)]
    pub priority: u32,
}

impl RaceTarget {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.name.trim().is_empty(),
            "race target name cannot be empty"
        );
        ensure!(
            !self.profile.trim().is_empty(),
            "race target profile cannot be empty"
        );
        ensure!(
            !self.model.trim().is_empty(),
            "race target model cannot be empty"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuzzConfig {
    pub command: String,
    #[serde(default = "default_fuzz_timeout")]
    pub timeout_secs: u64,
}

fn default_fuzz_timeout() -> u64 {
    DEFAULT_FUZZ_TIMEOUT_SECS
}

impl FuzzConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.command.trim().is_empty(),
            "race fuzz command cannot be empty"
        );
        ensure!(
            self.timeout_secs > 0,
            "race fuzz timeout must be greater than zero"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RaceSpec {
    pub schema_version: u32,
    pub run_id: String,
    pub task: String,
    pub targets: Vec<RaceTarget>,
    pub verify_commands: Vec<String>,
    #[serde(default)]
    pub fuzz: Option<FuzzConfig>,
    #[serde(default = "default_max_candidates")]
    pub max_candidates: u32,
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    pub workspace_digest: String,
}

fn default_max_candidates() -> u32 {
    DEFAULT_MAX_CANDIDATES
}

fn default_max_concurrency() -> usize {
    2
}

impl RaceSpec {
    pub fn new(task: impl Into<String>, targets: Vec<RaceTarget>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            run_id: uuid::Uuid::new_v4().to_string(),
            task: task.into(),
            targets,
            verify_commands: Vec::new(),
            fuzz: None,
            max_candidates: DEFAULT_MAX_CANDIDATES,
            max_concurrency: 2,
            workspace_digest: String::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == SCHEMA_VERSION,
            "unsupported race schema version {}",
            self.schema_version
        );
        ensure!(!self.task.trim().is_empty(), "race task cannot be empty");
        ensure!(self.targets.len() >= 2, "a race needs at least two targets");
        ensure!(
            self.targets.len() <= MAX_CANDIDATES as usize,
            "a race may have at most {MAX_CANDIDATES} targets"
        );
        ensure!(
            self.max_candidates >= 2,
            "race max_candidates must be at least two"
        );
        ensure!(
            self.max_candidates <= MAX_CANDIDATES,
            "race max_candidates exceeds {MAX_CANDIDATES}"
        );
        ensure!(
            self.max_concurrency > 0,
            "race max_concurrency must be greater than zero"
        );
        ensure!(
            !self.verify_commands.is_empty(),
            "a race requires at least one verification command"
        );
        for command in &self.verify_commands {
            ensure!(
                !command.trim().is_empty(),
                "race verification commands cannot be empty"
            );
        }
        for target in &self.targets {
            target.validate()?;
        }
        if let Some(fuzz) = &self.fuzz {
            fuzz.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotFile {
    pub path: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceSnapshot {
    pub root: PathBuf,
    pub revision: Option<String>,
    pub tracked_patch: Vec<u8>,
    pub untracked_files: Vec<SnapshotFile>,
    pub digest: String,
}

impl WorkspaceSnapshot {
    /// Reconstruct the exact dirty workspace state in `destination`, which is
    /// used by race worktrees before a candidate starts.
    pub fn materialize_into(&self, destination: &Path) -> Result<()> {
        ensure!(
            destination.is_dir(),
            "race snapshot destination is not a directory"
        );
        if !self.tracked_patch.is_empty() {
            let mut child = Command::new("git")
                .arg("-C")
                .arg(destination)
                .args(["apply", "--whitespace=nowarn"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .context("starting git apply for race workspace snapshot")?;
            use std::io::Write;
            child
                .stdin
                .take()
                .context("opening git apply input")?
                .write_all(&self.tracked_patch)?;
            let output = child.wait_with_output()?;
            ensure!(
                output.status.success(),
                "could not materialize tracked race snapshot: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        for file in &self.untracked_files {
            let relative = safe_relative_path(&file.path)?;
            let target = destination.join(relative);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(target, &file.bytes)?;
        }
        Ok(())
    }
}

pub fn capture_workspace_snapshot(root: &Path) -> Result<WorkspaceSnapshot> {
    ensure!(root.is_dir(), "workspace root is not a directory");
    let revision = git_output(root, &["rev-parse", "HEAD"]).ok();
    let tracked_patch = git_output_bytes(root, &["diff", "--binary", "HEAD"]).unwrap_or_default();
    let names = git_output_bytes(root, &["ls-files", "--others", "--exclude-standard", "-z"])
        .unwrap_or_default();
    let mut untracked_files = Vec::new();
    for raw in names
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        let path =
            String::from_utf8(raw.to_vec()).context("untracked workspace path is not UTF-8")?;
        let relative = safe_relative_path(&path)?;
        let bytes = fs::read(root.join(relative))
            .with_context(|| format!("reading untracked workspace file {path}"))?;
        untracked_files.push(SnapshotFile { path, bytes });
    }
    untracked_files.sort_by(|left, right| left.path.cmp(&right.path));
    let mut hasher = blake3::Hasher::new();
    hasher.update(&tracked_patch);
    for file in &untracked_files {
        hasher.update(file.path.as_bytes());
        hasher.update(&file.bytes);
    }
    Ok(WorkspaceSnapshot {
        root: root.to_path_buf(),
        revision,
        tracked_patch,
        untracked_files,
        digest: hasher.finalize().to_hex().to_string(),
    })
}

fn git_output(root: &Path, args: &[&str]) -> Result<String> {
    String::from_utf8(git_output_bytes(root, args)?).context("git output is not UTF-8")
}

fn git_output_bytes(root: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()?;
    ensure!(output.status.success(), "git {} failed", args.join(" "));
    Ok(output.stdout)
}

fn safe_relative_path(path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    ensure!(!path.is_absolute(), "race snapshot path must be relative");
    for component in path.components() {
        ensure!(
            !matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            ),
            "race snapshot path escapes workspace: {}",
            path.display()
        );
    }
    Ok(path.to_path_buf())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateState {
    #[default]
    Pending,
    Running,
    Verifying,
    Fuzzing,
    Passed,
    Failed,
    Cancelled,
    Abandoned,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageResult {
    pub name: String,
    pub command: String,
    pub passed: bool,
    pub timed_out: bool,
    pub duration_ms: u128,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateReport {
    pub candidate_id: String,
    pub target: RaceTarget,
    pub state: CandidateState,
    pub process_succeeded: bool,
    pub report_matches_diff: bool,
    pub actual_changes: Vec<String>,
    pub changed_lines: u64,
    pub verify: Vec<StageResult>,
    pub fuzz: Option<StageResult>,
    pub wall_clock_ms: u128,
    pub cost_microusd: Option<u64>,
    pub artifact_ref: Option<String>,
    pub failure_reason: Option<String>,
}

impl CandidateReport {
    pub fn eligible(&self) -> bool {
        self.process_succeeded
            && self.report_matches_diff
            && !self.actual_changes.is_empty()
            && self.verify.iter().all(|stage| stage.passed)
            && self.fuzz.as_ref().is_none_or(|stage| stage.passed)
            && matches!(self.state, CandidateState::Passed)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RaceScore {
    pub candidate_id: String,
    pub eligible: bool,
    pub changed_files: usize,
    pub changed_lines: u64,
    pub wall_clock_ms: u128,
    pub cost_microusd: Option<u64>,
    pub target_priority: u32,
}

pub fn score(candidate: &CandidateReport) -> RaceScore {
    RaceScore {
        candidate_id: candidate.candidate_id.clone(),
        eligible: candidate.eligible(),
        changed_files: candidate.actual_changes.len(),
        changed_lines: candidate.changed_lines,
        wall_clock_ms: candidate.wall_clock_ms,
        cost_microusd: candidate.cost_microusd,
        target_priority: candidate.target.priority,
    }
}

/// Return the deterministic winner without mutating candidate order. Failed
/// candidates are always excluded before any quality tie-breaker is applied.
pub fn select_winner(candidates: &[CandidateReport]) -> Option<String> {
    candidates
        .iter()
        .filter(|candidate| candidate.eligible())
        .min_by_key(|candidate| {
            let candidate_score = score(candidate);
            (
                candidate_score.changed_files,
                candidate_score.changed_lines,
                candidate_score.wall_clock_ms,
                candidate_score.cost_microusd.unwrap_or(u64::MAX),
                candidate_score.target_priority,
                candidate_score.candidate_id.clone(),
            )
        })
        .map(|candidate| candidate.candidate_id.clone())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RaceStatus {
    #[default]
    Pending,
    Running,
    Ready,
    Applied,
    NoWinner,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RaceSnapshot {
    pub schema_version: u32,
    pub run_id: String,
    pub status: RaceStatus,
    pub workspace_digest: String,
    pub candidates: Vec<CandidateReport>,
    pub selected_candidate: Option<String>,
    pub artifact_root: Option<PathBuf>,
    pub error: Option<String>,
}

impl RaceSnapshot {
    pub fn pending(spec: &RaceSpec) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            run_id: spec.run_id.clone(),
            status: RaceStatus::Pending,
            workspace_digest: spec.workspace_digest.clone(),
            candidates: spec
                .targets
                .iter()
                .enumerate()
                .map(|(index, target)| CandidateReport {
                    candidate_id: format!("candidate-{index}"),
                    target: target.clone(),
                    state: CandidateState::Pending,
                    process_succeeded: false,
                    report_matches_diff: false,
                    actual_changes: Vec::new(),
                    changed_lines: 0,
                    verify: Vec::new(),
                    fuzz: None,
                    wall_clock_ms: 0,
                    cost_microusd: None,
                    artifact_ref: None,
                    failure_reason: None,
                })
                .collect(),
            selected_candidate: None,
            artifact_root: None,
            error: None,
        }
    }
}

fn run_stage_with_timeout(
    root: &Path,
    name: &str,
    command: &str,
    timeout: Option<Duration>,
) -> StageResult {
    let started = Instant::now();
    let mut process = Command::new("sh");
    process
        .args(["-c", command])
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        process.process_group(0);
    }
    let mut child = match process.spawn() {
        Ok(child) => child,
        Err(error) => {
            return StageResult {
                name: name.to_string(),
                command: command.to_string(),
                passed: false,
                timed_out: false,
                duration_ms: started.elapsed().as_millis(),
                detail: error.to_string(),
            };
        }
    };
    let stdout = child.stdout.take().expect("stage stdout is piped");
    let stderr = child.stderr.take().expect("stage stderr is piped");
    let stdout_reader = thread::spawn(move || drain_stage_output(stdout));
    let stderr_reader = thread::spawn(move || drain_stage_output(stderr));
    let mut timed_out = false;
    let mut wait_error = None;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if timeout.is_some_and(|timeout| started.elapsed() >= timeout) => {
                timed_out = true;
                kill_stage_process_group(&mut child);
                break;
            }
            Ok(None) => thread::sleep(Duration::from_millis(20)),
            Err(error) => {
                kill_stage_process_group(&mut child);
                wait_error = Some(error);
                break;
            }
        }
    }
    // A shell can exit after daemonizing a descendant that still owns its
    // stdout/stderr pipes. Reap the private group before draining output so a
    // completed stage cannot leak work or wedge `wait_with_output` forever.
    kill_stage_process_group(&mut child);
    let status = child.wait();
    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();
    let (passed, detail) = match (wait_error, status) {
        (Some(error), _) => (false, error.to_string()),
        (None, Ok(status)) => (
            !timed_out && status.success(),
            bounded_detail(&format!(
                "{}{}",
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&stderr)
            )),
        ),
        (None, Err(error)) => (false, error.to_string()),
    };
    StageResult {
        name: name.to_string(),
        command: command.to_string(),
        passed,
        timed_out,
        duration_ms: started.elapsed().as_millis(),
        detail,
    }
}

const MAX_STAGE_CAPTURE_BYTES: usize = 64 * 1024;

/// Drain a child pipe continuously while retaining bounded head/tail evidence.
/// Continuous draining prevents a noisy healthy verifier from blocking on the
/// OS pipe buffer; bounded retention keeps continual runs memory-safe.
fn drain_stage_output(mut reader: impl Read) -> Vec<u8> {
    const HEAD_BYTES: usize = MAX_STAGE_CAPTURE_BYTES / 2;
    const TAIL_BYTES: usize = MAX_STAGE_CAPTURE_BYTES - HEAD_BYTES;
    const OMITTED: &[u8] = b"\n[... stage output truncated ...]\n";

    let mut head = Vec::with_capacity(HEAD_BYTES);
    let mut tail = std::collections::VecDeque::with_capacity(TAIL_BYTES);
    let mut total = 0usize;
    let mut buffer = [0u8; 8 * 1024];
    loop {
        let count = match reader.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        total = total.saturating_add(count);
        for &byte in &buffer[..count] {
            if head.len() < HEAD_BYTES {
                head.push(byte);
            } else {
                if tail.len() == TAIL_BYTES {
                    tail.pop_front();
                }
                tail.push_back(byte);
            }
        }
    }

    if total > MAX_STAGE_CAPTURE_BYTES {
        head.extend_from_slice(OMITTED);
    }
    head.extend(tail);
    head
}

#[cfg(unix)]
fn kill_stage_process_group(child: &mut std::process::Child) {
    let process_group = child.id() as libc::pid_t;
    // SAFETY: a negative PID addresses only the private process group created
    // for this child. ESRCH is harmless when the leader and descendants have
    // already exited.
    unsafe {
        libc::kill(-process_group, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_stage_process_group(child: &mut std::process::Child) {
    let _ = child.kill();
}

/// Run a stage with an explicit caller-selected wall-clock limit.
pub fn run_stage(root: &Path, name: &str, command: &str, timeout: Duration) -> StageResult {
    run_stage_with_timeout(root, name, command, Some(timeout))
}

/// Run verifier stages without an implicit wall-clock ceiling. A verifier is
/// productive work, so the ordinary path waits for completion; callers that
/// need a bounded fault campaign can use [`run_stage`] with an explicit limit.
pub fn run_verification(root: &Path, commands: &[String]) -> Vec<StageResult> {
    commands
        .iter()
        .enumerate()
        .scan(true, |continue_running, (index, command)| {
            if !*continue_running {
                return None;
            }
            let stage =
                run_stage_with_timeout(root, &format!("verify-{}", index + 1), command, None);
            *continue_running = stage.passed;
            Some(stage)
        })
        .collect()
}

pub fn run_fuzz(root: &Path, config: &FuzzConfig) -> StageResult {
    run_stage(
        root,
        "fuzz",
        &config.command,
        Duration::from_secs(config.timeout_secs),
    )
}

pub fn changed_lines(root: &Path) -> u64 {
    let Ok(output) = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["diff", "--numstat"])
        .output()
    else {
        return 0;
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let mut fields = line.split('\t');
            let added = fields.next()?.parse::<u64>().ok()?;
            let removed = fields.next()?.parse::<u64>().ok()?;
            Some(added.saturating_add(removed))
        })
        .sum()
}

fn bounded_detail(detail: &str) -> String {
    const MAX_DETAIL: usize = 4096;
    if detail.len() <= MAX_DETAIL {
        detail.trim().to_string()
    } else {
        let truncated = detail.chars().take(MAX_DETAIL).collect::<String>();
        format!("{truncated}…")
    }
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn target(name: &str, priority: u32) -> RaceTarget {
        RaceTarget {
            name: name.into(),
            profile: "profile".into(),
            model: name.into(),
            priority,
        }
    }

    fn candidate(id: &str, target: RaceTarget, files: usize, lines: u64) -> CandidateReport {
        CandidateReport {
            candidate_id: id.into(),
            target,
            state: CandidateState::Passed,
            process_succeeded: true,
            report_matches_diff: true,
            actual_changes: (0..files).map(|i| format!("file-{i}")).collect(),
            changed_lines: lines,
            verify: vec![StageResult {
                name: "verify".into(),
                command: "true".into(),
                passed: true,
                timed_out: false,
                duration_ms: 1,
                detail: String::new(),
            }],
            fuzz: None,
            wall_clock_ms: 10,
            cost_microusd: Some(1),
            artifact_ref: None,
            failure_reason: None,
        }
    }

    #[test]
    fn ranking_excludes_failed_candidates_and_prefers_smallest_diff() {
        let mut failed = candidate("failed", target("failed", 0), 0, 0);
        failed.state = CandidateState::Failed;
        let winner = candidate("winner", target("winner", 1), 1, 5);
        let larger = candidate("larger", target("larger", 0), 2, 2);
        assert_eq!(
            select_winner(&[failed, larger, winner]),
            Some("winner".into())
        );
    }

    #[test]
    fn spec_requires_verification_and_two_targets() {
        let mut spec = RaceSpec::new("fix it", vec![target("fast", 0)]);
        assert!(spec.validate().is_err());
        spec.targets.push(target("strong", 1));
        spec.verify_commands.push("true".into());
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn snapshot_captures_untracked_files_and_materializes_them() {
        let source = tempdir().unwrap();
        let output = Command::new("git")
            .args(["init", "-q"])
            .current_dir(source.path())
            .output()
            .unwrap();
        assert!(output.status.success());
        fs::write(source.path().join("new.txt"), b"hello").unwrap();
        let snapshot = capture_workspace_snapshot(source.path()).unwrap();
        assert_eq!(snapshot.untracked_files[0].path, "new.txt");
        let destination = tempdir().unwrap();
        let init = Command::new("git")
            .args(["init", "-q"])
            .current_dir(destination.path())
            .output()
            .unwrap();
        assert!(init.status.success());
        snapshot.materialize_into(destination.path()).unwrap();
        assert_eq!(
            fs::read(destination.path().join("new.txt")).unwrap(),
            b"hello"
        );
    }

    #[test]
    fn snapshot_captures_tracked_edits_against_head() {
        let source = tempdir().unwrap();
        for args in [
            &["init", "-q"][..],
            &["config", "user.email", "race@example.test"][..],
            &["config", "user.name", "Race Test"][..],
        ] {
            let output = Command::new("git")
                .args(args)
                .current_dir(source.path())
                .output()
                .unwrap();
            assert!(output.status.success());
        }
        fs::write(source.path().join("tracked.txt"), b"before\n").unwrap();
        assert!(
            Command::new("git")
                .args(["add", "tracked.txt"])
                .current_dir(source.path())
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .args(["commit", "-qm", "initial"])
                .current_dir(source.path())
                .status()
                .unwrap()
                .success()
        );
        fs::write(source.path().join("tracked.txt"), b"after\n").unwrap();
        let snapshot = capture_workspace_snapshot(source.path()).unwrap();
        assert!(!snapshot.tracked_patch.is_empty());
        let destination = tempdir().unwrap();
        assert!(
            Command::new("git")
                .args(["clone", "-q"])
                .arg(source.path())
                .arg(destination.path())
                .status()
                .unwrap()
                .success()
        );
        snapshot.materialize_into(destination.path()).unwrap();
        assert_eq!(
            fs::read(destination.path().join("tracked.txt")).unwrap(),
            b"after\n"
        );
    }

    #[test]
    fn stage_timeout_is_visible() {
        let dir = tempdir().unwrap();
        let result = run_stage(dir.path(), "timeout", "sleep 1", Duration::from_millis(10));
        assert!(result.timed_out);
        assert!(!result.passed);
    }

    #[test]
    fn verification_uses_the_unlimited_stage_path() {
        let dir = tempdir().unwrap();
        let results = run_verification(dir.path(), &["sleep 0.02".into()]);
        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
        assert!(!results[0].timed_out);
    }

    #[test]
    fn unlimited_verification_continuously_drains_noisy_output() {
        let dir = tempdir().unwrap();
        let command = "i=0; while [ $i -lt 20000 ]; do printf 'stdout-abcdefghijklmnopqrstuvwxyz-0123456789\\n'; printf 'stderr-abcdefghijklmnopqrstuvwxyz-0123456789\\n' >&2; i=$((i + 1)); done";
        let results = run_verification(dir.path(), &[command.into()]);

        assert_eq!(results.len(), 1);
        assert!(results[0].passed, "noisy verifier failed: {results:?}");
        assert!(!results[0].timed_out);
        assert!(
            results[0].detail.len() <= 4_100,
            "stage report must remain bounded"
        );
    }

    #[cfg(unix)]
    #[test]
    fn completed_stage_reaps_descendants_that_hold_output_pipes() {
        let dir = tempdir().unwrap();
        let started = Instant::now();
        let result = run_stage(
            dir.path(),
            "detached-child",
            "sleep 3 & echo $! > child.pid",
            Duration::from_secs(1),
        );

        assert!(result.passed, "top-level shell should complete: {result:?}");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "output drain waited for the detached descendant"
        );
        let pid: libc::pid_t = fs::read_to_string(dir.path().join("child.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            // SAFETY: signal 0 only probes the PID written by this test and
            // never changes process state.
            let alive = unsafe { libc::kill(pid, 0) == 0 };
            if !alive || Instant::now() >= deadline {
                assert!(!alive, "stage descendant {pid} was left alive");
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
}
