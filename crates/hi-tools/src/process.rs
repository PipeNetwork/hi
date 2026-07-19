use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::process::Command;

use crate::{ProcessOutcome, ToolStatus, TruncationState};

/// Maximum bytes retained from each process stream before the middle is
/// discarded. The reader continues draining after the cap so a noisy child can
/// never deadlock on a full pipe.
const MAX_CAPTURE_BYTES: usize = 2 * 1024 * 1024;

/// Environment variables which must never be inherited by model-controlled
/// processes. Everything else is retained so compilers and project-local tool
/// chains keep working.
const SECRET_ENV_VARS: &[&str] = &[
    "HI_API_KEY",
    "HI_WEB_SEARCH_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "PIPENETWORK_API_KEY",
    "OLLAMA_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_TOKEN",
];

/// The structured result returned by [`ProcessRunner`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessExecution {
    pub status: ToolStatus,
    pub outcome: ProcessOutcome,
    pub truncation: TruncationState,
}

impl ProcessExecution {
    /// Compatibility-friendly model text. Status remains authoritative.
    pub fn model_content(&self) -> String {
        let mut out = self.outcome.stdout_summary.clone();
        if !self.outcome.stderr_summary.is_empty() {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&self.outcome.stderr_summary);
        }
        match self.status {
            ToolStatus::Failed => {
                if let Some(code) = self.outcome.exit_code {
                    if !out.is_empty() && !out.ends_with('\n') {
                        out.push('\n');
                    }
                    out.push_str(&format!("[exit code {code}]"));
                }
            }
            ToolStatus::TimedOut => {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("[timed out — process killed]");
            }
            _ => {}
        }
        if out.is_empty() {
            out.push_str("[no output]");
        }
        out
    }
}

/// Hardened process runner bound to one explicit workspace root.
///
/// Every child gets a closed stdin, bounded/drained stdout and stderr, a
/// sanitized environment, kill-on-drop, and (on Unix) its own process group so
/// cancellation and timeout remove descendants as well as the shell.
#[derive(Clone, Debug)]
pub struct ProcessRunner {
    root: PathBuf,
    /// Resolved OS sandbox for shell commands (`HI_SANDBOX`). Off by default so
    /// home-dir tool caches keep working; set `HI_SANDBOX=workspace` to confine
    /// writes (macOS enforced today — see `docs/sandbox.md`).
    sandbox: crate::sandbox::SandboxProfile,
}

impl ProcessRunner {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let metadata = std::fs::metadata(root)
            .with_context(|| format!("reading workspace root {}", root.display()))?;
        anyhow::ensure!(
            metadata.is_dir(),
            "workspace root is not a directory: {}",
            root.display()
        );
        let root = root
            .canonicalize()
            .with_context(|| format!("canonicalizing workspace root {}", root.display()))?;
        let sandbox = crate::sandbox::SandboxProfile::new(
            crate::sandbox::SandboxPolicy::from_env(),
            &[root.as_path()],
        );
        Ok(Self { root, sandbox })
    }

    /// Whether shell commands from this runner are OS-sandboxed on this platform.
    pub fn sandbox_enforced(&self) -> bool {
        self.sandbox.is_enforced()
    }

    #[cfg(test)]
    pub(crate) fn from_current_dir() -> Result<Self> {
        Self::new(std::env::current_dir().context("determining working directory")?)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub async fn run_shell(&self, command: &str, timeout: Duration) -> Result<ProcessExecution> {
        self.run_shell_streaming(command, timeout, &mut |_| {})
            .await
    }

    pub async fn run_shell_streaming(
        &self,
        command: &str,
        timeout: Duration,
        on_line: &mut (dyn FnMut(&str) + Send),
    ) -> Result<ProcessExecution> {
        let started = Instant::now();
        let child = self.spawn_shell(command)?;
        capture_child(child, timeout, on_line, started).await
    }

    /// Run a shell command in the foreground up to `foreground_budget`; if it is
    /// still running at the deadline, return the live child for adoption into the
    /// background registry instead of killing it. A command that finishes in
    /// time yields a normal [`ProcessExecution`] (full 2 MB output + condense),
    /// identical to [`run_shell_streaming`].
    pub async fn run_shell_adoptable(
        &self,
        command: &str,
        foreground_budget: Duration,
        on_line: &mut (dyn FnMut(&str) + Send),
    ) -> Result<AdoptableOutcome> {
        let started = Instant::now();
        let child = self.spawn_shell(command)?;
        capture_child_adoptable(child, foreground_budget, on_line, started).await
    }

    /// Run an executable directly, keeping filesystem-derived arguments out of
    /// a shell parser. This is used for filename-sensitive syntax checks and
    /// other internal commands.
    pub async fn run_program<I, S>(
        &self,
        program: impl AsRef<OsStr>,
        args: I,
        timeout: Duration,
    ) -> Result<ProcessExecution>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let started = Instant::now();
        let mut command = Command::new(program);
        command.args(args);
        self.configure(&mut command);
        let child = command.spawn().context("failed to spawn program")?;
        capture_child(child, timeout, &mut |_| {}, started).await
    }

    /// Run a trusted executable directly with explicit environment overrides.
    ///
    /// The inherited environment is sanitized first; only the supplied values
    /// are added back. This is intended for internal child processes which need
    /// a narrowly scoped credential without exposing every parent-process
    /// secret.
    pub async fn run_program_with_env<I, S, E, K, V>(
        &self,
        program: impl AsRef<OsStr>,
        args: I,
        environment: E,
        timeout: Duration,
    ) -> Result<ProcessExecution>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
        E: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        let started = Instant::now();
        let mut command = Command::new(program);
        command.args(args);
        self.configure(&mut command);
        command.envs(environment);
        let child = command.spawn().context("failed to spawn program")?;
        capture_child(child, timeout, &mut |_| {}, started).await
    }

    fn configure(&self, command: &mut Command) {
        command
            .current_dir(&self.root)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .env("AI_AGENT", "hi");
        for var in SECRET_ENV_VARS {
            command.env_remove(var);
        }
        for (name, _) in std::env::vars_os() {
            if sensitive_environment_name(&name) {
                command.env_remove(name);
            }
        }
        command
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("PYTHONDONTWRITEBYTECODE", "1");
        // Pager neutralization: point every pager a common tool might launch at
        // a passthrough (`cat`) and blank the ones with no passthrough form, so
        // `git log`, `gh`, `man`, `systemctl`, `aws`, … stream their output
        // instead of blocking on an interactive pager the agent can't drive.
        // stdin is already null; this covers pagers that ignore a closed stdin.
        command
            .env("PAGER", "cat")
            .env("GIT_PAGER", "cat")
            .env("GH_PAGER", "cat")
            .env("MANPAGER", "cat")
            .env("SYSTEMD_PAGER", "")
            .env("AWS_PAGER", "");
        #[cfg(unix)]
        command.process_group(0);
    }

    /// Spawn a child for the background registry. The registry is responsible
    /// for draining and reaping it. When a sandbox policy is active (and the
    /// platform enforces it), the command runs confined via the sandbox wrapper
    /// (e.g. `sandbox-exec` on macOS); otherwise it's a plain `sh -c`.
    pub(crate) fn spawn_shell(&self, command: &str) -> Result<tokio::process::Child> {
        let (program, args) = self.sandbox.wrap(command);
        let mut cmd = Command::new(program);
        cmd.args(args);
        self.configure(&mut cmd);
        cmd.spawn().context("failed to spawn command")
    }
}

fn sensitive_environment_name(name: &OsStr) -> bool {
    let name = name.to_string_lossy().to_ascii_uppercase();
    [
        "API_KEY",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "AUTH_COOKIE",
        "SESSION_COOKIE",
    ]
    .iter()
    .any(|marker| name.contains(marker))
}

async fn capture_child(
    mut child: tokio::process::Child,
    timeout: Duration,
    on_line: &mut (dyn FnMut(&str) + Send),
    started: Instant,
) -> Result<ProcessExecution> {
    let group_guard = ProcessGroupDropGuard::for_child(&child);
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();

    let callback: &Mutex<&mut (dyn FnMut(&str) + Send)> = &Mutex::new(on_line);
    let stdout_buf = Mutex::new(BoundedBuffer::default());
    let stderr_buf = Mutex::new(BoundedBuffer::default());

    let (status, exit_code) = {
        let combined = async {
            tokio::join!(
                read_stream(&mut stdout, callback, &stdout_buf),
                read_stream(&mut stderr, callback, &stderr_buf),
            );
            child.wait().await
        };
        match tokio::time::timeout(timeout, combined).await {
            Ok(Ok(exit)) if exit.success() => (ToolStatus::Succeeded, exit.code()),
            Ok(Ok(exit)) => (ToolStatus::Failed, exit.code()),
            Ok(Err(err)) => return Err(err).context("waiting for command"),
            Err(_) => {
                kill_process_group(&child);
                let _ = child.kill().await;
                (ToolStatus::TimedOut, None)
            }
        }
    };
    drop(group_guard);

    Ok(build_execution(
        stdout_buf.into_inner().unwrap_or_default(),
        stderr_buf.into_inner().unwrap_or_default(),
        status,
        exit_code,
        started,
    ))
}

/// A live child handed back by [`ProcessRunner::run_shell_adoptable`] because it
/// exceeded its foreground budget while still running. The caller adopts it into
/// the background registry (keeping it alive) rather than killing it.
pub struct RunningChild {
    pub child: tokio::process::Child,
    pub stdout: Option<tokio::process::ChildStdout>,
    pub stderr: Option<tokio::process::ChildStderr>,
    pub pgid: Option<i32>,
    /// The combined stdout+stderr produced while in the foreground, to seed the
    /// background handle so a later poll shows the whole run.
    pub partial_output: String,
}

/// Either the command completed within the foreground budget, or it is still
/// running and eligible for adoption into the background.
pub enum AdoptableOutcome {
    Completed(ProcessExecution),
    StillRunning(RunningChild),
}

fn build_execution(
    stdout: BoundedBuffer,
    stderr: BoundedBuffer,
    status: ToolStatus,
    exit_code: Option<i32>,
    started: Instant,
) -> ProcessExecution {
    let stdout_summary = crate::condense::condense(stdout.data.trim_end());
    let stderr_summary = crate::condense::condense(stderr.data.trim_end());
    let original_bytes = stdout.total_bytes.saturating_add(stderr.total_bytes) as u64;
    let retained_bytes = stdout_summary.len().saturating_add(stderr_summary.len()) as u64;
    let truncation = if stdout.truncated || stderr.truncated || retained_bytes < original_bytes {
        TruncationState::Truncated {
            original_bytes,
            retained_bytes,
        }
    } else {
        TruncationState::Complete
    };
    ProcessExecution {
        status,
        outcome: ProcessOutcome {
            exit_code,
            stdout_summary,
            stderr_summary,
            duration_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        },
        truncation,
    }
}

/// Like [`capture_child`], but on hitting the foreground budget the still-running
/// child is returned for adoption instead of being killed. The process-group
/// kill guard is defused on that path so the child survives the handoff.
async fn capture_child_adoptable(
    mut child: tokio::process::Child,
    foreground_budget: Duration,
    on_line: &mut (dyn FnMut(&str) + Send),
    started: Instant,
) -> Result<AdoptableOutcome> {
    let mut group_guard = ProcessGroupDropGuard::for_child(&child);
    let pgid = child.id().map(|pid| pid as i32);
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();

    let callback: &Mutex<&mut (dyn FnMut(&str) + Send)> = &Mutex::new(on_line);
    let stdout_buf = Mutex::new(BoundedBuffer::default());
    let stderr_buf = Mutex::new(BoundedBuffer::default());

    let timed_out = {
        let combined = async {
            tokio::join!(
                read_stream(&mut stdout, callback, &stdout_buf),
                read_stream(&mut stderr, callback, &stderr_buf),
            );
            child.wait().await
        };
        match tokio::time::timeout(foreground_budget, combined).await {
            Ok(Ok(exit)) if exit.success() => {
                drop(group_guard);
                return Ok(AdoptableOutcome::Completed(build_execution(
                    stdout_buf.into_inner().unwrap_or_default(),
                    stderr_buf.into_inner().unwrap_or_default(),
                    ToolStatus::Succeeded,
                    exit.code(),
                    started,
                )));
            }
            Ok(Ok(exit)) => {
                drop(group_guard);
                return Ok(AdoptableOutcome::Completed(build_execution(
                    stdout_buf.into_inner().unwrap_or_default(),
                    stderr_buf.into_inner().unwrap_or_default(),
                    ToolStatus::Failed,
                    exit.code(),
                    started,
                )));
            }
            Ok(Err(err)) => return Err(err).context("waiting for command"),
            Err(_) => true,
        }
    };

    debug_assert!(timed_out);
    // Still running at the budget: hand the live child off. Defuse the guard so
    // dropping it here does not kill the group the registry is about to own.
    group_guard.defuse();
    let partial = {
        let stdout = stdout_buf.into_inner().unwrap_or_default();
        let stderr = stderr_buf.into_inner().unwrap_or_default();
        let mut combined = stdout.data;
        if !stderr.data.is_empty() {
            if !combined.is_empty() && !combined.ends_with('\n') {
                combined.push('\n');
            }
            combined.push_str(&stderr.data);
        }
        combined
    };
    Ok(AdoptableOutcome::StillRunning(RunningChild {
        child,
        stdout,
        stderr,
        pgid,
        partial_output: partial,
    }))
}

#[derive(Default)]
struct BoundedBuffer {
    data: String,
    total_bytes: usize,
    truncated: bool,
}

impl BoundedBuffer {
    fn push(&mut self, text: &str) {
        self.total_bytes = self.total_bytes.saturating_add(text.len());
        self.data.push_str(text);
        if self.data.len() <= MAX_CAPTURE_BYTES {
            return;
        }
        self.truncated = true;
        let head_target = MAX_CAPTURE_BYTES * 3 / 5;
        let tail_target = MAX_CAPTURE_BYTES - head_target;
        let head_end = char_boundary_at_or_before(&self.data, head_target);
        let tail_start =
            char_boundary_at_or_after(&self.data, self.data.len().saturating_sub(tail_target));
        self.data = format!(
            "{}\n… [process output middle truncated] …\n{}",
            &self.data[..head_end],
            &self.data[tail_start..]
        );
        if self.data.len() > MAX_CAPTURE_BYTES {
            let end = char_boundary_at_or_before(&self.data, MAX_CAPTURE_BYTES);
            self.data.truncate(end);
        }
    }
}

async fn read_stream<R: tokio::io::AsyncRead + Unpin>(
    pipe: &mut Option<R>,
    on_line: &Mutex<&mut (dyn FnMut(&str) + Send)>,
    buffer: &Mutex<BoundedBuffer>,
) {
    let Some(pipe) = pipe.as_mut() else { return };
    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut reader = BufReader::new(pipe);
    let mut bytes = Vec::new();
    loop {
        bytes.clear();
        match reader.read_until(b'\n', &mut bytes).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let mut line = String::from_utf8_lossy(&bytes)
            .trim_end_matches(['\r', '\n'])
            .to_string();
        line.push('\n');
        if let Ok(mut callback) = on_line.lock() {
            (*callback)(&line);
        }
        if let Ok(mut buffer) = buffer.lock() {
            buffer.push(&line);
        }
    }
}

fn char_boundary_at_or_before(text: &str, mut offset: usize) -> usize {
    offset = offset.min(text.len());
    while offset > 0 && !text.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

fn char_boundary_at_or_after(text: &str, mut offset: usize) -> usize {
    offset = offset.min(text.len());
    while offset < text.len() && !text.is_char_boundary(offset) {
        offset += 1;
    }
    offset
}

#[cfg(unix)]
struct ProcessGroupDropGuard {
    pgid: Option<i32>,
}

#[cfg(unix)]
impl ProcessGroupDropGuard {
    fn for_child(child: &tokio::process::Child) -> Self {
        Self {
            pgid: child.id().map(|pid| pid as i32),
        }
    }

    /// Disarm the guard so dropping it does not kill the process group. Used
    /// when the still-running child is handed off (auto-background-on-timeout)
    /// — the new owner is now responsible for its lifecycle.
    fn defuse(&mut self) {
        self.pgid = None;
    }
}

#[cfg(unix)]
impl Drop for ProcessGroupDropGuard {
    fn drop(&mut self) {
        if let Some(pgid) = self.pgid {
            kill_group(pgid);
        }
    }
}

#[cfg(not(unix))]
struct ProcessGroupDropGuard;

#[cfg(not(unix))]
impl ProcessGroupDropGuard {
    fn for_child(_child: &tokio::process::Child) -> Self {
        Self
    }

    fn defuse(&mut self) {}
}

#[cfg(unix)]
pub(crate) fn kill_group(pgid: i32) {
    // SAFETY: a negative pid addresses the process group and has no memory
    // safety implications. A stale group simply returns an OS error.
    unsafe {
        libc::kill(-pgid, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
pub(crate) fn kill_group(_pgid: i32) {}

#[cfg(unix)]
fn kill_process_group(child: &tokio::process::Child) {
    if let Some(pid) = child.id() {
        kill_group(pid as i32);
    }
}

#[cfg(not(unix))]
fn kill_process_group(_child: &tokio::process::Child) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn explicit_root_and_structured_failure() {
        let root = std::env::temp_dir().join(format!("hi-process-root-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("marker"), "ok").unwrap();
        let runner = ProcessRunner::new(&root).unwrap();
        let run = runner
            .run_shell(
                "pwd; cat marker; printf problem >&2; exit 7",
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert_eq!(run.status, ToolStatus::Failed);
        assert_eq!(run.outcome.exit_code, Some(7));
        assert!(
            run.outcome
                .stdout_summary
                .contains(root.to_string_lossy().as_ref())
        );
        assert!(run.outcome.stdout_summary.contains("ok"));
        assert!(run.outcome.stderr_summary.contains("problem"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn timeout_is_typed() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let run = runner
            .run_shell("sleep 60", Duration::from_millis(50))
            .await
            .unwrap();
        assert_eq!(run.status, ToolStatus::TimedOut);
        assert_eq!(run.outcome.exit_code, None);
    }

    #[tokio::test]
    async fn direct_program_treats_filename_as_one_argument() {
        let root = std::env::temp_dir().join(format!(
            "hi-process-argv-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let name = "input; touch INJECTED.txt";
        std::fs::write(root.join(name), "safe\n").unwrap();
        let runner = ProcessRunner::new(&root).unwrap();
        let run = runner
            .run_program("cat", [name], Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(run.status, ToolStatus::Succeeded);
        assert_eq!(run.outcome.stdout_summary, "safe");
        assert!(!root.join("INJECTED.txt").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn explicit_environment_is_added_after_sanitization() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let run = runner
            .run_program_with_env(
                "sh",
                ["-c", "printf %s \"$HI_API_KEY\""],
                [("HI_API_KEY", "child-only-key")],
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert_eq!(run.status, ToolStatus::Succeeded);
        assert_eq!(run.outcome.stdout_summary, "child-only-key");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn timeout_kills_process_group_descendants() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let run = runner
            .run_shell(
                "sleep 60 & child=$!; printf '%s\\n' \"$child\"; wait",
                Duration::from_millis(100),
            )
            .await
            .unwrap();
        assert_eq!(run.status, ToolStatus::TimedOut);
        let pid = run.outcome.stdout_summary.trim().parse::<u32>().unwrap();
        let proc_stat = format!("/proc/{pid}/stat");
        for _ in 0..100 {
            let gone_or_zombie = match std::fs::read_to_string(&proc_stat) {
                Ok(stat) => {
                    stat.rsplit_once(") ")
                        .and_then(|(_, rest)| rest.chars().next())
                        == Some('Z')
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
                Err(_) => false,
            };
            if gone_or_zombie {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed-out descendant {pid} remained alive");
    }

    #[tokio::test]
    async fn pagers_are_neutralized_for_child_commands() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let mut sink = |_: &str| {};
        // The child sees PAGER=cat and a blanked AWS_PAGER — paging tools
        // stream instead of blocking.
        let exec = runner
            .run_shell_streaming(
                "printf 'PAGER=%s GIT_PAGER=%s AWS_PAGER=[%s]' \"$PAGER\" \"$GIT_PAGER\" \"$AWS_PAGER\"",
                Duration::from_secs(10),
                &mut sink,
            )
            .await
            .unwrap();
        let out = exec.model_content();
        assert!(out.contains("PAGER=cat"), "PAGER neutralized: {out}");
        assert!(
            out.contains("GIT_PAGER=cat"),
            "GIT_PAGER neutralized: {out}"
        );
        assert!(out.contains("AWS_PAGER=[]"), "AWS_PAGER blanked: {out}");
    }

    #[test]
    fn sensitive_environment_names_are_removed_conservatively() {
        assert!(sensitive_environment_name(OsStr::new("GITHUB_TOKEN")));
        assert!(sensitive_environment_name(OsStr::new(
            "AWS_SECRET_ACCESS_KEY"
        )));
        assert!(sensitive_environment_name(OsStr::new("DATABASE_PASSWORD")));
        assert!(!sensitive_environment_name(OsStr::new("PATH")));
        assert!(!sensitive_environment_name(OsStr::new("RUSTUP_HOME")));
    }

    #[tokio::test]
    async fn adoptable_completes_within_budget_like_normal() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let mut sink = |_: &str| {};
        let outcome = runner
            .run_shell_adoptable("echo adopt-hello", Duration::from_secs(10), &mut sink)
            .await
            .expect("ok");
        match outcome {
            AdoptableOutcome::Completed(exec) => {
                assert_eq!(exec.status, ToolStatus::Succeeded);
                assert!(
                    exec.model_content().contains("adopt-hello"),
                    "got: {}",
                    exec.model_content()
                );
            }
            AdoptableOutcome::StillRunning(_) => panic!("fast command must complete in budget"),
        }
    }

    // Multi-thread flavor so the foreground-budget timer fires independently of
    // the blocking child under CI load (see the bash-tool auto-background test).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn adoptable_hands_off_a_running_child_with_partial_output() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let mut sink = |_: &str| {};
        let outcome = runner
            .run_shell_adoptable(
                "echo seedline; sleep 600",
                Duration::from_millis(400),
                &mut sink,
            )
            .await
            .expect("ok");
        match outcome {
            AdoptableOutcome::StillRunning(mut running) => {
                assert!(
                    running.partial_output.contains("seedline"),
                    "seed carries foreground output: {:?}",
                    running.partial_output
                );
                assert!(running.pgid.is_some(), "pgid captured for tree-kill");
                // The handed-off child is still alive; clean it up (the guard
                // was defused, so nothing killed it for us).
                kill_process_group(&running.child);
                let _ = running.child.kill().await;
            }
            AdoptableOutcome::Completed(_) => panic!("a 600s sleep must outlast a 400ms budget"),
        }
    }
}
