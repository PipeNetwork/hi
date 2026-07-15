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
    ///
    /// Invariant: success annotations must never contain "[exit code " —
    /// steering classifies any result containing that marker as a failure.
    pub fn model_content(&self) -> String {
        let stdout_empty = self.outcome.stdout_summary.is_empty();
        let stderr_empty = self.outcome.stderr_summary.is_empty();
        let mut out = self.outcome.stdout_summary.clone();
        if !stderr_empty {
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
                    if stdout_empty && stderr_empty {
                        out.push_str(&format!(
                            "[exit code {code} — no output on stdout or stderr]"
                        ));
                    } else {
                        out.push_str(&format!("[exit code {code}]"));
                    }
                }
            }
            ToolStatus::TimedOut => {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("[timed out — process killed]");
            }
            // Empty and stderr-only results are where models form false
            // premises ("did it work?" / "stderr means it failed") — state
            // the verdict explicitly instead of leaving it implied.
            ToolStatus::Succeeded => {
                if stdout_empty && stderr_empty {
                    out.push_str("[no output — command succeeded (exit 0)]");
                } else if stdout_empty {
                    if !out.ends_with('\n') {
                        out.push('\n');
                    }
                    out.push_str("[command succeeded (exit 0) — output above is stderr]");
                }
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
        Ok(Self { root })
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
        #[cfg(unix)]
        command.process_group(0);
    }

    /// Spawn a child for the background registry. The registry is responsible
    /// for draining and reaping it.
    pub(crate) fn spawn_shell(&self, command: &str) -> Result<tokio::process::Child> {
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command);
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
    let _group_guard = ProcessGroupDropGuard::for_child(&child);
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let callback: &Mutex<&mut (dyn FnMut(&str) + Send)> = &Mutex::new(on_line);
    let stdout_buf = Mutex::new(BoundedBuffer::default());
    let stderr_buf = Mutex::new(BoundedBuffer::default());

    let combined = async {
        tokio::join!(
            read_stream(stdout, callback, &stdout_buf),
            read_stream(stderr, callback, &stderr_buf),
        );
        child.wait().await
    };

    let (status, exit_code) = match tokio::time::timeout(timeout, combined).await {
        Ok(Ok(exit)) if exit.success() => (ToolStatus::Succeeded, exit.code()),
        Ok(Ok(exit)) => (ToolStatus::Failed, exit.code()),
        Ok(Err(err)) => return Err(err).context("waiting for command"),
        Err(_) => {
            kill_process_group(&child);
            let _ = child.kill().await;
            (ToolStatus::TimedOut, None)
        }
    };

    let stdout = stdout_buf.into_inner().unwrap_or_default();
    let stderr = stderr_buf.into_inner().unwrap_or_default();
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

    Ok(ProcessExecution {
        status,
        outcome: ProcessOutcome {
            exit_code,
            stdout_summary,
            stderr_summary,
            duration_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        },
        truncation,
    })
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
    pipe: Option<R>,
    on_line: &Mutex<&mut (dyn FnMut(&str) + Send)>,
    buffer: &Mutex<BoundedBuffer>,
) {
    let Some(pipe) = pipe else { return };
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
    async fn empty_success_is_annotated_with_exit_zero() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let run = runner
            .run_shell("true", Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(run.status, ToolStatus::Succeeded);
        assert_eq!(
            run.model_content(),
            "[no output — command succeeded (exit 0)]"
        );
    }

    #[tokio::test]
    async fn stderr_only_success_is_annotated() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let run = runner
            .run_shell("printf warning >&2", Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(run.status, ToolStatus::Succeeded);
        let content = run.model_content();
        assert!(content.contains("warning"), "got: {content:?}");
        assert!(
            content.ends_with("[command succeeded (exit 0) — output above is stderr]"),
            "got: {content:?}"
        );
        assert!(
            !content.contains("[exit code "),
            "success must never carry the failure marker: {content:?}"
        );
    }

    #[tokio::test]
    async fn empty_failure_annotates_no_output() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let run = runner
            .run_shell("exit 3", Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(run.status, ToolStatus::Failed);
        assert_eq!(
            run.model_content(),
            "[exit code 3 — no output on stdout or stderr]"
        );
        // Failures with output keep the bare marker.
        let noisy = runner
            .run_shell("printf problem >&2; exit 7", Duration::from_secs(5))
            .await
            .unwrap();
        assert!(noisy.model_content().ends_with("[exit code 7]"));
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
}
