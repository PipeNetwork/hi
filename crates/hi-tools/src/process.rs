use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::process::Command;

use crate::{ProcessOutcome, ToolStatus, TruncationState};

mod environment;
mod execution;
mod foreground;
mod hermetic;

use environment::{SECRET_ENV_VARS, sensitive_environment_name, workspace_cargo_home};
pub(crate) use execution::kill_group;
#[cfg(test)]
use execution::kill_process_group;
pub use execution::{AdoptableOutcome, RunningChild, preserve_detached_descendants};
use execution::{capture_child, capture_child_adoptable, capture_child_maybe_timeout};
pub use foreground::ForegroundProcessRegistry;

/// The structured result returned by [`ProcessRunner`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessExecution {
    pub status: ToolStatus,
    pub outcome: ProcessOutcome,
    pub truncation: TruncationState,
}

impl ProcessExecution {
    /// Output intended for a human-facing UI. ANSI styling is retained so a
    /// terminal frontend can render compiler diagnostics and diffs with their
    /// original colors.
    pub fn display_content(&self) -> String {
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

    /// Compatibility-friendly model text. Status remains authoritative and
    /// terminal control sequences never enter provider context.
    pub fn model_content(&self) -> String {
        strip_ansi(&self.display_content())
    }

    /// Sanitized process metadata for tool/session records. The display path
    /// keeps the raw summaries separately through [`Self::display_content`].
    pub fn model_outcome(&self) -> ProcessOutcome {
        let mut outcome = self.outcome.clone();
        outcome.stdout_summary = strip_ansi(&outcome.stdout_summary);
        outcome.stderr_summary = strip_ansi(&outcome.stderr_summary);
        outcome
    }
}

/// Strip CSI/OSC ANSI sequences before process output is persisted or sent to
/// a provider. UI callers that want styling should use `display_content`.
pub(crate) fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            // CSI: ESC [ … final byte in @-~
            Some('[') => {
                chars.next();
                for c in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&c) {
                        break;
                    }
                }
            }
            // OSC: ESC ] … BEL (or ESC \\)
            Some(']') => {
                chars.next();
                while let Some(c) = chars.next() {
                    if c == '\u{7}' {
                        break;
                    }
                    if c == '\u{1b}' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            _ => {
                chars.next();
            }
        }
    }
    out
}

/// Hardened process runner bound to one explicit workspace root. Children get
/// closed stdin, bounded output, a sanitized environment, kill-on-drop, and on
/// Unix their own process group for complete cancellation.
#[derive(Clone, Debug)]
pub struct ProcessRunner {
    root: PathBuf,
    foreground: ForegroundProcessRegistry,
    /// Resolved OS sandbox (`HI_SANDBOX`), workspace-confined by default with a
    /// per-workspace Cargo home to protect shared toolchain caches.
    sandbox: crate::sandbox::SandboxProfile,
    cargo_home: Option<PathBuf>,
    private_temp: Option<PathBuf>,
}

impl ProcessRunner {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let policy = crate::sandbox::SandboxPolicy::from_env().map_err(anyhow::Error::msg)?;
        Self::new_with_policy(root, policy)
    }

    /// Construct with a caller-owned policy, avoiding process-environment
    /// coupling between embedded agents and test fixtures.
    pub fn new_with_policy(
        root: impl AsRef<Path>,
        policy: crate::sandbox::SandboxPolicy,
    ) -> Result<Self> {
        Self::new_with_policy_and_config(root, policy, crate::sandbox::SandboxConfig::default())
    }

    /// Construct with caller-owned policy and hermetic profile configuration.
    pub fn new_with_policy_and_config(
        root: impl AsRef<Path>,
        policy: crate::sandbox::SandboxPolicy,
        sandbox_config: crate::sandbox::SandboxConfig,
    ) -> Result<Self> {
        hermetic::build_process_runner(root.as_ref(), policy, sandbox_config)
    }

    /// Whether shell commands from this runner are OS-sandboxed on this platform.
    pub fn sandbox_enforced(&self) -> bool {
        self.sandbox.is_enforced()
    }

    /// Selected sandbox backend state for reports and lifecycle events.
    pub fn sandbox_backend_status(&self) -> crate::sandbox::SandboxBackendStatus {
        self.sandbox.backend_status()
    }

    /// Stable backend label (`seatbelt`, `pipe-wrap`, or `none`).
    pub fn sandbox_backend_name(&self) -> &'static str {
        self.sandbox.backend_name()
    }

    /// Policy requested via `HI_SANDBOX` (may be unenforced on this OS).
    pub fn sandbox_policy(&self) -> crate::sandbox::SandboxPolicy {
        self.sandbox.policy()
    }

    #[cfg(test)]
    pub(crate) fn from_current_dir() -> Result<Self> {
        Self::new(std::env::current_dir().context("determining working directory")?)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn foreground_registry(&self) -> ForegroundProcessRegistry {
        self.foreground.clone()
    }

    pub async fn run_shell(&self, command: &str, timeout: Duration) -> Result<ProcessExecution> {
        self.run_shell_streaming(command, timeout, &mut |_| {})
            .await
    }

    /// Run a shell command with an optional outer deadline. `None` leaves the
    /// command active until it exits or the future is cancelled/dropped; the
    /// process-group guard still reaps the command tree on cancellation.
    pub async fn run_shell_maybe_timeout(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> Result<ProcessExecution> {
        self.run_shell_streaming_maybe_timeout(command, timeout, &mut |_| {})
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
        capture_child(child, timeout, on_line, started, &self.foreground).await
    }

    /// Streaming variant of [`Self::run_shell_maybe_timeout`].
    pub async fn run_shell_streaming_maybe_timeout(
        &self,
        command: &str,
        timeout: Option<Duration>,
        on_line: &mut (dyn FnMut(&str) + Send),
    ) -> Result<ProcessExecution> {
        let started = Instant::now();
        let child = self.spawn_shell(command)?;
        capture_child_maybe_timeout(child, timeout, on_line, started, &self.foreground).await
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
        capture_child_adoptable(child, foreground_budget, on_line, started, &self.foreground).await
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
        self.run_program_maybe_timeout(program, args, Some(timeout))
            .await
    }

    /// Run an executable directly with an optional outer deadline.
    ///
    /// `None` leaves productive work active until the program exits or the
    /// returned future is cancelled. Cancellation still drops the process
    /// group guard and removes the direct child and all of its descendants.
    pub async fn run_program_maybe_timeout<I, S>(
        &self,
        program: impl AsRef<OsStr>,
        args: I,
        timeout: Option<Duration>,
    ) -> Result<ProcessExecution>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let started = Instant::now();
        let (wrapped_program, wrapped_args) =
            self.sandbox
                .wrap_program_in(program.as_ref(), args, &self.root);
        let mut command = Command::new(wrapped_program);
        command.args(wrapped_args);
        self.configure(&mut command);
        if self.sandbox.is_enforced() {
            command.env(crate::sandbox::NESTED_SANDBOX_ENV, "1");
        }
        let child = command.spawn().context("failed to spawn program")?;
        capture_child_maybe_timeout(child, timeout, &mut |_| {}, started, &self.foreground).await
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
        let (wrapped_program, wrapped_args) =
            self.sandbox
                .wrap_program_in(program.as_ref(), args, &self.root);
        let mut command = Command::new(wrapped_program);
        command.args(wrapped_args);
        self.configure(&mut command);
        command.envs(environment);
        if self.sandbox.is_enforced() {
            command.env(crate::sandbox::NESTED_SANDBOX_ENV, "1");
        }
        let child = command.spawn().context("failed to spawn program")?;
        capture_child(child, timeout, &mut |_| {}, started, &self.foreground).await
    }

    /// Run a trusted executable with explicit environment overrides and an
    /// optional outer deadline. `None` leaves the process running until it
    /// exits or the returned future is cancelled/dropped; the process-group
    /// guard still removes the child and its descendants on cancellation.
    pub async fn run_program_with_env_maybe_timeout<I, S, E, K, V>(
        &self,
        program: impl AsRef<OsStr>,
        args: I,
        environment: E,
        timeout: Option<Duration>,
    ) -> Result<ProcessExecution>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
        E: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        let started = Instant::now();
        let (wrapped_program, wrapped_args) =
            self.sandbox
                .wrap_program_in(program.as_ref(), args, &self.root);
        let mut command = Command::new(wrapped_program);
        command.args(wrapped_args);
        self.configure(&mut command);
        command.envs(environment);
        if self.sandbox.is_enforced() {
            command.env(crate::sandbox::NESTED_SANDBOX_ENV, "1");
        }
        let child = command.spawn().context("failed to spawn program")?;
        capture_child_maybe_timeout(child, timeout, &mut |_| {}, started, &self.foreground).await
    }

    /// Spawn a long-lived direct child with piped stdin/stdout. This is the
    /// process boundary used by stdio MCP servers; it retains the same
    /// cwd, environment sanitization, wrapper selection, process group, and
    /// kill-on-drop behavior as ordinary tool execution.
    pub fn spawn_program_piped<I, S, E, K, V>(
        &self,
        program: impl AsRef<OsStr>,
        args: I,
        environment: E,
    ) -> Result<tokio::process::Child>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
        E: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        let (wrapped_program, wrapped_args) =
            self.sandbox
                .wrap_program_in(program.as_ref(), args, &self.root);
        let mut command = Command::new(wrapped_program);
        command.args(wrapped_args);
        self.configure(&mut command);
        command
            .stdin(std::process::Stdio::piped())
            .envs(environment);
        if self.sandbox.is_enforced() {
            command.env(crate::sandbox::NESTED_SANDBOX_ENV, "1");
        }
        command.spawn().context("failed to spawn piped program")
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
            // Cargo suppresses ANSI diagnostics when stdout/stderr are pipes;
            // the TUI captures both streams, so request the same colored
            // compiler output the user would see in an interactive terminal.
            .env("CARGO_TERM_COLOR", "always")
            .env("PYTHONDONTWRITEBYTECODE", "1");
        if let Some(cargo_home) = &self.cargo_home {
            command.env("CARGO_HOME", cargo_home);
        }
        if let Some(private_temp) = &self.private_temp {
            command
                .env("TMPDIR", private_temp)
                .env("TMP", private_temp)
                .env("TEMP", private_temp);
        }
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
        let (program, args) =
            self.sandbox
                .wrap_program_in(OsStr::new("sh"), ["-c", command], &self.root);
        let mut cmd = Command::new(program);
        cmd.args(args);
        self.configure(&mut cmd);
        if self.sandbox.is_enforced() {
            // Mark the confined process tree so a nested hi (e.g. this repo's
            // own test suite under verify) skips the re-wrap macOS would
            // reject — the outer profile already confines every descendant.
            cmd.env(crate::sandbox::NESTED_SANDBOX_ENV, "1");
        }
        cmd.spawn().context("failed to spawn command")
    }
}

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
    async fn timeout_retains_unterminated_output_from_both_streams() {
        let root = tempfile::tempdir().unwrap();
        let runner =
            ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
                .unwrap();
        let mut streamed = String::new();
        let run = runner
            .run_shell_streaming(
                "printf pending-stdout; printf pending-stderr >&2; exec sleep 600",
                Duration::from_millis(400),
                &mut |text| streamed.push_str(text),
            )
            .await
            .unwrap();

        assert_eq!(run.status, ToolStatus::TimedOut);
        assert_eq!(run.outcome.stdout_summary, "pending-stdout");
        assert_eq!(run.outcome.stderr_summary, "pending-stderr");
        assert!(streamed.contains("pending-stdout"));
        assert!(streamed.contains("pending-stderr"));
    }

    #[tokio::test]
    async fn direct_program_deadline_is_optional_and_explicit() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let completed = tokio::time::timeout(
            Duration::from_secs(2),
            runner.run_program_maybe_timeout("sh", ["-c", "sleep 0.03; printf completed"], None),
        )
        .await
        .expect("the unbounded direct program should complete normally")
        .unwrap();
        assert_eq!(completed.status, ToolStatus::Succeeded);
        assert_eq!(completed.outcome.stdout_summary, "completed");

        let timed_out = runner
            .run_program_maybe_timeout("sh", ["-c", "sleep 1"], Some(Duration::from_millis(25)))
            .await
            .unwrap();
        assert_eq!(timed_out.status, ToolStatus::TimedOut);
        assert_eq!(timed_out.outcome.exit_code, None);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_an_unbounded_process_future_kills_its_group() {
        let root = std::env::temp_dir().join(format!(
            "hi-process-unbounded-cancel-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let marker = root.join("leaked");
        let runner = ProcessRunner::new(&root).unwrap();
        let command = format!("sleep 0.15; touch {}", marker.display());

        assert!(
            tokio::time::timeout(
                Duration::from_millis(20),
                runner.run_shell_maybe_timeout(&command, None)
            )
            .await
            .is_err(),
            "the test must cancel the still-running unbounded process future"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(
            !marker.exists(),
            "dropping an unbounded verifier future must kill its process group"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_an_unbounded_direct_program_kills_its_group() {
        let root = std::env::temp_dir().join(format!(
            "hi-process-unbounded-direct-cancel-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let marker = root.join("leaked");
        let runner = ProcessRunner::new(&root).unwrap();
        let command = format!("sleep 0.15; touch {}", marker.display());

        assert!(
            tokio::time::timeout(
                Duration::from_millis(20),
                runner.run_program_maybe_timeout("sh", ["-c", command.as_str()], None)
            )
            .await
            .is_err(),
            "the test must cancel the still-running unbounded direct program"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(
            !marker.exists(),
            "dropping the direct-program future must kill its process group"
        );
        let _ = std::fs::remove_dir_all(root);
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

    #[test]
    fn sandboxed_cargo_uses_workspace_local_home() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let default = workspace_cargo_home(&root, crate::sandbox::SandboxPolicy::Workspace)
            .expect("workspace mode needs an isolated Cargo cache");

        assert_eq!(default, root.join(".hi/state/cargo-home"));

        std::fs::create_dir_all(root.join(".cargo-home")).unwrap();
        assert_eq!(
            workspace_cargo_home(&root, crate::sandbox::SandboxPolicy::Workspace),
            Some(root.join(".cargo-home")),
            "retain compatibility with existing project-local Cargo caches"
        );
        assert_eq!(
            workspace_cargo_home(&root, crate::sandbox::SandboxPolicy::Off),
            None,
            "sandbox-off commands retain the user's normal Cargo environment"
        );
    }

    #[tokio::test]
    async fn process_children_receive_the_isolated_cargo_home() {
        let temp = tempfile::tempdir().unwrap();
        let runner = ProcessRunner::new(temp.path()).unwrap();
        let Some(expected) = runner.cargo_home.clone() else {
            // A parent hi sandbox deliberately resolves nested runners to Off;
            // the ancestor already owns environment confinement in that case.
            return;
        };

        let run = runner
            .run_program(
                "sh",
                ["-c", "printf %s \"$CARGO_HOME\""],
                Duration::from_secs(5),
            )
            .await
            .unwrap();

        assert_eq!(run.status, ToolStatus::Succeeded);
        assert_eq!(run.outcome.stdout_summary, expected.to_string_lossy());
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

    #[tokio::test]
    async fn cargo_diagnostics_keep_color_when_output_is_piped() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let mut sink = |_: &str| {};
        let exec = runner
            .run_shell_streaming(
                "printf %s \"$CARGO_TERM_COLOR\"",
                Duration::from_secs(5),
                &mut sink,
            )
            .await
            .unwrap();
        assert_eq!(exec.model_content(), "always");
    }

    #[tokio::test]
    async fn model_content_strips_ansi_but_display_content_preserves_it() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let exec = runner
            .run_shell("printf '\\033[31mred\\033[0m'", Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(exec.model_content(), "red");
        assert_eq!(exec.display_content(), "\u{1b}[31mred\u{1b}[0m");
        assert_eq!(exec.model_outcome().stdout_summary, "red");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn newline_free_output_stays_bounded() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        // Four megabytes without a newline used to make read_until allocate
        // the whole record before BoundedBuffer could clip it.
        //
        // Generate the record with `yes | tr -d | head -c` rather than
        // `dd bs=1m`: the lowercase `1m` suffix is not portable (some CI
        // runners' `dd` rejects it), and `/dev/zero` may be unreachable under
        // a confined sandbox. Both failure modes write nothing while the
        // pipeline still exits 0, which previously made the run report
        // `Complete` instead of `Truncated` on Linux CI.
        let run = runner
            .run_shell(
                "yes x | tr -d '\\n' | head -c 4194304",
                Duration::from_secs(10),
            )
            .await
            .unwrap();
        assert_eq!(run.status, ToolStatus::Succeeded);
        assert!(
            matches!(run.truncation, TruncationState::Truncated { .. }),
            "expected truncation; got {:?} (stdout {} bytes)",
            run.truncation,
            run.outcome.stdout_summary.len()
        );
        // The human-readable truncation marker can make the returned string a
        // little larger than the nominal character budget; it must still be
        // tiny compared with the four-megabyte source record.
        assert!(run.outcome.stdout_summary.chars().count() < 10_000);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn secrets_split_across_stream_chunks_are_redacted() {
        let runner = ProcessRunner::from_current_dir().unwrap();
        let run = runner
            .run_shell(
                "printf '%*s' 65533 '' | tr ' ' x; printf 'OPENAI_API_KEY=sk-example-secret-value-123456789'",
                Duration::from_secs(10),
            )
            .await
            .unwrap();
        assert!(
            !run.model_content()
                .contains("sk-example-secret-value-123456789")
        );
        assert!(run.model_content().contains("[REDACTED_SECRET]"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn entropy_gated_secret_split_across_stream_chunks_is_redacted() {
        // The entropy gate needs the credential key name AND the value in one
        // redaction window. When a long secret straddles the 64 KiB
        // pseudo-line boundary, the streaming per-chunk redact sees only a
        // fragment, which the value-length floor rejects. The final
        // re-redaction over the reassembled buffer must still catch the whole
        // assignment. The key sits at a line boundary, matching real logs.
        let runner = ProcessRunner::from_current_dir().unwrap();
        // 65490-char line + newline, then `token=` (6) + a 60-char value pushes
        // the value across the 65536 flush boundary mid-token.
        let value = "dGhpc2lzYXJhbmRvbWJhc2U2NHNlY3JldGRHaHBjMmx6WVhKaGJtUnZiVQ==";
        let cmd = format!("printf '%*s\\n' 65490 ''; printf 'token={value}'");
        let run = runner
            .run_shell(&cmd, Duration::from_secs(10))
            .await
            .unwrap();
        let content = run.model_content();
        assert!(
            !content.contains(value),
            "entropy-gated secret leaked across chunk split: ...{}",
            &content[content.len().saturating_sub(90)..]
        );
        assert!(
            content.contains("[REDACTED_SECRET]"),
            "expected redaction marker in reassembled output: ...{}",
            &content[content.len().saturating_sub(90)..]
        );
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
                "printf seedline; printf diagnostic >&2; sleep 600",
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
                assert!(
                    running.partial_output.contains("diagnostic"),
                    "unterminated stderr survives adoption: {:?}",
                    running.partial_output
                );
            }
            AdoptableOutcome::Completed(_) => panic!("a 600s sleep must outlast a 400ms budget"),
        }
    }

    #[tokio::test]
    async fn exit_is_reported_even_when_a_descendant_holds_the_pipes() {
        // `cmd &` inside the shell: the shell exits instantly but the
        // detached sleep inherits stdout, so pipe-EOF never arrives on its
        // own. The reap must not wait for EOF — this used to report a
        // full-budget timeout and discard the real exit status.
        let root = tempfile::tempdir().unwrap();
        let runner =
            ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
                .unwrap();
        let started = Instant::now();
        let exec = runner
            .run_shell("printf done; sleep 30 &", Duration::from_millis(400))
            .await
            .expect("ok");
        assert_eq!(exec.status, ToolStatus::Succeeded);
        assert_eq!(exec.outcome.exit_code, Some(0));
        assert!(
            exec.outcome.stdout_summary.contains("done"),
            "foreground output captured: {:?}",
            exec.outcome.stdout_summary
        );
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "must not burn the budget waiting for the descendant: {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn adoptable_does_not_adopt_exited_children_with_inherited_pipes() {
        let root = tempfile::tempdir().unwrap();
        let runner =
            ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
                .unwrap();
        for exit_code in [0, 7] {
            let command = format!("printf done; sleep 30 & exit {exit_code}");
            let outcome = runner
                .run_shell_adoptable(&command, Duration::from_millis(400), &mut |_| {})
                .await
                .unwrap();
            match outcome {
                AdoptableOutcome::Completed(execution) => {
                    assert_eq!(execution.outcome.exit_code, Some(exit_code));
                    assert_eq!(
                        execution.status,
                        if exit_code == 0 {
                            ToolStatus::Succeeded
                        } else {
                            ToolStatus::Failed
                        }
                    );
                    assert_eq!(execution.outcome.stdout_summary, "done");
                }
                AdoptableOutcome::StillRunning(mut running) => {
                    if let Some(pgid) = running.pgid {
                        kill_group(pgid);
                    }
                    let _ = running.child.kill().await;
                    panic!("an exited child must not be adopted as still running");
                }
            }
        }
    }
}
