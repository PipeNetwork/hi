use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::process::Command;

use crate::{ProcessOutcome, ToolStatus, TruncationState};

mod environment;
mod execution;
mod foreground;
mod hermetic;
mod program;

use environment::{SECRET_ENV_VARS, sensitive_environment_name, workspace_cargo_home};
#[cfg(test)]
use execution::kill_process_group;
pub use execution::{AdoptableOutcome, RunningChild, preserve_detached_descendants};
pub(crate) use execution::{PIPE_DRAIN_GRACE, detached_descendants_preserved, kill_group};
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

#[derive(Default)]
struct EvidenceReducerState {
    config: crate::EvidenceReducerConfig,
    hook: Option<crate::EvidenceReducerHook>,
}

fn reduce_condensed_stream(
    config: &crate::EvidenceReducerConfig,
    hook: Option<&crate::EvidenceReducerHook>,
    condensed: &str,
    is_error: bool,
) -> String {
    if condensed.is_empty() {
        return String::new();
    }
    let receipt = match hook {
        Some(hook) => hook(condensed, is_error),
        None => return condensed.to_string(),
    };
    crate::evidence_reducer::after_condense(condensed, is_error, config, receipt)
}

/// Hardened process runner bound to one explicit workspace root. Children get
/// closed stdin, bounded output, a sanitized environment, kill-on-drop, and on
/// Unix their own process group for complete cancellation.
#[derive(Clone)]
pub struct ProcessRunner {
    root: PathBuf,
    foreground: ForegroundProcessRegistry,
    /// Resolved OS sandbox (`HI_SANDBOX`), workspace-confined by default with a
    /// per-workspace Cargo home to protect shared toolchain caches.
    sandbox: crate::sandbox::SandboxProfile,
    cargo_home: Option<PathBuf>,
    private_temp: Option<PathBuf>,
    evidence_reducer: Arc<Mutex<EvidenceReducerState>>,
}

impl std::fmt::Debug for ProcessRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessRunner")
            .field("root", &self.root)
            .field("foreground", &self.foreground)
            .field("sandbox", &self.sandbox)
            .field("cargo_home", &self.cargo_home)
            .field("private_temp", &self.private_temp)
            .finish_non_exhaustive()
    }
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

    /// Install quote-checked reduction after diagnostic condense. Clones share
    /// this hook. `None` keeps condensed output (fail-open, no nested model).
    pub fn set_evidence_reducer(
        &self,
        config: crate::EvidenceReducerConfig,
        hook: Option<crate::EvidenceReducerHook>,
    ) {
        let mut state = self
            .evidence_reducer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.config = config;
        state.hook = hook;
    }

    pub(super) fn apply_diagnostic_evidence_reducer(
        &self,
        mut execution: ProcessExecution,
    ) -> ProcessExecution {
        let (config, hook) = {
            let state = self
                .evidence_reducer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !state.config.enabled {
                return execution;
            }
            (state.config.clone(), state.hook.clone())
        };
        let is_error = !matches!(execution.status, ToolStatus::Succeeded);
        execution.outcome.stdout_summary = reduce_condensed_stream(
            &config,
            hook.as_ref(),
            &execution.outcome.stdout_summary,
            is_error,
        );
        execution.outcome.stderr_summary = reduce_condensed_stream(
            &config,
            hook.as_ref(),
            &execution.outcome.stderr_summary,
            is_error,
        );
        execution
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
        let execution = capture_child(child, timeout, on_line, started, &self.foreground).await?;
        Ok(self.apply_diagnostic_evidence_reducer(execution))
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
        let execution =
            capture_child_maybe_timeout(child, timeout, on_line, started, &self.foreground).await?;
        Ok(self.apply_diagnostic_evidence_reducer(execution))
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
        match capture_child_adoptable(child, foreground_budget, on_line, started, &self.foreground)
            .await?
        {
            AdoptableOutcome::Completed(execution) => Ok(AdoptableOutcome::Completed(
                self.apply_diagnostic_evidence_reducer(execution),
            )),
            other => Ok(other),
        }
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
        let execution =
            capture_child(child, timeout, &mut |_| {}, started, &self.foreground).await?;
        Ok(self.apply_diagnostic_evidence_reducer(execution))
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
        let execution =
            capture_child_maybe_timeout(child, timeout, &mut |_| {}, started, &self.foreground)
                .await?;
        Ok(self.apply_diagnostic_evidence_reducer(execution))
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

    /// Spawn a shell child with the ordinary foreground drop policy.
    pub(crate) fn spawn_shell(&self, command: &str) -> Result<tokio::process::Child> {
        self.spawn_shell_with_drop_policy(command, true)
    }

    /// Spawn a child whose lifetime is owned by the background registry.
    ///
    /// Tokio's child handle must not kill the process merely because its
    /// driver task is dropped after a successful `--keep-background` release.
    /// The registry retains the fail-safe process-group kill until that
    /// explicit ownership transfer, and all ordinary stop paths still kill and
    /// reap the child themselves.
    pub(crate) fn spawn_background_shell(&self, command: &str) -> Result<tokio::process::Child> {
        self.spawn_shell_with_drop_policy(command, false)
    }

    fn spawn_shell_with_drop_policy(
        &self,
        command: &str,
        kill_on_drop: bool,
    ) -> Result<tokio::process::Child> {
        let (program, args) =
            self.sandbox
                .wrap_program_in(OsStr::new("sh"), ["-c", command], &self.root);
        let mut cmd = Command::new(program);
        cmd.args(args);
        self.configure(&mut cmd);
        cmd.kill_on_drop(kill_on_drop);
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
mod tests;
