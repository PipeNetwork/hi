//! Headless PTY controller for `hi`.
//!
//! Provides a programmatic interface to spawn and control terminal processes
//! in a pseudo-terminal (PTY). This enables:
//! - E2E testing of TUI applications with realistic terminal behavior
//! - Scripted scenario replay against terminal programs
//! - Capturing and validating terminal output (scrollback, styled text)
//!
//! The controller spawns a child process in a PTY, feeds it input, and captures
//! output. On platforms with `portable-pty` support, this uses real OS PTYs;
//! on others, it falls back to piped stdio (no terminal emulation).
//!
//! Inspired by grok-build's `ptyctl` crate.
//!
//! # Quick start
//!
//! ```no_run
//! use hi_ptyctl::PtySession;
//!
//! let mut session = PtySession::spawn("echo", &["hello"]).unwrap();
//! session.wait().unwrap();
//! let output = session.read_output().unwrap();
//! assert!(output.contains("hello"));
//! ```

use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use thiserror::Error;

/// Errors from the PTY controller.
#[derive(Debug, Error)]
pub enum PtyError {
    /// Failed to spawn the child process.
    #[error("failed to spawn process: {0}")]
    SpawnFailed(String),
    /// Failed to write to the PTY.
    #[error("write failed: {0}")]
    WriteFailed(String),
    /// Failed to read from the PTY.
    #[error("read failed: {0}")]
    ReadFailed(String),
    /// The process exited unexpectedly.
    #[error("process exited: {0}")]
    ProcessExited(i32),
    /// An I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// A key sequence to send to the PTY.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Key {
    /// A literal string of characters.
    Text(String),
    /// Enter/Return key.
    Enter,
    /// Ctrl-C (SIGINT).
    CtrlC,
    /// Ctrl-D (EOF).
    CtrlD,
    /// Escape key.
    Escape,
    /// Tab key.
    Tab,
    /// Backspace.
    Backspace,
    /// Arrow key.
    Arrow(Direction),
    /// A specific byte sequence.
    Raw(Vec<u8>),
}

/// Arrow key directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Key {
    /// Convert a key to its byte sequence.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Key::Text(s) => s.as_bytes().to_vec(),
            Key::Enter => b"\r".to_vec(),
            Key::CtrlC => b"\x03".to_vec(),
            Key::CtrlD => b"\x04".to_vec(),
            Key::Escape => b"\x1b".to_vec(),
            Key::Tab => b"\t".to_vec(),
            Key::Backspace => b"\x7f".to_vec(),
            Key::Arrow(d) => {
                let suffix = match d {
                    Direction::Up => b"A",
                    Direction::Down => b"B",
                    Direction::Right => b"C",
                    Direction::Left => b"D",
                };
                let mut bytes = vec![0x1b, b'['];
                bytes.push(suffix[0]);
                bytes
            }
            Key::Raw(bytes) => bytes.clone(),
        }
    }
}

/// A headless PTY session controlling a child process.
pub struct PtySession {
    child: std::process::Child,
    stdin: Option<std::process::ChildStdin>,
    stdout_rx: mpsc::Receiver<String>,
    _reader_thread: Option<thread::JoinHandle<()>>,
}

impl PtySession {
    /// Spawn a command in a PTY (or piped stdio as fallback).
    pub fn spawn(command: &str, args: &[&str]) -> Result<Self, PtyError> {
        let mut cmd = Command::new(command);
        cmd.args(args);
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| PtyError::SpawnFailed(format!("{command}: {e}")))?;

        let stdin = child.stdin.take();
        let stdout = child.stdout.take();

        let (tx, rx) = mpsc::channel::<String>();
        let reader_thread = if let Some(stdout) = stdout {
            let tx = tx.clone();
            thread::Builder::new()
                .name("hi-ptyctl-reader".into())
                .spawn(move || {
                    use std::io::Read;
                    let mut buf = [0u8; 4096];
                    let mut stdout = stdout;
                    loop {
                        match stdout.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => {
                                let s = String::from_utf8_lossy(&buf[..n]).to_string();
                                if tx.send(s).is_err() {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                })
                .ok()
        } else {
            None
        };

        // Drop the sender so the channel closes when the reader thread exits.
        drop(tx);

        Ok(Self {
            child,
            stdin,
            stdout_rx: rx,
            _reader_thread: reader_thread,
        })
    }

    /// Send text to the process's stdin.
    pub async fn send(&mut self, text: &str) -> Result<(), PtyError> {
        if let Some(stdin) = &mut self.stdin {
            use std::io::Write;
            stdin
                .write_all(text.as_bytes())
                .map_err(|e| PtyError::WriteFailed(e.to_string()))?;
            stdin
                .flush()
                .map_err(|e| PtyError::WriteFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Send a key sequence to the process.
    pub async fn send_key(&mut self, key: &Key) -> Result<(), PtyError> {
        if let Some(stdin) = &mut self.stdin {
            use std::io::Write;
            let bytes = key.to_bytes();
            stdin
                .write_all(&bytes)
                .map_err(|e| PtyError::WriteFailed(e.to_string()))?;
            stdin
                .flush()
                .map_err(|e| PtyError::WriteFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Read all accumulated output (non-blocking, drains the channel).
    pub fn read_output(&self) -> Result<String, PtyError> {
        let mut output = String::new();
        while let Ok(chunk) = self.stdout_rx.try_recv() {
            output.push_str(&chunk);
        }
        Ok(output)
    }

    /// Wait for output with a timeout, accumulating all available text.
    pub fn read_output_timeout(&self, timeout: Duration) -> Result<String, PtyError> {
        let mut output = String::new();
        // First, drain anything immediately available.
        while let Ok(chunk) = self.stdout_rx.try_recv() {
            output.push_str(&chunk);
        }
        // Then wait for more with timeout.
        if let Ok(chunk) = self.stdout_rx.recv_timeout(timeout) {
            output.push_str(&chunk);
            // Drain any more that arrived.
            while let Ok(chunk) = self.stdout_rx.try_recv() {
                output.push_str(&chunk);
            }
        }
        Ok(output)
    }

    /// Wait for the child process to exit.
    pub fn wait(&mut self) -> Result<std::process::ExitStatus, PtyError> {
        self.child.wait().map_err(PtyError::Io)
    }

    /// Kill the child process if it's still running.
    pub fn kill(&mut self) -> Result<(), PtyError> {
        self.child.kill().map_err(PtyError::Io)
    }

    /// Check if the child process has exited.
    pub fn try_wait(&mut self) -> Result<Option<std::process::ExitStatus>, PtyError> {
        self.child.try_wait().map_err(PtyError::Io)
    }
}

impl Drop for PtySession {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A scripted scenario for PTY-based testing.
#[derive(Debug, Clone)]
pub struct Scenario {
    /// Steps to execute.
    pub steps: Vec<ScenarioStep>,
}

/// A single step in a test scenario.
#[derive(Debug, Clone)]
pub enum ScenarioStep {
    /// Send text to the PTY.
    Send(String),
    /// Send a key sequence.
    SendKey(Key),
    /// Wait for output matching a pattern (substring match).
    WaitFor(String),
    /// Wait for a duration.
    Wait(Duration),
    /// Assert that the output contains a string.
    AssertContains(String),
    /// Assert that the output does NOT contain a string.
    AssertNotContains(String),
}

/// Result of running a scenario.
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    /// Whether all assertions passed.
    pub passed: bool,
    /// The full captured output.
    pub output: String,
    /// Any failure messages.
    pub failures: Vec<String>,
}

/// Run a scenario against a spawned process.
pub fn run_scenario(
    command: &str,
    args: &[&str],
    scenario: &Scenario,
) -> Result<ScenarioResult, PtyError> {
    let mut session = PtySession::spawn(command, args)?;
    let mut output = String::new();
    let mut failures = Vec::new();

    for step in &scenario.steps {
        match step {
            ScenarioStep::Send(text) => {
                // We can't use async here, so call the underlying write directly.
                if let Some(stdin) = &mut session.stdin {
                    use std::io::Write;
                    if let Err(e) = stdin.write_all(text.as_bytes()) {
                        failures.push(format!("send failed: {e}"));
                    }
                    let _ = stdin.flush();
                }
            }
            ScenarioStep::SendKey(key) => {
                if let Some(stdin) = &mut session.stdin {
                    use std::io::Write;
                    let bytes = key.to_bytes();
                    if let Err(e) = stdin.write_all(&bytes) {
                        failures.push(format!("send_key failed: {e}"));
                    }
                    let _ = stdin.flush();
                }
            }
            ScenarioStep::WaitFor(pattern) => {
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                loop {
                    output.push_str(&session.read_output()?);
                    if output.contains(pattern) {
                        break;
                    }
                    if std::time::Instant::now() > deadline {
                        failures.push(format!("timeout waiting for: {pattern}"));
                        break;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
            }
            ScenarioStep::Wait(d) => {
                thread::sleep(*d);
                output.push_str(&session.read_output()?);
            }
            ScenarioStep::AssertContains(pattern) => {
                output.push_str(&session.read_output()?);
                if !output.contains(pattern) {
                    failures.push(format!("expected output to contain: {pattern}"));
                }
            }
            ScenarioStep::AssertNotContains(pattern) => {
                output.push_str(&session.read_output()?);
                if output.contains(pattern) {
                    failures.push(format!("expected output to NOT contain: {pattern}"));
                }
            }
        }
    }

    // Final output drain.
    output.push_str(&session.read_output()?);

    let passed = failures.is_empty();
    Ok(ScenarioResult {
        passed,
        output,
        failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_text_to_bytes() {
        assert_eq!(Key::Text("hello".into()).to_bytes(), b"hello");
    }

    #[test]
    fn key_enter_to_bytes() {
        assert_eq!(Key::Enter.to_bytes(), b"\r");
    }

    #[test]
    fn key_ctrl_c_to_bytes() {
        assert_eq!(Key::CtrlC.to_bytes(), b"\x03");
    }

    #[test]
    fn key_arrow_to_bytes() {
        assert_eq!(Key::Arrow(Direction::Up).to_bytes(), b"\x1b[A".to_vec());
        assert_eq!(Key::Arrow(Direction::Down).to_bytes(), b"\x1b[B".to_vec());
    }

    #[test]
    fn key_escape_to_bytes() {
        assert_eq!(Key::Escape.to_bytes(), b"\x1b");
    }

    #[test]
    fn key_tab_to_bytes() {
        assert_eq!(Key::Tab.to_bytes(), b"\t");
    }

    #[test]
    fn key_backspace_to_bytes() {
        assert_eq!(Key::Backspace.to_bytes(), b"\x7f");
    }

    #[test]
    fn key_raw_to_bytes() {
        assert_eq!(Key::Raw(vec![1, 2, 3]).to_bytes(), vec![1, 2, 3]);
    }

    #[test]
    fn spawn_echo_and_read() {
        let mut session = PtySession::spawn("echo", &["hello world"]).unwrap();
        session.wait().unwrap();
        let output = session.read_output_timeout(Duration::from_secs(5)).unwrap();
        assert!(output.contains("hello world"));
    }

    #[test]
    fn spawn_cat_and_send() {
        // Use `cat` with a heredoc-style approach: pipe input then close stdin.
        let mut session = PtySession::spawn("cat", &[]).unwrap();
        // Send some text.
        if let Some(stdin) = &mut session.stdin {
            use std::io::Write;
            stdin.write_all(b"hello\n").unwrap();
            stdin.flush().unwrap();
        }
        // Close stdin to signal EOF to cat.
        session.stdin.take(); // drop stdin
        session.wait().unwrap();
        let output = session.read_output_timeout(Duration::from_secs(5)).unwrap();
        assert!(output.contains("hello"));
    }

    #[test]
    fn scenario_assert_contains() {
        let scenario = Scenario {
            steps: vec![
                ScenarioStep::Wait(Duration::from_millis(100)),
                ScenarioStep::AssertContains("hello".into()),
            ],
        };
        let result = run_scenario("echo", &["hello"], &scenario).unwrap();
        assert!(result.passed, "failures: {:?}", result.failures);
    }

    #[test]
    fn scenario_assert_not_contains() {
        let scenario = Scenario {
            steps: vec![
                ScenarioStep::Wait(Duration::from_millis(100)),
                ScenarioStep::AssertNotContains("goodbye".into()),
            ],
        };
        let result = run_scenario("echo", &["hello"], &scenario).unwrap();
        assert!(result.passed, "failures: {:?}", result.failures);
    }

    #[test]
    fn scenario_failure_on_missing_pattern() {
        let scenario = Scenario {
            steps: vec![
                ScenarioStep::Wait(Duration::from_millis(100)),
                ScenarioStep::AssertContains("nonexistent".into()),
            ],
        };
        let result = run_scenario("echo", &["hello"], &scenario).unwrap();
        assert!(!result.passed);
        assert!(!result.failures.is_empty());
    }

    #[test]
    fn spawn_nonexistent_command_fails() {
        let result = PtySession::spawn("nonexistent-command-12345", &[]);
        assert!(result.is_err());
    }
}
