//! Job-control spawn. One wait API: tokio `child.wait()` — never waitpid(WNOHANG).

use std::cell::Cell;
use std::io::{self, IsTerminal, Write};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::process::{Child, Command};

use crate::config::SupervisorConfig;
use crate::fsutil;

pub struct TerminalGuard {
    tty_fd: Option<i32>,
    orig: Option<libc::termios>,
    supervisor_pgid: i32,
    restored: Cell<bool>,
}

pub struct Spawned {
    pub child: Child,
    pub pid: u32,
    pub pgid: i32,
    pub terminal: TerminalGuard,
}

pub fn snapshot_terminal() -> TerminalGuard {
    let supervisor_pgid = unsafe { libc::getpgrp() };
    let tty_fd = stdin_tty_fd();
    let orig = tty_fd.and_then(|fd| {
        let mut termios = std::mem::MaybeUninit::<libc::termios>::uninit();
        let rc = unsafe { libc::tcgetattr(fd, termios.as_mut_ptr()) };
        if rc == 0 {
            Some(unsafe { termios.assume_init() })
        } else {
            None
        }
    });
    ignore_job_signals();
    TerminalGuard {
        tty_fd,
        orig,
        supervisor_pgid,
        restored: Cell::new(false),
    }
}

impl TerminalGuard {
    pub fn restore(&self) {
        if self.restored.replace(true) {
            return;
        }
        // POSIX: `tcsetpgrp` / tty writes from a background process group raise
        // SIGTTOU and *stop* the process (`zsh: suspended (tty output)`). After
        // the child exits we are that background group until we take the tty
        // back, so keep TTOU ignored through handoff + the leave-alt-screen
        // write, then restore SIG_DFL.
        debug_assert!(
            ttou_is_ignored(),
            "SIGTTOU must stay ignored until after tty handoff"
        );
        if let (Some(fd), Some(orig)) = (self.tty_fd, self.orig.as_ref()) {
            unsafe {
                // Drop leftover TUI keystrokes so they cannot answer a later [y/N].
                libc::tcsetattr(fd, libc::TCSAFLUSH, orig);
                libc::tcsetpgrp(fd, self.supervisor_pgid);
            }
        }
        if io::stdout().is_terminal() {
            let mut out = io::stdout();
            let _ = out.write_all(
                b"\x1b[?25h\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1004l\x1b[?2004l\x1b[?1049l",
            );
            let _ = out.flush();
        }
        restore_job_signals();
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

pub fn spawn_child(
    cfg: &SupervisorConfig,
    env_pairs: &[(String, String)],
    terminal: TerminalGuard,
) -> Result<Spawned> {
    let mut command = Command::new(&cfg.child_program);
    command.args(&cfg.child_args);
    command.current_dir(&cfg.workspace);
    command.kill_on_drop(false);
    if cfg.inherit_stdio {
        command
            .stdin(std::process::Stdio::inherit())
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit());
    }
    for (key, value) in env_pairs {
        command.env(key, value);
    }
    // SAFETY: pre_exec runs in the child between fork and exec, still
    // single-threaded, so setpgid cannot race other threads.
    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) != 0 {
                return Err(io::Error::last_os_error());
            }
            // Parent ignores job-control signals; the child must not inherit that.
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGTSTP, libc::SIG_DFL);
            libc::signal(libc::SIGTTOU, libc::SIG_DFL);
            libc::signal(libc::SIGTTIN, libc::SIG_DFL);
            Ok(())
        });
    }
    let child = command
        .spawn()
        .with_context(|| format!("spawning supervised child {}", cfg.child_program.display()))?;
    let pid = child.id().context("child pid missing after spawn")?;
    let pgid = pid as i32;
    // Parent setpgid is the fallback if the child has not yet run pre_exec.
    unsafe {
        libc::setpgid(pgid, pgid);
    }
    if let Some(fd) = terminal.tty_fd {
        unsafe {
            libc::tcsetpgrp(fd, pgid);
        }
    }
    Ok(Spawned {
        child,
        pid,
        pgid,
        terminal,
    })
}

pub fn signal_group(pgid: i32, sig: i32) {
    unsafe {
        libc::kill(-pgid, sig);
    }
}

static INSTALL_LOCK: Mutex<()> = Mutex::new(());
static INSTALL_PGID: AtomicI32 = AtomicI32::new(0);
static INSTALL_INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_install_signal(_sig: libc::c_int) {
    abort_install_group();
}

fn abort_install_group() {
    let pgid = INSTALL_PGID.load(Ordering::SeqCst);
    if pgid > 1 {
        unsafe {
            libc::kill(-pgid, libc::SIGTERM);
        }
    }
    INSTALL_INTERRUPTED.store(true, Ordering::SeqCst);
}

struct InstallSignalGuard {
    prev_int: libc::sighandler_t,
    prev_term: libc::sighandler_t,
}

impl Drop for InstallSignalGuard {
    fn drop(&mut self) {
        INSTALL_PGID.store(0, Ordering::SeqCst);
        unsafe {
            libc::signal(libc::SIGINT, self.prev_int);
            libc::signal(libc::SIGTERM, self.prev_term);
        }
    }
}

/// Wait for a process-group child. SIGINT/SIGTERM kill that group instead of leaking it.
pub fn wait_install_child(
    child: &mut std::process::Child,
    pgid: i32,
    timeout: Duration,
    term_grace: Duration,
) -> io::Result<std::process::ExitStatus> {
    let _lock = INSTALL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    INSTALL_INTERRUPTED.store(false, Ordering::SeqCst);
    let prev_int = unsafe {
        libc::signal(
            libc::SIGINT,
            on_install_signal as *const () as libc::sighandler_t,
        )
    };
    let prev_term = unsafe {
        libc::signal(
            libc::SIGTERM,
            on_install_signal as *const () as libc::sighandler_t,
        )
    };
    let _guard = InstallSignalGuard {
        prev_int,
        prev_term,
    };
    INSTALL_PGID.store(pgid, Ordering::SeqCst);
    let deadline = Instant::now() + timeout;
    loop {
        let status = child.try_wait()?;
        if INSTALL_INTERRUPTED.load(Ordering::SeqCst) {
            if status.is_none() {
                terminate_group(child, pgid, term_grace);
            }
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "cargo install interrupted",
            ));
        }
        if let Some(status) = status {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            terminate_group(child, pgid, term_grace);
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "cargo install timed out",
            ));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn terminate_group(child: &mut std::process::Child, pgid: i32, grace: Duration) {
    signal_group(pgid, libc::SIGTERM);
    let kill_at = Instant::now() + grace;
    loop {
        if child.try_wait().ok().flatten().is_some() {
            return;
        }
        if Instant::now() >= kill_at {
            signal_group(pgid, libc::SIGKILL);
            let _ = child.wait();
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(test)]
pub fn interrupt_install() {
    abort_install_group();
}

#[cfg(test)]
pub fn install_pgid() -> i32 {
    INSTALL_PGID.load(Ordering::SeqCst)
}

pub async fn abort_harness_child(
    child: &mut Child,
    pgid: i32,
    current_tool_pgid: Option<i32>,
    grace: Duration,
) -> Result<std::process::ExitStatus> {
    // A job-stopped child (Ctrl-Z / SIGSTOP) will not observe SIGTERM until
    // it is continued. Continue first so a user stop actually ends the session.
    signal_group(pgid, libc::SIGCONT);
    signal_group(pgid, libc::SIGTERM);
    if let Some(tool) = current_tool_pgid
        && tool != pgid
    {
        signal_group(tool, libc::SIGCONT);
        signal_group(tool, libc::SIGTERM);
    }
    tokio::select! {
        status = child.wait() => Ok(status?),
        _ = tokio::time::sleep(grace) => {
            signal_group(pgid, libc::SIGKILL);
            if let Some(tool) = current_tool_pgid
                && tool != pgid
            {
                signal_group(tool, libc::SIGKILL);
            }
            Ok(child.wait().await?)
        }
    }
}

pub fn pid_alive(pid: u32) -> bool {
    let rc = unsafe { libc::kill(pid as i32, 0) };
    if rc == 0 {
        return true;
    }
    io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

fn stdin_tty_fd() -> Option<i32> {
    let fd = io::stdin().as_raw_fd();
    if unsafe { libc::isatty(fd) } == 1 {
        Some(fd)
    } else {
        None
    }
}

fn ignore_job_signals() {
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_IGN);
        libc::signal(libc::SIGTTOU, libc::SIG_IGN);
        libc::signal(libc::SIGTTIN, libc::SIG_IGN);
        // SIGTSTP: owned by the tokio handler in `supervise`. Do not SIG_IGN.
    }
}

/// Undo [`ignore_job_signals`] for Ctrl-C / TTOU / TTIN. Do not touch SIGTSTP:
/// `supervise` installs a tokio handler once (`get_or_init`), and
/// `libc::signal(SIGTSTP, …)` would replace it. SIG_DFL *stops* this process
/// (`T`) so repair never runs; SIG_IGN leaves later generations unable to
/// observe Ctrl-Z. The tokio handler both swallows the default stop and
/// delivers user-stop while `hi` is running.
fn restore_job_signals() {
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_DFL);
        libc::signal(libc::SIGTTOU, libc::SIG_DFL);
        libc::signal(libc::SIGTTIN, libc::SIG_DFL);
    }
}

fn ttou_is_ignored() -> bool {
    unsafe {
        let previous = libc::signal(libc::SIGTTOU, libc::SIG_IGN);
        libc::signal(libc::SIGTTOU, previous);
        previous == libc::SIG_IGN
    }
}

/// Job-control ignore is for the live child. The apply prompt and cargo install must be interruptible.
pub fn prepare_interactive_prompt() {
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_DFL);
    }
    let Some(fd) = stdin_tty_fd() else {
        return;
    };
    let mut termios = std::mem::MaybeUninit::<libc::termios>::uninit();
    let rc = unsafe { libc::tcgetattr(fd, termios.as_mut_ptr()) };
    if rc == 0 {
        let termios = unsafe { termios.assume_init() };
        unsafe {
            libc::tcsetattr(fd, libc::TCSAFLUSH, &termios);
        }
    }
}

pub fn write_supervisor_log(runtime: &Path, line: &str) {
    let path = runtime.join("supervisor.log");
    let ts = hi_liveness::unix_ms();
    let body = format!("{ts} {line}\n");
    if path.exists() {
        if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(&path) {
            let _ = file.write_all(body.as_bytes());
            let _ = fsutil::chmod_0600(&path);
        }
        return;
    }
    let _ = fsutil::write_0600(&path, body.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SignalRestore {
        int: libc::sighandler_t,
        ttou: libc::sighandler_t,
        ttin: libc::sighandler_t,
    }

    impl Drop for SignalRestore {
        fn drop(&mut self) {
            unsafe {
                libc::signal(libc::SIGINT, self.int);
                libc::signal(libc::SIGTTOU, self.ttou);
                libc::signal(libc::SIGTTIN, self.ttin);
            }
        }
    }

    fn capture_job_signals() -> SignalRestore {
        unsafe {
            SignalRestore {
                int: libc::signal(libc::SIGINT, libc::SIG_DFL),
                ttou: libc::signal(libc::SIGTTOU, libc::SIG_DFL),
                ttin: libc::signal(libc::SIGTTIN, libc::SIG_DFL),
            }
        }
    }

    #[test]
    fn snapshot_ignores_ttou_and_restore_rearms_it_after_tty_handoff() {
        let _signals = capture_job_signals();
        let guard = snapshot_terminal();
        let while_live = unsafe { libc::signal(libc::SIGTTOU, libc::SIG_IGN) };
        assert_eq!(
            while_live,
            libc::SIG_IGN,
            "supervisor must ignore SIGTTOU while the child owns the tty"
        );
        guard.restore();
        let after = unsafe { libc::signal(libc::SIGTTOU, libc::SIG_DFL) };
        assert_eq!(
            after,
            libc::SIG_DFL,
            "SIGTTOU is rearmed only after tcsetpgrp and the leave-screen write"
        );
    }
}
