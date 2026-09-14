//! Job-control spawn. One wait API: tokio `child.wait()` — never waitpid(WNOHANG).

use std::io::{self, Write};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::{Child, Command};

use crate::config::SupervisorConfig;
use crate::fsutil;

pub struct TerminalGuard {
    tty_fd: Option<i32>,
    orig: Option<libc::termios>,
    supervisor_pgid: i32,
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

pub fn restore_terminal(terminal: &TerminalGuard) {
    if let (Some(fd), Some(orig)) = (terminal.tty_fd, terminal.orig.as_ref()) {
        unsafe {
            libc::tcsetattr(fd, libc::TCSANOW, orig);
            libc::tcsetpgrp(fd, terminal.supervisor_pgid);
        }
    }
    let mut out = io::stdout();
    let _ = out.write_all(
        b"\x1b[?25h\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1004l\x1b[?2004l\x1b[?1049l",
    );
    let _ = out.flush();
}

pub fn signal_group(pgid: i32, sig: i32) {
    unsafe {
        libc::kill(-pgid, sig);
    }
}

pub async fn abort_harness_child(
    child: &mut Child,
    pgid: i32,
    current_tool_pgid: Option<i32>,
    grace: Duration,
) -> Result<std::process::ExitStatus> {
    signal_group(pgid, libc::SIGTERM);
    if let Some(tool) = current_tool_pgid
        && tool != pgid
    {
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
        libc::signal(libc::SIGTSTP, libc::SIG_IGN);
        libc::signal(libc::SIGTTOU, libc::SIG_IGN);
        libc::signal(libc::SIGTTIN, libc::SIG_IGN);
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
