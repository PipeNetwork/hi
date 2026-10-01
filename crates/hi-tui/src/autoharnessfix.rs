//! `/autoharnessfix` TUI dispatch. `exec(2)` skips Drop, so the tty is restored
//! before replacing this process.

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::Result;
use hi_harness::Harness;
use hi_sentinel::{SlashOutcome, dispatch_slash, exec_repair, exec_supervisor, restart_line};
use ratatui::text::Line;

use crate::App;
use crate::event::Restore;
use crate::render::dim;

pub(crate) struct AutoharnessfixOpts {
    pub session_path: Option<PathBuf>,
    pub no_save: bool,
    pub sentinel_blocked: Option<String>,
}

/// Ordered tty restore used by `/autoharnessfix on` and by the unit test.
pub trait TtyRestore {
    fn disable_raw_mode(&mut self);
    fn leave_alternate_screen(&mut self);
    fn restore_termios(&mut self);
    fn print_restart(&mut self);
    fn exec(&mut self) -> io::Error;
}

pub fn restore_then_exec(host: &mut impl TtyRestore) -> io::Error {
    host.disable_raw_mode();
    host.leave_alternate_screen();
    host.restore_termios();
    host.print_restart();
    host.exec()
}

pub(crate) struct CrosstermRestore<'a> {
    restore: Option<Restore>,
    termios: Option<&'a libc::termios>,
    kind: ExecKind,
}

pub(crate) enum ExecKind {
    Supervisor {
        session_file: PathBuf,
    },
    Repair {
        incident: PathBuf,
    },
    /// `/reload`: the rebuilt `hi` on disk, resuming `session_file`.
    Reload {
        binary: PathBuf,
        session_file: PathBuf,
    },
}

impl TtyRestore for CrosstermRestore<'_> {
    fn disable_raw_mode(&mut self) {
        if let Some(restore) = self.restore.take() {
            restore.restore_now();
        } else {
            Restore::restore_now_static();
        }
    }

    fn leave_alternate_screen(&mut self) {
        // `Restore::restore_now` already left the alternate screen.
    }

    fn restore_termios(&mut self) {
        restore_termios(self.termios);
    }

    fn print_restart(&mut self) {
        match &self.kind {
            ExecKind::Reload { binary, .. } => {
                println!("reloading {} with this session…", binary.display())
            }
            _ => println!("{}", restart_line()),
        }
        let _ = io::stdout().flush();
    }

    fn exec(&mut self) -> io::Error {
        match &self.kind {
            ExecKind::Supervisor { session_file } => exec_supervisor(session_file),
            ExecKind::Repair { incident } => exec_repair(incident),
            ExecKind::Reload {
                binary,
                session_file,
            } => exec_reload(binary, session_file),
        }
    }
}

/// Same argv minus the one-shot prompt and cwd resume flags, pinned to the
/// session file, so the new build reopens exactly this conversation.
#[cfg(unix)]
fn exec_reload(binary: &std::path::Path, session_file: &std::path::Path) -> io::Error {
    use std::os::unix::process::CommandExt;
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let args = hi_sentinel::relaunch_args(&args, Some(session_file));
    hi_sentinel::unlink_empty_crash_markers_for_current_pid();
    std::process::Command::new(binary).args(args).exec()
}

#[cfg(not(unix))]
fn exec_reload(_binary: &std::path::Path, _session_file: &std::path::Path) -> io::Error {
    io::Error::other("/reload is Unix-only")
}

/// The `hi` path this process was started from. After a rebuild Linux reports
/// the replaced inode as `… (deleted)`; the path itself holds the new build.
pub(crate) fn binary_on_disk() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let text = exe.to_string_lossy();
    let path = PathBuf::from(text.strip_suffix(" (deleted)").unwrap_or(&text).to_string());
    path.is_file().then_some(path)
}

/// Notices when the `hi` binary on disk is no longer the build this process
/// runs. Every fix during the drme stalls needed a restart that was easy to
/// miss, so the old build kept stalling.
pub(crate) struct BuildWatch {
    running: Option<u64>,
    last_check: std::time::Instant,
    reported: bool,
}

const BUILD_CHECK_EVERY: std::time::Duration = std::time::Duration::from_secs(15);

impl BuildWatch {
    pub(crate) fn new() -> Self {
        Self {
            running: running_inode(),
            last_check: std::time::Instant::now(),
            reported: false,
        }
    }

    /// True once, the first time the file on disk is a different build.
    pub(crate) fn newer_build_on_disk(&mut self) -> bool {
        if self.reported || self.last_check.elapsed() < BUILD_CHECK_EVERY {
            return false;
        }
        self.last_check = std::time::Instant::now();
        let (Some(running), Some(disk)) = (self.running, binary_on_disk().and_then(|p| inode(&p)))
        else {
            return false;
        };
        self.reported = running != disk;
        self.reported
    }
}

#[cfg(unix)]
fn inode(path: &std::path::Path) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path).ok().map(|meta| meta.ino())
}

#[cfg(not(unix))]
fn inode(_path: &std::path::Path) -> Option<u64> {
    None
}

/// Inode of the running executable. `/proc/self/exe` still resolves after the
/// file is replaced; elsewhere the path is read once at startup.
fn running_inode() -> Option<u64> {
    let proc_exe = std::path::Path::new("/proc/self/exe");
    if proc_exe.exists() {
        return inode(proc_exe);
    }
    binary_on_disk().and_then(|path| inode(&path))
}

/// `/reload`: replace this process with the `hi` binary on disk and reopen
/// the current session.
pub(crate) fn reload(
    app: &mut App,
    harness: &Harness,
    restore: &mut Option<Restore>,
    termios: Option<&libc::termios>,
) -> Result<Option<String>> {
    let Some(session_file) = harness.session_path().map(std::path::Path::to_path_buf) else {
        app.push(Line::styled(
            "this session is not saved, so /reload cannot reopen it; exit and run `hi` again",
            dim(),
        ));
        return Ok(None);
    };
    let Some(binary) = binary_on_disk() else {
        app.push(Line::styled(
            "cannot find the hi binary on disk to reload",
            dim(),
        ));
        return Ok(None);
    };
    exec_after_restore(
        restore,
        termios,
        ExecKind::Reload {
            binary,
            session_file,
        },
    )
}

pub(crate) fn snapshot_termios() -> Option<libc::termios> {
    #[cfg(unix)]
    {
        let fd = libc::STDIN_FILENO;
        if unsafe { libc::isatty(fd) } != 1 {
            return None;
        }
        let mut termios = std::mem::MaybeUninit::<libc::termios>::uninit();
        let rc = unsafe { libc::tcgetattr(fd, termios.as_mut_ptr()) };
        if rc == 0 {
            Some(unsafe { termios.assume_init() })
        } else {
            None
        }
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn restore_termios(orig: Option<&libc::termios>) {
    #[cfg(unix)]
    {
        let Some(orig) = orig else {
            return;
        };
        let fd = libc::STDIN_FILENO;
        unsafe {
            libc::tcsetattr(fd, libc::TCSAFLUSH, orig);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = orig;
    }
}

pub(crate) fn handle(
    app: &mut App,
    harness: &Harness,
    arg: &str,
    opts: &AutoharnessfixOpts,
    restore: &mut Option<Restore>,
    termios: Option<&libc::termios>,
) -> Result<Option<String>> {
    let outcome = dispatch_slash(
        arg,
        opts.session_path.as_deref(),
        opts.no_save,
        opts.sentinel_blocked.as_deref(),
        harness.workspace_root(),
    )?;
    match outcome {
        SlashOutcome::Message(text) => {
            for line in text.lines() {
                app.push(Line::styled(line.to_string(), dim()));
            }
            Ok(None)
        }
        SlashOutcome::ExecSupervisor { session_file } => {
            exec_after_restore(restore, termios, ExecKind::Supervisor { session_file })
        }
        SlashOutcome::ExecRepair { incident } => {
            exec_after_restore(restore, termios, ExecKind::Repair { incident })
        }
    }
}

fn exec_after_restore(
    restore: &mut Option<Restore>,
    termios: Option<&libc::termios>,
    kind: ExecKind,
) -> Result<Option<String>> {
    let mut host = CrosstermRestore {
        restore: restore.take(),
        termios,
        kind,
    };
    let target = match &host.kind {
        ExecKind::Reload { binary, .. } => binary.display().to_string(),
        _ => "hi-sentinel".to_string(),
    };
    let err = restore_then_exec(&mut host);
    Err(anyhow::anyhow!("exec {target}: {err}"))
}

/// Mirrors `CrosstermRestore`: `Restore::restore_now_static` already left the
/// alternate screen, so `leave_alternate_screen` is a no-op.
#[cfg(test)]
struct RecordingCrossterm {
    order: Vec<&'static str>,
}

#[cfg(test)]
impl TtyRestore for RecordingCrossterm {
    fn disable_raw_mode(&mut self) {
        self.order.push("disable_raw_mode");
        self.order.push("LeaveAlternateScreen");
    }
    fn leave_alternate_screen(&mut self) {}
    fn restore_termios(&mut self) {
        self.order.push("restore_termios");
    }
    fn print_restart(&mut self) {
        self.order.push("print_restart");
    }
    fn exec(&mut self) -> io::Error {
        self.order.push("exec");
        io::Error::other("fake exec")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_happens_before_exec() {
        let mut host = RecordingCrossterm { order: Vec::new() };
        let err = restore_then_exec(&mut host);
        assert_eq!(err.to_string(), "fake exec");
        assert_eq!(
            host.order,
            [
                "disable_raw_mode",
                "LeaveAlternateScreen",
                "restore_termios",
                "print_restart",
                "exec",
            ]
        );
    }

    #[test]
    fn restore_now_static_sequence_includes_leave_alternate_screen() {
        let mut buf = Vec::new();
        crate::event::write_leave_session_screen(&mut buf).unwrap();
        let text = String::from_utf8_lossy(&buf);
        assert!(
            text.contains("1049l"),
            "CrosstermRestore disable_raw_mode must emit LeaveAlternateScreen: {text:?}"
        );
    }
}
