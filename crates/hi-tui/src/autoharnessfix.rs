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
    Supervisor { session_file: PathBuf },
    Repair { incident: PathBuf },
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
        println!("{}", restart_line());
        let _ = io::stdout().flush();
    }

    fn exec(&mut self) -> io::Error {
        match &self.kind {
            ExecKind::Supervisor { session_file } => exec_supervisor(session_file),
            ExecKind::Repair { incident } => exec_repair(incident),
        }
    }
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
    let err = restore_then_exec(&mut host);
    Err(anyhow::anyhow!("exec hi-sentinel: {err}"))
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
