use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use portable_pty::{Child, CommandBuilder, ExitStatus, MasterPty, PtySize, native_pty_system};

pub(crate) const MAX_RAW_TERMINAL_BYTES: usize = 4 * 1024 * 1024;
const RUN_MARKER_ENV: &str = "HI_SMOKE_RUN_MARKER";
static NEXT_RUN_MARKER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Default)]
pub(crate) struct RawTerminal {
    pub bytes: Vec<u8>,
    pub truncated: bool,
    pub total_bytes: u64,
}

#[derive(Debug)]
struct ReaderFailure(String);

pub(crate) struct PtyProcess {
    master: Option<Box<dyn MasterPty + Send>>,
    writer: Option<Box<dyn Write + Send>>,
    child: Box<dyn Child + Send + Sync>,
    raw: Arc<Mutex<RawTerminal>>,
    parser: Arc<Mutex<vt100::Parser>>,
    reader_failure: Arc<Mutex<Option<ReaderFailure>>>,
    reader_thread: Option<JoinHandle<()>>,
    status: Option<ExitStatus>,
    cleanup_complete: bool,
    process_id: Option<u32>,
    run_marker: String,
    #[cfg(unix)]
    process_group: Option<libc::pid_t>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MarkedProcess {
    pub pid: libc::pid_t,
    pub ppid: libc::pid_t,
    pub pgid: libc::pid_t,
    pub command: String,
}

pub(crate) struct SpawnSpec<'a> {
    pub executable: &'a Path,
    pub args: &'a [String],
    pub cwd: &'a Path,
    pub env: &'a BTreeMap<String, String>,
    pub cols: u16,
    pub rows: u16,
}

impl PtyProcess {
    pub(crate) fn spawn(spec: SpawnSpec<'_>) -> Result<Self> {
        let system = native_pty_system();
        let pair = system
            .openpty(PtySize {
                rows: spec.rows,
                cols: spec.cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("opening pseudo-terminal")?;

        let mut command = CommandBuilder::new(spec.executable);
        command.args(spec.args);
        command.cwd(spec.cwd);
        command.env_clear();
        for (key, value) in spec.env {
            command.env(key, value);
        }
        let run_marker = next_run_marker();
        // This is deliberately written after the scenario environment. It is
        // harness-owned identity, not a scenario customization point. Unlike
        // a process group it survives `setsid`, so daemonized descendants can
        // still be found, killed, and reported as leaks.
        command.env(RUN_MARKER_ENV, &run_marker);

        let child = pair.slave.spawn_command(command).with_context(|| {
            format!("spawning {} in pseudo-terminal", spec.executable.display())
        })?;
        drop(pair.slave);

        let process_id = child.process_id();
        #[cfg(unix)]
        // portable-pty calls `setsid` before exec, so the child PID is the
        // stable process-group identifier for the launched session. The
        // terminal foreground group can change as soon as the program starts
        // (or become stale when a short-lived leader exits), so it is only a
        // fallback when the child implementation cannot expose its PID.
        let process_group = process_id
            .and_then(|pid| libc::pid_t::try_from(pid).ok())
            .or_else(|| pair.master.process_group_leader());
        let mut reader = pair
            .master
            .try_clone_reader()
            .context("cloning pseudo-terminal reader")?;
        let writer = pair
            .master
            .take_writer()
            .context("taking pseudo-terminal writer")?;

        let raw = Arc::new(Mutex::new(RawTerminal::default()));
        let parser = Arc::new(Mutex::new(vt100::Parser::new(spec.rows, spec.cols, 5_000)));
        let reader_failure = Arc::new(Mutex::new(None));
        let thread_raw = Arc::clone(&raw);
        let thread_parser = Arc::clone(&parser);
        let thread_failure = Arc::clone(&reader_failure);
        let reader_thread = std::thread::Builder::new()
            .name("hi-smoke-pty-reader".into())
            .spawn(move || {
                let mut buffer = [0_u8; 16 * 1024];
                loop {
                    match reader.read(&mut buffer) {
                        Ok(0) => break,
                        Ok(length) => {
                            if let Ok(mut terminal) = thread_raw.lock() {
                                terminal.total_bytes =
                                    terminal.total_bytes.saturating_add(length as u64);
                                let remaining =
                                    MAX_RAW_TERMINAL_BYTES.saturating_sub(terminal.bytes.len());
                                let keep = remaining.min(length);
                                terminal.bytes.extend_from_slice(&buffer[..keep]);
                                terminal.truncated |= keep < length;
                            }
                            if let Ok(mut parser) = thread_parser.lock() {
                                parser.process(&buffer[..length]);
                            }
                        }
                        Err(error) if is_expected_pty_eof(&error) => break,
                        Err(error) => {
                            if let Ok(mut slot) = thread_failure.lock() {
                                *slot = Some(ReaderFailure(error.to_string()));
                            }
                            break;
                        }
                    }
                }
            })
            .context("starting pseudo-terminal reader")?;

        Ok(Self {
            master: Some(pair.master),
            writer: Some(writer),
            child,
            raw,
            parser,
            reader_failure,
            reader_thread: Some(reader_thread),
            status: None,
            cleanup_complete: false,
            process_id,
            run_marker,
            #[cfg(unix)]
            process_group,
        })
    }

    pub(crate) fn send_line(&mut self, text: &str) -> Result<()> {
        // Bracketed paste prevents terminal escape sequences in scenario text from becoming keys.
        self.send_bytes(b"\x1b[200~")?;
        self.send_bytes(text.as_bytes())?;
        self.send_bytes(b"\x1b[201~\r")
    }

    pub(crate) fn send_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("PTY input is closed"))?;
        writer.write_all(bytes).context("writing PTY input")?;
        writer.flush().context("flushing PTY input")
    }

    pub(crate) fn resize(&mut self, cols: u16, rows: u16) -> Result<()> {
        self.master
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("PTY is closed"))?
            .resize(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("resizing pseudo-terminal")?;
        self.parser
            .lock()
            .map_err(|_| anyhow::anyhow!("virtual terminal parser lock was poisoned"))?
            .screen_mut()
            .set_size(rows, cols);
        Ok(())
    }

    pub(crate) fn screen(&self) -> Result<String> {
        self.parser
            .lock()
            .map(|parser| parser.screen().contents())
            .map_err(|_| anyhow::anyhow!("virtual terminal parser lock was poisoned"))
    }

    pub(crate) fn raw(&self) -> Result<RawTerminal> {
        self.raw
            .lock()
            .map(|raw| raw.clone())
            .map_err(|_| anyhow::anyhow!("raw terminal evidence lock was poisoned"))
    }

    #[cfg(test)]
    pub(crate) fn poison_evidence_locks_for_test(&self) {
        let parser = Arc::clone(&self.parser);
        let _ = std::thread::spawn(move || {
            let _guard = parser.lock().expect("locking parser before poison");
            panic!("poison virtual terminal evidence");
        })
        .join();
        let raw = Arc::clone(&self.raw);
        let _ = std::thread::spawn(move || {
            let _guard = raw.lock().expect("locking raw capture before poison");
            panic!("poison raw terminal evidence");
        })
        .join();
    }

    pub(crate) fn process_id(&self) -> Option<u32> {
        self.process_id
    }

    pub(crate) fn run_marker(&self) -> &str {
        &self.run_marker
    }

    #[cfg(unix)]
    pub(crate) fn process_group(&self) -> Option<libc::pid_t> {
        self.process_group
    }

    fn try_wait_child(&mut self) -> Result<Option<ExitStatus>> {
        if let Some(status) = &self.status {
            return Ok(Some(status.clone()));
        }
        let status = self.child.try_wait().context("polling hi process")?;
        if let Some(status) = status {
            self.status = Some(status.clone());
            Ok(Some(status))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        let status = self.try_wait_child()?;
        self.check_reader()?;
        Ok(status)
    }

    #[cfg(test)]
    pub(crate) fn wait_until(&mut self, deadline: Instant) -> Result<Option<ExitStatus>> {
        while Instant::now() < deadline {
            if let Some(status) = self.try_wait()? {
                return Ok(Some(status));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        Ok(None)
    }

    fn wait_until_child(&mut self, deadline: Instant) -> Result<Option<ExitStatus>> {
        while Instant::now() < deadline {
            if let Some(status) = self.try_wait_child()? {
                return Ok(Some(status));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        Ok(None)
    }

    /// Terminates the entire PTY session/process group, allowing two seconds for graceful exit.
    pub(crate) fn terminate_group(&mut self) -> Result<Option<ExitStatus>> {
        if self.cleanup_complete {
            self.check_reader()?;
            return Ok(self.status.clone());
        }

        // Reader failures are evidence failures, not permission to abandon a
        // live process tree. Poll and reap the child without consulting that
        // evidence until TERM/KILL cleanup and PTY closure are complete, then
        // surface the original reader failure to the caller.
        let _ = self.try_wait_child()?;

        #[cfg(unix)]
        if self.status.is_none()
            && let Some(group) = self.process_group
        {
            signal_process_group(group, libc::SIGTERM)?;
        }
        #[cfg(unix)]
        signal_marked_processes(&self.run_marker, libc::SIGTERM)?;
        #[cfg(not(unix))]
        self.child.kill().context("terminating hi process")?;

        let grace_deadline = Instant::now() + Duration::from_secs(2);
        if self.status.is_none() {
            let _ = self.wait_until_child(grace_deadline)?;
        }

        #[cfg(unix)]
        loop {
            if self.status.is_none() {
                let _ = self.try_wait_child()?;
            }
            // Once the leader has been reaped, its numeric PID/PGID can be
            // reused by an unrelated process. At that point only the unique
            // run marker is safe ownership evidence for remaining children.
            let group_exists =
                self.status.is_none() && self.process_group.is_some_and(process_group_exists);
            let marked = collect_marked_processes(&self.run_marker)?;
            if !group_exists && marked.is_empty() {
                break;
            }
            if Instant::now() >= grace_deadline {
                break;
            }
            // A child can call `setsid` while shutdown is in flight. Rescan
            // and signal on every pass so that race cannot escape cleanup.
            for process in marked {
                signal_process(process.pid, libc::SIGTERM)?;
            }
            std::thread::sleep(Duration::from_millis(20));
        }

        #[cfg(unix)]
        {
            if self.status.is_none() {
                let _ = self.try_wait_child()?;
            }
            if self.status.is_none()
                && let Some(group) = self
                    .process_group
                    .filter(|group| process_group_exists(*group))
            {
                signal_process_group(group, libc::SIGKILL)?;
            }
        }
        #[cfg(unix)]
        signal_marked_processes(&self.run_marker, libc::SIGKILL)?;
        // A direct kill is a fallback when the platform could not resolve a process group.
        if self.status.is_none() {
            let _ = self.child.kill();
            let _ = self.wait_until_child(Instant::now() + Duration::from_secs(2))?;
        }

        #[cfg(unix)]
        {
            let kill_deadline = Instant::now() + Duration::from_secs(2);
            loop {
                let remaining = collect_marked_processes(&self.run_marker)?;
                if remaining.is_empty() {
                    break;
                }
                for process in &remaining {
                    signal_process(process.pid, libc::SIGKILL)?;
                }
                if Instant::now() >= kill_deadline {
                    bail!(
                        "processes bearing run marker {} survived SIGKILL: {:?}",
                        self.run_marker,
                        remaining
                    );
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }

        // Closing our PTY handles lets the reader finish even when the child
        // exited without flushing. Never wait forever on a descendant that
        // inherited the slave: bounded cleanup is itself a harness invariant.
        self.writer.take();
        self.master.take();
        self.finish_reader(Duration::from_secs(2));
        self.cleanup_complete = true;
        self.check_reader()?;
        Ok(self.status.clone())
    }

    fn finish_reader(&mut self, timeout: Duration) {
        if let Some(thread) = self.reader_thread.take() {
            let deadline = Instant::now() + timeout;
            while !thread.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            if thread.is_finished() {
                let _ = thread.join();
            }
            // Dropping a still-running JoinHandle detaches it. That is safer
            // than wedging the complete smoke run on a leaked slave handle;
            // descendant leak detection remains authoritative.
        }
    }

    fn check_reader(&self) -> Result<()> {
        if let Ok(failure) = self.reader_failure.lock()
            && let Some(failure) = &*failure
        {
            bail!("PTY reader failed: {}", failure.0);
        }
        Ok(())
    }
}

impl Drop for PtyProcess {
    fn drop(&mut self) {
        let _ = self.terminate_group();
        self.finish_reader(Duration::ZERO);
    }
}

#[cfg(unix)]
fn signal_process_group(group: libc::pid_t, signal: libc::c_int) -> Result<()> {
    if group <= 1 {
        bail!("refusing to signal invalid process group {group}");
    }
    // SAFETY: kill is called with a validated positive process-group identifier and signal.
    let result = unsafe { libc::kill(-group, signal) };
    if result == -1 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            return Err(error).with_context(|| format!("signalling process group {group}"));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn signal_process(pid: libc::pid_t, signal: libc::c_int) -> Result<()> {
    if pid <= 1 || pid == std::process::id() as libc::pid_t {
        bail!("refusing to signal invalid harness descendant pid {pid}");
    }
    // SAFETY: `pid` is validated above and the signal is supplied by the
    // cleanup implementation.
    let result = unsafe { libc::kill(pid, signal) };
    if result == -1 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            return Err(error).with_context(|| format!("signalling descendant process {pid}"));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn signal_marked_processes(marker: &str, signal: libc::c_int) -> Result<()> {
    for process in collect_marked_processes(marker)? {
        signal_process(process.pid, signal)?;
    }
    Ok(())
}

#[cfg(unix)]
fn process_group_exists(group: libc::pid_t) -> bool {
    if group <= 1 {
        return false;
    }
    // SAFETY: signal 0 performs an existence/permission check and does not
    // deliver a signal; `group` was obtained from the PTY implementation.
    let result = unsafe { libc::kill(-group, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

fn next_run_marker() -> String {
    let sequence = NEXT_RUN_MARKER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{nanos:x}-{sequence:x}", std::process::id())
}

#[cfg(target_os = "linux")]
pub(crate) fn collect_marked_processes(marker: &str) -> Result<Vec<MarkedProcess>> {
    let expected = format!("{RUN_MARKER_ENV}={marker}");
    let mut matches = Vec::new();
    for entry in std::fs::read_dir("/proc").context("enumerating /proc for smoke descendants")? {
        let Ok(entry) = entry else {
            continue;
        };
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<libc::pid_t>().ok())
        else {
            continue;
        };
        let process_dir = entry.path();
        let Ok(environ) = std::fs::read(process_dir.join("environ")) else {
            continue;
        };
        if !environ
            .split(|byte| *byte == 0)
            .any(|value| value == expected.as_bytes())
        {
            continue;
        }
        let ppid = std::fs::read_to_string(process_dir.join("status"))
            .ok()
            .and_then(|status| {
                status
                    .lines()
                    .find_map(|line| line.strip_prefix("PPid:"))
                    .and_then(|value| value.trim().parse::<libc::pid_t>().ok())
            })
            .unwrap_or_default();
        // SAFETY: getpgid is a read-only query for a pid obtained from /proc.
        let pgid = unsafe { libc::getpgid(pid) };
        let command = std::fs::read(process_dir.join("cmdline"))
            .ok()
            .map(|bytes| {
                bytes
                    .split(|byte| *byte == 0)
                    .filter(|part| !part.is_empty())
                    .map(String::from_utf8_lossy)
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default();
        matches.push(MarkedProcess {
            pid,
            ppid,
            pgid,
            command,
        });
    }
    Ok(matches)
}

#[cfg(target_os = "macos")]
pub(crate) fn collect_marked_processes(marker: &str) -> Result<Vec<MarkedProcess>> {
    use std::ffi::CStr;
    use std::mem;
    use std::os::raw::c_void;

    // SAFETY: the first call asks libproc for the required pid capacity.
    let capacity = unsafe { libc::proc_listallpids(std::ptr::null_mut(), 0) };
    if capacity < 1 {
        return Err(std::io::Error::last_os_error()).context("enumerating macOS processes");
    }
    let mut pids = vec![0 as libc::pid_t; capacity as usize + 32];
    // SAFETY: `pids` is initialized writable storage and the byte length
    // matches its allocation.
    let count = unsafe {
        libc::proc_listallpids(
            pids.as_mut_ptr().cast::<c_void>(),
            (pids.len() * mem::size_of::<libc::pid_t>()) as libc::c_int,
        )
    };
    if count < 1 {
        return Err(std::io::Error::last_os_error()).context("reading macOS process list");
    }
    pids.truncate(count as usize);

    let expected = format!("{RUN_MARKER_ENV}={marker}");
    let mut matches = Vec::new();
    for pid in pids.into_iter().filter(|pid| *pid > 1) {
        let Some(arguments) = macos_process_arguments(pid) else {
            continue;
        };
        if !arguments
            .split(|byte| *byte == 0)
            .any(|value| value == expected.as_bytes())
        {
            continue;
        }

        // SAFETY: proc_pidinfo writes exactly one proc_bsdinfo into valid,
        // initialized storage when it returns the expected size.
        let mut info = unsafe { mem::zeroed::<libc::proc_bsdinfo>() };
        let size = unsafe {
            libc::proc_pidinfo(
                pid,
                libc::PROC_PIDTBSDINFO,
                0,
                (&raw mut info).cast::<c_void>(),
                mem::size_of::<libc::proc_bsdinfo>() as libc::c_int,
            )
        };
        if size != mem::size_of::<libc::proc_bsdinfo>() as libc::c_int {
            continue;
        }
        // SAFETY: pbi_name is a fixed-size NUL-terminated C buffer supplied
        // by libproc.
        let command = unsafe { CStr::from_ptr(info.pbi_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        matches.push(MarkedProcess {
            pid,
            ppid: info.pbi_ppid as libc::pid_t,
            pgid: info.pbi_pgid as libc::pid_t,
            command,
        });
    }
    Ok(matches)
}

#[cfg(target_os = "macos")]
fn macos_process_arguments(pid: libc::pid_t) -> Option<Vec<u8>> {
    let mut mib = [libc::CTL_KERN, libc::KERN_PROCARGS2, pid];
    let mut length = 0_usize;
    // SAFETY: a null output pointer asks sysctl for the required allocation.
    if unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            mib.len() as libc::c_uint,
            std::ptr::null_mut(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    } == -1
        || length == 0
    {
        return None;
    }
    let mut bytes = vec![0_u8; length];
    // SAFETY: `bytes` is writable for `length` bytes, which was supplied by
    // the immediately preceding sysctl query.
    if unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            mib.len() as libc::c_uint,
            bytes.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    } == -1
    {
        return None;
    }
    bytes.truncate(length);
    Some(bytes)
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
pub(crate) fn collect_marked_processes(_marker: &str) -> Result<Vec<MarkedProcess>> {
    bail!("run-marker process discovery supports Linux and macOS")
}

#[cfg(not(unix))]
pub(crate) fn collect_marked_processes(_marker: &str) -> Result<Vec<MarkedProcess>> {
    Ok(Vec::new())
}

fn is_expected_pty_eof(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::UnexpectedEof
        || error.raw_os_error() == Some(libc::EIO)
        || error.raw_os_error() == Some(libc::EBADF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    const ESCAPE_HELPER_ENV: &str = "HI_SMOKE_TEST_ESCAPE_HELPER";

    #[test]
    fn raw_terminal_defaults_to_bounded_empty_capture() {
        let capture = RawTerminal::default();
        assert!(capture.bytes.is_empty());
        assert!(!capture.truncated);
        assert_eq!(capture.total_bytes, 0);
    }

    #[test]
    fn poisoned_terminal_evidence_locks_fail_closed() {
        let workspace = tempfile::tempdir().unwrap();
        let args = vec!["-c".to_string(), "sleep 30".to_string()];
        let mut process = PtyProcess::spawn(SpawnSpec {
            executable: Path::new("/bin/sh"),
            args: &args,
            cwd: workspace.path(),
            env: &BTreeMap::new(),
            cols: 80,
            rows: 24,
        })
        .unwrap();

        let parser = Arc::clone(&process.parser);
        let _ = std::thread::spawn(move || {
            let _guard = parser.lock().unwrap();
            panic!("poison virtual terminal evidence");
        })
        .join();
        let raw = Arc::clone(&process.raw);
        let _ = std::thread::spawn(move || {
            let _guard = raw.lock().unwrap();
            panic!("poison raw terminal evidence");
        })
        .join();

        assert!(
            process
                .screen()
                .unwrap_err()
                .to_string()
                .contains("parser lock was poisoned")
        );
        assert!(
            process
                .raw()
                .unwrap_err()
                .to_string()
                .contains("evidence lock was poisoned")
        );
        process.terminate_group().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn cleanup_preserves_an_already_observed_leader_exit_status() {
        let workspace = tempfile::tempdir().unwrap();
        let args = vec!["-c".to_string(), "exit 23".to_string()];
        let mut process = PtyProcess::spawn(SpawnSpec {
            executable: Path::new("/bin/sh"),
            args: &args,
            cwd: workspace.path(),
            env: &BTreeMap::new(),
            cols: 80,
            rows: 24,
        })
        .unwrap();

        let observed = process
            .wait_until(Instant::now() + Duration::from_secs(2))
            .unwrap()
            .expect("shell leader should exit");
        assert_eq!(observed.exit_code(), 23);
        let cleanup = process
            .terminate_group()
            .unwrap()
            .expect("cleanup should return the cached leader status");
        assert_eq!(cleanup.exit_code(), 23);
    }

    #[cfg(unix)]
    #[test]
    fn cleanup_does_not_signal_a_stale_group_after_the_leader_exits() {
        let workspace = tempfile::tempdir().unwrap();
        let args = vec!["-c".to_string(), "exit 0".to_string()];
        let mut process = PtyProcess::spawn(SpawnSpec {
            executable: Path::new("/bin/sh"),
            args: &args,
            cwd: workspace.path(),
            env: &BTreeMap::new(),
            cols: 80,
            rows: 24,
        })
        .unwrap();

        process
            .wait_until(Instant::now() + Duration::from_secs(2))
            .unwrap()
            .expect("shell leader should exit");
        // Simulate the cached numeric PGID becoming foreign after the leader
        // was reaped. Group 1 is deliberately rejected if signalling occurs.
        process.process_group = Some(1);
        process.terminate_group().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn exited_leader_does_not_block_reader_or_leave_its_process_group() {
        let workspace = tempfile::tempdir().unwrap();
        let args = vec!["-c".to_string(), "sleep 30 & exit 0".to_string()];
        let mut process = PtyProcess::spawn(SpawnSpec {
            executable: Path::new("/bin/sh"),
            args: &args,
            cwd: workspace.path(),
            env: &BTreeMap::new(),
            cols: 80,
            rows: 24,
        })
        .unwrap();
        let group = process.process_group().unwrap();
        assert_eq!(
            process
                .process_id()
                .and_then(|pid| libc::pid_t::try_from(pid).ok()),
            Some(group),
            "portable-pty's setsid child PID must be the stored process group"
        );
        let started = Instant::now();
        let status = process
            .wait_until(Instant::now() + Duration::from_secs(2))
            .unwrap()
            .expect("shell leader should exit");
        assert!(status.success());
        assert!(started.elapsed() < Duration::from_secs(2));
        process.terminate_group().unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while !collect_marked_processes(process.run_marker())
            .unwrap()
            .is_empty()
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            collect_marked_processes(process.run_marker())
                .unwrap()
                .is_empty(),
            "the exited leader's marked descendants must be gone even if its numeric PGID is reused"
        );
    }

    #[cfg(unix)]
    #[test]
    fn run_marker_detects_descendant_that_escaped_process_group() {
        let (mut process, escaped) = spawn_escaped_descendant();
        assert_ne!(escaped.pgid, process.process_group().unwrap());
        assert!(escaped.pid > 1);
        process.terminate_group().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn termination_kills_descendant_that_escaped_process_group() {
        let (mut process, escaped) = spawn_escaped_descendant();
        process.terminate_group().unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        while collect_marked_processes(process.run_marker())
            .unwrap()
            .iter()
            .any(|candidate| candidate.pid == escaped.pid)
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            collect_marked_processes(process.run_marker())
                .unwrap()
                .is_empty()
        );
    }

    #[cfg(unix)]
    #[test]
    fn reader_failure_does_not_skip_sleeping_descendant_cleanup() {
        let workspace = tempfile::tempdir().unwrap();
        let args = vec![
            "-c".to_string(),
            "/bin/sleep 30 & child=$!; printf 'reader-evidence:%s\\n' \"$child\"; wait".to_string(),
        ];
        let mut process = PtyProcess::spawn(SpawnSpec {
            executable: Path::new("/bin/sh"),
            args: &args,
            cwd: workspace.path(),
            env: &BTreeMap::new(),
            cols: 80,
            rows: 24,
        })
        .unwrap();
        let leader = process.process_id().unwrap() as libc::pid_t;
        let group = process.process_group().unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        let sleeping_descendant = loop {
            let raw = process.raw().unwrap();
            let output = String::from_utf8_lossy(&raw.bytes);
            if let Some(pid) = output
                .lines()
                .find_map(|line| line.trim().strip_prefix("reader-evidence:"))
                .and_then(|pid| pid.parse::<libc::pid_t>().ok())
            {
                assert_ne!(pid, leader, "shell reported itself instead of its child");
                assert!(process_exists(pid), "reported sleeping child is not alive");
                break pid;
            }
            assert!(
                Instant::now() < deadline,
                "shell did not emit its sleeping descendant pid; raw={output:?}"
            );
            std::thread::sleep(Duration::from_millis(20));
        };
        assert!(process_group_exists(group));

        *process.reader_failure.lock().unwrap() =
            Some(ReaderFailure("injected read failure".into()));
        assert!(
            process
                .try_wait()
                .unwrap_err()
                .to_string()
                .contains("injected read failure")
        );

        let cleanup_error = process.terminate_group().unwrap_err().to_string();
        assert!(cleanup_error.contains("PTY reader failed: injected read failure"));
        assert!(
            process.cleanup_complete,
            "cleanup must finish before reporting"
        );
        assert!(process.status.is_some(), "the PTY leader must be reaped");
        assert!(
            collect_marked_processes(process.run_marker())
                .unwrap()
                .is_empty(),
            "reader failure left marked processes behind; sleeping descendant was {sleeping_descendant}"
        );
        let descendant_deadline = Instant::now() + Duration::from_secs(2);
        while process_exists(sleeping_descendant) && Instant::now() < descendant_deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            !process_exists(sleeping_descendant),
            "sleeping descendant {sleeping_descendant} survived cleanup"
        );
        assert!(
            !process_group_exists(group),
            "PTY process group survived cleanup"
        );
        assert!(
            process
                .raw()
                .unwrap()
                .bytes
                .windows(b"reader-evidence".len())
                .any(|window| window == b"reader-evidence"),
            "terminal evidence must remain readable after cleanup"
        );
        assert!(
            process
                .terminate_group()
                .unwrap_err()
                .to_string()
                .contains("injected read failure"),
            "the reader failure must remain reportable after cleanup"
        );
    }

    #[cfg(unix)]
    fn process_exists(pid: libc::pid_t) -> bool {
        // SAFETY: signal 0 only queries a PID reported by the test child.
        let result = unsafe { libc::kill(pid, 0) };
        result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }

    #[cfg(unix)]
    fn spawn_escaped_descendant() -> (PtyProcess, MarkedProcess) {
        let workspace = tempfile::tempdir().unwrap();
        let executable = std::env::current_exe().unwrap();
        let args = vec![
            "--exact".to_string(),
            "pty::tests::escaped_descendant_helper".to_string(),
            "--test-threads=1".to_string(),
        ];
        let env = BTreeMap::from([(ESCAPE_HELPER_ENV.to_string(), "1".to_string())]);
        let mut process = PtyProcess::spawn(SpawnSpec {
            executable: &executable,
            args: &args,
            cwd: workspace.path(),
            env: &env,
            cols: 80,
            rows: 24,
        })
        .unwrap();
        let status = process
            .wait_until(Instant::now() + Duration::from_secs(5))
            .unwrap()
            .expect("escape helper leader should exit");
        assert!(
            status.success(),
            "escape helper failed before its descendant became ready: status={status:?}, output={}",
            String::from_utf8_lossy(&process.raw().unwrap().bytes)
        );

        let original_group = process.process_group().unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(escaped) = collect_marked_processes(process.run_marker())
                .unwrap()
                .into_iter()
                .find(|candidate| candidate.pgid != original_group)
            {
                return (process, escaped);
            }
            assert!(
                Instant::now() < deadline,
                "setsid descendant was not discovered by its inherited run marker"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Subprocess-only helper for the escaped-descendant regression tests.
    /// The surrounding test process forks once, waits on a pipe until the
    /// child confirms that `setsid` succeeded, then exits normally and leaves
    /// the fork in a new session until the parent smoke harness cleans it up.
    #[cfg(unix)]
    #[test]
    fn escaped_descendant_helper() {
        if std::env::var_os(ESCAPE_HELPER_ENV).is_none() {
            return;
        }
        // SAFETY: the forked child calls only async-signal-safe libc routines
        // before it is terminated by the outer test. A pipe handshake makes
        // the parent's successful exit proof that the child completed
        // `setsid`; process-list polling is no longer used as a readiness
        // signal and cannot race the fork.
        unsafe {
            let mut ready_pipe = [-1; 2];
            assert_eq!(
                libc::pipe(ready_pipe.as_mut_ptr()),
                0,
                "pipe failed: {}",
                std::io::Error::last_os_error()
            );
            let child = libc::fork();
            if child < 0 {
                let error = std::io::Error::last_os_error();
                let _ = libc::close(ready_pipe[0]);
                let _ = libc::close(ready_pipe[1]);
                panic!("fork failed: {error}");
            }
            if child == 0 {
                let _ = libc::close(ready_pipe[0]);
                if libc::setsid() == -1 {
                    libc::_exit(101);
                }
                let ready = [1_u8];
                if libc::write(ready_pipe[1], ready.as_ptr().cast(), ready.len())
                    != ready.len() as isize
                {
                    libc::_exit(102);
                }
                let _ = libc::close(ready_pipe[1]);
                loop {
                    libc::pause();
                }
            }
            let _ = libc::close(ready_pipe[1]);
            let mut ready = [0_u8];
            let received = loop {
                let received = libc::read(ready_pipe[0], ready.as_mut_ptr().cast(), ready.len());
                if received == -1
                    && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR)
                {
                    continue;
                }
                break received;
            };
            let _ = libc::close(ready_pipe[0]);
            assert_eq!(received, 1, "setsid child closed its readiness pipe");
            assert_eq!(ready, [1], "setsid child sent an invalid readiness byte");
        }
    }
}
