//! Unix crash handler — SIGBUS/SIGSEGV via `sigaction(2)`.
//!
//! Captures crash PC + frame-pointer chain. All handler operations are
//! minimal (raw pointer reads, direct file I/O, atomics — no allocation).
//! The crash PC is written to disk before frame walking so a secondary
//! fault during the walk still produces a usable report.

use std::cell::RefCell;
use std::ffi::CString;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};

use crate::{CrashReport, MAX_FRAMES};

/// Raw crash info written to a process-scoped marker by the signal handler.
#[repr(C)]
pub struct CrashInfo {
    /// Magic bytes for validation.
    magic: [u8; 4],
    /// Signal number.
    signum: i32,
    /// `si_code` from `siginfo_t`.
    si_code: i32,
    /// Explicit alignment bytes keep the on-disk record fully initialized.
    _align_pc: [u8; 4],
    /// Crash instruction pointer.
    crash_pc: u64,
    /// Unix timestamp (seconds since epoch).
    timestamp: u64,
    /// Number of valid frames in the backtrace.
    frame_count: u32,
    /// Explicit alignment bytes before the address array.
    _align_frames: [u8; 4],
    /// Frame-pointer chain (instruction addresses).
    frames: [u64; MAX_FRAMES],
    /// App version string (null-terminated, up to 64 bytes).
    app_version: [u8; 64],
}

const CRASH_MAGIC: [u8; 4] = *b"HICR";
/// Legacy single-process marker name, retained for startup migration.
const CRASH_FILE: &str = "last-crash.bin";
const CRASH_FILE_PREFIX: &str = "last-crash-";
const CRASH_FILE_SUFFIX: &str = ".bin";
const ALT_STACK_BYTES: usize = 64 * 1024;

static INSTALLED: AtomicBool = AtomicBool::new(false);
static REPORT_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct ThreadAltStack {
    // Keeps the registered memory alive until the owning thread exits.
    _memory: Box<[u8]>,
}

impl Drop for ThreadAltStack {
    fn drop(&mut self) {
        // Alternate stacks are per-thread. Disable this thread's registration
        // before releasing its backing allocation during TLS teardown.
        let disabled = libc::stack_t {
            ss_sp: std::ptr::null_mut(),
            ss_flags: libc::SS_DISABLE,
            ss_size: 0,
        };
        // SAFETY: `disabled` is a valid stack_t and the output pointer is null.
        unsafe {
            libc::sigaltstack(&disabled, std::ptr::null_mut());
        }
    }
}

thread_local! {
    static THREAD_ALT_STACK: RefCell<Option<ThreadAltStack>> = const { RefCell::new(None) };
}

/// Install an alternate signal stack for the current thread.
///
/// Signal dispositions are process-wide, but `sigaltstack(2)` is per-thread.
/// Applications must call this on every thread that may execute application
/// work (runtime thread-start hooks are the usual integration point).
pub fn install_thread_alt_stack() -> bool {
    THREAD_ALT_STACK.with(|slot| {
        if slot.borrow().is_some() {
            return true;
        }

        let mut memory = vec![0u8; ALT_STACK_BYTES].into_boxed_slice();
        let stack = libc::stack_t {
            ss_sp: memory.as_mut_ptr().cast(),
            ss_flags: 0,
            ss_size: memory.len(),
        };
        // SAFETY: the boxed allocation is stable and is retained in TLS for
        // the remainder of this thread's lifetime on success.
        if unsafe { libc::sigaltstack(&stack, std::ptr::null_mut()) } != 0 {
            return false;
        }
        *slot.borrow_mut() = Some(ThreadAltStack { _memory: memory });
        true
    })
}

/// Install the SIGBUS/SIGSEGV handler. Creates `crash_dir` and opens
/// a process-scoped marker without following symlinks or truncating an
/// unverified inode. Process-scoped names are essential: a second live `hi`
/// session must not unlink the inode held open by the first. Returns `true` on
/// success.
pub fn install(crash_dir: &Path, app_version: &str) -> bool {
    // sigaltstack state is per-thread, not process-wide. Always cover the
    // caller even if another thread installed the process signal actions.
    if !install_thread_alt_stack() {
        eprintln!("hi-crash-handler: failed to install alternate signal stack");
        return false;
    }
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return true; // already installed
    }

    // Open the marker now, while allocation and path handling are safe. The
    // descriptor intentionally remains open for the process lifetime so the
    // signal handler needs only async-signal-safe descriptor operations.
    let crash_fd = match open_crash_marker(crash_dir) {
        Ok(marker) => marker.file.into_raw_fd(),
        Err(error) => {
            eprintln!("hi-crash-handler: failed to open crash marker: {error}");
            INSTALLED.store(false, Ordering::SeqCst);
            return false;
        }
    };
    let version_bytes = app_version.as_bytes();
    let mut version_buf = [0u8; 64];
    let copy_len = version_bytes.len().min(63);
    version_buf[..copy_len].copy_from_slice(&version_bytes[..copy_len]);

    // SAFETY: install_signal_handler is a C FFI call that sets up sigaction.
    // It's called once at startup before threads are spawned.
    unsafe {
        install_signal_handler(crash_fd, version_buf);
    }

    true
}

fn invalid_input(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
}

fn path_cstring(path: &Path) -> std::io::Result<CString> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| invalid_input("path contains NUL"))
}

fn name_cstring(name: &str) -> std::io::Result<CString> {
    if name.as_bytes().contains(&b'/') {
        return Err(invalid_input("directory entry contains a slash"));
    }
    CString::new(name).map_err(|_| invalid_input("directory entry contains NUL"))
}

fn open_crash_directory(crash_dir: &Path, create: bool) -> std::io::Result<OwnedFd> {
    if create {
        std::fs::create_dir_all(crash_dir)?;
    }
    let path = path_cstring(crash_dir)?;
    // SAFETY: path is NUL-terminated and live for this call. O_NOFOLLOW keeps
    // a planted final-component symlink from redirecting marker/report I/O.
    let raw = unsafe {
        libc::open(
            path.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: open returned a fresh descriptor now owned by this value.
    let directory = unsafe { OwnedFd::from_raw_fd(raw) };
    let stat = fd_stat(directory.as_raw_fd())?;
    if (stat.st_mode & libc::S_IFMT) != libc::S_IFDIR {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "crash path is not a directory",
        ));
    }
    if stat.st_uid != unsafe { libc::geteuid() } {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "crash directory is not owned by the current user",
        ));
    }
    // SAFETY: directory is a live descriptor owned by this process.
    if unsafe { libc::fchmod(directory.as_raw_fd(), 0o700) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(directory)
}

fn fd_stat(fd: libc::c_int) -> std::io::Result<libc::stat> {
    // SAFETY: stat is fully initialized by fstat on success.
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(stat)
}

fn validate_owned_regular_fd(fd: libc::c_int) -> std::io::Result<libc::stat> {
    let stat = fd_stat(fd)?;
    if (stat.st_mode & libc::S_IFMT) != libc::S_IFREG {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "crash artifact is not a regular file",
        ));
    }
    if stat.st_uid != unsafe { libc::geteuid() } {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "crash artifact is not owned by the current user",
        ));
    }
    if stat.st_nlink != 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "crash artifact has multiple hard links",
        ));
    }
    Ok(stat)
}

fn open_crash_marker(crash_dir: &Path) -> std::io::Result<OpenCrashMarker> {
    open_crash_marker_for_pid(crash_dir, std::process::id())
}

#[derive(Debug)]
struct OpenCrashMarker {
    file: OwnedFd,
    _name: String,
}

#[derive(Debug, Clone)]
struct MarkerIdentity {
    pid: u32,
    /// Validated filename scope used to keep reports collision-free too.
    scope: String,
}

fn marker_name(pid: u32, nonce: &str) -> String {
    format!("{CRASH_FILE_PREFIX}{pid}-{nonce}{CRASH_FILE_SUFFIX}")
}

fn marker_identity(name: &str) -> Option<MarkerIdentity> {
    let scope = name
        .strip_prefix(CRASH_FILE_PREFIX)?
        .strip_suffix(CRASH_FILE_SUFFIX)?;
    let (pid, nonce) = match scope.split_once('-') {
        Some((pid, nonce)) => (pid, Some(nonce)),
        None => (scope, None), // legacy PID-only marker
    };
    let pid = pid.parse().ok()?;
    if let Some(nonce) = nonce {
        // New markers use UUID v4 simple form. Validation prevents arbitrary
        // directory entry text from becoming part of a report filename.
        if nonce.len() != 32 || uuid::Uuid::parse_str(nonce).is_err() {
            return None;
        }
    }
    Some(MarkerIdentity {
        pid,
        scope: scope.to_string(),
    })
}

fn open_crash_marker_for_pid(crash_dir: &Path, pid: u32) -> std::io::Result<OpenCrashMarker> {
    let directory = open_crash_directory(crash_dir, true)?;
    for _ in 0..32 {
        // PID alone is not globally unique across PID namespaces or hosts
        // sharing a home directory. A random nonce plus O_EXCL means no live
        // process ever detaches another process's marker inode.
        let name = marker_name(pid, &uuid::Uuid::new_v4().simple().to_string());
        match open_named_crash_marker(&directory, &name) {
            Ok(file) => return Ok(OpenCrashMarker { file, _name: name }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a unique crash marker",
    ))
}

fn open_named_crash_marker(directory: &OwnedFd, marker_name: &str) -> std::io::Result<OwnedFd> {
    let name = name_cstring(marker_name)?;
    let raw = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            0o600,
        )
    };
    if raw < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: openat returned a fresh descriptor now owned by this value.
    let marker = unsafe { OwnedFd::from_raw_fd(raw) };
    validate_owned_regular_fd(marker.as_raw_fd())?;
    if unsafe { libc::fchmod(marker.as_raw_fd(), 0o600) } != 0
        || unsafe { libc::lseek(marker.as_raw_fd(), 0, libc::SEEK_SET) } < 0
    {
        return Err(std::io::Error::last_os_error());
    }
    Ok(marker)
}

fn read_crash_info(directory_fd: libc::c_int, marker_name: &str) -> std::io::Result<CrashInfo> {
    let name = name_cstring(marker_name)?;
    let raw = unsafe {
        libc::openat(
            directory_fd,
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
        )
    };
    if raw < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: openat returned a fresh descriptor now owned by this value.
    let marker = unsafe { OwnedFd::from_raw_fd(raw) };
    let stat = validate_owned_regular_fd(marker.as_raw_fd())?;
    if stat.st_size != std::mem::size_of::<CrashInfo>() as libc::off_t {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "crash marker has an invalid size",
        ));
    }
    if unsafe { libc::fchmod(marker.as_raw_fd(), 0o600) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut file = std::fs::File::from(marker);
    let mut data = vec![0u8; std::mem::size_of::<CrashInfo>()];
    file.read_exact(&mut data)?;
    // Vec<u8> does not promise CrashInfo alignment; the fixed-size read above
    // guarantees a complete record for an unaligned copy.
    Ok(unsafe { std::ptr::read_unaligned(data.as_ptr().cast::<CrashInfo>()) })
}

fn write_report_safely(
    directory_fd: libc::c_int,
    report_name: &str,
    contents: &[u8],
) -> std::io::Result<()> {
    let report_name = name_cstring(report_name)?;
    for _ in 0..32 {
        let sequence = REPORT_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temp_name = format!(".crash-report-{}-{sequence}.tmp", std::process::id());
        let temp_name = name_cstring(&temp_name)?;
        let raw = unsafe {
            libc::openat(
                directory_fd,
                temp_name.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                0o600,
            )
        };
        if raw < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                continue;
            }
            return Err(error);
        }
        // SAFETY: openat returned a fresh descriptor now owned by this value.
        let output = unsafe { OwnedFd::from_raw_fd(raw) };
        let result = (|| {
            validate_owned_regular_fd(output.as_raw_fd())?;
            if unsafe { libc::fchmod(output.as_raw_fd(), 0o600) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
            let mut output = std::fs::File::from(output);
            output.write_all(contents)?;
            output.sync_all()?;
            drop(output);
            if unsafe {
                libc::renameat(
                    directory_fd,
                    temp_name.as_ptr(),
                    directory_fd,
                    report_name.as_ptr(),
                )
            } != 0
            {
                return Err(std::io::Error::last_os_error());
            }
            // Best effort: the report contents are already fsynced, and some
            // filesystems reject fsync on directories.
            unsafe {
                libc::fsync(directory_fd);
            }
            Ok(())
        })();
        if result.is_err() {
            unsafe {
                libc::unlinkat(directory_fd, temp_name.as_ptr(), 0);
            }
        }
        return result;
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a private crash report temporary file",
    ))
}

fn marker_process_is_alive(pid: u32) -> bool {
    // `kill(0, 0)` addresses the current process group and casting a large
    // unsigned value can produce a negative process-group selector. Neither
    // is a valid process-scoped marker owner.
    if pid == 0 || pid > libc::pid_t::MAX as u32 {
        return false;
    }
    let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if result == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

fn marker_candidates(crash_dir: &Path) -> Vec<(String, Option<MarkerIdentity>)> {
    let Ok(entries) = std::fs::read_dir(crash_dir) else {
        return Vec::new();
    };
    let mut candidates = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            if name == CRASH_FILE {
                Some((name, None))
            } else {
                marker_identity(&name).map(|identity| (name, Some(identity)))
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.0.cmp(&right.0));
    candidates
}

/// Check for a previous crash by scanning legacy and process-scoped markers.
pub fn check_previous_crash(crash_dir: &Path) -> Option<CrashReport> {
    let directory = open_crash_directory(crash_dir, false).ok()?;
    for (marker, identity) in marker_candidates(crash_dir) {
        let info = match read_crash_info(directory.as_raw_fd(), &marker) {
            Ok(info) if info.magic == CRASH_MAGIC => info,
            _ => {
                // Clean exits leave an empty pre-opened marker. Reclaim only
                // markers whose owning process no longer exists; a concurrent
                // live process may still need its descriptor for a later crash.
                if identity
                    .as_ref()
                    .is_none_or(|identity| !marker_process_is_alive(identity.pid))
                    && let Ok(name) = name_cstring(&marker)
                {
                    unsafe {
                        libc::unlinkat(directory.as_raw_fd(), name.as_ptr(), 0);
                    }
                }
                continue;
            }
        };

        let signal_name = signal_name(info.signum);
        let report_name = match &identity {
            Some(identity) => format!("crash-{}-{}.txt", info.timestamp, identity.scope),
            None => format!("crash-{}.txt", info.timestamp),
        };
        let report_path = crash_dir.join(&report_name);

        let report_text = format!(
            "hi crash report\n\
             Signal: {signal_name}\n\
             si_code: {}\n\
             Crash PC: 0x{:x}\n\
             Timestamp: {}\n\
             Version: {}\n\
             Frames: {}\n",
            info.si_code,
            info.crash_pc,
            info.timestamp,
            std::str::from_utf8(&info.app_version)
                .unwrap_or("unknown")
                .trim_end_matches('\0'),
            info.frame_count,
        );
        write_report_safely(directory.as_raw_fd(), &report_name, report_text.as_bytes()).ok()?;

        // Remove exactly the marker that produced this report.
        if let Ok(name) = name_cstring(&marker) {
            unsafe {
                libc::unlinkat(directory.as_raw_fd(), name.as_ptr(), 0);
            }
        }

        return Some(CrashReport {
            signal_name,
            si_code: info.si_code,
            timestamp: info.timestamp,
            app_version: std::str::from_utf8(&info.app_version)
                .unwrap_or("unknown")
                .trim_end_matches('\0')
                .to_string(),
            report_path,
        });
    }
    None
}

fn signal_name(signum: i32) -> &'static str {
    match signum {
        libc::SIGSEGV => "SIGSEGV (Segmentation fault)",
        libc::SIGBUS => "SIGBUS (Bus error)",
        libc::SIGILL => "SIGILL (Illegal instruction)",
        libc::SIGFPE => "SIGFPE (Floating point exception)",
        libc::SIGABRT => "SIGABRT (Abort)",
        _ => "Unknown signal",
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};

    fn test_crash_info(timestamp: u64) -> CrashInfo {
        let mut app_version = [0u8; 64];
        app_version[..5].copy_from_slice(b"test\0");
        CrashInfo {
            magic: CRASH_MAGIC,
            signum: libc::SIGSEGV,
            si_code: 1,
            _align_pc: [0; 4],
            crash_pc: 0x1234,
            timestamp,
            frame_count: 0,
            _align_frames: [0; 4],
            frames: [0; MAX_FRAMES],
            app_version,
        }
    }

    fn write_crash_info_for_test(path: &Path, info: &CrashInfo) {
        // SAFETY: CrashInfo has no implicit padding (asserted below) and every
        // field in this test value is initialized.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (info as *const CrashInfo).cast::<u8>(),
                std::mem::size_of::<CrashInfo>(),
            )
        };
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn crash_record_has_no_implicit_padding() {
        let initialized_field_bytes = 4 + 4 + 4 + 4 + 8 + 8 + 4 + 4 + (8 * MAX_FRAMES) + 64;
        assert_eq!(std::mem::size_of::<CrashInfo>(), initialized_field_bytes);
    }

    #[test]
    fn marker_open_refuses_occupied_symlink_and_hardlink_without_touching_target() {
        let temp = tempfile::tempdir().unwrap();
        let crash_dir = temp.path().join("crash");
        std::fs::create_dir(&crash_dir).unwrap();
        let directory = open_crash_directory(&crash_dir, false).unwrap();
        let victim = temp.path().join("victim.txt");
        std::fs::write(&victim, "keep me").unwrap();

        let marker = marker_name(std::process::id(), "00000000000040008000000000000001");
        symlink(&victim, crash_dir.join(&marker)).unwrap();
        assert!(open_named_crash_marker(&directory, &marker).is_err());
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep me");
        std::fs::remove_file(crash_dir.join(&marker)).unwrap();

        std::fs::hard_link(&victim, crash_dir.join(&marker)).unwrap();
        assert!(open_named_crash_marker(&directory, &marker).is_err());
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep me");
    }

    #[test]
    fn crash_directory_final_symlink_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let actual = temp.path().join("actual");
        std::fs::create_dir(&actual).unwrap();
        let linked = temp.path().join("linked");
        symlink(&actual, &linked).unwrap();
        assert!(open_crash_marker(&linked).is_err());
        assert_eq!(std::fs::read_dir(actual).unwrap().count(), 0);
    }

    #[test]
    fn same_pid_sessions_get_unique_markers_without_detaching_each_other() {
        let temp = tempfile::tempdir().unwrap();
        let crash_dir = temp.path().join("crash");
        let shared_pid = 1_000_001;

        let first = open_crash_marker_for_pid(&crash_dir, shared_pid).unwrap();
        let second = open_crash_marker_for_pid(&crash_dir, shared_pid).unwrap();
        assert_ne!(first._name, second._name);
        let info = test_crash_info(99);
        // SAFETY: `first` remains a live writable descriptor for this call.
        unsafe { write_crash_info(first.file.as_raw_fd(), &info) };

        assert!(crash_dir.join(&first._name).exists());
        assert!(crash_dir.join(&second._name).exists());
        let directory = open_crash_directory(&crash_dir, false).unwrap();
        let persisted = read_crash_info(directory.as_raw_fd(), &first._name).unwrap();
        assert_eq!(persisted.magic, CRASH_MAGIC);
        assert_eq!(persisted.timestamp, 99);
        drop(second);
        drop(first);
    }

    #[test]
    fn startup_scan_preserves_an_empty_marker_owned_by_a_live_process() {
        let temp = tempfile::tempdir().unwrap();
        let open_marker = open_crash_marker(temp.path()).unwrap();
        let marker = open_marker._name.clone();

        assert!(check_previous_crash(temp.path()).is_none());
        assert!(temp.path().join(marker).exists());
        drop(open_marker);
    }

    #[test]
    fn marker_identity_accepts_nonce_and_legacy_names_only() {
        let nonce = "00000000000040008000000000000001";
        let current = marker_identity(&marker_name(42, nonce)).unwrap();
        assert_eq!(current.pid, 42);
        assert_eq!(current.scope, format!("42-{nonce}"));
        assert_eq!(marker_identity("last-crash-42.bin").unwrap().pid, 42);
        assert!(marker_identity("last-crash-42-not-a-uuid.bin").is_none());
    }

    #[test]
    fn alternate_signal_stacks_are_distinct_per_thread() {
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let spawn = |barrier: std::sync::Arc<std::sync::Barrier>| {
            std::thread::spawn(move || {
                assert!(install_thread_alt_stack());
                // SAFETY: the output stack_t is initialized by sigaltstack.
                let mut current: libc::stack_t = unsafe { std::mem::zeroed() };
                assert_eq!(
                    unsafe { libc::sigaltstack(std::ptr::null(), &mut current) },
                    0
                );
                assert_eq!(current.ss_flags & libc::SS_DISABLE, 0);
                let first = current.ss_sp as usize;
                assert!(install_thread_alt_stack());
                let mut repeated: libc::stack_t = unsafe { std::mem::zeroed() };
                assert_eq!(
                    unsafe { libc::sigaltstack(std::ptr::null(), &mut repeated) },
                    0
                );
                assert_eq!(first, repeated.ss_sp as usize);
                barrier.wait();
                (first, current.ss_size)
            })
        };
        let first = spawn(barrier.clone());
        let second = spawn(barrier.clone());
        barrier.wait();
        let first = first.join().unwrap();
        let second = second.join().unwrap();
        assert_ne!(first.0, second.0);
        assert!(first.1 >= ALT_STACK_BYTES && second.1 >= ALT_STACK_BYTES);
    }

    #[test]
    fn previous_crash_report_replaces_symlink_without_following_it() {
        let temp = tempfile::tempdir().unwrap();
        let crash_dir = temp.path().join("crash");
        std::fs::create_dir(&crash_dir).unwrap();
        std::fs::set_permissions(&crash_dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        write_crash_info_for_test(&crash_dir.join(CRASH_FILE), &test_crash_info(42));

        let victim = temp.path().join("victim.txt");
        std::fs::write(&victim, "keep me").unwrap();
        let report_path = crash_dir.join("crash-42.txt");
        symlink(&victim, &report_path).unwrap();

        let report = check_previous_crash(&crash_dir).expect("valid crash report");
        assert_eq!(report.report_path, report_path);
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep me");
        assert!(
            !std::fs::symlink_metadata(&report_path)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(
            std::fs::read_to_string(&report_path)
                .unwrap()
                .contains("Crash PC: 0x1234")
        );
        assert_eq!(
            std::fs::metadata(&report_path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        assert_eq!(
            std::fs::metadata(&crash_dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert!(!crash_dir.join(CRASH_FILE).exists());
    }
}

// ---------------------------------------------------------------------------
// Signal handler installation (platform-specific)
// ---------------------------------------------------------------------------

/// Pre-opened marker and app version, set once before handlers are registered.
static CRASH_FD: AtomicI32 = AtomicI32::new(-1);
static mut APP_VERSION_BUF: [u8; 64] = [0u8; 64];

/// Install the SIGBUS/SIGSEGV signal handler.
///
/// # Safety
/// Must be called once at startup before threads are spawned.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn install_signal_handler(crash_fd: libc::c_int, version_buf: [u8; 64]) {
    CRASH_FD.store(crash_fd, Ordering::Release);
    APP_VERSION_BUF = version_buf;

    let mut action: libc::sigaction = std::mem::zeroed();
    action.sa_sigaction = crash_handler as *const () as usize;
    // Reset before entering the handler. If frame walking itself faults, the
    // nested signal terminates with the default disposition instead of
    // recursively re-entering arbitrary-pointer walking.
    action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_RESTART | libc::SA_RESETHAND;
    libc::sigemptyset(&mut action.sa_mask);

    for &sig in &[libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGFPE] {
        libc::sigaction(sig, &action, std::ptr::null_mut());
    }
}

/// The signal handler. Writes crash info to its pre-opened process marker,
/// then re-raises the signal so the default handler produces a core dump.
///
/// # Safety
/// This is a signal handler — only async-signal-safe operations are allowed.
#[allow(unsafe_op_in_unsafe_fn)]
extern "C" fn crash_handler(
    signum: libc::c_int,
    info: *mut libc::siginfo_t,
    _ctx: *mut libc::c_void,
) {
    // SAFETY: we only do async-signal-safe operations here.
    unsafe {
        let si_code = if info.is_null() { 0 } else { (*info).si_code };
        let crash_pc = extract_pc(_ctx);

        let now = libc::time(std::ptr::null_mut());
        let timestamp = if now < 0 { 0 } else { now as u64 };

        let mut info = CrashInfo {
            magic: CRASH_MAGIC,
            signum,
            si_code,
            _align_pc: [0; 4],
            crash_pc,
            timestamp,
            frame_count: 0,
            _align_frames: [0; 4],
            frames: [0u64; MAX_FRAMES],
            app_version: APP_VERSION_BUF,
        };

        let fd = CRASH_FD.load(Ordering::Acquire);
        if fd >= 0 {
            // Persist the crash site before walking arbitrary frame pointers: a
            // secondary fault during unwinding still leaves a usable marker.
            write_crash_info(fd, &info);
            info.frame_count = capture_backtrace(&mut info.frames) as u32;
            write_crash_info(fd, &info);
        }

        // Re-raise the signal with the default handler to produce a core dump.
        libc::signal(signum, libc::SIG_DFL);
        libc::raise(signum);
    }
}

/// Rewrite the pre-opened marker without allocating or resolving a path.
///
/// # Safety
/// `fd` must be a live writable descriptor owned for the process lifetime.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn write_crash_info(fd: libc::c_int, info: &CrashInfo) {
    let _ = libc::lseek(fd, 0, libc::SEEK_SET);
    let _ = libc::ftruncate(fd, 0);
    let bytes = std::slice::from_raw_parts(
        info as *const CrashInfo as *const u8,
        std::mem::size_of::<CrashInfo>(),
    );
    let mut written = 0usize;
    while written < bytes.len() {
        let count = libc::write(
            fd,
            bytes.as_ptr().add(written) as *const libc::c_void,
            bytes.len() - written,
        );
        if count <= 0 {
            break;
        }
        written = written.saturating_add(count as usize);
    }
    let _ = libc::fsync(fd);
}

/// Extract the instruction pointer from the signal context.
///
/// # Safety
/// `ctx` is the raw `ucontext_t` pointer from the signal handler.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn extract_pc(ctx: *mut libc::c_void) -> u64 {
    if ctx.is_null() {
        return 0;
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // macOS ARM64: ucontext_t is not in libc, define minimal types.
        #[repr(C)]
        struct Arm64ThreadState {
            regs: [u64; 29],
            fp: u64,
            lr: u64,
            sp: u64,
            pc: u64,
            cpsr: u32,
            _pad: u32,
        }
        #[repr(C)]
        struct MachMcontext {
            _es: [u8; 16],
            _ss: Arm64ThreadState,
        }
        #[repr(C)]
        struct Ucontext {
            uc_onstack: i32,
            uc_sigmask: u32,
            uc_stack: libc::stack_t,
            uc_link: *mut Ucontext,
            uc_mcsize: u64,
            uc_mcontext: *mut MachMcontext,
        }
        let uc = ctx as *const Ucontext;
        if (*uc).uc_mcontext.is_null() {
            return 0;
        }
        (*(*uc).uc_mcontext)._ss.pc
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let uc = ctx as *const libc::ucontext_t;
        let gregs = &(*uc).uc_mcontext.gregs;
        gregs[libc::REG_RIP as usize] as u64
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        let uc = ctx as *const libc::ucontext_t;
        (*uc).uc_mcontext.pc as u64
    }

    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
    )))]
    {
        let _ = ctx;
        0
    }
}

/// Capture a frame-pointer backtrace (async-signal-safe, no allocation).
///
/// Walks the frame pointer chain: `[rbp/x29] -> saved_rbp, return_addr`.
/// Returns the number of frames captured.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn capture_backtrace(frames: &mut [u64; MAX_FRAMES]) -> usize {
    let mut count = 0;
    let mut fp: usize;

    // Get the current frame pointer.
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::asm!("mov {}, rbp", out(reg) fp);
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::asm!("mov {}, x29", out(reg) fp);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return 0;
    }

    while count < MAX_FRAMES && fp != 0 && fp.is_multiple_of(std::mem::size_of::<usize>()) {
        // Frame layout: [saved_fp, return_addr]
        let saved_fp = *(fp as *const usize);
        let return_addr = *((fp + std::mem::size_of::<usize>()) as *const usize);

        if return_addr == 0 {
            break;
        }

        frames[count] = return_addr as u64;
        count += 1;

        if saved_fp <= fp {
            break; // prevent infinite loop
        }
        fp = saved_fp;
    }

    count
}
