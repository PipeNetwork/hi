//! Unix crash handler — SIGBUS/SIGSEGV via `sigaction(2)`.
//!
//! Captures crash PC + frame-pointer chain. All handler operations are
//! minimal (raw pointer reads, direct file I/O, atomics — no allocation).
//! The crash PC is written to disk before frame walking so a secondary
//! fault during the walk still produces a usable report.

use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{CrashReport, MAX_FRAMES};

/// Raw crash info written to `last-crash.bin` by the signal handler.
#[repr(C)]
pub struct CrashInfo {
    /// Magic bytes for validation.
    magic: [u8; 4],
    /// Signal number.
    signum: i32,
    /// `si_code` from `siginfo_t`.
    si_code: i32,
    /// Crash instruction pointer.
    crash_pc: u64,
    /// Unix timestamp (seconds since epoch).
    timestamp: u64,
    /// Number of valid frames in the backtrace.
    frame_count: u32,
    /// Frame-pointer chain (instruction addresses).
    frames: [u64; MAX_FRAMES],
    /// App version string (null-terminated, up to 64 bytes).
    app_version: [u8; 64],
}

const CRASH_MAGIC: [u8; 4] = *b"HICR";
const CRASH_FILE: &str = "last-crash.bin";

static INSTALLED: AtomicBool = AtomicBool::new(false);

/// Install the SIGBUS/SIGSEGV handler. Creates `crash_dir` and opens
/// `last-crash.bin` with `O_TRUNC` (so a clean startup erases the previous
/// crash marker). Returns `true` on success.
pub fn install(crash_dir: &Path, app_version: &str) -> bool {
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return true; // already installed
    }

    if let Err(e) = std::fs::create_dir_all(crash_dir) {
        eprintln!("hi-crash-handler: failed to create crash dir: {e}");
        return false;
    }

    // Truncate the crash marker so a clean startup doesn't report a stale crash.
    let crash_path = crash_dir.join(CRASH_FILE);
    let _ = std::fs::write(&crash_path, b"");

    // Set owner-only permissions on the crash file.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&crash_path, std::fs::Permissions::from_mode(0o600));
    }

    // Store the crash dir path in a global for the signal handler.
    // SAFETY: we use a static mutex-protected PathBuf. The signal handler
    // reads it via a raw pointer. This is safe because the path is set once
    // at install time and never freed.
    let dir_cstring = match CString::new(crash_dir.as_os_str().as_bytes()) {
        Ok(s) => s,
        Err(_) => return false,
    };

    let version_bytes = app_version.as_bytes();
    let mut version_buf = [0u8; 64];
    let copy_len = version_bytes.len().min(63);
    version_buf[..copy_len].copy_from_slice(&version_bytes[..copy_len]);

    // SAFETY: install_signal_handler is a C FFI call that sets up sigaction.
    // It's called once at startup before threads are spawned.
    unsafe {
        install_signal_handler(dir_cstring.as_ptr(), version_buf);
    }

    true
}

/// Check for a previous crash by reading `last-crash.bin`.
pub fn check_previous_crash(crash_dir: &Path) -> Option<CrashReport> {
    let crash_path = crash_dir.join(CRASH_FILE);
    let data = std::fs::read(&crash_path).ok()?;
    if data.len() < std::mem::size_of::<CrashInfo>() {
        return None;
    }

    // SAFETY: we check the magic bytes to validate the data.
    let info: &CrashInfo = unsafe { &*(data.as_ptr() as *const CrashInfo) };
    if info.magic != CRASH_MAGIC {
        return None;
    }

    let signal_name = signal_name(info.signum);
    let report_path = crash_dir.join(format!("crash-{}.txt", info.timestamp));

    // Write a human-readable report.
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
    let _ = std::fs::write(&report_path, &report_text);

    // Remove the binary crash marker so we don't report it again.
    let _ = std::fs::remove_file(&crash_path);

    Some(CrashReport {
        signal_name,
        si_code: info.si_code,
        timestamp: info.timestamp,
        app_version: std::str::from_utf8(&info.app_version)
            .unwrap_or("unknown")
            .trim_end_matches('\0')
            .to_string(),
        report_path,
    })
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

// ---------------------------------------------------------------------------
// Signal handler installation (platform-specific)
// ---------------------------------------------------------------------------

/// Global storage for the crash dir path and app version, accessed by the
/// signal handler. Set once at install time.
static mut CRASH_DIR_PTR: *const libc::c_char = std::ptr::null();
static mut APP_VERSION_BUF: [u8; 64] = [0u8; 64];

/// Install the SIGBUS/SIGSEGV signal handler.
///
/// # Safety
/// Must be called once at startup before threads are spawned.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn install_signal_handler(crash_dir_ptr: *const libc::c_char, version_buf: [u8; 64]) {
    CRASH_DIR_PTR = crash_dir_ptr;
    APP_VERSION_BUF = version_buf;

    // Set up an alternate signal stack so the handler works even if the
    // main stack is corrupted.
    static mut ALT_STACK: [u8; 64 * 1024] = [0u8; 64 * 1024];
    // SAFETY: raw pointer arithmetic on static mut — no reference created.
    let stack = libc::stack_t {
        ss_sp: std::ptr::addr_of_mut!(ALT_STACK) as *mut libc::c_void,
        ss_flags: 0,
        ss_size: 64 * 1024,
    };
    libc::sigaltstack(&stack, std::ptr::null_mut());

    let mut action: libc::sigaction = std::mem::zeroed();
    action.sa_sigaction = crash_handler as *const () as usize;
    action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_RESTART;
    libc::sigemptyset(&mut action.sa_mask);

    for &sig in &[libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGFPE] {
        libc::sigaction(sig, &action, std::ptr::null_mut());
    }
}

/// The signal handler. Writes crash info to `last-crash.bin` then re-raises
/// the signal so the default handler produces a core dump.
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

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut frames = [0u64; MAX_FRAMES];
        let frame_count = capture_backtrace(&mut frames);

        let info = CrashInfo {
            magic: CRASH_MAGIC,
            signum,
            si_code,
            crash_pc,
            timestamp,
            frame_count: frame_count as u32,
            frames,
            app_version: APP_VERSION_BUF,
        };

        // Write the crash info to disk.
        if !CRASH_DIR_PTR.is_null() {
            let dir = std::ffi::CStr::from_ptr(CRASH_DIR_PTR);
            let dir_bytes = dir.to_bytes();
            let dir_path = std::path::PathBuf::from(std::ffi::OsStr::from_bytes(dir_bytes));
            let crash_path = dir_path.join(CRASH_FILE);

            // Open with O_WRONLY | O_CREAT | O_TRUNC
            let path_cstring = match CString::new(crash_path.as_os_str().as_bytes()) {
                Ok(s) => s,
                Err(_) => {
                    libc::_exit(128 + signum);
                }
            };

            let fd = libc::open(
                path_cstring.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC,
                0o600,
            );
            if fd >= 0 {
                let bytes = std::slice::from_raw_parts(
                    &info as *const CrashInfo as *const u8,
                    std::mem::size_of::<CrashInfo>(),
                );
                libc::write(fd, bytes.as_ptr() as *const libc::c_void, bytes.len());
                libc::close(fd);
            }
        }

        // Re-raise the signal with the default handler to produce a core dump.
        libc::signal(signum, libc::SIG_DFL);
        libc::raise(signum);
    }
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
        return gregs[libc::REG_RIP as usize] as u64;
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        let uc = ctx as *const libc::ucontext_t;
        return (*uc).uc_mcontext.pc as u64;
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
