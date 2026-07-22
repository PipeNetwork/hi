//! Cross-platform crash handler with startup crash detection.
//!
//! - **Unix**: SIGBUS/SIGSEGV via `sigaction(2)`.
//! - **Other platforms**: no-op (returns `false` from [`install`]).
//!
//! # Usage
//!
//! Call [`check_previous_crash`] first to detect crashes from the previous
//! session, then [`install`] early in `main()`, before any async runtime or
//! thread spawning.
//!
//! ```rust,no_run
//! use std::path::PathBuf;
//!
//! let crash_dir = PathBuf::from("~/.hi/crash");
//!
//! if let Some(report) = hi_crash_handler::check_previous_crash(&crash_dir) {
//!     eprintln!("hi crashed during your last session: {}", report.signal_name);
//!     eprintln!("  Report: {}", report.report_path.display());
//! }
//!
//! hi_crash_handler::install(hi_crash_handler::CrashHandlerConfig {
//!     app_version: env!("CARGO_PKG_VERSION").to_string(),
//!     crash_dir: crash_dir.clone(),
//! });
//! ```

use std::path::{Path, PathBuf};

const MAX_FRAMES: usize = 64;

/// Configuration for the crash handler.
pub struct CrashHandlerConfig {
    /// Application version string.
    pub app_version: String,
    /// Directory where crash dumps are written. Created if it does not exist.
    pub crash_dir: PathBuf,
}

/// Information about a crash from the previous session.
#[derive(Debug)]
pub struct CrashReport {
    /// Human-readable signal name (e.g. "SIGSEGV (Segmentation fault)").
    pub signal_name: &'static str,
    /// The `si_code` from `siginfo_t`.
    pub si_code: i32,
    /// Unix timestamp of the crash.
    pub timestamp: u64,
    /// Application version at crash time.
    pub app_version: String,
    /// Path to the saved human-readable crash report.
    pub report_path: PathBuf,
}

/// Install the crash handler for SIGBUS and SIGSEGV.
///
/// Must be called early in `main()`, before any async runtime or thread
/// spawning. Creates `crash_dir` if it does not exist.
///
/// Returns `true` if the handler was installed successfully.
/// On unsupported platforms, this is a no-op that returns `false`.
pub fn install(config: CrashHandlerConfig) -> bool {
    #[cfg(unix)]
    {
        handler::install(&config.crash_dir, &config.app_version)
    }
    #[cfg(not(unix))]
    {
        let _ = config;
        false
    }
}

/// Check for a crash from the previous session.
///
/// Reads `last-crash.bin` from `crash_dir`. If present, parses it into a
/// [`CrashReport`], writes a human-readable report file, and removes the
/// binary crash marker. Returns `None` if no previous crash is found.
///
/// Must be called before [`install`] — `install` opens `last-crash.bin`
/// with `O_TRUNC`, which would erase the previous crash marker.
pub fn check_previous_crash(crash_dir: &Path) -> Option<CrashReport> {
    #[cfg(unix)]
    {
        handler::check_previous_crash(crash_dir)
    }
    #[cfg(not(unix))]
    {
        let _ = crash_dir;
        None
    }
}

#[cfg(unix)]
mod handler;

#[cfg(unix)]
pub use handler::CrashInfo;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_previous_crash_returns_none_when_no_file() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(check_previous_crash(tmp.path()).is_none());
    }
}
