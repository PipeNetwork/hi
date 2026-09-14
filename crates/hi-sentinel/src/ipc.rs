//! Diagnose/repair requests live next to the heartbeat so macOS without
//! `XDG_RUNTIME_DIR` still reaches the supervisor.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use hi_liveness::ENV_HEARTBEAT;

use crate::fsutil;

pub const REQUEST_DIAGNOSE: &str = "request-diagnose";
pub const REQUEST_DIAGNOSE_DONE: &str = "request-diagnose.done";
pub const REQUEST_REPAIR: &str = "request-repair";
pub const REQUEST_REPAIR_DONE: &str = "request-repair.done";

/// Directory that already holds `heartbeat.json` for this child, if supervised.
pub fn runtime_dir_from_env() -> Option<PathBuf> {
    let heartbeat = PathBuf::from(std::env::var_os(ENV_HEARTBEAT)?);
    heartbeat.parent().map(Path::to_path_buf)
}

pub fn write_request(runtime: &Path, name: &str, done: &str) -> io::Result<()> {
    fsutil::mkdir_0700(runtime)?;
    let _ = fs::remove_file(runtime.join(done));
    fsutil::write_0600(&runtime.join(name), b"{}\n")
}

pub fn take_request(runtime: &Path, name: &str) -> bool {
    fs::remove_file(runtime.join(name)).is_ok()
}

pub fn write_done(runtime: &Path, name: &str, body: &str) -> io::Result<()> {
    fsutil::mkdir_0700(runtime)?;
    fsutil::write_0600(&runtime.join(name), body.as_bytes())
}

pub fn read_done(runtime: &Path, name: &str) -> Option<String> {
    fs::read_to_string(runtime.join(name)).ok()
}

pub fn wait_done(runtime: &Path, name: &str, timeout: Duration) -> io::Result<String> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(body) = read_done(runtime, name) {
            return Ok(body);
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("{name} was not acknowledged"),
            ));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths;

    #[test]
    fn runtime_dir_falls_back_without_xdg_runtime_dir() {
        let _lock = crate::ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let previous_runtime = std::env::var_os("XDG_RUNTIME_DIR");
        let previous_state = std::env::var_os("XDG_STATE_HOME");
        unsafe {
            std::env::remove_var("XDG_RUNTIME_DIR");
            std::env::set_var("XDG_STATE_HOME", &state);
            std::env::remove_var(crate::ENV_STATE_DIR);
        }
        let runtime = paths::runtime_dir(&paths::state_dir(), "ipc-test");
        unsafe {
            match previous_runtime {
                Some(value) => std::env::set_var("XDG_RUNTIME_DIR", value),
                None => std::env::remove_var("XDG_RUNTIME_DIR"),
            }
            match previous_state {
                Some(value) => std::env::set_var("XDG_STATE_HOME", value),
                None => std::env::remove_var("XDG_STATE_HOME"),
            }
        }
        assert!(
            runtime.starts_with(state.join("hi/sentinel/run/ipc-test")),
            "expected state-dir fallback, got {}",
            runtime.display()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&runtime).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700);
        }
    }

    #[test]
    fn request_and_done_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        write_request(dir.path(), REQUEST_DIAGNOSE, REQUEST_DIAGNOSE_DONE).unwrap();
        assert!(take_request(dir.path(), REQUEST_DIAGNOSE));
        assert!(!take_request(dir.path(), REQUEST_DIAGNOSE));
        write_done(dir.path(), REQUEST_DIAGNOSE_DONE, "/tmp/incident\n").unwrap();
        assert_eq!(
            read_done(dir.path(), REQUEST_DIAGNOSE_DONE).as_deref(),
            Some("/tmp/incident\n")
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(dir.path().join(REQUEST_DIAGNOSE_DONE))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }
    }
}
