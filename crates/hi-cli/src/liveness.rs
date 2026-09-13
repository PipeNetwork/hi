//! Supervised heartbeat writer, incident-scoped crash dir, and panic file.

use std::path::PathBuf;

use hi_liveness::{ENV_CRASH_DIR, ENV_PANIC_FILE, ENV_SUPERVISED, HarnessState, WriterHandle};

pub fn crash_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(ENV_CRASH_DIR) {
        return PathBuf::from(dir);
    }
    std::env::var("HOME")
        .map(|home| PathBuf::from(home).join(".hi/crash"))
        .unwrap_or_else(|_| PathBuf::from(".hi/crash"))
}

pub fn install_supervised() -> Option<WriterHandle> {
    install_panic_hook();
    hi_liveness::spawn_from_env()
}

pub async fn awaiting_user<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    hi_liveness::set_state_if_installed(HarnessState::AwaitingUser);
    let out = fut.await;
    hi_liveness::set_state_if_installed(HarnessState::Idle);
    out
}

fn install_panic_hook() {
    let supervised = std::env::var(ENV_SUPERVISED).ok();
    if supervised
        .as_deref()
        .is_none_or(|v| !hi_liveness::env_flag_on(v))
    {
        return;
    }
    let Some(path) = std::env::var_os(ENV_PANIC_FILE) else {
        return;
    };
    let path = PathBuf::from(path);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = panic_text(info);
        let redacted = hi_secrets::redact_secrets(&payload);
        let _ = hi_liveness::write_private_file(&path, redacted.as_bytes());
        previous(info);
    }));
}

fn panic_text(info: &std::panic::PanicHookInfo<'_>) -> String {
    let mut text = info.to_string();
    text.truncate(64 * 1024);
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crash_dir_prefers_hi_crash_dir() {
        let _lock = crate::CWD_LOCK.lock().unwrap();
        let previous = std::env::var_os(ENV_CRASH_DIR);
        unsafe {
            std::env::set_var(ENV_CRASH_DIR, "/tmp/hi-incident-crash");
        }
        let dir = crash_dir();
        unsafe {
            match previous {
                Some(value) => std::env::set_var(ENV_CRASH_DIR, value),
                None => std::env::remove_var(ENV_CRASH_DIR),
            }
        }
        assert_eq!(dir, PathBuf::from("/tmp/hi-incident-crash"));
    }
}
