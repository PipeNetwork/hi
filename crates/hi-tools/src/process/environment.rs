//! Child-process environment sanitization and isolated Cargo cache selection.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tokio::process::Command;

pub(super) fn strip_inherited_secrets(command: &mut Command) {
    for var in hi_secrets::SECRET_ENV_NAMES {
        command.env_remove(var);
    }
    for (name, _) in std::env::vars_os() {
        if sensitive_environment_name(&name) || is_supervisor_env(&name) {
            command.env_remove(&name);
        }
    }
    command.env_remove("HI_CRASH_DIR");
}

fn is_supervisor_env(name: &OsStr) -> bool {
    name.to_string_lossy().starts_with("HI_SENTINEL_")
}

impl super::ProcessRunner {
    #[allow(clippy::unused_self)]
    pub fn detached_descendants_preserved(&self) -> bool {
        super::execution::detached_descendants_preserved()
    }
}

/// Cargo needs a writable registry/cache, while the sandbox intentionally
/// protects the user's shared `~/.cargo` from dependency-cache poisoning.
/// Isolate a cache by canonical workspace identity. Existing project-local
/// `.cargo-home` directories remain supported, but new projects do not gain
/// untracked cache trees.
pub(super) fn workspace_cargo_home(
    root: &Path,
    policy: crate::sandbox::SandboxPolicy,
) -> Option<PathBuf> {
    if matches!(
        policy,
        crate::sandbox::SandboxPolicy::Off | crate::sandbox::SandboxPolicy::ReadOnly
    ) {
        return None;
    }
    if let Some(configured) = std::env::var_os("CARGO_HOME").filter(|value| !value.is_empty()) {
        let configured = PathBuf::from(configured);
        let configured = if configured.is_absolute() {
            configured
        } else {
            root.join(configured)
        };
        if configured.starts_with(root) {
            return Some(configured);
        }
    }
    let legacy = root.join(".cargo-home");
    if legacy.is_dir() {
        return Some(legacy);
    }
    // Test fixtures and genuinely ephemeral projects should clean their cache
    // up with the workspace instead of leaving one hashed directory per run.
    let temp = std::env::temp_dir()
        .canonicalize()
        .unwrap_or_else(|_| std::env::temp_dir());
    if root.starts_with(&temp)
        || root.starts_with("/tmp")
        || root.starts_with("/private/tmp")
        || root.starts_with("/var/tmp")
        || root.starts_with("/private/var/tmp")
    {
        return Some(root.join(".hi/state/cargo-home"));
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|value| !value.is_empty())
                .map(|home| PathBuf::from(home).join(".cache"))
        })
        .unwrap_or_else(std::env::temp_dir)
        .join("hi/cargo");
    let digest = Sha256::digest(root.as_os_str().as_encoded_bytes());
    let digest = format!("{:x}", digest);
    Some(base.join(&digest[..24]))
}

pub(super) fn sensitive_environment_name(name: &OsStr) -> bool {
    let name = name.to_string_lossy().to_ascii_uppercase();
    [
        "API_KEY",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "AUTH_COOKIE",
        "SESSION_COOKIE",
    ]
    .iter()
    .any(|marker| name.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn supervisor_env_names_are_classified() {
        assert!(is_supervisor_env(OsStr::new("HI_SENTINEL_SUPERVISED")));
        assert!(is_supervisor_env(OsStr::new("HI_SENTINEL_INSTANCE")));
        assert!(!is_supervisor_env(OsStr::new("HI_SANDBOX")));
        assert!(!is_supervisor_env(OsStr::new("HI_CRASH_DIR")));
    }

    #[test]
    fn sensitive_environment_names_are_removed_conservatively() {
        assert!(sensitive_environment_name(OsStr::new("GITHUB_TOKEN")));
        assert!(sensitive_environment_name(OsStr::new(
            "AWS_SECRET_ACCESS_KEY"
        )));
        assert!(sensitive_environment_name(OsStr::new("DATABASE_PASSWORD")));
        assert!(!sensitive_environment_name(OsStr::new("PATH")));
        assert!(!sensitive_environment_name(OsStr::new("RUSTUP_HOME")));
    }

    struct EnvRestore {
        keys: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (key, previous) in self.keys.drain(..) {
                unsafe {
                    match previous {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn tool_children_do_not_inherit_sentinel_env() {
        let _lock = ENV_LOCK.lock().unwrap();
        let keys = [
            "HI_SENTINEL_SUPERVISED",
            "HI_SENTINEL_INSTANCE",
            "HI_CRASH_DIR",
        ];
        let restore = EnvRestore {
            keys: keys
                .iter()
                .map(|key| (*key, std::env::var_os(key)))
                .collect(),
        };
        unsafe {
            std::env::set_var("HI_SENTINEL_SUPERVISED", "1");
            std::env::set_var("HI_SENTINEL_INSTANCE", "nested-token");
            std::env::set_var("HI_CRASH_DIR", "/tmp/hi-crash-should-not-leak");
        }
        let dir = tempfile::tempdir().unwrap();
        let runner = super::super::ProcessRunner::new(dir.path()).unwrap();
        let run = runner
            .run_program(
                "sh",
                [
                    "-c",
                    "printf %s \"$HI_SENTINEL_SUPERVISED|$HI_SENTINEL_INSTANCE|$HI_CRASH_DIR\"",
                ],
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        drop(restore);
        assert_eq!(run.status, crate::ToolStatus::Succeeded);
        assert_eq!(
            run.outcome.stdout_summary, "||",
            "tool children must not inherit HI_SENTINEL_* or HI_CRASH_DIR"
        );
    }
}
