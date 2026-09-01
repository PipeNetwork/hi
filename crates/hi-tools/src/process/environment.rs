//! Child-process environment sanitization and isolated Cargo cache selection.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Environment variables which must never be inherited by model-controlled
/// processes. Everything else is retained so compilers and project-local tool
/// chains keep working.
pub(super) const SECRET_ENV_VARS: &[&str] = &[
    "HI_API_KEY",
    "HI_WEB_SEARCH_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "PIPENETWORK_API_KEY",
    "OLLAMA_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_TOKEN",
];
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
