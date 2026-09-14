//! XDG state / runtime locations for Sentinel.

use std::path::{Path, PathBuf};

use crate::ENV_STATE_DIR;
use crate::fsutil;

pub fn state_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(ENV_STATE_DIR) {
        return PathBuf::from(dir);
    }
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state")))
        .unwrap_or_else(|| PathBuf::from(".local/state"))
        .join("hi")
        .join("sentinel")
}

pub fn incidents_dir(state: &Path) -> PathBuf {
    state.join("incidents")
}

pub fn history_path(state: &Path) -> PathBuf {
    state.join("history.jsonl")
}

pub fn next_id_path(state: &Path) -> PathBuf {
    state.join("next-id")
}

/// Live heartbeat + crash dir for this supervisor instance.
pub fn runtime_dir(state: &Path, instance: &str) -> PathBuf {
    if let Some(runtime) = std::env::var_os("XDG_RUNTIME_DIR") {
        let dir = PathBuf::from(runtime).join("hi-sentinel").join(instance);
        if fsutil::mkdir_0700(&dir).is_ok() {
            return dir;
        }
    }
    let fallback = state.join("run").join(instance);
    if fsutil::mkdir_0700(&fallback).is_ok() {
        return fallback;
    }
    let tmp = std::env::temp_dir()
        .join(format!("hi-sentinel-{}", uid()))
        .join(instance);
    let _ = fsutil::mkdir_0700(&tmp);
    tmp
}

fn uid() -> u32 {
    #[cfg(unix)]
    {
        unsafe { libc::getuid() }
    }
    #[cfg(not(unix))]
    {
        0
    }
}

pub fn default_config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join("hi").join("config.toml"))
}

pub fn cache_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("hi")
}

/// Detached autofix worktrees. Not inside the Hi checkout (that pollutes `git status`).
pub fn worktrees_dir() -> PathBuf {
    cache_dir().join("autofix-worktrees")
}
