//! Folder trust — prompt before running tools/hooks in an untrusted workspace.
//!
//! Inspired by grok-build's `xai-grok-workspace/folder_trust` module. The trust
//! gate prevents hi from executing repo-local code (hooks, MCP servers, custom
//! tools) in a workspace the user hasn't explicitly trusted — important now that
//! `.hi/hooks/` can run arbitrary commands.
//!
//! ## Precedence (canonical — see [`decide`])
//! 1. Feature flag OFF → trusted (no gating).
//! 2. Store (self/ancestor recorded trusted) → trusted.
//! 3. Key unrecordable (over-broad root like `$HOME`) → trusted.
//! 4. No repo-local code-exec configs present → trusted (nothing to gate).
//! 5. Interactive TTY → prompt the user (y/N).
//! 6. Otherwise (headless) → untrusted.
//!
//! Trust state is persisted in `~/.hi/trusted_folders.toml`.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The pure trust outcome for a set of inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustOutcome {
    /// Repo-local code execution allowed.
    Trusted,
    /// Repo-local code execution blocked.
    Untrusted,
    /// Interactive: ask the user.
    Prompt,
}

/// Inputs to the pure [`decide`] precedence function.
#[derive(Debug, Clone, Copy)]
pub struct DecideInputs {
    pub store_trusted: bool,
    pub repo_configs_present: bool,
    pub is_interactive: bool,
    /// False when the workspace key is an over-broad root the store refuses to
    /// record — home / filesystem root / non-absolute.
    pub key_recordable: bool,
}

/// Pure trust-decision precedence. No I/O; unit-tested directly.
pub fn decide(feature_enabled: bool, i: &DecideInputs) -> TrustOutcome {
    if !feature_enabled {
        return TrustOutcome::Trusted;
    }
    if i.store_trusted {
        return TrustOutcome::Trusted;
    }
    if !i.key_recordable {
        return TrustOutcome::Trusted;
    }
    if !i.repo_configs_present {
        return TrustOutcome::Trusted;
    }
    if i.is_interactive {
        return TrustOutcome::Prompt;
    }
    TrustOutcome::Untrusted
}

/// Whether the folder-trust system is inert for this binary.
///
/// Local/dev builds (no `HI_VERSION` release stamp) auto-trust everything.
/// Folder-trust applies only to shipped, release-stamped binaries.
pub fn folder_trust_inert() -> bool {
    is_local_build()
}

/// Whether this is a local/dev build (no release version stamp).
fn is_local_build() -> bool {
    option_env!("HI_VERSION").is_none()
}

/// Resolve whether the folder-trust gate is enabled.
///
/// On a local/dev build the feature is OFF regardless of env — a self-built hi
/// auto-trusts. On a release build, `HI_FOLDER_TRUST` env var controls it
/// (default: on).
pub fn feature_enabled() -> bool {
    if is_local_build() {
        return false;
    }
    match std::env::var("HI_FOLDER_TRUST") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false" | "no" | ""),
        Err(_) => true,
    }
}

/// Gather the [`DecideInputs`] for `cwd`, keyed by `key`.
pub fn decide_inputs(cwd: &Path, key: &Path) -> DecideInputs {
    decide_inputs_with_interactive(cwd, key, is_interactive())
}

/// Like [`decide_inputs`] but with caller-supplied interactivity.
pub fn decide_inputs_with_interactive(
    cwd: &Path,
    key: &Path,
    is_interactive: bool,
) -> DecideInputs {
    DecideInputs {
        store_trusted: TrustStore::load().is_trusted(key),
        repo_configs_present: repo_configs_present(cwd),
        is_interactive,
        key_recordable: !is_unsafe_trust_root(key),
    }
}

/// Whether repo-local code-exec configs are present (`.hi/hooks/` directory).
fn repo_configs_present(cwd: &Path) -> bool {
    cwd.join(".hi/hooks").is_dir()
}

/// An over-broad root that the store refuses to record: `$HOME`, filesystem
/// root, or non-absolute path.
fn is_unsafe_trust_root(key: &Path) -> bool {
    if !key.is_absolute() {
        return true;
    }
    if key == Path::new("/") {
        return true;
    }
    if let Some(home) = std::env::var("HOME").ok() {
        if key == Path::new(&home) {
            return true;
        }
    }
    false
}

fn is_interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

/// The workspace key for trust storage — the `.git` directory's parent,
/// or `cwd` itself if not in a git repo. The result is canonicalized so
/// firmlink aliases (`/tmp` → `/private/tmp`) don't bypass trust checks.
pub fn workspace_key(cwd: &Path) -> PathBuf {
    // Walk up to find a .git directory.
    let mut current = cwd;
    loop {
        if current.join(".git").exists() {
            return current.canonicalize().unwrap_or_else(|_| current.to_path_buf());
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }
    cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf())
}

// ---------------------------------------------------------------------------
// Trust store — persisted in ~/.hi/trusted_folders.toml
// ---------------------------------------------------------------------------

/// Durable trust store: a set of trusted workspace paths.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TrustStoreFile {
    #[serde(default)]
    trusted: Vec<String>,
}

/// In-memory trust store loaded from disk.
pub struct TrustStore {
    trusted: Vec<PathBuf>,
    path: PathBuf,
}

impl TrustStore {
    /// Load the trust store from `~/.hi/trusted_folders.toml`.
    pub fn load() -> Self {
        let path = trust_store_path();
        let trusted = match std::fs::read_to_string(&path) {
            Ok(content) => {
                let file: TrustStoreFile = toml::from_str(&content).unwrap_or_default();
                file.trusted.into_iter().map(PathBuf::from).collect()
            }
            Err(_) => Vec::new(),
        };
        Self { trusted, path }
    }

    /// Whether `key` or any ancestor is in the trusted set.
    pub fn is_trusted(&self, key: &Path) -> bool {
        self.trusted.iter().any(|t| key.starts_with(t))
    }

    /// Add `key` to the trusted set and persist.
    pub fn grant(&mut self, key: &Path) -> std::io::Result<()> {
        if !self.trusted.iter().any(|t| t == key) {
            self.trusted.push(key.to_path_buf());
        }
        self.persist()
    }

    /// Remove `key` (and any descendants) from the trusted set and persist.
    pub fn revoke(&mut self, key: &Path) -> std::io::Result<bool> {
        let before = self.trusted.len();
        self.trusted.retain(|t| !t.starts_with(key));
        let changed = self.trusted.len() != before;
        if changed {
            self.persist()?;
        }
        Ok(changed)
    }

    fn persist(&self) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = TrustStoreFile {
            trusted: self.trusted.iter().map(|t| t.to_string_lossy().to_string()).collect(),
        };
        let content = toml::to_string_pretty(&file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&self.path, content)
    }
}

/// Path to the trust store file.
fn trust_store_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".hi/trusted_folders.toml")
}

/// Grant trust for `cwd` and persist to the store.
pub fn grant_folder_trust(cwd: &Path) -> std::io::Result<()> {
    let key = workspace_key(cwd);
    let mut store = TrustStore::load();
    store.grant(&key)
}

/// Revoke trust for `cwd` and persist to the store. Returns true if any
/// entries were removed.
pub fn revoke_folder_trust(cwd: &Path) -> bool {
    let key = workspace_key(cwd);
    let mut store = TrustStore::load();
    store.revoke(&key).unwrap_or(false)
}

/// Resolve trust for `cwd`: gather inputs, decide, and if `Prompt`, ask the
/// user via stderr. Returns `Trusted` or `Untrusted` (never `Prompt`).
pub fn resolve_trust(cwd: &Path) -> TrustOutcome {
    let key = workspace_key(cwd);
    let inputs = decide_inputs(cwd, &key);
    match decide(feature_enabled(), &inputs) {
        TrustOutcome::Prompt => {
            // Prompt the user via stderr.
            eprintln!(
                "This workspace contains .hi/hooks/ — trust it and allow repo-local code execution? [y/N]"
            );
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_ok() {
                let answer = input.trim().to_ascii_lowercase();
                if answer == "y" || answer == "yes" {
                    let _ = grant_folder_trust(cwd);
                    TrustOutcome::Trusted
                } else {
                    TrustOutcome::Untrusted
                }
            } else {
                TrustOutcome::Untrusted
            }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_off_trusts_everything() {
        let inputs = DecideInputs {
            store_trusted: false,
            repo_configs_present: true,
            is_interactive: true,
            key_recordable: true,
        };
        assert_eq!(decide(false, &inputs), TrustOutcome::Trusted);
    }

    #[test]
    fn store_trusted_short_circuits() {
        let inputs = DecideInputs {
            store_trusted: true,
            repo_configs_present: true,
            is_interactive: true,
            key_recordable: true,
        };
        assert_eq!(decide(true, &inputs), TrustOutcome::Trusted);
    }

    #[test]
    fn unrecordable_key_trusts() {
        let inputs = DecideInputs {
            store_trusted: false,
            repo_configs_present: true,
            is_interactive: true,
            key_recordable: false,
        };
        assert_eq!(decide(true, &inputs), TrustOutcome::Trusted);
    }

    #[test]
    fn no_configs_trusts() {
        let inputs = DecideInputs {
            store_trusted: false,
            repo_configs_present: false,
            is_interactive: true,
            key_recordable: true,
        };
        assert_eq!(decide(true, &inputs), TrustOutcome::Trusted);
    }

    #[test]
    fn interactive_with_configs_prompts() {
        let inputs = DecideInputs {
            store_trusted: false,
            repo_configs_present: true,
            is_interactive: true,
            key_recordable: true,
        };
        assert_eq!(decide(true, &inputs), TrustOutcome::Prompt);
    }

    #[test]
    fn headless_with_configs_untrusted() {
        let inputs = DecideInputs {
            store_trusted: false,
            repo_configs_present: true,
            is_interactive: false,
            key_recordable: true,
        };
        assert_eq!(decide(true, &inputs), TrustOutcome::Untrusted);
    }

    #[test]
    fn is_unsafe_trust_root_rejects_home_and_root() {
        assert!(is_unsafe_trust_root(Path::new("/")));
        assert!(is_unsafe_trust_root(Path::new("relative")));
        if let Ok(home) = std::env::var("HOME") {
            assert!(is_unsafe_trust_root(Path::new(&home)));
        }
        assert!(!is_unsafe_trust_root(Path::new("/Users/someone/projects/repo")));
    }

    #[test]
    fn workspace_key_finds_git_root() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().canonicalize().unwrap();
        let subdir = repo.join("src/nested");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(repo.join(".git"), "gitdir: /fake").unwrap();

        let key = workspace_key(&subdir);
        assert_eq!(key, repo);
    }

    #[test]
    fn workspace_key_falls_back_to_cwd_without_git() {
        let tmp = tempfile::tempdir().unwrap();
        let canon = tmp.path().canonicalize().unwrap();
        let key = workspace_key(tmp.path());
        assert_eq!(key, canon);
    }
}
