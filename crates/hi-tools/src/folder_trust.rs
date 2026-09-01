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
//! 3. No repo-local code-exec configs present → trusted (nothing to gate).
//! 4. Key unrecordable (over-broad root like `$HOME`) → untrusted.
//! 5. Interactive TTY → prompt the user (y/N).
//! 6. Otherwise (headless) → untrusted.
//!
//! Trust state is persisted in `~/.hi/trusted_folders.toml` (or the explicit
//! `HI_TRUST_STORE` path). All frontends use this store; a second independent
//! trust database would make `/trust on` disagree with MCP/hook admission.

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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
    if !i.repo_configs_present {
        return TrustOutcome::Trusted;
    }
    // Never turn an inability to persist a narrowly-scoped grant into an
    // implicit grant. In particular, running from $HOME or / must not allow a
    // planted .mcp.json or hook tree to execute without consent.
    if !i.key_recordable {
        return TrustOutcome::Untrusted;
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
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "false" | "no" | ""
        ),
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

/// Whether repo-local code-exec configs are present (hooks or MCP servers).
fn repo_configs_present(cwd: &Path) -> bool {
    cwd.join(".hi/hooks").is_dir() || mcp_server_configs_present(cwd)
}

fn mcp_server_configs_present(cwd: &Path) -> bool {
    // Claude/Cursor's project-root config can launch stdio commands and expand
    // environment variables into HTTP headers just like `.hi/mcp/*.json`.
    // Treat it as repo-local code execution for the same trust decision.
    if cwd.join(".mcp.json").is_file() {
        return true;
    }
    let dir = cwd.join(".hi").join("mcp");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .path()
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    })
}

/// An over-broad root that the store refuses to record: `$HOME`, filesystem
/// root, or non-absolute path.
fn is_unsafe_trust_root(key: &Path) -> bool {
    if !key.is_absolute() {
        return true;
    }
    // `parent() == None` is the portable root test: it catches `/` on Unix,
    // drive roots such as `C:\\`, and UNC share roots on Windows. Recording any
    // of those as an ancestor grant would effectively disable folder trust for
    // the whole filesystem/share.
    if key.parent().is_none() {
        return true;
    }
    if let Some(profile) = user_profile_dir() {
        let profile = profile
            .canonicalize()
            .unwrap_or_else(|_| profile.to_path_buf());
        if paths_equal_for_platform(key, &profile) {
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
            return current
                .canonicalize()
                .unwrap_or_else(|_| current.to_path_buf());
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
    /// Absent when the platform has no trustworthy user-profile directory.
    /// Reads then return no grants and mutations fail closed.
    path: Option<PathBuf>,
}

impl TrustStore {
    /// Load the trust store from `~/.hi/trusted_folders.toml`.
    pub fn load() -> Self {
        let path = trust_store_path();
        let trusted = path
            .as_deref()
            .map(|path| load_trusted_paths(path, should_import_legacy(path)))
            .unwrap_or_default();
        Self { trusted, path }
    }

    /// Whether `key` or any ancestor is in the trusted set.
    pub fn is_trusted(&self, key: &Path) -> bool {
        self.trusted.iter().any(|t| key.starts_with(t))
    }

    /// Add `key` to the trusted set and persist.
    pub fn grant(&mut self, key: &Path) -> std::io::Result<()> {
        if is_unsafe_trust_root(key) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "refusing to trust over-broad workspace root {}",
                    key.display()
                ),
            ));
        }

        let path = self.path.as_deref().ok_or_else(profile_unavailable_error)?;
        let _lock = TrustStoreLock::acquire(path)?;
        // The object may have been loaded well before this mutation. Reload
        // only after taking the inter-process lock so a stale writer cannot
        // restore an authorization another process has just revoked.
        let mut trusted = load_trusted_paths(path, should_import_legacy(path));
        if !trusted.iter().any(|trusted_key| trusted_key == key) {
            trusted.push(key.to_path_buf());
        }
        persist_trusted_paths(path, &trusted)?;
        self.trusted = trusted;
        Ok(())
    }

    /// Remove `key` (and any descendants) from the trusted set and persist.
    pub fn revoke(&mut self, key: &Path) -> std::io::Result<bool> {
        let path = self.path.as_deref().ok_or_else(profile_unavailable_error)?;
        let _lock = TrustStoreLock::acquire(path)?;
        // As with grant, make the decision from the latest serialized state,
        // not from this potentially stale in-memory snapshot.
        let mut trusted = load_trusted_paths(path, should_import_legacy(path));
        let before = trusted.len();
        trusted.retain(|trusted_key| !trusted_key.starts_with(key));
        let changed = trusted.len() != before;
        if changed {
            persist_trusted_paths(path, &trusted)?;
        }
        self.trusted = trusted;
        Ok(changed)
    }
}

/// A separate, persistent advisory-lock file serializes trust-store writers.
/// The file may remain after a crash, but the OS releases the advisory lock
/// with the process/file descriptor, so no stale-lock recovery is needed.
#[derive(Debug)]
struct TrustStoreLock {
    _file: std::fs::File,
}

impl TrustStoreLock {
    fn acquire(store_path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = store_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let path = lock_path(store_path);
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // Ensure a pre-planted final-component symlink/reparse point is
            // opened as the reparse point itself, then rejected below.
            const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
            options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
        }

        let file = options.open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "trust-store lock is not a regular file",
            ));
        }
        file.lock()?;
        Ok(Self { _file: file })
    }
}

fn lock_path(store_path: &Path) -> PathBuf {
    let mut path = store_path.as_os_str().to_os_string();
    path.push(".lock");
    PathBuf::from(path)
}

fn should_import_legacy(path: &Path) -> bool {
    std::env::var_os("HI_TRUST_STORE").is_none()
        && default_trust_store_path().as_deref() == Some(path)
}

fn load_trusted_paths(path: &Path, import_legacy: bool) -> Vec<PathBuf> {
    let content = match read_regular_file(path) {
        Ok(content) => Some(content),
        // Before trust handling was centralized, hi-agent used a newline file
        // in XDG config. Import it only when the canonical store does not yet
        // exist. Other read failures (including symlinks) fail closed.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && import_legacy => {
            legacy_agent_trust_store_path().and_then(|legacy| read_regular_file(&legacy).ok())
        }
        Err(_) => None,
    };

    content
        .as_deref()
        .map(parse_trusted_paths)
        .unwrap_or_default()
        .into_iter()
        .filter(|path| !is_unsafe_trust_root(path))
        .collect()
}

/// Read without following a final-component symlink. A symlinked trust file
/// must neither inject grants nor redirect a later overwrite to another file.
fn read_regular_file(path: &Path) -> std::io::Result<String> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }

    let mut input = options.open(path)?;
    let metadata = input.metadata()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "trust store is not a regular file",
        ));
    }
    let mut content = String::new();
    input.read_to_string(&mut content)?;
    Ok(content)
}

fn persist_trusted_paths(path: &Path, trusted: &[PathBuf]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = TrustStoreFile {
        trusted: trusted
            .iter()
            .map(|trusted_path| trusted_path.to_string_lossy().to_string())
            .collect(),
    };
    let content = toml::to_string_pretty(&file).map_err(std::io::Error::other)?;

    // Trust grants are authorization state. Replace them atomically with a
    // private regular file rather than following a pre-planted destination
    // symlink or briefly exposing a truncated/partially-written store.
    let (temp, mut output) = create_private_temp(path)?;
    let result = (|| {
        output.write_all(content.as_bytes())?;
        output.sync_all()?;
        drop(output);
        replace_file(&temp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

fn create_private_temp(path: &Path) -> std::io::Result<(PathBuf, std::fs::File)> {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    // A crashed process may leave a temp file behind and its PID can later be
    // reused. Skip occupied names instead of letting that stale file wedge the
    // next authorization update.
    for _ in 0..128 {
        let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let temp = path.with_extension(format!("tmp-{}-{suffix}", std::process::id()));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        }
        match options.open(&temp) {
            Ok(file) => return Ok((temp, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a unique trust-store temp file",
    ))
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x0000_0001;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x0000_0008;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(
            existing_file_name: *const u16,
            new_file_name: *const u16,
            flags: u32,
        ) -> i32;
    }

    fn wide_path(path: &Path) -> std::io::Result<Vec<u16>> {
        let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
        if wide.contains(&0) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "path contains an interior NUL",
            ));
        }
        wide.push(0);
        Ok(wide)
    }

    let source = wide_path(source)?;
    let destination = wide_path(destination)?;
    // SAFETY: both vectors are NUL-terminated and remain alive for the call.
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn profile_unavailable_error() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "cannot resolve a trusted user profile directory for folder-trust state",
    )
}

/// Resolve the native user profile without ever falling back to the current
/// working directory, which may itself be the untrusted repository.
fn user_profile_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    let profile = std::env::var_os("USERPROFILE")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            let drive = std::env::var_os("HOMEDRIVE").filter(|value| !value.is_empty())?;
            let path = std::env::var_os("HOMEPATH").filter(|value| !value.is_empty())?;
            let mut combined = drive;
            combined.push(path);
            Some(PathBuf::from(combined))
        });
    #[cfg(not(windows))]
    let profile = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);

    // A relative profile would resolve inside the untrusted current working
    // directory and recreate the authorization-store vulnerability.
    profile.filter(|path| path.is_absolute())
}

fn paths_equal_for_platform(left: &Path, right: &Path) -> bool {
    #[cfg(windows)]
    {
        left.to_string_lossy()
            .eq_ignore_ascii_case(&right.to_string_lossy())
    }
    #[cfg(not(windows))]
    {
        left == right
    }
}

fn default_trust_store_path() -> Option<PathBuf> {
    user_profile_dir().map(|profile| profile.join(".hi/trusted_folders.toml"))
}

/// Path to the trust store file, if the platform profile can be resolved.
fn trust_store_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("HI_TRUST_STORE").filter(|path| !path.is_empty()) {
        let path = PathBuf::from(path);
        return path.is_absolute().then_some(path);
    }
    default_trust_store_path()
}

fn legacy_agent_trust_store_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| user_profile_dir().map(|home| home.join(".config")))?;
    Some(base.join("hi").join("trusted-workspaces.txt"))
}

fn parse_trusted_paths(content: &str) -> Vec<PathBuf> {
    if let Ok(file) = toml::from_str::<TrustStoreFile>(content) {
        return file.trusted.into_iter().map(PathBuf::from).collect();
    }
    // Legacy hi-agent and explicit HI_TRUST_STORE files were one path per line.
    content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .collect()
}

/// Canonical trust-store location, for user-facing status output. `None`
/// means the native user profile was unavailable and trust fails closed.
pub fn trust_store_file() -> Option<PathBuf> {
    trust_store_path()
}

/// Whether the canonical store grants this workspace (including an explicitly
/// trusted ancestor). This is a non-interactive query and never prompts.
pub fn folder_trust_granted(cwd: &Path) -> bool {
    let key = workspace_key(cwd);
    !is_unsafe_trust_root(&key) && TrustStore::load().is_trusted(&key)
}

/// Resolve a persisted trust grant for repository configuration that can
/// redirect prompts, repository contents, or credentials to a remote
/// endpoint. Unlike the ordinary code-exec gate, this intentionally does not
/// honor the local-build feature shortcut: remote data routing always needs a
/// durable, explicit grant.
pub fn resolve_sensitive_config_trust(cwd: &Path) -> TrustOutcome {
    let key = workspace_key(cwd);
    if !is_unsafe_trust_root(&key) && TrustStore::load().is_trusted(&key) {
        return TrustOutcome::Trusted;
    }
    if is_unsafe_trust_root(&key) || !is_interactive() {
        return TrustOutcome::Untrusted;
    }
    eprintln!(
        "This workspace's hi.toml configures a remote provider route that can receive \
         prompts or repository data — trust it and persist that authorization? [y/N]"
    );
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    match std::io::stdin().read_line(&mut answer) {
        Ok(_) if matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") => {
            match grant_folder_trust(cwd) {
                Ok(()) => TrustOutcome::Trusted,
                Err(error) => {
                    eprintln!("Could not persist folder trust: {error}");
                    TrustOutcome::Untrusted
                }
            }
        }
        _ => TrustOutcome::Untrusted,
    }
}

/// Grant trust for `cwd` and persist to the store.
pub fn grant_folder_trust(cwd: &Path) -> std::io::Result<()> {
    let key = workspace_key(cwd);
    if is_unsafe_trust_root(&key) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "refusing to trust over-broad workspace root {}",
                key.display()
            ),
        ));
    }
    let mut store = TrustStore::load();
    store.grant(&key)
}

/// Revoke trust for `cwd` and persist to the store. Returns true if any
/// entries were removed.
pub fn try_revoke_folder_trust(cwd: &Path) -> std::io::Result<bool> {
    let key = workspace_key(cwd);
    let mut store = TrustStore::load();
    store.revoke(&key)
}

/// Backwards-compatible best-effort revoke. Interactive callers that need to
/// report persistence failures should use [`try_revoke_folder_trust`].
pub fn revoke_folder_trust(cwd: &Path) -> bool {
    try_revoke_folder_trust(cwd).unwrap_or(false)
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
                "This workspace contains repo-local hooks or MCP configuration — trust it and allow repo-local code execution? [y/N]"
            );
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_ok() {
                let answer = input.trim().to_ascii_lowercase();
                if answer == "y" || answer == "yes" {
                    match grant_folder_trust(cwd) {
                        Ok(()) => TrustOutcome::Trusted,
                        Err(error) => {
                            eprintln!("Could not persist folder trust: {error}");
                            TrustOutcome::Untrusted
                        }
                    }
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
    fn unrecordable_key_with_code_exec_config_is_untrusted() {
        let inputs = DecideInputs {
            store_trusted: false,
            repo_configs_present: true,
            is_interactive: true,
            key_recordable: false,
        };
        assert_eq!(decide(true, &inputs), TrustOutcome::Untrusted);
    }

    #[test]
    fn unrecordable_key_without_code_exec_config_is_trusted() {
        let inputs = DecideInputs {
            store_trusted: false,
            repo_configs_present: false,
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
        if let Some(profile) = user_profile_dir() {
            assert!(is_unsafe_trust_root(&profile));
        }
        assert!(!is_unsafe_trust_root(Path::new(
            "/Users/someone/projects/repo"
        )));
    }

    #[cfg(windows)]
    #[test]
    fn is_unsafe_trust_root_rejects_windows_volume_and_share_roots() {
        assert!(is_unsafe_trust_root(Path::new(r"C:\")));
        assert!(is_unsafe_trust_root(Path::new(r"\\server\share\")));
        assert!(is_unsafe_trust_root(Path::new(r"\\?\C:\")));
        assert!(is_unsafe_trust_root(Path::new(r"\\?\UNC\server\share\")));
        assert!(!is_unsafe_trust_root(Path::new(r"C:\work\repo")));
    }

    #[test]
    fn overbroad_root_cannot_be_persisted_as_trusted() {
        let error = grant_folder_trust(Path::new("/"))
            .expect_err("filesystem root must never become an ancestor trust grant");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn trust_store_parser_accepts_canonical_and_legacy_formats() {
        assert_eq!(
            parse_trusted_paths("trusted = [\"/work/one\", \"/work/two\"]"),
            [PathBuf::from("/work/one"), PathBuf::from("/work/two")]
        );
        assert_eq!(
            parse_trusted_paths("/work/one\n\n/work/two\n"),
            [PathBuf::from("/work/one"), PathBuf::from("/work/two")]
        );
    }

    fn test_store(path: &Path) -> TrustStore {
        TrustStore {
            trusted: load_trusted_paths(path, false),
            path: Some(path.to_path_buf()),
        }
    }

    #[test]
    fn stale_store_mutations_reload_latest_state_under_the_lock() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("trusted.toml");
        let revoked = tmp.path().join("revoked");
        let granted = tmp.path().join("granted");
        let later = tmp.path().join("later");

        test_store(&path).grant(&revoked).unwrap();
        let mut stale_grant = test_store(&path);
        let mut revoker = test_store(&path);
        assert!(revoker.revoke(&revoked).unwrap());

        // This object's snapshot still contains `revoked`. Its grant must
        // reload the post-revoke file instead of restoring that stale grant.
        stale_grant.grant(&granted).unwrap();
        let after_stale_grant = test_store(&path);
        assert!(!after_stale_grant.is_trusted(&revoked));
        assert!(after_stale_grant.is_trusted(&granted));

        let mut stale_revoke = test_store(&path);
        let mut granter = test_store(&path);
        granter.grant(&later).unwrap();

        // Conversely, the stale revoke must preserve a grant committed after
        // it loaded its snapshot.
        assert!(stale_revoke.revoke(&granted).unwrap());
        let after_stale_revoke = test_store(&path);
        assert!(!after_stale_revoke.is_trusted(&granted));
        assert!(after_stale_revoke.is_trusted(&later));
    }

    #[test]
    fn advisory_lock_serializes_independent_openers() {
        use std::sync::mpsc;
        use std::time::Duration;

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("trusted.toml");
        let held = TrustStoreLock::acquire(&path).unwrap();
        let (attempting_tx, attempting_rx) = mpsc::sync_channel(0);
        let (acquired_tx, acquired_rx) = mpsc::sync_channel(0);
        let thread_path = path.clone();
        let thread = std::thread::spawn(move || {
            attempting_tx.send(()).unwrap();
            let _lock = TrustStoreLock::acquire(&thread_path).unwrap();
            acquired_tx.send(()).unwrap();
        });

        attempting_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(
            acquired_rx
                .recv_timeout(Duration::from_millis(100))
                .is_err()
        );
        drop(held);
        acquired_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        thread.join().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn lock_acquisition_rejects_a_symlink_without_touching_its_target() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("trusted.toml");
        let target = tmp.path().join("unrelated");
        std::fs::write(&target, "keep me").unwrap();
        symlink(&target, lock_path(&path)).unwrap();

        TrustStoreLock::acquire(&path).expect_err("symlinked lock must fail closed");
        assert_eq!(std::fs::read_to_string(target).unwrap(), "keep me");
        assert!(!path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn persisting_trust_replaces_symlink_without_touching_its_target() {
        use std::os::unix::fs::{MetadataExt, symlink};

        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("unrelated");
        let path = tmp.path().join("trusted.toml");
        std::fs::write(&target, "keep me").unwrap();
        symlink(&target, &path).unwrap();
        let _lock = TrustStoreLock::acquire(&path).unwrap();
        persist_trusted_paths(&path, &[PathBuf::from("/work/repo")]).unwrap();

        assert_eq!(std::fs::read_to_string(target).unwrap(), "keep me");
        assert!(
            !std::fs::symlink_metadata(&path)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        assert!(
            std::fs::read_to_string(path)
                .unwrap()
                .contains("/work/repo")
        );
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

    #[test]
    fn mcp_json_counts_as_repo_local_code_exec() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!mcp_server_configs_present(tmp.path()));
        std::fs::write(tmp.path().join(".mcp.json"), r#"{"mcpServers":{}}"#).unwrap();
        assert!(mcp_server_configs_present(tmp.path()));
        std::fs::remove_file(tmp.path().join(".mcp.json")).unwrap();
        assert!(!mcp_server_configs_present(tmp.path()));
        let dir = tmp.path().join(".hi/mcp");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!mcp_server_configs_present(tmp.path()));
        std::fs::write(dir.join("echo.json"), r#"{"command":"true"}"#).unwrap();
        assert!(mcp_server_configs_present(tmp.path()));
        assert!(repo_configs_present(tmp.path()));
    }
}
