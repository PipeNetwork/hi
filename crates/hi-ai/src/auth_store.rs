//! Persisted OAuth credentials, one entry per provider.
//!
//! Deliberately *not* `config.toml`. Config resolution layers a project-local
//! `./hi.toml` over the user file, so a refresh token written through the normal
//! config path could land in a repo and be committed. This file lives only in
//! the user config dir and is created 0600 before any secret reaches it.
//!
//! API keys are not stored here — they stay in `config.toml`/env, where they
//! already live. This is for credentials that expire and get rewritten.

use std::collections::HashMap;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const MAX_AUTH_BYTES: u64 = 1024 * 1024;
#[cfg(unix)]
const PRIVATE_DIR_MODE: u32 = 0o700;
#[cfg(unix)]
const PRIVATE_FILE_MODE: u32 = 0o600;
static WRITE_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// An OAuth credential for one provider.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredToken {
    pub access: String,
    pub refresh: String,
    /// Unix seconds after which `access` should be re-minted. Written with a
    /// safety margin subtracted, so a token that is "valid" here is valid long
    /// enough to finish a request rather than expiring mid-flight.
    pub expires: u64,
}

impl StoredToken {
    /// Refresh slightly before the reported expiry: a token that dies between
    /// the check and the response is indistinguishable from a revoked one.
    const REFRESH_SKEW_SECS: u64 = 5 * 60;

    /// Build from a token response's `expires_in`, applying the skew.
    pub fn expiring_in(access: String, refresh: String, expires_in_secs: u64) -> Self {
        let now = now_secs();
        Self {
            access,
            refresh,
            expires: now + expires_in_secs.saturating_sub(Self::REFRESH_SKEW_SECS),
        }
    }

    pub fn is_expired(&self) -> bool {
        now_secs() >= self.expires
    }
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `~/.config/hi/auth.json`, alongside `config.toml` and `models-cache.json`.
pub fn auth_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join("hi").join("auth.json"))
}

fn read_all() -> HashMap<String, StoredToken> {
    let Some(path) = auth_path() else {
        return HashMap::new();
    };
    read_private(&path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// The stored credential for `provider`, expired or not. Callers decide whether
/// to refresh; returning expired tokens is what makes refresh possible at all.
pub fn load(provider: &str) -> Option<StoredToken> {
    read_all().remove(provider)
}

/// Replace `provider`'s credential, preserving every other provider's entry.
pub fn save(provider: &str, token: &StoredToken) -> Result<()> {
    let _lock = AuthLock::acquire()?;
    let mut all = read_all();
    all.insert(provider.to_string(), token.clone());
    write_all(&all)
}

/// Remove `provider`'s credential (logout). Absent entries are not an error.
pub fn delete(provider: &str) -> Result<()> {
    let _lock = AuthLock::acquire()?;
    let mut all = read_all();
    if all.remove(provider).is_none() {
        return Ok(());
    }
    write_all(&all)
}

struct AuthLock {
    file: std::fs::File,
}

impl AuthLock {
    fn acquire() -> Result<Self> {
        let auth = auth_path().context("could not determine config directory")?;
        let parent = auth.parent().context("auth path has no parent")?;
        ensure_private_directory(parent)
            .with_context(|| format!("securing config dir {}", parent.display()))?;
        let path = auth.with_extension("json.lock");
        let file = open_lock_file(&path).context("opening credential lock")?;
        for _ in 0..500 {
            match file.try_lock() {
                Ok(()) => return Ok(Self { file }),
                Err(std::fs::TryLockError::WouldBlock) => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(std::fs::TryLockError::Error(error)) => {
                    return Err(error).context("acquiring credential lock");
                }
            }
        }
        anyhow::bail!("timed out acquiring credential lock")
    }
}

impl Drop for AuthLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// Write via a 0600 temp file and rename, so a reader never sees a partial file
/// and the secret is never briefly world-readable (which a write-then-chmod
/// would allow).
fn write_all(all: &HashMap<String, StoredToken>) -> Result<()> {
    let path = auth_path().context("could not determine config directory")?;
    let parent = path.parent().context("auth path has no parent")?;
    ensure_private_directory(parent)
        .with_context(|| format!("securing config dir {}", parent.display()))?;

    let json = serde_json::to_string_pretty(all).context("serializing credentials")?;
    write_private_atomic(&path, json.as_bytes())
        .with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

fn write_private_atomic(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "auth path has no parent")
    })?;
    for _ in 0..16 {
        let nonce = WRITE_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("auth");
        let temp = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options
                .mode(PRIVATE_FILE_MODE)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        }
        let mut file = match options.open(&temp) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        };
        let result = (|| {
            tighten_private_file(&file)?;
            file.write_all(contents)?;
            file.sync_all()?;
            drop(file);
            replace_file(&temp, path)
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        return result;
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a private credential temp file",
    ))
}

fn open_lock_file(path: &Path) -> std::io::Result<std::fs::File> {
    loop {
        let before = match std::fs::symlink_metadata(path) {
            Ok(metadata) => Some(metadata),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error),
        };
        if before
            .as_ref()
            .is_some_and(|metadata| metadata.file_type().is_symlink() || !metadata.is_file())
        {
            return Err(invalid_auth_file(path));
        }
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true);
        if before.is_none() {
            options.create_new(true);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options
                .mode(PRIVATE_FILE_MODE)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        }
        let file = match options.open(path) {
            Ok(file) => file,
            Err(error) if before.is_none() && error.kind() == std::io::ErrorKind::AlreadyExists => {
                continue;
            }
            Err(error) => return Err(error),
        };
        let opened = file.metadata()?;
        if !opened.is_file()
            || before
                .as_ref()
                .is_some_and(|before| !same_file(before, &opened))
        {
            return Err(invalid_auth_file(path));
        }
        tighten_private_file(&file)?;
        return Ok(file);
    }
}

#[cfg(not(windows))]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::rename(from, to)
}

#[cfg(windows)]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    match std::fs::rename(from, to) {
        Ok(()) => return Ok(()),
        Err(error) if std::fs::symlink_metadata(to).is_err() => return Err(error),
        Err(_) => {}
    }
    let parent = to.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "auth path has no parent")
    })?;
    let name = to
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("auth");
    for _ in 0..16 {
        let nonce = WRITE_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let backup = parent.join(format!(".{name}.bak-{}-{nonce}", std::process::id()));
        if std::fs::symlink_metadata(&backup).is_ok() {
            continue;
        }
        std::fs::rename(to, &backup)?;
        return match std::fs::rename(from, to) {
            Ok(()) => {
                let _ = std::fs::remove_file(backup);
                Ok(())
            }
            Err(error) => {
                let restore = std::fs::rename(&backup, to);
                if let Err(restore_error) = restore {
                    return Err(std::io::Error::other(format!(
                        "credential replacement failed ({error}); restoring the previous file also failed ({restore_error})"
                    )));
                }
                Err(error)
            }
        };
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a credential backup path",
    ))
}

fn read_private(path: &Path) -> std::io::Result<String> {
    let before = std::fs::symlink_metadata(path)?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() > MAX_AUTH_BYTES {
        return Err(invalid_auth_file(path));
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    let file = options.open(path)?;
    let opened = file.metadata()?;
    if !opened.is_file() || !same_file(&before, &opened) || opened.len() > MAX_AUTH_BYTES {
        return Err(invalid_auth_file(path));
    }
    tighten_private_file(&file)?;
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    file.take(MAX_AUTH_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_AUTH_BYTES {
        return Err(invalid_auth_file(path));
    }
    String::from_utf8(bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

fn ensure_private_directory(path: &Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(PRIVATE_DIR_MODE);
    }
    match builder.create(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    let before = std::fs::symlink_metadata(path)?;
    if before.file_type().is_symlink() || !before.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "credential directory is not a directory: {}",
                path.display()
            ),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _};
        let mut options = std::fs::OpenOptions::new();
        options
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC);
        let directory = options.open(path)?;
        let opened = directory.metadata()?;
        if before.dev() != opened.dev() || before.ino() != opened.ino() || !opened.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "credential directory changed during open: {}",
                    path.display()
                ),
            ));
        }
        directory.set_permissions(std::fs::Permissions::from_mode(PRIVATE_DIR_MODE))?;
    }
    Ok(())
}

fn tighten_private_file(file: &std::fs::File) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        file.set_permissions(std::fs::Permissions::from_mode(PRIVATE_FILE_MODE))?;
    }
    Ok(())
}

fn invalid_auth_file(path: &Path) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!(
            "credential store is not a stable regular file: {}",
            path.display()
        ),
    )
}

#[cfg(unix)]
fn same_file(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file(_left: &std::fs::Metadata, _right: &std::fs::Metadata) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Point HOME at a scratch dir so the real `~/.config/hi` is never touched.
    /// Serialized because it mutates process-wide env.
    fn with_temp_home<T>(body: impl FnOnce() -> T) -> T {
        // Crate-wide, not module-local: the models-cache tests redirect HOME as
        // well, and a lock per module would not serialize against them. These
        // tests are synchronous, so there is no runtime to block.
        let _lock = crate::ENV_HOME_LOCK.blocking_lock();

        let dir = std::env::temp_dir().join(format!("hi-auth-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let prev_home = std::env::var_os("HOME");
        let prev_xdg = std::env::var_os("XDG_CONFIG_HOME");
        unsafe {
            std::env::set_var("HOME", &dir);
            std::env::remove_var("XDG_CONFIG_HOME");
        }
        let out = body();
        unsafe {
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            if let Some(v) = prev_xdg {
                std::env::set_var("XDG_CONFIG_HOME", v);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
        out
    }

    fn token(access: &str) -> StoredToken {
        StoredToken {
            access: access.into(),
            refresh: "refresh-value".into(),
            expires: now_secs() + 3600,
        }
    }

    #[test]
    fn saves_and_loads_a_credential() {
        with_temp_home(|| {
            let stored = token("access-1");
            save("xai", &stored).unwrap();
            assert_eq!(load("xai"), Some(stored));
            assert_eq!(load("other"), None, "unrelated providers stay absent");
        });
    }

    /// Storing one provider's token must not evict another's.
    #[test]
    fn saving_one_provider_preserves_the_others() {
        with_temp_home(|| {
            save("xai", &token("xai-access")).unwrap();
            save("anthropic", &token("anthropic-access")).unwrap();
            assert_eq!(load("xai").unwrap().access, "xai-access");
            assert_eq!(load("anthropic").unwrap().access, "anthropic-access");
        });
    }

    #[test]
    fn concurrent_saves_preserve_both_providers() {
        with_temp_home(|| {
            let threads = (0..8)
                .map(|index| {
                    std::thread::spawn(move || {
                        save(
                            &format!("provider-{index}"),
                            &token(&format!("token-{index}")),
                        )
                        .unwrap();
                    })
                })
                .collect::<Vec<_>>();
            for thread in threads {
                thread.join().unwrap();
            }
            for index in 0..8 {
                assert_eq!(
                    load(&format!("provider-{index}")).unwrap().access,
                    format!("token-{index}")
                );
            }
        });
    }

    #[test]
    fn delete_removes_only_the_named_provider_and_tolerates_absence() {
        with_temp_home(|| {
            save("xai", &token("a")).unwrap();
            save("anthropic", &token("b")).unwrap();
            delete("xai").unwrap();
            assert_eq!(load("xai"), None);
            assert!(load("anthropic").is_some());
            delete("never-stored").unwrap();
        });
    }

    /// The file holds a refresh token; it must not be readable by other users.
    #[cfg(unix)]
    #[test]
    fn the_credential_file_is_not_world_readable() {
        with_temp_home(|| {
            use std::os::unix::fs::PermissionsExt;
            save("xai", &token("secret")).unwrap();
            let mode = std::fs::metadata(auth_path().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600, "auth.json must be owner-only, got {mode:o}");
        });
    }

    /// A rewrite must leave no readable leftover behind.
    #[cfg(unix)]
    #[test]
    fn rewriting_leaves_no_temp_file() {
        with_temp_home(|| {
            save("xai", &token("first")).unwrap();
            save("xai", &token("second")).unwrap();
            let temp = auth_path().unwrap().with_extension("json.tmp");
            assert!(!temp.exists(), "temp file should be renamed away");
            assert_eq!(load("xai").unwrap().access, "second");
        });
    }

    #[test]
    fn replacing_an_existing_credential_file_works() {
        with_temp_home(|| {
            save("xai", &token("first")).unwrap();
            save("xai", &token("second")).unwrap();
            assert_eq!(load("xai").unwrap().access, "second");
        });
    }

    #[test]
    fn stale_lock_file_does_not_block_writes() {
        with_temp_home(|| {
            let lock = auth_path().unwrap().with_extension("json.lock");
            std::fs::create_dir_all(lock.parent().unwrap()).unwrap();
            std::fs::write(&lock, b"left by a crashed process").unwrap();

            save("xai", &token("after-crash")).unwrap();
            assert_eq!(load("xai").unwrap().access, "after-crash");
        });
    }

    #[cfg(unix)]
    #[test]
    fn planted_temp_symlink_cannot_overwrite_its_target() {
        with_temp_home(|| {
            use std::os::unix::fs::symlink;

            let path = auth_path().unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let target = path.parent().unwrap().join("outside");
            std::fs::write(&target, b"do not overwrite").unwrap();
            symlink(&target, path.with_extension("json.tmp")).unwrap();

            save("xai", &token("secret")).unwrap();
            assert_eq!(std::fs::read(target).unwrap(), b"do not overwrite");
            assert_eq!(load("xai").unwrap().access, "secret");
        });
    }

    #[cfg(unix)]
    #[test]
    fn credential_read_rejects_symlinks_without_importing_their_contents() {
        with_temp_home(|| {
            use std::os::unix::fs::symlink;

            let path = auth_path().unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let target = path.parent().unwrap().join("outside.json");
            let mut attacker = HashMap::new();
            attacker.insert("xai".to_string(), token("injected"));
            std::fs::write(&target, serde_json::to_vec(&attacker).unwrap()).unwrap();
            symlink(&target, &path).unwrap();

            assert_eq!(load("xai"), None);
            save("xai", &token("real")).unwrap();
            assert_eq!(load("xai").unwrap().access, "real");
            let target_contents = std::fs::read_to_string(target).unwrap();
            assert!(target_contents.contains("injected"));
            assert!(!target_contents.contains("real"));
        });
    }

    #[test]
    fn expiry_applies_a_safety_margin() {
        let fresh = StoredToken::expiring_in("a".into(), "r".into(), 21_600);
        assert!(!fresh.is_expired());

        // A token whose remaining life is under the skew is already "expired",
        // so it gets replaced before it can die mid-request.
        let nearly_gone = StoredToken::expiring_in("a".into(), "r".into(), 60);
        assert!(
            nearly_gone.is_expired(),
            "a token expiring within the skew window must be refreshed early"
        );
    }

    #[test]
    fn a_missing_or_corrupt_file_reads_as_empty_rather_than_failing() {
        with_temp_home(|| {
            assert_eq!(load("xai"), None);
            let path = auth_path().unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "{ not json").unwrap();
            assert_eq!(load("xai"), None, "corrupt store should not panic");
            // And it must still be recoverable by writing a fresh credential.
            save("xai", &token("recovered")).unwrap();
            assert_eq!(load("xai").unwrap().access, "recovered");
        });
    }
}
