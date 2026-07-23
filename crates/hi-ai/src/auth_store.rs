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
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

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
    std::fs::read_to_string(&path)
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
    path: PathBuf,
}

impl AuthLock {
    fn acquire() -> Result<Self> {
        let auth = auth_path().context("could not determine config directory")?;
        let parent = auth.parent().context("auth path has no parent")?;
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating config dir {}", parent.display()))?;
        let path = auth.with_extension("json.lock");
        for _ in 0..500 {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(error) => return Err(error).context("acquiring credential lock"),
            }
        }
        anyhow::bail!("timed out acquiring credential lock")
    }
}

impl Drop for AuthLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Write via a 0600 temp file and rename, so a reader never sees a partial file
/// and the secret is never briefly world-readable (which a write-then-chmod
/// would allow).
fn write_all(all: &HashMap<String, StoredToken>) -> Result<()> {
    let path = auth_path().context("could not determine config directory")?;
    let parent = path.parent().context("auth path has no parent")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("creating config dir {}", parent.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
    }

    let json = serde_json::to_string_pretty(all).context("serializing credentials")?;
    let temp = path.with_extension("json.tmp");
    write_private(&temp, &json).with_context(|| format!("writing {}", temp.display()))?;
    std::fs::rename(&temp, &path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

#[cfg(unix)]
fn write_private(path: &std::path::Path, contents: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    // `create_new` would fail on a leftover temp file from a crashed run, so
    // truncate instead — the mode still applies when this call creates the file,
    // and set_permissions covers the pre-existing case.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    use std::os::unix::fs::PermissionsExt;
    file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    file.write_all(contents.as_bytes())
}

#[cfg(not(unix))]
fn write_private(path: &std::path::Path, contents: &str) -> std::io::Result<()> {
    std::fs::write(path, contents)
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
            save("xai", &token("access-1")).unwrap();
            assert_eq!(load("xai"), Some(token("access-1")));
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
