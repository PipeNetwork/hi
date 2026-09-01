//! Persisted x402 credit tokens, stored separately from pairing keys.

use anyhow::Context;

use crate::auth_store::{self, StoredToken};
use crate::token::PersistableToken;

/// `auth.json` key for issued `x402_…` credit tokens. Must not collide with
/// [`crate::pipenetwork_auth::PROVIDER_ID`] (`pk_live_…` pairing keys).
pub const PROVIDER_ID: &str = "pipenetwork-x402";

pub const X402_PROVIDER_ID: &str = PROVIDER_ID;

pub fn load_credit_token() -> Option<String> {
    auth_store::load(PROVIDER_ID)
        .map(|token| token.access)
        .filter(|access| access.starts_with(crate::x402::X402_CREDIT_TOKEN_PREFIX))
}

pub fn has_credit_token() -> bool {
    load_credit_token().is_some()
}

pub fn validate_keypair_file(path: &std::path::Path) -> anyhow::Result<()> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let bytes: Vec<u8> = serde_json::from_str(text.trim())
        .with_context(|| format!("{} is not a JSON byte array", path.display()))?;
    anyhow::ensure!(
        bytes.len() == 64,
        "Solana keypair {} must be a 64-byte JSON array",
        path.display()
    );
    Ok(())
}

pub fn save_credit_token(token: &str) -> anyhow::Result<()> {
    let token = token.trim();
    anyhow::ensure!(
        token.starts_with(crate::x402::X402_CREDIT_TOKEN_PREFIX),
        "refusing to store a non-x402 credential under {PROVIDER_ID}"
    );
    auth_store::save(
        PROVIDER_ID,
        &StoredToken {
            access: token.to_string(),
            refresh: String::new(),
            expires: u64::MAX / 2,
        },
    )
}

pub fn logout_quiet() -> anyhow::Result<bool> {
    let had = auth_store::load(PROVIDER_ID).is_some();
    auth_store::delete(PROVIDER_ID)?;
    Ok(had)
}

pub fn logout() -> anyhow::Result<()> {
    if logout_quiet()? {
        println!("cleared pipenetwork x402 credit token");
    } else {
        println!("no pipenetwork x402 credit token stored");
    }
    Ok(())
}

/// In-process source that starts from any stored credit token and writes new
/// grants back to `auth.json`.
pub fn credit_token_source() -> PersistableToken {
    PersistableToken::persisting(PROVIDER_ID, load_credit_token().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth_store::StoredToken;

    fn with_temp_home<T>(body: impl FnOnce() -> T) -> T {
        let _lock = crate::ENV_HOME_LOCK.blocking_lock();
        let dir = std::env::temp_dir().join(format!("hi-x402-auth-{}", std::process::id()));
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
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            if let Some(value) = prev_xdg {
                std::env::set_var("XDG_CONFIG_HOME", value);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
        out
    }

    #[test]
    fn pairing_and_x402_tokens_do_not_overwrite_each_other() {
        with_temp_home(|| {
            crate::auth_store::save(
                crate::pipenetwork_auth::PROVIDER_ID,
                &StoredToken {
                    access: "pk_live_pair".into(),
                    refresh: String::new(),
                    expires: u64::MAX / 2,
                },
            )
            .unwrap();
            save_credit_token("x402_credits").unwrap();
            assert_eq!(
                crate::auth_store::load(crate::pipenetwork_auth::PROVIDER_ID)
                    .unwrap()
                    .access,
                "pk_live_pair"
            );
            assert_eq!(load_credit_token().as_deref(), Some("x402_credits"));
        });
    }

    #[test]
    fn validate_keypair_file_requires_a_64_byte_json_array() {
        let dir = std::env::temp_dir().join(format!(
            "hi-x402-keypair-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("id.json");
        std::fs::write(&path, "[1,2,3]").unwrap();
        let error = validate_keypair_file(&path).unwrap_err().to_string();
        assert!(error.contains("64-byte"), "{error}");
        let bytes: Vec<u8> = (0..64).collect();
        std::fs::write(&path, serde_json::to_string(&bytes).unwrap()).unwrap();
        validate_keypair_file(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
