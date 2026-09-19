use argon2::password_hash::rand_core::OsRng;
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::{Algorithm, Argon2, Params, Version};
use std::sync::OnceLock;

/// Argon2 parameters used for both hashing and verification.
///
/// The crate default (19 MiB, 2 passes) is a deliberate brute-force cost, but it
/// also makes every login attempt expensive for *us*. Keeping the cost explicit
/// in one place lets the login path bound how many of these run at once.
fn argon2() -> Argon2<'static> {
    Argon2::new(Algorithm::Argon2id, Version::V0x13, Params::default())
}

/// Hash a plaintext password with Argon2id and a random salt.
pub fn hash_password(password: &str) -> Result<String, argon2::password_hash::Error> {
    let salt = SaltString::generate(&mut OsRng);
    let hash = argon2().hash_password(password.as_bytes(), &salt)?;
    Ok(hash.to_string())
}

/// A real hash of a throwaway password, computed once on first use.
///
/// Verifying against this costs the same as a genuine check, so a login attempt
/// for a username that does not exist takes about as long as one for a username
/// that does. Without it, `LOGIN` doubles as a user-enumeration oracle: unknown
/// users fail instantly while known ones burn a full Argon2 verify.
static DUMMY_HASH: OnceLock<String> = OnceLock::new();

fn dummy_hash() -> &'static str {
    DUMMY_HASH.get_or_init(|| {
        hash_password("timing-equalisation-placeholder")
            .expect("hashing the placeholder password must succeed")
    })
}

/// Verify a plaintext password against a stored Argon2 PHC string.
pub fn verify_password(password: &str, stored_hash: &str) -> bool {
    let Ok(parsed) = PasswordHash::new(stored_hash) else {
        return false;
    };
    argon2()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok()
}

/// Verify `password` against `stored`, or against a throwaway hash when the user
/// is unknown, so both outcomes cost roughly the same wall-clock time.
///
/// Always returns `false` for `None`, which is the only useful answer — it exists
/// for its side effect of spending the verification time.
pub fn verify_password_or_dummy(password: &str, stored: Option<&str>) -> bool {
    match stored {
        Some(hash) => verify_password(password, hash),
        None => {
            let _ = verify_password(password, dummy_hash());
            false
        }
    }
}

/// A 256-bit random token, hex-encoded.
///
/// Used for one-shot bridge tickets so a browser can authenticate without
/// putting the password in the websocket URL, where it would land in proxy and
/// access logs.
pub fn random_token() -> String {
    use argon2::password_hash::rand_core::RngCore;
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_and_verify_roundtrip() {
        let hash = hash_password("hunter2").unwrap();
        assert!(verify_password("hunter2", &hash));
        assert!(!verify_password("wrong", &hash));
    }

    #[test]
    fn tokens_are_unique_and_hex() {
        let a = random_token();
        let b = random_token();
        assert_eq!(a.len(), 64);
        assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }

    #[test]
    fn hashes_are_unique_per_salt() {
        let a = hash_password("same").unwrap();
        let b = hash_password("same").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn unknown_user_verification_always_fails() {
        // Spends a real verification and still reports failure for both an
        // empty and a plausible password.
        assert!(!verify_password_or_dummy("", None));
        assert!(!verify_password_or_dummy("hunter2", None));
    }

    #[test]
    fn known_user_verification_matches_verify_password() {
        let hash = hash_password("hunter2").unwrap();
        assert!(verify_password_or_dummy("hunter2", Some(&hash)));
        assert!(!verify_password_or_dummy("wrong", Some(&hash)));
    }
}
