//! Bearer-token supply for providers.
//!
//! An API key never changes, so most providers hold one for the life of the
//! process. OAuth credentials don't work that way: they expire on a timer
//! (xAI's are six hours) and a long agent session outlives them. Threading the
//! credential through this trait lets a provider re-read it per request and ask
//! for a fresh one when the endpoint rejects the current token, without the
//! provider knowing anything about OAuth.

use async_trait::async_trait;

/// Supplies the bearer token for provider requests.
///
/// [`StaticToken`] covers the API-key case. Implementations backed by expiring
/// credentials override [`refresh`](TokenSource::refresh) so the provider can
/// recover mid-session instead of failing the turn.
#[async_trait]
pub trait TokenSource: Send + Sync {
    /// The token to send on the next request.
    async fn token(&self) -> String;

    /// Called after the endpoint rejects the current token. Return `true` if a
    /// *different* token is now available and the request is worth retrying.
    ///
    /// The provider calls this at most once per request, so an implementation
    /// that keeps returning `true` without actually changing the token cannot
    /// spin the request loop.
    async fn refresh(&self) -> bool {
        false
    }

    /// Persist a newly issued credential (x402 credit token). Return `true`
    /// when later [`token`](Self::token) calls will observe `token`.
    async fn store(&self, token: String) -> bool {
        let _ = token;
        false
    }
}

/// A fixed credential — an API key, or a keyless local server's placeholder.
pub struct StaticToken(pub String);

#[async_trait]
impl TokenSource for StaticToken {
    async fn token(&self) -> String {
        self.0.clone()
    }
}

/// A bearer token that can be replaced mid-session (x402 credit grants).
///
/// When `persist_as` is set, [`store`](TokenSource::store) also writes
/// `auth.json` under that provider id. Pairing keys live under `pipenetwork`;
/// x402 credit tokens use `pipenetwork-x402`.
pub struct PersistableToken {
    current: tokio::sync::Mutex<String>,
    persist_as: Option<&'static str>,
}

impl PersistableToken {
    pub fn in_memory(initial: impl Into<String>) -> Self {
        Self {
            current: tokio::sync::Mutex::new(initial.into()),
            persist_as: None,
        }
    }

    pub fn persisting(provider_id: &'static str, initial: impl Into<String>) -> Self {
        Self {
            current: tokio::sync::Mutex::new(initial.into()),
            persist_as: Some(provider_id),
        }
    }
}

#[async_trait]
impl TokenSource for PersistableToken {
    async fn token(&self) -> String {
        self.current.lock().await.clone()
    }

    async fn store(&self, token: String) -> bool {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return false;
        }
        *self.current.lock().await = trimmed.to_string();
        if let Some(provider_id) = self.persist_as {
            let stored = crate::auth_store::StoredToken {
                access: trimmed.to_string(),
                refresh: String::new(),
                expires: u64::MAX / 2,
            };
            if let Err(error) = crate::auth_store::save(provider_id, &stored) {
                tracing::warn!(
                    target: "hi::x402",
                    error = %error,
                    "failed to persist x402 credit token"
                );
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn static_token_returns_its_key_and_never_refreshes() {
        let source = StaticToken("sk-test".to_string());
        assert_eq!(source.token().await, "sk-test");
        assert!(
            !source.refresh().await,
            "an API key has nothing to refresh to; claiming otherwise would make \
             the provider retry a request that cannot succeed"
        );
    }

    #[tokio::test]
    async fn persistable_token_store_replaces_the_next_read() {
        let source = PersistableToken::in_memory("");
        assert_eq!(source.token().await, "");
        assert!(source.store("x402_live".to_string()).await);
        assert_eq!(source.token().await, "x402_live");
    }
}
