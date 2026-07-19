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
}

/// A fixed credential — an API key, or a keyless local server's placeholder.
pub struct StaticToken(pub String);

#[async_trait]
impl TokenSource for StaticToken {
    async fn token(&self) -> String {
        self.0.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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
    async fn static_token_is_usable_as_a_trait_object() {
        let source: Arc<dyn TokenSource> = Arc::new(StaticToken("sk-test".to_string()));
        assert_eq!(source.token().await, "sk-test");
    }
}
