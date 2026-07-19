//! xAI (Grok) OAuth credentials.
//!
//! A grok.com subscription — SuperGrok or X Premium — can authenticate `hi`
//! instead of a metered `XAI_API_KEY` from console.x.ai. This module holds the
//! shared OAuth constants and the token refresh, plus the [`TokenSource`] that
//! lets a live session survive its access token expiring.
//!
//! Endpoint note: the resulting token is sent to `https://api.x.ai/v1` as an
//! ordinary bearer, exactly like an API key. xAI's own CLI routes subscription
//! traffic to a separate `cli-chat-proxy.grok.com` host instead, but that host
//! is gated on a Grok CLI version claim and serves a strictly smaller model
//! catalogue, so there is nothing to gain by imitating it.

use crate::token::TokenSource;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::Deserialize;

use crate::auth_store::{self, StoredToken};

/// xAI's public OAuth client id, as used by their own CLI. Public by design:
/// device-code clients cannot hold a secret.
pub const CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
pub const SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
pub const DEVICE_CODE_URL: &str = "https://auth.x.ai/oauth2/device/code";
pub const TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";

/// The provider key this module's credentials are stored under.
pub const PROVIDER_ID: &str = "xai";

/// Fallback lifetime when a token response omits `expires_in`. Observed
/// responses carry 21600 (6h), so this is only a floor for a malformed reply.
const DEFAULT_LIFETIME_SECS: u64 = 3600;

#[derive(Debug, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    /// Absent when the server does not rotate the refresh token — see
    /// [`TokenResponse::into_stored`].
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_in: Option<u64>,
}

impl TokenResponse {
    /// Convert to a storable credential.
    ///
    /// `previous_refresh` is retained when the response omits `refresh_token`:
    /// xAI only returns one when it actually rotates, so treating the field as
    /// required would discard a still-valid refresh token and silently log the
    /// user out on the next expiry.
    pub fn into_stored(self, previous_refresh: Option<&str>) -> Result<StoredToken> {
        let refresh = match (self.refresh_token, previous_refresh) {
            (Some(rotated), _) => rotated,
            (None, Some(kept)) => kept.to_string(),
            (None, None) => bail!("xAI token response omitted refresh_token"),
        };
        Ok(StoredToken::expiring_in(
            self.access_token,
            refresh,
            self.expires_in.unwrap_or(DEFAULT_LIFETIME_SECS),
        ))
    }
}

/// RFC 8628 default when the server omits `interval`.
const DEFAULT_POLL_INTERVAL_SECS: u64 = 5;
/// RFC 8628 §3.5: a `slow_down` response requires backing off by 5 seconds.
const SLOW_DOWN_INCREMENT_SECS: u64 = 5;

/// A pending device authorization: show the user the code, then poll.
#[derive(Debug, Deserialize)]
pub struct DeviceCode {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    /// Pre-filled with the user code, when the server provides it.
    #[serde(default)]
    pub verification_uri_complete: Option<String>,
    #[serde(default)]
    pub interval: Option<u64>,
    pub expires_in: u64,
}

impl DeviceCode {
    /// The URL to show. Prefers the pre-filled form so the user does not have
    /// to retype the code.
    pub fn url(&self) -> &str {
        self.verification_uri_complete
            .as_deref()
            .unwrap_or(&self.verification_uri)
    }

    /// RFC 8628 permits `interval: 0` (no minimum wait); fall back rather than
    /// treating it as invalid, and never poll faster than once a second.
    fn poll_interval_secs(&self) -> u64 {
        self.interval
            .filter(|i| *i > 0)
            .unwrap_or(DEFAULT_POLL_INTERVAL_SECS)
            .max(1)
    }

    /// Reject a non-HTTPS verification URI. The URL is printed for the user to
    /// open, so a tampered response must not be able to point them at an
    /// arbitrary scheme or a plaintext page that would capture their login.
    fn validate(&self) -> Result<()> {
        for url in [
            Some(self.verification_uri.as_str()),
            self.verification_uri_complete.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if !url.starts_with("https://") {
                bail!("xAI returned an untrusted (non-HTTPS) verification URL");
            }
        }
        Ok(())
    }
}

/// Start a device authorization. The caller shows [`DeviceCode::url`] and
/// [`DeviceCode::user_code`], then calls [`poll_for_token`].
pub async fn request_device_code() -> Result<DeviceCode> {
    let response = crate::http::agent_http_client()
        .post(DEVICE_CODE_URL)
        .header("Accept", "application/json")
        // Identify honestly rather than posing as another client.
        .form(&[
            ("client_id", CLIENT_ID),
            ("scope", SCOPE),
            ("referrer", "hi"),
        ])
        .send()
        .await
        .context("xAI device authorization request failed")?;

    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("xAI device authorization failed (HTTP {status}): {text}");
    }
    let device: DeviceCode = serde_json::from_str(&text)
        .context("xAI device authorization returned an unexpected body")?;
    device.validate()?;
    Ok(device)
}

#[derive(Debug, Deserialize)]
struct TokenError {
    error: String,
    #[serde(default)]
    error_description: Option<String>,
}

/// Poll until the user approves in their browser, or the code expires.
pub async fn poll_for_token(device: &DeviceCode) -> Result<StoredToken> {
    let mut interval = device.poll_interval_secs();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(device.expires_in);

    loop {
        // Wait first: the user cannot possibly have approved yet, and polling
        // immediately just earns a `slow_down`.
        tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
        if std::time::Instant::now() >= deadline {
            bail!("xAI device code expired before it was approved");
        }

        let response = crate::http::agent_http_client()
            .post(TOKEN_URL)
            .header("Accept", "application/json")
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("client_id", CLIENT_ID),
                ("device_code", device.device_code.as_str()),
            ])
            .send()
            .await
            .context("xAI token polling request failed")?;

        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if status.is_success() {
            return serde_json::from_str::<TokenResponse>(&text)
                .context("xAI token response was not understood")?
                // A first login has no previous refresh token to fall back on.
                .into_stored(None);
        }

        let Ok(error) = serde_json::from_str::<TokenError>(&text) else {
            bail!("xAI token polling failed (HTTP {status}): {text}");
        };
        match error.error.as_str() {
            "authorization_pending" => {}
            "slow_down" => interval += SLOW_DOWN_INCREMENT_SECS,
            "access_denied" | "authorization_denied" => {
                bail!("the xAI sign-in was denied");
            }
            "expired_token" => bail!("the xAI device code expired — run /login xai again"),
            other => {
                let detail = error.error_description.unwrap_or_default();
                bail!(
                    "xAI sign-in failed: {other}{}",
                    if detail.is_empty() {
                        String::new()
                    } else {
                        format!(" ({detail})")
                    }
                );
            }
        }
    }
}

/// Exchange a refresh token for a fresh access token.
pub async fn refresh(refresh_token: &str) -> Result<StoredToken> {
    let response = crate::http::agent_http_client()
        .post(TOKEN_URL)
        .header("Accept", "application/json")
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", CLIENT_ID),
            ("refresh_token", refresh_token),
        ])
        .send()
        .await
        .context("xAI token refresh request failed")?;

    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("xAI token refresh failed (HTTP {status}): {text}");
    }
    serde_json::from_str::<TokenResponse>(&text)
        .context("xAI token refresh returned an unexpected body")?
        .into_stored(Some(refresh_token))
}

/// Run the full device-code sign-in and persist the credential.
///
/// Prints the URL and code, then blocks until the user approves in a browser.
/// No local callback server and no browser launch — the user opens the link
/// themselves, which also makes this work over SSH.
pub async fn login() -> Result<()> {
    let device = request_device_code().await?;

    println!("\n  Open this URL and approve the sign-in:\n");
    println!("    \x1b[4m{}\x1b[0m\n", device.url());
    println!("  Code: \x1b[1m{}\x1b[0m", device.user_code);
    println!(
        "\n  \x1b[2mUse the account with your SuperGrok or X Premium subscription.\n  \
         Waiting for approval…\x1b[0m"
    );

    let token = poll_for_token(&device).await?;
    auth_store::save(PROVIDER_ID, &token).context("saving the xAI credential")?;
    println!("\n\x1b[32m  ✓ signed in to xAI\x1b[0m");
    println!(
        "  \x1b[2mCredential stored in ~/.config/hi/auth.json. \
         Use it with `hi --provider xai`.\x1b[0m\n"
    );
    Ok(())
}

/// Discard the stored credential. Returns whether one was actually present, so
/// a caller can distinguish "signed out" from "was not signed in".
pub fn logout_quiet() -> Result<bool> {
    let had_credential = auth_store::load(PROVIDER_ID).is_some();
    auth_store::delete(PROVIDER_ID)?;
    Ok(had_credential)
}

/// Discard the stored credential, reporting on stdout (line-REPL frontend).
pub fn logout() -> Result<()> {
    if logout_quiet()? {
        println!("signed out of xAI — the stored credential was removed");
    } else {
        println!("not signed in to xAI");
    }
    Ok(())
}

/// Supplies the stored xAI OAuth access token, re-minting it when it expires.
///
/// Holds the credential in memory so the common path costs no file I/O, and
/// writes through to `auth.json` on every refresh so other `hi` processes and
/// later sessions see the new token.
pub struct XaiTokenSource {
    current: tokio::sync::RwLock<StoredToken>,
}

impl XaiTokenSource {
    pub fn new(initial: StoredToken) -> Self {
        Self {
            current: tokio::sync::RwLock::new(initial),
        }
    }

    /// Load the stored credential, if the user has logged in.
    pub fn from_store() -> Option<Self> {
        auth_store::load(PROVIDER_ID).map(Self::new)
    }

    /// Re-mint the access token and persist it. Returns false if the refresh
    /// failed, which the provider treats as "nothing more to try".
    async fn renew(&self) -> bool {
        let mut current = self.current.write().await;

        // Another process may have refreshed while we waited for the lock, or
        // since this process started. Adopting its token avoids a second
        // refresh — which matters because a rotated refresh token invalidates
        // the one we hold, and the loser of that race would be logged out.
        if let Some(stored) = auth_store::load(PROVIDER_ID)
            && stored.expires > current.expires
            && !stored.is_expired()
        {
            *current = stored;
            return true;
        }

        match refresh(&current.refresh).await {
            Ok(fresh) => {
                if let Err(error) = auth_store::save(PROVIDER_ID, &fresh) {
                    // The token is still usable this session; only persistence
                    // failed, so warn rather than dropping a valid credential.
                    eprintln!(
                        "\x1b[33mwarning: could not save refreshed xAI token: {error:#}\x1b[0m"
                    );
                }
                *current = fresh;
                true
            }
            Err(error) => {
                eprintln!(
                    "\x1b[33mxAI token refresh failed: {error:#}\n  Run `/login xai` to sign in again.\x1b[0m"
                );
                false
            }
        }
    }
}

#[async_trait]
impl TokenSource for XaiTokenSource {
    async fn token(&self) -> String {
        // Refresh ahead of expiry so a request is not spent discovering it.
        let expired = self.current.read().await.is_expired();
        if expired {
            self.renew().await;
        }
        self.current.read().await.access.clone()
    }

    async fn refresh(&self) -> bool {
        // Reactive path: the endpoint rejected the token even though we thought
        // it was valid (revoked, or clock skew). Renew unconditionally.
        self.renew().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(refresh_token: Option<&str>, expires_in: Option<u64>) -> TokenResponse {
        TokenResponse {
            access_token: "new-access".into(),
            refresh_token: refresh_token.map(str::to_string),
            expires_in,
        }
    }

    /// xAI omits `refresh_token` when it does not rotate. Dropping the old one
    /// would log the user out at the next expiry.
    #[test]
    fn an_omitted_refresh_token_keeps_the_previous_one() {
        let stored = response(None, Some(21_600))
            .into_stored(Some("original-refresh"))
            .unwrap();
        assert_eq!(stored.refresh, "original-refresh");
        assert_eq!(stored.access, "new-access");
    }

    #[test]
    fn a_rotated_refresh_token_replaces_the_previous_one() {
        let stored = response(Some("rotated"), Some(21_600))
            .into_stored(Some("original-refresh"))
            .unwrap();
        assert_eq!(stored.refresh, "rotated");
    }

    #[test]
    fn a_response_with_no_refresh_token_at_all_is_an_error() {
        assert!(response(None, Some(21_600)).into_stored(None).is_err());
    }

    /// Only a floor for malformed replies; real responses carry expires_in.
    #[test]
    fn a_missing_expiry_falls_back_to_an_hour() {
        let stored = response(Some("r"), None).into_stored(None).unwrap();
        let lifetime = stored.expires - auth_store::now_secs();
        assert!(
            lifetime <= DEFAULT_LIFETIME_SECS && lifetime > 0,
            "expected a bounded fallback lifetime, got {lifetime}s"
        );
    }

    #[test]
    fn the_requested_scope_covers_both_subscription_and_api_access() {
        // grok-cli:access authenticates the subscription; api:access is what
        // makes the resulting token usable at api.x.ai.
        assert!(SCOPE.contains("grok-cli:access"));
        assert!(SCOPE.contains("api:access"));
        assert!(
            SCOPE.contains("offline_access"),
            "without offline_access no refresh token is issued and the session \
             would die at the first expiry"
        );
    }

    fn device(interval: Option<u64>, uri: &str, complete: Option<&str>) -> DeviceCode {
        DeviceCode {
            device_code: "dc".into(),
            user_code: "ABCD-EFGH".into(),
            verification_uri: uri.into(),
            verification_uri_complete: complete.map(str::to_string),
            interval,
            expires_in: 1800,
        }
    }

    /// RFC 8628 allows `interval: 0`; treating it as "poll immediately, forever"
    /// would hammer the endpoint into `slow_down` on every login.
    #[test]
    fn a_zero_or_missing_interval_falls_back_to_the_rfc_default() {
        assert_eq!(
            device(Some(0), "https://x", None).poll_interval_secs(),
            DEFAULT_POLL_INTERVAL_SECS
        );
        assert_eq!(
            device(None, "https://x", None).poll_interval_secs(),
            DEFAULT_POLL_INTERVAL_SECS
        );
        assert_eq!(device(Some(7), "https://x", None).poll_interval_secs(), 7);
    }

    /// The pre-filled URL saves the user retyping the code.
    #[test]
    fn the_prefilled_verification_url_is_preferred() {
        let d = device(
            Some(5),
            "https://accounts.x.ai/oauth2/device",
            Some("https://accounts.x.ai/oauth2/device?user_code=ABCD-EFGH"),
        );
        assert!(d.url().contains("user_code=ABCD-EFGH"));
        assert_eq!(
            device(Some(5), "https://accounts.x.ai/oauth2/device", None).url(),
            "https://accounts.x.ai/oauth2/device"
        );
    }

    /// This URL is handed to the user to open and sign in at, so a tampered
    /// response must not be able to redirect them somewhere unencrypted.
    #[test]
    fn a_non_https_verification_url_is_rejected() {
        assert!(
            device(Some(5), "https://accounts.x.ai/d", None)
                .validate()
                .is_ok()
        );
        assert!(
            device(Some(5), "http://evil.example/d", None)
                .validate()
                .is_err()
        );
        assert!(
            device(
                Some(5),
                "https://accounts.x.ai/d",
                Some("http://evil.example/d")
            )
            .validate()
            .is_err(),
            "the pre-filled URL is the one actually opened, so it must be checked too"
        );
    }

    #[tokio::test]
    async fn a_valid_token_is_returned_without_touching_the_network() {
        let source = XaiTokenSource::new(StoredToken {
            access: "live-access".into(),
            refresh: "r".into(),
            expires: auth_store::now_secs() + 3600,
        });
        assert_eq!(source.token().await, "live-access");
    }
}
