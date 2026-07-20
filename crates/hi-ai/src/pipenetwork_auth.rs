//! Pipe Network (pipenetwork) browser pairing for the hi CLI.
//!
//! Unlike xAI's OAuth device flow, pipenetwork does not hand hi a refreshable
//! access token. The browser signs the user in, the control plane mints a
//! project API key (`pk_live_…`), and hi stores that key under the
//! `pipenetwork` provider in `auth.json`. Subsequent runs treat it like any
//! other API key — no refresh cycle.

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::auth_store::{self, StoredToken};

/// Default control-plane origin. Override with `PIPENETWORK_API_BASE` when
/// pointing at a staging or local deployment (no trailing `/v1`).
pub const DEFAULT_API_BASE: &str = "https://api.pipenetwork.ai";

/// The provider key credentials are stored under in `auth.json`.
pub const PROVIDER_ID: &str = "pipenetwork";

/// Placeholder refresh field — project API keys do not rotate. Kept so the
/// shared [`StoredToken`] shape stays uniform with OAuth providers.
const NO_REFRESH: &str = "";

/// Far-future expiry so the key is never treated as an expiring OAuth token.
const NEVER_EXPIRES: u64 = u64::MAX / 2;

#[derive(Debug, Deserialize)]
pub struct PairingIssue {
    pub pairing_id: String,
    pub pairing_secret: String,
    pub user_code: String,
    pub verification_url: String,
    #[serde(default)]
    pub poll_interval_ms: Option<u64>,
}

impl PairingIssue {
    pub fn poll_interval_secs(&self) -> u64 {
        let ms = self.poll_interval_ms.unwrap_or(1_500).max(250);
        // Floor at 1s so we don't hammer a misconfigured server.
        (ms.div_ceil(1000)).max(1)
    }

    pub fn validate(&self) -> Result<()> {
        if self.pairing_id.trim().is_empty() || self.pairing_secret.trim().is_empty() {
            bail!("pipenetwork pairing response omitted pairing credentials");
        }
        if self.user_code.trim().is_empty() {
            bail!("pipenetwork pairing response omitted user_code");
        }
        if !self.verification_url.starts_with("https://")
            && !self.verification_url.starts_with("http://127.0.0.1")
            && !self.verification_url.starts_with("http://localhost")
        {
            bail!(
                "pipenetwork verification URL must be https (or local http): {}",
                self.verification_url
            );
        }
        Ok(())
    }

    pub fn url(&self) -> &str {
        &self.verification_url
    }
}

#[derive(Debug, Deserialize)]
struct PollEnvelope {
    status: String,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    project_name: Option<String>,
}

/// Resolve the control-plane base URL (no `/v1` suffix).
pub fn api_base() -> String {
    std::env::var("PIPENETWORK_API_BASE")
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_API_BASE.to_string())
}

/// Start a pairing session. Returns the user-facing code + verification URL.
pub async fn request_pairing() -> Result<PairingIssue> {
    let url = format!("{}/v1/platform/desktop/hi/pairings", api_base());
    let response = crate::http::agent_http_client()
        .post(&url)
        .header("Accept", "application/json")
        .send()
        .await
        .context("pipenetwork pairing request failed")?;
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("pipenetwork pairing failed (HTTP {status}): {text}");
    }
    let issue: PairingIssue = serde_json::from_str(&text)
        .with_context(|| format!("pipenetwork pairing returned unexpected body: {text}"))?;
    issue.validate()?;
    Ok(issue)
}

/// Poll until the browser approves (or the pairing expires/cancels).
pub async fn poll_for_key(issue: &PairingIssue) -> Result<StoredToken> {
    let url = format!("{}/v1/platform/desktop/hi/pairings/poll", api_base());
    let mut interval = std::time::Duration::from_secs(issue.poll_interval_secs());
    // Cap the wait so a forgotten browser tab doesn't hang forever (~20 min).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20 * 60);

    loop {
        if std::time::Instant::now() > deadline {
            bail!("pipenetwork pairing timed out — run /login pipenetwork again");
        }
        tokio::time::sleep(interval).await;

        let response = crate::http::agent_http_client()
            .post(&url)
            .header("Accept", "application/json")
            .json(&serde_json::json!({
                "pairing_id": issue.pairing_id,
                "pairing_secret": issue.pairing_secret,
            }))
            .send()
            .await
            .context("pipenetwork pairing poll failed")?;
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("pipenetwork pairing poll failed (HTTP {status}): {text}");
        }
        let envelope: PollEnvelope = serde_json::from_str(&text)
            .with_context(|| format!("pipenetwork poll returned unexpected body: {text}"))?;

        match envelope.status.as_str() {
            "pending" => {
                // Stay on the server-suggested cadence; slow down slightly if we
                // keep getting pending for a long stretch.
                interval = std::time::Duration::from_secs(issue.poll_interval_secs());
            }
            "approved" => {
                let api_key = envelope
                    .api_key
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| anyhow::anyhow!("approved pairing omitted api_key"))?;
                let _ = (envelope.project_id, envelope.project_name);
                return Ok(StoredToken {
                    access: api_key,
                    refresh: NO_REFRESH.to_string(),
                    expires: NEVER_EXPIRES,
                });
            }
            "expired" => bail!("the pipenetwork pairing expired — run /login pipenetwork again"),
            "cancelled" => bail!("the pipenetwork pairing was cancelled"),
            other => bail!("pipenetwork pairing failed with status '{other}'"),
        }
    }
}

/// Run the full browser pairing and persist the project API key.
pub async fn login() -> Result<()> {
    let issue = request_pairing().await?;

    println!("\n  Open this URL and sign in to connect hi:\n");
    println!("    \x1b[4m{}\x1b[0m\n", issue.url());
    println!("  Code: \x1b[1m{}\x1b[0m", issue.user_code);
    println!(
        "\n  \x1b[2mApprove in your browser. Waiting for approval…\x1b[0m"
    );

    let token = poll_for_key(&issue).await?;
    auth_store::save(PROVIDER_ID, &token).context("saving the pipenetwork credential")?;
    println!("\n\x1b[32m  ✓ signed in to pipenetwork\x1b[0m");
    println!(
        "  \x1b[2mAPI key stored in ~/.config/hi/auth.json. \
         Use it with `hi --provider pipenetwork` or `/provider pipenetwork`.\x1b[0m\n"
    );
    Ok(())
}

/// Discard the stored credential. Returns whether one was present.
pub fn logout_quiet() -> Result<bool> {
    let had_credential = auth_store::load(PROVIDER_ID).is_some();
    auth_store::delete(PROVIDER_ID)?;
    Ok(had_credential)
}

/// Discard the stored credential and print a short status line.
pub fn logout() -> Result<()> {
    if logout_quiet()? {
        println!("signed out of pipenetwork");
    } else {
        println!("not signed in to pipenetwork");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_issue(url: &str) -> PairingIssue {
        PairingIssue {
            pairing_id: "hpair_test".into(),
            pairing_secret: "secret".into(),
            user_code: "ABCD-EFGH".into(),
            verification_url: url.into(),
            poll_interval_ms: Some(1_500),
        }
    }

    #[test]
    fn https_verification_url_is_accepted() {
        assert!(sample_issue("https://pipenetwork.ai/signup?hi_pairing=x").validate().is_ok());
    }

    #[test]
    fn local_http_verification_url_is_accepted() {
        assert!(sample_issue("http://localhost:3000/signup?hi_pairing=x").validate().is_ok());
        assert!(sample_issue("http://127.0.0.1:3000/signup?hi_pairing=x").validate().is_ok());
    }

    #[test]
    fn non_https_remote_url_is_rejected() {
        assert!(sample_issue("http://evil.example/signup").validate().is_err());
    }

    #[test]
    fn poll_interval_floors_at_one_second() {
        let mut issue = sample_issue("https://example.com/x");
        issue.poll_interval_ms = Some(100);
        assert_eq!(issue.poll_interval_secs(), 1);
        issue.poll_interval_ms = Some(2_500);
        assert_eq!(issue.poll_interval_secs(), 3);
    }
}
