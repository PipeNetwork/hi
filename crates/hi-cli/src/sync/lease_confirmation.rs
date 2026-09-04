use anyhow::{Context, Result, anyhow};

use super::{
    LEASE_CONFIRMATION_TIMEOUT, LEASE_CONFIRMED_FRESH_SECS, RemoteSessionSink, lock_recover,
    note_endpoint_outcome, unix_now,
};

impl RemoteSessionSink {
    /// Writer lease token for authenticated long-polls and heartbeats.
    pub fn writer_lease_token(&self) -> Option<String> {
        self.lease_token()
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn writer_lease_generation(&self) -> u64 {
        self.store
            .status(Some(&self.session_id))
            .map(|status| status.lease_generation)
            .unwrap_or_default()
    }

    pub fn writer_lease_is_lost(&self) -> bool {
        self.lease_lost.is_lost()
    }

    pub(crate) fn subscribe_writer_lease_status(
        &self,
    ) -> tokio::sync::watch::Receiver<hi_pipefs::PipeFsLeaseStatus> {
        self.lease_lost.subscribe()
    }

    /// Prove this writer still owns the server lease before admitting a
    /// native filesystem mutation. Ambiguous transport failures immediately
    /// publish uncertainty so live writers can be stopped.
    pub(crate) async fn confirm_writer_lease(&self) -> Result<()> {
        if self.lease_lost.is_lost() {
            anyhow::bail!("lease_lost: this session was taken over by another writer");
        }
        let token = self
            .lease_token()
            .ok_or_else(|| anyhow!("writer lease is unavailable"))?;
        let url = format!(
            "{}/hi/sessions/{}/heartbeat",
            self.config.base_url, self.session_id
        );
        let telemetry = lock_recover(&self.telemetry).clone();
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("x-hi-lease-token", token)
            .json(&serde_json::json!({
                "model": telemetry.model,
                "context_used_tokens": telemetry.context_used_tokens,
                "context_max_tokens": telemetry.context_max_tokens,
            }))
            .timeout(LEASE_CONFIRMATION_TIMEOUT)
            .send()
            .await;
        note_endpoint_outcome(&self.store, response.as_ref().err());
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                self.lease_lost.mark_uncertain();
                return Err(error).with_context(|| format!("confirming writer lease at {url}"));
            }
        };
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::CONFLICT {
                self.lease_lost.mark_lost();
            } else {
                self.lease_lost.mark_uncertain();
            }
            anyhow::bail!("writer lease confirmation failed: HTTP {status} {body}");
        }
        let confirmed_until = (unix_now().max(0) as u64).saturating_add(LEASE_CONFIRMED_FRESH_SECS);
        self.store
            .renew_lease_expiry(&self.session_id, confirmed_until)
            .context("recording confirmed writer lease freshness")?;
        self.lease_lost.mark_synchronously_confirmed();
        Ok(())
    }

    pub fn lease_token(&self) -> Option<String> {
        self.store.lease_token(&self.session_id).ok().flatten()
    }
}
