use std::collections::BTreeMap;
use std::time::Duration;

use reqwest::{Method, StatusCode, Url};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;
use uuid::Uuid;

use crate::RevisionKind;

const API_ATTEMPTS: usize = 3;

#[derive(Clone, Debug)]
pub struct PipeFsClientConfig {
    pub base_url: String,
    pub api_key: String,
    pub request_timeout: Duration,
    pub transfer_timeout: Duration,
}

impl PipeFsClientConfig {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            request_timeout: Duration::from_secs(30),
            transfer_timeout: Duration::from_secs(300),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PipeFsLease {
    pub token: String,
    pub generation: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PipeFsCapabilities {
    pub enabled: bool,
    pub archive_version: u16,
    pub transfer_modes: Vec<String>,
    pub maximum_revision_bytes: u64,
    pub maximum_workspace_bytes: u64,
    pub maximum_delta_chain: u32,
    pub transfer_expiry_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ArtifactDescriptor {
    pub blake3: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub media_type: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RestoreRevision {
    pub revision_id: Uuid,
    pub base_revision_id: Option<Uuid>,
    pub revision_type: RevisionKind,
    pub sequence: u64,
    pub artifact: ArtifactDescriptor,
    pub manifest_digest: String,
    pub logical_size_bytes: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PipeFsRemoteState {
    pub session_id: String,
    pub enabled: bool,
    pub current_head: Option<Uuid>,
    pub sequence: u64,
    pub manifest_digest: Option<String>,
    pub logical_size_bytes: u64,
    pub restore_chain: Vec<RestoreRevision>,
}

#[derive(Debug, Error)]
pub enum PipeFsError {
    #[error("authentication_error: {0}")]
    Authentication(String),
    #[error("pipefs_disabled: {0}")]
    Disabled(String),
    #[error("lease_lost: {0}")]
    LeaseLost(String),
    #[error("head_conflict: {0}")]
    Conflict(String),
    #[error("storage_error: {0}")]
    Storage(String),
    #[error("missing_revision: {0}")]
    MissingRevision(String),
    #[error("network_error: {0}")]
    Network(String),
    #[error("protocol_error: {0}")]
    Protocol(String),
    #[error("corruption_error: {0}")]
    Corruption(String),
}

#[derive(Clone)]
pub struct PipeFsClient {
    config: PipeFsClientConfig,
    base_url: Url,
    http: reqwest::Client,
}

#[derive(Clone, Debug, Deserialize)]
struct Transfer {
    transfer: String,
    method: String,
    url: String,
    #[serde(default)]
    required_headers: BTreeMap<String, String>,
    expires_at_unix: i64,
}

#[derive(Debug, Deserialize)]
struct PreparedRevision {
    revision_id: Uuid,
    upload_session_id: Uuid,
    revision_type: RevisionKind,
    transfer: Transfer,
}

#[derive(Debug, Deserialize)]
struct DownloadAuthorization {
    revision_id: Uuid,
    artifact: ArtifactDescriptor,
    transfer: Transfer,
}

#[derive(Debug, Serialize)]
struct PrepareRequest<'a> {
    expected_base_revision_id: Option<Uuid>,
    revision_type: RevisionKind,
    artifact: ArtifactDescriptor,
    manifest_digest: &'a str,
    logical_size_bytes: u64,
    idempotency_key: &'a str,
}

impl PipeFsClient {
    pub fn new(config: PipeFsClientConfig) -> Result<Self, PipeFsError> {
        let normalized = format!("{}/", config.base_url.trim_end_matches('/'));
        let base_url = Url::parse(&normalized)
            .map_err(|error| PipeFsError::Protocol(format!("invalid IPOP URL: {error}")))?;
        if !matches!(base_url.scheme(), "http" | "https") {
            return Err(PipeFsError::Protocol(
                "IPOP URL must use http or https".to_string(),
            ));
        }
        if config.api_key.trim().is_empty() {
            return Err(PipeFsError::Authentication(
                "an IPOP API key is required".to_string(),
            ));
        }
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| PipeFsError::Protocol(error.to_string()))?;
        Ok(Self {
            config,
            base_url,
            http,
        })
    }

    pub async fn capabilities(&self) -> Result<PipeFsCapabilities, PipeFsError> {
        self.api_json(Method::GET, "hi/pipefs/capabilities", None, None)
            .await
    }

    pub async fn state(&self, session_id: &str) -> Result<PipeFsRemoteState, PipeFsError> {
        self.api_json(
            Method::GET,
            &format!("hi/sessions/{session_id}/pipefs"),
            None,
            None,
        )
        .await
    }

    pub async fn set_enabled(
        &self,
        session_id: &str,
        lease: &PipeFsLease,
        enabled: bool,
    ) -> Result<PipeFsRemoteState, PipeFsError> {
        self.api_json(
            Method::PUT,
            &format!("hi/sessions/{session_id}/pipefs"),
            Some(lease),
            Some(serde_json::json!({ "enabled": enabled })),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn commit_archive(
        &self,
        session_id: &str,
        lease: &PipeFsLease,
        expected_base_revision_id: Option<Uuid>,
        revision_type: RevisionKind,
        archive: &[u8],
        archive_blake3: &str,
        manifest_digest: &str,
        logical_size_bytes: u64,
        idempotency_key: &str,
    ) -> Result<PipeFsRemoteState, PipeFsError> {
        let request = PrepareRequest {
            expected_base_revision_id,
            revision_type,
            artifact: ArtifactDescriptor {
                blake3: archive_blake3.to_string(),
                size_bytes: archive.len() as u64,
                media_type: String::new(),
            },
            manifest_digest,
            logical_size_bytes,
            idempotency_key,
        };
        let prepared: PreparedRevision = self
            .api_json(
                Method::POST,
                &format!("hi/sessions/{session_id}/pipefs/revisions"),
                Some(lease),
                Some(
                    serde_json::to_value(request)
                        .map_err(|error| PipeFsError::Protocol(error.to_string()))?,
                ),
            )
            .await?;
        if prepared.revision_type != revision_type {
            return Err(PipeFsError::Protocol(
                "server changed the prepared revision type".to_string(),
            ));
        }
        let upload = self
            .upload_transfer(session_id, lease, &prepared, archive)
            .await;
        let commit = self
            .api_json_with_timeout(
                Method::POST,
                &format!(
                    "hi/sessions/{session_id}/pipefs/revisions/{}/commit",
                    prepared.revision_id
                ),
                Some(lease),
                None,
                self.config.transfer_timeout,
            )
            .await;
        match (upload, commit) {
            (Ok(()), result) => result,
            (Err(_), Ok(state)) => Ok(state),
            (Err(upload_error), Err(PipeFsError::MissingRevision(_)))
            | (Err(upload_error), Err(PipeFsError::Storage(_))) => Err(upload_error),
            (Err(_), Err(commit_error)) => Err(commit_error),
        }
    }

    pub async fn download_revision(
        &self,
        session_id: &str,
        revision: &RestoreRevision,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, PipeFsError> {
        if revision.artifact.size_bytes > maximum_bytes {
            return Err(PipeFsError::Corruption(format!(
                "revision {} declares {} bytes, exceeding the negotiated limit of {}",
                revision.revision_id, revision.artifact.size_bytes, maximum_bytes
            )));
        }
        let authorization: DownloadAuthorization = self
            .api_json(
                Method::POST,
                &format!(
                    "hi/sessions/{session_id}/pipefs/revisions/{}/download",
                    revision.revision_id
                ),
                None,
                None,
            )
            .await?;
        if authorization.revision_id != revision.revision_id
            || authorization.artifact.blake3 != revision.artifact.blake3
            || authorization.artifact.size_bytes != revision.artifact.size_bytes
        {
            return Err(PipeFsError::Protocol(
                "download authorization does not match restore-chain metadata".to_string(),
            ));
        }
        let bytes = self
            .download_transfer(
                session_id,
                revision.revision_id,
                &authorization.transfer,
                revision.artifact.size_bytes,
            )
            .await?;
        if bytes.len() as u64 != revision.artifact.size_bytes {
            return Err(PipeFsError::Corruption(format!(
                "revision {} has size {}, expected {}",
                revision.revision_id,
                bytes.len(),
                revision.artifact.size_bytes
            )));
        }
        let actual = blake3::hash(&bytes).to_hex().to_string();
        if actual != revision.artifact.blake3 {
            return Err(PipeFsError::Corruption(format!(
                "revision {} failed BLAKE3 verification",
                revision.revision_id
            )));
        }
        Ok(bytes)
    }

    async fn upload_transfer(
        &self,
        session_id: &str,
        lease: &PipeFsLease,
        prepared: &PreparedRevision,
        bytes: &[u8],
    ) -> Result<(), PipeFsError> {
        self.ensure_transfer_fresh(&prepared.transfer)?;
        if prepared.transfer.method != "PUT" {
            return Err(PipeFsError::Protocol(
                "upload authorization did not use PUT".to_string(),
            ));
        }
        let proxy = prepared.transfer.transfer == "proxy";
        let url = self.transfer_url(&prepared.transfer, proxy)?;
        if proxy {
            let expected_suffix = format!(
                "/hi/sessions/{session_id}/pipefs/uploads/{}",
                prepared.upload_session_id
            );
            if !url.path().ends_with(&expected_suffix) {
                return Err(PipeFsError::Protocol(
                    "proxy upload URL does not match the prepared upload session".to_string(),
                ));
            }
        }
        let transfer = prepared.transfer.clone();
        let payload = bytes.to_vec();
        let response = self
            .retry_request(|| {
                let mut request = self
                    .http
                    .put(url.clone())
                    .timeout(self.config.transfer_timeout)
                    .body(payload.clone());
                for (name, value) in &transfer.required_headers {
                    request = request.header(name, value);
                }
                if proxy {
                    request = request
                        .header("x-api-key", &self.config.api_key)
                        .header("x-hi-lease-token", &lease.token);
                }
                request
            })
            .await?;
        consume_empty(response, "uploading PipeFS revision")
            .await
            .map_err(|error| self.classify(error))?;
        // The session id is intentionally part of this method's API: it makes
        // accidental cross-session transfer reuse difficult to hide in callers.
        let _ = session_id;
        Ok(())
    }

    async fn download_transfer(
        &self,
        session_id: &str,
        revision_id: Uuid,
        transfer: &Transfer,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, PipeFsError> {
        self.ensure_transfer_fresh(transfer)?;
        if transfer.method != "GET" {
            return Err(PipeFsError::Protocol(
                "download authorization did not use GET".to_string(),
            ));
        }
        let proxy = transfer.transfer == "proxy";
        let url = self.transfer_url(transfer, proxy)?;
        if proxy {
            let expected_suffix =
                format!("/hi/sessions/{session_id}/pipefs/revisions/{revision_id}/content");
            if !url.path().ends_with(&expected_suffix) {
                return Err(PipeFsError::Protocol(
                    "proxy download URL does not match the authorized revision".to_string(),
                ));
            }
        }
        let transfer = transfer.clone();
        let response = self
            .retry_request(|| {
                let mut request = self
                    .http
                    .get(url.clone())
                    .timeout(self.config.transfer_timeout);
                for (name, value) in &transfer.required_headers {
                    request = request.header(name, value);
                }
                if proxy {
                    request = request.header("x-api-key", &self.config.api_key);
                }
                request
            })
            .await?;
        let mut response = checked_response(response, "downloading PipeFS revision")
            .await
            .map_err(|error| self.classify(error))?;
        if response
            .content_length()
            .is_some_and(|length| length > maximum_bytes)
        {
            return Err(PipeFsError::Corruption(format!(
                "download response exceeds the negotiated limit of {maximum_bytes} bytes"
            )));
        }
        let capacity = usize::try_from(maximum_bytes.min(8 * 1024 * 1024)).unwrap_or_default();
        let mut bytes = Vec::with_capacity(capacity);
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| PipeFsError::Network(error.to_string()))?
        {
            let next_size = (bytes.len() as u64)
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| PipeFsError::Corruption("download size overflow".to_string()))?;
            if next_size > maximum_bytes {
                return Err(PipeFsError::Corruption(format!(
                    "download response exceeds the negotiated limit of {maximum_bytes} bytes"
                )));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }

    async fn api_json<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        lease: Option<&PipeFsLease>,
        body: Option<serde_json::Value>,
    ) -> Result<T, PipeFsError> {
        self.api_json_with_timeout(method, path, lease, body, self.config.request_timeout)
            .await
    }

    /// Use the transfer-scale deadline for operations that make storage bytes
    /// durable.  Revision finalization can stream/verify a large object and
    /// commit PostgreSQL metadata, so applying the small control-plane timeout
    /// here would leave a successfully uploaded revision stuck in Pending.
    async fn api_json_with_timeout<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        lease: Option<&PipeFsLease>,
        body: Option<serde_json::Value>,
        timeout: Duration,
    ) -> Result<T, PipeFsError> {
        let url = self
            .base_url
            .join(path)
            .map_err(|error| PipeFsError::Protocol(error.to_string()))?;
        let response = self
            .retry_request(|| {
                let mut request = self
                    .http
                    .request(method.clone(), url.clone())
                    .header("x-api-key", &self.config.api_key)
                    .timeout(timeout);
                if let Some(lease) = lease {
                    request = request.header("x-hi-lease-token", &lease.token);
                }
                if let Some(body) = &body {
                    request = request.json(body);
                }
                request
            })
            .await?;
        let response = checked_response(response, path)
            .await
            .map_err(|error| self.classify(error))?;
        response
            .json::<T>()
            .await
            .map_err(|error| PipeFsError::Protocol(format!("invalid response for {path}: {error}")))
    }

    async fn retry_request<F>(&self, mut build: F) -> Result<reqwest::Response, PipeFsError>
    where
        F: FnMut() -> reqwest::RequestBuilder,
    {
        let mut delay = Duration::from_millis(150);
        let mut last = None;
        for attempt in 0..API_ATTEMPTS {
            match build().send().await {
                Ok(response) => return Ok(response),
                Err(error) if error.is_connect() || error.is_timeout() => {
                    last = Some(error);
                    if attempt + 1 < API_ATTEMPTS {
                        tokio::time::sleep(delay).await;
                        delay *= 2;
                    }
                }
                Err(error) => return Err(PipeFsError::Network(error.to_string())),
            }
        }
        Err(PipeFsError::Network(last.map_or_else(
            || "request failed".to_string(),
            |error| error.to_string(),
        )))
    }

    fn transfer_url(&self, transfer: &Transfer, proxy: bool) -> Result<Url, PipeFsError> {
        match Url::parse(&transfer.url) {
            Ok(url) if proxy => {
                if url.origin() != self.base_url.origin() {
                    return Err(PipeFsError::Protocol(
                        "proxy transfer URL changed origin".to_string(),
                    ));
                }
                Ok(url)
            }
            Ok(url) => Ok(url),
            Err(_) if proxy => self
                .base_url
                .join(transfer.url.trim_start_matches('/'))
                .map_err(|error| PipeFsError::Protocol(error.to_string())),
            Err(error) => Err(PipeFsError::Protocol(format!(
                "invalid presigned transfer URL: {error}"
            ))),
        }
    }

    fn ensure_transfer_fresh(&self, transfer: &Transfer) -> Result<(), PipeFsError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        if transfer.expires_at_unix <= now {
            return Err(PipeFsError::Network(
                "transfer authorization expired before use".to_string(),
            ));
        }
        if !matches!(transfer.transfer.as_str(), "proxy" | "presigned") {
            return Err(PipeFsError::Protocol(format!(
                "unsupported transfer mode {:?}",
                transfer.transfer
            )));
        }
        Ok(())
    }

    fn classify(&self, error: HttpFailure) -> PipeFsError {
        let detail = format!("{}: {}", error.status, error.body);
        match error.status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => PipeFsError::Authentication(detail),
            StatusCode::CONFLICT if error.body.contains("lease_lost") => {
                PipeFsError::LeaseLost(detail)
            }
            StatusCode::CONFLICT => PipeFsError::Conflict(detail),
            StatusCode::NOT_FOUND => PipeFsError::MissingRevision(detail),
            StatusCode::SERVICE_UNAVAILABLE if error.body.contains("pipefs_disabled") => {
                PipeFsError::Disabled(detail)
            }
            StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::BAD_GATEWAY
            | StatusCode::GATEWAY_TIMEOUT => PipeFsError::Storage(detail),
            _ => PipeFsError::Protocol(detail),
        }
    }
}

#[derive(Debug)]
struct HttpFailure {
    status: StatusCode,
    body: String,
}

async fn checked_response(
    response: reqwest::Response,
    _operation: &str,
) -> Result<reqwest::Response, HttpFailure> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(HttpFailure { status, body })
}

async fn consume_empty(response: reqwest::Response, operation: &str) -> Result<(), HttpFailure> {
    checked_response(response, operation).await.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_credentials_without_a_safe_endpoint() {
        let mut config = PipeFsClientConfig::new("file:///tmp/api", "secret");
        assert!(PipeFsClient::new(config.clone()).is_err());
        config.base_url = "https://example.test/v1".into();
        config.api_key.clear();
        assert!(PipeFsClient::new(config).is_err());
    }
}
