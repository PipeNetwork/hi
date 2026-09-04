use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use reqwest::{Method, StatusCode, Url, header};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tokio_util::io::ReaderStream;
use uuid::Uuid;

use crate::RevisionKind;

mod validation;
use validation::validate_remote_state;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipeFsLease {
    pub token: String,
    pub generation: u64,
}

/// Opaque, non-secret identity for the authenticated IPOP authority that owns
/// a local PipeFS cache.
///
/// The scope deliberately combines the normalized URL origin with a
/// fingerprint of the credential. Session IDs are caller-selected and are not
/// globally unique across IPOP deployments or billing accounts, so they cannot
/// safely name recovery data on their own.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipeFsCacheScope(String);

impl PipeFsCacheScope {
    fn from_authority(base_url: &Url, api_key: &str) -> Self {
        let mut credential =
            blake3::Hasher::new_derive_key("hi.pipefs.cache-credential-fingerprint.v1");
        credential.update(api_key.as_bytes());
        let credential = credential.finalize();

        let mut authority = blake3::Hasher::new_derive_key("hi.pipefs.cache-authority-scope.v1");
        authority.update(base_url.origin().ascii_serialization().as_bytes());
        authority.update(&[0]);
        authority.update(credential.as_bytes());
        Self(format!("v1-{}", authority.finalize().to_hex()))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn directory_name(&self) -> String {
        format!("authority-{}", self.0)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct PipeFsCapabilities {
    pub enabled: bool,
    /// Independently-drainable rollout controls. `None` means an older server
    /// whose aggregate `enabled` bit governs that operation.
    #[serde(default)]
    pub enrollment_enabled: Option<bool>,
    #[serde(default)]
    pub writes_enabled: Option<bool>,
    #[serde(default)]
    pub restore_enabled: Option<bool>,
    #[serde(default)]
    pub garbage_collection_enabled: Option<bool>,
    pub archive_version: u16,
    pub transfer_modes: Vec<String>,
    pub maximum_revision_bytes: u64,
    pub maximum_workspace_bytes: u64,
    pub maximum_delta_chain: u32,
    pub transfer_expiry_seconds: u64,
    /// Versioned optional features advertised by newer servers.
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub writer_protocols: Vec<u16>,
    #[serde(default)]
    pub writer_protocol: Option<u16>,
}

impl PipeFsCapabilities {
    pub fn enrollment_available(&self) -> bool {
        self.enrollment_enabled.unwrap_or(self.enabled)
            && self.writes_enabled.unwrap_or(self.enabled)
    }

    pub fn writes_available(&self) -> bool {
        self.writes_enabled.unwrap_or(self.enabled)
    }

    pub fn restore_available(&self) -> bool {
        self.restore_enabled.unwrap_or(self.enabled)
    }

    pub fn causal_commit_available(&self) -> bool {
        self.capabilities
            .iter()
            .any(|value| value == crate::CAUSAL_COMMIT_CAPABILITY)
            && self.supports_writer_protocol(crate::CAUSAL_WRITER_PROTOCOL)
    }

    pub fn supports_writer_protocol(&self, protocol: u16) -> bool {
        self.writer_protocol == Some(protocol) || self.writer_protocols.contains(&protocol)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArtifactDescriptor {
    pub blake3: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub media_type: String,
}

/// Immutable revision uploaded for a protocol-2 causal commit but not yet
/// installed as the workspace head.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UploadedRevision {
    pub revision_id: Uuid,
    pub artifact: ArtifactDescriptor,
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
    pub(crate) config: PipeFsClientConfig,
    pub(crate) base_url: Url,
    cache_scope: PipeFsCacheScope,
    pub(crate) http: reqwest::Client,
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
        if !safe_transport_url(&base_url) {
            return Err(PipeFsError::Protocol(
                "IPOP URL must use HTTPS unless it is a loopback development endpoint".to_string(),
            ));
        }
        if !base_url.username().is_empty()
            || base_url.password().is_some()
            || base_url.query().is_some()
            || base_url.fragment().is_some()
        {
            return Err(PipeFsError::Protocol(
                "IPOP URL cannot contain credentials, a query, or a fragment".to_string(),
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
        let cache_scope = PipeFsCacheScope::from_authority(&base_url, &config.api_key);
        Ok(Self {
            config,
            base_url,
            cache_scope,
            http,
        })
    }

    /// Return the opaque cache identity for this authenticated IPOP authority.
    /// It is safe to persist; neither the origin nor credential can be recovered
    /// from it.
    pub fn cache_scope(&self) -> PipeFsCacheScope {
        self.cache_scope.clone()
    }

    pub async fn capabilities(&self) -> Result<PipeFsCapabilities, PipeFsError> {
        self.api_json(Method::GET, "hi/pipefs/capabilities", None, None)
            .await
    }

    pub async fn state(&self, session_id: &str) -> Result<PipeFsRemoteState, PipeFsError> {
        let state = self
            .api_json(
                Method::GET,
                &format!("hi/sessions/{session_id}/pipefs"),
                None,
                None,
            )
            .await?;
        validate_remote_state(session_id, state)
    }

    pub async fn set_enabled(
        &self,
        session_id: &str,
        lease: &PipeFsLease,
        enabled: bool,
    ) -> Result<PipeFsRemoteState, PipeFsError> {
        let state = self
            .api_json(
                Method::PUT,
                &format!("hi/sessions/{session_id}/pipefs"),
                Some(lease),
                Some(serde_json::json!({ "enabled": enabled })),
            )
            .await?;
        validate_remote_state(session_id, state)
    }

    /// Atomically publish a workspace operation and its transcript records.
    /// Transport retries are safe because the server deduplicates by the
    /// operation ID embedded in `request`.
    pub async fn causal_commit(
        &self,
        session_id: &str,
        lease: &PipeFsLease,
        request: &crate::CausalCommitRequest,
    ) -> Result<crate::CausalCommitReceipt, PipeFsError> {
        request.validate(lease)?;
        let body = serde_json::to_value(request)
            .map_err(|error| PipeFsError::Protocol(error.to_string()))?;
        let receipt: crate::CausalCommitReceipt = self
            .api_json_with_timeout(
                Method::POST,
                &format!(
                    "hi/sessions/{session_id}/pipefs/operations/{}/commit",
                    request.operation.operation_id
                ),
                Some(lease),
                Some(body),
                self.config.transfer_timeout,
            )
            .await?;
        // The workspace layer repeats this validation with its persisted
        // previous cursor. This zero-baseline check still rejects malformed
        // acknowledgements before they can leave the transport boundary.
        receipt.validate_for_request(request, 0)?;
        Ok(receipt)
    }

    pub async fn acknowledge_operation_intent(
        &self,
        session_id: &str,
        lease: &PipeFsLease,
        request: &crate::CausalIntentRequest,
    ) -> Result<crate::CausalIntentReceipt, PipeFsError> {
        request.validate(lease)?;
        let receipt: crate::CausalIntentReceipt = self
            .api_json_with_timeout(
                Method::POST,
                &format!(
                    "hi/sessions/{session_id}/pipefs/operations/{}/intent",
                    request.operation_id
                ),
                Some(lease),
                Some(serde_json::to_value(request).map_err(|error| {
                    PipeFsError::Protocol(format!("encoding operation intent: {error}"))
                })?),
                self.config.transfer_timeout,
            )
            .await?;
        receipt.validate(request, lease)?;
        Ok(receipt)
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
        let commit: Result<PipeFsRemoteState, PipeFsError> = self
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
            (Ok(()), result) => result.and_then(|state| validate_remote_state(session_id, state)),
            (Err(_), Ok(state)) => validate_remote_state(session_id, state),
            (Err(upload_error), Err(PipeFsError::MissingRevision(_)))
            | (Err(upload_error), Err(PipeFsError::Storage(_))) => Err(upload_error),
            (Err(_), Err(commit_error)) => Err(commit_error),
        }
    }

    /// Commit a staged archive by streaming it from disk for every upload
    /// attempt. This keeps retry memory bounded by the HTTP transport buffer.
    #[allow(clippy::too_many_arguments)]
    pub async fn commit_archive_file(
        &self,
        session_id: &str,
        lease: &PipeFsLease,
        expected_base_revision_id: Option<Uuid>,
        revision_type: RevisionKind,
        archive_path: &Path,
        archive_size_bytes: u64,
        archive_blake3: &str,
        manifest_digest: &str,
        logical_size_bytes: u64,
        idempotency_key: &str,
    ) -> Result<PipeFsRemoteState, PipeFsError> {
        let (prepared, upload) = self
            .prepare_and_upload_archive_file(
                session_id,
                lease,
                expected_base_revision_id,
                revision_type,
                archive_path,
                archive_size_bytes,
                archive_blake3,
                manifest_digest,
                logical_size_bytes,
                idempotency_key,
            )
            .await?;
        // Commit is deliberately attempted even if the transfer returned an
        // ambiguous network error: the immutable object may already exist.
        let commit: Result<PipeFsRemoteState, PipeFsError> = self
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
            (Ok(()), result) => result.and_then(|state| validate_remote_state(session_id, state)),
            (Err(_), Ok(state)) => validate_remote_state(session_id, state),
            (Err(upload_error), Err(PipeFsError::MissingRevision(_)))
            | (Err(upload_error), Err(PipeFsError::Storage(_))) => Err(upload_error),
            (Err(_), Err(commit_error)) => Err(commit_error),
        }
    }

    /// Upload an immutable revision without moving the workspace head. Only a
    /// subsequent causal operation commit may publish this revision.
    #[allow(clippy::too_many_arguments)]
    pub async fn upload_archive_file(
        &self,
        session_id: &str,
        lease: &PipeFsLease,
        expected_base_revision_id: Option<Uuid>,
        revision_type: RevisionKind,
        archive_path: &Path,
        archive_size_bytes: u64,
        archive_blake3: &str,
        manifest_digest: &str,
        logical_size_bytes: u64,
        idempotency_key: &str,
    ) -> Result<UploadedRevision, PipeFsError> {
        let (prepared, upload) = self
            .prepare_and_upload_archive_file(
                session_id,
                lease,
                expected_base_revision_id,
                revision_type,
                archive_path,
                archive_size_bytes,
                archive_blake3,
                manifest_digest,
                logical_size_bytes,
                idempotency_key,
            )
            .await?;
        upload?;
        Ok(UploadedRevision {
            revision_id: prepared.revision_id,
            artifact: ArtifactDescriptor {
                blake3: archive_blake3.to_owned(),
                size_bytes: archive_size_bytes,
                media_type: String::new(),
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn prepare_and_upload_archive_file(
        &self,
        session_id: &str,
        lease: &PipeFsLease,
        expected_base_revision_id: Option<Uuid>,
        revision_type: RevisionKind,
        archive_path: &Path,
        archive_size_bytes: u64,
        archive_blake3: &str,
        manifest_digest: &str,
        logical_size_bytes: u64,
        idempotency_key: &str,
    ) -> Result<(PreparedRevision, Result<(), PipeFsError>), PipeFsError> {
        let request = PrepareRequest {
            expected_base_revision_id,
            revision_type,
            artifact: ArtifactDescriptor {
                blake3: archive_blake3.to_owned(),
                size_bytes: archive_size_bytes,
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
                "server changed the prepared revision type".to_owned(),
            ));
        }
        let upload = self
            .upload_file_transfer(
                session_id,
                lease,
                &prepared,
                archive_path,
                archive_size_bytes,
            )
            .await;
        Ok((prepared, upload))
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

    /// Download and verify a revision directly into a newly-created file.
    /// Partial files are removed and transient transfer failures retry from
    /// byte zero, so a caller never observes an unverified archive.
    pub async fn download_revision_to_file(
        &self,
        session_id: &str,
        revision: &RestoreRevision,
        maximum_bytes: u64,
        destination: &Path,
    ) -> Result<(), PipeFsError> {
        if revision.artifact.size_bytes > maximum_bytes {
            return Err(PipeFsError::Corruption(format!(
                "revision {} declares {} bytes, exceeding the negotiated limit of {}",
                revision.revision_id, revision.artifact.size_bytes, maximum_bytes
            )));
        }
        if destination.exists() {
            return Err(PipeFsError::Protocol(format!(
                "download destination already exists: {}",
                destination.display()
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
        self.download_transfer_to_file(
            session_id,
            revision.revision_id,
            &authorization.transfer,
            &revision.artifact,
            maximum_bytes,
            destination,
        )
        .await
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

    async fn upload_file_transfer(
        &self,
        session_id: &str,
        lease: &PipeFsLease,
        prepared: &PreparedRevision,
        archive_path: &Path,
        archive_size_bytes: u64,
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
        let metadata = std::fs::symlink_metadata(archive_path)
            .map_err(|error| PipeFsError::Storage(error.to_string()))?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.len() != archive_size_bytes
        {
            return Err(PipeFsError::Corruption(
                "staged PipeFS archive changed before upload".to_string(),
            ));
        }

        let mut delay = Duration::from_millis(150);
        let mut last = None;
        for attempt in 0..API_ATTEMPTS {
            self.ensure_transfer_fresh(&prepared.transfer)?;
            let file = tokio::fs::File::open(archive_path)
                .await
                .map_err(|error| PipeFsError::Storage(error.to_string()))?;
            let body = reqwest::Body::wrap_stream(ReaderStream::new(file));
            let mut request = self
                .http
                .put(url.clone())
                .timeout(self.config.transfer_timeout)
                .body(body)
                .header(header::CONTENT_LENGTH, archive_size_bytes);
            for (name, value) in &prepared.transfer.required_headers {
                request = request.header(name, value);
            }
            if proxy {
                request = request
                    .header("x-api-key", &self.config.api_key)
                    .header("x-hi-lease-token", &lease.token);
            }
            match request.send().await {
                Ok(response) => {
                    return consume_empty(response, "uploading PipeFS revision")
                        .await
                        .map_err(|error| self.classify(error));
                }
                Err(error) if error.is_connect() || error.is_timeout() || error.is_body() => {
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
            || "upload failed".to_string(),
            |error| error.to_string(),
        )))
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

    async fn download_transfer_to_file(
        &self,
        session_id: &str,
        revision_id: Uuid,
        transfer: &Transfer,
        artifact: &ArtifactDescriptor,
        maximum_bytes: u64,
        destination: &Path,
    ) -> Result<(), PipeFsError> {
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
        let mut delay = Duration::from_millis(150);
        let mut last = None;
        for attempt in 0..API_ATTEMPTS {
            self.ensure_transfer_fresh(transfer)?;
            let _ = tokio::fs::remove_file(destination).await;
            match self
                .download_transfer_to_file_once(
                    &url,
                    transfer,
                    proxy,
                    artifact,
                    maximum_bytes,
                    destination,
                )
                .await
            {
                Ok(()) => return Ok(()),
                Err(error @ PipeFsError::Network(_)) if attempt + 1 < API_ATTEMPTS => {
                    last = Some(error);
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
                Err(error) => {
                    let _ = tokio::fs::remove_file(destination).await;
                    return Err(error);
                }
            }
        }
        let _ = tokio::fs::remove_file(destination).await;
        Err(last.unwrap_or_else(|| PipeFsError::Network("download failed".to_string())))
    }

    async fn download_transfer_to_file_once(
        &self,
        url: &Url,
        transfer: &Transfer,
        proxy: bool,
        artifact: &ArtifactDescriptor,
        maximum_bytes: u64,
        destination: &Path,
    ) -> Result<(), PipeFsError> {
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
        let response = request
            .send()
            .await
            .map_err(|error| PipeFsError::Network(error.to_string()))?;
        let mut response = checked_response(response, "downloading PipeFS revision")
            .await
            .map_err(|error| self.classify(error))?;
        if response
            .content_length()
            .is_some_and(|length| length > maximum_bytes || length != artifact.size_bytes)
        {
            return Err(PipeFsError::Corruption(
                "download response length does not match revision metadata".to_string(),
            ));
        }
        let mut file = create_private_download(destination)?;
        let mut size = 0_u64;
        let mut hasher = blake3::Hasher::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| PipeFsError::Network(error.to_string()))?
        {
            size = size
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| PipeFsError::Corruption("download size overflow".to_string()))?;
            if size > maximum_bytes || size > artifact.size_bytes {
                return Err(PipeFsError::Corruption(format!(
                    "download response exceeds the negotiated limit of {maximum_bytes} bytes"
                )));
            }
            hasher.update(&chunk);
            file.write_all(&chunk)
                .await
                .map_err(|error| PipeFsError::Storage(error.to_string()))?;
        }
        if size != artifact.size_bytes {
            return Err(PipeFsError::Corruption(format!(
                "download has size {size}, expected {}",
                artifact.size_bytes
            )));
        }
        if hasher.finalize().to_hex().as_str() != artifact.blake3 {
            return Err(PipeFsError::Corruption(
                "download failed BLAKE3 verification".to_string(),
            ));
        }
        file.sync_all()
            .await
            .map_err(|error| PipeFsError::Storage(error.to_string()))?;
        Ok(())
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
        let url = match Url::parse(&transfer.url) {
            Ok(url) if proxy => {
                if url.origin() != self.base_url.origin() {
                    return Err(PipeFsError::Protocol(
                        "proxy transfer URL changed origin".to_string(),
                    ));
                }
                url
            }
            Ok(url) => url,
            Err(_) if proxy => self
                .base_url
                .join(transfer.url.trim_start_matches('/'))
                .map_err(|error| PipeFsError::Protocol(error.to_string()))?,
            Err(error) => {
                return Err(PipeFsError::Protocol(format!(
                    "invalid presigned transfer URL: {error}"
                )));
            }
        };
        if !safe_transport_url(&url) {
            return Err(PipeFsError::Protocol(
                "PipeFS transfer URL must use HTTPS unless it is a loopback development endpoint"
                    .to_string(),
            ));
        }
        Ok(url)
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

fn safe_transport_url(url: &Url) -> bool {
    if url.scheme() == "https" {
        return true;
    }
    if url.scheme() != "http" {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    host.eq_ignore_ascii_case("localhost")
        || host.to_ascii_lowercase().ends_with(".localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn create_private_download(path: &Path) -> Result<tokio::fs::File, PipeFsError> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(path)
        .map_err(|error| PipeFsError::Storage(error.to_string()))?;
    Ok(tokio::fs::File::from_std(file))
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
        config.base_url = "http://example.test/v1".into();
        assert!(PipeFsClient::new(config.clone()).is_err());
        config.base_url = "http://127.0.0.1:8080/v1".into();
        assert!(PipeFsClient::new(config.clone()).is_ok());
        config.base_url = "https://example.test/v1".into();
        config.api_key.clear();
        assert!(PipeFsClient::new(config).is_err());
    }

    #[test]
    fn cache_scope_normalizes_origin_and_separates_credentials() {
        let scope = |base_url, api_key| {
            PipeFsClient::new(PipeFsClientConfig::new(base_url, api_key))
                .unwrap()
                .cache_scope()
        };

        assert_eq!(
            scope("https://EXAMPLE.test/v1", "account-a"),
            scope("https://example.test:443/another/api/path", "account-a")
        );
        assert_ne!(
            scope("https://example.test/v1", "account-a"),
            scope("https://example.test/v1", "account-b")
        );
        assert_ne!(
            scope("https://example.test/v1", "account-a"),
            scope("https://other.example.test/v1", "account-a")
        );
    }

    #[test]
    fn refuses_plaintext_non_loopback_transfer_urls() {
        let client = PipeFsClient::new(PipeFsClientConfig::new(
            "https://api.example.test/v1",
            "secret",
        ))
        .unwrap();
        let transfer = Transfer {
            transfer: "presigned".to_string(),
            method: "GET".to_string(),
            url: "http://storage.example.test/object".to_string(),
            required_headers: BTreeMap::new(),
            expires_at_unix: i64::MAX,
        };
        assert!(client.transfer_url(&transfer, false).is_err());
    }

    #[test]
    fn split_capabilities_fall_back_to_legacy_enabled_for_old_servers() {
        let old: PipeFsCapabilities = serde_json::from_value(serde_json::json!({
            "enabled": true,
            "archive_version": 1,
            "transfer_modes": ["proxy"],
            "maximum_revision_bytes": 1024,
            "maximum_workspace_bytes": 4096,
            "maximum_delta_chain": 20,
            "transfer_expiry_seconds": 60
        }))
        .unwrap();
        assert!(old.enrollment_available());
        assert!(old.writes_available());
        assert!(old.restore_available());
        assert!(!old.causal_commit_available());

        let draining: PipeFsCapabilities = serde_json::from_value(serde_json::json!({
            "enabled": true,
            "enrollment_enabled": false,
            "writes_enabled": false,
            "restore_enabled": true,
            "garbage_collection_enabled": false,
            "archive_version": 1,
            "transfer_modes": ["presigned"],
            "maximum_revision_bytes": 1024,
            "maximum_workspace_bytes": 4096,
            "maximum_delta_chain": 20,
            "transfer_expiry_seconds": 60
        }))
        .unwrap();
        assert!(!draining.enrollment_available());
        assert!(!draining.writes_available());
        assert!(draining.restore_available());

        let protocol_two: PipeFsCapabilities = serde_json::from_value(serde_json::json!({
            "enabled": true,
            "archive_version": 1,
            "transfer_modes": ["proxy"],
            "maximum_revision_bytes": 1024,
            "maximum_workspace_bytes": 4096,
            "maximum_delta_chain": 20,
            "transfer_expiry_seconds": 60,
            "capabilities": ["causal_commit_v1"],
            "writer_protocols": [1, 2]
        }))
        .unwrap();
        assert!(protocol_two.causal_commit_available());
    }
}
