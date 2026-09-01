//! Self-update version discovery and manual update guidance.
//!
//! Remote installation is intentionally disabled until this repository defines
//! a signed update-manifest format and embeds trusted verification keys.
//!
//! # Quick start
//!
//! ```no_run
//! use hi_update::{check_for_update, UpdateConfig};
//!
//! # async fn run() {
//! let config = UpdateConfig::default();
//! let status = check_for_update(&config).await;
//! if status.update_available {
//!     println!("Update available: {} -> {}", status.current_version, status.latest_version.unwrap());
//! }
//! # }
//! ```

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const UPDATE_MANIFEST_SCHEMA: u16 = 1;
pub const MAX_UPDATE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignedUpdateManifest {
    pub schema: u16,
    pub key_id: String,
    pub version: String,
    pub target: String,
    pub asset_name: String,
    pub asset_url: String,
    pub size: u64,
    pub sha256: String,
}

impl SignedUpdateManifest {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).context("serializing update manifest")
    }

    pub fn validate(&self, expected_version: &str) -> Result<()> {
        if self.schema != UPDATE_MANIFEST_SCHEMA {
            bail!("unsupported update manifest schema {}", self.schema);
        }
        if self.version.trim_start_matches('v') != expected_version.trim_start_matches('v') {
            bail!("update manifest version does not match the release");
        }
        if self.target != platform_asset_name() {
            bail!("update manifest target does not match this platform");
        }
        if self.size == 0 || self.size > MAX_UPDATE_BYTES {
            bail!("update asset size is outside the allowed range");
        }
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!("update manifest has an invalid SHA-256 digest");
        }
        let url = reqwest::Url::parse(&self.asset_url).context("invalid update asset URL")?;
        if url.scheme() != "https" {
            bail!("update asset URL must use HTTPS");
        }
        Ok(())
    }
}

pub fn verify_signed_manifest(
    manifest: &SignedUpdateManifest,
    signature_hex: &str,
    trusted_keys: &[(&str, [u8; 32])],
) -> Result<()> {
    let result = verify_signed_manifest_inner(manifest, signature_hex, trusted_keys);
    if result.is_err() {
        hi_observability::record(hi_observability::ReliabilityEvent::UpdateSignatureFailure);
    }
    result
}

fn verify_signed_manifest_inner(
    manifest: &SignedUpdateManifest,
    signature_hex: &str,
    trusted_keys: &[(&str, [u8; 32])],
) -> Result<()> {
    let (_, key) = trusted_keys
        .iter()
        .find(|(id, _)| *id == manifest.key_id)
        .context("update manifest key is not trusted")?;
    let key = VerifyingKey::from_bytes(key).context("invalid trusted update key")?;
    let bytes = hex::decode(signature_hex.trim()).context("invalid update signature encoding")?;
    let signature = Signature::from_slice(&bytes).context("invalid update signature")?;
    key.verify(&manifest.canonical_bytes()?, &signature)
        .context("update manifest signature verification failed")
}

pub fn verify_asset_bytes(manifest: &SignedUpdateManifest, bytes: &[u8]) -> Result<()> {
    if bytes.len() as u64 != manifest.size {
        bail!("downloaded update size does not match signed manifest");
    }
    let digest = hex::encode(Sha256::digest(bytes));
    if !digest.eq_ignore_ascii_case(&manifest.sha256) {
        bail!("downloaded update digest does not match signed manifest");
    }
    Ok(())
}

/// Configuration for the update system.
#[derive(Debug, Clone)]
pub struct UpdateConfig {
    /// GitHub repo to check for releases (e.g. "owner/hi").
    pub repo: String,
    /// Current installed version (e.g. "0.3.1").
    pub current_version: String,
    /// HTTP timeout for version checks.
    pub timeout: Duration,
    /// GitHub API base URL. Overridable for tests and GitHub Enterprise.
    pub api_base: String,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            // Must match the repo that actually publishes releases (git
            // origin): a default under anyone else's GitHub namespace lets
            // that account steer every `hi update` user to arbitrary
            // "Install manually from:" URLs.
            repo: "PipeNetwork/hi".to_string(),
            current_version: env!("CARGO_PKG_VERSION").to_string(),
            timeout: Duration::from_secs(10),
            api_base: "https://api.github.com".to_string(),
        }
    }
}

/// The result of an update check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStatus {
    /// Currently installed version.
    pub current_version: String,
    /// Latest available version (if the check succeeded).
    pub latest_version: Option<String>,
    /// Whether an update is available.
    pub update_available: bool,
    /// URL to download the latest release, if an update is available.
    pub download_url: Option<String>,
    /// Error message if the check failed.
    pub error: Option<String>,
}

/// A GitHub release (minimal subset of the API response).
#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

#[derive(Debug, Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

/// Check for an available update. Does not download or install — just compares
/// the current version against the latest GitHub release.
pub async fn check_for_update(config: &UpdateConfig) -> UpdateStatus {
    let status = check_for_update_inner(config).await;
    if status.error.is_some() {
        hi_observability::record(hi_observability::ReliabilityEvent::UpdateCheckFailure);
    }
    status
}

async fn check_for_update_inner(config: &UpdateConfig) -> UpdateStatus {
    let client = reqwest::Client::builder()
        .timeout(config.timeout)
        .user_agent(format!("hi/{}", config.current_version))
        .build()
        .unwrap_or_else(|_| {
            reqwest::Client::builder()
                .timeout(config.timeout)
                .build()
                .expect("failed to build timed reqwest Client")
        });

    let url = format!(
        "{}/repos/{}/releases/latest",
        config.api_base.trim_end_matches('/'),
        config.repo
    );

    match client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
    {
        Ok(resp) => {
            if !resp.status().is_success() {
                return UpdateStatus {
                    current_version: config.current_version.clone(),
                    latest_version: None,
                    update_available: false,
                    download_url: None,
                    error: Some(format!("GitHub API returned {}", resp.status())),
                };
            }
            // Cap the buffered body: the response source is the network, and
            // `json()` would otherwise buffer without bound under the request
            // timeout.
            let body = read_capped_body(resp).await;
            let release = match body {
                Ok(body) => serde_json::from_slice::<GitHubRelease>(&body)
                    .map_err(|e| format!("parsing release response: {e}")),
                Err(error) => Err(error),
            };
            match release {
                Ok(release) => {
                    let latest = release.tag_name.trim_start_matches('v').to_string();
                    let update_available = is_newer(&latest, &config.current_version);
                    let download_url = find_asset_url(&release.assets);
                    UpdateStatus {
                        current_version: config.current_version.clone(),
                        latest_version: Some(latest),
                        update_available,
                        download_url,
                        error: None,
                    }
                }
                Err(error) => UpdateStatus {
                    current_version: config.current_version.clone(),
                    latest_version: None,
                    update_available: false,
                    download_url: None,
                    error: Some(error),
                },
            }
        }
        Err(e) => UpdateStatus {
            current_version: config.current_version.clone(),
            latest_version: None,
            update_available: false,
            download_url: None,
            error: Some(format!("checking for update: {e}")),
        },
    }
}

pub const UNSUPPORTED_INSTALL_ERROR: &str = "automatic update installation is unsupported: this build has no embedded-key signed update-manifest contract; download and install a release manually";

/// Remote installation is unavailable until a signed manifest contract and
/// embedded verification keys are part of this repository.
pub async fn install_update(_config: &UpdateConfig, _destination: &Path) -> Result<String> {
    bail!(UNSUPPORTED_INSTALL_ERROR)
}

const MAX_RELEASE_RESPONSE_BYTES: usize = 1024 * 1024;

async fn read_capped_body(resp: reqwest::Response) -> Result<Vec<u8>, String> {
    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("reading release response: {e}"))?;
        if body.len().saturating_add(chunk.len()) > MAX_RELEASE_RESPONSE_BYTES {
            return Err(format!(
                "release response exceeds {MAX_RELEASE_RESPONSE_BYTES} bytes"
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Find the download URL for the current platform's binary, preferring an
/// asset that names this machine's architecture — an OS-only match can hand
/// an arm64 user an x86_64 binary when a release ships both.
fn find_asset_url(assets: &[GitHubAsset]) -> Option<String> {
    let target = platform_asset_name();
    let platform_match = |asset: &&GitHubAsset| {
        asset.name.contains(target)
            && !asset.name.ends_with(".sha256")
            && !asset.name.ends_with(".sha256sum")
            && !asset.name.ends_with(".txt")
    };
    assets
        .iter()
        .filter(platform_match)
        .find(|asset| asset.name.contains(std::env::consts::ARCH))
        .or_else(|| assets.iter().find(platform_match))
        .map(|asset| asset.browser_download_url.clone())
}

/// The asset name pattern for the current platform.
fn platform_asset_name() -> &'static str {
    let _os = if cfg!(target_os = "macos") {
        "apple-darwin"
    } else if cfg!(target_os = "linux") {
        "unknown-linux"
    } else if cfg!(target_os = "windows") {
        "pc-windows"
    } else {
        "unknown"
    };
    // This is a static string but we need to return a &'static str.
    // Since we can't format at compile time easily, use a match.
    match () {
        _ if cfg!(target_os = "macos") => "apple-darwin",
        _ if cfg!(target_os = "linux") => "unknown-linux",
        _ if cfg!(target_os = "windows") => "pc-windows",
        _ => "unknown",
    }
}

/// Compare two semver-like version strings. Returns true if `latest` is newer
/// than `current`.
///
/// Parses `major.minor.patch` numerically, ignores build metadata (`+…`), and
/// orders a release after its own pre-releases (`1.0.0` > `1.0.0-rc.1`).
pub fn is_newer(latest: &str, current: &str) -> bool {
    parse_version(latest) > parse_version(current)
}

/// Parse a version string into a tuple of (major, minor, patch, is_release).
fn parse_version(v: &str) -> (u32, u32, u32, bool) {
    let v = v.trim_start_matches('v');
    // Build metadata does not participate in ordering; without stripping it,
    // "1.2.3+build" parses as (1, 2, 0) and the identical release looks newer
    // forever.
    let v = v.split('+').next().unwrap_or(v);
    let (core, is_release) = match v.split_once('-') {
        Some((core, _)) => (core, false),
        None => (v, true),
    };
    let mut iter = core.split('.').map(|p| p.parse().unwrap_or(0));
    (
        iter.next().unwrap_or(0),
        iter.next().unwrap_or(0),
        iter.next().unwrap_or(0),
        is_release,
    )
}

/// Print an [`UpdateStatus`] to stdout in human-readable format.
pub fn print_update_status(status: &UpdateStatus) {
    if let Some(error) = &status.error {
        println!(
            "hi v{} — update check failed: {error}",
            status.current_version
        );
        return;
    }
    if status.update_available {
        if let Some(latest) = &status.latest_version {
            println!(
                "hi v{} — update available: v{}",
                status.current_version, latest
            );
            if let Some(url) = &status.download_url {
                println!("  Install manually from: {url}");
            }
            println!("  Automatic installation is unavailable in this build.");
        }
    } else {
        println!("hi v{} — up to date", status.current_version);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use std::fs;

    fn asset(name: &str) -> GitHubAsset {
        GitHubAsset {
            name: name.into(),
            browser_download_url: "http://localhost/asset".into(),
        }
    }

    #[test]
    fn is_newer_true_for_higher_version() {
        assert!(is_newer("1.0.0", "0.9.0"));
        assert!(is_newer("0.3.2", "0.3.1"));
        assert!(is_newer("1.0.0", "0.99.99"));
    }

    #[test]
    fn is_newer_false_for_same_or_lower() {
        assert!(!is_newer("0.3.1", "0.3.1"));
        assert!(!is_newer("0.3.0", "0.3.1"));
        assert!(!is_newer("0.2.9", "0.3.1"));
    }

    #[test]
    fn is_newer_handles_v_prefix() {
        assert!(is_newer("v1.0.0", "0.9.0"));
        assert!(is_newer("1.0.0", "v0.9.0"));
    }

    #[test]
    fn is_newer_ignores_prerelease_suffix() {
        assert!(is_newer("1.0.0-alpha.1", "0.9.0"));
        assert!(!is_newer("0.3.1-alpha.1", "0.3.1"));
    }

    #[test]
    fn parse_version_handles_missing_parts() {
        assert_eq!(parse_version("1"), (1, 0, 0, true));
        assert_eq!(parse_version("1.2"), (1, 2, 0, true));
        assert_eq!(parse_version("1.2.3"), (1, 2, 3, true));
    }

    #[test]
    fn parse_version_handles_invalid_parts() {
        assert_eq!(parse_version("a.b.c"), (0, 0, 0, true));
        assert_eq!(parse_version("1.x.3"), (1, 0, 3, true));
    }

    #[test]
    fn build_metadata_does_not_affect_ordering() {
        assert_eq!(parse_version("1.2.3+build.5"), (1, 2, 3, true));
        // Without stripping metadata, the identical release looked newer than
        // the running "1.2.3+build" binary forever.
        assert!(!is_newer("1.2.3", "1.2.3+build.5"));
    }

    #[test]
    fn release_is_newer_than_its_own_prerelease() {
        assert!(is_newer("1.0.0", "1.0.0-rc.1"));
        assert!(!is_newer("1.0.0-rc.1", "1.0.0"));
    }

    #[test]
    fn update_status_serializes_to_json() {
        let status = UpdateStatus {
            current_version: "0.3.1".into(),
            latest_version: Some("0.4.0".into()),
            update_available: true,
            download_url: Some("https://example.com/hi".into()),
            error: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("0.3.1"));
        assert!(json.contains("0.4.0"));
        assert!(json.contains("update_available"));
    }

    #[test]
    fn asset_selection_ignores_checksum_assets() {
        let platform = platform_asset_name();
        let assets = vec![
            asset(&format!("hi-{platform}.sha256")),
            asset(&format!("hi-{platform}")),
        ];
        assert_eq!(
            find_asset_url(&assets).as_deref(),
            Some("http://localhost/asset")
        );
    }

    #[tokio::test]
    async fn unsigned_remote_install_is_rejected_before_network_or_disk_access() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("hi");
        fs::write(&destination, b"old").unwrap();
        let config = UpdateConfig {
            api_base: "http://127.0.0.1:1".into(),
            ..UpdateConfig::default()
        };
        let error = install_update(&config, &destination).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("embedded-key signed update-manifest")
        );
        assert_eq!(fs::read(destination).unwrap(), b"old");
    }

    #[test]
    fn signed_manifest_verifies_canonical_content_and_asset() {
        let signing = SigningKey::from_bytes(&[7; 32]);
        let bytes = b"new hi binary";
        let manifest = SignedUpdateManifest {
            schema: UPDATE_MANIFEST_SCHEMA,
            key_id: "test-1".into(),
            version: "1.2.3".into(),
            target: platform_asset_name().into(),
            asset_name: "hi".into(),
            asset_url: "https://example.com/hi".into(),
            size: bytes.len() as u64,
            sha256: hex::encode(Sha256::digest(bytes)),
        };
        manifest.validate("v1.2.3").unwrap();
        let signature = signing.sign(&manifest.canonical_bytes().unwrap());
        verify_signed_manifest(
            &manifest,
            &hex::encode(signature.to_bytes()),
            &[("test-1", signing.verifying_key().to_bytes())],
        )
        .unwrap();
        verify_asset_bytes(&manifest, bytes).unwrap();
        assert!(verify_asset_bytes(&manifest, b"tampered").is_err());
    }

    #[test]
    fn signed_manifest_fails_closed_for_unknown_key_or_insecure_url() {
        let manifest = SignedUpdateManifest {
            schema: UPDATE_MANIFEST_SCHEMA,
            key_id: "unknown".into(),
            version: "1.2.3".into(),
            target: platform_asset_name().into(),
            asset_name: "hi".into(),
            asset_url: "http://example.com/hi".into(),
            size: 1,
            sha256: "00".repeat(32),
        };
        assert!(manifest.validate("1.2.3").is_err());
        assert!(verify_signed_manifest(&manifest, &"00".repeat(64), &[]).is_err());
    }
}
