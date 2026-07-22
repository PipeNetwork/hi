//! Self-update — check for new versions and download/install updates.
//!
//! Inspired by grok-build's `xai-grok-update` crate. Simplified for hi:
//! checks a GitHub releases API for the latest version, compares against the
//! installed version, and optionally downloads the binary for the current
//! platform.
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

use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Configuration for the update system.
#[derive(Debug, Clone)]
pub struct UpdateConfig {
    /// GitHub repo to check for releases (e.g. "owner/hi").
    pub repo: String,
    /// Current installed version (e.g. "0.3.1").
    pub current_version: String,
    /// HTTP timeout for version checks.
    pub timeout: Duration,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            repo: "david/hi".to_string(),
            current_version: env!("CARGO_PKG_VERSION").to_string(),
            timeout: Duration::from_secs(10),
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
    html_url: String,
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
    let client = reqwest::Client::builder()
        .timeout(config.timeout)
        .user_agent(format!("hi/{}", config.current_version))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let url = format!("https://api.github.com/repos/{}/releases/latest", config.repo);

    match client.get(&url).header("Accept", "application/vnd.github+json").send().await {
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
            match resp.json::<GitHubRelease>().await {
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
                Err(e) => UpdateStatus {
                    current_version: config.current_version.clone(),
                    latest_version: None,
                    update_available: false,
                    download_url: None,
                    error: Some(format!("parsing release response: {e}")),
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

/// Find the download URL for the current platform's binary.
fn find_asset_url(assets: &[GitHubAsset]) -> Option<String> {
    let target = platform_asset_name();
    assets
        .iter()
        .find(|a| a.name.contains(target))
        .map(|a| a.browser_download_url.clone())
}

/// The asset name pattern for the current platform.
fn platform_asset_name() -> &'static str {
    let os = if cfg!(target_os = "macos") {
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
/// Parses `major.minor.patch` and compares numerically. Pre-release suffixes
/// (e.g. `-alpha.1`) are ignored for the comparison.
pub fn is_newer(latest: &str, current: &str) -> bool {
    let latest_parts = parse_version(latest);
    let current_parts = parse_version(current);
    latest_parts > current_parts
}

/// Parse a version string into a tuple of (major, minor, patch).
fn parse_version(v: &str) -> (u32, u32, u32) {
    let v = v.trim_start_matches('v');
    let v = v.split('-').next().unwrap_or(v);
    let parts: Vec<u32> = v
        .split('.')
        .map(|p| p.parse().unwrap_or(0))
        .collect();
    let mut iter = parts.into_iter();
    (
        iter.next().unwrap_or(0),
        iter.next().unwrap_or(0),
        iter.next().unwrap_or(0),
    )
}

/// Print an [`UpdateStatus`] to stdout in human-readable format.
pub fn print_update_status(status: &UpdateStatus) {
    if let Some(error) = &status.error {
        println!("hi v{} — update check failed: {error}", status.current_version);
        return;
    }
    if status.update_available {
        if let Some(latest) = &status.latest_version {
            println!("hi v{} — update available: v{}", status.current_version, latest);
            if let Some(url) = &status.download_url {
                println!("  Download: {url}");
            }
            println!("  Run `hi update` to install.");
        }
    } else {
        println!("hi v{} — up to date", status.current_version);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(parse_version("1"), (1, 0, 0));
        assert_eq!(parse_version("1.2"), (1, 2, 0));
        assert_eq!(parse_version("1.2.3"), (1, 2, 3));
    }

    #[test]
    fn parse_version_handles_invalid_parts() {
        assert_eq!(parse_version("a.b.c"), (0, 0, 0));
        assert_eq!(parse_version("1.x.3"), (1, 0, 3));
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
}
