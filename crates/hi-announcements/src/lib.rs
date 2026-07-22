//! Shared announcement types, persistence, and formatting for `hi` CLI apps.
//!
//! Provides the wire types for remote announcements (e.g. release notes, tips,
//! important changes surfaced to users on startup), plus persistence for
//! dismissed/hidden announcement IDs and filtering of expired entries.
//!
//! Inspired by grok-build's `xai-grok-announcements` crate.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn run() -> anyhow::Result<()> {
//! use hi_announcements::{RemoteAnnouncement, read_hidden_announcement_ids};
//!
//! let hidden = read_hidden_announcement_ids("~/.hi").await?;
//! let ann = RemoteAnnouncement {
//!     id: Some("release-0.3".to_string()),
//!     message: Some("hi 0.3 is out!".to_string()),
//!     ..Default::default()
//! };
//! assert!(!hi_announcements::visible_announcements(&[ann]).is_empty());
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// A call-to-action link attached to an announcement.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnnouncementCta {
    /// The display label for the link (e.g. `"Read more"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// The URL to open when the user activates the CTA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Optional caption/tooltip text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
}

/// Severity level for an announcement.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnnouncementSeverity {
    /// Low-priority informational message.
    #[default]
    Info,
    /// Something the user should pay attention to.
    Warning,
    /// Critical: action required.
    Critical,
}

/// A remote announcement fetched from a server or override.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteAnnouncement {
    /// Unique identifier. If absent, a content-based key is derived.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The announcement body text. Entries with empty/absent messages are
    /// filtered out by [`visible_announcements`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Severity level.
    #[serde(default)]
    pub severity: AnnouncementSeverity,
    /// Optional title/header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Optional call-to-action link.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cta: Option<AnnouncementCta>,
    /// Unix timestamp (seconds) of last update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<u64>,
    /// Unix timestamp (seconds) after which the announcement is expired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    /// Whether the user can dismiss the announcement.
    #[serde(default = "default_true")]
    pub dismissible: bool,
    /// Whether the announcement persists across sessions (not auto-dismissed).
    #[serde(default)]
    pub persistent: bool,
}

fn default_true() -> bool {
    true
}

/// Notification that announcements have been refreshed from a remote source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnnouncementsRefreshed {
    /// Monotonically increasing generation number.
    #[serde(rename = "gen")]
    pub r#gen: u64,
    /// The full set of announcements from the refresh.
    pub announcements: Vec<RemoteAnnouncement>,
}

// ---------------------------------------------------------------------------
// Hidden/dismissed ID persistence
// ---------------------------------------------------------------------------

/// Derive the hide key for an announcement: the `id` if present, otherwise a
/// content-based fallback using the message text.
pub fn announcement_hide_key(a: &RemoteAnnouncement) -> String {
    if let Some(id) = &a.id {
        if !id.is_empty() {
            return id.clone();
        }
    }
    // Content-based fallback: hash of the message text, separated by a unit
    // separator to avoid collisions between id-based and content-based keys.
    let msg = a.message.as_deref().unwrap_or("");
    format!("content\x1f{msg}")
}

/// Parse a comma-separated list of hidden announcement IDs into a `BTreeSet`.
pub fn parse_hidden_announcement_ids(s: &str) -> BTreeSet<String> {
    s.split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// Serialize a `BTreeSet` of hidden IDs into a comma-separated string.
/// Returns `None` if the set is empty.
pub fn serialize_hidden_announcement_ids(ids: &BTreeSet<String>) -> Option<String> {
    if ids.is_empty() {
        return None;
    }
    Some(ids.iter().cloned().collect::<Vec<_>>().join(","))
}

/// Remove hidden IDs that no longer appear in the active set. Returns `true`
/// if any IDs were pruned.
pub fn prune_hidden_announcement_ids(
    hidden: &mut BTreeSet<String>,
    active: &[&RemoteAnnouncement],
) -> bool {
    let active_keys: BTreeSet<String> = active.iter().map(|a| announcement_hide_key(a)).collect();
    let before = hidden.len();
    hidden.retain(|k| active_keys.contains(k));
    hidden.len() != before
}

/// Path to the hidden-announcements file within a hi home directory.
fn hidden_file_path(hi_home: impl AsRef<Path>) -> std::path::PathBuf {
    hi_home.as_ref().join("announcements.json")
}

/// Read the set of hidden/dismissed announcement IDs from disk.
/// Returns an empty set if the file doesn't exist.
pub async fn read_hidden_announcement_ids(hi_home: impl AsRef<Path>) -> Result<BTreeSet<String>> {
    let path = hidden_file_path(hi_home);
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let data = tokio::fs::read_to_string(&path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(parse_hidden_announcement_ids(&data))
}

/// Write the set of hidden/dismissed announcement IDs to disk.
/// If the set is empty, the file is removed.
pub async fn write_hidden_announcement_ids(
    hi_home: impl AsRef<Path>,
    ids: &BTreeSet<String>,
) -> Result<()> {
    let path = hidden_file_path(hi_home);
    if ids.is_empty() {
        let _ = tokio::fs::remove_file(&path).await;
        return Ok(());
    }
    let data = serialize_hidden_announcement_ids(ids).unwrap_or_default();
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    tokio::fs::write(&path, data)
        .await
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

/// Filter to announcements that have a non-empty message.
pub fn visible_announcements(announcements: &[RemoteAnnouncement]) -> Vec<&RemoteAnnouncement> {
    announcements
        .iter()
        .filter(|a| a.message.as_deref().is_some_and(|m| !m.trim().is_empty()))
        .collect()
}

/// Whether an announcement is expired at the given Unix timestamp.
pub fn is_expired_at(a: &RemoteAnnouncement, now: u64) -> bool {
    a.expires_at.is_some_and(|exp| exp <= now)
}

/// Filter out expired announcements, using the current system time.
pub fn filter_expired(
    announcements: impl IntoIterator<Item = RemoteAnnouncement>,
) -> Vec<RemoteAnnouncement> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    filter_expired_at(announcements, now)
}

/// Filter out expired announcements at a specific timestamp.
pub fn filter_expired_at(
    announcements: impl IntoIterator<Item = RemoteAnnouncement>,
    now: u64,
) -> Vec<RemoteAnnouncement> {
    announcements
        .into_iter()
        .filter(|a| !is_expired_at(a, now))
        .collect()
}

/// Resolve startup announcements from a remote fetch result.
///
/// Honors the `HI_ANNOUNCEMENTS_OVERRIDE` environment variable: if set to
/// valid JSON, it replaces the remote result entirely (for testing/dev).
pub fn resolve_startup_announcements(
    remote: Result<Vec<RemoteAnnouncement>>,
) -> Option<Vec<RemoteAnnouncement>> {
    let override_str = std::env::var("HI_ANNOUNCEMENTS_OVERRIDE").ok();
    resolve_startup_announcements_with_override(remote, override_str.as_deref())
}

/// Like [`resolve_startup_announcements`] but takes the override string
/// explicitly, for testability without env-var races.
pub fn resolve_startup_announcements_with_override(
    remote: Result<Vec<RemoteAnnouncement>>,
    override_str: Option<&str>,
) -> Option<Vec<RemoteAnnouncement>> {
    if let Some(s) = override_str {
        if let Ok(parsed) = serde_json::from_str::<Vec<RemoteAnnouncement>>(s) {
            return Some(parsed);
        }
    }
    match remote {
        Ok(anns) if !anns.is_empty() => Some(anns),
        Ok(_) => None,
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ann(id: &str, msg: &str) -> RemoteAnnouncement {
        RemoteAnnouncement {
            id: Some(id.to_string()),
            message: Some(msg.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn hide_key_uses_id_when_present() {
        let a = ann("release-1", "hi 1.0 is out");
        assert_eq!(announcement_hide_key(&a), "release-1");
    }

    #[test]
    fn hide_key_falls_back_to_content() {
        let a = RemoteAnnouncement {
            id: None,
            message: Some("hello".to_string()),
            ..Default::default()
        };
        assert_eq!(announcement_hide_key(&a), "content\x1fhello");
    }

    #[test]
    fn hide_key_falls_back_when_id_empty() {
        let a = RemoteAnnouncement {
            id: Some("".to_string()),
            message: Some("hello".to_string()),
            ..Default::default()
        };
        assert_eq!(announcement_hide_key(&a), "content\x1fhello");
    }

    #[test]
    fn parse_and_serialize_roundtrip() {
        let mut ids = BTreeSet::new();
        ids.insert("a".to_string());
        ids.insert("b".to_string());
        ids.insert("c".to_string());
        let s = serialize_hidden_announcement_ids(&ids).unwrap();
        let back = parse_hidden_announcement_ids(&s);
        assert_eq!(ids, back);
    }

    #[test]
    fn serialize_empty_returns_none() {
        let ids = BTreeSet::new();
        assert!(serialize_hidden_announcement_ids(&ids).is_none());
    }

    #[test]
    fn prune_removes_stale_ids() {
        let mut hidden = BTreeSet::from(["old".to_string(), "keep".to_string()]);
        let active = vec![ann("keep", "msg")];
        let active_refs: Vec<&RemoteAnnouncement> = active.iter().collect();
        assert!(prune_hidden_announcement_ids(&mut hidden, &active_refs));
        assert!(hidden.contains("keep"));
        assert!(!hidden.contains("old"));
    }

    #[test]
    fn prune_noop_when_all_active() {
        let mut hidden = BTreeSet::from(["a".to_string()]);
        let active = vec![ann("a", "msg")];
        let active_refs: Vec<&RemoteAnnouncement> = active.iter().collect();
        assert!(!prune_hidden_announcement_ids(&mut hidden, &active_refs));
    }

    #[test]
    fn visible_filters_empty_messages() {
        let anns = vec![
            ann("1", "hello"),
            RemoteAnnouncement {
                id: Some("2".to_string()),
                message: Some("".to_string()),
                ..Default::default()
            },
            RemoteAnnouncement {
                id: Some("3".to_string()),
                message: None,
                ..Default::default()
            },
            ann("4", "  "), // whitespace-only
        ];
        let visible = visible_announcements(&anns);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id.as_deref(), Some("1"));
    }

    #[test]
    fn filter_expired_at_removes_past_expiry() {
        let anns = vec![
            RemoteAnnouncement {
                id: Some("expired".to_string()),
                message: Some("old".to_string()),
                expires_at: Some(100),
                ..Default::default()
            },
            RemoteAnnouncement {
                id: Some("active".to_string()),
                message: Some("new".to_string()),
                expires_at: Some(200),
                ..Default::default()
            },
            RemoteAnnouncement {
                id: Some("noexpiry".to_string()),
                message: Some("forever".to_string()),
                ..Default::default()
            },
        ];
        let filtered = filter_expired_at(anns, 150);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|a| a.id.as_deref() != Some("expired")));
    }

    #[test]
    fn resolve_startup_with_override() {
        let result = resolve_startup_announcements_with_override(
            Ok(vec![]),
            Some(r#"[{"id":"test","message":"override"}]"#),
        );
        assert!(result.is_some());
        let anns = result.unwrap();
        assert_eq!(anns.len(), 1);
        assert_eq!(anns[0].id.as_deref(), Some("test"));
    }

    #[test]
    fn resolve_startup_empty_remote_returns_none() {
        assert!(resolve_startup_announcements_with_override(Ok(vec![]), None).is_none());
    }

    #[test]
    fn resolve_startup_error_returns_none() {
        assert!(
            resolve_startup_announcements_with_override(Err(anyhow::anyhow!("network")), None)
                .is_none()
        );
    }

    #[test]
    fn resolve_startup_override_with_nonempty_remote_uses_override() {
        let remote = vec![ann("remote", "remote msg")];
        let result = resolve_startup_announcements_with_override(
            Ok(remote),
            Some(r#"[{"id":"override","message":"ov"}]"#),
        );
        // Override takes precedence.
        assert_eq!(result.unwrap()[0].id.as_deref(), Some("override"));
    }

    #[test]
    fn resolve_startup_invalid_override_falls_back_to_remote() {
        let remote = vec![ann("remote", "remote msg")];
        let result = resolve_startup_announcements_with_override(Ok(remote), Some("not json"));
        assert_eq!(result.unwrap()[0].id.as_deref(), Some("remote"));
    }

    #[tokio::test]
    async fn read_write_hidden_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let ids = BTreeSet::from(["a".to_string(), "b".to_string()]);
        write_hidden_announcement_ids(tmp.path(), &ids)
            .await
            .unwrap();
        let back = read_hidden_announcement_ids(tmp.path()).await.unwrap();
        assert_eq!(ids, back);
    }

    #[tokio::test]
    async fn write_empty_removes_file() {
        let tmp = tempfile::tempdir().unwrap();
        let ids = BTreeSet::from(["a".to_string()]);
        write_hidden_announcement_ids(tmp.path(), &ids)
            .await
            .unwrap();
        assert!(hidden_file_path(tmp.path()).exists());
        let empty = BTreeSet::new();
        write_hidden_announcement_ids(tmp.path(), &empty)
            .await
            .unwrap();
        assert!(!hidden_file_path(tmp.path()).exists());
    }

    #[tokio::test]
    async fn read_nonexistent_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let ids = read_hidden_announcement_ids(tmp.path()).await.unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn announcements_refreshed_serde() {
        let ar = AnnouncementsRefreshed {
            r#gen: 5,
            announcements: vec![ann("1", "hi")],
        };
        let json = serde_json::to_string(&ar).unwrap();
        assert!(json.contains("\"gen\":5"));
        let back: AnnouncementsRefreshed = serde_json::from_str(&json).unwrap();
        assert_eq!(ar, back);
    }
}
