//! Activity log — a persisted, cross-restart feed of the loud things loops
//! notice, so `/digest` can answer "what changed while I was away?"
//!
//! Every *loud* loop event (a real change a firing reported, a budget pause, an
//! expiry) is appended as one JSON line to a per-project `activity.jsonl`, next
//! to `loops.json`. `/digest` groups the feed by loop, shows recent changes, and
//! marks which are new since you last looked (a millis watermark in
//! `activity.seen`). The entry carries a free-form `source`, so fleet/goal can
//! append to the same feed later without a schema change.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Keep at most this many entries in the file (trimmed oldest-first on append
/// once the file grows past the check threshold).
const MAX_ENTRIES: usize = 500;
/// Only bother trimming when the file is larger than this (bytes).
const TRIM_ABOVE_BYTES: u64 = 256 * 1024;

/// One recorded loud event.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct ActivityEntry {
    /// Unix millis.
    pub(crate) at_ms: u64,
    /// The originating loop id (0 if not a loop).
    #[serde(default)]
    pub(crate) loop_id: u64,
    /// A short human label for the source (a loop's name, or "loop#3").
    pub(crate) source: String,
    /// What happened, in one line.
    pub(crate) text: String,
}

/// The activity feed path for a project (sibling of its `loops.json`).
pub(crate) fn activity_path(loops_file: &Path) -> PathBuf {
    loops_file.with_file_name("activity.jsonl")
}

/// The "last looked" watermark path (sibling of `loops.json`).
pub(crate) fn seen_path(loops_file: &Path) -> PathBuf {
    loops_file.with_file_name("activity.seen")
}

/// Append one entry, creating the file/dir as needed. Best-effort: failures are
/// swallowed (the transcript still shows the event). Trims the file when it has
/// grown large so it can't accumulate without bound.
pub(crate) fn append(path: &Path, entry: &ActivityEntry) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) > TRIM_ABOVE_BYTES {
        let mut kept = load(path);
        let drop = kept.len().saturating_sub(MAX_ENTRIES);
        if drop > 0 {
            kept.drain(..drop);
            let body: String = kept
                .iter()
                .filter_map(|e| serde_json::to_string(e).ok())
                .map(|l| l + "\n")
                .collect();
            let _ = std::fs::write(path, body);
        }
    }
    if let Ok(line) = serde_json::to_string(entry) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// Load all entries (oldest first); malformed lines are skipped.
pub(crate) fn load(path: &Path) -> Vec<ActivityEntry> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// The "last looked" watermark (0 if never).
pub(crate) fn load_seen(path: &Path) -> u64 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(0)
}

/// Record that the feed has been viewed up to `at_ms`.
pub(crate) fn save_seen(path: &Path, at_ms: u64) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, at_ms.to_string());
}

/// One source's rolled-up activity for the digest (a loop, a fleet row, goal…).
pub(crate) struct SourceDigest {
    /// The grouping key + display label ("loop#3 check CI", "fleet#5 …", "goal").
    pub(crate) source: String,
    /// Number of events in the window.
    pub(crate) count: usize,
    /// How many of those are newer than the seen watermark.
    pub(crate) fresh: usize,
    /// The most recent few events (newest first): (at_ms, text, fresh).
    pub(crate) recent: Vec<(u64, String, bool)>,
}

/// Group the feed by source for display: only entries at/after `since_ms` are
/// included; `seen_ms` marks which of those are new. Grouping by the `source`
/// string keeps loops, fleet rows, and goal distinct in one feed. Groups are
/// ordered by most recent activity first. Returns `(groups, total, total_fresh)`.
pub(crate) fn digest(
    entries: &[ActivityEntry],
    since_ms: u64,
    seen_ms: u64,
    recent_per_source: usize,
) -> (Vec<SourceDigest>, usize, usize) {
    let mut order: Vec<String> = Vec::new();
    let mut groups: std::collections::HashMap<String, SourceDigest> =
        std::collections::HashMap::new();
    let mut total = 0usize;
    let mut total_fresh = 0usize;
    for e in entries.iter().filter(|e| e.at_ms >= since_ms) {
        total += 1;
        let fresh = e.at_ms > seen_ms;
        if fresh {
            total_fresh += 1;
        }
        let g = groups.entry(e.source.clone()).or_insert_with(|| {
            order.push(e.source.clone());
            SourceDigest {
                source: e.source.clone(),
                count: 0,
                fresh: 0,
                recent: Vec::new(),
            }
        });
        g.count += 1;
        if fresh {
            g.fresh += 1;
        }
        g.recent.push((e.at_ms, e.text.clone(), fresh));
    }
    // Newest-first recents, capped; groups ordered by their latest event.
    for g in groups.values_mut() {
        g.recent.reverse();
        g.recent.truncate(recent_per_source);
    }
    let mut out: Vec<SourceDigest> = order
        .into_iter()
        .filter_map(|s| groups.remove(&s))
        .collect();
    out.sort_by(|a, b| {
        let (la, lb) = (
            a.recent.first().map(|r| r.0).unwrap_or(0),
            b.recent.first().map(|r| r.0).unwrap_or(0),
        );
        lb.cmp(&la)
    });
    (out, total, total_fresh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(at_ms: u64, loop_id: u64, text: &str) -> ActivityEntry {
        ActivityEntry {
            at_ms,
            loop_id,
            source: format!("loop#{loop_id}"),
            text: text.into(),
        }
    }

    #[test]
    fn append_load_and_seen_round_trip() {
        let dir = std::env::temp_dir().join(format!("hi-activity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let loops = dir.join("loops.json");
        let path = activity_path(&loops);
        append(&path, &e(1000, 3, "CI went red"));
        append(&path, &e(2000, 3, "CI green again"));
        append(&path, &e(1500, 5, "p99 spiked"));
        let all = load(&path);
        assert_eq!(all.len(), 3);

        let seen = seen_path(&loops);
        assert_eq!(load_seen(&seen), 0);
        save_seen(&seen, 1200);
        assert_eq!(load_seen(&seen), 1200);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn digest_groups_by_loop_and_marks_fresh() {
        let entries = vec![
            e(1000, 3, "CI went red"),
            e(1500, 5, "p99 spiked"),
            e(2000, 3, "CI green again"),
            e(2500, 3, "CI red again"),
        ];
        // seen at 1600 → the two loop#3 events at 2000/2500 are fresh; loop#5 is not.
        let (groups, total, fresh) = digest(&entries, 0, 1600, 3);
        assert_eq!(total, 4);
        assert_eq!(fresh, 2);
        // loop#3 sorts first (its latest event 2500 > loop#5's 1500).
        assert_eq!(groups[0].source, "loop#3");
        assert_eq!(groups[0].count, 3);
        assert_eq!(groups[0].fresh, 2);
        // Newest-first recents.
        assert_eq!(groups[0].recent[0].1, "CI red again");
        assert!(groups[0].recent[0].2, "newest is fresh");
        assert_eq!(groups[1].source, "loop#5");
        assert_eq!(groups[1].fresh, 0);
    }

    #[test]
    fn digest_keeps_distinct_sources_separate() {
        // A loop and a fleet row sharing the numeric id 3 must NOT merge — the
        // whole point of grouping the unified feed by the `source` string.
        let entries = vec![
            ActivityEntry {
                at_ms: 1000,
                loop_id: 3,
                source: "loop#3 watch CI".into(),
                text: "CI red".into(),
            },
            ActivityEntry {
                at_ms: 2000,
                loop_id: 0,
                source: "fleet#3 port module".into(),
                text: "merged 2 files".into(),
            },
        ];
        let (groups, total, _) = digest(&entries, 0, 0, 3);
        assert_eq!(total, 2);
        assert_eq!(groups.len(), 2, "loop#3 and fleet#3 stay distinct");
    }

    #[test]
    fn digest_window_filters_old_entries() {
        let entries = vec![e(1000, 1, "old"), e(5000, 1, "recent")];
        let (groups, total, _) = digest(&entries, 4000, 0, 3);
        assert_eq!(total, 1);
        assert_eq!(groups[0].recent[0].1, "recent");
    }
}
