//! Shared prompt-queue wire types for `hi`.
//!
//! Defines the serializable types used to communicate the prompt queue between
//! the agent loop and frontends (TUI, CLI). The queue lets users stack prompts
//! that execute sequentially after the current turn completes — durable across
//! compaction boundaries and frontend reconnects.
//!
//! Inspired by grok-build's `xai-prompt-queue` crate.
//!
//! # Quick start
//!
//! ```
//! use hi_prompt_queue::{QueueEntryWire, QueueEntryMeta};
//!
//! let entry = QueueEntryWire {
//!     id: "q-1".to_string(),
//!     text: "Fix the failing tests".to_string(),
//!     meta: QueueEntryMeta::default(),
//! };
//! let json = serde_json::to_string(&entry).unwrap();
//! assert!(json.contains("q-1"));
//! ```

use serde::{Deserialize, Serialize};

/// Metadata for a queued prompt entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueueEntryMeta {
    /// Unix timestamp (seconds) when the entry was enqueued.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enqueued_at: Option<u64>,
    /// Whether the entry was submitted while a turn was in progress (deferred).
    #[serde(default)]
    pub deferred: bool,
}

/// A single entry in the prompt queue, as transmitted over the wire.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueueEntryWire {
    /// Unique identifier for this entry (e.g. `"q-1"`).
    pub id: String,
    /// The prompt text to send when this entry is dequeued.
    pub text: String,
    /// Additional metadata about the entry.
    #[serde(default)]
    pub meta: QueueEntryMeta,
}

/// Notification that the queue has changed (entries added, removed, or
/// reordered). Frontends use this to refresh their queue display.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueueChanged {
    /// Monotonically increasing generation number for the queue state.
    #[serde(rename = "gen")]
    pub r#gen: u64,
    /// The full current queue, in execution order (index 0 = next to run).
    pub entries: Vec<QueueEntryWire>,
}

impl QueueChanged {
    /// Create a `QueueChanged` with generation 0 and an empty queue.
    pub fn empty() -> Self {
        Self {
            r#gen: 0,
            entries: Vec::new(),
        }
    }

    /// Create a `QueueChanged` with the given generation and entries.
    pub fn new(r#gen: u64, entries: Vec<QueueEntryWire>) -> Self {
        Self { r#gen, entries }
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of entries in the queue.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

impl Default for QueueChanged {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_queue_entry() {
        let entry = QueueEntryWire {
            id: "q-1".to_string(),
            text: "Fix the tests".to_string(),
            meta: QueueEntryMeta {
                enqueued_at: Some(1700000000),
                deferred: true,
            },
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: QueueEntryWire = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn roundtrip_queue_changed() {
        let qc = QueueChanged::new(
            3,
            vec![
                QueueEntryWire {
                    id: "a".to_string(),
                    text: "first".to_string(),
                    meta: QueueEntryMeta::default(),
                },
                QueueEntryWire {
                    id: "b".to_string(),
                    text: "second".to_string(),
                    meta: QueueEntryMeta::default(),
                },
            ],
        );
        let json = serde_json::to_string(&qc).unwrap();
        let back: QueueChanged = serde_json::from_str(&json).unwrap();
        assert_eq!(qc, back);
    }

    #[test]
    fn empty_queue() {
        let q = QueueChanged::empty();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert_eq!(q.r#gen, 0);
    }

    #[test]
    fn meta_defaults_are_minimal_json() {
        let entry = QueueEntryWire {
            id: "x".to_string(),
            text: "hi".to_string(),
            meta: QueueEntryMeta::default(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        // enqueued_at is None and should be skipped.
        assert!(!json.contains("enqueued_at"));
        // deferred is false (default) and should be present as a bool.
        assert!(json.contains("\"deferred\":false"));
    }
}
