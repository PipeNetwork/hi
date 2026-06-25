//! Persistence seam. The agent records newly-produced messages after each turn
//! through a [`SessionSink`]; the CLI provides a JSONL-file implementation.

use anyhow::Result;
use hi_ai::{Message, Usage};

/// Records conversation messages durably. Implementations do their own IO.
pub trait SessionSink: Send {
    /// Append `messages` (the ones produced since the last call) to storage.
    fn record(&mut self, messages: &[Message], usage: Usage, cost_usd: Option<f64>) -> Result<()>;

    /// Persist a compaction boundary: the compacted messages replace all prior
    /// messages in storage, so a resumed session starts from the compacted state.
    fn record_compaction(&mut self, messages: &[Message]) -> Result<()>;

    /// Persist a long-horizon goal's state so a resumed session picks it up at
    /// its active sub-goal. Last write wins (the goal is replaced wholesale).
    /// Default no-op so existing mock sinks don't need to implement it.
    fn record_goal(&mut self, _goal: &crate::Goal) -> Result<()> {
        Ok(())
    }
}
