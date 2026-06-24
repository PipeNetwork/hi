//! Persistence seam. The agent records newly-produced messages after each turn
//! through a [`SessionSink`]; the CLI provides a JSONL-file implementation.

use anyhow::Result;
use hi_ai::{Message, Usage};

/// Records conversation messages durably. Implementations do their own IO.
pub trait SessionSink: Send {
    /// Append `messages` (the ones produced since the last call) to storage.
    fn record(&mut self, messages: &[Message], usage: Usage, cost_usd: Option<f64>) -> Result<()>;
}
