//! Persistence seam. The agent records newly-produced messages after each turn
//! through a [`SessionSink`]; the CLI provides a JSONL-file implementation.

use anyhow::Result;
use hi_ai::{Message, Usage};

/// Records conversation messages durably. Implementations do their own IO.
pub trait SessionSink: Send {
    /// Append `messages` (the ones produced since the last call) to storage.
    fn record(&mut self, messages: &[Message], usage: Usage) -> Result<()>;

    /// Persist a compaction boundary: the compacted messages replace all prior
    /// messages in storage, so a resumed session starts from the compacted state.
    fn record_compaction(&mut self, messages: &[Message]) -> Result<()>;

    /// Persist an explicit replacement of the durable conversational state.
    ///
    /// This is used by `/retry` and interrupted-turn discard: the visible
    /// transcript, structured goal, and decision log must rewind together. A
    /// JSONL implementation can write this as one metadata record so resume
    /// cannot observe a discarded transcript with stale side-channel state.
    fn record_state_replacement(
        &mut self,
        messages: &[Message],
        goal: Option<&crate::Goal>,
        decisions: &crate::DecisionLog,
    ) -> Result<()> {
        self.record_compaction(messages)?;
        match goal {
            Some(goal) => self.record_goal(goal)?,
            None => self.clear_goal()?,
        }
        self.record_decisions(decisions)
    }

    /// Persist the retained git checkpoint refs so `/undo` still has the same
    /// stack after resume. Last write wins.
    fn record_checkpoints(&mut self, _refs: &[String]) -> Result<()> {
        Ok(())
    }

    /// Persist a long-horizon goal's state so a resumed session picks it up at
    /// its active sub-goal. Last write wins (the goal is replaced wholesale).
    /// Default no-op so existing mock sinks don't need to implement it.
    fn record_goal(&mut self, _goal: &crate::Goal) -> Result<()> {
        Ok(())
    }

    /// Persist that the long-horizon goal was cleared. Default no-op so
    /// existing mock sinks don't need to implement it.
    fn clear_goal(&mut self) -> Result<()> {
        Ok(())
    }

    /// Persist the intra-session decision log so a resumed session keeps the
    /// same key decisions in its rebuilt system prompt. Last write wins.
    fn record_decisions(&mut self, _decisions: &crate::DecisionLog) -> Result<()> {
        Ok(())
    }
}
