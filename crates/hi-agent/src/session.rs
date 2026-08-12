//! Persistence seam. The agent records newly-produced messages after each turn
//! through a [`SessionSink`]; the CLI provides a JSONL-file implementation.

use anyhow::Result;
use hi_ai::{Message, Usage};

/// Records conversation messages durably. Implementations do their own IO.
pub trait SessionSink: Send {
    /// Stable identifier of the durable session this sink writes to — the
    /// transcript file stem for JSONL sessions. `None` for sinks with no
    /// durable identity (tests, ephemeral runs). Findings-ledger records use
    /// this so post-mortems can point at the exact transcript.
    fn id(&self) -> Option<String> {
        None
    }

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
        plan: &[crate::PlanStep],
    ) -> Result<()> {
        self.record_compaction(messages)?;
        match goal {
            Some(goal) => self.record_goal(goal)?,
            None => self.clear_goal()?,
        }
        self.record_decisions(decisions)?;
        if plan.is_empty() {
            self.clear_plan()
        } else {
            self.record_plan(plan)
        }
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

    /// Persist the current unfinished task plan. Last write wins.
    fn record_plan(&mut self, _plan: &[crate::PlanStep]) -> Result<()> {
        Ok(())
    }

    /// Persist that no task plan should be restored.
    fn clear_plan(&mut self) -> Result<()> {
        Ok(())
    }

    /// Persist plan-drive pause and stall. Last write wins. Default no-op.
    fn record_plan_drive(&mut self, _paused: bool, _stall: u32) -> Result<()> {
        Ok(())
    }

    /// Persist goal-drive stall. Last write wins. Default no-op.
    fn record_goal_drive(&mut self, _stall: u32) -> Result<()> {
        Ok(())
    }

    /// Persist the intra-session decision log so a resumed session keeps the
    /// same key decisions in its rebuilt system prompt. Last write wins.
    fn record_decisions(&mut self, _decisions: &crate::DecisionLog) -> Result<()> {
        Ok(())
    }

    /// Persist one turn's final outcome — status, verification, review, stop
    /// reason — plus why review produced no verdict when it didn't. Appended
    /// per turn; a diagnostic record for post-mortems, ignored on resume.
    /// Default no-op so existing mock sinks don't need to implement it.
    fn record_turn_outcome(
        &mut self,
        _outcome: &crate::TurnOutcome,
        _review_unavailable_reason: Option<&str>,
    ) -> Result<()> {
        Ok(())
    }
}
