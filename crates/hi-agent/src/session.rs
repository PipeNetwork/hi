//! Persistence seam. The agent records newly-produced messages after each turn
//! through a [`SessionSink`]; the CLI provides a JSONL-file implementation.

use anyhow::Result;
use hi_ai::{Content, Message, Usage};

/// One provider-facing result sealed into a workspace settlement transcript.
///
/// This record is deliberately independent of the ordinary JSONL message
/// projection. PipeFS stages it in the remote outbox before workspace
/// settlement, so the causal commit can acknowledge the exact result that is
/// about to become visible to the model. The ordinary assistant/tool messages
/// are appended only after that acknowledgement.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkspaceTranscriptCall {
    pub call_id: String,
    pub name: String,
    pub result: String,
}

/// Durable execution evidence paired with one workspace operation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WorkspaceTranscriptExecution {
    pub schema_version: u16,
    pub operation_id: hi_workspace::OperationId,
    pub assistant_content: Vec<Content>,
    pub calls: Vec<WorkspaceTranscriptCall>,
    pub execution: hi_workspace::ExecutionReport,
}

impl WorkspaceTranscriptExecution {
    pub const SCHEMA_VERSION: u16 = 1;
}

/// Records conversation messages durably. Implementations do their own IO.
pub trait SessionSink: Send {
    /// Stable identifier of the durable session this sink writes to — the
    /// transcript file stem for JSONL sessions. `None` for sinks with no
    /// durable identity (tests, ephemeral runs). Findings-ledger records use
    /// this so post-mortems can point at the exact transcript.
    fn id(&self) -> Option<String> {
        None
    }

    /// The model this session currently runs and its context window, for
    /// remote viewers. Called when the sink is attached and again on
    /// `/provider` switches, so a heartbeat never advertises a stale model.
    /// Default no-op: only sinks that mirror to a remote care.
    fn record_model_context(&mut self, _model: &str, _context_window: Option<u32>) {}

    /// Append `messages` (the ones produced since the last call) to storage.
    fn record(&mut self, messages: &[Message], usage: Usage) -> Result<()>;

    /// Stage the exact result of an admitted workspace operation for a causal
    /// durability commit.
    ///
    /// Local-only sessions never call this hook. A PipeFS host must override
    /// it and durably enqueue the record before returning; the default fails
    /// closed so a remote workspace can never silently settle against an
    /// empty or one-step-behind transcript batch.
    fn stage_workspace_execution(&mut self, _record: &WorkspaceTranscriptExecution) -> Result<()> {
        anyhow::bail!("this session sink cannot stage a workspace execution transcript")
    }

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

    /// Persist whether the remote PipeFS workspace is authoritative for this
    /// session. Last write wins. Implementations that predate PipeFS may keep
    /// the default no-op; the host also maintains a cache-local safety hint.
    fn record_pipefs_mode(&mut self, _enabled: bool) -> Result<()> {
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

    /// Persist plan-drive state plus an evidence-ledger delta. `reset_evidence`
    /// starts a new structural scope; `evidence_add` contains only fixed-size
    /// hashes newly credited in that scope. The default preserves compatibility
    /// with sinks that only store pause/stall state.
    fn record_plan_drive_state(
        &mut self,
        paused: bool,
        stall: u32,
        _reset_evidence: bool,
        _evidence_add: &[String],
    ) -> Result<()> {
        self.record_plan_drive(paused, stall)
    }

    /// Persist plan-drive state together with whether an interruption pause is
    /// consumed by the next real user turn. The compatibility default drops
    /// only that policy bit while retaining the established pause/stall data.
    fn record_plan_drive_state_with_policy(
        &mut self,
        paused: bool,
        stall: u32,
        _resume_on_user_input: bool,
        reset_evidence: bool,
        evidence_add: &[String],
    ) -> Result<()> {
        self.record_plan_drive_state(paused, stall, reset_evidence, evidence_add)
    }

    /// Persist whether plan approval is pending, including drafts, revisions,
    /// and visible or parked cards. The legacy name is retained for storage
    /// compatibility. Reopening or revising keeps the gate set until a final
    /// decision. This is independent from the pause controlled by `/plan pause`.
    fn record_plan_approval_parked(&mut self, _parked: bool) -> Result<()> {
        Ok(())
    }

    /// Persist goal-drive stall. Last write wins. Default no-op.
    fn record_goal_drive(&mut self, _stall: u32) -> Result<()> {
        Ok(())
    }

    /// Goal-drive counterpart to [`Self::record_plan_drive_state`].
    fn record_goal_drive_state(
        &mut self,
        stall: u32,
        _reset_evidence: bool,
        _evidence_add: &[String],
    ) -> Result<()> {
        self.record_goal_drive(stall)
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
