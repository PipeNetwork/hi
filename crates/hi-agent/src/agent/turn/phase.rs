//! Explicit turn-loop phases.
//!
//! `run_turn` still owns the big async control flow, but every major region of
//! that method maps to a [`TurnPhase`]. Prefer matching on this enum (or naming
//! locals/helpers after it) over inventing new boolean "force next" flags when
//! adding behavior — the goal is a readable state machine, not a flag soup.
//!
//! Pipeline (see `docs/architecture.md`):
//!
//! ```text
//! Setup → ( Model → Tools → Steer )* → WorkspaceRepair → Settle → Finalize → Done
//! ```
//!
//! Two distinct "repair" concepts touch this pipeline:
//! - [`TurnPhase::WorkspaceRepair`] — compile/lint/test via
//!   [`crate::verify::WorkspaceRepairVerifier`]; failures re-enter Model.
//! - Review-answer repair — quality nudges inside [`TurnPhase::Steer`] via
//!   [`crate::steering::ReviewRepairMode`] / `ReviewRepairState`; never runs
//!   shell stages.

/// Major phase of one interactive agent turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TurnPhase {
    /// Per-turn caches, ledger baseline, task contract, plan preserve, verifier construct.
    Setup,
    /// Build `ChatRequest`, stream the model, handle provider retries.
    Model,
    /// Execute tool calls (parallel/serial), record effects and progress.
    Tools,
    /// Post-tool / post-text policy: inspection sprawl, review repair, implementation incomplete, etc.
    Steer,
    /// Run [`crate::verify::WorkspaceRepairVerifier`] stages; may loop back to [`Self::Model`].
    WorkspaceRepair,
    /// Seal checkpoint, reconcile ledger, keep or wipe a green verify.
    Settle,
    /// Tool-free recap call and usage/steer lines (optional).
    Finalize,
    /// Assemble [`crate::TurnOutcome`] and clear turn-scoped agent fields.
    Done,
}

impl TurnPhase {
    /// Stable snake_case label for telemetry / status lines.
    #[allow(dead_code)]
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::Model => "model",
            Self::Tools => "tools",
            Self::Steer => "steer",
            Self::WorkspaceRepair => "workspace_repair",
            Self::Settle => "settle",
            Self::Finalize => "finalize",
            Self::Done => "done",
        }
    }
}

/// How the model/tool inner loop should proceed after a steering or verify decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Reserved for extracting steer/verify branches from `run_turn`.
pub(crate) enum LoopControl {
    /// Another model round (possibly after a nudge).
    ContinueModel,
    /// Leave the model/tool loop; enter settle/finalize.
    BreakTurn,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_stable_wire_values() {
        assert_eq!(TurnPhase::WorkspaceRepair.label(), "workspace_repair");
        assert_eq!(TurnPhase::Steer.label(), "steer");
    }
}
