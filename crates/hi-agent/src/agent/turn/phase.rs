//! Explicit turn-loop phases.
//!
//! `run_turn` stamps [`TurnPhase`] onto [`crate::Agent::turn_phase`] at every
//! control-flow boundary. The public getter is read by the TUI debug panel and
//! is safe after errors: the outer `run_turn` wrapper always finishes on
//! [`TurnPhase::Done`].
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TurnPhase {
    /// Per-turn caches, ledger baseline, task contract, plan preserve, verifier construct.
    #[default]
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
    /// Stable snake_case label for telemetry / status lines / debug panel.
    pub fn label(self) -> &'static str {
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

impl crate::Agent {
    /// Record the active turn phase (debug panel + tests).
    #[inline]
    pub(crate) fn set_turn_phase(&mut self, phase: TurnPhase) {
        self.turn_phase = phase;
    }

    /// Phase of the in-flight or most recently finished turn.
    pub fn turn_phase(&self) -> TurnPhase {
        self.turn_phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_are_stable_wire_values() {
        assert_eq!(TurnPhase::WorkspaceRepair.label(), "workspace_repair");
        assert_eq!(TurnPhase::Steer.label(), "steer");
        assert_eq!(TurnPhase::Done.label(), "done");
    }

    #[test]
    fn pipeline_order_matches_architecture_doc() {
        let order = [
            TurnPhase::Setup,
            TurnPhase::Model,
            TurnPhase::Tools,
            TurnPhase::Steer,
            TurnPhase::WorkspaceRepair,
            TurnPhase::Settle,
            TurnPhase::Finalize,
            TurnPhase::Done,
        ];
        let labels: Vec<_> = order.iter().map(|p| p.label()).collect();
        assert_eq!(
            labels,
            [
                "setup",
                "model",
                "tools",
                "steer",
                "workspace_repair",
                "settle",
                "finalize",
                "done",
            ]
        );
    }

    #[test]
    fn done_is_terminal_label() {
        assert_eq!(TurnPhase::Done.label(), "done");
    }
}
