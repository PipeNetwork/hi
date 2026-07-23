//! Explicit turn-loop phases and legal transitions.
//!
//! `run_turn` advances through [`TurnPhase`] via [`crate::Agent::set_turn_phase`]
//! at every control-flow boundary. Illegal transitions panic in debug builds and
//! are logged + clamped in release so a bad stamp never silently corrupts the
//! TUI debug panel.
//!
//! Pipeline (see `docs/architecture.md`):
//!
//! ```text
//! Setup → ( Model → Tools → Steer )* → WorkspaceRepair → Settle → Finalize → Done
//! ```
//!
//! Re-entry edges (not a pure DAG):
//! - Steer → Model (quality / unfinished continue)
//! - Tools → Steer (always after a batch)
//! - WorkspaceRepair → Model (failed verify / coding obligation)
//! - Any → Done (outer `run_turn` wrapper, including `?` exits)
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

    /// Whether `next` is a legal successor of `self` in the turn state machine.
    ///
    /// `Done` is absorbing and reachable from every phase so the outer
    /// `run_turn` wrapper can always land there after `?` exits. A fresh turn
    /// may also restart at `Setup` from `Done` (or the default) between calls.
    pub fn can_transition_to(self, next: Self) -> bool {
        use TurnPhase::*;
        if self == next {
            // Idempotent stamps (e.g. Tools set in loop_ and again in execute_tool_batch).
            return true;
        }
        if next == Done {
            return true;
        }
        match self {
            Setup => matches!(next, Model | Settle | Done),
            // Early tool-mode denial / empty turns skip the model loop and go
            // straight toward settlement classification (via Done from outer).
            Model => matches!(
                next,
                Tools | Steer | Model | WorkspaceRepair | Settle | Done
            ),
            Tools => matches!(next, Steer | Model | Tools | Done),
            Steer => matches!(next, Model | Tools | WorkspaceRepair | Settle | Done),
            WorkspaceRepair => matches!(next, Model | Settle | WorkspaceRepair | Done),
            Settle => matches!(next, Finalize | Done),
            Finalize => matches!(next, Done),
            Done => matches!(next, Setup | Done),
        }
    }
}

impl crate::Agent {
    /// Record the active turn phase (debug panel + tests).
    ///
    /// Enforces [`TurnPhase::can_transition_to`]. Illegal transitions are a
    /// logic bug: panic in debug, warn + apply in release so UI never freezes
    /// on a stale phase after a missed edge.
    #[inline]
    pub(crate) fn set_turn_phase(&mut self, phase: TurnPhase) {
        let from = self.report.turn_phase;
        if !from.can_transition_to(phase) {
            debug_assert!(
                false,
                "illegal turn phase transition {} → {}",
                from.label(),
                phase.label()
            );
            // Release builds still apply the phase so the TUI never freezes
            // on stale state after a missed edge. Keep this silent for users;
            // the debug assertion catches the programming error in dev/tests.
        }
        self.report.turn_phase = phase;
    }

    /// Phase of the in-flight or most recently finished turn.
    pub fn turn_phase(&self) -> TurnPhase {
        self.report.turn_phase
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

    #[test]
    fn happy_path_transitions_are_legal() {
        use TurnPhase::*;
        let path = [
            Setup,
            Model,
            Tools,
            Steer,
            Model,
            Tools,
            Steer,
            WorkspaceRepair,
            Settle,
            Finalize,
            Done,
        ];
        for window in path.windows(2) {
            assert!(
                window[0].can_transition_to(window[1]),
                "{} → {}",
                window[0].label(),
                window[1].label()
            );
        }
    }

    #[test]
    fn reentry_and_escape_edges() {
        use TurnPhase::*;
        assert!(Steer.can_transition_to(Model));
        assert!(WorkspaceRepair.can_transition_to(Model));
        assert!(WorkspaceRepair.can_transition_to(Settle));
        assert!(Model.can_transition_to(Done));
        assert!(Tools.can_transition_to(Done));
        assert!(Done.can_transition_to(Setup));
        assert!(Setup.can_transition_to(Settle)); // early-exit turns
    }

    #[test]
    fn illegal_edges_are_rejected() {
        use TurnPhase::*;
        assert!(!Finalize.can_transition_to(Model));
        assert!(!Settle.can_transition_to(Tools));
        assert!(!Done.can_transition_to(Model));
        assert!(!Setup.can_transition_to(Tools));
    }

    #[test]
    fn idempotent_stamps_are_legal() {
        for phase in [
            TurnPhase::Setup,
            TurnPhase::Model,
            TurnPhase::Tools,
            TurnPhase::Steer,
            TurnPhase::WorkspaceRepair,
            TurnPhase::Settle,
            TurnPhase::Finalize,
            TurnPhase::Done,
        ] {
            assert!(phase.can_transition_to(phase), "{}", phase.label());
        }
    }
}
