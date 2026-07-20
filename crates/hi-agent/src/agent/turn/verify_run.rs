//! Run one **workspace** repair-verification check ([`TurnPhase::WorkspaceRepair`]).

use anyhow::Result;

use super::phase::TurnPhase;
use crate::ui::Ui;
use crate::verify::{Snapshot, VerifyOutcome, VerifyWorkspace, WorkspaceRepairVerifier};

impl crate::Agent {
    /// Kill turn-scoped background processes, reconcile the ledger, and run the
    /// configured [`WorkspaceRepairVerifier`] stages ([`TurnPhase::WorkspaceRepair`]).
    /// Returns [`VerifyOutcome::NotRun`] when verification is off.
    pub(super) async fn run_workspace_repair_verification(
        &mut self,
        verifier: &mut WorkspaceRepairVerifier,
        turn_background_baseline: &[String],
        turn_snapshot: &mut Option<Snapshot>,
        turn_checkpoint_created: bool,
        turn_ledger_revision: u64,
        fast_feedback: &super::fast_feedback::FastFeedbackState,
        ui: &mut dyn Ui,
    ) -> Result<VerifyOutcome> {
        // Caller stamps WorkspaceRepair before invoking; keep phase sticky here.
        debug_assert_eq!(self.turn_phase(), TurnPhase::WorkspaceRepair);
        let killed_backgrounds = self
            .runtime
            .background()
            .kill_started_after(turn_background_baseline);
        if killed_backgrounds > 0 {
            ui.status(&format!(
                "stopped {killed_backgrounds} live background process(es) before final verification"
            ));
            // Process-group termination is signalled synchronously. Yield so
            // the driver tasks can observe it before the final filesystem
            // reconciliation and verifier snapshot.
            tokio::task::yield_now().await;
            self.invalidate_snapshot();
            self.reconcile_workspace_changes()?;
        }
        if !verifier.is_on() {
            return Ok(VerifyOutcome::NotRun);
        }
        let baseline = self.ensure_turn_snapshot(turn_snapshot).await?;
        let pre_turn_checkpoint = turn_checkpoint_created
            .then(|| self.workspace.checkpoints.last())
            .flatten()
            .and_then(|reference| {
                hi_tools::checkpoint::parse_reference(reference)
                    .ok()
                    .map(|(target, _)| target.to_string())
            });
        let lsp = self.runtime.lsp();
        self.reconcile_workspace_changes()?;
        let (ledger_touched_files, ledger_mutation_seen, current_revision) = {
            let ledger = self.runtime.ledger();
            (
                ledger.touched_paths_since(turn_ledger_revision),
                ledger.had_mutation_since(turn_ledger_revision),
                ledger.revision(),
            )
        };
        // Phase I: packages mid-turn already sealed green at this revision —
        // WorkspaceRepair drops matching affected-check/test stages.
        let skip_checks = fast_feedback.skippable_check_packages(current_revision);
        let skip_tests = fast_feedback.skippable_test_packages(current_revision);
        let workspace = VerifyWorkspace::new(
            self.runtime.root(),
            self.runtime.state_root(),
            pre_turn_checkpoint.as_deref(),
            &lsp,
        )
        .with_changed_files(&ledger_touched_files)
        .with_mutation_seen(ledger_mutation_seen)
        .with_skippable_affected(&skip_checks, &skip_tests);
        Ok(verifier
            .check(&workspace, &baseline, &mut self.snapshot_cache, ui)
            .await)
    }
}
