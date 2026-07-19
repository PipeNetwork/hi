//! Run one repair-verification check against the current workspace state.

use anyhow::Result;

use crate::ui::Ui;
use crate::verify::{RepairVerifier, Snapshot, VerifyOutcome, VerifyWorkspace};

impl crate::Agent {
    /// Kill turn-scoped background processes, reconcile the ledger, and run the
    /// configured [`RepairVerifier`] stages. Returns [`VerifyOutcome::NotRun`]
    /// when verification is off.
    pub(super) async fn run_repair_verification(
        &mut self,
        verifier: &mut RepairVerifier,
        turn_background_baseline: &[String],
        turn_snapshot: &mut Option<Snapshot>,
        turn_checkpoint_created: bool,
        turn_ledger_revision: u64,
        ui: &mut dyn Ui,
    ) -> Result<VerifyOutcome> {
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
            .then(|| self.checkpoints.last())
            .flatten()
            .and_then(|reference| {
                hi_tools::checkpoint::parse_reference(reference)
                    .ok()
                    .map(|(target, _)| target.to_string())
            });
        let lsp = self.runtime.lsp();
        self.reconcile_workspace_changes()?;
        let (ledger_touched_files, ledger_mutation_seen) = {
            let ledger = self.runtime.ledger();
            (
                ledger.touched_paths_since(turn_ledger_revision),
                ledger.had_mutation_since(turn_ledger_revision),
            )
        };
        let workspace = VerifyWorkspace::new(
            self.runtime.root(),
            self.runtime.state_root(),
            pre_turn_checkpoint.as_deref(),
            &lsp,
        )
        .with_changed_files(&ledger_touched_files)
        .with_mutation_seen(ledger_mutation_seen);
        Ok(verifier
            .check(&workspace, &baseline, &mut self.snapshot_cache, ui)
            .await)
    }
}
