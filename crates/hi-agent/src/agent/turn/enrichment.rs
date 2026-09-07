//! Publish verifier-backed workspace learning before the absorbing settlement.

use anyhow::Result;

use crate::domain::VerifyEvidence;
use crate::verify::is_internal_runtime_artifact_path;
use crate::{ReviewStatus, Ui};

use super::state::TurnState;

pub(super) struct EnrichmentOutcome {
    pub inputs_changed: bool,
    /// This is a post-maintenance guard, not a replacement review seal. The
    /// original task diff must remain untouched through deterministic recheck.
    pub preserved_review_at: Option<String>,
}

impl crate::Agent {
    /// A successful stage permits the optional enrichment writers. Their writes
    /// do not inherit that stage's seal: changed canonical input needs another
    /// deterministic pass before settlement, under the same verifier ceiling.
    pub(super) async fn enrich_verified_turn(
        &mut self,
        turn: &mut TurnState,
        verified_revision: u64,
        verified_digest: String,
        ui: &mut dyn Ui,
    ) -> Result<EnrichmentOutcome> {
        self.report.verify = VerifyEvidence::pass(verified_revision, verified_digest.clone());
        self.runtime
            .ledger()
            .retain_verification_baseline(verified_revision);
        self.reconcile_workspace_changes().await?;
        let changes = self
            .runtime
            .ledger()
            .changes_since(turn.turn_ledger_revision);
        let reviewed_changes = changes.clone();
        let before_writers = self.runtime.ledger().revision();
        let review_was_current = turn.independent_review_status == ReviewStatus::Passed
            && self.runtime.ledger().workspace_revision() == verified_digest;
        let reviewed_content = if review_was_current {
            crate::hygiene::reviewable_content_revision(
                self.runtime.root(),
                &reviewed_changes,
                self.turn_cancellation.clone(),
            )
            .await
        } else {
            None
        };
        self.workspace.last_file_changes = changes
            .into_iter()
            .filter(|change| !is_internal_runtime_artifact_path(&change.path))
            .collect();
        self.workspace.last_changed_files = self
            .workspace
            .last_file_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        if self.workspace.last_changed_files.is_empty() {
            return Ok(EnrichmentOutcome {
                inputs_changed: false,
                preserved_review_at: None,
            });
        }

        let mut published_paths = Vec::new();
        if self.config.memory.curate_skills && !self.task_recovery.exhausted {
            published_paths.extend(self.curate_turn_end(turn.turn_start, ui).await);
        }
        published_paths.extend(self.record_coding_facts_turn_end(ui).await);
        self.reconcile_workspace_changes().await?;
        let current_digest = self.runtime.ledger().workspace_revision();
        let changed_by_writers = self.runtime.ledger().changes_since(before_writers);
        let only_owned_metadata = !changed_by_writers.is_empty()
            && changed_by_writers.iter().all(|change| {
                let path = self.runtime.root().join(&change.path);
                published_paths.contains(&path)
                    && !reviewed_changes
                        .iter()
                        .any(|prior| prior.path == change.path)
                    && !turn.task_contract.referenced_paths.contains(&change.path)
            });
        // Completion review covered the task diff. Preserve that review only
        // when acknowledged maintenance changed separate metadata paths and
        // the exact reviewed bytes remain unchanged. Source edits, explicitly
        // requested metadata, and inconclusive reads still invalidate it.
        let preserve_review = if reviewed_content.is_some() && only_owned_metadata {
            crate::hygiene::reviewable_content_revision(
                self.runtime.root(),
                &reviewed_changes,
                self.turn_cancellation.clone(),
            )
            .await
                == reviewed_content
        } else {
            false
        };
        let mut review_after_reconciliation = turn.independent_review_status;
        let changed = super::settlement::reconcile_verified_revision(
            &mut self.report.verify,
            &mut review_after_reconciliation,
            current_digest.clone(),
            ui,
        );
        if !preserve_review {
            turn.independent_review_status = review_after_reconciliation;
        }
        if changed {
            ui.status(
                "workspace learning changed verification inputs; checking the final revision",
            );
        }
        Ok(EnrichmentOutcome {
            inputs_changed: changed,
            preserved_review_at: (changed && preserve_review).then_some(current_digest),
        })
    }
}
