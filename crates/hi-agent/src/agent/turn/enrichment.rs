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
    /// A successful stage permits the optional enrichment writers. Prose-only
    /// owned memory/skill files rebind the seal without another suite run. Any
    /// other canonical-input change — including files that landed after the
    /// attested digest — still needs another deterministic pass.
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
        let post_verify_changes = self.runtime.ledger().changes_since(verified_revision);
        let only_owned_metadata = changes_are_owned_metadata(
            changed_by_writers.iter().map(|change| change.path.as_str()),
            &published_paths,
            &reviewed_changes,
            &turn.task_contract.referenced_paths,
            self.runtime.root(),
        );
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
        let stages = self
            .config
            .gates
            .verification
            .resolved_stages(self.runtime.root());
        let recheck = enrichment_requires_revalidation(
            post_verify_changes
                .iter()
                .map(|change| change.path.as_str()),
            &stages,
        );
        // Inspect every path that moved after the attested digest, not just the
        // enrichment writers. A source file that lands during review must not
        // inherit the previous green seal.
        let only_owned_post_verify = changes_are_owned_metadata(
            post_verify_changes
                .iter()
                .map(|change| change.path.as_str()),
            &published_paths,
            &reviewed_changes,
            &turn.task_contract.referenced_paths,
            self.runtime.root(),
        );
        let changed = if !recheck && only_owned_post_verify {
            // Skill/memory prose is not checked input for cargo test. Rebind
            // the seal instead of a second suite run the model cannot repair.
            let revision = self.runtime.ledger().revision();
            self.report.verify = VerifyEvidence::pass(revision, current_digest.clone());
            self.runtime.ledger().retain_verification_baseline(revision);
            ui.status("workspace learning did not change checked verification inputs");
            false
        } else {
            super::settlement::reconcile_verified_revision(
                &mut self.report.verify,
                &mut review_after_reconciliation,
                current_digest.clone(),
                ui,
            )
        };
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

fn changes_are_owned_metadata<'a>(
    paths: impl IntoIterator<Item = &'a str>,
    published_paths: &[std::path::PathBuf],
    reviewed_changes: &[hi_tools::FileChange],
    referenced_paths: &[String],
    root: &std::path::Path,
) -> bool {
    let mut any = false;
    for path in paths {
        any = true;
        let absolute = root.join(path);
        if !published_paths.contains(&absolute)
            || reviewed_changes.iter().any(|prior| prior.path == path)
            || referenced_paths.iter().any(|referenced| referenced == path)
        {
            return false;
        }
    }
    any
}

fn enrichment_requires_revalidation<'a>(
    paths: impl IntoIterator<Item = &'a str>,
    stages: &[crate::config::VerifyStage],
) -> bool {
    let paths: Vec<String> = paths
        .into_iter()
        .map(|path| path.replace('\\', "/"))
        .collect();
    if paths.is_empty() {
        return false;
    }
    if paths.iter().any(|path| {
        !crate::verify::is_prose_only_path(path)
            && !crate::verify::is_internal_runtime_artifact_path(path)
    }) {
        return true;
    }
    stages.iter().any(|stage| {
        paths
            .iter()
            .any(|path| stage.command.contains(path.as_str()))
    })
}

#[cfg(test)]
mod tests {
    use super::enrichment_requires_revalidation;
    use crate::config::VerifyStage;

    #[test]
    fn cargo_stages_ignore_prose_memory_writes() {
        let stages = [VerifyStage::new("test", "cargo test --quiet")];
        assert!(!enrichment_requires_revalidation(
            [".hi/memory.md"],
            &stages
        ));
        assert!(enrichment_requires_revalidation(["src/ws.rs"], &stages));
    }

    #[test]
    fn explicit_memory_checks_still_revalidate() {
        let stages = [VerifyStage::new("check", "test ! -e .hi/memory.md")];
        assert!(enrichment_requires_revalidation([".hi/memory.md"], &stages));
    }

    #[test]
    fn source_that_lands_with_memory_still_requires_revalidation() {
        let stages = [VerifyStage::new("test", "true")];
        assert!(enrichment_requires_revalidation(
            [".hi/memory.md", "late.rs"],
            &stages
        ));
    }
}
