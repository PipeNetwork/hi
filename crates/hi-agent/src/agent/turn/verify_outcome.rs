//! React to one [`VerifyOutcome`] from workspace repair.
//!
//! Extracted from the main turn loop so orchestration stays thin routing:
//! run verify → handle outcome → re-enter Model or leave the `'turn` loop.

use std::collections::BTreeSet;

use anyhow::Result;

use crate::transcript::NudgeKind;
use crate::verify::{VerifyOutcome, WorkspaceRepairVerifier, stage_guidance};
use crate::{ReviewStatus, Ui};

use super::helpers::fallback_review_line_count;

/// What the outer `'turn` loop should do after handling a verify outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VerifyOutcomeControl {
    /// Leave the model/verify loop and proceed to Settle.
    BreakTurn,
    /// Re-enter Model → Tools (obligation, failed repair, skeptic objection, …).
    ReenterModel,
}

/// Mutable turn-locals the verify-outcome handler may update.
pub(super) struct VerifyOutcomeState<'a> {
    pub(super) obligation_nudge_fired: &'a mut bool,
    pub(super) force_tools_next: &'a mut bool,
    pub(super) verified_at: &'a mut Option<(u64, String)>,
    pub(super) independent_review_status: &'a mut ReviewStatus,
    pub(super) independent_review_repairs: &'a mut u32,
    pub(super) stalled_unfinished: &'a mut bool,
    pub(super) verification_infrastructure_error: &'a mut bool,
    pub(super) verification_unstable: &'a mut bool,
    pub(super) last_verify_attributions: &'a mut Vec<hi_tools::Attribution>,
    pub(super) ranked_context_paths: &'a mut BTreeSet<String>,
    pub(super) context_generation_seen: &'a mut u64,
    pub(super) indexed_ledger_revision: &'a mut u64,
}

impl crate::Agent {
    /// Apply one workspace-repair [`VerifyOutcome`]. Returns whether the outer
    /// loop should break to Settle or re-enter Model.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_workspace_repair_outcome(
        &mut self,
        outcome: VerifyOutcome,
        verifier: &mut WorkspaceRepairVerifier,
        turn_ledger_revision: u64,
        expected_mutation: bool,
        context_task: &str,
        repository_context_enabled: bool,
        state: &mut VerifyOutcomeState<'_>,
        ui: &mut dyn Ui,
    ) -> Result<VerifyOutcomeControl> {
        match outcome {
            VerifyOutcome::NotRun => {
                // Phase C obligation: one re-entry when a coding turn still
                // owes green evidence (failed budget or never sealed).
                let (changed_now, mutation_now) = {
                    let ledger = self.runtime.ledger();
                    (
                        ledger
                            .changes_since(turn_ledger_revision)
                            .into_iter()
                            .map(|c| c.path)
                            .collect::<Vec<_>>(),
                        ledger.had_mutation_since(turn_ledger_revision),
                    )
                };
                if !*state.obligation_nudge_fired
                    && let Some(reason) = super::obligation::coding_verify_obligation(
                        self.task.last_task_contract.as_ref(),
                        &self.config.gates.verification,
                        expected_mutation,
                        &changed_now,
                        mutation_now,
                        self.report.last_verify,
                        verifier.executions().len(),
                    )
                {
                    match reason {
                        // Never sealed green after a code mutation — one more
                        // model round to run checks / fix. Failed-verify budget
                        // exhaustion already spent its repair rounds above.
                        super::obligation::ObligationReason::UnverifiedMutation => {
                            *state.obligation_nudge_fired = true;
                            ui.status(reason.ui_status());
                            ui.nudge(reason.ui_status());
                            self.messages
                                .push_nudge(NudgeKind::Continue, reason.nudge_body());
                            *state.force_tools_next = true;
                            return Ok(VerifyOutcomeControl::ReenterModel);
                        }
                        super::obligation::ObligationReason::FailedVerify => {
                            *state.stalled_unfinished = true;
                            ui.status(reason.ui_status());
                        }
                    }
                }
                if self.report.last_verify == Some(false) {
                    *state.stalled_unfinished = true;
                    ui.status(
                        "verification still failed after the retry budget; the task may be incomplete. /retry, or send 'continue'.",
                    );
                }
                return Ok(VerifyOutcomeControl::BreakTurn);
            }
            VerifyOutcome::SkippedNoChanges { first } => {
                if first {
                    ui.status("verification skipped — no files changed this turn");
                }
                // Mutation-shaped coding turns that somehow report no file
                // delta still owe evidence when mutation_seen (e.g. restored
                // bytes) or the contract expected edits — one obligation nudge.
                let mutation_now = self
                    .runtime
                    .ledger()
                    .had_mutation_since(turn_ledger_revision);
                if !*state.obligation_nudge_fired
                    && let Some(reason) = super::obligation::coding_verify_obligation(
                        self.task.last_task_contract.as_ref(),
                        &self.config.gates.verification,
                        expected_mutation,
                        &[],
                        mutation_now,
                        self.report.last_verify,
                        verifier.executions().len(),
                    )
                {
                    if matches!(
                        reason,
                        super::obligation::ObligationReason::UnverifiedMutation
                    ) {
                        *state.obligation_nudge_fired = true;
                        ui.status(reason.ui_status());
                        ui.nudge(reason.ui_status());
                        self.messages
                            .push_nudge(NudgeKind::Continue, reason.nudge_body());
                        *state.force_tools_next = true;
                        return Ok(VerifyOutcomeControl::ReenterModel);
                    }
                }
                return Ok(VerifyOutcomeControl::BreakTurn);
            }
            VerifyOutcome::SkippedProseOnly { first } => {
                if first {
                    ui.status("verification skipped — prose-only files changed this turn");
                }
                return Ok(VerifyOutcomeControl::BreakTurn);
            }
            VerifyOutcome::Passed => {
                ui.status("✓ verification passed");
                self.report.set_verify(Some(true));
                self.reconcile_workspace_changes().await?;
                let (verified_revision, verified_digest, current_changes) = {
                    let mut ledger = self.runtime.ledger();
                    (
                        ledger.revision(),
                        ledger.workspace_revision(),
                        ledger.changes_since(turn_ledger_revision),
                    )
                };
                *state.verified_at = Some((verified_revision, verified_digest.clone()));
                let current_files = current_changes
                    .iter()
                    .map(|change| change.path.clone())
                    .collect::<Vec<_>>();
                let mut diff = self.turn_diff().await;
                let diff_lines = if diff.trim().is_empty() {
                    fallback_review_line_count(self.runtime.root(), &current_changes)
                } else {
                    diff.lines().count()
                };
                let (review_required, large_diff_review) = self
                    .task
                    .last_task_contract
                    .as_ref()
                    .map_or((false, false), |contract| {
                        let required = contract.requires_review(
                            self.config.gates.review,
                            &current_files,
                            diff_lines,
                            self.config.subagents.long_horizon
                                || self.config.subagents.write_subagents.is_enabled(),
                        );
                        let large = contract.is_large_mutation(&current_files, diff_lines);
                        (required, large)
                    });
                if review_required {
                    self.refresh_active_task_context(
                        &context_task,
                        repository_context_enabled,
                        turn_ledger_revision,
                        state.ranked_context_paths,
                        state.context_generation_seen,
                        state.indexed_ledger_revision,
                    );
                    if diff.chars().count() > 50_000 {
                        diff = diff.chars().take(50_000).collect();
                        diff.push_str("\n… (bounded review diff truncated)");
                    }
                    let contract = self
                        .task
                        .last_task_contract
                        .as_ref()
                        .and_then(|contract| serde_json::to_string_pretty(contract).ok())
                        .unwrap_or_else(|| "(task contract unavailable)".into());
                    let instructions = self.task.task_context.as_deref().unwrap_or("(none)");
                    let stages = verifier.stages_summary().unwrap_or_else(|| "(none)".into());
                    let context = format!(
                        "Task contract:\n{contract}\n\nScoped instructions and relevant repository context:\n{instructions}\n\nChanged files ({file_count}):\n{files}\n\nDiff size: {diff_lines} lines\nDeterministic verification: PASSED\nStages: {stages}\nVerified workspace revision: {verified_digest}\n\nComplete bounded turn diff:\n{diff}",
                        file_count = current_files.len(),
                        files = current_files.join("\n"),
                    );
                    // Phase L: large multi-file diffs get the hole-focused
                    // skeptic prompt; other risk reviews keep the general one.
                    let review_label = if large_diff_review {
                        "large-diff skeptic"
                    } else {
                        "independent completion review"
                    };
                    ui.status(&format!("running {review_label}"));
                    let verdict = if diff.trim().is_empty() && !current_files.is_empty() {
                        super::super::skeptic::SkepticVerdict::Unavailable(
                            "a complete turn diff was unavailable for the current changes".into(),
                        )
                    } else if large_diff_review {
                        self.large_diff_review(&context).await
                    } else {
                        self.independent_review(&context).await
                    };
                    match verdict {
                        super::super::skeptic::SkepticVerdict::Approve => {
                            *state.independent_review_status = ReviewStatus::Passed;
                            if large_diff_review {
                                ui.status("✓ large-diff skeptic approved");
                            }
                        }
                        super::super::skeptic::SkepticVerdict::Unavailable(reason) => {
                            *state.independent_review_status = ReviewStatus::Unavailable;
                            ui.status(&format!(
                                "{review_label} unavailable after deterministic pass: {reason}"
                            ));
                        }
                        super::super::skeptic::SkepticVerdict::Object(objections)
                            if *state.independent_review_repairs == 0 =>
                        {
                            *state.independent_review_repairs = 1;
                            *state.independent_review_status = ReviewStatus::Objected;
                            self.report.set_verify(None);
                            *state.verified_at = None;
                            verifier.allow_review_revalidation();
                            let headline = if large_diff_review {
                                "Large-diff skeptic found concrete multi-file defects"
                            } else {
                                "Independent review found concrete completion defects"
                            };
                            self.messages.push_nudge(
                                NudgeKind::Review,
                                format!(
                                    "{headline}. Repair them now, then re-run deterministic validation.\n\n{}",
                                    objections
                                        .iter()
                                        .map(|objection| format!("- {objection}"))
                                        .collect::<Vec<_>>()
                                        .join("\n")
                                ),
                            );
                            ui.nudge(&format!(
                                "{review_label} objected; allowing one repair cycle"
                            ));
                            return Ok(VerifyOutcomeControl::ReenterModel);
                        }
                        super::super::skeptic::SkepticVerdict::Object(objections) => {
                            *state.independent_review_status = ReviewStatus::Objected;
                            *state.stalled_unfinished = true;
                            ui.status(&format!(
                                "{review_label} objected again after repair: {}",
                                objections.join("; ")
                            ));
                        }
                        // The independent-review prompt defines no ESCALATE
                        // verdict; treat a stray one as a final objection
                        // (no extra repair cycle — escalation means
                        // retrying can't fix it).
                        super::super::skeptic::SkepticVerdict::Escalate(objections) => {
                            *state.independent_review_status = ReviewStatus::Objected;
                            *state.stalled_unfinished = true;
                            ui.status(&format!(
                                "{review_label} escalated — needs your judgment: {}",
                                objections.join("; ")
                            ));
                        }
                    }
                }
                return Ok(VerifyOutcomeControl::BreakTurn);
            }
            VerifyOutcome::Failed {
                stage,
                output,
                round,
            } => {
                ui.status(&format!("✗ {} failed; iterating", stage.name));
                self.report.set_verify(Some(false));
                *state.verified_at = None;
                let guidance = stage_guidance(&stage);
                // Structured failure: attributions + condensed output + optional
                // diagnostic snippet. Enrich-only relative to the raw blob.
                let structured = hi_tools::format_structured_failure(
                    &format!(
                        "Verification stage `{}` failed (`{}`).",
                        stage.name, stage.command
                    ),
                    &output,
                    Some(guidance),
                );
                *state.last_verify_attributions = structured.attributions.clone();
                // Replace the previous verify nudge instead of accumulating.
                // Only the latest verification output belongs in context.
                // `replace_last_nudge` pops trailing tool/assistant messages
                // from the prior verify cycle and the prior nudge itself
                // (located by typed kind, not string-matching), then pushes
                // the new one. On the first round there's no prior nudge, so
                // nothing is popped — the model's just-finished turn stays.
                self.messages
                    .replace_last_nudge(NudgeKind::Verify { round }, structured.body);
                // Re-enter Model → Tools with the verify nudge in context.
                // The verifier's round counter enforces max_verify_repairs.
                return Ok(VerifyOutcomeControl::ReenterModel);
            }
            VerifyOutcome::InfrastructureError {
                stage,
                output,
                round,
            } => {
                *state.verification_infrastructure_error = true;
                self.report.set_verify(None);
                *state.verified_at = None;
                ui.status(&format!(
                    "verification infrastructure failed at {} (round {round}): {output}",
                    stage.name,
                ));
                return Ok(VerifyOutcomeControl::BreakTurn);
            }
            VerifyOutcome::Unstable {
                stage,
                changed_files,
                round,
            } => {
                *state.verification_unstable = true;
                *state.stalled_unfinished = true;
                self.report.set_verify(Some(false));
                *state.verified_at = None;
                ui.status(&format!(
                    "verification is unstable in round {round}: stage {} modified {}",
                    stage.name,
                    changed_files.join(", ")
                ));
                return Ok(VerifyOutcomeControl::BreakTurn);
            }
        }
    }
}
