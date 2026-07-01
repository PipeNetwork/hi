//! The verification-in-the-loop subsystem, extracted from `run_turn`.
//!
//! After the model stops calling tools, [`Verifier`] runs the configured
//! pipeline stages in order (cheap compile/typecheck first, then lint, then
//! tests); the first to fail stops the turn and its output is fed back to the
//! model for another attempt, up to `max_rounds`. A passing pipeline ends the
//! turn. The "only verify turns that changed files" gating lives here too — a
//! turn that edited nothing can't have introduced a failure.
//!
//! Extracted so the verify state machine (round counter, outcome) is owned by
//! one small type instead of entangled with the main loop's locals and the
//! `Agent`'s shared mutable fields.

use hi_tools::run_check;

use crate::config::VerifyStage;
use crate::snapshot::{FileFingerprint, SnapshotCache, changed_files_between};
use crate::ui::Ui;

/// The snapshot type the verifier compares against.
pub(crate) type Snapshot = std::collections::BTreeMap<String, FileFingerprint>;

/// The outcome of one verify check.
#[derive(Debug)]
pub(crate) enum VerifyOutcome {
    /// All stages passed — the turn is done.
    Passed,
    /// No files changed since the turn baseline, so verification was skipped
    /// (a turn that edited nothing can't have introduced a failure). `first`
    /// is true only on the first round, so the caller can surface a one-time
    /// "skipped" status.
    SkippedNoChanges { first: bool },
    /// A stage failed; its output is fed back to the model. The caller records
    /// the nudge and loops. Carries the 1-based round number.
    Failed {
        stage: VerifyStage,
        output: String,
        round: u32,
    },
    /// Verification didn't run: no stages configured, or the round cap was
    /// already reached.
    NotRun,
}

/// Owns the verify state machine for one turn: the configured stages, the
/// round cap, and the current round counter.
pub(crate) struct Verifier {
    stages: Vec<VerifyStage>,
    max_rounds: u32,
    round: u32,
}

impl Verifier {
    /// Construct from the agent's config. `stages` empty means verification is
    /// off; `max_rounds` caps the retry rounds.
    pub(crate) fn new(stages: Vec<VerifyStage>, max_rounds: u32) -> Self {
        Self {
            stages,
            max_rounds,
            round: 0,
        }
    }

    /// Whether any verification stage is configured.
    #[allow(dead_code)]
    pub(crate) fn is_on(&self) -> bool {
        !self.stages.is_empty()
    }

    /// The current round (0 before any verify run, 1-based after).
    #[allow(dead_code)]
    pub(crate) fn round(&self) -> u32 {
        self.round
    }

    /// Run one verification check against the current workspace snapshot,
    /// compared to the turn baseline. Gates on file changes: if nothing
    /// changed, returns [`VerifyOutcome::SkippedNoChanges`] (and does NOT
    /// consume a round). Otherwise runs the stages in order and returns the
    /// first failure, or [`VerifyOutcome::Passed`].
    ///
    /// `snapshot_cache` is invalidated-on-mutation cache the verifier reads
    /// through; the caller passes the turn baseline separately.
    pub(crate) async fn check(
        &mut self,
        turn_snapshot: &Snapshot,
        snapshot_cache: &mut SnapshotCache,
        ui: &mut dyn Ui,
    ) -> VerifyOutcome {
        if self.stages.is_empty() || self.round >= self.max_rounds {
            return VerifyOutcome::NotRun;
        }
        let current = snapshot_cache.get().await;
        let changed_files = changed_files_between(turn_snapshot, &current);
        if changed_files.is_empty() {
            let first = self.round == 0;
            return VerifyOutcome::SkippedNoChanges { first };
        }
        self.round += 1;
        let round = self.round;
        let max_rounds = self.max_rounds;

        // LSP fast path: if enabled, check diagnostics on changed files before
        // running any shell stages. This catches type errors in ~1s instead of
        // a full `cargo test`/build, and gives line-level errors.
        if hi_tools::lsp_enabled().await
            && let Some(mgr) = hi_tools::lsp_manager_handle()
        {
            let mut lsp_errors = Vec::new();
            for file in &changed_files {
                let path = std::path::Path::new(file);
                if let Ok(text) = tokio::fs::read_to_string(path).await {
                    let _ = mgr.sync_document(path, &text).await;
                }
                if let Ok(diags) = mgr.diagnostics(path).await {
                    for d in diags {
                        if d.severity == "error" {
                            lsp_errors.push(format!(
                                "{}:{}:{}: {}",
                                file,
                                d.line + 1,
                                d.col + 1,
                                d.message
                            ));
                        }
                    }
                }
            }
            if !lsp_errors.is_empty() {
                let output = format!(
                    "LSP diagnostics ({} error(s)):\n{}",
                    lsp_errors.len(),
                    lsp_errors.join("\n")
                );
                return VerifyOutcome::Failed {
                    stage: VerifyStage::new("lsp", "diagnostics"),
                    output,
                    round,
                };
            }
        }

        let mut failure = None;
        for stage in &self.stages {
            ui.status(&format!(
                "verifying ({round}/{max_rounds}) · {}: {}",
                stage.name, stage.command
            ));
            let (passed, output) = run_check(&stage.command).await;
            if !passed {
                failure = Some((stage.clone(), output));
                break;
            }
        }
        match failure {
            None => VerifyOutcome::Passed,
            Some((stage, output)) => VerifyOutcome::Failed {
                stage,
                output,
                round,
            },
        }
    }
}

/// Tailor the failure guidance to the stage kind: test failures imply a rule
/// to infer; compile/lint errors point at a root cause to fix first. Used by
/// the caller when building the verify nudge body.
pub(crate) fn stage_guidance(stage: &VerifyStage) -> &'static str {
    if stage.is_test() {
        "These checks define the exact required behavior. Compare the expected \
         and actual values to infer the precise rule — including edge cases and \
         tie-breaking — then make the smallest edit that satisfies every case."
    } else {
        "Read the error above and fix its root cause (a type, name, or syntax \
         problem) before anything else — the later stages can't run until this \
         passes."
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_is_off_when_no_stages() {
        let v = Verifier::new(Vec::new(), 2);
        assert!(!v.is_on());
        assert_eq!(v.round(), 0);
    }

    #[test]
    fn verifier_is_on_with_stages() {
        let v = Verifier::new(vec![VerifyStage::new("check", "true")], 2);
        assert!(v.is_on());
    }

    #[test]
    fn stage_guidance_differs_tests_vs_compile() {
        let test_stage = VerifyStage::new("test", "pytest");
        let compile_stage = VerifyStage::new("check", "cargo check");
        assert_ne!(stage_guidance(&test_stage), stage_guidance(&compile_stage));
        assert!(stage_guidance(&test_stage).contains("required behavior"));
        assert!(stage_guidance(&compile_stage).contains("root cause"));
    }
}
