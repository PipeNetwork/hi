//! Headless `hi --spec-review` and the plain REPL's `/review` chaining.
//!
//! The drive itself lives in `hi_harness::ReviewDrive`; this module runs its
//! turns back to back with a `Ui`, renders the `review` report object, and
//! maps the final state to an exit code. (`--review` was already taken by the
//! independent-review policy, hence `--spec-review` for the headless entry.)

use std::fmt;

use anyhow::Result;
use hi_harness::{
    Harness, ReviewDrive, ReviewPhase, ReviewStep, TurnCancellation, TurnOutcome, Ui,
    review_take_all,
};

/// Coverage complete and no P0/P1 findings.
pub const EXIT_COMPLETE: i32 = 0;
/// Harness error (request failure) before the loop settled.
pub const EXIT_HARNESS_ERROR: i32 = 1;
/// Usage error: an explicit plan/spec path does not exist.
pub const EXIT_USAGE: i32 = 2;
/// Loop finished; plan/spec items are partial or missing.
pub const EXIT_INCOMPLETE_COVERAGE: i32 = 3;
/// Loop finished; P0/P1 findings are still open (audit-only or capped).
pub const EXIT_OPEN_DEFECTS: i32 = 4;
/// Loop stopped early: no progress, pass cap, unparseable verdict, turn error.
pub const EXIT_STOPPED: i32 = 5;

/// Default fix passes when `--spec-review-passes` is not given.
pub const DEFAULT_PASSES: u32 = hi_harness::DEFAULT_REVIEW_PASSES;

/// Nonzero `hi --spec-review` result carried through `anyhow` so `main` can
/// exit with the mapped code after a clean shutdown.
#[derive(Debug)]
pub struct ReviewExit {
    pub code: i32,
    pub summary: String,
}

impl fmt::Display for ReviewExit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (exit {})", self.summary, self.code)
    }
}

impl std::error::Error for ReviewExit {}

/// Result of a headless review run.
pub struct ReviewRun {
    pub drive: ReviewDrive,
    /// The last turn's outcome, for the turn report.
    pub last_outcome: Option<TurnOutcome>,
    /// Model turns run, including format re-asks.
    pub turns: u32,
}

impl ReviewRun {
    pub fn exit_code(&self) -> i32 {
        exit_code(&self.drive)
    }

    /// `Ok(())` on exit 0, otherwise a [`ReviewExit`] with the mapped code.
    pub fn into_result(self) -> Result<()> {
        let code = self.exit_code();
        if code == EXIT_COMPLETE {
            return Ok(());
        }
        Err(ReviewExit {
            code,
            summary: exit_summary(&self.drive),
        }
        .into())
    }
}

/// Exit code for a settled drive. Open P0/P1 outranks incomplete coverage.
pub fn exit_code(drive: &ReviewDrive) -> i32 {
    match drive.phase {
        ReviewPhase::Done if !drive.open_blocking().is_empty() => EXIT_OPEN_DEFECTS,
        ReviewPhase::Done if !drive.coverage_complete() => EXIT_INCOMPLETE_COVERAGE,
        ReviewPhase::Done => EXIT_COMPLETE,
        ReviewPhase::Stopped => EXIT_STOPPED,
        ReviewPhase::Idle | ReviewPhase::Audit | ReviewPhase::Fix | ReviewPhase::Reaudit => {
            EXIT_HARNESS_ERROR
        }
    }
}

fn exit_summary(drive: &ReviewDrive) -> String {
    match exit_code(drive) {
        EXIT_OPEN_DEFECTS => format!(
            "spec review: {} P0/P1 finding(s) still open",
            drive.open_blocking().len()
        ),
        EXIT_INCOMPLETE_COVERAGE => {
            let (implemented, total) = drive.coverage_counts();
            if total == 0 {
                format!(
                    "spec review: no coverage rows reported for {}",
                    drive.inputs.summary()
                )
            } else {
                format!("spec review: {implemented}/{total} plan/spec items implemented")
            }
        }
        EXIT_STOPPED => format!(
            "spec review stopped: {}",
            drive.stop_reason.as_deref().unwrap_or("stopped")
        ),
        EXIT_COMPLETE => "spec review complete".to_string(),
        _ => "spec review did not finish".to_string(),
    }
}

/// The `review` object for `--report`.
pub fn review_report_json(drive: &ReviewDrive) -> serde_json::Value {
    let verdict = drive.last_verdict.as_ref();
    serde_json::json!({
        "inputs": {
            "files": drive.inputs.files,
            "scope": drive.inputs.scope,
            "git": drive.inputs.git,
            "chunks": drive.inputs.chunks,
            "notice": drive.inputs.notice,
        },
        "phase": drive.phase.label(),
        "audit_only": drive.audit_only,
        "passes": drive.pass,
        "max_passes": drive.max_passes,
        "verdict": verdict.map(|v| if v.complete { "complete" } else { "incomplete" }),
        "coverage_complete": drive.coverage_complete(),
        "coverage": verdict.map(|v| v.coverage.clone()).unwrap_or_default(),
        "findings": verdict.map(|v| v.findings.clone()).unwrap_or_default(),
        "open_blocking": drive.open_blocking().len(),
        "residual": verdict.and_then(|v| v.residual.clone()),
        "unverified_citations": drive.unverified_citations,
        "changed_files": drive.all_changed_files,
        "audit_changed_files": drive.audit_changed_files,
        "fix_turn_error": drive.fix_turn_error,
        "stop_reason": drive.stop_reason,
        "summary": drive.status_line(),
        "exit_code": exit_code(drive),
    })
}

/// Run the whole audit -> fix -> re-audit loop with `ui`, printing the
/// summary and the coverage/findings report to stdout at the end. `paths`
/// are the `--spec-review` values; `all` among them chunks the whole repo.
pub async fn run_headless(
    harness: &mut Harness,
    ui: &mut dyn Ui,
    paths: &[String],
    max_passes: u32,
    audit_only: bool,
) -> Result<ReviewRun> {
    let mut paths = paths.to_vec();
    let all = review_take_all(&mut paths);
    let mut prompt = match harness.begin_review(&paths, all, audit_only, max_passes) {
        Ok(prompt) => prompt,
        Err(message) => {
            return Err(ReviewExit {
                code: EXIT_USAGE,
                summary: message,
            }
            .into());
        }
    };
    // The turn itself announces the status line; only the notice is extra.
    if let Some(notice) = &harness.review_drive().inputs.notice {
        eprintln!("spec review: {notice}");
    }
    let mut turns = 0u32;
    let last_outcome;
    loop {
        let outcome = harness
            .run_turn_cancellable(&prompt, ui, TurnCancellation::new())
            .await?;
        turns += 1;
        let step = harness.review_next_step(&outcome, ui);
        match step {
            ReviewStep::RunPrompt(next) => prompt = next,
            ReviewStep::Done(summary) | ReviewStep::Stopped(summary) => {
                println!("{summary}");
                for line in harness.review_drive().report_lines() {
                    println!("{line}");
                }
                last_outcome = Some(outcome);
                break;
            }
            ReviewStep::Paused(text) => {
                // No interactive resume in headless mode: treat as stopped.
                println!("{text}");
                println!("{}", harness.stop_review());
                last_outcome = Some(outcome);
                break;
            }
            ReviewStep::Idle => {
                last_outcome = Some(outcome);
                break;
            }
        }
    }
    Ok(ReviewRun {
        drive: harness.review_drive().clone(),
        last_outcome,
        turns,
    })
}

/// Plain REPL: after a turn, keep running review turns while the drive asks
/// for them. `None` means the turn future failed before an outcome.
pub async fn chain(harness: &mut Harness, ui: &mut dyn Ui, outcome: Option<TurnOutcome>) {
    if !harness.review_drive().is_active() {
        return;
    }
    let Some(mut outcome) = outcome else {
        println!("{}", harness.stop_review());
        return;
    };
    loop {
        match harness.review_next_step(&outcome, ui) {
            ReviewStep::RunPrompt(prompt) => {
                match harness
                    .run_turn_cancellable(&prompt, ui, TurnCancellation::new())
                    .await
                {
                    Ok(next) => outcome = next,
                    Err(err) => {
                        eprintln!("{err:#}");
                        println!("{}", harness.stop_review());
                        return;
                    }
                }
            }
            ReviewStep::Done(text) | ReviewStep::Stopped(text) | ReviewStep::Paused(text) => {
                println!("{text}");
                if harness.review_drive().phase != ReviewPhase::Fix {
                    for line in harness.review_drive().report_lines() {
                        println!("{line}");
                    }
                }
                return;
            }
            ReviewStep::Idle => return,
        }
    }
}

#[cfg(test)]
#[path = "review_cli_tests.rs"]
mod tests;
