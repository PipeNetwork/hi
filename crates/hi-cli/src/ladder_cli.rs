//! Headless `hi --ladder` and the plain REPL's `/ladder` chaining.
//!
//! The ladder itself lives in `hi_harness::Ladder`; this module runs its
//! attempts back to back with a `Ui`, renders the `ladder` report object,
//! and maps the final phase to an exit code.

use std::fmt;

use anyhow::Result;
use hi_harness::{
    Harness, Ladder, LadderCommand, LadderPhase, LadderStep, TurnCancellation, TurnOutcome, Ui,
};

/// The verify command passed.
pub const EXIT_VERIFIED: i32 = 0;
/// Harness error (request failure) before the ladder settled.
pub const EXIT_HARNESS_ERROR: i32 = 1;
/// Usage error: no check to judge attempts, `pipe/auto`, or a bad rung.
pub const EXIT_USAGE: i32 = 2;
/// Every rung failed verification.
pub const EXIT_EXHAUSTED: i32 = 3;
/// Stopped early: a cancelled turn, or no check ran after a turn.
pub const EXIT_STOPPED: i32 = 5;

/// Stop reason when a turn fails before it produces an outcome.
const TURN_FAILED: &str = "the turn failed before it finished";

/// Nonzero `hi --ladder` result carried through `anyhow` so `main` can exit
/// with the mapped code after a clean shutdown.
#[derive(Debug)]
pub struct LadderExit {
    pub code: i32,
    pub summary: String,
}

impl fmt::Display for LadderExit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (exit {})", self.summary, self.code)
    }
}

impl std::error::Error for LadderExit {}

/// Result of a headless ladder run.
pub struct LadderRun {
    pub ladder: Ladder,
    /// The last attempt's outcome, for the turn report.
    pub last_outcome: Option<TurnOutcome>,
}

impl LadderRun {
    /// `Ok(())` when the check passed, otherwise a [`LadderExit`].
    pub fn into_result(self) -> Result<()> {
        let code = exit_code(&self.ladder);
        if code == EXIT_VERIFIED {
            return Ok(());
        }
        Err(LadderExit {
            code,
            summary: self.ladder.status_line(),
        }
        .into())
    }
}

/// Exit code for a settled ladder.
pub fn exit_code(ladder: &Ladder) -> i32 {
    match ladder.phase {
        LadderPhase::Verified => EXIT_VERIFIED,
        LadderPhase::Exhausted => EXIT_EXHAUSTED,
        LadderPhase::Stopped => EXIT_STOPPED,
        LadderPhase::Idle | LadderPhase::Running => EXIT_HARNESS_ERROR,
    }
}

/// The `ladder` object for `--report`.
pub fn ladder_report_json(ladder: &Ladder) -> serde_json::Value {
    let (input_tokens, output_tokens) = ladder.total_tokens();
    let attempts: Vec<serde_json::Value> = ladder
        .attempts
        .iter()
        .map(|attempt| {
            serde_json::json!({
                "rung": attempt.rung + 1,
                "label": attempt.label,
                "result": attempt.result.label(),
                "changed_files": attempt.changed_files,
                "input_tokens": attempt.input_tokens,
                "output_tokens": attempt.output_tokens,
            })
        })
        .collect();
    serde_json::json!({
        "task": ladder.task,
        "phase": ladder.phase.label(),
        "verify": ladder.verify_command,
        "base_model": ladder.base_model,
        "rungs": (0..ladder.rungs.len())
            .map(|index| ladder.rung_label(index))
            .collect::<Vec<_>>(),
        "rung": ladder.rung + 1,
        "start_note": ladder.start_note,
        "attempts": attempts,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stop_reason": ladder.stop_reason,
        "summary": ladder.status_line(),
        "exit_code": exit_code(ladder),
    })
}

/// `/ladder [status|stop|<task>]` in the plain REPL: prints what the command
/// reports and returns the prompt to run when it starts a ladder.
pub async fn command(harness: &mut Harness, arg: &str) -> Option<String> {
    match harness.ladder_command(arg).await {
        LadderCommand::Start { prompt, notes } => {
            for note in notes {
                println!("{note}");
            }
            Some(prompt)
        }
        LadderCommand::Message(text) => {
            println!("{text}");
            None
        }
    }
}

/// Run `task` up the ladder with `ui`, printing the summary and one line
/// per attempt to stdout at the end.
pub async fn run_headless(harness: &mut Harness, ui: &mut dyn Ui, task: &str) -> Result<LadderRun> {
    let notes = harness
        .begin_ladder(task)
        .await
        .map_err(|summary| LadderExit {
            code: EXIT_USAGE,
            summary,
        })?;
    for note in notes {
        eprintln!("{note}");
    }
    let mut prompt = task.trim().to_string();
    loop {
        let outcome = match harness
            .run_turn_cancellable(&prompt, ui, TurnCancellation::new())
            .await
        {
            Ok(outcome) => outcome,
            Err(err) => {
                eprintln!("{}", harness.stop_ladder(TURN_FAILED));
                return Err(err);
            }
        };
        match harness.ladder_next_step(&outcome, ui) {
            LadderStep::RunPrompt(next) => prompt = next,
            LadderStep::Verified(summary) | LadderStep::Stopped(summary) => {
                println!("{summary}");
                for line in harness.ladder().report_lines() {
                    println!("{line}");
                }
                return Ok(LadderRun {
                    ladder: harness.ladder().clone(),
                    last_outcome: Some(outcome),
                });
            }
            LadderStep::Idle => {
                return Ok(LadderRun {
                    ladder: harness.ladder().clone(),
                    last_outcome: Some(outcome),
                });
            }
        }
    }
}

/// Plain REPL: after a turn, keep running attempts while the ladder asks
/// for them. `None` means the turn future failed before an outcome.
pub async fn chain(harness: &mut Harness, ui: &mut dyn Ui, outcome: Option<TurnOutcome>) {
    if !harness.ladder().is_active() {
        return;
    }
    let Some(mut outcome) = outcome else {
        println!("{}", harness.stop_ladder(TURN_FAILED));
        return;
    };
    loop {
        match harness.ladder_next_step(&outcome, ui) {
            LadderStep::RunPrompt(prompt) => {
                match harness
                    .run_turn_cancellable(&prompt, ui, TurnCancellation::new())
                    .await
                {
                    Ok(next) => outcome = next,
                    Err(err) => {
                        eprintln!("{err:#}");
                        println!("{}", harness.stop_ladder(TURN_FAILED));
                        return;
                    }
                }
            }
            LadderStep::Verified(summary) | LadderStep::Stopped(summary) => {
                println!("{summary}");
                for line in harness.ladder().report_lines() {
                    println!("{line}");
                }
                return;
            }
            LadderStep::Idle => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_harness::{LadderRung, LadderTurnFacts, TurnStopReason};

    fn settled(phase: LadderPhase) -> Ladder {
        Ladder {
            phase,
            ..Ladder::default()
        }
    }

    fn checked(verification: &str, input_tokens: u64, output_tokens: u64) -> LadderTurnFacts {
        LadderTurnFacts {
            stop_reason: TurnStopReason::Completed,
            changed_files: vec!["src/lib.rs".into()],
            verification: Some(verification.into()),
            usage: hi_ai::Usage {
                input_tokens,
                output_tokens,
                ..hi_ai::Usage::default()
            },
            ..LadderTurnFacts::default()
        }
    }

    #[test]
    fn phases_map_to_exit_codes() {
        assert_eq!(exit_code(&settled(LadderPhase::Verified)), EXIT_VERIFIED);
        assert_eq!(exit_code(&settled(LadderPhase::Exhausted)), EXIT_EXHAUSTED);
        assert_eq!(exit_code(&settled(LadderPhase::Stopped)), EXIT_STOPPED);
        assert_eq!(
            exit_code(&settled(LadderPhase::Running)),
            EXIT_HARNESS_ERROR
        );
        assert!(
            LadderRun {
                ladder: settled(LadderPhase::Verified),
                last_outcome: None,
            }
            .into_result()
            .is_ok()
        );
        let err = LadderRun {
            ladder: settled(LadderPhase::Exhausted),
            last_outcome: None,
        }
        .into_result()
        .unwrap_err();
        let exit = err.downcast_ref::<LadderExit>().expect("typed exit");
        assert_eq!(exit.code, EXIT_EXHAUSTED);
    }

    #[test]
    fn report_carries_rungs_attempts_and_tokens() {
        let rungs = vec![
            LadderRung::parse("pipe/deepseek-v4-flash low").unwrap(),
            LadderRung::parse("pipe/gpt-6-astra high").unwrap(),
        ];
        let mut ladder = Ladder::start("fix it", "pipe/base", rungs, 1, "cargo test", 0);
        ladder.after_turn(&checked("failed", 1_200, 80));
        ladder.after_turn(&checked("passed", 3_000, 200));
        let report = ladder_report_json(&ladder);
        assert_eq!(report["phase"], "verified");
        assert_eq!(report["exit_code"], EXIT_VERIFIED);
        assert_eq!(
            report["rungs"],
            serde_json::json!(["pipe/deepseek-v4-flash · low", "pipe/gpt-6-astra · high"])
        );
        assert_eq!(report["attempts"][0]["result"], "verify failed");
        assert_eq!(report["attempts"][1]["rung"], 2);
        assert_eq!(report["input_tokens"], 4_200);
        assert_eq!(report["output_tokens"], 280);
    }
}
