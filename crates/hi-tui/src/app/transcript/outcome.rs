//! Typed turn-outcome presentation and terminal failure messaging.

use std::time::Duration;

use hi_agent::{ReviewStatus, TurnOutcome, TurnStatus, TurnStopReason, VerificationStatus};
use ratatui::style::Style;

use crate::TurnState;
use crate::render::{accent_line, dim};
use crate::theme::theme;
use crate::util::fmt_rate_limits;

impl crate::App {
    /// Apply the authoritative typed result returned by `Agent::run_turn`.
    ///
    /// `Ui::turn_end` carries token accounting only and can arrive before final
    /// workspace reconciliation. It must therefore never decide whether a turn
    /// succeeded. This is the sole success-state transition for a normal turn.
    pub(crate) fn note_turn_outcome(&mut self, outcome: &TurnOutcome) {
        self.last_stop_reason = Some(outcome.stop_reason);
        let detail = outcome_detail(outcome);
        match outcome_state(outcome) {
            OutcomeState::Done => {
                self.status = format!("done · {detail}");
                self.last_turn_state = TurnState::Done(detail.clone());
                self.last_error = None;
                // “No applicable checks” is a non-event. Keep the typed state
                // for /status, but don't paint a green receipt into the pane.
                if outcome.verification == VerificationStatus::Passed {
                    self.push(accent_line(
                        theme().accent_success,
                        format!("✓ done · {detail}"),
                        dim(),
                    ));
                }
            }
            OutcomeState::Warning => {
                let label = match outcome.status {
                    TurnStatus::Blocked => format!("blocked · {detail}"),
                    TurnStatus::Incomplete => format!("incomplete · {detail}"),
                    _ => detail,
                };
                self.status = format!("warning · {label}");
                self.last_turn_state = TurnState::Warning(label.clone());
                self.last_error = Some(label.clone());
                self.push(accent_line(
                    theme().warning,
                    format!("⚠ {label}"),
                    Style::default().fg(theme().warning),
                ));
            }
            OutcomeState::Failed => {
                self.status = format!("failed · {detail}");
                self.last_turn_state = TurnState::Failed(detail.clone());
                self.last_error = Some(detail.clone());
                // Infrastructure failures are internal (provider/runner/session).
                // Keep typed state for reports/eval, but don't dump the jargon
                // banner into the user transcript.
                if !is_infrastructure_failure_detail(&detail) {
                    self.push(accent_line(
                        theme().accent_error,
                        format!("✗ failed · {detail}"),
                        Style::default().fg(theme().accent_error),
                    ));
                }
            }
            OutcomeState::Cancelled => {
                self.status = "cancelled".to_string();
                self.last_turn_state = TurnState::Cancelled;
                self.last_error = None;
                self.push(accent_line(
                    theme().warning,
                    "⚠ cancelled",
                    Style::default().fg(theme().warning),
                ));
            }
        }
        // No follow(): preserve a reader's scroll position at turn end.
    }

    pub(crate) fn note_turn_failed(&mut self, error: &str, kind: &str, guidance: &str) {
        self.status = format!("failed · {kind}").to_string();
        self.last_turn_state = TurnState::Failed(error.to_string());
        self.last_error = Some(error.to_string());
        let guidance_line = if guidance.is_empty() {
            String::new()
        } else {
            format!("\n  💡 {guidance}")
        };
        let limits = fmt_rate_limits(self.rate_limits)
            .map(|limits| format!("\n  {limits}"))
            .unwrap_or_default();
        self.push(accent_line(
            theme().accent_error,
            format!("✗ failed · {kind}: {error}{guidance_line}{limits}"),
            Style::default().fg(theme().accent_error),
        ));
        self.follow();
    }

    pub(crate) fn note_backend_waiting(&mut self, idle: Duration, threshold: Duration) {
        let _ = (idle, threshold);
        self.push(accent_line(
            theme().warning,
            "⚠ Still thinking. Ctrl-C cancels; keep waiting to continue.",
            Style::default().fg(theme().warning),
        ));
        self.follow();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutcomeState {
    Done,
    Warning,
    Failed,
    Cancelled,
}

fn outcome_state(outcome: &TurnOutcome) -> OutcomeState {
    let stalled_without_project_change =
        outcome.stop_reason == TurnStopReason::Stalled && outcome.changed_files.is_empty();
    if outcome.status == TurnStatus::Cancelled {
        OutcomeState::Cancelled
    } else if outcome.status == TurnStatus::Failed
        || outcome.verification == VerificationStatus::InfrastructureError
    {
        OutcomeState::Failed
    } else if outcome.status == TurnStatus::Completed
        && matches!(
            outcome.verification,
            VerificationStatus::Passed | VerificationStatus::NotApplicable
        )
        && !stalled_without_project_change
        && outcome.review != ReviewStatus::Objected
    {
        // Escalated is a completed scar, not a defect objection.
        OutcomeState::Done
    } else {
        OutcomeState::Warning
    }
}

fn is_infrastructure_failure_detail(detail: &str) -> bool {
    detail == "infrastructure failure"
        || detail == "verification infrastructure failure"
        || detail.starts_with("infrastructure failure")
        || detail.starts_with("verification infrastructure failure")
}

fn outcome_detail(outcome: &TurnOutcome) -> String {
    // A verified project edit may remain successful after a late interaction
    // stall. Baseline checks on an unchanged project do not turn a stalled,
    // answerless turn into success.
    let green_settled = outcome.status == TurnStatus::Completed
        && outcome.verification == VerificationStatus::Passed
        && (outcome.stop_reason != TurnStopReason::Stalled || !outcome.changed_files.is_empty())
        && outcome.review != ReviewStatus::Objected;
    let base = if green_settled {
        "verified".to_string()
    } else if outcome.status == TurnStatus::Incomplete {
        if let Some(leftover) = outcome
            .leftover
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            leftover.to_string()
        } else {
            match outcome.stop_reason {
                TurnStopReason::Completed => match outcome.verification {
                    VerificationStatus::Passed => "verified",
                    VerificationStatus::NotApplicable => "no applicable checks",
                    VerificationStatus::Unverified => "checks did not settle",
                    VerificationStatus::Failed => "verification failed",
                    VerificationStatus::InfrastructureError => {
                        "verification infrastructure failure"
                    }
                },
                TurnStopReason::NoApplicableVerification => "no applicable checks",
                TurnStopReason::VerificationUnavailable => "checks did not settle",
                TurnStopReason::VerificationFailed => "verification failed",
                TurnStopReason::VerificationUnstable => "verification was unstable",
                TurnStopReason::ReviewObjected => "review objected",
                TurnStopReason::ReviewEscalated => "review escalated",
                TurnStopReason::ToolModeDenied => "required tool was denied",
                TurnStopReason::StepLimit => "step limit reached",
                TurnStopReason::TimeLimit => "time budget reached",
                TurnStopReason::TurnLimit => "turn limit reached",
                TurnStopReason::Stalled => "stalled",
                TurnStopReason::Cancelled => "cancelled",
                TurnStopReason::InfrastructureFailure => "infrastructure failure",
            }
            .to_string()
        }
    } else {
        match outcome.stop_reason {
            TurnStopReason::Completed => match outcome.verification {
                VerificationStatus::Passed => "verified",
                VerificationStatus::NotApplicable => "no applicable checks",
                VerificationStatus::Unverified => "checks did not settle",
                VerificationStatus::Failed => "verification failed",
                VerificationStatus::InfrastructureError => "verification infrastructure failure",
            },
            TurnStopReason::NoApplicableVerification => "no applicable checks",
            TurnStopReason::VerificationUnavailable => "checks did not settle",
            TurnStopReason::VerificationFailed => "verification failed",
            TurnStopReason::VerificationUnstable => "verification was unstable",
            TurnStopReason::ReviewObjected => "review objected",
            TurnStopReason::ReviewEscalated => "review escalated",
            TurnStopReason::ToolModeDenied => "required tool was denied",
            TurnStopReason::StepLimit => "step limit reached",
            TurnStopReason::TimeLimit => "time budget reached",
            TurnStopReason::TurnLimit => "turn limit reached",
            TurnStopReason::Stalled => "stalled",
            TurnStopReason::Cancelled => "cancelled",
            TurnStopReason::InfrastructureFailure => "infrastructure failure",
        }
        .to_string()
    };
    match outcome.review {
        ReviewStatus::Passed if outcome.verification == VerificationStatus::Passed => {
            format!("{base} · reviewed")
        }
        // A review transport failure is non-blocking after deterministic
        // verification passes. Keep it in the report/debug telemetry rather
        // than turning a green result into a noisy warning banner.
        ReviewStatus::Unavailable if outcome.verification == VerificationStatus::Passed => base,
        ReviewStatus::Objected if base == "review objected" => base,
        ReviewStatus::Objected => format!("{base} · review objected"),
        ReviewStatus::Escalated => format!("{base} · review escalated"),
        _ => base,
    }
}
