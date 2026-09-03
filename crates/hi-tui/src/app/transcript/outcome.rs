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
                    _ => format!("stopped · {detail}"),
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
                self.push(accent_line(
                    theme().accent_error,
                    format!("✗ failed · {detail}"),
                    Style::default().fg(theme().accent_error),
                ));
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
    let limit = matches!(
        outcome.stop_reason,
        TurnStopReason::StepLimit
            | TurnStopReason::ToolLimit
            | TurnStopReason::TimeLimit
            | TurnStopReason::TurnLimit
    );
    let canonicalized_legacy_limit = outcome.status == TurnStatus::Failed
        && limit
        && matches!(
            outcome.verification,
            VerificationStatus::Passed | VerificationStatus::NotApplicable
        );
    if outcome.status == TurnStatus::Cancelled && outcome.stop_reason == TurnStopReason::Cancelled {
        OutcomeState::Cancelled
    } else if matches!(
        outcome.verification,
        VerificationStatus::Failed | VerificationStatus::InfrastructureError
    ) || (outcome.status == TurnStatus::Failed && !canonicalized_legacy_limit)
    {
        // Failure evidence outranks a coincident or legacy limit reason.
        OutcomeState::Failed
    } else if limit {
        // Limits are expected execution boundaries, not failures. The
        // canonicalized legacy pair Failed+StepLimit is neutral only when its
        // verification evidence was passed/not-applicable.
        OutcomeState::Warning
    } else if outcome.status == TurnStatus::Cancelled {
        OutcomeState::Cancelled
    } else if outcome.status == TurnStatus::Completed
        && matches!(
            outcome.verification,
            VerificationStatus::Passed | VerificationStatus::NotApplicable
        )
        && outcome.review != ReviewStatus::Objected
        && matches!(
            outcome.stop_reason,
            TurnStopReason::Completed
                | TurnStopReason::NoApplicableVerification
                | TurnStopReason::ReviewEscalated
        )
    {
        // Escalated is a completed scar, not a defect objection.
        OutcomeState::Done
    } else {
        OutcomeState::Warning
    }
}

fn outcome_detail(outcome: &TurnOutcome) -> String {
    let green_settled = outcome.status == TurnStatus::Completed
        && outcome.verification == VerificationStatus::Passed
        && matches!(
            outcome.stop_reason,
            TurnStopReason::Completed | TurnStopReason::ReviewEscalated
        )
        && outcome.review != ReviewStatus::Objected;
    let base = if outcome.verification == VerificationStatus::InfrastructureError {
        "verification infrastructure failure".to_string()
    } else if outcome.verification == VerificationStatus::Failed {
        "verification failed".to_string()
    } else if green_settled {
        "verified".to_string()
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
            TurnStopReason::NoProgress => "no progress",
            TurnStopReason::StepLimit => "step limit reached",
            TurnStopReason::ToolLimit => "tool-call limit reached",
            TurnStopReason::TimeLimit => "time budget reached",
            TurnStopReason::TurnLimit => "turn limit reached",
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
