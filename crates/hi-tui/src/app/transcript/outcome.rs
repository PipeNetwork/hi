//! Typed turn-outcome presentation and terminal failure messaging.

use hi_harness::{TurnOutcome, TurnStopReason};
use ratatui::style::Style;

use crate::TurnState;
use crate::render::{accent_line, dim};
use crate::theme::theme;
use crate::util::fmt_rate_limits;

impl crate::App {
    pub(crate) fn note_turn_outcome(&mut self, outcome: &TurnOutcome) {
        let detail = match outcome.stop_reason {
            TurnStopReason::Completed => "done".to_string(),
            TurnStopReason::Cancelled => "cancelled".to_string(),
            TurnStopReason::Error => outcome.error.clone().unwrap_or_else(|| "error".to_string()),
        };
        match outcome.stop_reason {
            TurnStopReason::Completed => {
                self.status = format!("done · {detail}");
                self.last_turn_state = TurnState::Done(detail.clone());
                self.last_error = None;
                self.push(accent_line(
                    theme().accent_success,
                    format!("✓ {detail}"),
                    dim(),
                ));
            }
            TurnStopReason::Cancelled => {
                self.status = "cancelled".to_string();
                self.last_turn_state = TurnState::Cancelled;
                self.last_error = None;
                self.push(accent_line(
                    theme().warning,
                    "⚠ cancelled",
                    Style::default().fg(theme().warning),
                ));
            }
            TurnStopReason::Error => {
                self.status = format!("failed · {detail}");
                self.last_turn_state = TurnState::Failed(detail.clone());
                self.last_error = Some(detail.clone());
                self.push(accent_line(
                    theme().accent_error,
                    format!("✗ failed · {detail}"),
                    Style::default().fg(theme().accent_error),
                ));
            }
        }
    }

    pub(crate) fn note_turn_failed(&mut self, error: &str, kind: &str, guidance: &str) {
        if matches!(kind, "workspace" | "recovery") {
            self.note_workspace_admission_error(error, kind, guidance);
            return;
        }
        self.status = format!("failed · {kind}");
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

    pub(crate) fn note_workspace_admission_error(
        &mut self,
        error: &str,
        kind: &str,
        guidance: &str,
    ) {
        self.last_error = Some(error.to_string());
        let guidance_line = if guidance.is_empty() {
            String::new()
        } else {
            format!("\n  💡 {guidance}")
        };
        self.push(accent_line(
            theme().warning,
            format!("⚠ {kind}: {error}{guidance_line}"),
            Style::default().fg(theme().warning),
        ));
        self.follow();
    }
}
