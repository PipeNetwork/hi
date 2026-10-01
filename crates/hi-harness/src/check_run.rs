//! Running a check command from a turn: the configured verify command at
//! the end of an editing turn, and the `/review` seed command before an
//! audit turn ([`crate::review_seed`]). Both are shown as a tool call, hold
//! the liveness `Verifying` state, and end early when the turn is cancelled.

use crate::ui::Ui;
use crate::{Harness, TurnCancellation};

/// Result of a verify or seed command run by the harness.
pub(crate) enum CheckRun {
    Cancelled,
    Finished { passed: bool, output: String },
}

impl Harness {
    /// Run the configured verify command; `None` when there is none or the
    /// turn was already cancelled, else `passed`, `failed`, or `cancelled`.
    /// A failing run's output is kept for the ladder's retry prompt.
    pub(crate) async fn run_verify(
        &mut self,
        ui: &mut dyn Ui,
        cancel: &TurnCancellation,
    ) -> Option<String> {
        self.last_verify_output = None;
        let command = self.verify_command.clone()?;
        let run = self.run_check(&command, "verify", ui, cancel).await?;
        Some(
            match run {
                CheckRun::Cancelled => "cancelled",
                CheckRun::Finished { passed: true, .. } => "passed",
                CheckRun::Finished {
                    passed: false,
                    output,
                } => {
                    self.last_verify_output = Some(output);
                    "failed"
                }
            }
            .into(),
        )
    }

    /// Whether a `bash` call runs every `&&`/`;`/`|` segment of the
    /// configured verify command. That counts as verification even when the
    /// command is not a runner `looks_like_verify` knows (`./check.sh`).
    pub(crate) fn runs_verify_command(&self, name: &str, arguments: &str) -> bool {
        let Some(verify) = self.verify_command.as_deref() else {
            return false;
        };
        if !name.eq_ignore_ascii_case("bash") {
            return false;
        }
        let Some(command) = serde_json::from_str::<serde_json::Value>(arguments)
            .ok()
            .and_then(|args| args.get("command")?.as_str().map(str::to_string))
        else {
            return false;
        };
        let ran = shell_segments(&command);
        let wanted = shell_segments(verify);
        !wanted.is_empty() && wanted.iter().all(|segment| ran.contains(segment))
    }

    /// Run `command` as a check (verify, or the review seed), shown as a
    /// `label` tool call. `None` when the turn was already cancelled.
    pub(crate) async fn run_check(
        &self,
        command: &str,
        label: &str,
        ui: &mut dyn Ui,
        cancel: &TurnCancellation,
    ) -> Option<CheckRun> {
        if cancel.is_cancelled() {
            return None;
        }
        ui.status(&format!("{label} · {command}"));
        self.liveness
            .set_state(hi_liveness::HarnessState::Verifying);
        self.liveness
            .emit(hi_liveness::EventCode::VerifyStart, None, None, None);
        // A check has no default wall clock, so Esc must be able to end it.
        // Dropping the run future kills the whole process group.
        let run = hi_tools::run_check_in_with_runner(self.tools.runner_ref(), command);
        let ran = tokio::select! {
            ran = run => Some(ran),
            _ = cancel.cancelled() => None,
        };
        let result = match ran {
            Some(Ok(execution)) => {
                let text = execution.display_content();
                ui.tool_call(label, command);
                ui.tool_result(label, &text);
                self.liveness.note_progress();
                CheckRun::Finished {
                    passed: execution.status == hi_tools::ToolStatus::Succeeded,
                    output: execution.model_content(),
                }
            }
            Some(Err(err)) => {
                ui.status(&format!("{label} failed to start: {err:#}"));
                CheckRun::Finished {
                    passed: false,
                    output: format!("{label} failed to start: {err:#}"),
                }
            }
            None => {
                ui.status(&format!("{label} cancelled"));
                CheckRun::Cancelled
            }
        };
        self.liveness
            .emit(hi_liveness::EventCode::VerifyEnd, None, None, None);
        self.liveness.set_state(hi_liveness::HarnessState::Idle);
        Some(result)
    }
}

/// Whitespace-normalized pipeline segments of a shell line.
fn shell_segments(command: &str) -> Vec<String> {
    command
        .split(['\n', ';', '|'])
        .flat_map(|part| part.split("&&"))
        .map(|segment| segment.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|segment| !segment.is_empty())
        .collect()
}
