//! Settlement bridge for background lifecycle effects hidden by `run_program`.

use std::collections::BTreeSet;

use anyhow::Result;
use hi_workflow::ProgramCall;

use super::program_failed_result;

type ResolvedProgramCall = (
    std::result::Result<hi_workflow::ProgramToolResult, String>,
    hi_tools::ToolOutcome,
);

impl crate::Agent {
    /// Background polls are read-only as calls, but observing their terminal
    /// state closes the lifecycle of a previously admitted live writer.
    pub(super) async fn observe_program_effect(
        &self,
        call: &ProgramCall,
        resolved: ResolvedProgramCall,
    ) -> (ResolvedProgramCall, Option<String>) {
        let terminal = resolved
            .1
            .background
            .as_ref()
            .filter(|background| {
                matches!(
                    background.state,
                    hi_tools::BackgroundState::Exited
                        | hi_tools::BackgroundState::Killed
                        | hi_tools::BackgroundState::Failed
                )
            })
            .map(|background| background.id.clone());
        if let Some(background) = &resolved.1.background
            && let Err(error) = self.observe_durable_background_process(background).await
        {
            return (
                program_failed_result(
                    call,
                    format!(
                        "background process state could not be durably recorded: {error:#}; run /pipefs retry"
                    ),
                ),
                terminal,
            );
        }
        (resolved, terminal)
    }

    /// A terminal writer may be reconciled only after its lifecycle callback
    /// has made the exact job durability-pending. Running writers stay outside
    /// parent mutation admission and continue to own their job permit.
    pub(super) async fn admit_terminal_program_reconciliation(
        &self,
        mut intent: Option<hi_workspace::MutationIntent>,
        terminal: &BTreeSet<String>,
    ) -> Result<(Option<hi_workspace::MutationIntent>, bool)> {
        if intent.is_some() || terminal.is_empty() {
            return Ok((intent, false));
        }
        let pending = self.runtime.background().pending_job_settlements().await;
        if !pending.iter().any(|job| terminal.contains(&job.handle)) {
            return Ok((intent, false));
        }
        let reconciliation = hi_workspace::MutationIntent::reconciliation();
        self.workspace_coordination
            .begin_intent(self.workspace_durability.clone(), reconciliation.clone())
            .await?;
        intent = Some(reconciliation);
        Ok((intent, true))
    }
}
