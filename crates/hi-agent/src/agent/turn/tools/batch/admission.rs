//! Sealed workspace admission and pre-execution denial helpers.

use anyhow::Result;

use super::outcome::{ToolProtocolFailure, tool_protocol_failure_content};
use super::{ProgressKind, ProgressTracker, ToolProgressLabel, ToolTimeline};
use crate::Ui;
use crate::agent::turn::helpers::{synthetic_tool_outcome, tool_entry};
use crate::heuristics::emit_tool_output;
use crate::steering::inspection_signature;

pub(super) enum ToolBatchAdmission {
    NotRequired,
    Admitted(hi_workspace::MutationIntent),
    Stale(String),
}

pub(super) fn stale_workspace_protocol_failure(
    error: &anyhow::Error,
    tool: &str,
) -> Option<ToolProtocolFailure> {
    error
        .downcast_ref::<crate::workspace_coordination::SealedWorkspaceAdmissionError>()
        .map(|stale| ToolProtocolFailure::stale_workspace(tool, stale.message()))
}

impl crate::Agent {
    /// Close an admission retained by a batch which returned before the normal
    /// transcript/terminal settlement fence. Before any executor boundary the
    /// failure is definite and reopens the workspace. Once an effect-capable
    /// call was dispatched, leave the permit with the ordinary failed-turn
    /// cleanup path so it can quiesce processes, reconcile the final image,
    /// and avoid manufacturing recovery when the result is still provable.
    pub(super) async fn finalize_failed_tool_batch_admission(
        &mut self,
        error: anyhow::Error,
        active_operation_before: Option<&hi_workspace::OperationId>,
        effects_may_have_begun: bool,
    ) -> anyhow::Error {
        if effects_may_have_begun {
            return error;
        }
        let Some(active) = self.workspace_coordination.active_mutation_record() else {
            return error;
        };
        // Never settle a permit which predates this batch. An admission error
        // caused by an already-unsettled operation belongs to its original
        // owner and must remain fenced for explicit recovery.
        if active_operation_before == Some(&active.operation_id) {
            return error;
        }
        let detail = format!("tool batch failed before normal settlement: {error:#}");
        let mut execution = hi_workspace::ExecutionReport {
            disposition: hi_workspace::ExecutionDisposition::Failed,
            workspace_may_have_changed: false,
            external_effect_may_have_occurred: false,
            content_digest: None,
            changed_paths: Vec::new(),
            artifacts: Vec::new(),
            detail: Some(detail),
        };
        let record = [hi_ai::Content::Text(
            "Tool batch failed before any admitted workspace or external effect was dispatched."
                .into(),
        )];
        let stage_error = self
            .stage_active_workspace_execution(&[], &record, &[], &execution)
            .err();
        if let Some(stage_error) = &stage_error {
            execution.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
            execution.detail = Some(format!(
                "pre-execution failure transcript could not be staged: {stage_error:#}"
            ));
        }
        let settlement = self
            .checkpoint_durable_workspace_with_execution(execution)
            .await;
        match (stage_error, settlement) {
            (None, Ok(())) => error.context(
                "tool batch failed before any admitted workspace or external effect; workspace admission was settled",
            ),
            (None, Err(settlement_error)) => error.context(format!(
                "tool batch failed before execution and its admission could not be settled cleanly: {settlement_error:#}"
            )),
            (Some(stage_error), Ok(())) => error.context(format!(
                "tool batch failed before execution; its audit record could not be staged even though settlement returned: {stage_error:#}"
            )),
            (Some(stage_error), Err(settlement_error)) => error.context(format!(
                "tool batch failed before execution; audit staging failed ({stage_error:#}) and settlement requires recovery: {settlement_error:#}"
            )),
        }
    }

    pub(super) async fn admit_sealed_tool_batch(
        &self,
        calls: &[(String, String, String)],
        envelope: &hi_tools::envelope::ToolEnvelope,
        allow_process_feedback: bool,
    ) -> Result<ToolBatchAdmission> {
        if self.config.gates.dry_run {
            return Ok(ToolBatchAdmission::NotRequired);
        }
        let Some(intent) = super::policy::workspace_intent_for_admitted_calls(
            calls,
            self.pipefs_workspace_active(),
            allow_process_feedback,
            self.config.gates.proactive_verify,
        ) else {
            return Ok(ToolBatchAdmission::NotRequired);
        };
        match self
            .begin_sealed_workspace_operation(intent.clone(), &envelope.payload.workspace)
            .await
        {
            Ok(()) => Ok(ToolBatchAdmission::Admitted(intent)),
            Err(error) => stale_workspace_protocol_failure(&error, "tool batch")
                .map(|stale| ToolBatchAdmission::Stale(stale.message))
                .ok_or(error),
        }
    }

    pub(super) async fn begin_sealed_workspace_operation(
        &self,
        intent: hi_workspace::MutationIntent,
        sealed: &hi_tools::envelope::WorkspaceEnvelope,
    ) -> Result<()> {
        self.workspace_coordination
            .begin_sealed_intent(self.workspace_durability.clone(), intent, sealed)
            .await?;
        if let Err(error) =
            hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::ToolBeforeStart)
        {
            self.workspace_coordination.abandon_active()?;
            return Err(error.into());
        }
        Ok(())
    }

    pub(super) async fn admit_sealed_program(
        &self,
        envelope: &hi_tools::envelope::ToolEnvelope,
    ) -> Result<Option<hi_workspace::MutationIntent>> {
        let intent = super::policy::workspace_program_intent_for_envelope(envelope);
        if let Some(intent) = intent.as_ref() {
            self.begin_sealed_workspace_operation(intent.clone(), &envelope.payload.workspace)
                .await?;
        }
        Ok(intent)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_stale_workspace_denials(
    calls: &[(String, String, String)],
    permitted_prefix: usize,
    message: &str,
    results: &mut [Option<(String, String)>],
    completed: &mut [bool],
    completion_order: &mut Vec<usize>,
    protocol_errors: &mut Vec<ToolProtocolFailure>,
    progress_tracker: &mut ProgressTracker,
    progress_labels: &mut Vec<ToolProgressLabel>,
    tool_timeline: &mut ToolTimeline,
    ui: &mut dyn Ui,
) {
    for (index, (id, name, arguments)) in calls.iter().enumerate().take(permitted_prefix) {
        if completed[index] {
            continue;
        }
        let failure = ToolProtocolFailure::stale_workspace(name, message);
        ui.tool_call_id(id, name, arguments);
        let content = tool_protocol_failure_content(&failure);
        protocol_errors.push(failure);
        let output = synthetic_tool_outcome(content.clone(), hi_tools::ToolStatus::Denied);
        emit_tool_output(ui, id, name, &output);
        let label = ToolProgressLabel::new(
            ProgressKind::None,
            "tool denied by stale workspace envelope",
            inspection_signature(name, arguments),
        );
        progress_tracker.record_tool(&label);
        progress_labels.push(label.clone());
        tool_timeline.push(tool_entry(
            name.clone(),
            hi_tools::target_path(name, arguments).unwrap_or_default(),
            0,
            &output,
            &label,
        ));
        results[index] = Some((id.clone(), content));
        completed[index] = true;
        completion_order.push(index);
        if let Some(entry) = tool_timeline.last_mut() {
            entry.completion_index = completion_order.len() as u32;
        }
    }
}
