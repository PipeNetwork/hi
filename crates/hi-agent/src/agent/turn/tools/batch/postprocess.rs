//! One observation boundary for every physically completed native tool call.

use std::collections::HashSet;

use anyhow::Result;

use crate::agent::turn::helpers::{tool_entry_with_args, tool_satisfies_validation};
use crate::agent::turn::progress::{
    ProgressKind, ProgressTracker, ToolProgressLabel, classify_tool_progress, signature_seen,
};
use crate::agent::turn::retention::ToolTimeline;
use crate::recovery::ValidationObservation;
#[cfg(test)]
use crate::recovery::ValidationResult;
use crate::steering::{
    EvidenceTracker, ImplementationTracker, bash_command, implementation_tool_call_validates,
    inspection_signature, validation_exit_status_is_reliable,
};

pub(super) async fn validation_input_revision<'a>(
    agent: &crate::Agent,
    calls: impl Iterator<Item = (&'a str, &'a str)>,
) -> Result<Option<String>> {
    let mut validation = false;
    let mut goal_input = false;
    for (name, arguments) in calls {
        if implementation_tool_call_validates(name, arguments) {
            validation = true;
            goal_input |= bash_command(arguments).is_some_and(|command| {
                crate::goal_export::is_referenced(&command, agent.runtime.root())
            });
        }
    }
    if !validation {
        return Ok(None);
    }
    agent.runtime.ensure_ledger_scan_complete_async().await?;
    if goal_input {
        agent.runtime.register_goal_validation_input().await?;
    }
    agent.runtime.reconcile_ledger_async().await?;
    Ok(Some(agent.runtime.ledger().workspace_revision()))
}

pub(super) struct CompletedTool<'a> {
    pub index: usize,
    pub name: &'a str,
    pub arguments: &'a str,
    pub output: &'a hi_tools::ToolOutcome,
    pub input_revision: Option<&'a str>,
    pub path: String,
    pub duration_ms: u64,
    pub plan_changed: bool,
}

pub(super) struct ToolObservations {
    batch_id: uuid::Uuid,
    seen: HashSet<usize>,
    pub hashable_idempotent_results: usize,
    pub repeated_idempotent_results: usize,
    pub running_background_poll_results: usize,
    pub actionable_poll_results: usize,
    pub wait_flavored_results: usize,
}

impl Default for ToolObservations {
    fn default() -> Self {
        Self {
            batch_id: uuid::Uuid::new_v4(),
            seen: HashSet::new(),
            hashable_idempotent_results: 0,
            repeated_idempotent_results: 0,
            running_background_poll_results: 0,
            actionable_poll_results: 0,
            wait_flavored_results: 0,
        }
    }
}

impl ToolObservations {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn record(
        &mut self,
        agent: &mut crate::Agent,
        completed: CompletedTool<'_>,
        evidence: &mut EvidenceTracker,
        implementation: &mut ImplementationTracker,
        progress_tracker: &mut ProgressTracker,
        labels: &mut Vec<ToolProgressLabel>,
        timeline: &mut ToolTimeline,
    ) -> Result<()> {
        if !self.seen.insert(completed.index) {
            return Ok(());
        }
        let CompletedTool {
            index,
            name,
            arguments,
            output,
            input_revision,
            path,
            duration_ms,
            plan_changed,
        } = completed;
        let input_stable = if let Some(revision) = input_revision {
            crate::agent::turn::fast_feedback_observations::unchanged(
                &agent.runtime,
                Some(revision),
            )
            .await
        } else {
            true
        };
        let error = output.status != hi_tools::ToolStatus::Succeeded;
        let semantic_output = if error && !output.content.starts_with("Error:") {
            std::borrow::Cow::Owned(format!("Error: {}", output.content))
        } else {
            std::borrow::Cow::Borrowed(output.content.as_str())
        };
        let signature = inspection_signature(name, arguments);
        let signature_was_seen = signature_seen(evidence, &signature);
        let before = implementation.clone();
        let validation_succeeded =
            input_stable && tool_satisfies_validation(name, arguments, output);
        evidence.record_success(name, arguments, &semantic_output);
        implementation.record_tool_result(
            name,
            arguments,
            &semantic_output,
            validation_succeeded,
            output.effects.mutation_applied,
        );
        progress_tracker
            .tool_guardrail
            .observe_workspace_revision(agent.runtime.ledger().revision());
        let progress = progress_tracker
            .tool_guardrail
            .record_tool_result_with_effects(
                name,
                arguments,
                &semantic_output,
                output.effects.mutation_applied,
            );
        self.running_background_poll_results += usize::from(progress.running_background_poll);
        self.actionable_poll_results += usize::from(progress.actionable_background_output);
        self.wait_flavored_results +=
            usize::from(super::policy::wait_flavored_call(name, arguments, output));
        self.hashable_idempotent_results += usize::from(progress.hashable_idempotent);
        self.repeated_idempotent_results +=
            usize::from(progress.hashable_idempotent && progress.repeated_idempotent_result);
        let label = if name == "delegate" && output.effects.mutation_applied {
            ToolProgressLabel::new(
                ProgressKind::Meaningful,
                "successful delegated mutation",
                signature,
            )
        } else {
            classify_tool_progress(
                name,
                arguments,
                &semantic_output,
                error,
                validation_succeeded,
                signature,
                signature_was_seen,
                progress.repeated_idempotent_result,
                &before,
                plan_changed,
                agent.runtime.root(),
            )
        };
        progress_tracker.record_tool(&label);
        labels.push(label.clone());
        timeline.push(tool_entry_with_args(
            name.into(),
            path,
            duration_ms,
            output,
            &label,
            arguments,
        ));
        if let Some(input_revision) = input_revision
            && let Some(observation) = validation_observation(
                format!("tool:{}:{index}", self.batch_id),
                name,
                arguments,
                input_revision,
                input_stable,
                output,
                agent.runtime.root(),
            )
        {
            agent.observe_validation(observation).await?;
        }
        Ok(())
    }
}

fn validation_observation(
    execution_id: String,
    name: &str,
    arguments: &str,
    input_revision: &str,
    input_stable: bool,
    output: &hi_tools::ToolOutcome,
    root: &std::path::Path,
) -> Option<ValidationObservation> {
    if !implementation_tool_call_validates(name, arguments) {
        return None;
    }
    let command = bash_command(arguments)?;
    let stable = input_stable
        && !output.effects.mutation_applied
        && validation_exit_status_is_reliable(name, arguments);
    let status = crate::agent::turn::fast_feedback_observations::process_status(output, stable);
    Some(ValidationObservation::command(
        execution_id,
        &command,
        input_revision.into(),
        status,
        &output.content,
        root,
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::turn::helpers::synthetic_tool_outcome;
    use crate::tests::common::{Canned, IsolatedWorkspace};
    use std::sync::{Arc, Mutex};

    fn process(status: hi_tools::ToolStatus, exit_code: Option<i32>) -> hi_tools::ToolOutcome {
        let mut output =
            synthetic_tool_outcome("test parser::rejects_bad_input ... FAILED".into(), status);
        output.process = Some(hi_tools::ProcessOutcome {
            exit_code,
            stdout_summary: String::new(),
            stderr_summary: String::new(),
            duration_ms: 1,
        });
        output
    }

    #[test]
    fn only_actual_stable_process_results_establish_validation() {
        let root = std::path::Path::new("/workspace");
        let args = r#"{"command":"cargo test --workspace"}"#;
        for (status, code, stable, expected) in [
            (
                hi_tools::ToolStatus::Succeeded,
                Some(0),
                true,
                ValidationResult::Passed,
            ),
            (
                hi_tools::ToolStatus::Failed,
                Some(1),
                true,
                ValidationResult::Failed,
            ),
            (
                hi_tools::ToolStatus::Failed,
                None,
                true,
                ValidationResult::Infrastructure,
            ),
            (
                hi_tools::ToolStatus::TimedOut,
                None,
                true,
                ValidationResult::Infrastructure,
            ),
            (
                hi_tools::ToolStatus::Cancelled,
                None,
                true,
                ValidationResult::Deferred,
            ),
            (
                hi_tools::ToolStatus::Succeeded,
                Some(0),
                false,
                ValidationResult::Deferred,
            ),
        ] {
            let observation = validation_observation(
                "execution".into(),
                "bash",
                args,
                "revision",
                stable,
                &process(status, code),
                root,
            )
            .unwrap();
            assert_eq!(observation.status, expected);
            assert!(!observation.required_stage);
        }
        let synthetic =
            synthetic_tool_outcome("program succeeded".into(), hi_tools::ToolStatus::Succeeded);
        assert!(
            validation_observation(
                "envelope".into(),
                "run_program",
                "{}",
                "revision",
                true,
                &synthetic,
                root
            )
            .is_none()
        );
        assert_eq!(
            validation_observation(
                "launch".into(),
                "bash",
                args,
                "revision",
                true,
                &synthetic,
                root
            )
            .unwrap()
            .status,
            ValidationResult::Infrastructure
        );
    }

    #[test]
    fn exact_execution_and_scope_preserve_recovery_obligations() {
        let root = std::path::Path::new("/workspace");
        let mut recovery = crate::TaskRecoveryState::new("repair".into(), 3);
        let failed = validation_observation(
            "first".into(),
            "bash",
            r#"{"command":"cargo test --workspace"}"#,
            "revision",
            true,
            &process(hi_tools::ToolStatus::Failed, Some(1)),
            root,
        )
        .unwrap();
        recovery.observe(&failed);
        recovery.observe(&failed);
        assert!(
            !recovery.exhausted,
            "replaying one physical result is not another failed execution"
        );
        recovery.intervene("repair");
        let remaining = recovery.remaining;
        for command in ["cargo test -p core", "cargo check --workspace"] {
            let args = serde_json::json!({"command":command}).to_string();
            recovery.observe(
                &validation_observation(
                    command.into(),
                    "bash",
                    &args,
                    "revision",
                    true,
                    &process(hi_tools::ToolStatus::Succeeded, Some(0)),
                    root,
                )
                .unwrap(),
            );
        }
        assert_eq!(
            recovery.remaining, remaining,
            "narrow or unrelated green does not discharge workspace tests"
        );
        recovery.observe(
            &validation_observation(
                "resolved".into(),
                "bash",
                r#"{"command":"cargo test --workspace"}"#,
                "revision",
                true,
                &process(hi_tools::ToolStatus::Succeeded, Some(0)),
                root,
            )
            .unwrap(),
        );
        assert_eq!(recovery.remaining, 3);
    }

    #[tokio::test]
    async fn replaying_one_completion_updates_all_trackers_once() {
        let workspace = IsolatedWorkspace::new("tool-observation-dedupe");
        let mut agent =
            crate::Agent::new(Arc::new(Canned(Mutex::new(Vec::new()))), workspace.config())
                .unwrap();
        let mut observations = ToolObservations::default();
        let mut evidence = EvidenceTracker::default();
        let mut implementation = ImplementationTracker::default();
        let mut progress = ProgressTracker::default();
        let mut labels = Vec::new();
        let mut timeline = ToolTimeline::default();
        let output = synthetic_tool_outcome("fn main() {}".into(), hi_tools::ToolStatus::Succeeded);
        for _ in 0..2 {
            observations
                .record(
                    &mut agent,
                    CompletedTool {
                        index: 7,
                        name: "read",
                        arguments: r#"{"path":"src/main.rs"}"#,
                        output: &output,
                        input_revision: None,
                        path: "src/main.rs".into(),
                        duration_ms: 1,
                        plan_changed: false,
                    },
                    &mut evidence,
                    &mut implementation,
                    &mut progress,
                    &mut labels,
                    &mut timeline,
                )
                .await
                .unwrap();
        }
        assert_eq!(evidence.file_reads, 1);
        assert_eq!(implementation.pre_mutation_tool_calls, 1);
        assert_eq!(labels.len(), 1);
        assert_eq!(timeline.len(), 1);
    }
}
