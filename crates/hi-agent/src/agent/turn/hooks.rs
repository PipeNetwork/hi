//! Controller-guarded repository lifecycle hooks.

use anyhow::{Context, Result};

use crate::session_ops::HookExecution;

/// A trusted repository hook is arbitrary project code. Admit it before the
/// process is spawned, then reconcile its opaque workspace effects and settle
/// the executor's real outcome even when the hook itself failed.
impl crate::Agent {
    pub(super) async fn run_workspace_lifecycle_hook(
        &mut self,
        name: &str,
        input: &str,
        cancellation: &crate::TurnCancellation,
    ) -> Result<Option<String>> {
        let workspace = self.workspace_root().to_path_buf();
        let hook = crate::session_ops::run_hook_cancellable(&workspace, name, input, cancellation);
        self.run_admitted_lifecycle_hook(name, input, cancellation, hook)
            .await
    }

    async fn run_admitted_lifecycle_hook<F>(
        &mut self,
        name: &str,
        input: &str,
        cancellation: &crate::TurnCancellation,
        hook: F,
    ) -> Result<Option<String>>
    where
        F: std::future::Future<Output = HookExecution>,
    {
        // Do not create an operation for a hook which was skipped before
        // admission. Once admitted, cancellation is a real terminal execution
        // result and must be reconciled and settled below.
        if cancellation.is_cancelled() {
            return Ok(None);
        }

        let intent = hi_workspace::MutationIntent {
            effect_scope: hi_workspace::EffectScope::LiveWriter,
            replay_class: hi_workspace::ReplayClass::NonReplayableExternal,
            dirty_paths: None,
            description: Some(format!("repository lifecycle hook: {name}")),
        };
        self.begin_classified_workspace_operation(intent)
            .await
            .with_context(|| format!("workspace controller refused lifecycle hook {name}"))?;

        let ledger_revision = self.runtime.ledger().revision();
        let hook = hook.await;
        let reconciliation = self.reconcile_workspace_changes().await;
        let changes = if reconciliation.is_ok() {
            self.runtime.ledger().changes_since(ledger_revision)
        } else {
            Vec::new()
        };
        let disposition = hook_execution_disposition(&hook, reconciliation.is_err());
        let mut execution = hi_workspace::ExecutionReport {
            disposition,
            workspace_may_have_changed: reconciliation.is_err()
                || matches!(&hook, HookExecution::Indeterminate(_))
                || !changes.is_empty(),
            // Admission happened before the opaque executable was polled. A
            // failed or cancelled response therefore never permits replay.
            external_effect_may_have_occurred: true,
            content_digest: None,
            changed_paths: changes
                .iter()
                .map(|change| std::path::PathBuf::from(&change.path))
                .collect(),
            artifacts: Vec::new(),
            detail: hook_execution_detail(name, &hook, reconciliation.as_ref().err()),
        };
        let operation_id = self
            .workspace_coordination
            .active_parent_operation()
            .expect("lifecycle hook operation was admitted above");
        let call_id = format!("lifecycle-hook:{name}:{operation_id}");
        let arguments = serde_json::json!({ "name": name, "input": input }).to_string();
        let calls = [(
            call_id.clone(),
            "workspace_lifecycle_hook".into(),
            arguments.clone(),
        )];
        let assistant_content = [hi_ai::Content::ToolCall {
            id: call_id.clone(),
            name: "workspace_lifecycle_hook".into(),
            arguments,
        }];
        let results = [(call_id, hook_transcript_result(&hook))];
        let stage_error = self
            .stage_active_workspace_execution(&calls, &assistant_content, &results, &execution)
            .err();
        if let Some(error) = &stage_error {
            execution.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
            execution.detail = Some(match execution.detail.take() {
                Some(detail) => format!(
                    "{detail}; lifecycle hook transcript evidence could not be staged: {error:#}"
                ),
                None => {
                    format!("lifecycle hook transcript evidence could not be staged: {error:#}")
                }
            });
        }
        let settlement = self
            .checkpoint_durable_workspace_with_execution(execution)
            .await;

        let mut failures = Vec::new();
        let output = match hook {
            HookExecution::Completed(Ok(output)) => Some(output),
            HookExecution::Completed(Err(error)) => {
                failures.push(format!("lifecycle hook {name} failed: {error:#}"));
                None
            }
            HookExecution::Cancelled => None,
            HookExecution::Indeterminate(error) => {
                failures.push(format!(
                    "lifecycle hook {name} process cleanup was indeterminate: {error:#}"
                ));
                None
            }
        };
        if let Err(error) = reconciliation {
            failures.push(format!(
                "lifecycle hook {name} workspace effects could not be reconciled: {error:#}"
            ));
        }
        if let Some(error) = stage_error {
            failures.push(format!(
                "lifecycle hook {name} transcript evidence could not be staged: {error:#}"
            ));
        }
        if let Err(error) = settlement {
            failures.push(format!(
                "lifecycle hook {name} workspace settlement failed: {error:#}"
            ));
        }
        if failures.is_empty() {
            Ok(output)
        } else {
            Err(anyhow::anyhow!(failures.join("; ")))
        }
    }

    /// Deterministic controller tests use an inert executor so they can prove
    /// admission ordering without changing process-global folder trust.
    #[cfg(test)]
    pub(crate) async fn run_workspace_lifecycle_hook_for_test<F>(
        &mut self,
        name: &str,
        input: &str,
        cancellation: &crate::TurnCancellation,
        hook: F,
    ) -> Result<Option<String>>
    where
        F: std::future::Future<Output = HookExecution>,
    {
        self.run_admitted_lifecycle_hook(name, input, cancellation, hook)
            .await
    }
}

fn hook_transcript_result(hook: &HookExecution) -> String {
    match hook {
        HookExecution::Completed(Ok(output)) => output.clone(),
        HookExecution::Completed(Err(error)) => format!("failed: {error:#}"),
        HookExecution::Cancelled => "cancelled after process reap".into(),
        HookExecution::Indeterminate(error) => format!("indeterminate: {error:#}"),
    }
}

fn hook_execution_disposition(
    hook: &HookExecution,
    reconciliation_failed: bool,
) -> hi_workspace::ExecutionDisposition {
    if reconciliation_failed {
        return hi_workspace::ExecutionDisposition::Indeterminate;
    }
    match hook {
        HookExecution::Completed(Ok(_)) => hi_workspace::ExecutionDisposition::Succeeded,
        HookExecution::Completed(Err(_)) => hi_workspace::ExecutionDisposition::Failed,
        HookExecution::Cancelled => hi_workspace::ExecutionDisposition::Cancelled,
        HookExecution::Indeterminate(_) => hi_workspace::ExecutionDisposition::Indeterminate,
    }
}

fn hook_execution_detail(
    name: &str,
    hook: &HookExecution,
    reconciliation: Option<&anyhow::Error>,
) -> Option<String> {
    match (hook, reconciliation) {
        (_, Some(error)) => Some(format!(
            "lifecycle hook {name} effects are indeterminate because reconciliation failed: {error:#}"
        )),
        (HookExecution::Completed(Err(error)), None) => {
            Some(format!("lifecycle hook {name} failed: {error:#}"))
        }
        (HookExecution::Cancelled, None) => Some(format!(
            "lifecycle hook {name} was cancelled after process reap"
        )),
        (HookExecution::Indeterminate(error), None) => Some(format!(
            "lifecycle hook {name} process cleanup was indeterminate: {error:#}"
        )),
        (HookExecution::Completed(Ok(_)), None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_hook_results_keep_their_real_execution_disposition() {
        assert_eq!(
            hook_execution_disposition(&HookExecution::Completed(Ok("ok".into())), false),
            hi_workspace::ExecutionDisposition::Succeeded
        );
        assert_eq!(
            hook_execution_disposition(
                &HookExecution::Completed(Err(anyhow::anyhow!("exit 9"))),
                false,
            ),
            hi_workspace::ExecutionDisposition::Failed
        );
        assert_eq!(
            hook_execution_disposition(&HookExecution::Cancelled, false),
            hi_workspace::ExecutionDisposition::Cancelled
        );
        assert_eq!(
            hook_execution_disposition(&HookExecution::Completed(Ok("ok".into())), true),
            hi_workspace::ExecutionDisposition::Indeterminate
        );
        assert_eq!(
            hook_execution_disposition(
                &HookExecution::Indeterminate(anyhow::anyhow!("reap unknown")),
                false,
            ),
            hi_workspace::ExecutionDisposition::Indeterminate
        );
    }
}
