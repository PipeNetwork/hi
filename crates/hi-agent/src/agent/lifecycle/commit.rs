//! Controller-guarded session-scoped Git commits.

use anyhow::{Context, Result};

impl crate::Agent {
    /// Commit paths recorded by this session and publish their exact outcome.
    ///
    /// This is the single `/commit` implementation for all frontends. Git is
    /// admitted before it can stage files or run filters/hooks, and its output
    /// is not returned as success until reconciliation, transcript staging,
    /// and workspace settlement have all completed.
    pub async fn commit_session_changes(&mut self, paths: &[String]) -> Result<String> {
        let intent = hi_workspace::MutationIntent {
            effect_scope: hi_workspace::EffectScope::LiveWriter,
            replay_class: hi_workspace::ReplayClass::NonReplayableExternal,
            dirty_paths: Some(paths.iter().map(std::path::PathBuf::from).collect()),
            description: Some("commit session workspace changes".into()),
        };
        self.begin_classified_workspace_operation(intent)
            .await
            .context("workspace controller refused the Git commit")?;

        let ledger_revision = self.runtime.ledger().revision();
        let commit = hi_tools::commit_in_typed(self.workspace_root(), paths).await;
        let reconciliation = self.reconcile_workspace_changes().await;
        let changes = if reconciliation.is_ok() {
            self.runtime.ledger().changes_since(ledger_revision)
        } else {
            Vec::new()
        };

        let mut execution = hi_workspace::ExecutionReport {
            disposition: commit_disposition(commit.status, reconciliation.is_err()),
            workspace_may_have_changed: commit.workspace_may_have_changed
                || reconciliation.is_err()
                || !changes.is_empty(),
            external_effect_may_have_occurred: commit.external_effect_may_have_occurred,
            content_digest: None,
            changed_paths: changes
                .iter()
                .map(|change| std::path::PathBuf::from(&change.path))
                .collect(),
            artifacts: Vec::new(),
            detail: commit_detail(&commit, reconciliation.as_ref().err()),
        };

        let operation_id = self
            .workspace_coordination
            .active_parent_operation()
            .expect("Git commit operation was admitted above");
        let call_id = format!("session-git-commit:{operation_id}");
        let arguments = serde_json::json!({ "paths": paths }).to_string();
        let calls = [(
            call_id.clone(),
            "session_git_commit".into(),
            arguments.clone(),
        )];
        let assistant_content = [hi_ai::Content::ToolCall {
            id: call_id.clone(),
            name: "session_git_commit".into(),
            arguments,
        }];
        let results = [(call_id, commit.content.clone())];
        let stage_error = self
            .stage_active_workspace_execution(&calls, &assistant_content, &results, &execution)
            .err();
        if let Some(error) = &stage_error {
            execution.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
            execution.detail = Some(match execution.detail.take() {
                Some(detail) => {
                    format!("{detail}; exact Git commit evidence could not be staged: {error:#}")
                }
                None => format!("exact Git commit evidence could not be staged: {error:#}"),
            });
        }

        let settlement = self
            .checkpoint_durable_workspace_with_execution(execution)
            .await;
        combine_commit_result(commit, reconciliation.err(), stage_error, settlement)
    }
}

fn commit_disposition(
    status: hi_tools::ToolStatus,
    reconciliation_failed: bool,
) -> hi_workspace::ExecutionDisposition {
    if reconciliation_failed {
        return hi_workspace::ExecutionDisposition::Indeterminate;
    }
    match status {
        hi_tools::ToolStatus::Succeeded => hi_workspace::ExecutionDisposition::Succeeded,
        hi_tools::ToolStatus::Cancelled => hi_workspace::ExecutionDisposition::Cancelled,
        hi_tools::ToolStatus::Failed
        | hi_tools::ToolStatus::Denied
        | hi_tools::ToolStatus::TimedOut => hi_workspace::ExecutionDisposition::Failed,
    }
}

fn commit_detail(
    commit: &hi_tools::CommitOutcome,
    reconciliation: Option<&anyhow::Error>,
) -> Option<String> {
    match (commit.succeeded_typed(), reconciliation) {
        (true, None) => None,
        (false, None) => Some(format!("Git commit failed: {}", commit.content)),
        (true, Some(error)) => Some(format!(
            "Git commit returned success, but its workspace effects could not be reconciled: {error:#}"
        )),
        (false, Some(error)) => Some(format!(
            "Git commit failed ({}), and its workspace effects could not be reconciled: {error:#}",
            commit.content
        )),
    }
}

fn combine_commit_result(
    commit: hi_tools::CommitOutcome,
    reconciliation: Option<anyhow::Error>,
    stage: Option<anyhow::Error>,
    settlement: anyhow::Result<()>,
) -> anyhow::Result<String> {
    if commit.succeeded_typed() && reconciliation.is_none() && stage.is_none() && settlement.is_ok()
    {
        return Ok(commit.content);
    }

    let mut failures = Vec::new();
    if !commit.succeeded_typed() {
        failures.push(format!("Git commit failed: {}", commit.content));
    }
    if let Some(error) = reconciliation {
        failures.push(format!(
            "Git commit effects could not be reconciled: {error:#}"
        ));
    }
    if let Some(error) = stage {
        failures.push(format!(
            "Git commit transcript could not be staged: {error:#}"
        ));
    }
    if let Err(error) = settlement {
        failures.push(format!("Git commit settlement failed: {error:#}"));
    }
    Err(anyhow::anyhow!(failures.join("; ")))
}
