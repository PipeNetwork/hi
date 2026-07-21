//! Write-capable `delegate` subagent dispatch.
//!
//! The heavy lifting — a git worktree, a child `hi` subprocess, verification, and
//! applying only the verified diff back — lives behind the frontend-supplied
//! [`DelegateRunner`](crate::DelegateRunner) (it needs provider credentials and
//! subprocess/git plumbing the agent loop doesn't have). This method is the thin
//! dispatch: parse, budget, callout, invoke the runner, refresh the snapshot.

use serde_json::Value;

use crate::Ui;

fn delegate_tool_outcome(
    content: impl Into<String>,
    status: hi_tools::ToolStatus,
    mutation_attempted: bool,
    mutation_applied: bool,
) -> hi_tools::ToolOutcome {
    hi_tools::ToolOutcome {
        content: content.into(),
        display: None,
        plan: None,
        status,
        process: None,
        background: None,
        effects: hi_tools::ToolEffects {
            mutation_attempted,
            mutation_applied,
            file_changes: Vec::new(),
        },
        truncation: hi_tools::TruncationState::Complete,
    }
}

/// Cap on `delegate` subagents per session — lower than explore, since each is a
/// full write+verify run in an isolated worktree.
pub(crate) const MAX_DELEGATE_SUBAGENTS_PER_SESSION: u32 = 4;

impl crate::Agent {
    /// Run one write-capable `delegate` subagent and return a summary. The runner
    /// isolates it in a worktree and applies its changes back only if verification
    /// passes; on failure nothing touches the real tree (spatial isolation).
    pub(crate) async fn handle_delegate(
        &mut self,
        arguments: &str,
        ui: &mut dyn Ui,
    ) -> hi_tools::ToolOutcome {
        let parsed = serde_json::from_str::<Value>(arguments).ok();
        let task = parsed
            .as_ref()
            .and_then(|v| v.get("task").and_then(Value::as_str))
            .unwrap_or_default()
            .to_string();
        if task.trim().is_empty() {
            return delegate_tool_outcome(
                "delegate error: missing required \"task\" argument",
                hi_tools::ToolStatus::Failed,
                false,
                false,
            );
        }
        // Budget before runner so exhausted sessions get a clear budget message
        // even when a runner is attached (and tests that only set the counter).
        if self.subagents.delegate_subagents_used >= MAX_DELEGATE_SUBAGENTS_PER_SESSION {
            return delegate_tool_outcome(
                format!(
                    "delegate budget exhausted ({MAX_DELEGATE_SUBAGENTS_PER_SESSION} this session); \
                     implement the rest directly instead."
                ),
                hi_tools::ToolStatus::Denied,
                false,
                false,
            );
        }
        let Some(runner) = self.subagents.delegate_runner.clone() else {
            return delegate_tool_outcome(
                "delegate unavailable: no subagent runner is attached in this context; \
                 implement it directly instead.",
                hi_tools::ToolStatus::Denied,
                false,
                false,
            );
        };
        let n = self
            .subagents
            .try_begin_delegate(MAX_DELEGATE_SUBAGENTS_PER_SESSION)
            .expect("budget checked above");

        let verify = parsed
            .as_ref()
            .and_then(|v| v.get("verify").and_then(Value::as_str))
            .map(str::to_string)
            .filter(|s| !s.trim().is_empty());
        let summary: String = task.chars().take(72).collect();
        let ellipsis = if task.chars().count() > 72 { "…" } else { "" };
        ui.subagent_note(&format!(
            "↳ delegate subagent {n}/{MAX_DELEGATE_SUBAGENTS_PER_SESSION}: {summary}{ellipsis}"
        ));

        let ledger_revision = self.runtime.ledger().revision();
        let outcome = runner.run(&task, verify.as_deref()).await;
        let expected_paths = outcome
            .changed_files
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let mut output =
            delegate_tool_outcome(outcome.summary, outcome.status, true, outcome.applied);

        // The frontend applies through git/transaction plumbing outside the
        // normal tool engine. Reconcile it here, then attribute only the paths
        // the verified delegate reported. Concurrent user/editor changes still
        // enter the turn-level ledger, but never masquerade as delegate effects.
        match self.reconcile_workspace_changes() {
            Ok(()) => {
                let changes = self.runtime.ledger().changes_since(ledger_revision);
                let delegate_changes = changes
                    .into_iter()
                    .filter(|change| expected_paths.contains(&change.path))
                    .collect::<Vec<_>>();
                let actual_paths = delegate_changes
                    .iter()
                    .map(|change| change.path.clone())
                    .collect::<std::collections::BTreeSet<_>>();
                let exact_application =
                    outcome.applied && !expected_paths.is_empty() && actual_paths == expected_paths;
                output.effects.mutation_applied = !delegate_changes.is_empty();
                output.effects.file_changes = delegate_changes;
                if output.status == hi_tools::ToolStatus::Succeeded && !exact_application {
                    output.status = hi_tools::ToolStatus::Failed;
                    output.content.push_str(
                        "\nDelegate reported success without the exact applied workspace changes.",
                    );
                } else if output.status != hi_tools::ToolStatus::Succeeded
                    && output.effects.mutation_applied
                {
                    output.content.push_str(
                        "\nWarning: declared workspace changes remained after delegate failure.",
                    );
                }
            }
            Err(error) => {
                output.status = hi_tools::ToolStatus::Failed;
                output.content.push_str(&format!(
                    "\nFailed to reconcile delegate workspace effects: {error:#}"
                ));
                output.effects.file_changes.clear();
            }
        }

        if output.status == hi_tools::ToolStatus::Succeeded {
            ui.subagent_note(&format!(
                "↳ delegate subagent {n} applied — {} file(s) changed",
                output.effects.file_changes.len()
            ));
        } else {
            ui.subagent_note(&format!("↳ delegate subagent {n} rolled back"));
        }
        // The runner may have applied a diff to the working tree; refresh the
        // parent's snapshot AND clear the read cache so change detection, verify,
        // and any later `read` see the merged content — the merge writes files via
        // `git apply`, outside the edit-tool layer that normally invalidates.
        self.invalidate_snapshot();
        if output.effects.mutation_applied {
            self.runtime.clear_read_cache();
        }
        output
    }
}
