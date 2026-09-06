//! Terminal UI publication fence for settlement-bearing ordinary tool calls.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context as _, Result};
use hi_ai::Content;

use crate::Ui;
use crate::heuristics::emit_tool_output;

use super::policy::{
    merge_reconciled_changes, workspace_execution_report, workspace_operation_requires_settlement,
};

struct PendingTerminal {
    id: String,
    name: String,
    output: Option<hi_tools::ToolOutcome>,
}

/// Holds terminal UI events whose effects still need transcript/workspace
/// settlement. Tool-start and streaming events remain live; only the terminal
/// success/failure is fenced. Independent read-only calls bypass this queue,
/// while a read that depends on a fenced mutation stays behind the same fence
/// so terminal results cannot be published out of dependency order.
pub(super) struct DeferredToolTerminals {
    admitted: BTreeSet<String>,
    index: BTreeMap<String, usize>,
    pending: Vec<PendingTerminal>,
}

impl DeferredToolTerminals {
    pub(super) fn for_admitted_calls(calls: &[(String, String, String)]) -> Self {
        let deps = crate::heuristics::tool_deps(calls);
        let mut waits_for_settlement = vec![false; calls.len()];
        for (index, (_, name, arguments)) in calls.iter().enumerate() {
            waits_for_settlement[index] = workspace_operation_requires_settlement(name, arguments)
                || name == "bash_kill"
                || deps[index]
                    .iter()
                    .any(|dependency| waits_for_settlement[*dependency]);
        }
        let admitted = calls
            .iter()
            .zip(waits_for_settlement)
            .filter(|(_, waits)| *waits)
            .map(|((id, _, _), _)| id.clone())
            .collect();
        Self {
            admitted,
            index: BTreeMap::new(),
            pending: Vec::new(),
        }
    }

    /// Remember an admitted live lifecycle before execution. If a later
    /// bookkeeping/reconciliation step fails before producing its ToolOutcome,
    /// the failure path can still close this started call exactly once.
    pub(super) fn started(&mut self, id: &str, name: &str) {
        if self.admitted.contains(id) {
            self.ensure_pending(id, name);
        }
    }

    pub(super) fn emit(
        &mut self,
        ui: &mut dyn Ui,
        id: &str,
        name: &str,
        output: &hi_tools::ToolOutcome,
    ) {
        if !self.admitted.contains(id) {
            emit_tool_output(ui, id, name, output);
            return;
        }
        let index = self.ensure_pending(id, name);
        let first = self.pending[index].output.replace(output.clone()).is_none();
        debug_assert!(first, "terminal tool result queued more than once for {id}");
    }

    pub(super) fn publish(&mut self, ui: &mut dyn Ui) {
        for terminal in self.pending.drain(..) {
            let output = terminal.output.unwrap_or_else(|| {
                unresolved_terminal("workspace settlement completed without a tool outcome")
            });
            emit_tool_output(ui, &terminal.id, &terminal.name, &output);
        }
    }

    pub(super) fn fail(&mut self, ui: &mut dyn Ui, detail: impl std::fmt::Display) {
        let detail = detail.to_string();
        for terminal in self.pending.drain(..) {
            let executed = terminal
                .output
                .unwrap_or_else(|| unresolved_terminal("tool execution outcome is unavailable"));
            let output = publication_failure(&executed, &detail);
            emit_tool_output(ui, &terminal.id, &terminal.name, &output);
        }
    }

    pub(super) fn guard<T>(&mut self, ui: &mut dyn Ui, result: Result<T>) -> Result<T> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                self.fail(ui, format!("post-execution lifecycle failed: {error:#}"));
                let detail = format!("{error:#}");
                Err(error).context(format!("post-execution tool lifecycle failed: {detail}"))
            }
        }
    }

    fn ensure_pending(&mut self, id: &str, name: &str) -> usize {
        if let Some(index) = self.index.get(id) {
            return *index;
        }
        let index = self.pending.len();
        self.pending.push(PendingTerminal {
            id: id.to_owned(),
            name: name.to_owned(),
            output: None,
        });
        self.index.insert(id.to_owned(), index);
        index
    }
}

/// Stage the exact provider-facing batch, settle its workspace operation, and
/// only then publish terminal UI events. An ambiguous boundary replaces every
/// queued terminal with one failed/indeterminate result before returning.
#[allow(clippy::too_many_arguments)]
pub(super) async fn publish_after_settlement(
    agent: &mut crate::Agent,
    terminals: &mut DeferredToolTerminals,
    ui: &mut dyn Ui,
    intent: Option<&hi_workspace::MutationIntent>,
    calls: &[(String, String, String)],
    completion_content: &[Content],
    results: &[(String, String)],
    batch_entries: &[crate::ToolCallEntry],
    feedback_changes: &[hi_tools::FileChange],
) -> Result<()> {
    let Some(intent) = intent else {
        terminals.publish(ui);
        return Ok(());
    };
    let mut execution = workspace_execution_report(intent, batch_entries, calls.len());
    merge_reconciled_changes(&mut execution, feedback_changes);
    if let Err(stage_error) =
        agent.stage_visible_workspace_execution(calls, completion_content, results, &execution)
    {
        let mut indeterminate = execution;
        indeterminate.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
        indeterminate.detail = Some(format!(
            "workspace effects ran, but their transcript could not be staged: {stage_error:#}"
        ));
        let settlement = agent
            .checkpoint_durable_workspace_with_execution(indeterminate)
            .await;
        terminals.fail(
            ui,
            format!(
                "workspace effects ran, but their transcript could not be staged: {stage_error:#}"
            ),
        );
        return match settlement {
            Err(settlement_error) => Err(settlement_error).context(format!(
                "workspace transcript staging failed before settlement: {stage_error:#}"
            )),
            Ok(()) => Err(stage_error)
                .context("workspace transcript staging failed; execution remains indeterminate"),
        };
    }
    if let Err(settlement_error) = agent
        .checkpoint_durable_workspace_with_execution(execution)
        .await
    {
        terminals.fail(
            ui,
            format!(
                "workspace/transcript settlement acknowledgement was not durable: {settlement_error:#}"
            ),
        );
        return Err(settlement_error).context("tool batch workspace settlement failed");
    }
    terminals.publish(ui);
    Ok(())
}

fn publication_failure(output: &hi_tools::ToolOutcome, detail: &str) -> hi_tools::ToolOutcome {
    let combined = if output.content.is_empty() {
        format!("Error: workspace publication is indeterminate: {detail}")
    } else {
        format!(
            "Error: workspace publication is indeterminate: {detail}\n\nTool execution output:\n{}",
            output.content
        )
    };
    let (content, truncation) = hi_tools::bound_tool_content(combined);
    let mut terminal = output.clone();
    terminal.content = content;
    terminal.display = None;
    terminal.plan = None;
    terminal.status = hi_tools::ToolStatus::Failed;
    terminal.truncation = truncation;
    terminal
}

fn unresolved_terminal(content: &str) -> hi_tools::ToolOutcome {
    hi_tools::ToolOutcome {
        content: content.to_owned(),
        display: None,
        plan: None,
        status: hi_tools::ToolStatus::Failed,
        process: None,
        background: None,
        effects: hi_tools::ToolEffects::default(),
        truncation: hi_tools::TruncationState::Complete,
        images: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::turn::helpers::synthetic_tool_outcome;
    use crate::tests::common::RecordingUi;

    fn calls() -> Vec<(String, String, String)> {
        vec![
            (
                "write-1".into(),
                "write".into(),
                r#"{"path":"out.txt","content":"ok"}"#.into(),
            ),
            (
                "read-1".into(),
                "read".into(),
                r#"{"path":"in.txt"}"#.into(),
            ),
        ]
    }

    #[test]
    fn successful_mutation_terminal_waits_for_publication_but_read_does_not() {
        let mut subject = DeferredToolTerminals::for_admitted_calls(&calls());
        let mut ui = RecordingUi::default();
        let success = synthetic_tool_outcome("ok".into(), hi_tools::ToolStatus::Succeeded);

        subject.emit(&mut ui, "write-1", "write", &success);
        assert!(ui.tool_results.is_empty());
        subject.emit(&mut ui, "read-1", "read", &success);
        assert_eq!(ui.tool_results.len(), 1);
        assert_eq!(ui.tool_results[0].0, "read-1");

        subject.publish(&mut ui);
        assert_eq!(ui.tool_results.len(), 2);
        assert_eq!(ui.tool_results[1].0, "write-1");
        assert_eq!(ui.tool_results[1].3, hi_tools::ToolStatus::Succeeded);
    }

    #[test]
    fn ambiguous_publication_emits_one_failed_terminal() {
        let mut subject = DeferredToolTerminals::for_admitted_calls(&calls());
        let mut ui = RecordingUi::default();
        let success =
            synthetic_tool_outcome("bytes written".into(), hi_tools::ToolStatus::Succeeded);

        subject.emit(&mut ui, "write-1", "write", &success);
        subject.fail(&mut ui, "transcript stage unavailable");
        subject.fail(&mut ui, "duplicate settlement error");

        assert_eq!(ui.tool_results.len(), 1);
        assert_eq!(ui.tool_results[0].0, "write-1");
        assert_eq!(ui.tool_results[0].3, hi_tools::ToolStatus::Failed);
        assert!(
            ui.tool_results[0]
                .2
                .contains("publication is indeterminate")
        );
        assert!(
            ui.tool_results[0]
                .2
                .contains("transcript stage unavailable")
        );
    }

    #[test]
    fn dependent_read_terminal_waits_behind_denied_mutation() {
        let calls = vec![
            (
                "write-1".into(),
                "write".into(),
                r#"{"path":"out.txt","content":"ok"}"#.into(),
            ),
            (
                "read-1".into(),
                "read".into(),
                r#"{"path":"out.txt"}"#.into(),
            ),
        ];
        let mut subject = DeferredToolTerminals::for_admitted_calls(&calls);
        let mut ui = RecordingUi::default();
        let denied = synthetic_tool_outcome("skipped".into(), hi_tools::ToolStatus::Denied);
        let success = synthetic_tool_outcome("ok".into(), hi_tools::ToolStatus::Succeeded);

        subject.emit(&mut ui, "write-1", "write", &denied);
        subject.emit(&mut ui, "read-1", "read", &success);
        assert!(ui.tool_results.is_empty());

        subject.publish(&mut ui);
        assert_eq!(ui.tool_results.len(), 2);
        assert_eq!(ui.tool_results[0].0, "write-1");
        assert_eq!(ui.tool_results[0].3, hi_tools::ToolStatus::Denied);
        assert_eq!(ui.tool_results[1].0, "read-1");
        assert_eq!(ui.tool_results[1].3, hi_tools::ToolStatus::Succeeded);
    }
}
