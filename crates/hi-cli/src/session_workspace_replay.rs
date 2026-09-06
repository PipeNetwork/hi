use std::collections::HashMap;

use anyhow::{Result, ensure};
use hi_ai::Message;

struct StagedExecution {
    execution: hi_agent::WorkspaceTranscriptExecution,
    visible_on_resume: bool,
    after_message_index: usize,
    settled: bool,
    retired: bool,
}

#[derive(Default)]
pub(super) struct WorkspaceExecutionReplay {
    entries: Vec<StagedExecution>,
    by_operation: HashMap<String, usize>,
}

impl WorkspaceExecutionReplay {
    pub(super) fn stage(
        &mut self,
        execution: hi_agent::WorkspaceTranscriptExecution,
        visible_on_resume: bool,
        after_message_index: usize,
    ) -> Result<()> {
        ensure!(
            execution.schema_version == hi_agent::WorkspaceTranscriptExecution::SCHEMA_VERSION,
            "unsupported workspace execution stage schema {}",
            execution.schema_version
        );
        let operation = execution.operation_id.to_string();
        if let Some(index) = self.by_operation.get(&operation).copied() {
            let existing = &self.entries[index];
            ensure!(
                existing.visible_on_resume == visible_on_resume
                    && serde_json::to_value(&existing.execution)?
                        == serde_json::to_value(&execution)?,
                "workspace execution {operation} was staged with conflicting evidence"
            );
            return Ok(());
        }
        self.by_operation.insert(operation, self.entries.len());
        self.entries.push(StagedExecution {
            execution,
            visible_on_resume,
            after_message_index,
            settled: false,
            retired: false,
        });
        Ok(())
    }

    pub(super) fn settle(&mut self, operation_id: hi_workspace::OperationId) {
        if let Some(index) = self.by_operation.get(operation_id.as_str()).copied() {
            self.entries[index].settled = true;
        }
    }

    /// A later replacement/compaction already contains the intended durable
    /// conversation, so older settled stages must not be resurrected.
    pub(super) fn retire_settled(&mut self) {
        for entry in &mut self.entries {
            if entry.settled {
                entry.retired = true;
            }
        }
    }

    pub(super) fn finish(self, messages: &mut Vec<Message>) -> bool {
        let mut inserted = 0usize;
        let mut recovered_settled_execution = false;
        for entry in self.entries.into_iter().filter(|entry| !entry.retired) {
            let boundary = entry
                .after_message_index
                .saturating_add(inserted)
                .min(messages.len());
            if entry.settled && entry.visible_on_resume {
                let expected = execution_messages(&entry.execution);
                let restored = merge_at(messages, &expected, boundary);
                recovered_settled_execution |= restored != 0;
                inserted = inserted.saturating_add(restored);
            }
            if !entry.settled {
                let warning = Message::assistant(vec![hi_ai::Content::Text(recovery_warning(
                    &entry.execution,
                ))]);
                inserted = inserted.saturating_add(merge_at(messages, &[warning], boundary));
            }
        }
        recovered_settled_execution
    }
}

fn execution_messages(execution: &hi_agent::WorkspaceTranscriptExecution) -> Vec<Message> {
    let mut messages = Vec::with_capacity(1 + execution.calls.len());
    if !execution.assistant_content.is_empty() {
        messages.push(Message::assistant(execution.assistant_content.clone()));
    }
    messages.extend(execution.calls.iter().map(|call| {
        let (result, _) = hi_tools::bound_tool_content(call.result.clone());
        Message::tool_result(call.call_id.clone(), result)
    }));
    messages
}

fn merge_at(messages: &mut Vec<Message>, expected: &[Message], boundary: usize) -> usize {
    if expected.is_empty() {
        return 0;
    }
    let available = messages.len().saturating_sub(boundary).min(expected.len());
    let matching_prefix = (0..available)
        .take_while(|offset| message_eq(&messages[boundary + *offset], &expected[*offset]))
        .count();
    let missing = expected.len().saturating_sub(matching_prefix);
    if missing != 0 {
        messages.splice(
            boundary + matching_prefix..boundary + matching_prefix,
            expected[matching_prefix..].iter().cloned(),
        );
    }
    missing
}

fn message_eq(left: &Message, right: &Message) -> bool {
    serde_json::to_value(left).ok() == serde_json::to_value(right).ok()
}

fn recovery_warning(execution: &hi_agent::WorkspaceTranscriptExecution) -> String {
    format!(
        "[workspace recovery pending] Operation {} completed execution but its local settlement acknowledgement was interrupted. The exact result is retained; inspect `hi workspace status` and `hi workspace recover list` before retrying the effect.",
        execution.operation_id
    )
}
