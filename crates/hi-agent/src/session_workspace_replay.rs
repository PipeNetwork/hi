//! Pure recovery of workspace execution outbox records into resumed history.

use std::collections::HashMap;

use hi_ai::Message;
use serde::{Deserialize, Serialize};

use super::SessionReduceError;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StagedExecution {
    execution: crate::WorkspaceTranscriptExecution,
    visible_on_resume: bool,
    after_message_index: usize,
    settled: bool,
    retired: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(super) struct WorkspaceExecutionReplay {
    entries: Vec<StagedExecution>,
    #[serde(skip)]
    by_operation: HashMap<String, usize>,
    #[serde(skip)]
    active: Vec<usize>,
}

impl WorkspaceExecutionReplay {
    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(super) fn has_pending(&self) -> bool {
        self.active
            .iter()
            .any(|index| !self.entries[*index].settled)
    }

    pub(super) fn restore_indexes(
        &mut self,
        message_count: usize,
    ) -> Result<(), SessionReduceError> {
        self.by_operation.clear();
        self.active.clear();
        let mut previous_boundary = 0;
        for (index, entry) in self.entries.iter().enumerate() {
            validate_execution(&entry.execution)?;
            if self
                .by_operation
                .insert(entry.execution.operation_id.to_string(), index)
                .is_some()
            {
                return Err(SessionReduceError::WorkspaceExecution(
                    "duplicate workspace operation in session snapshot".into(),
                ));
            }
            if !entry.retired {
                if entry.after_message_index > message_count
                    || entry.after_message_index < previous_boundary
                {
                    return Err(SessionReduceError::WorkspaceExecution(
                        "invalid workspace execution boundary in session snapshot".into(),
                    ));
                }
                previous_boundary = entry.after_message_index;
                self.active.push(index);
            } else if !entry.settled {
                return Err(SessionReduceError::WorkspaceExecution(
                    "unsettled workspace operation was retired in session snapshot".into(),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn stage(
        &mut self,
        execution: crate::WorkspaceTranscriptExecution,
        visible_on_resume: bool,
        after_message_index: usize,
    ) -> Result<(), SessionReduceError> {
        validate_execution(&execution)?;
        let operation = execution.operation_id.to_string();
        if let Some(index) = self.by_operation.get(&operation).copied() {
            let existing = &self.entries[index];
            if existing.visible_on_resume != visible_on_resume
                || serde_json::to_value(&existing.execution).ok()
                    != serde_json::to_value(&execution).ok()
            {
                return Err(SessionReduceError::WorkspaceExecution(format!(
                    "workspace execution {operation} was staged with conflicting evidence"
                )));
            }
            return Ok(());
        }
        self.by_operation.insert(operation, self.entries.len());
        self.active.push(self.entries.len());
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

    /// A replacement owns the visible conversation. Settled outbox records
    /// cannot resurrect discarded turns; unresolved operations remain visible.
    pub(super) fn replace_messages(&mut self, replacement_len: usize) {
        self.active.retain(|index| {
            let entry = &mut self.entries[*index];
            if entry.settled {
                entry.retired = true;
                false
            } else {
                entry.after_message_index = entry.after_message_index.min(replacement_len);
                true
            }
        });
    }

    /// Merge once, in order, instead of repeatedly splicing the entire suffix.
    /// Consuming the reducer makes recovery a projection, never an executed effect.
    pub(super) fn finish(self, messages: &mut Vec<Message>) -> bool {
        if self.active.is_empty() {
            return false;
        }
        let original = std::mem::take(messages);
        let mut source = original.into_iter().peekable();
        let mut consumed = 0usize;
        let mut restored_settled_execution = false;
        for entry in self.entries.into_iter().filter(|entry| !entry.retired) {
            while consumed < entry.after_message_index {
                let Some(message) = source.next() else { break };
                messages.push(message);
                consumed += 1;
            }
            let expected = if entry.settled && entry.visible_on_resume {
                execution_messages(&entry.execution)
            } else if !entry.settled {
                vec![Message::assistant(vec![hi_ai::Content::Text(format!(
                    "[workspace recovery pending] Operation {} completed execution but its local settlement acknowledgement was interrupted. The exact result is retained; inspect `hi workspace status` and `hi workspace recover list` before retrying the effect.",
                    entry.execution.operation_id
                ))])]
            } else {
                Vec::new()
            };
            let mut matching_prefix = true;
            for expected_message in expected {
                if matching_prefix
                    && source
                        .peek()
                        .is_some_and(|actual| message_eq(actual, &expected_message))
                {
                    messages.push(source.next().expect("matched source message"));
                    consumed += 1;
                } else {
                    matching_prefix = false;
                    restored_settled_execution |= entry.settled;
                    messages.push(expected_message);
                }
            }
        }
        messages.extend(source);
        restored_settled_execution
    }
}

fn validate_execution(
    execution: &crate::WorkspaceTranscriptExecution,
) -> Result<(), SessionReduceError> {
    if execution.schema_version != crate::WorkspaceTranscriptExecution::SCHEMA_VERSION {
        return Err(SessionReduceError::WorkspaceExecution(format!(
            "unsupported workspace execution stage schema {}",
            execution.schema_version
        )));
    }
    Ok(())
}

fn execution_messages(execution: &crate::WorkspaceTranscriptExecution) -> Vec<Message> {
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

fn message_eq(left: &Message, right: &Message) -> bool {
    serde_json::to_value(left).ok() == serde_json::to_value(right).ok()
}
