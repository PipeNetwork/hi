//! Public compatibility types for background subagent tasks.

use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackgroundTaskCapacityError {
    pub maximum: usize,
    pub running: usize,
    pub unobserved_terminal: usize,
}

impl std::fmt::Display for BackgroundTaskCapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "background task capacity reached (max {}): {} running and {} completed/failed result(s) not yet observed; call get_task_output or wait_tasks for existing task IDs, then retry (or use kill_task for work no longer needed)",
            self.maximum, self.running, self.unobserved_terminal
        )
    }
}

impl std::error::Error for BackgroundTaskCapacityError {}

pub const MAX_WAIT_TIMEOUT: Duration = Duration::from_secs(600);
pub const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundTaskState {
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl BackgroundTaskState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackgroundTaskOutcome {
    pub id: String,
    pub description: String,
    pub subagent_type: String,
    pub state: BackgroundTaskState,
    pub output: String,
    pub applied: bool,
    pub changed_files: Vec<String>,
}

impl BackgroundTaskOutcome {
    pub fn running(id: &str, description: &str, subagent_type: &str) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            subagent_type: subagent_type.to_string(),
            state: BackgroundTaskState::Running,
            output: String::new(),
            applied: false,
            changed_files: Vec::new(),
        }
    }

    pub fn with_registry_identity(
        mut self,
        id: &str,
        description: &str,
        subagent_type: &str,
    ) -> Self {
        if self.id.is_empty() {
            self.id = id.to_string();
        }
        if self.description.is_empty() {
            self.description = description.to_string();
        }
        if self.subagent_type.is_empty() {
            self.subagent_type = subagent_type.to_string();
        }
        self
    }

    pub fn tool_status(&self) -> crate::ToolStatus {
        match self.state {
            BackgroundTaskState::Completed | BackgroundTaskState::Running => {
                crate::ToolStatus::Succeeded
            }
            BackgroundTaskState::Cancelled => crate::ToolStatus::Cancelled,
            BackgroundTaskState::Failed => crate::ToolStatus::Failed,
        }
    }
}

pub(crate) fn not_found_outcome(id: &str) -> BackgroundTaskOutcome {
    BackgroundTaskOutcome {
        id: id.to_string(),
        description: String::new(),
        subagent_type: String::new(),
        state: BackgroundTaskState::Failed,
        output: format!("no task with id \"{id}\""),
        applied: false,
        changed_files: Vec::new(),
    }
}

pub type BgFuture =
    std::pin::Pin<Box<dyn std::future::Future<Output = BackgroundTaskOutcome> + 'static>>;
