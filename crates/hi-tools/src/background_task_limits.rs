use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;

use super::{
    BACKGROUND_QUEUE_TIMEOUT, BackgroundTaskOutcome, BackgroundTaskState, BgFutureFactory,
    MAX_BG_TASKS, MAX_CONCURRENT_PREPARATIONS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackgroundTaskLimits {
    pub max_tasks: usize,
    pub max_concurrent_preparations: usize,
    pub queue_timeout: Duration,
}

impl Default for BackgroundTaskLimits {
    fn default() -> Self {
        Self {
            max_tasks: MAX_BG_TASKS,
            max_concurrent_preparations: MAX_CONCURRENT_PREPARATIONS,
            queue_timeout: BACKGROUND_QUEUE_TIMEOUT,
        }
    }
}

pub(super) async fn run_with_execution_slot(
    slots: Arc<Semaphore>,
    queue_timeout: Duration,
    future_factory: BgFutureFactory,
) -> BackgroundTaskOutcome {
    let permit = match tokio::time::timeout(queue_timeout, slots.acquire_owned()).await {
        Ok(Ok(permit)) => permit,
        Ok(Err(_)) => return registry_dropped_outcome(),
        Err(_) => return queue_timeout_outcome(queue_timeout),
    };
    let outcome = future_factory().await;
    drop(permit);
    outcome
}

pub(super) fn queue_timeout_outcome(timeout: Duration) -> BackgroundTaskOutcome {
    BackgroundTaskOutcome {
        id: String::new(),
        description: String::new(),
        subagent_type: String::new(),
        state: BackgroundTaskState::Failed,
        output: format!(
            "Background task exceeded its {:.1}-second queue limit.",
            timeout.as_secs_f64()
        ),
        applied: false,
        changed_files: Vec::new(),
    }
}

fn registry_dropped_outcome() -> BackgroundTaskOutcome {
    BackgroundTaskOutcome {
        id: String::new(),
        description: String::new(),
        subagent_type: String::new(),
        state: BackgroundTaskState::Cancelled,
        output: "Task cancelled because its registry was dropped.".into(),
        applied: false,
        changed_files: Vec::new(),
    }
}
