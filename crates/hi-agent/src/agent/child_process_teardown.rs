//! Cancellation-safe native-process teardown for throwaway child agents.

use std::sync::Arc;

use anyhow::{Context as _, Result};

pub(super) struct ReapingChild {
    child: crate::Agent,
    teardown: Option<hi_tools::BackgroundTaskTeardown>,
    reaping_started: bool,
}

impl ReapingChild {
    pub(super) fn new(
        child: crate::Agent,
        teardown: Option<hi_tools::BackgroundTaskTeardown>,
    ) -> Self {
        if let Some(teardown) = &teardown {
            teardown.arm();
        }
        Self {
            child,
            teardown,
            reaping_started: false,
        }
    }

    pub(super) fn child(&self) -> &crate::Agent {
        &self.child
    }

    pub(super) fn child_mut(&mut self) -> &mut crate::Agent {
        &mut self.child
    }

    /// Signal all child-owned processes and await their detached drivers. The
    /// actual reap is spawned before this await, so dropping the caller cannot
    /// cancel cleanup or strand a task-teardown acknowledgement.
    pub(super) async fn stop_and_reap(&mut self) -> Result<()> {
        self.child.kill_background_processes();
        let background = self.child.background_process_registry();
        self.reaping_started = true;
        await_reap(background, self.teardown.clone()).await
    }

    pub(super) async fn failure_after_reap(&mut self, detail: impl Into<String>) -> String {
        let detail = detail.into();
        match self.stop_and_reap().await {
            Ok(()) => detail,
            Err(error) => format!("{detail}; child process teardown failed: {error:#}"),
        }
    }
}

impl Drop for ReapingChild {
    fn drop(&mut self) {
        if self.reaping_started {
            return;
        }
        self.child.kill_background_processes();
        let background = self.child.background_process_registry();
        let teardown = self.teardown.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = reap(background, teardown).await;
            });
        } else if let Some(teardown) = teardown {
            teardown.finish(Err(
                "child runtime disappeared before native processes could be reaped".into(),
            ));
        }
    }
}

async fn await_reap(
    background: Arc<hi_tools::BackgroundRegistry>,
    teardown: Option<hi_tools::BackgroundTaskTeardown>,
) -> Result<()> {
    let fallback = teardown.clone();
    match tokio::spawn(reap(background, teardown)).await {
        Ok(result) => result,
        Err(error) => {
            let detail = format!("child process reaper task failed: {error}");
            if let Some(teardown) = fallback {
                teardown.finish(Err(detail.clone()));
            }
            Err(anyhow::anyhow!(detail))
        }
    }
}

async fn reap(
    background: Arc<hi_tools::BackgroundRegistry>,
    teardown: Option<hi_tools::BackgroundTaskTeardown>,
) -> Result<()> {
    let result = background
        .ensure_quiescent_and_reaped()
        .await
        .context("waiting for child-owned native processes to be reaped");
    if let Some(teardown) = teardown {
        teardown.finish(
            result
                .as_ref()
                .map(|_| ())
                .map_err(|error| format!("{error:#}")),
        );
    }
    result
}
