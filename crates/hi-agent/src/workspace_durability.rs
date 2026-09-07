use anyhow::Result;

/// Keep native backend staging off runtime workers. The owned callback retains
/// its record after a cancelled waiter; its admitted operation remains fenced.
pub(crate) async fn stage_execution_owned(
    backend: std::sync::Arc<dyn WorkspaceDurability>,
    record: crate::WorkspaceTranscriptExecution,
) -> Result<()> {
    tokio::task::spawn_blocking(move || backend.stage_workspace_execution(&record))
        .await
        .map_err(|error| anyhow::anyhow!("workspace staging owner failed: {error}"))?
}

/// Host-provided durability fence for a materialized workspace.
///
/// The agent remains unaware of archives or remote storage. It only marks the
/// start of a mutation and waits for the host to durably acknowledge the
/// resulting workspace before the tool batch is committed to the transcript.
#[async_trait::async_trait]
pub trait WorkspaceDurability: Send + Sync {
    /// Refuse when a previous revision is pending or this writer's lease is
    /// stale, then record a recovery marker before local bytes can change.
    async fn mutation_started(&self, dirty_paths: Option<Vec<String>>) -> Result<()>;

    /// Reconcile the materialized tree and durably commit any changed bytes.
    async fn checkpoint(&self) -> Result<()>;

    /// Durably stage the execution record which must be published with the
    /// next remote workspace receipt. Local durability backends never need
    /// this hook. The default is deliberately fail-closed because a remote
    /// controller must not settle bytes against a transcript batch that omits
    /// the native verifier which produced them.
    fn stage_workspace_execution(
        &self,
        _record: &crate::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        anyhow::bail!("this workspace durability backend cannot stage execution evidence")
    }

    /// Start or stop periodic reconciliation for a native background process.
    /// Implementations that do not need it may ignore the notification.
    async fn background_process_state(&self, _id: &str, _running: bool) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    };

    struct HeldStager {
        entered: tokio::sync::Notify,
        gate: (Mutex<bool>, Condvar),
        committed: AtomicBool,
    }
    #[async_trait::async_trait]
    impl WorkspaceDurability for HeldStager {
        async fn mutation_started(&self, _: Option<Vec<String>>) -> Result<()> {
            Ok(())
        }
        async fn checkpoint(&self) -> Result<()> {
            Ok(())
        }
        fn stage_workspace_execution(&self, _: &crate::WorkspaceTranscriptExecution) -> Result<()> {
            self.entered.notify_one();
            let mut released = self.gate.0.lock().unwrap();
            while !*released {
                released = self.gate.1.wait(released).unwrap();
            }
            self.committed.store(true, Ordering::Release);
            Ok(())
        }
    }

    #[tokio::test]
    async fn cancelled_native_staging_waiter_keeps_callback_off_runtime_and_owned() {
        let backend = Arc::new(HeldStager {
            entered: tokio::sync::Notify::new(),
            gate: (Mutex::new(false), Condvar::new()),
            committed: AtomicBool::new(false),
        });
        let record = crate::WorkspaceTranscriptExecution {
            schema_version: crate::WorkspaceTranscriptExecution::SCHEMA_VERSION,
            operation_id: hi_workspace::OperationId::new("held-native-stage"),
            assistant_content: Vec::new(),
            calls: Vec::new(),
            execution: hi_workspace::ExecutionReport::succeeded(None),
        };
        let mut stage = Box::pin(stage_execution_owned(backend.clone(), record));
        tokio::select! {
            result = stage.as_mut() => panic!("staging gate did not block: {result:?}"),
            _ = backend.entered.notified() => {},
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        drop(stage);
        assert!(!backend.committed.load(Ordering::Acquire));
        *backend.gate.0.lock().unwrap() = true;
        backend.gate.1.notify_all();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while !backend.committed.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}
