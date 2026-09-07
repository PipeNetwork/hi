//! A bounded owner for synchronous session stores. Runtime callers await commits
//! without occupying Tokio workers; cancelling a waiter does not cancel a write.
use crate::SessionSink;
use anyhow::{Result, anyhow};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, mpsc};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

const CAPACITY: usize = 128;
type Operation = Box<dyn FnOnce(&mut dyn SessionSink) -> Result<()> + Send>;

#[derive(Default)]
struct FailureLog {
    receipts: BTreeMap<u64, String>,
    overflow: u64,
    acknowledged_overflow: u64,
}
struct Reply {
    result: std::result::Result<(), String>,
    failures: Vec<u64>,
    overflow: u64,
}
enum ReplySender {
    Async(oneshot::Sender<Reply>),
    Blocking(mpsc::Sender<Reply>),
}
impl ReplySender {
    fn send(self, reply: Reply) {
        match self {
            Self::Async(sender) => {
                let _ = sender.send(reply);
            }
            Self::Blocking(sender) => {
                let _ = sender.send(reply);
            }
        }
    }
}
struct Work {
    operation: Option<Operation>,
    reply: ReplySender,
    _permit: OwnedSemaphorePermit,
}
struct Inner {
    sender: mpsc::Sender<Work>,
    capacity: Arc<Semaphore>,
    failures: Arc<Mutex<FailureLog>>,
}

#[derive(Clone)]
pub struct SessionIoHandle(Arc<Inner>);

impl SessionIoHandle {
    /// Return only after this operation committed or reported its failure.
    pub async fn write<F>(&self, operation: F) -> Result<()>
    where
        F: FnOnce(&mut dyn SessionSink) -> Result<()> + Send + 'static,
    {
        self.submit(Some(Box::new(operation))).await
    }

    /// Drain earlier accepted work, surfacing failures whose waiter disappeared.
    pub async fn barrier(&self) -> Result<()> {
        self.submit(None).await
    }

    async fn submit(&self, operation: Option<Operation>) -> Result<()> {
        let permit = self
            .0
            .capacity
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| anyhow!("session writer closed"))?;
        let (sender, receiver) = oneshot::channel();
        self.0
            .sender
            .send(Work {
                operation,
                reply: ReplySender::Async(sender),
                _permit: permit,
            })
            .map_err(|_| anyhow!("session writer stopped before accepting work"))?;
        let reply = receiver
            .await
            .map_err(|_| anyhow!("session writer stopped before acknowledging work"))?;
        self.acknowledge(reply)
    }

    fn write_blocking<F>(&self, operation: F) -> Result<()>
    where
        F: FnOnce(&mut dyn SessionSink) -> Result<()> + Send + 'static,
    {
        // Compatibility calls may commit only while the owner is idle. Acquiring
        // all permits atomically prevents blocking a runtime/UI thread behind
        // an accepted append whose original async waiter has disappeared.
        let permit = self
            .0
            .capacity
            .clone()
            .try_acquire_many_owned(CAPACITY as u32)
            .map_err(|_| {
                anyhow!(
                    "session writer has pending work; await its barrier before a synchronous update"
                )
            })?;
        let (sender, receiver) = mpsc::channel();
        self.0
            .sender
            .send(Work {
                operation: Some(Box::new(operation)),
                reply: ReplySender::Blocking(sender),
                _permit: permit,
            })
            .map_err(|_| anyhow!("session writer stopped before accepting work"))?;
        let reply = receiver
            .recv()
            .map_err(|_| anyhow!("session writer stopped before acknowledging work"))?;
        self.acknowledge(reply)
    }

    fn acknowledge(&self, reply: Reply) -> Result<()> {
        let mut failures = self
            .0
            .failures
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for id in reply.failures {
            failures.receipts.remove(&id);
        }
        failures.acknowledged_overflow = failures.acknowledged_overflow.max(reply.overflow);
        reply.result.map_err(anyhow::Error::msg)
    }
}

/// Explicit blocking compatibility for startup/public synchronous setters.
/// Active turns obtain `io_handle()` and await owned operations instead.
pub(crate) struct OwnedSessionSink {
    handle: SessionIoHandle,
    id: Option<String>,
    local_stage: bool,
}
impl OwnedSessionSink {
    pub(crate) fn new(mut sink: Box<dyn SessionSink>) -> Self {
        let id = sink.id();
        let local_stage = sink.requires_local_workspace_execution_stage();
        let (sender, receiver) = mpsc::channel::<Work>();
        let failures = Arc::new(Mutex::new(FailureLog::default()));
        let worker_failures = failures.clone();
        std::thread::Builder::new()
            .name("hi-session-writer".into())
            .spawn(move || {
                let mut sequence = 0_u64;
                while let Ok(work) = receiver.recv() {
                    sequence = sequence.saturating_add(1);
                    let reply = if let Some(operation) = work.operation {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            operation(&mut *sink)
                        }))
                        .map_err(|_| "session writer operation panicked".to_owned())
                        .and_then(|result| result.map_err(|error| format!("{error:#}")));
                        if let Err(error) = &result {
                            let mut log = worker_failures
                                .lock()
                                .unwrap_or_else(|error| error.into_inner());
                            if log.receipts.len() < CAPACITY {
                                log.receipts
                                    .insert(sequence, error.chars().take(8_192).collect());
                            } else {
                                log.overflow = log.overflow.saturating_add(1);
                            }
                        }
                        Reply {
                            result,
                            failures: vec![sequence],
                            overflow: 0,
                        }
                    } else {
                        let log = worker_failures
                            .lock()
                            .unwrap_or_else(|error| error.into_inner());
                        let result = if let Some(error) = log.receipts.values().next() {
                            Err(error.clone())
                        } else if log.overflow > log.acknowledged_overflow {
                            Err("earlier session writes failed without acknowledgement".into())
                        } else {
                            Ok(())
                        };
                        Reply {
                            result,
                            failures: log.receipts.keys().copied().collect(),
                            overflow: log.overflow,
                        }
                    };
                    drop(work._permit);
                    work.reply.send(reply);
                }
            })
            .expect("start session writer");
        Self {
            handle: SessionIoHandle(Arc::new(Inner {
                sender,
                capacity: Arc::new(Semaphore::new(CAPACITY)),
                failures,
            })),
            id,
            local_stage,
        }
    }
}

/// Provides ownership for optional metadata exports without creating a saved
/// transcript. Ordinary no-session persistence never submits work here.
struct EphemeralMetadataSink;
impl SessionSink for EphemeralMetadataSink {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> Result<()> {
        Ok(())
    }
    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> Result<()> {
        Ok(())
    }
}

impl crate::Agent {
    pub(crate) fn write_session<F>(
        &self,
        operation: F,
    ) -> impl std::future::Future<Output = Result<()>> + Send + 'static + use<F>
    where
        F: FnOnce(&mut dyn SessionSink) -> Result<()> + Send + 'static,
    {
        let handle = self
            .session
            .as_ref()
            .map(|sink| {
                sink.io_handle()
                    .ok_or_else(|| anyhow!("session sink has no I/O owner"))
            })
            .transpose();
        async move {
            match handle? {
                Some(handle) => handle.write(operation).await,
                None => Ok(()),
            }
        }
    }
    /// Goal metadata can exist without a saved transcript. Start its owner only
    /// when an export is requested, and retain it if a real sink is attached.
    pub(crate) fn write_session_metadata<F>(
        &mut self,
        operation: F,
    ) -> impl std::future::Future<Output = Result<()>> + Send + 'static + use<F>
    where
        F: FnOnce(&mut dyn SessionSink) -> Result<()> + Send + 'static,
    {
        let saved = self
            .session
            .as_ref()
            .map(|sink| {
                sink.io_handle()
                    .ok_or_else(|| anyhow!("session sink has no I/O owner"))
            })
            .transpose();
        if self.session.is_none() && self.session_metadata_io.is_none() {
            self.session_metadata_io =
                Some(OwnedSessionSink::new(Box::new(EphemeralMetadataSink)).handle);
        }
        let metadata = self.session_metadata_io.clone();
        async move {
            match saved? {
                Some(saved) => {
                    // An old accepted export must finish before the newly
                    // attached sink can publish a newer view of the same goal.
                    if let Some(metadata) = metadata {
                        metadata.barrier().await?;
                    }
                    saved.write(operation).await
                }
                None => {
                    metadata
                        .expect("metadata owner initialized")
                        .write(operation)
                        .await
                }
            }
        }
    }

    pub(crate) fn has_session_io(&self) -> bool {
        self.session.is_some() || self.session_metadata_io.is_some()
    }

    pub fn session_barrier(
        &self,
    ) -> impl std::future::Future<Output = Result<()>> + Send + 'static + use<> {
        let saved = self
            .session
            .as_ref()
            .map(|sink| {
                sink.io_handle()
                    .ok_or_else(|| anyhow!("session sink has no I/O owner"))
            })
            .transpose();
        let metadata = self.session_metadata_io.clone();
        async move {
            let (saved, metadata) = tokio::join!(
                async {
                    match saved? {
                        Some(handle) => handle.barrier().await,
                        None => Ok(()),
                    }
                },
                async {
                    match metadata {
                        Some(handle) => handle.barrier().await,
                        None => Ok(()),
                    }
                }
            );
            match (saved, metadata) {
                (Err(saved), Err(metadata)) => {
                    Err(anyhow!("{saved:#}; metadata export: {metadata:#}"))
                }
                (saved, metadata) => saved.and(metadata),
            }
        }
    }
}

impl SessionSink for OwnedSessionSink {
    fn id(&self) -> Option<String> {
        self.id.clone()
    }
    fn requires_local_workspace_execution_stage(&self) -> bool {
        self.local_stage
    }
    fn io_handle(&self) -> Option<SessionIoHandle> {
        Some(self.handle.clone())
    }
    fn record_model_context(&mut self, model: &str, window: Option<u32>) {
        let model = model.to_owned();
        if let Err(error) = self.handle.write_blocking(move |sink| {
            sink.record_model_context(&model, window);
            Ok(())
        }) {
            tracing::warn!(%error, "session model context publication failed");
        }
    }
    fn record(&mut self, messages: &[hi_ai::Message], usage: hi_ai::Usage) -> Result<()> {
        let messages = messages.to_vec();
        self.handle
            .write_blocking(move |sink| sink.record(&messages, usage))
    }
    fn record_compaction(&mut self, messages: &[hi_ai::Message]) -> Result<()> {
        let messages = messages.to_vec();
        self.handle
            .write_blocking(move |sink| sink.record_compaction(&messages))
    }
    fn stage_workspace_execution(
        &mut self,
        record: &crate::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        let record = record.clone();
        self.handle
            .write_blocking(move |sink| sink.stage_workspace_execution(&record))
    }
    fn stage_local_workspace_execution(
        &mut self,
        record: &crate::WorkspaceTranscriptExecution,
        visible: bool,
    ) -> Result<()> {
        let record = record.clone();
        self.handle
            .write_blocking(move |sink| sink.stage_local_workspace_execution(&record, visible))
    }
    fn settle_local_workspace_execution(&mut self, id: &hi_workspace::OperationId) -> Result<()> {
        let id = id.clone();
        self.handle
            .write_blocking(move |sink| sink.settle_local_workspace_execution(&id))
    }
    fn record_state_replacement(
        &mut self,
        messages: &[hi_ai::Message],
        goal: Option<&crate::Goal>,
        decisions: &crate::DecisionLog,
        plan: &[crate::PlanStep],
    ) -> Result<()> {
        let messages = messages.to_vec();
        let goal = goal.cloned();
        let decisions = decisions.clone();
        let plan = plan.to_vec();
        self.handle.write_blocking(move |sink| {
            sink.record_state_replacement(&messages, goal.as_ref(), &decisions, &plan)
        })
    }
    fn record_checkpoints(&mut self, refs: &[String]) -> Result<()> {
        let refs = refs.to_vec();
        self.handle
            .write_blocking(move |sink| sink.record_checkpoints(&refs))
    }
    fn record_pipefs_mode(&mut self, enabled: bool) -> Result<()> {
        self.handle
            .write_blocking(move |sink| sink.record_pipefs_mode(enabled))
    }
    fn record_goal(&mut self, goal: &crate::Goal) -> Result<()> {
        let goal = goal.clone();
        self.handle
            .write_blocking(move |sink| sink.record_goal(&goal))
    }
    fn clear_goal(&mut self) -> Result<()> {
        self.handle.write_blocking(move |sink| sink.clear_goal())
    }
    fn record_plan(&mut self, plan: &[crate::PlanStep]) -> Result<()> {
        let plan = plan.to_vec();
        self.handle
            .write_blocking(move |sink| sink.record_plan(&plan))
    }
    fn clear_plan(&mut self) -> Result<()> {
        self.handle.write_blocking(move |sink| sink.clear_plan())
    }
    fn record_plan_drive(&mut self, paused: bool, stall: u32) -> Result<()> {
        self.handle
            .write_blocking(move |sink| sink.record_plan_drive(paused, stall))
    }
    fn record_plan_drive_state(
        &mut self,
        paused: bool,
        stall: u32,
        reset: bool,
        evidence: &[String],
    ) -> Result<()> {
        let evidence = evidence.to_vec();
        self.handle.write_blocking(move |sink| {
            sink.record_plan_drive_state(paused, stall, reset, &evidence)
        })
    }
    fn record_plan_drive_state_with_policy(
        &mut self,
        paused: bool,
        stall: u32,
        resume: bool,
        reset: bool,
        evidence: &[String],
    ) -> Result<()> {
        let evidence = evidence.to_vec();
        self.handle.write_blocking(move |sink| {
            sink.record_plan_drive_state_with_policy(paused, stall, resume, reset, &evidence)
        })
    }
    fn record_plan_approval_parked(&mut self, parked: bool) -> Result<()> {
        self.handle
            .write_blocking(move |sink| sink.record_plan_approval_parked(parked))
    }
    fn record_task_recovery(&mut self, state: &crate::TaskRecoveryState) -> Result<()> {
        let state = state.clone();
        self.handle
            .write_blocking(move |sink| sink.record_task_recovery(&state))
    }
    fn record_goal_drive(&mut self, stall: u32) -> Result<()> {
        self.handle
            .write_blocking(move |sink| sink.record_goal_drive(stall))
    }
    fn record_goal_drive_state(
        &mut self,
        stall: u32,
        reset: bool,
        evidence: &[String],
    ) -> Result<()> {
        let evidence = evidence.to_vec();
        self.handle
            .write_blocking(move |sink| sink.record_goal_drive_state(stall, reset, &evidence))
    }
    fn record_decisions(&mut self, decisions: &crate::DecisionLog) -> Result<()> {
        let decisions = decisions.clone();
        self.handle
            .write_blocking(move |sink| sink.record_decisions(&decisions))
    }
    fn record_turn_outcome(
        &mut self,
        outcome: &crate::TurnOutcome,
        unavailable: Option<&str>,
    ) -> Result<()> {
        let outcome = outcome.clone();
        let unavailable = unavailable.map(str::to_owned);
        self.handle
            .write_blocking(move |sink| sink.record_turn_outcome(&outcome, unavailable.as_deref()))
    }

    fn record_turn_settlement(
        &mut self,
        outcome: &crate::TurnOutcome,
        review_unavailable_reason: Option<&str>,
        task_recovery: Option<&crate::TaskRecoveryState>,
        settled_goal: Option<&crate::Goal>,
    ) -> Result<()> {
        let outcome = outcome.clone();
        let unavailable = review_unavailable_reason.map(str::to_owned);
        let recovery = task_recovery.cloned();
        let goal = settled_goal.cloned();
        self.handle.write_blocking(move |sink| {
            sink.record_turn_settlement(
                &outcome,
                unavailable.as_deref(),
                recovery.as_ref(),
                goal.as_ref(),
            )
        })
    }
}

#[cfg(test)]
mod tests;
