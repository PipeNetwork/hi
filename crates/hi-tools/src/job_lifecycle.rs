//! Storage-neutral lifecycle callbacks for legacy background handles.

use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackgroundJobId {
    pub source_id: String,
    pub handle: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundJobKind {
    Process,
    ReadAgent,
    WriteCandidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundJobEffect {
    ReadOnly,
    CandidateOnly,
    LiveWriter,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackgroundJobRegistration {
    pub id: BackgroundJobId,
    pub kind: BackgroundJobKind,
    pub effect: BackgroundJobEffect,
    pub name: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundJobTerminal {
    Succeeded,
    ReadyToMerge,
    Failed,
    FailedBeforeStart,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackgroundJobPublication {
    Published,
    DurabilityPending,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackgroundCandidateTransition {
    Merging,
    Settling,
    Succeeded,
    Failed,
    RecoveryRequired,
    Stale,
}

#[async_trait]
pub trait BackgroundJobLifecycle: Send + Sync {
    async fn register(&self, registration: BackgroundJobRegistration) -> Result<(), String>;

    /// Called only after the underlying process/task has stopped. A live
    /// writer returns `DurabilityPending`; success remains unpublished until
    /// a later workspace receipt settles the returned pending identity.
    async fn observe_terminal(
        &self,
        id: &BackgroundJobId,
        terminal: BackgroundJobTerminal,
        detail: Option<String>,
    ) -> Result<BackgroundJobPublication, String>;

    async fn observe_terminal_with_artifacts(
        &self,
        id: &BackgroundJobId,
        terminal: BackgroundJobTerminal,
        detail: Option<String>,
        _artifacts: Vec<hi_workspace::ArtifactRef>,
    ) -> Result<BackgroundJobPublication, String> {
        self.observe_terminal(id, terminal, detail).await
    }

    async fn pending(&self, source_id: &str) -> Vec<BackgroundJobId>;

    async fn settle_after_workspace(&self, pending: &[BackgroundJobId]) -> Result<(), String>;

    async fn workspace_job_id(&self, _id: &BackgroundJobId) -> Option<String> {
        None
    }

    async fn workspace_job_verification_ms(&self, _id: &BackgroundJobId) -> Option<u64> {
        None
    }

    async fn transition_candidate(
        &self,
        _id: &BackgroundJobId,
        _transition: BackgroundCandidateTransition,
        _detail: Option<String>,
    ) -> Result<(), String> {
        Err("candidate lifecycle transitions are unavailable".into())
    }
}

pub(crate) struct BackgroundJobLifecycleSlot {
    source_id: String,
    lifecycle: RwLock<Option<Arc<dyn BackgroundJobLifecycle>>>,
}

impl Default for BackgroundJobLifecycleSlot {
    fn default() -> Self {
        Self {
            source_id: uuid::Uuid::new_v4().to_string(),
            lifecycle: RwLock::new(None),
        }
    }
}

impl BackgroundJobLifecycleSlot {
    pub(crate) fn set(&self, lifecycle: Arc<dyn BackgroundJobLifecycle>) {
        *self
            .lifecycle
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(lifecycle);
    }

    pub(crate) async fn register(
        &self,
        handle: &str,
        kind: BackgroundJobKind,
        effect: BackgroundJobEffect,
        name: &str,
    ) -> Result<Option<ManagedBackgroundJob>, String> {
        let lifecycle = self
            .lifecycle
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let Some(lifecycle) = lifecycle else {
            return Ok(None);
        };
        let id = BackgroundJobId {
            source_id: self.source_id.clone(),
            handle: handle.to_owned(),
        };
        lifecycle
            .register(BackgroundJobRegistration {
                id: id.clone(),
                kind,
                effect,
                name: name.to_owned(),
            })
            .await?;
        Ok(Some(ManagedBackgroundJob { id, lifecycle }))
    }

    pub(crate) async fn pending(&self) -> Vec<BackgroundJobId> {
        let lifecycle = self
            .lifecycle
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        match lifecycle {
            Some(lifecycle) => lifecycle.pending(&self.source_id).await,
            None => Vec::new(),
        }
    }

    pub(crate) async fn settle_after_workspace(
        &self,
        pending: &[BackgroundJobId],
    ) -> Result<(), String> {
        let lifecycle = self
            .lifecycle
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        match lifecycle {
            Some(lifecycle) => lifecycle.settle_after_workspace(pending).await,
            None => Ok(()),
        }
    }
}

#[derive(Clone)]
pub(crate) struct ManagedBackgroundJob {
    id: BackgroundJobId,
    lifecycle: Arc<dyn BackgroundJobLifecycle>,
}

impl ManagedBackgroundJob {
    pub(crate) async fn observe(
        &self,
        terminal: BackgroundJobTerminal,
        detail: Option<String>,
    ) -> Result<BackgroundJobPublication, String> {
        self.lifecycle
            .observe_terminal(&self.id, terminal, detail)
            .await
    }

    pub(crate) async fn observe_with_artifacts(
        &self,
        terminal: BackgroundJobTerminal,
        detail: Option<String>,
        artifacts: Vec<hi_workspace::ArtifactRef>,
    ) -> Result<BackgroundJobPublication, String> {
        self.lifecycle
            .observe_terminal_with_artifacts(&self.id, terminal, detail, artifacts)
            .await
    }

    pub(crate) async fn workspace_job_id(&self) -> Option<String> {
        self.lifecycle.workspace_job_id(&self.id).await
    }

    pub(crate) async fn workspace_job_verification_ms(&self) -> Option<u64> {
        self.lifecycle.workspace_job_verification_ms(&self.id).await
    }

    pub(crate) async fn transition_candidate(
        &self,
        transition: BackgroundCandidateTransition,
        detail: Option<String>,
    ) -> Result<(), String> {
        self.lifecycle
            .transition_candidate(&self.id, transition, detail)
            .await
    }
}
