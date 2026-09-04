use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::watch;

use crate::job_output::{DEFAULT_OUTPUT_BYTES, JobOutputChunk, JobOutputStream, truncate_output};
use crate::{
    ArtifactRef, BarrierKind, BarrierReceipt, BarrierStatus, BindingId, ControllerId, EffectScope,
    JobCompletion, JobId, JobKind, JobPermit, JobSealOutcome, JobSealStatus, JobSpec, JobState,
    JobTerminal, RecoveryId, WORKSPACE_CONTRACT_SCHEMA_VERSION, WorkspaceBinding,
};

#[path = "job_registry_recovery.rs"]
mod recovery;

pub const DEFAULT_MAX_PREPARATIONS: usize = 4;
pub const DEFAULT_MAX_ACTIVE_JOBS: usize = 16;

/// Limits enforced by the shared registry rather than by an individual UI or
/// process implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRegistryLimits {
    pub max_preparations: usize,
    pub max_active_jobs: usize,
}

impl Default for JobRegistryLimits {
    fn default() -> Self {
        Self {
            max_preparations: DEFAULT_MAX_PREPARATIONS,
            max_active_jobs: DEFAULT_MAX_ACTIVE_JOBS,
        }
    }
}

/// A caller must present the controller, binding, and epoch it observed for
/// every lifecycle mutation. This prevents a callback from an old binding
/// from updating a new workspace that happens to reuse the same job ID.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobFence {
    pub controller_id: ControllerId,
    pub binding_id: BindingId,
    pub epoch: u64,
}

impl JobFence {
    pub fn from_binding(binding: &WorkspaceBinding) -> Self {
        Self {
            controller_id: binding.controller_id.clone(),
            binding_id: binding.binding_id.clone(),
            epoch: binding.epoch,
        }
    }

    fn matches_binding(&self, binding: &WorkspaceBinding) -> bool {
        self.controller_id == binding.controller_id
            && self.binding_id == binding.binding_id
            && self.epoch == binding.epoch
    }

    fn matches_permit(&self, permit: &JobPermit) -> bool {
        self.controller_id == permit.controller_id
            && self.binding_id == permit.binding_id
            && self.epoch == permit.epoch
    }
}

/// Serializable projection suitable for the control journal and all clients.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceJobSnapshot {
    pub permit: JobPermit,
    pub state: JobState,
    pub revision: u64,
    pub registered_at_ms: u64,
    pub updated_at_ms: u64,
    pub finalized_at_ms: Option<u64>,
    pub terminal: Option<JobTerminal>,
    #[serde(default)]
    pub recovery_id: Option<RecoveryId>,
    pub detail: Option<String>,
    pub artifacts: Vec<ArtifactRef>,
    pub output: Vec<JobOutputChunk>,
    pub output_truncated: bool,
    pub output_bytes: usize,
    pub next_output_sequence: u64,
}

impl WorkspaceJobSnapshot {
    pub fn is_actionable(&self) -> bool {
        self.state.is_terminal()
            || matches!(
                self.state,
                JobState::ReadyToMerge | JobState::DurabilityPending | JobState::RecoveryRequired
            )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceJobsSnapshot {
    pub controller_id: ControllerId,
    pub binding_id: BindingId,
    pub epoch: u64,
    pub sequence: u64,
    pub limits: JobRegistryLimits,
    pub jobs: Vec<WorkspaceJobSnapshot>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRestartReport {
    pub recovery_required: Vec<JobId>,
    pub orphaned: Vec<JobId>,
    pub stale: Vec<JobId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobWaitCondition {
    RevisionAfter(u64),
    Actionable,
    Terminal,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum JobRegistryError {
    #[error("job registry limits must both be greater than zero")]
    InvalidLimits,
    #[error("job callback is fenced by a different workspace binding or epoch")]
    StaleFence,
    #[error("job {0} was not found")]
    NotFound(JobId),
    #[error("job {job_id} is already terminal in state {state:?}")]
    AlreadyTerminal { job_id: JobId, state: JobState },
    #[error("job {job_id} expected state {expected:?}, but is {actual:?}")]
    UnexpectedState {
        job_id: JobId,
        expected: JobState,
        actual: JobState,
    },
    #[error("illegal job transition for {job_id}: {from:?} -> {to:?}")]
    InvalidTransition {
        job_id: JobId,
        from: JobState,
        to: JobState,
    },
    #[error("active job limit reached ({limit})")]
    ActiveLimitReached { limit: usize },
    #[error("candidate preparation limit reached ({limit})")]
    PreparationLimitReached { limit: usize },
    #[error("a live writer job is already active")]
    ActiveLiveWriter,
    #[error("wait for job state timed out")]
    WaitTimedOut,
    #[error("job recovery {0} was not found")]
    RecoveryNotFound(RecoveryId),
    #[error("restored job {job_id} uses unsupported schema version {schema_version}")]
    UnsupportedSchema { job_id: JobId, schema_version: u16 },
    #[error("restored job ID {0} appeared more than once")]
    DuplicateJob(JobId),
}

#[derive(Clone)]
pub struct WorkspaceJobRegistry {
    inner: Arc<Inner>,
}

struct Inner {
    state: Mutex<State>,
    changed_tx: watch::Sender<u64>,
}

struct State {
    binding: WorkspaceBinding,
    limits: JobRegistryLimits,
    sequence: u64,
    jobs: BTreeMap<JobId, WorkspaceJobSnapshot>,
}

impl WorkspaceJobRegistry {
    pub fn new(binding: WorkspaceBinding) -> Self {
        Self::with_limits(binding, JobRegistryLimits::default())
            .expect("default workspace job limits are valid")
    }

    pub fn with_limits(
        binding: WorkspaceBinding,
        limits: JobRegistryLimits,
    ) -> Result<Self, JobRegistryError> {
        if limits.max_preparations == 0 || limits.max_active_jobs == 0 {
            return Err(JobRegistryError::InvalidLimits);
        }
        let (changed_tx, _) = watch::channel(0);
        Ok(Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    binding,
                    limits,
                    sequence: 0,
                    jobs: BTreeMap::new(),
                }),
                changed_tx,
            }),
        })
    }

    /// Restore durable projections and conservatively settle records that
    /// were active when the previous process stopped.
    pub fn restore(
        binding: WorkspaceBinding,
        limits: JobRegistryLimits,
        jobs: impl IntoIterator<Item = WorkspaceJobSnapshot>,
    ) -> Result<(Self, JobRestartReport), JobRegistryError> {
        let registry = Self::with_limits(binding, limits)?;
        let mut report = JobRestartReport::default();
        {
            let mut state = lock(&registry.inner.state);
            let binding = state.binding.clone();
            for mut job in jobs {
                let job_id = job.permit.job_id.clone();
                if job.permit.schema_version > WORKSPACE_CONTRACT_SCHEMA_VERSION {
                    return Err(JobRegistryError::UnsupportedSchema {
                        job_id,
                        schema_version: job.permit.schema_version,
                    });
                }
                if state.jobs.contains_key(&job_id) {
                    return Err(JobRegistryError::DuplicateJob(job_id));
                }

                if !job.state.is_terminal() {
                    let current_binding = job.permit.controller_id == binding.controller_id
                        && job.permit.binding_id == binding.binding_id
                        && job.permit.epoch == binding.epoch;
                    let next = if !current_binding {
                        if is_write_job(&job.permit.spec) {
                            report.stale.push(job_id.clone());
                            JobState::Stale
                        } else {
                            report.orphaned.push(job_id.clone());
                            JobState::Orphaned
                        }
                    } else if is_write_job(&job.permit.spec) {
                        report.recovery_required.push(job_id.clone());
                        JobState::RecoveryRequired
                    } else {
                        report.orphaned.push(job_id.clone());
                        JobState::Orphaned
                    };
                    job.state = next;
                    if next == JobState::RecoveryRequired && job.recovery_id.is_none() {
                        job.recovery_id = Some(recovery::recovery_id_for(&job.permit));
                    }
                    job.revision = job.revision.saturating_add(1);
                    job.updated_at_ms = now_ms();
                    if next.is_terminal() {
                        job.finalized_at_ms = Some(job.updated_at_ms);
                    }
                    job.detail = Some(
                        match next {
                            JobState::RecoveryRequired => {
                                "writer was active across restart; recovery is required"
                            }
                            JobState::Stale => "job belongs to a stale workspace binding",
                            _ => "job was active when the harness restarted",
                        }
                        .to_owned(),
                    );
                }
                state.jobs.insert(job_id, job);
            }
            publish(&registry.inner, &mut state);
        }
        Ok((registry, report))
    }

    pub fn fence(&self) -> JobFence {
        JobFence::from_binding(&lock(&self.inner.state).binding)
    }

    pub fn register(&self, fence: &JobFence, spec: JobSpec) -> Result<JobPermit, JobRegistryError> {
        let mut state = lock(&self.inner.state);
        ensure_fence(&state, fence)?;
        if active_count(&state.jobs) >= state.limits.max_active_jobs {
            return Err(JobRegistryError::ActiveLimitReached {
                limit: state.limits.max_active_jobs,
            });
        }
        if matches!(spec.effect_scope, EffectScope::LiveWriter)
            && state.jobs.values().any(|job| {
                !job.state.is_terminal()
                    && matches!(job.permit.spec.effect_scope, EffectScope::LiveWriter)
            })
        {
            return Err(JobRegistryError::ActiveLiveWriter);
        }

        let now = now_ms();
        let permit = JobPermit {
            schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
            controller_id: state.binding.controller_id.clone(),
            job_id: JobId::new(uuid::Uuid::new_v4().to_string()),
            binding_id: state.binding.binding_id.clone(),
            epoch: state.binding.epoch,
            spec,
            issued_at_ms: now,
        };
        let snapshot = WorkspaceJobSnapshot {
            permit: permit.clone(),
            state: JobState::Queued,
            revision: 0,
            registered_at_ms: now,
            updated_at_ms: now,
            finalized_at_ms: None,
            terminal: None,
            recovery_id: None,
            detail: None,
            artifacts: Vec::new(),
            output: Vec::new(),
            output_truncated: false,
            output_bytes: 0,
            next_output_sequence: 0,
        };
        state.jobs.insert(permit.job_id.clone(), snapshot);
        publish(&self.inner, &mut state);
        Ok(permit)
    }

    pub fn transition(
        &self,
        fence: &JobFence,
        job_id: &JobId,
        expected: JobState,
        next: JobState,
        detail: Option<String>,
        artifacts: Vec<ArtifactRef>,
    ) -> Result<WorkspaceJobSnapshot, JobRegistryError> {
        let mut state = lock(&self.inner.state);
        ensure_fence(&state, fence)?;
        let current = state
            .jobs
            .get(job_id)
            .ok_or_else(|| JobRegistryError::NotFound(job_id.clone()))?;
        ensure_job_fence(current, fence)?;
        if current.state.is_terminal() {
            return Err(JobRegistryError::AlreadyTerminal {
                job_id: job_id.clone(),
                state: current.state,
            });
        }
        if current.state != expected {
            return Err(JobRegistryError::UnexpectedState {
                job_id: job_id.clone(),
                expected,
                actual: current.state,
            });
        }
        if !transition_allowed(current, next) {
            return Err(JobRegistryError::InvalidTransition {
                job_id: job_id.clone(),
                from: current.state,
                to: next,
            });
        }
        if next == JobState::Starting
            && is_preparation(&current.permit.spec)
            && preparation_count(&state.jobs) >= state.limits.max_preparations
        {
            return Err(JobRegistryError::PreparationLimitReached {
                limit: state.limits.max_preparations,
            });
        }

        let job = state.jobs.get_mut(job_id).expect("job was just found");
        job.state = next;
        if next == JobState::RecoveryRequired && job.recovery_id.is_none() {
            job.recovery_id = Some(recovery::recovery_id_for(&job.permit));
        }
        job.revision = job.revision.saturating_add(1);
        job.updated_at_ms = now_ms();
        if let Some(detail) = detail {
            job.detail = Some(detail);
        }
        job.artifacts.extend(artifacts);
        if next.is_terminal() {
            job.finalized_at_ms = Some(job.updated_at_ms);
        }
        let snapshot = job.clone();
        publish(&self.inner, &mut state);
        Ok(snapshot)
    }

    pub fn seal(&self, fence: &JobFence, job_id: &JobId, terminal: JobTerminal) -> JobSealOutcome {
        let mut state = lock(&self.inner.state);
        if ensure_fence(&state, fence).is_err() {
            return seal_rejected(job_id, None, "job callback has a stale binding fence");
        }
        let Some(current) = state.jobs.get(job_id) else {
            return JobSealOutcome {
                job_id: job_id.clone(),
                status: JobSealStatus::NotFound,
                state: None,
                recovery_id: None,
                detail: Some("job was not registered".to_owned()),
            };
        };
        if !fence.matches_permit(&current.permit) {
            return seal_rejected(
                job_id,
                Some(current.state),
                "job permit has a stale binding fence",
            );
        }
        if current.state.is_terminal() {
            return JobSealOutcome {
                job_id: job_id.clone(),
                status: JobSealStatus::AlreadySealed,
                state: Some(current.state),
                recovery_id: current.recovery_id.clone(),
                detail: current.detail.clone(),
            };
        }

        let next = completion_state(terminal.completion);
        if !transition_allowed(current, next) {
            return seal_rejected(
                job_id,
                Some(current.state),
                format!("illegal job transition {:?} -> {next:?}", current.state),
            );
        }
        let job = state.jobs.get_mut(job_id).expect("job was just found");
        job.state = next;
        if next == JobState::RecoveryRequired && job.recovery_id.is_none() {
            job.recovery_id = Some(recovery::recovery_id_for(&job.permit));
        }
        job.revision = job.revision.saturating_add(1);
        job.updated_at_ms = now_ms();
        job.detail.clone_from(&terminal.detail);
        job.artifacts.extend(terminal.artifacts.clone());
        if next.is_terminal() {
            job.finalized_at_ms = Some(job.updated_at_ms);
            job.terminal = Some(terminal.clone());
        }
        let recovery_id = job.recovery_id.clone();
        publish(&self.inner, &mut state);
        JobSealOutcome {
            job_id: job_id.clone(),
            status: JobSealStatus::Sealed,
            state: Some(next),
            recovery_id,
            detail: terminal.detail,
        }
    }

    pub fn request_cancel(
        &self,
        fence: &JobFence,
        job_id: &JobId,
    ) -> Result<WorkspaceJobSnapshot, JobRegistryError> {
        let current = self.status(fence, job_id)?;
        let next = match current.state {
            JobState::Queued | JobState::ReadyToMerge => JobState::Cancelled,
            JobState::Starting | JobState::Running => JobState::CancelRequested,
            state if state.is_terminal() => {
                return Err(JobRegistryError::AlreadyTerminal {
                    job_id: job_id.clone(),
                    state,
                });
            }
            state => {
                return Err(JobRegistryError::InvalidTransition {
                    job_id: job_id.clone(),
                    from: state,
                    to: JobState::CancelRequested,
                });
            }
        };
        self.transition(
            fence,
            job_id,
            current.state,
            next,
            Some("cancellation requested".to_owned()),
            Vec::new(),
        )
    }

    pub fn append_output(
        &self,
        fence: &JobFence,
        job_id: &JobId,
        stream: JobOutputStream,
        text: impl Into<String>,
    ) -> Result<WorkspaceJobSnapshot, JobRegistryError> {
        let mut state = lock(&self.inner.state);
        ensure_fence(&state, fence)?;
        let job = state
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| JobRegistryError::NotFound(job_id.clone()))?;
        ensure_job_fence(job, fence)?;
        if job.state.is_terminal() {
            return Err(JobRegistryError::AlreadyTerminal {
                job_id: job_id.clone(),
                state: job.state,
            });
        }
        let limit = job
            .permit
            .spec
            .limits
            .output_bytes
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(DEFAULT_OUTPUT_BYTES);
        let chunk = JobOutputChunk {
            sequence: job.next_output_sequence,
            stream,
            text: text.into(),
        };
        job.next_output_sequence = job.next_output_sequence.saturating_add(1);
        job.output_bytes = job.output_bytes.saturating_add(chunk.text.len());
        job.output.push(chunk);
        truncate_output(job, limit);
        job.revision = job.revision.saturating_add(1);
        job.updated_at_ms = now_ms();
        let snapshot = job.clone();
        publish(&self.inner, &mut state);
        Ok(snapshot)
    }

    pub fn status(
        &self,
        fence: &JobFence,
        job_id: &JobId,
    ) -> Result<WorkspaceJobSnapshot, JobRegistryError> {
        let state = lock(&self.inner.state);
        ensure_fence(&state, fence)?;
        let job = state
            .jobs
            .get(job_id)
            .ok_or_else(|| JobRegistryError::NotFound(job_id.clone()))?;
        ensure_job_fence(job, fence)?;
        Ok(job.clone())
    }

    /// Compatibility alias for `job output`: the output is part of the same
    /// immutable status projection, so callers cannot observe mismatched state.
    pub fn output(
        &self,
        fence: &JobFence,
        job_id: &JobId,
    ) -> Result<WorkspaceJobSnapshot, JobRegistryError> {
        self.status(fence, job_id)
    }

    pub fn snapshot(&self) -> WorkspaceJobsSnapshot {
        let state = lock(&self.inner.state);
        WorkspaceJobsSnapshot {
            controller_id: state.binding.controller_id.clone(),
            binding_id: state.binding.binding_id.clone(),
            epoch: state.binding.epoch,
            sequence: state.sequence,
            limits: state.limits,
            jobs: state.jobs.values().cloned().collect(),
        }
    }

    pub fn barrier(
        &self,
        fence: &JobFence,
        reason: BarrierKind,
        deadline: Instant,
    ) -> Result<BarrierReceipt, JobRegistryError> {
        let state = lock(&self.inner.state);
        ensure_fence(&state, fence)?;
        let pending_jobs: Vec<_> = state
            .jobs
            .iter()
            .filter(|(_, job)| !job.state.is_terminal())
            .map(|(id, _)| id.clone())
            .collect();
        let recovery_required = state
            .jobs
            .values()
            .any(|job| job.state == JobState::RecoveryRequired);
        let recovery_id = state.jobs.values().find_map(|job| {
            (job.state == JobState::RecoveryRequired)
                .then(|| job.recovery_id.clone())
                .flatten()
        });
        let status = if recovery_required {
            BarrierStatus::RecoveryRequired
        } else if pending_jobs.is_empty() {
            BarrierStatus::Passed
        } else if Instant::now() >= deadline {
            BarrierStatus::TimedOut
        } else {
            BarrierStatus::Blocked
        };
        Ok(BarrierReceipt {
            kind: reason,
            status,
            binding_id: state.binding.binding_id.clone(),
            epoch: state.binding.epoch,
            active_operation: None,
            pending_jobs,
            recovery_id,
            detail: recovery_required
                .then(|| "one or more writer jobs require workspace recovery".to_owned()),
        })
    }

    pub async fn wait(
        &self,
        fence: &JobFence,
        job_id: &JobId,
        condition: JobWaitCondition,
        deadline: Option<Instant>,
    ) -> Result<WorkspaceJobSnapshot, JobRegistryError> {
        let mut changed = self.inner.changed_tx.subscribe();
        loop {
            let snapshot = self.status(fence, job_id)?;
            let ready = match condition {
                JobWaitCondition::RevisionAfter(revision) => snapshot.revision > revision,
                JobWaitCondition::Actionable => snapshot.is_actionable(),
                JobWaitCondition::Terminal => snapshot.state.is_terminal(),
            };
            if ready {
                return Ok(snapshot);
            }

            if let Some(deadline) = deadline {
                if Instant::now() >= deadline {
                    return Err(JobRegistryError::WaitTimedOut);
                }
                tokio::select! {
                    changed_result = changed.changed() => {
                        if changed_result.is_err() {
                            return Err(JobRegistryError::WaitTimedOut);
                        }
                    }
                    () = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {
                        return Err(JobRegistryError::WaitTimedOut);
                    }
                }
            } else if changed.changed().await.is_err() {
                return Err(JobRegistryError::WaitTimedOut);
            }
        }
    }

    // Compatibility-friendly names matching the public command family.
    pub fn job_status(
        &self,
        fence: &JobFence,
        job_id: &JobId,
    ) -> Result<WorkspaceJobSnapshot, JobRegistryError> {
        self.status(fence, job_id)
    }

    pub fn job_output(
        &self,
        fence: &JobFence,
        job_id: &JobId,
    ) -> Result<WorkspaceJobSnapshot, JobRegistryError> {
        self.output(fence, job_id)
    }

    pub async fn job_wait(
        &self,
        fence: &JobFence,
        job_id: &JobId,
        deadline: Option<Instant>,
    ) -> Result<WorkspaceJobSnapshot, JobRegistryError> {
        self.wait(fence, job_id, JobWaitCondition::Actionable, deadline)
            .await
    }

    pub fn job_cancel(
        &self,
        fence: &JobFence,
        job_id: &JobId,
    ) -> Result<WorkspaceJobSnapshot, JobRegistryError> {
        self.request_cancel(fence, job_id)
    }
}

fn ensure_fence(state: &State, fence: &JobFence) -> Result<(), JobRegistryError> {
    if fence.matches_binding(&state.binding) {
        Ok(())
    } else {
        Err(JobRegistryError::StaleFence)
    }
}

fn ensure_job_fence(job: &WorkspaceJobSnapshot, fence: &JobFence) -> Result<(), JobRegistryError> {
    if fence.matches_permit(&job.permit) {
        Ok(())
    } else {
        Err(JobRegistryError::StaleFence)
    }
}

fn transition_allowed(job: &WorkspaceJobSnapshot, next: JobState) -> bool {
    if !job.state.can_transition_to(next) {
        return false;
    }
    if is_candidate(&job.permit.spec) {
        if job.state == JobState::Running && next == JobState::Succeeded {
            return false;
        }
        if next == JobState::Succeeded && job.state != JobState::Settling {
            return false;
        }
    } else if matches!(job.permit.spec.effect_scope, EffectScope::LiveWriter)
        && next == JobState::Succeeded
        && job.state != JobState::Settling
    {
        return false;
    }
    true
}

fn completion_state(completion: JobCompletion) -> JobState {
    match completion {
        JobCompletion::Succeeded => JobState::Succeeded,
        JobCompletion::ReadyToMerge => JobState::ReadyToMerge,
        JobCompletion::Merging => JobState::Merging,
        JobCompletion::Settling => JobState::Settling,
        JobCompletion::Failed => JobState::Failed,
        JobCompletion::Cancelled => JobState::Cancelled,
        JobCompletion::DurabilityPending => JobState::DurabilityPending,
        JobCompletion::RecoveryRequired => JobState::RecoveryRequired,
        JobCompletion::Stale => JobState::Stale,
    }
}

fn is_candidate(spec: &JobSpec) -> bool {
    spec.kind == JobKind::WriteCandidate || matches!(spec.effect_scope, EffectScope::CandidateOnly)
}

fn is_write_job(spec: &JobSpec) -> bool {
    is_candidate(spec) || matches!(spec.effect_scope, EffectScope::LiveWriter)
}

fn is_preparation(spec: &JobSpec) -> bool {
    is_candidate(spec)
}

fn active_count(jobs: &BTreeMap<JobId, WorkspaceJobSnapshot>) -> usize {
    jobs.values().filter(|job| !job.state.is_terminal()).count()
}

fn preparation_count(jobs: &BTreeMap<JobId, WorkspaceJobSnapshot>) -> usize {
    jobs.values()
        .filter(|job| {
            is_preparation(&job.permit.spec)
                && matches!(job.state, JobState::Starting | JobState::Running)
        })
        .count()
}

fn seal_rejected(
    job_id: &JobId,
    state: Option<JobState>,
    detail: impl Into<String>,
) -> JobSealOutcome {
    JobSealOutcome {
        job_id: job_id.clone(),
        status: JobSealStatus::Rejected,
        state,
        recovery_id: None,
        detail: Some(detail.into()),
    }
}

fn publish(inner: &Inner, state: &mut State) {
    state.sequence = state.sequence.saturating_add(1);
    inner.changed_tx.send_replace(state.sequence);
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
