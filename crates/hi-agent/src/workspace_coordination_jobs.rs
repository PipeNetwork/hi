//! Adapter from compatibility background handles to workspace-controller jobs.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use hi_tools::{
    BackgroundCandidateTransition, BackgroundJobEffect, BackgroundJobId, BackgroundJobKind,
    BackgroundJobLifecycle, BackgroundJobPublication, BackgroundJobRegistration,
    BackgroundJobTerminal,
};
use hi_workspace::{
    ArtifactRef, EffectScope, JobCompletion, JobId, JobKind, JobLimits, JobSealOutcome,
    JobSealStatus, JobSpec, JobState, JobTerminal, WorkspaceController,
};

use super::WorkspaceCoordination;

pub(crate) struct WorkspaceJobLifecycleBridge {
    coordination: WorkspaceCoordination,
    admission_generation: u64,
    jobs: tokio::sync::Mutex<HashMap<BackgroundJobId, Arc<TrackedJob>>>,
    registrations: tokio::sync::Mutex<()>,
}

struct TrackedJob {
    controller: Arc<dyn WorkspaceController>,
    job_id: JobId,
    effect: BackgroundJobEffect,
    verification_ms: Option<u64>,
    gate: tokio::sync::Mutex<()>,
    observed_exit: std::sync::Mutex<Option<BackgroundJobTerminal>>,
    artifacts: std::sync::Mutex<Vec<ArtifactRef>>,
}

impl TrackedJob {
    fn state(&self) -> Result<JobState, String> {
        self.controller
            .job_state(&self.job_id)
            .ok_or_else(|| format!("workspace job {} has no authoritative state", self.job_id))
    }
}

impl WorkspaceJobLifecycleBridge {
    pub(crate) fn new(coordination: WorkspaceCoordination) -> Self {
        let admission_generation = coordination.admission_generation();
        Self {
            coordination,
            admission_generation,
            jobs: tokio::sync::Mutex::new(HashMap::new()),
            registrations: tokio::sync::Mutex::new(()),
        }
    }

    fn supports_effect_now(&self, effect: BackgroundJobEffect) -> bool {
        if !self
            .coordination
            .admission_generation_is_current(self.admission_generation)
        {
            return false;
        }
        let controller = self.coordination.job_controller();
        let capabilities = controller.capabilities();
        let harness = self.coordination.harness_settings();
        let background_candidates_available = harness.features.candidate_jobs_v2
            && match controller.binding().authority {
                hi_workspace::WorkspaceAuthority::Local => true,
                hi_workspace::WorkspaceAuthority::PipeFs {
                    writer_protocol, ..
                } => writer_protocol >= 2,
            };
        match effect {
            BackgroundJobEffect::ReadOnly => true,
            BackgroundJobEffect::CandidateOnly => {
                capabilities.candidate_apply && background_candidates_available
            }
            BackgroundJobEffect::LiveWriter => capabilities.background_writers,
        }
    }
}

impl WorkspaceCoordination {
    pub(crate) fn bind_background_registries(
        &self,
        processes: &hi_tools::BackgroundRegistry,
        tasks: &hi_tools::BackgroundTaskRegistry,
    ) {
        let lifecycle: Arc<dyn BackgroundJobLifecycle> =
            Arc::new(WorkspaceJobLifecycleBridge::new(self.clone()));
        processes.set_job_lifecycle(lifecycle.clone());
        tasks.set_job_lifecycle(lifecycle);
    }
}

#[async_trait]
impl BackgroundJobLifecycle for WorkspaceJobLifecycleBridge {
    fn supports_effect(&self, effect: BackgroundJobEffect) -> bool {
        self.supports_effect_now(effect)
    }

    async fn register(&self, registration: BackgroundJobRegistration) -> Result<(), String> {
        // Always take admission before the bridge job map. Rebind holds the
        // exclusive side while draining lifecycle callbacks, which may need
        // the map but never need to admit another job.
        let _admission = self.coordination.acquire_admission().await;
        if !self
            .coordination
            .admission_generation_is_current(self.admission_generation)
        {
            return Err("background registry belongs to a stale workspace binding".into());
        }
        let _registration = self.registrations.lock().await;
        let jobs = self.jobs.lock().await;
        if jobs.contains_key(&registration.id) {
            return Err(format!(
                "background job {} was registered more than once",
                registration.id.handle
            ));
        }
        let controller = self.coordination.job_controller();
        let harness = self.coordination.harness_settings();
        if !self.supports_effect_now(registration.effect) {
            return Err("background job effect is unavailable for this workspace binding".into());
        }
        let active = jobs
            .values()
            .filter(|job| job.state().is_ok_and(|state| !state.is_terminal()))
            .count();
        if active >= harness.jobs.max_active {
            return Err(format!(
                "managed job concurrency reached ({})",
                harness.jobs.max_active
            ));
        }
        drop(jobs);
        let parent_operation = self.coordination.active_parent_operation();
        let permit = controller
            .register_job(JobSpec {
                kind: job_kind(registration.kind),
                effect_scope: effect_scope(registration.effect),
                name: registration.name,
                limits: managed_limits(&harness.jobs, registration.kind),
                parent_operation,
            })
            .await
            .map_err(|error| error.to_string())?;
        let verification_ms = permit.spec.limits.verification_ms;
        self.jobs.lock().await.insert(
            registration.id,
            Arc::new(TrackedJob {
                controller,
                job_id: permit.job_id,
                effect: registration.effect,
                verification_ms,
                gate: tokio::sync::Mutex::new(()),
                observed_exit: std::sync::Mutex::new(None),
                artifacts: std::sync::Mutex::new(Vec::new()),
            }),
        );
        Ok(())
    }

    async fn observe_terminal_with_artifacts(
        &self,
        id: &BackgroundJobId,
        terminal: BackgroundJobTerminal,
        detail: Option<String>,
        artifacts: Vec<ArtifactRef>,
    ) -> Result<BackgroundJobPublication, String> {
        if !artifacts.is_empty() {
            let job = self
                .jobs
                .lock()
                .await
                .get(id)
                .cloned()
                .ok_or_else(|| format!("unregistered background job {}", id.handle))?;
            *job.artifacts.lock().unwrap_or_else(|p| p.into_inner()) = artifacts;
        }
        self.observe_terminal(id, terminal, detail).await
    }

    async fn observe_terminal(
        &self,
        id: &BackgroundJobId,
        terminal: BackgroundJobTerminal,
        detail: Option<String>,
    ) -> Result<BackgroundJobPublication, String> {
        let job = self
            .jobs
            .lock()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| format!("unregistered background job {}", id.handle))?;
        let _job = job.gate.lock().await;
        match job.state()? {
            state if state.is_terminal() => return Ok(BackgroundJobPublication::Published),
            JobState::RecoveryRequired => {
                return Err(format!("workspace job {} requires recovery", job.job_id));
            }
            JobState::ReadyToMerge if terminal == BackgroundJobTerminal::Cancelled => {
                seal(&job, JobCompletion::Cancelled, detail).await?;
                return Ok(BackgroundJobPublication::Published);
            }
            JobState::DurabilityPending
            | JobState::ReadyToMerge
            | JobState::Merging
            | JobState::Settling => {
                return Ok(BackgroundJobPublication::DurabilityPending);
            }
            _ => {}
        }
        if terminal == BackgroundJobTerminal::Orphaned {
            seal(&job, JobCompletion::Orphaned, detail).await?;
            return Ok(BackgroundJobPublication::Published);
        }
        if job.effect == BackgroundJobEffect::LiveWriter
            && terminal != BackgroundJobTerminal::FailedBeforeStart
        {
            // Execution evidence is retained independently from the controller's
            // lifecycle. A failed/dropped acknowledgment can be retried exactly.
            *job.observed_exit.lock().unwrap_or_else(|p| p.into_inner()) = Some(terminal);
            seal(&job, JobCompletion::DurabilityPending, detail).await?;
            return Ok(BackgroundJobPublication::DurabilityPending);
        }
        if job.effect == BackgroundJobEffect::CandidateOnly
            && terminal == BackgroundJobTerminal::Succeeded
        {
            seal(&job, JobCompletion::ReadyToMerge, detail).await?;
            return Ok(BackgroundJobPublication::DurabilityPending);
        }
        seal(&job, completion(terminal), detail).await?;
        Ok(BackgroundJobPublication::Published)
    }

    async fn pending(&self, source_id: &str) -> Vec<BackgroundJobId> {
        self.jobs
            .lock()
            .await
            .iter()
            .filter(|(id, job)| {
                id.source_id == source_id
                    && matches!(
                        job.state(),
                        Ok(JobState::DurabilityPending | JobState::Settling)
                    )
                    && job
                        .observed_exit
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .is_some()
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    async fn settle_after_workspace(&self, pending: &[BackgroundJobId]) -> Result<(), String> {
        let jobs = {
            let tracked = self.jobs.lock().await;
            pending
                .iter()
                .filter_map(|id| tracked.get(id).cloned())
                .collect::<Vec<_>>()
        };
        let results = futures_util::future::join_all(jobs.into_iter().map(|job| async move {
            let _job = job.gate.lock().await;
            let terminal = *job.observed_exit.lock().unwrap_or_else(|p| p.into_inner());
            let Some(terminal) = terminal else {
                return Ok::<(), String>(());
            };
            let state = job.state()?;
            if state.is_terminal() {
                return Ok::<(), String>(());
            }
            if state == JobState::DurabilityPending {
                seal(
                    &job,
                    JobCompletion::Settling,
                    Some("workspace durability receipt acknowledged".into()),
                )
                .await?;
            } else if state != JobState::Settling {
                return Err(format!(
                    "workspace job {} cannot publish from {state:?}",
                    job.job_id
                ));
            }
            seal(
                &job,
                completion(terminal),
                Some("workspace durability receipt acknowledged".into()),
            )
            .await?;
            *job.observed_exit.lock().unwrap_or_else(|p| p.into_inner()) = None;
            Ok(())
        }))
        .await;
        let errors = results
            .into_iter()
            .filter_map(Result::err)
            .collect::<Vec<_>>();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }

    async fn workspace_job_id(&self, id: &BackgroundJobId) -> Option<String> {
        self.jobs
            .lock()
            .await
            .get(id)
            .map(|job| job.job_id.to_string())
    }

    async fn workspace_job_verification_ms(&self, id: &BackgroundJobId) -> Option<u64> {
        self.jobs
            .lock()
            .await
            .get(id)
            .and_then(|job| job.verification_ms)
    }

    async fn transition_candidate(
        &self,
        id: &BackgroundJobId,
        transition: BackgroundCandidateTransition,
        detail: Option<String>,
    ) -> Result<(), String> {
        let job = self
            .jobs
            .lock()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| format!("unregistered background candidate {}", id.handle))?;
        let _job = job.gate.lock().await;
        let state = job.state()?;
        let (expected, completion) = match transition {
            BackgroundCandidateTransition::Merging => {
                (JobState::ReadyToMerge, JobCompletion::Merging)
            }
            BackgroundCandidateTransition::Settling => (JobState::Merging, JobCompletion::Settling),
            BackgroundCandidateTransition::Succeeded => {
                (JobState::Settling, JobCompletion::Succeeded)
            }
            BackgroundCandidateTransition::Failed => (state, JobCompletion::Failed),
            BackgroundCandidateTransition::RecoveryRequired => {
                (state, JobCompletion::RecoveryRequired)
            }
            BackgroundCandidateTransition::Stale => (state, JobCompletion::Stale),
        };
        if state.is_terminal() || state == JobState::RecoveryRequired {
            return Err(format!(
                "candidate {} is already terminal; refusing transition {transition:?}",
                id.handle
            ));
        }
        if state != expected {
            return Err(format!(
                "candidate {} transition {transition:?} expected {expected:?}, observed {state:?}",
                id.handle
            ));
        }
        seal(&job, completion, detail).await
    }
}

async fn seal(
    job: &TrackedJob,
    completion: JobCompletion,
    detail: Option<String>,
) -> Result<(), String> {
    let expected_state = completion_state(completion);
    let artifacts = job
        .artifacts
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let outcome = job
        .controller
        .seal_job(
            job.job_id.clone(),
            JobTerminal {
                completion,
                detail,
                artifacts,
            },
        )
        .await;
    acknowledge_seal(&job.job_id, expected_state, &outcome)
}

fn acknowledge_seal(
    job_id: &JobId,
    expected_state: JobState,
    outcome: &JobSealOutcome,
) -> Result<(), String> {
    match (outcome.status, outcome.state) {
        (JobSealStatus::Sealed | JobSealStatus::AlreadySealed, Some(state))
            if state == expected_state =>
        {
            Ok(())
        }
        (JobSealStatus::Sealed | JobSealStatus::AlreadySealed, observed) => Err(format!(
            "workspace job {job_id} settlement returned {:?} in state {observed:?}, but the requested terminal state was {expected_state:?}",
            outcome.status
        )),
        (status, _) => Err(format!(
            "workspace job {} settlement was rejected ({status:?}): {}",
            job_id,
            outcome.detail.as_deref().unwrap_or("no detail")
        )),
    }
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
        JobCompletion::Orphaned => JobState::Orphaned,
        JobCompletion::Stale => JobState::Stale,
    }
}

fn job_kind(kind: BackgroundJobKind) -> JobKind {
    match kind {
        BackgroundJobKind::Process => JobKind::Process,
        BackgroundJobKind::ReadAgent => JobKind::ReadAgent,
        BackgroundJobKind::WriteCandidate => JobKind::WriteCandidate,
    }
}

fn managed_limits(
    settings: &hi_workspace::HarnessJobSettings,
    kind: BackgroundJobKind,
) -> JobLimits {
    JobLimits {
        queue_ms: Some(duration_millis(settings.queue_timeout)),
        execution_ms: Some(duration_millis(settings.candidate_timeout)),
        verification_ms: (kind == BackgroundJobKind::WriteCandidate)
            .then_some(duration_millis(settings.verifier_timeout)),
        output_bytes: None,
    }
}

fn duration_millis(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn effect_scope(effect: BackgroundJobEffect) -> EffectScope {
    match effect {
        BackgroundJobEffect::ReadOnly => EffectScope::ReadOnly,
        BackgroundJobEffect::CandidateOnly => EffectScope::CandidateOnly,
        BackgroundJobEffect::LiveWriter => EffectScope::LiveWriter,
    }
}

fn completion(terminal: BackgroundJobTerminal) -> JobCompletion {
    match terminal {
        BackgroundJobTerminal::Succeeded => JobCompletion::Succeeded,
        BackgroundJobTerminal::ReadyToMerge => JobCompletion::ReadyToMerge,
        BackgroundJobTerminal::Failed | BackgroundJobTerminal::FailedBeforeStart => {
            JobCompletion::Failed
        }
        BackgroundJobTerminal::Cancelled => JobCompletion::Cancelled,
        BackgroundJobTerminal::Orphaned => JobCompletion::Orphaned,
    }
}

#[cfg(test)]
#[path = "workspace_coordination_jobs_keep_background_tests.rs"]
mod keep_background_tests;

#[cfg(test)]
#[path = "workspace_coordination_jobs_tests.rs"]
mod tests;
