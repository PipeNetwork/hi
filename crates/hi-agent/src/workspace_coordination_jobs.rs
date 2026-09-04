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
    jobs: tokio::sync::Mutex<HashMap<BackgroundJobId, TrackedJob>>,
}

struct TrackedJob {
    controller: Arc<dyn WorkspaceController>,
    job_id: JobId,
    effect: BackgroundJobEffect,
    verification_ms: Option<u64>,
    state: BridgeState,
    artifacts: Vec<ArtifactRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BridgeState {
    Running,
    ReadyToMerge,
    Merging,
    Settling,
    DurabilityPending(BackgroundJobTerminal),
    Published,
}

impl WorkspaceJobLifecycleBridge {
    pub(crate) fn new(coordination: WorkspaceCoordination) -> Self {
        let admission_generation = coordination.admission_generation();
        Self {
            coordination,
            admission_generation,
            jobs: tokio::sync::Mutex::new(HashMap::new()),
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
        let mut jobs = self.jobs.lock().await;
        if jobs.contains_key(&registration.id) {
            return Err(format!(
                "background job {} was registered more than once",
                registration.id.handle
            ));
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
        let effect_available = match registration.effect {
            BackgroundJobEffect::ReadOnly => true,
            BackgroundJobEffect::CandidateOnly => {
                capabilities.candidate_apply && background_candidates_available
            }
            BackgroundJobEffect::LiveWriter => capabilities.background_writers,
        };
        if !effect_available {
            return Err("background job effect is unavailable for this workspace binding".into());
        }
        let active = jobs
            .values()
            .filter(|job| job.state != BridgeState::Published)
            .count();
        if active >= harness.jobs.max_active {
            return Err(format!(
                "managed job concurrency reached ({})",
                harness.jobs.max_active
            ));
        }
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
        jobs.insert(
            registration.id,
            TrackedJob {
                controller,
                job_id: permit.job_id,
                effect: registration.effect,
                verification_ms,
                state: BridgeState::Running,
                artifacts: Vec::new(),
            },
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
            let mut jobs = self.jobs.lock().await;
            let job = jobs
                .get_mut(id)
                .ok_or_else(|| format!("unregistered background job {}", id.handle))?;
            job.artifacts = artifacts;
        }
        self.observe_terminal(id, terminal, detail).await
    }

    async fn observe_terminal(
        &self,
        id: &BackgroundJobId,
        terminal: BackgroundJobTerminal,
        detail: Option<String>,
    ) -> Result<BackgroundJobPublication, String> {
        let mut jobs = self.jobs.lock().await;
        let job = jobs
            .get_mut(id)
            .ok_or_else(|| format!("unregistered background job {}", id.handle))?;
        match job.state {
            BridgeState::Published => return Ok(BackgroundJobPublication::Published),
            BridgeState::DurabilityPending(_) => {
                return Ok(BackgroundJobPublication::DurabilityPending);
            }
            BridgeState::ReadyToMerge => {
                if terminal == BackgroundJobTerminal::Cancelled {
                    seal(job, JobCompletion::Cancelled, detail).await?;
                    job.state = BridgeState::Published;
                    return Ok(BackgroundJobPublication::Published);
                }
                return Ok(BackgroundJobPublication::DurabilityPending);
            }
            BridgeState::Merging | BridgeState::Settling => {
                return Ok(BackgroundJobPublication::DurabilityPending);
            }
            BridgeState::Running => {}
        }

        if job.effect == BackgroundJobEffect::LiveWriter
            && terminal != BackgroundJobTerminal::FailedBeforeStart
        {
            seal(job, JobCompletion::DurabilityPending, detail).await?;
            job.state = BridgeState::DurabilityPending(terminal);
            return Ok(BackgroundJobPublication::DurabilityPending);
        }

        if job.effect == BackgroundJobEffect::CandidateOnly
            && terminal == BackgroundJobTerminal::Succeeded
        {
            seal(job, JobCompletion::ReadyToMerge, detail).await?;
            job.state = BridgeState::ReadyToMerge;
            return Ok(BackgroundJobPublication::DurabilityPending);
        }
        let completion = completion(terminal);
        seal(job, completion, detail).await?;
        job.state = BridgeState::Published;
        Ok(BackgroundJobPublication::Published)
    }

    async fn pending(&self, source_id: &str) -> Vec<BackgroundJobId> {
        self.jobs
            .lock()
            .await
            .iter()
            .filter(|(id, job)| {
                id.source_id == source_id && matches!(job.state, BridgeState::DurabilityPending(_))
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    async fn settle_after_workspace(&self, pending: &[BackgroundJobId]) -> Result<(), String> {
        let mut jobs = self.jobs.lock().await;
        for id in pending {
            let Some(job) = jobs.get_mut(id) else {
                continue;
            };
            let BridgeState::DurabilityPending(terminal) = job.state else {
                continue;
            };
            // A durability acknowledgement is the boundary between an
            // observed process exit and publication. Move through Settling so
            // the controller itself can reject any adapter that attempts to
            // publish a live-writer success directly from Running or Pending.
            seal(
                job,
                JobCompletion::Settling,
                Some(
                    "workspace durability receipt acknowledged; publishing terminal job state"
                        .into(),
                ),
            )
            .await?;
            seal(
                job,
                completion(terminal),
                Some("workspace durability receipt acknowledged".into()),
            )
            .await?;
            job.state = BridgeState::Published;
        }
        Ok(())
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
        let mut jobs = self.jobs.lock().await;
        let job = jobs
            .get_mut(id)
            .ok_or_else(|| format!("unregistered background candidate {}", id.handle))?;
        let (expected, completion, next) = match transition {
            BackgroundCandidateTransition::Merging => (
                BridgeState::ReadyToMerge,
                JobCompletion::Merging,
                BridgeState::Merging,
            ),
            BackgroundCandidateTransition::Settling => (
                BridgeState::Merging,
                JobCompletion::Settling,
                BridgeState::Settling,
            ),
            BackgroundCandidateTransition::Succeeded => (
                BridgeState::Settling,
                JobCompletion::Succeeded,
                BridgeState::Published,
            ),
            BackgroundCandidateTransition::Failed => {
                (job.state, JobCompletion::Failed, BridgeState::Published)
            }
            BackgroundCandidateTransition::RecoveryRequired => (
                job.state,
                JobCompletion::RecoveryRequired,
                BridgeState::Published,
            ),
            BackgroundCandidateTransition::Stale => {
                (job.state, JobCompletion::Stale, BridgeState::Published)
            }
        };
        if job.state == BridgeState::Published {
            return Err(format!(
                "candidate {} is already terminal; refusing transition {transition:?}",
                id.handle
            ));
        }
        if job.state != expected {
            return Err(format!(
                "candidate {} transition {transition:?} expected {expected:?}, observed {:?}",
                id.handle, job.state
            ));
        }
        seal(job, completion, detail).await?;
        job.state = next;
        Ok(())
    }
}

async fn seal(
    job: &TrackedJob,
    completion: JobCompletion,
    detail: Option<String>,
) -> Result<(), String> {
    let expected_state = completion_state(completion);
    let outcome = job
        .controller
        .seal_job(
            job.job_id.clone(),
            JobTerminal {
                completion,
                detail,
                artifacts: job.artifacts.clone(),
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_control::{ControlJobState, ControlStore};

    fn coordination_with_candidates(
        root: &std::path::Path,
        state: &std::path::Path,
    ) -> WorkspaceCoordination {
        let mut harness = hi_workspace::ResolvedHarnessSettings::default();
        harness.features.candidate_jobs_v2 = true;
        WorkspaceCoordination::new_local_with_settings(root, state, harness)
    }

    fn registration(effect: BackgroundJobEffect) -> BackgroundJobRegistration {
        BackgroundJobRegistration {
            id: BackgroundJobId {
                source_id: "process-registry".into(),
                handle: "server_1".into(),
            },
            kind: match effect {
                BackgroundJobEffect::ReadOnly => BackgroundJobKind::ReadAgent,
                BackgroundJobEffect::CandidateOnly => BackgroundJobKind::WriteCandidate,
                BackgroundJobEffect::LiveWriter => BackgroundJobKind::Process,
            },
            effect,
            name: "test background process".into(),
        }
    }

    #[test]
    fn already_sealed_success_does_not_acknowledge_recovery_required() {
        let job_id = JobId::new("job-1");
        let outcome = JobSealOutcome {
            job_id: job_id.clone(),
            status: JobSealStatus::AlreadySealed,
            state: Some(JobState::Succeeded),
            recovery_id: None,
            detail: Some("inner controller had already published success".into()),
        };

        let error = acknowledge_seal(&job_id, JobState::RecoveryRequired, &outcome).unwrap_err();
        assert!(error.contains("AlreadySealed"));
        assert!(error.contains("Succeeded"));
        assert!(error.contains("RecoveryRequired"));
    }

    #[tokio::test]
    async fn live_writer_success_waits_for_the_workspace_receipt_exactly_once() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("workspace");
        let state = directory.path().join("state");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        let coordination = WorkspaceCoordination::new_local(&root, &state);
        let store = ControlStore::open_for_state(&state).unwrap();
        let bridge = WorkspaceJobLifecycleBridge::new(coordination.clone());
        coordination.begin(None, None).await.unwrap();

        let registration = registration(BackgroundJobEffect::LiveWriter);
        bridge.register(registration.clone()).await.unwrap();
        let job_id = bridge
            .jobs
            .lock()
            .await
            .get(&registration.id)
            .unwrap()
            .job_id
            .clone();
        assert_eq!(
            store.get_job(job_id.as_str()).unwrap().unwrap().state,
            ControlJobState::Running
        );

        assert_eq!(
            bridge
                .observe_terminal(&registration.id, BackgroundJobTerminal::Succeeded, None)
                .await
                .unwrap(),
            BackgroundJobPublication::DurabilityPending
        );
        let pending_record = store.get_job(job_id.as_str()).unwrap().unwrap();
        assert_eq!(pending_record.state, ControlJobState::DurabilityPending);
        assert_eq!(
            bridge
                .observe_terminal(&registration.id, BackgroundJobTerminal::Succeeded, None)
                .await
                .unwrap(),
            BackgroundJobPublication::DurabilityPending
        );
        assert_eq!(
            store.get_job(job_id.as_str()).unwrap().unwrap().revision,
            pending_record.revision,
            "a repeated process callback must not write a second transition"
        );

        let pending = bridge.pending(&registration.id.source_id).await;
        coordination
            .checkpoint(
                None,
                hi_workspace::ExecutionReport::succeeded(Some("workspace-receipt".into())),
            )
            .await
            .unwrap();
        assert_eq!(
            store.get_job(job_id.as_str()).unwrap().unwrap().state,
            ControlJobState::DurabilityPending,
            "workspace settlement alone must not publish a job omitted from the frozen set"
        );
        bridge.settle_after_workspace(&pending).await.unwrap();
        let succeeded = store.get_job(job_id.as_str()).unwrap().unwrap();
        assert_eq!(succeeded.state, ControlJobState::Succeeded);

        bridge.settle_after_workspace(&pending).await.unwrap();
        assert_eq!(
            store.get_job(job_id.as_str()).unwrap().unwrap().revision,
            succeeded.revision,
            "repeated settlement must be idempotent"
        );
    }

    #[tokio::test]
    async fn mixed_batch_children_inherit_the_active_parent_operation() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("workspace");
        let state = directory.path().join("state");
        std::fs::create_dir_all(&root).unwrap();
        let coordination = coordination_with_candidates(&root, &state);
        let store = ControlStore::open_for_state(&state).unwrap();
        let bridge = WorkspaceJobLifecycleBridge::new(coordination.clone());
        coordination.begin(None, None).await.unwrap();
        let reader = registration(BackgroundJobEffect::ReadOnly);
        bridge.register(reader.clone()).await.unwrap();
        let job_id = bridge.jobs.lock().await[&reader.id].job_id.clone();

        assert_eq!(
            bridge
                .observe_terminal(&reader.id, BackgroundJobTerminal::Succeeded, None)
                .await
                .unwrap(),
            BackgroundJobPublication::Published
        );
        assert_eq!(
            store.get_job(job_id.as_str()).unwrap().unwrap().state,
            ControlJobState::Succeeded
        );

        let mut candidate = registration(BackgroundJobEffect::CandidateOnly);
        candidate.id.handle = "candidate_1".into();
        bridge.register(candidate.clone()).await.unwrap();
        let candidate_id = bridge.jobs.lock().await[&candidate.id].job_id.clone();
        assert_eq!(
            bridge
                .observe_terminal(&candidate.id, BackgroundJobTerminal::Succeeded, None)
                .await
                .unwrap(),
            BackgroundJobPublication::DurabilityPending
        );
        assert_eq!(
            store.get_job(candidate_id.as_str()).unwrap().unwrap().state,
            ControlJobState::ReadyToMerge
        );
        coordination
            .checkpoint(None, hi_workspace::ExecutionReport::succeeded(None))
            .await
            .unwrap();
        for (transition, expected) in [
            (
                BackgroundCandidateTransition::Merging,
                ControlJobState::Merging,
            ),
            (
                BackgroundCandidateTransition::Settling,
                ControlJobState::Settling,
            ),
            (
                BackgroundCandidateTransition::Succeeded,
                ControlJobState::Succeeded,
            ),
        ] {
            bridge
                .transition_candidate(&candidate.id, transition, None)
                .await
                .unwrap();
            assert_eq!(
                store.get_job(candidate_id.as_str()).unwrap().unwrap().state,
                expected
            );
        }
    }

    #[tokio::test]
    async fn cancelled_candidate_cannot_reenter_merging() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("workspace");
        let state = directory.path().join("state");
        std::fs::create_dir_all(&root).unwrap();
        let coordination = coordination_with_candidates(&root, &state);
        let bridge = WorkspaceJobLifecycleBridge::new(coordination);
        let candidate = registration(BackgroundJobEffect::CandidateOnly);
        bridge.register(candidate.clone()).await.unwrap();
        bridge
            .observe_terminal(&candidate.id, BackgroundJobTerminal::Succeeded, None)
            .await
            .unwrap();
        bridge
            .observe_terminal(&candidate.id, BackgroundJobTerminal::Cancelled, None)
            .await
            .unwrap();

        let error = bridge
            .transition_candidate(&candidate.id, BackgroundCandidateTransition::Merging, None)
            .await
            .unwrap_err();
        assert!(error.contains("already terminal"));
    }

    #[tokio::test]
    async fn resolved_limits_gate_admission_and_populate_job_deadlines() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("workspace");
        let state = directory.path().join("state");
        std::fs::create_dir_all(&root).unwrap();
        let mut harness = hi_workspace::ResolvedHarnessSettings::default();
        harness.jobs.max_active = 1;
        harness.jobs.queue_timeout = std::time::Duration::from_millis(11);
        harness.jobs.candidate_timeout = std::time::Duration::from_millis(22);
        harness.jobs.verifier_timeout = std::time::Duration::from_millis(33);
        harness.features.candidate_jobs_v2 = true;
        let coordination = WorkspaceCoordination::new_local_with_settings(&root, &state, harness);
        let bridge = WorkspaceJobLifecycleBridge::new(coordination);

        let first = registration(BackgroundJobEffect::ReadOnly);
        bridge.register(first.clone()).await.unwrap();
        let mut second = registration(BackgroundJobEffect::ReadOnly);
        second.id.handle = "reader_2".into();
        let error = bridge.register(second).await.unwrap_err();
        assert!(error.contains("concurrency reached (1)"));

        let settings = bridge.coordination.harness_settings();
        assert_eq!(
            managed_limits(&settings.jobs, BackgroundJobKind::WriteCandidate),
            JobLimits {
                queue_ms: Some(11),
                execution_ms: Some(22),
                verification_ms: Some(33),
                output_bytes: None,
            }
        );
    }

    #[tokio::test]
    async fn pipefs_protocol_one_rejects_background_candidates_in_the_lifecycle_bridge() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("workspace");
        let state = directory.path().join("state");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        let coordination = coordination_with_candidates(&root, &state);
        coordination
            .install_pipefs("protocol-one", 1, false, &root, &state)
            .unwrap();
        let bridge = WorkspaceJobLifecycleBridge::new(coordination);

        let error = bridge
            .register(registration(BackgroundJobEffect::CandidateOnly))
            .await
            .unwrap_err();
        assert!(error.contains("unavailable"));
    }

    #[tokio::test]
    async fn local_candidates_are_also_closed_until_the_rollout_gate_is_enabled() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("workspace");
        let state = directory.path().join("state");
        std::fs::create_dir_all(&root).unwrap();
        let coordination = WorkspaceCoordination::new_local(&root, &state);
        let bridge = WorkspaceJobLifecycleBridge::new(coordination);

        let error = bridge
            .register(registration(BackgroundJobEffect::CandidateOnly))
            .await
            .unwrap_err();
        assert!(error.contains("unavailable"));
    }

    #[tokio::test]
    async fn resolved_limits_reach_the_local_controller_for_direct_jobs() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("workspace");
        let state = directory.path().join("state");
        std::fs::create_dir_all(&root).unwrap();
        let mut harness = hi_workspace::ResolvedHarnessSettings::default();
        harness.jobs.max_preparations = 1;
        harness.jobs.max_active = 2;
        let coordination = WorkspaceCoordination::new_local_with_settings(&root, &state, harness);
        let controller = coordination.job_controller();
        let candidate = |name: &str| JobSpec {
            kind: JobKind::WriteCandidate,
            effect_scope: EffectScope::CandidateOnly,
            name: name.into(),
            limits: JobLimits::default(),
            parent_operation: None,
        };

        controller.register_job(candidate("first")).await.unwrap();
        let preparation_error = controller
            .register_job(candidate("second"))
            .await
            .unwrap_err();
        assert!(
            preparation_error
                .detail
                .contains("candidate preparation limit reached (1)")
        );
        controller
            .register_job(JobSpec {
                kind: JobKind::ReadAgent,
                effect_scope: EffectScope::ReadOnly,
                name: "reader".into(),
                limits: JobLimits::default(),
                parent_operation: None,
            })
            .await
            .unwrap();
        let active_error = controller
            .register_job(JobSpec {
                kind: JobKind::ReadAgent,
                effect_scope: EffectScope::ReadOnly,
                name: "overflow".into(),
                limits: JobLimits::default(),
                parent_operation: None,
            })
            .await
            .unwrap_err();
        assert!(active_error.detail.contains("active job limit reached (2)"));
    }
}
