use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use hi_workspace::{
    AdmissionDenied, BarrierKind, BarrierReceipt, EffectScope, ExecutionReport,
    InMemoryWorkspaceController, JobCompletion, JobId, JobKind, JobLimits, JobPermit,
    JobSealOutcome, JobSpec, JobTerminal, MutationIntent, MutationPermit, RecoveryId,
    RecoveryOutcome, SettlementOutcome, WorkspaceBinding, WorkspaceCapabilities,
    WorkspaceController, WorkspaceState, WorkspaceStatus,
};
use tokio::sync::{Notify, watch};

use super::WorkspaceCoordination;

fn subject() -> (tempfile::TempDir, WorkspaceCoordination) {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let subject = WorkspaceCoordination::new_local(root.path(), &state);
    (root, subject)
}

#[tokio::test]
async fn hidden_controller_job_blocks_rebind_barrier_until_terminal() {
    let (_root, subject) = subject();
    let controller: Arc<dyn WorkspaceController> = subject.job_controller();
    let job = controller
        .register_job(JobSpec {
            kind: JobKind::ReadAgent,
            effect_scope: EffectScope::ReadOnly,
            name: "hidden reader".into(),
            limits: JobLimits::default(),
            parent_operation: None,
        })
        .await
        .unwrap();

    let error = subject
        .require_barrier_before(BarrierKind::Rebind, Instant::now())
        .await
        .unwrap_err();
    assert!(error.to_string().contains(job.job_id.as_str()));

    controller
        .seal_job(
            job.job_id,
            JobTerminal {
                completion: JobCompletion::Cancelled,
                detail: None,
                artifacts: Vec::new(),
            },
        )
        .await;
    subject
        .require_barrier_before(BarrierKind::Rebind, Instant::now() + Duration::from_secs(1))
        .await
        .unwrap();
}

#[tokio::test]
async fn recovery_fence_is_reported_by_exit_barrier() {
    let (_root, subject) = subject();
    let permit = subject
        .job_controller()
        .begin(MutationIntent::workspace("barrier test"))
        .await
        .unwrap();
    drop(permit);
    assert_eq!(subject.status().state, WorkspaceState::RecoveryRequired);
    let recovery = subject.status().recovery_id.unwrap();

    let error = subject
        .require_barrier_before(BarrierKind::Exit, Instant::now())
        .await
        .unwrap_err();
    assert!(error.to_string().contains(recovery.as_str()));
}

struct BlockingBarrierController {
    inner: Arc<dyn WorkspaceController>,
    entered: Notify,
    release: Notify,
}

impl BlockingBarrierController {
    fn new(inner: Arc<dyn WorkspaceController>) -> Self {
        Self {
            inner,
            entered: Notify::new(),
            release: Notify::new(),
        }
    }
}

#[async_trait]
impl WorkspaceController for BlockingBarrierController {
    fn binding(&self) -> WorkspaceBinding {
        self.inner.binding()
    }

    fn capabilities(&self) -> WorkspaceCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> WorkspaceStatus {
        self.inner.status()
    }

    fn subscribe(&self) -> watch::Receiver<WorkspaceStatus> {
        self.inner.subscribe()
    }

    async fn begin(&self, intent: MutationIntent) -> Result<MutationPermit, AdmissionDenied> {
        self.inner.begin(intent).await
    }

    async fn settle(
        &self,
        permit: MutationPermit,
        execution: ExecutionReport,
    ) -> SettlementOutcome {
        self.inner.settle(permit, execution).await
    }

    async fn register_job(&self, spec: JobSpec) -> Result<JobPermit, AdmissionDenied> {
        self.inner.register_job(spec).await
    }

    async fn seal_job(&self, job: JobId, terminal: JobTerminal) -> JobSealOutcome {
        self.inner.seal_job(job, terminal).await
    }

    async fn barrier(&self, reason: BarrierKind, deadline: Instant) -> BarrierReceipt {
        self.entered.notify_one();
        self.release.notified().await;
        self.inner.barrier(reason, deadline).await
    }

    async fn reconcile(&self, recovery: RecoveryId) -> RecoveryOutcome {
        self.inner.reconcile(recovery).await
    }
}

#[tokio::test]
async fn barrier_rejects_a_controller_swap_before_accepting_passed() {
    let (_root, subject) = subject();
    let original: Arc<dyn WorkspaceController> = subject.job_controller();
    let blocking = Arc::new(BlockingBarrierController::new(original));
    subject.install_controller(blocking.clone()).unwrap();

    let waiting = subject.clone();
    let barrier = tokio::spawn(async move {
        waiting
            .require_barrier_before(BarrierKind::Rebind, Instant::now() + Duration::from_secs(5))
            .await
    });
    blocking.entered.notified().await;

    let replacement: Arc<dyn WorkspaceController> = Arc::new(
        InMemoryWorkspaceController::new_local("replacement", "/replacement", "/state"),
    );
    subject.install_controller(replacement).unwrap();
    blocking.release.notify_waiters();

    let error = barrier.await.unwrap().unwrap_err();
    assert!(error.to_string().contains("controller changed"));
}
