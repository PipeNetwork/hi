use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde::{Serialize, Serializer};
use thiserror::Error;
use tokio::sync::watch;

use crate::{
    AdmissionDenied, BarrierKind, BarrierReceipt, ExecutionReport, JobId, JobPermit,
    JobSealOutcome, JobSpec, JobTerminal, MutationIntent, MutationPermitRecord, RecoveryId,
    RecoveryOutcome, SettlementOutcome, WorkspaceBinding, WorkspaceCapabilities, WorkspaceStatus,
};

/// Synchronous recovery hook invoked when an admitted mutation is abandoned.
///
/// Implementations must not perform I/O or await. Durable work should be
/// scheduled separately; this callback's job is to close admission and publish
/// recovery-required state in a bounded critical section.
pub trait PermitAbandonment: Send + Sync + 'static {
    fn mutation_abandoned(&self, permit: &MutationPermitRecord);
}

#[derive(Clone)]
pub struct PermitIssuer {
    controller_id: crate::ControllerId,
    abandonment: Arc<dyn PermitAbandonment>,
}

impl fmt::Debug for PermitIssuer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PermitIssuer")
            .field("controller_id", &self.controller_id)
            .finish_non_exhaustive()
    }
}

impl PermitIssuer {
    pub fn new(abandonment: Arc<dyn PermitAbandonment>) -> Self {
        Self {
            controller_id: crate::ControllerId::new(uuid::Uuid::new_v4().to_string()),
            abandonment,
        }
    }

    pub fn controller_id(&self) -> &crate::ControllerId {
        &self.controller_id
    }

    pub fn issue_mutation(&self, mut record: MutationPermitRecord) -> MutationPermit {
        record.controller_id = self.controller_id.clone();
        MutationPermit {
            record,
            abandonment: Some(self.abandonment.clone()),
        }
    }

    /// Validate that a live permit belongs to this issuer and disarm its
    /// synchronous abandonment fence for settlement.
    pub fn claim_mutation(
        &self,
        permit: &mut MutationPermit,
    ) -> Result<MutationPermitRecord, PermitClaimError> {
        if permit.record.controller_id != self.controller_id {
            return Err(PermitClaimError::WrongController);
        }
        if permit.abandonment.is_none() {
            return Err(PermitClaimError::AlreadyClaimed);
        }
        permit.abandonment = None;
        Ok(permit.record.clone())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum PermitClaimError {
    #[error("mutation permit belongs to a different controller")]
    WrongController,
    #[error("mutation permit has already been claimed")]
    AlreadyClaimed,
}

/// A live, non-cloneable mutation capability.
///
/// Serialization intentionally emits only the durable record. There is no
/// `Deserialize` implementation because persisted records must never recreate
/// an executable capability or bypass the issuer's abandonment fence.
pub struct MutationPermit {
    record: MutationPermitRecord,
    abandonment: Option<Arc<dyn PermitAbandonment>>,
}

impl MutationPermit {
    pub fn record(&self) -> &MutationPermitRecord {
        &self.record
    }

    pub fn snapshot(&self) -> MutationPermitRecord {
        self.record.clone()
    }
}

impl fmt::Debug for MutationPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MutationPermit")
            .field("record", &self.record)
            .field("armed", &self.abandonment.is_some())
            .finish()
    }
}

impl Serialize for MutationPermit {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.record.serialize(serializer)
    }
}

impl Drop for MutationPermit {
    fn drop(&mut self) {
        if let Some(abandonment) = self.abandonment.take() {
            abandonment.mutation_abandoned(&self.record);
        }
    }
}

#[async_trait]
pub trait WorkspaceController: Send + Sync {
    fn binding(&self) -> WorkspaceBinding;

    fn capabilities(&self) -> WorkspaceCapabilities;

    fn status(&self) -> WorkspaceStatus;

    fn subscribe(&self) -> watch::Receiver<WorkspaceStatus>;

    async fn begin(&self, intent: MutationIntent) -> Result<MutationPermit, AdmissionDenied>;

    async fn settle(&self, permit: MutationPermit, execution: ExecutionReport)
    -> SettlementOutcome;

    async fn register_job(&self, spec: JobSpec) -> Result<JobPermit, AdmissionDenied>;

    async fn seal_job(&self, job: JobId, terminal: JobTerminal) -> JobSealOutcome;

    async fn barrier(&self, reason: BarrierKind, deadline: Instant) -> BarrierReceipt;

    async fn reconcile(&self, recovery: RecoveryId) -> RecoveryOutcome;
}
