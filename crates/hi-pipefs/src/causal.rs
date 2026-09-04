use hi_workspace::{ExecutionReport, MutationPermitRecord, ReplayClass};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{ArtifactDescriptor, PipeFsError, PipeFsLease};

pub const CAUSAL_COMMIT_CAPABILITY: &str = "causal_commit_v1";
pub const CAUSAL_WRITER_PROTOCOL: u16 = 2;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalOperationReceipt {
    pub operation_id: String,
    pub idempotency_key: String,
    #[serde(default)]
    pub binding_id: String,
    #[serde(default)]
    pub binding_epoch: u64,
    pub replay_class: ReplayClass,
    pub execution: ExecutionReport,
}

impl CausalOperationReceipt {
    pub(crate) fn has_valid_recovery_fence(&self) -> bool {
        !self.operation_id.trim().is_empty()
            && !self.idempotency_key.trim().is_empty()
            && !self.binding_id.trim().is_empty()
            && self.binding_epoch > 0
            && match &self.replay_class {
                ReplayClass::IdempotentExternal { key } => key.as_str() == self.idempotency_key,
                ReplayClass::PureWorkspace | ReplayClass::NonReplayableExternal => true,
            }
    }

    pub(crate) fn validate_binding(
        &self,
        expected_binding_id: &str,
        expected_epoch: u64,
    ) -> Result<(), PipeFsError> {
        if self.binding_id.is_empty()
            || self.binding_epoch == 0
            || self.binding_id != expected_binding_id
            || self.binding_epoch != expected_epoch
        {
            return Err(PipeFsError::Protocol(
                "causal operation is fenced by a different workspace binding or epoch".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalIntentRequest {
    pub expected_head: Option<Uuid>,
    pub lease_generation: u64,
    pub operation_id: String,
    pub idempotency_key: String,
    #[serde(default)]
    pub binding_id: String,
    #[serde(default)]
    pub binding_epoch: u64,
    pub replay_class: ReplayClass,
}

impl CausalIntentRequest {
    pub(crate) fn for_operation(
        expected_head: Option<Uuid>,
        lease_generation: u64,
        operation: &MutationPermitRecord,
    ) -> Self {
        Self {
            expected_head,
            lease_generation,
            operation_id: operation.operation_id.to_string(),
            idempotency_key: operation.idempotency_key.to_string(),
            binding_id: operation.binding_id.to_string(),
            binding_epoch: operation.epoch,
            replay_class: operation.intent.replay_class.clone(),
        }
    }

    pub(crate) fn validate(&self, lease: &PipeFsLease) -> Result<(), PipeFsError> {
        if self.lease_generation == 0 || self.lease_generation != lease.generation {
            return Err(PipeFsError::Protocol(
                "operation intent lease generation does not match the authenticated lease".into(),
            ));
        }
        if self.operation_id.trim().is_empty()
            || self.idempotency_key.trim().is_empty()
            || self.binding_id.trim().is_empty()
            || self.binding_epoch == 0
        {
            return Err(PipeFsError::Protocol(
                "operation intent requires operation, idempotency, binding, and epoch fences"
                    .into(),
            ));
        }
        if self.replay_class != ReplayClass::NonReplayableExternal {
            return Err(PipeFsError::Protocol(
                "remote intent acknowledgement is reserved for non-replayable effects".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalIntentReceipt {
    pub operation_id: String,
    pub lease_generation: u64,
    #[serde(default)]
    pub binding_id: String,
    #[serde(default)]
    pub binding_epoch: u64,
    pub acknowledged: bool,
    #[serde(default)]
    pub replayed: bool,
}

impl CausalIntentReceipt {
    pub(crate) fn validate(
        &self,
        request: &CausalIntentRequest,
        lease: &PipeFsLease,
    ) -> Result<(), PipeFsError> {
        if self.operation_id != request.operation_id
            || self.lease_generation != lease.generation
            || self.binding_id != request.binding_id
            || self.binding_epoch != request.binding_epoch
            || !self.acknowledged
        {
            return Err(PipeFsError::Protocol(
                "operation intent acknowledgement does not match the request fence".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalTranscriptRecord {
    pub record_id: u64,
    #[serde(default)]
    pub client_record_id: String,
    pub record_type: String,
    pub payload: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalCommitRequest {
    pub expected_head: Option<Uuid>,
    pub lease_generation: u64,
    pub uploaded_revision: Option<Uuid>,
    pub uploaded_artifact: Option<ArtifactDescriptor>,
    pub operation: CausalOperationReceipt,
    pub transcript_records: Vec<CausalTranscriptRecord>,
}

impl CausalCommitRequest {
    pub(crate) fn validate(&self, lease: &PipeFsLease) -> Result<(), PipeFsError> {
        if self.lease_generation == 0 || self.lease_generation != lease.generation {
            return Err(PipeFsError::Protocol(
                "causal commit lease generation does not match the authenticated lease".into(),
            ));
        }
        if self.operation.operation_id.trim().is_empty()
            || self.operation.idempotency_key.trim().is_empty()
            || self.operation.binding_id.trim().is_empty()
            || self.operation.binding_epoch == 0
        {
            return Err(PipeFsError::Protocol(
                "causal commit requires operation, idempotency, binding, and epoch fences".into(),
            ));
        }
        if self.uploaded_revision.is_some() != self.uploaded_artifact.is_some() {
            return Err(PipeFsError::Protocol(
                "uploaded revision and artifact descriptor must be supplied together".into(),
            ));
        }
        if self
            .transcript_records
            .windows(2)
            .any(|pair| pair[0].record_id >= pair[1].record_id)
        {
            return Err(PipeFsError::Protocol(
                "causal transcript record IDs must be strictly increasing".into(),
            ));
        }
        if self.transcript_records.iter().any(|record| {
            record.record_id == 0
                || record.client_record_id.trim().is_empty()
                || record.record_type.trim().is_empty()
        }) {
            return Err(PipeFsError::Protocol(
                "causal transcript records require stable IDs and record types".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalCommitReceipt {
    pub head: Option<Uuid>,
    pub manifest_digest: Option<String>,
    pub transcript_cursor: u64,
    pub operation_id: String,
    #[serde(default)]
    pub replayed: bool,
}

impl CausalCommitReceipt {
    pub(crate) fn validate_for_request(
        &self,
        request: &CausalCommitRequest,
        previous_transcript_cursor: u64,
    ) -> Result<(), PipeFsError> {
        if self.operation_id != request.operation.operation_id {
            return Err(PipeFsError::Protocol(
                "causal commit receipt names a different operation".into(),
            ));
        }
        let submitted = u64::try_from(request.transcript_records.len()).map_err(|_| {
            PipeFsError::Protocol("causal transcript record count exceeds u64".into())
        })?;
        if submitted > 0 && self.transcript_cursor == 0 {
            return Err(PipeFsError::Protocol(
                "causal commit returned a zero cursor for submitted transcript records".into(),
            ));
        }
        let minimum_cursor = if self.replayed {
            previous_transcript_cursor
        } else {
            previous_transcript_cursor
                .checked_add(submitted)
                .ok_or_else(|| PipeFsError::Protocol("causal transcript cursor overflow".into()))?
        };
        if self.transcript_cursor < minimum_cursor {
            return Err(PipeFsError::Protocol(format!(
                "causal commit cursor {} does not acknowledge {} submitted record(s) after cursor {}",
                self.transcript_cursor, submitted, previous_transcript_cursor
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use hi_workspace::{ExecutionReport, ReplayClass};

    use super::*;

    fn request() -> CausalCommitRequest {
        CausalCommitRequest {
            expected_head: None,
            lease_generation: 7,
            uploaded_revision: None,
            uploaded_artifact: None,
            operation: CausalOperationReceipt {
                operation_id: "operation-1".into(),
                idempotency_key: "key-1".into(),
                binding_id: "binding-1".into(),
                binding_epoch: 3,
                replay_class: ReplayClass::PureWorkspace,
                execution: ExecutionReport::succeeded(None),
            },
            transcript_records: vec![CausalTranscriptRecord {
                record_id: 4,
                client_record_id: "record-4".into(),
                record_type: "assistant".into(),
                payload: serde_json::json!({"text": "done"}),
            }],
        }
    }

    #[test]
    fn request_is_fenced_to_the_authenticated_lease_generation() {
        let lease = PipeFsLease {
            token: "secret".into(),
            generation: 7,
        };
        request().validate(&lease).unwrap();
        let mut stale = request();
        stale.lease_generation = 6;
        assert!(stale.validate(&lease).is_err());
    }

    #[test]
    fn operation_is_fenced_to_the_exact_binding_epoch() {
        let operation = request().operation;
        operation.validate_binding("binding-1", 3).unwrap();
        assert!(operation.validate_binding("binding-1", 4).is_err());
        assert!(operation.validate_binding("binding-2", 3).is_err());

        let lease = PipeFsLease {
            token: "secret".into(),
            generation: 7,
        };
        let mut unfenced = request();
        unfenced.operation.binding_epoch = 0;
        assert!(unfenced.validate(&lease).is_err());
    }

    #[test]
    fn transcript_records_must_be_causally_ordered() {
        let lease = PipeFsLease {
            token: "secret".into(),
            generation: 7,
        };
        let mut invalid = request();
        invalid.transcript_records.push(CausalTranscriptRecord {
            record_id: 4,
            client_record_id: "record-4-duplicate".into(),
            record_type: "tool".into(),
            payload: serde_json::Value::Null,
        });
        assert!(invalid.validate(&lease).is_err());
    }

    #[test]
    fn intent_is_only_valid_for_non_replayable_effects() {
        let lease = PipeFsLease {
            token: "secret".into(),
            generation: 7,
        };
        let mut intent = CausalIntentRequest {
            expected_head: None,
            lease_generation: 7,
            operation_id: "operation-1".into(),
            idempotency_key: "key-1".into(),
            binding_id: "binding-1".into(),
            binding_epoch: 3,
            replay_class: ReplayClass::NonReplayableExternal,
        };
        intent.validate(&lease).unwrap();
        intent.replay_class = ReplayClass::PureWorkspace;
        assert!(intent.validate(&lease).is_err());
    }

    #[test]
    fn intent_acknowledgement_replay_keeps_the_exact_binding_fence() {
        let lease = PipeFsLease {
            token: "secret".into(),
            generation: 7,
        };
        let request = CausalIntentRequest {
            expected_head: None,
            lease_generation: 7,
            operation_id: "operation-1".into(),
            idempotency_key: "key-1".into(),
            binding_id: "binding-1".into(),
            binding_epoch: 3,
            replay_class: ReplayClass::NonReplayableExternal,
        };
        let mut receipt = CausalIntentReceipt {
            operation_id: "operation-1".into(),
            lease_generation: 7,
            binding_id: "binding-1".into(),
            binding_epoch: 3,
            acknowledged: true,
            replayed: true,
        };
        receipt.validate(&request, &lease).unwrap();
        receipt.binding_epoch = 2;
        assert!(receipt.validate(&request, &lease).is_err());
        receipt.binding_epoch = 3;
        receipt.binding_id = "binding-2".into();
        assert!(receipt.validate(&request, &lease).is_err());
    }

    #[test]
    fn receipt_must_advance_past_every_submitted_transcript_record() {
        let request = request();
        let stale = CausalCommitReceipt {
            head: None,
            manifest_digest: None,
            transcript_cursor: 9,
            operation_id: request.operation.operation_id.clone(),
            replayed: false,
        };
        assert!(stale.validate_for_request(&request, 9).is_err());

        let acknowledged = CausalCommitReceipt {
            transcript_cursor: 10,
            ..stale
        };
        acknowledged.validate_for_request(&request, 9).unwrap();

        let replayed = CausalCommitReceipt {
            transcript_cursor: 9,
            replayed: true,
            ..acknowledged
        };
        replayed.validate_for_request(&request, 9).unwrap();
    }
}
