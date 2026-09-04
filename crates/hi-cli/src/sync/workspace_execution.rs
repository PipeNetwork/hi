//! Durable staging for workspace execution records that must be published
//! with the next PipeFS settlement boundary.

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use super::{RECORD_TYPE_USAGE, RemoteSessionSink, lock_recover};

pub(super) const WORKSPACE_EXECUTION_RECORD_TYPE: &str = "workspace_execution";

/// Protocol-1 servers predate `workspace_execution` and reject unknown record
/// types. Carry the evidence in a known, non-message record so it is durable
/// without creating a second visible assistant/tool message. Existing readers
/// treat the unknown payload tag as an opaque boundary. Protocol 2 unwraps the
/// carrier before submitting its causal transcript batch.
#[derive(Serialize)]
struct CompatibilityWorkspaceExecution<'a> {
    #[serde(rename = "type")]
    record_type: &'static str,
    #[serde(flatten)]
    execution: &'a hi_agent::WorkspaceTranscriptExecution,
}

#[derive(Clone)]
pub(super) struct RequiredWorkspaceStageFailure {
    record: hi_agent::WorkspaceTranscriptExecution,
    detail: String,
}

impl RemoteSessionSink {
    /// Durably enqueue the exact execution record which must share the next
    /// PipeFS publication boundary. The deterministic id makes an identical
    /// retry idempotent; a different payload receives a different digest and
    /// remains visible rather than overwriting evidence.
    pub(crate) fn stage_workspace_execution(
        &self,
        record: &hi_agent::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        // Staging is a local durability action. It must remain possible after
        // the remote lease disappears so the ensuing recovery still carries
        // the exact result whose effect may already have happened.
        let result = self.enqueue_workspace_execution(record);
        if let Err(error) = &result {
            let mut failure = lock_recover(&self.required_workspace_stage_failure);
            *failure = Some(RequiredWorkspaceStageFailure {
                record: record.clone(),
                detail: format!("{error:#}"),
            });
        } else {
            *lock_recover(&self.required_workspace_stage_failure) = None;
        }
        result
    }

    pub(crate) fn ensure_workspace_execution_staged(&self) -> Result<()> {
        let pending = lock_recover(&self.required_workspace_stage_failure).clone();
        if let Some(pending) = pending {
            match self.enqueue_workspace_execution(&pending.record) {
                Ok(()) => {
                    *lock_recover(&self.required_workspace_stage_failure) = None;
                }
                Err(error) => {
                    let original_detail = pending.detail;
                    let detail = format!("{error:#}");
                    *lock_recover(&self.required_workspace_stage_failure) =
                        Some(RequiredWorkspaceStageFailure {
                            record: pending.record,
                            detail: detail.clone(),
                        });
                    anyhow::bail!(
                        "a required workspace execution transcript was not staged: {}; retry failed: {detail}",
                        original_detail
                    );
                }
            }
        }
        Ok(())
    }

    /// Prove that compatibility publication has an exact durable transcript
    /// counterpart. The proof may be either the still-queued carrier or an
    /// acknowledgement retained atomically when that carrier left the outbox.
    pub(crate) fn ensure_compatibility_workspace_execution(
        &self,
        operation: &hi_pipefs::CausalOperationReceipt,
    ) -> Result<()> {
        let execution = serde_json::to_value(&operation.execution)?;
        let digest =
            crate::sync_store::records::execution_digest(&operation.operation_id, &execution)?;
        let queued = self.store.pending_workspace_execution(
            &self.session_id,
            &operation.operation_id,
            &digest,
        )?;
        let acknowledged = self.store.workspace_execution_ack_cursor(
            &self.session_id,
            &operation.operation_id,
            &digest,
        )?;
        ensure!(
            queued || acknowledged.is_some(),
            "compatibility recovery has no exact durable transcript proof for operation {}; recovery cache retained",
            operation.operation_id
        );
        Ok(())
    }

    pub(crate) fn compatibility_workspace_execution_cursor(
        &self,
        operation: &hi_pipefs::CausalOperationReceipt,
    ) -> Result<u64> {
        let execution = serde_json::to_value(&operation.execution)?;
        let digest =
            crate::sync_store::records::execution_digest(&operation.operation_id, &execution)?;
        self.store
            .workspace_execution_ack_cursor(
                &self.session_id,
                &operation.operation_id,
                &digest,
            )?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "compatibility transcript acknowledgement did not retain exact proof for operation {}; recovery cache retained",
                    operation.operation_id
                )
            })
    }

    fn enqueue_workspace_execution(
        &self,
        record: &hi_agent::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        anyhow::ensure!(
            self.pipefs_sync_required(),
            "PipeFS transcript durability is not pinned for this session"
        );
        let payload = serde_json::to_string(&CompatibilityWorkspaceExecution {
            record_type: WORKSPACE_EXECUTION_RECORD_TYPE,
            execution: record,
        })
        .context("serializing workspace execution transcript")?;
        let digest = blake3::hash(payload.as_bytes()).to_hex().to_string();
        let id = format!(
            "workspace-execution:{}:{}",
            record.operation_id,
            &digest[..16]
        );
        self.enqueue_reconciled(&id, RECORD_TYPE_USAGE, &payload)
            .context("enqueueing required workspace execution transcript")
    }
}

/// Convert the protocol-1-safe outbox representation back to the native
/// protocol-2 causal record. The outbox remains portable across a crash and a
/// subsequent activation that negotiates a different writer protocol.
pub(super) fn project_causal_workspace_execution(
    record_type: &str,
    mut payload: serde_json::Value,
) -> Result<(String, serde_json::Value)> {
    if record_type != RECORD_TYPE_USAGE
        || payload.get("type").and_then(serde_json::Value::as_str)
            != Some(WORKSPACE_EXECUTION_RECORD_TYPE)
    {
        return Ok((record_type.to_string(), payload));
    }
    let object = payload
        .as_object_mut()
        .context("workspace execution compatibility carrier is not an object")?;
    object.remove("type");
    ensure!(
        object.contains_key("operation_id") && object.contains_key("execution"),
        "workspace execution compatibility carrier is incomplete"
    );
    Ok((WORKSPACE_EXECUTION_RECORD_TYPE.to_string(), payload))
}
