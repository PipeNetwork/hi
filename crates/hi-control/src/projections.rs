//! Durable workspace/session projections coupled to the required event log.
//!
//! The records in this module are intentionally local to `hi-control`. The
//! workspace orchestrator may adapt its richer domain types at the boundary
//! without introducing dependency cycles into the persistence crate.

use hi_events::{EventDurability, EventReceipt, RunEvent};
use rusqlite::{OptionalExtension, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{ControlError, ControlStore, Result, append_event_in_transaction, validate_id};

#[path = "projection_write.rs"]
mod write;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceAuthority {
    Local,
    #[serde(rename = "pipefs")]
    PipeFs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceProjectionState {
    Ready,
    Mutating,
    Settling,
    PendingRemote,
    LeaseUncertain,
    LeaseLost,
    Conflict,
    TranscriptPending,
    CleanupPending,
    RecoveryRequired,
    JournalCorrupt,
    Incompatible,
    LocalAuditDegraded,
    Closed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceBindingRecord {
    pub binding_id: String,
    pub workspace_id: String,
    pub session_id: Option<String>,
    pub epoch: u64,
    pub authority: WorkspaceAuthority,
    pub state: WorkspaceProjectionState,
    pub workspace_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<serde_json::Value>,
    pub revision: u64,
    pub opened_at_ms: u64,
    pub updated_at_ms: u64,
    pub closed_at_ms: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationReplayClass {
    PureWorkspace,
    IdempotentExternal,
    NonReplayableExternal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceOperationStatus {
    Admitted,
    Executing,
    ExecutionRecorded,
    Settling,
    Durable,
    NoChange,
    Pending,
    Indeterminate,
    LeaseLost,
    Conflict,
    TranscriptPending,
    RecoveryRequired,
    LocalAuditDegraded,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceOperationRecord {
    pub operation_id: String,
    pub binding_id: String,
    pub epoch: u64,
    pub session_id: Option<String>,
    pub run_id: Option<String>,
    pub attempt_id: Option<String>,
    pub job_id: Option<String>,
    pub kind: String,
    pub replay_class: OperationReplayClass,
    pub status: WorkspaceOperationStatus,
    pub operation_digest: String,
    pub idempotency_key: String,
    pub base_version: Option<String>,
    pub result_version: Option<String>,
    pub execution_ref: Option<String>,
    pub settlement_ref: Option<String>,
    pub error: Option<String>,
    pub revision: u64,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub settled_at_ms: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlJobKind {
    Process,
    ReadAgent,
    WriteCandidate,
    Hook,
    Compaction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlEffectScope {
    ReadOnly,
    CandidateOnly,
    LiveWriter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlJobState {
    Queued,
    Starting,
    Running,
    ReadyToMerge,
    Merging,
    Settling,
    CancelRequested,
    Succeeded,
    Failed,
    Cancelled,
    DurabilityPending,
    RecoveryRequired,
    Orphaned,
    Stale,
}

impl ControlJobState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::Orphaned | Self::Stale
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlJobRecord {
    pub job_id: String,
    pub session_id: Option<String>,
    pub run_id: Option<String>,
    pub attempt_id: Option<String>,
    pub binding_id: Option<String>,
    pub epoch: Option<u64>,
    pub kind: ControlJobKind,
    pub effect_scope: ControlEffectScope,
    pub state: ControlJobState,
    pub application_state: Option<String>,
    pub operation_digest: Option<String>,
    pub idempotency_key: Option<String>,
    pub candidate_ref: Option<String>,
    pub result_ref: Option<String>,
    pub workspace_version: Option<String>,
    pub error: Option<String>,
    pub revision: u64,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub cancel_requested_at_ms: Option<u64>,
    pub finished_at_ms: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceRecoveryStatus {
    Required,
    Inspecting,
    Retrying,
    Resolved,
    Discarded,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceRecoveryRecord {
    pub recovery_id: String,
    pub binding_id: Option<String>,
    pub workspace_id: String,
    pub session_id: Option<String>,
    pub operation_id: Option<String>,
    pub job_id: Option<String>,
    pub kind: String,
    pub status: WorkspaceRecoveryStatus,
    pub digest: Option<String>,
    pub artifact_ref: Option<String>,
    pub detail: Option<String>,
    pub error: Option<String>,
    pub revision: u64,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub resolved_at_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionSnapshotRecord {
    pub snapshot_id: String,
    pub session_id: String,
    pub reducer_version: u32,
    pub through_sequence: u64,
    pub state_ref: String,
    pub state_digest: String,
    pub state_bytes: u64,
    pub revision: u64,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "projection")]
pub enum ProjectionTransition {
    WorkspaceBinding(WorkspaceBindingRecord),
    WorkspaceOperation(WorkspaceOperationRecord),
    Job(ControlJobRecord),
    WorkspaceRecovery(WorkspaceRecoveryRecord),
    SessionSnapshot(SessionSnapshotRecord),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionKind {
    WorkspaceBinding,
    WorkspaceOperation,
    Job,
    WorkspaceRecovery,
    SessionSnapshot,
}

impl ProjectionKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::WorkspaceBinding => "workspace_binding",
            Self::WorkspaceOperation => "workspace_operation",
            Self::Job => "job",
            Self::WorkspaceRecovery => "workspace_recovery",
            Self::SessionSnapshot => "session_snapshot",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionEventReceipt {
    pub projection_kind: ProjectionKind,
    pub projection_id: String,
    pub projection_revision: u64,
    pub event: EventReceipt,
}

impl ProjectionTransition {
    fn identity(&self) -> (ProjectionKind, &str, u64) {
        match self {
            Self::WorkspaceBinding(record) => (
                ProjectionKind::WorkspaceBinding,
                &record.binding_id,
                record.revision,
            ),
            Self::WorkspaceOperation(record) => (
                ProjectionKind::WorkspaceOperation,
                &record.operation_id,
                record.revision,
            ),
            Self::Job(record) => (ProjectionKind::Job, &record.job_id, record.revision),
            Self::WorkspaceRecovery(record) => (
                ProjectionKind::WorkspaceRecovery,
                &record.recovery_id,
                record.revision,
            ),
            Self::SessionSnapshot(record) => (
                ProjectionKind::SessionSnapshot,
                &record.snapshot_id,
                record.revision,
            ),
        }
    }

    fn validate(&self) -> Result<()> {
        let (_, id, revision) = self.identity();
        validate_id(id, "projection id")?;
        if revision == 0 {
            return Err(ControlError::Invalid(
                "projection revision must start at one".into(),
            ));
        }
        match self {
            Self::WorkspaceBinding(record) => {
                validate_id(&record.workspace_id, "workspace_id")?;
                validate_times(record.opened_at_ms, record.updated_at_ms)?;
            }
            Self::WorkspaceOperation(record) => {
                validate_id(&record.binding_id, "binding_id")?;
                validate_id(&record.idempotency_key, "idempotency_key")?;
                validate_times(record.created_at_ms, record.updated_at_ms)?;
            }
            Self::Job(record) => validate_times(record.created_at_ms, record.updated_at_ms)?,
            Self::WorkspaceRecovery(record) => {
                validate_id(&record.workspace_id, "workspace_id")?;
                validate_times(record.created_at_ms, record.updated_at_ms)?;
            }
            Self::SessionSnapshot(record) => {
                validate_id(&record.session_id, "session_id")?;
                if record.state_ref.is_empty() || record.state_digest.is_empty() {
                    return Err(ControlError::Invalid(
                        "session snapshot requires state_ref and state_digest".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl ControlStore {
    /// Atomically append a required event and advance exactly one durable
    /// projection. Projection revisions are compare-and-swap tokens: revision
    /// one creates a record, and every later transition must increment by one.
    pub fn commit_projection_event(
        &self,
        transition: ProjectionTransition,
        mut event: RunEvent,
    ) -> Result<ProjectionEventReceipt> {
        transition.validate()?;
        if event.schema_version != hi_events::EVENT_SCHEMA_VERSION {
            return Err(ControlError::Invalid(format!(
                "unsupported event schema version {}",
                event.schema_version
            )));
        }
        if event.durability != EventDurability::Required {
            return Err(ControlError::Invalid(
                "projection transitions require a required-durability event".into(),
            ));
        }

        let (kind, projection_id, revision) = transition.identity();
        let projection_id = projection_id.to_owned();
        let transition_json = serde_json::to_vec(&transition)?;
        let digest = blake3::hash(&transition_json).to_hex().to_string();
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        if let Some(existing) = existing_projection_event(&tx, &event.event_id)? {
            event.sequence = existing.event.sequence;
            if existing.kind != kind.as_str()
                || existing.id != projection_id
                || existing.revision as u64 != revision
                || existing.digest != digest
                || existing.event != event
            {
                return Err(ControlError::Invalid(format!(
                    "event {} was reused for a different projection transition",
                    event.event_id
                )));
            }
            tx.commit()?;
            return Ok(ProjectionEventReceipt {
                projection_kind: kind,
                projection_id,
                projection_revision: revision,
                event: EventReceipt {
                    event_id: event.event_id,
                    sequence: event.sequence,
                },
            });
        }

        let event_exists: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM run_events WHERE event_id = ?1)",
            [&event.event_id],
            |row| row.get(0),
        )?;
        if event_exists {
            return Err(ControlError::Invalid(format!(
                "event {} is already committed without this projection",
                event.event_id
            )));
        }

        let event_receipt = append_event_in_transaction(&tx, &mut event)?;
        write::apply_transition(&tx, &transition, &event.event_id)?;
        tx.execute(
            "INSERT INTO control_projection_events
             (event_id, projection_kind, projection_id, projection_revision, projection_digest)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                event.event_id,
                kind.as_str(),
                projection_id,
                revision as i64,
                digest
            ],
        )?;
        tx.commit()?;

        Ok(ProjectionEventReceipt {
            projection_kind: kind,
            projection_id,
            projection_revision: revision,
            event: event_receipt,
        })
    }

    pub fn get_workspace_binding(&self, id: &str) -> Result<Option<WorkspaceBindingRecord>> {
        self.get_projection("control_workspace_bindings", "binding_id", id)
    }

    pub fn latest_workspace_binding(
        &self,
        workspace_id: &str,
    ) -> Result<Option<WorkspaceBindingRecord>> {
        let connection = self.lock()?;
        let json = connection
            .query_row(
                "SELECT record_json FROM control_workspace_bindings
                 WHERE workspace_id = ?1 AND authority = 'local'
                 ORDER BY epoch DESC, updated_at_ms DESC LIMIT 1",
                [workspace_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        json.map(parse_record).transpose()
    }

    pub fn latest_pipefs_binding(
        &self,
        session_id: &str,
    ) -> Result<Option<WorkspaceBindingRecord>> {
        let connection = self.lock()?;
        let json = connection
            .query_row(
                "SELECT record_json FROM control_workspace_bindings
                 WHERE session_id = ?1 AND authority = 'pipefs'
                 ORDER BY epoch DESC, updated_at_ms DESC LIMIT 1",
                [session_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        json.map(parse_record).transpose()
    }

    pub fn unsettled_workspace_bindings(
        &self,
        workspace_id: &str,
    ) -> Result<Vec<WorkspaceBindingRecord>> {
        self.query_projections(
            "SELECT b.record_json FROM control_workspace_bindings b
             WHERE b.workspace_id = ?1 AND b.authority = 'local'
               AND (EXISTS (
                     SELECT 1 FROM control_jobs j WHERE j.binding_id = b.binding_id
                       AND j.state NOT IN ('succeeded','failed','cancelled','orphaned','stale')
                   ) OR EXISTS (
                   SELECT 1 FROM control_workspace_operations o
                     WHERE o.binding_id = b.binding_id
                       AND o.status NOT IN ('durable','no_change','local_audit_degraded','failed')
                  ) OR EXISTS (
                    SELECT 1 FROM control_workspace_recoveries r
                     WHERE r.binding_id = b.binding_id
                       AND r.status NOT IN ('resolved','discarded')
                  ))
             ORDER BY b.epoch, b.updated_at_ms",
            workspace_id,
        )
    }

    pub fn unsettled_pipefs_bindings(
        &self,
        session_id: &str,
    ) -> Result<Vec<WorkspaceBindingRecord>> {
        self.query_projections(
            "SELECT b.record_json FROM control_workspace_bindings b
             WHERE b.session_id = ?1 AND b.authority = 'pipefs'
               AND (EXISTS (
                     SELECT 1 FROM control_jobs j WHERE j.binding_id = b.binding_id
                       AND j.state NOT IN ('succeeded','failed','cancelled','orphaned','stale')
                   ) OR EXISTS (
                   SELECT 1 FROM control_workspace_operations o
                     WHERE o.binding_id = b.binding_id
                       AND o.status NOT IN ('durable','no_change','local_audit_degraded','failed')
                  ) OR EXISTS (
                    SELECT 1 FROM control_workspace_recoveries r
                     WHERE r.binding_id = b.binding_id
                       AND r.status NOT IN ('resolved','discarded')
                  ))
             ORDER BY b.epoch, b.updated_at_ms",
            session_id,
        )
    }

    pub fn get_workspace_operation(&self, id: &str) -> Result<Option<WorkspaceOperationRecord>> {
        self.get_projection("control_workspace_operations", "operation_id", id)
    }

    pub fn operations_for_binding(
        &self,
        binding_id: &str,
    ) -> Result<Vec<WorkspaceOperationRecord>> {
        self.query_projections(
            "SELECT record_json FROM control_workspace_operations
             WHERE binding_id = ?1 ORDER BY created_at_ms, operation_id",
            binding_id,
        )
    }

    pub fn get_job(&self, id: &str) -> Result<Option<ControlJobRecord>> {
        self.get_projection("control_jobs", "job_id", id)
    }

    pub fn get_workspace_recovery(&self, id: &str) -> Result<Option<WorkspaceRecoveryRecord>> {
        self.get_projection("control_workspace_recoveries", "recovery_id", id)
    }

    pub fn jobs_for_binding(&self, binding_id: &str) -> Result<Vec<ControlJobRecord>> {
        self.query_projections(
            "SELECT record_json FROM control_jobs
             WHERE binding_id = ?1 ORDER BY created_at_ms, job_id",
            binding_id,
        )
    }

    pub fn unsettled_jobs(&self) -> Result<Vec<ControlJobRecord>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare(
            "SELECT record_json FROM control_jobs
             WHERE state NOT IN ('succeeded', 'failed', 'cancelled', 'orphaned', 'stale')
             ORDER BY updated_at_ms, job_id",
        )?;
        collect_records(statement.query_map([], |row| row.get::<_, String>(0))?)
    }

    pub fn recoveries_for_workspace(
        &self,
        workspace_id: &str,
    ) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.query_projections(
            "SELECT record_json FROM control_workspace_recoveries
             WHERE workspace_id = ?1 ORDER BY updated_at_ms DESC, recovery_id",
            workspace_id,
        )
    }

    pub fn recoveries_for_binding(&self, binding_id: &str) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.query_projections(
            "SELECT record_json FROM control_workspace_recoveries
             WHERE binding_id = ?1 ORDER BY updated_at_ms, recovery_id",
            binding_id,
        )
    }

    pub fn recoveries_for_session(&self, session_id: &str) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.query_projections(
            "SELECT record_json FROM control_workspace_recoveries
             WHERE session_id = ?1 ORDER BY updated_at_ms DESC, recovery_id",
            session_id,
        )
    }

    pub fn recoveries_for_operation(
        &self,
        operation_id: &str,
    ) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.query_projections(
            "SELECT record_json FROM control_workspace_recoveries
             WHERE operation_id = ?1 ORDER BY updated_at_ms, recovery_id",
            operation_id,
        )
    }

    pub fn recoveries_for_job(&self, job_id: &str) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.query_projections(
            "SELECT record_json FROM control_workspace_recoveries
             WHERE job_id = ?1 ORDER BY updated_at_ms, recovery_id",
            job_id,
        )
    }

    pub fn latest_session_snapshot(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionSnapshotRecord>> {
        let connection = self.lock()?;
        let json = connection
            .query_row(
                "SELECT record_json FROM session_snapshots
                 WHERE session_id = ?1 ORDER BY through_sequence DESC LIMIT 1",
                [session_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        json.map(parse_record).transpose()
    }

    fn get_projection<T: DeserializeOwned>(
        &self,
        table: &str,
        id_column: &str,
        id: &str,
    ) -> Result<Option<T>> {
        // Callers are private fixed literals; user-controlled table names are
        // never accepted here.
        let sql = format!("SELECT record_json FROM {table} WHERE {id_column} = ?1");
        let connection = self.lock()?;
        let json = connection
            .query_row(&sql, [id], |row| row.get::<_, String>(0))
            .optional()?;
        json.map(parse_record).transpose()
    }

    fn query_projections<T: DeserializeOwned>(&self, sql: &str, value: &str) -> Result<Vec<T>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare(sql)?;
        collect_records(statement.query_map([value], |row| row.get::<_, String>(0))?)
    }
}

struct ExistingProjectionEvent {
    kind: String,
    id: String,
    revision: i64,
    digest: String,
    event: RunEvent,
}

fn existing_projection_event(
    tx: &Transaction<'_>,
    event_id: &str,
) -> Result<Option<ExistingProjectionEvent>> {
    let row = tx
        .query_row(
            "SELECT p.projection_kind, p.projection_id, p.projection_revision,
                    p.projection_digest, e.event_json
             FROM control_projection_events p
             JOIN run_events e ON e.event_id = p.event_id
             WHERE p.event_id = ?1",
            [event_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )
        .optional()?;
    row.map(|(kind, id, revision, digest, json)| {
        Ok(ExistingProjectionEvent {
            kind,
            id,
            revision,
            digest,
            event: serde_json::from_str(&json)?,
        })
    })
    .transpose()
}

fn validate_times(created: u64, updated: u64) -> Result<()> {
    if updated < created {
        return Err(ControlError::Invalid(
            "projection update predates its creation".into(),
        ));
    }
    Ok(())
}

pub(super) fn parse_record<T: DeserializeOwned>(json: String) -> Result<T> {
    serde_json::from_str(&json).map_err(Into::into)
}

fn collect_records<T: DeserializeOwned>(
    rows: rusqlite::MappedRows<'_, impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<String>>,
) -> Result<Vec<T>> {
    rows.map(|row| parse_record(row?)).collect()
}

#[cfg(test)]
#[path = "projection_tests.rs"]
mod tests;
