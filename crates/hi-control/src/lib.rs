//! Durable local control-plane state for hi.
//!
//! The control store is deliberately independent of the CLI, agent loop, and
//! RSI trust domain. It owns the records needed to coordinate local processes
//! safely; large prompts, diffs, logs, and model responses remain artifacts
//! referenced by hash rather than being copied into this database.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use hi_events::{EventBus, EventError, EventReceipt, EventSink, RunEvent};
use hi_policy::{
    ApprovalDecision, ApprovalRecord, ApprovalState, ApprovalStore, CapabilityKind,
    CapabilityRequest, OperationDigest,
};
pub use hi_policy::{PolicySnapshot, Principal, Provenance, ScopeKind, ScopeRef};
use rusqlite::{Connection, OptionalExtension, Row, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const CONTROL_SCHEMA_VERSION: i64 = 1;
pub const DEFAULT_LEASE_TTL_MS: u64 = 30_000;

#[derive(Debug, Error)]
pub enum ControlError {
    #[error("control store database: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("control store serialization: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("control store io: {0}")]
    Io(#[from] std::io::Error),
    #[error("run not found: {0}")]
    RunNotFound(String),
    #[error("attempt not found: {0}")]
    AttemptNotFound(String),
    #[error("effect not found: {0}")]
    EffectNotFound(String),
    #[error("run is terminal: {0}")]
    RunTerminal(String),
    #[error("attempt already has an active lease: {0}")]
    AttemptBusy(String),
    #[error("attempt lease is lost or fenced: {0}")]
    LeaseLost(String),
    #[error("invalid control record: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, ControlError>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunKind {
    Interactive,
    Workflow,
    Loop,
    Delegation,
    Verification,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Queued,
    Running,
    Waiting,
    Succeeded,
    Failed,
    Cancelled,
    Abandoned,
}

impl RunStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Waiting => "waiting",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Abandoned => "abandoned",
        }
    }

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "queued" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "waiting" => Ok(Self::Waiting),
            "succeeded" => Ok(Self::Succeeded),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            "abandoned" => Ok(Self::Abandoned),
            other => Err(ControlError::Invalid(format!("unknown run status {other}"))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesiredState {
    Run,
    Pause,
    Cancel,
}

impl DesiredState {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Pause => "pause",
            Self::Cancel => "cancel",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteSnapshot {
    pub harness: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub capability_digest: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: String,
    pub kind: RunKind,
    pub workspace_id: Option<String>,
    pub scope: Option<ScopeRef>,
    pub session_id: Option<String>,
    pub parent_run_id: Option<String>,
    pub status: RunStatus,
    pub desired_state: DesiredState,
    pub policy_snapshot: Option<PolicySnapshot>,
    pub route_snapshot: Option<RouteSnapshot>,
    pub provenance: Option<Provenance>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewRun {
    pub run_id: Option<String>,
    pub kind: RunKind,
    pub workspace_id: Option<String>,
    pub scope: Option<ScopeRef>,
    pub session_id: Option<String>,
    pub parent_run_id: Option<String>,
    pub policy_snapshot: Option<PolicySnapshot>,
    pub route_snapshot: Option<RouteSnapshot>,
    pub provenance: Option<Provenance>,
    pub desired_state: DesiredState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptStatus {
    Running,
    Waiting,
    Succeeded,
    Failed,
    Lost,
    Cancelled,
}

impl AttemptStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Waiting => "waiting",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Lost => "lost",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub attempt_id: String,
    pub run_id: String,
    pub number: u32,
    pub worker_id: String,
    pub status: AttemptStatus,
    pub lease_generation: u64,
    pub lease_expires_at_ms: u64,
    pub last_heartbeat_at_ms: u64,
    pub started_at_ms: u64,
    pub finished_at_ms: Option<u64>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptLease {
    pub attempt: Attempt,
    pub fencing_token: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectStatus {
    Planned,
    Started,
    Succeeded,
    Failed,
    Denied,
    Unknown,
    Reconciled,
}

impl EffectStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Planned => "planned",
            Self::Started => "started",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Denied => "denied",
            Self::Unknown => "unknown",
            Self::Reconciled => "reconciled",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub hash: String,
    pub media_type: String,
    pub size_bytes: Option<u64>,
    pub scope_id: Option<String>,
    pub sensitivity: String,
    pub producer_run_id: Option<String>,
    pub producer_attempt_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    Memory,
    Skill,
    Artifact,
    CredentialReference,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopedResource {
    pub resource_id: String,
    pub kind: ResourceKind,
    pub scope: ScopeRef,
    pub owner_id: String,
    pub digest: String,
    pub sensitivity: String,
    pub provenance: Option<Provenance>,
    pub expires_at_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectRecord {
    pub effect_id: String,
    pub run_id: String,
    pub attempt_id: String,
    pub fencing_token: u64,
    pub capability: CapabilityKind,
    pub tool: String,
    pub operation_digest: OperationDigest,
    pub idempotency_key: String,
    pub scope: Option<ScopeRef>,
    pub provenance: Option<Provenance>,
    pub status: EffectStatus,
    pub input_ref: Option<ArtifactRef>,
    pub output_ref: Option<ArtifactRef>,
    pub mutation_ref: Option<ArtifactRef>,
    pub external_ref: Option<String>,
    pub error: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewEffect {
    pub effect_id: Option<String>,
    pub run_id: String,
    pub attempt_id: String,
    pub fencing_token: u64,
    pub capability: CapabilityKind,
    pub tool: String,
    pub operation_digest: OperationDigest,
    pub idempotency_key: String,
    pub scope: Option<ScopeRef>,
    pub provenance: Option<Provenance>,
    pub input_ref: Option<ArtifactRef>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditRecord {
    pub audit_id: String,
    pub decision: String,
    pub actor: Principal,
    pub source: String,
    pub scope: Option<ScopeRef>,
    pub provenance: Option<Provenance>,
    pub policy_snapshot: Option<PolicySnapshot>,
    pub operation_digest: Option<OperationDigest>,
    pub approval_id: Option<String>,
    pub route: Option<RouteSnapshot>,
    pub effect_id: Option<String>,
    pub event_id: Option<String>,
    pub detail: Option<String>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectOutcome {
    pub status: EffectStatus,
    pub output_ref: Option<ArtifactRef>,
    pub mutation_ref: Option<ArtifactRef>,
    pub external_ref: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct ControlStore {
    connection: Arc<Mutex<Connection>>,
    path: PathBuf,
}

impl std::fmt::Debug for ControlStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlStore")
            .field("path", &self.path)
            .finish()
    }
}

impl ControlStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let connection = hi_sqlite_journal::JournalMode::for_db_path(&path)
            .open(&path)
            .map_err(|error| ControlError::Invalid(error.to_string()))?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        migrate(&connection)?;
        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
            path,
        })
    }

    pub fn open_for_state(state_root: impl AsRef<Path>) -> Result<Self> {
        Self::open(state_root.as_ref().join("events.sqlite3"))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn create_run(&self, mut new_run: NewRun) -> Result<RunRecord> {
        let run_id = new_run
            .run_id
            .take()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        validate_id(&run_id, "run_id")?;
        let now = now_ms();
        let run = RunRecord {
            run_id,
            kind: new_run.kind,
            workspace_id: new_run.workspace_id,
            scope: new_run.scope,
            session_id: new_run.session_id,
            parent_run_id: new_run.parent_run_id,
            status: RunStatus::Queued,
            desired_state: new_run.desired_state,
            policy_snapshot: new_run.policy_snapshot,
            route_snapshot: new_run.route_snapshot,
            provenance: new_run.provenance,
            created_at_ms: now,
            updated_at_ms: now,
        };
        let connection = self.lock()?;
        connection.execute(
            "INSERT INTO control_runs
             (run_id, kind, workspace_id, scope_json, session_id, parent_run_id,
              status, desired_state, policy_json, route_json, provenance_json,
              created_at_ms, updated_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                run.run_id,
                enum_json(&run.kind)?,
                run.workspace_id,
                optional_json(&run.scope)?,
                run.session_id,
                run.parent_run_id,
                run.status.as_str(),
                run.desired_state.as_str(),
                optional_json(&run.policy_snapshot)?,
                optional_json(&run.route_snapshot)?,
                optional_json(&run.provenance)?,
                run.created_at_ms as i64,
                run.updated_at_ms as i64,
            ],
        )?;
        Ok(run)
    }

    pub fn get_run(&self, run_id: &str) -> Result<Option<RunRecord>> {
        let connection = self.lock()?;
        connection
            .query_row(
                "SELECT run_id, kind, workspace_id, scope_json, session_id, parent_run_id,
                        status, desired_state, policy_json, route_json, provenance_json,
                        created_at_ms, updated_at_ms
                 FROM control_runs WHERE run_id = ?1",
                [run_id],
                row_run,
            )
            .optional()
            .map_err(Into::into)
    }

    /// Re-open a previously terminal or abandoned run for an explicit new
    /// invocation while preserving all prior attempts and effects.
    pub fn requeue_run(&self, run_id: &str, now: u64) -> Result<()> {
        let connection = self.lock()?;
        let changed = connection.execute(
            "UPDATE control_runs SET status = 'queued', desired_state = 'run',
             updated_at_ms = ?1
             WHERE run_id = ?2 AND status IN ('failed', 'cancelled', 'abandoned', 'succeeded')",
            params![now as i64, run_id],
        )?;
        if changed == 0 {
            let exists: Option<String> = connection
                .query_row(
                    "SELECT run_id FROM control_runs WHERE run_id = ?1",
                    [run_id],
                    |row| row.get(0),
                )
                .optional()?;
            return exists
                .map(|_| ())
                .ok_or_else(|| ControlError::RunNotFound(run_id.into()));
        }
        Ok(())
    }

    pub fn claim_attempt(
        &self,
        run_id: &str,
        worker_id: &str,
        now: u64,
        lease_ttl_ms: u64,
    ) -> Result<AttemptLease> {
        validate_id(worker_id, "worker_id")?;
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let run_status: Option<String> = tx
            .query_row(
                "SELECT status FROM control_runs WHERE run_id = ?1",
                [run_id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(run_status) = run_status else {
            return Err(ControlError::RunNotFound(run_id.into()));
        };
        if matches!(
            run_status.as_str(),
            "succeeded" | "failed" | "cancelled" | "abandoned"
        ) {
            return Err(ControlError::RunTerminal(run_id.into()));
        }
        let active: Option<String> = tx
            .query_row(
                "SELECT attempt_id FROM control_attempts
                 WHERE run_id = ?1 AND status IN ('running', 'waiting')
                 LIMIT 1",
                [run_id],
                |row| row.get(0),
            )
            .optional()?;
        if active.is_some() {
            return Err(ControlError::AttemptBusy(run_id.into()));
        }
        let number: u32 = tx.query_row(
            "SELECT COALESCE(MAX(number), 0) + 1 FROM control_attempts WHERE run_id = ?1",
            [run_id],
            |row| row.get::<_, i64>(0).map(|value| value as u32),
        )?;
        let fencing_token: u64 = tx.query_row(
            "SELECT COALESCE(MAX(lease_generation), 0) + 1
             FROM control_attempts WHERE run_id = ?1",
            [run_id],
            |row| row.get::<_, i64>(0).map(|value| value as u64),
        )?;
        let attempt_id = uuid::Uuid::new_v4().to_string();
        let expires = now.saturating_add(lease_ttl_ms.max(1));
        tx.execute(
            "INSERT INTO control_attempts
             (attempt_id, run_id, number, worker_id, status, lease_generation,
              lease_expires_at_ms, last_heartbeat_at_ms, started_at_ms)
             VALUES (?1, ?2, ?3, ?4, 'running', ?5, ?6, ?7, ?8)",
            params![
                attempt_id,
                run_id,
                number,
                worker_id,
                fencing_token as i64,
                expires as i64,
                now as i64,
                now as i64
            ],
        )?;
        tx.execute(
            "UPDATE control_runs SET status = 'running', updated_at_ms = ?1 WHERE run_id = ?2",
            params![now as i64, run_id],
        )?;
        tx.commit()?;
        Ok(AttemptLease {
            attempt: Attempt {
                attempt_id,
                run_id: run_id.into(),
                number,
                worker_id: worker_id.into(),
                status: AttemptStatus::Running,
                lease_generation: fencing_token,
                lease_expires_at_ms: expires,
                last_heartbeat_at_ms: now,
                started_at_ms: now,
                finished_at_ms: None,
                error: None,
            },
            fencing_token,
        })
    }

    pub fn renew_attempt(
        &self,
        attempt_id: &str,
        fencing_token: u64,
        now: u64,
        lease_ttl_ms: u64,
    ) -> Result<AttemptLease> {
        let connection = self.lock()?;
        let expires = now.saturating_add(lease_ttl_ms.max(1));
        let changed = connection.execute(
            "UPDATE control_attempts
             SET lease_expires_at_ms = ?1, last_heartbeat_at_ms = ?2
             WHERE attempt_id = ?3 AND lease_generation = ?4
               AND status IN ('running', 'waiting') AND lease_expires_at_ms >= ?2",
            params![expires as i64, now as i64, attempt_id, fencing_token as i64],
        )?;
        if changed != 1 {
            return Err(ControlError::LeaseLost(attempt_id.into()));
        }
        self.load_attempt_locked(&connection, attempt_id)?
            .ok_or_else(|| ControlError::AttemptNotFound(attempt_id.into()))
            .map(|attempt| AttemptLease {
                attempt,
                fencing_token,
            })
    }

    pub fn complete_attempt(
        &self,
        attempt_id: &str,
        fencing_token: u64,
        status: AttemptStatus,
        now: u64,
        error: Option<&str>,
    ) -> Result<Attempt> {
        if matches!(status, AttemptStatus::Running | AttemptStatus::Waiting) {
            return Err(ControlError::Invalid(
                "completion status must be terminal".into(),
            ));
        }
        let connection = self.lock()?;
        let changed = connection.execute(
            "UPDATE control_attempts
             SET status = ?1, finished_at_ms = ?2, lease_expires_at_ms = 0,
                 error = ?3
             WHERE attempt_id = ?4 AND lease_generation = ?5
               AND status IN ('running', 'waiting') AND lease_expires_at_ms >= ?2",
            params![
                status.as_str(),
                now as i64,
                error,
                attempt_id,
                fencing_token as i64
            ],
        )?;
        if changed != 1 {
            return Err(ControlError::LeaseLost(attempt_id.into()));
        }
        let attempt = self
            .load_attempt_locked(&connection, attempt_id)?
            .ok_or_else(|| ControlError::AttemptNotFound(attempt_id.into()))?;
        let run_status = match status {
            AttemptStatus::Succeeded => RunStatus::Succeeded,
            AttemptStatus::Cancelled => RunStatus::Cancelled,
            AttemptStatus::Failed => RunStatus::Failed,
            AttemptStatus::Lost => RunStatus::Queued,
            AttemptStatus::Running | AttemptStatus::Waiting => unreachable!(),
        };
        connection.execute(
            "UPDATE control_runs SET status = ?1, updated_at_ms = ?2 WHERE run_id = ?3",
            params![run_status.as_str(), now as i64, attempt.run_id],
        )?;
        Ok(attempt)
    }

    pub fn recover_expired_attempts(&self, now: u64) -> Result<Vec<String>> {
        let mut connection = self.lock()?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut statement = tx.prepare(
            "SELECT attempt_id, run_id FROM control_attempts
             WHERE status IN ('running', 'waiting') AND lease_expires_at_ms < ?1",
        )?;
        let expired = statement
            .query_map([now as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(statement);
        for (attempt_id, run_id) in &expired {
            tx.execute(
                "UPDATE control_attempts SET status = 'lost', finished_at_ms = ?1,
                 lease_expires_at_ms = 0, error = 'lease expired'
                 WHERE attempt_id = ?2 AND status IN ('running', 'waiting')",
                params![now as i64, attempt_id],
            )?;
            tx.execute(
                "UPDATE control_effects SET status = 'unknown', updated_at_ms = ?1,
                 error = COALESCE(error, 'attempt lease expired while effect was active')
                 WHERE attempt_id = ?2 AND status IN ('planned', 'started')",
                params![now as i64, attempt_id],
            )?;
            tx.execute(
                "UPDATE control_runs SET status = 'queued', updated_at_ms = ?1
                 WHERE run_id = ?2 AND status = 'running'",
                params![now as i64, run_id],
            )?;
        }
        tx.commit()?;
        Ok(expired.into_iter().map(|(attempt, _)| attempt).collect())
    }

    pub fn record_effect(&self, effect: NewEffect, now: u64) -> Result<EffectRecord> {
        validate_id(&effect.idempotency_key, "idempotency_key")?;
        let connection = self.lock()?;
        let attempt = self
            .load_attempt_locked(&connection, &effect.attempt_id)?
            .ok_or_else(|| ControlError::AttemptNotFound(effect.attempt_id.clone()))?;
        if attempt.run_id != effect.run_id
            || attempt.lease_generation != effect.fencing_token
            || !matches!(
                attempt.status,
                AttemptStatus::Running | AttemptStatus::Waiting
            )
            || attempt.lease_expires_at_ms < now
        {
            return Err(ControlError::LeaseLost(effect.attempt_id));
        }
        let effect_id = effect
            .effect_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let record = EffectRecord {
            effect_id,
            run_id: effect.run_id,
            attempt_id: effect.attempt_id,
            fencing_token: effect.fencing_token,
            capability: effect.capability,
            tool: effect.tool,
            operation_digest: effect.operation_digest,
            idempotency_key: effect.idempotency_key,
            scope: effect.scope,
            provenance: effect.provenance,
            status: EffectStatus::Planned,
            input_ref: effect.input_ref,
            output_ref: None,
            mutation_ref: None,
            external_ref: None,
            error: None,
            created_at_ms: now,
            updated_at_ms: now,
        };
        if let Some(existing_id) = connection
            .query_row(
                "SELECT effect_id FROM control_effects WHERE idempotency_key = ?1",
                [&record.idempotency_key],
                |row| row.get::<_, String>(0),
            )
            .optional()?
        {
            let existing = self
                .load_effect_locked(&connection, &existing_id)?
                .ok_or_else(|| ControlError::EffectNotFound(existing_id.clone()))?;
            if existing.run_id != record.run_id
                || existing.attempt_id != record.attempt_id
                || existing.operation_digest != record.operation_digest
            {
                return Err(ControlError::Invalid(
                    "idempotency key was reused for a different effect".into(),
                ));
            }
            return Ok(existing);
        }
        connection.execute(
            "INSERT INTO control_effects
             (effect_id, run_id, attempt_id, fencing_token, capability, tool,
              operation_digest, idempotency_key, scope_json, provenance_json,
              status, input_ref_json, created_at_ms, updated_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 'planned', ?11, ?12, ?13)",
            params![
                record.effect_id,
                record.run_id,
                record.attempt_id,
                record.fencing_token as i64,
                enum_json(&record.capability)?,
                record.tool,
                record.operation_digest.0,
                record.idempotency_key,
                optional_json(&record.scope)?,
                optional_json(&record.provenance)?,
                optional_json(&record.input_ref)?,
                now as i64,
                now as i64,
            ],
        )?;
        Ok(record)
    }

    pub fn start_effect(
        &self,
        effect_id: &str,
        fencing_token: u64,
        now: u64,
    ) -> Result<EffectRecord> {
        let connection = self.lock()?;
        let effect = self
            .load_effect_locked(&connection, effect_id)?
            .ok_or_else(|| ControlError::EffectNotFound(effect_id.into()))?;
        let attempt = self
            .load_attempt_locked(&connection, &effect.attempt_id)?
            .ok_or_else(|| ControlError::AttemptNotFound(effect.attempt_id.clone()))?;
        if attempt.lease_generation != fencing_token
            || attempt.lease_expires_at_ms < now
            || !matches!(
                attempt.status,
                AttemptStatus::Running | AttemptStatus::Waiting
            )
        {
            return Err(ControlError::LeaseLost(effect.attempt_id));
        }
        let changed = connection.execute(
            "UPDATE control_effects SET status = 'started', updated_at_ms = ?1
             WHERE effect_id = ?2 AND fencing_token = ?3 AND status = 'planned'",
            params![now as i64, effect_id, fencing_token as i64],
        )?;
        if changed != 1 {
            return self
                .load_effect_locked(&connection, effect_id)?
                .ok_or_else(|| ControlError::EffectNotFound(effect_id.into()));
        }
        self.load_effect_locked(&connection, effect_id)?
            .ok_or_else(|| ControlError::EffectNotFound(effect_id.into()))
    }

    pub fn complete_effect(
        &self,
        effect_id: &str,
        fencing_token: u64,
        outcome: EffectOutcome,
        now: u64,
    ) -> Result<EffectRecord> {
        let connection = self.lock()?;
        let effect = self
            .load_effect_locked(&connection, effect_id)?
            .ok_or_else(|| ControlError::EffectNotFound(effect_id.into()))?;
        let attempt = self
            .load_attempt_locked(&connection, &effect.attempt_id)?
            .ok_or_else(|| ControlError::AttemptNotFound(effect.attempt_id.clone()))?;
        if attempt.lease_generation != fencing_token
            || attempt.lease_expires_at_ms < now
            || !matches!(
                attempt.status,
                AttemptStatus::Running | AttemptStatus::Waiting
            )
        {
            return Err(ControlError::LeaseLost(effect.attempt_id));
        }
        connection.execute(
            "UPDATE control_effects
             SET status = ?1, output_ref_json = ?2, mutation_ref_json = ?3,
                 external_ref = ?4, error = ?5, updated_at_ms = ?6
             WHERE effect_id = ?7 AND fencing_token = ?8",
            params![
                outcome.status.as_str(),
                optional_json(&outcome.output_ref)?,
                optional_json(&outcome.mutation_ref)?,
                outcome.external_ref,
                outcome.error,
                now as i64,
                effect_id,
                fencing_token as i64,
            ],
        )?;
        self.load_effect_locked(&connection, effect_id)?
            .ok_or_else(|| ControlError::EffectNotFound(effect_id.into()))
    }

    pub fn reconcile_effect(
        &self,
        effect_id: &str,
        outcome: EffectOutcome,
        now: u64,
    ) -> Result<EffectRecord> {
        if !matches!(
            outcome.status,
            EffectStatus::Succeeded | EffectStatus::Failed | EffectStatus::Reconciled
        ) {
            return Err(ControlError::Invalid(
                "reconciliation must resolve an effect".into(),
            ));
        }
        let connection = self.lock()?;
        let changed = connection.execute(
            "UPDATE control_effects
             SET status = 'reconciled', output_ref_json = ?1, mutation_ref_json = ?2,
                 external_ref = ?3, error = ?4, updated_at_ms = ?5
             WHERE effect_id = ?6 AND status = 'unknown'",
            params![
                optional_json(&outcome.output_ref)?,
                optional_json(&outcome.mutation_ref)?,
                outcome.external_ref,
                outcome.error,
                now as i64,
                effect_id,
            ],
        )?;
        if changed != 1 {
            return Err(ControlError::EffectNotFound(effect_id.into()));
        }
        self.load_effect_locked(&connection, effect_id)?
            .ok_or_else(|| ControlError::EffectNotFound(effect_id.into()))
    }

    pub fn record_audit(&self, record: &AuditRecord) -> Result<()> {
        let connection = self.lock()?;
        connection.execute(
            "INSERT INTO control_audit
             (audit_id, decision, actor_json, source, scope_json, provenance_json,
              policy_json, operation_digest, approval_id, route_json, effect_id,
              event_id, detail, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                record.audit_id,
                record.decision,
                serde_json::to_string(&record.actor)?,
                record.source,
                optional_json(&record.scope)?,
                optional_json(&record.provenance)?,
                optional_json(&record.policy_snapshot)?,
                record.operation_digest.as_ref().map(|value| &value.0),
                record.approval_id,
                optional_json(&record.route)?,
                record.effect_id,
                record.event_id,
                record.detail,
                record.created_at_ms as i64,
            ],
        )?;
        Ok(())
    }

    pub fn register_scope(&self, scope: &ScopeRef) -> Result<()> {
        validate_id(&scope.scope_id, "scope_id")?;
        let connection = self.lock()?;
        if let Some(parent) = &scope.parent_scope_id {
            let exists: Option<String> = connection
                .query_row(
                    "SELECT scope_id FROM control_scopes WHERE scope_id = ?1",
                    [parent],
                    |row| row.get(0),
                )
                .optional()?;
            if exists.is_none() {
                return Err(ControlError::Invalid(format!(
                    "scope parent does not exist: {parent}"
                )));
            }
        }
        connection.execute(
            "INSERT INTO control_scopes
             (scope_id, kind, parent_scope_id, workspace_id, owner_id, inherited, expires_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                scope.scope_id,
                serde_json::to_string(&scope.kind)?,
                scope.parent_scope_id,
                scope.workspace_id,
                scope.owner_id,
                scope.inherited as i64,
                scope.expires_at_ms.map(|value| value as i64),
            ],
        )?;
        Ok(())
    }

    pub fn register_resource(&self, resource: &ScopedResource) -> Result<()> {
        validate_id(&resource.resource_id, "resource_id")?;
        let connection = self.lock()?;
        connection.execute(
            "INSERT INTO control_resources
             (resource_id, kind, scope_id, owner_id, digest, sensitivity,
              provenance_json, expires_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                resource.resource_id,
                serde_json::to_string(&resource.kind)?,
                resource.scope.scope_id,
                resource.owner_id,
                resource.digest,
                resource.sensitivity,
                optional_json(&resource.provenance)?,
                resource.expires_at_ms.map(|value| value as i64),
            ],
        )?;
        Ok(())
    }

    /// Register metadata for an existing content-addressed artifact. The
    /// bytes stay in the owning CAS/trace/replay store; this row only binds
    /// the hash to scope and producer provenance.
    pub fn register_artifact(&self, artifact: &ArtifactRef, now: u64) -> Result<()> {
        let connection = self.lock()?;
        connection.execute(
            "INSERT OR IGNORE INTO control_artifacts
             (artifact_hash, media_type, size_bytes, scope_id, sensitivity,
              producer_run_id, producer_attempt_id, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                artifact.hash,
                artifact.media_type,
                artifact.size_bytes.map(|value| value as i64),
                artifact.scope_id,
                artifact.sensitivity,
                artifact.producer_run_id,
                artifact.producer_attempt_id,
                now as i64,
            ],
        )?;
        Ok(())
    }

    /// Return whether a resource scope is visible from a consumer scope.
    /// Exact scope access is always allowed for the same owner; ancestor
    /// resources require their explicit `inherited` flag.
    pub fn scope_allows(
        &self,
        resource_scope_id: &str,
        consumer_scope_id: &str,
        owner_id: &str,
        now: u64,
    ) -> Result<bool> {
        let connection = self.lock()?;
        let resource: Option<(String, Option<String>, String, i64, Option<i64>)> = connection
            .query_row(
                "SELECT kind, parent_scope_id, owner_id, inherited, expires_at_ms
                 FROM control_scopes WHERE scope_id = ?1",
                [resource_scope_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .optional()?;
        let Some((_, _, resource_owner, inherited, expires)) = resource else {
            return Ok(false);
        };
        if resource_owner != owner_id || expires.is_some_and(|value| value as u64 <= now) {
            return Ok(false);
        }
        if resource_scope_id == consumer_scope_id {
            return Ok(true);
        }
        if inherited == 0 {
            return Ok(false);
        }
        let mut cursor = Some(consumer_scope_id.to_string());
        while let Some(scope_id) = cursor {
            cursor = connection
                .query_row(
                    "SELECT parent_scope_id FROM control_scopes WHERE scope_id = ?1",
                    [&scope_id],
                    |row| row.get::<_, Option<String>>(0),
                )
                .optional()?
                .flatten();
            if cursor.as_deref() == Some(resource_scope_id) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn append_event(&self, mut event: RunEvent) -> Result<EventReceipt> {
        let mut connection = self.lock()?;
        append_event_locked(&mut connection, &mut event)
    }

    pub fn replay_events(&self, after_sequence: u64) -> Result<Vec<RunEvent>> {
        let connection = self.lock()?;
        let mut statement = connection.prepare(
            "SELECT event_json FROM run_events WHERE sequence > ?1 ORDER BY sequence ASC",
        )?;
        let rows = statement.query_map([after_sequence as i64], |row| row.get::<_, String>(0))?;
        rows.map(|row| {
            let json = row?;
            serde_json::from_str(&json).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    json.len(),
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })
        })
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
    }

    pub fn max_event_sequence(&self) -> Result<u64> {
        let connection = self.lock()?;
        Ok(connection.query_row(
            "SELECT COALESCE(MAX(sequence), 0) FROM run_events",
            [],
            |row| row.get::<_, i64>(0),
        )? as u64)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.connection
            .lock()
            .map_err(|_| ControlError::Invalid("control store lock poisoned".into()))
    }

    fn load_attempt_locked(&self, connection: &Connection, id: &str) -> Result<Option<Attempt>> {
        connection
            .query_row(
                "SELECT attempt_id, run_id, number, worker_id, status, lease_generation,
                        lease_expires_at_ms, last_heartbeat_at_ms, started_at_ms,
                        finished_at_ms, error
                 FROM control_attempts WHERE attempt_id = ?1",
                [id],
                row_attempt,
            )
            .optional()
            .map_err(Into::into)
    }

    fn load_effect_locked(
        &self,
        connection: &Connection,
        id: &str,
    ) -> Result<Option<EffectRecord>> {
        connection
            .query_row(
                "SELECT effect_id, run_id, attempt_id, fencing_token, capability, tool,
                        operation_digest, idempotency_key, scope_json, provenance_json,
                        status, input_ref_json, output_ref_json, mutation_ref_json,
                        external_ref, error, created_at_ms, updated_at_ms
                 FROM control_effects WHERE effect_id = ?1",
                [id],
                row_effect,
            )
            .optional()
            .map_err(Into::into)
    }
}

impl EventSink for ControlStore {
    fn publish(&self, event: RunEvent) -> std::result::Result<EventReceipt, EventError> {
        self.append_event(event)
            .map_err(|error| EventError::Persistence(error.to_string()))
    }
}

impl EventBus for ControlStore {
    fn replay_since(&self, sequence: u64) -> std::result::Result<Vec<RunEvent>, EventError> {
        self.replay_events(sequence)
            .map_err(|error| EventError::Persistence(error.to_string()))
    }
}

impl ApprovalStore for ControlStore {
    fn create(&self, request: CapabilityRequest) -> anyhow::Result<ApprovalRecord> {
        let connection = self
            .lock()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let json = serde_json::to_string(&request)?;
        connection.execute(
            "INSERT INTO approvals
             (approval_id, run_id, request_json, state, created_at_ms, expires_at_ms)
             VALUES (?1, ?2, ?3, 'pending', ?4, ?5)",
            params![
                request.approval_id.0,
                request.run_id,
                json,
                request.created_at_ms as i64,
                request.expires_at_ms as i64,
            ],
        )?;
        Ok(ApprovalRecord {
            request,
            state: ApprovalState::Pending,
            decided_at_ms: None,
            consumed_at_ms: None,
        })
    }

    fn get(&self, id: &hi_policy::ApprovalId) -> anyhow::Result<Option<ApprovalRecord>> {
        let connection = self
            .lock()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let now = hi_policy::now_ms() as i64;
        connection.execute(
            "UPDATE approvals SET state = 'expired'
             WHERE approval_id = ?1 AND state IN ('pending', 'approved') AND expires_at_ms <= ?2",
            params![&id.0, now],
        )?;
        approval_row(&connection, &id.0).map_err(Into::into)
    }

    fn decide(
        &self,
        id: &hi_policy::ApprovalId,
        decision: ApprovalDecision,
    ) -> anyhow::Result<ApprovalRecord> {
        let mut connection = self
            .lock()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut current = approval_row(&tx, &id.0)?
            .ok_or_else(|| anyhow::anyhow!("approval request not found"))?;
        if current.state != ApprovalState::Pending {
            anyhow::bail!("approval is not pending: {:?}", current.state);
        }
        let now = hi_policy::now_ms();
        let next = if now >= current.request.expires_at_ms {
            ApprovalState::Expired
        } else {
            match decision {
                ApprovalDecision::Approved => ApprovalState::Approved,
                ApprovalDecision::Denied
                | ApprovalDecision::Cancelled
                | ApprovalDecision::Unavailable => ApprovalState::Denied,
            }
        };
        tx.execute(
            "UPDATE approvals SET state = ?1, decided_at_ms = ?2 WHERE approval_id = ?3",
            params![approval_state_name(&next), now as i64, id.0],
        )?;
        tx.commit()?;
        current.state = next;
        current.decided_at_ms = Some(now);
        Ok(current)
    }

    fn claim(
        &self,
        id: &hi_policy::ApprovalId,
        digest: &OperationDigest,
    ) -> anyhow::Result<ApprovalRecord> {
        let mut connection = self
            .lock()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut current = approval_row(&tx, &id.0)?
            .ok_or_else(|| anyhow::anyhow!("approval request not found"))?;
        if current.request.operation_digest != *digest {
            anyhow::bail!("approval operation digest mismatch");
        }
        if current.state != ApprovalState::Approved {
            anyhow::bail!("approval is not approved: {:?}", current.state);
        }
        let now = hi_policy::now_ms();
        if now >= current.request.expires_at_ms {
            tx.execute(
                "UPDATE approvals SET state = 'expired' WHERE approval_id = ?1",
                [&id.0],
            )?;
            tx.commit()?;
            anyhow::bail!("approval expired");
        }
        let changed = tx.execute(
            "UPDATE approvals SET state = 'consumed', consumed_at_ms = ?1
             WHERE approval_id = ?2 AND state = 'approved'",
            params![now as i64, id.0],
        )?;
        if changed != 1 {
            anyhow::bail!("approval was consumed concurrently");
        }
        tx.commit()?;
        current.state = ApprovalState::Consumed;
        current.consumed_at_ms = Some(now);
        Ok(current)
    }

    fn abandon_run(&self, run_id: &str) -> anyhow::Result<u64> {
        let connection = self
            .lock()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        Ok(connection.execute(
            "UPDATE approvals SET state = 'abandoned'
             WHERE run_id = ?1 AND state IN ('pending', 'approved')",
            [run_id],
        )? as u64)
    }

    fn abandon_interactive(&self) -> anyhow::Result<u64> {
        let connection = self
            .lock()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        Ok(connection.execute(
            "UPDATE approvals SET state = 'abandoned'
             WHERE run_id IS NULL AND state IN ('pending', 'approved')",
            [],
        )? as u64)
    }

    fn pending(&self) -> anyhow::Result<Vec<ApprovalRecord>> {
        let connection = self
            .lock()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let now = hi_policy::now_ms() as i64;
        connection.execute(
            "UPDATE approvals SET state = 'expired'
             WHERE state IN ('pending', 'approved') AND expires_at_ms <= ?1",
            [now],
        )?;
        let mut statement = connection.prepare(
            "SELECT request_json, state, decided_at_ms, consumed_at_ms
             FROM approvals WHERE state IN ('pending', 'approved') ORDER BY created_at_ms ASC",
        )?;
        let rows = statement.query_map([], approval_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }
}

fn approval_row(connection: &Connection, id: &str) -> rusqlite::Result<Option<ApprovalRecord>> {
    connection
        .query_row(
            "SELECT request_json, state, decided_at_ms, consumed_at_ms
             FROM approvals WHERE approval_id = ?1",
            [id],
            approval_from_row,
        )
        .optional()
}

fn approval_from_row(row: &Row<'_>) -> rusqlite::Result<ApprovalRecord> {
    let request_json: String = row.get(0)?;
    let request: CapabilityRequest = serde_json::from_str(&request_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            request_json.len(),
            rusqlite::types::Type::Text,
            Box::new(error),
        )
    })?;
    let state = match row.get::<_, String>(1)?.as_str() {
        "pending" => ApprovalState::Pending,
        "approved" => ApprovalState::Approved,
        "denied" => ApprovalState::Denied,
        "expired" => ApprovalState::Expired,
        "consumed" => ApprovalState::Consumed,
        "abandoned" => ApprovalState::Abandoned,
        other => {
            return Err(rusqlite::Error::InvalidParameterName(format!(
                "unknown approval state {other}"
            )));
        }
    };
    Ok(ApprovalRecord {
        request,
        state,
        decided_at_ms: row.get::<_, Option<i64>>(2)?.map(|value| value as u64),
        consumed_at_ms: row.get::<_, Option<i64>>(3)?.map(|value| value as u64),
    })
}

fn approval_state_name(state: &ApprovalState) -> &'static str {
    match state {
        ApprovalState::Pending => "pending",
        ApprovalState::Approved => "approved",
        ApprovalState::Denied => "denied",
        ApprovalState::Expired => "expired",
        ApprovalState::Consumed => "consumed",
        ApprovalState::Abandoned => "abandoned",
    }
}

fn migrate(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS control_meta (
           key TEXT PRIMARY KEY,
           value TEXT NOT NULL
         );
         INSERT OR IGNORE INTO control_meta(key, value)
           VALUES ('schema_version', '1');
         CREATE TABLE IF NOT EXISTS control_runs (
           run_id TEXT PRIMARY KEY,
           kind TEXT NOT NULL,
           workspace_id TEXT,
           scope_json TEXT,
           session_id TEXT,
           parent_run_id TEXT,
           status TEXT NOT NULL,
           desired_state TEXT NOT NULL,
           policy_json TEXT,
           route_json TEXT,
           provenance_json TEXT,
           created_at_ms INTEGER NOT NULL,
           updated_at_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS control_runs_status ON control_runs(status, updated_at_ms);
         CREATE TABLE IF NOT EXISTS control_attempts (
           attempt_id TEXT PRIMARY KEY,
           run_id TEXT NOT NULL REFERENCES control_runs(run_id),
           number INTEGER NOT NULL,
           worker_id TEXT NOT NULL,
           status TEXT NOT NULL,
           lease_generation INTEGER NOT NULL,
           lease_expires_at_ms INTEGER NOT NULL,
           last_heartbeat_at_ms INTEGER NOT NULL,
           started_at_ms INTEGER NOT NULL,
           finished_at_ms INTEGER,
           error TEXT,
           UNIQUE(run_id, number)
         );
         CREATE INDEX IF NOT EXISTS control_attempts_lease
           ON control_attempts(status, lease_expires_at_ms);
         CREATE TABLE IF NOT EXISTS control_effects (
           effect_id TEXT PRIMARY KEY,
           run_id TEXT NOT NULL REFERENCES control_runs(run_id),
           attempt_id TEXT NOT NULL REFERENCES control_attempts(attempt_id),
           fencing_token INTEGER NOT NULL,
           capability TEXT NOT NULL,
           tool TEXT NOT NULL,
           operation_digest TEXT NOT NULL,
           idempotency_key TEXT NOT NULL UNIQUE,
           scope_json TEXT,
           provenance_json TEXT,
           status TEXT NOT NULL,
           input_ref_json TEXT,
           output_ref_json TEXT,
           mutation_ref_json TEXT,
           external_ref TEXT,
           error TEXT,
           created_at_ms INTEGER NOT NULL,
           updated_at_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS control_effects_attempt
           ON control_effects(attempt_id, created_at_ms);
         CREATE TABLE IF NOT EXISTS control_audit (
           audit_id TEXT PRIMARY KEY,
           decision TEXT NOT NULL,
           actor_json TEXT NOT NULL,
           source TEXT NOT NULL,
           scope_json TEXT,
           provenance_json TEXT,
           policy_json TEXT,
           operation_digest TEXT,
           approval_id TEXT,
           route_json TEXT,
           effect_id TEXT,
           event_id TEXT,
           detail TEXT,
           created_at_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS control_audit_created ON control_audit(created_at_ms);
         CREATE TABLE IF NOT EXISTS control_scopes (
           scope_id TEXT PRIMARY KEY,
           kind TEXT NOT NULL,
           parent_scope_id TEXT,
           workspace_id TEXT,
           owner_id TEXT NOT NULL,
           inherited INTEGER NOT NULL,
           expires_at_ms INTEGER
         );
         CREATE INDEX IF NOT EXISTS control_scopes_parent ON control_scopes(parent_scope_id);
         CREATE TABLE IF NOT EXISTS control_resources (
           resource_id TEXT PRIMARY KEY,
           kind TEXT NOT NULL,
           scope_id TEXT NOT NULL,
           owner_id TEXT NOT NULL,
           digest TEXT NOT NULL,
           sensitivity TEXT NOT NULL,
           provenance_json TEXT,
           expires_at_ms INTEGER
         );
         CREATE INDEX IF NOT EXISTS control_resources_scope ON control_resources(scope_id, kind);
         CREATE TABLE IF NOT EXISTS control_artifacts (
           artifact_hash TEXT PRIMARY KEY,
           media_type TEXT NOT NULL,
           size_bytes INTEGER,
           scope_id TEXT,
           sensitivity TEXT NOT NULL,
           producer_run_id TEXT,
           producer_attempt_id TEXT,
           created_at_ms INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS control_artifacts_scope ON control_artifacts(scope_id, sensitivity);
         CREATE TABLE IF NOT EXISTS approvals (
           approval_id TEXT PRIMARY KEY,
           run_id TEXT,
           request_json TEXT NOT NULL,
           state TEXT NOT NULL,
           created_at_ms INTEGER NOT NULL,
           expires_at_ms INTEGER NOT NULL,
           decided_at_ms INTEGER,
           consumed_at_ms INTEGER
         );
         CREATE INDEX IF NOT EXISTS approvals_pending ON approvals(state, expires_at_ms);
         CREATE INDEX IF NOT EXISTS approvals_run ON approvals(run_id, state);
         CREATE TABLE IF NOT EXISTS run_events (
           sequence INTEGER PRIMARY KEY,
           event_id TEXT NOT NULL UNIQUE,
           occurred_at_ms INTEGER NOT NULL,
           event_json TEXT NOT NULL,
           event_bytes INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS run_events_event_id ON run_events(event_id);",
    )?;
    Ok(())
}

fn append_event_locked(connection: &mut Connection, event: &mut RunEvent) -> Result<EventReceipt> {
    if event.schema_version != hi_events::EVENT_SCHEMA_VERSION {
        return Err(ControlError::Invalid(format!(
            "unsupported event schema version {}",
            event.schema_version
        )));
    }
    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let existing: Option<i64> = tx
        .query_row(
            "SELECT sequence FROM run_events WHERE event_id = ?1",
            [&event.event_id],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(sequence) = existing {
        tx.commit()?;
        event.sequence = sequence as u64;
        return Ok(EventReceipt {
            event_id: event.event_id.clone(),
            sequence: sequence as u64,
        });
    }
    let sequence = tx.query_row(
        "SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_events",
        [],
        |row| row.get::<_, i64>(0),
    )?;
    event.sequence = sequence as u64;
    let json = serde_json::to_string(event)?;
    tx.execute(
        "INSERT INTO run_events(sequence, event_id, occurred_at_ms, event_json, event_bytes)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            sequence,
            event.event_id,
            event.occurred_at_ms as i64,
            json,
            json.len() as i64
        ],
    )?;
    tx.commit()?;
    Ok(EventReceipt {
        event_id: event.event_id.clone(),
        sequence: sequence as u64,
    })
}

fn row_run(row: &Row<'_>) -> rusqlite::Result<RunRecord> {
    Ok(RunRecord {
        run_id: row.get(0)?,
        kind: serde_json::from_str(&row.get::<_, String>(1)?)
            .map_err(|error| to_sql_error(ControlError::Serialization(error)))?,
        workspace_id: row.get(2)?,
        scope: parse_optional(row.get(3)?)?,
        session_id: row.get(4)?,
        parent_run_id: row.get(5)?,
        status: RunStatus::from_str(&row.get::<_, String>(6)?).map_err(to_sql_error)?,
        desired_state: match row.get::<_, String>(7)?.as_str() {
            "run" => DesiredState::Run,
            "pause" => DesiredState::Pause,
            "cancel" => DesiredState::Cancel,
            other => {
                return Err(to_sql_error(ControlError::Invalid(format!(
                    "unknown desired state {other}"
                ))));
            }
        },
        policy_snapshot: parse_optional(row.get(8)?)?,
        route_snapshot: parse_optional(row.get(9)?)?,
        provenance: parse_optional(row.get(10)?)?,
        created_at_ms: row.get::<_, i64>(11)? as u64,
        updated_at_ms: row.get::<_, i64>(12)? as u64,
    })
}

fn row_attempt(row: &Row<'_>) -> rusqlite::Result<Attempt> {
    Ok(Attempt {
        attempt_id: row.get(0)?,
        run_id: row.get(1)?,
        number: row.get::<_, i64>(2)? as u32,
        worker_id: row.get(3)?,
        status: match row.get::<_, String>(4)?.as_str() {
            "running" => AttemptStatus::Running,
            "waiting" => AttemptStatus::Waiting,
            "succeeded" => AttemptStatus::Succeeded,
            "failed" => AttemptStatus::Failed,
            "lost" => AttemptStatus::Lost,
            "cancelled" => AttemptStatus::Cancelled,
            other => {
                return Err(to_sql_error(ControlError::Invalid(format!(
                    "unknown attempt status {other}"
                ))));
            }
        },
        lease_generation: row.get::<_, i64>(5)? as u64,
        lease_expires_at_ms: row.get::<_, i64>(6)? as u64,
        last_heartbeat_at_ms: row.get::<_, i64>(7)? as u64,
        started_at_ms: row.get::<_, i64>(8)? as u64,
        finished_at_ms: row.get::<_, Option<i64>>(9)?.map(|value| value as u64),
        error: row.get(10)?,
    })
}

fn row_effect(row: &Row<'_>) -> rusqlite::Result<EffectRecord> {
    Ok(EffectRecord {
        effect_id: row.get(0)?,
        run_id: row.get(1)?,
        attempt_id: row.get(2)?,
        fencing_token: row.get::<_, i64>(3)? as u64,
        capability: serde_json::from_str(&row.get::<_, String>(4)?)
            .map_err(|error| to_sql_error(ControlError::Serialization(error)))?,
        tool: row.get(5)?,
        operation_digest: OperationDigest(row.get(6)?),
        idempotency_key: row.get(7)?,
        scope: parse_optional(row.get(8)?)?,
        provenance: parse_optional(row.get(9)?)?,
        status: effect_status(&row.get::<_, String>(10)?)?,
        input_ref: parse_optional(row.get(11)?)?,
        output_ref: parse_optional(row.get(12)?)?,
        mutation_ref: parse_optional(row.get(13)?)?,
        external_ref: row.get(14)?,
        error: row.get(15)?,
        created_at_ms: row.get::<_, i64>(16)? as u64,
        updated_at_ms: row.get::<_, i64>(17)? as u64,
    })
}

fn effect_status(value: &str) -> rusqlite::Result<EffectStatus> {
    match value {
        "planned" => Ok(EffectStatus::Planned),
        "started" => Ok(EffectStatus::Started),
        "succeeded" => Ok(EffectStatus::Succeeded),
        "failed" => Ok(EffectStatus::Failed),
        "denied" => Ok(EffectStatus::Denied),
        "unknown" => Ok(EffectStatus::Unknown),
        "reconciled" => Ok(EffectStatus::Reconciled),
        other => Err(to_sql_error(ControlError::Invalid(format!(
            "unknown effect status {other}"
        )))),
    }
}

fn enum_json<T: Serialize>(value: &T) -> Result<String> {
    Ok(serde_json::to_string(value)?)
}

fn optional_json<T: Serialize>(value: &Option<T>) -> Result<Option<String>> {
    value
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(Into::into)
}

fn parse_optional<T: for<'de> Deserialize<'de>>(
    value: Option<String>,
) -> rusqlite::Result<Option<T>> {
    value
        .map(|json| {
            serde_json::from_str(&json)
                .map_err(|error| to_sql_error(ControlError::Serialization(error)))
        })
        .transpose()
}

fn to_sql_error(error: ControlError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
}

fn validate_id(value: &str, label: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(ControlError::Invalid(format!("invalid {label}")));
    }
    Ok(())
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_events::{
        ActivityObject, ActivityState, ActivityVerb, EventContext, EventKind, SemanticActivity,
    };

    fn store() -> ControlStore {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.sqlite3");
        let store = ControlStore::open(path).unwrap();
        std::mem::forget(dir);
        store
    }

    fn run(store: &ControlStore) -> RunRecord {
        store
            .create_run(NewRun {
                run_id: None,
                kind: RunKind::Interactive,
                workspace_id: Some("workspace".into()),
                scope: None,
                session_id: Some("session".into()),
                parent_run_id: None,
                policy_snapshot: None,
                route_snapshot: None,
                provenance: None,
                desired_state: DesiredState::Run,
            })
            .unwrap()
    }

    #[test]
    fn claim_renew_and_fence_attempts() {
        let store = store();
        let run = run(&store);
        let lease = store.claim_attempt(&run.run_id, "worker", 100, 30).unwrap();
        assert!(
            store
                .renew_attempt(&lease.attempt.attempt_id, 1, 110, 30)
                .is_ok()
        );
        assert!(matches!(
            store.complete_attempt(
                &lease.attempt.attempt_id,
                2,
                AttemptStatus::Succeeded,
                120,
                None
            ),
            Err(ControlError::LeaseLost(_))
        ));
        store
            .complete_attempt(
                &lease.attempt.attempt_id,
                1,
                AttemptStatus::Succeeded,
                120,
                None,
            )
            .unwrap();
        assert!(matches!(
            store.claim_attempt(&run.run_id, "worker-2", 130, 30),
            Err(ControlError::RunTerminal(_))
        ));
    }

    #[test]
    fn expired_attempts_are_recoverable() {
        let store = store();
        let run = run(&store);
        let lease = store.claim_attempt(&run.run_id, "worker", 100, 10).unwrap();
        assert_eq!(
            store.recover_expired_attempts(111).unwrap(),
            vec![lease.attempt.attempt_id]
        );
        let next = store
            .claim_attempt(&run.run_id, "worker-2", 112, 10)
            .unwrap();
        assert_eq!(next.attempt.number, 2);
        assert_eq!(next.fencing_token, 2);
    }

    #[test]
    fn events_are_idempotent_and_not_trimmed() {
        let store = store();
        let event = RunEvent::new(
            EventKind::RunStarted,
            EventContext::default(),
            SemanticActivity {
                verb: ActivityVerb::Start,
                object: ActivityObject::Run,
                state: ActivityState::Running,
                group_key: "run".into(),
                title: "started".into(),
                detail: None,
                refs: vec![],
                progress: None,
            },
        );
        let first = store.append_event(event.clone()).unwrap();
        let second = store.append_event(event.clone()).unwrap();
        assert_eq!(first, second);
        assert_eq!(store.replay_events(0).unwrap().len(), 1);
        assert_eq!(event.sequence, 0);
    }

    #[test]
    fn stale_effects_are_rejected() {
        let store = store();
        let run = run(&store);
        let lease = store.claim_attempt(&run.run_id, "worker", 100, 30).unwrap();
        let effect = store.record_effect(
            NewEffect {
                effect_id: None,
                run_id: run.run_id,
                attempt_id: lease.attempt.attempt_id,
                fencing_token: 2,
                capability: CapabilityKind::WorkspaceWrite,
                tool: "edit".into(),
                operation_digest: OperationDigest("digest".into()),
                idempotency_key: "effect-1".into(),
                scope: None,
                provenance: None,
                input_ref: None,
            },
            110,
        );
        assert!(matches!(effect, Err(ControlError::LeaseLost(_))));
    }

    #[test]
    fn expired_attempts_fence_active_effects_as_unknown() {
        let store = store();
        let run = run(&store);
        let lease = store.claim_attempt(&run.run_id, "worker", 100, 10).unwrap();
        let effect = store
            .record_effect(
                NewEffect {
                    effect_id: None,
                    run_id: run.run_id,
                    attempt_id: lease.attempt.attempt_id.clone(),
                    fencing_token: lease.fencing_token,
                    capability: CapabilityKind::ProcessExecution,
                    tool: "bash".into(),
                    operation_digest: OperationDigest("digest".into()),
                    idempotency_key: "effect-unknown".into(),
                    scope: None,
                    provenance: None,
                    input_ref: None,
                },
                105,
            )
            .unwrap();
        store
            .start_effect(&effect.effect_id, lease.fencing_token, 106)
            .unwrap();
        store.recover_expired_attempts(111).unwrap();
        let result = store.complete_effect(
            &effect.effect_id,
            lease.fencing_token,
            EffectOutcome {
                status: EffectStatus::Succeeded,
                output_ref: None,
                mutation_ref: None,
                external_ref: None,
                error: None,
            },
            112,
        );
        assert!(matches!(result, Err(ControlError::LeaseLost(_))));
    }

    #[test]
    fn approvals_are_one_shot_in_the_shared_store() {
        let store = store();
        let request = hi_policy::approval_request(
            CapabilityKind::WorkspaceWrite,
            hi_policy::ResourceScope::Operation {
                workspace_id: "workspace".into(),
                label: "edit".into(),
            },
            OperationDigest("approval-digest".into()),
            "edit",
            None,
            Some("session".into()),
            "edit file",
            "redacted",
        );
        let id = request.approval_id.clone();
        store.create(request).unwrap();
        store.decide(&id, ApprovalDecision::Approved).unwrap();
        let claimed = store
            .claim(&id, &OperationDigest("approval-digest".into()))
            .unwrap();
        assert_eq!(claimed.state, ApprovalState::Consumed);
        assert!(
            store
                .claim(&id, &OperationDigest("approval-digest".into()))
                .is_err()
        );
    }

    #[test]
    fn inherited_scope_requires_same_owner_and_explicit_grant() {
        let store = store();
        store
            .register_scope(&ScopeRef {
                scope_id: "workspace-scope".into(),
                kind: ScopeKind::Workspace,
                parent_scope_id: None,
                workspace_id: Some("workspace".into()),
                owner_id: "user".into(),
                inherited: true,
                expires_at_ms: None,
            })
            .unwrap();
        store
            .register_scope(&ScopeRef {
                scope_id: "run-scope".into(),
                kind: ScopeKind::Run,
                parent_scope_id: Some("workspace-scope".into()),
                workspace_id: Some("workspace".into()),
                owner_id: "user".into(),
                inherited: false,
                expires_at_ms: None,
            })
            .unwrap();
        assert!(
            store
                .scope_allows("workspace-scope", "run-scope", "user", now_ms())
                .unwrap()
        );
        assert!(
            !store
                .scope_allows("workspace-scope", "run-scope", "other", now_ms())
                .unwrap()
        );
    }
}
