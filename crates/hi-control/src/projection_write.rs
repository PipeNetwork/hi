use rusqlite::{OptionalExtension, Transaction, params};
use serde::{Serialize, de::DeserializeOwned};

use super::{
    ControlJobRecord, ProjectionTransition, SessionSnapshotRecord, WorkspaceBindingRecord,
    WorkspaceOperationRecord, WorkspaceRecoveryRecord, parse_record,
};
use crate::{ControlError, Result};

pub(super) fn apply_transition(
    tx: &Transaction<'_>,
    transition: &ProjectionTransition,
    event_id: &str,
) -> Result<()> {
    match transition {
        ProjectionTransition::WorkspaceBinding(record) => apply_binding(tx, record, event_id),
        ProjectionTransition::WorkspaceOperation(record) => apply_operation(tx, record, event_id),
        ProjectionTransition::Job(record) => apply_job(tx, record, event_id),
        ProjectionTransition::WorkspaceRecovery(record) => apply_recovery(tx, record, event_id),
        ProjectionTransition::SessionSnapshot(record) => apply_snapshot(tx, record, event_id),
    }
}

fn apply_binding(
    tx: &Transaction<'_>,
    record: &WorkspaceBindingRecord,
    event_id: &str,
) -> Result<()> {
    check_revision::<WorkspaceBindingRecord>(
        tx,
        "control_workspace_bindings",
        "binding_id",
        &record.binding_id,
        record.revision,
        |old| {
            old.workspace_id == record.workspace_id
                && old.session_id == record.session_id
                && old.epoch == record.epoch
                && old.authority == record.authority
                && old.opened_at_ms == record.opened_at_ms
        },
    )?;
    let json = serde_json::to_string(record)?;
    let authority = enum_name(&record.authority)?;
    let state = enum_name(&record.state)?;
    if record.revision == 1 {
        tx.execute(
            "INSERT INTO control_workspace_bindings
             (binding_id, workspace_id, session_id, epoch, authority, state, workspace_version,
              revision, record_json, last_event_id, opened_at_ms, updated_at_ms, closed_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                record.binding_id,
                record.workspace_id,
                record.session_id,
                record.epoch as i64,
                authority,
                state,
                record.workspace_version,
                record.revision as i64,
                json,
                event_id,
                record.opened_at_ms as i64,
                record.updated_at_ms as i64,
                record.closed_at_ms.map(|v| v as i64)
            ],
        )?;
    } else {
        tx.execute(
            "UPDATE control_workspace_bindings SET state=?1, workspace_version=?2, revision=?3,
             record_json=?4, last_event_id=?5, updated_at_ms=?6, closed_at_ms=?7 WHERE binding_id=?8",
            params![state, record.workspace_version, record.revision as i64, json, event_id,
                record.updated_at_ms as i64, record.closed_at_ms.map(|v| v as i64), record.binding_id],
        )?;
    }
    Ok(())
}

fn apply_operation(
    tx: &Transaction<'_>,
    record: &WorkspaceOperationRecord,
    event_id: &str,
) -> Result<()> {
    check_revision::<WorkspaceOperationRecord>(
        tx,
        "control_workspace_operations",
        "operation_id",
        &record.operation_id,
        record.revision,
        |old| {
            old.binding_id == record.binding_id
                && old.epoch == record.epoch
                && old.idempotency_key == record.idempotency_key
                && old.operation_digest == record.operation_digest
                && old.created_at_ms == record.created_at_ms
        },
    )?;
    let json = serde_json::to_string(record)?;
    let replay = enum_name(&record.replay_class)?;
    let status = enum_name(&record.status)?;
    if record.revision == 1 {
        tx.execute(
            "INSERT INTO control_workspace_operations
             (operation_id,binding_id,epoch,session_id,run_id,attempt_id,job_id,kind,replay_class,
              status,operation_digest,idempotency_key,base_version,result_version,revision,record_json,
              last_event_id,created_at_ms,updated_at_ms,settled_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20)",
            params![record.operation_id,record.binding_id,record.epoch as i64,record.session_id,
                record.run_id,record.attempt_id,record.job_id,record.kind,replay,status,
                record.operation_digest,record.idempotency_key,record.base_version,record.result_version,
                record.revision as i64,json,event_id,record.created_at_ms as i64,
                record.updated_at_ms as i64,record.settled_at_ms.map(|v| v as i64)],
        )?;
    } else {
        tx.execute(
            "UPDATE control_workspace_operations SET status=?1,result_version=?2,revision=?3,
             record_json=?4,last_event_id=?5,updated_at_ms=?6,settled_at_ms=?7 WHERE operation_id=?8",
            params![status,record.result_version,record.revision as i64,json,event_id,
                record.updated_at_ms as i64,record.settled_at_ms.map(|v| v as i64),record.operation_id],
        )?;
    }
    Ok(())
}

fn apply_job(tx: &Transaction<'_>, record: &ControlJobRecord, event_id: &str) -> Result<()> {
    check_revision::<ControlJobRecord>(
        tx,
        "control_jobs",
        "job_id",
        &record.job_id,
        record.revision,
        |old| {
            old.binding_id == record.binding_id
                && old.epoch == record.epoch
                && old.kind == record.kind
                && old.effect_scope == record.effect_scope
                && old.idempotency_key == record.idempotency_key
                && old.created_at_ms == record.created_at_ms
                && !old.state.is_terminal()
        },
    )?;
    let json = serde_json::to_string(record)?;
    let kind = enum_name(&record.kind)?;
    let scope = enum_name(&record.effect_scope)?;
    let state = enum_name(&record.state)?;
    if record.revision == 1 {
        tx.execute(
            "INSERT INTO control_jobs
             (job_id,session_id,run_id,attempt_id,binding_id,epoch,kind,effect_scope,state,
              application_state,operation_digest,idempotency_key,workspace_version,revision,record_json,
              last_event_id,created_at_ms,updated_at_ms,cancel_requested_at_ms,finished_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20)",
            params![record.job_id,record.session_id,record.run_id,record.attempt_id,record.binding_id,
                record.epoch.map(|v| v as i64),kind,scope,state,record.application_state,
                record.operation_digest,record.idempotency_key,record.workspace_version,
                record.revision as i64,json,event_id,record.created_at_ms as i64,
                record.updated_at_ms as i64,record.cancel_requested_at_ms.map(|v| v as i64),
                record.finished_at_ms.map(|v| v as i64)],
        )?;
    } else {
        tx.execute(
            "UPDATE control_jobs SET state=?1,application_state=?2,workspace_version=?3,revision=?4,
             record_json=?5,last_event_id=?6,updated_at_ms=?7,cancel_requested_at_ms=?8,
             finished_at_ms=?9 WHERE job_id=?10",
            params![state,record.application_state,record.workspace_version,record.revision as i64,json,
                event_id,record.updated_at_ms as i64,record.cancel_requested_at_ms.map(|v| v as i64),
                record.finished_at_ms.map(|v| v as i64),record.job_id],
        )?;
    }
    Ok(())
}

fn apply_recovery(
    tx: &Transaction<'_>,
    record: &WorkspaceRecoveryRecord,
    event_id: &str,
) -> Result<()> {
    check_revision::<WorkspaceRecoveryRecord>(
        tx,
        "control_workspace_recoveries",
        "recovery_id",
        &record.recovery_id,
        record.revision,
        |old| {
            old.binding_id == record.binding_id
                && old.workspace_id == record.workspace_id
                && old.operation_id == record.operation_id
                && old.job_id == record.job_id
                && old.kind == record.kind
                && old.created_at_ms == record.created_at_ms
        },
    )?;
    let json = serde_json::to_string(record)?;
    let status = enum_name(&record.status)?;
    if record.revision == 1 {
        tx.execute(
            "INSERT INTO control_workspace_recoveries
             (recovery_id,binding_id,workspace_id,session_id,operation_id,job_id,kind,status,digest,
              artifact_ref,revision,record_json,last_event_id,created_at_ms,updated_at_ms,resolved_at_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
            params![record.recovery_id,record.binding_id,record.workspace_id,record.session_id,
                record.operation_id,record.job_id,record.kind,status,record.digest,record.artifact_ref,
                record.revision as i64,json,event_id,record.created_at_ms as i64,
                record.updated_at_ms as i64,record.resolved_at_ms.map(|v| v as i64)],
        )?;
    } else {
        tx.execute(
            "UPDATE control_workspace_recoveries SET status=?1,digest=?2,artifact_ref=?3,revision=?4,
             record_json=?5,last_event_id=?6,updated_at_ms=?7,resolved_at_ms=?8 WHERE recovery_id=?9",
            params![status,record.digest,record.artifact_ref,record.revision as i64,json,event_id,
                record.updated_at_ms as i64,record.resolved_at_ms.map(|v| v as i64),record.recovery_id],
        )?;
    }
    Ok(())
}

fn apply_snapshot(
    tx: &Transaction<'_>,
    record: &SessionSnapshotRecord,
    event_id: &str,
) -> Result<()> {
    if record.revision != 1 {
        return Err(ControlError::Invalid(
            "session snapshots are immutable".into(),
        ));
    }
    check_revision::<SessionSnapshotRecord>(
        tx,
        "session_snapshots",
        "snapshot_id",
        &record.snapshot_id,
        record.revision,
        |_| false,
    )?;
    let json = serde_json::to_string(record)?;
    tx.execute(
        "INSERT INTO session_snapshots
         (snapshot_id,session_id,reducer_version,through_sequence,state_ref,state_digest,state_bytes,
          revision,record_json,last_event_id,created_at_ms)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
        params![record.snapshot_id,record.session_id,record.reducer_version as i64,
            record.through_sequence as i64,record.state_ref,record.state_digest,record.state_bytes as i64,
            record.revision as i64,json,event_id,record.created_at_ms as i64],
    )?;
    Ok(())
}

fn check_revision<T: DeserializeOwned>(
    tx: &Transaction<'_>,
    table: &str,
    id_column: &str,
    id: &str,
    revision: u64,
    immutable_matches: impl FnOnce(&T) -> bool,
) -> Result<()> {
    let sql = format!("SELECT revision, record_json FROM {table} WHERE {id_column} = ?1");
    let current = tx
        .query_row(&sql, [id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })
        .optional()?;
    match current {
        None if revision == 1 => Ok(()),
        None => Err(ControlError::Invalid(format!(
            "projection {id} must begin at revision one"
        ))),
        Some((current_revision, json)) if revision == current_revision as u64 + 1 => {
            let current: T = parse_record(json)?;
            if immutable_matches(&current) {
                Ok(())
            } else {
                Err(ControlError::Invalid(format!(
                    "projection {id} changed immutable identity fields"
                )))
            }
        }
        Some((current_revision, _)) => Err(ControlError::Invalid(format!(
            "stale projection revision for {id}: expected {}, received {revision}",
            current_revision + 1
        ))),
    }
}

fn enum_name<T: Serialize>(value: &T) -> Result<String> {
    serde_json::to_value(value)?
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            ControlError::Invalid("projection enum did not serialize as a string".into())
        })
}
