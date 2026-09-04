use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};

use crate::{ControlError, Result};

pub const CONTROL_SCHEMA_VERSION: i64 = 2;

pub(crate) fn migrate(connection: &mut Connection) -> Result<()> {
    // Foreign-key enforcement is connection-local and SQLite ignores attempts
    // to change it while a transaction is active, so enable and verify it
    // before taking the migration write lock.
    connection.pragma_update(None, "foreign_keys", "ON")?;
    let foreign_keys: i64 = connection.query_row("PRAGMA foreign_keys", [], |row| row.get(0))?;
    if foreign_keys != 1 {
        return Err(ControlError::Invalid(
            "SQLite foreign-key enforcement could not be enabled".into(),
        ));
    }

    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    tx.execute_batch(
        "CREATE TABLE IF NOT EXISTS control_meta (
           key TEXT PRIMARY KEY,
           value TEXT NOT NULL
         );",
    )?;

    let stored_version = tx
        .query_row(
            "SELECT value FROM control_meta WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .map(|value| {
            value.parse::<i64>().map_err(|_| {
                ControlError::Invalid(format!("invalid control schema version {value:?}"))
            })
        })
        .transpose()?
        .unwrap_or(0);

    if stored_version < 0 {
        return Err(ControlError::Invalid(format!(
            "invalid control schema version {stored_version}"
        )));
    }
    if stored_version > CONTROL_SCHEMA_VERSION {
        return Err(ControlError::IncompatibleSchema {
            found: stored_version,
            supported: CONTROL_SCHEMA_VERSION,
        });
    }

    if stored_version < 2 {
        // Re-run the idempotent v1 DDL as part of the upgrade so databases
        // left partially initialized by the former non-transactional migrator
        // are repaired before v2 foreign keys are introduced.
        tx.execute_batch(V1_SCHEMA)?;
        tx.execute_batch(V2_SCHEMA)?;
    }

    // Updating the version is deliberately the final statement in the same
    // transaction as all schema changes. A crash exposes either the complete
    // old schema or the complete new schema, never a falsely upgraded store.
    hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::SchemaBeforeVersionUpdate)
        .map_err(|error| ControlError::Invalid(error.to_string()))?;
    tx.execute(
        "INSERT INTO control_meta(key, value) VALUES ('schema_version', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![CONTROL_SCHEMA_VERSION.to_string()],
    )?;
    tx.commit()?;
    Ok(())
}

const V1_SCHEMA: &str = r#"
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
CREATE INDEX IF NOT EXISTS control_artifacts_scope
  ON control_artifacts(scope_id, sensitivity);
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
CREATE INDEX IF NOT EXISTS run_events_event_id ON run_events(event_id);
"#;

const V2_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS control_workspace_bindings (
  binding_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  session_id TEXT,
  epoch INTEGER NOT NULL,
  authority TEXT NOT NULL,
  state TEXT NOT NULL,
  workspace_version TEXT,
  revision INTEGER NOT NULL CHECK(revision >= 1),
  record_json TEXT NOT NULL,
  last_event_id TEXT NOT NULL REFERENCES run_events(event_id),
  opened_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  closed_at_ms INTEGER
);
CREATE INDEX IF NOT EXISTS control_workspace_bindings_workspace
  ON control_workspace_bindings(workspace_id, epoch DESC);
CREATE INDEX IF NOT EXISTS control_workspace_bindings_session
  ON control_workspace_bindings(session_id, updated_at_ms DESC);

CREATE TABLE IF NOT EXISTS control_workspace_operations (
  operation_id TEXT PRIMARY KEY,
  binding_id TEXT NOT NULL REFERENCES control_workspace_bindings(binding_id),
  epoch INTEGER NOT NULL,
  session_id TEXT,
  run_id TEXT,
  attempt_id TEXT,
  job_id TEXT,
  kind TEXT NOT NULL,
  replay_class TEXT NOT NULL,
  status TEXT NOT NULL,
  operation_digest TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  base_version TEXT,
  result_version TEXT,
  revision INTEGER NOT NULL CHECK(revision >= 1),
  record_json TEXT NOT NULL,
  last_event_id TEXT NOT NULL REFERENCES run_events(event_id),
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  settled_at_ms INTEGER
);
CREATE INDEX IF NOT EXISTS control_workspace_operations_binding
  ON control_workspace_operations(binding_id, epoch, status, updated_at_ms);
CREATE INDEX IF NOT EXISTS control_workspace_operations_session
  ON control_workspace_operations(session_id, updated_at_ms);

CREATE TABLE IF NOT EXISTS control_jobs (
  job_id TEXT PRIMARY KEY,
  session_id TEXT,
  run_id TEXT,
  attempt_id TEXT,
  binding_id TEXT REFERENCES control_workspace_bindings(binding_id),
  epoch INTEGER,
  kind TEXT NOT NULL,
  effect_scope TEXT NOT NULL,
  state TEXT NOT NULL,
  application_state TEXT,
  operation_digest TEXT,
  idempotency_key TEXT UNIQUE,
  workspace_version TEXT,
  revision INTEGER NOT NULL CHECK(revision >= 1),
  record_json TEXT NOT NULL,
  last_event_id TEXT NOT NULL REFERENCES run_events(event_id),
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  cancel_requested_at_ms INTEGER,
  finished_at_ms INTEGER
);
CREATE INDEX IF NOT EXISTS control_jobs_workspace_state
  ON control_jobs(binding_id, epoch, state, updated_at_ms);
CREATE INDEX IF NOT EXISTS control_jobs_session
  ON control_jobs(session_id, updated_at_ms);

CREATE TABLE IF NOT EXISTS control_workspace_recoveries (
  recovery_id TEXT PRIMARY KEY,
  binding_id TEXT REFERENCES control_workspace_bindings(binding_id),
  workspace_id TEXT NOT NULL,
  session_id TEXT,
  operation_id TEXT REFERENCES control_workspace_operations(operation_id),
  job_id TEXT REFERENCES control_jobs(job_id),
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  digest TEXT,
  artifact_ref TEXT,
  revision INTEGER NOT NULL CHECK(revision >= 1),
  record_json TEXT NOT NULL,
  last_event_id TEXT NOT NULL REFERENCES run_events(event_id),
  created_at_ms INTEGER NOT NULL,
  updated_at_ms INTEGER NOT NULL,
  resolved_at_ms INTEGER
);
CREATE INDEX IF NOT EXISTS control_workspace_recoveries_workspace
  ON control_workspace_recoveries(workspace_id, status, updated_at_ms);
CREATE INDEX IF NOT EXISTS control_workspace_recoveries_session
  ON control_workspace_recoveries(session_id, status, updated_at_ms);

CREATE TABLE IF NOT EXISTS session_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  reducer_version INTEGER NOT NULL,
  through_sequence INTEGER NOT NULL,
  state_ref TEXT NOT NULL,
  state_digest TEXT NOT NULL,
  state_bytes INTEGER NOT NULL,
  revision INTEGER NOT NULL CHECK(revision >= 1),
  record_json TEXT NOT NULL,
  last_event_id TEXT NOT NULL REFERENCES run_events(event_id),
  created_at_ms INTEGER NOT NULL,
  UNIQUE(session_id, through_sequence)
);
CREATE INDEX IF NOT EXISTS session_snapshots_latest
  ON session_snapshots(session_id, through_sequence DESC);

CREATE TABLE IF NOT EXISTS control_projection_events (
  event_id TEXT PRIMARY KEY REFERENCES run_events(event_id),
  projection_kind TEXT NOT NULL,
  projection_id TEXT NOT NULL,
  projection_revision INTEGER NOT NULL,
  projection_digest TEXT NOT NULL,
  UNIQUE(projection_kind, projection_id, projection_revision)
);
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_v1_and_reopens_idempotently() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.sqlite3");
        let mut connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE control_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO control_meta(key, value) VALUES ('schema_version', '1');",
            )
            .unwrap();
        connection.execute_batch(V1_SCHEMA).unwrap();
        connection
            .execute(
                "INSERT INTO control_runs
                 (run_id, kind, status, desired_state, created_at_ms, updated_at_ms)
                 VALUES ('legacy-run', '\"interactive\"', 'queued', 'run', 1, 1)",
                [],
            )
            .unwrap();

        migrate(&mut connection).unwrap();
        migrate(&mut connection).unwrap();

        let version: i64 = connection
            .query_row(
                "SELECT CAST(value AS INTEGER) FROM control_meta
                 WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, CONTROL_SCHEMA_VERSION);
        let table_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table' AND name = 'control_jobs'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(table_count, 1);
        let legacy_rows: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM control_runs WHERE run_id = 'legacy-run'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(legacy_rows, 1);
    }

    #[test]
    fn refuses_a_future_schema_without_rewriting_it() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE control_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO control_meta(key, value) VALUES ('schema_version', '99');",
            )
            .unwrap();

        assert!(matches!(
            migrate(&mut connection),
            Err(ControlError::IncompatibleSchema {
                found: 99,
                supported: CONTROL_SCHEMA_VERSION
            })
        ));
        let version: String = connection
            .query_row(
                "SELECT value FROM control_meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, "99");
    }

    #[test]
    fn failed_migration_rolls_back_schema_and_version_together() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE control_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO control_meta(key, value) VALUES ('schema_version', '1');",
            )
            .unwrap();
        connection.execute_batch(V1_SCHEMA).unwrap();
        // Force v2's job index creation to fail after earlier v2 DDL has run.
        connection
            .execute_batch("CREATE TABLE control_jobs (job_id TEXT PRIMARY KEY);")
            .unwrap();

        assert!(matches!(
            migrate(&mut connection),
            Err(ControlError::Database(_))
        ));
        let version: String = connection
            .query_row(
                "SELECT value FROM control_meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, "1");
        let partial_v2_tables: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table' AND name = 'control_workspace_bindings'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(partial_v2_tables, 0);
    }
}
