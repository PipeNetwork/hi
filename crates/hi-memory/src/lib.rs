//! Trusted, scoped memory persistence. Candidate writes remain hypotheses.

use std::{fs, path::Path};

use anyhow::{Result, ensure};
use hi_rsi_runtime::{MemoryClass, MemoryEntry};
use rusqlite::{Connection, params};

pub struct MemoryStore {
    tenant_id: String,
    connection: Connection,
}

impl MemoryStore {
    pub fn open(root: &Path, tenant_id: &str) -> Result<Self> {
        validate_scope(tenant_id)?;
        let directory = root.join(tenant_id);
        fs::create_dir_all(&directory)?;
        let connection = Connection::open(directory.join("memory.sqlite"))?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.execute_batch("CREATE TABLE IF NOT EXISTS memories(
            id INTEGER PRIMARY KEY, class TEXT NOT NULL, content_json TEXT NOT NULL,
            tenant TEXT NOT NULL, repository_scope TEXT, candidate TEXT NOT NULL,
            artifacts_json TEXT NOT NULL, confidence INTEGER NOT NULL,
            created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
            supervisor_verified INTEGER NOT NULL, provenance_hash TEXT NOT NULL UNIQUE);
            CREATE INDEX IF NOT EXISTS memory_scope ON memories(class,tenant,repository_scope,candidate,expires_at);")?;
        Ok(Self {
            tenant_id: tenant_id.into(),
            connection,
        })
    }

    pub fn write_candidate_hypothesis(&self, entry: &MemoryEntry) -> Result<String> {
        entry.validate_candidate_authored()?;
        self.write(entry)
    }

    pub fn write_supervisor_verified(&self, entry: &MemoryEntry) -> Result<String> {
        ensure!(
            entry.supervisor_verified,
            "trusted write must be marked verified"
        );
        ensure!(
            entry.confidence_millionths <= 1_000_000
                && entry.expires_at_unix_ms > entry.created_at_unix_ms,
            "invalid trusted memory"
        );
        self.write(entry)
    }

    fn write(&self, entry: &MemoryEntry) -> Result<String> {
        ensure!(
            entry.tenant_id == self.tenant_id,
            "cross-tenant memory write denied"
        );
        validate_entry_scope(entry)?;
        ensure!(
            !entry.supporting_artifacts.is_empty() || !entry.supervisor_verified,
            "verified durable memory requires supporting artifacts"
        );
        let bytes = serde_json::to_vec(entry)?;
        let hash = blake3_hash(&bytes);
        self.connection.execute("INSERT OR IGNORE INTO memories(class,content_json,tenant,repository_scope,candidate,artifacts_json,confidence,created_at,expires_at,supervisor_verified,provenance_hash) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)", params![class_name(entry.memory_class), serde_json::to_string(&entry.content)?, entry.tenant_id, entry.repository_scope, entry.candidate_id, serde_json::to_string(&entry.supporting_artifacts)?, entry.confidence_millionths, entry.created_at_unix_ms, entry.expires_at_unix_ms, entry.supervisor_verified, hash])?;
        Ok(hash)
    }

    pub fn query(
        &self,
        class: MemoryClass,
        repository: Option<&str>,
        candidate: &str,
        now_unix_ms: u64,
        limit: u32,
    ) -> Result<Vec<MemoryEntry>> {
        ensure!(
            !candidate.is_empty() && limit > 0 && limit <= 1000,
            "invalid memory query"
        );
        let mut statement = self.connection.prepare("SELECT content_json,repository_scope,candidate,artifacts_json,confidence,created_at,expires_at,supervisor_verified FROM memories WHERE class=?1 AND tenant=?2 AND expires_at>?3 ORDER BY supervisor_verified DESC,confidence DESC,created_at DESC LIMIT ?4")?;
        let rows = statement.query_map(
            params![class_name(class), self.tenant_id, now_unix_ms, limit],
            |row| {
                Ok(MemoryEntry {
                    memory_class: class,
                    content: serde_json::from_str(&row.get::<_, String>(0)?).map_err(sql_json)?,
                    tenant_id: self.tenant_id.clone(),
                    repository_scope: row.get(1)?,
                    candidate_id: row.get(2)?,
                    supporting_artifacts: serde_json::from_str(&row.get::<_, String>(3)?)
                        .map_err(sql_json)?,
                    confidence_millionths: row.get(4)?,
                    created_at_unix_ms: row.get(5)?,
                    expires_at_unix_ms: row.get(6)?,
                    supervisor_verified: row.get(7)?,
                })
            },
        )?;
        let entries = rows
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|entry| visible(entry, repository, candidate))
            .collect();
        Ok(entries)
    }
}

fn visible(entry: &MemoryEntry, repository: Option<&str>, candidate: &str) -> bool {
    match entry.memory_class {
        MemoryClass::Working | MemoryClass::Attempt => entry.candidate_id == candidate,
        MemoryClass::Repository => entry.repository_scope.as_deref() == repository,
        MemoryClass::Episodic => true,
        MemoryClass::Procedural => entry.candidate_id == candidate,
    }
}

fn validate_entry_scope(entry: &MemoryEntry) -> Result<()> {
    validate_scope(&entry.tenant_id)?;
    ensure!(
        !entry.candidate_id.is_empty() && entry.candidate_id.len() <= 256,
        "invalid candidate memory scope"
    );
    if entry.memory_class == MemoryClass::Repository {
        ensure!(
            entry
                .repository_scope
                .as_ref()
                .is_some_and(|v| !v.is_empty()),
            "repository memory requires repository scope"
        );
    }
    Ok(())
}

fn validate_scope(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b)),
        "invalid memory tenant scope"
    );
    Ok(())
}

fn class_name(class: MemoryClass) -> &'static str {
    match class {
        MemoryClass::Working => "working",
        MemoryClass::Attempt => "attempt",
        MemoryClass::Repository => "repository",
        MemoryClass::Episodic => "episodic",
        MemoryClass::Procedural => "procedural",
    }
}
fn blake3_hash(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}
fn sql_json(error: serde_json::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn entry(tenant: &str, candidate: &str) -> MemoryEntry {
        MemoryEntry {
            memory_class: MemoryClass::Procedural,
            content: json!({"fact":"run tests"}),
            tenant_id: tenant.into(),
            repository_scope: None,
            candidate_id: candidate.into(),
            supporting_artifacts: vec![],
            confidence_millionths: 500_000,
            created_at_unix_ms: 1,
            expires_at_unix_ms: 100,
            supervisor_verified: false,
        }
    }
    #[test]
    fn enforces_tenant_candidate_and_verification_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let store = MemoryStore::open(tmp.path(), "tenant-a").unwrap();
        store
            .write_candidate_hypothesis(&entry("tenant-a", "candidate-a"))
            .unwrap();
        assert_eq!(
            store
                .query(MemoryClass::Procedural, None, "candidate-a", 2, 10)
                .unwrap()
                .len(),
            1
        );
        assert!(
            store
                .query(MemoryClass::Procedural, None, "candidate-b", 2, 10)
                .unwrap()
                .is_empty()
        );
        let mut forged = entry("tenant-a", "candidate-a");
        forged.supervisor_verified = true;
        assert!(store.write_candidate_hypothesis(&forged).is_err());
        assert!(
            store
                .write_candidate_hypothesis(&entry("tenant-b", "candidate-a"))
                .is_err()
        );
    }
}
