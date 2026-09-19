//! Store creation, connection opening, permission setup, and initial schema gating.

use std::path::{Path, PathBuf};
use std::time::Instant;

use hi_sqlite_journal::JournalMode;

use super::WorkspaceStore;
use crate::error::{Result, StoreError, classify_open_error};
use crate::owner_only::{
    create_dir_owner_only, create_owner_only, sibling_path, tighten_owner_only,
};
use crate::schema::{self, SchemaInit, USER_VERSION, read_user_version};
use crate::types::SchemaState;

impl WorkspaceStore {
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            create_dir_owner_only(parent)?;
        }
        let mode = JournalMode::for_db_path(db_path);
        create_owner_only(db_path)?;
        let opened_at = Instant::now();
        let mut conn = mode.open(db_path).map_err(|error| {
            classify_open_error(
                StoreError::Io(std::io::Error::other(format!("{error:#}"))),
                opened_at,
                db_path,
            )
        })?;
        tighten_owner_only(db_path)?;
        for suffix in ["-wal", "-shm", "-journal"] {
            tighten_owner_only(&sibling_path(db_path, suffix))?;
        }

        let found = read_user_version(&conn)
            .map_err(|error| classify_open_error(error, opened_at, db_path))?;
        if found > USER_VERSION {
            return Self::open_newer_schema(conn, db_path.to_path_buf(), found, opened_at);
        }
        match schema::init_schema(&mut conn)
            .map_err(|error| classify_open_error(error, opened_at, db_path))?
        {
            SchemaInit::Newer { user_version } => {
                Self::open_newer_schema(conn, db_path.to_path_buf(), user_version, opened_at)
            }
            SchemaInit::Ready { created } => {
                if created {
                    tracing::info!(
                        path = %db_path.display(),
                        journal_mode = mode.as_str(),
                        user_version = USER_VERSION,
                        "workspace store created"
                    );
                }
                Ok(Self {
                    conn,
                    schema: SchemaState::Current,
                    path: db_path.to_path_buf(),
                })
            }
        }
    }

    fn open_newer_schema(
        conn: rusqlite::Connection,
        effective: PathBuf,
        found: u32,
        opened_at: Instant,
    ) -> Result<Self> {
        conn.pragma_update(None, "query_only", true)
            .map_err(|error| classify_open_error(error.into(), opened_at, &effective))?;
        tracing::warn!(
            path = %effective.display(),
            found,
            supported = USER_VERSION,
            "workspace store written by a newer hi; opening read-only"
        );
        Ok(Self {
            conn,
            schema: SchemaState::NewerReadOnly {
                user_version: found,
            },
            path: effective,
        })
    }
}
