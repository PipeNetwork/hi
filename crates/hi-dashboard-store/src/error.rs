//! Typed failures for the workspace store.

pub type Result<T> = std::result::Result<T, StoreError>;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("workspace store is schema v{found}; this build supports v{supported} (read-only)")]
    NewerSchema { found: u32, supported: u32 },
    #[error("workspace is full ({capacity} members) and every member is pinned")]
    AllPinned { capacity: usize },
    #[error("no workspace member {session_id} ({kind})")]
    MemberNotFound { session_id: String, kind: String },
    #[error("invalid workspace layout patch: {reason}")]
    InvalidLayoutPatch { reason: &'static str },
    #[error("invalid session id: {reason}")]
    InvalidSessionId { reason: &'static str },
    #[error("cwd is required for build members")]
    CwdRequired,
    #[error("cwd must be an absolute path")]
    CwdNotAbsolute,
    #[error("cwd exceeds {max} bytes")]
    CwdTooLong { max: usize },
    #[error("invalid unknown {column} value: {reason}")]
    InvalidEnumValue {
        column: &'static str,
        reason: &'static str,
    },
    #[error("unknown {column} value exceeds {max} bytes")]
    EnumValueTooLong { column: &'static str, max: usize },
    #[error("workspace store busy after {waited_ms}ms")]
    Busy { waited_ms: u64 },
    #[error("workspace store file is unusable: {source}")]
    Unusable {
        #[source]
        source: rusqlite::Error,
    },
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

fn is_busy_code(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(f, _)
            if matches!(f.code, rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked)
    )
}

fn is_unusable_db_error(error: &rusqlite::Error) -> bool {
    match error {
        rusqlite::Error::SqliteFailure(f, message) => {
            matches!(
                f.code,
                rusqlite::ErrorCode::NotADatabase | rusqlite::ErrorCode::DatabaseCorrupt
            ) || message.as_deref().is_some_and(|message| {
                let message = message.to_ascii_lowercase();
                message.contains("disk image is malformed")
                    || message.contains("malformed database schema")
                    || message.contains("is not a database")
            })
        }
        _ => false,
    }
}

pub(crate) fn classify_unusable(error: rusqlite::Error) -> StoreError {
    if is_unusable_db_error(&error) {
        StoreError::Unusable { source: error }
    } else {
        StoreError::Sqlite(error)
    }
}

pub(crate) fn classify_busy(
    error: StoreError,
    op: &'static str,
    started: std::time::Instant,
) -> StoreError {
    match error {
        StoreError::Sqlite(e) if is_busy_code(&e) => {
            let waited_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
            tracing::warn!(op, waited_ms, "workspace store busy budget exhausted");
            StoreError::Busy { waited_ms }
        }
        other => other,
    }
}

pub(crate) fn classify_open_error(
    error: StoreError,
    started: std::time::Instant,
    path: &std::path::Path,
) -> StoreError {
    let error = match error {
        StoreError::Sqlite(error) => classify_unusable(error),
        other => other,
    };
    if matches!(error, StoreError::Unusable { .. }) {
        tracing::error!(path = %path.display(), error = %error, "workspace store file is unusable");
    }
    classify_busy(error, "open", started)
}
