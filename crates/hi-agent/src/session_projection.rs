//! Versioned snapshots and atomic tail patches for presentation clients.
//!
//! A projection is only a transport around [`SessionReducer`]. The reducer
//! remains the sole state authority; clients either install a complete,
//! integrity-checked snapshot or apply an exact-base event tail atomically.

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    SESSION_REDUCER_VERSION, SessionEvent, SessionReduceError, SessionReducer,
    SessionReducerSnapshot,
};

pub const SESSION_PROJECTION_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionProjectionSnapshot {
    pub schema_version: u16,
    pub reducer_version: u32,
    pub revision: u64,
    pub digest: String,
    pub reducer: SessionReducerSnapshot,
}

impl SessionProjectionSnapshot {
    pub fn validate(&self) -> Result<(), SessionProjectionError> {
        validate_versions(self.schema_version, self.reducer_version)?;
        if self.reducer_version != self.reducer.reducer_version {
            return Err(SessionProjectionError::UnsupportedReducer {
                found: self.reducer.reducer_version,
                supported: self.reducer_version,
            });
        }
        if self.revision != self.reducer.through_sequence {
            return Err(SessionProjectionError::RevisionMismatch {
                declared: self.revision,
                actual: self.reducer.through_sequence,
            });
        }
        SessionReducer::from_snapshot(self.reducer.clone())?;
        let actual = reducer_digest(&self.reducer);
        if self.digest != actual {
            return Err(SessionProjectionError::DigestMismatch {
                expected: self.digest.clone(),
                actual,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionProjectionPatch {
    pub schema_version: u16,
    pub reducer_version: u32,
    pub base_revision: u64,
    pub target_revision: u64,
    pub base_digest: String,
    pub target_digest: String,
    pub events: Vec<SessionEvent>,
}

/// In-memory projection used by CLI, TUI, remote views, and inspection tools.
#[derive(Clone, Debug, Default)]
pub struct SessionProjection {
    reducer: SessionReducer,
}

impl SessionProjection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_snapshot(
        snapshot: SessionProjectionSnapshot,
    ) -> Result<Self, SessionProjectionError> {
        snapshot.validate()?;
        Ok(Self {
            reducer: SessionReducer::from_snapshot(snapshot.reducer)?,
        })
    }

    pub fn reducer(&self) -> &SessionReducer {
        &self.reducer
    }

    pub fn snapshot(&self) -> SessionProjectionSnapshot {
        snapshot_for(&self.reducer)
    }

    /// Build a patch without changing the live projection. Every event is
    /// validated by a cloned reducer and the resulting digest is sealed.
    pub fn prepare_patch(
        &self,
        events: Vec<SessionEvent>,
    ) -> Result<SessionProjectionPatch, SessionProjectionError> {
        let base = self.snapshot();
        let mut candidate = self.reducer.clone();
        candidate.apply_all(events.clone())?;
        let target = snapshot_for(&candidate);
        Ok(SessionProjectionPatch {
            schema_version: SESSION_PROJECTION_SCHEMA_VERSION,
            reducer_version: SESSION_REDUCER_VERSION,
            base_revision: base.revision,
            target_revision: target.revision,
            base_digest: base.digest,
            target_digest: target.digest,
            events,
        })
    }

    /// Apply an exact-base patch transactionally. A stale, malformed, or
    /// tampered patch leaves the current reducer untouched.
    pub fn apply_patch(
        &mut self,
        patch: SessionProjectionPatch,
    ) -> Result<SessionProjectionSnapshot, SessionProjectionError> {
        validate_versions(patch.schema_version, patch.reducer_version)?;
        // A v2 snapshot can be migrated, but an old producer's patch digests
        // describe old reduction rules. Request a current snapshot instead.
        if patch.reducer_version != SESSION_REDUCER_VERSION {
            return Err(SessionProjectionError::UnsupportedReducer {
                found: patch.reducer_version,
                supported: SESSION_REDUCER_VERSION,
            });
        }
        let current = self.snapshot();
        if patch.base_revision != current.revision || patch.base_digest != current.digest {
            return Err(SessionProjectionError::StaleBase {
                expected_revision: current.revision,
                found_revision: patch.base_revision,
            });
        }
        let mut candidate = self.reducer.clone();
        candidate.apply_all(patch.events)?;
        let target = snapshot_for(&candidate);
        if patch.target_revision != target.revision {
            return Err(SessionProjectionError::RevisionMismatch {
                declared: patch.target_revision,
                actual: target.revision,
            });
        }
        if patch.target_digest != target.digest {
            return Err(SessionProjectionError::DigestMismatch {
                expected: patch.target_digest,
                actual: target.digest,
            });
        }
        self.reducer = candidate;
        Ok(self.snapshot())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionProjectionError {
    UnsupportedSchema {
        found: u16,
        supported: u16,
    },
    UnsupportedReducer {
        found: u32,
        supported: u32,
    },
    RevisionMismatch {
        declared: u64,
        actual: u64,
    },
    DigestMismatch {
        expected: String,
        actual: String,
    },
    StaleBase {
        expected_revision: u64,
        found_revision: u64,
    },
    Reduce(SessionReduceError),
}

impl fmt::Display for SessionProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema { found, supported } => write!(
                formatter,
                "unsupported session projection schema {found}; supported {supported}"
            ),
            Self::UnsupportedReducer { found, supported } => write!(
                formatter,
                "unsupported session projection reducer {found}; supported {supported}"
            ),
            Self::RevisionMismatch { declared, actual } => write!(
                formatter,
                "session projection revision mismatch: declared {declared}, actual {actual}"
            ),
            Self::DigestMismatch { expected, actual } => write!(
                formatter,
                "session projection digest mismatch: expected {expected}, actual {actual}"
            ),
            Self::StaleBase {
                expected_revision,
                found_revision,
            } => write!(
                formatter,
                "stale session patch: current revision {expected_revision}, patch base {found_revision}"
            ),
            Self::Reduce(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SessionProjectionError {}

impl From<SessionReduceError> for SessionProjectionError {
    fn from(value: SessionReduceError) -> Self {
        Self::Reduce(value)
    }
}

fn validate_versions(schema: u16, reducer: u32) -> Result<(), SessionProjectionError> {
    if schema != SESSION_PROJECTION_SCHEMA_VERSION {
        return Err(SessionProjectionError::UnsupportedSchema {
            found: schema,
            supported: SESSION_PROJECTION_SCHEMA_VERSION,
        });
    }
    if !crate::session_reducer::supported_reducer_version(reducer) {
        return Err(SessionProjectionError::UnsupportedReducer {
            found: reducer,
            supported: SESSION_REDUCER_VERSION,
        });
    }
    Ok(())
}

fn snapshot_for(reducer: &SessionReducer) -> SessionProjectionSnapshot {
    let snapshot = reducer.snapshot();
    SessionProjectionSnapshot {
        schema_version: SESSION_PROJECTION_SCHEMA_VERSION,
        reducer_version: SESSION_REDUCER_VERSION,
        revision: snapshot.through_sequence,
        digest: reducer_digest(&snapshot),
        reducer: snapshot,
    }
}

#[cfg(test)]
thread_local! { static DIGEST_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }

#[cfg(test)]
pub(crate) fn digest_calls() -> u64 {
    DIGEST_CALLS.with(std::cell::Cell::get)
}

fn reducer_digest(snapshot: &SessionReducerSnapshot) -> String {
    #[cfg(test)]
    DIGEST_CALLS.with(|calls| calls.set(calls.get() + 1));
    let bytes = serde_json::to_vec(snapshot).expect("session reducer snapshot serializes");
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use hi_ai::Message;

    use super::*;
    use crate::SessionEventKind;

    fn block_id(value: &str) -> crate::TranscriptBlockId {
        crate::TranscriptBlockId::new(value).unwrap()
    }

    fn message(sequence: u64, text: &str) -> SessionEvent {
        SessionEvent::new(SessionEventKind::Message {
            message: Message::user(text),
        })
        .at_sequence(sequence)
    }

    #[test]
    fn exact_base_patch_is_deterministic_and_round_trips() {
        let mut projection = SessionProjection::new();
        let patch = projection
            .prepare_patch(vec![message(1, "one"), message(2, "two")])
            .unwrap();
        let expected_digest = patch.target_digest.clone();
        let snapshot = projection.apply_patch(patch).unwrap();
        assert_eq!(snapshot.revision, 2);
        assert_eq!(snapshot.digest, expected_digest);

        let restored = SessionProjection::from_snapshot(snapshot.clone()).unwrap();
        assert_eq!(restored.snapshot().digest, snapshot.digest);
        assert!(
            restored
                .reducer()
                .state()
                .semantically_eq(projection.reducer().state())
        );
    }

    #[test]
    fn version_two_snapshot_digest_is_checked_before_migration() {
        let mut reducer = SessionReducer::new();
        reducer.apply_event(message(1, "old conversation")).unwrap();
        let mut old = snapshot_for(&reducer);
        old.reducer_version = 2;
        old.reducer.reducer_version = 2;
        old.reducer.state.reducer_version = 2;
        old.digest = reducer_digest(&old.reducer);
        let encoded = serde_json::to_value(&old).unwrap();
        assert!(encoded["reducer"]["state"].get("task_recovery").is_none());
        assert!(encoded["reducer"].get("workspace_execution").is_none());
        let decoded = serde_json::from_value(encoded).unwrap();
        let migrated = SessionProjection::from_snapshot(decoded).unwrap();
        assert_eq!(migrated.snapshot().reducer_version, SESSION_REDUCER_VERSION);
        assert_eq!(
            migrated.reducer().state().messages[0].text(),
            "old conversation"
        );
        assert_eq!(
            migrated.reducer().state().task_recovery,
            crate::TaskRecoveryState::default()
        );

        old.reducer.state.name = Some("tampered".into());
        assert!(matches!(
            SessionProjection::from_snapshot(old),
            Err(SessionProjectionError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn an_ephemeral_callback_cache_cannot_change_snapshot_digest() {
        let mut recovery = crate::TaskRecoveryState::default();
        recovery.observe(&crate::recovery::ValidationObservation::command(
            "callback".into(),
            "unrelated check",
            "revision".into(),
            crate::recovery::ValidationResult::Passed,
            "",
            std::path::Path::new("."),
            false,
        ));
        let mut reducer = SessionReducer::new();
        reducer
            .apply_event(SessionEvent::new(SessionEventKind::TaskRecovery {
                state: recovery,
            }))
            .unwrap();
        let snapshot = snapshot_for(&reducer);
        let encoded = serde_json::to_value(&snapshot).unwrap();
        assert!(encoded["reducer"]["state"].get("task_recovery").is_none());
        let restored =
            SessionProjection::from_snapshot(serde_json::from_value(encoded).unwrap()).unwrap();
        assert_eq!(restored.snapshot().digest, snapshot.digest);
    }

    #[test]
    fn recovery_v1_digest_is_checked_before_obligation_migration() {
        // Literal V1 struct field order and presence are intentional: hashing a
        // sorted Value or serializing migrated fields would change this wire.
        let legacy = r#"{"schema_version":1,"objective":"repair suite","limit":3,"remaining":2,"exhausted":false,"interventions":1,"last_reason":"repair","mutation_credited":true,"validations":{"full":{"best_failures":["suite::failure"],"passed":true,"failed_states":["[\"rev-a\",[\"suite::failure\"]]"]}},"observed_executions":["old-execution"]}"#;
        let recovery: crate::TaskRecoveryState = serde_json::from_str(legacy).unwrap();
        assert_eq!(serde_json::to_string(&recovery).unwrap(), legacy);
        let mut reducer = SessionReducer::new();
        reducer
            .apply_event(message(1, "retained conversation"))
            .unwrap();
        let mut old = snapshot_for(&reducer);
        old.reducer.state.task_recovery = recovery;
        old.digest = reducer_digest(&old.reducer);
        let encoded = serde_json::to_string(&old).unwrap();
        let decoded: SessionProjectionSnapshot = serde_json::from_str(&encoded).unwrap();
        let mut tampered = decoded.clone();
        tampered.reducer.state.task_recovery.remaining = 3;
        assert!(matches!(
            tampered.validate(),
            Err(SessionProjectionError::DigestMismatch { .. })
        ));
        let migrated = SessionProjection::from_snapshot(decoded).unwrap();
        assert_eq!(
            migrated.reducer().state().messages[0].text(),
            "retained conversation"
        );
        let state = &migrated.reducer().state().task_recovery;
        assert_eq!(state.schema_version, 2);
        assert_eq!(state.remaining, 2);
        assert_eq!(
            state.unresolved_validation_status("rev-a"),
            Some(crate::recovery::ValidationResult::Deferred)
        );
        assert_ne!(migrated.snapshot().digest, old.digest);
        assert!(
            serde_json::to_value(state)
                .unwrap()
                .get("observed_executions")
                .is_none()
        );
    }

    #[test]
    fn mismatched_snapshot_versions_and_old_rule_patches_require_new_snapshot() {
        let mut snapshot = SessionProjection::new().snapshot();
        snapshot.reducer_version = 2;
        assert!(snapshot.validate().is_err());
        let mut projection = SessionProjection::new();
        let mut patch = projection
            .prepare_patch(vec![message(1, "old producer")])
            .unwrap();
        patch.reducer_version = 2;
        assert!(matches!(
            projection.apply_patch(patch),
            Err(SessionProjectionError::UnsupportedReducer { found: 2, .. })
        ));
        assert_eq!(projection.snapshot().revision, 0);
    }

    #[test]
    fn stale_patch_is_rejected_without_partial_application() {
        let projection = SessionProjection::new();
        let patch = projection.prepare_patch(vec![message(1, "stale")]).unwrap();
        let mut advanced = SessionProjection::new();
        let first = advanced.prepare_patch(vec![message(1, "current")]).unwrap();
        advanced.apply_patch(first).unwrap();
        let before = advanced.snapshot();

        assert!(matches!(
            advanced.apply_patch(patch),
            Err(SessionProjectionError::StaleBase { .. })
        ));
        assert_eq!(advanced.snapshot().digest, before.digest);
    }

    #[test]
    fn tampered_target_is_rejected_without_partial_application() {
        let mut projection = SessionProjection::new();
        let mut patch = projection.prepare_patch(vec![message(1, "one")]).unwrap();
        patch.target_digest = "sha256:tampered".into();

        assert!(matches!(
            projection.apply_patch(patch),
            Err(SessionProjectionError::DigestMismatch { .. })
        ));
        assert_eq!(projection.snapshot().revision, 0);
    }

    #[test]
    fn lifecycle_patch_is_atomic_and_survives_snapshot_restore() {
        let id = block_id("turn-1:reply");
        let mut projection = SessionProjection::new();
        let patch = projection
            .prepare_patch(vec![
                SessionEvent::new(SessionEventKind::TranscriptBlockOpened {
                    block_id: id.clone(),
                    kind: crate::TranscriptBlockKind::Assistant,
                    content: "answer".into(),
                })
                .at_sequence(1),
                SessionEvent::new(SessionEventKind::TranscriptBlockSettled {
                    block_id: id.clone(),
                    terminal: crate::TranscriptBlockTerminal::Completed,
                })
                .at_sequence(2),
            ])
            .unwrap();
        let snapshot = projection.apply_patch(patch).unwrap();
        let restored = SessionProjection::from_snapshot(snapshot).unwrap();
        assert_eq!(restored.reducer().state().transcript_blocks[0].id, id);
        assert!(
            restored.reducer().state().transcript_blocks[0]
                .lifecycle
                .is_terminal()
        );
    }

    #[test]
    fn duplicate_terminal_transition_cannot_partially_advance_projection() {
        let id = block_id("turn-2:tool");
        let mut projection = SessionProjection::new();
        let initial = projection
            .prepare_patch(vec![
                SessionEvent::new(SessionEventKind::TranscriptBlockOpened {
                    block_id: id.clone(),
                    kind: crate::TranscriptBlockKind::Tool,
                    content: String::new(),
                })
                .at_sequence(1),
                SessionEvent::new(SessionEventKind::TranscriptBlockSettled {
                    block_id: id.clone(),
                    terminal: crate::TranscriptBlockTerminal::Completed,
                })
                .at_sequence(2),
            ])
            .unwrap();
        projection.apply_patch(initial).unwrap();
        let before = projection.snapshot();

        assert!(matches!(
            projection.prepare_patch(vec![
                SessionEvent::new(SessionEventKind::TranscriptBlockSettled {
                    block_id: id,
                    terminal: crate::TranscriptBlockTerminal::Failed,
                })
                .at_sequence(3),
            ]),
            Err(SessionProjectionError::Reduce(
                SessionReduceError::TranscriptBlock(
                    crate::TranscriptBlockTransitionError::AlreadySettled { .. }
                )
            ))
        ));
        assert_eq!(projection.snapshot().digest, before.digest);
        assert_eq!(projection.snapshot().revision, before.revision);
    }

    #[test]
    fn message_only_projection_snapshot_keeps_legacy_wire_shape() {
        let mut projection = SessionProjection::new();
        let patch = projection
            .prepare_patch(vec![message(1, "legacy")])
            .unwrap();
        projection.apply_patch(patch).unwrap();
        let encoded = serde_json::to_value(projection.snapshot()).unwrap();
        assert!(
            encoded["reducer"]["state"]
                .get("transcript_blocks")
                .is_none()
        );

        let decoded: SessionProjectionSnapshot = serde_json::from_value(encoded).unwrap();
        let restored = SessionProjection::from_snapshot(decoded).unwrap();
        assert!(restored.reducer().state().transcript_blocks.is_empty());
        assert_eq!(restored.reducer().state().messages[0].text(), "legacy");
    }
}
