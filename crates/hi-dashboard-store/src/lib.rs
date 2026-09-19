//! SQLite-backed persistent dashboard workspace: membership, layout ranks, grouping.
//!
//! Port of Grok Build `xai-grok-dashboard-store` (Apache-2.0).

use std::path::{Path, PathBuf};

mod error;
mod owner_only;
mod schema;
mod store;
mod store_open;
mod types;

pub use error::{Result, StoreError};
pub use schema::USER_VERSION;
pub use store::WorkspaceStore;
pub use types::{
    Grouping, InsertOutcome, LayoutApplyOutcome, LayoutGrouping, LayoutPatch, MAX_CWD_BYTES,
    MAX_ENUM_BYTES, MAX_MODEL_BYTES, MAX_SESSION_ID_BYTES, MAX_SUMMARY_BYTES, MAX_TITLE_BYTES,
    Member, MemberKey, MemberKind, MemberMetadata, MemberOrigin, NewMember, PinAssignment,
    RANK_GAP, RankAssignment, RekeyOutcome, RemoveOutcome, SchemaState, SessionId,
    WORKSPACE_CAPACITY, WorkspaceSnapshot,
};

/// `{hi_data_dir}/dashboard/workspace.db`
pub fn default_db_path(hi_data_dir: &Path) -> PathBuf {
    hi_data_dir.join("dashboard").join("workspace.db")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn now_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
    }

    fn member(id: &str, title: &str, cwd: &str) -> NewMember {
        NewMember {
            key: MemberKey {
                session_id: SessionId::new(id).unwrap(),
                kind: MemberKind::Build,
            },
            origin: MemberOrigin::Local,
            metadata: MemberMetadata {
                cwd: Some(cwd.to_string()),
                title: Some(title.to_string()),
                model: Some("pipe/deepseek-v4-flash-0731".into()),
                last_turn_summary: None,
                is_worktree: false,
                last_change_unix_ms: now_ms(),
            },
        }
    }

    #[test]
    fn insert_snapshot_update_remove() {
        let dir = tempfile::tempdir().unwrap();
        let path = default_db_path(dir.path());
        let mut store = WorkspaceStore::open(&path).unwrap();
        assert_eq!(store.schema_state(), SchemaState::Current);
        assert!(matches!(
            store
                .insert_member(member("sess-a", "fix login", "/tmp/proj"))
                .unwrap(),
            InsertOutcome::Inserted
        ));
        let snap = store.snapshot().unwrap();
        assert_eq!(snap.members.len(), 1);
        assert_eq!(snap.members[0].title.as_deref(), Some("fix login"));
        assert_eq!(
            snap.members[0].model.as_deref(),
            Some("pipe/deepseek-v4-flash-0731")
        );
        assert_eq!(snap.grouping, Grouping::State);

        store
            .update_member_metadata(
                &MemberKey {
                    session_id: SessionId::new("sess-a").unwrap(),
                    kind: MemberKind::Build,
                },
                &MemberMetadata {
                    cwd: Some("/tmp/proj".into()),
                    title: Some("fix login (wip)".into()),
                    model: Some("gpt-6-astra".into()),
                    last_turn_summary: Some("running tests".into()),
                    is_worktree: true,
                    last_change_unix_ms: now_ms(),
                },
            )
            .unwrap();
        let snap = store.snapshot().unwrap();
        assert_eq!(snap.members[0].title.as_deref(), Some("fix login (wip)"));
        assert_eq!(snap.members[0].model.as_deref(), Some("gpt-6-astra"));
        assert!(snap.members[0].is_worktree);

        let key = MemberKey {
            session_id: SessionId::new("sess-a").unwrap(),
            kind: MemberKind::Build,
        };
        store
            .set_pin_rank(&[RankAssignment {
                key: key.clone(),
                rank: Some(RANK_GAP),
            }])
            .unwrap();
        assert!(store.snapshot().unwrap().members[0].pin_rank.is_some());
        store.set_grouping(&Grouping::Directory).unwrap();
        assert_eq!(store.snapshot().unwrap().grouping, Grouping::Directory);
        assert_eq!(store.remove_member(&key).unwrap(), RemoveOutcome::Removed);
        assert_eq!(
            store.remove_member(&key).unwrap(),
            RemoveOutcome::NotPresent
        );
        assert!(store.snapshot().unwrap().members.is_empty());
    }

    #[test]
    fn insert_existing_updates_metadata_keeps_origin() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = WorkspaceStore::open(&default_db_path(dir.path())).unwrap();
        store
            .insert_member(member("sess-b", "one", "/tmp/a"))
            .unwrap();
        assert!(matches!(
            store
                .insert_member(member("sess-b", "two", "/tmp/a"))
                .unwrap(),
            InsertOutcome::UpdatedExisting
        ));
        let snap = store.snapshot().unwrap();
        assert_eq!(snap.members.len(), 1);
        assert_eq!(snap.members[0].title.as_deref(), Some("two"));
        assert_eq!(snap.members[0].origin, MemberOrigin::Local);
    }
}
