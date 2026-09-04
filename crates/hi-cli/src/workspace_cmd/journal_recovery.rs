use std::collections::BTreeSet;
use std::fs;

use anyhow::{Context, Result, bail, ensure};
use hi_control::{
    ControlStore, OperationReplayClass, WorkspaceOperationRecord, WorkspaceProjectionJournal,
    WorkspaceRecoveryRecord, WorkspaceRecoveryStatus,
};
use hi_workspace::{
    BindingId, ControllerId, OperationId, RecoveryId, RecoveryOutcome, RecoveryStatus, ReplayClass,
    WORKSPACE_CONTRACT_SCHEMA_VERSION, WorkspaceAuthority, WorkspaceBinding, WorkspaceId,
    WorkspaceVersion, restart_operation_recovery_id,
};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(super) struct JournalRecoveryView {
    pub schema_version: u16,
    pub session_id: String,
    pub recovery_id: String,
    pub recovery_cache_id: String,
    pub cache_confirmation_digest: Option<String>,
    pub binding_id: Option<String>,
    pub operation_id: Option<String>,
    pub job_id: Option<String>,
    pub kind: String,
    pub status: WorkspaceRecoveryStatus,
    pub digest: Option<String>,
    pub artifact_ref: Option<String>,
    pub detail: Option<String>,
    pub error: Option<String>,
    pub remote_retry_safe: bool,
    pub safe_resolution: String,
}

#[derive(Clone, Debug)]
pub(super) struct RecoveryTarget {
    pub requested_id: String,
    pub cache_id: String,
    pub remote_retry_safe: bool,
    pub journal_recovery_id: Option<String>,
}

pub(super) fn list(
    scope: &hi_pipefs::PipeFsCacheScope,
    caches: &[hi_pipefs::PipeFsRecoveryCache],
    session_id: &str,
) -> Result<Vec<JournalRecoveryView>> {
    let mut views = Vec::new();
    for cache in caches {
        let state_root = cache.path.join("runtime-state");
        let state_metadata = match fs::symlink_metadata(&state_root) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        ensure!(
            state_metadata.is_dir() && !state_metadata.file_type().is_symlink(),
            "PipeFS recovery runtime state is not a real directory: {}",
            state_root.display()
        );
        let cache_root = cache.path.canonicalize()?;
        let canonical_state = state_root.canonicalize()?;
        ensure!(
            canonical_state.starts_with(&cache_root) && canonical_state != cache_root,
            "PipeFS recovery runtime state escapes its validated cache"
        );
        let database = state_root.join("events.sqlite3");
        let metadata = match fs::symlink_metadata(&database) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "PipeFS recovery control journal is not a regular file: {}",
            database.display()
        );
        let store = ControlStore::open_for_state(&state_root)
            .with_context(|| format!("opening recovery control journal in cache {}", cache.id))?;
        let evidence = hi_pipefs::recovery_cache_operation_evidence(scope, session_id, &cache.id)?;
        for record in store.recoveries_for_session(session_id)? {
            if record.status == WorkspaceRecoveryStatus::Discarded {
                continue;
            }
            let remote_retry_safe = recovery_matches(&store, &record, evidence.as_ref())?;
            if record.status == WorkspaceRecoveryStatus::Resolved && !remote_retry_safe {
                continue;
            }
            views.push(JournalRecoveryView {
                schema_version: 1,
                session_id: session_id.to_owned(),
                recovery_id: record.recovery_id,
                recovery_cache_id: cache.id.clone(),
                cache_confirmation_digest: cache.confirmation_digest.clone(),
                binding_id: record.binding_id,
                operation_id: record.operation_id,
                job_id: record.job_id,
                kind: record.kind,
                status: record.status,
                digest: record.digest,
                artifact_ref: record.artifact_ref,
                detail: record.detail,
                error: record.error,
                remote_retry_safe,
                safe_resolution: resolution(remote_retry_safe),
            });
        }
    }
    apply_cache_retry_policy(&mut views);
    views.sort_by(|left, right| {
        left.recovery_id
            .cmp(&right.recovery_id)
            .then_with(|| left.recovery_cache_id.cmp(&right.recovery_cache_id))
    });
    Ok(views)
}

fn apply_cache_retry_policy(views: &mut [JournalRecoveryView]) {
    let blocked_caches = views
        .iter()
        .filter(|view| !view.remote_retry_safe)
        .map(|view| view.recovery_cache_id.clone())
        .collect::<BTreeSet<_>>();
    for view in views {
        if blocked_caches.contains(&view.recovery_cache_id) {
            view.remote_retry_safe = false;
            view.safe_resolution = resolution(false);
        }
    }
}

pub(super) fn find_loaded(
    views: &[JournalRecoveryView],
    recovery_id: &str,
) -> Result<Option<JournalRecoveryView>> {
    let mut matches = views.iter().filter(|view| view.recovery_id == recovery_id);
    let found = matches.next();
    if matches.next().is_some() {
        bail!(
            "journal recovery {recovery_id} exists in multiple caches; inspect the listed cache IDs explicitly"
        );
    }
    Ok(found.cloned())
}

pub(super) fn resolve_loaded(
    views: &[JournalRecoveryView],
    requested_id: &str,
    cache_alias_exists: bool,
) -> Result<RecoveryTarget> {
    let mut matching = views.iter().filter(|view| view.recovery_id == requested_id);
    if let Some(view) = matching.next() {
        ensure!(
            matching.next().is_none(),
            "journal recovery {requested_id} exists in multiple caches"
        );
        return Ok(RecoveryTarget {
            requested_id: requested_id.to_owned(),
            cache_id: view.recovery_cache_id.clone(),
            remote_retry_safe: view.remote_retry_safe,
            journal_recovery_id: Some(view.recovery_id.clone()),
        });
    }
    if cache_alias_exists {
        let remote_retry_safe = views
            .iter()
            .filter(|view| view.recovery_cache_id == requested_id)
            .all(|view| view.remote_retry_safe);
        return Ok(RecoveryTarget {
            requested_id: requested_id.to_owned(),
            cache_id: requested_id.to_owned(),
            remote_retry_safe,
            journal_recovery_id: None,
        });
    }
    bail!("recovery {requested_id:?} was not found for this authority and session")
}

pub(super) fn mark_recovered_before_release(
    scope: &hi_pipefs::PipeFsCacheScope,
    session_id: &str,
    cache_id: &str,
    recovery_id: &str,
    operation: &hi_pipefs::CausalOperationReceipt,
    revision: Option<uuid::Uuid>,
    transcript_cursor: u64,
) -> Result<()> {
    let cache = hi_pipefs::inspect_recovery_cache(scope, session_id, cache_id)?;
    let evidence = hi_pipefs::recovery_cache_operation_evidence(scope, session_id, cache_id)?
        .context("recovery cache lost its pending operation evidence")?;
    ensure!(
        evidence == *operation,
        "recovery operation evidence changed before journal settlement"
    );
    let state_root = cache.path.join("runtime-state");
    let store = ControlStore::open_for_state(&state_root)?;
    let recovery = store
        .get_workspace_recovery(recovery_id)?
        .with_context(|| format!("journal recovery {recovery_id} disappeared"))?;
    ensure!(
        recovery.status != WorkspaceRecoveryStatus::Discarded
            && recovery_matches(&store, &recovery, Some(operation))?,
        "journal recovery no longer matches the exact pending operation"
    );
    let binding_id = recovery
        .binding_id
        .as_deref()
        .context("journal recovery has no workspace binding")?;
    let persisted = store
        .get_workspace_binding(binding_id)?
        .with_context(|| format!("workspace binding {binding_id} disappeared"))?;
    ensure!(
        persisted.session_id.as_deref() == Some(session_id)
            && persisted.epoch == operation.binding_epoch,
        "journal binding no longer matches the recovery session and epoch"
    );
    let binding = WorkspaceBinding {
        schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
        controller_id: ControllerId::new(uuid::Uuid::new_v4().to_string()),
        binding_id: BindingId::new(persisted.binding_id),
        workspace_id: WorkspaceId::new(persisted.workspace_id),
        epoch: persisted.epoch,
        workspace_root: cache
            .workspace_root
            .context("recovery cache has no workspace root")?,
        state_root,
        authority: WorkspaceAuthority::PipeFs {
            session_id: session_id.to_owned(),
            writer_protocol: 0,
        },
        version: WorkspaceVersion::PipeFs {
            lease_generation: 0,
            head: revision.map(|revision| revision.to_string()),
            manifest_digest: None,
            transcript_cursor: Some(transcript_cursor),
        },
    };
    WorkspaceProjectionJournal::from_control_store(store).record_recovery_outcome(
        &binding,
        &RecoveryOutcome {
            recovery_id: RecoveryId::new(recovery_id),
            status: RecoveryStatus::Recovered,
            binding: binding.clone(),
            detail: Some("remote workspace and transcript acknowledgements were verified".into()),
        },
    )?;
    Ok(())
}

fn resolution(remote_retry_safe: bool) -> String {
    format!(
        "export or confirmation-discard this recovery ID's entire owning cache; retry is {}",
        if remote_retry_safe {
            "available because every journal fence in the cache has exact operation proof"
        } else {
            "blocked because this cache contains journal work without exact operation proof"
        }
    )
}

fn recovery_matches(
    store: &ControlStore,
    recovery: &WorkspaceRecoveryRecord,
    evidence: Option<&hi_pipefs::CausalOperationReceipt>,
) -> Result<bool> {
    let (Some(operation_id), Some(evidence)) = (&recovery.operation_id, evidence) else {
        return Ok(false);
    };
    let Some(operation) = store.get_workspace_operation(operation_id)? else {
        return Ok(false);
    };
    Ok(operation_matches(&operation, recovery, evidence))
}

fn operation_matches(
    operation: &WorkspaceOperationRecord,
    recovery: &WorkspaceRecoveryRecord,
    evidence: &hi_pipefs::CausalOperationReceipt,
) -> bool {
    let expected_recovery = restart_operation_recovery_id(
        &BindingId::new(evidence.binding_id.clone()),
        evidence.binding_epoch,
        &OperationId::new(evidence.operation_id.clone()),
    );
    recovery.recovery_id == expected_recovery.as_str()
        && recovery.job_id.is_none()
        && recovery.operation_id.as_deref() == Some(evidence.operation_id.as_str())
        && recovery.binding_id.as_deref() == Some(evidence.binding_id.as_str())
        && operation.operation_id == evidence.operation_id
        && operation.binding_id == evidence.binding_id
        && operation.epoch == evidence.binding_epoch
        && operation.idempotency_key == evidence.idempotency_key
        && match (operation.replay_class, &evidence.replay_class) {
            (OperationReplayClass::PureWorkspace, ReplayClass::PureWorkspace)
            | (OperationReplayClass::NonReplayableExternal, ReplayClass::NonReplayableExternal) => {
                true
            }
            (OperationReplayClass::IdempotentExternal, ReplayClass::IdempotentExternal { key }) => {
                key.as_str() == evidence.idempotency_key
            }
            _ => false,
        }
}

pub(super) fn print_summary(view: &JournalRecoveryView) {
    println!("{}", format_summary(view));
}

pub(super) fn format_summary(view: &JournalRecoveryView) -> String {
    let target = view
        .operation_id
        .as_deref()
        .map(|id| format!("operation {id}"))
        .or_else(|| view.job_id.as_deref().map(|id| format!("job {id}")))
        .unwrap_or_else(|| "unknown lifecycle".into());
    format!(
        "- {}: {} ({target}); owning cache {}; whole-cache confirmation {}",
        view.recovery_id,
        view.kind,
        view.recovery_cache_id,
        view.cache_confirmation_digest
            .as_deref()
            .unwrap_or("unavailable")
    )
}

pub(super) fn print_detail(view: &JournalRecoveryView, session_id: &str) {
    println!("{}", format_detail(view, session_id));
}

pub(super) fn format_detail(view: &JournalRecoveryView, session_id: &str) -> String {
    format!(
        "PipeFS journal recovery {}\nsession: {session_id}\nrecovery cache: {}\nwhole-cache discard confirmation: {}\nkind: {}\nstatus: {:?}\noperation: {}\njob: {}\ndetail: {}\nsafe resolution: {}",
        view.recovery_id,
        view.recovery_cache_id,
        view.cache_confirmation_digest
            .as_deref()
            .unwrap_or("unavailable"),
        view.kind,
        view.status,
        view.operation_id.as_deref().unwrap_or("none"),
        view.job_id.as_deref().unwrap_or("none"),
        view.detail.as_deref().unwrap_or("none"),
        view.safe_resolution
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(id: &str, retry_safe: bool) -> JournalRecoveryView {
        JournalRecoveryView {
            schema_version: 1,
            session_id: "session".into(),
            recovery_id: id.into(),
            recovery_cache_id: "cache-alias".into(),
            cache_confirmation_digest: Some("blake3:whole-cache".into()),
            binding_id: Some("binding".into()),
            operation_id: Some(format!("operation-{id}")),
            job_id: None,
            kind: "crashed_foreground_operation".into(),
            status: WorkspaceRecoveryStatus::Required,
            digest: None,
            artifact_ref: None,
            detail: Some("inspect me".into()),
            error: None,
            remote_retry_safe: retry_safe,
            safe_resolution: resolution(retry_safe),
        }
    }

    #[test]
    fn unmatched_sibling_blocks_journal_and_cache_alias_retry_but_keeps_mapping() {
        let mut views = vec![view("matched", true), view("unmatched", false)];
        apply_cache_retry_policy(&mut views);
        assert!(views.iter().all(|view| !view.remote_retry_safe));

        for id in ["matched", "unmatched"] {
            let target = resolve_loaded(&views, id, false).unwrap();
            assert_eq!(target.cache_id, "cache-alias");
            assert!(!target.remote_retry_safe);
        }
        let alias = resolve_loaded(&views, "cache-alias", true).unwrap();
        assert_eq!(alias.cache_id, "cache-alias");
        assert!(!alias.remote_retry_safe);
        assert_eq!(
            views[1].cache_confirmation_digest.as_deref(),
            Some("blake3:whole-cache")
        );
        assert!(views[1].safe_resolution.contains("entire owning cache"));
    }

    #[test]
    fn displayed_journal_id_maps_to_whole_cache_export_and_discard_target() {
        let views = vec![view("stable-recovery-id", false)];
        let target = resolve_loaded(&views, "stable-recovery-id", false).unwrap();
        assert_eq!(target.requested_id, "stable-recovery-id");
        assert_eq!(target.cache_id, "cache-alias");
        let json = serde_json::to_string(&views[0]).unwrap();
        assert!(json.contains("blake3:whole-cache"));
        assert!(json.contains("cache-alias"));
        let summary = format_summary(&views[0]);
        assert!(summary.contains("stable-recovery-id"));
        assert!(summary.contains("owning cache cache-alias"));
        assert!(summary.contains("whole-cache confirmation blake3:whole-cache"));
    }

    #[test]
    fn retry_proof_requires_the_deterministic_operation_recovery_id() {
        let binding_id = "binding";
        let operation_id = "operation";
        let epoch = 7;
        let expected = restart_operation_recovery_id(
            &BindingId::new(binding_id),
            epoch,
            &OperationId::new(operation_id),
        );
        let operation = WorkspaceOperationRecord {
            operation_id: operation_id.into(),
            binding_id: binding_id.into(),
            epoch,
            session_id: Some("session".into()),
            run_id: None,
            attempt_id: None,
            job_id: None,
            kind: "tool".into(),
            replay_class: OperationReplayClass::PureWorkspace,
            status: hi_control::WorkspaceOperationStatus::RecoveryRequired,
            operation_digest: "digest".into(),
            idempotency_key: "key".into(),
            base_version: None,
            result_version: None,
            execution_ref: None,
            settlement_ref: None,
            error: None,
            revision: 1,
            created_at_ms: 1,
            updated_at_ms: 1,
            settled_at_ms: None,
        };
        let mut recovery = WorkspaceRecoveryRecord {
            recovery_id: expected.to_string(),
            binding_id: Some(binding_id.into()),
            workspace_id: "workspace".into(),
            session_id: Some("session".into()),
            operation_id: Some(operation_id.into()),
            job_id: None,
            kind: "crashed_foreground_operation".into(),
            status: WorkspaceRecoveryStatus::Required,
            digest: None,
            artifact_ref: None,
            detail: None,
            error: None,
            revision: 1,
            created_at_ms: 1,
            updated_at_ms: 1,
            resolved_at_ms: None,
        };
        let evidence = hi_pipefs::CausalOperationReceipt {
            operation_id: operation_id.into(),
            idempotency_key: "key".into(),
            binding_id: binding_id.into(),
            binding_epoch: epoch,
            replay_class: ReplayClass::PureWorkspace,
            execution: hi_workspace::ExecutionReport::succeeded(None),
        };
        assert!(operation_matches(&operation, &recovery, &evidence));
        recovery.recovery_id = "legacy-random-journal-id".into();
        assert!(!operation_matches(&operation, &recovery, &evidence));
    }

    #[cfg(unix)]
    #[test]
    fn journal_discovery_rejects_a_symlinked_runtime_state_parent() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let cache_path = temporary.path().join("cache");
        let outside = temporary.path().join("outside");
        fs::create_dir_all(&cache_path).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("events.sqlite3"), b"not a database").unwrap();
        symlink(&outside, cache_path.join("runtime-state")).unwrap();
        let client = hi_pipefs::PipeFsClient::new(hi_pipefs::PipeFsClientConfig::new(
            "https://sync.example",
            "secret",
        ))
        .unwrap();
        let cache = hi_pipefs::PipeFsRecoveryCache {
            id: "cache-alias".into(),
            confirmation_digest: None,
            path: cache_path,
            workspace_root: None,
            phase: None,
            logical_size_bytes: 0,
            pending_archive_bytes: 0,
            last_error: None,
        };

        let error = list(&client.cache_scope(), &[cache], "session").unwrap_err();
        assert!(error.to_string().contains("not a real directory"));
    }
}
