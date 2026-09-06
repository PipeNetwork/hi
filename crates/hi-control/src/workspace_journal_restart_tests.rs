use super::*;

#[tokio::test]
async fn local_restart_orphans_live_process_and_retires_legacy_recoveries() {
    let (_directory, store) = store();
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        "local-process-workspace",
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach_store(inner, store.clone()).unwrap();
    let job = controller
        .register_job(live_process_spec("development server"))
        .await
        .unwrap();
    let fresh_job = controller
        .register_job(live_process_spec("second development server"))
        .await
        .unwrap();
    let binding = controller.binding();
    let journal = controller.projection_journal();

    // Reproduce the projection written by older controller-v2 builds after
    // they encountered the same live process on startup.
    let mut persisted_job = store.get_job(job.job_id.as_str()).unwrap().unwrap();
    persisted_job.state = ControlJobState::RecoveryRequired;
    persisted_job.revision = persisted_job.revision.saturating_add(1);
    persisted_job.updated_at_ms = hi_events::now_ms();
    journal
        .commit_job(persisted_job, binding.workspace_id.as_str())
        .unwrap();
    let recovery_id =
        hi_workspace::restart_job_recovery_id(&binding.binding_id, binding.epoch, &job.job_id);
    let now = hi_events::now_ms();
    journal
        .commit_recovery(process_recovery(
            &binding,
            &job.job_id,
            &recovery_id,
            "crashed_writer_job",
            now,
        ))
        .unwrap();
    let legacy_recovery_id = hi_workspace::RecoveryId::new("legacy-local-process-crash");
    let mut legacy = process_recovery(
        &binding,
        &job.job_id,
        &legacy_recovery_id,
        "workspace_reconciliation",
        now,
    );
    legacy.digest = Some("legacy-observation".into());
    legacy.artifact_ref = Some("artifact://legacy/process-observation".into());
    legacy.error = Some("old lifecycle error".into());
    journal.commit_recovery(legacy).unwrap();

    let report = journal.reconcile_jobs_after_restart(&binding).unwrap();
    assert_eq!(report.orphaned.len(), 2);
    assert!(report.orphaned.contains(&job.job_id));
    assert!(report.orphaned.contains(&fresh_job.job_id));
    assert!(report.recovery_required.is_empty());
    assert!(report.recovery_ids.is_empty());
    for job_id in [&job.job_id, &fresh_job.job_id] {
        let job = store.get_job(job_id.as_str()).unwrap().unwrap();
        assert_eq!(job.state, ControlJobState::Orphaned);
        assert!(job.finished_at_ms.is_some());
    }
    assert_eq!(
        store
            .get_workspace_recovery(recovery_id.as_str())
            .unwrap()
            .unwrap()
            .status,
        WorkspaceRecoveryStatus::Resolved
    );
    let legacy = store
        .get_workspace_recovery(legacy_recovery_id.as_str())
        .unwrap()
        .unwrap();
    assert_eq!(legacy.status, WorkspaceRecoveryStatus::Resolved);
    assert_eq!(
        legacy.artifact_ref.as_deref(),
        Some("artifact://legacy/process-observation")
    );
    assert!(
        store
            .unsettled_workspace_bindings(binding.workspace_id.as_str())
            .unwrap()
            .is_empty()
    );

    let sequence = store.max_event_sequence().unwrap();
    let repeated = journal.reconcile_jobs_after_restart(&binding).unwrap();
    assert!(repeated.recovery_ids.is_empty());
    assert_eq!(store.max_event_sequence().unwrap(), sequence);
}

#[tokio::test]
async fn local_process_terminal_journal_failure_degrades_without_recovery_fence() {
    let (_directory, store) = store();
    let fault = FaultStore::new(store.clone());
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        "local-process-journal-failure",
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach(inner, Arc::new(fault.clone())).unwrap();
    let job = controller
        .register_job(live_process_spec("development server"))
        .await
        .unwrap();
    let binding = controller.binding();
    assert_eq!(
        controller
            .seal_job(
                job.job_id.clone(),
                JobTerminal {
                    completion: JobCompletion::Settling,
                    detail: None,
                    artifacts: Vec::new(),
                },
            )
            .await
            .status,
        JobSealStatus::Sealed
    );

    fault.fail_next_succeeded_job_commit();
    let outcome = controller
        .seal_job(
            job.job_id.clone(),
            JobTerminal {
                completion: JobCompletion::Succeeded,
                detail: Some("server exited cleanly".into()),
                artifacts: Vec::new(),
            },
        )
        .await;
    assert_eq!(outcome.status, JobSealStatus::Sealed);
    assert!(outcome.recovery_id.is_none());
    assert_eq!(
        controller.status().state,
        WorkspaceState::LocalAuditDegraded
    );
    assert!(
        store
            .recoveries_for_binding(binding.binding_id.as_str())
            .unwrap()
            .is_empty()
    );

    let permit = controller
        .begin(MutationIntent::workspace(
            "foreground work remains available",
        ))
        .await
        .unwrap();
    assert_eq!(
        controller
            .settle(permit, ExecutionReport::succeeded(None))
            .await
            .status,
        SettlementStatus::LocalAuditDegraded
    );

    // The failed final audit write left Settling in SQLite. A healthy restart
    // terminalizes that old process without manufacturing recovery.
    assert_eq!(
        store.get_job(job.job_id.as_str()).unwrap().unwrap().state,
        ControlJobState::Settling
    );
    let report = controller
        .projection_journal()
        .reconcile_jobs_after_restart(&binding)
        .unwrap();
    assert_eq!(report.orphaned, vec![job.job_id.clone()]);
    assert!(report.recovery_ids.is_empty());
    assert_eq!(
        store.get_job(job.job_id.as_str()).unwrap().unwrap().state,
        ControlJobState::Orphaned
    );
}

#[tokio::test]
async fn pipefs_restart_keeps_live_process_recovery_fail_closed() {
    let (_directory, store) = store();
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_pipefs(
        "pipefs-process-workspace",
        "session",
        2,
        true,
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach_store(inner, store.clone()).unwrap();
    let job = controller
        .register_job(live_process_spec("remote writer"))
        .await
        .unwrap();
    let binding = controller.binding();

    let report = controller
        .projection_journal()
        .reconcile_jobs_after_restart(&binding)
        .unwrap();
    assert_eq!(report.recovery_required, vec![job.job_id.clone()]);
    assert_eq!(report.recovery_ids.len(), 1);
    assert_eq!(
        store.get_job(job.job_id.as_str()).unwrap().unwrap().state,
        ControlJobState::RecoveryRequired
    );
    assert_eq!(
        store
            .get_workspace_recovery(report.recovery_ids[0].as_str())
            .unwrap()
            .unwrap()
            .status,
        WorkspaceRecoveryStatus::Required
    );
}

fn process_recovery(
    binding: &hi_workspace::WorkspaceBinding,
    job_id: &hi_workspace::JobId,
    recovery_id: &hi_workspace::RecoveryId,
    kind: &str,
    now: u64,
) -> WorkspaceRecoveryRecord {
    WorkspaceRecoveryRecord {
        recovery_id: recovery_id.to_string(),
        binding_id: Some(binding.binding_id.to_string()),
        workspace_id: binding.workspace_id.to_string(),
        session_id: None,
        operation_id: None,
        job_id: Some(job_id.to_string()),
        kind: kind.into(),
        status: WorkspaceRecoveryStatus::Required,
        digest: None,
        artifact_ref: None,
        detail: Some("old local process recovery fence".into()),
        error: None,
        revision: 1,
        created_at_ms: now,
        updated_at_ms: now,
        resolved_at_ms: None,
    }
}
