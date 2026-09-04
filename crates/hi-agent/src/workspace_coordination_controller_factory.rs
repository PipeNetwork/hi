use super::*;

pub(super) fn local_controller(
    workspace_root: &Path,
    state_root: &Path,
    epoch: u64,
    job_limits: hi_workspace::JobRegistryLimits,
) -> Arc<dyn WorkspaceController> {
    let workspace_id = workspace_id(workspace_root);
    let store = match hi_control::ControlStore::open_for_state(state_root) {
        Ok(store) => store,
        Err(error) => {
            let raw: Arc<dyn WorkspaceController> = Arc::new(
                InMemoryWorkspaceController::new_local_at_epoch_with_job_limits(
                    workspace_id,
                    workspace_root,
                    state_root,
                    epoch,
                    job_limits,
                ),
            );
            return Arc::new(
                hi_control::JournaledWorkspaceController::local_without_store(
                    raw,
                    error.to_string(),
                )
                .expect("local degraded controller accepts an unavailable journal"),
            );
        }
    };
    match build_local_journaled_controller(
        workspace_id.clone(),
        workspace_root,
        state_root,
        epoch,
        store,
        job_limits,
    ) {
        Ok(controller) => controller,
        Err(error) => {
            let raw: Arc<dyn WorkspaceController> = Arc::new(
                InMemoryWorkspaceController::new_local_at_epoch_with_job_limits(
                    workspace_id,
                    workspace_root,
                    state_root,
                    epoch,
                    job_limits,
                ),
            );
            Arc::new(
                hi_control::JournaledWorkspaceController::local_without_store(
                    raw,
                    error.to_string(),
                )
                .expect("local degraded controller accepts an unavailable journal"),
            )
        }
    }
}

fn build_local_journaled_controller(
    workspace_id: String,
    workspace_root: &Path,
    state_root: &Path,
    minimum_epoch: u64,
    store: hi_control::ControlStore,
    job_limits: hi_workspace::JobRegistryLimits,
) -> Result<Arc<dyn WorkspaceController>> {
    let latest = store.latest_workspace_binding(&workspace_id)?;
    let epoch = latest.as_ref().map_or(minimum_epoch, |binding| {
        minimum_epoch.max(binding.epoch.saturating_add(1))
    });
    let raw = Arc::new(
        InMemoryWorkspaceController::new_local_at_epoch_with_job_limits(
            workspace_id.clone(),
            workspace_root,
            state_root,
            epoch,
            job_limits,
        ),
    );
    let journal = hi_control::WorkspaceProjectionJournal::from_control_store(store.clone());
    let historical = store.unsettled_workspace_bindings(&workspace_id)?;
    candidate_recovery::reconcile(&raw, &journal, &store)?;
    seed_restart_recoveries(&raw, &journal, &store, &historical)?;
    let inner: Arc<dyn WorkspaceController> = raw;
    Ok(Arc::new(
        hi_control::JournaledWorkspaceController::attach_store(inner, store)?,
    ))
}

pub(super) fn pipefs_controller(
    session_id: &str,
    writer_protocol: u16,
    causal_commit: bool,
    workspace_root: &Path,
    state_root: &Path,
    minimum_epoch: u64,
    job_limits: hi_workspace::JobRegistryLimits,
) -> Result<Arc<dyn WorkspaceController>> {
    let store = hi_control::ControlStore::open_for_state(state_root)
        .context("opening the PipeFS control journal")?;
    let latest = store.latest_pipefs_binding(session_id)?;
    let epoch = latest.as_ref().map_or(minimum_epoch, |binding| {
        minimum_epoch.max(binding.epoch.saturating_add(1))
    });
    let raw = Arc::new(
        InMemoryWorkspaceController::new_pipefs_at_epoch_with_job_limits(
            workspace_id(workspace_root),
            session_id,
            writer_protocol,
            causal_commit,
            workspace_root,
            state_root,
            epoch,
            job_limits,
        ),
    );
    let journal = hi_control::WorkspaceProjectionJournal::from_control_store(store.clone());
    let historical = store.unsettled_pipefs_bindings(session_id)?;
    candidate_recovery::reconcile(&raw, &journal, &store)?;
    seed_restart_recoveries(&raw, &journal, &store, &historical)?;
    let inner: Arc<dyn WorkspaceController> = raw;
    Ok(Arc::new(
        hi_control::JournaledWorkspaceController::attach_store(inner, store)?,
    ))
}

pub(super) fn resolved_job_limits(
    harness: &hi_workspace::ResolvedHarnessSettings,
) -> hi_workspace::JobRegistryLimits {
    hi_workspace::JobRegistryLimits {
        max_preparations: harness.jobs.max_preparations,
        max_active_jobs: harness.jobs.max_active,
    }
}
