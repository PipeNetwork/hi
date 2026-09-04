use super::*;

impl InMemoryWorkspaceController {
    pub fn new_local_at_epoch_with_job_limits(
        workspace_id: impl Into<WorkspaceId>,
        workspace_root: impl Into<PathBuf>,
        state_root: impl Into<PathBuf>,
        epoch: u64,
        job_limits: JobRegistryLimits,
    ) -> Self {
        assert_valid_job_limits(job_limits);
        let workspace_id = workspace_id.into();
        let workspace_root = workspace_root.into();
        let state_root = state_root.into();
        let inner = Arc::new_cyclic(|weak| {
            let issuer = PermitIssuer::new(Arc::new(AbandonmentHandler {
                inner: weak.clone(),
            }));
            let mut binding = WorkspaceBinding::new_local(
                issuer.controller_id().clone(),
                workspace_id,
                workspace_root,
                state_root,
            );
            binding.epoch = epoch;
            let status = WorkspaceStatus::ready(&binding);
            let (status_tx, _) = watch::channel(status.clone());
            Inner {
                state: Mutex::new(State {
                    binding,
                    capabilities: WorkspaceCapabilities::in_memory(),
                    status,
                    active_operation: None,
                    job_limits,
                    jobs: BTreeMap::new(),
                    recoveries: BTreeMap::new(),
                }),
                status_tx,
                issuer,
            }
        });
        Self { inner }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_pipefs_at_epoch_with_job_limits(
        workspace_id: impl Into<WorkspaceId>,
        session_id: impl Into<String>,
        writer_protocol: u16,
        causal_commit: bool,
        workspace_root: impl Into<PathBuf>,
        state_root: impl Into<PathBuf>,
        epoch: u64,
        job_limits: JobRegistryLimits,
    ) -> Self {
        assert_valid_job_limits(job_limits);
        let workspace_id = workspace_id.into();
        let workspace_root = workspace_root.into();
        let state_root = state_root.into();
        let inner = Arc::new_cyclic(|weak| {
            let issuer = PermitIssuer::new(Arc::new(AbandonmentHandler {
                inner: weak.clone(),
            }));
            let mut binding = WorkspaceBinding::new_pipefs(
                issuer.controller_id().clone(),
                workspace_id,
                session_id.into(),
                writer_protocol,
                workspace_root,
                state_root,
            );
            binding.epoch = epoch;
            let status = WorkspaceStatus::ready(&binding);
            let (status_tx, _) = watch::channel(status.clone());
            Inner {
                state: Mutex::new(State {
                    binding,
                    capabilities: WorkspaceCapabilities::pipefs(causal_commit),
                    status,
                    active_operation: None,
                    job_limits,
                    jobs: BTreeMap::new(),
                    recoveries: BTreeMap::new(),
                }),
                status_tx,
                issuer,
            }
        });
        Self { inner }
    }
}

fn assert_valid_job_limits(limits: JobRegistryLimits) {
    assert!(
        limits.max_preparations > 0 && limits.max_active_jobs > 0,
        "workspace job limits must both be greater than zero"
    );
}
