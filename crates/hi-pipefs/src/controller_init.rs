use super::*;

impl PipeFsWorkspaceController {
    pub async fn new(
        workspace: PipeFsWorkspace,
        session: Arc<dyn PipeFsSessionBridge>,
        config: PipeFsControllerConfig,
    ) -> Self {
        Self::new_with_job_limits(workspace, session, config, JobRegistryLimits::default()).await
    }

    pub async fn new_with_job_limits(
        workspace: PipeFsWorkspace,
        session: Arc<dyn PipeFsSessionBridge>,
        config: PipeFsControllerConfig,
        job_limits: JobRegistryLimits,
    ) -> Self {
        let mode = config.writer_mode();
        let remote_status = workspace.status().await;
        let persisted = workspace.persisted_causal_recovery().await;
        let persisted_compatibility = workspace.persisted_compatibility_recovery().await;
        let inner = Arc::new_cyclic(|weak| {
            let issuer = PermitIssuer::new(Arc::new(AbandonmentHandler {
                inner: weak.clone(),
            }));
            let mut binding = WorkspaceBinding::new_pipefs(
                issuer.controller_id().clone(),
                config.workspace_id,
                config.session_id,
                config.writer_protocol,
                config.workspace_root,
                config.state_root,
            );
            binding.epoch = config.epoch;
            binding.version = pipefs_version(&remote_status, remote_status.transcript_cursor);
            let mut status = WorkspaceStatus::ready(&binding);
            let mut recoveries = BTreeMap::new();
            if let Some(pending) = persisted {
                let transcript_pending = pending.receipt.is_some();
                let fence_valid = pending.operation.has_valid_recovery_fence();
                let incompatible = mode != PipeFsWriterMode::Causal || !fence_valid;
                let operation = MutationPermitRecord {
                    schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
                    controller_id: issuer.controller_id().clone(),
                    operation_id: OperationId::new(pending.operation.operation_id.clone()),
                    idempotency_key: hi_workspace::IdempotencyKey::new(
                        pending.operation.idempotency_key.clone(),
                    ),
                    binding_id: pending.operation.binding_id.clone().into(),
                    epoch: pending.operation.binding_epoch,
                    base_version: binding.version.clone(),
                    intent: MutationIntent {
                        effect_scope: hi_workspace::EffectScope::LiveWriter,
                        replay_class: pending.operation.replay_class.clone(),
                        dirty_paths: (!pending.operation.execution.changed_paths.is_empty())
                            .then(|| pending.operation.execution.changed_paths.clone()),
                        description: Some("restored pending PipeFS causal operation".into()),
                    },
                    issued_at_ms: now_ms(),
                };
                let recovery_id = hi_workspace::restart_operation_recovery_id(
                    &operation.binding_id,
                    operation.epoch,
                    &operation.operation_id,
                );
                let detail = if !fence_valid {
                    "pending causal recovery lacks a valid binding epoch fence"
                } else if incompatible {
                    "pending causal recovery requires writer protocol 2 and causal_commit_v1"
                } else if transcript_pending {
                    "remote commit landed; transcript acknowledgement is pending"
                } else {
                    "causal PipeFS operation requires remote reconciliation"
                };
                let record = RecoveryRecord {
                    schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
                    recovery_id: recovery_id.clone(),
                    kind: if incompatible {
                        RecoveryKind::IncompatibleState
                    } else if transcript_pending {
                        RecoveryKind::TranscriptPending
                    } else {
                        RecoveryKind::UnsettledMutation
                    },
                    binding_id: operation.binding_id.clone(),
                    epoch: operation.epoch,
                    operation_id: Some(operation.operation_id.clone()),
                    job_id: None,
                    detail: detail.into(),
                    created_at_ms: now_ms(),
                    resolved: false,
                };
                recoveries.insert(
                    recovery_id.clone(),
                    RecoveryEntry {
                        record,
                        operation: Some(operation),
                        execution: Some(pending.operation.execution),
                        batch: Some(CausalTranscriptBatch {
                            records: pending.transcript_records,
                        }),
                    },
                );
                status.state = if incompatible {
                    WorkspaceState::Incompatible
                } else if transcript_pending {
                    WorkspaceState::TranscriptPending
                } else {
                    WorkspaceState::PendingRemote
                };
                status.recovery_id = Some(recovery_id);
                status.detail = Some(detail.into());
            } else if let Some(pending) = persisted_compatibility {
                let fence_valid = pending.operation.has_valid_recovery_fence();
                let incompatible = mode != PipeFsWriterMode::Compatibility || !fence_valid;
                let operation = MutationPermitRecord {
                    schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
                    controller_id: issuer.controller_id().clone(),
                    operation_id: OperationId::new(pending.operation.operation_id.clone()),
                    idempotency_key: hi_workspace::IdempotencyKey::new(
                        pending.operation.idempotency_key.clone(),
                    ),
                    binding_id: pending.operation.binding_id.clone().into(),
                    epoch: pending.operation.binding_epoch,
                    base_version: binding.version.clone(),
                    intent: MutationIntent {
                        effect_scope: hi_workspace::EffectScope::LiveWriter,
                        replay_class: pending.operation.replay_class.clone(),
                        dirty_paths: (!pending.operation.execution.changed_paths.is_empty())
                            .then(|| pending.operation.execution.changed_paths.clone()),
                        description: Some("restored pending PipeFS compatibility operation".into()),
                    },
                    issued_at_ms: now_ms(),
                };
                let recovery_id = hi_workspace::restart_operation_recovery_id(
                    &operation.binding_id,
                    operation.epoch,
                    &operation.operation_id,
                );
                let detail = if !fence_valid {
                    "pending compatibility recovery lacks a valid binding epoch fence"
                } else if incompatible {
                    "pending compatibility recovery requires writer protocol 1 fallback"
                } else if pending.remote_commit_landed {
                    "remote commit landed; compatibility transcript acknowledgement is pending"
                } else {
                    "compatibility PipeFS operation requires remote reconciliation"
                };
                let record = RecoveryRecord {
                    schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
                    recovery_id: recovery_id.clone(),
                    kind: if incompatible {
                        RecoveryKind::IncompatibleState
                    } else if pending.remote_commit_landed {
                        RecoveryKind::TranscriptPending
                    } else {
                        RecoveryKind::UnsettledMutation
                    },
                    binding_id: operation.binding_id.clone(),
                    epoch: operation.epoch,
                    operation_id: Some(operation.operation_id.clone()),
                    job_id: None,
                    detail: detail.into(),
                    created_at_ms: now_ms(),
                    resolved: false,
                };
                recoveries.insert(
                    recovery_id.clone(),
                    RecoveryEntry {
                        record,
                        operation: Some(operation),
                        execution: Some(pending.operation.execution),
                        batch: None,
                    },
                );
                status.state = if incompatible {
                    WorkspaceState::Incompatible
                } else if pending.remote_commit_landed {
                    WorkspaceState::TranscriptPending
                } else {
                    WorkspaceState::PendingRemote
                };
                status.recovery_id = Some(recovery_id);
                status.detail = Some(detail.into());
            }
            let (status_tx, _) = watch::channel(status.clone());
            Inner {
                workspace,
                session,
                mode,
                issuer,
                jobs: WorkspaceJobRegistry::with_limits(binding.clone(), job_limits)
                    .expect("resolved PipeFS job limits are valid"),
                state: Mutex::new(State {
                    binding,
                    status,
                    active: None,
                    recoveries,
                }),
                status_tx,
            }
        });
        let controller = Self { inner };
        controller.start_lease_monitor();
        controller
    }
}
