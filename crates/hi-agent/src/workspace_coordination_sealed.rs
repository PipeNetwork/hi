//! Exact workspace-binding validation at mutation admission.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, bail};
use hi_workspace::{
    ExecutionDisposition, ExecutionReport, MutationIntent, MutationPermit, WorkspaceAuthority,
    WorkspaceBinding, WorkspaceController, WorkspaceState,
};

use super::{ActiveMutation, WorkspaceCoordination};
use crate::WorkspaceDurability;

/// A provider response was sealed against a workspace state which no longer
/// matches the mutation permit that would authorize its effects.
///
/// This concrete error lets tool dispatch use the same bounded
/// `stale_workspace` retry as the earlier envelope precheck.
#[derive(Debug)]
pub(crate) struct SealedWorkspaceAdmissionError {
    message: String,
}

impl SealedWorkspaceAdmissionError {
    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for SealedWorkspaceAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SealedWorkspaceAdmissionError {}

impl WorkspaceCoordination {
    pub(crate) async fn begin(
        &self,
        durability: Option<Arc<dyn WorkspaceDurability>>,
        dirty_paths: Option<Vec<String>>,
    ) -> Result<()> {
        let mut intent = MutationIntent::workspace("tool or lifecycle workspace mutation");
        intent.dirty_paths = dirty_paths.map(|paths| paths.into_iter().map(Into::into).collect());
        self.begin_intent(durability, intent).await
    }

    pub(crate) async fn begin_intent(
        &self,
        durability: Option<Arc<dyn WorkspaceDurability>>,
        intent: MutationIntent,
    ) -> Result<()> {
        self.begin_intent_inner(durability, intent, None).await
    }

    /// Admit a provider-request mutation only if the authority-bearing state
    /// sealed into that exact request remains current inside the rebind gate.
    /// The issued permit is checked before effects too, closing a version
    /// change during the controller's asynchronous `begin` implementation.
    pub(crate) async fn begin_sealed_intent(
        &self,
        durability: Option<Arc<dyn WorkspaceDurability>>,
        intent: MutationIntent,
        sealed: &hi_tools::envelope::WorkspaceEnvelope,
    ) -> Result<()> {
        self.begin_intent_inner(durability, intent, Some(sealed))
            .await
    }

    async fn begin_intent_inner(
        &self,
        mut durability: Option<Arc<dyn WorkspaceDurability>>,
        intent: MutationIntent,
        sealed: Option<&hi_tools::envelope::WorkspaceEnvelope>,
    ) -> Result<()> {
        let _admission = self.acquire_admission().await;
        if self.lock_active()?.is_some() {
            bail!("a workspace mutation is already admitted and awaiting settlement");
        }
        let controller = self.controller();
        if let Some(sealed) = sealed {
            ensure_sealed_workspace_current(sealed, &controller.binding())?;
        }
        let dirty_paths = intent.dirty_paths.as_ref().map(|paths| {
            paths
                .iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
        });
        if !self.harness.features.workspace_controller_v2 {
            let status = controller.status();
            if status.recovery_id.is_some()
                || !matches!(
                    status.state,
                    WorkspaceState::Ready | WorkspaceState::LeaseUncertain
                )
            {
                bail!(
                    "workspace recovery remains required while controller-v2 admission is disabled: {:?}",
                    status.state
                );
            }
            match durability.as_ref() {
                Some(durability) => durability
                    .mutation_started(dirty_paths)
                    .await
                    .context("legacy workspace mutation admission failed")?,
                None if matches!(
                    controller.binding().authority,
                    WorkspaceAuthority::PipeFs { .. }
                ) =>
                {
                    bail!(
                        "PipeFS mutation admission requires the legacy durability fence while controller-v2 admission is disabled"
                    )
                }
                None => {}
            }
            // A legacy PipeFS admission can observe a lease/head change while
            // awaiting its backend. Do not execute against the old request;
            // leave any pending backend evidence intact for recovery.
            if let Some(sealed) = sealed {
                ensure_sealed_workspace_current(sealed, &controller.binding())?;
            }
            return Ok(());
        }
        if self.controller_settles_backend.load(Ordering::Acquire) {
            durability = None;
        }
        let permit = controller.begin(intent).await?;
        let permit = match sealed
            .and_then(|sealed| sealed_permit_mismatch(sealed, &controller.binding(), &permit))
        {
            Some(error) => {
                return Err(self.settle_rejected_permit(controller, permit, error).await);
            }
            None => permit,
        };
        if let Some(durability) = durability.as_ref()
            && let Err(error) = durability.mutation_started(dirty_paths).await
        {
            let report = no_effect_report(format!(
                "mutation admission backend failed before execution: {error:#}"
            ));
            let _ = controller.settle(permit, report).await;
            return Err(
                error.context("workspace mutation admission backend rejected the operation")
            );
        }
        let permit = match sealed
            .and_then(|sealed| sealed_permit_mismatch(sealed, &controller.binding(), &permit))
        {
            Some(error) => {
                return Err(self.settle_rejected_permit(controller, permit, error).await);
            }
            None => permit,
        };
        *self.lock_active()? = Some(ActiveMutation { controller, permit });
        Ok(())
    }

    async fn settle_rejected_permit(
        &self,
        controller: Arc<dyn WorkspaceController>,
        permit: MutationPermit,
        error: SealedWorkspaceAdmissionError,
    ) -> anyhow::Error {
        let settlement = self
            .settle_owned(
                ActiveMutation { controller, permit },
                None,
                no_effect_report(
                    "sealed workspace changed during admission; no tool effect was started",
                ),
            )
            .await;
        let message = match settlement {
            Ok(()) => error.message,
            Err(settlement_error) => format!(
                "{}; rejecting permit also failed to reach a known settlement state: {settlement_error:#}",
                error.message
            ),
        };
        SealedWorkspaceAdmissionError { message }.into()
    }
}

pub(crate) fn sealed_workspace_mismatch(
    sealed: &hi_tools::envelope::WorkspaceEnvelope,
    current: &WorkspaceBinding,
) -> Option<SealedWorkspaceAdmissionError> {
    let current = hi_tools::envelope::WorkspaceEnvelope::from(current);
    if sealed == &current {
        return None;
    }
    Some(SealedWorkspaceAdmissionError {
        message: format!(
            "workspace changed after this model request was sealed; discard the call and retry with a fresh request (sealed authority={:?}, binding={}, epoch={}, version={:?}; current authority={:?}, binding={}, epoch={}, version={:?})",
            sealed.authority,
            sealed.binding_id,
            sealed.epoch,
            sealed.version,
            current.authority,
            current.binding_id,
            current.epoch,
            current.version,
        ),
    })
}

fn ensure_sealed_workspace_current(
    sealed: &hi_tools::envelope::WorkspaceEnvelope,
    current: &WorkspaceBinding,
) -> Result<()> {
    sealed_workspace_mismatch(sealed, current).map_or(Ok(()), |error| Err(error.into()))
}

fn sealed_permit_mismatch(
    sealed: &hi_tools::envelope::WorkspaceEnvelope,
    current: &WorkspaceBinding,
    permit: &MutationPermit,
) -> Option<SealedWorkspaceAdmissionError> {
    if let Some(error) = sealed_workspace_mismatch(sealed, current) {
        return Some(error);
    }
    let record = permit.record();
    if record.schema_version == hi_workspace::WORKSPACE_CONTRACT_SCHEMA_VERSION
        && record.controller_id == current.controller_id
        && record.binding_id.as_str() == sealed.binding_id
        && record.epoch == sealed.epoch
        && record.base_version == sealed.version
    {
        return None;
    }
    Some(SealedWorkspaceAdmissionError {
        message: format!(
            "workspace changed while this model request was being admitted; discard the call and retry with a fresh request (sealed binding={}, epoch={}, version={:?}; issued binding={}, epoch={}, version={:?})",
            sealed.binding_id,
            sealed.epoch,
            sealed.version,
            record.binding_id,
            record.epoch,
            record.base_version,
        ),
    })
}

fn no_effect_report(detail: impl Into<String>) -> ExecutionReport {
    ExecutionReport {
        disposition: ExecutionDisposition::Failed,
        workspace_may_have_changed: false,
        external_effect_may_have_occurred: false,
        content_digest: None,
        changed_paths: Vec::new(),
        artifacts: Vec::new(),
        detail: Some(detail.into()),
    }
}
