use std::sync::Arc;

use hi_workspace::{RecoveryKind, WorkspaceState};

use super::PipeFsLeaseStatus;
use super::state::{Inner, lock, publish, require_recovery};

pub(super) fn start(inner: &Arc<Inner>) {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let mut lease_status = inner.session.subscribe_lease_status();
    let weak = Arc::downgrade(inner);
    runtime.spawn(async move {
        loop {
            let observed = *lease_status.borrow_and_update();
            let Some(inner) = weak.upgrade() else {
                return;
            };
            match observed {
                PipeFsLeaseStatus::Valid => reopen_after_refresh(&inner),
                PipeFsLeaseStatus::Uncertain => fence_uncertain(&inner),
                PipeFsLeaseStatus::Lost => {
                    fence_lost(&inner).await;
                    return;
                }
            }
            if lease_status.changed().await.is_err() {
                return;
            }
        }
    });
}

fn reopen_after_refresh(inner: &Inner) {
    let mut state = lock(&inner.state);
    if state.status.state == WorkspaceState::LeaseUncertain
        && state.status.recovery_id.is_none()
        && state.active.is_none()
    {
        state.status.state = WorkspaceState::Ready;
        state.status.detail = None;
        publish(inner, &mut state);
    }
}

fn fence_uncertain(inner: &Inner) {
    let detail = "the shared HI writer lease could not be refreshed; live writers were stopped until authority is proven";
    let mut state = lock(&inner.state);
    if state.status.recovery_id.is_none() && state.status.state != WorkspaceState::LeaseLost {
        state.status.state = WorkspaceState::LeaseUncertain;
        state.status.detail = Some(detail.into());
        publish(inner, &mut state);
    }
}

async fn fence_lost(inner: &Inner) {
    let base_detail = "the shared HI writer lease was taken over by another writer";
    // Close admission before potentially expensive filesystem inspection.
    {
        let mut state = lock(&inner.state);
        state.status.state = if state.status.recovery_id.is_some() {
            WorkspaceState::RecoveryRequired
        } else {
            WorkspaceState::LeaseLost
        };
        state.status.detail = Some(base_detail.into());
        publish(inner, &mut state);
    }
    let marker_error = inner
        .workspace
        .mark_lease_lost(base_detail)
        .await
        .err()
        .map(|error| format!("; recovery marker failed: {error:#}"));
    let recovery_required =
        marker_error.is_some() || inner.workspace.lease_loss_recovery_required().await;
    let detail = format!("{base_detail}{}", marker_error.unwrap_or_default());
    let mut state = lock(&inner.state);
    if let Some(operation) = state.active.clone() {
        require_recovery(
            inner,
            &mut state,
            RecoveryKind::LeaseLost,
            Some(operation),
            None,
            None,
            detail,
        );
    } else if state.status.recovery_id.is_some() {
        state.status.state = WorkspaceState::RecoveryRequired;
        state.status.detail = Some(detail);
        publish(inner, &mut state);
    } else if recovery_required {
        require_recovery(
            inner,
            &mut state,
            RecoveryKind::LeaseLost,
            None,
            None,
            None,
            detail,
        );
    } else {
        state.status.state = WorkspaceState::LeaseLost;
        state.status.detail = Some(detail);
        publish(inner, &mut state);
    }
}
