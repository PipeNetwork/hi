use super::*;
use tokio_util::sync::CancellationToken;

pub(super) fn reset(app: &mut App, idx: usize) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.operation_cancel = Some(
        row.workflow_run_id
            .as_deref()
            .and_then(|run_id| app.workflow_runs.get(run_id))
            .map(|run| run.cancel.child_token())
            .unwrap_or_default(),
    );
}

pub(super) fn token(app: &mut App, idx: usize) -> Option<CancellationToken> {
    if app.fleet.get(idx)?.operation_cancel.is_none() {
        reset(app, idx);
    }
    app.fleet.get(idx)?.operation_cancel.clone()
}

pub(super) fn request(app: &mut App, idx: usize) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    if row.state != RowState::Working {
        return;
    }
    if let Some(cancellation) = &row.operation_cancel {
        cancellation.cancel();
    }
    if let Some(kill) = row.kill.take() {
        let _ = kill.send(());
    }
}

pub(super) fn requested(row: &FleetRow) -> bool {
    row.operation_cancel
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
}

/// A late cancellation must still fence the next queued prompt, even when the
/// operation completed successfully just before its UI result was processed.
pub(super) fn settle(app: &mut App, idx: usize) -> bool {
    let Some(row) = app.fleet.get_mut(idx) else {
        return false;
    };
    if !requested(row) && row.workflow_status != Some(WorkflowJobStatus::Cancelled) {
        return false;
    }
    let elapsed = row
        .started
        .take()
        .map(|t| t.elapsed().as_millis() as u64)
        .unwrap_or(0);
    row.state = RowState::Failed;
    row.activity.clear();
    row.kill = None;
    if matches!(row.merge, MergeState::Merged(_)) && !row.changed.is_empty() {
        row.stale = true;
    }
    if row.workflow_status == Some(WorkflowJobStatus::Cancelled) {
        return true;
    }
    row.push_line("⚠ cancelled — candidate and queued replies retained".into());
    let completion = row.workflow_reply.take().map(|reply| {
        row.workflow_status = Some(WorkflowJobStatus::Cancelled);
        let delivered = reply
            .send(Ok(hi_workflow::AgentResult {
                agent_id: format!("#{}", row.id),
                success: false,
                output: serde_json::json!({"summary": "cancelled"}),
                cancelled: true,
                tokens_used: row.usage,
                duration_ms: elapsed,
            }))
            .is_ok();
        WorkflowReplyCompletion {
            run_id: row.workflow_run_id.clone(),
            delivered,
        }
    });
    settle_workflow_reply(app, completion);
    flag_attention(app, idx);
    true
}
