use super::*;
use tokio_util::sync::CancellationToken;

#[cfg(test)]
mod tests;

/// Refresh only after the combined tree passes. A failed check must leave the
/// original candidate available for recovery, even if the verifier changed the
/// already-merged workspace before it failed.
pub(super) fn queue(
    app: &mut App,
    idx: usize,
    launcher: &FleetLauncher,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    let cancellation = cancellation::token(app, idx);
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    if row.workflow_status == Some(WorkflowJobStatus::Cancelled) {
        row.state = RowState::Failed;
        row.activity.clear();
        return;
    }
    let worktree = row.worktree.clone();
    let workspace = app.workspace_root.clone();
    let verify = launcher.verify.clone();
    row.state = RowState::Working;
    row.activity = "post-merge check…".into();
    in_flight.push(Box::pin(async move {
        (idx, check(workspace, worktree, verify, cancellation).await)
    }));
}

async fn check(
    workspace: PathBuf,
    candidate: PathBuf,
    verify: Option<String>,
    cancellation: Option<CancellationToken>,
) -> RowDone {
    let mut verify_ok = None;
    let new_base = async {
        ensure_running(cancellation.as_ref())?;
        if let Some(verify) = verify {
            verify_ok = Some(
                worktree::verify_passes_async(&workspace, &verify, cancellation.as_ref()).await,
            );
        }
        ensure_running(cancellation.as_ref())?;
        if verify_ok == Some(false) {
            return Err("combined-tree verification failed after merge".into());
        }
        let checkpoint = hi_tools::checkpoint::create_detailed(&workspace);
        let captured = if let Some(cancellation) = &cancellation {
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => return Err("post-merge check cancelled".into()),
                captured = checkpoint => captured,
            }
        } else {
            checkpoint.await
        };
        let base = match captured {
            hi_tools::checkpoint::CreateResult::Created(base) => base,
            hi_tools::checkpoint::CreateResult::Unavailable(error)
            | hi_tools::checkpoint::CreateResult::Failed(error) => {
                return Err(format!("could not snapshot the merged workspace: {error}"));
            }
        };
        ensure_running(cancellation.as_ref())?;
        tokio::task::spawn_blocking(move || {
            ensure_running(cancellation.as_ref())?;
            worktree::reset_to(&candidate, &base)
                .map_err(|error| format!("could not refresh the candidate base: {error:#}"))?;
            Ok(base)
        })
        .await
        .map_err(|error| format!("candidate refresh worker failed: {error}"))?
    }
    .await;
    RowDone::PostVerify {
        verify_ok,
        new_base,
    }
}

fn ensure_running(cancellation: Option<&CancellationToken>) -> Result<(), String> {
    if cancellation.is_some_and(CancellationToken::is_cancelled) {
        Err("post-merge check cancelled".into())
    } else {
        Ok(())
    }
}

/// Only a confirmed refresh may advance the row's base or release queued work.
/// A failed refresh leaves both the candidate and pending prompts parked.
pub(super) fn finish(
    app: &mut App,
    idx: usize,
    verify_ok: Option<bool>,
    new_base: Result<String, String>,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.started = None;
    row.activity.clear();
    let failure = match new_base {
        Ok(base) => {
            // Even cancellation racing a completed reset must retain its
            // confirmed base so a later explicit resume cannot use stale data.
            row.base = base;
            row.changed.clear();
            row.stale = false;
            (verify_ok == Some(false))
                .then(|| "combined-tree verification failed after merge".to_string())
        }
        Err(error) => {
            row.stale = true;
            Some(error)
        }
    };
    if cancellation::settle(app, idx) {
        return;
    }
    let row = &mut app.fleet[idx];
    if let Some(error) = failure {
        row.state = RowState::Failed;
        let message = format!(
            "{error}; changes are already merged — inspect your tree; candidate retained at {}. Refresh its base (r) before resuming queued work",
            row.worktree.display()
        );
        row.push_line(format!("⚠ {message}"));
        record_fleet(launcher, row.id, &row.title, &message);
        let completion = finish_workflow_agent(row, false, message);
        settle_workflow_reply(app, completion);
        flag_attention(app, idx);
        return;
    }
    row.state = RowState::Idle;
    let completion = finish_workflow_agent(
        row,
        true,
        "verified changes merged into the workspace".into(),
    );
    if !settle_workflow_reply(app, completion) {
        continue_row(app, idx, launcher, line_tx, in_flight);
    }
}
