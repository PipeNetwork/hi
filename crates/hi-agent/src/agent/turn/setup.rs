//! Turn setup: task context refresh, checkpoints, snapshots, empty-response nudge.

use std::collections::BTreeSet;

use anyhow::{Context, Result};

use crate::Ui;
use crate::steering::POST_TOOL_EMPTY_RESPONSE_NUDGE;
use crate::transcript::NudgeKind;
use crate::verify::Snapshot;

impl crate::Agent {
    /// Refresh the active task index at most once per context generation.
    /// Workspace edits advance both the ledger and the generation, while a
    /// transcript-only compaction advances only the generation — that
    /// distinction avoids rescanning the repository after compaction.
    ///
    /// This deliberately rewrites **no transcript message** in the common
    /// case: the refreshed memory ranking and task index land in the *next
    /// turn's* volatile context block, and `refresh_system_message` is
    /// hash-gated on the now-stable system content. Keeping every mid-turn
    /// round append-only is what lets provider prompt caches hit round after
    /// round; the model doesn't need a mid-turn index refresh because the
    /// only thing changing the workspace is its own tool calls, whose
    /// results it already sees.
    #[allow(
        clippy::too_many_arguments,
        reason = "context refresh updates the turn-local generation and index markers together"
    )]
    pub(super) async fn refresh_active_task_context(
        &mut self,
        task: &str,
        repository_context_enabled: bool,
        refresh_repository_context: bool,
        turn_ledger_revision: u64,
        ranked_paths: &mut BTreeSet<String>,
        seen_generation: &mut u64,
        indexed_ledger_revision: &mut u64,
    ) -> Result<()> {
        let generation = self.runtime.context_generation();
        let indexed_revision_is_stale = repository_context_refresh_needed(
            generation,
            *seen_generation,
            repository_context_enabled,
            refresh_repository_context,
            self.runtime.ledger().revision(),
            *indexed_ledger_revision,
        );
        // Phase P: memory is cheap and task-dependent — always re-rank for the
        // current prompt even when the context generation hasn't advanced
        // (same ledger, new user task in a multi-turn session).
        self.refresh_memory_context(task);

        // A deferred mid-turn refresh may have consumed the generation marker
        // while intentionally leaving the repository index stale. Completion
        // review explicitly opts into rebuilding that index, so the generation
        // fast path must not hide a stale ledger revision.
        if generation == *seen_generation && !indexed_revision_is_stale {
            // Hash-gated: a no-op unless the stable system content changed.
            self.refresh_system_message();
            return Ok(());
        }

        let (ledger_revision, touched_paths, current_paths) = {
            let ledger = self.runtime.ledger();
            (
                ledger.revision(),
                ledger.touched_paths_since(turn_ledger_revision),
                ledger.changed_paths_since(turn_ledger_revision),
            )
        };
        ranked_paths.extend(touched_paths);
        ranked_paths.extend(current_paths);

        // Repository ranking and indexing walk the workspace synchronously.
        // During model/tool rounds the model already sees its own tool results,
        // so defer that expensive refresh until the next turn. Completion review
        // can opt in when it needs a fresh context block immediately.
        let refreshed_task_context = if refresh_repository_context
            && repository_context_enabled
            && ledger_revision != *indexed_ledger_revision
        {
            // Repository ranking and indexing are synchronous filesystem work.
            // Keep completion review from blocking the async executor or the
            // foreground UI while rebuilding context after a mutation.
            let root = self.runtime.root().to_path_buf();
            let task = task.to_string();
            let exclusions = self.config.memory.context_exclusions.clone();
            let repo_map = self.runtime.repo_map_arc();
            let paths = std::mem::take(ranked_paths);
            let (paths, refreshed) = tokio::task::spawn_blocking(move || {
                let mut paths = paths;
                for path in hi_tools::ranked_paths_for_task(&root, &task, repo_map.as_ref(), 12) {
                    paths.insert(path);
                }
                let selected = paths.iter().cloned().collect::<Vec<_>>();
                let index = crate::context_index::build_task_context_index(
                    &root,
                    &task,
                    &selected,
                    &exclusions,
                );
                let orientation = hi_tools::orientation_for_task(&root, &task, repo_map.as_ref());
                let refreshed = match (orientation, index) {
                    (Some(seed), Some(index)) => Some(format!("{seed}\n\n{index}")),
                    (Some(seed), None) => Some(seed),
                    (None, index) => index,
                };
                (paths, refreshed)
            })
            .await
            .context("task-context refresh worker failed")?;
            *ranked_paths = paths;
            Some(refreshed)
        } else {
            None
        };
        let indexed_repository_context = refreshed_task_context.is_some();
        if let Some(refreshed) = refreshed_task_context
            && refreshed != self.task.task_context
        {
            self.task.set_task_context(refreshed);
        }

        // Hash-gated slot-zero refresh: only fires when the stable system
        // content itself changed (it no longer carries the task index).
        self.refresh_system_message();
        if let Err(err) = self.messages.validate_and_repair_for_provider() {
            anyhow::bail!(
                "Transcript validation failed after context refresh; recovery did not resolve it: {err}"
            );
        }
        *seen_generation = generation;
        // A deferred mid-turn refresh must not mark the repository index as
        // current. Completion review can request the rebuild immediately; if
        // we advanced this marker here, that later opt-in would incorrectly
        // believe the post-mutation index was already fresh.
        if indexed_repository_context {
            *indexed_ledger_revision = ledger_revision;
        }
        Ok(())
    }

    pub(super) async fn reconcile_error_turn_changes(&mut self, turn_revision: u64) -> Result<()> {
        self.reconcile_workspace_changes().await?;
        let changes = self.runtime.ledger().changes_since(turn_revision);
        self.workspace.last_changed_files =
            changes.iter().map(|change| change.path.clone()).collect();
        self.workspace.last_file_changes = changes;
        Ok(())
    }

    pub(super) fn nudge_after_post_tool_empty_response(
        &mut self,
        force_tools_next: &mut bool,
        force_tool_call: bool,
    ) {
        self.messages
            .push_nudge_or_fold(NudgeKind::Continue, POST_TOOL_EMPTY_RESPONSE_NUDGE);
        if force_tool_call {
            *force_tools_next = true;
        }
    }

    pub(super) async fn ensure_turn_checkpoint(
        &mut self,
        checkpoint_allowed: &mut Option<bool>,
        checkpoint_created: &mut bool,
        ui: &mut dyn Ui,
    ) -> bool {
        if let Some(allowed) = *checkpoint_allowed {
            return allowed;
        }

        // Snapshot lazily, immediately before the first mutating tool. YOLO
        // mode means checkpoint failure never asks for permission; it does not
        // mean skipping a recoverable /undo point when one can be created.
        let reason = match hi_tools::checkpoint::create_detailed_with_state(
            self.runtime.root(),
            self.runtime.state_root(),
        )
        .await
        {
            hi_tools::checkpoint::CreateResult::Created(sha) => {
                let mut next = self.workspace.checkpoints.clone();
                next.push(sha);
                if next.len() > crate::MAX_CHECKPOINTS {
                    next.drain(0..next.len() - crate::MAX_CHECKPOINTS);
                }
                if let Some(session) = self.session.as_mut()
                    && let Err(err) = session.record_checkpoints(&next)
                {
                    format!(
                        "checkpoint was created but its reference could not be persisted: {err:#}"
                    )
                } else {
                    self.workspace.checkpoints = next;
                    *checkpoint_created = true;
                    *checkpoint_allowed = Some(true);
                    return true;
                }
            }
            hi_tools::checkpoint::CreateResult::Unavailable(reason)
            | hi_tools::checkpoint::CreateResult::Failed(reason) => reason,
        };
        let allowed = self.config.gates.allow_no_checkpoint;
        *checkpoint_allowed = Some(allowed);
        if !allowed {
            ui.status(&format!(
                "mutation skipped: a checkpoint is required but unavailable: {reason}"
            ));
        }
        allowed
    }

    /// Bind the pre-turn checkpoint to the exact post-turn workspace state.
    /// `/undo` will refuse this record if an editor or another process changes
    /// any tracked path after the turn completes.
    pub(super) async fn seal_turn_checkpoint(&mut self, ui: &mut dyn Ui) -> Result<bool> {
        let Some(target) = self.workspace.checkpoints.last().cloned() else {
            return Ok(false);
        };
        match hi_tools::checkpoint::create_detailed_with_state(
            self.runtime.root(),
            self.runtime.state_root(),
        )
        .await
        {
            hi_tools::checkpoint::CreateResult::Created(expected_current) => {
                let sealed = hi_tools::checkpoint::sealed_reference(&target, &expected_current);
                if let Some(last) = self.workspace.checkpoints.last_mut() {
                    *last = sealed;
                }
                if let Some(session) = self.session.as_mut() {
                    session.record_checkpoints(&self.workspace.checkpoints)?;
                }
                Ok(true)
            }
            hi_tools::checkpoint::CreateResult::Unavailable(reason)
            | hi_tools::checkpoint::CreateResult::Failed(reason) => {
                // An unsealed 0.2 undo record could overwrite edits made after
                // this turn, so always drop it. Strict mode fails the operation;
                // YOLO continues silently and exposes the loss in telemetry.
                self.workspace.checkpoints.pop();
                if let Some(session) = self.session.as_mut() {
                    session.record_checkpoints(&self.workspace.checkpoints)?;
                }
                if !self.config.gates.allow_no_checkpoint {
                    ui.checkpoint_warning(&format!(
                        "⚠ could not seal this turn's undo record: {reason}"
                    ));
                }
                Ok(false)
            }
        }
    }

    pub(super) async fn ensure_turn_snapshot(
        &mut self,
        turn_snapshot: &mut Option<Snapshot>,
    ) -> Result<Snapshot> {
        if let Some(snapshot) = turn_snapshot.as_ref() {
            return Ok(snapshot.clone());
        }
        let snapshot = self.snapshot_cached().await?;
        *turn_snapshot = Some(snapshot.clone());
        Ok(snapshot)
    }
}

fn repository_context_refresh_needed(
    generation: u64,
    seen_generation: u64,
    repository_context_enabled: bool,
    refresh_repository_context: bool,
    ledger_revision: u64,
    indexed_ledger_revision: u64,
) -> bool {
    generation != seen_generation
        || (repository_context_enabled
            && refresh_repository_context
            && ledger_revision != indexed_ledger_revision)
}

#[cfg(test)]
mod tests {
    use super::repository_context_refresh_needed;

    #[test]
    fn completion_refresh_bypasses_seen_generation_when_index_is_stale() {
        assert!(repository_context_refresh_needed(4, 4, true, true, 9, 8));
        assert!(!repository_context_refresh_needed(4, 4, true, true, 8, 8));
        assert!(!repository_context_refresh_needed(4, 4, true, false, 9, 8));
        assert!(repository_context_refresh_needed(5, 4, true, false, 8, 8));
    }
}
