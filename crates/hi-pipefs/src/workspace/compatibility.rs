use super::*;

impl PipeFsWorkspace {
    /// Publish workspace bytes for the legacy CAS + transcript-flush
    /// compatibility protocol without releasing the local archive or recovery
    /// marker. The controller must follow with
    /// [`finish_compatibility_checkpoint`](Self::finish_compatibility_checkpoint)
    /// only after `flush_through` has returned an acknowledged cursor.
    pub async fn checkpoint_for_compatibility_transcript(
        &self,
        operation: CausalOperationReceipt,
    ) -> Result<Option<Uuid>> {
        let mut state = self.inner.state.lock().await;
        ensure!(
            state.pending_causal.is_none(),
            "a causal PipeFS operation is awaiting transcript acknowledgement"
        );
        if let Some(pending_operation) = &state.pending_compatibility {
            ensure!(
                pending_operation.operation == operation,
                "compatibility PipeFS retry does not match the persisted operation"
            );
            if state.phase == WorkspacePhase::Pending {
                if let Some(pending) = &state.pending
                    && state.remote.as_ref().is_some_and(|remote| {
                        remote.current_head.is_some()
                            && remote.manifest_digest.as_deref()
                                == Some(pending.manifest_digest.as_str())
                            && remote.logical_size_bytes == pending.logical_size_bytes
                    })
                {
                    return Ok(state.remote.as_ref().and_then(|remote| remote.current_head));
                }
                if state.pending.is_none() {
                    return Ok(state.remote.as_ref().and_then(|remote| remote.current_head));
                }
            }
        } else {
            let background_terminal_generation = covered_background_terminal_generation(&state);
            let minimum_transcript_cursor = state
                .transcript_cursor
                .unwrap_or_default()
                .checked_add(1)
                .ok_or_else(|| anyhow!("PipeFS compatibility transcript cursor overflow"))?;
            state.pending_compatibility = Some(PendingCompatibilityOperation {
                operation,
                minimum_transcript_cursor: Some(minimum_transcript_cursor),
                background_terminal_generation,
            });
            write_private(
                &self.inner.recovery_marker,
                b"compatibility workspace operation requires acknowledgement\n",
            )?;
            self.persist_locked(&state)?;
        }
        if state.phase == WorkspacePhase::Pending {
            return self
                .retry_locked(&mut state, true)
                .await
                .map(|(head, _)| Some(head));
        }
        if !self.stage_checkpoint_locked(&mut state).await? {
            state.phase = WorkspacePhase::Pending;
            state.last_error =
                Some("workspace unchanged; transcript acknowledgement pending".into());
            write_private(
                &self.inner.recovery_marker,
                b"compatibility operation requires transcript acknowledgement\n",
            )?;
            self.persist_locked(&state)?;
            return Ok(state.remote.as_ref().and_then(|remote| remote.current_head));
        }
        self.retry_locked(&mut state, true)
            .await
            .map(|(head, _)| Some(head))
    }

    /// Complete the compatibility publication only after the transcript
    /// outbox has durably accepted the server cursor.
    pub async fn finish_compatibility_checkpoint(
        &self,
        operation_id: &str,
        transcript_cursor: u64,
    ) -> Result<()> {
        let mut state = self.inner.state.lock().await;
        let pending = state.pending_compatibility.as_ref().ok_or_else(|| {
            anyhow!("PipeFS has no compatibility operation awaiting transcript acknowledgement")
        })?;
        ensure!(
            pending.operation.operation_id == operation_id,
            "compatibility transcript acknowledgement does not match the pending operation"
        );
        let minimum_transcript_cursor = pending.minimum_transcript_cursor.ok_or_else(|| {
            anyhow!(
                "compatibility transcript acknowledgement has no persisted cursor fence; recovery evidence retained"
            )
        })?;
        let previous_transcript_cursor = state.transcript_cursor.unwrap_or_default();
        ensure!(
            transcript_cursor >= minimum_transcript_cursor
                && transcript_cursor >= previous_transcript_cursor,
            "compatibility transcript cursor {transcript_cursor} does not advance the pending operation beyond cursor {previous_transcript_cursor}; recovery evidence retained"
        );
        let background_terminal_generation = pending.background_terminal_generation;
        state.pending = None;
        state.pending_compatibility = None;
        state.transcript_cursor = Some(transcript_cursor);
        state.phase = WorkspacePhase::Clean;
        state.dirty_paths.clear();
        acknowledge_covered_background_terminals(&mut state, background_terminal_generation);
        state.last_error = None;
        self.persist_locked(&state)?;
        let _ = fs::remove_file(&self.inner.pending_archive);
        self.clear_recovery_marker_if_safe(&state);
        Ok(())
    }
}
