//! Best-effort learning intake after turn settlement.

impl crate::Agent {
    pub(super) fn record_terminal_finding(&self, outcome: &crate::TurnOutcome) {
        // Automatic post-mortem intake: bad outcomes become findings-ledger
        // records so `hi metrics` surfaces failure patterns without anyone
        // spelunking raw transcripts. Best-effort by design.
        if self.config.memory.learning && crate::learning::outcome_warrants_finding(outcome) {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let state_root = self.runtime.state_root().to_path_buf();
            let finding = crate::learning::Finding {
                ts,
                session_id: self.session.as_deref().and_then(crate::SessionSink::id),
                turn: Some(self.turn_count),
                status: outcome.status,
                stop_reason: outcome.stop_reason,
                verification: outcome.verification,
                review: outcome.review,
                review_unavailable_reason: self
                    .report
                    .last_turn_telemetry
                    .review_unavailable_reason
                    .clone(),
                last_no_progress_reason: self
                    .report
                    .last_turn_telemetry
                    .last_no_progress_reason
                    .clone(),
                changed_files: outcome.changed_files.len(),
                model: outcome.effective_route.model.clone(),
                hint_active: self.task.active_hint_shape.clone(),
                failure_shape: crate::learning::tool_failure_shape(
                    &self.report.last_turn_telemetry.tool_timeline,
                ),
            };
            tokio::task::spawn_blocking(move || {
                crate::learning::append_finding(&state_root, &finding);
            });
        }
    }
}
