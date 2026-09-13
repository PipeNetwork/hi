//! Turn-start persist, heartbeat snapshot updates, and invariant reporting.

use hi_liveness::{EventCode, HarnessState, InvariantCode, TurnIntent, unix_ms};

use crate::session::PendingTurn;
use crate::{Harness, TurnStopReason};

impl Harness {
    /// Persist the user line + `PendingTurn` immediately, then return the
    /// message length so finish does not append a second copy.
    pub(crate) fn begin_persisted_turn(&mut self, input: &str, pre: Option<&str>) -> usize {
        self.turn_index = self.turn_index.saturating_add(1);
        let pending = PendingTurn {
            turn_index: self.turn_index,
            started_unix_ms: unix_ms(),
            pre_checkpoint: pre.map(str::to_string),
        };
        self.pending_turn = Some(pending.clone());
        self.liveness.reset_turn_tools();
        self.liveness.set_state(HarnessState::AwaitingModel);
        self.liveness.set_turn_index(self.turn_index);
        self.liveness
            .set_pre_checkpoint(pending.pre_checkpoint.clone());
        self.liveness.note_progress();
        self.liveness.emit(EventCode::TurnStart, None, None, None);

        if let Some(session) = &mut self.session {
            let failed = match self.messages.last() {
                Some(last) => session.record_messages(std::slice::from_ref(last)).is_err(),
                None => false,
            };
            let pending_failed = session.record_pending_turn(&pending).is_err();
            if failed || pending_failed {
                hi_liveness::report_invariant(&self.liveness, InvariantCode::SessionAppendFailed);
            } else {
                self.liveness.note_progress();
                self.liveness
                    .emit(EventCode::SessionAppend, None, None, None);
            }
        }

        let intent = TurnIntent {
            schema_version: hi_liveness::SCHEMA_VERSION,
            turn_index: pending.turn_index,
            prompt: input.to_string(),
            session_path: self
                .session
                .as_ref()
                .map(|session| session.path().display().to_string()),
            pre_checkpoint: pending.pre_checkpoint.clone(),
            started_unix_ms: pending.started_unix_ms,
            workspace: self.workspace_root.display().to_string(),
            oneshot: false,
            plain: false,
        };
        let _ = hi_liveness::write_turn_intent_from_env(&intent);
        self.messages.len()
    }

    pub(crate) fn close_persisted_turn(&mut self, from: usize, reason: TurnStopReason) {
        let turn_index = self.pending_turn.take().map(|pending| pending.turn_index);
        self.persist_new_messages(from);
        if let Some(turn_index) = turn_index
            && let Some(session) = &mut self.session
            && session.record_turn_closed(turn_index).is_err()
        {
            hi_liveness::report_invariant(&self.liveness, InvariantCode::SessionAppendFailed);
        }
        let code = match reason {
            TurnStopReason::Completed => EventCode::TurnEnd,
            TurnStopReason::Cancelled => EventCode::TurnCancel,
            TurnStopReason::Error => EventCode::TurnError,
        };
        self.liveness.emit(code, None, None, None);
        self.liveness.set_state(HarnessState::Idle);
        self.liveness.finish_turn_tools(true);
    }

    pub(crate) fn persist_new_messages(&mut self, from: usize) {
        if let Some(session) = &mut self.session {
            let failed = session.record_messages(&self.messages[from..]).is_err()
                || session.record_usage(self.session_usage).is_err()
                || session.record_checkpoints(&self.checkpoints).is_err();
            if failed {
                hi_liveness::report_invariant(&self.liveness, InvariantCode::SessionAppendFailed);
            } else {
                self.liveness.note_progress();
            }
        }
    }
}

pub(crate) fn tool_fingerprint(name: &str, arguments: &str) -> String {
    let canonical = serde_json::from_str::<serde_json::Value>(arguments)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| arguments.to_string());
    format!("{name}:{}", blake3::hash(canonical.as_bytes()).to_hex())
}
