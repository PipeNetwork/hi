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
        self.turn_open = true;
        self.liveness.reset_turn_tools();
        self.liveness.set_state(HarnessState::AwaitingModel);
        self.liveness.set_turn_index(self.turn_index);
        self.liveness
            .set_pre_checkpoint(pending.pre_checkpoint.clone());
        self.liveness.note_progress();
        self.liveness.emit(EventCode::TurnStart, None, None, None);

        let persisted_before_user = self.messages.len().saturating_sub(1);
        let mut persisted_before = self.messages.len();
        if let Some(session) = &mut self.session {
            let Some(last) = self.messages.last() else {
                return self.messages.len();
            };
            if session.record_turn_start(last, &pending).is_err() {
                // Do not report `SessionAppendFailed` here: that invariant
                // auto-repairs and would kill a turn that can still persist
                // on the next append or at close.
                if session.last_recorded_user_text().as_deref() != Some(input) {
                    persisted_before = persisted_before_user;
                }
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
            oneshot: self.turn_oneshot,
            plain: self.turn_plain,
        };
        let _ = hi_liveness::write_turn_intent_from_env(&intent);
        persisted_before
    }

    pub(crate) fn close_persisted_turn(&mut self, from: usize, reason: TurnStopReason) {
        self.turn_open = false;
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
            let had_messages = from < self.messages.len();
            let failed = session.record_messages(&self.messages[from..]).is_err()
                || session.record_usage(self.session_usage).is_err()
                || session.record_checkpoints(&self.checkpoints).is_err();
            if failed {
                hi_liveness::report_invariant(&self.liveness, InvariantCode::SessionAppendFailed);
            } else {
                self.liveness.note_progress();
                if had_messages {
                    self.liveness
                        .emit(EventCode::SessionAppend, None, None, None);
                }
            }
        }
    }

    /// Fsync messages added since `persisted_before` so a mid-turn crash still
    /// has a valid transcript to resume. Callers must only pass a complete
    /// round (every `tool_call` already has a result). Also rewrites
    /// `PendingTurn` so a failed turn-start persist can still resume.
    /// Advances the cursor on success. A failed write is retried on the next
    /// call and at close — do not report `SessionAppendFailed` here: that
    /// invariant auto-repairs and would kill a turn that can still persist.
    pub(crate) fn persist_turn_progress(&mut self, persisted_before: &mut usize) {
        if *persisted_before >= self.messages.len() {
            return;
        }
        let pending = self.pending_turn.clone();
        if let Some(session) = &mut self.session {
            if session
                .record_messages(&self.messages[*persisted_before..])
                .is_ok()
            {
                if let Some(pending) = pending.as_ref() {
                    let _ = session.record_pending_turn(pending);
                }
                self.liveness.note_progress();
                self.liveness
                    .emit(EventCode::SessionAppend, None, None, None);
                *persisted_before = self.messages.len();
            }
        } else {
            *persisted_before = self.messages.len();
        }
    }
}

pub struct AwaitingUserGuard {
    liveness: hi_liveness::Publisher,
}

impl Drop for AwaitingUserGuard {
    fn drop(&mut self) {
        self.liveness.set_state(HarnessState::Idle);
    }
}

impl Harness {
    pub fn awaiting_user(&self) -> AwaitingUserGuard {
        self.liveness.set_state(HarnessState::AwaitingUser);
        AwaitingUserGuard {
            liveness: self.liveness.clone(),
        }
    }
}

pub(crate) fn tool_fingerprint(name: &str, arguments: &str) -> String {
    let canonical = serde_json::from_str::<serde_json::Value>(arguments)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| arguments.to_string());
    format!("{name}:{}", blake3::hash(canonical.as_bytes()).to_hex())
}
