//! Opt-in comparison between the established session loaders and reducer v2.
//!
//! The established JSONL/remote reducers remain authoritative during rollout.
//! This hook observes the same records and reports projection drift without
//! changing resume behavior or the legacy wire format.

use super::LoadedSession;
use anyhow::{Result, anyhow};
use std::sync::atomic::{AtomicBool, Ordering};

static REDUCER_V2_ENABLED: AtomicBool = AtomicBool::new(true);
static PROJECTION_V2_ENABLED: AtomicBool = AtomicBool::new(false);

pub(crate) fn configure(reducer_v2: bool, projection_v2: bool) {
    REDUCER_V2_ENABLED.store(reducer_v2, Ordering::Release);
    PROJECTION_V2_ENABLED.store(projection_v2, Ordering::Release);
}

pub(super) struct SessionReducerShadow {
    reducer: Option<hi_agent::SessionReducer>,
    projection: Option<hi_agent::SessionProjection>,
    error: Option<String>,
}

impl SessionReducerShadow {
    pub(super) fn new() -> Self {
        let legacy_override = std::env::var("HI_SESSION_REDUCER_V2_SHADOW")
            .ok()
            .is_some_and(|value| matches!(value.trim(), "1" | "on" | "true" | "yes"));
        Self {
            reducer: (REDUCER_V2_ENABLED.load(Ordering::Acquire) || legacy_override)
                .then(hi_agent::SessionReducer::new),
            projection: PROJECTION_V2_ENABLED
                .load(Ordering::Acquire)
                .then(hi_agent::SessionProjection::new),
            error: None,
        }
    }

    pub(super) fn observe_legacy_json(&mut self, line: &str) {
        if let Some(event) = hi_agent::SessionEvent::from_legacy_json(line) {
            self.observe(event);
        }
    }

    pub(super) fn observe_remote(&mut self, record_type: &str, payload_json: &str) {
        self.observe(hi_agent::SessionEvent::from_remote_record(
            record_type,
            payload_json,
        ));
    }

    pub(super) fn observe_opaque_boundary(&mut self) {
        self.observe(hi_agent::SessionEvent::opaque_boundary());
    }

    fn observe(&mut self, event: hi_agent::SessionEvent) {
        if self.error.is_some() {
            return;
        }
        if let Some(reducer) = self.reducer.as_mut()
            && let Err(error) = reducer.apply(event.clone())
        {
            self.error = Some(error.to_string());
            return;
        }
        if let Some(projection) = self.projection.as_mut() {
            let result = projection
                .prepare_patch(vec![event])
                .and_then(|patch| projection.apply_patch(patch));
            if let Err(error) = result {
                self.error = Some(error.to_string());
            }
        }
    }

    /// Finish replay. Reducer-only mode remains observational; projection mode
    /// is an explicit promotion gate and therefore returns the validated v2
    /// projection as the restored session. Any drift or replay failure fails
    /// closed instead of silently falling back to a different state model.
    pub(super) fn finish(
        self,
        established: LoadedSession,
        source: &'static str,
    ) -> Result<LoadedSession> {
        let projection_promoted = self.projection.is_some();
        let reducer = self
            .projection
            .as_ref()
            .map(hi_agent::SessionProjection::reducer)
            .or(self.reducer.as_ref());
        let Some(reducer) = reducer else {
            return Ok(established);
        };
        if let Some(error) = &self.error {
            if projection_promoted {
                return Err(anyhow!(
                    "session projection v2 {source} replay failed: {error}"
                ));
            }
            tracing::warn!(source, %error, "session reducer v2 shadow replay failed");
            return Ok(established);
        }
        let expected = state_from_loaded(&established);
        if !reducer.state().semantically_eq(&expected) {
            if projection_promoted {
                return Err(anyhow!(
                    "session projection v2 diverged from the established {source} replay"
                ));
            }
            tracing::warn!(
                source,
                reducer_version = hi_agent::SESSION_REDUCER_VERSION,
                through_sequence = reducer.through_sequence(),
                "session reducer v2 shadow projection diverged"
            );
            return Ok(established);
        }
        if projection_promoted {
            let mut loaded = loaded_from_state(reducer.state());
            loaded.harness_settings = established.harness_settings;
            return Ok(loaded);
        }
        Ok(established)
    }
}

fn state_from_loaded(loaded: &LoadedSession) -> hi_agent::SessionState {
    hi_agent::SessionState {
        reducer_version: hi_agent::SESSION_REDUCER_VERSION,
        messages: loaded.messages.clone(),
        usage: loaded.usage,
        checkpoint_refs: loaded.checkpoint_refs.clone(),
        remote_session_id: loaded.remote_session_id.clone(),
        pipefs_enabled: loaded.pipefs_enabled,
        name: loaded.name.clone(),
        goal: loaded.goal.clone(),
        decisions: loaded.decisions.entries().to_vec(),
        plan: loaded.plan.clone(),
        plan_drive_paused: loaded.plan_drive_paused,
        plan_drive_resume_on_user_input: loaded.plan_drive_resume_on_user_input,
        plan_approval_parked: loaded.plan_approval_parked,
        plan_drive_stall: loaded.plan_drive_stall,
        goal_drive_stall: loaded.goal_drive_stall,
        plan_drive_evidence: loaded.plan_drive_evidence.iter().cloned().collect(),
        goal_drive_evidence: loaded.goal_drive_evidence.iter().cloned().collect(),
        transcript_blocks: Vec::new(),
    }
}

fn loaded_from_state(state: &hi_agent::SessionState) -> LoadedSession {
    LoadedSession {
        messages: state.messages.clone(),
        usage: state.usage,
        checkpoint_refs: state.checkpoint_refs.clone(),
        harness_settings: crate::session_harness::empty_layer(),
        remote_session_id: state.remote_session_id.clone(),
        pipefs_enabled: state.pipefs_enabled,
        name: state.name.clone(),
        goal: state.goal.clone(),
        decisions: hi_agent::DecisionLog::from_entries(state.decisions.clone()),
        plan: state.plan.clone(),
        plan_drive_paused: state.plan_drive_paused,
        plan_drive_resume_on_user_input: state.plan_drive_resume_on_user_input,
        plan_approval_parked: state.plan_approval_parked,
        plan_drive_stall: state.plan_drive_stall,
        goal_drive_stall: state.goal_drive_stall,
        plan_drive_evidence: state.plan_drive_evidence.iter().cloned().collect(),
        goal_drive_evidence: state.goal_drive_evidence.iter().cloned().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::Message;

    fn empty_loaded(messages: Vec<Message>) -> LoadedSession {
        LoadedSession {
            messages,
            usage: hi_ai::Usage::default(),
            checkpoint_refs: Vec::new(),
            harness_settings: crate::session_harness::empty_layer(),
            remote_session_id: None,
            pipefs_enabled: None,
            name: None,
            goal: None,
            decisions: hi_agent::DecisionLog::default(),
            plan: Vec::new(),
            plan_drive_paused: false,
            plan_drive_resume_on_user_input: false,
            plan_approval_parked: false,
            plan_drive_stall: 0,
            goal_drive_stall: 0,
            plan_drive_evidence: Vec::new(),
            goal_drive_evidence: Vec::new(),
        }
    }

    fn promoted_shadow() -> SessionReducerShadow {
        SessionReducerShadow {
            reducer: Some(hi_agent::SessionReducer::new()),
            projection: Some(hi_agent::SessionProjection::new()),
            error: None,
        }
    }

    #[test]
    fn promoted_projection_becomes_the_restored_state_after_parity() {
        let message = Message::user("restore me");
        let mut shadow = promoted_shadow();
        shadow.observe(hi_agent::SessionEvent::new(
            hi_agent::SessionEventKind::Message {
                message: message.clone(),
            },
        ));

        let restored = shadow
            .finish(empty_loaded(vec![message]), "test_records")
            .unwrap();
        assert_eq!(restored.messages.len(), 1);
        assert_eq!(restored.messages[0].text(), "restore me");
    }

    #[test]
    fn promoted_projection_fails_closed_on_parity_drift() {
        let mut shadow = promoted_shadow();
        shadow.observe(hi_agent::SessionEvent::new(
            hi_agent::SessionEventKind::Message {
                message: Message::user("projected"),
            },
        ));

        let error = match shadow.finish(
            empty_loaded(vec![Message::user("established")]),
            "test_records",
        ) {
            Ok(_) => panic!("projection drift must fail closed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("diverged"));
    }
}
