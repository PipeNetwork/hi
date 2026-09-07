//! One preparation/apply policy for committed sync and async transcript rewinds.

#[derive(Clone)]
pub(super) struct PreparedRewind {
    messages: std::sync::Arc<Vec<hi_ai::Message>>,
    free_goal: Option<String>,
    structured_goal: Option<crate::Goal>,
    decisions: crate::DecisionLog,
    plan: Vec<hi_tools::PlanStep>,
    scope_changed: bool,
    durable_pause: bool,
    resume_on_input: bool,
}

impl PreparedRewind {
    pub(super) fn record(&self, session: &mut dyn crate::SessionSink) -> anyhow::Result<()> {
        if self.scope_changed {
            // Clear old scope first: an interrupted append must never give the
            // new next step the previous step's stall/evidence ledger.
            session.record_plan_drive_state_with_policy(
                self.durable_pause,
                0,
                self.resume_on_input,
                true,
                &[],
            )?;
        }
        // Fully finished checklists remain visible live but must not reappear
        // as unfinished work when the session is loaded again.
        let session_plan = if crate::heuristics::plan_has_pending_steps(&self.plan) {
            self.plan.as_slice()
        } else {
            &[]
        };
        session.record_state_replacement(
            &self.messages,
            self.structured_goal.as_ref(),
            &self.decisions,
            session_plan,
        )
    }

    pub(super) fn apply(self, agent: &mut crate::Agent) {
        agent
            .messages
            .replace_all(std::sync::Arc::unwrap_or_clone(self.messages));
        agent.persisted = agent.messages.len();
        agent
            .goals
            .restore_triple(self.free_goal, self.structured_goal, self.plan);
        if self.scope_changed {
            agent.plan_drive_stall = 0;
            agent.plan_drive_evidence.clear();
        }
        agent.decisions = self.decisions;
    }
}

impl crate::Agent {
    pub(super) fn prepare_snapshot_rewind(
        &self,
        len: usize,
        snapshot: &crate::AgentStateSnapshot,
        workspace_rolled_back: bool,
    ) -> PreparedRewind {
        let mut messages = self.messages.as_slice()[..len.min(self.messages.len())].to_vec();
        // Prompt-injected goal/decision state lives in volatile turn context.
        let system = self.system_message_for();
        if let Some(first) = messages.first_mut() {
            *first = system;
        } else {
            messages.push(system);
        }
        let plan = if workspace_rolled_back {
            crate::domain::GoalState::prefer_plan_progress_after_workspace_rollback(
                &snapshot.last_plan,
                self.goals.plan(),
            )
        } else {
            crate::domain::GoalState::prefer_plan_progress(&snapshot.last_plan, self.goals.plan())
        };
        let scope_changed = crate::heuristics::next_plan_step_title(&snapshot.last_plan)
            != crate::heuristics::next_plan_step_title(&plan);
        PreparedRewind {
            messages: std::sync::Arc::new(messages),
            free_goal: snapshot.goal.clone(),
            structured_goal: self
                .config
                .subagents
                .long_horizon
                .then_some(snapshot.structured_goal.clone())
                .flatten(),
            decisions: snapshot.decisions.clone(),
            plan,
            scope_changed,
            // A transactional resume hides the badge, but restart must still
            // restore its durable pause until successful settlement commits.
            durable_pause: self.durable_plan_drive_paused(),
            resume_on_input: self.plan_drive_resumes_on_user_input(),
        }
    }
}
