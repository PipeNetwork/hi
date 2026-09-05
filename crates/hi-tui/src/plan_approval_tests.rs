use super::*;
use crate::tests::test_app;
use hi_agent::{AgentConfig, AgentGates, AgentPaths, AgentSubagents, PlanStatus, PlanStep};
use std::sync::{Arc, Mutex};

fn fixture() -> (tempfile::TempDir, Agent, App) {
    let root = tempfile::tempdir().unwrap();
    let provider = Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "unused".into(),
    ));
    let mut agent = Agent::new(
        provider,
        AgentConfig {
            paths: AgentPaths {
                workspace_root: root.path().to_path_buf(),
                state_root: root.path().join(".hi-state"),
            },
            gates: AgentGates {
                lsp_mode: hi_agent::LspMode::Off,
                ..AgentGates::default()
            },
            subagents: AgentSubagents {
                long_horizon: true,
                ..AgentSubagents::default()
            },
            ..AgentConfig::default()
        },
    )
    .unwrap();
    agent.set_plan_mode(true);
    agent.restore_plan(vec![PlanStep {
        title: "Implement the scheduler".into(),
        status: PlanStatus::Pending,
    }]);
    let mut app = test_app("custom", "test-model");
    app.refresh_goal(&agent);
    (root, agent, app)
}

fn completed() -> hi_agent::TurnOutcome {
    hi_agent::TurnOutcome {
        status: hi_agent::TurnStatus::Completed,
        verification: hi_agent::VerificationStatus::NotApplicable,
        review: hi_agent::ReviewStatus::NotRequired,
        stop_reason: hi_agent::TurnStopReason::Completed,
        changed_files: Vec::new(),
        verified_workspace_revision: None,
        effective_route: hi_agent::EffectiveModelRoute {
            provider: None,
            model: "test-model".into(),
        },
        review_same_model: false,
        leftover: None,
        plan_leftover: Some("Implement the scheduler".into()),
    }
}

#[test]
fn completed_draft_opens_approval_without_leaving_plan_mode() {
    let (_root, mut agent, mut app) = fixture();
    app.input.set("/status");
    app.sync_completion();
    assert!(app.completion.is_some());

    app.finish_plan_draft(true, Some(&completed()));
    app.push_session_face(&mut agent);
    app.maybe_queue_drive(&agent, Some(&completed()));

    assert!(app.plan_approval_capturing());
    assert!(app.plan_mode && agent.plan_mode());
    assert!(agent.plan_approval_parked());
    assert!(app.queue.is_empty());
    assert!(app.completion.is_none());
    assert_eq!(app.input.text(), "/status");
}

#[test]
fn draft_requires_successful_settlement_and_unfinished_steps() {
    let (_root, _agent, mut app) = fixture();
    for status in [
        hi_agent::TurnStatus::Failed,
        hi_agent::TurnStatus::Blocked,
        hi_agent::TurnStatus::Cancelled,
    ] {
        let mut outcome = completed();
        outcome.status = status;
        app.finish_plan_draft(true, Some(&outcome));
        assert!(app.plan_approval.is_none());
    }
    app.finish_plan_draft(true, None);
    app.finish_plan_draft(false, Some(&completed()));
    assert!(app.plan_approval.is_none());
    app.plan[0].status = PlanStatus::Done;
    app.finish_plan_draft(true, Some(&completed()));
    assert!(app.plan_approval.is_none());
}

#[test]
fn settled_turn_does_not_reopen_already_approved_or_parked_card() {
    let (_root, mut agent, mut app) = fixture();
    app.finish_plan_draft(true, Some(&completed()));
    app.park_plan_approval(&mut agent);
    app.finish_plan_draft(true, Some(&completed()));
    assert!(app.plan_approval.as_ref().unwrap().parked);

    app.unpark_plan_approval();
    app.apply_plan_approve(&mut agent);
    app.finish_plan_draft(true, Some(&completed()));
    assert!(app.plan_approval.is_none());
    assert!(!agent.plan_approval_parked());
    assert!(!agent.plan_mode());
}

#[test]
fn successful_revision_reopens_previously_parked_approval_without_losing_feedback() {
    let (_root, mut agent, mut app) = fixture();
    app.finish_plan_draft(true, Some(&completed()));
    app.plan_approval
        .as_mut()
        .unwrap()
        .comments
        .push(PlanComment {
            step: 0,
            text: "Preserve the cancellation API.".into(),
        });
    app.park_plan_approval(&mut agent);

    // An ordinary submitted revision (or /retry) starts a new plan-mode turn.
    app.begin_plan_draft(agent.plan_mode());
    agent.restore_plan(vec![PlanStep {
        title: "Implement the revised scheduler".into(),
        status: PlanStatus::Pending,
    }]);
    app.refresh_goal(&agent);
    app.finish_plan_draft(true, Some(&completed()));
    app.push_session_face(&mut agent);
    app.maybe_queue_drive(&agent, Some(&completed()));

    assert!(app.plan_approval_visible());
    assert_eq!(app.plan[0].title, "Implement the revised scheduler");
    assert_eq!(
        app.plan_approval.as_ref().unwrap().comments[0].text,
        "Preserve the cancellation API."
    );
    assert!(app.plan_mode && agent.plan_mode());
    assert!(agent.plan_approval_parked());
    assert!(app.queue.is_empty());
}

#[test]
fn unsuccessful_or_nonplanning_revision_does_not_reopen_parked_review() {
    for status in [
        hi_agent::TurnStatus::Failed,
        hi_agent::TurnStatus::Blocked,
        hi_agent::TurnStatus::Cancelled,
    ] {
        let (_root, mut agent, mut app) = fixture();
        app.finish_plan_draft(true, Some(&completed()));
        app.park_plan_approval(&mut agent);
        app.begin_plan_draft(true);
        let mut outcome = completed();
        outcome.status = status;
        app.finish_plan_draft(true, Some(&outcome));
        assert!(app.plan_approval.as_ref().unwrap().parked);
    }
    let (_root, mut agent, mut app) = fixture();
    app.finish_plan_draft(true, Some(&completed()));
    app.park_plan_approval(&mut agent);
    app.begin_plan_draft(false);
    app.finish_plan_draft(false, Some(&completed()));
    assert!(app.plan_approval.as_ref().unwrap().parked);
}

#[test]
fn explicitly_parking_during_revision_remains_respected_at_settlement() {
    let (_root, mut agent, mut app) = fixture();
    app.finish_plan_draft(true, Some(&completed()));
    app.park_plan_approval(&mut agent);
    app.begin_plan_draft(true);
    assert!(app.unpark_plan_approval());
    app.park_plan_approval(&mut agent);
    app.finish_plan_draft(true, Some(&completed()));
    assert!(app.plan_approval.as_ref().unwrap().parked);
    assert!(agent.plan_approval_parked());
}

struct ApprovalRecorder(Arc<Mutex<Vec<bool>>>);

impl hi_agent::SessionSink for ApprovalRecorder {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_plan_approval_parked(&mut self, pending: bool) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(pending);
        Ok(())
    }
}

struct UnwritableApproval;

impl hi_agent::SessionSink for UnwritableApproval {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_plan_approval_parked(&mut self, _: bool) -> anyhow::Result<()> {
        anyhow::bail!("approval store unavailable")
    }
}

#[test]
fn failed_approval_write_keeps_card_and_read_only_mode() {
    let (_root, mut agent, mut app) = fixture();
    app.finish_plan_draft(true, Some(&completed()));
    app.push_session_face(&mut agent);
    agent.set_session(Box::new(UnwritableApproval));

    assert!(!app.apply_plan_approve(&mut agent));
    assert!(app.plan_approval_capturing());
    assert!(agent.plan_approval_parked());
    assert!(agent.plan_mode());
    assert!(app.status.contains("approval store unavailable"));

    // A deferred choice also cannot release work when its durable transition
    // fails after the turn gives the agent back to the frontend.
    app.apply_plan_approve_local();
    app.push_session_face(&mut agent);
    assert!(app.plan_approval_capturing());
    assert!(agent.plan_approval_parked());
    assert!(agent.plan_mode());
    assert!(app.status.contains("approval store unavailable"));
}

#[test]
fn visible_approval_is_persisted_and_reopening_does_not_release_it() {
    let (_root, mut agent, mut app) = fixture();
    let records = Arc::new(Mutex::new(Vec::new()));
    agent.set_session(Box::new(ApprovalRecorder(records.clone())));
    app.finish_plan_draft(true, Some(&completed()));
    app.push_session_face(&mut agent);
    assert_eq!(records.lock().unwrap().last(), Some(&true));

    // Plan mode is transient across restart. Its durable approval gate must
    // still hold the reconstructed session until the user actually decides.
    agent.set_plan_mode(false);
    let mut restored = test_app("custom", "test-model");
    restored.refresh_goal(&agent);
    restored.maybe_queue_drive(&agent, None);
    assert!(restored.plan_approval.as_ref().unwrap().parked);
    assert!(restored.queue.is_empty());
    restored.unpark_plan_approval();
    restored.push_session_face(&mut agent);
    assert!(agent.plan_approval_parked());
    assert_eq!(records.lock().unwrap().last(), Some(&true));
    restored.apply_plan_approve(&mut agent);
    assert_eq!(records.lock().unwrap().last(), Some(&false));
}

#[tokio::test]
async fn resume_cannot_dismiss_pending_approval_and_new_draft_can_replace_it() {
    let (_root, mut agent, mut app) = fixture();
    app.finish_plan_draft(true, Some(&completed()));
    app.park_plan_approval(&mut agent);
    app.handle_command(&mut agent, hi_agent::Command::Plan("resume".into()))
        .await;
    assert!(app.plan_approval.as_ref().unwrap().parked);
    assert!(agent.plan_approval_parked());
    assert!(app.queue.is_empty());

    app.handle_command(
        &mut agent,
        hi_agent::Command::Plan("revise the scheduler".into()),
    )
    .await;
    assert!(app.plan_approval.is_none());
    assert!(agent.plan_mode());
    assert!(agent.plan_approval_parked());
    app.finish_plan_draft(true, Some(&completed()));
    assert!(app.plan_approval_capturing());
}

#[test]
fn request_changes_keeps_typed_feedback_and_opens_review_after_revision() {
    for deferred in [false, true] {
        let (_root, mut agent, mut app) = fixture();
        app.finish_plan_draft(true, Some(&completed()));
        app.input.set("Keep the public API stable.");
        app.plan_approval
            .as_mut()
            .unwrap()
            .comments
            .push(PlanComment {
                step: 0,
                text: "Add cancellation coverage.".into(),
            });
        if deferred {
            app.apply_plan_request_changes_local();
            app.push_session_face(&mut agent);
        } else {
            app.apply_plan_request_changes(&mut agent);
        }
        assert!(app.input.text().contains("Keep the public API stable."));
        assert!(app.input.text().contains("Add cancellation coverage."));
        assert!(agent.plan_mode());
        assert!(agent.plan_approval_parked());
        // The in-process drafting state does not restore a parked card, but
        // a restart (which loses transient plan mode) must still require it.
        app.refresh_goal(&agent);
        assert!(app.plan_approval.is_none());
        agent.set_plan_mode(false);
        let mut restored = test_app("custom", "test-model");
        restored.refresh_goal(&agent);
        restored.maybe_queue_drive(&agent, None);
        assert!(restored.plan_approval.as_ref().unwrap().parked);
        assert!(restored.queue.is_empty());
        agent.set_plan_mode(true);
        app.finish_plan_draft(true, Some(&completed()));
        assert!(app.plan_approval_capturing());
    }
}

#[test]
fn explicit_draft_with_active_goal_still_requires_approval_and_quit_pauses_both() {
    for deferred in [false, true] {
        let (_root, mut agent, mut app) = fixture();
        agent
            .set_structured_goal(Some(hi_agent::Goal::new(
                "Ship the scheduler",
                vec!["Implement the scheduler".into()],
            )))
            .unwrap();
        app.refresh_goal(&agent);
        app.finish_plan_draft(true, Some(&completed()));
        assert!(app.plan_approval_capturing());
        if deferred {
            app.apply_plan_quit_local();
            app.push_session_face(&mut agent);
        } else {
            app.apply_plan_quit(&mut agent);
        }
        assert!(agent.plan_drive_paused());
        assert!(agent.structured_goal().unwrap().is_paused());
        assert!(agent.drive_decision(None).prompt().is_none());

        // Reopening and explicitly approving the work resumes the goal too.
        app.open_plan_approval();
        app.push_session_face(&mut agent);
        assert!(app.apply_plan_approve(&mut agent));
        assert!(!agent.structured_goal().unwrap().is_paused());
        assert_eq!(
            agent.explicit_goal_drive_decision().prompt(),
            Some(hi_agent::GOAL_CONTINUE_PROMPT)
        );
    }
}

#[tokio::test]
async fn clearing_plan_removes_visible_card_and_exiting_clears_draft_gate() {
    let (_root, mut agent, mut app) = fixture();
    app.finish_plan_draft(true, Some(&completed()));
    app.push_session_face(&mut agent);
    app.handle_command(&mut agent, hi_agent::Command::Plan("clear".into()))
        .await;
    assert!(app.plan_approval.is_none());
    assert!(agent.plan_approval_parked());
    app.handle_command(&mut agent, hi_agent::Command::Plan("off".into()))
        .await;
    assert!(!agent.plan_approval_parked());
}

#[tokio::test]
async fn initial_plan_command_persists_gate_before_any_checklist_exists() {
    let (_root, mut agent, mut app) = fixture();
    agent.clear_pinned_plan();
    agent.set_plan_mode(false);
    app.refresh_goal(&agent);
    let records = Arc::new(Mutex::new(Vec::new()));
    agent.set_session(Box::new(ApprovalRecorder(records.clone())));
    app.handle_command(
        &mut agent,
        hi_agent::Command::Plan("draft the scheduler".into()),
    )
    .await;
    assert!(agent.plan_mode());
    assert!(agent.plan_approval_parked());
    assert_eq!(*records.lock().unwrap(), vec![true]);
    assert!(app.plan_approval.is_none());
}

struct DecisionRecorder {
    records: Arc<Mutex<Vec<&'static str>>>,
    fail: Option<&'static str>,
}

impl DecisionRecorder {
    fn record_transition(&self, kind: &'static str) -> anyhow::Result<()> {
        self.records.lock().unwrap().push(kind);
        if self.fail == Some(kind) {
            anyhow::bail!("{kind} unavailable");
        }
        Ok(())
    }
}

impl hi_agent::SessionSink for DecisionRecorder {
    fn record(&mut self, _: &[hi_ai::Message], _: hi_ai::Usage) -> anyhow::Result<()> {
        Ok(())
    }
    fn record_compaction(&mut self, _: &[hi_ai::Message]) -> anyhow::Result<()> {
        Ok(())
    }
    fn record_plan_drive(&mut self, _: bool, _: u32) -> anyhow::Result<()> {
        self.record_transition("plan_pause")
    }
    fn record_goal(&mut self, _: &hi_agent::Goal) -> anyhow::Result<()> {
        self.record_transition("goal_pause")
    }
    fn record_plan_approval_parked(&mut self, pending: bool) -> anyhow::Result<()> {
        self.record_transition(if pending {
            "approval_pending"
        } else {
            "approval_clear"
        })
    }
}

#[test]
fn quit_saves_all_pauses_before_releasing_approval_and_fails_closed() {
    for deferred in [false, true] {
        for fail in [
            None,
            Some("plan_pause"),
            Some("goal_pause"),
            Some("approval_clear"),
        ] {
            let (_root, mut agent, mut app) = fixture();
            agent
                .set_structured_goal(Some(hi_agent::Goal::new(
                    "Ship the scheduler",
                    vec!["Implement the scheduler".into()],
                )))
                .unwrap();
            app.refresh_goal(&agent);
            app.finish_plan_draft(true, Some(&completed()));
            app.push_session_face(&mut agent);
            let records = Arc::new(Mutex::new(Vec::new()));
            agent.set_session(Box::new(DecisionRecorder {
                records: records.clone(),
                fail,
            }));
            if deferred {
                app.apply_plan_quit_local();
                app.push_session_face(&mut agent);
            } else {
                app.apply_plan_quit(&mut agent);
            }
            if let Some(failure) = fail {
                assert!(app.plan_approval_capturing());
                assert!(agent.plan_approval_parked());
                assert!(agent.plan_mode());
                assert!(app.status.contains(&format!("{failure} unavailable")));
                assert_eq!(records.lock().unwrap().last(), Some(&failure));
            } else {
                assert_eq!(
                    *records.lock().unwrap(),
                    vec!["plan_pause", "goal_pause", "approval_clear"]
                );
                assert!(app.plan_approval.is_none());
                assert!(agent.plan_drive_paused());
                assert!(agent.structured_goal().unwrap().is_paused());
            }
            assert!(agent.drive_decision(None).prompt().is_none());
        }
    }
}
