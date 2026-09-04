use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_agent::Agent;

use crate::{App, plan_approval::PlanApprovalOutcome};

/// Handle approval before composer completion or any other idle key handler.
/// Draft text and queued prompts are left intact until the user decides.
pub(super) fn handle_idle_plan_approval_key(
    app: &mut App,
    agent: &mut Agent,
    key: &KeyEvent,
) -> Option<String> {
    if !app.plan_approval_visible() {
        return None;
    }
    match crate::plan_approval::handle_key(app, key) {
        crate::plan_approval::PlanApprovalOutcome::Approve => {
            if !app.plan_has_leftover() || !agent.plan_incomplete() {
                app.refresh_goal(agent);
                return None;
            }
            if app.apply_plan_approve(agent) {
                agent
                    .explicit_goal_drive_decision()
                    .prompt()
                    .map(str::to_string)
            } else {
                None
            }
        }
        crate::plan_approval::PlanApprovalOutcome::Park => {
            app.park_plan_approval(agent);
            None
        }
        crate::plan_approval::PlanApprovalOutcome::RequestChanges => {
            app.apply_plan_request_changes(agent);
            None
        }
        crate::plan_approval::PlanApprovalOutcome::Quit => {
            app.apply_plan_quit(agent);
            None
        }
        crate::plan_approval::PlanApprovalOutcome::Continue => None,
    }
}

/// During a turn, the visible plan card owns navigation and Esc. Ctrl-C
/// remains the explicit whole-turn cancellation shortcut.
pub(super) fn handle_working_plan_approval_key(app: &mut App, key: &KeyEvent) -> bool {
    if !app.plan_approval_visible()
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
    {
        return false;
    }
    match crate::plan_approval::handle_key(app, key) {
        PlanApprovalOutcome::Approve if app.plan_has_leftover() => {
            app.apply_plan_approve_local();
            let prompt = if app
                .goal
                .as_ref()
                .is_some_and(hi_agent::Goal::has_drive_work)
            {
                hi_agent::GOAL_CONTINUE_PROMPT
            } else {
                hi_agent::PLAN_DRIVE_PROMPT
            };
            let _ = app.enqueue_prompt_front(prompt.to_string());
        }
        PlanApprovalOutcome::Park => app.park_plan_approval_local(),
        PlanApprovalOutcome::RequestChanges => app.apply_plan_request_changes_local(),
        PlanApprovalOutcome::Quit => app.apply_plan_quit_local(),
        PlanApprovalOutcome::Continue | PlanApprovalOutcome::Approve => {}
    }
    true
}

#[cfg(test)]
#[path = "plan_input_tests.rs"]
mod tests;
