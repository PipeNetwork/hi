use super::{Goal, GoalStatus};
use std::collections::{HashMap, VecDeque};

impl Goal {
    /// Revoke only completion claims introduced after the trusted baseline.
    /// Preserve plan edits, diagnostics, blocked work, and user settings.
    pub(crate) fn revoke_unsupported_completion(&mut self, before: Option<&Goal>) -> bool {
        let before = before.filter(|goal| goal.objective == self.objective);
        let mut previous = HashMap::<&str, VecDeque<GoalStatus>>::new();
        if let Some(before) = before {
            for step in &before.sub_goals {
                previous
                    .entry(&step.description)
                    .or_default()
                    .push_back(step.status);
            }
        }
        let mut changed = false;
        for step in &mut self.sub_goals {
            // Consume each old occurrence once: a newly appended duplicate
            // cannot borrow an earlier step's completion evidence.
            let status = previous
                .get_mut(step.description.as_str())
                .and_then(VecDeque::pop_front);
            if step.status == GoalStatus::Done && status != Some(GoalStatus::Done) {
                step.status = status.unwrap_or(GoalStatus::Pending);
                changed = true;
            }
        }
        if self.objective_complete && !before.is_some_and(|goal| goal.objective_complete) {
            self.objective_complete = false;
            changed = true;
        }
        if changed {
            self.rederive_status();
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::GoalPauseReason;

    #[test]
    fn revocation_preserves_requested_plan_edits_and_prior_completion() {
        let mut before = Goal::new(
            "ship it",
            vec!["done".into(), "working".into(), "blocked".into()],
        );
        before.advance();
        before.sub_goals[2].status = GoalStatus::Blocked;
        let mut current = before.clone();
        current.sub_goals[1].status = GoalStatus::Done;
        current.sub_goals[2].status = GoalStatus::Done;
        current.sub_goals[1].notes.push("keep diagnostic".into());
        current.append_missing(&["new requested step".into()]);
        current.sub_goals.push(before.sub_goals[0].clone());
        current.pause(GoalPauseReason::User);
        current.turn_budget = Some(17);
        current.objective_complete = true;
        assert!(current.revoke_unsupported_completion(Some(&before)));
        assert_eq!(current.sub_goals[0].status, GoalStatus::Done);
        assert_eq!(current.sub_goals[1].status, GoalStatus::Active);
        assert_eq!(current.sub_goals[2].status, GoalStatus::Blocked);
        assert_eq!(current.sub_goals[3].description, "new requested step");
        assert_eq!(current.sub_goals[4].status, GoalStatus::Pending);
        assert_eq!(current.sub_goals[1].notes, ["keep diagnostic"]);
        assert_eq!(current.pause_reason, GoalPauseReason::User);
        assert_eq!(current.turn_budget, Some(17));
        assert!(!current.objective_complete);
        assert!(!current.revoke_unsupported_completion(Some(&before)));
    }

    #[test]
    fn replacement_objective_cannot_borrow_old_completion() {
        let mut before = Goal::new("old", vec!["same description".into()]);
        before.advance();
        let mut current = Goal::new("new request", vec!["same description".into()]);
        current.advance();
        assert!(current.revoke_unsupported_completion(Some(&before)));
        assert_eq!(current.objective, "new request");
        assert_eq!(current.sub_goals[0].status, GoalStatus::Active);
    }
}
