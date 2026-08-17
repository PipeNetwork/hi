//! Session face: the prompt is a mode machine (ask → plan → always-approve).
//!
//! Shift-Tab cycles those three. `/auto` stays slash-only; if the session is
//! already on Auto, Shift-Tab jumps to Plan rather than inserting Auto into
//! the cycle. Cycling into plan only flips flags — it does not queue the
//! `/plan` kickoff prompt.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_agent::{Agent, PermissionMode};

use crate::App;
use crate::domain::OverlayDomain;

/// Visible permission/plan face drawn on the composer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SessionFace {
    Ask,
    Plan,
    Always,
}

impl SessionFace {
    pub(crate) fn cycle(self) -> Self {
        match self {
            Self::Ask => Self::Plan,
            Self::Plan => Self::Always,
            Self::Always => Self::Ask,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Plan => "plan",
            Self::Always => "always-approve",
        }
    }
}

pub(crate) fn is_cycle_key(key: &KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) || key.modifiers.contains(KeyModifiers::ALT) {
        return false;
    }
    matches!(key.code, KeyCode::BackTab)
        || (matches!(key.code, KeyCode::Tab) && key.modifiers.contains(KeyModifiers::SHIFT))
}

/// Shift-Tab cycles on the composer, including while a plan-approval card is
/// parked. The live card owns Tab / Shift-Tab itself.
pub(crate) fn cycle_allowed(app: &App) -> bool {
    !OverlayDomain::any_hard(app)
}

impl App {
    pub(crate) fn session_face(&self) -> SessionFace {
        if self.plan_mode {
            SessionFace::Plan
        } else if self.permission_mode == PermissionMode::Always {
            SessionFace::Always
        } else {
            SessionFace::Ask
        }
    }

    pub(crate) fn cycle_session_face(&mut self) {
        let was_plan = self.plan_mode;
        let next = if !self.plan_mode && self.permission_mode == PermissionMode::Auto {
            SessionFace::Plan
        } else {
            self.session_face().cycle()
        };
        self.set_session_face(next);
        if was_plan && !self.plan_mode {
            self.maybe_open_plan_approval();
        }
    }

    pub(crate) fn set_session_face(&mut self, face: SessionFace) {
        match face {
            SessionFace::Ask => {
                self.plan_mode = false;
                self.permission_mode = PermissionMode::Ask;
            }
            SessionFace::Plan => {
                self.plan_mode = true;
                self.permission_mode = PermissionMode::Ask;
            }
            SessionFace::Always => {
                self.plan_mode = false;
                self.permission_mode = PermissionMode::Always;
            }
        }
        self.session_face_dirty = true;
        self.status = format!("mode: {}", face.label());
    }

    /// Push composer flags to the live agent after Shift-Tab (or a mid-turn
    /// cycle that could not borrow the agent until the turn settled).
    pub(crate) fn push_session_face(&mut self, agent: &mut Agent) {
        if !self.session_face_dirty {
            return;
        }
        agent.set_plan_mode(self.plan_mode);
        agent.set_permission_mode(self.permission_mode);
        agent.set_plan_drive_paused(self.plan_drive_paused);
        self.session_face_dirty = false;
        self.refresh_goal(agent);
    }

    pub(crate) fn plan_has_leftover(&self) -> bool {
        self.plan.iter().any(|step| {
            matches!(
                step.status,
                hi_agent::PlanStatus::Pending | hi_agent::PlanStatus::Active
            )
        })
    }

    pub(crate) fn composer_flags(&self) -> Vec<&'static str> {
        let mut flags = Vec::new();
        if self.plan_mode {
            flags.push("plan");
        } else if self.permission_mode == PermissionMode::Always {
            flags.push("always-approve");
        } else if self.permission_mode == PermissionMode::Auto {
            flags.push("auto");
        }
        flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_app;

    #[test]
    fn cycle_is_ask_plan_always() {
        assert_eq!(SessionFace::Ask.cycle(), SessionFace::Plan);
        assert_eq!(SessionFace::Plan.cycle(), SessionFace::Always);
        assert_eq!(SessionFace::Always.cycle(), SessionFace::Ask);
    }

    #[test]
    fn auto_jumps_to_plan() {
        let mut app = test_app("openai", "gpt-4o");
        app.permission_mode = PermissionMode::Auto;
        app.plan_mode = false;
        app.cycle_session_face();
        assert!(app.plan_mode);
        assert_eq!(app.permission_mode, PermissionMode::Ask);
        assert_eq!(app.session_face(), SessionFace::Plan);
    }

    #[test]
    fn default_always_cycles_to_ask() {
        let mut app = test_app("openai", "gpt-4o");
        assert_eq!(app.session_face(), SessionFace::Always);
        app.cycle_session_face();
        assert_eq!(app.session_face(), SessionFace::Ask);
        app.cycle_session_face();
        assert_eq!(app.session_face(), SessionFace::Plan);
        assert!(app.plan_mode);
        app.cycle_session_face();
        assert_eq!(app.session_face(), SessionFace::Always);
        assert!(!app.plan_mode);
    }

    #[test]
    fn cycle_is_blocked_while_the_live_card_owns_keys() {
        let mut app = test_app("openai", "gpt-4o");
        app.open_plan_approval();
        assert!(app.plan_approval_capturing());
        assert!(!cycle_allowed(&app));
        app.plan_approval.as_mut().unwrap().parked = true;
        assert!(!app.plan_approval_capturing());
        assert!(cycle_allowed(&app));
    }
}
