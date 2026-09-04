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

    /// Mirror the permission mode that an interactive synthetic drive will
    /// use once `Agent::begin_drive_turn` starts polling. The Agent temporarily
    /// demotes Always to Auto so routine edits remain checkpointed and can be
    /// auto-approved by the frontend; the App must see that same mode while it
    /// owns confirmation handling.
    ///
    /// This is transient state, not a user choice, so it deliberately does not
    /// mark the session face dirty. The Agent's starting mode is authoritative
    /// after the turn (Always restores; Auto/Ask remain), and is returned when
    /// a temporary drive demotion needs restoring. Shift-Tab still wins by
    /// marking a real mid-turn choice dirty.
    pub(crate) fn sync_synthetic_drive_permission(
        &mut self,
        kind: hi_agent::DriveKind,
        agent_mode: PermissionMode,
    ) -> Option<PermissionMode> {
        let active_mode = if kind.is_drive() && agent_mode == PermissionMode::Always {
            PermissionMode::Auto
        } else {
            agent_mode
        };
        self.permission_mode = active_mode;
        (active_mode != agent_mode).then_some(agent_mode)
    }

    pub(crate) fn restore_synthetic_drive_permission(
        &mut self,
        restore_mode: Option<PermissionMode>,
    ) {
        if !self.session_face_dirty
            && let Some(restore_mode) = restore_mode
        {
            self.permission_mode = restore_mode;
        }
    }

    /// Push composer flags to the live agent after Shift-Tab (or a mid-turn
    /// cycle that could not borrow the agent until the turn settled).
    pub(crate) fn push_session_face(&mut self, agent: &mut Agent) -> bool {
        if !self.session_face_dirty && !self.plan_drive_pause_dirty {
            return true;
        }
        let saved = (|| -> anyhow::Result<()> {
            if self.plan_drive_pause_dirty {
                // Save both execution states before releasing approval. A
                // failed write or restart between records must retain a gate.
                agent.try_set_plan_drive_paused(self.plan_drive_paused)?;
                agent.try_set_goal_pause_reason(if self.plan_drive_paused {
                    hi_agent::GoalPauseReason::User
                } else {
                    hi_agent::GoalPauseReason::None
                })?;
            }
            if self.session_face_dirty {
                // The legacy persisted "parked" flag is the approval gate. Keep
                // it set while drafting/revising too: plan mode itself is not
                // durable, and restart must not execute an unapproved checklist.
                agent
                    .try_set_plan_approval_parked(self.plan_approval.is_some() || self.plan_mode)?;
            }
            Ok(())
        })();
        if let Err(error) = saved {
            self.plan_mode = true;
            agent.set_plan_mode(true);
            if self.plan_approval.is_none() && self.plan_has_leftover() {
                self.open_plan_approval();
            }
            self.status = format!("could not save plan approval: {error:#}");
            return false;
        }
        if self.session_face_dirty {
            agent.set_plan_mode(self.plan_mode);
            agent.set_permission_mode(self.permission_mode);
        }
        self.session_face_dirty = false;
        self.plan_drive_pause_dirty = false;
        self.refresh_goal(agent);
        true
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

    #[test]
    fn synthetic_drive_uses_auto_for_safe_edits_then_restores_always() {
        let mut app = test_app("openai", "gpt-4o");
        assert_eq!(app.permission_mode, PermissionMode::Always);
        let previous =
            app.sync_synthetic_drive_permission(hi_agent::DriveKind::Plan, PermissionMode::Always);
        let safe = hi_agent::ConfirmationRequest::FileEdit {
            path: "src/lib.rs".into(),
            diff: "+fn smoke() {}\n".into(),
        };

        assert_eq!(app.permission_mode, PermissionMode::Auto);
        assert!(app.should_auto_approve(&safe));
        assert!(!app.session_face_dirty);

        app.restore_synthetic_drive_permission(previous);
        assert_eq!(app.permission_mode, PermissionMode::Always);
        assert!(!app.session_face_dirty);
    }

    #[test]
    fn mid_turn_shift_tab_choice_wins_over_drive_permission_restore() {
        let mut app = test_app("openai", "gpt-4o");
        let previous =
            app.sync_synthetic_drive_permission(hi_agent::DriveKind::Plan, PermissionMode::Always);
        assert_eq!(app.permission_mode, PermissionMode::Auto);

        // Auto is slash-only, so Shift-Tab enters the visible Plan face.
        app.cycle_session_face();
        assert!(app.plan_mode);
        assert_eq!(app.permission_mode, PermissionMode::Ask);
        assert!(app.session_face_dirty);

        app.restore_synthetic_drive_permission(previous);
        assert!(app.plan_mode);
        assert_eq!(app.permission_mode, PermissionMode::Ask);
        assert!(app.session_face_dirty);

        // Once the turn releases its Agent borrow, the pending choice must
        // become the durable agent mode instead of the drive's restored
        // Always value winning the race.
        let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
            "http://127.0.0.1:1/v1".into(),
            "unused".into(),
        ));
        let mut agent = Agent::new(provider, hi_agent::AgentConfig::default()).unwrap();
        agent.set_permission_mode(PermissionMode::Always);
        app.push_session_face(&mut agent);
        assert!(agent.plan_mode());
        assert_eq!(agent.permission_mode(), PermissionMode::Ask);
        assert!(!app.session_face_dirty);
    }

    #[test]
    fn ordinary_user_turn_does_not_change_permission_face() {
        let mut app = test_app("openai", "gpt-4o");
        assert_eq!(
            app.sync_synthetic_drive_permission(hi_agent::DriveKind::User, PermissionMode::Always,),
            None
        );
        assert_eq!(app.permission_mode, PermissionMode::Always);
    }

    #[test]
    fn stale_app_mode_converges_to_the_agents_post_turn_mode() {
        let mut app = test_app("openai", "gpt-4o");

        // Auto is not a temporary drive mode when the Agent started there.
        app.permission_mode = PermissionMode::Always;
        let restore =
            app.sync_synthetic_drive_permission(hi_agent::DriveKind::Plan, PermissionMode::Auto);
        assert_eq!(app.permission_mode, PermissionMode::Auto);
        assert_eq!(restore, None);
        app.restore_synthetic_drive_permission(restore);
        assert_eq!(app.permission_mode, PermissionMode::Auto);

        // User turns do not have a drive demotion, but still synchronize a
        // stale clean App mirror to the live Agent mode.
        app.permission_mode = PermissionMode::Always;
        let restore =
            app.sync_synthetic_drive_permission(hi_agent::DriveKind::User, PermissionMode::Ask);
        assert_eq!(app.permission_mode, PermissionMode::Ask);
        assert_eq!(restore, None);
        app.restore_synthetic_drive_permission(restore);
        assert_eq!(app.permission_mode, PermissionMode::Ask);

        // A stale App value must not become the restore target: Always came
        // from the Agent, so it is what returns after temporary Auto.
        app.permission_mode = PermissionMode::Ask;
        let restore =
            app.sync_synthetic_drive_permission(hi_agent::DriveKind::Goal, PermissionMode::Always);
        assert_eq!(app.permission_mode, PermissionMode::Auto);
        assert_eq!(restore, Some(PermissionMode::Always));
        app.restore_synthetic_drive_permission(restore);
        assert_eq!(app.permission_mode, PermissionMode::Always);
    }
}
