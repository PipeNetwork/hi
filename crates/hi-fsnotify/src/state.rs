//! Lock-state state machine for git operation tracking.
//!
//! Pure data + pure transition function. No I/O. The state machine merges
//! rapid lock cycles (e.g. rebase/squash picking) into a single operation.

#![allow(dead_code)]

use std::time::{Duration, Instant};

/// After a lock release, wait this long before declaring the operation
/// complete: a lock reappearing within the window (rebase/squash cycles
/// `index.lock` per pick) is the *same* operation, so rapid cycles merge.
pub const SETTLE_MS: u64 = 500;

/// Drop transient OS events for this window after a head-changing op.
const COOLDOWN_MS: u64 = 500;

/// Diagnostic threshold — fires a one-time warning when a lock is held
/// longer than this. `git gc` on huge repos can exceed legitimately.
const STALE_LOCK_SECS: u64 = 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LockState {
    Idle,
    Locked {
        head_at_start: Option<String>,
        since: Instant,
    },
    Settling {
        head_at_start: Option<String>,
        since: Instant,
        until: Instant,
    },
    Cooldown {
        until: Instant,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LockTransition {
    None,
    Started,
    /// Emitted on any `Locked → !Locked` transition. `head_changed` is the HEAD
    /// comparison; cooldown begins iff true.
    Completed {
        head_changed: bool,
    },
    /// Cooldown timer expired; consumer never sees this — internal only.
    CooldownEnded,
}

/// One step. Pure; mutates `state` from freshly-observed FS facts.
pub(crate) fn drive(
    state: &mut LockState,
    lock_present: bool,
    head_now: Option<String>,
    now: Instant,
    cooldown: Duration,
) -> LockTransition {
    let settle = Duration::from_millis(SETTLE_MS);
    match (&*state, lock_present) {
        // Idle or Cooldown + lock appears → Locked, emit Started.
        (LockState::Idle, true) | (LockState::Cooldown { .. }, true) => {
            *state = LockState::Locked {
                head_at_start: head_now,
                since: now,
            };
            LockTransition::Started
        }

        // Settling + lock reappears → back to Locked, preserving original
        // head_at_start and since. No transition emitted (already Started).
        (
            LockState::Settling {
                head_at_start,
                since,
                ..
            },
            true,
        ) => {
            *state = LockState::Locked {
                head_at_start: head_at_start.clone(),
                since: *since,
            };
            LockTransition::None
        }

        // Locked + lock gone → Settling. No transition yet.
        (
            LockState::Locked {
                head_at_start,
                since,
            },
            false,
        ) => {
            *state = LockState::Settling {
                head_at_start: head_at_start.clone(),
                since: *since,
                until: now + settle,
            };
            LockTransition::None
        }

        // Settling + still no lock + settle elapsed → Completed.
        (
            LockState::Settling {
                head_at_start,
                until,
                ..
            },
            false,
        ) if *until <= now => {
            let head_changed = head_at_start.as_ref() != head_now.as_ref();
            if head_changed {
                *state = LockState::Cooldown {
                    until: now + cooldown,
                };
            } else {
                *state = LockState::Idle;
            }
            LockTransition::Completed { head_changed }
        }

        // Cooldown + elapsed → Idle.
        (LockState::Cooldown { until }, false) if *until <= now => {
            *state = LockState::Idle;
            LockTransition::CooldownEnded
        }

        _ => LockTransition::None,
    }
}

/// `check` fires once per stale period; resets when the lock releases.
#[derive(Debug, Default)]
pub(crate) struct StaleWarn {
    warned: bool,
}

impl StaleWarn {
    pub(crate) fn check(&mut self, state: &LockState, now: Instant) -> Option<Duration> {
        let since = match state {
            LockState::Locked { since, .. } | LockState::Settling { since, .. } => *since,
            _ => {
                self.warned = false;
                return None;
            }
        };
        let elapsed = now.duration_since(since);
        if !self.warned && elapsed > Duration::from_secs(STALE_LOCK_SECS) {
            self.warned = true;
            Some(elapsed)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cooldown() -> Duration {
        Duration::from_millis(COOLDOWN_MS)
    }

    #[test]
    fn idle_to_locked_emits_started() {
        let mut s = LockState::Idle;
        let t = drive(&mut s, true, Some("abc".into()), Instant::now(), cooldown());
        assert_eq!(t, LockTransition::Started);
        assert!(matches!(s, LockState::Locked { .. }));
    }

    #[test]
    fn locked_to_settling_no_transition() {
        let mut s = LockState::Locked {
            head_at_start: Some("abc".into()),
            since: Instant::now(),
        };
        let t = drive(
            &mut s,
            false,
            Some("abc".into()),
            Instant::now(),
            cooldown(),
        );
        assert_eq!(t, LockTransition::None);
        assert!(matches!(s, LockState::Settling { .. }));
    }

    #[test]
    fn settling_elapsed_emits_completed() {
        let now = Instant::now();
        let mut s = LockState::Settling {
            head_at_start: Some("abc".into()),
            since: now,
            until: now,
        };
        let t = drive(&mut s, false, Some("def".into()), now, cooldown());
        assert_eq!(t, LockTransition::Completed { head_changed: true });
        assert!(matches!(s, LockState::Cooldown { .. }));
    }

    #[test]
    fn settling_elapsed_no_head_change_goes_idle() {
        let now = Instant::now();
        let mut s = LockState::Settling {
            head_at_start: Some("abc".into()),
            since: now,
            until: now,
        };
        let t = drive(&mut s, false, Some("abc".into()), now, cooldown());
        assert_eq!(
            t,
            LockTransition::Completed {
                head_changed: false
            }
        );
        assert_eq!(s, LockState::Idle);
    }

    #[test]
    fn settling_relock_preserves_head_at_start() {
        let now = Instant::now();
        let mut s = LockState::Settling {
            head_at_start: Some("original".into()),
            since: now,
            until: now + Duration::from_secs(10),
        };
        // Lock reappears before settle — should go back to Locked with original head.
        let t = drive(&mut s, true, Some("different".into()), now, cooldown());
        assert_eq!(t, LockTransition::None);
        match s {
            LockState::Locked { head_at_start, .. } => {
                assert_eq!(head_at_start.as_deref(), Some("original"));
            }
            _ => panic!("expected Locked"),
        }
    }

    #[test]
    fn cooldown_elapsed_emits_cooldown_ended() {
        let now = Instant::now();
        let mut s = LockState::Cooldown { until: now };
        let t = drive(&mut s, false, None, now, cooldown());
        assert_eq!(t, LockTransition::CooldownEnded);
        assert_eq!(s, LockState::Idle);
    }

    #[test]
    fn cooldown_relock_emits_started() {
        let now = Instant::now();
        let mut s = LockState::Cooldown {
            until: now + Duration::from_secs(10),
        };
        let t = drive(&mut s, true, Some("abc".into()), now, cooldown());
        assert_eq!(t, LockTransition::Started);
        assert!(matches!(s, LockState::Locked { .. }));
    }

    #[test]
    fn settling_not_elapsed_stays() {
        let now = Instant::now();
        let mut s = LockState::Settling {
            head_at_start: Some("abc".into()),
            since: now,
            until: now + Duration::from_secs(10),
        };
        let t = drive(&mut s, false, Some("abc".into()), now, cooldown());
        assert_eq!(t, LockTransition::None);
        assert!(matches!(s, LockState::Settling { .. }));
    }

    #[test]
    fn stale_warn_fires_once() {
        let mut w = StaleWarn::default();
        let now = Instant::now();
        let s = LockState::Locked {
            head_at_start: None,
            since: now - Duration::from_secs(120),
        };
        assert!(w.check(&s, now).is_some());
        // Second check should not fire again.
        assert!(w.check(&s, now).is_none());
    }

    #[test]
    fn stale_warn_resets_on_idle() {
        let mut w = StaleWarn::default();
        let now = Instant::now();
        let s_locked = LockState::Locked {
            head_at_start: None,
            since: now - Duration::from_secs(120),
        };
        assert!(w.check(&s_locked, now).is_some());
        // Goes idle — resets.
        assert!(w.check(&LockState::Idle, now).is_none());
        // New lock — can warn again.
        let s_locked2 = LockState::Locked {
            head_at_start: None,
            since: now - Duration::from_secs(120),
        };
        assert!(w.check(&s_locked2, now).is_some());
    }

    #[test]
    fn stale_warn_no_fire_under_threshold() {
        let mut w = StaleWarn::default();
        let now = Instant::now();
        let s = LockState::Locked {
            head_at_start: None,
            since: now - Duration::from_secs(30),
        };
        assert!(w.check(&s, now).is_none());
    }
}
