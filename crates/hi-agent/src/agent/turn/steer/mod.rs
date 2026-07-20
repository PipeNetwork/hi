//! Post-model / post-tool Steer policy ([`super::phase::TurnPhase::Steer`]).
//!
//! - [`review`] — text-only path (unfinished continues, review-answer repairs,
//!   implementation completeness when no tools were called)
//! - [`implementation`] — post-tool path (mutation recovery, repeat/no-progress)
//!
//! Workspace compile/lint/test repair stays in [`super::verify_run`].

mod cascade;
mod implementation;
mod review;

/// Whether the inner Model→Tools→Steer loop should continue or stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RoundControl {
    Continue,
    /// `true` means step-cap; `false` means natural end / stalled end of tools loop.
    BreakInner(bool),
}
