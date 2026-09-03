//! The main turn loop and its helpers: `run_turn` (user message → model →
//! tool calls → results → repeat, then workspace repair), `finalize_turn`, and
//! the per-turn steering/tool-selection helpers.
//!
//! Pipeline phases are named in [`phase::TurnPhase`]:
//! `Setup → (Model → Tools → Steer)* → WorkspaceRepair → Settle → Finalize → Done`.
//!
//! Split by responsibility:
//! - [`phase`] — explicit phase enum (WorkspaceRepair vs review Steer repair)
//! - [`progress`] — progress kinds, stall tracking, tool progress labels
//! - [`retry`] — provider/output-cap retry state and **review**-repair budgets
//! - [`helpers`] — telemetry, routing, tool-entry construction
//! - [`setup`] — checkpoints, snapshots, task-context refresh
//! - [`finalize`] — recap call, usage/steer lines, text-tool cleanup
//! - [`verify_run`] — background teardown + [`crate::verify::WorkspaceRepairVerifier`]
//! - [`verify_outcome`] — react to one `VerifyOutcome` (re-enter Model or break to Settle)
//! - [`settlement`] — keep/invalidate a green verify when the tree moves after
//! - [`tools`] — one-round tool-batch scheduler (TurnPhase::Tools)
//! - [`steer`] — post-model / post-tool policy (TurnPhase::Steer)
//! - [`model_round`] — Model phase stream/retries/guards/text-steer
//! - [`loop_`] — `run_turn` orchestration (phase stamps; outcome classification in [`finalize`])

pub(crate) mod btw;
mod btw_snapshot;
mod entry;
mod fast_feedback;
mod finalize;
mod helpers;
mod loop_;
mod model_request;
mod model_retry;
mod model_round;
mod obligation;
pub mod phase;
mod progress;
pub(in crate::agent) mod retention;
mod retry;
mod settlement;
mod setup;
mod speculation;
mod state;
mod steer;
mod suggest;
mod tools;
mod verify_outcome;
mod verify_run;

pub use phase::TurnPhase;

// Re-export nothing publicly; sibling agent modules call Agent methods directly.
// Auxiliary provider work shares the primary provider's transport policy by
// default. A finite deadline remains available for tests and explicitly
// bounded integrations, but ordinary work must not stop because an arbitrary
// process-local wall-clock budget elapsed.
pub(crate) const DEFAULT_SIDE_CALL_TIMEOUT: Option<std::time::Duration> = None;

/// Await auxiliary work with an optional, explicitly supplied deadline.
///
/// Returning the elapsed duration keeps timeout reporting accurate without
/// inventing a sentinel duration for the unbounded case.
pub(crate) async fn await_side_call<F>(
    timeout: Option<std::time::Duration>,
    future: F,
) -> Result<F::Output, std::time::Duration>
where
    F: std::future::Future,
{
    match timeout {
        Some(timeout) => tokio::time::timeout(timeout, future)
            .await
            .map_err(|_| timeout),
        None => Ok(future.await),
    }
}

impl crate::Agent {
    /// Return an explicitly configured deadline for auxiliary provider work.
    /// `None` lets the provider's transport policy and turn cancellation own
    /// the lifetime, which is the ordinary/default behavior.
    pub(crate) fn side_call_timeout(&self) -> Option<std::time::Duration> {
        self.side_call_timeout
    }
}

#[cfg(test)]
mod side_call_tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn default_side_call_waits_past_the_legacy_deadline() {
        assert!(DEFAULT_SIDE_CALL_TIMEOUT.is_none());

        let completed = await_side_call(DEFAULT_SIDE_CALL_TIMEOUT, async {
            tokio::time::sleep(std::time::Duration::from_secs(16)).await;
            "completed"
        })
        .await;

        assert_eq!(completed, Ok("completed"));
    }

    #[tokio::test(start_paused = true)]
    async fn explicitly_bounded_side_call_still_times_out() {
        let timeout = std::time::Duration::from_secs(3);
        let elapsed = await_side_call(Some(timeout), std::future::pending::<()>())
            .await
            .expect_err("explicit deadline must remain enforceable");

        assert_eq!(elapsed, timeout);
    }
}
