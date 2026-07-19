//! The main turn loop and its helpers: `run_turn` (user message → model →
//! tool calls → results → repeat, then verify), `finalize_turn`, and the
//! per-turn steering/tool-selection helpers.
//!
//! Split by responsibility:
//! - [`progress`] — progress kinds, stall tracking, tool progress labels
//! - [`retry`] — provider/output-cap retry state and review-repair budgets
//! - [`helpers`] — telemetry, routing, tool-entry construction
//! - [`setup`] — checkpoints, snapshots, task-context refresh
//! - [`finalize`] — recap call, usage/steer lines, text-tool cleanup
//! - [`verify_run`] — background teardown + [`crate::verify::RepairVerifier`] check
//! - [`settlement`] — keep/invalidate a green verify when the tree moves after
//! - [`loop_`] — `run_turn` orchestration

mod finalize;
mod helpers;
mod loop_;
mod progress;
mod retry;
mod setup;
mod settlement;
mod verify_run;

// Re-export nothing publicly; sibling agent modules call Agent methods directly.
