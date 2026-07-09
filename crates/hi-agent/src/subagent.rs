//! The write-`delegate` subagent seam.
//!
//! Running a write-capable subagent in isolation needs a git worktree, a child
//! `hi` subprocess, and the provider credentials to authenticate it — all of which
//! live in the frontend (hi-cli), not in the agent loop. So the agent depends only
//! on this trait; the frontend supplies a [`DelegateRunner`] that does the
//! worktree + subprocess + verify + apply-back dance. If none is attached, the
//! `delegate` tool reports itself unavailable.

use async_trait::async_trait;

/// Outcome of one `delegate` write-subagent run.
pub struct DelegateOutcome {
    /// Whether the subagent's verified changes were applied to the working tree.
    pub applied: bool,
    /// Files the applied change touched (empty when nothing was applied).
    pub changed_files: Vec<String>,
    /// A summary fed back to the model (what happened + why kept/rolled back).
    pub summary: String,
}

/// Runs a write-capable subagent in isolation, verifying before merging its work
/// back into the parent's working tree. Implemented by the frontend.
#[async_trait]
pub trait DelegateRunner: Send + Sync {
    /// Carry out `task` in an isolated worktree, gating the result on `verify`
    /// (or the session's default when `None`), and apply the diff back only if it
    /// passes.
    async fn run(&self, task: &str, verify: Option<&str>) -> DelegateOutcome;
}
