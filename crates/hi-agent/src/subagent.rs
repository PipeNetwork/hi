//! The write-`delegate` subagent seam.
//!
//! Running a write-capable subagent in isolation needs a git worktree, a child
//! `hi` subprocess, and the provider credentials to authenticate it — all of which
//! live in the frontend (hi-cli), not in the agent loop. So the agent depends only
//! on this trait; the frontend supplies a [`DelegateRunner`] that does the
//! worktree + subprocess + verify + apply-back dance. If none is attached, the
//! `delegate` tool reports itself unavailable.

use async_trait::async_trait;

/// A per-role model route override (team roles): which model — and optionally
/// which OpenAI-compatible endpoint — a subagent runs on, independent of the
/// driver's route. All-`None` means "inherit the driver". This is what lets a
/// big cloud driver plan and integrate while local models do the execution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SubagentRoute {
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
}

impl SubagentRoute {
    /// Whether this route overrides anything at all.
    pub fn is_inherited(&self) -> bool {
        self.model.is_none() && self.base_url.is_none()
    }

    /// Short display label for role tables: `model @ endpoint` with inherited
    /// parts elided.
    pub fn label(&self, driver_model: &str) -> String {
        let model = self.model.as_deref().unwrap_or(driver_model);
        match self.base_url.as_deref() {
            Some(url) => format!("{model} @ {url}"),
            None => format!("{model} (driver route)"),
        }
    }
}

/// Outcome of one `delegate` write-subagent run.
pub struct DelegateOutcome {
    /// Authoritative result of the isolated delegate run. A rolled-back,
    /// unavailable, timed-out, or otherwise rejected candidate is never
    /// successful merely because it produced a textual summary.
    pub status: hi_tools::ToolStatus,
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

    /// Cancellation-aware delegate execution. Existing frontends retain their
    /// behavior through the default implementation; runners that own child
    /// processes should override this and cooperatively terminate and clean up
    /// when `cancellation` is requested.
    async fn run_cancellable(
        &self,
        task: &str,
        verify: Option<&str>,
        cancellation: crate::TurnCancellation,
    ) -> DelegateOutcome {
        if cancellation.is_cancelled() {
            return DelegateOutcome {
                status: hi_tools::ToolStatus::Cancelled,
                applied: false,
                changed_files: Vec::new(),
                summary: "delegate cancelled before execution".into(),
            };
        }
        self.run(task, verify).await
    }

    /// Route-aware delegate execution (team roles). The default ignores the
    /// route and preserves existing runner behavior; runners that spawn child
    /// processes should apply `route` so delegate work can run on a different
    /// model/endpoint than the driver.
    async fn run_routed(
        &self,
        task: &str,
        verify: Option<&str>,
        route: &SubagentRoute,
        cancellation: crate::TurnCancellation,
    ) -> DelegateOutcome {
        let _ = route;
        self.run_cancellable(task, verify, cancellation).await
    }
}
