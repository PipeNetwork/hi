//! Write-capable `delegate` subagent dispatch.
//!
//! The heavy lifting — a git worktree, a child `hi` subprocess, verification, and
//! applying only the verified diff back — lives behind the frontend-supplied
//! [`DelegateRunner`](crate::DelegateRunner) (it needs provider credentials and
//! subprocess/git plumbing the agent loop doesn't have). This method is the thin
//! dispatch: parse, budget, callout, invoke the runner, refresh the snapshot.

use serde_json::Value;

use crate::Ui;

/// Cap on `delegate` subagents per session — lower than explore, since each is a
/// full write+verify run in an isolated worktree.
pub(crate) const MAX_DELEGATE_SUBAGENTS_PER_SESSION: u32 = 4;

impl crate::Agent {
    /// Run one write-capable `delegate` subagent and return a summary. The runner
    /// isolates it in a worktree and applies its changes back only if verification
    /// passes; on failure nothing touches the real tree (spatial isolation).
    pub(crate) async fn handle_delegate(&mut self, arguments: &str, ui: &mut dyn Ui) -> String {
        let parsed = serde_json::from_str::<Value>(arguments).ok();
        let task = parsed
            .as_ref()
            .and_then(|v| v.get("task").and_then(Value::as_str))
            .unwrap_or_default()
            .to_string();
        if task.trim().is_empty() {
            return "delegate error: missing required \"task\" argument".to_string();
        }
        if self.delegate_subagents_used >= MAX_DELEGATE_SUBAGENTS_PER_SESSION {
            return format!(
                "delegate budget exhausted ({MAX_DELEGATE_SUBAGENTS_PER_SESSION} this session); \
                 implement the rest directly instead."
            );
        }
        let Some(runner) = self.delegate_runner.clone() else {
            return "delegate unavailable: no subagent runner is attached in this context; \
                    implement it directly instead."
                .to_string();
        };

        let verify = parsed
            .as_ref()
            .and_then(|v| v.get("verify").and_then(Value::as_str))
            .map(str::to_string)
            .filter(|s| !s.trim().is_empty());

        self.delegate_subagents_used += 1;
        let n = self.delegate_subagents_used;
        let summary: String = task.chars().take(72).collect();
        let ellipsis = if task.chars().count() > 72 { "…" } else { "" };
        ui.subagent_note(&format!(
            "↳ delegate subagent {n}/{MAX_DELEGATE_SUBAGENTS_PER_SESSION}: {summary}{ellipsis}"
        ));

        let outcome = runner.run(&task, verify.as_deref()).await;
        if outcome.applied {
            ui.subagent_note(&format!(
                "↳ delegate subagent {n} applied — {} file(s) changed",
                outcome.changed_files.len()
            ));
        } else {
            ui.subagent_note(&format!("↳ delegate subagent {n} rolled back"));
        }
        // The runner may have applied a diff to the working tree; refresh the
        // parent's snapshot AND clear the read cache so change detection, verify,
        // and any later `read` see the merged content — the merge writes files via
        // `git apply`, outside the edit-tool layer that normally invalidates.
        self.invalidate_snapshot();
        if outcome.applied {
            hi_tools::clear_read_cache();
        }
        outcome.summary
    }
}
