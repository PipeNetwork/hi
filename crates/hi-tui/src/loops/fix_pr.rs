use std::path::Path;

use hi_tools::worktree;
use tokio_util::sync::CancellationToken;

use super::{LoopSpec, now_ms, truncate};

pub(super) struct Outcome {
    pub result: (String, bool),
    /// Cleanup is safe only once all remaining candidate work is on a branch.
    pub committed: bool,
}

/// Commit before publishing. A failed push or missing `gh` still leaves a
/// durable local branch; a failed commit must preserve the worktree itself.
pub(super) fn open(
    worktree: &Path,
    spec: &LoopSpec,
    summary: &str,
    changed: &[String],
    cancellation: &CancellationToken,
) -> Outcome {
    if cancellation.is_cancelled() {
        return Outcome {
            result: ("cancelled".to_string(), false),
            committed: false,
        };
    }
    let name = spec.name();
    let branch = format!("hi-autofix/loop{}-{}", spec.id, now_ms());
    let commit_msg = format!("hi auto-fix: {name}\n\n{}", truncate(summary, 500));
    if let Err(e) = worktree::commit_to_branch(worktree, &branch, &commit_msg) {
        return Outcome {
            result: (
                format!("verified, but couldn't prepare the PR branch: {e}"),
                true,
            ),
            committed: false,
        };
    }
    let mut result = publish(worktree, &branch, &name, summary, changed, cancellation);
    // Successful commit hooks can leave edits outside the commit. Preserve
    // those edits too, and fail closed if the final inspection is unavailable.
    let committed =
        matches!(worktree::changed_files(worktree, "HEAD"), Ok(paths) if paths.is_empty());
    if !committed {
        result
            .0
            .push_str("; additional work remains outside the commit");
    }
    Outcome { result, committed }
}

fn publish(
    worktree: &Path,
    branch: &str,
    name: &str,
    summary: &str,
    changed: &[String],
    cancellation: &CancellationToken,
) -> (String, bool) {
    if cancellation.is_cancelled() {
        return (
            format!("cancelled after committing fix to local branch {branch}"),
            false,
        );
    }
    if let Err(e) = worktree::push_branch(worktree, branch) {
        return (
            format!("fix committed to branch {branch} (couldn't push: {e}) — review it locally"),
            true,
        );
    }
    if cancellation.is_cancelled() {
        return (
            format!("cancelled after pushing fix branch {branch}; no PR was opened"),
            false,
        );
    }
    // The pushed branch stands alone if `gh` is absent or fails.
    let title = format!("hi auto-fix: {name}");
    let body = format!(
        "A recurring `hi` watch (\"{name}\") detected a problem and an agent produced a \
         verify-passing fix.\n\n**Problem**\n\n{summary}\n\n**Changed files**\n\n{}\n",
        changed
            .iter()
            .map(|f| format!("- `{f}`"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    match std::process::Command::new("gh")
        .current_dir(worktree)
        .args([
            "pr", "create", "--head", branch, "--title", &title, "--body", &body,
        ])
        .output()
    {
        Ok(o) if o.status.success() => {
            let url = String::from_utf8_lossy(&o.stdout).trim().to_string();
            (format!("opened PR: {url}"), true)
        }
        _ => (
            format!("fix pushed to branch {branch} — open a PR to land it"),
            true,
        ),
    }
}
