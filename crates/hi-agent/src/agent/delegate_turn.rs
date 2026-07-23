//! Write-capable `delegate` subagent dispatch.
//!
//! The heavy lifting — a git worktree, a child `hi` subprocess, verification, and
//! applying only the verified diff back — lives behind the frontend-supplied
//! [`DelegateRunner`](crate::DelegateRunner) (it needs provider credentials and
//! subprocess/git plumbing the agent loop doesn't have). This method is the thin
//! dispatch: parse, budget, callout, invoke the runner, refresh the snapshot.

use serde_json::Value;

use crate::Ui;

fn delegate_tool_outcome(
    content: impl Into<String>,
    status: hi_tools::ToolStatus,
    mutation_attempted: bool,
    mutation_applied: bool,
) -> hi_tools::ToolOutcome {
    hi_tools::ToolOutcome {
        content: content.into(),
        display: None,
        plan: None,
        status,
        process: None,
        background: None,
        effects: hi_tools::ToolEffects {
            mutation_attempted,
            mutation_applied,
            file_changes: Vec::new(),
        },
        truncation: hi_tools::TruncationState::Complete,
    }
}

/// Cap on `delegate` subagents per session — lower than explore, since each is a
/// full write+verify run in an isolated worktree.
pub(crate) const MAX_DELEGATE_SUBAGENTS_PER_SESSION: u32 = 4;

/// Maximum number of delegate subagents to run concurrently within a single
/// tool batch. Each spawns a child `hi` subprocess in its own worktree, so cap
/// the fan-out to avoid resource exhaustion. The apply-back step is serialized
/// by the global `MERGE_LOCK`, so only the child execution is parallel.
pub(crate) const MAX_PARALLEL_DELEGATES: usize = 2;

/// A prepared-but-not-yet-running delegate subagent job. Extracted from the
/// parent `Agent` so the heavy work (`runner.run()`) can run concurrently
/// across multiple delegates without holding `&mut self`.
pub(crate) struct DelegateJob {
    pub(crate) slot: u32,
    pub(crate) task: String,
    pub(crate) verify: Option<String>,
    pub(crate) runner: std::sync::Arc<dyn crate::DelegateRunner>,
    /// File paths extracted from the task description (best-effort). Used to
    /// detect overlap between parallel delegates — only disjoint file sets
    /// are safe to run in parallel.
    pub(crate) file_set: std::collections::BTreeSet<String>,
}

/// The result of running a delegate job — the runner outcome plus the slot
/// number for reconciliation.
pub(crate) struct DelegateJobResult {
    pub(crate) slot: u32,
    pub(crate) outcome: crate::DelegateOutcome,
}

/// Extract file-like paths from a task description string. Best-effort: looks
/// for strings that resemble file paths (contain `/` or known extensions).
/// Used to detect whether two delegate tasks target disjoint file sets — when
/// they do, they can run in parallel safely.
pub(crate) fn extract_file_set(task: &str) -> std::collections::BTreeSet<String> {
    let mut paths = std::collections::BTreeSet::new();
    // Match tokens that look like file paths: contain at least one `/` and end
    // with a typical source extension, OR are relative paths like `src/foo.rs`.
    // This is deliberately conservative — false negatives just mean we fall
    // back to serial execution, which is always safe.
    for token in task.split_whitespace() {
        // Strip leading/trailing punctuation that wouldn't be part of a path.
        let cleaned = token
            .trim_matches(|c: char| {
                !c.is_alphanumeric() && c != '/' && c != '.' && c != '-' && c != '_'
            })
            .trim_end_matches(|c: char| c == ',' || c == ';' || c == '.' || c == ':' || c == '!');
        if cleaned.contains('/') && has_file_extension(cleaned) {
            paths.insert(cleaned.to_string());
        }
    }
    paths
}

/// Check if a string ends with a known source file extension.
fn has_file_extension(s: &str) -> bool {
    const EXTENSIONS: &[&str] = &[
        ".rs", ".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".java", ".kt",
        ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cc", ".mm", ".m",
        ".swift", ".scala", ".clj", ".ex", ".exs", ".erl", ".hs", ".ml",
        ".lua", ".r", ".sh", ".bash", ".zsh", ".fish", ".ps1",
        ".toml", ".yaml", ".yml", ".json", ".xml", ".html", ".css", ".scss",
        ".md", ".txt", ".cfg", ".ini", ".conf", ".sql", ".proto", ".thrift",
        ".dockerfile", ".makefile", ".cmake",
    ];
    let lower = s.to_lowercase();
    EXTENSIONS.iter().any(|ext| lower.ends_with(ext))
}

/// Check if two file sets are disjoint (no common paths).
pub(crate) fn file_sets_disjoint(
    a: &std::collections::BTreeSet<String>,
    b: &std::collections::BTreeSet<String>,
) -> bool {
    if a.is_empty() || b.is_empty() {
        // If either set is empty, we can't confirm disjointness — fall back
        // to serial to be safe.
        return false;
    }
    a.intersection(b).count() == 0
}

impl crate::Agent {
    /// Prepare a delegate subagent job: check budget, extract the runner and
    /// verify command, and extract the file set from the task description.
    /// Returns `None` if the budget is exhausted, no runner is attached, or
    /// the task is empty.
    pub(crate) fn prepare_delegate(
        &mut self,
        arguments: &str,
    ) -> Option<(DelegateJob, u64)> {
        let parsed = serde_json::from_str::<Value>(arguments).ok();
        let task = parsed
            .as_ref()
            .and_then(|v| v.get("task").and_then(Value::as_str))
            .unwrap_or_default()
            .to_string();
        if task.trim().is_empty() {
            return None;
        }
        if self.subagents.delegate_subagents_used >= MAX_DELEGATE_SUBAGENTS_PER_SESSION {
            return None;
        }
        let runner = self.subagents.delegate_runner.clone()?;
        let n = self
            .subagents
            .try_begin_delegate(MAX_DELEGATE_SUBAGENTS_PER_SESSION)
            .expect("budget checked above");
        let verify = parsed
            .as_ref()
            .and_then(|v| v.get("verify").and_then(Value::as_str))
            .map(str::to_string)
            .filter(|s| !s.trim().is_empty());
        let file_set = extract_file_set(&task);
        let ledger_revision = self.runtime.ledger().revision();
        Some((
            DelegateJob {
                slot: n,
                task,
                verify,
                runner,
                file_set,
            },
            ledger_revision,
        ))
    }

    /// Run one write-capable `delegate` subagent and return a summary. The runner
    /// isolates it in a worktree and applies its changes back only if verification
    /// passes; on failure nothing touches the real tree (spatial isolation).
    pub(crate) async fn handle_delegate(
        &mut self,
        arguments: &str,
        ui: &mut dyn Ui,
    ) -> hi_tools::ToolOutcome {
        let parsed = serde_json::from_str::<Value>(arguments).ok();
        let task = parsed
            .as_ref()
            .and_then(|v| v.get("task").and_then(Value::as_str))
            .unwrap_or_default()
            .to_string();
        if task.trim().is_empty() {
            return delegate_tool_outcome(
                "delegate error: missing required \"task\" argument",
                hi_tools::ToolStatus::Failed,
                false,
                false,
            );
        }
        // Budget before runner so exhausted sessions get a clear budget message
        // even when a runner is attached (and tests that only set the counter).
        if self.subagents.delegate_subagents_used >= MAX_DELEGATE_SUBAGENTS_PER_SESSION {
            return delegate_tool_outcome(
                format!(
                    "delegate budget exhausted ({MAX_DELEGATE_SUBAGENTS_PER_SESSION} this session); \
                     implement the rest directly instead."
                ),
                hi_tools::ToolStatus::Denied,
                false,
                false,
            );
        }
        let Some(runner) = self.subagents.delegate_runner.clone() else {
            return delegate_tool_outcome(
                "delegate unavailable: no subagent runner is attached in this context; \
                 implement it directly instead.",
                hi_tools::ToolStatus::Denied,
                false,
                false,
            );
        };
        let n = self
            .subagents
            .try_begin_delegate(MAX_DELEGATE_SUBAGENTS_PER_SESSION)
            .expect("budget checked above");

        let verify = parsed
            .as_ref()
            .and_then(|v| v.get("verify").and_then(Value::as_str))
            .map(str::to_string)
            .filter(|s| !s.trim().is_empty());
        let summary: String = task.chars().take(72).collect();
        let ellipsis = if task.chars().count() > 72 { "…" } else { "" };
        ui.subagent_note(&format!(
            "↳ delegate subagent {n}/{MAX_DELEGATE_SUBAGENTS_PER_SESSION}: {summary}{ellipsis}"
        ));

        let ledger_revision = self.runtime.ledger().revision();
        let outcome = runner.run(&task, verify.as_deref()).await;
        let expected_paths = outcome
            .changed_files
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let mut output =
            delegate_tool_outcome(outcome.summary, outcome.status, true, outcome.applied);
        let mut reconciliation_failed = false;

        // The frontend applies through git/transaction plumbing outside the
        // normal tool engine. Reconcile it here, then attribute only the paths
        // the verified delegate reported. Concurrent user/editor changes still
        // enter the turn-level ledger, but never masquerade as delegate effects.
        match self.reconcile_workspace_changes().await {
            Ok(()) => {
                let changes = self.runtime.ledger().changes_since(ledger_revision);
                let delegate_changes = changes
                    .into_iter()
                    .filter(|change| expected_paths.contains(&change.path))
                    .collect::<Vec<_>>();
                let actual_paths = delegate_changes
                    .iter()
                    .map(|change| change.path.clone())
                    .collect::<std::collections::BTreeSet<_>>();
                let exact_application =
                    outcome.applied && !expected_paths.is_empty() && actual_paths == expected_paths;
                output.effects.mutation_applied = !delegate_changes.is_empty();
                output.effects.file_changes = delegate_changes;
                if output.status == hi_tools::ToolStatus::Succeeded && !exact_application {
                    output.status = hi_tools::ToolStatus::Failed;
                    output.content.push_str(
                        "\nDelegate reported success without the exact applied workspace changes.",
                    );
                } else if output.status != hi_tools::ToolStatus::Succeeded
                    && output.effects.mutation_applied
                {
                    output.content.push_str(
                        "\nWarning: declared workspace changes remained after delegate failure.",
                    );
                }
            }
            Err(error) => {
                reconciliation_failed = true;
                output.status = hi_tools::ToolStatus::Failed;
                output.content.push_str(&format!(
                    "\nFailed to reconcile delegate workspace effects: {error:#}\n\
                     Warning: workspace state is unknown; inspect the working tree before continuing."
                ));
                output.effects.file_changes.clear();
            }
        }

        if output.status == hi_tools::ToolStatus::Succeeded {
            ui.subagent_note(&format!(
                "↳ delegate subagent {n} applied — {} file(s) changed",
                output.effects.file_changes.len()
            ));
        } else if reconciliation_failed {
            ui.subagent_note(&format!(
                "↳ delegate subagent {n} failed — workspace state unknown"
            ));
        } else if output.effects.mutation_applied {
            ui.subagent_note(&format!(
                "↳ delegate subagent {n} failed — workspace changes remain"
            ));
        } else {
            ui.subagent_note(&format!("↳ delegate subagent {n} rolled back"));
        }
        // The runner may have applied a diff to the working tree; refresh the
        // parent's snapshot AND clear the read cache so change detection, verify,
        // and any later `read` see the merged content — the merge writes files via
        // `git apply`, outside the edit-tool layer that normally invalidates.
        self.invalidate_snapshot();
        self.runtime.clear_read_cache();
        output
    }

    /// Finish a completed delegate job: reconcile workspace changes, attribute
    /// file changes, and refresh the snapshot. Called after parallel delegates
    /// complete in the batch scheduler. Returns the final tool outcome.
    /// `ledger_revision` is captured before the delegate runs (in
    /// `prepare_delegate`) so reconciliation only attributes changes made by
    /// this delegate, not concurrent ones.
    pub(crate) async fn finish_delegate(
        &mut self,
        result: DelegateJobResult,
        ledger_revision: u64,
        ui: &mut dyn Ui,
    ) -> hi_tools::ToolOutcome {
        let DelegateJobResult { slot: n, outcome } = result;
        let expected_paths = outcome
            .changed_files
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let mut output =
            delegate_tool_outcome(outcome.summary, outcome.status, true, outcome.applied);
        let mut reconciliation_failed = false;

        match self.reconcile_workspace_changes().await {
            Ok(()) => {
                let changes = self.runtime.ledger().changes_since(ledger_revision);
                let delegate_changes = changes
                    .into_iter()
                    .filter(|change| expected_paths.contains(&change.path))
                    .collect::<Vec<_>>();
                let actual_paths = delegate_changes
                    .iter()
                    .map(|change| change.path.clone())
                    .collect::<std::collections::BTreeSet<_>>();
                let exact_application =
                    outcome.applied && !expected_paths.is_empty() && actual_paths == expected_paths;
                output.effects.mutation_applied = !delegate_changes.is_empty();
                output.effects.file_changes = delegate_changes;
                if output.status == hi_tools::ToolStatus::Succeeded && !exact_application {
                    output.status = hi_tools::ToolStatus::Failed;
                    output.content.push_str(
                        "\nDelegate reported success without the exact applied workspace changes.",
                    );
                } else if output.status != hi_tools::ToolStatus::Succeeded
                    && output.effects.mutation_applied
                {
                    output.content.push_str(
                        "\nWarning: declared workspace changes remained after delegate failure.",
                    );
                }
            }
            Err(error) => {
                reconciliation_failed = true;
                output.status = hi_tools::ToolStatus::Failed;
                output.content.push_str(&format!(
                    "\nFailed to reconcile delegate workspace effects: {error:#}\n\
                     Warning: workspace state is unknown; inspect the working tree before continuing."
                ));
                output.effects.file_changes.clear();
            }
        }

        if output.status == hi_tools::ToolStatus::Succeeded {
            ui.subagent_note(&format!(
                "↳ delegate subagent {n} applied — {} file(s) changed",
                output.effects.file_changes.len()
            ));
        } else if reconciliation_failed {
            ui.subagent_note(&format!(
                "↳ delegate subagent {n} failed — workspace state unknown"
            ));
        } else if output.effects.mutation_applied {
            ui.subagent_note(&format!(
                "↳ delegate subagent {n} failed — workspace changes remain"
            ));
        } else {
            ui.subagent_note(&format!("↳ delegate subagent {n} rolled back"));
        }
        self.invalidate_snapshot();
        self.runtime.clear_read_cache();
        output
    }

    /// Release a delegate budget slot when the job failed before running.
    pub(crate) fn release_delegate_slot(&mut self) {
        self.subagents.release_delegate();
    }
}

/// Run a prepared delegate job to completion. This is a free function (not a
/// method on `Agent`) so it can run concurrently across multiple jobs without
/// holding `&mut self`. The `DelegateRunner` is `Send + Sync`, so multiple
/// `runner.run()` calls can execute in parallel — each creates its own worktree
/// and runs its own child subprocess. The apply-back step is serialized by the
/// global `MERGE_LOCK` in the candidate merge infrastructure.
pub(crate) async fn run_delegate_job(job: DelegateJob) -> DelegateJobResult {
    let DelegateJob {
        slot,
        task,
        verify,
        runner,
        file_set: _,
    } = job;
    let outcome = runner.run(&task, verify.as_deref()).await;
    DelegateJobResult { slot, outcome }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_file_set_finds_paths_with_extensions() {
        let paths = extract_file_set(
            "Update crates/hi-agent/src/agent/delegate_turn.rs and crates/hi-tools/src/lib.rs",
        );
        assert!(paths.contains("crates/hi-agent/src/agent/delegate_turn.rs"));
        assert!(paths.contains("crates/hi-tools/src/lib.rs"));
    }

    #[test]
    fn extract_file_set_ignores_non_path_tokens() {
        let paths = extract_file_set("Refactor the delegate runner to use parallel worktrees");
        assert!(paths.is_empty(), "no file paths in this task: {paths:?}");
    }

    #[test]
    fn extract_file_set_handles_trailing_punctuation() {
        let paths = extract_file_set("Fix the bug in src/main.rs, then update src/lib.rs.");
        assert!(paths.contains("src/main.rs"));
        assert!(paths.contains("src/lib.rs"));
    }

    #[test]
    fn disjoint_file_sets_detected() {
        let a = extract_file_set("Update src/foo.rs and src/bar.rs");
        let b = extract_file_set("Update src/baz.rs and src/qux.rs");
        assert!(file_sets_disjoint(&a, &b));
    }

    #[test]
    fn overlapping_file_sets_not_disjoint() {
        let a = extract_file_set("Update src/foo.rs and src/bar.rs");
        let b = extract_file_set("Update src/bar.rs and src/baz.rs");
        assert!(!file_sets_disjoint(&a, &b));
    }

    #[test]
    fn empty_file_sets_not_disjoint() {
        let a: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let b = extract_file_set("Update src/foo.rs");
        // Empty set → can't confirm disjointness → false (fall back to serial).
        assert!(!file_sets_disjoint(&a, &b));
        assert!(!file_sets_disjoint(&b, &a));
    }
}
