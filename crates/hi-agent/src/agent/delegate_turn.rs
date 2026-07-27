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

/// Default cap on `delegate` subagents per turn — lower than explore, since
/// each is a full write+verify run in an isolated worktree. Refilled every
/// turn ([`crate::domain::SubagentSessionState::begin_turn`]).
pub(crate) const MAX_DELEGATE_SUBAGENTS_PER_TURN: u32 = 4;
const MAX_CONFIGURED_DELEGATES: u32 = 16;

/// The per-turn delegate cap. `HI_DELEGATE_SESSION_LIMIT` keeps its name for
/// compatibility but now bounds each turn, not the whole session.
pub(crate) fn delegate_turn_limit() -> u32 {
    configured_delegate_limit(
        std::env::var("HI_DELEGATE_SESSION_LIMIT").ok().as_deref(),
        MAX_DELEGATE_SUBAGENTS_PER_TURN as usize,
    ) as u32
}

/// Default maximum number of delegate subagents in one tool batch.
pub(crate) const MAX_PARALLEL_DELEGATES: usize = 4;

pub(crate) fn parallel_delegate_limit() -> usize {
    configured_delegate_limit(
        std::env::var("HI_PARALLEL_DELEGATES").ok().as_deref(),
        MAX_PARALLEL_DELEGATES,
    )
}

fn configured_delegate_limit(value: Option<&str>, default: usize) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
        .clamp(1, MAX_CONFIGURED_DELEGATES as usize)
}

/// A prepared-but-not-yet-running delegate subagent job. Extracted from the
/// parent `Agent` so the heavy work (`runner.run()`) can run concurrently
/// across multiple delegates without holding `&mut self`.
pub(crate) struct DelegateJob {
    pub(crate) slot: u32,
    pub(crate) task: String,
    pub(crate) verify: Option<String>,
    pub(crate) runner: std::sync::Arc<dyn crate::DelegateRunner>,
    /// Team-role route override for this executor (all-`None` = driver route).
    pub(crate) route: crate::SubagentRoute,
    pub(crate) cancellation: crate::TurnCancellation,
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

fn structured_file_set(parsed: Option<&Value>) -> Option<std::collections::BTreeSet<String>> {
    let scope = parsed?.get("scope")?.as_array()?;
    let paths = scope
        .iter()
        .filter_map(Value::as_str)
        .map(normalize_scope)
        .filter(|path| !path.is_empty() && path != "." && !path.starts_with("../"))
        .collect::<std::collections::BTreeSet<_>>();
    (!paths.is_empty()).then_some(paths)
}

/// Extract conservative workspace scopes from a task description. Exact file
/// paths are preferred; directory-like paths are also retained so delegates
/// targeting separate modules can run concurrently without requiring every
/// filename to be listed.
pub(crate) fn extract_file_set(task: &str) -> std::collections::BTreeSet<String> {
    let mut paths = std::collections::BTreeSet::new();
    for token in task.split_whitespace() {
        let cleaned = token
            .trim_matches(|c: char| {
                !c.is_alphanumeric() && c != '/' && c != '.' && c != '-' && c != '_'
            })
            .trim_end_matches([',', ';', '.', ':', '!']);
        if cleaned.contains('/') && (has_file_extension(cleaned) || looks_like_directory(cleaned)) {
            paths.insert(normalize_scope(cleaned));
        }
    }
    paths
}

fn looks_like_directory(path: &str) -> bool {
    !path.rsplit('/').next().unwrap_or_default().contains('.')
}

fn normalize_scope(path: &str) -> String {
    path.trim_matches('/').to_string()
}

/// Check if a string ends with a known source file extension.
fn has_file_extension(s: &str) -> bool {
    const EXTENSIONS: &[&str] = &[
        ".rs",
        ".py",
        ".ts",
        ".js",
        ".tsx",
        ".jsx",
        ".go",
        ".java",
        ".kt",
        ".rb",
        ".php",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cc",
        ".mm",
        ".m",
        ".swift",
        ".scala",
        ".clj",
        ".ex",
        ".exs",
        ".erl",
        ".hs",
        ".ml",
        ".lua",
        ".r",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".xml",
        ".html",
        ".css",
        ".scss",
        ".md",
        ".txt",
        ".cfg",
        ".ini",
        ".conf",
        ".sql",
        ".proto",
        ".thrift",
        ".dockerfile",
        ".makefile",
        ".cmake",
    ];
    let lower = s.to_lowercase();
    EXTENSIONS.iter().any(|ext| lower.ends_with(ext))
}

/// The task shape of a `delegate` call, from its optional `kind` argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DelegateKind {
    /// Open-ended implementation (the default).
    Author,
    /// A mechanical, precisely-specified change.
    Edit,
}

/// Parse the `kind` argument; anything but an explicit `"edit"` is authoring,
/// so an unknown value can never accidentally land on the smaller editor model.
pub(crate) fn delegate_kind(parsed: Option<&Value>) -> DelegateKind {
    match parsed
        .and_then(|v| v.get("kind"))
        .and_then(Value::as_str)
        .map(str::trim)
    {
        Some(kind) if kind.eq_ignore_ascii_case("edit") => DelegateKind::Edit,
        _ => DelegateKind::Author,
    }
}

/// Check whether two declared workspace scopes cannot overlap. Unknown scopes
/// remain conservative: worktree isolation prevents concurrent writes, but the
/// destination merge must not guess when either task omitted its target paths.
pub(crate) fn file_sets_disjoint(
    a: &std::collections::BTreeSet<String>,
    b: &std::collections::BTreeSet<String>,
) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    a.iter().all(|left| {
        b.iter().all(|right| {
            left != right
                && !left.starts_with(&format!("{right}/"))
                && !right.starts_with(&format!("{left}/"))
        })
    })
}

impl crate::Agent {
    /// Prepare a delegate subagent job: check budget, extract the runner and
    /// verify command, and extract the file set from the task description.
    /// Returns `None` if the budget is exhausted, no runner is attached, or
    /// the task is empty.
    pub(crate) fn prepare_delegate(&mut self, arguments: &str) -> Option<(DelegateJob, u64)> {
        let parsed = serde_json::from_str::<Value>(arguments).ok();
        let task = parsed
            .as_ref()
            .and_then(|v| v.get("task").and_then(Value::as_str))
            .unwrap_or_default()
            .to_string();
        if task.trim().is_empty() {
            return None;
        }
        let session_limit = delegate_turn_limit();
        if self.subagents.delegate_turn_used >= session_limit {
            return None;
        }
        let runner = self.subagents.delegate_runner.clone()?;
        let n = self
            .subagents
            .try_begin_delegate(session_limit)
            .expect("budget checked above");
        let verify = parsed
            .as_ref()
            .and_then(|v| v.get("verify").and_then(Value::as_str))
            .map(str::to_string)
            .filter(|s| !s.trim().is_empty());
        let file_set =
            structured_file_set(parsed.as_ref()).unwrap_or_else(|| extract_file_set(&task));
        let route = self.route_for_kind(delegate_kind(parsed.as_ref()));
        let ledger_revision = self.runtime.ledger().revision();
        Some((
            DelegateJob {
                slot: n,
                task,
                verify,
                runner,
                route,
                cancellation: crate::TurnCancellation::new(),
                file_set,
            },
            ledger_revision,
        ))
    }

    /// The configured executor route for `delegate` children (team roles).
    /// All-`None` inherits the driver's provider/model.
    pub(crate) fn delegate_route(&self) -> crate::SubagentRoute {
        crate::SubagentRoute {
            model: self.config.subagents.delegate_model.clone(),
            base_url: self.config.subagents.delegate_endpoint.clone(),
            api_key: self.config.subagents.delegate_endpoint_key.clone(),
        }
    }

    /// The route for a delegate call of `kind`: mechanical edits ride the
    /// editor lane when one is configured (team-bench: small fast models win
    /// precise edits; only big coders author reliably), everything else — and
    /// edits with no editor set — rides the delegate route.
    pub(crate) fn route_for_kind(&self, kind: DelegateKind) -> crate::SubagentRoute {
        if kind == DelegateKind::Edit {
            let sub = &self.config.subagents;
            if sub.editor_model.is_some() || sub.editor_endpoint.is_some() {
                return crate::SubagentRoute {
                    model: sub.editor_model.clone(),
                    base_url: sub.editor_endpoint.clone(),
                    api_key: sub.editor_endpoint_key.clone(),
                };
            }
        }
        self.delegate_route()
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
        // Budget before runner so exhausted turns get a clear budget message
        // even when a runner is attached (and tests that only set the counter).
        let session_limit = delegate_turn_limit();
        if self.subagents.delegate_turn_used >= session_limit {
            return delegate_tool_outcome(
                format!(
                    "delegate budget exhausted ({session_limit} this turn); \
                     implement the rest directly for this turn."
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
            .try_begin_delegate(session_limit)
            .expect("budget checked above");

        let verify = parsed
            .as_ref()
            .and_then(|v| v.get("verify").and_then(Value::as_str))
            .map(str::to_string)
            .filter(|s| !s.trim().is_empty());
        let summary: String = task.chars().take(72).collect();
        let ellipsis = if task.chars().count() > 72 { "…" } else { "" };
        ui.subagent_note(&format!("↳ delegate subagent {n}: {summary}{ellipsis}"));

        let ledger_revision = self.runtime.ledger().revision();
        // Route-aware even on the direct tool path: `/team delegate|editor`
        // assignments must apply here exactly as they do to background tasks
        // (plain `run` silently dropped the team route).
        let route = self.route_for_kind(delegate_kind(parsed.as_ref()));
        let outcome = runner
            .run_routed(
                &task,
                verify.as_deref(),
                &route,
                crate::TurnCancellation::new(),
            )
            .await;
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
        route,
        cancellation,
        file_set: _,
    } = job;
    let outcome = runner
        .run_routed(&task, verify.as_deref(), &route, cancellation)
        .await;
    DelegateJobResult { slot, outcome }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ConcurrentRunner {
        active: AtomicUsize,
        peak: AtomicUsize,
        release: tokio::sync::Semaphore,
    }

    #[async_trait::async_trait]
    impl crate::DelegateRunner for ConcurrentRunner {
        async fn run(&self, task: &str, _verify: Option<&str>) -> crate::DelegateOutcome {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            let permit = self.release.acquire().await.unwrap();
            permit.forget();
            self.active.fetch_sub(1, Ordering::SeqCst);
            crate::DelegateOutcome {
                status: hi_tools::ToolStatus::Failed,
                applied: false,
                changed_files: Vec::new(),
                summary: format!("finished {task}"),
            }
        }
    }

    #[test]
    fn structured_scope_is_authoritative() {
        let parsed = serde_json::json!({
            "task": "Update src/wrong.rs",
            "scope": ["crates/hi-agent", "docs/guide.md"]
        });
        let paths = structured_file_set(Some(&parsed)).unwrap();
        assert_eq!(
            paths,
            std::collections::BTreeSet::from([
                "crates/hi-agent".to_string(),
                "docs/guide.md".to_string()
            ])
        );
        assert!(!paths.contains("src/wrong.rs"));
    }

    #[test]
    fn configured_limits_are_clamped() {
        assert_eq!(configured_delegate_limit(Some("999"), 4), 16);
        assert_eq!(configured_delegate_limit(Some("0"), 4), 4);
        assert_eq!(configured_delegate_limit(Some("bad"), 4), 4);
    }

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
    fn extract_file_set_finds_directory_scopes() {
        let paths = extract_file_set("Update crates/hi-agent and docs/guides independently");
        assert!(paths.contains("crates/hi-agent"));
        assert!(paths.contains("docs/guides"));
    }

    #[test]
    fn directory_and_child_file_scopes_overlap() {
        let a = extract_file_set("Update crates/hi-agent");
        let b = extract_file_set("Update crates/hi-agent/src/lib.rs");
        assert!(!file_sets_disjoint(&a, &b));
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

    #[tokio::test]
    async fn delegate_jobs_can_fill_the_four_job_wave() {
        let runner = std::sync::Arc::new(ConcurrentRunner {
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            release: tokio::sync::Semaphore::new(0),
        });
        let mut tasks = Vec::new();
        for slot in 1..=MAX_PARALLEL_DELEGATES as u32 {
            let runner_for_job: std::sync::Arc<dyn crate::DelegateRunner> = runner.clone();
            tasks.push(tokio::spawn(run_delegate_job(DelegateJob {
                slot,
                task: format!("update src/module-{slot}.rs"),
                verify: None,
                runner: runner_for_job,
                route: crate::SubagentRoute::default(),
                cancellation: crate::TurnCancellation::new(),
                file_set: std::collections::BTreeSet::from([format!("src/module-{slot}.rs")]),
            })));
        }
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while runner.peak.load(Ordering::SeqCst) < MAX_PARALLEL_DELEGATES {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("all delegate jobs should enter the runner concurrently");
        assert_eq!(runner.peak.load(Ordering::SeqCst), 4);
        runner.release.add_permits(MAX_PARALLEL_DELEGATES);
        for task in tasks {
            task.await.unwrap();
        }
    }

    #[test]
    fn empty_file_sets_not_disjoint() {
        let a: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let b = extract_file_set("Update src/foo.rs");
        // Unknown scope cannot prove isolation, so execution remains serial.
        assert!(!file_sets_disjoint(&a, &b));
        assert!(!file_sets_disjoint(&b, &a));
    }
}
