//! Built-in tools and workspace effects for the interactive agent.
//!
//! # Module layout
//!
//! - [`protocol`] — agent tool protocol and workspace effects (catalog, execute,
//!   checkpoint, guard, sandbox, transactions, process helpers).
//! - [`infra`] — product infrastructure used by the CLI/TUI but not required to
//!   speak the tool protocol (HF download/run, local server lifecycle, web
//!   fetch/search, repo-map orientation, polyglot fast-feedback, LSP status).
//!
//! The crate root still re-exports the historical public surface so existing
//! `hi_tools::…` call sites keep compiling. Prefer `hi_tools::protocol::…` /
//! `hi_tools::infra::…` in new code when the boundary matters.
//!
//! # Firewall policy (soft)
//!
//! | Prefer | Contents |
//! |--------|----------|
//! | [`protocol`] | tool catalog/execute, checkpoint, guard, sandbox, worktree, process, transactions, `ToolOutcome` family |
//! | [`infra`] | hf, local_server, web, repo_map, fast_feedback, LSP status |
//!
//! `hi-agent` turn/verify/mutation paths should depend on protocol symbols;
//! product/CLI orientation and download helpers belong under infra. Root paths
//! remain valid indefinitely for compatibility.
//!
//! Richer capabilities still come from subprocess CLI tools the model invokes
//! via `bash` — not a plugin runtime — so the advertised tool set stays small.

use serde::{Deserialize, Serialize};

/// Agent tool protocol and recoverable workspace effects.
pub mod protocol {
    pub mod checkpoint {
        pub use crate::checkpoint::*;
    }
    pub mod guard {
        pub use crate::guard::*;
    }
    pub mod sandbox {
        pub use crate::sandbox::*;
    }
    pub mod stub_scan {
        pub use crate::stub_scan::*;
    }
    pub mod worktree {
        pub use crate::worktree::*;
    }
    pub use crate::attribution::{AttrKind, Attribution, parse_attributions};
    pub use crate::background::BackgroundRegistry;
    pub use crate::condense::condense_diagnostics;
    pub use crate::paths::ReadCache;
    pub use crate::process::{AdoptableOutcome, ProcessExecution, ProcessRunner, RunningChild};
    pub use crate::structured_failure::{
        StructuredFailure, format_structured_failure, format_structured_failure_with_limit,
        render_cause_section,
    };
    pub use crate::tools::{
        MAX_WRITE_OVERWRITE_BYTES, MINIMAL_TOOL_SPECS, PreparedMutation, TOOL_CATALOG, TOOL_SPECS,
        ToolCapability, ToolMetadata, commit_in, delegate_tool_spec, execute_in_runtime,
        execute_prepared_in_runtime, execute_streaming_in_runtime, explore_tool_spec, fast_check_for,
        is_coordination, is_filesystem_mutating, is_known_tool, is_read_only,
        prepare_mutation_in_with_state, prepare_verify_workdir, run_check_in, run_fast_check_in,
        target_path, tool_metadata, working_tree_diff_in, working_tree_diff_plain_in,
    };
    pub use crate::transaction::{MutationPlan, PlannedFileMutation, recover_workspace_transactions};
}

/// Product infrastructure outside the core tool protocol.
pub mod infra {
    pub use crate::fast_feedback::{
        CargoCheckOutcome, CargoCommandOutcome, affected_any_package_dirs, affected_cargo_package_dirs,
        affected_go_package_dirs, affected_javascript_package_dirs, affected_package_dirs,
        affected_python_package_dirs, format_lsp_error_feedback, go_source_paths,
        is_python_package_root, javascript_source_paths, lsp_source_paths, python_source_paths,
        run_affected_cargo_checks, run_affected_cargo_tests, run_affected_polyglot_checks,
        run_affected_polyglot_tests, rust_source_paths,
    };
    pub use crate::hf::{
        HfCommandResult, HfCommandState, HfMlxRun, download_repo_keep_foreground, handle_hf_command,
        handle_hf_command_result,
    };
    pub use crate::local_server::{
        LocalServerHandle, skeptic_model_dir, start_local_server, stop_all_local_servers,
        stop_local_server,
    };
    pub use crate::lsp::lsp_status_report_for;
    pub use crate::repo_map::{RepoMapCache, orientation_for_task, ranked_paths_for_task};
    pub use crate::web::{run_web_fetch, run_web_search};
}

// --- implementation modules (private layout; public surface via root + namespaces) ---
pub mod checkpoint;
pub mod guard;
mod lsp;
pub mod sandbox;
pub mod stub_scan;
pub mod worktree;

mod attribution;
mod background;
mod condense;
mod edit;
mod effects;
mod fast_feedback;
mod hf;
mod internal_snapshot;
mod local_server;
mod paths;
mod process;
mod read;
mod repo_map;
mod structured_failure;
mod catalog;
mod tools;
mod transaction;
mod web;

pub use background::BackgroundRegistry;
pub use condense::condense_diagnostics;
pub use hf::{
    HfCommandResult, HfCommandState, HfMlxRun, download_repo_keep_foreground, handle_hf_command,
    handle_hf_command_result,
};
pub use local_server::{
    LocalServerHandle, skeptic_model_dir, start_local_server, stop_all_local_servers,
    stop_local_server,
};
pub use lsp::lsp_status_report_for;
pub use fast_feedback::{
    CargoCheckOutcome, CargoCommandOutcome, affected_any_package_dirs, affected_cargo_package_dirs,
    affected_go_package_dirs, affected_javascript_package_dirs, affected_package_dirs,
    affected_python_package_dirs, format_lsp_error_feedback, go_source_paths,
    is_python_package_root, javascript_source_paths, lsp_source_paths, python_source_paths,
    run_affected_cargo_checks, run_affected_cargo_tests, run_affected_polyglot_checks,
    run_affected_polyglot_tests, rust_source_paths,
};
pub use paths::ReadCache;
pub use process::{AdoptableOutcome, ProcessExecution, ProcessRunner, RunningChild};
pub use repo_map::{RepoMapCache, orientation_for_task, ranked_paths_for_task};
pub use tools::{
    MAX_WRITE_OVERWRITE_BYTES, MINIMAL_TOOL_SPECS, PreparedMutation, TOOL_CATALOG, TOOL_SPECS,
    ToolCapability, ToolMetadata, commit_in, delegate_tool_spec, execute_in_runtime,
    execute_prepared_in_runtime, execute_streaming_in_runtime, explore_tool_spec, fast_check_for,
    is_coordination, is_filesystem_mutating, is_known_tool, is_read_only,
    prepare_mutation_in_with_state, prepare_verify_workdir, run_check_in, run_fast_check_in,
    target_path, tool_metadata, working_tree_diff_in, working_tree_diff_plain_in,
};
#[cfg(test)]
pub(crate) use tools::{execute, execute_in, preview_edit_in};
pub use transaction::{MutationPlan, PlannedFileMutation, recover_workspace_transactions};
pub use web::{run_web_fetch, run_web_search};

pub use attribution::{AttrKind, Attribution, parse_attributions};
pub use structured_failure::{
    StructuredFailure, format_structured_failure, format_structured_failure_with_limit,
    render_cause_section,
};

// `ToolOutcome`'s constructors (`plain`/`shown`/`planned`) are crate-private and
// used by `tools`/`read`; they live here because the type is part of the public
// API and is small enough to stay shared.

/// Machine-readable completion state for a tool invocation.
///
/// Human-facing prose in [`ToolOutcome::content`] is never the source of truth
/// for success. Callers must use this status (and, for commands, the associated
/// [`ProcessOutcome`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    Succeeded,
    Failed,
    Denied,
    TimedOut,
    Cancelled,
}

/// Machine-readable lifecycle state for a detached process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundState {
    Started,
    Running,
    Exited,
    Killed,
    Failed,
}

/// Structured state returned by `bash`, `bash_output`, and `bash_kill` for a
/// detached process. A started or still-running process is never verification
/// evidence; callers should use [`ToolOutcome::satisfies_validation`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackgroundOutcome {
    pub id: String,
    pub state: BackgroundState,
    pub exit_code: Option<i32>,
}

/// Structured data captured from a completed foreground process.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessOutcome {
    /// `None` when no exit code exists (for example, a signal or timeout).
    pub exit_code: Option<i32>,
    pub stdout_summary: String,
    pub stderr_summary: String,
    pub duration_ms: u64,
}

/// The kind of workspace mutation represented by a [`FileChange`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileChangeKind {
    Create,
    Modify,
    Delete,
}

/// Exact before/after metadata for one file affected by a tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileChange {
    pub path: String,
    pub kind: FileChangeKind,
    pub before_digest: Option<String>,
    pub after_digest: Option<String>,
    pub before_len: Option<u64>,
    pub after_len: Option<u64>,
    pub before_mode: Option<u32>,
    pub after_mode: Option<u32>,
}

/// Workspace effects independently of the model-facing result text.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolEffects {
    pub mutation_attempted: bool,
    pub mutation_applied: bool,
    pub file_changes: Vec<FileChange>,
}

/// Whether output was clipped before being returned to the model/UI.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TruncationState {
    #[default]
    Complete,
    Truncated {
        original_bytes: u64,
        retained_bytes: u64,
    },
}

/// The result of a tool call, split into `content` shown to the model and an
/// optional richer `display` for the UI (e.g. a colored diff). This keeps
/// edit/write feedback terse for the model while showing the user what changed.
/// `plan`, when set, drives the live plan tracker instead of a transcript echo.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutcome {
    pub content: String,
    pub display: Option<String>,
    pub plan: Option<Vec<PlanStep>>,
    pub status: ToolStatus,
    pub process: Option<ProcessOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<BackgroundOutcome>,
    pub effects: ToolEffects,
    pub truncation: TruncationState,
}

/// Bound arbitrary model-facing tool content to the shared result budget.
///
/// The returned byte counts describe the original string and the exact
/// returned string, including the human-readable truncation marker. The marker
/// can make a just-over-limit result byte-larger than its input, so truncation
/// is determined from the character budget rather than byte-length ordering.
pub fn bound_tool_content(content: String) -> (String, TruncationState) {
    let original_bytes = content.len() as u64;
    let was_truncated = content.chars().count() > *crate::condense::MAX_OUTPUT_CHARS;
    let bounded = crate::condense::truncate(&content);
    let truncation = if was_truncated {
        TruncationState::Truncated {
            original_bytes,
            retained_bytes: bounded.len() as u64,
        }
    } else {
        TruncationState::Complete
    };
    (bounded, truncation)
}

impl ToolOutcome {
    pub(crate) fn plain(content: String) -> Self {
        Self {
            content,
            display: None,
            plan: None,
            status: ToolStatus::Succeeded,
            process: None,
            background: None,
            effects: ToolEffects::default(),
            truncation: TruncationState::Complete,
        }
    }

    /// Plain model/UI content clipped to the shared per-tool context budget,
    /// with authoritative truncation metadata. Use this for results whose
    /// producer can return an unbounded body (notably a repository-wide diff).
    pub(crate) fn bounded_plain(content: String) -> Self {
        let (bounded, truncation) = bound_tool_content(content);
        let mut outcome = Self::plain(bounded);
        outcome.truncation = truncation;
        outcome
    }

    pub(crate) fn failed(content: String) -> Self {
        Self {
            status: ToolStatus::Failed,
            ..Self::plain(content)
        }
    }

    pub(crate) fn denied(content: String) -> Self {
        Self {
            status: ToolStatus::Denied,
            ..Self::plain(content)
        }
    }

    pub(crate) fn shown(content: String, display: String) -> Self {
        Self {
            content,
            display: Some(display),
            plan: None,
            status: ToolStatus::Succeeded,
            process: None,
            background: None,
            effects: ToolEffects::default(),
            truncation: TruncationState::Complete,
        }
    }

    /// A result that updates the user-facing plan checklist. The model sees only
    /// `content` (a terse confirmation); the steps drive the pinned tracker.
    pub(crate) fn planned(content: String, steps: Vec<PlanStep>) -> Self {
        Self {
            content,
            display: None,
            plan: Some(steps),
            status: ToolStatus::Succeeded,
            process: None,
            background: None,
            effects: ToolEffects::default(),
            truncation: TruncationState::Complete,
        }
    }

    /// Whether this outcome may count as successful validation evidence.
    /// Detached work must have exited successfully, and foreground commands
    /// must expose a successful exit code rather than only optimistic prose.
    pub fn satisfies_validation(&self) -> bool {
        if self.status != ToolStatus::Succeeded {
            return false;
        }
        if let Some(process) = &self.process
            && process.exit_code != Some(0)
        {
            return false;
        }
        match &self.background {
            None => true,
            Some(background) => {
                background.state == BackgroundState::Exited && background.exit_code == Some(0)
            }
        }
    }
}

/// One line of the task plan/checklist surfaced by the `update_plan` tool. The
/// model resubmits the whole list (with updated statuses) on every call, so
/// there is no per-step index or state to drift out of sync.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanStep {
    pub title: String,
    pub status: PlanStatus,
}

/// The progress state of a single [`PlanStep`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanStatus {
    Pending,
    Active,
    Done,
}

impl PlanStatus {
    /// Map the model's free-form status string onto a state, tolerating the
    /// common synonyms models reach for ("in_progress", "completed", "todo", …).
    pub(crate) fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "done" | "complete" | "completed" | "finished" => PlanStatus::Done,
            "active" | "in_progress" | "in-progress" | "doing" | "current" | "started" => {
                PlanStatus::Active
            }
            _ => PlanStatus::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    // Integration-style tests that exercise the public `execute` entry point
    // across the split modules (dispatch + read/grep/list/bash + edit + plan).
    use crate::{PlanStatus, execute};

    #[test]
    fn only_completed_successful_processes_satisfy_validation() {
        let mut outcome = super::ToolOutcome::plain("started".into());
        outcome.background = Some(super::BackgroundOutcome {
            id: "bg_1".into(),
            state: super::BackgroundState::Started,
            exit_code: None,
        });
        assert!(!outcome.satisfies_validation());

        outcome.background = Some(super::BackgroundOutcome {
            id: "bg_1".into(),
            state: super::BackgroundState::Exited,
            exit_code: Some(0),
        });
        assert!(outcome.satisfies_validation());

        outcome.background = None;
        outcome.process = Some(super::ProcessOutcome {
            exit_code: Some(1),
            stdout_summary: String::new(),
            stderr_summary: String::new(),
            duration_ms: 1,
        });
        assert!(!outcome.satisfies_validation());
    }

    #[tokio::test]
    async fn multi_edit_applies_in_order_and_is_atomic() {
        let dir = std::env::temp_dir().join(format!("hi-medit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.txt");
        std::fs::write(&path, "one\ntwo\nthree\n").unwrap();
        // Two edits in one atomic call.
        let args = r#"{"path":"x.txt","edits":[{"old_string":"one","new_string":"1"},{"old_string":"three","new_string":"3"}]}"#;
        let out = crate::execute_in(&dir, "multi_edit", args).await;
        assert!(out.content.contains("Applied 2 edits"), "{}", out.content);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "1\ntwo\n3\n");

        // A failing edit in the batch leaves the file untouched (atomic).
        let bad = r#"{"path":"x.txt","edits":[{"old_string":"1","new_string":"X"},{"old_string":"nope","new_string":"Y"}]}"#;
        let out = crate::execute_in(&dir, "multi_edit", bad).await;
        assert!(
            out.content.contains("Error"),
            "should fail: {}",
            out.content
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "1\ntwo\n3\n",
            "file unchanged after a failed batch"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn failures_and_denials_are_typed_not_inferred_from_prose() {
        let malformed = execute("read", "not json").await;
        assert_eq!(malformed.status, crate::ToolStatus::Failed);

        let denied = execute("bash", r#"{"command":"rm -rf /"}"#).await;
        assert_eq!(denied.status, crate::ToolStatus::Denied);
        assert!(denied.effects.mutation_attempted);
        assert!(!denied.effects.mutation_applied);
    }

    #[tokio::test]
    async fn nonzero_process_has_structured_failure_and_separate_streams() {
        let outcome = execute(
            "bash",
            r#"{"command":"printf out; printf err >&2; exit 9"}"#,
        )
        .await;
        assert_eq!(outcome.status, crate::ToolStatus::Failed);
        let process = outcome.process.expect("foreground process outcome");
        assert_eq!(process.exit_code, Some(9));
        assert_eq!(process.stdout_summary, "out");
        assert_eq!(process.stderr_summary, "err");
    }

    #[tokio::test]
    async fn successful_write_reports_exact_effects() {
        let dir = std::env::temp_dir().join(format!("hi-effects-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let arguments = serde_json::json!({"path": "file.txt", "content": "hello\n"}).to_string();
        let outcome = crate::execute_in(&dir, "write", &arguments).await;
        assert_eq!(
            outcome.status,
            crate::ToolStatus::Succeeded,
            "{}",
            outcome.content
        );
        assert!(outcome.effects.mutation_attempted);
        assert!(outcome.effects.mutation_applied);
        assert_eq!(outcome.effects.file_changes.len(), 1);
        let change = &outcome.effects.file_changes[0];
        assert_eq!(change.kind, crate::FileChangeKind::Create);
        assert!(change.before_digest.is_none());
        assert!(
            change
                .after_digest
                .as_deref()
                .is_some_and(|digest| digest.starts_with("sha256:"))
        );
        assert_eq!(change.after_len, Some(6));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn foreground_bash_reports_exact_effects_even_when_process_fails() {
        let dir = unique_test_dir("hi-bash-effects-failed");
        std::fs::create_dir_all(&dir).unwrap();

        let outcome = crate::execute_in(
            &dir,
            "bash",
            r#"{"command":"printf 'written\\n' > result.txt; exit 7"}"#,
        )
        .await;

        assert_eq!(outcome.status, crate::ToolStatus::Failed);
        assert_eq!(
            outcome
                .process
                .as_ref()
                .and_then(|process| process.exit_code),
            Some(7)
        );
        assert!(outcome.effects.mutation_attempted);
        assert!(outcome.effects.mutation_applied);
        assert_eq!(outcome.effects.file_changes.len(), 1);
        let change = &outcome.effects.file_changes[0];
        assert_eq!(change.path, "result.txt");
        assert_eq!(change.kind, crate::FileChangeKind::Create);
        assert_eq!(change.before_digest, None);
        assert_eq!(change.after_len, Some(8));
        assert!(
            change
                .after_digest
                .as_deref()
                .is_some_and(|digest| digest.starts_with("sha256:"))
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn successful_noop_bash_is_attempted_but_not_applied() {
        let dir = unique_test_dir("hi-bash-effects-noop");
        std::fs::create_dir_all(&dir).unwrap();

        let outcome = crate::execute_in(&dir, "bash", r#"{"command":"true"}"#).await;

        assert_eq!(outcome.status, crate::ToolStatus::Succeeded);
        assert!(outcome.effects.mutation_attempted);
        assert!(!outcome.effects.mutation_applied);
        assert!(outcome.effects.file_changes.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn interactive_bash_denial_is_a_typed_attempt_without_effects() {
        let dir = unique_test_dir("hi-bash-effects-interactive");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("app.py"),
            "from textual.app import App\nApp().run()\n",
        )
        .unwrap();

        let outcome = crate::execute_in(&dir, "bash", r#"{"command":"python3 app.py"}"#).await;

        assert_eq!(outcome.status, crate::ToolStatus::Denied);
        assert!(outcome.effects.mutation_attempted);
        assert!(!outcome.effects.mutation_applied);
        assert!(outcome.effects.file_changes.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_effect_snapshot_failure_is_typed_and_prevents_execution() {
        use std::os::unix::net::UnixListener;

        let dir = unique_test_dir("hi-bash-effects-special");
        std::fs::create_dir_all(&dir).unwrap();
        let listener = UnixListener::bind(dir.join("live.sock")).unwrap();

        let outcome = crate::execute_in(
            &dir,
            "bash",
            r#"{"command":"printf should-not-run > marker"}"#,
        )
        .await;

        assert_eq!(outcome.status, crate::ToolStatus::Failed);
        assert!(outcome.effects.mutation_attempted);
        assert!(!outcome.satisfies_validation());
        assert!(outcome.content.contains("special workspace entry"));
        assert!(!dir.join("marker").exists());
        drop(listener);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn post_process_effect_snapshot_failure_is_not_reported_as_no_change() {
        let dir = unique_test_dir("hi-bash-effects-post-special");
        std::fs::create_dir_all(&dir).unwrap();

        let outcome =
            crate::execute_in(&dir, "bash", r#"{"command":"mkfifo generated.pipe"}"#).await;

        assert_eq!(
            outcome
                .process
                .as_ref()
                .and_then(|process| process.exit_code),
            Some(0),
            "the process itself completed: {}",
            outcome.content
        );
        assert_eq!(outcome.status, crate::ToolStatus::Failed);
        assert!(outcome.effects.mutation_attempted);
        assert!(
            outcome.effects.mutation_applied,
            "an unavailable postimage must be conservative, not a false clean result"
        );
        assert!(outcome.effects.file_changes.is_empty());
        assert!(outcome.content.contains("infrastructure failure"));
        assert!(!outcome.satisfies_validation());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn background_bash_reports_and_seals_terminal_file_effects() {
        let dir = unique_test_dir("hi-background-effects");
        let state = dir.join(".hi/state");
        std::fs::create_dir_all(&state).unwrap();
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&dir).unwrap());
        let background = crate::BackgroundRegistry::default();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());

        let started = crate::execute_in_runtime(
            &dir,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "bash",
            r#"{"command":"printf background > bg.txt","run_in_background":true}"#,
        )
        .await;
        assert_eq!(started.status, crate::ToolStatus::Succeeded);
        assert!(started.effects.mutation_attempted);
        assert!(!started.effects.mutation_applied);
        assert!(!started.satisfies_validation());
        let id = started.background.as_ref().unwrap().id.clone();

        let terminal = loop {
            let polled = crate::execute_in_runtime(
                &dir,
                &state,
                &lsp,
                &background,
                &cache,
                &repo_map,
                "bash_output",
                &serde_json::json!({"id": id}).to_string(),
            )
            .await;
            if polled
                .background
                .as_ref()
                .is_some_and(|state| state.state == crate::BackgroundState::Exited)
            {
                break polled;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        assert_eq!(terminal.status, crate::ToolStatus::Succeeded);
        assert!(terminal.effects.mutation_attempted);
        assert!(terminal.effects.mutation_applied);
        assert_eq!(terminal.effects.file_changes.len(), 1);
        assert_eq!(terminal.effects.file_changes[0].path, "bg.txt");

        // The first terminal observation seals attribution. An unrelated edit
        // after exit must not alter the process's recorded effects.
        let sealed = terminal.effects.clone();
        std::fs::write(dir.join("later.txt"), "external\n").unwrap();
        let repolled = crate::execute_in_runtime(
            &dir,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "bash_output",
            &serde_json::json!({"id": id}).to_string(),
        )
        .await;
        assert_eq!(repolled.effects, sealed);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn background_kill_waits_for_reap_and_reports_effects() {
        let dir = unique_test_dir("hi-background-kill-effects");
        let state = dir.join(".hi/state");
        std::fs::create_dir_all(&state).unwrap();
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&dir).unwrap());
        let background = crate::BackgroundRegistry::default();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
        let started = crate::execute_in_runtime(
            &dir,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "bash",
            r#"{"command":"printf killed > killed.txt; sleep 600","run_in_background":true}"#,
        )
        .await;
        let id = started.background.as_ref().unwrap().id.clone();
        for _ in 0..400 {
            if dir.join("killed.txt").exists() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert!(
            dir.join("killed.txt").exists(),
            "background command should have written killed.txt before kill"
        );

        let killed = crate::execute_in_runtime(
            &dir,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "bash_kill",
            &serde_json::json!({"id": id}).to_string(),
        )
        .await;
        // Lifecycle is authoritative: a successful kill is Cancelled even if
        // effect inspection is slow under suite load.
        assert_eq!(
            killed.status,
            crate::ToolStatus::Cancelled,
            "kill status={:?} content={} effects={:?}",
            killed.status,
            killed.content,
            killed.effects
        );
        assert_eq!(
            killed.background.as_ref().map(|state| state.state),
            Some(crate::BackgroundState::Killed)
        );
        assert!(killed.effects.mutation_attempted);
        // Effects may be empty if inspection failed; when present they must
        // include the file written before kill.
        if killed.effects.mutation_applied {
            assert!(
                killed
                    .effects
                    .file_changes
                    .iter()
                    .any(|c| c.path == "killed.txt"),
                "expected killed.txt in {:?}",
                killed.effects.file_changes
            );
        }
        assert!(!killed.satisfies_validation());
        let _ = std::fs::remove_dir_all(dir);
    }

    fn unique_test_dir(label: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    #[tokio::test]
    async fn explicit_roots_do_not_share_files_or_process_cwds() {
        let base = std::env::temp_dir().join(format!("hi-two-roots-{}", std::process::id()));
        let one = base.join("one");
        let two = base.join("two");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&one).unwrap();
        std::fs::create_dir_all(&two).unwrap();
        std::fs::write(one.join("value"), "one").unwrap();
        std::fs::write(two.join("value"), "two").unwrap();

        let read_one = crate::execute_in(&one, "read", r#"{"path":"value"}"#).await;
        let read_two = crate::execute_in(&two, "read", r#"{"path":"value"}"#).await;
        assert!(read_one.content.contains("one"));
        assert!(read_two.content.contains("two"));

        let pwd_one = crate::execute_in(&one, "bash", r#"{"command":"pwd"}"#).await;
        let pwd_two = crate::execute_in(&two, "bash", r#"{"command":"pwd"}"#).await;
        // `pwd` reports the physical working directory, so compare against the
        // canonical form — on macOS the temp dir is reached via the `/var` →
        // `/private/var` symlink alias.
        assert_eq!(
            pwd_one.process.unwrap().stdout_summary,
            one.canonicalize().unwrap().to_string_lossy()
        );
        assert_eq!(
            pwd_two.process.unwrap().stdout_summary,
            two.canonicalize().unwrap().to_string_lossy()
        );

        let write =
            crate::execute_in(&one, "write", r#"{"path":"value","content":"changed"}"#).await;
        assert_eq!(write.status, crate::ToolStatus::Succeeded);
        assert_eq!(
            std::fs::read_to_string(one.join("value")).unwrap(),
            "changed"
        );
        assert_eq!(std::fs::read_to_string(two.join("value")).unwrap(), "two");
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn grep_finds_a_known_symbol() {
        // Searches the repo's own source via the real tool.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let out = crate::execute_in(root, "grep", r#"{"pattern":"fn tool_specs"}"#).await;
        assert!(out.content.contains("tool_specs"), "grep: {}", out.content);
    }

    #[tokio::test]
    async fn grep_glob_filters_by_extension() {
        // `fn tool_specs` is in lib.rs but not in any .py file. With a `*.py`
        // glob, grep should find no matches; without a glob it finds the .rs hit.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let py =
            crate::execute_in(root, "grep", r#"{"pattern":"fn tool_specs","glob":"*.py"}"#).await;
        assert!(
            py.content.contains("no matches"),
            "glob *.py excludes the .rs hit: {}",
            py.content
        );
        let rs =
            crate::execute_in(root, "grep", r#"{"pattern":"fn tool_specs","glob":"*.rs"}"#).await;
        assert!(
            rs.content.contains("tool_specs"),
            "glob *.rs finds the hit: {}",
            rs.content
        );
    }

    #[tokio::test]
    async fn grep_skips_binary_files() {
        // A file with a NUL byte should be skipped, not error out.
        let dir = std::env::temp_dir().join(format!("hi-grep-bin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "hello world\n").unwrap();
        std::fs::write(dir.join("b.bin"), b"hello\x00world\n").unwrap();
        let out = crate::execute_in(&dir, "grep", r#"{"pattern":"hello","path":"."}"#).await;
        // Should find the match in a.txt but not error on b.bin.
        assert!(
            out.content.contains("a.txt"),
            "found text file: {}",
            out.content
        );
        assert!(
            !out.content.contains("Error"),
            "no error on binary: {}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn update_plan_records_steps_and_statuses() {
        let args = r#"{"steps":[
            {"title":"find leak","status":"done"},
            {"title":"fix walkers","status":"in_progress"},
            {"title":"add tests","status":"pending"}
        ]}"#;
        let out = execute("update_plan", args).await;
        let plan = out.plan.expect("plan is set");
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].status, PlanStatus::Done);
        // "in_progress" is a common model synonym for active.
        assert_eq!(plan[1].status, PlanStatus::Active);
        assert_eq!(plan[2].status, PlanStatus::Pending);
        // Model-facing content is a terse confirmation; the steps drive the
        // pinned tracker, so there's no transcript-echo display.
        assert!(out.content.contains("1/3"), "content: {}", out.content);
        assert!(out.display.is_none(), "plan should not echo to transcript");
    }

    #[tokio::test]
    async fn update_plan_rejects_empty() {
        let out = execute("update_plan", r#"{"steps":[]}"#).await;
        assert!(out.content.contains("Error"), "{}", out.content);
        assert!(out.plan.is_none());
    }

    #[tokio::test]
    async fn list_excludes_git_metadata_but_keeps_dotfiles() {
        // Regression: the model called `list` during a review and got flooded
        // with `.git/objects/...` paths. We walk hidden files (so real dotfiles
        // like `.github/` are visible) but must prune VCS metadata directories.
        let dir = std::env::temp_dir().join(format!("hi-list-vcs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(".git/objects/ab")).unwrap();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::create_dir_all(dir.join(".github/workflows")).unwrap();
        std::fs::write(dir.join(".git/config"), "[core]\n").unwrap();
        std::fs::write(dir.join(".git/objects/ab/cdef"), b"\x00\x01obj").unwrap();
        std::fs::write(dir.join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(dir.join(".github/workflows/ci.yml"), "name: ci\n").unwrap();

        let out = crate::execute_in(&dir, "list", r#"{"path":"."}"#).await;

        assert!(
            !out.content.contains("/.git/"),
            ".git metadata must be pruned: {}",
            out.content
        );
        assert!(
            out.content.contains("main.rs"),
            "normal files still listed: {}",
            out.content
        );
        // Legit dotfiles must survive — we prune VCS dirs, not all hidden entries.
        assert!(
            out.content.contains(".github"),
            "non-VCS dotfiles preserved: {}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_excludes_git_metadata() {
        // The same needle lives in a tracked file and in `.git` internals; grep
        // must surface only the tracked hit (covers whichever path runs — the
        // `rg` fast-path's exclusion globs or the inline walker's filter).
        let dir = std::env::temp_dir().join(format!("hi-grep-vcs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/lib.rs"), "let UNIQNEEDLE = 1;\n").unwrap();
        std::fs::write(dir.join(".git/config"), "UNIQNEEDLE\n").unwrap();

        let out = crate::execute_in(&dir, "grep", r#"{"pattern":"UNIQNEEDLE","path":"."}"#).await;

        assert!(
            out.content.contains("lib.rs"),
            "tracked hit found: {}",
            out.content
        );
        assert!(
            !out.content.contains(".git/config"),
            ".git must not be searched: {}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_rejects_binary_with_a_clear_message() {
        let dir = std::env::temp_dir().join(format!("hi-read-bin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob.bin");
        std::fs::write(&path, b"\x00\x01\x02binary\xff").unwrap();
        let out = crate::execute_in(&dir, "read", r#"{"path":"blob.bin"}"#).await;
        assert!(
            out.content.contains("binary file"),
            "clear binary message: {}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn list_includes_source_files() {
        // Pass an explicit path instead of relying on the process cwd: other
        // tests mutate the shared cwd via set_current_dir, which makes a
        // default-path `.` listing racy under parallel execution.
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let out = crate::execute_in(manifest, "list", r#"{"path":"."}"#).await;
        assert!(out.content.contains("lib.rs"), "list: {}", out.content);
    }

    #[tokio::test]
    async fn bash_refuses_catastrophic_but_runs_safe() {
        let refused = execute("bash", r#"{"command":"rm -rf /"}"#).await;
        assert!(refused.content.contains("refused"), "{}", refused.content);
        let ok = execute("bash", r#"{"command":"echo hello-guard"}"#).await;
        assert!(ok.content.contains("hello-guard"), "{}", ok.content);
    }

    #[tokio::test]
    async fn background_bash_round_trips_through_execute() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let state = std::env::temp_dir().join(format!("hi-bg-state-{}", std::process::id()));
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(root).unwrap());
        let background = crate::BackgroundRegistry::default();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
        // Start detached: execute returns a handle without waiting for exit.
        let started = crate::execute_in_runtime(
            root,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "bash",
            r#"{"command":"echo bg-roundtrip","run_in_background":true}"#,
        )
        .await;
        let id = started
            .content
            .split('`')
            .nth(1)
            .expect("handle id in start message")
            .to_string();
        assert!(id.starts_with("bg_"), "got: {}", started.content);

        // Poll until we see the line or the process has exited.
        let mut seen = String::new();
        for _ in 0..200 {
            let out = crate::execute_in_runtime(
                root,
                &state,
                &lsp,
                &background,
                &cache,
                &repo_map,
                "bash_output",
                &format!(r#"{{"id":"{id}"}}"#),
            )
            .await;
            seen.push_str(&out.content);
            if seen.contains("bg-roundtrip") && seen.contains("exited") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(seen.contains("bg-roundtrip"), "polled output: {seen:?}");

        // Killing an already-exited process is reported, not an error.
        let killed = crate::execute_in_runtime(
            root,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "bash_kill",
            &format!(r#"{{"id":"{id}"}}"#),
        )
        .await;
        assert!(!killed.content.starts_with("Error"), "{}", killed.content);
    }

    #[tokio::test]
    async fn bash_output_unknown_id_is_a_recoverable_error() {
        // Tool failures come back as content (so the model can recover), not panics.
        let out = execute("bash_output", r#"{"id":"bg_nope"}"#).await;
        assert!(out.content.starts_with("Error"), "{}", out.content);
        assert!(out.content.contains("bg_nope"), "{}", out.content);
    }
}
