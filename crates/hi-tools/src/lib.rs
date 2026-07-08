//! Built-in tools: `read`, `write`, `edit`, `bash`.
//!
//! Richer capabilities come from subprocess CLI tools the model invokes via
//! `bash` — not a plugin runtime — so this set stays intentionally small.

pub mod checkpoint;
pub mod guard;
mod lsp;

mod attribution;
mod background;
mod condense;
mod edit;
mod hf;
mod paths;
mod read;
mod tools;
mod web;

// Re-exports preserving the crate's pre-split public surface.
pub use background::{
    ids as background_process_ids, kill_all as kill_background_processes,
    kill_started_after as kill_background_processes_started_after,
};
pub use condense::condense_diagnostics;
pub use hf::{
    HfCommandResult, HfCommandState, HfMlxRun, download_repo_keep_foreground, handle_hf_command,
    handle_hf_command_result,
};
pub use lsp::{
    lsp_enabled, lsp_manager_handle, lsp_status, lsp_status_report, lsp_status_sync,
    set_lsp_manager, set_lsp_manager_arc, sync_lsp_document,
};
pub use paths::clear_read_cache;
pub use tools::{
    MINIMAL_TOOL_SPECS, TOOL_SPECS, commit, execute, execute_streaming, explore_tool_spec,
    fast_check_for, is_filesystem_mutating, is_read_only, prepare_verify_workdir, run_check,
    target_path, working_tree_diff, working_tree_diff_plain,
};
pub use web::{run_web_download, run_web_fetch, run_web_search};

pub use attribution::{AttrKind, Attribution, parse_attributions};

// `ToolOutput`'s constructors (`plain`/`shown`/`planned`) are crate-private and
// used by `tools`/`read`; they live here because the type is part of the public
// API and is small enough to stay shared.

/// The result of a tool call, split into `content` shown to the model and an
/// optional richer `display` for the UI (e.g. a colored diff). This keeps
/// edit/write feedback terse for the model while showing the user what changed.
/// `plan`, when set, drives the live plan tracker instead of a transcript echo.
pub struct ToolOutput {
    pub content: String,
    pub display: Option<String>,
    pub plan: Option<Vec<PlanStep>>,
}

impl ToolOutput {
    pub(crate) fn plain(content: String) -> Self {
        Self {
            content,
            display: None,
            plan: None,
        }
    }

    pub(crate) fn shown(content: String, display: String) -> Self {
        Self {
            content,
            display: Some(display),
            plan: None,
        }
    }

    /// A result that updates the user-facing plan checklist. The model sees only
    /// `content` (a terse confirmation); the steps drive the pinned tracker.
    pub(crate) fn planned(content: String, steps: Vec<PlanStep>) -> Self {
        Self {
            content,
            display: None,
            plan: Some(steps),
        }
    }
}

/// One line of the task plan/checklist surfaced by the `update_plan` tool. The
/// model resubmits the whole list (with updated statuses) on every call, so
/// there is no per-step index or state to drift out of sync.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanStep {
    pub title: String,
    pub status: PlanStatus,
}

/// The progress state of a single [`PlanStep`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    #[tokio::test]
    async fn multi_edit_applies_in_order_and_is_atomic() {
        let dir = std::env::temp_dir().join(format!("hi-medit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.txt");
        std::fs::write(&path, "one\ntwo\nthree\n").unwrap();
        let p = path.to_string_lossy();

        // Bypass path guard — temp files live outside the project workspace.
        let had_guard = std::env::var_os("HI_NO_PATH_GUARD");
        unsafe {
            std::env::set_var("HI_NO_PATH_GUARD", "1");
        }

        // Two edits in one atomic call.
        let args = format!(
            r#"{{"path":"{p}","edits":[{{"old_string":"one","new_string":"1"}},{{"old_string":"three","new_string":"3"}}]}}"#
        );
        let out = execute("multi_edit", &args).await;
        assert!(out.content.contains("Applied 2 edits"), "{}", out.content);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "1\ntwo\n3\n");

        // A failing edit in the batch leaves the file untouched (atomic).
        let bad = format!(
            r#"{{"path":"{p}","edits":[{{"old_string":"1","new_string":"X"}},{{"old_string":"nope","new_string":"Y"}}]}}"#
        );
        let out = execute("multi_edit", &bad).await;
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
        // Restore env.
        unsafe {
            if had_guard.is_none() {
                std::env::remove_var("HI_NO_PATH_GUARD");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_finds_a_known_symbol() {
        // Searches the repo's own source via the real tool.
        let out = execute("grep", r#"{"pattern":"fn tool_specs"}"#).await;
        assert!(out.content.contains("tool_specs"), "grep: {}", out.content);
    }

    #[tokio::test]
    async fn grep_glob_filters_by_extension() {
        // `fn tool_specs` is in lib.rs but not in any .py file. With a `*.py`
        // glob, grep should find no matches; without a glob it finds the .rs hit.
        let py = execute("grep", r#"{"pattern":"fn tool_specs","glob":"*.py"}"#).await;
        assert!(
            py.content.contains("no matches"),
            "glob *.py excludes the .rs hit: {}",
            py.content
        );
        let rs = execute("grep", r#"{"pattern":"fn tool_specs","glob":"*.rs"}"#).await;
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
        let p = dir.to_string_lossy();
        let out = execute("grep", &format!(r#"{{"pattern":"hello","path":"{p}"}}"#)).await;
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

        let p = dir.to_string_lossy();
        let out = execute("list", &format!(r#"{{"path":"{p}"}}"#)).await;

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

        let p = dir.to_string_lossy();
        let out = execute(
            "grep",
            &format!(r#"{{"pattern":"UNIQNEEDLE","path":"{p}"}}"#),
        )
        .await;

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
        let p = path.to_string_lossy();
        // Bypass path guard — temp files live outside the project workspace.
        let had_guard = std::env::var_os("HI_NO_PATH_GUARD");
        unsafe {
            std::env::set_var("HI_NO_PATH_GUARD", "1");
        }
        let out = execute("read", &format!(r#"{{"path":"{p}"}}"#)).await;
        unsafe {
            if had_guard.is_none() {
                std::env::remove_var("HI_NO_PATH_GUARD");
            }
        }
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
        let manifest = env!("CARGO_MANIFEST_DIR");
        let out = execute("list", &format!(r#"{{"path":"{manifest}"}}"#)).await;
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
        let _guard = crate::background::TEST_LOCK.lock().await;
        // Start detached: execute returns a handle without waiting for exit.
        let started = execute(
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
            let out = execute("bash_output", &format!(r#"{{"id":"{id}"}}"#)).await;
            seen.push_str(&out.content);
            if seen.contains("bg-roundtrip") && seen.contains("exited") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(seen.contains("bg-roundtrip"), "polled output: {seen:?}");

        // Killing an already-exited process is reported, not an error.
        let killed = execute("bash_kill", &format!(r#"{{"id":"{id}"}}"#)).await;
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
