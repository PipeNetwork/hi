use std::time::Duration;

use super::{
    BashArgs, RuntimeResources, definitely_read_only_shell, file_dump_read_arguments,
    managed_handoff_enabled, process_tool_outcome, run_bash_tool_with_auto_background,
};
use crate::{ProcessExecution, ProcessOutcome, ToolStatus, TruncationState};

struct RejectingLifecycle {
    live_writer_supported: bool,
}

#[test]
fn managed_handoff_requires_an_unlimited_command_and_workspace_capability() {
    assert!(managed_handoff_enabled(true, None, true));
    assert!(!managed_handoff_enabled(
        true,
        Some(Duration::from_secs(600)),
        true
    ));
    assert!(!managed_handoff_enabled(true, None, false));
    assert!(!managed_handoff_enabled(false, None, true));
}

#[async_trait::async_trait]
impl crate::BackgroundJobLifecycle for RejectingLifecycle {
    fn supports_effect(&self, effect: crate::BackgroundJobEffect) -> bool {
        effect != crate::BackgroundJobEffect::LiveWriter || self.live_writer_supported
    }

    async fn register(&self, _: crate::BackgroundJobRegistration) -> Result<(), String> {
        Err("background writers unavailable for this binding".into())
    }

    async fn observe_terminal(
        &self,
        _: &crate::BackgroundJobId,
        _: crate::BackgroundJobTerminal,
        _: Option<String>,
    ) -> Result<crate::BackgroundJobPublication, String> {
        unreachable!("a rejected job has no terminal lifecycle")
    }

    async fn pending(&self, _: &str) -> Vec<crate::BackgroundJobId> {
        Vec::new()
    }

    async fn settle_after_workspace(&self, _: &[crate::BackgroundJobId]) -> Result<(), String> {
        Ok(())
    }
}

#[test]
fn read_only_shell_allowlist_is_conservative() {
    for command in ["rg TODO src", "head -20 README.md", "printf 'done\\n'"] {
        assert!(definitely_read_only_shell(command), "{command:?}");
    }
    for command in [
        "echo hi > marker.txt",
        "sed -i s/old/new/ src/lib.rs",
        "find . -exec rm {} \\;",
        "sort -o sorted.txt input.txt",
        // Even observational Git commands can refresh the index, invoke
        // fsmonitor/textconv/external-diff helpers, or start a pager. Keep
        // them on the live-writer reconciliation path unless a future
        // broker can prove the complete invocation hermetic.
        "git status --short",
        "git -C nested/repo diff",
        "git diff --output=patch.txt",
        "git -C /tmp/repo diff",
        "./scripts/check.sh",
        "cargo test",
    ] {
        assert!(!definitely_read_only_shell(command), "{command:?}");
    }
}

#[test]
fn file_dump_commands_map_to_read_arguments() {
    fn parsed(command: &str) -> Option<serde_json::Value> {
        file_dump_read_arguments(command).map(|json| serde_json::from_str(&json).unwrap())
    }
    assert_eq!(
        parsed("cat SPEC.md"),
        Some(serde_json::json!({"path":"SPEC.md"}))
    );
    assert_eq!(
        parsed("sed -n '200,400p' SPEC.md"),
        Some(serde_json::json!({"path":"SPEC.md","offset":200,"limit":201}))
    );
    assert_eq!(
        parsed("head -n 50 crates/api/src/lib.rs"),
        Some(serde_json::json!({"path":"crates/api/src/lib.rs","limit":50}))
    );
    assert!(parsed("cat file | wc -l").is_none());
    assert!(parsed("sed -i s/a/b/ SPEC.md").is_none());
    assert!(parsed("echo hello").is_none());
    assert!(parsed("cat *.md").is_none());
    assert_eq!(
        parsed("printf -- '---\\n' && cat SPEC.md"),
        Some(serde_json::json!({"path":"SPEC.md"}))
    );
    assert_eq!(
        parsed("echo banner; cat SPEC.md"),
        Some(serde_json::json!({"path":"SPEC.md"}))
    );
    assert!(parsed("cat SPEC.md && rm SPEC.md").is_none());
    assert!(parsed("rm SPEC.md && cat SPEC.md").is_none());
    assert_eq!(
        parsed(r#"grep -n "" src/ws.rs"#),
        Some(serde_json::json!({"path":"src/ws.rs"}))
    );
    assert_eq!(
        parsed("grep -n '^' src/ws.rs | sed -n '206,300p'"),
        Some(serde_json::json!({"path":"src/ws.rs","offset":206,"limit":95}))
    );
    assert!(parsed(r#"grep -n TODO src/ws.rs"#).is_none());
    assert!(parsed(r#"grep -n "" src/ws.rs | wc -l"#).is_none());
    assert!(parsed("cat SPEC.md | wc -l").is_none());
}

#[test]
fn process_tool_outcome_separates_model_and_display_text() {
    let execution = ProcessExecution {
        status: ToolStatus::Succeeded,
        outcome: ProcessOutcome {
            exit_code: Some(0),
            stdout_summary: "\u{1b}[31mred\u{1b}[0m".into(),
            stderr_summary: String::new(),
            duration_ms: 1,
        },
        truncation: TruncationState::Complete,
    };

    let outcome = process_tool_outcome(execution, None);
    assert_eq!(outcome.content, "red");
    assert_eq!(outcome.display.as_deref(), Some("\u{1b}[31mred\u{1b}[0m"));
    assert_eq!(
        outcome.process.unwrap().stdout_summary,
        "red",
        "serialized process metadata stays plain"
    );
}

#[test]
fn pipeline_status_warning_prevents_false_exit_zero_inference() {
    let execution = ProcessExecution {
        status: ToolStatus::Succeeded,
        outcome: ProcessOutcome {
            exit_code: Some(0),
            stdout_summary: "exit=0".into(),
            stderr_summary: String::new(),
            duration_ms: 1,
        },
        truncation: TruncationState::Complete,
    };

    let outcome =
        process_tool_outcome(execution, Some("timeout 10 ./app | head -30; echo exit=$?"));
    assert!(outcome.content.contains("final command's status"));
    assert!(outcome.content.contains("Capture the program status"));
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn denied_managed_handoff_kills_reaps_and_returns_a_tool_failure() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state_root = directory.path().join("state");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&state_root).unwrap();
    let pid_file = root.join("child.pid");
    let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root).unwrap());
    let background = crate::BackgroundRegistry::default();
    background.set_foreground_handoff_budget(Some(Duration::from_secs(1)));
    background.set_job_lifecycle(std::sync::Arc::new(RejectingLifecycle {
        live_writer_supported: true,
    }));
    let read_cache = std::sync::Mutex::new(crate::ReadCache::new());
    let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
    let runner = crate::ProcessRunner::new(&root).unwrap();
    let command = format!("printf '%s' $$ > {}; sleep 600", pid_file.display());

    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        run_bash_tool_with_auto_background(
            &root,
            &state_root,
            RuntimeResources {
                process_runner: Some(&runner),
                lsp: &lsp,
                background: &background,
                read_cache: &read_cache,
                repo_map: &repo_map,
                repo_map_arc: None,
                mcp: None,
                memory: None,
                skill: None,
                hunk_tracker: None,
            },
            BashArgs {
                command,
                timeout: None,
                run_in_background: false,
            },
            &mut |_| {},
            true,
        ),
    )
    .await
    .expect("handoff rejection must reap promptly")
    .unwrap();

    assert_eq!(outcome.status, ToolStatus::Failed);
    assert!(
        outcome
            .content
            .contains("managed background handoff failed")
    );
    assert!(outcome.background.is_none());
    assert!(background.ids().is_empty());
    let pid: i32 = std::fs::read_to_string(pid_file)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH),
        "rejected handoff left foreground process {pid} alive"
    );
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unsupported_live_writer_handoff_remains_foreground() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state_root = directory.path().join("state");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&state_root).unwrap();
    let output_path = root.join("finished.txt");
    let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root).unwrap());
    let background = crate::BackgroundRegistry::default();
    background.set_foreground_handoff_budget(Some(Duration::from_millis(10)));
    background.set_job_lifecycle(std::sync::Arc::new(RejectingLifecycle {
        live_writer_supported: false,
    }));
    let read_cache = std::sync::Mutex::new(crate::ReadCache::new());
    let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
    let runner = crate::ProcessRunner::new(&root).unwrap();
    let command = format!("sleep 0.08; printf finished > {}", output_path.display());

    let outcome = tokio::time::timeout(
        Duration::from_secs(2),
        run_bash_tool_with_auto_background(
            &root,
            &state_root,
            RuntimeResources {
                process_runner: Some(&runner),
                lsp: &lsp,
                background: &background,
                read_cache: &read_cache,
                repo_map: &repo_map,
                repo_map_arc: None,
                mcp: None,
                memory: None,
                skill: None,
                hunk_tracker: None,
            },
            BashArgs {
                command,
                timeout: None,
                run_in_background: false,
            },
            &mut |_| {},
            true,
        ),
    )
    .await
    .expect("unsupported handoff must not impose the attachment budget")
    .unwrap();

    assert_eq!(outcome.status, ToolStatus::Succeeded);
    assert!(outcome.background.is_none());
    assert!(background.ids().is_empty());
    assert_eq!(std::fs::read_to_string(output_path).unwrap(), "finished");
}
