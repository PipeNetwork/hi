use super::super::{finish_ripgrep_execution, ripgrep_binary_unavailable};

#[test]
fn pipe_wrap_missing_rg_is_distinct_from_other_launch_failures() {
    let missing = "pipe-wrap: child setup failed: exec rg: ENOENT: No such file or directory";
    for (status, code, stdout, stderr, unavailable) in [
        (crate::ToolStatus::Failed, 1, "", missing, true),
        (
            crate::ToolStatus::Failed,
            1,
            "",
            "pipe-wrap: child setup failed: exec rg: EACCES: Permission denied",
            false,
        ),
        (
            crate::ToolStatus::Failed,
            1,
            "",
            "pipe-wrap: child setup failed: mount: ENOENT: No such file or directory",
            false,
        ),
        (
            crate::ToolStatus::Failed,
            1,
            "",
            "pipe-wrap: child setup failed: exec sh: ENOENT: No such file or directory",
            false,
        ),
        (crate::ToolStatus::Failed, 2, "", missing, false),
        (crate::ToolStatus::TimedOut, 1, "", missing, false),
        (
            crate::ToolStatus::Failed,
            1,
            "partial search output",
            missing,
            false,
        ),
    ] {
        let execution = crate::ProcessExecution {
            status,
            outcome: crate::ProcessOutcome {
                exit_code: Some(code),
                stdout_summary: stdout.into(),
                stderr_summary: stderr.into(),
                duration_ms: 1,
            },
            truncation: crate::TruncationState::Complete,
        };
        assert_eq!(ripgrep_binary_unavailable(&execution), unavailable);
        let result = finish_ripgrep_execution(execution, ".", "needle");
        if unavailable {
            assert!(result.unwrap().is_none());
        } else {
            assert!(
                result.is_err(),
                "unexpected fallback or no-match: {result:?}"
            );
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
fn missing_ripgrep_uses_fallback_with_the_selected_sandbox() {
    const CHILD: &str = "HI_GREP_MISSING_BINARY_TEST_CHILD";
    const ENFORCED: &str = "HI_GREP_MISSING_BINARY_EXPECT_ENFORCED";
    const TEST: &str =
        "read::tests::grep_process::missing_ripgrep_uses_fallback_with_the_selected_sandbox";
    let root = tempfile::tempdir().unwrap();
    let runner = crate::ProcessRunner::new_with_policy(
        root.path(),
        crate::sandbox::SandboxPolicy::Workspace,
    )
    .unwrap();
    if std::env::var_os(CHILD).is_none() {
        if std::env::var_os("HI_PIPE_WRAP").is_some() {
            assert!(runner.sandbox_enforced(), "configured helper must enforce");
        }
        // The parent resolves executables before wrapping them. Use a fresh
        // process with only the helper's `true` probe on PATH, so both lookups
        // observe rg as absent without mutating other tests' environment.
        let restricted_path = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink("/bin/true", restricted_path.path().join("true")).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST, "--nocapture"])
            .env(CHILD, "1")
            .env(ENFORCED, if runner.sandbox_enforced() { "1" } else { "0" })
            .env("PATH", restricted_path.path())
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stdout}\n{stderr}");
        assert!(
            stdout.contains("1 passed"),
            "test child did not run: {stdout}"
        );
        return;
    }

    let expect_enforced = std::env::var(ENFORCED).unwrap() == "1";
    assert_eq!(runner.sandbox_enforced(), expect_enforced);
    std::fs::write(root.path().join("source.txt"), "source_needle\n").unwrap();
    std::fs::create_dir(root.path().join(".cargo-home")).unwrap();
    std::fs::write(
        root.path().join(".cargo-home/vendor.txt"),
        "source_needle\n",
    )
    .unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        if expect_enforced {
            let execution = runner
                .run_program_plain_maybe_timeout("rg", ["--version"], None)
                .await
                .unwrap();
            assert_eq!(execution.status, crate::ToolStatus::Failed);
            assert!(
                ripgrep_binary_unavailable(&execution),
                "unexpected helper response: {execution:?}"
            );
        }
        let output = super::super::run_grep_with_runner(
            root.path(),
            Some(&runner),
            r#"{"pattern":"source_needle"}"#,
        )
        .await
        .unwrap();
        assert!(output.content.contains("source.txt:1: source_needle"));
        assert!(!output.content.contains("vendor.txt"));
    });
}

#[test]
fn sandboxed_missing_rg_is_treated_as_unavailable() {
    let execution = crate::ProcessExecution {
        status: crate::ToolStatus::Failed,
        outcome: crate::ProcessOutcome {
            exit_code: Some(71),
            stdout_summary: String::new(),
            stderr_summary: "sandbox-exec: execvp() of 'rg' failed: No such file or directory"
                .into(),
            duration_ms: 1,
        },
        truncation: crate::TruncationState::Complete,
    };
    assert!(ripgrep_binary_unavailable(&execution));
    assert!(
        finish_ripgrep_execution(execution, ".", "needle")
            .unwrap()
            .is_none()
    );
}

#[cfg(unix)]
#[tokio::test]
async fn grep_retains_failed_process_diagnostics_instead_of_reporting_no_matches() {
    let root = tempfile::tempdir().unwrap();
    let runner = crate::ProcessRunner::new(root.path()).unwrap();
    for (script, expected) in [
        ("exit 1", Ok("no matches for needle")),
        (
            "printf 'search launcher failed before execution' >&2; exit 1",
            Err("search launcher failed before execution"),
        ),
        ("printf 'source_symbol'", Ok("source_symbol")),
    ] {
        let execution = runner
            .run_program_plain_maybe_timeout("sh", &["-c", script], None)
            .await
            .unwrap();
        let result = finish_ripgrep_execution(execution, ".", "needle");
        match expected {
            Ok(content) => assert_eq!(result.unwrap().unwrap().content, content),
            Err(diagnostic) => {
                let error = result.unwrap_err().to_string();
                assert!(error.contains(diagnostic), "{error}");
                assert!(!error.contains("no matches"), "{error}");
            }
        }
    }
}
