use super::*;

#[tokio::test]
async fn explicit_root_and_structured_failure() {
    let root = std::env::temp_dir().join(format!("hi-process-root-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("marker"), "ok").unwrap();
    let runner = ProcessRunner::new(&root).unwrap();
    let run = runner
        .run_shell(
            "pwd; cat marker; printf problem >&2; exit 7",
            Duration::from_secs(5),
        )
        .await
        .unwrap();
    assert_eq!(run.status, ToolStatus::Failed);
    assert_eq!(run.outcome.exit_code, Some(7));
    assert!(
        run.outcome
            .stdout_summary
            .contains(root.to_string_lossy().as_ref())
    );
    assert!(run.outcome.stdout_summary.contains("ok"));
    assert!(run.outcome.stderr_summary.contains("problem"));
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn timeout_is_typed() {
    let runner = ProcessRunner::from_current_dir().unwrap();
    let run = runner
        .run_shell("sleep 60", Duration::from_millis(50))
        .await
        .unwrap();
    assert_eq!(run.status, ToolStatus::TimedOut);
    assert_eq!(run.outcome.exit_code, None);
}

#[tokio::test]
async fn timeout_retains_unterminated_output_from_both_streams() {
    let root = tempfile::tempdir().unwrap();
    let runner =
        ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off).unwrap();
    let mut streamed = String::new();
    let run = runner
        .run_shell_streaming(
            "printf pending-stdout; printf pending-stderr >&2; exec sleep 600",
            Duration::from_millis(400),
            &mut |text| streamed.push_str(text),
        )
        .await
        .unwrap();

    assert_eq!(run.status, ToolStatus::TimedOut);
    assert_eq!(run.outcome.stdout_summary, "pending-stdout");
    assert_eq!(run.outcome.stderr_summary, "pending-stderr");
    assert!(streamed.contains("pending-stdout"));
    assert!(streamed.contains("pending-stderr"));
}

#[tokio::test]
async fn direct_program_deadline_is_optional_and_explicit() {
    let runner = ProcessRunner::from_current_dir().unwrap();
    let completed = tokio::time::timeout(
        Duration::from_secs(2),
        runner.run_program_maybe_timeout("sh", ["-c", "sleep 0.03; printf completed"], None),
    )
    .await
    .expect("the unbounded direct program should complete normally")
    .unwrap();
    assert_eq!(completed.status, ToolStatus::Succeeded);
    assert_eq!(completed.outcome.stdout_summary, "completed");

    let timed_out = runner
        .run_program_maybe_timeout("sh", ["-c", "sleep 1"], Some(Duration::from_millis(25)))
        .await
        .unwrap();
    assert_eq!(timed_out.status, ToolStatus::TimedOut);
    assert_eq!(timed_out.outcome.exit_code, None);
}

#[cfg(unix)]
#[tokio::test]
async fn cancelling_an_unbounded_process_future_kills_its_group() {
    let root = std::env::temp_dir().join(format!(
        "hi-process-unbounded-cancel-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let marker = root.join("leaked");
    let runner = ProcessRunner::new(&root).unwrap();
    let command = format!("sleep 0.15; touch {}", marker.display());

    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            runner.run_shell_maybe_timeout(&command, None)
        )
        .await
        .is_err(),
        "the test must cancel the still-running unbounded process future"
    );
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert!(
        !marker.exists(),
        "dropping an unbounded verifier future must kill its process group"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(unix)]
#[tokio::test]
async fn cancelling_an_unbounded_direct_program_kills_its_group() {
    let root = std::env::temp_dir().join(format!(
        "hi-process-unbounded-direct-cancel-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let marker = root.join("leaked");
    let runner = ProcessRunner::new(&root).unwrap();
    let command = format!("sleep 0.15; touch {}", marker.display());

    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            runner.run_program_maybe_timeout("sh", ["-c", command.as_str()], None)
        )
        .await
        .is_err(),
        "the test must cancel the still-running unbounded direct program"
    );
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert!(
        !marker.exists(),
        "dropping the direct-program future must kill its process group"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn direct_program_treats_filename_as_one_argument() {
    let root = std::env::temp_dir().join(format!(
        "hi-process-argv-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let name = "input; touch INJECTED.txt";
    std::fs::write(root.join(name), "safe\n").unwrap();
    let runner = ProcessRunner::new(&root).unwrap();
    let run = runner
        .run_program("cat", [name], Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(run.status, ToolStatus::Succeeded);
    assert_eq!(run.outcome.stdout_summary, "safe");
    assert!(!root.join("INJECTED.txt").exists());
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn explicit_environment_is_added_after_sanitization() {
    let runner = ProcessRunner::from_current_dir().unwrap();
    let run = runner
        .run_program_with_env(
            "sh",
            ["-c", "printf %s \"$HI_API_KEY\""],
            [("HI_API_KEY", "child-only-key")],
            Duration::from_secs(5),
        )
        .await
        .unwrap();
    assert_eq!(run.status, ToolStatus::Succeeded);
    assert_eq!(run.outcome.stdout_summary, "child-only-key");
}

#[test]
fn sandboxed_cargo_uses_workspace_local_home() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let default = workspace_cargo_home(&root, crate::sandbox::SandboxPolicy::Workspace)
        .expect("workspace mode needs an isolated Cargo cache");

    assert_eq!(default, root.join(".hi/state/cargo-home"));

    std::fs::create_dir_all(root.join(".cargo-home")).unwrap();
    assert_eq!(
        workspace_cargo_home(&root, crate::sandbox::SandboxPolicy::Workspace),
        Some(root.join(".cargo-home")),
        "retain compatibility with existing project-local Cargo caches"
    );
    assert_eq!(
        workspace_cargo_home(&root, crate::sandbox::SandboxPolicy::Off),
        None,
        "sandbox-off commands retain the user's normal Cargo environment"
    );
}

#[tokio::test]
async fn process_children_receive_the_isolated_cargo_home() {
    let temp = tempfile::tempdir().unwrap();
    let runner = ProcessRunner::new(temp.path()).unwrap();
    let Some(expected) = runner.cargo_home.clone() else {
        // A parent hi sandbox deliberately resolves nested runners to Off;
        // the ancestor already owns environment confinement in that case.
        return;
    };

    let run = runner
        .run_program(
            "sh",
            ["-c", "printf %s \"$CARGO_HOME\""],
            Duration::from_secs(5),
        )
        .await
        .unwrap();

    assert_eq!(run.status, ToolStatus::Succeeded);
    assert_eq!(run.outcome.stdout_summary, expected.to_string_lossy());
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn timeout_kills_process_group_descendants() {
    let runner = ProcessRunner::from_current_dir().unwrap();
    let run = runner
        .run_shell(
            "sleep 60 & child=$!; printf '%s\\n' \"$child\"; wait",
            Duration::from_millis(100),
        )
        .await
        .unwrap();
    assert_eq!(run.status, ToolStatus::TimedOut);
    let pid = run.outcome.stdout_summary.trim().parse::<u32>().unwrap();
    let proc_stat = format!("/proc/{pid}/stat");
    for _ in 0..100 {
        let gone_or_zombie = match std::fs::read_to_string(&proc_stat) {
            Ok(stat) => {
                stat.rsplit_once(") ")
                    .and_then(|(_, rest)| rest.chars().next())
                    == Some('Z')
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
            Err(_) => false,
        };
        if gone_or_zombie {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed-out descendant {pid} remained alive");
}

#[tokio::test]
async fn pagers_are_neutralized_for_child_commands() {
    let runner = ProcessRunner::from_current_dir().unwrap();
    let mut sink = |_: &str| {};
    // The child sees PAGER=cat and a blanked AWS_PAGER — paging tools
    // stream instead of blocking.
    let exec = runner
            .run_shell_streaming(
                "printf 'PAGER=%s GIT_PAGER=%s AWS_PAGER=[%s]' \"$PAGER\" \"$GIT_PAGER\" \"$AWS_PAGER\"",
                Duration::from_secs(10),
                &mut sink,
            )
            .await
            .unwrap();
    let out = exec.model_content();
    assert!(out.contains("PAGER=cat"), "PAGER neutralized: {out}");
    assert!(
        out.contains("GIT_PAGER=cat"),
        "GIT_PAGER neutralized: {out}"
    );
    assert!(out.contains("AWS_PAGER=[]"), "AWS_PAGER blanked: {out}");
}

#[tokio::test]
async fn cargo_diagnostics_keep_color_when_output_is_piped() {
    let runner = ProcessRunner::from_current_dir().unwrap();
    let mut sink = |_: &str| {};
    let exec = runner
        .run_shell_streaming(
            "printf %s \"$CARGO_TERM_COLOR\"",
            Duration::from_secs(5),
            &mut sink,
        )
        .await
        .unwrap();
    assert_eq!(exec.model_content(), "always");
}

#[tokio::test]
async fn model_content_strips_ansi_but_display_content_preserves_it() {
    let runner = ProcessRunner::from_current_dir().unwrap();
    let exec = runner
        .run_shell("printf '\\033[31mred\\033[0m'", Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(exec.model_content(), "red");
    assert_eq!(exec.display_content(), "\u{1b}[31mred\u{1b}[0m");
    assert_eq!(exec.model_outcome().stdout_summary, "red");
}

#[cfg(unix)]
#[tokio::test]
async fn newline_free_output_stays_bounded() {
    let runner = ProcessRunner::from_current_dir().unwrap();
    // Four megabytes without a newline used to make read_until allocate
    // the whole record before BoundedBuffer could clip it.
    //
    // Generate the record with `yes | tr -d | head -c` rather than
    // `dd bs=1m`: the lowercase `1m` suffix is not portable (some CI
    // runners' `dd` rejects it), and `/dev/zero` may be unreachable under
    // a confined sandbox. Both failure modes write nothing while the
    // pipeline still exits 0, which previously made the run report
    // `Complete` instead of `Truncated` on Linux CI.
    let run = runner
        .run_shell(
            "yes x | tr -d '\\n' | head -c 4194304",
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    assert_eq!(run.status, ToolStatus::Succeeded);
    assert!(
        matches!(run.truncation, TruncationState::Truncated { .. }),
        "expected truncation; got {:?} (stdout {} bytes)",
        run.truncation,
        run.outcome.stdout_summary.len()
    );
    // The human-readable truncation marker can make the returned string a
    // little larger than the nominal character budget; it must still be
    // tiny compared with the four-megabyte source record.
    assert!(run.outcome.stdout_summary.chars().count() < 10_000);
}

#[cfg(unix)]
#[tokio::test]
async fn secrets_split_across_stream_chunks_are_redacted() {
    let runner = ProcessRunner::from_current_dir().unwrap();
    let run = runner
            .run_shell(
                "printf '%*s' 65533 '' | tr ' ' x; printf 'OPENAI_API_KEY=sk-example-secret-value-123456789'",
                Duration::from_secs(10),
            )
            .await
            .unwrap();
    assert!(
        !run.model_content()
            .contains("sk-example-secret-value-123456789")
    );
    assert!(run.model_content().contains("[REDACTED_SECRET]"));
}

#[cfg(unix)]
#[tokio::test]
async fn entropy_gated_secret_split_across_stream_chunks_is_redacted() {
    // The entropy gate needs the credential key name AND the value in one
    // redaction window. When a long secret straddles the 64 KiB
    // pseudo-line boundary, the streaming per-chunk redact sees only a
    // fragment, which the value-length floor rejects. The final
    // re-redaction over the reassembled buffer must still catch the whole
    // assignment. The key sits at a line boundary, matching real logs.
    let runner = ProcessRunner::from_current_dir().unwrap();
    // 65490-char line + newline, then `token=` (6) + a 60-char value pushes
    // the value across the 65536 flush boundary mid-token.
    let value = "dGhpc2lzYXJhbmRvbWJhc2U2NHNlY3JldGRHaHBjMmx6WVhKaGJtUnZiVQ==";
    let cmd = format!("printf '%*s\\n' 65490 ''; printf 'token={value}'");
    let run = runner
        .run_shell(&cmd, Duration::from_secs(10))
        .await
        .unwrap();
    let content = run.model_content();
    assert!(
        !content.contains(value),
        "entropy-gated secret leaked across chunk split: ...{}",
        &content[content.len().saturating_sub(90)..]
    );
    assert!(
        content.contains("[REDACTED_SECRET]"),
        "expected redaction marker in reassembled output: ...{}",
        &content[content.len().saturating_sub(90)..]
    );
}

#[test]
fn sensitive_environment_names_are_removed_conservatively() {
    assert!(sensitive_environment_name(OsStr::new("GITHUB_TOKEN")));
    assert!(sensitive_environment_name(OsStr::new(
        "AWS_SECRET_ACCESS_KEY"
    )));
    assert!(sensitive_environment_name(OsStr::new("DATABASE_PASSWORD")));
    assert!(!sensitive_environment_name(OsStr::new("PATH")));
    assert!(!sensitive_environment_name(OsStr::new("RUSTUP_HOME")));
}

#[tokio::test]
async fn adoptable_completes_within_budget_like_normal() {
    let runner = ProcessRunner::from_current_dir().unwrap();
    let mut sink = |_: &str| {};
    let outcome = runner
        .run_shell_adoptable("echo adopt-hello", Duration::from_secs(10), &mut sink)
        .await
        .expect("ok");
    match outcome {
        AdoptableOutcome::Completed(exec) => {
            assert_eq!(exec.status, ToolStatus::Succeeded);
            assert!(
                exec.model_content().contains("adopt-hello"),
                "got: {}",
                exec.model_content()
            );
        }
        AdoptableOutcome::StillRunning(_) => panic!("fast command must complete in budget"),
    }
}

// Multi-thread flavor so the foreground-budget timer fires independently of
// the blocking child under CI load (see the bash-tool auto-background test).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn adoptable_hands_off_a_running_child_with_partial_output() {
    let runner = ProcessRunner::from_current_dir().unwrap();
    let mut sink = |_: &str| {};
    let outcome = runner
        .run_shell_adoptable(
            "printf seedline; printf diagnostic >&2; sleep 600",
            Duration::from_millis(400),
            &mut sink,
        )
        .await
        .expect("ok");
    match outcome {
        AdoptableOutcome::StillRunning(mut running) => {
            assert!(
                running.partial_output.contains("seedline"),
                "seed carries foreground output: {:?}",
                running.partial_output
            );
            assert!(running.pgid.is_some(), "pgid captured for tree-kill");
            // The handed-off child is still alive; clean it up (the guard
            // was defused, so nothing killed it for us).
            kill_process_group(&running.child);
            let _ = running.child.kill().await;
            assert!(
                running.partial_output.contains("diagnostic"),
                "unterminated stderr survives adoption: {:?}",
                running.partial_output
            );
        }
        AdoptableOutcome::Completed(_) => panic!("a 600s sleep must outlast a 400ms budget"),
    }
}

#[tokio::test]
async fn exit_is_reported_even_when_a_descendant_holds_the_pipes() {
    // `cmd &` inside the shell: the shell exits instantly but the
    // detached sleep inherits stdout, so pipe-EOF never arrives on its
    // own. The reap must not wait for EOF — this used to report a
    // full-budget timeout and discard the real exit status.
    let root = tempfile::tempdir().unwrap();
    let runner =
        ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off).unwrap();
    let started = Instant::now();
    let exec = runner
        .run_shell("printf done; sleep 30 &", Duration::from_millis(400))
        .await
        .expect("ok");
    assert_eq!(exec.status, ToolStatus::Succeeded);
    assert_eq!(exec.outcome.exit_code, Some(0));
    assert!(
        exec.outcome.stdout_summary.contains("done"),
        "foreground output captured: {:?}",
        exec.outcome.stdout_summary
    );
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "must not burn the budget waiting for the descendant: {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn adoptable_does_not_adopt_exited_children_with_inherited_pipes() {
    let root = tempfile::tempdir().unwrap();
    let runner =
        ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off).unwrap();
    for exit_code in [0, 7] {
        let command = format!("printf done; sleep 30 & exit {exit_code}");
        let outcome = runner
            .run_shell_adoptable(&command, Duration::from_millis(400), &mut |_| {})
            .await
            .unwrap();
        match outcome {
            AdoptableOutcome::Completed(execution) => {
                assert_eq!(execution.outcome.exit_code, Some(exit_code));
                assert_eq!(
                    execution.status,
                    if exit_code == 0 {
                        ToolStatus::Succeeded
                    } else {
                        ToolStatus::Failed
                    }
                );
                assert_eq!(execution.outcome.stdout_summary, "done");
            }
            AdoptableOutcome::StillRunning(mut running) => {
                if let Some(pgid) = running.pgid {
                    kill_group(pgid);
                }
                let _ = running.child.kill().await;
                panic!("an exited child must not be adopted as still running");
            }
        }
    }
}

fn diagnostic_failure_command() -> &'static str {
    "printf 'running 3 tests\\nerror: mismatch\\nFAILED tests::it_breaks\\n'; \
i=0; while [ \"$i\" -lt 200 ]; do printf 'ok noise\\n'; i=$((i+1)); done; \
printf 'UNIQUE_MIDDLE_LINE\\n'; \
i=0; while [ \"$i\" -lt 200 ]; do printf 'ok noise\\n'; i=$((i+1)); done; \
printf 'test result: FAILED\\n'; exit 1"
}

fn canned_mismatch_hook() -> crate::EvidenceReducerHook {
    std::sync::Arc::new(|source: &str, is_error: bool| {
        let quote = "error: mismatch";
        if !source.contains(quote) {
            return Err("missing-quote");
        }
        Ok(serde_json::json!({
            "schema": crate::REDUCER_RECEIPT_SCHEMA,
            "source_sha256": crate::sha256_hex(source.as_bytes()),
            "status": if is_error { "failure" } else { "success" },
            "uncertain": false,
            "evidence": [{"kind": "failure", "quote": quote}]
        })
        .to_string())
    })
}

#[tokio::test]
async fn diagnostic_shell_reduces_after_condense_with_canned_receipt() {
    let root = tempfile::tempdir().unwrap();
    let runner =
        ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off).unwrap();
    runner.set_evidence_reducer(
        crate::EvidenceReducerConfig {
            enabled: true,
            min_bytes: 16,
        },
        Some(canned_mismatch_hook()),
    );
    let run = runner
        .run_shell(diagnostic_failure_command(), Duration::from_secs(10))
        .await
        .unwrap();
    assert_eq!(run.status, ToolStatus::Failed);
    assert!(
        run.outcome
            .stdout_summary
            .starts_with(crate::EVIDENCE_RECEIPT_PREFIX),
        "{}",
        run.outcome.stdout_summary
    );
    assert!(
        run.outcome.stdout_summary.contains("error: mismatch"),
        "{}",
        run.outcome.stdout_summary
    );
    assert!(
        !run.outcome.stdout_summary.contains("UNIQUE_MIDDLE_LINE"),
        "condense must run first so omitted middle lines cannot appear: {}",
        run.outcome.stdout_summary
    );
}

#[tokio::test]
async fn diagnostic_shell_fail_opens_without_receipt_and_when_disabled() {
    let root = tempfile::tempdir().unwrap();
    let runner =
        ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off).unwrap();
    runner.set_evidence_reducer(
        crate::EvidenceReducerConfig {
            enabled: true,
            min_bytes: 16,
        },
        None,
    );
    let enabled_no_hook = runner
        .run_shell(diagnostic_failure_command(), Duration::from_secs(10))
        .await
        .unwrap();
    assert_eq!(enabled_no_hook.status, ToolStatus::Failed);
    assert!(
        enabled_no_hook
            .outcome
            .stdout_summary
            .contains("error: mismatch"),
        "{}",
        enabled_no_hook.outcome.stdout_summary
    );
    assert!(
        !enabled_no_hook
            .outcome
            .stdout_summary
            .contains(crate::EVIDENCE_RECEIPT_PREFIX)
    );

    let disabled =
        ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off).unwrap();
    let json = serde_json::json!({"schema": "unused"}).to_string();
    disabled.set_evidence_reducer(
        crate::EvidenceReducerConfig::default(),
        Some(std::sync::Arc::new(move |_, _| Ok(json.clone()))),
    );
    let run = disabled
        .run_shell(diagnostic_failure_command(), Duration::from_secs(10))
        .await
        .unwrap();
    assert!(run.outcome.stdout_summary.contains("error: mismatch"));
    assert!(
        !run.outcome
            .stdout_summary
            .contains(crate::EVIDENCE_RECEIPT_PREFIX)
    );
}

#[tokio::test]
async fn adoptable_completed_and_run_program_diagnostics_use_reducer() {
    let root = tempfile::tempdir().unwrap();
    let runner =
        ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off).unwrap();
    runner.set_evidence_reducer(
        crate::EvidenceReducerConfig {
            enabled: true,
            min_bytes: 16,
        },
        Some(canned_mismatch_hook()),
    );
    let adopted = runner
        .run_shell_adoptable(
            diagnostic_failure_command(),
            Duration::from_secs(10),
            &mut |_| {},
        )
        .await
        .unwrap();
    match adopted {
        AdoptableOutcome::Completed(run) => {
            assert!(
                run.outcome
                    .stdout_summary
                    .starts_with(crate::EVIDENCE_RECEIPT_PREFIX),
                "{}",
                run.outcome.stdout_summary
            );
        }
        AdoptableOutcome::StillRunning(_) => {
            panic!("diagnostic command should finish inside the foreground budget")
        }
    }

    let program = runner
        .run_program(
            "sh",
            ["-c", diagnostic_failure_command()],
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    assert!(
        program
            .outcome
            .stdout_summary
            .starts_with(crate::EVIDENCE_RECEIPT_PREFIX),
        "{}",
        program.outcome.stdout_summary
    );
}
