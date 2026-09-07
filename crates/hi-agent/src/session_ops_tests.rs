use super::*;
use hi_ai::{Message, Role};

#[test]
fn pipefs_write_forms_are_identified_before_sync_helpers_run() {
    for command in [
        Command::Fork(String::new()),
        Command::Remember("note".into()),
        Command::UndoMemory,
        Command::Marketplace("install /tmp/skill.md".into()),
        Command::Worktree("gc".into()),
        Command::Inspect("bundle".into()),
        Command::Cd("nested".into()),
        Command::Trust("on".into()),
    ] {
        assert!(session_command_mutates_workspace(&command), "{command:?}");
    }
    for command in [
        Command::Marketplace("status".into()),
        Command::Worktree("list".into()),
        Command::Inspect("json".into()),
        Command::Cd(String::new()),
        Command::Trust("status".into()),
    ] {
        assert!(!session_command_mutates_workspace(&command), "{command:?}");
    }
    assert!(matches!(
        session_command_replay_class(&Command::Trust("on".into())),
        hi_workspace::ReplayClass::IdempotentExternal { .. }
    ));
    assert!(matches!(
        session_command_replay_class(&Command::Remember("note".into())),
        hi_workspace::ReplayClass::NonReplayableExternal
    ));
}

fn u(t: &str) -> Message {
    Message::user(t)
}
fn a(t: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![hi_ai::Content::Text(t.into())],
    }
}

#[test]
fn mode_prompts_are_delimited_and_current_turn_authoritative() {
    let plan = plan_mode_prompt("build profiles");
    assert!(plan.starts_with(crate::transcript::TURN_CONTROL_START));
    assert!(plan.contains("Plan mode is ON for this turn"));
    assert!(plan.contains("User request:\nbuild profiles"));
    assert_eq!(plan_mode_prompt(&plan), plan, "plan wrapping is idempotent");

    let normal = normal_mode_prompt("build all of that");
    assert!(normal.starts_with(crate::transcript::TURN_CONTROL_START));
    assert!(normal.contains("Plan mode is OFF for this turn"));
    assert!(normal.contains("Earlier plan-mode controls are historical"));
    assert!(normal.ends_with("build all of that"));
}

#[test]
fn lists_and_rewinds_user_turns() {
    let msgs = vec![
        Message::system("sys"),
        u("one"),
        a("ok1"),
        u("two"),
        a("ok2"),
    ];
    let turns = list_user_turns(&msgs);
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0].n, 1);
    assert_eq!(rewind_len_before_user_turn(&msgs, 2).unwrap(), 3);
}

#[test]
fn parse_fork_and_remember_flags() {
    assert_eq!(
        parse_fork_args("--no-worktree try rustc"),
        (false, "try rustc".into())
    );
    assert_eq!(
        parse_remember_args("--global prefer pnpm"),
        (true, "prefer pnpm".into())
    );
}

#[test]
fn tasks_report_names_background_task_ids() {
    let report = format_tasks_report(
        &["pid-9".into()],
        &["task_1".into(), "task_2".into()],
        0,
        false,
        PermissionMode::Ask,
        &[],
    );
    assert!(report.contains("task_1"));
    assert!(report.contains("task_2"));
    assert!(report.contains("background tasks:"));
}

#[test]
fn search_finds_assistant_text() {
    let msgs = vec![u("hi"), a("unique-token-xyz")];
    let r = search_messages(&msgs, "unique-token");
    assert!(r.contains("unique-token-xyz"));
}

#[test]
fn agents_show_clips_a_huge_persona() {
    let root = std::env::temp_dir().join(format!(
        "hi-agents-show-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join(".hi/agents")).unwrap();
    std::fs::write(
        root.join(".hi/agents/huge.md"),
        "P".repeat(crate::learning::MAX_LEDGER_READ_BYTES + 8_000),
    )
    .unwrap();
    let out = agents_report(&root, "show huge");
    assert!(
        out.len() <= crate::learning::MAX_LEDGER_READ_BYTES,
        "persona dump must be prefix-capped: {}",
        out.len()
    );
    assert!(out.starts_with('P'), "{out}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn claude_mcp_inspect_lists_servers_and_skips_a_huge_file() {
    let dir = std::env::temp_dir().join(format!(
        "hi-claude-json-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let small = dir.join("small.json");
    std::fs::write(
        &small,
        r#"{"mcpServers":{"sqlite":{},"github":{},"extra":{}}}"#,
    )
    .unwrap();
    let listed = inspect_claude_mcp_servers(&small);
    assert!(listed.contains("sqlite"), "{listed}");
    assert!(listed.contains("github"), "{listed}");

    let huge = dir.join("huge.json");
    std::fs::write(
        &huge,
        format!(
            "{{\"mcpServers\":{{}},\"pad\":\"{}\"}}",
            "x".repeat(crate::learning::MAX_LEDGER_READ_BYTES + 8_000)
        ),
    )
    .unwrap();
    let skipped = inspect_claude_mcp_servers(&huge);
    assert!(
        skipped.len() < 4_096,
        "huge claude.json must not be dumped: {}",
        skipped.len()
    );
    assert!(
        skipped.contains("too large") || skipped.contains("mcpServers"),
        "{skipped}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn hook_command_lists_missing_directory() {
    let root = std::env::temp_dir().join(format!("hi-hook-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let report = hooks_command(&root, "list");
    assert!(report.contains("hooks"));
    assert!(report.contains("no "));
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn untrusted_workspace_blocks_hook_execution() {
    let root = std::env::temp_dir().join(format!(
        "hi-untrusted-hook-test-{}-{}",
        std::process::id(),
        chrono_like_stamp()
    ));
    std::fs::create_dir_all(root.join(".hi/hooks")).unwrap();
    std::fs::write(
        root.join(".hi/hooks/pre-turn"),
        "#!/bin/sh\necho should-not-run\n",
    )
    .unwrap();
    let error = run_hook(&root, "pre-turn", "x")
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("untrusted"), "{error}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn hook_timeout_is_opt_in_and_zero_means_unlimited() {
    assert_eq!(hook_timeout_from_value(None), None);
    assert_eq!(hook_timeout_from_value(Some("")), None);
    assert_eq!(hook_timeout_from_value(Some("0")), None);
    assert_eq!(hook_timeout_from_value(Some("invalid")), None);
    assert_eq!(
        hook_timeout_from_value(Some("17")),
        Some(std::time::Duration::from_secs(17))
    );
}

#[test]
fn closed_hook_stdin_is_benign_but_other_write_errors_are_not() {
    assert!(
        tolerate_closed_hook_stdin(Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe)))
            .is_ok()
    );
    let error = tolerate_closed_hook_stdin(Err(std::io::Error::from(
        std::io::ErrorKind::PermissionDenied,
    )))
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
}

#[cfg(unix)]
fn write_test_hook(root: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt as _;

    let hooks = root.join(".hi/hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    let path = hooks.join("pre-turn");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(unix)]
fn process_is_alive(pid: libc::pid_t) -> bool {
    // SAFETY: signal 0 only probes a PID created by the test and does not
    // alter process state.
    unsafe { libc::kill(pid, 0) == 0 }
}

#[cfg(unix)]
async fn assert_process_exits(pid: libc::pid_t) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    while process_is_alive(pid) && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(!process_is_alive(pid), "hook descendant {pid} leaked");
}

#[cfg(unix)]
#[tokio::test]
async fn completed_hook_reaps_daemonized_descendants_without_waiting_for_their_pipes() {
    let root = tempfile::tempdir().unwrap();
    write_test_hook(
        root.path(),
        "sleep 30 &\necho $! > child.pid\nprintf '{\"version\":1,\"decision\":\"allow\",\"message\":\"done\"}\\n'",
    );
    // Keep the watchdog strictly outside the implementation's bounded
    // pipe-drain grace. Equal deadlines made this test flaky under load:
    // the outer timer could win before cleanup returned (or before its
    // specific error surfaced), even though the process group was killed.
    let watchdog = HOOK_PIPE_DRAIN_GRACE + std::time::Duration::from_secs(3);
    let report = tokio::time::timeout(
        watchdog,
        run_hook_process(root.path(), "pre-turn", "input", None),
    )
    .await
    .expect("daemonized hook must not wedge output drain")
    .unwrap();
    assert!(report.contains("done"), "{report}");

    let pid: libc::pid_t = std::fs::read_to_string(root.path().join("child.pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_process_exits(pid).await;
}

#[cfg(unix)]
#[tokio::test]
async fn explicit_hook_timeout_reaps_the_complete_process_group() {
    let root = tempfile::tempdir().unwrap();
    // Leave enough startup headroom for a heavily loaded test host. The
    // timeout begins after `spawn`, but the OS may not schedule the shell
    // quickly enough to create `child.pid` under concurrent PTY campaigns.
    // Keep the descendant far beyond the deadline so the assertion still
    // proves timeout-driven process-group cleanup rather than natural exit.
    write_test_hook(root.path(), "sleep 30 &\necho $! > child.pid\nwait");
    let error = run_hook_process(
        root.path(),
        "pre-turn",
        "input",
        Some(std::time::Duration::from_secs(5)),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(error.contains("timed out"), "{error}");

    let pid: libc::pid_t = std::fs::read_to_string(root.path().join("child.pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_process_exits(pid).await;
}

#[cfg(unix)]
#[tokio::test]
async fn hook_output_is_drained_concurrently_and_retained_with_a_bound() {
    let root = tempfile::tempdir().unwrap();
    write_test_hook(
        root.path(),
        "i=0\nwhile [ $i -lt 30000 ]; do\n  printf 'stdout-abcdefghijklmnopqrstuvwxyz-0123456789\\n'\n  printf 'stderr-abcdefghijklmnopqrstuvwxyz-0123456789\\n' >&2\n  i=$((i + 1))\ndone",
    );
    let report = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        run_hook_process(root.path(), "pre-turn", "input", None),
    )
    .await
    .expect("noisy hook must not block on full pipes")
    .unwrap();
    assert!(
        report.len() <= MAX_HOOK_OUTPUT_BYTES + 128,
        "{}",
        report.len()
    );
    assert!(report.contains("hook output truncated"));
}

#[test]
fn marketplace_installs_skill_file() {
    let root = std::env::temp_dir().join(format!("hi-market-test-{}", std::process::id()));
    let source_dir = root.join("source-pack");
    std::fs::create_dir_all(&source_dir).unwrap();
    std::fs::write(source_dir.join("SKILL.md"), "---\nname: test\n---\n").unwrap();
    let report = marketplace_report(&root, &format!("install {}", source_dir.display()));
    assert!(report.contains("installed"), "{report}");
    assert!(root.join(".hi/skills/source-pack/SKILL.md").is_file());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn mcp_admin_has_doctor_route() {
    assert!(mcp_admin_report("doctor").contains("/doctor"));
}

#[test]
fn remember_note_stamps_bullet_id_and_mentions_undo() {
    let root = std::env::temp_dir().join(format!(
        "hi-remember-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join(".hi")).unwrap();
    let msg = remember_note(&root, "prefer unique-remember-token", false).unwrap();
    assert!(msg.contains("[#"), "{msg}");
    assert!(msg.contains("/undo-memory"), "{msg}");
    let body = crate::memory::read_layer(&root.join(".hi/memory.md"));
    assert!(body.contains("[#"), "{body}");
    assert!(body.contains("prefer unique-remember-token"), "{body}");
    let _ = std::fs::remove_dir_all(root);
}
