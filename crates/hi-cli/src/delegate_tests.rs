use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::candidate_gate::{
    independently_verify_candidate_cached, inspect_child_report, is_destination_verify_cancelled,
    is_verifier_cancelled, parse_name_status, staged_candidate_diff,
};
use crate::candidate_merge::{
    apply_candidate_and_reverify, apply_candidate_and_reverify_cancellable,
    apply_candidate_and_reverify_cancellable_at_base,
};

static TEST_ID: AtomicU64 = AtomicU64::new(0);

#[test]
fn delegate_child_budget_precedes_outer_kill_and_preserves_explicit_cap() {
    let uncapped = crate::delegate::delegate_child_budget_arguments(None, None, None)
        .into_iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert!(
        uncapped.is_empty(),
        "ordinary delegates must inherit no model, tool, or time ceiling"
    );

    let capped = crate::delegate::delegate_child_budget_arguments(Some(7), Some(13), Some(120))
        .into_iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(
        capped,
        [
            "--turn-deadline",
            "60",
            "--max-steps",
            "7",
            "--max-tool-calls",
            "13"
        ]
    );

    assert_eq!(
        crate::delegate::delegate_timeout_secs_from_value(Some("120")),
        Some(120),
        "ordinary configured delegate timeouts must remain exact"
    );
    assert_eq!(
        crate::delegate::delegate_timeout_secs_from_value(Some("0")),
        Some(900),
        "zero falls back to the mandatory managed timeout"
    );
    assert_eq!(
        crate::delegate::delegate_timeout_secs_from_value(None),
        Some(900)
    );
    assert_eq!(
        crate::delegate::delegate_timeout_secs_from_value(Some("invalid")),
        Some(900)
    );
    assert_eq!(
        crate::delegate::delegate_timeout_secs_from_value(Some("1")),
        Some(1)
    );

    let short = crate::delegate::delegate_child_budget_arguments(None, None, Some(1))
        .into_iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert!(short.is_empty());

    let managed_zero = crate::delegate::delegate_child_budget_arguments(None, Some(0), None)
        .into_iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(managed_zero, ["--max-tool-calls", "0"]);
}

#[test]
fn delegate_capacity_wait_is_finite_by_default_and_explicit_when_requested() {
    assert_eq!(
        crate::delegate::delegate_queue_timeout_secs_from_value(None),
        Some(300)
    );
    assert_eq!(
        crate::delegate::delegate_queue_timeout_secs_from_value(Some("0")),
        Some(300)
    );
    assert_eq!(
        crate::delegate::delegate_queue_timeout_secs_from_value(Some("invalid")),
        Some(300)
    );
    assert_eq!(
        crate::delegate::delegate_queue_timeout_secs_from_value(Some("3")),
        Some(3)
    );
    assert!(crate::delegate::queue_wait_timed_out(
        Duration::from_secs(300),
        Some(Duration::from_secs(300))
    ));
    assert!(crate::delegate::queue_wait_timed_out(
        Duration::from_millis(1),
        Some(Duration::from_millis(1))
    ));
}

#[test]
fn delegate_runner_runtime_step_limit_can_be_cleared_and_reinstalled() {
    let root = temp_path("dynamic-step-limit");
    let workspace = root.join("workspace");
    let state = root.join("state");
    std::fs::create_dir_all(&workspace).unwrap();
    let runner = crate::delegate::CliDelegateRunner::new(
        PathBuf::from("hi"),
        "openai".into(),
        "test-model".into(),
        "http://127.0.0.1:1/v1".into(),
        "test-key".into(),
        None,
        Some(7),
        Some(13),
        0,
        workspace,
        state,
    )
    .unwrap();
    assert_eq!(runner.configured_max_steps(), Some(7));
    assert_eq!(runner.configured_max_tool_calls(), Some(13));

    hi_agent::DelegateRunner::set_max_steps(&runner, None);
    assert_eq!(runner.configured_max_steps(), None);
    hi_agent::DelegateRunner::set_max_steps(&runner, Some(9));
    assert_eq!(runner.configured_max_steps(), Some(9));
    assert_eq!(
        runner.configured_max_tool_calls(),
        Some(13),
        "changing the model-round cap must preserve the tool-call cap"
    );

    hi_agent::DelegateRunner::set_max_tool_calls(&runner, None);
    assert_eq!(runner.configured_max_tool_calls(), None);
    hi_agent::DelegateRunner::set_max_tool_calls(&runner, Some(0));
    assert_eq!(
        runner.configured_max_tool_calls(),
        Some(0),
        "a managed zero-tool budget must not collide with unlimited"
    );

    let _ = std::fs::remove_dir_all(root);
}

fn report_json(verification: &str, review: &str) -> Value {
    serde_json::json!({
        "schema_version": 2,
        "outcome": {
            "status": "completed",
            "verification": verification,
            "review": review,
            "stop_reason": "completed",
            "changed_files": ["src/lib.rs"],
            "verified_workspace_revision": "sha256:abc",
            "effective_route": { "provider": "fake", "model": "test" }
        },
        "verification": {
            "status": verification,
            "stages": [{ "name": "verify_1", "command": "true" }]
        },
        "review": { "status": review },
        "route": { "provider": "fake", "model": "test" },
        "changes_complete": true,
        "changes": [{
            "path": "src/lib.rs",
            "kind": "modify",
            "before_digest": "sha256:before",
            "after_digest": "sha256:after",
            "before_len": 10,
            "after_len": 11,
            "before_mode": 420,
            "after_mode": 420
        }]
    })
}

fn temp_path(label: &str) -> PathBuf {
    let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "hi-delegate-test-{label}-{}-{id}",
        std::process::id()
    ))
}

#[test]
fn typed_child_gate_rejects_unverified_and_objected_outcomes() {
    let path = temp_path("report");
    std::fs::write(
        &path,
        serde_json::to_vec(&report_json("passed", "passed")).unwrap(),
    )
    .unwrap();
    assert!(inspect_child_report(&path).is_ok());

    std::fs::write(
        &path,
        serde_json::to_vec(&report_json("unverified", "passed")).unwrap(),
    )
    .unwrap();
    assert!(inspect_child_report(&path).is_err());

    std::fs::write(
        &path,
        serde_json::to_vec(&report_json("passed", "objected")).unwrap(),
    )
    .unwrap();
    assert!(inspect_child_report(&path).is_err());

    std::fs::write(
        &path,
        serde_json::to_vec(&report_json("passed", "unavailable")).unwrap(),
    )
    .unwrap();
    assert!(
        inspect_child_report(&path).is_ok(),
        "review infrastructure is fail-open only after deterministic verification"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn name_status_parser_is_nul_safe_and_rejects_traversal() {
    let parsed = parse_name_status(b"M\0src/a.rs\0A\0space name.txt\0").unwrap();
    assert_eq!(
        parsed,
        vec![PathBuf::from("src/a.rs"), PathBuf::from("space name.txt")]
    );
    assert!(parse_name_status(b"A\0../escape\0").is_err());
}

#[test]
fn immutable_base_keeps_candidate_commits_in_the_diff() {
    let (root, worktree) = candidate_fixture("committed-diff");
    let base = git_stdout(&root, &["rev-parse", "HEAD"]);
    std::fs::write(worktree.join("value.txt"), "committed candidate\n").unwrap();
    git_ok(&worktree, &["add", "value.txt"]);
    git_ok(&worktree, &["commit", "-qm", "candidate commit"]);

    assert!(
        staged_candidate_diff(&worktree, "HEAD")
            .unwrap()
            .paths
            .is_empty(),
        "a moving HEAD would hide a child-created commit"
    );
    assert_eq!(
        staged_candidate_diff(&worktree, &base)
            .unwrap()
            .display_paths,
        vec!["value.txt"]
    );

    hi_tools::worktree::cleanup(&root, &[worktree]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn destination_verification_failure_rolls_back_transaction() {
    let (root, worktree) = candidate_fixture("rollback");
    std::fs::write(worktree.join("value.txt"), "after\n").unwrap();

    let error = apply_candidate_and_reverify(
        &worktree,
        "HEAD",
        &root,
        &root.join(".hi-test-state"),
        "false",
    )
    .expect_err("failed destination verification must reject candidate");
    assert!(format!("{error:#}").contains("rolled back"));
    assert_eq!(
        std::fs::read_to_string(root.join("value.txt")).unwrap(),
        "before\n"
    );

    hi_tools::worktree::cleanup(&root, &[worktree]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn passing_destination_revision_is_applied_with_candidate_mode() {
    let (root, worktree) = candidate_fixture("success");
    std::fs::write(worktree.join("value.txt"), "after\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            worktree.join("value.txt"),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
    }

    let changed = apply_candidate_and_reverify(
        &worktree,
        "HEAD",
        &root,
        &root.join(".hi-test-state"),
        "grep -qx after value.txt",
    )
    .expect("passing destination revision is accepted");
    assert_eq!(changed.changes, vec!["value.txt"]);
    assert_eq!(
        std::fs::read_to_string(root.join("value.txt")).unwrap(),
        "after\n"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(root.join("value.txt"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
    }

    hi_tools::worktree::cleanup(&root, &[worktree]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn detached_candidate_rejects_a_changed_complete_source_base() {
    let container = temp_path("detached-stale-base");
    let source = container.join("source");
    let state = container.join("state");
    let candidate_owner = container.join("candidate");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::create_dir(&state).unwrap();
    git_ok(&source, &["init", "-q"]);
    git_ok(&source, &["config", "user.email", "test@example.invalid"]);
    git_ok(&source, &["config", "user.name", "Hi Test"]);
    std::fs::write(source.join("value.txt"), "before\n").unwrap();
    git_ok(&source, &["add", "value.txt"]);
    git_ok(&source, &["commit", "-qm", "base"]);
    let candidate = hi_tools::candidate_workspace::CandidateWorkspace::create(
        &source,
        &state,
        &candidate_owner,
    )
    .unwrap();
    std::fs::write(candidate.root().join("value.txt"), "candidate\n").unwrap();

    // Even a disjoint parent edit changes the complete base version and must
    // prevent safe auto-apply.
    std::fs::write(source.join("other.txt"), "concurrent\n").unwrap();
    let error = apply_candidate_and_reverify_cancellable_at_base(
        candidate.root(),
        candidate.baseline_commit(),
        &source,
        &state,
        "true",
        Some(candidate.source_snapshot_id()),
        None,
    )
    .expect_err("a stale complete base must not auto-apply");

    assert!(format!("{error:#}").contains("candidate base is stale"));
    assert_eq!(
        std::fs::read_to_string(source.join("value.txt")).unwrap(),
        "before\n"
    );
    drop(candidate);
    let _ = std::fs::remove_dir_all(container);
}

#[test]
fn scoped_workspace_diff_and_merge_paths_remain_relative() {
    let (root, worktree) = candidate_fixture("scoped-workspace");
    let destination = root.join("nested");
    let candidate = worktree.join("nested");
    std::fs::create_dir_all(&destination).unwrap();
    std::fs::create_dir_all(&candidate).unwrap();
    std::fs::write(candidate.join("created.txt"), "scoped\n").unwrap();

    let diff = staged_candidate_diff(&candidate, "HEAD").unwrap();
    assert_eq!(diff.display_paths, vec!["created.txt"]);
    assert!(
        String::from_utf8_lossy(&diff.patch).contains("b/created.txt"),
        "patch paths must be relative to the explicit workspace"
    );

    let changed = apply_candidate_and_reverify(
        &candidate,
        "HEAD",
        &destination,
        &root.join(".hi-test-state"),
        "test -f created.txt",
    )
    .expect("scoped candidate is applied inside the scoped destination");
    assert_eq!(changed.changes, vec!["created.txt"]);
    assert_eq!(
        std::fs::read_to_string(destination.join("created.txt")).unwrap(),
        "scoped\n"
    );

    hi_tools::worktree::cleanup(&root, &[worktree]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn verifier_mutation_is_unstable_and_preserved_for_review() {
    let (root, worktree) = candidate_fixture("unstable");
    std::fs::write(worktree.join("value.txt"), "after\n").unwrap();

    let error = apply_candidate_and_reverify(
        &worktree,
        "HEAD",
        &root,
        &root.join(".hi-test-state"),
        "printf 'verifier mutation\\n' > value.txt",
    )
    .expect_err("a verifier-mutated revision must not be accepted");
    assert!(format!("{error:#}").contains("unstable"));
    assert!(format!("{error:#}").contains("rollback was refused"));
    assert_eq!(
        std::fs::read_to_string(root.join("value.txt")).unwrap(),
        "verifier mutation\n"
    );

    hi_tools::worktree::cleanup(&root, &[worktree]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn candidate_rollback_preserves_concurrent_user_edit() {
    let (root, worktree) = candidate_fixture("concurrent-verifier-edit");
    std::fs::write(worktree.join("value.txt"), "candidate\n").unwrap();
    std::fs::create_dir_all(root.join(".hi")).unwrap();
    let writer_root = root.clone();
    let writer = std::thread::spawn(move || {
        let started = Instant::now();
        while !writer_root.join(".hi/verification-started").exists() {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "verifier did not start"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        std::fs::write(writer_root.join("value.txt"), "user's concurrent edit\n").unwrap();
        std::fs::write(writer_root.join(".hi/verification-continue"), "").unwrap();
    });
    let result = apply_candidate_and_reverify(
        &worktree,
        "HEAD",
        &root,
        &root.join(".hi-test-state"),
        "touch .hi/verification-started; n=0; while test ! -f .hi/verification-continue && test $n -lt 500; do n=$((n+1)); sleep 0.01; done; false",
    );
    writer.join().unwrap();
    let content = std::fs::read_to_string(root.join("value.txt")).unwrap();
    hi_tools::worktree::cleanup(&root, &[worktree]);
    let _ = std::fs::remove_dir_all(root);
    assert!(result.is_err());
    assert_eq!(content, "user's concurrent edit\n");
}

#[test]
fn independently_verify_candidate_cached_revalidates_corrupt_and_mismatched_records() {
    let (root, worktree) = candidate_fixture("verify-cache-invalid");
    std::fs::write(worktree.join("value.txt"), "after\n").unwrap();
    let cache = root.join(".hi-test-verify-cache");
    let marker = worktree.join(".hi/allow-verification");
    std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
    std::fs::write(&marker, "allowed").unwrap();
    let verify = "test -f .hi/allow-verification";
    independently_verify_candidate_cached(&worktree, "HEAD", verify, &cache, None).unwrap();
    let entry = std::fs::read_dir(&cache)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .unwrap();
    let valid: Value = serde_json::from_slice(&std::fs::read(&entry).unwrap()).unwrap();
    std::fs::remove_file(&marker).unwrap();

    let mut invalid_records = vec![b"{truncated".to_vec()];
    for field in ["schema_version", "key", "base", "verify", "fingerprint"] {
        let mut record = valid.clone();
        record[field] = Value::Null;
        invalid_records.push(serde_json::to_vec(&record).unwrap());
    }
    for record in invalid_records {
        std::fs::write(&entry, record).unwrap();
        let error = independently_verify_candidate_cached(&worktree, "HEAD", verify, &cache, None)
            .expect_err("an invalid cache entry must rerun the now-failing verifier");
        assert!(format!("{error:#}").contains("configured verifier"));
        assert!(!entry.with_extension("lock").exists());
    }

    // A rerun can repair the invalid cache, and a valid cache hit still works.
    std::fs::write(&marker, "allowed").unwrap();
    independently_verify_candidate_cached(&worktree, "HEAD", verify, &cache, None).unwrap();
    std::fs::remove_file(&marker).unwrap();
    independently_verify_candidate_cached(&worktree, "HEAD", verify, &cache, None).unwrap();

    let cancel = hi_agent::TurnCancellation::new();
    cancel.cancel();
    let error =
        independently_verify_candidate_cached(&worktree, "HEAD", verify, &cache, Some(cancel))
            .expect_err("cancellation must take precedence over a valid cached pass");
    assert!(is_verifier_cancelled(&error));

    hi_tools::worktree::cleanup(&root, &[worktree]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn independently_verify_candidate_cached_cancel_skips_cache() {
    let original_sandbox = std::env::var_os("HI_SANDBOX");
    unsafe { std::env::set_var("HI_SANDBOX", "off") };
    let (root, worktree) = candidate_fixture("verify-cache-cancel");
    std::fs::write(worktree.join("value.txt"), "after\n").unwrap();
    let cache = root.join(".hi-test-verify-cache");
    let cancel = hi_agent::TurnCancellation::new();
    let cancel_thread = cancel.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        cancel_thread.cancel();
    });
    let started = Instant::now();
    let error =
        independently_verify_candidate_cached(&worktree, "HEAD", "sleep 30", &cache, Some(cancel))
            .expect_err("cancel must stop independent verify");
    let elapsed = started.elapsed();
    match original_sandbox {
        Some(value) => unsafe { std::env::set_var("HI_SANDBOX", value) },
        None => unsafe { std::env::remove_var("HI_SANDBOX") },
    }
    assert!(
        elapsed < Duration::from_secs(3),
        "cancel must kill the verifier instead of waiting: {elapsed:?}"
    );
    assert!(
        is_verifier_cancelled(&error),
        "cancel must be distinct from a generic verify fail: {error:#}"
    );
    let rendered = format!("{error:#}");
    assert!(
        !rendered.contains("configured verifier"),
        "cancel must not be wrapped as a verify failure: {rendered}"
    );
    let wrote_cache = std::fs::read_dir(&cache)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("json"))
        })
        .unwrap_or(false);
    assert!(
        !wrote_cache,
        "cancelled verify must not write a cache record"
    );

    hi_tools::worktree::cleanup(&root, &[worktree]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn apply_candidate_and_reverify_cancel_rolls_back_destination() {
    let original_sandbox = std::env::var_os("HI_SANDBOX");
    unsafe { std::env::set_var("HI_SANDBOX", "off") };
    let (root, worktree) = candidate_fixture("apply-verify-cancel");
    std::fs::write(worktree.join("value.txt"), "after\n").unwrap();
    let dest_value = root.join("value.txt");
    let cancel = hi_agent::TurnCancellation::new();
    let cancel_thread = cancel.clone();
    let watch = dest_value.clone();
    std::thread::spawn(move || {
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(5) {
            if std::fs::read_to_string(&watch).ok().as_deref() == Some("after\n") {
                std::thread::sleep(Duration::from_millis(50));
                cancel_thread.cancel();
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        cancel_thread.cancel();
    });
    let started = Instant::now();
    let error = apply_candidate_and_reverify_cancellable(
        &worktree,
        "HEAD",
        &root,
        &root.join(".hi-test-state"),
        "sleep 30",
        Some(cancel),
    )
    .expect_err("cancel during destination verify must reject the candidate");
    let elapsed = started.elapsed();
    match original_sandbox {
        Some(value) => unsafe { std::env::set_var("HI_SANDBOX", value) },
        None => unsafe { std::env::remove_var("HI_SANDBOX") },
    }
    assert!(
        elapsed < Duration::from_secs(8),
        "cancel must kill destination verify instead of waiting: {elapsed:?}"
    );
    assert!(
        is_destination_verify_cancelled(&error),
        "cancel after apply must be distinct from a generic verify fail: {error:#}"
    );
    let rendered = format!("{error:#}");
    assert!(
        !rendered.contains("destination verification `sleep 30` failed"),
        "cancel must not be wrapped as a verify failure: {rendered}"
    );
    assert_eq!(
        std::fs::read_to_string(&dest_value).unwrap(),
        "before\n",
        "destination must be rolled back after cancel during verify"
    );

    hi_tools::worktree::cleanup(&root, &[worktree]);
    let _ = std::fs::remove_dir_all(root);
}

fn candidate_fixture(label: &str) -> (PathBuf, PathBuf) {
    let root = temp_path(label);
    std::fs::create_dir_all(&root).unwrap();
    git_ok(&root, &["init", "-q"]);
    git_ok(&root, &["config", "user.email", "test@example.invalid"]);
    git_ok(&root, &["config", "user.name", "Hi Test"]);
    std::fs::write(root.join("value.txt"), "before\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            root.join("value.txt"),
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();
    }
    git_ok(&root, &["add", "value.txt"]);
    git_ok(&root, &["commit", "-qm", "base"]);

    let worktree = root.join("candidate");
    git_ok(
        &root,
        &[
            "worktree",
            "add",
            "--detach",
            worktree.to_str().unwrap(),
            "HEAD",
        ],
    );
    (root, worktree)
}

fn git_ok(root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn git_stdout(root: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}

#[test]
fn delegate_route_overrides_switch_the_child_to_the_openai_compat_route() {
    let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("hi-delegate-route-{}-{id}", std::process::id()));
    let workspace = base.join("ws");
    let state = base.join("state");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&state).unwrap();
    let runner = crate::delegate::CliDelegateRunner::new(
        PathBuf::from("hi"),
        "pipenetwork".into(),
        "pipe/glm-5.2".into(),
        "https://api.pipenetwork.ai/v1".into(),
        "cloud-key".into(),
        Some("true".into()),
        None,
        None,
        1,
        workspace,
        state,
    )
    .unwrap();

    // No override → the driver's route, untouched.
    let inherited = runner.effective_route(&hi_agent::SubagentRoute::default());
    assert_eq!(
        inherited,
        (
            "pipenetwork".into(),
            "pipe/glm-5.2".into(),
            "https://api.pipenetwork.ai/v1".into(),
            "cloud-key".into()
        )
    );

    // Model-only override stays on the driver's provider.
    let model_only = runner.effective_route(&hi_agent::SubagentRoute {
        model: Some("pipe/glm-4-flash".into()),
        base_url: None,
        api_key: None,
    });
    assert_eq!(model_only.0, "pipenetwork");
    assert_eq!(model_only.1, "pipe/glm-4-flash");

    // Endpoint override moves the child to the generic OpenAI-compatible
    // provider (local MLX/Ollama/llama.cpp servers all speak it), and the
    // cloud key must NOT leak to the local endpoint.
    let local = runner.effective_route(&hi_agent::SubagentRoute {
        model: Some("qwen3-coder".into()),
        base_url: Some("http://127.0.0.1:18080/v1".into()),
        api_key: None,
    });
    assert_eq!(
        local,
        (
            "openai".into(),
            "qwen3-coder".into(),
            "http://127.0.0.1:18080/v1".into(),
            String::new()
        )
    );
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn missing_verify_pipeline_derives_a_build_gate_for_known_project_types() {
    use std::path::PathBuf;
    let base = std::env::temp_dir().join(format!("hi-derive-verify-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);

    // Rust workspace → cargo check gate.
    let rust_ws = base.join("rust/ws");
    std::fs::create_dir_all(&rust_ws).unwrap();
    std::fs::write(rust_ws.join("Cargo.toml"), "[workspace]\n").unwrap();
    assert_eq!(
        crate::delegate::derive_default_verify(&rust_ws).as_deref(),
        Some("cargo check --workspace --all-targets")
    );

    // Unknown project type → still no gate (delegate stays unavailable).
    let plain = base.join("plain/ws");
    std::fs::create_dir_all(&plain).unwrap();
    assert_eq!(crate::delegate::derive_default_verify(&plain), None);

    // The runner picks the derived gate up when the session configured none —
    // and an explicit (even trivial) pipeline still wins.
    let state = base.join("state");
    std::fs::create_dir_all(&state).unwrap();
    let derived = crate::delegate::CliDelegateRunner::new(
        PathBuf::from("hi"),
        "pipenetwork".into(),
        "pipe/glm-5.2".into(),
        "https://api.pipenetwork.ai/v1".into(),
        "k".into(),
        None,
        None,
        None,
        1,
        rust_ws.clone(),
        state.clone(),
    )
    .unwrap();
    assert_eq!(
        derived.default_verify_for_tests().as_deref(),
        Some("cargo check --workspace --all-targets")
    );
    let explicit = crate::delegate::CliDelegateRunner::new(
        PathBuf::from("hi"),
        "pipenetwork".into(),
        "pipe/glm-5.2".into(),
        "https://api.pipenetwork.ai/v1".into(),
        "k".into(),
        Some("true".into()),
        None,
        None,
        1,
        rust_ws,
        state,
    )
    .unwrap();
    assert_eq!(explicit.default_verify_for_tests().as_deref(), Some("true"));
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn delegate_runner_rebinds_candidates_to_the_new_authoritative_root() {
    use hi_agent::DelegateRunner;

    let temporary = tempfile::tempdir().unwrap();
    let first = temporary.path().join("first");
    let second = temporary.path().join("second");
    let first_state = temporary.path().join("first-state");
    let second_state = temporary.path().join("second-state");
    for path in [&first, &second, &first_state, &second_state] {
        std::fs::create_dir_all(path).unwrap();
    }
    let runner = crate::delegate::CliDelegateRunner::new(
        PathBuf::from("hi"),
        "openai".into(),
        "model".into(),
        "http://127.0.0.1".into(),
        "key".into(),
        Some("true".into()),
        None,
        None,
        1,
        first.clone(),
        first_state.clone(),
    )
    .unwrap();
    assert!(runner.is_bound_to_workspace(&first, &first_state));
    assert!(runner.bind_workspace(&second, &second_state));
    assert!(!runner.is_bound_to_workspace(&first, &first_state));
    assert!(runner.is_bound_to_workspace(&second, &second_state));
}
