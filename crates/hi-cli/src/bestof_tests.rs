use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

fn git_ok(root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn init_candidate_repository(root: &Path) {
    std::fs::create_dir(root).unwrap();
    git_ok(root, &["init", "--quiet"]);
    std::fs::write(root.join("tracked.txt"), "base\n").unwrap();
    git_ok(root, &["add", "tracked.txt"]);
    git_ok(
        root,
        &[
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@invalid",
            "commit",
            "--quiet",
            "-m",
            "base",
        ],
    );
}

fn worktree_registry(root: &Path) -> Vec<u8> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["worktree", "list", "--porcelain"])
        .output()
        .unwrap();
    assert!(output.status.success());
    output.stdout
}

fn test_opts<'a>(exe: &'a Path, verify: &'a str) -> BestOf<'a> {
    BestOf {
        exe,
        provider: "openai",
        model: "test-model",
        base_url: "http://127.0.0.1:9/v1",
        api_key: "test-key",
        verify,
        prompt: "do the thing",
        candidates: 1,
        max_steps: Some(1),
        max_tool_calls: Some(0),
        max_verify: 1,
        workspace_root: Path::new("/"),
        state_root: Path::new("/tmp"),
        report: None,
        targets: None,
        max_concurrency: 1,
        apply: true,
        fuzz: None,
        expected_workspace_digest: None,
        judge: hi_research::JudgeChoice::Tests,
        research_id: None,
        snippet_block: String::new(),
    }
}

fn temp_file(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "hi-bestof-{label}-{}-candidate-0.report.json",
        std::process::id()
    ))
}

#[test]
fn managed_best_of_limits_use_typed_defaults_and_positive_legacy_overrides() {
    let defaults = hi_workspace::ResolvedHarnessSettings::default().jobs;
    assert_eq!(
        managed_timeout_from_value(None, defaults.candidate_timeout),
        Duration::from_secs(15 * 60)
    );
    assert_eq!(
        managed_timeout_from_value(Some("0"), defaults.queue_timeout),
        Duration::from_secs(5 * 60)
    );
    assert_eq!(
        managed_timeout_from_value(Some("invalid"), defaults.verifier_timeout),
        Duration::from_secs(2 * 60)
    );
    assert_eq!(
        managed_timeout_from_value(Some("17"), defaults.candidate_timeout),
        Duration::from_secs(17)
    );
}

#[test]
fn parallel_best_of_preparation_never_registers_source_worktrees() {
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    let state = temporary.path().join("state");
    std::fs::create_dir(&state).unwrap();
    init_candidate_repository(&source);
    std::fs::write(source.join("tracked.txt"), "dirty\n").unwrap();
    std::fs::write(source.join("untracked.txt"), "new\n").unwrap();
    let before = worktree_registry(&source);

    for round in 0..8 {
        let candidates = std::thread::scope(|scope| {
            (0..4)
                .map(|index| {
                    let owner = temporary.path().join(format!("candidate-{round}-{index}"));
                    let source = &source;
                    let state = &state;
                    scope.spawn(move || prepare_detached_candidate(source, state, &owner).unwrap())
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(worktree_registry(&source), before);
        assert!(
            candidates
                .windows(2)
                .all(|pair| { pair[0].source_snapshot_id() == pair[1].source_snapshot_id() })
        );
        assert!(candidates.iter().all(|candidate| {
            candidate.root().join(".git").is_dir()
                && std::fs::read_to_string(candidate.root().join("tracked.txt")).unwrap()
                    == "dirty\n"
                && candidate.root().join("untracked.txt").is_file()
        }));
        drop(candidates);
        assert_eq!(worktree_registry(&source), before);
    }
}

#[test]
fn run_candidate_rejects_nonzero_exit_even_without_a_report() {
    let exe = Path::new("/bin/false");
    if !exe.exists() {
        return;
    }
    let opts = test_opts(exe, "true");
    let report = temp_file("failure");
    let log = report.with_extension("log");
    let workspace = std::env::current_dir().unwrap().canonicalize().unwrap();
    let source = temp_file("source-root");
    std::fs::create_dir_all(&source).unwrap();
    let runtime_owner = tempfile::tempdir().unwrap();
    let child_paths = crate::child_process::CandidateChildPaths::prepare_test(
        &workspace,
        &runtime_owner.path().join("runtime"),
    )
    .unwrap();
    let execution = run_candidate(&opts, 0, 0.2, &report, &log, &source, &child_paths);
    assert!(!execution.process_succeeded);
    assert!(!execution.typed_child_succeeded);
    assert!(log.exists(), "candidate log must be persisted");
    let _ = std::fs::remove_file(report);
    let _ = std::fs::remove_file(log);
    let _ = std::fs::remove_dir_all(source);
}

#[test]
fn candidate_arguments_preserve_both_explicit_caps_including_zero_tools() {
    let mut arguments = Vec::new();
    append_execution_cap_arguments(&mut arguments, Some(7), Some(0));
    let arguments = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(arguments, ["--max-steps", "7", "--max-tool-calls", "0"]);
    let mut arguments = Vec::new();
    append_execution_cap_arguments(&mut arguments, None, None);
    assert!(arguments.is_empty());
}

#[test]
fn exit_zero_without_typed_report_is_not_eligible() {
    let exe = Path::new("/bin/true");
    if !exe.exists() {
        return;
    }
    let opts = test_opts(exe, "true");
    let report = temp_file("missing-report");
    let log = report.with_extension("log");
    let workspace = std::env::current_dir().unwrap().canonicalize().unwrap();
    let source = temp_file("source-root-missing");
    std::fs::create_dir_all(&source).unwrap();
    let runtime_owner = tempfile::tempdir().unwrap();
    let child_paths = crate::child_process::CandidateChildPaths::prepare_test(
        &workspace,
        &runtime_owner.path().join("runtime"),
    )
    .unwrap();
    let execution = run_candidate(&opts, 0, 0.2, &report, &log, &source, &child_paths);
    assert!(execution.process_succeeded);
    assert!(!execution.typed_child_succeeded);
    assert!(
        execution
            .child_gate_reason
            .contains("typed child gate failed")
    );
    let _ = std::fs::remove_file(report);
    let _ = std::fs::remove_file(log);
    let _ = std::fs::remove_dir_all(source);
}

#[test]
fn empty_verifier_is_rejected_before_candidate_setup() {
    let opts = test_opts(Path::new("/bin/true"), "  ");
    let error = run(&opts).expect_err("empty verifier must be a usage error");
    assert!(format!("{error:#}").contains("resolved non-empty"));
}

#[test]
fn model_judge_skips_empty_verifier_usage_error() {
    let mut opts = test_opts(Path::new("/bin/true"), "  ");
    opts.judge = hi_research::JudgeChoice::Model;
    let error = run(&opts).expect_err("workspace setup still fails");
    assert!(
        !format!("{error:#}").contains("resolved non-empty"),
        "{error:#}"
    );
}

#[test]
fn bounded_map_preserves_order_and_limits_parallelism() {
    let items = [0usize, 1, 2, 3];
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let first_wave = Arc::new(Barrier::new(2));
    let results = bounded_ordered_map(&items, 2, |item| {
        let current = active.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(current, Ordering::SeqCst);
        if *item < 2 {
            first_wave.wait();
        }
        std::thread::sleep(Duration::from_millis((3 - item) as u64));
        active.fetch_sub(1, Ordering::SeqCst);
        item * 10
    });
    assert_eq!(results, [0, 10, 20, 30]);
    assert_eq!(peak.load(Ordering::SeqCst), 2);
}

#[test]
fn percentiles_use_nearest_rank() {
    let values = latency_percentiles([1, 2, 3, 4, 100]);
    assert_eq!(values.samples, 5);
    assert_eq!(values.p50_ms, 3);
    assert_eq!(values.p95_ms, 100);
}

#[test]
fn verify_concurrency_clamps_and_falls_back() {
    assert_eq!(
        configured_verify_concurrency(Some("999"), 2),
        MAX_VERIFY_CONCURRENCY
    );
    assert_eq!(configured_verify_concurrency(Some("0"), 2), 1);
    assert_eq!(configured_verify_concurrency(Some("invalid"), 3), 3);
    assert_eq!(configured_verify_concurrency(None, 3), 3);
}
