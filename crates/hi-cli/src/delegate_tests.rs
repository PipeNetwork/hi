use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use crate::candidate_gate::{inspect_child_report, parse_name_status, staged_candidate_diff};
use crate::candidate_merge::apply_candidate_and_reverify;

static TEST_ID: AtomicU64 = AtomicU64::new(0);

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
    assert_eq!(changed, vec!["value.txt"]);
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
    assert_eq!(changed, vec!["created.txt"]);
    assert_eq!(
        std::fs::read_to_string(destination.join("created.txt")).unwrap(),
        "scoped\n"
    );

    hi_tools::worktree::cleanup(&root, &[worktree]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn verifier_mutation_is_unstable_and_rolls_back() {
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
    assert_eq!(
        std::fs::read_to_string(root.join("value.txt")).unwrap(),
        "before\n"
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
