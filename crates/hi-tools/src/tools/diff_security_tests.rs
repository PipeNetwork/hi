#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use super::working_tree_diff_plain_in;

fn git(root: &std::path::Path, args: &[&str]) {
    let status = std::process::Command::new("git")
        .current_dir(root)
        .args(args)
        .status()
        .expect("git command starts");
    assert!(status.success(), "git {args:?} failed");
}

#[cfg(unix)]
#[tokio::test]
async fn diff_tool_never_invokes_repository_external_diff_driver() {
    let root = tempfile::tempdir().unwrap();
    git(root.path(), &["init", "-q"]);
    git(
        root.path(),
        &["config", "user.email", "test@example.invalid"],
    );
    git(root.path(), &["config", "user.name", "Test"]);
    std::fs::write(root.path().join("tracked.txt"), "before\n").unwrap();
    git(root.path(), &["add", "tracked.txt"]);
    git(root.path(), &["commit", "-qm", "base"]);

    let marker = root.path().join("external-diff-ran");
    let helper = root.path().join("external-diff.sh");
    std::fs::write(
        &helper,
        format!("#!/bin/sh\nprintf ran > '{}'\n", marker.display()),
    )
    .unwrap();
    std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o700)).unwrap();
    git(
        root.path(),
        &["config", "diff.external", helper.to_str().unwrap()],
    );
    std::fs::write(root.path().join("tracked.txt"), "after\n").unwrap();

    let output = working_tree_diff_plain_in(root.path()).await;

    assert!(
        !marker.exists(),
        "read-only diff executed a repository helper"
    );
    assert!(output.contains("-before"), "{output}");
    assert!(output.contains("+after"), "{output}");
}
