use super::*;
use crate::VerificationMode;
use std::process::Command;

fn change(path: &str, kind: FileChangeKind, after_len: Option<u64>) -> FileChange {
    change_with_before(path, kind, None, after_len)
}

fn change_with_before(
    path: &str,
    kind: FileChangeKind,
    before_len: Option<u64>,
    after_len: Option<u64>,
) -> FileChange {
    FileChange {
        path: path.into(),
        kind,
        before_digest: None,
        after_digest: None,
        before_len,
        after_len,
        before_mode: None,
        after_mode: None,
    }
}

fn git(root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
async fn ignored_untracked_runtime_changes_are_not_reviewable() {
    let root = tempfile::tempdir().unwrap();
    git(root.path(), &["init", "-q"]);
    git(root.path(), &["config", "user.email", "test@example.com"]);
    git(root.path(), &["config", "user.name", "Test"]);
    std::fs::write(
        root.path().join(".gitignore"),
        "ignored.db\ntracked.db\ndeleted-ignored.db\n",
    )
    .unwrap();
    std::fs::write(root.path().join("tracked.db"), "tracked baseline\n").unwrap();
    std::fs::write(root.path().join("deleted-ignored.db"), "tracked deletion\n").unwrap();
    git(root.path(), &["add", ".gitignore"]);
    git(
        root.path(),
        &["add", "-f", "tracked.db", "deleted-ignored.db"],
    );
    git(root.path(), &["commit", "-qm", "baseline"]);
    git(root.path(), &["rm", "-q", "deleted-ignored.db"]);
    std::fs::write(root.path().join("ignored.db"), "runtime\n").unwrap();
    std::fs::write(root.path().join("candidate.rs"), "fn candidate() {}\n").unwrap();

    let ignored = change_with_before(
        "ignored.db",
        FileChangeKind::Modify,
        Some(31 * 1024 * 1024),
        Some(40 * 1024),
    );
    let tracked = change_with_before(
        "tracked.db",
        FileChangeKind::Modify,
        Some(31 * 1024 * 1024),
        Some(40 * 1024),
    );
    let untracked = change("candidate.rs", FileChangeKind::Create, Some(20));
    let staged_deletion =
        change_with_before("deleted-ignored.db", FileChangeKind::Delete, Some(17), None);
    let changes = vec![ignored.clone(), tracked, untracked, staged_deletion];

    let reviewable = reviewable_changes(root.path(), &changes).await;
    assert_eq!(
        reviewable
            .iter()
            .map(|change| change.path.as_str())
            .collect::<Vec<_>>(),
        vec!["tracked.db", "candidate.rs", "deleted-ignored.db"],
        "tracked files, staged deletions, and non-ignored untracked files remain reviewable"
    );

    let ignored_before =
        reviewable_content_revision(root.path(), std::slice::from_ref(&ignored), None)
            .await
            .unwrap();
    std::fs::write(root.path().join("ignored.db"), "changed\n").unwrap();
    let ignored_after =
        reviewable_content_revision(root.path(), std::slice::from_ref(&ignored), None)
            .await
            .unwrap();
    assert_eq!(
        ignored_before, ignored_after,
        "an ignored-only runtime rewrite must not look like review progress"
    );

    let contract = TaskContract::derive("reset the database", VerificationMode::Auto);
    assert!(
        assess_reviewable(root.path(), &contract, &[ignored], "reset the database")
            .await
            .is_empty(),
        "an ignored runtime database must not be diagnosed as a large source rewrite"
    );
}

#[tokio::test]
async fn exact_revision_detects_same_length_oversized_tracked_rewrite() {
    const LEN: usize = 16 * 1024 * 1024 + 1;
    let root = tempfile::tempdir().unwrap();
    git(root.path(), &["init", "-q"]);
    git(root.path(), &["config", "user.email", "test@example.com"]);
    git(root.path(), &["config", "user.name", "Test"]);
    std::fs::write(root.path().join("large.bin"), vec![b'a'; LEN]).unwrap();
    git(root.path(), &["add", "large.bin"]);
    git(root.path(), &["commit", "-qm", "baseline"]);
    let changes = vec![change_with_before(
        "large.bin",
        FileChangeKind::Modify,
        Some(LEN as u64),
        Some(LEN as u64),
    )];

    let before = reviewable_content_revision(root.path(), &changes, None)
        .await
        .unwrap();
    std::fs::write(root.path().join("large.bin"), vec![b'b'; LEN]).unwrap();
    let after = reviewable_content_revision(root.path(), &changes, None)
        .await
        .unwrap();

    assert_ne!(
        before, after,
        "large files must be streamed exactly rather than inheriting the ledger's size-only digest"
    );
}

#[tokio::test]
async fn staged_ignored_deletion_is_reviewable_from_nested_workspace_root() {
    let repo = tempfile::tempdir().unwrap();
    git(repo.path(), &["init", "-q"]);
    git(repo.path(), &["config", "user.email", "test@example.com"]);
    git(repo.path(), &["config", "user.name", "Test"]);
    let workspace = repo.path().join("nested");
    std::fs::create_dir(&workspace).unwrap();
    std::fs::write(workspace.join(".gitignore"), "gone.db\n").unwrap();
    std::fs::write(workspace.join("gone.db"), "tracked\n").unwrap();
    git(&workspace, &["add", ".gitignore"]);
    git(&workspace, &["add", "-f", "gone.db"]);
    git(&workspace, &["commit", "-qm", "baseline"]);
    git(&workspace, &["rm", "-q", "gone.db"]);

    let deletion = change_with_before("gone.db", FileChangeKind::Delete, Some(8), None);
    let reviewable = reviewable_changes(&workspace, &[deletion]).await;
    assert_eq!(
        reviewable.len(),
        1,
        "staged deletion must survive ignore filtering"
    );
    assert_eq!(reviewable[0].path, "gone.db");
}

#[tokio::test]
async fn staged_rename_keeps_ignored_old_path_reviewable() {
    let root = tempfile::tempdir().unwrap();
    git(root.path(), &["init", "-q"]);
    git(root.path(), &["config", "user.email", "test@example.com"]);
    git(root.path(), &["config", "user.name", "Test"]);
    std::fs::write(root.path().join(".gitignore"), "old.db\n").unwrap();
    std::fs::write(root.path().join("old.db"), "tracked\n").unwrap();
    git(root.path(), &["add", ".gitignore"]);
    git(root.path(), &["add", "-f", "old.db"]);
    git(root.path(), &["commit", "-qm", "baseline"]);
    git(root.path(), &["mv", "old.db", "new.db"]);

    let deletion = change_with_before("old.db", FileChangeKind::Delete, Some(8), None);
    let reviewable = reviewable_changes(root.path(), &[deletion]).await;
    assert_eq!(
        reviewable.len(),
        1,
        "rename detection must not hide the removed tracked path"
    );
    assert_eq!(reviewable[0].path, "old.db");
}

#[tokio::test]
async fn exact_revision_distinguishes_stable_missing_nodes() {
    let root = tempfile::tempdir().unwrap();
    let changes = vec![change("gone.txt", FileChangeKind::Delete, None)];

    let missing = reviewable_content_revision(root.path(), &changes, None)
        .await
        .unwrap();
    std::fs::write(root.path().join("gone.txt"), "present").unwrap();
    let present = reviewable_content_revision(root.path(), &changes, None)
        .await
        .unwrap();
    std::fs::remove_file(root.path().join("gone.txt")).unwrap();
    let missing_again = reviewable_content_revision(root.path(), &changes, None)
        .await
        .unwrap();

    assert_ne!(missing, present);
    assert_eq!(missing, missing_again);
}

#[tokio::test]
async fn exact_revision_is_inconclusive_when_turn_is_cancelled() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("source.rs"), "fn main() {}\n").unwrap();
    let cancellation = crate::TurnCancellation::new();
    cancellation.cancel();

    assert!(
        reviewable_content_revision(
            root.path(),
            &[change("source.rs", FileChangeKind::Modify, Some(13))],
            Some(cancellation),
        )
        .await
        .is_none(),
        "a cancelled exact read must not publish an equality fingerprint"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn exact_revision_does_not_follow_intermediate_symlink_outside_workspace() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("secret.txt"), "outside\n").unwrap();
    symlink(outside.path(), root.path().join("alias")).unwrap();

    assert!(
        reviewable_content_revision(
            root.path(),
            &[change("alias/secret.txt", FileChangeKind::Modify, Some(8),)],
            None,
        )
        .await
        .is_none(),
        "an intermediate symlink escape must be inconclusive"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn exact_revision_includes_symlink_target_and_mode() {
    use std::os::unix::fs::{PermissionsExt as _, symlink};

    let root = tempfile::tempdir().unwrap();
    let link_change = change("link", FileChangeKind::Modify, None);
    symlink("one", root.path().join("link")).unwrap();
    let link_one =
        reviewable_content_revision(root.path(), std::slice::from_ref(&link_change), None)
            .await
            .unwrap();
    std::fs::remove_file(root.path().join("link")).unwrap();
    symlink("two", root.path().join("link")).unwrap();
    let link_two = reviewable_content_revision(root.path(), &[link_change], None)
        .await
        .unwrap();
    assert_ne!(link_one, link_two, "same-length link targets must differ");

    let script = root.path().join("script.sh");
    std::fs::write(&script, "#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o644)).unwrap();
    let script_change = change("script.sh", FileChangeKind::Modify, Some(10));
    let mode_644 =
        reviewable_content_revision(root.path(), std::slice::from_ref(&script_change), None)
            .await
            .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mode_755 = reviewable_content_revision(root.path(), &[script_change], None)
        .await
        .unwrap();
    assert_ne!(
        mode_644, mode_755,
        "executable-bit changes are review progress"
    );
}

#[tokio::test]
async fn non_git_workspace_retains_conservative_hygiene_coverage() {
    let root = tempfile::tempdir().unwrap();
    let contract = TaskContract::derive("rewrite the database fixture", VerificationMode::Auto);
    let changes = vec![change_with_before(
        "fixture.db",
        FileChangeKind::Modify,
        Some(31 * 1024 * 1024),
        Some(40 * 1024),
    )];

    let findings = assess_reviewable(
        root.path(),
        &contract,
        &changes,
        "rewrite the database fixture",
    )
    .await;
    assert!(
        findings
            .iter()
            .any(|finding| finding.reason.contains("fixture.db")),
        "a Git classification failure must retain changes, not fail open"
    );
}

#[test]
fn unreferenced_creates_need_a_narrow_contract() {
    let contract = TaskContract::derive("fix src/parser.rs", VerificationMode::Auto);
    assert!(!contract.referenced_paths.is_empty());
    let changes = vec![
        change("src/parser.rs", FileChangeKind::Modify, Some(100)),
        change("a.rs", FileChangeKind::Create, Some(10)),
        change("b.rs", FileChangeKind::Create, Some(10)),
        change("c.rs", FileChangeKind::Create, Some(10)),
    ];
    let findings = assess(&contract, &changes, "fix src/parser.rs");
    assert!(
        findings.iter().any(|f| f.reason.contains("unreferenced")),
        "{findings:?}"
    );
    let broad = TaskContract::derive("implement the feature", VerificationMode::Auto);
    assert!(broad.referenced_paths.is_empty());
    assert!(
        assess(&broad, &changes, "implement the feature")
            .iter()
            .all(|f| !f.reason.contains("unreferenced"))
    );
}

#[test]
fn dependency_manifest_is_flagged_unless_asked() {
    let contract = TaskContract::derive("fix the parser in src/lib.rs", VerificationMode::Auto);
    let changes = vec![change("Cargo.toml", FileChangeKind::Modify, Some(200))];
    let findings = assess(&contract, &changes, "fix the parser in src/lib.rs");
    assert!(
        findings
            .iter()
            .any(|f| f.reason.contains("dependency manifest")),
        "{findings:?}"
    );
    let asked = assess(
        &contract,
        &changes,
        "add crate serde to Cargo.toml for the parser",
    );
    assert!(
        asked
            .iter()
            .all(|f| !f.reason.contains("dependency manifest")),
        "{asked:?}"
    );

    let broad = TaskContract::derive(
        "connect the app to the inference API and make it work",
        VerificationMode::Auto,
    );
    assert!(broad.referenced_paths.is_empty());
    let broad_findings = assess(
        &broad,
        &changes,
        "connect the app to the inference API and make it work",
    );
    assert!(
        broad_findings
            .iter()
            .all(|f| !f.reason.contains("dependency manifest")),
        "broad integration work may naturally require a dependency: {broad_findings:?}"
    );
}

#[test]
fn oversized_create_is_flagged() {
    let contract = TaskContract::derive("add a helper", VerificationMode::Auto);
    let changes = vec![change(
        "src/huge.rs",
        FileChangeKind::Create,
        Some(LARGE_FILE_BYTES + 1),
    )];
    let findings = assess(&contract, &changes, "add a helper");
    assert!(
        findings.iter().any(|f| f.reason.contains("bytes")),
        "{findings:?}"
    );
}

#[test]
fn small_patch_on_already_large_file_is_not_flagged() {
    let contract =
        TaskContract::derive("fold stream_area into the Run row", VerificationMode::Auto);
    let changes = vec![change_with_before(
        "crates/hi-tui/src/app/render.rs",
        FileChangeKind::Modify,
        Some(113_013),
        Some(113_050),
    )];
    let findings = assess(&contract, &changes, "fold stream_area into the Run row");
    assert!(
        findings
            .iter()
            .all(|f| !f.reason.contains("rewriting large files")),
        "small delta on a large file must not look like a rewrite: {findings:?}"
    );
}

#[test]
fn modify_without_before_len_is_not_flagged() {
    let contract = TaskContract::derive("edit render.rs", VerificationMode::Auto);
    let changes = vec![change(
        "crates/hi-tui/src/app/render.rs",
        FileChangeKind::Modify,
        Some(LARGE_FILE_BYTES + 1),
    )];
    let findings = assess(&contract, &changes, "edit render.rs");
    assert!(
        findings
            .iter()
            .all(|f| !f.reason.contains("rewriting large files")),
        "unknown baseline must not flag an already-large file: {findings:?}"
    );
}

#[test]
fn large_growth_on_modify_is_flagged() {
    let contract = TaskContract::derive("rewrite the renderer", VerificationMode::Auto);
    let changes = vec![change_with_before(
        "src/render.rs",
        FileChangeKind::Modify,
        Some(1_024),
        Some(LARGE_FILE_BYTES + 2_048),
    )];
    let findings = assess(&contract, &changes, "rewrite the renderer");
    assert!(
        findings.iter().any(|f| f.reason.contains("changed by")),
        "{findings:?}"
    );
}
