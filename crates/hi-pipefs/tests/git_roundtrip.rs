use std::fs;
use std::path::Path;
use std::process::Command;

use hi_pipefs::{apply_archive, build_revision};

fn git(cwd: &Path, arguments: &[&str]) -> String {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(cwd)
        .output()
        .expect("git must be installed for the PipeFS Git acceptance test");
    assert!(
        output.status.success(),
        "git {:?} failed:\nstdout: {}\nstderr: {}",
        arguments,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn native_git_clone_checkout_add_commit_survives_restore() {
    if Command::new("git").arg("--version").output().is_err() {
        return;
    }

    let fixture = tempfile::tempdir().unwrap();
    let origin = fixture.path().join("origin.git");
    let seed = fixture.path().join("seed");
    fs::create_dir(&seed).unwrap();
    git(
        fixture.path(),
        &["init", "--bare", origin.to_str().unwrap()],
    );
    git(&seed, &["init"]);
    git(&seed, &["config", "user.name", "PipeFS Test"]);
    git(&seed, &["config", "user.email", "pipefs@example.invalid"]);
    fs::write(seed.join("tracked.txt"), "main\n").unwrap();
    git(&seed, &["add", "tracked.txt"]);
    git(&seed, &["commit", "-m", "initial"]);
    git(&seed, &["branch", "-M", "main"]);
    git(
        &seed,
        &["remote", "add", "origin", origin.to_str().unwrap()],
    );
    git(&seed, &["push", "-u", "origin", "main"]);

    let source = tempfile::tempdir().unwrap();
    git(
        source.path(),
        &["clone", origin.to_str().unwrap(), "repository"],
    );
    let repository = source.path().join("repository");
    // A bare repository created for a test may not advertise its main branch.
    git(&repository, &["checkout", "main"]);
    git(&repository, &["config", "user.name", "PipeFS Test"]);
    git(
        &repository,
        &["config", "user.email", "pipefs@example.invalid"],
    );

    // Machine A's first durable checkpoint is a full archive. Subsequent Git
    // work is intentionally represented as deltas so the test exercises the
    // same bounded restore-chain shape used for cross-machine resume.
    let full = build_revision(source.path(), None, true).unwrap();
    git(&repository, &["checkout", "-b", "feature"]);
    fs::write(repository.join("tracked.txt"), "feature\n").unwrap();
    git(&repository, &["add", "tracked.txt"]);
    git(&repository, &["commit", "-m", "feature change"]);
    assert!(git(&repository, &["status", "--porcelain"]).is_empty());

    let feature_delta = build_revision(source.path(), Some(&full.snapshot), false).unwrap();
    let machine_b = tempfile::tempdir().unwrap();
    let full_snapshot = apply_archive(machine_b.path(), &full.bytes, None).unwrap();
    let feature_snapshot =
        apply_archive(machine_b.path(), &feature_delta.bytes, Some(&full_snapshot)).unwrap();
    let restored_repository = machine_b.path().join("repository");

    assert_eq!(
        git(&restored_repository, &["branch", "--show-current"]).trim(),
        "feature"
    );
    assert!(git(&restored_repository, &["log", "-1", "--format=%s"]).contains("feature change"));
    assert!(git(&restored_repository, &["status", "--porcelain"]).is_empty());
    git(&restored_repository, &["checkout", "main"]);
    assert_eq!(
        fs::read_to_string(restored_repository.join("tracked.txt")).unwrap(),
        "main\n"
    );
    git(&restored_repository, &["checkout", "feature"]);
    fs::write(
        restored_repository.join("after-restore.txt"),
        "native git\n",
    )
    .unwrap();
    git(&restored_repository, &["add", "after-restore.txt"]);
    git(
        &restored_repository,
        &["commit", "-m", "commit after restore"],
    );
    assert!(git(&restored_repository, &["status", "--porcelain"]).is_empty());

    // Machine B checkpoints its new commit. Machine C must reconstruct the
    // repository from the full-plus-deltas chain, including .git metadata.
    let post_restore_delta =
        build_revision(machine_b.path(), Some(&feature_snapshot), false).unwrap();
    let machine_c = tempfile::tempdir().unwrap();
    let machine_c_full = apply_archive(machine_c.path(), &full.bytes, None).unwrap();
    let machine_c_feature = apply_archive(
        machine_c.path(),
        &feature_delta.bytes,
        Some(&machine_c_full),
    )
    .unwrap();
    apply_archive(
        machine_c.path(),
        &post_restore_delta.bytes,
        Some(&machine_c_feature),
    )
    .unwrap();
    let machine_c_repository = machine_c.path().join("repository");
    assert_eq!(
        git(&machine_c_repository, &["branch", "--show-current"]).trim(),
        "feature"
    );
    assert_eq!(
        git(&machine_c_repository, &["log", "-1", "--format=%s"]).trim(),
        "commit after restore"
    );
    assert_eq!(
        fs::read_to_string(machine_c_repository.join("after-restore.txt")).unwrap(),
        "native git\n"
    );
    assert!(git(&machine_c_repository, &["status", "--porcelain"]).is_empty());
}
