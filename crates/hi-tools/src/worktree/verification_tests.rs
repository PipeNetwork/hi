#[cfg(unix)]
#[test]
fn completed_verifier_stops_background_descendants() {
    let dir = tempfile::tempdir().unwrap();
    let command = "(sleep 0.3; touch leaked) >/dev/null 2>&1 & exit 0";
    assert!(super::verify_passes_with_timeout(
        dir.path(),
        command,
        Some(std::time::Duration::from_secs(5)),
    ));
    std::thread::sleep(std::time::Duration::from_millis(450));
    assert!(
        !dir.path().join("leaked").exists(),
        "verification returned success but a descendant kept modifying the tree"
    );
}

fn git(root: &std::path::Path, args: &[&str]) {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn repository() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    git(dir.path(), &["init", "-q"]);
    git(dir.path(), &["config", "user.name", "test"]);
    git(
        dir.path(),
        &["config", "user.email", "test@example.invalid"],
    );
    std::fs::write(dir.path().join("source.rs"), "original\n").unwrap();
    std::fs::write(dir.path().join(".gitignore"), "ignored-output/\n.env\n").unwrap();
    git(dir.path(), &["add", "-A"]);
    git(dir.path(), &["commit", "-qm", "base"]);
    dir
}

#[test]
fn successful_verifier_source_mutations_cannot_pass() {
    for command in [
        "printf changed > source.rs",
        "printf new > added.rs",
        "rm source.rs",
        "printf changed > source.rs; git add source.rs",
    ] {
        let dir = repository();
        std::fs::write(dir.path().join("source.rs"), "candidate\n").unwrap();
        assert!(
            !super::verify_passes(dir.path(), command),
            "accepted mutating verifier: {command}"
        );
    }
}

#[test]
fn stable_verification_preserves_the_users_index_and_ignores_regenerable_output() {
    let dir = repository();
    std::fs::write(dir.path().join("source.rs"), "staged\n").unwrap();
    git(dir.path(), &["add", "source.rs"]);
    std::fs::write(dir.path().join("source.rs"), "candidate\n").unwrap();
    let index = dir.path().join(".git/index");
    let before = std::fs::read(&index).unwrap();
    assert!(super::verify_passes(
        dir.path(),
        "test \"$(cat source.rs)\" = candidate; mkdir -p ignored-output __pycache__; printf cache > ignored-output/cache; printf bytecode > __pycache__/source.pyc"
    ));
    assert_eq!(
        std::fs::read(index).unwrap(),
        before,
        "verification changed staged user work"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("source.rs")).unwrap(),
        "candidate\n"
    );
}

#[test]
fn staged_ignored_source_is_included_in_the_verification_seal() {
    let dir = repository();
    std::fs::write(dir.path().join(".env"), "before\n").unwrap();
    git(dir.path(), &["add", "-f", ".env"]);
    let index = std::fs::read(dir.path().join(".git/index")).unwrap();
    assert!(!super::verify_passes(dir.path(), "printf after > .env"));
    assert_eq!(std::fs::read(dir.path().join(".git/index")).unwrap(), index);
    assert_eq!(
        std::fs::read_to_string(dir.path().join(".env")).unwrap(),
        "after"
    );
}

#[test]
fn stable_verification_supports_split_indexes_and_plain_directories() {
    let dir = repository();
    git(dir.path(), &["update-index", "--split-index"]);
    assert!(super::verify_passes(dir.path(), "true"));
    let plain = tempfile::tempdir().unwrap();
    assert!(super::verify_passes(plain.path(), "true"));
    assert!(!super::verify_passes(plain.path(), "false"));
}

#[test]
fn failed_snapshot_does_not_run_verification_or_modify_the_index() {
    let dir = repository();
    std::fs::write(
        dir.path().join(".gitattributes"),
        "source.rs filter=reject\n",
    )
    .unwrap();
    git(dir.path(), &["config", "filter.reject.clean", "false"]);
    git(dir.path(), &["config", "filter.reject.required", "true"]);
    std::fs::write(dir.path().join("source.rs"), "candidate\n").unwrap();
    let index = std::fs::read(dir.path().join(".git/index")).unwrap();
    assert!(!super::verify_passes(dir.path(), "touch verifier-ran"));
    assert!(!dir.path().join("verifier-ran").exists());
    assert_eq!(std::fs::read(dir.path().join(".git/index")).unwrap(), index);
}

fn slow_filter(dir: &std::path::Path, signal: &std::path::Path) {
    let signal = format!("'{}'", signal.to_str().unwrap().replace('\'', "'\\''"));
    std::fs::write(dir.join(".gitattributes"), "source.rs filter=slow\n").unwrap();
    git(
        dir,
        &[
            "config",
            "filter.slow.clean",
            &format!("printf '%s' \"$GIT_INDEX_FILE\" > {signal}; sleep 30; cat"),
        ],
    );
    git(dir, &["config", "filter.slow.required", "true"]);
    std::fs::write(dir.join("source.rs"), "candidate\n").unwrap();
}

#[tokio::test]
async fn cancellation_stops_snapshot_filters_and_removes_the_temporary_index() {
    let dir = repository();
    let signals = tempfile::tempdir().unwrap();
    let signal = signals.path().join("snapshot-started");
    slow_filter(dir.path(), &signal);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let worker = super::verify_passes_async(dir.path(), "touch verifier-ran", Some(&cancellation));
    tokio::pin!(worker);
    let ready = async {
        loop {
            if let Ok(index) = std::fs::read_to_string(&signal)
                && !index.is_empty()
            {
                return std::path::PathBuf::from(index);
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    };
    let index = tokio::select! {
        result = &mut worker => panic!("verification returned before filter started: {result}"),
        index = tokio::time::timeout(std::time::Duration::from_secs(5), ready) => index.unwrap(),
    };
    cancellation.cancel();
    assert!(
        !tokio::time::timeout(std::time::Duration::from_secs(3), worker)
            .await
            .unwrap()
    );
    assert!(!dir.path().join("verifier-ran").exists());
    assert!(
        !index.parent().unwrap().exists(),
        "private index leaked after cancellation"
    );
}

#[test]
fn explicit_verification_deadline_also_bounds_snapshot_filters() {
    let dir = repository();
    let signals = tempfile::tempdir().unwrap();
    slow_filter(dir.path(), &signals.path().join("started"));
    let started = std::time::Instant::now();
    assert!(!super::verify_passes_with_timeout(
        dir.path(),
        "touch verifier-ran",
        Some(std::time::Duration::from_millis(250))
    ));
    assert!(started.elapsed() < std::time::Duration::from_secs(3));
    assert!(!dir.path().join("verifier-ran").exists());
}

#[cfg(unix)]
#[test]
fn completed_snapshot_filter_cannot_leave_a_background_writer() {
    let dir = repository();
    let signals = tempfile::tempdir().unwrap();
    let release = signals.path().join("release");
    let leaked = signals.path().join("leaked");
    let quoted =
        |path: &std::path::Path| format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"));
    std::fs::write(
        dir.path().join(".gitattributes"),
        "source.rs filter=background\n",
    )
    .unwrap();
    git(
        dir.path(),
        &[
            "config",
            "filter.background.clean",
            &format!(
                "(while [ ! -e {} ]; do sleep 0.01; done; touch {}) >/dev/null 2>&1 & cat",
                quoted(&release),
                quoted(&leaked)
            ),
        ],
    );
    git(
        dir.path(),
        &["config", "filter.background.required", "true"],
    );
    std::fs::write(dir.path().join("source.rs"), "candidate\n").unwrap();
    assert!(super::verify_passes_with_timeout(
        dir.path(),
        "true",
        Some(std::time::Duration::from_secs(5))
    ));
    std::fs::write(release, "go").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(150));
    assert!(
        !leaked.exists(),
        "Git snapshot returned with a live filter descendant"
    );
}
