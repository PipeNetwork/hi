#![cfg(unix)]

use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;

use super::*;

fn git(root: &Path, args: &[&str]) {
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

fn repository(root: &Path) {
    std::fs::create_dir_all(root).unwrap();
    git(root, &["init", "-q"]);
    git(root, &["config", "user.name", "test"]);
    git(root, &["config", "user.email", "test@example.invalid"]);
    std::fs::write(root.join("source.txt"), "base\n").unwrap();
    std::fs::write(root.join(".gitattributes"), "*.txt filter=hi_test\n").unwrap();
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "base"]);
}

fn wait_for(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !std::fs::metadata(path).is_ok_and(|meta| meta.len() > 0) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        std::fs::metadata(path).is_ok_and(|meta| meta.len() > 0),
        "process did not signal readiness"
    );
}

fn index_path(worktree: &Path) -> std::path::PathBuf {
    let output = Command::new("git")
        .current_dir(worktree)
        .args(["rev-parse", "--path-format=absolute", "--git-path", "index"])
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim_end().into()
}

#[test]
fn cancellation_interrupts_merge_staging_filter() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path().join("root");
    let candidate = fixture.path().join("candidate");
    repository(&root);
    add_worktree(&root, &candidate, "HEAD").unwrap();
    let real_index = index_path(&candidate);
    let before_index = std::fs::read(&real_index).unwrap();
    let ready = fixture.path().join("ready");
    let release = fixture.path().join("release");
    let script = fixture.path().join("filter.sh");
    std::fs::write(
        &script,
        format!(
            "printf '%s' \"$GIT_INDEX_FILE\" > {}\nwhile ! test -e {}; do sleep 0.02; done\ncat\n",
            crate::edit::sh_quote(&ready.to_string_lossy()),
            crate::edit::sh_quote(&release.to_string_lossy())
        ),
    )
    .unwrap();
    git(
        &root,
        &[
            "config",
            "filter.hi_test.clean",
            &format!("sh {}", crate::edit::sh_quote(&script.to_string_lossy())),
        ],
    );
    std::fs::write(candidate.join("source.txt"), "candidate\n").unwrap();
    let cancellation = CancellationToken::new();
    let worker_cancel = cancellation.clone();
    let worker_root = root.clone();
    let worker_candidate = candidate.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        tx.send(apply_changes_impl_with_timeout_and_cancel(
            &worker_candidate,
            "HEAD",
            &worker_root,
            None,
            Some(&worker_cancel),
        ))
        .unwrap();
    });
    wait_for(&ready);
    let private_index = std::path::PathBuf::from(std::fs::read_to_string(&ready).unwrap());
    cancellation.cancel();
    let observed = rx.recv_timeout(Duration::from_millis(500));
    // Release the baseline's stuck filter before asserting so even the
    // failing regression leaves no blocked worker/process behind.
    std::fs::write(&release, "release").unwrap();
    let settled_promptly = observed.is_ok();
    let result = observed.unwrap_or_else(|_| rx.recv_timeout(Duration::from_secs(5)).unwrap());
    worker.join().unwrap();
    assert!(result.is_err(), "cancelled merge must not succeed");
    assert_eq!(
        std::fs::read_to_string(root.join("source.txt")).unwrap(),
        "base\n"
    );
    assert_eq!(std::fs::read(&real_index).unwrap(), before_index);
    assert!(!std::path::PathBuf::from(format!("{}.lock", real_index.display())).exists());
    assert!(
        !private_index.parent().unwrap().exists(),
        "cancelled merge leaked its private index/lock"
    );
    git(&root, &["config", "filter.hi_test.clean", "cat"]);
    assert!(
        apply_changes_to(&candidate, "HEAD", &root).unwrap(),
        "cancelled candidate must be retryable"
    );
    cleanup(&root, &[candidate]);
    assert!(
        settled_promptly,
        "cancellation waited for the staging filter to finish"
    );
}

#[tokio::test]
async fn async_inspection_cancellation_cleans_its_private_index() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path().join("root");
    repository(&root);
    let real_index = index_path(&root);
    let before_index = std::fs::read(&real_index).unwrap();
    let ready = fixture.path().join("ready");
    let release = fixture.path().join("release");
    let filter = format!(
        "printf '%s' \"$GIT_INDEX_FILE\" > {}; while ! test -e {}; do sleep 0.02; done; cat",
        crate::edit::sh_quote(&ready.to_string_lossy()),
        crate::edit::sh_quote(&release.to_string_lossy())
    );
    git(&root, &["config", "filter.hi_test.clean", &filter]);
    std::fs::write(root.join("source.txt"), "candidate\n").unwrap();
    let cancellation = CancellationToken::new();
    let worker_root = root.clone();
    let worker_cancel = cancellation.clone();
    let mut task = tokio::spawn(async move {
        changed_files_async_with_cancel(&worker_root, "HEAD", Some(&worker_cancel)).await
    });
    for _ in 0..250 {
        if std::fs::metadata(&ready).is_ok_and(|meta| meta.len() > 0) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let private_index = std::path::PathBuf::from(std::fs::read_to_string(&ready).unwrap());
    cancellation.cancel();
    let observed = tokio::time::timeout(Duration::from_secs(2), &mut task).await;
    std::fs::write(&release, "release").unwrap();
    let timely = observed.is_ok();
    let result = match observed {
        Ok(result) => result,
        Err(_) => task.await,
    }
    .unwrap();
    assert!(timely, "inspection did not honor cancellation");
    assert!(result.is_err());
    assert_eq!(std::fs::read(&real_index).unwrap(), before_index);
    assert!(!private_index.parent().unwrap().exists());
    git(&root, &["config", "filter.hi_test.clean", "cat"]);
    assert_eq!(
        changed_files_async(&root, "HEAD").await.unwrap(),
        ["source.txt"]
    );
    assert_eq!(std::fs::read(&real_index).unwrap(), before_index);
}

#[test]
fn cancellation_interrupts_large_patch_delivery() {
    let fixture = tempfile::tempdir().unwrap();
    let ready = fixture.path().join("ready");
    let release = fixture.path().join("release");
    let mut command = Command::new("sh");
    command.arg("-c").arg(format!(
        "printf ready > {}; while ! test -e {}; do sleep 0.02; done; cat >/dev/null",
        crate::edit::sh_quote(&ready.to_string_lossy()),
        crate::edit::sh_quote(&release.to_string_lossy())
    ));
    configure_private_process_group(&mut command);
    let cancellation = CancellationToken::new();
    let worker_cancel = cancellation.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let patch = vec![b'x'; 8 * 1024 * 1024];
        tx.send(run_apply_command(
            &mut command,
            &patch,
            None,
            Some(&worker_cancel),
        ))
        .unwrap();
    });
    wait_for(&ready);
    cancellation.cancel();
    let observed = rx.recv_timeout(Duration::from_millis(500));
    std::fs::write(&release, "release").unwrap();
    let settled_promptly = observed.is_ok();
    let result = observed.unwrap_or_else(|_| rx.recv_timeout(Duration::from_secs(5)).unwrap());
    worker.join().unwrap();
    assert!(
        settled_promptly,
        "cancellation blocked behind the patch stdin pipe"
    );
    assert!(result.is_err(), "cancelled patch delivery must not succeed");
}

#[test]
fn large_patch_and_large_stderr_do_not_deadlock() {
    let mut command = Command::new("sh");
    command.args(["-c", "head -c 1048576 /dev/zero >&2; cat"]);
    let patch = vec![b'x'; 1024 * 1024];
    let output =
        run_apply_command(&mut command, &patch, Some(Duration::from_secs(5)), None).unwrap();
    assert!(output.status.success());
    assert_eq!(output.stdout, patch);
    assert_eq!(output.stderr.len(), 1024 * 1024);
}

#[test]
fn completed_merge_command_cleans_background_writers() {
    let fixture = tempfile::tempdir().unwrap();
    let mut process = Command::new("sh");
    process.current_dir(fixture.path()).args([
        "-c",
        "(while ! test -f release; do sleep 0.02; done; touch leaked) & printf finished",
    ]);
    let output = command::run(
        &mut process,
        None,
        command::Budget::new(Some(Duration::from_secs(2))),
        None,
    )
    .unwrap();
    assert!(output.status.success());
    assert_eq!(output.stdout, b"finished");
    std::fs::write(fixture.path().join("release"), "release").unwrap();
    std::thread::sleep(Duration::from_millis(100));
    assert!(!fixture.path().join("leaked").exists());
}

#[test]
fn merge_uses_machine_patch_despite_display_diff_configuration() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path().join("root");
    let candidate = fixture.path().join("candidate");
    repository(&root);
    add_worktree(&root, &candidate, "HEAD").unwrap();
    let display = fixture.path().join("display.sh");
    std::fs::write(&display, "printf 'custom diff output\\n'\n").unwrap();
    git(
        &root,
        &[
            "config",
            "diff.external",
            &format!("sh {}", crate::edit::sh_quote(&display.to_string_lossy())),
        ],
    );
    git(&root, &["config", "color.ui", "always"]);
    git(&root, &["config", "diff.noprefix", "true"]);
    std::fs::write(candidate.join("source.txt"), "candidate\n").unwrap();
    assert!(apply_changes_to(&candidate, "HEAD", &root).unwrap());
    assert_eq!(
        std::fs::read_to_string(root.join("source.txt")).unwrap(),
        "candidate\n"
    );
    cleanup(&root, &[candidate]);
}

#[test]
fn explicit_merge_deadline_covers_the_lock_wait() {
    let held = MERGE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result =
            acquire_merge_lock(None, command::Budget::new(Some(Duration::from_millis(25))))
                .map(|_| ());
        tx.send(result).unwrap();
    });
    let observed = rx.recv_timeout(Duration::from_millis(500));
    drop(held);
    let timely = observed.is_ok();
    let result = observed.unwrap_or_else(|_| rx.recv_timeout(Duration::from_secs(5)).unwrap());
    worker.join().unwrap();
    assert!(timely, "deadline did not interrupt the merge lock wait");
    assert!(result.unwrap_err().to_string().contains("timed out"));
}
