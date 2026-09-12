use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

use super::JournalMode;

const PROBE_PATH: &str = "HI_SQLITE_JOURNAL_LOCK_PROBE_PATH";

// POSIX locks are process-scoped, so a same-process try-lock is not a probe.
// Every assertion uses a fresh process against the actual SQLite-owned inode.
fn assert_peer_is_locked(path: &Path) {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "lock_tests::lock_probe_child", "--nocapture"])
        .env(PROBE_PATH, path)
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("test result: ok. 1 passed;"),
        "lock probe did not run exactly one test: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        output.status.success(),
        "SQLite lock was lost for {}: {}{}",
        path.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn lock_probe_child() {
    let Some(path) = std::env::var_os(PROBE_PATH) else {
        return;
    };
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut lock: libc::flock = unsafe { std::mem::zeroed() };
    lock.l_type = libc::F_WRLCK as _;
    lock.l_whence = libc::SEEK_SET as _;
    lock.l_start = 0;
    lock.l_len = 0;
    let result = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETLK, &lock) };
    assert_eq!(
        result, -1,
        "peer acquired a lock still owned by the SQLite reader"
    );
    let error = std::io::Error::last_os_error();
    assert!(
        matches!(error.raw_os_error(), Some(libc::EACCES | libc::EAGAIN)),
        "unexpected lock denial: {error}"
    );
}

fn active_reader(path: &Path) -> rusqlite::Connection {
    let connection = JournalMode::Wal.open(path).unwrap();
    connection
        .execute_batch("CREATE TABLE t(value INTEGER); INSERT INTO t VALUES (1); BEGIN;")
        .unwrap();
    let count: i64 = connection
        .query_row("SELECT count(*) FROM t", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 1);
    connection
}

fn assert_mode_600(path: &Path) {
    assert_eq!(
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
        0o600
    );
}

fn check_reopen(tighten: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("events.sqlite3");
    let reader = active_reader(&path);
    let shm = super::sidecar_path(&path, "-shm");
    assert_peer_is_locked(&path);
    assert_peer_is_locked(&shm);
    if tighten {
        for file in [&path, &shm] {
            std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o666)).unwrap();
        }
    }
    let reopened = JournalMode::Wal.open(&path).unwrap();
    assert_mode_600(&path);
    assert_mode_600(&shm);
    assert_peer_is_locked(&path);
    assert_peer_is_locked(&shm);
    // SQLite's own second connection close must retain the first connection's
    // read transaction too; permission adjustment cannot corrupt its bookkeeping.
    drop(reopened);
    assert_peer_is_locked(&path);
    assert_peer_is_locked(&shm);
    let count: i64 = reader
        .query_row("SELECT count(*) FROM t", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn reopening_owner_only_database_preserves_live_sqlite_locks() {
    check_reopen(false);
}

#[test]
fn tightening_database_and_wal_index_preserves_live_sqlite_locks() {
    check_reopen(true);
}

#[test]
fn portable_permission_child_preserves_live_sqlite_locks() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("events.sqlite3");
    let _reader = active_reader(&path);
    let shm = super::sidecar_path(&path, "-shm");
    assert_peer_is_locked(&path);
    assert_peer_is_locked(&shm);
    for file in [&path, &shm] {
        std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o666)).unwrap();
        super::permissions::tighten_in_child_for_test(file).unwrap();
        assert_mode_600(file);
        assert_peer_is_locked(&path);
        assert_peer_is_locked(&shm);
    }
}

#[test]
fn portable_permission_child_refuses_replacement_inodes_and_links() {
    use std::os::unix::fs::symlink;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("state.sqlite3");
    let target = directory.path().join("other.sqlite3");
    std::fs::write(&path, "first").unwrap();
    std::fs::write(&target, "second").unwrap();
    std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o666)).unwrap();
    let expected = std::fs::metadata(&path).unwrap();
    std::fs::rename(&target, &path).unwrap();
    assert!(super::permissions::tighten_in_child_with_metadata_for_test(&path, &expected).is_err());
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o666
    );
    symlink(&path, &target).unwrap();
    assert!(super::permissions::tighten_in_child_for_test(&target).is_err());
    std::fs::remove_file(&target).unwrap();
    std::fs::hard_link(&path, &target).unwrap();
    assert!(super::permissions::tighten_in_child_for_test(&path).is_err());
}

#[test]
fn permission_paths_refuse_replacements_and_preserve_optional_disappearance() {
    use std::os::unix::ffi::OsStrExt;
    type Tighten = fn(&Path, &std::fs::Metadata) -> anyhow::Result<()>;
    let operations: [Tighten; 2] = [
        super::permissions::tighten_with_metadata_for_test,
        super::permissions::tighten_in_child_with_metadata_for_test,
    ];
    for tighten in operations {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state-shm");
        let replacement = directory.path().join("other");
        std::fs::write(&path, "original").unwrap();
        let expected = std::fs::metadata(&path).unwrap();
        std::fs::write(&replacement, "replacement").unwrap();
        std::fs::set_permissions(&replacement, std::fs::Permissions::from_mode(0o666)).unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        assert!(tighten(&path, &expected).is_err());
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o666
        );
        std::fs::remove_file(&path).unwrap();
        let missing = tighten(&path, &expected).unwrap_err();
        assert_eq!(
            missing.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::NotFound
        );
        let encoded = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(encoded.as_ptr(), 0o600) }, 0);
        let started = std::time::Instant::now();
        assert!(tighten(&path, &expected).is_err());
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "replacement FIFO blocked permission validation"
        );
    }
}
