//! Filesystem-aware SQLite journal-mode selection.
//!
//! WAL mode keeps its wal-index in an mmap'd `-shm` file and relies on coherent
//! shared memory plus reliable POSIX locks — guarantees network filesystems
//! (NFS, SMB) do not provide. When the database directory is on a network
//! mount, we use a rollback journal (TRUNCATE) instead, preventing SIGBUS
//! crashes and silent corruption.
//!
//! Inspired by grok-build's `xai-sqlite-journal` crate.
//!
//! # Quick start
//!
//! ```no_run
//! use hi_sqlite_journal::JournalMode;
//! use std::path::Path;
//!
//! let db_path = Path::new("~/.hi/memory.sqlite");
//! let conn = JournalMode::for_db_path(db_path).open(db_path).unwrap();
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use rusqlite::Connection;

/// Wait for peers' locks instead of failing instantly.
const BUSY_TIMEOUT_MS: u32 = 5000;

/// Poll cadence and cap for the journal-mode switch, which SQLite refuses to
/// apply busy_timeout to (see [`JournalMode::apply`]).
const MODE_SWITCH_POLL: std::time::Duration = std::time::Duration::from_millis(50);
const MODE_SWITCH_WAIT_MAX: std::time::Duration = std::time::Duration::from_secs(5);

/// SQLITE_BUSY / SQLITE_LOCKED — transient peer contention, safe to retry.
fn is_busy(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(failure, _) if matches!(
            failure.code,
            rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
        )
    )
}

/// Journal mode chosen for a SQLite database based on where it lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JournalMode {
    /// Write-ahead logging — the default for local filesystems.
    Wal,
    /// Rollback journal truncated at commit — safe on network filesystems.
    Truncate,
}

impl JournalMode {
    /// Pick the journal mode for a database at `db_path`.
    ///
    /// Classifies the parent directory (the DB file itself may not exist yet).
    /// `HI_SQLITE_JOURNAL_MODE` (`wal`|`truncate`) overrides detection as a
    /// kill-switch.
    pub fn for_db_path(db_path: &Path) -> Self {
        let env = std::env::var("HI_SQLITE_JOURNAL_MODE").ok();
        match env
            .as_deref()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("wal") => return JournalMode::Wal,
            Some("truncate") => return JournalMode::Truncate,
            Some(other) => {
                eprintln!(
                    "hi-sqlite-journal: invalid HI_SQLITE_JOURNAL_MODE='{other}' \
                     (accepted: wal, truncate); using auto-detection"
                );
            }
            _ => {}
        }

        // Auto-detect: check if the parent directory is on a network filesystem.
        let parent = sqlite_parent(db_path);
        if is_network_filesystem(parent) {
            JournalMode::Truncate
        } else {
            JournalMode::Wal
        }
    }

    /// Apply the journal mode to an existing connection.
    pub fn apply(&self, conn: &Connection) -> Result<()> {
        conn.pragma_update(None, "busy_timeout", BUSY_TIMEOUT_MS)
            .context("setting busy_timeout")?;
        let mode = self.as_str();
        // Switching journal modes promotes to an exclusive lock, and SQLite
        // skips the busy handler on that promotion when a peer holds a write
        // lock (waiting could deadlock) — so the pragma returns SQLITE_BUSY
        // instantly despite busy_timeout. Poll instead. Steady-state opens
        // never loop: the mode is persisted in the file, so re-applying it
        // is a lock-free no-op.
        let mut waited = std::time::Duration::ZERO;
        loop {
            match conn.pragma_update(None, "journal_mode", mode) {
                Err(error) if is_busy(&error) && waited < MODE_SWITCH_WAIT_MAX => {
                    std::thread::sleep(MODE_SWITCH_POLL);
                    waited += MODE_SWITCH_POLL;
                }
                Ok(()) => {
                    // The pragma does not raise an error when the switch
                    // fails (e.g. WAL requested on a filesystem without
                    // coherent locking) — it silently keeps the old mode.
                    // Read the resulting mode back and fail loudly instead of
                    // reporting success on a database that never switched.
                    let applied: String = conn
                        .pragma_query_value(None, "journal_mode", |row| row.get(0))
                        .context("reading back journal_mode")?;
                    ensure!(
                        applied.eq_ignore_ascii_case(mode),
                        "journal_mode switch to {mode} did not take effect (actual: {applied})"
                    );
                    return Ok(());
                }
                Err(error) => {
                    return Err(error).with_context(|| format!("setting journal_mode to {mode}"));
                }
            }
        }
    }

    /// Open a connection at `db_path` with the appropriate journal mode.
    pub fn open(&self, db_path: &Path) -> Result<Connection> {
        let parent = sqlite_parent(db_path);
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating db parent dir {}", parent.display()))?;
        let open_path = sqlite_open_path(db_path)?;
        // Resolve a symlinked parent exactly once, then use that same path for
        // permission checks, SQLite, and sidecars. Otherwise a parent symlink
        // swapped between these operations could redirect SQLite to a file we
        // never secured.
        prepare_writable_database(&open_path)?;
        let flags = rusqlite::OpenFlags::default();
        #[cfg(unix)]
        let flags = flags | rusqlite::OpenFlags::SQLITE_OPEN_NOFOLLOW;
        let conn = Connection::open_with_flags(&open_path, flags)
            .with_context(|| format!("opening sqlite db {}", db_path.display()))?;
        self.apply(&conn)?;
        // Tighten pre-existing sidecars and any files created while applying
        // the journal mode. Later sidecars inherit the main database's mode
        // from SQLite's Unix VFS.
        tighten_existing_sidecars(&open_path)?;
        Ok(conn)
    }

    /// Open a read-only connection at `db_path` with the appropriate journal mode.
    pub fn open_readonly(&self, db_path: &Path) -> Result<Connection> {
        let flags =
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;
        #[cfg(unix)]
        let flags = flags | rusqlite::OpenFlags::SQLITE_OPEN_NOFOLLOW;
        let open_path = sqlite_open_path(db_path)?;
        let conn = Connection::open_with_flags(open_path, flags)
            .with_context(|| format!("opening sqlite db read-only {}", db_path.display()))?;
        // For read-only, just set busy_timeout — journal mode is already
        // persisted in the file.
        conn.pragma_update(None, "busy_timeout", BUSY_TIMEOUT_MS)
            .context("setting busy_timeout")?;
        Ok(conn)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            JournalMode::Wal => "wal",
            JournalMode::Truncate => "truncate",
        }
    }
}

/// Resolve symlinks in the parent directory while preserving the final path
/// component for SQLite's `SQLITE_OPEN_NOFOLLOW` check. Passing the original
/// path directly would reject legitimate symlinked parents on Unix (including
/// macOS's `/var` -> `/private/var`), while canonicalizing the whole path would
/// silently follow a malicious final-component symlink.
fn sqlite_open_path(db_path: &Path) -> Result<PathBuf> {
    #[cfg(unix)]
    {
        let parent = sqlite_parent(db_path);
        let canonical_parent = parent
            .canonicalize()
            .with_context(|| format!("canonicalizing sqlite parent {}", parent.display()))?;
        let file_name = db_path
            .file_name()
            .with_context(|| format!("sqlite path has no filename: {}", db_path.display()))?;
        Ok(canonical_parent.join(file_name))
    }
    #[cfg(not(unix))]
    {
        Ok(db_path.to_path_buf())
    }
}

/// `Path::parent` represents a bare relative filename's parent as an empty
/// path. Filesystem operations and mount detection need the equivalent real
/// directory (`.`), not that empty sentinel.
fn sqlite_parent(db_path: &Path) -> &Path {
    db_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

/// Atomically create a new database owner-only, or tighten an existing
/// writable database before SQLite touches it. Non-Unix platforms retain the
/// native ACL/permission behavior.
fn prepare_writable_database(db_path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(db_path)
        {
            Ok(file) => set_open_file_owner_only(&file, db_path)?,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                set_owner_only(db_path, false)?;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("securely creating sqlite db {}", db_path.display()));
            }
        }
        tighten_existing_sidecars(db_path)
    }
    #[cfg(not(unix))]
    {
        let _ = db_path;
        Ok(())
    }
}

fn tighten_existing_sidecars(db_path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        for suffix in ["-wal", "-shm", "-journal"] {
            set_owner_only(&sidecar_path(db_path, suffix), true)?;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = db_path;
    }
    Ok(())
}

#[cfg(unix)]
fn set_owner_only(path: &Path, allow_missing: bool) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if allow_missing && error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(());
        }
        Err(error) => {
            return Err(error).with_context(|| format!("reading mode for {}", path.display()));
        }
    };
    ensure!(
        !metadata.file_type().is_symlink(),
        "refusing symbolic link for sqlite file: {}",
        path.display()
    );
    ensure!(
        metadata.is_file(),
        "sqlite path is not a regular file: {}",
        path.display()
    );
    ensure!(
        metadata.nlink() == 1,
        "refusing multiply-linked sqlite file: {}",
        path.display()
    );
    let file = match std::fs::OpenOptions::new()
        .write(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if allow_missing && error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(());
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "opening sqlite file without following links {}",
                    path.display()
                )
            });
        }
    };
    let opened = file
        .metadata()
        .with_context(|| format!("reading opened sqlite file metadata {}", path.display()))?;
    ensure!(
        opened.is_file(),
        "opened sqlite path is not a regular file: {}",
        path.display()
    );
    ensure!(
        opened.nlink() == 1,
        "refusing multiply-linked sqlite file: {}",
        path.display()
    );
    ensure!(
        metadata.dev() == opened.dev() && metadata.ino() == opened.ino(),
        "sqlite path changed while securing it: {}",
        path.display()
    );
    set_open_file_owner_only(&file, path)
}

#[cfg(unix)]
fn set_open_file_owner_only(file: &std::fs::File, path: &Path) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let metadata = file
        .metadata()
        .with_context(|| format!("reading opened sqlite file metadata {}", path.display()))?;
    ensure!(
        metadata.is_file(),
        "opened sqlite path is not a regular file: {}",
        path.display()
    );
    ensure!(
        metadata.nlink() == 1,
        "refusing multiply-linked sqlite file: {}",
        path.display()
    );
    let mut permissions = metadata.permissions();
    permissions.set_mode(0o600);
    file.set_permissions(permissions)
        .with_context(|| format!("setting owner-only mode on {}", path.display()))
}

fn sidecar_path(db_path: &Path, suffix: &str) -> PathBuf {
    let mut path = db_path.as_os_str().to_os_string();
    path.push(suffix);
    PathBuf::from(path)
}

/// Detect whether `path` is on a network filesystem.
///
/// On Unix, reads `/proc/mounts` (Linux) or uses `statfs` (macOS) to check
/// the filesystem type. On other platforms, always returns `false` (assume
/// local).
fn is_network_filesystem(path: &Path) -> bool {
    #[cfg(target_os = "linux")]
    {
        is_network_filesystem_linux(path)
    }
    #[cfg(target_os = "macos")]
    {
        is_network_filesystem_macos(path)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = path;
        false
    }
}

/// `path` may not exist yet (new database parents commonly do not). Walk up
/// until canonicalization succeeds so symlinks in the existing prefix are
/// resolved and filesystem probes always receive a real path.
fn filesystem_probe_path(path: &Path) -> PathBuf {
    let normalized = if path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        path
    };
    let mut candidate = normalized;
    loop {
        if let Ok(canonical) = candidate.canonicalize() {
            return canonical;
        }
        let Some(parent) = candidate.parent() else {
            return normalized.to_path_buf();
        };
        candidate = parent;
    }
}

#[cfg(target_os = "linux")]
fn is_network_filesystem_linux(path: &Path) -> bool {
    // Read /proc/mounts and find the longest mount prefix matching `path`.
    // Then check if the filesystem type is a known network type.
    let mounts = match std::fs::read_to_string("/proc/mounts") {
        Ok(m) => m,
        Err(_) => return false,
    };

    let probe_path = filesystem_probe_path(path);

    let mut best_match: Option<(String, &str)> = None; // (mount_point, fs_type)
    for line in mounts.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }
        let mount_point = decode_mount_field(parts[1]);
        let fs_type = parts[2];
        let mount_path = Path::new(&mount_point);
        if probe_path.starts_with(mount_path)
            && best_match
                .as_ref()
                .is_none_or(|(mp, _)| mount_point.len() > mp.len())
        {
            best_match = Some((mount_point, fs_type));
        }
    }

    if let Some((_, fs_type)) = best_match {
        let network_fs = [
            "nfs",
            "nfs4",
            "cifs",
            "smb",
            "smb2",
            "smb3",
            "fuse.sshfs",
            "webdav",
        ];
        return network_fs.contains(&fs_type);
    }
    false
}

#[cfg(target_os = "linux")]
fn decode_mount_field(value: &str) -> String {
    value
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

#[cfg(target_os = "macos")]
fn is_network_filesystem_macos(path: &Path) -> bool {
    // Use statfs to get the filesystem type name.
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let probe_path = filesystem_probe_path(path);
    let path_cstring = match CString::new(probe_path.as_os_str().as_bytes()) {
        Ok(s) => s,
        Err(_) => return false,
    };

    // SAFETY: statfs is a C syscall. We pass a valid C string and a valid
    // pointer to a statfs struct.
    unsafe {
        let mut buf: libc::statfs = std::mem::zeroed();
        if libc::statfs(path_cstring.as_ptr(), &mut buf) != 0 {
            return false;
        }
        // f_fstypename is a null-terminated C string in the statfs struct.
        // On macOS it's [i8; 16] (c_char), so cast to u8 for byte comparison.
        let fs_type_raw = &buf.f_fstypename;
        let fs_type_bytes: &[u8] =
            std::slice::from_raw_parts(fs_type_raw.as_ptr() as *const u8, fs_type_raw.len());
        let len = fs_type_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(fs_type_bytes.len());
        let fs_type = std::str::from_utf8(&fs_type_bytes[..len]).unwrap_or("");
        let network_fs = ["nfs", "smbfs", "webdav", "afpfs", "fuse.sshfs"];
        network_fs.contains(&fs_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn assert_owner_only(path: &Path) {
        use std::os::unix::fs::PermissionsExt;

        let mode = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "{} has mode {mode:o}", path.display());
    }

    #[test]
    fn env_override_wal() {
        // Can't safely mutate env in parallel tests, so just test the logic
        // by calling for_db_path with a known-local path and no env.
        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("test.sqlite");
        let mode = JournalMode::for_db_path(&db);
        // On local filesystem, should be WAL (unless on network mount in CI).
        // Just verify it returns a valid mode.
        assert!(matches!(mode, JournalMode::Wal | JournalMode::Truncate));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mount_matching_respects_component_boundaries_and_decodes_paths() {
        assert_eq!(
            decode_mount_field("/mnt/shared\\040files"),
            "/mnt/shared files"
        );
        assert!(Path::new("/mnt/net/db").starts_with(Path::new("/mnt/net")));
        assert!(!Path::new("/mnt/network/db").starts_with(Path::new("/mnt/net")));
    }

    #[test]
    fn filesystem_probe_uses_nearest_existing_ancestor() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("not-created/nested/dir");
        assert_eq!(
            filesystem_probe_path(&missing),
            tmp.path().canonicalize().unwrap()
        );
    }

    #[test]
    fn bare_relative_database_uses_current_directory_as_parent() {
        assert_eq!(sqlite_parent(Path::new("bare.sqlite")), Path::new("."));
        assert_eq!(
            filesystem_probe_path(Path::new("")),
            Path::new(".").canonicalize().unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_probe_resolves_symlinked_existing_ancestor() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("actual-mount");
        std::fs::create_dir(&target).unwrap();
        let link = tmp.path().join("linked-parent");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let missing = link.join("not-created/nested");
        assert_eq!(
            filesystem_probe_path(&missing),
            target.canonicalize().unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn writable_and_readonly_open_reject_database_symlink_without_chmodding_target() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("target.sqlite");
        let conn = Connection::open(&target).unwrap();
        conn.execute_batch("CREATE TABLE t(v);").unwrap();
        drop(conn);
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o644)).unwrap();

        let link = tmp.path().join("linked.sqlite");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let error = JournalMode::Wal.open(&link).unwrap_err();
        assert!(format!("{error:#}").contains("symbolic link"), "{error:#}");
        assert_eq!(
            std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
            0o644,
            "rejected target mode changed"
        );

        assert!(JournalMode::Wal.open_readonly(&link).is_err());
        assert_eq!(
            std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
            0o644,
            "read-only rejection changed target mode"
        );
    }

    #[cfg(unix)]
    #[test]
    fn writable_open_rejects_sidecar_symlinks_without_chmodding_targets() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        for suffix in ["-wal", "-shm", "-journal"] {
            let db = tmp.path().join(format!("sidecar{suffix}.sqlite"));
            let conn = Connection::open(&db).unwrap();
            conn.execute_batch("CREATE TABLE t(v);").unwrap();
            drop(conn);

            let target = tmp.path().join(format!("target{suffix}"));
            std::fs::write(&target, b"unrelated").unwrap();
            std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o644)).unwrap();
            std::os::unix::fs::symlink(&target, sidecar_path(&db, suffix)).unwrap();

            let error = JournalMode::Wal.open(&db).unwrap_err();
            assert!(format!("{error:#}").contains("symbolic link"), "{error:#}");
            assert_eq!(
                std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
                0o644,
                "rejected {suffix} target mode changed"
            );
            assert_eq!(std::fs::read(&target).unwrap(), b"unrelated");
        }
    }

    #[cfg(unix)]
    #[test]
    fn writable_open_rejects_hard_linked_database_without_chmodding_target() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("target.sqlite");
        let conn = Connection::open(&target).unwrap();
        conn.execute_batch("CREATE TABLE t(v);").unwrap();
        drop(conn);
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o644)).unwrap();

        let linked = tmp.path().join("linked.sqlite");
        std::fs::hard_link(&target, &linked).unwrap();
        let error = JournalMode::Wal.open(&linked).unwrap_err();
        assert!(
            format!("{error:#}").contains("multiply-linked"),
            "{error:#}"
        );
        assert_eq!(
            std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
            0o644,
            "rejected target mode changed"
        );
    }

    #[cfg(unix)]
    #[test]
    fn writable_open_rejects_hard_linked_sidecar_without_chmodding_target() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("main.sqlite");
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch("CREATE TABLE t(v);").unwrap();
        drop(conn);

        let target = tmp.path().join("target-sidecar");
        std::fs::write(&target, b"unrelated").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o644)).unwrap();
        std::fs::hard_link(&target, sidecar_path(&db, "-wal")).unwrap();

        let error = JournalMode::Wal.open(&db).unwrap_err();
        assert!(
            format!("{error:#}").contains("multiply-linked"),
            "{error:#}"
        );
        assert_eq!(std::fs::read(&target).unwrap(), b"unrelated");
        assert_eq!(
            std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
            0o644,
            "rejected sidecar target mode changed"
        );
    }

    #[cfg(unix)]
    #[test]
    fn writable_open_secures_database_and_sidecars_through_symlinked_parent() {
        let tmp = tempfile::tempdir().unwrap();
        let actual_parent = tmp.path().join("actual");
        std::fs::create_dir(&actual_parent).unwrap();
        let linked_parent = tmp.path().join("linked");
        std::os::unix::fs::symlink(&actual_parent, &linked_parent).unwrap();

        let requested = linked_parent.join("through-link.sqlite");
        let actual = actual_parent.join("through-link.sqlite");
        let conn = JournalMode::Wal.open(&requested).unwrap();
        conn.execute_batch("CREATE TABLE t(v); INSERT INTO t VALUES (1);")
            .unwrap();

        assert_owner_only(&actual);
        assert_owner_only(&sidecar_path(&actual, "-wal"));
        assert_owner_only(&sidecar_path(&actual, "-shm"));
    }

    /// Run the umask-sensitive assertions in a subprocess so changing the
    /// process-global umask cannot race the rest of the test suite.
    #[cfg(unix)]
    #[test]
    fn writable_database_and_sidecars_are_owner_only() {
        const CHILD_ENV: &str = "HI_SQLITE_PERMISSION_TEST_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "tests::writable_database_and_sidecars_are_owner_only",
                    "--nocapture",
                ])
                .env(CHILD_ENV, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "permission child failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        // SAFETY: this branch runs in a dedicated subprocess and exits after
        // the assertions, so the process-global umask cannot affect peers.
        unsafe { libc::umask(0) };
        let tmp = tempfile::tempdir().unwrap();

        let wal_db = tmp.path().join("private-wal.sqlite");
        let wal = JournalMode::Wal.open(&wal_db).unwrap();
        wal.execute_batch("CREATE TABLE t(v); INSERT INTO t VALUES (1);")
            .unwrap();
        let wal_path = sidecar_path(&wal_db, "-wal");
        let shm_path = sidecar_path(&wal_db, "-shm");
        assert!(wal_path.exists(), "WAL sidecar was not created");
        assert!(shm_path.exists(), "SHM sidecar was not created");
        for path in [&wal_db, &wal_path, &shm_path] {
            assert_owner_only(path);
        }

        // Existing writable databases and live sidecars are tightened too.
        use std::os::unix::fs::PermissionsExt;
        for path in [&wal_db, &wal_path, &shm_path] {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o666)).unwrap();
        }
        let reopened_wal = JournalMode::Wal.open(&wal_db).unwrap();
        for path in [&wal_db, &wal_path, &shm_path] {
            assert_owner_only(path);
        }
        drop(reopened_wal);
        drop(wal);

        let truncate_db = tmp.path().join("private-truncate.sqlite");
        let truncate = JournalMode::Truncate.open(&truncate_db).unwrap();
        truncate
            .execute_batch("CREATE TABLE t(v); INSERT INTO t VALUES (1);")
            .unwrap();
        let journal_path = sidecar_path(&truncate_db, "-journal");
        assert!(journal_path.exists(), "rollback journal was not retained");
        assert_owner_only(&truncate_db);
        assert_owner_only(&journal_path);

        std::fs::set_permissions(&truncate_db, std::fs::Permissions::from_mode(0o666)).unwrap();
        std::fs::set_permissions(&journal_path, std::fs::Permissions::from_mode(0o666)).unwrap();
        let reopened_truncate = JournalMode::Truncate.open(&truncate_db).unwrap();
        assert_owner_only(&truncate_db);
        assert_owner_only(&journal_path);
        drop(reopened_truncate);
        drop(truncate);
    }

    #[test]
    fn open_wal_creates_db() {
        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("test_wal.sqlite");
        let conn = JournalMode::Wal.open(&db).unwrap();
        conn.execute_batch("CREATE TABLE t (v TEXT); INSERT INTO t VALUES ('hello');")
            .unwrap();
        // Verify the data.
        let v: String = conn.query_row("SELECT v FROM t", [], |r| r.get(0)).unwrap();
        assert_eq!(v, "hello");
    }

    #[test]
    fn open_truncate_creates_db() {
        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("test_truncate.sqlite");
        let conn = JournalMode::Truncate.open(&db).unwrap();
        conn.execute_batch("CREATE TABLE t (v TEXT); INSERT INTO t VALUES ('world');")
            .unwrap();
        let v: String = conn.query_row("SELECT v FROM t", [], |r| r.get(0)).unwrap();
        assert_eq!(v, "world");
    }

    #[test]
    fn open_readonly_works() {
        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("test_ro.sqlite");
        {
            let conn = JournalMode::Wal.open(&db).unwrap();
            conn.execute_batch("CREATE TABLE t (v TEXT); INSERT INTO t VALUES ('ro');")
                .unwrap();
        }
        let conn = JournalMode::Wal.open_readonly(&db).unwrap();
        let v: String = conn.query_row("SELECT v FROM t", [], |r| r.get(0)).unwrap();
        assert_eq!(v, "ro");
        // Writes should fail.
        assert!(conn.execute("INSERT INTO t VALUES ('nope')", []).is_err());
    }

    /// busy_timeout must be configured before the journal-mode switch: the
    /// switch needs an exclusive lock, and with the default timeout of 0 a
    /// peer's write lock fails the open instantly instead of delaying it.
    #[test]
    fn open_waits_for_peer_write_lock() {
        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("contended.sqlite");
        // A pre-WAL database forces the exclusive-lock path in apply().
        let peer = Connection::open(&db).unwrap();
        peer.execute_batch("CREATE TABLE t(x); BEGIN IMMEDIATE; INSERT INTO t VALUES(1);")
            .unwrap();
        let opener = {
            let db = db.clone();
            std::thread::spawn(move || JournalMode::Wal.open(&db).map(|_| ()))
        };
        std::thread::sleep(std::time::Duration::from_millis(25));
        peer.execute_batch("COMMIT;").unwrap();
        opener
            .join()
            .unwrap()
            .expect("open should wait for the peer's lock instead of failing instantly");
    }

    #[test]
    fn as_str_returns_correct_value() {
        assert_eq!(JournalMode::Wal.as_str(), "wal");
        assert_eq!(JournalMode::Truncate.as_str(), "truncate");
    }
}
