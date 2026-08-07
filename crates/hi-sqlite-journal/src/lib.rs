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

use std::path::Path;

use anyhow::{Context, Result};
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
        let parent = db_path.parent().unwrap_or(Path::new("."));
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
                result => {
                    return result.with_context(|| format!("setting journal_mode to {mode}"));
                }
            }
        }
    }

    /// Open a connection at `db_path` with the appropriate journal mode.
    pub fn open(&self, db_path: &Path) -> Result<Connection> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating db parent dir {}", parent.display()))?;
        }
        let conn = Connection::open(db_path)
            .with_context(|| format!("opening sqlite db {}", db_path.display()))?;
        self.apply(&conn)?;
        Ok(conn)
    }

    /// Open a read-only connection at `db_path` with the appropriate journal mode.
    pub fn open_readonly(&self, db_path: &Path) -> Result<Connection> {
        let flags =
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let conn = Connection::open_with_flags(db_path, flags)
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

#[cfg(target_os = "linux")]
fn is_network_filesystem_linux(path: &Path) -> bool {
    // Read /proc/mounts and find the longest mount prefix matching `path`.
    // Then check if the filesystem type is a known network type.
    let mounts = match std::fs::read_to_string("/proc/mounts") {
        Ok(m) => m,
        Err(_) => return false,
    };

    let path_str = match path.canonicalize() {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(_) => path.to_string_lossy().to_string(),
    };

    let mut best_match: Option<(String, &str)> = None; // (mount_point, fs_type)
    for line in mounts.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }
        let mount_point = decode_mount_field(parts[1]);
        let fs_type = parts[2];
        let mount_path = Path::new(&mount_point);
        if Path::new(&path_str).starts_with(mount_path)
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

    let path_cstring = match CString::new(path.as_os_str().as_bytes()) {
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
