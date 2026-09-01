use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::WorkflowOutcome;

pub const RUN_MANIFEST_VERSION: u32 = 1;
pub const MAX_RUN_ID_LEN: usize = 96;
pub const MAX_SCRIPT_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_ARGS_BYTES: usize = 1024 * 1024;
pub const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
pub const MAX_RESTORED_RUNS: usize = 256;
static REGISTER_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static WRITE_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
#[cfg(unix)]
const PRIVATE_DIR_MODE: u32 = 0o700;
#[cfg(unix)]
const PRIVATE_FILE_MODE: u32 = 0o600;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredRunStatus {
    Running,
    Paused,
    /// The exact pending approval has been durably selected for one-shot
    /// consumption, but the runnable state has not been published yet.
    ApprovalClaiming,
    Interrupted,
    Completed,
    BudgetExceeded,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowEngineKind {
    #[default]
    Rhai,
    Declarative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowRunManifest {
    pub version: u32,
    pub run_id: String,
    pub workflow_name: String,
    #[serde(default)]
    pub engine: WorkflowEngineKind,
    pub status: StoredRunStatus,
    pub current_phase: Option<String>,
    pub agent_budget: u64,
    pub agent_spent: u64,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub outcome: Option<WorkflowOutcome>,
    /// A workflow-side effect can pause with a durable, digest-bound approval.
    /// These identifiers are references only; the approval record lives in the
    /// project policy store.
    #[serde(default)]
    pub pending_approval_id: Option<String>,
    #[serde(default)]
    pub pending_operation_digest: Option<String>,
}

impl WorkflowRunManifest {
    pub fn new(
        run_id: String,
        workflow_name: String,
        agent_budget: u64,
    ) -> Result<Self, StoreError> {
        validate_run_id(&run_id)?;
        let now = now_ms();
        Ok(Self {
            version: RUN_MANIFEST_VERSION,
            run_id,
            workflow_name,
            engine: WorkflowEngineKind::Rhai,
            status: StoredRunStatus::Running,
            current_phase: None,
            agent_budget,
            agent_spent: 0,
            created_at_ms: now,
            updated_at_ms: now,
            outcome: None,
            pending_approval_id: None,
            pending_operation_digest: None,
        })
    }

    pub fn status(&self) -> crate::WorkflowRunStatus {
        match self.status {
            StoredRunStatus::Running => crate::WorkflowRunStatus::Active,
            StoredRunStatus::Paused => match self.outcome.as_ref() {
                Some(WorkflowOutcome::Paused { kind, .. }) => (*kind).into(),
                _ => crate::WorkflowRunStatus::UserPaused,
            },
            StoredRunStatus::ApprovalClaiming => crate::WorkflowRunStatus::Blocked,
            StoredRunStatus::Interrupted => crate::WorkflowRunStatus::Interrupted,
            StoredRunStatus::Completed => crate::WorkflowRunStatus::Complete,
            StoredRunStatus::BudgetExceeded => crate::WorkflowRunStatus::BudgetLimited,
            StoredRunStatus::Cancelled => crate::WorkflowRunStatus::Cancelled,
            StoredRunStatus::Failed => crate::WorkflowRunStatus::Failed,
        }
    }

    pub fn finish(&mut self, outcome: WorkflowOutcome) {
        self.status = match &outcome {
            WorkflowOutcome::Completed { .. } => StoredRunStatus::Completed,
            WorkflowOutcome::Paused { .. } => StoredRunStatus::Paused,
            WorkflowOutcome::BudgetExceeded { .. } => StoredRunStatus::BudgetExceeded,
            WorkflowOutcome::Cancelled => StoredRunStatus::Cancelled,
            WorkflowOutcome::Failed { .. } => StoredRunStatus::Failed,
        };
        self.outcome = Some(outcome);
        self.pending_approval_id = None;
        self.pending_operation_digest = None;
        self.updated_at_ms = now_ms();
    }

    pub fn set_pending_approval(
        &mut self,
        approval_id: impl Into<String>,
        digest: impl Into<String>,
    ) {
        self.status = StoredRunStatus::Paused;
        self.outcome = Some(WorkflowOutcome::Paused {
            kind: crate::PauseKind::Approval,
            message: "workflow is waiting for an approval".into(),
        });
        self.pending_approval_id = Some(approval_id.into());
        self.pending_operation_digest = Some(digest.into());
        self.updated_at_ms = now_ms();
    }

    pub fn begin_approval_claim(&mut self) {
        self.status = StoredRunStatus::ApprovalClaiming;
        self.updated_at_ms = now_ms();
    }
}

#[derive(Debug, Clone)]
pub struct StoredWorkflowRun {
    pub manifest: WorkflowRunManifest,
    pub script: String,
    pub args: serde_json::Value,
    pub journal_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct WorkflowRunStore {
    root: PathBuf,
}

#[derive(Debug)]
pub struct WorkflowRunOwnership {
    file: std::fs::File,
}

impl Drop for WorkflowRunOwnership {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("invalid workflow run id: {0}")]
    InvalidRunId(String),
    #[error("unsupported workflow run manifest version {0}")]
    Version(u32),
    #[error("workflow run artifact exceeds its {limit}-byte limit: {name}")]
    TooLarge { name: &'static str, limit: u64 },
    #[error("workflow run store io: {0}")]
    Io(#[from] std::io::Error),
    #[error("workflow run manifest json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("workflow run already exists: {0}")]
    AlreadyExists(String),
    #[error("workflow run store rejected symbolic link: {0}")]
    Symlink(PathBuf),
    #[error("workflow run store contains corrupt entry {run_id}: {message}")]
    CorruptEntry { run_id: String, message: String },
}

impl WorkflowRunStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn register(
        &self,
        manifest: &WorkflowRunManifest,
        script: &str,
        args: &serde_json::Value,
    ) -> Result<(), StoreError> {
        validate_run_id(&manifest.run_id)?;
        if script.len() > MAX_SCRIPT_BYTES {
            return Err(StoreError::TooLarge {
                name: "script",
                limit: MAX_SCRIPT_BYTES as u64,
            });
        }
        let args = serde_json::to_vec(args)?;
        if args.len() > MAX_ARGS_BYTES {
            return Err(StoreError::TooLarge {
                name: "args",
                limit: MAX_ARGS_BYTES as u64,
            });
        }
        validate_manifest(manifest)?;
        ensure_private_directory(&self.root, true)?;
        let dir = self.run_dir(&manifest.run_id)?;
        if std::fs::symlink_metadata(&dir).is_ok() {
            return Err(StoreError::AlreadyExists(manifest.run_id.clone()));
        }
        let staging = self.root.join(format!(
            ".{}.register-{}-{}",
            manifest.run_id,
            now_ms(),
            REGISTER_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        create_private_directory(&staging)?;
        let result = (|| {
            atomic_write(&staging.join("script.rhai"), script.as_bytes())?;
            atomic_write(&staging.join("args.json"), &args)?;
            write_manifest(&staging, manifest)?;
            match std::fs::rename(&staging, &dir) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists || dir.exists() => {
                    Err(StoreError::AlreadyExists(manifest.run_id.clone()))
                }
                Err(error) => Err(error.into()),
            }
        })();
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        result
    }

    pub fn persist(&self, manifest: &WorkflowRunManifest) -> Result<(), StoreError> {
        validate_manifest(manifest)?;
        let dir = self.run_dir(&manifest.run_id)?;
        ensure_private_directory(&self.root, false)?;
        ensure_private_directory(&dir, false)?;
        write_manifest(&dir, manifest)
    }

    pub fn load(&self, run_id: &str) -> Result<StoredWorkflowRun, StoreError> {
        let dir = self.run_dir(run_id)?;
        ensure_private_directory(&self.root, false)?;
        ensure_private_directory(&dir, false)?;
        let manifest_bytes = read_bounded(&dir.join("state.json"), MAX_MANIFEST_BYTES)?;
        let manifest: WorkflowRunManifest = serde_json::from_slice(&manifest_bytes)?;
        if manifest.version != RUN_MANIFEST_VERSION {
            return Err(StoreError::Version(manifest.version));
        }
        validate_run_id(&manifest.run_id)?;
        if manifest.run_id != run_id {
            return Err(StoreError::InvalidRunId(manifest.run_id));
        }
        let script = String::from_utf8(read_bounded(
            &dir.join("script.rhai"),
            MAX_SCRIPT_BYTES as u64,
        )?)
        .map_err(|e| StoreError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        let args = serde_json::from_slice(&read_bounded(
            &dir.join("args.json"),
            MAX_ARGS_BYTES as u64,
        )?)?;
        Ok(StoredWorkflowRun {
            manifest,
            script,
            args,
            journal_path: dir.join("journal.jsonl"),
        })
    }

    pub fn recover(&self, run_id: &str) -> Result<StoredWorkflowRun, StoreError> {
        let mut run = self.load(run_id)?;
        // Paused is an intentional durable state. Only a run that was actively
        // executing when the process disappeared should become Interrupted.
        if run.manifest.status == StoredRunStatus::Running {
            run.manifest.status = StoredRunStatus::Interrupted;
            run.manifest.outcome = None;
            run.manifest.updated_at_ms = now_ms();
            self.persist(&run.manifest)?;
        }
        Ok(run)
    }

    pub fn list(&self) -> Result<Vec<StoredWorkflowRun>, StoreError> {
        let mut runs = Vec::new();
        match ensure_private_directory(&self.root, false) {
            Ok(()) => {}
            Err(StoreError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(runs);
            }
            Err(error) => return Err(error),
        }
        let entries = match std::fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(runs),
            Err(error) => return Err(error.into()),
        };
        let mut ids = entries
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<Result<Vec<_>, _>>()?;
        ids.sort();
        for name in ids {
            let Some(id) = name.to_str().map(str::to_owned) else {
                continue;
            };
            if id.starts_with('.') {
                continue;
            }
            let run_dir = self.root.join(&id);
            if std::fs::symlink_metadata(run_dir.join("cleared")).is_ok() {
                continue;
            }
            match self.load(&id) {
                Ok(run) => runs.push(run),
                Err(error) => {
                    return Err(StoreError::CorruptEntry {
                        run_id: id,
                        message: error.to_string(),
                    });
                }
            }
        }
        runs.sort_by(|a, b| {
            b.manifest
                .updated_at_ms
                .cmp(&a.manifest.updated_at_ms)
                .then_with(|| a.manifest.run_id.cmp(&b.manifest.run_id))
        });
        runs.truncate(MAX_RESTORED_RUNS);
        Ok(runs)
    }

    pub fn delete(&self, run_id: &str) -> Result<(), StoreError> {
        let dir = self.run_dir(run_id)?;
        match ensure_private_directory(&dir, false) {
            Ok(()) => {}
            Err(StoreError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(());
            }
            Err(error) => return Err(error),
        }
        // Publish a durable tombstone before removing the run. A crash after
        // this write cannot make a half-deleted run visible to `list` again.
        atomic_write(&dir.join("cleared"), b"")?;
        match std::fs::remove_dir_all(dir) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    pub fn journal_path(&self, run_id: &str) -> Result<PathBuf, StoreError> {
        Ok(self.run_dir(run_id)?.join("journal.jsonl"))
    }

    /// Try to become the sole process allowed to transition or execute this
    /// run. The lock file is permanent; OS advisory ownership disappears when
    /// the guard/file descriptor closes, including after a crash.
    pub fn try_claim(&self, run_id: &str) -> Result<Option<WorkflowRunOwnership>, StoreError> {
        let dir = self.run_dir(run_id)?;
        ensure_private_directory(&self.root, false)?;
        ensure_private_directory(&dir, false)?;
        let path = dir.join("owner.lock");
        let file = open_private_lock_file(&path)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(WorkflowRunOwnership { file })),
            Err(std::fs::TryLockError::WouldBlock) => Ok(None),
            Err(std::fs::TryLockError::Error(error)) => Err(error.into()),
        }
    }

    fn run_dir(&self, run_id: &str) -> Result<PathBuf, StoreError> {
        validate_run_id(run_id)?;
        Ok(self.root.join(run_id))
    }
}

pub fn validate_run_id(run_id: &str) -> Result<(), StoreError> {
    if run_id.is_empty()
        || run_id.len() > MAX_RUN_ID_LEN
        || !run_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    {
        return Err(StoreError::InvalidRunId(run_id.to_owned()));
    }
    Ok(())
}

fn validate_manifest(manifest: &WorkflowRunManifest) -> Result<(), StoreError> {
    validate_run_id(&manifest.run_id)?;
    if manifest.version != RUN_MANIFEST_VERSION {
        return Err(StoreError::Version(manifest.version));
    }
    Ok(())
}

fn write_manifest(dir: &Path, manifest: &WorkflowRunManifest) -> Result<(), StoreError> {
    let bytes = serde_json::to_vec_pretty(manifest)?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(StoreError::TooLarge {
            name: "manifest",
            limit: MAX_MANIFEST_BYTES,
        });
    }
    atomic_write(&dir.join("state.json"), &bytes)
}

fn create_private_directory(path: &Path) -> Result<(), StoreError> {
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(PRIVATE_DIR_MODE);
    }
    builder.create(path)?;
    ensure_private_directory(path, false)
}

fn open_private_lock_file(path: &Path) -> Result<std::fs::File, StoreError> {
    loop {
        let before = match std::fs::symlink_metadata(path) {
            Ok(metadata) => Some(metadata),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        if before
            .as_ref()
            .is_some_and(|metadata| metadata.file_type().is_symlink() || !metadata.is_file())
        {
            return Err(StoreError::Symlink(path.to_owned()));
        }
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true);
        if before.is_none() {
            options.create_new(true);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options
                .mode(PRIVATE_FILE_MODE)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        }
        let file = match options.open(path) {
            Ok(file) => file,
            Err(error) if before.is_none() && error.kind() == std::io::ErrorKind::AlreadyExists => {
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let opened = file.metadata()?;
        if !opened.is_file() {
            return Err(StoreError::Symlink(path.to_owned()));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
            if before
                .as_ref()
                .is_some_and(|before| before.dev() != opened.dev() || before.ino() != opened.ino())
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "workflow ownership file changed during open: {}",
                        path.display()
                    ),
                )
                .into());
            }
            file.set_permissions(std::fs::Permissions::from_mode(PRIVATE_FILE_MODE))?;
        }
        return Ok(file);
    }
}

fn ensure_private_directory(path: &Path, create: bool) -> Result<(), StoreError> {
    if create {
        match create_private_directory(path) {
            Ok(()) => return Ok(()),
            Err(StoreError::Io(error)) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    let before = std::fs::symlink_metadata(path)?;
    if before.file_type().is_symlink() {
        return Err(StoreError::Symlink(path.to_owned()));
    }
    if !before.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotADirectory,
            format!("workflow run path is not a directory: {}", path.display()),
        )
        .into());
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _};
        let mut options = std::fs::OpenOptions::new();
        options
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC);
        let directory = options.open(path)?;
        let opened = directory.metadata()?;
        if before.dev() != opened.dev() || before.ino() != opened.ino() || !opened.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "workflow run directory changed during open: {}",
                    path.display()
                ),
            )
            .into());
        }
        directory.set_permissions(std::fs::Permissions::from_mode(PRIVATE_DIR_MODE))?;
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    #[cfg(test)]
    if fail_this_atomic_write_for_test() {
        return Err(std::io::Error::other("injected workflow artifact write failure").into());
    }
    let parent = path.parent().ok_or_else(|| {
        StoreError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "workflow artifact path has no parent",
        ))
    })?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    for _ in 0..16 {
        let nonce = WRITE_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options
                .mode(PRIVATE_FILE_MODE)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        }
        let mut file = match options.open(&tmp) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        };
        let result = (|| -> Result<(), StoreError> {
            use std::io::Write as _;
            file.write_all(bytes)?;
            file.sync_all()?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                file.set_permissions(std::fs::Permissions::from_mode(PRIVATE_FILE_MODE))?;
            }
            drop(file);
            replace_file(&tmp, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        return result;
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a private workflow artifact temp file",
    )
    .into())
}

#[cfg(not(windows))]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::rename(from, to)
}

#[cfg(windows)]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    match std::fs::rename(from, to) {
        Ok(()) => return Ok(()),
        Err(error) if std::fs::symlink_metadata(to).is_err() => return Err(error),
        Err(_) => {}
    }
    let parent = to.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "workflow artifact path has no parent",
        )
    })?;
    let name = to
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    for _ in 0..16 {
        let nonce = WRITE_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let backup = parent.join(format!(".{name}.bak-{}-{nonce}", std::process::id()));
        if std::fs::symlink_metadata(&backup).is_ok() {
            continue;
        }
        std::fs::rename(to, &backup)?;
        return match std::fs::rename(from, to) {
            Ok(()) => {
                let _ = std::fs::remove_file(backup);
                Ok(())
            }
            Err(error) => {
                if let Err(restore_error) = std::fs::rename(&backup, to) {
                    return Err(std::io::Error::other(format!(
                        "workflow artifact replacement failed ({error}); restoring the previous file also failed ({restore_error})"
                    )));
                }
                Err(error)
            }
        };
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a workflow artifact backup path",
    ))
}

#[cfg(test)]
thread_local! {
    static FAIL_ATOMIC_WRITE_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn fail_nth_atomic_write_for_test(nth: usize) {
    assert!(nth > 0, "failure injection count must be positive");
    FAIL_ATOMIC_WRITE_AFTER.with(|remaining| remaining.set(Some(nth)));
}

#[cfg(test)]
pub(crate) fn clear_atomic_write_failure_for_test() {
    FAIL_ATOMIC_WRITE_AFTER.with(|remaining| remaining.set(None));
}

#[cfg(test)]
fn fail_this_atomic_write_for_test() -> bool {
    FAIL_ATOMIC_WRITE_AFTER.with(|remaining| match remaining.get() {
        Some(1) => {
            remaining.set(None);
            true
        }
        Some(count) => {
            remaining.set(Some(count - 1));
            false
        }
        None => false,
    })
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, StoreError> {
    let before = std::fs::symlink_metadata(path)?;
    if before.file_type().is_symlink() {
        return Err(StoreError::Symlink(path.to_owned()));
    }
    if !before.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "workflow run artifact is not a regular file: {}",
                path.display()
            ),
        )
        .into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    let file = options.open(path)?;
    let opened = file.metadata()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if before.dev() != opened.dev() || before.ino() != opened.ino() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "workflow run artifact changed during open: {}",
                    path.display()
                ),
            )
            .into());
        }
        file.set_permissions(std::fs::Permissions::from_mode(PRIVATE_FILE_MODE))?;
    }
    if !opened.is_file() || opened.len() > limit {
        return Err(StoreError::TooLarge {
            name: "stored artifact",
            limit,
        });
    }
    use std::io::Read as _;
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(StoreError::TooLarge {
            name: "stored artifact",
            limit,
        });
    }
    Ok(bytes)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_ignores_tombstoned_runs() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let manifest = WorkflowRunManifest::new("cleared-run".into(), "test".into(), 8).unwrap();
        store
            .register(&manifest, "complete(1);", &serde_json::json!({}))
            .unwrap();
        atomic_write(&dir.path().join("cleared-run/cleared"), b"").unwrap();

        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn load_is_pure_and_recovery_persists_interruption() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let manifest = WorkflowRunManifest::new("run-1".into(), "review".into(), 128).unwrap();
        store
            .register(&manifest, "complete(args);", &serde_json::json!({"x": 1}))
            .unwrap();
        let restored = store.load("run-1").unwrap();
        assert_eq!(restored.manifest.status, StoredRunStatus::Running);
        let restored = store.recover("run-1").unwrap();
        assert_eq!(restored.manifest.status, StoredRunStatus::Interrupted);
        assert_eq!(
            store.load("run-1").unwrap().manifest.status,
            StoredRunStatus::Interrupted
        );
        assert_eq!(restored.args["x"], 1);
        assert!(restored.journal_path.ends_with("journal.jsonl"));
    }

    #[test]
    fn persist_replaces_an_existing_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manifest = WorkflowRunManifest::new("replace".into(), "first".into(), 8).unwrap();
        store
            .register(&manifest, "complete(1);", &serde_json::json!({}))
            .unwrap();
        manifest.workflow_name = "second".into();
        store.persist(&manifest).unwrap();
        assert_eq!(
            store.load("replace").unwrap().manifest.workflow_name,
            "second"
        );
    }

    #[test]
    fn run_ownership_is_exclusive_and_released_with_the_guard() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let manifest = WorkflowRunManifest::new("owned".into(), "test".into(), 8).unwrap();
        store
            .register(&manifest, "complete(1);", &serde_json::json!({}))
            .unwrap();

        let first = store.try_claim("owned").unwrap().unwrap();
        assert!(store.try_claim("owned").unwrap().is_none());
        drop(first);
        assert!(store.try_claim("owned").unwrap().is_some());
    }

    #[test]
    fn recovery_preserves_paused_runs_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manifest = WorkflowRunManifest::new("paused".into(), "review".into(), 8).unwrap();
        manifest.finish(WorkflowOutcome::Paused {
            kind: crate::PauseKind::User,
            message: "waiting".into(),
        });
        store
            .register(&manifest, "complete(1);", &serde_json::json!({}))
            .unwrap();

        let first = store.recover("paused").unwrap();
        assert_eq!(first.manifest.status, StoredRunStatus::Paused);
        assert!(matches!(
            first.manifest.outcome,
            Some(WorkflowOutcome::Paused {
                kind: crate::PauseKind::User,
                ..
            })
        ));
        let updated_at = first.manifest.updated_at_ms;
        let second = store.recover("paused").unwrap();
        assert_eq!(second.manifest.status, StoredRunStatus::Paused);
        assert_eq!(second.manifest.updated_at_ms, updated_at);
    }

    #[test]
    fn rejects_unsafe_ids() {
        for id in ["", "../x", "a/b", ".", "x y"] {
            assert!(validate_run_id(id).is_err(), "accepted {id:?}");
        }
    }

    #[test]
    fn preserves_terminal_status_and_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manifest = WorkflowRunManifest::new("done".into(), "review".into(), 8).unwrap();
        manifest.finish(WorkflowOutcome::Completed {
            result: serde_json::json!({"ok": true}),
        });
        store
            .register(&manifest, "complete(args);", &serde_json::json!({}))
            .unwrap();
        assert_eq!(
            store.load("done").unwrap().manifest.status,
            StoredRunStatus::Completed
        );
        store.delete("done").unwrap();
        assert!(store.load("done").is_err());
    }

    #[test]
    fn registration_is_no_clobber() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let manifest = WorkflowRunManifest::new("same".into(), "first".into(), 8).unwrap();
        store
            .register(
                &manifest,
                "complete(1);",
                &serde_json::json!({"first": true}),
            )
            .unwrap();
        let mut replacement = manifest.clone();
        replacement.workflow_name = "second".into();
        assert!(matches!(
            store.register(&replacement, "complete(2);", &serde_json::json!({})),
            Err(StoreError::AlreadyExists(_))
        ));
        let loaded = store.load("same").unwrap();
        assert_eq!(loaded.manifest.workflow_name, "first");
        assert_eq!(loaded.script, "complete(1);");
    }

    #[test]
    fn list_reports_corrupt_entries() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("bad")).unwrap();
        std::fs::write(dir.path().join("bad/state.json"), b"not json").unwrap();
        let error = WorkflowRunStore::new(dir.path()).list().unwrap_err();
        assert!(matches!(error, StoreError::CorruptEntry { run_id, .. } if run_id == "bad"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_run_directories_and_artifacts() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), dir.path().join("linked")).unwrap();
        let store = WorkflowRunStore::new(dir.path());
        assert!(matches!(store.load("linked"), Err(StoreError::Symlink(_))));

        let manifest = WorkflowRunManifest::new("artifact".into(), "review".into(), 8).unwrap();
        store
            .register(&manifest, "complete(1);", &serde_json::json!({}))
            .unwrap();
        std::fs::remove_file(dir.path().join("artifact/script.rhai")).unwrap();
        let target = outside.path().join("script.rhai");
        std::fs::write(&target, "complete(2);").unwrap();
        symlink(target, dir.path().join("artifact/script.rhai")).unwrap();
        assert!(matches!(
            store.load("artifact"),
            Err(StoreError::Symlink(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn run_artifacts_are_owner_only_and_existing_modes_are_tightened() {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let manifest = WorkflowRunManifest::new("private".into(), "review".into(), 8).unwrap();
        store
            .register(
                &manifest,
                "complete(args);",
                &serde_json::json!({"secret": true}),
            )
            .unwrap();

        assert_eq!(std::fs::metadata(dir.path()).unwrap().mode() & 0o777, 0o700);
        let run_dir = dir.path().join("private");
        assert_eq!(std::fs::metadata(&run_dir).unwrap().mode() & 0o777, 0o700);
        for name in ["state.json", "script.rhai", "args.json"] {
            let path = run_dir.join(name);
            assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).unwrap();
        }

        store.load("private").unwrap();
        for name in ["state.json", "script.rhai", "args.json"] {
            assert_eq!(
                std::fs::metadata(run_dir.join(name)).unwrap().mode() & 0o777,
                0o600
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn stale_or_symlinked_legacy_temp_does_not_block_or_overwrite_persist() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let outside = dir.path().join("outside");
        std::fs::write(&outside, b"do not overwrite").unwrap();
        let store = WorkflowRunStore::new(dir.path().join("runs"));
        let mut manifest = WorkflowRunManifest::new("safe".into(), "review".into(), 8).unwrap();
        store
            .register(&manifest, "complete(1);", &serde_json::json!({}))
            .unwrap();
        let tmp = dir.path().join("runs/safe/state.tmp");
        symlink(&outside, &tmp).unwrap();
        manifest.workflow_name = "changed".into();

        store.persist(&manifest).unwrap();
        assert_eq!(std::fs::read(&outside).unwrap(), b"do not overwrite");
        assert_eq!(
            store.load("safe").unwrap().manifest.workflow_name,
            "changed"
        );

        std::fs::remove_file(tmp).unwrap();
        std::fs::write(dir.path().join("runs/safe/state.tmp"), b"stale crash temp").unwrap();
        manifest.workflow_name = "changed-again".into();
        store.persist(&manifest).unwrap();
        assert_eq!(
            store.load("safe").unwrap().manifest.workflow_name,
            "changed-again"
        );
    }

    #[test]
    fn list_sorts_before_applying_cap() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        for index in 0..=MAX_RESTORED_RUNS {
            let id = format!("run-{index:03}");
            let mut manifest = WorkflowRunManifest::new(id, "review".into(), 8).unwrap();
            manifest.updated_at_ms = index as u64;
            store
                .register(&manifest, "complete(1);", &serde_json::json!({}))
                .unwrap();
        }
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), MAX_RESTORED_RUNS);
        assert_eq!(
            listed.first().unwrap().manifest.run_id,
            format!("run-{:03}", MAX_RESTORED_RUNS)
        );
        assert_eq!(listed.last().unwrap().manifest.run_id, "run-001");
    }

    #[test]
    fn rejects_unknown_manifest_version() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manifest = WorkflowRunManifest::new("run".into(), "review".into(), 8).unwrap();
        manifest.version += 1;
        assert!(matches!(
            store.persist(&manifest),
            Err(StoreError::Version(_))
        ));
    }
}
