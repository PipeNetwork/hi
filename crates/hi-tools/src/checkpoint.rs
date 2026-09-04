//! Git-backed working-tree checkpoints for `/undo`.
//!
//! Before each turn the agent snapshots the full non-ignored working tree into a
//! *dangling* commit — built in a throwaway index so it never touches the user's
//! staging area, branch, or history. `/undo` restores the latest snapshot,
//! reverting every file the turn created, modified, or deleted in one step. This
//! is what makes running with no confirmation prompts safe: anything is undoable.
//!
//! Git checkpoints cover the non-ignored tree below the explicit workspace
//! plus small gitignored inputs (`.env`, vendored sources) — but never
//! gitignored regenerable artifacts (`target/`, `node_modules/`, caches) or
//! gitignored bulk files over [`MAX_IGNORED_CHECKPOINT_FILE_BYTES`] (dataset
//! shards, model weights), and never the runtime's own state root. Excluding
//! those keeps a data-heavy workspace checkpointable at all and keeps build
//! output out of the user's `.git/objects`. Internal checkpoints (non-git
//! workspaces) cover the complete bounded tree. Both preserve executable modes
//! and symlink targets. If the covered tree exceeds the checkpoint limits,
//! mutation is denied unless the caller explicitly allows no checkpoint.
//! Neither can undo non-file side effects such as network changes or deletes
//! outside the workspace; those are what the catastrophic-operation guard is
//! for.

use std::ffi::{OsStr, OsString};
use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::process::Output;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

const SEALED_REFERENCE_PREFIX: &str = "sealed:v1:";
static ISOLATED_ID: AtomicU64 = AtomicU64::new(0);
static ISOLATED_NAME_SALT: OnceLock<String> = OnceLock::new();

// Checkpoint Git commands are local and normally finish in milliseconds, but
// large repositories can legitimately need time to hash hundreds of MiB. Keep
// a generous finite ceiling while making cancellation immediate and ensuring a
// wedged hook/filter can never pin a turn forever.
const GIT_COMMAND_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const GIT_PIPE_DRAIN_GRACE: Duration = Duration::from_secs(2);
const GIT_REAP_GRACE: Duration = Duration::from_secs(2);

const MAX_CHECKPOINT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_CHECKPOINT_ENTRIES: usize = 200_000;

/// Explicit result of attempting to create a working-tree checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CreateResult {
    Created(String),
    /// Checkpointing is not available in this workspace (normally non-Git).
    Unavailable(String),
    /// Git was available, but snapshot creation actually failed.
    Failed(String),
}

/// Minimal raw-output Git runner for checkpoint plumbing.
///
/// `ProcessRunner` intentionally turns subprocess output into bounded UTF-8
/// summaries. Checkpoint operations need the exact bytes produced by Git
/// (`-z` path lists and blobs included), so this runner keeps `Output` binary
/// while retaining the same non-interactive lifecycle. On Unix, each Git
/// command gets its own process group so cancellation tears down descendants;
/// on other platforms, cancellation is limited to the direct Git child.
#[derive(Clone, Debug)]
struct GitRunner {
    program: OsString,
    timeout: Duration,
}

impl Default for GitRunner {
    fn default() -> Self {
        Self {
            program: OsString::from("git"),
            timeout: GIT_COMMAND_TIMEOUT,
        }
    }
}

impl GitRunner {
    #[cfg(test)]
    fn new(program: impl Into<OsString>, timeout: Duration) -> Self {
        Self {
            program: program.into(),
            timeout,
        }
    }

    async fn output(
        &self,
        dir: &Path,
        args: &[OsString],
        extra_env: &[(&str, &str)],
    ) -> Result<Output> {
        self.output_with_completion(dir, args, extra_env, None)
            .await
    }

    async fn output_with_completion(
        &self,
        dir: &Path,
        args: &[OsString],
        extra_env: &[(&str, &str)],
        completion: Option<BlockingWorkSignal>,
    ) -> Result<Output> {
        let mut command = Command::new(&self.program);
        command
            .arg("-C")
            .arg(dir)
            .args(args)
            .envs(extra_env.iter().copied())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GCM_INTERACTIVE", "never")
            .env("PAGER", "cat")
            .env("GIT_PAGER", "cat")
            .env("GIT_EDITOR", "true")
            .env("GIT_SEQUENCE_EDITOR", "true");
        #[cfg(unix)]
        command.process_group(0);

        let child = command.spawn().context("spawning git")?;
        let mut child = ReapingGitChild::new(child, completion);
        let mut group = GitProcessGroupGuard::for_child(child.child());
        let stdout = child
            .child_mut()
            .stdout
            .take()
            .context("capturing git stdout")?;
        let stderr = child
            .child_mut()
            .stderr
            .take()
            .context("capturing git stderr")?;

        let execution = async {
            let drains = async {
                tokio::try_join!(read_raw_pipe(stdout), read_raw_pipe(stderr))
                    .context("reading git output")
            };
            tokio::pin!(drains);
            let wait = child.child_mut().wait();
            tokio::pin!(wait);

            tokio::select! {
                status = &mut wait => {
                    let status = status.context("waiting for git")?;
                    // A Git hook/filter may have left a descendant holding the
                    // inherited pipes. The direct child is settled, so remove
                    // any such strays and give buffered bytes a bounded drain.
                    group.terminate();
                    let (stdout, stderr) = tokio::time::timeout(
                        GIT_PIPE_DRAIN_GRACE,
                        &mut drains,
                    )
                    .await
                    .context("timed out draining git output")??;
                    Ok(Output { status, stdout, stderr })
                }
                captured = &mut drains => {
                    let (stdout, stderr) = captured?;
                    let status = wait.await.context("waiting for git")?;
                    group.terminate();
                    Ok(Output { status, stdout, stderr })
                }
            }
        };

        match tokio::time::timeout(self.timeout, execution).await {
            Ok(Ok(output)) => {
                child.mark_reaped();
                Ok(output)
            }
            Ok(Err(error)) => {
                if terminate_and_reap_git(child.child_mut(), &mut group).await {
                    child.mark_reaped();
                }
                Err(error)
            }
            Err(_) => {
                if terminate_and_reap_git(child.child_mut(), &mut group).await {
                    child.mark_reaped();
                }
                bail!(
                    "git command timed out after {:.3}s",
                    self.timeout.as_secs_f64()
                )
            }
        }
    }
}

/// Owns a spawned Git process until it has actually been waited. Dropping a
/// Tokio Child with kill-on-drop requests termination but does not wait for
/// process exit. A cancelled checkpoint future can therefore otherwise release
/// its temporary-index or worktree guards while Git still has those resources
/// open. The optional signal is released only after a successful reap.
struct ReapingGitChild {
    child: Option<tokio::process::Child>,
    completion: Option<BlockingWorkSignal>,
}

impl ReapingGitChild {
    fn new(child: tokio::process::Child, completion: Option<BlockingWorkSignal>) -> Self {
        Self {
            child: Some(child),
            completion,
        }
    }

    fn child(&self) -> &tokio::process::Child {
        self.child.as_ref().expect("Git child must be present")
    }

    fn child_mut(&mut self) -> &mut tokio::process::Child {
        self.child.as_mut().expect("Git child must be present")
    }

    fn mark_reaped(&mut self) {
        self.child.take();
        // Dropping the signal wakes every dependent cleanup waiter. Do this
        // only after the OS child has been waited successfully.
        self.completion.take();
    }
}

impl Drop for ReapingGitChild {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        #[cfg(unix)]
        if let Some(pid) = child.id() {
            crate::process::kill_group(pid as i32);
        }
        let _ = child.start_kill();

        // ManuallyDrop is deliberate: if the OS cannot start a reaper thread,
        // retaining the signal leaves dependent cleanup safely disarmed rather
        // than deleting resources beneath a child whose exit is unproven.
        let completion = self.completion.take().map(std::mem::ManuallyDrop::new);
        let spawned = std::thread::Builder::new()
            .name("hi-git-reaper".into())
            .spawn(move || {
                loop {
                    match child.try_wait() {
                        Ok(Some(_)) => break,
                        Ok(None) => {
                            #[cfg(unix)]
                            if let Some(pid) = child.id() {
                                crate::process::kill_group(pid as i32);
                            }
                            let _ = child.start_kill();
                            std::thread::sleep(Duration::from_millis(10));
                        }
                        Err(error) => {
                            eprintln!("warning: could not reap cancelled Git child: {error}");
                            return;
                        }
                    }
                }
                if let Some(completion) = completion {
                    drop(std::mem::ManuallyDrop::into_inner(completion));
                }
            });
        if let Err(error) = spawned {
            eprintln!("warning: could not start cancelled Git child reaper: {error}");
        }
    }
}

async fn read_raw_pipe<R>(mut pipe: R) -> std::io::Result<Vec<u8>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;

    let mut bytes = Vec::new();
    pipe.read_to_end(&mut bytes).await?;
    Ok(bytes)
}

#[cfg(unix)]
struct GitProcessGroupGuard {
    pgid: Option<i32>,
}

#[cfg(unix)]
impl GitProcessGroupGuard {
    fn for_child(child: &tokio::process::Child) -> Self {
        Self {
            pgid: child.id().map(|pid| pid as i32),
        }
    }

    fn terminate(&mut self) {
        if let Some(pgid) = self.pgid.take() {
            crate::process::kill_group(pgid);
        }
    }
}

#[cfg(unix)]
impl Drop for GitProcessGroupGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[cfg(not(unix))]
struct GitProcessGroupGuard;

#[cfg(not(unix))]
impl GitProcessGroupGuard {
    fn for_child(_child: &tokio::process::Child) -> Self {
        Self
    }

    fn terminate(&mut self) {}
}

async fn terminate_and_reap_git(
    child: &mut tokio::process::Child,
    group: &mut GitProcessGroupGuard,
) -> bool {
    group.terminate();
    let _ = child.start_kill();
    matches!(
        tokio::time::timeout(GIT_REAP_GRACE, child.wait()).await,
        Ok(Ok(_))
    )
}

/// Owns short-lived checkpoint plumbing files across async Git calls. Git
/// writes an adjacent `<index>.lock`; cancellation can kill Git before it
/// unlinks that lock, so the index guard owns both paths.
struct TemporaryFileGuard {
    paths: Vec<PathBuf>,
    blocking_work: Option<BlockingWorkCompletion>,
}

impl TemporaryFileGuard {
    fn file(path: PathBuf) -> Self {
        Self {
            paths: vec![path],
            blocking_work: None,
        }
    }

    fn git_index(path: PathBuf) -> Self {
        let lock = path_with_suffix(&path, ".lock");
        Self {
            paths: vec![path, lock],
            blocking_work: None,
        }
    }

    fn path(&self) -> &Path {
        &self.paths[0]
    }

    fn wait_for(&mut self, completion: BlockingWorkCompletion) {
        self.blocking_work = Some(completion);
    }
}

impl Drop for TemporaryFileGuard {
    fn drop(&mut self) {
        let paths = std::mem::take(&mut self.paths);
        let Some(completion) = self.blocking_work.take() else {
            remove_temporary_files(paths);
            return;
        };
        if completion.is_finished() {
            remove_temporary_files(paths);
            return;
        }

        // A cancelled Git future hands its child to a detached reaper. Keep
        // the index, adjacent lock, and pathspec alive until that reaper proves
        // the child has exited.
        if let Err(error) = std::thread::Builder::new()
            .name("hi-checkpoint-temp-cleanup".into())
            .spawn(move || {
                completion.wait_blocking();
                remove_temporary_files(paths);
            })
        {
            eprintln!("warning: could not start checkpoint temporary-file cleanup: {error}");
        }
    }
}

fn remove_temporary_files(paths: Vec<PathBuf>) {
    for path in paths {
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => eprintln!(
                "warning: could not remove checkpoint temporary file {}: {error}",
                path.display()
            ),
        }
    }
}

fn path_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn isolated_name_salt() -> &'static str {
    ISOLATED_NAME_SALT.get_or_init(|| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let address = (&now as *const u128) as usize as u128;
        format!(
            "{:032x}",
            now ^ address ^ (std::process::id() as u128) << 64
        )
    })
}

fn cleanup_record_path(sandbox: &Path) -> Result<PathBuf> {
    let parent = sandbox
        .parent()
        .context("isolated sandbox has no allocation parent")?;
    let name = sandbox
        .file_name()
        .context("isolated sandbox has no file name")?;
    let mut record_name = name.to_os_string();
    record_name.push(".cleanup.json");
    Ok(parent.join(record_name))
}

/// Completion handshake for blocking work that may outlive its awaiting
/// future. `spawn_blocking` tasks cannot be force-cancelled once running, so an
/// isolated directory must not be removed until its materializer has stopped
/// writing to it.
#[derive(Debug, Default)]
struct BlockingWorkState {
    finished: bool,
    waiters: usize,
}

#[derive(Clone, Debug)]
struct BlockingWorkCompletion(Arc<(Mutex<BlockingWorkState>, Condvar)>);

struct BlockingWorkSignal(BlockingWorkCompletion);

impl BlockingWorkCompletion {
    fn pair() -> (Self, BlockingWorkSignal) {
        let completion = Self(Arc::new((
            Mutex::new(BlockingWorkState::default()),
            Condvar::new(),
        )));
        let signal = BlockingWorkSignal(completion.clone());
        (completion, signal)
    }

    async fn wait(&self) -> Result<()> {
        let completion = self.clone();
        tokio::task::spawn_blocking(move || completion.wait_blocking())
            .await
            .context("blocking work completion waiter failed")?;
        Ok(())
    }

    fn wait_blocking(&self) {
        let (state, wake) = &*self.0;
        let mut state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.waiters = state.waiters.saturating_add(1);
        wake.notify_all();
        while !state.finished {
            state = wake
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn is_finished(&self) -> bool {
        self.0
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .finished
    }

    #[cfg(test)]
    fn waiter_count(&self) -> usize {
        self.0
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .waiters
    }
}

impl Drop for BlockingWorkSignal {
    fn drop(&mut self) {
        let (state, wake) = &*self.0.0;
        state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .finished = true;
        wake.notify_all();
    }
}

#[derive(Clone, Debug)]
struct IsolatedCleanup {
    path: PathBuf,
    git_repo: Option<PathBuf>,
    registered_worktree: bool,
    git: GitRunner,
    blocking_work: Option<BlockingWorkCompletion>,
    record_path: Option<PathBuf>,
}

/// Durable breadcrumb retained when detached cleanup cannot finish. These files
/// are intentionally not auto-consumed: the shared temporary directory is an
/// untrusted boundary, and safely authenticating a dead process's repository
/// ownership would require a broader recovery protocol.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct IsolatedCleanupRecord {
    version: u32,
    owner_pid: u32,
    path: IsolatedRecordPath,
    git_repo: Option<IsolatedRecordPath>,
    registered_worktree: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "encoding", content = "value", rename_all = "snake_case")]
enum IsolatedRecordPath {
    UnixBytes(String),
    WindowsWide(Vec<u16>),
    Utf8(String),
}

impl IsolatedRecordPath {
    fn encode(path: &Path) -> Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;

            let value = path
                .as_os_str()
                .as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            Ok(Self::UnixBytes(value))
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;

            Ok(Self::WindowsWide(path.as_os_str().encode_wide().collect()))
        }
        #[cfg(all(not(unix), not(windows)))]
        {
            Ok(Self::Utf8(
                path.to_str()
                    .context("isolated cleanup path is not Unicode")?
                    .to_string(),
            ))
        }
    }

    #[cfg(test)]
    fn decode(&self) -> Result<PathBuf> {
        match self {
            Self::UnixBytes(value) => {
                #[cfg(unix)]
                {
                    use std::os::unix::ffi::OsStringExt;

                    ensure!(
                        value.len().is_multiple_of(2)
                            && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
                        "invalid encoded Unix cleanup path"
                    );
                    let bytes = (0..value.len())
                        .step_by(2)
                        .map(|index| u8::from_str_radix(&value[index..index + 2], 16))
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    Ok(PathBuf::from(OsString::from_vec(bytes)))
                }
                #[cfg(not(unix))]
                {
                    let _ = value;
                    bail!("Unix cleanup path cannot be decoded on this platform")
                }
            }
            Self::WindowsWide(value) => {
                #[cfg(windows)]
                {
                    use std::os::windows::ffi::OsStringExt;

                    Ok(PathBuf::from(OsString::from_wide(value)))
                }
                #[cfg(not(windows))]
                {
                    let _ = value;
                    bail!("Windows cleanup path cannot be decoded on this platform")
                }
            }
            Self::Utf8(value) => Ok(PathBuf::from(value)),
        }
    }
}

struct IsolatedGuard {
    path: PathBuf,
    git_repo: Option<PathBuf>,
    registered_worktree: bool,
    git: GitRunner,
    blocking_work: Option<BlockingWorkCompletion>,
    record_path: Option<PathBuf>,
    cleaned: bool,
}

impl IsolatedGuard {
    #[cfg(test)]
    fn directory(path: PathBuf) -> Self {
        Self {
            path,
            git_repo: None,
            registered_worktree: false,
            git: GitRunner::default(),
            blocking_work: None,
            record_path: None,
            cleaned: false,
        }
    }

    fn directory_after(path: PathBuf, completion: BlockingWorkCompletion) -> Result<Self> {
        let mut guard = Self {
            path,
            git_repo: None,
            registered_worktree: false,
            git: GitRunner::default(),
            blocking_work: Some(completion),
            record_path: None,
            // Keep Drop disarmed until the breadcrumb is durable. A
            // constructor error means no materialization has started and
            // therefore owns nothing that needs asynchronous cleanup.
            cleaned: true,
        };
        guard.persist_cleanup_record()?;
        guard.cleaned = false;
        Ok(guard)
    }

    fn worktree(path: PathBuf, git_repo: PathBuf, git: GitRunner) -> Result<Self> {
        let mut guard = Self {
            path,
            git_repo: Some(git_repo),
            registered_worktree: true,
            git,
            blocking_work: None,
            record_path: None,
            // In particular, do not run `git worktree prune` if breadcrumb
            // persistence itself fails before `git worktree add` is attempted.
            cleaned: true,
        };
        guard.persist_cleanup_record()?;
        guard.cleaned = false;
        Ok(guard)
    }

    fn persist_cleanup_record(&mut self) -> Result<()> {
        let record_path = cleanup_record_path(&self.path)?;
        let record = IsolatedCleanupRecord {
            version: 1,
            owner_pid: std::process::id(),
            path: IsolatedRecordPath::encode(&self.path)?,
            git_repo: self
                .git_repo
                .as_deref()
                .map(IsolatedRecordPath::encode)
                .transpose()?,
            registered_worktree: self.registered_worktree,
        };
        let bytes = serde_json::to_vec(&record).context("serializing isolated cleanup record")?;
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&record_path)
            .with_context(|| format!("creating {}", record_path.display()))?;
        let persist = file
            .write_all(&bytes)
            .with_context(|| format!("writing {}", record_path.display()))
            .and_then(|()| {
                file.sync_all()
                    .with_context(|| format!("syncing {}", record_path.display()))
            });
        if let Err(error) = persist {
            drop(file);
            let _ = std::fs::remove_file(&record_path);
            return Err(error).context("persisting isolated cleanup record");
        }
        drop(file);
        self.record_path = Some(record_path);
        Ok(())
    }

    fn cleanup_plan(&self) -> IsolatedCleanup {
        IsolatedCleanup {
            path: self.path.clone(),
            git_repo: self.git_repo.clone(),
            registered_worktree: self.registered_worktree,
            git: self.git.clone(),
            blocking_work: self.blocking_work.clone(),
            record_path: self.record_path.clone(),
        }
    }

    fn wait_for(&mut self, completion: BlockingWorkCompletion) {
        self.blocking_work = Some(completion);
    }

    async fn cleanup(&mut self) -> Result<()> {
        if self.cleaned {
            return Ok(());
        }
        let cleanup = self.cleanup_plan();
        cleanup_isolated_physical(&cleanup).await?;
        // `self.path` is the only directory this guard owns. The parent is a
        // shared allocation root: another verifier may have created it but not
        // yet materialized its child. Removing that empty parent here creates
        // a TOCTOU race where the other verifier reports an infrastructure
        // failure instead of the stage's real result.
        //
        // Physical ownership has ended at this point. Disarm before removing
        // the durable breadcrumb so an unlink failure cannot make Drop run a
        // second destructive Git removal/prune sequence.
        self.cleaned = true;
        remove_isolated_cleanup_record(&cleanup)
    }
}

impl Drop for IsolatedGuard {
    fn drop(&mut self) {
        if self.cleaned {
            return;
        }
        self.cleaned = true;
        let cleanup = self.cleanup_plan();
        // Cancellation drops this guard on the async turn path. Cleanup must
        // outlive that cancelled future without making Drop itself wait on Git
        // or a recursive filesystem traversal, so give recovery its own small
        // runtime on a detached OS thread.
        if let Err(error) = spawn_isolated_cleanup(cleanup) {
            eprintln!(
                "warning: could not start isolated verification cleanup thread for {}: {error}",
                self.path.display()
            );
        }
    }
}

fn spawn_isolated_cleanup(cleanup: IsolatedCleanup) -> std::io::Result<()> {
    let path = cleanup.path.clone();
    std::thread::Builder::new()
        .name("hi-isolated-cleanup".into())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            match runtime {
                Ok(runtime) => {
                    if let Err(error) = runtime.block_on(cleanup_isolated(cleanup)) {
                        eprintln!(
                            "warning: isolated verification cleanup failed for {}: {error:#}",
                            path.display()
                        );
                    }
                }
                Err(error) => eprintln!(
                    "warning: could not start isolated verification cleanup runtime for {}: {error}",
                    path.display()
                ),
            }
        })
        .map(|_| ())
}

async fn cleanup_isolated(cleanup: IsolatedCleanup) -> Result<()> {
    cleanup_isolated_physical(&cleanup).await?;
    remove_isolated_cleanup_record(&cleanup)
}

async fn cleanup_isolated_physical(cleanup: &IsolatedCleanup) -> Result<()> {
    if let Some(completion) = &cleanup.blocking_work {
        completion.wait().await?;
    }
    if cleanup.registered_worktree {
        let repo = cleanup
            .git_repo
            .as_ref()
            .context("isolated worktree has no source repository")?;
        let remove_args = vec![
            OsString::from("worktree"),
            OsString::from("remove"),
            OsString::from("--force"),
            cleanup.path.clone().into_os_string(),
        ];
        let removed = cleanup
            .git
            .output(repo, &remove_args, &[])
            .await
            .context("removing isolated verification worktree");
        if !matches!(&removed, Ok(output) if output.status.success()) {
            // Removing the directory and pruning the now-missing worktree is
            // an idempotent fallback, including cancellation during `add` or a
            // verification command that damaged its own `.git` file.
            let directory_removal = remove_dir_all_async(cleanup.path.clone()).await;
            let prune_args = [
                OsString::from("worktree"),
                OsString::from("prune"),
                OsString::from("--expire"),
                OsString::from("now"),
            ];
            let prune = cleanup
                .git
                .output(repo, &prune_args, &[])
                .await
                .context("pruning isolated verification worktree")?;
            if !prune.status.success() {
                let remove_error = match removed {
                    Ok(output) => String::from_utf8_lossy(&output.stderr).trim().to_owned(),
                    Err(error) => format!("{error:#}"),
                };
                bail!(
                    "could not remove isolated verification worktree: {remove_error}; prune also failed: {}",
                    String::from_utf8_lossy(&prune.stderr).trim()
                );
            }
            directory_removal
                .context("Git worktree removal failed and its directory fallback also failed")?;
        }
    } else if cleanup.path.exists() {
        remove_dir_all_async(cleanup.path.clone()).await?;
    }
    if std::fs::symlink_metadata(&cleanup.path).is_ok() {
        remove_dir_all_async(cleanup.path.clone())
            .await
            .context("isolated cleanup reported success but its sandbox survived")?;
    }
    Ok(())
}

fn remove_isolated_cleanup_record(cleanup: &IsolatedCleanup) -> Result<()> {
    if let Some(record_path) = &cleanup.record_path {
        match std::fs::remove_file(record_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("removing isolated cleanup record {}", record_path.display())
                });
            }
        }
    }
    Ok(())
}

async fn remove_dir_all_async(path: PathBuf) -> Result<()> {
    let display = path.display().to_string();
    tokio::task::spawn_blocking(move || match std::fs::remove_dir_all(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    })
    .await
    .context("isolated cleanup worker failed")?
    .with_context(|| format!("removing isolated verification copy {display}"))
}

/// Run an operation in a fresh copy of an immutable checkpoint and remove the
/// copy afterwards. Git checkpoints use a detached temporary worktree so
/// commands that inspect repository metadata behave normally; internal
/// checkpoints are reconstructed directly from their content-addressed store.
/// Neither path writes to the destination workspace.
pub async fn with_isolated_checkpoint<T, F, Fut>(
    dir: &Path,
    reference: &str,
    state_root: &Path,
    operation: F,
) -> Result<T>
where
    F: FnOnce(PathBuf) -> Fut,
    Fut: Future<Output = Result<T>>,
{
    with_isolated_checkpoint_with_runner(
        dir,
        reference,
        state_root,
        GitRunner::default(),
        operation,
    )
    .await
}

async fn with_isolated_checkpoint_with_runner<T, F, Fut>(
    dir: &Path,
    reference: &str,
    state_root: &Path,
    git_runner: GitRunner,
    operation: F,
) -> Result<T>
where
    F: FnOnce(PathBuf) -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let (target, _) = parse_reference(reference)?;
    // The state root is control-plane data and is intentionally protected by
    // the default command sandbox. A verification worktree needs to create
    // compiler/test artifacts, so placing it below state_root makes ordinary
    // builds fail with EPERM and falsely look like baseline code failures.
    let parent = std::env::temp_dir().join("hi-verification-sandboxes");
    std::fs::create_dir_all(&parent)
        .with_context(|| format!("creating verification sandbox root {}", parent.display()))?;
    ensure!(
        std::fs::symlink_metadata(&parent)?.file_type().is_dir(),
        "verification sandbox root is not a real directory: {}",
        parent.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("securing verification sandbox root {}", parent.display()))?;
    }
    let sandbox = parent.join(format!(
        "verify-{}-{}-{}",
        std::process::id(),
        isolated_name_salt(),
        ISOLATED_ID.fetch_add(1, Ordering::Relaxed)
    ));
    if sandbox.exists() {
        remove_dir_all_async(sandbox.clone())
            .await
            .with_context(|| format!("removing stale sandbox {}", sandbox.display()))?;
    }

    let (mut guard, operation_root) = if crate::internal_snapshot::is_internal_id(target) {
        let (materialization, materialization_signal) = BlockingWorkCompletion::pair();
        let guard = IsolatedGuard::directory_after(sandbox.clone(), materialization)?;
        let source = dir.to_path_buf();
        let state = state_root.to_path_buf();
        let target = target.to_string();
        let destination = sandbox.clone();
        tokio::task::spawn_blocking(move || {
            // Kept alive for the complete blocking call. On success, failure,
            // panic, or cancellation before the task starts, dropping this
            // signal releases any detached cleanup waiter.
            let _materialization_signal = materialization_signal;
            crate::internal_snapshot::materialize(&source, &state, &target, &destination)
        })
        .await
        .context("isolated snapshot materialization task failed")??;
        (guard, sandbox)
    } else {
        let repo = toplevel_with_runner(dir, &git_runner)
            .await
            .context("not in a git work tree")?;
        let source = dir
            .canonicalize()
            .with_context(|| format!("canonicalizing workspace root {}", dir.display()))?;
        let repo = repo
            .canonicalize()
            .with_context(|| format!("canonicalizing Git root {}", repo.display()))?;
        let relative_root = source.strip_prefix(&repo).with_context(|| {
            format!(
                "workspace {} is outside Git root {}",
                source.display(),
                repo.display()
            )
        })?;
        // Arm recovery before `worktree add`: cancellation may arrive after
        // Git registers the worktree but before the command returns.
        let mut guard = IsolatedGuard::worktree(sandbox.clone(), repo.clone(), git_runner.clone())?;
        let (git_completion, git_signal) = BlockingWorkCompletion::pair();
        guard.wait_for(git_completion);
        let add_args = vec![
            OsString::from("worktree"),
            OsString::from("add"),
            OsString::from("--detach"),
            OsString::from("--force"),
            sandbox.clone().into_os_string(),
            OsString::from(target),
        ];
        let output = git_runner
            .output_with_completion(&repo, &add_args, &[], Some(git_signal))
            .await
            .context("creating isolated verification worktree")?;
        if !output.status.success() {
            let error = anyhow::anyhow!(
                "creating isolated verification worktree failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
            return match guard.cleanup().await {
                Ok(()) => Err(error),
                Err(cleanup) => Err(error.context(format!(
                    "isolated worktree recovery also failed: {cleanup:#}"
                ))),
            };
        }
        let operation_root = sandbox.join(relative_root);
        (guard, operation_root)
    };

    let operation_result = operation(operation_root).await;
    let cleanup_result = guard.cleanup().await;
    match (operation_result, cleanup_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Err(error), Err(cleanup)) => Err(error.context(format!(
            "isolated verification cleanup also failed: {cleanup:#}"
        ))),
    }
}

/// Encode a pre-turn checkpoint together with the immutable post-turn snapshot
/// it is allowed to replace.  Session files intentionally keep storing strings
/// so checkpoint ids written by 0.1 remain readable; only newly-created undo
/// records use this envelope.
pub fn sealed_reference(target: &str, expected_current: &str) -> String {
    format!(
        "{SEALED_REFERENCE_PREFIX}{}:{target}{expected_current}",
        target.len()
    )
}

/// Decode a checkpoint session reference. Historical bare ids have no seal and
/// are returned unchanged for migration compatibility.
pub fn parse_reference(reference: &str) -> Result<(&str, Option<&str>)> {
    let Some(encoded) = reference.strip_prefix(SEALED_REFERENCE_PREFIX) else {
        return Ok((reference, None));
    };
    let (target_len, payload) = encoded
        .split_once(':')
        .context("malformed sealed checkpoint reference")?;
    let target_len = target_len
        .parse::<usize>()
        .context("malformed sealed checkpoint target length")?;
    ensure_reference_boundary(payload, target_len)?;
    let (target, expected_current) = payload.split_at(target_len);
    if target.is_empty() || expected_current.is_empty() {
        bail!("malformed sealed checkpoint reference");
    }
    Ok((target, Some(expected_current)))
}

fn ensure_reference_boundary(payload: &str, offset: usize) -> Result<()> {
    if offset > payload.len() || !payload.is_char_boundary(offset) {
        bail!("malformed sealed checkpoint target length");
    }
    Ok(())
}

async fn git(dir: &Path, args: &[&str]) -> Result<Output> {
    let args = args.iter().map(OsString::from).collect::<Vec<_>>();
    GitRunner::default()
        .output(dir, &args, &[])
        .await
        .context("running git")
}

async fn git_indexed_with_completion(
    dir: &Path,
    index: &str,
    args: &[String],
    completion: BlockingWorkSignal,
) -> Result<Output> {
    let args = args.iter().map(OsString::from).collect::<Vec<_>>();
    GitRunner::default()
        .output_with_completion(dir, &args, &[("GIT_INDEX_FILE", index)], Some(completion))
        .await
        .context("running git")
}

async fn git_owned(dir: &Path, args: Vec<OsString>) -> Result<Output> {
    GitRunner::default()
        .output(dir, &args, &[])
        .await
        .context("running git")
}

#[derive(Clone, Debug)]
struct GitScope {
    root: PathBuf,
    repo: PathBuf,
    repo_relative: PathBuf,
}

async fn git_scope(dir: &Path) -> Result<GitScope> {
    let root = dir
        .canonicalize()
        .with_context(|| format!("canonicalizing workspace root {}", dir.display()))?;
    ensure!(root.is_dir(), "workspace root is not a directory");
    let output = git(&root, &["rev-parse", "--show-toplevel"]).await?;
    ensure_git_success(output.status.success(), &output.stderr, "locating Git root")?;
    let repo = PathBuf::from(
        String::from_utf8(output.stdout)
            .context("Git root is not valid UTF-8")?
            .trim(),
    )
    .canonicalize()
    .context("canonicalizing Git root")?;
    let repo_relative = root
        .strip_prefix(&repo)
        .with_context(|| {
            format!(
                "workspace {} is outside Git root {}",
                root.display(),
                repo.display()
            )
        })?
        .to_path_buf();
    Ok(GitScope {
        root,
        repo,
        repo_relative,
    })
}

async fn toplevel_with_runner(dir: &Path, runner: &GitRunner) -> Option<PathBuf> {
    let args = [
        OsString::from("rev-parse"),
        OsString::from("--show-toplevel"),
    ];
    let out = runner.output(dir, &args, &[]).await.ok()?;
    if !out.status.success() {
        return None;
    }
    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!p.is_empty()).then(|| PathBuf::from(p))
}

/// Snapshot the working tree of the repo containing `dir` into a dangling commit,
/// returning its SHA. `None` if `dir` isn't in a git work tree (so there's
/// nothing to checkpoint against).
pub async fn create(dir: &Path) -> Option<String> {
    match create_detailed(dir).await {
        CreateResult::Created(sha) => Some(sha),
        CreateResult::Unavailable(_) | CreateResult::Failed(_) => None,
    }
}

/// Snapshot with a diagnostic outcome suitable for an interactive safety gate.
pub async fn create_detailed(dir: &Path) -> CreateResult {
    create_detailed_with_state(dir, &default_state_root()).await
}

/// Snapshot using Git when available, otherwise a content-addressed internal
/// store rooted at `state_root`.
pub async fn create_detailed_with_state(dir: &Path, state_root: &Path) -> CreateResult {
    match create_git_detailed(dir, state_root).await {
        created @ CreateResult::Created(_) => created,
        git_result @ (CreateResult::Unavailable(_) | CreateResult::Failed(_)) => {
            let root = dir.to_path_buf();
            let state = state_root.to_path_buf();
            match tokio::task::spawn_blocking(move || {
                crate::internal_snapshot::create(&root, &state)
            })
            .await
            {
                Ok(Ok(id)) => CreateResult::Created(id),
                Ok(Err(error)) => CreateResult::Failed(format!(
                    "Git checkpoint unavailable ({git_result:?}); internal snapshot failed: {error:#}"
                )),
                Err(error) => {
                    CreateResult::Failed(format!("internal snapshot task failed: {error}"))
                }
            }
        }
    }
}

async fn create_git_detailed(dir: &Path, state_root: &Path) -> CreateResult {
    let probe = match git(dir, &["rev-parse", "--is-inside-work-tree"]).await {
        Ok(output) => output,
        Err(err) => return CreateResult::Unavailable(format!("Git is unavailable: {err:#}")),
    };
    if !probe.status.success() {
        return CreateResult::Unavailable(
            "the working directory is not inside a Git work tree".into(),
        );
    }
    let scope = match git_scope(dir).await {
        Ok(scope) => scope,
        Err(error) => return CreateResult::Failed(format!("invalid Git workspace: {error:#}")),
    };
    let scan_root = scope.root.clone();
    let scan_state = state_root.to_path_buf();
    let (preflight_bytes, preflight_entries) =
        match tokio::task::spawn_blocking(move || checkpoint_preflight(&scan_root, &scan_state))
            .await
        {
            Ok(Ok(totals)) => totals,
            Ok(Err(error)) => {
                return CreateResult::Failed(format!(
                    "workspace cannot be checkpointed: {error:#}"
                ));
            }
            Err(error) => {
                return CreateResult::Failed(format!("checkpoint preflight failed: {error}"));
            }
        };
    // Small ignored inputs (`.env`, vendored sources) stay covered by undo;
    // regenerable artifact trees and bulk data (gitignored `target/`, dataset
    // shards, model weights) are excluded so a data-heavy workspace neither
    // pours artifacts into `.git/objects` nor loses checkpointing entirely to
    // the size ceiling.
    let ignored_inputs = match qualifying_ignored_inputs(
        &scope.root,
        state_root,
        preflight_bytes,
        preflight_entries,
    )
    .await
    {
        Ok(paths) => paths,
        Err(error) => {
            return CreateResult::Failed(format!("workspace cannot be checkpointed: {error:#}"));
        }
    };
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!("hi-checkpoint-{}-{n}", std::process::id()));
    let mut temp_index = TemporaryFileGuard::git_index(tmp);
    let Some(index) = temp_index.path().to_str().map(str::to_owned) else {
        return CreateResult::Failed("temporary checkpoint index path is not valid UTF-8".into());
    };

    // Seed the throwaway index from HEAD so `add -A` is a fast incremental
    // (harmlessly fails in a repo with no commits yet).
    let (completion, signal) = BlockingWorkCompletion::pair();
    temp_index.wait_for(completion.clone());
    let _ = git_indexed_with_completion(
        &scope.root,
        &index,
        &["read-tree".into(), "HEAD".into()],
        signal,
    )
    .await;
    if !completion.is_finished() {
        return CreateResult::Failed(
            "git read-tree did not terminate cleanly; checkpoint index retained for recovery"
                .into(),
        );
    }
    // Limit the throwaway index update to the explicit workspace root. The
    // index is seeded from HEAD, so paths elsewhere in a containing monorepo
    // remain at HEAD even when they have unrelated dirty user changes. This
    // add respects gitignore; the qualifying ignored inputs enumerated above
    // are force-added separately below. The runtime state root is explicitly
    // excluded to avoid recursively checkpointing hi's own snapshots and
    // journals.
    let mut add_args = vec![
        "add".to_string(),
        "-A".to_string(),
        "--".to_string(),
        ".".to_string(),
    ];
    // `**/name` also matches at the root (gitignore glob semantics), and the
    // bare-directory form must be excluded too or `git add` names the ignored
    // directory itself and fails with "Use -f".
    for name in REGENERABLE_DIR_NAMES {
        add_args.push(format!(":(exclude,glob)**/{name}"));
        add_args.push(format!(":(exclude,glob)**/{name}/**"));
    }
    if let Some(relative_state) = contained_relative_path(&scope.root, state_root) {
        let relative_state = relative_state.to_string_lossy().replace('\\', "/");
        add_args.push(format!(":(exclude){relative_state}"));
        add_args.push(format!(":(exclude){relative_state}/**"));
    }
    let (completion, signal) = BlockingWorkCompletion::pair();
    temp_index.wait_for(completion);
    let add = match git_indexed_with_completion(&scope.root, &index, &add_args, signal).await {
        Ok(output) => output,
        Err(err) => {
            return CreateResult::Failed(format!("git add failed: {err:#}"));
        }
    };
    if !add.status.success() {
        return CreateResult::Failed(format!(
            "git add failed: {}",
            String::from_utf8_lossy(&add.stderr).trim()
        ));
    }
    if !ignored_inputs.is_empty() {
        // NUL-delimited literal pathspecs so globs/whitespace in ignored file
        // names cannot change what gets added.
        let mut pathspecs = Vec::new();
        for path in &ignored_inputs {
            pathspecs.extend_from_slice(b":(literal)");
            pathspecs.extend_from_slice(os_str_bytes(path.as_os_str()));
            pathspecs.push(0);
        }
        let mut spec_file = TemporaryFileGuard::file(
            std::env::temp_dir().join(format!("hi-checkpoint-pathspec-{}-{n}", std::process::id())),
        );
        if let Err(error) = std::fs::write(spec_file.path(), pathspecs) {
            return CreateResult::Failed(format!("writing checkpoint pathspec file: {error}"));
        }
        let (completion, signal) = BlockingWorkCompletion::pair();
        temp_index.wait_for(completion.clone());
        spec_file.wait_for(completion);
        let force_add = git_indexed_with_completion(
            &scope.root,
            &index,
            &[
                "add".into(),
                "-f".into(),
                "--pathspec-file-nul".into(),
                format!("--pathspec-from-file={}", spec_file.path().display()),
            ],
            signal,
        )
        .await;
        drop(spec_file);
        let force_add = match force_add {
            Ok(output) => output,
            Err(err) => {
                return CreateResult::Failed(format!("git add of ignored inputs failed: {err:#}"));
            }
        };
        if !force_add.status.success() {
            return CreateResult::Failed(format!(
                "git add of ignored inputs failed: {}",
                String::from_utf8_lossy(&force_add.stderr).trim()
            ));
        }
    }
    let (completion, signal) = BlockingWorkCompletion::pair();
    temp_index.wait_for(completion);
    let tree_out = match git_indexed_with_completion(
        &scope.root,
        &index,
        &["write-tree".into()],
        signal,
    )
    .await
    {
        Ok(output) => output,
        Err(err) => {
            return CreateResult::Failed(format!("git write-tree failed: {err:#}"));
        }
    };
    drop(temp_index);
    if !tree_out.status.success() {
        return CreateResult::Failed(format!(
            "git write-tree failed: {}",
            String::from_utf8_lossy(&tree_out.stderr).trim()
        ));
    }
    let tree = String::from_utf8_lossy(&tree_out.stdout).trim().to_string();

    let commit = match git(&scope.root, &["commit-tree", &tree, "-m", "hi checkpoint"]).await {
        Ok(output) => output,
        Err(err) => return CreateResult::Failed(format!("git commit-tree failed: {err:#}")),
    };
    if !commit.status.success() {
        return CreateResult::Failed(format!(
            "git commit-tree failed: {}",
            String::from_utf8_lossy(&commit.stderr).trim()
        ));
    }
    let sha = String::from_utf8_lossy(&commit.stdout).trim().to_string();
    if sha.is_empty() {
        CreateResult::Failed("git commit-tree returned an empty checkpoint id".into())
    } else {
        CreateResult::Created(sha)
    }
}

fn contained_relative_path(root: &Path, candidate: &Path) -> Option<PathBuf> {
    let root = root.canonicalize().ok()?;
    let candidate = candidate.canonicalize().ok()?;
    candidate
        .strip_prefix(root)
        .ok()
        .filter(|relative| !relative.as_os_str().is_empty())
        .map(Path::to_path_buf)
}

/// Ignored files larger than this are presumed regenerable artifacts or bulk
/// data (model weights, dataset shards) and are excluded from checkpoints and
/// the preflight ceiling. Small ignored files (`.env`, vendored sources,
/// configs) remain covered — they're plausible task inputs undo must revert.
const MAX_IGNORED_CHECKPOINT_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Directory names whose gitignored contents are always regenerable build or
/// dependency artifacts — never checkpointed and never counted toward the
/// ceiling. This is what let a Rust workspace pour hundreds of MB of `target/`
/// rlibs into `.git/objects` on every turn checkpoint before the exclusion.
pub(crate) const REGENERABLE_DIR_NAMES: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    ".jj",
    ".cargo-home",
    "target",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".turbo",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "hi-test-scratch",
];

pub(crate) fn regenerable_dir_name(name: &std::ffi::OsStr) -> bool {
    name.to_str()
        .is_some_and(|name| REGENERABLE_DIR_NAMES.contains(&name))
}

/// Walk the non-ignored workspace (what a plain `git add -A` ingests),
/// enforcing the entry/byte ceilings and bailing on nested repositories and
/// special files. Gitignored trees are skipped here — the qualifying subset of
/// ignored inputs is enumerated and accounted separately by
/// [`qualifying_ignored_inputs`]. Returns the running `(bytes, entries)`
/// totals so that second phase continues against the same ceilings.
fn checkpoint_preflight(root: &Path, state_root: &Path) -> Result<(u64, usize)> {
    let root = root
        .canonicalize()
        .with_context(|| format!("canonicalizing checkpoint root {}", root.display()))?;
    ensure!(root.is_dir(), "checkpoint root is not a directory");
    let state = state_root
        .canonicalize()
        .unwrap_or_else(|_| state_root.to_path_buf());
    let mut bytes = 0u64;
    let mut entries = 0usize;
    let filter_root = root.clone();
    let filter_state = state.clone();
    for result in ignore::WalkBuilder::new(&root)
        .hidden(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .parents(true)
        .filter_entry(move |entry| {
            // Prune the workspace's own VCS metadata; a *nested* repository's
            // metadata must flow through so the consumer can bail on it.
            if matches!(
                entry.file_name().to_str(),
                Some(".git" | ".hg" | ".svn" | ".jj")
            ) && entry.path().parent() == Some(filter_root.as_path())
            {
                return false;
            }
            // Regenerable artifact names are outside checkpoint scope whether
            // or not the repo ignores them — a bootstrap workspace without a
            // .gitignore must not checkpoint its build output. Name-based and
            // type-agnostic, matching the change ledger's prune and the add
            // pathspecs below (git pathspecs cannot distinguish a file from a
            // directory).
            if regenerable_dir_name(entry.file_name()) {
                return false;
            }
            // The runtime state root must never be checkpointed.
            if !entry.path_is_symlink() {
                let canonical = entry
                    .path()
                    .canonicalize()
                    .unwrap_or_else(|_| entry.path().to_path_buf());
                if canonical == filter_state || canonical.starts_with(&filter_state) {
                    return false;
                }
            }
            true
        })
        .build()
    {
        let entry =
            result.with_context(|| format!("walking checkpoint workspace {}", root.display()))?;
        let path = entry.path();
        if path == root {
            continue;
        }
        if matches!(
            path.file_name().and_then(|name| name.to_str()),
            Some(".git" | ".hg" | ".svn" | ".jj")
        ) {
            // A parent Git tree stores a nested repository as a gitlink and
            // cannot represent its dirty working files. Force the unified
            // creator to fall back to the no-follow internal backend instead
            // of claiming incomplete undo coverage.
            bail!(
                "nested repository metadata at {} is not representable by a Git checkpoint",
                path.display()
            );
        }
        let metadata = std::fs::symlink_metadata(path)
            .with_context(|| format!("reading checkpoint metadata {}", path.display()))?;
        entries = entries.saturating_add(1);
        ensure!(
            entries <= MAX_CHECKPOINT_ENTRIES,
            "workspace checkpoint exceeds {MAX_CHECKPOINT_ENTRIES} entries"
        );
        let file_type = metadata.file_type();
        if file_type.is_dir() {
            // The walker descends on its own.
        } else if file_type.is_file() {
            bytes = bytes.saturating_add(metadata.len());
        } else if file_type.is_symlink() {
            let target = std::fs::read_link(path)
                .with_context(|| format!("reading checkpoint symlink {}", path.display()))?;
            bytes = bytes.saturating_add(os_str_bytes(target.as_os_str()).len() as u64);
        } else {
            bail!(
                "cannot checkpoint special filesystem entry {}",
                path.display()
            );
        }
        ensure!(
            bytes <= MAX_CHECKPOINT_BYTES,
            "workspace checkpoint exceeds {} MiB ceiling",
            MAX_CHECKPOINT_BYTES / 1024 / 1024
        );
    }
    Ok((bytes, entries))
}

/// Enumerate the gitignored files that still belong in the checkpoint: small
/// plausible task inputs (`.env`, vendored sources) outside regenerable
/// artifact directories. Uses `git ls-files --others --ignored --directory` so
/// large ignored trees are seen as one entry and either skipped wholesale (by
/// artifact name) or expanded with the per-file size filter — bulk data and
/// build output never get hashed into `.git/objects`. Continues the byte/entry
/// accounting started by [`checkpoint_preflight`] against the same ceilings.
async fn qualifying_ignored_inputs(
    scope_root: &Path,
    state_root: &Path,
    bytes: u64,
    entries: usize,
) -> Result<Vec<PathBuf>> {
    let listing = git(
        scope_root,
        &[
            "ls-files",
            "-z",
            "--others",
            "--ignored",
            "--directory",
            "--exclude-standard",
        ],
    )
    .await?;
    ensure_git_success(
        listing.status.success(),
        &listing.stderr,
        "git ls-files --ignored",
    )?;
    let relatives: Vec<PathBuf> = listing
        .stdout
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
        .map(safe_git_relative)
        .collect::<Result<_>>()?;
    let root = scope_root.to_path_buf();
    let state = state_root
        .canonicalize()
        .unwrap_or_else(|_| state_root.to_path_buf());
    tokio::task::spawn_blocking(move || {
        let mut out = Vec::new();
        let mut bytes = bytes;
        let mut entries = entries;
        for relative in relatives {
            collect_ignored_input(&root, &relative, &state, &mut out, &mut bytes, &mut entries)?;
        }
        Ok(out)
    })
    .await
    .context("ignored-input scan task failed")?
}

fn collect_ignored_input(
    root: &Path,
    relative: &Path,
    state_root: &Path,
    out: &mut Vec<PathBuf>,
    bytes: &mut u64,
    entries: &mut usize,
) -> Result<()> {
    if relative.components().any(|component| match component {
        std::path::Component::Normal(name) => regenerable_dir_name(name),
        _ => false,
    }) {
        return Ok(());
    }
    let path = root.join(relative);
    let Ok(metadata) = std::fs::symlink_metadata(&path) else {
        // Deleted between listing and scan — a normal race.
        return Ok(());
    };
    if !metadata.file_type().is_symlink() {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        if canonical == state_root || canonical.starts_with(state_root) {
            return Ok(());
        }
    }
    let file_type = metadata.file_type();
    let added_bytes = if file_type.is_dir() {
        for child in std::fs::read_dir(&path)
            .with_context(|| format!("reading ignored directory {}", path.display()))?
        {
            let child = child.with_context(|| format!("walking {}", path.display()))?;
            collect_ignored_input(
                root,
                &relative.join(child.file_name()),
                state_root,
                out,
                bytes,
                entries,
            )?;
        }
        return Ok(());
    } else if file_type.is_file() {
        if metadata.len() > MAX_IGNORED_CHECKPOINT_FILE_BYTES {
            return Ok(());
        }
        metadata.len()
    } else if file_type.is_symlink() {
        std::fs::read_link(&path)
            .map(|target| os_str_bytes(target.as_os_str()).len() as u64)
            .unwrap_or(0)
    } else {
        // Ignored sockets/fifos are simply not checkpointable inputs.
        return Ok(());
    };
    *entries = entries.saturating_add(1);
    ensure!(
        *entries <= MAX_CHECKPOINT_ENTRIES,
        "workspace checkpoint exceeds {MAX_CHECKPOINT_ENTRIES} entries"
    );
    *bytes = bytes.saturating_add(added_bytes);
    ensure!(
        *bytes <= MAX_CHECKPOINT_BYTES,
        "workspace checkpoint exceeds {} MiB ceiling",
        MAX_CHECKPOINT_BYTES / 1024 / 1024
    );
    out.push(relative.to_path_buf());
    Ok(())
}

#[cfg(unix)]
fn os_str_bytes(value: &OsStr) -> &[u8] {
    use std::os::unix::ffi::OsStrExt;
    value.as_bytes()
}

#[cfg(not(unix))]
fn os_str_bytes(value: &OsStr) -> &[u8] {
    value.to_str().unwrap_or("").as_bytes()
}

/// A unified diff of the working tree (of the repo containing `dir`) against
/// checkpoint `target` — everything that changed since that checkpoint, including
/// new and deleted files. Best-effort: `None` if not in a work tree, git errors,
/// or nothing changed. Used to show a reviewer what a turn actually did.
pub async fn diff(dir: &Path, target: &str) -> Option<String> {
    diff_with_state(dir, target, &default_state_root()).await
}

pub async fn diff_with_state(dir: &Path, target: &str, state_root: &Path) -> Option<String> {
    if crate::internal_snapshot::is_internal_id(target) {
        let root = dir.to_path_buf();
        let state = state_root.to_path_buf();
        let target = target.to_string();
        return tokio::task::spawn_blocking(move || {
            crate::internal_snapshot::diff(&root, &state, &target)
        })
        .await
        .ok()
        .and_then(Result::ok)
        .flatten();
    }
    let scope = git_scope(dir).await.ok()?;
    // Snapshot the current tree (captures untracked files too, via `add -A`) and
    // diff the checkpoint against it — the same technique `restore` uses, so new
    // files show up rather than being invisible to a bare `git diff <commit>`.
    let current = match create_git_detailed(&scope.root, state_root).await {
        CreateResult::Created(id) => id,
        CreateResult::Unavailable(_) | CreateResult::Failed(_) => return None,
    };
    let out = git(
        &scope.root,
        &[
            "diff",
            "--no-renames",
            "--relative",
            target,
            &current,
            "--",
            ".",
        ],
    )
    .await
    .ok()?;
    if !out.status.success() {
        return None;
    }
    let patch = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!patch.is_empty()).then_some(patch)
}

/// Restore the working tree to checkpoint `target`, undoing every change made
/// since. Returns the number of files restored or removed.
pub async fn restore(dir: &Path, target: &str) -> Result<usize> {
    restore_with_state(dir, target, &default_state_root()).await
}

pub async fn restore_with_state(dir: &Path, target: &str, state_root: &Path) -> Result<usize> {
    if crate::internal_snapshot::is_internal_id(target) {
        return restore_internal_with_state(dir, target, None, state_root, || {}, || {}).await;
    }
    let scope = git_scope(dir).await.context("not in a Git work tree")?;
    // Snapshot the current state and diff against the target, then prepare all
    // blobs/symlink targets before the transaction touches the workspace.
    let current = match create_git_detailed(&scope.root, state_root).await {
        CreateResult::Created(id) => id,
        CreateResult::Unavailable(reason) | CreateResult::Failed(reason) => {
            bail!("couldn't snapshot current state: {reason}")
        }
    };
    let (plan, changed) = prepare_git_restore(&scope, target, &current, state_root).await?;
    if let Some(plan) = plan {
        run_uncancellable_filesystem_boundary(move || plan.commit())?;
    }
    Ok(changed)
}

/// Prepare an internal restore off the async runtime, but retain ownership of
/// the live-tree commit in an explicit uncancellable filesystem boundary. A
/// cancelled `spawn_blocking` join detaches its worker, so committing inside a
/// normally-awaited blocking task would let `/undo` keep changing the
/// workspace after its async future had returned cancellation. Preparation is
/// read-only and remains cancellable. Once finalization starts, a multithreaded
/// Tokio runtime replaces the blocked scheduler worker and the caller cannot
/// observe cancellation until recovery, revalidation, and commit are settled.
async fn restore_internal_with_state<F, G>(
    dir: &Path,
    target: &str,
    expected_current: Option<&str>,
    state_root: &Path,
    after_prepare: F,
    before_commit: G,
) -> Result<usize>
where
    F: FnOnce() + Send + 'static,
    G: FnOnce() + Send + 'static,
{
    // Journal recovery may repair an interrupted transaction, so perform it
    // before creating the detachable preparation worker. It uses the same
    // uncancellable boundary as commit: recovery may mutate the live tree and
    // must settle before this API can report cancellation.
    run_uncancellable_filesystem_boundary(|| {
        crate::transaction::recover_workspace_transactions(dir, state_root)
    })?;
    let root = dir.to_path_buf();
    let state = state_root.to_path_buf();
    let target = target.to_string();
    let expected = expected_current.map(str::to_string);
    let expected_for_commit = expected.clone();
    let (plan, changed) = tokio::task::spawn_blocking(move || {
        let prepared = crate::internal_snapshot::prepare_restore_after_recovery(
            &root,
            &state,
            &target,
            expected.as_deref(),
        );
        after_prepare();
        prepared
    })
    .await
    .context("internal restore preparation task failed")??;
    // Give cancellation one final scheduling point while all completed work is
    // still read-only. From the next statement through commit there is no
    // detachable mutation worker and no cancellation boundary.
    tokio::task::yield_now().await;
    run_uncancellable_filesystem_boundary(move || -> Result<()> {
        // Another process may have left a journal while the detachable scan
        // was in flight. Recover again inside the owned commit boundary;
        // MutationPlan's preimage seal rejects overlap changed by recovery.
        crate::transaction::recover_workspace_transactions(dir, state_root)?;
        if let Some(expected) = expected_for_commit {
            crate::internal_snapshot::ensure_current_matches(dir, state_root, &expected)?;
        }
        before_commit();
        if let Some(plan) = plan {
            plan.commit()?;
        }
        Ok(())
    })?;
    Ok(changed)
}

/// Execute a synchronous filesystem transaction without starving unrelated
/// async work or allowing task cancellation to detach live-tree mutations.
///
/// `block_in_place` temporarily hands the scheduler worker's async duties to a
/// replacement thread. A current-thread runtime has no replacement-worker
/// facility, so it intentionally falls back to the same inline, uncancellable
/// semantics; callers still never observe completion while writes are active.
fn run_uncancellable_filesystem_boundary<T>(operation: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(handle)
            if matches!(
                handle.runtime_flavor(),
                tokio::runtime::RuntimeFlavor::MultiThread
            ) =>
        {
            tokio::task::block_in_place(operation)
        }
        _ => operation(),
    }
}

async fn prepare_git_restore(
    scope: &GitScope,
    target: &str,
    current: &str,
    state_root: &Path,
) -> Result<(Option<crate::transaction::MutationPlan>, usize)> {
    use crate::transaction::{MutationPlan, RestoreMutation};

    let diff = git(
        &scope.root,
        &[
            "diff",
            "--no-renames",
            "--name-status",
            "-z",
            "--relative",
            target,
            current,
            "--",
            ".",
        ],
    )
    .await?;
    ensure_git_success(diff.status.success(), &diff.stderr, "git restore diff")?;

    let mut fields = diff.stdout.split(|byte| *byte == 0);
    let mut mutations = Vec::new();
    while let Some(status) = fields.next() {
        if status.is_empty() {
            break;
        }
        let path = fields
            .next()
            .context("malformed NUL-delimited Git diff (missing path)")?;
        ensure!(!path.is_empty(), "malformed Git diff (empty path)");
        let relative = safe_git_relative(path)?;
        let postimage = match status {
            b"A" => None,
            b"M" | b"D" | b"T" => Some(git_restore_node(scope, target, &relative).await?),
            _ => bail!(
                "unsupported Git restore status {:?} for {}",
                String::from_utf8_lossy(status),
                relative.display()
            ),
        };
        mutations.push(RestoreMutation {
            path: relative,
            postimage,
        });
    }
    if mutations.is_empty() {
        return Ok((None, 0));
    }
    let changed = mutations.len();
    let plan = MutationPlan::new_restore_with_state(&scope.root, state_root, mutations)?;
    Ok((Some(plan), changed))
}

async fn git_restore_node(
    scope: &GitScope,
    checkpoint: &str,
    relative: &Path,
) -> Result<crate::transaction::RestoreNode> {
    use crate::transaction::RestoreNode;

    let repository_path = scope.repo_relative.join(relative);
    let tree = git_owned(
        &scope.repo,
        vec![
            "ls-tree".into(),
            "-z".into(),
            "--full-tree".into(),
            checkpoint.into(),
            "--".into(),
            repository_path.as_os_str().to_os_string(),
        ],
    )
    .await?;
    ensure_git_success(tree.status.success(), &tree.stderr, "git ls-tree")?;
    let records: Vec<&[u8]> = tree
        .stdout
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
        .collect();
    ensure!(
        records.len() == 1,
        "checkpoint has {} tree entries for {}",
        records.len(),
        relative.display()
    );
    let separator = records[0]
        .iter()
        .position(|byte| *byte == b'\t')
        .context("malformed git ls-tree output")?;
    let (metadata, returned_path) = records[0].split_at(separator);
    let returned_path = &returned_path[1..];
    let fields: Vec<&[u8]> = metadata.split(|byte| *byte == b' ').collect();
    ensure!(fields.len() == 3, "malformed git ls-tree metadata");
    let returned_path = safe_git_relative(returned_path)?;
    ensure!(
        returned_path == repository_path,
        "Git returned an out-of-scope tree path {}",
        returned_path.display()
    );
    ensure!(fields[1] == b"blob", "unsupported non-blob Git tree entry");
    let object = std::str::from_utf8(fields[2]).context("invalid Git object id")?;
    let blob = git(&scope.repo, &["cat-file", "blob", object]).await?;
    ensure_git_success(blob.status.success(), &blob.stderr, "git cat-file")?;
    match fields[0] {
        b"100644" => Ok(RestoreNode::File {
            bytes: blob.stdout,
            mode: 0o644,
        }),
        b"100755" => Ok(RestoreNode::File {
            bytes: blob.stdout,
            mode: 0o755,
        }),
        b"120000" => Ok(RestoreNode::Symlink {
            target: path_from_bytes(&blob.stdout),
        }),
        mode => bail!(
            "unsupported Git mode {} for {}",
            String::from_utf8_lossy(mode),
            relative.display()
        ),
    }
}

fn safe_git_relative(bytes: &[u8]) -> Result<PathBuf> {
    let path = path_from_bytes(bytes);
    ensure!(
        !path.as_os_str().is_empty()
            && !path.is_absolute()
            && path
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "Git returned unsafe workspace path {:?}",
        path
    );
    Ok(path)
}

#[cfg(unix)]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStringExt;
    PathBuf::from(OsString::from_vec(bytes.to_vec()))
}

#[cfg(not(unix))]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(bytes).into_owned())
}

/// Restore only if the workspace still equals `expected_current`. This seals an
/// undo record against post-turn user/editor changes.
pub async fn restore_sealed_with_state(
    dir: &Path,
    target: &str,
    expected_current: &str,
    state_root: &Path,
) -> Result<usize> {
    if crate::internal_snapshot::is_internal_id(target)
        && crate::internal_snapshot::is_internal_id(expected_current)
    {
        return restore_internal_with_state(
            dir,
            target,
            Some(expected_current),
            state_root,
            || {},
            || {},
        )
        .await;
    }
    // Git callers seal against an immutable tree, prepare all postimages, then
    // sample the tree once more before the transaction's own per-node digest
    // revalidation. A changed file is therefore never silently overwritten.
    let scope = git_scope(dir).await.context("not in a Git work tree")?;
    let current = create_git_detailed(&scope.root, state_root).await;
    match current {
        CreateResult::Created(id) => {
            ensure!(
                !crate::internal_snapshot::is_internal_id(expected_current),
                "checkpoint backend changed after the turn"
            );
            ensure!(
                git_tree_id(&scope.root, &id).await?
                    == git_tree_id(&scope.root, expected_current).await?,
                "undo conflict: workspace changed externally after the turn (expected {expected_current}, found {id})"
            );
            let (plan, changed) = prepare_git_restore(&scope, target, &id, state_root).await?;
            let observed = match create_git_detailed(&scope.root, state_root).await {
                CreateResult::Created(observed) => observed,
                CreateResult::Unavailable(reason) | CreateResult::Failed(reason) => {
                    bail!("could not revalidate undo restore: {reason}")
                }
            };
            ensure!(
                git_tree_id(&scope.root, &observed).await? == git_tree_id(&scope.root, &id).await?,
                "undo conflict: workspace changed externally while preparing restore"
            );
            if let Some(plan) = plan {
                run_uncancellable_filesystem_boundary(move || plan.commit())?;
            }
            Ok(changed)
        }
        CreateResult::Unavailable(reason) | CreateResult::Failed(reason) => {
            bail!("could not seal undo restore: {reason}")
        }
    }
}

async fn git_tree_id(dir: &Path, checkpoint: &str) -> Result<String> {
    let spec = format!("{checkpoint}^{{tree}}");
    let output = git(dir, &["rev-parse", &spec]).await?;
    ensure_git_success(
        output.status.success(),
        &output.stderr,
        "git rev-parse tree",
    )?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn ensure_git_success(success: bool, stderr: &[u8], operation: &str) -> Result<()> {
    if success {
        Ok(())
    } else {
        bail!(
            "{operation} failed: {}",
            String::from_utf8_lossy(stderr).trim()
        )
    }
}

/// Default persistent state directory used by compatibility APIs. New runtimes
/// should pass their explicit state root to the `*_with_state` functions.
pub fn default_state_root() -> PathBuf {
    if let Some(path) = std::env::var_os("HI_STATE_ROOT") {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("XDG_STATE_HOME") {
        return PathBuf::from(path).join("hi");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".local").join("state").join("hi");
    }
    std::env::temp_dir().join("hi-state")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn write_executable(path: &Path, contents: &str) {
        use std::os::unix::fs::PermissionsExt;

        std::fs::write(path, contents).unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[cfg(unix)]
    fn process_is_alive(pid: i32) -> bool {
        // SAFETY: signal 0 only probes process existence and does not mutate
        // memory or deliver a signal.
        unsafe { libc::kill(pid, 0) == 0 }
    }

    #[cfg(unix)]
    async fn wait_for_process_exit(pid: i32) -> bool {
        for _ in 0..200 {
            if !process_is_alive(pid) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        false
    }

    #[cfg(unix)]
    async fn wait_for_path(path: &Path, should_exist: bool) -> bool {
        for _ in 0..200 {
            if path.exists() == should_exist {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        false
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn raw_git_runner_preserves_binary_output_and_disables_interaction() {
        let temp = tempfile::tempdir().unwrap();
        let fake_git = temp.path().join("fake-git");
        write_executable(
            &fake_git,
            "#!/bin/sh\n\
             printf 'caf\\303\\251\\000\\377'\n\
             printf '%s|%s|%s|%s' \"$GIT_TERMINAL_PROMPT\" \"$GCM_INTERACTIVE\" \"$PAGER\" \"$GIT_PAGER\" >&2\n",
        );
        let runner = GitRunner::new(&fake_git, Duration::from_secs(2));

        let output = runner.output(temp.path(), &[], &[]).await.unwrap();

        assert!(output.status.success());
        assert_eq!(output.stdout, b"caf\xc3\xa9\0\xff");
        assert_eq!(output.stderr, b"0|never|cat|cat");
    }

    #[tokio::test]
    async fn cancelling_checkpoint_temp_scope_removes_index_lock_and_pathspec() {
        let temp = tempfile::tempdir().unwrap();
        let index = temp.path().join("hi-checkpoint-index");
        let index_lock = path_with_suffix(&index, ".lock");
        let pathspec = temp.path().join("hi-checkpoint-pathspec");
        let task_index = index.clone();
        let task_index_lock = index_lock.clone();
        let task_pathspec = pathspec.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let _index_cleanup = TemporaryFileGuard::git_index(task_index.clone());
            let _pathspec_cleanup = TemporaryFileGuard::file(task_pathspec.clone());
            std::fs::write(task_index, b"index").unwrap();
            std::fs::write(task_index_lock, b"lock left by killed git").unwrap();
            std::fs::write(task_pathspec, b"pathspec").unwrap();
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
        });

        started_rx.await.unwrap();
        assert!(index.exists());
        assert!(index_lock.exists());
        assert!(pathspec.exists());
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());

        assert!(!index.exists(), "temporary checkpoint index survived");
        assert!(!index_lock.exists(), "temporary Git index lock survived");
        assert!(!pathspec.exists(), "temporary pathspec file survived");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn temporary_file_cleanup_waits_for_child_completion() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("owned-by-git");
        std::fs::write(&path, b"temporary index").unwrap();
        let (completion, signal) = BlockingWorkCompletion::pair();
        let mut guard = TemporaryFileGuard::file(path.clone());
        guard.wait_for(completion.clone());

        drop(guard);
        tokio::time::timeout(Duration::from_secs(2), async {
            while completion.waiter_count() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("temporary-file cleanup did not wait for its owner");
        assert!(
            path.exists(),
            "temporary file was removed before its owner completed"
        );

        drop(signal);
        assert!(
            wait_for_path(&path, false).await,
            "temporary file survived after its owner completed"
        );
    }

    #[tokio::test]
    async fn cancelled_internal_restore_drops_prepared_plan_without_committing() {
        for sealed in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let workspace = temp.path().join("workspace");
            let state = temp.path().join("state");
            std::fs::create_dir_all(&workspace).unwrap();
            std::fs::create_dir_all(&state).unwrap();
            let file = workspace.join("file");
            std::fs::write(&file, "before").unwrap();
            let before = crate::internal_snapshot::create(&workspace, &state).unwrap();
            std::fs::write(&file, "after").unwrap();
            let expected =
                sealed.then(|| crate::internal_snapshot::create(&workspace, &state).unwrap());

            let task_workspace = workspace.clone();
            let task_state = state.clone();
            let task_before = before.clone();
            let task_expected = expected.clone();
            let (prepared_tx, prepared_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
            let worker_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let task_worker_done = worker_done.clone();
            let task = tokio::spawn(async move {
                restore_internal_with_state(
                    &task_workspace,
                    &task_before,
                    task_expected.as_deref(),
                    &task_state,
                    move || {
                        let _ = prepared_tx.send(());
                        release_rx
                            .recv_timeout(Duration::from_secs(5))
                            .expect("test did not release prepared restore worker");
                        task_worker_done.store(true, Ordering::Release);
                    },
                    || {},
                )
                .await
            });

            tokio::time::timeout(Duration::from_secs(5), prepared_rx)
                .await
                .expect("restore preparation did not reach the test barrier")
                .expect("restore preparation task exited before the test barrier");
            task.abort();
            let cancellation = tokio::time::timeout(Duration::from_secs(1), task)
                .await
                .expect("restore cancellation waited for its detached worker")
                .expect_err("restore unexpectedly completed before cancellation");
            assert!(cancellation.is_cancelled());
            assert_eq!(
                std::fs::read_to_string(&file).unwrap(),
                "after",
                "{sealed:?} restore committed before its cancelled API returned"
            );

            release_tx.send(()).unwrap();
            tokio::time::timeout(Duration::from_secs(5), async {
                while !worker_done.load(Ordering::Acquire) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("detached restore preparation worker did not finish");
            tokio::task::yield_now().await;
            assert_eq!(
                std::fs::read_to_string(&file).unwrap(),
                "after",
                "{sealed:?} restore committed after its cancelled API returned"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn internal_restore_commit_boundary_is_owned_and_does_not_starve_runtime() {
        let temp = tempfile::tempdir().unwrap();
        let workspace = temp.path().join("workspace");
        let state = temp.path().join("state");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        let file = workspace.join("file");
        std::fs::write(&file, "before").unwrap();
        let before = crate::internal_snapshot::create(&workspace, &state).unwrap();
        std::fs::write(&file, "after").unwrap();
        let after = crate::internal_snapshot::create(&workspace, &state).unwrap();

        let task_workspace = workspace.clone();
        let task_state = state.clone();
        let task_before = before.clone();
        let task_after = after.clone();
        let (commit_tx, commit_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let mut task = tokio::spawn(async move {
            restore_internal_with_state(
                &task_workspace,
                &task_before,
                Some(&task_after),
                &task_state,
                || {},
                move || {
                    let _ = commit_tx.send(());
                    release_rx
                        .recv_timeout(Duration::from_secs(5))
                        .expect("test did not release owned restore commit");
                },
            )
            .await
        });

        tokio::time::timeout(Duration::from_secs(5), commit_rx)
            .await
            .expect("restore did not enter its commit boundary")
            .expect("restore exited before its commit boundary");
        tokio::time::timeout(
            Duration::from_millis(100),
            tokio::time::sleep(Duration::from_millis(1)),
        )
        .await
        .expect("the synchronous commit boundary starved the single-worker runtime");

        task.abort();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut task)
                .await
                .is_err(),
            "task cancellation returned while an owned commit could still mutate the workspace"
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "after");

        release_tx.send(()).unwrap();
        let joined = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("owned restore commit did not settle after release");
        match joined {
            Ok(result) => assert_eq!(result.unwrap(), 1),
            Err(error) => assert!(error.is_cancelled()),
        }
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "before");
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "before",
            "restore kept mutating after its task settled"
        );
    }

    #[cfg(unix)]
    #[test]
    fn isolated_cleanup_record_round_trips_non_utf8_paths_losslessly() {
        use std::os::unix::ffi::OsStringExt;

        let path = PathBuf::from(OsString::from_vec(b"/tmp/hi-sandbox-\xff".to_vec()));
        let repo = PathBuf::from(OsString::from_vec(b"/tmp/hi-repo-\xfe".to_vec()));
        let record = IsolatedCleanupRecord {
            version: 1,
            owner_pid: 123,
            path: IsolatedRecordPath::encode(&path).unwrap(),
            git_repo: Some(IsolatedRecordPath::encode(&repo).unwrap()),
            registered_worktree: true,
        };

        let encoded = serde_json::to_vec(&record).unwrap();
        let decoded: IsolatedCleanupRecord = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded.path.decode().unwrap(), path);
        assert_eq!(
            decoded.git_repo.unwrap().decode().unwrap(),
            repo,
            "non-UTF-8 repository path changed in durable cleanup record"
        );
    }

    #[cfg(unix)]
    #[test]
    fn worktree_guard_record_failure_does_not_run_git_cleanup() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repo");
        let sandbox = temp.path().join("verify-guard");
        let fake_git = temp.path().join("fake-git");
        let invoked = path_with_suffix(&fake_git, ".called");
        std::fs::create_dir(&repo).unwrap();
        write_executable(
            &fake_git,
            "#!/bin/sh\nprintf called > \"$0.called\"\nexit 0\n",
        );
        // Force breadcrumb persistence to fail before `git worktree add` can
        // begin. Constructor unwinding must leave Drop disarmed.
        std::fs::write(cleanup_record_path(&sandbox).unwrap(), b"occupied").unwrap();

        let error = IsolatedGuard::worktree(
            sandbox,
            repo,
            GitRunner::new(fake_git, Duration::from_secs(1)),
        )
        .err()
        .expect("an occupied cleanup-record path must reject construction");
        assert!(error.to_string().contains("creating"));
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !invoked.exists(),
            "constructor failure unexpectedly ran Git cleanup/prune"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cleanup_record_unlink_failure_disarms_worktree_guard() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repo");
        let sandbox = temp.path().join("verify-unlink-failure");
        let record = temp.path().join("retained-cleanup-record");
        let fake_git = temp.path().join("fake-git");
        let calls = repo.join("calls");
        std::fs::create_dir(&repo).unwrap();
        std::fs::create_dir(&sandbox).unwrap();
        // remove_file on a directory fails portably, deterministically
        // exercising a breadcrumb-unlink error after physical cleanup.
        std::fs::create_dir(&record).unwrap();
        write_executable(
            &fake_git,
            "#!/bin/sh\n\
             repo=$2\n\
             shift 2\n\
             printf '%s:%s\\n' \"$1\" \"$2\" >> \"$repo/calls\"\n\
             if [ \"$1:$2\" = 'worktree:remove' ]; then\n\
               rm -rf \"$4\"\n\
               exit 0\n\
             fi\n\
             exit 2\n",
        );
        let mut guard = IsolatedGuard {
            path: sandbox.clone(),
            git_repo: Some(repo),
            registered_worktree: true,
            git: GitRunner::new(fake_git, Duration::from_secs(1)),
            blocking_work: None,
            record_path: Some(record.clone()),
            cleaned: false,
        };

        let error = guard
            .cleanup()
            .await
            .expect_err("cleanup-record unlink unexpectedly succeeded");

        assert!(
            error
                .to_string()
                .contains("removing isolated cleanup record"),
            "unexpected cleanup error: {error:#}"
        );
        assert!(
            !sandbox.exists(),
            "physical worktree cleanup did not finish"
        );
        assert!(
            record.exists(),
            "failed cleanup breadcrumb was not retained"
        );
        assert!(guard.cleaned, "physically cleaned guard remained armed");
        drop(guard);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            std::fs::read_to_string(calls).unwrap(),
            "worktree:remove\n",
            "Drop repeated destructive Git cleanup after breadcrumb unlink failure"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn raw_git_timeout_kills_and_reaps_its_process_group() {
        let temp = tempfile::tempdir().unwrap();
        let fake_git = temp.path().join("fake-git");
        let parent_pid = temp.path().join("parent.pid");
        let child_pid = temp.path().join("child.pid");
        write_executable(
            &fake_git,
            "#!/bin/sh\n\
             shift 2\n\
             printf '%s' \"$$\" > \"$1\"\n\
             sleep 60 &\n\
             printf '%s' \"$!\" > \"$2\"\n\
             wait\n",
        );
        let runner = GitRunner::new(&fake_git, Duration::from_secs(1));
        let args = [
            parent_pid.clone().into_os_string(),
            child_pid.clone().into_os_string(),
        ];
        let started = std::time::Instant::now();

        let error = runner.output(temp.path(), &args, &[]).await.unwrap_err();

        assert!(error.to_string().contains("timed out"), "{error:#}");
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "timeout path exceeded its bounded reap grace"
        );
        let parent = std::fs::read_to_string(&parent_pid)
            .unwrap()
            .parse::<i32>()
            .unwrap();
        let child = std::fs::read_to_string(&child_pid)
            .unwrap()
            .parse::<i32>()
            .unwrap();
        assert!(
            wait_for_process_exit(parent).await,
            "Git parent survived timeout"
        );
        assert!(
            wait_for_process_exit(child).await,
            "Git descendant survived timeout"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_git_runner_reaps_before_releasing_completion() {
        let temp = tempfile::tempdir().unwrap();
        let fake_git = temp.path().join("fake-git");
        let parent_pid = temp.path().join("parent.pid");
        let child_pid = temp.path().join("child.pid");
        write_executable(
            &fake_git,
            "#!/bin/sh\n\
             shift 2\n\
             printf '%s' \"$$\" > \"$1\"\n\
             sleep 60 &\n\
             printf '%s' \"$!\" > \"$2\"\n\
             wait\n",
        );
        let runner = GitRunner::new(&fake_git, Duration::from_secs(30));
        let args = [
            parent_pid.clone().into_os_string(),
            child_pid.clone().into_os_string(),
        ];
        let (completion, signal) = BlockingWorkCompletion::pair();
        let git_dir = temp.path().to_path_buf();
        let task = tokio::spawn(async move {
            runner
                .output_with_completion(&git_dir, &args, &[], Some(signal))
                .await
        });
        assert!(
            wait_for_path(&child_pid, true).await,
            "fake Git never started"
        );
        let parent = std::fs::read_to_string(&parent_pid)
            .unwrap()
            .parse::<i32>()
            .unwrap();
        let child = std::fs::read_to_string(&child_pid)
            .unwrap()
            .parse::<i32>()
            .unwrap();

        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(5), completion.wait())
            .await
            .expect("cancelled Git child was never reaped")
            .expect("Git reaper completion waiter failed");
        assert!(
            !process_is_alive(parent),
            "completion was released before the direct Git child was reaped"
        );
        assert!(
            wait_for_process_exit(child).await,
            "Git descendant survived cancellation"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_worktree_add_kills_git_tree_and_schedules_recovery() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repo");
        let state = temp.path().join("state");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        let fake_git = temp.path().join("fake-git");
        let registered = repo.join("registered");
        let parent_pid = repo.join("add-parent.pid");
        let child_pid = repo.join("add-child.pid");
        let sandbox_record = repo.join("sandbox.path");
        write_executable(
            &fake_git,
            "#!/bin/sh\n\
             repo=$2\n\
             shift 2\n\
             if [ \"$1:$2\" = 'rev-parse:--show-toplevel' ]; then\n\
               printf '%s\\n' \"$repo\"\n\
               exit 0\n\
             fi\n\
             if [ \"$1:$2\" = 'worktree:add' ]; then\n\
               sandbox=$5\n\
               : > \"$repo/registered\"\n\
               printf '%s' \"$sandbox\" > \"$repo/sandbox.path\"\n\
               mkdir -p \"$sandbox\"\n\
               printf '%s' \"$$\" > \"$repo/add-parent.pid\"\n\
               sleep 60 &\n\
               printf '%s' \"$!\" > \"$repo/add-child.pid\"\n\
               wait\n\
             fi\n\
             if [ \"$1:$2\" = 'worktree:remove' ]; then\n\
               rm -f \"$repo/registered\"\n\
               rm -rf \"$4\"\n\
               exit 0\n\
             fi\n\
             if [ \"$1:$2\" = 'worktree:prune' ]; then\n\
               rm -f \"$repo/registered\"\n\
               exit 0\n\
             fi\n\
             exit 2\n",
        );
        let runner = GitRunner::new(&fake_git, Duration::from_secs(30));
        let repo_for_task = repo.clone();
        let state_for_task = state.clone();
        let task = tokio::spawn(async move {
            with_isolated_checkpoint_with_runner(
                &repo_for_task,
                "checkpoint-id",
                &state_for_task,
                runner,
                |_| async { Ok(()) },
            )
            .await
        });
        assert!(
            wait_for_path(&registered, true).await,
            "fake Git never registered the worktree"
        );
        assert!(wait_for_path(&child_pid, true).await);
        let parent = std::fs::read_to_string(&parent_pid)
            .unwrap()
            .parse::<i32>()
            .unwrap();
        let child = std::fs::read_to_string(&child_pid)
            .unwrap()
            .parse::<i32>()
            .unwrap();
        let sandbox = PathBuf::from(std::fs::read_to_string(&sandbox_record).unwrap());
        let cancelled_at = std::time::Instant::now();

        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());

        assert!(
            cancelled_at.elapsed() < Duration::from_secs(1),
            "dropping the isolated guard blocked on cleanup"
        );
        assert!(
            wait_for_path(&registered, false).await,
            "recovery did not unregister the interrupted add"
        );
        assert!(
            wait_for_process_exit(parent).await,
            "worktree-add Git process survived cancellation"
        );
        assert!(
            wait_for_process_exit(child).await,
            "worktree-add descendant survived cancellation"
        );
        assert!(wait_for_path(&sandbox, false).await, "sandbox survived");
    }

    #[tokio::test]
    async fn cancelling_isolated_operation_unregisters_worktree_off_drop_path() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repo");
        let state = temp.path().join("state");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        sh(
            &repo,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(repo.join("value.txt"), "checkpoint\n").unwrap();
        let checkpoint = create(&repo).await.expect("checkpoint");
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let repo_for_task = repo.clone();
        let state_for_task = state.clone();
        let task = tokio::spawn(async move {
            with_isolated_checkpoint(
                &repo_for_task,
                &checkpoint,
                &state_for_task,
                move |isolated| async move {
                    let _ = started_tx.send(isolated);
                    std::future::pending::<Result<()>>().await
                },
            )
            .await
        });
        let sandbox = tokio::time::timeout(Duration::from_secs(5), started_rx)
            .await
            .expect("isolated operation did not start")
            .expect("isolated operation dropped its start signal");
        let listed_sandbox = sandbox.canonicalize().unwrap();
        let listed = std::process::Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["worktree", "list", "--porcelain"])
            .output()
            .unwrap();
        assert!(
            String::from_utf8_lossy(&listed.stdout).contains(&listed_sandbox.to_string_lossy()[..]),
            "isolated worktree was not registered before cancellation: {}",
            String::from_utf8_lossy(&listed.stdout)
        );
        let cancelled_at = std::time::Instant::now();

        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());

        assert!(
            cancelled_at.elapsed() < Duration::from_secs(1),
            "guard Drop synchronously waited for worktree cleanup"
        );
        let mut unregistered = false;
        for _ in 0..200 {
            let listed = std::process::Command::new("git")
                .arg("-C")
                .arg(&repo)
                .args(["worktree", "list", "--porcelain"])
                .output()
                .unwrap();
            if !String::from_utf8_lossy(&listed.stdout)
                .contains(&listed_sandbox.to_string_lossy()[..])
                && !sandbox.exists()
            {
                unregistered = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            unregistered,
            "cancelled operation left its isolated worktree registered"
        );
    }

    #[tokio::test]
    async fn isolated_cleanup_waits_for_detached_materialization_before_removal() {
        let temp = tempfile::tempdir().unwrap();
        let sandbox = temp.path().join("internal-sandbox");
        std::fs::create_dir(&sandbox).unwrap();
        let (completion, signal) = BlockingWorkCompletion::pair();
        let cleanup = IsolatedCleanup {
            path: sandbox.clone(),
            git_repo: None,
            registered_worktree: false,
            git: GitRunner::default(),
            blocking_work: Some(completion.clone()),
            record_path: None,
        };
        let cleanup = tokio::spawn(cleanup_isolated(cleanup));

        tokio::time::timeout(Duration::from_secs(2), async {
            while completion.waiter_count() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("isolated cleanup never waited for the materializer");
        assert!(
            sandbox.exists(),
            "cleanup removed the directory while materialization was active"
        );
        std::fs::write(
            sandbox.join("late-object"),
            b"materialized after cancellation",
        )
        .unwrap();

        // In production this signal is owned by the spawn_blocking closure and
        // drops only after materialize() returns or unwinds.
        drop(signal);
        tokio::time::timeout(Duration::from_secs(2), cleanup)
            .await
            .expect("isolated cleanup did not finish after materialization")
            .expect("isolated cleanup task panicked")
            .expect("isolated cleanup failed");
        assert!(
            !sandbox.exists(),
            "completed materialization sandbox survived cleanup"
        );
    }

    #[test]
    fn sealed_reference_round_trips_internal_and_git_ids() {
        for (target, current) in [
            ("0123456789abcdef", "fedcba9876543210"),
            (
                "internal:v1:workspace:before",
                "internal:v1:workspace:after",
            ),
        ] {
            let encoded = sealed_reference(target, current);
            assert_eq!(parse_reference(&encoded).unwrap(), (target, Some(current)));
        }
        assert_eq!(
            parse_reference("legacy-checkpoint").unwrap(),
            ("legacy-checkpoint", None)
        );
    }

    #[test]
    fn malformed_sealed_reference_is_rejected() {
        assert!(parse_reference("sealed:v1:nope:payload").is_err());
        assert!(parse_reference("sealed:v1:999:payload").is_err());
        assert!(parse_reference("sealed:v1:0:payload").is_err());
    }

    fn sh(dir: &Path, cmd: &str) {
        let ok = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(dir)
            .status()
            .unwrap()
            .success();
        assert!(ok, "command failed: {cmd}");
    }

    #[tokio::test]
    async fn checkpoint_restores_modified_created_and_deleted_files() {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "hi-ckpt-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        sh(
            &dir,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(dir.join("keep.txt"), "v1\n").unwrap();
        std::fs::write(dir.join("gone.txt"), "stays\n").unwrap();

        // Checkpoint the v1 state.
        let cp = create(&dir).await.expect("checkpoint");

        // A turn modifies one file, deletes another, and creates a third.
        std::fs::write(dir.join("keep.txt"), "v2 changed\n").unwrap();
        std::fs::remove_file(dir.join("gone.txt")).unwrap();
        std::fs::write(dir.join("new.txt"), "created by the turn\n").unwrap();

        let n = restore(&dir, &cp).await.expect("restore");
        assert_eq!(n, 3, "modified + deleted + created");
        assert_eq!(
            std::fs::read_to_string(dir.join("keep.txt")).unwrap(),
            "v1\n"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("gone.txt")).unwrap(),
            "stays\n"
        );
        assert!(!dir.join("new.txt").exists(), "created file removed");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn git_checkpoint_covers_ignored_files_but_excludes_runtime_state() {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "hi-ckpt-ignored-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let state = dir.join(".hi/state");
        std::fs::create_dir_all(&state).unwrap();
        sh(
            &dir,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(dir.join(".gitignore"), "secret.env\n.hi/state/\n").unwrap();
        std::fs::write(dir.join("secret.env"), "before\n").unwrap();
        std::fs::write(state.join("runtime"), "state-before\n").unwrap();

        let checkpoint = match create_detailed_with_state(&dir, &state).await {
            CreateResult::Created(id) => id,
            other => panic!("checkpoint failed: {other:?}"),
        };
        std::fs::write(dir.join("secret.env"), "after\n").unwrap();
        std::fs::write(state.join("runtime"), "state-after\n").unwrap();

        restore_with_state(&dir, &checkpoint, &state).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("secret.env")).unwrap(),
            "before\n"
        );
        assert_eq!(
            std::fs::read_to_string(state.join("runtime")).unwrap(),
            "state-after\n",
            "undo must not overwrite the runtime's own state store"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn git_and_internal_checkpoints_restore_ignored_vendor_sources() {
        static N: AtomicU64 = AtomicU64::new(0);
        for git_backed in [true, false] {
            let base = std::env::temp_dir().join(format!(
                "hi-ckpt-vendor-{git_backed}-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let workspace = base.join("workspace");
            let state = base.join("state");
            std::fs::create_dir_all(workspace.join("vendor")).unwrap();
            std::fs::create_dir_all(&state).unwrap();
            if git_backed {
                sh(
                    &workspace,
                    "git init -q && git config user.email t@t && git config user.name t",
                );
                std::fs::write(workspace.join(".gitignore"), "vendor/\n").unwrap();
            }
            std::fs::write(workspace.join("vendor/base.rs"), "before\n").unwrap();
            let checkpoint = match create_detailed_with_state(&workspace, &state).await {
                CreateResult::Created(id) => id,
                other => panic!("checkpoint failed: {other:?}"),
            };
            assert_eq!(
                checkpoint.starts_with("internal:v1:"),
                !git_backed,
                "test did not exercise the intended backend"
            );
            std::fs::write(workspace.join("vendor/base.rs"), "after\n").unwrap();
            std::fs::write(workspace.join("vendor/new.rs"), "created\n").unwrap();

            restore_with_state(&workspace, &checkpoint, &state)
                .await
                .unwrap();

            assert_eq!(
                std::fs::read_to_string(workspace.join("vendor/base.rs")).unwrap(),
                "before\n"
            );
            assert!(
                !workspace.join("vendor/new.rs").exists(),
                "{git_backed:?} checkpoint did not remove ignored created source"
            );
            let _ = std::fs::remove_dir_all(base);
        }
    }

    #[tokio::test]
    async fn ignored_target_artifacts_are_not_checkpointed_and_undo_leaves_them_alone() {
        // The old policy force-added gitignored target/ into every checkpoint,
        // pouring build artifacts into the user's `.git/objects` on every
        // turn. Regenerable artifact dirs are now excluded: undo neither
        // reverts nor deletes them.
        static N: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir().join(format!(
            "hi-ckpt-generated-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let workspace = base.join("workspace");
        let state = base.join("state");
        std::fs::create_dir_all(workspace.join("target")).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        sh(
            &workspace,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(workspace.join(".gitignore"), "/target/\n").unwrap();
        std::fs::write(workspace.join("src.rs"), "before\n").unwrap();
        std::fs::write(workspace.join("target/existing.rs"), "before\n").unwrap();
        let checkpoint = match create_detailed_with_state(&workspace, &state).await {
            CreateResult::Created(id) => id,
            other => panic!("checkpoint failed: {other:?}"),
        };
        assert!(!checkpoint.starts_with("internal:v1:"));
        std::fs::write(workspace.join("src.rs"), "after\n").unwrap();
        std::fs::write(workspace.join("target/existing.rs"), "after\n").unwrap();
        std::fs::write(workspace.join("target/new.rs"), "created\n").unwrap();

        restore_with_state(&workspace, &checkpoint, &state)
            .await
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(workspace.join("src.rs")).unwrap(),
            "before\n",
            "tracked-side sources are still reverted"
        );
        assert_eq!(
            std::fs::read_to_string(workspace.join("target/existing.rs")).unwrap(),
            "after\n",
            "regenerable artifacts are outside undo coverage"
        );
        assert!(workspace.join("target/new.rs").exists());
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn unignored_artifact_directories_stay_outside_checkpoints() {
        // A bootstrap workspace with no .gitignore at all: the verifier's
        // `cargo test` creates target/, which must neither enter checkpoints
        // (post-apply vs post-verify trees would differ — "verification
        // unstable") nor count toward the ceiling. A root *file* named like
        // an artifact dir (a `build` script) keeps its undo coverage.
        static N: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir().join(format!(
            "hi-ckpt-unignored-target-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let workspace = base.join("workspace");
        let state = base.join("state");
        std::fs::create_dir_all(workspace.join("target/debug")).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        sh(
            &workspace,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(workspace.join("src.rs"), "fn main() {}\n").unwrap();
        std::fs::write(workspace.join("build"), "#!/bin/sh\n").unwrap();
        let huge = std::fs::File::create(workspace.join("target/debug/huge.rlib")).unwrap();
        huge.set_len(MAX_CHECKPOINT_BYTES + 1).unwrap();

        let before = match create_detailed_with_state(&workspace, &state).await {
            CreateResult::Created(id) => id,
            other => panic!("checkpoint must ignore unignored build output: {other:?}"),
        };
        // Simulate the destination verifier mutating build output only: the
        // sealed tree must be unchanged (stability), so a second checkpoint
        // equals the first.
        std::fs::write(workspace.join("target/debug/new.o"), "obj\n").unwrap();
        let after = match create_detailed_with_state(&workspace, &state).await {
            CreateResult::Created(id) => id,
            other => panic!("checkpoint failed after build mutation: {other:?}"),
        };
        let trees = |id: &str| {
            let output = std::process::Command::new("git")
                .args(["rev-parse", &format!("{id}^{{tree}}")])
                .current_dir(&workspace)
                .output()
                .unwrap();
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        };
        assert_eq!(
            trees(&before),
            trees(&after),
            "build-output churn must not destabilize checkpoint trees"
        );

        std::fs::write(workspace.join("build"), "#!/bin/sh\nchanged\n").unwrap();
        std::fs::write(workspace.join("src.rs"), "fn main() { changed(); }\n").unwrap();
        restore_with_state(&workspace, &before, &state)
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(workspace.join("src.rs")).unwrap(),
            "fn main() {}\n",
            "source files keep undo coverage"
        );
        assert_eq!(
            std::fs::read_to_string(workspace.join("build")).unwrap(),
            "#!/bin/sh\nchanged\n",
            "artifact-named paths are outside undo scope, matching the ledger's name prune"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn oversized_ignored_data_does_not_block_checkpoints() {
        // The live failure: a workspace with a multi-GB gitignored dataset
        // could never be checkpointed because ignored bytes counted toward the
        // ceiling — every turn silently ran without undo. Oversized ignored
        // files are now excluded from the snapshot and the ceiling; small
        // ignored inputs alongside them stay covered.
        static N: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir().join(format!(
            "hi-ckpt-generated-limit-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let workspace = base.join("workspace");
        let state = base.join("state");
        std::fs::create_dir_all(workspace.join("target")).unwrap();
        std::fs::create_dir_all(workspace.join("data")).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        sh(
            &workspace,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(workspace.join(".gitignore"), "/target/\n/data/\n").unwrap();
        let huge = std::fs::File::create(workspace.join("target/huge.bin")).unwrap();
        huge.set_len(MAX_CHECKPOINT_BYTES + 1).unwrap();
        let shard = std::fs::File::create(workspace.join("data/shard.npy")).unwrap();
        shard
            .set_len(MAX_IGNORED_CHECKPOINT_FILE_BYTES + 1)
            .unwrap();
        std::fs::write(workspace.join("data/meta.json"), "before\n").unwrap();
        std::fs::write(workspace.join("src.rs"), "before\n").unwrap();

        let checkpoint = match create_detailed_with_state(&workspace, &state).await {
            CreateResult::Created(id) => id,
            other => panic!("checkpoint must succeed despite oversized ignored data: {other:?}"),
        };
        assert!(!checkpoint.starts_with("internal:v1:"));
        std::fs::write(workspace.join("data/meta.json"), "after\n").unwrap();
        std::fs::write(workspace.join("src.rs"), "after\n").unwrap();

        restore_with_state(&workspace, &checkpoint, &state)
            .await
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(workspace.join("src.rs")).unwrap(),
            "before\n"
        );
        assert_eq!(
            std::fs::read_to_string(workspace.join("data/meta.json")).unwrap(),
            "before\n",
            "small ignored inputs stay covered by undo"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn sealed_git_restore_refuses_post_seal_edit() {
        static N: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir().join(format!(
            "hi-ckpt-seal-conflict-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let workspace = base.join("workspace");
        let state = base.join("state");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        sh(
            &workspace,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(workspace.join("file"), "before").unwrap();
        let before = match create_detailed_with_state(&workspace, &state).await {
            CreateResult::Created(id) => id,
            other => panic!("checkpoint failed: {other:?}"),
        };
        std::fs::write(workspace.join("file"), "turn").unwrap();
        let after = match create_detailed_with_state(&workspace, &state).await {
            CreateResult::Created(id) => id,
            other => panic!("checkpoint failed: {other:?}"),
        };
        std::fs::write(workspace.join("file"), "external").unwrap();

        let error = restore_sealed_with_state(&workspace, &before, &after, &state)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("undo conflict"));
        assert_eq!(
            std::fs::read_to_string(workspace.join("file")).unwrap(),
            "external"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn checkpoint_restores_non_ascii_filenames() {
        // Regression: git octal-quotes non-ASCII paths in `--name-status` unless
        // `-z` is used, which made /undo silently skip files like `café.txt`.
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "hi-ckpt-utf8-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        sh(
            &dir,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(dir.join("café.txt"), "v1\n").unwrap();
        std::fs::write(dir.join("naïve.txt"), "stays\n").unwrap();
        let cp = create(&dir).await.expect("checkpoint");

        // Modify one non-ASCII file, delete another, create a third.
        std::fs::write(dir.join("café.txt"), "v2\n").unwrap();
        std::fs::remove_file(dir.join("naïve.txt")).unwrap();
        std::fs::write(dir.join("résumé.txt"), "new\n").unwrap();

        let n = restore(&dir, &cp).await.expect("restore");
        assert_eq!(n, 3, "all three non-ASCII files handled");
        assert_eq!(
            std::fs::read_to_string(dir.join("café.txt")).unwrap(),
            "v1\n",
            "modified non-ASCII file reverted"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("naïve.txt")).unwrap(),
            "stays\n",
            "deleted non-ASCII file restored"
        );
        assert!(
            !dir.join("résumé.txt").exists(),
            "created non-ASCII file removed"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn git_checkpoint_is_scoped_to_explicit_workspace_subdirectory() {
        static N: AtomicU64 = AtomicU64::new(0);
        let repo = std::env::temp_dir().join(format!(
            "hi-ckpt-scope-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let workspace = repo.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        sh(
            &repo,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(workspace.join("inside.txt"), "committed inside\n").unwrap();
        std::fs::write(repo.join("outside.txt"), "committed outside\n").unwrap();
        sh(&repo, "git add -A && git commit -qm baseline");

        std::fs::write(workspace.join("inside.txt"), "checkpoint inside\n").unwrap();
        std::fs::write(repo.join("outside.txt"), "user outside before\n").unwrap();
        let checkpoint = create(&workspace).await.expect("scoped checkpoint");

        std::fs::write(workspace.join("inside.txt"), "turn inside\n").unwrap();
        std::fs::write(repo.join("outside.txt"), "user outside after\n").unwrap();
        assert_eq!(restore(&workspace, &checkpoint).await.unwrap(), 1);
        assert_eq!(
            std::fs::read_to_string(workspace.join("inside.txt")).unwrap(),
            "checkpoint inside\n"
        );
        assert_eq!(
            std::fs::read_to_string(repo.join("outside.txt")).unwrap(),
            "user outside after\n",
            "checkpoint restore must not overwrite changes outside the explicit root"
        );
        let _ = std::fs::remove_dir_all(repo);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn git_checkpoint_restores_mode_and_symlink_target() {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "hi-ckpt-mode-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        sh(
            &dir,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(dir.join("run.sh"), "#!/bin/sh\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir.join("run.sh"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        std::os::unix::fs::symlink("run.sh", dir.join("link")).unwrap();
        let cp = create(&dir).await.unwrap();
        std::fs::set_permissions(dir.join("run.sh"), std::fs::Permissions::from_mode(0o644))
            .unwrap();
        std::fs::remove_file(dir.join("link")).unwrap();
        std::os::unix::fs::symlink("missing", dir.join("link")).unwrap();
        restore(&dir, &cp).await.unwrap();
        assert_eq!(
            std::fs::metadata(dir.join("run.sh"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
        assert_eq!(
            std::fs::read_link(dir.join("link")).unwrap(),
            PathBuf::from("run.sh")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn internal_checkpoint_handles_non_git_workspace() {
        let dir = std::env::temp_dir().join(format!("hi-non-git-{}", std::process::id()));
        let state = std::env::temp_dir().join(format!("hi-non-git-state-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&state);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("file"), "before").unwrap();
        assert!(matches!(
            create_detailed_with_state(&dir, &state).await,
            CreateResult::Created(ref id) if id.starts_with("internal:v1:")
        ));
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(state);
    }

    #[tokio::test]
    async fn isolated_git_checkpoint_is_read_only_and_unregisters_worktree() {
        static N: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir().join(format!(
            "hi-isolated-ckpt-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let dir = base.join("workspace");
        let state = base.join("state");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        sh(
            &dir,
            "git init -q && git config user.email t@t && git config user.name t",
        );
        std::fs::write(dir.join("value.txt"), "before\n").unwrap();
        let checkpoint = match create_detailed_with_state(&dir, &state).await {
            CreateResult::Created(id) => id,
            other => panic!("checkpoint failed: {other:?}"),
        };
        std::fs::write(dir.join("value.txt"), "after\n").unwrap();

        let protected_state = state.clone();
        with_isolated_checkpoint(&dir, &checkpoint, &state, |isolated| async move {
            assert!(
                !isolated.starts_with(&protected_state),
                "buildable verification worktree must not be nested under protected state: {}",
                isolated.display()
            );
            assert!(isolated.join(".git").exists());
            assert_eq!(
                std::fs::read_to_string(isolated.join("value.txt"))?,
                "before\n"
            );
            std::fs::write(isolated.join("value.txt"), "sandbox mutation\n")?;
            Ok(())
        })
        .await
        .unwrap();

        assert_eq!(
            std::fs::read_to_string(dir.join("value.txt")).unwrap(),
            "after\n"
        );
        let worktrees = std::process::Command::new("git")
            .arg("-C")
            .arg(&dir)
            .args(["worktree", "list", "--porcelain"])
            .output()
            .unwrap();
        let listed = String::from_utf8_lossy(&worktrees.stdout);
        assert_eq!(
            listed
                .lines()
                .filter(|line| line.starts_with("worktree "))
                .count(),
            1,
            "temporary worktree remained registered: {listed}"
        );
        assert!(
            !state.join("verification-sandboxes").exists(),
            "sandbox directory should be removed after attribution"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn isolated_cleanup_keeps_its_shared_allocation_parent() {
        static N: AtomicU64 = AtomicU64::new(0);
        let parent = std::env::temp_dir().join(format!(
            "hi-isolated-parent-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let sandbox = parent.join("first");
        std::fs::create_dir_all(&sandbox).unwrap();

        let mut guard = IsolatedGuard::directory(sandbox.clone());
        guard.cleanup().await.unwrap();

        assert!(!sandbox.exists(), "the owned sandbox must be removed");
        assert!(
            parent.exists(),
            "cleanup must not remove the shared parent another verifier may be using"
        );
        let _ = std::fs::remove_dir(parent);
    }
}
