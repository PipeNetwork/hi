//! Git-backed working-tree checkpoints for `/undo`.
//!
//! Before each turn the agent snapshots the full non-ignored working tree into a
//! *dangling* commit — built in a throwaway index so it never touches the user's
//! staging area, branch, or history. `/undo` restores the latest snapshot,
//! reverting every file the turn created, modified, or deleted in one step. This
//! is what makes running with no confirmation prompts safe: anything is undoable.
//!
//! Git and internal checkpoints cover the complete bounded tree below the
//! explicit workspace, including ignored/generated/vendor paths and excluding
//! only the runtime's own state root. Both preserve executable modes and
//! symlink targets. If that complete tree exceeds the checkpoint limits,
//! mutation is denied unless the caller explicitly allows no checkpoint.
//! Neither can undo non-file side effects such as network changes or deletes
//! outside the workspace; those are what the catastrophic-operation guard is
//! for.

use std::ffi::{OsStr, OsString};
use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::process::Output;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail, ensure};
use tokio::process::Command;

const SEALED_REFERENCE_PREFIX: &str = "sealed:v1:";
static ISOLATED_ID: AtomicU64 = AtomicU64::new(0);

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

struct IsolatedGuard {
    path: PathBuf,
    git_repo: Option<PathBuf>,
    registered_worktree: bool,
    cleaned: bool,
}

impl IsolatedGuard {
    fn directory(path: PathBuf) -> Self {
        Self {
            path,
            git_repo: None,
            registered_worktree: false,
            cleaned: false,
        }
    }

    fn worktree(path: PathBuf, git_repo: PathBuf) -> Self {
        Self {
            path,
            git_repo: Some(git_repo),
            registered_worktree: true,
            cleaned: false,
        }
    }

    async fn cleanup(&mut self) -> Result<()> {
        if self.cleaned {
            return Ok(());
        }
        if self.registered_worktree {
            let repo = self
                .git_repo
                .as_ref()
                .context("isolated worktree has no source repository")?;
            let output = Command::new("git")
                .arg("-C")
                .arg(repo)
                .args(["worktree", "remove", "--force"])
                .arg(&self.path)
                .output()
                .await
                .context("removing isolated verification worktree")?;
            if !output.status.success() {
                // Removing the directory and pruning the now-missing worktree
                // is an idempotent fallback, including after cancellation or a
                // verification command that damaged its own `.git` file.
                let _ = std::fs::remove_dir_all(&self.path);
                let prune = Command::new("git")
                    .arg("-C")
                    .arg(repo)
                    .args(["worktree", "prune", "--expire", "now"])
                    .output()
                    .await
                    .context("pruning isolated verification worktree")?;
                if !prune.status.success() {
                    bail!(
                        "could not remove isolated verification worktree: {}; prune also failed: {}",
                        String::from_utf8_lossy(&output.stderr).trim(),
                        String::from_utf8_lossy(&prune.stderr).trim()
                    );
                }
            }
        } else if self.path.exists() {
            std::fs::remove_dir_all(&self.path).with_context(|| {
                format!(
                    "removing isolated verification copy {}",
                    self.path.display()
                )
            })?;
        }
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::remove_dir(parent);
        }
        self.cleaned = true;
        Ok(())
    }

    fn cleanup_sync(&mut self) {
        if self.cleaned {
            return;
        }
        if self.registered_worktree
            && let Some(repo) = &self.git_repo
        {
            let removed = std::process::Command::new("git")
                .arg("-C")
                .arg(repo)
                .args(["worktree", "remove", "--force"])
                .arg(&self.path)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|status| status.success());
            if !removed {
                let _ = std::fs::remove_dir_all(&self.path);
                let _ = std::process::Command::new("git")
                    .arg("-C")
                    .arg(repo)
                    .args(["worktree", "prune", "--expire", "now"])
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
            }
        } else {
            let _ = std::fs::remove_dir_all(&self.path);
        }
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::remove_dir(parent);
        }
        self.cleaned = true;
    }
}

impl Drop for IsolatedGuard {
    fn drop(&mut self) {
        self.cleanup_sync();
    }
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
    let (target, _) = parse_reference(reference)?;
    let parent = state_root.join("verification-sandboxes");
    std::fs::create_dir_all(&parent)
        .with_context(|| format!("creating verification sandbox root {}", parent.display()))?;
    let sandbox = parent.join(format!(
        "verify-{}-{}",
        std::process::id(),
        ISOLATED_ID.fetch_add(1, Ordering::Relaxed)
    ));
    if sandbox.exists() {
        std::fs::remove_dir_all(&sandbox)
            .with_context(|| format!("removing stale sandbox {}", sandbox.display()))?;
    }

    let (mut guard, operation_root) = if crate::internal_snapshot::is_internal_id(target) {
        let guard = IsolatedGuard::directory(sandbox.clone());
        let source = dir.to_path_buf();
        let state = state_root.to_path_buf();
        let target = target.to_string();
        let destination = sandbox.clone();
        tokio::task::spawn_blocking(move || {
            crate::internal_snapshot::materialize(&source, &state, &target, &destination)
        })
        .await
        .context("isolated snapshot materialization task failed")??;
        (guard, sandbox)
    } else {
        let repo = toplevel(dir).await.context("not in a git work tree")?;
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
        let output = Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["worktree", "add", "--detach", "--force"])
            .arg(&sandbox)
            .arg(target)
            .output()
            .await
            .context("creating isolated verification worktree")?;
        if !output.status.success() {
            let _ = std::fs::remove_dir_all(&sandbox);
            let _ = Command::new("git")
                .arg("-C")
                .arg(&repo)
                .args(["worktree", "prune", "--expire", "now"])
                .output()
                .await;
            bail!(
                "creating isolated verification worktree failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let operation_root = sandbox.join(relative_root);
        (IsolatedGuard::worktree(sandbox, repo), operation_root)
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
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .await
        .context("running git")
}

async fn git_indexed(dir: &Path, index: &str, args: &[String]) -> Result<Output> {
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_INDEX_FILE", index)
        .output()
        .await
        .context("running git")
}

async fn git_owned(dir: &Path, args: Vec<OsString>) -> Result<Output> {
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
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

async fn toplevel(dir: &Path) -> Option<std::path::PathBuf> {
    let out = git(dir, &["rev-parse", "--show-toplevel"]).await.ok()?;
    if !out.status.success() {
        return None;
    }
    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!p.is_empty()).then(|| std::path::PathBuf::from(p))
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
    match tokio::task::spawn_blocking(move || checkpoint_preflight(&scan_root, &scan_state)).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            return CreateResult::Failed(format!("workspace cannot be checkpointed: {error:#}"));
        }
        Err(error) => return CreateResult::Failed(format!("checkpoint preflight failed: {error}")),
    }
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!("hi-checkpoint-{}-{n}", std::process::id()));
    let Some(index) = tmp.to_str() else {
        return CreateResult::Failed("temporary checkpoint index path is not valid UTF-8".into());
    };

    // Seed the throwaway index from HEAD so `add -A` is a fast incremental
    // (harmlessly fails in a repo with no commits yet).
    let _ = git_indexed(&scope.root, index, &["read-tree".into(), "HEAD".into()]).await;
    // Limit the throwaway index update to the explicit workspace root. The
    // index is seeded from HEAD, so paths elsewhere in a containing monorepo
    // remain at HEAD even when they have unrelated dirty user changes. `-f`
    // includes ignored workspace inputs so Git and internal checkpoints provide
    // equivalent undo coverage. A bounded no-follow preflight above prevents
    // force-adding an unbounded generated tree; if the complete workspace is
    // too large, checkpoint creation fails and mutation is denied. The runtime
    // state root is explicitly excluded to avoid recursively checkpointing
    // hi's own snapshots and journals.
    let mut add_args = vec![
        "add".to_string(),
        "-f".to_string(),
        "-A".to_string(),
        "--".to_string(),
        ".".to_string(),
    ];
    if let Some(relative_state) = contained_relative_path(&scope.root, state_root) {
        let relative_state = relative_state.to_string_lossy().replace('\\', "/");
        add_args.push(format!(":(exclude){relative_state}"));
        add_args.push(format!(":(exclude){relative_state}/**"));
    }
    let add = match git_indexed(&scope.root, index, &add_args).await {
        Ok(output) => output,
        Err(err) => {
            let _ = std::fs::remove_file(&tmp);
            return CreateResult::Failed(format!("git add failed: {err:#}"));
        }
    };
    if !add.status.success() {
        let _ = std::fs::remove_file(&tmp);
        return CreateResult::Failed(format!(
            "git add failed: {}",
            String::from_utf8_lossy(&add.stderr).trim()
        ));
    }
    let tree_out = match git_indexed(&scope.root, index, &["write-tree".into()]).await {
        Ok(output) => output,
        Err(err) => {
            let _ = std::fs::remove_file(&tmp);
            return CreateResult::Failed(format!("git write-tree failed: {err:#}"));
        }
    };
    let _ = std::fs::remove_file(&tmp);
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

fn checkpoint_preflight(root: &Path, state_root: &Path) -> Result<()> {
    let root = root
        .canonicalize()
        .with_context(|| format!("canonicalizing checkpoint root {}", root.display()))?;
    ensure!(root.is_dir(), "checkpoint root is not a directory");
    let state = state_root
        .canonicalize()
        .unwrap_or_else(|_| state_root.to_path_buf());
    let mut bytes = 0u64;
    let mut entries = 0usize;
    checkpoint_preflight_dir(&root, &root, &state, &mut bytes, &mut entries)
}

fn checkpoint_preflight_dir(
    root: &Path,
    directory: &Path,
    state_root: &Path,
    bytes: &mut u64,
    entries: &mut usize,
) -> Result<()> {
    for entry in std::fs::read_dir(directory)
        .with_context(|| format!("reading checkpoint directory {}", directory.display()))?
    {
        let entry = entry.with_context(|| format!("walking {}", directory.display()))?;
        let path = entry.path();
        if matches!(
            entry.file_name().to_str(),
            Some(".git" | ".hg" | ".svn" | ".jj")
        ) {
            if directory == root {
                continue;
            }
            // A parent Git tree stores a nested repository as a gitlink and
            // cannot represent its dirty working files. Force the unified
            // creator to fall back to the no-follow internal backend instead
            // of claiming incomplete undo coverage.
            bail!(
                "nested repository metadata at {} is not representable by a Git checkpoint",
                path.display()
            );
        }
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("reading checkpoint metadata {}", path.display()))?;
        if !metadata.file_type().is_symlink() {
            let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
            if canonical == state_root || canonical.starts_with(state_root) {
                continue;
            }
        }
        *entries = entries.saturating_add(1);
        ensure!(
            *entries <= MAX_CHECKPOINT_ENTRIES,
            "workspace checkpoint exceeds {MAX_CHECKPOINT_ENTRIES} entries"
        );
        let file_type = metadata.file_type();
        if file_type.is_dir() {
            checkpoint_preflight_dir(root, &path, state_root, bytes, entries)?;
        } else if file_type.is_file() {
            *bytes = bytes.saturating_add(metadata.len());
        } else if file_type.is_symlink() {
            let target = std::fs::read_link(&path)
                .with_context(|| format!("reading checkpoint symlink {}", path.display()))?;
            *bytes = bytes.saturating_add(os_str_bytes(target.as_os_str()).len() as u64);
        } else {
            bail!(
                "cannot checkpoint special filesystem entry {}",
                path.display()
            );
        }
        ensure!(
            *bytes <= MAX_CHECKPOINT_BYTES,
            "workspace checkpoint exceeds {} MiB ceiling",
            MAX_CHECKPOINT_BYTES / 1024 / 1024
        );
    }
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
        let root = dir.to_path_buf();
        let state = state_root.to_path_buf();
        let target = target.to_string();
        return tokio::task::spawn_blocking(move || {
            crate::internal_snapshot::restore(&root, &state, &target)
        })
        .await
        .context("internal restore task failed")?;
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
        plan.commit()?;
    }
    Ok(changed)
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
        let root = dir.to_path_buf();
        let state = state_root.to_path_buf();
        let target = target.to_string();
        let expected = expected_current.to_string();
        return tokio::task::spawn_blocking(move || {
            crate::internal_snapshot::restore_sealed(&root, &state, &target, &expected)
        })
        .await
        .context("sealed internal restore task failed")?;
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
                plan.commit()?;
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
    async fn bounded_git_checkpoint_covers_ignored_target_files() {
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
        std::fs::write(workspace.join("target/existing.rs"), "before\n").unwrap();
        let checkpoint = match create_detailed_with_state(&workspace, &state).await {
            CreateResult::Created(id) => id,
            other => panic!("checkpoint failed: {other:?}"),
        };
        assert!(!checkpoint.starts_with("internal:v1:"));
        std::fs::write(workspace.join("target/existing.rs"), "after\n").unwrap();
        std::fs::write(workspace.join("target/new.rs"), "created\n").unwrap();

        restore_with_state(&workspace, &checkpoint, &state)
            .await
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(workspace.join("target/existing.rs")).unwrap(),
            "before\n"
        );
        assert!(!workspace.join("target/new.rs").exists());
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn oversized_ignored_target_fails_closed_without_partial_checkpoint() {
        static N: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir().join(format!(
            "hi-ckpt-generated-limit-{}-{}",
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
        let huge = std::fs::File::create(workspace.join("target/huge.bin")).unwrap();
        huge.set_len(MAX_CHECKPOINT_BYTES + 1).unwrap();

        let result = create_detailed_with_state(&workspace, &state).await;

        assert!(matches!(
            result,
            CreateResult::Failed(ref reason) if reason.contains("512 MiB")
        ));
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

        with_isolated_checkpoint(&dir, &checkpoint, &state, |isolated| async move {
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
}
