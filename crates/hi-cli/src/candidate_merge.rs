//! Transactional destination merge shared by delegate and best-of execution.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::candidate_gate::{
    ensure_command, run_async_thread, run_verifier_sync, staged_candidate_diff,
};

static SCRATCH_ID: AtomicU64 = AtomicU64::new(0);

/// Apply through the transaction engine and verify the exact destination
/// revision. A failure restores the sealed pre-apply checkpoint.
pub(crate) fn apply_candidate_and_reverify(
    worktree: &Path,
    base: &str,
    destination: &Path,
    state_root: &Path,
    verify: &str,
) -> Result<Vec<String>> {
    ensure!(!verify.trim().is_empty(), "destination verifier is empty");
    let pre_apply = create_checkpoint_sync(destination, state_root)
        .context("creating the destination pre-apply checkpoint")?;

    let applied = apply_candidate_transactionally(worktree, base, destination, state_root)
        .context("transactional candidate apply failed")?;
    let post_apply = match create_checkpoint_sync(destination, state_root) {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            let rollback = restore_checkpoint_sync(destination, &pre_apply, state_root);
            return Err(combine_rollback_error(
                error.context("checkpointing the applied destination revision failed"),
                rollback,
            ));
        }
    };

    let verifier_result = run_verifier_sync(destination, verify);
    let post_verify = match create_checkpoint_sync(destination, state_root) {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            let rollback = restore_checkpoint_sync(destination, &pre_apply, state_root);
            return Err(combine_rollback_error(
                error.context("checkpointing the destination after verification failed"),
                rollback,
            ));
        }
    };
    let stable = match same_checkpoint_tree(destination, &post_apply, &post_verify) {
        Ok(stable) => stable,
        Err(error) => {
            let rollback =
                restore_checkpoint_sealed_sync(destination, &pre_apply, &post_verify, state_root);
            return Err(combine_rollback_error(
                error.context("comparing destination verification revisions failed"),
                rollback,
            ));
        }
    };
    if verifier_result.is_ok() && stable {
        return Ok(applied);
    }

    let reason = if let Err(error) = verifier_result {
        error.context(format!("destination verification `{verify}` failed"))
    } else {
        anyhow!("destination verifier modified relevant files (verification unstable)")
    };
    let rollback =
        restore_checkpoint_sealed_sync(destination, &pre_apply, &post_verify, state_root);
    Err(combine_rollback_error(reason, rollback))
}

fn combine_rollback_error(error: anyhow::Error, rollback: Result<usize>) -> anyhow::Error {
    match rollback {
        Ok(_) => error.context("applied candidate changes were rolled back"),
        Err(rollback_error) => error.context(format!(
            "rollback was refused or failed to avoid overwriting concurrent edits: {rollback_error:#}"
        )),
    }
}

/// Preview the patch against a scratch copy of the destination, then commit the
/// exact postimages with the shared digest-sealed transaction engine.
///
/// Intentionally blocking (`std::process` git calls): invoked synchronously via
/// `run_async_thread`, which runs it on a dedicated worker runtime — never on
/// the main async executor.
fn apply_candidate_transactionally(
    worktree: &Path,
    base: &str,
    destination: &Path,
    state_root: &Path,
) -> Result<Vec<String>> {
    let diff = staged_candidate_diff(worktree, base)?;
    ensure!(!diff.paths.is_empty(), "candidate diff is empty");

    let scratch = ScratchDir::new()?;
    for relative in &diff.paths {
        copy_regular_preimage(destination, scratch.path(), relative)?;
    }
    apply_patch_to_scratch(scratch.path(), &diff.patch)?;

    let mut mutations = Vec::with_capacity(diff.paths.len());
    for relative in &diff.paths {
        let before = regular_metadata(&destination.join(relative))?;
        let after_path = scratch.path().join(relative);
        let after = regular_metadata(&after_path)?;
        match (&before, &after) {
            (Some(before_metadata), Some(after)) => {
                mutations.push(hi_tools::PlannedFileMutation::update_with_mode(
                    relative,
                    std::fs::read(&after_path).with_context(|| {
                        format!("reading candidate postimage {}", relative.display())
                    })?,
                    candidate_transaction_mode(worktree, relative, Some(before_metadata), after)?,
                ));
            }
            (None, Some(after)) => {
                mutations.push(hi_tools::PlannedFileMutation::add_with_mode(
                    relative,
                    std::fs::read(&after_path).with_context(|| {
                        format!("reading candidate postimage {}", relative.display())
                    })?,
                    candidate_transaction_mode(worktree, relative, None, after)?,
                ));
            }
            (Some(_), None) => mutations.push(hi_tools::PlannedFileMutation::delete(relative)),
            (None, None) => bail!(
                "candidate change for {} has no destination postimage",
                relative.display()
            ),
        }
    }

    let plan = hi_tools::MutationPlan::new_with_state(destination, state_root, mutations)?;
    ensure!(
        !plan.is_noop(),
        "candidate patch produces no new destination changes"
    );
    let changes = plan.commit()?;
    ensure!(
        !changes.is_empty(),
        "candidate transaction committed no changes"
    );
    Ok(changes.into_iter().map(|change| change.path).collect())
}

fn copy_regular_preimage(destination: &Path, scratch: &Path, relative: &Path) -> Result<()> {
    let source = destination.join(relative);
    let Some(metadata) = regular_metadata(&source)? else {
        return Ok(());
    };
    let target = scratch.join(relative);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating scratch directory {}", parent.display()))?;
    }
    std::fs::copy(&source, &target)
        .with_context(|| format!("copying destination preimage {}", relative.display()))?;
    std::fs::set_permissions(&target, metadata.permissions())
        .with_context(|| format!("preserving mode for {}", relative.display()))?;
    Ok(())
}

fn regular_metadata(path: &Path) -> Result<Option<std::fs::Metadata>> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                !metadata.file_type().is_symlink() && metadata.is_file(),
                "candidate changes unsupported non-regular path {}",
                path.display()
            );
            Ok(Some(metadata))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => {
            Err(error).with_context(|| format!("reading metadata for {}", path.display()))
        }
    }
}

#[cfg(unix)]
fn candidate_transaction_mode(
    worktree: &Path,
    relative: &Path,
    before: Option<&std::fs::Metadata>,
    _after: &std::fs::Metadata,
) -> Result<u32> {
    use std::os::unix::fs::PermissionsExt;
    // Git tracks executable class, not every permission bit. Preserve the
    // destination's other permissions and apply the candidate index's intent.
    let output = Command::new("git")
        .current_dir(worktree)
        .args(["ls-files", "--stage", "-z", "--"])
        .arg(relative)
        .output()
        .with_context(|| format!("reading candidate mode for {}", relative.display()))?;
    let output = ensure_command(output, "git ls-files --stage in candidate worktree")?;
    let mode = output
        .stdout
        .split(|byte| *byte == b' ')
        .next()
        .context("candidate index omitted file mode")?;
    let executable = match mode {
        b"100755" => true,
        b"100644" => false,
        _ => bail!(
            "candidate path {} has unsupported Git mode {}",
            relative.display(),
            String::from_utf8_lossy(mode)
        ),
    };
    let base = before
        .map(|metadata| metadata.permissions().mode() & 0o7777)
        .unwrap_or(0o644);
    Ok(if executable {
        base | ((base & 0o444) >> 2)
    } else {
        base & !0o111
    })
}

#[cfg(not(unix))]
fn candidate_transaction_mode(
    _worktree: &Path,
    _relative: &Path,
    _before: Option<&std::fs::Metadata>,
    after: &std::fs::Metadata,
) -> Result<u32> {
    Ok(if after.permissions().readonly() {
        0o444
    } else {
        0o644
    })
}

fn apply_patch_to_scratch(scratch: &Path, patch: &[u8]) -> Result<()> {
    let mut child = Command::new("git")
        .current_dir(scratch)
        .args(["apply", "--whitespace=nowarn"])
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("launching transactional patch preview")?;
    child
        .stdin
        .take()
        .context("opening transactional patch stdin")?
        .write_all(patch)
        .context("writing transactional patch preview")?;
    let output = child
        .wait_with_output()
        .context("waiting for transactional patch preview")?;
    ensure!(
        output.status.success(),
        "candidate patch conflicts with the destination: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Result<Self> {
        let id = SCRATCH_ID.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("hi-merge-preview-{}-{id}", std::process::id()));
        std::fs::create_dir(&path)
            .with_context(|| format!("creating transaction preview {}", path.display()))?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn create_checkpoint_sync(root: &Path, state_root: &Path) -> Result<String> {
    let root = root.to_path_buf();
    let state_root = state_root.to_path_buf();
    run_async_thread(move || async move {
        match hi_tools::checkpoint::create_detailed_with_state(&root, &state_root).await {
            hi_tools::checkpoint::CreateResult::Created(id) => Ok(id),
            hi_tools::checkpoint::CreateResult::Unavailable(reason)
            | hi_tools::checkpoint::CreateResult::Failed(reason) => bail!(reason),
        }
    })
}

fn restore_checkpoint_sync(root: &Path, target: &str, state_root: &Path) -> Result<usize> {
    let root = root.to_path_buf();
    let target = target.to_string();
    let state_root = state_root.to_path_buf();
    run_async_thread(move || async move {
        hi_tools::checkpoint::restore_with_state(&root, &target, &state_root).await
    })
}

fn restore_checkpoint_sealed_sync(
    root: &Path,
    target: &str,
    expected: &str,
    state_root: &Path,
) -> Result<usize> {
    let root = root.to_path_buf();
    let target = target.to_string();
    let expected = expected.to_string();
    let state_root = state_root.to_path_buf();
    run_async_thread(move || async move {
        hi_tools::checkpoint::restore_sealed_with_state(&root, &target, &expected, &state_root)
            .await
    })
}

fn same_checkpoint_tree(root: &Path, left: &str, right: &str) -> Result<bool> {
    let left_internal = left.starts_with("internal:v1:");
    let right_internal = right.starts_with("internal:v1:");
    if left_internal || right_internal {
        ensure!(
            left_internal && right_internal,
            "candidate merge checkpoint backend changed during verification"
        );
        // Internal ids contain the workspace id and deterministic manifest
        // digest, so equality is the same current-revision comparison as Git
        // tree equality.
        return Ok(left == right);
    }

    fn tree(root: &Path, checkpoint: &str) -> Result<String> {
        let spec = format!("{checkpoint}^{{tree}}");
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["rev-parse", &spec])
            .output()
            .context("resolving checkpoint tree")?;
        let output = ensure_command(output, "git rev-parse checkpoint tree")?;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
    Ok(tree(root, left)? == tree(root, right)?)
}
