//! Parent-only publication of detached candidates into a live destination.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, ensure};
use hi_workspace::{CandidateApplyError, HarnessFailpoint, WorkspaceBinding};
use tokio_util::sync::CancellationToken;

use super::{DetachedVerifiedCandidate, apply_verified_candidate};

/// Whether a failed candidate publication is known-clean or needs recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidatePublicationErrorKind {
    /// The exact binding/base or publication contract is no longer eligible.
    Stale,
    /// Publication failed and the destination is proven restored/unchanged.
    Failed,
    /// The destination could not be proven restored. Admission must fail closed.
    RecoveryRequired,
}

/// Typed publication failure used by the job/controller lifecycle.
#[derive(Debug)]
pub struct CandidatePublicationError {
    kind: CandidatePublicationErrorKind,
    detail: String,
}

impl CandidatePublicationError {
    pub fn kind(&self) -> CandidatePublicationErrorKind {
        self.kind
    }

    fn new(kind: CandidatePublicationErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for CandidatePublicationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for CandidatePublicationError {}

/// Apply a candidate transactionally, run its content-bound verification
/// pipeline against the resulting live destination, and accept only a stable
/// green revision. Any known verifier rejection restores the sealed pre-apply
/// checkpoint; a refused/failed rollback is explicitly recovery-required.
pub async fn apply_verified_candidate_and_reverify(
    detached: &DetachedVerifiedCandidate,
    binding: &WorkspaceBinding,
    state_root: &Path,
    runner: &crate::ProcessRunner,
    cancellation: &CancellationToken,
) -> std::result::Result<Vec<crate::FileChange>, CandidatePublicationError> {
    if let Err(error) = detached.candidate.validate_for_apply(binding) {
        let kind = match error {
            CandidateApplyError::Invalid(_) => CandidatePublicationErrorKind::Failed,
            CandidateApplyError::StaleBinding
            | CandidateApplyError::StaleEpoch { .. }
            | CandidateApplyError::StaleBaseVersion
            | CandidateApplyError::MissingDestinationVerification => {
                CandidatePublicationErrorKind::Stale
            }
        };
        return Err(CandidatePublicationError::new(
            kind,
            format!("candidate cannot publish to the current destination: {error}"),
        ));
    }
    if let Err(error) = validate_runtime_binding(binding, state_root, runner) {
        return Err(CandidatePublicationError::new(
            CandidatePublicationErrorKind::Failed,
            format!("candidate destination runtime is invalid: {error:#}"),
        ));
    }
    if cancellation.is_cancelled() {
        return Err(CandidatePublicationError::new(
            CandidatePublicationErrorKind::Failed,
            "candidate publication was cancelled before destination apply",
        ));
    }

    let pre_apply = create_checkpoint(&binding.workspace_root, state_root)
        .await
        .map_err(|error| {
            CandidatePublicationError::new(
                CandidatePublicationErrorKind::Failed,
                format!("candidate destination pre-apply checkpoint failed: {error:#}"),
            )
        })?;
    if let Err(error) = hi_workspace::hit_harness_failpoint(HarnessFailpoint::CandidateBeforeApply)
    {
        return Err(CandidatePublicationError::new(
            CandidatePublicationErrorKind::Failed,
            format!("candidate publication stopped before apply: {error}"),
        ));
    }

    let apply_detached = detached.clone();
    let apply_binding = binding.clone();
    let apply_state = state_root.to_path_buf();
    let applied = tokio::task::spawn_blocking(move || {
        apply_verified_candidate(&apply_detached, &apply_binding, &apply_state)
    })
    .await;
    let changes = match applied {
        Ok(Ok(changes)) => changes,
        Ok(Err(error)) => {
            return Err(classify_apply_error(
                &binding.workspace_root,
                state_root,
                &pre_apply,
                error,
            )
            .await);
        }
        Err(error) => {
            return Err(classify_apply_error(
                &binding.workspace_root,
                state_root,
                &pre_apply,
                anyhow!("candidate apply worker failed: {error}"),
            )
            .await);
        }
    };

    let post_apply = create_checkpoint(&binding.workspace_root, state_root)
        .await
        .map_err(|error| {
            CandidatePublicationError::new(
                CandidatePublicationErrorKind::RecoveryRequired,
                format!(
                    "candidate bytes were applied but their rollback seal could not be created: {error:#}"
                ),
            )
        })?;

    let mut verifier_failure =
        hi_workspace::hit_harness_failpoint(HarnessFailpoint::CandidateAfterApply)
            .err()
            .map(|error| format!("candidate publication interrupted after apply: {error}"));
    if verifier_failure.is_none() && cancellation.is_cancelled() {
        verifier_failure =
            Some("candidate publication was cancelled after destination apply".into());
    }
    if verifier_failure.is_none() {
        let pipeline_budget =
            Duration::from_millis(detached.candidate.destination_verification_budget_ms);
        let pipeline_deadline = Instant::now()
            .checked_add(pipeline_budget)
            .expect("validated destination verification budget fits Instant");
        for verifier in &detached.candidate.destination_verification {
            if cancellation.is_cancelled() {
                verifier_failure = Some(format!(
                    "candidate publication was cancelled before destination verifier `{}`",
                    verifier.name
                ));
                break;
            }
            let remaining = pipeline_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                verifier_failure = Some(format!(
                    "destination verification exceeded its {:.3}s total budget before `{}`",
                    pipeline_budget.as_secs_f64(),
                    verifier.name
                ));
                break;
            }
            let execution = {
                let execution = runner.run_shell(
                    &verifier.command,
                    Duration::from_millis(verifier.timeout_ms).min(remaining),
                );
                tokio::pin!(execution);
                tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => {
                        runner.foreground_registry().kill_current();
                        let _ = execution.await;
                        verifier_failure = Some(format!(
                            "candidate publication was cancelled during destination verifier `{}`",
                            verifier.name
                        ));
                        break;
                    }
                    execution = &mut execution => execution,
                }
            };
            match execution {
                Ok(execution) if execution.status == crate::ToolStatus::Succeeded => {}
                Ok(execution) => {
                    verifier_failure = Some(format!(
                        "destination verifier `{}` failed with {:?}: {}",
                        verifier.name,
                        execution.status,
                        execution.model_content()
                    ));
                    break;
                }
                Err(error) => {
                    verifier_failure = Some(format!(
                        "destination verifier `{}` could not run: {error:#}",
                        verifier.name
                    ));
                    break;
                }
            }
        }
    }

    let post_verify = match create_checkpoint(&binding.workspace_root, state_root).await {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            return Err(CandidatePublicationError::new(
                CandidatePublicationErrorKind::RecoveryRequired,
                format!(
                    "candidate destination could not be sealed after verification; rollback is ambiguous: {error:#}"
                ),
            ));
        }
    };
    if verifier_failure.is_none() && cancellation.is_cancelled() {
        verifier_failure = Some("candidate publication was cancelled after verification".into());
    }
    let stable = same_checkpoint_tree(&binding.workspace_root, &post_apply, &post_verify).await;
    if verifier_failure.is_none() && matches!(stable, Ok(true)) {
        return Ok(changes);
    }

    let reason = match (verifier_failure, stable) {
        (Some(reason), _) => reason,
        (None, Ok(false)) => {
            "destination verifier modified relevant workspace files (verification unstable)".into()
        }
        (None, Err(error)) => {
            format!("destination verification revisions could not be compared: {error:#}")
        }
        (None, Ok(true)) => unreachable!("stable successful verification returned above"),
    };
    match rollback_sealed(
        &binding.workspace_root,
        state_root,
        &pre_apply,
        &post_verify,
    )
    .await
    {
        Ok(_) => Err(CandidatePublicationError::new(
            CandidatePublicationErrorKind::Failed,
            format!("{reason}; applied candidate changes were rolled back"),
        )),
        Err(rollback) => Err(CandidatePublicationError::new(
            CandidatePublicationErrorKind::RecoveryRequired,
            format!(
                "{reason}; rollback was refused or failed to avoid overwriting concurrent edits: {rollback:#}"
            ),
        )),
    }
}

fn validate_runtime_binding(
    binding: &WorkspaceBinding,
    state_root: &Path,
    runner: &crate::ProcessRunner,
) -> Result<()> {
    ensure!(
        state_root == binding.state_root,
        "candidate state root differs from the controller binding"
    );
    let expected = binding
        .workspace_root
        .canonicalize()
        .context("canonicalizing bound candidate destination")?;
    let actual = runner
        .root()
        .canonicalize()
        .context("canonicalizing verifier destination")?;
    ensure!(
        actual == expected,
        "candidate verifier is bound to a different workspace root"
    );
    Ok(())
}

async fn classify_apply_error(
    root: &Path,
    state_root: &Path,
    pre_apply: &str,
    error: anyhow::Error,
) -> CandidatePublicationError {
    let stale = error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<CandidateApplyError>(),
            Some(
                CandidateApplyError::StaleBinding
                    | CandidateApplyError::StaleEpoch { .. }
                    | CandidateApplyError::StaleBaseVersion
                    | CandidateApplyError::MissingDestinationVerification
            )
        )
    }) || error.to_string().contains("candidate base is stale");
    let current = create_checkpoint(root, state_root).await;
    let unchanged = match current {
        Ok(current) => same_checkpoint_tree(root, pre_apply, &current).await,
        Err(error) => Err(error),
    };
    match unchanged {
        Ok(true) => CandidatePublicationError::new(
            if stale {
                CandidatePublicationErrorKind::Stale
            } else {
                CandidatePublicationErrorKind::Failed
            },
            format!("candidate apply failed with destination proven unchanged: {error:#}"),
        ),
        Ok(false) => CandidatePublicationError::new(
            CandidatePublicationErrorKind::RecoveryRequired,
            format!("candidate apply failed after changing the destination: {error:#}"),
        ),
        Err(reconcile) => CandidatePublicationError::new(
            CandidatePublicationErrorKind::RecoveryRequired,
            format!(
                "candidate apply failed and the destination could not be reconciled: {error:#}; {reconcile:#}"
            ),
        ),
    }
}

async fn create_checkpoint(root: &Path, state_root: &Path) -> Result<String> {
    match crate::checkpoint::create_detailed_with_state(root, state_root).await {
        crate::checkpoint::CreateResult::Created(checkpoint) => Ok(checkpoint),
        crate::checkpoint::CreateResult::Unavailable(reason)
        | crate::checkpoint::CreateResult::Failed(reason) => Err(anyhow!(reason)),
    }
}

async fn same_checkpoint_tree(root: &Path, left: &str, right: &str) -> Result<bool> {
    let left_internal = crate::internal_snapshot::is_internal_id(left);
    let right_internal = crate::internal_snapshot::is_internal_id(right);
    ensure!(
        left_internal == right_internal,
        "checkpoint backend changed while publishing candidate"
    );
    if left_internal {
        return Ok(left == right);
    }
    let left = git_checkpoint_tree(root.to_path_buf(), left.to_owned()).await?;
    let right = git_checkpoint_tree(root.to_path_buf(), right.to_owned()).await?;
    Ok(left == right)
}

async fn git_checkpoint_tree(root: PathBuf, checkpoint: String) -> Result<String> {
    tokio::task::spawn_blocking(move || {
        let revision = format!("{checkpoint}^{{tree}}");
        let output = Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(["rev-parse", &revision])
            .output()
            .with_context(|| format!("resolving checkpoint tree in {}", root.display()))?;
        ensure!(
            output.status.success(),
            "git rev-parse checkpoint tree failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        Ok(String::from_utf8(output.stdout)
            .context("Git returned a non-UTF-8 checkpoint tree")?
            .trim()
            .to_owned())
    })
    .await
    .context("checkpoint tree worker failed")?
}

async fn rollback_sealed(
    root: &Path,
    state_root: &Path,
    target: &str,
    expected: &str,
) -> Result<usize> {
    hi_workspace::hit_harness_failpoint(HarnessFailpoint::RollbackBeforeRestore)?;
    crate::checkpoint::restore_sealed_with_state(root, target, expected, state_root).await
}
