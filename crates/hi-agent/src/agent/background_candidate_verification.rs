//! Content-bound live verifier contract for background candidates.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use hi_workspace::{CandidateApplyError, WorkspaceBinding};
use tokio_util::sync::CancellationToken;

pub(super) type PublicationResult = std::result::Result<
    Vec<hi_tools::FileChange>,
    hi_tools::candidate_workspace::CandidatePublicationError,
>;

pub(super) struct CandidateSettlementGuard {
    abandonment: Arc<CandidateAbandonment>,
    armed: bool,
}

pub(super) struct CandidateAbandonment {
    claimed: AtomicBool,
    registry: Arc<hi_tools::BackgroundTaskRegistry>,
    task_id: String,
    evidence: hi_tools::candidate_workspace::PersistedDetachedCandidate,
}

impl CandidateSettlementGuard {
    pub(super) fn new(
        registry: Arc<hi_tools::BackgroundTaskRegistry>,
        task_id: String,
        evidence: hi_tools::candidate_workspace::PersistedDetachedCandidate,
    ) -> (Self, Arc<CandidateAbandonment>) {
        let abandonment = Arc::new(CandidateAbandonment {
            claimed: AtomicBool::new(false),
            registry,
            task_id,
            evidence,
        });
        (
            Self {
                abandonment: Arc::clone(&abandonment),
                armed: true,
            },
            abandonment,
        )
    }

    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CandidateSettlementGuard {
    fn drop(&mut self) {
        if self.armed {
            self.abandonment.recover(
                "candidate publication caller disappeared before lifecycle settlement completed",
            );
        }
    }
}

impl CandidateAbandonment {
    pub(super) fn recover(&self, detail: impl Into<String>) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            // The immutable artifact remains on disk for startup recovery.
            return;
        };
        if self.claimed.swap(true, Ordering::AcqRel) {
            return;
        }
        let registry = Arc::clone(&self.registry);
        let task_id = self.task_id.clone();
        let evidence = self.evidence.clone();
        let detail = detail.into();
        let publication = registry.track_candidate_publication();
        runtime.spawn(async move {
            let transition = registry
                .transition_candidate(
                    &task_id,
                    hi_tools::BackgroundCandidateTransition::RecoveryRequired,
                    Some(detail.clone()),
                )
                .await;
            let detail = match transition {
                Ok(()) => detail,
                Err(error) => format!(
                    "{detail}; recovery-required job transition was not acknowledged: {error}"
                ),
            };
            registry.restore_ready_candidate(&task_id, evidence);
            registry.resolve_candidate_retained(&task_id, detail);
            drop(publication);
        });
    }
}

pub(super) async fn supervised_publication(
    registry: Arc<hi_tools::BackgroundTaskRegistry>,
    detached: hi_tools::candidate_workspace::PersistedDetachedCandidate,
    binding: WorkspaceBinding,
    runner: hi_tools::ProcessRunner,
    turn_cancellation: Option<crate::TurnCancellation>,
    abandonment: Arc<CandidateAbandonment>,
) -> Result<PublicationResult, String> {
    let cancellation = CancellationToken::new();
    let cancellation_for_worker = cancellation.clone();
    let publication = registry.track_candidate_publication();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let watcher = turn_cancellation.map(|signal| {
            tokio::spawn(async move {
                while !signal.is_cancelled() {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
                cancellation.cancel();
            })
        });
        let result = hi_tools::candidate_workspace::apply_verified_candidate_and_reverify(
            &detached,
            &binding,
            &binding.state_root,
            &runner,
            &cancellation_for_worker,
        )
        .await;
        if let Some(watcher) = watcher {
            watcher.abort();
        }
        if let Err(result) = result_tx.send(result) {
            let detail = match result {
                Ok(_) => "candidate publication caller disappeared after destination verification"
                    .to_owned(),
                Err(error) => format!(
                    "candidate publication caller disappeared while handling publication failure: {error:#}"
                ),
            };
            abandonment.recover(detail);
        }
        drop(publication);
    });
    result_rx
        .await
        .map_err(|_| "candidate publication supervisor stopped without a result".into())
}

pub(super) struct CandidateRejection {
    pub(super) transition: hi_tools::BackgroundCandidateTransition,
    pub(super) detail: String,
}

pub(super) fn candidate_preflight(
    detached: &hi_tools::candidate_workspace::DetachedVerifiedCandidate,
    binding: &WorkspaceBinding,
    job_verification_ms: Option<u64>,
) -> Result<(), CandidateRejection> {
    if let Err(error) = detached.candidate.validate_for_apply(binding) {
        return Err(CandidateRejection {
            transition: if matches!(error, CandidateApplyError::Invalid(_)) {
                hi_tools::BackgroundCandidateTransition::Failed
            } else {
                hi_tools::BackgroundCandidateTransition::Stale
            },
            detail: format!("candidate cannot apply to the current binding: {error}"),
        });
    }
    let Some(job_verification_ms) = job_verification_ms else {
        return Err(CandidateRejection {
            transition: hi_tools::BackgroundCandidateTransition::RecoveryRequired,
            detail: "candidate job has no authoritative verification budget".into(),
        });
    };
    if detached.candidate.destination_verification_budget_ms != job_verification_ms {
        return Err(CandidateRejection {
            transition: hi_tools::BackgroundCandidateTransition::Stale,
            detail: format!(
                "candidate verifier budget {}ms differs from its workspace job budget {job_verification_ms}ms",
                detached.candidate.destination_verification_budget_ms
            ),
        });
    }
    hi_tools::candidate_workspace::ensure_workspace_matches(
        &binding.workspace_root,
        &binding.state_root,
        &detached.source_snapshot_id,
    )
    .map_err(|error| CandidateRejection {
        transition: hi_tools::BackgroundCandidateTransition::Stale,
        detail: format!("candidate base is stale: {error:#}"),
    })
}

pub(super) fn candidate_execution_report(
    evidence: &hi_tools::candidate_workspace::PersistedDetachedCandidate,
    disposition: hi_workspace::ExecutionDisposition,
    workspace_may_have_changed: bool,
    changes: &[hi_tools::FileChange],
    detail: Option<String>,
) -> hi_workspace::ExecutionReport {
    let mut changed_paths = changes
        .iter()
        .map(|change| std::path::PathBuf::from(&change.path))
        .collect::<Vec<_>>();
    changed_paths.sort();
    changed_paths.dedup();
    let mut artifacts = vec![evidence.artifact.clone()];
    artifacts.extend(evidence.candidate.artifacts.iter().cloned());
    artifacts.extend(
        evidence
            .candidate
            .verification
            .iter()
            .flat_map(|verification| verification.artifacts.iter().cloned()),
    );
    artifacts.sort_by(|left, right| left.uri.cmp(&right.uri));
    artifacts.dedup_by(|left, right| left.uri == right.uri);
    hi_workspace::ExecutionReport {
        disposition,
        workspace_may_have_changed,
        external_effect_may_have_occurred: false,
        content_digest: None,
        changed_paths,
        artifacts,
        detail,
    }
}

pub(super) fn candidate_call_arguments(
    task_id: &str,
    evidence: &hi_tools::candidate_workspace::PersistedDetachedCandidate,
) -> String {
    serde_json::json!({
        "task_id": task_id,
        "job_id": &evidence.candidate.job_id,
        "candidate_id": &evidence.candidate.candidate_id,
        "candidate_digest": &evidence.candidate.candidate_digest,
        "source_binding_id": &evidence.candidate.source_binding_id,
        "source_epoch": evidence.candidate.source_epoch,
        "base_version": &evidence.candidate.base_version,
        "effective_route": &evidence.candidate.effective_route,
        "artifact": &evidence.artifact,
    })
    .to_string()
}

/// Recover the exact ordered executable stages from the final successful
/// child verification round. Earlier repair attempts, LSP-only diagnostics,
/// and a timed-out attempt followed by a successful retry are not replayed.
pub(super) fn from_successful_round(
    executions: &[crate::VerificationExecution],
    timeout_ms: u64,
) -> Vec<hi_workspace::CandidateDestinationVerifier> {
    let Some(final_round) = executions.iter().map(|execution| execution.round).max() else {
        return Vec::new();
    };
    let mut seen = std::collections::BTreeSet::new();
    executions
        .iter()
        .filter(|execution| {
            execution.round == final_round
                && execution.process.is_some()
                && execution.status == hi_tools::ToolStatus::Succeeded
        })
        .filter_map(|execution| {
            let identity = (execution.name.clone(), execution.command.clone());
            seen.insert(identity.clone())
                .then_some(hi_workspace::CandidateDestinationVerifier {
                    name: identity.0,
                    command: identity.1,
                    timeout_ms,
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn execution(
        round: u32,
        name: &str,
        command: &str,
        status: hi_tools::ToolStatus,
    ) -> crate::VerificationExecution {
        crate::VerificationExecution {
            round,
            name: name.into(),
            command: command.into(),
            status,
            process: Some(hi_tools::ProcessOutcome {
                exit_code: Some(if status == hi_tools::ToolStatus::Succeeded {
                    0
                } else {
                    1
                }),
                stdout_summary: String::new(),
                stderr_summary: String::new(),
                duration_ms: 1,
            }),
            truncation: Some(hi_tools::TruncationState::Complete),
        }
    }

    #[test]
    fn contract_uses_only_ordered_successes_from_final_round() {
        let mut executions = vec![
            execution(1, "check", "old-check", hi_tools::ToolStatus::Failed),
            execution(2, "check", "new-check", hi_tools::ToolStatus::Succeeded),
            execution(2, "test", "new-test", hi_tools::ToolStatus::Succeeded),
            execution(2, "test", "new-test", hi_tools::ToolStatus::Succeeded),
        ];
        let mut lsp = execution(2, "lsp", "diagnostics", hi_tools::ToolStatus::Succeeded);
        lsp.process = None;
        executions.push(lsp);

        let contract = from_successful_round(&executions, 12_345);
        assert_eq!(contract.len(), 2);
        assert_eq!(contract[0].command, "new-check");
        assert_eq!(contract[1].command, "new-test");
        assert!(contract.iter().all(|stage| stage.timeout_ms == 12_345));
    }
}
