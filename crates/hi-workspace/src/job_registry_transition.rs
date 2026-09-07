//! Canonical job transition constraints for every workspace backend.

use super::*;

pub(super) fn transition_allowed(job: &WorkspaceJobSnapshot, next: JobState) -> bool {
    if !job.state.can_transition_to(next) {
        return false;
    }
    if is_candidate(&job.permit.spec) {
        if job.state == JobState::Running && next == JobState::Succeeded {
            return false;
        }
        if next == JobState::Succeeded && job.state != JobState::Settling {
            return false;
        }
    } else if matches!(job.permit.spec.effect_scope, EffectScope::LiveWriter)
        && next == JobState::Succeeded
        && job.state != JobState::Settling
    {
        return false;
    }
    true
}

pub(super) fn completion_state(completion: JobCompletion) -> JobState {
    match completion {
        JobCompletion::Succeeded => JobState::Succeeded,
        JobCompletion::ReadyToMerge => JobState::ReadyToMerge,
        JobCompletion::Merging => JobState::Merging,
        JobCompletion::Settling => JobState::Settling,
        JobCompletion::Failed => JobState::Failed,
        JobCompletion::Cancelled => JobState::Cancelled,
        JobCompletion::DurabilityPending => JobState::DurabilityPending,
        JobCompletion::RecoveryRequired => JobState::RecoveryRequired,
        JobCompletion::Orphaned => JobState::Orphaned,
        JobCompletion::Stale => JobState::Stale,
    }
}
