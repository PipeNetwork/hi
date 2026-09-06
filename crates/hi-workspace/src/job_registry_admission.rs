use crate::{EffectScope, JobKind, JobSpec, WorkspaceAuthority, WorkspaceBinding};

pub(super) fn is_candidate(spec: &JobSpec) -> bool {
    spec.kind == JobKind::WriteCandidate || spec.effect_scope == EffectScope::CandidateOnly
}

pub(super) fn is_write_job(spec: &JobSpec) -> bool {
    is_candidate(spec) || spec.effect_scope == EffectScope::LiveWriter
}

pub(super) fn is_preparation(spec: &JobSpec) -> bool {
    is_candidate(spec)
}

pub(super) fn is_local_live_process(binding: &WorkspaceBinding, spec: &JobSpec) -> bool {
    matches!(&binding.authority, WorkspaceAuthority::Local)
        && spec.kind == JobKind::Process
        && spec.effect_scope == EffectScope::LiveWriter
}

pub(super) fn live_writers_conflict(
    binding: &WorkspaceBinding,
    incoming: &JobSpec,
    existing: &JobSpec,
) -> bool {
    existing.effect_scope == EffectScope::LiveWriter
        && !(is_local_live_process(binding, incoming) && is_local_live_process(binding, existing))
}
