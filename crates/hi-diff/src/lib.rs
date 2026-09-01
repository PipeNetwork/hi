//! Differential execution primitives shared by the TUI and headless runners.
//!
//! The crate deliberately knows nothing about ratatui, provider credentials, or
//! a particular model runtime. Backends feed it normalized outcomes and
//! checkpoints; it owns the reproducibility, comparison, corpus, and artifact
//! semantics that make those runs useful.

mod compare;
mod corpus;
mod provider;
mod runner;
mod store;
mod types;
mod workspace;

pub use compare::{compare_local, compare_response, compare_tensor, normalize_text};
pub use corpus::{CaseGenerator, MutationKind, shrink_local_case};
pub use provider::run_provider_targets;
pub use runner::{run_api_case, run_local_case};
pub use store::{ArtifactStore, RunStore, default_root};
pub use types::*;
pub use workspace::{
    capture_workspace_snapshot, create_isolated_worktree, remove_isolated_worktree,
};

/// Version of the serialized run/case format. Increment this when changing
/// semantics rather than silently reading old data as if it were current.
pub const SCHEMA_VERSION: u32 = 1;

/// A backend capable of running a normalized local case.
pub trait LocalImplementation: Send + Sync {
    fn metadata(&self) -> ImplementationMetadata;
    fn capabilities(&self) -> CheckpointCapabilities;
    fn run_case(
        &self,
        case: &LocalCase,
        probe: ProbeLevel,
        sink: &mut dyn CheckpointSink,
    ) -> anyhow::Result<LocalOutcome>;
}

/// A sink for intermediate states. Summary mode must be cheap enough to use in
/// the hot fuzzing loop; full values are written to an artifact store only when
/// the caller explicitly requests them.
pub trait CheckpointSink {
    fn checkpoint(&mut self, checkpoint: Checkpoint<'_>) -> anyhow::Result<()>;
}

/// A host-provided executor for full agent comparisons. The core only needs the
/// normalized task/outcome boundary and never launches an agent itself.
#[async_trait::async_trait]
pub trait AgentExecutor: Send + Sync {
    async fn run(
        &self,
        target: &AgentTarget,
        snapshot: &WorkspaceSnapshot,
        task: &AgentCase,
    ) -> anyhow::Result<AgentOutcome>;
}

/// A provider-independent API target. The actual Provider is injected by the
/// caller so credentials and config remain outside persisted run specs.
#[async_trait::async_trait]
pub trait ApiExecutor: Send + Sync {
    async fn run_response(&self, target: &ApiTarget, case: &ApiCase) -> anyhow::Result<ApiOutcome>;
}

/// A deterministic engine smoke run used by the TUI before concrete model
/// adapters are selected. It exercises case identity, comparison, and
/// snapshot semantics without loading a model or touching the workspace.
pub fn run_smoke(spec: &DiffRunSpec) -> anyhow::Result<DiffRunSnapshot> {
    use std::time::Instant;

    anyhow::ensure!(
        spec.targets.len() >= 2,
        "smoke runs need at least two targets"
    );
    let started = Instant::now();
    let mut generator = CaseGenerator::new(spec.seed).with_limits(128, 16);
    let mut snapshot = DiffRunSnapshot::pending(spec);
    snapshot.status = RunStatus::Running;
    let mut failures = Vec::new();
    for _ in 0..spec.case_count {
        let case = generator.next_case();
        let token = case.input_tokens.first().copied().unwrap_or_default();
        let outcome = LocalOutcome {
            generated_tokens: vec![token],
            next_token: Some(token),
            logits: None,
            checkpoints: Vec::new(),
            checkpoint_values: std::collections::BTreeMap::new(),
        };
        let verdict = compare_local(
            case.id.clone(),
            spec.targets[0].name(),
            &outcome,
            &[(spec.targets[1].name().to_string(), outcome.clone())],
            &spec.contract,
        );
        if verdict.verdict == Verdict::Mismatch {
            snapshot.mismatches += 1;
            failures.push(verdict);
        }
        snapshot.cases_completed += 1;
    }
    snapshot.status = if snapshot.mismatches == 0 {
        RunStatus::Completed
    } else {
        RunStatus::Failed
    };
    snapshot.cases_per_second =
        snapshot.cases_completed as f64 / started.elapsed().as_secs_f64().max(0.000_001);
    snapshot.recent_failures = failures.into_iter().rev().take(8).collect();
    Ok(snapshot)
}
