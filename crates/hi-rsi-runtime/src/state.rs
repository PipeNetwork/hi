use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{BudgetUsage, StageId};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub hash: String,
    pub size_bytes: u64,
    pub media_type: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryState {
    pub repository_snapshot_hash: String,
    pub starting_commit: String,
    pub source_tree_hash: String,
    pub worktree_root: String,
    pub submodule_commits: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryClass {
    Working,
    Attempt,
    Repository,
    Episodic,
    Procedural,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryEntry {
    pub memory_class: MemoryClass,
    pub content: Value,
    pub tenant_id: String,
    pub repository_scope: Option<String>,
    pub candidate_id: String,
    pub supporting_artifacts: Vec<ArtifactRef>,
    pub confidence_millionths: u32,
    pub created_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub supervisor_verified: bool,
}

impl MemoryEntry {
    pub fn validate_candidate_authored(&self) -> Result<()> {
        ensure!(
            !self.supervisor_verified,
            "candidate-authored memory cannot claim supervisor verification"
        );
        ensure!(
            self.confidence_millionths <= 1_000_000,
            "memory confidence exceeds one"
        );
        ensure!(
            self.expires_at_unix_ms > self.created_at_unix_ms,
            "memory expiry must follow creation"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    TaskClassifier,
    RequirementNormalizer,
    RepositoryExplorer,
    Diagnostician,
    Planner,
    Implementer,
    TestGenerator,
    Reviewer,
    Repairer,
    Summarizer,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextTrust {
    TrustedInstruction,
    UserInstruction,
    UntrustedRepository,
    UntrustedToolOutput,
    UntrustedExternal,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextItem {
    pub source: String,
    pub content_hash: String,
    pub byte_start: Option<u64>,
    pub byte_end: Option<u64>,
    pub selection_reason: String,
    pub relevance_millionths: Option<u32>,
    pub trust: ContextTrust,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextManifest {
    pub items: Vec<ContextItem>,
    pub token_estimate: u64,
    pub selection_policy_version: String,
    pub omitted_candidates: Vec<String>,
}

impl ContextManifest {
    pub fn canonical_hash(&self) -> Result<String> {
        Ok(blake3::hash(&serde_json::to_vec(self)?)
            .to_hex()
            .to_string())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EngineeringPlan {
    pub objective: String,
    pub assumptions: Vec<String>,
    pub affected_components: Vec<String>,
    pub evidence: Vec<ArtifactRef>,
    pub proposed_changes: Vec<String>,
    pub tests: Vec<String>,
    pub risks: Vec<String>,
    pub rollback: String,
    pub revision: u32,
    pub revision_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    Passed,
    Failed,
    InfrastructureError,
    SkippedByPolicy,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationCheck {
    pub name: String,
    pub command_hash: String,
    pub status: VerificationStatus,
    pub exit_code: Option<i32>,
    pub duration_ms: u64,
    pub output: Option<ArtifactRef>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationReport {
    pub report_version: u16,
    pub run_id: String,
    pub candidate_id: String,
    pub environment_hash: String,
    pub source_tree_hash: String,
    pub checks: Vec<VerificationCheck>,
    pub passed: bool,
    pub policy_violations: Vec<String>,
    pub artifacts: Vec<ArtifactRef>,
    /// Present only on a report returned by the trusted supervisor. Candidate
    /// protocol messages must leave it unset.
    pub supervisor_attestation: Option<String>,
}

impl VerificationReport {
    pub fn validate_candidate_proposal(&self) -> Result<()> {
        ensure!(
            self.supervisor_attestation.is_none(),
            "candidate cannot supply a supervisor attestation"
        );
        Ok(())
    }

    pub fn validate_supervisor_report(&self) -> Result<()> {
        ensure!(
            self.supervisor_attestation
                .as_ref()
                .is_some_and(|value| !value.is_empty()),
            "trusted verification report has no attestation"
        );
        let checks_passed = self
            .checks
            .iter()
            .all(|check| check.status == VerificationStatus::Passed);
        ensure!(
            self.passed == (checks_passed && self.policy_violations.is_empty()),
            "verification summary does not match its checks"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureDomain {
    Candidate,
    ModelProvider,
    Tool,
    Repository,
    BuildEnvironment,
    Evaluator,
    Policy,
    Budget,
    WorkerInfrastructure,
    Backend,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FailureEvidence {
    pub domain: FailureDomain,
    pub subcategory: String,
    pub retryable: bool,
    pub causal_event_hash: Option<String>,
    pub stage: StageId,
    pub artifacts: Vec<ArtifactRef>,
    pub counts_against_candidate: bool,
}

impl FailureEvidence {
    pub fn validate(&self) -> Result<()> {
        if matches!(
            self.domain,
            FailureDomain::ModelProvider
                | FailureDomain::BuildEnvironment
                | FailureDomain::Evaluator
                | FailureDomain::WorkerInfrastructure
                | FailureDomain::Backend
        ) {
            ensure!(
                !self.counts_against_candidate,
                "infrastructure failure cannot count against candidate quality"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunState {
    pub task_id: String,
    pub run_id: String,
    pub candidate_id: String,
    pub repository: RepositoryState,
    pub current_stages: BTreeSet<StageId>,
    pub attempts: BTreeMap<StageId, u32>,
    pub working_memory: Vec<MemoryEntry>,
    pub plan: Option<EngineeringPlan>,
    pub patches: Vec<ArtifactRef>,
    pub verification: Vec<VerificationReport>,
    pub budget: BudgetUsage,
    pub failure_evidence: Vec<FailureEvidence>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub schema_version: u16,
    pub run_id: String,
    pub candidate_id: String,
    pub state: RunState,
    pub workspace_tree_hash: String,
    pub workflow_position: BTreeSet<StageId>,
    pub context_manifests: Vec<ArtifactRef>,
    pub response_artifacts: Vec<ArtifactRef>,
    pub created_at_sequence: u64,
}

impl Checkpoint {
    pub fn canonical_hash(&self) -> Result<String> {
        ensure!(
            self.run_id == self.state.run_id && self.candidate_id == self.state.candidate_id,
            "checkpoint identity does not match run state"
        );
        Ok(blake3::hash(&serde_json::to_vec(self)?)
            .to_hex()
            .to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_cannot_forge_verified_memory_or_attestation() {
        let memory = MemoryEntry {
            memory_class: MemoryClass::Working,
            content: Value::String("hypothesis".into()),
            tenant_id: "tenant".into(),
            repository_scope: None,
            candidate_id: "candidate".into(),
            supporting_artifacts: vec![],
            confidence_millionths: 500_000,
            created_at_unix_ms: 1,
            expires_at_unix_ms: 2,
            supervisor_verified: true,
        };
        assert!(memory.validate_candidate_authored().is_err());

        let report = VerificationReport {
            report_version: 1,
            run_id: "run".into(),
            candidate_id: "candidate".into(),
            environment_hash: "1".repeat(64),
            source_tree_hash: "2".repeat(64),
            checks: vec![],
            passed: true,
            policy_violations: vec![],
            artifacts: vec![],
            supervisor_attestation: Some("forged".into()),
        };
        assert!(report.validate_candidate_proposal().is_err());
    }

    #[test]
    fn infrastructure_failures_are_censored() {
        let evidence = FailureEvidence {
            domain: FailureDomain::WorkerInfrastructure,
            subcategory: "launcher".into(),
            retryable: true,
            causal_event_hash: None,
            stage: StageId::from("verify"),
            artifacts: vec![],
            counts_against_candidate: true,
        };
        assert!(evidence.validate().is_err());
    }
}
