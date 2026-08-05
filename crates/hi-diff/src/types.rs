use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::SCHEMA_VERSION;

pub type RunId = String;
pub type CaseId = String;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiffMode {
    LocalParity,
    ApiResponse,
    AgentOutcome,
}

impl DiffMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::LocalParity => "local parity",
            Self::ApiResponse => "API response",
            Self::AgentOutcome => "agent outcome",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProbeLevel {
    #[default]
    Summary,
    FullOnFailure,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    Cpu,
    Cuda,
    Mlx,
    Api,
    Agent,
    Custom,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImplementationMetadata {
    pub name: String,
    pub backend: BackendKind,
    pub implementation_version: String,
    pub model_fingerprint: Option<String>,
    pub tokenizer_fingerprint: Option<String>,
    pub details: BTreeMap<String, String>,
}

impl ImplementationMetadata {
    pub fn new(name: impl Into<String>, backend: BackendKind) -> Self {
        Self {
            name: name.into(),
            backend,
            implementation_version: env!("CARGO_PKG_VERSION").to_string(),
            model_fingerprint: None,
            tokenizer_fingerprint: None,
            details: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CheckpointCapabilities {
    pub checkpoints: Vec<String>,
    pub supports_full_values: bool,
}

impl CheckpointCapabilities {
    pub fn supports(&self, name: &str) -> bool {
        self.checkpoints.iter().any(|candidate| candidate == name)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TargetSpec {
    Local(LocalTarget),
    Api(ApiTarget),
    Agent(AgentTarget),
}

impl TargetSpec {
    pub fn name(&self) -> &str {
        match self {
            Self::Local(target) => &target.name,
            Self::Api(target) => &target.name,
            Self::Agent(target) => &target.name,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalTarget {
    pub name: String,
    pub backend: BackendKind,
    pub model_path: PathBuf,
    pub model_fingerprint: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiTarget {
    pub name: String,
    pub profile: String,
    pub model: String,
    pub provider: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentTarget {
    pub name: String,
    pub profile: String,
    pub model: String,
    pub provider: String,
    pub verify_commands: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquivalenceContract {
    pub mode: DiffMode,
    pub required_checkpoints: Vec<String>,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
    pub exact_text: bool,
    pub normalize_whitespace: bool,
    pub require_schema_valid: bool,
    pub require_same_tool_calls: bool,
}

impl Default for EquivalenceContract {
    fn default() -> Self {
        Self {
            mode: DiffMode::LocalParity,
            required_checkpoints: Vec::new(),
            absolute_tolerance: 1e-3,
            relative_tolerance: 1e-3,
            exact_text: false,
            normalize_whitespace: true,
            require_schema_valid: true,
            require_same_tool_calls: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiffRunSpec {
    pub schema_version: u32,
    pub run_id: RunId,
    pub mode: DiffMode,
    pub seed: u64,
    pub case_count: u64,
    pub max_concurrency: usize,
    pub probe: ProbeLevel,
    pub targets: Vec<TargetSpec>,
    pub contract: EquivalenceContract,
    pub artifact_root: Option<PathBuf>,
}

impl DiffRunSpec {
    pub fn new(mode: DiffMode, seed: u64, targets: Vec<TargetSpec>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            run_id: uuid::Uuid::new_v4().to_string(),
            mode,
            seed,
            case_count: 1,
            max_concurrency: 1,
            probe: ProbeLevel::default(),
            targets,
            contract: EquivalenceContract {
                mode,
                ..EquivalenceContract::default()
            },
            artifact_root: None,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema_version == SCHEMA_VERSION,
            "unsupported diff schema version {}",
            self.schema_version
        );
        anyhow::ensure!(
            self.targets.len() >= 2,
            "differential runs need at least two targets"
        );
        anyhow::ensure!(self.case_count > 0, "case_count must be greater than zero");
        anyhow::ensure!(
            self.max_concurrency > 0,
            "max_concurrency must be greater than zero"
        );
        anyhow::ensure!(
            self.contract.mode == self.mode,
            "contract mode does not match run mode"
        );
        let targets_match_mode = self
            .targets
            .iter()
            .all(|target| match (&self.mode, target) {
                (DiffMode::LocalParity, TargetSpec::Local(_))
                | (DiffMode::ApiResponse, TargetSpec::Api(_))
                | (DiffMode::AgentOutcome, TargetSpec::Agent(_)) => true,
                _ => false,
            });
        anyhow::ensure!(
            targets_match_mode,
            "targets do not match diff mode {}",
            self.mode.label()
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DiffCase {
    Local(LocalCase),
    Api(ApiCase),
    Agent(AgentCase),
}

impl DiffCase {
    pub fn id(&self) -> &str {
        match self {
            Self::Local(case) => &case.id,
            Self::Api(case) => &case.id,
            Self::Agent(case) => &case.id,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalCase {
    pub id: CaseId,
    pub input_tokens: Vec<u32>,
    pub decode_steps: usize,
    pub seed: u64,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiCase {
    pub id: CaseId,
    pub request: serde_json::Value,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentCase {
    pub id: CaseId,
    pub task: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorSummary {
    pub shape: Vec<usize>,
    pub len: usize,
    pub nan_count: usize,
    pub inf_count: usize,
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub l2: f32,
    pub samples: Vec<f32>,
}

impl TensorSummary {
    pub fn from_values(shape: Vec<usize>, values: &[f32]) -> Self {
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut min = None;
        let mut max = None;
        let mut l2 = 0.0f64;
        for &value in values {
            if value.is_nan() {
                nan_count += 1;
            } else if value.is_infinite() {
                inf_count += 1;
            } else {
                min = Some(min.map_or(value, |old: f32| old.min(value)));
                max = Some(max.map_or(value, |old: f32| old.max(value)));
                l2 += f64::from(value) * f64::from(value);
            }
        }
        let samples = if values.len() <= 16 {
            values.to_vec()
        } else {
            let mut result = Vec::with_capacity(16);
            for i in 0..16 {
                result.push(values[i * (values.len() - 1) / 15]);
            }
            result
        };
        Self {
            shape,
            len: values.len(),
            nan_count,
            inf_count,
            min,
            max,
            l2: l2.sqrt() as f32,
            samples,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Checkpoint<'a> {
    pub name: &'a str,
    pub step: usize,
    pub shape: Vec<usize>,
    pub values: Option<&'a [f32]>,
    pub summary: TensorSummary,
    pub artifact: Option<ArtifactRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointRecord {
    pub name: String,
    pub step: usize,
    pub summary: TensorSummary,
    pub artifact: Option<ArtifactRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub digest: String,
    pub relative_path: String,
    pub kind: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalOutcome {
    pub generated_tokens: Vec<u32>,
    pub next_token: Option<u32>,
    pub logits: Option<Vec<f32>>,
    pub checkpoints: Vec<CheckpointRecord>,
    /// Full checkpoint values are retained only for the in-memory comparison
    /// of one case and are skipped from persisted JSON artifacts.
    #[serde(skip)]
    pub checkpoint_values: BTreeMap<String, Vec<f32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiOutcome {
    pub text: String,
    pub json: Option<serde_json::Value>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub finish_reason: Option<String>,
    pub error_category: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub latency_ms: u64,
    pub schema_valid: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentOutcome {
    pub success: bool,
    pub patch: String,
    pub changed_files: Vec<String>,
    pub tool_events: Vec<String>,
    pub test_output: String,
    pub verifier_output: String,
    pub timed_out: bool,
    pub policy_violation: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceSnapshot {
    pub root: PathBuf,
    pub source_revision: Option<String>,
    pub dirty_patch: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    Equivalent,
    EquivalentDifferentImplementation,
    Mismatch,
    Inconclusive,
    ExecutionError,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Difference {
    pub location: String,
    pub message: String,
    pub max_error: Option<f32>,
    pub rms_error: Option<f32>,
    pub first_bad_index: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseVerdict {
    pub case_id: CaseId,
    pub verdict: Verdict,
    pub differences: Vec<Difference>,
    pub target_errors: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiffRunSnapshot {
    pub schema_version: u32,
    pub run_id: RunId,
    pub mode: DiffMode,
    pub status: RunStatus,
    pub cases_completed: u64,
    pub cases_total: u64,
    pub mismatches: u64,
    pub errors: u64,
    pub cases_per_second: f64,
    pub recent_failures: Vec<CaseVerdict>,
    pub selected_case: Option<CaseId>,
    pub artifact_root: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl DiffRunSnapshot {
    pub fn pending(spec: &DiffRunSpec) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            run_id: spec.run_id.clone(),
            mode: spec.mode,
            status: RunStatus::Pending,
            cases_completed: 0,
            cases_total: spec.case_count,
            mismatches: 0,
            errors: 0,
            cases_per_second: 0.0,
            recent_failures: Vec::new(),
            selected_case: None,
            artifact_root: spec.artifact_root.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DiffEvent {
    Started(DiffRunSnapshot),
    Progress(DiffRunSnapshot),
    CaseFinished(CaseVerdict),
    Checkpoint {
        case_id: CaseId,
        checkpoint: CheckpointRecord,
    },
    Finished(DiffRunSnapshot),
}
