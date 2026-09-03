use serde::{Deserialize, Serialize};

pub const TASK_OBJECT: &str = "task";
pub const TASK_MIN_COST_USD: f64 = 0.01;
pub const TASK_MAX_COST_USD: f64 = 25.0;
pub const TASK_DEFAULT_DEADLINE_SECS: u64 = 1_800;
/// Interactive clients fail open if a task stays `queued` this long with no worker.
pub const QUEUE_STALL_SECS: u64 = 20;
pub const TASK_MIN_DEADLINE_SECS: u64 = 30;
pub const TASK_MAX_DEADLINE_SECS: u64 = 3_600;
pub const TASK_DEFAULT_ATTEMPTS: u32 = 3;
pub const TASK_REVIEW_RUBRIC_SECURE_RUST_V2: &str = "secure-rust-v2";
pub const TASKS_UNAVAILABLE_CODE: &str = "tasks_unavailable";
pub const ROUTE_QUALITY: &str = "route_quality";
pub const ROUTE_CHEAP: &str = "route_cheap";
pub const TASKS_PATH: &str = "/v1/tasks";
pub const QUOTES_PATH: &str = "/v1/quotes";
pub const REPAIRS_PATH: &str = "/v1/repairs";
pub const VERIFICATIONS_PATH: &str = "/v1/verifications";
pub const RECEIPTS_VERIFY_PATH: &str = "/v1/receipts/verify";
pub const LEDGER_EVENTS_PATH: &str = "/v1/ledger/events";
pub const HI_WORKER_HEARTBEAT_SERVICE: &str = "rsi-hi-worker";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    #[default]
    #[serde(rename = "code.change")]
    CodeChange,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    #[default]
    Queued,
    Planning,
    Executing,
    Verifying,
    Repairing,
    Succeeded,
    Failed,
    BudgetExhausted,
    DeadlineExceeded,
    Canceled,
}

impl TaskStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Planning => "planning",
            Self::Executing => "executing",
            Self::Verifying => "verifying",
            Self::Repairing => "repairing",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::BudgetExhausted => "budget_exhausted",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::Canceled => "canceled",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded
                | Self::Failed
                | Self::BudgetExhausted
                | Self::DeadlineExceeded
                | Self::Canceled
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskVerifierKind {
    CargoTest,
    CargoClippy,
    Review,
    JsonSchema,
}

impl TaskVerifierKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CargoTest => "cargo_test",
            Self::CargoClippy => "cargo_clippy",
            Self::Review => "review",
            Self::JsonSchema => "json_schema",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskVerifier {
    #[serde(rename = "type")]
    pub kind: TaskVerifierKind,
    #[serde(default = "default_true")]
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rubric: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRepositoryInput {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskInputs {
    pub repository: TaskRepositoryInput,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instructions: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskResultContract {
    #[serde(default)]
    pub format: TaskResultFormat,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskResultFormat {
    #[default]
    GitPatch,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskOutcomeContract {
    pub result: TaskResultContract,
    pub verifiers: Vec<TaskVerifier>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeObjective {
    #[default]
    MinimizeCostPerSuccess,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskExecutionPolicy {
    pub maximum_cost_usd: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_seconds: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_attempts: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCreateRequest {
    #[serde(rename = "type")]
    pub task_type: TaskType,
    pub goal: String,
    pub inputs: TaskInputs,
    pub outcome_contract: TaskOutcomeContract,
    pub execution_policy: TaskExecutionPolicy,
    #[serde(default, skip_serializing_if = "is_default_objective")]
    pub objective: OutcomeObjective,
}

fn is_default_objective(value: &OutcomeObjective) -> bool {
    *value == OutcomeObjective::MinimizeCostPerSuccess
}

impl TaskCreateRequest {
    pub fn code_change(
        goal: impl Into<String>,
        repository_id: impl Into<String>,
        verifiers: Vec<TaskVerifier>,
        maximum_cost_usd: f64,
        deadline_seconds: u64,
    ) -> Self {
        Self {
            task_type: TaskType::CodeChange,
            goal: goal.into(),
            inputs: TaskInputs {
                repository: TaskRepositoryInput {
                    id: repository_id.into(),
                    revision: None,
                },
                instructions: Vec::new(),
            },
            outcome_contract: TaskOutcomeContract {
                result: TaskResultContract {
                    format: TaskResultFormat::GitPatch,
                },
                verifiers,
            },
            execution_policy: TaskExecutionPolicy {
                maximum_cost_usd,
                deadline_seconds: Some(deadline_seconds),
                maximum_attempts: Some(TASK_DEFAULT_ATTEMPTS),
            },
            objective: OutcomeObjective::MinimizeCostPerSuccess,
        }
    }
}

pub fn cargo_test_verifier() -> TaskVerifier {
    TaskVerifier {
        kind: TaskVerifierKind::CargoTest,
        required: true,
        rubric: None,
        minimum_score: None,
        schema: None,
    }
}

pub fn cargo_clippy_verifier() -> TaskVerifier {
    TaskVerifier {
        kind: TaskVerifierKind::CargoClippy,
        required: true,
        rubric: None,
        minimum_score: None,
        schema: None,
    }
}

pub fn json_schema_verifier(schema: serde_json::Value) -> TaskVerifier {
    TaskVerifier {
        kind: TaskVerifierKind::JsonSchema,
        required: true,
        rubric: None,
        minimum_score: None,
        schema: Some(schema),
    }
}

pub fn review_verifier() -> TaskVerifier {
    TaskVerifier {
        kind: TaskVerifierKind::Review,
        required: true,
        rubric: Some(TASK_REVIEW_RUBRIC_SECURE_RUST_V2.into()),
        minimum_score: None,
        schema: None,
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct TaskView {
    #[serde(default)]
    pub object: String,
    pub id: String,
    pub status: TaskStatus,
    #[serde(default)]
    pub goal: String,
    #[serde(default)]
    pub maximum_cost_usd: f64,
    #[serde(default)]
    pub result: Option<TaskResult>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub contract_hash: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TaskResult {
    #[serde(default)]
    pub artifact: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TaskEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(default)]
    pub sequence: u64,
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default)]
    pub verifier: Option<String>,
    #[serde(default)]
    pub passed: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TaskEvents {
    #[serde(default)]
    pub events: Vec<TaskEvent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuoteCreateRequest {
    pub requirements: QuoteRequirements,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuoteRequirements {
    pub maximum_cost_usd: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QuoteOffer {
    pub route: String,
    #[serde(default)]
    pub price_cap_usd: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QuoteList {
    #[serde(default)]
    pub quotes: Vec<QuoteOffer>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairCreateRequest {
    pub task_id: String,
    pub remaining_budget_usd: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RepairView {
    pub id: String,
    #[serde(default)]
    pub task_id: String,
    #[serde(default)]
    pub status: TaskStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptVerifyRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ExecutionReceipt {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub task_id: String,
    #[serde(default)]
    pub contract_hash: String,
    #[serde(default)]
    pub outcome: String,
    #[serde(default)]
    pub verified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationCreateRequest {
    pub result: VerificationSubject,
    pub contract: TaskOutcomeContract,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationSubject {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerEventCreateRequest {
    pub task_id: String,
    pub outcome: String,
    pub provider_cost_usd: f64,
    pub customer_price_usd: f64,
    pub billing_unit: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct VerificationView {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub status: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RepositoryCreated {
    #[serde(alias = "id")]
    pub repository_id: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PublicRsiStatus {
    #[serde(default)]
    pub ready: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutcomeOffer {
    Quality,
    Cheap,
}

impl OutcomeOffer {
    pub fn as_route(self) -> &'static str {
        match self {
            Self::Quality => ROUTE_QUALITY,
            Self::Cheap => ROUTE_CHEAP,
        }
    }

    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "cheap" | "route_cheap" => Self::Cheap,
            _ => Self::Quality,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OutcomeMode {
    Auto,
    Tasks,
    /// Ordinary `hi` runs stay on the direct provider route. Outcome tasks
    /// carry paid-job cost/deadline/attempt policies and must be opted into.
    #[default]
    Chat,
}

impl OutcomeMode {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "tasks" | "task" => Self::Tasks,
            "chat" => Self::Chat,
            _ => Self::Chat,
        }
    }
}
