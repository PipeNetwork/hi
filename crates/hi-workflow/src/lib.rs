pub mod declarative;
pub mod engine;
pub mod host;
pub mod journal;
pub mod meta;
pub mod registry;
pub mod run;
pub mod runtime;
pub mod schema;
pub mod snapshot;
pub mod store;
pub mod trigger;
pub mod validate;

pub const MAX_WORKFLOW_NAME_LEN: usize = 64;
pub const MAX_WORKFLOW_DESCRIPTION_LEN: usize = 1_024;
pub const MAX_WORKFLOW_WHEN_TO_USE_LEN: usize = 2_048;
pub const MAX_WORKFLOW_PHASES: usize = 64;
pub const MAX_PHASE_TITLE_LEN: usize = 128;
pub const MAX_PHASE_DETAIL_LEN: usize = 1_024;
pub const MAX_WORKFLOW_TRIGGERS: usize = 16;
pub const MAX_PARALLEL: usize = 1_024;
pub const DEFAULT_AGENT_BUDGET: u64 = 128;
pub const MAX_AGENT_BUDGET: u64 = 1_024;
pub const MAX_HOST_CALLS: u64 = 10_000;

pub(crate) fn with_rhai_hint(msg: String) -> String {
    let hint = if msg.contains("Expression exceeds maximum complexity") {
        "a single expression nests too deep — usually one long chained `+` string \
         concatenation. Split it into multiple `+=` statements."
    } else if msg.contains("reserved keyword") {
        "Rhai reserves identifiers it doesn't use — `shared`, `sync`, `async`, `await`, \
         `spawn`, `go`, `thread`, `new`, `match`, `case`, `default`, `void`, `null`, \
         `nil`, `exit`, `static`, `var` — rename the variable (`shared` → `has_shared`)."
    } else if msg.contains("getter is not registered for type 'char'") {
        "indexing a string yields a `char`, so field access on it fails — you likely \
         indexed a string you expected to be an array (e.g. unparsed JSON in an agent \
         output). Check with `type_of(x)`; slice strings with `s.sub_string(start, len)`."
    } else {
        return msg;
    };
    format!("{msg}\nhint: {hint}")
}

pub use declarative::{
    AgentJob, AgentStepOutcome, DeclarativeExecutionError, DeclarativeOutcome,
    DeclarativeRunParams, DeclarativeStep, DeclarativeValidationError, DeclarativeWorkflow,
    DeclarativeWorkflowMetadata, MAX_DECLARATIVE_ID_LEN, MAX_DECLARATIVE_STEPS, MAX_TEMPLATE_LEN,
    run_declarative_workflow,
};
pub use engine::{WorkflowRunParams, run_workflow};
pub use host::{AgentOpts, AgentResult, BudgetState, HostError, WorkflowHostRequest};
pub use journal::{Journal, JournalEntry, JournalError};
pub use meta::{MetaError, PhaseMeta, WorkflowMeta, extract_meta};
pub use registry::{
    MAX_WORKFLOW_SOURCE_BYTES, RegisteredWorkflow, RegistryError, WorkflowRegistry, WorkflowSource,
    save_project_workflow, user_workflows_dir, valid_workflow_name,
};
pub use run::{PauseKind, WorkflowOutcome};
pub use runtime::{ManagedWorkflowRun, RuntimeError, WorkflowRuntimeManager};
pub use schema::validate_output_schema;
pub use snapshot::{
    WORKFLOW_HISTORY_MAX, WorkflowAgentSnapshot, WorkflowHistoryEntry, WorkflowPhaseSnapshot,
    WorkflowRunSnapshot, WorkflowRunStatus,
};
pub use store::{
    RUN_MANIFEST_VERSION, StoreError, StoredRunStatus, StoredWorkflowRun, WorkflowRunManifest,
    WorkflowRunStore, validate_run_id,
};
pub use trigger::{
    InMemoryTriggerLedger, TriggerDispatcher, TriggerLedger, TriggerResult, TriggerSpec,
};
pub use validate::{
    ValidationError, ValidationReport, validate_script, validate_script_with_agent_budget,
};
