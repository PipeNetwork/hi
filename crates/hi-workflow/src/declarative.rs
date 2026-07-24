use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::{AgentOpts, AgentResult, HostError, MAX_PARALLEL, PauseKind, WorkflowHostRequest};

pub const MAX_DECLARATIVE_STEPS: usize = 1_024;
pub const MAX_DECLARATIVE_ID_LEN: usize = 128;
pub const MAX_TEMPLATE_LEN: usize = 64 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeclarativeWorkflowMetadata {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclarativeWorkflow {
    pub metadata: DeclarativeWorkflowMetadata,
    #[serde(default)]
    pub steps: Vec<DeclarativeStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DeclarativeStep {
    Agent {
        id: String,
        #[serde(flatten)]
        opts: AgentOpts,
    },
    ParallelAgents {
        jobs: Vec<AgentJob>,
    },
    Phase {
        title: String,
    },
    Log {
        message: String,
    },
    Pause {
        kind: PauseKind,
        message: String,
    },
    Complete {
        #[serde(default)]
        result: Value,
    },
    IfAgentSuccess {
        agent: String,
        #[serde(default, rename = "then")]
        then_steps: Vec<DeclarativeStep>,
        #[serde(default, rename = "else")]
        else_steps: Vec<DeclarativeStep>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentJob {
    pub id: String,
    #[serde(flatten)]
    pub opts: AgentOpts,
}

#[derive(Debug, Clone)]
pub struct DeclarativeRunParams {
    pub workflow: DeclarativeWorkflow,
    pub args: Value,
    pub host_tx: mpsc::UnboundedSender<WorkflowHostRequest>,
    pub cancel: CancellationToken,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStepOutcome {
    pub id: String,
    pub result: AgentResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum DeclarativeOutcome {
    Completed {
        result: Value,
        agents: Vec<AgentStepOutcome>,
    },
    Paused {
        kind: PauseKind,
        message: String,
        agents: Vec<AgentStepOutcome>,
    },
    Cancelled {
        agents: Vec<AgentStepOutcome>,
    },
    BudgetExceeded {
        message: String,
        agents: Vec<AgentStepOutcome>,
    },
    Failed {
        error: DeclarativeExecutionError,
        agents: Vec<AgentStepOutcome>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "kind", content = "message", rename_all = "snake_case")]
pub enum DeclarativeExecutionError {
    #[error("invalid workflow: {0}")]
    Validation(String),
    #[error("template error: {0}")]
    Template(String),
    #[error("host channel closed while sending {0}")]
    HostChannelClosed(String),
    #[error("host dropped reply for {0}")]
    HostReplyDropped(String),
    #[error("host error: {0}")]
    Host(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error, PartialEq, Eq)]
#[error("declarative workflow validation failed: {errors:?}")]
pub struct DeclarativeValidationError {
    pub errors: Vec<String>,
}

impl DeclarativeWorkflow {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn validate(&self) -> Result<(), DeclarativeValidationError> {
        let mut errors = Vec::new();
        if self.metadata.name.trim().is_empty() {
            errors.push("metadata.name must not be empty".into());
        }
        if self.metadata.name.len() > crate::MAX_WORKFLOW_NAME_LEN {
            errors.push(format!(
                "metadata.name exceeds {} bytes",
                crate::MAX_WORKFLOW_NAME_LEN
            ));
        }
        if self.metadata.description.len() > crate::MAX_WORKFLOW_DESCRIPTION_LEN {
            errors.push(format!(
                "metadata.description exceeds {} bytes",
                crate::MAX_WORKFLOW_DESCRIPTION_LEN
            ));
        }
        let mut count = 0usize;
        let mut ids = BTreeSet::new();
        validate_steps(&self.steps, &mut count, &mut ids, &mut errors);
        if count > MAX_DECLARATIVE_STEPS {
            errors.push(format!(
                "workflow has {count} steps; maximum is {MAX_DECLARATIVE_STEPS}"
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(DeclarativeValidationError { errors })
        }
    }
}

fn validate_steps(
    steps: &[DeclarativeStep],
    count: &mut usize,
    ids: &mut BTreeSet<String>,
    errors: &mut Vec<String>,
) {
    for step in steps {
        *count = count.saturating_add(1);
        match step {
            DeclarativeStep::Agent { id, opts } => validate_job(id, opts, ids, errors),
            DeclarativeStep::ParallelAgents { jobs } => {
                if jobs.is_empty() {
                    errors.push("parallel_agents.jobs must not be empty".into());
                }
                if jobs.len() > MAX_PARALLEL {
                    errors.push(format!(
                        "parallel_agents has {} jobs; maximum is {MAX_PARALLEL}",
                        jobs.len()
                    ));
                }
                for job in jobs {
                    validate_job(&job.id, &job.opts, ids, errors);
                }
            }
            DeclarativeStep::Phase { title } => validate_template("phase.title", title, errors),
            DeclarativeStep::Log { message } => validate_template("log.message", message, errors),
            DeclarativeStep::Pause { message, .. } => {
                validate_template("pause.message", message, errors)
            }
            DeclarativeStep::Complete { .. } => {}
            DeclarativeStep::IfAgentSuccess {
                agent,
                then_steps,
                else_steps,
            } => {
                if !ids.contains(agent) {
                    errors.push(format!(
                        "condition references agent `{agent}` before it is defined"
                    ));
                }
                validate_steps(then_steps, count, ids, errors);
                validate_steps(else_steps, count, ids, errors);
            }
        }
    }
}

fn validate_job(id: &str, opts: &AgentOpts, ids: &mut BTreeSet<String>, errors: &mut Vec<String>) {
    if id.is_empty() || id.len() > MAX_DECLARATIVE_ID_LEN {
        errors.push(format!(
            "agent id `{id}` must be 1..={MAX_DECLARATIVE_ID_LEN} bytes"
        ));
    }
    if !ids.insert(id.to_owned()) {
        errors.push(format!("duplicate agent id `{id}`"));
    }
    validate_template(&format!("agent `{id}` prompt"), &opts.prompt, errors);
}

fn validate_template(field: &str, template: &str, errors: &mut Vec<String>) {
    if template.len() > MAX_TEMPLATE_LEN {
        errors.push(format!("{field} exceeds {MAX_TEMPLATE_LEN} bytes"));
    }
    if let Err(error) = template_keys(template) {
        errors.push(format!("{field}: {error}"));
    }
}

pub async fn run_declarative_workflow(params: DeclarativeRunParams) -> DeclarativeOutcome {
    if let Err(error) = params.workflow.validate() {
        return DeclarativeOutcome::Failed {
            error: DeclarativeExecutionError::Validation(error.to_string()),
            agents: Vec::new(),
        };
    }
    let mut executor = Executor {
        args: params.args,
        host_tx: params.host_tx,
        cancel: params.cancel,
        results: BTreeMap::new(),
        ordered: Vec::new(),
    };
    match executor.run_steps(&params.workflow.steps).await {
        Ok(Control::Continue) => DeclarativeOutcome::Completed {
            result: Value::Null,
            agents: executor.ordered,
        },
        Ok(Control::Complete(result)) => DeclarativeOutcome::Completed {
            result,
            agents: executor.ordered,
        },
        Ok(Control::Pause(kind, message)) => DeclarativeOutcome::Paused {
            kind,
            message,
            agents: executor.ordered,
        },
        Err(RunError::Cancelled) => DeclarativeOutcome::Cancelled {
            agents: executor.ordered,
        },
        Err(RunError::Budget(message)) => DeclarativeOutcome::BudgetExceeded {
            message,
            agents: executor.ordered,
        },
        Err(RunError::Failed(error)) => DeclarativeOutcome::Failed {
            error,
            agents: executor.ordered,
        },
    }
}

struct Executor {
    args: Value,
    host_tx: mpsc::UnboundedSender<WorkflowHostRequest>,
    cancel: CancellationToken,
    results: BTreeMap<String, AgentResult>,
    ordered: Vec<AgentStepOutcome>,
}

enum Control {
    Continue,
    Complete(Value),
    Pause(PauseKind, String),
}
enum RunError {
    Cancelled,
    Budget(String),
    Failed(DeclarativeExecutionError),
}

impl Executor {
    fn run_steps<'a>(
        &'a mut self,
        steps: &'a [DeclarativeStep],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Control, RunError>> + Send + 'a>> {
        Box::pin(async move {
            for step in steps {
                self.check_cancel()?;
                let control = match step {
                    DeclarativeStep::Agent { id, opts } => {
                        self.run_agents(&[AgentJob {
                            id: id.clone(),
                            opts: opts.clone(),
                        }])
                        .await?;
                        Control::Continue
                    }
                    DeclarativeStep::ParallelAgents { jobs } => {
                        self.run_agents(jobs).await?;
                        Control::Continue
                    }
                    DeclarativeStep::Phase { title } => {
                        let title = self.interpolate(title)?;
                        self.send(
                            WorkflowHostRequest::Phase {
                                title,
                                replayed: false,
                            },
                            "phase",
                        )?;
                        Control::Continue
                    }
                    DeclarativeStep::Log { message } => {
                        let message = self.interpolate(message)?;
                        self.send(
                            WorkflowHostRequest::Log {
                                message,
                                replayed: false,
                            },
                            "log",
                        )?;
                        Control::Continue
                    }
                    DeclarativeStep::Pause { kind, message } => {
                        Control::Pause(*kind, self.interpolate(message)?)
                    }
                    DeclarativeStep::Complete { result } => {
                        Control::Complete(interpolate_value(result, &self.args, &self.results)?)
                    }
                    DeclarativeStep::IfAgentSuccess {
                        agent,
                        then_steps,
                        else_steps,
                    } => {
                        let success = self
                            .results
                            .get(agent)
                            .is_some_and(|result| result.success && !result.cancelled);
                        self.run_steps(if success { then_steps } else { else_steps })
                            .await?
                    }
                };
                if !matches!(control, Control::Continue) {
                    return Ok(control);
                }
            }
            Ok(Control::Continue)
        })
    }

    async fn run_agents(&mut self, jobs: &[AgentJob]) -> Result<(), RunError> {
        let count = u64::try_from(jobs.len()).map_err(|_| {
            RunError::Failed(DeclarativeExecutionError::Validation(
                "agent count overflowed".into(),
            ))
        })?;
        self.request_reservation(count).await?;
        let mut pending = Vec::with_capacity(jobs.len());
        for job in jobs {
            if let Err(error) = self.check_cancel() {
                self.release(count).await;
                return Err(error);
            }
            let mut opts = job.opts.clone();
            opts.prompt = self.interpolate(&opts.prompt)?;
            if let Some(label) = &opts.label {
                opts.label = Some(self.interpolate(label)?);
            }
            let (reply, rx) = oneshot::channel();
            if self
                .send(
                    WorkflowHostRequest::SpawnAgent { opts, reply },
                    "spawn_agent",
                )
                .is_err()
            {
                self.release(count).await;
                return Err(RunError::Failed(
                    DeclarativeExecutionError::HostChannelClosed("spawn_agent".into()),
                ));
            }
            pending.push((job.id.clone(), rx));
        }
        let mut first_error = None;
        for (id, rx) in pending {
            let response = tokio::select! { biased; _ = self.cancel.cancelled() => Err(RunError::Cancelled), value = rx => map_reply(value, "spawn_agent") };
            match response {
                Ok(result) => {
                    self.results.insert(id.clone(), result.clone());
                    self.ordered.push(AgentStepOutcome { id, result });
                }
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        self.release(count).await;
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    async fn request_reservation(&self, count: u64) -> Result<(), RunError> {
        let (reply, rx) = oneshot::channel();
        self.send(
            WorkflowHostRequest::ReserveAgentCalls { count, reply },
            "reserve_agent_calls",
        )?;
        tokio::select! { biased; _ = self.cancel.cancelled() => Err(RunError::Cancelled), value = rx => map_unit_reply(value, "reserve_agent_calls") }
    }

    async fn release(&self, count: u64) {
        let (reply, rx) = oneshot::channel();
        if self
            .host_tx
            .send(WorkflowHostRequest::ReleaseAgentCalls { count, reply })
            .is_ok()
        {
            let _ = rx.await;
        }
    }

    fn check_cancel(&self) -> Result<(), RunError> {
        if self.cancel.is_cancelled() {
            Err(RunError::Cancelled)
        } else {
            Ok(())
        }
    }
    fn send(&self, request: WorkflowHostRequest, kind: &str) -> Result<(), RunError> {
        self.host_tx.send(request).map_err(|_| {
            RunError::Failed(DeclarativeExecutionError::HostChannelClosed(kind.into()))
        })
    }
    fn interpolate(&self, input: &str) -> Result<String, RunError> {
        interpolate(input, &self.args, &self.results).map_err(RunError::Failed)
    }
}

fn map_reply(
    value: Result<Result<AgentResult, HostError>, oneshot::error::RecvError>,
    kind: &str,
) -> Result<AgentResult, RunError> {
    match value {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(error)) => Err(map_host(error)),
        Err(_) => Err(RunError::Failed(
            DeclarativeExecutionError::HostReplyDropped(kind.into()),
        )),
    }
}
fn map_unit_reply(
    value: Result<Result<(), HostError>, oneshot::error::RecvError>,
    kind: &str,
) -> Result<(), RunError> {
    match value {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(map_host(error)),
        Err(_) => Err(RunError::Failed(
            DeclarativeExecutionError::HostReplyDropped(kind.into()),
        )),
    }
}
fn map_host(error: HostError) -> RunError {
    match error {
        HostError::Cancelled => RunError::Cancelled,
        HostError::BudgetExceeded => RunError::Budget("workflow agent budget exceeded".into()),
        HostError::AgentCallQuotaExceeded { requested, maximum } => RunError::Budget(format!(
            "workflow agent budget exceeded: requested {requested}, maximum {maximum}"
        )),
        other => RunError::Failed(DeclarativeExecutionError::Host(other.to_string())),
    }
}

fn interpolate_value(
    value: &Value,
    args: &Value,
    results: &BTreeMap<String, AgentResult>,
) -> Result<Value, RunError> {
    match value {
        Value::String(value) => interpolate(value, args, results)
            .map(Value::String)
            .map_err(RunError::Failed),
        Value::Array(values) => values
            .iter()
            .map(|value| interpolate_value(value, args, results))
            .collect(),
        Value::Object(values) => values
            .iter()
            .map(|(key, value)| Ok((key.clone(), interpolate_value(value, args, results)?)))
            .collect(),
        value => Ok(value.clone()),
    }
}

fn interpolate(
    input: &str,
    args: &Value,
    results: &BTreeMap<String, AgentResult>,
) -> Result<String, DeclarativeExecutionError> {
    let mut output = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find("{{") {
        output.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after
            .find("}}")
            .ok_or_else(|| DeclarativeExecutionError::Template("unclosed `{{`".into()))?;
        let key = after[..end].trim();
        let value = lookup(key, args, results).ok_or_else(|| {
            DeclarativeExecutionError::Template(format!("unknown variable `{key}`"))
        })?;
        output.push_str(&display_value(value));
        rest = &after[end + 2..];
    }
    output.push_str(rest);
    Ok(output)
}

fn template_keys(input: &str) -> Result<Vec<&str>, &'static str> {
    let mut keys = Vec::new();
    let mut rest = input;
    while let Some(start) = rest.find("{{") {
        let after = &rest[start + 2..];
        let end = after.find("}}").ok_or("unclosed `{{`")?;
        let key = after[..end].trim();
        if key.is_empty() {
            return Err("empty template variable");
        }
        keys.push(key);
        rest = &after[end + 2..];
    }
    Ok(keys)
}

fn lookup<'a>(
    key: &str,
    args: &'a Value,
    results: &'a BTreeMap<String, AgentResult>,
) -> Option<&'a Value> {
    let (root, path) = key.split_once('.').unwrap_or((key, ""));
    let mut value = if root == "args" {
        args
    } else if root == "agents" {
        let (id, tail) = path.split_once('.').unwrap_or((path, ""));
        let result = results.get(id)?;
        if tail == "output" {
            return Some(&result.output);
        }
        if let Some(tail) = tail.strip_prefix("output.") {
            return descend(&result.output, tail);
        }
        return None;
    } else {
        return None;
    };
    if !path.is_empty() {
        value = descend(value, path)?;
    }
    Some(value)
}
fn descend<'a>(mut value: &'a Value, path: &str) -> Option<&'a Value> {
    for part in path.split('.') {
        value = value.get(part)?;
    }
    Some(value)
}
fn display_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => "null".into(),
        other => other.to_string(),
    }
}
