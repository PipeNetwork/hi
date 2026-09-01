//! Opt-in, semantic-event workflow triggers.
//!
//! The dispatcher claims a source event before starting a workflow. The CLI
//! supplies the durable project ledger; an in-memory ledger is provided for
//! embedders and tests.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use hi_events::{
    ActivityObject, ActivityState, ActivityVerb, EventContext, EventKind, EventSink, RunEvent,
    SemanticActivity,
};
use hi_policy::{ApprovalStore, CapabilityKind, OperationDigest, ResourceScope};
use serde::{Deserialize, Serialize};

use crate::{RegisteredWorkflow, WorkflowRegistry, WorkflowRuntimeManager, WorkflowSource};

pub const MAX_TRIGGER_ID_LEN: usize = 96;
pub const MAX_TRIGGER_EVENT_LEN: usize = 64;
pub const MAX_TRIGGER_PREDICATES: usize = 16;
pub const MAX_TRIGGER_ARGUMENTS: usize = 16;
pub const MAX_TRIGGER_DEPTH: u8 = 4;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TriggerSpec {
    pub id: String,
    /// Snake-case semantic event name, for example `verification_completed`.
    pub event: String,
    /// Equality predicates over bounded fields such as `activity.state`.
    #[serde(default)]
    pub when: BTreeMap<String, String>,
    /// Workflow argument name -> approved event field path.
    #[serde(default)]
    pub arguments: BTreeMap<String, String>,
    #[serde(default)]
    pub cooldown_secs: u64,
    /// A literal key or a bounded event field path.
    #[serde(default)]
    pub concurrency_key: Option<String>,
    #[serde(default)]
    pub approval_required: bool,
    /// Triggers are inert until explicitly enabled in project settings.
    #[serde(default)]
    pub enabled: bool,
}

impl TriggerSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() || self.id.len() > MAX_TRIGGER_ID_LEN {
            return Err(format!("trigger id must be 1..{MAX_TRIGGER_ID_LEN} bytes"));
        }
        if self.event.len() > MAX_TRIGGER_EVENT_LEN || event_kind(&self.event).is_none() {
            return Err(format!("unsupported trigger event {:?}", self.event));
        }
        if self.when.len() > MAX_TRIGGER_PREDICATES {
            return Err(format!(
                "at most {MAX_TRIGGER_PREDICATES} trigger predicates are allowed"
            ));
        }
        if self.arguments.len() > MAX_TRIGGER_ARGUMENTS {
            return Err(format!(
                "at most {MAX_TRIGGER_ARGUMENTS} trigger arguments are allowed"
            ));
        }
        for (field, value) in &self.when {
            if field.len() > 128 || value.len() > 512 {
                return Err("trigger field names and values are bounded".into());
            }
            if field.trim().is_empty() {
                return Err("trigger argument names must not be empty".into());
            }
            if !approved_field(field) {
                return Err(format!(
                    "trigger field {field:?} is not an approved event field"
                ));
            }
        }
        for (field, value) in &self.arguments {
            if field.len() > 128 || value.len() > 512 {
                return Err("trigger field names and values are bounded".into());
            }
            if field.trim().is_empty() {
                return Err("trigger argument names must not be empty".into());
            }
            if !approved_field(value) {
                return Err(format!(
                    "trigger field {value:?} is not an approved event field"
                ));
            }
        }
        if let Some(key) = &self.concurrency_key
            && key.len() > 128
        {
            return Err("trigger concurrency keys are bounded".into());
        }
        Ok(())
    }

    pub fn matches(&self, event: &RunEvent) -> bool {
        event_kind(&self.event).is_some_and(|kind| kind == event.kind)
            && self.when.iter().all(|(field, expected)| {
                event_field(event, field).is_some_and(|actual| actual == *expected)
            })
    }

    pub fn extract_arguments(&self, event: &RunEvent) -> serde_json::Value {
        let mut args = serde_json::Map::new();
        for (name, field) in &self.arguments {
            if let Some(value) = event_field(event, field) {
                args.insert(name.clone(), serde_json::Value::String(value));
            }
        }
        args.insert("source_event_id".into(), event.event_id.clone().into());
        args.insert("source_sequence".into(), event.sequence.into());
        serde_json::Value::Object(args)
    }
}

/// Durable delivery/checkpoint state owned by the project state layer.
pub trait TriggerLedger: Send + Sync {
    fn claim(&self, trigger_id: &str, source_event_id: &str) -> Result<bool, String>;
    fn high_watermark(&self) -> Result<u64, String>;
    fn set_high_watermark(&self, sequence: u64) -> Result<(), String>;
    fn last_fired(&self, trigger_id: &str, key: &str) -> Result<Option<u64>, String>;
    fn record_fired(&self, trigger_id: &str, key: &str, at_ms: u64) -> Result<(), String>;
    fn mark_failed(&self, trigger_id: &str, source_event_id: &str) -> Result<(), String> {
        let _ = (trigger_id, source_event_id);
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryTriggerLedger {
    deliveries: Arc<std::sync::Mutex<HashSet<(String, String)>>>,
    fired: Arc<std::sync::Mutex<HashMap<(String, String), u64>>>,
    high_watermark: Arc<std::sync::atomic::AtomicU64>,
}

impl TriggerLedger for InMemoryTriggerLedger {
    fn claim(&self, trigger_id: &str, source_event_id: &str) -> Result<bool, String> {
        Ok(self
            .deliveries
            .lock()
            .map_err(|_| "trigger ledger lock poisoned")?
            .insert((trigger_id.into(), source_event_id.into())))
    }

    fn high_watermark(&self) -> Result<u64, String> {
        Ok(self
            .high_watermark
            .load(std::sync::atomic::Ordering::Acquire))
    }

    fn set_high_watermark(&self, sequence: u64) -> Result<(), String> {
        self.high_watermark
            .fetch_max(sequence, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    fn last_fired(&self, trigger_id: &str, key: &str) -> Result<Option<u64>, String> {
        Ok(self
            .fired
            .lock()
            .map_err(|_| "trigger ledger lock poisoned")?
            .get(&(trigger_id.into(), key.into()))
            .copied())
    }

    fn record_fired(&self, trigger_id: &str, key: &str, at_ms: u64) -> Result<(), String> {
        self.fired
            .lock()
            .map_err(|_| "trigger ledger lock poisoned")?
            .insert((trigger_id.into(), key.into()), at_ms);
        Ok(())
    }

    fn mark_failed(&self, trigger_id: &str, source_event_id: &str) -> Result<(), String> {
        self.deliveries
            .lock()
            .map_err(|_| "trigger ledger lock poisoned")?
            .remove(&(trigger_id.into(), source_event_id.into()));
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriggerResult {
    Disabled {
        trigger_id: String,
    },
    Skipped {
        trigger_id: String,
        reason: String,
    },
    Started {
        trigger_id: String,
        run_id: String,
    },
    PausedForApproval {
        trigger_id: String,
        run_id: String,
        approval_id: String,
    },
    Failed {
        trigger_id: String,
        error: String,
    },
}

pub struct TriggerDispatcher {
    ledger: Arc<dyn TriggerLedger>,
    sink: Option<Arc<dyn EventSink>>,
    approval_store: Option<Arc<dyn ApprovalStore>>,
    enabled: bool,
    trusted_workspace: bool,
    active_keys: HashSet<String>,
    active_runs: HashMap<String, String>,
}

impl TriggerDispatcher {
    pub fn new(
        ledger: Arc<dyn TriggerLedger>,
        sink: Option<Arc<dyn EventSink>>,
        enabled: bool,
        trusted_workspace: bool,
    ) -> Self {
        Self {
            ledger,
            sink,
            approval_store: None,
            enabled,
            trusted_workspace,
            active_keys: HashSet::new(),
            active_runs: HashMap::new(),
        }
    }

    pub fn with_approval_store(mut self, store: Arc<dyn ApprovalStore>) -> Self {
        self.approval_store = Some(store);
        self
    }

    /// Seed the checkpoint at enablement so enabling a trigger never backfills
    /// historical events.
    pub fn enable_from(&mut self, current_sequence: u64) -> Result<(), String> {
        self.enabled = true;
        self.ledger.set_high_watermark(current_sequence)
    }

    pub fn high_watermark(&self) -> Result<u64, String> {
        self.ledger.high_watermark()
    }

    /// Replay canonical events after the persisted checkpoint. The input is
    /// sorted defensively because remote/event-bus adapters may deliver a
    /// batch in arrival order rather than stream order.
    pub fn replay<I>(
        &mut self,
        events: I,
        registry: &WorkflowRegistry,
        runtime: &mut WorkflowRuntimeManager,
    ) -> Vec<TriggerResult>
    where
        I: IntoIterator<Item = RunEvent>,
    {
        let checkpoint = self.high_watermark().unwrap_or(0);
        let mut events = events.into_iter().collect::<Vec<_>>();
        events.sort_by_key(|event| event.sequence);
        events
            .into_iter()
            .filter(|event| event.sequence == 0 || event.sequence > checkpoint)
            .flat_map(|event| self.dispatch_event(&event, registry, runtime))
            .collect()
    }

    pub fn dispatch_event(
        &mut self,
        event: &RunEvent,
        registry: &WorkflowRegistry,
        runtime: &mut WorkflowRuntimeManager,
    ) -> Vec<TriggerResult> {
        if !self.enabled {
            if event.sequence > 0 {
                let _ = self.ledger.set_high_watermark(event.sequence);
            }
            return vec![TriggerResult::Disabled {
                trigger_id: "project-triggers".into(),
            }];
        }
        if matches!(
            event.kind,
            EventKind::WorkflowCompleted | EventKind::WorkflowFailed
        ) && let Some(run_id) = event.context.workflow_id.as_deref()
            && let Some(key) = self.active_runs.remove(run_id)
        {
            self.active_keys.remove(&key);
        }

        let mut results = Vec::new();
        for workflow in registry.list() {
            if !eligible(workflow, self.trusted_workspace) {
                continue;
            }
            for spec in &workflow.meta.triggers {
                let trigger_id = format!("{}:{}", workflow.name, spec.id);
                if !spec.enabled || !spec.matches(event) {
                    continue;
                }
                let key = concurrency_key(spec, event);
                if !self
                    .ledger
                    .claim(&trigger_id, &event.event_id)
                    .unwrap_or(false)
                {
                    results.push(TriggerResult::Skipped {
                        trigger_id,
                        reason: "duplicate source event".into(),
                    });
                    continue;
                }
                let cooldown_ms = spec.cooldown_secs.saturating_mul(1_000);
                if cooldown_ms > 0
                    && self
                        .ledger
                        .last_fired(&trigger_id, &key)
                        .ok()
                        .flatten()
                        .is_some_and(|last| event.occurred_at_ms.saturating_sub(last) < cooldown_ms)
                {
                    self.emit(event, EventKind::TriggerSkipped, &trigger_id, "cooldown");
                    results.push(TriggerResult::Skipped {
                        trigger_id,
                        reason: "cooldown active".into(),
                    });
                    continue;
                }
                if !self.active_keys.insert(key.clone()) {
                    self.emit(
                        event,
                        EventKind::TriggerSkipped,
                        &trigger_id,
                        "concurrency key active",
                    );
                    results.push(TriggerResult::Skipped {
                        trigger_id,
                        reason: "concurrency key active".into(),
                    });
                    continue;
                }
                self.emit(event, EventKind::TriggerAccepted, &trigger_id, "accepted");
                if spec.approval_required {
                    self.active_keys.remove(&key);
                    let Some(approval_store) = self.approval_store.clone() else {
                        self.emit(
                            event,
                            EventKind::TriggerSkipped,
                            &trigger_id,
                            "approval required",
                        );
                        results.push(TriggerResult::Skipped {
                            trigger_id,
                            reason: "approval required; background trigger will not auto-approve"
                                .into(),
                        });
                        continue;
                    };
                    let args = spec.extract_arguments(event);
                    let workspace_id = event
                        .context
                        .workspace_id
                        .clone()
                        .unwrap_or_else(|| "project".into());
                    let run_id = runtime.allocate_run_id();
                    let scope = ResourceScope::Workflow {
                        workflow_id: workflow.name.clone(),
                        run_id: run_id.clone(),
                    };
                    let digest = OperationDigest::calculate(
                        &CapabilityKind::WorkflowExecution,
                        "workflow",
                        &serde_json::json!({
                            "workflow": workflow.name,
                            "script": workflow.script,
                            "args": args,
                        }),
                        &workspace_id,
                        &scope,
                        None,
                    );
                    let request = hi_policy::approval_request(
                        CapabilityKind::WorkflowExecution,
                        scope,
                        digest.clone(),
                        "workflow",
                        Some(run_id.clone()),
                        event.context.session_id.clone(),
                        format!("Approve triggered workflow {}", workflow.name),
                        "background workflow approval requested".to_string(),
                    );
                    let Ok(record) = approval_store.create(request) else {
                        self.emit(
                            event,
                            EventKind::TriggerFailed,
                            &trigger_id,
                            "approval unavailable",
                        );
                        let _ = self.ledger.mark_failed(&trigger_id, &event.event_id);
                        results.push(TriggerResult::Failed {
                            trigger_id,
                            error: "approval store unavailable".into(),
                        });
                        continue;
                    };
                    match runtime.start_paused_for_approval(
                        run_id.clone(),
                        workflow.name.clone(),
                        workflow.script.clone(),
                        spec.extract_arguments(event),
                        crate::DEFAULT_AGENT_BUDGET,
                        (&record.request.approval_id.0, &digest.0),
                    ) {
                        Ok(run_id) => {
                            let _ =
                                self.ledger
                                    .record_fired(&trigger_id, &key, event.occurred_at_ms);
                            self.emit(
                                event,
                                EventKind::TriggerSkipped,
                                &trigger_id,
                                "approval pending",
                            );
                            results.push(TriggerResult::PausedForApproval {
                                trigger_id,
                                run_id,
                                approval_id: record.request.approval_id.0,
                            });
                        }
                        Err(error) => {
                            let _ = approval_store.abandon_run(&run_id);
                            self.emit(
                                event,
                                EventKind::TriggerFailed,
                                &trigger_id,
                                "workflow pause failed",
                            );
                            let _ = self.ledger.mark_failed(&trigger_id, &event.event_id);
                            results.push(TriggerResult::Failed {
                                trigger_id,
                                error: error.to_string(),
                            });
                        }
                    }
                    continue;
                }
                let args = spec.extract_arguments(event);
                match runtime.start(
                    workflow.name.clone(),
                    workflow.script.clone(),
                    args,
                    crate::DEFAULT_AGENT_BUDGET,
                ) {
                    Ok(run_id) => {
                        let _ = self
                            .ledger
                            .record_fired(&trigger_id, &key, event.occurred_at_ms);
                        self.active_runs.insert(run_id.clone(), key);
                        self.emit(event, EventKind::TriggerStarted, &trigger_id, "started");
                        results.push(TriggerResult::Started { trigger_id, run_id });
                    }
                    Err(error) => {
                        self.active_keys.remove(&key);
                        self.emit(
                            event,
                            EventKind::TriggerFailed,
                            &trigger_id,
                            "workflow start failed",
                        );
                        let _ = self.ledger.mark_failed(&trigger_id, &event.event_id);
                        results.push(TriggerResult::Failed {
                            trigger_id,
                            error: error.to_string(),
                        });
                    }
                }
            }
        }
        if !results
            .iter()
            .any(|result| matches!(result, TriggerResult::Failed { .. }))
            && event.sequence > 0
        {
            let _ = self.ledger.set_high_watermark(event.sequence);
        }
        results
    }

    fn emit(&self, source: &RunEvent, kind: EventKind, trigger_id: &str, detail: &str) {
        let Some(sink) = &self.sink else { return };
        let event = RunEvent::new(
            kind,
            EventContext {
                parent_event_id: Some(source.event_id.clone()),
                correlation_id: Some(trigger_id.to_string()),
                ..source.context.clone()
            },
            SemanticActivity {
                verb: ActivityVerb::Trigger,
                object: ActivityObject::Trigger,
                state: if detail == "started" {
                    ActivityState::Running
                } else if detail == "workflow start failed" {
                    ActivityState::Failed
                } else {
                    ActivityState::Waiting
                },
                group_key: format!("trigger:{trigger_id}:{}", source.event_id),
                title: format!("trigger {detail}"),
                detail: Some(trigger_id.to_string()),
                refs: Vec::new(),
                progress: None,
            },
        );
        let _ = sink.publish(event);
    }
}

fn eligible(workflow: &RegisteredWorkflow, trusted: bool) -> bool {
    match &workflow.source {
        WorkflowSource::Builtin => true,
        WorkflowSource::Project(_) => trusted,
        WorkflowSource::User(_) => false,
    }
}

fn event_kind(name: &str) -> Option<EventKind> {
    [
        EventKind::VerificationCompleted,
        EventKind::RunCompleted,
        EventKind::RunFailed,
        EventKind::WorkflowCompleted,
        EventKind::WorkflowFailed,
        EventKind::ApprovalDecided,
        EventKind::LoopFired,
        EventKind::GitChanged,
    ]
    .into_iter()
    .find(|kind| {
        serde_json::to_value(kind).ok().and_then(|value| {
            value
                .get("type")
                .and_then(|value| value.as_str())
                .map(str::to_owned)
        }) == Some(name.to_string())
    })
}

fn approved_field(field: &str) -> bool {
    matches!(
        field,
        "event_id"
            | "sequence"
            | "occurred_at_ms"
            | "activity.state"
            | "activity.verb"
            | "activity.object"
            | "activity.group_key"
            | "context.workspace_id"
            | "context.session_id"
            | "context.run_id"
            | "context.workflow_id"
            | "payload.source"
            | "payload.status"
            | "payload.branch"
    )
}

fn event_field(event: &RunEvent, field: &str) -> Option<String> {
    match field {
        "event_id" => Some(event.event_id.clone()),
        "sequence" => Some(event.sequence.to_string()),
        "occurred_at_ms" => Some(event.occurred_at_ms.to_string()),
        "activity.state" => serde_json::to_value(&event.activity.state)
            .ok()?
            .as_str()
            .map(str::to_owned),
        "activity.verb" => serde_json::to_value(&event.activity.verb)
            .ok()?
            .as_str()
            .map(str::to_owned),
        "activity.object" => serde_json::to_value(&event.activity.object)
            .ok()?
            .as_str()
            .map(str::to_owned),
        "activity.group_key" => Some(event.activity.group_key.clone()),
        "context.workspace_id" => event.context.workspace_id.clone(),
        "context.session_id" => event.context.session_id.clone(),
        "context.run_id" => event.context.run_id.clone(),
        "context.workflow_id" => event.context.workflow_id.clone(),
        _ => field.strip_prefix("payload.").and_then(|key| {
            event
                .payload
                .fields
                .get(key)
                .and_then(|value| value.as_str().map(str::to_owned))
        }),
    }
}

fn concurrency_key(spec: &TriggerSpec, event: &RunEvent) -> String {
    spec.concurrency_key
        .as_deref()
        .and_then(|key| event_field(event, key).or_else(|| Some(key.to_string())))
        .unwrap_or_else(|| "project".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event() -> RunEvent {
        RunEvent::new(
            EventKind::VerificationCompleted,
            EventContext::default(),
            SemanticActivity {
                verb: ActivityVerb::Verify,
                object: ActivityObject::Verification,
                state: ActivityState::Succeeded,
                group_key: "verify:1".into(),
                title: "verification passed".into(),
                detail: None,
                refs: vec![],
                progress: None,
            },
        )
    }

    #[test]
    fn trigger_matches_only_bounded_semantic_fields() {
        let trigger = TriggerSpec {
            id: "after-verify".into(),
            event: "verification_completed".into(),
            when: [("activity.state".into(), "succeeded".into())]
                .into_iter()
                .collect(),
            arguments: [("source".into(), "event_id".into())].into_iter().collect(),
            cooldown_secs: 0,
            concurrency_key: None,
            approval_required: false,
            enabled: true,
        };
        trigger.validate().unwrap();
        let source = event();
        assert!(trigger.matches(&source));
        assert_eq!(
            trigger.extract_arguments(&source)["source"],
            source.event_id
        );
    }

    #[test]
    fn unknown_or_raw_fields_are_rejected() {
        let trigger = TriggerSpec {
            id: "x".into(),
            event: "verification_completed".into(),
            when: BTreeMap::new(),
            arguments: [("raw".into(), "payload.output".into())]
                .into_iter()
                .collect(),
            cooldown_secs: 0,
            concurrency_key: None,
            approval_required: false,
            enabled: false,
        };
        assert!(trigger.validate().is_err());
    }

    #[test]
    fn in_memory_ledger_deduplicates() {
        let ledger = InMemoryTriggerLedger::default();
        assert!(ledger.claim("t", "e").unwrap());
        assert!(!ledger.claim("t", "e").unwrap());
    }
}
