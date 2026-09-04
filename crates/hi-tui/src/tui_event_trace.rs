//! Structured, redacted lifecycle trace for interactive TUI smoke tests.
//!
//! This is deliberately separate from delegate `--events-jsonl`: delegate
//! events are a parent/child progress protocol, while these records describe
//! the assembled interactive frontend. Every record is flushed before the
//! call returns so an abruptly terminated PTY still leaves useful evidence.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use serde_json::{Value, json};

use crate::event::UiEvent;

pub const TUI_EVENT_TRACE_SCHEMA_VERSION: u32 = 1;
const SMOKE_RUN_MARKER_ENV: &str = "HI_SMOKE_RUN_MARKER";

#[derive(Clone)]
pub struct TuiEventTrace {
    inner: Arc<Mutex<TraceState>>,
}

struct TraceState {
    writer: BufWriter<File>,
    sequence: u64,
    run_id: Option<String>,
    failure: Option<String>,
}

#[derive(Serialize)]
struct TraceRecord<'a> {
    schema_version: u32,
    sequence: u64,
    process_id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_id: Option<&'a str>,
    timestamp_ms: u64,
    event: &'a str,
    data: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PromptOrigin {
    User,
    PlanDrive,
    GoalDrive,
    CommandFollowUp,
}

impl PromptOrigin {
    pub(crate) fn from_prompt(prompt: &str) -> Self {
        match hi_agent::DriveKind::from_prompt(prompt) {
            hi_agent::DriveKind::Plan => Self::PlanDrive,
            hi_agent::DriveKind::Goal => Self::GoalDrive,
            hi_agent::DriveKind::User if prompt.trim_start().starts_with('/') => {
                Self::CommandFollowUp
            }
            hi_agent::DriveKind::User => Self::User,
        }
    }
}

impl TuiEventTrace {
    /// Create or append to a TUI trace. Parent directories are created so a
    /// harness can point directly at a fresh artifact bundle. Sequence numbers
    /// continue from the largest valid prior record, preserving evidence when
    /// a smoke scenario quits and restarts against the same artifact path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_run_id(path, smoke_run_id_from_env())
    }

    fn open_with_run_id(path: impl AsRef<Path>, run_id: Option<String>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("creating TUI event trace directory {}", parent.display())
            })?;
        }
        let (sequence, needs_separator) = prior_trace_state(path)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening TUI event trace {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        if needs_separator {
            writer
                .write_all(b"\n")
                .with_context(|| format!("separating prior TUI event trace {}", path.display()))?;
            writer
                .flush()
                .with_context(|| format!("flushing TUI event trace {}", path.display()))?;
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(TraceState {
                writer,
                sequence,
                run_id,
                failure: None,
            })),
        })
    }

    pub(crate) fn emit(&self, event: &'static str, data: Value) -> Result<()> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| anyhow!("TUI event trace lock poisoned"))?;
        if let Some(failure) = &state.failure {
            return Err(anyhow!(failure.clone()));
        }
        let run_id = state.run_id.clone();
        let record = TraceRecord {
            schema_version: TUI_EVENT_TRACE_SCHEMA_VERSION,
            sequence: state.sequence,
            process_id: std::process::id(),
            run_id: run_id.as_deref(),
            timestamp_ms: unix_timestamp_ms(),
            event,
            data,
        };
        let result = (|| -> Result<()> {
            let mut line =
                serde_json::to_vec(&record).context("serializing TUI event trace record")?;
            line.push(b'\n');
            state
                .writer
                .write_all(&line)
                .context("writing TUI event trace record")?;
            state
                .writer
                .flush()
                .context("flushing TUI event trace record")?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                state.sequence = state.sequence.saturating_add(1);
                Ok(())
            }
            Err(error) => {
                let message = format!("TUI event trace write failed: {error:#}");
                state.failure = Some(message.clone());
                Err(anyhow!(message))
            }
        }
    }

    /// Return a prior asynchronous tap failure, if any.
    pub(crate) fn check(&self) -> Result<()> {
        let state = self
            .inner
            .lock()
            .map_err(|_| anyhow!("TUI event trace lock poisoned"))?;
        match &state.failure {
            Some(failure) => Err(anyhow!(failure.clone())),
            None => Ok(()),
        }
    }

    pub(crate) fn emit_ui_event(&self, event: &UiEvent) -> Result<()> {
        match event {
            UiEvent::ProviderRequest { audit } => self.emit("provider_request", audit.clone()),
            _ => self.emit("ui_event", ui_event_summary(event)),
        }
    }
}

fn smoke_run_id_from_env() -> Option<String> {
    std::env::var(SMOKE_RUN_MARKER_ENV)
        .ok()
        .filter(|marker| valid_smoke_run_marker(marker))
}

fn valid_smoke_run_marker(marker: &str) -> bool {
    if marker.len() > 128 {
        return false;
    }
    let mut parts = marker.split('-');
    let (Some(pid), Some(nanos), Some(sequence), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    pid.parse::<u32>().is_ok_and(|pid| pid > 0)
        && u128::from_str_radix(nanos, 16).is_ok_and(|nanos| nanos > 0)
        && u64::from_str_radix(sequence, 16).is_ok_and(|sequence| sequence > 0)
}

pub(crate) fn compose_remote_event_tap(
    base: Option<crate::RemoteEventTap>,
    trace: Option<TuiEventTrace>,
) -> Option<crate::RemoteEventTap> {
    match (base, trace) {
        (None, None) => None,
        (Some(base), None) => Some(base),
        (None, Some(trace)) => Some(Arc::new(move |event| {
            let _ = trace.emit_ui_event(event);
        })),
        (Some(base), Some(trace)) => Some(Arc::new(move |event| {
            base(event);
            let _ = trace.emit_ui_event(event);
        })),
    }
}

pub(crate) fn prompt_summary(prompt: &str, origin: PromptOrigin, queue_depth: usize) -> Value {
    json!({
        "origin": origin,
        // Correlate lifecycle events without writing the prompt body into the
        // semantic trace. The smoke harness uses the digest as a multiset key,
        // so repeated identical prompts remain distinguishable by count.
        "prompt_fingerprint": blake3::hash(prompt.as_bytes()).to_hex().to_string(),
        "prompt_chars": prompt.chars().count().min(1_000_000),
        "command": prompt.trim_start().starts_with('/'),
        "queue_depth": queue_depth,
    })
}

pub(crate) fn approval_kind(request: &hi_agent::ConfirmationRequest) -> &'static str {
    match request {
        hi_agent::ConfirmationRequest::FileEdit { .. } => "file_edit",
        hi_agent::ConfirmationRequest::ShellMutation { .. } => "shell_mutation",
        hi_agent::ConfirmationRequest::DelegateApply { .. } => "delegate_apply",
        hi_agent::ConfirmationRequest::AskUser { .. } => "ask_user",
        hi_agent::ConfirmationRequest::External { .. } => "external",
    }
}

pub(crate) fn plan_summary(agent: &hi_agent::Agent) -> Value {
    let mut pending = 0usize;
    let mut active = 0usize;
    let mut done = 0usize;
    for step in agent.current_plan() {
        match step.status {
            hi_agent::PlanStatus::Pending => pending += 1,
            hi_agent::PlanStatus::Active => active += 1,
            hi_agent::PlanStatus::Done => done += 1,
        }
    }
    json!({
        "total": pending + active + done,
        "pending": pending,
        "active": active,
        "done": done,
        "incomplete": agent.plan_incomplete(),
        "mode": agent.plan_mode(),
        "paused": agent.plan_drive_paused(),
        "approval_parked": agent.plan_approval_parked(),
        "stall": agent.plan_drive_stall(),
    })
}

pub(crate) fn drive_summary(action: hi_agent::DriveAction) -> Value {
    match action {
        hi_agent::DriveAction::Enqueue(kind) => json!({
            "action": "enqueue",
            "kind": match kind {
                hi_agent::DriveKind::Plan => "plan_drive",
                hi_agent::DriveKind::Goal => "goal_drive",
                hi_agent::DriveKind::User => "user",
            },
        }),
        hi_agent::DriveAction::Idle { reason } => json!({
            "action": "idle",
            "reason": match reason {
                hi_agent::DriveIdleReason::None => "none",
                hi_agent::DriveIdleReason::PlanMode => "plan_mode",
                hi_agent::DriveIdleReason::PlanApprovalParked => "plan_approval_parked",
                hi_agent::DriveIdleReason::GoalPaused => "goal_paused",
                hi_agent::DriveIdleReason::GoalParked => "goal_parked",
                hi_agent::DriveIdleReason::PlanPaused => "plan_paused",
                hi_agent::DriveIdleReason::PlanParked => "plan_parked",
                hi_agent::DriveIdleReason::Cancelled => "cancelled",
                hi_agent::DriveIdleReason::Blocked => "blocked",
                hi_agent::DriveIdleReason::NoProgress => "no_progress",
                hi_agent::DriveIdleReason::Infrastructure => "infrastructure",
            },
        }),
    }
}

pub(crate) fn outcome_summary(outcome: Option<&hi_agent::TurnOutcome>) -> Value {
    let Some(outcome) = outcome else {
        return Value::Null;
    };
    json!({
        "status": outcome.status,
        "verification": outcome.verification,
        "review": outcome.review,
        "stop_reason": outcome.stop_reason,
        "changed_file_count": outcome.changed_files.len(),
        "has_verified_revision": outcome.verified_workspace_revision.is_some(),
        "has_leftover": outcome.leftover.is_some(),
        "has_plan_leftover": outcome.plan_leftover.is_some(),
    })
}

/// Typed representation of the model-round limit that actually governed a
/// settled turn. Keep the unlimited sentinel out of smoke assertions: the
/// sentinel is an implementation detail, while `mode` is durable lifecycle
/// evidence that remains clear to humans reading a failure bundle.
pub(crate) fn step_limit_summary(effective_max_steps: u32) -> Value {
    if effective_max_steps == u32::MAX {
        json!({"mode": "unlimited"})
    } else {
        json!({
            "mode": "finite",
            "max_steps": effective_max_steps,
        })
    }
}

impl crate::App {
    pub(crate) fn check_tui_event_trace(&self) -> Result<()> {
        if let Some(trace) = &self.tui_event_trace {
            trace.check()?;
        }
        Ok(())
    }

    pub(crate) fn trace_ready(&self, width: u16, height: u16) -> Result<()> {
        if let Some(trace) = &self.tui_event_trace {
            trace.emit(
                "ready",
                json!({
                    "width": width,
                    "height": height,
                    "queue_depth": self.queue.len(),
                }),
            )?;
        }
        Ok(())
    }

    /// Record that the full-screen event loop consumed a terminal resize.
    /// Harnesses wait for this acknowledgement before sending subsequent
    /// keystrokes, avoiding an input-vs-SIGWINCH race at startup.
    pub(crate) fn trace_resized(&self, width: u16, height: u16) -> Result<()> {
        if let Some(trace) = &self.tui_event_trace {
            trace.emit(
                "resized",
                json!({
                    "width": width,
                    "height": height,
                    "queue_depth": self.queue.len(),
                }),
            )?;
        }
        Ok(())
    }

    pub(crate) fn trace_prompt_queued(&self, prompt: &str) {
        let Some(trace) = &self.tui_event_trace else {
            return;
        };
        let origin = PromptOrigin::from_prompt(prompt);
        let _ = trace.emit(
            "prompt_queued",
            prompt_summary(prompt, origin, self.queue.len()),
        );
    }

    pub(crate) fn trace_prompt_dequeued(&self, prompt: &str) -> Result<()> {
        if let Some(trace) = &self.tui_event_trace {
            let origin = PromptOrigin::from_prompt(prompt);
            trace.emit(
                "prompt_dequeued",
                prompt_summary(prompt, origin, self.queue.len()),
            )?;
        }
        Ok(())
    }

    pub(crate) fn trace_prompt_removed(&self, prompt: &str) {
        let Some(trace) = &self.tui_event_trace else {
            return;
        };
        let origin = PromptOrigin::from_prompt(prompt);
        let _ = trace.emit(
            "prompt_removed",
            prompt_summary(prompt, origin, self.queue.len()),
        );
    }

    /// An idle user submission crosses the queue boundary synchronously. Emit
    /// both records so harness assertions do not need a special prompt path.
    pub(crate) fn trace_immediate_prompt(&self, prompt: &str) -> Result<()> {
        if let Some(trace) = &self.tui_event_trace {
            let origin = PromptOrigin::from_prompt(prompt);
            trace.emit("prompt_queued", prompt_summary(prompt, origin, 1))?;
            trace.emit("prompt_dequeued", prompt_summary(prompt, origin, 0))?;
        }
        Ok(())
    }

    pub(crate) fn trace_turn_started(&self, agent: &hi_agent::Agent, prompt: &str) -> Result<()> {
        if let Some(trace) = &self.tui_event_trace {
            let mut summary =
                prompt_summary(prompt, PromptOrigin::from_prompt(prompt), self.queue.len());
            summary["plan"] = plan_summary(agent);
            trace.emit("turn_started", summary)?;
        }
        Ok(())
    }

    pub(crate) fn trace_turn_settled(
        &self,
        agent: &hi_agent::Agent,
        outcome: Option<&hi_agent::TurnOutcome>,
    ) -> Result<()> {
        let Some(trace) = &self.tui_event_trace else {
            return Ok(());
        };
        let next_drive = agent.drive_decision(outcome);
        let effective_max_steps = agent.last_turn_telemetry().effective_max_steps;
        trace.emit(
            "turn_settled",
            json!({
                "outcome": outcome_summary(outcome),
                "step_limit": step_limit_summary(effective_max_steps),
                "queue_depth": self.queue.len(),
                "next_drive": drive_summary(next_drive),
                "plan": plan_summary(agent),
            }),
        )?;
        let mut plan_pause_emitted = false;
        match next_drive {
            hi_agent::DriveAction::Idle {
                reason: hi_agent::DriveIdleReason::GoalParked,
            } => trace.emit("drive_parked", json!({"kind": "goal_drive"}))?,
            hi_agent::DriveAction::Idle {
                reason: hi_agent::DriveIdleReason::PlanParked,
            } => trace.emit("drive_parked", json!({"kind": "plan_drive"}))?,
            hi_agent::DriveAction::Idle {
                reason: hi_agent::DriveIdleReason::PlanApprovalParked,
            } => trace.emit(
                "drive_parked",
                json!({"kind": "plan_approval", "reason": "awaiting_approval"}),
            )?,
            hi_agent::DriveAction::Idle {
                reason: hi_agent::DriveIdleReason::GoalPaused,
            } => trace.emit("drive_paused", json!({"kind": "goal_drive"}))?,
            hi_agent::DriveAction::Idle {
                reason: hi_agent::DriveIdleReason::PlanPaused,
            } => {
                trace.emit("drive_paused", json!({"kind": "plan_drive"}))?;
                plan_pause_emitted = true;
            }
            _ => {}
        }
        // A cancelled turn explicitly pauses an unfinished plan before the
        // next-drive decision is computed.  That decision may report the
        // cancellation itself instead of `PlanPaused`; preserve the actual
        // persisted drive state in the semantic trace either way.
        if agent.plan_incomplete() && agent.plan_drive_paused() && !plan_pause_emitted {
            trace.emit(
                "drive_paused",
                json!({"kind": "plan_drive", "reason": "turn_settlement"}),
            )?;
        }
        Ok(())
    }

    pub(crate) fn trace_approval_shown(&self, kind: &str) -> Result<()> {
        if let Some(trace) = &self.tui_event_trace {
            trace.emit("approval_shown", json!({"kind": clip_label(kind)}))?;
        }
        Ok(())
    }

    pub(crate) fn trace_approval_decided(&self, kind: &str, decision: &str) {
        let Some(trace) = &self.tui_event_trace else {
            return;
        };
        let _ = trace.emit(
            "approval_decided",
            json!({
                "kind": clip_label(kind),
                "decision": clip_label(decision),
            }),
        );
    }

    pub(crate) fn trace_drive_state(&self, event: &'static str, kind: &str, reason: &str) {
        let Some(trace) = &self.tui_event_trace else {
            return;
        };
        let _ = trace.emit(
            event,
            json!({"kind": clip_label(kind), "reason": clip_label(reason)}),
        );
    }

    pub(crate) fn trace_session_ended(&self, agent: &hi_agent::Agent) -> Result<()> {
        if let Some(trace) = &self.tui_event_trace {
            trace.emit(
                "session_ended",
                json!({
                    "queue_depth": self.queue.len(),
                    "plan": plan_summary(agent),
                    "turn_count": agent.turn_count(),
                }),
            )?;
            trace.check()?;
        }
        Ok(())
    }
}

fn ui_event_summary(event: &UiEvent) -> Value {
    match event {
        UiEvent::ProviderRequest { .. } => json!({"kind": "provider_request"}),
        UiEvent::Text { text } => text_event("text", text),
        UiEvent::BtwAnswer { text } => text_event("btw_answer", text),
        UiEvent::BtwQuestion { question } => text_event("btw_question", question),
        UiEvent::BtwToolStarted { name, arguments } => {
            tool_event("btw_tool_started", name, arguments)
        }
        UiEvent::BtwToolResult { name, result } => tool_event("btw_tool_result", name, result),
        UiEvent::BtwEnd => json!({"kind": "btw_end"}),
        UiEvent::Reasoning { text } => text_event("reasoning", text),
        UiEvent::AssistantEnd => json!({"kind": "assistant_end"}),
        UiEvent::ToolStarted { name, arguments } => tool_event("tool_started", name, arguments),
        UiEvent::ToolCall { name, arguments } => tool_event("tool_call", name, arguments),
        UiEvent::ToolResult { name, result } => tool_event("tool_result", name, result),
        UiEvent::ToolStream { name, line } => tool_event("tool_stream", name, line),
        UiEvent::Status { text } => text_event("status", text),
        UiEvent::TopStatus { text } => text_event("top_status", text),
        UiEvent::CheckpointWarning { text } => text_event("checkpoint_warning", text),
        UiEvent::Plan { steps } => {
            let pending = steps
                .iter()
                .filter(|step| step.status == hi_agent::PlanStatus::Pending)
                .count();
            let active = steps
                .iter()
                .filter(|step| step.status == hi_agent::PlanStatus::Active)
                .count();
            let done = steps
                .iter()
                .filter(|step| step.status == hi_agent::PlanStatus::Done)
                .count();
            json!({"kind": "plan", "total": steps.len(), "pending": pending, "active": active, "done": done})
        }
        UiEvent::Usage {
            prompt,
            generated,
            ctx_used,
            ctx_window,
            estimated,
        } => json!({
            "kind": "usage", "prompt": prompt, "generated": generated,
            "ctx_used": ctx_used, "ctx_window": ctx_window, "estimated": estimated,
        }),
        UiEvent::SessionUsage { usage } => json!({
            "kind": "session_usage",
            "prompt": usage.input_tokens,
            "generated": usage.output_tokens,
        }),
        UiEvent::RateLimits { rate_limits } => {
            json!({"kind": "rate_limits", "present": rate_limits.is_some()})
        }
        UiEvent::TurnEnd { summary } => text_event("turn_end", summary),
        UiEvent::TurnError {
            error_kind,
            message,
            guidance,
        } => json!({
            "kind": "turn_error", "error_kind": clip_label(error_kind),
            "message_chars": message.chars().count(), "guidance_chars": guidance.chars().count(),
        }),
        UiEvent::ChangedFiles { files } => json!({"kind": "changed_files", "count": files.len()}),
        UiEvent::SuggestedPrompt { text } => text_event("suggested_prompt", text),
        UiEvent::SubagentSpawned {
            subagent_kind,
            background,
            ..
        } => json!({
            "kind": "subagent_spawned", "subagent_kind": clip_label(subagent_kind), "background": background,
        }),
        UiEvent::SubagentProgress { line, .. } => json!({
            "kind": "subagent_progress", "has_line": line.is_some(),
        }),
        UiEvent::SubagentFinished {
            status,
            elapsed_ms,
            summary,
            ..
        } => json!({
            "kind": "subagent_finished", "status": clip_label(status),
            "elapsed_ms": elapsed_ms, "summary_chars": summary.chars().count(),
        }),
        UiEvent::WorkflowUpdated { snapshot } => json!({
            "kind": "workflow_updated", "status": format!("{:?}", snapshot.status),
        }),
        UiEvent::DiffRunUpdated { snapshot } => json!({
            "kind": "diff_run_updated", "status": format!("{:?}", snapshot.status),
        }),
    }
}

/// Keep the PTY trace useful for live-provider diagnosis without turning it
/// into a second full request trace. Deserializing through the typed audit
/// contract drops unknown fields, and removing `request_body` is a
/// defense-in-depth boundary in case a future telemetry producer forgets to
/// strip it before the TUI sees the record.
pub(crate) fn provider_request_summary(audit: &Value) -> Value {
    let Ok(mut audit) = serde_json::from_value::<hi_ai::WireAudit>(audit.clone()) else {
        return json!({"audit_valid": false});
    };
    audit.request_body = None;
    let mut value = serde_json::to_value(audit).unwrap_or_else(|_| json!({"audit_valid": false}));
    if let Some(object) = value.as_object_mut() {
        object.remove("request_body");
    }
    value
}

fn text_event(kind: &'static str, text: &str) -> Value {
    json!({"kind": kind, "chars": text.chars().count().min(1_000_000)})
}

fn tool_event(kind: &'static str, name: &str, payload: &str) -> Value {
    json!({
        "kind": kind,
        "name": clip_label(name),
        "payload_chars": payload.chars().count().min(1_000_000),
    })
}

fn clip_label(text: &str) -> String {
    text.chars()
        .filter(|ch| !ch.is_control())
        .take(80)
        .collect()
}

fn prior_trace_state(path: &Path) -> Result<(u64, bool)> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok((0, false)),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("reading prior TUI event trace {}", path.display()));
        }
    };
    let mut largest = None::<u64>;
    for line in bytes.split(|byte| *byte == b'\n') {
        let Ok(value) = serde_json::from_slice::<Value>(line) else {
            // A process can be killed between writes. Preserve that evidence
            // and continue after the last complete record.
            continue;
        };
        if let Some(sequence) = value.get("sequence").and_then(Value::as_u64) {
            largest = Some(largest.map_or(sequence, |current| current.max(sequence)));
        }
    }
    let sequence = largest.map_or(0, |sequence| sequence.saturating_add(1));
    let needs_separator = bytes.last().is_some_and(|byte| *byte != b'\n');
    Ok((sequence, needs_separator))
}

fn unix_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    fn wire_audit_with_secret() -> hi_ai::WireAudit {
        hi_ai::WireAudit {
            provider: "openai_compatible".into(),
            route: "chat_completions".into(),
            model: "pipe/test".into(),
            output_token_parameter: "max_tokens".into(),
            max_output_tokens: 512,
            temperature: Some(0.2),
            top_p: None,
            reasoning_request: Some("high".into()),
            reasoning_replay: None,
            native_tools_enabled: true,
            tool_count: 7,
            strict_schema: true,
            tool_choice: Some("auto".into()),
            request_attempt: 2,
            compatibility_fallback: Some("stream_usage".into()),
            accepted: true,
            request_body: Some(json!({
                "messages": [{"role": "user", "content": "private prompt"}],
                "authorization": "Bearer private-key"
            })),
            response_status: Some(200),
        }
    }

    #[test]
    fn trace_is_versioned_sequenced_and_flushed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/events.jsonl");
        let trace = TuiEventTrace::open_with_run_id(&path, None).unwrap();
        trace.emit("ready", json!({"width": 80})).unwrap();
        trace
            .emit_ui_event(&UiEvent::Text {
                text: "secret body".into(),
            })
            .unwrap();

        let mut contents = String::new();
        File::open(path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        let rows = contents
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["schema_version"], 1);
        assert_eq!(rows[0]["sequence"], 0);
        assert!(rows[0].get("run_id").is_none());
        assert_eq!(rows[1]["sequence"], 1);
        assert_eq!(rows[1]["event"], "ui_event");
        assert_eq!(rows[1]["data"]["kind"], "text");
        assert_eq!(rows[1]["data"]["chars"], 11);
        assert!(!contents.contains("secret body"));
    }

    #[test]
    fn reopening_trace_appends_and_continues_sequence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        {
            let trace =
                TuiEventTrace::open_with_run_id(&path, Some("123-18fdb42-1".into())).unwrap();
            trace.emit("ready", json!({})).unwrap();
            trace.emit("session_ended", json!({})).unwrap();
        }
        {
            let trace =
                TuiEventTrace::open_with_run_id(&path, Some("123-18fdb42-2".into())).unwrap();
            trace.emit("ready", json!({})).unwrap();
        }
        let rows = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["sequence"], 0);
        assert_eq!(rows[1]["sequence"], 1);
        assert_eq!(rows[2]["sequence"], 2);
        assert!(rows.iter().all(|row| row["process_id"].is_u64()));
        assert_eq!(rows[0]["run_id"], "123-18fdb42-1");
        assert_eq!(rows[1]["run_id"], "123-18fdb42-1");
        assert_eq!(rows[2]["run_id"], "123-18fdb42-2");
    }

    #[test]
    fn smoke_run_marker_validation_accepts_only_harness_shape() {
        assert!(valid_smoke_run_marker("123-18fdb42-7"));
        for marker in [
            "",
            "0-18fdb42-7",
            "123-0-7",
            "123-18fdb42-0",
            "123-not-hex-7",
            "123-18fdb42",
            "123-18fdb42-7-extra",
        ] {
            assert!(!valid_smoke_run_marker(marker), "accepted {marker:?}");
        }
    }

    #[test]
    fn reopening_after_partial_record_separates_new_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        std::fs::write(
            &path,
            b"{\"schema_version\":1,\"sequence\":7,\"event\":\"truncated",
        )
        .unwrap();
        let trace = TuiEventTrace::open(&path).unwrap();
        trace.emit("ready", json!({})).unwrap();
        let contents = std::fs::read_to_string(path).unwrap();
        let last = contents.lines().last().unwrap();
        let row: Value = serde_json::from_str(last).unwrap();
        assert_eq!(row["event"], "ready");
        assert_eq!(row["sequence"], 0);
    }

    #[test]
    fn open_and_deferred_write_failures_are_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let open_error = TuiEventTrace::open(dir.path()).err().unwrap();
        assert!(format!("{open_error:#}").contains("TUI event trace"));

        let trace = TuiEventTrace::open(dir.path().join("events.jsonl")).unwrap();
        trace.inner.lock().unwrap().failure = Some("simulated disk failure".into());
        assert!(format!("{:#}", trace.check().unwrap_err()).contains("simulated disk failure"));
        assert!(
            format!("{:#}", trace.emit("ready", json!({})).unwrap_err())
                .contains("simulated disk failure")
        );
    }

    #[test]
    fn prompt_origins_cover_drive_user_and_command_paths() {
        assert_eq!(
            PromptOrigin::from_prompt(hi_agent::PLAN_DRIVE_PROMPT),
            PromptOrigin::PlanDrive
        );
        assert_eq!(
            PromptOrigin::from_prompt(hi_agent::GOAL_CONTINUE_PROMPT),
            PromptOrigin::GoalDrive
        );
        assert_eq!(
            PromptOrigin::from_prompt("/status"),
            PromptOrigin::CommandFollowUp
        );
        assert_eq!(PromptOrigin::from_prompt("fix it"), PromptOrigin::User);
    }

    #[test]
    fn prompt_summary_correlates_without_exposing_prompt_text() {
        let first = prompt_summary("private prompt", PromptOrigin::User, 73);
        let repeated = prompt_summary("private prompt", PromptOrigin::User, 72);
        let different = prompt_summary("different prompt", PromptOrigin::User, 71);

        assert_eq!(first["queue_depth"], 73);
        assert_eq!(first["prompt_fingerprint"], repeated["prompt_fingerprint"]);
        assert_ne!(first["prompt_fingerprint"], different["prompt_fingerprint"]);
        assert!(
            !serde_json::to_string(&first)
                .unwrap()
                .contains("private prompt")
        );
    }

    #[test]
    fn step_limit_summary_names_unlimited_without_exposing_the_sentinel() {
        assert_eq!(step_limit_summary(u32::MAX), json!({"mode": "unlimited"}));
        assert_eq!(
            step_limit_summary(2),
            json!({"mode": "finite", "max_steps": 2})
        );
    }

    #[test]
    fn composed_tap_preserves_existing_sink_and_redacts_ui_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        let trace = TuiEventTrace::open(&path).unwrap();
        let seen = Arc::new(Mutex::new(0usize));
        let sink_seen = seen.clone();
        let base: crate::RemoteEventTap = Arc::new(move |_| {
            *sink_seen.lock().unwrap() += 1;
        });
        let tap = compose_remote_event_tap(Some(base), Some(trace)).unwrap();
        tap(&UiEvent::Reasoning {
            text: "private reasoning".into(),
        });
        assert_eq!(*seen.lock().unwrap(), 1);
        let contents = std::fs::read_to_string(path).unwrap();
        assert!(contents.contains("reasoning"));
        assert!(!contents.contains("private reasoning"));
    }

    #[test]
    fn provider_request_trace_keeps_only_typed_scalar_audit_fields() {
        let audit = wire_audit_with_secret();
        let mut value = serde_json::to_value(audit).unwrap();
        value["future_nested_field"] = json!({"secret": "future-secret"});

        let summary = provider_request_summary(&value);
        assert_eq!(summary["provider"], "openai_compatible");
        assert_eq!(summary["model"], "pipe/test");
        assert_eq!(summary["request_attempt"], 2);
        assert_eq!(summary["response_status"], 200);
        assert!(summary.get("request_body").is_none());
        assert!(summary.get("future_nested_field").is_none());
        assert!(summary.as_object().unwrap().values().all(|value| matches!(
            value,
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
        )));
        let encoded = serde_json::to_string(&summary).unwrap();
        assert!(!encoded.contains("private prompt"));
        assert!(!encoded.contains("private-key"));
        assert!(!encoded.contains("future-secret"));
    }

    #[test]
    fn provider_request_is_flushed_before_turn_settlement_without_renderable_secrets() {
        use hi_agent::Ui as _;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        let trace = TuiEventTrace::open(&path).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (confirmations, _confirmation_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut ui = crate::event::ChannelUi {
            tx,
            confirmations,
            event_sink: None,
            approval_store: None,
        };

        ui.provider_request(&wire_audit_with_secret());
        let event = rx.try_recv().expect("provider audit event");
        let serialized = serde_json::to_string(&event).unwrap();
        assert!(serialized.contains("provider_request"));
        assert!(!serialized.contains("private prompt"));
        assert!(!serialized.contains("private-key"));

        // The full-screen event tap performs this call while the turn future
        // is still running. No turn-settlement callback is involved.
        trace.emit_ui_event(&event).unwrap();
        let contents = std::fs::read_to_string(path).unwrap();
        let row: Value = serde_json::from_str(contents.trim()).unwrap();
        assert_eq!(row["event"], "provider_request");
        assert_eq!(row["data"]["response_status"], 200);
        assert!(!contents.contains("private prompt"));
        assert!(!contents.contains("private-key"));
    }

    #[test]
    fn malformed_provider_audit_emits_safe_scalar_evidence() {
        assert_eq!(
            provider_request_summary(&json!({"request_body": {"secret": "value"}})),
            json!({"audit_valid": false})
        );
    }
}
