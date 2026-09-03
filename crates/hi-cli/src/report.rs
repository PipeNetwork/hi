//! One-shot reports, exit codes, and RSI trace finish helpers.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use hi_agent::{Agent, Observation, ObservationSink, ReviewStatus, TurnOutcome, VerifyStage};
use hi_rsi_runtime::ManagedRuntimeDescriptor;
use hi_trace::{TraceIdentity, TraceMode, TraceSummary, TraceWriter};

use crate::commands::tool_mode_label;
use crate::config::{Cli, RsiRequested};
use crate::goal_report;
use crate::rsi_observation::TraceObservationSink;

pub(crate) fn pipeline_command(stages: &[VerifyStage]) -> Option<String> {
    if stages.is_empty() {
        return None;
    }
    Some(
        stages
            .iter()
            .map(|s| s.command.as_str())
            .collect::<Vec<_>>()
            .join(" && "),
    )
}

pub(crate) fn one_shot_exit_code(
    outcome: &TurnOutcome,
    allow_unverified: bool,
    leftover: bool,
) -> i32 {
    let code = outcome.exit_code(allow_unverified);
    if code != 0 {
        return code;
    }
    if leftover { 1 } else { 0 }
}

pub(crate) fn report_verification_stages(
    executions: &[hi_agent::VerificationExecution],
    review: ReviewStatus,
) -> Vec<serde_json::Value> {
    let mut stages: Vec<serde_json::Value> = executions
        .iter()
        .map(|execution| {
            let mut value =
                serde_json::to_value(execution).expect("verification execution serializes");
            if let Some(name) = value.get("name").and_then(|name| name.as_str()) {
                let mapped = match name {
                    "test" => "cargo_test",
                    "clippy" => "cargo_clippy",
                    other => other,
                };
                value["name"] = serde_json::Value::String(mapped.to_string());
            }
            value
        })
        .collect();
    if !matches!(review, ReviewStatus::NotRequired) {
        let status = match review {
            ReviewStatus::Passed => "passed",
            ReviewStatus::Objected | ReviewStatus::Escalated | ReviewStatus::Unavailable => {
                "failed"
            }
            ReviewStatus::NotRequired => unreachable!(),
        };
        stages.push(serde_json::json!({
            "name": "review",
            "status": status,
        }));
    }
    stages
}

pub(crate) fn write_initialization_failure_report(
    path: &Path,
    model: &str,
    provider: &str,
    error: &anyhow::Error,
    rsi: Option<&TraceSummary>,
    effective_max_steps: u32,
    effective_max_tool_calls: u32,
) -> Result<()> {
    let outcome =
        TurnOutcome::infrastructure_failure(model, Some(provider.to_string()), Vec::new());
    let report = serde_json::json!({
        "schema_version": 2,
        "outcome": outcome,
        "verification": {
            "mode": "unavailable",
            "status": outcome.verification,
            "planned_stages": [],
            "stages": [],
            "rounds": 0,
            "attributions": [],
        },
        "review": { "status": outcome.review },
        "tools": [],
        "route": outcome.effective_route,
        "usage": {
            "session": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 },
            "turn": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 },
        },
        "changes": [],
        "changes_complete": true,
        "provider_error": {
            "kind": "infrastructure",
            "message": error.to_string(),
        },
        "compat_fallbacks": [],
        "telemetry": {
            "effective_max_steps": effective_max_steps,
            "effective_max_tool_calls": effective_max_tool_calls,
            "tool_calls": 0,
        },
        "rsi": rsi_report_block(rsi),
        "assistant_response": serde_json::Value::Null,
    });
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating report directory {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("writing report {}", path.display()))
}

/// Drive a one-shot turn through Ctrl-C without dropping the live agent
/// future. The caller owns the cancellation token passed to
/// `Agent::run_turn_cancellable`; after signalling it we keep polling until
/// that method completes its bounded cooperative settlement and rollback. The
/// boolean records whether this driver observed Ctrl-C; callers use it to stop
/// synthetic goal/plan drive even when the turn crossed its commit boundary
/// just before the interrupt and therefore returns a normal outcome.
pub(crate) async fn run_one_shot_cancellable<F>(
    future: F,
    cancellation: hi_agent::TurnCancellation,
) -> (Result<TurnOutcome>, bool)
where
    F: std::future::Future<Output = Result<TurnOutcome>>,
{
    tokio::pin!(future);
    tokio::select! {
        result = &mut future => (result, false),
        _ = tokio::signal::ctrl_c() => {
            cancellation.cancel();
            (future.await, true)
        },
    }
}

pub(crate) fn start_rsi_trace(
    cli: &Cli,
    requested: RsiRequested,
    runtime: Option<&ManagedRuntimeDescriptor>,
) -> Result<Option<TraceWriter>> {
    // Local metadata traces are the default; `HI_TRACE_CAPTURE=off` is the
    // explicit escape hatch for installations that do not want local trace
    // files. Full capture remains opt-in (or is selected by evaluation).
    let capture_requested = !std::env::var("HI_TRACE_CAPTURE")
        .ok()
        .is_some_and(|value| value.eq_ignore_ascii_case("off"));
    let result = match requested {
        RsiRequested::Off if !capture_requested => return Ok(None),
        RsiRequested::Off => {
            let state_home = std::env::var_os("XDG_STATE_HOME")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME")
                        .map(std::path::PathBuf::from)
                        .map(|home| home.join(".local/state"))
                })
                .unwrap_or_else(|| std::path::PathBuf::from("."));
            // Self-hosted runs sign their trace with the local ed25519 key
            // (`local-signed:`), proving the trace is unmodified since signing
            // without claiming worker attestation. Managed runs leave
            // attestation to the external worker.
            TraceWriter::create_local(
                &state_home,
                cli.rsi_max_bytes.unwrap_or(hi_trace::DEFAULT_RUN_MAX_BYTES),
            )
            .map(|trace| {
                trace.with_attestor(std::sync::Arc::new(hi_trace::LocalAttestor::default()))
            })
        }
        RsiRequested::Managed => {
            let runtime = runtime.ok_or_else(|| anyhow!("managed RSI runtime is unavailable"))?;
            let trace = TraceWriter::create_bound(
                cli.rsi_trace_dir.as_ref().expect("clap requires trace dir"),
                TraceMode::Managed,
                cli.rsi_max_bytes.expect("clap requires trace size"),
                TraceIdentity {
                    run_id: runtime.identity.run_id.clone(),
                    task_id: runtime.identity.task_id.clone(),
                    candidate_id: runtime.identity.candidate_id.clone(),
                    manifest_hash: runtime.identity.manifest_hash.clone(),
                    agent_artifact_hash: runtime.identity.agent_artifact_hash.clone(),
                    repository_snapshot_hash: runtime.identity.repository_snapshot_hash.clone(),
                    runtime_descriptor_hash: runtime.content_hash()?,
                },
            );
            // Worker attestation: when the deployment exposes a signing socket
            // (HI_RSI_TRACE_SIGNING_SOCKET), attach a WorkerAttestor so the
            // terminal root_hash is signed with the worker's key — the anchor
            // that turns the local chain into worker-anchored evidence. Without
            // it the managed trace is recorded unattested (today's behavior).
            #[cfg(unix)]
            let trace = trace.map(
                |trace| match std::env::var_os("HI_RSI_TRACE_SIGNING_SOCKET") {
                    Some(socket) => trace.with_attestor(std::sync::Arc::new(
                        hi_trace::WorkerAttestor::from_socket(std::path::PathBuf::from(socket)),
                    )),
                    None => trace,
                },
            );
            trace
        }
        RsiRequested::Remote => return Ok(None),
    };
    match result {
        Ok(trace) => Ok(Some(trace)),
        Err(error) if requested == RsiRequested::Managed => Err(error),
        Err(error) => {
            eprintln!("\x1b[33mRSI trace warning: {error:#}; this turn will be unobserved\x1b[0m");
            Ok(None)
        }
    }
}

pub(crate) fn unix_time_ms() -> Result<u64> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

pub(crate) fn finish_initialization_trace(
    observer: Option<&std::sync::Arc<TraceObservationSink>>,
    error: &anyhow::Error,
) -> Result<Option<TraceSummary>> {
    let Some(observer) = observer else {
        return Ok(None);
    };
    let mut terminal = Observation::json(
        "run_completed",
        "initialization",
        1,
        "turn-1",
        &serde_json::json!({"status":"infrastructure_error", "error":format!("{error:#}")}),
    )?;
    terminal.metadata = serde_json::json!({"status":"infrastructure_error"});
    observer.finish(terminal)
}

pub(crate) fn finish_turn_trace(
    observer: Option<&std::sync::Arc<TraceObservationSink>>,
    agent: &Agent,
    prompt: &str,
    outcome: Option<&TurnOutcome>,
    error: Option<&anyhow::Error>,
) -> Result<Option<TraceSummary>> {
    let Some(observer) = observer else {
        return Ok(None);
    };
    let context_payload = if observer.full_capture() {
        serde_json::to_value(prompt)?
    } else {
        serde_json::json!({
            "byte_len": prompt.len(),
            "blake3": blake3::hash(prompt.as_bytes()).to_hex().to_string(),
        })
    };
    let verification_payload = if observer.full_capture() {
        serde_json::to_value(agent.last_verification_executions())?
    } else {
        serde_json::json!({
            "count": agent.last_turn_telemetry().diagnostic_retention.verification_executions_total,
            "retained_count": agent.last_verification_executions().len(),
            "dropped_count": agent.last_turn_telemetry().diagnostic_retention.verification_executions_dropped,
            "statuses": agent.last_verification_executions()
                .iter()
                .map(|execution| format!("{:?}", execution.status))
                .collect::<Vec<_>>(),
        })
    };
    let terminal_payload = if observer.full_capture() {
        serde_json::json!({
            "outcome": outcome,
            "error": error.map(|error| format!("{error:#}")),
        })
    } else {
        serde_json::json!({
            "outcome": outcome,
            "error_present": error.is_some(),
        })
    };
    for (kind, stage, payload) in [
        ("context_built", "intake", context_payload),
        (
            "repository_observation",
            "repository",
            serde_json::to_value(agent.last_file_changes())?,
        ),
        (
            "verification_completed",
            "verification",
            verification_payload,
        ),
        (
            "checkpoint_created",
            "checkpoint",
            serde_json::json!({"available": agent.last_turn_telemetry().checkpoint_available}),
        ),
    ] {
        observer.observe(Observation::json(kind, stage, 1, "turn-1", &payload)?)?;
    }
    let checkpoint_terminal_kind =
        if outcome.is_some_and(|turn| turn.stop_reason == hi_agent::TurnStopReason::Cancelled) {
            "checkpoint_rollback"
        } else if error.is_some() {
            "checkpoint_preserved"
        } else if outcome.is_some_and(|turn| turn.status == hi_agent::TurnStatus::Completed) {
            "checkpoint_sealed"
        } else {
            "checkpoint_updated"
        };
    observer.observe(Observation::json(
        checkpoint_terminal_kind,
        "checkpoint",
        1,
        "turn-1",
        &serde_json::json!({
            "available": agent.last_turn_telemetry().checkpoint_available,
            "error_present": error.is_some(),
        }),
    )?)?;
    observer.observe(Observation::json(
        "stage_exited",
        "verify",
        1,
        "turn-1",
        &serde_json::json!({"stage":"verify"}),
    )?)?;
    let terminal = Observation::json("run_completed", "complete", 1, "turn-1", &terminal_payload)?;
    observer.finish(terminal)
}

pub(crate) fn rsi_report_block(summary: Option<&TraceSummary>) -> serde_json::Value {
    summary.map_or_else(
        || {
            serde_json::json!({
                "mode": "off",
                "trace_schema": hi_trace::TRACE_SCHEMA_VERSION,
                "trace_id": null,
                "event_count": 0,
                "root_hash": null,
                "complete": false,
                "fully_observed": false,
                "candidate_evidence": true,
                // Keep the key present (null) so consumers can rely on the
                // field regardless of whether a trace was recorded.
                "attestation": null,
            })
        },
        |summary| serde_json::to_value(summary).expect("RSI summary serializes"),
    )
}

pub(crate) fn finish_interactive_trace(
    observer: Option<&std::sync::Arc<TraceObservationSink>>,
    agent: &Agent,
) -> Result<()> {
    let prompt = agent.last_user_message().unwrap_or_default();
    let summary = finish_turn_trace(observer, agent, &prompt, agent.last_turn_outcome(), None);
    summary?;
    Ok(())
}

/// Write a machine-readable run report (tokens, verify outcome) for the
/// eval harness and other automation.
#[allow(clippy::too_many_arguments)] // report writer carries each manifest input explicitly
pub(crate) fn write_report(
    path: &std::path::Path,
    agent: &Agent,
    user_prompt: Option<&str>,
    outcome: Option<&TurnOutcome>,
    error: Option<&anyhow::Error>,
    rsi: Option<&TraceSummary>,
    input_mode: &str,
    transcript_messages: Option<usize>,
    prompt_characters: Option<usize>,
    output_mode: &str,
) -> Result<()> {
    let totals = agent.totals();
    let turn = agent.last_turn_usage();
    let normalized_usage = agent.last_usage_telemetry();
    let sandbox = hi_tools::ProcessRunner::new(agent.workspace_root())
        .ok()
        .map(|runner| {
            serde_json::json!({
                "backend": runner.sandbox_backend_name(),
                "status": format!("{:?}", runner.sandbox_backend_status()).to_ascii_lowercase(),
                "enforced": runner.sandbox_enforced(),
            })
        })
        .unwrap_or_else(
            || serde_json::json!({"backend":"unknown","status":"unavailable","enforced":false}),
        );
    let tel = agent.last_turn_telemetry();
    let outcome = outcome.cloned().unwrap_or_else(|| {
        let route = agent.last_effective_route();
        TurnOutcome::infrastructure_failure(
            route.model.clone(),
            route.provider.clone(),
            agent.last_changed_files().to_vec(),
        )
    });
    let goal = goal_report::report_goal(agent.structured_goal(), agent.goal_drive_stall());
    let plan = report_plan(agent);
    let failure_mode = report_failure_mode(&outcome, error, tel);
    let partial_artifact = write_partial_artifact(path, agent, &outcome, error)?;
    let model_outcome = serde_json::json!({
        "model_requests": tel.model_requests,
        "accepted_completions": tel.accepted_completions,
        "tool_calls_before_stop": tel.tool_calls,
        "tool_call_channel": tel.tool_call_channel,
        "stop_reason": tel.last_stop_reason,
        "refusal_source": tel.refusal_source,
        "reasoning_requested": tel.reasoning_requested,
        "reasoning_received": tel.reasoning_received,
        "reasoning_replayed": tel.reasoning_replayed,
        "reasoning_signature_replayed": tel.reasoning_signature_replayed,
        "reasoning_fallback": tel.reasoning_fallback,
        "wire_audit": tel.wire_audit,
        "wire_audit_dropped": tel.diagnostic_retention.wire_audit_dropped,
    });
    let telemetry = serde_json::json!({
        "effective_max_steps": tel.effective_max_steps,
        "effective_max_tool_calls": agent.max_tool_calls_limit(),
        "verify_rounds": tel.verify_rounds,
        "recovery_retries": tel.recovery_retries,
        "repeat_nudges": tel.repeat_nudges,
        "continue_nudges": tel.continue_nudges,
        "truncation_retries": tel.truncation_retries,
        "no_progress_streak": tel.no_progress_streak,
        "forced_final_answer_attempts": tel.forced_final_answer_attempts,
        "last_progress_reason": tel.last_progress_reason,
        "last_no_progress_reason": tel.last_no_progress_reason,
        "hit_step_cap": tel.hit_step_cap,
        "hit_tool_cap": tel.hit_tool_cap,
        "verify_attributions": tel.verify_attributions,
        "tool_calls": tel.tool_calls,
        "max_concurrent_batch": tel.max_concurrent_batch,
        "serial_runs": tel.serial_runs,
        "tool_timeline": tel.tool_timeline,
        "progress_events": tel.progress_events,
        "plan_drive_stall": agent.plan_drive_stall(),
        "goal_drive_stall": agent.goal_drive_stall(),
        "file_reads": tel.file_reads,
        "targeted_searches": tel.targeted_searches,
        "listing_only": tel.listing_only,
        "first_tool_kind": tel.first_tool_kind,
        "discovery_depth": tel.discovery_depth,
        "quality_repair_nudges": tel.quality_repair_nudges,
        "review_repair_exhaustion_reason": tel.review_repair_exhaustion_reason,
        "review_repair_counts": tel.review_repair_counts,
        "review_repair_stopped_by_exhaustion": tel.review_repair_stopped_by_exhaustion,
        "skeptic_unavailable_count": tel.skeptic_unavailable_count,
        "skeptic_last_status": tel.skeptic_last_status,
        "checkpoint_available": tel.checkpoint_available,
        "advertised_tools": tel.advertised_tools,
        "tool_schema_tokens": tel.tool_schema_tokens,
        "stopped_by_step_cap": tel.hit_step_cap,
        "stopped_by_tool_cap": tel.hit_tool_cap,
        "prefix_stable_rounds": tel.prefix_stable_rounds,
        "prefix_break_rounds": tel.prefix_break_rounds,
        "tool_prefix_break_rounds": tel.tool_prefix_break_rounds,
        "earliest_prefix_break": tel.earliest_prefix_break,
        "model_requests": tel.model_requests,
        "accepted_completions": tel.accepted_completions,
        "last_stop_reason": tel.last_stop_reason,
        "tool_call_channel": tel.tool_call_channel,
        "reasoning_requested": tel.reasoning_requested,
        "reasoning_received": tel.reasoning_received,
        "reasoning_replayed": tel.reasoning_replayed,
        "reasoning_signature_replayed": tel.reasoning_signature_replayed,
        "reasoning_fallback": tel.reasoning_fallback,
        "refusal_source": tel.refusal_source,
        "requests": tel.requests,
        "compaction": tel.compaction,
        "diagnostic_retention": tel.diagnostic_retention,
        "ledger_events_dropped": agent.ledger_events_dropped(),
    });
    let planned_stages = agent
        .resolved_verification_stages()
        .into_iter()
        .map(|stage| serde_json::json!({ "name": stage.name, "command": stage.command }))
        .collect::<Vec<_>>();
    let stages = report_verification_stages(agent.last_verification_executions(), outcome.review);
    let tools = report_tool_records(&tel.tool_timeline);
    let exact_changes = agent
        .last_file_changes()
        .iter()
        .map(|change| serde_json::to_value(change).expect("file change serializes"))
        .collect::<Vec<_>>();
    let outcome_paths = outcome
        .changed_files
        .iter()
        .collect::<std::collections::BTreeSet<_>>();
    let exact_paths = agent
        .last_file_changes()
        .iter()
        .map(|change| &change.path)
        .collect::<std::collections::BTreeSet<_>>();
    let report = serde_json::json!({
        "schema_version": 2,
        "outcome": outcome,
        "failure_mode": failure_mode,
        "model_outcome": model_outcome,
        "partial_artifact": partial_artifact,
        "verification": {
            "mode": agent.verification_mode(),
            "status": outcome.verification,
            "planned_stages": planned_stages,
            "stages": stages,
            "rounds": tel.verify_rounds,
            "attributions": tel.verify_attributions,
        },
        "review": {
            "status": outcome.review,
        },
        "tools": tools,
        "route": outcome.effective_route,
        "sandbox": sandbox,
        "usage": {
            "session": {
                "input_tokens": totals.input_tokens,
                "output_tokens": totals.output_tokens,
                "total_tokens": totals.total(),
                "cache_read_tokens": totals.cache_read_tokens,
                "cache_creation_tokens": totals.cache_creation_tokens,
                "estimated": totals.estimated,
            },
            "turn": {
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
                "total_tokens": turn.total(),
                "cache_read_tokens": turn.cache_read_tokens,
                "cache_creation_tokens": turn.cache_creation_tokens,
                "cache_read_ratio": if turn.input_tokens > 0 {
                    Some(
                        (turn.cache_read_tokens as f64 / turn.input_tokens as f64 * 1000.0).round()
                            / 1000.0,
                    )
                } else {
                    None
                },
                "user_prompt_estimated_tokens": agent.last_user_prompt_tokens(),
                "raw_user_prompt_estimated_tokens": user_prompt.map(hi_ai::estimate_text_tokens),
                "estimated": turn.estimated,
            },
            "normalized": normalized_usage,
        },
        "changes": exact_changes,
        "changes_complete": outcome_paths == exact_paths,
        "provider_error": error.map(|err| serde_json::json!({
            "kind": hi_ai::provider_error_kind(err).map(|kind| kind.as_str()),
            "message": err.to_string(),
            "code": err.downcast_ref::<hi_ai::ProviderError>().and_then(|error| error.code.clone()),
            "http_status": err.downcast_ref::<hi_ai::ProviderError>().and_then(|error| error.http_status),
        })),
        "compat_fallbacks": agent.last_compat_fallbacks(),
        "tool_mode": tool_mode_label(agent.tool_mode()),
        "goal": goal,
        "plan": plan,
        "telemetry": telemetry,
        "rsi": rsi_report_block(rsi),
        "assistant_response": agent.messages().iter().rev()
            .find(|message| message.role == hi_ai::Role::Assistant)
            .map(|message| message.text()),
        "evaluation": {
            "input_mode": input_mode,
            "output_mode": output_mode,
            "transcript_messages": transcript_messages,
            "prompt_characters": prompt_characters,
        },
    });
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating report directory {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("writing report {}", path.display()))?;
    Ok(())
}

fn report_plan(agent: &hi_agent::Agent) -> Option<serde_json::Value> {
    let steps = agent.current_plan();
    if steps.is_empty() {
        return None;
    }
    let done = steps
        .iter()
        .filter(|step| step.status == hi_agent::PlanStatus::Done)
        .count();
    Some(serde_json::json!({
        "done": done,
        "total": steps.len(),
        "next": agent.next_plan_step_title(),
        "pending": agent.plan_incomplete(),
        "drive": agent.plan_drive_status(),
    }))
}

fn report_failure_mode(
    outcome: &TurnOutcome,
    error: Option<&anyhow::Error>,
    telemetry: &hi_agent::TurnTelemetry,
) -> &'static str {
    if let Some(error) = error {
        return match hi_ai::provider_error_kind(error) {
            Some(hi_ai::ProviderErrorKind::PolicyBlocked) => "api_policy_blocked",
            Some(hi_ai::ProviderErrorKind::ToolProtocol) => "tool_protocol_error",
            Some(hi_ai::ProviderErrorKind::EmptyCompletion) => "empty_completion",
            Some(hi_ai::ProviderErrorKind::RequestTooLarge)
                if telemetry.last_stop_reason.as_deref() == Some("length") =>
            {
                "output_truncated"
            }
            Some(_) => "provider_transport_error",
            None => "infrastructure_error",
        };
    }
    if telemetry.refusal_source.is_some() {
        return if telemetry.tool_calls > 0 {
            "model_refusal_after_tools"
        } else {
            "model_refusal_before_tools"
        };
    }
    if matches!(outcome.stop_reason, hi_agent::TurnStopReason::Cancelled) {
        return "user_cancelled";
    }
    if matches!(
        outcome.verification,
        hi_agent::VerificationStatus::Failed | hi_agent::VerificationStatus::InfrastructureError
    ) {
        return "verification_failed";
    }
    if matches!(
        telemetry.last_stop_reason.as_deref(),
        Some("length" | "max_tokens")
    ) {
        return "output_truncated";
    }
    if outcome.changed_files.is_empty()
        && !matches!(outcome.status, hi_agent::TurnStatus::Completed)
    {
        return "no_edits";
    }
    "completed"
}

fn write_partial_artifact(
    report_path: &Path,
    agent: &Agent,
    outcome: &TurnOutcome,
    error: Option<&anyhow::Error>,
) -> Result<serde_json::Value> {
    let evidence_dir = Path::new(&format!("{}.evidence", report_path.display())).to_path_buf();
    std::fs::create_dir_all(&evidence_dir)
        .with_context(|| format!("creating evidence directory {}", evidence_dir.display()))?;
    let changes = agent.last_file_changes();
    let changes_json = serde_json::to_vec_pretty(changes)?;
    let changes_hash = blake3::hash(&changes_json).to_hex().to_string();
    let changes_path = evidence_dir.join(format!("{changes_hash}.changes.json"));
    if !changes_path.exists() {
        let temp = evidence_dir.join(format!(".{changes_hash}.tmp"));
        std::fs::write(&temp, &changes_json)?;
        std::fs::rename(&temp, &changes_path)?;
    }
    let before_material = changes
        .iter()
        .map(|change| {
            format!(
                "{}:{}",
                change.path,
                change.before_digest.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let after_material = changes
        .iter()
        .map(|change| {
            format!(
                "{}:{}",
                change.path,
                change.after_digest.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let last_completion_ref = agent
        .messages()
        .iter()
        .rev()
        .find(|message| message.role == hi_ai::Role::Assistant)
        .map(|message| blake3::hash(message.text().as_bytes()).to_hex().to_string());
    let checkpoint_ref = agent.checkpoint_refs().last().cloned();
    let cancelled = matches!(outcome.stop_reason, hi_agent::TurnStopReason::Cancelled);
    let status = if cancelled {
        "rolled_back"
    } else if error.is_some() || !matches!(outcome.status, hi_agent::TurnStatus::Completed) {
        "partial"
    } else {
        "complete"
    };
    let manifest = serde_json::json!({
        "schema_version": 1,
        "status": status,
        "workspace_before_digest": blake3::hash(before_material.as_bytes()).to_hex().to_string(),
        "workspace_after_digest": blake3::hash(after_material.as_bytes()).to_hex().to_string(),
        "changed_files": changes.iter().map(|change| change.path.clone()).collect::<Vec<_>>(),
        "patch_ref": changes_path.display().to_string(),
        "checkpoint_ref": checkpoint_ref.clone(),
        "last_completion_ref": last_completion_ref.clone(),
        "preserved_changes": error.is_some() && !cancelled && !changes.is_empty(),
        "resume_available": agent.last_turn_telemetry().checkpoint_available == Some(true),
        "rollback_reason": cancelled.then_some("user_cancelled"),
        "provider_error": error.map(|err| err.to_string()),
    });
    let manifest_path = evidence_dir.join("manifest.json");
    let temp = evidence_dir.join(".manifest.json.tmp");
    std::fs::write(&temp, serde_json::to_vec_pretty(&manifest)?)?;
    std::fs::rename(&temp, &manifest_path)?;
    Ok(serde_json::json!({
        "status": status,
        "directory": evidence_dir,
        "manifest": manifest_path,
        "patch_ref": changes_path,
        "checkpoint_ref": checkpoint_ref,
        "last_completion_ref": last_completion_ref,
        "changed_files": changes.iter().map(|change| change.path.clone()).collect::<Vec<_>>(),
        "resume_available": agent.last_turn_telemetry().checkpoint_available == Some(true),
    }))
}

/// Additive schema-v2 goal detail used by long-horizon drivers to distinguish
/// a genuinely unchanged turn from progress that does not advance `done` yet
/// (for example, recording a retry or moving the active plan cursor).
pub(crate) fn report_tool_records(entries: &[hi_agent::ToolCallEntry]) -> Vec<serde_json::Value> {
    entries
        .iter()
        .map(|entry| {
            serde_json::json!({
                "name": entry.tool,
                "path": entry.path,
                "duration_ms": entry.duration_ms,
                "status": entry.status,
                "process": entry.process,
                "background": entry.background,
                "effects": entry.effects,
                "truncation": entry.truncation,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A finished trace's attestation label must surface in the report block.
    /// Uses a legacy `local-unattested` fixture — the display path renders any
    /// label verbatim, so this proves the field is not dropped regardless of
    /// scheme (current local runs emit `local-signed:`).
    #[test]
    fn rsi_report_block_surfaces_attestation_label() {
        let summary = TraceSummary {
            mode: TraceMode::Local,
            trace_schema: hi_trace::TRACE_SCHEMA_VERSION,
            trace_id: "a".repeat(32),
            event_count: 1,
            root_hash: "b".repeat(64),
            complete: true,
            fully_observed: true,
            candidate_evidence: true,
            artifact_path: None,
            identity: None,
            attestation: Some(format!("local-unattested:{}", "b".repeat(64))),
        };
        let block = rsi_report_block(Some(&summary));
        assert_eq!(
            block["attestation"].as_str().unwrap(),
            format!("local-unattested:{}", "b".repeat(64)),
            "report block dropped the attestation label: {block}"
        );
    }

    /// The no-trace ("off") branch keeps the `attestation` key present (null)
    /// so the report schema is stable whether or not a trace was recorded.
    #[test]
    fn rsi_report_block_off_keeps_attestation_key() {
        let block = rsi_report_block(None);
        assert!(
            block.get("attestation").is_some(),
            "off branch must keep the attestation key: {block}"
        );
        assert!(block["attestation"].is_null());
    }
}
