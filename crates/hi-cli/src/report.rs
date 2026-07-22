//! One-shot reports, exit codes, and RSI trace finish helpers.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use hi_agent::{Agent, Observation, ObservationSink, TurnOutcome, VerifyStage};
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

pub(crate) fn one_shot_exit_code(outcome: &TurnOutcome, allow_unverified: bool) -> i32 {
    outcome.exit_code(allow_unverified)
}

pub(crate) fn report_verification_stages(
    executions: &[hi_agent::VerificationExecution],
    _resolved: Vec<VerifyStage>,
) -> Vec<serde_json::Value> {
    executions
        .iter()
        .map(|execution| {
            serde_json::to_value(execution).expect("verification execution serializes")
        })
        .collect()
}

pub(crate) fn write_initialization_failure_report(
    path: &Path,
    model: &str,
    provider: &str,
    error: &anyhow::Error,
    rsi: Option<&TraceSummary>,
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
            "effective_max_steps": 0,
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

pub(crate) async fn run_one_shot_cancellable<F>(future: F) -> Option<Result<TurnOutcome>>
where
    F: std::future::Future<Output = Result<TurnOutcome>>,
{
    tokio::pin!(future);
    tokio::select! {
        result = &mut future => Some(result),
        _ = tokio::signal::ctrl_c() => None,
    }
}

pub(crate) fn start_rsi_trace(
    cli: &Cli,
    requested: RsiRequested,
    runtime: Option<&ManagedRuntimeDescriptor>,
) -> Result<Option<TraceWriter>> {
    let result = match requested {
        RsiRequested::Off => return Ok(None),
        RsiRequested::Managed => {
            let runtime = runtime.ok_or_else(|| anyhow!("managed RSI runtime is unavailable"))?;
            TraceWriter::create_bound(
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
            )
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
    for (kind, stage, payload) in [
        ("context_built", "intake", serde_json::to_value(prompt)?),
        (
            "repository_observation",
            "repository",
            serde_json::to_value(agent.last_file_changes())?,
        ),
        (
            "verification_completed",
            "verification",
            serde_json::to_value(agent.last_verification_executions())?,
        ),
        (
            "checkpoint_created",
            "checkpoint",
            serde_json::json!({"available": agent.last_turn_telemetry().checkpoint_available}),
        ),
    ] {
        observer.observe(Observation::json(kind, stage, 1, "turn-1", &payload)?)?;
    }
    observer.observe(Observation::json(
        "stage_exited",
        "verify",
        1,
        "turn-1",
        &serde_json::json!({"stage":"verify"}),
    )?)?;
    let terminal = Observation::json(
        "run_completed",
        "complete",
        1,
        "turn-1",
        &serde_json::json!({
            "outcome": outcome,
            "error": error.map(|error| format!("{error:#}")),
        }),
    )?;
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
pub(crate) fn write_report(
    path: &std::path::Path,
    agent: &Agent,
    user_prompt: Option<&str>,
    outcome: Option<&TurnOutcome>,
    error: Option<&anyhow::Error>,
    rsi: Option<&TraceSummary>,
) -> Result<()> {
    let totals = agent.totals();
    let turn = agent.last_turn_usage();
    let tel = agent.last_turn_telemetry();
    let outcome = outcome.cloned().unwrap_or_else(|| {
        let route = agent.last_effective_route();
        TurnOutcome::infrastructure_failure(
            route.model.clone(),
            route.provider.clone(),
            agent.last_changed_files().to_vec(),
        )
    });
    let goal = goal_report::report_goal(agent.structured_goal());
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
        "last_stall_reason": tel.last_stall_reason,
        "hit_step_cap": tel.hit_step_cap,
        "stalled_unfinished": tel.stalled_unfinished,
        "stalled_repeating": tel.stalled_repeating,
        "verify_attributions": tel.verify_attributions,
        "tool_calls": tel.tool_calls,
        "max_concurrent_batch": tel.max_concurrent_batch,
        "serial_runs": tel.serial_runs,
        "tool_timeline": tel.tool_timeline,
        "progress_events": tel.progress_events,
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
    });
    let planned_stages = agent
        .resolved_verification_stages()
        .into_iter()
        .map(|stage| serde_json::json!({ "name": stage.name, "command": stage.command }))
        .collect::<Vec<_>>();
    let stages = report_verification_stages(agent.last_verification_executions(), Vec::new());
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
                "user_prompt_estimated_tokens": agent.last_user_prompt_tokens(),
                "raw_user_prompt_estimated_tokens": user_prompt.map(hi_ai::estimate_text_tokens),
                "estimated": turn.estimated,
            },
        },
        "changes": exact_changes,
        "changes_complete": outcome_paths == exact_paths,
        "provider_error": error.map(|err| serde_json::json!({
            "kind": hi_ai::provider_error_kind(err).map(|kind| kind.as_str()),
            "message": err.to_string(),
        })),
        "compat_fallbacks": agent.last_compat_fallbacks(),
        "tool_mode": tool_mode_label(agent.tool_mode()),
        "goal": goal,
        "telemetry": telemetry,
        "rsi": rsi_report_block(rsi),
        "assistant_response": agent.messages().iter().rev()
            .find(|message| message.role == hi_ai::Role::Assistant)
            .map(|message| message.text()),
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
