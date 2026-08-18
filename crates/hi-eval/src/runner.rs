use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::Semaphore;

use crate::artifacts::{
    CapturedOracle, DirectorySnapshot, candidate_changed_paths, candidate_runtime_artifact_paths,
    command_output_with_timeout, copy_dir, directory_snapshot, forbidden_changes, make_workdir,
    output_summary, run_final_oracle_without_artifacts, verify_output_in,
};
use crate::config::{EvalProfile, Task, WorkspaceKind};
use crate::results::{
    AgentProcessOutcome, Candidate, CandidateCheck, FailKind, RunResult, Trajectory,
    TrajectoryAttribution, TrajectoryToolCall, TurnMetric, classify, looks_like_build_error,
};

/// A no-follow content snapshot of `dir`. Integrity failures never become an
/// empty snapshot, even in the historical test helper.
#[cfg(test)]
pub fn dir_snapshot(dir: &Path) -> Result<DirectorySnapshot> {
    directory_snapshot(dir)
}

/// Run all of a config's candidates; the config solves the task if any passes.
/// Tokens are summed. Candidates run in parallel since each gets its own
/// isolated workdir — wall-clock is the max, not the sum.
#[allow(clippy::too_many_arguments)]
pub async fn run_config(
    hi: &Path,
    task_dir: &Path,
    task: &Task,
    config_name: &str,
    use_verify: bool,
    temperatures: &[f32],
    env: &'static [(&'static str, &'static str)],
    profile: EvalProfile,
    model_override: Option<String>,
    candidate_semaphore: Arc<Semaphore>,
) -> Result<RunResult> {
    let model_label = model_override
        .clone()
        .or_else(|| std::env::var("HI_MODEL").ok())
        .unwrap_or_else(|| "(unset)".to_string());
    let mut result = RunResult {
        config: config_name.to_string(),
        model: model_label,
        task: String::new(),
        trial: 0,
        passed: false,
        fail: None,
        provider_error_kind: None,
        compat_fallbacks_used: Vec::new(),
        changed_files: Vec::new(),
        verify_output_summary: String::new(),
        failure_confidence: None,
        candidates: Vec::new(),
        tokens: 0,
        input_tokens: 0,
        seconds: 0.0,
        mcp_model: None,
        trajectory: Trajectory::default(),
        growth: Vec::new(),
    };

    // Run candidates in parallel — each gets its own temp workdir.
    // `run_config` is already executing on the eval runtime (called from
    // `async_main` via `tokio::spawn`), so we must not create a second Tokio
    // runtime here and `block_on` from within it — nesting runtimes panics.
    // Just await the candidate futures directly on this runtime.
    let candidates: Vec<Candidate> = async {
        let mut futs = Vec::new();
        for (index, &temperature) in temperatures.iter().enumerate() {
            let hi = hi.to_path_buf();
            let task_dir = task_dir.to_path_buf();
            let task = task.clone();
            let model_override = model_override.clone();
            let semaphore = candidate_semaphore.clone();
            futs.push(tokio::spawn(async move {
                let permit = semaphore
                    .acquire_owned()
                    .await
                    .context("global candidate semaphore closed")?;
                tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    let started = Instant::now();
                    match run_candidate(
                        index,
                        &hi,
                        &task_dir,
                        &task,
                        use_verify,
                        temperature,
                        env,
                        profile,
                        model_override.as_deref(),
                    ) {
                        Ok(candidate) => candidate,
                        Err(error) => infrastructure_candidate(
                            index,
                            temperature,
                            model_override.as_deref(),
                            started.elapsed().as_secs_f64(),
                            &error,
                        ),
                    }
                })
                .await
                .context("joining blocking candidate")
            }));
        }
        let mut out = Vec::with_capacity(futs.len());
        for fut in futs {
            out.push(fut.await.context("joining candidate task")??);
        }
        Ok::<_, anyhow::Error>(out)
    }
    .await?;

    let mut fails: Vec<FailKind> = Vec::new();
    let mut summaries = Vec::new();
    // Track the representative candidate (furthest-progressing) so its
    // trajectory is surfaced, mirroring how `result.fail` is chosen.
    let mut best_rank: i32 = -1;
    let mut representative_trajectory: Option<Trajectory> = None;
    let mut representative_growth: Vec<TurnMetric> = Vec::new();
    for candidate in &candidates {
        let cand_rank = candidate
            .fail
            .map(|k| k.rank() as i32)
            .unwrap_or(if candidate.passed { 4 } else { -1 });
        if cand_rank > best_rank {
            best_rank = cand_rank;
            representative_trajectory = Some(candidate.trajectory.clone());
            representative_growth = candidate.growth.clone();
            result.failure_confidence = candidate.failure_confidence;
        }
        result.passed |= candidate.passed;
        if let Some(k) = candidate.fail {
            fails.push(k);
        }
        if result.provider_error_kind.is_none() {
            result.provider_error_kind = candidate.provider_error_kind.clone();
        }
        result
            .compat_fallbacks_used
            .extend(candidate.compat_fallbacks_used.clone());
        result.changed_files.extend(candidate.changed_files.clone());
        if !candidate.verify_output_summary.is_empty() {
            summaries.push(candidate.verify_output_summary.clone());
        }
        result.tokens += candidate.tokens;
        result.input_tokens += candidate.input_tokens;
        result.seconds = result.seconds.max(candidate.seconds);
    }
    // When the config failed (no candidate passed), its representative failure is
    // the one that got furthest (e.g. logic over no-edits).
    if !result.passed {
        result.fail = fails.into_iter().max_by_key(|k| k.rank());
    }
    result.trajectory = representative_trajectory.unwrap_or_default();
    result.growth = representative_growth;
    result.changed_files.sort();
    result.changed_files.dedup();
    result.compat_fallbacks_used.sort();
    result.compat_fallbacks_used.dedup();
    result.verify_output_summary = summaries.join("\n--- candidate ---\n");
    result.candidates = candidates;
    Ok(result)
}

/// The report's `goal` block as a canonical JSON string (drive/stall signal).
fn read_report_goal(report: &Path) -> Option<String> {
    let text = std::fs::read_to_string(report).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let goal = value.get("goal").filter(|g| !g.is_null())?;
    Some(goal.to_string())
}

/// A single turn's context-growth snapshot from the just-written report: context
/// (input) tokens sent this turn, total booked, tool calls, and goal progress.
/// Best-effort — a missing/short report yields zeroed fields, never an error.
fn read_turn_metric(report: &Path, turn: u32) -> TurnMetric {
    let mut m = TurnMetric {
        turn,
        ..Default::default()
    };
    let Ok(text) = std::fs::read_to_string(report) else {
        return m;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return m;
    };
    m.input_tokens = v
        .pointer("/usage/turn/input_tokens")
        .and_then(|value| value.as_u64())
        .or_else(|| {
            v.pointer("/turn_usage/input_tokens")
                .and_then(|value| value.as_u64())
        })
        .or_else(|| v["turn_input_tokens"].as_u64())
        .or_else(|| v["input_tokens"].as_u64())
        .unwrap_or(0);
    m.total_tokens = v
        .pointer("/usage/turn/total_tokens")
        .and_then(|value| value.as_u64())
        .or_else(|| {
            v.pointer("/turn_usage/total_tokens")
                .and_then(|value| value.as_u64())
        })
        .or_else(|| v["turn_total_tokens"].as_u64())
        .or_else(|| v["total_tokens"].as_u64())
        .unwrap_or(0);
    m.tool_calls = v["telemetry"]["tool_calls"].as_u64().unwrap_or(0) as u32;
    // The report's `goal` block is a summary: {objective, status, done, total}.
    if let Some(goal) = v.get("goal").filter(|g| !g.is_null()) {
        m.goal_done = goal["done"].as_u64().unwrap_or(0) as u32;
        m.goal_total = goal["total"].as_u64().unwrap_or(0) as u32;
    }
    m
}

/// One independent attempt in an isolated copy of the fixture.
#[allow(clippy::too_many_arguments)]
pub fn run_candidate(
    index: usize,
    hi: &Path,
    task_dir: &Path,
    task: &Task,
    use_verify: bool,
    temperature: f32,
    env: &[(&str, &str)],
    profile: EvalProfile,
    model_override: Option<&str>,
) -> Result<Candidate> {
    task.validate()?;
    let final_message_mode = std::env::var("HI_EVAL_OUTPUT")
        .ok()
        .is_some_and(|value| value.eq_ignore_ascii_case("final_message"));
    // Capture before creating or launching anything candidate-controlled.
    let oracle = CapturedOracle::capture(task_dir, &task.final_oracle)?;
    let work = make_workdir()?;
    let fixture = task_dir.join("fixture");
    copy_dir(&fixture, &work)?;
    initialize_workspace(&work, task)?;
    let before = directory_snapshot(&work)?;
    let report_path = work.join(".hi-eval-report.json");

    // Long-task drive (HI_EVAL_TURNS > 1): goal mode runs N session-continuing
    // turns of TURN_STEPS each — the fleet/goal cadence — while plain mode gets
    // the SAME total step budget in one turn, so the A/B compares structure,
    // not budget. HI_EVAL_TURNS=1 (default) is today's single-run behavior.
    let goal_mode = std::env::var_os("HI_EVAL_GOAL").is_some();
    let turns: u32 = std::env::var("HI_EVAL_TURNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1);
    let turn_steps: u32 = std::env::var("HI_EVAL_TURN_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(25);
    let session = work.join(".hi-eval-session.jsonl");
    let harness_run = harness_run_config(task_dir);
    let resume_steps = harness_resume_steps(&harness_run);
    let persist_session = turns > 1 || !resume_steps.is_empty() || harness_run.needs_session();
    if harness_run.needs_seed() {
        seed_harness_session(
            &session,
            harness_run.seed_image_chars,
            harness_run.seed_tool_result_chars,
        )?;
    }
    const GOAL_CONTINUE_PROMPT: &str = "Continue the long-horizon goal: complete the active \
sub-goal now, then update the plan with update_plan — including any newly discovered steps.";

    let build_cmd = |prompt: &str, first: bool, max_steps: Option<u32>| {
        let mut cmd = Command::new(hi);
        cmd.current_dir(&work)
            .arg("--report")
            .arg(&report_path)
            .arg("--trace-full")
            .arg("--temperature")
            .arg(temperature.to_string());
        if persist_session {
            // Multi-turn / harness resume: persist the conversation across child runs.
            cmd.arg("--session-file").arg(&session);
        } else {
            cmd.arg("--no-save");
        }
        if let Some(steps) = max_steps {
            cmd.arg("--max-steps").arg(steps.to_string());
        }
        for arg in profile.hi_args() {
            cmd.arg(arg);
        }
        if let Some(model) = model_override {
            cmd.env("HI_MODEL", model);
        }
        if final_message_mode {
            cmd.env("HI_EVAL_OUTPUT", "final_message");
        }
        // Per-config env — the orchestration lever under test (e.g. the skeptic
        // gate). Same for the first turn and every drive turn.
        for (key, value) in env {
            cmd.env(key, value);
        }
        if use_verify {
            if let Some(feedback) = &task.visible_feedback {
                cmd.arg("--verify").arg(&feedback.command);
            }
        } else {
            cmd.arg("--no-verify").arg("--allow-unverified");
        }
        if goal_mode && first {
            cmd.arg("--goal").arg(&task.prompt);
        }
        cmd.arg(prompt);
        cmd
    };

    let started = Instant::now();
    let timeout = Duration::from_secs(task.timeouts.candidate_seconds);
    let first_max_steps = resume_steps.first().copied().or_else(|| {
        (turns > 1).then(|| {
            if goal_mode {
                turn_steps
            } else {
                turn_steps * turns
            }
        })
    });
    let mut first_command = build_cmd(&task.prompt, true, first_max_steps);
    let mut output =
        command_output_with_timeout(&mut first_command, timeout).context("failed to launch hi")?;
    let mut merged_tape = read_report_value(&report_path);
    // Per-turn context-growth series (multi-turn drive only). Turn 1 first, then
    // one snapshot per session-continuing drive turn — the growth curve.
    let mut growth: Vec<TurnMetric> = Vec::new();
    if turns > 1 {
        growth.push(read_turn_metric(&report_path, 1));
    }
    if goal_mode && turns > 1 {
        // Drive while the report says the goal is still active (stall guard: two
        // consecutive turns with an unchanged goal park the drive).
        let mut last_goal = read_report_goal(&report_path);
        let mut stall = 0u32;
        for turn in 1..turns {
            let active = last_goal.as_ref().is_some_and(|g| {
                g.contains("\"status\":\"Active\"") || g.contains("\"status\": \"Active\"")
            });
            if !active || stall >= 2 || output.timed_out {
                break;
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            let mut command = build_cmd(GOAL_CONTINUE_PROMPT, false, Some(turn_steps));
            output = command_output_with_timeout(&mut command, remaining)
                .context("failed to launch hi (drive turn)")?;
            growth.push(read_turn_metric(&report_path, turn + 1));
            let goal = read_report_goal(&report_path);
            if goal == last_goal {
                stall += 1;
            } else {
                stall = 0;
            }
            last_goal = goal;
        }
    }
    // Harness resume: additional invocations on the same --session-file.
    // Default single-shot tasks never take this path.
    if resume_steps.len() > 1 && !output.timed_out {
        for (offset, &steps) in resume_steps.iter().enumerate().skip(1) {
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                break;
            }
            let mut command = build_cmd(&task.prompt, false, Some(steps));
            output = command_output_with_timeout(&mut command, remaining)
                .context("failed to launch hi (harness resume)")?;
            if persist_session {
                growth.push(read_turn_metric(&report_path, (offset + 1) as u32));
            }
            if let Some(next) = read_report_value(&report_path) {
                merged_tape = Some(match merged_tape.take() {
                    Some(prev) => merge_harness_tapes(prev, next),
                    None => next,
                });
            }
        }
    }
    if let Some(merged) = &merged_tape
        && persist_session
    {
        let _ = std::fs::write(
            &report_path,
            serde_json::to_vec_pretty(merged).unwrap_or_default(),
        );
    }
    if !output.success() {
        eprintln!(
            "    (hi {}{}): {}",
            output.status,
            if output.timed_out { ", timed out" } else { "" },
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    // Freeze the candidate revision before running any feedback or scorer.
    let after = directory_snapshot(&work)?;
    let mut changed_files = candidate_changed_paths(&before, &after, &task.allowed_changes)?;
    if !harness_run.ignore_change_prefixes.is_empty() {
        changed_files.retain(|path| {
            !harness_run
                .ignore_change_prefixes
                .iter()
                .any(|prefix| hi_eval::judge::path_under_prefix(path, prefix))
        });
    }
    let runtime_artifacts =
        candidate_runtime_artifact_paths(&before, &after, &task.allowed_changes)?;
    let edited = !changed_files.is_empty();
    let report = read_report(&report_path);
    let mut final_message_available = !final_message_mode;
    if final_message_mode
        && let Ok(value) = std::fs::read_to_string(&report_path)
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .map(|value| {
                value["assistant_response"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string()
            })
            .ok_or(())
    {
        final_message_available = !value.trim().is_empty();
        std::fs::write(work.join(".hi-eval-final-message"), value)?;
    }
    let forbidden = forbidden_changes(&changed_files, &task.allowed_changes)?;
    let (patch, patch_truncated) = render_patch(&before, &after, &changed_files);

    // Inner-task trees (e.g. bug/) may have been mutated by the candidate's
    // driver via host.py. Restore the fixture copy so the oracle scores the
    // driver, not a leftover inner patch.
    if !harness_run.ignore_change_prefixes.is_empty() {
        restore_ignored_inner_paths(&fixture, &work, &harness_run.ignore_change_prefixes)?;
    }
    // The score comes only from the captured oracle in a new copy. A visible
    // feedback check, when configured, is recorded separately and cannot make
    // a candidate pass.
    let oracle_output = run_final_oracle_without_artifacts(
        &work,
        &oracle,
        task.timeouts.oracle_seconds,
        &runtime_artifacts,
    )?;
    let mut verify_output = String::from_utf8_lossy(&oracle_output.stdout).into_owned();
    verify_output.push_str(&String::from_utf8_lossy(&oracle_output.stderr));
    let mut checks = Vec::new();
    checks.push(CandidateCheck {
        name: "allowed_changes".to_string(),
        passed: forbidden.is_empty() && (edited || final_message_available),
        timed_out: false,
        output_truncated: false,
        duration_seconds: 0.0,
        output_summary: if !edited {
            "candidate made no task-file changes".to_string()
        } else if forbidden.is_empty() {
            format!("{} changed path(s) allowed", changed_files.len())
        } else {
            format!("forbidden changes: {}", forbidden.join(", "))
        },
    });
    if let Some(feedback) = &task.visible_feedback {
        let feedback_output = run_check_in_fresh_copy(
            &work,
            &feedback.command,
            task.timeouts.visible_feedback_seconds,
            &runtime_artifacts,
        )?;
        checks.push(check_artifact("visible_feedback", &feedback_output));
    }
    checks.push(check_artifact("final_oracle", &oracle_output));

    let passed = oracle_output.success()
        && forbidden.is_empty()
        && (edited || final_message_available)
        && !output.timed_out;
    let hi_ok = output.success();
    let fail = classify(passed, hi_ok, edited || final_message_mode, &verify_output);
    let failure_confidence = fail.map(|kind| match kind {
        FailKind::Error if report.provider_error_kind.is_some() => "high",
        FailKind::NoEdits if changed_files.is_empty() => "high",
        FailKind::Compile if looks_like_build_error(&verify_output) => "high",
        FailKind::Logic => "medium",
        _ => "low",
    });

    let agent_process = if output.timed_out {
        AgentProcessOutcome::TimedOut
    } else if output.status.success() {
        AgentProcessOutcome::ExitedSuccessfully
    } else {
        AgentProcessOutcome::ExitedWithFailure
    };
    let false_verified = report.reported_success && !passed;
    let seconds = started.elapsed().as_secs_f64();
    let mut agent_output = String::from_utf8_lossy(&output.stdout).into_owned();
    agent_output.push_str(&String::from_utf8_lossy(&output.stderr));
    let turn_tokens = if growth.is_empty() {
        report.tokens
    } else {
        growth.iter().map(|turn| turn.total_tokens).sum()
    };
    let turn_input_tokens = if growth.is_empty() {
        report.input_tokens
    } else {
        growth.iter().map(|turn| turn.input_tokens).sum()
    };

    // Judge before deleting the workdir: the report and any written driver.py
    // must still be readable. Live judge is a no-op when judge.toml is absent.
    let max_request_tokens = max_request_tokens_from_report(&report_path);
    let judge = maybe_judge(task_dir, &report_path, &work);
    if let Some(report) = &judge {
        append_harness_finding(report, model_override.unwrap_or(""));
    }
    let tape = std::fs::read_to_string(&report_path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok());

    let _ = std::fs::remove_dir_all(&work);

    Ok(Candidate {
        index,
        temperature,
        seed: None,
        passed,
        fail,
        agent_process,
        agent_exit_code: output.status.code(),
        agent_output_summary: summarize_output(&agent_output),
        agent_output_truncated: output.output_truncated,
        reported_success: report.reported_success,
        false_verified,
        actual_model_route: report.actual_model_route,
        turn_outcome: report.turn_outcome,
        provider_error_kind: report.provider_error_kind,
        failure_mode: report.failure_mode,
        model_outcome: report.model_outcome,
        partial_artifact: report.partial_artifact,
        trace: report.trace,
        compat_fallbacks_used: report.compat_fallbacks_used,
        changed_files,
        verify_output_summary: summarize_output(&verify_output),
        failure_confidence,
        tokens: turn_tokens,
        input_tokens: turn_input_tokens,
        session_tokens: report.session_tokens,
        session_input_tokens: report.session_input_tokens,
        cost: report.cost,
        seconds,
        patch,
        patch_truncated,
        checks,
        trajectory: report.trajectory,
        growth,
        judge,
        max_request_tokens,
        tape,
    })
}

fn maybe_judge(task_dir: &Path, report_path: &Path, work: &Path) -> Option<serde_json::Value> {
    let rules_path = task_dir.join("judge.toml");
    if !rules_path.is_file() {
        return None;
    }
    let rules = hi_eval::judge::load_rules(&rules_path).ok()?;
    let raw = std::fs::read_to_string(report_path).ok()?;
    let mut report = serde_json::from_str::<serde_json::Value>(&raw).ok()?;
    if rules.process.require_output_slice {
        let src = std::fs::read_to_string(work.join("driver.py")).unwrap_or_default();
        report["harness"]["driver_bounds_output"] =
            serde_json::Value::Bool(hi_eval::judge::driver_bounds_output(&src));
    }
    serde_json::to_value(hi_eval::judge::judge_report(&report, &rules)).ok()
}

fn max_request_tokens_from_report(path: &Path) -> Option<u64> {
    let raw = std::fs::read_to_string(path).ok()?;
    let report: serde_json::Value = serde_json::from_str(&raw).ok()?;
    report
        .pointer("/telemetry/requests")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|r| r.get("input_tokens_est").and_then(|v| v.as_u64()))
        .max()
}

fn append_harness_finding(judge: &serde_json::Value, model: &str) {
    let process_fail = judge.get("process").and_then(|v| v.as_str()) == Some("fail");
    let budget_fail = judge.get("budget").and_then(|v| v.as_str()) == Some("fail");
    if !process_fail && !budget_fail {
        return;
    }
    let Ok(root) = std::env::var("HI_STATE_DIR") else {
        return;
    };
    if root.trim().is_empty() {
        return;
    }
    let dir = std::path::PathBuf::from(root).join("learning");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let shape = if process_fail {
        "harness_process"
    } else {
        "harness_budget"
    };
    let line = serde_json::json!({
        "ts": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "status": "failed",
        "stop_reason": "completed",
        "verification": "failed",
        "review": "not_required",
        "changed_files": 0,
        "model": model,
        "failure_shape": shape,
    });
    let Ok(body) = serde_json::to_string(&line) else {
        return;
    };
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("findings.jsonl"))
    {
        use std::io::Write;
        let _ = writeln!(file, "{body}");
    }
}

fn infrastructure_candidate(
    index: usize,
    temperature: f32,
    model: Option<&str>,
    seconds: f64,
    error: &anyhow::Error,
) -> Candidate {
    let message = format!("{error:#}");
    Candidate {
        index,
        temperature,
        seed: None,
        passed: false,
        fail: Some(FailKind::Error),
        agent_process: AgentProcessOutcome::InfrastructureError,
        agent_exit_code: None,
        agent_output_summary: summarize_output(&message),
        agent_output_truncated: false,
        reported_success: false,
        false_verified: false,
        actual_model_route: model.map(str::to_string),
        turn_outcome: None,
        provider_error_kind: Some("evaluator_infrastructure".to_string()),
        failure_mode: Some("infrastructure_error".to_string()),
        model_outcome: None,
        partial_artifact: None,
        trace: None,
        compat_fallbacks_used: Vec::new(),
        changed_files: Vec::new(),
        verify_output_summary: summarize_output(&message),
        failure_confidence: Some("high"),
        tokens: 0,
        input_tokens: 0,
        session_tokens: 0,
        session_input_tokens: 0,
        cost: None,
        seconds,
        patch: String::new(),
        patch_truncated: false,
        checks: vec![CandidateCheck {
            name: "evaluator_infrastructure".to_string(),
            passed: false,
            timed_out: false,
            output_truncated: false,
            duration_seconds: seconds,
            output_summary: summarize_output(&message),
        }],
        trajectory: Trajectory::default(),
        growth: Vec::new(),
        judge: None,
        max_request_tokens: None,
        tape: None,
    }
}

#[derive(Clone, Debug, Default)]
struct HarnessRunConfig {
    steps: Vec<u32>,
    seed_image_chars: Option<u64>,
    seed_tool_result_chars: Option<u64>,
    ignore_change_prefixes: Vec<String>,
}

impl HarnessRunConfig {
    fn needs_session(&self) -> bool {
        !self.steps.is_empty()
            || self.seed_image_chars.is_some()
            || self.seed_tool_result_chars.is_some()
    }

    fn needs_seed(&self) -> bool {
        self.seed_image_chars.is_some() || self.seed_tool_result_chars.is_some()
    }
}

fn harness_run_config(task_dir: &Path) -> HarnessRunConfig {
    let path = task_dir.join("judge.toml");
    let from_file = std::fs::read_to_string(&path)
        .ok()
        .and_then(|raw| toml::from_str::<hi_eval::judge::JudgeRules>(&raw).ok())
        .map(|rules| HarnessRunConfig {
            steps: rules.run.steps,
            seed_image_chars: rules.run.seed_image_chars,
            seed_tool_result_chars: rules.run.seed_tool_result_chars,
            ignore_change_prefixes: rules.run.ignore_change_prefixes,
        })
        .unwrap_or_default();
    if !from_file.steps.is_empty() || from_file.needs_seed() {
        return from_file;
    }
    let resume = std::env::var("HI_EVAL_RESUME")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    if resume {
        HarnessRunConfig {
            steps: vec![3, 8],
            ..from_file
        }
    } else {
        from_file
    }
}

fn restore_ignored_inner_paths(fixture: &Path, work: &Path, prefixes: &[String]) -> Result<()> {
    for prefix in prefixes {
        let rel = prefix.trim_end_matches('/');
        if rel.is_empty() || rel.contains("..") || Path::new(rel).is_absolute() {
            continue;
        }
        let src = fixture.join(rel);
        let dst = work.join(rel);
        if src.is_dir() {
            if dst.exists() {
                std::fs::remove_dir_all(&dst)?;
            }
            copy_dir(&src, &dst)?;
        } else if src.is_file() {
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

fn harness_resume_steps(cfg: &HarnessRunConfig) -> Vec<u32> {
    cfg.steps.iter().copied().filter(|&n| n >= 1).collect()
}

fn seed_harness_session(
    path: &Path,
    image_chars: Option<u64>,
    tool_result_chars: Option<u64>,
) -> Result<()> {
    use hi_ai::{Content, Message, Role};
    const MAX_SEED_IMAGE: u64 = 2_000_000;
    const MAX_SEED_RESULT: u64 = 200_000;
    let mut messages = Vec::new();
    if let Some(n) = image_chars {
        let n = n.min(MAX_SEED_IMAGE) as usize;
        messages.push(Message {
            role: Role::User,
            content: vec![
                Content::Text("see this screenshot".into()),
                Content::Image {
                    data: "A".repeat(n),
                    media_type: "image/png".into(),
                },
            ],
        });
        messages.push(Message::assistant(vec![Content::Text("ok".into())]));
    }
    if let Some(n) = tool_result_chars {
        let n = n.min(MAX_SEED_RESULT) as usize;
        messages.push(Message::user("read the blob"));
        messages.push(Message::assistant(vec![Content::ToolCall {
            id: "seed-read".into(),
            name: "read".into(),
            arguments: r#"{"path":"blob.txt"}"#.into(),
        }]));
        messages.push(Message::tool_result("seed-read", "X".repeat(n)));
        messages.push(Message::assistant(vec![Content::Text("noted".into())]));
    }
    // Pad past DEFAULT_KEEP_RECENT (3) so the seed is old on resume.
    for i in 0..3 {
        messages.push(Message::user(format!("continue {i}")));
        messages.push(Message::assistant(vec![Content::Text("ok".into())]));
    }
    let mut payload = String::new();
    for message in &messages {
        payload.push_str(&serde_json::to_string(message)?);
        payload.push('\n');
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, payload).with_context(|| format!("seeding session {}", path.display()))?;
    Ok(())
}

pub(crate) fn initialize_workspace(work: &Path, task: &Task) -> Result<()> {
    if task.workspace.kind == WorkspaceKind::Plain {
        return Ok(());
    }
    for args in [
        vec!["init", "-q"],
        vec!["add", "-A"],
        vec![
            "-c",
            "user.name=hi-eval",
            "-c",
            "user.email=hi-eval@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ] {
        let mut command = Command::new("git");
        command.current_dir(work).args(args);
        let output = command_output_with_timeout(&mut command, Duration::from_secs(30))?;
        if !output.success() {
            anyhow::bail!(
                "initializing task Git workspace: {}",
                output_summary(&output)
            );
        }
    }
    if task.workspace.kind == WorkspaceKind::DirtyRepository {
        let path = work.join(
            task.workspace
                .dirty_path
                .as_deref()
                .expect("validated dirty path"),
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            path,
            task.workspace
                .dirty_contents
                .as_deref()
                .expect("validated dirty contents"),
        )?;
    }
    Ok(())
}

fn run_check_in_fresh_copy(
    candidate_dir: &Path,
    command: &str,
    timeout_seconds: u64,
    runtime_artifacts: &[String],
) -> Result<crate::artifacts::TimedOutput> {
    let verification = make_workdir()?;
    let result = (|| {
        copy_dir(candidate_dir, &verification)?;
        crate::artifacts::prune_candidate_runtime_artifacts(&verification, runtime_artifacts)?;
        verify_output_in(&verification, command, timeout_seconds)
            .context("launching feedback check")
    })();
    let _ = std::fs::remove_dir_all(verification);
    result
}

fn check_artifact(name: &str, output: &crate::artifacts::TimedOutput) -> CandidateCheck {
    CandidateCheck {
        name: name.to_string(),
        passed: output.success(),
        timed_out: output.timed_out,
        output_truncated: output.output_truncated,
        duration_seconds: output.duration.as_secs_f64(),
        output_summary: summarize_output(&output_summary(output)),
    }
}

fn summarize_output(output: &str) -> String {
    const MAX: usize = 4000;
    let trimmed = output.trim();
    if trimmed.chars().count() <= MAX {
        return trimmed.to_string();
    }
    let head: String = trimmed.chars().take(1800).collect();
    let tail: String = trimmed
        .chars()
        .rev()
        .take(1800)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{head}\n\n[... verify output truncated ...]\n\n{tail}")
}

fn render_patch(
    before: &DirectorySnapshot,
    after: &DirectorySnapshot,
    changed: &[String],
) -> (String, bool) {
    const MAX_PATCH_BYTES: usize = 256 * 1024;
    let mut patch = String::new();
    let mut truncated = false;
    for path in changed {
        let old = before.get(path);
        let new = after.get(path);
        patch.push_str(&format!(
            "--- {}\n+++ {}\n",
            old.map_or("/dev/null", |_| path.as_str()),
            new.map_or("/dev/null", |_| path.as_str())
        ));
        match (old, new) {
            (Some(old), Some(new)) if old.kind != new.kind => patch.push_str(&format!(
                "old type {:?}\nnew type {:?}\n",
                old.kind, new.kind
            )),
            (None, Some(new)) => patch.push_str(&format!("new type {:?}\n", new.kind)),
            (Some(old), None) => patch.push_str(&format!("deleted type {:?}\n", old.kind)),
            _ => {}
        }
        match (old, new) {
            (Some(old), Some(new)) if old.bytes.contains(&0) || new.bytes.contains(&0) => {
                patch.push_str(&format!(
                    "Binary files differ ({} -> {} bytes)\n",
                    old.bytes.len(),
                    new.bytes.len()
                ));
            }
            (Some(old), Some(new)) => {
                #[cfg(unix)]
                if old.mode != new.mode {
                    patch.push_str(&format!(
                        "old mode {:o}\nnew mode {:o}\n",
                        old.mode & 0o7777,
                        new.mode & 0o7777
                    ));
                }
                for line in String::from_utf8_lossy(&old.bytes).lines() {
                    patch.push('-');
                    patch.push_str(line);
                    patch.push('\n');
                }
                for line in String::from_utf8_lossy(&new.bytes).lines() {
                    patch.push('+');
                    patch.push_str(line);
                    patch.push('\n');
                }
            }
            (None, Some(new)) if new.bytes.contains(&0) => {
                patch.push_str(&format!(
                    "Binary file created ({} bytes)\n",
                    new.bytes.len()
                ));
            }
            (None, Some(new)) => {
                for line in String::from_utf8_lossy(&new.bytes).lines() {
                    patch.push('+');
                    patch.push_str(line);
                    patch.push('\n');
                }
            }
            (Some(old), None) if old.bytes.contains(&0) => {
                patch.push_str(&format!(
                    "Binary file deleted ({} bytes)\n",
                    old.bytes.len()
                ));
            }
            (Some(old), None) => {
                for line in String::from_utf8_lossy(&old.bytes).lines() {
                    patch.push('-');
                    patch.push_str(line);
                    patch.push('\n');
                }
            }
            (None, None) => {}
        }
        if patch.len() > MAX_PATCH_BYTES {
            patch.truncate(MAX_PATCH_BYTES);
            patch.push_str("\n[... candidate patch truncated ...]\n");
            truncated = true;
            break;
        }
    }
    (patch, truncated)
}

#[derive(Default)]
struct ReportInfo {
    tokens: u64,
    input_tokens: u64,
    session_tokens: u64,
    session_input_tokens: u64,
    cost: Option<f64>,
    reported_success: bool,
    actual_model_route: Option<String>,
    turn_outcome: Option<serde_json::Value>,
    provider_error_kind: Option<String>,
    failure_mode: Option<String>,
    model_outcome: Option<serde_json::Value>,
    partial_artifact: Option<serde_json::Value>,
    trace: Option<serde_json::Value>,
    compat_fallbacks_used: Vec<String>,
    trajectory: Trajectory,
}

fn read_report_value(path: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn merge_harness_tapes(prev: serde_json::Value, next: serde_json::Value) -> serde_json::Value {
    let prev_tel = prev
        .get("telemetry")
        .cloned()
        .unwrap_or(serde_json::json!({}));
    let next_tel = next
        .get("telemetry")
        .cloned()
        .unwrap_or(serde_json::json!({}));
    let concat = |key: &str| -> serde_json::Value {
        let mut items = prev_tel
            .get(key)
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        items.extend(
            next_tel
                .get(key)
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default(),
        );
        serde_json::Value::Array(items)
    };
    let mut merged = next;
    if let Some(tel) = merged.get_mut("telemetry") {
        tel["tool_timeline"] = concat("tool_timeline");
        tel["requests"] = concat("requests");
        tel["compaction"] = concat("compaction");
        let prev_rounds = prev_tel
            .get("verify_rounds")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let next_rounds = next_tel
            .get("verify_rounds")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        tel["verify_rounds"] = serde_json::json!(prev_rounds.max(next_rounds));
    }
    merged
}

fn read_report(path: &Path) -> ReportInfo {
    let Ok(text) = std::fs::read_to_string(path) else {
        return ReportInfo::default();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return ReportInfo::default();
    };
    let tel = &value["telemetry"];
    let stopped_by_step_cap = tel["stopped_by_step_cap"]
        .as_bool()
        .unwrap_or_else(|| tel["hit_step_cap"].as_bool().unwrap_or(false));
    let trajectory = Trajectory {
        verify_rounds: tel["verify_rounds"].as_u64().unwrap_or(0) as u32,
        recovery_retries: tel["recovery_retries"].as_u64().unwrap_or(0) as u32,
        repeat_nudges: tel["repeat_nudges"].as_u64().unwrap_or(0) as u32,
        continue_nudges: tel["continue_nudges"].as_u64().unwrap_or(0) as u32,
        truncation_retries: tel["truncation_retries"].as_u64().unwrap_or(0) as u32,
        effective_max_steps: tel["effective_max_steps"].as_u64().unwrap_or(0) as u32,
        hit_step_cap: tel["hit_step_cap"].as_bool().unwrap_or(false),
        stopped_by_step_cap,
        stalled_unfinished: tel["stalled_unfinished"].as_bool().unwrap_or(false),
        stalled_repeating: tel["stalled_repeating"].as_bool().unwrap_or(false),
        quality_repair_nudges: tel["quality_repair_nudges"].as_u64().unwrap_or(0) as u32,
        review_repair_counts: tel["review_repair_counts"]
            .as_object()
            .map(|counts| {
                counts
                    .iter()
                    .map(|(mode, value)| (mode.clone(), value.as_u64().unwrap_or(0) as u32))
                    .collect()
            })
            .unwrap_or_default(),
        review_repair_exhaustion_reason: tel["review_repair_exhaustion_reason"]
            .as_str()
            .unwrap_or("")
            .to_string(),
        review_repair_stopped_by_exhaustion: tel["review_repair_stopped_by_exhaustion"]
            .as_bool()
            .unwrap_or(false),
        verify_attributions: tel["verify_attributions"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .map(|a| TrajectoryAttribution {
                        path: a["path"].as_str().unwrap_or("").to_string(),
                        line: a["line"].as_u64().map(|n| n as u32),
                        column: a["column"].as_u64().map(|n| n as u32),
                        message: a["message"].as_str().unwrap_or("").to_string(),
                        kind: a["kind"].as_str().unwrap_or("").to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        tool_calls: tel["tool_calls"].as_u64().unwrap_or(0) as u32,
        max_concurrent_batch: tel["max_concurrent_batch"].as_u64().unwrap_or(0) as u32,
        serial_runs: tel["serial_runs"].as_u64().unwrap_or(0) as u32,
        tool_timeline: tel
            .get("tool_timeline")
            .and_then(|v| v.as_array())
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|t| {
                let t = t.as_object()?;
                Some(TrajectoryToolCall {
                    tool: t
                        .get("tool")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    path: t
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    duration_ms: t.get("duration_ms").and_then(|v| v.as_u64()).unwrap_or(0),
                    error: t.get("error").and_then(|v| v.as_bool()).unwrap_or(false),
                    progress_kind: t
                        .get("progress_kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                })
            })
            .collect(),
        repeated_verify_failures: tel["repeated_verify_failures"].as_u64().unwrap_or(0) as u32,
        no_progress_streak: tel["no_progress_streak"].as_u64().unwrap_or(0) as u32,
        last_stall_reason: tel["last_stall_reason"].as_str().unwrap_or("").to_string(),
        progress_events: tel
            .get("progress_events")
            .and_then(|v| v.as_array())
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|e| {
                let e = e.as_object()?;
                Some(crate::results::TrajectoryProgressEvent {
                    kind: e
                        .get("kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    reason: e
                        .get("reason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    signature: e
                        .get("signature")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                })
            })
            .collect(),
    };
    let turn_outcome = value
        .get("outcome")
        .or_else(|| value.get("turn_outcome"))
        .or_else(|| value.pointer("/turn/outcome"))
        .filter(|outcome| !outcome.is_null())
        .cloned();
    let reported_success = turn_outcome
        .as_ref()
        .and_then(|outcome| outcome.get("verification"))
        .and_then(|verification| verification.as_str())
        .is_some_and(|verification| verification.eq_ignore_ascii_case("passed"))
        || value["verify_passed"].as_bool().unwrap_or(false);
    let route = turn_outcome
        .as_ref()
        .and_then(|outcome| outcome.get("effective_route"))
        .or_else(|| value.get("route"));
    let actual_model_route = route
        .and_then(|route| {
            let model = route.get("model")?.as_str()?;
            let provider = route.get("provider").and_then(|provider| provider.as_str());
            Some(provider.map_or_else(
                || model.to_string(),
                |provider| format!("{provider}/{model}"),
            ))
        })
        .or_else(|| value["model"].as_str().map(str::to_string));
    ReportInfo {
        tokens: value
            .pointer("/usage/turn/total_tokens")
            .and_then(|value| value.as_u64())
            .or_else(|| {
                value
                    .pointer("/turn_usage/total_tokens")
                    .and_then(|value| value.as_u64())
            })
            .or_else(|| value["turn_total_tokens"].as_u64())
            .or_else(|| value["total_tokens"].as_u64())
            .unwrap_or(0),
        input_tokens: value
            .pointer("/usage/turn/input_tokens")
            .and_then(|value| value.as_u64())
            .or_else(|| {
                value
                    .pointer("/turn_usage/input_tokens")
                    .and_then(|value| value.as_u64())
            })
            .or_else(|| value["turn_input_tokens"].as_u64())
            .or_else(|| value["input_tokens"].as_u64())
            .unwrap_or(0),
        session_tokens: value
            .pointer("/usage/session/total_tokens")
            .and_then(|value| value.as_u64())
            .or_else(|| value["session_total_tokens"].as_u64())
            .or_else(|| value["total_tokens"].as_u64())
            .unwrap_or(0),
        session_input_tokens: value
            .pointer("/usage/session/input_tokens")
            .and_then(|value| value.as_u64())
            .or_else(|| value["session_input_tokens"].as_u64())
            .or_else(|| value["input_tokens"].as_u64())
            .unwrap_or(0),
        cost: value
            .pointer("/usage/turn/cost")
            .and_then(|value| value.as_f64())
            .or_else(|| value["turn_cost"].as_f64())
            .or_else(|| value["cost"].as_f64()),
        reported_success,
        actual_model_route,
        turn_outcome,
        provider_error_kind: value
            .pointer("/provider_error/kind")
            .and_then(|kind| kind.as_str())
            .or_else(|| value["provider_error_kind"].as_str())
            .map(str::to_string),
        failure_mode: value["failure_mode"].as_str().map(str::to_string),
        model_outcome: value.get("model_outcome").cloned(),
        partial_artifact: value.get("partial_artifact").cloned(),
        trace: value.get("rsi").filter(|value| !value.is_null()).cloned(),
        compat_fallbacks_used: string_array(&value["compat_fallbacks_used"]),
        trajectory,
    }
}

fn string_array(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{dir_snapshot, read_report, run_candidate};
    use crate::artifacts::candidate_changed_paths;
    use crate::config::{EvalProfile, Task};

    #[test]
    fn candidate_changes_classify_only_known_new_build_artifacts() {
        let root = std::env::temp_dir().join(format!("hi-eval-snapshot-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("target/debug/deps")).unwrap();
        std::fs::write(root.join("target/preexisting.txt"), "keep\n").unwrap();
        std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
        let before = dir_snapshot(&root).unwrap();

        std::fs::write(root.join("target/debug/deps/build.log"), "compiled\n").unwrap();
        let after_build = dir_snapshot(&root).unwrap();
        assert!(
            candidate_changed_paths(&before, &after_build, &["main.rs".to_string()])
                .unwrap()
                .is_empty(),
            "known new build artifacts must not count as edits"
        );

        std::fs::write(root.join("target/preexisting.txt"), "changed\n").unwrap();
        std::fs::write(root.join("target/forbidden.txt"), "source-like\n").unwrap();
        let after_tamper = dir_snapshot(&root).unwrap();
        let tampering =
            candidate_changed_paths(&before, &after_tamper, &["main.rs".to_string()]).unwrap();
        assert!(tampering.contains(&"target/preexisting.txt".to_string()));
        assert!(tampering.contains(&"target/forbidden.txt".to_string()));

        std::fs::write(root.join("main.rs"), "fn main(){println!(\"x\");}\n").unwrap();
        let after_source = dir_snapshot(&root).unwrap();
        assert!(
            candidate_changed_paths(&before, &after_source, &["main.rs".to_string()])
                .unwrap()
                .contains(&"main.rs".to_string()),
            "source edits must still count"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn read_report_parses_review_repair_trajectory_fields() {
        let path = std::env::temp_dir().join(format!(
            "hi-eval-review-repair-report-{}.json",
            std::process::id()
        ));
        let report = serde_json::json!({
            "total_tokens": 123,
            "input_tokens": 100,
            "changed_files": [],
            "compat_fallbacks_used": [],
            "telemetry": {
                "effective_max_steps": 12,
                "verify_rounds": 0,
                "recovery_retries": 0,
                "repeat_nudges": 0,
                "continue_nudges": 0,
                "truncation_retries": 0,
                "hit_step_cap": false,
                "stopped_by_step_cap": false,
                "stalled_unfinished": true,
                "stalled_repeating": false,
                "quality_repair_nudges": 4,
                "review_repair_counts": {
                    "review_listing_only": 4
                },
                "review_repair_exhaustion_reason": "review_listing_only_exhausted",
                "review_repair_stopped_by_exhaustion": true,
                "verify_attributions": []
            }
        });
        std::fs::write(&path, serde_json::to_string(&report).unwrap()).unwrap();

        let parsed = read_report(&path);

        assert_eq!(parsed.tokens, 123);
        assert_eq!(parsed.input_tokens, 100);
        assert_eq!(parsed.trajectory.effective_max_steps, 12);
        assert_eq!(parsed.trajectory.quality_repair_nudges, 4);
        assert_eq!(
            parsed.trajectory.review_repair_counts["review_listing_only"],
            4
        );
        assert_eq!(
            parsed.trajectory.review_repair_exhaustion_reason,
            "review_listing_only_exhausted"
        );
        assert!(parsed.trajectory.review_repair_stopped_by_exhaustion);
        assert!(!parsed.trajectory.stopped_by_step_cap);
        assert!(!parsed.trajectory.hit_step_cap);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn read_report_parses_progress_and_repeated_failure_telemetry() {
        let path = std::env::temp_dir().join(format!(
            "hi-eval-progress-report-{}.json",
            std::process::id()
        ));
        let report = serde_json::json!({
            "total_tokens": 1,
            "input_tokens": 1,
            "changed_files": [],
            "compat_fallbacks_used": [],
            "telemetry": {
                "verify_rounds": 3,
                "repeated_verify_failures": 2,
                "no_progress_streak": 1,
                "last_stall_reason": "no new evidence",
                "progress_events": [
                    {"kind": "meaningful", "reason": "edited file", "signature": "edit:a.rs"},
                    {"kind": "none", "reason": "re-ran search", "signature": null}
                ],
                "tool_timeline": [
                    {"tool": "bash", "path": "", "duration_ms": 40, "error": false,
                     "progress_kind": "weak"}
                ]
            }
        });
        std::fs::write(&path, serde_json::to_string(&report).unwrap()).unwrap();

        let parsed = read_report(&path);
        let t = &parsed.trajectory;

        assert_eq!(t.verify_rounds, 3);
        assert_eq!(t.repeated_verify_failures, 2);
        assert_eq!(t.no_progress_streak, 1);
        assert_eq!(t.last_stall_reason, "no new evidence");
        assert_eq!(t.progress_events.len(), 2);
        assert_eq!(t.progress_events[0].kind, "meaningful");
        assert_eq!(t.progress_events[0].signature, "edit:a.rs");
        assert_eq!(t.progress_events[1].kind, "none");
        assert_eq!(t.progress_events[1].signature, "");
        assert_eq!(t.tool_timeline.len(), 1);
        assert_eq!(t.tool_timeline[0].progress_kind, "weak");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn read_report_prefers_v2_turn_usage_outcome_and_route() {
        let path =
            std::env::temp_dir().join(format!("hi-eval-v2-report-{}.json", std::process::id()));
        let report = serde_json::json!({
            "schema_version": 2,
            "usage": {
                "turn": {"input_tokens": 70, "total_tokens": 100, "cost": 0.25},
                "session": {"input_tokens": 700, "total_tokens": 1000}
            },
            "outcome": {
                "status": "completed",
                "verification": "passed",
                "review": "not_required",
                "stop_reason": "completed",
                "changed_files": ["solution.py"],
                "effective_route": {"provider": "example", "model": "coder"}
            },
            "telemetry": {}
        });
        std::fs::write(&path, serde_json::to_string(&report).unwrap()).unwrap();
        let parsed = read_report(&path);
        assert_eq!(parsed.input_tokens, 70);
        assert_eq!(parsed.tokens, 100);
        assert_eq!(parsed.cost, Some(0.25));
        assert!(parsed.reported_success);
        assert_eq!(parsed.actual_model_route.as_deref(), Some("example/coder"));
        assert!(parsed.turn_outcome.is_some());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn read_turn_metric_captures_tokens_and_goal_progress() {
        use super::read_turn_metric;
        let path =
            std::env::temp_dir().join(format!("hi-eval-turn-metric-{}.json", std::process::id()));
        let report = serde_json::json!({
            "total_tokens": 9000,
            "input_tokens": 8000,
            "telemetry": { "tool_calls": 7 },
            // The report's goal block is a summary, not the full sub_goals list.
            "goal": { "objective": "x", "status": "Active", "done": 2, "total": 4 }
        });
        std::fs::write(&path, serde_json::to_string(&report).unwrap()).unwrap();

        let m = read_turn_metric(&path, 3);
        assert_eq!(m.turn, 3);
        assert_eq!(m.input_tokens, 8000);
        assert_eq!(m.total_tokens, 9000);
        assert_eq!(m.tool_calls, 7);
        assert_eq!(m.goal_total, 4);
        assert_eq!(m.goal_done, 2);

        let _ = std::fs::remove_file(&path);

        // Missing report → zeroed metric, never a panic (fail-open).
        let absent = read_turn_metric(std::path::Path::new("/no/such/report.json"), 5);
        assert_eq!(absent.turn, 5);
        assert_eq!(absent.input_tokens, 0);
        assert_eq!(absent.goal_total, 0);
    }

    #[cfg(unix)]
    fn candidate_test_task() -> (std::path::PathBuf, Task) {
        let root = crate::artifacts::make_workdir().unwrap();
        std::fs::create_dir_all(root.join("fixture")).unwrap();
        std::fs::write(root.join("fixture/solution.py"), "VALUE = 0\n").unwrap();
        let task = toml::from_str(
            r#"
schema_version = 2
prompt = "set VALUE to 42"
allowed_changes = ["solution.py"]
[final_oracle]
command = "PYTHONPATH=. python3 -c 'from solution import VALUE; assert VALUE == 42'"
"#,
        )
        .unwrap();
        (root, task)
    }

    #[cfg(unix)]
    fn fake_hi(root: &std::path::Path, body: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = root.join("fake-hi");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&path, permissions).unwrap();
        path
    }

    #[test]
    #[cfg(unix)]
    fn no_op_candidate_cannot_pass_external_oracle() {
        let (root, task) = candidate_test_task();
        let hi = fake_hi(&root, "exit 0");
        let candidate = run_candidate(
            0,
            &hi,
            &root,
            &task,
            false,
            0.0,
            &[],
            EvalProfile::Default,
            None,
        )
        .unwrap();
        assert!(!candidate.passed);
        assert!(candidate.changed_files.is_empty());
        assert_eq!(candidate.fail, Some(crate::results::FailKind::NoEdits));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(unix)]
    fn forbidden_change_fails_even_when_oracle_behavior_passes() {
        let (root, task) = candidate_test_task();
        let hi = fake_hi(
            &root,
            "printf 'VALUE = 42\\n' > solution.py\nprintf hacked > tests.py\nexit 0",
        );
        let candidate = run_candidate(
            0,
            &hi,
            &root,
            &task,
            false,
            0.0,
            &[],
            EvalProfile::Default,
            None,
        )
        .unwrap();
        assert!(!candidate.passed);
        assert!(candidate.changed_files.contains(&"solution.py".to_string()));
        assert!(candidate.changed_files.contains(&"tests.py".to_string()));
        assert!(
            candidate
                .checks
                .iter()
                .any(|check| { check.name == "final_oracle" && check.passed })
        );
        assert!(
            candidate
                .checks
                .iter()
                .any(|check| { check.name == "allowed_changes" && !check.passed })
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(unix)]
    fn vcs_metadata_tampering_is_a_forbidden_candidate_change() {
        let (root, mut task) = candidate_test_task();
        task.workspace.kind = crate::config::WorkspaceKind::CleanRepository;
        let hi = fake_hi(
            &root,
            "printf 'VALUE = 42\\n' > solution.py\nprintf '\\n[evil]\\nvalue = true\\n' >> .git/config\nexit 0",
        );
        let candidate = run_candidate(
            0,
            &hi,
            &root,
            &task,
            false,
            0.0,
            &[],
            EvalProfile::Default,
            None,
        )
        .unwrap();

        assert!(!candidate.passed);
        assert!(candidate.changed_files.contains(&"solution.py".to_string()));
        assert!(candidate.changed_files.contains(&".git/config".to_string()));
        assert!(
            candidate
                .checks
                .iter()
                .any(|check| check.name == "final_oracle" && check.passed)
        );
        assert!(
            candidate
                .checks
                .iter()
                .any(|check| check.name == "allowed_changes" && !check.passed)
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(unix)]
    fn timed_out_candidate_is_preserved_and_cannot_pass() {
        let (root, mut task) = candidate_test_task();
        task.timeouts.candidate_seconds = 1;
        let hi = fake_hi(&root, "sleep 5");
        let candidate = run_candidate(
            0,
            &hi,
            &root,
            &task,
            false,
            0.0,
            &[],
            EvalProfile::Default,
            None,
        )
        .unwrap();
        assert!(!candidate.passed);
        assert_eq!(
            candidate.agent_process,
            crate::results::AgentProcessOutcome::TimedOut
        );
        assert!(candidate.seconds < 3.0, "timeout cleanup took too long");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(unix)]
    fn claimed_verification_that_fails_oracle_is_flagged() {
        let (root, task) = candidate_test_task();
        let hi = fake_hi(
            &root,
            r#"while [ "$#" -gt 0 ]; do if [ "$1" = "--report" ]; then shift; printf '%s' '{"schema_version":2,"outcome":{"status":"completed","verification":"passed","review":"not_required","stop_reason":"completed","changed_files":[],"effective_route":{"provider":"fake","model":"test"}},"usage":{"turn":{"input_tokens":1,"total_tokens":2}},"telemetry":{}}' > "$1"; fi; shift; done; exit 0"#,
        );
        let candidate = run_candidate(
            0,
            &hi,
            &root,
            &task,
            true,
            0.0,
            &[],
            EvalProfile::Default,
            None,
        )
        .unwrap();
        assert!(candidate.reported_success);
        assert!(candidate.false_verified);
        assert!(!candidate.passed);
        assert_eq!(candidate.actual_model_route.as_deref(), Some("fake/test"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn config_preserves_all_heterogeneous_candidate_results() {
        let (root, task) = candidate_test_task();
        let hi = fake_hi(&root, "printf 'VALUE = 42\\n' > solution.py\nexit 0");
        let temperatures = [0.2, 0.7, 1.0];
        let result = super::run_config(
            &hi,
            &root,
            &task,
            "test-best-of",
            false,
            &temperatures,
            &[],
            EvalProfile::Default,
            None,
            std::sync::Arc::new(tokio::sync::Semaphore::new(2)),
        )
        .await
        .unwrap();
        assert!(result.passed);
        assert_eq!(result.candidates.len(), 3);
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.temperature)
                .collect::<Vec<_>>(),
            temperatures
        );
        assert!(result.candidates.iter().all(|candidate| candidate.passed));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn merge_harness_tapes_unions_timeline_and_requests() {
        let first = serde_json::json!({
            "telemetry": {
                "verify_rounds": 0,
                "tool_timeline": [{"tool": "edit", "path": "solution.py"}],
                "requests": [{"input_tokens_est": 100}]
            }
        });
        let second = serde_json::json!({
            "telemetry": {
                "verify_rounds": 1,
                "tool_timeline": [{"tool": "bash", "path": ""}],
                "requests": [{"input_tokens_est": 200}]
            }
        });
        let merged = super::merge_harness_tapes(first, second);
        let tools: Vec<&str> = merged["telemetry"]["tool_timeline"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["tool"].as_str())
            .collect();
        assert_eq!(tools, ["edit", "bash"]);
        assert_eq!(merged["telemetry"]["verify_rounds"], 1);
        assert_eq!(merged["telemetry"]["requests"][1]["input_tokens_est"], 200);
    }

    #[test]
    #[cfg(unix)]
    fn harness_run_steps_use_session_file_and_judge_before_cleanup() {
        let (root, task) = candidate_test_task();
        std::fs::write(
            root.join("judge.toml"),
            r#"
[process]
require_tools = ["edit"]
[budget]
max_tool_result_chars = 50000
[run]
steps = [3, 8]
"#,
        )
        .unwrap();
        let args_log = root.join("args.log");
        let hi = fake_hi(
            &root,
            &format!(
                r#"printf '%s\n' "$*" >> "{log}"
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--report" ]; then
    shift
    printf '%s' '{{"schema_version":2,"outcome":{{"status":"completed"}},"telemetry":{{"tool_timeline":[{{"tool":"edit","path":"solution.py"}}],"requests":[{{"input_tokens_est":1234,"max_tool_result_chars":10}}]}}}}' > "$1"
  fi
  shift
done
printf 'VALUE = 42\n' > solution.py
exit 0
"#,
                log = args_log.display()
            ),
        );
        let candidate = run_candidate(
            0,
            &hi,
            &root,
            &task,
            false,
            0.0,
            &[],
            EvalProfile::Default,
            None,
        )
        .unwrap();
        let args = std::fs::read_to_string(&args_log).unwrap();
        assert!(
            args.contains("--session-file"),
            "resume helper must persist a session: {args}"
        );
        assert!(
            args.contains("--max-steps 3") && args.contains("--max-steps 8"),
            "both run.steps must be invoked: {args}"
        );
        assert!(candidate.judge.is_some(), "judge must run before cleanup");
        assert_eq!(candidate.max_request_tokens, Some(1234));
        assert_eq!(
            candidate
                .judge
                .as_ref()
                .and_then(|j| j.get("process"))
                .and_then(|v| v.as_str()),
            Some("pass")
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
