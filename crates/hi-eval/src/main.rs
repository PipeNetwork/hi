//! `hi-eval` — coding-task benchmark runner for `hi`.
//!
//! Runs each task under each config in an isolated copy of its fixture, then
//! scores a fresh copy with a pre-captured immutable oracle. Reports preserve
//! every candidate and distinguish candidate pass rate, solve@N, and standard
//! pass@k for exchangeable samples. This is how we measure whether a
//! lever (e.g. verification-in-the-loop) actually beats a baseline — including
//! a real backend like `openrouter/fusion`.
//!
//! **Agent-level path:** candidates always invoke the full `hi` binary (tools +
//! turn loop + optional `--verify`), never bare `hi-ai` completions. Use
//! `--configs=verify` for the repair-loop A/B, and `--agent-path` for a
//! model-free smoke of report schema + `--verify` wiring (see `agent_path`).
//!
//! Model selection flows through to `hi` via the usual env vars
//! (HI_MODEL / HI_BASE_URL / HI_API_KEY), so you compare backends by swapping
//! env, not code:
//!
//!   HI_MODEL=openrouter/fusion HI_API_KEY=… cargo run -p hi-eval -- bench/tasks
//!   cargo run -p hi-eval -- --agent-path
//!
//! Usage: hi-eval [TASKS_DIR]   (default: bench/tasks). Set HI_BIN to override
//! the hi binary path.

mod ab;
mod artifacts;
mod baseline;
mod comparison;
mod config;
mod reporting;
mod results;
mod runner;
mod selftest;
mod skeptic_detector;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use tokio::sync::Semaphore;

use artifacts::{
    default_artifacts_dir, dir_name, discover_tasks, find_hi, validate_tasks, write_artifact,
};
use baseline::{
    SuiteKind, capture_from_summary_file, capture_process_from_summary_file, classify_suites,
    compare_exit_code, compare_to_baseline, compare_to_process_baseline, default_baseline_path,
    default_baseline_path_for_suites, default_north_star_suites, ensure_placeholder,
    is_process_baseline_path, load_baseline, load_process_baseline, print_compare_report,
};
use config::{CONFIGS, Config, EvalProfile, Task};
use reporting::{evaluation_summary, print_summary, write_summary};
use results::{McpModelArtifact, RunResult};
use runner::run_config;
use selftest::run_self_test;

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new().context("creating tokio runtime")?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "judge") {
        return hi_eval::judge::run_cli(&args);
    }
    if args.first().is_some_and(|arg| arg == "compare") {
        let left = flag_value(&args, "--left")
            .ok_or_else(|| anyhow::anyhow!("compare requires --left <artifact-dir>"))?;
        let right = flag_value(&args, "--right")
            .ok_or_else(|| anyhow::anyhow!("compare requires --right <artifact-dir>"))?;
        let report = comparison::compare_dirs(Path::new(&left), Path::new(&right))?;
        let output = serde_json::to_string_pretty(&report)?;
        if let Some(path) = flag_value(&args, "--output") {
            std::fs::write(&path, &output)
                .with_context(|| format!("writing comparison report {path}"))?;
        } else {
            println!("{output}");
        }
        return Ok(());
    }
    if args.first().is_some_and(|arg| arg == "ab") {
        return run_ab(&args).await;
    }
    let validate = args.iter().any(|a| a == "--validate");
    let self_test = args.iter().any(|a| a == "--self-test");
    let agent_path_smoke = args.iter().any(|a| a == "--agent-path");
    // Offline baseline tools (no model required).
    if let Some(summary) = args
        .iter()
        .find_map(|a| a.strip_prefix("--write-baseline="))
    {
        let suites = args
            .iter()
            .find_map(|a| a.strip_prefix("--suites="))
            .map(|s| {
                s.split(',')
                    .map(str::trim)
                    .filter(|x| !x.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_else(default_north_star_suites);
        let baseline_path = args
            .iter()
            .find_map(|a| a.strip_prefix("--baseline-out="))
            .map(PathBuf::from)
            .unwrap_or_else(|| default_baseline_path_for_suites(&suites));
        let trials = args
            .iter()
            .find_map(|a| a.strip_prefix("--trials="))
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        let model_route = std::env::var("HI_MODEL")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let configs = args
            .iter()
            .find_map(|a| a.strip_prefix("--configs="))
            .map(|s| {
                s.split(',')
                    .map(str::trim)
                    .filter(|x| !x.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_else(|| vec!["baseline".into(), "verify".into()]);
        if is_process_baseline_path(&baseline_path)
            || matches!(
                classify_suites(&suites),
                SuiteKind::Quality | SuiteKind::Harness
            )
        {
            let baseline = capture_process_from_summary_file(Path::new(summary), &baseline_path)?;
            println!(
                "wrote process baseline {} (process_pass_rate={:?}, budget_pass_rate={:?})",
                baseline_path.display(),
                baseline.process_pass_rate,
                baseline.budget_pass_rate
            );
            return Ok(());
        }
        let baseline = capture_from_summary_file(
            Path::new(summary),
            &baseline_path,
            model_route,
            trials,
            suites,
            configs,
        )?;
        println!(
            "wrote baseline {} (solve_rate={:?}, false_verified_rate={:?})",
            baseline_path.display(),
            baseline.solve_rate,
            baseline.false_verified_rate
        );
        return Ok(());
    }
    if let Some(summary) = args
        .iter()
        .find_map(|a| a.strip_prefix("--compare-baseline="))
    {
        let baseline_path = args
            .iter()
            .find_map(|a| a.strip_prefix("--baseline="))
            .map(PathBuf::from)
            .unwrap_or_else(default_baseline_path);
        let text =
            std::fs::read_to_string(summary).with_context(|| format!("reading {summary}"))?;
        let summary: reporting::EvaluationSummary =
            serde_json::from_str(&text).with_context(|| format!("parsing {summary}"))?;
        let report = if is_process_baseline_path(&baseline_path) {
            let baseline = load_process_baseline(&baseline_path)?;
            compare_to_process_baseline(&baseline, &summary, 0.05)
        } else {
            let baseline = load_baseline(&baseline_path)?;
            compare_to_baseline(&baseline, &summary, 0.05, 0.05)
        };
        eprintln!("baseline {}", baseline_path.display());
        print_compare_report(&report);
        let code = compare_exit_code(&report);
        if code != 0 {
            std::process::exit(code);
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--ensure-baseline") {
        let path = args
            .iter()
            .find_map(|a| a.strip_prefix("--baseline-out="))
            .map(PathBuf::from)
            .unwrap_or_else(default_baseline_path);
        let b = ensure_placeholder(&path)?;
        println!(
            "{} {} (captured={})",
            if path.exists() { "baseline" } else { "wrote" },
            path.display(),
            b.is_captured()
        );
        return Ok(());
    }
    let north_star = args.iter().any(|a| a == "--north-star");
    let suite_roots: Vec<String> = if north_star {
        args.iter()
            .find_map(|a| a.strip_prefix("--suites="))
            .map(|s| {
                s.split(',')
                    .map(str::trim)
                    .filter(|x| !x.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_else(default_north_star_suites)
    } else {
        vec![
            args.iter()
                .find(|a| !a.starts_with("--"))
                .cloned()
                .unwrap_or_else(|| "bench/tasks".to_string()),
        ]
    };
    let profile = EvalProfile::parse(args.iter().find_map(|a| a.strip_prefix("--profile=")))?;
    let requested_models: Option<Vec<String>> = args
        .iter()
        .find_map(|a| a.strip_prefix("--models="))
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect()
        });
    // Skeptic detector eval: measure the reviewer's precision/recall on labeled
    // (forward/reversed) bug-fix diffs — independent of the task/config matrix.
    if args.iter().any(|a| a == "--skeptic-detector") {
        profile.validate_env()?;
        let repo = args
            .iter()
            .find_map(|a| a.strip_prefix("--repo="))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let n = args
            .iter()
            .find_map(|a| a.strip_prefix("--n="))
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let reviewer = args
            .iter()
            .find_map(|a| a.strip_prefix("--reviewer="))
            .unwrap_or("pipe/glm-5.2-fast")
            .to_string();
        let max_diff_lines = args
            .iter()
            .find_map(|a| a.strip_prefix("--max-diff-lines="))
            .and_then(|s| s.parse().ok())
            .unwrap_or(350);
        return skeptic_detector::run(skeptic_detector::Options {
            repo,
            hi_bin: find_hi()?,
            reviewer,
            provider_args: profile.hi_args().iter().map(|s| s.to_string()).collect(),
            n,
            max_diff_lines,
            concurrency: 4,
        })
        .await;
    }

    let artifacts_dir = args
        .iter()
        .find_map(|a| a.strip_prefix("--artifacts="))
        .map(PathBuf::from)
        .unwrap_or_else(default_artifacts_dir);

    // --configs=baseline,verify selects a subset of configs (default: all).
    let configs_filter: Option<Vec<String>> = args
        .iter()
        .find_map(|a| a.strip_prefix("--configs="))
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let active: Vec<&Config> = CONFIGS
        .iter()
        .filter(|c| {
            configs_filter
                .as_ref()
                .is_none_or(|f| f.iter().any(|n| n == c.name))
        })
        .collect();
    if active.is_empty() {
        bail!(
            "no configs match --configs; known: baseline, verify, best-of-3, goal-team, research-best-of-3"
        );
    }

    if self_test {
        return run_self_test(&active, profile).await;
    }
    if agent_path_smoke {
        return hi_eval::agent_path::run_agent_path_smoke();
    }

    // --trials=N repeats the whole matrix N times so the summary can report a
    // mean ± spread and pass@k (single runs are too noisy to trust).
    let trials: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--trials="))
        .map(|s| s.parse().context("--trials must be a positive integer"))
        .transpose()?
        .unwrap_or(3);
    if trials == 0 {
        bail!("--trials must be greater than zero");
    }

    let mut tasks = Vec::new();
    for root in &suite_roots {
        let found = discover_tasks(Path::new(root))?;
        if found.is_empty() {
            eprintln!("hi-eval: warning: no tasks under {root}");
        }
        tasks.extend(found);
    }
    // Stable order for reproducible artifacts.
    tasks.sort_by(|a, b| a.0.cmp(&b.0));
    tasks.dedup_by(|a, b| a.0 == b.0);
    if tasks.is_empty() {
        bail!(
            "no tasks (with task.toml) found under {}",
            suite_roots.join(", ")
        );
    }
    if north_star {
        eprintln!(
            "hi-eval: north-star ladder · {} suite root(s): {}",
            suite_roots.len(),
            suite_roots.join(", ")
        );
    }

    if validate {
        return validate_tasks(&tasks);
    }
    let timeout_override = |name: &str| -> Result<Option<u64>> {
        let value = args
            .iter()
            .find_map(|arg| arg.strip_prefix(&format!("--{name}=")))
            .map(|value| {
                value
                    .parse::<u64>()
                    .with_context(|| format!("--{name} must be a positive integer"))
            })
            .transpose()?;
        if value == Some(0) {
            bail!("--{name} must be greater than zero");
        }
        Ok(value)
    };
    let candidate_timeout = timeout_override("candidate-timeout")?;
    let feedback_timeout = timeout_override("feedback-timeout")?;
    let oracle_timeout = timeout_override("oracle-timeout")?;
    for (_, task) in &mut tasks {
        if let Some(value) = candidate_timeout {
            task.timeouts.candidate_seconds = value;
        }
        if let Some(value) = feedback_timeout {
            task.timeouts.visible_feedback_seconds = value;
        }
        if let Some(value) = oracle_timeout {
            task.timeouts.oracle_seconds = value;
        }
    }
    profile.validate_env()?;
    std::fs::create_dir_all(&artifacts_dir)
        .with_context(|| format!("creating artifacts dir {}", artifacts_dir.display()))?;

    let hi = find_hi()?;
    let model = default_eval_model(profile);
    let mcp_catalog = fetch_mcp_catalog(profile).await;
    let models_to_run = resolve_models_to_run(requested_models, &model, mcp_catalog.as_ref())?;
    let mcp_models: std::collections::HashMap<String, Option<McpModelArtifact>> = models_to_run
        .iter()
        .map(|model_id| {
            let meta = mcp_catalog
                .as_ref()
                .and_then(|catalog| catalog.get(model_id))
                .map(mcp_model_artifact);
            (model_id.clone(), meta)
        })
        .collect();
    // Mirror hi's env toggles so the run header and artifacts label which side of
    // each A/B this run is: `HI_CONDENSE=0` / `HI_RECOVERY_SAMPLING=0` to disable.
    let env_on = |name: &str| {
        !matches!(
            std::env::var(name).ok().as_deref(),
            Some("0" | "off" | "false" | "no")
        )
    };
    let condense_on = env_on("HI_CONDENSE");
    let recovery_on = env_on("HI_RECOVERY_SAMPLING");
    // Off by default (unlike condense/recovery) — on only when the var is present,
    // matching hi's own gating. The child `hi` inherits it via the env.
    let write_subagents_on = std::env::var_os("HI_WRITE_SUBAGENTS").is_some();
    // Off by default; on when set — each run becomes a planner-decomposed goal.
    let goal_mode_on = std::env::var_os("HI_EVAL_GOAL").is_some();
    eprintln!(
        "hi-eval: {} task(s) × {} config(s) × {} model(s) × {trials} trial(s) · models={} · profile={} · condense={} · recovery={} · write_subagents={} · goal_mode={} · hi={} · artifacts={}",
        tasks.len(),
        active.len(),
        models_to_run.len(),
        models_to_run.join(","),
        profile.label(),
        if condense_on { "on" } else { "off" },
        if recovery_on { "on" } else { "off" },
        if write_subagents_on { "on" } else { "off" },
        if goal_mode_on { "on" } else { "off" },
        hi.display(),
        artifacts_dir.display()
    );

    let mut results = Vec::new();
    // Cap concurrent candidates to avoid overwhelming the provider with parallel
    // requests. Each candidate is a subprocess that makes its own HTTP calls, so
    // the real limit is the provider's rate limit, not local CPU.
    let concurrency_arg = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--concurrency="));
    let concurrency_env = std::env::var("HI_EVAL_CONCURRENCY").ok();
    let concurrency = resolve_concurrency(concurrency_arg, concurrency_env.as_deref())?;
    let semaphore = Arc::new(Semaphore::new(concurrency));

    for trial in 0..trials {
        if trials > 1 {
            eprintln!("--- trial {}/{trials} ---", trial + 1);
        }
        let trial_results = run_eval_trial(EvalTrial {
            hi: &hi,
            trial,
            tasks: &tasks,
            active: &active,
            models_to_run: &models_to_run,
            mcp_models: &mcp_models,
            profile,
            artifacts_dir: &artifacts_dir,
            condense_on,
            recovery_on,
            write_subagents_on,
            goal_mode_on,
            semaphore: semaphore.clone(),
        })
        .await?;
        results.extend(trial_results);
    }

    print_summary(&results, tasks.len(), &active, trials);
    write_summary(&artifacts_dir, &results, tasks.len(), trials)?;

    // Suite-aware baseline: quality/harness lock process/budget; SWE uses
    // core-0.2.json. Do not compare a 7-row routing suite to coding solve_rate.
    let baseline_path = args
        .iter()
        .find_map(|a| a.strip_prefix("--baseline="))
        .map(PathBuf::from)
        .unwrap_or_else(|| default_baseline_path_for_suites(&suite_roots));
    let summary = evaluation_summary(&results, tasks.len(), trials);
    let summary_path = artifacts_dir.join("summary.json");
    let write_baseline_flag = args.iter().any(|a| a == "--write-baseline");
    let process_suite = matches!(
        classify_suites(&suite_roots),
        SuiteKind::Quality | SuiteKind::Harness
    ) || is_process_baseline_path(&baseline_path);
    let auto_capture_placeholder = args.iter().any(|a| a == "--capture-baseline-if-empty")
        || std::env::var_os("HI_EVAL_CAPTURE_BASELINE").is_some();
    if process_suite {
        if write_baseline_flag {
            match capture_process_from_summary_file(&summary_path, &baseline_path) {
                Ok(b) => eprintln!(
                    "process baseline captured → {} (process_pass_rate={:?}, budget_pass_rate={:?})",
                    baseline_path.display(),
                    b.process_pass_rate,
                    b.budget_pass_rate
                ),
                Err(err) => eprintln!("process baseline capture failed: {err:#}"),
            }
        }
        match load_process_baseline(&baseline_path) {
            Ok(baseline) => {
                let report = compare_to_process_baseline(&baseline, &summary, 0.05);
                eprintln!("baseline {}", baseline_path.display());
                print_compare_report(&report);
                if args.iter().any(|a| a == "--fail-on-baseline-regression") {
                    let code = compare_exit_code(&report);
                    if code != 0 {
                        std::process::exit(code);
                    }
                }
            }
            Err(err) => eprintln!(
                "note: no process baseline at {} ({err:#}) — use scripts/check_quality_regression.sh or --write-baseline",
                baseline_path.display()
            ),
        }
    } else if write_baseline_flag
        || (auto_capture_placeholder
            && load_baseline(&baseline_path)
                .map(|b| !b.is_captured())
                .unwrap_or(true))
    {
        let suites = suite_roots.clone();
        let configs: Vec<String> = active.iter().map(|c| c.name.to_string()).collect();
        let model_route = models_to_run.first().cloned().filter(|m| m != "(unset)");
        match capture_from_summary_file(
            &summary_path,
            &baseline_path,
            model_route,
            trials as u32,
            suites,
            configs,
        ) {
            Ok(b) => eprintln!(
                "baseline captured → {} (solve_rate={:?}, false_verified_rate={:?})",
                baseline_path.display(),
                b.solve_rate,
                b.false_verified_rate
            ),
            Err(err) => eprintln!("baseline capture failed: {err:#}"),
        }
    }
    if !process_suite {
        if let Ok(baseline) = load_baseline(&baseline_path) {
            let report = compare_to_baseline(&baseline, &summary, 0.05, 0.05);
            eprintln!("baseline {}", baseline_path.display());
            print_compare_report(&report);
            if args.iter().any(|a| a == "--fail-on-baseline-regression") {
                let code = compare_exit_code(&report);
                if code != 0 {
                    std::process::exit(code);
                }
            }
        } else {
            eprintln!(
                "note: no baseline at {} — run with --write-baseline after a matrix, or `hi-eval --ensure-baseline`",
                baseline_path.display()
            );
        }
    }
    Ok(())
}

struct EvalTrial<'a> {
    hi: &'a Path,
    trial: usize,
    tasks: &'a [(PathBuf, Task)],
    active: &'a [&'a Config],
    models_to_run: &'a [String],
    mcp_models: &'a std::collections::HashMap<String, Option<McpModelArtifact>>,
    profile: EvalProfile,
    artifacts_dir: &'a Path,
    condense_on: bool,
    recovery_on: bool,
    write_subagents_on: bool,
    goal_mode_on: bool,
    semaphore: Arc<Semaphore>,
}

async fn run_eval_trial(spec: EvalTrial<'_>) -> Result<Vec<RunResult>> {
    let mut futs = Vec::new();
    for model_id in spec.models_to_run {
        let mcp_model = spec.mcp_models.get(model_id).cloned().flatten();
        for (dir, task) in spec.tasks {
            let label = task.name.clone().unwrap_or_else(|| dir_name(dir));
            for config in spec.active {
                let hi = spec.hi.to_path_buf();
                let dir = dir.clone();
                let task = task.clone();
                let config_name = config.name.to_string();
                let use_verify = config.use_verify;
                let temperatures = config.temperatures.to_vec();
                let config_env = config.env;
                let candidate_semaphore = spec.semaphore.clone();
                let artifacts_dir = spec.artifacts_dir.to_path_buf();
                let label2 = label.clone();
                let model_for_run = model_id.clone();
                let mcp_model = mcp_model.clone();
                let profile = spec.profile;
                let condense_on = spec.condense_on;
                let recovery_on = spec.recovery_on;
                let write_subagents_on = spec.write_subagents_on;
                let goal_mode_on = spec.goal_mode_on;
                let trial = spec.trial;
                futs.push(tokio::spawn(async move {
                    let model_override =
                        (model_for_run != "(unset)").then_some(model_for_run.clone());
                    let mut result = run_config(
                        &hi,
                        &dir,
                        &task,
                        &config_name,
                        use_verify,
                        &temperatures,
                        config_env,
                        profile,
                        model_override,
                        candidate_semaphore,
                    )
                    .await
                    .with_context(|| {
                        format!(
                            "running task '{}' [{}] model={}",
                            label2, config_name, model_for_run
                        )
                    })?;
                    result.task = label2.clone();
                    result.trial = trial;
                    result.model = model_for_run.clone();
                    result.mcp_model = mcp_model;
                    write_artifact(
                        &artifacts_dir,
                        profile,
                        condense_on,
                        recovery_on,
                        write_subagents_on,
                        goal_mode_on,
                        &result,
                    )?;
                    eprintln!(
                        "  {:10} {:4} {}  model={} ({} cand, {} tok, {:.1}s)",
                        config_name,
                        if result.passed { "PASS" } else { "FAIL" },
                        label2,
                        model_for_run,
                        result.candidates.len(),
                        result.tokens,
                        result.seconds
                    );
                    Ok::<_, anyhow::Error>(result)
                }));
            }
        }
    }
    let mut results = Vec::new();
    for fut in futs {
        match fut.await.context("joining eval task")? {
            Ok(result) => results.push(result),
            Err(err) => {
                eprintln!("  eval error: {err:#}");
                return Err(err);
            }
        }
    }
    Ok(results)
}

async fn run_ab(args: &[String]) -> Result<()> {
    let rest: Vec<String> = args.iter().skip(1).cloned().collect();
    let baseline_raw = flag_value(&rest, "--baseline-bin")
        .or_else(|| {
            std::env::var("HI_BIN_BASELINE")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .context("ab requires --baseline-bin <abs-path> or HI_BIN_BASELINE")?;
    let candidate_raw = flag_value(&rest, "--candidate-bin")
        .or_else(|| {
            std::env::var("HI_BIN_CANDIDATE")
                .ok()
                .filter(|s| !s.is_empty())
        })
        .context("ab requires --candidate-bin <abs-path> or HI_BIN_CANDIDATE")?;
    let baseline_bin = ab::require_absolute_executable(&baseline_raw, "baseline")?;
    let candidate_bin = ab::require_absolute_executable(&candidate_raw, "candidate")?;
    let baseline_meta = ab::inspect_binary(&baseline_bin)?;
    let candidate_meta = ab::inspect_binary(&candidate_bin)?;

    let profile = EvalProfile::parse(rest.iter().find_map(|a| a.strip_prefix("--profile=")))?;
    let requested_models: Option<Vec<String>> = rest
        .iter()
        .find_map(|a| a.strip_prefix("--models="))
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect()
        });
    let artifacts_root = rest
        .iter()
        .find_map(|a| a.strip_prefix("--artifacts="))
        .map(PathBuf::from)
        .unwrap_or_else(default_artifacts_dir);
    let configs_filter: Option<Vec<String>> = rest
        .iter()
        .find_map(|a| a.strip_prefix("--configs="))
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let active: Vec<&Config> = CONFIGS
        .iter()
        .filter(|c| {
            configs_filter
                .as_ref()
                .is_none_or(|f| f.iter().any(|n| n == c.name))
        })
        .collect();
    if active.is_empty() {
        bail!(
            "no configs match --configs; known: baseline, verify, best-of-3, goal-team, research-best-of-3"
        );
    }
    let trials: usize = rest
        .iter()
        .find_map(|a| a.strip_prefix("--trials="))
        .map(|s| s.parse().context("--trials must be a positive integer"))
        .transpose()?
        .unwrap_or(3);
    if trials == 0 {
        bail!("--trials must be greater than zero");
    }
    let suite_root = rest
        .iter()
        .enumerate()
        .find_map(|(i, a)| {
            if a.starts_with("--") {
                return None;
            }
            if i > 0 && matches!(rest[i - 1].as_str(), "--baseline-bin" | "--candidate-bin") {
                return None;
            }
            Some(a.clone())
        })
        .unwrap_or_else(|| "bench/quality".to_string());

    let mut tasks = discover_tasks(Path::new(&suite_root))?;
    tasks.sort_by(|a, b| a.0.cmp(&b.0));
    if tasks.is_empty() {
        bail!("no tasks (with task.toml) found under {suite_root}");
    }
    profile.validate_env()?;
    std::fs::create_dir_all(&artifacts_root)
        .with_context(|| format!("creating artifacts dir {}", artifacts_root.display()))?;

    let model = default_eval_model(profile);
    let mcp_catalog = fetch_mcp_catalog(profile).await;
    let models_to_run = resolve_models_to_run(requested_models, &model, mcp_catalog.as_ref())?;
    let mcp_models: std::collections::HashMap<String, Option<McpModelArtifact>> = models_to_run
        .iter()
        .map(|model_id| {
            let meta = mcp_catalog
                .as_ref()
                .and_then(|catalog| catalog.get(model_id))
                .map(mcp_model_artifact);
            (model_id.clone(), meta)
        })
        .collect();
    let env_on = |name: &str| {
        !matches!(
            std::env::var(name).ok().as_deref(),
            Some("0" | "off" | "false" | "no")
        )
    };
    let condense_on = env_on("HI_CONDENSE");
    let recovery_on = env_on("HI_RECOVERY_SAMPLING");
    let write_subagents_on = std::env::var_os("HI_WRITE_SUBAGENTS").is_some();
    let goal_mode_on = std::env::var_os("HI_EVAL_GOAL").is_some();
    let concurrency_arg = rest
        .iter()
        .find_map(|arg| arg.strip_prefix("--concurrency="));
    let concurrency_env = std::env::var("HI_EVAL_CONCURRENCY").ok();
    let concurrency = resolve_concurrency(concurrency_arg, concurrency_env.as_deref())?;
    let semaphore = Arc::new(Semaphore::new(concurrency));

    let meta = ab::AbMeta {
        schema_version: 1,
        baseline: baseline_meta.clone(),
        candidate: candidate_meta.clone(),
        trials,
        configs: active.iter().map(|c| c.name.to_string()).collect(),
        models: models_to_run.clone(),
        tasks: tasks
            .iter()
            .map(|(dir, task)| task.name.clone().unwrap_or_else(|| dir_name(dir)))
            .collect(),
        trial_order_rule: "even trials: baseline then candidate; odd trials: swapped",
        env: ab::snapshot_eval_env(),
    };
    std::fs::write(
        artifacts_root.join("ab_meta.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;

    eprintln!(
        "hi-eval ab: {} task(s) × {} config(s) × {} model(s) × {trials} trial(s) · baseline={} ({}) · candidate={} ({}) · artifacts={}",
        tasks.len(),
        active.len(),
        models_to_run.len(),
        baseline_bin.display(),
        &baseline_meta.sha256[..12.min(baseline_meta.sha256.len())],
        candidate_bin.display(),
        &candidate_meta.sha256[..12.min(candidate_meta.sha256.len())],
        artifacts_root.display()
    );

    let mut baseline_results = Vec::new();
    let mut candidate_results = Vec::new();
    for trial in 0..trials {
        if trials > 1 {
            eprintln!("--- trial {}/{trials} ---", trial + 1);
        }
        for side in ab::create_trial_order(trial) {
            let (hi, arm_dir, sink) = match side {
                ab::AbSide::Baseline => (
                    baseline_bin.as_path(),
                    artifacts_root.join("baseline"),
                    &mut baseline_results,
                ),
                ab::AbSide::Candidate => (
                    candidate_bin.as_path(),
                    artifacts_root.join("candidate"),
                    &mut candidate_results,
                ),
            };
            std::fs::create_dir_all(&arm_dir)?;
            eprintln!("  arm {}", side.as_str());
            let trial_results = run_eval_trial(EvalTrial {
                hi,
                trial,
                tasks: &tasks,
                active: &active,
                models_to_run: &models_to_run,
                mcp_models: &mcp_models,
                profile,
                artifacts_dir: &arm_dir,
                condense_on,
                recovery_on,
                write_subagents_on,
                goal_mode_on,
                semaphore: semaphore.clone(),
            })
            .await?;
            sink.extend(trial_results);
        }
    }

    write_summary(
        &artifacts_root.join("baseline"),
        &baseline_results,
        tasks.len(),
        trials,
    )?;
    write_summary(
        &artifacts_root.join("candidate"),
        &candidate_results,
        tasks.len(),
        trials,
    )?;
    print_summary(&baseline_results, tasks.len(), &active, trials);
    print_summary(&candidate_results, tasks.len(), &active, trials);

    let report = ab::build_ab_report(
        baseline_meta,
        candidate_meta,
        trials,
        &baseline_results,
        &candidate_results,
    );
    let report_path = artifacts_root.join("ab_report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    eprintln!(
        "ab_report: process Δ {:?}  solve@N Δ {:.3}  → {}",
        report.overall.process_pass_rate_delta,
        report.overall.solve_at_n_delta,
        report_path.display()
    );
    Ok(())
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1).cloned())
        .or_else(|| {
            args.iter()
                .find_map(|arg| arg.strip_prefix(&format!("{flag}=")).map(str::to_string))
        })
}

fn resolve_concurrency(cli: Option<&str>, env: Option<&str>) -> Result<usize> {
    let value = cli
        .or(env)
        .map(str::parse::<usize>)
        .transpose()
        .context("candidate concurrency must be a positive integer")?
        .unwrap_or(4);
    if value == 0 {
        bail!("candidate concurrency must be greater than zero");
    }
    Ok(value)
}

fn default_eval_model(profile: EvalProfile) -> String {
    std::env::var("HI_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| match profile {
            EvalProfile::Pipenetwork | EvalProfile::PipenetworkMcp => {
                Some("pipe/deepseek-v4-flash-vision-exp".to_string())
            }
            EvalProfile::Default => None,
        })
        .unwrap_or_else(|| "(unset)".to_string())
}

async fn fetch_mcp_catalog(
    profile: EvalProfile,
) -> Option<std::collections::HashMap<String, hi_ai::PipeMcpModelMetadata>> {
    if !profile.uses_mcp_metadata() {
        return None;
    }
    let key = std::env::var("HI_API_KEY")
        .or_else(|_| std::env::var("PIPENETWORK_API_KEY"))
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .ok()?;
    let url = std::env::var("HI_MCP_URL").unwrap_or_else(|_| hi_ai::PIPE_MCP_DEFAULT_URL.into());
    let client = hi_ai::PipeMcpClient::new(url, key);
    match client.model_metadata().await {
        Ok(models) => Some(models),
        Err(err) => {
            eprintln!("hi-eval: MCP model metadata not loaded: {err:#}");
            None
        }
    }
}

fn resolve_models_to_run(
    requested: Option<Vec<String>>,
    default_model: &str,
    mcp_catalog: Option<&std::collections::HashMap<String, hi_ai::PipeMcpModelMetadata>>,
) -> Result<Vec<String>> {
    let mut models = requested.unwrap_or_else(|| vec![default_model.to_string()]);
    if models.is_empty() {
        models.push(default_model.to_string());
    }
    models.sort();
    models.dedup();
    if let Some(catalog) = mcp_catalog {
        let missing: Vec<String> = models
            .iter()
            .filter(|model| !catalog.contains_key(model.as_str()))
            .cloned()
            .collect();
        if !missing.is_empty() {
            bail!(
                "--models contains id(s) not visible through MCP: {}",
                missing.join(", ")
            );
        }
    }
    Ok(models)
}

fn mcp_model_artifact(model: &hi_ai::PipeMcpModelMetadata) -> McpModelArtifact {
    McpModelArtifact {
        model_id: model.id.clone(),
        provider_label: model.provider_label.clone(),
        available: model.available,
        status: model.status.clone(),
        unavailable_reasons: model.unavailable_reasons.clone(),
        capabilities: model.capabilities.clone(),
    }
}

#[cfg(test)]
mod main_tests {
    use super::resolve_concurrency;

    #[test]
    fn concurrency_default_override_and_zero_rejection() {
        assert_eq!(resolve_concurrency(None, None).unwrap(), 4);
        assert_eq!(resolve_concurrency(Some("2"), Some("9")).unwrap(), 2);
        assert!(resolve_concurrency(Some("0"), None).is_err());
        assert!(resolve_concurrency(None, Some("0")).is_err());
        assert!(resolve_concurrency(Some("many"), None).is_err());
    }
}
