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

mod artifacts;
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
use config::{CONFIGS, Config, EvalProfile};
use reporting::{print_summary, write_summary};
use results::McpModelArtifact;
use runner::run_config;
use selftest::run_self_test;

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new().context("creating tokio runtime")?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let validate = args.iter().any(|a| a == "--validate");
    let self_test = args.iter().any(|a| a == "--self-test");
    let agent_path_smoke = args.iter().any(|a| a == "--agent-path");
    let tasks_dir = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "bench/tasks".to_string());
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
        bail!("no configs match --configs; known: baseline, verify, best-of-3, goal-team");
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

    let mut tasks = discover_tasks(Path::new(&tasks_dir))?;
    if tasks.is_empty() {
        bail!("no tasks (with task.toml) found under {tasks_dir}");
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
    let mcp_model_metadata = |model_id: &str| {
        mcp_catalog
            .as_ref()
            .and_then(|catalog| catalog.get(model_id))
            .map(mcp_model_artifact)
    };
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
        // Run all (model, task, config) cells for this trial concurrently.
        let mut futs = Vec::new();
        for model_id in &models_to_run {
            let mcp_model = mcp_model_metadata(model_id);
            for (dir, task) in &tasks {
                let label = task.name.clone().unwrap_or_else(|| dir_name(dir));
                for config in &active {
                    let hi = hi.clone();
                    let dir = dir.clone();
                    let task = task.clone();
                    let config_name = config.name.to_string();
                    let use_verify = config.use_verify;
                    let temperatures = config.temperatures.to_vec();
                    let config_env = config.env;
                    let candidate_semaphore = semaphore.clone();
                    let artifacts_dir = artifacts_dir.clone();
                    let label2 = label.clone();
                    let model_for_run = model_id.clone();
                    let mcp_model = mcp_model.clone();
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
        for fut in futs {
            match fut.await.context("joining eval task")? {
                Ok(result) => results.push(result),
                Err(err) => {
                    eprintln!("  eval error: {err:#}");
                    return Err(err);
                }
            }
        }
    }

    print_summary(&results, tasks.len(), &active, trials);
    write_summary(&artifacts_dir, &results, tasks.len(), trials)?;
    Ok(())
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
                Some("ipop/coder-balanced".to_string())
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
