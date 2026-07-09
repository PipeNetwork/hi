//! `hi-eval` — coding-task benchmark runner for `hi`.
//!
//! Runs each task under each config in an isolated copy of its fixture, scores
//! pass/fail by the task's own verify command (ground truth), and reports
//! pass-rate and time per config. This is how we measure whether a
//! lever (e.g. verification-in-the-loop) actually beats a baseline — including
//! a real backend like `openrouter/fusion`.
//!
//! Model selection flows through to `hi` via the usual env vars
//! (HI_MODEL / HI_BASE_URL / HI_API_KEY), so you compare backends by swapping
//! env, not code:
//!
//!   HI_MODEL=openrouter/fusion HI_API_KEY=… cargo run -p hi-eval -- bench/tasks
//!
//! Usage: hi-eval [TASKS_DIR]   (default: bench/tasks). Set HI_BIN to override
//! the hi binary path.

mod artifacts;
mod config;
mod reporting;
mod results;
mod runner;
mod selftest;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use tokio::sync::Semaphore;

use artifacts::{
    default_artifacts_dir, dir_name, discover_tasks, find_hi, validate_tasks, write_artifact,
};
use config::{CONFIGS, Config, EvalProfile, Task};
use reporting::print_summary;
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
        bail!("no configs match --configs; known: baseline, verify, best-of-3");
    }

    if self_test {
        return run_self_test(&active, profile).await;
    }

    // --trials=N repeats the whole matrix N times so the summary can report a
    // mean ± spread and pass@k (single runs are too noisy to trust).
    let trials: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--trials="))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);

    let tasks = discover_tasks(Path::new(&tasks_dir))?;
    if tasks.is_empty() {
        bail!("no tasks (with task.toml) found under {tasks_dir}");
    }

    if validate {
        return validate_tasks(&tasks);
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
    eprintln!(
        "hi-eval: {} task(s) × {} config(s) × {} model(s) × {trials} trial(s) · models={} · profile={} · condense={} · recovery={} · write_subagents={} · hi={} · artifacts={}",
        tasks.len(),
        active.len(),
        models_to_run.len(),
        models_to_run.join(","),
        profile.label(),
        if condense_on { "on" } else { "off" },
        if recovery_on { "on" } else { "off" },
        if write_subagents_on { "on" } else { "off" },
        hi.display(),
        artifacts_dir.display()
    );

    let mut results = Vec::new();
    // Cap concurrent candidates to avoid overwhelming the provider with parallel
    // requests. Each candidate is a subprocess that makes its own HTTP calls, so
    // the real limit is the provider's rate limit, not local CPU.
    let semaphore = Arc::new(Semaphore::new(
        std::env::var("HI_EVAL_CONCURRENCY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4),
    ));

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
                    let task_prompt = task.prompt.clone();
                    let task_verify = task.verify.clone();
                    let config_name = config.name.to_string();
                    let use_verify = config.use_verify;
                    let temperatures = config.temperatures.to_vec();
                    let sem = semaphore.clone();
                    let artifacts_dir = artifacts_dir.clone();
                    let label2 = label.clone();
                    let model_for_run = model_id.clone();
                    let mcp_model = mcp_model.clone();
                    futs.push(tokio::spawn(async move {
                        let _permit = sem.acquire().await;
                        let task = Task {
                            name: Some(label2.clone()),
                            prompt: task_prompt,
                            verify: task_verify,
                        };
                        let model_override =
                            (model_for_run != "(unset)").then_some(model_for_run.clone());
                        let mut result = run_config(
                            &hi,
                            &dir,
                            &task,
                            &config_name,
                            use_verify,
                            &temperatures,
                            profile,
                            model_override,
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
                            &result,
                        )?;
                        eprintln!(
                            "  {:10} {:4} {}  model={} ({} cand, {} tok, {:.1}s)",
                            config_name,
                            if result.passed { "PASS" } else { "FAIL" },
                            label2,
                            model_for_run,
                            result.candidates,
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
    Ok(())
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
