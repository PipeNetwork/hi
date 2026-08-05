//! Headless Diff Lab commands and the provider-backed TUI runner.
//!
//! This is the configuration boundary for API comparisons: it resolves named
//! profiles, constructs `hi-ai` providers, and passes only non-secret target
//! metadata into `hi-diff`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use hi_ai::{ChatRequest, CompatMode, DeepSeekCompat, Message, RequestProfile, ToolMode};
use hi_diff::{
    ApiTarget, BackendKind, CaseVerdict, DiffMode, DiffRunSnapshot, DiffRunSpec,
    EquivalenceContract, LocalTarget, RunStatus, RunStore, TargetSpec, Verdict, default_root,
    run_provider_targets, run_smoke,
};

use crate::config::{self, Config};
use crate::provider::{build_provider, provider_label};

pub(crate) async fn run_cli(args: &[String]) -> Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("help");
    match command {
        "run" => run(args.get(1..).unwrap_or_default()).await,
        "inspect" => inspect(args.get(1..).unwrap_or_default()),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => bail!("unknown diff-lab command '{other}' (try `hi diff-lab help`)"),
    }
}

async fn run(args: &[String]) -> Result<()> {
    let mode = value(args, "--mode").unwrap_or_else(|| "local".into());
    let mode = match mode.as_str() {
        "local" => DiffMode::LocalParity,
        "api" | "response" => DiffMode::ApiResponse,
        "agent" | "agents" => DiffMode::AgentOutcome,
        other => bail!("unknown diff-lab mode '{other}' (use local, api, or agent)"),
    };
    let seed = value(args, "--seed")
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(42);
    let cases = value(args, "--cases")
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(256);
    let root = value(args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(default_root);

    if mode == DiffMode::ApiResponse {
        return run_api(args, seed, cases, root).await;
    }

    let targets = smoke_targets_for(mode);
    let mut spec = DiffRunSpec::new(mode, seed, targets);
    spec.case_count = cases;
    spec.artifact_root = Some(root.clone());
    spec.validate()?;
    let store = RunStore::new(&root)?;
    store.write_spec(&spec)?;
    store.write_named_spec(&spec)?;
    let snapshot = run_smoke(&spec)?;
    let snapshot_path = store.write_named_snapshot(&snapshot)?;
    print_summary(&spec, &snapshot, &snapshot_path);
    if snapshot.mismatches > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Build the callback consumed by the TUI overlay. `Config` contains no
/// resolved secrets in the callback type; key lookup happens only inside the
/// spawned run and never enters a `DiffRunSpec` or artifact.
pub(crate) fn build_tui_api_runner(config: Config) -> hi_tui::DiffApiRunner {
    Arc::new(move |request| {
        let config = config.clone();
        Box::pin(run_api_request(config, request, default_root()))
    })
}

async fn run_api(args: &[String], seed: u64, cases: u64, root: PathBuf) -> Result<()> {
    let prompt = value(args, "--prompt")
        .or_else(|| value(args, "--request"))
        .or_else(|| {
            value(args, "--prompt-file").and_then(|path| std::fs::read_to_string(path).ok())
        })
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .context("API mode requires --prompt TEXT or --prompt-file PATH")?;
    let raw_targets = values(args, "--target");
    anyhow::ensure!(
        raw_targets.len() >= 2,
        "API mode requires at least two --target PROFILE:MODEL values"
    );
    let targets = raw_targets
        .iter()
        .enumerate()
        .map(|(index, raw)| parse_target(raw, index))
        .collect::<Result<Vec<_>>>()?;
    let max_requests = value(args, "--max-requests")
        .context("API mode requires an explicit --max-requests ceiling")?
        .parse::<u64>()
        .context("--max-requests must be an integer")?;
    let max_concurrency = value(args, "--concurrency")
        .or_else(|| value(args, "--max-concurrency"))
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(targets.len().min(4).max(1));
    let request = hi_tui::DiffApiRunRequest {
        prompt,
        targets,
        seed,
        cases,
        max_concurrency,
        max_requests,
        max_tokens: value(args, "--max-tokens")
            .map(|value| value.parse())
            .transpose()?
            .unwrap_or(4096),
    };
    let config_path = value(args, "--config").map(PathBuf::from);
    let snapshot = run_api_request(
        config::load_config(config_path.as_deref())?,
        request,
        root.clone(),
    )
    .await?;
    let spec_path = root.join(format!("run-{}-spec.json", snapshot.run_id));
    print!("API targets completed; manifest: {}\n", spec_path.display());
    if snapshot.mismatches > 0 {
        std::process::exit(1);
    }
    Ok(())
}

pub(crate) async fn run_api_request(
    config: Config,
    request: hi_tui::DiffApiRunRequest,
    root: PathBuf,
) -> Result<DiffRunSnapshot> {
    anyhow::ensure!(
        request.targets.len() >= 2,
        "API runs need at least two targets"
    );
    anyhow::ensure!(request.cases > 0, "API run cases must be greater than zero");
    anyhow::ensure!(
        request.max_tokens > 0,
        "API max tokens must be greater than zero"
    );
    anyhow::ensure!(
        request.max_concurrency > 0,
        "API concurrency must be greater than zero"
    );
    let request_count = request
        .cases
        .checked_mul(request.targets.len() as u64)
        .context("API request count overflowed")?;
    anyhow::ensure!(
        request.max_requests >= request_count,
        "request ceiling {} is below the planned {} provider requests",
        request.max_requests,
        request_count
    );

    let mut target_specs = Vec::with_capacity(request.targets.len());
    for target in &request.targets {
        let settings = config::resolve_named_profile(&config, &target.profile)
            .with_context(|| format!("resolving Diff Lab target profile '{}'", target.profile))?;
        target_specs.push(TargetSpec::Api(ApiTarget {
            name: target.name.clone(),
            profile: target.profile.clone(),
            model: target.model.clone(),
            provider: provider_label(settings.provider).to_string(),
        }));
    }

    let mut spec = DiffRunSpec::new(DiffMode::ApiResponse, request.seed, target_specs);
    spec.case_count = request.cases;
    spec.max_concurrency = request.max_concurrency;
    spec.contract = EquivalenceContract {
        mode: DiffMode::ApiResponse,
        exact_text: false,
        normalize_whitespace: true,
        require_schema_valid: false,
        require_same_tool_calls: true,
        ..EquivalenceContract::default()
    };
    spec.artifact_root = Some(root.clone());
    spec.validate()?;
    let store = RunStore::new(&root)?;
    store.write_spec(&spec)?;
    store.write_named_spec(&spec)?;
    let mut snapshot = DiffRunSnapshot::pending(&spec);
    snapshot.status = RunStatus::Running;
    store.append_event(&spec.run_id, &hi_diff::DiffEvent::Started(snapshot.clone()))?;

    let prompt = request.prompt.trim().to_string();
    anyhow::ensure!(!prompt.is_empty(), "API prompt must not be empty");
    let seed = request.seed;
    let targets = request.targets.clone();
    let contract = spec.contract.clone();
    let config_for_cases = config.clone();
    let run_id = spec.run_id.clone();
    let mut results = futures_util::stream::iter(0..request.cases)
        .map(move |case_index| {
            let config = config_for_cases.clone();
            let targets = targets.clone();
            let prompt = prompt.clone();
            let contract = contract.clone();
            let case_id = format!("api-{seed}-{case_index:08x}");
            let request_id = format!("diff-{run_id}-{case_index}");
            async move {
                run_one_api_case(
                    config,
                    &targets,
                    &prompt,
                    &request_id,
                    &case_id,
                    &contract,
                    request.max_tokens,
                )
                .await
            }
        })
        .buffer_unordered(request.max_concurrency);

    while let Some(result) = results.next().await {
        let verdict = result?;
        snapshot.cases_completed += 1;
        match verdict.verdict {
            Verdict::Mismatch => snapshot.mismatches += 1,
            Verdict::ExecutionError => snapshot.errors += 1,
            _ => {}
        }
        if verdict.verdict != Verdict::Equivalent {
            snapshot.recent_failures.push(verdict.clone());
            if snapshot.recent_failures.len() > 8 {
                snapshot.recent_failures.remove(0);
            }
        }
        store.append_event(&spec.run_id, &hi_diff::DiffEvent::CaseFinished(verdict))?;
    }
    snapshot.status = if snapshot.mismatches == 0 && snapshot.errors == 0 {
        RunStatus::Completed
    } else {
        RunStatus::Failed
    };
    store.write_named_snapshot(&snapshot)?;
    store.append_event(
        &spec.run_id,
        &hi_diff::DiffEvent::Finished(snapshot.clone()),
    )?;
    Ok(snapshot)
}

async fn run_one_api_case(
    config: Config,
    targets: &[hi_tui::DiffApiTarget],
    prompt: &str,
    request_id: &str,
    case_id: &str,
    contract: &EquivalenceContract,
    max_tokens: u32,
) -> Result<CaseVerdict> {
    let mut providers = Vec::with_capacity(targets.len());
    let mut api_targets = Vec::with_capacity(targets.len());
    for target in targets {
        let settings = config::resolve_named_profile(&config, &target.profile)
            .with_context(|| format!("resolving Diff Lab target profile '{}';", target.profile))?;
        let provider = build_provider(&settings);
        api_targets.push(ApiTarget {
            name: target.name.clone(),
            profile: target.profile.clone(),
            model: target.model.clone(),
            provider: provider_label(settings.provider).to_string(),
        });
        providers.push((api_targets.last().cloned().unwrap(), provider));
    }
    let request = ChatRequest {
        model: api_targets
            .first()
            .map(|target| target.model.clone())
            .unwrap_or_default(),
        request_id: Some(request_id.to_string()),
        retry_attempt: 0,
        user_turn: true,
        canonical_objective: Some(prompt.to_string()),
        messages: Arc::new(vec![Message::user(prompt)]),
        tools: Arc::from(Vec::<hi_ai::ToolSpec>::new()),
        max_tokens,
        temperature: Some(0.0),
        top_p: None,
        frequency_penalty: None,
        thinking_budget: None,
        reasoning_effort: None,
        profile: RequestProfile {
            compat: CompatMode::Auto,
            tool_mode: ToolMode::ChatOnly,
            stream_usage: Some(true),
            deepseek_compat: DeepSeekCompat::Auto,
            deepseek_strict: None,
            deepseek_thinking: None,
        },
    };
    Ok(run_provider_targets(case_id, request, providers, contract).await?)
}

fn parse_target(raw: &str, index: usize) -> Result<hi_tui::DiffApiTarget> {
    let (profile, model) = raw
        .split_once(':')
        .context("targets must use PROFILE:MODEL, e.g. pipenetwork:pipe/glm-5.2")?;
    anyhow::ensure!(!profile.trim().is_empty(), "target profile cannot be empty");
    anyhow::ensure!(!model.trim().is_empty(), "target model cannot be empty");
    Ok(hi_tui::DiffApiTarget {
        name: format!("target-{}", index + 1),
        profile: profile.trim().to_string(),
        model: model.trim().to_string(),
    })
}

fn smoke_targets_for(mode: DiffMode) -> Vec<TargetSpec> {
    match mode {
        DiffMode::LocalParity => vec![
            TargetSpec::Local(LocalTarget {
                name: "reference".into(),
                backend: BackendKind::Cpu,
                model_path: ".".into(),
                model_fingerprint: None,
            }),
            TargetSpec::Local(LocalTarget {
                name: "candidate".into(),
                backend: BackendKind::Custom,
                model_path: ".".into(),
                model_fingerprint: None,
            }),
        ],
        DiffMode::AgentOutcome => vec![
            TargetSpec::Agent(hi_diff::AgentTarget {
                name: "agent-a".into(),
                profile: "default".into(),
                model: "current".into(),
                provider: "configured".into(),
                verify_commands: Vec::new(),
            }),
            TargetSpec::Agent(hi_diff::AgentTarget {
                name: "agent-b".into(),
                profile: "default".into(),
                model: "current".into(),
                provider: "configured".into(),
                verify_commands: Vec::new(),
            }),
        ],
        DiffMode::ApiResponse => unreachable!("API targets are built from profiles"),
    }
}

fn inspect(args: &[String]) -> Result<()> {
    let root = value(args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(default_root);
    let run_id = args
        .iter()
        .find(|arg| !arg.starts_with('-'))
        .context("usage: hi diff-lab inspect <run-id> [--out DIR]")?;
    let path = root.join(format!("run-{run_id}-snapshot.json"));
    let body =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    println!("{body}");
    Ok(())
}

fn print_summary(spec: &DiffRunSpec, snapshot: &DiffRunSnapshot, snapshot_path: &std::path::Path) {
    println!(
        "diff-lab {} · run {} · {:?} · {}/{} cases · {} mismatches · {} errors\n{}",
        spec.mode.label(),
        spec.run_id,
        snapshot.status,
        snapshot.cases_completed,
        snapshot.cases_total,
        snapshot.mismatches,
        snapshot.errors,
        snapshot_path.display(),
    );
}

fn values(args: &[String], flag: &str) -> Vec<String> {
    args.windows(2)
        .filter(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
        .collect()
}

fn value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn print_help() {
    println!(
        "hi diff-lab\n\nCommands:\n  run --mode api --prompt TEXT --target PROFILE:MODEL --target PROFILE:MODEL --max-requests N [--max-tokens N] [--cases N] [--concurrency N] [--config PATH] [--out DIR]\n  run [--mode local|agent] [--seed N] [--cases N] [--out DIR]\n  inspect <run-id> [--out DIR]\n\nAPI targets resolve the normal hi profiles, so PipeNetwork can be used with e.g. `pipenetwork:pipe/glm-5.2` and `pipenetwork:pipe/kimi3`. Keys are read from the existing config/environment and never persisted by Diff Lab."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_profile_and_model_target_without_touching_credentials() {
        let target = parse_target("pipe-work:pipe/glm-5.2", 0).unwrap();
        assert_eq!(target.name, "target-1");
        assert_eq!(target.profile, "pipe-work");
        assert_eq!(target.model, "pipe/glm-5.2");
    }

    #[test]
    fn target_manifest_contains_profile_and_model_but_not_api_key() {
        let spec = DiffRunSpec::new(
            DiffMode::ApiResponse,
            7,
            vec![
                TargetSpec::Api(ApiTarget {
                    name: "glm".into(),
                    profile: "pipe-work".into(),
                    model: "pipe/glm-5.2".into(),
                    provider: "pipenetwork".into(),
                }),
                TargetSpec::Api(ApiTarget {
                    name: "kimi".into(),
                    profile: "pipe-work".into(),
                    model: "pipe/kimi3".into(),
                    provider: "pipenetwork".into(),
                }),
            ],
        );
        let encoded = serde_json::to_string(&spec).unwrap();
        assert!(encoded.contains("pipe/glm-5.2"));
        assert!(encoded.contains("pipe/kimi3"));
        assert!(!encoded.contains("api_key"));
        assert!(!encoded.contains("Authorization"));
    }
}
