//! Profile-driven third-party evaluation commands.
//!
//! This command layer owns orchestration only. Format knowledge lives in
//! `hi-eval-adapters`; execution remains in `hi-eval` backends and the legacy
//! `hi-eval` binary during the migration.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use hi_eval::backends::{DockerMount, DockerRunSpec, HarborDockerBackend};
use hi_eval::{
    ArtifactSpec, AttemptRecord, AttemptStatus, ClaimLevel, EnvironmentSpec, EvalEvidence,
    EvalInput, EvalManifest, EvalProfile, EvalScore, EvalStateStore, IdentityDetails, ImportStore,
    PreparationReceipt, ProgressEvent, RunIdentity, RunRecord, RunStatus, TaskPackage,
    VerifierSpec, command_output_with_timeout,
};
use hi_eval_adapters::{ADAPTER_API_VERSION, plan_directory};

use crate::review_target::resolve_runtime_roots;

pub(crate) async fn run_eval_cli(args: &[String]) -> Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("help");
    match command {
        "import" | "prepare" => prepare(args).await,
        "run" => run_profile(args).await,
        "status" => status(args),
        "report" => report(args),
        "stop" => stop(args),
        "cleanup" => cleanup(args),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => bail!("unknown eval command {other:?}; run `hi eval --help`"),
    }
}

async fn prepare(args: &[String]) -> Result<()> {
    let manifest_path =
        flag_path(args, "--manifest").unwrap_or_else(|| PathBuf::from("evals/manifest.toml"));
    let manifest = EvalManifest::load(&manifest_path)?;
    let profile_name = flag_string(args, "--profile")
        .or_else(|| args.get(1).filter(|value| !value.starts_with('-')).cloned())
        .or_else(|| manifest.profiles.keys().next().cloned())
        .context("manifest has no profile; pass --profile <name>")?;
    let profile = manifest.profile(&profile_name)?;
    let (workspace_root, state_root) = resolve_runtime_roots()?;
    let state_root = flag_path(args, "--state").unwrap_or_else(|| state_root.join("evals"));
    let store_root = flag_path(args, "--store").unwrap_or_else(|| state_root.join("imports"));
    let docker_backend = if matches!(profile.backend.as_str(), "harbor" | "docker") {
        let backend = HarborDockerBackend::from_environment();
        backend.check_runtime()?;
        Some(backend)
    } else {
        None
    };
    let manifest_root = manifest_path
        .canonicalize()
        .with_context(|| format!("resolving manifest {}", manifest_path.display()))?
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| workspace_root.clone());
    let store = ImportStore::new(&store_root);
    let mut dataset_digests = std::collections::BTreeMap::new();
    let mut task_datasets = std::collections::BTreeMap::new();
    for dataset_name in &profile.datasets {
        let source = manifest
            .datasets
            .get(dataset_name)
            .with_context(|| format!("profile references unknown dataset {dataset_name:?}"))?;
        let source_path = if source.source.is_absolute() {
            source.source.clone()
        } else {
            manifest_root.join(&source.source)
        };
        let plan = plan_directory(
            dataset_name.clone(),
            source.adapter.clone(),
            source_path,
            source.revision.clone(),
            profile.claim_level,
        )?;
        let imported = store.import(&plan)?;
        if let Some(backend) = &docker_backend {
            for task in &imported.tasks {
                let task_root = imported.root.join(&task.path);
                let package = task.package.as_ref().with_context(|| {
                    format!("Docker task {:?} has no normalized package", task.id)
                })?;
                backend.ensure_image(package, &task_root)?;
                if let Some(verifier) = profile.verifier.as_ref().or(package.verifier.as_ref()) {
                    docker_verifier_image(backend, package, verifier, &task_root)?;
                }
            }
        }
        for task in &imported.tasks {
            if let Some(previous) = task_datasets.insert(task.id.clone(), dataset_name.clone()) {
                bail!(
                    "task id {:?} appears in both datasets {:?} and {:?}; task ids must be unique within a profile",
                    task.id,
                    previous,
                    dataset_name
                );
            }
        }
        dataset_digests.insert(dataset_name.clone(), imported.digest);
    }
    let manifest_digest = manifest.digest()?;
    let identity = build_identity(
        &profile_name,
        &manifest_digest,
        dataset_digests.clone(),
        profile,
    )?;
    let receipt = PreparationReceipt {
        schema_version: hi_eval::PLATFORM_SCHEMA_VERSION,
        profile: profile_name.clone(),
        manifest_digest,
        datasets: dataset_digests,
        store_root: store
            .root()
            .canonicalize()
            .context("resolving prepared import store")?,
        identity,
    };
    let path = EvalStateStore::new(&state_root).write_preparation(&receipt)?;
    println!(
        "prepared profile {profile_name} → {} (identity {})",
        path.display(),
        receipt.identity.digest
    );
    Ok(())
}

async fn run_profile(args: &[String]) -> Result<()> {
    let manifest_path =
        flag_path(args, "--manifest").unwrap_or_else(|| PathBuf::from("evals/manifest.toml"));
    let manifest = EvalManifest::load(&manifest_path)?;
    let profile_name = flag_string(args, "--profile")
        .or_else(|| args.get(1).filter(|value| !value.starts_with('-')).cloned())
        .or_else(|| args.get(1).filter(|value| !value.starts_with('-')).cloned())
        .or_else(|| manifest.profiles.keys().next().cloned())
        .context("run requires --profile <name>")?;
    let profile = manifest.profile(&profile_name)?;
    if !matches!(
        profile.backend.as_str(),
        "host" | "legacy-host" | "harbor" | "docker"
    ) {
        bail!(
            "profile backend {:?} is not enabled by the host CLI yet; use the corresponding hi-eval backend directly",
            profile.backend
        );
    }
    let (_, default_state_root) = resolve_runtime_roots()?;
    let state_root = flag_path(args, "--state").unwrap_or_else(|| default_state_root.join("evals"));
    let state = EvalStateStore::new(&state_root);
    let receipt = state.read_preparation(&profile_name).with_context(|| {
        format!("profile {profile_name:?} is not prepared; run `hi eval prepare --profile {profile_name}`")
    })?;
    let manifest_digest = manifest.digest()?;
    let expected = build_identity(
        &profile_name,
        &manifest_digest,
        receipt.datasets.clone(),
        profile,
    )?;
    if receipt.identity.digest != expected.digest {
        bail!(
            "prepared profile is stale (manifest, task package, binary, policy, or runtime changed); re-run `hi eval prepare --profile {profile_name}`"
        );
    }
    let store_root = flag_path(args, "--store")
        .or_else(|| {
            (!receipt.store_root.as_os_str().is_empty()).then(|| receipt.store_root.clone())
        })
        .unwrap_or_else(|| state_root.join("imports"));
    let store = ImportStore::new(&store_root);
    let eval_binary = (!matches!(profile.backend.as_str(), "harbor" | "docker"))
        .then(find_eval_binary)
        .transpose()?;
    let profile_root = state.profile_root(&profile_name)?;
    let fresh = args.iter().any(|arg| arg == "--fresh");
    if fresh {
        let _ = fs::remove_file(profile_root.join("stop.requested"));
        let _ = fs::remove_dir_all(profile_root.join("attempts"));
        let _ = fs::remove_dir_all(profile_root.join("evidence"));
        let _ = fs::remove_dir_all(profile_root.join("comparisons"));
        let _ = fs::remove_file(profile_root.join("progress.jsonl"));
        let _ = fs::remove_file(profile_root.join("report.json"));
    } else {
        // A stop request is a one-run control signal. Consuming it here makes
        // the documented `stop` → `run` resume flow work without requiring
        // `--fresh` (which would discard durable attempts).
        let _ = fs::remove_file(profile_root.join("stop.requested"));
    }
    let now = now_unix();
    let previous_start = (!fresh)
        .then(|| state.read_run(&profile_name))
        .transpose()?
        .flatten()
        .map(|run| run.started_at_unix);
    state.write_run(&RunRecord {
        schema_version: hi_eval::PLATFORM_SCHEMA_VERSION,
        profile: profile_name.clone(),
        identity: expected.clone(),
        status: RunStatus::Running,
        started_at_unix: previous_start.unwrap_or(now),
        updated_at_unix: now,
    })?;
    let artifact_root = profile_root.join("evidence");
    fs::create_dir_all(&artifact_root)?;
    let arms = if profile.arms.is_empty() {
        if profile.models.len() > 1 {
            profile
                .models
                .iter()
                .enumerate()
                .map(|(index, model)| hi_eval::DifferentialArmConfig {
                    name: format!("model-{index}-{}", model.replace(['/', '\\'], "_")),
                    command: None,
                    environment: Default::default(),
                    model: Some(model.clone()),
                })
                .collect()
        } else {
            vec![hi_eval::DifferentialArmConfig {
                name: "default".into(),
                command: None,
                environment: Default::default(),
                model: None,
            }]
        }
    } else {
        profile.arms.clone()
    };
    let mut failures = Vec::new();
    let mut stopped = false;
    for dataset_name in &profile.datasets {
        let digest = receipt
            .datasets
            .get(dataset_name)
            .with_context(|| format!("prepared profile has no dataset {dataset_name:?}"))?;
        let dataset = store.load(store.root().join(dataset_name).join(digest))?;
        let dataset_root = dataset.root.join("tasks");
        if !dataset_root.is_dir() {
            bail!(
                "prepared dataset is missing task root: {}",
                dataset_root.display()
            );
        }
        for task in dataset.tasks {
            if !profile.selectors.is_empty()
                && !profile
                    .selectors
                    .iter()
                    .any(|selector| selector == &task.id)
            {
                continue;
            }
            let task_root = dataset.root.join(&task.path);
            for arm in &arms {
                for trial in 0..profile.trials {
                    if profile_root.join("stop.requested").is_file() {
                        stopped = true;
                        break;
                    }
                    if let Some(existing) =
                        state.read_attempt(&profile_name, &task.id, &arm.name, trial)?
                        && existing.identity_digest == expected.digest
                        && matches!(
                            existing.status,
                            AttemptStatus::Passed | AttemptStatus::Failed
                        )
                    {
                        if existing.status == AttemptStatus::Failed {
                            failures.push(format!("{dataset_name}/{}/{}", task.id, arm.name));
                        }
                        continue;
                    }
                    let output = artifact_root
                        .join(dataset_name)
                        .join(&task.id)
                        .join(&arm.name)
                        .join(trial.to_string());
                    rotate_attempt_output(&output)?;
                    fs::create_dir_all(&output)?;
                    let record = match execute_attempt(
                        eval_binary.as_deref(),
                        &task_root,
                        &task,
                        profile,
                        arm,
                        trial,
                        &expected,
                        &output,
                    ) {
                        Ok(record) => record,
                        Err(error) => AttemptRecord {
                            profile: profile_name.clone(),
                            task: task.id.clone(),
                            arm: arm.name.clone(),
                            trial,
                            status: AttemptStatus::InfrastructureFailed,
                            identity_digest: expected.digest.clone(),
                            claim_level: profile.claim_level,
                            score: None,
                            evidence: Some(EvalEvidence {
                                claim_level: profile.claim_level,
                                backend: Some(profile.backend.clone()),
                                runtime: Some("attempt-launch-failed".into()),
                                task_digest: Some(task.digest.clone()),
                                error: Some(error.to_string()),
                                ..EvalEvidence::default()
                            }),
                        },
                    };
                    let terminal = record.status;
                    state.write_attempt(&record)?;
                    state.append_progress(
                        &profile_name,
                        &ProgressEvent {
                            task: task.id.clone(),
                            arm: arm.name.clone(),
                            trial,
                            status: terminal,
                            identity_digest: expected.digest.clone(),
                            message: None,
                        },
                    )?;
                    if !matches!(terminal, AttemptStatus::Passed) {
                        failures.push(format!("{dataset_name}/{}/{}", task.id, arm.name));
                    }
                }
                if stopped {
                    break;
                }
            }
            if arms.len() > 1 {
                for trial in 0..profile.trials {
                    let mut comparison_arms = Vec::with_capacity(arms.len());
                    for arm in &arms {
                        let Some(record) =
                            state.read_attempt(&profile_name, &task.id, &arm.name, trial)?
                        else {
                            comparison_arms.clear();
                            break;
                        };
                        comparison_arms.push(hi_eval::DifferentialArm {
                            name: arm.name.clone(),
                            attempt: hi_eval::EvalAttempt {
                                task: task.id.clone(),
                                arm: arm.name.clone(),
                                trial,
                                identity_digest: expected.digest.clone(),
                            },
                            score: record.score,
                            evidence: record
                                .evidence
                                .map(|evidence| evidence.artifacts)
                                .unwrap_or_default(),
                        });
                    }
                    if !comparison_arms.is_empty() {
                        state.write_comparison(
                            &profile_name,
                            &hi_eval::DifferentialComparison {
                                schema_version: hi_eval::PLATFORM_SCHEMA_VERSION,
                                task: task.id.clone(),
                                trial,
                                identity_digest: expected.digest.clone(),
                                arms: comparison_arms,
                            },
                        )?;
                    }
                }
            }
            if stopped {
                break;
            }
        }
        if stopped {
            break;
        }
    }
    let final_status = if stopped {
        RunStatus::Stopped
    } else if failures.is_empty() {
        RunStatus::Completed
    } else {
        RunStatus::Failed
    };
    state.write_run(&RunRecord {
        schema_version: hi_eval::PLATFORM_SCHEMA_VERSION,
        profile: profile_name.clone(),
        identity: expected,
        status: final_status,
        started_at_unix: state
            .read_run(&profile_name)?
            .map(|run| run.started_at_unix)
            .unwrap_or(now),
        updated_at_unix: now_unix(),
    })?;
    if stopped {
        bail!("evaluation profile {profile_name} stopped; resume with `hi eval run`")
    } else if failures.is_empty() {
        println!(
            "evaluation profile {profile_name} completed; evidence: {}",
            artifact_root.display()
        );
        Ok(())
    } else {
        bail!(
            "evaluation profile {profile_name} failed attempts: {}",
            failures.join(", ")
        )
    }
}

fn build_identity(
    profile_name: &str,
    manifest_digest: &str,
    dataset_digests: std::collections::BTreeMap<String, String>,
    profile: &EvalProfile,
) -> Result<RunIdentity> {
    let hi_digest = binary_identity_digest()?;
    let digest_value = |value: &serde_json::Value| -> Result<String> {
        Ok(blake3::hash(&serde_json::to_vec(value)?)
            .to_hex()
            .to_string())
    };
    let configuration_digest = blake3::hash(&serde_json::to_vec(profile)?)
        .to_hex()
        .to_string();
    let mcp_configuration_digest = digest_value(&serde_json::to_value(&profile.mcp_servers)?)?;
    let provider_policy_digest = digest_value(&serde_json::json!({
        "policy": profile.provider_policy,
        "sampling": profile.sampling,
        "models": profile.models,
    }))?;
    let scoring_policy_digest = digest_value(&serde_json::to_value(&profile.scoring)?)?;
    let secret_configuration_digest = profile
        .secret_configuration_digest
        .clone()
        .unwrap_or_default();
    RunIdentity::new_with_details(
        profile_name,
        manifest_digest,
        dataset_digests,
        profile.models.clone(),
        &profile.backend,
        scoring_policy_digest,
        configuration_digest,
        IdentityDetails {
            adapter_version: ADAPTER_API_VERSION.into(),
            hi_binary_digest: hi_digest,
            provider_policy_digest,
            mcp_configuration_digest,
            secret_configuration_digest,
            runtime_identity: format!(
                "{}-{}-{}",
                std::env::consts::OS,
                std::env::consts::ARCH,
                profile.backend
            ),
        },
    )
}

fn binary_identity_digest() -> Result<String> {
    let candidate = std::env::var_os("HI_BIN")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| std::env::current_exe().ok().filter(|path| path.is_file()));
    let evaluator = std::env::var_os("HI_EVAL_BIN")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|current| current.parent().map(|parent| parent.join("hi-eval")))
                .filter(|path| path.is_file())
        });
    let digest = |path: Option<PathBuf>| -> Result<Option<String>> {
        path.map(|path| Ok(blake3::hash(&fs::read(&path)?).to_hex().to_string()))
            .transpose()
    };
    let value = serde_json::json!({
        "candidate": digest(candidate)?,
        "evaluator": digest(evaluator)?,
    });
    Ok(blake3::hash(&serde_json::to_vec(&value)?)
        .to_hex()
        .to_string())
}

#[allow(clippy::too_many_arguments)]
fn execute_attempt(
    eval_binary: Option<&Path>,
    task_root: &Path,
    task: &hi_eval::ImportedTask,
    profile: &EvalProfile,
    arm: &hi_eval::DifferentialArmConfig,
    trial: u32,
    identity: &RunIdentity,
    output: &Path,
) -> Result<AttemptRecord> {
    let task_root = task_root
        .canonicalize()
        .with_context(|| format!("resolving imported task {}", task_root.display()))?;
    let output = output
        .canonicalize()
        .with_context(|| format!("resolving attempt evidence {}", output.display()))?;
    if matches!(profile.backend.as_str(), "harbor" | "docker") {
        return execute_docker_attempt(&task_root, task, profile, arm, trial, identity, &output);
    }
    let eval_binary = eval_binary.context("host evaluation requires the hi-eval binary")?;
    let started = std::time::Instant::now();
    let package = task
        .package
        .as_ref()
        .context("imported case has no task package")?;
    package.validate()?;
    let verifier = profile.verifier.as_ref().or(package.verifier.as_ref());
    if !package.artifacts.is_empty() || verifier.is_some_and(|spec| !spec.artifacts.is_empty()) {
        bail!(
            "declared artifact transfer is not available in the host evaluator; select an artifact-capable backend"
        );
    }
    if matches!(package.output, hi_eval::EvalOutput::Workspace)
        && !task_root.join("task.toml").is_file()
        && arm.command.is_none()
    {
        bail!(
            "workspace task {:?} is a normalized package without task.toml; the legacy host runner cannot execute it yet",
            task.id
        );
    }
    if matches!(package.output, hi_eval::EvalOutput::Workspace) && profile.verifier.is_some() {
        bail!("profile-level verifier overrides are not available for the legacy workspace runner");
    }
    let mut final_message_mode = false;
    let mut final_report_path = None;
    let candidate_task_root = output.join("candidate-task");
    if arm.command.is_some() || matches!(package.output, hi_eval::EvalOutput::Workspace) {
        copy_eval_tree(&task_root, &candidate_task_root)?;
    }
    let mut command = if let Some(command) = &arm.command {
        let shell = if cfg!(windows) { "cmd" } else { "sh" };
        let shell_arg = if cfg!(windows) { "/C" } else { "-c" };
        let mut command_process = Command::new(shell);
        command_process.arg(shell_arg).arg(command);
        command_process
    } else {
        if matches!(package.output, hi_eval::EvalOutput::FinalMessage) {
            final_message_mode = true;
            let input_path = output.join("input.json");
            fs::write(&input_path, serde_json::to_vec_pretty(&package.input)?)?;
            let report_path = output.join("hi-report.json");
            final_report_path = Some(report_path.clone());
            let hi_binary = std::env::var_os("HI_BIN")
                .map(PathBuf::from)
                .filter(|path| path.is_file())
                .or_else(|| std::env::current_exe().ok())
                .context("final-message evaluation requires the hi binary")?;
            let mut command_process = Command::new(hi_binary);
            command_process
                .arg("--eval-input")
                .arg(input_path)
                .arg("--report")
                .arg(report_path)
                .arg("--eval-output")
                .arg("final_message")
                .arg("--no-save")
                .arg("--no-verify")
                .arg("--allow-unverified")
                .arg("--quiet");
            command_process
        } else {
            let mut command_process = Command::new(eval_binary);
            command_process.arg(&candidate_task_root);
            command_process.arg(format!("--artifacts={}", output.display()));
            command_process.arg("--trials=1");
            let treatments = if profile.treatments.is_empty() {
                "baseline".to_string()
            } else {
                profile.treatments.join(",")
            };
            command_process.arg(format!("--configs={treatments}"));
            command_process
        }
    };
    let candidate_workdir = if final_message_mode {
        let candidate_root = output.join("candidate-workspace");
        if task_root.join("fixture").is_dir() {
            copy_eval_tree(&task_root.join("fixture"), &candidate_root)?;
        } else {
            fs::create_dir_all(&candidate_root)?;
        }
        candidate_root
    } else {
        candidate_task_root.clone()
    };
    command.current_dir(&candidate_workdir);
    for (key, value) in &arm.environment {
        command.env(key, value);
    }
    if let Some(model) = arm
        .model
        .as_deref()
        .or_else(|| profile.models.first().map(String::as_str))
    {
        command.env("HI_MODEL", model);
    }
    let result = command_output_with_timeout(
        &mut command,
        Duration::from_secs(hi_eval::config::DEFAULT_CANDIDATE_TIMEOUT_SECONDS),
    )
    .with_context(|| format!("running evaluation attempt for task {}", task.id))?;
    if final_message_mode {
        let _ = fs::remove_dir_all(&candidate_workdir);
    }
    fs::write(output.join("runner.stdout.log"), &result.stdout)?;
    fs::write(output.join("runner.stderr.log"), &result.stderr)?;
    let reports = collect_json_with_paths(&output)?;
    let mut scored = reports.iter().find_map(|(path, value)| {
        value
            .get("passed")
            .and_then(serde_json::Value::as_bool)
            .map(|passed| {
                let rewards = value
                    .get("rewards")
                    .and_then(serde_json::Value::as_object)
                    .map(|object| {
                        object
                            .iter()
                            .filter_map(|(name, value)| {
                                value.as_f64().map(|reward| (name.clone(), reward))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                (path.clone(), passed, rewards)
            })
    });
    let mut verifier_timed_out = false;
    if final_message_mode {
        let report_path = final_report_path
            .clone()
            .context("final-message report path was not initialized")?;
        let report = serde_json::from_slice::<serde_json::Value>(&fs::read(&report_path)?)
            .with_context(|| format!("parsing final-message report {}", report_path.display()))?;
        let final_message = report
            .get("assistant_response")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        fs::write(output.join("final_message.txt"), &final_message)?;
        let verifier_passed = if let Some(verifier) = verifier {
            let shell = if cfg!(windows) { "cmd" } else { "sh" };
            let shell_arg = if cfg!(windows) { "/C" } else { "-c" };
            let mut verifier_command = Command::new(shell);
            let verifier_workspace = output.join("verifier-workspace");
            copy_eval_tree(&task_root, &verifier_workspace)?;
            verifier_command
                .arg(shell_arg)
                .arg(&verifier.command)
                .current_dir(&verifier_workspace)
                .env("HI_EVAL_FINAL_MESSAGE", output.join("final_message.txt"))
                .env("HI_EVAL_OUTPUT", &output);
            let verifier_output = command_output_with_timeout(
                &mut verifier_command,
                Duration::from_secs(hi_eval::config::DEFAULT_CHECK_TIMEOUT_SECONDS),
            )?;
            verifier_timed_out = verifier_output.timed_out;
            fs::write(output.join("verifier.stdout.log"), &verifier_output.stdout)?;
            fs::write(output.join("verifier.stderr.log"), &verifier_output.stderr)?;
            verifier_output.success()
        } else {
            result.status.success() && !final_message.trim().is_empty()
        };
        let rewards = fs::read(output.join("score.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|value| value.get("rewards").cloned())
            .and_then(|value| value.as_object().cloned())
            .map(|object| {
                object
                    .into_iter()
                    .filter_map(|(name, value)| value.as_f64().map(|reward| (name, reward)))
                    .collect()
            })
            .unwrap_or_default();
        scored = Some((report_path, verifier_passed, rewards));
    }
    let status = if result.timed_out || verifier_timed_out {
        AttemptStatus::InfrastructureFailed
    } else if let Some((_, verifier_passed, rewards)) = &scored {
        if profile.scoring.classify(*verifier_passed, rewards) {
            AttemptStatus::Passed
        } else {
            AttemptStatus::Failed
        }
    } else {
        // No verifier score available — pass/fail is indeterminate either way.
        AttemptStatus::InfrastructureFailed
    };
    let score = scored.map(|(_, verifier_passed, rewards)| {
        EvalScore::from_rewards(&profile.scoring, verifier_passed, rewards)
    });
    let report = if final_message_mode {
        final_report_path.as_deref().map(evidence_reference)
    } else {
        reports.first().map(|(path, _)| evidence_reference(path))
    };
    let verifier_log_path = if final_message_mode {
        output.join("verifier.stderr.log")
    } else {
        output.join("runner.stderr.log")
    };
    Ok(AttemptRecord {
        profile: identity.profile.clone(),
        task: task.id.clone(),
        arm: arm.name.clone(),
        trial,
        status,
        identity_digest: identity.digest.clone(),
        claim_level: profile.claim_level,
        score,
        evidence: Some(EvalEvidence {
            report,
            verifier_log: Some(evidence_reference(&verifier_log_path)),
            artifacts: reports
                .into_iter()
                .map(|(path, _)| evidence_reference(&path))
                .collect(),
            claim_level: profile.claim_level,
            source_digest: Some(package.source.digest.clone()),
            task_digest: Some(task.digest.clone()),
            backend: Some(profile.backend.clone()),
            runtime: Some(format!(
                "{}-{}",
                std::env::consts::OS,
                std::env::consts::ARCH
            )),
            preparation_seconds: None,
            verifier_seconds: Some(started.elapsed().as_secs_f64()),
            transcript_messages: match &package.input {
                hi_eval::EvalInput::Prompt { .. } => None,
                hi_eval::EvalInput::Transcript { messages, .. } => Some(messages.len()),
            },
            prompt_characters: match &package.input {
                hi_eval::EvalInput::Prompt { prompt } => Some(prompt.chars().count()),
                hi_eval::EvalInput::Transcript {
                    messages,
                    final_prompt,
                } => Some(
                    messages
                        .iter()
                        .map(|message| message.content.to_string().chars().count())
                        .sum::<usize>()
                        + final_prompt.as_deref().unwrap_or_default().chars().count(),
                ),
            },
            scoring_policy_digest: Some(
                blake3::hash(&serde_json::to_vec(&profile.scoring)?)
                    .to_hex()
                    .to_string(),
            ),
            input_mode: Some(
                match &package.input {
                    hi_eval::EvalInput::Prompt { .. } => "prompt",
                    hi_eval::EvalInput::Transcript { .. } => "transcript",
                }
                .into(),
            ),
            output_mode: Some(
                match package.output {
                    hi_eval::EvalOutput::Workspace => "workspace",
                    hi_eval::EvalOutput::FinalMessage => "final_message",
                }
                .into(),
            ),
            error: (result.timed_out || verifier_timed_out)
                .then(|| "evaluation attempt timed out".to_string()),
        }),
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_docker_attempt(
    task_root: &Path,
    task: &hi_eval::ImportedTask,
    profile: &EvalProfile,
    arm: &hi_eval::DifferentialArmConfig,
    trial: u32,
    identity: &RunIdentity,
    output: &Path,
) -> Result<AttemptRecord> {
    let started = std::time::Instant::now();
    let package = task
        .package
        .as_ref()
        .context("Docker evaluation case has no normalized task package")?;
    package.validate()?;
    let command = arm.command.as_deref().with_context(|| {
        format!(
            "Docker evaluation arm {:?} requires an explicit command; set arms.command to the benchmark runner or hi entrypoint inside the image",
            arm.name
        )
    })?;

    let backend = HarborDockerBackend::from_environment();
    backend.check_runtime()?;
    let image = backend.ensure_image(package, task_root)?;

    let attempt_parent = output
        .parent()
        .context("Docker attempt evidence has no parent directory")?;
    let attempt_name = output
        .file_name()
        .context("Docker attempt evidence has no directory name")?
        .to_string_lossy();
    let candidate_root = attempt_parent.join(format!(".{attempt_name}.candidate-workspace"));
    let candidate_evidence = attempt_parent.join(format!(".{attempt_name}.candidate-evidence"));
    let candidate_input = attempt_parent.join(format!(".{attempt_name}.candidate-input.json"));
    let _ = fs::remove_dir_all(&candidate_root);
    let _ = fs::remove_dir_all(&candidate_evidence);
    let _ = fs::remove_file(&candidate_input);
    fs::create_dir_all(&candidate_evidence)?;
    copy_eval_tree(task_root, &candidate_root)?;
    let mut candidate_mounts = vec![
        DockerMount::new(&candidate_root, "/workspace"),
        DockerMount::new(&candidate_evidence, "/evidence"),
    ];
    let mut environment = arm.environment.clone();
    if let Some(model) = arm
        .model
        .as_deref()
        .or_else(|| profile.models.first().map(String::as_str))
    {
        environment.insert("HI_MODEL".into(), model.into());
    }

    let final_message_mode = matches!(package.output, hi_eval::EvalOutput::FinalMessage);
    if final_message_mode {
        fs::write(&candidate_input, serde_json::to_vec_pretty(&package.input)?)?;
        candidate_mounts
            .push(DockerMount::new(&candidate_input, "/input/eval-input.json").read_only());
        environment.insert("HI_EVAL_INPUT".into(), "/input/eval-input.json".into());
        environment.insert("HI_EVAL_OUTPUT".into(), "/evidence".into());
        environment.insert(
            "HI_EVAL_FINAL_MESSAGE".into(),
            "/evidence/final_message.txt".into(),
        );
    }

    let candidate = backend.run(DockerRunSpec {
        image: image.clone(),
        name: docker_container_name(identity, task, arm, trial, "candidate"),
        workdir: "/workspace".into(),
        command: command.to_string_lossy().into_owned(),
        mounts: candidate_mounts,
        environment,
        network: profile.network.clone(),
        resources: profile.resources.clone(),
        enforce_storage: docker_storage_enforced(),
        timeout: Duration::from_secs(hi_eval::config::DEFAULT_CANDIDATE_TIMEOUT_SECONDS),
    })?;
    copy_eval_tree(&candidate_evidence, output)?;
    fs::write(
        output.join("candidate.stdout.log"),
        &candidate.output.stdout,
    )?;
    fs::write(
        output.join("candidate.stderr.log"),
        &candidate.output.stderr,
    )?;

    let final_message = if final_message_mode {
        read_or_extract_final_message(output)?
    } else {
        None
    };

    let verifier = profile.verifier.as_ref().or(package.verifier.as_ref());
    let verifier_result = if let Some(verifier) = verifier {
        let verifier_image = docker_verifier_image(&backend, package, verifier, task_root)?;
        let artifact_specs = package
            .artifacts
            .iter()
            .chain(verifier.artifacts.iter())
            .collect::<Vec<_>>();
        let artifact_root = output.join("verifier-artifacts");
        if !artifact_specs.is_empty() {
            fs::create_dir_all(&artifact_root)?;
            for artifact in artifact_specs {
                copy_declared_artifact(&candidate_root, &artifact_root, artifact)?;
            }
        }
        let mut verifier_mounts = vec![
            DockerMount::new(&candidate_root, "/workspace").read_only(),
            DockerMount::new(output, "/evidence"),
        ];
        if artifact_root.is_dir() {
            verifier_mounts.push(DockerMount::new(&artifact_root, "/artifacts").read_only());
        }
        let mut verifier_environment = std::collections::BTreeMap::new();
        verifier_environment.insert("HI_EVAL_OUTPUT".into(), "/evidence".into());
        verifier_environment.insert("HI_EVAL_ARTIFACTS".into(), "/artifacts".into());
        if final_message_mode {
            verifier_environment.insert(
                "HI_EVAL_FINAL_MESSAGE".into(),
                "/evidence/final_message.txt".into(),
            );
        }
        let result = backend.run(DockerRunSpec {
            image: verifier_image,
            name: docker_container_name(identity, task, arm, trial, "verifier"),
            workdir: "/workspace".into(),
            command: verifier.command.clone(),
            mounts: verifier_mounts,
            environment: verifier_environment,
            network: verifier.network.clone(),
            resources: profile.resources.clone(),
            enforce_storage: docker_storage_enforced(),
            timeout: Duration::from_secs(hi_eval::config::DEFAULT_CHECK_TIMEOUT_SECONDS),
        })?;
        fs::write(output.join("verifier.stdout.log"), &result.output.stdout)?;
        fs::write(output.join("verifier.stderr.log"), &result.output.stderr)?;
        Some(result)
    } else {
        None
    };
    let _ = fs::remove_dir_all(&candidate_root);
    let _ = fs::remove_dir_all(&candidate_evidence);
    let _ = fs::remove_file(&candidate_input);

    let reports = collect_json_with_paths(output)?;
    let rewards = reports
        .iter()
        .find_map(|(_, value)| value.get("rewards"))
        .and_then(serde_json::Value::as_object)
        .map(|object| {
            object
                .iter()
                .filter_map(|(name, value)| value.as_f64().map(|reward| (name.clone(), reward)))
                .collect()
        })
        .unwrap_or_default();
    let candidate_timed_out = candidate.output.timed_out;
    let verifier_timed_out = verifier_result
        .as_ref()
        .is_some_and(|result| result.output.timed_out);
    let candidate_passed = candidate.output.success();
    let verifier_passed = verifier_result
        .as_ref()
        .map(|result| candidate_passed && result.output.success())
        .unwrap_or_else(|| {
            if final_message_mode {
                candidate_passed
                    && final_message
                        .as_deref()
                        .is_some_and(|message| !message.trim().is_empty())
            } else {
                candidate_passed
            }
        });
    let status = if candidate_timed_out || verifier_timed_out || !candidate_passed {
        // A timed-out or failed candidate run yields no scoreable evidence.
        AttemptStatus::InfrastructureFailed
    } else if profile.scoring.classify(verifier_passed, &rewards) {
        AttemptStatus::Passed
    } else {
        AttemptStatus::Failed
    };
    let score = EvalScore::from_rewards(&profile.scoring, verifier_passed, rewards);
    let report = reports
        .iter()
        .find(|(_, value)| value.get("passed").is_some())
        .map(|(path, _)| evidence_reference(path))
        .or_else(|| {
            output
                .join("hi-report.json")
                .is_file()
                .then(|| evidence_reference(&output.join("hi-report.json")))
        });
    let verifier_log = if verifier_result.is_some() {
        output.join("verifier.stderr.log")
    } else {
        output.join("candidate.stderr.log")
    };
    let mut evidence_artifacts = reports
        .into_iter()
        .map(|(path, _)| evidence_reference(&path))
        .collect::<Vec<_>>();
    for path in [
        output.join("final_message.txt"),
        output.join("candidate.stdout.log"),
        output.join("candidate.stderr.log"),
        output.join("verifier.stdout.log"),
        output.join("verifier.stderr.log"),
    ] {
        if path.is_file() {
            evidence_artifacts.push(evidence_reference(&path));
        }
    }
    Ok(AttemptRecord {
        profile: identity.profile.clone(),
        task: task.id.clone(),
        arm: arm.name.clone(),
        trial,
        status,
        identity_digest: identity.digest.clone(),
        claim_level: profile.claim_level,
        score: Some(score),
        evidence: Some(EvalEvidence {
            report,
            verifier_log: Some(evidence_reference(&verifier_log)),
            artifacts: evidence_artifacts,
            claim_level: profile.claim_level,
            source_digest: Some(package.source.digest.clone()),
            task_digest: Some(task.digest.clone()),
            backend: Some(profile.backend.clone()),
            runtime: Some(format!(
                "docker:{image}:storage-{}",
                if docker_storage_enforced() {
                    "enforced"
                } else {
                    "unbounded"
                }
            )),
            preparation_seconds: None,
            verifier_seconds: Some(started.elapsed().as_secs_f64()),
            transcript_messages: match &package.input {
                EvalInput::Prompt { .. } => None,
                EvalInput::Transcript { messages, .. } => Some(messages.len()),
            },
            prompt_characters: Some(eval_input_characters(&package.input)),
            scoring_policy_digest: Some(
                blake3::hash(&serde_json::to_vec(&profile.scoring)?)
                    .to_hex()
                    .to_string(),
            ),
            input_mode: Some(
                match &package.input {
                    EvalInput::Prompt { .. } => "prompt",
                    EvalInput::Transcript { .. } => "transcript",
                }
                .into(),
            ),
            output_mode: Some(
                match package.output {
                    hi_eval::EvalOutput::Workspace => "workspace",
                    hi_eval::EvalOutput::FinalMessage => "final_message",
                }
                .into(),
            ),
            error: (candidate_timed_out || verifier_timed_out)
                .then(|| "Docker evaluation attempt timed out".into()),
        }),
    })
}

fn docker_verifier_image(
    backend: &HarborDockerBackend,
    package: &TaskPackage,
    verifier: &VerifierSpec,
    task_root: &Path,
) -> Result<String> {
    if matches!(&verifier.environment, EnvironmentSpec::Host) {
        return backend.ensure_image(package, task_root);
    }
    let mut verifier_package = package.clone();
    verifier_package.environment = verifier.environment.clone();
    backend.ensure_image(&verifier_package, task_root)
}

fn docker_container_name(
    identity: &RunIdentity,
    task: &hi_eval::ImportedTask,
    arm: &hi_eval::DifferentialArmConfig,
    trial: u32,
    role: &str,
) -> String {
    let material = format!(
        "{}:{}:{}:{}:{}",
        identity.digest, task.id, arm.name, trial, role
    );
    let digest = blake3::hash(material.as_bytes()).to_hex().to_string();
    format!("hi-eval-{}-{}", &digest[..20], role)
}

fn docker_storage_enforced() -> bool {
    matches!(
        std::env::var("HI_DOCKER_ENFORCE_STORAGE").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn read_or_extract_final_message(output: &Path) -> Result<Option<String>> {
    let direct = output.join("final_message.txt");
    if direct.is_file() {
        return Ok(Some(fs::read_to_string(&direct)?));
    }
    let report = output.join("hi-report.json");
    if !report.is_file() {
        return Ok(None);
    }
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&report)?)?;
    let response = value
        .get("assistant_response")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    if let Some(response) = &response {
        fs::write(&direct, response)?;
    }
    Ok(response)
}

fn eval_input_characters(input: &EvalInput) -> usize {
    match input {
        EvalInput::Prompt { prompt } => prompt.chars().count(),
        EvalInput::Transcript {
            messages,
            final_prompt,
        } => {
            messages
                .iter()
                .map(|message| message.content.to_string().chars().count())
                .sum::<usize>()
                + final_prompt.as_deref().unwrap_or_default().chars().count()
        }
    }
}

fn copy_declared_artifact(
    source_root: &Path,
    destination_root: &Path,
    artifact: &ArtifactSpec,
) -> Result<()> {
    let source = source_root.join(&artifact.source);
    let source = source
        .canonicalize()
        .with_context(|| format!("resolving declared artifact {}", artifact.source.display()))?;
    let source_root = source_root.canonicalize()?;
    if !source.starts_with(&source_root) {
        bail!("declared artifact escapes the candidate workspace");
    }
    let destination = destination_root.join(&artifact.source);
    copy_artifact_tree(&source, &destination, &artifact.exclude, Path::new(""))
}

fn copy_artifact_tree(
    source: &Path,
    destination: &Path,
    excludes: &[PathBuf],
    relative: &Path,
) -> Result<()> {
    if excludes
        .iter()
        .any(|excluded| relative == excluded || relative.starts_with(excluded))
    {
        return Ok(());
    }
    let metadata = fs::symlink_metadata(source)?;
    if metadata.file_type().is_symlink() {
        bail!("declared artifact contains a symlink: {}", source.display());
    }
    if metadata.is_dir() {
        fs::create_dir_all(destination)?;
        let mut entries = fs::read_dir(source)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let name = entry.file_name();
            copy_artifact_tree(
                &entry.path(),
                &destination.join(&name),
                excludes,
                &relative.join(name),
            )?;
        }
    } else if metadata.is_file() {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(source, destination)?;
    } else {
        bail!(
            "declared artifact contains an unsupported node: {}",
            source.display()
        );
    }
    Ok(())
}

fn collect_json_with_paths(root: &Path) -> Result<Vec<(PathBuf, serde_json::Value)>> {
    let mut output = Vec::new();
    collect_json_with_paths_inner(root, &mut output)?;
    Ok(output)
}

fn copy_eval_tree(source: &Path, destination: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(source)?;
    if metadata.file_type().is_symlink() {
        bail!("evaluation input contains a symlink: {}", source.display());
    }
    if metadata.is_dir() {
        fs::create_dir_all(destination)?;
        let mut entries = fs::read_dir(source)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            copy_eval_tree(&entry.path(), &destination.join(entry.file_name()))?;
        }
    } else if metadata.is_file() {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(source, destination)?;
    } else {
        bail!(
            "evaluation input contains an unsupported filesystem node: {}",
            source.display()
        );
    }
    Ok(())
}

fn rotate_attempt_output(output: &Path) -> Result<()> {
    if !output.exists() {
        return Ok(());
    }
    let Some(parent) = output.parent() else {
        bail!(
            "evaluation attempt output has no parent: {}",
            output.display()
        );
    };
    let name = output
        .file_name()
        .context("evaluation attempt output has no name")?
        .to_string_lossy();
    let retired = parent.join(format!(
        "{name}.previous-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default()
    ));
    fs::rename(output, &retired).with_context(|| {
        format!(
            "preserving previous evaluation attempt {} as {}",
            output.display(),
            retired.display()
        )
    })?;
    Ok(())
}

fn evidence_reference(path: &Path) -> PathBuf {
    let mut reference = PathBuf::new();
    let mut under_evidence = false;
    for component in path.components() {
        let component = component.as_os_str();
        if under_evidence {
            reference.push(component);
        } else if component == std::ffi::OsStr::new("evidence") {
            under_evidence = true;
            reference.push(component);
        }
    }
    if reference.as_os_str().is_empty() {
        path.file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("evidence"))
    } else {
        reference
    }
}

fn collect_json_with_paths_inner(
    root: &Path,
    output: &mut Vec<(PathBuf, serde_json::Value)>,
) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    let mut entries = fs::read_dir(root)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_json_with_paths_inner(&path, output)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
            && let Ok(value) = serde_json::from_slice(&fs::read(&path)?)
        {
            output.push((path, value));
        }
    }
    Ok(())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn status(args: &[String]) -> Result<()> {
    let profile = required_profile(args)?;
    let (_, default_state_root) = resolve_runtime_roots()?;
    let state = EvalStateStore::new(
        flag_path(args, "--state").unwrap_or_else(|| default_state_root.join("evals")),
    );
    let receipt = state.read_preparation(&profile)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "preparation": receipt,
            "run": state.read_run(&profile)?,
        }))?
    );
    Ok(())
}

fn report(args: &[String]) -> Result<()> {
    let profile = required_profile(args)?;
    let (_, default_state_root) = resolve_runtime_roots()?;
    let state = EvalStateStore::new(
        flag_path(args, "--state").unwrap_or_else(|| default_state_root.join("evals")),
    );
    let root = state.profile_root(&profile)?;
    let mut reports = Vec::new();
    collect_json(&root.join("attempts"), &mut reports)?;
    collect_json(&root.join("evidence"), &mut reports)?;
    let report = serde_json::json!({
        "schema_version": hi_eval::PLATFORM_SCHEMA_VERSION,
        "profile": profile,
        "preparation": state.read_preparation(&profile).ok(),
        "run": state.read_run(&profile)?,
        "records": reports,
    });
    let path = state.write_report(&profile, &report)?;
    println!(
        "{}\nreport: {}",
        serde_json::to_string_pretty(&report)?,
        path.display()
    );
    Ok(())
}

fn stop(args: &[String]) -> Result<()> {
    let profile = required_profile(args)?;
    let (_, default_state_root) = resolve_runtime_roots()?;
    let root = EvalStateStore::new(
        flag_path(args, "--state").unwrap_or_else(|| default_state_root.join("evals")),
    )
    .profile_root(&profile)?;
    fs::create_dir_all(&root)?;
    fs::write(root.join("stop.requested"), b"requested\n")?;
    println!("stop requested for profile {profile}");
    Ok(())
}

fn cleanup(args: &[String]) -> Result<()> {
    let profile = required_profile(args)?;
    let (_, default_state_root) = resolve_runtime_roots()?;
    let state = EvalStateStore::new(
        flag_path(args, "--state").unwrap_or_else(|| default_state_root.join("evals")),
    );
    state.cleanup_profile(&profile)?;
    println!("cleaned evaluation state for profile {profile}");
    Ok(())
}

fn required_profile(args: &[String]) -> Result<String> {
    flag_string(args, "--profile")
        .or_else(|| args.get(1).filter(|value| !value.starts_with('-')).cloned())
        .context("command requires --profile <name>")
}

fn flag_string(args: &[String], name: &str) -> Option<String> {
    let prefix = format!("{name}=");
    args.iter()
        .find_map(|arg| arg.strip_prefix(&prefix).map(str::to_string))
        .or_else(|| {
            args.iter()
                .position(|arg| arg == name)
                .and_then(|index| args.get(index + 1).cloned())
        })
}

fn flag_path(args: &[String], name: &str) -> Option<PathBuf> {
    flag_string(args, name).map(PathBuf::from)
}

fn find_eval_binary() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("HI_EVAL_BIN") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path.canonicalize().unwrap_or(path));
        }
    }
    let current = std::env::current_exe().context("resolving hi executable")?;
    if let Some(sibling) = current.parent().map(|parent| parent.join("hi-eval"))
        && sibling.is_file()
    {
        return Ok(sibling);
    }
    if let Some(path) = std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .map(|directory| directory.join("hi-eval"))
        .find(|path| path.is_file())
    {
        return Ok(path);
    }
    bail!("could not find hi-eval; set HI_EVAL_BIN to the evaluator binary")
}

fn collect_json(root: &Path, output: &mut Vec<serde_json::Value>) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    let mut entries = fs::read_dir(root)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_json(&path, output)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            let bytes = fs::read(&path)?;
            if let Ok(value) = serde_json::from_slice(&bytes) {
                output.push(value);
            }
        }
    }
    Ok(())
}

fn print_help() {
    println!(
        "hi eval — profile-driven evaluation\n\n\
         hi eval import|prepare [--manifest PATH] [--profile NAME] [--store PATH]\n\
         hi eval run --profile NAME [--manifest PATH] [--state PATH] [--store PATH] [--fresh]\n\
         hi eval status|report|stop|cleanup --profile NAME [--state PATH]\n\n\
         Sources are imported into immutable content-addressed packages.\n\
         Host profiles use HI_EVAL_BIN when the standalone hi-eval binary is\n\
         not beside hi. Docker/Harbor profiles use HI_DOCKER_BIN when Docker\n\
         is not on PATH."
    );
}

#[allow(dead_code)]
fn _claim_level_name(level: ClaimLevel) -> &'static str {
    match level {
        ClaimLevel::Official => "official",
        ClaimLevel::PublicReproduction => "public_reproduction",
        ClaimLevel::Smoke => "smoke",
        ClaimLevel::EvidenceOnly => "evidence_only",
    }
}
