//! Thin interactive-path RSI hooks: managed descriptor, remote switch, provider wrap.
//!
//! The interactive CLI **must not** drive `hi_agent_runtime::WorkflowExecutor` or
//! `hi_verifier::AttestingVerifier`. This module only:
//! - resolves / validates `--rsi-managed` / remote RSI request
//! - loads the expiring managed runtime descriptor
//! - starts trace observation
//! - optionally wraps the provider with budget + remote RSI
//!
//! See `docs/architecture.md` and `docs/adr/001-rsi-runtime-boundary.md`.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;
use hi_agent::{Observation, ObservationSink, RsiControl};
use hi_ai::Provider;
use hi_rsi_runtime::{EffectiveRuntime, ManagedRuntimeDescriptor, SharedBudgetLedger};

use crate::config::{Cli, Config, ProviderName, QualitySettings, RsiRequested, Settings};
use crate::report::{start_rsi_trace, unix_time_ms};
use crate::rsi_observation::{ObservedProvider, TraceObservationSink};
use crate::rsi_remote::{PersistRsiConfig, RsiRemoteProvider, RsiSettings};

/// Validated RSI mode + optional managed descriptor + optional observer.
pub(crate) struct RsiBootstrap {
    pub requested: RsiRequested,
    pub managed_runtime: Option<ManagedRuntimeDescriptor>,
    pub observer: Option<Arc<TraceObservationSink>>,
}

impl RsiBootstrap {
    /// Resolve CLI/config RSI mode, enforce interactive invariants, load managed
    /// descriptor when required, and open the managed trace writer.
    pub(crate) fn initialize(
        cli: &Cli,
        file: &Config,
        prompt_input: Option<&str>,
        launches_external_workflow: bool,
    ) -> Result<Self> {
        let requested = crate::config::resolve_rsi(cli, file)?;
        validate_managed_process_topology(requested, cli.best_of, launches_external_workflow)?;
        if requested == RsiRequested::Managed && prompt_input.is_none() {
            anyhow::bail!("managed RSI requires a noninteractive one-shot prompt");
        }
        if requested == RsiRequested::Remote {
            eprintln!(
                "\x1b[33mRSI candidate channel is enabled: this turn uploads the repository and bounded conversation context to Pipe. Operational evidence is retained 30 days; training is off without separate consent.\x1b[0m"
            );
        }
        if cli.api_unix_socket.is_some() && requested != RsiRequested::Managed {
            anyhow::bail!("--api-unix-socket is available only with --rsi-managed");
        }
        let managed_runtime = if requested == RsiRequested::Managed {
            Some(ManagedRuntimeDescriptor::read(
                cli.rsi_runtime_descriptor
                    .as_deref()
                    .expect("clap requires RSI runtime descriptor"),
                unix_time_ms()?,
            )?)
        } else {
            None
        };
        let full_capture = cli.trace_full
            || cli.trace_capture == Some(crate::config::CliTraceCapture::Full)
            || requested == RsiRequested::Managed
            || std::env::var("HI_TRACE_CAPTURE")
                .ok()
                .is_some_and(|value| value.eq_ignore_ascii_case("full"));
        let observer = start_rsi_trace(cli, requested, managed_runtime.as_ref())?.map(|writer| {
            TraceObservationSink::new(writer, requested == RsiRequested::Managed, full_capture)
        });
        if let Some(observer) = &observer {
            emit_run_started(observer, cli, requested, managed_runtime.as_ref())?;
        }
        Ok(Self {
            requested,
            managed_runtime,
            observer,
        })
    }

    pub(crate) fn is_managed(&self) -> bool {
        self.requested == RsiRequested::Managed
    }
}

/// Managed evidence and budgets are process-scoped. `--best-of` launches
/// independent `hi` processes which cannot share the worker's signed ledger or
/// trace, so accepting that combination would silently escape the descriptor.
fn validate_managed_process_topology(
    requested: RsiRequested,
    best_of: u32,
    launches_external_workflow: bool,
) -> Result<()> {
    if requested == RsiRequested::Managed && best_of > 1 {
        anyhow::bail!(
            "managed RSI does not support --best-of: candidate processes cannot share the signed runtime budget and trace"
        );
    }
    if requested == RsiRequested::Managed && launches_external_workflow {
        anyhow::bail!(
            "managed RSI does not support automatic plan workflows: child processes cannot share the signed runtime budget and trace"
        );
    }
    Ok(())
}

/// An external delegate is another independent `hi` process, so managed RSI
/// must keep delegation in-process where the signed budget ledger is shared.
pub(crate) fn external_delegate_allowed(requested: RsiRequested, is_subagent: bool) -> bool {
    requested != RsiRequested::Managed && !is_subagent
}

fn emit_run_started(
    observer: &Arc<TraceObservationSink>,
    cli: &Cli,
    requested: RsiRequested,
    managed_runtime: Option<&ManagedRuntimeDescriptor>,
) -> Result<()> {
    let mut policy = Observation::json(
        "run_started",
        "initialization",
        1,
        "turn-1",
        &serde_json::json!({
            "max_steps": cli.max_steps,
            "max_tool_calls": cli.max_tool_calls,
            "managed": requested == RsiRequested::Managed,
            "runtime_descriptor_hash": managed_runtime
                .map(ManagedRuntimeDescriptor::content_hash)
                .transpose()?,
        }),
    )?;
    policy.causation_hash =
        Some("0000000000000000000000000000000000000000000000000000000000000000".into());
    observer.observe(policy)?;
    observer.observe(Observation::json(
        "stage_entered",
        "intake",
        1,
        "turn-1",
        &serde_json::json!({"stage":"intake"}),
    )?)?;
    if let Some(runtime) = managed_runtime {
        observer.observe(Observation::json(
            "context_built",
            "initialization",
            1,
            "turn-1",
            runtime,
        )?)?;
    }
    Ok(())
}

/// Wire remote RSI (if any) and optional managed budget observation around `base`.
pub(crate) struct RsiProviderBundle {
    pub provider: Arc<dyn Provider>,
    pub rsi_control: Option<Arc<dyn RsiControl>>,
    pub rsi_remote_switch: Option<Arc<AtomicBool>>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn wrap_provider(
    cli: &Cli,
    file: &Config,
    settings: &Settings,
    quality: &QualitySettings,
    workspace_root: PathBuf,
    state_root: PathBuf,
    bootstrap: &RsiBootstrap,
    base_provider: Arc<dyn Provider>,
) -> Result<RsiProviderBundle> {
    let workspace_for_outcome = workspace_root.clone();
    let state_for_outcome = state_root.clone();
    let remote_settings = if bootstrap.is_managed() {
        None
    } else {
        let section = file.rsi.as_ref();
        let active_pipe_key = if settings.provider == ProviderName::Pipenetwork {
            settings.api_key.as_str()
        } else {
            ""
        };
        let referenced_key = section
            .and_then(|rsi| rsi.api_key_ref.as_deref())
            .map(|reference| crate::config::resolve_credential_reference(reference, false, false))
            .transpose()?;
        match RsiSettings::resolve(
            section.and_then(|rsi| rsi.base_url.as_deref()),
            referenced_key
                .as_deref()
                .or_else(|| section.and_then(|rsi| rsi.api_key.as_deref())),
            section.and_then(|rsi| rsi.api_key_env.as_deref()),
            section.and_then(|rsi| rsi.maximum_cost_microusd),
            section.and_then(|rsi| rsi.channel.as_deref()),
            active_pipe_key,
            &settings.base_url,
        ) {
            Ok(settings) => Some(settings),
            Err(error) if bootstrap.requested == RsiRequested::Remote => return Err(error),
            Err(_) => None,
        }
    };
    let rsi_remote_switch = remote_settings
        .as_ref()
        .map(|_| Arc::new(AtomicBool::new(bootstrap.requested == RsiRequested::Remote)));
    let persist_rsi_config: PersistRsiConfig = {
        let file = std::sync::Mutex::new(file.clone());
        let config_path = cli.config.clone();
        Arc::new(move |enabled, maximum_cost_microusd, channel| {
            crate::config::set_rsi_config(
                &mut file.lock().unwrap(),
                enabled,
                maximum_cost_microusd,
                channel,
                config_path.as_deref(),
            )
        })
    };
    let outcome_cost_microusd = remote_settings
        .as_ref()
        .map(|settings| settings.maximum_cost_microusd())
        .or_else(|| {
            file.rsi
                .as_ref()
                .and_then(|section| section.maximum_cost_microusd)
        })
        .unwrap_or(15_000_000);
    let remote_provider = match (remote_settings, &rsi_remote_switch) {
        (Some(remote), Some(enabled)) => Some(Arc::new(RsiRemoteProvider::new(
            base_provider.clone(),
            enabled.clone(),
            workspace_root,
            state_root,
            remote,
            persist_rsi_config,
        )?)),
        _ => None,
    };
    let rsi_control = remote_provider
        .as_ref()
        .map(|provider| provider.clone() as Arc<dyn RsiControl>);
    let base_provider: Arc<dyn Provider> = match remote_provider {
        Some(provider) => provider,
        None => base_provider,
    };
    let (base_provider, rsi_control) = crate::outcome_route::wrap_outcome(
        cli,
        file,
        settings,
        quality,
        workspace_for_outcome,
        state_for_outcome,
        base_provider,
        rsi_control,
        outcome_cost_microusd,
    )?;
    let managed_budget = bootstrap
        .managed_runtime
        .as_ref()
        .map(|runtime| SharedBudgetLedger::new(&runtime.budgets));
    let provider: Arc<dyn Provider> = match &bootstrap.observer {
        Some(observer) => Arc::new(ObservedProvider::new(
            base_provider,
            observer.clone() as Arc<dyn ObservationSink>,
            managed_budget,
            observer.full_capture(),
        )),
        None => base_provider,
    };
    Ok(RsiProviderBundle {
        provider,
        rsi_control,
        rsi_remote_switch,
    })
}

/// Bind the process's effective limits to the managed descriptor (fail-closed).
#[allow(clippy::too_many_arguments)] // mirrors every signed runtime field at the binding boundary
pub(crate) fn bind_managed_effective(
    managed: Option<&ManagedRuntimeDescriptor>,
    settings: &Settings,
    quality_max_verify_repairs: u32,
    quality_tool_set_label: &str,
    cli: &Cli,
    effective_max_steps: u32,
    effective_max_tool_calls: u32,
    max_tokens: u32,
) -> Result<()> {
    let Some(runtime) = managed else {
        return Ok(());
    };
    runtime.bind_effective(&EffectiveRuntime {
        model_role: &settings.model,
        max_model_calls: effective_max_steps,
        max_tool_calls: effective_max_tool_calls,
        max_output_tokens: max_tokens,
        max_repair_iterations: quality_max_verify_repairs,
        trace_bytes: cli.rsi_max_bytes.expect("clap requires RSI trace size"),
        tool_set: quality_tool_set_label,
        tool_mode: settings.tool_mode.label(),
    })?;
    Ok(())
}

/// Resolve the actual per-turn model-call limit once, before validating or
/// constructing the agent. Ordinary sessions are unlimited by default. A
/// managed worker remains bounded by its signed descriptor when the user did
/// not request a smaller explicit cap.
pub(crate) fn effective_max_steps(
    explicit: Option<u32>,
    managed: Option<&ManagedRuntimeDescriptor>,
) -> u32 {
    match (explicit, managed) {
        // `u32::MAX` is the agent's unlimited sentinel. A signed managed
        // descriptor must remain enforceable even when it carries that numeric
        // value, so reserve the sentinel and clamp managed work to max - 1.
        (Some(limit), Some(_)) => limit.min(u32::MAX - 1),
        (None, Some(runtime)) => runtime.budgets.model_calls.min(u32::MAX - 1),
        (Some(limit), None) => limit,
        (None, None) => hi_agent::MAX_MODEL_ROUNDS,
    }
}

/// Resolve the per-turn tool-execution cap. Ordinary sessions are unlimited;
/// managed workers inherit the signed descriptor unless the caller requests a
/// smaller explicit cap.
pub(crate) fn effective_max_tool_calls(
    explicit: Option<u32>,
    managed: Option<&ManagedRuntimeDescriptor>,
) -> u32 {
    match (explicit, managed) {
        // Managed execution must always carry a finite in-process cap. Reserve
        // u32::MAX for ordinary-session "unlimited", even when a descriptor's
        // u64 budget is larger than the agent counter can represent.
        (Some(limit), Some(_)) => limit.min(u32::MAX - 1),
        (None, Some(runtime)) => runtime.budgets.tool_calls.min(u64::from(u32::MAX - 1)) as u32,
        (Some(limit), None) => limit,
        (None, None) => hi_agent::MAX_TOOL_CALLS,
    }
}

/// Resolve productive verification repairs. Ordinary sessions use the public
/// unlimited sentinel. A managed worker inherits its signed finite budget only
/// when the operator/project did not explicitly choose a value; explicit
/// values remain visible to `bind_managed_effective`, which rejects violations.
pub(crate) fn effective_max_verify_repairs(
    configured: u32,
    explicitly_configured: bool,
    managed: Option<&ManagedRuntimeDescriptor>,
) -> u32 {
    match (explicitly_configured, managed) {
        (false, Some(runtime)) => runtime.budgets.repair_iterations.min(u32::MAX - 1),
        (true, Some(_)) => configured.min(u32::MAX - 1),
        (_, None) => configured,
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use hi_rsi_runtime::{
        CandidateIdentity, IsolationProfile, ManagedRuntimeDescriptor, MutationLevel,
        RuntimeBudgets, RuntimePolicy,
    };

    use crate::config::RsiRequested;

    use super::{
        RsiBootstrap, bind_managed_effective, effective_max_steps, effective_max_tool_calls,
        effective_max_verify_repairs, external_delegate_allowed, validate_managed_process_topology,
    };

    fn descriptor(model_calls: u32) -> ManagedRuntimeDescriptor {
        ManagedRuntimeDescriptor {
            schema_version: 1,
            protocol_major: 1,
            identity: CandidateIdentity {
                run_id: "run-1".into(),
                task_id: "task-1".into(),
                candidate_id: "candidate-1".into(),
                manifest_hash: "1".repeat(64),
                agent_artifact_hash: "2".repeat(64),
                repository_snapshot_hash: "3".repeat(64),
                source_repository: "pipe/hi".into(),
                source_commit: "abc123".into(),
            },
            budgets: RuntimeBudgets {
                wall_time_seconds: 60,
                cpu_time_seconds: 60,
                memory_bytes: 1024,
                disk_bytes: 1024,
                input_tokens: 100,
                output_tokens: 100,
                tool_calls: 10,
                cost_microusd: 100,
                model_calls,
                repair_iterations: 2,
                trace_bytes: 4096,
            },
            policy: RuntimePolicy {
                task_policy_version: "task-v1".into(),
                mutation_level: MutationLevel::Workflow,
                workflow_entrypoint: "intake".into(),
                model_role: "implementer".into(),
                tool_set: "minimal".into(),
                tool_mode: "auto".into(),
                filesystem_mode: "worktree-write".into(),
                allowed_tools: vec!["read".into(), "write".into()],
                network_allowlist: vec![],
                isolation: IsolationProfile::Namespace,
                trusted_launcher: true,
            },
            runtime_package: None,
            issued_at_unix_ms: 1_000,
            expires_at_unix_ms: 2_000,
        }
    }

    fn cli(max_steps: Option<u32>, max_tool_calls: Option<u32>) -> crate::config::Cli {
        let mut args = vec![
            "hi".to_string(),
            "--provider".to_string(),
            "openai".to_string(),
            "--model".to_string(),
            "implementer".to_string(),
            "--base-url".to_string(),
            "http://127.0.0.1:9/v1".to_string(),
            "--api-key".to_string(),
            "test-key".to_string(),
            "--rsi-max-bytes".to_string(),
            "4096".to_string(),
        ];
        if let Some(max_steps) = max_steps {
            args.push("--max-steps".to_string());
            args.push(max_steps.to_string());
        }
        if let Some(max_tool_calls) = max_tool_calls {
            args.push("--max-tool-calls".to_string());
            args.push(max_tool_calls.to_string());
        }
        crate::config::Cli::try_parse_from(args).unwrap()
    }

    fn bind(
        runtime: &ManagedRuntimeDescriptor,
        explicit_steps: Option<u32>,
        explicit_tools: Option<u32>,
    ) -> anyhow::Result<(u32, u32)> {
        let cli = cli(explicit_steps, explicit_tools);
        let settings = crate::config::resolve(&cli, &crate::config::Config::default())?;
        let max_steps = effective_max_steps(cli.max_steps, Some(runtime));
        let max_tool_calls = effective_max_tool_calls(cli.max_tool_calls, Some(runtime));
        bind_managed_effective(
            Some(runtime),
            &settings,
            2,
            "minimal",
            &cli,
            max_steps,
            max_tool_calls,
            100,
        )?;
        Ok((max_steps, max_tool_calls))
    }

    #[test]
    fn ordinary_default_is_unlimited_and_explicit_cap_wins() {
        assert_eq!(hi_agent::MAX_MODEL_ROUNDS, u32::MAX);
        assert_eq!(hi_agent::MAX_TOOL_CALLS, u32::MAX);
        assert_eq!(hi_agent::UNLIMITED_REPAIR_CYCLES, u32::MAX);
        assert_eq!(effective_max_steps(None, None), u32::MAX);
        assert_eq!(effective_max_steps(Some(7), None), 7);
        assert_eq!(effective_max_tool_calls(None, None), u32::MAX);
        assert_eq!(effective_max_tool_calls(Some(9), None), 9);
        assert_eq!(
            effective_max_verify_repairs(hi_agent::UNLIMITED_REPAIR_CYCLES, false, None),
            u32::MAX
        );
    }

    #[test]
    fn managed_default_matches_descriptor_and_explicit_cap_is_validated() {
        let runtime = descriptor(12);
        assert_eq!(
            effective_max_verify_repairs(hi_agent::UNLIMITED_REPAIR_CYCLES, false, Some(&runtime)),
            runtime.budgets.repair_iterations,
            "an ordinary managed worker must inherit its signed repair budget"
        );
        assert_eq!(
            effective_max_verify_repairs(1, true, Some(&runtime)),
            1,
            "an explicit finite repair cap must remain explicit for binding"
        );
        let inherited = effective_max_steps(None, Some(&runtime));
        assert_eq!(inherited, 12);
        assert_eq!(
            bind(&runtime, None, None).expect("the descriptor's exact budgets must bind"),
            (12, 10)
        );

        let smaller = effective_max_steps(Some(7), Some(&runtime));
        assert_eq!(smaller, 7);
        assert_eq!(
            bind(&runtime, Some(7), Some(8)).expect("explicit smaller caps must bind"),
            (7, 8)
        );

        let larger = effective_max_steps(Some(13), Some(&runtime));
        assert!(
            bind(&runtime, Some(larger), None).is_err(),
            "an explicit cap above the signed descriptor must be rejected"
        );

        let inherited_tools = effective_max_tool_calls(None, Some(&runtime));
        assert_eq!(inherited_tools, 10);
        assert!(
            bind(&runtime, None, Some(11)).is_err(),
            "an explicit tool cap above the signed descriptor must be rejected"
        );

        let sentinel_sized = descriptor(u32::MAX);
        assert_eq!(
            effective_max_steps(None, Some(&sentinel_sized)),
            u32::MAX - 1,
            "a managed budget must never turn into the ordinary unlimited sentinel"
        );
        assert_eq!(
            bind(&sentinel_sized, None, None)
                .expect("the sentinel-sized signed budget should bind")
                .0,
            u32::MAX - 1
        );
    }

    #[test]
    fn managed_workers_reject_multi_process_model_call_topologies() {
        assert!(validate_managed_process_topology(RsiRequested::Managed, 2, false).is_err());
        assert!(validate_managed_process_topology(RsiRequested::Managed, 1, false).is_ok());
        assert!(validate_managed_process_topology(RsiRequested::Off, 4, true).is_ok());

        assert!(!external_delegate_allowed(RsiRequested::Managed, false));
        assert!(!external_delegate_allowed(RsiRequested::Off, true));
        assert!(external_delegate_allowed(RsiRequested::Off, false));
    }

    #[test]
    fn managed_best_of_is_rejected_before_runtime_descriptor_io() {
        let cli = crate::config::Cli::try_parse_from([
            "hi",
            "--rsi-managed",
            "--rsi-trace-dir",
            "/definitely/missing/trace-dir",
            "--rsi-max-bytes",
            "4096",
            "--rsi-runtime-descriptor",
            "/definitely/missing/runtime-descriptor.json",
            "--best-of",
            "2",
        ])
        .expect("the combination is structurally valid and rejected by RSI bootstrap");

        let error = match RsiBootstrap::initialize(
            &cli,
            &crate::config::Config::default(),
            Some("implement the task"),
            false,
        ) {
            Ok(_) => panic!("managed workers cannot launch independent best-of processes"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("does not support --best-of"),
            "topology validation must happen before descriptor I/O: {error:#}"
        );
    }

    #[test]
    fn managed_automatic_workflow_is_rejected_before_runtime_descriptor_io() {
        let cli = crate::config::Cli::try_parse_from([
            "hi",
            "--rsi-managed",
            "--rsi-trace-dir",
            "/definitely/missing/trace-dir",
            "--rsi-max-bytes",
            "4096",
            "--rsi-runtime-descriptor",
            "/definitely/missing/runtime-descriptor.json",
            "plan.md",
        ])
        .expect("the combination is structurally valid and rejected by RSI bootstrap");

        let error = match RsiBootstrap::initialize(
            &cli,
            &crate::config::Config::default(),
            Some("plan.md"),
            true,
        ) {
            Ok(_) => panic!("managed workers cannot launch independent workflow processes"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("automatic plan workflows"),
            "topology validation must happen before descriptor I/O: {error:#}"
        );
    }
}
