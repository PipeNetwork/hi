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

use crate::config::{Cli, Config, ProviderName, RsiRequested, Settings};
use crate::provider::build_chain;
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
    pub(crate) fn initialize(cli: &Cli, file: &Config, prompt_input: Option<&str>) -> Result<Self> {
        let requested = crate::config::resolve_rsi(cli, file)?;
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
        let observer = start_rsi_trace(cli, requested, managed_runtime.as_ref())?
            .map(|writer| TraceObservationSink::new(writer, requested == RsiRequested::Managed));
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

pub(crate) fn wrap_provider(
    cli: &Cli,
    file: &Config,
    settings: &Settings,
    workspace_root: PathBuf,
    state_root: PathBuf,
    bootstrap: &RsiBootstrap,
    base_provider: Arc<dyn Provider>,
) -> Result<RsiProviderBundle> {
    let remote_settings = if bootstrap.is_managed() {
        None
    } else {
        let section = file.rsi.as_ref();
        let active_pipe_key = if settings.provider == ProviderName::Pipenetwork {
            settings.api_key.as_str()
        } else {
            ""
        };
        match RsiSettings::resolve(
            section.and_then(|rsi| rsi.base_url.as_deref()),
            section.and_then(|rsi| rsi.maximum_cost_microusd),
            section.and_then(|rsi| rsi.channel.as_deref()),
            active_pipe_key,
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
        None => build_chain(settings, crate::config::resolve_fallbacks(cli, file)).into(),
    };
    let managed_budget = bootstrap
        .managed_runtime
        .as_ref()
        .map(|runtime| SharedBudgetLedger::new(&runtime.budgets));
    let provider: Arc<dyn Provider> = match &bootstrap.observer {
        Some(observer) => Arc::new(ObservedProvider::new(
            base_provider,
            observer.clone() as Arc<dyn ObservationSink>,
            managed_budget,
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
pub(crate) fn bind_managed_effective(
    managed: Option<&ManagedRuntimeDescriptor>,
    settings: &Settings,
    quality_max_verify_repairs: u32,
    quality_tool_set_label: &str,
    cli: &Cli,
    max_tokens: u32,
) -> Result<()> {
    let Some(runtime) = managed else {
        return Ok(());
    };
    runtime.bind_effective(&EffectiveRuntime {
        model_role: &settings.model,
        max_model_calls: cli.max_steps.unwrap_or(u32::MAX),
        max_tool_calls: cli.max_tool_calls.unwrap_or(u32::MAX),
        max_output_tokens: max_tokens,
        max_repair_iterations: quality_max_verify_repairs,
        trace_bytes: cli.rsi_max_bytes.expect("clap requires RSI trace size"),
        tool_set: quality_tool_set_label,
        tool_mode: settings.tool_mode.label(),
    })?;
    Ok(())
}
