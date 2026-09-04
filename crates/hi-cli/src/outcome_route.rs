//! Interactive Outcome routing: cargo mutation turns go to `POST /v1/tasks`.
//! Interactive `hi` stays a **client**. Fail open to the inner provider (local
//! chat, or RSI `/v1/rsi/runs` when `--rsi` is on) when the task plane is
//! missing. `--rsi-managed` never enters this path.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
#[cfg(test)]
use hi_agent::VerificationMode;
use hi_agent::{ReviewPolicy, RsiControl, TaskContract};
use hi_ai::{
    ChatRequest, Completion, Content, Provider, Role, ServedModel, StreamEvent, ToolCallChannel,
    Usage, estimate_text_tokens,
};
use hi_outcome::{
    OutcomeClient, OutcomeClientConfig, OutcomeError, OutcomeMode, OutcomeOffer, TaskCreateRequest,
    TaskStatus, cargo_clippy_verifier, cargo_test_verifier, json_schema_verifier, review_verifier,
};
use serde::{Deserialize, Serialize};

use crate::config::{Cli, Config, ProviderName, QualitySettings, Settings};
use crate::rsi_policy::{
    DEFAULT_BASE_URL, DEFAULT_COMPRESSED_BYTES, DEFAULT_ENTRIES, SnapshotLimits,
};
use crate::rsi_remote::{apply_exact_patch, capture_snapshot};

const DEFAULT_JSON_SCHEMA: &str = r#"{"type":"object"}"#;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RepoCache {
    blake3: String,
    repository_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LastTask {
    id: String,
    status: String,
    contract_hash: Option<String>,
    origin: String,
}

pub(crate) struct OutcomeRouteProvider {
    inner: Arc<dyn Provider>,
    inner_rsi: Option<Arc<dyn RsiControl>>,
    client: OutcomeClient,
    workspace_root: PathBuf,
    state_root: PathBuf,
    mode: OutcomeMode,
    offer: OutcomeOffer,
    quality: QualitySettings,
    turn_deadline_secs: Option<u64>,
    maximum_cost_microusd: u64,
}

pub(crate) fn resolve_outcome_mode(cli: &Cli, file: &Config) -> OutcomeMode {
    if cli.no_tasks {
        return OutcomeMode::Chat;
    }
    if cli.tasks {
        return OutcomeMode::Tasks;
    }
    file.outcome
        .as_ref()
        .and_then(|section| section.mode.as_deref())
        .map(OutcomeMode::parse)
        .unwrap_or_default()
}

pub(crate) fn should_submit_outcome(
    mode: OutcomeMode,
    user_turn: bool,
    has_cargo: bool,
    contract: &TaskContract,
) -> bool {
    if !user_turn || mode == OutcomeMode::Chat {
        return false;
    }
    if mode == OutcomeMode::Tasks {
        return true;
    }
    has_cargo && contract.is_code_change_turn()
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn wrap_outcome(
    cli: &Cli,
    file: &Config,
    settings: &Settings,
    quality: &QualitySettings,
    workspace_root: PathBuf,
    state_root: PathBuf,
    inner: Arc<dyn Provider>,
    inner_rsi: Option<Arc<dyn RsiControl>>,
    maximum_cost_microusd: u64,
) -> Result<(Arc<dyn Provider>, Option<Arc<dyn RsiControl>>)> {
    if cli.rsi_managed {
        return Ok((inner, inner_rsi));
    }
    let mode = resolve_outcome_mode(cli, file);
    if mode == OutcomeMode::Chat {
        return Ok((inner, inner_rsi));
    }
    let (origin, api_key) = outcome_credentials(file, settings);
    let client = match OutcomeClient::new(OutcomeClientConfig { origin, api_key }) {
        Ok(client) => client,
        Err(error) if error.is_fail_open() => {
            return Ok((inner, inner_rsi));
        }
        Err(error) => return Err(anyhow!(error)),
    };
    let offer = file
        .outcome
        .as_ref()
        .and_then(|section| section.offer.as_deref())
        .map(OutcomeOffer::parse)
        .unwrap_or(OutcomeOffer::Quality);
    let provider = Arc::new(OutcomeRouteProvider {
        inner,
        inner_rsi,
        client,
        workspace_root,
        state_root,
        mode,
        offer,
        quality: quality.clone(),
        turn_deadline_secs: cli.turn_deadline,
        maximum_cost_microusd,
    });
    let control = provider.clone() as Arc<dyn RsiControl>;
    Ok((provider as Arc<dyn Provider>, Some(control)))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutcomeOriginSource {
    Config,
    Environment,
    Provider,
    Default,
}

fn outcome_origin(file: &Config, settings: &Settings) -> (String, OutcomeOriginSource) {
    if let Some(url) = file
        .outcome
        .as_ref()
        .and_then(|section| section.base_url.as_deref())
        .filter(|url| !url.trim().is_empty())
    {
        return (url.to_string(), OutcomeOriginSource::Config);
    }
    if let Ok(url) = std::env::var("HI_OUTCOME_BASE_URL")
        && !url.trim().is_empty()
    {
        return (url, OutcomeOriginSource::Environment);
    }
    if settings.provider == ProviderName::Pipenetwork && !settings.base_url.trim().is_empty() {
        return (settings.base_url.clone(), OutcomeOriginSource::Provider);
    }
    (DEFAULT_BASE_URL.to_string(), OutcomeOriginSource::Default)
}

fn outcome_credentials(file: &Config, settings: &Settings) -> (String, String) {
    let (origin, source) = outcome_origin(file, settings);
    let section_key = file.outcome.as_ref().and_then(outcome_section_api_key);
    let api_key = resolve_outcome_api_key(
        source,
        &origin,
        section_key,
        settings.provider,
        &settings.base_url,
        &settings.api_key,
        std::env::var("HI_OUTCOME_API_KEY").ok(),
        std::env::var("PIPENETWORK_API_KEY").ok(),
    );
    (origin, api_key)
}

fn outcome_section_api_key(section: &crate::config::OutcomeSection) -> Option<String> {
    if let Some(reference) = section.api_key_ref.as_deref() {
        return crate::config::resolve_credential_reference(
            reference,
            section.project_local,
            section.project_local,
        )
        .ok()
        .filter(|key| !key.trim().is_empty());
    }
    let literal = section.api_key.clone().filter(|key| !key.trim().is_empty());
    if literal.is_some() {
        return literal;
    }
    if section.project_local && section.api_key_env.is_some() {
        return None;
    }
    section
        .api_key_env
        .as_deref()
        .and_then(|name| std::env::var(name).ok())
        .filter(|key| !key.trim().is_empty())
}

#[allow(clippy::too_many_arguments)]
fn resolve_outcome_api_key(
    source: OutcomeOriginSource,
    origin: &str,
    section_key: Option<String>,
    provider: ProviderName,
    provider_origin: &str,
    provider_key: &str,
    outcome_env_key: Option<String>,
    pipenetwork_env_key: Option<String>,
) -> String {
    let source_paired = match source {
        OutcomeOriginSource::Config => section_key,
        OutcomeOriginSource::Environment => outcome_env_key,
        OutcomeOriginSource::Provider => Some(provider_key.to_string()),
        OutcomeOriginSource::Default => None,
    }
    .filter(|key| !key.trim().is_empty());
    if let Some(key) = source_paired {
        return key;
    }

    // A resolved Pipe key is already authorized for its own route, so it may
    // be reused only on the same authenticated origin.
    if provider == ProviderName::Pipenetwork
        && hi_provider_config::same_endpoint_origin(origin, provider_origin)
        && !provider_key.trim().is_empty()
    {
        return provider_key.to_string();
    }
    // Provider-specific ambient credentials are confined to Pipe's official
    // service origin, regardless of who selected the Outcome URL.
    if hi_provider_config::is_official_provider_endpoint(
        hi_provider_config::ProviderName::Pipenetwork,
        origin,
    ) && let Some(key) = pipenetwork_env_key.filter(|key| !key.trim().is_empty())
    {
        return key;
    }
    String::new()
}

/// Pipe research uses the same origin/key resolution as Outcome, plus the
/// stored `pipenetwork` pairing key so a local chat model can still call
/// `POST /v1/research`.
pub(crate) fn install_research_defaults(file: &Config, settings: &Settings) {
    let (outcome_origin, outcome_key) = outcome_credentials(file, settings);
    let explicit_origin = std::env::var("PIPENETWORK_API_BASE")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let origin = explicit_origin
        .clone()
        .unwrap_or_else(|| outcome_origin.clone());
    let official_pipe = hi_provider_config::is_official_provider_endpoint(
        hi_provider_config::ProviderName::Pipenetwork,
        &origin,
    );
    let api_key = if explicit_origin.is_some() {
        std::env::var("HI_RESEARCH_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                official_pipe
                    .then(|| std::env::var("PIPENETWORK_API_KEY").ok())
                    .flatten()
                    .filter(|value| !value.trim().is_empty())
            })
            .or_else(|| {
                hi_provider_config::same_endpoint_origin(&origin, &outcome_origin)
                    .then_some(outcome_key.clone())
                    .filter(|value| !value.trim().is_empty())
            })
            .unwrap_or_default()
    } else {
        outcome_key
    };
    let api_key = if api_key.trim().is_empty() && official_pipe {
        hi_ai::auth_store::load(hi_ai::pipenetwork_auth::PROVIDER_ID)
            .map(|stored| stored.access)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_default()
    } else {
        api_key
    };
    if api_key.trim().is_empty() {
        return;
    }
    hi_research::install_process_defaults(origin, api_key);
}

fn user_prompt(request: &ChatRequest) -> String {
    request
        .canonical_objective
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            request
                .messages
                .iter()
                .rev()
                .find(|message| message.role == Role::User)
                .map(|message| message.text())
                .unwrap_or_default()
        })
}

fn fail_open_warning(error: &OutcomeError) -> String {
    format!("Outcome unavailable ({error}); falling back to local chat.")
}

impl OutcomeRouteProvider {
    fn has_cargo(&self) -> bool {
        self.workspace_root.join("Cargo.toml").exists()
    }

    fn last_task_path(&self) -> PathBuf {
        self.state_root.join("outcome-last.json")
    }

    fn repo_cache_path(&self) -> PathBuf {
        self.workspace_root.join(".hi/outcome-repo.json")
    }

    fn persist_last(&self, last: &LastTask) -> Result<()> {
        if let Some(parent) = self.last_task_path().parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(self.last_task_path(), serde_json::to_vec_pretty(last)?)?;
        Ok(())
    }

    fn load_last(&self) -> Option<LastTask> {
        let bytes = std::fs::read(self.last_task_path()).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    fn cost_usd(&self) -> f64 {
        hi_outcome::cost_usd_from_microusd(self.maximum_cost_microusd)
    }

    fn deadline_secs(&self) -> u64 {
        hi_outcome::clamp_deadline_secs(self.turn_deadline_secs)
    }

    fn verifiers(&self, contract: &TaskContract, has_cargo: bool) -> Vec<hi_outcome::TaskVerifier> {
        let mut verifiers = Vec::new();
        if has_cargo {
            verifiers.push(cargo_test_verifier());
            if self.quality.clippy || matches!(self.mode, OutcomeMode::Auto | OutcomeMode::Tasks) {
                verifiers.push(cargo_clippy_verifier());
            }
        } else {
            let schema = serde_json::from_str(DEFAULT_JSON_SCHEMA)
                .unwrap_or_else(|_| serde_json::json!({"type": "object"}));
            verifiers.push(json_schema_verifier(schema));
        }
        let include_review = match self.quality.review {
            ReviewPolicy::Always => true,
            ReviewPolicy::Risk => contract.is_code_change_turn(),
            ReviewPolicy::Off => false,
        };
        if include_review {
            verifiers.push(review_verifier());
        }
        verifiers
    }

    async fn submit_turn(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion, OutcomeError> {
        let prompt = user_prompt(&request);
        let contract = TaskContract::derive(&prompt, self.quality.verification.clone());
        let has_cargo = self.has_cargo();
        if !should_submit_outcome(self.mode, request.user_turn, has_cargo, &contract) {
            return Err(OutcomeError::fail_open(
                "turn is not an Outcome code.change",
            ));
        }
        if has_cargo && !self.client.rsi_ready().await.unwrap_or(false) {
            return Err(OutcomeError::fail_open(
                "RSI worker heartbeat is not ready for cargo-backed code.change",
            ));
        }
        sink(StreamEvent::Status("Outcome: packing workspace".into()));
        let workspace = self.workspace_root.clone();
        let snapshot = tokio::task::spawn_blocking(move || {
            capture_snapshot(
                &workspace,
                SnapshotLimits {
                    compressed_bytes: DEFAULT_COMPRESSED_BYTES,
                    entries: DEFAULT_ENTRIES,
                    uncompressed_bytes: 20 * 1024 * 1024 * 1024,
                },
            )
        })
        .await
        .map_err(|error| OutcomeError::hard(error.to_string()))?
        .map_err(|error| OutcomeError::hard(error.to_string()))?;
        let repository_id = self
            .ensure_repository(&snapshot.bytes, &snapshot.blake3)
            .await?;
        let _ = self
            .client
            .create_quote(self.cost_usd())
            .await
            .ok()
            .and_then(|quotes| {
                OutcomeClient::pick_offer(&quotes, self.offer).map(|offer| offer.route.clone())
            });
        sink(StreamEvent::Status("Outcome: creating task".into()));
        let created = self
            .client
            .create_task(&TaskCreateRequest::code_change(
                prompt,
                repository_id,
                self.verifiers(&contract, has_cargo),
                self.cost_usd(),
                self.deadline_secs(),
            ))
            .await?;
        self.persist_last(&LastTask {
            id: created.id.clone(),
            status: created.status.as_str().into(),
            contract_hash: created.contract_hash.clone(),
            origin: self.client.origin().into(),
        })
        .ok();
        self.poll_and_finish(created.id, created.status, sink).await
    }

    async fn ensure_repository(&self, gzip: &[u8], blake3: &str) -> Result<String, OutcomeError> {
        if let Ok(bytes) = std::fs::read(self.repo_cache_path())
            && let Ok(cache) = serde_json::from_slice::<RepoCache>(&bytes)
            && cache.blake3 == blake3
        {
            return Ok(cache.repository_id);
        }
        let created = self
            .client
            .upload_repository(gzip.to_vec(), blake3, blake3)
            .await?;
        if let Some(parent) = self.repo_cache_path().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(
            self.repo_cache_path(),
            serde_json::to_vec_pretty(&RepoCache {
                blake3: blake3.to_string(),
                repository_id: created.repository_id.clone(),
            })
            .unwrap_or_default(),
        );
        Ok(created.repository_id)
    }

    async fn poll_and_finish(
        &self,
        task_id: String,
        mut status: TaskStatus,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion, OutcomeError> {
        let started = Instant::now();
        let mut last_sequence = 0u64;
        loop {
            if let Some(error) =
                OutcomeClient::classify_queue_stall(status, started.elapsed().as_secs())
            {
                let _ = self.client.cancel_task(&task_id).await;
                return Err(error);
            }
            if let Ok(events) = self.client.task_events(&task_id).await {
                for event in events.events {
                    if event.sequence <= last_sequence {
                        continue;
                    }
                    last_sequence = event.sequence;
                    let label = event.stage.or(event.verifier).unwrap_or(event.event_type);
                    sink(StreamEvent::Status(format!("Outcome: {label}")));
                }
            }
            if status.is_terminal() {
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
            let task = self.client.get_task(&task_id).await?;
            status = task.status;
            self.persist_last(&LastTask {
                id: task.id.clone(),
                status: task.status.as_str().into(),
                contract_hash: task.contract_hash.clone(),
                origin: self.client.origin().into(),
            })
            .ok();
        }
        match status {
            TaskStatus::Succeeded => self.apply_success(&task_id, sink).await,
            TaskStatus::Failed | TaskStatus::BudgetExhausted | TaskStatus::DeadlineExceeded => {
                let receipt = self.client.verify_receipt(&task_id).await.ok();
                let hint = format!(
                    "Outcome task {task_id} ended ({status}). Repair with `/rsi repair` while budget remains.",
                    status = status.as_str()
                );
                sink(StreamEvent::Text(hint.clone()));
                if let Some(receipt) = receipt {
                    sink(StreamEvent::Status(format!(
                        "Outcome receipt {} verified={}",
                        receipt.contract_hash, receipt.verified
                    )));
                }
                Ok(text_completion(&hint))
            }
            TaskStatus::Canceled => Err(OutcomeError::fail_open("Outcome task was canceled")),
            _ => Err(OutcomeError::fail_open(
                "Outcome task ended without a result",
            )),
        }
    }

    async fn apply_success(
        &self,
        task_id: &str,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion, OutcomeError> {
        sink(StreamEvent::Status("Outcome: applying patch".into()));
        let patch = self.client.download_task_patch(task_id).await?;
        let root = self.workspace_root.clone();
        tokio::task::spawn_blocking(move || apply_exact_patch(&root, &patch))
            .await
            .map_err(|error| OutcomeError::hard(error.to_string()))?
            .map_err(|error| OutcomeError::hard(error.to_string()))?;
        let receipt = self.client.verify_receipt(task_id).await.ok();
        let summary = match receipt {
            Some(receipt) => format!(
                "Outcome task {task_id} succeeded. contract_hash={} verified={}",
                receipt.contract_hash, receipt.verified
            ),
            None => format!("Outcome task {task_id} succeeded."),
        };
        sink(StreamEvent::Text(summary.clone()));
        Ok(text_completion(&summary))
    }

    async fn outcome_command(&self, argument: &str) -> Result<String> {
        let mut parts = argument.split_whitespace();
        let action = parts.next().unwrap_or("status");
        match action {
            "repair" => {
                let task_id = parts
                    .next()
                    .map(str::to_string)
                    .or_else(|| self.load_last().map(|last| last.id))
                    .ok_or_else(|| anyhow!("usage: /rsi repair [task_id]"))?;
                let repair = self
                    .client
                    .create_repair(&task_id, self.cost_usd())
                    .await
                    .map_err(|error| anyhow!(error))?;
                Ok(format!(
                    "Outcome repair {} for {} ({})",
                    repair.id,
                    repair.task_id,
                    repair.status.as_str()
                ))
            }
            "status" | "cancel" | "apply" | "artifacts" | "feedback" => {
                let rest = parts.collect::<Vec<_>>().join(" ");
                self.task_command(action, &rest).await
            }
            "list" => {
                if let Some(last) = self.load_last() {
                    Ok(format!(
                        "Outcome last {} · {} · {}",
                        last.id, last.status, last.origin
                    ))
                } else if let Some(inner) = &self.inner_rsi {
                    inner.command("list").await
                } else {
                    Ok("no Outcome tasks in this workspace".into())
                }
            }
            _ => {
                if let Some(inner) = &self.inner_rsi {
                    inner.command(argument).await
                } else {
                    bail!(
                        "usage: /rsi <list|status RUN|cancel RUN|apply RUN|artifacts RUN|feedback [RUN] good|bad [reason]|repair [TASK]>"
                    )
                }
            }
        }
    }

    async fn task_command(&self, action: &str, rest: &str) -> Result<String> {
        let mut parts = rest.split_whitespace();
        let maybe_id = parts.next().unwrap_or("");
        let task_id = if maybe_id.starts_with("task_") {
            maybe_id.to_string()
        } else {
            self.load_last()
                .map(|last| last.id)
                .ok_or_else(|| anyhow!("no Outcome task id"))?
        };
        match action {
            "status" => {
                let task = self
                    .client
                    .get_task(&task_id)
                    .await
                    .map_err(|error| anyhow!(error))?;
                let receipt = self.client.verify_receipt(&task_id).await.ok();
                Ok(format!(
                    "Outcome task {}: {} · contract_hash={}",
                    task.id,
                    task.status.as_str(),
                    receipt
                        .map(|receipt| receipt.contract_hash)
                        .or(task.contract_hash)
                        .unwrap_or_else(|| "n/a".into())
                ))
            }
            "cancel" => {
                let task = self
                    .client
                    .cancel_task(&task_id)
                    .await
                    .map_err(|error| anyhow!(error))?;
                Ok(format!(
                    "Outcome task {}: {}",
                    task.id,
                    task.status.as_str()
                ))
            }
            "apply" => {
                let patch = self
                    .client
                    .download_task_patch(&task_id)
                    .await
                    .map_err(|error| anyhow!(error))?;
                let root = self.workspace_root.clone();
                tokio::task::spawn_blocking(move || apply_exact_patch(&root, &patch))
                    .await
                    .context("Outcome patch apply task failed")??;
                Ok(format!("applied Outcome task {task_id}"))
            }
            "artifacts" => {
                let destination = self.state_root.join("outcome/downloads").join(&task_id);
                std::fs::create_dir_all(&destination)?;
                let patch = self
                    .client
                    .download_task_patch(&task_id)
                    .await
                    .map_err(|error| anyhow!(error))?;
                std::fs::write(destination.join("patch"), patch)?;
                Ok(format!("downloaded patch to {}", destination.display()))
            }
            "feedback" => {
                let mut tokens = rest.split_whitespace();
                if maybe_id.starts_with("task_") {
                    tokens.next();
                }
                let outcome = tokens.next().unwrap_or("good");
                let reason = tokens.collect::<Vec<_>>().join(" ");
                self.client
                    .feedback(
                        &task_id,
                        outcome,
                        (!reason.is_empty()).then_some(reason.as_str()),
                    )
                    .await
                    .map_err(|error| anyhow!(error))?;
                Ok(format!(
                    "recorded {outcome} feedback for Outcome task {task_id}"
                ))
            }
            _ => unreachable!(),
        }
    }
}

fn text_completion(text: &str) -> Completion {
    Completion {
        content: vec![Content::Text(text.to_string())],
        usage: Usage {
            output_tokens: estimate_text_tokens(text),
            estimated: true,
            ..Usage::default()
        },
        stop_reason: Some("stop".into()),
        tool_call_channel: ToolCallChannel::None,
        ..Completion::default()
    }
}

#[async_trait]
impl Provider for OutcomeRouteProvider {
    crate::provider::forward_provider_capabilities!(self, inner);
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        if !request.user_turn {
            return self.inner.stream(request, sink).await;
        }
        match self.submit_turn(request.clone(), sink).await {
            Ok(completion) => Ok(completion),
            Err(error) if error.is_fail_open() => {
                if error.message != "turn is not an Outcome code.change" {
                    sink(StreamEvent::Warning(fail_open_warning(&error)));
                }
                self.inner.stream(request, sink).await
            }
            Err(error) => Err(anyhow!(error)),
        }
    }

    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        self.inner.list_models().await
    }
}

#[async_trait]
impl RsiControl for OutcomeRouteProvider {
    async fn validate(&self) -> Result<()> {
        if !self.client.rsi_ready().await.unwrap_or(false) {
            bail!("Outcome RSI worker is not ready");
        }
        Ok(())
    }

    async fn status(&self) -> Result<String> {
        let ready = self.client.rsi_ready().await.unwrap_or(false);
        let last = self.load_last();
        Ok(format!(
            "Outcome origin {} · rsi_ready={ready} · last {}",
            self.client.origin(),
            last.map(|last| format!("{} ({})", last.id, last.status))
                .unwrap_or_else(|| "none".into())
        ))
    }

    async fn command(&self, argument: &str) -> Result<String> {
        self.outcome_command(argument).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_default_stays_on_the_direct_unlimited_provider_route() {
        assert_eq!(OutcomeMode::default(), OutcomeMode::Chat);
        assert_eq!(OutcomeMode::parse("unknown"), OutcomeMode::Chat);
        assert_eq!(OutcomeMode::parse("auto"), OutcomeMode::Auto);
    }

    #[test]
    fn auto_routes_cargo_mutation_and_keeps_qa_on_chat() {
        let qa = TaskContract::derive("what does this parser do?", VerificationMode::Auto);
        assert!(!should_submit_outcome(OutcomeMode::Auto, true, true, &qa));
        let fix = TaskContract::derive(
            "fix the failing tests in the parser",
            VerificationMode::Auto,
        );
        assert!(should_submit_outcome(OutcomeMode::Auto, true, true, &fix));
        assert!(!should_submit_outcome(OutcomeMode::Auto, true, false, &fix));
        assert!(should_submit_outcome(OutcomeMode::Tasks, true, false, &qa));
        assert!(!should_submit_outcome(OutcomeMode::Chat, true, true, &fix));
        assert!(!should_submit_outcome(OutcomeMode::Auto, false, true, &fix));
    }

    #[test]
    fn project_outcome_endpoint_cannot_consume_ambient_or_provider_key() {
        let key = resolve_outcome_api_key(
            OutcomeOriginSource::Config,
            "https://attacker.example/v1",
            None,
            ProviderName::Pipenetwork,
            "https://api.pipenetwork.ai/v1",
            "resolved-provider-key",
            Some("ambient-outcome-key".into()),
            Some("ambient-pipe-key".into()),
        );
        assert!(key.is_empty());

        let paired = resolve_outcome_api_key(
            OutcomeOriginSource::Config,
            "https://project-outcome.example/v1",
            Some("repository-test-key".into()),
            ProviderName::Pipenetwork,
            "https://api.pipenetwork.ai/v1",
            "resolved-provider-key",
            Some("ambient-outcome-key".into()),
            Some("ambient-pipe-key".into()),
        );
        assert_eq!(paired, "repository-test-key");
    }

    #[test]
    fn outcome_reuses_provider_or_ambient_key_only_on_matching_official_origin() {
        let same_origin = resolve_outcome_api_key(
            OutcomeOriginSource::Config,
            "https://api.pipenetwork.ai/tasks",
            None,
            ProviderName::Pipenetwork,
            "https://api.pipenetwork.ai/v1",
            "resolved-provider-key",
            None,
            Some("ambient-pipe-key".into()),
        );
        assert_eq!(same_origin, "resolved-provider-key");

        let official = resolve_outcome_api_key(
            OutcomeOriginSource::Default,
            "https://api.pipenetwork.ai",
            None,
            ProviderName::Openai,
            "https://openrouter.ai/api/v1",
            "openrouter-key",
            None,
            Some("ambient-pipe-key".into()),
        );
        assert_eq!(official, "ambient-pipe-key");
    }

    #[test]
    fn project_outcome_section_cannot_read_environment_but_can_use_literal_key() {
        let blocked = crate::config::OutcomeSection {
            project_local: true,
            api_key_env: Some("PATH".into()),
            ..Default::default()
        };
        assert_eq!(outcome_section_api_key(&blocked), None);

        let literal = crate::config::OutcomeSection {
            project_local: true,
            api_key: Some("repository-test-key".into()),
            api_key_env: Some("PATH".into()),
            ..Default::default()
        };
        assert_eq!(
            outcome_section_api_key(&literal).as_deref(),
            Some("repository-test-key")
        );
    }
}
