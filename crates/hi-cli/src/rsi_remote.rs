//! Public RSI remote-execution client (HTTP transport + run lifecycle).
//!
//! **Trust-boundary policy** (path safety, pack/context budgets, base URL rules)
//! lives in [`crate::rsi_policy`]. This module must not loosen those checks.
//! Interactive managed RSI bootstrap is [`crate::rsi_bootstrap`].
//!
//! See `docs/adr/001-rsi-runtime-boundary.md`.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use flate2::{Compression, GzBuilder};
use hi_ai::{
    ChatRequest, Completion, Content, Provider, Role, ServedModel, StreamEvent, Usage,
    estimate_text_tokens,
};
use ignore::WalkBuilder;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use crate::rsi_policy::{
    self, DEFAULT_BASE_URL, DEFAULT_COMPRESSED_BYTES, DEFAULT_CONTEXT_BYTES, DEFAULT_ENTRIES,
    MANAGED_CONTEXT_BYTES, MAX_OBJECTIVE_BYTES, SnapshotLimits,
};
const POLL_INTERVAL: Duration = Duration::from_secs(1);
/// Default wall-clock bound for waiting on a remote RSI run. Override with
/// `HI_RSI_WAIT_TIMEOUT_SECS`.
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const MAX_POLL_INTERVAL: Duration = Duration::from_secs(10);
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn wait_timeout() -> Duration {
    std::env::var("HI_RSI_WAIT_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_WAIT_TIMEOUT)
}

#[derive(Clone, Debug)]
pub(crate) struct RsiSettings {
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    maximum_cost_microusd: Arc<AtomicU64>,
    channel: Arc<AtomicU8>,
}

impl RsiSettings {
    pub(crate) fn resolve(
        configured_base_url: Option<&str>,
        configured_maximum_cost_microusd: Option<u64>,
        configured_channel: Option<&str>,
        active_provider_key: &str,
    ) -> Result<Self> {
        let api_key = std::env::var("PIPENETWORK_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| active_provider_key.to_owned());
        ensure!(
            !api_key.trim().is_empty(),
            "RSI requires PIPENETWORK_API_KEY or an active Pipe provider key"
        );
        let base_url = configured_base_url
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(DEFAULT_BASE_URL)
            .trim_end_matches('/')
            .to_owned();
        rsi_policy::validate_rsi_base_url(&base_url)?;
        let channel = match configured_channel.unwrap_or("stable") {
            "stable" => 0,
            "beta" => 1,
            _ => bail!("RSI channel must be stable or beta"),
        };
        Ok(Self {
            base_url,
            api_key,
            maximum_cost_microusd: Arc::new(AtomicU64::new(
                configured_maximum_cost_microusd
                    .unwrap_or(15_000_000)
                    .clamp(1, 15_000_000),
            )),
            channel: Arc::new(AtomicU8::new(channel)),
        })
    }

    fn maximum_cost_microusd(&self) -> u64 {
        self.maximum_cost_microusd.load(Ordering::SeqCst)
    }

    fn channel(&self) -> &'static str {
        if self.channel.load(Ordering::SeqCst) == 1 {
            "beta"
        } else {
            "stable"
        }
    }
}

pub(crate) type PersistRsiConfig =
    Arc<dyn Fn(Option<bool>, Option<u64>, Option<String>) -> Result<()> + Send + Sync + 'static>;

#[derive(Clone)]
pub(crate) struct RsiRemoteProvider {
    inner: Arc<dyn Provider>,
    enabled: Arc<AtomicBool>,
    workspace_root: PathBuf,
    state_root: PathBuf,
    settings: RsiSettings,
    persist_config: PersistRsiConfig,
    http: Client,
}

impl RsiRemoteProvider {
    pub(crate) fn new(
        inner: Arc<dyn Provider>,
        enabled: Arc<AtomicBool>,
        workspace_root: PathBuf,
        state_root: PathBuf,
        settings: RsiSettings,
        persist_config: PersistRsiConfig,
    ) -> Result<Self> {
        Ok(Self {
            inner,
            enabled,
            workspace_root,
            state_root,
            settings,
            persist_config,
            http: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(90))
                .build()?,
        })
    }

    async fn remote_stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let capabilities = self.capabilities().await?;
        ensure!(capabilities.ready, "RSI service is not ready");
        ensure!(
            capabilities.candidate_compatible,
            "RSI candidate and worker are not compatible"
        );
        sink(StreamEvent::Status("RSI: capturing workspace".into()));
        let workspace = self.workspace_root.clone();
        let limits = SnapshotLimits {
            compressed_bytes: capabilities.maximum_compressed_upload_bytes,
            entries: capabilities.maximum_repository_entries,
            uncompressed_bytes: capabilities.maximum_disk_bytes,
        };
        let snapshot = tokio::task::spawn_blocking(move || capture_snapshot(&workspace, limits))
            .await
            .context("RSI snapshot task failed")??;
        sink(StreamEvent::Status(format!(
            "RSI: uploading {} bytes",
            snapshot.bytes.len()
        )));
        let repository = self.upload_repository(&snapshot).await?;
        let (objective, context) = bounded_context(&request, capabilities.maximum_context_bytes)?;
        let mut submission_material = blake3::Hasher::new();
        submission_material.update(snapshot.blake3.as_bytes());
        submission_material.update(objective.as_bytes());
        submission_material.update(&serde_json::to_vec(&context)?);
        let maximum_cost_microusd = self.settings.maximum_cost_microusd();
        let channel = self.settings.channel();
        submission_material.update(&maximum_cost_microusd.to_le_bytes());
        submission_material.update(channel.as_bytes());
        let submission_key = idempotency_key("run", submission_material.finalize().as_bytes());
        let created: RunView = self
            .send_json(
                self.http
                    .post(format!("{}/v1/rsi/runs", self.settings.base_url))
                    .header("idempotency-key", &submission_key)
                    .json(&RunSubmission {
                        repository_id: &repository.repository_id,
                        objective: &objective,
                        context: &context,
                        maximum_cost_microusd,
                        channel,
                    }),
            )
            .await?;
        sink(StreamEvent::Status(format!(
            "RSI: {} · channel {}",
            format_candidate(created.candidate.as_ref()),
            channel
        )));
        let pending_path = self.pending_path(&created.run_id);
        write_json_atomic(
            &pending_path,
            &PendingRun {
                schema_version: 1,
                run_id: created.run_id.clone(),
                repository_blake3: snapshot.blake3,
                objective,
                state: created.state.clone(),
            },
        )?;
        let mut cancel = CancelOnDrop::new(
            self.http.clone(),
            self.settings.clone(),
            created.run_id.clone(),
        );
        let terminal = self.wait_for_run(&created.run_id, sink).await?;
        cancel.disarm();
        if terminal.state == "canceled" {
            bail!("RSI run {} was canceled", terminal.run_id);
        }
        if terminal.state != "completed" {
            bail!(
                "RSI run {} ended as {}{}",
                terminal.run_id,
                terminal.state,
                terminal
                    .error
                    .as_deref()
                    .map(|value| format!(": {value}"))
                    .unwrap_or_default()
            );
        }
        sink(StreamEvent::Status("RSI: validating result".into()));
        if terminal.artifacts.patch {
            let bytes = self.download_artifact(&terminal.run_id, "patch").await?;
            let root = self.workspace_root.clone();
            tokio::task::spawn_blocking(move || apply_exact_patch(&root, &bytes))
                .await
                .context("RSI patch application task failed")??;
        }
        let answer = terminal
            .assistant_response
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| anyhow!("completed RSI run omitted its assistant response"))?;
        sink(StreamEvent::Text(answer.clone()));
        let _ = fs::remove_file(&pending_path);
        write_json_atomic(
            &self.summary_path(&terminal.run_id),
            &serde_json::json!({
                "schema_version": 1,
                "run_id": terminal.run_id,
                "state": terminal.state,
                "assistant_response": answer,
                "artifacts": terminal.artifacts,
                "candidate": terminal.candidate,
                "channel": channel,
            }),
        )?;
        Ok(Completion {
            content: vec![Content::Text(answer.clone())],
            usage: Usage {
                output_tokens: estimate_text_tokens(&answer),
                estimated: true,
                ..Usage::default()
            },
            stop_reason: Some("rsi_remote_completed".into()),
        })
    }

    async fn capabilities(&self) -> Result<Capabilities> {
        self.send_json(
            self.http
                .get(format!("{}/v1/rsi/capabilities", self.settings.base_url)),
        )
        .await
    }

    async fn upload_repository(&self, snapshot: &Snapshot) -> Result<RepositoryCreated> {
        let request = self
            .http
            .post(format!("{}/v1/rsi/repositories", self.settings.base_url))
            .header(
                "idempotency-key",
                idempotency_key("upload", snapshot.blake3.as_bytes()),
            )
            .header("content-type", "application/gzip")
            .header("x-content-blake3", &snapshot.blake3)
            .body(snapshot.bytes.clone());
        self.send_json(request).await
    }

    async fn wait_for_run(
        &self,
        run_id: &str,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<RunView> {
        let mut last_state = String::new();
        let mut last_event = 0_u64;
        let mut last_billing = String::new();
        let mut status_failures = 0_u8;
        let mut poll_delay = POLL_INTERVAL;
        let deadline = tokio::time::Instant::now() + wait_timeout();
        loop {
            if tokio::time::Instant::now() >= deadline {
                bail!(
                    "RSI run {run_id} did not finish within {:?}; recover it with /rsi status {run_id}",
                    wait_timeout()
                );
            }
            // Event batches are replayable, so reconnects resume from the last sequence seen.
            // Status polling remains authoritative and is also the fallback if events are
            // temporarily unavailable.
            if let Ok(events) = self
                .send_json::<RunEvents>(self.http.get(format!(
                    "{}/v1/rsi/runs/{run_id}/events",
                    self.settings.base_url
                )))
                .await
            {
                for event in events.events {
                    if event.sequence <= last_event {
                        continue;
                    }
                    last_event = event.sequence;
                    if let Some(label) = event.status_label() {
                        sink(StreamEvent::Status(format!("RSI: {label}")));
                    }
                }
            }
            let run_result: Result<RunView> = self
                .send_json(
                    self.http
                        .get(format!("{}/v1/rsi/runs/{run_id}", self.settings.base_url)),
                )
                .await;
            let run = match run_result {
                Ok(run) => {
                    status_failures = 0;
                    run
                }
                Err(_) if status_failures < 12 => {
                    status_failures += 1;
                    if status_failures == 1 {
                        sink(StreamEvent::Status(
                            "RSI: connection interrupted; reconnecting".into(),
                        ));
                    }
                    tokio::time::sleep(poll_delay).await;
                    poll_delay = (poll_delay * 2).min(MAX_POLL_INTERVAL);
                    continue;
                }
                Err(error) => {
                    return Err(error).context(format!(
                        "lost contact with RSI run {run_id}; recover it with /rsi status {run_id}"
                    ));
                }
            };
            let state_changed = run.state != last_state;
            if state_changed {
                sink(StreamEvent::Status(format!("RSI: {}", run.state)));
                last_state.clone_from(&run.state);
                poll_delay = POLL_INTERVAL;
            }
            if let Some(billing) = &run.billing {
                let label = billing.label();
                if label != last_billing {
                    sink(StreamEvent::Status(format!("RSI billing: {label}")));
                    last_billing = label;
                }
            }
            if is_terminal(&run.state) {
                return Ok(run);
            }
            if !state_changed {
                poll_delay = (poll_delay * 2).min(MAX_POLL_INTERVAL);
            }
            tokio::time::sleep(poll_delay).await;
        }
    }

    async fn download_artifact(&self, run_id: &str, kind: &str) -> Result<Vec<u8>> {
        let response = self
            .authorized(self.http.get(format!(
                "{}/v1/rsi/runs/{run_id}/artifacts/{kind}",
                self.settings.base_url
            )))
            .send()
            .await?;
        ensure_success(response)
            .await?
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(Into::into)
    }

    fn pending_path(&self, run_id: &str) -> PathBuf {
        self.state_root
            .join("rsi/pending")
            .join(format!("{run_id}.json"))
    }

    fn summary_path(&self, run_id: &str) -> PathBuf {
        self.state_root
            .join("rsi/runs")
            .join(format!("{run_id}.json"))
    }

    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        request.bearer_auth(&self.settings.api_key)
    }

    async fn send_json<T: for<'de> Deserialize<'de>>(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<T> {
        let response = ensure_success(self.authorized(request).send().await?).await?;
        response.json().await.context("decoding RSI response")
    }
}

#[async_trait]
impl Provider for RsiRemoteProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        // Auxiliary requests (compaction, memory, planning, or finalization) stay on their
        // normal route. The primary user turn is remote even in chat-only tool mode.
        if !self.enabled.load(Ordering::SeqCst) || !request.user_turn {
            return self.inner.stream(request, sink).await;
        }
        self.remote_stream(request, sink).await
    }

    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        self.inner.list_models().await
    }
}

#[derive(Clone, Debug, Deserialize)]
struct Capabilities {
    #[serde(default)]
    ready: bool,
    #[serde(default = "default_compressed_bytes")]
    maximum_compressed_upload_bytes: u64,
    #[serde(default = "default_entries")]
    maximum_repository_entries: usize,
    #[serde(default = "default_disk_bytes")]
    maximum_disk_bytes: u64,
    #[serde(default = "default_context_bytes")]
    maximum_context_bytes: usize,
    #[serde(default)]
    candidate_compatible: bool,
}

fn default_compressed_bytes() -> u64 {
    DEFAULT_COMPRESSED_BYTES
}
fn default_entries() -> usize {
    DEFAULT_ENTRIES
}
fn default_disk_bytes() -> u64 {
    20 * 1024 * 1024 * 1024
}
fn default_context_bytes() -> usize {
    DEFAULT_CONTEXT_BYTES
}

#[derive(Debug, Deserialize)]
struct RepositoryCreated {
    repository_id: String,
}

#[derive(Serialize)]
struct RunSubmission<'a> {
    repository_id: &'a str,
    objective: &'a str,
    context: &'a serde_json::Value,
    maximum_cost_microusd: u64,
    channel: &'a str,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RsiContextDocument {
    schema_version: u16,
    messages: Vec<RsiContextMessage>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RsiContextMessage {
    role: String,
    content: String,
}

pub(crate) fn load_managed_context(path: &Path) -> Result<String> {
    let before = fs::symlink_metadata(path).context("reading managed RSI context metadata")?;
    ensure!(
        before.is_file() && !before.file_type().is_symlink(),
        "managed RSI context must be a regular file"
    );
    ensure!(
        before.len() <= MANAGED_CONTEXT_BYTES as u64,
        "managed RSI context exceeds 768 KiB"
    );
    let bytes = fs::read(path).context("reading managed RSI context")?;
    let after = fs::symlink_metadata(path).context("rechecking managed RSI context")?;
    ensure!(
        bytes.len() as u64 == before.len() && same_file_version(&before, &after),
        "managed RSI context changed while reading"
    );
    let document: RsiContextDocument =
        serde_json::from_slice(&bytes).context("decoding managed RSI context")?;
    ensure!(
        document.schema_version == 1,
        "unsupported managed RSI context schema"
    );
    ensure!(
        document.messages.len() <= 1024,
        "managed RSI context contains too many messages"
    );
    for message in &document.messages {
        ensure!(
            matches!(message.role.as_str(), "user" | "assistant"),
            "managed RSI context contains an unsupported role"
        );
    }
    serde_json::to_string_pretty(&document).context("rendering managed RSI context")
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct ArtifactAvailability {
    #[serde(default)]
    patch: bool,
    #[serde(default)]
    report: bool,
    #[serde(default)]
    trace: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct RunView {
    run_id: String,
    state: String,
    #[serde(default)]
    assistant_response: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    artifacts: ArtifactAvailability,
    #[serde(default)]
    billing: Option<RunBilling>,
    #[serde(default)]
    candidate: Option<CandidateSummary>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CandidateSummary {
    candidate_id: String,
    version: String,
    mutation_class: String,
    rollout_phase: String,
}

#[derive(Debug, Deserialize)]
struct PublicStatus {
    ready: bool,
    default_channel: String,
    #[serde(default)]
    stable: Option<CandidateSummary>,
    #[serde(default)]
    beta: Option<CandidateSummary>,
    learning_loop: LearningLoopStatus,
    spend_ceiling_microusd: u64,
    evidence_policy: EvidencePolicyStatus,
    training_enabled: bool,
}

#[derive(Debug, Deserialize)]
struct LearningLoopStatus {
    healthy: bool,
    status: String,
}

#[derive(Debug, Deserialize)]
struct EvidencePolicyStatus {
    repository_and_context_uploaded: bool,
    operational_retention_days: u16,
    training_requires_separate_consent: bool,
}

#[derive(Serialize)]
struct FeedbackSubmission<'a> {
    outcome: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'a str>,
}

fn format_candidate(candidate: Option<&CandidateSummary>) -> String {
    candidate.map_or_else(
        || "candidate unavailable".into(),
        |candidate| {
            format!(
                "candidate {} v{} · {} · {}",
                candidate.candidate_id,
                candidate.version,
                candidate.mutation_class,
                candidate.rollout_phase
            )
        },
    )
}

fn format_public_status(status: &PublicStatus, selected_channel: &str) -> String {
    let selected = if selected_channel == "beta" {
        status.beta.as_ref().or(status.stable.as_ref())
    } else {
        status.stable.as_ref()
    };
    format!(
        "RSI candidate channel: {} · selected {} (default {})\nLearning loop: {} ({})\nSpend ceiling: ${:.2}/run\nEvidence: repository/context upload {}; operational retention {} days; training {}{}\n{}",
        if status.ready { "ready" } else { "not ready" },
        selected_channel,
        status.default_channel,
        status.learning_loop.status,
        if status.learning_loop.healthy {
            "healthy"
        } else {
            "fail-closed"
        },
        status.spend_ceiling_microusd as f64 / 1_000_000.0,
        if status.evidence_policy.repository_and_context_uploaded {
            "required"
        } else {
            "disabled"
        },
        status.evidence_policy.operational_retention_days,
        if status.training_enabled { "on" } else { "off" },
        if status.evidence_policy.training_requires_separate_consent {
            " (separate consent required)"
        } else {
            ""
        },
        format_candidate(selected),
    )
}

#[derive(Clone, Debug, Deserialize)]
struct RunBilling {
    reserved_microusd: u64,
    #[serde(default)]
    charged_microusd: Option<u64>,
    settlement_state: String,
}

impl RunBilling {
    fn label(&self) -> String {
        let dollars = |value: u64| value as f64 / 1_000_000.0;
        match self.charged_microusd {
            Some(charged) => format!(
                "{} ${:.6} (reserved ${:.2})",
                self.settlement_state,
                dollars(charged),
                dollars(self.reserved_microusd)
            ),
            None => format!(
                "{}; reserved up to ${:.2}",
                self.settlement_state,
                dollars(self.reserved_microusd)
            ),
        }
    }
}

#[derive(Debug, Deserialize)]
struct RunEvents {
    #[serde(default)]
    events: Vec<RunEvent>,
}

#[derive(Debug, Deserialize)]
struct RunEvent {
    sequence: u64,
    #[serde(default, rename = "type")]
    event_type: Option<String>,
    #[serde(default)]
    stage: Option<String>,
    #[serde(default)]
    state: Option<String>,
    #[serde(default)]
    passed: Option<bool>,
}

impl RunEvent {
    fn status_label(&self) -> Option<String> {
        let mut label = self
            .stage
            .as_deref()
            .or(self.state.as_deref())
            .or(self.event_type.as_deref())?
            .to_owned();
        if let Some(passed) = self.passed {
            label.push_str(if passed { " passed" } else { " failed" });
        }
        Some(label)
    }
}

#[derive(Serialize)]
struct PendingRun {
    schema_version: u16,
    run_id: String,
    repository_blake3: String,
    objective: String,
    state: String,
}

struct CancelOnDrop {
    armed: bool,
    http: Client,
    settings: RsiSettings,
    run_id: String,
}

impl CancelOnDrop {
    fn new(http: Client, settings: RsiSettings, run_id: String) -> Self {
        Self {
            armed: true,
            http,
            settings,
            run_id,
        }
    }
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let http = self.http.clone();
        let settings = self.settings.clone();
        let run_id = self.run_id.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = http
                    .post(format!("{}/v1/rsi/runs/{run_id}/cancel", settings.base_url))
                    .bearer_auth(settings.api_key)
                    .header(
                        "idempotency-key",
                        idempotency_key("cancel", run_id.as_bytes()),
                    )
                    .send()
                    .await;
            });
        }
    }
}

async fn ensure_success(response: reqwest::Response) -> Result<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    if status == StatusCode::PAYMENT_REQUIRED {
        bail!("RSI requires sufficient Pipe credits for the run reservation");
    }
    bail!(
        "RSI request failed ({status}): {}",
        body.chars().take(512).collect::<String>()
    )
}

fn is_terminal(state: &str) -> bool {
    matches!(
        state,
        "completed" | "failed" | "canceled" | "infrastructure_failed" | "budget_exhausted"
    )
}

struct Snapshot {
    bytes: Vec<u8>,
    blake3: String,
}

fn capture_snapshot(root: &Path, limits: SnapshotLimits) -> Result<Snapshot> {
    let root = root
        .canonicalize()
        .context("canonicalizing RSI workspace")?;
    let mut paths = Vec::new();
    let mut walk = WalkBuilder::new(&root);
    walk.hidden(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .follow_links(false);
    for entry in walk.build() {
        let entry = entry.context("walking RSI workspace")?;
        let path = entry.path();
        if path == root {
            continue;
        }
        let relative = path
            .strip_prefix(&root)
            .context("workspace path escaped root")?;
        validate_relative_path(relative)?;
        if rsi_policy::is_reserved_workspace_root(relative) {
            continue;
        }
        paths.push(relative.to_path_buf());
        ensure!(
            paths.len() <= limits.entries,
            "workspace exceeds RSI entry limit"
        );
    }
    paths.sort();
    let encoder = GzBuilder::new()
        .mtime(0)
        .write(Vec::new(), Compression::default());
    let mut archive = tar::Builder::new(encoder);
    archive.mode(tar::HeaderMode::Deterministic);
    let mut observed = 0u64;
    for relative in paths {
        let absolute = root.join(&relative);
        let before = fs::symlink_metadata(&absolute)?;
        ensure!(
            !before.file_type().is_symlink(),
            "snapshot refuses links: {}",
            relative.display()
        );
        let mut header = tar::Header::new_gnu();
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(0);
        if before.is_dir() {
            header.set_entry_type(tar::EntryType::Directory);
            header.set_size(0);
            header.set_mode(0o700);
            header.set_cksum();
            archive.append_data(&mut header, &relative, std::io::empty())?;
        } else if before.is_file() {
            observed = observed
                .checked_add(before.len())
                .ok_or_else(|| anyhow!("snapshot size overflow"))?;
            ensure!(
                observed <= limits.uncompressed_bytes,
                "workspace exceeds RSI disk limit"
            );
            let mut file = File::open(&absolute)?;
            let mut bytes = Vec::with_capacity(usize::try_from(before.len()).unwrap_or(0));
            Read::by_ref(&mut file)
                .take(before.len() + 1)
                .read_to_end(&mut bytes)?;
            let after = file.metadata()?;
            ensure!(
                bytes.len() as u64 == before.len() && same_file_version(&before, &after),
                "file changed during RSI capture: {}",
                relative.display()
            );
            header.set_entry_type(tar::EntryType::Regular);
            header.set_size(bytes.len() as u64);
            header.set_mode(if is_executable(&before) { 0o700 } else { 0o600 });
            header.set_cksum();
            archive.append_data(&mut header, &relative, bytes.as_slice())?;
        } else {
            bail!("snapshot refuses special entry: {}", relative.display());
        }
    }
    let encoder = archive.into_inner()?;
    let bytes = encoder.finish()?;
    ensure!(
        bytes.len() as u64 <= limits.compressed_bytes,
        "snapshot exceeds RSI upload limit"
    );
    Ok(Snapshot {
        blake3: blake3::hash(&bytes).to_hex().to_string(),
        bytes,
    })
}

#[cfg(unix)]
fn is_executable(metadata: &fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    metadata.permissions().mode() & 0o100 != 0
}
#[cfg(not(unix))]
fn is_executable(_: &fs::Metadata) -> bool {
    false
}

#[cfg(unix)]
fn same_file_version(a: &fs::Metadata, b: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    a.dev() == b.dev()
        && a.ino() == b.ino()
        && a.len() == b.len()
        && a.mtime() == b.mtime()
        && a.mtime_nsec() == b.mtime_nsec()
}
#[cfg(not(unix))]
fn same_file_version(a: &fs::Metadata, b: &fs::Metadata) -> bool {
    a.len() == b.len() && a.modified().ok() == b.modified().ok()
}

fn bounded_context(
    request: &ChatRequest,
    maximum_bytes: usize,
) -> Result<(String, serde_json::Value)> {
    let objective = request
        .canonical_objective
        .clone()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("RSI turn has no user objective"))?;
    ensure!(
        objective.len() <= MAX_OBJECTIVE_BYTES,
        "RSI objective exceeds 64 KiB"
    );
    let active_user = request
        .messages
        .iter()
        .rposition(|message| message.role == Role::User)
        .ok_or_else(|| anyhow!("RSI turn has no active user message"))?;
    let mut kept: Vec<RsiContextMessage> = Vec::new();
    let budget = maximum_bytes.min(DEFAULT_CONTEXT_BYTES);
    let mut used = serde_json::to_vec(&RsiContextDocument {
        schema_version: 1,
        messages: Vec::new(),
    })?
    .len();
    for message in request.messages[..active_user].iter().rev() {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System | Role::Tool => continue,
        };
        let content = strip_prompt_shaping(&message.text()).trim().to_owned();
        if content.is_empty() {
            continue;
        }
        let value = RsiContextMessage {
            role: role.into(),
            content,
        };
        let size = serde_json::to_vec(&value)?.len();
        if used.saturating_add(size).saturating_add(1) > budget {
            continue;
        }
        used = used.saturating_add(size).saturating_add(1);
        kept.push(value);
    }
    kept.reverse();
    let context = RsiContextDocument {
        schema_version: 1,
        messages: kept,
    };
    Ok((objective, serde_json::to_value(context)?))
}

fn strip_prompt_shaping(value: &str) -> &str {
    for marker in ["\n\nRead-only review guard:", "\n\nImplementation guard:"] {
        if let Some((canonical, _)) = value.split_once(marker) {
            return canonical;
        }
    }
    value
}

#[derive(Deserialize)]
struct PatchDocument {
    schema_version: u16,
    files: Vec<PatchFile>,
}

#[derive(Deserialize)]
struct PatchFile {
    path: String,
    kind: String,
    before_blake3: Option<String>,
    after_blake3: Option<String>,
    before_mode: Option<u32>,
    after_mode: Option<u32>,
    after_encoding: Option<String>,
    after_content: Option<String>,
}

enum Desired {
    Absent,
    Directory(u32),
    File(Vec<u8>, u32),
}
struct ValidatedChange {
    path: PathBuf,
    desired: Desired,
}

fn apply_exact_patch(root: &Path, bytes: &[u8]) -> Result<()> {
    let document: PatchDocument =
        serde_json::from_slice(bytes).context("decoding exact RSI patch")?;
    ensure!(document.schema_version == 1, "unsupported RSI patch schema");
    let root = root.canonicalize()?;
    let mut changes = Vec::new();
    for file in document.files {
        let relative = PathBuf::from(&file.path);
        validate_relative_path(&relative)?;
        ensure!(
            !rsi_policy::is_reserved_workspace_root(&relative),
            "RSI patch targets protected state"
        );
        let target = root.join(&relative);
        validate_baseline(&target, &file)?;
        let desired = desired_entry(&file)?;
        changes.push(ValidatedChange {
            path: target,
            desired,
        });
    }
    let journal = capture_journal(&changes)?;
    if let Err(error) = apply_changes(&changes) {
        let rollback = restore_journal(&journal);
        return match rollback {
            Ok(()) => Err(error.context("RSI patch rolled back")),
            Err(rollback) => Err(error.context(format!(
                "RSI patch failed and rollback failed: {rollback:#}"
            ))),
        };
    }
    Ok(())
}

fn validate_baseline(target: &Path, file: &PatchFile) -> Result<()> {
    let current = fs::symlink_metadata(target);
    if let Ok(metadata) = &current {
        ensure!(
            !metadata.file_type().is_symlink(),
            "RSI patch refuses links"
        );
    }
    match file.kind.as_str() {
        "added" => ensure!(
            current.is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound),
            "RSI apply conflict at {}",
            file.path
        ),
        "modified" | "deleted" | "type_changed" => {
            let metadata =
                current.with_context(|| format!("RSI apply conflict at {}", file.path))?;
            if let Some(hash) = &file.before_blake3 {
                ensure!(
                    metadata.is_file(),
                    "RSI apply type conflict at {}",
                    file.path
                );
                ensure!(
                    hash_regular(target)? == *hash,
                    "RSI apply content conflict at {}",
                    file.path
                );
            } else {
                ensure!(
                    metadata.is_dir(),
                    "RSI apply type conflict at {}",
                    file.path
                );
            }
            if let Some(mode) = file.before_mode {
                ensure!(
                    safe_mode(&metadata) == mode,
                    "RSI apply mode conflict at {}",
                    file.path
                );
            }
        }
        other => bail!("unsupported RSI patch operation {other}"),
    }
    Ok(())
}

fn desired_entry(file: &PatchFile) -> Result<Desired> {
    if file.after_mode.is_none() {
        ensure!(
            file.after_blake3.is_none() && file.after_content.is_none(),
            "invalid deletion patch"
        );
        return Ok(Desired::Absent);
    }
    let mode = file.after_mode.unwrap();
    ensure!(matches!(mode, 0o600 | 0o700), "unsafe RSI result mode");
    match (&file.after_encoding, &file.after_content) {
        (None, None) => {
            ensure!(file.after_blake3.is_none(), "directory has a content hash");
            Ok(Desired::Directory(mode))
        }
        (Some(encoding), Some(content)) => {
            let bytes = match encoding.as_str() {
                "utf-8" => content.as_bytes().to_vec(),
                "base64" => BASE64
                    .decode(content)
                    .context("decoding RSI patch content")?,
                _ => bail!("unsupported RSI patch encoding"),
            };
            let hash = file
                .after_blake3
                .as_deref()
                .ok_or_else(|| anyhow!("file patch omitted its hash"))?;
            ensure!(
                blake3::hash(&bytes).to_hex().as_str() == hash,
                "RSI patch content hash mismatch"
            );
            Ok(Desired::File(bytes, mode))
        }
        _ => bail!("incomplete RSI patch content"),
    }
}

enum JournalEntry {
    Missing(PathBuf),
    Directory(PathBuf, u32),
    File(PathBuf, Vec<u8>, u32),
}

fn capture_journal(changes: &[ValidatedChange]) -> Result<Vec<JournalEntry>> {
    changes
        .iter()
        .map(|change| match fs::symlink_metadata(&change.path) {
            Ok(metadata) if metadata.is_dir() => Ok(JournalEntry::Directory(
                change.path.clone(),
                safe_mode(&metadata),
            )),
            Ok(metadata) if metadata.is_file() => Ok(JournalEntry::File(
                change.path.clone(),
                fs::read(&change.path)?,
                safe_mode(&metadata),
            )),
            Ok(_) => bail!("RSI apply encountered a special entry"),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(JournalEntry::Missing(change.path.clone()))
            }
            Err(error) => Err(error.into()),
        })
        .collect()
}

fn apply_changes(changes: &[ValidatedChange]) -> Result<()> {
    let mut ordered = changes.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| match (&left.desired, &right.desired) {
        (Desired::Absent, Desired::Absent) => right
            .path
            .components()
            .count()
            .cmp(&left.path.components().count()),
        (Desired::Absent, _) => std::cmp::Ordering::Greater,
        (_, Desired::Absent) => std::cmp::Ordering::Less,
        (Desired::Directory(_), Desired::File(_, _)) => std::cmp::Ordering::Less,
        (Desired::File(_, _), Desired::Directory(_)) => std::cmp::Ordering::Greater,
        _ => left
            .path
            .components()
            .count()
            .cmp(&right.path.components().count()),
    });
    for change in ordered {
        match &change.desired {
            Desired::Directory(mode) => {
                if change.path.exists() && !change.path.is_dir() {
                    fs::remove_file(&change.path)?;
                }
                fs::create_dir_all(&change.path)?;
                set_mode(&change.path, *mode)?;
            }
            Desired::File(bytes, mode) => {
                if change.path.is_dir() {
                    fs::remove_dir_all(&change.path)?;
                }
                if let Some(parent) = change.path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let temp = temporary_sibling(&change.path);
                let result = (|| {
                    let mut out = fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&temp)?;
                    out.write_all(bytes)?;
                    out.sync_all()?;
                    set_mode(&temp, *mode)?;
                    fs::rename(&temp, &change.path)?;
                    Ok::<_, anyhow::Error>(())
                })();
                if result.is_err() {
                    let _ = fs::remove_file(&temp);
                }
                result?;
            }
            Desired::Absent => match fs::symlink_metadata(&change.path) {
                Ok(metadata) if metadata.is_dir() => fs::remove_dir(&change.path)?,
                Ok(_) => fs::remove_file(&change.path)?,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            },
        }
    }
    Ok(())
}

fn restore_journal(journal: &[JournalEntry]) -> Result<()> {
    let mut entries = journal.iter().collect::<Vec<_>>();
    entries.sort_by_key(|entry| std::cmp::Reverse(journal_path(entry).components().count()));
    for entry in &entries {
        let path = journal_path(entry);
        if let Ok(metadata) = fs::symlink_metadata(path) {
            if metadata.is_dir() {
                fs::remove_dir_all(path)?;
            } else {
                fs::remove_file(path)?;
            }
        }
    }
    entries.sort_by_key(|entry| journal_path(entry).components().count());
    for entry in entries {
        match entry {
            JournalEntry::Missing(_) => {}
            JournalEntry::Directory(path, mode) => {
                fs::create_dir_all(path)?;
                set_mode(path, *mode)?;
            }
            JournalEntry::File(path, bytes, mode) => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(path, bytes)?;
                set_mode(path, *mode)?;
            }
        }
    }
    Ok(())
}

fn journal_path(entry: &JournalEntry) -> &Path {
    match entry {
        JournalEntry::Missing(path)
        | JournalEntry::Directory(path, _)
        | JournalEntry::File(path, _, _) => path,
    }
}

fn validate_relative_path(path: &Path) -> Result<()> {
    rsi_policy::validate_relative_path(path)
}

fn hash_regular(path: &Path) -> Result<String> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "not a regular file"
    );
    Ok(blake3::hash(&fs::read(path)?).to_hex().to_string())
}

#[cfg(unix)]
fn safe_mode(metadata: &fs::Metadata) -> u32 {
    if metadata.is_dir() || is_executable(metadata) {
        0o700
    } else {
        0o600
    }
}
#[cfg(not(unix))]
fn safe_mode(metadata: &fs::Metadata) -> u32 {
    if metadata.is_dir() { 0o700 } else { 0o600 }
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    Ok(())
}
#[cfg(not(unix))]
fn set_mode(_: &Path, _: u32) -> Result<()> {
    Ok(())
}

fn temporary_sibling(path: &Path) -> PathBuf {
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    path.with_extension(format!("hi-rsi-{}-{id}.tmp", std::process::id()))
}

fn idempotency_key(scope: &str, material: &[u8]) -> String {
    format!("hi-rsi-{scope}-{}", blake3::hash(material).to_hex())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = temporary_sibling(path);
    let result = (|| {
        fs::write(&temp, serde_json::to_vec_pretty(value)?)?;
        fs::rename(&temp, path)?;
        Ok::<_, anyhow::Error>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

#[async_trait]
impl hi_agent::RsiControl for RsiRemoteProvider {
    async fn validate(&self) -> Result<()> {
        let status: PublicStatus = self
            .send_json(
                self.http
                    .get(format!("{}/v1/rsi/status", self.settings.base_url)),
            )
            .await?;
        ensure!(status.ready, "RSI candidate channel is not ready");
        Ok(())
    }

    async fn status(&self) -> Result<String> {
        let status: PublicStatus = self
            .send_json(
                self.http
                    .get(format!("{}/v1/rsi/status", self.settings.base_url)),
            )
            .await?;
        Ok(format_public_status(&status, self.settings.channel()))
    }

    async fn command(&self, argument: &str) -> Result<String> {
        let mut parts = argument.split_whitespace();
        let action = parts.next().unwrap_or("list");
        match action {
            "list" => self.local_run_list(),
            "status" => {
                let Some(run_id) = parts.next() else {
                    return self.local_run_list();
                };
                ensure!(parts.next().is_none(), "usage: /rsi status <run_id>");
                let run: RunView = self
                    .send_json(
                        self.http
                            .get(format!("{}/v1/rsi/runs/{run_id}", self.settings.base_url)),
                    )
                    .await?;
                Ok(format!(
                    "RSI run {}: {} · {} (patch {}, report {}, trace {})",
                    run.run_id,
                    run.state,
                    format_candidate(run.candidate.as_ref()),
                    run.artifacts.patch,
                    run.artifacts.report,
                    run.artifacts.trace
                ))
            }
            "cancel" => {
                let run_id = parts
                    .next()
                    .ok_or_else(|| anyhow!("usage: /rsi cancel <run_id>"))?;
                ensure!(parts.next().is_none(), "usage: /rsi cancel <run_id>");
                let run: RunView = self
                    .send_json(
                        self.http
                            .post(format!(
                                "{}/v1/rsi/runs/{run_id}/cancel",
                                self.settings.base_url
                            ))
                            .header(
                                "idempotency-key",
                                idempotency_key("cancel", run_id.as_bytes()),
                            )
                            .json(&serde_json::json!({})),
                    )
                    .await?;
                Ok(format!("RSI run {}: {}", run.run_id, run.state))
            }
            "artifacts" => {
                let run_id = parts
                    .next()
                    .ok_or_else(|| anyhow!("usage: /rsi artifacts <run_id>"))?;
                ensure!(parts.next().is_none(), "usage: /rsi artifacts <run_id>");
                let destination = self.state_root.join("rsi/downloads").join(run_id);
                fs::create_dir_all(&destination)?;
                let run: RunView = self
                    .send_json(
                        self.http
                            .get(format!("{}/v1/rsi/runs/{run_id}", self.settings.base_url)),
                    )
                    .await?;
                let mut downloaded = Vec::new();
                for (kind, available) in [
                    ("patch", run.artifacts.patch),
                    ("report", run.artifacts.report),
                    ("trace", run.artifacts.trace),
                ] {
                    if available {
                        fs::write(
                            destination.join(kind),
                            self.download_artifact(run_id, kind).await?,
                        )?;
                        downloaded.push(kind);
                    }
                }
                Ok(format!(
                    "downloaded {} to {}",
                    downloaded.join(", "),
                    destination.display()
                ))
            }
            "apply" => {
                let run_id = parts
                    .next()
                    .ok_or_else(|| anyhow!("usage: /rsi apply <run_id>"))?;
                ensure!(parts.next().is_none(), "usage: /rsi apply <run_id>");
                let patch = self.download_artifact(run_id, "patch").await?;
                let root = self.workspace_root.clone();
                tokio::task::spawn_blocking(move || apply_exact_patch(&root, &patch))
                    .await
                    .context("RSI patch application task failed")??;
                Ok(format!("applied RSI run {run_id}"))
            }
            "feedback" => {
                let values = parts.collect::<Vec<_>>();
                let (run_id, outcome, reason_start) = match values.as_slice() {
                    [outcome, ..] if matches!(*outcome, "good" | "bad") => {
                        (self.latest_run_id()?, *outcome, 1)
                    }
                    [run_id, outcome, ..] if matches!(*outcome, "good" | "bad") => {
                        ((*run_id).to_string(), *outcome, 2)
                    }
                    _ => bail!("usage: /rsi feedback [RUN] good|bad [reason]"),
                };
                let reason = values[reason_start..].join(" ");
                ensure!(reason.len() <= 2_000, "RSI feedback reason is too long");
                let material = format!("{run_id}\0{outcome}\0{reason}");
                let _: serde_json::Value = self
                    .send_json(
                        self.http
                            .post(format!(
                                "{}/v1/rsi/runs/{run_id}/feedback",
                                self.settings.base_url
                            ))
                            .header(
                                "idempotency-key",
                                idempotency_key("feedback", material.as_bytes()),
                            )
                            .json(&FeedbackSubmission {
                                outcome,
                                reason: (!reason.is_empty()).then_some(reason.as_str()),
                            }),
                    )
                    .await?;
                Ok(format!("recorded {outcome} feedback for RSI run {run_id}"))
            }
            _ => bail!(
                "usage: /rsi <list|status RUN|cancel RUN|apply RUN|artifacts RUN|feedback [RUN] good|bad [reason]>"
            ),
        }
    }

    fn maximum_cost_microusd(&self) -> u64 {
        self.settings.maximum_cost_microusd()
    }

    fn set_maximum_cost_microusd(&self, value: u64) -> Result<()> {
        ensure!(
            (1..=15_000_000).contains(&value),
            "RSI spend limit must be greater than $0 and no more than $15"
        );
        (self.persist_config)(None, Some(value), None)?;
        self.settings
            .maximum_cost_microusd
            .store(value, Ordering::SeqCst);
        Ok(())
    }

    fn persist_enabled(&self, enabled: bool) -> Result<()> {
        (self.persist_config)(Some(enabled), None, None)
    }

    fn channel(&self) -> &'static str {
        self.settings.channel()
    }

    fn set_channel(&self, channel: &str) -> Result<()> {
        let value = match channel {
            "stable" => 0,
            "beta" => 1,
            _ => bail!("RSI channel must be stable or beta"),
        };
        (self.persist_config)(None, None, Some(channel.to_string()))?;
        self.settings.channel.store(value, Ordering::SeqCst);
        Ok(())
    }
}

impl RsiRemoteProvider {
    fn latest_run_id(&self) -> Result<String> {
        let mut latest: Option<(std::time::SystemTime, String)> = None;
        for directory in [
            self.state_root.join("rsi/pending"),
            self.state_root.join("rsi/runs"),
        ] {
            let Ok(entries) = fs::read_dir(directory) else {
                continue;
            };
            for entry in entries {
                let entry = entry?;
                if !entry.file_type()?.is_file() {
                    continue;
                }
                let modified = entry
                    .metadata()?
                    .modified()
                    .unwrap_or(std::time::UNIX_EPOCH);
                let value: serde_json::Value = serde_json::from_slice(&fs::read(entry.path())?)?;
                let Some(run_id) = value.get("run_id").and_then(serde_json::Value::as_str) else {
                    continue;
                };
                if latest.as_ref().is_none_or(|(prior, _)| modified > *prior) {
                    latest = Some((modified, run_id.to_string()));
                }
            }
        }
        latest
            .map(|(_, run_id)| run_id)
            .ok_or_else(|| anyhow!("no local RSI run is available; specify RUN explicitly"))
    }

    fn local_run_list(&self) -> Result<String> {
        let mut rows = BTreeMap::new();
        for (directory, fallback_state) in [
            (self.state_root.join("rsi/pending"), "pending"),
            (self.state_root.join("rsi/runs"), "completed"),
        ] {
            let Ok(entries) = fs::read_dir(directory) else {
                continue;
            };
            for entry in entries {
                let entry = entry?;
                if !entry.file_type()?.is_file() {
                    continue;
                }
                let value: serde_json::Value = serde_json::from_slice(&fs::read(entry.path())?)?;
                if let Some(id) = value.get("run_id").and_then(serde_json::Value::as_str) {
                    let state = value
                        .get("state")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or(fallback_state);
                    rows.insert(id.to_owned(), state.to_owned());
                }
            }
        }
        if rows.is_empty() {
            return Ok("no local RSI run metadata".into());
        }
        Ok(rows
            .into_iter()
            .map(|(id, state)| format!("{id}  {state}"))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hi-rsi-{name}-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn snapshots_are_deterministic_and_exclude_state() {
        let root = temp_dir("snapshot");
        fs::write(root.join("b.txt"), "b").unwrap();
        fs::write(root.join("a.txt"), "a").unwrap();
        fs::create_dir(root.join(".hi")).unwrap();
        fs::write(root.join(".hi/secret"), "no").unwrap();
        let limits = SnapshotLimits {
            compressed_bytes: 1024 * 1024,
            entries: 20,
            uncompressed_bytes: 1024 * 1024,
        };
        let one = capture_snapshot(&root, limits).unwrap();
        let two = capture_snapshot(&root, limits).unwrap();
        assert_eq!(one.bytes, two.bytes);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_patch_conflict_is_non_mutating() {
        let root = temp_dir("conflict");
        fs::write(root.join("a.txt"), "local").unwrap();
        let patch = serde_json::json!({"schema_version":1,"files":[{
            "path":"a.txt","kind":"modified","before_blake3":blake3::hash(b"baseline").to_hex().to_string(),
            "after_blake3":blake3::hash(b"remote").to_hex().to_string(),"before_mode":384,"after_mode":384,
            "after_encoding":"utf-8","after_content":"remote"}]});
        assert!(apply_exact_patch(&root, &serde_json::to_vec(&patch).unwrap()).is_err());
        assert_eq!(fs::read_to_string(root.join("a.txt")).unwrap(), "local");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_patch_applies_after_full_validation() {
        let root = temp_dir("apply");
        fs::write(root.join("a.txt"), "before").unwrap();
        let patch = serde_json::json!({"schema_version":1,"files":[{
            "path":"a.txt","kind":"modified","before_blake3":blake3::hash(b"before").to_hex().to_string(),
            "after_blake3":blake3::hash(b"after").to_hex().to_string(),"before_mode":384,"after_mode":384,
            "after_encoding":"utf-8","after_content":"after"}]});
        apply_exact_patch(&root, &serde_json::to_vec(&patch).unwrap()).unwrap();
        assert_eq!(fs::read_to_string(root.join("a.txt")).unwrap(), "after");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_context_uses_canonical_objective_and_stops_before_active_turn() {
        let request = ChatRequest {
            model: "m".into(),
            user_turn: true,
            canonical_objective: Some("Inspect Cargo.toml and make no changes".into()),
            messages: Arc::new(vec![
                hi_ai::Message::system("secret system"),
                hi_ai::Message::user(
                    "prior question\n\nImplementation guard: make concrete file changes",
                ),
                hi_ai::Message::assistant(vec![Content::Text("prior answer".into())]),
                hi_ai::Message::user(
                    "Inspect Cargo.toml\n\nRead-only review guard: use inspection",
                ),
                hi_ai::Message::assistant(vec![Content::ToolCall {
                    id: "hi_preflight_1".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                }]),
                hi_ai::Message {
                    role: Role::Tool,
                    content: vec![Content::ToolResult {
                        call_id: "hi_preflight_1".into(),
                        output: "local preflight".into(),
                    }],
                },
            ]),
            tools: Arc::new([]),
            max_tokens: 10,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: hi_ai::RequestProfile::default(),
        };

        let (objective, context) = bounded_context(&request, 64 * 1024).unwrap();
        assert_eq!(objective, "Inspect Cargo.toml and make no changes");
        let encoded = serde_json::to_string(&context).unwrap();
        assert!(encoded.contains("prior question"));
        assert!(encoded.contains("prior answer"));
        assert!(!encoded.contains("Implementation guard"));
        assert!(!encoded.contains("Read-only review guard"));
        assert!(!encoded.contains("local preflight"));
        assert!(!encoded.contains("secret system"));
    }

    #[test]
    fn managed_context_rejects_bad_schema_and_oversize() {
        let root = temp_dir("managed-context");
        let path = root.join("context.json");
        fs::write(&path, br#"{"schema_version":2,"messages":[]}"#).unwrap();
        assert!(load_managed_context(&path).is_err());
        fs::write(&path, vec![b'x'; MANAGED_CONTEXT_BYTES + 1]).unwrap();
        assert!(load_managed_context(&path).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
