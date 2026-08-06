//! Auto-managed local model for the `/goal` skeptic review (`/config
//! skeptic-local on`).
//!
//! The skeptic gate runs a bounded critique call before a turn may advance a
//! sub-goal. Transport failures yield `Unavailable`; product policy
//! (`skeptic_fail_open`, default fail-closed) decides whether that blocks
//! advance. It fires often, so routing it to a small local model keeps the
//! coding driver and planner on the main model while making the review free and
//! private. Turning the feature on detects the machine's local-inference
//! backend (Apple-Silicon MLX or NVIDIA CUDA), fetches a small default review
//! model if it isn't already cached, launches a `hi-local` server, waits for it
//! to become healthy, and points `skeptic_endpoint`/`skeptic_model` at it. Every
//! step degrades gracefully: a missing backend, missing binary, failed
//! download, or unhealthy server leaves the skeptic on the main provider and
//! reports why.

use crate::Agent;
use anyhow::{Context, Result, bail};
use futures_util::stream::{self, StreamExt};
use hi_ai::Provider;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

/// A local-inference backend that `hi-local serve` can drive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalBackend {
    /// Apple-Silicon MLX. Serves a model *directory*.
    Mlx,
    /// NVIDIA CUDA. Serves a single GGUF *file*.
    Cuda,
}

impl LocalBackend {
    /// The `--backend` value for `hi-local serve`.
    pub fn serve_flag(self) -> &'static str {
        match self {
            LocalBackend::Mlx => "mlx",
            LocalBackend::Cuda => "cuda",
        }
    }

    /// The `hi-local` cargo feature that compiles this backend in. The
    /// backends are opt-in features (`default = []`), so a plain
    /// `cargo build -p hi-local` produces a binary that instantly rejects
    /// `--backend mlx` — the exact failure users saw as a silent dead server.
    pub fn cargo_feature(self) -> &'static str {
        match self {
            LocalBackend::Mlx => "mlx",
            LocalBackend::Cuda => "native-cuda",
        }
    }
}

/// Choose a backend from probed hardware facts. Pure so it can be unit tested;
/// the environment probe is [`detect_backend`]. MLX wins on Apple Silicon;
/// otherwise CUDA when an NVIDIA runtime is present; otherwise none.
pub fn pick_backend(is_apple_silicon: bool, has_nvidia: bool) -> Option<LocalBackend> {
    if is_apple_silicon {
        Some(LocalBackend::Mlx)
    } else if has_nvidia {
        Some(LocalBackend::Cuda)
    } else {
        None
    }
}

/// Probe the host for a usable local backend. `HI_LOCAL_BACKEND`
/// (`mlx`/`cuda`/`none`) overrides the probe for tests and debugging.
pub fn detect_backend() -> Option<LocalBackend> {
    match std::env::var("HI_LOCAL_BACKEND").ok().as_deref() {
        Some("mlx") => return Some(LocalBackend::Mlx),
        Some("cuda") => return Some(LocalBackend::Cuda),
        Some("none") => return None,
        _ => {}
    }
    let is_apple_silicon = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    let has_nvidia = !is_apple_silicon && nvidia_present();
    pick_backend(is_apple_silicon, has_nvidia)
}

/// Cached form for synchronous UI paths such as `/team` pickers. Hardware
/// probing is normally cheap, but `nvidia-smi` can take noticeable time when
/// a driver is waking up or unavailable. Do not make every role-menu action
/// pay that subprocess cost. Explicit test/debug overrides always bypass the
/// cache so changing `HI_LOCAL_BACKEND` takes effect immediately.
pub fn detect_backend_cached() -> Option<LocalBackend> {
    if std::env::var_os("HI_LOCAL_BACKEND").is_some() {
        return detect_backend();
    }
    const CACHE_TTL: Duration = Duration::from_secs(30);
    let now = Instant::now();
    if let Ok(cache) = BACKEND_CACHE.lock()
        && let Some((checked_at, backend)) = *cache
        && now.duration_since(checked_at) < CACHE_TTL
    {
        return backend;
    }
    let backend = detect_backend();
    if let Ok(mut cache) = BACKEND_CACHE.lock() {
        *cache = Some((now, backend));
    }
    backend
}

static BACKEND_CACHE: LazyLock<Mutex<Option<(Instant, Option<LocalBackend>)>>> =
    LazyLock::new(|| Mutex::new(None));

fn nvidia_present() -> bool {
    // `nvidia-smi` on PATH is the cheapest reliable signal a CUDA runtime exists.
    std::process::Command::new("nvidia-smi")
        .arg("-L")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// `detect_backend` runs a blocking `nvidia-smi` subprocess; offload it so it
/// doesn't stall the async executor when called from an async context.
pub async fn detect_backend_offload() -> Option<LocalBackend> {
    tokio::task::spawn_blocking(detect_backend_cached)
        .await
        .unwrap_or(None)
}

/// A default local review model for a backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalModelSpec {
    /// HuggingFace repo id to fetch when the weights are absent.
    pub repo: String,
    /// Model id the server advertises over the OpenAI API (the skeptic model).
    pub model_id: String,
    /// For CUDA/GGUF, the single weight filename to serve inside the repo.
    /// `None` for MLX, where the whole downloaded directory is the model.
    pub gguf_file: Option<String>,
    /// The backend this spec targets.
    pub backend: LocalBackend,
}

/// The bundled default review model for a backend: a ~3B instruct model, 4-bit
/// quantized — strong enough to catch premature "done", small enough to run
/// beside the coding model.
pub fn default_model(backend: LocalBackend) -> LocalModelSpec {
    match backend {
        LocalBackend::Mlx => LocalModelSpec {
            repo: "mlx-community/Qwen2.5-3B-Instruct-4bit".to_string(),
            model_id: "Qwen2.5-3B-Instruct-4bit".to_string(),
            gguf_file: None,
            backend,
        },
        LocalBackend::Cuda => LocalModelSpec {
            repo: "Qwen/Qwen2.5-3B-Instruct-GGUF".to_string(),
            model_id: "qwen2.5-3b-instruct".to_string(),
            gguf_file: Some("qwen2.5-3b-instruct-q4_k_m.gguf".to_string()),
            backend,
        },
    }
}

/// [`default_model`] overlaid with any `HI_SKEPTIC_LOCAL_*` env overrides, so a
/// user can point the skeptic at their own local model.
pub fn resolve_model(backend: LocalBackend) -> LocalModelSpec {
    let env = |key: &str| {
        std::env::var(key)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    let mut spec = default_model(backend);
    let id_override = env("HI_SKEPTIC_LOCAL_MODEL_ID");
    if let Some(repo) = env("HI_SKEPTIC_LOCAL_REPO") {
        // A fresh repo without an explicit id defaults to the repo's last path
        // segment, which is what MLX servers advertise for a model directory.
        if id_override.is_none() {
            spec.model_id = repo.rsplit('/').next().unwrap_or(&repo).to_string();
        }
        spec.repo = repo;
    }
    if let Some(id) = id_override {
        spec.model_id = id;
    }
    if let Some(file) = env("HI_SKEPTIC_LOCAL_GGUF") {
        spec.gguf_file = Some(file);
    }
    spec
}

/// Whether the model's weights are already cached in `dir`.
pub fn model_present(dir: &Path, spec: &LocalModelSpec) -> bool {
    match &spec.gguf_file {
        // MLX: a loadable model directory carries a config.json (matches
        // `/hf`) — and when the repo ships a safetensors index, every weight
        // shard it names. Delegate content validation to the shared HF helper;
        // existence alone would accept a saved redirect/error body.
        None => hi_tools::mlx_model_present(dir),
        // CUDA: the specific GGUF file must be on disk.
        Some(file) => dir.join(file).exists(),
    }
}

/// The path passed to `hi-local serve`: the model *directory* for MLX, the GGUF
/// *file* for CUDA.
pub fn serve_model_path(dir: &Path, spec: &LocalModelSpec) -> PathBuf {
    match &spec.gguf_file {
        None => dir.to_path_buf(),
        Some(file) => dir.join(file),
    }
}

/// The OpenAI-compatible base URL for a served model.
pub fn endpoint_url(host: &str, port: u16) -> String {
    format!("http://{host}:{port}/v1")
}

/// Build the `hi-local serve …` argument vector.
pub fn serve_args(model_path: &Path, spec: &LocalModelSpec, host: &str, port: u16) -> Vec<String> {
    vec![
        "serve".to_string(),
        model_path.to_string_lossy().into_owned(),
        "--backend".to_string(),
        spec.backend.serve_flag().to_string(),
        "--host".to_string(),
        host.to_string(),
        "--port".to_string(),
        port.to_string(),
        "--model-id".to_string(),
        spec.model_id.clone(),
    ]
}

/// Locate the `hi-local` binary: `$HI_LOCAL_BIN`, else a sibling of the current
/// executable, else the bare name resolved on `PATH` at spawn.
pub fn find_hi_local() -> PathBuf {
    if let Some(path) = std::env::var_os("HI_LOCAL_BIN") {
        return PathBuf::from(path);
    }
    if let Ok(current) = std::env::current_exe()
        && let Some(dir) = current.parent()
    {
        let sibling = dir.join(format!("hi-local{}", std::env::consts::EXE_SUFFIX));
        if sibling.exists() {
            return sibling;
        }
    }
    PathBuf::from("hi-local")
}

/// Pick a free localhost port for the server, honoring `HI_SKEPTIC_LOCAL_PORT`
/// as the starting point (default 8080).
fn pick_free_port() -> u16 {
    let start = std::env::var("HI_SKEPTIC_LOCAL_PORT")
        .ok()
        .and_then(|v| v.trim().parse::<u16>().ok())
        .unwrap_or(8080);
    let end = start.saturating_add(64);
    for port in start..=end {
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    start
}

/// Outcome of turning the local skeptic on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalSkepticOutcome {
    /// The server is up and the skeptic now routes to `endpoint`.
    Ready { endpoint: String, model_id: String },
    /// Weights aren't cached and an inline download wasn't allowed (the TUI).
    /// The caller should fetch `repo` into `dir` once, then re-run.
    NeedsDownload { repo: String, dir: PathBuf },
    /// No local-inference backend was detected on this machine.
    NoBackend,
}

/// Session state for an active local skeptic, kept so it can be torn down and
/// the prior skeptic settings restored on `/config skeptic-local off`.
pub(crate) struct LocalSkepticState {
    pub(crate) process_id: String,
    pub(crate) endpoint: String,
    pub(crate) model_id: String,
    pub(crate) prev_skeptic_model: Option<String>,
    pub(crate) prev_endpoint: Option<String>,
    pub(crate) prev_endpoint_key: Option<String>,
}

/// Build the optional skeptic provider from an endpoint config. Shared by the
/// constructor and the runtime toggle so their wiring can't drift.
pub(crate) fn build_skeptic_provider(config: &crate::AgentConfig) -> Option<Arc<dyn Provider>> {
    config.subagents.skeptic_endpoint.as_deref().map(|url| {
        let key = config
            .subagents
            .skeptic_endpoint_key
            .clone()
            .unwrap_or_else(|| "local".to_string());
        Arc::new(hi_ai::OpenAiProvider::new(url.to_string(), key)) as Arc<dyn Provider>
    })
}

impl Agent {
    /// Whether an auto-managed local skeptic is currently running, and at what
    /// endpoint.
    pub fn local_skeptic_endpoint(&self) -> Option<&str> {
        self.local_skeptic
            .as_ref()
            .filter(|state| self.local_skeptic_server_is_running(state))
            .map(|s| s.endpoint.as_str())
    }

    /// Rebuild [`Agent::skeptic_provider`] from the current config after the
    /// skeptic endpoint changes at runtime.
    pub(crate) fn rebuild_skeptic_provider(&mut self) {
        self.skeptic_provider = build_skeptic_provider(&self.config);
    }

    /// Turn the auto-managed local skeptic on: detect a backend, fetch the
    /// default review model if needed, launch `hi-local`, wait for health, and
    /// route the skeptic review to it. `allow_download` gates the blocking,
    /// progress-to-terminal model fetch — the plain CLI passes `true`; the TUI
    /// passes `false` (a multi-GB download would corrupt its alternate screen)
    /// and gets [`LocalSkepticOutcome::NeedsDownload`] when the model is absent.
    ///
    /// Idempotent: a second call while already on just reports `Ready`. On any
    /// failure it returns `Err` and leaves the skeptic on the main provider.
    pub async fn enable_local_skeptic(
        &mut self,
        allow_download: bool,
    ) -> Result<LocalSkepticOutcome> {
        if let Some(state) = &self.local_skeptic
            && self.local_skeptic_server_is_running(state)
        {
            return Ok(LocalSkepticOutcome::Ready {
                endpoint: state.endpoint.clone(),
                model_id: state.model_id.clone(),
            });
        }
        if let Some(state) = self.local_skeptic.take() {
            // A managed child can exit independently (OOM, crash, or an
            // external kill). Restore the route that existed before the dead
            // local skeptic was installed before trying the team-server reuse
            // path or provisioning a replacement.
            if !state.process_id.is_empty() {
                hi_tools::stop_local_server(&state.process_id);
            }
            self.config.subagents.skeptic_model = state.prev_skeptic_model;
            self.config.subagents.skeptic_endpoint = state.prev_endpoint;
            self.config.subagents.skeptic_endpoint_key = state.prev_endpoint_key;
            self.rebuild_skeptic_provider();
        }
        // A provisioned team executor (laguna, coder-32b…) is already serving
        // locally? Review on it for free — no second download, no second
        // server stacked in RAM. The team registry keeps owning the process:
        // the empty process id makes disable restore routes without stopping
        // a server the executors still depend on.
        if let Some((endpoint, model_id)) = self.any_team_local_server() {
            let prev_skeptic_model = self.config.subagents.skeptic_model.clone();
            let prev_endpoint = self.config.subagents.skeptic_endpoint.clone();
            let prev_endpoint_key = self.config.subagents.skeptic_endpoint_key.clone();
            self.config.subagents.skeptic_endpoint = Some(endpoint.clone());
            self.config.subagents.skeptic_endpoint_key = Some("local".to_string());
            self.config.subagents.skeptic_model = Some(model_id.clone());
            self.rebuild_skeptic_provider();
            self.local_skeptic = Some(LocalSkepticState {
                process_id: String::new(),
                endpoint: endpoint.clone(),
                model_id: model_id.clone(),
                prev_skeptic_model,
                prev_endpoint,
                prev_endpoint_key,
            });
            return Ok(LocalSkepticOutcome::Ready { endpoint, model_id });
        }
        let Some(backend) = detect_backend_offload().await else {
            return Ok(LocalSkepticOutcome::NoBackend);
        };
        let spec = resolve_model(backend);
        let dir = hi_tools::skeptic_model_dir(&spec.repo);
        if !model_present(&dir, &spec) {
            if !allow_download {
                return Ok(LocalSkepticOutcome::NeedsDownload {
                    repo: spec.repo.clone(),
                    dir,
                });
            }
            hi_tools::download_repo_keep_foreground(&spec.repo, &dir)
                .await
                .with_context(|| format!("downloading local skeptic model {}", spec.repo))?;
            if !model_present(&dir, &spec) {
                bail!(
                    "downloaded {} but its weights are still missing under {}",
                    spec.repo,
                    dir.display()
                );
            }
        }
        let abs_dir = std::fs::canonicalize(&dir).unwrap_or(dir);
        let model_path = serve_model_path(&abs_dir, &spec);
        let bin = ensure_hi_local_binary(backend).await?;
        let host = "127.0.0.1";
        let port = pick_free_port();
        let args = serve_args(&model_path, &spec, host, port);
        let deadline = health_deadline_for_model(model_dir_bytes(&abs_dir));
        let handle =
            hi_tools::start_local_server_with_deadline(&bin, &args, host, port, deadline)
                .await
                .with_context(|| {
                    format!(
                        "hi-local ({}) did not become ready within {}s — is it built with the {} backend?",
                        bin.display(),
                        deadline.as_secs(),
                        spec.backend.serve_flag()
                    )
                })?;

        let prev_skeptic_model = self.config.subagents.skeptic_model.clone();
        let prev_endpoint = self.config.subagents.skeptic_endpoint.clone();
        let prev_endpoint_key = self.config.subagents.skeptic_endpoint_key.clone();
        self.config.subagents.skeptic_endpoint = Some(handle.endpoint.clone());
        self.config.subagents.skeptic_endpoint_key = Some("local".to_string());
        self.config.subagents.skeptic_model = Some(spec.model_id.clone());
        self.rebuild_skeptic_provider();
        self.local_skeptic = Some(LocalSkepticState {
            process_id: handle.process_id,
            endpoint: handle.endpoint.clone(),
            model_id: spec.model_id.clone(),
            prev_skeptic_model,
            prev_endpoint,
            prev_endpoint_key,
        });
        Ok(LocalSkepticOutcome::Ready {
            endpoint: handle.endpoint,
            model_id: spec.model_id,
        })
    }

    /// Turn the local skeptic off: stop the server and restore the prior skeptic
    /// settings. Returns whether one was running.
    pub fn disable_local_skeptic(&mut self) -> bool {
        let Some(state) = self.local_skeptic.take() else {
            return false;
        };
        // Empty id = the skeptic was riding a team server it doesn't own.
        if !state.process_id.is_empty() {
            hi_tools::stop_local_server(&state.process_id);
        }
        self.config.subagents.skeptic_model = state.prev_skeptic_model;
        self.config.subagents.skeptic_endpoint = state.prev_endpoint;
        self.config.subagents.skeptic_endpoint_key = state.prev_endpoint_key;
        self.rebuild_skeptic_provider();
        // A skeptic may have been riding a team server whose executor route
        // was cleared while the skeptic was enabled. Reconcile after restoring
        // the prior route, or the now-unreferenced server keeps consuming RAM
        // until the whole session exits.
        self.release_unreferenced_team_servers();
        true
    }

    /// Stop any auto-managed local skeptic server without touching config, for
    /// session shutdown. Called from [`Agent::kill_background_processes`].
    pub(crate) fn stop_local_skeptic_server(&self) {
        if let Some(state) = &self.local_skeptic
            && !state.process_id.is_empty()
        {
            hi_tools::stop_local_server(&state.process_id);
        }
    }
}

// ---- Supported team-role local models ------------------------------------
//
// `/team <role> <name>` provisioning: the user picks a short supported name
// (or just `local`), and hi does everything — hardware-sized selection,
// download, server spawn, health wait, role wiring. Nobody types endpoints.

/// One quantization of a model's MLX form. pipenetwork publishes whole quant
/// ladders (2bit→8bit) per model, so the same model fits very different
/// machines — a 3bit Laguna runs where the 4bit can't.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlxQuant {
    /// Short quant tag users can force with `name@quant` (e.g. `4bit`).
    pub quant: &'static str,
    /// Comfortable-fit floor for this quant (weights + KV/OS headroom).
    pub min_ram_gb: u64,
    /// Approximate download size in GiB. Used by the provider picker for a
    /// disk-fit check; the downloader remains authoritative at runtime.
    pub download_gb: u64,
    pub repo: &'static str,
    pub model_id: &'static str,
}

/// The source of a managed local runtime. Hub sources are downloaded into
/// hi's cache; directory sources are validated and served in place.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LocalModelSource {
    Hub { repo: String },
    Directory { path: PathBuf },
}

impl LocalModelSource {
    pub fn identity(&self) -> String {
        match self {
            Self::Hub { repo } => format!("hub:{repo}"),
            Self::Directory { path } => format!("dir:{}", path.display()),
        }
    }
}

/// Capability information shown before a local model is started. `Unknown`
/// is intentional for live Hub rows whose model card does not advertise a
/// reliable tool contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LocalToolSupport {
    ToolCapable,
    ChatOnly,
    #[default]
    Unknown,
}

/// Structured data for a local-model picker row. Keeping this richer than a
/// `(name, detail, id)` tuple lets the TUI explain memory and capability
/// tradeoffs before a potentially multi-gigabyte download starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalModelOption {
    pub display_name: String,
    pub model_id: String,
    pub source: LocalModelSource,
    pub quantization: Option<String>,
    pub download_bytes: Option<u64>,
    pub resident_bytes: Option<u64>,
    pub min_ram_gb: Option<u64>,
    pub context_window: Option<u32>,
    pub tool_support: LocalToolSupport,
    pub installed: bool,
}

/// A live model entry discovered from the Pipe Network Hugging Face
/// collections. The exact file total is fetched from the Hub and is used for
/// both disk filtering and download progress.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LocalCatalogModel {
    pub collection: String,
    pub display_name: String,
    pub repo: String,
    pub model_id: String,
    pub quant: String,
    pub pipeline_tag: String,
    pub download_bytes: u64,
    pub resident_bytes: u64,
    pub note: Option<String>,
    #[serde(default)]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub tool_support: LocalToolSupport,
}

impl LocalCatalogModel {
    const GIB: u64 = 1024 * 1024 * 1024;

    /// RAM available to a local model after reserving 8 GiB for the OS, the
    /// TUI, and request KV cache. A fixed reserve makes a 96 GiB machine useful
    /// for the larger Nemotron variants without admitting genuinely oversized
    /// checkpoints.
    pub fn fits_ram(&self, system_ram_gb: u64) -> bool {
        let host_reserve = 8 * Self::GIB;
        self.resident_bytes.saturating_add(host_reserve) <= system_ram_gb.saturating_mul(Self::GIB)
    }

    pub fn fits_disk(&self, available_bytes: Option<u64>) -> bool {
        available_bytes
            .is_none_or(|available| available >= self.download_bytes.saturating_add(Self::GIB))
    }

    pub fn fits_machine(&self, system_ram_gb: u64, available_bytes: Option<u64>) -> bool {
        self.fits_ram(system_ram_gb) && self.fits_disk(available_bytes)
    }
}

static PIPENETWORK_CATALOG: LazyLock<Mutex<Option<Vec<LocalCatalogModel>>>> =
    LazyLock::new(|| Mutex::new(None));

pub fn cached_pipenetwork_catalog() -> Option<Vec<LocalCatalogModel>> {
    PIPENETWORK_CATALOG
        .lock()
        .ok()
        .and_then(|catalog| catalog.clone())
}

fn cache_pipenetwork_catalog(catalog: Vec<LocalCatalogModel>) {
    if let Ok(mut cached) = PIPENETWORK_CATALOG.lock() {
        *cached = Some(catalog);
    }
}

#[derive(Clone, Debug)]
struct PipeCatalogCandidate {
    collection: String,
    repo: String,
    pipeline_tag: String,
    note: Option<String>,
}

/// Refresh the chat-capable MLX subset of Pipe Network's public Hub.
///
/// The collections endpoint is useful because it carries Pipe Network's
/// human-written quant notes, but it is not a complete inventory: newly
/// published collections and some older collection revisions are absent from
/// the owner's collection feed. Merge it with the author's model listing so a
/// model is never hidden merely because its collection page has not appeared
/// in that feed yet. This runs in the TUI's background task, so `/provider`
/// never waits on the network. Models with image/video pipelines are excluded
/// because the OpenAI-compatible coding runtime cannot serve them.
pub async fn refresh_pipenetwork_catalog() -> Result<Vec<LocalCatalogModel>> {
    let client = hi_ai::HuggingFaceHubClient::from_env();
    let (collections_result, author_models_result) = tokio::join!(
        client.list_collections("pipenetwork"),
        client.author_models("pipenetwork", 500),
    );

    let collections = collections_result.unwrap_or_default();
    let author_models = author_models_result.unwrap_or_default();
    if collections.is_empty() && author_models.is_empty() {
        bail!("Pipe Network returned no model collections or author models");
    }

    let mut candidates = HashMap::<String, PipeCatalogCandidate>::new();

    // Collection entries win when the same repo appears in both sources:
    // they carry the best display title and the quant-specific size/quality
    // notes written by Pipe Network.
    for collection in collections {
        for item in collection.items {
            let is_model = item.kind.as_deref().is_none_or(|kind| kind == "model");
            let tags = item
                .pipeline_tag
                .as_deref()
                .map(|tag| vec![tag.to_string()])
                .unwrap_or_default();
            if !is_supported_pipenetwork_model(&item.id, item.pipeline_tag.as_deref(), &tags) {
                continue;
            }
            if is_model {
                candidates.insert(
                    item.id.clone(),
                    PipeCatalogCandidate {
                        collection: collection.title.clone(),
                        repo: item.id,
                        pipeline_tag: item
                            .pipeline_tag
                            .unwrap_or_else(|| "text-generation".to_string()),
                        note: item.note.and_then(|note| note.text),
                    },
                );
            }
        }
    }

    // The author listing is the complete current inventory. It also catches
    // repositories whose id does not contain `MLX` but whose Hub tags mark
    // them as MLX exports (for example some LongCat variants).
    for model in author_models {
        if !is_supported_pipenetwork_model(&model.id, None, &model.tags) {
            continue;
        }
        let pipeline_tag = supported_pipeline_tag(None, &model.tags);
        let candidate = PipeCatalogCandidate {
            collection: "Pipe Network models".to_string(),
            repo: model.id.clone(),
            pipeline_tag,
            note: None,
        };
        candidates.entry(model.id).or_insert(candidate);
    }

    let candidates: Vec<PipeCatalogCandidate> = candidates.into_values().collect();

    let mut catalog: Vec<LocalCatalogModel> = stream::iter(candidates)
        .map(|candidate| {
            let client = client.clone();
            async move {
                let repo = hi_ai::HfRepoRef::parse(&candidate.repo)?;
                let files = client.list_files(&repo).await?;
                let download_bytes = files.iter().filter_map(|file| file.size).sum::<u64>();
                if download_bytes == 0 {
                    bail!(
                        "Hugging Face returned no sized files for {}",
                        candidate.repo
                    );
                }
                let resident_bytes = advertised_resident_bytes(candidate.note.as_deref())
                    .unwrap_or_else(|| download_bytes.saturating_mul(5) / 4);
                let quant = quantization_for_repo(&candidate.repo);
                let display_name =
                    display_name_for_repo(&candidate.collection, &candidate.repo, &quant);
                Ok(LocalCatalogModel {
                    collection: candidate.collection,
                    display_name,
                    repo: candidate.repo.clone(),
                    model_id: candidate
                        .repo
                        .rsplit('/')
                        .next()
                        .unwrap_or(&candidate.repo)
                        .to_string(),
                    quant,
                    pipeline_tag: candidate.pipeline_tag,
                    download_bytes,
                    resident_bytes,
                    note: candidate.note,
                    context_window: None,
                    tool_support: LocalToolSupport::Unknown,
                })
            }
        })
        .buffer_unordered(8)
        .filter_map(|result| async { result.ok() })
        .collect()
        .await;

    catalog.sort_by_key(|model| (model.resident_bytes, model.download_bytes));
    catalog.dedup_by(|left, right| left.repo == right.repo);
    cache_pipenetwork_catalog(catalog.clone());
    Ok(catalog)
}

fn supported_pipeline_tag(pipeline_tag: Option<&str>, tags: &[String]) -> String {
    pipeline_tag
        .filter(|tag| matches!(*tag, "text-generation" | "image-text-to-text"))
        .or_else(|| {
            tags.iter()
                .map(String::as_str)
                .find(|tag| matches!(*tag, "text-generation" | "image-text-to-text"))
        })
        .unwrap_or("text-generation")
        .to_string()
}

fn is_supported_pipenetwork_model(repo: &str, pipeline_tag: Option<&str>, tags: &[String]) -> bool {
    let lower_repo = repo.to_ascii_lowercase();
    let lower_tags: Vec<String> = tags.iter().map(|tag| tag.to_ascii_lowercase()).collect();
    let is_mlx = lower_repo.contains("mlx") || lower_tags.iter().any(|tag| tag == "mlx");
    let is_chat_capable = matches!(
        pipeline_tag,
        Some("text-generation") | Some("image-text-to-text")
    ) || lower_tags
        .iter()
        .any(|tag| tag == "text-generation" || tag == "image-text-to-text");
    let is_supported_quant =
        !lower_repo.contains("nvfp4") && !lower_tags.iter().any(|tag| tag == "nvfp4");
    is_mlx && is_chat_capable && is_supported_quant
}

fn quantization_for_repo(repo: &str) -> String {
    let lower = repo.to_ascii_lowercase();
    for quant in [
        "mixed-4_8bit",
        "mixed-3_6bit",
        "reapgraded",
        "reap50",
        "reap37",
        "reap25",
        "reap12",
        "reap",
        "mxfp4-q8",
        "nvfp4",
        "bf16",
        "8bit",
        "6bit",
        "5bit",
        "4bit",
        "3bit",
        "2bit",
    ] {
        if lower.contains(quant) {
            return quant.to_string();
        }
    }
    "unknown".to_string()
}

fn display_name_for_repo(collection: &str, repo: &str, quant: &str) -> String {
    let family = repo
        .rsplit('/')
        .next()
        .unwrap_or(repo)
        .replace("-MLX", "")
        .replace("-mlx", "")
        .replace("-context", "")
        .replace("-mixed-4_8bit", "")
        .replace("-mixed-3_6bit", "")
        .replace("-REAPgraded", "")
        .replace("-REAP50", "")
        .replace("-REAP37", "")
        .replace("-REAP25", "")
        .replace("-REAP12", "")
        .replace("-mxfp4-q8", "")
        .replace("-4bit", "")
        .replace("-6bit", "")
        .replace("-8bit", "")
        .replace("-5bit", "")
        .replace("-bf16", "")
        .replace("-nvfp4", "");
    format!("{family} · {quant} · {collection}")
}

fn advertised_resident_bytes(note: Option<&str>) -> Option<u64> {
    let note = note?.to_ascii_lowercase();
    let (before_resident, _) = note.split_once("resident")?;
    parse_last_size(before_resident)
}

fn parse_last_size(text: &str) -> Option<u64> {
    let words: Vec<&str> = text.split_whitespace().collect();
    for index in (0..words.len()).rev() {
        let word = words[index].trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '.');
        let lower = word.to_ascii_lowercase();
        let (number, multiplier) = if lower.len() > 2
            && let Some(number) = lower.strip_suffix("gb")
        {
            (number, 1.0)
        } else if lower.len() > 2
            && let Some(number) = lower.strip_suffix("tb")
        {
            (number, 1024.0)
        } else if matches!(lower.as_str(), "gb" | "gib") && index > 0 {
            (words[index - 1], 1.0)
        } else if matches!(lower.as_str(), "tb" | "tib") && index > 0 {
            (words[index - 1], 1024.0)
        } else {
            continue;
        };
        let number = number
            .trim_matches(|ch: char| !ch.is_ascii_digit() && ch != '.')
            .parse::<f64>()
            .ok()?;
        return Some((number * multiplier * LocalCatalogModel::GIB as f64) as u64);
    }
    None
}

/// A verified GGUF form for the CUDA backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaGguf {
    pub min_ram_gb: u64,
    pub repo: &'static str,
    pub gguf_file: &'static str,
    pub model_id: &'static str,
}

/// One curated local model users can select by short name. MLX is the
/// primary form (verified pipenetwork/mlx-community conversions), carried as
/// a quant ladder ordered best-quality-first; the CUDA GGUF form is optional
/// — entries without a verified GGUF are honestly MLX-only rather than
/// pointing at guessed repos.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupportedLocalModel {
    /// What the user types: `/team delegate <name>` (or `name@quant`).
    pub name: &'static str,
    /// One-line human description for the picker.
    pub label: &'static str,
    /// MLX quant ladder, **best quality first** — selection takes the first
    /// quant whose floor fits, so bigger machines get better weights.
    pub mlx: &'static [MlxQuant],
    pub cuda: Option<CudaGguf>,
}

impl SupportedLocalModel {
    /// Highest-quality MLX quant that fits `ram_gb`.
    pub fn pick_mlx(&self, ram_gb: u64) -> Option<&'static MlxQuant> {
        self.mlx.iter().find(|quant| ram_gb >= quant.min_ram_gb)
    }

    /// The smallest MLX quant — what an explicit oversized pick falls to.
    pub fn smallest_mlx(&self) -> Option<&'static MlxQuant> {
        self.mlx.iter().min_by_key(|quant| quant.min_ram_gb)
    }

    /// Smallest workable floor on `backend` (`None` = backend unknown:
    /// smallest floor across all published forms).
    pub fn min_ram_gb(&self, backend: Option<LocalBackend>) -> u64 {
        let mlx_floor = self.smallest_mlx().map(|quant| quant.min_ram_gb);
        let cuda_floor = self.cuda.map(|cuda| cuda.min_ram_gb);
        match backend {
            Some(LocalBackend::Mlx) => mlx_floor.unwrap_or(u64::MAX),
            Some(LocalBackend::Cuda) => cuda_floor.unwrap_or(u64::MAX),
            None => mlx_floor.min(cuda_floor).unwrap_or(u64::MAX),
        }
    }

    /// Whether any published form of this model fits `ram_gb` on `backend`.
    pub fn fits(&self, ram_gb: u64, backend: Option<LocalBackend>) -> bool {
        ram_gb >= self.min_ram_gb(backend)
    }

    /// Compact ladder summary for the picker, e.g. `8/6/4/3/2bit`.
    pub fn quant_summary(&self) -> String {
        let tags: Vec<&str> = self
            .mlx
            .iter()
            .map(|quant| quant.quant.strip_suffix("bit").unwrap_or(quant.quant))
            .collect();
        let all_bits = self.mlx.iter().all(|quant| quant.quant.ends_with("bit"));
        if all_bits {
            format!("{}bit", tags.join("/"))
        } else {
            tags.join("/")
        }
    }
}

/// A catalog entry resolved for a specific machine: the family plus the MLX
/// quant chosen for it (highest quality that fits, an explicit `@quant`, or
/// the smallest quant when nothing fits an explicit pick).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedLocalModel {
    pub entry: &'static SupportedLocalModel,
    pub mlx: Option<&'static MlxQuant>,
}

/// An explicitly selected local runtime. Unlike [`LocalModelSpec`], this
/// carries the stable identity used by profiles and the TUI; the process id
/// and port are deliberately not part of the persisted identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalRuntimeSpec {
    pub repo: String,
    pub model_id: String,
    pub backend: LocalBackend,
    pub model_dir: PathBuf,
    pub profile_name: String,
    pub source: LocalModelSource,
    pub quantization: Option<String>,
    pub context_window: Option<u32>,
    pub tool_support: LocalToolSupport,
}

/// A local server that has completed provisioning and verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManagedLocalRuntime {
    pub runtime_id: String,
    pub profile_name: String,
    pub repo: String,
    pub model_id: String,
    pub base_url: String,
    pub process_id: String,
    pub model_dir: PathBuf,
    pub backend: LocalBackend,
    pub source: LocalModelSource,
    pub quantization: Option<String>,
    pub context_window: Option<u32>,
    pub tool_support: LocalToolSupport,
}

/// Progress phases for driver-local runtime provisioning. The loading phase
/// carries the managed process handle and model size so frontends can show a
/// live memory-based progress bar while the server becomes ready.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalRuntimePhase {
    Resolving,
    Downloading,
    PreparingRuntime,
    StartingServer,
    LoadingModel {
        deadline_secs: u64,
        /// The managed-process handle used by the TUI to sample load progress.
        server_handle: String,
        /// On-disk model size, used as the approximate memory-load target.
        expected_bytes: u64,
    },
    Verifying,
    Ready,
}

/// Build a managed runtime spec from one catalog selection.
pub fn local_runtime_spec(
    name: &str,
    ram_gb: u64,
    backend: LocalBackend,
) -> Result<LocalRuntimeSpec> {
    if backend != LocalBackend::Mlx {
        bail!("managed local driver runtimes currently support MLX only");
    }
    let (repo, model_id, quantization, context_window, tool_support) =
        if let Some(resolved) = resolve_team_local_model(name, ram_gb, Some(backend)) {
            let spec = team_model_spec(resolved, backend)?;
            let quantization = resolved.mlx.map(|quant| quant.quant.to_string());
            (
                spec.repo,
                spec.model_id,
                quantization,
                None,
                LocalToolSupport::ToolCapable,
            )
        } else if name.contains('/') {
            // Provider-picker rows use the repository id as their stable action
            // key for live collection entries. The repository itself is enough to
            // reproduce the runtime identity after the picker refreshes; no
            // server port or process state is embedded in the selection.
            let repo = hi_ai::HfRepoRef::parse(name)
                .with_context(|| format!("invalid local model repository '{name}'"))?
                .repo_id;
            let model_id = repo.rsplit('/').next().unwrap_or(&repo).to_string();
            let metadata = cached_pipenetwork_catalog()
                .and_then(|catalog| catalog.into_iter().find(|model| model.repo == repo));
            (
                repo,
                model_id,
                metadata.as_ref().map(|model| model.quant.clone()),
                metadata.as_ref().and_then(|model| model.context_window),
                metadata.map(|model| model.tool_support).unwrap_or_default(),
            )
        } else {
            bail!("unknown local model '{name}'")
        };
    let model_dir = hi_tools::skeptic_model_dir(&repo);
    let profile_name = format!(
        "mlx-{}",
        hi_tools::safe_path(&model_id).to_ascii_lowercase()
    );
    Ok(LocalRuntimeSpec {
        repo: repo.clone(),
        model_id,
        backend,
        model_dir,
        profile_name,
        source: LocalModelSource::Hub { repo: repo.clone() },
        quantization,
        context_window,
        tool_support,
    })
}

/// Build a runtime spec for an already-downloaded MLX model directory. The
/// directory is validated before a server is spawned so a stale profile fails
/// with an actionable path error instead of a generic server-start timeout.
pub fn local_runtime_spec_from_directory(
    path: &Path,
    model_id: Option<&str>,
) -> Result<LocalRuntimeSpec> {
    if !path.is_dir() {
        bail!(
            "local MLX model directory does not exist: {}",
            path.display()
        );
    }
    let model_dir = std::fs::canonicalize(path)
        .with_context(|| format!("resolving local MLX model directory {}", path.display()))?;
    let model_id = model_id
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .or_else(|| {
            model_dir
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .ok_or_else(|| {
            anyhow::anyhow!("could not derive a model id from {}", model_dir.display())
        })?;
    let probe = LocalModelSpec {
        repo: model_dir.to_string_lossy().into_owned(),
        model_id: model_id.clone(),
        gguf_file: None,
        backend: LocalBackend::Mlx,
    };
    if !model_present(&model_dir, &probe) {
        bail!(
            "{} is not a complete MLX model directory (expected config.json and all weight shards)",
            model_dir.display()
        );
    }
    let profile_name = format!(
        "mlx-{}",
        hi_tools::safe_path(&model_id).to_ascii_lowercase()
    );
    Ok(LocalRuntimeSpec {
        repo: model_dir.to_string_lossy().into_owned(),
        model_id,
        backend: LocalBackend::Mlx,
        model_dir: model_dir.clone(),
        profile_name,
        source: LocalModelSource::Directory { path: model_dir },
        quantization: None,
        context_window: None,
        tool_support: LocalToolSupport::Unknown,
    })
}

/// Build the persisted-directory form of a runtime without probing the
/// filesystem. Startup uses this form so a moved directory can be shown in
/// the TUI recovery screen instead of aborting before the TUI paints.
pub fn local_runtime_spec_from_directory_source(
    path: PathBuf,
    model_id: String,
    quantization: Option<String>,
    context_window: Option<u32>,
    tool_support: LocalToolSupport,
) -> LocalRuntimeSpec {
    let profile_name = format!(
        "mlx-{}",
        hi_tools::safe_path(&model_id).to_ascii_lowercase()
    );
    LocalRuntimeSpec {
        repo: path.to_string_lossy().into_owned(),
        model_id,
        backend: LocalBackend::Mlx,
        model_dir: path.clone(),
        profile_name,
        source: LocalModelSource::Directory { path },
        quantization,
        context_window,
        tool_support,
    }
}

/// Stable identity for a runtime, independent of its ephemeral endpoint.
pub fn local_runtime_id(spec: &LocalRuntimeSpec) -> String {
    format!(
        "{}:{}:{}",
        spec.backend.serve_flag(),
        spec.source.identity(),
        spec.model_id
    )
}

impl ResolvedLocalModel {
    /// User-facing name: `laguna-s@3bit` when the family has a real ladder,
    /// the bare family name when there is only one form.
    pub fn display(&self) -> String {
        match self.mlx {
            Some(quant) if self.entry.mlx.len() > 1 => {
                format!("{}@{}", self.entry.name, quant.quant)
            }
            _ => self.entry.name.to_string(),
        }
    }
}

/// The supported catalog. **Order = auto-selection preference**: `local`
/// takes the first entry with any quant that fits the machine, so
/// newer/faster models come before older peers. Within an entry the quant
/// ladder is best-quality-first. Entries after `mini` (which fits almost
/// everywhere) are explicit-pick / picker-only. Every repo below was
/// verified against the pipenetwork / mlx-community HF listings — no
/// guessed names.
pub const SUPPORTED_LOCAL_MODELS: &[SupportedLocalModel] = &[
    // Laguna floors encode the live-verified serving rule: the weights must
    // fit wired GPU memory (~72% of RAM) or the MoE expert path degrades to
    // ~0 tok/s. Proven on a 64GB Mac: 2bit (35GB) scores 4/4 on team-bench,
    // 3bit (51.5GB) exceeds the wired limit and cannot generate.
    SupportedLocalModel {
        name: "laguna-s",
        label: "Poolside Laguna S 2.1 — 118B MoE coding flagship (1M context; 4/4 on team-bench)",
        mlx: &[
            MlxQuant {
                quant: "8bit",
                min_ram_gb: 256,
                download_gb: 304,
                repo: "pipenetwork/Laguna-S-2.1-MLX-8bit",
                model_id: "Laguna-S-2.1-MLX-8bit",
            },
            MlxQuant {
                quant: "6bit",
                min_ram_gb: 192,
                download_gb: 230,
                repo: "pipenetwork/Laguna-S-2.1-MLX-6bit",
                model_id: "Laguna-S-2.1-MLX-6bit",
            },
            MlxQuant {
                quant: "4bit",
                min_ram_gb: 96,
                download_gb: 155,
                repo: "pipenetwork/Laguna-S-2.1-MLX-4bit",
                model_id: "Laguna-S-2.1-MLX-4bit",
            },
            MlxQuant {
                quant: "3bit",
                min_ram_gb: 96,
                download_gb: 120,
                repo: "pipenetwork/Laguna-S-2.1-MLX-3bit",
                model_id: "Laguna-S-2.1-MLX-3bit",
            },
            MlxQuant {
                quant: "2bit",
                min_ram_gb: 64,
                download_gb: 80,
                repo: "pipenetwork/Laguna-S-2.1-MLX-2bit",
                model_id: "Laguna-S-2.1-MLX-2bit",
            },
        ],
        cuda: None,
    },
    SupportedLocalModel {
        name: "coder-32b",
        label: "Qwen2.5-Coder 32B — dense coder (~19GB)",
        mlx: &[MlxQuant {
            quant: "4bit",
            min_ram_gb: 40,
            download_gb: 19,
            repo: "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
            model_id: "Qwen2.5-Coder-32B-Instruct-4bit",
        }],
        cuda: Some(CudaGguf {
            min_ram_gb: 40,
            repo: "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
            gguf_file: "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
            model_id: "qwen2.5-coder-32b-instruct",
        }),
    },
    SupportedLocalModel {
        name: "coder-14b",
        label: "Qwen2.5-Coder 14B — dense coder (~9GB)",
        mlx: &[MlxQuant {
            quant: "4bit",
            min_ram_gb: 24,
            download_gb: 9,
            repo: "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
            model_id: "Qwen2.5-Coder-14B-Instruct-4bit",
        }],
        cuda: Some(CudaGguf {
            min_ram_gb: 24,
            repo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
            gguf_file: "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
            model_id: "qwen2.5-coder-14b-instruct",
        }),
    },
    SupportedLocalModel {
        name: "coder-7b",
        label: "Qwen2.5-Coder 7B — dense coder (~5GB)",
        mlx: &[MlxQuant {
            quant: "4bit",
            min_ram_gb: 12,
            download_gb: 5,
            repo: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            model_id: "Qwen2.5-Coder-7B-Instruct-4bit",
        }],
        cuda: Some(CudaGguf {
            min_ram_gb: 12,
            repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            gguf_file: "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            model_id: "qwen2.5-coder-7b-instruct",
        }),
    },
    SupportedLocalModel {
        name: "nemotron-4b",
        label: "NVIDIA Nemotron 3 Nano 4B — tiny modern executor (~2-4GB)",
        mlx: &[
            MlxQuant {
                quant: "8bit",
                min_ram_gb: 8,
                download_gb: 4,
                repo: "pipenetwork/NVIDIA-Nemotron-3-Nano-4B-MLX-8bit",
                model_id: "NVIDIA-Nemotron-3-Nano-4B-MLX-8bit",
            },
            MlxQuant {
                quant: "4bit",
                min_ram_gb: 6,
                download_gb: 2,
                repo: "pipenetwork/NVIDIA-Nemotron-3-Nano-4B-MLX-4bit",
                model_id: "NVIDIA-Nemotron-3-Nano-4B-MLX-4bit",
            },
        ],
        cuda: None,
    },
    SupportedLocalModel {
        name: "mini",
        label: "Qwen2.5 3B — tiny generalist for recon/review (~2GB)",
        mlx: &[MlxQuant {
            quant: "4bit",
            min_ram_gb: 6,
            download_gb: 2,
            repo: "mlx-community/Qwen2.5-3B-Instruct-4bit",
            model_id: "Qwen2.5-3B-Instruct-4bit",
        }],
        cuda: Some(CudaGguf {
            min_ram_gb: 6,
            repo: "Qwen/Qwen2.5-3B-Instruct-GGUF",
            gguf_file: "qwen2.5-3b-instruct-q4_k_m.gguf",
            model_id: "qwen2.5-3b-instruct",
        }),
    },
    SupportedLocalModel {
        name: "deepseek-coder-v2-lite",
        label: "DeepSeek Coder V2 Lite — MLX 4-bit coding model (~8.8GB)",
        mlx: &[MlxQuant {
            quant: "4bit",
            min_ram_gb: 16,
            download_gb: 9,
            repo: "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
            model_id: "DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
        }],
        cuda: None,
    },
    // Explicit-pick / picker-only from here down (mini already fits
    // almost every machine, so auto never reaches these). A model enters the
    // auto section above only after `hi team-bench` has served it and passed
    // tasks on real hardware — live verification found nemotron-30b writing
    // non-compiling code (1/4) and qwen3.6's nvfp4 quant unsupported by
    // hi-mlx, so they wait here, picker-visible and honestly labeled, until
    // the gaps close. (laguna-s earned its way back: 4/4 at 2bit.)
    SupportedLocalModel {
        name: "nemotron-30b",
        label: "NVIDIA Nemotron 3 Nano 30B-A3B — unreliable under hi-local (0/6 on team-bench)",
        mlx: &[MlxQuant {
            quant: "4bit",
            min_ram_gb: 24,
            download_gb: 20,
            repo: "pipenetwork/Nemotron-3-Nano-30B-A3B-context-mlx-4bit",
            model_id: "Nemotron-3-Nano-30B-A3B-context-mlx-4bit",
        }],
        cuda: None,
    },
    SupportedLocalModel {
        name: "qwen3.6-35b",
        label: "Qwen3.6 35B-A3B — needs nvfp4 support hi-local doesn't have yet",
        mlx: &[MlxQuant {
            quant: "nvfp4",
            min_ram_gb: 32,
            download_gb: 20,
            repo: "pipenetwork/Qwen3.6-35B-A3B-mlx-nvfp4",
            model_id: "Qwen3.6-35B-A3B-mlx-nvfp4",
        }],
        cuda: None,
    },
    SupportedLocalModel {
        name: "deepseek-v4-flash",
        label: "DeepSeek V4 Flash — 284B MoE (128GB+ Macs; hi-local speaks V4 natively, unbenched here)",
        mlx: &[
            MlxQuant {
                quant: "8bit",
                min_ram_gb: 512,
                download_gb: 304,
                repo: "mlx-community/DeepSeek-V4-Flash-8bit",
                model_id: "DeepSeek-V4-Flash-8bit",
            },
            MlxQuant {
                quant: "4bit",
                min_ram_gb: 256,
                download_gb: 162,
                repo: "mlx-community/DeepSeek-V4-Flash-4bit",
                model_id: "DeepSeek-V4-Flash-4bit",
            },
            MlxQuant {
                quant: "3bit",
                min_ram_gb: 192,
                download_gb: 130,
                repo: "mlx-community/DeepSeek-V4-Flash-3bit-DQ",
                model_id: "DeepSeek-V4-Flash-3bit-DQ",
            },
            MlxQuant {
                quant: "2bit",
                min_ram_gb: 128,
                download_gb: 90,
                repo: "mlx-community/DeepSeek-V4-Flash-2bit-DQ",
                model_id: "DeepSeek-V4-Flash-2bit-DQ",
            },
        ],
        cuda: None,
    },
    SupportedLocalModel {
        name: "glm-5.2-reap50",
        label: "GLM-5.2 with half the experts REAP-pruned (~195GB; 256GB Macs)",
        mlx: &[MlxQuant {
            quant: "4bit",
            min_ram_gb: 256,
            download_gb: 195,
            repo: "pipenetwork/GLM-5.2-REAP50-MLX-4bit",
            model_id: "GLM-5.2-REAP50-MLX-4bit",
        }],
        cuda: None,
    },
    SupportedLocalModel {
        name: "glm-5.2",
        label: "GLM-5.2 — the cloud driver's own family, local (~390GB; 512GB Macs)",
        mlx: &[MlxQuant {
            quant: "4bit",
            min_ram_gb: 448,
            download_gb: 430,
            repo: "pipenetwork/GLM-5.2-MLX-4bit",
            model_id: "GLM-5.2-MLX-4bit",
        }],
        cuda: None,
    },
    SupportedLocalModel {
        name: "nemotron-550b",
        label: "NVIDIA Nemotron 3 Ultra 550B-A55B — frontier MoE (~275GB; 512GB Macs)",
        mlx: &[MlxQuant {
            quant: "4bit",
            min_ram_gb: 512,
            download_gb: 275,
            repo: "pipenetwork/NVIDIA-Nemotron-3-Ultra-550B-A55B-MLX-4bit",
            model_id: "NVIDIA-Nemotron-3-Ultra-550B-A55B-MLX-4bit",
        }],
        cuda: None,
    },
];

/// Physical memory in GiB, best-effort (0 when unknown).
pub fn system_ram_gb() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output();
        if let Ok(out) = out
            && let Ok(text) = String::from_utf8(out.stdout)
            && let Ok(bytes) = text.trim().parse::<u64>()
        {
            return bytes / (1024 * 1024 * 1024);
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo")
            && let Some(kb) = parse_meminfo_total_kb(&text)
        {
            return kb / (1024 * 1024);
        }
    }
    0
}

/// Parse `MemTotal:  16384 kB` from /proc/meminfo. Pure for testing.
pub fn parse_meminfo_total_kb(meminfo: &str) -> Option<u64> {
    meminfo
        .lines()
        .find(|line| line.starts_with("MemTotal:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// Resolve a `/team` model selection to a catalog entry plus the MLX quant
/// chosen for this machine. `local`/`coder`/`auto` take the first catalog
/// entry with any form that fits `ram_gb` on `backend` (quant = highest
/// quality that fits); exact names are honored even oversized (quant = the
/// smallest published, the closest to feasible); `name@quant` forces an
/// exact quant. `None` = not a supported local model (the caller treats the
/// input as a cloud model id instead).
pub fn resolve_team_local_model(
    name: &str,
    ram_gb: u64,
    backend: Option<LocalBackend>,
) -> Option<ResolvedLocalModel> {
    let name = name.trim().to_ascii_lowercase();
    if matches!(name.as_str(), "local" | "coder" | "auto") {
        let entry = SUPPORTED_LOCAL_MODELS
            .iter()
            .find(|entry| entry.fits(ram_gb, backend))
            // Nothing fits → the smallest supported model on this backend,
            // so tiny machines still get a working executor rather than an
            // arbitrary entry.
            .or_else(|| {
                SUPPORTED_LOCAL_MODELS
                    .iter()
                    .min_by_key(|entry| entry.min_ram_gb(backend))
            })?;
        return Some(ResolvedLocalModel {
            entry,
            mlx: entry.pick_mlx(ram_gb).or_else(|| entry.smallest_mlx()),
        });
    }
    // `name@quant` forces an exact rung of the ladder.
    if let Some((family, quant)) = name.split_once('@') {
        let entry = SUPPORTED_LOCAL_MODELS
            .iter()
            .find(|entry| entry.name == family)?;
        let quant = entry
            .mlx
            .iter()
            .find(|candidate| candidate.quant == quant)?;
        return Some(ResolvedLocalModel {
            entry,
            mlx: Some(quant),
        });
    }
    let entry = SUPPORTED_LOCAL_MODELS
        .iter()
        .find(|entry| entry.name == name)?;
    Some(ResolvedLocalModel {
        entry,
        mlx: entry.pick_mlx(ram_gb).or_else(|| entry.smallest_mlx()),
    })
}

/// The backend-specific serve spec for a resolved catalog selection. Errors
/// on entries that have no verified GGUF for the CUDA backend.
pub fn team_model_spec(
    resolved: ResolvedLocalModel,
    backend: LocalBackend,
) -> Result<LocalModelSpec> {
    match backend {
        LocalBackend::Mlx => {
            let Some(quant) = resolved.mlx else {
                bail!("{} isn't packaged for MLX", resolved.entry.name);
            };
            Ok(LocalModelSpec {
                repo: quant.repo.to_string(),
                model_id: quant.model_id.to_string(),
                gguf_file: None,
                backend,
            })
        }
        LocalBackend::Cuda => {
            let Some(cuda) = resolved.entry.cuda else {
                bail!(
                    "{} isn't packaged for CUDA yet — pick coder-14b, coder-7b, coder-32b, or mini",
                    resolved.entry.name
                );
            };
            Ok(LocalModelSpec {
                repo: cuda.repo.to_string(),
                model_id: cuda.model_id.to_string(),
                gguf_file: Some(cuda.gguf_file.to_string()),
                backend,
            })
        }
    }
}

/// Where an in-flight `/team` local-model setup currently is. Published on a
/// watch channel so the UI can narrate honestly: minutes of silence during a
/// 19 GB download or a multi-minute model load are indistinguishable from a
/// hang without this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProvisionPhase {
    /// Backend detection / cache checks.
    Resolving,
    /// Fetching weights from the hub (quiet; the UI reports bytes on disk).
    Downloading,
    /// Compiling `hi-local` from a development checkout.
    BuildingServer,
    /// Server spawned; weights are loading into memory. `deadline_secs` is
    /// the readiness window; `server_handle` lets the UI sample the server's
    /// memory growth against `expected_bytes` for a real progress bar.
    LoadingModel {
        deadline_secs: u64,
        server_handle: String,
        expected_bytes: u64,
    },
}

/// Fully provision a supported local model: detect the backend, download the
/// weights when absent, spawn a managed `hi-local` server, and wait for
/// health. Returns `(endpoint, model_id, process_id)`. Standalone (no Agent
/// borrow) so frontends can run it on a background task and wire the role
/// when it completes — a 15 GB download must never block the UI. Progress
/// phases are published on `progress` as they begin.
pub async fn provision_team_local_model(
    resolved: ResolvedLocalModel,
    progress: tokio::sync::watch::Sender<ProvisionPhase>,
) -> Result<(String, String, String)> {
    let _ = progress.send(ProvisionPhase::Resolving);
    let Some(backend) = detect_backend_offload().await else {
        bail!(
            "no local-inference backend detected (needs Apple Silicon MLX or an NVIDIA CUDA runtime)"
        );
    };
    let spec = team_model_spec(resolved, backend)?;
    let dir = hi_tools::skeptic_model_dir(&spec.repo);
    if !model_present(&dir, &spec) {
        let _ = progress.send(ProvisionPhase::Downloading);
        // Quiet by contract: this runs behind a live TUI — raw downloader
        // output painted over the alternate screen once; never again.
        hi_tools::download_repo_keep_quiet(&spec.repo, &dir)
            .await
            .with_context(|| format!("downloading {}", spec.repo))?;
        if !model_present(&dir, &spec) {
            bail!(
                "downloaded {} but its weights are still missing under {}",
                spec.repo,
                dir.display()
            );
        }
    }
    let abs_dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    let model_path = serve_model_path(&abs_dir, &spec);
    let bin = ensure_hi_local_binary_with_progress(backend, Some(&progress)).await?;
    let host = "127.0.0.1";
    let port = pick_free_port();
    let args = serve_args(&model_path, &spec, host, port);
    // Big checkpoints take minutes to load; scale the readiness window to
    // the weights instead of failing a healthy-but-still-loading server.
    let expected_bytes = model_dir_bytes(&abs_dir);
    let deadline = health_deadline_for_model(expected_bytes);
    // Spawn first, then wait: publishing the handle lets the UI sample the
    // server's memory growth for a real progress bar while weights load.
    let process_id = hi_tools::spawn_local_server(&bin, &args)?;
    let _ = progress.send(ProvisionPhase::LoadingModel {
        deadline_secs: deadline.as_secs(),
        server_handle: process_id.clone(),
        expected_bytes,
    });
    hi_tools::await_local_server_health(&process_id, host, port, deadline)
        .await
        .with_context(|| {
            format!(
                "hi-local ({}) failed to become ready (allowed {}s)",
                bin.display(),
                deadline.as_secs()
            )
        })?;
    Ok((endpoint_url(host, port), spec.model_id, process_id))
}

/// Provision a driver-local runtime. This is the shared path used by the TUI
/// provider picker and the legacy `/hf run --mlx` command. It intentionally
/// leaves the current provider untouched; callers perform the provider swap
/// only after this function returns a verified runtime.
pub async fn provision_local_runtime(
    runtime: LocalRuntimeSpec,
    progress: tokio::sync::watch::Sender<LocalRuntimePhase>,
) -> Result<ManagedLocalRuntime> {
    let _ = progress.send(LocalRuntimePhase::Resolving);
    let model_spec = LocalModelSpec {
        repo: runtime.repo.clone(),
        model_id: runtime.model_id.clone(),
        gguf_file: None,
        backend: runtime.backend,
    };
    if matches!(&runtime.source, LocalModelSource::Hub { .. })
        && !model_present(&runtime.model_dir, &model_spec)
    {
        let _ = progress.send(LocalRuntimePhase::Downloading);
        hi_tools::download_repo_keep_quiet(&runtime.repo, &runtime.model_dir)
            .await
            .with_context(|| format!("downloading {}", runtime.repo))?;
        if !model_present(&runtime.model_dir, &model_spec) {
            bail!(
                "downloaded {} but its weights are still missing under {}",
                runtime.repo,
                runtime.model_dir.display()
            );
        }
    }

    let _ = progress.send(LocalRuntimePhase::PreparingRuntime);
    if !model_present(&runtime.model_dir, &model_spec) {
        bail!(
            "local MLX model is incomplete under {}",
            runtime.model_dir.display()
        );
    }
    let model_dir = std::fs::canonicalize(&runtime.model_dir).unwrap_or(runtime.model_dir.clone());
    let binary = ensure_hi_local_binary(runtime.backend).await?;
    let host = "127.0.0.1";
    let port = pick_free_port();
    let args = serve_args(&model_dir, &model_spec, host, port);
    let expected_bytes = model_dir_bytes(&model_dir);
    let deadline = health_deadline_for_model(expected_bytes);
    let _ = progress.send(LocalRuntimePhase::StartingServer);
    let process_id = hi_tools::spawn_local_server(&binary, &args)?;
    // Aborting the provisioning task (for example, `/provider cancel` or app
    // shutdown) must not orphan a server that was spawned midway through
    // setup. The guard is disarmed only after the verified runtime is returned
    // to the owner.
    struct LocalRuntimeProcessGuard(Option<String>);
    impl Drop for LocalRuntimeProcessGuard {
        fn drop(&mut self) {
            if let Some(process_id) = self.0.take() {
                hi_tools::stop_local_server(&process_id);
            }
        }
    }
    let mut process_guard = LocalRuntimeProcessGuard(Some(process_id.clone()));
    let _ = progress.send(LocalRuntimePhase::LoadingModel {
        deadline_secs: deadline.as_secs(),
        server_handle: process_id.clone(),
        expected_bytes,
    });
    let endpoint = endpoint_url(host, port);
    if let Err(error) = hi_tools::await_local_server_health(&process_id, host, port, deadline).await
    {
        hi_tools::stop_local_server(&process_id);
        return Err(error).with_context(|| {
            format!(
                "hi-local ({}) failed to become ready for {}",
                binary.display(),
                runtime.model_id
            )
        });
    }

    let _ = progress.send(LocalRuntimePhase::Verifying);
    if let Err(error) = hi_tools::verify_local_server(&endpoint, &runtime.model_id).await {
        hi_tools::stop_local_server(&process_id);
        return Err(error).with_context(|| {
            format!("local runtime verification failed for {}", runtime.model_id)
        });
    }
    let _ = progress.send(LocalRuntimePhase::Ready);
    process_guard.0 = None;
    Ok(ManagedLocalRuntime {
        runtime_id: local_runtime_id(&runtime),
        profile_name: runtime.profile_name,
        repo: runtime.repo,
        model_id: runtime.model_id,
        base_url: endpoint,
        process_id,
        model_dir: runtime.model_dir,
        backend: runtime.backend,
        source: runtime.source,
        quantization: runtime.quantization,
        context_window: runtime.context_window,
        tool_support: runtime.tool_support,
    })
}

#[cfg(test)]
mod team_catalog_tests {
    use super::*;

    const MLX: Option<LocalBackend> = Some(LocalBackend::Mlx);
    const CUDA: Option<LocalBackend> = Some(LocalBackend::Cuda);

    #[test]
    fn auto_sizing_only_lands_on_serve_verified_models() {
        // laguna-s earned auto placement by scoring 4/4 on team-bench at
        // 2bit; its floors encode the verified wired-memory rule, so every
        // tier that resolves to it would hold the weights GPU-resident.
        let sixty_four = resolve_team_local_model("local", 64, MLX).unwrap();
        assert_eq!(
            (sixty_four.entry.name, sixty_four.mlx.unwrap().quant),
            ("laguna-s", "2bit"),
            "64GB auto-picks the bench-proven flagship quant"
        );
        assert_eq!(
            resolve_team_local_model("local", 128, MLX)
                .unwrap()
                .mlx
                .unwrap()
                .quant,
            "4bit",
            "128GB gets the best laguna quant that fits wired memory"
        );
        assert_eq!(
            resolve_team_local_model("local", 40, MLX)
                .unwrap()
                .entry
                .name,
            "coder-32b",
            "below every laguna floor the verified dense coder takes over"
        );
        assert_eq!(
            resolve_team_local_model("coder", 24, MLX)
                .unwrap()
                .entry
                .name,
            "coder-14b",
            "nemotron-30b scored 1/4 on team-bench — the dense coder owns the 24GB tier"
        );
        assert_eq!(
            resolve_team_local_model("auto", 16, MLX)
                .unwrap()
                .entry
                .name,
            "coder-7b"
        );
        assert_eq!(
            resolve_team_local_model("local", 4, MLX)
                .unwrap()
                .entry
                .name,
            "nemotron-4b",
            "tiny machines still get a working executor"
        );
    }

    #[test]
    fn laguna_floors_keep_weights_inside_wired_gpu_memory() {
        // Proven live on 64GB: 2bit (35GB) generates, 3bit (51.5GB) exceeds
        // the ~72%-of-RAM wired limit and stalls to ~0 tok/s. Every floor
        // must keep its quant on the working side of that line.
        assert_eq!(
            resolve_team_local_model("laguna-s", 64, MLX)
                .unwrap()
                .display(),
            "laguna-s@2bit",
            "3bit must NOT resolve on 64GB — it stalled there in live testing"
        );
        assert_eq!(
            resolve_team_local_model("laguna-s", 96, MLX)
                .unwrap()
                .mlx
                .unwrap()
                .quant,
            "4bit"
        );
        assert_eq!(
            resolve_team_local_model("laguna-s", 192, MLX)
                .unwrap()
                .mlx
                .unwrap()
                .quant,
            "6bit"
        );
        assert_eq!(
            resolve_team_local_model("laguna-s", 256, MLX)
                .unwrap()
                .mlx
                .unwrap()
                .quant,
            "8bit",
            "highest quality that fits wins"
        );
    }

    #[test]
    fn auto_sizing_on_cuda_skips_mlx_only_families() {
        assert_eq!(
            resolve_team_local_model("local", 128, CUDA)
                .unwrap()
                .entry
                .name,
            "coder-32b",
            "the biggest verified-GGUF entry wins on a big CUDA box"
        );
        assert_eq!(
            resolve_team_local_model("local", 32, CUDA)
                .unwrap()
                .entry
                .name,
            "coder-14b",
            "nemotron-30b has no verified GGUF — auto must not pick it on CUDA"
        );
        assert_eq!(
            resolve_team_local_model("local", 4, CUDA)
                .unwrap()
                .entry
                .name,
            "mini",
            "the CUDA fallback is the smallest entry that actually serves on CUDA"
        );
    }

    #[test]
    fn explicit_quants_and_oversized_picks_are_honored() {
        let forced = resolve_team_local_model("laguna-s@2bit", 512, MLX).unwrap();
        assert_eq!(
            forced.mlx.unwrap().quant,
            "2bit",
            "@quant beats auto quality pick"
        );
        assert!(
            resolve_team_local_model("laguna-s@5bit", 512, MLX).is_none(),
            "unpublished quants don't resolve"
        );
        let oversized = resolve_team_local_model("laguna-s", 8, MLX).unwrap();
        assert_eq!(
            oversized.mlx.unwrap().quant,
            "2bit",
            "an explicit pick below every floor falls to the smallest quant"
        );
        assert_eq!(
            resolve_team_local_model("coder-32b", 8, MLX)
                .unwrap()
                .entry
                .name,
            "coder-32b",
            "an explicit pick is honored even below the sizing hint"
        );
        assert_eq!(
            resolve_team_local_model("glm-5.2-reap50", 8, MLX)
                .unwrap()
                .entry
                .name,
            "glm-5.2-reap50"
        );
        // DeepSeek V4 Flash: 284B MoE, explicit-pick only (unbenched), quant
        // floors follow the wired-memory rule (4bit is 151GB on disk).
        let flash = resolve_team_local_model("deepseek-v4-flash", 128, MLX).unwrap();
        assert_eq!(
            flash.mlx.unwrap().quant,
            "2bit",
            "128GB gets the 2bit-DQ rung"
        );
        let forced = resolve_team_local_model("deepseek-v4-flash@3bit", 512, MLX).unwrap();
        assert_eq!(
            team_model_spec(forced, LocalBackend::Mlx).unwrap().repo,
            "mlx-community/DeepSeek-V4-Flash-3bit-DQ"
        );
        assert_eq!(
            resolve_team_local_model("local", 192, MLX)
                .unwrap()
                .entry
                .name,
            "laguna-s",
            "unbenched giants never enter auto selection"
        );
        assert_eq!(
            resolve_team_local_model("coder-7b", 64, MLX)
                .unwrap()
                .display(),
            "coder-7b",
            "single-form families display without a quant suffix"
        );
        assert!(resolve_team_local_model("pipe/glm-4-flash", 64, MLX).is_none());
        assert!(resolve_team_local_model("qwen3-anything", 64, MLX).is_none());
    }

    #[test]
    fn specs_map_backend_forms_and_cuda_gaps_are_honest() {
        let resolved = resolve_team_local_model("coder-7b", 64, MLX).unwrap();
        let mlx = team_model_spec(resolved, LocalBackend::Mlx).unwrap();
        assert!(mlx.repo.starts_with("mlx-community/"));
        assert_eq!(mlx.gguf_file, None);
        let cuda = team_model_spec(resolved, LocalBackend::Cuda).unwrap();
        assert!(cuda.gguf_file.is_some());

        let laguna = resolve_team_local_model("laguna-s", 128, MLX).unwrap();
        let spec = team_model_spec(laguna, LocalBackend::Mlx).unwrap();
        assert_eq!(
            spec.repo, "pipenetwork/Laguna-S-2.1-MLX-4bit",
            "the spec serves exactly the quant the resolution chose"
        );
        assert!(
            team_model_spec(laguna, LocalBackend::Cuda).is_err(),
            "no guessed GGUF repos: CUDA gap is an honest error"
        );
    }

    #[test]
    fn deepseek_coder_v2_lite_is_a_selectable_mlx_runtime() {
        let resolved = resolve_team_local_model("deepseek-coder-v2-lite", 96, MLX).unwrap();
        let quant = resolved.mlx.unwrap();
        assert_eq!(quant.quant, "4bit");
        assert_eq!(
            quant.repo,
            "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
        );
        let runtime = local_runtime_spec("deepseek-coder-v2-lite", 96, LocalBackend::Mlx).unwrap();
        assert_eq!(runtime.backend, LocalBackend::Mlx);
        assert_eq!(runtime.model_id, quant.model_id);
        assert!(runtime.profile_name.starts_with("mlx-"));
    }

    #[test]
    fn live_pipenetwork_repository_is_a_selectable_mlx_runtime() {
        let runtime = local_runtime_spec(
            "pipenetwork/DeepSeek-V4-Flash-MLX-4bit",
            96,
            LocalBackend::Mlx,
        )
        .unwrap();
        assert_eq!(runtime.repo, "pipenetwork/DeepSeek-V4-Flash-MLX-4bit");
        assert_eq!(runtime.model_id, "DeepSeek-V4-Flash-MLX-4bit");
        assert_eq!(runtime.backend, LocalBackend::Mlx);
    }

    #[test]
    fn pipenetwork_catalog_accepts_tagged_mlx_repos_and_rejects_unsupported_pipelines() {
        assert!(is_supported_pipenetwork_model(
            "pipenetwork/LongCat-2.0-2bit",
            Some("text-generation"),
            &["mlx".into(), "text-generation".into()]
        ));
        assert!(is_supported_pipenetwork_model(
            "pipenetwork/MiniMax-M3-MLX-3bit",
            None,
            &["mlx".into(), "text-generation".into()]
        ));
        assert!(!is_supported_pipenetwork_model(
            "pipenetwork/MiniMax-H3-MLX-4bit",
            Some("image-text-to-video"),
            &["mlx".into(), "image-text-to-video".into()]
        ));
        assert!(!is_supported_pipenetwork_model(
            "pipenetwork/Qwen3.6-35B-A3B-mlx-nvfp4",
            Some("text-generation"),
            &["mlx".into(), "text-generation".into(), "nvfp4".into()]
        ));
    }

    #[test]
    fn pipenetwork_quant_labels_include_pruned_and_mixed_variants() {
        assert_eq!(
            quantization_for_repo("pipenetwork/Foo-MLX-REAP50"),
            "reap50"
        );
        assert_eq!(
            quantization_for_repo("pipenetwork/Foo-MLX-mixed-3_6bit"),
            "mixed-3_6bit"
        );
        assert_eq!(
            display_name_for_repo(
                "Pipe Network models",
                "pipenetwork/DeepSeek-V4-Flash-MLX-REAP50",
                "reap50"
            ),
            "DeepSeek-V4-Flash · reap50 · Pipe Network models"
        );
    }

    #[test]
    fn model_present_rejects_partial_downloads() {
        let dir = std::env::temp_dir().join(format!("hi-present-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let spec = LocalModelSpec {
            repo: "x/y".into(),
            model_id: "y".into(),
            gguf_file: None,
            backend: LocalBackend::Mlx,
        };
        assert!(!model_present(&dir, &spec), "empty dir");
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        assert!(!model_present(&dir, &spec), "config alone is not a model");
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let valid_shard = || {
            let header = b"{}";
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes
        };
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), valid_shard()).unwrap();
        assert!(
            !model_present(&dir, &spec),
            "a shard the index names is missing"
        );
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), valid_shard()).unwrap();
        assert!(model_present(&dir, &spec), "all shards present");
        std::fs::write(dir.join("model-00002-of-00002.safetensors.aria2"), "ctl").unwrap();
        assert!(
            !model_present(&dir, &spec),
            "an aria2 control file means the download isn't done"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn meminfo_parses() {
        assert_eq!(
            parse_meminfo_total_kb("MemTotal:       16332852 kB\nMemFree: 1 kB"),
            Some(16_332_852)
        );
        assert_eq!(parse_meminfo_total_kb("nope"), None);
    }
}

/// Health deadline scaled to the model on disk: loading weights into memory
/// dominates startup, so allow ~15s per GiB on top of a 60s floor, capped at
/// ten minutes. A 3B review model waits ~90s; a 19 GB 32B executor gets the
/// minutes it genuinely needs instead of failing at a flat 15s.
pub fn health_deadline_for_model(model_bytes: u64) -> std::time::Duration {
    const FLOOR_SECS: u64 = 60;
    const PER_GIB_SECS: u64 = 15;
    const CAP_SECS: u64 = 600;
    let gib = model_bytes / (1024 * 1024 * 1024);
    std::time::Duration::from_secs((FLOOR_SECS + gib * PER_GIB_SECS).min(CAP_SECS))
}

/// Total bytes directly inside `dir` (model repos download flat).
fn model_dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| entry.metadata().ok())
                .filter(|meta| meta.is_file())
                .map(|meta| meta.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Whether `name` resolves to an executable on PATH.
fn binary_on_path(name: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| {
        let candidate = dir.join(name);
        candidate.is_file()
    })
}

/// Locate the `hi-local` serving binary, building it when running from a
/// development checkout. Full automation means nobody gets told to go run
/// cargo by hand: if the binary is missing but the current executable lives
/// under a cargo `target/` directory, build `hi-local` for that same profile
/// (quietly — output captured) and use the fresh sibling.
pub async fn ensure_hi_local_binary(backend: LocalBackend) -> Result<PathBuf> {
    ensure_hi_local_binary_with_progress(backend, None).await
}

/// The cargo invocation that builds `hi-local` with `backend` compiled in.
/// Pure for testing — getting this wrong shipped a binary that couldn't
/// serve MLX at all.
pub fn hi_local_build_args(profile: &str, backend: LocalBackend) -> Vec<String> {
    let mut args = vec![
        "build".to_string(),
        "-p".to_string(),
        "hi-local".to_string(),
        "--features".to_string(),
        backend.cargo_feature().to_string(),
    ];
    if profile == "release" {
        args.push("--release".to_string());
    }
    args
}

/// [`ensure_hi_local_binary`], announcing a dev-checkout compile on
/// `progress`. In a development checkout (`target/<profile>/hi`) this
/// ALWAYS runs cargo with the backend feature — a stale or feature-less
/// sibling binary must never be served (cargo is a fast no-op when the
/// build is fresh). Installed layouts trust the sibling/PATH binary.
pub async fn ensure_hi_local_binary_with_progress(
    backend: LocalBackend,
    progress: Option<&tokio::sync::watch::Sender<ProvisionPhase>>,
) -> Result<PathBuf> {
    // Development checkout: <workspace>/target/<profile>/hi → (re)build the
    // serving binary with the right backend feature, into the same profile.
    if let Ok(current) = std::env::current_exe()
        && let Some(profile_dir) = current.parent()
        && let Some(target_dir) = profile_dir.parent()
        && target_dir.file_name().is_some_and(|name| name == "target")
        && let Some(workspace) = target_dir.parent()
    {
        if let Some(progress) = progress {
            let _ = progress.send(ProvisionPhase::BuildingServer);
        }
        let profile = profile_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("release");
        let args = hi_local_build_args(profile, backend);
        let output = tokio::process::Command::new("cargo")
            .args(&args)
            .current_dir(workspace)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .await
            .context("building hi-local from the development checkout")?;
        if !output.status.success() {
            let tail: String = String::from_utf8_lossy(&output.stderr)
                .chars()
                .rev()
                .take(400)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            bail!(
                "building hi-local (--features {}) failed: {tail}",
                backend.cargo_feature()
            );
        }
        let built = profile_dir.join(format!("hi-local{}", std::env::consts::EXE_SUFFIX));
        if built.exists() {
            return Ok(built);
        }
    }
    let bin = find_hi_local();
    if bin.components().count() > 1 {
        if bin.exists() {
            return Ok(bin);
        }
    } else if binary_on_path(&bin.to_string_lossy()) {
        return Ok(bin);
    }
    bail!(
        "the hi-local serving binary isn't available (not beside hi, not on PATH); reinstall hi with its local-serving component"
    )
}

#[cfg(test)]
mod provisioning_support_tests {
    use super::*;

    #[test]
    fn health_deadline_scales_with_model_size() {
        let gib = 1024 * 1024 * 1024;
        assert_eq!(health_deadline_for_model(0).as_secs(), 60);
        assert_eq!(health_deadline_for_model(2 * gib).as_secs(), 90);
        assert_eq!(health_deadline_for_model(19 * gib).as_secs(), 345);
        assert_eq!(
            health_deadline_for_model(200 * gib).as_secs(),
            600,
            "capped at ten minutes"
        );
    }

    #[test]
    fn build_args_always_carry_the_backend_feature() {
        let release = hi_local_build_args("release", LocalBackend::Mlx);
        assert!(release.contains(&"--features".to_string()));
        assert!(release.contains(&"mlx".to_string()));
        assert!(release.contains(&"--release".to_string()));
        let debug = hi_local_build_args("debug", LocalBackend::Cuda);
        assert!(debug.contains(&"native-cuda".to_string()));
        assert!(!debug.contains(&"--release".to_string()));
    }

    #[test]
    fn binary_on_path_finds_sh_and_rejects_nonsense() {
        assert!(binary_on_path("sh"));
        assert!(!binary_on_path("hi-definitely-not-a-real-binary-xyz"));
    }

    #[test]
    fn resident_note_parsing_uses_the_size_before_resident() {
        assert_eq!(
            parse_last_size("25.3 GB download, 11.5 GB "),
            Some(11 * 1024 * 1024 * 1024 + 512 * 1024 * 1024)
        );
        assert_eq!(parse_last_size("162 GB"), Some(162 * 1024 * 1024 * 1024));
        assert_eq!(parse_last_size("near-lossless (+2.3% PPL)"), None);
        assert_eq!(
            advertised_resident_bytes(Some(
                "162 GB · ppl 6.3005 · superseded by mixed-4_8bit — 3 GB more",
            )),
            None,
            "download/PPL notes must fall back to the measured file total"
        );
        assert_eq!(
            advertised_resident_bytes(Some("25.3 GB download, 11.5 GB resident.")),
            Some(11 * 1024 * 1024 * 1024 + 512 * 1024 * 1024)
        );
    }
}
