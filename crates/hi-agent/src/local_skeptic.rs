//! Auto-managed local model for the `/goal` skeptic review (`/config
//! skeptic-local on`).
//!
//! The skeptic gate runs a bounded, fail-open critique call before a turn may
//! advance a sub-goal. It fires often, so routing it to a small local model
//! keeps the coding driver and planner on the main model while making the review
//! free and private. Turning the feature on detects the machine's
//! local-inference backend (Apple-Silicon MLX or NVIDIA CUDA), fetches a small
//! default review model if it isn't already cached, launches a `hi-local`
//! server, waits for it to become healthy, and points
//! `skeptic_endpoint`/`skeptic_model` at it. Every step degrades gracefully: a
//! missing backend, missing binary, failed download, or unhealthy server leaves
//! the skeptic on the main provider and reports why.

use crate::Agent;
use anyhow::{Context, Result, bail};
use hi_ai::Provider;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    tokio::task::spawn_blocking(detect_backend)
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
    // An in-flight or interrupted fetch leaves `<file>.aria2` control files
    // behind; their presence means the weights can't be trusted yet, no
    // matter what else already landed.
    if aria2_remnants(dir) {
        return false;
    }
    match &spec.gguf_file {
        // MLX: a loadable model directory carries a config.json (matches
        // `/hf`) — and when the repo ships a safetensors index, every weight
        // shard it names. config.json downloads first, so checking it alone
        // would bless a directory whose later shards never started.
        None => {
            if !dir.join("config.json").exists() {
                return false;
            }
            match indexed_weight_files(dir) {
                Some(files) => files.iter().all(|file| dir.join(file).exists()),
                None => any_safetensors(dir),
            }
        }
        // CUDA: the specific GGUF file must be on disk.
        Some(file) => dir.join(file).exists(),
    }
}

/// Whether `dir` holds any aria2c control files (partial downloads).
fn aria2_remnants(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries
            .flatten()
            .any(|entry| entry.path().extension().is_some_and(|ext| ext == "aria2"))
    })
}

/// The unique weight files named by `model.safetensors.index.json`, when the
/// repo ships one (multi-shard models do).
fn indexed_weight_files(dir: &Path) -> Option<Vec<String>> {
    let raw = std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?;
    let index: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let map = index.get("weight_map")?.as_object()?;
    let mut files: Vec<String> = map
        .values()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect();
    files.sort();
    files.dedup();
    Some(files)
}

fn any_safetensors(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        })
    })
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
    prev_skeptic_model: Option<String>,
    prev_endpoint: Option<String>,
    prev_endpoint_key: Option<String>,
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
        self.local_skeptic.as_ref().map(|s| s.endpoint.as_str())
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
        if let Some(state) = &self.local_skeptic {
            return Ok(LocalSkepticOutcome::Ready {
                endpoint: state.endpoint.clone(),
                model_id: state.model_id.clone(),
            });
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
        true
    }

    /// Stop any auto-managed local skeptic server without touching config, for
    /// session shutdown. Called from [`Agent::kill_background_processes`].
    pub(crate) fn stop_local_skeptic_server(&self) {
        if let Some(state) = &self.local_skeptic {
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
    pub repo: &'static str,
    pub model_id: &'static str,
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
                repo: "pipenetwork/Laguna-S-2.1-MLX-8bit",
                model_id: "Laguna-S-2.1-MLX-8bit",
            },
            MlxQuant {
                quant: "6bit",
                min_ram_gb: 192,
                repo: "pipenetwork/Laguna-S-2.1-MLX-6bit",
                model_id: "Laguna-S-2.1-MLX-6bit",
            },
            MlxQuant {
                quant: "4bit",
                min_ram_gb: 96,
                repo: "pipenetwork/Laguna-S-2.1-MLX-4bit",
                model_id: "Laguna-S-2.1-MLX-4bit",
            },
            MlxQuant {
                quant: "3bit",
                min_ram_gb: 96,
                repo: "pipenetwork/Laguna-S-2.1-MLX-3bit",
                model_id: "Laguna-S-2.1-MLX-3bit",
            },
            MlxQuant {
                quant: "2bit",
                min_ram_gb: 64,
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
                repo: "pipenetwork/NVIDIA-Nemotron-3-Nano-4B-MLX-8bit",
                model_id: "NVIDIA-Nemotron-3-Nano-4B-MLX-8bit",
            },
            MlxQuant {
                quant: "4bit",
                min_ram_gb: 6,
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
                repo: "mlx-community/DeepSeek-V4-Flash-8bit",
                model_id: "DeepSeek-V4-Flash-8bit",
            },
            MlxQuant {
                quant: "4bit",
                min_ram_gb: 256,
                repo: "mlx-community/DeepSeek-V4-Flash-4bit",
                model_id: "DeepSeek-V4-Flash-4bit",
            },
            MlxQuant {
                quant: "3bit",
                min_ram_gb: 192,
                repo: "mlx-community/DeepSeek-V4-Flash-3bit-DQ",
                model_id: "DeepSeek-V4-Flash-3bit-DQ",
            },
            MlxQuant {
                quant: "2bit",
                min_ram_gb: 128,
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
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), "w").unwrap();
        assert!(
            !model_present(&dir, &spec),
            "a shard the index names is missing"
        );
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), "w").unwrap();
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
}
