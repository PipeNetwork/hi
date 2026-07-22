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

/// Probe the host for a usable local backend.
pub fn detect_backend() -> Option<LocalBackend> {
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
async fn detect_backend_offload() -> Option<LocalBackend> {
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
    match &spec.gguf_file {
        // MLX: a loadable model directory carries a config.json (matches `/hf`).
        None => dir.join("config.json").exists(),
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
        let bin = find_hi_local();
        let host = "127.0.0.1";
        let port = pick_free_port();
        let args = serve_args(&model_path, &spec, host, port);
        let handle = hi_tools::start_local_server(&bin, &args, host, port)
            .await
            .with_context(|| {
                format!(
                    "starting hi-local ({}) — is the `hi-local` binary built with the {} backend?",
                    bin.display(),
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
        hi_tools::stop_local_server(&state.process_id);
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
