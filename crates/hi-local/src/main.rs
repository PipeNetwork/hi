// The server-launch and generation entry points take many tuning parameters by
// design; bundling them would only move the argument list elsewhere.
#![allow(clippy::too_many_arguments)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use hi_local_core::InferenceBackend;

#[derive(Parser, Debug)]
#[command(name = "hi-local", version, about = "Native local sidecar for hi")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Inspect a local GGUF file or MLX model directory.
    Inspect {
        model_path: PathBuf,
        #[arg(long)]
        model_id: Option<String>,
    },
    /// Serve a local model through an OpenAI-compatible API.
    Serve {
        model_path: PathBuf,
        #[arg(long, value_enum)]
        backend: BackendKind,
        #[arg(long, value_enum)]
        execution: Option<CudaExecutionKind>,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long)]
        model_id: Option<String>,
        #[arg(long, default_value_t = 8)]
        max_batch_size: usize,
        #[arg(long)]
        max_active_requests: Option<usize>,
        #[arg(long, default_value_t = 8192)]
        max_batched_tokens: usize,
        #[arg(long, default_value_t = 2000)]
        max_wait_us: u64,
        #[arg(long, value_enum, default_value_t = CudaKvCacheModeKind::Paged)]
        kv_cache_mode: CudaKvCacheModeKind,
        #[arg(long, default_value_t = 16)]
        kv_page_size: usize,
        #[arg(long)]
        mmproj_path: Option<PathBuf>,
        #[arg(long)]
        allow_http_image_url: bool,
        #[arg(long)]
        allow_local_image_url: bool,
    },
    /// Run the dense Qwen GGUF CPU reference path for CUDA parity checks.
    QwenCpu {
        model_path: PathBuf,
        #[arg(long, conflicts_with = "prompt")]
        tokens: Option<String>,
        #[arg(long, conflicts_with = "tokens")]
        prompt: Option<String>,
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        include_logits: bool,
        /// Run the GPU provider instead of the CPU reference (deepseek4 only;
        /// requires a native-cuda build). Same flags and output contract.
        #[arg(long)]
        gpu: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendKind {
    Cuda,
    Mlx,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CudaExecutionKind {
    Gpu,
    CpuReference,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CudaKvCacheModeKind {
    Legacy,
    Paged,
}

impl From<CudaExecutionKind> for hi_cuda::CudaExecution {
    fn from(value: CudaExecutionKind) -> Self {
        match value {
            CudaExecutionKind::Gpu => Self::Gpu,
            CudaExecutionKind::CpuReference => Self::CpuReference,
        }
    }
}

impl From<CudaKvCacheModeKind> for hi_cuda::CudaKvCacheMode {
    fn from(value: CudaKvCacheModeKind) -> Self {
        match value {
            CudaKvCacheModeKind::Legacy => Self::Legacy,
            CudaKvCacheModeKind::Paged => Self::Paged,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hi_local=info,hi_cuda=info".into()),
        )
        .init();
    run(Cli::parse()).await
}

async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Inspect {
            model_path,
            model_id,
        } => inspect(&model_path, model_id),
        Command::Serve {
            model_path,
            backend,
            execution,
            host,
            port,
            model_id,
            max_batch_size,
            max_active_requests,
            max_batched_tokens,
            max_wait_us,
            kv_cache_mode,
            kv_page_size,
            mmproj_path,
            allow_http_image_url,
            allow_local_image_url,
        } => {
            serve(
                model_path,
                backend,
                execution,
                host,
                port,
                model_id,
                max_batch_size,
                max_active_requests,
                max_batched_tokens,
                max_wait_us,
                kv_cache_mode,
                kv_page_size,
                mmproj_path,
                allow_http_image_url,
                allow_local_image_url,
            )
            .await
        }
        Command::QwenCpu {
            model_path,
            tokens,
            prompt,
            max_tokens,
            top_k,
            temperature,
            top_p,
            seed,
            include_logits,
            gpu,
        } => qwen_cpu(
            model_path,
            tokens,
            prompt,
            max_tokens,
            top_k,
            temperature,
            top_p,
            seed,
            include_logits,
            gpu,
        ),
    }
}

fn inspect(model_path: &Path, model_id: Option<String>) -> Result<()> {
    if is_gguf_path(model_path) {
        let gguf = hi_gguf::GgufFile::open(model_path)?;
        let summary = gguf.summary()?;
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    if model_path.is_dir() {
        #[cfg(feature = "mlx")]
        {
            let info = hi_mlx::manifest::inspect_model(model_path, model_id)?;
            println!("{}", serde_json::to_string_pretty(&info)?);
            return Ok(());
        }
        #[cfg(not(feature = "mlx"))]
        {
            let _ = &model_id;
            bail!(
                "inspecting an MLX model directory requires the `mlx` feature; \
                 rebuild hi-local with --features mlx (Apple Silicon macOS only)"
            );
        }
    }
    bail!(
        "unsupported model path {}; expected a .gguf file or an MLX model directory",
        model_path.display()
    )
}

fn qwen_cpu(
    model_path: PathBuf,
    tokens: Option<String>,
    prompt: Option<String>,
    max_tokens: usize,
    top_k: usize,
    temperature: f32,
    top_p: f32,
    seed: Option<u64>,
    include_logits: bool,
    gpu: bool,
) -> Result<()> {
    if !is_gguf_path(&model_path) {
        bail!(
            "Qwen CPU reference expects a .gguf model file, got {}",
            model_path.display()
        );
    }
    let options = hi_cuda::qwen_cpu::QwenCpuRunOptions {
        max_tokens,
        top_k,
        temperature,
        top_p,
        seed,
        include_logits,
    };
    let input = match (tokens, prompt) {
        (Some(tokens), None) => {
            let tokens = parse_token_ids(&tokens)?;
            if tokens.is_empty() {
                bail!("--tokens must include at least one token id");
            }
            CpuReferenceInput::Tokens(tokens)
        }
        (None, Some(prompt)) => CpuReferenceInput::Prompt(prompt),
        (None, None) => bail!("provide either --tokens or --prompt"),
        (Some(_), Some(_)) => bail!("provide only one of --tokens or --prompt"),
    };
    // deepseek4 uses its own CPU reference (hyper-connections, latent MQA,
    // compressed KV) and, under --gpu, the host-orchestrated CUDA engine;
    // everything else keeps the existing Qwen path.
    let gguf = hi_gguf::GgufFile::open(&model_path)
        .with_context(|| format!("loading GGUF model from {}", model_path.display()))?;
    let is_deepseek4 = gguf.qwen_config()?.is_deepseek4();
    if gpu && !is_deepseek4 {
        bail!("--gpu is currently supported only for deepseek4 GGUFs (DeepSeek-V4-Flash)");
    }
    let output = if gpu {
        run_dsv4_gpu(gguf, &model_path, &input, options)?
    } else if is_deepseek4 {
        let model =
            hi_cuda::dsv4_cpu::DeepSeekV4CpuReference::from_gguf(gguf).with_context(|| {
                format!(
                    "loading DeepSeek-V4 GGUF model from {}",
                    model_path.display()
                )
            })?;
        match &input {
            CpuReferenceInput::Tokens(tokens) => model.run_tokens(tokens, options)?,
            CpuReferenceInput::Prompt(prompt) => model.run_prompt(prompt, options)?,
        }
    } else {
        let model = hi_cuda::qwen_cpu::QwenCpuReference::from_gguf(&gguf)
            .with_context(|| format!("loading Qwen GGUF model from {}", model_path.display()))?;
        match &input {
            CpuReferenceInput::Tokens(tokens) => model.run_tokens(tokens, options)?,
            CpuReferenceInput::Prompt(prompt) => model.run_prompt(prompt, options)?,
        }
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

#[cfg(feature = "native-cuda")]
fn run_dsv4_gpu(
    gguf: hi_gguf::GgufFile,
    model_path: &Path,
    input: &CpuReferenceInput,
    options: hi_cuda::qwen_cpu::QwenCpuRunOptions,
) -> Result<hi_cuda::qwen_cpu::QwenCpuRunOutput> {
    let model = hi_cuda::dsv4_gpu::DeepSeekV4GpuEngine::from_gguf(gguf).with_context(|| {
        format!(
            "loading DeepSeek-V4 GGUF model onto the GPU from {}",
            model_path.display()
        )
    })?;
    Ok(match input {
        CpuReferenceInput::Tokens(tokens) => model.run_tokens(tokens, options)?,
        CpuReferenceInput::Prompt(prompt) => model.run_prompt(prompt, options)?,
    })
}

#[cfg(not(feature = "native-cuda"))]
fn run_dsv4_gpu(
    _gguf: hi_gguf::GgufFile,
    _model_path: &Path,
    _input: &CpuReferenceInput,
    _options: hi_cuda::qwen_cpu::QwenCpuRunOptions,
) -> Result<hi_cuda::qwen_cpu::QwenCpuRunOutput> {
    bail!(
        "--gpu requires a native-cuda build of hi-local; rebuild with \
         --features native-cuda and a CUDA Toolkit installation"
    )
}

enum CpuReferenceInput {
    Tokens(Vec<u32>),
    Prompt(String),
}

async fn serve(
    model_path: PathBuf,
    backend: BackendKind,
    execution: Option<CudaExecutionKind>,
    host: String,
    port: u16,
    model_id: Option<String>,
    max_batch_size: usize,
    max_active_requests: Option<usize>,
    max_batched_tokens: usize,
    max_wait_us: u64,
    kv_cache_mode: CudaKvCacheModeKind,
    kv_page_size: usize,
    mmproj_path: Option<PathBuf>,
    allow_http_image_url: bool,
    allow_local_image_url: bool,
) -> Result<()> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("invalid listen address {host}:{port}"))?;

    match backend {
        BackendKind::Cuda => {
            if !is_gguf_path(&model_path) {
                bail!(
                    "CUDA backend expects a .gguf model file, got {}",
                    model_path.display()
                );
            }
            let execution = execution
                .map(hi_cuda::CudaExecution::from)
                .unwrap_or_else(hi_cuda::CudaExecution::default_for_build);
            let config = hi_cuda::CudaBackendConfig {
                max_batch_size,
                max_active_requests: max_active_requests.unwrap_or(max_batch_size),
                max_batched_tokens,
                max_wait_us,
                kv_cache_mode: kv_cache_mode.into(),
                kv_page_size,
                mmproj_path,
            };
            let backend = Arc::new(
                hi_cuda::CudaBackend::load_with_config(&model_path, model_id, execution, config)
                    .with_context(|| format!("loading GGUF model from {}", model_path.display()))?,
            );
            let listener = bind_server_listener(addr).await?;
            tracing::info!(
                "serving {} on http://{addr}/v1 with cuda execution={}",
                backend.model().id,
                backend.execution().label()
            );
            axum::serve(
                listener,
                hi_local_core::server::app_with_config(
                    backend,
                    hi_local_core::server::ServerConfig {
                        image_url_policy: hi_local_core::server::ImageUrlPolicy {
                            allow_http_urls: allow_http_image_url,
                            allow_local_urls: allow_local_image_url,
                        },
                    },
                ),
            )
            .await?;
        }
        BackendKind::Mlx => {
            if execution.is_some() {
                bail!("--execution is only supported with --backend cuda");
            }
            if mmproj_path.is_some() {
                bail!("--mmproj-path is only supported with --backend cuda");
            }
            #[cfg(not(feature = "mlx"))]
            {
                bail!(
                    "the MLX backend is not compiled in; rebuild hi-local with \
                     --features mlx (Apple Silicon macOS only)"
                );
            }
            #[cfg(feature = "mlx")]
            {
                if !hi_mlx::backend::platform_supported() {
                    bail!("MLX inference requires Apple Silicon macOS");
                }
                let backend = Arc::new(
                    hi_mlx::backend::MlxBackend::load(&model_path, model_id).with_context(
                        || format!("loading MLX model from {}", model_path.display()),
                    )?,
                );
                let listener = bind_server_listener(addr).await?;
                tracing::info!("serving {} on http://{addr}/v1", backend.model().id);
                axum::serve(
                    listener,
                    hi_local_core::server::app_with_config(
                        backend,
                        hi_local_core::server::ServerConfig {
                            image_url_policy: hi_local_core::server::ImageUrlPolicy {
                                allow_http_urls: allow_http_image_url,
                                allow_local_urls: allow_local_image_url,
                            },
                        },
                    ),
                )
                .await?;
            }
        }
    }
    Ok(())
}

async fn bind_server_listener(addr: SocketAddr) -> Result<tokio::net::TcpListener> {
    tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))
}

fn is_gguf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

fn parse_token_ids(value: &str) -> Result<Vec<u32>> {
    value
        .split(|ch: char| ch == ',' || ch.is_ascii_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u32>()
                .with_context(|| format!("invalid token id {part:?}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn serve_validates_cuda_config_before_binding_socket() {
        let occupied = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = occupied.local_addr().unwrap().port();

        let err = serve(
            PathBuf::from("missing.gguf"),
            BackendKind::Cuda,
            Some(CudaExecutionKind::CpuReference),
            "127.0.0.1".to_string(),
            port,
            None,
            0,
            None,
            8192,
            2000,
            CudaKvCacheModeKind::Paged,
            16,
            None,
            false,
            false,
        )
        .await
        .unwrap_err();
        let err = format!("{err:#}");

        assert!(err.contains("--max-batch-size must be greater than zero"));
        assert!(!err.contains("binding"));
    }

    #[tokio::test]
    async fn serve_validates_mlx_options_before_binding_socket() {
        let occupied = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = occupied.local_addr().unwrap().port();

        let err = serve(
            PathBuf::from("model-dir"),
            BackendKind::Mlx,
            Some(CudaExecutionKind::Gpu),
            "127.0.0.1".to_string(),
            port,
            None,
            8,
            None,
            8192,
            2000,
            CudaKvCacheModeKind::Paged,
            16,
            None,
            false,
            false,
        )
        .await
        .unwrap_err();
        let err = format!("{err:#}");

        assert!(err.contains("--execution is only supported with --backend cuda"));
        assert!(!err.contains("binding"));
    }
}
