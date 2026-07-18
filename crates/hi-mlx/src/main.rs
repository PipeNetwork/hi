use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use hi_mlx::backend::{InferenceBackend, MlxBackend, platform_supported};
use hi_mlx::manifest::{inspect_model, list_models};

#[derive(Parser, Debug)]
#[command(name = "hi-mlx", version, about = "Native MLX sidecar for hi")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Serve one local MLX model through an OpenAI-compatible API.
    Serve {
        model_path: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long)]
        model_id: Option<String>,
        /// Optional draft model for greedy speculative decoding (must share the target's tokenizer).
        #[arg(long)]
        draft: Option<PathBuf>,
        /// Speculative decoding lookahead (drafts proposed per round).
        #[arg(long, default_value_t = 3)]
        spec_k: usize,
    },
    /// Inspect one local MLX model directory.
    Inspect {
        model_path: PathBuf,
        #[arg(long)]
        model_id: Option<String>,
    },
    /// List model directories under a root, usually .hi/models.
    List {
        #[arg(default_value = ".hi/models")]
        root: PathBuf,
    },
    /// Benchmark greedy speculative decoding (draft proposes, target verifies) vs target-only greedy.
    Spec {
        target_path: PathBuf,
        draft_path: PathBuf,
        #[arg(long, default_value = "Write a short paragraph about the ocean.")]
        prompt: String,
        #[arg(long, default_value_t = 200)]
        max_tokens: u32,
        #[arg(long, default_value_t = 4)]
        k: usize,
    },
    /// Repack shard files so expert slabs are contiguous on disk, reducing
    /// seek overhead during streaming MoE inference. Writes a new set of
    /// shard files to `<model_path>/repacked/` and updates the index.
    Repack {
        model_path: PathBuf,
        /// Output directory (defaults to `<model_path>/repacked/`).
        #[arg(long)]
        output: Option<PathBuf>,
        /// Target shard size in GB (default 4, matching typical MLX shards).
        #[arg(long, default_value_t = 4)]
        shard_size_gb: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hi_mlx=info".into()),
        )
        .init();
    run(Cli::parse()).await
}

async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Serve {
            model_path,
            host,
            port,
            model_id,
            draft,
            spec_k,
        } => {
            if !platform_supported() {
                bail!("MLX inference requires Apple Silicon macOS");
            }
            let backend = Arc::new(
                MlxBackend::load_with_draft(&model_path, model_id, draft.as_ref(), spec_k)
                    .with_context(|| format!("loading MLX model from {}", model_path.display()))?,
            );
            let addr: SocketAddr = format!("{host}:{port}")
                .parse()
                .with_context(|| format!("invalid listen address {host}:{port}"))?;
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .with_context(|| format!("binding {addr}"))?;
            if let Some(draft) = &draft {
                tracing::info!(
                    "speculative decoding enabled (draft {}, k={spec_k})",
                    draft.display()
                );
            }
            tracing::info!("serving {} on http://{addr}/v1", backend.model().id);
            axum::serve(listener, hi_mlx::server::app(backend)).await?;
        }
        Command::Inspect {
            model_path,
            model_id,
        } => {
            let info = inspect_model(model_path, model_id)?;
            println!("{}", serde_json::to_string_pretty(&info)?);
        }
        Command::List { root } => {
            let models = list_models(root)?;
            if models.is_empty() {
                println!("No local MLX models found.");
            } else {
                for model in models {
                    let mark = if model.supported { "ok" } else { "unsupported" };
                    println!("{mark}\t{}\t{}", model.path.display(), model.summary);
                }
            }
        }
        Command::Spec {
            target_path,
            draft_path,
            prompt,
            max_tokens,
            k,
        } => {
            if !platform_supported() {
                bail!("MLX inference requires Apple Silicon macOS");
            }
            use hi_mlx::backend::GenerationRequest;
            use hi_mlx::models::NativeRuntime;
            let req = |p: &str| GenerationRequest {
                prompt: p.to_string(),
                max_tokens,
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                seed: None,
                stop_sequences: vec![],
                media_inputs: vec![],
            };
            let mut target = NativeRuntime::from_path(&target_path)
                .with_context(|| format!("loading target {}", target_path.display()))?;
            let mut draft = NativeRuntime::from_path(&draft_path)
                .with_context(|| format!("loading draft {}", draft_path.display()))?;

            let t0 = std::time::Instant::now();
            let (spec_out, stats) =
                target.speculative_generate(&mut draft, req(&prompt), k, |_| Ok(()))?;
            let spec_dt = t0.elapsed().as_secs_f64();
            let spec_tps = spec_out.completion_tokens as f64 / spec_dt.max(1e-6);
            let accept_rate = stats.accepted as f64 / stats.proposed.max(1) as f64 * 100.0;
            let avg_per_round = spec_out.completion_tokens as f64 / stats.rounds.max(1) as f64;

            let t1 = std::time::Instant::now();
            let base_out = target.generate(req(&prompt))?;
            let base_dt = t1.elapsed().as_secs_f64();
            let base_tps = base_out.completion_tokens as f64 / base_dt.max(1e-6);

            println!("=== speculative (k={k}) ===\n{}", spec_out.text.trim());
            println!(
                "\n--- speculative: {} tok, {spec_dt:.1}s => {spec_tps:.1} tok/s | \
                 accept {accept_rate:.0}% | {avg_per_round:.2} tok/round over {} rounds ---",
                spec_out.completion_tokens, stats.rounds
            );
            println!(
                "--- target greedy: {} tok, {base_dt:.1}s => {base_tps:.1} tok/s ---",
                base_out.completion_tokens
            );
            println!(
                "=== SPEEDUP {:.2}x  |  output identical to greedy: {} ===",
                spec_tps / base_tps.max(1e-6),
                spec_out.text == base_out.text
            );
        }
        Command::Repack {
            model_path,
            output,
            shard_size_gb,
        } => {
            if !platform_supported() {
                bail!("MLX repack requires Apple Silicon macOS");
            }
            let output_dir = output.unwrap_or_else(|| model_path.join("repacked"));
            tracing::info!(
                "repacking {} -> {} (shard_size={}GB)",
                model_path.display(),
                output_dir.display(),
                shard_size_gb
            );
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
            hi_mlx::repack::repack_model(&model_path, &output_dir, shard_size_gb)?;
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
            bail!("MLX repack requires Apple Silicon macOS");
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
            tracing::info!("repack complete: {}", output_dir.display());
        }
    }
    Ok(())
}
