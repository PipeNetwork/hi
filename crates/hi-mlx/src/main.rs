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
        } => {
            if !platform_supported() {
                bail!("MLX inference requires Apple Silicon macOS");
            }
            let backend = Arc::new(
                MlxBackend::load(&model_path, model_id)
                    .with_context(|| format!("loading MLX model from {}", model_path.display()))?,
            );
            let addr: SocketAddr = format!("{host}:{port}")
                .parse()
                .with_context(|| format!("invalid listen address {host}:{port}"))?;
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .with_context(|| format!("binding {addr}"))?;
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
    }
    Ok(())
}
