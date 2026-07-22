//! Auto-managed local model server for the `/goal` skeptic review.
//!
//! Spawns `hi-local serve …` into a private background registry, waits for its
//! `/health` endpoint to report ready, and hands back the OpenAI-compatible base
//! URL. The policy around it — which backend, which default model, when to turn
//! it on — lives in `hi-agent`'s `local_skeptic` module; this file owns only the
//! process and HTTP mechanics, mirroring the proven `/hf run --mlx` path.

use anyhow::{Result, anyhow, bail};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;

// Private registry so these servers stay isolated from any workspace runtime and
// can be stopped by id when the user turns the feature off. Like the `/hf`
// sidecar registry, entries are never adopted by an agent's tool registry.
static LOCAL_SERVERS: LazyLock<crate::BackgroundRegistry> =
    LazyLock::new(crate::BackgroundRegistry::default);

/// A running local model server routed to the skeptic review.
pub struct LocalServerHandle {
    /// Background-registry handle, used to stop the server later.
    pub process_id: String,
    /// OpenAI-compatible base URL (e.g. `http://127.0.0.1:8080/v1`).
    pub endpoint: String,
}

/// Cache directory for a downloaded local model, matching `/hf run --mlx`'s
/// layout (`$HI_MLX_MODELS_DIR` or `./.hi/models`, repo id sanitized) so a model
/// fetched by either path is reused rather than downloaded twice. Uses the
/// `main` revision (no `@rev` suffix).
pub fn skeptic_model_dir(repo_id: &str) -> PathBuf {
    let root = std::env::var_os("HI_MLX_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".hi").join("models"));
    root.join(crate::hf::safe_path(repo_id))
}

/// Spawn `bin serve <args…>` in the background and wait for `/health` to report
/// ready. On failure the process is killed and its captured output is folded
/// into the error so the caller can surface a real diagnosis. `host`/`port` must
/// match the `--host`/`--port` already present in `serve_args`.
pub async fn start_local_server(
    bin: &Path,
    serve_args: &[String],
    host: &str,
    port: u16,
) -> Result<LocalServerHandle> {
    let mut command = crate::web::shell_quote(&bin.to_string_lossy());
    for arg in serve_args {
        command.push(' ');
        command.push_str(&crate::web::shell_quote(arg));
    }
    let runner = crate::ProcessRunner::new(std::env::current_dir()?)?;
    let process_id = LOCAL_SERVERS.spawn(&runner, &command)?;
    match wait_for_health(host, port).await {
        Ok(()) => Ok(LocalServerHandle {
            process_id,
            endpoint: format!("http://{host}:{port}/v1"),
        }),
        Err(err) => {
            let output = LOCAL_SERVERS.poll(&process_id).unwrap_or_default();
            let _ = LOCAL_SERVERS.kill(&process_id);
            bail!("hi-local did not become healthy at http://{host}:{port}: {err}\n{output}");
        }
    }
}

/// Stop a server started by [`start_local_server`]. No-op if already gone.
pub fn stop_local_server(process_id: &str) {
    let _ = LOCAL_SERVERS.kill(process_id);
}

/// Stop every local model server started by this process. For session shutdown:
/// the registry only ever holds `/goal` skeptic servers, so a frontend can call
/// this from a drop guard to cover all exit paths without tracking ids.
pub fn stop_all_local_servers() {
    LOCAL_SERVERS.kill_all();
}

async fn wait_for_health(host: &str, port: u16) -> Result<()> {
    let url = format!("http://{host}:{port}/health");
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(1))
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(1, 2));
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    let mut last_error = None;
    while tokio::time::Instant::now() < deadline {
        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => match response.json().await {
                Ok(body) if health_ready(&body) => return Ok(()),
                Ok(body) => last_error = Some(anyhow!("health returned not-ready body: {body}")),
                Err(err) => last_error = Some(anyhow!("health response was not valid JSON: {err}")),
            },
            Ok(response) => last_error = Some(anyhow!("health returned {}", response.status())),
            Err(err) => last_error = Some(anyhow!(err)),
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    Err(last_error.unwrap_or_else(|| anyhow!("health check timed out")))
}

fn health_ready(body: &serde_json::Value) -> bool {
    body.get("ready").and_then(serde_json::Value::as_bool) == Some(true)
}
