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

/// Cache directory for a downloaded local model (`$HI_MLX_MODELS_DIR`, else
/// `~/.hi/models`, repo id sanitized) so a model fetched by any project is
/// reused rather than downloaded twice — a 50 GB checkpoint must never be
/// duplicated per working directory. Downloads that predate the shared root
/// (in the old cwd-relative `./.hi/models`) keep working via a per-repo
/// fallback. Uses the `main` revision (no `@rev` suffix).
pub fn skeptic_model_dir(repo_id: &str) -> PathBuf {
    let safe = crate::hf::safe_path(repo_id);
    if let Some(root) = std::env::var_os("HI_MLX_MODELS_DIR") {
        return PathBuf::from(root).join(safe);
    }
    let legacy = PathBuf::from(".hi").join("models").join(&safe);
    let shared = std::env::var_os("HOME")
        .map(|home| PathBuf::from(home).join(".hi").join("models").join(&safe));
    match shared {
        Some(shared) if !shared.is_dir() && legacy.is_dir() => legacy,
        Some(shared) => shared,
        None => legacy,
    }
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
    start_local_server_with_deadline(bin, serve_args, host, port, Duration::from_secs(15)).await
}

/// [`start_local_server`] with an explicit health deadline. Large models
/// need it: loading a 19 GB 32B checkpoint into memory takes minutes, not
/// the 15 seconds that suit a small review model.
pub async fn start_local_server_with_deadline(
    bin: &Path,
    serve_args: &[String],
    host: &str,
    port: u16,
    health_deadline: Duration,
) -> Result<LocalServerHandle> {
    let process_id = spawn_local_server(bin, serve_args)?;
    await_local_server_health(&process_id, host, port, health_deadline).await?;
    Ok(LocalServerHandle {
        process_id,
        endpoint: format!("http://{host}:{port}/v1"),
    })
}

/// Spawn the server process without waiting for readiness. Callers that want
/// live load progress use this, then poll [`await_local_server_health`] —
/// the handle is available immediately, so its RSS can be sampled while the
/// model loads.
pub fn spawn_local_server(bin: &Path, serve_args: &[String]) -> Result<String> {
    let mut command = crate::web::shell_quote(&bin.to_string_lossy());
    for arg in serve_args {
        command.push(' ');
        command.push_str(&crate::web::shell_quote(arg));
    }
    let runner = crate::ProcessRunner::new(std::env::current_dir()?)?;
    LOCAL_SERVERS.spawn(&runner, &command)
}

/// The OS pid of a spawned local server (for RSS-based load progress).
pub fn local_server_os_pid(process_id: &str) -> Option<i32> {
    LOCAL_SERVERS.os_pid(process_id)
}

/// Wait for a spawned server to become healthy. Fails FAST when the process
/// exits during startup — a crashed server (wrong backend feature, bad
/// weights) must error in seconds with its output, not after staring at a
/// dead port for the whole multi-minute deadline.
pub async fn await_local_server_health(
    process_id: &str,
    host: &str,
    port: u16,
    health_deadline: Duration,
) -> Result<()> {
    let deadline = tokio::time::Instant::now() + health_deadline;
    loop {
        if let Ok(outcome) = LOCAL_SERVERS.outcome(process_id)
            && outcome.state != crate::BackgroundState::Running
        {
            let output = LOCAL_SERVERS.poll(process_id).unwrap_or_default();
            let tail: String = output.chars().rev().take(500).collect::<String>().chars().rev().collect();
            anyhow::bail!("the local model server exited during startup: {tail}");
        }
        if try_health_once(host, port).await {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(400)).await;
    }
    match wait_for_health(host, port, Duration::from_millis(1)).await {
        Ok(()) => Ok(()),
        Err(err) => {
            let output = LOCAL_SERVERS.poll(process_id).unwrap_or_default();
            let _ = LOCAL_SERVERS.kill(process_id);
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

async fn wait_for_health(host: &str, port: u16, health_deadline: Duration) -> Result<()> {
    let url = format!("http://{host}:{port}/health");
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(1))
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(1, 2));
    let deadline = tokio::time::Instant::now() + health_deadline;
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

/// One health probe attempt: true when the server answers ready.
async fn try_health_once(host: &str, port: u16) -> bool {
    let url = format!("http://{host}:{port}/health");
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(1))
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(1, 2));
    match client.get(&url).send().await {
        Ok(response) if response.status().is_success() => match response.json().await {
            Ok(body) => health_ready(&body),
            Err(_) => false,
        },
        _ => false,
    }
}

fn health_ready(body: &serde_json::Value) -> bool {
    body.get("ready").and_then(serde_json::Value::as_bool) == Some(true)
}

#[cfg(test)]
mod model_dir_tests {
    // SAFETY: nextest runs each test in its own process, so env/cwd mutation
    // can't race other tests.
    #[test]
    fn model_dir_prefers_env_then_shared_home_with_legacy_fallback() {
        let scratch = std::env::temp_dir().join(format!("hi-modeldir-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        std::env::set_current_dir(&scratch).unwrap();
        let home = scratch.join("home");
        std::fs::create_dir_all(&home).unwrap();
        unsafe { std::env::set_var("HOME", &home) };

        unsafe { std::env::set_var("HI_MLX_MODELS_DIR", scratch.join("override")) };
        assert!(super::skeptic_model_dir("a/b").starts_with(scratch.join("override")));
        unsafe { std::env::remove_var("HI_MLX_MODELS_DIR") };

        let shared = home.join(".hi").join("models").join("a_b");
        assert_eq!(super::skeptic_model_dir("a/b"), shared, "default is the shared home root");

        std::fs::create_dir_all(scratch.join(".hi").join("models").join("a_b")).unwrap();
        assert_eq!(
            super::skeptic_model_dir("a/b"),
            std::path::PathBuf::from(".hi").join("models").join("a_b"),
            "a pre-existing cwd-local download keeps working (cwd-relative, as always)"
        );

        std::fs::create_dir_all(&shared).unwrap();
        assert_eq!(
            super::skeptic_model_dir("a/b"),
            shared,
            "once the shared copy exists it wins over the legacy one"
        );
        let _ = std::fs::remove_dir_all(&scratch);
    }
}
