//! Auto-managed local model server for the `/goal` skeptic review.
//!
//! Spawns `hi-local serve …` into a private background registry, waits for its
//! `/health` endpoint to report ready, and hands back the OpenAI-compatible base
//! URL. The policy around it — which backend, which default model, when to turn
//! it on — lives in `hi-agent`'s `local_skeptic` module; this file owns only the
//! process and HTTP mechanics, mirroring the proven `/hf run --mlx` path.

use anyhow::{Result, anyhow, bail};
use serde_json::json;
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
    skeptic_model_dir_in(
        repo_id,
        std::env::var_os("HI_MLX_MODELS_DIR"),
        std::env::var_os("HOME"),
        Path::new("."),
    )
}

/// Pure core of [`skeptic_model_dir`]: every environmental input is an
/// explicit parameter so tests can exercise the resolution rules without
/// mutating process-global env or cwd (those writes raced other tests — the
/// sandbox tests read `HOME` concurrently under one-process `cargo test`).
/// The legacy path stays cwd-relative in the return value, as always; `cwd`
/// is only used to check whether that legacy download exists.
fn skeptic_model_dir_in(
    repo_id: &str,
    models_dir_override: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
    cwd: &Path,
) -> PathBuf {
    let safe = crate::hf::safe_path(repo_id);
    if let Some(root) = models_dir_override {
        return PathBuf::from(root).join(safe);
    }
    let legacy = PathBuf::from(".hi").join("models").join(&safe);
    let legacy_exists = cwd.join(&legacy).is_dir();
    let shared = home.map(|home| PathBuf::from(home).join(".hi").join("models").join(&safe));
    match shared {
        Some(shared) if !shared.is_dir() && legacy_exists => legacy,
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
    // Release bundles place the precompiled Metal library beside the binary at
    // ../lib/mlx/mlx.metallib. The vendored pmetal loader honors this variable,
    // so packaged users do not need a writable Cargo/Python cache or a manual
    // MLX environment setup. Source/PATH installs simply use their normal
    // colocated or user-cache discovery.
    if let Some(metallib) = bundled_metallib_path(bin) {
        command = format!(
            "PMETAL_METALLIB_PATH={} {command}",
            crate::web::shell_quote(&metallib.to_string_lossy())
        );
    }
    for arg in serve_args {
        command.push(' ');
        command.push_str(&crate::web::shell_quote(arg));
    }
    let runner = crate::ProcessRunner::new(std::env::current_dir()?)?;
    LOCAL_SERVERS.spawn(&runner, &command)
}

fn bundled_metallib_path(bin: &Path) -> Option<PathBuf> {
    let path = bin.parent()?.join("../lib/mlx/mlx.metallib");
    path.is_file().then_some(path)
}

/// The OS pid of a spawned local server (for RSS-based load progress).
pub fn local_server_os_pid(process_id: &str) -> Option<i32> {
    LOCAL_SERVERS.os_pid(process_id)
}

/// Whether a server owned by this process is still running.
///
/// Team routes are kept in the agent for the lifetime of a session, but the
/// child can exit independently (OOM, bad weights, or a backend crash). Do
/// not treat a stale handle as a reusable local server.
pub fn local_server_is_running(process_id: &str) -> bool {
    LOCAL_SERVERS
        .outcome(process_id)
        .is_ok_and(|outcome| outcome.state == crate::BackgroundState::Running)
}

fn health_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(1))
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(1, 2))
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
    // Reuse one connection pool for the entire readiness wait. Model loading
    // can take minutes; constructing a new reqwest client every 400ms needlessly
    // allocates clients and throws away keep-alive connections.
    let client = health_client();
    loop {
        if let Ok(outcome) = LOCAL_SERVERS.outcome(process_id)
            && outcome.state != crate::BackgroundState::Running
        {
            let output = LOCAL_SERVERS.poll(process_id).unwrap_or_default();
            let tail: String = output
                .chars()
                .rev()
                .take(500)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            // A process can exit before the readiness timeout. Reap it here
            // as well as on timeout; otherwise every failed model load leaves
            // a dead registry entry (and, depending on the runner, its child
            // process tree) behind until session shutdown.
            let _ = LOCAL_SERVERS.kill(process_id);
            anyhow::bail!("the local model server exited during startup: {tail}");
        }
        if try_health_once(&client, host, port).await {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(400)).await;
    }
    match wait_for_health(&client, host, port, Duration::from_millis(1)).await {
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

/// Stop every local model server started by this process. The registry holds
/// only hi-owned `/goal` skeptic and `/team` role servers, so a frontend can
/// call this from a drop guard to cover all exit paths without tracking ids.
pub fn stop_all_local_servers() {
    LOCAL_SERVERS.kill_all();
}

/// Verify the OpenAI-compatible surface after `/health` turns ready. A
/// healthy process can still be serving the wrong model or a binary built
/// without the requested backend, so the driver switches only after both the
/// model list and a minimal non-streaming completion succeed.
pub async fn verify_local_server(endpoint: &str, model_id: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(2, 30));
    let models_url = format!("{}/models", endpoint.trim_end_matches('/'));
    let response = client.get(&models_url).send().await?;
    let status = response.status();
    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|error| anyhow!("/v1/models returned invalid JSON: {error}"))?;
    if !status.is_success() {
        bail!("/v1/models returned {status}: {body}");
    }
    let advertised = body
        .get("data")
        .and_then(serde_json::Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|model| model.get("id").and_then(|id| id.as_str()))
        })
        .into_iter()
        .flatten()
        .any(|id| id == model_id);
    if !advertised {
        bail!("local server did not advertise model '{model_id}'");
    }

    let completions_url = format!("{}/chat/completions", endpoint.trim_end_matches('/'));
    let response = client
        .post(&completions_url)
        .json(&json!({
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 1,
            "stream": false,
        }))
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        bail!("local server completion probe returned {status}: {text}");
    }

    // Probe the tool-bearing request shape separately. `tool_choice: none`
    // keeps this a compatibility check rather than asking the model to emit a
    // real tool call, while still catching servers that silently implement
    // chat-only requests.
    let response = client
        .post(completions_url)
        .json(&json!({
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "hi_runtime_probe",
                    "description": "Compatibility probe; do not call it.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": "none",
            "max_tokens": 1,
            "stream": false,
        }))
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        bail!("local server tool compatibility probe returned {status}: {text}");
    }
    Ok(())
}

async fn wait_for_health(
    client: &reqwest::Client,
    host: &str,
    port: u16,
    health_deadline: Duration,
) -> Result<()> {
    let url = format!("http://{host}:{port}/health");
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
async fn try_health_once(client: &reqwest::Client, host: &str, port: u16) -> bool {
    let url = format!("http://{host}:{port}/health");
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
    use std::ffi::OsString;

    // All environmental inputs are passed explicitly — no process-global
    // env/cwd mutation, which raced the sandbox tests (they read `HOME`)
    // under single-process `cargo test`.
    #[test]
    fn model_dir_prefers_env_then_shared_home_with_legacy_fallback() {
        let scratch = std::env::temp_dir().join(format!("hi-modeldir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch).unwrap();
        let home = scratch.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let home_env = || Some(OsString::from(&home));

        assert!(
            super::skeptic_model_dir_in(
                "a/b",
                Some(OsString::from(scratch.join("override"))),
                home_env(),
                &scratch,
            )
            .starts_with(scratch.join("override")),
            "the models-dir override wins"
        );

        let shared = home.join(".hi").join("models").join("a_b");
        assert_eq!(
            super::skeptic_model_dir_in("a/b", None, home_env(), &scratch),
            shared,
            "default is the shared home root"
        );

        std::fs::create_dir_all(scratch.join(".hi").join("models").join("a_b")).unwrap();
        assert_eq!(
            super::skeptic_model_dir_in("a/b", None, home_env(), &scratch),
            std::path::PathBuf::from(".hi").join("models").join("a_b"),
            "a pre-existing cwd-local download keeps working (cwd-relative, as always)"
        );

        std::fs::create_dir_all(&shared).unwrap();
        assert_eq!(
            super::skeptic_model_dir_in("a/b", None, home_env(), &scratch),
            shared,
            "once the shared copy exists it wins over the legacy one"
        );

        assert_eq!(
            super::skeptic_model_dir_in("a/b", None, None, &scratch),
            std::path::PathBuf::from(".hi").join("models").join("a_b"),
            "no home falls back to the cwd-local path"
        );
        let _ = std::fs::remove_dir_all(&scratch);
    }

    #[test]
    fn bundled_metallib_is_found_relative_to_hi_local() {
        let scratch = std::env::temp_dir().join(format!("hi-bundle-{}", std::process::id()));
        let bin = scratch.join("bin");
        let lib = scratch.join("lib/mlx");
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::write(lib.join("mlx.metallib"), b"fixture").unwrap();
        let expected = bin.join("../lib/mlx/mlx.metallib");

        assert_eq!(
            super::bundled_metallib_path(&bin.join("hi-local")).as_deref(),
            Some(expected.as_path())
        );
        let _ = std::fs::remove_dir_all(&scratch);
    }
}
