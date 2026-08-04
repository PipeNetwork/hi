//! Unit tests for the deterministic core of the auto-managed local skeptic:
//! backend selection, the default-model registry, weight-presence checks, the
//! `hi-local serve` argument builder, and `/config skeptic-local` parsing. The
//! live orchestration (download + spawn + health) needs real hardware and is
//! exercised manually, not here.

use crate::command::{ConfigArg, config_is_skeptic_local, parse_config_arg};
use crate::local_skeptic::{
    LocalBackend, default_model, endpoint_url, model_present, pick_backend, serve_args,
    serve_model_path,
};
use std::path::{Path, PathBuf};

fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hi-local-skeptic-test-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn pick_backend_prefers_mlx_then_cuda_then_none() {
    assert_eq!(pick_backend(true, false), Some(LocalBackend::Mlx));
    // Apple Silicon wins even if an NVIDIA runtime is somehow also present.
    assert_eq!(pick_backend(true, true), Some(LocalBackend::Mlx));
    assert_eq!(pick_backend(false, true), Some(LocalBackend::Cuda));
    assert_eq!(pick_backend(false, false), None);
}

#[test]
fn default_model_matches_backend() {
    let mlx = default_model(LocalBackend::Mlx);
    assert_eq!(mlx.backend, LocalBackend::Mlx);
    assert_eq!(mlx.backend.serve_flag(), "mlx");
    // MLX serves a whole directory — no single GGUF file.
    assert!(mlx.gguf_file.is_none());
    assert!(!mlx.repo.is_empty());
    assert!(!mlx.model_id.is_empty());

    let cuda = default_model(LocalBackend::Cuda);
    assert_eq!(cuda.backend, LocalBackend::Cuda);
    assert_eq!(cuda.backend.serve_flag(), "cuda");
    // CUDA serves one GGUF file inside the repo.
    assert!(
        cuda.gguf_file
            .as_deref()
            .is_some_and(|f| f.ends_with(".gguf"))
    );
}

#[test]
fn model_present_checks_config_json_for_mlx() {
    let dir = scratch_dir("mlx");
    let spec = default_model(LocalBackend::Mlx);
    assert!(!model_present(&dir, &spec), "empty dir is not present");
    std::fs::write(dir.join("config.json"), "{}").unwrap();
    assert!(
        !model_present(&dir, &spec),
        "config.json alone is a partial download, not a model"
    );
    std::fs::write(dir.join("model.safetensors"), b"w").unwrap();
    assert!(
        model_present(&dir, &spec),
        "config + weights mark MLX present"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn model_present_checks_the_gguf_file_for_cuda() {
    let dir = scratch_dir("cuda");
    let spec = default_model(LocalBackend::Cuda);
    let file = spec.gguf_file.clone().unwrap();
    // A config.json is not enough for CUDA — the specific GGUF must exist.
    std::fs::write(dir.join("config.json"), "{}").unwrap();
    assert!(!model_present(&dir, &spec));
    std::fs::write(dir.join(&file), b"gguf").unwrap();
    assert!(model_present(&dir, &spec));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn serve_model_path_is_dir_for_mlx_and_file_for_cuda() {
    let dir = Path::new("/models/repo");
    let mlx = default_model(LocalBackend::Mlx);
    assert_eq!(serve_model_path(dir, &mlx), dir.to_path_buf());

    let cuda = default_model(LocalBackend::Cuda);
    let expected = dir.join(cuda.gguf_file.clone().unwrap());
    assert_eq!(serve_model_path(dir, &cuda), expected);
}

#[test]
fn serve_args_builds_the_expected_invocation() {
    let spec = default_model(LocalBackend::Mlx);
    let path = Path::new("/models/repo");
    let args = serve_args(path, &spec, "127.0.0.1", 8123);
    assert_eq!(
        args,
        vec![
            "serve".to_string(),
            "/models/repo".to_string(),
            "--backend".to_string(),
            "mlx".to_string(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            "8123".to_string(),
            "--model-id".to_string(),
            spec.model_id.clone(),
        ]
    );
}

#[test]
fn endpoint_url_is_openai_compatible() {
    assert_eq!(endpoint_url("127.0.0.1", 8080), "http://127.0.0.1:8080/v1");
}

#[test]
fn config_parses_skeptic_local_on_off_and_invalid() {
    assert_eq!(
        parse_config_arg("skeptic-local on"),
        ConfigArg::SkepticLocal(true)
    );
    assert_eq!(
        parse_config_arg("skeptic-local off"),
        ConfigArg::SkepticLocal(false)
    );
    // Alias + case-insensitive value.
    assert_eq!(
        parse_config_arg("local-skeptic ON"),
        ConfigArg::SkepticLocal(true)
    );
    assert!(matches!(
        parse_config_arg("skeptic-local"),
        ConfigArg::Invalid(_)
    ));
    assert!(matches!(
        parse_config_arg("skeptic-local maybe"),
        ConfigArg::Invalid(_)
    ));

    assert!(config_is_skeptic_local("skeptic-local on"));
    assert!(config_is_skeptic_local("skeptic-local off"));
    // A different /config option must not be misrouted to the async handler.
    assert!(!config_is_skeptic_local("reasoning high"));
    assert!(!config_is_skeptic_local("skeptic-local nonsense"));
}

#[tokio::test]
async fn skeptic_local_reuses_a_running_team_server_instead_of_spawning() {
    let mut agent = crate::tests::common::agent(vec![], crate::AgentConfig::default());
    let process_id = hi_tools::spawn_local_server(
        std::path::Path::new("/bin/sh"),
        &["-c".into(), "sleep 60".into()],
    )
    .expect("test local server process");
    agent.register_team_local_server(
        "http://127.0.0.1:9481/v1".into(),
        "Laguna-S-2.1-MLX-2bit".into(),
        process_id.clone(),
    );
    agent.set_delegate_route(
        Some("Laguna-S-2.1-MLX-2bit".into()),
        Some("http://127.0.0.1:9481/v1".into()),
        None,
    );

    let outcome = agent.enable_local_skeptic(false).await.unwrap();
    match outcome {
        crate::LocalSkepticOutcome::Ready { endpoint, model_id } => {
            assert_eq!(endpoint, "http://127.0.0.1:9481/v1");
            assert_eq!(model_id, "Laguna-S-2.1-MLX-2bit");
        }
        other => panic!("expected Ready on a running team server, got {other:?}"),
    }
    let skeptic = agent
        .team_roles()
        .into_iter()
        .find(|role| role.role == "skeptic")
        .unwrap();
    assert_eq!(
        skeptic.model, "Laguna-S-2.1-MLX-2bit",
        "skeptic route points at the team executor"
    );

    // Turning the skeptic off must NOT stop the team server (the executors
    // still depend on it) — the registry entry survives.
    assert!(agent.disable_local_skeptic());
    assert!(
        agent
            .running_local_model_server("Laguna-S-2.1-MLX-2bit")
            .is_some(),
        "team server survives skeptic disable"
    );
    hi_tools::stop_local_server(&process_id);
    let delegate = agent
        .team_roles()
        .into_iter()
        .find(|role| role.role == "delegate")
        .unwrap();
    assert!(
        delegate.inherited,
        "a dead managed endpoint must fall back to the driver route"
    );
    assert!(
        agent.local_skeptic_endpoint().is_none(),
        "a dead local skeptic must not remain advertised as running"
    );
}

#[tokio::test]
async fn disabling_a_team_server_skeptic_releases_an_unreferenced_server() {
    let mut agent = crate::tests::common::agent(vec![], crate::AgentConfig::default());
    let process_id = hi_tools::spawn_local_server(
        std::path::Path::new("/bin/sh"),
        &["-c".into(), "sleep 60".into()],
    )
    .expect("test local server process");
    let endpoint = "http://127.0.0.1:9485/v1".to_string();
    let model = "borrowed-team-model".to_string();
    agent.register_team_local_server(endpoint.clone(), model.clone(), process_id.clone());
    agent.set_delegate_route(Some(model.clone()), Some(endpoint), None);
    agent.enable_local_skeptic(false).await.unwrap();

    // The executor is cleared while the skeptic still borrows the server, so
    // it must survive until the skeptic is disabled.
    agent.set_delegate_route(None, None, None);
    assert!(hi_tools::local_server_is_running(&process_id));
    assert!(agent.disable_local_skeptic());

    assert!(!hi_tools::local_server_is_running(&process_id));
    assert!(
        agent.running_local_model_server(&model).is_none(),
        "an unreferenced team server must be removed after skeptic disable"
    );
}

#[tokio::test]
async fn dead_managed_skeptic_is_recovered_before_team_reuse() {
    let mut agent = crate::tests::common::agent(vec![], crate::AgentConfig::default());
    let stale_process = hi_tools::spawn_local_server(
        std::path::Path::new("/bin/sh"),
        &["-c".into(), "sleep 60".into()],
    )
    .expect("stale skeptic process");
    agent.local_skeptic = Some(crate::local_skeptic::LocalSkepticState {
        process_id: stale_process.clone(),
        endpoint: "http://127.0.0.1:9482/v1".into(),
        model_id: "stale-model".into(),
        prev_skeptic_model: None,
        prev_endpoint: None,
        prev_endpoint_key: None,
    });
    hi_tools::stop_local_server(&stale_process);

    let team_process = hi_tools::spawn_local_server(
        std::path::Path::new("/bin/sh"),
        &["-c".into(), "sleep 60".into()],
    )
    .expect("team server process");
    agent.register_team_local_server(
        "http://127.0.0.1:9483/v1".into(),
        "team-model".into(),
        team_process.clone(),
    );

    let outcome = agent.enable_local_skeptic(false).await.unwrap();
    assert_eq!(
        outcome,
        crate::LocalSkepticOutcome::Ready {
            endpoint: "http://127.0.0.1:9483/v1".into(),
            model_id: "team-model".into(),
        }
    );

    let driver_model = agent.config.routing.model.clone();
    hi_tools::stop_local_server(&team_process);
    assert_eq!(
        agent.effective_skeptic_model(),
        driver_model,
        "goal reviews must fall back to the driver when their managed local route dies"
    );

    agent.disable_local_skeptic();
}

#[tokio::test]
async fn team_table_hides_a_dead_dedicated_skeptic_route() {
    let mut agent = crate::tests::common::agent(vec![], crate::AgentConfig::default());
    let process_id = hi_tools::spawn_local_server(
        std::path::Path::new("/bin/sh"),
        &["-c".into(), "sleep 60".into()],
    )
    .expect("dedicated skeptic process");
    let endpoint = "http://127.0.0.1:9484/v1".to_string();
    let model_id = "dedicated-stale-model".to_string();
    agent.config.subagents.skeptic_model = Some(model_id.clone());
    agent.config.subagents.skeptic_endpoint = Some(endpoint.clone());
    agent.config.subagents.skeptic_endpoint_key = Some("local".into());
    agent.rebuild_skeptic_provider();
    agent.local_skeptic = Some(crate::local_skeptic::LocalSkepticState {
        process_id: process_id.clone(),
        endpoint,
        model_id,
        prev_skeptic_model: None,
        prev_endpoint: None,
        prev_endpoint_key: None,
    });

    hi_tools::stop_local_server(&process_id);

    let skeptic = agent
        .team_roles()
        .into_iter()
        .find(|role| role.role == "skeptic")
        .expect("skeptic row");
    assert!(
        skeptic.inherited,
        "dead dedicated route falls back to driver"
    );
    assert_eq!(skeptic.model, agent.config.routing.model);
    assert_eq!(skeptic.route, "driver provider");
}
