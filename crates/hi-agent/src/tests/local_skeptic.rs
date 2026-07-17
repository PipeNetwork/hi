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
    assert!(model_present(&dir, &spec), "config.json marks MLX present");
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
