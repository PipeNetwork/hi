use super::{
    Cli, Config, DEFAULT_MAX_TOKENS, LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS, LocalRuntimeProfile,
    PIPENETWORK_DEFAULT_MAX_TOKENS, Profile, ProviderName, RsiRequested, RsiSection, auto_select,
    auto_selected_env, configured_max_tokens, curate_skills_default, detect_verify_pipeline,
    detect_verify_pipeline_with, explore_subagents_default, is_official_deepseek_url,
    max_tokens_is_explicit, needs_setup, permits_missing_checkpoint, planner_model_default,
    read_config_file, resolve, resolve_active_profile, resolve_fallbacks, resolve_named_profile,
    resolve_quality, resolve_reasoning_effort, resolve_rsi, resolve_x402_settings, save_config_to,
    set_rsi_config, suggest_next_prompt_default, upsert_profile_project_local,
    write_subagents_default, x402_configured,
};
use clap::Parser;
use hi_agent::{LspMode, ReviewPolicy, ToolSet, VerificationMode};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

fn temp_dir_with(marker: &str) -> std::path::PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "hi-detect-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    if !marker.is_empty() {
        std::fs::write(dir.join(marker), "").unwrap();
    }
    dir
}

fn tmp_leftovers(path: &Path) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    std::fs::read_dir(parent).ok().is_some_and(|entries| {
        entries
            .flatten()
            .any(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
    })
}

#[test]
fn detects_layered_pipeline_by_marker() {
    // (marker, expected stage commands in order). Bare package.json, Makefile,
    // and testless Python markers detect nothing: stages come from declared
    // scripts/targets or an actual test suite, never from assuming a
    // convention the repo doesn't state.
    let cases: [(&str, Vec<&str>); 6] = [
        (
            "Cargo.toml",
            vec!["cargo check --quiet", "cargo test --quiet"],
        ),
        ("go.mod", vec!["go build ./...", "go test ./..."]),
        ("pyproject.toml", vec![]),
        ("package.json", vec![]),
        ("Makefile", vec![]),
        ("", vec![]),
    ];
    for (marker, expected) in cases {
        let dir = temp_dir_with(marker);
        let got: Vec<String> = detect_verify_pipeline(&dir)
            .into_iter()
            .map(|s| s.command)
            .collect();
        assert_eq!(got, expected, "marker={marker:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn python_pipeline_requires_a_collectable_test_file() {
    let dir = temp_dir_with("pyproject.toml");
    std::fs::write(dir.join("test_smoke.py"), "def test_smoke(): pass\n").unwrap();
    let commands: Vec<String> = detect_verify_pipeline(&dir)
        .into_iter()
        .map(|s| s.command)
        .collect();
    assert_eq!(commands, ["pytest -q"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn javascript_pipeline_follows_declared_scripts_and_lockfile() {
    let dir = temp_dir_with("");
    std::fs::write(
        dir.join("package.json"),
        r#"{"scripts": {"test": "vitest run", "lint": "eslint ."}}"#,
    )
    .unwrap();
    let commands: Vec<String> = detect_verify_pipeline(&dir)
        .into_iter()
        .map(|s| s.command)
        .collect();
    assert_eq!(commands, ["npm run lint", "npm test --silent"]);

    // A pnpm lockfile switches the runner.
    std::fs::write(dir.join("pnpm-lock.yaml"), "").unwrap();
    let commands: Vec<String> = detect_verify_pipeline(&dir)
        .into_iter()
        .map(|s| s.command)
        .collect();
    assert_eq!(commands, ["pnpm run lint", "pnpm test"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn makefile_pipeline_requires_declared_targets() {
    let dir = temp_dir_with("");
    std::fs::write(
        dir.join("Makefile"),
        "check:\n\techo check\n\ntest:\n\techo test\n",
    )
    .unwrap();
    let commands: Vec<String> = detect_verify_pipeline(&dir)
        .into_iter()
        .map(|s| s.command)
        .collect();
    assert_eq!(commands, ["make check", "make test"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn quality_defaults_to_automatic_safe_policy() {
    let dir = temp_dir_with("");
    let cli = super::Cli::try_parse_from(["hi"]).unwrap();
    let quality = resolve_quality(&cli, &dir).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(quality.verification, VerificationMode::Auto);
    assert_eq!(quality.max_verify_repairs, 2);
    assert_eq!(quality.review, ReviewPolicy::Risk);
    assert_eq!(quality.lsp_mode, LspMode::Auto);
    assert_eq!(quality.tool_set, ToolSet::Dynamic);
    assert!(!cli.allow_no_checkpoint);
    assert!(permits_missing_checkpoint(&cli));
}

#[test]
fn project_race_config_loads_targets_and_fuzz_without_credentials() {
    let dir = temp_dir_with("");
    std::fs::create_dir_all(dir.join(".hi")).unwrap();
    std::fs::write(
        dir.join(".hi/config.toml"),
        r#"
[race]
max_candidates = 2
fuzz_command = "cargo fuzz run parser -- -runs=10"
fuzz_timeout_secs = 9

[[race.targets]]
name = "fast"
profile = "local"
model = "model-a"

[[race.targets]]
name = "strong"
profile = "cloud"
model = "model-b"
priority = 1
"#,
    )
    .unwrap();
    let cli = super::Cli::try_parse_from(["hi"]).unwrap();
    let quality = resolve_quality(&cli, &dir).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(quality.race.enabled);
    assert_eq!(quality.race.targets.len(), 2);
    assert_eq!(quality.race.targets[1].model, "model-b");
    assert_eq!(quality.race.fuzz.as_ref().unwrap().timeout_secs, 9);
}

#[test]
fn parses_deepseek_compatibility_override() {
    let cli = Cli::try_parse_from(["hi", "--deepseek-compat", "on"]).unwrap();
    assert_eq!(cli.deepseek_compat, Some(super::CliDeepSeekCompat::On));

    let cli = Cli::try_parse_from(["hi", "--deepseek-compat", "off"]).unwrap();
    assert_eq!(cli.deepseek_compat, Some(super::CliDeepSeekCompat::Off));
}

#[test]
fn durable_execution_can_be_selected_by_cli_or_profile() {
    let cli = Cli::try_parse_from(["hi", "--durable"]).unwrap();
    let mut config = Config::default();
    config.profiles.insert(
        "durable".into(),
        Profile {
            provider: Some(ProviderName::Openai),
            model: Some("gpt-4o".into()),
            api_key: Some("test".into()),
            execution: Some(hi_agent::ExecutionMode::Durable),
            ..Profile::default()
        },
    );
    config.default_profile = Some("durable".into());

    assert_eq!(
        resolve(&cli, &config).unwrap().execution,
        hi_agent::ExecutionMode::Durable
    );

    // Select the profile explicitly so another parallel test's workspace
    // last-session fixture cannot override this test's route.
    let cli = Cli::try_parse_from(["hi", "--profile", "durable"]).unwrap();
    assert_eq!(
        resolve(&cli, &config).unwrap().execution,
        hi_agent::ExecutionMode::Durable
    );
}

#[test]
fn profile_serializes_deepseek_compatibility_override() {
    let config = Config {
        profiles: [(
            "deepseek".to_string(),
            Profile {
                deepseek_compat: Some(hi_ai::DeepSeekCompat::On),
                ..Profile::default()
            },
        )]
        .into_iter()
        .collect(),
        ..Config::default()
    };
    let encoded = toml::to_string(&config).unwrap();
    assert!(encoded.contains("deepseek_compat = \"on\""));
}

#[test]
#[allow(clippy::field_reassign_with_default)] // test config assembled field-by-field
fn managed_local_runtime_profile_round_trips_and_reaches_settings() {
    let mut config = Config::default();
    config.default_profile = Some("deepseek-mlx".into());
    config.profiles.insert(
        "deepseek-mlx".into(),
        Profile {
            provider: Some(ProviderName::Openai),
            model: Some("DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx".into()),
            base_url: Some("http://127.0.0.1:8080/v1".into()),
            api_key: Some("local".into()),
            runtime: Some(LocalRuntimeProfile {
                kind: "mlx".into(),
                repo: "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx".into(),
                backend: Some("mlx".into()),
                autostart: true,
                model_path: None,
                quantization: None,
                context_window: None,
                tool_mode: None,
            }),
            ..Default::default()
        },
    );
    let encoded = toml::to_string(&config).unwrap();
    let decoded: Config = toml::from_str(&encoded).unwrap();
    assert_eq!(
        decoded.profiles["deepseek-mlx"].runtime,
        config.profiles["deepseek-mlx"].runtime
    );
    let cli = Cli::try_parse_from(["hi", "--profile", "deepseek-mlx"]).unwrap();
    let settings = resolve(&cli, &decoded).unwrap();
    assert_eq!(settings.runtime, decoded.profiles["deepseek-mlx"].runtime);
}

#[test]
fn checkpoint_policy_is_yolo_unless_edit_confirmation_is_strict() {
    let default = super::Cli::try_parse_from(["hi"]).unwrap();
    assert!(permits_missing_checkpoint(&default));

    let strict = super::Cli::try_parse_from(["hi", "--confirm-edits"]).unwrap();
    assert!(!permits_missing_checkpoint(&strict));

    let override_cli =
        super::Cli::try_parse_from(["hi", "--confirm-edits", "--allow-no-checkpoint"]).unwrap();
    assert!(permits_missing_checkpoint(&override_cli));
}

#[test]
fn cli_quality_overrides_project_config_and_verify_is_repeatable() {
    let dir = temp_dir_with("");
    std::fs::create_dir_all(dir.join(".hi")).unwrap();
    std::fs::write(
        dir.join(".hi/config.toml"),
        r#"[quality]
verification = "disabled"
max_verify_repairs = 7
review = "off"
lsp = "off"
tool_set = "full"
context_exclusions = ["generated/**"]
"#,
    )
    .unwrap();
    let cli = super::Cli::try_parse_from([
        "hi",
        "--verify",
        "cargo check",
        "--verify",
        "cargo test",
        "--max-verify-repairs",
        "1",
        "--review",
        "always",
        "--lsp",
        "on",
        "--tool-set",
        "minimal",
    ])
    .unwrap();
    let quality = resolve_quality(&cli, &dir).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        quality.verification,
        VerificationMode::Explicit(vec![
            hi_agent::VerifyStage::new("verify_1", "cargo check"),
            hi_agent::VerifyStage::new("verify_2", "cargo test"),
        ])
    );
    assert_eq!(quality.max_verify_repairs, 1);
    assert_eq!(quality.review, ReviewPolicy::Always);
    assert_eq!(quality.lsp_mode, LspMode::On);
    assert_eq!(quality.tool_set, ToolSet::Minimal);
    assert_eq!(quality.context_exclusions, vec!["generated/**"]);
}

#[test]
fn removed_quality_flags_are_usage_errors() {
    for flag in ["--auto-verify", "--max-verify", "--minimal-tools"] {
        assert!(
            super::Cli::try_parse_from(["hi", flag]).is_err(),
            "obsolete flag still accepted: {flag}"
        );
    }
}

#[test]
fn empty_verification_commands_are_configuration_errors() {
    let dir = temp_dir_with("");
    let cli = super::Cli::try_parse_from(["hi", "--verify", "   "]).unwrap();
    assert!(
        resolve_quality(&cli, &dir)
            .unwrap_err()
            .to_string()
            .contains("must not be empty")
    );

    std::fs::create_dir_all(dir.join(".hi")).unwrap();
    std::fs::write(
        dir.join(".hi/config.toml"),
        "[quality]\nverification = \"explicit\"\nstages = [\"\"]\n",
    )
    .unwrap();
    let cli = super::Cli::try_parse_from(["hi"]).unwrap();
    assert!(
        resolve_quality(&cli, &dir)
            .unwrap_err()
            .to_string()
            .contains("must not be empty")
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cargo_pipeline_runs_compile_gate_before_tests() {
    let dir = temp_dir_with("Cargo.toml");
    let stages = detect_verify_pipeline(&dir);
    // The cheap compile gate must come first so errors localize fast.
    assert_eq!(stages[0].name, "check");
    assert!(stages[0].command.contains("cargo check"));
    assert!(stages.last().unwrap().command.contains("cargo test"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cargo_pipeline_with_clippy_inserts_clippy_between_check_and_test() {
    let dir = temp_dir_with("Cargo.toml");
    let names: Vec<String> = detect_verify_pipeline_with(&dir, true)
        .into_iter()
        .map(|stage| stage.name)
        .collect();
    assert_eq!(names, ["check", "clippy", "test"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn onboarding_mentions_real_interactive_flags() {
    assert!(
        !super::ONBOARDING.contains("--tui"),
        "there is no --tui flag; the TUI is the default"
    );
    assert!(
        super::ONBOARDING.contains("--plain"),
        "onboarding should point to the actual opt-out flag"
    );
}

#[test]
fn pipenetwork_prefers_provider_specific_api_key_env() {
    assert_eq!(
        ProviderName::Pipenetwork.key_envs(),
        &["PIPENETWORK_API_KEY", "HI_API_KEY", "OPENAI_API_KEY"]
    );
}

/// `/provider pipe` is the short form of `/provider pipenetwork`, matching
/// `/login pipe` so users aren't bounced with "no profile or provider".
#[test]
fn pipe_is_an_alias_for_pipenetwork_provider() {
    assert_eq!(
        "pipe".parse::<ProviderName>(),
        Ok(ProviderName::Pipenetwork)
    );
    assert_eq!(
        "pipenetwork".parse::<ProviderName>(),
        Ok(ProviderName::Pipenetwork)
    );
    // Canonical spelling is unchanged — alias is input-only.
    assert_eq!(ProviderName::Pipenetwork.as_str(), "pipenetwork");
}

/// `/provider xai` should work with nothing configured — otherwise a user
/// who just ran `/login xai` has to hand-write a profile to use it.
#[test]
fn a_bare_provider_name_resolves_without_a_profile() {
    let config = Config::default();
    // Through the shared guard: `XAI_API_KEY` is also read by `auto_select`,
    // so setting it unsynchronized would race the `needs_setup` tests below.
    let env = ClearedSetupEnv::new();
    env.set("XAI_API_KEY", "test-key");
    let settings = resolve_named_profile(&config, "xai").unwrap();
    drop(env);
    assert_eq!(settings.provider, ProviderName::Xai);
    assert_eq!(settings.base_url, "https://api.x.ai/v1");
    assert_eq!(settings.model, "grok-4.6");
}

/// Switching back with `/provider pipenetwork` after `/provider xai` must
/// reuse a key stored on a differently-named profile (typically
/// `default_profile = "default"` with `provider = "pipenetwork"`). The bare
/// preset path used to ignore other profiles and only check auth.json + env.
#[test]
#[allow(clippy::field_reassign_with_default)] // test config assembled field-by-field
fn bare_provider_reuses_key_from_default_profile_for_that_provider() {
    let mut config = Config::default();
    config.default_profile = Some("default".into());
    config.profiles.insert(
        "default".into(),
        Profile {
            provider: Some(ProviderName::Pipenetwork),
            model: Some("ipop/coder-balanced".into()),
            api_key: Some("profile-pipe-key".into()),
            ..Default::default()
        },
    );
    let env = ClearedSetupEnv::new();
    let settings = resolve_named_profile(&config, "pipenetwork").unwrap();
    let via_alias = resolve_named_profile(&config, "pipe").unwrap();
    drop(env);
    assert_eq!(settings.provider, ProviderName::Pipenetwork);
    assert_eq!(settings.api_key, "profile-pipe-key");
    assert_eq!(via_alias.api_key, "profile-pipe-key");
    // Preset path keeps provider defaults for model — it is not a silent
    // rename of the profile; only the credential is borrowed.
    assert_eq!(settings.model, "pipe/deepseek-v4-flash-vision-exp");
}

/// When default_profile targets a different provider, still borrow from any
/// profile that does match the bare provider name.
#[test]
#[allow(clippy::field_reassign_with_default)] // test config assembled field-by-field
fn bare_provider_reuses_key_from_any_matching_profile() {
    let mut config = Config::default();
    config.default_profile = Some("local".into());
    config.profiles.insert(
        "local".into(),
        Profile {
            provider: Some(ProviderName::Ollama),
            model: Some("qwen".into()),
            ..Default::default()
        },
    );
    config.profiles.insert(
        "work".into(),
        Profile {
            provider: Some(ProviderName::Pipenetwork),
            api_key: Some("work-pipe-key".into()),
            ..Default::default()
        },
    );
    let env = ClearedSetupEnv::new();
    let settings = resolve_named_profile(&config, "pipenetwork").unwrap();
    drop(env);
    assert_eq!(settings.api_key, "work-pipe-key");
    assert_eq!(settings.provider, ProviderName::Pipenetwork);
}

/// A profile is explicit configuration, so it must win over the preset of
/// the same name.
#[test]
fn a_profile_shadows_a_same_named_provider() {
    let mut config = Config::default();
    config.profiles.insert(
        "xai".into(),
        Profile {
            provider: Some(ProviderName::Xai),
            model: Some("grok-4.5".into()),
            api_key: Some("profile-key".into()),
            ..Default::default()
        },
    );
    let settings = resolve_named_profile(&config, "xai").unwrap();
    assert_eq!(settings.model, "grok-4.5", "the profile's model must win");
    assert_eq!(settings.api_key, "profile-key");
}

#[test]
fn project_custom_endpoint_cannot_consume_global_ambient_key() {
    let mut config = Config {
        default_profile: Some("global".into()),
        profiles: std::collections::HashMap::from([(
            "global".into(),
            Profile {
                provider: Some(ProviderName::Openai),
                model: Some("global/model".into()),
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    // Model a repository hi.toml selecting its own endpoint without pairing a
    // credential. The user's ambient generic/provider keys must not follow it.
    super::merge_config_with_project_trust(
        &mut config,
        Config {
            default_profile: Some("project".into()),
            profiles: std::collections::HashMap::from([(
                "project".into(),
                Profile {
                    provider: Some(ProviderName::Openai),
                    model: Some("attacker/model".into()),
                    base_url: Some("https://attacker.example/v1".into()),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        },
        true,
    );
    let env = ClearedSetupEnv::new();
    env.set("HI_API_KEY", "global-generic-key");
    env.set("OPENROUTER_API_KEY", "global-openrouter-key");
    let selected = config.default_profile.as_deref().unwrap();
    let error = resolve_named_profile(&config, selected)
        .unwrap_err()
        .to_string();
    drop(env);

    assert!(error.contains("custom openai endpoint"), "{error}");
    assert!(error.contains("api_key_env"), "{error}");
    assert!(error.contains("--api-key"), "{error}");
}

#[test]
fn project_profile_cannot_name_user_environment_key_even_for_official_endpoint() {
    let mut config = Config::default();
    super::merge_config(
        &mut config,
        Config {
            default_profile: Some("project".into()),
            profiles: std::collections::HashMap::from([(
                "project".into(),
                Profile {
                    provider: Some(ProviderName::Openai),
                    model: Some("project/model".into()),
                    base_url: Some("https://openrouter.ai/api/v1".into()),
                    api_key_env: Some("OPENROUTER_API_KEY".into()),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        },
    );
    // Isolate the credential-source boundary from the independent route-trust
    // boundary: even a trusted project cannot name an ambient user secret.
    config.profiles.get_mut("project").unwrap().project_trusted = true;
    let env = ClearedSetupEnv::new();
    env.set("OPENROUTER_API_KEY", "user-secret");
    let error = resolve_named_profile(&config, "project")
        .unwrap_err()
        .to_string();
    assert!(error.contains("project-local profiles"), "{error}");
    assert!(error.contains("--api-key"), "{error}");

    let cli =
        Cli::try_parse_from(["hi", "--profile", "project", "--api-key", "explicit-key"]).unwrap();
    let settings = resolve(&cli, &config).unwrap();
    drop(env);
    assert_eq!(settings.api_key, "explicit-key");
}

#[test]
fn project_profile_custom_endpoint_requires_persisted_folder_trust() {
    let mut config = Config::default();
    super::merge_config(
        &mut config,
        Config {
            default_profile: Some("project".into()),
            profiles: std::collections::HashMap::from([(
                "project".into(),
                Profile {
                    provider: Some(ProviderName::Openai),
                    model: Some("project/model".into()),
                    base_url: Some("https://project-gateway.example/v1".into()),
                    api_key: Some("repository-test-key".into()),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        },
    );
    let error = resolve_named_profile(&config, "project")
        .unwrap_err()
        .to_string();
    assert!(error.contains("persisted folder trust"), "{error}");

    config.profiles.get_mut("project").unwrap().project_trusted = true;
    let settings = resolve_named_profile(&config, "project").unwrap();
    assert_eq!(settings.api_key, "repository-test-key");
}

#[test]
fn project_profile_official_remote_route_requires_persisted_folder_trust() {
    let mut config = Config::default();
    super::merge_config(
        &mut config,
        Config {
            default_profile: Some("project".into()),
            profiles: std::collections::HashMap::from([(
                "project".into(),
                Profile {
                    provider: Some(ProviderName::Anthropic),
                    model: Some("claude-project".into()),
                    api_key: Some("repository-owned-key".into()),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        },
    );

    let error = resolve_named_profile(&config, "project")
        .unwrap_err()
        .to_string();
    assert!(error.contains("persisted folder trust"), "{error}");

    config.profiles.get_mut("project").unwrap().project_trusted = true;
    let settings = resolve_named_profile(&config, "project").unwrap();
    assert_eq!(settings.api_key, "repository-owned-key");
}

#[test]
fn project_custom_mcp_cannot_receive_ambient_provider_key() {
    let mut config = Config::default();
    super::merge_config_with_project_trust(
        &mut config,
        Config {
            default_profile: Some("project".into()),
            profiles: std::collections::HashMap::from([(
                "project".into(),
                Profile {
                    provider: Some(ProviderName::Pipenetwork),
                    model: Some("pipe/model".into()),
                    base_url: Some("https://api.pipenetwork.ai/v1".into()),
                    mcp_url: Some("https://attacker.example/mcp".into()),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        },
        true,
    );
    let env = ClearedSetupEnv::new();
    env.set("PIPENETWORK_API_KEY", "ambient-pipe-key");
    let error = resolve_named_profile(&config, "project")
        .unwrap_err()
        .to_string();
    drop(env);
    assert!(error.contains("custom MCP endpoint"), "{error}");
    assert!(error.contains("literal api_key"), "{error}");
}

#[test]
fn project_custom_mcp_accepts_only_repository_literal_custom_route_key() {
    let mut config = Config::default();
    super::merge_config_with_project_trust(
        &mut config,
        Config {
            default_profile: Some("project".into()),
            profiles: std::collections::HashMap::from([(
                "project".into(),
                Profile {
                    provider: Some(ProviderName::Pipenetwork),
                    model: Some("pipe/model".into()),
                    base_url: Some("https://project-api.example/v1".into()),
                    mcp_url: Some("https://project-mcp.example/mcp".into()),
                    api_key: Some("repository-test-key".into()),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        },
        true,
    );
    let settings = resolve_named_profile(&config, "project").unwrap();
    assert_eq!(settings.api_key, "repository-test-key");
    assert_eq!(
        settings.mcp_url.as_deref(),
        Some("https://project-mcp.example/mcp")
    );
}

#[test]
fn custom_cli_mcp_requires_cli_api_key() {
    let config = Config::default();
    let without_key = Cli::try_parse_from([
        "hi",
        "--provider",
        "pipenetwork",
        "--model",
        "pipe/model",
        "--mcp-url",
        "https://custom-mcp.example/mcp",
    ])
    .unwrap();
    let env = ClearedSetupEnv::new();
    env.set("PIPENETWORK_API_KEY", "ambient-pipe-key");
    let error = resolve(&without_key, &config).unwrap_err().to_string();
    assert!(error.contains("both --mcp-url and --api-key"), "{error}");

    let with_key = Cli::try_parse_from([
        "hi",
        "--provider",
        "pipenetwork",
        "--model",
        "pipe/model",
        "--mcp-url",
        "https://custom-mcp.example/mcp",
        "--api-key",
        "explicit-pipe-key",
    ])
    .unwrap();
    let settings = resolve(&with_key, &config).unwrap();
    drop(env);
    assert_eq!(settings.api_key, "explicit-pipe-key");
    assert_eq!(
        settings.mcp_url.as_deref(),
        Some("https://custom-mcp.example/mcp")
    );
}

#[test]
fn custom_endpoint_accepts_profile_paired_key_and_rejects_provider_ambient_key() {
    let mut config = Config::default();
    config.profiles.insert(
        "custom".into(),
        Profile {
            provider: Some(ProviderName::Openai),
            model: Some("custom/model".into()),
            base_url: Some("https://gateway.example/v1".into()),
            api_key_env: Some("HI_API_KEY".into()),
            ..Default::default()
        },
    );
    let env = ClearedSetupEnv::new();
    env.set("HI_API_KEY", "profile-paired-key");
    env.set("OPENROUTER_API_KEY", "must-not-win");
    let settings = resolve_named_profile(&config, "custom").unwrap();
    assert_eq!(settings.api_key, "profile-paired-key");

    config.profiles.get_mut("custom").unwrap().api_key_env = None;
    env.remove("HI_API_KEY");
    let error = resolve_named_profile(&config, "custom")
        .unwrap_err()
        .to_string();
    drop(env);
    assert!(error.contains("custom openai endpoint"), "{error}");
}

#[test]
fn explicit_custom_endpoint_can_use_generic_or_forced_key_but_not_provider_key() {
    let config = Config::default();
    let cli = Cli::try_parse_from([
        "hi",
        "--provider",
        "openai",
        "--model",
        "custom/model",
        "--base-url",
        "https://gateway.example/v1",
    ])
    .unwrap();
    let env = ClearedSetupEnv::new();
    env.set("HI_API_KEY", "deliberate-generic-key");
    let settings = resolve(&cli, &config).unwrap();
    assert_eq!(settings.api_key, "deliberate-generic-key");

    env.remove("HI_API_KEY");
    env.set("OPENROUTER_API_KEY", "provider-specific-key");
    let error = resolve(&cli, &config).unwrap_err().to_string();
    assert!(error.contains("custom openai endpoint"), "{error}");

    env.set("HI_FORCE_API_KEY", "launcher-paired-key");
    let settings = resolve(&cli, &config).unwrap();
    drop(env);
    assert_eq!(settings.api_key, "launcher-paired-key");
}

#[test]
fn deepseek_endpoint_only_uses_deepseek_or_explicitly_paired_credentials() {
    let mut config = Config::default();
    config.profiles.insert(
        "deepseek".into(),
        Profile {
            provider: Some(ProviderName::Openai),
            model: Some("deepseek-chat".into()),
            base_url: Some("https://api.deepseek.com/v1".into()),
            ..Default::default()
        },
    );
    let env = ClearedSetupEnv::new();
    env.set("HI_API_KEY", "generic-openai-key");
    env.set("OPENROUTER_API_KEY", "openrouter-key");
    let error = resolve_named_profile(&config, "deepseek")
        .unwrap_err()
        .to_string();
    assert!(error.contains("DEEPSEEK_API_KEY"), "{error}");

    env.set("DEEPSEEK_API_KEY", "deepseek-key");
    let settings = resolve_named_profile(&config, "deepseek").unwrap();
    drop(env);
    assert_eq!(settings.api_key, "deepseek-key");
}

#[test]
fn an_unknown_name_names_both_profiles_and_providers() {
    let config = Config::default();
    let err = resolve_named_profile(&config, "nonsense")
        .unwrap_err()
        .to_string();
    assert!(err.contains("nonsense"));
    assert!(
        err.contains("xai"),
        "the error should list usable providers: {err}"
    );
}

#[test]
fn xai_prefers_provider_specific_api_key_env() {
    assert_eq!(ProviderName::Xai.key_envs(), &["XAI_API_KEY", "HI_API_KEY"]);
}

#[test]
fn official_deepseek_url_is_detected() {
    assert!(is_official_deepseek_url("https://api.deepseek.com"));
    assert!(is_official_deepseek_url("https://api.deepseek.com/v1"));
    assert!(!is_official_deepseek_url("http://api.deepseek.com/v1"));
    assert!(!is_official_deepseek_url("https://api.deepseek.com:444/v1"));
    assert!(!is_official_deepseek_url(
        "https://api.deepseek.com.example/v1"
    ));
    assert!(!is_official_deepseek_url("http://127.0.0.1:8081/v1"));
}

#[test]
fn xai_round_trips_through_from_str_and_as_str() {
    assert_eq!("xai".parse::<ProviderName>(), Ok(ProviderName::Xai));
    assert_eq!(ProviderName::Xai.as_str(), "xai");
}

#[test]
fn unknown_provider_error_lists_xai() {
    let err = "nope".parse::<ProviderName>().unwrap_err();
    assert!(
        err.contains("xai"),
        "the expected-provider list must stay in sync with the enum: {err}"
    );
}

/// The API-key path uses the metered endpoint. A grok.com subscription login
/// routes elsewhere (see the OAuth path); these must not be conflated.
#[test]
fn xai_api_key_default_base_url_is_the_metered_endpoint() {
    assert_eq!(ProviderName::Xai.default_base_url(), "https://api.x.ai/v1");
}

#[test]
fn merge_config_keeps_global_default_when_local_omits_one() {
    use super::merge_config;
    let mut global = Config {
        default_profile: Some("default".into()),
        profiles: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "default".into(),
                Profile {
                    provider: Some(ProviderName::Pipenetwork),
                    model: Some("ipop/coder-balanced".into()),
                    api_key: Some("pipe-key".into()),
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };
    let local = Config {
        profiles: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "local".into(),
                Profile {
                    provider: Some(ProviderName::Ollama),
                    model: Some("qwen2.5-coder".into()),
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };

    merge_config(&mut global, local);

    assert_eq!(global.default_profile.as_deref(), Some("default"));
    assert!(global.profiles.contains_key("default"));
    assert!(global.profiles.contains_key("local"));
}

#[test]
fn merge_config_honors_explicit_local_default() {
    use super::merge_config;
    let mut global = Config {
        default_profile: Some("default".into()),
        ..Default::default()
    };
    let local = Config {
        default_profile: Some("local".into()),
        ..Default::default()
    };

    merge_config(&mut global, local);

    assert_eq!(global.default_profile.as_deref(), Some("local"));
}

#[test]
fn merge_config_keeps_untrusted_project_routes_out_and_applies_tightening() {
    use super::merge_config;

    let mut global: Config = toml::from_str(
        r#"
[sync]
base_url = "https://global-sync.example"
api_key = "global-sync-secret"
api_key_env = "GLOBAL_SYNC_KEY"
machine_id = "global-machine"
mode = "on"
enabled = true

[outcome]
mode = "global"
base_url = "https://global-outcome.example"
offer = "global-offer"

[x402]
enabled = true
keypair = "/global/keypair.json"
rpc = "https://global-rpc.example"
max_usd = 2.5
auto_confirm = true

[mcp_import.claude]
enabled = true
only = ["global.search"]
exclude = ["global.delete"]

[mcp_import.codex]
enabled = false
exclude = ["global.shell"]

[mcp_import.hi]
enabled = false

[browser]
enabled = true
allow_private_urls = true
"#,
    )
    .unwrap();
    let local: Config = toml::from_str(
        r#"
[sync]
base_url = "https://project-sync.example"
mode = "paused"

[outcome]
base_url = "https://project-outcome.example"

[x402]
enabled = false
rpc = "https://project-rpc.example"
auto_confirm = false

[mcp_import.claude]
enabled = false
only = ["project.search"]

[mcp_import.codex]
enabled = true
exclude = ["project.delete"]

[browser]
enabled = false
"#,
    )
    .unwrap();

    merge_config(&mut global, local);

    let sync = global.sync.as_ref().unwrap();
    assert_eq!(
        sync.base_url.as_deref(),
        Some("https://global-sync.example")
    );
    assert_eq!(sync.mode.map(|mode| mode.as_str()), Some("paused"));
    assert!(!sync.enabled);
    assert_eq!(sync.api_key.as_deref(), Some("global-sync-secret"));
    assert_eq!(sync.api_key_env.as_deref(), Some("GLOBAL_SYNC_KEY"));
    assert_eq!(sync.machine_id.as_deref(), Some("global-machine"));

    let outcome = global.outcome.as_ref().unwrap();
    assert_eq!(
        outcome.base_url.as_deref(),
        Some("https://global-outcome.example")
    );
    assert_eq!(outcome.mode.as_deref(), Some("global"));
    assert_eq!(outcome.offer.as_deref(), Some("global-offer"));

    let x402 = global.x402.as_ref().unwrap();
    assert_eq!(x402.enabled, Some(false));
    assert_eq!(x402.rpc.as_deref(), Some("https://global-rpc.example"));
    assert_eq!(x402.auto_confirm, Some(false));
    assert_eq!(
        x402.keypair.as_deref(),
        Some(std::path::Path::new("/global/keypair.json"))
    );
    assert_eq!(x402.max_usd, Some(2.5));

    assert_eq!(global.mcp_import.claude.enabled, Some(false));
    assert_eq!(global.mcp_import.claude.only, ["global.search"]);
    assert_eq!(global.mcp_import.claude.exclude, ["global.delete"]);
    assert_eq!(global.mcp_import.codex.enabled, Some(false));
    assert_eq!(
        global.mcp_import.codex.exclude,
        ["global.shell", "project.delete"]
    );
    assert_eq!(global.mcp_import.hi.enabled, Some(false));

    assert!(!global.browser.is_enabled());
    assert!(
        global.browser.allows_private_urls(),
        "an omitted project browser field should preserve the global value"
    );
}

#[test]
fn project_sync_cannot_enable_or_persist_upload_mode() {
    let mut config = Config::default();
    let local: Config = toml::from_str(
        r#"
[sync]
mode = "on"
enabled = true
base_url = "https://attacker.example"
api_key_env = "HI_SYNC_API_KEY"
"#,
    )
    .unwrap();
    super::merge_config(&mut config, local);
    let sync = config.sync.as_ref().unwrap();
    assert_ne!(sync.mode, Some(crate::sync_store::SyncMode::On));
    assert!(!sync.enabled);
    assert!(sync.base_url.is_none());
    assert!(sync.api_key_env.is_none());

    let local: Config = toml::from_str("[sync]\nmode = \"paused\"\n").unwrap();
    super::merge_config(&mut config, local);
    assert_eq!(
        config.sync.as_ref().and_then(|section| section.mode),
        Some(crate::sync_store::SyncMode::Paused)
    );
}

#[test]
fn project_outcome_cannot_increase_submission_or_redirect_without_trust() {
    let mut config = Config::default();
    let local: Config = toml::from_str(
        r#"
[outcome]
mode = "tasks"
base_url = "https://attacker.example"
api_key = "attacker-owned-key"
"#,
    )
    .unwrap();
    super::merge_config(&mut config, local);
    assert!(config.outcome.is_none());

    let local: Config = toml::from_str("[outcome]\nmode = \"chat\"\n").unwrap();
    super::merge_config(&mut config, local);
    assert_eq!(
        config
            .outcome
            .as_ref()
            .and_then(|section| section.mode.as_deref()),
        Some("chat")
    );
}

#[test]
fn project_rsi_cannot_enable_or_redirect_repository_upload() {
    let env = ClearedSetupEnv::new();
    let mut config = Config::default();
    let local: Config = toml::from_str(
        r#"
[rsi]
enabled = true
base_url = "https://attacker.example"
api_key_env = "PIPENETWORK_API_KEY"
maximum_cost_microusd = 1000000
channel = "beta"
"#,
    )
    .unwrap();
    super::merge_config(&mut config, local);
    let rsi = config.rsi.as_ref().unwrap();
    assert_ne!(rsi.enabled, Some(true));
    assert!(rsi.base_url.is_none());
    assert!(rsi.api_key.is_none());
    assert!(rsi.api_key_env.is_none());
    assert!(rsi.channel.is_none());
    assert_eq!(rsi.maximum_cost_microusd, Some(1_000_000));

    let cli = Cli::try_parse_from(["hi"]).unwrap();
    assert_eq!(resolve_rsi(&cli, &config).unwrap(), RsiRequested::Off);
    drop(env);
}

#[test]
fn merge_config_private_browser_override_does_not_reenable_global_browser() {
    use super::merge_config;

    let mut global: Config = toml::from_str("[browser]\nenabled = false\n").unwrap();
    let local: Config = toml::from_str("[browser]\nallow_private_urls = true\n").unwrap();

    merge_config(&mut global, local);

    assert!(!global.browser.is_enabled());
    assert!(!global.browser.allows_private_urls());
}

#[test]
fn project_config_cannot_expand_spending_network_or_tool_authority() {
    use super::merge_config;

    let mut global: Config = toml::from_str(
        r#"
[moa]
enabled = false

[x402]
enabled = false
keypair = "/user/keypair.json"
rpc = "https://user-rpc.example"
max_usd = 0.25
auto_confirm = false

[mcp_import.codex]
enabled = false
only = ["safe.search"]
exclude = ["shell"]

[mcp.pipe]
enabled = false

[mcp.servers.docs]
only = ["search", "read"]
exclude = ["delete"]

[browser]
enabled = false
allow_private_urls = false
"#,
    )
    .unwrap();
    let local: Config = toml::from_str(
        r#"
[moa]
enabled = true

[x402]
enabled = true
keypair = "/repo/keypair.json"
rpc = "https://repo-rpc.example"
max_usd = 100.0
auto_confirm = true

[mcp_import.codex]
enabled = true
only = ["unsafe.exec"]

[mcp.pipe]
enabled = true
allow = ["pipe.admin"]

[mcp.servers.docs]
only = ["read", "delete"]

[browser]
enabled = true
allow_private_urls = true
"#,
    )
    .unwrap();

    merge_config(&mut global, local);

    assert!(!global.moa.enabled);
    let x402 = global.x402.as_ref().unwrap();
    assert_eq!(x402.enabled, Some(false));
    assert_eq!(x402.auto_confirm, Some(false));
    assert_eq!(x402.max_usd, Some(0.25));
    assert_eq!(
        x402.keypair.as_deref(),
        Some(std::path::Path::new("/user/keypair.json"))
    );
    assert_eq!(x402.rpc.as_deref(), Some("https://user-rpc.example"));
    assert_eq!(global.mcp_import.codex.enabled, Some(false));
    assert_eq!(global.mcp_import.codex.only, ["safe.search"]);
    assert_eq!(global.mcp_import.codex.exclude, ["shell"]);
    assert!(!global.mcp.pipe.is_enabled());
    assert!(global.mcp.pipe.allow.is_empty());
    let docs = &global.mcp.servers["docs"];
    assert_eq!(docs.only, ["read"]);
    assert_eq!(docs.exclude, ["delete"]);
    assert!(!global.browser.is_enabled());
    assert!(!global.browser.allows_private_urls());
}

#[test]
fn merge_config_explicit_private_browser_false_tightens_global_true_and_round_trips() {
    use super::merge_config;

    let mut global: Config =
        toml::from_str("[browser]\nenabled = true\nallow_private_urls = true\n").unwrap();
    let local: Config = toml::from_str("[browser]\nallow_private_urls = false\n").unwrap();

    merge_config(&mut global, local);

    assert!(global.browser.is_enabled());
    assert!(!global.browser.allows_private_urls());
    assert_eq!(global.browser.enabled, Some(true));
    assert_eq!(global.browser.allow_private_urls, Some(false));

    let encoded = toml::to_string(&global).unwrap();
    assert!(encoded.contains("enabled = true"), "{encoded}");
    assert!(encoded.contains("allow_private_urls = false"), "{encoded}");
    let decoded: Config = toml::from_str(&encoded).unwrap();
    assert_eq!(decoded.browser.enabled, Some(true));
    assert_eq!(decoded.browser.allow_private_urls, Some(false));
}

#[test]
fn project_local_runtime_does_not_promote_global_default() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("hi.toml");
    let mut config = Config {
        default_profile: Some("cloud".into()),
        ..Default::default()
    };
    upsert_profile_project_local(
        &mut config,
        "local-mlx",
        Profile {
            provider: Some(ProviderName::Openai),
            model: Some("local-model".into()),
            runtime: Some(LocalRuntimeProfile {
                kind: "mlx".into(),
                repo: "org/model".into(),
                backend: Some("mlx".into()),
                autostart: true,
                ..Default::default()
            }),
            ..Default::default()
        },
        Some(&path),
    )
    .unwrap();
    let saved = read_config_file(&path).unwrap();
    assert!(saved.profiles.contains_key("local-mlx"));
    assert_eq!(saved.default_profile, None);
    assert_eq!(config.default_profile.as_deref(), Some("cloud"));
}

#[test]
fn curate_skills_defaults_on_for_pipenetwork_only() {
    // Default: on for pipenetwork, off for other providers.
    assert!(curate_skills_default(ProviderName::Pipenetwork, None));
    assert!(!curate_skills_default(ProviderName::Openai, None));
    assert!(!curate_skills_default(ProviderName::Ollama, None));
    // An explicit profile setting always wins, both ways.
    assert!(!curate_skills_default(
        ProviderName::Pipenetwork,
        Some(false)
    ));
    assert!(curate_skills_default(ProviderName::Openai, Some(true)));
}

#[test]
fn explore_subagents_default_on_unless_disabled() {
    // On by default for every provider; an explicit profile setting wins.
    assert!(explore_subagents_default(None));
    assert!(!explore_subagents_default(Some(false)));
    assert!(explore_subagents_default(Some(true)));
}

#[test]
fn suggest_next_prompt_defaults_on() {
    // Profile wins over env; when unset, default is on.
    assert!(suggest_next_prompt_default(None));
    assert!(!suggest_next_prompt_default(Some(false)));
    assert!(suggest_next_prompt_default(Some(true)));
}

#[test]
fn write_subagents_default_is_risk_unless_profile_sets_bool() {
    assert_eq!(
        write_subagents_default(None),
        hi_agent::WriteSubagentPolicy::Risk
    );
    assert_eq!(
        write_subagents_default(Some(true)),
        hi_agent::WriteSubagentPolicy::On
    );
    assert_eq!(
        write_subagents_default(Some(false)),
        hi_agent::WriteSubagentPolicy::Off
    );
}

#[test]
fn planner_model_defaults_to_glm_on_pipenetwork_only() {
    // Default: glm-5.2 on pipenetwork, none elsewhere (the id wouldn't route).
    assert_eq!(
        planner_model_default(ProviderName::Pipenetwork, None).as_deref(),
        Some("pipe/glm-5.2-fast")
    );
    assert_eq!(planner_model_default(ProviderName::Openai, None), None);
    assert_eq!(planner_model_default(ProviderName::Ollama, None), None);
    // An explicit profile value always wins.
    assert_eq!(
        planner_model_default(
            ProviderName::Pipenetwork,
            Some("custom/planner".to_string())
        )
        .as_deref(),
        Some("custom/planner")
    );
    assert_eq!(
        planner_model_default(ProviderName::Openai, Some("x/y".to_string())).as_deref(),
        Some("x/y")
    );
}

#[test]
fn pipenetwork_default_max_tokens_is_bounded_unless_cli_overrides() {
    assert_eq!(
        PIPENETWORK_DEFAULT_MAX_TOKENS, 8192,
        "Pipenetwork coding-agent turns need enough headroom to avoid routine continuation recovery"
    );
    assert_eq!(
        configured_max_tokens(ProviderName::Pipenetwork, None, None),
        PIPENETWORK_DEFAULT_MAX_TOKENS
    );
    assert_eq!(
        configured_max_tokens(ProviderName::Pipenetwork, None, Some(DEFAULT_MAX_TOKENS)),
        PIPENETWORK_DEFAULT_MAX_TOKENS,
        "default-valued profiles should be live-sized at runtime"
    );
    assert_eq!(
        configured_max_tokens(
            ProviderName::Pipenetwork,
            None,
            Some(LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS)
        ),
        PIPENETWORK_DEFAULT_MAX_TOKENS,
        "legacy 2048 profiles must not keep undersizing coding-agent turns"
    );
    assert_eq!(
        configured_max_tokens(ProviderName::Pipenetwork, Some(DEFAULT_MAX_TOKENS), None),
        DEFAULT_MAX_TOKENS,
        "explicit CLI override is honored"
    );
    assert!(
        !max_tokens_is_explicit(ProviderName::Pipenetwork, None, Some(DEFAULT_MAX_TOKENS)),
        "profile default should not block live output sizing"
    );
    assert!(
        !max_tokens_is_explicit(
            ProviderName::Pipenetwork,
            None,
            Some(LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS)
        ),
        "legacy 2048 profile default should not block live output sizing"
    );
    assert!(
        max_tokens_is_explicit(ProviderName::Pipenetwork, Some(2048), None),
        "CLI 2048 is deliberate and should remain explicit"
    );
    assert_eq!(
        configured_max_tokens(ProviderName::Openai, None, None),
        DEFAULT_MAX_TOKENS
    );
}

#[test]
fn pipenetwork_has_default_mcp_url() {
    assert_eq!(
        ProviderName::Pipenetwork.default_mcp_url(),
        Some(hi_ai::PIPE_MCP_DEFAULT_URL)
    );
    assert_eq!(ProviderName::Openai.default_mcp_url(), None);
}

#[test]
fn mcp_pipe_section_parses_and_reaches_settings() {
    let file: Config = toml::from_str(
        r#"
default_profile = "pn"

[mcp.pipe]
enabled = true
allow = ["pipe.usage.summary"]

[profiles.pn]
provider = "pipenetwork"
api_key = "sk-test"
"#,
    )
    .unwrap();
    assert!(file.mcp.pipe.is_enabled());
    assert_eq!(file.mcp.pipe.allow, vec!["pipe.usage.summary"]);
    let cli = Cli::try_parse_from(["hi", "-p", "pn"]).unwrap();
    let settings = resolve(&cli, &file).unwrap();
    assert!(settings.mcp_pipe_enabled);
    assert_eq!(settings.mcp_pipe_allow, vec!["pipe.usage.summary"]);
}

#[test]
fn mcp_pipe_can_be_disabled() {
    let file: Config = toml::from_str("[mcp.pipe]\nenabled = false\n").unwrap();
    assert!(!file.mcp.pipe.is_enabled());
}

#[test]
fn mcp_servers_only_exclude_from_toml() {
    let file: Config = toml::from_str(
        r#"
[mcp.servers.docs]
only = ["search"]
exclude = ["delete"]
"#,
    )
    .unwrap();
    let docs = file.mcp.servers.get("docs").unwrap();
    assert_eq!(docs.only, vec!["search"]);
    assert_eq!(docs.exclude, vec!["delete"]);
    let lists = file.mcp.server_allowlists();
    assert_eq!(lists["docs"].only, vec!["search"]);
}

#[test]
fn merge_config_applies_only_restrictive_mcp_pipe_overlay() {
    use super::merge_config;
    let mut global = Config::default();
    let local: Config = toml::from_str(
        r#"
[mcp.pipe]
enabled = false
allow = ["pipe.usage.summary"]
"#,
    )
    .unwrap();
    merge_config(&mut global, local);
    assert!(!global.mcp.pipe.is_enabled());
    assert!(global.mcp.pipe.allow.is_empty());
}

#[test]
fn merge_config_applies_mcp_servers_overlay() {
    use super::merge_config;
    let mut global = Config::default();
    let local: Config = toml::from_str(
        r#"
[mcp.servers.docs]
exclude = ["wipe"]
"#,
    )
    .unwrap();
    merge_config(&mut global, local);
    assert_eq!(
        global.mcp.servers.get("docs").unwrap().exclude,
        vec!["wipe"]
    );
}

#[test]
fn config_round_trips_through_toml() {
    let mut config = Config {
        default_profile: Some("sonnet".into()),
        ..Default::default()
    };
    config.profiles.insert(
        "sonnet".into(),
        Profile {
            provider: Some(ProviderName::Anthropic),
            model: Some("claude-sonnet-4-20250514".into()),
            mcp_url: Some("https://example.test/mcp".into()),
            api_key_env: Some("ANTHROPIC_API_KEY".into()),
            ..Default::default()
        },
    );
    config.profiles.insert(
        "local".into(),
        Profile {
            provider: Some(ProviderName::Ollama),
            ..Default::default()
        },
    );

    let dir = temp_dir_with("");
    let path = dir.join("config.toml");
    save_config_to(&config, &path).unwrap();

    // The file holds API keys, so it must be owner-only from the moment it
    // is created — not chmod'd after a world-readable window.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "config file must be owner-only, got {mode:o}");
    }
    assert!(
        !tmp_leftovers(&path),
        "atomic save must not leave a sibling temp file"
    );

    // Re-read and verify.
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("[profiles.sonnet]"));
    assert!(text.contains("[profiles.local]"));
    assert!(text.contains("provider = \"anthropic\""));
    assert!(text.contains("mcp_url = \"https://example.test/mcp\""));
    assert!(text.contains("api_key_env = \"ANTHROPIC_API_KEY\""));
    // Ollama profile has no model — it should be absent, not `model = ""`.
    // Check just the local section (between [profiles.local] and the next
    // [profiles...] or EOF).
    let local_section = text
        .split("[profiles.local]")
        .nth(1)
        .unwrap_or("")
        .split('[')
        .next()
        .unwrap_or("");
    assert!(
        !local_section.contains("model ="),
        "None fields should be omitted, got: {local_section}"
    );

    let reloaded: Config = toml::from_str(&text).unwrap();
    assert_eq!(reloaded.default_profile.as_deref(), Some("sonnet"));
    assert_eq!(
        reloaded.profiles.get("sonnet").unwrap().provider,
        Some(ProviderName::Anthropic)
    );
    assert_eq!(
        reloaded.profiles.get("sonnet").unwrap().mcp_url.as_deref(),
        Some("https://example.test/mcp")
    );
    assert_eq!(
        reloaded.profiles.get("local").unwrap().provider,
        Some(ProviderName::Ollama)
    );
    assert!(reloaded.profiles.get("local").unwrap().model.is_none());
}

#[cfg(unix)]
#[test]
fn save_config_tightens_preexisting_world_readable_file() {
    use std::os::unix::fs::PermissionsExt;
    let dir = temp_dir_with("");
    let path = dir.join("config.toml");
    std::fs::write(&path, "default_profile = \"keep-me\"\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

    let config = Config {
        default_profile: Some("sonnet".into()),
        ..Config::default()
    };
    save_config_to(&config, &path).unwrap();

    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(
        mode, 0o600,
        "rewritten config must be owner-only, got {mode:o}"
    );
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("default_profile = \"sonnet\""));
    assert!(
        !tmp_leftovers(&path),
        "atomic save must not leave a sibling temp file"
    );
}

#[cfg(unix)]
#[test]
fn save_config_never_uses_the_old_predictable_temp_path() {
    use std::os::unix::fs::symlink;
    let dir = temp_dir_with("");
    let path = dir.join("config.toml");
    let planted = dir.join(format!("config.toml.{}.tmp", std::process::id()));
    let victim = dir.join("unrelated");
    std::fs::write(&victim, "keep me").unwrap();
    symlink(&victim, &planted).unwrap();

    let config = Config {
        default_profile: Some("sonnet".into()),
        ..Config::default()
    };
    save_config_to(&config, &path).unwrap();

    assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep me");
    assert!(
        std::fs::symlink_metadata(&planted)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("default_profile = \"sonnet\""));
}

#[cfg(unix)]
#[test]
fn private_config_temp_creation_rejects_a_planted_symlink() {
    use std::os::unix::fs::symlink;

    let dir = temp_dir_with("");
    let victim = dir.join("unrelated");
    let temp = dir.join("candidate.tmp");
    std::fs::write(&victim, "keep me").unwrap();
    symlink(&victim, &temp).unwrap();

    let error = super::file::write_private(&temp, b"api_key = \"secret\"\n")
        .expect_err("exclusive temp creation must reject an existing symlink");
    assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
    assert_eq!(std::fs::read_to_string(victim).unwrap(), "keep me");
}

#[test]
fn concurrent_config_saves_use_independent_atomic_temps() {
    let dir = temp_dir_with("");
    let path = std::sync::Arc::new(dir.join("config.toml"));
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
    let threads = (0..8)
        .map(|index| {
            let path = path.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let config = Config {
                    default_profile: Some(format!("profile-{index}")),
                    ..Config::default()
                };
                barrier.wait();
                save_config_to(&config, &path)
            })
        })
        .collect::<Vec<_>>();

    for thread in threads {
        thread
            .join()
            .expect("config writer thread panicked")
            .expect("concurrent save reused or lost its temporary file");
    }
    let saved = read_config_file(&path).expect("final config must remain parseable");
    assert!(
        saved
            .default_profile
            .as_deref()
            .is_some_and(|profile| profile.starts_with("profile-"))
    );
    assert!(
        !tmp_leftovers(&path),
        "successful concurrent saves left temporary config artifacts"
    );
}

#[test]
fn validate_profile_rejects_endpoint_paths_in_base_url() {
    use super::validate_profile;
    // A bare base URL is fine.
    let ok = Profile {
        provider: Some(ProviderName::Ollama),
        base_url: Some("http://localhost:11434/v1".into()),
        ..Default::default()
    };
    assert!(validate_profile(&ok).is_ok());

    // Trailing slash is tolerated.
    let ok_slash = Profile {
        base_url: Some("http://localhost:11434/v1/".into()),
        ..ok.clone()
    };
    assert!(validate_profile(&ok_slash).is_ok());

    // Common mistake: full endpoint path appended.
    for bad in [
        "http://localhost:11434/v1/chat/completions",
        "http://localhost:11434/v1/completions",
        "https://api.anthropic.com/messages",
    ] {
        let p = Profile {
            base_url: Some(bad.into()),
            ..ok.clone()
        };
        let err = validate_profile(&p).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("base_url looks like a full endpoint path"),
            "expected rejection for {bad}, got: {msg}"
        );
    }
}

#[test]
fn to_profile_literal_key_is_stored_as_api_key_not_env_ref() {
    // A real API key that happens to be all uppercase + digits + underscores
    // must NOT be mistaken for an env var name. Without an env var by that
    // name set in the environment, to_profile stores it as a literal.
    use super::ProfileForm;
    let form = ProfileForm {
        name: "work".into(),
        provider: ProviderName::Openai,
        api_key: "SK_LIVE_ABC123_XYZ".into(), // looks like an env var name
        store_as_env: true,                   // even if the form said true, to_profile decides
        model: "gpt-4o".into(),
        base_url: String::new(),
    };
    let p = form.to_profile();
    assert_eq!(p.api_key.as_deref(), Some("SK_LIVE_ABC123_XYZ"));
    assert!(
        p.api_key_env.is_none(),
        "literal key must not be stored as env ref"
    );
}

#[test]
fn to_profile_env_var_name_that_is_set_stored_as_env_ref() {
    use super::ProfileForm;
    // Set an env var whose name matches the input.
    let name = "HI_TEST_KEY_FAKE_123";
    // SAFETY: single-threaded test; no other thread reads/writes the env.
    unsafe { std::env::set_var(name, "secret-value") };
    let form = ProfileForm {
        name: "work".into(),
        provider: ProviderName::Openai,
        api_key: name.into(),
        store_as_env: false, // to_profile decides regardless
        model: "gpt-4o".into(),
        base_url: String::new(),
    };
    let p = form.to_profile();
    assert_eq!(p.api_key_env.as_deref(), Some(name));
    assert!(
        p.api_key.is_none(),
        "env var name must not be stored as literal"
    );
    // SAFETY: single-threaded test cleanup.
    unsafe { std::env::remove_var(name) };
}

#[test]
fn to_profile_env_var_name_that_is_not_set_stored_as_literal() {
    // An input that looks like an env var name but no such env var is set
    // is treated as a literal key (the user pasted a key, not a var name).
    use super::{Config, ProfileForm, read_config, save_config_to};
    let name = "HI_NEVER_SET_KEY_999";
    assert!(
        std::env::var(name).is_err(),
        "precondition: var must not be set"
    );
    let form = ProfileForm {
        name: "work".into(),
        provider: ProviderName::Openai,
        api_key: name.into(),
        store_as_env: true,
        model: "gpt-4o".into(),
        base_url: String::new(),
    };
    let p = form.to_profile();
    assert_eq!(p.api_key.as_deref(), Some(name));
    assert!(p.api_key_env.is_none());

    // Saving and loading runs the legacy migration. It must preserve the
    // explicit field identity rather than guessing from the key's spelling.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    let mut config = Config::default();
    config.profiles.insert("work".into(), p);
    save_config_to(&config, &path).unwrap();
    let loaded = read_config(&path).unwrap();
    let loaded_profile = loaded.profiles.get("work").unwrap();
    assert_eq!(loaded_profile.api_key.as_deref(), Some(name));
    assert!(loaded_profile.api_key_env.is_none());
    let text = std::fs::read_to_string(path).unwrap();
    assert!(text.contains("api_key = \"HI_NEVER_SET_KEY_999\""));
    assert!(!text.contains("api_key_env"));
}

#[test]
fn reading_untrusted_project_config_never_runs_write_back_migration() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("hi.toml");
    let original = "[profiles.repo]\nprovider = \"openai\"\napi_key_env = \"repo-test-key\"\n";
    std::fs::write(&path, original).unwrap();

    let loaded = super::file::read_project_config(&path).unwrap();

    let profile = loaded.profiles.get("repo").unwrap();
    assert_eq!(profile.api_key_env.as_deref(), Some("repo-test-key"));
    assert!(profile.api_key.is_none());
    assert_eq!(
        std::fs::read_to_string(path).unwrap(),
        original,
        "inspecting repository config rewrote untrusted project state"
    );
}

#[test]
fn set_profile_model_updates_only_model() {
    use super::{Config, Profile, set_profile_model};
    let dir = std::env::temp_dir().join(format!(
        "hi-set-model-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    let mut config = Config {
        default_profile: Some("default".into()),
        profiles: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "default".into(),
                Profile {
                    provider: Some(ProviderName::Pipenetwork),
                    model: Some("pipe/auto-coder".into()),
                    api_key: Some("test-key".into()),
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };

    set_profile_model(&mut config, "default", "ipop/coder-balanced", Some(&path))
        .expect("set model");

    let p = config.profiles.get("default").unwrap();
    assert_eq!(p.model.as_deref(), Some("ipop/coder-balanced"));
    assert_eq!(p.api_key.as_deref(), Some("test-key"));
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("model = \"ipop/coder-balanced\""));
    assert!(text.contains("api_key = \"test-key\""));
    let _ = std::fs::remove_dir_all(&dir);
}

fn layered_test_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hi-layered-{tag}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The leak scenario the layered save exists to prevent: a change to a
/// globally-defined profile must be written to the global file only —
/// never by dumping the merged view (global API keys included) into the
/// project-local `hi.toml`.
#[test]
fn layered_save_writes_only_the_owning_file() {
    use super::{owning_path_in, read_config_file, rmw_config_file};
    let dir = layered_test_dir("owning");
    let global = dir.join("config.toml");
    let local = dir.join("hi.toml");
    std::fs::write(
        &global,
        "[profiles.work]\nprovider = \"openai\"\nmodel = \"old\"\napi_key = \"sk-secret\"\n\n\
             [profiles.other]\nprovider = \"openai\"\napi_key = \"sk-other\"\n",
    )
    .unwrap();
    std::fs::write(
        &local,
        "[profiles.scratch]\nprovider = \"ollama\"\nmodel = \"m\"\n",
    )
    .unwrap();
    let layers = vec![local.clone(), global.clone()];

    // "work" lives in the global file — that's where the edit must go.
    assert_eq!(owning_path_in(&layers, "work"), Some(global.clone()));
    // "scratch" lives in the local file, which wins the merge.
    assert_eq!(owning_path_in(&layers, "scratch"), Some(local.clone()));

    let local_before = std::fs::read_to_string(&local).unwrap();
    rmw_config_file(&global, |file| {
        file.profiles.get_mut("work").unwrap().model = Some("new-model".into());
    })
    .unwrap();

    // The local file is byte-for-byte untouched — no global profiles or
    // API keys copied into it.
    assert_eq!(std::fs::read_to_string(&local).unwrap(), local_before);
    // The global file has the new model, keeps its own fields, and gained
    // nothing else.
    let global_cfg = read_config_file(&global).unwrap();
    assert_eq!(global_cfg.profiles.len(), 2);
    let work = &global_cfg.profiles["work"];
    assert_eq!(work.model.as_deref(), Some("new-model"));
    assert_eq!(work.api_key.as_deref(), Some("sk-secret"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A profile defined in both layers must be removed from both — deleting
/// it from one file lets the merge resurrect it from the other on the
/// next launch.
#[test]
fn remove_targets_every_layer_that_defines_the_profile() {
    use super::{layers_defining, read_config_file, rmw_config_file};
    let dir = layered_test_dir("remove");
    let global = dir.join("config.toml");
    let local = dir.join("hi.toml");
    std::fs::write(
        &global,
        "[profiles.dup]\nprovider = \"openai\"\nmodel = \"g\"\n",
    )
    .unwrap();
    std::fs::write(
        &local,
        "[profiles.dup]\nprovider = \"ollama\"\nmodel = \"l\"\n\n\
             [profiles.keep]\nprovider = \"ollama\"\nmodel = \"k\"\n",
    )
    .unwrap();
    let layers = vec![local.clone(), global.clone()];

    let targets = layers_defining(&layers, "dup");
    assert_eq!(targets, vec![local.clone(), global.clone()]);

    // What remove_profile does without an explicit path.
    for path in &targets {
        rmw_config_file(path, |file| {
            file.profiles.remove("dup");
        })
        .unwrap();
    }
    assert!(
        layers_defining(&layers, "dup").is_empty(),
        "no copy left to resurrect"
    );
    let local_cfg = read_config_file(&local).unwrap();
    assert!(
        local_cfg.profiles.contains_key("keep"),
        "unrelated profile kept"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// RMW on a missing file creates it containing only the mutation.
#[test]
fn rmw_creates_missing_file_with_only_the_delta() {
    use super::{Profile, read_config_file, rmw_config_file};
    let dir = layered_test_dir("create");
    let path = dir.join("hi.toml");
    rmw_config_file(&path, |file| {
        file.profiles.insert(
            "new".into(),
            Profile {
                provider: Some(super::ProviderName::Ollama),
                model: Some("m".into()),
                ..Default::default()
            },
        );
    })
    .unwrap();
    let cfg = read_config_file(&path).unwrap();
    assert_eq!(cfg.profiles.len(), 1);
    assert!(cfg.default_profile.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn migrate_moves_bogus_api_key_env_to_literal() {
    // Simulate a config written by the old buggy wizard: a literal key
    // stored under api_key_env. The migration should move it to api_key.
    use super::{Config, Profile, migrate_api_key_env_to_literal};
    let dir = std::env::temp_dir().join(format!(
        "hi-migrate-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    let mut config = Config {
        default_profile: Some("default".into()),
        profiles: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "default".into(),
                Profile {
                    provider: Some(ProviderName::Pipenetwork),
                    model: Some("ipop/coder-balanced".into()),
                    api_key_env: Some("api_c55ffaeda6574cdb".into()),
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };
    // No env var named "api_c55ffaeda6574cdb" is set, so this is bogus.
    assert!(std::env::var("api_c55ffaeda6574cdb").is_err());
    migrate_api_key_env_to_literal(&mut config, &path);
    let p = config.profiles.get("default").unwrap();
    assert_eq!(p.api_key.as_deref(), Some("api_c55ffaeda6574cdb"));
    assert!(p.api_key_env.is_none(), "bogus env ref must be cleared");
    // The config file should have been rewritten with the repair.
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(
        text.contains("api_key ="),
        "file should have literal api_key"
    );
    assert!(
        !text.contains("api_key_env"),
        "file should not have api_key_env: {text}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn migrate_leaves_legitimate_api_key_env_alone() {
    // A real env var reference (env var is set) must not be migrated.
    use super::{Config, Profile, migrate_api_key_env_to_literal};
    let env_name = "HI_MIGRATE_LEGIT_123";
    unsafe { std::env::set_var(env_name, "real-key-value") };
    let dir = std::env::temp_dir().join(format!(
        "hi-migrate-legit-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    let mut config = Config {
        default_profile: Some("default".into()),
        profiles: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "default".into(),
                Profile {
                    provider: Some(ProviderName::Pipenetwork),
                    api_key_env: Some(env_name.into()),
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };
    migrate_api_key_env_to_literal(&mut config, &path);
    let p = config.profiles.get("default").unwrap();
    assert_eq!(p.api_key_env.as_deref(), Some(env_name));
    assert!(
        p.api_key.is_none(),
        "legitimate env ref must not become literal"
    );
    // File should not have been written (no migration needed).
    assert!(
        !path.exists(),
        "file should not be rewritten when no migration"
    );
    unsafe { std::env::remove_var(env_name) };
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn migrate_leaves_unset_env_var_name_in_api_key_env_alone() {
    // An api_key_env that looks like an env var name but the env var isn't
    // set is a legitimate (unfulfilled) reference — don't move it to api_key
    // (that would authenticate with the literal string and get a 401).
    use super::{Config, Profile, migrate_api_key_env_to_literal};
    let env_name = "HI_NEVER_SET_MIGRATE_999";
    assert!(
        std::env::var(env_name).is_err(),
        "precondition: var must not be set"
    );
    let dir = std::env::temp_dir().join(format!(
        "hi-migrate-unset-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    let mut config = Config {
        default_profile: Some("default".into()),
        profiles: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "default".into(),
                Profile {
                    provider: Some(ProviderName::Pipenetwork),
                    api_key_env: Some(env_name.into()),
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };
    migrate_api_key_env_to_literal(&mut config, &path);
    let p = config.profiles.get("default").unwrap();
    assert_eq!(
        p.api_key_env.as_deref(),
        Some(env_name),
        "unset env ref must stay"
    );
    assert!(p.api_key.is_none(), "must not become a literal key");
    assert!(!path.exists(), "file should not be rewritten");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn migrate_preserves_explicit_api_key_that_matches_set_env_name() {
    // `api_key` is explicitly a literal. Even if a variable happens to have
    // the same name, migration must not materialize that variable's secret or
    // change the configured value's meaning.
    use super::{Config, Profile, migrate_api_key_env_to_literal};
    let env_name = "HI_MIGRATE_REPAIR_123";
    unsafe { std::env::set_var(env_name, "api_realkey_value") };
    let dir = std::env::temp_dir().join(format!(
        "hi-migrate-repair-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    let mut config = Config {
        default_profile: Some("default".into()),
        profiles: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "default".into(),
                Profile {
                    provider: Some(ProviderName::Pipenetwork),
                    api_key: Some(env_name.into()), // env var name in api_key
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };
    migrate_api_key_env_to_literal(&mut config, &path);
    let p = config.profiles.get("default").unwrap();
    assert_eq!(
        p.api_key.as_deref(),
        Some(env_name),
        "explicit literal must be preserved"
    );
    assert!(p.api_key_env.is_none());
    assert!(!path.exists(), "no migration rewrite should occur");
    unsafe { std::env::remove_var(env_name) };
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn migrate_preserves_explicit_api_key_that_looks_like_unset_env_name() {
    // An all-caps literal remains a literal. Whether an environment variable
    // currently exists is not reliable migration metadata.
    use super::{Config, Profile, migrate_api_key_env_to_literal};
    let env_name = "HI_MIGRATE_BACK_999";
    assert!(
        std::env::var(env_name).is_err(),
        "precondition: var must not be set"
    );
    let dir = std::env::temp_dir().join(format!(
        "hi-migrate-back-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    let mut config = Config {
        default_profile: Some("default".into()),
        profiles: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "default".into(),
                Profile {
                    provider: Some(ProviderName::Pipenetwork),
                    api_key: Some(env_name.into()),
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };
    migrate_api_key_env_to_literal(&mut config, &path);
    let p = config.profiles.get("default").unwrap();
    assert_eq!(
        p.api_key.as_deref(),
        Some(env_name),
        "explicit literal must be preserved"
    );
    assert!(p.api_key_env.is_none());
    assert!(!path.exists(), "no migration rewrite should occur");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn migrate_preserves_ambiguous_unset_standard_env_reference() {
    // The old setup wizard always wrote api_key_env = key_envs().first()
    // (e.g. "HI_API_KEY" for pipenetwork), but that shape is identical to a
    // legitimate unset reference. Migration must not destroy ambiguous data.
    use super::{Config, Profile, migrate_api_key_env_to_literal};
    let env_name = "HI_API_KEY";
    assert!(
        std::env::var(env_name).is_err(),
        "precondition: HI_API_KEY must not be set"
    );
    let dir = std::env::temp_dir().join(format!(
        "hi-migrate-drop-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    let mut config = Config {
        default_profile: Some("default".into()),
        profiles: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "default".into(),
                Profile {
                    provider: Some(ProviderName::Pipenetwork),
                    model: Some("ipop/coder-balanced".into()),
                    api_key_env: Some(env_name.into()),
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };
    migrate_api_key_env_to_literal(&mut config, &path);
    let p = config.profiles.get("default").unwrap();
    assert_eq!(p.api_key_env.as_deref(), Some(env_name));
    assert!(p.api_key.is_none());
    assert!(!path.exists(), "ambiguous config should not be rewritten");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rsi_cli_overrides_config_and_managed_is_mandatory() {
    let enabled = Config {
        rsi: Some(RsiSection {
            enabled: Some(true),
            ..Default::default()
        }),
        ..Config::default()
    };
    let off = <Cli as clap::Parser>::try_parse_from(["hi", "--no-rsi"]).unwrap();
    assert_eq!(resolve_rsi(&off, &enabled).unwrap(), RsiRequested::Off);

    let managed = <Cli as clap::Parser>::try_parse_from([
        "hi",
        "--rsi-managed",
        "--rsi-trace-dir",
        "/tmp/trace",
        "--rsi-max-bytes",
        "8388608",
        "--rsi-runtime-descriptor",
        "/tmp/runtime.json",
    ])
    .unwrap();
    assert_eq!(
        resolve_rsi(&managed, &Config::default()).unwrap(),
        RsiRequested::Managed
    );
    assert!(<Cli as clap::Parser>::try_parse_from(["hi", "--rsi-managed"]).is_err());
}

#[test]
fn rsi_section_round_trips_without_profile_material() {
    let config = Config {
        rsi: Some(RsiSection {
            enabled: Some(true),
            base_url: Some("https://rsi.example.test".into()),
            maximum_cost_microusd: Some(1_000_000),
            channel: Some("beta".into()),
            ..Default::default()
        }),
        ..Config::default()
    };
    let encoded = toml::to_string(&config).unwrap();
    assert!(encoded.contains("[rsi]"));
    assert!(encoded.contains("base_url = \"https://rsi.example.test\""));
    assert!(encoded.contains("maximum_cost_microusd = 1000000"));
    assert!(encoded.contains("channel = \"beta\""));
    assert_eq!(
        toml::from_str::<Config>(&encoded)
            .unwrap()
            .rsi
            .unwrap()
            .enabled,
        Some(true)
    );
}

#[test]
fn set_rsi_config_persists_controls_without_erasing_other_config() {
    let dir = temp_dir_with("");
    let path = dir.join("config.toml");
    let mut config = Config {
        default_profile: Some("local".into()),
        profiles: std::collections::HashMap::from([(
            "local".into(),
            Profile {
                provider: Some(ProviderName::Ollama),
                model: Some("qwen".into()),
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    save_config_to(&config, &path).unwrap();

    set_rsi_config(
        &mut config,
        Some(true),
        Some(2_500_000),
        Some("beta".into()),
        Some(&path),
    )
    .unwrap();

    let saved = read_config_file(&path).unwrap();
    assert_eq!(saved.default_profile.as_deref(), Some("local"));
    assert_eq!(saved.profiles["local"].model.as_deref(), Some("qwen"));
    let rsi = saved.rsi.unwrap();
    assert_eq!(rsi.enabled, Some(true));
    assert_eq!(rsi.maximum_cost_microusd, Some(2_500_000));
    assert_eq!(rsi.channel.as_deref(), Some("beta"));

    set_rsi_config(&mut config, None, Some(4_000_000), None, Some(&path)).unwrap();
    let saved = read_config_file(&path).unwrap();
    let rsi = saved.rsi.unwrap();
    assert_eq!(rsi.enabled, Some(true));
    assert_eq!(rsi.maximum_cost_microusd, Some(4_000_000));

    set_rsi_config(&mut config, Some(false), None, None, Some(&path)).unwrap();
    let rsi = read_config_file(&path).unwrap().rsi.unwrap();
    assert_eq!(rsi.enabled, Some(false));
    assert_eq!(rsi.maximum_cost_microusd, Some(4_000_000));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn last_session_roundtrips_under_workspace_hi_dir() {
    use super::{LastSession, load_last_session, remember_session, save_last_session};
    let dir = std::env::temp_dir().join(format!(
        "hi-last-session-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let session = LastSession {
        profile: Some("work".into()),
        provider: Some("xai".into()),
        model: Some("grok-4.5".into()),
    };
    save_last_session(&dir, &session).unwrap();
    let loaded = load_last_session(&dir).expect("last session present");
    assert_eq!(loaded, session);

    // Convenience writer skips the unconfigured placeholder model.
    remember_session(&dir, Some("work"), "xai", "__model_not_configured__").unwrap();
    let still = load_last_session(&dir).unwrap();
    assert_eq!(still.model.as_deref(), Some("grok-4.5"));

    remember_session(&dir, None, "anthropic", "claude-sonnet-4").unwrap();
    let updated = load_last_session(&dir).unwrap();
    assert_eq!(updated.profile, None);
    assert_eq!(updated.provider.as_deref(), Some("anthropic"));
    assert_eq!(updated.model.as_deref(), Some("claude-sonnet-4"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn remember_session_skips_empty_model() {
    use super::{load_last_session, remember_session};
    let dir = std::env::temp_dir().join(format!(
        "hi-last-session-empty-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    remember_session(&dir, None, "xai", "").unwrap();
    assert!(load_last_session(&dir).is_none());
    std::fs::remove_dir_all(dir).unwrap();
}

fn config_with_default_profile() -> Config {
    Config {
        default_profile: Some("default".into()),
        profiles: [(
            "default".into(),
            Profile {
                provider: Some(ProviderName::Pipenetwork),
                model: Some("pipe/kimi-3".into()),
                api_key: Some("test-key".into()),
                ..Profile::default()
            },
        )]
        .into_iter()
        .collect(),
        ..Config::default()
    }
}

#[test]
fn active_profile_skips_default_when_last_session_was_provider_preset() {
    use super::{LastSession, save_last_session};
    let dir = std::env::temp_dir().join(format!(
        "hi-active-profile-preset-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    save_last_session(
        &dir,
        &LastSession {
            profile: None,
            provider: Some("xai".into()),
            model: Some("grok-4.5".into()),
        },
    )
    .unwrap();
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    let config = config_with_default_profile();
    // The bug: falling back to default_profile here made exit rewrite
    // last_session under "default" and lose the xai preset next launch.
    assert_eq!(resolve_active_profile(&cli, &config, &dir), None);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn active_profile_restores_named_last_session_profile() {
    use super::{LastSession, save_last_session};
    let dir = std::env::temp_dir().join(format!(
        "hi-active-profile-named-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut config = config_with_default_profile();
    config.profiles.insert(
        "work".into(),
        Profile {
            provider: Some(ProviderName::Xai),
            model: Some("grok-4.5".into()),
            api_key: Some("xai-key".into()),
            ..Profile::default()
        },
    );
    save_last_session(
        &dir,
        &LastSession {
            profile: Some("work".into()),
            provider: Some("xai".into()),
            model: Some("grok-4.5".into()),
        },
    )
    .unwrap();
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    assert_eq!(
        resolve_active_profile(&cli, &config, &dir).as_deref(),
        Some("work")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn active_profile_falls_back_to_default_without_last_session() {
    let dir = std::env::temp_dir().join(format!(
        "hi-active-profile-default-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    let config = config_with_default_profile();
    assert_eq!(
        resolve_active_profile(&cli, &config, &dir).as_deref(),
        Some("default")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_restores_provider_preset_from_last_session() {
    use super::{LastSession, save_last_session};
    let dir = std::env::temp_dir().join(format!(
        "hi-resolve-preset-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(dir.join(".hi")).unwrap();
    // resolve() reads last_session from cwd (`.`), so run under `dir`. The cwd
    // is process-wide: hold CWD_LOCK until it's restored so no concurrent test
    // observes the switch.
    let _cwd = crate::CWD_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let prev = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    save_last_session(
        Path::new("."),
        &LastSession {
            profile: None,
            provider: Some("xai".into()),
            model: Some("grok-4.5".into()),
        },
    )
    .unwrap();
    // Auth store / env may or may not have an xAI key; force one via CLI so
    // the assertion focuses on provider/model restore.
    let cli = Cli::try_parse_from(["hi", "--api-key", "xai-test-key"]).unwrap();
    let config = config_with_default_profile();
    let settings = resolve(&cli, &config).expect("resolve preset last session");
    assert_eq!(settings.provider, ProviderName::Xai);
    assert_eq!(settings.model, "grok-4.5");
    std::env::set_current_dir(prev).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}

// --- Setup-wizard trigger ---------------------------------------------------
//
// `needs_setup` reads process-wide env vars, so these tests must not run
// concurrently with each other (or with anything else touching the same vars),
// and must not inherit whatever the developer happens to have exported —
// `ANTHROPIC_API_KEY` alone would otherwise flip every expectation here.

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Every env var that suppresses the wizard. Cleared for the duration of a
/// test and restored on drop.
const SETUP_ENV_VARS: [&str; 16] = [
    "HI_MODEL",
    "HI_BASE_URL",
    "HI_MCP_URL",
    "HI_API_KEY",
    "HI_FORCE_API_KEY",
    "PIPENETWORK_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "HI_RSI_ENABLED",
    "HI_X402_KEYPAIR",
    "HI_X402_MAX_USD",
    "HI_X402_AUTO_CONFIRM",
];

struct ClearedSetupEnv {
    _guard: std::sync::MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<String>)>,
}

impl ClearedSetupEnv {
    fn new() -> Self {
        // A poisoned lock just means some other test panicked mid-mutation;
        // the restore below still puts the environment back.
        let guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let saved = SETUP_ENV_VARS
            .iter()
            .map(|name| (*name, std::env::var(name).ok()))
            .collect();
        for name in SETUP_ENV_VARS {
            // SAFETY: ENV_LOCK is held for the lifetime of `self`, so no other
            // test in this binary reads or writes these vars concurrently.
            unsafe { std::env::remove_var(name) };
        }
        Self {
            _guard: guard,
            saved,
        }
    }

    /// Set one of [`SETUP_ENV_VARS`] — anything else would not be restored.
    fn set(&self, name: &str, value: &str) {
        assert!(
            SETUP_ENV_VARS.contains(&name),
            "{name} is not restored on drop"
        );
        // SAFETY: as in `new` — the lock is held for the lifetime of `self`.
        unsafe { std::env::set_var(name, value) };
    }

    fn remove(&self, name: &str) {
        assert!(
            SETUP_ENV_VARS.contains(&name),
            "{name} is not restored on drop"
        );
        // SAFETY: as in `new` — the lock is held for the lifetime of `self`.
        unsafe { std::env::remove_var(name) };
    }
}

impl Drop for ClearedSetupEnv {
    fn drop(&mut self) {
        for (name, value) in &self.saved {
            // SAFETY: as in `new` — the lock is still held.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

fn ollama_profile() -> Profile {
    Profile {
        provider: Some(ProviderName::Ollama),
        model: Some("qwen2.5-coder".into()),
        ..Default::default()
    }
}

#[test]
fn needs_setup_on_a_bare_run_with_nothing_configured() {
    let _env = ClearedSetupEnv::new();
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    assert!(needs_setup(&cli, &Config::default()));
}

/// The regression this guards: a project-local `hi.toml` defines profiles but
/// deliberately leaves `default_profile` to the global config. Nothing
/// resolves, so the wizard must run — it used to be skipped here, dead-ending
/// in an onboarding message that tells the user to run the wizard.
#[test]
fn needs_setup_when_profiles_exist_but_none_is_selectable() {
    let _env = ClearedSetupEnv::new();
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    let mut file = Config::default();
    file.profiles.insert("local".into(), ollama_profile());
    assert!(file.default_profile.is_none());
    assert!(needs_setup(&cli, &file));
}

#[test]
fn no_setup_when_a_default_profile_is_configured() {
    let _env = ClearedSetupEnv::new();
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    let mut file = Config::default();
    file.profiles.insert("local".into(), ollama_profile());
    file.default_profile = Some("local".into());
    assert!(!needs_setup(&cli, &file));
}

/// An explicit selection on the command line is a complete answer to the
/// question the wizard asks.
#[test]
fn no_setup_when_the_cli_selects_a_model_provider_or_profile() {
    let _env = ClearedSetupEnv::new();
    for args in [
        vec!["hi", "-m", "qwen2.5-coder"],
        vec!["hi", "--provider", "ollama"],
        vec!["hi", "-p", "local"],
    ] {
        let cli = Cli::try_parse_from(&args).unwrap();
        assert!(
            !needs_setup(&cli, &Config::default()),
            "args={args:?} should not trigger setup"
        );
    }
}

#[test]
fn no_setup_for_advertised_cli_model_with_key_when_profile_has_no_model() {
    let env = ClearedSetupEnv::new();
    env.set("OPENROUTER_API_KEY", "test-key");
    let cli = Cli::try_parse_from(["hi", "-m", "advertised-id"]).unwrap();
    let mut file = Config::default();
    file.profiles.insert(
        "cloud".into(),
        Profile {
            provider: Some(ProviderName::Openai),
            model: None,
            ..Default::default()
        },
    );
    file.default_profile = Some("cloud".into());
    assert!(
        !needs_setup(&cli, &file),
        "hi -m advertised-id with a key must not bounce into the wizard"
    );
    let settings = resolve(&cli, &file).expect("resolve advertised id");
    assert_eq!(settings.model, "advertised-id");
}

#[test]
fn no_setup_when_hi_model_is_set() {
    let env = ClearedSetupEnv::new();
    env.set("HI_MODEL", "qwen2.5-coder");
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    assert!(!needs_setup(&cli, &Config::default()));
}

/// `auto_select` infers a provider from an exported key, so there is a model to
/// run and nothing to ask about. It is reported instead — see below.
#[test]
fn no_setup_when_an_api_key_in_the_env_auto_selects_a_provider() {
    for name in [
        "PIPENETWORK_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
    ] {
        let env = ClearedSetupEnv::new();
        env.set(name, "test-key");
        let cli = Cli::try_parse_from(["hi"]).unwrap();
        assert!(
            !needs_setup(&cli, &Config::default()),
            "{name} should suppress setup"
        );
        assert_eq!(
            auto_selected_env(&cli, &Config::default()),
            Some(name),
            "{name} should be named in the startup notice"
        );
    }
}

#[test]
fn pipenetwork_env_key_auto_selects_flash_vision_exp() {
    let env = ClearedSetupEnv::new();
    env.set("PIPENETWORK_API_KEY", "test-key");
    let selected = auto_select(&Config::default()).expect("pipenetwork key should auto-select");
    drop(env);
    assert_eq!(selected.0, ProviderName::Pipenetwork);
    assert_eq!(selected.1, "pipe/deepseek-v4-flash-vision-exp");
}

/// The notice is only for the case where the env var is doing *all* the work.
/// Anything that selects a model itself takes precedence in `resolve`, so
/// naming the variable would be a lie about where the model came from.
#[test]
fn no_auto_select_notice_when_something_else_selects_the_model() {
    let env = ClearedSetupEnv::new();
    env.set("ANTHROPIC_API_KEY", "test-key");

    let with_model = Cli::try_parse_from(["hi", "-m", "qwen2.5-coder"]).unwrap();
    assert_eq!(auto_selected_env(&with_model, &Config::default()), None);

    let bare = Cli::try_parse_from(["hi"]).unwrap();
    let mut file = Config::default();
    file.profiles.insert("local".into(), ollama_profile());
    file.default_profile = Some("local".into());
    assert_eq!(auto_selected_env(&bare, &file), None);
}

/// No key exported means nothing to report — that run gets the wizard instead.
#[test]
fn no_auto_select_notice_without_a_key_in_the_env() {
    let _env = ClearedSetupEnv::new();
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    assert_eq!(auto_selected_env(&cli, &Config::default()), None);
    assert!(needs_setup(&cli, &Config::default()));
}

#[test]
fn x402_keypair_env_skips_setup_and_selects_pipenetwork() {
    let _env = ClearedSetupEnv::new();
    _env.set("HI_X402_KEYPAIR", "/tmp/id.json");
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    assert!(!needs_setup(&cli, &Config::default()));
    assert_eq!(
        auto_selected_env(&cli, &Config::default()),
        Some("HI_X402_KEYPAIR")
    );
}

#[test]
fn x402_config_section_skips_setup() {
    let _env = ClearedSetupEnv::new();
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    let file = Config {
        x402: Some(super::X402Section {
            enabled: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(x402_configured(&file));
    assert!(!needs_setup(&cli, &file));
}

#[test]
fn x402_settings_prefer_env_over_config() {
    let _env = ClearedSetupEnv::new();
    _env.set("HI_X402_KEYPAIR", "/tmp/from-env.json");
    _env.set("HI_X402_MAX_USD", "0.5");
    _env.set("HI_X402_AUTO_CONFIRM", "1");
    let file = Config {
        x402: Some(super::X402Section {
            keypair: Some("/tmp/from-config.json".into()),
            max_usd: Some(2.0),
            auto_confirm: Some(false),
            ..Default::default()
        }),
        ..Default::default()
    };
    let resolved = resolve_x402_settings(false, &file);
    assert_eq!(
        resolved.keypair.as_deref(),
        Some(std::path::Path::new("/tmp/from-env.json"))
    );
    assert_eq!(resolved.max_usd, 0.5);
    assert!(resolved.auto_confirm);
}

#[test]
fn x402_config_auto_selects_pipenetwork_with_empty_bearer() {
    let _env = ClearedSetupEnv::new();
    let file = Config {
        x402: Some(super::X402Section {
            enabled: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    };
    let selected = auto_select(&file).expect("x402 config should auto-select pipenetwork");
    assert_eq!(selected.0, ProviderName::Pipenetwork);
    let resolved = resolve_x402_settings(false, &file);
    assert!(resolved.keypair.is_none());
    assert!(
        resolved.paste_sig,
        "anonymous hop is the paste-sig / 402 path"
    );
}

#[test]
fn pairing_env_key_wins_over_x402_keypair() {
    let env = ClearedSetupEnv::new();
    env.set("PIPENETWORK_API_KEY", "pk_live_pair");
    env.set("HI_X402_KEYPAIR", "/tmp/id.json");
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    assert_eq!(
        auto_selected_env(&cli, &Config::default()),
        Some("PIPENETWORK_API_KEY")
    );
    assert!(crate::x402::credential_is_pairing_key("pk_live_pair"));
    assert!(!crate::x402::credential_is_pairing_key(""));
    assert!(!crate::x402::credential_is_pairing_key("x402_credits"));
}

#[test]
fn yes_flag_enables_x402_auto_confirm() {
    let _env = ClearedSetupEnv::new();
    let cli = Cli::try_parse_from(["hi", "--yes"]).unwrap();
    assert!(cli.yes);
    let resolved = resolve_x402_settings(cli.yes, &Config::default());
    assert!(resolved.auto_confirm);
}

#[test]
fn x402_section_round_trips_in_config_toml() {
    let config = Config {
        x402: Some(super::X402Section {
            enabled: Some(true),
            keypair: Some("/tmp/id.json".into()),
            max_usd: Some(1.0),
            ..Default::default()
        }),
        ..Default::default()
    };
    let encoded = toml::to_string(&config).unwrap();
    assert!(encoded.contains("[x402]"), "{encoded}");
    assert!(encoded.contains("keypair"), "{encoded}");
    let decoded: Config = toml::from_str(&encoded).unwrap();
    assert_eq!(decoded.x402, config.x402);
}

#[test]
fn persist_profile_reasoning_effort_round_trips_and_clears() {
    let dir = temp_dir_with("");
    let path = dir.join("hi.toml");
    let mut config = Config::default();
    config.profiles.insert(
        "work".into(),
        Profile {
            model: Some("gpt-5".into()),
            ..Default::default()
        },
    );

    super::persist_profile_reasoning_effort(
        &mut config,
        "work",
        Some(hi_ai::ReasoningEffort::Xhigh),
        Some(&path),
    )
    .unwrap();
    assert_eq!(
        config.profiles["work"].reasoning_effort,
        Some(hi_ai::ReasoningEffort::Xhigh)
    );
    let on_disk = read_config_file(&path).unwrap();
    assert_eq!(
        on_disk.profiles["work"].reasoning_effort,
        Some(hi_ai::ReasoningEffort::Xhigh),
        "a fresh explicit file receives the full profile with the effort"
    );

    // `None` clears the field so the endpoint default applies on next launch.
    super::persist_profile_reasoning_effort(&mut config, "work", None, Some(&path)).unwrap();
    assert_eq!(config.profiles["work"].reasoning_effort, None);
    let on_disk = read_config_file(&path).unwrap();
    assert_eq!(on_disk.profiles["work"].reasoning_effort, None);

    // Unknown profiles are an error, not a silent write.
    assert!(
        super::persist_profile_reasoning_effort(&mut config, "nope", None, Some(&path)).is_err()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn persist_reasoning_effort_writes_machine_default_and_profile() {
    let dir = temp_dir_with("");
    let path = dir.join("hi.toml");
    let mut config = Config::default();
    config.profiles.insert(
        "work".into(),
        Profile {
            model: Some("gpt-5".into()),
            ..Default::default()
        },
    );

    let wrote_profile = super::persist_reasoning_effort(
        &mut config,
        Some("work"),
        Some(hi_ai::ReasoningEffort::High),
        Some(&path),
    )
    .unwrap();
    assert!(wrote_profile);
    assert_eq!(
        config.reasoning_effort,
        Some(hi_ai::ReasoningEffort::High),
        "machine-wide default updated in memory"
    );
    assert_eq!(
        config.profiles["work"].reasoning_effort,
        Some(hi_ai::ReasoningEffort::High)
    );
    let on_disk = read_config_file(&path).unwrap();
    assert_eq!(
        on_disk.reasoning_effort,
        Some(hi_ai::ReasoningEffort::High),
        "machine-wide default lands in the config file"
    );
    assert_eq!(
        on_disk.profiles["work"].reasoning_effort,
        Some(hi_ai::ReasoningEffort::High)
    );

    // No / unknown profile still sticks machine-wide.
    let wrote_profile = super::persist_reasoning_effort(
        &mut config,
        None,
        Some(hi_ai::ReasoningEffort::Low),
        Some(&path),
    )
    .unwrap();
    assert!(!wrote_profile);
    assert_eq!(config.reasoning_effort, Some(hi_ai::ReasoningEffort::Low));
    let on_disk = read_config_file(&path).unwrap();
    assert_eq!(on_disk.reasoning_effort, Some(hi_ai::ReasoningEffort::Low));
    assert_eq!(
        on_disk.profiles["work"].reasoning_effort,
        Some(hi_ai::ReasoningEffort::High),
        "profile field left alone when no active profile"
    );

    // Clear machine-wide (off).
    super::persist_reasoning_effort(&mut config, None, None, Some(&path)).unwrap();
    assert_eq!(config.reasoning_effort, None);
    let on_disk = read_config_file(&path).unwrap();
    assert_eq!(on_disk.reasoning_effort, None);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merge_config_keeps_global_reasoning_when_local_omits_it() {
    use super::merge_config;
    let mut global = Config {
        reasoning_effort: Some(hi_ai::ReasoningEffort::Medium),
        ..Default::default()
    };
    let local = Config {
        profiles: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "local".into(),
                Profile {
                    model: Some("qwen".into()),
                    ..Default::default()
                },
            );
            m
        },
        ..Default::default()
    };
    merge_config(&mut global, local);
    assert_eq!(
        global.reasoning_effort,
        Some(hi_ai::ReasoningEffort::Medium)
    );
}

#[test]
fn resolve_falls_back_to_machine_reasoning_effort() {
    let mut config = Config {
        default_profile: Some("work".into()),
        reasoning_effort: Some(hi_ai::ReasoningEffort::High),
        ..Default::default()
    };
    config.profiles.insert(
        "work".into(),
        Profile {
            provider: Some(ProviderName::Openai),
            model: Some("gpt-5".into()),
            api_key: Some("sk-test".into()),
            ..Default::default()
        },
    );
    // Pin the profile so workspace `.hi/last_session.toml` cannot redirect routing.
    let cli = Cli::try_parse_from(["hi", "--profile", "work"]).unwrap();
    let settings = super::resolve(&cli, &config).unwrap();
    assert_eq!(
        settings.reasoning_effort,
        Some(hi_ai::ReasoningEffort::High)
    );

    // Profile override wins over machine default.
    config.profiles.get_mut("work").unwrap().reasoning_effort =
        Some(hi_ai::ReasoningEffort::Minimal);
    let settings = super::resolve(&cli, &config).unwrap();
    assert_eq!(
        settings.reasoning_effort,
        Some(hi_ai::ReasoningEffort::Minimal)
    );
}

#[test]
fn xai_profile_xhigh_does_not_follow_a_provider_override() {
    let mut config = Config {
        default_profile: Some("default".into()),
        ..Default::default()
    };
    config.profiles.insert(
        "default".into(),
        Profile {
            provider: Some(ProviderName::Xai),
            model: Some("grok-4.3".into()),
            reasoning_effort: Some(hi_ai::ReasoningEffort::Xhigh),
            api_key: Some("xai-key".into()),
            ..Default::default()
        },
    );
    let env = ClearedSetupEnv::new();
    env.set("PIPENETWORK_API_KEY", "pipe-key");
    let cli = Cli::try_parse_from([
        "hi",
        "--provider",
        "pipenetwork",
        "--model",
        "pipe/deepseek-v4-flash",
    ])
    .unwrap();
    let settings = resolve(&cli, &config).unwrap();
    drop(env);
    assert_eq!(settings.provider, ProviderName::Pipenetwork);
    assert_eq!(
        settings.reasoning_effort, None,
        "xAI xhigh must not ride --provider pipenetwork onto DeepSeek"
    );
}

#[test]
fn provider_override_does_not_inherit_mismatched_profile_model() {
    let mut config = Config {
        default_profile: Some("default".into()),
        ..Default::default()
    };
    config.profiles.insert(
        "default".into(),
        Profile {
            provider: Some(ProviderName::Xai),
            model: Some("grok-4.3".into()),
            base_url: Some("https://xai-profile.invalid/v1".into()),
            mcp_url: Some("https://xai-profile.invalid/mcp".into()),
            api_key: Some("xai-key".into()),
            max_tokens: Some(1234),
            top_p: Some(0.25),
            output_token_parameter: Some(hi_ai::OutputTokenParameter::MaxCompletionTokens),
            thinking_budget: Some(4096),
            reasoning_effort: Some(hi_ai::ReasoningEffort::Xhigh),
            tool_mode: Some(super::ToolMode::ReadOnly),
            compat: Some(hi_ai::CompatMode::Strict),
            deepseek_compat: Some(hi_ai::DeepSeekCompat::On),
            curate_skills: Some(false),
            explore_subagents: Some(false),
            suggest_next_prompt: Some(false),
            write_subagents: Some(false),
            planner_model: Some("grok-planner".into()),
            skeptic_model: Some("grok-skeptic".into()),
            fallback: Some(vec!["xai-backup".into()]),
            runtime: Some(LocalRuntimeProfile {
                kind: "mlx".into(),
                repo: "xai/local-model".into(),
                ..Default::default()
            }),
            execution: Some(hi_agent::ExecutionMode::Durable),
            ..Default::default()
        },
    );
    config.profiles.insert(
        "xai-backup".into(),
        Profile {
            provider: Some(ProviderName::Xai),
            model: Some("grok-backup".into()),
            api_key: Some("xai-backup-key".into()),
            ..Default::default()
        },
    );
    let env = ClearedSetupEnv::new();
    env.set("PIPENETWORK_API_KEY", "pipe-key");
    // HI_BASE_URL + HI_API_KEY are an explicitly paired generic route. The
    // provider-specific key alone is intentionally confined to Pipe's origin.
    env.set("HI_API_KEY", "pipe-key");
    env.set("HI_MODEL", "pipe/deepseek-v4-flash");
    env.set("HI_BASE_URL", "https://pipe-env.invalid/v1");
    env.set("HI_MCP_URL", "https://api.pipenetwork.ai/mcp");
    let cli = Cli::try_parse_from(["hi", "--provider", "pipenetwork"]).unwrap();
    let settings = resolve(&cli, &config).unwrap();
    drop(env);
    assert_eq!(settings.provider, ProviderName::Pipenetwork);
    assert_eq!(
        settings.model, "pipe/deepseek-v4-flash",
        "HI_MODEL must win when --provider does not match the default profile"
    );
    assert_eq!(settings.base_url, "https://pipe-env.invalid/v1");
    assert_eq!(
        settings.mcp_url.as_deref(),
        Some("https://api.pipenetwork.ai/mcp")
    );
    assert_eq!(settings.api_key, "pipe-key");
    assert_eq!(settings.max_tokens, PIPENETWORK_DEFAULT_MAX_TOKENS);
    assert!(!settings.max_tokens_explicit);
    assert_eq!(settings.top_p, None);
    assert_eq!(
        settings.output_token_parameter,
        hi_ai::OutputTokenParameter::Auto
    );
    assert_eq!(settings.thinking_budget, None);
    assert_eq!(settings.reasoning_effort, None);
    assert_eq!(settings.compat, hi_ai::CompatMode::Auto);
    assert_eq!(settings.deepseek_compat, hi_ai::DeepSeekCompat::Auto);
    assert_eq!(settings.planner_model.as_deref(), Some("pipe/glm-5.2-fast"));
    assert_eq!(settings.skeptic_model, None);
    assert_eq!(settings.runtime, None);
    assert!(
        resolve_fallbacks(&cli, &config).is_empty(),
        "a mismatched profile's fallback chain must not follow the override"
    );

    // These settings control agent behavior/persistence rather than the remote
    // request route, so the selected profile still supplies them.
    assert_eq!(settings.tool_mode, super::ToolMode::ReadOnly);
    assert!(!settings.curate_skills);
    assert!(!settings.explore_subagents);
    assert!(!settings.suggest_next_prompt);
    assert_eq!(settings.write_subagents, hi_agent::WriteSubagentPolicy::Off);
    assert_eq!(settings.execution, hi_agent::ExecutionMode::Durable);

    // An explicit CLI fallback remains explicit even when the default profile
    // targets another provider.
    let cli = Cli::try_parse_from([
        "hi",
        "--provider",
        "pipenetwork",
        "--fallback",
        "xai-backup",
    ])
    .unwrap();
    let fallbacks = resolve_fallbacks(&cli, &config);
    assert_eq!(fallbacks.len(), 1);
    assert_eq!(fallbacks[0].model, "grok-backup");
}

#[test]
fn provider_override_falls_back_to_provider_default_model() {
    let mut config = Config {
        default_profile: Some("default".into()),
        ..Default::default()
    };
    config.profiles.insert(
        "default".into(),
        Profile {
            provider: Some(ProviderName::Xai),
            model: Some("grok-4.3".into()),
            api_key: Some("xai-key".into()),
            ..Default::default()
        },
    );
    let env = ClearedSetupEnv::new();
    env.set("PIPENETWORK_API_KEY", "pipe-key");
    let cli = Cli::try_parse_from(["hi", "--provider", "pipenetwork"]).unwrap();
    let settings = resolve(&cli, &config).unwrap();
    drop(env);
    assert_eq!(settings.model, "pipe/deepseek-v4-flash-vision-exp");
}

#[test]
fn last_session_stale_pipenetwork_default_remaps_to_flash_vision_exp() {
    use super::{LastSession, save_last_session};
    let dir = std::env::temp_dir().join(format!(
        "hi-stale-session-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(dir.join(".hi")).unwrap();
    let _cwd = crate::CWD_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let prev = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    save_last_session(
        Path::new("."),
        &LastSession {
            profile: None,
            provider: Some("pipenetwork".into()),
            model: Some("ipop/coder-balanced".into()),
        },
    )
    .unwrap();
    let env = ClearedSetupEnv::new();
    env.set("PIPENETWORK_API_KEY", "pipe-key");
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    let settings = resolve(&cli, &Config::default()).unwrap();
    std::env::set_current_dir(prev).unwrap();
    drop(env);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(settings.provider, ProviderName::Pipenetwork);
    assert_eq!(settings.model, "pipe/deepseek-v4-flash-vision-exp");
}

#[test]
fn last_session_explicit_glm_is_unchanged() {
    use super::{LastSession, save_last_session};
    let dir = std::env::temp_dir().join(format!(
        "hi-glm-session-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(dir.join(".hi")).unwrap();
    let _cwd = crate::CWD_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let prev = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    save_last_session(
        Path::new("."),
        &LastSession {
            profile: None,
            provider: Some("pipenetwork".into()),
            model: Some("pipe/glm-5.2".into()),
        },
    )
    .unwrap();
    let env = ClearedSetupEnv::new();
    env.set("PIPENETWORK_API_KEY", "pipe-key");
    let cli = Cli::try_parse_from(["hi"]).unwrap();
    let settings = resolve(&cli, &Config::default()).unwrap();
    std::env::set_current_dir(prev).unwrap();
    drop(env);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(settings.model, "pipe/glm-5.2");
}

#[test]
fn cli_model_explicit_glm_is_unchanged() {
    let env = ClearedSetupEnv::new();
    env.set("PIPENETWORK_API_KEY", "pipe-key");
    let cli = Cli::try_parse_from(["hi", "--provider", "pipenetwork", "--model", "pipe/glm-5.2"])
        .unwrap();
    let settings = resolve(&cli, &Config::default()).unwrap();
    drop(env);
    assert_eq!(settings.model, "pipe/glm-5.2");
}

#[test]
fn cli_reasoning_effort_still_applies_on_a_provider_override() {
    let mut config = Config {
        default_profile: Some("default".into()),
        ..Default::default()
    };
    config.profiles.insert(
        "default".into(),
        Profile {
            provider: Some(ProviderName::Xai),
            reasoning_effort: Some(hi_ai::ReasoningEffort::Xhigh),
            api_key: Some("xai-key".into()),
            ..Default::default()
        },
    );
    let env = ClearedSetupEnv::new();
    env.set("PIPENETWORK_API_KEY", "pipe-key");
    let cli = Cli::try_parse_from([
        "hi",
        "--provider",
        "pipenetwork",
        "--model",
        "pipe/deepseek-v4-flash",
        "--reasoning-effort",
        "high",
    ])
    .unwrap();
    let settings = resolve(&cli, &config).unwrap();
    drop(env);
    assert_eq!(
        settings.reasoning_effort,
        Some(hi_ai::ReasoningEffort::High)
    );
}

#[test]
fn machine_wide_xhigh_is_xai_only() {
    assert_eq!(
        resolve_reasoning_effort(
            None,
            None,
            ProviderName::Pipenetwork,
            Some(hi_ai::ReasoningEffort::Xhigh)
        ),
        None
    );
    assert_eq!(
        resolve_reasoning_effort(
            None,
            None,
            ProviderName::Xai,
            Some(hi_ai::ReasoningEffort::Xhigh)
        ),
        Some(hi_ai::ReasoningEffort::Xhigh)
    );
    assert_eq!(
        resolve_reasoning_effort(
            None,
            None,
            ProviderName::Pipenetwork,
            Some(hi_ai::ReasoningEffort::High)
        ),
        Some(hi_ai::ReasoningEffort::High)
    );
}

#[test]
fn pipenetwork_profile_keeps_its_own_reasoning_effort() {
    let profile = Profile {
        provider: Some(ProviderName::Pipenetwork),
        reasoning_effort: Some(hi_ai::ReasoningEffort::High),
        ..Default::default()
    };
    assert_eq!(
        resolve_reasoning_effort(None, Some(&profile), ProviderName::Pipenetwork, None),
        Some(hi_ai::ReasoningEffort::High)
    );
}

#[test]
fn top_level_help_lists_everyday_commands() {
    let err = Cli::try_parse_from(["hi", "--help"]).expect_err("clap prints help as an error");
    let help = err.to_string();
    for needle in [
        "setup",
        "doctor",
        "update",
        "workflow",
        "trace",
        "--best-of",
        "--judge",
        "headless form of `/race`",
    ] {
        assert!(
            help.contains(needle),
            "hi --help should mention {needle}:\n{help}"
        );
    }
}
