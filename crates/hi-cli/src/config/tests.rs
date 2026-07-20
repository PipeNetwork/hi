
    use super::{
        Cli, Config, DEFAULT_MAX_TOKENS, LEGACY_PIPENETWORK_DEFAULT_MAX_TOKENS,
        PIPENETWORK_DEFAULT_MAX_TOKENS, Profile, ProviderName, RsiRequested, RsiSection,
        configured_max_tokens, curate_skills_default, detect_verify_pipeline,
        explore_subagents_default, max_tokens_is_explicit, permits_missing_checkpoint,
        planner_model_default, read_config_file, resolve_named_profile, resolve_quality,
        resolve_rsi, save_config_to, set_rsi_config, write_subagents_default,
    };
    use clap::Parser;
    use hi_agent::{LspMode, ReviewPolicy, ToolSet, VerificationMode};
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

    #[test]
    fn detects_layered_pipeline_by_marker() {
        // (marker, expected stage commands in order)
        let cases: [(&str, Vec<&str>); 6] = [
            (
                "Cargo.toml",
                vec!["cargo check --quiet", "cargo test --quiet"],
            ),
            ("go.mod", vec!["go build ./...", "go test ./..."]),
            ("pyproject.toml", vec!["pytest -q"]),
            ("package.json", vec!["npm test --silent"]),
            ("Makefile", vec!["make test"]),
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

    /// `/provider xai` should work with nothing configured — otherwise a user
    /// who just ran `/login xai` has to hand-write a profile to use it.
    #[test]
    fn a_bare_provider_name_resolves_without_a_profile() {
        let config = Config::default();
        unsafe { std::env::set_var("XAI_API_KEY", "test-key") };
        let settings = resolve_named_profile(&config, "xai").unwrap();
        unsafe { std::env::remove_var("XAI_API_KEY") };
        assert_eq!(settings.provider, ProviderName::Xai);
        assert_eq!(settings.base_url, "https://api.x.ai/v1");
        assert_eq!(settings.model, "grok-4.3");
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
        use super::ProfileForm;
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
    fn migrate_repairs_env_var_name_misplaced_in_api_key() {
        // The previous version of the migration moved an env var name like
        // "HI_API_KEY" from api_key_env to api_key when the env var wasn't set,
        // causing 401s. If the env var IS set, the migration should replace
        // api_key with the env var's value.
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
            Some("api_realkey_value"),
            "should be replaced with env var value"
        );
        assert!(p.api_key_env.is_none());
        unsafe { std::env::remove_var(env_name) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_moves_unset_env_var_name_from_api_key_back_to_env_ref() {
        // If api_key holds an env var name and the env var is NOT set, move it
        // back to api_key_env so the user gets the right error ("env var … is
        // not set") instead of a 401 from authenticating with the var name.
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
            p.api_key_env.as_deref(),
            Some(env_name),
            "should move back to env ref"
        );
        assert!(p.api_key.is_none(), "api_key should be cleared");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migrate_drops_standard_env_name_from_buggy_save_config() {
        // The old setup wizard always wrote api_key_env = key_envs().first()
        // (e.g. "HI_API_KEY" for pipenetwork) regardless of what the user pasted.
        // When that env var isn't set, the migration should drop the bogus
        // reference so resolve falls through to the onboarding error, prompting
        // the user to re-enter their key (the new wizard stores it as api_key).
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
        assert!(
            p.api_key_env.is_none(),
            "bogus standard env ref must be dropped"
        );
        assert!(p.api_key.is_none(), "no literal key to recover");
        // File should have been rewritten without api_key_env.
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            !text.contains("api_key_env"),
            "file should not have api_key_env: {text}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rsi_cli_overrides_config_and_managed_is_mandatory() {
        let enabled = Config {
            rsi: Some(RsiSection {
                enabled: Some(true),
                base_url: None,
                maximum_cost_microusd: None,
                channel: None,
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
