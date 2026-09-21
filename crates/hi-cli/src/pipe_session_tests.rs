use super::*;
use crate::config::resolve;
use clap::Parser;

fn openai_profile(api_key: &str) -> Profile {
    Profile {
        provider: Some(ProviderName::Openai),
        api_key: Some(api_key.into()),
        model: Some("gpt-4o".into()),
        ..Default::default()
    }
}

fn openai_settings_with(config: &Config) -> Settings {
    let cli = Cli::try_parse_from(["hi", "--profile", "openai"]).unwrap();
    resolve(&cli, config).expect("openai profile should resolve")
}

#[test]
fn pipe_route_uses_pipenetwork_profile_not_openai_default() {
    let mut config = Config {
        default_profile: Some("openai".into()),
        ..Default::default()
    };
    config
        .profiles
        .insert("openai".into(), openai_profile("sk-openai"));
    config.profiles.insert(
        "pipenetwork".into(),
        Profile {
            provider: Some(ProviderName::Pipenetwork),
            api_key: Some("pk_pipe".into()),
            model: Some("pipe/custom".into()),
            ..Default::default()
        },
    );
    let cli = Cli::try_parse_from(["hi", "--profile", "openai"]).unwrap();
    let settings = openai_settings_with(&config);
    assert_eq!(settings.api_key, "sk-openai");
    let route = resolve_pipe_route(&cli, &config, &settings).unwrap();
    assert_eq!(route.api_key, "pk_pipe");
    assert_eq!(route.model, "pipe/custom");
    assert_eq!(route.base_url, hi_harness::DEFAULT_BASE_URL);
}

#[test]
fn pipe_route_cli_api_key_wins() {
    let mut config = Config {
        default_profile: Some("openai".into()),
        ..Default::default()
    };
    config
        .profiles
        .insert("openai".into(), openai_profile("sk-openai"));
    config.profiles.insert(
        "pipenetwork".into(),
        Profile {
            provider: Some(ProviderName::Pipenetwork),
            api_key: Some("pk_pipe".into()),
            ..Default::default()
        },
    );
    let cli = Cli::try_parse_from(["hi", "--profile", "openai", "--api-key", "cli-key"]).unwrap();
    let settings = openai_settings_with(&config);
    let route = resolve_pipe_route(&cli, &config, &settings).unwrap();
    assert_eq!(route.api_key, "cli-key");
}

#[test]
fn session_path_prefers_explicit_file() {
    let cli = Cli::try_parse_from(["hi", "--session-file", "/tmp/explicit.jsonl"]).unwrap();
    let path = resolve_session_path(&cli).unwrap();
    assert_eq!(
        path.as_deref(),
        Some(std::path::Path::new("/tmp/explicit.jsonl"))
    );
}

#[test]
fn session_path_no_save_skips_persistence() {
    let cli = Cli::try_parse_from(["hi", "--no-save"]).unwrap();
    assert_eq!(resolve_session_path(&cli).unwrap(), None);
}

#[test]
fn session_path_resume_id_is_used() {
    let cli = Cli::try_parse_from(["hi", "--resume", "abc-123"]).unwrap();
    let path = resolve_session_path(&cli).unwrap().expect("path");
    assert!(
        path.ends_with("abc-123.jsonl"),
        "unexpected resume path {}",
        path.display()
    );
}

#[test]
fn pipe_route_login_profile_ref_does_not_leak_openai_key() {
    let mut config = Config {
        default_profile: Some("openai".into()),
        ..Default::default()
    };
    config
        .profiles
        .insert("openai".into(), openai_profile("sk-openai"));
    config.profiles.insert(
        "pipenetwork".into(),
        Profile {
            provider: Some(ProviderName::Pipenetwork),
            api_key_ref: Some("auth-store://pipenetwork".into()),
            ..Default::default()
        },
    );
    let cli = Cli::try_parse_from(["hi", "--profile", "openai"]).unwrap();
    let settings = openai_settings_with(&config);
    match resolve_pipe_route(&cli, &config, &settings) {
        Ok(route) => {
            assert_ne!(
                route.api_key, "sk-openai",
                "openai default profile key must not be sent to Pipe"
            );
        }
        Err(err) => {
            let text = format!("{err:#}");
            assert!(
                !text.contains("sk-openai"),
                "openai key must not be sent to Pipe: {text}"
            );
            assert!(
                text.contains("hi login pipenetwork"),
                "missing pairing key should mention login: {text}"
            );
        }
    }
}

#[test]
fn turn_report_records_silent_inspect_stop_fields() {
    let ui = StdoutUi {
        assistant: String::new(),
        turn_end: "stopped repeating the same inspect".into(),
        statuses: vec!["shrunk tool results".into()],
        tool_calls: vec![serde_json::json!({
            "name": "read",
            "arguments": "{\"path\":\"src/server.rs\"}",
            "output": "read src/server.rs · 12000 chars · omitted",
        })],
        ..StdoutUi::default()
    };
    let outcome = TurnOutcome {
        stop_reason: TurnStopReason::Completed,
        usage: hi_ai::Usage {
            input_tokens: 12,
            output_tokens: 0,
            ..Default::default()
        },
        changed_files: Vec::new(),
        error: None,
        verification: None,
    };
    let body = turn_report_json(&outcome, &ui, 4, &[], None);
    assert!(
        body.get("review").is_none(),
        "no review key without --spec-review"
    );
    assert_eq!(body["assistant_response"], "");
    assert_eq!(body["turn_end"], "stopped repeating the same inspect");
    assert_eq!(
        body["outcome"]["turn_end"],
        "stopped repeating the same inspect"
    );
    assert_eq!(body["usage"]["input_tokens"], 12);
    let output = body["tools"][0]["output"].as_str().unwrap();
    assert!(
        output.contains("omitted"),
        "report must keep stubbed read output: {output}"
    );
}
