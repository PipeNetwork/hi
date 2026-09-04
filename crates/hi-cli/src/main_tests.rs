use super::{
    automatic_workflow_plan_path, canonical_session_identity, completed_session_switch,
    pipefs_startup_authority_required, sync_session_enabled, top_level_error_code,
    validate_tui_event_trace_request,
};
use crate::config::{Cli, ProviderName, Settings};
use crate::landing::write_landing;
use crate::project_context::{auto_memory_enabled, memory_context};
use crate::provider::{
    default_skeptic_model, effective_max_tokens_for_model, startup_live_model_metadata,
};
use crate::report::{
    one_shot_exit_code, report_tool_records, report_verification_stages,
    write_initialization_failure_report,
};
use crate::review_target::review_target_dir_from_prompt_at;
use anyhow::Result;
use async_trait::async_trait;
use clap::Parser;
use hi_ai::{ChatRequest, CompatMode, Completion, Provider, ServedModel, StreamEvent, ToolMode};
use std::path::PathBuf;

#[test]
fn canonical_remote_session_identity_survives_a_random_local_cache_name() {
    let local = std::path::Path::new("/tmp/random-local-continuation.jsonl");
    assert_eq!(
        canonical_session_identity(None, Some("remote-session"), local),
        "remote-session"
    );
    assert_eq!(
        canonical_session_identity(Some("explicit-session"), Some("remote-session"), local),
        "explicit-session"
    );
    assert_eq!(
        canonical_session_identity(None, None, local),
        "random-local-continuation"
    );
    assert_eq!(
        completed_session_switch("remote-session".to_string(), "summary".to_string()).id,
        "remote-session"
    );
}

#[test]
fn resumed_remote_identity_always_requires_authoritative_pipefs_probe() {
    assert!(pipefs_startup_authority_required(
        true, false, false, true, false
    ));
    assert!(pipefs_startup_authority_required(
        true, false, false, false, true
    ));
    assert!(!pipefs_startup_authority_required(
        false, false, false, true, false
    ));
    assert!(!pipefs_startup_authority_required(
        true, false, false, false, false
    ));
}

#[test]
fn tui_event_trace_accepts_only_the_full_interactive_frontend() {
    let interactive =
        Cli::try_parse_from(["hi", "--tui-events-jsonl", "/tmp/hi-tui-events-test.jsonl"]).unwrap();
    validate_tui_event_trace_request(&interactive, true, true).unwrap();
    assert!(validate_tui_event_trace_request(&interactive, false, true).is_err());

    let explicit_session = Cli::try_parse_from([
        "hi",
        "--session-file",
        "/tmp/hi-tui-session-test.jsonl",
        "--tui-events-jsonl",
        "/tmp/hi-tui-events-test.jsonl",
    ])
    .unwrap();
    validate_tui_event_trace_request(&explicit_session, true, true).unwrap();

    let plain = Cli::try_parse_from([
        "hi",
        "--plain",
        "--tui-events-jsonl",
        "/tmp/hi-tui-events-test.jsonl",
    ])
    .unwrap();
    assert!(validate_tui_event_trace_request(&plain, true, true).is_err());

    let one_shot = Cli::try_parse_from([
        "hi",
        "--tui-events-jsonl",
        "/tmp/hi-tui-events-test.jsonl",
        "fix it",
    ])
    .unwrap();
    assert!(validate_tui_event_trace_request(&one_shot, true, true).is_err());
}

#[test]
fn tui_event_trace_does_not_alias_delegate_progress_jsonl() {
    let error = Cli::try_parse_from([
        "hi",
        "--events-jsonl",
        "/tmp/delegate.jsonl",
        "--tui-events-jsonl",
        "/tmp/tui.jsonl",
    ])
    .unwrap_err();
    assert!(error.to_string().contains("cannot be used with"));
}

#[test]
fn automatic_workflow_detection_matches_plain_and_moa_plan_prompts() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("plan.md"), "- [ ] implement the parser\n").unwrap();
    let workspace = root.path().canonicalize().unwrap();

    let plain = Cli::try_parse_from(["hi", "implement plan.md"]).unwrap();
    assert_eq!(
        automatic_workflow_plan_path(&plain, &workspace, Some("implement plan.md")).as_deref(),
        Some("plan.md")
    );

    let moa = Cli::try_parse_from(["hi", "/moa implement plan.md"]).unwrap();
    assert_eq!(
        automatic_workflow_plan_path(&moa, &workspace, Some("/moa implement plan.md")).as_deref(),
        Some("plan.md")
    );

    let fleet = Cli::try_parse_from([
        "hi",
        "--session-file",
        "/tmp/hi-fleet-session.jsonl",
        "implement plan.md",
    ])
    .unwrap();
    assert!(automatic_workflow_plan_path(&fleet, &workspace, Some("implement plan.md")).is_none());
}

#[test]
fn optional_sync_storage_failure_keeps_the_local_session_running() {
    assert!(!sync_session_enabled(false, false, None, true));
    assert!(!sync_session_enabled(
        false,
        false,
        Some(crate::sync_store::SyncMode::On),
        false,
    ));
}

#[test]
fn available_sync_storage_honors_explicit_persisted_and_provider_modes() {
    assert!(sync_session_enabled(true, true, None, false));
    assert!(sync_session_enabled(
        true,
        false,
        Some(crate::sync_store::SyncMode::On),
        false,
    ));
    assert!(sync_session_enabled(true, false, None, true));
    assert!(!sync_session_enabled(
        true,
        false,
        Some(crate::sync_store::SyncMode::Off),
        true,
    ));
}

#[test]
fn skeptic_defaults_to_glm_on_pipenetwork_and_session_model_elsewhere() {
    assert_eq!(
        default_skeptic_model(ProviderName::Pipenetwork, "ipop/coder-balanced"),
        "pipe/glm-5.2"
    );
    assert_eq!(
        default_skeptic_model(ProviderName::Pipenetwork, "pipe/deepseek-v4-flash-0731"),
        "pipe/glm-5.2"
    );
    assert_eq!(
        default_skeptic_model(ProviderName::Anthropic, "claude-sonnet-5"),
        "claude-sonnet-5"
    );
    assert_eq!(
        default_skeptic_model(ProviderName::Ollama, "qwen2.5-coder"),
        "qwen2.5-coder"
    );
    // xAI: pin grok-4.6 as the reviewer, not a weaker session model.
    assert_eq!(
        default_skeptic_model(ProviderName::Xai, "grok-4.3"),
        "grok-4.6"
    );
    assert_eq!(
        default_skeptic_model(ProviderName::Xai, "grok-code-fast-1"),
        "grok-4.6"
    );
}

struct HangingModelListProvider;

#[async_trait]
impl Provider for HangingModelListProvider {
    async fn stream(
        &self,
        _request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        unreachable!("metadata discovery must not start a chat request")
    }

    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        std::future::pending().await
    }
}

#[tokio::test]
async fn hanging_optional_model_metadata_cannot_delay_startup_preparation() {
    let provider = HangingModelListProvider;
    let mut discovery = std::pin::pin!(provider.list_models());
    assert!(matches!(
        futures_util::poll!(&mut discovery),
        std::task::Poll::Pending
    ));

    let started = std::time::Instant::now();
    let metadata = startup_live_model_metadata();

    assert_eq!(metadata.context_window, None);
    assert_eq!(metadata.max_output_tokens, None);
    assert!(started.elapsed() < std::time::Duration::from_millis(50));
}

#[test]
fn auto_memory_off_when_disabled_or_unsaved() {
    assert!(auto_memory_enabled(false, false), "default on");
    assert!(!auto_memory_enabled(true, false), "--no-memory disables");
    assert!(!auto_memory_enabled(false, true), "--no-save disables");
}

#[test]
fn one_shot_exit_codes_follow_v2_outcomes() {
    let outcome = |status, verification| hi_agent::TurnOutcome {
        status,
        verification,
        review: hi_agent::ReviewStatus::NotRequired,
        stop_reason: hi_agent::TurnStopReason::Completed,
        changed_files: Vec::new(),
        verified_workspace_revision: None,
        effective_route: hi_agent::EffectiveModelRoute {
            provider: Some("test".into()),
            model: "model".into(),
        },
        review_same_model: false,
        leftover: None,
        plan_leftover: None,
    };
    assert_eq!(
        one_shot_exit_code(
            &outcome(
                hi_agent::TurnStatus::Completed,
                hi_agent::VerificationStatus::Passed,
            ),
            false,
            false,
        ),
        0
    );
    assert_eq!(
        one_shot_exit_code(
            &outcome(
                hi_agent::TurnStatus::Completed,
                hi_agent::VerificationStatus::Unverified,
            ),
            true,
            false,
        ),
        0
    );
    assert_eq!(
        one_shot_exit_code(
            &outcome(
                hi_agent::TurnStatus::Failed,
                hi_agent::VerificationStatus::Failed,
            ),
            false,
            false,
        ),
        1
    );
    assert_eq!(
        one_shot_exit_code(
            &outcome(
                hi_agent::TurnStatus::Failed,
                hi_agent::VerificationStatus::InfrastructureError,
            ),
            false,
            false,
        ),
        3
    );
    assert_eq!(
        one_shot_exit_code(
            &outcome(
                hi_agent::TurnStatus::Cancelled,
                hi_agent::VerificationStatus::Unverified,
            ),
            false,
            false,
        ),
        130
    );
    assert_eq!(
        one_shot_exit_code(
            &outcome(
                hi_agent::TurnStatus::Completed,
                hi_agent::VerificationStatus::Passed,
            ),
            false,
            true,
        ),
        1
    );

    let mut explicit_cap = outcome(
        hi_agent::TurnStatus::Failed,
        hi_agent::VerificationStatus::Passed,
    );
    explicit_cap.stop_reason = hi_agent::TurnStopReason::StepLimit;
    assert_eq!(
        one_shot_exit_code(&explicit_cap, false, false),
        1,
        "an explicit productive-work cap must not report successful completion"
    );

    let mut accepted_read_only_wrap_up = outcome(
        hi_agent::TurnStatus::Completed,
        hi_agent::VerificationStatus::NotApplicable,
    );
    accepted_read_only_wrap_up.stop_reason = hi_agent::TurnStopReason::ToolLimit;
    assert_eq!(
        one_shot_exit_code(&accepted_read_only_wrap_up, false, false),
        0,
        "a usable read-only cap wrap-up remains a completed answer"
    );
}

#[test]
fn report_stages_prefer_actual_execution_evidence() {
    let execution = hi_agent::VerificationExecution {
        round: 2,
        name: "test".into(),
        command: "cargo test".into(),
        status: hi_tools::ToolStatus::TimedOut,
        process: Some(hi_tools::ProcessOutcome {
            exit_code: None,
            stdout_summary: "partial output".into(),
            stderr_summary: String::new(),
            duration_ms: 30_000,
        }),
        truncation: Some(hi_tools::TruncationState::Truncated {
            original_bytes: 40_000,
            retained_bytes: 8_000,
        }),
    };
    let stages = report_verification_stages(&[execution], hi_agent::ReviewStatus::NotRequired);

    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0]["round"], 2);
    assert_eq!(stages[0]["status"], "timed_out");
    assert_eq!(stages[0]["process"]["duration_ms"], 30_000);
    assert_eq!(stages[0]["truncation"]["state"], "truncated");
    assert_ne!(stages[0]["name"], "configured");
    assert_eq!(stages[0]["name"], "cargo_test");
}

#[test]
fn report_stages_map_local_rust_names_to_public_verifiers() {
    let execution = hi_agent::VerificationExecution {
        round: 1,
        name: "clippy".into(),
        command: "cargo clippy".into(),
        status: hi_tools::ToolStatus::Succeeded,
        process: None,
        truncation: None,
    };
    let stages = report_verification_stages(&[execution], hi_agent::ReviewStatus::Passed);
    assert_eq!(stages[0]["name"], "cargo_clippy");
    assert_eq!(stages[1]["name"], "review");
    assert_eq!(stages[1]["status"], "passed");
}

#[test]
fn report_stages_do_not_claim_planned_checks_executed() {
    let stages = report_verification_stages(&[], hi_agent::ReviewStatus::NotRequired);
    assert!(stages.is_empty());
}

#[test]
fn report_tool_records_preserve_typed_evidence() {
    let entry = hi_agent::ToolCallEntry {
        tool: "bash".into(),
        path: String::new(),
        duration_ms: 17,
        queue_delay_ms: 0,
        completion_index: 1,
        status: hi_tools::ToolStatus::Failed,
        background: None,
        process: Some(hi_tools::ProcessOutcome {
            exit_code: Some(9),
            stdout_summary: "partial stdout".into(),
            stderr_summary: "failed".into(),
            duration_ms: 17,
        }),
        effects: hi_tools::ToolEffects {
            mutation_attempted: true,
            mutation_applied: true,
            file_changes: vec![hi_tools::FileChange {
                path: "src/lib.rs".into(),
                kind: hi_tools::FileChangeKind::Modify,
                before_digest: Some("sha256:before".into()),
                after_digest: Some("sha256:after".into()),
                before_len: Some(1),
                after_len: Some(2),
                before_mode: Some(0o100644),
                after_mode: Some(0o100644),
            }],
        },
        truncation: hi_tools::TruncationState::Truncated {
            original_bytes: 100,
            retained_bytes: 20,
        },
        error: true,
        progress_kind: "weak".into(),
        progress_reason: "tool returned an error".into(),
        normalized_signature: None,
        command: None,
        arg_chars: 0,
        result_chars: 0,
        truncated: true,
        kind: "shell".into(),
    };

    let records = report_tool_records(&[entry]);
    assert_eq!(records[0]["status"], "failed");
    assert_eq!(records[0]["process"]["exit_code"], 9);
    assert_eq!(records[0]["effects"]["mutation_applied"], true);
    assert_eq!(
        records[0]["effects"]["file_changes"][0]["path"],
        "src/lib.rs"
    );
    assert_eq!(records[0]["truncation"]["state"], "truncated");
}

#[test]
fn top_level_errors_never_fall_back_to_outcome_exit_one() {
    assert_eq!(top_level_error_code(&anyhow::anyhow!("usage: bad flag")), 2);
    assert_eq!(
        top_level_error_code(&anyhow::anyhow!("workspace runner crashed")),
        3
    );
}

#[test]
fn initialization_failure_still_writes_a_v2_report() {
    let path = std::env::temp_dir().join(format!(
        "hi-init-failure-report-{}-{:?}.json",
        std::process::id(),
        std::thread::current().id()
    ));
    write_initialization_failure_report(
        &path,
        "test-model",
        "test-provider",
        &anyhow::anyhow!("state root denied"),
        None,
        11,
        7,
    )
    .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let _ = std::fs::remove_file(path);
    assert_eq!(report["schema_version"], 2);
    assert_eq!(report["outcome"]["status"], "failed");
    assert_eq!(report["outcome"]["verification"], "unverified");
    assert_eq!(report["route"]["provider"], "test-provider");
    assert_eq!(report["changes"], serde_json::json!([]));
    assert_eq!(report["rsi"]["mode"], "off");
    assert_eq!(report["rsi"]["candidate_evidence"], true);
    assert_eq!(report["telemetry"]["effective_max_steps"], 11);
    assert_eq!(report["telemetry"]["effective_max_tool_calls"], 7);
}

#[test]
fn memory_context_wraps_nonempty_and_skips_blank() {
    let section = memory_context("- run cargo fmt before commits").unwrap();
    assert!(section.starts_with("# Memory (from past sessions)"));
    assert!(section.contains("- run cargo fmt before commits"));
    assert!(memory_context("   \n  ").is_none(), "blank → no section");
}

fn test_settings() -> Settings {
    Settings {
        execution: hi_agent::ExecutionMode::Ephemeral,
        provider: ProviderName::Openai,
        model: "gpt-4o".into(),
        base_url: String::new(),
        mcp_url: None,
        api_key: String::new(),
        max_tokens: 4096,
        max_tokens_explicit: true,
        top_p: None,
        output_token_parameter: hi_ai::OutputTokenParameter::Auto,
        thinking_budget: None,
        reasoning_effort: None,
        tool_mode: ToolMode::default(),
        compat: CompatMode::default(),
        deepseek_compat: hi_ai::DeepSeekCompat::default(),
        curate_skills: false,
        explore_subagents: true,
        suggest_next_prompt: true,
        write_subagents: hi_agent::WriteSubagentPolicy::Risk,
        planner_model: None,
        skeptic_model: None,
        moa: hi_ai::MoaConfig::default(),
        api_unix_socket: None,
        runtime: None,
        x402: Default::default(),
        browser_enabled: true,
        browser_allow_private: false,
        mcp_pipe_enabled: true,
        mcp_pipe_allow: Vec::new(),
    }
}

fn pipenetwork_settings(model: &str, max_tokens: u32, explicit: bool) -> Settings {
    Settings {
        execution: hi_agent::ExecutionMode::Ephemeral,
        provider: ProviderName::Pipenetwork,
        model: model.into(),
        base_url: String::new(),
        mcp_url: None,
        api_key: String::new(),
        max_tokens,
        max_tokens_explicit: explicit,
        top_p: None,
        output_token_parameter: hi_ai::OutputTokenParameter::Auto,
        thinking_budget: None,
        reasoning_effort: None,
        tool_mode: ToolMode::default(),
        compat: CompatMode::default(),
        deepseek_compat: hi_ai::DeepSeekCompat::default(),
        curate_skills: false,
        explore_subagents: true,
        suggest_next_prompt: true,
        write_subagents: hi_agent::WriteSubagentPolicy::Risk,
        planner_model: None,
        skeptic_model: None,
        moa: hi_ai::MoaConfig::default(),
        api_unix_socket: None,
        runtime: None,
        x402: Default::default(),
        browser_enabled: true,
        browser_allow_private: false,
        mcp_pipe_enabled: true,
        mcp_pipe_allow: Vec::new(),
    }
}

fn temp_review_dir(name: &str) -> PathBuf {
    let unique = format!(
        "hi-target-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dir = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&dir).unwrap();
    dir.canonicalize().unwrap()
}

#[test]
fn review_target_detects_absolute_directory() {
    let dir = temp_review_dir("absolute");
    let cwd = std::env::current_dir().unwrap();

    let found = review_target_dir_from_prompt_at(
        &format!("review {} and discuss only", dir.display()),
        &cwd,
        None,
    )
    .unwrap();

    assert_eq!(found, dir);
    let _ = std::fs::remove_dir_all(found);
}

#[test]
fn review_target_expands_home_directory() {
    let home = temp_review_dir("home");
    let repo = home.join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    let cwd = std::env::current_dir().unwrap();

    let found =
        review_target_dir_from_prompt_at("security review ~/repo read only", &cwd, Some(&home))
            .unwrap();

    assert_eq!(found, repo.canonicalize().unwrap());
    let _ = std::fs::remove_dir_all(home);
}

#[test]
fn review_target_ignores_non_review_prompt() {
    let dir = temp_review_dir("non-review");
    let cwd = std::env::current_dir().unwrap();

    let found = review_target_dir_from_prompt_at(&format!("fix {}", dir.display()), &cwd, None);

    assert!(found.is_none());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn review_target_ignores_paths_only_in_folded_stdin() {
    let dir = temp_review_dir("stdin");
    let cwd = std::env::current_dir().unwrap();
    let prompt = format!("review codebase\n\nstdin:\n```\n{}\n```", dir.display());

    let found = review_target_dir_from_prompt_at(&prompt, &cwd, None);

    assert!(found.is_none());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn pipenetwork_coding_routes_apply_live_output_limits() {
    let balanced = pipenetwork_settings("ipop/coder-balanced", 8192, false);
    assert_eq!(
        effective_max_tokens_for_model(&balanced, Some(131_072)),
        131_072
    );

    let auto_code = pipenetwork_settings("pipe/auto-coder", 8192, false);
    assert_eq!(
        effective_max_tokens_for_model(&auto_code, Some(16_384)),
        16_384
    );

    let flash = pipenetwork_settings("pipe/deepseek-v4-flash-0731", 8192, false);
    assert_eq!(effective_max_tokens_for_model(&flash, Some(16_384)), 16_384);
}

#[test]
fn explicit_max_tokens_survive_live_metadata_but_clamp_down() {
    let lower = pipenetwork_settings("ipop/coder-balanced", 4096, true);
    assert_eq!(effective_max_tokens_for_model(&lower, Some(131_072)), 4096);

    let too_high = pipenetwork_settings("pipe/auto-coder", 65_536, true);
    assert_eq!(
        effective_max_tokens_for_model(&too_high, Some(16_384)),
        16_384
    );
}

/// `write_landing` renders the block-letter "hi" banner.
/// We render into a `Vec<u8>`, strip ANSI escapes, and assert the banner
/// shape (5 figlet rows), the trailing model/cwd lines, and that the raw
/// output carries the orange SGR escape — no real file descriptors touched.
#[test]
fn write_landing_shows_hi_wordmark() {
    let mut buf: Vec<u8> = Vec::new();
    write_landing(&mut buf, &test_settings(), Some(128_000)).expect("render landing");

    let raw = String::from_utf8(buf).expect("utf8");
    let stripped = strip_ansi(&raw);
    let lines: Vec<&str> = stripped.lines().collect();

    // 5 banner rows + model line + cwd line = 7 content rows.
    assert!(
        lines.len() >= 7,
        "expected ≥7 lines (5 banner + model + cwd), got {}: {lines:?}",
        lines.len()
    );

    // The banner rows are the figlet art — they contain block-letter
    // strokes (pipes, underscores, slashes) and span 5 consecutive rows.
    let banner = &lines[0..5];
    // Every banner row is non-empty and carries pipe/underscore strokes.
    for (i, row) in banner.iter().enumerate() {
        assert!(
            row.contains('|') || row.contains('_'),
            "banner row {i} should carry figlet strokes, got: {row:?}"
        );
    }

    // Row 6 (index 5): model + provider + context window.
    let model_line = lines[5];
    assert!(
        model_line.contains("gpt-4o"),
        "model line missing model: {model_line:?}"
    );
    assert!(
        model_line.contains("openai"),
        "model line missing provider: {model_line:?}"
    );
    assert!(
        model_line.contains("128K context"),
        "model line missing context window: {model_line:?}"
    );

    // Row 7 (index 6): cwd — at minimum, non-empty (a path).
    assert!(
        !lines[6].is_empty(),
        "cwd line should be non-empty, got: {:?}",
        lines[6]
    );

    // The raw output must carry the orange SGR escape on banner rows.
    let orange_count = raw.matches("\x1b[38;2;255;140;0m").count();
    assert!(
        orange_count >= 5,
        "expected ≥5 orange SGR escapes (one per banner row), got {orange_count}"
    );
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            // Skip until we pass a letter (the terminator of a CSI sequence).
            i += 2;
            while i < bytes.len() && !bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
            i += 1;
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}
