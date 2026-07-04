use super::common::*;
use super::*;
use crate::steering::{
    GAPS_INSPECTION_CAP, REVIEW_INSPECTION_CAP, ROADMAP_INSPECTION_CAP, ReviewRepairMode,
    SECURITY_INSPECTION_CAP, STATUS_INSPECTION_CAP, active_read_only_inspection_cap,
    default_read_only_inspection_cap, explicit_read_only_inspection_cap, read_only_turn_prompt,
    repair_nudge_with_required_next,
};

#[test]
fn explicit_controls_classify_as_read_only_intents() {
    let status_macro = command::expand_prompt_macro("/status codebase state").unwrap();
    assert_eq!(
        classify_read_only_intent(&status_macro),
        Some(ReviewIntent::Status)
    );
    let security_macro = command::expand_prompt_macro("/security unsafe unwraps").unwrap();
    assert_eq!(
        classify_read_only_intent(&security_macro),
        Some(ReviewIntent::Security)
    );
    let audit_macro = command::expand_prompt_macro("/audit token leaks").unwrap();
    assert_eq!(
        classify_read_only_intent(&audit_macro),
        Some(ReviewIntent::Security)
    );
    let gaps_macro = command::expand_prompt_macro("/gaps missing coverage").unwrap();
    assert_eq!(
        classify_read_only_intent(&gaps_macro),
        Some(ReviewIntent::Gaps)
    );
    let roadmap_macro = command::expand_prompt_macro("/roadmap next work").unwrap();
    assert_eq!(
        classify_read_only_intent(&roadmap_macro),
        Some(ReviewIntent::Roadmap)
    );
    assert_eq!(
        classify_read_only_intent("review this code for auth leaks but do not edit"),
        Some(ReviewIntent::Security)
    );
    assert_eq!(
        classify_read_only_intent(
            "Review this codebase for issues related to ipop/coder-balanced API routing or latency. Use at most 4 file inspections. Do not modify files. Return concise findings only."
        ),
        Some(ReviewIntent::Review)
    );
    assert_eq!(
        classify_read_only_intent("review codebase and discuss status and state"),
        None
    );
    assert_eq!(classify_read_only_intent("status"), None);
    assert_eq!(classify_read_only_intent("fix the unsafe unwraps"), None);
}

#[test]
fn read_only_inspection_caps_are_intent_specific() {
    assert_eq!(
        default_read_only_inspection_cap(ReviewIntent::Review),
        REVIEW_INSPECTION_CAP
    );
    assert_eq!(
        default_read_only_inspection_cap(ReviewIntent::Status),
        STATUS_INSPECTION_CAP
    );
    assert_eq!(
        default_read_only_inspection_cap(ReviewIntent::Roadmap),
        ROADMAP_INSPECTION_CAP
    );
    assert_eq!(
        default_read_only_inspection_cap(ReviewIntent::Gaps),
        GAPS_INSPECTION_CAP
    );
    assert_eq!(
        default_read_only_inspection_cap(ReviewIntent::Security),
        SECURITY_INSPECTION_CAP
    );
}

#[test]
fn review_repair_modes_map_stable_metadata() {
    let expected = [
        (
            ReviewRepairMode::NoEvidence,
            "review_no_evidence",
            "review_no_evidence_exhausted",
            "inspect_files_before_answering",
            "no_evidence",
            4,
        ),
        (
            ReviewRepairMode::ListingOnly,
            "review_listing_only",
            "review_listing_only_exhausted",
            "inspect_one_concrete_file_before_answering",
            "listing",
            4,
        ),
        (
            ReviewRepairMode::GenericTemplate,
            "review_generic_template",
            "review_generic_disclaimer_exhausted",
            "produce_concrete_bounded_review",
            "generic",
            4,
        ),
        (
            ReviewRepairMode::InspectedDisclaimer,
            "review_inspected_disclaimer",
            "review_generic_disclaimer_exhausted",
            "chat_only_bounded_answer_from_inspected_files",
            "disclaimer",
            4,
        ),
        (
            ReviewRepairMode::InspectedDisclaimerChatAttempt,
            "review_inspected_disclaimer_chat_attempt",
            "review_generic_disclaimer_exhausted",
            "chat_only_bounded_answer_from_inspected_files",
            "disclaimer_chat",
            2,
        ),
        (
            ReviewRepairMode::ConcreteAnswer,
            "review_concrete_answer",
            "review_concrete_answer_exhausted",
            "cite_findings_plus_limits",
            "concrete",
            4,
        ),
        (
            ReviewRepairMode::ReadAfterSearch,
            "review_read_after_search",
            "review_read_after_search_exhausted",
            "read_one_matching_file_before_answering",
            "read_after_search",
            2,
        ),
        (
            ReviewRepairMode::SecurityBroadSearch,
            "review_security_broad_search",
            "review_security_broad_search_exhausted",
            "search_required_security_patterns_before_answering",
            "security_broad",
            4,
        ),
        (
            ReviewRepairMode::SecurityScope,
            "review_security_scope",
            "review_security_scope_exhausted",
            "bound_security_claims_to_inspected_evidence",
            "security_scope",
            5,
        ),
        (
            ReviewRepairMode::GapSearchOverclaim,
            "review_gap_search_overclaim",
            "review_gap_search_overclaim_exhausted",
            "cite_search_matches_plus_limits",
            "gap_overclaim",
            3,
        ),
    ];

    assert_eq!(ReviewRepairMode::ALL.len(), expected.len());
    for (mode, key, exhaustion, required_next, compact, limit) in expected {
        assert!(ReviewRepairMode::ALL.contains(&mode));
        assert_eq!(mode.key(), key);
        assert_eq!(mode.exhaustion_key(), exhaustion);
        assert_eq!(mode.required_next(), required_next);
        assert_eq!(mode.compact_label(), compact);
        assert_eq!(mode.default_limit(), limit);
        assert_eq!(crate::compact_review_repair_label(key), compact);
    }
    assert_eq!(
        crate::compact_review_repair_label("review_listing_only_exhausted"),
        "listing"
    );
    assert_eq!(
        crate::compact_review_repair_label("review_generic_disclaimer_exhausted"),
        "generic"
    );
}

#[test]
fn visible_review_repair_nudges_repeat_required_next_action() {
    let nudge = repair_nudge_with_required_next(
        ReviewRepairMode::ReadAfterSearch,
        "The targeted search result is already in the transcript.",
    );

    assert!(nudge.contains("Required next action `read_one_matching_file_before_answering`"));
    assert!(nudge.contains("read one matching file from the search results before answering"));
}

#[test]
fn explicit_read_only_inspection_cap_phrases_parse() {
    assert_eq!(
        explicit_read_only_inspection_cap("Review this codebase. Use at most 4 file inspections."),
        Some(4)
    );
    assert_eq!(
        explicit_read_only_inspection_cap("use no more than 4 reads"),
        Some(4)
    );
    assert_eq!(
        explicit_read_only_inspection_cap("max 4 file reads"),
        Some(4)
    );
    assert_eq!(
        explicit_read_only_inspection_cap("At most 20 file inspections; max 12 reads."),
        Some(12)
    );
}

#[test]
fn explicit_read_only_inspection_cap_ignores_non_caps() {
    assert_eq!(
        explicit_read_only_inspection_cap("read 4 files before answering"),
        None
    );
    assert_eq!(
        explicit_read_only_inspection_cap("use 4 reads if needed"),
        None
    );
    assert_eq!(
        explicit_read_only_inspection_cap("inspect at least 4 files"),
        None
    );
}

#[test]
fn active_read_only_inspection_cap_clamps_to_lower_prompt_limit() {
    assert_eq!(
        active_read_only_inspection_cap(
            "Security review. Use at most 4 file inspections.",
            ReviewIntent::Security
        ),
        4
    );
    assert_eq!(
        active_read_only_inspection_cap(
            "Security review. Use at most 40 file inspections.",
            ReviewIntent::Security
        ),
        SECURITY_INSPECTION_CAP
    );
    let prompt = read_only_turn_prompt(
        "Review this codebase. Use at most 4 file inspections.",
        ReviewIntent::Review,
    );
    assert!(prompt.contains("Active inspection cap: at most 4 file reads/searches"));
}

#[test]
fn build_macro_classifies_as_implementation_without_stealing_discussion() {
    let build_macro = command::expand_prompt_macro("/build gpu training TUI calculator").unwrap();
    let intent = classify_implementation_intent(&build_macro).expect("implementation prompt");
    assert!(intent.tui);
    assert!(intent.gpu_training_estimator);

    assert!(
        classify_implementation_intent(
            "discuss whats its missing and what we should considering building and implimenting"
        )
        .is_none()
    );
    assert_eq!(
        classify_read_only_intent(
            "discuss whats its missing and what we should considering building and implimenting"
        ),
        None
    );

    let prompt = implementation_turn_prompt(
        "/build gpu training calculator",
        ImplementationIntent {
            tui: true,
            gpu_training_estimator: true,
        },
    );
    assert!(prompt.contains("Ratatui"));
    assert!(prompt.contains("cargo init --bin ."));
    assert!(prompt.contains("training_flops = 6 * params * tokens"));
    assert!(prompt.contains("H100 80GB"));
    assert!(prompt.contains("validation command"));
}

#[test]
fn discuss_without_explicit_review_signal_stays_conversational() {
    for prompt in [
        "discuss status and state",
        r#"discuss this auth token status json: HealthLive{ "status": "ok", "auth": { "token": "redacted" } }"#,
        "discuss missing auth token json",
    ] {
        assert_eq!(
            classify_read_only_intent(prompt),
            None,
            "plain discuss prompt must not enter read-only review mode: {prompt:?}"
        );
        assert_eq!(
            classify_implementation_intent(prompt),
            None,
            "plain discuss prompt must not become an implementation request: {prompt:?}"
        );
    }

    assert_eq!(
        classify_read_only_intent("discuss only: review this code for auth leaks"),
        Some(ReviewIntent::Security)
    );
    assert_eq!(
        classify_read_only_intent("review codebase and discuss status and state"),
        None
    );
}

#[test]
fn ux_cleanup_with_live_json_stays_normal_agent_mode() {
    let prompt = r#"clean up UX so its not showing a bunch of json: HealthLive{
      "status": "ok",
      "ready": true,
      "secret_canary_enforced": false,
      "auth": { "token": "redacted" }
    } StatsLive{ "nodes_online": 1, "requests_failed": 0 }"#;

    assert_eq!(
        classify_implementation_intent(prompt),
        None,
        "ordinary UX cleanup prose should stay in normal agent mode"
    );
    assert_eq!(
        classify_read_only_intent(prompt),
        None,
        "pasted JSON must not trigger security review mode"
    );
}

#[test]
fn mutating_review_and_diagnostic_prompts_do_not_enter_read_only_review() {
    for prompt in [
        "review and fix auth token display in the login page",
        "review for security issues and fix them",
        "audit for token leaks and patch the backend route",
        "fix review page auth token display",
        "update status page UI",
    ] {
        assert_eq!(
            classify_implementation_intent(prompt),
            None,
            "ordinary mutating prose should stay in normal agent mode: {prompt:?}"
        );
        assert_eq!(
            classify_read_only_intent(prompt),
            None,
            "mutating prompt must not enter read-only review mode: {prompt:?}"
        );
    }

    for prompt in [
        r#"what is happening here: HealthLive{ "status": "ok", "auth": { "token": "redacted" } }"#,
        "update me on backend api status",
        "give me an update on the provider route state",
    ] {
        assert_eq!(
            classify_implementation_intent(prompt),
            None,
            "informational prompt must not become an implementation request: {prompt:?}"
        );
        assert_eq!(
            classify_read_only_intent(prompt),
            None,
            "pasted diagnostics/status wording must not invent a read-only repo review: {prompt:?}"
        );
    }

    assert_eq!(
        classify_read_only_intent("review this code for auth leaks but do not edit"),
        Some(ReviewIntent::Security)
    );
    assert_eq!(
        classify_read_only_intent("review codebase and discuss status and state"),
        None
    );
    assert_eq!(classify_read_only_intent("status"), None);
}

#[test]
fn plain_implementation_prose_does_not_trigger_implementation_mode() {
    for prompt in [
        "finish the av1 implementation",
        "finish the parser implementation",
        "finish the av1 implimentation",
        "implement the parser",
        "discuss the implementation",
        "analyze the implementation",
        "assess the implementation",
    ] {
        assert_eq!(
            classify_implementation_intent(prompt),
            None,
            "ordinary prose should not trigger implementation mode: {prompt:?}"
        );
    }
}

#[test]
fn implementation_preflight_detects_rust_validation() {
    let dir = std::env::temp_dir().join(format!(
        "hi-implementation-preflight-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(dir.join("src/main.rs"), "fn main() {}\n").unwrap();
    std::fs::write(dir.join("README.md"), "# demo\n").unwrap();
    std::fs::create_dir_all(dir.join("models/nested")).unwrap();
    std::fs::write(dir.join("models/nested/Cargo.toml"), "[package]\n").unwrap();
    std::fs::create_dir_all(dir.join(".turbo/docs")).unwrap();
    std::fs::write(dir.join(".turbo/docs/README.md"), "# generated\n").unwrap();

    let output = std::process::Command::new("sh")
        .arg("-lc")
        .arg(implementation_preflight_command())
        .current_dir(&dir)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let _ = std::fs::remove_dir_all(&dir);

    assert!(output.status.success());
    assert!(stdout.contains("[workspace_manifests]"));
    assert!(stdout.contains("./Cargo.toml"));
    assert!(stdout.contains("[likely_entrypoints]"));
    assert!(stdout.contains("./src/main.rs"));
    assert!(!stdout.contains("./models/nested/Cargo.toml"));
    assert!(!stdout.contains("./.turbo/docs/README.md"));
    assert_eq!(
        preferred_validation_from_preflight(&stdout),
        Some("cargo test".to_string())
    );
}

#[test]
fn gpu_training_estimator_cli_bootstrap_compiles_and_tests() {
    let dir = std::env::temp_dir().join(format!(
        "hi-gpu-estimator-cli-bootstrap-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (path, content) in gpu_training_estimator_bootstrap_files(ImplementationIntent {
        tui: false,
        gpu_training_estimator: true,
    }) {
        let path = dir.join(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    let output = std::process::Command::new("cargo")
        .arg("test")
        .current_dir(&dir)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        output.status.success(),
        "generated CLI project should pass cargo test\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

#[test]
fn gpu_training_estimator_tui_bootstrap_uses_ratatui_and_respects_existing_workspace() {
    let files = gpu_training_estimator_bootstrap_files(ImplementationIntent {
        tui: true,
        gpu_training_estimator: true,
    });
    let cargo_toml = files
        .iter()
        .find(|(path, _)| *path == "Cargo.toml")
        .map(|(_, content)| content)
        .unwrap();
    let main_rs = files
        .iter()
        .find(|(path, _)| *path == "src/main.rs")
        .map(|(_, content)| content)
        .unwrap();
    assert!(cargo_toml.contains("ratatui"));
    assert!(cargo_toml.contains("crossterm"));
    assert!(main_rs.contains("GPU Training Time Estimator"));
    assert!(main_rs.contains("estimate_seconds"));

    let dir = std::env::temp_dir().join(format!(
        "hi-gpu-estimator-existing-workspace-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("Cargo.toml"), "[package]\nname=\"existing\"\n").unwrap();
    let intent = ImplementationIntent {
        tui: true,
        gpu_training_estimator: true,
    };
    let can_bootstrap = intent.gpu_training_estimator
        && implementation_workspace_can_accept_rust_bootstrap_at(&dir);
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        !can_bootstrap,
        "bootstrap must not run over an existing manifest"
    );
}

#[tokio::test]
async fn implementation_turn_repairs_no_changes_and_missing_validation() {
    let path = temp_file("implementation-repair");
    let path_string = path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        ),
        write_completion(&path_string),
        completion(
            vec![Content::Text("Implemented the calculator.".into())],
            1,
            1,
        ),
        bash_completion("cargo test --help"),
        completion(
            vec![Content::Text(format!(
                "Changed {path_string} and validated with cargo test --help."
            ))],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecordingUi::default();
    agent
        .run_turn("/build a small CLI GPU training time estimator", &mut ui)
        .await
        .unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("no file changes")),
        "expected no-change repair status: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("without validation")),
        "expected validation repair status: {:?}",
        ui.statuses
    );
    assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 2);
    assert!(
        agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("validated with cargo test --help")
    );
}

#[tokio::test]
async fn stalled_implementation_does_not_finalize_with_stale_recap() {
    let path = temp_file("implementation-no-finalize");
    let path_string = path.to_string_lossy().to_string();
    let mut cfg = config();
    cfg.finalize = true;
    let responses = vec![
        write_completion(&path_string),
        completion(vec![Content::Text("Implemented it.".into())], 1, 1),
        completion(vec![Content::Text("Done.".into())], 1, 1),
        completion(vec![Content::Text("Final recap.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecordingUi::default();
    agent
        .run_turn("/build a small CLI GPU training time estimator", &mut ui)
        .await
        .unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(
        !agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("Final recap"),
        "stalled implementation should not finalize with a recap"
    );
    assert!(agent.last_turn_telemetry().stalled_unfinished);
}

#[tokio::test]
async fn scaffold_only_implementation_gets_source_edit_nudge() {
    let dir = temp_file("implementation-scaffold-only");
    let dir_string = dir.to_string_lossy().to_string();
    let responses = vec![
        bash_completion(&format!("mkdir -p {dir_string}")),
        completion(vec![Content::Text("Implemented it.".into())], 1, 1),
        completion(vec![Content::Text("Done.".into())], 1, 1),
        completion(vec![Content::Text("Final recap.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecordingUi::default();
    agent
        .run_turn("/build a small CLI GPU training time estimator", &mut ui)
        .await
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("only scaffolded setup files")),
        "expected scaffold-only repair status: {:?}",
        ui.statuses
    );
    assert!(
        !agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("Final recap"),
        "stalled implementation should not finalize with a recap"
    );
    assert!(agent.last_turn_telemetry().stalled_unfinished);
}

#[tokio::test]
async fn scaffold_only_repair_can_use_text_tool_fallback_for_source_edit() {
    let scaffold_dir = temp_file("implementation-scaffold-text-fallback-dir");
    let scaffold_dir_string = scaffold_dir.to_string_lossy().to_string();
    let source_path = temp_file("implementation-scaffold-text-fallback-src");
    let source_path_string = source_path.to_string_lossy().to_string();
    let xmlish_write = format!(
        "<tool_call>write<arg_key>path</arg_key><arg_value>{source_path_string}</arg_value><arg_key>content</arg_key><arg_value>implemented\n</arg_value></tool_call>"
    );
    let responses = vec![
        bash_completion(&format!("mkdir -p {scaffold_dir_string}")),
        completion(vec![Content::Text("Implemented it.".into())], 1, 1),
        completion(vec![Content::Text("Done.".into())], 1, 1),
        completion(vec![Content::Text(xmlish_write)], 1, 1),
        bash_completion("cargo test --help"),
        completion(
            vec![Content::Text(format!(
                "Changed {source_path_string} and validated with cargo test --help."
            ))],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecordingUi::default();
    agent
        .run_turn("/build a small CLI GPU training time estimator", &mut ui)
        .await
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(&source_path).unwrap(),
        "implemented\n"
    );
    let _ = std::fs::remove_dir_all(&scaffold_dir);
    let _ = std::fs::remove_file(&source_path);

    assert!(
        agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("validated with cargo test --help")
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("only scaffolded setup files")),
        "expected scaffold repair status: {:?}",
        ui.statuses
    );
}

#[test]
fn security_search_family_detection_covers_required_patterns() {
    let unsafe_only = security_search_families_for_tool(
        "grep",
        r#"{"pattern":"unwrap|expect|panic","glob":"*.rs"}"#,
    );
    assert!(unsafe_only.unsafe_or_panic);
    assert!(!unsafe_only.execution_or_fs_env);
    assert!(!unsafe_only.secret_or_auth);

    let path_does_not_count = security_search_families_for_tool(
        "grep",
        r#"{"pattern":"unwrap","path":"src/file_utils.rs"}"#,
    );
    assert!(path_does_not_count.unsafe_or_panic);
    assert!(!path_does_not_count.execution_or_fs_env);

    let broad = security_search_families_for_tool(
        "grep",
        r#"{"pattern":"unsafe|unwrap|expect|panic|command|std::process|spawn|std::fs|read_to_string|std::env|secret|token|auth|api_key|password|bearer","glob":"*.rs"}"#,
    );
    assert_eq!(
        broad,
        SecuritySearchFamilies {
            unsafe_or_panic: true,
            execution_or_fs_env: true,
            secret_or_auth: true,
        }
    );

    let shell = security_search_families_for_tool(
        "bash",
        r#"{"command":"rg 'exec|spawn|token|auth' crates"}"#,
    );
    assert!(!shell.unsafe_or_panic);
    assert!(shell.execution_or_fs_env);
    assert!(shell.secret_or_auth);
}

#[test]
fn incomplete_security_search_requires_broadening_after_read() {
    let mut evidence = EvidenceTracker::default();
    evidence.record_success(
        "grep",
        r#"{"pattern":"unwrap|expect|panic","glob":"*.rs"}"#,
        "src/lib.rs:1: value.unwrap()\n",
    );
    evidence.record_success("read", r#"{"path":"src/lib.rs"}"#, "1\tfn main() {}\n");

    assert!(should_nudge_security_broad_search(
        Some(ReviewIntent::Security),
        &evidence,
        "src/lib.rs: no command execution or secret issues were found."
    ));
    assert!(!evidence.security_search_complete());
}

#[test]
fn security_scope_overclaim_requires_bounded_answer() {
    let mut evidence = EvidenceTracker::default();
    evidence.record_success(
        "grep",
        r#"{"pattern":"unsafe|unwrap|expect|panic|command|std::process|spawn|std::fs|std::env|secret|token|auth","glob":"*.rs"}"#,
        "src/lib.rs:1: fn main() {}\n",
    );
    evidence.record_success("read", r#"{"path":"src/lib.rs"}"#, "1\tfn main() {}\n");

    assert!(should_nudge_security_scope(
        Some(ReviewIntent::Security),
        &evidence,
        "The codebase appears to be secure. There are no hardcoded secrets or direct command execution issues. Specifically, in `src/lib.rs`, no unsafe unwraps were found."
    ));
    assert!(!should_nudge_security_scope(
        Some(ReviewIntent::Security),
        &evidence,
        "Based on the inspected `src/lib.rs` and searched patterns, I did not establish a concrete unsafe unwrap finding. This is not a complete audit."
    ));
}

#[test]
fn generic_inventory_summary_with_path_is_not_accepted_as_status_review() {
    let mut evidence = EvidenceTracker::default();
    evidence.record_success("read", r#"{"path":"Cargo.toml"}"#, "[workspace]\n");
    evidence.record_success(
        "read",
        r#"{"path":"crates/hi-agent/src/lib.rs"}"#,
        "pub struct Agent;\n",
    );

    let generic = "The codebase is a Rust project structured with multiple crates. \
It has a workspace setup with Cargo.toml defining dependencies, and the main functionality \
revolves around an agent loop with tool calling capabilities.";
    assert!(should_nudge_concrete_review_answer(
        Some(ReviewIntent::Status),
        &evidence,
        generic
    ));

    let bounded = "Status:\n- Based on the inspected Cargo.toml and \
crates/hi-agent/src/lib.rs, the workspace exposes the agent crate and the current status \
surface is the agent loop.\n\nEvidence:\n- Cargo.toml and crates/hi-agent/src/lib.rs \
were inspected.\n\nRisks/Validation:\n- This is not a complete repo audit.";
    assert!(!should_nudge_concrete_review_answer(
        Some(ReviewIntent::Status),
        &evidence,
        bounded
    ));
}

#[test]
fn review_answer_needs_bounded_review_shape_not_just_a_path() {
    let mut evidence = EvidenceTracker::default();
    evidence.record_success("read", r#"{"path":"src/lib.rs"}"#, "fn main() {}\n");

    assert!(should_nudge_concrete_review_answer(
        Some(ReviewIntent::Review),
        &evidence,
        "src/lib.rs is part of the project and contains Rust code."
    ));
    assert!(!should_nudge_concrete_review_answer(
        Some(ReviewIntent::Review),
        &evidence,
        "Findings:\n- Based on the inspected src/lib.rs, no concrete issue was established in that file.\n\nEvidence:\n- src/lib.rs was read.\n\nFollow-up:\n- Inspect callers before making broader claims."
    ));
}

#[test]
fn concrete_review_accepts_distinctive_inspected_file_aliases() {
    let mut evidence = EvidenceTracker::default();
    evidence.record_success(
        "read",
        r#"{"path":"src/pages/top-up.tsx"}"#,
        "export function TopUp() { return null; }\n",
    );

    let basename_answer = "Findings:\n- top-up.tsx: Based on the inspected top-up page and security search patterns, no confirmed auth/token issue was established from this file alone.\n\nLimits:\n- This is limited to inspected evidence.";
    assert_eq!(
        concrete_review_answer_problem(Some(ReviewIntent::Security), &evidence, basename_answer),
        None
    );

    let stem_answer = "Findings:\n- top-up page: Based on the inspected file and security search patterns, no confirmed auth/token issue was established from this file alone.\n\nLimits:\n- This is limited to inspected evidence.";
    assert_eq!(
        concrete_review_answer_problem(Some(ReviewIntent::Security), &evidence, stem_answer),
        None
    );

    assert_eq!(
        concrete_review_answer_problem(
            Some(ReviewIntent::Security),
            &evidence,
            "Findings:\n- The inspected page has no confirmed token issue from the reviewed evidence."
        ),
        Some(ConcreteReviewAnswerProblem::MissingInspectedCitation)
    );

    assert_eq!(
        concrete_review_answer_problem(
            Some(ReviewIntent::Security),
            &evidence,
            "top-up.tsx is part of the project and contains React code."
        ),
        Some(ConcreteReviewAnswerProblem::MissingReviewShape)
    );
}

#[test]
fn concrete_review_accepts_concise_cited_answers_with_limits() {
    let mut security_evidence = EvidenceTracker::default();
    security_evidence.record_success(
        "read",
        r#"{"path":"src/lib.rs"}"#,
        "pub fn value(input: Option<i32>) -> i32 { input.unwrap_or_default() }\n",
    );
    assert_eq!(
        concrete_review_answer_problem(
            Some(ReviewIntent::Security),
            &security_evidence,
            "src/lib.rs: reviewed for unsafe/unwrap issues from inspected evidence; no confirmed finding from this file alone. Limits: only src/lib.rs was inspected."
        ),
        None
    );

    let mut status_evidence = EvidenceTracker::default();
    status_evidence.record_success("read", r#"{"path":"Cargo.toml"}"#, "[workspace]\n");
    assert_eq!(
        concrete_review_answer_problem(
            Some(ReviewIntent::Status),
            &status_evidence,
            "Cargo.toml: status reviewed from the inspected manifest; no current blocker was established. Limits: only Cargo.toml was inspected, and no validation was run."
        ),
        None
    );
}

#[test]
fn round_adds_evidence_detects_re_reads_and_re_searches() {
    let mut evidence = EvidenceTracker::default();
    // Record one read and one grep.
    evidence.record_success("read", r#"{"path":"src/lib.rs"}"#, "fn main() {}\n");
    evidence.record_success(
        "grep",
        r#"{"pattern":"unwrap","glob":"*.rs"}"#,
        "src/lib.rs:1: x.unwrap()\n",
    );
    evidence.record_success("bash_kill", r#"{"id":"bg_1"}"#, "[bg_1] already killed");

    let call = |name: &str, args: &str| (String::new(), name.to_string(), args.to_string());

    // Re-reading the same file adds no new evidence.
    assert!(
        !evidence.round_adds_evidence(&[call("read", r#"{"path":"src/lib.rs"}"#)]),
        "re-read of an inspected path adds no evidence"
    );
    assert!(
        evidence.round_adds_evidence(&[call("read", r#"{"path":"src/lib.rs","offset":241}"#)]),
        "a new read page from an inspected path adds evidence"
    );
    // Re-running the same grep adds no new evidence.
    assert!(
        !evidence.round_adds_evidence(&[call("grep", r#"{"pattern":"unwrap","glob":"*.rs"}"#)]),
        "re-run of a seen grep adds no evidence"
    );
    assert!(
        evidence.round_adds_evidence(&[call(
            "grep",
            r#"{"pattern":"unwrap","glob":"*.rs","context":2}"#
        )]),
        "grep with new context adds evidence"
    );
    // Reading a new file adds evidence.
    assert!(
        evidence.round_adds_evidence(&[call("read", r#"{"path":"src/main.rs"}"#)]),
        "read of a new path adds evidence"
    );
    // A new grep pattern adds evidence.
    assert!(
        evidence.round_adds_evidence(&[call("grep", r#"{"pattern":"panic","glob":"*.rs"}"#)]),
        "a new grep pattern adds evidence"
    );
    assert!(
        !evidence.round_adds_evidence(&[call("bash_kill", r#"{"id":"bg_1"}"#)]),
        "reusing a known-terminal background kill handle adds no evidence"
    );
    assert!(
        evidence.round_adds_evidence(&[call("bash_kill", r#"{"id":"bg_2"}"#)]),
        "a first kill attempt for a new background handle should execute"
    );
    // A mix of re-read and new read adds evidence (the new one).
    assert!(
        evidence.round_adds_evidence(&[
            call("read", r#"{"path":"src/lib.rs"}"#),
            call("read", r#"{"path":"src/main.rs"}"#),
        ]),
        "a mix of re-read and new read adds evidence"
    );
    // A mutating tool always adds evidence.
    assert!(
        evidence.round_adds_evidence(&[call("write", r#"{"path":"x","content":"y"}"#)]),
        "a mutating tool adds evidence"
    );
    // An empty round is treated as adding evidence (not a cycle).
    assert!(
        evidence.round_adds_evidence(&[]),
        "empty round is not a cycle"
    );
    assert!(
        evidence.round_adds_evidence(&[call("read", r#"{"path":42}"#)]),
        "un-signable read calls should execute and surface their tool error"
    );
}

#[test]
fn inspection_signature_is_stable_and_tool_specific() {
    assert_eq!(
        inspection_signature("read", r#"{"path":"src/lib.rs"}"#),
        Some("read:src/lib.rs:1:default".into())
    );
    assert_eq!(
        inspection_signature("read", r#"{"path":"src/lib.rs","limit":240,"offset":10}"#),
        Some("read:src/lib.rs:10:240".into())
    );
    assert_eq!(
        inspection_signature("read", r#"{"path":"src/lib.rs","offset":0}"#),
        Some("read:src/lib.rs:1:default".into())
    );
    assert_eq!(
        inspection_signature("read", r#"{"path":"src/lib.rs","offset":null}"#),
        Some("read:src/lib.rs:1:default".into())
    );
    assert_eq!(
        inspection_signature("list", r#"{"path":"."}"#),
        Some("list:.".into())
    );
    // list with no path defaults to ".".
    assert_eq!(inspection_signature("list", r#"{}"#), Some("list:.".into()));
    assert_eq!(
        inspection_signature("grep", r#"{"pattern":"unwrap","glob":"*.rs"}"#),
        Some("grep:unwrap:*.rs::0".into())
    );
    assert_eq!(
        inspection_signature("grep", r#"{"pattern":"unwrap","glob":"*.rs","context":2}"#),
        Some("grep:unwrap:*.rs::2".into())
    );
    assert_eq!(
        inspection_signature("grep", r#"{"pattern":"unwrap","context":null}"#),
        Some("grep:unwrap:::0".into())
    );
    assert_eq!(
        inspection_signature("glob", r#"{"pattern":"**/*.rs","path":"src"}"#),
        Some("glob:**/*.rs:src".into())
    );
    assert_eq!(
        inspection_signature("bash_output", r#"{"id":"bg_1"}"#),
        Some("bash_output:bg_1".into())
    );
    assert_eq!(
        inspection_signature("bash_kill", r#"{"id":"bg_1"}"#),
        Some("bash_kill:bg_1".into())
    );
    // Mutating/unclassified tools have no signature.
    assert_eq!(inspection_signature("write", r#"{"path":"x"}"#), None);
    assert_eq!(inspection_signature("bash", r#"{"command":"ls"}"#), None);
    assert_eq!(inspection_signature("read", r#"{"path":42}"#), None);
    assert_eq!(inspection_signature("bash_output", r#"{"id":""}"#), None);
    assert_eq!(inspection_signature("bash_kill", r#"{"id":""}"#), None);
    assert_eq!(
        inspection_signature("grep", r#"{"pattern":"unwrap","context":"two"}"#),
        None
    );
}

#[test]
fn search_hit_snippets_keep_late_high_signal_matches() {
    let inspected_path = temp_file("repair-search-ranking");
    std::fs::write(
        &inspected_path,
        "fn token() { let value = std::env::var(\"API_KEY\").unwrap(); }\n",
    )
    .unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let mut output = String::new();
    for line in 1..=12 {
        output.push_str(&format!("{inspected}:{line}:/// token budget note\n"));
    }
    output.push_str(&format!(
        "{inspected}:99:fn token() {{ let value = std::env::var(\"API_KEY\").unwrap(); }}\n"
    ));

    let mut evidence = EvidenceTracker::default();
    evidence.record_success(
        "grep",
        &serde_json::json!({
            "pattern": "unwrap|std::env|api_key|token",
            "glob": "*.rs"
        })
        .to_string(),
        &output,
    );

    assert_eq!(evidence.search_hit_snippets.len(), 8);
    assert!(
        evidence.search_hit_snippets[0].contains("std::env::var"),
        "late high-signal hit should outrank early token-only lines: {:?}",
        evidence.search_hit_snippets
    );
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn security_review_prompts_advertise_only_read_only_tools() {
    let responses = vec![
        completion(
            vec![Content::Text(
                "I need to inspect targeted search results or file reads first.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: r#"{"path":"Cargo.toml"}"#.into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Findings:\n- Cargo.toml was inspected as security review context.\n\nLimits:\n- Limited to inspected evidence.".into(),
            )],
            1,
            1,
        ),
    ];
    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(responses),
        tool_names: tool_names.clone(),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(Box::new(provider), config());
    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut NullUi,
        )
        .await
        .unwrap();

    let names = tool_names.lock().unwrap();
    let first = names.first().expect("request recorded");
    assert!(first.iter().any(|name| name == "read"));
    assert!(first.iter().any(|name| name == "grep"));
    assert!(first.iter().any(|name| name == "list"));
    assert!(!first.iter().any(|name| matches!(
        name.as_str(),
        "write" | "edit" | "multi_edit" | "apply_patch" | "bash"
    )));
    assert_eq!(modes.lock().unwrap()[0], ToolMode::Auto);
}

#[tokio::test]
async fn discuss_only_security_review_blocks_mutating_tool_call_execution() {
    let path = temp_file("readonly-block");
    std::fs::write(&path, "old\n").unwrap();
    let edit_args = serde_json::json!({
        "path": path.to_string_lossy().to_string(),
        "old_string": "old\n",
        "new_string": "new\n",
    })
    .to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "edit".into(),
                name: "edit".into(),
                arguments: edit_args,
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": path.to_string_lossy().to_string() })
                    .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Findings:\n- {}: inspected evidence only; no file changes were made.",
                path.to_string_lossy()
            ))],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert_eq!(std::fs::read_to_string(&path).unwrap(), "old\n");
    assert!(
        ui.tool_results
            .iter()
            .any(|(name, result)| { name == "edit" && result.contains("Tool `edit` blocked") }),
        "expected blocked edit tool result in transcript"
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn listing_only_review_final_gets_deepen_review_nudge() {
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "list".into(),
                name: "list".into(),
                arguments: r#"{"path":"."}"#.into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: r#"{"path":"Cargo.toml"}"#.into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Findings:\n- Cargo.toml defines the workspace members and gives concrete status context for this review.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("/status codebase state", &mut ui)
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("only a listing")),
        "expected deepen-review nudge status: {:?}",
        ui.statuses
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.quality_repair_nudges, 1);
    assert_eq!(telemetry.targeted_searches, 0);
    assert_eq!(telemetry.file_reads, 1);
    assert!(!telemetry.listing_only);
    assert_eq!(telemetry.discovery_depth, "mixed");
    assert!(
        agent
            .usage_summary(agent.totals())
            .contains("review-repair")
    );
}

#[tokio::test]
async fn read_only_review_generic_final_gets_concrete_evidence_nudge() {
    let inspected_path = temp_file("concrete-review");
    std::fs::write(&inspected_path, "fn main() { println!(\"ok\"); }\n").unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "No unsafe unwrap issues were found in the inspected code.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Findings:\n- {inspected}: inspected for unsafe, unwrap, expect, and panic patterns; no security-critical issue was established from that file alone."
            ))],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("lacked concrete inspected files")),
        "expected concrete-evidence nudge status: {:?}",
        ui.statuses
    );
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| message.role == Role::Assistant && message.text().contains(&inspected)),
        "final answer should cite inspected path"
    );
    assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 1);
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn security_review_accepts_inspected_filename_alias_in_final_answer() {
    let base = temp_file("security-alias-dir");
    let inspected_path = base.join("src/pages/top-up.tsx");
    std::fs::create_dir_all(inspected_path.parent().unwrap()).unwrap();
    std::fs::write(
        &inspected_path,
        "export function TopUp() { return <button>top up</button>; }\n",
    )
    .unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Findings:\n- top-up.tsx: Based on the inspected top-up page, no confirmed token/auth or command-execution issue was established from this file alone.\n\nLimits:\n- This is limited to inspected evidence and is not a complete audit."
                    .into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        !ui.assistant.contains("fallback summary"),
        "filename alias should be accepted instead of fallback: {}",
        ui.assistant
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("lacked concrete inspected files")),
        "should not nudge when final cites inspected filename alias: {:?}",
        ui.statuses
    );
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| message.role == Role::Assistant
                && message.text().contains("top-up.tsx")),
        "final answer should be recorded: {:?}",
        agent
            .messages()
            .iter()
            .map(|message| message.text())
            .collect::<Vec<_>>()
    );
    assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 0);
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn read_only_review_text_final_without_evidence_gets_inspection_nudge() {
    let inspected_path = temp_file("no-evidence-review");
    std::fs::write(&inspected_path, "fn main() { println!(\"ok\"); }\n").unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Findings:\n- {inspected}: inspected as the status evidence for this read-only review."
            ))],
            1,
            1,
        ),
    ];
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(Box::new(provider), config());
    let mut ui = RecUi::default();

    agent
        .run_turn("/status codebase state", &mut ui)
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("no inspected evidence")),
        "expected no-evidence nudge: {:?}",
        ui.statuses
    );
    let modes = modes.lock().unwrap();
    assert_eq!(modes[0], ToolMode::Auto);
    assert_eq!(modes[1], ToolMode::Required);
    assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 1);
    assert_eq!(agent.last_turn_telemetry().file_reads, 1);
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn read_only_status_preflight_seeds_first_request_with_evidence() {
    let mut cfg = config();
    cfg.read_only_preflight = true;
    let (mut agent, requests) = scripted_agent(
        vec![ProviderStep::Completion(completion(
            vec![Content::Text(
                "Status:\n- Cargo.toml and README.md were inspected as the workspace manifest and project overview for this status review."
                    .into(),
            )],
            10,
            4,
        ))],
        cfg,
    );

    let mut ui = RecUi::default();
    agent
        .run_turn("/status codebase state", &mut ui)
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("read-only preflight")),
        "expected preflight status: {:?}",
        ui.statuses
    );
    let requests = requests.lock().unwrap();
    let first = requests.first().expect("provider request");
    let mut tool_names = Vec::new();
    let mut tool_results = String::new();
    for message in first {
        for content in &message.content {
            match content {
                Content::ToolCall { name, .. } => tool_names.push(name.clone()),
                Content::ToolResult { output, .. } => {
                    tool_results.push_str(output);
                    tool_results.push('\n');
                }
                _ => {}
            }
        }
    }
    assert!(
        tool_names.iter().any(|name| name == "diff"),
        "{tool_names:?}"
    );
    assert!(
        tool_names.iter().any(|name| name == "read"),
        "{tool_names:?}"
    );
    assert!(tool_results.contains("[package]") || tool_results.contains("[workspace]"));
    let telemetry = agent.last_turn_telemetry();
    assert!(telemetry.tool_calls >= 3, "{telemetry:?}");
    assert!(telemetry.file_reads >= 2, "{telemetry:?}");
    assert_eq!(telemetry.targeted_searches, 0, "{telemetry:?}");
    assert!(!telemetry.listing_only, "{telemetry:?}");
    assert_eq!(telemetry.first_tool_kind, "listing");
}

#[tokio::test]
async fn ux_cleanup_with_live_json_does_not_enter_read_only_preflight() {
    let mut cfg = config();
    cfg.read_only_preflight = true;
    let path = temp_file("ux-json-implementation");
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_completion(&path.to_string_lossy())),
            ProviderStep::Completion(bash_completion("cargo --version # cargo check")),
            ProviderStep::Completion(completion(
                vec![Content::Text("Implemented the overview summary UI.".into())],
                10,
                4,
            )),
        ],
        cfg,
    );

    let mut ui = RecUi::default();
    agent
        .run_turn(
            r#"clean up UX so its not showing a bunch of json: HealthLive{
              "status": "ok",
              "ready": true,
              "secret_canary_enforced": false,
              "auth": { "token": "redacted" }
            } StatsLive{ "nodes_online": 1, "requests_failed": 0 }"#,
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("read-only preflight")),
        "UX cleanup must not run read-only preflight: {:?}",
        ui.statuses
    );
    assert!(
        !ui.assistant.contains("fallback summary"),
        "implementation prompts must not return review fallback summaries: {}",
        ui.assistant
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn security_preflight_is_code_scoped_and_bounded() {
    let calls = read_only_preflight_initial_calls(ReviewIntent::Security);
    let mut read_paths = Vec::new();
    let mut grep_args = String::new();
    for call in &calls {
        if call.name == "read" {
            if let Some(path) = hi_tools::target_path(call.name, &call.arguments) {
                read_paths.push(path);
            }
        } else if call.name == "grep" {
            grep_args = call.arguments.clone();
        }
    }

    assert!(read_paths.iter().any(|path| path == "Cargo.toml"));
    assert!(!read_paths.iter().any(|path| path == "README.md"));
    assert!(grep_args.contains(r#""glob":"*.rs""#), "{grep_args}");
    assert!(grep_args.contains(r#""context":0"#), "{grep_args}");
    assert!(preflight_path_relevant_for_intent(
        ReviewIntent::Security,
        "crates/hi-agent/src/lib.rs"
    ));
    assert!(!preflight_path_relevant_for_intent(
        ReviewIntent::Security,
        "README.md"
    ));

    let long_grep = (0..40)
        .map(|i| format!("src/lib.rs:{i}:unwrap()"))
        .collect::<Vec<_>>()
        .join("\n");
    let compacted = compact_preflight_tool_output("grep", &long_grep);
    assert!(compacted.contains("preflight grep output truncated"));
    assert!(compacted.lines().count() <= READ_ONLY_PREFLIGHT_GREP_MAX_LINES + 1);

    let long_diff = (0..(READ_ONLY_PREFLIGHT_DIFF_MAX_LINES + 25))
        .map(|i| format!("diff line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let compacted = compact_preflight_tool_output("diff", &long_diff);
    assert!(compacted.contains("preflight diff output truncated"));
    assert!(compacted.lines().count() <= READ_ONLY_PREFLIGHT_DIFF_MAX_LINES + 1);
}

#[tokio::test]
async fn read_only_review_no_evidence_repair_exhaustion_stops_incomplete() {
    let responses = vec![
        completion(
            vec![Content::Text(
                "Not enough evidence to review without inspecting files.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Not enough evidence to review without inspecting files.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Not enough evidence to review without inspecting files.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Not enough evidence to review without inspecting files.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Not enough evidence to review without inspecting files.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.assistant.trim().is_empty(),
        "guardrail should not emit canned assistant text: {}",
        ui.assistant
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("no inspected evidence after repair")),
        "expected exhausted no-evidence status: {:?}",
        ui.statuses
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.quality_repair_nudges, 4);
    assert_eq!(telemetry.discovery_depth, "none");
    assert_eq!(telemetry.last_stall_reason, "review_no_evidence_exhausted");
    assert!(telemetry.stalled_unfinished);
}

#[tokio::test]
async fn listing_only_review_gets_full_budget_after_no_evidence_repair() {
    let responses = vec![
        completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "list".into(),
                name: "list".into(),
                arguments: r#"{"path":"."}"#.into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("/status codebase state", &mut ui)
        .await
        .unwrap();

    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.quality_repair_nudges, 5);
    assert_eq!(telemetry.review_repair_counts["review_no_evidence"], 1);
    assert_eq!(telemetry.review_repair_counts["review_listing_only"], 4);
    assert_eq!(
        telemetry.review_repair_exhaustion_reason,
        "review_listing_only_exhausted"
    );
    assert_eq!(telemetry.last_stall_reason, "review_listing_only_exhausted");
    assert!(telemetry.stalled_unfinished);
}

#[tokio::test]
async fn rejected_text_only_review_draft_is_compacted_in_repair_context() {
    let responses = vec![
        ProviderStep::Completion(completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        )),
        ProviderStep::Completion(completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: r#"{"path":"Cargo.toml"}"#.into(),
            }],
            1,
            1,
        )),
        ProviderStep::Completion(completion(
            vec![Content::Text(
                "Status:\n- Based on the inspected Cargo.toml, the workspace manifest was reviewed.\n\nEvidence:\n- Cargo.toml was read.\n\nRisks/Validation:\n- Limited to inspected evidence.".into(),
            )],
            1,
            1,
        )),
    ];
    let (mut agent, requests) = scripted_agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("/status codebase state", &mut ui)
        .await
        .unwrap();

    let requests = requests.lock().unwrap();
    assert!(
        requests.len() >= 2,
        "expected a repair request: {requests:?}"
    );
    let second_request_text = requests[1]
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        second_request_text.contains(
            "[review retry: reason=review_no_evidence; required_next=inspect_files_before_answering; do_not_repeat_previous_draft]"
        ),
        "repair note missing from provider context: {second_request_text}"
    );
    assert!(
        !second_request_text.contains("The repository looks healthy and organized."),
        "rejected draft should not be fed back verbatim: {second_request_text}"
    );
    assert!(
        !second_request_text
            .to_ascii_lowercase()
            .contains("insufficient evidence"),
        "repair note should avoid noisy provider/review trigger wording: {second_request_text}"
    );

    assert!(
        agent.messages().iter().any(|message| {
            message.role == Role::Assistant
                && message.content.iter().any(
                    |content| matches!(content, Content::ToolCall { name, .. } if name == "read"),
                )
        }),
        "assistant tool-call turns must still be stored normally"
    );
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| message.role == Role::Assistant
                && message.text().contains("Based on the inspected Cargo.toml")),
        "accepted final review answer should be stored normally"
    );
}

#[tokio::test]
async fn read_only_review_repair_template_final_is_not_accepted() {
    let inspected_path = temp_file("repair-template");
    std::fs::write(&inspected_path, "# hi\n\nA terminal coding assistant.\n").unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Findings/Status:\n- The inspected context points to these concrete review targets: {inspected}, ./Cargo.toml.\n- Review observations should stay tied to those files or modules instead of only summarizing the repository layout.\n\nConcrete Follow-up:\n- Convert any broad status claims into file-specific findings before recommending changes."
            ))],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Status:\n- `{inspected}` identifies this as a terminal coding assistant.\n\nEvidence:\n- Read `{inspected}` during this review.\n\nBuild Next:\n- Inspect command routing and tool execution modules before making broader status claims.\n\nRisks/Validation:\n- Limited to inspected evidence; not a full repository review."
            ))],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("/status codebase state", &mut ui)
        .await
        .unwrap();

    assert!(
        ui.assistant.contains(&inspected),
        "repaired model answer should cite inspected evidence: {}",
        ui.assistant
    );
    assert!(
        !ui.assistant.contains("Findings/Status"),
        "old repair template must not be surfaced: {}",
        ui.assistant
    );
    assert!(!agent.last_turn_telemetry().stalled_unfinished);
    assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 1);
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn read_only_review_repair_exhaustion_reports_inspected_evidence() {
    let inspected_path = temp_file("repair-exhaustion-evidence");
    std::fs::write(&inspected_path, "pub fn value() -> i32 { 1 }\n").unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        ),
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        ),
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        ),
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        ),
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.assistant.trim().is_empty(),
        "guardrail should not emit canned assistant text: {}",
        ui.assistant
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.quality_repair_nudges, 4);
    assert_eq!(
        telemetry.last_stall_reason,
        "review_concrete_answer_exhausted"
    );
    assert!(telemetry.stalled_unfinished);
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn read_only_review_generic_insufficient_after_read_reports_evidence() {
    let inspected_path = temp_file("generic-insufficient-after-read");
    std::fs::write(
        &inspected_path,
        "pub fn value(input: Option<i32>) -> i32 { input.unwrap_or_default() }\n",
    )
    .unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "grep".into(),
                name: "grep".into(),
                arguments: serde_json::json!({
                    "pattern": "unsafe|unwrap|expect|panic|std::process|std::fs|std::env|secret|token|auth",
                    "glob": "*.rs",
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Not enough evidence: I inspected `{inspected}`, but cannot make concrete security findings."
            ))],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Not enough evidence: I inspected `{inspected}`, but still cannot make concrete security findings."
            ))],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Not enough evidence: I inspected `{inspected}`, but still cannot make concrete security findings."
            ))],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Not enough evidence: I inspected `{inspected}`, but still cannot make concrete security findings."
            ))],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Not enough evidence: I inspected `{inspected}`, but still cannot make concrete security findings."
            ))],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.assistant.trim().is_empty(),
        "guardrail should not emit canned assistant text: {}",
        ui.assistant
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("nudging the model to answer from inspected files")),
        "expected summarize-evidence repair status: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("generic evidence disclaimer after inspection")),
        "expected replacement status: {:?}",
        ui.statuses
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(
        telemetry.last_stall_reason,
        "review_generic_disclaimer_exhausted"
    );
    assert!(telemetry.stalled_unfinished);
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn read_only_review_generic_insufficient_after_read_gets_summary_repair() {
    let inspected_path = temp_file("generic-insufficient-summary-repair");
    std::fs::write(
        &inspected_path,
        "pub fn value(input: Option<i32>) -> i32 { input.unwrap_or_default() }\n",
    )
    .unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "grep".into(),
                name: "grep".into(),
                arguments: serde_json::json!({
                    "pattern": "unsafe|unwrap|expect|panic|std::process|std::fs|std::env|secret|token|auth",
                    "glob": "*.rs",
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Not enough evidence: I inspected `{inspected}`, but cannot make concrete security findings."
            ))],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Findings:\n- `{inspected}` uses `unwrap_or_default`; from the inspected file this is a fallback conversion, not a panic-prone unwrap.\n\nInspected Evidence:\n- `{inspected}` was read after the targeted search.\n\nLimits:\n- This is not a complete audit of uninspected files."
            ))],
            1,
            1,
        ),
    ];
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(Box::new(provider), config());
    let mut ui = RecUi::default();

    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("nudging the model to answer from inspected files")),
        "expected summarize-evidence repair status: {:?}",
        ui.statuses
    );
    assert!(
        ui.assistant.contains(&inspected),
        "final answer should cite inspected path: {}",
        ui.assistant
    );
    assert!(
        !ui.assistant.contains("fallback summary"),
        "accepted repaired answer should not fall back: {}",
        ui.assistant
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.quality_repair_nudges, 1);
    assert!(!telemetry.stalled_unfinished);
    let modes = modes.lock().unwrap();
    assert_eq!(
        modes.last(),
        Some(&ToolMode::ChatOnly),
        "summary repair should force a chat-only answer attempt: {modes:?}"
    );
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn inspected_disclaimer_chat_attempts_do_not_share_unrelated_repair_budget() {
    let inspected_path = temp_file("inspected-disclaimer-independent");
    std::fs::write(&inspected_path, "[workspace]\n").unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Not enough evidence to provide a review without more file reads.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Not enough evidence to provide a review without more file reads.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Status:\n- Based on the inspected {inspected}, the workspace manifest was reviewed.\n\nEvidence:\n- {inspected} was read.\n\nRisks/Validation:\n- Limited to inspected evidence."
            ))],
            1,
            1,
        ),
    ];
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(Box::new(provider), config());
    let mut ui = RecUi::default();

    agent
        .run_turn("/status codebase state", &mut ui)
        .await
        .unwrap();

    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.review_repair_counts["review_no_evidence"], 1);
    assert_eq!(
        telemetry.review_repair_counts["review_inspected_disclaimer"],
        2
    );
    assert_eq!(
        telemetry.review_repair_counts["review_inspected_disclaimer_chat_attempt"],
        2
    );
    assert_eq!(telemetry.quality_repair_nudges, 3);
    assert!(!telemetry.stalled_unfinished);
    let modes = modes.lock().unwrap();
    assert_eq!(
        modes.as_slice(),
        &[
            ToolMode::Auto,
            ToolMode::Required,
            ToolMode::Auto,
            ToolMode::ChatOnly,
            ToolMode::ChatOnly,
        ],
        "inspected-disclaimer repairs should force independent chat-only answer attempts: {modes:?}"
    );
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn read_only_review_repeat_exhaustion_reports_inspected_evidence() {
    let inspected_path = temp_file("repeat-exhaustion-evidence");
    std::fs::write(
        &inspected_path,
        "pub fn value() -> Option<i32> { Some(1) }\n",
    )
    .unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let grep_args = serde_json::json!({
        "pattern": "unwrap\\(",
        "glob": "*.rs",
    })
    .to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "grep1".into(),
                name: "grep".into(),
                arguments: grep_args.clone(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "grep2".into(),
                name: "grep".into(),
                arguments: grep_args.clone(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "grep3".into(),
                name: "grep".into(),
                arguments: grep_args.clone(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "grep4".into(),
                name: "grep".into(),
                arguments: grep_args,
            }],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.assistant.trim().is_empty(),
        "guardrail should not emit canned assistant text: {}",
        ui.assistant
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("turn stopped incomplete")),
        "expected forced final recovery to stop incomplete: {:?}",
        ui.statuses
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.repeat_nudges, 2);
    assert_eq!(telemetry.forced_final_answer_attempts, 1);
    assert!(telemetry.stalled_unfinished);
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn gap_review_search_match_blocks_no_gap_overclaim() {
    let inspected_path = temp_file("gap-overclaim-evidence");
    std::fs::write(
        &inspected_path,
        "// TODO: add provider retry coverage\npub fn value() {}\n",
    )
    .unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "grep".into(),
                name: "grep".into(),
                arguments: serde_json::json!({
                    "pattern": "TODO|FIXME|missing|gap",
                    "path": inspected.clone(),
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "{inspected}: The project appears mature with no obvious gaps and no TODO/FIXME markers."
            ))],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "{inspected}: The project appears mature with no obvious gaps and no TODO/FIXME markers."
            ))],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "{inspected}: The project appears mature with no obvious gaps and no TODO/FIXME markers."
            ))],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "{inspected}: The project appears mature with no obvious gaps and no TODO/FIXME markers."
            ))],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("/gaps missing coverage and build-next work", &mut ui)
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("contradicted search matches")),
        "expected gap overclaim nudge: {:?}",
        ui.statuses
    );
    assert!(
        ui.assistant.trim().is_empty(),
        "guardrail should not emit canned assistant text: {}",
        ui.assistant
    );
    let telemetry = agent.last_turn_telemetry();
    assert!(telemetry.quality_repair_nudges >= 1);
    assert!(telemetry.stalled_unfinished);
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn security_review_with_partial_search_gets_broad_search_nudge() {
    let inspected_path = temp_file("security-broad-search");
    std::fs::write(
        &inspected_path,
        "fn run() { let value = Some(1).unwrap(); }\n",
    )
    .unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "list".into(),
                name: "list".into(),
                arguments: r#"{"path":"."}"#.into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "No security issues or unsafe unwraps were found.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "grep".into(),
                name: "grep".into(),
                arguments: serde_json::json!({
                    "pattern": "unwrap|expect|panic",
                    "glob": "*.rs",
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Findings:\n- {inspected}: no unsafe unwrap, command execution, filesystem/env, or secret/token/auth risks were found."
            ))],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "grep-broad".into(),
                name: "grep".into(),
                arguments: serde_json::json!({
                    "pattern": "unsafe|unwrap|expect|panic|command|std::process|process::|shell|exec|spawn|filesystem|std::fs|fs::|read_to_string|write|remove_file|std::env|env::|secret|token|auth|api_key|apikey|password|credential|bearer",
                    "glob": "*.rs",
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read-again".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Findings:\n- {inspected}: searched unsafe/unwrap/panic, command/filesystem/env, and secret/token/auth patterns; this file contains a direct unwrap but no broader conclusion is made beyond inspected evidence."
            ))],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("missed required pattern families")),
        "expected security broad-search nudge: {:?}",
        ui.statuses
    );
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| message.role == Role::Assistant
                && message.text().contains(&inspected)
                && message.text().contains("direct unwrap")),
        "final answer should cite inspected path after broad search"
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.quality_repair_nudges, 2);
    assert_eq!(telemetry.targeted_searches, 2);
    assert_eq!(telemetry.file_reads, 2);
    assert!(!telemetry.listing_only);
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn security_review_overbroad_all_clear_gets_scope_nudge() {
    let inspected_path = temp_file("security-scope");
    std::fs::write(&inspected_path, "fn main() { println!(\"ok\"); }\n").unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "grep".into(),
                name: "grep".into(),
                arguments: serde_json::json!({
                    "pattern": "unsafe|unwrap|expect|panic|command|std::process|spawn|std::fs|std::env|secret|token|auth",
                    "glob": "*.rs",
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "The codebase appears to be secure. There are no hardcoded secrets or direct command execution issues. Specifically, in `{inspected}`, no unsafe unwraps were found."
            ))],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Findings:\n- {inspected}: Based on the inspected file and searched security patterns, I did not establish a concrete unsafe/unwrap finding in this file. This is not a complete audit and does not rule out issues outside the inspected evidence."
            ))],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("overclaimed repo-wide safety")),
        "expected security scope nudge: {:?}",
        ui.statuses
    );
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| message.role == Role::Assistant
                && message.text().contains("not a complete audit")),
        "final answer should be bounded"
    );
    assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 1);
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn read_only_review_repeated_search_without_read_stops_incomplete() {
    let grep_call = || {
        completion(
            vec![Content::ToolCall {
                id: "grep".into(),
                name: "grep".into(),
                arguments: serde_json::json!({
                    "pattern": "fn run_turn",
                    "glob": "*.rs",
                })
                .to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![grep_call(), grep_call(), grep_call(), grep_call()];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("nudging it to read a matching file")),
        "expected read-after-search nudge: {:?}",
        ui.statuses
    );
    assert!(
        ui.assistant.trim().is_empty(),
        "guardrail should not emit canned assistant text: {}",
        ui.assistant
    );
    assert!(agent.last_turn_telemetry().stalled_unfinished);
}

#[tokio::test]
async fn read_only_review_search_then_generic_final_requires_file_read() {
    let inspected_path = temp_file("search-then-read-review");
    std::fs::write(&inspected_path, "pub fn run_turn() {}\n").unwrap();
    let inspected = inspected_path.to_string_lossy().to_string();
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "grep".into(),
                name: "grep".into(),
                arguments: serde_json::json!({
                    "pattern": "unwrap|expect|panic",
                    "glob": "*.rs",
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Targeted search ran, but I have not read a matching file yet.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "read".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Findings:\n- `{inspected}` was read after targeted search and contains the reviewed entrypoint.\n\nEvidence:\n- Read `{inspected}`.\n\nLimits:\n- Limited to inspected evidence."
            ))],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent.run_turn("/review codebase", &mut ui).await.unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("targeted search but no file reads")),
        "expected search-without-read nudge: {:?}",
        ui.statuses
    );
    assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 2);
    assert_eq!(agent.last_turn_telemetry().file_reads, 1);
    assert!(
        ui.assistant.contains(&inspected),
        "final answer should cite the file read after the nudge: {}",
        ui.assistant
    );
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn listing_only_review_repair_exhaustion_stops_incomplete() {
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "list".into(),
                name: "list".into(),
                arguments: r#"{"path":"."}"#.into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Only a directory listing is available; I need to inspect files before reviewing."
                    .into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Findings/Status:\n- The inspected context points to `src/lib.rs`.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The repository looks healthy and organized.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    agent
        .run_turn("/status codebase state", &mut ui)
        .await
        .unwrap();

    assert!(
        ui.assistant.trim().is_empty(),
        "guardrail should not emit canned assistant text: {}",
        ui.assistant
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("only listing evidence after repair")),
        "expected exhausted repair status: {:?}",
        ui.statuses
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.quality_repair_nudges, 4);
    assert!(telemetry.listing_only);
    assert_eq!(telemetry.last_stall_reason, "review_listing_only_exhausted");
    assert!(telemetry.stalled_unfinished);
    assert!(agent.usage_summary(agent.totals()).contains("stalled"));
}
