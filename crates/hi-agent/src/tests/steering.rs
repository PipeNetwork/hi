use super::common::*;
use super::*;
use crate::steering::{
    ReviewRepairMode, is_read_only_inspection_tool, read_only_preflight_initial_calls_in,
    read_only_turn_prompt, repair_nudge_with_required_next,
};

#[tokio::test]
async fn review_answers_are_preserved_without_evidence_or_format_repair() {
    // These answers used to trigger separate no-evidence, listing-only,
    // disclaimer, format, security-scope, and gap-overclaim repair loops.
    for (index, prompt, tool, answer) in [
        (
            0,
            "review this code, do not edit",
            None,
            "Insufficient evidence to determine the cause; the relevant service is unavailable.",
        ),
        (
            1,
            "review this code, do not edit",
            Some(("list", r#"{"path":"."}"#)),
            "The workspace contains src/lib.rs.",
        ),
        (
            2,
            "review this code, do not edit",
            Some(("read", r#"{"path":"src/lib.rs"}"#)),
            "Insufficient evidence for the remote caller, but src/lib.rs returns 42.",
        ),
        (
            3,
            "review security, do not edit",
            Some(("read", r#"{"path":"src/lib.rs"}"#)),
            "No security issues found in src/lib.rs.",
        ),
        (
            4,
            "review gaps, do not edit",
            Some(("grep", r#"{"path":"src/lib.rs","pattern":"TODO"}"#)),
            "No missing implementations; the TODO in src/lib.rs is a documentation reminder.",
        ),
    ] {
        let workspace = IsolatedWorkspace::new(&format!("preserve-review-{index}"));
        std::fs::create_dir_all(workspace.path("src")).unwrap();
        std::fs::write(
            workspace.path("src/lib.rs"),
            "// TODO: document answer\npub fn answer() -> u32 { 42 }\n",
        )
        .unwrap();
        let mut responses = Vec::new();
        if let Some((name, arguments)) = tool {
            responses.push(completion(
                vec![Content::ToolCall {
                    id: "inspect".into(),
                    name: name.into(),
                    arguments: arguments.into(),
                }],
                1,
                1,
            ));
        }
        responses.push(completion(vec![Content::Text(answer.into())], 1, 1));
        let mut cfg = workspace.config();
        cfg.gates.read_only_preflight = false;
        let mut agent = agent(responses, cfg);
        let outcome = agent.run_turn(prompt, &mut NullUi).await.unwrap();
        assert_eq!(
            outcome.status,
            TurnStatus::Completed,
            "case {index}: {outcome:?}"
        );
        assert_eq!(agent.messages().last().unwrap().text(), answer);
        assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 0);
        assert!(agent.last_turn_telemetry().review_repair_counts.is_empty());
        assert_eq!(agent.task_recovery.interventions, 0);
    }
}

#[test]
fn scoped_no_edit_carveouts_never_flip_implementation_to_read_only() {
    // Found by a controlled A/B on Multi-SWE-bench: adding "Do NOT modify any
    // existing test files" to an implement-the-fix prompt sent every instance
    // into read-only preflight. A negated edit verb with a *scoped* object is
    // a carve-out, not a no-mutation request — in any phrasing, not just the
    // ones an enumerated list happened to contain.
    for carveout in [
        "Fix this bug. Do NOT modify any existing test files.",
        "Implement the feature but don't touch the existing specs.",
        "Fix the parser without changing the documentation.",
        "Apply the fix; never edit the changelog.",
        "Fix it. Avoid updating any docs.",
        "Write driver.py for the included host.py. Do not rewrite host.py or the oracle. Do not edit bug/ yourself — only talk to host.py.",
        "Fix the bug. Do not edit src/lib.rs.",
        "Implement the feature. Don't rewrite host.py.",
    ] {
        assert_eq!(
            classify_read_only_intent(carveout),
            None,
            "scoped carve-out misread as read-only: {carveout:?}"
        );
    }
    // Global no-mutation requests still classify read-only.
    for global in [
        "review the module read only and do not modify anything",
        "audit this code, do not change any files",
        "assess the design and make no changes",
    ] {
        assert!(
            classify_read_only_intent(global).is_some(),
            "global no-mutation request lost: {global:?}"
        );
    }
    // Descriptive "no changes" in a bug report is evidence about the past,
    // not an instruction (found via the SWE-bench prompt corpus).
    assert_eq!(
        classify_read_only_intent(
            "Fix this: expected ax1 dataLims to stay put since I made no changes to it, \
             but they get replaced by inf"
        ),
        None,
        "descriptive past-tense 'no changes' misread as read-only"
    );
}

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
fn review_repair_modes_map_stable_metadata() {
    let expected = [
        (
            ReviewRepairMode::NoEvidence,
            "review_no_evidence",
            "review_no_evidence_exhausted",
            "inspect_files_before_answering",
            "no_evidence",
        ),
        (
            ReviewRepairMode::ListingOnly,
            "review_listing_only",
            "review_listing_only_exhausted",
            "inspect_one_concrete_file_before_answering",
            "listing",
        ),
        (
            ReviewRepairMode::GenericTemplate,
            "review_generic_template",
            "review_generic_disclaimer_exhausted",
            "produce_concrete_bounded_review",
            "generic",
        ),
        (
            ReviewRepairMode::InspectedDisclaimer,
            "review_inspected_disclaimer",
            "review_generic_disclaimer_exhausted",
            "chat_only_bounded_answer_from_inspected_files",
            "disclaimer",
        ),
        (
            ReviewRepairMode::InspectedDisclaimerChatAttempt,
            "review_inspected_disclaimer_chat_attempt",
            "review_generic_disclaimer_exhausted",
            "chat_only_bounded_answer_from_inspected_files",
            "disclaimer_chat",
        ),
        (
            ReviewRepairMode::ConcreteAnswer,
            "review_concrete_answer",
            "review_concrete_answer_exhausted",
            "cite_findings_plus_limits",
            "concrete",
        ),
        (
            ReviewRepairMode::ReadAfterSearch,
            "review_read_after_search",
            "review_read_after_search_exhausted",
            "read_one_matching_file_before_answering",
            "read_after_search",
        ),
        (
            ReviewRepairMode::SecurityBroadSearch,
            "review_security_broad_search",
            "review_security_broad_search_exhausted",
            "search_required_security_patterns_before_answering",
            "security_broad",
        ),
        (
            ReviewRepairMode::SecurityScope,
            "review_security_scope",
            "review_security_scope_exhausted",
            "bound_security_claims_to_inspected_evidence",
            "security_scope",
        ),
        (
            ReviewRepairMode::GapSearchOverclaim,
            "review_gap_search_overclaim",
            "review_gap_search_overclaim_exhausted",
            "cite_search_matches_plus_limits",
            "gap_overclaim",
        ),
        (
            ReviewRepairMode::SprawlForceAnswer,
            "review_sprawl_force_answer",
            "review_sprawl_force_answer_exhausted",
            "chat_only_bounded_answer_from_inspected_files",
            "sprawl_force",
        ),
    ];

    assert_eq!(ReviewRepairMode::ALL.len(), expected.len());
    for (mode, key, exhaustion, required_next, compact) in expected {
        assert!(ReviewRepairMode::ALL.contains(&mode));
        assert_eq!(mode.key(), key);
        assert_eq!(mode.exhaustion_key(), exhaustion);
        assert_eq!(mode.required_next(), required_next);
        assert_eq!(mode.compact_label(), compact);
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
fn read_only_reviews_ignore_inspection_count_language() {
    let prompt = "Review only crates/hi-ai/src/openai/request.rs and crates/hi-ai/src/openai/stream.rs for one concrete bug. Do not edit files.";
    let guarded = read_only_turn_prompt(
        "Review this codebase. Use at most 4 file inspections.",
        ReviewIntent::Review,
    );
    assert!(!guarded.contains("inspection cap"));
    assert!(guarded.contains("Continue inspecting whenever additional evidence is relevant"));
    let exact = read_only_turn_prompt(prompt, ReviewIntent::Review);
    assert!(exact.contains("bounded exact-file review"));
}

#[test]
fn read_only_inspection_tools_include_context_efficient_discovery() {
    for name in [
        "read",
        "list",
        "grep",
        "glob",
        "explore",
        "repo_map",
        "find_symbol",
    ] {
        assert!(
            is_read_only_inspection_tool(name),
            "{name} must count as inspection for sprawl"
        );
    }
    assert!(!is_read_only_inspection_tool("write"));
    assert!(!is_read_only_inspection_tool("bash"));
}

#[test]
fn build_macro_classifies_as_implementation_without_stealing_discussion() {
    let build_macro = command::expand_prompt_macro("/build gpu training TUI calculator").unwrap();
    let intent = classify_implementation_intent(&build_macro).expect("implementation prompt");
    assert!(intent.tui);

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
        ImplementationIntent { tui: true },
    );
    assert!(prompt.contains("Ratatui"));
    assert!(prompt.contains("cargo init --bin ."));
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
fn explicit_benchmark_implementation_prompt_enters_implementation_mode() {
    let prompt = "Implementation task. You are explicitly allowed and expected to edit files in this disposable benchmark workspace, apply patches, and run the verification command. Do not treat this as a read-only review.\n\nA tiny Rust task used to smoke-test PipeBench against the hi coding-agent harness.\n\nImplement `pub fn add(a: i32, b: i32) -> i32` in src/lib.rs.\nKeep the existing test passing. Do not change the test.";

    assert!(
        classify_implementation_intent(prompt).is_some(),
        "explicit benchmark implementation prompt should use implementation steering"
    );
    assert_eq!(
        classify_read_only_intent(prompt),
        None,
        "do not treat as read-only review must not become a read-only guard"
    );
}

#[test]
fn explicit_no_mutation_still_blocks_implementation_mode() {
    let prompt =
        "Implementation task, but do not edit files; inspect only and explain what would change.";

    assert_eq!(classify_implementation_intent(prompt), None);
    assert_eq!(
        classify_read_only_intent(prompt),
        Some(ReviewIntent::Review)
    );
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

#[tokio::test]
async fn final_verifier_owns_post_edit_validation_without_an_extra_model_round() {
    let mut cfg = config();
    let path = cfg.paths.workspace_root.join("answer.txt");
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![crate::VerifyStage::new(
        "check",
        "python3 -c 'from pathlib import Path; assert Path(\"answer.txt\").read_text() == \"x\"'",
    )]);
    let responses = vec![
        completion(vec![Content::Text("I will implement it.".into())], 1, 1),
        write_completion(path.to_str().unwrap()),
        completion(
            vec![Content::Text("Implemented the requested change.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecordingUi::default();
    let outcome = agent
        .run_turn("/build a small CLI project tracker", &mut ui)
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed, "{:?}", ui.statuses);
    assert_eq!(outcome.verification, crate::VerificationStatus::Passed);
    assert_eq!(agent.last_turn_telemetry().model_requests, 3);
    assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 1);
    assert!(!ui.statuses.iter().any(|s| s.contains("without validation")));
}

#[tokio::test]
async fn generic_chat_completion_is_hidden_and_retried_for_a_real_answer() {
    let workspace = IsolatedWorkspace::new("generic-chat-completion-retry");
    let provider = StreamingCanned(Mutex::new(vec![
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Yes. The project has a web UI under `web/`.".into(),
            )],
            1,
            1,
        ),
    ]));
    let mut agent = Agent::new(std::sync::Arc::new(provider), workspace.config()).unwrap();
    let mut ui = RecUi::default();

    let outcome = agent.run_turn("is there a web UI?", &mut ui).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(agent.last_turn_telemetry().model_requests, 2);
    assert!(
        ui.assistant.contains("project has a web UI"),
        "{}",
        ui.assistant
    );
    assert!(
        !ui.assistant.contains("Completed the requested action"),
        "the rejected streamed placeholder leaked into the UI: {}",
        ui.assistant
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("completion placeholder"))
    );
}

#[tokio::test]
async fn no_change_challenge_keeps_structured_tools_and_accepts_edit() {
    let workspace = IsolatedWorkspace::new("no-change-structured-edit");
    let source = workspace.path("src/main.rs");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    let source_text = source.to_string_lossy();
    let responses = vec![
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        ),
        completion(
            vec![Content::Text("I will edit the file now.".into())],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "write-source".into(),
                name: "write".into(),
                arguments: serde_json::json!({"path": source_text, "content": "fn main() {}\n"})
                    .to_string(),
            }],
            1,
            1,
        ),
        completion(vec![Content::Text("Implemented the app.".into())], 1, 1),
        bash_completion("python3 -c 'assert 2 + 2 == 4'"),
        completion(
            vec![Content::Text(
                "Implemented src/main.rs and validated it successfully.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut cfg = workspace.config();
    cfg.gates.allow_unverified = true;
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("Build a small command-line app.", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(std::fs::read_to_string(&source).unwrap(), "fn main() {}\n");
    assert_eq!(modes.lock().unwrap().get(2), Some(&ToolMode::Auto));
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("plain-text tool fallback")),
        "a valid text answer is not a tool protocol failure: {:?}",
        ui.statuses
    );
    assert_eq!(agent.last_turn_telemetry().no_progress_streak, 0);
}

#[tokio::test]
async fn implementation_with_accepted_answer_settles_without_recap_inference() {
    let path = temp_file("implementation-no-finalize");
    let path_string = path.to_string_lossy().to_string();
    let mut cfg = config();
    cfg.memory.finalize = true;
    cfg.gates.allow_unverified = true;
    let responses = vec![
        write_completion(&path_string),
        completion(vec![Content::Text("Implemented it.".into())], 1, 1),
        completion(vec![Content::Text("Done.".into())], 1, 1),
        completion(vec![Content::Text("Final recap.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecordingUi::default();
    let outcome = agent
        .run_turn("/build a small CLI project tracker", &mut ui)
        .await
        .unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(
        agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("Implemented it."),
        "normal settlement preserves the accepted model answer"
    );
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.stop_reason, TurnStopReason::VerificationUnavailable);
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("incomplete") || status.contains("stalled")),
        "normal settlement must not emit a legacy failure label: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn shell_generated_source_is_a_complete_mutation_without_scaffold_or_xml_repair() {
    let mut cfg = config();
    cfg.gates.verification = crate::VerificationMode::Disabled;
    cfg.gates.allow_unverified = true;
    let source = cfg.paths.workspace_root.join("source.txt");
    let mut agent = agent(
        vec![
            bash_completion(
                "python3 -c 'from pathlib import Path; Path(\"source.txt\").write_text(\"implemented\")'",
            ),
            completion(
                vec![Content::Text(
                    "Created source.txt with the requested content.".into(),
                )],
                1,
                1,
            ),
        ],
        cfg,
    );
    let mut ui = RecordingUi::default();
    let outcome = agent
        .run_turn("create source.txt containing implemented", &mut ui)
        .await
        .unwrap();
    assert_eq!(std::fs::read_to_string(source).unwrap(), "implemented");
    assert_eq!(outcome.status, TurnStatus::Completed, "{:?}", ui.statuses);
    assert_eq!(agent.last_turn_telemetry().model_requests, 2);
    assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 0);
    assert!(
        !agent
            .messages()
            .iter()
            .any(|m| m.text().contains("<tool_call>"))
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
fn guessed_background_handle_is_steered_without_user_facing_status() {
    // The registry records a handle named while it was empty as *guessed*:
    // nothing has ever run under it, so the model invented it. The steer
    // corrects the model with a nudge only — no user-facing status line.
    let registry = hi_tools::BackgroundRegistry::default();
    assert!(registry.poll("ghost_1").is_err());
    let unknown = registry.unknown_handles();
    assert_eq!(unknown.len(), 1);
    assert_eq!(unknown[0].id, "ghost_1");
    assert!(
        unknown[0].registry_was_empty,
        "an empty registry means the id was never real"
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
        inspection_signature("read", r#"{"path":"src/lib.rs","offset":1,"limit":2000}"#,),
        Some("read:src/lib.rs:1:default".into())
    );
    assert_eq!(
        inspection_signature("read", r#"{"path":"src/lib.rs","offset":0,"limit":0}"#,),
        Some("read:src/lib.rs:1:1".into())
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
        inspection_signature("bash_output", r#"{"id":"sh_1"}"#),
        Some("bash_output:sh_1".into())
    );
    assert_eq!(
        inspection_signature("bash_kill", r#"{"id":"sh_1"}"#),
        Some("bash_kill:sh_1".into())
    );
    assert_eq!(
        inspection_signature("bash", r#"{"command":"ls"}"#),
        Some("bash:inspection:ls".into())
    );
    let paged_listing = r#"{"command":"for f in blog_posts/txt/*.txt; do echo \"=== $f ===\"; head -2 \"$f\" | tr '\\n' ' '; echo; done | sed -n '20,46p'"}"#;
    assert!(
        inspection_signature("bash", paged_listing)
            .is_some_and(|signature| signature.starts_with("bash:inspection:for f in "))
    );
    // Mutating/unclassified tools have no signature.
    assert_eq!(inspection_signature("write", r#"{"path":"x"}"#), None);
    assert_eq!(
        inspection_signature("bash", r#"{"command":"rm -f generated.txt"}"#),
        None
    );
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
    let manifest = temp_workspace_path("Cargo.toml");
    std::fs::write(
        &manifest,
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    let manifest_path = manifest.to_string_lossy().to_string();
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
                arguments: serde_json::json!({"path": manifest_path}).to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(format!(
                "Findings:\n- {manifest_path} was inspected as security review context.\n\nLimits:\n- Limited to inspected evidence."
            ))],
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
    let mut agent = Agent::new(std::sync::Arc::new(provider), config()).unwrap();
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
async fn read_only_status_preflight_seeds_first_request_with_evidence() {
    let manifest = temp_workspace_path("Cargo.toml");
    let readme = temp_workspace_path("README.md");
    std::fs::write(&manifest, "[workspace]\nmembers = []\n").unwrap();
    std::fs::write(&readme, "# Fixture\n").unwrap();
    let mut cfg = config();
    cfg.gates.read_only_preflight = true;
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
    let path = temp_file("ux-json-implementation");
    let mut cfg = config();
    cfg.gates.read_only_preflight = true;
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
    let calls =
        read_only_preflight_initial_calls_in(std::path::Path::new("."), ReviewIntent::Security);
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
async fn repeated_completion_placeholder_gets_one_retry_without_evidence_failure() {
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
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert_eq!(agent.task_recovery().interventions, 1);
    assert!(!agent.task_recovery().exhausted);
    assert!(!ui.assistant.contains("Automatic recovery stopped."));
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn read_only_review_repeat_exhaustion_returns_typed_no_progress() {
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
    let mut responses = vec![
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
                arguments: grep_args.clone(),
            }],
            1,
            1,
        ),
    ];
    for retry in 0..config().loop_limits.max_empty_retries {
        responses.push(completion(
            vec![Content::ToolCall {
                id: format!("grep-final-retry-{retry}"),
                name: "grep".into(),
                arguments: grep_args.clone(),
            }],
            1,
            1,
        ));
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(
            "review for security issues or unsafe unwraps. then disucss only",
            &mut ui,
        )
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert!(ui.assistant.contains("Automatic recovery stopped."));
    let _ = std::fs::remove_file(inspected_path);
}

#[tokio::test]
async fn read_only_review_repeated_search_returns_typed_no_progress() {
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
    let mut responses = vec![grep_call(), grep_call(), grep_call(), grep_call()];
    for _ in 0..config().loop_limits.max_empty_retries {
        responses.push(grep_call());
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();

    let outcome = agent
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
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert!(ui.assistant.contains("Automatic recovery stopped."));
}

#[tokio::test]
async fn repeated_inspection_challenges_allow_a_plain_text_explanation() {
    let workspace = IsolatedWorkspace::new("repeat-inspection-explanation");
    let mut responses = Vec::new();
    for n in 0..3 {
        let path = workspace.path(format!("empty-{n}"));
        std::fs::create_dir(&path).unwrap();
        responses.push(completion(
            vec![Content::ToolCall {
                id: format!("list-{n}"),
                name: "list".into(),
                arguments: serde_json::json!({"path": path}).to_string(),
            }],
            1,
            1,
        ));
    }
    responses.push(completion(
        vec![Content::Text(
            "No file changes are needed because the requested empty directories already exist."
                .into(),
        )],
        1,
        1,
    ));
    let mut cfg = workspace.config();
    cfg.loop_limits.max_repeat_nudges = 0;
    cfg.gates.allow_unverified = true;
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("Build three empty directories if missing", &mut ui)
        .await
        .unwrap();
    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "{outcome:?}: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .filter(|s| {
                s.contains("requesting an edit or explanation")
                    || s.contains("inspection has not produced a file change")
            })
            .count()
            >= 1,
        "expected an implementation challenge before accepting the explanation: {:?}",
        ui.statuses
    );
    assert!(
        modes
            .lock()
            .unwrap()
            .iter()
            .all(|mode| *mode == ToolMode::Auto)
    );
    assert!(
        !agent
            .messages()
            .iter()
            .flat_map(|m| &m.content)
            .any(|c| matches!(c, Content::Text(text) if text.contains("<tool_call>")))
    );
}

#[tokio::test]
async fn successful_edits_between_rereads_do_not_force_an_incomplete_closeout() {
    let workspace = IsolatedWorkspace::new("chat-productive-rereads");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"productive_rereads\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    std::fs::write(
        workspace.path("src/lib.rs"),
        "pub fn answer() -> u32 { 0 }\n",
    )
    .unwrap();
    std::fs::write(
        workspace.path("context.txt"),
        "Neighboring API used by every implementation step.\n",
    )
    .unwrap();
    assert!(
        tokio::process::Command::new("cargo")
            .args(["generate-lockfile", "--offline"])
            .current_dir(workspace.path(""))
            .status()
            .await
            .unwrap()
            .success()
    );
    let read_context = || {
        completion(
            vec![Content::ToolCall {
                id: "read-context".into(),
                name: "read".into(),
                arguments: r#"{"path":"context.txt"}"#.into(),
            }],
            1,
            1,
        )
    };
    let mut responses = vec![read_context()];
    for step in 1..=5 {
        responses.push(completion(vec![Content::ToolCall {
            id: format!("edit-{step}"), name: "edit".into(),
            arguments: serde_json::json!({"path":"src/lib.rs", "old_string": format!("pub fn answer() -> u32 {{ {} }}", step - 1), "new_string": format!("pub fn answer() -> u32 {{ {step} }}")}).to_string(),
        }], 1, 1));
        responses.push(read_context());
    }
    responses.push(completion(
        vec![Content::Text(
            "Implemented all five changes. Cargo check passed for the final source.".into(),
        )],
        1,
        1,
    ));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut cfg = workspace.config();
    cfg.gates.lsp_mode = LspMode::Off;
    cfg.gates.verification = VerificationMode::Explicit(vec![
        VerifyStage::new("check", "cargo check --quiet"),
        VerifyStage::new("test", "cargo test --quiet"),
    ]);
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("fix and build all of that", &mut ui)
        .await
        .unwrap();
    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "{outcome:?}: {:?}",
        ui.statuses
    );
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn answer() -> u32 { 5 }\n"
    );
    assert!(!agent.task_recovery().exhausted);
    assert_eq!(
        agent.task_recovery().interventions,
        0,
        "post-edit context refreshes must not consume recovery"
    );
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 0);
    assert!(
        modes
            .lock()
            .unwrap()
            .iter()
            .all(|mode| *mode != ToolMode::ChatOnly)
    );
}

#[tokio::test]
async fn post_edit_context_refresh_does_not_spend_the_stall_budget() {
    let workspace = IsolatedWorkspace::new("chat-post-edit-context");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"context_refresh\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    std::fs::write(
        workspace.path("src/lib.rs"),
        "pub mod server; pub mod db; pub fn answer() -> u32 { 0 }\n",
    )
    .unwrap();
    std::fs::write(
        workspace.path("src/server.rs"),
        "pub const LIMIT: u32 = 1;\n",
    )
    .unwrap();
    std::fs::write(workspace.path("src/db.rs"), "pub const LIMIT: u32 = 2;\n").unwrap();
    assert!(
        tokio::process::Command::new("cargo")
            .args(["generate-lockfile", "--offline"])
            .current_dir(workspace.path(""))
            .status()
            .await
            .unwrap()
            .success()
    );
    let read = |path: &str| {
        completion(
            vec![Content::ToolCall {
                id: "read-context".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path":path}).to_string(),
            }],
            1,
            1,
        )
    };
    let edit = |path: &str, old: &str, new: &str| {
        completion(
            vec![Content::ToolCall {
                id: "edit-context".into(),
                name: "edit".into(),
                arguments: serde_json::json!({"path":path,"old_string":old,"new_string":new})
                    .to_string(),
            }],
            1,
            1,
        )
    };
    // The live session: inspect the project, land two green edits, refresh
    // neighboring context, repeat one read, then continue the remaining fix.
    let responses = vec![
        read("Cargo.toml"),
        read("src/lib.rs"),
        read("src/server.rs"),
        read("src/db.rs"),
        edit("src/lib.rs", "{ 0 }", "{ 1 }"),
        read("src/lib.rs"),
        edit("src/lib.rs", "{ 1 }", "{ 2 }"),
        read("Cargo.toml"),
        read("src/server.rs"),
        read("src/db.rs"),
        read("src/db.rs"),
        edit("src/server.rs", "= 1;", "= 3;"),
        completion(
            vec![Content::Text(
                "Updated the parser and server limits; all requested changes are complete.".into(),
            )],
            1,
            1,
        ),
    ];
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut cfg = workspace.config();
    cfg.gates.lsp_mode = LspMode::Off;
    cfg.gates.verification = VerificationMode::Explicit(vec![
        VerifyStage::new("check", "cargo check --quiet"),
        VerifyStage::new("test", "cargo test --quiet"),
    ]);
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("fix and build all of that", &mut ui)
        .await
        .unwrap();
    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "{outcome:?}; {:?}",
        ui.statuses
    );
    assert_eq!(
        outcome.verification,
        VerificationStatus::Passed,
        "{outcome:?}; {:?}",
        ui.statuses
    );
    assert!(
        std::fs::read_to_string(workspace.path("src/server.rs"))
            .unwrap()
            .contains("= 3;")
    );
    assert!(!agent.task_recovery().exhausted);
    assert_eq!(agent.last_turn_telemetry().forced_final_answer_attempts, 0);
    assert!(
        modes
            .lock()
            .unwrap()
            .iter()
            .all(|mode| *mode != ToolMode::ChatOnly)
    );
    assert!(!ui.assistant.contains("Automatic recovery stopped."));
}
