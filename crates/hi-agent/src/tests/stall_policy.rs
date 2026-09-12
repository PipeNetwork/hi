use super::common::{
    IsolatedWorkspace, ProviderStep, RecUi, bash_completion, completion, echo_call, scripted_agent,
    write_content_completion,
};
use super::*;
use hi_ai::Content;

fn read_completion(path: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: "r".into(),
            name: "read".into(),
            arguments: format!(r#"{{"path":{path:?}}}"#),
        }],
        1,
        1,
    )
}

#[tokio::test]
async fn identical_reads_after_write_end_as_stationarity_not_no_progress() {
    let workspace = IsolatedWorkspace::new("stationarity-after-write");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn f() {}\n").unwrap();
    let path = workspace.path("src/lib.rs").to_string_lossy().into_owned();
    let mut steps = vec![ProviderStep::Completion(write_content_completion(
        &path,
        "pub fn f() { 1 }\n",
    ))];
    for _ in 0..crate::steering::MAX_CONSECUTIVE_IDENTICAL_PROBLEMATIC_TOOL_CALLS {
        steps.push(ProviderStep::Completion(read_completion(&path)));
    }
    steps.push(ProviderStep::Completion(completion(
        vec![Content::Text("Stopped repeating the read.".into())],
        1,
        1,
    )));
    let mut cfg = workspace.config();
    cfg.loop_limits.max_repeat_nudges = 16;
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let (mut agent, _) = scripted_agent(steps, cfg);
    let mut ui = RecUi::default();
    let outcome = agent.run_turn("build all of that", &mut ui).await.unwrap();
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "stationarity after edits must not be no_progress: {outcome:?}; {:?}",
        ui.statuses
    );
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn f() { 1 }\n"
    );
}

fn last_request_text(request: &[hi_ai::Message]) -> String {
    request.last().map(hi_ai::Message::text).unwrap_or_default()
}

#[tokio::test]
async fn false_completion_claim_without_tool_evidence_injects_laziness() {
    let workspace = IsolatedWorkspace::new("false-completion-laziness");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn f() {}\n").unwrap();
    let path = workspace.path("src/lib.rs").to_string_lossy().into_owned();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "SUCCESS: cargo test --quiet is all green and production-ready.".into(),
                )],
                1,
                1,
            )),
            ProviderStep::Completion(write_content_completion(&path, "pub fn f() { 1 }\n")),
            ProviderStep::Completion(completion(
                vec![Content::Text("Wrote the function body.".into())],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn("build all of that", &mut ui).await.unwrap();
    let sent = requests.lock().unwrap();
    assert!(
        sent.len() >= 2,
        "laziness continue must send another provider request: {}",
        sent.len()
    );
    let follow_up = last_request_text(&sent[1]);
    assert_eq!(
        sent[1].last().map(|message| message.role),
        Some(hi_ai::Role::User),
        "laziness nudge must be the last message on the next request"
    );
    assert!(
        follow_up.contains("[hi:nudge:laziness]"),
        "next provider request must end on the laziness user nudge, got: {follow_up}"
    );
    assert!(
        !follow_up.contains("SUCCESS: cargo test"),
        "false-completion claim must not be folded into the laziness user message: {follow_up}"
    );
    assert_ne!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert_ne!(outcome.stop_reason, TurnStopReason::InfrastructureFailure);
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn f() { 1 }\n"
    );
}

#[tokio::test]
async fn write_then_false_completion_without_tests_injects_laziness() {
    let workspace = IsolatedWorkspace::new("write-then-false-completion");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn f() {}\n").unwrap();
    let path = workspace.path("src/lib.rs").to_string_lossy().into_owned();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_content_completion(&path, "pub fn f() { 1 }\n")),
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "SUCCESS: cargo test --quiet is all green and production-ready.".into(),
                )],
                1,
                1,
            )),
            ProviderStep::Completion(super::common::bash_completion(
                "python3 -c 'assert 2 + 2 == 4'",
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "Edited src/lib.rs and ran a local check.".into(),
                )],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn("build all of that", &mut ui).await.unwrap();
    let sent = requests.lock().unwrap();
    assert!(
        sent.len() >= 3,
        "write then false completion must continue after laziness: {}",
        sent.len()
    );
    let follow_up = last_request_text(&sent[2]);
    assert_eq!(
        sent[2].last().map(|message| message.role),
        Some(hi_ai::Role::User),
        "laziness nudge must be the last message after write-then-false-completion"
    );
    assert!(
        follow_up.contains("[hi:nudge:laziness]"),
        "next provider request after the false recap must be the laziness user nudge, got: {follow_up}"
    );
    assert_ne!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn f() { 1 }\n"
    );
}

#[tokio::test]
async fn todo_gate_continues_then_falls_through_without_keep_working() {
    let mut cfg = super::common::config();
    cfg.loop_limits.max_silent_continues = 1;
    cfg.loop_limits.max_keep_working = 2;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    let plan_call = |id: &str, s1: &str, s2: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"Review a","status":"{s1}"}},{{"title":"Review b","status":"{s2}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(plan_call("p1", "active", "pending")),
            ProviderStep::Completion(completion(
                vec![Content::Text("Step 1 recap.".into())],
                1,
                1,
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text("Still recapping leftover work.".into())],
                1,
                1,
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text("Would have been keep-working.".into())],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("implement the remaining plan step", &mut ui)
        .await
        .unwrap();
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| message.text().contains("[hi:nudge:todogate]")),
        "TodoGate reminder missing: {:?}",
        agent
            .messages()
            .iter()
            .map(|m| m.text())
            .collect::<Vec<_>>()
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("still working")),
        "keep-working must not stack on TodoGate: {:?}",
        ui.statuses
    );
    assert_ne!(outcome.stop_reason, TurnStopReason::InfrastructureFailure);
    assert!(agent.plan_incomplete());
}

fn request_blob(request: &[hi_ai::Message]) -> String {
    request.iter().map(hi_ai::Message::text).collect()
}

#[tokio::test]
async fn bulky_grep_is_not_packed_into_an_obs_recall_loop() {
    let workspace = IsolatedWorkspace::new("stall-grep-not-packed");
    let body = format!("offline status {}\n", "x".repeat(80)).repeat(80);
    std::fs::write(workspace.path("web.rs"), &body).unwrap();
    let mut cfg = workspace.config();
    cfg.memory.observation_pack = true;
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let mut steps = vec![ProviderStep::Completion(completion(
        vec![Content::ToolCall {
            id: "g".into(),
            name: "grep".into(),
            arguments: r#"{"pattern":"offline"}"#.into(),
        }],
        1,
        1,
    ))];
    for _ in 0..3 {
        steps.push(ProviderStep::Completion(echo_call()));
    }
    steps.push(ProviderStep::Completion(write_content_completion(
        "web.rs",
        "offline status fixed\n",
    )));
    for _ in 0..8 {
        steps.push(ProviderStep::Completion(completion(
            vec![Content::Text("Inspected the offline status.".into())],
            1,
            1,
        )));
    }
    let (mut agent, requests) = scripted_agent(steps, cfg);
    agent
        .run_turn(
            "it shows it offline and i don't see how to make an account. fix",
            &mut RecUi::default(),
        )
        .await
        .unwrap();
    let sent = requests.lock().unwrap();
    assert!(
        sent.len() > 3,
        "need later requests after the grep: {}",
        sent.len()
    );
    for (index, request) in sent.iter().enumerate().skip(1) {
        let blob = request_blob(request);
        assert!(
            !blob.contains("id: obs_"),
            "request {index} packed grep into obs_recall: {blob}"
        );
        assert!(
            !request
                .iter()
                .any(|message| message.text().contains("obs_recall")),
            "request {index} advertised paging for a discovery result"
        );
    }
}

#[tokio::test]
async fn obs_recall_only_round_nudges_an_implementation_instead_of_paging() {
    let workspace = IsolatedWorkspace::new("stall-obs-recall-nudge");
    std::fs::write(workspace.path("index.html"), "<span>offline</span>\n").unwrap();
    let mut cfg = workspace.config();
    cfg.memory.observation_pack = true;
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(
                vec![Content::ToolCall {
                    id: "or".into(),
                    name: "obs_recall".into(),
                    arguments: r#"{"id":"obs_aaaaaaaaaaaaaaaaaaaaaaaa","offset":0}"#.into(),
                }],
                1,
                1,
            )),
            ProviderStep::Completion(write_content_completion(
                "index.html",
                "<button>Create account</button>\n",
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text("Added a Create account button.".into())],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn(
            "it shows it offline and i don't see how to make an account. fix",
            &mut ui,
        )
        .await
        .unwrap();
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("observation paging is not an implementation")),
        "obs_recall-only rounds must be steered to edit: {:?}",
        ui.statuses
    );
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "paging must not settle as no_progress: {outcome:?}; {:?}",
        ui.statuses
    );
    assert!(
        std::fs::read_to_string(workspace.path("index.html"))
            .unwrap()
            .contains("Create account"),
        "implementation must land after the paging nudge"
    );
}

fn read_named(id: &str, path: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: id.into(),
            name: "read".into(),
            arguments: format!(r#"{{"path":{path:?}}}"#),
        }],
        1,
        1,
    )
}

#[tokio::test]
async fn unique_file_inspection_sprawl_on_a_fix_prompt_stops() {
    let workspace = IsolatedWorkspace::new("stall-unique-read-sprawl");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    let cfg = stall_cfg(&workspace);
    let mut steps = Vec::new();
    for index in 0..12 {
        let relative = format!("src/context-{index}.rs");
        std::fs::write(
            workspace.path(&relative),
            format!("pub fn context_{index}() {{}}\n"),
        )
        .unwrap();
        steps.push(ProviderStep::Completion(read_named(
            &format!("r{index}"),
            &relative,
        )));
    }
    steps.push(ProviderStep::Completion(write_content_completion(
        "src/context-0.rs",
        "pub fn context_0() { 1 }\n",
    )));
    steps.push(ProviderStep::Completion(completion(
        vec![Content::Text("Edited after inspecting every file.".into())],
        1,
        1,
    )));
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let mut ui = RecUi::default();
    let outcome = agent.run_turn(FIX_PROMPT, &mut ui).await.unwrap();
    let sent = requests.lock().unwrap();
    assert!(
        sent.len() < 16,
        "unique-file reads must not keep requesting: {} {:?}",
        sent.len(),
        ui.statuses
    );
    assert!(
        ui.statuses.iter().any(|status| status
            .contains("inspection has not produced a file change")
            || status.contains("kept inspecting after the edit challenge")),
        "unique-file sprawl must demand an edit: {:?}",
        ui.statuses
    );
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "unique-file inspection until leftover budget is a stall: {outcome:?}; {:?}",
        ui.statuses
    );
    assert_eq!(outcome.status, TurnStatus::Failed, "{outcome:?}");
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/context-0.rs")).unwrap(),
        "pub fn context_0() {}\n",
        "the late write must not land after unique-read exhaustion"
    );
}

#[tokio::test]
async fn review_and_fix_insufficient_evidence_dump_is_not_a_todo_continue() {
    let workspace = IsolatedWorkspace::new("stall-review-fix-dump");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    let cfg = stall_cfg(&workspace);
    let files = ["Cargo.toml", "src/main.rs", "src/server.rs", "src/state.rs"];
    let mut steps = Vec::new();
    for (index, relative) in files.iter().enumerate() {
        if relative.starts_with("src/") {
            std::fs::write(
                workspace.path(relative),
                format!("pub fn f{index}() {{}}\n"),
            )
            .unwrap();
        } else {
            std::fs::write(workspace.path(relative), "[package]\nname = \"app\"\n").unwrap();
        }
        steps.push(ProviderStep::Completion(read_named(
            &format!("r{index}"),
            relative,
        )));
    }
    steps.push(ProviderStep::Completion(insufficient_evidence_dump()));
    steps.push(ProviderStep::Completion(insufficient_evidence_dump()));
    let (mut agent, _) = scripted_agent(steps, cfg);
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("review for any major issues and fix", &mut ui)
        .await
        .unwrap();
    let withhold_at = ui.statuses.iter().position(|status| {
        status.contains("withholding inspection tools until a file change lands")
    });
    let dump_at = ui
        .statuses
        .iter()
        .position(|status| status.contains("review-shaped wrap-up is not an implementation"));
    assert!(
        withhold_at.is_none() || dump_at.is_some_and(|dump| withhold_at.unwrap() >= dump),
        "unique-file review-and-fix must not withhold inspection before the dump is rejected: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("review-shaped wrap-up is not an implementation")),
        "the insufficient-evidence dump must be rejected: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("outstanding todos remain")),
        "todo-gate must not keep the dump alive: {:?}",
        ui.statuses
    );
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "this transcript must not look like leftover inspection: {outcome:?}; {:?}",
        ui.statuses
    );
    assert_eq!(outcome.status, TurnStatus::Failed, "{outcome:?}");
    assert!(
        !ui.assistant.contains("Insufficient evidence"),
        "the dump must not be the user-visible answer: {}",
        ui.assistant
    );
}

#[tokio::test]
async fn three_inspection_rounds_on_a_fix_prompt_nudge_an_edit() {
    let workspace = IsolatedWorkspace::new("stall-inspect-then-edit");
    std::fs::write(workspace.path("index.html"), "<span>offline</span>\n").unwrap();
    std::fs::write(workspace.path("web.rs"), "pub fn page() {}\n").unwrap();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let html = workspace.path("index.html").to_string_lossy().into_owned();
    let web = workspace.path("web.rs").to_string_lossy().into_owned();
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(read_named("r1", &html)),
            ProviderStep::Completion(read_named("r2", &web)),
            ProviderStep::Completion(read_named("r3", &html)),
            ProviderStep::Completion(read_named("r4", &html)),
            ProviderStep::Completion(write_content_completion(
                "index.html",
                "<button>Create account</button>\n",
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text("Added a Create account button.".into())],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn(
            "it shows it offline and i don't see how to make an account. fix",
            &mut ui,
        )
        .await
        .unwrap();
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("inspection has not produced a file change")),
        "three inspection rounds must demand an edit: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("withholding inspection tools until a file change lands")),
        "discovery tools must be withheld after the edit nudge: {:?}",
        ui.statuses
    );
    let html_reads = ui
        .tool_results
        .iter()
        .filter(|(name, result)| name == "read" && result.contains("offline"))
        .count();
    assert_eq!(
        html_reads, 2,
        "the post-withhold read must not succeed: {:?}",
        ui.tool_results
    );
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}"
    );
    assert!(
        std::fs::read_to_string(workspace.path("index.html"))
            .unwrap()
            .contains("Create account")
    );
}

#[tokio::test]
async fn compiler_failure_in_a_successful_shell_forces_an_edit() {
    let workspace = IsolatedWorkspace::new("stall-compiler-in-success");
    std::fs::write(workspace.path("web.rs"), "fn page() {}\n").unwrap();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(
                vec![Content::ToolCall {
                    id: "chk".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({
                        "command": "printf 'error[E0425]: cannot find function register\\n'; true"
                    })
                    .to_string(),
                }],
                1,
                1,
            )),
            ProviderStep::Completion(write_content_completion("web.rs", "fn register() {}\n")),
            ProviderStep::Completion(completion(
                vec![Content::Text("Defined register().".into())],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn(
            "it shows it offline and i don't see how to make an account. fix",
            &mut ui,
        )
        .await
        .unwrap();
    assert!(
        ui.statuses
            .iter()
            .any(|status| { status.contains("compiler or test failure is already in context") }),
        "exit-0 cargo output with rustc errors must demand an edit: {:?}",
        ui.statuses
    );
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}"
    );
    assert!(
        std::fs::read_to_string(workspace.path("web.rs"))
            .unwrap()
            .contains("fn register()")
    );
}

fn insufficient_evidence_dump() -> Completion {
    completion(
        vec![Content::Text(
            "Insufficient evidence: I inspected src/web/index.html, src/web.rs, src/state.rs, REDACTED_SECRET, but that evidence is not enough to make concrete gap, roadmap, or status claims for this request. A useful answer would need targeted reads or searches of the owning implementation modules, tests, and validation surface before recommending build-next work.\n\
Evidence summary: files_read=3, searches=0, listings=1, commands=4, failed_tools=1, inspected_paths=src/web/index.html, src/web.rs, src/state.rs, REDACTED_SECRET.\n\
Insufficient evidence cause: inaccessible_path"
                .into(),
        )],
        1,
        1,
    )
}

#[tokio::test]
async fn bash_sed_after_inspection_nudge_is_withheld_until_an_edit() {
    let workspace = IsolatedWorkspace::new("stall-bash-sed-withhold");
    std::fs::write(workspace.path("index.html"), "<span>offline</span>\n").unwrap();
    std::fs::write(workspace.path("web.rs"), "pub fn page() {}\n").unwrap();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let html = workspace.path("index.html").to_string_lossy().into_owned();
    let web = workspace.path("web.rs").to_string_lossy().into_owned();
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(read_named("r1", &html)),
            ProviderStep::Completion(read_named("r2", &web)),
            ProviderStep::Completion(read_named("r3", &html)),
            ProviderStep::Completion(bash_completion("sed -n '1,20p' index.html")),
            ProviderStep::Completion(write_content_completion(
                "index.html",
                "<button>Create account</button>\n",
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text("Added a Create account button.".into())],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn(
            "it shows it offline and i don't see how to make an account. fix",
            &mut ui,
        )
        .await
        .unwrap();
    assert!(
        ui.tool_results.iter().any(|(name, result)| {
            name == "bash" && result.contains("withheld until a file change")
        }),
        "bash sed after the edit nudge must not dump the file: {:?}",
        ui.tool_results
    );
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}"
    );
    assert!(
        std::fs::read_to_string(workspace.path("index.html"))
            .unwrap()
            .contains("Create account")
    );
}

#[tokio::test]
async fn trivial_edit_then_insufficient_evidence_review_is_not_verified_success() {
    let workspace = IsolatedWorkspace::new("stall-review-dump-after-button");
    std::fs::write(workspace.path("index.html"), "<span>offline</span>\n").unwrap();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(write_content_completion(
                "index.html",
                "<button>Create account</button>\n",
            )),
            ProviderStep::Completion(bash_completion("sed -n '1,20p' index.html")),
            ProviderStep::Completion(insufficient_evidence_dump()),
            ProviderStep::Completion(insufficient_evidence_dump()),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn(
            "it shows it offline and i don't see how to make an account. fix",
            &mut ui,
        )
        .await
        .unwrap();
    assert!(
        ui.statuses
            .iter()
            .any(|status| { status.contains("review-shaped wrap-up is not an implementation") }),
        "the gap/roadmap dump after a one-line edit must be rejected: {:?}",
        ui.statuses
    );
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "this transcript must not settle as verified success: {outcome:?}; {:?}",
        ui.statuses
    );
    assert_eq!(outcome.status, TurnStatus::Failed, "{outcome:?}");
    assert!(
        !ui.assistant.contains("Insufficient evidence"),
        "the review dump must not be the user-visible answer: {}",
        ui.assistant
    );
}

#[tokio::test]
async fn insufficient_evidence_nudge_then_a_real_fix_recap_can_complete() {
    let workspace = IsolatedWorkspace::new("stall-review-dump-then-fix");
    std::fs::write(workspace.path("index.html"), "<span>offline</span>\n").unwrap();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(write_content_completion(
                "index.html",
                "<button>Create account</button>\n",
            )),
            ProviderStep::Completion(insufficient_evidence_dump()),
            ProviderStep::Completion(write_content_completion(
                "index.html",
                "<button id=\"register\">Create account</button>\n<script>fetch('/register',{method:'POST'})</script>\n",
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "Wired the Create account button to POST /register.".into(),
                )],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn(
            "it shows it offline and i don't see how to make an account. fix",
            &mut ui,
        )
        .await
        .unwrap();
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}"
    );
    assert_eq!(outcome.status, TurnStatus::Completed, "{outcome:?}");
    assert!(
        std::fs::read_to_string(workspace.path("index.html"))
            .unwrap()
            .contains("/register")
    );
}

const FIX_PROMPT: &str = "it shows it offline and i don't see how to make an account. fix";

const LOGIN_HTML: &str = r#"<!DOCTYPE html>
<form>
<label>password <input id="password" class="secret" type="password" autocomplete="current-password"></label>
<style>input.secret { width:9rem; }</style>
<span id="status">offline</span>
</form>
"#;

fn stall_cfg(workspace: &IsolatedWorkspace) -> AgentConfig {
    let mut cfg = workspace.config();
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    cfg
}

/// Production leftover budgets. Canned stall tests that claim to catch the
/// TUI leftover closeout must not disable `max_keep_working`.
fn prod_recovery_cfg(workspace: &IsolatedWorkspace) -> AgentConfig {
    let mut cfg = stall_cfg(workspace);
    cfg.loop_limits.max_keep_working = crate::MAX_KEEP_WORKING;
    cfg.loop_limits.max_recovery_interventions = crate::DEFAULT_RECOVERY_INTERVENTIONS;
    cfg.loop_limits.max_silent_continues = crate::MAX_SILENT_CONTINUES;
    cfg
}

const REVIEW_FIX_PROMPT: &str = "review for any major issues and fix";

const DECLINE_RECAP: &str =
    "No file changes are needed because cargo test is green and I found no correctness bugs.";

const LEFTOVER_EMPTY_CLOSEOUT: &str = "Automatic recovery stopped. No file changes were made.";

fn list_named(id: &str, path: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: id.into(),
            name: "list".into(),
            arguments: format!(r#"{{"path":{path:?}}}"#),
        }],
        1,
        1,
    )
}

fn python_assert() -> Completion {
    bash_completion("python3 -c 'assert 2 + 2 == 4'")
}

fn decline_recap() -> Completion {
    completion(vec![Content::Text(DECLINE_RECAP.into())], 1, 1)
}

fn seed_unique_sources(workspace: &IsolatedWorkspace, count: usize) -> Vec<String> {
    seed_unique_sources_from(workspace, 0, count)
}

fn seed_unique_sources_from(
    workspace: &IsolatedWorkspace,
    start: usize,
    count: usize,
) -> Vec<String> {
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    (start..start + count)
        .map(|index| {
            let relative = format!("src/mod_{index}.rs");
            std::fs::write(
                workspace.path(&relative),
                format!("pub fn f{index}() {{}}\n"),
            )
            .unwrap();
            relative
        })
        .collect()
}

fn unique_read_steps(paths: &[String]) -> Vec<ProviderStep> {
    unique_read_steps_prefixed(paths, "r")
}

fn unique_read_steps_prefixed(paths: &[String], prefix: &str) -> Vec<ProviderStep> {
    paths
        .iter()
        .enumerate()
        .map(|(index, relative)| {
            ProviderStep::Completion(read_named(&format!("{prefix}{index}"), relative))
        })
        .collect()
}

fn batched_unique_reads(paths: &[String]) -> ProviderStep {
    let calls = paths
        .iter()
        .enumerate()
        .map(|(index, relative)| Content::ToolCall {
            id: format!("r{index}"),
            name: "read".into(),
            arguments: format!(r#"{{"path":{relative:?}}}"#),
        })
        .collect();
    ProviderStep::Completion(completion(calls, 1, 1))
}

fn pad_decline_recaps(steps: &mut Vec<ProviderStep>, count: usize) {
    for _ in 0..count {
        steps.push(ProviderStep::Completion(decline_recap()));
    }
}

fn review_fix_inspect_then_tests(paths: &[String], batched: bool) -> Vec<ProviderStep> {
    let mut steps = if batched {
        vec![batched_unique_reads(paths)]
    } else {
        unique_read_steps(paths)
    };
    steps.push(ProviderStep::Completion(list_named("ls", "src")));
    steps.push(ProviderStep::Completion(python_assert()));
    steps
}

fn inspect_then_bash_then_edit(command: &str) -> Vec<ProviderStep> {
    vec![
        ProviderStep::Completion(read_named("r1", "index.html")),
        ProviderStep::Completion(read_named("r2", "web.rs")),
        ProviderStep::Completion(read_named("r3", "index.html")),
        ProviderStep::Completion(bash_completion(command)),
        ProviderStep::Completion(write_content_completion(
            "index.html",
            "<button>Create account</button>\n",
        )),
        ProviderStep::Completion(completion(
            vec![Content::Text("Added a Create account button.".into())],
            1,
            1,
        )),
    ]
}

#[tokio::test]
async fn insufficient_evidence_dump_is_no_progress_even_when_verify_is_green() {
    let workspace = IsolatedWorkspace::new("stall-dump-green-verify");
    std::fs::write(workspace.path("index.html"), LOGIN_HTML).unwrap();
    let mut cfg = stall_cfg(&workspace);
    cfg.gates.allow_unverified = false;
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(write_content_completion(
                "index.html",
                "<button>Create account</button>\n",
            )),
            ProviderStep::Completion(insufficient_evidence_dump()),
            ProviderStep::Completion(insufficient_evidence_dump()),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn(FIX_PROMPT, &mut ui).await.unwrap();
    assert_ne!(
        outcome.status,
        TurnStatus::Completed,
        "green verify must not turn a review dump into success: {outcome:?}; {:?}",
        ui.statuses
    );
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "expected no_progress even if checks passed: {outcome:?}; {:?}",
        ui.statuses
    );
    assert!(
        !ui.assistant.contains("Insufficient evidence"),
        "the review dump must not be the user-visible answer: {}",
        ui.assistant
    );
}

#[tokio::test]
async fn login_html_read_does_not_invent_a_redacted_secret_path() {
    let workspace = IsolatedWorkspace::new("stall-html-secret-class");
    std::fs::write(workspace.path("index.html"), LOGIN_HTML).unwrap();
    let cfg = stall_cfg(&workspace);
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(read_named("r1", "index.html")),
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "The login form shows offline and has no create-account control.".into(),
                )],
                1,
                1,
            )),
            ProviderStep::Completion(write_content_completion(
                "index.html",
                "<button>Create account</button>\n",
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text("Added a Create account button.".into())],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn(FIX_PROMPT, &mut ui).await.unwrap();
    let html_reads: Vec<&str> = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "read")
        .map(|(_, result)| result.as_str())
        .collect();
    assert!(
        html_reads
            .iter()
            .any(|result| result.contains("offline") && result.contains(r#"class="secret""#)),
        "login markup must stay visible to the model: {:?}",
        ui.tool_results
    );
    assert!(
        html_reads
            .iter()
            .all(|result| !result.contains("[REDACTED_SECRET]")),
        "class=secret must not become a redacted path: {:?}",
        ui.tool_results
    );
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}"
    );
}

#[tokio::test]
async fn cat_head_and_rg_after_inspection_nudge_are_withheld_until_an_edit() {
    for command in [
        "cat index.html",
        "head -n 20 index.html",
        "rg offline index.html",
        r#"grep -nE 'offline' index.html 2>&1; echo "EXIT=$?""#,
    ] {
        let workspace = IsolatedWorkspace::new(&format!(
            "stall-bash-{}",
            command.split_whitespace().next().unwrap()
        ));
        std::fs::write(workspace.path("index.html"), LOGIN_HTML).unwrap();
        std::fs::write(workspace.path("web.rs"), "pub fn page() {}\n").unwrap();
        let cfg = stall_cfg(&workspace);
        let (mut agent, _) = scripted_agent(inspect_then_bash_then_edit(command), cfg);
        let mut ui = RecUi::default();
        let outcome = agent.run_turn(FIX_PROMPT, &mut ui).await.unwrap();
        assert!(
            ui.tool_results.iter().any(|(name, result)| {
                name == "bash" && result.contains("withheld until a file change")
            }),
            "{command} after the edit nudge must not dump the file: {:?}",
            ui.tool_results
        );
        assert_ne!(
            outcome.stop_reason,
            TurnStopReason::NoProgress,
            "{command}: {outcome:?}"
        );
        assert!(
            std::fs::read_to_string(workspace.path("index.html"))
                .unwrap()
                .contains("Create account"),
            "{command}"
        );
    }
}

#[tokio::test]
async fn mixed_write_and_bash_sed_after_withhold_applies_only_the_edit() {
    let workspace = IsolatedWorkspace::new("stall-mixed-write-sed");
    std::fs::write(workspace.path("index.html"), LOGIN_HTML).unwrap();
    std::fs::write(workspace.path("web.rs"), "pub fn page() {}\n").unwrap();
    let cfg = stall_cfg(&workspace);
    let html = workspace.path("index.html").to_string_lossy().into_owned();
    let web = workspace.path("web.rs").to_string_lossy().into_owned();
    let mixed = completion(
        vec![
            Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: serde_json::json!({
                    "path": "index.html",
                    "content": "<button>Create account</button>\n"
                })
                .to_string(),
            },
            Content::ToolCall {
                id: "sed".into(),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": "sed -n '1,20p' index.html"
                })
                .to_string(),
            },
        ],
        1,
        1,
    );
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(read_named("r1", &html)),
            ProviderStep::Completion(read_named("r2", &web)),
            ProviderStep::Completion(read_named("r3", &html)),
            ProviderStep::Completion(mixed),
            ProviderStep::Completion(completion(
                vec![Content::Text("Added a Create account button.".into())],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn(FIX_PROMPT, &mut ui).await.unwrap();
    assert!(
        std::fs::read_to_string(workspace.path("index.html"))
            .unwrap()
            .contains("Create account"),
        "the write in a mixed withheld batch must land"
    );
    assert!(
        ui.tool_results.iter().any(|(name, result)| {
            name == "bash" && result.contains("withheld until a file change")
        }),
        "sed in the same batch must skip: {:?}",
        ui.tool_results
    );
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}"
    );
    assert_eq!(outcome.status, TurnStatus::Completed, "{outcome:?}");
}

#[tokio::test]
async fn review_shaped_insufficient_evidence_dump_is_kept_on_a_review_turn() {
    let workspace = IsolatedWorkspace::new("stall-review-dump-ok");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("src/lib.rs"),
        "pub fn answer() -> u32 { 42 }\n",
    )
    .unwrap();
    let cfg = stall_cfg(&workspace);
    let dump = "Insufficient evidence: I inspected src/lib.rs, but that evidence is not enough to make concrete gap, roadmap, or status claims for this request. A useful answer would need targeted reads or searches of the owning implementation modules, tests, and validation surface before recommending build-next work.\n\
Evidence summary: files_read=1, searches=0, listings=0, commands=0, failed_tools=0, inspected_paths=src/lib.rs.\n\
Insufficient evidence cause: inaccessible_path";
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(read_named("r1", "src/lib.rs")),
            ProviderStep::Completion(completion(vec![Content::Text(dump.into())], 1, 1)),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("review this code, do not edit", &mut ui)
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed, "{outcome:?}");
    assert!(
        ui.assistant.contains("Insufficient evidence")
            || agent
                .messages()
                .iter()
                .any(|message| message.text().contains("Insufficient evidence")),
        "review turns must keep the dump: {}",
        ui.assistant
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("review-shaped wrap-up is not an implementation")),
        "fix-turn wrap-up gate must not fire on a review: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn trivial_html_edit_with_a_normal_recap_is_not_no_progress() {
    let workspace = IsolatedWorkspace::new("stall-button-recap-ok");
    std::fs::write(workspace.path("index.html"), LOGIN_HTML).unwrap();
    let cfg = stall_cfg(&workspace);
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(write_content_completion(
                "index.html",
                "<button>Create account</button>\n",
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text("Added a Create account button.".into())],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn(FIX_PROMPT, &mut ui).await.unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed, "{outcome:?}");
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}"
    );
    assert!(
        std::fs::read_to_string(workspace.path("index.html"))
            .unwrap()
            .contains("Create account")
    );
}

#[tokio::test]
async fn truncated_extra_page_executes_then_a_completed_reread_withholds() {
    let workspace = IsolatedWorkspace::new("stall-truncated-page");
    let large: String = (1..=2_000)
        .map(|n| format!("source line {n} with enough text to blow the char budget"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(workspace.path("big.rs"), &large).unwrap();
    std::fs::write(workspace.path("tiny.rs"), "pub fn ready() {}\n").unwrap();
    let cfg = stall_cfg(&workspace);
    let big = workspace.path("big.rs").to_string_lossy().into_owned();
    let tiny = workspace.path("tiny.rs").to_string_lossy().into_owned();
    let page = completion(
        vec![Content::ToolCall {
            id: "page".into(),
            name: "read".into(),
            arguments: serde_json::json!({ "path": big, "offset": 400 }).to_string(),
        }],
        1,
        1,
    );
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(read_named("tiny", &tiny)),
            ProviderStep::Completion(read_named("big1", &big)),
            ProviderStep::Completion(page),
            ProviderStep::Completion(read_named("tiny2", &tiny)),
            ProviderStep::Completion(write_content_completion(
                "tiny.rs",
                "pub fn ready() { 1 }\n",
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "Paged the large file and edited tiny.rs.".into(),
                )],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn(FIX_PROMPT, &mut ui).await.unwrap();
    let page_result = ui
        .tool_results
        .iter()
        .find(|(name, result)| name == "read" && result.contains("source line 400"))
        .map(|(_, result)| result.as_str());
    assert!(
        page_result.is_some_and(|result| !result.contains("withheld until a file change")),
        "the extra truncated page must execute: {:?}",
        ui.tool_results
    );
    assert!(
        ui.statuses.iter().any(|status| {
            status.contains("inspection has not produced a file change")
                || status.contains("withholding inspection tools")
        }),
        "a completed-file reread after paging must demand an edit: {:?}",
        ui.statuses
    );
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}"
    );
    assert!(
        std::fs::read_to_string(workspace.path("tiny.rs"))
            .unwrap()
            .contains("pub fn ready() { 1 }")
    );
}

#[tokio::test]
async fn stderr_redirect_sed_pipeline_after_nudge_stops_instead_of_looping() {
    let workspace = IsolatedWorkspace::new("stall-stderr-sed-loop");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/db.rs"), "pub fn members() {}\n").unwrap();
    std::fs::write(workspace.path("index.html"), LOGIN_HTML).unwrap();
    let cfg = stall_cfg(&workspace);
    let html = workspace.path("index.html").to_string_lossy().into_owned();
    let db = workspace.path("src/db.rs").to_string_lossy().into_owned();
    let mut steps = vec![
        ProviderStep::Completion(read_named("r1", &html)),
        ProviderStep::Completion(read_named("r2", &db)),
        ProviderStep::Completion(read_named("r3", &html)),
    ];
    let mut sed = String::from("sed 's/^/GOT: /'");
    for index in 0..16 {
        let command =
            format!("grep -nE 'pub fn' src/db.rs | head -10 | {sed} 2>&1; echo \"EXIT=$?\"");
        steps.push(ProviderStep::Completion(bash_completion(&command)));
        sed.push_str(&format!(" | sed 's/{index}/{}/g'", index + 1));
    }
    steps.push(ProviderStep::Completion(completion(
        vec![Content::Text("still looking at db.rs".into())],
        1,
        1,
    )));
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let mut ui = RecUi::default();
    let outcome = agent.run_turn(FIX_PROMPT, &mut ui).await.unwrap();
    let sent = requests.lock().unwrap();
    assert!(
        sent.len() < 12,
        "growing 2>&1 sed dumps must not keep requesting: {} {:?}",
        sent.len(),
        ui.statuses
    );
    assert!(
        ui.tool_results.iter().any(|(name, result)| {
            name == "bash" && result.contains("withheld until a file change")
        }),
        "2>&1 grep/sed after the edit nudge must be withheld: {:?}",
        ui.tool_results
    );
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "this live loop must not run for hours: {outcome:?}; {:?}",
        ui.statuses
    );
    assert_eq!(outcome.status, TurnStatus::Failed, "{outcome:?}");
}

#[tokio::test]
async fn cargo_check_without_an_edit_on_a_fix_prompt_still_demands_a_mutation() {
    let workspace = IsolatedWorkspace::new("stall-validate-no-edit");
    std::fs::create_dir_all(workspace.path("src/web")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn ready() {}\n").unwrap();
    std::fs::write(
        workspace.path("src/web/index.html"),
        "<button id=\"register\">Create account</button><script>fetch('/register')</script>\n",
    )
    .unwrap();
    let cfg = stall_cfg(&workspace);
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(read_named("r1", "src/lib.rs")),
            ProviderStep::Completion(read_named("r2", "src/web/index.html")),
            ProviderStep::Completion(bash_completion("python3 -c 'assert 2 + 2 == 4'")),
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "No file changes are needed because the current implementation already wires Create account to POST /register."
                        .into(),
                )],
                1,
                1,
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "No file changes are needed because the current implementation already wires Create account to POST /register."
                        .into(),
                )],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn(
            "the create account flow is still missing from the web UI. add a visible register form and wire it to POST /register",
            &mut ui,
        )
        .await
        .unwrap();
    assert!(
        ui.statuses.iter().any(|status| {
            status.contains("validation without a file change does not finish a fix")
                || status.contains("inspection has not produced a file change")
                || status.contains("requesting an edit or explanation")
        }),
        "cargo check without an edit must not silently finish a fix: {:?}",
        ui.statuses
    );
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::InfrastructureFailure,
        "{outcome:?}"
    );
    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "an evidence-backed no-change answer after the nudge is allowed: {outcome:?}; {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn review_and_fix_nine_unique_reads_then_green_tests_then_decline_recap_completes() {
    // TUI: 9 unique reads, list, cargo test, leftover "Automatic recovery
    // stopped. No file changes were made." A healthy recap that declines
    // mutation must complete, not leftover-exhaust.
    let workspace = IsolatedWorkspace::new("stall-review-fix-tui-9");
    let paths = seed_unique_sources(&workspace, 9);
    let cfg = prod_recovery_cfg(&workspace);
    let mut steps = review_fix_inspect_then_tests(&paths, false);
    pad_decline_recaps(&mut steps, 12);
    let (mut agent, _) = scripted_agent(steps, cfg);
    let mut ui = RecUi::default();
    let outcome = agent.run_turn(REVIEW_FIX_PROMPT, &mut ui).await.unwrap();
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "healthy review-and-fix recap must not stall: {outcome:?}; {:?}",
        ui.statuses
    );
    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "{outcome:?}; {:?}",
        ui.statuses
    );
    assert!(
        ui.assistant.contains(DECLINE_RECAP) || ui.assistant.contains("no file changes are needed"),
        "user-visible answer must be the recap, not leftover closeout: {}",
        ui.assistant
    );
    assert!(
        !ui.assistant.contains(LEFTOVER_EMPTY_CLOSEOUT),
        "leftover empty closeout is a false stall: {}",
        ui.assistant
    );
    assert!(
        !agent.task_recovery().exhausted,
        "green tests plus a decline recap must not spend leftover recovery: {:?}",
        agent.task_recovery().last_reason
    );
}

#[tokio::test]
async fn review_and_fix_batched_reads_list_green_tests_then_decline_recap_completes() {
    let workspace = IsolatedWorkspace::new("stall-review-fix-tui-batch");
    let paths = seed_unique_sources(&workspace, 9);
    let cfg = prod_recovery_cfg(&workspace);
    let mut steps = review_fix_inspect_then_tests(&paths, true);
    pad_decline_recaps(&mut steps, 12);
    let (mut agent, _) = scripted_agent(steps, cfg);
    let mut ui = RecUi::default();
    let outcome = agent.run_turn(REVIEW_FIX_PROMPT, &mut ui).await.unwrap();
    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "batched 9-file inspection plus green tests plus a decline recap must complete: {outcome:?}; {:?}",
        ui.statuses
    );
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}"
    );
    assert!(
        !ui.assistant.contains(LEFTOVER_EMPTY_CLOSEOUT),
        "{}",
        ui.assistant
    );
    assert!(!agent.task_recovery().exhausted);
}

#[tokio::test]
async fn review_and_fix_nine_reads_then_dump_is_rejected() {
    let workspace = IsolatedWorkspace::new("stall-review-fix-9-dump");
    let paths = seed_unique_sources(&workspace, 9);
    let cfg = stall_cfg(&workspace);
    let mut steps = review_fix_inspect_then_tests(&paths, false);
    steps.push(ProviderStep::Completion(insufficient_evidence_dump()));
    steps.push(ProviderStep::Completion(insufficient_evidence_dump()));
    let (mut agent, _) = scripted_agent(steps, cfg);
    let mut ui = RecUi::default();
    let outcome = agent.run_turn(REVIEW_FIX_PROMPT, &mut ui).await.unwrap();
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("review-shaped wrap-up is not an implementation")),
        "the dump must be rejected: {:?}",
        ui.statuses
    );
    assert_eq!(outcome.status, TurnStatus::Failed, "{outcome:?}");
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}"
    );
    assert!(
        !ui.assistant.contains("Insufficient evidence:"),
        "the dump must not be the user-visible answer: {}",
        ui.assistant
    );
    assert!(
        !ui.assistant.contains(LEFTOVER_EMPTY_CLOSEOUT)
            || ui.statuses.iter().any(|status| {
                status.contains("review-shaped wrap-up is not an implementation")
            }),
        "dump rejection must not be hidden by leftover recovery: {}",
        ui.assistant
    );
}

#[tokio::test]
async fn review_and_fix_prod_recovery_without_a_recap_after_checks_completes() {
    let workspace = IsolatedWorkspace::new("stall-review-fix-prod-no-recap");
    let paths = seed_unique_sources(&workspace, 9);
    let extra = seed_unique_sources_from(&workspace, 9, 8);
    let cfg = prod_recovery_cfg(&workspace);
    let mut steps = review_fix_inspect_then_tests(&paths, false);
    steps.extend(unique_read_steps_prefixed(&extra, "x"));
    for index in 0..8 {
        steps.push(ProviderStep::Completion(completion(
            vec![Content::Text(format!("Still looking at file {index}."))],
            1,
            1,
        )));
    }
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let mut ui = RecUi::default();
    let outcome = agent.run_turn(REVIEW_FIX_PROMPT, &mut ui).await.unwrap();
    let sent = requests.lock().unwrap().len();
    assert!(
        sent < 40,
        "wrap-up after green checks must settle: {sent} {:?}",
        ui.statuses
    );
    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "green checks plus wrap-up without a model recap is a healthy review, not leftover: {outcome:?}; {:?}",
        ui.statuses
    );
    assert_ne!(
        outcome.stop_reason,
        TurnStopReason::NoProgress,
        "{outcome:?}; {:?}",
        ui.statuses
    );
    assert!(
        !ui.assistant.contains(LEFTOVER_EMPTY_CLOSEOUT),
        "{}",
        ui.assistant
    );
}
