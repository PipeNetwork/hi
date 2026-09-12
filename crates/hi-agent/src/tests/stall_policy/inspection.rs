use super::*;

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
