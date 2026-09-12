use super::*;

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
