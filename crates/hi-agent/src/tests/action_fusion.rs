use super::common::*;
use super::*;
use hi_ai::Content;

fn tool_outputs(agent: &Agent) -> Vec<(String, String)> {
    let mut names = Vec::new();
    let mut outputs = Vec::new();
    for message in agent.messages() {
        for content in &message.content {
            match content {
                Content::ToolCall { name, .. } => names.push(name.clone()),
                Content::ToolResult { output, .. } => {
                    let name = names.get(outputs.len()).cloned().unwrap_or_default();
                    outputs.push((name, output.clone()));
                }
                _ => {}
            }
        }
    }
    outputs
}

#[tokio::test]
async fn scheduler_fuses_write_then_bash_through_execute_mutation_then_command() {
    let mut cfg = config();
    cfg.memory.action_fusion = true;
    let mut agent = agent(
        vec![
            completion(
                vec![
                    Content::ToolCall {
                        id: "w".into(),
                        name: "write".into(),
                        arguments: r#"{"path":"fused.txt","content":"hello-fusion"}"#.into(),
                    },
                    Content::ToolCall {
                        id: "b".into(),
                        name: "bash".into(),
                        arguments: serde_json::json!({
                            "command": "printf 'fusion-command-output\\n'"
                        })
                        .to_string(),
                    },
                ],
                1,
                1,
            ),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    agent
        .run_turn("edit then check", &mut NullUi)
        .await
        .unwrap();
    let outputs = tool_outputs(&agent);
    let write = outputs
        .iter()
        .find(|(name, _)| name == "write")
        .map(|(_, output)| output.as_str())
        .unwrap_or_else(|| panic!("missing write observation: {outputs:?}"));
    let bash = outputs
        .iter()
        .find(|(name, _)| name == "bash")
        .map(|(_, output)| output.as_str())
        .unwrap_or_else(|| panic!("missing bash observation: {outputs:?}"));
    assert!(
        write.contains(hi_tools::FUSED_COMMAND_SUCCEEDED),
        "write observation should be the combined fusion body: {write}"
    );
    assert!(
        write.contains("fusion-command-output"),
        "command output belongs on the mutation observation: {write}"
    );
    assert!(
        bash.contains(hi_tools::FUSED_COMMAND_SUCCEEDED),
        "bash slot must keep the fused success marker: {bash}"
    );
    assert!(
        bash.contains("fusion-command-output"),
        "bash slot must keep the command output; a stub is how failed checks stall: {bash}"
    );
}

#[tokio::test]
async fn scheduler_skips_fused_command_when_mutation_fails() {
    let mut cfg = config();
    cfg.memory.action_fusion = true;
    cfg.loop_limits.max_recovery_interventions = 0;
    let mut agent = agent(
        vec![
            completion(
                vec![
                    Content::ToolCall {
                        id: "w".into(),
                        name: "write".into(),
                        arguments: r#"{"path":"blocked.txt","content":"nope"}"#.into(),
                    },
                    Content::ToolCall {
                        id: "b".into(),
                        name: "bash".into(),
                        arguments: serde_json::json!({
                            "command": "printf 'should-not-run\\n'"
                        })
                        .to_string(),
                    },
                ],
                1,
                1,
            ),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    std::fs::create_dir(agent.runtime.root().join("blocked.txt")).unwrap();
    agent
        .run_turn("edit then check", &mut NullUi)
        .await
        .unwrap();
    let outputs = tool_outputs(&agent);
    let write = outputs
        .iter()
        .find(|(name, _)| name == "write")
        .map(|(_, output)| output.as_str())
        .unwrap_or_else(|| panic!("missing write observation: {outputs:?}"));
    let bash = outputs
        .iter()
        .find(|(name, _)| name == "bash")
        .map(|(_, output)| output.as_str())
        .unwrap_or_else(|| panic!("missing bash observation: {outputs:?}"));
    assert!(
        write.contains(hi_tools::FUSED_COMMAND_SKIPPED),
        "failed mutation should skip the command: {write}"
    );
    assert!(
        !write.contains("should-not-run"),
        "skipped command must not run: {write}"
    );
    assert!(
        bash.contains(hi_tools::FUSED_COMMAND_SKIPPED),
        "bash stub should record the skip: {bash}"
    );
}

#[tokio::test]
async fn scheduler_keeps_failed_fused_command_output_on_bash() {
    let mut cfg = config();
    cfg.memory.action_fusion = true;
    cfg.loop_limits.max_recovery_interventions = 0;
    let mut agent = agent(
        vec![
            completion(
                vec![
                    Content::ToolCall {
                        id: "w".into(),
                        name: "write".into(),
                        arguments: r#"{"path":"fused.txt","content":"hello-fusion"}"#.into(),
                    },
                    Content::ToolCall {
                        id: "b".into(),
                        name: "bash".into(),
                        arguments: serde_json::json!({
                            "command": "printf 'error[E0425] cannot find function register\\n'; exit 1"
                        })
                        .to_string(),
                    },
                ],
                1,
                1,
            ),
            completion(vec![Content::Text("still broken".into())], 1, 1),
            completion(vec![Content::Text("still broken".into())], 1, 1),
            completion(vec![Content::Text("still broken".into())], 1, 1),
            completion(vec![Content::Text("still broken".into())], 1, 1),
        ],
        cfg,
    );
    agent
        .run_turn("edit then check", &mut NullUi)
        .await
        .unwrap();
    let outputs = tool_outputs(&agent);
    let write = outputs
        .iter()
        .find(|(name, _)| name == "write")
        .map(|(_, output)| output.as_str())
        .unwrap_or_else(|| panic!("missing write observation: {outputs:?}"));
    let bash = outputs
        .iter()
        .find(|(name, _)| name == "bash")
        .map(|(_, output)| output.as_str())
        .unwrap_or_else(|| panic!("missing bash observation: {outputs:?}"));
    assert!(
        write.contains(hi_tools::FUSED_COMMAND_FAILED),
        "combined mutation observation should keep the failed command: {write}"
    );
    assert!(
        write.contains("error[E0425] cannot find function register"),
        "combined mutation observation should keep the diagnostic: {write}"
    );
    assert!(
        bash.contains(hi_tools::FUSED_COMMAND_FAILED),
        "bash slot must not be a success stub after a failed check: {bash}"
    );
    assert!(
        bash.contains("error[E0425] cannot find function register"),
        "bash slot must keep the compiler diagnostic, not hide it behind a stub: {bash}"
    );
    assert!(
        !bash.contains("Command output is included in the preceding write observation."),
        "failed fused command must not be stubbed: {bash}"
    );
    assert!(
        write
            .trim_start()
            .starts_with(hi_tools::FUSED_COMMAND_FAILED),
        "failed check must lead the write observation so it is not read as success: {write}"
    );
}

fn fusion_config(workspace: &IsolatedWorkspace) -> AgentConfig {
    let mut cfg = workspace.config();
    cfg.memory.action_fusion = true;
    cfg.loop_limits.max_recovery_interventions = 0;
    cfg.loop_limits.max_keep_working = 0;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.gates.allow_unverified = true;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    cfg
}

fn write_then_bash(path: &str, content: &str, command: &str) -> Completion {
    completion(
        vec![
            Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: format!("{{\"path\":{path:?},\"content\":{content:?}}}"),
            },
            Content::ToolCall {
                id: "b".into(),
                name: "bash".into(),
                arguments: serde_json::json!({ "command": command }).to_string(),
            },
        ],
        1,
        1,
    )
}

fn done_steps(n: usize) -> Vec<ProviderStep> {
    (0..n)
        .map(|_| ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)))
        .collect()
}

fn request_tool_outputs(messages: &[hi_ai::Message]) -> Vec<String> {
    messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|content| match content {
            Content::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect()
}

fn request_blob(messages: &[hi_ai::Message]) -> String {
    request_tool_outputs(messages).join("\n---\n")
}

fn bulky_fail_command(marker: &str) -> String {
    format!(
        "printf '{marker}\\n'; i=0; while [ \"$i\" -lt 400 ]; do printf 'ok noise\\n'; i=$((i+1)); done; exit 1"
    )
}

/// The chat-session stall: the model claims cargo check passed after a fused
/// write+check that actually failed. Every later provider request must still
/// carry the diagnostic on both slots.
#[tokio::test]
async fn fused_failed_check_stays_in_every_later_provider_request() {
    let workspace = IsolatedWorkspace::new("fusion-stall-e0425");
    let path = workspace.path("src/web.rs").to_string_lossy().into_owned();
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    let mut steps = vec![ProviderStep::Completion(write_then_bash(
        &path,
        "fn serve() { register(); }\n",
        "printf 'error[E0425] cannot find function register in this scope\\n'; exit 1",
    ))];
    steps.push(ProviderStep::Completion(completion(
        vec![Content::Text(
            "Added a Create account button. Verification: cargo check --quiet passed (the earlier E0425 is resolved)."
                .into(),
        )],
        1,
        1,
    )));
    steps.extend(done_steps(4));
    let (mut agent, requests) = scripted_agent(steps, fusion_config(&workspace));
    let mut ui = RecUi::default();
    let _ = agent
        .run_turn("add a Create account button", &mut ui)
        .await
        .unwrap();
    let sent = requests.lock().unwrap();
    assert!(
        sent.len() >= 2,
        "tool round must produce a follow-up request: {}",
        sent.len()
    );
    for (index, request) in sent.iter().enumerate().skip(1) {
        let blob = request_blob(request);
        assert!(
            blob.contains("error[E0425] cannot find function register"),
            "provider request {index} dropped the failed check (stall): {blob}"
        );
        assert!(
            blob.contains(hi_tools::FUSED_COMMAND_FAILED),
            "provider request {index} lost the fused failure marker: {blob}"
        );
        assert!(
            !blob.contains("Command output is included in the preceding"),
            "provider request {index} stubbed the failed check: {blob}"
        );
    }
    let outputs = tool_outputs(&agent);
    let write = outputs
        .iter()
        .find(|(name, _)| name == "write")
        .map(|(_, output)| output.as_str())
        .unwrap();
    assert!(
        write
            .trim_start()
            .starts_with(hi_tools::FUSED_COMMAND_FAILED),
        "write observation must not lead with success: {write}"
    );
}

#[tokio::test]
async fn fused_stderr_only_and_empty_nonzero_exits_are_failures() {
    let workspace = IsolatedWorkspace::new("fusion-stderr-empty");
    let stderr_path = workspace.path("stderr.rs").to_string_lossy().into_owned();
    let empty_path = workspace.path("empty.rs").to_string_lossy().into_owned();
    let utf8_path = workspace.path("utf8.rs").to_string_lossy().into_owned();
    let mut steps = vec![ProviderStep::Completion(write_then_bash(
        &stderr_path,
        "fn a() {}",
        "printf 'error[E0425] cannot find function register\\n' >&2; exit 1",
    ))];
    steps.push(ProviderStep::Completion(write_then_bash(
        &empty_path,
        "fn b() {}",
        "exit 1",
    )));
    steps.push(ProviderStep::Completion(write_then_bash(
        &utf8_path,
        "fn c() {}",
        "printf 'error[E0425] cannot find function café\\n'; exit 1",
    )));
    steps.extend(done_steps(6));
    let (mut agent, _) = scripted_agent(steps, fusion_config(&workspace));
    agent
        .run_turn("edit then check", &mut NullUi)
        .await
        .unwrap();
    let outputs = tool_outputs(&agent);
    let blob = outputs
        .iter()
        .map(|(name, output)| format!("{name}:{output}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        blob.contains("error[E0425] cannot find function register"),
        "stderr-only rustc failure must reach the transcript: {blob}"
    );
    assert!(
        blob.contains("café") || blob.contains("caf"),
        "utf-8 diagnostic must not be dropped: {blob}"
    );
    let empty_write = outputs
        .iter()
        .filter(|(name, _)| name == "write")
        .map(|(_, output)| output.as_str())
        .find(|output| output.contains(hi_tools::FUSED_COMMAND_FAILED))
        .unwrap_or("");
    assert!(
        empty_write.contains(hi_tools::FUSED_COMMAND_FAILED),
        "empty nonzero exit must still be a fused failure: {blob}"
    );
    assert!(
        !blob.contains("Command output is included in the preceding"),
        "no fused failure may be stubbed: {blob}"
    );
}

#[tokio::test]
async fn fused_command_not_found_and_timeout_are_failures() {
    let workspace = IsolatedWorkspace::new("fusion-notfound-timeout");
    let missing_path = workspace.path("missing.rs").to_string_lossy().into_owned();
    let timeout_path = workspace.path("timeout.rs").to_string_lossy().into_owned();
    let mut steps = vec![ProviderStep::Completion(write_then_bash(
        &missing_path,
        "fn a() {}",
        "hi_fusion_definitely_not_a_command_9f3a",
    ))];
    steps.push(ProviderStep::Completion(completion(
        vec![
            Content::ToolCall {
                id: "w2".into(),
                name: "write".into(),
                arguments: format!("{{\"path\":{timeout_path:?},\"content\":\"fn b() {{}}\"}}"),
            },
            Content::ToolCall {
                id: "b2".into(),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": "sleep 30",
                    "timeout": 1
                })
                .to_string(),
            },
        ],
        1,
        1,
    )));
    steps.extend(done_steps(6));
    let (mut agent, _) = scripted_agent(steps, fusion_config(&workspace));
    agent
        .run_turn("edit then check", &mut NullUi)
        .await
        .unwrap();
    let outputs = tool_outputs(&agent);
    let blob = outputs
        .iter()
        .map(|(_, output)| output.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let failed = outputs
        .iter()
        .filter(|(name, output)| name == "bash" && output.contains(hi_tools::FUSED_COMMAND_FAILED))
        .count();
    assert!(
        failed >= 2,
        "not-found and timeout must both mark fused command failed: {blob}"
    );
    assert!(
        blob.contains("not found")
            || blob.contains("No such file")
            || blob.contains("timed out")
            || blob.contains(hi_tools::FUSED_COMMAND_FAILED),
        "random process errors must stay visible: {blob}"
    );
}

#[tokio::test]
async fn fused_gcc_failure_is_not_packed_off_later_requests() {
    let workspace = IsolatedWorkspace::new("fusion-pack-gcc");
    let path = workspace.path("web.rs").to_string_lossy().into_owned();
    let mut cfg = fusion_config(&workspace);
    cfg.memory.observation_pack = true;
    let mut steps = vec![ProviderStep::Completion(write_then_bash(
        &path,
        "fn serve() {}",
        &bulky_fail_command("src/web.rs:4:9: error: cannot find function register"),
    ))];
    for _ in 0..3 {
        steps.push(ProviderStep::Completion(echo_call()));
    }
    steps.extend(done_steps(4));
    let (mut agent, requests) = scripted_agent(steps, cfg);
    agent
        .run_turn("edit then check", &mut NullUi)
        .await
        .unwrap();
    let sent = requests.lock().unwrap();
    assert!(sent.len() >= 3, "need later packed rounds: {}", sent.len());
    let last = request_blob(sent.last().unwrap());
    assert!(
        last.contains("src/web.rs:4:9: error: cannot find function register")
            || last.contains(hi_tools::FUSED_COMMAND_FAILED),
        "later packed request must not drop the gcc failure: {last}"
    );
    assert!(
        !last.contains("id: obs_"),
        "unresolved compiler failure must not become a packed handle: {last}"
    );
}

#[tokio::test]
async fn parallel_fused_pairs_on_different_files_keep_both_outputs() {
    let workspace = IsolatedWorkspace::new("fusion-parallel-files");
    let a = workspace.path("a.rs").to_string_lossy().into_owned();
    let b = workspace.path("b.rs").to_string_lossy().into_owned();
    let mut steps = vec![ProviderStep::Completion(completion(
        vec![
            Content::ToolCall {
                id: "wa".into(),
                name: "write".into(),
                arguments: format!("{{\"path\":{a:?},\"content\":\"fn a() {{}}\"}}"),
            },
            Content::ToolCall {
                id: "ba".into(),
                name: "bash".into(),
                arguments: serde_json::json!({ "command": "printf 'alpha-ok\\n'" }).to_string(),
            },
            Content::ToolCall {
                id: "wb".into(),
                name: "write".into(),
                arguments: format!("{{\"path\":{b:?},\"content\":\"fn b() {{}}\"}}"),
            },
            Content::ToolCall {
                id: "bb".into(),
                name: "bash".into(),
                arguments: serde_json::json!({ "command": "printf 'beta-ok\\n'" }).to_string(),
            },
        ],
        1,
        1,
    ))];
    steps.extend(done_steps(3));
    let (mut agent, _) = scripted_agent(steps, fusion_config(&workspace));
    agent.run_turn("edit both", &mut NullUi).await.unwrap();
    let blob = tool_outputs(&agent)
        .iter()
        .map(|(_, output)| output.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        blob.contains("alpha-ok"),
        "first fused pair missing: {blob}"
    );
    assert!(
        blob.contains("beta-ok"),
        "second fused pair missing: {blob}"
    );
}

#[tokio::test]
async fn edit_then_failing_check_leads_with_failure() {
    let workspace = IsolatedWorkspace::new("fusion-edit-fail");
    let path = workspace.path("lib.rs").to_string_lossy().into_owned();
    std::fs::write(workspace.path("lib.rs"), "fn f() { old }\n").unwrap();
    let mut steps = vec![ProviderStep::Completion(completion(
        vec![
            Content::ToolCall {
                id: "e".into(),
                name: "edit".into(),
                arguments: serde_json::json!({
                    "path": path,
                    "old_string": "old",
                    "new_string": "new"
                })
                .to_string(),
            },
            Content::ToolCall {
                id: "b".into(),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": "printf 'error[E0425] cannot find function register\\n'; exit 1"
                })
                .to_string(),
            },
        ],
        1,
        1,
    ))];
    steps.extend(done_steps(4));
    let (mut agent, _) = scripted_agent(steps, fusion_config(&workspace));
    agent
        .run_turn("edit then check", &mut NullUi)
        .await
        .unwrap();
    let outputs = tool_outputs(&agent);
    let edit = outputs
        .iter()
        .find(|(name, _)| name == "edit")
        .map(|(_, output)| output.as_str())
        .unwrap_or("");
    let bash = outputs
        .iter()
        .find(|(name, _)| name == "bash")
        .map(|(_, output)| output.as_str())
        .unwrap_or("");
    assert!(
        edit.trim_start()
            .starts_with(hi_tools::FUSED_COMMAND_FAILED),
        "edit observation must lead with the failed check: {edit}"
    );
    assert!(
        bash.contains("error[E0425] cannot find function register"),
        "edit-then-check bash slot must keep the diagnostic: {bash}"
    );
}

fn named_write_then_bash(
    write_id: &str,
    bash_id: &str,
    path: &str,
    content: &str,
    command: &str,
) -> Completion {
    completion(
        vec![
            Content::ToolCall {
                id: write_id.into(),
                name: "write".into(),
                arguments: format!("{{\"path\":{path:?},\"content\":{content:?}}}"),
            },
            Content::ToolCall {
                id: bash_id.into(),
                name: "bash".into(),
                arguments: serde_json::json!({ "command": command }).to_string(),
            },
        ],
        1,
        1,
    )
}

fn transcript_blob(agent: &Agent) -> String {
    tool_outputs(agent)
        .iter()
        .map(|(name, output)| format!("{name}:{output}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn assert_visible_failure(blob: &str, marker: &str) {
    assert!(
        blob.contains(marker),
        "diagnostic {marker:?} missing from fused transcript: {blob}"
    );
    assert!(
        blob.contains(hi_tools::FUSED_COMMAND_FAILED)
            || blob.contains(hi_tools::FUSED_COMMAND_SKIPPED),
        "fused failure marker missing next to {marker:?}: {blob}"
    );
    assert!(
        !blob.contains("Command output is included in the preceding"),
        "fused failure was stubbed ({marker:?}): {blob}"
    );
}

#[tokio::test]
async fn fused_failure_table_covers_compiler_test_and_runtime_signatures() {
    let workspace = IsolatedWorkspace::new("fusion-failure-table");
    let cases = [
        (
            "rustc.rs",
            "printf 'error[E0425] cannot find function register\\n'; exit 1",
            "error[E0425] cannot find function register",
        ),
        (
            "gcc.rs",
            "printf 'src/web.rs:4:9: error: cannot find function register\\n'; exit 1",
            "src/web.rs:4:9: error:",
        ),
        (
            "cargo-test.rs",
            "printf 'running 3 tests\\ntest result: FAILED. 1 passed; 1 failed\\n'; exit 1",
            "test result: FAILED",
        ),
        (
            "panic.rs",
            "printf 'thread tests::it_breaks panicked at src/lib.rs:4:1:\\nassertion failed\\n'; exit 1",
            "panicked at",
        ),
        (
            "pytest.rs",
            "printf '===== FAILURES =====\\nAssertionError: mismatch\\n'; exit 1",
            "AssertionError",
        ),
        (
            "go.rs",
            "printf '\\nFAIL: TestRegister\\n'; exit 1",
            "FAIL: TestRegister",
        ),
        (
            "tsc.rs",
            "printf 'src/web.ts(4,9): error TS2322: Type mismatch\\n'; exit 1",
            "error TS2322",
        ),
        (
            "python.rs",
            "printf 'Traceback (most recent call last):\\nAssertionError: boom\\n'; exit 1",
            "AssertionError: boom",
        ),
        (
            "pipefail.rs",
            "false | printf 'error: cannot find function register\\n'; exit 1",
            "error: cannot find function register",
        ),
    ];
    let mut steps = Vec::new();
    for (i, (file, command, _)) in cases.iter().enumerate() {
        let path = workspace.path(file).to_string_lossy().into_owned();
        steps.push(ProviderStep::Completion(named_write_then_bash(
            &format!("w{i}"),
            &format!("b{i}"),
            &path,
            "fn f() {}",
            command,
        )));
    }
    steps.extend(done_steps(12));
    let (mut agent, _) = scripted_agent(steps, fusion_config(&workspace));
    agent
        .run_turn("edit then check each signature", &mut NullUi)
        .await
        .unwrap();
    let blob = transcript_blob(&agent);
    for (_, _, marker) in cases {
        assert_visible_failure(&blob, marker);
    }
}

#[tokio::test]
async fn apply_patch_and_multi_edit_then_failing_check_stay_visible() {
    let workspace = IsolatedWorkspace::new("fusion-patch-multiedit");
    let multi_path = workspace.path("multi.rs").to_string_lossy().into_owned();
    std::fs::write(workspace.path("multi.rs"), "fn f() { old }\n").unwrap();
    let mut steps = vec![
        ProviderStep::Completion(completion(
            vec![
                Content::ToolCall {
                    id: "p".into(),
                    name: "apply_patch".into(),
                    arguments: serde_json::json!({
                        "patch": "*** Begin Patch\n*** Add File: patched.rs\n+fn patched() {}\n*** End Patch"
                    })
                    .to_string(),
                },
                Content::ToolCall {
                    id: "pb".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({
                        "command": "printf 'error[E0425] cannot find function register\\n'; exit 1"
                    })
                    .to_string(),
                },
            ],
            1,
            1,
        )),
        ProviderStep::Completion(completion(
            vec![
                Content::ToolCall {
                    id: "m".into(),
                    name: "multi_edit".into(),
                    arguments: serde_json::json!({
                        "path": multi_path,
                        "edits": [{"old_string": "old", "new_string": "new"}]
                    })
                    .to_string(),
                },
                Content::ToolCall {
                    id: "mb".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({
                        "command": "printf 'src/web.rs:4:9: error: cannot find function register\\n'; exit 1"
                    })
                    .to_string(),
                },
            ],
            1,
            1,
        )),
    ];
    steps.extend(done_steps(6));
    let (mut agent, _) = scripted_agent(steps, fusion_config(&workspace));
    agent
        .run_turn("patch and multi-edit then check", &mut NullUi)
        .await
        .unwrap();
    let outputs = tool_outputs(&agent);
    let blob = transcript_blob(&agent);
    assert_visible_failure(&blob, "error[E0425] cannot find function register");
    assert_visible_failure(&blob, "src/web.rs:4:9: error:");
    for name in ["apply_patch", "multi_edit"] {
        let body = outputs
            .iter()
            .find(|(tool, _)| tool == name)
            .map(|(_, output)| output.as_str())
            .unwrap_or("");
        assert!(
            body.trim_start()
                .starts_with(hi_tools::FUSED_COMMAND_FAILED),
            "{name} must lead with the failed check: {body}"
        );
    }
}

#[tokio::test]
async fn write_read_bash_does_not_fuse_but_keeps_the_check_output() {
    let workspace = IsolatedWorkspace::new("fusion-write-read-bash");
    let path = workspace.path("lib.rs").to_string_lossy().into_owned();
    let mut steps = vec![ProviderStep::Completion(completion(
        vec![
            Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: format!("{{\"path\":{path:?},\"content\":\"fn f() {{}}\"}}"),
            },
            Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: format!("{{\"path\":{path:?}}}"),
            },
            Content::ToolCall {
                id: "b".into(),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": "printf 'error[E0425] cannot find function register\\n'; exit 1"
                })
                .to_string(),
            },
        ],
        1,
        1,
    ))];
    steps.extend(done_steps(4));
    let (mut agent, _) = scripted_agent(steps, fusion_config(&workspace));
    agent
        .run_turn("write, read, then check", &mut NullUi)
        .await
        .unwrap();
    let outputs = tool_outputs(&agent);
    let write = outputs
        .iter()
        .find(|(name, _)| name == "write")
        .map(|(_, output)| output.as_str())
        .unwrap_or("");
    let bash = outputs
        .iter()
        .find(|(name, _)| name == "bash")
        .map(|(_, output)| output.as_str())
        .unwrap_or("");
    assert!(
        !write.contains(hi_tools::FUSED_COMMAND_FAILED)
            && !write.contains(hi_tools::FUSED_COMMAND_SUCCEEDED),
        "read between write and bash must prevent fusion: {write}"
    );
    assert!(
        bash.contains("error[E0425] cannot find function register"),
        "unfused bash must still keep the diagnostic: {bash}"
    );
}

#[tokio::test]
async fn sequential_fused_failures_on_the_same_file_keep_both_diagnostics() {
    let workspace = IsolatedWorkspace::new("fusion-same-file-twice");
    let path = workspace.path("web.rs").to_string_lossy().into_owned();
    let mut steps = vec![
        ProviderStep::Completion(named_write_then_bash(
            "w1",
            "b1",
            &path,
            "fn serve() { register(); }",
            "printf 'error[E0425] cannot find function register\\n'; exit 1",
        )),
        ProviderStep::Completion(named_write_then_bash(
            "w2",
            "b2",
            &path,
            "fn serve() { login(); }",
            "printf 'error[E0425] cannot find function login\\n'; exit 1",
        )),
    ];
    steps.extend(done_steps(6));
    let (mut agent, requests) = scripted_agent(steps, fusion_config(&workspace));
    agent.run_turn("repair twice", &mut NullUi).await.unwrap();
    let blob = transcript_blob(&agent);
    assert_visible_failure(&blob, "cannot find function register");
    assert_visible_failure(&blob, "cannot find function login");
    let sent = requests.lock().unwrap();
    let last = request_blob(sent.last().expect("follow-up request"));
    assert!(
        last.contains("cannot find function login"),
        "latest same-file failure must remain on the last request: {last}"
    );
}

#[tokio::test]
async fn fused_skip_stays_on_later_requests_and_ui() {
    let workspace = IsolatedWorkspace::new("fusion-skip-later");
    std::fs::create_dir(workspace.path("blocked.txt")).unwrap();
    let path = workspace.path("blocked.txt").to_string_lossy().into_owned();
    let mut steps = vec![ProviderStep::Completion(write_then_bash(
        &path,
        "nope",
        "printf 'should-not-run\\n'",
    ))];
    for _ in 0..3 {
        steps.push(ProviderStep::Completion(echo_call()));
    }
    steps.extend(done_steps(3));
    let (mut agent, requests) = scripted_agent(steps, fusion_config(&workspace));
    let mut ui = RecUi::default();
    agent.run_turn("edit then check", &mut ui).await.unwrap();
    let sent = requests.lock().unwrap();
    for (index, request) in sent.iter().enumerate().skip(1) {
        let blob = request_blob(request);
        assert!(
            blob.contains(hi_tools::FUSED_COMMAND_SKIPPED),
            "request {index} dropped the skipped fused command: {blob}"
        );
        assert!(
            !blob.contains("should-not-run"),
            "skipped command ran on request {index}: {blob}"
        );
    }
    assert!(
        ui.tool_results
            .iter()
            .any(|(_, result)| result.contains(hi_tools::FUSED_COMMAND_SKIPPED)),
        "UI must show the skip, not a silent success: {:?}",
        ui.tool_results
    );
}

#[tokio::test]
async fn bulky_green_fused_log_may_pack_but_failures_must_not() {
    let workspace = IsolatedWorkspace::new("fusion-pack-green-vs-fail");
    let green_path = workspace.path("green.rs").to_string_lossy().into_owned();
    let fail_path = workspace.path("fail.rs").to_string_lossy().into_owned();
    let mut cfg = fusion_config(&workspace);
    cfg.memory.observation_pack = true;
    let mut steps = vec![
        ProviderStep::Completion(named_write_then_bash(
            "wg",
            "bg",
            &green_path,
            "fn ok() {}",
            "printf 'test result: ok. 3 passed; 0 failed\\n'; i=0; while [ \"$i\" -lt 400 ]; do printf 'ok noise\\n'; i=$((i+1)); done",
        )),
        ProviderStep::Completion(named_write_then_bash(
            "wf",
            "bf",
            &fail_path,
            "fn bad() {}",
            &bulky_fail_command("error[E0425] cannot find function register"),
        )),
    ];
    for _ in 0..3 {
        steps.push(ProviderStep::Completion(echo_call()));
    }
    steps.extend(done_steps(4));
    let (mut agent, requests) = scripted_agent(steps, cfg);
    agent
        .run_turn("green and failing checks", &mut NullUi)
        .await
        .unwrap();
    let sent = requests.lock().unwrap();
    assert!(sent.len() >= 3, "need later rounds: {}", sent.len());
    let last = request_blob(sent.last().unwrap());
    assert!(
        last.contains("error[E0425] cannot find function register")
            || last.contains(hi_tools::FUSED_COMMAND_FAILED),
        "failed fused check must not pack away: {last}"
    );
    let fail_chunks = last
        .matches("error[E0425] cannot find function register")
        .count()
        + last.matches(hi_tools::FUSED_COMMAND_FAILED).count();
    assert!(
        fail_chunks > 0,
        "failure evidence missing from packed later request: {last}"
    );
}
