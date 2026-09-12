use super::*;

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
