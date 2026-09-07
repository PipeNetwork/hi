use super::common::*;
use super::*;
use hi_workspace::WorkspaceController;

struct EnvelopeProbe {
    responses: Mutex<Vec<Completion>>,
    advertised: std::sync::Arc<Mutex<Vec<Vec<String>>>>,
    attachments: std::sync::Arc<Mutex<Vec<hi_ai::RequestToolEnvelope>>>,
    capabilities: hi_ai::ProviderCapabilities,
}

fn envelope_capabilities() -> hi_ai::ProviderCapabilities {
    let mut capabilities = hi_ai::ProviderCapabilities::native_tools(false);
    capabilities.parallel_tool_calls = true;
    capabilities.tool_choice.automatic = true;
    capabilities.actual_model_revision = Some("probe-model@2026-09-03".to_string());
    capabilities
}

#[async_trait::async_trait]
impl hi_ai::Provider for EnvelopeProbe {
    async fn stream(
        &self,
        request: hi_ai::ChatRequest,
        sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        self.advertised
            .lock()
            .unwrap()
            .push(request.tools.iter().map(|tool| tool.name.clone()).collect());
        self.attachments.lock().unwrap().push(
            request
                .tool_envelope
                .as_deref()
                .cloned()
                .expect("attached envelope"),
        );
        sink(hi_ai::StreamEvent::WireAudit(Box::default()));
        pop_canned_completion(&self.responses, "EnvelopeProbe")
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        self.capabilities.clone()
    }
}

#[tokio::test]
async fn unadvertised_known_tool_is_rejected_by_the_same_audited_envelope() {
    let workspace = IsolatedWorkspace::new("tool-envelope-boundary");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn old() {}\n").unwrap();
    let mut cfg = workspace.config();
    cfg.memory.tool_set = ToolSet::Dynamic;
    cfg.loop_limits.max_steps = 1;
    let advertised = std::sync::Arc::new(Mutex::new(Vec::new()));
    let attachments = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = EnvelopeProbe {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "outside-envelope".into(),
                    name: "update_plan".into(),
                    arguments: serde_json::json!({
                        "steps": [{"title": "bypass selection", "status": "active"}]
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(vec![Content::Text("The call was rejected.".into())], 1, 1),
        ]),
        advertised: advertised.clone(),
        attachments: attachments.clone(),
        capabilities: envelope_capabilities(),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let registry = hi_ai::ProviderCapabilityRegistry::default();
    let mut resolved_capabilities = envelope_capabilities();
    resolved_capabilities.structured_output = true;
    resolved_capabilities.actual_model_revision = Some("resolved-model@2026-09-04".into());
    registry.register(
        hi_ai::CapabilityRoute::new("unknown", "m"),
        resolved_capabilities.clone(),
    );
    agent.set_provider_capability_registry(registry);
    let mut ui = RecUi::default();

    let _ = agent
        .run_turn("Edit src/lib.rs to rename old to new.", &mut ui)
        .await;

    let first_tools = advertised.lock().unwrap().first().cloned().unwrap();
    let first_attachment = attachments.lock().unwrap().first().cloned().unwrap();
    assert!(!first_tools.iter().any(|name| name == "update_plan"));
    let denial = ui
        .tool_results
        .iter()
        .find(|(name, _)| name == "update_plan")
        .map(|(_, result)| serde_json::from_str::<serde_json::Value>(result).unwrap())
        .expect("outside-envelope call receives a typed result");
    assert_eq!(denial["error"]["kind"], "tool_protocol_error");
    assert_eq!(denial["error"]["reason"], "unavailable_tool");
    assert!(
        denial["error"]["message"]
            .as_str()
            .unwrap()
            .contains("sealed envelope")
    );
    assert!(
        ui.plans.is_empty(),
        "the omitted coordination tool must not run"
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("schema-corrected")
                || status.contains("DeepSeek tool arguments")
                || status.contains("plain-text tool call")),
        "an unavailable tool must not enter argument/schema recovery: {:?}",
        ui.statuses
    );

    let audit = agent
        .last_turn_telemetry()
        .wire_audit
        .first()
        .expect("provider audit is retained");
    let digest = audit["tool_envelope_digest"].as_str().unwrap();
    assert!(digest.starts_with("blake3:"));
    assert_eq!(digest, first_attachment.digest);
    assert_eq!(audit["tool_envelope"], first_attachment.payload);
    let audited_names = audit["tool_envelope"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|tool| tool["name"].as_str().map(str::to_string))
        .collect::<Vec<_>>();
    assert_eq!(audited_names, first_tools);
    let expected_record = hi_ai::EffectiveProviderCapabilities::conservative(
        hi_ai::CapabilityRoute::new("unknown", "m"),
        resolved_capabilities,
    );
    let expected_capability_digest = expected_record.canonical_digest();
    assert_eq!(
        audit["tool_envelope"]["provider"]["capability_digest"].as_str(),
        Some(expected_capability_digest.as_str())
    );
    assert_eq!(
        audit["tool_envelope"]["provider"]["actual_model_revision"].as_str(),
        Some("resolved-model@2026-09-04")
    );
    assert_eq!(
        audit["tool_envelope"]["provider"]["capability_record"],
        serde_json::to_value(expected_record).unwrap()
    );
}

#[tokio::test]
async fn malformed_mutation_is_denied_before_workspace_admission() {
    let workspace = IsolatedWorkspace::new("tool-envelope-before-admission");
    std::fs::write(workspace.path("src.rs"), "pub fn inspected() {}\n").unwrap();
    let mut cfg = workspace.config();
    cfg.memory.tool_set = ToolSet::Dynamic;
    cfg.loop_limits.max_steps = 1;
    let root = cfg.paths.workspace_root.clone();
    let state = cfg.paths.state_root.clone();
    let advertised = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = EnvelopeProbe {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "unavailable-bash".into(),
                    name: "bash".into(),
                    arguments: "{}".into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text("The unavailable call was denied.".into())],
                1,
                1,
            ),
        ]),
        advertised: advertised.clone(),
        attachments: std::sync::Arc::new(Mutex::new(Vec::new())),
        capabilities: envelope_capabilities(),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let controller = std::sync::Arc::new(hi_workspace::InMemoryWorkspaceController::new_pipefs(
        "envelope-workspace",
        "envelope-session",
        2,
        true,
        root,
        state,
    ));
    agent
        .install_workspace_controller(controller.clone())
        .unwrap();
    let existing = controller
        .begin(hi_workspace::MutationIntent::workspace("existing writer"))
        .await
        .unwrap();
    let active_operation = controller.status().active_operation;
    let mut ui = RecUi::default();

    let _ = agent
        .run_turn("Implement the requested change in src.rs.", &mut ui)
        .await;

    assert!(
        advertised.lock().unwrap()[0]
            .iter()
            .any(|name| name == "bash")
    );
    let denial = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "bash")
        .find_map(|(_, result)| serde_json::from_str::<serde_json::Value>(result).ok())
        .unwrap_or_else(|| panic!("sealed-envelope denial missing: {:?}", ui.tool_results));
    assert_eq!(denial["error"]["reason"], "invalid_arguments");
    assert_eq!(controller.status().active_operation, active_operation);
    assert!(!workspace.path("must-not-exist").exists());
    let settled = controller
        .settle(existing, hi_workspace::ExecutionReport::succeeded(None))
        .await;
    assert!(settled.receipt.is_some());
}

#[tokio::test]
async fn unadvertised_run_program_uses_typed_unavailable_recovery() {
    let workspace = IsolatedWorkspace::new("unadvertised-program-envelope");
    std::fs::write(workspace.path("source.txt"), "evidence\n").unwrap();
    let advertised = std::sync::Arc::new(Mutex::new(Vec::new()));
    let attachments = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = EnvelopeProbe {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "unavailable-program".into(),
                    name: "run_program".into(),
                    arguments: serde_json::json!({"source": "42"}).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "The unavailable workflow program was not executed.".into(),
                )],
                1,
                1,
            ),
        ]),
        advertised: advertised.clone(),
        attachments,
        capabilities: envelope_capabilities(),
    };
    let mut cfg = workspace.config();
    cfg.loop_limits.max_steps = 2;
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();

    let _ = agent
        .run_turn("Inspect source.txt without changing it.", &mut ui)
        .await;

    assert!(
        advertised
            .lock()
            .unwrap()
            .iter()
            .all(|tools| !tools.iter().any(|name| name == "run_program"))
    );
    let result = ui
        .tool_results
        .iter()
        .find(|(name, _)| name == "run_program")
        .map(|(_, result)| serde_json::from_str::<serde_json::Value>(result).unwrap())
        .expect("unavailable run_program receives a typed result");
    assert_eq!(result["error"]["kind"], "tool_protocol_error");
    assert_eq!(result["error"]["reason"], "unavailable_tool");
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("ordinary structured tools after run_program failure")),
        "an unavailable program must not enter the execution-failure fallback: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn read_only_envelope_recovers_from_unavailable_bash_with_an_admitted_read() {
    let workspace = IsolatedWorkspace::new("read-only-envelope-recovery");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn inspected() {}\n").unwrap();
    let mut cfg = workspace.config();
    cfg.memory.tool_set = ToolSet::Dynamic;
    cfg.routing.deepseek_compat = hi_ai::DeepSeekCompat::On;
    cfg.loop_limits.max_steps = 4;
    let advertised = std::sync::Arc::new(Mutex::new(Vec::new()));
    let attachments = std::sync::Arc::new(Mutex::new(Vec::new()));
    let mut auto_only_capabilities = envelope_capabilities();
    auto_only_capabilities.tool_choice.required = false;
    let provider = EnvelopeProbe {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "unavailable-bash".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({"command": "sed -n '1,80p' src/lib.rs"})
                        .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "admitted-read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": "src/lib.rs"}).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "Finding: `src/lib.rs` defines `inspected`. Limits: only that file was reviewed."
                        .into(),
                )],
                1,
                1,
            ),
        ]),
        advertised: advertised.clone(),
        attachments: attachments.clone(),
        capabilities: auto_only_capabilities,
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();

    let _ = agent
        .run_turn("Review src/lib.rs and report one bounded finding.", &mut ui)
        .await;

    let advertised = advertised.lock().unwrap();
    assert!(advertised.len() >= 2, "expected a corrected model request");
    assert!(advertised[0].iter().any(|name| name == "read"));
    assert!(!advertised[0].iter().any(|name| name == "bash"));
    assert!(advertised[1].iter().any(|name| name == "read"));
    assert!(!advertised[1].iter().any(|name| name == "bash"));
    assert_eq!(
        attachments.lock().unwrap()[1].payload["tool_mode"],
        serde_json::json!("auto"),
        "unavailable-tool recovery must not turn an Auto-only route into ChatOnly"
    );
    assert_eq!(
        ui.tool_results
            .iter()
            .filter(|(name, _)| name == "bash")
            .count(),
        1,
        "the forbidden call must not enter a retry loop"
    );
    assert!(
        ui.tool_results
            .iter()
            .any(|(name, result)| name == "read" && result.contains("pub fn inspected")),
        "the corrected admitted read must execute: {:?}",
        ui.tool_results
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("schema-corrected")
                || status.contains("DeepSeek tool arguments")
                || status.contains("plain-text tool call")),
        "unavailable-tool recovery must stay out of schema/plain-text fallbacks: {:?}",
        ui.statuses
    );
    let transcript = agent
        .messages()
        .iter()
        .map(hi_ai::Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(transcript.contains("not admitted by this request's sealed tool envelope"));
    assert!(transcript.contains("only admitted tool names"));
    assert!(!transcript.contains("Emit a new `bash` call"));
}

#[tokio::test]
async fn plain_text_fallback_stays_executable_on_an_auto_only_provider() {
    let workspace = IsolatedWorkspace::new("auto-only-text-tool-fallback");
    let destination = workspace.path("result.txt");
    let destination_text = destination.to_string_lossy();
    let invalid_write = |id: &str, arguments: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "write".into(),
                arguments: arguments.into(),
            }],
            1,
            1,
        )
    };
    let xmlish_write = format!(
        "<tool_call>write<arg_key>path</arg_key><arg_value>{destination_text}</arg_value><arg_key>content</arg_key><arg_value>ok\n</arg_value></tool_call>"
    );
    let advertised = std::sync::Arc::new(Mutex::new(Vec::new()));
    let attachments = std::sync::Arc::new(Mutex::new(Vec::new()));
    let mut capabilities = envelope_capabilities();
    capabilities.tool_choice.required = false;
    let provider = EnvelopeProbe {
        responses: Mutex::new(vec![
            invalid_write("invalid-write-1", "{}"),
            invalid_write(
                "invalid-write-2",
                &serde_json::json!({"path": destination_text}).to_string(),
            ),
            completion(vec![Content::Text(xmlish_write)], 1, 1),
            completion(
                vec![Content::Text(
                    "Created `result.txt` with the requested content.".into(),
                )],
                1,
                1,
            ),
            bash_completion("python3 -c 'assert 2 + 2 == 4'"),
            completion(
                vec![Content::Text("Created and validated `result.txt`.".into())],
                1,
                1,
            ),
        ]),
        advertised,
        attachments: attachments.clone(),
        capabilities,
    };
    let mut cfg = workspace.config();
    cfg.loop_limits.max_repeat_nudges = 1;
    cfg.gates.allow_unverified = true;
    cfg.memory.finalize = false;
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(
            &format!("Create {} containing ok.", destination.display()),
            &mut ui,
        )
        .await;

    assert!(
        outcome.is_ok(),
        "outcome={outcome:?}; statuses={:?}; tools={:?}; transcript={:?}",
        ui.statuses,
        ui.tool_results,
        agent
            .messages()
            .iter()
            .map(hi_ai::Message::text)
            .collect::<Vec<_>>()
    );
    assert_eq!(outcome.unwrap().status, TurnStatus::Completed);
    assert_eq!(std::fs::read_to_string(destination).unwrap(), "ok\n");
    let attachments = attachments.lock().unwrap();
    assert!(attachments.len() >= 3);
    assert_eq!(
        attachments[0].payload["tool_mode"],
        serde_json::json!("auto")
    );
    assert_eq!(
        attachments[1].payload["tool_mode"],
        serde_json::json!("auto")
    );
    assert_eq!(
        attachments[2].payload["tool_mode"],
        serde_json::json!("auto")
    );
    assert!(
        attachments[2].payload["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == serde_json::json!("write"))
    );
    assert!(attachments[2].requests_text_tool_fallback());
    assert!(!attachments[0].requests_text_tool_fallback());
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("DeepSeek tool arguments")),
        "DeepSeek schema fallback must not activate for an unrelated Auto route: {:?}",
        ui.statuses
    );
    let transcript = agent
        .messages()
        .iter()
        .map(hi_ai::Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(transcript.contains("current sealed envelope"));
    assert!(transcript.contains("<tool_call>"));
}

#[tokio::test]
async fn ordinary_narrative_tool_json_is_not_promoted() {
    let workspace = IsolatedWorkspace::new("narrative-tool-json");
    let destination = workspace.path("should-not-exist.txt");
    let textual_call = format!(
        r#"For example only: {{"name":"write","arguments":{{"path":"{}","content":"blocked"}}}}"#,
        destination.display()
    );
    let advertised = std::sync::Arc::new(Mutex::new(Vec::new()));
    let attachments = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = EnvelopeProbe {
        responses: Mutex::new(vec![completion(
            vec![Content::Text(textual_call.clone())],
            1,
            1,
        )]),
        advertised: advertised.clone(),
        attachments,
        capabilities: envelope_capabilities(),
    };
    let mut cfg = workspace.config();
    cfg.memory.tool_set = ToolSet::Full;
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();

    let _ = agent
        .run_turn("What does this hypothetical JSON object mean?", &mut ui)
        .await;

    assert!(!destination.exists());
    assert!(
        advertised
            .lock()
            .unwrap()
            .first()
            .is_some_and(|tools| tools.iter().any(|name| name == "write")),
        "write must be admitted so this proves call-channel gating"
    );
    assert!(ui.tool_results.iter().all(|(name, _)| name != "write"));
    let transcript = agent
        .messages()
        .iter()
        .map(hi_ai::Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        transcript.contains(&textual_call),
        "ordinary narrative must remain intact: {transcript:?}"
    );
}

struct ProvisionalProgramProvider {
    responses: Mutex<Vec<Completion>>,
    emitted_delta: std::sync::atomic::AtomicBool,
    advertised_program: std::sync::Arc<std::sync::atomic::AtomicBool>,
    speculative_call_seen: std::sync::Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for ProvisionalProgramProvider {
    async fn stream(
        &self,
        request: hi_ai::ChatRequest,
        sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        use std::sync::atomic::Ordering;

        let advertised = request.tools.iter().any(|tool| tool.name == "run_program");
        self.advertised_program.store(advertised, Ordering::Relaxed);
        if !self.emitted_delta.swap(true, Ordering::Relaxed) {
            let arguments = serde_json::json!({
                "source": r#"tool("read", #{uri: "mcp://probe/file:///sentinel"})"#
            })
            .to_string();
            sink(hi_ai::StreamEvent::ToolCallDelta {
                index: 0,
                id_delta: Some("provisional-program".into()),
                name_delta: Some("run_program".into()),
                arguments_delta: arguments,
            });
            let _ = tokio::time::timeout(
                std::time::Duration::from_millis(250),
                self.speculative_call_seen.notified(),
            )
            .await;
            sink(hi_ai::StreamEvent::ToolCallDelta {
                index: 0,
                id_delta: None,
                name_delta: Some("_not_admitted".into()),
                arguments_delta: String::new(),
            });
        }
        pop_canned_completion(&self.responses, "ProvisionalProgramProvider")
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        hi_ai::ProviderCapabilities::native_tools(true)
    }
}

struct CountingResourceMcp {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    called: std::sync::Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl hi_tools::McpBackend for CountingResourceMcp {
    async fn search(&self, _: Option<&str>) -> anyhow::Result<Vec<hi_tools::McpToolInfo>> {
        Ok(Vec::new())
    }

    async fn call(&self, _: &str, _: &str, _: &serde_json::Value) -> anyhow::Result<String> {
        unreachable!("the speculative program only requests an MCP resource read")
    }

    async fn read_resource(&self, _: &str, _: &str) -> anyhow::Result<String> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.called.notify_waiters();
        Ok("sentinel".into())
    }
}

#[tokio::test]
async fn provisional_streamed_program_never_crosses_execution_boundary() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    let workspace = IsolatedWorkspace::new("provisional-program-speculation");
    let calls = std::sync::Arc::new(AtomicUsize::new(0));
    let called = std::sync::Arc::new(tokio::sync::Notify::new());
    let advertised_program = std::sync::Arc::new(AtomicBool::new(false));
    let source = r#"tool("read", #{uri: "mcp://probe/file:///sentinel"})"#;
    let provider = ProvisionalProgramProvider {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "provisional-program".into(),
                    name: "run_program_not_admitted".into(),
                    arguments: serde_json::json!({"source": source}).to_string(),
                }],
                1,
                1,
            ),
            completion(vec![Content::Text("No program was executed.".into())], 1, 1),
        ]),
        emitted_delta: AtomicBool::new(false),
        advertised_program: advertised_program.clone(),
        speculative_call_seen: called.clone(),
    };
    let mut cfg = workspace.config();
    cfg.memory.tool_set = ToolSet::Full;
    cfg.program.mode = ProgramMode::Auto;
    cfg.program.speculative_ptc = true;
    cfg.loop_limits.max_steps = 2;
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    agent.attach_mcp(std::sync::Arc::new(CountingResourceMcp {
        calls: calls.clone(),
        called,
    }));

    let _ = agent
        .run_turn(
            "Inspect the available sentinel and summarize it.",
            &mut RecUi::default(),
        )
        .await;

    assert!(
        advertised_program.load(Ordering::Relaxed),
        "the fixture must advertise run_program so the zero-call result is meaningful"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "a provisional delta must not launch work before the final outer call is validated"
    );
}
