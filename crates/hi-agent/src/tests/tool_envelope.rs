use super::common::*;
use super::*;

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
        .map(|(_, result)| result)
        .expect("outside-envelope call receives a typed result");
    assert!(denial.contains("sealed envelope"), "{denial}");
    assert!(denial.contains(r#""reason":"unavailable_tool""#));
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
        attachments,
        capabilities: envelope_capabilities(),
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
async fn chat_only_envelope_rejects_a_text_promoted_write() {
    let workspace = IsolatedWorkspace::new("chat-only-text-tool-envelope");
    let destination = workspace.path("should-not-exist.txt");
    let textual_call = format!(
        "<tool_call>write<arg_key>path</arg_key><arg_value>{}</arg_value><arg_key>content</arg_key><arg_value>blocked</arg_value></tool_call>",
        destination.display()
    );
    let advertised = std::sync::Arc::new(Mutex::new(Vec::new()));
    let attachments = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = EnvelopeProbe {
        responses: Mutex::new(vec![
            completion(vec![Content::Text(textual_call)], 1, 1),
            completion(
                vec![Content::Text("No workspace changes were made.".into())],
                1,
                1,
            ),
        ]),
        advertised: advertised.clone(),
        attachments,
        capabilities: hi_ai::ProviderCapabilities::default(),
    };
    let mut cfg = workspace.config();
    cfg.loop_limits.max_steps = 2;
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();

    let _ = agent
        .run_turn(
            &format!("Create {} containing blocked.", destination.display()),
            &mut ui,
        )
        .await;

    assert!(!destination.exists());
    let denial = ui
        .tool_results
        .iter()
        .find(|(name, _)| name == "write")
        .map(|(_, result)| result)
        .expect("text-promoted call receives a typed denial");
    assert!(
        denial.contains("chat-only") || denial.contains("sealed envelope"),
        "{denial}"
    );
    assert!(denial.contains("envelope mode is chat_only"), "{denial}");
    assert!(
        advertised
            .lock()
            .unwrap()
            .iter()
            .all(|tools| !tools.is_empty()),
        "schemas may remain attached for audit/cache even though ChatOnly admits none"
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("schema-corrected")
                || status.contains("DeepSeek tool arguments")
                || status.contains("plain-text tool call")),
        "ChatOnly denial must exit tool-free instead of changing call formats: {:?}",
        ui.statuses
    );
    let transcript = agent
        .messages()
        .iter()
        .map(hi_ai::Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(transcript.contains("No tools are admitted in this request"));
    assert!(!transcript.contains("Emit a new `write` call"));
}
