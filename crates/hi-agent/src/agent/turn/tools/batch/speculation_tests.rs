use super::*;

use crate::agent::turn::model_request::ToolEnvelopeRequestLimits;
use crate::tests::common::{Canned, IsolatedWorkspace, RecUi};
use crate::ui::NullUi;
use crate::{Agent, ProgramMode};
use hi_ai::{CapabilityRoute, EffectiveProviderCapabilities, ProviderCapabilities, ToolMode};
use hi_tools::envelope::{ToolEnvelope, ToolEnvelopeContext};
use hi_workspace::WorkspaceController;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Default)]
struct RecordingDurability {
    process_states: Mutex<Vec<(String, bool)>>,
}

#[async_trait::async_trait]
impl crate::WorkspaceDurability for RecordingDurability {
    async fn mutation_started(&self, _dirty_paths: Option<Vec<String>>) -> anyhow::Result<()> {
        Ok(())
    }

    async fn checkpoint(&self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn background_process_state(&self, id: &str, running: bool) -> anyhow::Result<()> {
        self.process_states
            .lock()
            .unwrap()
            .push((id.to_string(), running));
        Ok(())
    }
}

fn agent(workspace: &IsolatedWorkspace, max_tool_calls: u32) -> Agent {
    let mut config = workspace.config();
    config.program.mode = ProgramMode::Auto;
    config.program.speculative_ptc = true;
    config.loop_limits.max_tool_calls = max_tool_calls;
    Agent::new(Arc::new(Canned(Mutex::new(Vec::new()))), config).unwrap()
}

fn envelope(agent: &Agent, mode: ToolMode) -> Arc<hi_tools::envelope::ToolEnvelope> {
    envelope_with_argument_limit(agent, mode, 64 * 1024)
}

fn envelope_with_argument_limit(
    agent: &Agent,
    mode: ToolMode,
    max_tool_argument_bytes: u32,
) -> Arc<hi_tools::envelope::ToolEnvelope> {
    envelope_with_limits(agent, mode, max_tool_argument_bytes, 0)
}

fn envelope_with_limits(
    agent: &Agent,
    mode: ToolMode,
    max_tool_argument_bytes: u32,
    executed_tool_calls: u32,
) -> Arc<hi_tools::envelope::ToolEnvelope> {
    agent.build_tool_envelope(
        &[hi_tools::run_program_tool_spec()],
        mode,
        mode,
        ToolEnvelopeRequestLimits {
            max_output_tokens: 100,
            executed_tool_calls,
            max_parallel_calls: 4,
            max_tool_argument_bytes,
            text_tool_fallback: false,
        },
        EffectiveProviderCapabilities::conservative(
            CapabilityRoute::new("test", "streaming-tools"),
            ProviderCapabilities::native_tools(true),
        ),
    )
}

async fn execute_program_for_test(
    agent: &mut Agent,
    envelope: &ToolEnvelope,
    source: &str,
    initial_tool_calls: u32,
    ui: &mut RecUi,
) -> anyhow::Result<(ToolBatchOutcome, u32)> {
    let id = "program".to_string();
    let arguments = serde_json::json!({"source": source}).to_string();
    let calls = vec![(id.clone(), "run_program".to_string(), arguments.clone())];
    let mut completion = vec![Content::ToolCall {
        id,
        name: "run_program".into(),
        arguments,
    }];
    let specs = vec![hi_tools::run_program_tool_spec()];
    let mut progress = ProgressTracker::default();
    let mut timeline = ToolTimeline::default();
    let mut tool_calls = initial_tool_calls;
    let mut max_concurrent = 0;
    let mut serial_runs = 0;
    let mut fallback_next = false;
    let mut fallback_used = false;
    let outcome = agent
        .execute_program_batch(
            &calls,
            &mut completion,
            &specs,
            envelope,
            None,
            &mut EvidenceTracker::default(),
            &mut ImplementationTracker::default(),
            &mut progress,
            &mut timeline,
            &mut tool_calls,
            &mut max_concurrent,
            &mut serial_runs,
            &registry(),
            &mut fallback_next,
            &mut fallback_used,
            ui,
        )
        .await?;
    Ok((outcome, tool_calls))
}

fn pipefs_controller(
    workspace: &IsolatedWorkspace,
) -> Arc<hi_workspace::InMemoryWorkspaceController> {
    Arc::new(hi_workspace::InMemoryWorkspaceController::new_pipefs(
        "program-workspace",
        "program-session",
        2,
        true,
        workspace.path("."),
        workspace.path(".state"),
    ))
}

#[tokio::test]
async fn dry_run_program_executes_no_bytes_and_requests_no_admission() {
    let workspace = IsolatedWorkspace::new("dry-run-program-preflight");
    let mut agent = agent(&workspace, 8);
    agent.config.gates.dry_run = true;
    let controller = pipefs_controller(&workspace);
    agent
        .install_workspace_controller(controller.clone())
        .unwrap();
    let envelope = envelope(&agent, ToolMode::Auto);
    let existing = controller
        .begin(hi_workspace::MutationIntent::workspace("existing writer"))
        .await
        .unwrap();
    let active_operation = controller.status().active_operation;
    let mut ui = RecUi::default();

    let (_, calls) = execute_program_for_test(
        &mut agent,
        &envelope,
        r#"tool("write", #{path: "forbidden.txt", content: "no"})"#,
        0,
        &mut ui,
    )
    .await
    .unwrap();

    assert_eq!(calls, 1);
    assert!(!workspace.path("forbidden.txt").exists());
    assert_eq!(controller.status().active_operation, active_operation);
    assert!(ui.tool_results.iter().any(|(name, result)| {
        name == "run_program" && result.contains("[dry-run]") && result.contains("not executed")
    }));
    assert!(
        controller
            .settle(existing, hi_workspace::ExecutionReport::succeeded(None))
            .await
            .receipt
            .is_some()
    );
}

#[tokio::test]
async fn exhausted_hard_budget_denies_program_before_rhai_or_admission() {
    let workspace = IsolatedWorkspace::new("budget-program-preflight");
    let mut agent = agent(&workspace, 1);
    let controller = pipefs_controller(&workspace);
    agent
        .install_workspace_controller(controller.clone())
        .unwrap();
    let envelope = envelope(&agent, ToolMode::Auto);
    let existing = controller
        .begin(hi_workspace::MutationIntent::workspace("existing writer"))
        .await
        .unwrap();
    let active_operation = controller.status().active_operation;
    let mut ui = RecUi::default();

    let (_, calls) = execute_program_for_test(
        &mut agent,
        &envelope,
        r#"tool("write", #{path: "forbidden.txt", content: "no"})"#,
        1,
        &mut ui,
    )
    .await
    .unwrap();

    assert_eq!(calls, 1, "a denied call must not consume budget twice");
    assert!(!workspace.path("forbidden.txt").exists());
    assert_eq!(controller.status().active_operation, active_operation);
    assert!(ui.tool_results.iter().any(|(name, result)| {
        name == "run_program" && result.contains("tool_budget_exhausted")
    }));
    assert!(
        controller
            .settle(existing, hi_workspace::ExecutionReport::succeeded(None))
            .await
            .receipt
            .is_some()
    );
}

#[tokio::test]
async fn real_program_host_obeys_the_smaller_sealed_nested_call_budget() {
    let workspace = IsolatedWorkspace::new("sealed-program-call-budget");
    std::fs::write(workspace.path("first.txt"), "first").unwrap();
    std::fs::write(workspace.path("second.txt"), "second").unwrap();
    let mut agent = agent(&workspace, 8);
    let envelope = envelope_with_limits(&agent, ToolMode::Auto, 64 * 1024, 6);
    assert_eq!(envelope.payload.limits.max_calls_per_round, 2);
    let mut ui = RecUi::default();

    let (_, calls) = execute_program_for_test(
        &mut agent,
        &envelope,
        r#"
            tool("read", #{path: "first.txt"});
            tool("read", #{path: "second.txt"});
        "#,
        0,
        &mut ui,
    )
    .await
    .unwrap();

    assert_eq!(
        calls, 2,
        "one outer and one nested call consumed budget: {:?}",
        ui.tool_results
    );
    assert_eq!(
        ui.tool_results
            .iter()
            .filter(|(name, _)| name == "read")
            .count(),
        1,
        "the second nested call must not cross the host boundary"
    );
    assert!(ui.tool_results.iter().any(|(name, result)| {
        name == "run_program" && result.contains("remaining budget of 1 tool calls")
    }));
}

#[tokio::test]
async fn terminal_nested_poll_observes_and_settles_its_live_writer_job() {
    let workspace = IsolatedWorkspace::new("terminal-nested-poll");
    let mut agent = agent(&workspace, 8);
    let durability = Arc::new(RecordingDurability::default());
    agent.set_workspace_durability(Some(durability.clone()));
    let mut ignore = |_: &str| {};
    let started = execute_streaming_in_runtime_with_runner(
        agent.runtime.process_runner(),
        agent.runtime.root(),
        agent.runtime.state_root(),
        &agent.runtime.lsp(),
        agent.runtime.background(),
        agent.runtime.read_cache(),
        agent.runtime.repo_map(),
        "bash",
        r#"{"command":"printf terminal > terminal.txt","run_in_background":true}"#,
        &mut ignore,
    )
    .await;
    let started_background = started.background.as_ref().expect("background process");
    let handle = started_background.id.clone();
    agent
        .observe_durable_background_process(started_background)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if agent
                .runtime
                .background()
                .pending_job_settlements()
                .await
                .iter()
                .any(|job| job.handle == handle)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("writer reaches durability-pending");
    let before = agent.workspace_controller_binding().version;
    let base = envelope(&agent, ToolMode::ReadOnly);
    let bash_output = hi_tools::TOOL_SPECS
        .iter()
        .find(|tool| tool.name == "bash_output")
        .unwrap()
        .clone();
    let minimal = ToolEnvelope::build_with_program_tools(
        &[hi_tools::run_program_tool_spec()],
        &[bash_output],
        ToolEnvelopeContext {
            provider: base.payload.provider.clone(),
            workspace: base.payload.workspace.clone(),
            trust: base.payload.trust.clone(),
            permissions: base.payload.permissions.clone(),
            limits: base.payload.limits.clone(),
            tool_mode: ToolMode::ReadOnly,
            execution_mode: ToolMode::ReadOnly,
            tool_versions: Default::default(),
        },
    );
    let mut ui = RecUi::default();

    execute_program_for_test(
        &mut agent,
        &minimal,
        &format!(r#"tool("bash_output", #{{id: "{handle}", wait_secs: 0}})"#),
        0,
        &mut ui,
    )
    .await
    .unwrap();

    assert!(
        agent
            .runtime
            .background()
            .pending_job_settlements()
            .await
            .is_empty(),
        "the writer job must not publish before workspace settlement; tools={:?}; durability={:?}; status={:?}",
        ui.tool_results,
        durability.process_states.lock().unwrap(),
        agent.workspace_controller_status(),
    );
    assert_ne!(agent.workspace_controller_binding().version, before);
    assert!(
        durability
            .process_states
            .lock()
            .unwrap()
            .iter()
            .any(|(id, running)| id == &handle && !running)
    );
    assert_eq!(
        std::fs::read_to_string(workspace.path("terminal.txt")).unwrap(),
        "terminal"
    );
}

#[tokio::test]
async fn dynamically_built_nested_arguments_obey_the_sealed_byte_limit() {
    let workspace = IsolatedWorkspace::new("program-nested-argument-limit");
    let mut agent = agent(&workspace, 8);
    let envelope = envelope_with_argument_limit(&agent, ToolMode::Auto, 32);
    let program_specs = envelope.program_specs();
    let call = ProgramCall {
        occurrence: 0,
        name: "read".into(),
        arguments: serde_json::json!({"path": "x".repeat(128)}),
    };
    let mut ui = NullUi;

    let (_, output) = agent
        .authorize_program_call(&call, &program_specs, &envelope, &mut ui)
        .await
        .expect("authorization persistence succeeds")
        .expect("oversized nested call must be denied before execution");

    assert_eq!(output.status, hi_tools::ToolStatus::Denied);
    assert!(output.content.contains("client tool-argument size limit"));
}

#[tokio::test]
async fn read_only_programs_cannot_admit_or_execute_mutating_nested_tools() {
    let workspace = IsolatedWorkspace::new("read-only-program-tools");
    let mut agent = agent(&workspace, 8);
    let read_only = envelope(&agent, ToolMode::ReadOnly);
    assert!(read_only.admits_program("read"));
    assert!(!read_only.admits_program("write"));
    assert!(!read_only.admits_program("bash"));

    // Defense in depth: even a valid envelope produced by an older/broken
    // sealer must not let the host cross the request's read-only boundary.
    let write = hi_tools::TOOL_SPECS
        .iter()
        .find(|tool| tool.name == "write")
        .expect("write tool spec")
        .clone();
    let legacy = ToolEnvelope::build_with_program_tools(
        &[hi_tools::run_program_tool_spec()],
        std::slice::from_ref(&write),
        ToolEnvelopeContext {
            provider: read_only.payload.provider.clone(),
            workspace: read_only.payload.workspace.clone(),
            trust: read_only.payload.trust.clone(),
            permissions: read_only.payload.permissions.clone(),
            limits: read_only.payload.limits.clone(),
            tool_mode: ToolMode::ReadOnly,
            execution_mode: ToolMode::ReadOnly,
            tool_versions: Default::default(),
        },
    );
    let call = ProgramCall {
        occurrence: 0,
        name: "write".into(),
        arguments: serde_json::json!({"path": "forbidden.txt", "content": "no"}),
    };
    let mut ui = NullUi;
    let (_, output) = agent
        .authorize_program_call(&call, &legacy.program_specs(), &legacy, &mut ui)
        .await
        .expect("authorization persistence succeeds")
        .expect("read-only nested write must be denied");
    assert_eq!(output.status, hi_tools::ToolStatus::Denied);
    assert!(output.content.contains("read-only"));
    assert!(!workspace.path("forbidden.txt").exists());
}

fn registry() -> SpeculationRegistry {
    SpeculationRegistry::new(8, 2, Duration::from_secs(30))
}

#[test]
fn speculation_enablement_uses_the_sealed_provider_capability() {
    let workspace = IsolatedWorkspace::new("sealed-streaming-program-speculation");
    let agent = agent(&workspace, 8);
    assert!(!agent.provider.capabilities().streamed_tool_call_deltas);
    let sealed = envelope(&agent, ToolMode::Auto);
    assert!(
        sealed
            .payload
            .provider
            .capability_record
            .capabilities
            .streamed_tool_call_deltas
    );

    assert!(agent.program_speculator(&sealed).enabled);
}

#[tokio::test]
async fn external_speculation_reports_that_an_effect_may_have_started() {
    let workspace = IsolatedWorkspace::new("external-program-speculation-evidence");
    let agent = agent(&workspace, 8);
    let base = envelope(&agent, ToolMode::Auto);
    let web_fetch = hi_tools::TOOL_SPECS
        .iter()
        .find(|tool| tool.name == "web_fetch")
        .unwrap()
        .clone();
    let sealed = ToolEnvelope::build_with_program_tools(
        &[hi_tools::run_program_tool_spec()],
        &[web_fetch],
        ToolEnvelopeContext {
            provider: base.payload.provider.clone(),
            workspace: base.payload.workspace.clone(),
            trust: base.payload.trust.clone(),
            permissions: base.payload.permissions.clone(),
            limits: base.payload.limits.clone(),
            tool_mode: ToolMode::Auto,
            execution_mode: ToolMode::Auto,
            tool_versions: Default::default(),
        },
    );
    let mut speculator = agent.program_speculator(&sealed);
    speculator.external_allowed = true;
    let registry = registry();

    assert!(speculator.launch(
        &registry,
        "program",
        r#"tool("web_fetch", #{url: "http://127.0.0.1:1/"})"#,
    ));
    registry.cancel_all();
}

#[tokio::test]
async fn auto_envelope_launches_a_valid_nested_read() {
    let workspace = IsolatedWorkspace::new("valid-program-speculation");
    std::fs::write(workspace.path("sentinel.txt"), "sentinel").unwrap();
    let agent = agent(&workspace, 8);
    let mut speculator = agent.program_speculator(&envelope(&agent, ToolMode::Auto));
    speculator.enabled = true;
    let registry = registry();

    speculator.launch(
        &registry,
        "program",
        r#"tool("read", #{path: "sentinel.txt"})"#,
    );

    assert_eq!(registry.telemetry().launched, 1);
    registry.cancel_all();
}

#[test]
fn chat_only_envelope_launches_no_speculative_calls() {
    let workspace = IsolatedWorkspace::new("chat-only-program-speculation");
    let agent = agent(&workspace, 8);
    let mut speculator = agent.program_speculator(&envelope(&agent, ToolMode::ChatOnly));
    // Isolate the envelope guard from the redundant construction-time guard.
    speculator.enabled = true;
    let registry = registry();

    speculator.launch(
        &registry,
        "program",
        r#"tool("read", #{path: "sentinel.txt"})"#,
    );

    assert_eq!(registry.telemetry().launched, 0);
}

#[test]
fn invalid_nested_arguments_launch_no_speculative_calls() {
    let workspace = IsolatedWorkspace::new("invalid-nested-program-speculation");
    let agent = agent(&workspace, 8);
    let mut speculator = agent.program_speculator(&envelope(&agent, ToolMode::Auto));
    speculator.enabled = true;
    let registry = registry();

    speculator.launch(
        &registry,
        "program",
        r#"tool("read", #{path: "sentinel.txt", unexpected: true})"#,
    );

    assert_eq!(registry.telemetry().launched, 0);
}

#[test]
fn background_polls_launch_no_speculative_calls() {
    let workspace = IsolatedWorkspace::new("background-program-speculation");
    let agent = agent(&workspace, 8);
    let mut speculator = agent.program_speculator(&envelope(&agent, ToolMode::Auto));
    speculator.enabled = true;
    let registry = registry();

    speculator.launch(
        &registry,
        "program",
        r#"tool("bash_output", #{id: "writer_1", wait_secs: 0})"#,
    );

    assert_eq!(registry.telemetry().launched, 0);
}

#[test]
fn non_executable_rhai_text_launches_no_speculative_calls() {
    let workspace = IsolatedWorkspace::new("lexically-ambiguous-program-speculation");
    let agent = agent(&workspace, 8);
    let mut speculator = agent.program_speculator(&envelope(&agent, ToolMode::Auto));
    speculator.enabled = true;

    for source in [
        r#"let text = `tool("read", #{path: "sentinel.txt"})`; text"#,
        "let text = #\"tool(\"read\", #{path: \"sentinel.txt\"})\"#; text",
        r#"/* outer /* nested */ tool("read", #{path: "sentinel.txt"}) */ 0"#,
        r#"obj . tool("read", #{path: "sentinel.txt"})"#,
        r#"namespace :: tool("read", #{path: "sentinel.txt"})"#,
        r#"throw("stop"); tool("read", #{path: "sentinel.txt"})"#,
    ] {
        let registry = registry();
        speculator.launch(&registry, "program", source);
        assert_eq!(registry.telemetry().launched, 0, "source: {source}");
    }
}

#[test]
fn outer_program_reserves_the_last_sealed_call_slot() {
    let workspace = IsolatedWorkspace::new("program-speculation-call-budget");
    let agent = agent(&workspace, 1);
    let mut speculator = agent.program_speculator(&envelope(&agent, ToolMode::Auto));
    speculator.enabled = true;
    let registry = registry();

    assert_eq!(speculator.max_calls, 0);
    speculator.launch(
        &registry,
        "program",
        r#"tool("read", #{path: "sentinel.txt"})"#,
    );
    assert_eq!(registry.telemetry().launched, 0);
}

#[tokio::test]
async fn program_progress_comes_from_real_nested_evidence_only() {
    let workspace = IsolatedWorkspace::new("program-progress-evidence");
    std::fs::write(workspace.path("input.txt"), "evidence").unwrap();
    let mut agent = agent(&workspace, 8);
    let envelope = envelope(&agent, ToolMode::Auto);
    let mut ui = RecUi::default();
    let (empty, _) =
        execute_program_for_test(&mut agent, &envelope, "let answer = 42;", 0, &mut ui)
            .await
            .unwrap();
    assert!(
        empty
            .tool_progress_labels
            .iter()
            .all(|label| label.kind != ProgressKind::Meaningful)
    );
    let (reads, _) = execute_program_for_test(
        &mut agent,
        &envelope,
        r#"tool("read", #{path: "input.txt"}); tool("read", #{path: "input.txt"});"#,
        0,
        &mut ui,
    )
    .await
    .unwrap();
    assert_eq!(
        reads
            .tool_progress_labels
            .iter()
            .filter(|label| label.kind == ProgressKind::Meaningful)
            .count(),
        1
    );
    assert_eq!(
        reads.tool_progress_labels.len(),
        3,
        "two physical reads and one envelope are each observed exactly once"
    );
}
