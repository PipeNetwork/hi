use super::common::*;
use super::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use hi_workspace::{
    AdmissionDenied, BarrierKind, BarrierReceipt, ExecutionReport, JobId, JobPermit,
    JobSealOutcome, JobSpec, JobTerminal, MutationIntent, MutationPermit, RecoveryId,
    RecoveryOutcome, SettlementOutcome, WorkspaceBinding, WorkspaceCapabilities,
    WorkspaceController, WorkspaceStatus,
};

#[derive(Clone)]
struct ObservedRequest {
    tools: Vec<String>,
    envelope: hi_ai::RequestToolEnvelope,
}

struct RepeatingForbiddenBash {
    requests: Arc<Mutex<Vec<ObservedRequest>>>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for RepeatingForbiddenBash {
    async fn stream(
        &self,
        request: hi_ai::ChatRequest,
        _sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        let mut requests = self.requests.lock().unwrap();
        let call_index = requests.len() + 1;
        requests.push(ObservedRequest {
            tools: request.tools.iter().map(|tool| tool.name.clone()).collect(),
            envelope: request
                .tool_envelope
                .as_deref()
                .cloned()
                .expect("every provider request must carry a sealed envelope"),
        });
        drop(requests);

        Ok(completion(
            vec![Content::ToolCall {
                id: format!("forbidden-bash-{call_index}"),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": "printf compromised > should-not-exist"
                })
                .to_string(),
            }],
            1,
            1,
        ))
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        let mut capabilities = test_provider_capabilities();
        capabilities.tool_choice.required = true;
        capabilities
    }
}

struct AdmissionCountingController {
    inner: Arc<hi_workspace::InMemoryWorkspaceController>,
    begins: Arc<AtomicUsize>,
    intents: Arc<Mutex<Vec<MutationIntent>>>,
}

#[async_trait::async_trait]
impl WorkspaceController for AdmissionCountingController {
    fn binding(&self) -> WorkspaceBinding {
        self.inner.binding()
    }

    fn capabilities(&self) -> WorkspaceCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> WorkspaceStatus {
        self.inner.status()
    }

    fn subscribe(&self) -> tokio::sync::watch::Receiver<WorkspaceStatus> {
        self.inner.subscribe()
    }

    async fn begin(&self, intent: MutationIntent) -> Result<MutationPermit, AdmissionDenied> {
        self.begins.fetch_add(1, Ordering::SeqCst);
        self.intents.lock().unwrap().push(intent.clone());
        self.inner.begin(intent).await
    }

    async fn settle(
        &self,
        permit: MutationPermit,
        execution: ExecutionReport,
    ) -> SettlementOutcome {
        self.inner.settle(permit, execution).await
    }

    async fn register_job(&self, spec: JobSpec) -> Result<JobPermit, AdmissionDenied> {
        self.inner.register_job(spec).await
    }

    async fn seal_job(&self, job: JobId, terminal: JobTerminal) -> JobSealOutcome {
        self.inner.seal_job(job, terminal).await
    }

    async fn barrier(&self, reason: BarrierKind, deadline: Instant) -> BarrierReceipt {
        self.inner.barrier(reason, deadline).await
    }

    async fn reconcile(&self, recovery: RecoveryId) -> RecoveryOutcome {
        self.inner.reconcile(recovery).await
    }
}

#[tokio::test]
async fn repeated_unadvertised_bash_exhausts_one_retry_without_fallback_or_admission() {
    let workspace = IsolatedWorkspace::new("persistent-unavailable-tool");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn inspect_me() {}\n").unwrap();
    let marker = workspace.path("should-not-exist");
    let mut config = workspace.config();
    config.memory.tool_set = ToolSet::Dynamic;
    config.loop_limits.max_steps = 8;
    config.loop_limits.max_repeat_nudges = 4;
    config.gates.review = ReviewPolicy::Off;

    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = RepeatingForbiddenBash {
        requests: requests.clone(),
    };
    let begins = Arc::new(AtomicUsize::new(0));
    let intents = Arc::new(Mutex::new(Vec::new()));
    let inner = Arc::new(hi_workspace::InMemoryWorkspaceController::new_local(
        "unavailable-tool-workspace",
        config.paths.workspace_root.clone(),
        config.paths.state_root.clone(),
    ));
    let original_binding = inner.binding();
    let controller = Arc::new(AdmissionCountingController {
        inner,
        begins: begins.clone(),
        intents: intents.clone(),
    });
    let mut agent = Agent::new(Arc::new(provider), config).unwrap();
    agent
        .install_workspace_controller(controller.clone())
        .unwrap();
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("Review src/lib.rs and report one bounded finding.", &mut ui)
        .await;

    let admitted_intents = intents.lock().unwrap();
    assert_eq!(begins.load(Ordering::SeqCst), admitted_intents.len());
    assert!(
        admitted_intents
            .iter()
            .all(MutationIntent::is_reconciliation),
        "a rejected call must not receive workspace mutation admission: {admitted_intents:?}"
    );
    let final_binding = controller.binding();
    assert_eq!(final_binding.controller_id, original_binding.controller_id);
    assert_eq!(final_binding.binding_id, original_binding.binding_id);
    assert_eq!(final_binding.epoch, original_binding.epoch);
    assert_eq!(
        final_binding.workspace_root,
        original_binding.workspace_root
    );
    assert!(controller.status().active_operation.is_none());
    assert!(!marker.exists(), "the forbidden shell command must not run");
    assert_eq!(
        std::fs::read_to_string(workspace.path("src/lib.rs")).unwrap(),
        "pub fn inspect_me() {}\n"
    );

    let requests = requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        2,
        "one initial call plus one correction; statuses={:?}; results={:?}; fallbacks={:?}",
        ui.statuses,
        ui.tool_results,
        requests
            .iter()
            .map(|request| request.envelope.requests_text_tool_fallback())
            .collect::<Vec<_>>()
    );
    for request in requests.iter() {
        assert!(!request.tools.iter().any(|tool| tool == "bash"));
        assert!(!request.envelope.requests_text_tool_fallback());
    }
    assert_eq!(
        ui.tool_results
            .iter()
            .filter(|(name, result)| {
                name == "bash" && result.contains("\"reason\":\"unavailable_tool\"")
            })
            .count(),
        2,
        "both forbidden calls must receive typed sealed-envelope denials"
    );
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|status| status.contains("retrying once with the admitted tool list"))
            .count(),
        1,
        "the unavailable-tool correction budget is exactly one"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| { status.contains("kept calling tools outside the sealed envelope") })
    );
    assert!(!ui.statuses.iter().any(|status| {
        status.contains("schema-corrected")
            || status.contains("DeepSeek tool arguments")
            || status.contains("trying one plain-text tool call")
    }));
    let transcript = agent
        .messages()
        .iter()
        .map(hi_ai::Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!transcript.contains("For this response only, emit exactly one plain-text call"));
    assert!(!transcript.contains("<tool_call>"));

    let outcome = outcome.expect("protocol exhaustion should settle as a typed failed outcome");
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::InfrastructureFailure);
    assert!(outcome.changed_files.is_empty());
}
