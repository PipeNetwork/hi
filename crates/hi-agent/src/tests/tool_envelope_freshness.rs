use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hi_ai::{ChatRequest, Completion, Content, Provider, ProviderCapabilities, StreamEvent};
use hi_workspace::{
    AdmissionDenied, BarrierKind, BarrierReceipt, ExecutionReport, JobId, JobPermit,
    JobSealOutcome, JobSpec, JobTerminal, MutationIntent, MutationPermit, RecoveryId,
    RecoveryOutcome, SettlementOutcome, WorkspaceBinding, WorkspaceCapabilities,
    WorkspaceController, WorkspaceStatus,
};

use super::common::{IsolatedWorkspace, RecUi, completion, write_content_completion};
use crate::{Agent, ToolSet};

#[derive(Clone, Copy)]
enum WorkspaceDrift {
    Rebind,
    VersionAdvance,
    AdmissionRebind,
    AdmissionVersionAdvance,
}

struct AdmissionDriftController {
    inner: Arc<hi_workspace::InMemoryWorkspaceController>,
    drift: WorkspaceDrift,
    drifted: AtomicBool,
}

#[async_trait::async_trait]
impl WorkspaceController for AdmissionDriftController {
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
        if !self.drifted.swap(true, Ordering::SeqCst) {
            match self.drift {
                WorkspaceDrift::AdmissionRebind => {
                    let current = self.inner.binding();
                    self.inner
                        .rebind(current.workspace_root, current.state_root)?;
                }
                WorkspaceDrift::AdmissionVersionAdvance => {
                    let permit = self
                        .inner
                        .begin(MutationIntent::workspace("admission-boundary settlement"))
                        .await?;
                    let outcome = self
                        .inner
                        .settle(
                            permit,
                            ExecutionReport::succeeded(Some(
                                "blake3:admission-boundary-version".into(),
                            )),
                        )
                        .await;
                    assert!(outcome.receipt.is_some(), "{outcome:?}");
                }
                WorkspaceDrift::Rebind | WorkspaceDrift::VersionAdvance => {}
            }
        }
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

struct WorkspaceDriftProbe {
    controller: Arc<hi_workspace::InMemoryWorkspaceController>,
    drift: WorkspaceDrift,
    drifted: AtomicBool,
    attachments: Arc<Mutex<Vec<hi_ai::RequestToolEnvelope>>>,
}

#[async_trait::async_trait]
impl Provider for WorkspaceDriftProbe {
    async fn stream(
        &self,
        request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        self.attachments.lock().unwrap().push(
            request
                .tool_envelope
                .as_deref()
                .cloned()
                .expect("attached envelope"),
        );
        if !self.drifted.swap(true, Ordering::SeqCst) {
            match self.drift {
                WorkspaceDrift::Rebind => {
                    let current = self.controller.binding();
                    self.controller
                        .rebind(current.workspace_root, current.state_root)
                        .expect("test controller rebind");
                }
                WorkspaceDrift::VersionAdvance => {
                    let permit = self
                        .controller
                        .begin(hi_workspace::MutationIntent::workspace(
                            "intervening test settlement",
                        ))
                        .await
                        .expect("intervening admission");
                    let settled = self
                        .controller
                        .settle(
                            permit,
                            hi_workspace::ExecutionReport::succeeded(Some(
                                "blake3:intervening-version".into(),
                            )),
                        )
                        .await;
                    assert!(settled.receipt.is_some(), "{settled:?}");
                }
                WorkspaceDrift::AdmissionRebind | WorkspaceDrift::AdmissionVersionAdvance => {}
            }
            return Ok(write_content_completion("stale-write.txt", "must not land"));
        }
        Ok(completion(
            vec![Content::Text(
                "The rejected call did not run; the workspace must be inspected again.".into(),
            )],
            1,
            1,
        ))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        let mut capabilities = ProviderCapabilities::native_tools(false);
        capabilities.parallel_tool_calls = true;
        capabilities.tool_choice.automatic = true;
        capabilities.tool_choice.required = false;
        capabilities.actual_model_revision = Some("freshness-probe@2026-09-04".into());
        capabilities
    }
}

async fn assert_workspace_drift_reseals_without_admission(drift: WorkspaceDrift, tag: &str) {
    let workspace = IsolatedWorkspace::new(tag);
    let mut config = workspace.config();
    config.memory.tool_set = ToolSet::Full;
    config.loop_limits.max_steps = 2;
    config.gates.allow_unverified = true;
    config.gates.proactive_verify = false;
    config.memory.finalize = false;
    let controller = Arc::new(hi_workspace::InMemoryWorkspaceController::new_local(
        format!("{tag}-workspace"),
        config.paths.workspace_root.clone(),
        config.paths.state_root.clone(),
    ));
    let attachments = Arc::new(Mutex::new(Vec::new()));
    let provider = WorkspaceDriftProbe {
        controller: controller.clone(),
        drift,
        drifted: AtomicBool::new(false),
        attachments: attachments.clone(),
    };
    let mut agent = Agent::new(Arc::new(provider), config).unwrap();
    let installed: Arc<dyn WorkspaceController> = match drift {
        WorkspaceDrift::AdmissionRebind | WorkspaceDrift::AdmissionVersionAdvance => {
            Arc::new(AdmissionDriftController {
                inner: controller.clone(),
                drift,
                drifted: AtomicBool::new(false),
            })
        }
        WorkspaceDrift::Rebind | WorkspaceDrift::VersionAdvance => controller.clone(),
    };
    agent.install_workspace_controller(installed).unwrap();
    let mut ui = RecUi::default();

    let _ = agent
        .run_turn(
            "Create stale-write.txt with the requested contents.",
            &mut ui,
        )
        .await;

    assert!(
        !workspace.path("stale-write.txt").exists(),
        "a call from a stale request must not mutate the workspace"
    );
    assert_eq!(
        controller.status().active_operation,
        None,
        "a stale call must not receive a mutation permit"
    );
    assert_eq!(
        controller.status().state,
        hi_workspace::WorkspaceState::Ready
    );
    let denial = ui
        .tool_results
        .iter()
        .find(|(name, _)| name == "write")
        .map(|(_, result)| serde_json::from_str::<serde_json::Value>(result).unwrap())
        .unwrap_or_else(|| panic!("stale-envelope denial missing: {:?}", ui.tool_results));
    assert_eq!(denial["error"]["kind"], "tool_protocol_error");
    assert_eq!(denial["error"]["reason"], "stale_workspace");
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("workspace changed after the request was sealed"))
    );
    let attachments = attachments.lock().unwrap();
    assert!(
        attachments.len() >= 2,
        "stale execution must trigger a newly sealed provider request: {attachments:?}"
    );
    assert_ne!(
        attachments[0].payload["workspace"], attachments[1].payload["workspace"],
        "the retry must carry the current workspace binding"
    );
}

#[tokio::test]
async fn rebind_after_provider_request_rejects_stale_epoch_before_admission() {
    assert_workspace_drift_reseals_without_admission(
        WorkspaceDrift::Rebind,
        "tool-envelope-stale-rebind",
    )
    .await;
}

#[tokio::test]
async fn version_advance_after_provider_request_rejects_stale_call_before_admission() {
    assert_workspace_drift_reseals_without_admission(
        WorkspaceDrift::VersionAdvance,
        "tool-envelope-stale-version",
    )
    .await;
}

#[tokio::test]
async fn rebind_inside_controller_begin_rejects_issued_stale_permit_before_effect() {
    assert_workspace_drift_reseals_without_admission(
        WorkspaceDrift::AdmissionRebind,
        "tool-envelope-admission-rebind",
    )
    .await;
}

#[tokio::test]
async fn version_advance_inside_controller_begin_rejects_issued_stale_permit_before_effect() {
    assert_workspace_drift_reseals_without_admission(
        WorkspaceDrift::AdmissionVersionAdvance,
        "tool-envelope-admission-version",
    )
    .await;
}
