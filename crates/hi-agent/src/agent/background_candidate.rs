//! Detached execution and sealing for general-purpose background tasks.

use std::sync::Arc;
#[cfg(test)]
use std::time::Duration;

use anyhow::Context as _;
use hi_ai::{Provider, ToolMode};
use hi_workspace::{
    CandidateRoute, CandidateVerification, EffectScope, ExecutionDisposition, ExecutionReport,
    JobId, MutationIntent, ReplayClass, WorkspaceBinding,
};

use super::background_candidate_verification::{
    candidate_call_arguments, candidate_execution_report,
};
use crate::{AgentConfig, Ui};

pub(super) struct BackgroundCandidatePlan {
    provider: Arc<dyn Provider>,
    config: AgentConfig,
    binding: WorkspaceBinding,
    source_root: std::path::PathBuf,
    source_state_root: std::path::PathBuf,
    capability_registry: hi_ai::ProviderCapabilityRegistry,
    provider_route: String,
    destination_verifier_timeout_ms: u64,
}

impl crate::Agent {
    pub(super) fn prepare_background_candidate(
        &self,
        slot: u32,
        verify: Option<&str>,
    ) -> BackgroundCandidatePlan {
        let source_root = self.runtime.root().to_path_buf();
        let source_state_root = self.runtime.state_root().to_path_buf();
        let route = self.delegate_route();
        let provider = super::explore_turn::routed_provider(
            route.base_url.as_deref(),
            route.api_key.as_deref(),
            &self.provider,
        );
        let mut config = self.config.clone();
        config.routing.model = route
            .model
            .unwrap_or_else(|| self.config.routing.model.clone());
        config.routing.provider_route = route
            .base_url
            .or_else(|| self.config.routing.provider_route.clone());
        config.routing.tool_mode = ToolMode::Auto;
        config.gates.verification = verify
            .map(str::trim)
            .filter(|command| !command.is_empty())
            .map(|command| {
                crate::VerificationMode::Explicit(vec![crate::VerifyStage::new(
                    "background candidate",
                    command,
                )])
            })
            .unwrap_or_else(|| self.config.gates.verification.clone());
        config.gates.allow_unverified = false;
        config.gates.confirm_edits = false;
        config.gates.dry_run = false;
        config.gates.read_only_preflight = false;
        config.gates.lsp_mode = crate::LspMode::Off;
        config.loop_limits.max_parallel_tools = 4;
        config.memory.project_context = None;
        config.memory.curate_skills = false;
        config.memory.learning = false;
        config.memory.inject_stack_skill = false;
        config.memory.inject_review_skill = false;
        config.memory.suggest_next_prompt = false;
        config.memory.offer_ask_user = false;
        config.memory.offer_mcp = false;
        config.memory.offer_memory = false;
        config.memory.offer_browser = false;
        config.memory.finalize = false;
        config.subagents.explore_subagents = false;
        config.subagents.write_subagents = crate::WriteSubagentPolicy::Off;
        config.subagents.is_subagent = true;
        config.subagents.long_horizon = false;
        config.rsi = crate::AgentRsi::default();
        config.sandbox_policy = Some(hi_tools::sandbox::SandboxPolicy::Strict);
        config.sandbox_config = Some(hi_tools::sandbox::SandboxConfig {
            // Strict mode intentionally exposes a few system roots for
            // toolchains. A checkout beneath one of them must still be wholly
            // invisible to the detached child.
            deny_read: vec![source_root.clone(), source_state_root.clone()],
            deny_host_temp: true,
            ..hi_tools::sandbox::SandboxConfig::default()
        });
        config.verification_timeout = Some(self.config.harness.jobs.verifier_timeout);
        config.suppress_initial_project_hooks = true;
        config.defer_initial_lsp = true;
        config.max_turns = Some(1);
        config.paths.state_root = source_state_root
            .join("subagents")
            .join(format!("background-candidate-{slot}"));
        let destination_timeout_ms =
            u64::try_from(self.config.harness.jobs.verifier_timeout.as_millis())
                .unwrap_or(u64::MAX);
        let provider_route = config
            .routing
            .provider_route
            .clone()
            .unwrap_or_else(|| "unknown".into());
        BackgroundCandidatePlan {
            provider,
            config,
            binding: self.workspace_controller_binding(),
            source_root,
            source_state_root,
            capability_registry: self.provider_capability_registry.clone(),
            provider_route,
            destination_verifier_timeout_ms: destination_timeout_ms,
        }
    }

    #[cfg(test)]
    pub(crate) fn background_candidate_plan_identity(
        &self,
    ) -> (
        Option<hi_tools::sandbox::SandboxPolicy>,
        bool,
        Vec<std::path::PathBuf>,
        String,
        String,
        Option<Duration>,
    ) {
        let plan = self.prepare_background_candidate(0, None);
        (
            plan.config.sandbox_policy,
            plan.config
                .sandbox_config
                .as_ref()
                .is_some_and(|config| config.deny_host_temp),
            plan.config
                .sandbox_config
                .as_ref()
                .map(|config| config.deny_read.clone())
                .unwrap_or_default(),
            plan.provider_route,
            plan.config.routing.model,
            plan.config.verification_timeout,
        )
    }
}

impl BackgroundCandidatePlan {
    pub(super) async fn run(
        mut self,
        task_id: &str,
        prompt: String,
        registry: Arc<hi_tools::BackgroundTaskRegistry>,
        teardown: hi_tools::BackgroundTaskTeardown,
        ui: &mut dyn Ui,
    ) -> hi_tools::BackgroundTaskOutcome {
        let Some(workspace_job_id) = registry.candidate_workspace_job_id(task_id).await else {
            return failed("candidate workspace job identity was unavailable");
        };
        let Some(workspace_verification_ms) =
            registry.candidate_workspace_verification_ms(task_id).await
        else {
            return failed("candidate workspace verification budget was unavailable");
        };
        if workspace_verification_ms != self.destination_verifier_timeout_ms {
            return failed(format!(
                "candidate verifier budget differs from its workspace job (candidate {}ms, job {}ms)",
                self.destination_verifier_timeout_ms, workspace_verification_ms
            ));
        }
        let owner = std::env::temp_dir().join(format!(
            "hi-background-candidate-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let candidate = match hi_tools::candidate_workspace::CandidateWorkspace::create(
            &self.source_root,
            &self.source_state_root,
            &owner,
        ) {
            Ok(candidate) => candidate,
            Err(error) => return failed(format!("candidate materialization failed: {error:#}")),
        };
        let capability_target =
            hi_ai::CapabilityRoute::new(&self.provider_route, &self.config.routing.model);
        let capability_candidates = self
            .provider
            .capability_candidates(&capability_target.route, &capability_target.model);
        let effective_capabilities = self
            .capability_registry
            .resolve_candidates(capability_target, &capability_candidates)
            .await;
        let capability_digest = effective_capabilities.canonical_digest();
        let actual_model_revision = effective_capabilities
            .actual_model_revision()
            .map(str::to_owned);
        self.config.paths.workspace_root = candidate.root().to_path_buf();
        self.config.paths.state_root = owner.join("runtime-state");
        let private_temp = self.config.paths.state_root.join("private-tmp");
        let sandbox_config = self
            .config
            .sandbox_config
            .get_or_insert_with(hi_tools::sandbox::SandboxConfig::default);
        sandbox_config.deny_host_temp = true;
        sandbox_config.private_temp = Some(private_temp);
        let execution_limit = self.config.harness.jobs.candidate_timeout;
        let child = match crate::Agent::new(self.provider.clone(), self.config) {
            Ok(child) => child,
            Err(error) => return failed(format!("candidate child creation failed: {error:#}")),
        };
        let mut child = super::child_process_teardown::ReapingChild::new(child, Some(teardown));
        child
            .child_mut()
            .set_provider_capability_registry(self.capability_registry);
        if !child.child().runtime.process_runner().sandbox_enforced() {
            let detail = child
                .failure_after_reap(
                    "candidate execution requires an enforced OS workspace sandbox; this host has no supported backend",
                )
                .await;
            return failed(detail);
        }
        let child_prompt = format!(
            "Work only in the detached candidate workspace named by your system prompt. Implement \
             the task completely, run the configured verification, and stop. Never access or \
             modify a parent/source checkout outside this workspace.\n\nTask: {}",
            prompt.trim()
        );
        let turn = match tokio::time::timeout(
            execution_limit,
            child.child_mut().run_turn(&child_prompt, ui),
        )
        .await
        {
            Ok(Ok(turn)) => turn,
            Ok(Err(error)) => {
                let detail = child
                    .failure_after_reap(format!("candidate child failed: {error:#}"))
                    .await;
                return failed(detail);
            }
            Err(_) => {
                let detail = child
                    .failure_after_reap(format!(
                        "candidate execution exceeded its {:.1}-second limit",
                        execution_limit.as_secs_f64()
                    ))
                    .await;
                return failed(detail);
            }
        };
        if let Err(error) = child.stop_and_reap().await {
            return failed(format!("candidate processes did not settle: {error:#}"));
        }
        if turn.status != crate::TurnStatus::Completed
            || turn.verification != crate::VerificationStatus::Passed
        {
            return failed(format!(
                "candidate child was not verified (status {:?}, verification {:?}): {:?}",
                turn.status,
                turn.verification,
                child.child().last_verification_executions()
            ));
        }
        let Some(verifier_digest) = turn.verified_workspace_revision.clone() else {
            return failed("candidate verification was not bound to a workspace revision");
        };
        let destination_verification =
            super::background_candidate_verification::from_successful_round(
                child.child().last_verification_executions(),
                workspace_verification_ms,
            );
        if destination_verification.is_empty() {
            return failed(
                "candidate verification passed without an executable destination-verifier contract",
            );
        }
        let summary = child
            .child()
            .last_assistant_text()
            .unwrap_or_else(|| "candidate child completed".into());
        let sealed =
            match candidate.seal_verified(hi_tools::candidate_workspace::CandidateSealContext {
                job_id: JobId::new(workspace_job_id),
                binding: self.binding,
                route: CandidateRoute {
                    provider: turn.effective_route.provider.unwrap_or(self.provider_route),
                    model: turn.effective_route.model,
                    actual_model_revision,
                    capability_digest,
                },
                verification: vec![CandidateVerification {
                    name: "candidate turn verification".into(),
                    passed: true,
                    verifier_digest,
                    detail: Some(format!("review: {:?}", turn.review)),
                    artifacts: Vec::new(),
                }],
                destination_verification,
                destination_verification_budget_ms: workspace_verification_ms,
            }) {
                Ok(sealed) => sealed,
                Err(error) => return failed(format!("candidate sealing failed: {error:#}")),
            };
        let changed_files = sealed
            .candidate
            .changes
            .iter()
            .map(|change| change.path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let sealed = match hi_tools::candidate_workspace::PersistedDetachedCandidate::persist(
            sealed,
            &self.source_state_root,
        ) {
            Ok(sealed) => sealed,
            Err(error) => {
                return failed(format!("candidate artifact persistence failed: {error:#}"));
            }
        };
        if let Err(error) = registry.publish_candidate(task_id, sealed) {
            return failed(format!("candidate publication failed: {error:#}"));
        }
        hi_tools::BackgroundTaskOutcome {
            id: String::new(),
            description: String::new(),
            subagent_type: "general-purpose".into(),
            state: hi_tools::BackgroundTaskState::Completed,
            output: format!(
                "{summary}\nCandidate verified in isolation; waiting for parent apply: {}",
                changed_files.join(", ")
            ),
            applied: false,
            changed_files,
        }
    }
}

impl crate::Agent {
    /// Apply candidates only between fully-settled foreground batches. Polling
    /// merely observes compatibility task state and can never mutate the live
    /// workspace.
    pub(crate) async fn settle_ready_candidates_at_boundary(&mut self) -> anyhow::Result<()> {
        for (task_id, detached) in self.bg_tasks.claim_ready_candidates() {
            let evidence = detached.clone();
            let (mut settlement_guard, abandonment) =
                super::background_candidate_verification::CandidateSettlementGuard::new(
                    Arc::clone(&self.bg_tasks),
                    task_id.clone(),
                    evidence.clone(),
                );
            let binding = self.workspace_controller_binding();
            let verification_ms = self
                .bg_tasks
                .candidate_workspace_verification_ms(&task_id)
                .await;
            if let Err(rejection) = super::background_candidate_verification::candidate_preflight(
                &detached,
                &binding,
                verification_ms,
            ) {
                self.bg_tasks.restore_ready_candidate(&task_id, detached);
                let rejected = self
                    .reject_candidate(&task_id, rejection.transition, rejection.detail)
                    .await;
                if rejected.is_ok() {
                    settlement_guard.disarm();
                }
                rejected?;
                continue;
            }

            let dirty_paths = detached
                .candidate
                .changes
                .iter()
                .map(|change| change.path.clone())
                .collect::<Vec<_>>();
            let intent = MutationIntent {
                effect_scope: EffectScope::LiveWriter,
                replay_class: ReplayClass::PureWorkspace,
                dirty_paths: Some(dirty_paths),
                description: Some(format!(
                    "apply verified background candidate {}",
                    detached.candidate.candidate_id
                )),
            };
            if let Err(error) = self.begin_classified_workspace_operation(intent).await {
                self.bg_tasks.restore_ready_candidate(&task_id, detached);
                settlement_guard.disarm();
                return Err(error.context("admitting background candidate apply"));
            }
            if let Err(error) = self
                .bg_tasks
                .transition_candidate(
                    &task_id,
                    hi_tools::BackgroundCandidateTransition::Merging,
                    None,
                )
                .await
            {
                self.bg_tasks.restore_ready_candidate(&task_id, detached);
                let detail = format!("candidate merge lifecycle failed: {error}");
                let report = candidate_execution_report(
                    &evidence,
                    ExecutionDisposition::Failed,
                    false,
                    &[],
                    Some(detail.clone()),
                );
                let settlement = self
                    .settle_and_reject_candidate(
                        &task_id,
                        &evidence,
                        report,
                        hi_tools::BackgroundCandidateTransition::Failed,
                        detail,
                    )
                    .await;
                if settlement.is_ok() {
                    settlement_guard.disarm();
                }
                settlement?;
                return Err(anyhow::anyhow!(error).context("starting candidate merge lifecycle"));
            }

            let verifier_runner = self.runtime.process_runner().clone();
            let publication = super::background_candidate_verification::supervised_publication(
                Arc::clone(&self.bg_tasks),
                detached.clone(),
                binding.clone(),
                verifier_runner,
                self.turn_cancellation.clone(),
                abandonment,
            )
            .await
            .map_err(anyhow::Error::msg)
            .context("waiting for candidate publication supervisor")?;
            let changes = match publication {
                Ok(changes) => changes,
                Err(error) => {
                    let failure_kind = error.kind();
                    self.bg_tasks.restore_ready_candidate(&task_id, detached);
                    let detail = format!("candidate publication failed: {error:#}");
                    let report = self
                        .candidate_failure_report(&evidence, detail.clone())
                        .await;
                    let transition = match (
                        failure_kind,
                        report.disposition,
                        report.workspace_may_have_changed,
                    ) {
                        (
                            hi_tools::candidate_workspace::CandidatePublicationErrorKind::Stale,
                            ExecutionDisposition::Failed,
                            false,
                        ) => hi_tools::BackgroundCandidateTransition::Stale,
                        (
                            hi_tools::candidate_workspace::CandidatePublicationErrorKind::Failed,
                            ExecutionDisposition::Failed,
                            false,
                        ) => hi_tools::BackgroundCandidateTransition::Failed,
                        _ => hi_tools::BackgroundCandidateTransition::RecoveryRequired,
                    };
                    let rejected = self
                        .settle_and_reject_candidate(
                            &task_id, &evidence, report, transition, detail,
                        )
                        .await;
                    if rejected.is_ok() {
                        settlement_guard.disarm();
                    }
                    rejected?;
                    continue;
                }
            };

            let effects = hi_tools::ToolEffects {
                mutation_attempted: true,
                mutation_applied: true,
                file_changes: changes.clone(),
            };
            if let Err(error) = self.record_tool_effects(&effects) {
                let detail =
                    format!("candidate applied but its effects could not be recorded: {error:#}");
                let reconciliation = self.runtime.reconcile_ledger_async().await;
                let report = match reconciliation {
                    Ok(_) => candidate_execution_report(
                        &evidence,
                        ExecutionDisposition::Failed,
                        true,
                        &changes,
                        Some(detail.clone()),
                    ),
                    Err(reconcile_error) => candidate_execution_report(
                        &evidence,
                        ExecutionDisposition::Indeterminate,
                        true,
                        &changes,
                        Some(format!(
                            "{detail}; final workspace reconciliation failed: {reconcile_error:#}"
                        )),
                    ),
                };
                let rejected = self
                    .settle_and_reject_candidate(
                        &task_id,
                        &evidence,
                        report,
                        hi_tools::BackgroundCandidateTransition::RecoveryRequired,
                        detail,
                    )
                    .await;
                if rejected.is_ok() {
                    settlement_guard.disarm();
                }
                rejected?;
                return Err(error.context("recording applied background candidate"));
            }
            if let Err(error) = self
                .bg_tasks
                .transition_candidate(
                    &task_id,
                    hi_tools::BackgroundCandidateTransition::Settling,
                    None,
                )
                .await
            {
                let detail = format!("candidate applied but settlement lifecycle failed: {error}");
                let report = candidate_execution_report(
                    &evidence,
                    ExecutionDisposition::Failed,
                    true,
                    &changes,
                    Some(detail.clone()),
                );
                let rejected = self
                    .settle_and_reject_candidate(
                        &task_id,
                        &evidence,
                        report,
                        hi_tools::BackgroundCandidateTransition::RecoveryRequired,
                        detail,
                    )
                    .await;
                if rejected.is_ok() {
                    settlement_guard.disarm();
                }
                rejected?;
                return Err(anyhow::anyhow!(error).context("settling candidate lifecycle"));
            }
            let report = candidate_execution_report(
                &evidence,
                ExecutionDisposition::Succeeded,
                true,
                &changes,
                None,
            );
            if let Err(error) = self
                .settle_candidate_execution(&task_id, &evidence, report)
                .await
            {
                let rejected = self
                    .reject_candidate(
                        &task_id,
                        hi_tools::BackgroundCandidateTransition::RecoveryRequired,
                        format!(
                            "candidate bytes were applied but durability is pending: {error:#}"
                        ),
                    )
                    .await;
                if rejected.is_ok() {
                    settlement_guard.disarm();
                }
                rejected?;
                return Err(error);
            }
            if let Err(error) = self
                .bg_tasks
                .transition_candidate(
                    &task_id,
                    hi_tools::BackgroundCandidateTransition::Succeeded,
                    None,
                )
                .await
            {
                let rejected = self
                    .reject_candidate(
                        &task_id,
                        hi_tools::BackgroundCandidateTransition::RecoveryRequired,
                        format!("candidate was durable but its job receipt failed: {error}"),
                    )
                    .await;
                if rejected.is_ok() {
                    settlement_guard.disarm();
                }
                rejected?;
                return Err(anyhow::anyhow!(error).context("publishing candidate job receipt"));
            }
            let changed_paths = changes.into_iter().map(|change| change.path).collect();
            if let Err(error) = detached.remove_after_terminal() {
                tracing::warn!(%error, %task_id, "candidate artifact cleanup failed after success");
            }
            self.bg_tasks
                .resolve_candidate_applied(&task_id, changed_paths);
            settlement_guard.disarm();
        }
        Ok(())
    }

    /// Reconcile an apply failure before publishing its typed result. A
    /// successful scan turns even a rollback error into a known failed
    /// execution with an exact postimage; scan failure remains indeterminate.
    async fn candidate_failure_report(
        &self,
        evidence: &hi_tools::candidate_workspace::PersistedDetachedCandidate,
        detail: String,
    ) -> ExecutionReport {
        match self.runtime.reconcile_ledger_async().await {
            Ok(changes) => candidate_execution_report(
                evidence,
                ExecutionDisposition::Failed,
                !changes.is_empty(),
                &changes,
                Some(detail),
            ),
            Err(error) => candidate_execution_report(
                evidence,
                ExecutionDisposition::Indeterminate,
                true,
                &[],
                Some(format!(
                    "{detail}; final workspace reconciliation failed: {error:#}"
                )),
            ),
        }
    }

    /// Stage the exact candidate/job result before handing the active permit
    /// to the controller. Local controllers skip the remote stage; PipeFS must
    /// acknowledge both this record and the workspace receipt.
    async fn settle_candidate_execution(
        &mut self,
        task_id: &str,
        evidence: &hi_tools::candidate_workspace::PersistedDetachedCandidate,
        mut execution: ExecutionReport,
    ) -> anyhow::Result<()> {
        if execution.workspace_may_have_changed
            && execution.disposition != ExecutionDisposition::Indeterminate
            && execution.content_digest.is_none()
        {
            execution.content_digest = Some(self.runtime.ledger().workspace_revision());
        }
        let operation_id = self
            .workspace_coordination
            .active_parent_operation()
            .context("candidate apply has no admitted workspace operation")?;
        let call_id = format!("candidate-apply:{operation_id}");
        let arguments = candidate_call_arguments(task_id, evidence);
        let result = match serde_json::to_string(&execution) {
            Ok(result) => result,
            Err(error) => {
                execution.disposition = ExecutionDisposition::Indeterminate;
                execution.detail = Some(match execution.detail.take() {
                    Some(detail) => format!(
                        "{detail}; candidate execution evidence could not be serialized: {error}"
                    ),
                    None => {
                        format!("candidate execution evidence could not be serialized: {error}")
                    }
                });
                let settlement = self
                    .checkpoint_durable_workspace_with_execution(execution)
                    .await;
                return match settlement {
                    Ok(()) => Err(error).context(
                        "candidate evidence serialization failed even though the controller returned a receipt",
                    ),
                    Err(settlement) => Err(settlement).context(format!(
                        "candidate evidence serialization failed before settlement: {error}"
                    )),
                };
            }
        };
        let calls = [(
            call_id.clone(),
            "apply_background_candidate".into(),
            arguments.clone(),
        )];
        let assistant_content = [hi_ai::Content::ToolCall {
            id: call_id.clone(),
            name: "apply_background_candidate".into(),
            arguments,
        }];
        let results = [(call_id, result)];
        let stage_error = self
            .stage_active_workspace_execution(&calls, &assistant_content, &results, &execution)
            .err();
        if let Some(error) = &stage_error {
            execution.disposition = ExecutionDisposition::Indeterminate;
            execution.detail = Some(match execution.detail.take() {
                Some(detail) => {
                    format!("{detail}; candidate transcript staging is ambiguous: {error:#}")
                }
                None => format!("candidate transcript staging is ambiguous: {error:#}"),
            });
        }
        let settlement = self
            .checkpoint_durable_workspace_with_execution(execution)
            .await;
        match (stage_error, settlement) {
            (None, Ok(())) => Ok(()),
            (Some(stage), Ok(())) => Err(stage).context(
                "candidate transcript staging failed even though the controller returned a receipt",
            ),
            (None, Err(settlement)) => Err(settlement),
            (Some(stage), Err(settlement)) => Err(settlement).context(format!(
                "candidate transcript staging failed before settlement: {stage:#}"
            )),
        }
    }

    async fn settle_and_reject_candidate(
        &mut self,
        task_id: &str,
        evidence: &hi_tools::candidate_workspace::PersistedDetachedCandidate,
        execution: ExecutionReport,
        transition: hi_tools::BackgroundCandidateTransition,
        detail: String,
    ) -> anyhow::Result<()> {
        let settlement = self
            .settle_candidate_execution(task_id, evidence, execution)
            .await;
        let (transition, detail) = match &settlement {
            Ok(()) => (transition, detail),
            Err(error) => (
                hi_tools::BackgroundCandidateTransition::RecoveryRequired,
                format!("{detail}; workspace/transcript settlement failed: {error:#}"),
            ),
        };
        let rejection = self.reject_candidate(task_id, transition, detail).await;
        match (settlement, rejection) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(settlement), Err(rejection)) => Err(settlement).context(format!(
                "candidate job recovery publication also failed: {rejection:#}"
            )),
        }
    }

    async fn reject_candidate(
        &self,
        task_id: &str,
        transition: hi_tools::BackgroundCandidateTransition,
        detail: String,
    ) -> anyhow::Result<()> {
        self.bg_tasks
            .transition_candidate(task_id, transition, Some(detail.clone()))
            .await
            .map_err(anyhow::Error::msg)
            .with_context(|| detail.clone())?;
        if matches!(
            transition,
            hi_tools::BackgroundCandidateTransition::RecoveryRequired
                | hi_tools::BackgroundCandidateTransition::Stale
        ) {
            self.bg_tasks.resolve_candidate_retained(task_id, detail);
        } else {
            self.bg_tasks.resolve_candidate_rejected(task_id, detail);
        }
        Ok(())
    }
}

fn failed(message: impl Into<String>) -> hi_tools::BackgroundTaskOutcome {
    hi_tools::BackgroundTaskOutcome {
        id: String::new(),
        description: String::new(),
        subagent_type: "general-purpose".into(),
        state: hi_tools::BackgroundTaskState::Failed,
        output: message.into(),
        applied: false,
        changed_files: Vec::new(),
    }
}
