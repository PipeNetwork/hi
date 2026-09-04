//! NativeDirector v2 shadow integration for the existing Rust turn loop.

use anyhow::Result;
use hi_engine_api::{
    EngineAction, EngineInput, EngineStateSnapshot, PresentationDirective, ToolDescriptor,
};
use hi_engine_host::{
    DirectorActionKind, DirectorPolicySignals, DirectorRequirement, DirectorTraceMode,
    ForcedToolSignal, NativeDirectorV2,
};

use super::state::TurnState;
use crate::Ui;
use crate::transcript::NudgeKind;

pub(super) struct TurnNativeDirector {
    director: Option<NativeDirectorV2>,
    mode: DirectorTraceMode,
    candidate_sequence: u64,
}

pub(super) struct DirectorTurnRequirements {
    pub plan: bool,
    pub goal: bool,
    pub reminder: bool,
    pub forced_tool: bool,
    pub verify_before_yield: bool,
}

impl TurnNativeDirector {
    fn start(
        input: Result<EngineInput>,
        signals: DirectorPolicySignals,
        mode: DirectorTraceMode,
    ) -> Self {
        let director = input.and_then(|input| {
            let mut director = NativeDirectorV2::new(signals);
            let evaluation = director.step_with_trace(
                &input,
                Some(&[DirectorActionKind::RequestModel]),
                mode,
            )?;
            emit_trace(&evaluation.trace)?;
            Ok(director)
        });
        let director = match director {
            Ok(director) => Some(director),
            Err(error) => {
                tracing::warn!(%error, "disabled NativeDirector v2 shadow for this turn");
                None
            }
        };
        Self {
            director,
            mode,
            candidate_sequence: 0,
        }
    }

    fn candidate_yield(
        &mut self,
        plan_pending: bool,
        goal_pending: bool,
        reminder_pending: bool,
        forced_tool_pending: bool,
        verify_pending: bool,
        promotion_allowed: bool,
    ) -> Option<String> {
        let director = self.director.as_mut()?;
        let mut signals = director.signals().clone();
        refresh_requirement(&mut signals.plan, plan_pending);
        refresh_requirement(&mut signals.goal, goal_pending);
        refresh_requirement(&mut signals.reminder, reminder_pending);
        refresh_requirement(&mut signals.forced_tool.requirement, forced_tool_pending);
        refresh_requirement(&mut signals.verify_before_yield, verify_pending);
        director.set_signals(signals);
        let request_id = format!("native-director-candidate-{}", self.candidate_sequence);
        self.candidate_sequence = self.candidate_sequence.saturating_add(1);
        let input = EngineInput::ProviderDelta {
            request_id,
            text: "candidate_yield".into(),
            reasoning: String::new(),
            tool_call_deltas: Vec::new(),
            done: true,
        };
        let legacy = if verify_pending {
            DirectorActionKind::Wait
        } else {
            DirectorActionKind::Complete
        };
        let mode = if !promotion_allowed && self.mode == DirectorTraceMode::Promoted {
            DirectorTraceMode::Shadow
        } else {
            self.mode
        };
        let evaluation = match director.step_with_trace(&input, Some(&[legacy]), mode) {
            Ok(evaluation) => evaluation,
            Err(error) => {
                tracing::warn!(%error, "disabled NativeDirector v2 shadow after step failure");
                self.director = None;
                return None;
            }
        };
        if let Err(error) = emit_trace(&evaluation.trace) {
            tracing::warn!(%error, "disabled NativeDirector v2 shadow after trace failure");
            self.director = None;
            return None;
        }
        if !promotion_allowed {
            return None;
        }
        evaluation
            .promoted_actions
            .as_deref()
            .and_then(promoted_model_instruction)
    }

    fn goal_required(&self) -> bool {
        self.director
            .as_ref()
            .is_some_and(|director| director.signals().goal.required)
    }
}

fn refresh_requirement(requirement: &mut DirectorRequirement, pending: bool) {
    requirement.required |= pending;
    requirement.satisfied = requirement.required && !pending;
}

fn promoted_model_instruction(actions: &[EngineAction]) -> Option<String> {
    let EngineAction::RequestModel { messages_json, .. } = actions.first()? else {
        return None;
    };
    serde_json::from_str::<serde_json::Value>(messages_json)
        .ok()?
        .get("instruction")?
        .as_str()
        .map(str::to_owned)
}

fn emit_trace(trace: &hi_engine_host::DirectorActionTrace) -> Result<()> {
    let trace_json = trace.to_json()?;
    tracing::debug!(
        target: "hi::native_director",
        director_version = trace.director_version,
        trace_schema_version = trace.trace_schema_version,
        sequence = trace.sequence,
        parity = ?trace.parity,
        trace = %trace_json,
        "NativeDirector v2 action trace"
    );
    Ok(())
}

const fn director_trace_mode(managed_rsi: bool, runtime_promotion: bool) -> DirectorTraceMode {
    if managed_rsi {
        DirectorTraceMode::HigherTrustRsi
    } else if runtime_promotion {
        DirectorTraceMode::Promoted
    } else {
        DirectorTraceMode::Shadow
    }
}

impl crate::Agent {
    pub(in crate::agent::turn) fn initialize_native_director_v2(
        &self,
        prompt: &str,
        workspace_context_generation: u64,
        ledger_revision: u64,
        requirements: DirectorTurnRequirements,
    ) -> TurnNativeDirector {
        let signals = DirectorPolicySignals {
            plan: DirectorRequirement::pending(requirements.plan),
            goal: DirectorRequirement::pending(requirements.goal),
            reminder: DirectorRequirement::pending(requirements.reminder),
            forced_tool: ForcedToolSignal {
                requirement: DirectorRequirement::pending(requirements.forced_tool),
                tool_name: None,
            },
            verify_before_yield: DirectorRequirement::pending(requirements.verify_before_yield),
        };
        TurnNativeDirector::start(
            self.engine_tool_descriptors()
                .map(|tools| EngineInput::TurnStarted {
                    snapshot: self.engine_snapshot(workspace_context_generation, ledger_revision),
                    prompt: prompt.into(),
                    tools,
                }),
            signals,
            director_trace_mode(
                self.config.rsi.managed,
                self.config.harness.features.native_director_v2,
            ),
        )
    }

    pub(in crate::agent::turn) fn native_director_candidate_yield(
        &mut self,
        turn: &mut TurnState,
        hit_cap: bool,
    ) -> bool {
        let plan_pending = self.plan_mode && self.goals.plan().is_empty();
        let reminder_pending = !self.plan_mode && self.goals.plan_incomplete();
        let goal_pending = turn.native_director.goal_required() && !turn.plan_updated_goal;
        let promotion_allowed =
            !hit_cap && !turn.flags.provider_exhausted && !turn.flags.ended_at_deadline;
        let instruction = turn.native_director.candidate_yield(
            plan_pending,
            goal_pending,
            reminder_pending,
            turn.flags.force_tools_next,
            (turn.expected_mutation || turn.requested_validation) && turn.verifier.is_on(),
            promotion_allowed,
        );
        let Some(instruction) = instruction else {
            return false;
        };
        self.messages
            .push_nudge_or_fold(NudgeKind::Continue, instruction);
        true
    }

    fn engine_snapshot(
        &self,
        workspace_context_generation: u64,
        ledger_revision: u64,
    ) -> EngineStateSnapshot {
        EngineStateSnapshot {
            api_major: hi_engine_api::ENGINE_API_MAJOR,
            api_minor: hi_engine_api::ENGINE_API_MINOR,
            state_schema_version: hi_engine_api::ENGINE_STATE_SCHEMA_VERSION,
            turn_id: format!("turn-{}", self.turn_count.saturating_add(1)),
            workspace_context_generation,
            ledger_revision,
            state: Vec::new(),
        }
    }

    fn engine_tool_descriptors(&self) -> Result<Vec<ToolDescriptor>> {
        self.tools
            .iter()
            .take(hi_engine_api::MAX_ENGINE_TOOLS)
            .map(|tool| {
                Ok(ToolDescriptor {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters_json: serde_json::to_string(&tool.parameters)?,
                })
            })
            .collect()
    }

    /// Run the guest's first decision step after the turn context is known.
    /// Effect actions remain rejected until a replay-equivalent router exists.
    pub(in crate::agent::turn) fn initialize_wasm_turn(
        &mut self,
        lease: &hi_engine_host::EngineLease,
        prompt: &str,
        workspace_context_generation: u64,
        ledger_revision: u64,
        ui: &mut dyn Ui,
    ) -> Result<()> {
        if self.config.engine.mode != hi_engine_api::EngineMode::Wasm {
            return Ok(());
        }
        let Some(mut engine) = lease.wasm_engine()? else {
            return Ok(());
        };
        let input = EngineInput::TurnStarted {
            snapshot: self.engine_snapshot(workspace_context_generation, ledger_revision),
            prompt: prompt.to_string(),
            tools: self.engine_tool_descriptors()?,
        };
        let module = engine.info().guest_version.clone();
        let actions = match engine.step(&input) {
            Ok(actions) => actions,
            Err(error) => {
                self.engine_runtime.rollback_active();
                tracing::warn!(guest = %module, %error, "WASM engine trapped during turn initialization");
                ui.status(&format!(
                    "logic {module} failed; retained previous module and using native engine"
                ));
                return Ok(());
            }
        };
        let ledger = hi_engine_host::ActionLedger::default();
        if let Err(error) = ledger.claim(&actions) {
            self.engine_runtime.rollback_active();
            tracing::warn!(guest = %module, %error, "WASM engine returned duplicate or invalid actions");
            ui.status(&format!(
                "logic {module} returned invalid actions; retained previous module and using native engine"
            ));
            return Ok(());
        }
        for action in actions {
            match action {
                EngineAction::RequestModel { .. }
                | EngineAction::Wait { .. }
                | EngineAction::UpdateState { .. }
                | EngineAction::Complete { .. } => {}
                EngineAction::Present { directive, .. } => match directive {
                    PresentationDirective::Status { text, .. }
                    | PresentationDirective::Activity { text, .. }
                    | PresentationDirective::Completion { text, .. } => ui.status(&text),
                    PresentationDirective::Warning { text, .. } => ui.top_status(&text),
                    PresentationDirective::ChangedFiles { .. } => {
                        tracing::debug!(guest = %module, "ignored pre-turn changed-files directive")
                    }
                },
                EngineAction::ExecuteTool { .. } | EngineAction::ExecuteParallel { .. } => {
                    self.engine_runtime.rollback_active();
                    tracing::warn!(guest = %module, "WASM engine requested an effect before the native action router was enabled");
                    ui.status(&format!(
                        "logic {module} requested an unavailable effect; retained previous module and using native engine"
                    ));
                    break;
                }
                EngineAction::Fail { code, message, .. } => {
                    self.engine_runtime.rollback_active();
                    tracing::warn!(guest = %module, code = %code, %message, "WASM engine reported initialization failure");
                    ui.status(&format!(
                        "logic {module} failed ({code}); retained previous module and using native engine"
                    ));
                    break;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_false_disables_promotion_and_managed_rsi_always_wins() {
        assert_eq!(
            director_trace_mode(true, true),
            DirectorTraceMode::HigherTrustRsi
        );
        assert_eq!(
            director_trace_mode(false, true),
            DirectorTraceMode::Promoted
        );
        assert_eq!(director_trace_mode(false, false), DirectorTraceMode::Shadow);
    }

    #[test]
    fn promotion_only_extracts_model_continuations() {
        let request = EngineAction::RequestModel {
            idempotency_key: "key".into(),
            request_id: "request".into(),
            messages_json: r#"{"instruction":"continue safely"}"#.into(),
        };
        assert_eq!(
            promoted_model_instruction(&[request]).as_deref(),
            Some("continue safely")
        );
        assert!(
            promoted_model_instruction(&[EngineAction::Wait {
                idempotency_key: "wait".into(),
            }])
            .is_none()
        );
    }

    #[test]
    fn shadow_initialization_failure_is_fail_open() {
        let mut director = TurnNativeDirector::start(
            Err(anyhow::anyhow!("oversized shadow input")),
            DirectorPolicySignals::default(),
            DirectorTraceMode::Shadow,
        );
        assert!(!director.goal_required());
        assert!(
            director
                .candidate_yield(true, true, true, true, true, true)
                .is_none()
        );
    }
}
