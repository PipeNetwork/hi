//! Deterministic native director used to shadow the interactive turn loop.
//!
//! The director proposes protocol actions only. The host remains the effect
//! owner, and promotion deliberately excludes effectful actions so enabling a
//! candidate cannot bypass the tool broker or the higher-trust RSI executor.

use anyhow::{Context, Result, anyhow};
use hi_engine_api::{EngineAction, EngineInput, EngineMode, encode_input};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::DecisionEngine;

pub const NATIVE_DIRECTOR_VERSION: u16 = 2;
pub const DIRECTOR_TRACE_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectorTraceMode {
    #[default]
    Shadow,
    Promoted,
    HigherTrustRsi,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectorRequirement {
    pub required: bool,
    pub satisfied: bool,
}

impl DirectorRequirement {
    pub const fn pending(required: bool) -> Self {
        Self {
            required,
            satisfied: false,
        }
    }

    const fn is_pending(self) -> bool {
        self.required && !self.satisfied
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForcedToolSignal {
    pub requirement: DirectorRequirement,
    /// `None` means any successful tool can satisfy the obligation.
    pub tool_name: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectorPolicySignals {
    pub plan: DirectorRequirement,
    pub goal: DirectorRequirement,
    pub reminder: DirectorRequirement,
    pub forced_tool: ForcedToolSignal,
    pub verify_before_yield: DirectorRequirement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectorDecision {
    RequestModel,
    AwaitProvider,
    AwaitTool,
    ResumeAfterTool,
    ForceTool,
    Plan,
    Goal,
    Reminder,
    VerifyBeforeYield,
    Yield,
    ApprovalDenied,
    Cancelled,
    TimedOut,
    HostError,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectorActionKind {
    RequestModel,
    ExecuteTool,
    ExecuteParallel,
    Present,
    UpdateState,
    Wait,
    Complete,
    Fail,
}

impl From<&EngineAction> for DirectorActionKind {
    fn from(action: &EngineAction) -> Self {
        match action {
            EngineAction::RequestModel { .. } => Self::RequestModel,
            EngineAction::ExecuteTool { .. } => Self::ExecuteTool,
            EngineAction::ExecuteParallel { .. } => Self::ExecuteParallel,
            EngineAction::Present { .. } => Self::Present,
            EngineAction::UpdateState { .. } => Self::UpdateState,
            EngineAction::Wait { .. } => Self::Wait,
            EngineAction::Complete { .. } => Self::Complete,
            EngineAction::Fail { .. } => Self::Fail,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectorParity {
    Match,
    Diverged,
    NotCompared,
    HigherTrustBypass,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectorActionTraceEntry {
    pub kind: DirectorActionKind,
    pub action_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectorActionTrace {
    pub trace_schema_version: u16,
    pub director_version: u16,
    pub sequence: u64,
    pub mode: DirectorTraceMode,
    pub input_kind: String,
    pub input_sha256: String,
    pub signals: DirectorPolicySignals,
    pub decision: DirectorDecision,
    pub proposed_actions: Vec<DirectorActionTraceEntry>,
    /// Semantic action classes observed at the legacy loop boundary. The
    /// legacy loop does not yet materialize full protocol actions.
    pub legacy_actions: Option<Vec<DirectorActionKind>>,
    pub parity: DirectorParity,
    pub promotion_applied: bool,
}

impl DirectorActionTrace {
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serializing native director trace")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectorEvaluation {
    pub proposed_actions: Vec<EngineAction>,
    /// Present only when promotion was requested and every action is a model
    /// continuation. Completion, waiting, presentation, and effects stay with
    /// the existing native loop during this migration stage.
    pub promoted_actions: Option<Vec<EngineAction>>,
    pub trace: DirectorActionTrace,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeDirectorState {
    director_version: u16,
    trace_schema_version: u16,
    turn_id: String,
    next_sequence: u64,
    signals: DirectorPolicySignals,
}

#[derive(Clone, Debug)]
pub struct NativeDirectorV2 {
    state: NativeDirectorState,
}

impl NativeDirectorV2 {
    pub fn new(signals: DirectorPolicySignals) -> Self {
        Self {
            state: NativeDirectorState {
                director_version: NATIVE_DIRECTOR_VERSION,
                trace_schema_version: DIRECTOR_TRACE_SCHEMA_VERSION,
                turn_id: "unbound-turn".into(),
                next_sequence: 0,
                signals,
            },
        }
    }

    pub fn signals(&self) -> &DirectorPolicySignals {
        &self.state.signals
    }

    pub fn set_signals(&mut self, signals: DirectorPolicySignals) {
        self.state.signals = signals;
    }

    pub fn step_with_trace(
        &mut self,
        input: &EngineInput,
        legacy_actions: Option<&[DirectorActionKind]>,
        mode: DirectorTraceMode,
    ) -> Result<DirectorEvaluation> {
        input
            .validate()
            .map_err(|error| anyhow!("invalid native director input: {error}"))?;
        if let EngineInput::TurnStarted { snapshot, .. } = input {
            self.state.turn_id.clone_from(&snapshot.turn_id);
            self.state.next_sequence = 0;
        }
        let sequence = self.state.next_sequence;
        self.state.next_sequence = self.state.next_sequence.saturating_add(1);
        let (decision, proposed_actions) = self.propose(input, sequence)?;
        for action in &proposed_actions {
            action
                .validate()
                .map_err(|error| anyhow!("native director returned invalid action: {error}"))?;
        }
        let proposed_kinds = proposed_actions
            .iter()
            .map(DirectorActionKind::from)
            .collect::<Vec<_>>();
        let parity = match (mode, legacy_actions) {
            (DirectorTraceMode::HigherTrustRsi, _) => DirectorParity::HigherTrustBypass,
            (_, None) => DirectorParity::NotCompared,
            (_, Some(legacy)) if legacy == proposed_kinds.as_slice() => DirectorParity::Match,
            (_, Some(_)) => DirectorParity::Diverged,
        };
        let promotion_applied =
            mode == DirectorTraceMode::Promoted && proposed_actions.iter().all(promotion_safe);
        let promoted_actions = promotion_applied.then(|| proposed_actions.clone());
        let proposed_actions_trace = proposed_actions
            .iter()
            .map(trace_action)
            .collect::<Result<Vec<_>>>()?;
        let trace = DirectorActionTrace {
            trace_schema_version: DIRECTOR_TRACE_SCHEMA_VERSION,
            director_version: NATIVE_DIRECTOR_VERSION,
            sequence,
            mode,
            input_kind: input_kind(input).into(),
            input_sha256: sha256_hex(encode_input(input)?.as_bytes()),
            signals: self.state.signals.clone(),
            decision,
            proposed_actions: proposed_actions_trace,
            legacy_actions: legacy_actions.map(<[DirectorActionKind]>::to_vec),
            parity,
            promotion_applied,
        };
        Ok(DirectorEvaluation {
            proposed_actions,
            promoted_actions,
            trace,
        })
    }

    fn propose(
        &mut self,
        input: &EngineInput,
        sequence: u64,
    ) -> Result<(DirectorDecision, Vec<EngineAction>)> {
        let result = match input {
            EngineInput::TurnStarted { .. } => (
                DirectorDecision::RequestModel,
                vec![self.request_model(sequence, DirectorDecision::RequestModel)?],
            ),
            EngineInput::ProviderDelta { done: false, .. } => (
                DirectorDecision::AwaitProvider,
                vec![self.wait(sequence, "provider")],
            ),
            EngineInput::ProviderDelta {
                tool_call_deltas, ..
            } if !tool_call_deltas.is_empty() => (
                DirectorDecision::AwaitTool,
                vec![self.wait(sequence, "tool")],
            ),
            EngineInput::ProviderDelta { .. } => self.candidate_yield(sequence)?,
            EngineInput::ToolResult { name, status, .. } => {
                self.observe_tool_result(name, status);
                (
                    DirectorDecision::ResumeAfterTool,
                    vec![self.request_model(sequence, DirectorDecision::ResumeAfterTool)?],
                )
            }
            EngineInput::ApprovalResult { approved: true, .. } => (
                DirectorDecision::AwaitTool,
                vec![self.wait(sequence, "approval")],
            ),
            EngineInput::ApprovalResult { .. } => (
                DirectorDecision::ApprovalDenied,
                vec![self.fail(sequence, "approval_denied", "tool approval was denied")],
            ),
            EngineInput::Cancelled { .. } => (
                DirectorDecision::Cancelled,
                vec![self.fail(sequence, "cancelled", "turn was cancelled")],
            ),
            EngineInput::TimedOut => (
                DirectorDecision::TimedOut,
                vec![self.fail(sequence, "timed_out", "turn timed out")],
            ),
            EngineInput::HostError { .. } => (
                DirectorDecision::HostError,
                vec![self.fail(sequence, "host_error", "host reported an error")],
            ),
        };
        Ok(result)
    }

    fn candidate_yield(&self, sequence: u64) -> Result<(DirectorDecision, Vec<EngineAction>)> {
        let decision = if self.state.signals.forced_tool.requirement.is_pending() {
            DirectorDecision::ForceTool
        } else if self.state.signals.plan.is_pending() {
            DirectorDecision::Plan
        } else if self.state.signals.goal.is_pending() {
            DirectorDecision::Goal
        } else if self.state.signals.reminder.is_pending() {
            DirectorDecision::Reminder
        } else if self.state.signals.verify_before_yield.is_pending() {
            DirectorDecision::VerifyBeforeYield
        } else {
            DirectorDecision::Yield
        };
        let actions = match decision {
            DirectorDecision::VerifyBeforeYield => vec![self.wait(sequence, "verification")],
            DirectorDecision::Yield => vec![EngineAction::Complete {
                idempotency_key: self.key(sequence, "complete"),
                result_json: serde_json::json!({
                    "director_version": NATIVE_DIRECTOR_VERSION,
                    "decision": "yield"
                })
                .to_string(),
            }],
            _ => vec![self.request_model(sequence, decision)?],
        };
        Ok((decision, actions))
    }

    fn observe_tool_result(&mut self, name: &str, status: &str) {
        if !successful_status(status) {
            return;
        }
        let forced = &mut self.state.signals.forced_tool;
        if forced.requirement.required
            && forced
                .tool_name
                .as_deref()
                .is_none_or(|required| required == name)
        {
            forced.requirement.satisfied = true;
        }
        if matches!(name, "verify" | "workspace_verify") {
            self.state.signals.verify_before_yield.satisfied = true;
        }
    }

    fn request_model(&self, sequence: u64, decision: DirectorDecision) -> Result<EngineAction> {
        let instruction = match decision {
            DirectorDecision::ForceTool => "call the required tool before yielding",
            DirectorDecision::Plan => "finish the active plan obligation before yielding",
            DirectorDecision::Goal => "finish the active goal obligation before yielding",
            DirectorDecision::Reminder => "address the outstanding reminder before yielding",
            DirectorDecision::ResumeAfterTool => "continue from the completed tool result",
            _ => "begin the model turn",
        };
        Ok(EngineAction::RequestModel {
            idempotency_key: self.key(sequence, "model"),
            request_id: self.key(sequence, "request"),
            messages_json: serde_json::to_string(&serde_json::json!({
                "director_version": NATIVE_DIRECTOR_VERSION,
                "policy_signal": decision,
                "instruction": instruction,
            }))?,
        })
    }

    fn wait(&self, sequence: u64, suffix: &str) -> EngineAction {
        EngineAction::Wait {
            idempotency_key: self.key(sequence, suffix),
        }
    }

    fn fail(&self, sequence: u64, code: &str, message: &str) -> EngineAction {
        EngineAction::Fail {
            idempotency_key: self.key(sequence, "fail"),
            code: code.into(),
            message: message.into(),
        }
    }

    fn key(&self, sequence: u64, suffix: &str) -> String {
        format!(
            "native-director-v{}:{}:{sequence}:{suffix}",
            NATIVE_DIRECTOR_VERSION, self.state.turn_id
        )
    }
}

impl DecisionEngine for NativeDirectorV2 {
    fn mode(&self) -> EngineMode {
        EngineMode::Native
    }

    fn step(&mut self, input: &EngineInput) -> Result<Vec<EngineAction>> {
        Ok(self
            .step_with_trace(input, None, DirectorTraceMode::Shadow)?
            .proposed_actions)
    }

    fn serialize_state(&mut self) -> Result<Vec<u8>> {
        serde_json::to_vec(&self.state).context("serializing native director state")
    }
}

fn input_kind(input: &EngineInput) -> &'static str {
    match input {
        EngineInput::TurnStarted { .. } => "turn_started",
        EngineInput::ProviderDelta { .. } => "provider_delta",
        EngineInput::ToolResult { .. } => "tool_result",
        EngineInput::ApprovalResult { .. } => "approval_result",
        EngineInput::Cancelled { .. } => "cancelled",
        EngineInput::TimedOut => "timed_out",
        EngineInput::HostError { .. } => "host_error",
    }
}

fn trace_action(action: &EngineAction) -> Result<DirectorActionTraceEntry> {
    Ok(DirectorActionTraceEntry {
        kind: DirectorActionKind::from(action),
        action_sha256: sha256_hex(&serde_json::to_vec(action)?),
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn successful_status(status: &str) -> bool {
    matches!(
        status.trim().to_ascii_lowercase().as_str(),
        "ok" | "success" | "succeeded" | "complete" | "completed" | "passed"
    )
}

fn promotion_safe(action: &EngineAction) -> bool {
    matches!(action, EngineAction::RequestModel { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_engine_api::{EngineStateSnapshot, PresentationDirective, ToolDescriptor, ToolRequest};

    fn start() -> EngineInput {
        EngineInput::TurnStarted {
            snapshot: EngineStateSnapshot::new("turn-7"),
            prompt: "implement the task".into(),
            tools: vec![ToolDescriptor {
                name: "read".into(),
                description: "read a file".into(),
                parameters_json: "{}".into(),
            }],
        }
    }

    fn candidate() -> EngineInput {
        EngineInput::ProviderDelta {
            request_id: "request-1".into(),
            text: "done".into(),
            reasoning: String::new(),
            tool_call_deltas: Vec::new(),
            done: true,
        }
    }

    fn pending(decision: DirectorDecision) -> DirectorPolicySignals {
        let mut signals = DirectorPolicySignals::default();
        match decision {
            DirectorDecision::ForceTool => {
                signals.forced_tool.requirement = DirectorRequirement::pending(true)
            }
            DirectorDecision::Plan => signals.plan = DirectorRequirement::pending(true),
            DirectorDecision::Goal => signals.goal = DirectorRequirement::pending(true),
            DirectorDecision::Reminder => signals.reminder = DirectorRequirement::pending(true),
            DirectorDecision::VerifyBeforeYield => {
                signals.verify_before_yield = DirectorRequirement::pending(true)
            }
            _ => unreachable!(),
        }
        signals
    }

    #[test]
    fn traces_are_versioned_and_deterministic() {
        let run = || {
            let mut director = NativeDirectorV2::new(DirectorPolicySignals::default());
            director
                .step_with_trace(
                    &start(),
                    Some(&[DirectorActionKind::RequestModel]),
                    DirectorTraceMode::Shadow,
                )
                .unwrap()
                .trace
                .to_json()
                .unwrap()
        };
        let first = run();
        assert_eq!(first, run());
        assert!(first.contains("\"trace_schema_version\":1"));
        assert!(first.contains("\"director_version\":2"));
    }

    #[test]
    fn traces_parity_and_divergence_against_legacy_actions() {
        let mut parity = NativeDirectorV2::new(DirectorPolicySignals::default());
        let matching = parity
            .step_with_trace(
                &start(),
                Some(&[DirectorActionKind::RequestModel]),
                DirectorTraceMode::Shadow,
            )
            .unwrap();
        assert_eq!(matching.trace.parity, DirectorParity::Match);

        let divergent = parity
            .step_with_trace(
                &candidate(),
                Some(&[DirectorActionKind::Wait]),
                DirectorTraceMode::Shadow,
            )
            .unwrap();
        assert_eq!(divergent.trace.decision, DirectorDecision::Yield);
        assert_eq!(divergent.trace.parity, DirectorParity::Diverged);
    }

    #[test]
    fn policy_stack_covers_each_yield_signal() {
        for decision in [
            DirectorDecision::ForceTool,
            DirectorDecision::Plan,
            DirectorDecision::Goal,
            DirectorDecision::Reminder,
            DirectorDecision::VerifyBeforeYield,
        ] {
            let mut director = NativeDirectorV2::new(pending(decision));
            director.step(&start()).unwrap();
            let evaluation = director
                .step_with_trace(&candidate(), None, DirectorTraceMode::Shadow)
                .unwrap();
            assert_eq!(evaluation.trace.decision, decision);
        }
    }

    #[test]
    fn policy_stack_uses_force_plan_goal_reminder_verify_precedence() {
        let mut signals = DirectorPolicySignals {
            plan: DirectorRequirement::pending(true),
            goal: DirectorRequirement::pending(true),
            reminder: DirectorRequirement::pending(true),
            verify_before_yield: DirectorRequirement::pending(true),
            ..DirectorPolicySignals::default()
        };
        signals.forced_tool.requirement = DirectorRequirement::pending(true);
        let mut director = NativeDirectorV2::new(signals);
        director.step(&start()).unwrap();
        assert_eq!(
            director
                .step_with_trace(&candidate(), None, DirectorTraceMode::Shadow)
                .unwrap()
                .trace
                .decision,
            DirectorDecision::ForceTool
        );
    }

    #[test]
    fn shadow_and_rsi_modes_never_promote() {
        for mode in [DirectorTraceMode::Shadow, DirectorTraceMode::HigherTrustRsi] {
            let mut director = NativeDirectorV2::new(pending(DirectorDecision::Plan));
            director.step(&start()).unwrap();
            let evaluation = director.step_with_trace(&candidate(), None, mode).unwrap();
            assert!(evaluation.promoted_actions.is_none());
            if mode == DirectorTraceMode::HigherTrustRsi {
                assert_eq!(evaluation.trace.parity, DirectorParity::HigherTrustBypass);
            }
        }
    }

    #[test]
    fn verification_waits_for_the_native_verifier_instead_of_requesting_an_effect() {
        let mut director = NativeDirectorV2::new(pending(DirectorDecision::VerifyBeforeYield));
        director.step(&start()).unwrap();
        let evaluation = director
            .step_with_trace(&candidate(), None, DirectorTraceMode::Promoted)
            .unwrap();
        assert!(matches!(
            evaluation.proposed_actions.as_slice(),
            [EngineAction::Wait { .. }]
        ));
        assert!(evaluation.promoted_actions.is_none());
    }

    #[test]
    fn promotion_is_limited_to_model_continuations() {
        let mut director = NativeDirectorV2::new(pending(DirectorDecision::Plan));
        director.step(&start()).unwrap();
        let evaluation = director
            .step_with_trace(&candidate(), None, DirectorTraceMode::Promoted)
            .unwrap();
        assert!(evaluation.promoted_actions.is_some());

        let effect = EngineAction::ExecuteTool {
            request: ToolRequest {
                idempotency_key: "effect-1".into(),
                request_id: "request-1".into(),
                occurrence_id: "occurrence-1".into(),
                name: "bash".into(),
                arguments_json: "{}".into(),
            },
        };
        assert!(!promotion_safe(&effect));
        assert!(!promotion_safe(&EngineAction::Wait {
            idempotency_key: "wait-1".into(),
        }));
        assert!(!promotion_safe(&EngineAction::Complete {
            idempotency_key: "complete-1".into(),
            result_json: "{}".into(),
        }));
        assert!(!promotion_safe(&EngineAction::Present {
            idempotency_key: "present-1".into(),
            directive: PresentationDirective::Status {
                activity_id: "activity-1".into(),
                text: "working".into(),
            },
        }));
    }

    #[test]
    fn successful_required_tool_satisfies_force_signal() {
        let mut signals = pending(DirectorDecision::ForceTool);
        signals.forced_tool.tool_name = Some("edit".into());
        let mut director = NativeDirectorV2::new(signals);
        director.step(&start()).unwrap();
        director
            .step(&EngineInput::ToolResult {
                request_id: "request-1".into(),
                occurrence_id: "occurrence-1".into(),
                name: "edit".into(),
                status: "success".into(),
                output: "done".into(),
                workspace_context_generation: 1,
                ledger_revision: 1,
            })
            .unwrap();
        assert!(director.signals().forced_tool.requirement.satisfied);
        assert_eq!(
            director
                .step_with_trace(&candidate(), None, DirectorTraceMode::Shadow)
                .unwrap()
                .trace
                .decision,
            DirectorDecision::Yield
        );
    }
}
