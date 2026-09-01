//! State and pure request-shaping helpers for one model round.

use std::collections::{BTreeSet, HashSet};

use hi_ai::Content;

use crate::steering::{
    EvidenceTracker, ImplementationIntent, ImplementationTracker, ReviewIntent, ToolLoopGuardrail,
    is_read_only_inspection_tool,
};
use crate::verify::WorkspaceRepairVerifier;

use super::super::progress::ProgressTracker;
use super::super::retry::{ReviewRepairState, TurnRetryState};

pub(super) fn deepseek_thinking_for_round(
    read_only_intent: Option<ReviewIntent>,
    request_text_answer: bool,
    request_cap_wrap_up: bool,
    empty_retries: u32,
) -> Option<bool> {
    if empty_retries > 0 {
        // A content-less DeepSeek round already demonstrated that the current
        // reasoning profile is not yielding a usable completion. Its bounded
        // retry should change more than sampling: disabling thinking makes
        // Flash gateways return the tool call or short answer directly.
        Some(false)
    } else {
        read_only_intent
            .is_some()
            .then_some(request_text_answer || request_cap_wrap_up)
    }
}

pub(super) fn merge_tool_call_channel(previous: &str, current: &str) -> String {
    if current == "none" {
        return previous.to_string();
    }
    if previous == "none" || previous.is_empty() {
        return current.to_string();
    }
    if previous == current {
        return previous.to_string();
    }
    "mixed".to_string()
}

/// Collapse duplicate read-only calls emitted in one model response.
///
/// The cross-round repetition guard cannot catch a model that puts the same
/// inspection call into a single tool batch multiple times. Executing those
/// copies wastes work and can make the transcript look like the agent is
/// stuck in a loop. Mutating and otherwise stateful calls are left untouched.
pub(super) fn collapse_duplicate_inspection_calls(
    content: &mut Vec<Content>,
    calls: Vec<(String, String, String)>,
) -> (Vec<(String, String, String)>, usize) {
    let mut seen = HashSet::<(String, String)>::new();
    let mut duplicate_indexes = HashSet::<usize>::new();
    let mut unique = Vec::with_capacity(calls.len());

    for (index, call) in calls.into_iter().enumerate() {
        let (_, name, arguments) = &call;
        if is_read_only_inspection_tool(name) && !seen.insert((name.clone(), arguments.clone())) {
            duplicate_indexes.insert(index);
        } else {
            unique.push(call);
        }
    }

    if duplicate_indexes.is_empty() {
        return (unique, 0);
    }

    // Keep the assistant transcript aligned with the calls that will be
    // executed. Results are paired with ToolCall blocks by position later.
    let mut tool_call_index = 0usize;
    content.retain(|block| match block {
        Content::ToolCall { .. } => {
            let keep = !duplicate_indexes.contains(&tool_call_index);
            tool_call_index += 1;
            keep
        }
        _ => true,
    });

    let duplicate_count = duplicate_indexes.len();
    (unique, duplicate_count)
}

pub(in crate::agent::turn) enum ModelRoundControl {
    Continue,
    BreakInner(bool),
    RunTools {
        calls: Vec<(String, String, String)>,
        completion_content: Vec<Content>,
        tool_specs: std::sync::Arc<[hi_ai::ToolSpec]>,
    },
}

pub(in crate::agent::turn) struct ModelRoundState<'a> {
    pub steps: &'a mut u32,
    pub empty_retries: &'a mut u32,
    pub truncation_retries: &'a mut u32,
    pub truncation_total_retries: &'a mut u32,
    pub silent_continues: &'a mut u32,
    pub generic_completion_retries: &'a mut u32,
    pub continue_total_nudges: &'a mut u32,
    pub repeat_nudges: &'a mut u32,
    pub repeat_sampling_rounds: &'a mut u32,
    pub force_tools_next: &'a mut bool,
    pub text_tool_fallback_next: &'a mut bool,
    pub force_text_answer_next: &'a mut bool,
    pub force_no_progress_final_answer_next: &'a mut bool,
    pub suppress_bookkeeping_tools_next: &'a mut bool,
    pub prev_added_no_evidence: &'a mut bool,
    pub made_tool_call: &'a mut bool,
    pub turn_start: &'a mut usize,
    pub stalled_repeating: &'a mut bool,
    pub stalled_unfinished: &'a mut bool,
    pub context_generation_seen: &'a mut u64,
    pub indexed_ledger_revision: &'a mut u64,
    pub sched_tool_calls: &'a mut u32,
    pub sched_max_concurrent: &'a mut u32,
    pub sched_serial_runs: &'a mut u32,
    pub tool_schema_tokens: &'a mut u64,
    pub ended_at_cap: &'a mut bool,
    pub cap_wrap_up_requested: &'a mut bool,
    pub review_wrap_up_requested: &'a mut bool,
    pub prev_call_sig: &'a mut Option<Vec<(String, String)>>,
    pub deepseek_strict_fallback_active: &'a mut bool,
    pub retry_state: &'a mut TurnRetryState,
    pub request_max_tokens_override: &'a mut Option<u32>,
    pub compat_fallbacks: &'a mut Vec<String>,
    pub effective_fallback_route: &'a mut Option<String>,
    pub ranked_context_paths: &'a mut BTreeSet<String>,
    pub progress_tracker: &'a mut ProgressTracker,
    pub evidence: &'a mut EvidenceTracker,
    pub implementation_tracker: &'a mut ImplementationTracker,
    pub review_repair: &'a mut ReviewRepairState,
    pub tool_guardrail: &'a mut ToolLoopGuardrail,
    pub last_verify_attributions: &'a mut Vec<hi_tools::Attribution>,
    pub tool_timeline: &'a mut Vec<crate::ToolCallEntry>,
    pub advertised_tool_names: &'a mut BTreeSet<String>,
    pub turn_snapshot: &'a mut Option<crate::verify::Snapshot>,
    pub max_steps: u32,
    pub context_task: &'a str,
    pub task_intent: crate::TaskIntent,
    pub repository_context_enabled: bool,
    pub turn_ledger_revision: u64,
    pub read_only_intent: Option<ReviewIntent>,
    pub implementation_intent: Option<ImplementationIntent>,
    pub read_only_inspection_cap: Option<u32>,
    pub expected_mutation: bool,
    pub requested_validation: bool,
    pub input: &'a str,
    pub user_prompt_tokens: u64,
    pub inspection_sprawl_intent: Option<ReviewIntent>,
    pub verifier: &'a WorkspaceRepairVerifier,
}
