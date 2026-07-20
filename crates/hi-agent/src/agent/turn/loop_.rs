//! The main turn loop: user message → model → tools → steer → workspace repair.
//!
//! Control flow still lives in one async method, but Tools and Steer phases are
//! delegated to [`super::tools`] and [`super::steer`]. Pipeline phases are named
//! in [`super::phase::TurnPhase`]:
//! `Setup → (Model → Tools → Steer)* → WorkspaceRepair → Settle → Finalize → Done`.
//!
//! Two repair systems (do not conflate):
//! - **Workspace repair** — [`crate::verify::WorkspaceRepairVerifier`] (tests/build)
//! - **Review repair** — [`crate::steering::ReviewRepairMode`] during Steer

use std::collections::BTreeSet;

use anyhow::Result;
use hi_ai::{
    ChatRequest, Content, ProviderErrorKind, RequestProfile, Role, StreamEvent, ToolMode,
    estimate_text_tokens, provider_error_kind,
};
use hi_tools::PlanStatus;

use crate::command;
use crate::compaction;
use crate::heuristics::{
    RECOVERY_SAMPLING, StallMode, looks_like_continue, looks_like_unfinished_step,
    parse_text_tool_calls, plan_has_pending_steps, recovery_sampling, recovery_telemetry,
    textcall_id_offset, tool_mode_label,
};
use crate::snapshot::changed_files_between;
use crate::steering::{
    BOOKKEEPING_REPOST_NUDGE, EvidenceTracker, IMPLEMENTATION_EMPTY_TUI_NUDGE,
    IMPLEMENTATION_NO_CHANGES_NUDGE, ImplementationIntent, ImplementationTracker, MutationRecovery,
    PLAN_REPOST_NUDGE, READ_AFTER_SEARCH_NUDGE, READ_ONLY_SAFE_CONTEXT_WINDOW, REPEAT_NUDGE,
    REREAD_NUDGE, ReviewIntent, SKIPPED_BOOKKEEPING_REPOST_RESULT, SKIPPED_PLAN_REPOST_RESULT,
    SKIPPED_REPEATED_CALL_RESULT, TOOL_PROTOCOL_RETRY_NUDGE, TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE,
    ToolLoopGuardrail, active_read_only_inspection_cap, bash_call_waits, bash_no_progress_signature,
    classify_implementation_intent, classify_read_only_intent, implementation_mentions_tui,
    implementation_text_tool_nudge, implementation_turn_prompt, inspected_paths_for_prompt,
    inspection_sprawl_exhausted, inspection_sprawl_nudge, read_only_turn_prompt,
    should_nudge_inspection_sprawl, should_nudge_read_after_repeated_search,
};
use crate::transcript::NudgeKind;
use crate::verify::{Snapshot, VerifyOutcome, WorkspaceRepairVerifier, stage_guidance};
use crate::{
    AUTO_KEEP_RECENT, MAX_TOOL_PROTOCOL_RETRIES, ReviewStatus, TRUNCATED_TOOL_CALL_NUDGE,
    TRUNCATION_NUDGE, TaskContract, TaskIntent, ToolCallEntry, TurnOutcome, TurnStatus,
    TurnStopReason, TurnTelemetry, Ui, VerificationMode, VerificationStatus,
};

use super::helpers::{
    build_turn_telemetry, effective_max_steps_for_turn, effective_model_route,
    fallback_review_line_count, task_needs_repository_context,
};
use super::phase::TurnPhase;
use super::progress::{
    NO_PROGRESS_FINAL_ANSWER_NUDGE, ProgressKind, ProgressTracker, forced_final_answer_is_unusable,
    no_progress_signature_for_calls,
};
use super::retry::{
    INCOMPLETE_STATUS, MAX_PROVIDER_OVERLOAD_RETRIES, MAX_TRANSIENT_ROUTE_RETRIES,
    ReviewRepairState, TurnRetryState, delay_label, estimate_tool_schema_tokens,
    output_cap_retry_tokens, provider_error_is_backoff_retryable, provider_overload_retry_delay,
    transient_route_retry_delay,
};

impl crate::Agent {
    /// Run one user turn to completion, emitting output through `ui`.
    ///
    /// Phases: [`TurnPhase::Setup`] → model/tool/steer loop →
    /// [`TurnPhase::WorkspaceRepair`] (optional stages; failures re-enter the
    /// model up to one initial check plus `max_verify_repairs` cycles) →
    /// [`TurnPhase::Settle`] → optional [`TurnPhase::Finalize`] →
    /// [`TurnPhase::Done`].
    pub async fn run_turn(&mut self, input: &str, ui: &mut dyn Ui) -> Result<TurnOutcome> {
        // Always land on Done, including `?` error exits mid-turn.
        let result = self.run_turn_body(input, ui).await;
        self.set_turn_phase(TurnPhase::Done);
        result
    }

    async fn run_turn_body(&mut self, input: &str, ui: &mut dyn Ui) -> Result<TurnOutcome> {
        // Phase stamp for the emerging state machine (see `phase.rs`).
        self.set_turn_phase(TurnPhase::Setup);
        let user_prompt_tokens = estimate_text_tokens(input);
        // Reset the per-turn file-read cache. It's invalidated per-key by the
        // edit tools and wholesale after `bash`, but clearing it here restores
        // its documented per-turn contract — so a file changed outside `hi`
        // between turns is re-read fresh, not served from a prior turn's cache.
        self.runtime.clear_read_cache();
        // Same per-turn contract for the diff / stub-scan caches: a new turn
        // recomputes both against its own baseline.
        self.turn_diff_cache = None;
        self.turn_stub_scan_cache = None;
        // Reconcile user/external edits before establishing this turn's
        // baseline so they are not attributed to the agent.
        self.runtime.ledger().reconcile()?;
        let turn_ledger_revision = self.runtime.ledger().revision();
        self.active_turn_ledger_revision = Some(turn_ledger_revision);
        self.active_turn_message_start = None;
        let turn_background_baseline = self.runtime.background().ids();
        let expanded_input =
            command::expand_prompt_macro(input).unwrap_or_else(|| input.to_string());
        // Synthetic goal-drive text is only transport. Contracts, context
        // ranking, review, and implementation guards need the real objective
        // and active milestone—especially explicit paths such as plan.md.
        let goal_context = self.goal_continuation_context(&expanded_input);
        let goal_drive_turn = goal_context.is_some();
        let context_task = goal_context.unwrap_or_else(|| expanded_input.clone());
        let structurally_read_only_subagent =
            self.config.subagents.is_subagent && self.config.routing.tool_mode == ToolMode::ReadOnly;
        let mut task_contract =
            TaskContract::derive(&context_task, self.config.gates.verification.clone());
        // Capability scope is authoritative for an explore child. Its quoted
        // question may contain mutation verbs ("what should we build next"),
        // but the child is an investigator, not an implementer. Letting prompt
        // wording override that scope activates mutation completion guards that
        // it can never satisfy and previously turned valid reads into denials.
        if structurally_read_only_subagent {
            task_contract.intent = TaskIntent::ReadOnly;
            task_contract.explicit_mutation = false;
        }
        self.refresh_tools_for_task(&context_task, task_contract.intent);
        let repository_context_enabled =
            task_needs_repository_context(&context_task, &task_contract);
        let mut ranked_context_paths = self
            .last_changed_files
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if repository_context_enabled {
            for path in hi_tools::ranked_paths_for_task(
                self.runtime.root(),
                &context_task,
                self.runtime.repo_map(),
                12,
            ) {
                ranked_context_paths.insert(path);
            }
        }
        self.task_context = repository_context_enabled
            .then(|| {
                let index = crate::context_index::build_task_context_index(
                    self.runtime.root(),
                    &context_task,
                    &ranked_context_paths.iter().cloned().collect::<Vec<_>>(),
                    &self.config.memory.context_exclusions,
                );
                let orientation = hi_tools::orientation_for_task(
                    self.runtime.root(),
                    &context_task,
                    self.runtime.repo_map(),
                );
                match (orientation, index) {
                    (Some(seed), Some(index)) => Some(format!("{seed}\n\n{index}")),
                    (Some(seed), None) => Some(seed),
                    (None, index) => index,
                }
            })
            .flatten();
        let mut context_generation_seen = self.runtime.context_generation();
        let mut indexed_ledger_revision = self.runtime.ledger().revision();
        let read_only_intent = classify_read_only_intent(&context_task);
        let implementation_candidate =
            if read_only_intent.is_some() || structurally_read_only_subagent {
                None
            } else if goal_drive_turn && task_contract.intent == TaskIntent::Mutation {
                Some(ImplementationIntent {
                    tui: implementation_mentions_tui(&context_task),
                })
            } else {
                classify_implementation_intent(&context_task)
            };
        let implementation_intent = implementation_candidate;
        // An explicit no-mutation request is authoritative even when broader
        // lexical contract classification saw a mutation-shaped verb. Prior
        // conversation is never consulted here.
        if read_only_intent.is_some() {
            task_contract.intent = TaskIntent::ReadOnly;
            task_contract.explicit_mutation = false;
        }
        self.last_task_contract = Some(task_contract.clone());
        self.last_task_prompt = Some(context_task.clone());
        self.refresh_system_message();
        // A turn is *expected* to mutate — and ends "incomplete · stalled"
        // when it changes no files — only for an explicit mutation request
        // ("fix the login bug"), a structured implementation task, or a goal
        // drive turn. The mutation-capable intent that ambiguous wording
        // ("how do users use it?") and tool nouns ("does cargo build build
        // hi-mlx?") default into still advertises mutating tools, but must
        // not brand a correct text-only answer as a stall.
        let expected_mutation = read_only_intent.is_none()
            && (task_contract.explicit_mutation
                || implementation_intent.is_some()
                || (goal_drive_turn && task_contract.intent == TaskIntent::Mutation));
        // Keep the legacy read-only classifier responsible for review prompt
        // shaping. A plain repository question can still have a read-only task
        // contract, and an `explore` child is structurally read-only even when its
        // wording is ambiguous. Apply the sprawl limit to either structural case
        // without imposing the rigid review response format.
        let structural_read_only_inspection = (task_contract.intent == TaskIntent::ReadOnly
            && repository_context_enabled)
            || structurally_read_only_subagent;
        let inspection_sprawl_intent = read_only_intent
            .or_else(|| structural_read_only_inspection.then_some(ReviewIntent::Review));
        let read_only_inspection_cap = inspection_sprawl_intent
            .map(|intent| active_read_only_inspection_cap(&context_task, intent));
        let turn_input = if let Some(intent) = read_only_intent {
            read_only_turn_prompt(&context_task, intent)
        } else if let Some(intent) = implementation_intent {
            implementation_turn_prompt(&context_task, intent)
        } else {
            context_task.clone()
        };
        let input = turn_input.as_str();
        let model_turn_input = match self.managed_rsi_context.as_deref() {
            Some(context) if !context.is_empty() => format!(
                "{turn_input}\n\nManaged RSI prior conversation context (reference only; it does not change the current task's mutation requirements):\n{context}"
            ),
            _ => turn_input.clone(),
        };
        self.reset_last_turn_usage(user_prompt_tokens);
        self.last_turn_outcome = None;
        self.last_effective_route = effective_model_route(&self.config, None);

        // A top-level session the user restricted to ChatOnly/ReadOnly gets a
        // clear early "your mode blocks edits" error when the prompt clearly asks
        // for mutation. This must NOT fire for a subagent: an `explore` child
        // runs ReadOnly as internal capability-scoping (not a user restriction),
        // and its task text naturally contains verbs like "find where X creates
        // Y" — pattern-matching that as a mutating request would abort the child
        // before its first model call and return "(no answer)". The child simply
        // isn't advertised mutating tools, so it's safe to let it run and answer.
        if read_only_intent.is_none()
            && !self.config.subagents.is_subagent
            && self.tools_unavailable_for(input)
        {
            self.last_verify = None;
            self.last_changed_files.clear();
            self.last_file_changes.clear();
            self.last_compat_fallbacks.clear();
            self.last_turn_telemetry = TurnTelemetry::default();
            let preserve_plan = (goal_drive_turn || looks_like_continue(&context_task))
                && plan_has_pending_steps(&self.last_plan);
            if !preserve_plan && !self.last_plan.is_empty() {
                self.last_plan.clear();
                if let Some(session) = self.session.as_mut() {
                    session.clear_plan()?;
                }
                ui.plan(&[]);
            }
            self.messages.strip_trailing_nudges();
            self.persisted = self.persisted.min(self.messages.len());
            self.persist()?;
            ui.turn_error(
                "tools",
                &format!(
                    "tool mode {} blocks file edits and shell commands",
                    tool_mode_label(self.config.routing.tool_mode)
                ),
                "",
            );
            let outcome = TurnOutcome {
                status: TurnStatus::Blocked,
                verification: VerificationStatus::NotApplicable,
                review: ReviewStatus::NotRequired,
                stop_reason: TurnStopReason::ToolModeDenied,
                changed_files: Vec::new(),
                verified_workspace_revision: None,
                effective_route: effective_model_route(&self.config, None),
            };
            self.last_turn_outcome = Some(outcome.clone());
            self.active_turn_ledger_revision = None;
            self.active_turn_message_start = None;
            return Ok(outcome);
        }
        let mut turn_checkpoint_allowed = None;
        let mut turn_checkpoint_created = false;

        // If the context window is filling up, reclaim room before adding more,
        // so the session keeps going instead of overflowing. Two tiers: a free,
        // deterministic elision of old tool output first; then, only if still
        // heavy, the configured summarizing strategy. Best-effort — a failed
        // model call just leaves the (already elided) history as-is.
        //
        // The outer trigger uses the provider-reported `context_used` (the last
        // request's occupancy — the most accurate signal, and only meaningful
        // once a real request has happened, so a fresh session isn't
        // over-eagerly compacted). Tier 2 below gates on a local token estimate
        // instead, because `context_used` is stale by then.
        if self.config.memory.auto_compact
            && let Some(window) = self.config.routing.context_window
            && window > 0
            && self.context_used * 100 >= u64::from(window) * self.config.memory.auto_compact_percent
        {
            ui.status(&format!(
                "context ~{}% full — compacting to free room",
                self.context_used * 100 / u64::from(window)
            ));
            // Tier 1: deterministic, no model call. Only old turns are eligible.
            if let Some(split) =
                compaction::recent_split(self.messages.as_slice(), AUTO_KEEP_RECENT)
                && compaction::elide_tool_outputs(self.messages.mutate_slice(), split) > 0
            {
                self.runtime.invalidate_context_after_compaction();
            }
            // Tier 2: only if still heavy. `context_used` reflects the
            // pre-elision request and is now stale, so gate on a local estimate.
            let target = u64::from(window) * self.config.memory.compact_target_percent / 100;
            if compaction::estimate_tokens(self.messages.as_slice()) > target {
                let _ = self.compact(ui).await;
            }
            self.context_used = 0;
        }

        self.messages.strip_trailing_nudges();
        self.persisted = self.persisted.min(self.messages.len());
        let mut turn_start = self.messages.len();
        self.active_turn_message_start = Some(turn_start);
        self.messages.push_user_or_fold(&model_turn_input);
        self.last_verify = None;
        self.last_changed_files.clear();
        self.last_file_changes.clear();
        self.last_compat_fallbacks.clear();
        self.last_turn_telemetry.verification_executions.clear();
        // Preserve only an unfinished plan that the user explicitly continues.
        // Clearing must also be emitted: the TUI owns a pinned copy and cannot
        // infer that the agent cleared its internal state.
        let preserve_plan = (goal_drive_turn || looks_like_continue(&context_task))
            && plan_has_pending_steps(&self.last_plan);
        if !preserve_plan && !self.last_plan.is_empty() {
            self.last_plan.clear();
            if let Some(session) = self.session.as_mut() {
                session.clear_plan()?;
            }
            ui.plan(&[]);
        }
        let mut compat_fallbacks = Vec::new();
        let mut effective_fallback_route: Option<String> = None;

        let resolved_verify_stages = self
            .config.gates.verification
            .resolved_stages(self.runtime.root());
        let verify_rounds = self.config.gates.max_verify_repairs.saturating_add(1);
        // Workspace repair only — not review-answer repair (see ReviewRepairState).
        let mut verifier = if matches!(&self.config.gates.verification, VerificationMode::Auto) {
            WorkspaceRepairVerifier::automatic(resolved_verify_stages, verify_rounds)
        } else {
            WorkspaceRepairVerifier::new(resolved_verify_stages, verify_rounds)
        };
        // Mid-turn LSP + affected cargo check state (dedupes packages across batches).
        let mut fast_feedback = super::fast_feedback::FastFeedbackState::default();
        let max_steps = effective_max_steps_for_turn(
            &self.config,
            task_contract.intent,
            read_only_intent,
            implementation_intent,
        );
        let max_parallel_tools = self.config.loop_limits.max_parallel_tools.max(1);
        let mut steps = 0u32;
        let mut empty_retries = 0u32;
        // Consecutive output-limit continuations. This is a stall budget, so it
        // resets after any non-truncated model response/tool progress.
        let mut truncation_retries = 0u32;
        // Cumulative truncation nudges for telemetry/UI summaries. Unlike the
        // consecutive budget above, this should not reset mid-turn.
        let mut truncation_total_retries = 0u32;
        let mut silent_continues = 0u32;
        let mut continue_total_nudges = 0u32;
        let mut repeat_nudges = 0u32;
        let mut progress_tracker = ProgressTracker::default();
        // Set after a silent-continue nudge: force the *next* round to call a
        // tool (`tool_choice: required`) instead of letting the model narrate
        // again or return an empty completion. Some models (e.g. weaker
        // OpenAI-compat coders) intermittently emit text-only or empty responses
        // when asked to continue; backing the "use your tools; act, don't
        // narrate" nudge with a hard tool-choice makes them actually act. Stays
        // set across empty-retries and re-nudges until the model emits a tool
        // call, then clears (see the made_tool_call path). Only takes effect when
        // tools are otherwise freely available (config tool_mode Auto).
        let mut force_tools_next = false;
        // Bounded discovery narrows the advertised catalog until the model
        // records a plan or makes the requested edit.
        let mut mutation_recovery = MutationRecovery::default();
        // A model-authored plan is only a proposal until deterministic
        // verification passes for the settled workspace revision. Keeping it
        // turn-local prevents failed, unverified, cancelled, or infrastructure-
        // error turns from leaking goal progress into the live session.
        let mut plan_updated_goal = false;
        let mut proposed_goal: Option<crate::Goal> = None;
        // The goal as it stood at turn start — so the skeptic gate can review
        // against the sub-goal that was active *before* the turn (update_plan may
        // have marked it done mid-turn) and, on an objection, revert the turn's
        // goal progress.
        let goal_before = self.structured_goal.clone();
        // Scheduler parallelism counters: how many calls ran this turn, the
        // largest concurrent ready-batch, and how many ran serially (bash or a
        // lone ready call). Flushed into telemetry so the dep-aware scheduler's
        // concurrency is measurable, not shipped on faith.
        let mut sched_tool_calls = 0u32;
        let mut sched_max_concurrent = 0u32;
        let mut sched_serial_runs = 0u32;
        // Per-tool-call timeline: each call's name, path, duration, and error
        // status, flushed into telemetry so `--report` can diagnose where time
        // went and which calls failed.
        let mut tool_timeline: Vec<ToolCallEntry> = Vec::new();
        let mut advertised_tool_names = BTreeSet::new();
        let mut tool_schema_tokens = 0_u64;
        let mut evidence = EvidenceTracker::default();
        let mut review_repair = ReviewRepairState::default();
        let mut independent_review_status = ReviewStatus::NotRequired;
        let mut independent_review_repairs = 0_u32;
        let mut verification_infrastructure_error = false;
        let mut verification_unstable = false;
        // A pass is bound to both the ledger event number and the full content
        // digest observed immediately after the verifier. Later workspace
        // activity must never inherit that pass.
        let mut verified_at: Option<(u64, String)> = None;
        // Whether the model or deterministic preflight has run a tool this
        // turn (kept for finalization gating — a plain Q&A turn doesn't need a
        // recap).
        let mut made_tool_call = false;
        let mut implementation_tracker = ImplementationTracker::default();
        let mut empty_tui_needs_project = false;
        if let Some(intent) = read_only_intent
            && self.config.gates.read_only_preflight
            && !self
                .config.rsi.remote_switch
                .as_ref()
                .is_some_and(|enabled| enabled.load(std::sync::atomic::Ordering::SeqCst))
            && !matches!(self.config.routing.tool_mode, ToolMode::ChatOnly)
        {
            let preflight = self
                .run_read_only_preflight(
                    intent,
                    read_only_inspection_cap.unwrap_or_else(|| evidence.inspection_attempt_count()),
                    ui,
                    &mut evidence,
                    &mut tool_timeline,
                    self.config.loop_limits.max_tool_calls.saturating_sub(sched_tool_calls),
                )
                .await;
            if preflight.executed > 0 {
                made_tool_call = true;
                sched_tool_calls = sched_tool_calls.saturating_add(preflight.executed);
                sched_serial_runs = sched_serial_runs.saturating_add(preflight.serial_runs);
                sched_max_concurrent = sched_max_concurrent.max(preflight.max_concurrent_batch);
            }
        }
        if implementation_intent.is_some()
            && !self
                .config.rsi.remote_switch
                .as_ref()
                .is_some_and(|enabled| enabled.load(std::sync::atomic::Ordering::SeqCst))
            && !matches!(self.config.routing.tool_mode, ToolMode::ChatOnly)
            && sched_tool_calls < self.config.loop_limits.max_tool_calls
        {
            let preflight_calls = self
                .run_implementation_preflight(ui, &mut implementation_tracker, &mut tool_timeline)
                .await;
            if preflight_calls > 0 {
                made_tool_call = true;
                sched_tool_calls = sched_tool_calls.saturating_add(preflight_calls);
                sched_serial_runs = sched_serial_runs.saturating_add(preflight_calls);
                sched_max_concurrent = sched_max_concurrent.max(1);
            }
            empty_tui_needs_project = implementation_intent.is_some_and(|intent| intent.tui)
                && implementation_tracker.preferred_validation.is_none();
        }
        // Signature (name, arguments) of the previous round's tool calls, to
        // spot a model re-issuing the exact same call and looping on it.
        let mut prev_call_sig: Option<Vec<(String, String)>> = None;
        // Whether the previous executed round added no new evidence (every call
        // was a read-only inspection already seen). Used by the no-new-evidence
        // cycle guard to fire only on the *second* consecutive wasted round,
        // preserving a single legitimate re-inspection after new evidence.
        let mut prev_added_no_evidence = false;
        let mut retry_state = TurnRetryState::default();
        let mut request_max_tokens_override: Option<u32> = None;
        let mut text_tool_fallback_next = false;
        let mut force_text_answer_next = false;
        let mut force_no_progress_final_answer_next = false;
        // After a bookkeeping-repost nudge, withhold the bookkeeping tools
        // (`update_plan`, `record_decision`) from the next request's tool
        // list. A bookkeeping-fixated model (observed live) keeps re-posting
        // meta-work through every nudge — and when only `update_plan` was
        // withheld it slid to repeating `record_decision` instead. Clear
        // feedback alone doesn't break the loop; removing the whole family
        // for one round forces a tool that does real work.
        let mut suppress_bookkeeping_tools_next = false;
        // Consecutive rounds skipped by the repeat guard, driving recovery
        // sampling: a model re-emitting the identical call each round is stuck
        // in a token-level loop that only hotter sampling breaks. Resets as
        // soon as the model issues a different round, so later rounds run at
        // the configured sampling again (unlike the cumulative
        // `repeat_nudges` budget, which never resets within a turn).
        let mut repeat_sampling_rounds = 0u32;
        let mut tool_guardrail = ToolLoopGuardrail::default();
        // Whether the turn ended because the model kept re-issuing the exact
        // same tool call through the whole repeat-nudge budget (drives the
        // stalled telemetry and skips the finalization recap).
        let mut stalled_repeating = false;
        // Whether the turn ended without enough evidence for a read-only review.
        let mut stalled_unfinished = false;
        // One-shot coding verify-obligation re-entry (Phase C). Prevents a
        // mutation-shaped turn from settling as "done" without green evidence
        // when a pipeline is configured — fires at most once per turn.
        let mut obligation_nudge_fired = false;
        // Whether the turn was cut short by the per-turn step cap, so the
        // finalization recap is skipped (the work may be incomplete).
        let mut ended_at_cap = false;
        // Attributions parsed from the most recent verify failure — captured
        // here so they survive to turn end and can be flushed into telemetry.
        let mut last_verify_attributions: Vec<hi_tools::Attribution> = Vec::new();
        // Snapshot the turn baseline lazily. Read-only/chat turns should not
        // walk the whole workspace just to prove nothing changed; the baseline
        // is captured before the first actual mutation, or before verification
        // when verify stages are configured.
        let mut turn_snapshot: Option<Snapshot> = None;
        // Snapshot from the most recent verify check. Reused at turn end to
        // avoid a second full tree walk when verify already took one.

        if empty_tui_needs_project {
            force_tools_next = true;
            self.messages
                .push_nudge(NudgeKind::Continue, IMPLEMENTATION_EMPTY_TUI_NUDGE);
        }

        'turn: loop {
            // Inner loop: Model → Tools → Steer until tools stop, or step cap.
            let hit_cap = loop {
                self.set_turn_phase(TurnPhase::Model);
                if steps >= max_steps {
                    break true;
                }
                steps += 1;

                // Mid-turn steering: inject any messages the user typed while
                // the turn was running, as genuine user messages, before the
                // next model round. This is a safe transcript boundary — the
                // prior round's tool calls are all resolved — so the folding
                // nudge push keeps provider alternation valid. The model
                // decides how to weigh them; we add no deferral directive.
                let interjected = self.interjections.drain();
                if !interjected.is_empty() {
                    for message in &interjected {
                        self.messages.push_nudge_or_fold(
                            NudgeKind::Interjection,
                            format!(
                                "The user sent this message while you were working — take it into account now:\n{message}"
                            ),
                        );
                    }
                    ui.status(&format!(
                        "✉ received {} message(s) from you mid-turn — factoring them in",
                        interjected.len()
                    ));
                }

                // After a content-less/garbled round, resample hotter and with
                // nucleus + frequency penalty on the retry to break out of the
                // low-entropy attractor that produced it (cf. minion's recovery
                // sampling). Bounded, and only while consecutively stalling —
                // `empty_retries` resets on real output, so a normal round runs at
                // the configured sampling. Toggleable via HI_RECOVERY_SAMPLING for
                // A/B-ing on the eval harness.
                let sampling_retries = empty_retries
                    .max(retry_state.protocol_retries)
                    .max(repeat_sampling_rounds);
                let (sampling_mode, sampling_budget) = if repeat_sampling_rounds > 0
                    && repeat_sampling_rounds >= empty_retries
                    && repeat_sampling_rounds >= retry_state.protocol_retries
                {
                    // The model is deterministically re-emitting the same tool
                    // call round after round (observed live: four byte-identical
                    // `update_plan` calls despite nudges and withheld tools).
                    // Hotter sampling + a frequency penalty is what actually
                    // breaks a token-level loop; nudge text alone doesn't.
                    (StallMode::Repeat, self.config.loop_limits.max_repeat_nudges)
                } else if retry_state.protocol_retries > empty_retries {
                    (StallMode::Empty, MAX_TOOL_PROTOCOL_RETRIES)
                } else {
                    (StallMode::Empty, self.config.loop_limits.max_empty_retries)
                };
                let (temperature, top_p, frequency_penalty) = recovery_sampling(
                    sampling_retries,
                    self.config.routing.temperature,
                    *RECOVERY_SAMPLING,
                );

                // Telemetry for the recovery-sampling A/B: emit a concise debug
                // line only when sampling is actually being changed (recovery on
                // and this is a retry), so ordinary runs stay quiet.
                if let Some(line) = recovery_telemetry(
                    sampling_mode,
                    sampling_retries,
                    sampling_budget,
                    temperature,
                    top_p,
                    frequency_penalty,
                    *RECOVERY_SAMPLING,
                ) {
                    ui.nudge(&line);
                }

                let context_safety_window = read_only_intent
                    .is_some()
                    .then_some(READ_ONLY_SAFE_CONTEXT_WINDOW);
                self.elide_in_turn_context_if_needed(ui, context_safety_window);

                self.refresh_active_task_context(
                    &context_task,
                    repository_context_enabled,
                    turn_ledger_revision,
                    &mut ranked_context_paths,
                    &mut context_generation_seen,
                    &mut indexed_ledger_revision,
                );

                self.messages.repair_invalid_tool_call_arguments();

                // Debug-mode invariant check: the transcript we're about to send
                // must be provider-safe (every tool_use answered, no consecutive
                // user messages). Cheap in release builds; in debug it catches
                // the orphan-tool_use class of bug at the source.
                debug_assert!(
                    self.messages.validate_for_provider().is_ok(),
                    "transcript invariant violated before provider send"
                );

                let request_text_tool_fallback = text_tool_fallback_next;
                text_tool_fallback_next = false;
                let request_text_answer = force_text_answer_next;
                force_text_answer_next = false;
                let request_no_progress_final_answer = force_no_progress_final_answer_next;
                if request_no_progress_final_answer {
                    progress_tracker.record_forced_final_answer_attempt();
                }
                force_no_progress_final_answer_next = false;

                // After a continue-nudge, force this round to call a tool rather
                // than narrate again or come back empty. Only when tools are
                // freely available (Auto): never override an intentional
                // ChatOnly/ReadOnly restriction, and Required already forces.
                let tool_mode = if request_text_tool_fallback
                    || request_text_answer
                    || request_no_progress_final_answer
                {
                    ToolMode::ChatOnly
                } else if force_tools_next && self.config.routing.tool_mode == ToolMode::Auto {
                    ToolMode::Required
                } else {
                    self.config.routing.tool_mode
                };
                let tool_availability_mode = if request_text_tool_fallback
                    || request_text_answer
                    || request_no_progress_final_answer
                {
                    ToolMode::ChatOnly
                } else if read_only_intent.is_some()
                    && !matches!(self.config.routing.tool_mode, ToolMode::ChatOnly)
                {
                    ToolMode::ReadOnly
                } else {
                    self.config.routing.tool_mode
                };
                let requested_request_max_tokens =
                    request_max_tokens_override.unwrap_or(self.config.routing.max_tokens);
                let mut request_tools = self.request_tools_for(tool_availability_mode);
                if suppress_bookkeeping_tools_next {
                    suppress_bookkeeping_tools_next = false;
                    // Only withhold when other tools remain — an empty tool
                    // list with tool_choice=required would be a provider error.
                    if request_tools
                        .iter()
                        .any(|tool| !hi_tools::is_coordination(&tool.name))
                    {
                        request_tools = request_tools
                            .iter()
                            .filter(|tool| !hi_tools::is_coordination(&tool.name))
                            .cloned()
                            .collect();
                    }
                }
                advertised_tool_names.extend(request_tools.iter().map(|tool| tool.name.clone()));
                let request_tool_schema_tokens = estimate_tool_schema_tokens(&request_tools);
                tool_schema_tokens = tool_schema_tokens.max(request_tool_schema_tokens);
                let context_preflight = match self.ensure_request_fits_context(
                    input,
                    turn_start,
                    requested_request_max_tokens,
                    request_tool_schema_tokens,
                    context_safety_window,
                    ui,
                ) {
                    Ok(context_preflight) => context_preflight,
                    Err(err) => {
                        self.reconcile_error_turn_changes(turn_ledger_revision)?;
                        self.truncate_messages(turn_start);
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        self.last_compat_fallbacks = compat_fallbacks.clone();
                        self.last_turn_telemetry = build_turn_telemetry(
                            max_steps,
                            verifier.round(),
                            empty_retries,
                            repeat_nudges,
                            continue_total_nudges,
                            truncation_total_retries,
                            &progress_tracker,
                            ended_at_cap,
                            stalled_unfinished,
                            stalled_repeating,
                            &last_verify_attributions,
                            verifier.executions(),
                            sched_tool_calls,
                            sched_max_concurrent,
                            sched_serial_runs,
                            &tool_timeline,
                            &evidence,
                            &review_repair,
                        );
                        let _ = self.persist();
                        let (kind, guidance) = crate::ui::classify_error(&err);
                        ui.turn_error(kind, &err.to_string(), guidance);
                        self.last_effective_route = effective_model_route(
                            &self.config,
                            effective_fallback_route.as_deref(),
                        );
                        return Err(err);
                    }
                };
                if context_preflight.dropped_prior_context {
                    turn_start = self.messages.len().saturating_sub(1);
                }
                // Context fitting may itself compact or elide the transcript.
                // Consume that generation before constructing the request.
                self.refresh_active_task_context(
                    &context_task,
                    repository_context_enabled,
                    turn_ledger_revision,
                    &mut ranked_context_paths,
                    &mut context_generation_seen,
                    &mut indexed_ledger_revision,
                );
                let request_max_tokens = context_preflight.max_tokens;
                if request_max_tokens != requested_request_max_tokens {
                    request_max_tokens_override = Some(request_max_tokens);
                }
                let request = ChatRequest {
                    model: self.config.routing.model.clone(),
                    user_turn: true,
                    canonical_objective: Some(context_task.clone()),
                    messages: self.messages.arc(),
                    tools: request_tools,
                    max_tokens: request_max_tokens,
                    temperature,
                    top_p,
                    frequency_penalty,
                    thinking_budget: self.config.routing.thinking_budget,
                    reasoning_effort: self.config.routing.reasoning_effort,
                    profile: RequestProfile {
                        compat: self.config.routing.compat,
                        tool_mode,
                        stream_usage: None,
                    },
                };

                let buffer_read_only_review_text =
                    read_only_intent.is_some() || implementation_intent.is_some();
                let mut buffered_assistant_text = String::new();
                let mut streamed_assistant_text = false;
                let mut sink = |event: StreamEvent| match event {
                    StreamEvent::Text(text) => {
                        if buffer_read_only_review_text {
                            buffered_assistant_text.push_str(&text);
                        } else {
                            streamed_assistant_text = true;
                            ui.assistant_text(&text);
                        }
                    }
                    StreamEvent::Reasoning(text) => ui.assistant_reasoning(&text),
                    StreamEvent::Status(text) => {
                        if let Some(fallback) = text.strip_prefix("compat: ") {
                            compat_fallbacks.push(fallback.to_string());
                        }
                        if let Some(route) = text.rsplit_once("falling back to ").map(|(_, r)| r) {
                            effective_fallback_route = Some(route.trim().to_string());
                        }
                        ui.status(&text);
                    }
                };
                let mut completion = match self.provider.stream(request, &mut sink).await {
                    Ok(completion) => {
                        retry_state.record_provider_success();
                        completion
                    }
                    Err(err)
                        if !retry_state.output_cap_retry_attempted
                            && hi_ai::provider_output_cap_error(&err)
                                .and_then(|cap| output_cap_retry_tokens(request_max_tokens, cap))
                                .is_some() =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        retry_state.output_cap_retry_attempted = true;
                        let new_max = hi_ai::provider_output_cap_error(&err)
                            .and_then(|cap| output_cap_retry_tokens(request_max_tokens, cap))
                            .expect("guard checked retry tokens");
                        request_max_tokens_override = Some(new_max);
                        ui.nudge(&format!(
                            "provider rejected the output budget; retrying this turn with max_tokens={new_max}"
                        ));
                        continue;
                    }
                    Err(err)
                        if retry_state.provider_overload_retries
                            < MAX_PROVIDER_OVERLOAD_RETRIES
                            && provider_error_is_backoff_retryable(&err) =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        retry_state.provider_overload_retries += 1;
                        let retry = retry_state.provider_overload_retries;
                        let delay = provider_overload_retry_delay(retry, &err);
                        let reason = if provider_error_kind(&err)
                            == Some(ProviderErrorKind::RateLimit)
                        {
                            "rate limited"
                        } else {
                            "request did not complete"
                        };
                        ui.nudge(&format!(
                            "{reason}; retrying {} ({retry}/{MAX_PROVIDER_OVERLOAD_RETRIES})",
                            delay_label(delay)
                        ));
                        if !delay.is_zero() {
                            tokio::time::sleep(delay).await;
                        }
                        continue;
                    }
                    Err(err)
                        if retry_state.transient_route_retries < MAX_TRANSIENT_ROUTE_RETRIES
                            && hi_ai::provider_route_error_is_retryable(&err) =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        retry_state.transient_route_retries += 1;
                        let retry = retry_state.transient_route_retries;
                        let delay = transient_route_retry_delay(retry, &err);
                        ui.nudge(&format!(
                            "request did not complete; retrying {} ({retry}/{MAX_TRANSIENT_ROUTE_RETRIES})",
                            delay_label(delay)
                        ));
                        if !delay.is_zero() {
                            tokio::time::sleep(delay).await;
                        }
                        continue;
                    }
                    Err(err)
                        if provider_error_kind(&err)
                            == Some(ProviderErrorKind::RequestTooLarge) =>
                    {
                        let mut context_drop_persistence_failed = false;
                        if !retry_state.request_too_large_retried {
                            match self.retry_after_request_too_large(input, turn_start, ui) {
                                Ok(true) => {
                                    retry_state.request_too_large_retried = true;
                                    turn_start = self.messages.len().saturating_sub(1);
                                    continue;
                                }
                                Ok(false) => {}
                                Err(persist_err) => {
                                    ui.status(&format!(
                                        "couldn't persist dropped-context retry state: {persist_err}"
                                    ));
                                    context_drop_persistence_failed = true;
                                }
                            }
                        }
                        self.truncate_messages(turn_start);
                        if context_drop_persistence_failed {
                            ui.status(
                                "request exceeds the provider limit, and prior context could not be \
                                 safely dropped because the session boundary was not persisted; fix \
                                 session storage or start a fresh/cleared session, then retry",
                            );
                        } else {
                            ui.status(
                                "request still exceeds the provider limit with prior context removed; \
                                 shorten the prompt or attached input, then retry",
                            );
                        }
                        self.add_error_usage(&err);
                        self.reconcile_error_turn_changes(turn_ledger_revision)?;
                        self.emit_usage(ui);
                        self.last_compat_fallbacks = compat_fallbacks.clone();
                        self.last_turn_telemetry = build_turn_telemetry(
                            max_steps,
                            verifier.round(),
                            empty_retries,
                            repeat_nudges,
                            continue_total_nudges,
                            truncation_total_retries,
                            &progress_tracker,
                            ended_at_cap,
                            stalled_unfinished,
                            stalled_repeating,
                            &last_verify_attributions,
                            verifier.executions(),
                            sched_tool_calls,
                            sched_max_concurrent,
                            sched_serial_runs,
                            &tool_timeline,
                            &evidence,
                            &review_repair,
                        );
                        let _ = self.persist();
                        let (kind, guidance) = crate::ui::classify_error(&err);
                        ui.turn_error(kind, &err.to_string(), guidance);
                        self.last_effective_route = effective_model_route(
                            &self.config,
                            effective_fallback_route.as_deref(),
                        );
                        return Err(err);
                    }
                    Err(err)
                        if provider_error_kind(&err) == Some(ProviderErrorKind::ToolProtocol)
                            && retry_state.protocol_retries < MAX_TOOL_PROTOCOL_RETRIES
                            && retry_state.protocol_failures_total
                                < crate::MAX_TOOL_PROTOCOL_FAILURES =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        retry_state.protocol_retries += 1;
                        retry_state.protocol_failures_total += 1;
                        let protocol_retries = retry_state.protocol_retries;
                        if implementation_intent.is_some() || made_tool_call {
                            force_tools_next = true;
                        }
                        ui.nudge(&format!(
                            "⚠ the model emitted an invalid tool turn — retrying with tool-format guidance ({protocol_retries}/{MAX_TOOL_PROTOCOL_RETRIES})"
                        ));
                        if self
                            .messages
                            .as_slice()
                            .last()
                            .is_some_and(|message| message.role == Role::User)
                        {
                            self.messages.push_user_or_fold(TOOL_PROTOCOL_RETRY_NUDGE);
                        } else {
                            self.messages
                                .push_nudge(NudgeKind::Continue, TOOL_PROTOCOL_RETRY_NUDGE);
                        }
                        continue;
                    }
                    Err(err)
                        if provider_error_kind(&err) == Some(ProviderErrorKind::ToolProtocol)
                            && implementation_intent.is_some()
                            && retry_state.protocol_text_fallbacks < 1 =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        retry_state.protocol_text_fallbacks += 1;
                        text_tool_fallback_next = true;
                        force_tools_next = false;
                        ui.status(
                            "structured tool calls kept failing; falling back to plain-text tool-call parsing",
                        );
                        if self
                            .messages
                            .as_slice()
                            .last()
                            .is_some_and(|message| message.role == Role::User)
                        {
                            self.messages
                                .push_user_or_fold(TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE);
                        } else {
                            self.messages
                                .push_nudge(NudgeKind::Continue, TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE);
                        }
                        continue;
                    }
                    Err(err)
                        if provider_error_kind(&err) == Some(ProviderErrorKind::ToolProtocol) =>
                    {
                        // Both the consecutive and cumulative invalid-tool-turn
                        // budgets are spent. A model that alternates a valid tool
                        // call with an invalid turn keeps resetting the consecutive
                        // counter, so without the cumulative cap this nudge-and-retry
                        // loop runs forever (spinning CPU, burning tokens). End the
                        // turn instead so the driver/user regains control; on a
                        // long-horizon drive the next turn resumes with a fresh budget.
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        ui.status(
                            "⚠ the model kept emitting invalid tool turns — ending the turn; /retry or continue to resume",
                        );
                        break false;
                    }
                    // A transient generation flake — a malformed/garbled stream or
                    // an empty completion. Treat it like a content-less response:
                    // flush, then silently re-run with hotter recovery sampling (a
                    // fresh request, with its own transport retries) up to the same
                    // budget, instead of failing the turn. Terminal errors (auth,
                    // rate limits, ...) fall through to the abort below. Invalid tool turns
                    // use the protocol-specific nudge path above.
                    Err(err)
                        if empty_retries < self.config.loop_limits.max_empty_retries
                            && matches!(
                                provider_error_kind(&err),
                                Some(
                                    ProviderErrorKind::MalformedStream
                                        | ProviderErrorKind::EmptyCompletion
                                )
                            ) =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        empty_retries += 1;
                        if made_tool_call {
                            self.nudge_after_post_tool_empty_response(
                                &mut force_tools_next,
                                implementation_intent.is_some(),
                            );
                        }
                        ui.nudge(&format!(
                            "⚠ the model's response didn't come through cleanly — \
                             retrying ({empty_retries}/{})",
                            self.config.loop_limits.max_empty_retries
                        ));
                        continue;
                    }
                    Err(err) => {
                        self.add_error_usage(&err);
                        self.reconcile_error_turn_changes(turn_ledger_revision)?;
                        self.emit_usage(ui);
                        if self.last_changed_files.is_empty()
                            && let Some(turn_snapshot) = turn_snapshot.as_ref()
                        {
                            self.messages.strip_trailing_nudges();
                            if let Ok(end_snapshot) = self.snapshot_cached().await {
                                self.last_changed_files =
                                    changed_files_between(turn_snapshot, &end_snapshot);
                            }
                        }
                        // With no model tool call, any concurrent workspace
                        // change was external to this failed attempt. Preserve
                        // it in the report, but never retain the failed user
                        // prompt or retry guidance in conversation history.
                        if !made_tool_call {
                            self.truncate_messages(turn_start);
                        }
                        self.last_compat_fallbacks = compat_fallbacks.clone();
                        self.last_turn_telemetry = build_turn_telemetry(
                            max_steps,
                            verifier.round(),
                            empty_retries,
                            repeat_nudges,
                            continue_total_nudges,
                            truncation_total_retries,
                            &progress_tracker,
                            ended_at_cap,
                            stalled_unfinished,
                            stalled_repeating,
                            &last_verify_attributions,
                            verifier.executions(),
                            sched_tool_calls,
                            sched_max_concurrent,
                            sched_serial_runs,
                            &tool_timeline,
                            &evidence,
                            &review_repair,
                        );
                        let _ = self.persist();
                        let (kind, guidance) = crate::ui::classify_error(&err);
                        ui.turn_error(kind, &err.to_string(), guidance);
                        self.last_effective_route = effective_model_route(
                            &self.config,
                            effective_fallback_route.as_deref(),
                        );
                        return Err(err);
                    }
                };
                if !buffer_read_only_review_text {
                    ui.assistant_end();
                }

                self.add_usage(completion.usage);
                // Let the frontend show the running total climb mid-turn.
                self.emit_usage(ui);

                // Truncation recovery: the model hit the output token cap
                // (`stop_reason: "length"` / `"max_tokens"`) mid-generation.
                // The response was cut off, not finished — record what it
                // produced and nudge it to continue from the cutoff, instead
                // of treating the truncation as a natural stop (which would
                // end the turn on a half-finished output and leave the model
                // "picking up where it stalled" on the next prompt). Bounded
                // by a *dedicated* truncation budget (separate from
                // `empty_retries`) so a big task that legitimately hits the
                // cap several times can still finish without the user typing
                // "continue".
                let truncated = matches!(
                    completion.stop_reason.as_deref(),
                    Some("length" | "max_tokens")
                );
                if truncated && truncation_retries < self.config.loop_limits.max_truncation_retries {
                    truncation_retries += 1;
                    truncation_total_retries += 1;
                    ui.nudge(&format!(
                        "⚠ the model hit the output token limit — continuing ({truncation_retries}/{})",
                        self.config.loop_limits.max_truncation_retries
                    ));
                    // Clean text-embedded tool-call JSON (local models) from the
                    // truncated content before recording. Complete tool calls are
                    // extracted and stripped; partial JSON (cut off mid-generation)
                    // stays as text so the model can continue from the cutoff.
                    // Structured ToolCall blocks are stripped: a truncated tool call
                    // has partial/malformed arguments and was never executed, so it
                    // has no matching tool_result. Leaving it in would create an
                    // orphan tool_use that providers reject on the next request.
                    let partial_tool_call =
                        self.clean_text_tool_calls_from_content(&mut completion.content);
                    let truncated_text = completion
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            Content::Text(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let active_tool_work = read_only_intent.is_none()
                        && (implementation_intent.is_some()
                            || made_tool_call
                            || implementation_tracker.mutation_seen
                            || plan_has_pending_steps(&self.last_plan)
                            || looks_like_unfinished_step(&truncated_text));
                    if (partial_tool_call || active_tool_work)
                        && self.config.routing.tool_mode == ToolMode::Auto
                    {
                        force_tools_next = true;
                    }
                    self.messages
                        .push_assistant_text_only(std::mem::take(&mut completion.content));
                    self.messages.push_nudge(
                        NudgeKind::Truncation,
                        if partial_tool_call || active_tool_work {
                            TRUNCATED_TOOL_CALL_NUDGE
                        } else {
                            TRUNCATION_NUDGE
                        },
                    );
                    continue;
                }
                // Truncation budget exhausted: the model kept hitting the output
                // token cap through the whole retry budget. Record the truncated
                // output (stripping partial tool calls, as above) and warn the
                // user — the task may be incomplete. Don't silently end the turn
                // on a half-finished output without surfacing what happened.
                if truncated {
                    self.clean_text_tool_calls_from_content(&mut completion.content);
                    self.messages
                        .push_assistant_text_only(std::mem::take(&mut completion.content));
                    stalled_unfinished = true;
                    ui.nudge(&format!(
                        "⚠ the model hit the output token limit {max} times — the task may be \
                         incomplete. /retry, or send 'continue'.",
                        max = self.config.loop_limits.max_truncation_retries,
                    ));
                    break false;
                }
                // A public RSI response is terminal, not a local planning round to nudge.
                if completion.stop_reason.as_deref() == Some("rsi_remote_completed") {
                    let answer = completion
                        .content
                        .iter()
                        .filter_map(|content| match content {
                            Content::Text(text) => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !answer.trim().is_empty()
                        && (buffer_read_only_review_text || !streamed_assistant_text)
                    {
                        ui.assistant_text(&answer);
                        ui.assistant_end();
                    }
                    self.messages
                        .push_assistant(std::mem::take(&mut completion.content));
                    progress_tracker.record_final_answer();
                    break false;
                }

                let calls: Vec<(String, String, String)> =
                    if request_text_answer || request_no_progress_final_answer {
                        Vec::new()
                    } else {
                        completion
                            .tool_calls()
                            .into_iter()
                            .map(|c| {
                                (
                                    c.id.to_string(),
                                    c.name.to_string(),
                                    c.arguments.to_string(),
                                )
                            })
                            .collect()
                    };

                // Fallback for local models (Ollama, llama.cpp, etc.) that emit
                // tool calls as text — raw JSON like {"name":"bash","arguments":…}
                // — instead of using the structured `tool_calls` API field. When
                // the API returned no structured calls, scan the assistant text
                // for tool-call JSON and promote any matches to real ToolCall
                // blocks so they actually execute. The raw JSON is stripped from
                // the recorded text so history stays clean.
                let calls = if calls.is_empty()
                    && !request_text_answer
                    && !request_no_progress_final_answer
                {
                    let full_text: String = completion
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            Content::Text(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let parsed =
                        parse_text_tool_calls(&full_text, textcall_id_offset(&self.messages));
                    if parsed.iter().any(|c| matches!(c, Content::ToolCall { .. })) {
                        // Replace text blocks with the interleaved content
                        // (prose segments + ToolCall blocks in emission order),
                        // preserving any Thinking blocks from the original.
                        let mut new_content = Vec::new();
                        let mut parsed_iter = parsed.into_iter().peekable();
                        for c in completion.content.iter() {
                            match c {
                                Content::Text(_) => {
                                    // Drain the parsed content that corresponds to
                                    // this text block (all of it — the original had
                                    // one Text block with the full raw text).
                                    for p in parsed_iter.by_ref() {
                                        new_content.push(p);
                                    }
                                }
                                Content::Thinking { .. } => new_content.push(c.clone()),
                                _ => {}
                            }
                        }
                        // If the original had no Text block (shouldn't happen for
                        // the local-model path, but be safe), drain remaining.
                        for p in parsed_iter {
                            new_content.push(p);
                        }
                        completion.content = new_content;
                        completion
                            .tool_calls()
                            .into_iter()
                            .map(|c| {
                                (
                                    c.id.to_string(),
                                    c.name.to_string(),
                                    c.arguments.to_string(),
                                )
                            })
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    calls
                };

                // Repetition guard: the model re-issued the exact same tool
                // calls (same names, same arguments, same order) as the previous
                // round. Re-running most tools can only reproduce the same
                // output, so don't execute — nudge the model to act on the output
                // it already has. `bash_output` is intentionally excluded from
                // this exact-match shortcut because a live background process is
                // time-dependent and can emit new output between identical polls;
                // completed/missing/pruned handles are caught below by the
                // stale-background no-new-evidence path. Bounded; past the
                // budget the turn ends with an honest "stuck repeating" notice
                // rather than looping until `max_steps`.
                let call_sig: Vec<(String, String)> = calls
                    .iter()
                    .map(|(_, name, args)| (name.clone(), args.clone()))
                    .collect();
                let has_background_output_poll = calls
                    .iter()
                    .any(|(_, name, _)| name.as_str() == "bash_output");
                let has_background_handle_call = calls
                    .iter()
                    .any(|(_, name, _)| matches!(name.as_str(), "bash_output" | "bash_kill"));
                let has_no_progress_bash = calls.iter().any(|(_, name, args)| {
                    name == "bash" && bash_no_progress_signature(args).is_some()
                });
                // A bash command that deliberately waits before sampling state
                // ("sleep 300 && du -sh models/") is time-dependent the same
                // way a `bash_output` poll is: re-running it verbatim is how
                // the model watches a slow external process (a download, a
                // long build, a warming server), and each run can return new
                // output. Exempt such rounds from the signature-based repeat
                // guards; the result-hash guard below still catches the
                // static case (the same poll returning byte-identical output),
                // so a wait loop stays bounded without punishing legitimate
                // progress-watching.
                let has_wait_poll_bash = calls
                    .iter()
                    .any(|(_, name, args)| name == "bash" && bash_call_waits(args));
                let exact_repeat = !calls.is_empty()
                    && !has_background_output_poll
                    && !has_wait_poll_bash
                    && prev_call_sig.as_ref() == Some(&call_sig);
                // No-new-evidence cycle guard: a round whose every call is a
                // read-only inspection (read/list/grep/glob) or stale background
                // handle operation already performed earlier this turn. This
                // catches multi-step cycles like
                // A→B→C→A→B→C — including grep/list cycles, not just re-reads —
                // that evade the exact-match check because each round differs
                // from the one right before it. On large workspaces such a cycle
                // can otherwise loop until `max_steps` without ever re-issuing an
                // identical round. `EvidenceTracker::round_adds_evidence` keys on
                // a stable per-inspection signature (read path/page, list path,
                // grep pattern/glob/path/context, stale background handle id), so
                // any re-inspection is caught regardless of cycle length or tool
                // mix. Shares the same
                // `repeat_nudges` budget as the exact-match guard so it stays
                // bounded.
                //
                // Fires only on the *second* consecutive no-new-evidence round
                // (`prev_added_no_evidence`): a single re-inspection right after
                // new evidence is allowed through (e.g. re-reading a file once a
                // broader search has surfaced something to re-examine, or paging
                // further into a file). Once the turn has made a successful
                // mutation, this guard is advisory only: after the nudge budget
                // is spent, execute the inspection rather than hard-stalling a
                // long implementation harness in the middle of a later plan step.
                let no_new_evidence = !calls.is_empty() && !evidence.round_adds_evidence(&calls);
                let stale_background_handle_call = no_new_evidence && has_background_handle_call;
                // A wait-poll round re-runs a seen inspection signature by
                // design, so it must not trip the no-new-evidence cycle guard
                // either — its staleness is judged by output, below.
                let is_repeat = exact_repeat
                    || (no_new_evidence
                        && !has_wait_poll_bash
                        && (prev_added_no_evidence || stale_background_handle_call));
                let no_new_after_mutation = is_repeat
                    && no_new_evidence
                    && implementation_tracker.mutation_seen
                    && !stale_background_handle_call;
                let repeat_budget_available = repeat_nudges < self.config.loop_limits.max_repeat_nudges;
                let should_skip_for_repeat =
                    is_repeat && (!no_new_after_mutation || repeat_budget_available);
                if should_skip_for_repeat {
                    // We deliberately do NOT execute the repeated tool calls,
                    // but the calls stay in the transcript, each paired with a
                    // synthetic result that says why it was skipped. Stripping
                    // them (as this path once did) left the model's turn as a
                    // bare placeholder with no result for the call it just
                    // made — weak models concluded the tool layer was broken
                    // ("my tool calls aren't producing visible output") and
                    // gave up instead of correcting course. Pairing every
                    // skipped `tool_use` with a `tool_result` also keeps the
                    // transcript in the shape providers require.
                    let all_plan_reposts = calls.iter().all(|(_, name, _)| name == "update_plan");
                    let all_bookkeeping_reposts = calls
                        .iter()
                        .all(|(_, name, _)| hi_tools::is_coordination(name));
                    let skip_results: Vec<(String, String)> = calls
                        .iter()
                        .map(|(id, name, _)| {
                            let note = if name == "update_plan" {
                                SKIPPED_PLAN_REPOST_RESULT
                            } else if hi_tools::is_coordination(name) {
                                SKIPPED_BOOKKEEPING_REPOST_RESULT
                            } else {
                                SKIPPED_REPEATED_CALL_RESULT
                            };
                            (id.clone(), note.to_string())
                        })
                        .collect();
                    self.messages.push_assistant_with_results(
                        std::mem::take(&mut completion.content),
                        skip_results,
                    );
                    if repeat_budget_available {
                        repeat_nudges += 1;
                        repeat_sampling_rounds += 1;
                        stalled_repeating = true;
                        let stall_reason = if all_plan_reposts {
                            "unchanged plan repost"
                        } else if all_bookkeeping_reposts {
                            "repeated bookkeeping call"
                        } else if stale_background_handle_call {
                            "stale background handle"
                        } else if has_no_progress_bash {
                            "semantic no-op bash command"
                        } else if no_new_evidence {
                            "repeated inspection signature"
                        } else {
                            "skipped repeated calls"
                        };
                        // Never force a chat-only "final answer" after bookkeeping
                        // loops on a mutation turn. That path exists for inspection
                        // stalls where the model already has evidence to summarize;
                        // on an edit request it just ends the turn incomplete with
                        // zero file changes (live: "I started the fix but didn't
                        // land the edit"). Keep tools required and let the
                        // budget-exhausted branch hand off to implementation repair.
                        let force_final_after_nudge = progress_tracker.record_no_progress_nudge(
                            stall_reason,
                            no_progress_signature_for_calls(&calls),
                        ) && !no_new_after_mutation
                            && implementation_intent.is_none()
                            && !(expected_mutation && all_bookkeeping_reposts);
                        let nudge = if all_bookkeeping_reposts {
                            if all_plan_reposts {
                                ui.nudge(&format!(
                                    "the model re-posted an unchanged plan — withholding \
                                     bookkeeping tools for a round and nudging it to execute \
                                     the next step ({repeat_nudges}/{})",
                                    self.config.loop_limits.max_repeat_nudges
                                ));
                            } else {
                                ui.nudge(&format!(
                                    "the model repeated bookkeeping calls without real work — \
                                     withholding bookkeeping tools for a round \
                                     ({repeat_nudges}/{})",
                                    self.config.loop_limits.max_repeat_nudges
                                ));
                            }
                            suppress_bookkeeping_tools_next = true;
                            force_tools_next = true;
                            // Cancel any prior force-final from a mixed stall so the
                            // bookkeeping withhold round still has real tools.
                            force_no_progress_final_answer_next = false;
                            if all_plan_reposts {
                                PLAN_REPOST_NUDGE.to_string()
                            } else {
                                BOOKKEEPING_REPOST_NUDGE.to_string()
                            }
                        } else if stale_background_handle_call {
                            if has_background_output_poll {
                                ui.nudge(&format!(
                                    "the model kept polling stale background process handles — \
                                     nudging it to stop polling them ({repeat_nudges}/{})",
                                    self.config.loop_limits.max_repeat_nudges
                                ));
                                "The background process handle you just polled is completed, missing, or pruned, so polling it again cannot produce new output. Do not call bash_output for that handle again. Continue from the available output, restart the command if you still need it, or finish with the current result.".to_string()
                            } else {
                                ui.nudge(&format!(
                                    "the model kept using stale background process handles — \
                                     nudging it to stop using them ({repeat_nudges}/{})",
                                    self.config.loop_limits.max_repeat_nudges
                                ));
                                "The background process handle you just used is already killed, already exited, missing, or pruned, so calling bash_kill for it again cannot change anything. Do not call bash_kill for that handle again. Continue from the available output, restart the command if you still need it, or finish with the current result.".to_string()
                            }
                        } else if should_nudge_read_after_repeated_search(
                            read_only_intent,
                            &evidence,
                        ) {
                            ui.nudge(&format!(
                                        "the model re-ran the same search — nudging it to read a matching file ({repeat_nudges}/{})",
                                        self.config.loop_limits.max_repeat_nudges
                                    ));
                            READ_AFTER_SEARCH_NUDGE.to_string()
                        } else if implementation_intent.is_some()
                            && no_new_evidence
                            && (evidence.saw_read || evidence.saw_search)
                        {
                            // Concrete, actionable nudge for implementation tasks:
                            // name the inspected files and the next plan step (if
                            // any) so the model has a specific action to take
                            // instead of a generic "start editing." A strong model
                            // responds to one concrete nudge; a weak one won't
                            // respond to any number, so the budget stays tight (2).
                            // Only fires for no-new-evidence cycles (re-reading
                            // already-inspected files); exact repeats of non-read
                            // tools (e.g. re-running a bash command) fall through
                            // to the generic REPEAT_NUDGE below, which says "don't
                            // re-run that command" — the right message for that case.
                            ui.nudge(&format!(
                                "the model re-read files it already inspected — their contents are \
                                 already above; nudging it to act on them ({repeat_nudges}/{})",
                                self.config.loop_limits.max_repeat_nudges
                            ));
                            let paths = inspected_paths_for_prompt(&evidence);
                            let plan_step = self
                                .last_plan
                                .iter()
                                .find(|s| {
                                    s.status == PlanStatus::Pending
                                        || s.status == PlanStatus::Active
                                })
                                .map(|s| s.title.as_str());
                            if let Some(step) = plan_step {
                                format!(
                                    "You already inspected these files: {paths}. Their contents are in the conversation above — do not re-read them. \
Your plan's next step is: \"{step}\". Execute it now with write/edit/multi_edit/apply_patch. \
Do not read more files first — you have enough context. Act on the next plan step immediately."
                                )
                            } else {
                                format!(
                                    "You already inspected these files: {paths}. Their contents are in the conversation above — do not re-read them. \
You have enough context to make progress. Edit one of the inspected files now with write/edit/multi_edit/apply_patch. \
If the task is already complete, stop and give your final recap."
                                )
                            }
                        } else if has_no_progress_bash {
                            ui.nudge(&format!(
                                "the model kept running no-op shell commands — nudging it to finish without more bash calls ({repeat_nudges}/{})",
                                self.config.loop_limits.max_repeat_nudges
                            ));
                            "The bash command you just called only says stop/quit/done or otherwise does no work. Do not call bash for that. If the task is complete, finish with a text answer; otherwise use a tool that inspects or changes the workspace.".to_string()
                        } else if no_new_evidence && !exact_repeat {
                            ui.nudge(&format!(
                                "the model re-read files it already inspected — their contents are \
                                 already above; nudging it to act on them ({repeat_nudges}/{})",
                                self.config.loop_limits.max_repeat_nudges
                            ));
                            REREAD_NUDGE.to_string()
                        } else {
                            ui.nudge(&format!(
                                "the model re-ran the same command — its output is already above; \
                                     nudging it to act on it ({repeat_nudges}/{})",
                                self.config.loop_limits.max_repeat_nudges
                            ));
                            REPEAT_NUDGE.to_string()
                        };
                        let nudge = if force_final_after_nudge {
                            force_no_progress_final_answer_next = true;
                            force_tools_next = false;
                            format!("{nudge}\n\n{NO_PROGRESS_FINAL_ANSWER_NUDGE}")
                        } else {
                            nudge
                        };
                        self.messages.push_nudge(NudgeKind::Repeat, nudge);
                        // Keep prev_call_sig as-is so a further repeat is still
                        // detected against the same signature.
                        continue;
                    }
                    if stale_background_handle_call {
                        ui.status(
                            "background process handles were completed, missing, or pruned (or already killed) and the model kept using them — the task may be incomplete. /retry, or send 'continue'.",
                        );
                        break false;
                    }
                    if has_no_progress_bash {
                        stalled_unfinished = true;
                        ui.nudge("model repeated no-op shell commands; stopping incomplete");
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if read_only_intent.is_some() && evidence.saw_search && !evidence.saw_read {
                        stalled_unfinished = true;
                        ui.nudge(
                            "review repeated the same search without reading files; stopping incomplete",
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if let Some(intent) = read_only_intent
                        && (evidence.saw_read || evidence.saw_search)
                    {
                        stalled_unfinished = true;
                        ui.nudge(
                            "review repeated the same command after inspection; stopping incomplete",
                        );
                        let _ = (intent, &evidence);
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    // Implementation / explicit-mutation turns that burned the
                    // repeat budget on non-mutating work must not hard-stop yet.
                    // Two live failure modes share this path:
                    //   1. re-reading already-inspected files without editing
                    //   2. pure bookkeeping loops (identical update_plan /
                    //      record_decision) that never even inspected the tree
                    // Case (2) used to fall through to the generic "kept
                    // re-running the same command" stop because the old gate
                    // required saw_read/saw_search. That branded turns as
                    // `incomplete · stalled` after two plan re-posts even when
                    // the model still had the implementation repair budget —
                    // exactly the "I started that fix but didn't land the
                    // edit" stall. Bookkeeping is zero-progress meta-work, not
                    // a dangerous inspection loop; hand it the same edit nudge.
                    let bookkeeping_only_stall = calls
                        .iter()
                        .all(|(_, name, _)| hi_tools::is_coordination(name));
                    let implementation_needs_mutation = !implementation_tracker.mutation_seen
                        && (implementation_intent.is_some() || expected_mutation)
                        && ((evidence.saw_read || evidence.saw_search) || bookkeeping_only_stall);
                    if implementation_needs_mutation {
                        if implementation_tracker.no_change_nudges < 2 {
                            implementation_tracker.no_change_nudges += 1;
                            evidence.quality_repair_nudges =
                                evidence.quality_repair_nudges.saturating_add(1);
                            let use_text_fallback = implementation_tracker.no_change_nudges >= 2;
                            force_tools_next = !use_text_fallback;
                            text_tool_fallback_next = use_text_fallback;
                            // Clear the sticky repeat stall: we are converting it
                            // into an implementation-repair continue, not ending
                            // the turn as stalled_repeating.
                            stalled_repeating = false;
                            // Drop the sticky prev signature so the next real
                            // tool call isn't immediately compared against the
                            // bookkeeping-only round that just exhausted the
                            // repeat budget.
                            prev_call_sig = None;
                            prev_added_no_evidence = false;
                            if bookkeeping_only_stall {
                                // Keep bookkeeping withheld while we demand real
                                // work — otherwise the model just re-posts the
                                // plan again on the repair round.
                                suppress_bookkeeping_tools_next = true;
                                ui.nudge(
                                    "implementation burned the bookkeeping-repeat budget without editing; nudging the model to edit or scaffold",
                                );
                            } else {
                                ui.nudge(
                                    "implementation kept repeating without editing; nudging the model to edit or scaffold",
                                );
                            }
                            let nudge = if use_text_fallback {
                                implementation_text_tool_nudge(IMPLEMENTATION_NO_CHANGES_NUDGE)
                            } else {
                                IMPLEMENTATION_NO_CHANGES_NUDGE.to_string()
                            };
                            self.messages.push_nudge(NudgeKind::Continue, nudge);
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.nudge(
                            "implementation kept repeating without editing; no file changes were made",
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    ui.status(
                        "⚠ the model kept re-running the same command without acting on the \
                         result — the task may be incomplete. /retry, or send 'continue'.",
                    );
                    break false;
                }
                // A different set of calls (or none) this round — the model moved
                // on, so clear any pending repeat-stall state. A wait-poll
                // round is not counted as the first wasted round of a cycle:
                // waiting on external state is progress-neutral, not evidence
                // of a loop.
                stalled_repeating = false;
                repeat_sampling_rounds = 0;
                prev_call_sig = Some(call_sig);
                prev_added_no_evidence = no_new_evidence && !has_wait_poll_bash;

                // Inspection-sprawl guard: a read-only review turn that keeps
                // reading *distinct* files (each a new inspection signature, so
                // the repeat/cycle guard above never fires) without ever
                // producing findings. Once enough evidence has accumulated,
                // nudge the model to answer; if it keeps sprawling past the
                // budget, stop incomplete rather than fabricate an answer. This is
                // the only guard that catches the "read 100 files, never
                // answer" failure mode — all review-quality guards fire only
                // on a final text answer, which never comes while the model
                // keeps issuing tool calls.
                if inspection_sprawl_exhausted(
                    inspection_sprawl_intent,
                    &evidence,
                    &calls,
                    read_only_inspection_cap,
                ) {
                    stalled_unfinished = true;
                    ui.nudge(
                            "review kept inspecting new files without producing findings; stopping incomplete",
                        );
                    ui.status(INCOMPLETE_STATUS);
                    break false;
                }
                if should_nudge_inspection_sprawl(
                    inspection_sprawl_intent,
                    &evidence,
                    &calls,
                    read_only_inspection_cap,
                ) {
                    evidence.inspection_sprawl_nudges =
                        evidence.inspection_sprawl_nudges.saturating_add(1);
                    force_text_answer_next = true;
                    let cap = read_only_inspection_cap
                        .unwrap_or_else(|| evidence.inspection_attempt_count());
                    ui.nudge(&format!(
                        "review inspected {} files/searches without answering; nudging it to produce findings",
                        evidence.inspection_attempt_count()
                    ));
                    self.messages
                        .push_assistant_text_only(std::mem::take(&mut completion.content));
                    self.messages.push_nudge(
                        NudgeKind::Continue,
                        inspection_sprawl_nudge(cap, evidence.inspection_attempt_count()),
                    );
                    continue;
                }

                // This round's assistant text, joined and captured before the
                // content is moved into history. Used both to detect a content-less
                // response (a reasoning model can return only reasoning tokens or
                // whitespace) and to spot an announced-but-unperformed next step.
                let assistant_text: String = completion
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_text = !assistant_text.trim().is_empty();

                if request_no_progress_final_answer {
                    let unusable = forced_final_answer_is_unusable(
                        &assistant_text,
                        plan_has_pending_steps(&self.last_plan),
                    );
                    if has_text && (buffer_read_only_review_text || !streamed_assistant_text) {
                        let text_to_emit = if buffered_assistant_text.is_empty() {
                            assistant_text.as_str()
                        } else {
                            buffered_assistant_text.as_str()
                        };
                        ui.assistant_text(text_to_emit);
                        ui.assistant_end();
                    }
                    if unusable {
                        self.messages
                            .push_assistant_text_only(std::mem::take(&mut completion.content));
                        stalled_unfinished = true;
                        progress_tracker.record(
                            ProgressKind::None,
                            "forced final-answer attempt was unusable",
                            None,
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    self.messages
                        .push_assistant(std::mem::take(&mut completion.content));
                    progress_tracker.record_final_answer();
                    break false;
                }

                // Auto-recover from a content-less response — no tool calls and no
                // text, i.e. a flaky provider returning only reasoning or an empty
                // message. Silently re-run a few times before giving up, each
                // retry resampling hotter (see the temperature bump above). The
                // dead round isn't recorded, so each retry re-runs with the
                // original context.
                if calls.is_empty() && !has_text {
                    if empty_retries < self.config.loop_limits.max_empty_retries {
                        empty_retries += 1;
                        if made_tool_call {
                            self.nudge_after_post_tool_empty_response(
                                &mut force_tools_next,
                                implementation_intent.is_some(),
                            );
                        }
                        ui.status(&format!(
                            "⚠ the model returned no response — retrying ({empty_retries}/{})",
                            self.config.loop_limits.max_empty_retries
                        ));
                        continue;
                    }
                    ui.status("⚠ the model returned no response after retrying — try /retry.");
                    break false;
                }
                // Real output this round — clear the retry counter so the
                // temperature bump is transient: a later, unrelated stall gets
                // its own budget rather than inheriting this one's elevation.
                empty_retries = 0;
                retry_state.protocol_retries = 0;
                truncation_retries = 0;

                if calls.is_empty() {
                    match self.steer_without_tools(
                        &assistant_text,
                        &mut completion.content,
                        read_only_intent,
                        implementation_intent,
                        &mut implementation_tracker,
                        &mut evidence,
                        &mut review_repair,
                        &mut progress_tracker,
                        &mut silent_continues,
                        &mut continue_total_nudges,
                        &mut force_tools_next,
                        &mut force_text_answer_next,
                        &mut text_tool_fallback_next,
                        &mut stalled_unfinished,
                        &mut buffered_assistant_text,
                        buffer_read_only_review_text,
                        steps,
                        ui,
                    ) {
                        super::steer::RoundControl::Continue => continue,
                        super::steer::RoundControl::BreakInner(hit) => break hit,
                    }
                }
                // The model requested tool calls — it's actively working.
                made_tool_call = true;
                // Real progress this round, so clear the silent-continue counter:
                // the budget bounds *consecutive* narrate-without-acting stalls,
                // not their total across the turn. A long, productive turn that
                // reads many files but occasionally narrates a step without the
                // tool call (a quirk of some models) recovers each time via the
                // nudge — without this reset the counter would creep up across
                // the whole turn and kill the turn mid-progress on the Nth stall
                // even though the model acted between every one. Mirrors the
                // `empty_retries = 0` reset above (a later stall gets its own
                // budget rather than inheriting an earlier one's).
                silent_continues = 0;
                // The model acted, so drop the forced-tool-choice we may have set
                // after a nudge — the next round is free to narrate or finish.
                force_tools_next = false;
                self.set_turn_phase(TurnPhase::Tools);
                let batch = self
                    .execute_tool_batch(
                        &calls,
                        &mut completion.content,
                        read_only_intent,
                        max_parallel_tools,
                        &task_contract,
                        &mut implementation_tracker,
                        &mut evidence,
                        &mut tool_guardrail,
                        &mut progress_tracker,
                        &mut tool_timeline,
                        &mut sched_tool_calls,
                        &mut sched_max_concurrent,
                        &mut sched_serial_runs,
                        &mut plan_updated_goal,
                        &mut proposed_goal,
                        &mut turn_snapshot,
                        &mut turn_checkpoint_allowed,
                        &mut turn_checkpoint_created,
                        &mut fast_feedback,
                        ui,
                    )
                    .await?;
                match self.steer_after_tools(
                    &calls,
                    &batch,
                    expected_mutation,
                    read_only_intent,
                    implementation_intent,
                    &mut implementation_tracker,
                    &mut evidence,
                    &mut mutation_recovery,
                    &mut progress_tracker,
                    &mut repeat_nudges,
                    &mut force_tools_next,
                    &mut text_tool_fallback_next,
                    &mut force_no_progress_final_answer_next,
                    &mut prev_added_no_evidence,
                    &mut stalled_repeating,
                    &mut stalled_unfinished,
                    ui,
                ) {
                    super::steer::RoundControl::Continue => {}
                    super::steer::RoundControl::BreakInner(hit) => break hit,
                }
            };

            if hit_cap {
                ui.status(&format!("reached step limit ({max_steps}); stopping turn"));
                ended_at_cap = true;
                break 'turn;
            }

            // TurnPhase::WorkspaceRepair — compile/lint/test stages; not review repair.
            // The state machine lives in WorkspaceRepairVerifier; this loop reacts.
            self.set_turn_phase(TurnPhase::WorkspaceRepair);
            let outcome = self
                .run_workspace_repair_verification(
                    &mut verifier,
                    &turn_background_baseline,
                    &mut turn_snapshot,
                    turn_checkpoint_created,
                    turn_ledger_revision,
                    &fast_feedback,
                    ui,
                )
                .await?;
            // Retain evidence immediately, not only in the common finalizer:
            // reconciliation or persistence can still fail after a successful
            // check, and reports for those error turns need the stages that
            // actually ran.
            self.last_turn_telemetry.verification_executions = verifier.executions().to_vec();
            match outcome {
                VerifyOutcome::NotRun => {
                    // Phase C obligation: one re-entry when a coding turn still
                    // owes green evidence (failed budget or never sealed).
                    let (changed_now, mutation_now) = {
                        let ledger = self.runtime.ledger();
                        (
                            ledger
                                .changes_since(turn_ledger_revision)
                                .into_iter()
                                .map(|c| c.path)
                                .collect::<Vec<_>>(),
                            ledger.had_mutation_since(turn_ledger_revision),
                        )
                    };
                    if !obligation_nudge_fired
                        && let Some(reason) = super::obligation::coding_verify_obligation(
                            self.last_task_contract.as_ref(),
                            &self.config.gates.verification,
                            expected_mutation,
                            &changed_now,
                            mutation_now,
                            self.last_verify,
                            verifier.executions().len(),
                        )
                    {
                        match reason {
                            // Never sealed green after a code mutation — one more
                            // model round to run checks / fix. Failed-verify budget
                            // exhaustion already spent its repair rounds above.
                            super::obligation::ObligationReason::UnverifiedMutation => {
                                obligation_nudge_fired = true;
                                ui.status(reason.ui_status());
                                ui.nudge(reason.ui_status());
                                self.messages
                                    .push_nudge(NudgeKind::Continue, reason.nudge_body());
                                force_tools_next = true;
                                continue 'turn;
                            }
                            super::obligation::ObligationReason::FailedVerify => {
                                stalled_unfinished = true;
                                ui.status(reason.ui_status());
                            }
                        }
                    }
                    if self.last_verify == Some(false) {
                        stalled_unfinished = true;
                        ui.status(
                            "verification still failed after the retry budget; the task may be incomplete. /retry, or send 'continue'.",
                        );
                    }
                    break 'turn;
                }
                VerifyOutcome::SkippedNoChanges { first } => {
                    if first {
                        ui.status("verification skipped — no files changed this turn");
                    }
                    // Mutation-shaped coding turns that somehow report no file
                    // delta still owe evidence when mutation_seen (e.g. restored
                    // bytes) or the contract expected edits — one obligation nudge.
                    let mutation_now = self
                        .runtime
                        .ledger()
                        .had_mutation_since(turn_ledger_revision);
                    if !obligation_nudge_fired
                        && let Some(reason) = super::obligation::coding_verify_obligation(
                            self.last_task_contract.as_ref(),
                            &self.config.gates.verification,
                            expected_mutation,
                            &[],
                            mutation_now,
                            self.last_verify,
                            verifier.executions().len(),
                        )
                    {
                        if matches!(
                            reason,
                            super::obligation::ObligationReason::UnverifiedMutation
                        ) {
                            obligation_nudge_fired = true;
                            ui.status(reason.ui_status());
                            ui.nudge(reason.ui_status());
                            self.messages
                                .push_nudge(NudgeKind::Continue, reason.nudge_body());
                            force_tools_next = true;
                            continue 'turn;
                        }
                    }
                    break 'turn;
                }
                VerifyOutcome::SkippedProseOnly { first } => {
                    if first {
                        ui.status("verification skipped — prose-only files changed this turn");
                    }
                    break 'turn;
                }
                VerifyOutcome::Passed => {
                    ui.status("✓ verification passed");
                    self.last_verify = Some(true);
                    self.reconcile_workspace_changes()?;
                    let (verified_revision, verified_digest, current_changes) = {
                        let mut ledger = self.runtime.ledger();
                        (
                            ledger.revision(),
                            ledger.workspace_revision(),
                            ledger.changes_since(turn_ledger_revision),
                        )
                    };
                    verified_at = Some((verified_revision, verified_digest.clone()));
                    let current_files = current_changes
                        .iter()
                        .map(|change| change.path.clone())
                        .collect::<Vec<_>>();
                    let mut diff = self.turn_diff().await;
                    let diff_lines = if diff.trim().is_empty() {
                        fallback_review_line_count(self.runtime.root(), &current_changes)
                    } else {
                        diff.lines().count()
                    };
                    let (review_required, large_diff_review) =
                        self.last_task_contract.as_ref().map_or((false, false), |contract| {
                            let required = contract.requires_review(
                                self.config.gates.review,
                                &current_files,
                                diff_lines,
                                self.config.subagents.long_horizon
                                    || self.config.subagents.write_subagents.is_enabled(),
                            );
                            let large = contract.is_large_mutation(&current_files, diff_lines);
                            (required, large)
                        });
                    if review_required {
                        self.refresh_active_task_context(
                            &context_task,
                            repository_context_enabled,
                            turn_ledger_revision,
                            &mut ranked_context_paths,
                            &mut context_generation_seen,
                            &mut indexed_ledger_revision,
                        );
                        if diff.chars().count() > 50_000 {
                            diff = diff.chars().take(50_000).collect();
                            diff.push_str("\n… (bounded review diff truncated)");
                        }
                        let contract = self
                            .last_task_contract
                            .as_ref()
                            .and_then(|contract| serde_json::to_string_pretty(contract).ok())
                            .unwrap_or_else(|| "(task contract unavailable)".into());
                        let instructions = self.task_context.as_deref().unwrap_or("(none)");
                        let stages = verifier.stages_summary().unwrap_or_else(|| "(none)".into());
                        let context = format!(
                            "Task contract:\n{contract}\n\nScoped instructions and relevant repository context:\n{instructions}\n\nChanged files ({file_count}):\n{files}\n\nDiff size: {diff_lines} lines\nDeterministic verification: PASSED\nStages: {stages}\nVerified workspace revision: {verified_digest}\n\nComplete bounded turn diff:\n{diff}",
                            file_count = current_files.len(),
                            files = current_files.join("\n"),
                        );
                        // Phase L: large multi-file diffs get the hole-focused
                        // skeptic prompt; other risk reviews keep the general one.
                        let review_label = if large_diff_review {
                            "large-diff skeptic"
                        } else {
                            "independent completion review"
                        };
                        ui.status(&format!("running {review_label}"));
                        let verdict = if diff.trim().is_empty() && !current_files.is_empty() {
                            super::super::skeptic::SkepticVerdict::Unavailable(
                                "a complete turn diff was unavailable for the current changes"
                                    .into(),
                            )
                        } else if large_diff_review {
                            self.large_diff_review(&context).await
                        } else {
                            self.independent_review(&context).await
                        };
                        match verdict {
                            super::super::skeptic::SkepticVerdict::Approve => {
                                independent_review_status = ReviewStatus::Passed;
                                if large_diff_review {
                                    ui.status("✓ large-diff skeptic approved");
                                }
                            }
                            super::super::skeptic::SkepticVerdict::Unavailable(reason) => {
                                independent_review_status = ReviewStatus::Unavailable;
                                ui.status(&format!(
                                    "{review_label} unavailable after deterministic pass: {reason}"
                                ));
                            }
                            super::super::skeptic::SkepticVerdict::Object(objections)
                                if independent_review_repairs == 0 =>
                            {
                                independent_review_repairs = 1;
                                independent_review_status = ReviewStatus::Objected;
                                self.last_verify = None;
                                verified_at = None;
                                verifier.allow_review_revalidation();
                                let headline = if large_diff_review {
                                    "Large-diff skeptic found concrete multi-file defects"
                                } else {
                                    "Independent review found concrete completion defects"
                                };
                                self.messages.push_nudge(
                                    NudgeKind::Review,
                                    format!(
                                        "{headline}. Repair them now, then re-run deterministic validation.\n\n{}",
                                        objections
                                            .iter()
                                            .map(|objection| format!("- {objection}"))
                                            .collect::<Vec<_>>()
                                            .join("\n")
                                    ),
                                );
                                ui.nudge(&format!(
                                    "{review_label} objected; allowing one repair cycle"
                                ));
                                continue 'turn;
                            }
                            super::super::skeptic::SkepticVerdict::Object(objections) => {
                                independent_review_status = ReviewStatus::Objected;
                                stalled_unfinished = true;
                                ui.status(&format!(
                                    "{review_label} objected again after repair: {}",
                                    objections.join("; ")
                                ));
                            }
                            // The independent-review prompt defines no ESCALATE
                            // verdict; treat a stray one as a final objection
                            // (no extra repair cycle — escalation means
                            // retrying can't fix it).
                            super::super::skeptic::SkepticVerdict::Escalate(objections) => {
                                independent_review_status = ReviewStatus::Objected;
                                stalled_unfinished = true;
                                ui.status(&format!(
                                    "{review_label} escalated — needs your judgment: {}",
                                    objections.join("; ")
                                ));
                            }
                        }
                    }
                    break 'turn;
                }
                VerifyOutcome::Failed {
                    stage,
                    output,
                    round,
                } => {
                    ui.status(&format!("✗ {} failed; iterating", stage.name));
                    self.last_verify = Some(false);
                    verified_at = None;
                    let guidance = stage_guidance(&stage);
                    // Structured failure: attributions + condensed output + optional
                    // diagnostic snippet. Enrich-only relative to the raw blob.
                    let structured = hi_tools::format_structured_failure(
                        &format!(
                            "Verification stage `{}` failed (`{}`).",
                            stage.name, stage.command
                        ),
                        &output,
                        Some(guidance),
                    );
                    last_verify_attributions = structured.attributions.clone();
                    // Replace the previous verify nudge instead of accumulating.
                    // Only the latest verification output belongs in context.
                    // `replace_last_nudge` pops trailing tool/assistant messages
                    // from the prior verify cycle and the prior nudge itself
                    // (located by typed kind, not string-matching), then pushes
                    // the new one. On the first round there's no prior nudge, so
                    // nothing is popped — the model's just-finished turn stays.
                    self.messages
                        .replace_last_nudge(NudgeKind::Verify { round }, structured.body);
                    // Re-enter Model → Tools with the verify nudge in context.
                    // The verifier's round counter enforces max_verify_repairs.
                    continue 'turn;
                }
                VerifyOutcome::InfrastructureError {
                    stage,
                    output,
                    round,
                } => {
                    verification_infrastructure_error = true;
                    self.last_verify = None;
                    verified_at = None;
                    ui.status(&format!(
                        "verification infrastructure failed at {} (round {round}): {output}",
                        stage.name,
                    ));
                    break 'turn;
                }
                VerifyOutcome::Unstable {
                    stage,
                    changed_files,
                    round,
                } => {
                    verification_unstable = true;
                    stalled_unfinished = true;
                    self.last_verify = Some(false);
                    verified_at = None;
                    ui.status(&format!(
                        "verification is unstable in round {round}: stage {} modified {}",
                        stage.name,
                        changed_files.join(", ")
                    ));
                    break 'turn;
                }
            }
        }

        // TurnPhase::Settle — seal checkpoint, then keep/wipe green verify.
        self.set_turn_phase(TurnPhase::Settle);
        // Seal first: checkpoint creation may take long enough for an owned
        // process or editor to move the tree. The authoritative reconciliation
        // below therefore happens after this final asynchronous safety step.
        if turn_checkpoint_created && !self.seal_turn_checkpoint(ui).await? {
            turn_checkpoint_created = false;
            // Default YOLO permits checkpoint-free mutation. A seal failure
            // must be silent and non-terminal there; strict confirmation mode
            // still treats loss of its promised undo record as incomplete.
            stalled_unfinished |= !self.config.gates.allow_no_checkpoint;
        }
        // The ledger is the authoritative source for exact effects, including
        // shell/delegate/background changes that did not flow through a file
        // mutation tool. Its revision is content-based and workspace-local.
        self.reconcile_workspace_changes()?;
        let (final_ledger_revision, final_workspace_revision, ledger_changes) = {
            let mut ledger = self.runtime.ledger();
            (
                ledger.revision(),
                ledger.workspace_revision(),
                ledger.changes_since(turn_ledger_revision),
            )
        };
        {
            let delta = {
                let ledger = self.runtime.ledger();
                match verified_at.as_ref() {
                    Some((revision, _)) => ledger.changes_since(*revision),
                    None => ledger_changes.clone(),
                }
            };
            super::settlement::reconcile_verified_revision(
                &mut self.last_verify,
                &mut verified_at,
                &mut independent_review_status,
                final_ledger_revision,
                final_workspace_revision.clone(),
                &delta,
                ui,
            );
        }
        self.last_changed_files = ledger_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        self.last_file_changes = ledger_changes;
        self.last_compat_fallbacks = compat_fallbacks;
        // Flush the per-turn counters (otherwise discarded locals) into
        // telemetry so `--report` / the eval harness can diagnose the turn's
        // trajectory: how many verify rounds, recovery retries, nudges fired,
        // and where the last verify failure pointed.
        self.last_turn_telemetry = build_turn_telemetry(
            max_steps,
            verifier.round(),
            empty_retries,
            repeat_nudges,
            continue_total_nudges,
            truncation_total_retries,
            &progress_tracker,
            ended_at_cap,
            stalled_unfinished,
            stalled_repeating,
            &last_verify_attributions,
            verifier.executions(),
            sched_tool_calls,
            sched_max_concurrent,
            sched_serial_runs,
            &tool_timeline,
            &evidence,
            &review_repair,
        );
        self.last_turn_telemetry.checkpoint_available =
            turn_checkpoint_allowed.map(|_| turn_checkpoint_created);
        self.last_turn_telemetry.advertised_tools = advertised_tool_names.into_iter().collect();
        self.last_turn_telemetry.tool_schema_tokens = tool_schema_tokens;

        // Verifier-gated skill auto-curation: after a turn that PASSED verification
        // and actually changed files, optionally distill a reusable technique into a
        // learned skill. The ground-truth verifier is the gate (safe with weak local
        // models); opt-in via `curate_skills`, and capped per session.
        if self.config.memory.curate_skills
            && self.last_verify == Some(true)
            && !self.last_changed_files.is_empty()
            && self.auto_skills_written < super::super::MAX_AUTO_SKILLS_PER_SESSION
        {
            self.curate_turn_end(turn_start, ui).await;
        }

        // Phase K: always-on (cheap, no model call) coding-fact extraction into
        // the decision log + project memory after a green file-changing turn.
        if self.last_verify == Some(true) && !self.last_changed_files.is_empty() {
            self.record_coding_facts_turn_end(ui);
        }

        // Surface the files this turn changed, so the user sees what was touched
        // without needing /diff. Skipped for read-only/Q&A turns (empty list).
        // Emitted BEFORE the finalize recap so the recap is the last text the
        // user sees (the "✓ done" marker follows it).
        if !self.last_changed_files.is_empty() {
            ui.changed_files(&self.last_changed_files);
        }

        // TurnPhase::Finalize — optional tool-free recap after mutating turns.
        // Requiring `made_tool_call` keeps plain Q&A from triggering it. Skipped
        // on step cap / stall (work may be incomplete).
        self.set_turn_phase(TurnPhase::Finalize);
        if self.config.memory.finalize
            && made_tool_call
            && !ended_at_cap
            && !stalled_unfinished
            && !stalled_repeating
            && !self.last_changed_files.is_empty()
            && steps < max_steps
        {
            self.finalize_turn(turn_start, ui).await;
            // finalize_turn appended a [user: finalize-nudge][assistant: recap]
            // pair. Strip it from the persisted transcript so the FINALIZE_PROMPT
            // ("don't take any further action") doesn't bleed into the next turn
            // and make the model emit summary text instead of executing the new
            // prompt. The recap was already shown to the user via the UI.
            self.messages.strip_finalize_pair();
        }

        // Tool-free curation/finalization calls and external editors can take
        // time after the first final reconciliation. Reconcile once more before
        // any long-horizon progress or typed outcome is committed.
        self.reconcile_workspace_changes()?;
        let (settled_revision, settled_digest, settled_changes) = {
            let mut ledger = self.runtime.ledger();
            (
                ledger.revision(),
                ledger.workspace_revision(),
                ledger.changes_since(turn_ledger_revision),
            )
        };
        {
            let delta = {
                let ledger = self.runtime.ledger();
                match verified_at.as_ref() {
                    Some((revision, _)) => ledger.changes_since(*revision),
                    None => settled_changes.clone(),
                }
            };
            super::settlement::reconcile_verified_revision(
                &mut self.last_verify,
                &mut verified_at,
                &mut independent_review_status,
                settled_revision,
                settled_digest.clone(),
                &delta,
                ui,
            );
        }
        self.last_changed_files = settled_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        self.last_file_changes = settled_changes;

        // Long-horizon progress happens only after the final settled revision
        // still matches deterministic verification.
        // Keep the pre-turn goal until every user/session callback has
        // finished. A late workspace mutation must also roll back progress
        // that this hook tentatively advances.
        let goal_before_final_settlement = goal_before.clone();
        let goal_invalidated_verification = self
            .goal_turn_end(
                super::super::goal_turn::GoalTurnState {
                    stalled_unfinished,
                    stalled_repeating,
                    hit_step_cap: ended_at_cap,
                    plan_updated_goal,
                    proposed_goal,
                    goal_before,
                    verified_at: verified_at.as_ref(),
                    turn_ledger_revision,
                },
                ui,
            )
            .await;
        if goal_invalidated_verification {
            verified_at = None;
            if independent_review_status == ReviewStatus::Passed {
                independent_review_status = ReviewStatus::Unavailable;
            }
        }

        // Report the user-prompt estimate and all turn-local model output; full request
        // context remains visible as the `ctx` gauge below.
        ui.turn_end(&self.usage_summary(&self.totals));
        // Strip any trailing synthetic nudge so it doesn't absorb the next
        // real prompt via `push_user_or_fold` (which folds a new user message
        // into a trailing user message). A stall (repeat-nudge, continue-
        // nudge, verify-fail, truncation) can leave a nudge as the last
        // entry; removing it here gives the next turn a clean transcript.
        self.messages.strip_trailing_nudges();
        self.persist()?;

        // `goal_turn_end`, `Ui::turn_end`, and a session sink are extension
        // points outside the verifier. Reconcile after all of them and before
        // constructing the typed outcome so none can create a false current-
        // revision pass. There are deliberately no callbacks after this
        // settlement point.
        self.reconcile_workspace_changes()?;
        let (outcome_revision, outcome_digest) = {
            let mut ledger = self.runtime.ledger();
            (ledger.revision(), ledger.workspace_revision())
        };
        let changed_after_final_hooks = self.last_verify == Some(true)
            && verified_at.as_ref().is_none_or(|(revision, digest)| {
                *revision != outcome_revision || digest != &outcome_digest
            });
        if changed_after_final_hooks {
            let delta = {
                let ledger = self.runtime.ledger();
                match verified_at.as_ref() {
                    Some((revision, _)) => ledger.changes_since(*revision),
                    None => ledger.changes_since(turn_ledger_revision),
                }
            };
            let wiped = super::settlement::reconcile_verified_revision_with_message(
                &mut self.last_verify,
                &mut verified_at,
                &mut independent_review_status,
                outcome_revision,
                outcome_digest.clone(),
                &delta,
                ui,
                "workspace changed during turn finalization; the previous pass and goal progress were invalidated",
            );
            if wiped {
                if self.config.subagents.long_horizon
                    && let Some(previous) = goal_before_final_settlement
                {
                    self.structured_goal = Some(previous);
                    self.refresh_system_message();
                    // The earlier persist may contain tentatively advanced goal
                    // state. Rewrite the goal record itself (message persistence
                    // does not include side-channel goal state) before returning.
                    if let Some(session) = self.session.as_mut()
                        && let Some(goal) = self.structured_goal.as_ref()
                    {
                        session.record_goal(goal)?;
                    }
                }
                // Capture any additional effects of the invalidation notification
                // or corrective persistence. No UI/session callback follows this.
                self.reconcile_workspace_changes()?;
            }
        }
        let (final_changes, turn_had_mutation) = {
            let ledger = self.runtime.ledger();
            (
                ledger.changes_since(turn_ledger_revision),
                ledger.had_mutation_since(turn_ledger_revision),
            )
        };
        self.last_changed_files = final_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        self.last_file_changes = final_changes;

        // `Unverified` is reserved for "checks should have run but did not
        // settle" (budget exhausted after a fail, post-pass code mutation, etc.).
        // When the pipeline never ran a stage — disabled, no auto markers, prose
        // only, empty effective stages — the honest public state is
        // `NotApplicable` ("no applicable checks"), not a scary incomplete
        // "unverified changes" warning. Users still get `Unverified` when a
        // check was expected and missing.
        let no_check_executed = self.last_turn_telemetry.verification_executions.is_empty();
        let (status, verification, review, stop_reason) = super::finalize::classify_turn_outcome(
            verification_infrastructure_error,
            verification_unstable,
            self.last_verify,
            &self.last_changed_files,
            turn_had_mutation,
            no_check_executed,
            independent_review_status,
            self.last_turn_telemetry.skeptic_last_status,
            ended_at_cap,
            stalled_unfinished,
            stalled_repeating,
            expected_mutation,
            self.config.gates.allow_unverified,
        );
        // Outer `run_turn` also stamps Done (covers `?` paths); keep the success path explicit.
        self.set_turn_phase(TurnPhase::Done);
        let outcome = TurnOutcome {
            status,
            verification,
            review,
            stop_reason,
            changed_files: self.last_changed_files.clone(),
            verified_workspace_revision: (verification == VerificationStatus::Passed)
                .then(|| verified_at.as_ref().map(|(_, digest)| digest.clone()))
                .flatten(),
            effective_route: effective_model_route(
                &self.config,
                effective_fallback_route.as_deref(),
            ),
        };
        self.last_effective_route = outcome.effective_route.clone();
        self.last_turn_outcome = Some(outcome.clone());
        self.active_turn_ledger_revision = None;
        self.active_turn_message_start = None;
        Ok(outcome)
    }
}
