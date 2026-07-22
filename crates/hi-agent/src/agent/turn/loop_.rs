//! The main turn loop: user message → model → tools → steer → workspace repair.
//!
//! Model I/O lives in [`super::model_round`]; Tools and Steer are delegated to
//! [`super::tools`] and [`super::steer`]. Pipeline phases are named in
//! [`super::phase::TurnPhase`]:
//! `Setup → (Model → Tools → Steer)* → WorkspaceRepair → Settle → Finalize → Done`.
//!
//! Two repair systems (do not conflate):
//! - **Workspace repair** — [`crate::verify::WorkspaceRepairVerifier`] (tests/build)
//! - **Review repair** — [`crate::steering::ReviewRepairMode`] during Steer

use std::collections::BTreeSet;

use anyhow::Result;
use hi_ai::{ToolMode, estimate_text_tokens};

use crate::command;
use crate::compaction;
use crate::domain::TurnControlFlags;
use crate::heuristics::{looks_like_continue, tool_mode_label};
use crate::steering::{
    EvidenceTracker, IMPLEMENTATION_EMPTY_TUI_NUDGE, ImplementationIntent, ImplementationTracker,
    MutationRecovery, ReviewIntent, ToolLoopGuardrail, classify_implementation_intent,
    classify_read_only_intent, implementation_mentions_tui, implementation_turn_prompt,
    read_only_turn_prompt, scaled_inspection_cap, workspace_source_file_count,
};
use crate::transcript::NudgeKind;
use crate::verify::{Snapshot, WorkspaceRepairVerifier};
use crate::{
    AUTO_KEEP_RECENT, ReviewStatus, TaskContract, TaskIntent, ToolCallEntry, TurnOutcome,
    TurnStatus, TurnStopReason, TurnTelemetry, Ui, VerificationMode, VerificationStatus,
};

use super::helpers::{
    build_turn_telemetry, effective_max_steps_for_turn, effective_model_route,
    task_needs_repository_context,
};
use super::phase::TurnPhase;
use super::progress::ProgressTracker;
use super::retry::{ReviewRepairState, TurnRetryState};

impl crate::Agent {
    /// Run one user turn to completion, emitting output through `ui`.
    ///
    /// Phases: [`TurnPhase::Setup`] → model/tool/steer loop →
    /// [`TurnPhase::WorkspaceRepair`] (optional stages; failures re-enter the
    /// model up to one initial check plus `max_verify_repairs` cycles) →
    /// [`TurnPhase::Settle`] → optional [`TurnPhase::Finalize`] →
    /// [`TurnPhase::Done`].
    pub async fn run_turn(&mut self, input: &str, ui: &mut dyn Ui) -> Result<TurnOutcome> {
        // User lifecycle hooks are intentionally outside the model/tool loop.
        // `pre-turn` is a gate; `post-turn` and `stop` are best-effort notices.
        let hooks = self.workspace_root().join(".hi/hooks");
        let hooks_trusted = crate::workspace_trusted(self.workspace_root());
        if hooks.join("pre-turn").is_file() && hooks_trusted {
            let report = crate::run_hook(self.workspace_root(), "pre-turn", input)
                .map_err(|e| anyhow::anyhow!("pre-turn hook blocked turn: {e:#}"))?;
            ui.status(&report);
        } else if hooks.join("pre-turn").is_file() {
            ui.status("project hooks skipped: workspace untrusted (run /trust on to enable)");
        }
        // Always land on Done, including `?` error exits mid-turn.
        // Phase stamps inside the body are validated by TurnPhase::can_transition_to.
        let result = self.run_turn_body(input, ui).await;
        self.set_turn_phase(TurnPhase::Done);
        let summary = match &result {
            Ok(outcome) => format!("status=ok\noutcome={outcome:?}\ninput={input}"),
            Err(error) => format!("status=error\nerror={error:#}\ninput={input}"),
        };
        if hooks.join("post-turn").is_file() && hooks_trusted {
            match crate::run_hook(self.workspace_root(), "post-turn", &summary) {
                Ok(report) => ui.status(&report),
                Err(error) => ui.status(&format!("post-turn hook failed: {error:#}")),
            }
        }
        if hooks.join("stop").is_file() && hooks_trusted {
            match crate::run_hook(self.workspace_root(), "stop", &summary) {
                Ok(report) => ui.status(&report),
                Err(error) => ui.status(&format!("stop hook failed: {error:#}")),
            }
        }
        result
    }

    async fn run_turn_body(&mut self, input: &str, ui: &mut dyn Ui) -> Result<TurnOutcome> {
        // Phase stamp for the emerging state machine (see `phase.rs`).
        self.set_turn_phase(TurnPhase::Setup);
        // A leftover `/btw` answer-pending flag (e.g. the model answered a side
        // question with tool calls only, or the prior turn was cancelled) must
        // not route this turn's first assistant text to `btw_answer`.
        self.btw_answer_pending = false;
        let user_prompt_tokens = estimate_text_tokens(input);
        // Reset the per-turn file-read cache. It's invalidated per-key by the
        // edit tools and wholesale after `bash`, but clearing it here restores
        // its documented per-turn contract — so a file changed outside `hi`
        // between turns is re-read fresh, not served from a prior turn's cache.
        self.runtime.clear_read_cache();
        // Reconcile user/external edits before establishing this turn's
        // baseline so they are not attributed to the agent.
        self.runtime.ledger().reconcile()?;
        let turn_ledger_revision = self.runtime.ledger().revision();
        let turn_background_baseline = self.runtime.background().ids();
        // Ledger + bg baselines + per-turn caches (cancel-safe finalizers).
        self.workspace
            .begin_turn(turn_ledger_revision, turn_background_baseline.clone());
        let expanded_input =
            command::expand_prompt_macro(input).unwrap_or_else(|| input.to_string());
        // Synthetic goal-drive text is only transport. Contracts, context
        // ranking, review, and implementation guards need the real objective
        // and active milestone—especially explicit paths such as plan.md.
        let goal_context = self.goal_continuation_context(&expanded_input);
        let goal_drive_turn = goal_context.is_some();
        let context_task = goal_context.unwrap_or_else(|| expanded_input.clone());
        let structurally_read_only_subagent = self.config.subagents.is_subagent
            && self.config.routing.tool_mode == ToolMode::ReadOnly;
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
            .workspace
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
        self.task.set_task_context(
            repository_context_enabled
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
                .flatten(),
        );
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
        self.task
            .set_task(Some(context_task.clone()), Some(task_contract.clone()));
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
        let indexed_file_count = workspace_source_file_count(self.runtime.root());
        let read_only_inspection_cap = inspection_sprawl_intent
            .map(|intent| scaled_inspection_cap(&context_task, intent, indexed_file_count));
        let turn_input = if let Some(intent) = read_only_intent {
            read_only_turn_prompt(&context_task, intent)
        } else if let Some(intent) = implementation_intent {
            implementation_turn_prompt(&context_task, intent)
        } else {
            context_task.clone()
        };
        let input = turn_input.as_str();
        let model_turn_input = match self.rsi_observe.take_managed_context() {
            Some(context) if !context.is_empty() => format!(
                "{turn_input}\n\nManaged RSI prior conversation context (reference only; it does not change the current task's mutation requirements):\n{context}"
            ),
            _ => turn_input.clone(),
        };
        self.reset_last_turn_usage(user_prompt_tokens);
        self.report.last_turn_outcome = None;
        self.report.last_effective_route = effective_model_route(&self.config, None);

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
            self.report.last_verify = None;
            self.workspace.last_changed_files.clear();
            self.workspace.last_file_changes.clear();
            self.report.last_compat_fallbacks.clear();
            self.report.last_turn_telemetry = TurnTelemetry::default();
            let preserve_plan = (goal_drive_turn || looks_like_continue(&context_task))
                && self.goals.plan_incomplete();
            if self.goals.clear_plan_unless(preserve_plan) {
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
            self.report.last_turn_outcome = Some(outcome.clone());
            self.workspace.clear_active_baselines();
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
            && self.report.context_used * 100
                >= u64::from(window) * self.config.memory.auto_compact_percent
        {
            ui.status(&format!(
                "context ~{}% full — compacting to free room",
                self.report.context_used * 100 / u64::from(window)
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
            self.report.context_used = 0;
        }

        self.messages.strip_trailing_nudges();
        self.persisted = self.persisted.min(self.messages.len());
        let mut turn_start = self.messages.len();
        self.workspace.set_message_start(turn_start);
        self.messages.push_user_or_fold(&model_turn_input);
        self.report.set_verify(None);
        self.workspace.last_changed_files.clear();
        self.workspace.last_file_changes.clear();
        self.report.last_compat_fallbacks.clear();
        self.report
            .last_turn_telemetry
            .verification_executions
            .clear();
        // Preserve only an unfinished plan that the user explicitly continues.
        // Clearing must also be emitted: the TUI owns a pinned copy and cannot
        // infer that the agent cleared its internal state.
        let preserve_plan =
            (goal_drive_turn || looks_like_continue(&context_task)) && self.goals.plan_incomplete();
        if self.goals.clear_plan_unless(preserve_plan) {
            if let Some(session) = self.session.as_mut() {
                session.clear_plan()?;
            }
            ui.plan(&[]);
        }
        let mut compat_fallbacks = Vec::new();
        let mut effective_fallback_route: Option<String> = None;

        let resolved_verify_stages = self
            .config
            .gates
            .verification
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
        // Per-turn control flags (force-next-tool, stalls, caps, obligation).
        // See [`TurnControlFlags`] — field projection keeps call sites direct.
        let mut flags = TurnControlFlags::default();
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
        let goal_before = self.goals.clone_structured();
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
        let mut implementation_tracker = ImplementationTracker::default();
        let mut empty_tui_needs_project = false;
        if let Some(intent) = read_only_intent
            && self.config.gates.read_only_preflight
            && !self
                .config
                .rsi
                .remote_switch
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
                    self.config
                        .loop_limits
                        .max_tool_calls
                        .saturating_sub(sched_tool_calls),
                )
                .await;
            if preflight.executed > 0 {
                flags.made_tool_call = true;
                sched_tool_calls = sched_tool_calls.saturating_add(preflight.executed);
                sched_serial_runs = sched_serial_runs.saturating_add(preflight.serial_runs);
                sched_max_concurrent = sched_max_concurrent.max(preflight.max_concurrent_batch);
            }
        }
        if implementation_intent.is_some()
            && !self
                .config
                .rsi
                .remote_switch
                .as_ref()
                .is_some_and(|enabled| enabled.load(std::sync::atomic::Ordering::SeqCst))
            && !matches!(self.config.routing.tool_mode, ToolMode::ChatOnly)
            && sched_tool_calls < self.config.loop_limits.max_tool_calls
        {
            let preflight_calls = self
                .run_implementation_preflight(ui, &mut implementation_tracker, &mut tool_timeline)
                .await;
            if preflight_calls > 0 {
                flags.made_tool_call = true;
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
        // After a bookkeeping-repost nudge, withhold the bookkeeping tools
        // (`update_plan`, `record_decision`) from the next request's tool
        // list. A bookkeeping-fixated model (observed live) keeps re-posting
        // meta-work through every nudge — and when only `update_plan` was
        // withheld it slid to repeating `record_decision` instead. Clear
        // feedback alone doesn't break the loop; removing the whole family
        // for one round forces a tool that does real work.
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
        // Whether the turn ended without enough evidence for a read-only review.
        // One-shot coding verify-obligation re-entry (Phase C). Prevents a
        // mutation-shaped turn from settling as "done" without green evidence
        // when a pipeline is configured — fires at most once per turn.
        // Whether the turn was cut short by the per-turn step cap, so the
        // finalization recap is skipped (the work may be incomplete).
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

        // Owned per-turn bag — Model/Tools/Steer/Verify project from this.
        let mut turn = super::state::TurnState {
            user_prompt_tokens,
            turn_ledger_revision,
            turn_background_baseline: turn_background_baseline.clone(),
            context_task: context_task.clone(),
            goal_drive_turn,
            task_contract: task_contract.clone(),
            repository_context_enabled,
            ranked_context_paths,
            context_generation_seen,
            indexed_ledger_revision,
            read_only_intent,
            implementation_intent,
            expected_mutation,
            inspection_sprawl_intent,
            read_only_inspection_cap,
            turn_input: input.to_string(),
            turn_checkpoint_allowed,
            turn_checkpoint_created,
            verifier,
            fast_feedback,
            max_steps,
            max_parallel_tools,
            steps,
            empty_retries,
            truncation_retries,
            truncation_total_retries,
            silent_continues,
            continue_total_nudges,
            repeat_nudges,
            repeat_sampling_rounds,
            flags,
            mutation_recovery,
            plan_updated_goal,
            proposed_goal,
            goal_before: goal_before.clone(),
            progress_tracker,
            evidence,
            implementation_tracker,
            review_repair,
            tool_guardrail,
            empty_tui_needs_project,
            sched_tool_calls,
            sched_max_concurrent,
            sched_serial_runs,
            tool_timeline,
            advertised_tool_names,
            tool_schema_tokens,
            prev_call_sig,
            prev_added_no_evidence,
            retry_state,
            request_max_tokens_override,
            compat_fallbacks,
            effective_fallback_route,
            independent_review_status,
            independent_review_repairs,
            verification_infrastructure_error,
            verification_unstable,
            verified_at,
            last_verify_attributions,
            turn_snapshot,
            turn_start,
        };

        if turn.empty_tui_needs_project {
            turn.flags.force_tools_next = true;
            self.messages
                .push_nudge(NudgeKind::Continue, IMPLEMENTATION_EMPTY_TUI_NUDGE);
        }

        'turn: loop {
            // Inner loop: Model → Tools → Steer until tools stop, or step cap.
            let hit_cap = loop {
                match self
                    .run_model_round(&mut turn.as_model_round_state(), ui)
                    .await?
                {
                    super::model_round::ModelRoundControl::Continue => continue,
                    super::model_round::ModelRoundControl::BreakInner(hit) => break hit,
                    super::model_round::ModelRoundControl::RunTools {
                        calls,
                        completion_content,
                    } => {
                        let mut completion_content = completion_content;
                        turn.flags.made_tool_call = true;
                        turn.silent_continues = 0;
                        // Tools ran — drop one-shot force flags for the next Model round.
                        turn.flags.clear_one_shot_forces();
                        self.set_turn_phase(TurnPhase::Tools);
                        let batch = self
                            .execute_tool_batch(
                                &calls,
                                &mut completion_content,
                                turn.read_only_intent,
                                turn.max_parallel_tools,
                                &turn.task_contract,
                                &mut turn.implementation_tracker,
                                &mut turn.evidence,
                                &mut turn.tool_guardrail,
                                &mut turn.progress_tracker,
                                &mut turn.tool_timeline,
                                &mut turn.sched_tool_calls,
                                &mut turn.sched_max_concurrent,
                                &mut turn.sched_serial_runs,
                                &mut turn.plan_updated_goal,
                                &mut turn.proposed_goal,
                                &mut turn.turn_snapshot,
                                &mut turn.turn_checkpoint_allowed,
                                &mut turn.turn_checkpoint_created,
                                &mut turn.fast_feedback,
                                ui,
                            )
                            .await?;
                        match self.steer_after_tools(
                            &calls,
                            &batch,
                            turn.expected_mutation,
                            turn.read_only_intent,
                            turn.implementation_intent,
                            &mut turn.implementation_tracker,
                            &mut turn.evidence,
                            &mut turn.mutation_recovery,
                            &mut turn.progress_tracker,
                            &mut turn.repeat_nudges,
                            &mut turn.flags.force_tools_next,
                            &mut turn.flags.text_tool_fallback_next,
                            &mut turn.flags.force_no_progress_final_answer_next,
                            &mut turn.prev_added_no_evidence,
                            &mut turn.flags.stalled_repeating,
                            &mut turn.flags.stalled_unfinished,
                            ui,
                        ) {
                            super::steer::RoundControl::Continue => {}
                            super::steer::RoundControl::BreakInner(hit) => break hit,
                        }
                    }
                }
            };

            if hit_cap {
                ui.status(&format!(
                    "reached step limit ({}); stopping turn",
                    turn.max_steps
                ));
                turn.flags.ended_at_cap = true;
                break 'turn;
            }

            // TurnPhase::WorkspaceRepair — compile/lint/test stages; not review repair.
            // The state machine lives in WorkspaceRepairVerifier; this loop reacts.
            self.set_turn_phase(TurnPhase::WorkspaceRepair);
            let outcome = self
                .run_workspace_repair_verification(
                    &mut turn.verifier,
                    &turn.turn_background_baseline,
                    &mut turn.turn_snapshot,
                    turn.turn_checkpoint_created,
                    turn.turn_ledger_revision,
                    &turn.fast_feedback,
                    ui,
                )
                .await?;
            // Retain turn.evidence immediately, not only in the common finalizer:
            // reconciliation or persistence can still fail after a successful
            // check, and reports for those error turns need the stages that
            // actually ran.
            self.report.last_turn_telemetry.verification_executions =
                turn.verifier.executions().to_vec();
            match self
                .handle_workspace_repair_outcome(
                    outcome,
                    &mut turn.verifier,
                    turn.turn_ledger_revision,
                    turn.expected_mutation,
                    &turn.context_task,
                    turn.repository_context_enabled,
                    &mut super::verify_outcome::VerifyOutcomeState {
                        obligation_nudge_fired: &mut turn.flags.obligation_nudge_fired,
                        force_tools_next: &mut turn.flags.force_tools_next,
                        verified_at: &mut turn.verified_at,
                        independent_review_status: &mut turn.independent_review_status,
                        independent_review_repairs: &mut turn.independent_review_repairs,
                        stalled_unfinished: &mut turn.flags.stalled_unfinished,
                        verification_infrastructure_error: &mut turn
                            .verification_infrastructure_error,
                        verification_unstable: &mut turn.verification_unstable,
                        last_verify_attributions: &mut turn.last_verify_attributions,
                        ranked_context_paths: &mut turn.ranked_context_paths,
                        context_generation_seen: &mut turn.context_generation_seen,
                        indexed_ledger_revision: &mut turn.indexed_ledger_revision,
                    },
                    ui,
                )
                .await?
            {
                super::verify_outcome::VerifyOutcomeControl::BreakTurn => break 'turn,
                super::verify_outcome::VerifyOutcomeControl::ReenterModel => continue 'turn,
            }
        }

        // TurnPhase::Settle — seal checkpoint, then keep/wipe green verify.
        self.set_turn_phase(TurnPhase::Settle);
        // Seal first: checkpoint creation may take long enough for an owned
        // process or editor to move the tree. The authoritative reconciliation
        // below therefore happens after this final asynchronous safety step.
        if turn.turn_checkpoint_created && !self.seal_turn_checkpoint(ui).await? {
            turn.turn_checkpoint_created = false;
            // Default YOLO permits checkpoint-free mutation. A seal failure
            // must be silent and non-terminal there; strict confirmation mode
            // still treats loss of its promised undo record as incomplete.
            turn.flags.stalled_unfinished |= !self.config.gates.allow_no_checkpoint;
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
                ledger.changes_since(turn.turn_ledger_revision),
            )
        };
        {
            let delta = {
                let ledger = self.runtime.ledger();
                match turn.verified_at.as_ref() {
                    Some((revision, _)) => ledger.changes_since(*revision),
                    None => ledger_changes.clone(),
                }
            };
            super::settlement::reconcile_verified_revision(
                &mut self.report.last_verify,
                &mut turn.verified_at,
                &mut turn.independent_review_status,
                final_ledger_revision,
                final_workspace_revision.clone(),
                &delta,
                ui,
            );
        }
        self.workspace.last_changed_files = ledger_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        self.workspace.last_file_changes = ledger_changes;
        self.report.last_compat_fallbacks = turn.compat_fallbacks.clone();
        // Flush the per-turn counters (otherwise discarded locals) into
        // telemetry so `--report` / the eval harness can diagnose the turn's
        // trajectory: how many verify rounds, recovery retries, nudges fired,
        // and where the last verify failure pointed.
        self.report.last_turn_telemetry = build_turn_telemetry(
            turn.max_steps,
            turn.verifier.round(),
            turn.empty_retries,
            turn.repeat_nudges,
            turn.continue_total_nudges,
            turn.truncation_total_retries,
            &turn.progress_tracker,
            turn.flags.ended_at_cap,
            turn.flags.stalled_unfinished,
            turn.flags.stalled_repeating,
            &turn.last_verify_attributions,
            turn.verifier.executions(),
            turn.sched_tool_calls,
            turn.sched_max_concurrent,
            turn.sched_serial_runs,
            &turn.tool_timeline,
            &turn.evidence,
            &turn.review_repair,
        );
        self.report.last_turn_telemetry.checkpoint_available = turn
            .turn_checkpoint_allowed
            .map(|_| turn.turn_checkpoint_created);
        self.report.last_turn_telemetry.advertised_tools =
            turn.advertised_tool_names.iter().cloned().collect();
        self.report.last_turn_telemetry.tool_schema_tokens = turn.tool_schema_tokens;

        // Verifier-gated skill auto-curation: after a turn that PASSED verification
        // and actually changed files, optionally distill a reusable technique into a
        // learned skill. The ground-truth turn.verifier is the gate (safe with weak local
        // models); opt-in via `curate_skills`, and capped per session.
        if self.config.memory.curate_skills
            && self.report.last_verify == Some(true)
            && !self.workspace.last_changed_files.is_empty()
            && self.subagents.auto_skills_written < super::super::MAX_AUTO_SKILLS_PER_SESSION
        {
            self.curate_turn_end(turn_start, ui).await;
        }

        // Phase K: always-on (cheap, no model call) coding-fact extraction into
        // the decision log + project memory after a green file-changing turn.
        if self.report.last_verify == Some(true) && !self.workspace.last_changed_files.is_empty() {
            self.record_coding_facts_turn_end(ui);
        }

        // Surface the files this turn changed, so the user sees what was touched
        // without needing /diff. Skipped for read-only/Q&A turns (empty list).
        // Emitted BEFORE the finalize recap so the recap is the last text the
        // user sees (the "✓ done" marker follows it).
        if !self.workspace.last_changed_files.is_empty() {
            ui.changed_files(&self.workspace.last_changed_files);
        }

        // TurnPhase::Finalize — optional tool-free recap after mutating turns.
        // Requiring `made_tool_call` keeps plain Q&A from triggering it. Skipped
        // on step cap / stall (work may be incomplete).
        self.set_turn_phase(TurnPhase::Finalize);
        if self.config.memory.finalize
            && turn.flags.made_tool_call
            && !turn.flags.ended_at_cap
            && !turn.flags.stalled_unfinished
            && !turn.flags.stalled_repeating
            && !self.workspace.last_changed_files.is_empty()
            && steps < turn.max_steps
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
                ledger.changes_since(turn.turn_ledger_revision),
            )
        };
        {
            let delta = {
                let ledger = self.runtime.ledger();
                match turn.verified_at.as_ref() {
                    Some((revision, _)) => ledger.changes_since(*revision),
                    None => settled_changes.clone(),
                }
            };
            super::settlement::reconcile_verified_revision(
                &mut self.report.last_verify,
                &mut turn.verified_at,
                &mut turn.independent_review_status,
                settled_revision,
                settled_digest.clone(),
                &delta,
                ui,
            );
        }
        self.workspace.last_changed_files = settled_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        self.workspace.last_file_changes = settled_changes;

        // Long-horizon progress happens only after the final settled revision
        // still matches deterministic verification.
        // Keep the pre-turn goal until every user/session callback has
        // finished. A late workspace mutation must also roll back progress
        // that this hook tentatively advances.
        let goal_before_final_settlement = turn.goal_before.clone();
        let goal_invalidated_verification = self
            .goal_turn_end(
                super::super::goal_turn::GoalTurnState {
                    stalled_unfinished: turn.flags.stalled_unfinished,
                    stalled_repeating: turn.flags.stalled_repeating,
                    hit_step_cap: turn.flags.ended_at_cap,
                    plan_updated_goal: turn.plan_updated_goal,
                    proposed_goal: turn.proposed_goal.clone(),
                    goal_before: turn.goal_before.clone(),
                    verified_at: turn.verified_at.as_ref(),
                    turn_ledger_revision: turn.turn_ledger_revision,
                },
                ui,
            )
            .await;
        if goal_invalidated_verification {
            turn.verified_at = None;
            if turn.independent_review_status == ReviewStatus::Passed {
                turn.independent_review_status = ReviewStatus::Unavailable;
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
        // points outside the turn.verifier. Reconcile after all of them and before
        // constructing the typed outcome so none can create a false current-
        // revision pass. There are deliberately no callbacks after this
        // settlement point.
        self.reconcile_workspace_changes()?;
        let (outcome_revision, outcome_digest) = {
            let mut ledger = self.runtime.ledger();
            (ledger.revision(), ledger.workspace_revision())
        };
        let changed_after_final_hooks = self.report.last_verify == Some(true)
            && turn.verified_at.as_ref().is_none_or(|(revision, digest)| {
                *revision != outcome_revision || digest != &outcome_digest
            });
        if changed_after_final_hooks {
            let delta = {
                let ledger = self.runtime.ledger();
                match turn.verified_at.as_ref() {
                    Some((revision, _)) => ledger.changes_since(*revision),
                    None => ledger.changes_since(turn.turn_ledger_revision),
                }
            };
            let wiped = super::settlement::reconcile_verified_revision_with_message(
                &mut self.report.last_verify,
                &mut turn.verified_at,
                &mut turn.independent_review_status,
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
                    self.goals.set_structured(Some(previous));
                    self.refresh_system_message();
                    // The earlier persist may contain tentatively advanced goal
                    // state. Rewrite the goal record itself (message persistence
                    // does not include side-channel goal state) before returning.
                    if let Some(session) = self.session.as_mut()
                        && let Some(goal) = self.goals.structured.as_ref()
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
                ledger.changes_since(turn.turn_ledger_revision),
                ledger.had_mutation_since(turn.turn_ledger_revision),
            )
        };
        self.workspace.last_changed_files = final_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        self.workspace.last_file_changes = final_changes;

        // `Unverified` is reserved for "checks should have run but did not
        // settle" (budget exhausted after a fail, post-pass code mutation, etc.).
        // When the pipeline never ran a stage — disabled, no auto markers, prose
        // only, empty effective stages — the honest public state is
        // `NotApplicable` ("no applicable checks"), not a scary incomplete
        // "unverified changes" warning. Users still get `Unverified` when a
        // check was expected and missing.
        let no_check_executed = self
            .report
            .last_turn_telemetry
            .verification_executions
            .is_empty();
        let (status, verification, review, stop_reason) = super::finalize::classify_turn_outcome(
            turn.verification_infrastructure_error,
            turn.verification_unstable,
            self.report.last_verify,
            &self.workspace.last_changed_files,
            turn_had_mutation,
            no_check_executed,
            turn.independent_review_status,
            self.report.last_turn_telemetry.skeptic_last_status,
            turn.flags.ended_at_cap,
            turn.flags.stalled_unfinished,
            turn.flags.stalled_repeating,
            turn.expected_mutation,
            self.config.gates.allow_unverified,
        );
        // Outer `run_turn` also stamps Done (covers `?` paths); keep the success path explicit.
        self.set_turn_phase(TurnPhase::Done);
        let outcome = TurnOutcome {
            status,
            verification,
            review,
            stop_reason,
            changed_files: self.workspace.last_changed_files.clone(),
            verified_workspace_revision: (verification == VerificationStatus::Passed)
                .then(|| turn.verified_at.as_ref().map(|(_, digest)| digest.clone()))
                .flatten(),
            effective_route: effective_model_route(
                &self.config,
                turn.effective_fallback_route.as_deref(),
            ),
        };
        self.report.set_outcome(outcome.clone());
        self.workspace.clear_active_baselines();
        Ok(outcome)
    }
}
