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

use anyhow::{Context, Result};

use crate::command;
use crate::domain::TurnControlFlags;
use crate::domain::VerifyEvidence;
use crate::heuristics::{looks_like_continue, looks_like_new_task, tool_mode_label};
use crate::steering::{
    EvidenceTracker, IMPLEMENTATION_EMPTY_TUI_NUDGE, ImplementationIntent, ImplementationTracker,
    MutationRecovery, ReviewIntent, ToolLoopGuardrail, classify_implementation_intent,
    classify_read_only_intent, implementation_mentions_tui, implementation_turn_prompt,
    implicit_read_only_review_intent, is_bounded_file_review, preflight_is_redundant_for_prompt,
    read_only_turn_prompt,
};
use crate::transcript::NudgeKind;
use crate::verify::{Snapshot, WorkspaceRepairVerifier, is_internal_runtime_artifact_path};
use crate::{
    ReviewStatus, TaskContract, TaskIntent, ToolCallEntry, TurnOutcome, TurnStatus, TurnStopReason,
    TurnTelemetry, Ui, VerificationMode, VerificationStatus,
};
use hi_ai::{ToolMode, estimate_text_tokens};

use super::entry::TurnCancellationRequested;
use super::helpers::{
    build_turn_telemetry, effective_max_steps_for_turn, effective_model_route,
    task_needs_repository_context,
};
use super::phase::TurnPhase;
use super::progress::ProgressTracker;
use super::retry::{ReviewRepairState, TurnRetryState};

impl crate::Agent {
    pub(in crate::agent::turn) async fn run_turn_core(
        &mut self,
        input: &str,
        ui: &mut dyn Ui,
    ) -> Result<TurnOutcome> {
        self.set_turn_phase(TurnPhase::Setup);
        // Immediate `/btw` launcher — TUI fires asides without waiting for a
        // model-round boundary. Refreshed each round; disarmed at join.
        self.arm_btw_dispatcher();
        // Repair-effort escalation is turn-scoped.
        self.repair_effort_escalated = false;
        // A leftover block request: `goal_turn_end` consumes it, so a turn that
        // errored out before reaching it would otherwise carry the request into
        // the next turn and set aside whatever step is active by then.
        self.pending_block = None;
        let user_prompt_tokens = estimate_text_tokens(input);
        // Reset the per-turn file-read cache. It's invalidated per-key by the
        // edit tools and wholesale after `bash`, but clearing it here restores
        // its documented per-turn contract — so a file changed outside `hi`
        // between turns is re-read fresh, not served from a prior turn's cache.
        self.runtime.clear_read_cache();
        // The initial ledger scan is allowed to run in the background during
        // startup, but a turn baseline must not be established against an
        // incomplete snapshot: otherwise external edits made during setup can
        // be absorbed into the scan and disappear from turn attribution.
        self.runtime.ensure_ledger_scan_complete_async().await?;
        // Reconcile user/external edits before establishing this turn's
        // baseline so they are not attributed to the agent. Off the drive
        // task's blocking path so a large workspace walk cannot freeze the TUI.
        let initial_external_changes = self.runtime.reconcile_ledger_async().await?;
        if !initial_external_changes.is_empty() {
            // The repository map is invalidated by tool effects and the
            // agent-level reconciliation path. This setup reconciliation is
            // intentionally direct, so keep the same invariant for edits
            // made by the user between turns.
            self.runtime.clear_repo_map_cache();
        }
        let turn_ledger_revision = self.runtime.ledger().revision();
        let turn_background_baseline = self.runtime.background().ids();
        // Ledger + bg baselines + per-turn caches (cancel-safe finalizers).
        self.workspace
            .begin_turn(turn_ledger_revision, turn_background_baseline.clone());
        let expanded_input =
            command::expand_prompt_macro(input).unwrap_or_else(|| input.to_string());
        self.begin_drive_turn(crate::DriveKind::from_prompt(&expanded_input));
        // Synthetic goal-drive text is only transport. Contracts, context
        // ranking, review, and implementation guards need the real objective
        // and active milestone—especially explicit paths such as plan.md.
        let goal_context = self.goal_continuation_context(&expanded_input);
        let goal_drive_turn = goal_context.is_some();
        let plan_context = self.plan_continuation_context(&expanded_input);
        let plan_drive_turn = plan_context.is_some();
        // Charge the turn budget up front: a turn that errors out still spent
        // the time, so counting only successful turns would let a failing goal
        // run past its ceiling indefinitely.
        if goal_drive_turn && let Some(goal) = self.goals.structured.as_mut() {
            goal.spend_turn();
        }
        let context_task = goal_context
            .or(plan_context)
            .unwrap_or_else(|| expanded_input.clone());
        let structurally_read_only_subagent = self.config.subagents.is_subagent
            && self.config.routing.tool_mode == ToolMode::ReadOnly;
        let mut task_contract =
            TaskContract::derive(&context_task, self.config.gates.verification.clone());
        let read_only_intent = classify_read_only_intent(&context_task).or_else(|| {
            implicit_read_only_review_intent(
                &context_task,
                task_contract.intent == TaskIntent::ReadOnly,
            )
        });
        // Capability scope is authoritative for an explore child. Its quoted
        // question may contain mutation verbs ("what should we build next"),
        // but the child is an investigator, not an implementer. Letting prompt
        // wording override that scope activates mutation completion guards that
        // it can never satisfy and previously turned valid reads into denials.
        // An explicit no-mutation request is equally authoritative. Apply both
        // scopes before refreshing tools so the first provider request does not
        // advertise mutation or broad-review schemas that the task cannot use.
        if structurally_read_only_subagent || read_only_intent.is_some() {
            task_contract.intent = TaskIntent::ReadOnly;
            task_contract.explicit_mutation = false;
        }
        self.refresh_tools_for_task(&context_task, task_contract.intent);
        // A closed-set exact-file review already tells the model precisely
        // which evidence to inspect. Building the ranked repository index for
        // those turns duplicates the same paths in the prompt, inflates every
        // DeepSeek request, and can push a small review into a huge context
        // before the first tool call. Keep repository context for broad reviews
        // and implementation work, where it provides useful orientation.
        let bounded_file_review =
            is_bounded_file_review(&context_task, task_contract.intent == TaskIntent::Mutation);
        let targeted_named_mutation = super::super::tool_selection::targeted_named_file_mutation(
            &context_task,
            task_contract.intent == TaskIntent::Mutation,
        );
        // Goal/plan continuations name a doc (plan.md) but still need the
        // repository map. Ordinary "write driver.py" prompts already named
        // the files and should not pay for a 6k-char index.
        let skip_index_for_named_edit =
            targeted_named_mutation && !goal_drive_turn && !plan_drive_turn;
        let repository_context_enabled = !bounded_file_review
            && !skip_index_for_named_edit
            && task_needs_repository_context(&context_task, &task_contract);
        let ranked_context_paths = self
            .workspace
            .last_changed_files
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let (ranked_context_paths, task_context) = if repository_context_enabled {
            let root = self.runtime.root().to_path_buf();
            let task = context_task.clone();
            let exclusions = self.config.memory.context_exclusions.clone();
            let repo_map = self.runtime.repo_map_arc();
            tokio::task::spawn_blocking(move || {
                let mut ranked_context_paths = ranked_context_paths;
                for path in hi_tools::ranked_paths_for_task(&root, &task, repo_map.as_ref(), 12) {
                    ranked_context_paths.insert(path);
                }
                let paths = ranked_context_paths.iter().cloned().collect::<Vec<_>>();
                let index = crate::context_index::build_task_context_index(
                    &root,
                    &task,
                    &paths,
                    &exclusions,
                );
                let orientation = hi_tools::orientation_for_task(&root, &task, repo_map.as_ref());
                let task_context = match (orientation, index) {
                    (Some(seed), Some(index)) => Some(format!("{seed}\n\n{index}")),
                    (Some(seed), None) => Some(seed),
                    (None, index) => index,
                };
                (ranked_context_paths, task_context)
            })
            .await
            .context("repository context worker failed")?
        } else {
            (ranked_context_paths, None)
        };
        self.task.set_task_context(task_context);
        let context_generation_seen = self.runtime.context_generation();
        let indexed_ledger_revision = self.runtime.ledger().revision();
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
        self.task
            .set_task(Some(context_task.clone()), Some(task_contract.clone()));
        // Rank memory for this task now: the volatile context block attached
        // to this turn's user message below must carry fresh memory, since
        // the per-round refresh no longer rewrites any message.
        self.refresh_memory_context(&context_task);
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
        // Validation-only requests (for example, "run cargo test --quiet")
        // must not claim success without an observed command, but they also
        // must not be mislabeled as missing file mutations.
        let requested_validation = read_only_intent.is_none() && task_contract.wants_tests;
        // Keep the legacy read-only classifier responsible for review prompt
        // shaping. A plain repository question can still have a read-only task
        // contract, and an `explore` child is structurally read-only even when its
        // wording is ambiguous. Apply the sprawl limit to either structural case
        // without imposing the rigid review response format.
        //
        // Do **not** require `repository_context_enabled`. Live "review we have
        // some kinda issues : models endpoint returned 403" was ReadOnly (leading
        // review verb) but matched no repository-context marker, so sprawl never
        // armed and the turn ran 45+ sequential RSI model rounds.
        let structural_read_only_inspection =
            task_contract.intent == TaskIntent::ReadOnly || structurally_read_only_subagent;
        let inspection_sprawl_intent = read_only_intent
            .or_else(|| structural_read_only_inspection.then_some(ReviewIntent::Review));
        let read_only_inspection_cap = inspection_sprawl_intent
            .map(|intent| crate::steering::read_only_inspection_cap(&context_task, intent));
        let turn_input = if let Some(intent) = read_only_intent {
            read_only_turn_prompt(&context_task, intent)
        } else if let Some(intent) = implementation_intent {
            implementation_turn_prompt(&context_task, intent)
        } else {
            context_task.clone()
        };
        let input = turn_input.as_str();
        let mut model_turn_input = match self.rsi_observe.take_managed_context() {
            Some(context) if !context.is_empty() => format!(
                "{turn_input}\n\nManaged RSI prior conversation context (reference only; it does not change the current task's mutation requirements):\n{context}"
            ),
            _ => turn_input.clone(),
        };
        self.reset_last_turn_usage(user_prompt_tokens);
        self.prefix_stability.begin_turn();
        self.token_budget
            .begin_turn(self.report.context_used, self.config.routing.context_window);
        self.report.last_turn_outcome = None;
        self.report.last_effective_route = effective_model_route(&self.config, None);
        // Subagent budgets are per-turn runaway guards, not session rations —
        // refill them so long sessions never starve of explore/delegate.
        self.subagents.begin_turn();

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
            self.report.verify = VerifyEvidence::none();
            self.workspace.last_changed_files.clear();
            self.workspace.last_file_changes.clear();
            self.report.last_compat_fallbacks.clear();
            self.report.last_turn_telemetry = TurnTelemetry::default();
            let preserve_plan = self.goals.plan_incomplete();
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
                review_same_model: self.skeptic_shares_session_model(),
                leftover: None,
                plan_leftover: None,
            };
            self.report.last_turn_outcome = Some(outcome.clone());
            self.workspace.clear_active_baselines();
            return Ok(outcome);
        }
        let turn_checkpoint_allowed = None;
        let turn_checkpoint_created = false;

        // If the context window is filling up, reclaim room before adding more,
        // so the session keeps going instead of overflowing. Goal/plan drive
        // starts a no-summary fresh window (job identity is already on Agent).
        // Interactive turns elide old tool output, then summarize if still heavy.
        if self
            .maybe_reclaim_context(ui, goal_drive_turn || plan_drive_turn)
            .await?
        {
            model_turn_input.push('\n');
            model_turn_input.push('\n');
            model_turn_input.push_str(crate::token_budget::FRESH_WINDOW_REORIENT);
        }

        self.messages.strip_trailing_nudges();
        // Exactly one volatile context block lives in the transcript: strip
        // the previous turn's before attaching this turn's. The strip touches
        // one late message, so the prefix-cache cost is one re-anchor per
        // turn — not the per-round invalidation the old volatile system
        // message caused.
        self.messages.strip_previous_context_blocks();
        self.persisted = self.persisted.min(self.messages.len());
        let turn_start = self.messages.len();
        self.workspace.set_message_start(turn_start);
        let model_turn_input = match self.volatile_context_block() {
            Some(block) => format!(
                "{}\n{block}\n{}\n\n{model_turn_input}",
                crate::transcript::CONTEXT_BLOCK_START,
                crate::transcript::CONTEXT_BLOCK_END
            ),
            None => model_turn_input,
        };
        let typed_prompt = self.pending_prompt.take();
        if let Some(prompt) = typed_prompt {
            let mut message = prompt.into_message();
            // The loop's volatile context and macro expansion are text-only
            // transport; replace the original text block while retaining all
            // image blocks for the provider adapter.
            if let Some(text) = message.content.iter_mut().find_map(|content| {
                if let hi_ai::Content::Text(text) = content {
                    Some(text)
                } else {
                    None
                }
            }) {
                *text = model_turn_input.clone();
            } else {
                message
                    .content
                    .insert(0, hi_ai::Content::Text(model_turn_input.clone()));
            }
            self.messages.push_user_or_fold_message(message);
        } else {
            self.messages.push_user_or_fold(&model_turn_input);
        }
        self.persist_durable_boundary("prompt")?;
        self.report.verify = VerifyEvidence::none();
        self.workspace.last_changed_files.clear();
        self.workspace.last_file_changes.clear();
        self.report.last_compat_fallbacks.clear();
        self.report
            .last_turn_telemetry
            .verification_executions
            .clear();
        // Preserve an unfinished plan across follow-ups so the checklist stays
        // pinned. Completed-only plans still clear. A new user message that is
        // not a continue/drive prompt gets a replace-plan fold so a new task
        // can replace stale steps instead of inheriting PLAN_CONTINUE_NUDGE.
        let preserve_plan = self.goals.plan_incomplete();
        if self.goals.clear_plan_unless(preserve_plan) {
            if let Some(session) = self.session.as_mut() {
                session.clear_plan()?;
            }
            ui.plan(&[]);
        }
        let plan_drive_turn =
            crate::DriveKind::from_prompt(&expanded_input) == crate::DriveKind::Plan;
        if preserve_plan
            && !goal_drive_turn
            && !plan_drive_turn
            && !looks_like_continue(&expanded_input)
            && !self.config.subagents.is_subagent
        {
            let nudge = if looks_like_new_task(&expanded_input) {
                crate::NEW_TASK_REPLACE_PLAN_NUDGE
            } else {
                crate::REPLACE_PLAN_NUDGE
            };
            self.messages.push_nudge_or_fold(NudgeKind::Continue, nudge);
        }
        let compat_fallbacks = Vec::new();
        let effective_fallback_route: Option<String> = None;

        let resolved_verify_stages = self
            .config
            .gates
            .verification
            .resolved_stages(self.runtime.root());
        let verify_rounds = self.config.gates.max_verify_repairs.saturating_add(1);
        // Workspace repair only — not review-answer repair (see ReviewRepairState).
        let verifier = if matches!(&self.config.gates.verification, VerificationMode::Auto) {
            WorkspaceRepairVerifier::automatic(resolved_verify_stages, verify_rounds)
        } else {
            WorkspaceRepairVerifier::new(resolved_verify_stages, verify_rounds)
        };
        // Mid-turn LSP + affected cargo check state (dedupes packages across batches).
        let fast_feedback = super::fast_feedback::FastFeedbackState::default();
        let max_steps = effective_max_steps_for_turn(&self.config);
        let max_parallel_tools = self.config.loop_limits.max_parallel_tools.max(1);
        let empty_retries = 0u32;
        // Consecutive output-limit continuations. This is a stall budget, so it
        // resets after any non-truncated model response/tool progress.
        let truncation_retries = 0u32;
        // Cumulative truncation nudges for telemetry/UI summaries. Unlike the
        // consecutive budget above, this should not reset mid-turn.
        let truncation_total_retries = 0u32;
        let silent_continues = 0u32;
        let continue_total_nudges = 0u32;
        let repeat_nudges = 0u32;
        let progress_tracker = ProgressTracker::default();
        // Per-turn control flags (force-next-tool, stalls, caps, obligation).
        // See [`TurnControlFlags`] — field projection keeps call sites direct.
        let mut flags = TurnControlFlags::default();
        // Bounded discovery narrows the advertised catalog until the model
        // records a plan or makes the requested edit.
        let mutation_recovery = MutationRecovery::default();
        // A model-authored plan is only a proposal until deterministic
        // verification passes for the settled workspace revision. Keeping it
        // turn-local prevents failed, unverified, cancelled, or infrastructure-
        // error turns from leaking goal progress into the live session.
        let plan_updated_goal = false;
        let proposed_goal: Option<crate::Goal> = None;
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
        let advertised_tool_names = BTreeSet::new();
        let tool_schema_tokens = 0_u64;
        let mut evidence = EvidenceTracker::default();
        let review_repair = ReviewRepairState::default();
        let independent_review_status = ReviewStatus::NotRequired;
        let independent_review_repairs = 0_u32;
        let review_unavailable_reason: Option<String> = None;
        let verification_infrastructure_error = false;
        let verification_unstable = false;
        // A pass is bound to both the ledger event number and the full content
        // digest observed immediately after the verifier. Later workspace
        // activity must never inherit that pass. (The bound evidence now lives
        // in `TurnReportState::verify` as `VerifyEvidence::Passed`, fused with
        // the verdict so the two cannot diverge.)
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
            && !preflight_is_redundant_for_prompt(self.runtime.root(), &context_task)
        {
            // Deterministic preflight is useful context, but it must not
            // consume the entire explicit tool budget before the model gets a
            // chance to act. Reserve at least half the cap (rounded up) for
            // model-ordered tools; with a cap of one, skip preflight entirely.
            let remaining_tool_budget = self
                .config
                .loop_limits
                .max_tool_calls
                .saturating_sub(sched_tool_calls);
            let model_tool_reserve = remaining_tool_budget.saturating_add(1) / 2;
            let preflight_tool_budget = remaining_tool_budget.saturating_sub(model_tool_reserve);
            let preflight = self
                .run_read_only_preflight(
                    intent,
                    &context_task,
                    read_only_inspection_cap.unwrap_or_else(|| evidence.inspection_attempt_count()),
                    ui,
                    &mut evidence,
                    &mut tool_timeline,
                    preflight_tool_budget,
                )
                .await;
            if preflight.executed > 0 {
                flags.made_tool_call = true;
                sched_tool_calls = sched_tool_calls.saturating_add(preflight.executed);
                sched_serial_runs = sched_serial_runs.saturating_add(preflight.serial_runs);
                sched_max_concurrent = sched_max_concurrent.max(preflight.max_concurrent_batch);
            }
            flags.force_tools_next |= preflight.interrupted;
        }
        if implementation_intent.is_some()
            && !self
                .config
                .rsi
                .remote_switch
                .as_ref()
                .is_some_and(|enabled| enabled.load(std::sync::atomic::Ordering::SeqCst))
            && !matches!(self.config.routing.tool_mode, ToolMode::ChatOnly)
            // Keep one model-ordered tool slot when the caller selected a
            // one-call hard cap; the deterministic validation probe must not
            // make a coding turn unable to edit anything.
            && self
                .config
                .loop_limits
                .max_tool_calls
                .saturating_sub(sched_tool_calls)
                > 1
        {
            let preflight = self
                .run_implementation_preflight(ui, &mut implementation_tracker, &mut tool_timeline)
                .await;
            if preflight.executed > 0 {
                flags.made_tool_call = true;
                sched_tool_calls = sched_tool_calls.saturating_add(preflight.executed);
                sched_serial_runs = sched_serial_runs.saturating_add(preflight.serial_runs);
                sched_max_concurrent = sched_max_concurrent.max(preflight.max_concurrent_batch);
            }
            flags.force_tools_next |= preflight.interrupted;
            empty_tui_needs_project = implementation_intent.is_some_and(|intent| intent.tui)
                && implementation_tracker.preferred_validation.is_none();
        }
        // Signature (name, arguments) of the previous round's tool calls, to
        // spot a model re-issuing the exact same call and looping on it.
        let prev_call_sig: Option<Vec<(String, String)>> = None;
        // Whether the previous executed round added no new evidence (every call
        // was a read-only inspection already seen). Used by the no-new-evidence
        // cycle guard to fire only on the *second* consecutive wasted round,
        // preserving a single legitimate re-inspection after new evidence.
        let prev_added_no_evidence = false;
        let retry_state = TurnRetryState::default();
        let request_max_tokens_override: Option<u32> = None;
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
        let repeat_sampling_rounds = 0u32;
        let tool_guardrail = ToolLoopGuardrail::default();
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
        let last_verify_attributions: Vec<hi_tools::Attribution> = Vec::new();
        // Snapshot the turn baseline lazily. Read-only/chat turns should not
        // walk the whole workspace just to prove nothing changed; the baseline
        // is captured before the first actual mutation, or before verification
        // when verify stages are configured.
        let turn_snapshot: Option<Snapshot> = None;
        // Snapshot from the most recent verify check. Reused at turn end to
        // avoid a second full tree walk when verify already took one.

        // Owned per-turn bag — Model/Tools/Steer/Verify project from this.
        let mut turn = super::state::TurnState {
            phase_latencies: crate::TurnPhaseLatencies::default(),
            user_prompt_tokens,
            turn_ledger_revision,
            turn_background_baseline: turn_background_baseline.clone(),
            context_task: context_task.clone(),
            task_contract: task_contract.clone(),
            repository_context_enabled,
            ranked_context_paths,
            context_generation_seen,
            indexed_ledger_revision,
            read_only_intent,
            implementation_intent,
            expected_mutation,
            requested_validation,
            inspection_sprawl_intent,
            read_only_inspection_cap,
            turn_input: input.to_string(),
            turn_checkpoint_allowed,
            turn_checkpoint_created,
            verifier,
            fast_feedback,
            max_steps,
            max_parallel_tools,
            steps: 0,
            empty_retries,
            truncation_retries,
            truncation_total_retries,
            silent_continues,
            generic_completion_retries: 0,
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
            deepseek_strict_fallback_active: false,
            deepseek_strict_fallback_used: false,
            retry_state,
            request_max_tokens_override,
            compat_fallbacks,
            effective_fallback_route,
            independent_review_status,
            independent_review_repairs,
            review_unavailable_reason,
            verification_infrastructure_error,
            verification_unstable,
            last_verify_attributions,
            turn_snapshot,
            turn_start,
        };

        if turn.empty_tui_needs_project {
            turn.flags.force_tools_next = true;
            self.messages
                .push_nudge(NudgeKind::Continue, IMPLEMENTATION_EMPTY_TUI_NUDGE);
        }

        // Soft wall-clock budget. Anchored here rather than at Setup so slow
        // context/indexing work before the first model call cannot consume the
        // whole allowance before any work is attempted.
        let turn_deadline_at = self
            .config
            .loop_limits
            .turn_soft_deadline
            .map(|budget| std::time::Instant::now() + budget);
        let deadline_expired =
            || turn_deadline_at.is_some_and(|deadline| std::time::Instant::now() >= deadline);
        'turn: loop {
            // Stop *starting* new work once the budget is spent; work already
            // in flight is never interrupted. Falling through to Settle means
            // the workspace is reconciled and the report written, instead of an
            // external kill freezing whatever happened to be on disk.
            if deadline_expired() {
                ui.status(
                    "wall-clock budget for this turn is spent; finishing with the current state",
                );
                turn.flags.ended_at_deadline = true;
                break 'turn;
            }
            // Whole-turn cancel (frontend Ctrl+C / Esc): leave the model loop
            // before the next provider round so cleanup_turn can run with a
            // coherent transcript rather than dropping mid-stream.
            if self
                .turn_cancellation
                .as_ref()
                .is_some_and(|c| c.is_cancelled())
            {
                return Err(TurnCancellationRequested.into());
            }
            // Inner loop: Model → Tools → Steer until tools stop, or step cap.
            let hit_cap = loop {
                // Checked per round, not just per outer iteration: a model that
                // keeps calling tools never returns to the outer loop, so an
                // outer-only check let a turn run to the external kill without
                // the budget ever being consulted. Breaking with `false` (not
                // the cap signal) still runs verification on what exists — the
                // settle path — and the ReenterModel gate below ends the turn
                // rather than starting another repair round.
                if deadline_expired() {
                    ui.status(
                        "wall-clock budget for this turn is spent; wrapping up with the current state",
                    );
                    turn.flags.ended_at_deadline = true;
                    // Breaking with `false` (not the cap signal) falls through
                    // to verification, which is the point: settle on what
                    // exists. But that path enters WorkspaceRepair, which is
                    // only a legal phase transition once a model round has run.
                    // The budget can expire in the window between the outer
                    // check and this one, so guard it: with no round yet there
                    // is also nothing new to verify.
                    if turn.steps == 0 {
                        break 'turn;
                    }
                    break false;
                }
                if self
                    .turn_cancellation
                    .as_ref()
                    .is_some_and(|c| c.is_cancelled())
                {
                    return Err(TurnCancellationRequested.into());
                }
                let model_started = std::time::Instant::now();
                let model_result = self
                    .run_model_round(&mut turn.as_model_round_state(), ui)
                    .await;
                turn.phase_latencies.model_request_ms = turn
                    .phase_latencies
                    .model_request_ms
                    .saturating_add(model_started.elapsed().as_millis() as u64);
                // Context fitting and request-too-large recovery can replace the
                // transcript and re-anchor the active user turn inside
                // `run_model_round`. Keep cancellation cleanup on the same
                // boundary instead of letting it truncate with a stale index.
                self.workspace.set_message_start(turn.turn_start);
                match model_result? {
                    super::model_round::ModelRoundControl::Continue => continue,
                    super::model_round::ModelRoundControl::BreakInner(hit) => break hit,
                    super::model_round::ModelRoundControl::RunTools {
                        calls,
                        completion_content,
                        tool_specs,
                    } => {
                        let mut completion_content = completion_content;
                        turn.flags.made_tool_call = true;
                        turn.silent_continues = 0;
                        // Tools ran — drop one-shot force flags for the next Model round.
                        turn.flags.clear_one_shot_forces();
                        self.set_turn_phase(TurnPhase::Tools);
                        let tool_started = std::time::Instant::now();
                        let batch_result = self
                            .execute_tool_batch(
                                &calls,
                                &mut completion_content,
                                &tool_specs,
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
                            .await;
                        turn.phase_latencies.tool_batch_ms = turn
                            .phase_latencies
                            .tool_batch_ms
                            .saturating_add(tool_started.elapsed().as_millis() as u64);
                        let batch = batch_result?;
                        self.persist_durable_boundary("tool")?;
                        let steer = self.steer_after_tools(
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
                            &mut turn.flags.suppress_bookkeeping_tools_next,
                            &mut turn.flags.text_tool_fallback_next,
                            &mut turn.flags.tool_validation_text_fallback_used,
                            &mut turn.flags.force_no_progress_final_answer_next,
                            &mut turn.prev_added_no_evidence,
                            &mut turn.prev_call_sig,
                            &mut turn.deepseek_strict_fallback_active,
                            &mut turn.deepseek_strict_fallback_used,
                            &mut turn.flags.stalled_repeating,
                            &mut turn.flags.stalled_unfinished,
                            ui,
                        );
                        if self.token_budget.take_pending_fresh_window() {
                            let task = self.task.last_task_prompt.clone();
                            self.apply_fresh_window(ui, task.as_deref())?;
                            turn.turn_start = self.messages.len().saturating_sub(1);
                            self.workspace.set_message_start(turn.turn_start);
                        }
                        match steer {
                            super::steer::RoundControl::Continue => {}
                            super::steer::RoundControl::BreakInner(hit) => break hit,
                        }
                    }
                }
            };

            if hit_cap {
                ui.status(
                    "reached turn work limit; verifying the current workspace before stopping",
                );
                turn.flags.ended_at_cap = true;
            }

            // TurnPhase::WorkspaceRepair — compile/lint/test stages; not review repair.
            // The state machine lives in WorkspaceRepairVerifier; this loop reacts.
            self.set_turn_phase(TurnPhase::WorkspaceRepair);
            ui.semantic_event(hi_events::RunEvent::new(
                hi_events::EventKind::VerificationStarted,
                hi_events::EventContext::default(),
                hi_events::SemanticActivity {
                    verb: hi_events::ActivityVerb::Verify,
                    object: hi_events::ActivityObject::Verification,
                    state: hi_events::ActivityState::Running,
                    group_key: format!("verification:turn:{}", self.turn_count),
                    title: "verification started".into(),
                    detail: None,
                    refs: Vec::new(),
                    progress: None,
                },
            ));
            let verify_started = std::time::Instant::now();
            let outcome_result = self
                .run_workspace_repair_verification(
                    &mut turn.verifier,
                    &turn.turn_background_baseline,
                    &mut turn.turn_snapshot,
                    turn.turn_checkpoint_created,
                    turn.turn_ledger_revision,
                    &turn.fast_feedback,
                    ui,
                )
                .await;
            turn.phase_latencies.verify_ms = turn
                .phase_latencies
                .verify_ms
                .saturating_add(verify_started.elapsed().as_millis() as u64);
            let outcome = outcome_result?;
            let (verification_state, verification_verb) = match &outcome {
                crate::verify::VerifyOutcome::Passed
                | crate::verify::VerifyOutcome::SkippedNoChanges { .. }
                | crate::verify::VerifyOutcome::SkippedProseOnly { .. } => (
                    hi_events::ActivityState::Succeeded,
                    hi_events::ActivityVerb::Complete,
                ),
                crate::verify::VerifyOutcome::Failed { .. }
                | crate::verify::VerifyOutcome::InfrastructureError { .. } => (
                    hi_events::ActivityState::Failed,
                    hi_events::ActivityVerb::Fail,
                ),
                crate::verify::VerifyOutcome::Unstable { .. } => (
                    hi_events::ActivityState::Waiting,
                    hi_events::ActivityVerb::Wait,
                ),
                crate::verify::VerifyOutcome::NotRun => (
                    hi_events::ActivityState::Waiting,
                    hi_events::ActivityVerb::Wait,
                ),
            };
            ui.semantic_event(hi_events::RunEvent::new(
                hi_events::EventKind::VerificationCompleted,
                hi_events::EventContext::default(),
                hi_events::SemanticActivity {
                    verb: verification_verb,
                    object: hi_events::ActivityObject::Verification,
                    state: verification_state,
                    group_key: format!("verification:turn:{}", self.turn_count),
                    title: "verification finished".into(),
                    detail: None,
                    refs: Vec::new(),
                    progress: None,
                },
            ));
            // Retain turn evidence immediately, not only in the common finalizer:
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
                        independent_review_status: &mut turn.independent_review_status,
                        independent_review_repairs: &mut turn.independent_review_repairs,
                        review_unavailable_reason: &mut turn.review_unavailable_reason,
                        stalled_unfinished: &mut turn.flags.stalled_unfinished,
                        verification_infrastructure_error: &mut turn
                            .verification_infrastructure_error,
                        verification_unstable: &mut turn.verification_unstable,
                        last_verify_attributions: &mut turn.last_verify_attributions,
                        validation_after_last_mutation: turn
                            .implementation_tracker
                            .validation_after_last_mutation,
                        ranked_context_paths: &mut turn.ranked_context_paths,
                        context_generation_seen: &mut turn.context_generation_seen,
                        indexed_ledger_revision: &mut turn.indexed_ledger_revision,
                        progress_tracker: &mut turn.progress_tracker,
                        continue_total_nudges: &mut turn.continue_total_nudges,
                    },
                    ui,
                )
                .await?
            {
                super::verify_outcome::VerifyOutcomeControl::BreakTurn => break 'turn,
                // The repair loop is the classic budget sink: a check that
                // keeps failing re-enters the model until something external
                // kills the process. Honor the deadline here too, so a turn
                // that ran out of time still settles on its own terms.
                super::verify_outcome::VerifyOutcomeControl::ReenterModel => {
                    if turn.flags.ended_at_cap {
                        ui.status(
                            "verification still needs repair, but the turn work limit is spent; ending incomplete",
                        );
                        turn.flags.stalled_unfinished = true;
                        break 'turn;
                    }
                    if deadline_expired() {
                        ui.status(
                            "wall-clock budget spent during repair; finishing with the current state",
                        );
                        turn.flags.ended_at_deadline = true;
                        break 'turn;
                    }
                    continue 'turn;
                }
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
        self.reconcile_workspace_changes().await?;
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
                match self.report.verify.bound_revision_digest() {
                    Some((revision, _)) => ledger.changes_since(revision),
                    None => ledger_changes.clone(),
                }
            };
            let review_was_passed = turn.independent_review_status == ReviewStatus::Passed;
            super::settlement::reconcile_verified_revision(
                &mut self.report.verify,
                &mut turn.independent_review_status,
                final_ledger_revision,
                final_workspace_revision.clone(),
                &delta,
                ui,
            );
            if review_was_passed && turn.independent_review_status == ReviewStatus::Unavailable {
                turn.review_unavailable_reason =
                    Some("a workspace change after the review pass invalidated it".into());
            }
        }
        let project_changes: Vec<_> = ledger_changes
            .into_iter()
            .filter(|change| !is_internal_runtime_artifact_path(&change.path))
            .collect();
        self.workspace.last_changed_files = project_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        self.workspace.last_file_changes = project_changes;
        self.report.last_compat_fallbacks = turn.compat_fallbacks.clone();
        // Flush the per-turn counters (otherwise discarded locals) into
        // telemetry so `--report` / the eval harness can diagnose the turn's
        // trajectory: how many verify rounds, recovery retries, nudges fired,
        // and where the last verify failure pointed.
        let model_telemetry = self.report.last_turn_telemetry.clone();
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
            &self.prefix_stability,
        );
        self.report.last_turn_telemetry.model_requests = model_telemetry.model_requests;
        self.report.last_turn_telemetry.accepted_completions = model_telemetry.accepted_completions;
        self.report.last_turn_telemetry.last_stop_reason = model_telemetry.last_stop_reason;
        self.report.last_turn_telemetry.tool_call_channel = model_telemetry.tool_call_channel;
        self.report.last_turn_telemetry.reasoning_requested = model_telemetry.reasoning_requested;
        self.report.last_turn_telemetry.reasoning_received = model_telemetry.reasoning_received;
        self.report.last_turn_telemetry.reasoning_replayed = model_telemetry.reasoning_replayed;
        self.report.last_turn_telemetry.reasoning_signature_replayed =
            model_telemetry.reasoning_signature_replayed;
        self.report.last_turn_telemetry.reasoning_fallback = model_telemetry.reasoning_fallback;
        self.report.last_turn_telemetry.refusal_source = model_telemetry.refusal_source;
        self.report.last_turn_telemetry.wire_audit = model_telemetry.wire_audit;
        self.report.last_turn_telemetry.requests = model_telemetry.requests;
        self.report.last_turn_telemetry.compaction = model_telemetry.compaction;
        self.report.last_turn_telemetry.phase_latencies = turn.phase_latencies.clone();
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
            && self.report.verify.passed()
            && !self.workspace.last_changed_files.is_empty()
            && self.subagents.auto_skills_written < super::super::MAX_AUTO_SKILLS_PER_SESSION
        {
            self.curate_turn_end(turn.turn_start, ui).await;
        }

        // Phase K: always-on (cheap, no model call) coding-fact extraction into
        // the decision log + project memory after a green file-changing turn.
        if self.report.verify.passed() && !self.workspace.last_changed_files.is_empty() {
            self.record_coding_facts_turn_end(ui);
        }

        // Surface the files this turn changed, so the user sees what was touched
        // without needing /diff. Skipped for read-only/Q&A turns (empty list).
        // Emitted BEFORE the finalize recap so the recap is the last text the
        // user sees (the "✓ done" marker follows it).
        if !self.workspace.last_changed_files.is_empty() {
            ui.changed_files(&self.workspace.last_changed_files);
        }

        // TurnPhase::Finalize — recap after mutating work, and as a close-out
        // when a tool-using turn produced no visible assistant text (live:
        // review stalled after tools, deferred recap never ran).
        self.set_turn_phase(TurnPhase::Finalize);
        let finalize_started = std::time::Instant::now();
        let needs_closeout = !super::finalize::turn_has_visible_assistant_text(
            self.messages.as_slice(),
            turn.turn_start,
        );
        let mut closeout_generated = false;
        if self.config.memory.finalize
            && turn.flags.made_tool_call
            && !turn.flags.ended_at_deadline
            && (needs_closeout
                || (!turn.flags.ended_at_cap
                    && !turn.flags.stalled_unfinished
                    && !turn.flags.stalled_repeating
                    && !self.workspace.last_changed_files.is_empty()
                    && turn.steps < turn.max_steps))
        {
            // Side questions may still be streaming — wait so their UI/usage land
            // before we close the turn, then disarm so idle `/btw` can't fire.
            self.join_btw_jobs(ui).await;
            self.disarm_btw_dispatcher();
            closeout_generated = self.finalize_turn(turn.turn_start, ui).await;
            // finalize_turn appended a [user: finalize-nudge][assistant: recap]
            // pair. Strip it from the persisted transcript so the FINALIZE_PROMPT
            // ("don't take any further action") doesn't bleed into the next turn
            // and make the model emit summary text instead of executing the new
            // prompt. The recap was already shown to the user via the UI.
            self.messages.strip_finalize_pair();
        }
        if needs_closeout && turn.flags.made_tool_call && !closeout_generated {
            turn.flags.stalled_unfinished = true;
            ui.status("turn ended without a user-facing answer");
        }
        turn.phase_latencies.finalize_ms = turn
            .phase_latencies
            .finalize_ms
            .saturating_add(finalize_started.elapsed().as_millis() as u64);

        // Tool-free curation/finalization calls and external editors can take
        // time after the first final reconciliation. Reconcile once more before
        // any long-horizon progress or typed outcome is committed.
        self.reconcile_workspace_changes().await?;
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
                match self.report.verify.bound_revision_digest() {
                    Some((revision, _)) => ledger.changes_since(revision),
                    None => settled_changes.clone(),
                }
            };
            let review_was_passed = turn.independent_review_status == ReviewStatus::Passed;
            super::settlement::reconcile_verified_revision(
                &mut self.report.verify,
                &mut turn.independent_review_status,
                settled_revision,
                settled_digest.clone(),
                &delta,
                ui,
            );
            if review_was_passed && turn.independent_review_status == ReviewStatus::Unavailable {
                turn.review_unavailable_reason =
                    Some("a workspace change after the review pass invalidated it".into());
            }
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
                    verified_at: self.report.verify.bound_revision_digest().as_ref(),
                    turn_ledger_revision: turn.turn_ledger_revision,
                    verification_infrastructure_error: turn.verification_infrastructure_error,
                },
                ui,
            )
            .await;
        if goal_invalidated_verification {
            self.report.verify.clear();
            if turn.independent_review_status == ReviewStatus::Passed {
                turn.independent_review_status = ReviewStatus::Unavailable;
                turn.review_unavailable_reason =
                    Some("goal settlement invalidated the review pass".into());
            }
        }
        // Budget check, after this turn's outcome is recorded so the report
        // reflects it. An objective with no reachable end state ("fully build
        // this" against a multi-phase plan) otherwise runs until someone
        // notices; a spent budget turns that into a stop with an account of
        // where it got to. Progress is intact — resuming continues from here.
        let budget_spent = self.goals.structured.as_ref().is_some_and(|goal| {
            goal.budget_exhausted()
                && goal.status == crate::goal::GoalStatus::Active
                && !goal.is_paused()
        });
        if budget_spent {
            let (report, spent, auto) = {
                let goal = self
                    .goals
                    .structured
                    .as_mut()
                    .expect("checked Some immediately above");
                let report = goal.progress_report();
                let spent = goal.turns_spent;
                let auto = goal.budget_auto;
                goal.pause(crate::goal::GoalPauseReason::Budget);
                (report, spent, auto)
            };
            // An automatic ceiling is a check-in, not a limit the user chose —
            // say so, or it reads as though they set something and forgot.
            let preamble = if auto {
                format!(
                    "⏸ automatic turn budget reached ({spent} turns) — stopping to check in rather than running on unattended."
                )
            } else {
                format!("⏸ turn budget spent ({spent} turns) — pausing with progress intact.")
            };
            ui.status(&format!(
                "{preamble}\n{report}\n`/goal budget <n>` to set your own, then `/goal resume`."
            ));
            self.refresh_system_message();
            self.persist_goal(ui);
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
        self.reconcile_workspace_changes().await?;
        let (outcome_revision, outcome_digest) = {
            let mut ledger = self.runtime.ledger();
            (ledger.revision(), ledger.workspace_revision())
        };
        let changed_after_final_hooks = self.report.verify.passed()
            && self
                .report
                .verify
                .bound_revision_digest()
                .is_none_or(|(revision, digest)| {
                    revision != outcome_revision || digest != outcome_digest
                });
        if changed_after_final_hooks {
            let delta = {
                let ledger = self.runtime.ledger();
                match self.report.verify.bound_revision_digest() {
                    Some((revision, _)) => ledger.changes_since(revision),
                    None => ledger.changes_since(turn.turn_ledger_revision),
                }
            };
            let review_was_passed = turn.independent_review_status == ReviewStatus::Passed;
            let wiped = super::settlement::reconcile_verified_revision_with_message(
                &mut self.report.verify,
                &mut turn.independent_review_status,
                outcome_revision,
                outcome_digest.clone(),
                &delta,
                ui,
                "workspace changed during turn finalization; the previous pass and goal progress were invalidated",
            );
            if wiped {
                if review_was_passed && turn.independent_review_status == ReviewStatus::Unavailable
                {
                    turn.review_unavailable_reason = Some(
                        "workspace changed during turn finalization; the review pass was invalidated"
                            .into(),
                    );
                }
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
                self.reconcile_workspace_changes().await?;
            }
        }
        let (final_changes, turn_had_mutation) = {
            let ledger = self.runtime.ledger();
            (
                ledger
                    .changes_since(turn.turn_ledger_revision)
                    .into_iter()
                    .filter(|change| !is_internal_runtime_artifact_path(&change.path))
                    .collect::<Vec<_>>(),
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
        // Carry the review-unavailable reason into telemetry. Merge, don't
        // overwrite: when the goal skeptic (not the independent review) was
        // the unavailable reviewer, it already wrote its reason directly.
        if let Some(reason) = &turn.review_unavailable_reason {
            self.report.last_turn_telemetry.review_unavailable_reason = Some(reason.clone());
        }
        // A read-only session may spend its final allowed round producing the
        // requested answer after inspection. That is a usable terminal result,
        // not unfinished workspace work: keep the cap as the diagnostic stop
        // reason while allowing the public outcome to be Completed. Mutation-
        // capable turns retain the stricter incomplete-at-cap contract.
        let accepted_read_only_cap_wrap_up = turn.flags.ended_at_cap
            && self.config.routing.tool_mode == ToolMode::ReadOnly
            && matches!(
                turn.progress_tracker.last_progress_reason.as_str(),
                "step-limit wrap-up report" | "tool-limit wrap-up report"
            )
            && !turn_had_mutation
            && !turn.flags.stalled_unfinished
            && !turn.flags.stalled_repeating;
        let classification_ended_at_cap =
            turn.flags.ended_at_cap && !accepted_read_only_cap_wrap_up;
        let (status, verification, review, classified_stop_reason) =
            super::finalize::classify_turn_outcome(
                turn.verification_infrastructure_error,
                turn.verification_unstable,
                self.report.verify.as_bool(),
                &self.workspace.last_changed_files,
                turn_had_mutation,
                no_check_executed,
                turn.independent_review_status,
                self.report.last_turn_telemetry.skeptic_last_status,
                classification_ended_at_cap,
                turn.flags.ended_at_deadline,
                turn.flags.stalled_unfinished,
                turn.flags.stalled_repeating,
                self.config.gates.allow_unverified,
            );
        let stop_reason = if accepted_read_only_cap_wrap_up {
            TurnStopReason::StepLimit
        } else {
            classified_stop_reason
        };
        // Outer `run_turn` also stamps Done (covers `?` paths); keep the success path explicit.
        self.set_turn_phase(TurnPhase::Done);
        let outcome = TurnOutcome {
            status,
            verification,
            review,
            stop_reason,
            changed_files: self.workspace.last_changed_files.clone(),
            verified_workspace_revision: (verification == VerificationStatus::Passed)
                .then(|| self.report.verify.digest().map(str::to_owned))
                .flatten(),
            effective_route: effective_model_route(
                &self.config,
                turn.effective_fallback_route.as_deref(),
            ),
            review_same_model: self.skeptic_shares_session_model(),
            leftover: self.goals.leftover_work(),
            plan_leftover: self.goals.plan_leftover_work(),
        };
        self.report.set_outcome(outcome.clone());
        // Claude-style suggested next prompt: cheap ChatOnly side call after
        // settlement. Never mutates history/workspace; frontends show ghost text.
        if self.should_suggest_next_prompt(&outcome) {
            self.suggest_next_prompt(turn.turn_start, ui).await;
        }
        // Cancellation can arrive during late settlement (for example while a
        // best-effort suggestion call is in flight). Do not clear the rollback
        // baselines and return a normal outcome merely because the main
        // model/tool loop had already ended; the outer selector has promised
        // the caller that a cancellation which won its race owns this turn.
        if self
            .turn_cancellation
            .as_ref()
            .is_some_and(|cancellation| cancellation.is_cancelled())
        {
            return Err(TurnCancellationRequested.into());
        }
        // The cancellation check above is the commit boundary for diagnostics:
        // do not durably publish a normal outcome (or synchronize it remotely)
        // while a cancellable late suggestion is still in flight. Everything
        // below is synchronous, so the body returns without another await and
        // a later cancellation belongs to the next outer scheduling point.
        if let Some(session) = self.session.as_mut() {
            let _ = session.record_turn_outcome(
                &outcome,
                self.report
                    .last_turn_telemetry
                    .review_unavailable_reason
                    .as_deref(),
            );
        }
        // Automatic post-mortem intake: bad outcomes become findings-ledger
        // records so `hi metrics` surfaces failure patterns without anyone
        // spelunking raw transcripts. Best-effort by design.
        if crate::learning::outcome_warrants_finding(&outcome) {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            crate::learning::append_finding(
                self.runtime.state_root(),
                &crate::learning::Finding {
                    ts,
                    session_id: self.session.as_deref().and_then(crate::SessionSink::id),
                    turn: Some(self.turn_count),
                    status: outcome.status,
                    stop_reason: outcome.stop_reason,
                    verification: outcome.verification,
                    review: outcome.review,
                    review_unavailable_reason: self
                        .report
                        .last_turn_telemetry
                        .review_unavailable_reason
                        .clone(),
                    last_stall_reason: self.report.last_turn_telemetry.last_stall_reason.clone(),
                    changed_files: outcome.changed_files.len(),
                    model: outcome.effective_route.model.clone(),
                    hint_active: self.task.active_hint_shape.clone(),
                    failure_shape: crate::learning::tool_failure_shape(
                        &self.report.last_turn_telemetry.tool_timeline,
                    ),
                },
            );
        }
        self.workspace.clear_active_baselines();
        let _ = self.maybe_requeue_goal_second_pass();
        Ok(outcome)
    }
}
