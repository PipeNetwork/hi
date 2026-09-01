//! Read-only `explore` subagent: delegate a bounded investigation to a child
//! agent that shares the parent's provider (via `Arc`) but runs with read-only
//! tools, its own fresh context, and a small step budget, then returns a concise
//! answer to the parent.
//!
//! Depth is capped at 1: the child is built with `explore_subagents = false`, and
//! because it runs in `ToolMode::ReadOnly` it never sees the (deliberately
//! non-read-only) `explore` tool — so a subagent cannot spawn another.

use hi_ai::ToolMode;
use serde_json::Value;

use crate::AgentConfig;
use crate::Ui;

pub(crate) fn explore_tool_outcome(
    content: impl Into<String>,
    status: hi_tools::ToolStatus,
) -> hi_tools::ToolOutcome {
    let (content, truncation) = hi_tools::bound_tool_content(content.into());
    hi_tools::ToolOutcome {
        content,
        display: None,
        plan: None,
        status,
        process: None,
        background: None,
        effects: hi_tools::ToolEffects::default(),
        truncation,
        images: Vec::new(),
    }
}

/// Cap on `explore` subagents per turn, to bound cost if the model
/// over-delegates within one task. Refilled every turn ([`crate::domain::SubagentSessionState::begin_turn`])
/// so long sessions never starve of exploration.
pub(crate) const MAX_EXPLORE_SUBAGENTS_PER_TURN: u32 = 8;

/// Per-round tool fan-out for one child explore turn. Children carry no step
/// ceiling: like the parent loop, they end via the repeat/no-progress/stall
/// budgets, so a hard cap can only truncate work that was still progressing
/// (live sessions showed capped children returning partial answers as
/// failures).
const EXPLORE_MAX_PARALLEL_TOOLS: usize = 4;

/// Maximum number of explore subagents to run concurrently within a single
/// tool batch. The turn budget is eight, so one batch can consume the whole
/// budget without waiting for a second wave.
pub(crate) const MAX_PARALLEL_EXPLORES: usize = 8;

/// A prepared-but-not-yet-running explore subagent job. Extracted from the
/// parent `Agent` so the heavy work (child `run_turn`) can run concurrently
/// across multiple explores without holding `&mut self`.
pub(crate) struct ExploreJob {
    pub(crate) slot: u32,
    pub(crate) task: String,
    pub(crate) provider: std::sync::Arc<dyn hi_ai::Provider>,
    pub(crate) child_config: AgentConfig,
}

/// The result of running an explore job — the tool outcome plus the child's
/// token usage (to fold into the parent's totals) and the slot (for budget
/// release on failure).
pub(crate) struct ExploreResult {
    pub(crate) slot: u32,
    pub(crate) outcome: hi_tools::ToolOutcome,
    pub(crate) usage: hi_ai::Usage,
}
impl crate::Agent {
    /// Prepare an explore subagent job: check budget, build the child config,
    /// and extract the provider. Returns `None` if the budget is exhausted or
    /// the task is empty. The returned job owns everything it needs to run
    /// concurrently with other explore jobs.
    pub(crate) fn prepare_explore(&mut self, arguments: &str) -> Option<ExploreJob> {
        let task = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|v| v.get("task").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_default();
        if task.trim().is_empty() {
            return None;
        }
        let n = self
            .subagents
            .try_begin_explore(MAX_EXPLORE_SUBAGENTS_PER_TURN)?;
        let child_model = self.effective_explore_child_model();
        let child_project_context = self
            .config
            .memory
            .project_context
            .as_deref()
            .map(|context| {
                const MAX_CHILD_CONTEXT_CHARS: usize = 4_000;
                context
                    .chars()
                    .take(MAX_CHILD_CONTEXT_CHARS)
                    .collect::<String>()
            });
        let child_config = AgentConfig {
            paths: crate::AgentPaths {
                workspace_root: self.runtime.root().to_path_buf(),
                state_root: self
                    .runtime
                    .state_root()
                    .join("subagents")
                    .join(format!("explore-{n}")),
            },
            routing: crate::AgentRouting {
                model: child_model,
                requested_max_tokens: self.config.routing.requested_max_tokens,
                max_tokens: self.config.routing.max_tokens,
                max_tokens_explicit: self.config.routing.max_tokens_explicit,
                temperature: self.config.routing.temperature,
                thinking_budget: self.config.routing.thinking_budget,
                reasoning_effort: self.config.routing.reasoning_effort,
                compat: self.config.routing.compat,
                deepseek_compat: self.config.routing.deepseek_compat,
                context_window: self.config.routing.context_window,
                tool_mode: ToolMode::ReadOnly,
                ..crate::AgentRouting::default()
            },
            gates: crate::AgentGates {
                verification: crate::VerificationMode::Disabled,
                read_only_preflight: false,
                lsp_mode: crate::LspMode::Off,
                ..crate::AgentGates::default()
            },
            loop_limits: crate::AgentLoopLimits {
                // Inherit the parent's step setting (off unless the operator
                // capped the session) — stall budgets end runaway children.
                max_steps: self.config.loop_limits.max_steps,
                max_parallel_tools: EXPLORE_MAX_PARALLEL_TOOLS,
                // A read-only explorer's text output IS its answer — don't nudge it to
                // keep going after it stops with text.
                max_silent_continues: 0,
                max_keep_working: 0,
                ..crate::AgentLoopLimits::default()
            },
            memory: crate::AgentMemory {
                project_context: child_project_context,
                finalize: false,
                curate_skills: false,
                suggest_next_prompt: false,
                ..crate::AgentMemory::default()
            },
            subagents: crate::AgentSubagents {
                explore_subagents: false,
                long_horizon: false,
                // Depth guard: a subagent is never advertised `explore`, so it can't
                // spawn another (depth ≤ 1), even in read-only mode.
                is_subagent: true,
                ..crate::AgentSubagents::default()
            },
            ..AgentConfig::default()
        };
        Some(ExploreJob {
            slot: n,
            task,
            provider: self.explore_child_provider(),
            child_config,
        })
    }

    /// The provider explore children run on. Shares the driver's connection
    /// unless an `explore_endpoint` is configured (team roles), in which case
    /// recon runs on its own OpenAI-compatible route — typically a local
    /// model, so read-heavy fan-out costs nothing.
    pub(crate) fn explore_child_provider(&self) -> std::sync::Arc<dyn hi_ai::Provider> {
        let stale = self.team_route_is_dead(
            self.config.subagents.explore_model.as_deref(),
            self.config.subagents.explore_endpoint.as_deref(),
        );
        routed_provider(
            (!stale)
                .then_some(self.config.subagents.explore_endpoint.as_deref())
                .flatten(),
            (!stale)
                .then_some(self.config.subagents.explore_endpoint_key.as_deref())
                .flatten(),
            &self.provider,
        )
    }

    /// The provider in-process background `delegate` tasks run on — the
    /// delegate route when configured, else the driver's provider. (The
    /// synchronous delegate path applies the same route in its child-process
    /// runner instead.)
    pub(crate) fn delegate_child_provider(&self) -> std::sync::Arc<dyn hi_ai::Provider> {
        let stale = self.team_route_is_dead(
            self.config.subagents.delegate_model.as_deref(),
            self.config.subagents.delegate_endpoint.as_deref(),
        );
        routed_provider(
            (!stale)
                .then_some(self.config.subagents.delegate_endpoint.as_deref())
                .flatten(),
            (!stale)
                .then_some(self.config.subagents.delegate_endpoint_key.as_deref())
                .flatten(),
            &self.provider,
        )
    }

    pub(crate) fn effective_explore_child_model(&self) -> String {
        let stale = self.team_route_is_dead(
            self.config.subagents.explore_model.as_deref(),
            self.config.subagents.explore_endpoint.as_deref(),
        );
        if stale {
            self.config.routing.model.clone()
        } else {
            explore_child_model(&self.config)
        }
    }

    /// Run one read-only `explore` subagent for the `{task}` argument and return
    /// its answer as the tool result. Best-effort: a provider/parse error becomes
    /// an error string fed back to the model, never fatal to the parent turn.
    ///
    /// This is the synchronous single-explore path. When multiple explore calls
    /// are ready in the same tool batch, the batch scheduler uses
    /// [`prepare_explore`] + [`run_explore_job`] + [`finish_explore`] to run
    /// them concurrently.
    pub(crate) async fn handle_explore(
        &mut self,
        arguments: &str,
        ui: &mut dyn Ui,
    ) -> hi_tools::ToolOutcome {
        let task = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|v| v.get("task").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_default();
        if task.trim().is_empty() {
            return explore_tool_outcome(
                "explore error: missing required \"task\" argument",
                hi_tools::ToolStatus::Failed,
            );
        }
        let Some(job) = self.prepare_explore(arguments) else {
            return explore_tool_outcome(
                format!(
                    "explore budget exhausted ({MAX_EXPLORE_SUBAGENTS_PER_TURN} subagents this \
                     turn); investigate directly for the rest of this turn."
                ),
                hi_tools::ToolStatus::Denied,
            );
        };
        let id = format!("explore-{}", job.slot);
        let summary = crate::clip_subagent_description(&task);
        let sink = ui.subagent_sink();
        ui.subagent_spawned(&id, "explore", &summary, false);
        let started = std::time::Instant::now();
        let result = if sink.is_some() {
            let mut child_ui = crate::subagent_progress::SubagentProgressUi {
                id: id.clone(),
                sink,
            };
            run_explore_job(job, &mut child_ui).await
        } else {
            let mut child_ui = crate::subagent_progress::SubagentParentUi {
                inner: crate::subagent_progress::SubagentProgressUi {
                    id: id.clone(),
                    sink: None,
                },
                parent: ui,
            };
            run_explore_job(job, &mut child_ui).await
        };
        // Fold the child's token usage into the parent's session totals.
        self.add_side_usage(result.usage);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let status = crate::subagent_finish_status(result.outcome.status);
        let finish_summary: String = result.outcome.content.chars().take(120).collect();
        ui.subagent_finished(&id, status, elapsed_ms, &finish_summary);
        result.outcome
    }

    /// Finish a completed explore job: fold usage into the parent's totals.
    /// Called after parallel explores complete in the batch scheduler.
    pub(crate) fn finish_explore(&mut self, result: ExploreResult) -> hi_tools::ToolOutcome {
        self.add_side_usage(result.usage);
        result.outcome
    }
}

/// Run a prepared explore job to completion. This is a free function (not a
/// method on `Agent`) so it can run concurrently across multiple jobs without
/// holding `&mut self`. Live status goes through [`crate::subagent_progress::SubagentProgressUi`].
pub(crate) async fn run_explore_job(job: ExploreJob, ui: &mut dyn Ui) -> ExploreResult {
    let ExploreJob {
        slot,
        task,
        provider,
        child_config,
    } = job;

    let mut child = match crate::Agent::new(provider, child_config) {
        Ok(child) => child,
        Err(error) => {
            return ExploreResult {
                slot,
                outcome: explore_tool_outcome(
                    format!("explore subagent runtime initialization failed: {error:#}"),
                    hi_tools::ToolStatus::Failed,
                ),
                usage: hi_ai::Usage::default(),
            };
        }
    };
    // `Box::pin` breaks the async-recursion cycle (`run_turn` → `handle_explore`
    // → child `run_turn`) that would otherwise make the future infinitely sized.
    let outcome = {
        match Box::pin(child.run_turn(&explore_child_prompt(&task), ui)).await {
            Ok(outcome) => {
                let answer = child.last_assistant_text();
                let mut status = match outcome.status {
                    crate::TurnStatus::Completed => hi_tools::ToolStatus::Succeeded,
                    crate::TurnStatus::Blocked => hi_tools::ToolStatus::Denied,
                    crate::TurnStatus::Cancelled => hi_tools::ToolStatus::Cancelled,
                    crate::TurnStatus::Incomplete | crate::TurnStatus::Failed => {
                        hi_tools::ToolStatus::Failed
                    }
                };
                let answer = answer.unwrap_or_else(|| {
                    status = hi_tools::ToolStatus::Failed;
                    "explore subagent produced no answer".to_string()
                });
                explore_tool_outcome(answer, status)
            }
            Err(err) => {
                // Nested escapes: typed fail cleanup (turn-scoped bg kill).
                let _ = child.cleanup_turn(crate::TurnCleanupKind::Fail).await;
                explore_tool_outcome(
                    format!("explore subagent error: {err}"),
                    hi_tools::ToolStatus::Failed,
                )
            }
        }
    };
    // Throwaway child runtime: full kill (local skeptic + any leftover bg).
    child.kill_background_processes();
    let usage = *child.totals();
    ExploreResult {
        slot,
        outcome,
        usage,
    }
}

const MAX_EXPLORE_TASK_CHARS: usize = 2_000;

fn explore_child_prompt(task: &str) -> String {
    // Deliberately plain phrasing: the child's read-only restriction and
    // inspection-sprawl cap come from its task contract and capability scope,
    // not legacy review-intent prompt shaping.
    let task = clip_chars(task.trim(), MAX_EXPLORE_TASK_CHARS);
    format!(
        "Answer this question about the codebase. Read and search the relevant files as needed, then \
         reply with a concise, self-contained answer that cites the specific files and locations \
         supporting it.\n\nQuestion: {task}"
    )
}

fn clip_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{clipped}…")
}

/// The model explore children run: `HI_EXPLORE_MODEL` env (highest, a live
/// escape hatch) → `subagents.explore_model` (team roles) → the driver model.
pub(crate) fn explore_child_model(config: &crate::AgentConfig) -> String {
    std::env::var("HI_EXPLORE_MODEL")
        .ok()
        .filter(|model| !model.trim().is_empty())
        .or_else(|| {
            config
                .subagents
                .explore_model
                .clone()
                .filter(|model| !model.trim().is_empty())
        })
        .unwrap_or_else(|| config.routing.model.clone())
}

/// Build the provider for a routed subagent role: a dedicated
/// OpenAI-compatible client when an endpoint override is set, else the
/// driver's shared provider. Construction is cheap (one HTTP client), so
/// routed children build per spawn rather than caching.
pub(crate) fn routed_provider(
    endpoint: Option<&str>,
    api_key: Option<&str>,
    parent: &std::sync::Arc<dyn hi_ai::Provider>,
) -> std::sync::Arc<dyn hi_ai::Provider> {
    match endpoint.map(str::trim).filter(|url| !url.is_empty()) {
        Some(url) => std::sync::Arc::new(hi_ai::OpenAiProvider::new(
            url.to_string(),
            api_key.unwrap_or_default().to_string(),
        )),
        None => parent.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explore_child_prompt_clips_a_huge_task() {
        let prompt = explore_child_prompt(&"TASK ".repeat(2_000));
        assert!(
            prompt.chars().count() < MAX_EXPLORE_TASK_CHARS + 400,
            "{}",
            prompt.chars().count()
        );
        assert!(!prompt.contains(&"TASK ".repeat(500)), "{prompt}");
    }

    #[test]
    fn explore_child_model_prefers_config_route_over_driver() {
        // Env is deliberately not exercised here (global, races other tests);
        // it keeps the highest precedence as a live escape hatch.
        let mut config = crate::AgentConfig::default();
        config.routing.model = "pipe/glm-5.2".into();
        assert_eq!(explore_child_model(&config), "pipe/glm-5.2", "inherits");
        config.subagents.explore_model = Some("qwen3-4b".into());
        assert_eq!(explore_child_model(&config), "qwen3-4b", "team route wins");
        config.subagents.explore_model = Some("  ".into());
        assert_eq!(
            explore_child_model(&config),
            "pipe/glm-5.2",
            "blank override is ignored"
        );
    }

    #[test]
    fn routed_provider_shares_the_driver_unless_an_endpoint_is_set() {
        let parent: std::sync::Arc<dyn hi_ai::Provider> = std::sync::Arc::new(
            hi_ai::OpenAiProvider::new("http://127.0.0.1:1/v1".into(), "k".into()),
        );
        let inherited = routed_provider(None, None, &parent);
        assert!(
            std::sync::Arc::ptr_eq(&parent, &inherited),
            "no endpoint → the driver's shared connection"
        );
        let routed = routed_provider(Some("http://127.0.0.1:18080/v1"), None, &parent);
        assert!(
            !std::sync::Arc::ptr_eq(&parent, &routed),
            "an endpoint override gets its own provider"
        );
        let blank = routed_provider(Some("   "), None, &parent);
        assert!(
            std::sync::Arc::ptr_eq(&parent, &blank),
            "blank endpoint is ignored"
        );
    }

    #[test]
    fn child_prompt_stays_plain_but_has_a_read_only_task_contract() {
        let prompt = explore_child_prompt("count the Rust source lines");
        assert!(crate::steering::classify_read_only_intent(&prompt).is_none());
        assert_eq!(
            crate::TaskContract::derive(&prompt, crate::VerificationMode::Disabled).intent,
            crate::TaskIntent::ReadOnly
        );
        assert_eq!(EXPLORE_MAX_PARALLEL_TOOLS, 4);
    }
}
