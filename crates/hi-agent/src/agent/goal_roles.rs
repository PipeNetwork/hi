//! Fail-open `/goal` roles: strategist (stuck runs) and summarizer (closing recap).
//!
//! Chat-only auxiliary calls, modeled on the planner/auditor. Never block
//! completion or advancement; a transport failure is ignored.

use std::sync::Arc;

use hi_ai::{ChatRequest, Content, Message, RequestProfile, StreamEvent};

use crate::Ui;
use crate::goal::clip_chars;

const STRATEGIST_EVERY: u32 = 3;
const STRATEGY_MAX_CHARS: usize = 800;
const SUMMARY_MAX_CHARS: usize = 1200;

const STRATEGIST_PROMPT: &str = "You are the Goal Strategist. The implementer has failed review \
several times in a row. Diagnose WHY the run is stuck and recommend ONE structural change to the \
HOW (not the WHAT). Do not change the objective or acceptance criteria. Reply with a short note: \
Diagnosis (1-3 sentences), then 2-4 mechanical next steps. No preamble.";

const SUMMARIZER_PROMPT: &str = "You are the Goal Summarizer. The goal is already complete. Write \
the closing message the user reads: one sentence naming WHAT was delivered, then HOW to use it \
(exact command or up to 3 bullets). At most 80 words. No preamble.";

impl crate::Agent {
    /// After a skeptic objection, maybe ask the strategist for a structural note.
    pub(crate) async fn maybe_run_goal_strategist(&mut self, ui: &mut dyn Ui) {
        let Some(goal) = self.goals.structured.as_ref() else {
            return;
        };
        if goal.skeptic_objections < goal.strategy_at.saturating_add(STRATEGIST_EVERY) {
            return;
        }
        let input = format!(
            "Objective: {}\nKind: {}\nConsecutive objections: {}\nLast gaps:\n{}\nActive: {}",
            goal.objective,
            goal.kind.as_str(),
            goal.skeptic_objections,
            if goal.last_gaps.is_empty() {
                "(none)"
            } else {
                goal.last_gaps.as_str()
            },
            goal.active_sub_goal()
                .map(|step| step.description.as_str())
                .unwrap_or("(none)"),
        );
        let objections = goal.skeptic_objections;
        match self.goal_role_call(STRATEGIST_PROMPT, &input).await {
            Ok(note) => {
                let note = clip_chars(note.trim(), STRATEGY_MAX_CHARS);
                if let Some(goal) = self.goals.structured.as_mut() {
                    goal.strategy_note = note.clone();
                    goal.strategy_at = objections;
                    goal.push_event("strategy", "strategist advised a restructure");
                }
                if !note.is_empty() {
                    ui.status(&format!(
                        "🧭 strategist: {}",
                        note.lines().next().unwrap_or("see strategy note")
                    ));
                }
            }
            Err(error) => {
                if let Some(goal) = self.goals.structured.as_mut() {
                    goal.strategy_at = objections;
                    goal.push_event("strategy", format!("unavailable ({error})"));
                }
            }
        }
    }

    /// After a successful completion audit, write a short closing recap.
    pub(crate) async fn maybe_run_goal_summarizer(&mut self, ui: &mut dyn Ui) {
        let Some(goal) = self.goals.structured.as_ref() else {
            return;
        };
        let input = format!(
            "Objective: {}\nKind: {}\nChecklist:\n{}",
            goal.objective,
            goal.kind.as_str(),
            goal.sub_goals
                .iter()
                .map(|step| {
                    let mark = match step.status {
                        crate::goal::GoalStatus::Done => "x",
                        crate::goal::GoalStatus::Active => ">",
                        crate::goal::GoalStatus::Failed => "!",
                        crate::goal::GoalStatus::Blocked => "-",
                        crate::goal::GoalStatus::Pending => " ",
                    };
                    format!("- [{mark}] {}", step.description)
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
        match self.goal_role_call(SUMMARIZER_PROMPT, &input).await {
            Ok(summary) => {
                let summary = clip_chars(summary.trim(), SUMMARY_MAX_CHARS);
                if let Some(goal) = self.goals.structured.as_mut() {
                    goal.closing_summary = summary.clone();
                    goal.push_event("summary", "closing recap written");
                }
                if !summary.is_empty() {
                    ui.status(&format!("📦 {summary}"));
                }
            }
            Err(error) => {
                if let Some(goal) = self.goals.structured.as_mut() {
                    goal.push_event("summary", format!("unavailable ({error})"));
                }
            }
        }
    }

    async fn goal_role_call(&mut self, system_prompt: &str, input: &str) -> anyhow::Result<String> {
        let model = self
            .config
            .subagents
            .planner_model
            .clone()
            .unwrap_or_else(|| self.effective_skeptic_model().to_string());
        let request_policy = self.seal_chat_only_auxiliary_request(&model, 1024).await;
        let request = ChatRequest {
            execution: self.request_execution(),
            model,
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: Arc::new(vec![
                Message::system(system_prompt.to_string()),
                Message::user(input.to_string()),
            ]),
            tools: request_policy.tools,
            tool_envelope: Some(request_policy.envelope),
            max_tokens: request_policy.max_tokens,
            temperature: self.config.routing.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                compat: self.config.routing.compat,
                tool_mode: request_policy.tool_mode,
                stream_usage: None,
                deepseek_compat: self.config.routing.deepseek_compat,
                deepseek_strict: None,
                deepseek_thinking: None,
                output_token_parameter: self.config.routing.output_token_parameter,
            },
        };
        let mut text = String::new();
        let mut sink = |event: StreamEvent| {
            if let StreamEvent::Text(chunk) = event {
                text.push_str(&chunk);
            }
        };
        let timeout = self.side_call_timeout();
        let completion = match crate::agent::turn::await_side_call(
            timeout,
            self.provider.stream(request, &mut sink),
        )
        .await
        {
            Err(timeout) => {
                anyhow::bail!("goal role timed out after {:.1}s", timeout.as_secs_f64());
            }
            Ok(Ok(completion)) => completion,
            Ok(Err(err)) => {
                self.add_side_error_usage(&err);
                return Err(err);
            }
        };
        self.add_side_usage(completion.usage);
        if text.trim().is_empty() {
            text = completion
                .content
                .iter()
                .filter_map(|block| match block {
                    Content::Text(body) => Some(body.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
        }
        Ok(text)
    }
}
