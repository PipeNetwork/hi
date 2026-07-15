//! The `/loop trio` workflow: a bounded plan → execute → review loop for
//! medium-sized tasks that need more structure than a bare prompt but less
//! than a full `/goal`. Modeled on the trio extension's single-session
//! planner→executor→reviewer pattern, but built as a transient loop (no
//! persistent `Goal` struct, no sub-goal checklist, no compaction survival).
//!
//! ## How it works
//!
//! 1. **Plan** — one bounded chat-only call to `planner_model` (same as
//!    `/goal` decomposition) produces a lightweight plan string. If no
//!    planner is configured, the prompt itself is the plan.
//! 2. **Execute** — the session model runs a normal `run_turn` with the plan
//!    + prompt as input. Full tools are available (unlike trio's
//!    tool-restricted executor phase — we keep the existing tool surface).
//! 3. **Review** — one bounded chat-only call to `skeptic_model` (same
//!    fail-open gate as `/goal team`) reviews the turn's diff. `Approve`
//!    stops the loop; `Object` sends it back for another execute round with
//!    the objections visible.
//! 4. **Stop** — the loop ends when the reviewer approves, `max_rounds` is
//!    hit, or the executor's turn fails verification.
//!
//! Unlike `/goal team`, this is a **per-task** review (one review of the whole
//! execution), not a per-step review. Unlike `/goal`, there's no growing
//! checklist, no autonomous drive, and no session persistence — it's a
//! bounded loop that stops.

use std::sync::Arc;

use anyhow::Result;
use hi_ai::{ChatRequest, Content, Message, RequestProfile, StreamEvent, ToolMode};

use super::skeptic::SkepticVerdict;

/// The planner prompt for the trio workflow. Lighter than `/goal`'s
/// decomposition — it produces a single plan block (not a sub-goal list),
/// since the executor works from it in one shot.
const TRIO_PLANNER_PROMPT: &str = "You are the planner in a plan→execute→review workflow. \
Understand the task and produce a concise, concrete implementation plan for the executor. \
The plan should be a short numbered list of steps (3–8 items), each one sentence. \
Do not write code — just the plan. The executor will implement it with full tool access.";

/// The reviewer prompt for the trio workflow. Same fail-open bias as the
/// `/goal team` skeptic: approve unless there's a concrete defect.
const TRIO_REVIEW_PROMPT: &str = "You are the reviewer in a plan→execute→review workflow. \
You see the task, the plan, and the executor's diff. Your job is to decide whether the \
implementation accomplishes the task. Bias strongly toward APPROVE. Reply APPROVE on the \
first line if the diff plausibly accomplishes the task, even if imperfect. Reply OBJECT on \
the first line ONLY when there is a concrete, specific defect: a real bug, a broken \
safeguard, or a requirement left unhandled. Put one concrete defect per following line. \
When uncertain, APPROVE.";

impl crate::Agent {
    /// Run the trio planner side-call. Returns the plan text, or the prompt
    /// itself if no planner model is configured (graceful fallback). Books
    /// usage; records no history.
    pub async fn trio_plan(&mut self, prompt: &str) -> Result<String> {
        let Some(model) = self.config.planner_model.clone() else {
            // No planner — use the prompt as the plan (single-step).
            return Ok(prompt.to_string());
        };
        let request = ChatRequest {
            model,
            messages: Arc::new(vec![
                Message::system(TRIO_PLANNER_PROMPT),
                Message::user(prompt.to_string()),
            ]),
            tools: Arc::new([]),
            max_tokens: 1024,
            temperature: self.config.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                compat: self.config.compat,
                tool_mode: ToolMode::ChatOnly,
                stream_usage: None,
            },
        };
        let mut text = String::new();
        let mut sink = |event: StreamEvent| {
            if let StreamEvent::Text(t) = event {
                text.push_str(&t);
            }
        };
        let completion = match self.provider.stream(request, &mut sink).await {
            Ok(c) => c,
            Err(err) => {
                self.add_side_error_usage(&err);
                // Planner failure is non-fatal — fall back to the prompt.
                return Ok(prompt.to_string());
            }
        };
        self.add_side_usage(completion.usage);
        if text.trim().is_empty() {
            text = content_text(&completion.content);
        }
        if text.trim().is_empty() {
            return Ok(prompt.to_string());
        }
        Ok(text)
    }

    /// Run the trio reviewer side-call. Returns the verdict (Approve/Object/
    /// Unavailable). Uses `skeptic_model` if configured, else the session
    /// model. Fail-open: errors → Approve. Books usage; records no history.
    pub async fn trio_review(
        &mut self,
        prompt: &str,
        plan: &str,
    ) -> SkepticVerdict {
        let model = self
            .config
            .skeptic_model
            .clone()
            .unwrap_or_else(|| self.config.model.clone());

        // Build the review context: task + plan + diff.
        let diff = self.turn_diff().await;
        let diff = if diff.len() > 6_000 {
            let mut end = 6_000;
            while !diff.is_char_boundary(end) {
                end -= 1;
            }
            let mut d = diff[..end].to_string();
            d.push_str("\n… (diff truncated)");
            d
        } else {
            diff
        };
        let context = format!(
            "Task: {prompt}\n\n\
             Plan:\n{plan}\n\n\
             Diff of the executor's changes:\n{diff}"
        );

        let request = ChatRequest {
            model,
            messages: Arc::new(vec![
                Message::system(TRIO_REVIEW_PROMPT),
                Message::user(context),
            ]),
            tools: Arc::new([]),
            max_tokens: 1024,
            temperature: self.config.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                compat: self.config.compat,
                tool_mode: ToolMode::ChatOnly,
                stream_usage: None,
            },
        };
        let mut text = String::new();
        let mut sink = |event: StreamEvent| {
            if let StreamEvent::Text(t) = event {
                text.push_str(&t);
            }
        };
        let completion = match self.provider.stream(request, &mut sink).await {
            Ok(c) => c,
            Err(err) => {
                self.add_side_error_usage(&err);
                // Fail-open: a reviewer error can't wedge the loop.
                return SkepticVerdict::Unavailable(format!("reviewer error: {err:#}"));
            }
        };
        self.add_side_usage(completion.usage);
        if text.trim().is_empty() {
            text = content_text(&completion.content);
        }
        parse_trio_verdict(&text)
    }
}

/// Extract the verdict from the reviewer's text output. Mirrors the skeptic
/// gate's `parse_verdict` but is kept local to avoid coupling trio to the
/// skeptic module's internals.
fn parse_trio_verdict(text: &str) -> SkepticVerdict {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let Some(first) = lines.first() else {
        return SkepticVerdict::Unavailable("reviewer returned empty output".into());
    };
    let first_clean = first.trim_matches(|c: char| matches!(c, '#' | '*' | '`' | '-' | '•' | ' '));
    let lower = first_clean.to_ascii_lowercase();
    if lower.starts_with("approve") {
        return SkepticVerdict::Approve;
    }
    if !(lower.starts_with("object") || lower.starts_with("reject")) {
        return SkepticVerdict::Unavailable("reviewer output did not contain a verdict".into());
    }
    let objs: Vec<String> = lines[1..]
        .iter()
        .map(|l| l.trim_matches(|c: char| matches!(c, '-' | '•' | '*' | ' ')))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    if objs.is_empty() {
        SkepticVerdict::Unavailable("reviewer objected without an actionable reason".into())
    } else {
        SkepticVerdict::Object(objs)
    }
}

/// Extract text from a completion's content array (same as plan_goal's helper).
fn content_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_trio_verdict_approve() {
        assert_eq!(
            parse_trio_verdict("APPROVE\nlooks good"),
            SkepticVerdict::Approve
        );
        assert_eq!(
            parse_trio_verdict("**APPROVE**"),
            SkepticVerdict::Approve
        );
    }

    #[test]
    fn parse_trio_verdict_object_with_reasons() {
        match parse_trio_verdict("OBJECT\nmissing test for foo\noff-by-one in bar()") {
            SkepticVerdict::Object(objs) => {
                assert_eq!(objs.len(), 2);
                assert!(objs[0].contains("missing test"));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn parse_trio_verdict_object_no_reasons_is_unavailable() {
        assert!(matches!(
            parse_trio_verdict("OBJECT"),
            SkepticVerdict::Unavailable(_)
        ));
    }

    #[test]
    fn parse_trio_verdict_empty_is_unavailable() {
        assert!(matches!(
            parse_trio_verdict(""),
            SkepticVerdict::Unavailable(_)
        ));
    }

    #[test]
    fn parse_trio_verdict_garbage_is_unavailable() {
        assert!(matches!(
            parse_trio_verdict("I think the code is fine"),
            SkepticVerdict::Unavailable(_)
        ));
    }
}
