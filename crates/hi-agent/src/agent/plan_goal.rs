//! Goal decomposition: one bounded planner-model call that turns a `/goal`
//! objective into an ordered list of sub-tasks for the long-horizon engine to
//! drive. A strong planner (e.g. glm-5.2) plans once; the session model executes
//! each sub-goal turn-by-turn. Modeled on the other bounded side-calls
//! ([`Agent::update_memory_at`], MoA's `reference_guidance`): a throwaway
//! chat-only request through `self.provider`, usage booked, no history recorded.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use hi_ai::{ChatRequest, Content, Message, RequestProfile, StreamEvent, ToolMode};

/// Upper bound on sub-tasks accepted from the planner — bias toward few so a
/// small objective stays small and we never over-decompose.
const MAX_SUB_GOALS: usize = 6;

const PLANNER_PROMPT: &str = "You are a planning assistant for a coding agent. Decompose the \
user's coding objective into 2 to 6 ordered, independently-verifiable sub-tasks. Output one \
imperative sub-task per line — no numbering, no bullet characters, no prose, no preamble, no \
blank lines. If the objective is genuinely a single step, output exactly one line.";

impl crate::Agent {
    /// Decompose `objective` into ordered sub-task descriptions via one bounded
    /// call to the configured `planner_model`. Returns the parsed list; errors if
    /// no planner is configured, the call fails, or nothing usable comes back — the
    /// caller then falls back to a single sub-goal equal to the objective. Books the
    /// call's token usage; records nothing into the session history.
    pub async fn decompose_goal(&mut self, objective: &str) -> Result<Vec<String>> {
        let Some(model) = self.config.planner_model.clone() else {
            return Err(anyhow!("no planner model configured"));
        };
        let request = ChatRequest {
            model,
            messages: Arc::new(vec![
                Message::system(PLANNER_PROMPT),
                Message::user(objective),
            ]),
            tools: Arc::new([]), // planning — no tool use
            max_tokens: 512,     // throwaway call — a short list of sub-tasks
            temperature: self.config.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
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
            Ok(completion) => completion,
            Err(err) => {
                self.add_error_usage(&err);
                return Err(err);
            }
        };
        self.add_usage(completion.usage);
        // Fall back to the completion content if the provider returned text only in
        // the final object rather than via stream deltas.
        if text.trim().is_empty() {
            text = content_text(&completion.content);
        }

        let steps = parse_sub_goals(&text);
        if steps.is_empty() {
            return Err(anyhow!("planner returned no sub-tasks"));
        }
        Ok(steps)
    }
}

/// Collect the text blocks of a completion (used only as the no-stream fallback).
fn content_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            Content::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse the planner's line-per-task output into clean sub-goal descriptions:
/// trim, strip any leading list marker, drop empties, cap at [`MAX_SUB_GOALS`].
fn parse_sub_goals(text: &str) -> Vec<String> {
    text.lines()
        .map(strip_list_marker)
        .filter(|s| !s.is_empty())
        .take(MAX_SUB_GOALS)
        .collect()
}

/// Strip a leading list marker — `- ` / `* ` / `• ` or a `12.` / `12)` number —
/// that a model tends to add despite being told not to.
fn strip_list_marker(line: &str) -> String {
    let s = line.trim();
    // Bullet forms.
    if let Some(rest) = s.strip_prefix(['-', '*', '•']) {
        return rest.trim_start().to_string();
    }
    // Numbered forms: leading ASCII digits followed by `.` or `)`.
    let digits = s.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0 && digits < s.len() && matches!(s.as_bytes()[digits], b'.' | b')') {
        return s[digits + 1..].trim_start().to_string();
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_cleans_planner_output() {
        let raw = "1. Add the parser module\n2) Wire it into main\n- Add a test\n* Update docs\n";
        assert_eq!(
            parse_sub_goals(raw),
            vec![
                "Add the parser module",
                "Wire it into main",
                "Add a test",
                "Update docs",
            ]
        );
    }

    #[test]
    fn drops_blank_lines_and_bounds_to_six() {
        let raw = "a\n\n  \nb\nc\nd\ne\nf\ng\nh\n"; // 8 non-empty
        let out = parse_sub_goals(raw);
        assert_eq!(out.len(), MAX_SUB_GOALS);
        assert_eq!(out.first().map(String::as_str), Some("a"));
    }

    #[test]
    fn single_line_stays_one_step() {
        assert_eq!(
            parse_sub_goals("Fix the off-by-one in count()\n"),
            vec!["Fix the off-by-one in count()"]
        );
    }

    #[test]
    fn empty_output_yields_nothing() {
        assert!(parse_sub_goals("   \n\n").is_empty());
    }
}
