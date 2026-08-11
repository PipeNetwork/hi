//! Post-turn "suggested next prompt" side call (Claude Code–style ghost text).
//!
//! A short ChatOnly request predicts one natural follow-up user prompt. The
//! result is emitted through [`Ui::suggested_prompt`] and never appended to the
//! conversation history.

use std::sync::Arc;

use hi_ai::{ChatRequest, Content, Message, RequestProfile, StreamEvent, ToolMode};

use crate::Ui;

/// Instruction for the throwaway suggest call. Kept terse so weak models still
/// return a single usable line (or `NONE`).
const SUGGEST_NEXT_PROMPT: &str = "Suggest ONE short next user prompt that continues this coding \
session naturally. Base it on what just happened in this turn.\n\
Rules:\n\
- Reply with exactly one line of prompt text the user could type next.\n\
- No quotes, numbering, markdown, or preamble.\n\
- Prefer a concrete follow-up (run tests, commit, fix remaining issue, open a PR, document, etc.).\n\
- If nothing useful comes to mind, reply with exactly: NONE";

/// Soft cap so a rambling model can't dump a paragraph into the input ghost.
const MAX_SUGGESTION_CHARS: usize = 160;

impl crate::Agent {
    /// Whether this settled turn should attempt a next-prompt suggestion.
    pub(super) fn should_suggest_next_prompt(&self, outcome: &crate::TurnOutcome) -> bool {
        if !self.config.memory.suggest_next_prompt {
            return false;
        }
        if self.config.subagents.is_subagent {
            return false;
        }
        if self.plan_mode {
            return false;
        }
        if self
            .goals
            .structured
            .as_ref()
            .is_some_and(crate::goal::Goal::should_auto_drive)
        {
            return false;
        }
        if std::env::var("HI_SUGGEST_NEXT_PROMPT").is_ok_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no" | "disable" | "disabled"
            )
        }) {
            return false;
        }
        matches!(
            outcome.status,
            crate::TurnStatus::Completed | crate::TurnStatus::Incomplete
        ) && !matches!(
            outcome.stop_reason,
            crate::TurnStopReason::Cancelled | crate::TurnStopReason::InfrastructureFailure
        )
    }

    /// Predict a single next user prompt from this turn's messages. Best-effort:
    /// failures are silent (no status spam); success emits [`Ui::suggested_prompt`].
    pub(super) async fn suggest_next_prompt(&mut self, turn_start: usize, ui: &mut dyn Ui) {
        let turn = &self.messages.as_slice()[turn_start..];
        if turn.is_empty() {
            return;
        }
        let mut messages = Vec::with_capacity(turn.len() + 2);
        messages.push(self.minimal_system_message());
        messages.extend_from_slice(turn);
        // Ground the suggest call with files touched this turn when available.
        if !self.workspace.last_changed_files.is_empty() {
            let list = self.workspace.last_changed_files.join(", ");
            messages.push(Message::user(format!("Files changed this turn: {list}")));
        }
        messages.push(Message::user(SUGGEST_NEXT_PROMPT));

        let request = ChatRequest {
            model: self.config.routing.model.clone(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: Arc::from(messages),
            tools: Arc::new([]),
            max_tokens: 64,
            temperature: self.config.routing.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                compat: self.config.routing.compat,
                tool_mode: ToolMode::ChatOnly,
                stream_usage: None,
                deepseek_compat: self.config.routing.deepseek_compat,
                deepseek_strict: None,
                deepseek_thinking: None,
                output_token_parameter: self.config.routing.output_token_parameter,
            },
        };

        let mut raw = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => raw.push_str(&text),
            StreamEvent::Status(_) | StreamEvent::Reasoning(_) | StreamEvent::WireAudit(_) => {}
        };
        let completion = match self.provider.stream(request, &mut sink).await {
            Ok(completion) => completion,
            Err(err) => {
                self.add_side_error_usage(&err);
                return;
            }
        };
        self.add_side_usage(completion.usage);

        if raw.trim().is_empty() {
            for c in &completion.content {
                if let Content::Text(t) = c {
                    raw.push_str(t);
                }
            }
        }

        if let Some(suggestion) = sanitize_suggestion(&raw) {
            // Suppress an identical back-to-back repeat: the suggest call is
            // grounded only in the current turn, so a stalled session makes the
            // model propose the same follow-up every turn, which users read as
            // duplicated ghost text. A *different* suggestion still replaces it.
            if is_repeat_suggestion(self.last_suggested_prompt.as_deref(), &suggestion) {
                return;
            }
            self.last_suggested_prompt = Some(suggestion.clone());
            ui.suggested_prompt(&suggestion);
        }
    }
}

/// True when `candidate` is the same suggestion as the one already shown
/// (`last`), so the UI should not be handed a duplicate ghost. Comparison is on
/// the trimmed, case-insensitive text so trivial whitespace/case wobble from the
/// model doesn't defeat the dedup.
pub(crate) fn is_repeat_suggestion(last: Option<&str>, candidate: &str) -> bool {
    let Some(last) = last else { return false };
    last.trim().eq_ignore_ascii_case(candidate.trim())
}

/// Normalize a model reply into a single ghost-text prompt, or `None` when the
/// model declined / produced garbage.
pub(crate) fn sanitize_suggestion(raw: &str) -> Option<String> {
    let line = raw
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .trim_matches(|c| c == '"' || c == '\'' || c == '`')
        .trim();
    // Strip common list/bullet prefixes weak models emit despite instructions.
    let line = line
        .strip_prefix("- ")
        .or_else(|| line.strip_prefix("* "))
        .or_else(|| line.strip_prefix("1. "))
        .or_else(|| line.strip_prefix("1) "))
        .unwrap_or(line)
        .trim();
    if line.is_empty() {
        return None;
    }
    if line.eq_ignore_ascii_case("none")
        || line.eq_ignore_ascii_case("n/a")
        || line.eq_ignore_ascii_case("na")
    {
        return None;
    }
    // Reject replies that look like the model narrating instead of prompting.
    let lower = line.to_ascii_lowercase();
    if lower.starts_with("i suggest")
        || lower.starts_with("you could")
        || lower.starts_with("you should")
        || lower.starts_with("the user")
        || lower.starts_with("suggested")
    {
        return None;
    }
    let clipped: String = line.chars().take(MAX_SUGGESTION_CHARS).collect();
    let clipped = clipped.trim().to_string();
    if clipped.is_empty() {
        None
    } else {
        Some(clipped)
    }
}

#[cfg(test)]
mod tests {
    use super::sanitize_suggestion;

    #[test]
    fn sanitize_accepts_plain_prompt() {
        assert_eq!(
            sanitize_suggestion("Run the unit tests for auth"),
            Some("Run the unit tests for auth".into())
        );
    }

    #[test]
    fn sanitize_rejects_none_and_narration() {
        assert_eq!(sanitize_suggestion("NONE"), None);
        assert_eq!(sanitize_suggestion("I suggest writing tests"), None);
        assert_eq!(sanitize_suggestion(""), None);
    }

    #[test]
    fn sanitize_strips_quotes_and_bullets() {
        assert_eq!(
            sanitize_suggestion("\"commit these changes\""),
            Some("commit these changes".into())
        );
        assert_eq!(sanitize_suggestion("- open a PR"), Some("open a PR".into()));
    }

    #[test]
    fn repeat_suggestion_detected_case_and_whitespace_insensitively() {
        use super::is_repeat_suggestion;
        assert!(is_repeat_suggestion(
            Some("Run the unit tests"),
            "Run the unit tests"
        ));
        assert!(is_repeat_suggestion(
            Some("Run the unit tests"),
            "  run the unit tests  "
        ));
        assert!(!is_repeat_suggestion(Some("Run the unit tests"), "Open a PR"));
        assert!(!is_repeat_suggestion(None, "Run the unit tests"));
    }
}
