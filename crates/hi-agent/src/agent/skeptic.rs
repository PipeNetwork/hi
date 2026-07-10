//! The `/goal team` skeptic gate: a bounded second-model review of a turn before
//! it advances a sub-goal. Modeled on the planner side-call ([`decompose_goal`]):
//! a throwaway chat-only request through `self.provider` at `skeptic_model`, usage
//! booked, no history recorded. The gate is **fail-open** — any error or
//! unparseable reply approves, so a flaky reviewer can only *catch* problems,
//! never wedge a goal.
//!
//! [`decompose_goal`]: crate::Agent::decompose_goal

use std::sync::Arc;

use hi_ai::{ChatRequest, Content, Message, RequestProfile, StreamEvent, ToolMode};

/// How much of the turn diff to show the skeptic (chars) — enough context without
/// blowing the bounded call's budget.
const SKEPTIC_DIFF_BUDGET: usize = 6_000;

const SKEPTIC_PROMPT: &str = "You are a code reviewer acting as a merge gate for a coding agent. \
You see the objective, the active sub-goal, the agent's verify result, and the diff it just \
produced. Your ONLY job is to block a change that fails to accomplish the active sub-goal — not to \
improve it or hold it to a higher standard. Bias strongly toward APPROVE. Reply APPROVE on the \
first line if the diff plausibly accomplishes the sub-goal, even if it is imperfect, could be more \
robust, lacks tests, or you cannot fully confirm it from the diff alone. Reply OBJECT on the first \
line ONLY when the diff has a concrete, specific defect that means the sub-goal is genuinely NOT \
accomplished: a real bug, a removed or broken safeguard, a case the sub-goal explicitly requires \
left unhandled, or a change that does the opposite of the sub-goal. Do NOT object over style, \
naming, missing tests (unless the sub-goal demands them), speculative edge cases, or anything you \
merely cannot verify from the diff. When uncertain, APPROVE — a wrong objection wastes a real \
retry. After OBJECT, put one concrete defect per line.";

/// The skeptic's verdict on whether the active sub-goal may advance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SkepticVerdict {
    /// Advance the sub-goal.
    Approve,
    /// Send it back to retry, carrying these concrete objections (fed into the
    /// sub-goal's notes so the next turn sees them).
    Object(Vec<String>),
}

impl crate::Agent {
    /// Run the skeptic gate against `sub_goal` (the sub-goal that was active at
    /// turn start — the current one may already be marked done via update_plan).
    /// Returns the joined objections if the skeptic wants the work retried, or
    /// `None` to let it stand (approve). Fail-open: no objection, a provider error,
    /// or an unparseable reply all return `None`. Books usage; records no history.
    pub(crate) async fn skeptic_gate(&mut self, objective: &str, sub_goal: &str) -> Option<String> {
        let context = self.skeptic_context(objective, sub_goal).await;
        match self.skeptic_review(&context).await {
            SkepticVerdict::Object(objs) if !objs.is_empty() => Some(objs.join("\n")),
            _ => None,
        }
    }

    /// Review an arbitrary `(objective, sub_goal, diff)` with the real skeptic —
    /// for offline *detector* evaluation of the reviewer (precision/recall on
    /// labeled diffs), independent of a live goal. Returns `(objected, objections)`.
    /// Uses the same prompt, model (`skeptic_model`), and fail-open behaviour as
    /// the gate; records no history.
    pub async fn review_diff(
        &mut self,
        objective: &str,
        sub_goal: &str,
        diff: &str,
    ) -> (bool, Vec<String>) {
        let mut diff = diff.to_string();
        if diff.len() > SKEPTIC_DIFF_BUDGET {
            let mut end = SKEPTIC_DIFF_BUDGET;
            while !diff.is_char_boundary(end) {
                end -= 1;
            }
            diff.truncate(end);
            diff.push_str("\n… (diff truncated)");
        }
        // Mirror the gate's context format so the reviewer sees the same shape.
        let context = format!(
            "Objective: {objective}\n\n\
             Active sub-goal (the one about to be marked done): {sub_goal}\n\n\
             verify result: (none configured)\n\
             Files changed this turn: (see diff)\n\n\
             Diff of this turn's changes:\n{diff}"
        );
        match self.skeptic_review(&context).await {
            SkepticVerdict::Object(objs) => (true, objs),
            SkepticVerdict::Approve => (false, Vec::new()),
        }
    }

    /// Assemble the review blob: objective + active sub-goal + verify result +
    /// changed files + a best-effort diff of this turn's changes (truncated).
    async fn skeptic_context(&self, objective: &str, sub_goal: &str) -> String {
        let verify = match self.last_verify {
            Some(true) => "verify result: PASSED",
            Some(false) => "verify result: FAILED",
            None => "verify result: (none configured)",
        };
        let files = if self.last_changed_files.is_empty() {
            "(none detected)".to_string()
        } else {
            self.last_changed_files.join(", ")
        };
        let mut diff = self.turn_diff().await;
        if diff.len() > SKEPTIC_DIFF_BUDGET {
            // Truncate on a char boundary so the format! below never panics.
            let mut end = SKEPTIC_DIFF_BUDGET;
            while !diff.is_char_boundary(end) {
                end -= 1;
            }
            diff.truncate(end);
            diff.push_str("\n… (diff truncated)");
        }
        format!(
            "Objective: {objective}\n\n\
             Active sub-goal (the one about to be marked done): {sub_goal}\n\n\
             {verify}\n\
             Files changed this turn: {files}\n\n\
             Diff of this turn's changes:\n{diff}"
        )
    }

    /// One bounded critique call to the configured `skeptic_model`. Fail-open: no
    /// model configured, a provider error, or an empty/unparseable reply all yield
    /// [`SkepticVerdict::Approve`].
    async fn skeptic_review(&mut self, context: &str) -> SkepticVerdict {
        let Some(model) = self.config.skeptic_model.clone() else {
            return SkepticVerdict::Approve;
        };
        let request = ChatRequest {
            model,
            messages: Arc::new(vec![
                Message::system(SKEPTIC_PROMPT),
                Message::user(context),
            ]),
            tools: Arc::new([]), // review only — no tool use
            max_tokens: 1024,
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
                return SkepticVerdict::Approve; // fail-open on a provider error
            }
        };
        self.add_side_usage(completion.usage);
        if text.trim().is_empty() {
            text = content_text(&completion.content);
        }
        parse_verdict(&text)
    }

    /// A best-effort unified diff of this turn's changes (against the turn's
    /// pre-edit checkpoint). Empty when there's no checkpoint or git can't produce
    /// one — the gate then reviews the sub-goal + verify result without a diff.
    async fn turn_diff(&self) -> String {
        match self.checkpoints.last() {
            Some(target) => hi_tools::checkpoint::diff(std::path::Path::new("."), target)
                .await
                .unwrap_or_default(),
            None => String::new(),
        }
    }
}

/// Collect the text blocks of a completion (the no-stream fallback).
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

/// Parse the skeptic's reply into a verdict. The first non-empty line decides:
/// `OBJECT`/`REJECT` (case-insensitive, markdown-tolerant) → the remaining
/// non-empty lines (plus any inline text after the keyword) are the objections;
/// anything else (`APPROVE`, empty, garbage) → `Approve`. Fail-open by
/// construction: an `OBJECT` with nothing actionable also approves.
fn parse_verdict(text: &str) -> SkepticVerdict {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let Some(first) = lines.first() else {
        return SkepticVerdict::Approve;
    };
    // Drop surrounding markdown emphasis/bullets on the verdict line.
    let first_clean = first.trim_matches(|c: char| matches!(c, '#' | '*' | '`' | '-' | '•' | ' '));
    let lower = first_clean.to_ascii_lowercase();
    if !(lower.starts_with("object") || lower.starts_with("reject")) {
        return SkepticVerdict::Approve;
    }
    // Objections: subsequent non-empty lines (bullets stripped) …
    let mut objs: Vec<String> = lines[1..]
        .iter()
        .map(|l| strip_bullet(l))
        .filter(|s| !s.is_empty())
        .collect();
    // … plus any inline objection after the keyword on the verdict line itself
    // (e.g. "OBJECT: the loop is off by one"). The leading keyword is ASCII, so
    // the byte index is a valid char boundary.
    let alpha_end = first_clean
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(first_clean.len());
    let inline =
        first_clean[alpha_end..].trim_matches(|c: char| matches!(c, ':' | '-' | '—' | '.' | ' '));
    if !inline.is_empty() {
        objs.insert(0, inline.to_string());
    }
    if objs.is_empty() {
        SkepticVerdict::Approve // OBJECT with nothing actionable → fail-open
    } else {
        SkepticVerdict::Object(objs)
    }
}

/// Strip a leading `-`/`*`/`•` bullet from an objection line.
fn strip_bullet(line: &str) -> String {
    let s = line.trim();
    s.strip_prefix(['-', '*', '•'])
        .map(|r| r.trim_start())
        .unwrap_or(s)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approve_variants() {
        assert_eq!(parse_verdict("APPROVE"), SkepticVerdict::Approve);
        assert_eq!(
            parse_verdict("  approve — looks correct\n"),
            SkepticVerdict::Approve
        );
        assert_eq!(parse_verdict("**APPROVE**"), SkepticVerdict::Approve);
        // Empty / garbage → fail-open approve.
        assert_eq!(parse_verdict("   \n\n"), SkepticVerdict::Approve);
        assert_eq!(parse_verdict("hmm, not sure"), SkepticVerdict::Approve);
    }

    #[test]
    fn object_with_listed_objections() {
        let v = parse_verdict("OBJECT\n- the loop is off by one\n- no test for the empty case\n");
        assert_eq!(
            v,
            SkepticVerdict::Object(vec![
                "the loop is off by one".to_string(),
                "no test for the empty case".to_string(),
            ])
        );
    }

    #[test]
    fn object_inline_objection() {
        // Objection on the verdict line after a separator.
        assert_eq!(
            parse_verdict("OBJECT: the sub-goal isn't actually satisfied"),
            SkepticVerdict::Object(vec!["the sub-goal isn't actually satisfied".to_string()])
        );
        // Markdown-wrapped keyword + a following bullet line.
        assert_eq!(
            parse_verdict("**OBJECT**\n* missing error handling"),
            SkepticVerdict::Object(vec!["missing error handling".to_string()])
        );
    }

    #[test]
    fn object_without_anything_actionable_is_fail_open() {
        // OBJECT with no objections to feed back → approve (nothing to retry on).
        assert_eq!(parse_verdict("OBJECT"), SkepticVerdict::Approve);
        assert_eq!(parse_verdict("OBJECT\n\n"), SkepticVerdict::Approve);
    }
}
