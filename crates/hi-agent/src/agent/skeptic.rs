//! The `/goal team` skeptic gate: a bounded second-model review of a turn before
//! it advances a sub-goal. Modeled on the planner side-call ([`decompose_goal`]):
//! a throwaway chat-only request through `self.provider` at the effective
//! skeptic model (`skeptic_model`, falling back to the session model so the
//! gate works unconfigured), usage booked, no history recorded. The gate is
//! **fail-open** — any error or unparseable reply approves, so a flaky
//! reviewer can only *catch* problems, never wedge a goal.
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
left unhandled, a change that does the opposite of the sub-goal, stub code standing in for \
behavior the sub-goal requires — todo!()/unimplemented!()/raise NotImplementedError or placeholder \
bodies where the sub-goal demands the real implementation; listed stub markers in the changed \
files are concrete evidence, not speculation — or the wrong artifact: when the sub-goal names a \
specific technology or file kind (a CUDA kernel, a Metal shader, a SQL schema) and the diff \
delivers a simulation or substitute in another language instead, the sub-goal is NOT \
accomplished. Do NOT object over style, \
naming, missing tests (unless the sub-goal demands them), speculative edge cases, or anything you \
merely cannot verify from the diff. When uncertain, APPROVE — a wrong objection wastes a real \
retry. After OBJECT, put one concrete defect per line.";

const INDEPENDENT_REVIEW_PROMPT: &str = "You are the independent completion reviewer for a coding \
agent. Review the task contract, scoped repository instructions, complete bounded diff, relevant \
context, and deterministic verification evidence. Reply APPROVE on the first line only when the \
change satisfies the stated acceptance contract without a concrete regression. Reply OBJECT on \
the first line when you find a specific correctness, security, compatibility, migration, or \
acceptance defect. Put one actionable defect per following line. Do not object over style or \
speculation; every objection must identify the affected behavior or file.";

/// The skeptic's verdict on whether the active sub-goal may advance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkepticVerdict {
    /// Advance the sub-goal.
    Approve,
    /// Send it back to retry, carrying these concrete objections (fed into the
    /// sub-goal's notes so the next turn sees them).
    Object(Vec<String>),
    /// Reviewer configuration, transport, or output could not yield a verdict.
    Unavailable(String),
}

impl crate::Agent {
    pub(crate) async fn independent_review(&mut self, context: &str) -> SkepticVerdict {
        let model = self.effective_skeptic_model().to_string();
        self.review_with_prompt(context, INDEPENDENT_REVIEW_PROMPT, model)
            .await
    }
    /// Run the skeptic gate against `sub_goal` (the sub-goal that was active at
    /// turn start — the current one may already be marked done via update_plan).
    /// Returns the joined objections if the skeptic wants the work retried, or
    /// `None` to let it stand (approve). Fail-open: no objection, a provider error,
    /// or an unparseable reply all return `None`. Books usage; records no history.
    pub(crate) async fn skeptic_gate(&mut self, objective: &str, sub_goal: &str) -> SkepticVerdict {
        let context = self.skeptic_context(objective, sub_goal).await;
        self.skeptic_review(&context).await
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
            SkepticVerdict::Unavailable(_) => (false, Vec::new()),
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
        let stub_findings =
            hi_tools::stub_scan::scan_paths(self.runtime.root(), &self.last_changed_files, 50);
        let stubs = if stub_findings.is_empty() {
            "(none detected)".to_string()
        } else {
            stub_findings
                .iter()
                .map(|f| format!("\n  {}:{}: {}", f.path, f.line, f.marker))
                .collect()
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
             Files changed this turn: {files}\n\
             Stub markers present in files changed this turn: {stubs}\n\n\
             Diff of this turn's changes:\n{diff}"
        )
    }

    /// One bounded critique call to the effective skeptic model —
    /// `skeptic_model` when configured, otherwise the session model, so the
    /// gate works with zero configuration. Fail-open: a provider error or an
    /// empty/unparseable reply approves.
    async fn skeptic_review(&mut self, context: &str) -> SkepticVerdict {
        let model = self.effective_skeptic_model().to_string();
        self.review_with_prompt(context, SKEPTIC_PROMPT, model)
            .await
    }

    async fn review_with_prompt(
        &mut self,
        context: &str,
        system_prompt: &str,
        model: String,
    ) -> SkepticVerdict {
        let request = ChatRequest {
            model,
            messages: Arc::new(vec![Message::system(system_prompt), Message::user(context)]),
            tools: Arc::new([]), // review only — no tool use
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

        // One bounded retry on a transient transport error (rate limit, brief
        // capacity/outage blip). A review that a single 429 could permanently
        // downgrade to "unavailable" is noise at the end of an otherwise-good
        // turn; anything persistent still reports unavailable after the retry.
        let mut attempts_left = 2u32;
        loop {
            attempts_left -= 1;
            let mut text = String::new();
            let mut sink = |event: StreamEvent| {
                if let StreamEvent::Text(t) = event {
                    text.push_str(&t);
                }
            };
            let completion = match self.provider.stream(request.clone(), &mut sink).await {
                Ok(completion) => completion,
                Err(err) => {
                    self.add_side_error_usage(&err);
                    if attempts_left > 0 && review_error_is_transient(&err) {
                        let delay = hi_ai::provider_retry_after_seconds(&err)
                            .unwrap_or(2)
                            .min(10);
                        tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                        continue;
                    }
                    return SkepticVerdict::Unavailable(format!("provider error: {err:#}"));
                }
            };
            self.add_side_usage(completion.usage);
            if text.trim().is_empty() {
                text = content_text(&completion.content);
            }
            return parse_verdict(&text);
        }
    }

    /// A best-effort unified diff of this turn's changes (against the turn's
    /// pre-edit checkpoint). Empty when there's no checkpoint or git can't produce
    /// one — the gate then reviews the sub-goal + verify result without a diff.
    pub(crate) async fn turn_diff(&self) -> String {
        match self.checkpoints.last() {
            Some(target) => hi_tools::checkpoint::diff_with_state(
                self.runtime.root(),
                target,
                self.runtime.state_root(),
            )
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
/// Transient transport errors worth one bounded retry before reporting the
/// review unavailable. Anything auth- or request-shape-related fails fast —
/// retrying cannot change those.
fn review_error_is_transient(err: &anyhow::Error) -> bool {
    use hi_ai::ProviderErrorKind as K;
    matches!(
        hi_ai::provider_error_kind(err),
        Some(
            K::RateLimit
                | K::CapacityUnavailable
                | K::Outage
                | K::MalformedStream
                | K::EmptyCompletion
        )
    ) || hi_ai::provider_route_error_is_retryable(err)
}

fn parse_verdict(text: &str) -> SkepticVerdict {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let Some(first) = lines.first() else {
        return SkepticVerdict::Unavailable("reviewer returned empty output".into());
    };
    // Drop surrounding markdown emphasis/bullets on the verdict line.
    let first_clean = first.trim_matches(|c: char| matches!(c, '#' | '*' | '`' | '-' | '•' | ' '));
    let lower = first_clean.to_ascii_lowercase();
    if lower.starts_with("approve") {
        return SkepticVerdict::Approve;
    }
    if !(lower.starts_with("object") || lower.starts_with("reject")) {
        return SkepticVerdict::Unavailable("reviewer output did not contain a verdict".into());
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
        SkepticVerdict::Unavailable("reviewer objected without an actionable reason".into())
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
        assert!(matches!(
            parse_verdict("   \n\n"),
            SkepticVerdict::Unavailable(_)
        ));
        assert!(matches!(
            parse_verdict("hmm, not sure"),
            SkepticVerdict::Unavailable(_)
        ));
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
    fn object_without_anything_actionable_is_unavailable() {
        assert!(matches!(
            parse_verdict("OBJECT"),
            SkepticVerdict::Unavailable(_)
        ));
        assert!(matches!(
            parse_verdict("OBJECT\n\n"),
            SkepticVerdict::Unavailable(_)
        ));
    }
}
