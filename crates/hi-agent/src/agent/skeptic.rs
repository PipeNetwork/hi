//! Second-model **completion / goal review** transport shared by three product gates.
//!
//! # Fail-open is transport-level, not global product policy
//!
//! [`SkepticVerdict::Unavailable`] is what the transport returns on provider
//! errors or unparseable replies. Product policy decides what that means:
//!
//! | Gate | Config | Unavailable behavior |
//! |---|---|---|
//! | Goal skeptic (`skeptic_gate`) | `AgentGates::skeptic_fail_open` (default **false**) | fail-closed: block goal advance |
//! | Independent / large-diff review | always records [`crate::ReviewStatus::Unavailable`] | visible on outcome; does **not** re-enter Model |
//!
//! The `/goal team` skeptic gate is a bounded second-model review of a turn
//! before it advances a sub-goal. Modeled on the planner side-call
//! ([`decompose_goal`]): a throwaway chat-only request through `self.provider`
//! at the effective skeptic model (`skeptic_model`, falling back to the session
//! model so the gate works unconfigured), usage booked, no history recorded.
//!
//! Distinct from Steer-phase **answer repair** (`ReviewRepairMode`) — this
//! module never nudges answer quality; it only yields a verdict after mutation
//! or at goal advance.
//!
//! [`decompose_goal`]: crate::Agent::decompose_goal

use std::sync::Arc;

use crate::domain::VerifyEvidence;

use hi_ai::{ChatRequest, Content, Message, RequestProfile, StreamEvent, ToolMode};

/// How much of the turn diff to show the **goal** skeptic, counted in **Unicode
/// chars** (not bytes). Intentionally smaller than completion-review's
/// [`COMPLETION_REVIEW_DIFF_BUDGET`] (50_000): goal `skeptic_gate` /
/// [`Agent::review_diff`] are bounded side-calls that must stay cheap, while
/// post-verify completion review can afford a fuller package.
const SKEPTIC_DIFF_BUDGET: usize = 6_000;
/// Bound on objective / sub-goal text copied into a side-call. The stored
/// goal and the user's transcript message are not rewritten.
const MAX_SKEPTIC_SIDE_CHARS: usize = 8_000;
const MAX_SKEPTIC_FILES: usize = 16;
const MAX_SKEPTIC_PATH_CHARS: usize = 200;

/// Diff char budget for independent / large-diff **completion** review (post
/// green WorkspaceRepair). Kept here next to the goal budget so the asymmetry
/// is obvious; applied in `verify_outcome`.
pub(crate) const COMPLETION_REVIEW_DIFF_BUDGET: usize = 50_000;

const SKEPTIC_PROMPT: &str = "You are a code reviewer acting as a merge gate for a coding agent. \
You see the objective, the active sub-goal, prior review notes on this step, the agent's verify \
result, and the diff it just \
produced. Your ONLY job is to block a change that fails to accomplish the active sub-goal — not to \
improve it or hold it to a higher standard. Judge the sub-goal's OUTCOME: do not object because \
the implementation's internal structure, naming, or approach differs from what you would have \
chosen — the how is the implementer's choice unless the sub-goal itself mandates it. Bias \
strongly toward APPROVE. Reply APPROVE on the \
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
accomplished. \
On a re-review (prior review notes are present), your PRIMARY job is to confirm the previously \
noted defects are addressed — the bar does NOT rise between rounds: a concern that earlier \
rounds accepted, or that you did not raise when you first saw this work, is not grounds to \
object now. Reply ESCALATE on the first line — instead of OBJECT — when retrying cannot fix the \
problem: the sub-goal contradicts the objective or the work already done, or completing/verifying \
it needs information or a decision only the user can provide. Escalation is rare; a fixable \
defect is an OBJECT. Do NOT object over style or naming. Missing tests ARE grounds \
to OBJECT when the sub-goal or task contract demands them; otherwise do not object over \
missing tests, speculative edge cases, or anything you merely cannot verify from the diff. \
When uncertain, APPROVE — a wrong objection wastes a real retry. After OBJECT or ESCALATE, \
put one concrete reason per line. The very first \
non-empty line of your reply must be the single word APPROVE, OBJECT, or ESCALATE — no preamble.";

const INDEPENDENT_REVIEW_PROMPT: &str = "You are the independent completion reviewer for a coding \
agent. Review the task contract, scoped repository instructions, complete bounded diff, relevant \
context, and deterministic verification evidence.\n\n\
Your reply MUST start with exactly one of these words on line 1 (nothing before it):\n\
APPROVE\n\
or\n\
OBJECT\n\n\
Use APPROVE only when the change satisfies the stated acceptance contract without a concrete \
regression. Use OBJECT when you find a specific correctness, security, compatibility, migration, \
or acceptance defect — then put one actionable defect per following line. Do not object over \
style or speculation; every objection must identify the affected behavior or file. Do not write \
preamble, analysis, or markdown headings before the verdict word.";

/// Phase L: same gate as independent review, biased toward catching multi-file
/// holes (missed call sites, partial renames, unfinished siblings) after tests
/// already passed. Still fail-open at the call site.
const LARGE_DIFF_REVIEW_PROMPT: &str = "You are reviewing a LARGE multi-file coding change that \
already passed deterministic compile/test checks. Focus on defects tests often miss:\n\
- call sites not updated after a rename/signature change\n\
- a required file left untouched while siblings changed\n\
- partial migrations (old and new paths both live incorrectly)\n\
- stubs/placeholders (todo!/unimplemented!/NotImplemented) where real behavior was required\n\
- acceptance criteria in the task contract still unsatisfied\n\n\
Your reply MUST start with exactly one of these words on line 1 (nothing before it):\n\
APPROVE\n\
or\n\
OBJECT\n\n\
Bias toward APPROVE when the diff plausibly completes the task. OBJECT only with concrete, \
file-specific defects (one per following line). No style nits, no speculation, no preamble.";

/// The skeptic's verdict on whether the active sub-goal may advance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkepticVerdict {
    /// Advance the sub-goal.
    Approve,
    /// Send it back to retry, carrying these concrete objections (fed into the
    /// sub-goal's notes so the next turn sees them).
    Object(Vec<String>),
    /// Retrying cannot fix it — the sub-goal contradicts the objective/prior
    /// work, or needs a user decision. The driver skips the step (visible
    /// `Failed` scar + note) instead of burning retries on an unwinnable loop.
    Escalate(Vec<String>),
    /// Reviewer configuration, transport, or output could not yield a verdict.
    Unavailable(String),
}

fn verify_ran_test_stage(executions: &[crate::VerificationExecution]) -> bool {
    executions.iter().any(|execution| {
        execution.status == hi_tools::ToolStatus::Succeeded
            && (execution.name.contains("test")
                || execution.command.contains("test")
                || execution.command.contains("pytest"))
    })
}

impl crate::Agent {
    pub(crate) async fn independent_review(&mut self, context: &str) -> SkepticVerdict {
        let model = self.effective_skeptic_model().to_string();
        self.review_with_prompt(context, INDEPENDENT_REVIEW_PROMPT, model)
            .await
    }

    /// Phase L large-diff skeptic — same transport/fail-open as independent
    /// review, with a prompt tuned for multi-file holes after green verify.
    pub(crate) async fn large_diff_review(&mut self, context: &str) -> SkepticVerdict {
        let model = self.effective_skeptic_model().to_string();
        self.review_with_prompt(context, LARGE_DIFF_REVIEW_PROMPT, model)
            .await
    }
    /// Run the skeptic gate against `sub_goal` (the sub-goal that was active at
    /// turn start — the current one may already be marked done via update_plan).
    /// `prior_notes` are the step's accumulated review/retry notes: on a
    /// re-review they anchor the anti-ratchet contract (confirm prior defects
    /// are fixed; the bar does not rise).
    ///
    /// Transport returns [`SkepticVerdict::Unavailable`] on provider error or
    /// unparseable reply; callers apply `skeptic_fail_open` (default fail-closed).
    /// Books usage; records no history.
    pub(crate) async fn skeptic_gate(
        &mut self,
        objective: &str,
        sub_goal: &str,
        prior_notes: &[String],
    ) -> SkepticVerdict {
        if let Some(reason) = self.missing_required_tests_objection(sub_goal) {
            return SkepticVerdict::Object(vec![reason]);
        }
        let context = self.skeptic_context(objective, sub_goal, prior_notes).await;
        self.skeptic_review(&context).await
    }

    /// Review an arbitrary `(objective, sub_goal, diff)` with the real skeptic —
    /// for offline *detector* evaluation of the reviewer (precision/recall on
    /// labeled diffs), independent of a live goal.
    ///
    /// Returns the raw [`SkepticVerdict`] so callers can distinguish transport
    /// failure ([`SkepticVerdict::Unavailable`]) from a real Approve. Product
    /// policy (e.g. `skeptic_fail_open`) is **not** applied here — same prompt
    /// and model as the gate, no history recorded.
    pub async fn review_diff(
        &mut self,
        objective: &str,
        sub_goal: &str,
        diff: &str,
    ) -> SkepticVerdict {
        let mut diff = diff.to_string();
        // Char-count budget (not bytes). Goal/eval path uses SKEPTIC_DIFF_BUDGET
        // (6k), not completion-review's larger package — see constant docs.
        if diff.chars().count() > SKEPTIC_DIFF_BUDGET {
            diff = diff.chars().take(SKEPTIC_DIFF_BUDGET).collect();
            diff.push_str("\n… (diff truncated)");
        }
        // Mirror the gate's context format so the reviewer sees the same shape.
        let context = review_diff_context(objective, sub_goal, &diff);
        self.skeptic_review(&context).await
    }

    /// Assemble the review blob: objective + active sub-goal + prior review
    /// notes + verify result + changed files + a best-effort diff of this
    /// turn's changes (truncated).
    async fn skeptic_context(
        &mut self,
        objective: &str,
        sub_goal: &str,
        prior_notes: &[String],
    ) -> String {
        let notes = if prior_notes.is_empty() {
            "(none — first review of this step)".to_string()
        } else {
            let clipped: String = prior_notes
                .iter()
                .take(crate::goal::MAX_NOTES_IN_PROMPT)
                .map(|n| {
                    format!(
                        "\n  — {}",
                        crate::goal::clip_chars(n, crate::goal::MAX_NOTE_CHARS)
                    )
                })
                .collect();
            if prior_notes.len() > crate::goal::MAX_NOTES_IN_PROMPT {
                format!(
                    "{clipped}\n  — … {} more",
                    prior_notes.len() - crate::goal::MAX_NOTES_IN_PROMPT
                )
            } else {
                clipped
            }
        };
        let verify = match self.report.verify {
            VerifyEvidence::Passed { .. } => "verify result: PASSED",
            VerifyEvidence::Failed => "verify result: FAILED",
            VerifyEvidence::None => "verify result: (none configured)",
        };
        let files = if self.workspace.last_changed_files.is_empty() {
            "(none detected)".to_string()
        } else {
            let mut parts: Vec<String> = self
                .workspace
                .last_changed_files
                .iter()
                .take(MAX_SKEPTIC_FILES)
                .map(|path| crate::goal::clip_chars(path, MAX_SKEPTIC_PATH_CHARS))
                .collect();
            if self.workspace.last_changed_files.len() > MAX_SKEPTIC_FILES {
                parts.push(format!(
                    "… {} more",
                    self.workspace.last_changed_files.len() - MAX_SKEPTIC_FILES
                ));
            }
            parts.join(", ")
        };
        let stub_findings = self.turn_stub_scan().await;
        let stubs = if stub_findings.is_empty() {
            "(none detected)".to_string()
        } else {
            stub_findings
                .iter()
                .map(|f| format!("\n  {}:{}: {}", f.path, f.line, f.marker))
                .collect()
        };
        let mut diff = self.turn_diff().await;
        // Char-count budget (not bytes). Goal gate stays on the smaller
        // SKEPTIC_DIFF_BUDGET; completion review uses COMPLETION_REVIEW_DIFF_BUDGET.
        if diff.chars().count() > SKEPTIC_DIFF_BUDGET {
            diff = diff.chars().take(SKEPTIC_DIFF_BUDGET).collect();
            diff.push_str("\n… (diff truncated)");
        }
        let acceptance = self
            .task
            .last_task_contract
            .as_ref()
            .and_then(|c| c.acceptance_section())
            .unwrap_or_else(|| "(none named)".into());
        format!(
            "Objective: {}\n\n\
             Active sub-goal (the one about to be marked done): {}\n\n\
             Prior review notes on this step (re-review: confirm these are addressed; \
             the bar does not rise): {notes}\n\n\
             {verify}\n\
             Files changed this turn: {files}\n\
             Stub markers present in files changed this turn: {stubs}\n\n\
             Acceptance criteria:\n{acceptance}\n\n\
             Diff of this turn's changes:\n{diff}",
            crate::goal::clip_chars(objective, MAX_SKEPTIC_SIDE_CHARS),
            crate::goal::clip_chars(sub_goal, MAX_SKEPTIC_SIDE_CHARS),
        )
    }

    /// Deterministic OBJECT when the contract/sub-goal is test-gated but this
    /// turn added no test files and ran no test verification stage.
    fn missing_required_tests_objection(&self, sub_goal: &str) -> Option<String> {
        let wants = self
            .task
            .last_task_contract
            .as_ref()
            .is_some_and(|c| c.wants_tests)
            || crate::task_contract::prompt_wants_tests(sub_goal);
        if !wants {
            return None;
        }
        if self
            .workspace
            .last_changed_files
            .iter()
            .any(|path| crate::task_contract::path_looks_like_test(path))
        {
            return None;
        }
        if verify_ran_test_stage(self.last_verification_executions()) {
            return None;
        }
        Some(
            "the task/sub-goal is test-gated but this turn added no tests and ran no test stage"
                .into(),
        )
    }

    /// One bounded critique call to the effective skeptic model —
    /// `skeptic_model` when configured, otherwise the session model, so the
    /// gate works with zero configuration.
    ///
    /// Transport-level only: provider errors and empty/unparseable replies
    /// become [`SkepticVerdict::Unavailable`]. Callers apply product policy
    /// (`skeptic_fail_open` for the goal gate; completion review records
    /// `ReviewStatus::Unavailable`).
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
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: Arc::new(vec![Message::system(system_prompt), Message::user(context)]),
            tools: Arc::new([]), // review only — no tool use
            max_tokens: 1024,
            // Deterministic structured verdict — do not inherit the coding turn's
            // sampling (higher temp makes first-line APPROVE/OBJECT less reliable
            // on non-GLM hosts such as xAI).
            temperature: Some(0.0),
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

        // One bounded retry on a transient transport error (rate limit, brief
        // capacity/outage blip). A review that a single 429 could permanently
        // downgrade to "unavailable" is noise at the end of an otherwise-good
        // turn; anything persistent still reports unavailable after the retry.
        // Route to the opt-in skeptic endpoint (a local model) when configured,
        // otherwise the session provider — cloned so the borrow doesn't overlap
        // the `&mut self` usage-accounting calls below.
        // A managed local team server can die after the route was installed
        // (OOM, bad weights, or an external kill). Executor routing already
        // falls back to the driver in that case; keep the goal skeptic
        // consistent so a dead sidecar does not fail-closed and park every
        // goal. Explicit external endpoints are not classified as dead.
        let provider = if self.skeptic_route_is_dead() {
            self.provider.clone()
        } else {
            self.skeptic_provider
                .clone()
                .unwrap_or_else(|| self.provider.clone())
        };
        let mut attempts_left = 2u32;
        loop {
            attempts_left -= 1;
            let mut text = String::new();
            let mut sink = |event: StreamEvent| {
                if let StreamEvent::Text(t) = event {
                    text.push_str(&t);
                }
            };
            let completion = match provider.stream(request.clone(), &mut sink).await {
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
    /// Cached per turn keyed by the ledger revision it was computed at: the
    /// skeptic gate, trio review, verify-review gate, and completion audit all
    /// need this diff, and shelling out to git per call is the expensive part.
    /// A reconcile that moves the revision makes the cache miss, never stale.
    pub(crate) async fn turn_diff(&mut self) -> String {
        let revision = self.runtime.ledger().revision();
        if let Some((cached_revision, diff)) = &self.workspace.turn_diff_cache
            && *cached_revision == revision
        {
            return diff.clone();
        }
        let diff = match self.workspace.checkpoints.last() {
            Some(target) => hi_tools::checkpoint::diff_with_state(
                self.runtime.root(),
                target,
                self.runtime.state_root(),
            )
            .await
            .unwrap_or_default(),
            None => String::new(),
        };
        self.workspace.turn_diff_cache = Some((revision, diff.clone()));
        diff
    }

    /// Stub markers in the files changed this turn — cached per turn (keyed by
    /// the ledger revision, like `turn_diff`): the skeptic gate and the
    /// completion audit scan the same paths, and the scan reads each file.
    pub(crate) async fn turn_stub_scan(&mut self) -> Vec<hi_tools::stub_scan::StubFinding> {
        let revision = self.runtime.ledger().revision();
        if let Some((cached_revision, findings)) = &self.workspace.turn_stub_scan_cache
            && *cached_revision == revision
        {
            return findings.clone();
        }
        // Stub scanning reads up to 50 changed source files. Keep that bounded
        // filesystem work off the async executor; completion review is otherwise
        // able to freeze the UI while it scans a large mutation set.
        let root = self.runtime.root().to_path_buf();
        let paths = self.workspace.last_changed_files.clone();
        let findings =
            tokio::task::spawn_blocking(move || hi_tools::stub_scan::scan_paths(&root, &paths, 50))
                .await
                .unwrap_or_default();
        self.workspace.turn_stub_scan_cache = Some((revision, findings.clone()));
        findings
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerdictKind {
    Approve,
    Object,
    Escalate,
}

/// Classify one non-empty line as a verdict keyword, if any.
///
/// Accepts leading markdown/bullets and common phrasings models emit when they
/// ignore "first line only" (e.g. `**Verdict:** APPROVE`, `I APPROVE this`).
/// Negated approve phrasing (`do not approve`, `cannot approve`, …) is ignored
/// so a reject-shaped sentence cannot false-pass the gate.
fn verdict_kind_from_line(line: &str) -> Option<VerdictKind> {
    let clean =
        line.trim_matches(|c: char| matches!(c, '#' | '*' | '`' | '-' | '•' | ' ' | '"' | '\''));
    let lower = clean.to_ascii_lowercase();
    // Prefer an explicit leading keyword (protocol-compliant replies).
    if lower.starts_with("approve") {
        return Some(VerdictKind::Approve);
    }
    if lower.starts_with("escalate") {
        return Some(VerdictKind::Escalate);
    }
    if lower.starts_with("object") || lower.starts_with("reject") {
        return Some(VerdictKind::Object);
    }
    // Tolerant scan: models on xAI/OpenAI often write a short preamble, then a
    // verdict word alone or after a label. Require the keyword as a whole token.
    // Skip negated "approve" so "I do not approve" cannot false-pass.
    for (idx, _) in lower.match_indices("approve") {
        if is_keyword_token(&lower, idx, "approve".len()) && !approve_is_negated(&lower, idx) {
            return Some(VerdictKind::Approve);
        }
    }
    for (idx, _) in lower.match_indices("escalate") {
        if is_keyword_token(&lower, idx, "escalate".len()) {
            return Some(VerdictKind::Escalate);
        }
    }
    for (word, kind) in [
        ("object", VerdictKind::Object),
        ("reject", VerdictKind::Object),
    ] {
        for (idx, _) in lower.match_indices(word) {
            if is_keyword_token(&lower, idx, word.len()) {
                return Some(kind);
            }
        }
    }
    None
}

fn is_keyword_token(lower: &str, idx: usize, len: usize) -> bool {
    let before_ok = idx == 0
        || !lower
            .as_bytes()
            .get(idx - 1)
            .copied()
            .is_some_and(|b| b.is_ascii_alphanumeric());
    let after = idx + len;
    let after_ok = after >= lower.len()
        || !lower
            .as_bytes()
            .get(after)
            .copied()
            .is_some_and(|b| b.is_ascii_alphanumeric());
    before_ok && after_ok
}

/// True when `approve` at `idx` is preceded by a local negation on the same line.
///
/// Covers the common reject-shaped phrasings models emit instead of `OBJECT`:
/// `do not approve`, `don't approve`, `cannot approve`, `can't approve`,
/// `never approve`, bare `not approve`.
fn approve_is_negated(lower: &str, approve_idx: usize) -> bool {
    let before = lower[..approve_idx].trim_end();
    for neg in [
        "do not", "don't", "cannot", "can't", "can not", "never", "not",
    ] {
        if !before.ends_with(neg) {
            continue;
        }
        let start = before.len() - neg.len();
        let boundary_ok = start == 0 || !before.as_bytes()[start - 1].is_ascii_alphanumeric();
        if boundary_ok {
            return true;
        }
    }
    false
}

/// Parse the skeptic's reply into a verdict.
///
/// Looks for the first line that contains a verdict keyword (`APPROVE` /
/// `OBJECT` / `REJECT` / `ESCALATE`), not only line 1 — protocol-compliant
/// reviewers put it first, but several hosts (notably same-model review on xAI)
/// emit a short analysis before the keyword. Remaining non-empty lines after
/// that verdict (plus any inline text after the keyword) are objections.
///
/// Fail-closed parse (not fail-open):
/// - empty / no keyword / garbage → [`SkepticVerdict::Unavailable`]
/// - `OBJECT` / `ESCALATE` with no actionable reason body → [`SkepticVerdict::Unavailable`]
/// - only a clear `APPROVE` keyword → [`SkepticVerdict::Approve`]
///
/// Product policy for [`SkepticVerdict::Unavailable`] is caller-side (goal gate
/// vs completion review); this parser never treats ambiguity as approve.
fn review_diff_context(objective: &str, sub_goal: &str, diff: &str) -> String {
    format!(
        "Objective: {}\n\n\
         Active sub-goal (the one about to be marked done): {}\n\n\
         verify result: (none configured)\n\
         Files changed this turn: (see diff)\n\n\
         Diff of this turn's changes:\n{diff}",
        crate::goal::clip_chars(objective, MAX_SKEPTIC_SIDE_CHARS),
        crate::goal::clip_chars(sub_goal, MAX_SKEPTIC_SIDE_CHARS),
    )
}

fn parse_verdict(text: &str) -> SkepticVerdict {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return SkepticVerdict::Unavailable("reviewer returned empty output".into());
    }

    let mut verdict_at = None;
    for (i, line) in lines.iter().enumerate() {
        if let Some(kind) = verdict_kind_from_line(line) {
            verdict_at = Some((i, kind, *line));
            break;
        }
    }
    let Some((idx, kind, verdict_line)) = verdict_at else {
        return SkepticVerdict::Unavailable("reviewer output did not contain a verdict".into());
    };

    if kind == VerdictKind::Approve {
        return SkepticVerdict::Approve;
    }

    let clean = verdict_line
        .trim_matches(|c: char| matches!(c, '#' | '*' | '`' | '-' | '•' | ' ' | '"' | '\''));
    // Reasons: subsequent non-empty lines after the verdict line (bullets stripped) …
    let mut objs: Vec<String> = lines[idx + 1..]
        .iter()
        .map(|l| strip_bullet(l))
        .filter(|s| !s.is_empty())
        // Don't treat a second APPROVE/OBJECT as an objection body.
        .filter(|s| verdict_kind_from_line(s).is_none())
        .collect();
    // … plus any inline reason after the keyword on the verdict line itself.
    if let Some(inline) = inline_reason_after_keyword(clean) {
        objs.insert(0, inline);
    }
    if objs.is_empty() {
        SkepticVerdict::Unavailable("reviewer objected without an actionable reason".into())
    } else if kind == VerdictKind::Escalate {
        SkepticVerdict::Escalate(objs)
    } else {
        SkepticVerdict::Object(objs)
    }
}

fn inline_reason_after_keyword(clean: &str) -> Option<String> {
    let lower = clean.to_ascii_lowercase();
    for word in ["escalate", "object", "reject", "approve"] {
        if let Some(idx) = lower.find(word)
            && is_keyword_token(&lower, idx, word.len())
        {
            let after = idx + word.len();
            let inline = clean[after..].trim_matches(|c: char| {
                matches!(c, ':' | '-' | '—' | '.' | ' ' | '*' | '`' | ')' | '(')
            });
            if !inline.is_empty()
                && !inline.eq_ignore_ascii_case("this")
                && !inline.eq_ignore_ascii_case("the change")
            {
                return Some(inline.to_string());
            }
            return None;
        }
    }
    None
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
    fn large_diff_prompt_targets_multi_file_holes() {
        assert!(
            LARGE_DIFF_REVIEW_PROMPT.contains("call sites"),
            "should mention missed call sites"
        );
        assert!(
            LARGE_DIFF_REVIEW_PROMPT.contains("APPROVE")
                && LARGE_DIFF_REVIEW_PROMPT.contains("OBJECT"),
            "must keep the same verdict protocol"
        );
        assert!(
            LARGE_DIFF_REVIEW_PROMPT.contains("LARGE"),
            "should mark large-diff context"
        );
    }

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
    fn review_diff_context_clips_huge_objective_and_sub_goal() {
        let ctx = review_diff_context(&"O".repeat(20_000), &"S".repeat(20_000), "diff-here");
        assert!(
            ctx.chars().count() < 20_000,
            "side-call context must stay bounded: {}",
            ctx.chars().count()
        );
        assert!(ctx.contains("diff-here"), "{ctx}");
        assert!(
            !ctx.contains(&"O".repeat(MAX_SKEPTIC_SIDE_CHARS + 1)),
            "{ctx}"
        );
        assert!(ctx.contains('…'), "{ctx}");
    }

    #[test]
    fn approve_after_preamble_like_xai_same_model_review() {
        // Session models on xAI/OpenAI often narrate before the keyword.
        assert_eq!(
            parse_verdict("I reviewed the diff and verification evidence.\n\nAPPROVE\n"),
            SkepticVerdict::Approve
        );
        assert_eq!(
            parse_verdict("**Verdict:** APPROVE\nLooks good overall."),
            SkepticVerdict::Approve
        );
        assert_eq!(
            parse_verdict("Summary: the change meets the contract. I APPROVE."),
            SkepticVerdict::Approve
        );
    }

    #[test]
    fn object_after_preamble() {
        assert_eq!(
            parse_verdict("Analysis follows.\nOBJECT\n- missing error path in parser.rs\n"),
            SkepticVerdict::Object(vec!["missing error path in parser.rs".to_string()])
        );
    }

    #[test]
    fn escalate_variants() {
        let v = parse_verdict("ESCALATE\n- the sub-goal contradicts the frozen plan\n");
        assert_eq!(
            v,
            SkepticVerdict::Escalate(vec!["the sub-goal contradicts the frozen plan".to_string()])
        );
        assert_eq!(
            parse_verdict("**Escalate**: needs a user decision on the schema"),
            SkepticVerdict::Escalate(vec!["needs a user decision on the schema".to_string()])
        );
        // An escalation without a reason is unusable — Unavailable (caller policy).
        assert!(matches!(
            parse_verdict("ESCALATE"),
            SkepticVerdict::Unavailable(_)
        ));
    }

    #[test]
    fn negated_approve_does_not_false_pass() {
        assert!(matches!(
            parse_verdict("I do not approve this change"),
            SkepticVerdict::Unavailable(_)
        ));
        assert!(matches!(
            parse_verdict("cannot approve until error handling lands"),
            SkepticVerdict::Unavailable(_)
        ));
        assert!(matches!(
            parse_verdict("I don't approve.\nThe parser still drops empty input."),
            SkepticVerdict::Unavailable(_)
        ));
        assert!(matches!(
            parse_verdict("never approve a stub stand-in"),
            SkepticVerdict::Unavailable(_)
        ));
        // A real OBJECT after a negated-approve preamble still objects.
        assert_eq!(
            parse_verdict(
                "I cannot approve this as-is.\nOBJECT\n- missing error path in parser.rs\n"
            ),
            SkepticVerdict::Object(vec!["missing error path in parser.rs".to_string()])
        );
        // Positive approve still works when not negated.
        assert_eq!(
            parse_verdict("I approve this change."),
            SkepticVerdict::Approve
        );
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
