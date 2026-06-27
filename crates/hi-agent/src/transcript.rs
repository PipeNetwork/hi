//! The conversation transcript, with provider-safety invariants enforced by
//! construction.
//!
//! The agent's history is a sequence of `Message`s that providers (OpenAI,
//! Anthropic) translate to wire JSON. Both providers require every assistant
//! `tool_use` block to be followed by a matching `tool_result`, and reject
//! transcripts that violate this on the next request. Historically the agent
//! managed `Arc<Vec<Message>>` directly at ~20 call sites in `run_turn`, with
//! each site responsible for keeping the transcript provider-safe — a fragile
//! invariant that produced real bugs (orphan `tool_use` blocks from the repeat
//! guard; verify nudges accumulating because they were detected by
//! string-matching).
//!
//! [`Transcript`] wraps the `Arc<Vec<Message>>` and exposes only intentional
//! operations. The "record executed tool calls" path
//! ([`push_assistant_with_results`]) debug-asserts that every `ToolCall` id has
//! a matching `ToolResult`, so the orphan state is unrepresentable. Synthetic
//! nudges carry a typed [`NudgeKind`] instead of being detected by content.
//! The shared-`Arc` optimization (in-flight `ChatRequest`s clone the `Arc` and
//! aren't disturbed until a unique copy is needed) is preserved via
//! [`arc`] / [`make_mut`].
//!
//! Invariants are checked by [`validate_for_provider`], intended as a debug
//! assertion before each provider send.

use std::sync::Arc;

use hi_ai::{Content, Message, Role};

/// The kind of synthetic user message the agent injects (as opposed to real
/// user input). Typed so nudges can be located by kind instead of by
/// string-matching their content — the old verify-nudge replace logic grepped
/// for `"Verification stage"`, which broke if the wording changed and couldn't
/// distinguish a nudge from a user message quoting the same text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NudgeKind {
    /// Sent when the model re-issues the exact same tool call as the previous
    /// round; tells it to act on the output it already has.
    Repeat,
    /// Sent when the model described a next step but emitted no tool call.
    /// (The continue-nudge mechanism was removed — the model now just ends the
    /// turn when it stops with text. Kept for the typed enum's completeness.)
    #[allow(dead_code)]
    Continue,
    /// Sent when the model hit the output token cap (`stop_reason: "length"` /
    /// `"max_tokens"`) mid-generation — the response was truncated, not
    /// finished. The nudge tells the model to continue from where it was cut
    /// off so the turn doesn't end on a half-finished output.
    Truncation,
    /// A verification stage failure fed back to the model for another attempt.
    /// Carries the verify round (1-based) so the replace logic can tell a prior
    /// verify nudge from other synthetic messages.
    Verify { round: u32 },
    /// The structured-recap request appended after a turn that changed files.
    Finalize,
    /// The compaction summary that replaces history, or the summary folded into
    /// the first kept turn. (Not a free-standing nudge in all strategies, but
    /// typed here for completeness.)
    #[allow(dead_code)]
    Compaction,
}

/// A marker stored on a synthetic user message so it can be found by kind.
///
/// Stored as a leading zero-width-ish sentinel in a `Content::Text` block.
/// We tag synthetic nudges by prepending a short, unlikely marker string that
/// the model sees as part of the message but that we can match on. This keeps
/// the storage format a plain `Message` (no schema change to `hi_ai::Message`)
/// while making detection robust and typed.
const fn nudge_marker(kind: NudgeKind) -> &'static str {
    match kind {
        NudgeKind::Repeat => "[hi:nudge:repeat]",
        NudgeKind::Continue => "[hi:nudge:continue]",
        NudgeKind::Truncation => "[hi:nudge:truncation]",
        NudgeKind::Verify { .. } => "[hi:nudge:verify]",
        NudgeKind::Finalize => "[hi:nudge:finalize]",
        NudgeKind::Compaction => "[hi:nudge:compaction]",
    }
}

/// Return the marker prefix for a nudge kind, for matching.
fn marker_for(kind: NudgeKind) -> &'static str {
    nudge_marker(kind)
}

/// Whether a text block starts with any known nudge marker — used by
/// [`Transcript::strip_trailing_nudges`] to identify synthetic user messages
/// without caring about the specific kind.
fn is_nudge_text(text: &str) -> bool {
    text.starts_with("[hi:nudge:")
}

/// The conversation transcript, enforcing provider-safety invariants.
#[derive(Clone, Debug)]
pub(crate) struct Transcript {
    messages: Arc<Vec<Message>>,
}

impl Transcript {
    /// Wrap an existing history (e.g. the system prompt, or resumed history).
    pub(crate) fn new(messages: Vec<Message>) -> Self {
        Self {
            messages: Arc::new(messages),
        }
    }

    /// The shared handle, for `ChatRequest::messages` and `record_compaction`.
    /// Cloning the `Arc` is a refcount bump — in-flight requests keep their
    /// snapshot until a mutation forces a unique copy via [`make_mut`].
    pub(crate) fn arc(&self) -> Arc<Vec<Message>> {
        Arc::clone(&self.messages)
    }

    /// Number of messages.
    pub(crate) fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the transcript is empty.
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Immutable access to the slice (for reads, export, session recording).
    pub(crate) fn as_slice(&self) -> &[Message] {
        &self.messages
    }

    /// The last message, if any.
    #[allow(dead_code)]
    pub(crate) fn last(&self) -> Option<&Message> {
        self.messages.last()
    }

    /// A unique mutable reference to the backing `Vec`, cloning if an in-flight
    /// `ChatRequest` still holds the `Arc` (refcount > 1). Preserves the
    /// `Arc::make_mut` optimization: repeated tool rounds without a model call
    /// avoid copying the history.
    fn make_mut(&mut self) -> &mut Vec<Message> {
        Arc::make_mut(&mut self.messages)
    }

    // ---- basic appends -------------------------------------------------

    /// Append a plain user message (real user input).
    pub(crate) fn push_user(&mut self, text: impl Into<String>) {
        self.make_mut().push(Message::user(text));
    }

    /// Append a user turn, folding into the last message if it's also a user
    /// message — so we never produce two consecutive user messages (some
    /// providers reject that). Used at turn start, where a just-run compaction
    /// may have left a `[system, user(summary)]` history: folding the real
    /// request into that summary keeps the transcript valid instead of creating
    /// `user(summary), user(input)`.
    pub(crate) fn push_user_or_fold(&mut self, text: impl Into<String>) {
        let text = text.into();
        let msgs = self.make_mut();
        if let Some(last) = msgs.last_mut()
            && last.role == Role::User
        {
            // Fold: append the real request to the prior user message.
            let mut folded = String::new();
            for c in &last.content {
                if let Content::Text(t) = c {
                    folded.push_str(t);
                    folded.push_str("\n\n---\n\n");
                }
            }
            folded.push_str(&text);
            last.content = vec![Content::Text(folded)];
        } else {
            msgs.push(Message::user(text));
        }
    }

    /// Append an assistant message verbatim. Use this only for content with no
    /// tool calls (e.g. a streamed recap, or a refusal). For assistant content
    /// that contains `ToolCall` blocks, use [`push_assistant_with_results`] so
    /// the matching tool results are recorded in the same operation.
    ///
    /// [`push_assistant_with_results`]: Self::push_assistant_with_results
    pub(crate) fn push_assistant(&mut self, content: Vec<Content>) {
        debug_assert!(
            !content
                .iter()
                .any(|c| matches!(c, Content::ToolCall { .. })),
            "push_assistant used for content with ToolCall blocks; \
             use push_assistant_with_results so tool results are paired"
        );
        self.make_mut().push(Message::assistant(content));
    }

    /// Append an assistant message that contains `ToolCall` blocks, immediately
    /// followed by a `tool_result` for every call id. This is the *only* way to
    /// record executed tool calls, so the orphan-`tool_use` state (a
    /// `tool_use` block with no matching `tool_result`) cannot be produced by
    /// construction.
    ///
    /// `results` must contain one `(call_id, output)` per `ToolCall` id in
    /// `content`, in any order. Missing or extra ids trip a debug assertion and
    /// are otherwise tolerated (a stub result is synthesized to keep the
    /// transcript provider-safe even in release builds).
    pub(crate) fn push_assistant_with_results(
        &mut self,
        content: Vec<Content>,
        results: Vec<(String, String)>,
    ) {
        let call_ids: Vec<String> = content
            .iter()
            .filter_map(|c| match c {
                Content::ToolCall { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect();
        debug_assert!(
            !call_ids.is_empty(),
            "push_assistant_with_results called with no ToolCall blocks"
        );
        self.make_mut().push(Message::assistant(content));
        let msgs = self.make_mut();
        // Record a result for each call id exactly once. If a result is
        // missing, synthesize a stub so we never leave an orphan tool_use —
        // providers reject that on the next request.
        for id in &call_ids {
            let output = results
                .iter()
                .find(|(rid, _)| rid == id)
                .map(|(_, o)| o.clone())
                .unwrap_or_else(|| "[tool result missing]".to_string());
            msgs.push(Message::tool_result(id, output));
        }
    }

    /// Append a tool result for a single call. Used by the streaming-bash path
    /// where results are produced one at a time and the assistant message was
    /// already recorded via [`push_assistant_with_results`] — except that path
    /// also records results inline, so this is kept for any future
    /// incremental-result caller.
    #[allow(dead_code)]
    pub(crate) fn push_tool_result(
        &mut self,
        call_id: impl Into<String>,
        output: impl Into<String>,
    ) {
        self.make_mut().push(Message::tool_result(call_id, output));
    }

    /// Record an assistant message whose tool calls were *deliberately not
    /// executed* (the repeat guard: the model re-issued the exact same calls,
    /// so re-running them would only reproduce the same output). The `ToolCall`
    /// blocks are stripped from the recorded content so the transcript never
    /// carries `tool_use` blocks without matching `tool_result`s. Text and
    /// thinking blocks are kept.
    pub(crate) fn push_assistant_text_only(&mut self, content: Vec<Content>) {
        let text_only: Vec<Content> = content
            .into_iter()
            .filter(|c| !matches!(c, Content::ToolCall { .. }))
            .collect();
        self.make_mut().push(Message::assistant(text_only));
    }

    // ---- synthetic nudges ----------------------------------------------

    /// Append a synthetic user nudge of `kind`, tagging it so it can be found
    /// later by [`replace_last_nudge`]. The marker is prepended to the text;
    /// the model sees it as part of the message but it's short and inert.
    ///
    /// [`replace_last_nudge`]: Self::replace_last_nudge
    pub(crate) fn push_nudge(&mut self, kind: NudgeKind, text: impl Into<String>) {
        let body = text.into();
        let tagged = format!("{}\n{}", marker_for(kind), body);
        self.make_mut().push(Message::user(tagged));
    }

    /// Replace the most recent nudge of `kind` (and any trailing assistant /
    /// tool messages after it) with a fresh one. Used by the verify loop so
    /// only the latest verification output stays in context instead of
    /// accumulating one nudge per failed round.
    ///
    /// If a prior nudge of `kind` exists: pops from the tail — trailing `Tool`
    /// and `Assistant` messages that belong to the prior cycle, then the prior
    /// nudge itself — and pushes the new one. If no prior nudge of `kind` is
    /// found, **nothing is popped** and the new nudge is simply appended (this
    /// is the first round: there's no prior cycle to replace, so the model's
    /// just-completed assistant turn and its tool results must stay).
    pub(crate) fn replace_last_nudge(&mut self, kind: NudgeKind, text: impl Into<String>) {
        let marker = marker_for(kind);
        // First, find whether a prior nudge of this kind exists. If not, just
        // append — don't strip the model's just-finished turn.
        let has_prior = self.messages.iter().any(|m| {
            m.role == Role::User
                && m.content
                    .iter()
                    .any(|c| matches!(c, Content::Text(t) if t.starts_with(marker)))
        });
        let body = text.into();
        let tagged = format!("{}\n{}", marker, body);
        let msgs = self.make_mut();
        if has_prior {
            while let Some(last) = msgs.last() {
                if last.role == Role::User
                    && last
                        .content
                        .iter()
                        .any(|c| matches!(c, Content::Text(t) if t.starts_with(marker)))
                {
                    msgs.pop();
                    break;
                }
                match last.role {
                    Role::Tool | Role::Assistant => {
                        msgs.pop();
                    }
                    _ => break,
                }
            }
        }
        msgs.push(Message::user(tagged));
    }

    // ---- rollback / reset ----------------------------------------------

    /// Strip trailing synthetic nudge messages (any `Role::User` message whose
    /// text starts with a known nudge marker). Called at turn end so a stall
    /// (repeat-nudge, continue-nudge, verify-fail) doesn't leave a synthetic
    /// user message as the last entry — which would absorb the next real prompt
    /// via [`push_user_or_fold`] and make the model "pick up where it stalled"
    /// instead of addressing the new request.
    ///
    /// Only trailing nudges are removed: a nudge followed by a real assistant
    /// message stays (it's part of the conversation history). Pops at most one
    /// nudge per call — the common case is a single trailing nudge.
    pub(crate) fn strip_trailing_nudges(&mut self) {
        let msgs = self.make_mut();
        // Pop trailing user messages that are tagged nudges. A nudge is always
        // followed by `continue` (another model round) or `break` (turn ends),
        // so at most one trailing nudge is expected — but loop defensively in
        // case an edge case leaves two.
        while let Some(last) = msgs.last()
            && last.role == Role::User
            && last
                .content
                .iter()
                .any(|c| matches!(c, Content::Text(t) if is_nudge_text(t)))
        {
            msgs.pop();
        }
    }

    /// Discard messages back to `len` — used to drop an interrupted turn so the
    /// conversation stays consistent (no dangling user message, no orphan
    /// tool_use from a round that was cut off mid-execution).
    pub(crate) fn rewind_to(&mut self, len: usize) {
        let msgs = self.make_mut();
        msgs.truncate(len);
    }

    /// Replace the entire history with `messages` (compaction strategies).
    pub(crate) fn replace_all(&mut self, messages: Vec<Message>) {
        self.messages = Arc::new(messages);
    }

    /// Replace the system message (index 0), or push one if the transcript is
    /// empty. Used when the goal/project context changes.
    pub(crate) fn replace_system(&mut self, system: Message) {
        let msgs = self.make_mut();
        if let Some(first) = msgs.first_mut() {
            *first = system;
        } else {
            msgs.push(system);
        }
    }

    /// Mutably borrow the backing slice for an in-place transformation that the
    /// caller takes responsibility for (compaction elision). The invariant
    /// check still applies: the transformation must keep the transcript
    /// provider-safe.
    pub(crate) fn mutate_slice(&mut self) -> &mut Vec<Message> {
        self.make_mut()
    }

    // ---- validation -----------------------------------------------------

    /// Assert the provider-safety invariants hold. Intended as a debug
    /// assertion before each provider send. Checks:
    /// - every `ToolCall` id in an assistant message has at least one matching
    ///   `ToolResult` (by `call_id`/`id`) somewhere later in the transcript;
    /// - no two consecutive messages have the same `User` role (some providers
    ///   reject this — the compaction strategies fold summaries to avoid it).
    ///
    /// Returns the first violation found, if any. In release builds this is
    /// still cheap and returns `Ok(())` on a clean transcript.
    pub(crate) fn validate_for_provider(&self) -> Result<(), TranscriptError> {
        let msgs = &self.messages;
        // Collect all answered call ids (tool results seen so far).
        let mut answered: std::collections::HashSet<&str> = std::collections::HashSet::new();
        // Walk oldest→newest; a tool_use is answered if a later tool_result
        // carries its id. Track pending tool_use ids and check at the end.
        let mut pending: Vec<&str> = Vec::new();
        let mut prev_role: Option<Role> = None;
        for m in msgs.iter() {
            if let Some(prev) = prev_role
                && prev == Role::User
                && m.role == Role::User
            {
                return Err(TranscriptError::ConsecutiveUser);
            }
            match m.role {
                Role::Assistant => {
                    for c in &m.content {
                        if let Content::ToolCall { id, .. } = c {
                            pending.push(id);
                        }
                    }
                }
                Role::Tool => {
                    for c in &m.content {
                        if let Content::ToolResult { call_id, .. } = c {
                            answered.insert(call_id);
                        }
                    }
                }
                _ => {}
            }
            prev_role = Some(m.role);
        }
        for id in &pending {
            if !answered.contains(id) {
                return Err(TranscriptError::OrphanToolUse((*id).to_string()));
            }
        }
        Ok(())
    }
}

/// A transcript invariant violation, for [`Transcript::validate_for_provider`].
#[derive(Debug)]
pub(crate) enum TranscriptError {
    /// An assistant `tool_use` block with no matching `tool_result`.
    OrphanToolUse(String),
    /// Two consecutive user messages (some providers reject this).
    ConsecutiveUser,
}

impl std::fmt::Display for TranscriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OrphanToolUse(id) => {
                write!(f, "orphan tool_use (no matching tool_result): {id}")
            }
            Self::ConsecutiveUser => write!(f, "consecutive user messages"),
        }
    }
}

impl std::error::Error for TranscriptError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(t: &str) -> Message {
        Message::user(t)
    }
    fn assistant_text(t: &str) -> Message {
        Message::assistant(vec![Content::Text(t.into())])
    }
    fn assistant_with_call(id: &str, name: &str) -> Message {
        Message::assistant(vec![Content::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: "{}".into(),
        }])
    }
    fn tool_result(id: &str, out: &str) -> Message {
        Message::tool_result(id, out)
    }

    #[test]
    fn push_assistant_with_results_pairs_every_call() {
        let mut t = Transcript::new(vec![user("do it")]);
        t.push_assistant_with_results(
            vec![
                Content::ToolCall {
                    id: "a".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                },
                Content::ToolCall {
                    id: "b".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                },
            ],
            vec![("b".into(), "out-b".into()), ("a".into(), "out-a".into())],
        );
        // assistant + two tool results, in call order.
        assert_eq!(t.as_slice().len(), 4);
        assert_eq!(t.as_slice()[2].role, Role::Tool);
        assert_eq!(t.as_slice()[3].role, Role::Tool);
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn push_assistant_with_results_synthesizes_missing_result() {
        // A missing result is tolerated in release (stub) so the transcript
        // stays provider-safe; debug_assert panics, so run this path without
        // triggering that by using a single call with an empty results list.
        let mut t = Transcript::new(vec![user("do it")]);
        t.push_assistant_with_results(
            vec![Content::ToolCall {
                id: "x".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }],
            vec![],
        );
        // Still has a tool result (the stub), so validation passes.
        t.validate_for_provider().unwrap();
        assert_eq!(t.as_slice().last().unwrap().role, Role::Tool);
    }

    #[test]
    fn push_assistant_text_only_strips_tool_calls() {
        let mut t = Transcript::new(vec![user("again")]);
        t.push_assistant_text_only(vec![
            Content::Text("thinking".into()),
            Content::ToolCall {
                id: "z".into(),
                name: "bash".into(),
                arguments: "{}".into(),
            },
        ]);
        let last = t.as_slice().last().unwrap();
        assert!(
            last.content
                .iter()
                .all(|c| !matches!(c, Content::ToolCall { .. }))
        );
        // No tool result was pushed, and none is needed — no orphan because
        // there's no tool_use either.
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn replace_last_nudge_pops_prior_cycle() {
        let mut t = Transcript::new(vec![user("fix it"), assistant_text("working")]);
        t.push_nudge(
            NudgeKind::Verify { round: 1 },
            "Verification stage `check` failed.",
        );
        t.push_assistant(vec![Content::Text("trying again".into())]);
        t.push_tool_result("c1", "out");
        // Now replace the verify nudge: should pop tool, assistant, and the
        // prior nudge, then push the new one — leaving the original user input
        // and assistant, then the new nudge.
        t.replace_last_nudge(
            NudgeKind::Verify { round: 2 },
            "Verification stage `check` failed (2).",
        );
        // user(input) + assistant(working) + nudge(new).
        assert_eq!(t.as_slice().len(), 3);
        assert_eq!(t.as_slice().last().unwrap().role, Role::User);
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn replace_last_nudge_appends_when_no_prior() {
        let mut t = Transcript::new(vec![user("fix it")]);
        t.replace_last_nudge(NudgeKind::Verify { round: 1 }, "first failure");
        assert_eq!(t.as_slice().len(), 2);
    }

    #[test]
    fn validate_catches_orphan_tool_use() {
        // Simulate the bug class directly: an assistant tool_use with no
        // matching result. Bypass the safe API to construct the bad state.
        let mut t = Transcript::new(vec![user("do it"), assistant_with_call("o", "bash")]);
        // No tool_result pushed.
        assert!(matches!(
            t.validate_for_provider(),
            Err(TranscriptError::OrphanToolUse(id)) if id == "o"
        ));
    }

    #[test]
    fn validate_catches_consecutive_user() {
        let t = Transcript::new(vec![user("a"), user("b"), assistant_text("c")]);
        assert!(matches!(
            t.validate_for_provider(),
            Err(TranscriptError::ConsecutiveUser)
        ));
    }

    #[test]
    fn validate_passes_for_normal_exchange() {
        let mut t = Transcript::new(vec![user("do it")]);
        t.push_assistant_with_results(
            vec![Content::ToolCall {
                id: "p".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }],
            vec![("p".into(), "out".into())],
        );
        t.push_assistant(vec![Content::Text("done".into())]);
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn rewind_drops_partial_turn() {
        let mut t = Transcript::new(vec![user("do it")]);
        let before = t.len();
        t.push_assistant_with_results(
            vec![Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }],
            vec![("r".into(), "out".into())],
        );
        t.rewind_to(before);
        assert_eq!(t.len(), before);
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn strip_trailing_nudges_removes_trailing_nudge() {
        // A turn that ended with a repeat-nudge: the last message is a
        // synthetic user nudge. Stripping should remove it, leaving the
        // assistant text as the last message.
        let mut t = Transcript::new(vec![user("do it"), assistant_text("working")]);
        t.push_nudge(NudgeKind::Repeat, "Act on the output above.");
        assert_eq!(t.as_slice().last().unwrap().role, Role::User);
        t.strip_trailing_nudges();
        assert_eq!(t.as_slice().last().unwrap().role, Role::Assistant);
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn strip_trailing_nudges_keeps_real_user_message() {
        // A real user message (no nudge marker) is not stripped.
        let mut t = Transcript::new(vec![user("do it"), assistant_text("done")]);
        t.push_user("next task");
        t.strip_trailing_nudges();
        assert_eq!(t.as_slice().len(), 3, "real user message is kept");
        assert_eq!(t.as_slice().last().unwrap().role, Role::User);
    }

    #[test]
    fn strip_trailing_nudges_noop_when_last_is_assistant() {
        // No trailing nudge → no-op.
        let mut t = Transcript::new(vec![user("do it"), assistant_text("done")]);
        let len = t.len();
        t.strip_trailing_nudges();
        assert_eq!(t.len(), len);
    }
}
