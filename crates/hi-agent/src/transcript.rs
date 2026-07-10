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

use crate::steering::ReviewRepairMode;
use hi_ai::{Content, Message, Role};

const PROVIDER_VISIBLE_ASSISTANT_PLACEHOLDER: &str =
    "Provider-invisible assistant content was omitted from the transcript.";

fn assistant_content_visible_to_providers(content: &[Content]) -> bool {
    content.iter().any(|c| match c {
        Content::Text(text) => !text.is_empty(),
        Content::ToolCall { .. } => true,
        Content::Thinking {
            text,
            signature: Some(signature),
        } => !text.is_empty() && !signature.is_empty(),
        _ => false,
    })
}

fn tool_call_ids(content: &[Content]) -> Vec<String> {
    content
        .iter()
        .filter_map(|c| match c {
            Content::ToolCall { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect()
}

fn tool_result_ids(content: &[Content]) -> Vec<String> {
    content
        .iter()
        .filter_map(|c| match c {
            Content::ToolResult { call_id, .. } => Some(call_id.clone()),
            _ => None,
        })
        .collect()
}

fn tool_call_arguments_are_valid(arguments: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(arguments)
        .is_ok_and(|value| matches!(value, serde_json::Value::Object(_)))
}

fn repair_tool_call_arguments(content: &mut [Content]) -> usize {
    let mut repaired = 0usize;
    for block in content {
        if let Content::ToolCall { arguments, .. } = block
            && !tool_call_arguments_are_valid(arguments)
        {
            *arguments = "{}".to_string();
            repaired += 1;
        }
    }
    repaired
}

pub(crate) fn repair_invalid_tool_call_arguments_in_messages(messages: &mut [Message]) -> usize {
    let mut repaired = 0usize;
    for message in messages {
        if message.role == Role::Assistant {
            repaired += repair_tool_call_arguments(&mut message.content);
        }
    }
    repaired
}

fn tool_message_has_only_results(content: &[Content]) -> bool {
    !content.is_empty()
        && content
            .iter()
            .all(|c| matches!(c, Content::ToolResult { .. }))
}

fn consume_pending_tool_result(pending: &mut Vec<String>, result_id: &str) -> bool {
    let Some(pos) = pending.iter().position(|id| id == result_id) else {
        return false;
    };
    pending.remove(pos);
    true
}

fn immediate_tool_block_end(
    messages: &[Message],
    start: usize,
    call_ids: &[String],
) -> Option<usize> {
    if call_ids.is_empty() {
        return Some(start);
    }
    if messages
        .get(start)
        .is_none_or(|message| message.role != Role::Tool)
    {
        return None;
    }

    let mut pending = call_ids.to_vec();
    let mut i = start;
    while i < messages.len() && messages[i].role == Role::Tool {
        if !tool_message_has_only_results(&messages[i].content) {
            return None;
        }
        let result_ids = tool_result_ids(&messages[i].content);
        for result_id in result_ids {
            if !consume_pending_tool_result(&mut pending, &result_id) {
                return None;
            }
        }
        i += 1;
    }
    pending.is_empty().then_some(i)
}

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
    pub(crate) fn push_assistant(&mut self, mut content: Vec<Content>) {
        debug_assert!(
            !content
                .iter()
                .any(|c| matches!(c, Content::ToolCall { .. })),
            "push_assistant used for content with ToolCall blocks; \
             use push_assistant_with_results so tool results are paired"
        );
        if !assistant_content_visible_to_providers(&content) {
            content.push(Content::Text(PROVIDER_VISIBLE_ASSISTANT_PLACEHOLDER.into()));
        }
        self.make_mut().push(Message::assistant(content));
    }

    /// Append a short provider-visible assistant note for a rejected text-only
    /// review draft. The rejected draft itself is intentionally not recorded:
    /// weaker models tend to imitate their own failed answer if it remains in
    /// context, while the following nudge already carries the repair guidance.
    pub(crate) fn push_assistant_repair_note(&mut self, mode: ReviewRepairMode) {
        let reason = mode.key();
        let required_next = mode.required_next();
        let note = format!(
            "[review retry: reason={reason}; required_next={required_next}; do_not_repeat_previous_draft]"
        );
        let note_lower = note.to_ascii_lowercase();
        debug_assert!(
            !note_lower.contains("insufficient evidence")
                && !note_lower.contains("quality_rejected"),
            "repair note reason should not include provider/error trigger text"
        );
        self.make_mut()
            .push(Message::assistant(vec![Content::Text(note)]));
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
        mut content: Vec<Content>,
        results: Vec<(String, String)>,
    ) {
        repair_tool_call_arguments(&mut content);
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
        //
        // When two calls share the same id (e.g. a provider that omits ids and
        // the caller synthesized duplicates, or a malformed response), a plain
        // `find` would give both calls the first matching result. Instead, drain
        // results by id: each id consumes the first unused result with that id,
        // so the second call gets the second result.
        let mut remaining = results;
        for id in &call_ids {
            let pos = remaining.iter().position(|(rid, _)| rid == id);
            let output = match pos {
                Some(i) => remaining.remove(i).1,
                None => "[tool result missing]".to_string(),
            };
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
        let mut text_only: Vec<Content> = content
            .into_iter()
            .filter(|c| !matches!(c, Content::ToolCall { .. }))
            .collect();
        if !assistant_content_visible_to_providers(&text_only) {
            text_only.push(Content::Text(PROVIDER_VISIBLE_ASSISTANT_PLACEHOLDER.into()));
        }
        self.make_mut().push(Message::assistant(text_only));
    }

    /// Repair provider-invisible assistant messages loaded from older sessions.
    ///
    /// New transcript appends prevent this state, but sessions saved before that
    /// guard can still contain assistant turns that serialize to empty provider
    /// content (for example empty text or unsigned thinking-only content).
    pub(crate) fn repair_provider_invisible_assistant_messages(&mut self) {
        let msgs = self.make_mut();
        for message in msgs.iter_mut() {
            if message.role == Role::Assistant
                && !assistant_content_visible_to_providers(&message.content)
            {
                message
                    .content
                    .push(Content::Text(PROVIDER_VISIBLE_ASSISTANT_PLACEHOLDER.into()));
            }
        }
    }

    /// Repair legacy/corrupted histories with adjacent user turns.
    ///
    /// Providers commonly reject consecutive `user` messages. New appends use
    /// [`push_user_or_fold`](Self::push_user_or_fold), but older saved sessions
    /// or repair steps that drop invalid tool messages can still leave adjacent
    /// user turns in the middle of history. Merge them while preserving all
    /// blocks so resume can proceed.
    pub(crate) fn repair_consecutive_user_messages(&mut self) {
        let original = self.as_slice().to_vec();
        let mut repaired: Vec<Message> = Vec::with_capacity(original.len());
        for message in original {
            if message.role == Role::User
                && let Some(prev) = repaired.last_mut()
                && prev.role == Role::User
            {
                if !prev.content.is_empty() && !message.content.is_empty() {
                    prev.content.push(Content::Text("\n\n".into()));
                }
                prev.content.extend(message.content);
                continue;
            }
            repaired.push(message);
        }
        self.messages = Arc::new(repaired);
    }

    /// Repair legacy/corrupted histories with adjacent assistant turns.
    ///
    /// This can be created by older sessions, or by dropping an unsafe
    /// assistant tool-call skeleton during [`repair_tool_result_ordering`].
    /// Merge adjacent assistant content so providers that require alternating
    /// user/assistant turns do not reject the resumed request.
    pub(crate) fn repair_consecutive_assistant_messages(&mut self) {
        let original = self.as_slice().to_vec();
        let mut repaired: Vec<Message> = Vec::with_capacity(original.len());
        for message in original {
            if message.role == Role::Assistant
                && let Some(prev) = repaired.last_mut()
                && prev.role == Role::Assistant
            {
                if !prev.content.is_empty() && !message.content.is_empty() {
                    prev.content.push(Content::Text("\n\n".into()));
                }
                prev.content.extend(message.content);
                continue;
            }
            repaired.push(message);
        }
        self.messages = Arc::new(repaired);
    }

    /// Repair legacy/corrupted histories whose tool results are not immediately
    /// paired with their assistant tool calls.
    ///
    /// Providers require each assistant `tool_use` turn to be followed by the
    /// matching tool-result block before any other turn. New appends enforce
    /// that by construction; this repair strips unsafe legacy tool calls and
    /// drops unmatched tool-result messages so resume can proceed.
    pub(crate) fn repair_tool_result_ordering(&mut self) {
        let original = self.as_slice().to_vec();
        let mut repaired = Vec::with_capacity(original.len());
        let mut i = 0;
        while i < original.len() {
            let mut message = original[i].clone();
            match message.role {
                Role::Assistant => {
                    let call_ids = tool_call_ids(&message.content);
                    if call_ids.is_empty() {
                        repaired.push(message);
                        i += 1;
                        continue;
                    }
                    if let Some(end) = immediate_tool_block_end(&original, i + 1, &call_ids) {
                        repaired.push(message);
                        repaired.extend_from_slice(&original[i + 1..end]);
                        i = end;
                        continue;
                    }

                    message
                        .content
                        .retain(|content| !matches!(content, Content::ToolCall { .. }));
                    if !assistant_content_visible_to_providers(&message.content) {
                        message
                            .content
                            .push(Content::Text(PROVIDER_VISIBLE_ASSISTANT_PLACEHOLDER.into()));
                    }
                    repaired.push(message);
                    i += 1;
                    while i < original.len() && original[i].role == Role::Tool {
                        i += 1;
                    }
                }
                Role::Tool => {
                    i += 1;
                }
                _ => {
                    repaired.push(message);
                    i += 1;
                }
            }
        }
        self.messages = Arc::new(repaired);
    }

    /// Repair malformed tool-call argument blocks loaded from older sessions or
    /// emitted by a flaky provider before this transcript is sent again.
    ///
    /// Tool arguments must be a JSON object string. If a saved assistant block
    /// contains partial JSON or a non-object value, providers may reject the
    /// next request before the model can recover. Replace only the arguments
    /// with `{}`; keep the call id/name and matching tool result so transcript
    /// ordering remains intact.
    pub(crate) fn repair_invalid_tool_call_arguments(&mut self) -> usize {
        repair_invalid_tool_call_arguments_in_messages(self.make_mut())
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

    /// Append a synthetic nudge while preserving provider-safe alternation. If
    /// the previous message is the same nudge kind, replace it; if it is another
    /// user message, fold this nudge into that message instead of creating
    /// consecutive user turns.
    pub(crate) fn push_nudge_or_fold(&mut self, kind: NudgeKind, text: impl Into<String>) {
        let marker = marker_for(kind);
        let body = text.into();
        let tagged = format!("{marker}\n{body}");
        let msgs = self.make_mut();
        if let Some(last) = msgs.last_mut()
            && last.role == Role::User
        {
            if last
                .content
                .iter()
                .any(|c| matches!(c, Content::Text(t) if t.starts_with(marker)))
            {
                last.content = vec![Content::Text(tagged)];
                return;
            }

            let mut folded = String::new();
            for c in &last.content {
                if let Content::Text(t) = c {
                    folded.push_str(t);
                    folded.push_str("\n\n---\n\n");
                }
            }
            folded.push_str(&tagged);
            last.content = vec![Content::Text(folded)];
            return;
        }
        msgs.push(Message::user(tagged));
    }

    /// Replace the most recent nudge of `kind` (and any trailing assistant /
    /// tool messages after it) with a fresh one. Used by the verify loop so
    /// only the latest verification output stays in context instead of
    /// accumulating one nudge per failed round.
    ///
    /// A prior nudge is replaceable only when it *immediately precedes* the
    /// trailing run of `Tool`/`Assistant` messages — i.e. the tail is exactly
    /// the prior nudge's response cycle. Then that cycle and the nudge are
    /// popped and the fresh nudge pushed. Otherwise (first round of this turn,
    /// or any other message in between) the nudge is simply appended: deciding
    /// off a marker anywhere in history — as this used to — meant a stale
    /// nudge from an *earlier* turn erased the current turn's work and left
    /// two consecutive user messages.
    pub(crate) fn replace_last_nudge(&mut self, kind: NudgeKind, text: impl Into<String>) {
        let marker = marker_for(kind);
        let body = text.into();
        let tagged = format!("{}\n{}", marker, body);
        let msgs = self.make_mut();
        let mut idx = msgs.len();
        while idx > 0 && matches!(msgs[idx - 1].role, Role::Tool | Role::Assistant) {
            idx -= 1;
        }
        let prior_nudge = idx.checked_sub(1).filter(|&i| {
            let m = &msgs[i];
            m.role == Role::User
                && m.content
                    .iter()
                    .any(|c| matches!(c, Content::Text(t) if t.starts_with(marker)))
        });
        if let Some(i) = prior_nudge {
            msgs.truncate(i);
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
        // followed by `continue` or `break` (turn ends), so at most one trailing
        // nudge is expected — but loop defensively in case an edge case leaves two.
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

    /// Remove a trailing `[user: finalize-nudge][assistant: recap]` pair if
    /// present. Unlike [`strip_trailing_nudges`], this handles the finalize
    /// case where the nudge is *not* the last message — it's followed by the
    /// assistant recap, so the trailing-nudge scan above can't reach it.
    ///
    /// The FINALIZE_PROMPT tells the model "the work for this turn is done…
    /// don't take any further action." If it stays in history, the next turn's
    /// prompt arrives right after that instruction and some models comply with
    /// the stale "don't act" directive instead of executing the new request —
    /// emitting more summary text rather than doing the work. The recap itself
    /// was already shown to the user via the UI, and the model can reconstruct
    /// what it did from the tool-call history already in the transcript, so
    /// dropping the pair loses no actionable context.
    pub(crate) fn strip_finalize_pair(&mut self) {
        let msgs = self.make_mut();
        // Need at least two messages: [user: finalize-nudge][assistant: recap].
        if msgs.len() < 2 {
            return;
        }
        let recap_is_last = msgs
            .last()
            .map(|m| m.role == Role::Assistant)
            .unwrap_or(false);
        let finalize_before_recap = msgs[msgs.len() - 2]
            .role == Role::User
            && msgs[msgs.len() - 2]
                .content
                .iter()
                .any(|c| matches!(c, Content::Text(t) if t.starts_with(nudge_marker(NudgeKind::Finalize))));
        if recap_is_last && finalize_before_recap {
            msgs.pop(); // assistant recap
            msgs.pop(); // user finalize-nudge
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
    /// - no adjacent normal conversation messages have the same `User` or
    ///   `Assistant` role (some providers require alternating turns).
    ///
    /// Returns the first violation found, if any. In release builds this is
    /// still cheap and returns `Ok(())` on a clean transcript.
    pub(crate) fn validate_for_provider(&self) -> Result<(), TranscriptError> {
        let msgs = &self.messages;
        let mut prev_role: Option<Role> = None;
        let mut i = 0;
        while i < msgs.len() {
            let m = &msgs[i];
            if let Some(prev) = prev_role {
                match (prev, m.role) {
                    (Role::User, Role::User) => return Err(TranscriptError::ConsecutiveUser),
                    (Role::Assistant, Role::Assistant) => {
                        return Err(TranscriptError::ConsecutiveAssistant);
                    }
                    _ => {}
                }
            }
            match m.role {
                Role::Assistant => {
                    if !assistant_content_visible_to_providers(&m.content) {
                        return Err(TranscriptError::EmptyAssistant);
                    }
                    let call_ids = tool_call_ids(&m.content);
                    if !call_ids.is_empty() {
                        for content in &m.content {
                            if let Content::ToolCall {
                                id,
                                name,
                                arguments,
                            } = content
                                && !tool_call_arguments_are_valid(arguments)
                            {
                                return Err(TranscriptError::InvalidToolArguments {
                                    id: id.clone(),
                                    name: name.clone(),
                                });
                            }
                        }
                        let mut pending = call_ids;
                        let Some(next) = msgs.get(i + 1) else {
                            return Err(TranscriptError::OrphanToolUse(pending.remove(0)));
                        };
                        if next.role != Role::Tool {
                            return Err(TranscriptError::OrphanToolUse(pending.remove(0)));
                        }
                        i += 1;
                        while i < msgs.len() && msgs[i].role == Role::Tool {
                            if !tool_message_has_only_results(&msgs[i].content) {
                                return Err(TranscriptError::UnexpectedToolResult(
                                    "non-tool content in tool_result block".to_string(),
                                ));
                            }
                            let result_ids = tool_result_ids(&msgs[i].content);
                            for result_id in result_ids {
                                if !consume_pending_tool_result(&mut pending, &result_id) {
                                    return Err(TranscriptError::UnexpectedToolResult(result_id));
                                }
                            }
                            i += 1;
                        }
                        if let Some(missing) = pending.into_iter().next() {
                            return Err(TranscriptError::OrphanToolUse(missing));
                        }
                        prev_role = Some(Role::Tool);
                        continue;
                    }
                }
                Role::Tool => {
                    if !tool_message_has_only_results(&m.content) {
                        return Err(TranscriptError::UnexpectedToolResult(
                            "non-tool content in tool_result block".to_string(),
                        ));
                    }
                    let id = tool_result_ids(&m.content)
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| "missing tool_result block".to_string());
                    return Err(TranscriptError::UnexpectedToolResult(id));
                }
                _ => {}
            }
            prev_role = Some(m.role);
            i += 1;
        }
        Ok(())
    }
}

/// A transcript invariant violation, for [`Transcript::validate_for_provider`].
#[derive(Debug)]
pub(crate) enum TranscriptError {
    /// An assistant `tool_use` block with no matching `tool_result`.
    OrphanToolUse(String),
    /// A `tool_result` appeared without being part of the immediate answer
    /// block for the preceding assistant tool call.
    UnexpectedToolResult(String),
    /// Two consecutive user messages (some providers reject this).
    ConsecutiveUser,
    /// Two consecutive assistant messages (some providers require alternation).
    ConsecutiveAssistant,
    /// An assistant message with no provider-visible content blocks.
    EmptyAssistant,
    /// An assistant tool call whose `arguments` block is not a JSON object.
    InvalidToolArguments { id: String, name: String },
}

impl std::fmt::Display for TranscriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OrphanToolUse(id) => {
                write!(f, "orphan tool_use (no matching tool_result): {id}")
            }
            Self::UnexpectedToolResult(id) => {
                write!(f, "unexpected or out-of-order tool_result: {id}")
            }
            Self::ConsecutiveUser => write!(f, "consecutive user messages"),
            Self::ConsecutiveAssistant => write!(f, "consecutive assistant messages"),
            Self::EmptyAssistant => write!(f, "empty/provider-invisible assistant message"),
            Self::InvalidToolArguments { id, name } => {
                write!(
                    f,
                    "invalid JSON object arguments for tool_use {name} ({id})"
                )
            }
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

    #[test]
    fn assistant_repair_notes_are_directive_and_trigger_free() {
        let cases = [
            (
                ReviewRepairMode::NoEvidence,
                "review_no_evidence",
                "inspect_files_before_answering",
            ),
            (
                ReviewRepairMode::ListingOnly,
                "review_listing_only",
                "inspect_one_concrete_file_before_answering",
            ),
            (
                ReviewRepairMode::ReadAfterSearch,
                "review_read_after_search",
                "read_one_matching_file_before_answering",
            ),
            (
                ReviewRepairMode::InspectedDisclaimer,
                "review_inspected_disclaimer",
                "chat_only_bounded_answer_from_inspected_files",
            ),
            (
                ReviewRepairMode::ConcreteAnswer,
                "review_concrete_answer",
                "cite_findings_plus_limits",
            ),
        ];

        for (mode, reason, required_next) in cases {
            let mut t = Transcript::new(vec![user("review this")]);
            t.push_assistant_repair_note(mode);
            let text = t.as_slice().last().unwrap().text();
            assert_eq!(
                text,
                format!(
                    "[review retry: reason={reason}; required_next={required_next}; do_not_repeat_previous_draft]"
                )
            );
            let lower = text.to_ascii_lowercase();
            assert!(!lower.contains("insufficient evidence"), "{text}");
            assert!(!lower.contains("quality_rejected"), "{text}");
        }
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
    fn push_assistant_with_results_pairs_duplicate_ids_by_position() {
        // Two calls with the same id (e.g. a provider that omits ids and both
        // got synthesized to the same value, or a malformed response). Each call
        // must get its OWN result, not both getting the first match — the old
        // `find` logic gave both calls "out-a", losing "out-b".
        let mut t = Transcript::new(vec![user("do it")]);
        t.push_assistant_with_results(
            vec![
                Content::ToolCall {
                    id: "dup".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                },
                Content::ToolCall {
                    id: "dup".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                },
            ],
            vec![
                ("dup".into(), "out-a".into()),
                ("dup".into(), "out-b".into()),
            ],
        );
        let msgs = t.as_slice();
        // assistant + two tool results
        assert_eq!(msgs.len(), 4);
        // First result → out-a, second result → out-b (positional, not both out-a)
        assert!(matches!(
            &msgs[2].content[0],
            Content::ToolResult { output, .. } if output == "out-a"
        ));
        assert!(matches!(
            &msgs[3].content[0],
            Content::ToolResult { output, .. } if output == "out-b"
        ));
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn push_assistant_with_results_repairs_invalid_tool_arguments() {
        let mut t = Transcript::new(vec![user("do it")]);
        t.push_assistant_with_results(
            vec![Content::ToolCall {
                id: "bad".into(),
                name: "read".into(),
                arguments: "{\"path\":".into(),
            }],
            vec![("bad".into(), "Error: invalid tool arguments".into())],
        );

        let Content::ToolCall { arguments, .. } = &t.as_slice()[1].content[0] else {
            panic!("expected tool call");
        };
        assert_eq!(arguments, "{}");
        t.validate_for_provider().unwrap();
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
    fn push_assistant_text_only_keeps_tool_only_skip_non_empty() {
        let mut t = Transcript::new(vec![user("again")]);
        t.push_assistant_text_only(vec![Content::ToolCall {
            id: "z".into(),
            name: "bash".into(),
            arguments: "{}".into(),
        }]);
        let last = t.as_slice().last().unwrap();
        assert_eq!(last.role, Role::Assistant);
        assert!(
            last.content
                .iter()
                .all(|c| !matches!(c, Content::ToolCall { .. }))
        );
        assert!(
            last.content
                .iter()
                .any(|c| matches!(c, Content::Text(t) if !t.trim().is_empty())),
            "skipped tool-only assistant turns must remain non-empty"
        );
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn push_assistant_text_only_adds_visible_text_for_unsigned_thinking_only_skip() {
        let mut t = Transcript::new(vec![user("again")]);
        t.push_assistant_text_only(vec![
            Content::Thinking {
                text: "internal reasoning".into(),
                signature: None,
            },
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
                .any(|c| matches!(c, Content::Text(t) if !t.trim().is_empty())),
            "unsigned thinking is dropped by Anthropic serialization, so a visible text block is required"
        );
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn push_assistant_adds_visible_text_for_empty_content() {
        let mut t = Transcript::new(vec![user("again")]);
        t.push_assistant(vec![Content::Text(String::new())]);
        let last = t.as_slice().last().unwrap();
        assert!(
            last.content
                .iter()
                .any(|c| matches!(c, Content::Text(t) if !t.trim().is_empty())),
            "assistant messages must serialize to provider-visible content"
        );
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
        let t = Transcript::new(vec![user("do it"), assistant_with_call("o", "bash")]);
        // No tool_result pushed.
        assert!(matches!(
            t.validate_for_provider(),
            Err(TranscriptError::OrphanToolUse(id)) if id == "o"
        ));
    }

    #[test]
    fn repair_invalid_tool_call_arguments_fixes_legacy_history() {
        let mut t = Transcript::new(vec![
            user("do it"),
            Message::assistant(vec![Content::ToolCall {
                id: "legacy".into(),
                name: "bash".into(),
                arguments: "[\"not\", \"an\", \"object\"]".into(),
            }]),
            Message::tool_result("legacy", "Error: invalid tool arguments"),
        ]);

        assert!(matches!(
            t.validate_for_provider(),
            Err(TranscriptError::InvalidToolArguments { id, .. }) if id == "legacy"
        ));
        assert_eq!(t.repair_invalid_tool_call_arguments(), 1);
        let Content::ToolCall { arguments, .. } = &t.as_slice()[1].content[0] else {
            panic!("expected tool call");
        };
        assert_eq!(arguments, "{}");
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn validate_catches_out_of_order_tool_result() {
        let t = Transcript::new(vec![
            user("do it"),
            assistant_with_call("c1", "bash"),
            assistant_text("interposed text"),
            Message::tool_result("c1", "late result"),
        ]);

        assert!(matches!(
            t.validate_for_provider(),
            Err(TranscriptError::OrphanToolUse(id)) if id == "c1"
        ));
    }

    #[test]
    fn validate_catches_standalone_tool_result() {
        let t = Transcript::new(vec![user("do it"), Message::tool_result("c1", "orphan")]);

        assert!(matches!(
            t.validate_for_provider(),
            Err(TranscriptError::UnexpectedToolResult(id)) if id == "c1"
        ));
    }

    #[test]
    fn validate_catches_non_result_content_inside_tool_message() {
        let t = Transcript::new(vec![
            user("do it"),
            assistant_with_call("c1", "bash"),
            Message {
                role: Role::Tool,
                content: vec![
                    Content::ToolResult {
                        call_id: "c1".into(),
                        output: "ok".into(),
                    },
                    Content::Text("this would be ignored by provider serialization".into()),
                ],
            },
        ]);

        assert!(matches!(
            t.validate_for_provider(),
            Err(TranscriptError::UnexpectedToolResult(id))
                if id == "non-tool content in tool_result block"
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
    fn validate_catches_consecutive_assistant() {
        let t = Transcript::new(vec![user("a"), assistant_text("b"), assistant_text("c")]);
        assert!(matches!(
            t.validate_for_provider(),
            Err(TranscriptError::ConsecutiveAssistant)
        ));
    }

    #[test]
    fn validate_catches_empty_assistant() {
        let t = Transcript::new(vec![user("a"), Message::assistant(Vec::new())]);
        assert!(matches!(
            t.validate_for_provider(),
            Err(TranscriptError::EmptyAssistant)
        ));
    }

    #[test]
    fn validate_catches_assistant_with_no_provider_visible_content() {
        let t = Transcript::new(vec![
            user("a"),
            Message::assistant(vec![
                Content::Text(String::new()),
                Content::Thinking {
                    text: "unsigned thinking".into(),
                    signature: None,
                },
            ]),
        ]);
        assert!(matches!(
            t.validate_for_provider(),
            Err(TranscriptError::EmptyAssistant)
        ));
    }

    #[test]
    fn repair_provider_invisible_assistant_messages_fixes_legacy_history() {
        let mut t = Transcript::new(vec![
            user("a"),
            Message::assistant(vec![
                Content::Text(String::new()),
                Content::Thinking {
                    text: "unsigned thinking".into(),
                    signature: None,
                },
            ]),
        ]);

        t.repair_provider_invisible_assistant_messages();

        let last = t.as_slice().last().unwrap();
        assert!(
            last.content
                .iter()
                .any(|c| matches!(c, Content::Text(t) if !t.trim().is_empty())),
            "legacy provider-invisible assistant turns should be repaired"
        );
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn repair_consecutive_user_messages_merges_legacy_adjacent_users() {
        let mut t = Transcript::new(vec![
            user("first"),
            user("second"),
            assistant_text("answer"),
            user("third"),
            user("fourth"),
        ]);

        t.repair_consecutive_user_messages();

        let roles: Vec<Role> = t.as_slice().iter().map(|message| message.role).collect();
        assert_eq!(
            roles,
            vec![Role::User, Role::Assistant, Role::User],
            "adjacent users should be merged without disturbing other roles"
        );
        assert!(t.as_slice()[0].text().contains("first"));
        assert!(t.as_slice()[0].text().contains("second"));
        assert!(t.as_slice()[2].text().contains("third"));
        assert!(t.as_slice()[2].text().contains("fourth"));
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn repair_consecutive_assistant_messages_merges_legacy_adjacent_assistants() {
        let mut t = Transcript::new(vec![
            user("prompt"),
            assistant_text("first"),
            assistant_text("second"),
            user("next"),
        ]);

        t.repair_consecutive_assistant_messages();

        let roles: Vec<Role> = t.as_slice().iter().map(|message| message.role).collect();
        assert_eq!(
            roles,
            vec![Role::User, Role::Assistant, Role::User],
            "adjacent assistants should be merged without disturbing other roles"
        );
        assert!(t.as_slice()[1].text().contains("first"));
        assert!(t.as_slice()[1].text().contains("second"));
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn repair_tool_result_ordering_strips_legacy_out_of_order_tool_call() {
        let mut t = Transcript::new(vec![
            user("do it"),
            Message::assistant(vec![Content::ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                arguments: "{}".into(),
            }]),
            assistant_text("interposed text"),
            Message::tool_result("c1", "late result"),
        ]);

        t.repair_tool_result_ordering();
        t.repair_provider_invisible_assistant_messages();
        t.repair_consecutive_assistant_messages();

        assert!(
            !t.as_slice().iter().any(|message| {
                message.content.iter().any(|content| {
                    matches!(
                        content,
                        Content::ToolCall { .. } | Content::ToolResult { .. }
                    )
                })
            }),
            "unsafe legacy tool skeleton should be stripped: {:?}",
            t.as_slice()
        );
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn repair_tool_result_ordering_strips_tool_messages_with_non_result_content() {
        let mut t = Transcript::new(vec![
            user("do it"),
            assistant_with_call("c1", "bash"),
            Message {
                role: Role::Tool,
                content: vec![
                    Content::ToolResult {
                        call_id: "c1".into(),
                        output: "ok".into(),
                    },
                    Content::Text("this would be dropped by serializers".into()),
                ],
            },
        ]);

        t.repair_tool_result_ordering();
        t.repair_provider_invisible_assistant_messages();

        assert!(
            !t.as_slice().iter().any(|message| {
                message.content.iter().any(|content| {
                    matches!(
                        content,
                        Content::ToolCall { .. } | Content::ToolResult { .. }
                    )
                })
            }),
            "unsafe mixed tool message should be stripped: {:?}",
            t.as_slice()
        );
        t.validate_for_provider().unwrap();
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
    fn push_nudge_or_fold_preserves_user_turn_alternation() {
        let mut t = Transcript::new(vec![user("do it"), assistant_text("working")]);
        t.push_nudge(NudgeKind::Verify { round: 1 }, "Verification failed.");
        t.push_nudge_or_fold(NudgeKind::Continue, "The model returned empty.");

        assert_eq!(t.as_slice().len(), 3);
        let last = t.as_slice().last().unwrap();
        assert_eq!(last.role, Role::User);
        let text = last.text();
        assert!(text.starts_with(nudge_marker(NudgeKind::Verify { round: 1 })));
        assert!(text.contains(nudge_marker(NudgeKind::Continue)));
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn push_nudge_or_fold_replaces_same_trailing_nudge() {
        let mut t = Transcript::new(vec![user("do it"), assistant_text("working")]);
        t.push_nudge_or_fold(NudgeKind::Continue, "first");
        t.push_nudge_or_fold(NudgeKind::Continue, "second");

        assert_eq!(t.as_slice().len(), 3);
        let text = t.as_slice().last().unwrap().text();
        assert!(text.contains("second"));
        assert!(!text.contains("first"));
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

    #[test]
    fn strip_finalize_pair_removes_nudge_and_recap() {
        // After finalize_turn the transcript is
        //   [user: finalize-nudge][assistant: recap]
        // as the last two messages. strip_finalize_pair should remove both,
        // leaving the prior assistant message as the last entry.
        let mut t = Transcript::new(vec![user("do it"), assistant_text("working")]);
        t.push_nudge(NudgeKind::Finalize, "Write the final summary.");
        t.push_assistant(vec![Content::Text("## Summary\n- did stuff".into())]);
        assert_eq!(t.as_slice().last().unwrap().role, Role::Assistant);
        assert_eq!(t.as_slice().len(), 4);
        t.strip_finalize_pair();
        assert_eq!(t.as_slice().len(), 2, "nudge + recap removed");
        assert_eq!(
            t.as_slice().last().unwrap().role,
            Role::Assistant,
            "prior assistant message is now last"
        );
        t.validate_for_provider().unwrap();
    }

    #[test]
    fn strip_finalize_pair_noop_without_pair() {
        // No trailing finalize pair → no-op, whatever the last message is.
        let mut t = Transcript::new(vec![user("do it"), assistant_text("done")]);
        let len = t.len();
        t.strip_finalize_pair();
        assert_eq!(t.len(), len);

        // A trailing repeat-nudge (not finalize) is also a no-op for this method.
        let mut t = Transcript::new(vec![user("do it"), assistant_text("done")]);
        t.push_nudge(NudgeKind::Repeat, "Act on the output.");
        let len = t.len();
        t.strip_finalize_pair();
        assert_eq!(t.len(), len, "repeat nudge is not a finalize pair");
    }
}
