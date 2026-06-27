//! Streaming response parsing: drain an OpenAI SSE stream into a
//! [`Completion`], reassemble fragmented tool calls, and parse usage chunks.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use futures_util::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::Value;

use crate::provider::{ProviderError, ProviderErrorKind};
use crate::types::{
    BillableBreakdown, Completion, Content, StreamEvent, Usage, estimate_messages_tokens,
};

/// Once the model reports a `finish_reason`, it has stopped generating; we wait
/// only this long for the trailing usage chunk / `[DONE]` before ending the
/// turn. Without it, a provider that emits `finish_reason` but never sends
/// `[DONE]` (nor closes the socket) would wedge the turn until the much longer
/// idle timeout expires — a completed answer left spinning for ~2 minutes.
const FINISH_GRACE: Duration = Duration::from_secs(3);

/// ChatML special tokens that some local models (Qwen, Yi, etc.) emit as text
/// content when the server doesn't strip them. They start with `<|` and end
/// with `|>`. We strip them from streamed text so they never reach the UI or
/// the recorded history.
const SPECIAL_TOKEN_PREFIX: &str = "<|";
const SPECIAL_TOKEN_SUFFIX: &str = "|>";

/// Check if `inner` (the text between `<|` and `|>`) looks like a special
/// token identifier: non-empty, alphanumeric + underscore, no spaces or
/// newlines. Real tokens: `im_start`, `im_end`, `endoftext`, `system`, etc.
fn is_token_identifier(inner: &str) -> bool {
    !inner.is_empty() && inner.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Find the start of a potential partial `<|` prefix at the end of `text`,
/// searching from `from` onward. Returns the index of the `<` if the text ends
/// with `<` (which could be the start of `<|` in the next chunk), or
/// `text.len()` if no partial prefix is present.
fn find_partial_prefix_start(text: &str, from: usize) -> usize {
    let bytes = text.as_bytes();
    if from >= bytes.len() {
        return bytes.len();
    }
    // Only the last character can be a partial `<|` start (a lone `<`).
    if *bytes.last().unwrap() == b'<' {
        bytes.len() - 1
    } else {
        bytes.len()
    }
}

/// A streaming filter that combines two text-cleaning passes for local models:
///
/// 1. **Special-token stripping**: removes ChatML special tokens (`<|im_start|>`,
///    `<|im_end|>`, …) that some local models emit as raw text.
/// 2. **Tool-call JSON suppression**: when the model emits tool calls as text
///    content (`{"name": "bash", …}`), suppresses the raw JSON from the live
///    display. The post-collection `parse_text_tool_calls` promotes it to real
///    `ToolCall` blocks; this filter only hides it while streaming so the user
///    doesn't see a screen full of JSON.
///
/// Text flows through both passes: raw → special-token strip → tool-call
/// suppression → sink. Combining them into one struct avoids the borrow-conflict
/// of chaining two `&mut dyn FnMut` filters.
struct StreamingTextFilter<'a> {
    inner: &'a mut (dyn FnMut(StreamEvent) + Send),
    // ── Special-token pass state ──
    /// Buffered text that might be the start of a special token (`<|` without
    /// a matching `|>` yet).
    st_pending: String,
    // ── Tool-call suppression pass state ──
    /// Buffered text starting from a `{` that might be a tool-call JSON.
    tc_pending: String,
    /// Brace depth inside `tc_pending` (1 after the opening `{`).
    tc_depth: i32,
    /// Whether we're inside a string literal inside `tc_pending`.
    tc_in_string: bool,
    /// Whether the previous char was a backslash (string escape).
    tc_escape: bool,
    /// Index in `tc_pending` up to which we've already scanned.
    tc_scanned: usize,
    /// Whether we've already validated the tool name in the buffered JSON.
    /// Once we've seen the `"name": "value"` pair and confirmed `value` is a
    /// valid tool name, this is set to `true`. If `value` is NOT a valid tool
    /// name, the buffer is flushed immediately as text (early exit).
    tc_name_checked: bool,
    /// Count of string literals seen so far in the buffered JSON (0 = before
    /// the first `"`, 1 = inside the key `"name"`, 2 = inside the value string,
    /// etc.). Used to locate the tool name value for early validation.
    tc_string_count: u32,
    /// Whether we're expecting a `:` after the `"name"` key. If the next
    /// non-whitespace char isn't `:`, this isn't a JSON key-value pair — flush.
    tc_expect_colon: bool,
}

impl<'a> StreamingTextFilter<'a> {
    fn new(inner: &'a mut (dyn FnMut(StreamEvent) + Send)) -> Self {
        Self {
            inner,
            st_pending: String::new(),
            tc_pending: String::new(),
            tc_depth: 0,
            tc_in_string: false,
            tc_escape: false,
            tc_scanned: 0,
            tc_name_checked: false,
            tc_string_count: 0,
            tc_expect_colon: false,
        }
    }

    /// Forward a non-text event directly, flushing all buffered text first.
    fn forward(&mut self, event: StreamEvent) {
        self.flush();
        (self.inner)(event);
    }

    /// Process a new text chunk: first strip special tokens, then suppress
    /// tool-call JSON from the cleaned text.
    fn text(&mut self, chunk: &str) {
        // Phase 1: Strip special tokens, collecting cleaned text.
        let cleaned = self.strip_special_tokens_chunk(chunk);
        // Phase 2: Suppress tool-call JSON from the cleaned text.
        if !cleaned.is_empty() {
            self.suppress_tool_calls(&cleaned);
        }
    }

    /// Strip ChatML special tokens from a chunk, buffering partial `<|` starts.
    /// Returns the cleaned text (may be empty if everything was buffered).
    fn strip_special_tokens_chunk(&mut self, chunk: &str) -> String {
        let combined = if self.st_pending.is_empty() {
            chunk.to_string()
        } else {
            let mut buf = std::mem::take(&mut self.st_pending);
            buf.push_str(chunk);
            buf
        };
        let bytes = combined.as_bytes();
        let mut out = String::with_capacity(combined.len());
        let mut last = 0;
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                if let Some(end) = combined[i..].find(SPECIAL_TOKEN_SUFFIX) {
                    let token_end = i + end + SPECIAL_TOKEN_SUFFIX.len();
                    let inner = &combined[i + SPECIAL_TOKEN_PREFIX.len()..i + end];
                    if is_token_identifier(inner) {
                        out.push_str(&combined[last..i]);
                        last = token_end;
                        i = token_end;
                        continue;
                    }
                } else {
                    // No closing `|>` found — buffer from `<|` onward.
                    out.push_str(&combined[last..i]);
                    self.st_pending = combined[i..].to_string();
                    return out;
                }
            }
            i += 1;
        }
        // Check for a partial `<` at the end (could be start of `<|`).
        let tail_start = find_partial_prefix_start(&combined, last);
        if tail_start < bytes.len() {
            out.push_str(&combined[last..tail_start]);
            self.st_pending = combined[tail_start..].to_string();
        } else {
            out.push_str(&combined[last..]);
        }
        out
    }

    /// Suppress tool-call JSON from cleaned text, forwarding prose to the sink.
    fn suppress_tool_calls(&mut self, chunk: &str) {
        if self.tc_depth > 0 {
            self.tc_pending.push_str(chunk);
            self.scan_tc_buffer();
        } else {
            self.scan_tc_for_start(chunk);
        }
    }

    /// Scan a chunk (when not buffering) for `{"name"`.
    fn scan_tc_for_start(&mut self, chunk: &str) {
        let bytes = chunk.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'{' {
                let rest = &chunk[i..];
                if looks_like_tool_call_start(rest) {
                    if i > 0 {
                        (self.inner)(StreamEvent::Text(chunk[..i].to_string()));
                    }
                    self.tc_pending = chunk[i..].to_string();
                    self.tc_depth = 1;
                    self.tc_in_string = false;
                    self.tc_escape = false;
                    self.tc_scanned = 1;
                    self.tc_name_checked = false;
                    self.tc_string_count = 0;
                    self.tc_expect_colon = false;
                    self.scan_tc_buffer();
                    return;
                }
            }
            i += 1;
        }
        (self.inner)(StreamEvent::Text(chunk.to_string()));
    }

    /// Continue scanning `tc_pending` from `tc_scanned`.
    fn scan_tc_buffer(&mut self) {
        let bytes = self.tc_pending.as_bytes();
        let mut i = self.tc_scanned;
        while i < bytes.len() {
            let b = bytes[i];
            if self.tc_in_string {
                if self.tc_escape {
                    self.tc_escape = false;
                } else if b == b'\\' {
                    self.tc_escape = true;
                } else if b == b'"' {
                    self.tc_in_string = false;
                    // We just closed a string. The 2nd string (string_count
                    // transitions 1→2) is the tool name value — validate it.
                    if !self.tc_name_checked && self.tc_string_count == 2 {
                        self.tc_name_checked = true;
                        if !self.buffered_tool_name_is_valid() {
                            // Not a valid tool name — this isn't a tool call.
                            // Flush the buffer as text and re-scan the rest.
                            self.flush_tc_as_text();
                            return;
                        }
                    }
                    // After the 1st string (the "name" key) closes, we expect
                    // a `:` next (with optional whitespace). If we see anything
                    // else, this isn't a JSON key-value pair — flush.
                    if self.tc_string_count == 1 {
                        self.tc_expect_colon = true;
                    }
                }
            } else if b == b'"' {
                self.tc_in_string = true;
                self.tc_string_count += 1;
                self.tc_expect_colon = false;
            } else if b == b'{' || b == b'[' {
                self.tc_depth += 1;
                self.tc_expect_colon = false;
            } else if b == b'}' || b == b']' {
                self.tc_depth -= 1;
                self.tc_expect_colon = false;
                if self.tc_depth == 0 && b == b'}' {
                    let json_str = &self.tc_pending[..=i];
                    let rest_start = i + 1;
                    if is_tool_call_json(json_str) {
                        // Confirmed tool call — suppress it.
                        let rest = self.tc_pending[rest_start..].to_string();
                        self.tc_pending.clear();
                        self.tc_scanned = 0;
                        self.tc_name_checked = false;
                        self.tc_string_count = 0;
                        self.tc_expect_colon = false;
                        if !rest.is_empty() {
                            self.scan_tc_for_start(&rest);
                        }
                    } else {
                        // Not a tool call — flush as text, re-scan the rest.
                        let rest = self.tc_pending[rest_start..].to_string();
                        (self.inner)(StreamEvent::Text(std::mem::take(&mut self.tc_pending)));
                        self.tc_scanned = 0;
                        self.tc_name_checked = false;
                        self.tc_string_count = 0;
                        self.tc_expect_colon = false;
                        if !rest.is_empty() {
                            self.scan_tc_for_start(&rest);
                        }
                    }
                    return;
                }
            } else if b == b':' {
                self.tc_expect_colon = false;
            } else if !b.is_ascii_whitespace() {
                // Any non-whitespace, non-colon char when we expect a colon
                // after the "name" key means this isn't a JSON key-value
                // pair (e.g. prose like `{"name" patterns}`). Flush.
                if self.tc_expect_colon {
                    self.flush_tc_as_text();
                    return;
                }
            }
            i += 1;
        }
        self.tc_scanned = bytes.len();
    }

    /// Extract the tool name from the buffered JSON and check if it's valid.
    /// The buffer starts with `{"name": "…"` — the 2nd quoted string is the
    /// tool name value. We only look at the portion up to `tc_scanned` (the
    /// current scan position), not the entire buffer, because the buffer may
    /// contain text beyond the JSON object (e.g. prose after the tool call).
    fn buffered_tool_name_is_valid(&self) -> bool {
        let s = &self.tc_pending;
        // Find all quoted strings in the buffer up to the current scan
        // position. The 2nd string is the tool name value.
        let mut strings: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut in_str = false;
        let mut escape = false;
        for b in s.bytes() {
            if in_str {
                if escape {
                    escape = false;
                    current.push(b as char);
                } else if b == b'\\' {
                    escape = true;
                } else if b == b'"' {
                    in_str = false;
                    strings.push(std::mem::take(&mut current));
                } else {
                    current.push(b as char);
                }
            } else if b == b'"' {
                in_str = true;
            }
        }
        // If we're still inside a string, it's incomplete — check partial.
        if in_str && !current.is_empty() {
            // Partial value string — check if it's a prefix of a valid tool name.
            return VALID_TOOL_NAMES
                .iter()
                .any(|name| name.starts_with(&current) || current.starts_with(name));
        }
        // The 2nd complete string is the tool name value.
        if strings.len() >= 2 {
            return is_valid_tool_name(&strings[1]);
        }
        // Not enough strings yet — keep buffering.
        true
    }

    /// Flush the tool-call buffer as text and reset state.
    fn flush_tc_as_text(&mut self) {
        if !self.tc_pending.is_empty() {
            (self.inner)(StreamEvent::Text(std::mem::take(&mut self.tc_pending)));
        }
        self.tc_depth = 0;
        self.tc_in_string = false;
        self.tc_escape = false;
        self.tc_scanned = 0;
        self.tc_name_checked = false;
        self.tc_string_count = 0;
        self.tc_expect_colon = false;
    }

    /// Flush all buffered text (both special-token and tool-call buffers).
    fn flush(&mut self) {
        // Flush special-token buffer first (it feeds into the tool-call pass).
        if !self.st_pending.is_empty() {
            let st = std::mem::take(&mut self.st_pending);
            self.suppress_tool_calls(&st);
        }
        // Flush tool-call buffer — might be a partial tool call that never
        // completed (e.g. truncated output). Flush as text so the user sees it.
        if !self.tc_pending.is_empty() {
            (self.inner)(StreamEvent::Text(std::mem::take(&mut self.tc_pending)));
        }
        self.tc_depth = 0;
        self.tc_in_string = false;
        self.tc_escape = false;
        self.tc_scanned = 0;
        self.tc_name_checked = false;
        self.tc_string_count = 0;
        self.tc_expect_colon = false;
    }
}

/// Check if `s` (starting at `{`) looks like the start of a tool-call JSON:
/// `{"name"` with optional whitespace. This is intentionally broad — it
/// triggers on any `{"name"` — but the caller validates the tool name value
/// as soon as it's available and flushes immediately if it's not a real tool
/// call, so false positives (prose mentioning `{"name"`) are short-lived.
fn looks_like_tool_call_start(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return false;
    }
    let mut i = 1;
    // Skip optional whitespace.
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'"' {
        let rest = &s[i + 1..];
        if let Some(after) = rest.strip_prefix("name")
            && after.starts_with('"')
        {
            return true;
        }
    }
    false
}

/// Check if a complete JSON string is a tool call: has a `"name"` field with a
/// valid tool name. Mirrors the logic in `parse_text_tool_calls` but operates
/// on a complete string.
fn is_tool_call_json(s: &str) -> bool {
    let value: serde_json::Value = match serde_json::from_str(s) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let obj = match value.as_object() {
        Some(o) => o,
        None => return false,
    };
    let name = match obj.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => return false,
    };
    is_valid_tool_name(name)
}

/// The set of valid tool names, mirroring `heuristics::VALID_TOOL_NAMES`. Kept
/// in sync so the streaming filter suppresses the same calls the post-collection
/// parser promotes.
const VALID_TOOL_NAMES: &[&str] = &[
    "update_plan",
    "record_decision",
    "read",
    "write",
    "edit",
    "multi_edit",
    "bash",
    "bash_output",
    "bash_kill",
    "list",
    "diff",
    "grep",
    "glob",
    "apply_patch",
];

fn is_valid_tool_name(name: &str) -> bool {
    VALID_TOOL_NAMES.contains(&name)
}

/// Strip ChatML special tokens (`<|im_start|>`, `<|im_end|>`, etc.) from a
/// complete string. Unlike [`StreamingTextFilter`], this operates on the full
/// text at once so there are no partial-token concerns. Used to clean the
/// accumulated `text` before it's recorded in history.
fn strip_special_tokens(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<'
            && i + 1 < bytes.len()
            && bytes[i + 1] == b'|'
            && let Some(end) = text[i..].find(SPECIAL_TOKEN_SUFFIX)
        {
            let inner = &text[i + SPECIAL_TOKEN_PREFIX.len()..i + end];
            if is_token_identifier(inner) {
                out.push_str(&text[last..i]);
                last = i + end + SPECIAL_TOKEN_SUFFIX.len();
                i = last;
                continue;
            }
        }
        i += 1;
    }
    out.push_str(&text[last..]);
    out
}

/// Drain an OpenAI SSE stream (already reduced to `data` strings) into a
/// [`Completion`], forwarding text/reasoning/tool tokens to `sink`.
///
/// Three deadlines bound the wait, all measured from the last real token
/// (content/reasoning/tool — keep-alive heartbeats carry none, so they don't
/// reset the clock):
/// - **cold start** (no output yet): wait up to `idle` — a request can be
///   legitimately queued before the first token.
/// - **stall** (output flowed, then stopped without a finish signal): end after
///   the shorter `stall`, returning what we have. A multi-second gap *between*
///   tokens means the stream has effectively ended; this stops a provider that
///   streams a full answer then holds the socket open from hanging the turn.
/// - **finish** (`finish_reason` seen): a short [`FINISH_GRACE`] catches any
///   trailing usage chunk, then stop even if `[DONE]` never comes.
pub(crate) async fn collect_completion<S>(
    mut stream: S,
    idle: Duration,
    stall: Duration,
    sink: &mut (dyn FnMut(StreamEvent) + Send),
) -> Result<Completion>
where
    S: Stream<Item = Result<String>> + Unpin,
{
    let mut text = String::new();
    let mut tool_calls: Vec<ToolCallBuilder> = Vec::new();
    let mut completion = Completion::default();
    let mut last_progress = Instant::now();
    let mut finished: Option<Instant> = None;
    let mut progressed = false;
    let mut output_chars = 0usize;
    // Wrap the sink so ChatML special tokens (<|im_start|>, <|im_end|>, …) are
    // stripped from streamed text, and tool-call JSON (`{"name":…}`) that local
    // models emit as text is suppressed from the live display. The
    // post-collection `parse_text_tool_calls` promotes the JSON to real
    // ToolCall blocks; this filter only hides it while streaming.
    //
    // Both passes are combined into a single `StreamingTextFilter` to avoid
    // borrow-conflict chaining of two `&mut dyn FnMut` filters.
    let mut filter = StreamingTextFilter::new(sink);

    loop {
        let budget = match finished {
            Some(at) => FINISH_GRACE.saturating_sub(at.elapsed()),
            None if progressed => stall.saturating_sub(last_progress.elapsed()),
            None => idle.saturating_sub(last_progress.elapsed()),
        };
        let data = match tokio::time::timeout(budget, stream.next()).await {
            Ok(Some(Ok(data))) => data,
            // A stream *read* error mid-flight: the provider reset the connection
            // or sent a truncated frame instead of closing cleanly / sending
            // `[DONE]`. If the answer already finished, or output has flowed,
            // treat it as an unclean end of a complete response and return what
            // we have — discarding a fully-streamed answer to force a retry is
            // worse than tolerating the unclean close. (This guard mirrors the
            // timeout branches below; without it a finished response was thrown
            // away as a "malformed stream" whenever the socket RST'd after the
            // last token.) With no output yet, it's a genuine failure: propagate.
            Ok(Some(Err(err))) => {
                if finished.is_some() || progressed {
                    break;
                }
                return Err(err);
            }
            Ok(None) => break,
            // Past finish_reason the answer is complete; don't let a provider that
            // omits `[DONE]` (or never closes the socket) hang a finished turn.
            Err(_) if finished.is_some() => break,
            // Output flowed then stalled with no finish signal: treat what we have
            // as the response rather than waiting out the full cold-start timeout.
            Err(_) if progressed => break,
            Err(_) => bail!(
                "model produced no output for {}s — the provider streamed only \
                 keep-alive heartbeats. It may be overloaded or the model unavailable; \
                 try again, or switch with /model.",
                idle.as_secs()
            ),
        };
        if data == "[DONE]" {
            break;
        }
        let chunk = serde_json::from_str::<ChatChunk>(&data).with_context(|| {
            format!(
                "malformed SSE JSON chunk: {}",
                data.chars().take(160).collect::<String>()
            )
        })?;
        if let Some(error) = chunk.error {
            let message = stream_error_message(&error);
            return Err(ProviderError::new(classify_stream_api_error(&message), message).into());
        }

        if let Some(usage) = chunk.usage {
            let cached = usage
                .prompt_tokens_details
                .map(|d| d.cached_tokens)
                .unwrap_or(0);
            completion.usage = Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                cache_read_tokens: cached,
                cache_creation_tokens: 0,
                // OpenAI's `prompt_tokens` already includes the cached subset
                // reported in `cached_tokens`, so the context window occupancy
                // is exactly `prompt_tokens` — not `prompt_tokens + cached`.
                input_includes_cache: true,
                context_occupancy: usage.prompt_tokens,
                // Normalized billable breakdown: prompt_tokens is the total, so
                // regular input is the non-cached remainder. No cache creation
                // is reported by OpenAI.
                billable: Some(BillableBreakdown {
                    regular_input: usage.prompt_tokens.saturating_sub(cached),
                    cached_input: cached,
                    cache_creation: 0,
                    output: usage.completion_tokens,
                }),
            };
        }
        for choice in chunk.choices {
            let delta = choice.delta;
            if let Some(reasoning) = delta.reasoning.or(delta.reasoning_content)
                && !reasoning.is_empty()
            {
                output_chars += reasoning.len();
                filter.forward(StreamEvent::Reasoning(reasoning));
                last_progress = Instant::now();
                progressed = true;
            }
            if let Some(content) = delta.content
                && !content.is_empty()
            {
                output_chars += content.len();
                text.push_str(&content);
                filter.text(&content);
                last_progress = Instant::now();
                progressed = true;
            }
            if let Some(deltas) = delta.tool_calls {
                last_progress = Instant::now();
                progressed = true;
                for tcd in deltas {
                    if tool_calls.len() <= tcd.index {
                        tool_calls.resize_with(tcd.index + 1, ToolCallBuilder::default);
                    }
                    let builder = &mut tool_calls[tcd.index];
                    if let Some(id) = tcd.id
                        && !id.is_empty()
                    {
                        builder.id = id;
                    }
                    if let Some(func) = tcd.function {
                        if let Some(name) = func.name {
                            output_chars += name.len();
                            builder.name.push_str(&name);
                        }
                        if let Some(args) = func.arguments {
                            output_chars += args.len();
                            builder.arguments.push_str(&args);
                        }
                    }
                }
            }
            if let Some(finish_reason) = choice.finish_reason {
                completion.stop_reason = Some(finish_reason);
                finished.get_or_insert_with(Instant::now);
            }
        }
    }

    // Flush any buffered partial token from the streaming filter (text that
    // looked like the start of a special token but never resolved into one).
    filter.flush();

    // Strip special tokens from the accumulated text so recorded history stays
    // clean. The streaming filter already removed them from the live display,
    // but the `text` accumulator holds the raw content.
    let text = strip_special_tokens(&text);
    if !text.is_empty() {
        completion.content.push(Content::Text(text));
    }
    for builder in tool_calls {
        if !builder.name.is_empty() {
            completion.content.push(builder.finish());
        }
    }
    if completion.usage.output_tokens == 0 && output_chars > 0 {
        completion.usage.output_tokens = output_chars.div_ceil(4) as u64;
    }
    Ok(completion)
}

pub(crate) fn classify_stream_error(err: anyhow::Error) -> ProviderError {
    if let Some(provider) = err.downcast_ref::<ProviderError>() {
        return ProviderError::new(provider.kind, provider.message.clone())
            .with_usage(provider.usage);
    }
    let text = err.to_string();
    let kind = if text.contains("no output") {
        ProviderErrorKind::StreamTimeout
    } else if text.contains("error reading stream") || text.contains("malformed SSE JSON chunk") {
        ProviderErrorKind::MalformedStream
    } else {
        ProviderErrorKind::Other
    };
    ProviderError::new(kind, text)
}

fn stream_error_message(error: &Value) -> String {
    match error {
        Value::String(message) => message.clone(),
        Value::Object(object) => object
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| object.get("error").and_then(Value::as_str))
            .or_else(|| object.get("code").and_then(Value::as_str))
            .map(str::to_string)
            .unwrap_or_else(|| error.to_string()),
        _ => error.to_string(),
    }
}

fn classify_stream_api_error(message: &str) -> ProviderErrorKind {
    let lower = message.to_ascii_lowercase();
    if lower.contains("rate limit") || lower.contains("too many requests") || lower.contains("429")
    {
        ProviderErrorKind::RateLimit
    } else if lower.contains("capacity")
        || lower.contains("temporarily unavailable")
        || lower.contains("service unavailable")
        || lower.contains("no route")
        || lower.contains("overloaded")
        || lower.contains("cooling down")
        || lower.contains("first_token_stall")
        || lower.contains("first token")
    {
        ProviderErrorKind::Outage
    } else {
        ProviderErrorKind::Other
    }
}

pub(crate) fn backfill_missing_usage(
    completion: &mut Completion,
    request: &crate::types::ChatRequest,
) {
    if completion.usage.input_tokens == 0 {
        completion.usage.input_tokens = estimate_messages_tokens(&request.messages);
    }
    if completion.usage.output_tokens == 0 {
        completion.usage.output_tokens =
            crate::types::estimate_completion_output_tokens(&completion.content);
    }
}

#[derive(Default)]
struct ToolCallBuilder {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallBuilder {
    fn finish(self) -> Content {
        Content::ToolCall {
            id: self.id,
            name: self.name,
            arguments: if self.arguments.is_empty() {
                "{}".into()
            } else {
                self.arguments
            },
        }
    }
}

// --- Streaming response shapes -------------------------------------------

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    /// OpenAI reports automatic prefix-cache hits here.
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokenDetails>,
}

#[derive(Deserialize)]
struct PromptTokenDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use anyhow::Result;
    use futures_util::{StreamExt, stream};

    use super::collect_completion;
    use crate::types::{Content, StreamEvent};

    /// A stream of SSE `data` strings that never ends (no `[DONE]`, socket stays
    /// open) — `pending()` models a provider that just stops talking.
    fn never_ending(events: Vec<&str>) -> impl futures_util::Stream<Item = Result<String>> + Unpin {
        let items: Vec<Result<String>> = events.into_iter().map(|s| Ok(s.to_string())).collect();
        stream::iter(items).chain(stream::pending())
    }

    /// A stream that yields the given `data` strings, then a read error — models
    /// a provider that resets the connection (or sends a truncated frame)
    /// instead of closing cleanly or sending `[DONE]`.
    fn errors_after(events: Vec<&str>) -> impl futures_util::Stream<Item = Result<String>> + Unpin {
        let mut items: Vec<Result<String>> =
            events.into_iter().map(|s| Ok(s.to_string())).collect();
        items.push(Err(anyhow::anyhow!("error reading stream")));
        stream::iter(items)
    }

    const STALL: Duration = Duration::from_secs(15);
    const IDLE: Duration = Duration::from_secs(120);

    #[tokio::test(start_paused = true)]
    async fn stops_after_finish_reason_without_done() {
        // The bug: pipenetwork.ai sends `finish_reason` then neither `[DONE]` nor a
        // socket close, so a finished answer used to spin until the 120s idle
        // timeout. Now the short finish-grace ends the turn promptly.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"the answer"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        assert_eq!(completion.stop_reason.as_deref(), Some("stop"));
        assert!(
            matches!(completion.content.first(), Some(Content::Text(t)) if t == "the answer"),
            "text is still collected: {:?}",
            completion.content
        );
    }

    #[tokio::test(start_paused = true)]
    async fn trailing_usage_chunk_after_finish_is_captured() {
        // Providers send the usage chunk right after `finish_reason`; the grace
        // window must be long enough to catch it.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}"#,
        ]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        assert_eq!(completion.usage.input_tokens, 10);
        assert_eq!(completion.usage.output_tokens, 2);
    }

    #[tokio::test(start_paused = true)]
    async fn returns_partial_when_stream_stalls_after_content() {
        // The reported bug: the model streams a full answer, then the provider
        // sends no finish_reason / [DONE] and holds the socket open. Once output
        // has flowed, the short stall window ends the turn with what we have,
        // instead of spinning out the full cold-start idle timeout.
        let stream = never_ending(vec![r#"{"choices":[{"delta":{"content":"the answer"}}]}"#]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        assert!(
            matches!(completion.content.first(), Some(Content::Text(t)) if t == "the answer"),
            "the streamed answer is returned: {:?}",
            completion.content
        );
    }

    #[tokio::test(start_paused = true)]
    async fn returns_content_when_stream_errors_after_finish() {
        // The reported "the model's response didn't come through cleanly" bug:
        // the model streams a full answer and `finish_reason`, then the provider
        // resets the connection instead of a clean close / `[DONE]`. The complete
        // answer must be returned, not discarded as a malformed stream and retried.
        let stream = errors_after(vec![
            r#"{"choices":[{"delta":{"content":"the answer"},"finish_reason":"stop"}]}"#,
        ]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        assert_eq!(completion.stop_reason.as_deref(), Some("stop"));
        assert!(
            matches!(completion.content.first(), Some(Content::Text(t)) if t == "the answer"),
            "the streamed answer survives an unclean close: {:?}",
            completion.content
        );
    }

    #[tokio::test(start_paused = true)]
    async fn returns_content_when_stream_errors_after_content_without_finish() {
        // Output flowed (no finish_reason yet), then the socket errored. We've
        // received real content, so return it rather than discarding it — same
        // policy as a post-content stall.
        let stream = errors_after(vec![r#"{"choices":[{"delta":{"content":"partial"}}]}"#]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        assert!(
            matches!(completion.content.first(), Some(Content::Text(t)) if t == "partial"),
            "content received before the error is kept: {:?}",
            completion.content
        );
    }

    #[tokio::test(start_paused = true)]
    async fn propagates_stream_error_before_any_output() {
        // A read error before any token is a genuine failure with nothing to
        // salvage — propagate it (the caller decides whether to retry).
        let stream = errors_after(vec![]);
        let mut sink = |_: StreamEvent| {};
        let err = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("error reading stream"),
            "got: {err}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn bails_when_no_output_at_all() {
        // Nothing ever streamed (cold-start): the long idle timeout trips and we
        // surface the "no output" error rather than returning an empty success.
        let stream = never_ending(vec![]);
        let mut sink = |_: StreamEvent| {};
        let err = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no output"), "got: {err}");
    }

    #[tokio::test(start_paused = true)]
    async fn fragmented_tool_calls_are_reassembled() {
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"ba","arguments":"{\"com"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"sh","arguments":"mand\":\"echo hi\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        ]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let calls = completion.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments, r#"{"command":"echo hi"}"#);
    }

    #[tokio::test(start_paused = true)]
    async fn cached_tokens_do_not_double_count_context_occupancy() {
        // OpenAI's `prompt_tokens` already includes the cached subset reported
        // in `cached_tokens`, so context_occupancy must equal prompt_tokens —
        // not prompt_tokens + cached_tokens (the double-counting bug this field
        // replaces).
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":400}}}"#,
        ]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        assert_eq!(completion.usage.input_tokens, 1000);
        assert_eq!(completion.usage.cache_read_tokens, 400);
        assert_eq!(
            completion.usage.context_occupancy, 1000,
            "context_occupancy == prompt_tokens, not prompt_tokens + cached"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn strips_chatml_special_tokens_from_streamed_text() {
        // Some local models (Qwen, Yi, etc.) emit ChatML special tokens like
        // <|im_start|> and <|im_end|> as raw text content. They must be stripped
        // from both the live stream and the recorded completion text.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"Let me check.\n<|im_start|>\n"}}]}"#,
            r#"{"choices":[{"delta":{"content":"{\"name\":\"bash\",\"arguments\":\"{}\"}\n<|im_end|>\nDone."}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mut collected = Vec::new();
        let mut sink = |e: StreamEvent| {
            if let StreamEvent::Text(t) = e {
                collected.push(t);
            }
        };
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let streamed = collected.join("");
        assert!(
            !streamed.contains("<|im_start|>") && !streamed.contains("<|im_end|>"),
            "special tokens must not appear in streamed text: {streamed:?}"
        );
        let text = completion
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        assert!(
            !text.contains("<|im_start|>") && !text.contains("<|im_end|>"),
            "special tokens must not appear in recorded text: {text:?}"
        );
        assert!(
            text.contains("Let me check.") && text.contains("Done."),
            "real text must survive: {text:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn strips_special_token_split_across_chunks() {
        // A special token can be split across streaming chunks: `<|im_` in one,
        // `start|>` in the next. The filter must buffer and resolve it.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"before <|im_"}}]}"#,
            r#"{"choices":[{"delta":{"content":"start|> after"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mut collected = Vec::new();
        let mut sink = |e: StreamEvent| {
            if let StreamEvent::Text(t) = e {
                collected.push(t);
            }
        };
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let streamed = collected.join("");
        assert!(
            !streamed.contains("<|im_start|>"),
            "split token must not appear in stream: {streamed:?}"
        );
        assert!(
            streamed.contains("before ") && streamed.contains(" after"),
            "surrounding text must survive: {streamed:?}"
        );
        let text = completion
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(text, "before  after");
    }

    #[tokio::test(start_paused = true)]
    async fn preserves_literal_angle_pipe_text() {
        // Text that looks like `<|foo bar|>` (with spaces) is NOT a special
        // token — it must be preserved as literal text.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"a <|foo bar|> b"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let text = completion
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(text, "a <|foo bar|> b");
    }

    #[tokio::test(start_paused = true)]
    async fn flushes_unresolved_partial_token_as_text() {
        // If the stream ends with a pending `<|` that never resolved into a
        // complete special token, it must be flushed as literal text.
        let stream = errors_after(vec![r#"{"choices":[{"delta":{"content":"hello <|"}}]}"#]);
        let mut collected = Vec::new();
        let mut sink = |e: StreamEvent| {
            if let StreamEvent::Text(t) = e {
                collected.push(t);
            }
        };
        let _completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let streamed = collected.join("");
        assert_eq!(streamed, "hello <|");
    }

    // ── Tool-call JSON suppression ──

    #[tokio::test(start_paused = true)]
    async fn suppresses_tool_call_json_from_streamed_text() {
        // A local model emits a tool call as text content. The raw JSON must
        // NOT appear in the live stream — the user should see only the prose.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"Let me check.\n"}}]}"#,
            r#"{"choices":[{"delta":{"content":"{\"name\": \"list\", \"arguments\": {}}"}}]}"#,
            r#"{"choices":[{"delta":{"content":"\nDone."}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mut collected = Vec::new();
        let mut sink = |e: StreamEvent| {
            if let StreamEvent::Text(t) = e {
                collected.push(t);
            }
        };
        let _completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let streamed = collected.join("");
        assert!(
            !streamed.contains("\"name\""),
            "tool-call JSON must not appear in stream: {streamed:?}"
        );
        assert!(
            streamed.contains("Let me check"),
            "prose before tool call preserved: {streamed:?}"
        );
        assert!(
            streamed.contains("Done"),
            "prose after tool call preserved: {streamed:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn suppresses_tool_call_json_split_across_chunks() {
        // The tool-call JSON is split across many small chunks — the filter
        // must buffer until the closing `}` and suppress the whole thing.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"{\"name\":"}}]}"#,
            r#"{"choices":[{"delta":{"content":" \"edit\","}}]}"#,
            r#"{"choices":[{"delta":{"content":" \"arguments\":"}}]}"#,
            r#"{"choices":[{"delta":{"content":" {\"path\":"}}]}"#,
            r#"{"choices":[{"delta":{"content":" \"a.rs\"}}}"}}]}"#,
            r#"{"choices":[{"delta":{"content":"All done."}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mut collected = Vec::new();
        let mut sink = |e: StreamEvent| {
            if let StreamEvent::Text(t) = e {
                collected.push(t);
            }
        };
        let _completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let streamed = collected.join("");
        assert!(
            !streamed.contains("\"name\"") && !streamed.contains("\"edit\""),
            "tool-call JSON must not appear in stream: {streamed:?}"
        );
        assert!(
            streamed.contains("All done."),
            "trailing prose preserved: {streamed:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn preserves_non_tool_json_in_streamed_text() {
        // JSON that isn't a tool call (no "name" key or unknown tool name)
        // must be preserved as literal text in the stream.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"Result: {\"foo\": 42} ok"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mut collected = Vec::new();
        let mut sink = |e: StreamEvent| {
            if let StreamEvent::Text(t) = e {
                collected.push(t);
            }
        };
        let _completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let streamed = collected.join("");
        assert!(
            streamed.contains("{\"foo\": 42}"),
            "non-tool JSON must appear in stream: {streamed:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn suppresses_multiple_tool_calls_in_stream() {
        // Multiple tool calls in one response — all should be suppressed.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"Starting.\n{\"name\": \"list\", \"arguments\": {}}\nMiddle.\n{\"name\": \"read\", \"arguments\": {\"path\": \"a.rs\"}}\nEnd."}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mut collected = Vec::new();
        let mut sink = |e: StreamEvent| {
            if let StreamEvent::Text(t) = e {
                collected.push(t);
            }
        };
        let _completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let streamed = collected.join("");
        assert!(
            !streamed.contains("\"name\""),
            "no tool-call JSON in stream: {streamed:?}"
        );
        assert!(
            streamed.contains("Starting.")
                && streamed.contains("Middle.")
                && streamed.contains("End."),
            "prose between calls preserved: {streamed:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn preserves_prose_containing_name_key_pattern() {
        // Prose that mentions {"name" but isn't a tool call (e.g. the model
        // describing the filter) must NOT be suppressed. The early-exit logic
        // detects that the "name" value isn't a valid tool name and flushes.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"The filter detects {\"name\" patterns in the stream and suppresses them."}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mut collected = Vec::new();
        let mut sink = |e: StreamEvent| {
            if let StreamEvent::Text(t) = e {
                collected.push(t);
            }
        };
        let _completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let streamed = collected.join("");
        assert!(
            streamed.contains("patterns"),
            "prose with {{\"name\"}} pattern must survive: {streamed:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn preserves_prose_with_name_colon_non_tool_value() {
        // `{"name": "something"}` where "something" isn't a valid tool name
        // must be preserved — the early-exit validates the tool name value.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"See {\"name\": \"not_a_tool\", \"x\": 1} here."}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mut collected = Vec::new();
        let mut sink = |e: StreamEvent| {
            if let StreamEvent::Text(t) = e {
                collected.push(t);
            }
        };
        let _completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let streamed = collected.join("");
        assert!(
            streamed.contains("not_a_tool"),
            "non-tool JSON with name key must survive: {streamed:?}"
        );
    }
}
