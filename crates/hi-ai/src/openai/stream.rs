//! Streaming response parsing: drain an OpenAI SSE stream into a
//! [`Completion`], reassemble fragmented tool calls, and parse usage chunks.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use futures_util::{Stream, StreamExt};
use serde::Deserialize;

use crate::provider::{ProviderError, ProviderErrorKind};
use crate::types::{BillableBreakdown, Completion, Content, StreamEvent, Usage, estimate_messages_tokens};

/// Once the model reports a `finish_reason`, it has stopped generating; we wait
/// only this long for the trailing usage chunk / `[DONE]` before ending the
/// turn. Without it, a provider that emits `finish_reason` but never sends
/// `[DONE]` (nor closes the socket) would wedge the turn until the much longer
/// idle timeout expires — a completed answer left spinning for ~2 minutes.
const FINISH_GRACE: Duration = Duration::from_secs(3);

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
                sink(StreamEvent::Reasoning(reasoning));
                last_progress = Instant::now();
                progressed = true;
            }
            if let Some(content) = delta.content
                && !content.is_empty()
            {
                output_chars += content.len();
                text.push_str(&content);
                sink(StreamEvent::Text(content));
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

pub(crate) fn backfill_missing_usage(completion: &mut Completion, request: &crate::types::ChatRequest) {
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
        let mut items: Vec<Result<String>> = events.into_iter().map(|s| Ok(s.to_string())).collect();
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
}
