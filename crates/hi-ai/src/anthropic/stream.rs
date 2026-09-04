//! Collect native Messages events without treating interrupted tool batches as complete.

use anyhow::{Context, Result};
use eventsource_stream::Event;
use futures_util::{Stream, StreamExt};
use serde_json::Value;

use super::{BlockBuilder, MAX_CONTENT_BLOCKS, backfill_missing_usage};
use crate::{
    ChatRequest, Completion, Content, ProviderError, ProviderErrorKind, StreamEvent,
    ToolCallChannel,
};

pub(super) async fn collect_completion<S, E>(
    mut stream: S,
    request: &ChatRequest,
    sink: &mut (dyn FnMut(StreamEvent) + Send),
) -> Result<Completion>
where
    S: Stream<Item = std::result::Result<Event, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    let mut blocks: Vec<Option<BlockBuilder>> = Vec::new();
    let mut completion = Completion::default();
    let mut stream_complete = false;
    let mut progressed = false;
    // Zero is a valid provider value for either field: a fully cached
    // Anthropic prompt can have `input_tokens == 0`, and an empty reply
    // can have `output_tokens == 0`. Track field presence separately so
    // the heuristic fallback never overwrites authoritative zeros.
    let mut input_usage_seen = false;
    let mut output_usage_seen = false;

    loop {
        let Some(event) = stream.next().await else {
            break;
        };
        let event = match event {
            Ok(event) => event,
            // Mirror the OpenAI path: an unclean mid-stream close AFTER the
            // answer finished or after content has already streamed must not
            // discard a (near-)complete response and force a full re-bill —
            // return what we have (the input tokens from `message_start` are
            // already in `completion.usage`; output is estimated below). With
            // no progress yet it's a genuine failure: propagate.
            Err(err) => {
                if stream_complete || progressed {
                    break;
                }
                return Err(err).context("error reading stream");
            }
        };
        let Ok(data) = serde_json::from_str::<Value>(&event.data) else {
            continue;
        };

        match event.event.as_str() {
            "message_start" => {
                if let Some(tokens) = data["message"]["usage"]["input_tokens"].as_u64() {
                    input_usage_seen = true;
                    completion.usage.input_tokens = tokens;
                }
                if let Some(tokens) = data["message"]["usage"]["cache_read_input_tokens"].as_u64() {
                    completion.usage.cache_read_tokens = tokens;
                }
                if let Some(tokens) =
                    data["message"]["usage"]["cache_creation_input_tokens"].as_u64()
                {
                    completion.usage.cache_creation_tokens = tokens;
                }
                // Anthropic reports cache tokens separately from
                // `input_tokens`, so the full context window occupancy is
                // the sum of all three. Saturating: the counts come straight
                // off the wire, so a corrupt frame can't overflow-panic here.
                completion.usage.context_occupancy = completion
                    .usage
                    .input_tokens
                    .saturating_add(completion.usage.cache_read_tokens)
                    .saturating_add(completion.usage.cache_creation_tokens);
            }
            "content_block_start" => {
                let index = data["index"].as_u64().unwrap_or(0) as usize;
                // The index comes straight off the wire — bound it so a
                // corrupt frame can't force a huge `resize_with` allocation.
                if index >= MAX_CONTENT_BLOCKS {
                    continue;
                }
                if blocks.len() <= index {
                    blocks.resize_with(index + 1, || None);
                }
                blocks[index] = Some(BlockBuilder::start(&data["content_block"]));
            }
            "content_block_delta" => {
                let index = data["index"].as_u64().unwrap_or(0) as usize;
                if let Some(Some(builder)) = blocks.get_mut(index) {
                    builder.apply_delta(index, &data["delta"], sink);
                    progressed = true;
                }
            }
            "message_delta" => {
                if let Some(reason) = data["delta"]["stop_reason"].as_str() {
                    completion.stop_reason = Some(reason.to_string());
                    stream_complete = true;
                }
                if let Some(tokens) = data["usage"]["output_tokens"].as_u64() {
                    output_usage_seen = true;
                    completion.usage.output_tokens = tokens;
                }
            }
            "error" => {
                let message = data["error"]["message"].as_str().unwrap_or("unknown error");
                let error_type = data["error"]["type"].as_str().unwrap_or("");
                let kind = match error_type {
                    "overloaded_error" | "rate_limit_error" => ProviderErrorKind::RateLimit,
                    "authentication_error" => ProviderErrorKind::Auth,
                    "invalid_request_error" => ProviderErrorKind::UnsupportedRequestShape,
                    _ => ProviderErrorKind::Other,
                };
                return Err(
                    ProviderError::new(kind, format!("Anthropic stream error: {message}"))
                        .with_usage(completion.usage)
                        .into(),
                );
            }
            _ => {}
        }
        if stream_complete {
            break;
        }
    }

    // A final block can contain valid JSON before the message finishes. Preserve
    // the partial response for usage/recovery, but never authorize that tool batch
    // as complete merely because the connection closed successfully.
    if !stream_complete {
        completion.stop_reason = Some("length".into());
    }
    completion.content = blocks
        .into_iter()
        .flatten()
        .filter_map(BlockBuilder::finish)
        .collect();
    if completion
        .content
        .iter()
        .any(|content| matches!(content, Content::ToolCall { .. }))
    {
        completion.tool_call_channel = ToolCallChannel::Native;
    }
    backfill_missing_usage(
        &mut completion,
        request,
        input_usage_seen,
        output_usage_seen,
    );
    // Keep the occupancy gauge alive on the estimate path too (matches the
    // OpenAI path's backfill): a proxy that omits `message_start` usage
    // would otherwise leave it at 0 all session.
    if completion.usage.context_occupancy == 0 {
        completion.usage.context_occupancy = completion.usage.input_tokens;
    }
    Ok(completion)
}
