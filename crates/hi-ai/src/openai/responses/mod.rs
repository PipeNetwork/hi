//! Stateless OpenAI Responses transport for GPT-6 Astra.
//!
//! Keep the generic Chat Completions compatibility ladder separate: Astra
//! function calls require Responses and must never fall back to text tools.

mod request;
mod stream;
#[cfg(test)]
mod tests;

use anyhow::{Context, Result};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;

use super::{
    OpenAiProvider, canonical_request_id, rate_limits_from_headers, retry_after_header_seconds,
};
use crate::provider::{ProviderError, ProviderErrorKind};
use crate::{ChatRequest, Completion, Content, StreamEvent, Usage};

pub(super) fn supports_model(model: &str) -> bool {
    let model = model.strip_prefix("openai/").unwrap_or(model);
    model == "gpt-6-astra"
}

pub(super) fn capabilities(model: &str) -> Option<crate::ProviderCapabilities> {
    supports_model(model).then(|| {
        let mut capabilities = crate::ProviderCapabilities::native_tools(true);
        capabilities.parallel_tool_calls = true;
        capabilities.modalities.image_input = true;
        capabilities.reasoning_replay.signed_or_encrypted = true;
        capabilities.usage_reporting = crate::UsageReporting::Final;
        capabilities.cancellation = crate::CancellationSupport::TransportAbort;
        capabilities
    })
}

impl OpenAiProvider {
    pub(super) async fn stream_responses(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let mut body = request::build_body(&request);
        if let Some(metadata) = self.request_metadata(&request) {
            body["metadata"] = metadata;
        }
        let url = format!("{}/responses", self.base_url);
        let correlation_id = canonical_request_id(request.request_id.as_deref());
        let mut auth_refreshed = false;
        loop {
            let response = match self
                .dispatch_chat(&url, &body, &correlation_id, &request.execution, sink)
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    sink(StreamEvent::WireAudit(Box::new(request::wire_audit(
                        &request,
                        &self.base_url,
                        &body,
                        false,
                        None,
                    ))));
                    return Err(error.into());
                }
            };
            let status = response.status();
            sink(StreamEvent::WireAudit(Box::new(request::wire_audit(
                &request,
                &self.base_url,
                &body,
                status.is_success(),
                Some(status.as_u16()),
            ))));
            let rate_limits = rate_limits_from_headers(response.headers());
            if status.is_success() {
                super::persist_x402_credit_token(&self.auth, response.headers()).await;
                let events = crate::http::idle_guard(
                    crate::http::debug_tap(response.bytes_stream()),
                    crate::http::stream_idle_window(),
                )
                .eventsource()
                .map(|event| {
                    event
                        .map_err(|error| anyhow::anyhow!(error))
                        .context("error reading Responses stream")
                });
                let mut completion = stream::collect_completion(Box::pin(events), sink)
                    .await
                    .map_err(|error| {
                        let mut error = super::stream::classify_stream_error(error);
                        if error.usage.is_zero() {
                            let tokens = crate::types::estimate_request_input_tokens(
                                &request.messages,
                                &request.tools,
                            );
                            error.usage = Usage {
                                input_tokens: tokens,
                                context_occupancy: tokens,
                                input_includes_cache: true,
                                estimated: true,
                                ..Default::default()
                            };
                        }
                        error.usage.rate_limits = error.usage.rate_limits.or(rate_limits);
                        error
                    })?;
                // Do not replace legitimate zero counts when usage was reported.
                if completion.usage.estimated {
                    super::stream::backfill_missing_usage(&mut completion, &request);
                }
                completion.usage.rate_limits = completion.usage.rate_limits.or(rate_limits);
                let visible = completion.content.iter().any(|content| match content {
                    Content::Text(text) | Content::Thinking { text, .. } => !text.is_empty(),
                    Content::ToolCall { .. } => true,
                    _ => false,
                });
                if !visible
                    && completion.refusal.is_none()
                    && !matches!(
                        completion.stop_reason.as_deref(),
                        Some("max_tokens" | "content_filter" | "refusal")
                    )
                {
                    return Err(ProviderError::new(
                        ProviderErrorKind::EmptyCompletion,
                        "model returned an empty completion",
                    )
                    .with_usage(completion.usage)
                    .into());
                }
                return Ok(completion);
            }
            let retry_after = retry_after_header_seconds(&response);
            let text = response.text().await.unwrap_or_default();
            let parsed = super::request::parse_api_error(Some(status), &text);
            let explicit_retryable = parsed.explicit_retryable;
            let mut error = parsed.into_provider_error(Some(status));
            if explicit_retryable != Some(false)
                && error.kind == ProviderErrorKind::Auth
                && !crate::is_billing_or_quota_text(&text)
                && !auth_refreshed
                && self.auth.refresh().await
            {
                auth_refreshed = true;
                sink(StreamEvent::Status(
                    "credential expired; refreshed it — retrying".into(),
                ));
                continue;
            }
            error.retry_after_seconds = error.retry_after_seconds.or(retry_after);
            error.usage.rate_limits = error.usage.rate_limits.or(rate_limits);
            return Err(error.into());
        }
    }
}
