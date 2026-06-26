//! OpenAI Chat Completions adapter.
//!
//! Covers OpenRouter, pipenetwork.ai, and local servers (Ollama, llama.cpp,
//! LM Studio, vLLM) — they differ only by base URL and API key.
//!
//! Request translation lives in [`request`], and SSE stream parsing in
//! [`stream`]; this module holds the [`OpenAiProvider`] struct and its
//! [`Provider`] impl, which wires the two together.

mod request;
mod stream;

use anyhow::{Context, Result};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;

use crate::provider::{Provider, ProviderError, ProviderErrorKind};
use crate::types::{
    ChatRequest, CompatMode, Completion, StreamEvent, Usage, estimate_messages_tokens,
};

pub struct OpenAiProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OpenAiProvider {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            http: crate::http::agent_http_client(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        }
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let url = format!("{}/chat/completions", self.base_url);
        let attempts = request::request_attempts(&request);
        let mut last_error: Option<ProviderError> = None;
        let mut resp = None;
        let mut idx = 0;
        while idx < attempts.len() {
            let attempt = attempts[idx];
            let body = request::build_body(&request, attempt);
            let response = crate::http::send_with_retry(
                self.http.post(&url).bearer_auth(&self.api_key).json(&body),
            )
            .await
            .context("request to model endpoint failed")?;

            if response.status().is_success() {
                if let Some(status) = attempt.status {
                    sink(StreamEvent::Status(status.into()));
                }
                resp = Some(response);
                break;
            }

            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let kind = request::classify_http_error(status, &text);
            last_error = Some(ProviderError::new(
                kind,
                format!("API error {status}: {text}"),
            ));
            if request.profile.compat == CompatMode::Strict {
                break;
            }
            // Degrade toward the attempt that actually addresses this error.
            // Tool rejection is surfaced: an agent turn that advertised tools
            // cannot safely continue chat-only because it would be unable to
            // inspect or modify the workspace.
            match request::next_degraded_attempt(&attempts, idx, kind, &text) {
                Some(next) => idx = next,
                None => break,
            }
        }
        let Some(resp) = resp else {
            return Err(last_error
                .unwrap_or_else(|| {
                    ProviderError::new(ProviderErrorKind::Other, "request failed before streaming")
                })
                .into());
        };

        // `debug_tap` optionally echoes the raw wire bytes when HI_DEBUG_STREAM
        // is set. Reduce the stream to its SSE `data` strings so the collection
        // loop is provider-agnostic and unit-testable.
        let stream = crate::http::debug_tap(resp.bytes_stream())
            .eventsource()
            .map(|res| res.map(|event| event.data).context("error reading stream"));
        let mut completion = stream::collect_completion(
            Box::pin(stream),
            crate::http::stream_idle_timeout(),
            crate::http::stream_stall_timeout(),
            sink,
        )
        .await
        .map_err(|err| {
            stream::classify_stream_error(err).with_usage(Usage {
                input_tokens: estimate_messages_tokens(&request.messages),
                output_tokens: request.max_tokens as u64,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                input_includes_cache: true,
                context_occupancy: estimate_messages_tokens(&request.messages),
                billable: None,
            })
        })?;
        stream::backfill_missing_usage(&mut completion, &request);
        if completion.content.is_empty() {
            return Err(ProviderError::new(
                ProviderErrorKind::EmptyCompletion,
                "model returned an empty completion",
            )
            .with_usage(completion.usage)
            .into());
        }
        Ok(completion)
    }

    async fn list_models(&self) -> Result<Vec<crate::provider::ServedModel>> {
        let url = format!("{}/models", self.base_url);
        crate::http::fetch_models(self.http.get(&url).bearer_auth(&self.api_key)).await
    }
}

#[cfg(test)]
mod tests {
    use super::OpenAiProvider;
    use crate::provider::{Provider, ProviderErrorKind, provider_error_kind};
    use crate::test_support::{FakeOpenAiServer, Response, sse_text};
    use crate::types::{
        ChatRequest, Content, Message, RequestProfile, StreamEvent, ToolMode, ToolSpec,
    };

    #[tokio::test]
    async fn fake_server_rejects_stream_options_then_succeeds() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(400, r#"{"error":"stream_options unsupported"}"#),
            Response::sse(sse_text("ok")),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let request = request(vec![], Default::default());
        let mut statuses = Vec::new();
        let mut sink = |event| {
            if let StreamEvent::Status(status) = event {
                statuses.push(status);
            }
        };
        let completion = provider.stream(request, &mut sink).await.unwrap();
        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "ok"));
        assert!(
            completion.usage.input_tokens > 0,
            "fallback request gets estimated input usage: {:?}",
            completion.usage
        );
        assert!(
            completion.usage.output_tokens > 0,
            "fallback request gets estimated output usage: {:?}",
            completion.usage
        );
        assert!(
            statuses.iter().any(|s| s.contains("stream_options")),
            "{statuses:?}"
        );
        let bodies = server.bodies();
        assert!(bodies[0].contains("stream_options"));
        assert!(!bodies[1].contains("stream_options"));
    }

    #[tokio::test]
    async fn fake_server_rejects_tools_fails_fast() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(
            400,
            r#"{"error":"tools unsupported"}"#,
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![tool()], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::UnsupportedTools)
        );
        let bodies = server.bodies();
        assert_eq!(bodies.len(), 1);
        assert!(bodies[0].contains("\"tools\""));
    }

    #[tokio::test]
    async fn required_tool_mode_does_not_degrade() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(
            400,
            r#"{"error":"tools unsupported"}"#,
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let profile = RequestProfile {
            tool_mode: ToolMode::Required,
            ..Default::default()
        };
        let err = provider
            .stream(request(vec![tool()], profile), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::UnsupportedTools)
        );
        assert_eq!(server.bodies().len(), 1);
    }

    #[tokio::test]
    async fn auth_rate_limit_and_malformed_stream_are_classified() {
        for (status, kind) in [
            (401, ProviderErrorKind::Auth),
            (403, ProviderErrorKind::Auth),
            (429, ProviderErrorKind::RateLimit),
        ] {
            let Some(server) =
                FakeOpenAiServer::new(vec![Response::json(status, r#"{"error":"nope"}"#)])
            else {
                return;
            };
            let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
            let err = provider
                .stream(request(vec![], Default::default()), &mut |_| {})
                .await
                .unwrap_err();
            assert_eq!(provider_error_kind(&err), Some(kind), "status {status}");
        }

        let Some(server) = FakeOpenAiServer::new(vec![Response::sse("data: {not-json}\n\n")])
        else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::MalformedStream)
        );
    }

    #[tokio::test]
    async fn request_too_large_400_is_classified() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(
            400,
            r#"{"error":"chat input exceeds the maximum allowed size of 131072 bytes","error_type":"invalid_request_error"}"#,
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::RequestTooLarge)
        );
    }

    #[tokio::test]
    async fn server_error_retries_then_succeeds() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(500, r#"{"error":"temporary outage"}"#),
            Response::sse(sse_text("recovered")),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let completion = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap();
        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "recovered"));
        assert_eq!(server.bodies().len(), 2);
    }

    #[tokio::test]
    async fn empty_completion_error_carries_usage() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":3}}\n\n",
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::EmptyCompletion)
        );
        assert_eq!(crate::provider::provider_error_usage(&err).input_tokens, 42);
        assert_eq!(crate::provider::provider_error_usage(&err).output_tokens, 3);
    }

    #[tokio::test]
    async fn streamed_error_payload_is_not_reported_as_empty_completion() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(
            "data: {\"error\":{\"message\":\"capacity temporarily unavailable\"}}\n\ndata: [DONE]\n\n",
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(provider_error_kind(&err), Some(ProviderErrorKind::Outage));
        assert!(
            err.to_string().contains("capacity temporarily unavailable"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn fake_server_stream_can_finish_without_done() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(
            "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n",
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let completion = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap();
        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "done"));
    }

    fn request(tools: Vec<ToolSpec>, profile: RequestProfile) -> ChatRequest {
        ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("hi")].into(),
            tools: tools.into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            profile,
        }
    }

    fn tool() -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Run shell command".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                },
                "required": ["command"]
            }),
        }
    }
}
