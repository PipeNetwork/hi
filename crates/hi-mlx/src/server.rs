use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::backend::{GenerationRequest, SharedBackend};
use crate::manifest::ModelInfo;
use crate::prompt::build_prompt;
use crate::tool_parser::{NormalizedToolCall, parse_tool_calls};

#[derive(Clone)]
pub struct AppState {
    backend: SharedBackend,
}

pub fn app(backend: SharedBackend) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(AppState { backend })
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    let health = state.backend.health();
    Json(json!({
        "status": "ok",
        "model": state.backend.model().id,
        "backend": "hi-mlx",
        "ready": health.ready,
        "family": health.family,
        "quantization": health.quantization,
        "context_length": health.context_length,
        "memory_estimate_bytes": health.memory_estimate_bytes,
    }))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [model_card(state.backend.model())]
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request.model != state.backend.model().id {
        return api_error(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("model '{}' is not loaded", request.model),
        );
    }

    let max_tokens = request
        .max_tokens
        .or(request.max_completion_tokens)
        .unwrap_or(state.backend.model().max_output_tokens);
    let generation = GenerationRequest {
        prompt: build_prompt(
            state.backend.model().family,
            &request.messages,
            &request.tools,
            &request.tool_choice,
        ),
        max_tokens,
        temperature: request.temperature.unwrap_or(0.6),
        top_p: request.top_p.unwrap_or(0.95),
    };
    let output = match state.backend.generate(generation).await {
        Ok(output) => output,
        Err(err) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                err.to_string(),
            );
        }
    };
    let created = unix_time();
    let usage = json!({
        "prompt_tokens": output.prompt_tokens,
        "completion_tokens": output.completion_tokens,
        "total_tokens": output.prompt_tokens + output.completion_tokens,
    });
    let tool_calls = parse_tool_calls(&output.text, &request.tools);
    if request.stream {
        streaming_response(
            &request.model,
            created,
            &output.text,
            tool_calls,
            request.stream_options.include_usage(),
            usage,
        )
    } else {
        completion_response(&request.model, created, &output.text, tool_calls, usage)
    }
}

fn completion_response(
    model: &str,
    created: u64,
    text: &str,
    tool_calls: Option<Vec<NormalizedToolCall>>,
    usage: Value,
) -> Response {
    let (message, finish_reason) = match tool_calls {
        Some(calls) => (
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": calls,
            }),
            "tool_calls",
        ),
        None => (
            json!({
                "role": "assistant",
                "content": text,
            }),
            "stop",
        ),
    };
    Json(json!({
        "id": completion_id(created),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }))
    .into_response()
}

fn streaming_response(
    model: &str,
    created: u64,
    text: &str,
    tool_calls: Option<Vec<NormalizedToolCall>>,
    include_usage: bool,
    usage: Value,
) -> Response {
    let id = completion_id(created);
    let mut chunks = vec![json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": null,
        }]
    })];
    match tool_calls {
        Some(calls) => {
            chunks.push(json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": calls},
                    "finish_reason": null,
                }]
            }));
            chunks.push(finish_chunk(&id, model, created, "tool_calls"));
        }
        None => {
            for piece in split_stream_text(text) {
                chunks.push(json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": null,
                    }]
                }));
            }
            chunks.push(finish_chunk(&id, model, created, "stop"));
        }
    }
    if include_usage {
        chunks.push(json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage,
        }));
    }

    let events = chunks
        .into_iter()
        .map(|chunk| Ok::<_, Infallible>(Event::default().data(chunk.to_string())))
        .chain(std::iter::once(Ok(Event::default().data("[DONE]"))));
    Sse::new(stream::iter(events)).into_response()
}

fn finish_chunk(id: &str, model: &str, created: u64, reason: &str) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": reason,
        }]
    })
}

fn split_stream_text(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if current.len() >= 512 {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn model_card(model: &ModelInfo) -> Value {
    json!({
        "id": model.id,
        "object": "model",
        "created": 0,
        "owned_by": "hi-mlx",
        "family": model.family.label(),
        "context_window": model.context_length,
        "max_output_tokens": model.max_output_tokens,
    })
}

fn api_error(status: StatusCode, code: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "code": code,
            }
        })),
    )
        .into_response()
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn completion_id(created: u64) -> String {
    format!("chatcmpl-hi-mlx-{created}")
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub tool_choice: Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: StreamOptions,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

impl StreamOptions {
    fn include_usage(&self) -> bool {
        self.include_usage.unwrap_or(false)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<Value>,
}

impl ChatMessage {
    pub fn content_text(&self) -> String {
        match self.content.as_ref() {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .and_then(Value::as_str)
                        .or_else(|| part.as_str())
                })
                .collect::<Vec<_>>()
                .join("\n"),
            Some(other) => other.to_string(),
            None => String::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Value,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use serde_json::json;
    use tower::ServiceExt;

    use super::*;
    use crate::backend::MockBackend;
    use crate::manifest::{inspect_model, test_support};

    #[tokio::test]
    async fn health_returns_ok() {
        let app = test_app("hello").await;

        let response = app
            .oneshot(
                Request::get("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn models_returns_loaded_model_id() {
        let app = test_app("hello").await;

        let response = app
            .oneshot(
                Request::get("/v1/models")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = body_json(response).await;
        assert_eq!(
            body["data"][0]["id"],
            "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX"
        );
    }

    #[tokio::test]
    async fn non_streaming_completion_returns_openai_json() {
        let app = test_app("hello from mlx").await;
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{"role":"user","content":"hi"}],
            "stream": false
        });

        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["choices"][0]["message"]["content"], "hello from mlx");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn streaming_completion_returns_sse_and_done() {
        let app = test_app("hello from mlx").await;
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{"role":"user","content":"hi"}],
            "stream": true
        });

        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains("data: {"));
        assert!(body.contains("hello from mlx"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn tool_output_is_normalized_to_openai_tool_calls() {
        let app = test_app(r#"{"name":"read","arguments":{"path":"README.md"}}"#).await;
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{"role":"user","content":"read README"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type":"object"}
                }
            }]
        });

        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = body_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read"
        );
    }

    async fn test_app(output: &str) -> Router {
        let dir = tempfile_path("server");
        test_support::write_qwen_fixture(&dir);
        let model = inspect_model(&dir, None).unwrap();
        let backend = Arc::new(MockBackend::new(model, output));
        app(backend)
    }

    async fn body_json(response: axum::response::Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn tempfile_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-mlx-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }
}
