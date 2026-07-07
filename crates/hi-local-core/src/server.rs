use std::collections::VecDeque;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use futures_util::StreamExt;
use futures_util::future::{self, Either};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::backend::{
    GenerationEvent, GenerationRequest, GenerationStream, ImageInput, ImageSource, ImageUrlKind,
    MultimodalInput, SharedBackend, VideoInput, VideoSource,
};
use crate::model::ModelInfo;
use crate::prompt::build_prompt_with_template;
use crate::tool_parser::{NormalizedToolCall, parse_tool_calls};

#[derive(Clone)]
pub struct AppState {
    backend: SharedBackend,
    config: ServerConfig,
}

const STREAM_ADMISSION_ERROR_WINDOW: Duration = Duration::from_millis(10);
type GenerationStreamFuture =
    Pin<Box<dyn Future<Output = anyhow::Result<GenerationStream>> + Send>>;

#[derive(Clone, Debug, Default)]
pub struct ServerConfig {
    pub image_url_policy: ImageUrlPolicy,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ImageUrlPolicy {
    pub allow_http_urls: bool,
    pub allow_local_urls: bool,
}

pub fn app(backend: SharedBackend) -> Router {
    app_with_config(backend, ServerConfig::default())
}

pub fn app_with_config(backend: SharedBackend, config: ServerConfig) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(AppState { backend, config })
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    let health = state.backend.health();
    let multimodal = state.backend.multimodal_support();
    Json(json!({
        "status": "ok",
        "model": state.backend.model().id,
        "backend": health.backend,
        "ready": health.ready,
        "family": health.family,
        "quantization": health.quantization,
        "execution": segment_status_health_details(&health.quantization, "execution"),
        "cuda_runtime": segment_status_health_details(&health.quantization, "cuda-runtime"),
        "gpu_weights": segment_status_health_details(&health.quantization, "gpu-weights"),
        "gpu_matrices": segment_status_health_details(&health.quantization, "gpu-matrices"),
        "gpu_vectors": segment_status_health_details(&health.quantization, "gpu-vectors"),
        "scheduler": scheduler_health_details(&health.quantization),
        "kv_cache": kv_cache_health_details(&health.quantization),
        "prefix_cache": segment_status_health_details(&health.quantization, "prefix-cache"),
        "attention": attention_health_details(&health.quantization),
        "sampling": segment_status_health_details(&health.quantization, "sampling"),
        "batch": batch_health_details(&health.quantization),
        "multimodal_batching": segment_status_health_details(
            &health.quantization,
            "multimodal-batching"
        ),
        "multimodal_metrics": multimodal_metrics_health_details(&health.quantization),
        "multimodal_projector": multimodal_projector_health_details(&health.quantization),
        "multimodal": {
            "image_inputs": multimodal.image_inputs,
            "video_inputs": multimodal.video_inputs,
            "generation": multimodal.generation,
            "status": multimodal.status,
        },
        "context_length": health.context_length,
        "memory_estimate_bytes": health.memory_estimate_bytes,
    }))
}

fn kv_cache_health_details(health: &str) -> Value {
    let Some(segment) = health_segment(health, "kv-cache") else {
        return json!({"status": "unavailable"});
    };
    parenthetical_health_details(segment)
}

fn attention_health_details(health: &str) -> Value {
    let Some(status) = health_segment(health, "attention") else {
        return json!({"status": "unavailable"});
    };
    let mut object = status_segment_details(status);
    if let Some(detail) = health_segment(health, "attention-detail") {
        if let Value::Object(fields) = &mut object {
            fields.insert("detail".to_string(), json!(detail));
            for field in split_top_level(detail, ',')
                .into_iter()
                .map(str::trim)
                .filter(|field| !field.is_empty())
            {
                let Some((key, value)) = field.split_once('=') else {
                    continue;
                };
                fields.insert(
                    key.trim().to_string(),
                    parse_health_field_value(value.trim()),
                );
            }
        }
    }
    object
}

fn batch_health_details(health: &str) -> Value {
    let Some(segment) = health_segment(health, "batch") else {
        return json!({"status": "unavailable"});
    };
    key_value_health_details(segment)
}

fn scheduler_health_details(health: &str) -> Value {
    let Some(segment) = health_segment(health, "scheduler") else {
        return json!({"status": "unavailable"});
    };
    let mut details = comma_health_details(segment);
    if let Value::Object(object) = &mut details {
        insert_nested_health_object(
            object,
            "fallbacks",
            &[
                ("total", "fallback_requests"),
                ("single_request", "fallback_single_request"),
                ("moe", "fallback_moe_requests"),
                ("multimodal", "fallback_multimodal_requests"),
                ("tokenizer", "fallback_tokenizer_requests"),
                ("bucket_singleton", "fallback_bucket_requests"),
                ("other", "fallback_other_requests"),
            ],
        );
        insert_nested_health_object(
            object,
            "admission",
            &[
                ("capacity_rejected_requests", "capacity_rejected_requests"),
                ("rejected_requests", "admission_rejected_requests"),
                ("errored_requests", "admission_errored_requests"),
                ("cancelled_requests", "admission_cancelled_requests"),
                (
                    "context_rejected_requests",
                    "admission_context_rejected_requests",
                ),
                (
                    "invalid_request_rejected_requests",
                    "admission_invalid_request_rejected_requests",
                ),
                (
                    "memory_rejected_requests",
                    "admission_memory_rejected_requests",
                ),
                (
                    "other_rejected_requests",
                    "admission_other_rejected_requests",
                ),
                ("requested_pages", "admission_requested_pages"),
                ("granted_pages", "admission_granted_pages"),
                ("rejected_pages", "admission_rejected_pages"),
                ("page_requests", "admission_page_requests"),
                ("requested_pages_max", "admission_requested_pages_max"),
                ("requested_pages_avg", "admission_requested_pages_avg"),
                ("granted_pages_max", "admission_granted_pages_max"),
                ("rejected_pages_max", "admission_rejected_pages_max"),
                ("page_grant_per_mille", "admission_page_grant_per_mille"),
                ("page_reject_per_mille", "admission_page_reject_per_mille"),
            ],
        );
        if let Some(Value::Object(admission)) = object.get_mut("admission") {
            insert_nested_health_object(
                admission,
                "by_reason",
                &[
                    ("context_length", "context_rejected_requests"),
                    ("invalid_request", "invalid_request_rejected_requests"),
                    ("insufficient_gpu_memory", "memory_rejected_requests"),
                    ("other", "other_rejected_requests"),
                ],
            );
        }
        insert_nested_health_object(
            object,
            "capacity",
            &[
                ("rejected_requests", "capacity_rejected_requests"),
                (
                    "max_active_rejected_requests",
                    "capacity_max_active_rejected_requests",
                ),
                (
                    "active_requests_total",
                    "capacity_rejected_active_requests_total",
                ),
                (
                    "active_requests_max",
                    "capacity_rejected_active_requests_max",
                ),
                (
                    "active_requests_avg",
                    "capacity_rejected_active_requests_avg",
                ),
            ],
        );
        insert_nested_health_object(
            object,
            "queue",
            &[
                ("peak_pending", "peak_pending"),
                ("pending", "pending"),
                ("queued_requests", "queued_requests"),
                ("waited_requests", "queue_waited_requests"),
                ("wait_micros_total", "queue_wait_micros_total"),
                ("wait_micros_max", "queue_wait_micros_max"),
                ("peak_inflight_requests", "peak_inflight_requests"),
                ("inflight_requests", "inflight_requests"),
                ("active_batch_size", "active"),
                ("active_requests", "active_requests"),
            ],
        );
        insert_nested_health_object(
            object,
            "idle",
            &[
                ("wakeups", "idle_wakeups"),
                ("micros_total", "idle_micros_total"),
                ("micros_max", "idle_micros_max"),
            ],
        );
        insert_nested_health_object(
            object,
            "batching",
            &[
                ("total_batches", "total_batches"),
                ("total_requests", "total_requests"),
                ("gpu_batched_batches", "gpu_batched_batches"),
                ("gpu_batched_requests", "gpu_batched_requests"),
                (
                    "paged_lease_batched_requests",
                    "paged_lease_batched_requests",
                ),
                (
                    "paged_internal_batched_requests",
                    "paged_internal_batched_requests",
                ),
                ("greedy_batched_requests", "greedy_batched_requests"),
                ("sampled_batched_requests", "sampled_batched_requests"),
                ("eligible_requests", "batch_plan_eligible_requests"),
                ("ineligible_requests", "batch_plan_ineligible_requests"),
                ("max_observed_batch", "max_observed"),
            ],
        );
        insert_nested_health_object(
            object,
            "token_budget",
            &[
                ("split_groups", "token_budget_split_groups"),
                ("split_requests", "token_budget_split_requests"),
                ("split_chunks", "token_budget_split_chunks"),
            ],
        );
        insert_nested_health_object(
            object,
            "continuous",
            &[
                ("step_admissions", "continuous_step_admissions"),
                ("admitted_requests", "continuous_admitted_requests"),
                ("decode_iterations", "continuous_decode_iterations"),
                ("prefill_batches", "continuous_prefill_batches"),
                ("prefill_requests", "continuous_prefill_requests"),
                ("append_decode_batches", "continuous_append_decode_batches"),
                (
                    "append_decode_requests",
                    "continuous_append_decode_requests",
                ),
                (
                    "mid_decode_retirements",
                    "continuous_mid_decode_retirements",
                ),
                (
                    "mid_decode_cancellations",
                    "continuous_mid_decode_cancellations",
                ),
            ],
        );
        insert_nested_health_object(
            object,
            "shared_prefix",
            &[
                ("batches", "shared_prefix_batches"),
                ("requests", "shared_prefix_requests"),
                ("tokens_total", "shared_prefix_tokens_total"),
            ],
        );
        insert_nested_health_object(
            object,
            "requests",
            &[
                ("total", "total_requests"),
                ("completed", "completed_requests"),
                ("errored", "errored_requests"),
                ("cancelled", "cancelled_requests"),
            ],
        );
        insert_nested_health_object(
            object,
            "throughput",
            &[
                ("prefill_tokens_total", "prefill_tokens_total"),
                ("decode_tokens_total", "decode_tokens_total"),
                ("prefill_micros_total", "prefill_micros_total"),
                ("decode_micros_total", "decode_micros_total"),
                ("tokens_per_sec_prefill", "tokens_per_sec_prefill"),
                ("tokens_per_sec_decode", "tokens_per_sec_decode"),
            ],
        );
    }
    details
}

fn multimodal_metrics_health_details(health: &str) -> Value {
    let Some(segment) = health_segment(health, "multimodal-metrics") else {
        return json!({"status": "unavailable"});
    };
    let mut details = comma_health_details(segment);
    if let Value::Object(object) = &mut details {
        insert_nested_health_object(
            object,
            "fallbacks",
            &[
                ("total", "multimodal_fallback_requests"),
                ("direct", "multimodal_fallback_direct_requests"),
                ("mixed_batch", "multimodal_fallback_mixed_batch_requests"),
                (
                    "decode_group_singleton",
                    "multimodal_fallback_decode_group_singleton_requests",
                ),
                (
                    "projection_failed",
                    "multimodal_fallback_projection_failed_requests",
                ),
                (
                    "projection_mismatch",
                    "multimodal_fallback_projection_mismatch_requests",
                ),
                (
                    "incompatible_prompt",
                    "multimodal_fallback_incompatible_prompt_requests",
                ),
                ("token_budget", "multimodal_fallback_token_budget_requests"),
            ],
        );
    }
    details
}

fn multimodal_projector_health_details(health: &str) -> Value {
    let Some(segment) = health_segment(health, "multimodal") else {
        return json!({"status": "unavailable"});
    };
    let segment = segment.trim();
    let Some(start) = segment.find('(') else {
        return status_segment_details(segment);
    };
    let status = segment[..start].trim();
    let fields = segment[start + 1..]
        .strip_suffix(')')
        .unwrap_or(&segment[start + 1..]);
    let mut object = serde_json::Map::new();
    object.insert(
        "status".to_string(),
        json!(if status.is_empty() { "unknown" } else { status }),
    );
    for section in split_top_level(fields, ';')
        .into_iter()
        .map(str::trim)
        .filter(|section| !section.is_empty())
    {
        for field in split_top_level(section, ',')
            .into_iter()
            .map(str::trim)
            .filter(|field| !field.is_empty())
        {
            let Some((key, value)) = field.split_once('=') else {
                continue;
            };
            let key = key.trim();
            if key == "vision-encoder" {
                object.insert(
                    "vision_encoder".to_string(),
                    parenthetical_health_details(value.trim()),
                );
            } else {
                object.insert(key.to_string(), parse_health_field_value(value.trim()));
            }
        }
    }
    Value::Object(object)
}

fn segment_status_health_details(health: &str, key: &str) -> Value {
    let Some(segment) = health_segment(health, key) else {
        return json!({"status": "unavailable"});
    };
    status_segment_details(segment)
}

fn status_segment_details(segment: &str) -> Value {
    if segment.contains('(') {
        parenthetical_health_details(segment)
    } else {
        json!({"status": if segment.trim().is_empty() { "unknown" } else { segment.trim() }})
    }
}

fn comma_health_details(segment: &str) -> Value {
    let mut fields = split_top_level(segment, ',')
        .into_iter()
        .map(str::trim)
        .filter(|field| !field.is_empty());
    let status = fields.next().unwrap_or("unknown");
    let mut object = serde_json::Map::new();
    object.insert("status".to_string(), json!(status));
    for field in fields {
        let Some((key, value)) = field.split_once('=') else {
            continue;
        };
        object.insert(
            key.trim().to_string(),
            parse_health_field_value(value.trim()),
        );
    }
    Value::Object(object)
}

fn insert_nested_health_object(
    parent: &mut serde_json::Map<String, Value>,
    key: &str,
    fields: &[(&str, &str)],
) {
    let mut nested = serde_json::Map::new();
    for (target, source) in fields {
        if let Some(value) = parent.get(*source).cloned() {
            nested.insert((*target).to_string(), value);
        }
    }
    if !nested.is_empty() {
        parent.insert(key.to_string(), Value::Object(nested));
    }
}

fn key_value_health_details(segment: &str) -> Value {
    let mut object = serde_json::Map::new();
    object.insert("status".to_string(), json!("available"));
    for field in split_top_level(segment, ',')
        .into_iter()
        .map(str::trim)
        .filter(|field| !field.is_empty())
    {
        let Some((key, value)) = field.split_once('=') else {
            continue;
        };
        object.insert(
            key.trim().to_string(),
            parse_health_field_value(value.trim()),
        );
    }
    Value::Object(object)
}

fn parenthetical_health_details(segment: &str) -> Value {
    let segment = segment.trim();
    let (status, fields) = match segment.find('(') {
        Some(start) => {
            let status = segment[..start].trim();
            let fields = segment[start + 1..]
                .strip_suffix(')')
                .unwrap_or(&segment[start + 1..]);
            (status, fields)
        }
        None => (segment, ""),
    };
    let mut object = serde_json::Map::new();
    object.insert(
        "status".to_string(),
        json!(if status.is_empty() { "unknown" } else { status }),
    );
    if !fields.trim().is_empty() && !fields.contains('=') {
        object.insert("reason".to_string(), json!(fields.trim()));
        return Value::Object(object);
    }
    for field in split_top_level(fields, ',')
        .into_iter()
        .map(str::trim)
        .filter(|field| !field.is_empty())
    {
        let Some((key, value)) = field.split_once('=') else {
            continue;
        };
        object.insert(
            key.trim().to_string(),
            parse_health_field_value(value.trim()),
        );
    }
    Value::Object(object)
}

fn health_segment<'a>(health: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    split_top_level(health, ';')
        .into_iter()
        .map(str::trim)
        .find_map(|segment| segment.strip_prefix(&prefix))
}

fn split_top_level(input: &str, delimiter: char) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, ch) in input.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if ch == delimiter && depth == 0 => {
                segments.push(&input[start..index]);
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    segments.push(&input[start..]);
    segments
}

fn parse_health_field_value(value: &str) -> Value {
    let value = value.trim();
    if value.contains('(') {
        parenthetical_health_details(value)
    } else {
        parse_health_value(value)
    }
}

fn parse_health_value(value: &str) -> Value {
    if let Ok(value) = value.parse::<u64>() {
        json!(value)
    } else if let Ok(value) = value.parse::<bool>() {
        json!(value)
    } else {
        json!(value)
    }
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    let backend = state.backend.health().backend;
    Json(json!({
        "object": "list",
        "data": [model_card(state.backend.model(), &backend)]
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
    if let Err(response) = validate_max_tokens(max_tokens, state.backend.model().max_output_tokens)
    {
        return response;
    }
    let media_inputs = match collect_media_inputs(&request.messages, &state.config.image_url_policy)
    {
        Ok(media_inputs) => media_inputs,
        Err(response) => return response,
    };
    if !media_inputs.is_empty() {
        let multimodal = state.backend.multimodal_support();
        let has_video = media_inputs
            .iter()
            .any(|input| matches!(input, MultimodalInput::Video(_)));
        if !multimodal.image_inputs || (has_video && !multimodal.video_inputs) {
            return api_error(
                StatusCode::BAD_REQUEST,
                "unsupported_multimodal",
                "multimodal inputs require a CUDA Qwen multimodal model with an mmproj path; this server has no multimodal projector loaded",
            );
        }
        if !multimodal.generation {
            return api_error(
                StatusCode::BAD_REQUEST,
                "unsupported_multimodal",
                format!(
                    "multimodal inputs reached a configured multimodal path ({}) but multimodal generation is not implemented in this build",
                    multimodal.status
                ),
            );
        }
    }
    let sampling_defaults = state.backend.sampling_defaults();
    let temperature = request.temperature.unwrap_or(sampling_defaults.temperature);
    let top_p = request.top_p.unwrap_or(sampling_defaults.top_p);
    if let Err(response) = validate_sampling_parameters(temperature, top_p, request.top_k) {
        return response;
    }
    let stop_sequences = request.stop_sequences();
    let generation = GenerationRequest {
        prompt: build_prompt_with_template(
            state.backend.model().family,
            state.backend.chat_template(),
            &request.messages,
            &request.tools,
            &request.tool_choice,
        ),
        max_tokens,
        temperature,
        top_p,
        top_k: request.top_k,
        seed: request.seed,
        stop_sequences: stop_sequences.clone(),
        media_inputs,
    };
    let created = unix_time();
    if request.stream {
        return streaming_response(
            state.backend.clone(),
            request.model,
            generation,
            request.tools,
            created,
            request.stream_options.include_usage(),
            stop_sequences,
        )
        .await;
    }

    let mut output = match state.backend.generate(generation).await {
        Ok(output) => output,
        Err(err) => return generation_error_response(err.to_string()),
    };
    output.text = truncate_at_stop(&output.text, &stop_sequences);
    let usage = json!({
        "prompt_tokens": output.prompt_tokens,
        "completion_tokens": output.completion_tokens,
        "total_tokens": output.prompt_tokens + output.completion_tokens,
    });
    let tool_calls = parse_tool_calls(&output.text, &request.tools);
    completion_response(&request.model, created, &output.text, tool_calls, usage)
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

async fn streaming_response(
    backend: SharedBackend,
    model: String,
    generation: GenerationRequest,
    tools: Vec<Tool>,
    created: u64,
    include_usage: bool,
    stop_sequences: Vec<String>,
) -> Response {
    let id = completion_id(created);
    let stream_future: GenerationStreamFuture =
        Box::pin(async move { backend.stream_generate(generation).await });
    let admission_delay = Box::pin(tokio::time::sleep(STREAM_ADMISSION_ERROR_WINDOW));
    match future::select(stream_future, admission_delay).await {
        Either::Left((Ok(stream), _)) => streaming_response_from_stream_future(
            id,
            model,
            tools,
            created,
            include_usage,
            stop_sequences,
            Box::pin(async move { Ok(stream) }),
        ),
        Either::Left((Err(err), _)) => generation_error_response(err.to_string()),
        Either::Right((_, stream_future)) => streaming_response_from_stream_future(
            id,
            model,
            tools,
            created,
            include_usage,
            stop_sequences,
            stream_future,
        ),
    }
}

fn streaming_response_from_stream_future(
    id: String,
    model: String,
    tools: Vec<Tool>,
    created: u64,
    include_usage: bool,
    stop_sequences: Vec<String>,
    stream_future: GenerationStreamFuture,
) -> Response {
    let mut pending = VecDeque::new();
    pending.push_back(sse_json_event(json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": null,
        }]
    })));
    let tool_mode = !tools.is_empty();
    let state = OpenAiSseState {
        id,
        model,
        tools,
        created,
        include_usage,
        tool_mode,
        buffered_text: String::new(),
        final_usage: None,
        stop_filter: StopStreamFilter::new(stop_sequences),
        stream_future: Some(stream_future),
        stream: None,
        pending,
        finished: false,
    };
    let events = futures_util::stream::unfold(state, |mut state| async move {
        state.next_event().await.map(|event| (event, state))
    });
    Sse::new(events).into_response()
}

struct OpenAiSseState {
    id: String,
    model: String,
    tools: Vec<Tool>,
    created: u64,
    include_usage: bool,
    tool_mode: bool,
    buffered_text: String,
    final_usage: Option<Value>,
    stop_filter: StopStreamFilter,
    stream_future: Option<GenerationStreamFuture>,
    stream: Option<GenerationStream>,
    pending: VecDeque<Result<Event, Infallible>>,
    finished: bool,
}

impl OpenAiSseState {
    async fn next_event(&mut self) -> Option<Result<Event, Infallible>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }
            if self.finished {
                return None;
            }
            if self.stream.is_none() {
                let Some(stream_future) = self.stream_future.take() else {
                    self.queue_finish();
                    continue;
                };
                match stream_future.await {
                    Ok(stream) => {
                        self.stream = Some(stream);
                    }
                    Err(err) => {
                        self.pending
                            .push_back(generation_error_event(err.to_string()));
                        self.pending.push_back(done_event());
                        self.finished = true;
                        continue;
                    }
                }
            }

            let event = {
                let stream = self.stream.as_mut().expect("stream was initialized");
                stream.next().await
            };
            match event {
                Some(Ok(GenerationEvent::TokenDelta { text, .. })) => {
                    self.queue_token_delta(text);
                    if self.stop_filter.stopped() {
                        self.stream.take();
                        self.queue_finish();
                    }
                }
                Some(Ok(GenerationEvent::Finished { output })) => {
                    self.final_usage = Some(json!({
                        "prompt_tokens": output.prompt_tokens,
                        "completion_tokens": output.completion_tokens,
                        "total_tokens": output.prompt_tokens + output.completion_tokens,
                    }));
                    if self.tool_mode {
                        self.buffered_text =
                            truncate_at_stop(&output.text, self.stop_filter.stops());
                    } else if !self.stop_filter.stopped() {
                        for piece in self.stop_filter.finish() {
                            self.queue_content_piece(piece);
                        }
                    }
                    self.stream.take();
                    self.queue_finish();
                }
                Some(Err(err)) => {
                    self.stream.take();
                    self.pending
                        .push_back(generation_error_event(err.to_string()));
                    self.pending.push_back(done_event());
                    self.finished = true;
                }
                None => {
                    self.stream.take();
                    self.queue_finish();
                }
            }
        }
    }

    fn queue_token_delta(&mut self, text: String) {
        let pieces = self.stop_filter.push(&text);
        if self.tool_mode {
            for piece in pieces {
                self.buffered_text.push_str(&piece);
            }
        } else {
            for piece in pieces {
                self.queue_content_piece(piece);
            }
        }
    }

    fn queue_content_piece(&mut self, piece: String) {
        if piece.is_empty() {
            return;
        }
        self.pending.push_back(sse_json_event(json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"content": piece},
                "finish_reason": null,
            }]
        })));
    }

    fn queue_finish(&mut self) {
        if self.finished {
            return;
        }
        let finish_reason = if self.tool_mode {
            match parse_tool_calls(&self.buffered_text, &self.tools) {
                Some(calls) => {
                    self.pending.push_back(sse_json_event(json!({
                        "id": self.id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": self.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": calls},
                            "finish_reason": null,
                        }]
                    })));
                    "tool_calls"
                }
                None => {
                    for piece in split_stream_text(&self.buffered_text) {
                        self.queue_content_piece(piece);
                    }
                    "stop"
                }
            }
        } else {
            "stop"
        };

        self.pending.push_back(sse_json_event(finish_chunk(
            &self.id,
            &self.model,
            self.created,
            finish_reason,
        )));
        if self.include_usage {
            if let Some(usage) = self.final_usage.take() {
                self.pending.push_back(sse_json_event(json!({
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [],
                    "usage": usage,
                })));
            }
        }
        self.pending.push_back(done_event());
        self.finished = true;
    }
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

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => sanitize_stop_sequences(vec![value]),
            Self::Many(values) => sanitize_stop_sequences(values),
        }
    }
}

fn sanitize_stop_sequences(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .filter(|value| !value.is_empty())
        .take(4)
        .collect()
}

fn truncate_at_stop(text: &str, stops: &[String]) -> String {
    let Some(index) = earliest_stop_index(text, stops) else {
        return text.to_string();
    };
    text[..index].to_string()
}

fn earliest_stop_index(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()
}

struct StopStreamFilter {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}

impl StopStreamFilter {
    fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
            stopped: false,
        }
    }

    fn stops(&self) -> &[String] {
        &self.stops
    }

    fn stopped(&self) -> bool {
        self.stopped
    }

    fn push(&mut self, text: &str) -> Vec<String> {
        if text.is_empty() || self.stopped {
            return Vec::new();
        }
        if self.stops.is_empty() {
            return vec![text.to_string()];
        }

        self.pending.push_str(text);
        if let Some(index) = earliest_stop_index(&self.pending, &self.stops) {
            let emit = self.pending[..index].to_string();
            self.pending.clear();
            self.stopped = true;
            return if emit.is_empty() {
                Vec::new()
            } else {
                vec![emit]
            };
        }

        let keep = longest_stop_prefix_suffix_len(&self.pending, &self.stops);
        let emit_len = self.pending.len().saturating_sub(keep);
        if emit_len == 0 {
            return Vec::new();
        }
        let emit = self.pending[..emit_len].to_string();
        self.pending = self.pending[emit_len..].to_string();
        vec![emit]
    }

    fn finish(&mut self) -> Vec<String> {
        if self.stopped || self.pending.is_empty() {
            return Vec::new();
        }
        let pending = std::mem::take(&mut self.pending);
        vec![pending]
    }
}

fn longest_stop_prefix_suffix_len(text: &str, stops: &[String]) -> usize {
    let mut keep = 0usize;
    for stop in stops {
        for end in stop
            .char_indices()
            .map(|(idx, _)| idx)
            .skip(1)
            .chain(std::iter::once(stop.len()))
        {
            let prefix = &stop[..end];
            if text.ends_with(prefix) {
                keep = keep.max(prefix.len());
            }
        }
    }
    keep
}

fn collect_media_inputs(
    messages: &[ChatMessage],
    policy: &ImageUrlPolicy,
) -> Result<Vec<MultimodalInput>, Response> {
    let mut inputs = Vec::new();
    for message in messages {
        let Some(Value::Array(parts)) = message.content.as_ref() else {
            continue;
        };
        for part in parts {
            match content_part_kind(part) {
                ContentPartKind::Text => {}
                ContentPartKind::Image => {
                    inputs.push(MultimodalInput::Image(parse_image_part(part, policy)?))
                }
                ContentPartKind::Video => {
                    inputs.push(MultimodalInput::Video(parse_video_part(part, policy)?))
                }
                ContentPartKind::Unsupported(kind) => {
                    return Err(api_error(
                        StatusCode::BAD_REQUEST,
                        "unsupported_content_part",
                        format!(
                            "unsupported chat content part type '{kind}'; only text, image_url, video, and video_url parts are supported"
                        ),
                    ));
                }
            }
        }
    }
    Ok(inputs)
}

enum ContentPartKind {
    Text,
    Image,
    Video,
    Unsupported(String),
}

fn content_part_kind(part: &Value) -> ContentPartKind {
    if part.as_str().is_some() {
        return ContentPartKind::Text;
    }
    let Some(object) = part.as_object() else {
        return ContentPartKind::Unsupported(part.to_string());
    };
    match object.get("type").and_then(Value::as_str) {
        Some("text") => ContentPartKind::Text,
        Some("image_url") => ContentPartKind::Image,
        Some("video") | Some("video_url") => ContentPartKind::Video,
        Some(kind) => ContentPartKind::Unsupported(kind.to_string()),
        None if object.contains_key("image_url") => ContentPartKind::Image,
        None if object.contains_key("video") || object.contains_key("video_url") => {
            ContentPartKind::Video
        }
        None if object.contains_key("text") => ContentPartKind::Text,
        None => ContentPartKind::Unsupported("unknown".to_string()),
    }
}

fn parse_image_part(part: &Value, policy: &ImageUrlPolicy) -> Result<ImageInput, Response> {
    let Some(image_url) = part.get("image_url") else {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_image_url",
            "image_url content part is missing the image_url field",
        ));
    };
    let (url, detail) = match image_url {
        Value::String(url) => (
            url.as_str(),
            part.get("detail")
                .and_then(Value::as_str)
                .map(str::to_string),
        ),
        Value::Object(object) => {
            let Some(url) = object.get("url").and_then(Value::as_str) else {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_image_url",
                    "image_url object must include a string url field",
                ));
            };
            let detail = object
                .get("detail")
                .and_then(Value::as_str)
                .or_else(|| part.get("detail").and_then(Value::as_str))
                .map(str::to_string);
            (url, detail)
        }
        _ => {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_image_url",
                "image_url content part must be a string or object",
            ));
        }
    };
    image_input_from_url(url, detail, policy)
}

fn parse_video_part(part: &Value, policy: &ImageUrlPolicy) -> Result<VideoInput, Response> {
    let video_value = part
        .get("video")
        .or_else(|| part.get("video_url"))
        .ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                "invalid_video_url",
                "video content part is missing the video or video_url field",
            )
        })?;
    let detail = part
        .get("detail")
        .and_then(Value::as_str)
        .map(str::to_string);
    let fps = number_field(part, "fps")?;
    let nframes = usize_field(part, "nframes")?;
    let min_frames = usize_field(part, "min_frames")?;
    let max_frames = usize_field(part, "max_frames")?;

    match video_value {
        Value::String(url) => {
            video_input_from_url(url, detail, fps, nframes, min_frames, max_frames, policy)
        }
        Value::Array(frames) => {
            let mut image_frames = Vec::with_capacity(frames.len());
            for frame in frames {
                image_frames.push(parse_video_frame_image(frame, policy)?);
            }
            if image_frames.is_empty() {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_video_url",
                    "video frame list must not be empty",
                ));
            }
            Ok(VideoInput {
                source: VideoSource::Frames(image_frames),
                detail,
                fps,
                nframes,
                min_frames,
                max_frames,
            })
        }
        Value::Object(object) => {
            if let Some(frames) = object.get("frames").and_then(Value::as_array) {
                let mut image_frames = Vec::with_capacity(frames.len());
                for frame in frames {
                    image_frames.push(parse_video_frame_image(frame, policy)?);
                }
                if image_frames.is_empty() {
                    return Err(api_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_video_url",
                        "video frame list must not be empty",
                    ));
                }
                return Ok(VideoInput {
                    source: VideoSource::Frames(image_frames),
                    detail: object
                        .get("detail")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .or(detail),
                    fps: number_field(video_value, "fps")?.or(fps),
                    nframes: usize_field(video_value, "nframes")?.or(nframes),
                    min_frames: usize_field(video_value, "min_frames")?.or(min_frames),
                    max_frames: usize_field(video_value, "max_frames")?.or(max_frames),
                });
            }
            let Some(url) = object.get("url").and_then(Value::as_str) else {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_video_url",
                    "video object must include a string url field or a frames array",
                ));
            };
            video_input_from_url(
                url,
                object
                    .get("detail")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or(detail),
                number_field(video_value, "fps")?.or(fps),
                usize_field(video_value, "nframes")?.or(nframes),
                usize_field(video_value, "min_frames")?.or(min_frames),
                usize_field(video_value, "max_frames")?.or(max_frames),
                policy,
            )
        }
        _ => Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_video_url",
            "video content part must be a string, object, or frame URL array",
        )),
    }
}

fn parse_video_frame_image(frame: &Value, policy: &ImageUrlPolicy) -> Result<ImageInput, Response> {
    match frame {
        Value::String(url) => image_input_from_url(url, None, policy),
        Value::Object(_) if frame.get("image_url").is_some() => parse_image_part(frame, policy),
        Value::Object(object) => {
            let Some(url) = object.get("url").and_then(Value::as_str) else {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_video_url",
                    "video frame object must include a string url field",
                ));
            };
            image_input_from_url(
                url,
                object
                    .get("detail")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                policy,
            )
        }
        _ => Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_video_url",
            "video frame entries must be URL strings or objects",
        )),
    }
}

fn number_field(value: &Value, field: &str) -> Result<Option<f32>, Response> {
    let Some(value) = value.get(field) else {
        return Ok(None);
    };
    let Some(number) = value.as_f64() else {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_video_url",
            format!("video field '{field}' must be numeric"),
        ));
    };
    if !number.is_finite() || number <= 0.0 {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_video_url",
            format!("video field '{field}' must be a positive finite number"),
        ));
    }
    Ok(Some(number as f32))
}

fn usize_field(value: &Value, field: &str) -> Result<Option<usize>, Response> {
    let Some(value) = value.get(field) else {
        return Ok(None);
    };
    let Some(number) = value.as_u64() else {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_video_url",
            format!("video field '{field}' must be a positive integer"),
        ));
    };
    if number == 0 {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_video_url",
            format!("video field '{field}' must be greater than zero"),
        ));
    }
    usize::try_from(number).map(Some).map_err(|_| {
        api_error(
            StatusCode::BAD_REQUEST,
            "invalid_video_url",
            format!("video field '{field}' is too large"),
        )
    })
}

fn image_input_from_url(
    url: &str,
    detail: Option<String>,
    policy: &ImageUrlPolicy,
) -> Result<ImageInput, Response> {
    if url.starts_with("data:") {
        return parse_data_image_url(url, detail);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        if !policy.allow_http_urls {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "unsupported_image_url",
                "HTTP image URLs are disabled; restart hi-local with --allow-http-image-url to enable them",
            ));
        }
        return Ok(ImageInput {
            source: ImageSource::Url {
                kind: ImageUrlKind::Http,
                url: url.to_string(),
            },
            detail,
        });
    }
    if url.starts_with("file://") || !url.contains("://") {
        if !policy.allow_local_urls {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "unsupported_image_url",
                "local image URLs are disabled; restart hi-local with --allow-local-image-url to enable them",
            ));
        }
        if url.trim().is_empty() {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_image_url",
                "local image URL must not be empty",
            ));
        }
        return Ok(ImageInput {
            source: ImageSource::Url {
                kind: ImageUrlKind::Local,
                url: url.to_string(),
            },
            detail,
        });
    }
    Err(api_error(
        StatusCode::BAD_REQUEST,
        "unsupported_image_url",
        "only data:, http(s), and local image URLs are supported",
    ))
}

fn video_input_from_url(
    url: &str,
    detail: Option<String>,
    fps: Option<f32>,
    nframes: Option<usize>,
    min_frames: Option<usize>,
    max_frames: Option<usize>,
    policy: &ImageUrlPolicy,
) -> Result<VideoInput, Response> {
    if url.starts_with("data:") {
        return parse_data_video_url(url, detail, fps, nframes, min_frames, max_frames);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        if !policy.allow_http_urls {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "unsupported_video_url",
                "HTTP video URLs are disabled; restart hi-local with --allow-http-image-url to enable them",
            ));
        }
        return Ok(VideoInput {
            source: VideoSource::Url {
                kind: ImageUrlKind::Http,
                url: url.to_string(),
            },
            detail,
            fps,
            nframes,
            min_frames,
            max_frames,
        });
    }
    if url.starts_with("file://") || !url.contains("://") {
        if !policy.allow_local_urls {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "unsupported_video_url",
                "local video URLs are disabled; restart hi-local with --allow-local-image-url to enable them",
            ));
        }
        if url.trim().is_empty() {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_video_url",
                "local video URL must not be empty",
            ));
        }
        return Ok(VideoInput {
            source: VideoSource::Url {
                kind: ImageUrlKind::Local,
                url: url.to_string(),
            },
            detail,
            fps,
            nframes,
            min_frames,
            max_frames,
        });
    }
    Err(api_error(
        StatusCode::BAD_REQUEST,
        "unsupported_video_url",
        "only data:, http(s), and local video URLs are supported",
    ))
}

fn parse_data_image_url(url: &str, detail: Option<String>) -> Result<ImageInput, Response> {
    let Some(payload) = url.strip_prefix("data:") else {
        unreachable!("parse_data_image_url is only called for data URLs")
    };
    let Some((metadata, encoded)) = payload.split_once(',') else {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_image_url",
            "data image URL must contain a comma-separated base64 payload",
        ));
    };
    let Some(media_type) = metadata.strip_suffix(";base64") else {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_image_url",
            "data image URL must use base64 encoding",
        ));
    };
    if !media_type.to_ascii_lowercase().starts_with("image/") {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_image_url",
            format!("unsupported data URL media type '{media_type}'; expected image/*"),
        ));
    }
    let bytes = BASE64_STANDARD.decode(encoded).map_err(|err| {
        api_error(
            StatusCode::BAD_REQUEST,
            "invalid_image_url",
            format!("data image URL contains invalid base64: {err}"),
        )
    })?;
    Ok(ImageInput {
        source: ImageSource::Data {
            media_type: media_type.to_string(),
            bytes,
        },
        detail,
    })
}

fn parse_data_video_url(
    url: &str,
    detail: Option<String>,
    fps: Option<f32>,
    nframes: Option<usize>,
    min_frames: Option<usize>,
    max_frames: Option<usize>,
) -> Result<VideoInput, Response> {
    let Some(payload) = url.strip_prefix("data:") else {
        unreachable!("parse_data_video_url is only called for data URLs")
    };
    let Some((metadata, encoded)) = payload.split_once(',') else {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_video_url",
            "data video URL must contain a comma-separated base64 payload",
        ));
    };
    let Some(media_type) = metadata.strip_suffix(";base64") else {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_video_url",
            "data video URL must use base64 encoding",
        ));
    };
    if !media_type.to_ascii_lowercase().starts_with("video/") {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_video_url",
            format!("unsupported data URL media type '{media_type}'; expected video/*"),
        ));
    }
    let bytes = BASE64_STANDARD.decode(encoded).map_err(|err| {
        api_error(
            StatusCode::BAD_REQUEST,
            "invalid_video_url",
            format!("data video URL contains invalid base64: {err}"),
        )
    })?;
    Ok(VideoInput {
        source: VideoSource::Data {
            media_type: media_type.to_string(),
            bytes,
        },
        detail,
        fps,
        nframes,
        min_frames,
        max_frames,
    })
}

fn sse_json_event(value: Value) -> Result<Event, Infallible> {
    Ok(Event::default().data(value.to_string()))
}

fn generation_error_event(message: String) -> Result<Event, Infallible> {
    let (code, error_type) = generation_error_code_and_type(&message);
    sse_json_event(json!({
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
    }))
}

fn done_event() -> Result<Event, Infallible> {
    Ok(Event::default().data("[DONE]"))
}

fn model_card(model: &ModelInfo, backend: &str) -> Value {
    json!({
        "id": model.id,
        "object": "model",
        "created": 0,
        "owned_by": format!("hi-{backend}"),
        "family": model.family.label(),
        "context_window": model.context_length,
        "max_output_tokens": model.max_output_tokens,
    })
}

fn validate_max_tokens(max_tokens: u32, max_output_tokens: u32) -> Result<(), Response> {
    if max_tokens == 0 {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            "max_tokens must be greater than 0",
        ));
    }
    if max_tokens > max_output_tokens {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            format!("max_tokens {max_tokens} exceeds model max_output_tokens {max_output_tokens}"),
        ));
    }
    Ok(())
}

fn validate_sampling_parameters(
    temperature: f32,
    top_p: f32,
    top_k: Option<u32>,
) -> Result<(), Response> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_sampling_parameter",
            "temperature must be a finite number greater than or equal to 0",
        ));
    }
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_sampling_parameter",
            "top_p must be a finite number between 0 and 1",
        ));
    }
    if top_k == Some(0) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_sampling_parameter",
            "top_k must be greater than 0 when provided",
        ));
    }
    Ok(())
}

fn generation_error_response(message: String) -> Response {
    let (code, error_type) = generation_error_code_and_type(&message);
    let status = match code {
        "context_length_exceeded" => StatusCode::BAD_REQUEST,
        "invalid_request_parameter" => StatusCode::BAD_REQUEST,
        "invalid_sampling_parameter" => StatusCode::BAD_REQUEST,
        "insufficient_gpu_memory" => StatusCode::SERVICE_UNAVAILABLE,
        "scheduler_over_capacity" => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    api_error_with_type_and_details(
        status,
        code,
        error_type,
        message.clone(),
        generation_error_details(code, &message),
    )
}

fn generation_error_code_and_type(message: &str) -> (&'static str, &'static str) {
    if message.contains("context_length_exceeded") {
        ("context_length_exceeded", "invalid_request_error")
    } else if message.contains("invalid_request_parameter") {
        ("invalid_request_parameter", "invalid_request_error")
    } else if message.contains("invalid_sampling_parameter") {
        ("invalid_sampling_parameter", "invalid_request_error")
    } else if message.contains("insufficient_gpu_memory") {
        ("insufficient_gpu_memory", "server_error")
    } else if message.contains("scheduler_over_capacity") {
        ("scheduler_over_capacity", "rate_limit_error")
    } else {
        ("generation_error", "server_error")
    }
}

fn generation_error_details(code: &str, message: &str) -> Option<Value> {
    match code {
        "insufficient_gpu_memory" => insufficient_gpu_memory_error_details(message),
        "scheduler_over_capacity" => scheduler_over_capacity_error_details(message),
        _ => None,
    }
}

fn insufficient_gpu_memory_error_details(message: &str) -> Option<Value> {
    let (required_pages, tail) = parse_u64_after(message, "requires ")?;
    let (token_count, tail) = parse_u64_after(tail, " for ")?;
    let (pages_free, tail) = parse_u64_after(tail, " but only ")?;
    let (pages_total, tail) = parse_u64_after(tail, " of ")?;
    let (page_size, _) = parse_u64_after(tail, "page_size=")?;
    Some(json!({
        "required_pages": required_pages,
        "token_count": token_count,
        "pages_free": pages_free,
        "pages_total": pages_total,
        "page_size": page_size,
    }))
}

fn scheduler_over_capacity_error_details(message: &str) -> Option<Value> {
    let (active_requests, tail) = parse_u64_after(message, "has ")?;
    let (max_active_requests, _) = parse_u64_after(tail, "max_active_requests=")?;
    Some(json!({
        "active_requests": active_requests,
        "max_active_requests": max_active_requests,
    }))
}

fn parse_u64_after<'a>(input: &'a str, prefix: &str) -> Option<(u64, &'a str)> {
    let (_, tail) = input.split_once(prefix)?;
    let digit_end = tail
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()?;
    let value = tail[..digit_end].parse().ok()?;
    Some((value, &tail[digit_end..]))
}

fn api_error(status: StatusCode, code: &str, message: impl Into<String>) -> Response {
    api_error_with_type(status, code, "invalid_request_error", message)
}

fn api_error_with_type(
    status: StatusCode,
    code: &str,
    error_type: &str,
    message: impl Into<String>,
) -> Response {
    api_error_with_type_and_details(status, code, error_type, message, None)
}

fn api_error_with_type_and_details(
    status: StatusCode,
    code: &str,
    error_type: &str,
    message: impl Into<String>,
    details: Option<Value>,
) -> Response {
    let mut error = json!({
        "message": message.into(),
        "type": error_type,
        "code": code,
    });
    if let (Some(details), Value::Object(fields)) = (details, &mut error) {
        fields.insert("details".to_string(), details);
    }
    (
        status,
        Json(json!({
            "error": error
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
    format!("chatcmpl-hi-local-{created}")
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
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopSequences>,
}

impl ChatCompletionRequest {
    fn stop_sequences(&self) -> Vec<String> {
        self.stop
            .clone()
            .map(StopSequences::into_vec)
            .unwrap_or_default()
    }
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
                    if part
                        .get("type")
                        .and_then(Value::as_str)
                        .is_some_and(|kind| kind == "image_url")
                        || part.get("image_url").is_some()
                    {
                        return Some("<|vision_start|><|image_pad|><|vision_end|>");
                    }
                    if part
                        .get("type")
                        .and_then(Value::as_str)
                        .is_some_and(|kind| kind == "video" || kind == "video_url")
                        || part.get("video").is_some()
                        || part.get("video_url").is_some()
                    {
                        return Some("<|vision_start|><|video_pad|><|vision_end|>");
                    }
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

    pub fn has_image_input(&self) -> bool {
        match self.content.as_ref() {
            Some(Value::Array(parts)) => parts.iter().any(|part| {
                part.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind == "image_url")
                    || part.get("image_url").is_some()
            }),
            _ => false,
        }
    }

    pub fn has_video_input(&self) -> bool {
        match self.content.as_ref() {
            Some(Value::Array(parts)) => parts.iter().any(|part| {
                part.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind == "video" || kind == "video_url")
                    || part.get("video").is_some()
                    || part.get("video_url").is_some()
            }),
            _ => false,
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
    use std::sync::atomic::{AtomicBool, Ordering};

    use async_trait::async_trait;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use futures_util::stream;
    use serde_json::json;
    use tower::ServiceExt;

    use crate::backend::{BackendHealth, InferenceBackend};
    use crate::model::{ModelFamily, ModelInfo, TokenizerInfo, WeightShard};
    use crate::test_support::MockBackend;

    use super::*;

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
        assert_eq!(body["backend"], "mock");
        assert_eq!(body["execution"]["status"], "unavailable");
        assert_eq!(body["cuda_runtime"]["status"], "unavailable");
        assert_eq!(body["gpu_weights"]["status"], "unavailable");
        assert_eq!(body["gpu_matrices"]["status"], "unavailable");
        assert_eq!(body["gpu_vectors"]["status"], "unavailable");
        assert_eq!(body["scheduler"]["status"], "unavailable");
        assert_eq!(body["kv_cache"]["status"], "unavailable");
        assert_eq!(body["attention"]["status"], "unavailable");
        assert_eq!(body["sampling"]["status"], "unavailable");
        assert_eq!(body["batch"]["status"], "unavailable");
        assert_eq!(body["multimodal_batching"]["status"], "unavailable");
        assert_eq!(body["multimodal_metrics"]["status"], "unavailable");
        assert_eq!(body["multimodal_projector"]["status"], "unavailable");
        assert_eq!(body["multimodal"]["status"], "text-only");
    }

    #[tokio::test]
    async fn health_returns_kv_cache_metrics_object() {
        let backend = Arc::new(MockBackend::new(test_model(), "hello").with_quantization(
            "Q4_0; execution=gpu; kv-cache=paged(page_size=16,pages_total=512,pages_free=384,pages_used=128,peak_pages_used=256,pages_free_per_mille=750,pages_used_per_mille=250,peak_pages_used_per_mille=500,pressure=normal,bytes_per_page=65536,bytes_total=33554432,allocations=7,allocation_failures=2,kernel_backend=paged-single-text,batched_backend=paged-text,multimodal_backend=legacy); scheduler=disabled",
        ));
        let app = app(backend);

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
        assert_eq!(body["kv_cache"]["status"], "paged");
        assert_eq!(body["kv_cache"]["page_size"], 16);
        assert_eq!(body["kv_cache"]["pages_total"], 512);
        assert_eq!(body["kv_cache"]["pages_free"], 384);
        assert_eq!(body["kv_cache"]["pages_used"], 128);
        assert_eq!(body["kv_cache"]["peak_pages_used"], 256);
        assert_eq!(body["kv_cache"]["pages_free_per_mille"], 750);
        assert_eq!(body["kv_cache"]["pages_used_per_mille"], 250);
        assert_eq!(body["kv_cache"]["peak_pages_used_per_mille"], 500);
        assert_eq!(body["kv_cache"]["pressure"], "normal");
        assert_eq!(body["kv_cache"]["bytes_per_page"], 65536);
        assert_eq!(body["kv_cache"]["bytes_total"], 33554432);
        assert_eq!(body["kv_cache"]["allocations"], 7);
        assert_eq!(body["kv_cache"]["allocation_failures"], 2);
        assert_eq!(body["kv_cache"]["kernel_backend"], "paged-single-text");
        assert_eq!(body["kv_cache"]["batched_backend"], "paged-text");
        assert_eq!(body["kv_cache"]["multimodal_backend"], "legacy");
    }

    #[tokio::test]
    async fn health_returns_prefix_cache_metrics_object() {
        let backend = Arc::new(MockBackend::new(test_model(), "hello").with_quantization(
            "Q4_0; execution=gpu; prefix-cache=enabled(scope=batch,backend=paged-shared-prefix,batches=3,requests=5,tokens_total=13); scheduler=disabled",
        ));
        let app = app(backend);

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
        assert_eq!(body["prefix_cache"]["status"], "enabled");
        assert_eq!(body["prefix_cache"]["scope"], "batch");
        assert_eq!(body["prefix_cache"]["backend"], "paged-shared-prefix");
        assert_eq!(body["prefix_cache"]["batches"], 3);
        assert_eq!(body["prefix_cache"]["requests"], 5);
        assert_eq!(body["prefix_cache"]["tokens_total"], 13);
    }

    #[tokio::test]
    async fn health_returns_scheduler_metrics_object() {
        let backend = Arc::new(MockBackend::new(test_model(), "hello").with_quantization(
            "Q4_0; execution=gpu; scheduler=enabled,mode=continuous-iteration,continuous_kv_backend=append-only-paged,total_batches=2,total_requests=3,gpu_batched_batches=1,gpu_batched_requests=2,paged_lease_batched_requests=3,paged_internal_batched_requests=4,greedy_batched_requests=5,sampled_batched_requests=6,shared_prefix_batches=7,shared_prefix_requests=8,shared_prefix_tokens_total=9,fallback_requests=10,fallback_single_request=11,fallback_moe_requests=12,fallback_multimodal_requests=13,fallback_tokenizer_requests=14,fallback_bucket_requests=15,fallback_other_requests=16,batch_plan_eligible_requests=17,batch_plan_ineligible_requests=18,token_budget_split_groups=53,token_budget_split_requests=54,token_budget_split_chunks=55,continuous_step_admissions=61,continuous_admitted_requests=62,continuous_decode_iterations=63,continuous_prefill_batches=66,continuous_prefill_requests=67,continuous_append_decode_batches=68,continuous_append_decode_requests=69,continuous_mid_decode_retirements=64,continuous_mid_decode_cancellations=65,completed_requests=2,errored_requests=1,cancelled_requests=19,capacity_rejected_requests=4,capacity_max_active_rejected_requests=49,capacity_rejected_active_requests_total=50,capacity_rejected_active_requests_max=51,capacity_rejected_active_requests_avg=52,admission_rejected_requests=20,admission_errored_requests=21,admission_cancelled_requests=22,admission_context_rejected_requests=45,admission_invalid_request_rejected_requests=46,admission_memory_rejected_requests=47,admission_other_rejected_requests=48,admission_requested_pages=23,admission_granted_pages=24,admission_rejected_pages=25,admission_page_requests=56,admission_requested_pages_max=57,admission_requested_pages_avg=58,admission_granted_pages_max=59,admission_rejected_pages_max=60,admission_page_grant_per_mille=489,admission_page_reject_per_mille=510,max_observed=26,peak_pending=27,pending=28,queued_requests=29,queue_waited_requests=30,queue_wait_micros_total=31,queue_wait_micros_max=32,peak_inflight_requests=33,inflight_requests=34,active=35,active_requests=36,prefill_tokens_total=37,decode_tokens_total=38,prefill_micros_total=39,decode_micros_total=40,tokens_per_sec_prefill=41,tokens_per_sec_decode=99,idle_wakeups=42,idle_micros_total=43,idle_micros_max=44; multimodal=text-only",
        ));
        let app = app(backend);

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
        assert_eq!(body["scheduler"]["status"], "enabled");
        assert_eq!(body["scheduler"]["mode"], "continuous-iteration");
        assert_eq!(
            body["scheduler"]["continuous_kv_backend"],
            "append-only-paged"
        );
        assert_eq!(body["scheduler"]["total_requests"], 3);
        assert_eq!(body["scheduler"]["completed_requests"], 2);
        assert_eq!(body["scheduler"]["errored_requests"], 1);
        assert_eq!(body["scheduler"]["capacity_rejected_requests"], 4);
        assert_eq!(body["scheduler"]["tokens_per_sec_decode"], 99);
        assert_eq!(body["scheduler"]["fallbacks"]["total"], 10);
        assert_eq!(body["scheduler"]["fallbacks"]["single_request"], 11);
        assert_eq!(body["scheduler"]["fallbacks"]["moe"], 12);
        assert_eq!(body["scheduler"]["fallbacks"]["multimodal"], 13);
        assert_eq!(body["scheduler"]["fallbacks"]["tokenizer"], 14);
        assert_eq!(body["scheduler"]["fallbacks"]["bucket_singleton"], 15);
        assert_eq!(body["scheduler"]["fallbacks"]["other"], 16);
        assert_eq!(
            body["scheduler"]["admission"]["capacity_rejected_requests"],
            4
        );
        assert_eq!(body["scheduler"]["capacity"]["rejected_requests"], 4);
        assert_eq!(
            body["scheduler"]["capacity"]["max_active_rejected_requests"],
            49
        );
        assert_eq!(body["scheduler"]["capacity"]["active_requests_total"], 50);
        assert_eq!(body["scheduler"]["capacity"]["active_requests_max"], 51);
        assert_eq!(body["scheduler"]["capacity"]["active_requests_avg"], 52);
        assert_eq!(body["scheduler"]["admission"]["rejected_requests"], 20);
        assert_eq!(body["scheduler"]["admission"]["errored_requests"], 21);
        assert_eq!(body["scheduler"]["admission"]["cancelled_requests"], 22);
        assert_eq!(body["scheduler"]["admission"]["requested_pages"], 23);
        assert_eq!(body["scheduler"]["admission"]["granted_pages"], 24);
        assert_eq!(body["scheduler"]["admission"]["rejected_pages"], 25);
        assert_eq!(body["scheduler"]["admission"]["page_requests"], 56);
        assert_eq!(body["scheduler"]["admission"]["requested_pages_max"], 57);
        assert_eq!(body["scheduler"]["admission"]["requested_pages_avg"], 58);
        assert_eq!(body["scheduler"]["admission"]["granted_pages_max"], 59);
        assert_eq!(body["scheduler"]["admission"]["rejected_pages_max"], 60);
        assert_eq!(body["scheduler"]["admission"]["page_grant_per_mille"], 489);
        assert_eq!(body["scheduler"]["admission"]["page_reject_per_mille"], 510);
        assert_eq!(
            body["scheduler"]["admission"]["by_reason"]["context_length"],
            45
        );
        assert_eq!(
            body["scheduler"]["admission"]["by_reason"]["invalid_request"],
            46
        );
        assert_eq!(
            body["scheduler"]["admission"]["by_reason"]["insufficient_gpu_memory"],
            47
        );
        assert_eq!(body["scheduler"]["admission"]["by_reason"]["other"], 48);
        assert_eq!(body["scheduler"]["queue"]["peak_pending"], 27);
        assert_eq!(body["scheduler"]["queue"]["pending"], 28);
        assert_eq!(body["scheduler"]["queue"]["queued_requests"], 29);
        assert_eq!(body["scheduler"]["queue"]["waited_requests"], 30);
        assert_eq!(body["scheduler"]["queue"]["wait_micros_total"], 31);
        assert_eq!(body["scheduler"]["queue"]["wait_micros_max"], 32);
        assert_eq!(body["scheduler"]["queue"]["peak_inflight_requests"], 33);
        assert_eq!(body["scheduler"]["queue"]["inflight_requests"], 34);
        assert_eq!(body["scheduler"]["queue"]["active_batch_size"], 35);
        assert_eq!(body["scheduler"]["queue"]["active_requests"], 36);
        assert_eq!(body["scheduler"]["idle"]["wakeups"], 42);
        assert_eq!(body["scheduler"]["idle"]["micros_total"], 43);
        assert_eq!(body["scheduler"]["idle"]["micros_max"], 44);
        assert_eq!(body["scheduler"]["batching"]["total_batches"], 2);
        assert_eq!(body["scheduler"]["batching"]["total_requests"], 3);
        assert_eq!(body["scheduler"]["batching"]["gpu_batched_batches"], 1);
        assert_eq!(body["scheduler"]["batching"]["gpu_batched_requests"], 2);
        assert_eq!(
            body["scheduler"]["batching"]["paged_lease_batched_requests"],
            3
        );
        assert_eq!(
            body["scheduler"]["batching"]["paged_internal_batched_requests"],
            4
        );
        assert_eq!(body["scheduler"]["batching"]["greedy_batched_requests"], 5);
        assert_eq!(body["scheduler"]["batching"]["sampled_batched_requests"], 6);
        assert_eq!(body["scheduler"]["batching"]["eligible_requests"], 17);
        assert_eq!(body["scheduler"]["batching"]["ineligible_requests"], 18);
        assert_eq!(body["scheduler"]["batching"]["max_observed_batch"], 26);
        assert_eq!(body["scheduler"]["token_budget"]["split_groups"], 53);
        assert_eq!(body["scheduler"]["token_budget"]["split_requests"], 54);
        assert_eq!(body["scheduler"]["token_budget"]["split_chunks"], 55);
        assert_eq!(body["scheduler"]["continuous"]["step_admissions"], 61);
        assert_eq!(body["scheduler"]["continuous"]["admitted_requests"], 62);
        assert_eq!(body["scheduler"]["continuous"]["decode_iterations"], 63);
        assert_eq!(body["scheduler"]["continuous"]["prefill_batches"], 66);
        assert_eq!(body["scheduler"]["continuous"]["prefill_requests"], 67);
        assert_eq!(body["scheduler"]["continuous"]["append_decode_batches"], 68);
        assert_eq!(
            body["scheduler"]["continuous"]["append_decode_requests"],
            69
        );
        assert_eq!(
            body["scheduler"]["continuous"]["mid_decode_retirements"],
            64
        );
        assert_eq!(
            body["scheduler"]["continuous"]["mid_decode_cancellations"],
            65
        );
        assert_eq!(body["scheduler"]["shared_prefix"]["batches"], 7);
        assert_eq!(body["scheduler"]["shared_prefix"]["requests"], 8);
        assert_eq!(body["scheduler"]["shared_prefix"]["tokens_total"], 9);
        assert_eq!(body["scheduler"]["requests"]["total"], 3);
        assert_eq!(body["scheduler"]["requests"]["completed"], 2);
        assert_eq!(body["scheduler"]["requests"]["errored"], 1);
        assert_eq!(body["scheduler"]["requests"]["cancelled"], 19);
        assert_eq!(body["scheduler"]["throughput"]["prefill_tokens_total"], 37);
        assert_eq!(body["scheduler"]["throughput"]["decode_tokens_total"], 38);
        assert_eq!(body["scheduler"]["throughput"]["prefill_micros_total"], 39);
        assert_eq!(body["scheduler"]["throughput"]["decode_micros_total"], 40);
        assert_eq!(
            body["scheduler"]["throughput"]["tokens_per_sec_prefill"],
            41
        );
        assert_eq!(body["scheduler"]["throughput"]["tokens_per_sec_decode"], 99);
    }

    #[tokio::test]
    async fn health_returns_multimodal_metrics_object() {
        let backend = Arc::new(MockBackend::new(test_model(), "hello").with_quantization(
            "Q4_0; execution=gpu; multimodal-metrics=enabled,vision_batched_requests=5,vision_batched_batches=2,multimodal_decode_batched_requests=4,multimodal_decode_ragged_batched_requests=3,vision_legacy_requests=7,vision_legacy_batches=3,vision_legacy_media_inputs=9,vision_legacy_projected_rows=128,multimodal_decode_legacy_requests=6,multimodal_decode_paged_requests=22,multimodal_requests=11,multimodal_media_inputs=13,multimodal_fallback_requests=14,multimodal_fallback_direct_requests=15,multimodal_fallback_mixed_batch_requests=16,multimodal_fallback_decode_group_singleton_requests=17,multimodal_fallback_projection_failed_requests=18,multimodal_fallback_projection_mismatch_requests=19,multimodal_fallback_incompatible_prompt_requests=20,multimodal_fallback_token_budget_requests=21; scheduler=disabled",
        ));
        let app = app(backend);

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
        assert_eq!(body["multimodal_metrics"]["status"], "enabled");
        assert_eq!(body["multimodal_metrics"]["vision_batched_requests"], 5);
        assert_eq!(body["multimodal_metrics"]["vision_batched_batches"], 2);
        assert_eq!(
            body["multimodal_metrics"]["multimodal_decode_batched_requests"],
            4
        );
        assert_eq!(
            body["multimodal_metrics"]["multimodal_decode_ragged_batched_requests"],
            3
        );
        assert_eq!(body["multimodal_metrics"]["vision_legacy_requests"], 7);
        assert_eq!(body["multimodal_metrics"]["vision_legacy_batches"], 3);
        assert_eq!(body["multimodal_metrics"]["vision_legacy_media_inputs"], 9);
        assert_eq!(
            body["multimodal_metrics"]["vision_legacy_projected_rows"],
            128
        );
        assert_eq!(
            body["multimodal_metrics"]["multimodal_decode_legacy_requests"],
            6
        );
        assert_eq!(
            body["multimodal_metrics"]["multimodal_decode_paged_requests"],
            22
        );
        assert_eq!(body["multimodal_metrics"]["multimodal_requests"], 11);
        assert_eq!(body["multimodal_metrics"]["multimodal_media_inputs"], 13);
        assert_eq!(
            body["multimodal_metrics"]["multimodal_fallback_requests"],
            14
        );
        assert_eq!(body["multimodal_metrics"]["fallbacks"]["total"], 14);
        assert_eq!(body["multimodal_metrics"]["fallbacks"]["direct"], 15);
        assert_eq!(body["multimodal_metrics"]["fallbacks"]["mixed_batch"], 16);
        assert_eq!(
            body["multimodal_metrics"]["fallbacks"]["decode_group_singleton"],
            17
        );
        assert_eq!(
            body["multimodal_metrics"]["fallbacks"]["projection_failed"],
            18
        );
        assert_eq!(
            body["multimodal_metrics"]["fallbacks"]["projection_mismatch"],
            19
        );
        assert_eq!(
            body["multimodal_metrics"]["fallbacks"]["incompatible_prompt"],
            20
        );
        assert_eq!(body["multimodal_metrics"]["fallbacks"]["token_budget"], 21);
    }

    #[tokio::test]
    async fn health_returns_cuda_runtime_metric_objects() {
        let backend = Arc::new(MockBackend::new(test_model(), "hello").with_quantization(
            "Q4_0; execution=gpu; cuda-runtime=available(devices=1,runtime=13000,driver=13010); gpu-weights=loaded(tensors=11,bytes=92); gpu-matrices=loaded(count=10,bytes=80,quantized=4); gpu-vectors=loaded(count=1,bytes=12); attention=tiled-paged; attention-detail=mode=tiled-paged,head_dim=128,head_dim_max=512,wide_kernel=tiled-wide,wide_head_dim_max=512,kv_cache=paged,decode=tiled-paged(paths=single-text|batched-text,fallback_reason=none),decode_fallback=paged(head_dim_min=513,fallback_reason=wide-head),prefill=tiled-contiguous(paths=text|batched-text|multimodal,fallback_reason=none),prefill_fallback=legacy(head_dim_min=513,fallback_reason=wide-head),multimodal=tiled-contiguous(prompt_embeddings=tiled-contiguous,fallback_reason=none); sampling=batched; batch=max_size=8,max_active_requests=4,max_batched_tokens=8192,max_wait_us=2500,max_wait_us_cap=60000000; multimodal-batching=enabled(vision_stage=batching-enabled,decode=ragged-paged|mrope-ragged-paged,fallback=projection-failed|unsupported-decoder-layout|page-exhaustion|malformed-or-no-placeholder); multimodal=text-only",
        ));
        let app = app(backend);

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
        assert_eq!(body["execution"]["status"], "gpu");
        assert_eq!(body["cuda_runtime"]["status"], "available");
        assert_eq!(body["cuda_runtime"]["devices"], 1);
        assert_eq!(body["cuda_runtime"]["runtime"], 13000);
        assert_eq!(body["cuda_runtime"]["driver"], 13010);
        assert_eq!(body["gpu_weights"]["status"], "loaded");
        assert_eq!(body["gpu_weights"]["tensors"], 11);
        assert_eq!(body["gpu_weights"]["bytes"], 92);
        assert_eq!(body["gpu_matrices"]["status"], "loaded");
        assert_eq!(body["gpu_matrices"]["count"], 10);
        assert_eq!(body["gpu_matrices"]["bytes"], 80);
        assert_eq!(body["gpu_matrices"]["quantized"], 4);
        assert_eq!(body["gpu_vectors"]["status"], "loaded");
        assert_eq!(body["gpu_vectors"]["count"], 1);
        assert_eq!(body["gpu_vectors"]["bytes"], 12);
        assert_eq!(body["attention"]["status"], "tiled-paged");
        assert_eq!(
            body["attention"]["detail"],
            "mode=tiled-paged,head_dim=128,head_dim_max=512,wide_kernel=tiled-wide,wide_head_dim_max=512,kv_cache=paged,decode=tiled-paged(paths=single-text|batched-text,fallback_reason=none),decode_fallback=paged(head_dim_min=513,fallback_reason=wide-head),prefill=tiled-contiguous(paths=text|batched-text|multimodal,fallback_reason=none),prefill_fallback=legacy(head_dim_min=513,fallback_reason=wide-head),multimodal=tiled-contiguous(prompt_embeddings=tiled-contiguous,fallback_reason=none)"
        );
        assert_eq!(body["attention"]["mode"], "tiled-paged");
        assert_eq!(body["attention"]["head_dim"], 128);
        assert_eq!(body["attention"]["head_dim_max"], 512);
        assert_eq!(body["attention"]["wide_kernel"], "tiled-wide");
        assert_eq!(body["attention"]["wide_head_dim_max"], 512);
        assert_eq!(body["attention"]["kv_cache"], "paged");
        assert_eq!(body["attention"]["decode"]["status"], "tiled-paged");
        assert_eq!(
            body["attention"]["decode"]["paths"],
            "single-text|batched-text"
        );
        assert_eq!(body["attention"]["decode"]["fallback_reason"], "none");
        assert_eq!(body["attention"]["decode_fallback"]["status"], "paged");
        assert_eq!(body["attention"]["decode_fallback"]["head_dim_min"], 513);
        assert_eq!(
            body["attention"]["decode_fallback"]["fallback_reason"],
            "wide-head"
        );
        assert_eq!(body["attention"]["prefill"]["status"], "tiled-contiguous");
        assert_eq!(
            body["attention"]["prefill"]["paths"],
            "text|batched-text|multimodal"
        );
        assert_eq!(body["attention"]["prefill"]["fallback_reason"], "none");
        assert_eq!(body["attention"]["prefill_fallback"]["status"], "legacy");
        assert_eq!(body["attention"]["prefill_fallback"]["head_dim_min"], 513);
        assert_eq!(
            body["attention"]["prefill_fallback"]["fallback_reason"],
            "wide-head"
        );
        assert_eq!(
            body["attention"]["multimodal"]["status"],
            "tiled-contiguous"
        );
        assert_eq!(
            body["attention"]["multimodal"]["prompt_embeddings"],
            "tiled-contiguous"
        );
        assert_eq!(body["attention"]["multimodal"]["fallback_reason"], "none");
        assert_eq!(body["sampling"]["status"], "batched");
        assert_eq!(body["batch"]["status"], "available");
        assert_eq!(body["batch"]["max_size"], 8);
        assert_eq!(body["batch"]["max_active_requests"], 4);
        assert_eq!(body["batch"]["max_batched_tokens"], 8192);
        assert_eq!(body["batch"]["max_wait_us"], 2500);
        assert_eq!(body["batch"]["max_wait_us_cap"], 60000000);
        assert_eq!(body["multimodal_batching"]["status"], "enabled");
        assert_eq!(
            body["multimodal_batching"]["decode"],
            "ragged-paged|mrope-ragged-paged"
        );
        assert_eq!(
            body["multimodal_batching"]["vision_stage"],
            "batching-enabled"
        );
        assert_eq!(
            body["multimodal_batching"]["fallback"],
            "projection-failed|unsupported-decoder-layout|page-exhaustion|malformed-or-no-placeholder"
        );
        assert_eq!(body["multimodal_projector"]["status"], "text-only");
    }

    #[tokio::test]
    async fn health_returns_attention_fallback_reason() {
        let backend = Arc::new(MockBackend::new(test_model(), "hello").with_quantization(
            "Q4_0; execution=gpu; attention=flash-online; attention-detail=mode=flash-online,head_dim=128,head_dim_max=512,wide_kernel=tiled-wide,wide_head_dim_max=512,kv_cache=legacy,decode=flash-online(paths=single-text|batched-text,fallback_reason=none),decode_fallback=legacy(head_dim_min=513,fallback_reason=wide-head),prefill=tiled-contiguous(paths=text|batched-text|multimodal,fallback_reason=none),prefill_fallback=legacy(head_dim_min=513,fallback_reason=wide-head),multimodal=text-only(prompt_embeddings=unavailable,fallback_reason=no-mmproj)",
        ));
        let app = app(backend);

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
        assert_eq!(body["attention"]["status"], "flash-online");
        assert_eq!(body["attention"]["mode"], "flash-online");
        assert_eq!(body["attention"]["head_dim"], 128);
        assert_eq!(body["attention"]["head_dim_max"], 512);
        assert_eq!(body["attention"]["wide_kernel"], "tiled-wide");
        assert_eq!(body["attention"]["wide_head_dim_max"], 512);
        assert_eq!(body["attention"]["kv_cache"], "legacy");
        assert_eq!(body["attention"]["decode"]["status"], "flash-online");
        assert_eq!(body["attention"]["decode"]["fallback_reason"], "none");
        assert_eq!(
            body["attention"]["decode_fallback"]["fallback_reason"],
            "wide-head"
        );
    }

    #[tokio::test]
    async fn health_returns_multimodal_projector_object() {
        let backend = Arc::new(MockBackend::new(test_model(), "hello").with_quantization(
            "Q4_0; execution=gpu; multimodal=mmproj-loaded(path=/models/vision.gguf,tensors=12,bytes=345,vision-encoder=loaded(variant=qwen2-vl,patch=14,temporal_patch=2,tokens_per_second=2,merge=2,hidden_dim=1280,ff_dim=3420,output_dim=1536,blocks=32,heads=16,device_bytes=123456,matrices=80,vectors=12,window_attention=true);vision=qwen2-vl,generation=enabled)",
        ));
        let app = app(backend);

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
        assert_eq!(body["multimodal_projector"]["status"], "mmproj-loaded");
        assert_eq!(body["multimodal_projector"]["path"], "/models/vision.gguf");
        assert_eq!(body["multimodal_projector"]["tensors"], 12);
        assert_eq!(body["multimodal_projector"]["bytes"], 345);
        assert_eq!(body["multimodal_projector"]["vision"], "qwen2-vl");
        assert_eq!(body["multimodal_projector"]["generation"], "enabled");
        assert_eq!(
            body["multimodal_projector"]["vision_encoder"]["status"],
            "loaded"
        );
        assert_eq!(
            body["multimodal_projector"]["vision_encoder"]["variant"],
            "qwen2-vl"
        );
        assert_eq!(body["multimodal_projector"]["vision_encoder"]["patch"], 14);
        assert_eq!(
            body["multimodal_projector"]["vision_encoder"]["temporal_patch"],
            2
        );
        assert_eq!(
            body["multimodal_projector"]["vision_encoder"]["tokens_per_second"],
            2
        );
        assert_eq!(body["multimodal_projector"]["vision_encoder"]["merge"], 2);
        assert_eq!(
            body["multimodal_projector"]["vision_encoder"]["hidden_dim"],
            1280
        );
        assert_eq!(
            body["multimodal_projector"]["vision_encoder"]["ff_dim"],
            3420
        );
        assert_eq!(
            body["multimodal_projector"]["vision_encoder"]["output_dim"],
            1536
        );
        assert_eq!(body["multimodal_projector"]["vision_encoder"]["blocks"], 32);
        assert_eq!(body["multimodal_projector"]["vision_encoder"]["heads"], 16);
        assert_eq!(
            body["multimodal_projector"]["vision_encoder"]["device_bytes"],
            123456
        );
        assert_eq!(
            body["multimodal_projector"]["vision_encoder"]["matrices"],
            80
        );
        assert_eq!(
            body["multimodal_projector"]["vision_encoder"]["vectors"],
            12
        );
        assert_eq!(
            body["multimodal_projector"]["vision_encoder"]["window_attention"],
            true
        );
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
    async fn generation_error_response_maps_insufficient_gpu_memory() {
        let response = generation_error_response(
            "insufficient_gpu_memory: CUDA paged KV cache requires 2 page(s) for 128 token(s), but only 1 of 8 page(s) are free (page_size=16)".to_string(),
        );

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "insufficient_gpu_memory");
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["details"]["required_pages"], 2);
        assert_eq!(body["error"]["details"]["token_count"], 128);
        assert_eq!(body["error"]["details"]["pages_free"], 1);
        assert_eq!(body["error"]["details"]["pages_total"], 8);
        assert_eq!(body["error"]["details"]["page_size"], 16);
    }

    #[tokio::test]
    async fn generation_error_response_maps_scheduler_over_capacity() {
        let response = generation_error_response(
            "scheduler_over_capacity: CUDA generation scheduler has 1 active request(s), max_active_requests=1"
                .to_string(),
        );

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "scheduler_over_capacity");
        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert_eq!(body["error"]["details"]["active_requests"], 1);
        assert_eq!(body["error"]["details"]["max_active_requests"], 1);
    }

    #[tokio::test]
    async fn generation_error_response_maps_context_length_exceeded() {
        let response = generation_error_response(
            "context_length_exceeded: prompt length 128 plus max_tokens 1 exceeds qwen context length 128"
                .to_string(),
        );

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "context_length_exceeded");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn generation_error_response_maps_invalid_request_parameter() {
        let response = generation_error_response(
            "invalid_request_parameter: max_tokens must be greater than 0".to_string(),
        );

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "invalid_request_parameter");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn generation_error_response_maps_invalid_sampling_parameter() {
        let response = generation_error_response(
            "invalid_sampling_parameter: top_p must be a finite number between 0 and 1".to_string(),
        );

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "invalid_sampling_parameter");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn streaming_completion_maps_admission_error_to_http_response() {
        let backend = Arc::new(MockBackend::new(test_model(), "unused").with_stream_error(
            "scheduler_over_capacity: CUDA generation scheduler has 1 active request(s), max_active_requests=1",
        ));
        let app = app(backend);
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

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "scheduler_over_capacity");
        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert_eq!(body["error"]["details"]["active_requests"], 1);
        assert_eq!(body["error"]["details"]["max_active_requests"], 1);
    }

    #[tokio::test]
    async fn completion_uses_backend_sampling_defaults() {
        let backend = Arc::new(
            MockBackend::new(test_model(), "hello").with_sampling_defaults(
                crate::SamplingDefaults {
                    temperature: 0.0,
                    top_p: 1.0,
                },
            ),
        );
        let app = app(backend.clone());
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
        let generation = backend.last_request().await.unwrap();
        assert_eq!(generation.temperature, 0.0);
        assert_eq!(generation.top_p, 1.0);
        assert_eq!(generation.top_k, None);
        assert_eq!(generation.seed, None);
    }

    #[tokio::test]
    async fn completion_parses_seed_and_top_k() {
        let backend = Arc::new(MockBackend::new(test_model(), "hello"));
        let app = app(backend.clone());
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{"role":"user","content":"hi"}],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 12,
            "seed": 99
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
        let generation = backend.last_request().await.unwrap();
        assert_eq!(generation.temperature, 0.7);
        assert_eq!(generation.top_p, 0.8);
        assert_eq!(generation.top_k, Some(12));
        assert_eq!(generation.seed, Some(99));
    }

    #[tokio::test]
    async fn completion_rejects_invalid_sampling_parameters() {
        for invalid in [
            json!({"temperature": -0.1}),
            json!({"top_p": 1.5}),
            json!({"top_k": 0}),
        ] {
            let backend = Arc::new(MockBackend::new(test_model(), "hello"));
            let app = app(backend.clone());
            let mut request = json!({
                "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
                "messages": [{"role":"user","content":"hi"}]
            });
            request
                .as_object_mut()
                .unwrap()
                .extend(invalid.as_object().unwrap().clone());

            let response = app
                .oneshot(
                    Request::post("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(request.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = body_json(response).await;
            assert_eq!(body["error"]["code"], "invalid_sampling_parameter");
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert!(backend.last_request().await.is_none());
        }
    }

    #[tokio::test]
    async fn completion_rejects_zero_max_tokens() {
        for invalid in [
            json!({"max_tokens": 0}),
            json!({"max_completion_tokens": 0}),
        ] {
            let backend = Arc::new(MockBackend::new(test_model(), "hello"));
            let app = app(backend.clone());
            let mut request = json!({
                "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
                "messages": [{"role":"user","content":"hi"}]
            });
            request
                .as_object_mut()
                .unwrap()
                .extend(invalid.as_object().unwrap().clone());

            let response = app
                .oneshot(
                    Request::post("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(request.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = body_json(response).await;
            assert_eq!(body["error"]["code"], "invalid_request_parameter");
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert!(backend.last_request().await.is_none());
        }
    }

    #[tokio::test]
    async fn completion_rejects_max_tokens_above_model_limit() {
        for invalid in [
            json!({"max_tokens": 3}),
            json!({"max_completion_tokens": 3}),
        ] {
            let mut model = test_model();
            model.max_output_tokens = 2;
            let backend = Arc::new(MockBackend::new(model, "hello"));
            let app = app(backend.clone());
            let mut request = json!({
                "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
                "messages": [{"role":"user","content":"hi"}]
            });
            request
                .as_object_mut()
                .unwrap()
                .extend(invalid.as_object().unwrap().clone());

            let response = app
                .oneshot(
                    Request::post("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(request.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = body_json(response).await;
            assert_eq!(body["error"]["code"], "invalid_request_parameter");
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert!(
                body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("max_output_tokens 2")
            );
            assert!(backend.last_request().await.is_none());
        }
    }

    #[tokio::test]
    async fn completion_uses_backend_chat_template() {
        let template = "{% for message in messages %}[{{ message['role'] }}] {{ message['content'] | trim }}\n{% endfor %}{% if add_generation_prompt %}[assistant] {% endif %}";
        let backend =
            Arc::new(MockBackend::new(test_model(), "hello").with_chat_template(template));
        let app = app(backend.clone());
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{"role":"user","content":" hi "}],
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
        let prompt = backend.last_prompt().await.unwrap();
        assert_eq!(prompt, "[user] hi\n[assistant] ");
    }

    #[tokio::test]
    async fn image_input_without_mmproj_returns_multimodal_error() {
        let app = test_app("unused").await;
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{
                "role":"user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                ]
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

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "unsupported_multimodal");
    }

    #[tokio::test]
    async fn data_image_input_is_passed_to_multimodal_backend() {
        let backend = Arc::new(
            MockBackend::new(test_model(), "described")
                .with_multimodal_support(crate::MultimodalSupport::image_generation("mock-mm")),
        );
        let app = app(backend.clone());
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{
                "role":"user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAEC", "detail": "low"}}
                ]
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

        assert_eq!(response.status(), StatusCode::OK);
        let generation = backend.last_request().await.unwrap();
        assert_eq!(generation.media_inputs.len(), 1);
        let MultimodalInput::Image(image) = &generation.media_inputs[0] else {
            panic!("expected image media input");
        };
        assert_eq!(image.detail.as_deref(), Some("low"));
        assert_eq!(
            image.source,
            ImageSource::Data {
                media_type: "image/png".to_string(),
                bytes: vec![0, 1, 2],
            }
        );
    }

    #[tokio::test]
    async fn data_video_input_is_passed_to_multimodal_backend() {
        let backend = Arc::new(
            MockBackend::new(test_model(), "described").with_multimodal_support(
                crate::MultimodalSupport::image_video_generation("mock-mm"),
            ),
        );
        let app = app(backend.clone());
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{
                "role":"user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "video", "video": {"url": "data:video/mp4;base64,AAEC", "fps": 2.0, "max_frames": 4}}
                ]
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

        assert_eq!(response.status(), StatusCode::OK);
        let generation = backend.last_request().await.unwrap();
        assert_eq!(generation.media_inputs.len(), 1);
        let MultimodalInput::Video(video) = &generation.media_inputs[0] else {
            panic!("expected video media input");
        };
        assert_eq!(video.fps, Some(2.0));
        assert_eq!(video.max_frames, Some(4));
        assert_eq!(
            video.source,
            VideoSource::Data {
                media_type: "video/mp4".to_string(),
                bytes: vec![0, 1, 2],
            }
        );
    }

    #[tokio::test]
    async fn http_image_url_requires_explicit_enablement() {
        let backend = Arc::new(
            MockBackend::new(test_model(), "unused")
                .with_multimodal_support(crate::MultimodalSupport::image_generation("mock-mm")),
        );
        let app = app(backend);
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{
                "role":"user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
                ]
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

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "unsupported_image_url");
    }

    #[tokio::test]
    async fn http_video_url_requires_explicit_enablement() {
        let backend = Arc::new(
            MockBackend::new(test_model(), "unused").with_multimodal_support(
                crate::MultimodalSupport::image_video_generation("mock-mm"),
            ),
        );
        let app = app(backend);
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{
                "role":"user",
                "content": [
                    {"type": "video", "video": {"url": "https://example.com/video.mp4"}}
                ]
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

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "unsupported_video_url");
    }

    #[tokio::test]
    async fn enabled_http_image_url_is_passed_to_multimodal_backend() {
        let backend = Arc::new(
            MockBackend::new(test_model(), "described")
                .with_multimodal_support(crate::MultimodalSupport::image_generation("mock-mm")),
        );
        let app = app_with_config(
            backend.clone(),
            ServerConfig {
                image_url_policy: ImageUrlPolicy {
                    allow_http_urls: true,
                    allow_local_urls: false,
                },
            },
        );
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{
                "role":"user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
                ]
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

        assert_eq!(response.status(), StatusCode::OK);
        let generation = backend.last_request().await.unwrap();
        let MultimodalInput::Image(image) = &generation.media_inputs[0] else {
            panic!("expected image media input");
        };
        assert_eq!(
            image.source,
            ImageSource::Url {
                kind: ImageUrlKind::Http,
                url: "https://example.com/image.png".to_string(),
            }
        );
    }

    #[tokio::test]
    async fn unsupported_media_part_returns_clear_error() {
        let app = test_app("unused").await;
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{
                "role":"user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}
                ]
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

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "unsupported_content_part");
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
        assert!(
            body.find(r#""delta":{"content":"hello from mlx"}"#)
                < body.find(r#""finish_reason":"stop""#),
            "content delta must be emitted before finish: {body}"
        );
    }

    #[tokio::test]
    async fn non_streaming_completion_applies_stop_sequence() {
        let app = test_app("alpha STOP beta").await;
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{"role":"user","content":"hi"}],
            "stop": [" STOP"]
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
        assert_eq!(body["choices"][0]["message"]["content"], "alpha");
    }

    #[tokio::test]
    async fn completion_passes_stop_sequences_to_backend_request() {
        let backend = Arc::new(MockBackend::new(test_model(), "alpha STOP beta"));
        let app = app(backend.clone());
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{"role":"user","content":"hi"}],
            "stop": [" STOP"]
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
        let generation = backend.last_request().await.unwrap();
        assert_eq!(generation.stop_sequences, vec![" STOP".to_string()]);
    }

    #[tokio::test]
    async fn streaming_completion_applies_stop_sequence_across_chunks() {
        let prefix = "x".repeat(511);
        let app = test_app(&format!("{prefix}STOP tail")).await;
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{"role":"user","content":"hi"}],
            "stream": true,
            "stop": "STOP"
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
        assert!(body.contains(&prefix));
        assert!(!body.contains("STOP"));
        assert!(!body.contains("tail"));
        assert!(body.contains(r#""finish_reason":"stop""#));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn streaming_completion_drops_backend_stream_after_stop_sequence() {
        let dropped_before_finished = Arc::new(AtomicBool::new(false));
        let backend = Arc::new(DropTrackingBackend {
            model: test_model(),
            dropped_before_finished: dropped_before_finished.clone(),
        });
        let app = app(backend);
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{"role":"user","content":"hi"}],
            "stream": true,
            "stop": "STOP"
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
        assert!(body.contains("alpha "));
        assert!(!body.contains("STOP"));
        assert!(!body.contains("tail"));
        assert!(body.contains(r#""finish_reason":"stop""#));
        assert!(body.contains("data: [DONE]"));
        assert!(dropped_before_finished.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn streaming_completion_drops_backend_stream_when_response_body_drops() {
        let dropped_before_finished = Arc::new(AtomicBool::new(false));
        let backend = Arc::new(DropTrackingBackend {
            model: test_model(),
            dropped_before_finished: dropped_before_finished.clone(),
        });
        let request = GenerationRequest {
            prompt: "hi".to_string(),
            max_tokens: 8,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            seed: None,
            stop_sequences: Vec::new(),
            media_inputs: Vec::new(),
        };
        let stream_future: GenerationStreamFuture =
            Box::pin(async move { backend.stream_generate(request).await });
        let mut state = OpenAiSseState {
            id: "chatcmpl-test".to_string(),
            model: "test-model".to_string(),
            tools: Vec::new(),
            created: 0,
            include_usage: false,
            tool_mode: false,
            buffered_text: String::new(),
            final_usage: None,
            stop_filter: StopStreamFilter::new(Vec::new()),
            stream_future: Some(stream_future),
            stream: None,
            pending: std::collections::VecDeque::new(),
            finished: false,
        };

        let event = state.next_event().await.unwrap().unwrap();
        assert!(format!("{event:?}").contains("alpha"));
        assert!(!dropped_before_finished.load(Ordering::SeqCst));

        drop(state);
        assert!(dropped_before_finished.load(Ordering::SeqCst));
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

    #[tokio::test]
    async fn streaming_tool_output_is_normalized_to_openai_tool_calls() {
        let app = test_app(r#"{"name":"read","arguments":{"path":"README.md"}}"#).await;
        let request = json!({
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX",
            "messages": [{"role":"user","content":"read README"}],
            "stream": true,
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

        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains(
            r#""tool_calls":[{"function":{"arguments":"{\"path\":\"README.md\"}","name":"read"}"#
        ));
        assert!(body.contains(r#""finish_reason":"tool_calls""#));
        assert!(body.contains("data: [DONE]"));
    }

    async fn test_app(output: &str) -> Router {
        let backend = Arc::new(MockBackend::new(test_model(), output));
        app(backend)
    }

    async fn body_json(response: axum::response::Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    struct DropTrackingBackend {
        model: ModelInfo,
        dropped_before_finished: Arc<AtomicBool>,
    }

    struct DropTrackingStreamState {
        step: usize,
        prompt_tokens: u64,
        dropped_before_finished: Arc<AtomicBool>,
        finished: bool,
    }

    impl Drop for DropTrackingStreamState {
        fn drop(&mut self) {
            if !self.finished {
                self.dropped_before_finished.store(true, Ordering::SeqCst);
            }
        }
    }

    #[async_trait]
    impl InferenceBackend for DropTrackingBackend {
        fn model(&self) -> &ModelInfo {
            &self.model
        }

        fn health(&self) -> BackendHealth {
            BackendHealth {
                backend: "drop-tracking".to_string(),
                ready: true,
                family: self.model.family.label().to_string(),
                quantization: "mock".to_string(),
                context_length: self.model.context_length,
                memory_estimate_bytes: Some(128),
            }
        }

        async fn stream_generate(
            &self,
            request: GenerationRequest,
        ) -> anyhow::Result<GenerationStream> {
            let state = DropTrackingStreamState {
                step: 0,
                prompt_tokens: (request.prompt.len() / 4).max(1) as u64,
                dropped_before_finished: self.dropped_before_finished.clone(),
                finished: false,
            };
            Ok(Box::pin(stream::unfold(state, |mut state| async move {
                match state.step {
                    0 => {
                        state.step = 1;
                        Some((
                            Ok(GenerationEvent::TokenDelta {
                                token_id: 1,
                                text: "alpha ".to_string(),
                            }),
                            state,
                        ))
                    }
                    1 => {
                        state.step = 2;
                        Some((
                            Ok(GenerationEvent::TokenDelta {
                                token_id: 2,
                                text: "STOP".to_string(),
                            }),
                            state,
                        ))
                    }
                    2 => {
                        state.step = 3;
                        Some((
                            Ok(GenerationEvent::TokenDelta {
                                token_id: 3,
                                text: " tail".to_string(),
                            }),
                            state,
                        ))
                    }
                    3 => {
                        state.step = 4;
                        state.finished = true;
                        Some((
                            Ok(GenerationEvent::Finished {
                                output: crate::backend::GenerationOutput {
                                    text: "alpha STOP tail".to_string(),
                                    prompt_tokens: state.prompt_tokens,
                                    completion_tokens: 3,
                                },
                            }),
                            state,
                        ))
                    }
                    _ => None,
                }
            })))
        }
    }

    fn test_model() -> ModelInfo {
        ModelInfo {
            id: "Qwen/Qwen2.5-Coder-1.5B-Instruct-MLX".to_string(),
            path: std::path::PathBuf::from("/tmp/qwen-mlx"),
            family: ModelFamily::Qwen2,
            model_type: "qwen2".to_string(),
            architecture: "Qwen2ForCausalLM".to_string(),
            context_length: Some(32768),
            max_output_tokens: 2048,
            tokenizer: TokenizerInfo {
                tokenizer_json: true,
                tokenizer_config: true,
                special_tokens_map: false,
            },
            chat_template: true,
            weight_shards: vec![WeightShard {
                path: "model.safetensors".to_string(),
                bytes: 128,
                tensor_count: Some(1),
            }],
        }
    }
}
