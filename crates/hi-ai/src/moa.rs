//! Opt-in Mixture-of-Agents routing.
//!
//! `MoaProvider` is a composite provider: ordinary model ids are forwarded to
//! the normal provider unchanged, while `moa/conservative` runs a bounded
//! private reference call before the acting aggregator call.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::provider::{Provider, ServedModel, provider_error_usage};
use crate::types::{
    ChatRequest, Completion, Content, Message, RequestProfile, Role, StreamEvent, ToolMode, Usage,
};

pub const MOA_MODEL_CONSERVATIVE: &str = "moa/conservative";
pub const MOA_PRESET_CONSERVATIVE: &str = "conservative";
pub const MOA_AGGREGATOR_CONSERVATIVE: &str = "ipop/coder-balanced";
pub const MOA_REFERENCE_CONSERVATIVE: &str = "pipe/auto-coder";

const REFERENCE_SYSTEM_PROMPT: &str = "You are a private advisory reference model in a bounded \
Mixture-of-Agents route. Review the conversation and provide concise implementation guidance, \
risks, and checks for the acting agent. Do not claim to have executed tools. Your response is \
private guidance, not the final answer to the user.";

const AGGREGATOR_GUIDANCE_PREFIX: &str = "[Private MoA guidance]\n\
The following advisory note came from a separate reference model. Treat it as non-authoritative: \
use it only when it is correct and useful, and do not quote or mention it unless relevant.\n\n";
const AGGREGATOR_GUIDANCE_SUFFIX: &str = "\n[/Private MoA guidance]";

fn default_enabled() -> bool {
    true
}

fn default_preset_name() -> String {
    MOA_PRESET_CONSERVATIVE.to_string()
}

fn default_presets() -> BTreeMap<String, MoaPreset> {
    let mut presets = BTreeMap::new();
    presets.insert(MOA_PRESET_CONSERVATIVE.to_string(), MoaPreset::default());
    presets
}

fn default_aggregator_model() -> String {
    MOA_AGGREGATOR_CONSERVATIVE.to_string()
}

fn default_reference_models() -> Vec<String> {
    vec![MOA_REFERENCE_CONSERVATIVE.to_string()]
}

fn default_reference_max_tokens() -> u32 {
    2048
}

fn default_reference_tool_result_budget_chars() -> usize {
    4000
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoaConfig {
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default = "default_preset_name")]
    pub default_preset: String,
    #[serde(default = "default_presets")]
    pub presets: BTreeMap<String, MoaPreset>,
}

impl Default for MoaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_preset: default_preset_name(),
            presets: default_presets(),
        }
    }
}

impl MoaConfig {
    pub fn validate(&self) -> Result<()> {
        if self.default_preset != MOA_PRESET_CONSERVATIVE {
            bail!(
                "unsupported MoA default_preset '{}'; v1 only supports '{}'",
                self.default_preset,
                MOA_PRESET_CONSERVATIVE
            );
        }
        let Some(preset) = self.presets.get(MOA_PRESET_CONSERVATIVE) else {
            bail!("MoA preset '{}' is required", MOA_PRESET_CONSERVATIVE);
        };
        preset.validate(MOA_PRESET_CONSERVATIVE)
    }

    pub fn preset_for_model(&self, model: &str) -> Option<&MoaPreset> {
        (self.enabled && model == MOA_MODEL_CONSERVATIVE)
            .then(|| self.presets.get(MOA_PRESET_CONSERVATIVE))
            .flatten()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoaPreset {
    #[serde(default = "default_aggregator_model")]
    pub aggregator_model: String,
    #[serde(default = "default_reference_models")]
    pub reference_models: Vec<String>,
    #[serde(default = "default_reference_max_tokens")]
    pub reference_max_tokens: u32,
    #[serde(default = "default_reference_tool_result_budget_chars")]
    pub reference_tool_result_budget_chars: usize,
}

impl Default for MoaPreset {
    fn default() -> Self {
        Self {
            aggregator_model: default_aggregator_model(),
            reference_models: default_reference_models(),
            reference_max_tokens: default_reference_max_tokens(),
            reference_tool_result_budget_chars: default_reference_tool_result_budget_chars(),
        }
    }
}

impl MoaPreset {
    pub fn validate(&self, name: &str) -> Result<()> {
        if is_recursive_moa_route(&self.aggregator_model) {
            bail!(
                "MoA preset '{name}' cannot use recursive aggregator model '{}'",
                self.aggregator_model
            );
        }
        if self.aggregator_model != MOA_AGGREGATOR_CONSERVATIVE {
            bail!(
                "MoA preset '{name}' aggregator_model must be '{}'",
                MOA_AGGREGATOR_CONSERVATIVE
            );
        }
        if self.reference_models != [MOA_REFERENCE_CONSERVATIVE.to_string()] {
            bail!(
                "MoA preset '{name}' reference_models must be ['{}']",
                MOA_REFERENCE_CONSERVATIVE
            );
        }
        if self
            .reference_models
            .iter()
            .any(|model| is_recursive_moa_route(model))
        {
            bail!("MoA preset '{name}' cannot reference another MoA route");
        }
        if self.reference_max_tokens == 0 {
            bail!("MoA preset '{name}' reference_max_tokens must be greater than 0");
        }
        Ok(())
    }
}

fn is_recursive_moa_route(model: &str) -> bool {
    model == MOA_MODEL_CONSERVATIVE || model.starts_with("moa/")
}

pub struct MoaProvider {
    passthrough: Box<dyn Provider>,
    routes: Box<dyn Provider>,
    config: MoaConfig,
}

impl MoaProvider {
    pub fn new(
        passthrough: Box<dyn Provider>,
        routes: Box<dyn Provider>,
        config: MoaConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            passthrough,
            routes,
            config,
        })
    }
}

#[async_trait]
impl Provider for MoaProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let Some(preset) = self.config.preset_for_model(&request.model).cloned() else {
            return self.passthrough.stream(request, sink).await;
        };

        sink(StreamEvent::Status(format!(
            "MoA reference: {}",
            MOA_REFERENCE_CONSERVATIVE
        )));
        let (guidance, reference_usage) = self.reference_guidance(&request, &preset).await;

        sink(StreamEvent::Status(format!(
            "MoA aggregating: {}",
            preset.aggregator_model
        )));
        let aggregate_request = aggregate_request(request, &preset, guidance);
        let mut completion = self.routes.stream(aggregate_request, sink).await?;
        add_reference_usage(&mut completion.usage, reference_usage);
        Ok(completion)
    }

    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        let mut models = self.passthrough.list_models().await?;
        if self.config.enabled && !models.iter().any(|m| m.id == MOA_MODEL_CONSERVATIVE) {
            models.push(virtual_moa_model());
        }
        Ok(models)
    }
}

impl MoaProvider {
    async fn reference_guidance(
        &self,
        request: &ChatRequest,
        preset: &MoaPreset,
    ) -> (String, Usage) {
        let reference_model = preset
            .reference_models
            .first()
            .cloned()
            .unwrap_or_else(|| MOA_REFERENCE_CONSERVATIVE.to_string());
        let reference_request = reference_request(request, preset, reference_model.clone());
        let mut private_sink = |_event: StreamEvent| {};

        match self
            .routes
            .stream(reference_request, &mut private_sink)
            .await
        {
            Ok(completion) => {
                let usage = completion.usage;
                (
                    guidance_from_completion(&reference_model, completion),
                    usage,
                )
            }
            Err(err) => {
                let usage = provider_error_usage(&err);
                (
                    unavailable_guidance(&reference_model, &err.to_string()),
                    usage,
                )
            }
        }
    }
}

fn virtual_moa_model() -> ServedModel {
    ServedModel {
        id: MOA_MODEL_CONSERVATIVE.to_string(),
        context_window: None,
        max_output_tokens: None,
        price: None,
        provider_label: Some("virtual MoA route".to_string()),
        status: Some("virtual".to_string()),
        available: true,
        availability_reason: Some(format!(
            "virtual MoA route: {} reference -> {} aggregator",
            MOA_REFERENCE_CONSERVATIVE, MOA_AGGREGATOR_CONSERVATIVE
        )),
        capabilities: vec!["tools".to_string(), "moa".to_string()],
    }
}

fn reference_request(
    request: &ChatRequest,
    preset: &MoaPreset,
    reference_model: String,
) -> ChatRequest {
    ChatRequest {
        model: reference_model,
        messages: Arc::new(reference_messages(
            &request.messages,
            preset.reference_tool_result_budget_chars,
        )),
        tools: Arc::from([]),
        max_tokens: request.max_tokens.min(preset.reference_max_tokens),
        temperature: request.temperature,
        top_p: request.top_p,
        frequency_penalty: request.frequency_penalty,
        thinking_budget: None,
        profile: RequestProfile {
            compat: request.profile.compat,
            tool_mode: ToolMode::ChatOnly,
            stream_usage: request.profile.stream_usage,
        },
    }
}

fn reference_messages(messages: &[Message], tool_result_budget_chars: usize) -> Vec<Message> {
    let mut out = vec![Message::system(REFERENCE_SYSTEM_PROMPT)];
    for message in messages {
        if message.role == Role::System {
            continue;
        }
        let text = flatten_message_for_reference(message, tool_result_budget_chars);
        if text.trim().is_empty() {
            continue;
        }
        let role = match message.role {
            Role::Assistant => Role::Assistant,
            Role::System | Role::User | Role::Tool => Role::User,
        };
        out.push(Message {
            role,
            content: vec![Content::Text(text)],
        });
    }
    out
}

fn flatten_message_for_reference(message: &Message, tool_result_budget_chars: usize) -> String {
    let mut parts = Vec::new();
    for block in &message.content {
        match block {
            Content::Text(text) => parts.push(text.clone()),
            Content::Thinking { text, .. } => {
                parts.push(format!(
                    "[assistant reasoning]\n{}",
                    truncate_chars(text, tool_result_budget_chars)
                ));
            }
            Content::ToolCall {
                id,
                name,
                arguments,
            } => parts.push(format!(
                "[assistant requested tool `{name}` id `{id}`]\n{}",
                truncate_chars(arguments, tool_result_budget_chars)
            )),
            Content::ToolResult { call_id, output } => parts.push(format!(
                "[tool result for `{call_id}`]\n{}",
                truncate_chars(output, tool_result_budget_chars)
            )),
            Content::Image { .. } => parts.push("[image omitted]".to_string()),
        }
    }
    parts.join("\n\n")
}

fn aggregate_request(
    mut request: ChatRequest,
    preset: &MoaPreset,
    reference_guidance: String,
) -> ChatRequest {
    request.model = preset.aggregator_model.clone();
    let mut messages = request.messages.as_ref().clone();
    append_guidance(&mut messages, reference_guidance);
    request.messages = Arc::new(messages);
    request
}

fn append_guidance(messages: &mut Vec<Message>, guidance: String) {
    let advisory = format!("{AGGREGATOR_GUIDANCE_PREFIX}{guidance}{AGGREGATOR_GUIDANCE_SUFFIX}");
    if let Some(message) = messages.iter_mut().rev().find(|m| m.role == Role::User) {
        message
            .content
            .push(Content::Text(format!("\n\n{advisory}")));
    } else {
        messages.push(Message::user(advisory));
    }
}

fn guidance_from_completion(reference_model: &str, completion: Completion) -> String {
    let text = completion_text(&completion.content);
    if text.trim().is_empty() {
        return unavailable_guidance(reference_model, "returned no usable text");
    }
    format!("Reference `{reference_model}` advisory:\n{text}")
}

fn unavailable_guidance(reference_model: &str, reason: &str) -> String {
    format!("Reference `{reference_model}` was unavailable ({reason}). Proceed without it.")
}

fn completion_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            Content::Text(text) => Some(text.clone()),
            Content::Thinking { text, .. } => Some(format!("[reasoning]\n{text}")),
            Content::ToolCall {
                id,
                name,
                arguments,
            } => Some(format!(
                "[unsupported reference tool request `{name}` id `{id}`]\n{arguments}"
            )),
            Content::ToolResult { call_id, output } => {
                Some(format!("[reference tool result `{call_id}`]\n{output}"))
            }
            Content::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let prefix: String = text.chars().take(max_chars).collect();
    let omitted = text.chars().count().saturating_sub(max_chars);
    format!("{prefix}\n[truncated {omitted} chars]")
}

fn add_reference_usage(aggregate: &mut Usage, reference: Usage) {
    let aggregate_context = aggregate.context_occupancy;
    let aggregate_includes_cache = aggregate.input_includes_cache;
    let aggregate_rate_limits = aggregate.rate_limits;
    let reference_rate_limits = reference.rate_limits;

    aggregate.input_tokens += reference.input_tokens;
    aggregate.output_tokens += reference.output_tokens;
    aggregate.cache_read_tokens += reference.cache_read_tokens;
    aggregate.cache_creation_tokens += reference.cache_creation_tokens;
    aggregate.context_occupancy = aggregate_context;
    aggregate.input_includes_cache = aggregate_includes_cache;
    aggregate.rate_limits = aggregate_rate_limits.or(reference_rate_limits);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{ProviderError, ProviderErrorKind};
    use crate::types::{RateLimitBucket, RateLimitState, ToolSpec};
    use serde_json::json;
    use std::sync::Mutex;

    #[derive(Clone)]
    struct RecordingProvider {
        requests: Arc<Mutex<Vec<ChatRequest>>>,
        results: Arc<Mutex<Vec<Result<Completion, String>>>>,
        models: Vec<ServedModel>,
    }

    impl RecordingProvider {
        fn new(results: Vec<Result<Completion, String>>) -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                results: Arc::new(Mutex::new(results)),
                models: Vec::new(),
            }
        }

        fn with_models(mut self, models: Vec<ServedModel>) -> Self {
            self.models = models;
            self
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Provider for RecordingProvider {
        async fn stream(
            &self,
            request: ChatRequest,
            sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            self.requests.lock().unwrap().push(request);
            sink(StreamEvent::Text("visible".to_string()));
            match self.results.lock().unwrap().remove(0) {
                Ok(completion) => Ok(completion),
                Err(message) => Err(ProviderError::new(ProviderErrorKind::Other, message).into()),
            }
        }

        async fn list_models(&self) -> Result<Vec<ServedModel>> {
            Ok(self.models.clone())
        }
    }

    fn request(model: &str) -> ChatRequest {
        ChatRequest {
            model: model.to_string(),
            messages: Arc::new(vec![
                Message::system("secret system"),
                Message::user("fix this"),
                Message::assistant(vec![Content::ToolCall {
                    id: "call-1".into(),
                    name: "read".into(),
                    arguments: "{\"path\":\"src/main.rs\"}".into(),
                }]),
                Message::tool_result("call-1", "x".repeat(20)),
            ]),
            tools: Arc::from([ToolSpec {
                name: "read".into(),
                description: "read a file".into(),
                parameters: json!({"type":"object"}),
            }]),
            max_tokens: 8192,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: Some(1024),
            profile: RequestProfile::default(),
        }
    }

    fn completion(text: &str, input: u64, output: u64) -> Completion {
        Completion {
            content: vec![Content::Text(text.to_string())],
            usage: Usage {
                input_tokens: input,
                output_tokens: output,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                input_includes_cache: false,
                context_occupancy: input,
                rate_limits: None,
            },
            stop_reason: Some("stop".into()),
        }
    }

    #[test]
    fn reference_view_drops_system_and_flattens_tools() {
        let mut messages = request(MOA_MODEL_CONSERVATIVE).messages.as_ref().clone();
        messages.push(Message::user_with_image("see image", "abc123", "image/png"));
        let out = reference_messages(&messages, 5);
        assert!(out.iter().all(|m| m.text() != "secret system"));
        assert!(out.iter().all(|m| m.role != Role::Tool));
        let text = out.iter().map(Message::text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("[assistant requested tool `read`"));
        assert!(text.contains("[tool result for `call-1`]"));
        assert!(text.contains("[truncated"));
        assert!(text.contains("[image omitted]"));
    }

    #[tokio::test]
    async fn normal_model_requests_bypass_moa() {
        let passthrough = RecordingProvider::new(vec![Ok(completion("direct", 1, 2))]);
        let passthrough_handle = passthrough.clone();
        let routes = RecordingProvider::new(vec![]);
        let routes_handle = routes.clone();
        let provider = MoaProvider::new(
            Box::new(passthrough),
            Box::new(routes),
            MoaConfig::default(),
        )
        .unwrap();
        let mut events = Vec::new();
        let mut sink = |event| events.push(event);
        let out = provider
            .stream(request("ipop/coder-balanced"), &mut sink)
            .await
            .unwrap();
        assert_eq!(completion_text(&out.content), "direct");
        assert_eq!(passthrough_handle.requests().len(), 1);
        assert!(routes_handle.requests().is_empty());
    }

    #[tokio::test]
    async fn aggregator_receives_original_tools_and_guidance() {
        let passthrough = RecordingProvider::new(vec![]);
        let routes = RecordingProvider::new(vec![
            Ok(completion("check the parser edge case", 10, 3)),
            Ok(completion("done", 4, 2)),
        ]);
        let routes_handle = routes.clone();
        let provider = MoaProvider::new(
            Box::new(passthrough),
            Box::new(routes),
            MoaConfig::default(),
        )
        .unwrap();
        let mut events = Vec::new();
        let mut sink = |event| match event {
            StreamEvent::Status(status) => events.push(format!("status:{status}")),
            StreamEvent::Text(text) => events.push(format!("text:{text}")),
            StreamEvent::Reasoning(text) => events.push(format!("reasoning:{text}")),
        };
        let out = provider
            .stream(request(MOA_MODEL_CONSERVATIVE), &mut sink)
            .await
            .unwrap();

        assert_eq!(out.usage.input_tokens, 14);
        assert_eq!(out.usage.output_tokens, 5);
        assert_eq!(
            events,
            vec![
                "status:MoA reference: pipe/auto-coder".to_string(),
                "status:MoA aggregating: ipop/coder-balanced".to_string(),
                "text:visible".to_string()
            ]
        );

        let requests = routes_handle.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].model, MOA_REFERENCE_CONSERVATIVE);
        assert_eq!(requests[0].profile.tool_mode, ToolMode::ChatOnly);
        assert!(requests[0].tools.is_empty());
        assert_eq!(requests[0].max_tokens, 2048);
        assert_eq!(requests[0].thinking_budget, None);

        assert_eq!(requests[1].model, MOA_AGGREGATOR_CONSERVATIVE);
        assert_eq!(requests[1].tools.len(), 1);
        assert_eq!(requests[1].profile.tool_mode, ToolMode::Auto);
        let guidance = requests[1]
            .messages
            .iter()
            .find(|m| m.role == Role::User)
            .unwrap()
            .text();
        assert!(guidance.contains("check the parser edge case"));
        assert!(guidance.contains("[Private MoA guidance]"));
    }

    #[tokio::test]
    async fn reference_failure_still_runs_aggregator() {
        let passthrough = RecordingProvider::new(vec![]);
        let routes = RecordingProvider::new(vec![
            Err("reference down".into()),
            Ok(completion("aggregate", 4, 2)),
        ]);
        let routes_handle = routes.clone();
        let provider = MoaProvider::new(
            Box::new(passthrough),
            Box::new(routes),
            MoaConfig::default(),
        )
        .unwrap();
        let mut sink = |_event| {};
        let out = provider
            .stream(request(MOA_MODEL_CONSERVATIVE), &mut sink)
            .await
            .unwrap();
        assert_eq!(completion_text(&out.content), "aggregate");
        let requests = routes_handle.requests();
        assert_eq!(requests.len(), 2);
        assert!(requests[1].messages.iter().any(|m| {
            m.role == Role::User
                && m.text()
                    .contains("Reference `pipe/auto-coder` was unavailable")
        }));
    }

    #[test]
    fn recursive_moa_routes_are_rejected() {
        let config = MoaConfig {
            presets: BTreeMap::from([(
                MOA_PRESET_CONSERVATIVE.to_string(),
                MoaPreset {
                    reference_models: vec![MOA_MODEL_CONSERVATIVE.to_string()],
                    ..MoaPreset::default()
                },
            )]),
            ..MoaConfig::default()
        };
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("reference_models"));
    }

    #[test]
    fn preserves_aggregator_rate_limits_when_summing_usage() {
        let mut aggregate = Usage {
            input_tokens: 4,
            output_tokens: 2,
            context_occupancy: 4,
            rate_limits: Some(RateLimitState {
                requests_min: RateLimitBucket {
                    limit: 10,
                    remaining: 8,
                    reset_seconds: 1,
                },
                ..RateLimitState::default()
            }),
            ..Usage::default()
        };
        let reference = Usage {
            input_tokens: 10,
            output_tokens: 3,
            context_occupancy: 10,
            rate_limits: Some(RateLimitState {
                requests_min: RateLimitBucket {
                    limit: 100,
                    remaining: 90,
                    reset_seconds: 2,
                },
                ..RateLimitState::default()
            }),
            ..Usage::default()
        };
        add_reference_usage(&mut aggregate, reference);
        assert_eq!(aggregate.input_tokens, 14);
        assert_eq!(aggregate.output_tokens, 5);
        assert_eq!(aggregate.context_occupancy, 4);
        assert_eq!(aggregate.rate_limits.unwrap().requests_min.limit, 10);
    }

    #[tokio::test]
    async fn list_models_adds_virtual_model() {
        let passthrough = RecordingProvider::new(vec![]).with_models(vec![ServedModel {
            id: "real".into(),
            context_window: None,
            max_output_tokens: None,
            price: None,
            provider_label: None,
            status: None,
            available: true,
            availability_reason: None,
            capabilities: Vec::new(),
        }]);
        let provider = MoaProvider::new(
            Box::new(passthrough),
            Box::new(RecordingProvider::new(vec![])),
            MoaConfig::default(),
        )
        .unwrap();
        let models = provider.list_models().await.unwrap();
        assert!(models.iter().any(|m| m.id == "real"));
        assert!(models.iter().any(|m| {
            m.id == MOA_MODEL_CONSERVATIVE
                && m.provider_label.as_deref() == Some("virtual MoA route")
        }));
    }
}
