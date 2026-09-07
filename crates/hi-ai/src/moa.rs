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

use crate::ProviderCapabilityCandidate;
use crate::provider::{
    Provider, ProviderError, ProviderErrorKind, ServedModel, provider_error_kind,
    provider_error_usage,
};

mod request_policy;
use crate::types::{ChatRequest, Completion, Content, Message, Role, StreamEvent, Usage};

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
    fn capabilities(&self) -> crate::ProviderCapabilities {
        request_policy::capabilities(self.passthrough.as_ref(), self.routes.as_ref())
    }

    fn capability_candidates(&self, route: &str, model: &str) -> Vec<ProviderCapabilityCandidate> {
        request_policy::candidates(
            &self.config,
            self.passthrough.as_ref(),
            self.routes.as_ref(),
            route,
            model,
        )
    }

    fn capability_candidates_for_request(
        &self,
        route: &str,
        model: &str,
        context: crate::ProviderRequestContext<'_>,
    ) -> Vec<ProviderCapabilityCandidate> {
        request_policy::candidates_for_request(
            &self.config,
            self.passthrough.as_ref(),
            self.routes.as_ref(),
            route,
            model,
            context,
        )
    }

    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let Some(preset) = self.config.preset_for_model(&request.model).cloned() else {
            return self.passthrough.stream(request, sink).await;
        };

        request
            .execution
            .plan_fanout(u32::try_from(preset.reference_models.len())?.saturating_add(1))?;
        sink(StreamEvent::Status(format!(
            "MoA reference: {}",
            MOA_REFERENCE_CONSERVATIVE
        )));
        let aggregator_reservation = request.execution.reserve(1);
        let (guidance, reference_usage) = self.reference_guidance(&request, &preset, sink).await;
        drop(aggregator_reservation);

        sink(StreamEvent::Status(format!(
            "MoA aggregating: {}",
            preset.aggregator_model
        )));
        let aggregate_request = aggregate_request(request, &preset, guidance);
        let mut completion = match self.routes.stream(aggregate_request, sink).await {
            Ok(completion) => completion,
            Err(err) => {
                // The reference call already ran and was billed. If the
                // aggregator fails, fold that usage into the error rather than
                // dropping it — otherwise the reference model's spend silently
                // vanishes from session accounting on every aggregator error.
                if reference_usage.is_zero() {
                    return Err(err);
                }
                let mut usage = provider_error_usage(&err);
                add_reference_usage(&mut usage, reference_usage);
                let error = crate::provider::provider_error_details(&err)
                    .cloned()
                    .unwrap_or_else(|| {
                        ProviderError::new(
                            provider_error_kind(&err).unwrap_or(ProviderErrorKind::Other),
                            err.to_string(),
                        )
                    })
                    .with_usage(usage);
                return Err(error.into());
            }
        };
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
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> (String, Usage) {
        let reference_model = preset
            .reference_models
            .first()
            .cloned()
            .unwrap_or_else(|| MOA_REFERENCE_CONSERVATIVE.to_string());
        let (max_tokens, envelope) = request_policy::reference_envelope(
            self.routes.as_ref(),
            request,
            preset,
            &reference_model,
        )
        .await;
        let reference_request = request_policy::build_reference_request(
            request,
            reference_model.clone(),
            max_tokens,
            preset.reference_tool_result_budget_chars,
            envelope,
        );
        let mut private_sink = |event: StreamEvent| {
            // Keep advisory content private while retaining physical send audit.
            if matches!(event, StreamEvent::ProviderAttempt(_)) {
                sink(event);
            }
        };

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
            Content::ProviderReplay { .. } => {}
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
            Content::Image { .. } | Content::ProviderReplay { .. } => None,
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

    aggregate.input_tokens = aggregate
        .input_tokens
        .saturating_add(reference.input_tokens);
    aggregate.output_tokens = aggregate
        .output_tokens
        .saturating_add(reference.output_tokens);
    aggregate.cache_read_tokens = aggregate
        .cache_read_tokens
        .saturating_add(reference.cache_read_tokens);
    aggregate.cache_creation_tokens = aggregate
        .cache_creation_tokens
        .saturating_add(reference.cache_creation_tokens);
    aggregate.context_occupancy = aggregate_context;
    aggregate.input_includes_cache = aggregate_includes_cache;
    aggregate.rate_limits = aggregate_rate_limits.or(reference_rate_limits);
    aggregate.estimated |= reference.estimated;
}

#[cfg(test)]
mod tests;
