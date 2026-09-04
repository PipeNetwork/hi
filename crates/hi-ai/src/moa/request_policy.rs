//! Capability projection and child-request sealing for the virtual MoA route.

use std::sync::Arc;

use crate::{
    ChatRequest, Provider, ProviderCapabilities, ProviderCapabilityCandidate, RequestProfile,
    RequestToolEnvelope, ToolMode,
};

use super::{MoaConfig, MoaPreset};

pub(super) fn capabilities(
    passthrough: &dyn Provider,
    routes: &dyn Provider,
) -> ProviderCapabilities {
    passthrough
        .capabilities()
        .conservative_intersection(&routes.capabilities())
}

pub(super) fn candidates(
    config: &MoaConfig,
    passthrough: &dyn Provider,
    routes: &dyn Provider,
    route: &str,
    model: &str,
) -> Vec<ProviderCapabilityCandidate> {
    match config.preset_for_model(model) {
        Some(preset) => routes.capability_candidates(route, &preset.aggregator_model),
        None => passthrough.capability_candidates(route, model),
    }
}

pub(super) async fn reference_envelope(
    routes: &dyn Provider,
    request: &crate::ChatRequest,
    preset: &MoaPreset,
    reference_model: &str,
) -> (u32, Arc<RequestToolEnvelope>) {
    let target = crate::CapabilityRoute::new("moa-reference", reference_model);
    let candidates = routes.capability_candidates(&target.route, &target.model);
    let effective = crate::ProviderCapabilityRegistry::default()
        .resolve_candidates(target, &candidates)
        .await;
    let max_tokens = effective
        .capabilities
        .request_limits
        .max_output_tokens
        .map_or(preset.reference_max_tokens, |limit| {
            preset.reference_max_tokens.min(limit)
        });
    let envelope = crate::request_envelope::derived_chat_only(
        request.tool_envelope.as_deref(),
        effective,
        max_tokens,
        "moa-reference",
    );
    (max_tokens, Arc::new(envelope))
}

pub(super) fn build_reference_request(
    request: &ChatRequest,
    reference_model: String,
    max_tokens: u32,
    tool_result_budget_chars: usize,
    tool_envelope: Arc<RequestToolEnvelope>,
) -> ChatRequest {
    ChatRequest {
        model: reference_model,
        request_id: request.request_id.clone(),
        retry_attempt: request.retry_attempt,
        user_turn: false,
        canonical_objective: None,
        messages: Arc::new(super::reference_messages(
            &request.messages,
            tool_result_budget_chars,
        )),
        tools: Arc::from([]),
        tool_envelope: Some(tool_envelope),
        max_tokens: request.max_tokens.min(max_tokens),
        temperature: request.temperature,
        top_p: request.top_p,
        frequency_penalty: request.frequency_penalty,
        thinking_budget: None,
        reasoning_effort: None,
        profile: RequestProfile {
            compat: request.profile.compat,
            tool_mode: ToolMode::ChatOnly,
            stream_usage: request.profile.stream_usage,
            deepseek_compat: request.profile.deepseek_compat,
            deepseek_strict: request.profile.deepseek_strict,
            deepseek_thinking: request.profile.deepseek_thinking,
            output_token_parameter: request.profile.output_token_parameter,
        },
    }
}
