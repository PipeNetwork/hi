//! Request sealing and executor admission for the `/btw` side loop.

use std::sync::Arc;

use hi_ai::{ToolCall, ToolMode};

use super::{
    BTW_MAX_PARALLEL_TOOLS, BTW_MAX_TOKENS, BTW_TOOL_ALLOWLIST, SealedRequestPolicy, btw_tool_specs,
};

pub(super) async fn seal(
    agent: &mut crate::Agent,
) -> (String, SealedRequestPolicy, SealedRequestPolicy) {
    let model = agent.config.routing.model.clone();
    let tools = btw_tool_specs(agent.request_tools_for(ToolMode::ReadOnly).as_ref());
    let read = agent
        .seal_auxiliary_request(
            &model,
            tools,
            ToolMode::ReadOnly,
            BTW_MAX_TOKENS,
            BTW_MAX_PARALLEL_TOOLS,
        )
        .await;
    let chat = agent
        .seal_auxiliary_request(&model, Arc::new([]), ToolMode::ChatOnly, BTW_MAX_TOKENS, 1)
        .await;
    (model, read, chat)
}

pub(super) fn rejection(policy: &SealedRequestPolicy, call: &ToolCall<'_>) -> Option<String> {
    if !policy.execution_envelope.digest_is_valid()
        || !policy.execution_envelope.matches_specs(&policy.tools)
        || !policy.execution_envelope.admits(call.name)
    {
        return Some(format!(
            "tool `{}` was outside the sealed /btw request envelope",
            call.name
        ));
    }
    if let Some(reason) = crate::heuristics::mode_blocks_tool(policy.tool_mode, call.name) {
        return Some(reason);
    }
    if !BTW_TOOL_ALLOWLIST.contains(&call.name) {
        return Some(format!(
            "tool `{}` is not available on /btw side questions (read-only inspection only)",
            call.name
        ));
    }
    hi_ai::validate_client_tool_call_with_limit(
        call.id,
        call.name,
        call.arguments,
        &policy.tools,
        policy
            .execution_envelope
            .payload
            .limits
            .max_tool_argument_bytes as usize,
    )
    .err()
    .map(|error| format!("tool call rejected by the sealed /btw policy: {error}"))
}
