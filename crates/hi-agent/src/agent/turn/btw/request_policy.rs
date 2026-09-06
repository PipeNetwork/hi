//! Request sealing and executor admission for the `/btw` side loop.

use std::sync::Arc;

use hi_ai::{ToolCall, ToolMode};

use super::{
    BTW_MAX_PARALLEL_TOOLS, BTW_MAX_TOKENS, BTW_TOOL_ALLOWLIST, SealedRequestPolicy, btw_tool_specs,
};

/// `/btw` has no workspace-controller settlement path. Keep its executor to
/// pure, non-mutating local/session inspection even if an allowlisted catalog
/// entry later changes policy. Network and MCP reads are excluded because
/// their observable external effect needs an operation receipt too.
pub(super) fn side_tool_is_pure(name: &str) -> bool {
    BTW_TOOL_ALLOWLIST.contains(&name)
        && hi_tools::tool_metadata(name).is_some_and(|metadata| {
            metadata.read_only
                && !metadata.filesystem_mutating
                && matches!(
                    metadata.policy.effect_scope,
                    hi_tools::catalog::EffectScope::ReadOnly
                )
                && matches!(
                    metadata.policy.replay_class,
                    hi_tools::catalog::ReplayClass::PureWorkspace
                )
                && !metadata.policy.resource_access.workspace_write
                && !metadata.policy.resource_access.network
                && !metadata.policy.resource_access.credentials
                && !metadata.policy.resource_access.mcp
        })
}

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
    if !side_tool_is_pure(call.name) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_cannot_admit_workspace_job_or_external_effects() {
        for name in BTW_TOOL_ALLOWLIST {
            assert!(side_tool_is_pure(name), "unsafe /btw tool policy: {name}");
        }
        for name in [
            "bash_output",
            "get_task_output",
            "wait_tasks",
            "diagnostics",
            "definition",
            "references",
            "hover",
            "web_search",
            "web_fetch",
            "research",
            "research_read",
            "search_tool",
            "memory_search",
            "memory_get",
        ] {
            assert!(!side_tool_is_pure(name), "{name} escaped /btw isolation");
        }
    }
}
