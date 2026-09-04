use std::sync::Arc;

use hi_ai::{ProviderCapabilities, ToolMode, ToolSpec};

pub(in crate::agent::turn) struct ProviderRequestShape {
    pub tools: Arc<[ToolSpec]>,
    pub tool_mode: ToolMode,
    pub max_output_tokens: u32,
    pub max_parallel_calls: usize,
    pub max_tool_argument_bytes: u32,
    pub max_input_tokens: Option<u32>,
}

impl ProviderRequestShape {
    pub(in crate::agent::turn) fn context_windows(
        &self,
        safety: Option<u32>,
    ) -> crate::agent::ContextWindowLimits {
        crate::agent::ContextWindowLimits {
            safety,
            provider: self.max_input_tokens,
        }
    }

    pub(in crate::agent::turn) fn envelope_limits(
        &self,
        max_output_tokens: u32,
        executed_tool_calls: u32,
    ) -> super::ToolEnvelopeRequestLimits {
        super::ToolEnvelopeRequestLimits {
            max_output_tokens,
            executed_tool_calls,
            max_parallel_calls: self.max_parallel_calls,
            max_tool_argument_bytes: self.max_tool_argument_bytes,
        }
    }
}

pub(in crate::agent::turn) fn constrain(
    tools: Arc<[ToolSpec]>,
    requested_mode: ToolMode,
    requested_output_tokens: u32,
    requested_parallel_calls: usize,
    capabilities: &ProviderCapabilities,
) -> ProviderRequestShape {
    let mut tools = tools.to_vec();
    if let Some(max_tools) = capabilities.request_limits.max_tools {
        tools.truncate(usize::try_from(max_tools).unwrap_or(usize::MAX));
    }

    let tool_mode_supported = capabilities.native_tool_calls
        && match requested_mode {
            ToolMode::Auto | ToolMode::ReadOnly => capabilities.tool_choice.automatic,
            ToolMode::Required => capabilities.tool_choice.required,
            ToolMode::ChatOnly => false,
        };
    let tool_mode =
        if requested_mode == ToolMode::ChatOnly || tools.is_empty() || !tool_mode_supported {
            ToolMode::ChatOnly
        } else {
            requested_mode
        };
    let max_output_tokens = capabilities
        .request_limits
        .max_output_tokens
        .map_or(requested_output_tokens, |limit| {
            requested_output_tokens.min(limit)
        });
    let max_parallel_calls = if capabilities.parallel_tool_calls {
        requested_parallel_calls.max(1)
    } else {
        1
    };
    let client_argument_limit = u32::try_from(hi_ai::MAX_TOOL_ARGUMENT_BYTES).unwrap_or(u32::MAX);
    let max_tool_argument_bytes = capabilities
        .request_limits
        .max_tool_argument_bytes
        .unwrap_or(client_argument_limit)
        .min(client_argument_limit);

    ProviderRequestShape {
        tools: tools.into(),
        tool_mode,
        max_output_tokens,
        max_parallel_calls,
        max_tool_argument_bytes,
        max_input_tokens: capabilities.request_limits.max_input_tokens,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tool(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: String::new(),
            parameters: json!({"type": "object"}),
        }
    }

    #[test]
    fn negotiated_limits_narrow_the_request_and_execution_envelope() {
        let mut capabilities = ProviderCapabilities::native_tools(true);
        capabilities.parallel_tool_calls = false;
        capabilities.request_limits.max_output_tokens = Some(128);
        capabilities.request_limits.max_tools = Some(1);
        capabilities.request_limits.max_tool_argument_bytes = Some(64);
        capabilities.request_limits.max_input_tokens = Some(4_096);

        let shape = constrain(
            vec![tool("read"), tool("grep")].into(),
            ToolMode::Auto,
            1_024,
            8,
            &capabilities,
        );

        assert_eq!(shape.tools.len(), 1);
        assert_eq!(shape.max_output_tokens, 128);
        assert_eq!(shape.max_parallel_calls, 1);
        assert_eq!(shape.max_tool_argument_bytes, 64);
        assert_eq!(shape.max_input_tokens, Some(4_096));
        assert_eq!(shape.tool_mode, ToolMode::Auto);
    }

    #[test]
    fn unknown_tool_protocol_fails_closed_without_discarding_audit_catalog() {
        let shape = constrain(
            vec![tool("read")].into(),
            ToolMode::Auto,
            1_024,
            8,
            &ProviderCapabilities::default(),
        );

        assert_eq!(shape.tools.len(), 1);
        assert_eq!(shape.tool_mode, ToolMode::ChatOnly);
        assert_eq!(shape.max_parallel_calls, 1);
    }
}
