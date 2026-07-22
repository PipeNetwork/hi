//! ChatRequest assembly helpers for the Model phase.
//!
//! Keeps tool-list filtering and schema accounting out of the main
//! `run_model_round` body.

use std::collections::BTreeSet;
use std::sync::Arc;

use hi_ai::ToolSpec;

use super::retry::estimate_tool_schema_tokens;

/// Drop coordination/bookkeeping tools when the one-shot suppress flag is set,
/// unless that would leave the list empty under a required tool choice.
pub(super) fn apply_bookkeeping_suppress(
    tools: Arc<[ToolSpec]>,
    suppress: bool,
) -> Arc<[ToolSpec]> {
    if !suppress {
        return tools;
    }
    if tools
        .iter()
        .any(|tool| !hi_tools::is_coordination(&tool.name))
    {
        tools
            .iter()
            .filter(|tool| !hi_tools::is_coordination(&tool.name))
            .cloned()
            .collect::<Vec<_>>()
            .into()
    } else {
        // Cannot suppress — would empty the list.
        tools
    }
}

/// Track advertised names and peak schema tokens for telemetry.
pub(super) fn note_advertised_tools(
    tools: &[ToolSpec],
    advertised: &mut BTreeSet<String>,
    tool_schema_tokens: &mut u64,
) -> u64 {
    advertised.extend(tools.iter().map(|tool| tool.name.clone()));
    let tokens = estimate_tool_schema_tokens(tools);
    *tool_schema_tokens = (*tool_schema_tokens).max(tokens);
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suppress_noop_when_flag_clear() {
        let tools: Arc<[ToolSpec]> = Arc::from([ToolSpec {
            name: "bash".into(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object"}),
        }]);
        let out = apply_bookkeeping_suppress(tools.clone(), false);
        assert_eq!(out.len(), 1);
    }
}
