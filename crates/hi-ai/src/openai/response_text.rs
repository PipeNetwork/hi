//! Final normalization of OpenAI-compatible text response channels.

use crate::{Content, ToolSpec};

pub(super) struct NormalizedResponseText {
    pub content: Vec<Content>,
    pub text_tool_calls: bool,
}

pub(super) fn normalize(
    text: String,
    fallback_tools: Option<&[ToolSpec]>,
    native_calls_empty: bool,
    dsml_enabled: bool,
    dsml_id_prefix: &str,
) -> Result<NormalizedResponseText, &'static str> {
    let fallback_content = if native_calls_empty {
        fallback_tools
            .map(|tools| {
                let prefix = format!("text_fallback_call_{}", uuid::Uuid::new_v4().simple());
                crate::text_tool_fallback::parse_admitted_calls(&text, tools, &prefix)
            })
            .transpose()?
            .flatten()
    } else {
        None
    };
    if let Some(content) = fallback_content {
        return Ok(NormalizedResponseText {
            content,
            text_tool_calls: true,
        });
    }

    let text = if fallback_tools.is_some() {
        super::stream::strip_text_tool_protocol_artifact(&text)
    } else {
        super::stream::strip_leading_open_brace_artifact(&text)
    };
    if dsml_enabled && native_calls_empty {
        let content =
            super::deepseek::parse_dsml_tool_calls(&text, dsml_id_prefix).unwrap_or_else(|| {
                let text = super::deepseek::strip_dsml_artifacts(&text);
                (!text.is_empty())
                    .then_some(Content::Text(text))
                    .into_iter()
                    .collect()
            });
        let text_tool_calls = content
            .iter()
            .any(|content| matches!(content, Content::ToolCall { .. }));
        return Ok(NormalizedResponseText {
            content,
            text_tool_calls,
        });
    }

    let text = if dsml_enabled {
        super::deepseek::strip_dsml_artifacts(&text)
    } else {
        text
    };
    Ok(NormalizedResponseText {
        content: (!text.is_empty())
            .then_some(Content::Text(text))
            .into_iter()
            .collect(),
        text_tool_calls: false,
    })
}
