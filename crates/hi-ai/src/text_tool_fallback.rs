//! Bounded normalization for the explicitly authorized plain-text tool channel.
//!
//! This parser is deliberately not a general assistant-text heuristic. Callers
//! must opt in for one sealed request and provide that request's exact tool set.

use crate::{Content, ToolSpec};

const MAX_FALLBACK_TEXT_BYTES: usize = 8 * 1024 * 1024;
const MAX_FALLBACK_TOOL_NAME_BYTES: usize = 256;
const MAX_FALLBACK_ARGUMENT_BYTES: usize = 4 * 1024 * 1024;

/// Promote admitted JSON or XML-ish calls into provider-neutral content.
///
/// `None` means no complete admitted call was found. Raw protocol text is never
/// returned on that path; the OpenAI collector owns its display/history
/// sanitization separately.
pub(crate) fn parse_admitted_calls(
    text: &str,
    admitted_tools: &[ToolSpec],
    id_prefix: &str,
) -> Result<Option<Vec<Content>>, &'static str> {
    if text.len() > MAX_FALLBACK_TEXT_BYTES {
        return Err("plain-text tool fallback exceeded its response-size limit");
    }
    if admitted_tools.is_empty() {
        return Ok(None);
    }

    let mut out = Vec::new();
    let mut call_count = 0usize;
    let mut cursor = 0usize;
    let mut search = 0usize;

    while search < text.len() {
        let byte = text.as_bytes()[search];
        let parsed = if byte == b'{' {
            parse_json_call(text, search)
        } else if byte == b'<' {
            parse_xml_call(text, search)
        } else {
            None
        };

        if let Some((name, arguments, end)) = parsed {
            if !admitted_tools.iter().any(|tool| tool.name == name) {
                return Err("plain-text tool fallback named a tool outside the sealed request");
            }
            // The recovery prompt grants one textual call, independent of the
            // normal native parallel-call limit. Reject the whole response
            // instead of executing an attacker/model-selected prefix.
            if call_count != 0 {
                return Err("plain-text tool fallback emitted more than one tool call");
            }
            let prose = text[cursor..search].trim_end();
            if !prose.is_empty() {
                out.push(Content::Text(prose.to_string()));
            }
            out.push(Content::ToolCall {
                id: format!("{id_prefix}_{call_count}"),
                name,
                arguments,
            });
            call_count += 1;
            cursor = end;
            search = end;
        } else {
            search += text[search..]
                .chars()
                .next()
                .expect("search remains inside text")
                .len_utf8();
        }
    }

    if call_count == 0 {
        return Ok(None);
    }
    let trailing = text[cursor..].trim_end();
    if !trailing.is_empty() {
        out.push(Content::Text(trailing.to_string()));
    }
    Ok(Some(out))
}

fn parse_json_call(text: &str, start: usize) -> Option<(String, String, usize)> {
    let end = balanced_json_object_end(text, start)?;
    let value: serde_json::Value = serde_json::from_str(&text[start..end]).ok()?;
    let object = value.as_object()?;
    let name = object.get("name")?.as_str()?;
    if name.is_empty() || name.len() > MAX_FALLBACK_TOOL_NAME_BYTES {
        return None;
    }
    let arguments = match object.get("arguments") {
        Some(serde_json::Value::String(arguments)) => arguments.clone(),
        Some(arguments) => arguments.to_string(),
        None => "{}".to_string(),
    };
    if arguments.len() > MAX_FALLBACK_ARGUMENT_BYTES {
        return None;
    }
    Some((name.to_string(), arguments, end))
}

fn balanced_json_object_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(start) != Some(&b'{') {
        return None;
    }
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (offset, byte) in bytes[start..].iter().copied().enumerate() {
        if in_string {
            if escape {
                escape = false;
            } else if byte == b'\\' {
                escape = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
                if depth == 0 {
                    return (byte == b'}').then_some(start + offset + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_xml_call(text: &str, start: usize) -> Option<(String, String, usize)> {
    const START: &str = "<tool_call>";
    const END: &str = "</tool_call>";
    const KEY_START: &str = "<arg_key>";
    const KEY_END: &str = "</arg_key>";
    const VALUE_START: &str = "<arg_value>";
    const VALUE_END: &str = "</arg_value>";

    let rest = text.get(start..)?;
    if !rest.starts_with(START) {
        return None;
    }
    let mut position = start + START.len();
    skip_ascii_whitespace(text, &mut position);
    let name_start = position;
    while let Some(character) = text[position..].chars().next() {
        if character == '<' || character.is_whitespace() {
            break;
        }
        position += character.len_utf8();
    }
    let name = text[name_start..position].trim();
    if name.is_empty() || name.len() > MAX_FALLBACK_TOOL_NAME_BYTES {
        return None;
    }

    let mut arguments = serde_json::Map::new();
    let mut saw_argument = false;
    loop {
        skip_ascii_whitespace(text, &mut position);
        if text[position..].starts_with(END) {
            position += END.len();
            break;
        }
        if !text[position..].starts_with(KEY_START) {
            break;
        }
        let key_start = position + KEY_START.len();
        let key_end = key_start + text[key_start..].find(KEY_END)?;
        let key = text[key_start..key_end].trim();
        if key.is_empty() || arguments.contains_key(key) {
            return None;
        }
        position = key_end + KEY_END.len();
        skip_ascii_whitespace(text, &mut position);
        if !text[position..].starts_with(VALUE_START) {
            return None;
        }
        let value_start = position + VALUE_START.len();
        let value_end = value_start + text[value_start..].find(VALUE_END)?;
        if value_end.saturating_sub(value_start) > MAX_FALLBACK_ARGUMENT_BYTES {
            return None;
        }
        arguments.insert(
            key.to_string(),
            serde_json::Value::String(text[value_start..value_end].to_string()),
        );
        saw_argument = true;
        position = value_end + VALUE_END.len();
    }
    if !saw_argument {
        return None;
    }
    let arguments = serde_json::Value::Object(arguments).to_string();
    if arguments.len() > MAX_FALLBACK_ARGUMENT_BYTES {
        return None;
    }
    Some((name.to_string(), arguments, position))
}

fn skip_ascii_whitespace(text: &str, position: &mut usize) {
    while text
        .as_bytes()
        .get(*position)
        .is_some_and(u8::is_ascii_whitespace)
    {
        *position += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object"}),
        }
    }

    #[test]
    fn parses_only_tools_admitted_by_the_exact_request() {
        let parsed = parse_admitted_calls(
            "before <tool_call>read<arg_key>path</arg_key><arg_value>README.md</arg_value></tool_call> after",
            &[tool("read")],
            "fallback",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            parsed.as_slice(),
            [Content::Text(before), Content::ToolCall { id, name, arguments }, Content::Text(after)]
                if before == "before"
                    && id == "fallback_0"
                    && name == "read"
                    && arguments == r#"{"path":"README.md"}"#
                    && after == " after"
        ));
        assert_eq!(
            parse_admitted_calls(
                r#"{"name":"bash","arguments":{"command":"true"}}"#,
                &[tool("read")],
                "fallback",
            )
            .unwrap_err(),
            "plain-text tool fallback named a tool outside the sealed request"
        );
    }

    #[test]
    fn parses_admitted_json_arguments_without_double_encoding() {
        let parsed = parse_admitted_calls(
            r#"{"name":"read","arguments":{"path":"README.md"}}"#,
            &[tool("read")],
            "fallback",
        )
        .unwrap()
        .unwrap();
        assert!(matches!(
            parsed.as_slice(),
            [Content::ToolCall { name, arguments, .. }]
                if name == "read" && arguments == r#"{"path":"README.md"}"#
        ));
    }

    #[test]
    fn rejects_multiple_calls_even_when_each_tool_is_admitted() {
        let parsed = parse_admitted_calls(
            concat!(
                "<tool_call>read<arg_key>path</arg_key><arg_value>a</arg_value></tool_call>",
                "<tool_call>read<arg_key>path</arg_key><arg_value>b</arg_value></tool_call>"
            ),
            &[tool("read")],
            "fallback",
        );
        assert_eq!(
            parsed.unwrap_err(),
            "plain-text tool fallback emitted more than one tool call"
        );
    }
}
