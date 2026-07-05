use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::server::Tool;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: NormalizedFunction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedFunction {
    pub name: String,
    pub arguments: String,
}

pub fn parse_tool_calls(text: &str, tools: &[Tool]) -> Option<Vec<NormalizedToolCall>> {
    let allowed = tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect::<Vec<_>>();
    if let Some(calls) = parse_json_tool_calls(text, &allowed) {
        return Some(calls);
    }
    parse_bracket_tool_calls(text, &allowed)
}

fn parse_json_tool_calls(text: &str, allowed: &[&str]) -> Option<Vec<NormalizedToolCall>> {
    let trimmed = strip_code_fence(text.trim());
    let value: Value = serde_json::from_str(trimmed).ok().or_else(|| {
        find_json_value(trimmed).and_then(|candidate| serde_json::from_str(candidate).ok())
    })?;
    let calls = if let Some(array) = value.get("tool_calls").and_then(Value::as_array) {
        array.iter().collect::<Vec<_>>()
    } else if let Some(call) = value.get("tool_call") {
        vec![call]
    } else {
        vec![&value]
    };
    let mut out = Vec::new();
    for call in calls {
        if let Some(function) = call.get("function") {
            let name = function.get("name").and_then(Value::as_str)?;
            let arguments = function.get("arguments").cloned().unwrap_or(Value::Null);
            out.push(normalize(name, arguments, allowed, out.len())?);
        } else {
            let name = call.get("name").and_then(Value::as_str)?;
            let arguments = call.get("arguments").cloned().unwrap_or(Value::Null);
            out.push(normalize(name, arguments, allowed, out.len())?);
        }
    }
    (!out.is_empty()).then_some(out)
}

fn find_json_value(text: &str) -> Option<&str> {
    for (start, ch) in text.char_indices() {
        let close = match ch {
            '{' => '}',
            '[' => ']',
            _ => continue,
        };
        if let Some(end) = balanced_json_end(&text[start..], ch, close) {
            return Some(&text[start..start + end]);
        }
    }
    None
}

fn balanced_json_end(text: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            c if c == open => depth += 1,
            c if c == close => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(idx + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_bracket_tool_calls(text: &str, allowed: &[&str]) -> Option<Vec<NormalizedToolCall>> {
    let trimmed = text.trim();
    let inner = trimmed.strip_prefix('[')?.strip_suffix(']')?.trim();
    let inner = inner
        .strip_prefix("tool_call:")
        .and_then(|rest| rest.split_once('\n').map(|(_, call)| call.trim()))
        .unwrap_or(inner);
    let open = inner.find('(')?;
    let close = inner.rfind(')')?;
    if close <= open {
        return None;
    }
    let name = inner[..open].trim();
    let args = inner[open + 1..close].trim();
    let arguments = if args.is_empty() {
        Value::Object(Default::default())
    } else {
        serde_json::from_str(args).ok()?
    };
    normalize(name, arguments, allowed, 0).map(|call| vec![call])
}

fn normalize(
    name: &str,
    arguments: Value,
    allowed: &[&str],
    index: usize,
) -> Option<NormalizedToolCall> {
    if !allowed.is_empty() && !allowed.contains(&name) {
        return None;
    }
    let arguments = match arguments {
        Value::String(s) => {
            if serde_json::from_str::<Value>(&s).is_ok() {
                s
            } else {
                serde_json::to_string(&Value::String(s)).unwrap()
            }
        }
        Value::Null => "{}".to_string(),
        other => serde_json::to_string(&other).ok()?,
    };
    Some(NormalizedToolCall {
        id: format!("call_mlx_{index}"),
        kind: "function".to_string(),
        function: NormalizedFunction {
            name: name.to_string(),
            arguments,
        },
    })
}

fn strip_code_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.trim_start_matches(|c: char| !c.is_whitespace());
    let rest = rest.trim_start();
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::server::{FunctionDef, Tool};

    fn tools() -> Vec<Tool> {
        vec![Tool {
            kind: "function".to_string(),
            function: FunctionDef {
                name: "read".to_string(),
                description: None,
                parameters: json!({"type":"object"}),
            },
        }]
    }

    #[test]
    fn json_tool_call_output_is_normalized() {
        let calls = parse_tool_calls(
            r#"{"name":"read","arguments":{"path":"README.md"}}"#,
            &tools(),
        )
        .unwrap();

        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"README.md"}"#);
    }

    #[test]
    fn bracket_tool_call_output_is_normalized() {
        let calls = parse_tool_calls(r#"[read({"path":"README.md"})]"#, &tools()).unwrap();

        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"README.md"}"#);
    }

    #[test]
    fn json_tool_call_with_reasoning_prefix_is_normalized() {
        let calls = parse_tool_calls(
            "<think>choosing a tool</think>\n\n{\"name\":\"read\",\"arguments\":{\"path\":\"README.md\"}}",
            &tools(),
        )
        .unwrap();

        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"README.md"}"#);
    }

    #[test]
    fn invalid_tool_json_falls_back_to_text() {
        assert!(parse_tool_calls(r#"{"name":"missing","arguments":{}}"#, &tools()).is_none());
        assert!(parse_tool_calls(r#"{"name":"read","arguments":"#, &tools()).is_none());
    }
}
