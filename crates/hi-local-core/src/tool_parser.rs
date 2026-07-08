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
    if let Some(calls) = parse_bracket_tool_calls(text, &allowed) {
        return Some(calls);
    }
    if let Some(calls) = parse_harmony_tool_calls(text, &allowed) {
        return Some(calls);
    }
    if let Some(calls) = parse_xml_attr_tool_calls(text, &allowed) {
        return Some(calls);
    }
    parse_element_tool_calls(text, &allowed)
}

// GPT-OSS harmony tool call: `...commentary to=functions.get_weather ...<|message|>{json}<|call|>`.
// The harmony delimiter tokens are dropped on decode, so match `functions.NAME` and take the next JSON
// object. Only the first call is used (a forced call may be followed by a hallucinated response echo).
fn parse_harmony_tool_calls(text: &str, allowed: &[&str]) -> Option<Vec<NormalizedToolCall>> {
    let pos = text.find("functions.")?;
    let after = &text[pos + "functions.".len()..];
    let name_end = after
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))
        .unwrap_or(after.len());
    let name = &after[..name_end];
    let arguments = find_json_value(&after[name_end..])
        .and_then(|json| serde_json::from_str::<Value>(json).ok())
        .unwrap_or(Value::Object(Default::default()));
    normalize(name, arguments, allowed, 0).map(|call| vec![call])
}

// OLMo-2-style `<tool_call><get_weather city="Paris"></tool_call>` — a nested element named after the
// function, with the arguments as XML attributes.
fn parse_element_tool_calls(text: &str, allowed: &[&str]) -> Option<Vec<NormalizedToolCall>> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find("<tool_call>") {
        let after = &rest[pos + "<tool_call>".len()..];
        let block_end = after.find("</tool_call>").unwrap_or(after.len());
        let block = &after[..block_end];
        rest = &after[block_end..];
        let Some(lt) = block.find('<') else {
            continue;
        };
        let inner = &block[lt + 1..];
        let gt = inner.find('>').unwrap_or(inner.len());
        let elem = inner[..gt].trim();
        let (name, attrs_str) = match elem.split_once(char::is_whitespace) {
            Some((n, a)) => (n.trim_end_matches('/'), a),
            None => (elem.trim_end_matches('/'), ""),
        };
        let arguments = Value::Object(parse_xml_attrs(attrs_str));
        if let Some(call) = normalize(name, arguments, allowed, out.len()) {
            out.push(call);
        }
    }
    (!out.is_empty()).then_some(out)
}

// Parse `key="value" key2='value2'` attribute pairs into a JSON object. Numeric/boolean values are
// parsed as such; everything else stays a string.
fn parse_xml_attrs(s: &str) -> serde_json::Map<String, Value> {
    let mut map = serde_json::Map::new();
    let mut rest = s.trim();
    while let Some(eq) = rest.find('=') {
        let key = rest[..eq]
            .rsplit(char::is_whitespace)
            .next()
            .unwrap_or("")
            .trim();
        let after = rest[eq + 1..].trim_start();
        let Some(quote) = after.chars().next().filter(|c| *c == '"' || *c == '\'') else {
            break;
        };
        let value_start = &after[quote.len_utf8()..];
        let Some(end) = value_start.find(quote) else {
            break;
        };
        let raw = &value_start[..end];
        if !key.is_empty() {
            let value = serde_json::from_str::<Value>(raw)
                .ok()
                .filter(|v| v.is_number() || v.is_boolean())
                .unwrap_or_else(|| Value::String(raw.to_string()));
            map.insert(key.to_string(), value);
        }
        rest = &value_start[end + quote.len_utf8()..];
    }
    map
}

// GPT-OSS-style `<tool_call name="get_weather" arguments='{"city":"Paris"}'>` (attributes rather than a
// JSON body). Extract each tag's name/arguments attributes.
fn parse_xml_attr_tool_calls(text: &str, allowed: &[&str]) -> Option<Vec<NormalizedToolCall>> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find("<tool_call") {
        let after = &rest[pos + "<tool_call".len()..];
        let end = after.find('>').unwrap_or(after.len());
        let tag = &after[..end];
        rest = &after[end..];
        let Some(name) = xml_attr(tag, "name") else {
            continue;
        };
        let args = xml_attr(tag, "arguments").unwrap_or("{}");
        let arguments =
            serde_json::from_str::<Value>(args).unwrap_or(Value::Object(Default::default()));
        if let Some(call) = normalize(name, arguments, allowed, out.len()) {
            out.push(call);
        }
    }
    (!out.is_empty()).then_some(out)
}

fn xml_attr<'a>(tag: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("{key}=");
    let start = tag.find(&needle)? + needle.len();
    let rest = &tag[start..];
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let inner = &rest[quote.len_utf8()..];
    let end = inner.find(quote)?;
    Some(&inner[..end])
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
    // Llama-style calls use `parameters` where OpenAI/hermes use `arguments`.
    let args_of = |obj: &Value| {
        obj.get("arguments")
            .or_else(|| obj.get("parameters"))
            .cloned()
            .unwrap_or(Value::Null)
    };
    let mut out = Vec::new();
    for call in calls {
        if let Some(function) = call.get("function") {
            let name = function.get("name").and_then(Value::as_str)?;
            out.push(normalize(name, args_of(function), allowed, out.len())?);
        } else {
            // Llama-4 Scout puts the function name under `type` rather than `name`.
            let name = call
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| call.get("type").and_then(Value::as_str))?;
            out.push(normalize(name, args_of(call), allowed, out.len())?);
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

    use crate::server::{FunctionDef, Tool};

    use super::*;

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
    fn olmo2_element_tool_call_is_normalized() {
        let calls = parse_tool_calls(
            "<tool_call>\n<read path=\"README.md\"></tool_call>\n\n<tool_response>",
            &tools(),
        )
        .unwrap();

        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"README.md"}"#);
    }

    #[test]
    fn harmony_tool_call_is_normalized() {
        let calls = parse_tool_calls(
            "commentary to=functions.read <|constrain|>json<|message|>{\"path\":\"README.md\"}<|call|>",
            &tools(),
        )
        .unwrap();

        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"README.md"}"#);
    }

    #[test]
    fn llama4_type_parameters_tool_call_is_normalized() {
        let calls = parse_tool_calls(
            r#"{"type":"read","parameters":{"path":"README.md"}}"#,
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
