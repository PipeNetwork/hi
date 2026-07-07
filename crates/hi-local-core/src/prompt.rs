use serde_json::{Value, json};

use crate::model::ModelFamily;
use crate::server::{ChatMessage, Tool};

pub fn build_prompt(
    family: ModelFamily,
    messages: &[ChatMessage],
    tools: &[Tool],
    tool_choice: &Value,
) -> String {
    match family {
        ModelFamily::Qwen2 | ModelFamily::Qwen3 => {
            build_chatml_prompt(messages, tools, tool_choice)
        }
        ModelFamily::Llama | ModelFamily::Mistral | ModelFamily::Mixtral => {
            build_llama_prompt(messages, tools, tool_choice)
        }
        ModelFamily::Gemma => build_gemma_prompt(messages, tools, tool_choice),
        ModelFamily::Phi => build_phi_prompt(messages, tools, tool_choice),
        ModelFamily::DeepSeek => build_deepseek_prompt(messages, tools, tool_choice),
        ModelFamily::GlmFlash => build_glm_prompt(messages, tools, tool_choice),
    }
}

pub fn build_prompt_with_template(
    family: ModelFamily,
    chat_template: Option<&str>,
    messages: &[ChatMessage],
    tools: &[Tool],
    tool_choice: &Value,
) -> String {
    if tools.is_empty() {
        if let Some(rendered) =
            chat_template.and_then(|template| render_gguf_chat_template(template, messages))
        {
            return rendered;
        }
    }
    build_prompt(family, messages, tools, tool_choice)
}

fn render_gguf_chat_template(template: &str, messages: &[ChatMessage]) -> Option<String> {
    let template = normalize_jinja_template(template);
    if template.contains("<|start_header_id|>")
        && template.contains("<|end_header_id|>")
        && template.contains("<|eot_id|>")
    {
        return Some(render_llama3_template(&template, messages));
    }
    if template.contains("<|im_start|>") && template.contains("<|im_end|>") {
        return render_simple_loop_template(&template, messages)
            .or_else(|| Some(build_chatml_prompt(messages, &[], &Value::Null)));
    }
    render_simple_loop_template(&template, messages)
}

fn normalize_jinja_template(template: &str) -> String {
    template
        .replace("\r\n", "\n")
        .replace("{%-", "{%")
        .replace("-%}", "%}")
        .replace("{{-", "{{")
        .replace("-}}", "}}")
}

fn render_llama3_template(template: &str, messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    if template.contains("<|begin_of_text|>") || template.contains("bos_token") {
        out.push_str("<|begin_of_text|>");
    }
    for message in messages {
        let role = normalize_chat_template_role(&message.role);
        out.push_str("<|start_header_id|>");
        out.push_str(role);
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(message.content_text().trim());
        if role == "assistant" && !message.tool_calls.is_empty() {
            if !message.content_text().trim().is_empty() {
                out.push('\n');
            }
            out.push_str(&json!(message.tool_calls).to_string());
        }
        out.push_str("<|eot_id|>");
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

fn render_simple_loop_template(template: &str, messages: &[ChatMessage]) -> Option<String> {
    const FOR_TAG: &str = "{% for message in messages %}";
    const ENDFOR_TAG: &str = "{% endfor %}";

    let loop_start = template.find(FOR_TAG)?;
    let body_start = loop_start + FOR_TAG.len();
    let loop_remainder = &template[body_start..];
    let body_end = loop_remainder.find(ENDFOR_TAG)?;
    let body = &loop_remainder[..body_end];
    let suffix = &loop_remainder[body_end + ENDFOR_TAG.len()..];
    let prefix = &template[..loop_start];

    if prefix.contains("{%") || body.contains("{%") {
        return None;
    }

    let mut out = render_jinja_prints(prefix, None)?;
    for message in messages {
        out.push_str(&render_jinja_prints(body, Some(message))?);
    }
    out.push_str(&render_generation_suffix(suffix)?);
    Some(out)
}

fn render_generation_suffix(suffix: &str) -> Option<String> {
    const IF_GENERATION_TAG: &str = "{% if add_generation_prompt %}";
    const ENDIF_TAG: &str = "{% endif %}";

    if let Some(if_start) = suffix.find(IF_GENERATION_TAG) {
        let before = &suffix[..if_start];
        let after_if = &suffix[if_start + IF_GENERATION_TAG.len()..];
        let endif_start = after_if.find(ENDIF_TAG)?;
        let generation_prompt = &after_if[..endif_start];
        let after = &after_if[endif_start + ENDIF_TAG.len()..];
        if before.contains("{%") || generation_prompt.contains("{%") || after.contains("{%") {
            return None;
        }
        let mut out = render_jinja_prints(before, None)?;
        out.push_str(&render_jinja_prints(generation_prompt, None)?);
        out.push_str(&render_jinja_prints(after, None)?);
        return Some(out);
    }

    if suffix.contains("{%") {
        return None;
    }
    render_jinja_prints(suffix, None)
}

fn render_jinja_prints(input: &str, message: Option<&ChatMessage>) -> Option<String> {
    let mut out = String::new();
    let mut remaining = input;
    while let Some(print_start) = remaining.find("{{") {
        out.push_str(&remaining[..print_start]);
        let after_start = &remaining[print_start + 2..];
        let print_end = after_start.find("}}")?;
        let expression = &after_start[..print_end];
        out.push_str(&eval_jinja_print(expression, message)?);
        remaining = &after_start[print_end + 2..];
    }
    out.push_str(remaining);
    Some(out)
}

fn eval_jinja_print(expression: &str, message: Option<&ChatMessage>) -> Option<String> {
    let expression = expression
        .trim()
        .trim_start_matches('-')
        .trim()
        .trim_end_matches('-')
        .trim();
    let (base, trim_output) = split_supported_filters(expression)?;
    let compact = base
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let mut value = match compact.as_str() {
        "bos_token" | "eos_token" => String::new(),
        "message['role']" | "message[\"role\"]" | "message.role" => {
            normalize_chat_template_role(&message?.role).to_string()
        }
        "message['content']" | "message[\"content\"]" | "message.content" => {
            message?.content_text()
        }
        _ => render_jinja_literal(base)?,
    };
    if trim_output {
        value = value.trim().to_string();
    }
    Some(value)
}

fn split_supported_filters(expression: &str) -> Option<(&str, bool)> {
    let mut parts = expression.split('|').map(str::trim);
    let base = parts.next()?.trim();
    let mut trim_output = false;
    for filter in parts {
        match filter {
            "" => {}
            "trim" => trim_output = true,
            _ => return None,
        }
    }
    Some((base, trim_output))
}

fn render_jinja_literal(expression: &str) -> Option<String> {
    let expression = expression.trim();
    let quote = expression.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    if !expression.ends_with(quote) || expression.len() < 2 {
        return None;
    }
    Some(unescape_jinja_string(
        &expression[quote.len_utf8()..expression.len() - quote.len_utf8()],
        quote,
    ))
}

fn unescape_jinja_string(value: &str, quote: char) -> String {
    let mut out = String::new();
    let mut escaped = false;
    for ch in value.chars() {
        if escaped {
            match ch {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '\\' => out.push('\\'),
                '\'' if quote == '\'' => out.push('\''),
                '"' if quote == '"' => out.push('"'),
                other => out.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }
    if escaped {
        out.push('\\');
    }
    out
}

fn normalize_chat_template_role(role: &str) -> &str {
    match role {
        "system" => "system",
        "assistant" => "assistant",
        "tool" => "tool",
        _ => "user",
    }
}

pub fn build_chatml_prompt(
    messages: &[ChatMessage],
    tools: &[Tool],
    tool_choice: &Value,
) -> String {
    let mut out = String::new();
    let tool_block = tool_instructions(tools, tool_choice);
    if !tool_block.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(&tool_block);
        out.push_str("\n<|im_end|>\n");
    }
    for message in messages {
        let role = match message.role.as_str() {
            "system" => "system",
            "assistant" => "assistant",
            "tool" => "tool",
            _ => "user",
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&message.content_text());
        if role == "assistant" && !message.tool_calls.is_empty() {
            if !message.content_text().is_empty() {
                out.push('\n');
            }
            out.push_str(&json!(message.tool_calls).to_string());
        }
        out.push_str("\n<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn build_llama_prompt(messages: &[ChatMessage], tools: &[Tool], tool_choice: &Value) -> String {
    let mut out = String::new();
    let tool_block = tool_instructions(tools, tool_choice);
    if !tool_block.is_empty() {
        out.push_str("<|system|>\n");
        out.push_str(&tool_block);
        out.push_str("</s>");
    }
    for message in messages {
        let role = match message.role.as_str() {
            "system" => "system",
            "assistant" => "assistant",
            "tool" => "tool",
            _ => "user",
        };
        out.push_str("<|");
        out.push_str(role);
        out.push_str("|>\n");
        out.push_str(&message.content_text());
        if role == "assistant" && !message.tool_calls.is_empty() {
            if !message.content_text().is_empty() {
                out.push('\n');
            }
            out.push_str(&json!(message.tool_calls).to_string());
        }
        out.push_str("</s>");
    }
    out.push_str("<|assistant|>\n");
    out
}

fn build_gemma_prompt(messages: &[ChatMessage], tools: &[Tool], tool_choice: &Value) -> String {
    let mut out = String::new();
    let tool_block = tool_instructions(tools, tool_choice);
    if !tool_block.is_empty() {
        out.push_str("<start_of_turn>user\n");
        out.push_str(&tool_block);
        out.push_str("<end_of_turn>\n");
    }
    for message in messages {
        let role = match message.role.as_str() {
            "assistant" | "model" => "model",
            _ => "user",
        };
        out.push_str("<start_of_turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&message.content_text());
        if role == "model" && !message.tool_calls.is_empty() {
            if !message.content_text().is_empty() {
                out.push('\n');
            }
            out.push_str(&json!(message.tool_calls).to_string());
        }
        out.push_str("<end_of_turn>\n");
    }
    out.push_str("<start_of_turn>model\n");
    out
}

fn build_phi_prompt(messages: &[ChatMessage], tools: &[Tool], tool_choice: &Value) -> String {
    let mut out = String::new();
    let tool_block = tool_instructions(tools, tool_choice);
    if !tool_block.is_empty() {
        out.push_str("<|system|>\n");
        out.push_str(&tool_block);
        out.push_str("<|end|>\n");
    }
    for message in messages {
        let role = match message.role.as_str() {
            "system" => "system",
            "assistant" => "assistant",
            "tool" => "user",
            _ => "user",
        };
        out.push_str("<|");
        out.push_str(role);
        out.push_str("|>\n");
        if message.role == "tool" {
            out.push_str("Tool result");
            if let Some(id) = &message.tool_call_id {
                out.push_str(" for ");
                out.push_str(id);
            }
            out.push_str(":\n");
        }
        out.push_str(&message.content_text());
        if role == "assistant" && !message.tool_calls.is_empty() {
            if !message.content_text().is_empty() {
                out.push('\n');
            }
            out.push_str(&json!(message.tool_calls).to_string());
        }
        out.push_str("<|end|>\n");
    }
    out.push_str("<|assistant|>\n");
    out
}

fn build_deepseek_prompt(messages: &[ChatMessage], tools: &[Tool], tool_choice: &Value) -> String {
    let mut out = String::new();
    let tool_block = tool_instructions(tools, tool_choice);
    if !tool_block.is_empty() {
        out.push_str(&tool_block);
        out.push('\n');
    }
    for message in messages {
        match message.role.as_str() {
            "system" => {
                out.push_str(message.content_text().trim());
                out.push('\n');
            }
            "assistant" => {
                out.push_str("<｜Assistant｜>");
                out.push_str(&message.content_text());
                if !message.tool_calls.is_empty() {
                    if !message.content_text().is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&json!(message.tool_calls).to_string());
                }
                out.push('\n');
            }
            "tool" => {
                out.push_str("<｜User｜>Tool result");
                if let Some(id) = &message.tool_call_id {
                    out.push_str(" for ");
                    out.push_str(id);
                }
                out.push_str(":\n");
                out.push_str(&message.content_text());
                out.push('\n');
            }
            _ => {
                out.push_str("<｜User｜>");
                out.push_str(&message.content_text());
                out.push('\n');
            }
        }
    }
    out.push_str("<｜Assistant｜>");
    out
}

fn build_glm_prompt(messages: &[ChatMessage], tools: &[Tool], tool_choice: &Value) -> String {
    let mut out = String::new();
    let tool_block = tool_instructions(tools, tool_choice);
    if !tool_block.is_empty() {
        out.push_str("<|system|>\n");
        out.push_str(&tool_block);
        out.push('\n');
    }
    for message in messages {
        let role = match message.role.as_str() {
            "system" => "system",
            "assistant" => "assistant",
            "tool" => "tool",
            _ => "user",
        };
        out.push_str("<|");
        out.push_str(role);
        out.push_str("|>\n");
        out.push_str(&message.content_text());
        if role == "assistant" && !message.tool_calls.is_empty() {
            if !message.content_text().is_empty() {
                out.push('\n');
            }
            out.push_str(&json!(message.tool_calls).to_string());
        }
        out.push('\n');
    }
    out.push_str("<|assistant|>\n");
    out
}

fn tool_instructions(tools: &[Tool], tool_choice: &Value) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str("You have access to tools. When a tool is needed, respond with a JSON object ");
    out.push_str(r#"like {"name":"tool_name","arguments":{...}} and no extra prose."#);
    if tool_choice == "required" {
        out.push_str(" You must call a tool.");
    }
    out.push_str("\n\n<tools>\n");
    out.push_str(&serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".to_string()));
    out.push_str("\n</tools>");
    out
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::server::{FunctionDef, Tool};

    use super::*;

    #[test]
    fn request_with_tools_injects_tool_schema() {
        let tools = vec![Tool {
            kind: "function".to_string(),
            function: FunctionDef {
                name: "read".to_string(),
                description: Some("Read a file".to_string()),
                parameters: json!({"type":"object"}),
            },
        }];
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!("inspect README")),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];

        let prompt = build_prompt(ModelFamily::Qwen3, &messages, &tools, &json!("required"));

        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("\"name\": \"read\""));
        assert!(prompt.contains("You must call a tool"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn deepseek_prompt_uses_deepseek_roles() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!("hi")),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];

        let prompt = build_prompt(ModelFamily::DeepSeek, &messages, &[], &Value::Null);

        assert!(prompt.contains("<｜User｜>hi"));
        assert!(prompt.ends_with("<｜Assistant｜>"));
    }

    #[test]
    fn llama_prompt_uses_zephyr_roles() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!("hi")),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];

        let prompt = build_prompt(ModelFamily::Llama, &messages, &[], &Value::Null);

        assert!(prompt.contains("<|user|>\nhi</s>"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn mistral_prompt_uses_llama_compatible_roles() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!("hi")),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];

        let prompt = build_prompt(ModelFamily::Mistral, &messages, &[], &Value::Null);

        assert!(prompt.contains("<|user|>\nhi</s>"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn mixtral_prompt_uses_mistral_compatible_roles() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!("hi")),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];

        let prompt = build_prompt(ModelFamily::Mixtral, &messages, &[], &Value::Null);

        assert!(prompt.contains("<|user|>\nhi</s>"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn gemma_prompt_uses_turn_markers() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!("hi")),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];

        let prompt = build_prompt(ModelFamily::Gemma, &messages, &[], &Value::Null);

        assert!(prompt.contains("<start_of_turn>user\nhi<end_of_turn>\n"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn phi_prompt_uses_phi_chat_markers() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!("hi")),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];

        let prompt = build_prompt(ModelFamily::Phi, &messages, &[], &Value::Null);

        assert!(prompt.contains("<|user|>\nhi<|end|>\n"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn llama3_gguf_chat_template_uses_header_tokens() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!("hi")),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];
        let template = "{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";

        let prompt = build_prompt_with_template(
            ModelFamily::Llama,
            Some(template),
            &messages,
            &[],
            &Value::Null,
        );

        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn simple_gguf_chat_template_renders_role_and_content() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some(json!(" setup ")),
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(json!(" hi ")),
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
        ];
        let template = "{{ bos_token }}{% for message in messages %}[{{ message['role'] }}] {{ message['content'] | trim }}\n{% endfor %}{% if add_generation_prompt %}[assistant] {% endif %}";

        let prompt = build_prompt_with_template(
            ModelFamily::Llama,
            Some(template),
            &messages,
            &[],
            &Value::Null,
        );

        assert_eq!(prompt, "[system] setup\n[user] hi\n[assistant] ");
    }

    #[test]
    fn template_with_tools_falls_back_to_family_prompt() {
        let tools = vec![Tool {
            kind: "function".to_string(),
            function: FunctionDef {
                name: "read".to_string(),
                description: Some("Read a file".to_string()),
                parameters: json!({"type":"object"}),
            },
        }];
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!("inspect README")),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];
        let template = "{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";

        let prompt = build_prompt_with_template(
            ModelFamily::Llama,
            Some(template),
            &messages,
            &tools,
            &json!("required"),
        );

        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("<|user|>\ninspect README</s>"));
        assert!(!prompt.contains("<|start_header_id|>"));
    }

    #[test]
    fn glm_flash_prompt_uses_glm_roles() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!("hi")),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];

        let prompt = build_prompt(ModelFamily::GlmFlash, &messages, &[], &Value::Null);

        assert!(prompt.contains("<|user|>\nhi"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }
}
