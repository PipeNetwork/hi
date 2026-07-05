use serde_json::{Value, json};

use crate::manifest::ModelFamily;
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
        ModelFamily::DeepSeek => build_deepseek_prompt(messages, tools, tool_choice),
        ModelFamily::GlmFlash => build_glm_prompt(messages, tools, tool_choice),
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

    use super::*;
    use crate::server::{FunctionDef, Tool};

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
