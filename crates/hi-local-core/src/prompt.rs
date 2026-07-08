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
        ModelFamily::Qwen2 | ModelFamily::Qwen3 | ModelFamily::NemotronH => {
            build_chatml_prompt(messages, tools, tool_choice)
        }
        ModelFamily::MiniMax => render_minimax_template(messages),
        ModelFamily::LongCat => render_longcat_template(messages),
        ModelFamily::Hy3 => build_hy3_prompt(messages, tools, tool_choice),
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
    // Custom Gemma-4 channel/turn format (e.g. pipenetwork fine-tunes): `<|turn>{role}\n{content}<turn|>`
    // with a `<|turn>model\n` generation prompt. Its full jinja is too complex for the loop renderer,
    // but the message framing is simple, so render it directly.
    if template.contains("<|turn>") && template.contains("<turn|>") {
        return Some(render_gemma_turn_template(messages));
    }
    // MiniMax-M3 custom format: `]~b]{role}\n{content}[e~[` framing, `]~b]ai\n</mm:think>` generation.
    if template.contains("]~b]") && template.contains("[e~[") {
        return Some(render_minimax_template(messages));
    }
    // LongCat-2.0 custom format: `<longcat_system|user|assistant>` turns.
    if template.contains("<longcat_assistant>") && template.contains("<longcat_user>") {
        return Some(render_longcat_template(messages));
    }
    // Granite 3.x: `<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>` turns.
    if template.contains("<|start_of_role|>") && template.contains("<|end_of_role|>") {
        return Some(render_granite_template(messages));
    }
    if template.contains("<|start_header_id|>")
        && template.contains("<|end_header_id|>")
        && template.contains("<|eot_id|>")
    {
        return Some(render_llama3_template(&template, messages));
    }
    // SmolLM3: chatml + a `reasoning_mode` toggle; render with thinking off (empty think block).
    if template.contains("reasoning_mode") && template.contains("<|im_start|>") {
        return Some(render_smollm3_template(messages));
    }
    // Seed-OSS: `<seed:bos>{role}\n...<seed:eos>` turns; set thinking_budget 0 for direct answers.
    if template.contains("<seed:bos>") && template.contains("thinking_budget") {
        return Some(render_seedoss_template(messages));
    }
    // GPT-OSS harmony format: `<|start|>{role}<|message|>...<|end|>`; prime the `final` channel.
    if template.contains("<|channel|>") && template.contains("<|start|>") {
        return Some(render_gptoss_template(messages));
    }
    // Cohere Command-R: `<|START_OF_TURN_TOKEN|><|{ROLE}_TOKEN|>...<|END_OF_TURN_TOKEN|>` turns.
    if template.contains("<|START_OF_TURN_TOKEN|>") && template.contains("<|END_OF_TURN_TOKEN|>") {
        return Some(render_cohere_template(messages));
    }
    // Llama-4: `<|header_start|>{role}<|header_end|>\n\n{content}<|eot|>` turns.
    if template.contains("<|header_start|>") && template.contains("<|header_end|>") {
        return Some(render_llama4_template(messages));
    }
    // Standard Gemma (2/3) format: `<start_of_turn>{role}\n{content}<end_of_turn>`. Its jinja is too
    // complex for the loop renderer; the framing is simple, so use the dedicated builder. Gemma is very
    // BOS-sensitive and the template emits `bos_token` first, so prepend it.
    if template.contains("<start_of_turn>") && template.contains("<end_of_turn>") {
        return Some(format!(
            "<bos>{}",
            build_gemma_prompt(messages, &[], &Value::Null)
        ));
    }
    if template.contains("<|im_start|>") && template.contains("<|im_end|>") {
        return render_simple_loop_template(&template, messages)
            .or_else(|| Some(build_chatml_prompt(messages, &[], &Value::Null)));
    }
    render_simple_loop_template(&template, messages)
}

fn render_gemma_turn_template(messages: &[ChatMessage]) -> String {
    // Leading BOS (the template emits bos_token first) is required — Gemma is sensitive to it.
    let mut out = String::from("<bos>");
    for message in messages {
        let role = match message.role.as_str() {
            "assistant" | "model" => "model",
            "system" | "developer" => "system",
            "tool" => "tool",
            _ => "user",
        };
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&message.content_text());
        out.push_str("<turn|>\n");
    }
    // Generation prompt: open the model turn and prime an empty thought channel (thinking disabled),
    // so the model proceeds straight to its final answer.
    out.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    out
}

fn render_minimax_template(messages: &[ChatMessage]) -> String {
    const BOD: &str = "]~!b[";
    const BOS: &str = "]~b]";
    const EOS: &str = "[e~[";
    let mut out = String::from(BOD);
    // Developer preamble: the system/developer message, or the default identity.
    let system = messages
        .iter()
        .find(|m| matches!(m.role.as_str(), "system" | "developer" | "root"))
        .map(|m| m.content_text());
    out.push_str(BOS);
    out.push_str("developer\n");
    out.push_str(&system.unwrap_or_else(|| "You are a helpful assistant.".to_string()));
    out.push_str(EOS);
    out.push('\n');
    for message in messages {
        let role = match message.role.as_str() {
            "assistant" | "ai" | "model" => "ai",
            "system" | "developer" | "root" => continue,
            "tool" => "tool",
            _ => "user",
        };
        out.push_str(BOS);
        out.push_str(role);
        out.push('\n');
        out.push_str(&message.content_text());
        out.push_str(EOS);
        out.push('\n');
    }
    // Generation prompt: open the ai turn and skip thinking (go straight to the answer).
    out.push_str(BOS);
    out.push_str("ai\n</mm:think>");
    out
}

// LongCat-2.0: `<longcat_system|user|assistant>` turns; the generation prompt opens the assistant turn
// with an empty think block (thinking disabled) so the model answers directly.
fn render_longcat_template(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for message in messages {
        match message.role.as_str() {
            "system" | "developer" | "root" => {
                out.push_str("<longcat_system>");
                out.push_str(&message.content_text());
            }
            "assistant" | "ai" | "model" => {
                out.push_str("<longcat_assistant>");
                out.push_str(&message.content_text());
            }
            "tool" => {
                out.push_str("<longcat_user>");
                out.push_str(&message.content_text());
            }
            _ => {
                out.push_str("<longcat_user>");
                out.push_str(&message.content_text());
            }
        }
    }
    out.push_str("<longcat_assistant><longcat_think>\n\n</longcat_think>\n");
    out
}

// Granite 3.x: `<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n` turns.
fn render_granite_template(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for message in messages {
        let role = match message.role.as_str() {
            "assistant" | "ai" | "model" => "assistant",
            "system" | "developer" | "root" => "system",
            "tool" => "tool",
            _ => "user",
        };
        out.push_str("<|start_of_role|>");
        out.push_str(role);
        out.push_str("<|end_of_role|>");
        out.push_str(&message.content_text());
        out.push_str("<|end_of_text|>\n");
    }
    out.push_str("<|start_of_role|>assistant<|end_of_role|>");
    out
}

// SmolLM3: chatml turns; the generation prompt primes an empty think block (/no_think) so the reasoning
// model answers directly instead of leaking a `<think>` monologue.
fn render_smollm3_template(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for message in messages {
        let role = match message.role.as_str() {
            "assistant" | "ai" | "model" => "assistant",
            "system" | "developer" | "root" => "system",
            "tool" => "tool",
            _ => "user",
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&message.content_text());
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n");
    out
}

// Seed-OSS: `<seed:bos>{role}\n...<seed:eos>` turns. Inject the thinking_budget=0 system instruction so
// the reasoning model answers directly instead of emitting a `<seed:think>` monologue.
fn render_seedoss_template(messages: &[ChatMessage]) -> String {
    const BUDGET0: &str = "\nYou are an intelligent assistant that can answer questions in one step \
        without the need for reasoning and thinking, that is, your thinking budget is 0. Next, please \
        skip the thinking process and directly start answering the user's question.";
    let mut out = String::new();
    let system = messages
        .iter()
        .find(|m| matches!(m.role.as_str(), "system" | "developer" | "root"))
        .map(|m| m.content_text())
        .unwrap_or_default();
    out.push_str("<seed:bos>system\n");
    out.push_str(&system);
    out.push_str(BUDGET0);
    out.push_str("<seed:eos>");
    for message in messages {
        let role = match message.role.as_str() {
            "assistant" | "ai" | "model" => "assistant",
            "system" | "developer" | "root" => continue,
            "tool" => "tool",
            _ => "user",
        };
        out.push_str("<seed:bos>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&message.content_text());
        out.push_str("<seed:eos>");
    }
    out.push_str("<seed:bos>assistant\n");
    out
}

// GPT-OSS harmony format: `<|start|>{role}<|message|>...<|end|>`. Prime the assistant `final` channel
// (low reasoning) so the model returns a direct answer rather than an analysis monologue.
fn render_gptoss_template(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    let system = messages
        .iter()
        .find(|m| matches!(m.role.as_str(), "system" | "developer" | "root"))
        .map(|m| m.content_text())
        .unwrap_or_else(|| "You are a helpful assistant.".to_string());
    out.push_str("<|start|>system<|message|>");
    out.push_str(&system);
    out.push_str("\nReasoning: low<|end|>");
    for message in messages {
        let role = match message.role.as_str() {
            "assistant" | "ai" | "model" => "assistant",
            "system" | "developer" | "root" => continue,
            "tool" => "tool",
            _ => "user",
        };
        out.push_str("<|start|>");
        out.push_str(role);
        out.push_str("<|message|>");
        out.push_str(&message.content_text());
        out.push_str("<|end|>");
    }
    out.push_str("<|start|>assistant<|channel|>final<|message|>");
    out
}

// Cohere Command-R: `<|START_OF_TURN_TOKEN|><|{ROLE}_TOKEN|>{content}<|END_OF_TURN_TOKEN|>` turns, with
// a leading BOS and a `<|CHATBOT_TOKEN|>` generation prompt.
fn render_cohere_template(messages: &[ChatMessage]) -> String {
    let mut out = String::from("<BOS_TOKEN>");
    for message in messages {
        let role = match message.role.as_str() {
            "assistant" | "ai" | "model" => "CHATBOT",
            "system" | "developer" | "root" => "SYSTEM",
            _ => "USER",
        };
        out.push_str("<|START_OF_TURN_TOKEN|><|");
        out.push_str(role);
        out.push_str("_TOKEN|>");
        out.push_str(&message.content_text());
        out.push_str("<|END_OF_TURN_TOKEN|>");
    }
    out.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
    out
}

// Llama-4: `<|header_start|>{role}<|header_end|>\n\n{content}<|eot|>` turns, leading begin-of-text and
// an assistant header generation prompt.
fn render_llama4_template(messages: &[ChatMessage]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for message in messages {
        let role = match message.role.as_str() {
            "assistant" | "ai" | "model" => "assistant",
            "system" | "developer" | "root" => "system",
            "tool" => "ipython",
            _ => "user",
        };
        out.push_str("<|header_start|>");
        out.push_str(role);
        out.push_str("<|header_end|>\n\n");
        out.push_str(&message.content_text());
        out.push_str("<|eot|>");
    }
    out.push_str("<|header_start|>assistant<|header_end|>\n\n");
    out
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

/// Render one tool call in Qwen2.5/Qwen3's native `<tool_call>` format. Accepts
/// either the OpenAI shape (`{"function":{"name","arguments"}}`, where
/// `arguments` is a JSON *string*) or a bare `{"name","arguments"}` object, and
/// emits `<tool_call>\n{"name": …, "arguments": {…}}\n</tool_call>` with the
/// arguments as a JSON object (matching what the model was trained to produce).
fn render_qwen_tool_call(tool_call: &Value) -> String {
    let function = tool_call.get("function").unwrap_or(tool_call);
    let name = function.get("name").and_then(Value::as_str).unwrap_or("");
    let arguments = match function.get("arguments").cloned().unwrap_or(Value::Null) {
        // OpenAI encodes arguments as a JSON string; Qwen wants the object.
        Value::String(text) => serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text)),
        other => other,
    };
    format!(
        "<tool_call>\n{{\"name\": {}, \"arguments\": {}}}\n</tool_call>",
        json!(name),
        arguments
    )
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
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        // Qwen has no ChatML `tool` role: tool results live inside a *user* turn,
        // each wrapped in <tool_response>…</tool_response>, with consecutive tool
        // messages grouped under one user turn. Rendering them as `<|im_start|>tool`
        // (an unknown role) is what made the model ignore the result and re-emit
        // the tool call.
        if message.role == "tool" {
            out.push_str("<|im_start|>user");
            loop {
                out.push_str("\n<tool_response>\n");
                out.push_str(&messages[index].content_text());
                out.push_str("\n</tool_response>");
                if messages.get(index + 1).is_some_and(|m| m.role == "tool") {
                    index += 1;
                } else {
                    break;
                }
            }
            out.push_str("<|im_end|>\n");
            index += 1;
            continue;
        }

        let role = match message.role.as_str() {
            "system" => "system",
            "assistant" => "assistant",
            _ => "user",
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&message.content_text());
        if role == "assistant" && !message.tool_calls.is_empty() {
            for tool_call in &message.tool_calls {
                if !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str(&render_qwen_tool_call(tool_call));
            }
        }
        out.push_str("<|im_end|>\n");
        index += 1;
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

/// Hy3 (Hunyuan-3) prompt format. Uses Hunyuan special tokens rather than ChatML:
/// `<｜hy_begin_of_sentence｜>{system}<｜hy_User｜>{user}<｜hy_Assistant｜>…`, and the model
/// ends its turn with `<｜hy_eos｜>` (token 120025, read into eos_token_ids from config.json).
fn build_hy3_prompt(messages: &[ChatMessage], tools: &[Tool], tool_choice: &Value) -> String {
    const BOS: &str = "<｜hy_begin_of_sentence:opensource｜>";
    const USER: &str = "<｜hy_User:opensource｜>";
    const ASSISTANT: &str = "<｜hy_Assistant:opensource｜>";
    const EOS: &str = "<｜hy_eos:opensource｜>";

    // System prompt is emitted once, right after BOS.
    let mut system = String::new();
    for message in messages {
        if message.role == "system" {
            system.push_str(&message.content_text());
        }
    }
    let tool_block = tool_instructions(tools, tool_choice);
    if !tool_block.is_empty() {
        if !system.is_empty() {
            system.push('\n');
        }
        system.push_str(&tool_block);
    }

    let mut out = String::new();
    out.push_str(BOS);
    out.push_str(&system);
    for message in messages {
        match message.role.as_str() {
            "system" => {} // folded into the system prompt above
            "assistant" => {
                out.push_str(ASSISTANT);
                out.push_str(&message.content_text());
                if !message.tool_calls.is_empty() {
                    out.push_str(&json!(message.tool_calls).to_string());
                }
                out.push_str(EOS);
            }
            _ => {
                // user + tool responses both open a user turn
                out.push_str(USER);
                out.push_str(&message.content_text());
            }
        }
    }
    out.push_str(ASSISTANT);
    out
}

fn tool_instructions(tools: &[Tool], tool_choice: &Value) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str("You have access to tools. To call one, emit a JSON object ");
    out.push_str(
        r#"{"name": "tool_name", "arguments": {...}} wrapped in <tool_call></tool_call> tags. "#,
    );
    out.push_str(
        "A tool's result is returned to you inside <tool_response></tool_response> tags; once you \
         have it, answer the user directly in plain text instead of calling the tool again.",
    );
    if tool_choice == "required" {
        out.push_str(" You must call a tool.");
    }
    // Emit each tool as compact JSON on its own line — matching Qwen2.5's native
    // template (`{{ tool | tojson }}` per tool) and avoiding the pretty-printer's
    // indentation/newline tokens, which balloon the prompt (re-prefilled every
    // turn) for no benefit.
    out.push_str("\n\n<tools>");
    for tool in tools {
        out.push('\n');
        out.push_str(&serde_json::to_string(tool).unwrap_or_else(|_| "{}".to_string()));
    }
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
        // Tools are emitted as compact JSON (no space after the colon).
        assert!(prompt.contains("\"name\":\"read\""));
        assert!(prompt.contains("You must call a tool"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_renders_tool_turns_in_qwen_native_format() {
        let tools = vec![Tool {
            kind: "function".to_string(),
            function: FunctionDef {
                name: "read_file".to_string(),
                description: Some("Read a file".to_string()),
                parameters: json!({"type":"object"}),
            },
        }];
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: Some(json!("read config.txt")),
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_call_id: None,
                // OpenAI shape: arguments is a JSON *string*.
                tool_calls: vec![json!({
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"config.txt\"}"}
                })],
            },
            ChatMessage {
                role: "tool".to_string(),
                content: Some(json!("port = 8443")),
                tool_call_id: Some("c1".to_string()),
                tool_calls: Vec::new(),
            },
        ];

        let prompt = build_prompt(ModelFamily::Qwen2, &messages, &tools, &Value::Null);

        // Assistant call uses <tool_call> tags with arguments as an object.
        assert!(
            prompt.contains("<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\":\"config.txt\"}}\n</tool_call>"),
            "assistant tool call not in Qwen format:\n{prompt}"
        );
        // Tool result is a <tool_response> inside a user turn — never a `tool` role.
        assert!(
            prompt.contains(
                "<|im_start|>user\n<tool_response>\nport = 8443\n</tool_response><|im_end|>"
            ),
            "tool result not wrapped as <tool_response> in a user turn:\n{prompt}"
        );
        assert!(
            !prompt.contains("<|im_start|>tool"),
            "must not emit an <|im_start|>tool role that Qwen never saw in training:\n{prompt}"
        );
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
