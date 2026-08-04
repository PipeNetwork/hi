//! DeepSeek-specific compatibility for OpenAI-compatible endpoints.

use serde_json::{Map, Value, json};
use std::collections::HashSet;

use crate::types::{DeepSeekCompat, ReasoningEffort};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolProtocol {
    OpenAiJson,
    NativeDsml,
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProviderCapabilities {
    pub(crate) deepseek: bool,
    pub(crate) official: bool,
    pub(crate) local_native_dsml: bool,
    pub(crate) supports_tool_choice: bool,
    pub(crate) supports_sampling_params: bool,
    pub(crate) requires_reasoning_content: bool,
    pub(crate) requires_assistant_content: bool,
    pub(crate) strict_tools: bool,
    pub(crate) default_thinking_enabled: bool,
    pub(crate) tool_protocol: ToolProtocol,
}

impl ProviderCapabilities {
    #[cfg(test)]
    pub(crate) const fn generic() -> Self {
        Self {
            deepseek: false,
            official: false,
            local_native_dsml: false,
            supports_tool_choice: true,
            supports_sampling_params: true,
            requires_reasoning_content: false,
            requires_assistant_content: false,
            strict_tools: false,
            default_thinking_enabled: false,
            tool_protocol: ToolProtocol::OpenAiJson,
        }
    }

    pub(crate) fn detect(base_url: &str, model: &str, mode: DeepSeekCompat) -> Self {
        let official = is_official_endpoint(base_url);
        let model_is_deepseek = is_deepseek_model(model);
        let deepseek = match mode {
            DeepSeekCompat::On => true,
            DeepSeekCompat::Off => false,
            DeepSeekCompat::Auto => official || model_is_deepseek,
        };
        let local_native_dsml = deepseek && is_local_endpoint(base_url);
        // PipeNetwork currently routes this model to a backend that rejects
        // strict tool schemas. Keep this known capability beside the other
        // wire-profile rules so one-shot CLI invocations do not pay a failed
        // strict request before the response-based cache can learn the same
        // fact.
        let known_non_strict_gateway = deepseek && is_pipenetwork_endpoint(base_url);
        Self {
            deepseek,
            official,
            local_native_dsml,
            supports_tool_choice: !deepseek,
            supports_sampling_params: !deepseek,
            requires_reasoning_content: deepseek,
            requires_assistant_content: deepseek,
            // The official endpoint exposes strict schemas through /beta. A
            // local native DSML route has no JSON strict flag to send; gateways
            // get the flag and the request layer provides one controlled
            // non-strict retry if they reject it.
            strict_tools: deepseek && !local_native_dsml && !known_non_strict_gateway,
            default_thinking_enabled: deepseek,
            tool_protocol: if local_native_dsml {
                ToolProtocol::NativeDsml
            } else if deepseek {
                ToolProtocol::Auto
            } else {
                ToolProtocol::OpenAiJson
            },
        }
    }

    pub(crate) fn model_for_request(&self, model: &str) -> String {
        if self.deepseek && self.official && is_deepseek_model(model) {
            "deepseek-v4-flash".to_string()
        } else {
            model.to_string()
        }
    }

    pub(crate) fn reasoning_wire_value(&self, effort: ReasoningEffort) -> &'static str {
        if !self.deepseek {
            return effort.as_str();
        }
        if self.official {
            // DeepSeek V4 accepts `high` and `max` for reasoning effort. The
            // neutral lower levels intentionally collapse to `high` because
            // sending `low` is rejected by the official V4 endpoint.
            return match effort {
                ReasoningEffort::Minimal
                | ReasoningEffort::Low
                | ReasoningEffort::Medium
                | ReasoningEffort::High => "high",
                ReasoningEffort::Xhigh => "max",
            };
        }
        match effort {
            ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
            ReasoningEffort::Medium | ReasoningEffort::High => "high",
            // Gateways commonly expose the older low/high compatibility
            // surface even when they route to DeepSeek V4.
            ReasoningEffort::Xhigh => "high",
        }
    }

    pub(crate) fn completion_url(&self, base_url: &str, strict_tools: bool) -> String {
        let mut base = base_url.trim_end_matches('/');
        if self.deepseek && self.official && base.ends_with("/v1") {
            base = &base[..base.len() - 3];
        }
        if strict_tools && self.deepseek && self.official && !base.ends_with("/beta") {
            format!("{base}/beta/chat/completions")
        } else {
            format!("{base}/chat/completions")
        }
    }

    pub(crate) fn diagnostic_status(&self, strict_tools: bool) -> String {
        let profile = if self.official {
            "official"
        } else if self.local_native_dsml {
            "local"
        } else {
            "gateway"
        };
        let protocol = match self.tool_protocol {
            ToolProtocol::OpenAiJson => "openai-json",
            ToolProtocol::NativeDsml => "native-dsml",
            ToolProtocol::Auto => "auto",
        };
        format!("deepseek profile={profile} protocol={protocol} strict={strict_tools}")
    }

    pub(crate) fn with_strict_tools(mut self, strict_tools: bool) -> Self {
        self.strict_tools = strict_tools;
        self
    }

    pub(crate) fn with_thinking_enabled(mut self, enabled: bool) -> Self {
        self.default_thinking_enabled = enabled;
        self
    }

    pub(crate) fn with_reasoning_content(mut self, enabled: bool) -> Self {
        self.requires_reasoning_content = enabled;
        self
    }
}

/// Apply a learned endpoint/model capability without overriding an explicit
/// compatibility choice. The cache is intentionally an Auto-mode optimization
/// because `on` and `off` are user-directed wire-format decisions.
pub(crate) fn apply_cached_strict_capability(
    capabilities: ProviderCapabilities,
    mode: DeepSeekCompat,
    cached_strict_tools: Option<bool>,
) -> ProviderCapabilities {
    if matches!(mode, DeepSeekCompat::Auto)
        && capabilities.strict_tools
        && cached_strict_tools == Some(false)
    {
        capabilities.with_strict_tools(false)
    } else {
        capabilities
    }
}

pub(crate) fn apply_cached_thinking_capability(
    capabilities: ProviderCapabilities,
    mode: DeepSeekCompat,
    cached_thinking_enabled: Option<bool>,
) -> ProviderCapabilities {
    if matches!(mode, DeepSeekCompat::Auto)
        && capabilities.deepseek
        && !capabilities.official
        && !capabilities.local_native_dsml
        && cached_thinking_enabled == Some(false)
    {
        capabilities.with_thinking_enabled(false)
    } else {
        capabilities
    }
}

pub(crate) fn strict_cache_key(base_url: &str, model: &str) -> String {
    format!(
        "{}\n{}",
        base_url.trim_end_matches('/').to_ascii_lowercase(),
        model.to_ascii_lowercase()
    )
}

pub(crate) fn is_deepseek_model(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase().replace(['_', ' '], "-");
    normalized.contains("deepseek") && (normalized.contains("v4") || normalized.contains("flash"))
}

pub(crate) fn is_official_endpoint(base_url: &str) -> bool {
    reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| host == "api.deepseek.com")
}

fn is_local_endpoint(base_url: &str) -> bool {
    let Some(host) = reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
    else {
        return false;
    };
    matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1" | "0.0.0.0")
}

fn is_pipenetwork_endpoint(base_url: &str) -> bool {
    reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| host == "api.pipenetwork.ai" || host.ends_with(".pipenetwork.ai"))
}

/// Convert a JSON Schema into the subset accepted by DeepSeek strict tools.
/// The original schema is never modified; this returns a request-only copy.
pub(crate) fn normalize_strict_schema(schema: &Value) -> Value {
    let defs = schema
        .get("$defs")
        .or_else(|| schema.get("definitions"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    normalize_schema(schema, &defs, &mut HashSet::new(), 0)
}

const MAX_SCHEMA_DEPTH: usize = 64;

fn normalize_schema(
    schema: &Value,
    defs: &Map<String, Value>,
    active_refs: &mut HashSet<String>,
    depth: usize,
) -> Value {
    if depth >= MAX_SCHEMA_DEPTH {
        return bounded_schema_fallback();
    }
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
        && let Some(name) = reference
            .strip_prefix("#/$defs/")
            .or_else(|| reference.strip_prefix("#/definitions/"))
        && let Some(target) = defs.get(name)
    {
        // Recursive local references cannot be expanded into a finite strict
        // schema. Stop at the cycle with a closed empty object rather than
        // overflowing the stack or sending `$ref` to a provider that cannot
        // consume it. The normalizer is request-only; client-side validation
        // still uses the original recursive schema.
        if !active_refs.insert(name.to_string()) {
            return bounded_schema_fallback();
        }
        let normalized = normalize_schema(target, defs, active_refs, depth + 1);
        active_refs.remove(name);
        return normalized;
    }
    if let Some(reference) = schema.get("$ref") {
        // Keep an unresolved/external reference intact. Silently changing it
        // to `type: string` corrupts the tool contract; if the strict route
        // cannot resolve it, the request layer will use its one controlled
        // non-strict retry with the original client schema.
        let mut out = Map::new();
        out.insert("$ref".to_string(), reference.clone());
        if let Some(description) = schema.get("description") {
            out.insert("description".to_string(), description.clone());
        }
        return Value::Object(out);
    }

    let union = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(Value::as_array);
    let has_object_shape = schema.get("type").and_then(Value::as_str) == Some("object")
        || schema.get("properties").is_some();
    if let Some(options) = union
        && !has_object_shape
    {
        let normalized: Vec<Value> = options
            .iter()
            .map(|option| normalize_schema(option, defs, active_refs, depth + 1))
            .collect();
        return json!({ "anyOf": normalized });
    }
    if let Some(types) = schema.get("type").and_then(Value::as_array) {
        let options: Vec<Value> = types
            .iter()
            .filter_map(Value::as_str)
            .map(|kind| {
                let mut branch = schema.clone();
                if let Some(branch) = branch.as_object_mut() {
                    branch.insert("type".to_string(), json!(kind));
                }
                normalize_schema(&branch, defs, active_refs, depth + 1)
            })
            .collect();
        if !options.is_empty() {
            let mut out = Map::new();
            out.insert("anyOf".to_string(), Value::Array(options));
            if let Some(description) = schema.get("description") {
                out.insert("description".to_string(), description.clone());
            }
            return Value::Object(out);
        }
    }

    let schema_type = schema.get("type").and_then(Value::as_str);
    let mut out = Map::new();
    match schema_type {
        Some("object") | None if schema.get("properties").is_some() => {
            out.insert("type".to_string(), json!("object"));
            let properties = schema
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let required: std::collections::HashSet<&str> = schema
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            let mut normalized_properties = Map::new();
            let mut normalized_required = Vec::new();
            for (name, value) in properties {
                let was_required = required.contains(name.as_str());
                let mut normalized = normalize_schema(&value, defs, active_refs, depth + 1);
                if !was_required {
                    normalized = nullable(normalized);
                }
                normalized_properties.insert(name.clone(), normalized);
                normalized_required.push(Value::String(name));
            }
            out.insert(
                "properties".to_string(),
                Value::Object(normalized_properties),
            );
            out.insert("required".to_string(), Value::Array(normalized_required));
            out.insert("additionalProperties".to_string(), json!(false));
            if let Some(options) = union {
                // Keep object properties alongside a root oneOf/anyOf. A
                // branch containing only `required` has no type information
                // to normalize; preserving that constraint directly avoids
                // turning it into the bogus fallback `type: string`.
                let normalized = options
                    .iter()
                    .map(|option| normalize_union_option(option, defs, active_refs, depth + 1))
                    .collect();
                out.insert("anyOf".to_string(), Value::Array(normalized));
            }
        }
        Some("object") => {
            // DeepSeek strict tools require closed object schemas. A schema
            // with no declared properties is still an object (commonly a
            // map/free-form payload); never silently turn it into a string.
            out.insert("type".to_string(), json!("object"));
            out.insert("properties".to_string(), json!({}));
            out.insert("required".to_string(), json!([]));
            out.insert("additionalProperties".to_string(), json!(false));
        }
        Some("array") => {
            out.insert("type".to_string(), json!("array"));
            if let Some(items) = schema.get("items") {
                out.insert(
                    "items".to_string(),
                    normalize_schema(items, defs, active_refs, depth + 1),
                );
            }
        }
        Some("null") => {
            out.insert("type".to_string(), json!("null"));
        }
        Some(kind @ ("string" | "number" | "integer" | "boolean")) => {
            out.insert("type".to_string(), json!(kind));
            copy_supported_scalar_keywords(schema, &mut out, kind);
        }
        _ => {
            if let Some(value) = schema.get("enum") {
                out.insert("enum".to_string(), value.clone());
            } else {
                out.insert("type".to_string(), json!("string"));
            }
        }
    }
    if let Some(description) = schema.get("description") {
        out.insert("description".to_string(), description.clone());
    }
    Value::Object(out)
}

fn normalize_union_option(
    option: &Value,
    defs: &Map<String, Value>,
    active_refs: &mut HashSet<String>,
    depth: usize,
) -> Value {
    if option.get("required").is_some()
        && option.get("type").is_none()
        && option.get("properties").is_none()
        && option.get("anyOf").is_none()
        && option.get("oneOf").is_none()
    {
        let required = option
            .get("required")
            .cloned()
            .unwrap_or_else(|| Value::Array(Vec::new()));
        json!({ "required": required })
    } else {
        normalize_schema(option, defs, active_refs, depth + 1)
    }
}

fn bounded_schema_fallback() -> Value {
    json!({
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": false
    })
}

fn copy_supported_scalar_keywords(schema: &Value, out: &mut Map<String, Value>, kind: &str) {
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        let values = values
            .iter()
            .filter(|value| json_value_matches_type(value, kind))
            .cloned()
            .collect::<Vec<_>>();
        if !values.is_empty() {
            out.insert("enum".to_string(), Value::Array(values));
        }
    }
    if let Some(value) = schema.get("const")
        && json_value_matches_type(value, kind)
    {
        out.insert("const".to_string(), value.clone());
    }
    if let Some(value) = schema.get("default")
        && json_value_matches_type(value, kind)
    {
        out.insert("default".to_string(), value.clone());
    }
    if kind == "string" {
        for key in ["format", "pattern"] {
            if let Some(value) = schema.get(key) {
                out.insert(key.to_string(), value.clone());
            }
        }
    } else if matches!(kind, "number" | "integer") {
        for key in [
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        ] {
            if let Some(value) = schema.get(key)
                && value.is_number()
            {
                out.insert(key.to_string(), value.clone());
            }
        }
    }
}

fn json_value_matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn nullable(schema: Value) -> Value {
    json!({ "anyOf": [schema, { "type": "null" }] })
}

/// Parse a complete native DSML response into prose/tool blocks. Streaming
/// suppression is handled by the caller; this parser is deliberately bounded
/// and only accepts closed invoke/parameter elements.
pub(crate) fn parse_dsml_tool_calls(
    text: &str,
    id_prefix: &str,
) -> Option<Vec<crate::types::Content>> {
    const OPEN: &str = "<｜DSML｜tool_calls>";
    const CLOSE: &str = "</｜DSML｜tool_calls>";
    const MAX_DSML_BYTES: usize = 8 * 1024 * 1024;
    if text.len() > MAX_DSML_BYTES || !text.contains(OPEN) || !text.contains(CLOSE) {
        return None;
    }
    let start = text.find(OPEN)?;
    let end = text[start + OPEN.len()..].find(CLOSE)? + start + OPEN.len();
    let payload = &text[start + OPEN.len()..end];
    let mut content = Vec::new();
    let before = text[..start].trim_end_matches(['\n', ' ']);
    if !before.is_empty() {
        content.push(crate::types::Content::Text(before.to_string()));
    }
    let mut cursor = 0;
    let mut call_index = 0;
    while let Some(relative) = payload[cursor..].find("<｜DSML｜invoke name=\"") {
        let invoke_start = cursor + relative;
        if !payload[cursor..invoke_start].trim().is_empty() {
            return None;
        }
        let name_start = invoke_start + "<｜DSML｜invoke name=\"".len();
        let name_end = payload[name_start..].find("\">")? + name_start;
        let name = &payload[name_start..name_end];
        if name.is_empty() {
            return None;
        }
        let invoke_close = payload[name_end + 2..].find("</｜DSML｜invoke>")? + name_end + 2;
        let body = &payload[name_end + 2..invoke_close];
        let mut args = Map::new();
        let mut body_cursor = 0;
        while let Some(relative_param) = body[body_cursor..].find("<｜DSML｜parameter name=\"") {
            let param_start = body_cursor + relative_param;
            if !body[body_cursor..param_start].trim().is_empty() {
                return None;
            }
            let param_name_start = param_start + "<｜DSML｜parameter name=\"".len();
            let param_name_end = body[param_name_start..].find("\" string=\"")? + param_name_start;
            let param_name = &body[param_name_start..param_name_end];
            if param_name.is_empty() {
                return None;
            }
            let string_start = param_name_end + "\" string=\"".len();
            let string_end = body[string_start..].find("\">")? + string_start;
            let is_string = match &body[string_start..string_end] {
                "true" => true,
                "false" => false,
                _ => return None,
            };
            let value_start = string_end + 2;
            let value_end = body[value_start..].find("</｜DSML｜parameter>")? + value_start;
            let raw = &body[value_start..value_end];
            let value = if is_string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw).ok()?
            };
            if args.insert(param_name.to_string(), value).is_some() {
                return None;
            }
            body_cursor = value_end + "</｜DSML｜parameter>".len();
        }
        if !body[body_cursor..].trim().is_empty() {
            return None;
        }
        content.push(crate::types::Content::ToolCall {
            id: format!("{id_prefix}_{call_index}"),
            name: name.to_string(),
            arguments: Value::Object(args).to_string(),
        });
        call_index += 1;
        cursor = invoke_close + "</｜DSML｜invoke>".len();
    }
    if !payload[cursor..].trim().is_empty() {
        return None;
    }
    if call_index == 0 {
        return None;
    }
    let sanitized_after = strip_dsml_artifacts(&text[end + CLOSE.len()..]);
    let after = sanitized_after.trim_matches(['\n', ' ']);
    if !after.is_empty() {
        content.push(crate::types::Content::Text(after.to_string()));
    }
    Some(content)
}

/// Remove native DSML markup from a text fallback without ever interpreting
/// it as a tool call. This is used when parsing fails, including incomplete or
/// oversized output, so rejected protocol text cannot leak into the replayed
/// transcript.
pub(crate) fn strip_dsml_artifacts(text: &str) -> String {
    const OPEN: &str = "<｜DSML｜tool_calls>";
    const CLOSE: &str = "</｜DSML｜tool_calls>";
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    while cursor < text.len() {
        let Some(relative_start) = text[cursor..].find(OPEN) else {
            out.push_str(&text[cursor..].replace(CLOSE, ""));
            break;
        };
        let start = cursor + relative_start;
        out.push_str(&text[cursor..start]);
        let body_start = start + OPEN.len();
        let Some(relative_end) = text[body_start..].find(CLOSE) else {
            // An unterminated block is rejected and removed through EOF.
            break;
        };
        cursor = body_start + relative_end + CLOSE.len();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        ProviderCapabilities, apply_cached_strict_capability, apply_cached_thinking_capability,
        is_deepseek_model, is_official_endpoint, normalize_strict_schema, parse_dsml_tool_calls,
        strict_cache_key,
    };
    use crate::types::{Content, ReasoningEffort};
    use serde_json::json;

    #[test]
    fn detects_aliases_and_local_routes() {
        assert!(is_deepseek_model("DeepSeek-V4-Flash-0731"));
        assert!(is_deepseek_model("deepseek/deepseek-v4-flash"));
        assert!(!is_deepseek_model("gpt-5"));

        let official = ProviderCapabilities::detect(
            "https://api.deepseek.com",
            "deepseek-v4-flash",
            crate::types::DeepSeekCompat::Auto,
        );
        assert!(official.deepseek);
        assert!(official.strict_tools);
        assert!(!official.supports_tool_choice);
        assert!(official.default_thinking_enabled);

        let local = ProviderCapabilities::detect(
            "http://127.0.0.1:8000/v1",
            "deepseek-v4-flash",
            crate::types::DeepSeekCompat::Auto,
        );
        assert!(local.local_native_dsml);
        assert!(!local.strict_tools);

        let gateway_auto = ProviderCapabilities::detect(
            "https://gateway.example/v1",
            "deepseek-v4-flash",
            crate::types::DeepSeekCompat::Auto,
        );
        assert!(gateway_auto.deepseek);
        assert!(gateway_auto.strict_tools);
        let pipenetwork = ProviderCapabilities::detect(
            "https://api.pipenetwork.ai/v1",
            "pipe/deepseek-v4-flash-0731",
            crate::types::DeepSeekCompat::Auto,
        );
        assert!(pipenetwork.deepseek);
        assert!(!pipenetwork.strict_tools);
        assert!(pipenetwork.default_thinking_enabled);
        assert!(
            !apply_cached_thinking_capability(
                pipenetwork,
                crate::types::DeepSeekCompat::Auto,
                Some(false)
            )
            .default_thinking_enabled
        );
        let gateway_off = ProviderCapabilities::detect(
            "https://gateway.example/v1",
            "deepseek-v4-flash",
            crate::types::DeepSeekCompat::Off,
        );
        assert!(!gateway_off.deepseek);
        let gateway_on = ProviderCapabilities::detect(
            "https://gateway.example/v1",
            "some-alias",
            crate::types::DeepSeekCompat::On,
        );
        assert!(gateway_on.deepseek);
        assert!(
            !apply_cached_strict_capability(
                gateway_on,
                crate::types::DeepSeekCompat::Auto,
                Some(false)
            )
            .strict_tools
        );
        assert!(
            apply_cached_strict_capability(
                gateway_on,
                crate::types::DeepSeekCompat::On,
                Some(false)
            )
            .strict_tools
        );
        assert!(
            !ProviderCapabilities::detect(
                "https://api.deepseek.com.example/v1",
                "deepseek-v4-flash",
                crate::types::DeepSeekCompat::Auto,
            )
            .official
        );
        assert_eq!(
            local.diagnostic_status(false),
            "deepseek profile=local protocol=native-dsml strict=false"
        );
        assert_eq!(
            official.completion_url("https://api.deepseek.com/v1", true),
            "https://api.deepseek.com/beta/chat/completions"
        );
        assert_eq!(
            official.completion_url("https://api.deepseek.com/v1", false),
            "https://api.deepseek.com/chat/completions"
        );
        assert_eq!(
            gateway_auto.completion_url("https://gateway.example/proxy/v1", true),
            "https://gateway.example/proxy/v1/chat/completions"
        );
        let disabled = ProviderCapabilities::detect(
            "https://api.deepseek.com/v1",
            "deepseek-v4-flash",
            crate::types::DeepSeekCompat::Off,
        );
        assert_eq!(
            disabled.completion_url("https://api.deepseek.com/v1", false),
            "https://api.deepseek.com/v1/chat/completions"
        );
        assert_eq!(
            strict_cache_key("https://Gateway.example/v1/", "DeepSeek-V4-Flash"),
            "https://gateway.example/v1\ndeepseek-v4-flash"
        );
        assert!(is_official_endpoint("https://api.deepseek.com/v1"));
    }

    #[test]
    fn official_flash_effort_mapping_uses_v4_values() {
        let caps = ProviderCapabilities::detect(
            "https://api.deepseek.com",
            "deepseek-v4-flash",
            crate::types::DeepSeekCompat::On,
        );
        assert_eq!(caps.reasoning_wire_value(ReasoningEffort::Minimal), "high");
        assert_eq!(caps.reasoning_wire_value(ReasoningEffort::Low), "high");
        assert_eq!(caps.reasoning_wire_value(ReasoningEffort::Medium), "high");
        assert_eq!(caps.reasoning_wire_value(ReasoningEffort::High), "high");
        assert_eq!(caps.reasoning_wire_value(ReasoningEffort::Xhigh), "max");
    }

    #[test]
    fn gateway_flash_effort_mapping_keeps_legacy_values() {
        let caps = ProviderCapabilities::detect(
            "https://gateway.example/v1",
            "deepseek-v4-flash",
            crate::types::DeepSeekCompat::Auto,
        );
        assert_eq!(caps.reasoning_wire_value(ReasoningEffort::Minimal), "low");
        assert_eq!(caps.reasoning_wire_value(ReasoningEffort::Low), "low");
        assert_eq!(caps.reasoning_wire_value(ReasoningEffort::Medium), "high");
        assert_eq!(caps.reasoning_wire_value(ReasoningEffort::High), "high");
        assert_eq!(caps.reasoning_wire_value(ReasoningEffort::Xhigh), "high");
    }

    #[test]
    fn strict_schema_expands_refs_and_makes_optional_properties_nullable() {
        let normalized = normalize_strict_schema(&json!({
            "$defs": {
                "path": {"type": "string", "minLength": 1}
            },
            "type": "object",
            "properties": {
                "path": {"$ref": "#/$defs/path"},
                "line": {"type": "integer"},
                "mode": {"type": ["string", "null"]}
            },
            "required": ["path", "mode"],
            "additionalProperties": true
        }));
        assert_eq!(normalized["additionalProperties"], false);
        assert_eq!(normalized["required"], json!(["line", "mode", "path"]));
        assert_eq!(normalized["properties"]["path"]["type"], "string");
        assert!(normalized["properties"]["path"].get("minLength").is_none());
        assert_eq!(normalized["properties"]["line"]["anyOf"][1]["type"], "null");
        assert_eq!(
            normalized["properties"]["mode"]["anyOf"][0]["type"],
            "string"
        );
    }

    #[test]
    fn strict_schema_keeps_propertyless_objects_as_objects() {
        let normalized = normalize_strict_schema(&json!({
            "type": "object",
            "description": "A dynamic payload"
        }));

        assert_eq!(normalized["type"], "object");
        assert_eq!(normalized["properties"], json!({}));
        assert_eq!(normalized["required"], json!([]));
        assert_eq!(normalized["additionalProperties"], false);
        assert_eq!(normalized["description"], "A dynamic payload");
    }

    #[test]
    fn strict_schema_bounds_recursive_local_refs() {
        let normalized = normalize_strict_schema(&json!({
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "next": {"$ref": "#/$defs/node"}
                    }
                }
            },
            "$ref": "#/$defs/node"
        }));
        assert!(normalized.to_string().len() < 4096);
        assert!(normalized.get("$ref").is_none());
        assert_eq!(
            normalized["properties"]["next"]["anyOf"][0]["additionalProperties"],
            false
        );
    }

    #[test]
    fn strict_schema_does_not_corrupt_unresolved_refs() {
        let normalized = normalize_strict_schema(&json!({
            "type": "object",
            "properties": {
                "payload": {"$ref": "#/components/schemas/Payload"}
            }
        }));

        assert_eq!(
            normalized["properties"]["payload"]["anyOf"][0]["$ref"],
            "#/components/schemas/Payload"
        );
        assert_ne!(
            normalized["properties"]["payload"]["anyOf"][0]["type"],
            "string"
        );
    }

    #[test]
    fn strict_schema_preserves_constraints_for_type_unions() {
        let normalized = normalize_strict_schema(&json!({
            "type": ["string", "null"],
            "enum": ["open", "closed", null],
            "description": "state"
        }));

        assert_eq!(normalized["description"], "state");
        assert_eq!(normalized["anyOf"][0]["type"], "string");
        assert_eq!(normalized["anyOf"][0]["enum"], json!(["open", "closed"]));
        assert_eq!(normalized["anyOf"][1]["type"], "null");
    }

    #[test]
    fn strict_schema_keeps_properties_when_object_uses_one_of() {
        let normalized = normalize_strict_schema(&json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer"}
            },
            "oneOf": [{"required": ["path"]}, {"required": ["paths"]}]
        }));

        assert_eq!(normalized["type"], "object");
        assert!(normalized["properties"].get("path").is_some());
        assert!(normalized["properties"].get("paths").is_some());
        assert!(normalized["properties"].get("limit").is_some());
        assert_eq!(normalized["required"], json!(["limit", "path", "paths"]));
        assert_eq!(
            normalized["anyOf"],
            json!([{"required": ["path"]}, {"required": ["paths"]}])
        );
    }

    #[test]
    fn strict_schema_preserves_supported_numeric_constraints() {
        let normalized = normalize_strict_schema(&json!({
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 11,
                    "multipleOf": 1,
                    "default": 2,
                    "minItems": 1,
                    "maxItems": 10
                }
            },
            "required": ["count"]
        }));
        let count = &normalized["properties"]["count"];
        assert_eq!(count["minimum"], 1);
        assert_eq!(count["maximum"], 10);
        assert_eq!(count["exclusiveMinimum"], 0);
        assert_eq!(count["exclusiveMaximum"], 11);
        assert_eq!(count["multipleOf"], 1);
        assert_eq!(count["default"], 2);
        assert!(count.get("minItems").is_none());
        assert!(count.get("maxItems").is_none());
    }

    #[test]
    fn dsml_parser_preserves_prose_and_json_parameters() {
        let text = "before\n<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\"><｜DSML｜parameter name=\"path\" string=\"true\">README.md</｜DSML｜parameter><｜DSML｜parameter name=\"line\" string=\"false\">12</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>\nafter";
        let content = parse_dsml_tool_calls(text, "dsml_test_4").expect("valid DSML");
        assert!(matches!(&content[0], Content::Text(text) if text == "before"));
        if let Content::ToolCall {
            id,
            name,
            arguments,
        } = &content[1]
        {
            let args: serde_json::Value = serde_json::from_str(arguments).unwrap();
            assert_eq!(id, "dsml_test_4_0");
            assert_eq!(name, "read");
            assert_eq!(args["path"], "README.md");
            assert_eq!(args["line"], 12);
        } else {
            panic!("expected DSML tool call: {content:?}");
        }
        assert!(matches!(&content[2], Content::Text(text) if text == "after"));
    }

    #[test]
    fn dsml_parser_rejects_malformed_attributes_and_body() {
        let malformed_attribute = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"read\">",
            "<｜DSML｜parameter name=\"path\" string=\"maybe\">src/lib.rs",
            "</｜DSML｜parameter></｜DSML｜invoke>",
            "</｜DSML｜tool_calls>"
        );
        assert!(parse_dsml_tool_calls(malformed_attribute, "dsml_test").is_none());

        let malformed_body = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"read\">unexpected",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls>"
        );
        assert!(parse_dsml_tool_calls(malformed_body, "dsml_test").is_none());

        let duplicate_parameter = concat!(
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\">",
            "<｜DSML｜parameter name=\"path\" string=\"true\">a</｜DSML｜parameter>",
            "<｜DSML｜parameter name=\"path\" string=\"true\">b</｜DSML｜parameter>",
            "</｜DSML｜invoke></｜DSML｜tool_calls>"
        );
        assert!(parse_dsml_tool_calls(duplicate_parameter, "dsml_test").is_none());
    }

    #[test]
    fn dsml_parser_rejects_oversized_payloads() {
        let oversized = format!(
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\"><｜DSML｜parameter name=\"path\" string=\"true\">{}</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
            "x".repeat(8 * 1024 * 1024)
        );
        assert!(parse_dsml_tool_calls(&oversized, "dsml_test").is_none());
    }

    #[test]
    fn dsml_parser_assigns_unique_ids_to_multiple_calls() {
        let text = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"read\"></｜DSML｜invoke>",
            "<｜DSML｜invoke name=\"grep\"></｜DSML｜invoke>",
            "</｜DSML｜tool_calls>"
        );
        let content = parse_dsml_tool_calls(text, "dsml_unique").expect("valid DSML");
        let ids = content
            .iter()
            .filter_map(|content| match content {
                Content::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(ids, ["dsml_unique_0", "dsml_unique_1"]);
    }
}
