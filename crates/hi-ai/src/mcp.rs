//! Pipe MCP client support used for metadata discovery.
//!
//! The hosted Pipe MCP endpoint is deliberately separate from the normal chat
//! execution path. `McpDiscoveryProvider` delegates streaming to the existing
//! provider and uses MCP only for model catalog/health metadata.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::provider::{Provider, ServedModel};
use crate::types::{ChatRequest, Completion, StreamEvent};

pub const PIPE_MCP_DEFAULT_URL: &str = "https://api.pipenetwork.ai/mcp";

const TOOL_MODELS_LIST: &str = "pipe.models.list";
const TOOL_MODELS_HEALTH: &str = "pipe.models.health";

#[derive(Clone)]
pub struct PipeMcpClient {
    http: reqwest::Client,
    url: String,
    api_key: String,
}

impl PipeMcpClient {
    pub fn new(url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: crate::http::agent_http_client_quick(),
            url: url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
        }
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub async fn initialize(&self) -> Result<Value> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "hi",
                    "version": env!("CARGO_PKG_VERSION"),
                },
            }),
        )
        .await
    }

    /// Handshake + parsed server identity, for callers that just want the
    /// display fields (server name, protocol version) without a `serde_json`
    /// dependency. Falls back to `"unknown"` when the server omits a field.
    pub async fn server_info(&self) -> Result<(String, String)> {
        let init = self.initialize().await?;
        let server = init
            .get("serverInfo")
            .and_then(|v| v.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let protocol = init
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        Ok((server, protocol))
    }

    pub async fn tools_list(&self) -> Result<Vec<McpTool>> {
        let result = self.request("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|value| serde_json::from_value(value).ok())
            .collect();
        Ok(tools)
    }

    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value> {
        let result = self
            .request(
                "tools/call",
                json!({
                    "name": name,
                    "arguments": arguments,
                }),
            )
            .await?;
        if result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let message = result
                .get("structuredContent")
                .and_then(|value| value.get("error"))
                .and_then(Value::as_str)
                .or_else(|| {
                    result
                        .get("content")
                        .and_then(Value::as_array)
                        .and_then(|items| items.first())
                        .and_then(|item| item.get("text"))
                        .and_then(Value::as_str)
                })
                .unwrap_or("MCP tool returned an error");
            bail!("{message}");
        }
        if let Some(value) = result.get("structuredContent") {
            return Ok(value.clone());
        }
        if let Some(text) = result
            .get("content")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .and_then(|item| item.get("text"))
            .and_then(Value::as_str)
            && let Ok(value) = serde_json::from_str::<Value>(text)
        {
            return Ok(value);
        }
        Ok(result)
    }

    pub async fn list_models(&self) -> Result<Vec<ServedModel>> {
        let value = self
            .call_tool(TOOL_MODELS_LIST, json!({ "include_unavailable": true }))
            .await?;
        Ok(parse_model_metadata(&value)
            .into_iter()
            .map(PipeMcpModelMetadata::into_served)
            .collect())
    }

    pub async fn model_metadata(&self) -> Result<HashMap<String, PipeMcpModelMetadata>> {
        let value = self
            .call_tool(TOOL_MODELS_LIST, json!({ "include_unavailable": true }))
            .await?;
        Ok(parse_model_metadata(&value)
            .into_iter()
            .map(|model| (model.id.clone(), model))
            .collect())
    }

    pub async fn models_health(&self) -> Result<HashMap<String, PipeMcpModelHealth>> {
        let value = self
            .call_tool(TOOL_MODELS_HEALTH, json!({ "include_unavailable": true }))
            .await?;
        Ok(parse_model_health(&value)
            .into_iter()
            .map(|model| (model.model_id.clone(), model))
            .collect())
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        });
        let response = crate::http::send_with_retry(
            self.http
                .post(&self.url)
                .bearer_auth(&self.api_key)
                .json(&body),
        )
        .await
        .with_context(|| format!("requesting Pipe MCP method {method}"))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            bail!("Pipe MCP endpoint returned {status}: {text}");
        }
        let value: Value = response.json().await.context("parsing MCP JSON-RPC")?;
        if let Some(error) = value.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("MCP JSON-RPC error");
            let code = error.get("code").and_then(Value::as_i64).unwrap_or(0);
            return Err(anyhow!("MCP error {code}: {message}"));
        }
        value
            .get("result")
            .cloned()
            .ok_or_else(|| anyhow!("MCP response missing result"))
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PipeMcpModelMetadata {
    pub id: String,
    pub provider_label: Option<String>,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    /// Pricing `(input, output)` in USD per 1M tokens.
    pub price: Option<(f64, f64)>,
    pub status: Option<String>,
    pub available: bool,
    pub availability_reason: Option<String>,
    pub unavailable_reasons: Vec<String>,
    pub capabilities: Vec<String>,
}

impl PipeMcpModelMetadata {
    pub fn into_served(self) -> ServedModel {
        let availability_reason = self
            .availability_reason
            .or_else(|| self.unavailable_reasons.first().cloned());
        ServedModel {
            id: self.id,
            context_window: self.context_window,
            max_output_tokens: self.max_output_tokens,
            price: self.price,
            provider_label: self.provider_label,
            status: self.status,
            available: self.available,
            availability_reason,
            capabilities: self.capabilities,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PipeMcpModelHealth {
    pub model_id: String,
    pub status: Option<String>,
    pub available: Option<bool>,
    pub chat_ready: Option<bool>,
    pub embeddings_ready: Option<bool>,
    pub tool_ready: Option<bool>,
    pub unavailable_reasons: Vec<String>,
}

/// Provider wrapper that uses MCP for `list_models` and falls back to the
/// wrapped provider's normal `/models` route when MCP is unavailable.
pub struct McpDiscoveryProvider {
    inner: Box<dyn Provider>,
    mcp: PipeMcpClient,
}

impl McpDiscoveryProvider {
    pub fn new(inner: Box<dyn Provider>, mcp: PipeMcpClient) -> Self {
        Self { inner, mcp }
    }
}

#[async_trait]
impl Provider for McpDiscoveryProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        self.inner.stream(request, sink).await
    }

    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        match self.mcp.list_models().await {
            Ok(models) if !models.is_empty() => Ok(models),
            _ => self.inner.list_models().await,
        }
    }
}

fn parse_model_metadata(value: &Value) -> Vec<PipeMcpModelMetadata> {
    value
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(parse_one_model_metadata)
        .collect()
}

fn parse_one_model_metadata(value: &Value) -> Option<PipeMcpModelMetadata> {
    let id = value.get("id").and_then(Value::as_str)?.to_string();
    let availability = value.get("availability").unwrap_or(&Value::Null);
    let pricing = value.get("pricing").unwrap_or(&Value::Null);
    let capabilities = capability_tags(value.get("capabilities"));
    Some(PipeMcpModelMetadata {
        id,
        provider_label: value
            .get("provider_label")
            .and_then(Value::as_str)
            .map(str::to_string),
        context_window: u32_metadata(value, &["context_window", "max_context_tokens"]),
        max_output_tokens: u32_metadata(
            value,
            &[
                "max_output_tokens",
                "max_completion_tokens",
                "output_token_limit",
                "/limits/max_output_tokens",
                "/limits/max_completion_tokens",
                "/limits/output_tokens",
            ],
        ),
        price: match (
            pricing.get("input_token_rate").and_then(Value::as_f64),
            pricing.get("output_token_rate").and_then(Value::as_f64),
        ) {
            (Some(input), Some(output)) => Some((input * 1_000_000.0, output * 1_000_000.0)),
            _ => None,
        },
        status: availability
            .get("status")
            .and_then(Value::as_str)
            .or_else(|| availability.get("label").and_then(Value::as_str))
            .map(str::to_string),
        available: availability
            .get("available")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        availability_reason: availability
            .get("reason")
            .and_then(Value::as_str)
            .map(str::to_string),
        unavailable_reasons: string_array(availability.get("unavailable_reasons")),
        capabilities,
    })
}

fn parse_model_health(value: &Value) -> Vec<PipeMcpModelHealth> {
    value
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let model_id = item
                .get("model_id")
                .or_else(|| item.get("id"))
                .and_then(Value::as_str)?
                .to_string();
            Some(PipeMcpModelHealth {
                model_id,
                status: item
                    .get("status")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                available: item.get("available").and_then(Value::as_bool),
                chat_ready: item.get("chat_ready").and_then(Value::as_bool),
                embeddings_ready: item.get("embeddings_ready").and_then(Value::as_bool),
                tool_ready: item.get("tool_ready").and_then(Value::as_bool),
                unavailable_reasons: string_array(item.get("unavailable_reasons")),
            })
        })
        .collect()
}

fn capability_tags(value: Option<&Value>) -> Vec<String> {
    let Some(Value::Object(map)) = value else {
        return Vec::new();
    };
    let mut tags: Vec<String> = map
        .iter()
        .filter(|(_, value)| value.as_bool().unwrap_or(false))
        .map(|(key, _)| normalize_capability(key))
        .collect();
    tags.sort();
    tags.dedup();
    tags
}

fn u32_metadata(value: &Value, fields_or_pointers: &[&str]) -> Option<u32> {
    fields_or_pointers.iter().find_map(|field| {
        let raw = if field.starts_with('/') {
            value.pointer(field)
        } else {
            value.get(field)
        }?;
        raw.as_u64().and_then(|n| u32::try_from(n).ok())
    })
}

fn normalize_capability(key: &str) -> String {
    match key {
        "parallel_tool_calls" => "parallel-tools".to_string(),
        "json_schema" | "tool_json_schema" => "json".to_string(),
        "image_generation" => "image".to_string(),
        "tool_call_normalization" => "tool-normalize".to_string(),
        other => other.replace('_', "-"),
    }
}

fn string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.as_str().map(str::to_string))
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parses_pipe_mcp_models_metadata() {
        let value = json!({
            "object": "pipe.models.list",
            "models": [{
                "id": "ipop/coder-balanced",
                "provider_label": "Pipe",
                "context_window": 1000000,
                "limits": {
                    "max_output_tokens": 131072
                },
                "capabilities": {
                    "chat": true,
                    "tools": true,
                    "reasoning": true,
                    "embeddings": false,
                    "parallel_tool_calls": true
                },
                "availability": {
                    "available": true,
                    "status": "available",
                    "label": "available",
                    "reason": null,
                    "unavailable_reasons": []
                },
                "pricing": {
                    "input_token_rate": 0.000001,
                    "output_token_rate": 0.000002
                }
            }]
        });
        let models = parse_model_metadata(&value);
        assert_eq!(models.len(), 1);
        let model = &models[0];
        assert_eq!(model.id, "ipop/coder-balanced");
        assert_eq!(model.provider_label.as_deref(), Some("Pipe"));
        assert_eq!(model.context_window, Some(1_000_000));
        assert_eq!(model.max_output_tokens, Some(131_072));
        assert_eq!(model.price, Some((1.0, 2.0)));
        assert!(model.capabilities.contains(&"tools".to_string()));
        assert!(model.capabilities.contains(&"parallel-tools".to_string()));

        let served = models.into_iter().next().unwrap().into_served();
        assert_eq!(served.provider_label.as_deref(), Some("Pipe"));
        assert_eq!(served.max_output_tokens, Some(131_072));
        assert_eq!(served.health(), None);
    }

    #[test]
    fn parses_pipe_mcp_health_metadata() {
        let value = json!({
            "object": "pipe.models.health",
            "models": [{
                "model_id": "down",
                "status": "unavailable",
                "available": false,
                "chat_ready": false,
                "tool_ready": false,
                "unavailable_reasons": ["provider disabled"]
            }]
        });
        let health = parse_model_health(&value);
        assert_eq!(health.len(), 1);
        assert_eq!(health[0].model_id, "down");
        assert_eq!(health[0].status.as_deref(), Some("unavailable"));
        assert_eq!(health[0].available, Some(false));
        assert_eq!(health[0].unavailable_reasons, vec!["provider disabled"]);
    }
}
