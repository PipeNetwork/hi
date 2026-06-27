//! Model registry backed by [models.dev](https://models.dev).
//!
//! `refresh()` fetches the full models.dev catalog, flattens it to the fields
//! we need, and writes a compact JSON cache. `Registry::load()` reads that
//! cache (fast to parse) and falls back to a small built-in table when no
//! cache exists — so the tool still works offline / before a first refresh.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const MODELS_DEV_URL: &str = "https://models.dev/api.json";

/// Capability and pricing metadata for a model. Costs are USD per 1M tokens.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub context_window: u32,
    pub max_output: u32,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub cost_input: f64,
    pub cost_output: f64,
}

impl ModelInfo {
    /// Estimated USD cost for a turn with the given token counts.
    pub fn cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        (input_tokens as f64 * self.cost_input + output_tokens as f64 * self.cost_output)
            / 1_000_000.0
    }
}

/// An in-memory set of known models with best-effort id lookup.
#[derive(Clone)]
pub struct Registry {
    models: Vec<ModelInfo>,
}

impl Registry {
    /// Load the cached models.dev catalog, merged over the built-in table.
    /// Never fails: a missing or corrupt cache just yields the built-ins.
    pub fn load() -> Self {
        let mut models = builtin();
        if let Some(cached) = read_cache() {
            // Cache wins; keep any built-ins it doesn't cover.
            let ids: std::collections::HashSet<&str> =
                cached.iter().map(|m| m.id.as_str()).collect();
            let extras: Vec<ModelInfo> = models
                .into_iter()
                .filter(|m| !ids.contains(m.id.as_str()))
                .collect();
            models = cached;
            models.extend(extras);
        }
        Self { models }
    }

    /// Pricing `(input, output)` per 1M tokens and context window for a model,
    /// each present only when known and non-zero.
    pub fn metadata(&self, model: &str) -> (Option<(f64, f64)>, Option<u32>) {
        match self.lookup(model) {
            Some(info) => (
                (info.cost_input > 0.0 || info.cost_output > 0.0)
                    .then_some((info.cost_input, info.cost_output)),
                (info.context_window > 0).then_some(info.context_window),
            ),
            None => (None, None),
        }
    }

    /// All known model ids, sorted and deduped — for the `/model` picker.
    pub fn model_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.models.iter().map(|m| m.id.clone()).collect();
        ids.sort();
        ids.dedup();
        ids
    }

    /// Best-effort lookup, tolerating provider prefixes and `.`/`-` differences
    /// between naming conventions (e.g. `anthropic/claude-3.5-sonnet`).
    pub fn lookup(&self, model: &str) -> Option<&ModelInfo> {
        let bare = model.rsplit('/').next().unwrap_or(model);
        if let Some(info) = self.models.iter().find(|m| m.id == model || m.id == bare) {
            return Some(info);
        }
        let norm = normalize(model);
        self.models.iter().find(|m| normalize(&m.id) == norm)
    }
}

fn normalize(s: &str) -> String {
    s.rsplit('/')
        .next()
        .unwrap_or(s)
        .replace('.', "-")
        .to_lowercase()
}

/// Fetch the live catalog and write the compact cache. Returns the model count.
pub async fn refresh() -> Result<usize> {
    let json = reqwest::Client::new()
        .get(MODELS_DEV_URL)
        .send()
        .await
        .context("fetching models.dev catalog")?
        .error_for_status()
        .context("models.dev returned an error")?
        .text()
        .await
        .context("reading models.dev response")?;

    let models = parse_catalog(&json)?;
    let path = cache_path().context("could not determine cache directory")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let compact = serde_json::to_string(&models)?;
    std::fs::write(&path, compact).with_context(|| format!("writing {}", path.display()))?;
    Ok(models.len())
}

/// Flatten the models.dev catalog (`{ provider: { models: { id: {...} } } }`)
/// into a flat, id-sorted list, first provider (alphabetically) winning on
/// duplicate model ids.
fn parse_catalog(json: &str) -> Result<Vec<ModelInfo>> {
    let root: BTreeMap<String, Value> =
        serde_json::from_str(json).context("parsing models.dev catalog")?;

    let mut by_id: BTreeMap<String, ModelInfo> = BTreeMap::new();
    for provider in root.values() {
        let Some(models) = provider.get("models").and_then(Value::as_object) else {
            continue;
        };
        for (id, m) in models {
            by_id.entry(id.clone()).or_insert_with(|| ModelInfo {
                id: id.clone(),
                name: m
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(id)
                    .to_string(),
                context_window: m
                    .pointer("/limit/context")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as u32,
                max_output: m
                    .pointer("/limit/output")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as u32,
                supports_tools: m.get("tool_call").and_then(Value::as_bool).unwrap_or(false),
                supports_reasoning: m.get("reasoning").and_then(Value::as_bool).unwrap_or(false),
                cost_input: m
                    .pointer("/cost/input")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                cost_output: m
                    .pointer("/cost/output")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
            });
        }
    }
    Ok(by_id.into_values().collect())
}

pub fn cache_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(base.join("hi").join("models.json"))
}

fn read_cache() -> Option<Vec<ModelInfo>> {
    let path = cache_path()?;
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// A few common models so lookups work before the first `--refresh-models`.
fn builtin() -> Vec<ModelInfo> {
    fn m(
        id: &str,
        name: &str,
        context_window: u32,
        max_output: u32,
        cost_input: f64,
        cost_output: f64,
    ) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            name: name.into(),
            context_window,
            max_output,
            supports_tools: true,
            supports_reasoning: false,
            cost_input,
            cost_output,
        }
    }
    vec![
        m(
            "claude-sonnet-4-20250514",
            "Claude Sonnet 4",
            200_000,
            64_000,
            3.0,
            15.0,
        ),
        m(
            "claude-3-5-haiku-20241022",
            "Claude 3.5 Haiku",
            200_000,
            8_192,
            0.8,
            4.0,
        ),
        m("gpt-4o", "GPT-4o", 128_000, 16_384, 2.5, 10.0),
        m("qwen2.5-coder", "Qwen2.5 Coder", 32_768, 8_192, 0.0, 0.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::{ModelInfo, Registry, parse_catalog};

    #[test]
    fn parse_catalog_extracts_limits_caps_and_cost() {
        let json = r#"{"anthropic":{"models":{"claude-x":{
            "name":"Claude X",
            "limit":{"context":200000,"output":64000},
            "tool_call":true,"reasoning":true,
            "cost":{"input":3.0,"output":15.0}}}}}"#;
        let models = parse_catalog(json).unwrap();
        let m = models.iter().find(|m| m.id == "claude-x").unwrap();
        assert_eq!(m.context_window, 200_000);
        assert_eq!(m.max_output, 64_000);
        assert!(m.supports_tools);
        assert!(m.supports_reasoning);
        assert_eq!(m.cost_input, 3.0);
        assert_eq!(m.cost_output, 15.0);
    }

    #[test]
    fn lookup_tolerates_provider_prefix_and_dots() {
        let info = |id: &str| ModelInfo {
            id: id.into(),
            name: String::new(),
            context_window: 0,
            max_output: 0,
            supports_tools: true,
            supports_reasoning: false,
            cost_input: 0.0,
            cost_output: 0.0,
        };
        let reg = Registry {
            models: vec![info("claude-3-5-sonnet"), info("gpt-4o")],
        };
        assert!(reg.lookup("claude-3-5-sonnet").is_some());
        assert!(reg.lookup("anthropic/claude-3.5-sonnet").is_some()); // prefix + dots
        assert!(reg.lookup("openai/gpt-4o").is_some());
        assert!(reg.lookup("mistral-large").is_none());
    }
}
