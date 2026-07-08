use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::manifest::ModelFamily;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlxModelConfig {
    pub path: PathBuf,
    pub raw: Value,
    pub family: ModelFamily,
    pub model_type: String,
    pub architectures: Vec<String>,
    pub hidden_size: u32,
    pub intermediate_size: Option<u32>,
    pub moe_intermediate_size: Option<u32>,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: Option<u32>,
    pub partial_rotary_factor: Option<f32>,
    // Qwen3.5 gated-delta-net (linear attention) hybrid fields.
    pub linear_num_value_heads: Option<u32>,
    pub linear_num_key_heads: Option<u32>,
    pub linear_key_head_dim: Option<u32>,
    pub linear_value_head_dim: Option<u32>,
    pub linear_conv_kernel_dim: Option<u32>,
    pub full_attention_interval: Option<u32>,
    pub num_nextn_predict_layers: Option<u32>,
    // Nemotron-H Mamba2 hybrid fields.
    pub hybrid_override_pattern: Option<String>,
    pub ssm_state_size: Option<u32>,
    pub mamba_conv_kernel: Option<u32>,
    pub mamba_n_groups: Option<u32>,
    pub mamba_num_heads: Option<u32>,
    pub mamba_head_dim: Option<u32>,
    pub qk_nope_head_dim: Option<u32>,
    pub qk_rope_head_dim: Option<u32>,
    pub v_head_dim: Option<u32>,
    pub q_lora_rank: Option<u32>,
    pub kv_lora_rank: Option<u32>,
    pub index_head_dim: Option<u32>,
    pub index_n_heads: Option<u32>,
    pub index_topk: Option<u32>,
    pub indexer_rope_interleave: bool,
    pub compress_ratios: Vec<u32>,
    pub compress_rope_theta: f32,
    pub sliding_window: Option<u32>,
    pub o_lora_rank: Option<u32>,
    pub o_groups: Option<u32>,
    pub swiglu_limit: Option<f32>,
    // Gemma-4 hybrid attention fields.
    pub layer_types: Vec<String>,
    pub final_logit_softcapping: Option<f32>,
    pub vocab_size: u32,
    pub context_length: Option<u32>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: Option<Value>,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    pub first_k_dense_replace: u32,
    pub moe_layer_freq: u32,
    pub n_routed_experts: Option<u32>,
    pub n_shared_experts: Option<u32>,
    pub num_experts_per_tok: Option<u32>,
    pub decoder_sparse_step: u32,
    pub mlp_only_layers: Vec<u32>,
    pub shared_expert_intermediate_size: Option<u32>,
    pub n_group: u32,
    pub topk_group: u32,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub num_hash_layers: u32,
    pub topk_method: Option<String>,
    pub scoring_func: Option<String>,
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
    pub hc_eps: f32,
    pub quantization: QuantizationConfig,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: Option<u32>,
}

impl MlxModelConfig {
    pub fn attention_head_dim(&self) -> u32 {
        self.head_dim
            .or_else(|| {
                self.qk_rope_head_dim
                    .zip(self.qk_nope_head_dim)
                    .map(|(a, b)| a + b)
            })
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads.max(1))
    }

    pub fn is_moe_layer(&self, layer_idx: u32) -> bool {
        self.n_routed_experts.is_some()
            && layer_idx >= self.first_k_dense_replace
            && layer_idx % self.moe_layer_freq.max(1) == 0
    }

    pub fn is_qwen_moe_layer(&self, layer_idx: u32) -> bool {
        if self.n_routed_experts.unwrap_or(0) == 0 {
            return false;
        }
        if self.family == ModelFamily::Hy3 {
            return layer_idx >= self.first_k_dense_replace;
        }
        if self.model_type.contains("qwen3") {
            !self.mlp_only_layers.contains(&layer_idx)
                && (layer_idx + 1) % self.decoder_sparse_step.max(1) == 0
        } else {
            self.model_type.contains("qwen2_moe")
        }
    }

    pub fn is_deepseek_v4(&self) -> bool {
        self.model_type.eq_ignore_ascii_case("deepseek_v4")
            || self
                .architectures
                .iter()
                .any(|arch| arch.contains("deepseekv4"))
    }

    pub fn quantization_label(&self) -> String {
        self.quantization.label()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationConfig {
    Dense,
    Mlx(QuantizationSpec),
    Mixed {
        default: Option<QuantizationSpec>,
        entries: BTreeMap<String, QuantizationSpec>,
        modes: Vec<String>,
    },
}

impl QuantizationConfig {
    pub fn label(&self) -> String {
        match self {
            Self::Dense => "dense".to_string(),
            Self::Mlx(spec) => spec.label(),
            Self::Mixed { modes, .. } => {
                if modes.is_empty() {
                    "mixed".to_string()
                } else {
                    format!("mixed:{}", modes.join(","))
                }
            }
        }
    }

    pub fn standard_mlx(&self) -> Result<Option<(u32, u32)>> {
        match self {
            Self::Mixed { modes, .. } => bail!(
                "unsupported quantization: mixed/dynamic formats ({}) are not mapped by hi-mlx yet",
                modes.join(", ")
            ),
            _ => self.standard_mlx_for("<default>"),
        }
    }

    pub fn standard_mlx_for(&self, prefix: &str) -> Result<Option<(u32, u32)>> {
        Ok(self
            .mlx_quantization_for(prefix)?
            .map(|spec| (spec.bits, spec.group_size)))
    }

    pub fn mlx_quantization_for(&self, prefix: &str) -> Result<Option<QuantizationSpec>> {
        match self {
            Self::Dense => Ok(None),
            Self::Mlx(spec) => spec.mlx_supported(prefix).map(Some),
            Self::Mixed {
                default, entries, ..
            } => match entries.get(prefix).or(default.as_ref()) {
                Some(spec) => spec.mlx_supported(prefix).map(Some),
                None => Ok(None),
            },
        }
    }

    pub fn validate_supported(&self) -> Result<()> {
        match self {
            Self::Dense | Self::Mlx(_) => {
                self.standard_mlx()?;
                Ok(())
            }
            Self::Mixed {
                default, entries, ..
            } => {
                if let Some(spec) = default {
                    spec.mlx_supported("<default>")?;
                }
                for (prefix, spec) in entries {
                    spec.mlx_supported(prefix)?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationSpec {
    pub bits: u32,
    pub group_size: u32,
    pub mode: QuantizationMode,
}

impl QuantizationSpec {
    pub fn label(&self) -> String {
        format!(
            "{}-bit/{}/{}",
            self.bits,
            self.group_size,
            self.mode.as_str()
        )
    }

    fn mlx_supported(&self, prefix: &str) -> Result<Self> {
        match self.mode {
            // MLX affine quantization supports these bit-widths; hi-mlx passes `bits` straight to the
            // MLX quantized ops, so dynamic/mixed-bit builds (e.g. GLM-5.2 3.5bpw: 3/4/6-bit) work.
            QuantizationMode::Affine if matches!(self.bits, 2 | 3 | 4 | 5 | 6 | 8) => Ok(self.clone()),
            QuantizationMode::Other(ref mode)
                if mode == "mxfp4" && self.bits == 4 && self.group_size == 32 =>
            {
                Ok(self.clone())
            }
            _ => bail!(
                "unsupported quantization for {prefix}: {}-bit {} is not mapped by hi-mlx; standard MLX affine 4-bit/8-bit, MXFP4 expert weights, and dense weights are supported",
                self.bits,
                self.mode.as_str()
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    Affine,
    Other(String),
}

impl QuantizationMode {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Affine => "affine",
            Self::Other(mode) => mode,
        }
    }
}

pub fn load_model_config(path: impl AsRef<Path>) -> Result<MlxModelConfig> {
    let path = path.as_ref();
    let raw = read_json(&path.join("config.json"))?;
    parse_model_config(path, raw)
}

pub fn parse_model_config(path: &Path, raw: Value) -> Result<MlxModelConfig> {
    // VL / multimodal configs (e.g. Qwen3.5) nest the text-model fields under `text_config`. Hoist
    // them to the top level (without clobbering top-level `model_type`/`architectures`/quantization)
    // so the rest of the parser finds `hidden_size`, `num_hidden_layers`, etc.
    let raw = match raw.get("text_config").and_then(Value::as_object).cloned() {
        Some(text_config) => {
            let mut merged = raw.clone();
            if let Some(obj) = merged.as_object_mut() {
                for (k, v) in text_config {
                    obj.entry(k).or_insert(v);
                }
            }
            merged
        }
        None => raw,
    };
    let model_type = str_field(&raw, "model_type").unwrap_or_default();
    let family = detect_family(&model_type, &raw).ok_or_else(|| {
        let found = if model_type.is_empty() {
            "missing".to_string()
        } else {
            format!("'{model_type}'")
        };
        anyhow!(
            "unsupported model_type {found}; hi-mlx supports these MLX model families: {}",
            supported_model_families()
        )
    })?;
    let architectures = architecture_strings(&raw);
    let context_length = ["max_position_embeddings", "seq_length", "n_ctx"]
        .iter()
        .find_map(|key| u32_field(&raw, key));
    let hidden_size = required_u32(&raw, "hidden_size")?;
    let num_attention_heads = required_u32(&raw, "num_attention_heads")?;
    Ok(MlxModelConfig {
        path: path.to_path_buf(),
        family,
        model_type,
        architectures,
        hidden_size,
        intermediate_size: u32_field(&raw, "intermediate_size"),
        moe_intermediate_size: u32_field(&raw, "moe_intermediate_size"),
        num_hidden_layers: required_u32(&raw, "num_hidden_layers")?,
        num_attention_heads,
        num_key_value_heads: u32_field(&raw, "num_key_value_heads").unwrap_or(num_attention_heads),
        head_dim: u32_field(&raw, "head_dim"),
        partial_rotary_factor: f32_field(&raw, "partial_rotary_factor")
            .or_else(|| {
                raw.get("rope_parameters")
                    .and_then(|p| p.get("partial_rotary_factor"))
                    .and_then(Value::as_f64)
                    .map(|v| v as f32)
            }),
        linear_num_value_heads: u32_field(&raw, "linear_num_value_heads"),
        linear_num_key_heads: u32_field(&raw, "linear_num_key_heads"),
        linear_key_head_dim: u32_field(&raw, "linear_key_head_dim"),
        linear_value_head_dim: u32_field(&raw, "linear_value_head_dim"),
        linear_conv_kernel_dim: u32_field(&raw, "linear_conv_kernel_dim"),
        full_attention_interval: u32_field(&raw, "full_attention_interval"),
        hybrid_override_pattern: str_field(&raw, "hybrid_override_pattern"),
        ssm_state_size: u32_field(&raw, "ssm_state_size"),
        mamba_conv_kernel: u32_field(&raw, "conv_kernel"),
        mamba_n_groups: u32_field(&raw, "n_groups"),
        mamba_num_heads: u32_field(&raw, "mamba_num_heads"),
        mamba_head_dim: u32_field(&raw, "mamba_head_dim"),
        num_nextn_predict_layers: u32_field(&raw, "num_nextn_predict_layers"),
        qk_nope_head_dim: u32_field(&raw, "qk_nope_head_dim"),
        qk_rope_head_dim: u32_field(&raw, "qk_rope_head_dim"),
        v_head_dim: u32_field(&raw, "v_head_dim"),
        q_lora_rank: u32_field(&raw, "q_lora_rank"),
        kv_lora_rank: u32_field(&raw, "kv_lora_rank"),
        index_head_dim: u32_field(&raw, "index_head_dim"),
        index_n_heads: u32_field(&raw, "index_n_heads"),
        index_topk: u32_field(&raw, "index_topk"),
        indexer_rope_interleave: bool_field(&raw, "indexer_rope_interleave").unwrap_or(false),
        compress_ratios: u32_list_field(&raw, "compress_ratios"),
        compress_rope_theta: f32_field(&raw, "compress_rope_theta").unwrap_or(160_000.0),
        sliding_window: u32_field(&raw, "sliding_window")
            .or_else(|| u32_field(&raw, "window_size")),
        o_lora_rank: u32_field(&raw, "o_lora_rank"),
        o_groups: u32_field(&raw, "o_groups"),
        swiglu_limit: f32_field(&raw, "swiglu_limit"),
        layer_types: raw
            .get("layer_types")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        final_logit_softcapping: f32_field(&raw, "final_logit_softcapping"),
        vocab_size: required_u32(&raw, "vocab_size")?,
        context_length,
        rms_norm_eps: f32_field(&raw, "rms_norm_eps").unwrap_or(1e-6),
        rope_theta: f32_field(&raw, "rope_theta")
            .or_else(|| {
                // Hy3 (hy_v3) nests rope_theta under `rope_parameters` instead of top-level.
                raw.get("rope_parameters")
                    .and_then(|p| p.get("rope_theta"))
                    .and_then(Value::as_f64)
                    .map(|v| v as f32)
            })
            .unwrap_or(1_000_000.0),
        rope_scaling: raw.get("rope_scaling").filter(|v| !v.is_null()).cloned(),
        attention_bias: bool_field(&raw, "attention_bias").unwrap_or(false),
        tie_word_embeddings: bool_field(&raw, "tie_word_embeddings").unwrap_or(true),
        first_k_dense_replace: u32_field(&raw, "first_k_dense_replace").unwrap_or(0),
        moe_layer_freq: u32_field(&raw, "moe_layer_freq").unwrap_or(1),
        n_routed_experts: u32_field(&raw, "n_routed_experts")
            .or_else(|| u32_field(&raw, "num_experts")),
        n_shared_experts: u32_field(&raw, "n_shared_experts"),
        num_experts_per_tok: u32_field(&raw, "num_experts_per_tok"),
        decoder_sparse_step: u32_field(&raw, "decoder_sparse_step").unwrap_or(1),
        mlp_only_layers: u32_list_field(&raw, "mlp_only_layers"),
        shared_expert_intermediate_size: u32_field(&raw, "shared_expert_intermediate_size"),
        n_group: u32_field(&raw, "n_group").unwrap_or(1),
        topk_group: u32_field(&raw, "topk_group").unwrap_or(1),
        norm_topk_prob: bool_field(&raw, "norm_topk_prob").unwrap_or(true),
        routed_scaling_factor: f32_field(&raw, "routed_scaling_factor")
            .or_else(|| f32_field(&raw, "router_scaling_factor")) // Hy3 (hy_v3) key
            .unwrap_or(1.0),
        num_hash_layers: u32_field(&raw, "num_hash_layers").unwrap_or(0),
        topk_method: str_field(&raw, "topk_method"),
        scoring_func: str_field(&raw, "scoring_func"),
        hc_mult: u32_field(&raw, "hc_mult").unwrap_or(4),
        hc_sinkhorn_iters: u32_field(&raw, "hc_sinkhorn_iters").unwrap_or(20),
        hc_eps: f32_field(&raw, "hc_eps").unwrap_or(1e-6),
        quantization: parse_quantization(&raw),
        eos_token_ids: token_ids(&raw, "eos_token_id"),
        pad_token_id: u32_field(&raw, "pad_token_id"),
        raw,
    })
}

pub fn read_generation_max_tokens(path: impl AsRef<Path>) -> Result<Option<u32>> {
    let path = path.as_ref().join("generation_config.json");
    if !path.exists() {
        return Ok(None);
    }
    let value = read_json(&path)?;
    Ok(value
        .get("max_new_tokens")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok()))
}

pub fn detect_family(model_type: &str, config: &Value) -> Option<ModelFamily> {
    let model_type = model_type.to_ascii_lowercase();
    let architectures = architecture_strings(config);
    let haystack = std::iter::once(model_type.as_str())
        .chain(architectures.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ");

    if matches!(
        model_type.as_str(),
        "qwen3" | "qwen3_moe" | "qwen3_next" | "qwen3_vl" | "qwen3_vl_moe"
    ) || haystack.contains("qwen3")
    {
        return Some(ModelFamily::Qwen3);
    }
    if matches!(model_type.as_str(), "qwen2" | "qwen2_moe" | "qwen2_5_vl")
        || haystack.contains("qwen2")
    {
        return Some(ModelFamily::Qwen2);
    }
    if model_type.starts_with("deepseek")
        || haystack.contains("deepseek")
        || matches!(
            model_type.as_str(),
            "deepseek_v2" | "deepseek_v3" | "deepseek_v31" | "deepseek_v32" | "deepseek_v4"
        )
        // GLM-5.2 (`glm_moe_dsa`) is DeepSeek-V3.2 architecturally: MLA + DeepSeek Sparse Attention
        // (the lightning indexer) + sigmoid/noaux MoE. Route it through the DeepSeek path.
        || matches!(model_type.as_str(), "glm_moe_dsa" | "glm_moe_dsa_mtp")
        || haystack.contains("glm_moe_dsa")
        // Kimi K2/K2.5/K2.7 are a thin DeepSeek-V3 wrapper (MLA + sigmoid/noaux MoE, with a nested
        // `language_model.` weight prefix that load_arrays already strips). Route through DeepSeek.
        || model_type.starts_with("kimi_k2")
        || haystack.contains("kimi_k2")
    {
        return Some(ModelFamily::DeepSeek);
    }
    if model_type.starts_with("glm4")
        || haystack.contains("glm4")
        || haystack.contains("glm-4")
        || matches!(
            model_type.as_str(),
            "glm4" | "glm4_moe" | "glm4_moe_lite" | "glm4v" | "glm4v_moe"
        )
    {
        return Some(ModelFamily::GlmFlash);
    }
    if matches!(model_type.as_str(), "hy_v3" | "hyv3")
        || haystack.contains("hy_v3")
        || haystack.contains("hyv3")
        || haystack.contains("hunyuan")
    {
        return Some(ModelFamily::Hy3);
    }
    // Nemotron-H: Mamba2 + attention + MLP/MoE hybrid (NVIDIA Nemotron-3 Nano/Ultra, TwoTower).
    if model_type.starts_with("nemotron_h")
        || haystack.contains("nemotron_h")
        || haystack.contains("nemotronh")
    {
        return Some(ModelFamily::NemotronH);
    }
    // Gemma-4 only (older gemma/gemma2/gemma3 remain unsupported); routed via the Gemma family and
    // dispatched to Gemma4TextLike in load_model.
    if model_type.starts_with("gemma4") || haystack.contains("gemma4") {
        return Some(ModelFamily::Gemma);
    }
    None
}

pub fn supported_model_families() -> &'static str {
    "qwen2/qwen2_moe, qwen3/qwen3_moe/qwen3_next/qwen3_5/qwen3_5_moe, deepseek_v2/deepseek_v3/deepseek_v32/deepseek_v4, glm_moe_dsa (GLM-5.2), kimi_k2* (Kimi K2), glm4/glm4_moe/glm4_moe_lite Flash, hy_v3 (Hunyuan-3)"
}

pub fn architecture_strings(config: &Value) -> Vec<String> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(|item| item.to_ascii_lowercase())
                .collect()
        })
        .unwrap_or_default()
}

fn parse_quantization(raw: &Value) -> QuantizationConfig {
    let Some(value) = raw
        .get("quantization_config")
        .or_else(|| raw.get("quantization"))
    else {
        return QuantizationConfig::Dense;
    };
    let Some(object) = value.as_object() else {
        return QuantizationConfig::Dense;
    };
    let default = parse_quantization_spec(value);
    let mut entries = BTreeMap::new();
    for (key, value) in object {
        if matches!(key.as_str(), "bits" | "group_size" | "mode") {
            continue;
        }
        if let Some(spec) = parse_quantization_spec(value) {
            entries.insert(key.clone(), spec);
        }
    }
    let mut modes = entries
        .values()
        .map(|spec| spec.mode.as_str().to_ascii_lowercase())
        .collect::<Vec<_>>();
    if let Some(default) = &default {
        modes.push(default.mode.as_str().to_ascii_lowercase());
    }
    modes.sort();
    modes.dedup();

    if !entries.is_empty() {
        return QuantizationConfig::Mixed {
            default,
            entries,
            modes,
        };
    }

    default.map_or(QuantizationConfig::Dense, QuantizationConfig::Mlx)
}

fn parse_quantization_spec(value: &Value) -> Option<QuantizationSpec> {
    let object = value.as_object()?;
    let bits = object
        .get("bits")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())?;
    let group_size = object
        .get("group_size")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())?;
    let mode = match object
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("affine")
        .to_ascii_lowercase()
        .as_str()
    {
        "affine" => QuantizationMode::Affine,
        other => QuantizationMode::Other(other.to_string()),
    };
    Some(QuantizationSpec {
        bits,
        group_size,
        mode,
    })
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn required_u32(value: &Value, key: &str) -> Result<u32> {
    u32_field(value, key).ok_or_else(|| anyhow!("config.json missing required numeric field {key}"))
}

fn u32_field(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
}

fn u32_list_field(value: &Value, key: &str) -> Vec<u32> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_u64)
                .filter_map(|n| u32::try_from(n).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn f32_field(value: &Value, key: &str) -> Option<f32> {
    value.get(key).and_then(Value::as_f64).map(|n| n as f32)
}

fn bool_field(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn str_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn token_ids(value: &Value, key: &str) -> Vec<u32> {
    match value.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_u64)
            .filter_map(|n| u32::try_from(n).ok())
            .collect(),
        Some(Value::Number(number)) => number
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parses_qwen3_config_shape() {
        let config = parse_model_config(
            Path::new("/tmp/qwen"),
            json!({
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "quantization": {"group_size": 64, "bits": 4},
                "eos_token_id": 151645
            }),
        )
        .unwrap();

        assert_eq!(config.family, ModelFamily::Qwen3);
        assert_eq!(config.attention_head_dim(), 128);
        assert_eq!(config.quantization.standard_mlx().unwrap(), Some((4, 64)));
        assert_eq!(config.eos_token_ids, vec![151645]);
    }

    #[test]
    fn routes_deepseek_and_glm_flash_automatically() {
        let deepseek = parse_model_config(
            Path::new("/tmp/deepseek"),
            json!({
                "architectures": ["DeepseekV32ForCausalLM"],
                "model_type": "deepseek_v32",
                "hidden_size": 7168,
                "num_hidden_layers": 61,
                "num_attention_heads": 128,
                "num_key_value_heads": 128,
                "qk_rope_head_dim": 64,
                "qk_nope_head_dim": 128,
                "v_head_dim": 128,
                "kv_lora_rank": 512,
                "q_lora_rank": 1536,
                "vocab_size": 129280,
                "n_routed_experts": 256,
                "num_experts_per_tok": 8
            }),
        )
        .unwrap();
        let glm = parse_model_config(
            Path::new("/tmp/glm"),
            json!({
                "architectures": ["Glm4MoeLiteForCausalLM"],
                "model_type": "glm4_moe_lite",
                "hidden_size": 2048,
                "num_hidden_layers": 47,
                "num_attention_heads": 20,
                "num_key_value_heads": 20,
                "qk_rope_head_dim": 64,
                "qk_nope_head_dim": 192,
                "v_head_dim": 256,
                "kv_lora_rank": 512,
                "q_lora_rank": 768,
                "vocab_size": 154880,
                "n_routed_experts": 64,
                "num_experts_per_tok": 4
            }),
        )
        .unwrap();

        assert_eq!(deepseek.family, ModelFamily::DeepSeek);
        assert_eq!(glm.family, ModelFamily::GlmFlash);
        assert!(deepseek.is_moe_layer(3));
        assert!(glm.is_moe_layer(1));
    }

    #[test]
    fn parses_deepseek_v32_indexer_config() {
        let config = parse_model_config(
            Path::new("/tmp/deepseek-v32"),
            json!({
                "architectures": ["DeepseekV32ForCausalLM"],
                "model_type": "deepseek_v32",
                "hidden_size": 7168,
                "num_hidden_layers": 61,
                "num_attention_heads": 128,
                "num_key_value_heads": 128,
                "qk_rope_head_dim": 64,
                "qk_nope_head_dim": 128,
                "v_head_dim": 128,
                "kv_lora_rank": 512,
                "q_lora_rank": 1536,
                "index_head_dim": 128,
                "index_n_heads": 64,
                "index_topk": 2048,
                "indexer_rope_interleave": false,
                "vocab_size": 129280
            }),
        )
        .unwrap();

        assert_eq!(config.family, ModelFamily::DeepSeek);
        assert_eq!(config.index_head_dim, Some(128));
        assert_eq!(config.index_n_heads, Some(64));
        assert_eq!(config.index_topk, Some(2048));
        assert!(!config.indexer_rope_interleave);
    }

    #[test]
    fn parses_qwen3_moe_routing_config() {
        let config = parse_model_config(
            Path::new("/tmp/qwen3-moe"),
            json!({
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "moe_intermediate_size": 768,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "num_experts": 64,
                "num_experts_per_tok": 8,
                "decoder_sparse_step": 2,
                "mlp_only_layers": [0],
                "norm_topk_prob": true,
                "vocab_size": 151936
            }),
        )
        .unwrap();

        assert_eq!(config.family, ModelFamily::Qwen3);
        assert_eq!(config.n_routed_experts, Some(64));
        assert!(!config.is_qwen_moe_layer(0));
        assert!(config.is_qwen_moe_layer(1));
        assert!(!config.is_qwen_moe_layer(2));
        assert!(config.is_qwen_moe_layer(3));
    }

    #[test]
    fn parses_deepseek_v4_flash_config_aliases() {
        let config = parse_model_config(
            Path::new("/tmp/deepseek-v4-flash"),
            json!({
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "hidden_size": 5120,
                "num_hidden_layers": 43,
                "num_attention_heads": 64,
                "num_key_value_heads": 1,
                "qk_rope_head_dim": 64,
                "qk_nope_head_dim": 128,
                "v_head_dim": 128,
                "kv_lora_rank": 512,
                "q_lora_rank": 1536,
                "index_head_dim": 128,
                "index_n_heads": 64,
                "index_topk": 512,
                "compress_ratios": [0, 0, 4, 128],
                "compress_rope_theta": 160000,
                "sliding_window": 128,
                "o_lora_rank": 1024,
                "o_groups": 8,
                "swiglu_limit": 10.0,
                "num_hash_layers": 3,
                "hc_mult": 4,
                "hc_sinkhorn_iters": 12,
                "hc_eps": 1e-5,
                "scoring_func": "sqrtsoftplus",
                "vocab_size": 129280
            }),
        )
        .unwrap();

        assert_eq!(config.family, ModelFamily::DeepSeek);
        assert_eq!(config.index_topk, Some(512));
        assert_eq!(config.compress_ratios, vec![0, 0, 4, 128]);
        assert_eq!(config.compress_rope_theta, 160000.0);
        assert_eq!(config.sliding_window, Some(128));
        assert_eq!(config.o_lora_rank, Some(1024));
        assert_eq!(config.o_groups, Some(8));
        assert_eq!(config.swiglu_limit, Some(10.0));
        assert_eq!(config.num_hash_layers, 3);
        assert_eq!(config.hc_mult, 4);
        assert_eq!(config.hc_sinkhorn_iters, 12);
        assert_eq!(config.hc_eps, 1e-5);
        assert_eq!(config.scoring_func.as_deref(), Some("sqrtsoftplus"));
    }

    #[test]
    fn parses_window_size_as_sliding_window_alias() {
        let config = parse_model_config(
            Path::new("/tmp/deepseek-v4-window"),
            json!({
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "hidden_size": 512,
                "num_hidden_layers": 1,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "window_size": 128,
                "vocab_size": 32000
            }),
        )
        .unwrap();

        assert_eq!(config.sliding_window, Some(128));
    }

    #[test]
    fn parses_mixed_quantization_modes_by_prefix() {
        let config = parse_model_config(
            Path::new("/tmp/v4"),
            json!({
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "hidden_size": 4096,
                "num_hidden_layers": 43,
                "num_attention_heads": 64,
                "num_key_value_heads": 1,
                "vocab_size": 129280,
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "model.layers.0.ffn.switch_mlp.gate_proj": {
                        "group_size": 32,
                        "bits": 4,
                        "mode": "mxfp4"
                    },
                    "model.layers.0.attn.wq_a": {
                        "group_size": 64,
                        "bits": 4,
                        "mode": "affine"
                    },
                    "model.layers.0.unsupported": {
                        "group_size": 16,
                        "bits": 4,
                        "mode": "dynamic"
                    }
                }
            }),
        )
        .unwrap();

        assert_eq!(
            config
                .quantization
                .standard_mlx_for("model.layers.0.attn.wq_a")
                .unwrap(),
            Some((4, 64))
        );
        let spec = config
            .quantization
            .mlx_quantization_for("model.layers.0.ffn.switch_mlp.gate_proj")
            .unwrap()
            .unwrap();
        assert_eq!(spec.bits, 4);
        assert_eq!(spec.group_size, 32);
        assert_eq!(spec.mode.as_str(), "mxfp4");
        let err = config.quantization.validate_supported().unwrap_err();
        assert!(err.to_string().contains("model.layers.0.unsupported"));
        assert!(err.to_string().contains("dynamic"));
    }
}
