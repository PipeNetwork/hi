use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::MlxModelConfig;
use crate::manifest::WeightShard;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightCatalog {
    pub root: PathBuf,
    pub tensors: BTreeSet<String>,
    pub weight_map: BTreeMap<String, String>,
    pub shards: Vec<WeightShard>,
    pub estimated_bytes: u64,
    pub quantization: WeightQuantization,
}

impl WeightCatalog {
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let shards = find_weight_shards(root)?;
        if shards.is_empty() {
            bail!(
                "bad model path {}: no .safetensors weight shards found",
                root.display()
            );
        }
        let weight_map = read_index_weight_map(root)?;
        let tensors = if weight_map.is_empty() {
            shard_tensor_names(root, &shards)?
        } else {
            weight_map.keys().cloned().collect()
        };
        let estimated_bytes = shards.iter().map(|shard| shard.bytes).sum();
        let quantization = WeightQuantization::from_tensors(&tensors);
        Ok(Self {
            root: root.to_path_buf(),
            tensors,
            weight_map,
            shards,
            estimated_bytes,
            quantization,
        })
    }

    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains(key)
    }

    pub fn shard_path_for(&self, key: &str) -> Option<PathBuf> {
        self.weight_map.get(key).map(|file| self.root.join(file))
    }

    pub fn linear(&self, prefix: &str) -> Option<LinearWeightKind> {
        let weight = format!("{prefix}.weight");
        if !self.has(&weight) {
            return None;
        }
        let scales = format!("{prefix}.scales");
        if self.has(&scales) {
            Some(LinearWeightKind::Quantized {
                prefix: prefix.to_string(),
            })
        } else {
            Some(LinearWeightKind::Dense {
                prefix: prefix.to_string(),
                bias: self.has(&format!("{prefix}.bias")),
            })
        }
    }

    pub fn validate_for_config(&self, config: &MlxModelConfig) -> Result<()> {
        config.quantization.validate_supported()?;
        // Nemotron-H uses `backbone.` naming (embeddings/norm_f) rather than `model.`; LongCat-2.0's
        // input embedding is the n-gram `ngram_embeddings.word_embeddings` (no `embed_tokens`).
        let required: &[&str] = match config.family {
            crate::manifest::ModelFamily::NemotronH => {
                &["backbone.embeddings.weight", "backbone.norm_f.weight"]
            }
            crate::manifest::ModelFamily::LongCat => &[
                "model.ngram_embeddings.word_embeddings.weight",
                "model.norm.weight",
            ],
            _ => &["model.embed_tokens.weight", "model.norm.weight"],
        };
        for key in required {
            // VL models (Qwen3.5) still carry the `language_model.` prefix at validation time.
            if !self.has(key) && !self.has(&format!("language_model.{key}")) {
                bail!(
                    "bad model path {}: missing required tensor {key}",
                    self.root.display()
                );
            }
        }
        match config.family {
            crate::manifest::ModelFamily::Qwen2
            | crate::manifest::ModelFamily::Qwen3
            | crate::manifest::ModelFamily::Hy3 => {
                if config.linear_num_value_heads.is_some() {
                    // Qwen3.5 gated-delta-net hybrid: layer 0 is a linear-attn (SSM) layer, and the
                    // weights may still carry the VL `language_model.` prefix at this point.
                    self.require_any(
                        "Qwen3.5 linear-attn projection",
                        &[
                            "model.layers.0.linear_attn.conv1d.weight",
                            "language_model.model.layers.0.linear_attn.conv1d.weight",
                        ],
                    )?;
                } else {
                    self.require_any(
                        "qwen attention projection",
                        &[
                            "model.layers.0.self_attn.q_proj.weight",
                            "model.layers.0.self_attn.q_proj.scales",
                        ],
                    )?;
                }
            }
            crate::manifest::ModelFamily::DeepSeek | crate::manifest::ModelFamily::GlmFlash => {
                if config.is_deepseek_v4()
                    || self.has("model.layers.0.attn.wkv.weight")
                    || self.has("model.layers.0.attn.wkv.scales")
                {
                    self.require_any(
                        "DeepSeek V4 attention projection",
                        &[
                            "model.layers.0.attn.wkv.weight",
                            "model.layers.0.attn.wkv.scales",
                        ],
                    )?;
                    self.require_any(
                        "DeepSeek V4 attention norm",
                        &["model.layers.0.attn_norm.weight"],
                    )?;
                    if config.is_deepseek_v4() {
                        self.validate_deepseek_v4_compressed_attention(config)?;
                    }
                } else if self.has("model.layers.0.self_attn.q_proj.weight")
                    || self.has("model.layers.0.self_attn.q_proj.scales")
                {
                    // Standard GQA GLM-4 (Glm4Like), not MLA.
                    self.require_any(
                        "GLM-4 attention projection",
                        &[
                            "model.layers.0.self_attn.q_proj.weight",
                            "model.layers.0.self_attn.q_proj.scales",
                        ],
                    )?;
                } else {
                    self.require_any(
                        "MLA attention projection",
                        &[
                            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                            "model.layers.0.self_attn.kv_a_proj_with_mqa.scales",
                        ],
                    )?;
                }
            }
            crate::manifest::ModelFamily::NemotronH => {
                self.require_any(
                    "Nemotron-H mixer projection",
                    &[
                        "backbone.layers.0.mixer.in_proj.weight",
                        "backbone.layers.0.mixer.in_proj.scales",
                    ],
                )?;
            }
            crate::manifest::ModelFamily::MiniMax => {
                self.require_any(
                    "MiniMax attention projection",
                    &[
                        "model.layers.0.self_attn.q_proj.weight",
                        "model.layers.0.self_attn.q_proj.scales",
                    ],
                )?;
            }
            crate::manifest::ModelFamily::LongCat => {
                self.require_any(
                    "LongCat-2.0 MLA projection",
                    &[
                        "model.layers.0.self_attn.0.kv_a_proj_with_mqa.weight",
                        "model.layers.0.self_attn.0.kv_a_proj_with_mqa.scales",
                    ],
                )?;
            }
            crate::manifest::ModelFamily::Gemma if config.model_type.starts_with("gemma") => {
                self.require_any(
                    "Gemma-3/4 attention projection",
                    &[
                        "model.layers.0.self_attn.q_proj.weight",
                        "model.layers.0.self_attn.q_proj.scales",
                        "language_model.model.layers.0.self_attn.q_proj.weight",
                        "language_model.model.layers.0.self_attn.q_proj.scales",
                    ],
                )?;
            }
            crate::manifest::ModelFamily::Laguna => {
                // Laguna gates attention output through a per-head g_proj; its absence means the
                // checkpoint is not the Laguna layout even if the config claims it.
                self.require_any(
                    "Laguna attention gate projection",
                    &[
                        "model.layers.0.self_attn.g_proj.weight",
                        "model.layers.0.self_attn.g_proj.scales",
                    ],
                )?;
            }
            crate::manifest::ModelFamily::Llama
            | crate::manifest::ModelFamily::Mistral
            | crate::manifest::ModelFamily::Mixtral
            | crate::manifest::ModelFamily::Gemma
            | crate::manifest::ModelFamily::Phi => {
                bail!(
                    "{} MLX weights are not supported by hi-mlx yet; use --backend cuda",
                    config.family.label()
                )
            }
        }
        if self.quantization.has_weight_shape_packed {
            bail!(
                "unsupported quantization: packed weight_shape/weight_packed tensors need a remap that is not implemented in hi-mlx yet"
            );
        }
        Ok(())
    }

    fn validate_deepseek_v4_compressed_attention(&self, config: &MlxModelConfig) -> Result<()> {
        for layer in 0..config.num_hidden_layers {
            let ratio = config
                .compress_ratios
                .get(layer as usize)
                .copied()
                .unwrap_or(0);
            if ratio == 0 {
                continue;
            }
            let prefix = format!("model.layers.{layer}.attn");
            self.require_v4_compressor(&format!("{prefix}.compressor"))?;
            if ratio == 4 {
                self.require_v4_compressor(&format!("{prefix}.indexer.compressor"))?;
                self.require_any(
                    "DeepSeek V4 CSA indexer query projection",
                    &[&format!("{prefix}.indexer.wq_b.weight")],
                )?;
                self.require_any(
                    "DeepSeek V4 CSA indexer weights projection",
                    &[&format!("{prefix}.indexer.weights_proj.weight")],
                )?;
            }
        }
        Ok(())
    }

    fn require_v4_compressor(&self, prefix: &str) -> Result<()> {
        self.require_any(
            "DeepSeek V4 compressor positional bias",
            &[&format!("{prefix}.ape")],
        )?;
        self.require_any(
            "DeepSeek V4 compressor norm",
            &[&format!("{prefix}.norm.weight")],
        )?;
        self.require_any(
            "DeepSeek V4 compressor gate projection",
            &[&format!("{prefix}.wgate.weight")],
        )?;
        self.require_any(
            "DeepSeek V4 compressor KV projection",
            &[&format!("{prefix}.wkv.weight")],
        )
    }

    fn require_any(&self, label: &str, keys: &[&str]) -> Result<()> {
        // VL models (e.g. Llama-4, Qwen3.5) carry the `language_model.` prefix at validation time.
        if keys
            .iter()
            .any(|key| self.has(key) || self.has(&format!("language_model.{key}")))
        {
            Ok(())
        } else {
            bail!(
                "bad model path {}: missing required {label}; looked for {}",
                self.root.display(),
                keys.join(", ")
            )
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearWeightKind {
    Dense { prefix: String, bias: bool },
    Quantized { prefix: String },
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightQuantization {
    pub dense_weights: usize,
    pub quantized_weights: usize,
    pub has_weight_shape_packed: bool,
    pub has_fp8_scale_inv: bool,
}

impl WeightQuantization {
    fn from_tensors(tensors: &BTreeSet<String>) -> Self {
        let mut quantized_weights = 0;
        let mut dense_weights = 0;
        for key in tensors {
            if let Some(prefix) = key.strip_suffix(".weight") {
                if tensors.contains(&format!("{prefix}.scales")) {
                    quantized_weights += 1;
                } else {
                    dense_weights += 1;
                }
            }
        }
        Self {
            dense_weights,
            quantized_weights,
            has_weight_shape_packed: tensors.iter().any(|key| {
                key.ends_with("weight_shape")
                    || key.ends_with("weight_packed")
                    || key.ends_with("weight_scale")
            }),
            has_fp8_scale_inv: tensors.iter().any(|key| key.ends_with("weight_scale_inv")),
        }
    }

    pub fn label(&self) -> &'static str {
        if self.quantized_weights > 0 && self.dense_weights > 0 {
            "mixed-dense-mlx"
        } else if self.quantized_weights > 0 {
            "mlx-quantized"
        } else {
            "dense"
        }
    }
}

fn read_index_weight_map(root: &Path) -> Result<BTreeMap<String, String>> {
    let path = root.join("model.safetensors.index.json");
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let Some(map) = value.get("weight_map").and_then(Value::as_object) else {
        bail!(
            "bad safetensors index {}: missing weight_map",
            path.display()
        );
    };
    Ok(map
        .iter()
        .filter_map(|(key, value)| value.as_str().map(|file| (key.clone(), file.to_string())))
        .collect())
}

fn shard_tensor_names(root: &Path, shards: &[WeightShard]) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    for shard in shards {
        let path = root.join(&shard.path);
        let header = safetensors_header(&path)?;
        let object = header
            .as_object()
            .ok_or_else(|| anyhow!("safetensors header is not an object"))?;
        out.extend(
            object
                .keys()
                .filter(|key| key.as_str() != "__metadata__")
                .cloned(),
        );
    }
    Ok(out)
}

pub fn find_weight_shards(path: &Path) -> Result<Vec<WeightShard>> {
    let mut shards = Vec::new();
    for entry in fs::read_dir(path).with_context(|| format!("reading {}", path.display()))? {
        let entry = entry?;
        let shard_path = entry.path();
        if shard_path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
            continue;
        }
        let meta = entry.metadata()?;
        let rel = shard_path
            .strip_prefix(path)
            .unwrap_or(&shard_path)
            .to_string_lossy()
            .to_string();
        shards.push(WeightShard {
            path: rel,
            bytes: meta.len(),
            tensor_count: safetensors_tensor_count(&shard_path).ok(),
        });
    }
    shards.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(shards)
}

pub fn safetensors_tensor_count(path: &Path) -> Result<usize> {
    let header = safetensors_header(path)?;
    let object = header
        .as_object()
        .ok_or_else(|| anyhow!("safetensors header is not an object"))?;
    Ok(object.keys().filter(|key| *key != "__metadata__").count())
}

fn safetensors_header(path: &Path) -> Result<Value> {
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)?;
    let header_len = u64::from_le_bytes(len);
    let header_len = usize::try_from(header_len).context("safetensors header too large")?;
    let mut header = vec![0; header_len];
    file.read_exact(&mut header)?;
    serde_json::from_slice(&header)
        .with_context(|| format!("parsing safetensors header {}", path.display()))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub mod mlx {
    use std::collections::HashMap;

    use anyhow::{Context, Result, bail};
    use mlx_rs::Array;

    use super::WeightCatalog;

    pub fn load_arrays(
        catalog: &WeightCatalog,
        skip_tensors: Option<&std::collections::BTreeSet<String>>,
    ) -> Result<HashMap<String, Array>> {
        let mut arrays = HashMap::new();
        for shard in &catalog.shards {
            let path = catalog.root.join(&shard.path);
            let loaded = Array::load_safetensors(&path)
                .with_context(|| format!("loading safetensors shard {}", path.display()))?;
            for (key, value) in loaded {
                // VL models (e.g. Qwen3.5) nest the language model under `language_model.`; strip it
                // and drop the vision tower / MTP heads we don't run.
                let key = key
                    .strip_prefix("language_model.")
                    .map(str::to_string)
                    .unwrap_or(key);
                if key.starts_with("visual.")
                    || key.starts_with("vision_")
                    || key.contains(".mtp.")
                    || key.starts_with("mtp.")
                {
                    continue;
                }
                // When expert streaming is enabled, skip loading the routed-expert
                // tensors — they'll be fetched on demand from the pool instead.
                if let Some(skip) = skip_tensors {
                    if skip.contains(&key) {
                        continue;
                    }
                }
                arrays.insert(key, value);
            }
        }
        if arrays.is_empty() {
            bail!("no tensors loaded from {}", catalog.root.display());
        }
        Ok(arrays)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use super::*;
    use crate::config::parse_model_config;

    #[test]
    fn catalog_reads_safetensors_index_and_classifies_quantized() {
        let dir = tempfile_path("catalog-index");
        fs::create_dir_all(&dir).unwrap();
        write_minimal_safetensors(&dir.join("model.safetensors"));
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
              "metadata": {"total_size": 1},
              "weight_map": {
                "model.embed_tokens.weight": "model.safetensors",
                "model.embed_tokens.scales": "model.safetensors",
                "model.embed_tokens.biases": "model.safetensors",
                "model.norm.weight": "model.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                "model.layers.0.self_attn.q_proj.scales": "model.safetensors",
                "model.layers.0.self_attn.q_proj.biases": "model.safetensors"
              }
            }"#,
        )
        .unwrap();

        let catalog = WeightCatalog::load(&dir).unwrap();

        assert!(catalog.has("model.embed_tokens.weight"));
        assert_eq!(
            catalog.linear("model.layers.0.self_attn.q_proj"),
            Some(LinearWeightKind::Quantized {
                prefix: "model.layers.0.self_attn.q_proj".to_string()
            })
        );
        assert_eq!(catalog.quantization.quantized_weights, 2);
    }

    #[test]
    fn catalog_reads_raw_safetensors_headers_without_index() {
        let dir = tempfile_path("catalog-header");
        fs::create_dir_all(&dir).unwrap();
        write_named_safetensors(
            &dir.join("model.safetensors"),
            &[
                "model.embed_tokens.weight",
                "model.norm.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ],
        );

        let catalog = WeightCatalog::load(&dir).unwrap();

        assert!(catalog.has("model.layers.0.self_attn.q_proj.weight"));
        assert_eq!(catalog.shards[0].tensor_count, Some(3));
    }

    #[test]
    fn catalog_classifies_mxfp4_weights_without_biases() {
        let dir = tempfile_path("catalog-mxfp4");
        fs::create_dir_all(&dir).unwrap();
        write_minimal_safetensors(&dir.join("model.safetensors"));
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
              "metadata": {"total_size": 1},
              "weight_map": {
                "model.embed_tokens.weight": "model.safetensors",
                "model.norm.weight": "model.safetensors",
                "model.layers.0.ffn.switch_mlp.gate_proj.weight": "model.safetensors",
                "model.layers.0.ffn.switch_mlp.gate_proj.scales": "model.safetensors"
              }
            }"#,
        )
        .unwrap();

        let catalog = WeightCatalog::load(&dir).unwrap();

        assert_eq!(
            catalog.linear("model.layers.0.ffn.switch_mlp.gate_proj"),
            Some(LinearWeightKind::Quantized {
                prefix: "model.layers.0.ffn.switch_mlp.gate_proj".to_string()
            })
        );
        assert_eq!(catalog.quantization.quantized_weights, 1);
    }

    #[test]
    fn catalog_validates_deepseek_v4_layout() {
        let dir = tempfile_path("catalog-v4");
        fs::create_dir_all(&dir).unwrap();
        write_minimal_safetensors(&dir.join("model.safetensors"));
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
              "metadata": {"total_size": 1},
              "weight_map": {
                "model.embed_tokens.weight": "model.safetensors",
                "model.norm.weight": "model.safetensors",
                "model.layers.0.attn.wkv.weight": "model.safetensors",
                "model.layers.0.attn.wkv.scales": "model.safetensors",
                "model.layers.0.attn_norm.weight": "model.safetensors"
              }
            }"#,
        )
        .unwrap();
        let config = parse_model_config(
            &dir,
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
                    }
                }
            }),
        )
        .unwrap();

        let catalog = WeightCatalog::load(&dir).unwrap();

        catalog.validate_for_config(&config).unwrap();
    }

    #[test]
    fn catalog_validates_deepseek_v4_compressor_and_indexer_keys() {
        let dir = tempfile_path("catalog-v4-compressed");
        fs::create_dir_all(&dir).unwrap();
        write_minimal_safetensors(&dir.join("model.safetensors"));
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
              "metadata": {"total_size": 1},
              "weight_map": {
                "model.embed_tokens.weight": "model.safetensors",
                "model.norm.weight": "model.safetensors",
                "model.layers.0.attn.wkv.weight": "model.safetensors",
                "model.layers.0.attn_norm.weight": "model.safetensors",
                "model.layers.0.attn.compressor.ape": "model.safetensors",
                "model.layers.0.attn.compressor.norm.weight": "model.safetensors",
                "model.layers.0.attn.compressor.wgate.weight": "model.safetensors",
                "model.layers.0.attn.compressor.wkv.weight": "model.safetensors",
                "model.layers.0.attn.indexer.compressor.ape": "model.safetensors",
                "model.layers.0.attn.indexer.compressor.norm.weight": "model.safetensors",
                "model.layers.0.attn.indexer.compressor.wgate.weight": "model.safetensors",
                "model.layers.0.attn.indexer.compressor.wkv.weight": "model.safetensors",
                "model.layers.0.attn.indexer.wq_b.weight": "model.safetensors",
                "model.layers.0.attn.indexer.weights_proj.weight": "model.safetensors",
                "model.layers.1.attn.compressor.ape": "model.safetensors",
                "model.layers.1.attn.compressor.norm.weight": "model.safetensors",
                "model.layers.1.attn.compressor.wgate.weight": "model.safetensors",
                "model.layers.1.attn.compressor.wkv.weight": "model.safetensors"
              }
            }"#,
        )
        .unwrap();
        let config = parse_model_config(
            &dir,
            json!({
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "hidden_size": 4096,
                "num_hidden_layers": 2,
                "num_attention_heads": 64,
                "num_key_value_heads": 1,
                "compress_ratios": [4, 128],
                "vocab_size": 129280
            }),
        )
        .unwrap();

        let catalog = WeightCatalog::load(&dir).unwrap();

        catalog.validate_for_config(&config).unwrap();
    }

    #[test]
    fn catalog_rejects_incomplete_deepseek_v4_csa_indexer() {
        let dir = tempfile_path("catalog-v4-missing-indexer");
        fs::create_dir_all(&dir).unwrap();
        write_minimal_safetensors(&dir.join("model.safetensors"));
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
              "metadata": {"total_size": 1},
              "weight_map": {
                "model.embed_tokens.weight": "model.safetensors",
                "model.norm.weight": "model.safetensors",
                "model.layers.0.attn.wkv.weight": "model.safetensors",
                "model.layers.0.attn_norm.weight": "model.safetensors",
                "model.layers.0.attn.compressor.ape": "model.safetensors",
                "model.layers.0.attn.compressor.norm.weight": "model.safetensors",
                "model.layers.0.attn.compressor.wgate.weight": "model.safetensors",
                "model.layers.0.attn.compressor.wkv.weight": "model.safetensors"
              }
            }"#,
        )
        .unwrap();
        let config = parse_model_config(
            &dir,
            json!({
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "hidden_size": 4096,
                "num_hidden_layers": 1,
                "num_attention_heads": 64,
                "num_key_value_heads": 1,
                "compress_ratios": [4],
                "vocab_size": 129280
            }),
        )
        .unwrap();

        let catalog = WeightCatalog::load(&dir).unwrap();
        let err = catalog.validate_for_config(&config).unwrap_err();

        assert!(err.to_string().contains("indexer.compressor.ape"));
    }

    fn write_minimal_safetensors(path: &Path) {
        write_named_safetensors(path, &[]);
    }

    fn write_named_safetensors(path: &Path, names: &[&str]) {
        let mut entries = String::from(r#"{"__metadata__":{"format":"pt"}"#);
        for name in names {
            entries.push(',');
            entries.push('"');
            entries.push_str(name);
            entries.push_str(r#"":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}"#);
        }
        entries.push('}');
        let header = entries.as_bytes();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        fs::write(path, bytes).unwrap();
    }

    fn tempfile_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-mlx-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }
}
