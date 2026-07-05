use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::config::{MlxModelConfig, load_model_config};
use crate::generate::TokenizerRuntime;
use crate::manifest::{ModelInfo, inspect_model};
use crate::models::NativeRuntime;
use crate::weights::WeightCatalog;

#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}

#[derive(Clone, Debug)]
pub struct GenerationOutput {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    fn model(&self) -> &ModelInfo;

    fn health(&self) -> BackendHealth;

    async fn generate(&self, request: GenerationRequest) -> Result<GenerationOutput>;
}

pub type SharedBackend = Arc<dyn InferenceBackend>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendHealth {
    pub ready: bool,
    pub family: String,
    pub quantization: String,
    pub context_length: Option<u32>,
    pub memory_estimate_bytes: Option<u64>,
}

pub struct MlxBackend {
    model: ModelInfo,
    config: MlxModelConfig,
    weights: WeightCatalog,
    runtime: Mutex<NativeRuntime>,
}

impl MlxBackend {
    pub fn load(path: impl AsRef<std::path::Path>, model_id: Option<String>) -> Result<Self> {
        ensure_native_generation_available()?;
        let path = path.as_ref();
        let model = inspect_model(path, model_id)?;
        let config = load_model_config(path)?;
        let weights = WeightCatalog::load(path)?;
        weights.validate_for_config(&config)?;
        let tokenizer = TokenizerRuntime::load(path)?;
        let runtime = NativeRuntime::load(config.clone(), weights.clone(), tokenizer)?;
        Ok(Self {
            model,
            config,
            weights,
            runtime: Mutex::new(runtime),
        })
    }
}

#[async_trait]
impl InferenceBackend for MlxBackend {
    fn model(&self) -> &ModelInfo {
        &self.model
    }

    fn health(&self) -> BackendHealth {
        BackendHealth {
            ready: true,
            family: self.config.family.label().to_string(),
            quantization: self.config.quantization_label(),
            context_length: self.config.context_length,
            memory_estimate_bytes: Some(self.weights.estimated_bytes),
        }
    }

    async fn generate(&self, request: GenerationRequest) -> Result<GenerationOutput> {
        let mut runtime = self.runtime.lock().await;
        runtime.generate(request)
    }
}

pub type NativeBackend = MlxBackend;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn ensure_native_generation_available() -> Result<()> {
    Ok(())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn ensure_native_generation_available() -> Result<()> {
    anyhow::bail!("native MLX inference requires Apple Silicon macOS")
}

pub fn platform_supported() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

#[cfg(test)]
pub struct MockBackend {
    model: ModelInfo,
    output: tokio::sync::Mutex<String>,
    last_prompt: tokio::sync::Mutex<Option<String>>,
}

#[cfg(test)]
impl MockBackend {
    pub fn new(model: ModelInfo, output: impl Into<String>) -> Self {
        Self {
            model,
            output: tokio::sync::Mutex::new(output.into()),
            last_prompt: tokio::sync::Mutex::new(None),
        }
    }

    pub async fn last_prompt(&self) -> Option<String> {
        self.last_prompt.lock().await.clone()
    }
}

#[cfg(test)]
#[async_trait]
impl InferenceBackend for MockBackend {
    fn model(&self) -> &ModelInfo {
        &self.model
    }

    fn health(&self) -> BackendHealth {
        BackendHealth {
            ready: true,
            family: self.model.family.label().to_string(),
            quantization: "mock".to_string(),
            context_length: self.model.context_length,
            memory_estimate_bytes: Some(self.model.weight_shards.iter().map(|s| s.bytes).sum()),
        }
    }

    async fn generate(&self, request: GenerationRequest) -> Result<GenerationOutput> {
        *self.last_prompt.lock().await = Some(request.prompt.clone());
        let text = self.output.lock().await.clone();
        Ok(GenerationOutput {
            prompt_tokens: (request.prompt.len() / 4).max(1) as u64,
            completion_tokens: (text.len() / 4).max(1) as u64,
            text,
        })
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    static MLX_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_compressed_deepseek_v4_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-deepseek-v4-compressed");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{
              "architectures": ["DeepseekV4ForCausalLM"],
              "model_type": "deepseek_v4",
              "hidden_size": 4,
              "intermediate_size": 8,
              "moe_intermediate_size": 4,
              "num_hidden_layers": 1,
              "num_attention_heads": 1,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "qk_rope_head_dim": 2,
              "q_lora_rank": 4,
              "index_head_dim": 2,
              "index_n_heads": 1,
              "index_topk": 1,
              "o_lora_rank": 4,
              "o_groups": 1,
              "n_routed_experts": 2,
              "n_shared_experts": 0,
              "num_experts_per_tok": 1,
              "num_hash_layers": 0,
              "scoring_func": "sqrtsoftplus",
              "norm_topk_prob": true,
              "routed_scaling_factor": 1.0,
              "swiglu_limit": 0.0,
              "hc_mult": 1,
              "hc_sinkhorn_iters": 1,
              "hc_eps": 1e-6,
              "compress_ratios": [4],
              "compress_rope_theta": 160000,
              "sliding_window": 2,
              "vocab_size": 4,
              "max_position_embeddings": 16,
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000,
              "tie_word_embeddings": false,
              "eos_token_id": 99
            }"#,
        )
        .unwrap();
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend =
            super::MlxBackend::load(&dir, Some("tiny-v4-compressed".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 4,
                temperature: 0.0,
                top_p: 1.0,
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert_eq!(output.completion_tokens, 4);
        assert!(!output.text.trim().is_empty());

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            let vocab = [
                -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
            ];
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert(
                "lm_head.weight".to_string(),
                Array::from_slice(
                    &[
                        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                        0.0, 0.0,
                    ],
                    &[4, 4],
                ),
            );
            arrays.insert("model.hc_head.fn".to_string(), zeros(&[1, 4]));
            arrays.insert("model.hc_head.base".to_string(), zeros(&[1]));
            arrays.insert("model.hc_head.scale".to_string(), zeros(&[1]));
            arrays.insert("model.norm.weight".to_string(), ones(4));

            let prefix = "model.layers.0";
            arrays.insert(format!("{prefix}.attn_norm.weight"), ones(4));
            let attn = format!("{prefix}.attn");
            arrays.insert(format!("{attn}.wq_a.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.q_norm.weight"), ones(4));
            arrays.insert(format!("{attn}.wq_b.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.wkv.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.kv_norm.weight"), ones(4));
            arrays.insert(format!("{attn}.wo_a.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.wo_b.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.compressor.ape"), zeros(&[4, 8]));
            arrays.insert(format!("{attn}.compressor.norm.weight"), ones(4));
            arrays.insert(format!("{attn}.compressor.wgate.weight"), zeros(&[8, 4]));
            arrays.insert(format!("{attn}.compressor.wkv.weight"), zeros(&[8, 4]));
            arrays.insert(format!("{attn}.indexer.compressor.ape"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.indexer.compressor.norm.weight"), ones(2));
            arrays.insert(
                format!("{attn}.indexer.compressor.wgate.weight"),
                zeros(&[4, 4]),
            );
            arrays.insert(
                format!("{attn}.indexer.compressor.wkv.weight"),
                zeros(&[4, 4]),
            );
            arrays.insert(format!("{attn}.indexer.wq_b.weight"), zeros(&[2, 4]));
            arrays.insert(
                format!("{attn}.indexer.weights_proj.weight"),
                zeros(&[1, 4]),
            );
            arrays.insert(format!("{prefix}.attn_hc.fn"), zeros(&[3, 4]));
            arrays.insert(format!("{prefix}.attn_hc.base"), zeros(&[3]));
            arrays.insert(format!("{prefix}.attn_hc.scale"), zeros(&[3]));

            arrays.insert(format!("{prefix}.ffn_norm.weight"), ones(4));
            arrays.insert(format!("{prefix}.ffn.gate.weight"), zeros(&[2, 4]));
            for name in ["gate_proj", "up_proj", "down_proj"] {
                arrays.insert(
                    format!("{prefix}.ffn.switch_mlp.{name}.weight"),
                    zeros(&[2, 4, 4]),
                );
            }
            arrays.insert(format!("{prefix}.ffn_hc.fn"), zeros(&[3, 4]));
            arrays.insert(format!("{prefix}.ffn_hc.base"), zeros(&[3]));
            arrays.insert(format!("{prefix}.ffn_hc.scale"), zeros(&[3]));

            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
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

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_deepseek_v4_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-deepseek-v4");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir);
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend = super::MlxBackend::load(&dir, Some("tiny-deepseek-v4".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert_eq!(output.completion_tokens, 2);
        assert!(!output.text.trim().is_empty());

        fn write_config(root: &Path) {
            fs::write(
                root.join("config.json"),
                r#"{
                  "architectures": ["DeepseekV4ForCausalLM"],
                  "model_type": "deepseek_v4",
                  "hidden_size": 4,
                  "intermediate_size": 8,
                  "moe_intermediate_size": 4,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "qk_rope_head_dim": 0,
                  "q_lora_rank": 4,
                  "o_lora_rank": 4,
                  "o_groups": 1,
                  "n_routed_experts": 2,
                  "n_shared_experts": 0,
                  "num_experts_per_tok": 1,
                  "num_hash_layers": 0,
                  "scoring_func": "sqrtsoftplus",
                  "norm_topk_prob": true,
                  "routed_scaling_factor": 1.0,
                  "swiglu_limit": 0.0,
                  "hc_mult": 1,
                  "hc_sinkhorn_iters": 1,
                  "hc_eps": 1e-6,
                  "compress_ratios": [0],
                  "vocab_size": 4,
                  "max_position_embeddings": 16,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 10000,
                  "tie_word_embeddings": false,
                  "eos_token_id": 99
                }"#,
            )
            .unwrap();
        }

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            let vocab = [
                -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
            ];
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert(
                "lm_head.weight".to_string(),
                Array::from_slice(
                    &[
                        -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                        0.0, 0.0,
                    ],
                    &[4, 4],
                ),
            );
            arrays.insert("model.hc_head.fn".to_string(), zeros(&[1, 4]));
            arrays.insert("model.hc_head.base".to_string(), zeros(&[1]));
            arrays.insert("model.hc_head.scale".to_string(), zeros(&[1]));
            arrays.insert("model.norm.weight".to_string(), ones(4));

            let prefix = "model.layers.0";
            arrays.insert(format!("{prefix}.attn_norm.weight"), ones(4));
            let attn = format!("{prefix}.attn");
            arrays.insert(format!("{attn}.wq_a.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.q_norm.weight"), ones(4));
            arrays.insert(format!("{attn}.wq_b.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.wkv.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.kv_norm.weight"), ones(4));
            arrays.insert(format!("{attn}.wo_a.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{attn}.wo_b.weight"), zeros(&[4, 4]));
            arrays.insert(format!("{prefix}.attn_hc.fn"), zeros(&[3, 4]));
            arrays.insert(format!("{prefix}.attn_hc.base"), zeros(&[3]));
            arrays.insert(format!("{prefix}.attn_hc.scale"), zeros(&[3]));

            arrays.insert(format!("{prefix}.ffn_norm.weight"), ones(4));
            arrays.insert(format!("{prefix}.ffn.gate.weight"), zeros(&[2, 4]));
            for name in ["gate_proj", "up_proj", "down_proj"] {
                arrays.insert(
                    format!("{prefix}.ffn.switch_mlp.{name}.weight"),
                    zeros(&[2, 4, 4]),
                );
            }
            arrays.insert(format!("{prefix}.ffn_hc.fn"), zeros(&[3, 4]));
            arrays.insert(format!("{prefix}.ffn_hc.base"), zeros(&[3]));
            arrays.insert(format!("{prefix}.ffn_hc.scale"), zeros(&[3]));

            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
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

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_qwen_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-qwen");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir);
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend = super::MlxBackend::load(&dir, Some("tiny-qwen".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert!((1..=2).contains(&output.completion_tokens));
        assert!(!output.text.trim().is_empty());

        fn write_config(root: &Path) {
            fs::write(
                root.join("config.json"),
                r#"{
                  "architectures": ["Qwen3ForCausalLM"],
                  "model_type": "qwen3",
                  "hidden_size": 4,
                  "intermediate_size": 8,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "vocab_size": 4,
                  "max_position_embeddings": 16,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 1000000,
                  "tie_word_embeddings": true,
                  "eos_token_id": 2
                }"#,
            )
            .unwrap();
        }

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(
                    &[
                        -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0,
                    ],
                    &[4, 4],
                ),
            );
            arrays.insert("model.norm.weight".to_string(), ones(4));
            arrays.insert("model.layers.0.input_layernorm.weight".to_string(), ones(4));
            arrays.insert(
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                ones(4),
            );
            arrays.insert(
                "model.layers.0.self_attn.q_norm.weight".to_string(),
                ones(4),
            );
            arrays.insert(
                "model.layers.0.self_attn.k_norm.weight".to_string(),
                ones(4),
            );
            for name in [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
            ] {
                arrays.insert(format!("model.layers.0.self_attn.{name}"), zeros(&[4, 4]));
            }
            arrays.insert(
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.up_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.down_proj.weight".to_string(),
                zeros(&[4, 8]),
            );
            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
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

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_qwen_moe_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-qwen-moe");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir);
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend = super::MlxBackend::load(&dir, Some("tiny-qwen-moe".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert_eq!(output.completion_tokens, 2);
        assert!(!output.text.trim().is_empty());

        fn write_config(root: &Path) {
            fs::write(
                root.join("config.json"),
                r#"{
                  "architectures": ["Qwen3MoeForCausalLM"],
                  "model_type": "qwen3_moe",
                  "hidden_size": 4,
                  "intermediate_size": 8,
                  "moe_intermediate_size": 4,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "num_experts": 2,
                  "num_experts_per_tok": 1,
                  "decoder_sparse_step": 1,
                  "mlp_only_layers": [],
                  "norm_topk_prob": true,
                  "vocab_size": 4,
                  "max_position_embeddings": 16,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 1000000,
                  "tie_word_embeddings": true,
                  "eos_token_id": 99
                }"#,
            )
            .unwrap();
        }

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(
                    &[
                        -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0,
                    ],
                    &[4, 4],
                ),
            );
            arrays.insert("model.norm.weight".to_string(), ones(4));
            arrays.insert("model.layers.0.input_layernorm.weight".to_string(), ones(4));
            arrays.insert(
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                ones(4),
            );
            arrays.insert(
                "model.layers.0.self_attn.q_norm.weight".to_string(),
                ones(4),
            );
            arrays.insert(
                "model.layers.0.self_attn.k_norm.weight".to_string(),
                ones(4),
            );
            for name in [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
            ] {
                arrays.insert(format!("model.layers.0.self_attn.{name}"), zeros(&[4, 4]));
            }
            arrays.insert("model.layers.0.mlp.gate.weight".to_string(), zeros(&[2, 4]));
            for expert in 0..2 {
                for name in ["gate_proj", "up_proj", "down_proj"] {
                    arrays.insert(
                        format!("model.layers.0.mlp.experts.{expert}.{name}.weight"),
                        zeros(&[4, 4]),
                    );
                }
            }
            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
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

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_glm_moe_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-glm-moe");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir);
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend = super::MlxBackend::load(&dir, Some("tiny-glm".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert!((1..=2).contains(&output.completion_tokens));
        assert!(!output.text.trim().is_empty());

        fn write_config(root: &Path) {
            fs::write(
                root.join("config.json"),
                r#"{
                  "architectures": ["Glm4MoeLiteForCausalLM"],
                  "model_type": "glm4_moe_lite",
                  "hidden_size": 4,
                  "intermediate_size": 8,
                  "moe_intermediate_size": 4,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "qk_nope_head_dim": 2,
                  "qk_rope_head_dim": 2,
                  "v_head_dim": 2,
                  "q_lora_rank": 3,
                  "kv_lora_rank": 3,
                  "vocab_size": 4,
                  "max_position_embeddings": 16,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 1000000,
                  "tie_word_embeddings": false,
                  "first_k_dense_replace": 1,
                  "moe_layer_freq": 1,
                  "n_routed_experts": 2,
                  "n_shared_experts": 1,
                  "num_experts_per_tok": 1,
                  "n_group": 1,
                  "topk_group": 1,
                  "topk_method": "noaux_tc",
                  "eos_token_id": 99
                }"#,
            )
            .unwrap();
        }

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            let vocab = [
                -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert(
                "lm_head.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert("model.norm.weight".to_string(), ones(4));
            for layer in 0..2 {
                let prefix = format!("model.layers.{layer}");
                arrays.insert(format!("{prefix}.input_layernorm.weight"), ones(4));
                arrays.insert(format!("{prefix}.post_attention_layernorm.weight"), ones(4));
                let attn = format!("{prefix}.self_attn");
                arrays.insert(format!("{attn}.q_a_proj.weight"), zeros(&[3, 4]));
                arrays.insert(format!("{attn}.q_a_layernorm.weight"), ones(3));
                arrays.insert(format!("{attn}.q_b_proj.weight"), zeros(&[4, 3]));
                arrays.insert(format!("{attn}.kv_a_proj_with_mqa.weight"), zeros(&[5, 4]));
                arrays.insert(format!("{attn}.kv_a_layernorm.weight"), ones(3));
                arrays.insert(format!("{attn}.embed_q.weight"), zeros(&[1, 3, 2]));
                arrays.insert(format!("{attn}.unembed_out.weight"), zeros(&[1, 2, 3]));
                arrays.insert(format!("{attn}.o_proj.weight"), zeros(&[4, 2]));
            }
            arrays.insert(
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.up_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.down_proj.weight".to_string(),
                zeros(&[4, 8]),
            );
            arrays.insert("model.layers.1.mlp.gate.weight".to_string(), zeros(&[2, 4]));
            arrays.insert(
                "model.layers.1.mlp.gate.e_score_correction_bias".to_string(),
                zeros(&[2]),
            );
            for name in ["gate_proj", "up_proj"] {
                arrays.insert(
                    format!("model.layers.1.mlp.switch_mlp.{name}.weight"),
                    zeros(&[2, 4, 4]),
                );
            }
            arrays.insert(
                "model.layers.1.mlp.switch_mlp.down_proj.weight".to_string(),
                zeros(&[2, 4, 4]),
            );
            for name in ["gate_proj", "up_proj"] {
                arrays.insert(
                    format!("model.layers.1.mlp.shared_experts.{name}.weight"),
                    zeros(&[4, 4]),
                );
            }
            arrays.insert(
                "model.layers.1.mlp.shared_experts.down_proj.weight".to_string(),
                zeros(&[4, 4]),
            );
            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
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

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    #[tokio::test]
    async fn native_backend_generates_from_tiny_deepseek_derived_mla_fixture() {
        use std::collections::HashMap;
        use std::fs;
        use std::path::{Path, PathBuf};

        use mlx_rs::Array;
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;

        let _guard = MLX_TEST_LOCK.lock().await;
        let dir = tempfile_path("native-deepseek-mla");
        fs::create_dir_all(&dir).unwrap();
        write_config(&dir);
        write_tokenizer(&dir);
        write_weights(&dir);

        let backend = super::MlxBackend::load(&dir, Some("tiny-deepseek".to_string())).unwrap();
        let output = super::InferenceBackend::generate(
            &backend,
            super::GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: 2,
                temperature: 0.0,
                top_p: 1.0,
            },
        )
        .await
        .unwrap();

        assert_eq!(output.prompt_tokens, 1);
        assert_eq!(output.completion_tokens, 2);
        assert!(!output.text.trim().is_empty());

        fn write_config(root: &Path) {
            fs::write(
                root.join("config.json"),
                r#"{
                  "architectures": ["DeepseekV32ForCausalLM"],
                  "model_type": "deepseek_v32",
                  "hidden_size": 4,
                  "intermediate_size": 8,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "qk_nope_head_dim": 2,
                  "qk_rope_head_dim": 2,
                  "v_head_dim": 2,
                  "q_lora_rank": 3,
                  "kv_lora_rank": 3,
                  "index_head_dim": 2,
                  "index_n_heads": 1,
                  "index_topk": 1,
                  "indexer_rope_interleave": false,
                  "vocab_size": 4,
                  "max_position_embeddings": 16,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 10000,
                  "tie_word_embeddings": false,
                  "eos_token_id": 99
                }"#,
            )
            .unwrap();
        }

        fn write_tokenizer(root: &Path) {
            let model = WordLevel::builder()
                .vocab(HashMap::from([
                    ("<unk>".to_string(), 0),
                    ("hello".to_string(), 1),
                    ("</s>".to_string(), 2),
                    ("world".to_string(), 3),
                ]))
                .unk_token("<unk>".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(root.join("tokenizer.json"), false)
                .unwrap();
        }

        fn write_weights(root: &Path) {
            let mut arrays = HashMap::new();
            let vocab = [
                -1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            arrays.insert(
                "model.embed_tokens.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert(
                "lm_head.weight".to_string(),
                Array::from_slice(&vocab, &[4, 4]),
            );
            arrays.insert("model.norm.weight".to_string(), ones(4));
            arrays.insert("model.layers.0.input_layernorm.weight".to_string(), ones(4));
            arrays.insert(
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                ones(4),
            );
            let attn = "model.layers.0.self_attn";
            arrays.insert(format!("{attn}.q_a_proj.weight"), zeros(&[3, 4]));
            arrays.insert(format!("{attn}.q_a_layernorm.weight"), ones(3));
            arrays.insert(format!("{attn}.q_b_proj.weight"), zeros(&[4, 3]));
            arrays.insert(format!("{attn}.kv_a_proj_with_mqa.weight"), zeros(&[5, 4]));
            arrays.insert(format!("{attn}.kv_a_layernorm.weight"), ones(3));
            arrays.insert(format!("{attn}.kv_b_proj.weight"), zeros(&[4, 3]));
            arrays.insert(format!("{attn}.o_proj.weight"), zeros(&[4, 2]));
            arrays.insert(format!("{attn}.indexer.wq_b.weight"), zeros(&[2, 3]));
            arrays.insert(format!("{attn}.indexer.wk.weight"), zeros(&[2, 4]));
            arrays.insert(format!("{attn}.indexer.k_norm.weight"), ones(2));
            arrays.insert(format!("{attn}.indexer.k_norm.bias"), zeros(&[2]));
            arrays.insert(
                format!("{attn}.indexer.weights_proj.weight"),
                zeros(&[1, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.up_proj.weight".to_string(),
                zeros(&[8, 4]),
            );
            arrays.insert(
                "model.layers.0.mlp.down_proj.weight".to_string(),
                zeros(&[4, 8]),
            );
            Array::save_safetensors(&arrays, None, root.join("model.safetensors")).unwrap();
        }

        fn ones(len: usize) -> Array {
            Array::from_slice(&vec![1.0f32; len], &[len as i32])
        }

        fn zeros(shape: &[i32]) -> Array {
            let len = shape.iter().product::<i32>() as usize;
            Array::from_slice(&vec![0.0f32; len], shape)
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
}
