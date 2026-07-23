//! Qwen-family GGUF config, tensor weight-name tables, and layout validation.
//!
//! Extracted from `lib.rs` as a pure code move; all public items are
//! re-exported from the crate root so `hi_gguf::X` paths are unchanged.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, anyhow, bail};
use hi_local_core::model::ModelFamily;
use serde::Serialize;

use crate::{GgufFile, GgufTensorType, MetadataValue, TensorInfo, array_len_as_u32};

#[derive(Clone, Debug, Serialize)]
pub struct QwenGgufConfig {
    pub architecture: String,
    pub family: ModelFamily,
    pub context_length: u32,
    pub embedding_length: u32,
    pub feed_forward_length: Option<u32>,
    pub expert_feed_forward_length: Option<u32>,
    /// Trunk decoder layers only. GGUFs whose `block_count` INCLUDES trailing
    /// NextN/MTP draft layers (`nextn_predict_layers` > 0, e.g. GLM-5.2) have
    /// those subtracted here; the MTP block's tensors live at the excluded
    /// indices and are ignored until speculative decoding consumes them.
    pub block_count: u32,
    /// Trailing NextN/MTP draft layers excluded from `block_count`.
    pub nextn_predict_layers: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,
    pub attention_key_length: Option<u32>,
    pub attention_value_length: Option<u32>,
    /// Sliding-window size for local attention layers (Gemma-2/3/4). Gemma-3
    /// interleaves local (windowed) and global (full) attention layers.
    pub attention_sliding_window: Option<u32>,
    /// Gemma-4 per-layer local/global pattern (`attention.sliding_window_pattern`,
    /// true = sliding). When absent, Gemma-3 falls back to the 5:1 pattern.
    pub attention_sliding_window_pattern: Option<Vec<bool>>,
    /// Gemma-4 per-layer KV head counts (`attention.head_count_kv` as an array;
    /// SWA layers use GQA, global layers MQA).
    pub attention_head_count_kv_per_layer: Option<Vec<u32>>,
    /// Gemma-4 head dims for SLIDING layers (`attention.key/value_length_swa`);
    /// `attention.key/value_length` then applies to global layers only.
    pub attention_key_length_swa: Option<u32>,
    pub attention_value_length_swa: Option<u32>,
    /// Gemma-4 KV sharing: the last N layers reuse an earlier layer's KV and
    /// carry no attn_k/attn_v projections.
    pub attention_shared_kv_layers: Option<u32>,
    /// Gemma-4 rope base/rot dims for SLIDING layers; the non-`_swa` keys then
    /// apply to global layers only.
    pub rope_freq_base_swa: Option<f32>,
    pub rope_dimension_count_swa: Option<u32>,
    /// Gemma-4 per-layer input embeddings width (gemma3n-style; 0/absent = none).
    pub embedding_length_per_layer_input: Option<u32>,
    /// Per-layer FFN widths (`feed_forward_length` as an array, gemma4 E-series).
    pub feed_forward_length_per_layer: Option<Vec<u32>>,
    pub attention_q_lora_rank: Option<u32>,
    pub attention_kv_lora_rank: Option<u32>,
    pub attention_qk_rope_head_dim: Option<u32>,
    pub attention_qk_nope_head_dim: Option<u32>,
    pub attention_v_head_dim: Option<u32>,
    pub attention_qk_head_dim: Option<u32>,
    pub attention_mla_tensor_layout: bool,
    // --- DeepSeek-V4 (`deepseek4`) ---
    /// Lightning-indexer geometry (`{arch}.attention.indexer.*`): head count,
    /// per-head key length, and the top-k tokens each query keeps.
    pub attention_indexer_head_count: Option<u32>,
    pub attention_indexer_key_length: Option<u32>,
    pub attention_indexer_top_k: Option<u32>,
    /// Grouped attention output projection (`attention.output_group_count` /
    /// `attention.output_lora_rank`): out = wo_b(concat_g wo_a[g](attn[g])).
    pub attention_output_group_count: Option<u32>,
    pub attention_output_lora_rank: Option<u32>,
    /// Per-layer KV-compressor ratios (`attention.compress_ratios`; 0 = no
    /// compressor). May carry one extra trailing entry for a stripped MTP layer.
    pub attention_compress_ratios: Option<Vec<u32>>,
    pub attention_compress_rope_freq_base: Option<f32>,
    /// Hyper-connection residual streams (`hyper_connection.count`,
    /// `.sinkhorn_iterations`, `.epsilon`).
    pub hyper_connection_count: Option<u32>,
    pub hyper_connection_sinkhorn_iterations: Option<u32>,
    pub hyper_connection_epsilon: Option<f32>,
    /// First N MoE layers route by token id via `ffn_gate_tid2eid` lookup
    /// tables instead of the learned router (`hash_layer_count`).
    pub hash_layer_count: Option<u32>,
    /// Per-layer SwiGLU activation clamps for routed / shared experts.
    pub swiglu_clamp_exp: Option<Vec<f32>>,
    pub swiglu_clamp_shexp: Option<Vec<f32>>,
    /// Explicit YARN correction betas (`rope.scaling.yarn_beta_fast`/`_slow`).
    pub rope_scaling_yarn_beta_fast: Option<f32>,
    pub rope_scaling_yarn_beta_slow: Option<f32>,
    pub recurrent_ssm_tensor_layout: bool,
    pub ssm_conv_kernel: Option<u32>,
    pub ssm_inner_size: Option<u32>,
    pub ssm_state_size: Option<u32>,
    pub ssm_time_step_rank: Option<u32>,
    pub ssm_group_count: Option<u32>,
    pub ssm_dt_b_c_rms: Option<bool>,
    pub full_attention_interval: Option<u32>,
    pub attention_recurrent_layers: Option<Vec<bool>>,
    pub expert_count: Option<u32>,
    pub expert_used_count: Option<u32>,
    /// Group-limited routing (DeepSeek-V3 `n_group`/`topk_group`): experts are
    /// partitioned into `expert_group_count` groups and only the top
    /// `expert_group_used_count` groups are searched. `> 1` is rejected at
    /// load (unimplemented in the qwen MoE paths; GLM-5.2 is n_group = 1 so
    /// the metadata is either absent or 1 there).
    pub expert_group_count: Option<u32>,
    pub expert_group_used_count: Option<u32>,
    /// Number of always-on shared experts (DeepSeek MoE `n_shared_experts`).
    /// Their fused MLP intermediate width is `expert_ff * expert_shared_count`.
    pub expert_shared_count: Option<u32>,
    pub expert_weights_norm: bool,
    /// llama.cpp expert gating function enum: 1 = softmax (default), 2 =
    /// sigmoid (DeepSeek-V3/GLM-5 class), 4 = sqrt-softplus (DeepSeek-V4).
    pub expert_gating_func: Option<u32>,
    /// Routed-expert weight multiplier (DeepSeek `routed_scaling_factor`).
    pub expert_weights_scale: Option<f32>,
    pub rope_freq_base: Option<f32>,
    pub rope_freq_scale: Option<f32>,
    /// YARN rope scaling (DeepSeek-V2/V3). `rope.scaling.type == "yarn"`; the
    /// factor extends context by `rope_scaling_factor` from
    /// `rope_scaling_original_context_length`, and `rope_yarn_log_multiplier`
    /// (llama.cpp `rope_yarn_log_mul`) sets the attention mscale.
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_original_context_length: Option<u32>,
    pub rope_yarn_log_multiplier: Option<f32>,
    pub rope_dimension_sections: Option<[u32; 4]>,
    /// Rotary dimension count (`{arch}.rope.dimension_count`). When smaller
    /// than the attention head dim (e.g. Qwen3.5 rotates 64 of 256 dims), only
    /// the leading `rope_dimension_count` dims of each head are rotated.
    pub rope_dimension_count: Option<u32>,
    pub rms_norm_eps: Option<f32>,
    /// Gemma-2 attention logit soft-cap (`cap * tanh(score/cap)` on the QK^T
    /// scores). In practice the scaled scores sit far inside the cap, so this is
    /// effectively a no-op for coherence; kept for metadata completeness.
    pub attn_logit_softcapping: Option<f32>,
    /// Gemma-2 final logit soft-cap applied to the lm_head output. Monotonic, so
    /// it does not change greedy argmax but compresses the sampling distribution.
    pub final_logit_softcapping: Option<f32>,
    pub vocab_size: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub file_type: Option<u32>,
    pub tensor_dtypes: Vec<String>,
    pub total_tensor_bytes: u64,
}

impl QwenGgufConfig {
    /// DeepSeek-V4 (`deepseek4` arch): latent-MQA attention (single shared KV
    /// head, no kv_b), attention sinks, per-layer compressors + lightning
    /// indexer, hyper-connection residual streams, grouped output projection,
    /// hash-routed leading MoE layers, sqrt-softplus gating.
    pub fn is_deepseek4(&self) -> bool {
        self.architecture == "deepseek4"
    }

    pub(crate) fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow!("GGUF metadata missing general.architecture"))?
            .to_ascii_lowercase();
        let family = ModelFamily::from_gguf_architecture(&architecture).ok_or_else(|| {
            anyhow!(
                "unsupported GGUF architecture '{architecture}'; only Qwen/Llama/Mistral/Mixtral/Gemma/Phi/DeepSeek/GLM GGUF is accepted"
            )
        })?;
        let prefix = architecture.clone();
        let context_length = gguf.required_u32(&format!("{prefix}.context_length"))?;
        let embedding_length = gguf.required_u32(&format!("{prefix}.embedding_length"))?;
        let raw_block_count = gguf.required_u32(&format!("{prefix}.block_count"))?;
        let nextn_predict_layers = gguf
            .metadata_u32(&format!("{prefix}.nextn_predict_layers"))
            .unwrap_or(0);
        if nextn_predict_layers >= raw_block_count {
            bail!(
                "{prefix}.nextn_predict_layers {nextn_predict_layers} must be less than block_count {raw_block_count}"
            );
        }
        let block_count = raw_block_count - nextn_predict_layers;
        reject_unsupported_mla_layout(gguf, family, &prefix, block_count)?;
        reject_unsupported_qwen_ssm_layout(gguf, family, &prefix)?;
        let attention_head_count = gguf.required_u32(&format!("{prefix}.attention.head_count"))?;
        let attention_key_length = gguf.metadata_u32(&format!("{prefix}.attention.key_length"));
        let attention_value_length = gguf.metadata_u32(&format!("{prefix}.attention.value_length"));
        let attention_sliding_window =
            gguf.metadata_u32(&format!("{prefix}.attention.sliding_window"));
        let attention_sliding_window_pattern = qwen_metadata_bool_array(
            gguf,
            &format!("{prefix}.attention.sliding_window_pattern"),
            block_count,
        )?;
        let attention_head_count_kv_per_layer = qwen_metadata_u32_array(
            gguf,
            &format!("{prefix}.attention.head_count_kv"),
            block_count,
        )?;
        let attention_key_length_swa =
            gguf.metadata_u32(&format!("{prefix}.attention.key_length_swa"));
        let attention_value_length_swa =
            gguf.metadata_u32(&format!("{prefix}.attention.value_length_swa"));
        let attention_shared_kv_layers =
            gguf.metadata_u32(&format!("{prefix}.attention.shared_kv_layers"));
        let rope_freq_base_swa = gguf.metadata_f32(&format!("{prefix}.rope.freq_base_swa"));
        let rope_dimension_count_swa =
            gguf.metadata_u32(&format!("{prefix}.rope.dimension_count_swa"));
        let embedding_length_per_layer_input =
            gguf.metadata_u32(&format!("{prefix}.embedding_length_per_layer_input"));
        let feed_forward_length_per_layer =
            qwen_metadata_u32_array(gguf, &format!("{prefix}.feed_forward_length"), block_count)?;
        let attention_q_lora_rank = gguf.metadata_u32(&format!("{prefix}.attention.q_lora_rank"));
        let attention_kv_lora_rank = gguf.metadata_u32(&format!("{prefix}.attention.kv_lora_rank"));
        // glm-dsa/DeepSeek-V3.2-class ggufs carry attention.key_length_mla /
        // value_length_mla + rope.dimension_count instead of the DeepSeek-2
        // qk_rope/qk_nope/v_head keys; derive the classic dims from them.
        let attention_key_length_mla =
            gguf.metadata_u32(&format!("{prefix}.attention.key_length_mla"));
        let attention_value_length_mla =
            gguf.metadata_u32(&format!("{prefix}.attention.value_length_mla"));
        let rope_dimension_count_for_mla =
            gguf.metadata_u32(&format!("{prefix}.rope.dimension_count"));
        // DeepSeek-V2/V2-Lite-class ggufs (no Q-LoRA) carry the classic
        // attention.key_length / value_length + rope.dimension_count rather than
        // the DeepSeek-V3.2/glm-dsa `_mla`-suffixed keys or the explicit
        // qk_rope/qk_nope/v_head keys. When a kv_lora_rank marks the layer as MLA,
        // derive qk_rope = rope.dimension_count, qk_nope = key_length - qk_rope,
        // and v_head = value_length from those plain keys.
        let mla_dims_from_plain_keys = attention_kv_lora_rank.is_some();
        let attention_qk_rope_head_dim = gguf
            .metadata_u32(&format!("{prefix}.attention.qk_rope_head_dim"))
            .or(attention_key_length_mla.and(rope_dimension_count_for_mla))
            .or(rope_dimension_count_for_mla.filter(|_| mla_dims_from_plain_keys));
        let attention_qk_nope_head_dim = gguf
            .metadata_u32(&format!("{prefix}.attention.qk_nope_head_dim"))
            .or(
                match (attention_key_length_mla, rope_dimension_count_for_mla) {
                    (Some(mla), Some(rope)) if mla > rope => Some(mla - rope),
                    _ => None,
                },
            )
            .or_else(
                || match (attention_key_length, rope_dimension_count_for_mla) {
                    (Some(key), Some(rope)) if mla_dims_from_plain_keys && key > rope => {
                        Some(key - rope)
                    }
                    _ => None,
                },
            );
        let attention_v_head_dim = gguf
            .metadata_u32(&format!("{prefix}.attention.v_head_dim"))
            .or(attention_value_length_mla)
            .or(attention_value_length.filter(|_| mla_dims_from_plain_keys));
        let attention_qk_head_dim = gguf.metadata_u32(&format!("{prefix}.attention.qk_head_dim"));
        let attention_mla_tensor_layout = qwen_mla_decoder_tensors_present(gguf, block_count);
        let attention_indexer_head_count =
            gguf.metadata_u32(&format!("{prefix}.attention.indexer.head_count"));
        let attention_indexer_key_length =
            gguf.metadata_u32(&format!("{prefix}.attention.indexer.key_length"));
        let attention_indexer_top_k =
            gguf.metadata_u32(&format!("{prefix}.attention.indexer.top_k"));
        let attention_output_group_count =
            gguf.metadata_u32(&format!("{prefix}.attention.output_group_count"));
        let attention_output_lora_rank =
            gguf.metadata_u32(&format!("{prefix}.attention.output_lora_rank"));
        let attention_compress_ratios =
            qwen_metadata_u32_array_lenient(gguf, &format!("{prefix}.attention.compress_ratios"))?;
        let attention_compress_rope_freq_base =
            gguf.metadata_f32(&format!("{prefix}.attention.compress_rope_freq_base"));
        let hyper_connection_count = gguf.metadata_u32(&format!("{prefix}.hyper_connection.count"));
        let hyper_connection_sinkhorn_iterations =
            gguf.metadata_u32(&format!("{prefix}.hyper_connection.sinkhorn_iterations"));
        let hyper_connection_epsilon =
            gguf.metadata_f32(&format!("{prefix}.hyper_connection.epsilon"));
        let hash_layer_count = gguf.metadata_u32(&format!("{prefix}.hash_layer_count"));
        let swiglu_clamp_exp = gguf.metadata_f32_array(&format!("{prefix}.swiglu_clamp_exp"))?;
        let swiglu_clamp_shexp =
            gguf.metadata_f32_array(&format!("{prefix}.swiglu_clamp_shexp"))?;
        let rope_scaling_yarn_beta_fast =
            gguf.metadata_f32(&format!("{prefix}.rope.scaling.yarn_beta_fast"));
        let rope_scaling_yarn_beta_slow =
            gguf.metadata_f32(&format!("{prefix}.rope.scaling.yarn_beta_slow"));
        let recurrent_ssm_tensor_layout =
            qwen_any_recurrent_ssm_layer_tensors_present(gguf, block_count);
        let attention_recurrent_layers =
            qwen_attention_recurrent_layers(gguf, &prefix, block_count)?;
        reject_unsupported_custom_attention_lengths(
            family,
            &prefix,
            embedding_length,
            attention_head_count,
            attention_key_length,
            attention_value_length.or(attention_v_head_dim),
        )?;
        let attention_head_count_kv = gguf
            .metadata_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(attention_head_count);
        let expert_count = gguf.metadata_u32(&format!("{prefix}.expert_count"));
        let expert_used_count = gguf.metadata_u32(&format!("{prefix}.expert_used_count"));
        let expert_group_count = gguf.metadata_u32(&format!("{prefix}.expert_group_count"));
        let expert_group_used_count =
            gguf.metadata_u32(&format!("{prefix}.expert_group_used_count"));
        let expert_shared_count = gguf.metadata_u32(&format!("{prefix}.expert_shared_count"));
        let vocab_size = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(array_len_as_u32);

        let mut tensor_dtypes = gguf
            .tensors
            .iter()
            .map(|tensor| tensor.dtype.label().to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        tensor_dtypes.sort();
        let total_tensor_bytes = gguf.tensors.iter().try_fold(0u64, |acc, tensor| {
            Ok::<_, anyhow::Error>(acc + tensor.byte_len()?)
        })?;

        let rope_dimension_sections = match gguf
            .metadata_i32_array(&format!("{prefix}.rope.dimension_sections"))?
        {
            Some(values) => {
                if values.len() != 4 {
                    bail!(
                        "GGUF metadata {prefix}.rope.dimension_sections must contain exactly 4 integers, got {}",
                        values.len()
                    );
                }
                let mut sections = [0u32; 4];
                for (idx, value) in values.into_iter().enumerate() {
                    if value < 0 {
                        bail!(
                            "GGUF metadata {prefix}.rope.dimension_sections contains negative value {value}"
                        );
                    }
                    sections[idx] = u32::try_from(value).with_context(|| {
                        format!("GGUF metadata {prefix}.rope.dimension_sections is out of range")
                    })?;
                }
                Some(sections)
            }
            None => None,
        };

        let feed_forward_length = gguf.metadata_u32(&format!("{prefix}.feed_forward_length"));
        let expert_feed_forward_length = gguf
            .metadata_u32(&format!("{prefix}.expert_feed_forward_length"))
            .or_else(|| gguf.metadata_u32(&format!("{prefix}.moe_feed_forward_length")))
            .or_else(|| gguf.metadata_u32(&format!("{prefix}.moe_intermediate_size")))
            .or_else(|| {
                if expert_count.is_some() {
                    feed_forward_length
                } else {
                    None
                }
            });

        Ok(Self {
            architecture,
            nextn_predict_layers,
            family,
            context_length,
            embedding_length,
            feed_forward_length,
            expert_feed_forward_length,
            block_count,
            attention_head_count,
            attention_head_count_kv,
            attention_key_length,
            attention_value_length,
            attention_sliding_window,
            attention_sliding_window_pattern,
            attention_head_count_kv_per_layer,
            attention_key_length_swa,
            attention_value_length_swa,
            attention_shared_kv_layers,
            rope_freq_base_swa,
            rope_dimension_count_swa,
            embedding_length_per_layer_input,
            feed_forward_length_per_layer,
            attention_q_lora_rank,
            attention_kv_lora_rank,
            attention_qk_rope_head_dim,
            attention_qk_nope_head_dim,
            attention_v_head_dim,
            attention_qk_head_dim,
            attention_mla_tensor_layout,
            attention_indexer_head_count,
            attention_indexer_key_length,
            attention_indexer_top_k,
            attention_output_group_count,
            attention_output_lora_rank,
            attention_compress_ratios,
            attention_compress_rope_freq_base,
            hyper_connection_count,
            hyper_connection_sinkhorn_iterations,
            hyper_connection_epsilon,
            hash_layer_count,
            swiglu_clamp_exp,
            swiglu_clamp_shexp,
            rope_scaling_yarn_beta_fast,
            rope_scaling_yarn_beta_slow,
            recurrent_ssm_tensor_layout,
            ssm_conv_kernel: gguf.metadata_u32(&format!("{prefix}.ssm.conv_kernel")),
            ssm_inner_size: gguf.metadata_u32(&format!("{prefix}.ssm.inner_size")),
            ssm_state_size: gguf.metadata_u32(&format!("{prefix}.ssm.state_size")),
            ssm_time_step_rank: gguf.metadata_u32(&format!("{prefix}.ssm.time_step_rank")),
            ssm_group_count: gguf.metadata_u32(&format!("{prefix}.ssm.group_count")),
            ssm_dt_b_c_rms: gguf.metadata_bool(&format!("{prefix}.ssm.dt_b_c_rms")),
            full_attention_interval: gguf
                .metadata_u32(&format!("{prefix}.full_attention_interval")),
            attention_recurrent_layers,
            expert_count,
            expert_used_count,
            expert_group_count,
            expert_group_used_count,
            expert_shared_count,
            expert_weights_norm: gguf
                .metadata_bool(&format!("{prefix}.expert_weights_norm"))
                .unwrap_or(true),
            expert_gating_func: gguf.metadata_u32(&format!("{prefix}.expert_gating_func")),
            expert_weights_scale: gguf.metadata_f32(&format!("{prefix}.expert_weights_scale")),
            rope_freq_base: gguf.metadata_f32(&format!("{prefix}.rope.freq_base")),
            rope_freq_scale: gguf.metadata_f32(&format!("{prefix}.rope.freq_scale")),
            rope_scaling_type: gguf
                .metadata_string(&format!("{prefix}.rope.scaling.type"))
                .map(str::to_string),
            rope_scaling_factor: gguf.metadata_f32(&format!("{prefix}.rope.scaling.factor")),
            rope_scaling_original_context_length: gguf
                .metadata_u32(&format!("{prefix}.rope.scaling.original_context_length")),
            rope_yarn_log_multiplier: gguf
                .metadata_f32(&format!("{prefix}.rope.scaling.yarn_log_multiplier")),
            rope_dimension_sections,
            rope_dimension_count: gguf.metadata_u32(&format!("{prefix}.rope.dimension_count")),
            rms_norm_eps: gguf.metadata_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon")),
            attn_logit_softcapping: gguf.metadata_f32(&format!("{prefix}.attn_logit_softcapping")),
            final_logit_softcapping: gguf
                .metadata_f32(&format!("{prefix}.final_logit_softcapping")),
            vocab_size,
            eos_token_id: gguf.metadata_u32("tokenizer.ggml.eos_token_id"),
            bos_token_id: gguf.metadata_u32("tokenizer.ggml.bos_token_id"),
            file_type: gguf.metadata_u32("general.file_type"),
            tensor_dtypes,
            total_tensor_bytes,
        })
    }

    /// Gemma family (Gemma-1/2/3). Gemma scales token embeddings by sqrt(hidden),
    /// uses a GeGLU MLP, and (Gemma-2+) adds post-attention/post-FFN norms and
    /// (Gemma-2) logit soft-capping — all handled specially by the CUDA forward pass.
    pub fn is_gemma(&self) -> bool {
        matches!(self.family, ModelFamily::Gemma)
    }

    /// Gemma-3 specifically. Gemma-3 interleaves local (sliding-window) and global
    /// (full) attention layers that use *different* RoPE bases (local 10000, global
    /// `rope.freq_base`), which the CUDA forward pass applies per layer.
    pub fn is_gemma3(&self) -> bool {
        self.architecture == "gemma3"
    }

    pub fn is_gemma4(&self) -> bool {
        self.architecture == "gemma4"
    }

    /// Whether `layer` uses sliding-window (local) attention. Gemma-4 reads the
    /// per-layer metadata pattern; Gemma-3 falls back to its fixed 5-local:1-global
    /// interleave. Models without windowing return false.
    pub fn layer_is_sliding(&self, layer: usize) -> bool {
        if self.attention_sliding_window.is_none() {
            return false;
        }
        if let Some(pattern) = &self.attention_sliding_window_pattern {
            return pattern.get(layer).copied().unwrap_or(false);
        }
        if self.is_gemma3() {
            return layer % 6 != 5;
        }
        false
    }

    /// Per-layer KV head count: gemma4 arrays override the scalar.
    pub fn layer_head_count_kv(&self, layer: usize) -> u32 {
        if let Some(per_layer) = &self.attention_head_count_kv_per_layer
            && let Some(count) = per_layer.get(layer)
        {
            return *count;
        }
        self.attention_head_count_kv
    }

    /// Per-layer qk head dim: gemma4 sliding layers use `key_length_swa`,
    /// global layers `key_length`.
    pub fn layer_key_head_dim(&self, layer: usize) -> Option<u32> {
        if self.layer_is_sliding(layer)
            && let Some(swa) = self.attention_key_length_swa
        {
            return (swa != 0).then_some(swa);
        }
        self.attention_key_head_dim()
    }

    pub fn layer_value_head_dim(&self, layer: usize) -> Option<u32> {
        if self.layer_is_sliding(layer)
            && let Some(swa) = self.attention_value_length_swa
        {
            return (swa != 0).then_some(swa);
        }
        self.attention_value_head_dim()
    }

    /// Per-layer FFN width: per-layer arrays (gemma4 E-series) override the scalar.
    pub fn layer_feed_forward_length(&self, layer: usize) -> Option<u32> {
        if let Some(per_layer) = &self.feed_forward_length_per_layer
            && let Some(width) = per_layer.get(layer)
        {
            return Some(*width);
        }
        self.feed_forward_length
    }

    pub fn quantization_label(&self) -> String {
        let mut labels = self.tensor_dtypes.clone();
        if labels.iter().any(|label| label == "Q4_K") {
            let replacement = match self.file_type {
                Some(14) => Some("Q4_K_S"),
                Some(15) => Some("Q4_K_M"),
                _ => None,
            };
            if let Some(replacement) = replacement {
                for label in &mut labels {
                    if label == "Q4_K" {
                        *label = replacement.to_string();
                    }
                }
            }
        }
        labels.sort();
        labels.dedup();
        labels.join(",")
    }

    pub fn default_rope_freq_base(&self) -> f32 {
        match self.family {
            ModelFamily::Llama
            | ModelFamily::Mistral
            | ModelFamily::Mixtral
            | ModelFamily::Gemma
            | ModelFamily::Phi
            | ModelFamily::DeepSeek => 10_000.0,
            _ => 1_000_000.0,
        }
    }

    /// Qwen3.5 pairs each gated-delta value head `h` with q/k group
    /// `h % group_count` (round-robin); Qwen3-Next block-repeats each group
    /// over `h / repeat`. Matches llama.cpp's per-arch head interleaving.
    pub fn ssm_kv_group_round_robin(&self) -> bool {
        self.architecture.starts_with("qwen35")
    }

    /// Rotary dims per head: `rope.dimension_count` when present (Qwen3.5
    /// rotates 64 of 256 dims), otherwise the full head dim.
    pub fn rope_rot_dim(&self, head_dim: usize) -> usize {
        match self.rope_dimension_count {
            Some(count) if count as usize != 0 && (count as usize) < head_dim => count as usize,
            _ => head_dim,
        }
    }

    pub fn attention_head_dim(&self) -> Option<u32> {
        let key = self.attention_key_head_dim()?;
        let value = self.attention_value_head_dim()?;
        if key == 0 || key != value {
            return None;
        }
        Some(key)
    }

    pub fn attention_key_head_dim(&self) -> Option<u32> {
        if let Some(qk) = self.attention_qk_head_dim {
            return (qk != 0).then_some(qk);
        }
        if let (Some(nope), Some(rope)) = (
            self.attention_qk_nope_head_dim,
            self.attention_qk_rope_head_dim,
        ) {
            return nope.checked_add(rope).filter(|value| *value != 0);
        }
        let dense = dense_attention_head_dim(self.embedding_length, self.attention_head_count);
        let key = self.attention_key_length.or(dense)?;
        (key != 0).then_some(key)
    }

    pub fn attention_value_head_dim(&self) -> Option<u32> {
        // MLA dim first, mirroring `attention_key_head_dim`: glm-dsa GGUFs carry
        // the latent compressed-KV width in plain `attention.value_length` (512)
        // alongside the true per-head dim in `value_length_mla` (256); preferring
        // the plain key poisons every downstream v_head consumer (attn_output
        // validation, KV page sizing, health reporting).
        if let Some(v) = self.attention_v_head_dim {
            return (v != 0).then_some(v);
        }
        let dense = dense_attention_head_dim(self.embedding_length, self.attention_head_count);
        let value = self.attention_value_length.or(dense)?;
        (value != 0).then_some(value)
    }
}
fn dense_attention_head_dim(embedding_length: u32, attention_head_count: u32) -> Option<u32> {
    if attention_head_count == 0 {
        return None;
    }
    embedding_length
        .checked_rem(attention_head_count)
        .filter(|remainder| *remainder == 0)
        .map(|_| embedding_length / attention_head_count)
}

fn cuda_model_family_label(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Qwen2 | ModelFamily::Qwen3 => "Qwen",
        ModelFamily::Llama => "Llama",
        ModelFamily::Mistral => "Mistral",
        ModelFamily::Mixtral => "Mixtral",
        ModelFamily::Gemma => "Gemma",
        ModelFamily::Phi => "Phi",
        ModelFamily::DeepSeek => "DeepSeek",
        ModelFamily::GlmFlash => "GLM",
        ModelFamily::Hy3 => "Hy3",
        ModelFamily::NemotronH => "NemotronH",
        // MLX-only families (never reach the CUDA/GGUF path) — arms kept explicit so a new
        // ModelFamily variant is a compile error here rather than a silent fallthrough.
        ModelFamily::MiniMax => "MiniMax",
        ModelFamily::LongCat => "LongCat",
        ModelFamily::Laguna => "Laguna",
        ModelFamily::Inkling => "Inkling",
    }
}

fn reject_unsupported_custom_attention_lengths(
    family: ModelFamily,
    prefix: &str,
    embedding_length: u32,
    attention_head_count: u32,
    attention_key_length: Option<u32>,
    attention_value_length: Option<u32>,
) -> Result<()> {
    let expected_head_dim = dense_attention_head_dim(embedding_length, attention_head_count);
    let effective_key = attention_key_length.or(expected_head_dim);
    let effective_value = attention_value_length.or(expected_head_dim);
    if attention_key_length.is_none() && attention_value_length.is_none() {
        return Ok(());
    }
    if let (Some(key), Some(value)) = (effective_key, effective_value)
        && key != 0
        && value != 0
    {
        return Ok(());
    }

    let family_label = cuda_model_family_label(family);
    let expected = match expected_head_dim {
        Some(head_dim) => format!(
            "expected dense per-head length {head_dim} from {prefix}.embedding_length={embedding_length} / {prefix}.attention.head_count={attention_head_count}"
        ),
        None => format!(
            "could not derive a dense per-head length from {prefix}.embedding_length={embedding_length} / {prefix}.attention.head_count={attention_head_count}"
        ),
    };
    let mut unsupported = Vec::new();
    if let Some(value) = attention_key_length {
        unsupported.push(format!("{prefix}.attention.key_length={value}"));
    }
    if let Some(value) = attention_value_length {
        unsupported.push(format!("{prefix}.attention.value_length={value}"));
    }
    if attention_key_length.is_none()
        && let Some(value) = effective_key
    {
        unsupported.push(format!("{prefix}.attention.key_length=<dense {value}>"));
    }
    if attention_value_length.is_none()
        && let Some(value) = effective_value
    {
        unsupported.push(format!("{prefix}.attention.value_length=<dense {value}>"));
    }
    bail!(
        "unsupported {family_label} GGUF metadata {}: attention key/value lengths must resolve to non-zero per-head dimensions for CUDA {family_label} support; {expected}",
        unsupported.join(", ")
    );
}

fn reject_unsupported_mla_layout(
    gguf: &GgufFile,
    family: ModelFamily,
    prefix: &str,
    block_count: u32,
) -> Result<()> {
    if family != ModelFamily::DeepSeek && family != ModelFamily::GlmFlash {
        return Ok(());
    }
    // DeepSeek-V4 (`deepseek4`) attention is latent-MQA: attn_q_a/attn_q_b plus a
    // single shared attn_kv latent and NO kv_b, which the V2/V3 MLA completeness
    // rules below would misread as an incomplete MLA set. Its tensor layout is
    // validated by the deepseek4 loader instead.
    if prefix == "deepseek4" {
        return Ok(());
    }
    let family_label = cuda_model_family_label(family);
    let dense_decoder_present = decoder_dense_attention_tensors_present(gguf, block_count);
    let mla_decoder_present = qwen_mla_decoder_tensors_present(gguf, block_count);
    let mut mla_metadata_keys = Vec::new();

    for suffix in [
        "attention.q_lora_rank",
        "attention.kv_lora_rank",
        "attention.qk_rope_head_dim",
        "attention.qk_nope_head_dim",
        "attention.v_head_dim",
        "attention.qk_head_dim",
    ] {
        let key = format!("{prefix}.{suffix}");
        if gguf.metadata.contains_key(&key) {
            mla_metadata_keys.push(key);
        }
    }

    for layer in 0..block_count {
        for layer_prefix in layer_prefix_variants(&format!("blk.{layer}")) {
            for suffix in [
                "attn_q_a.weight",
                "attn_q_a_norm.weight",
                "attn_q_b.weight",
                "attn_q_a_proj.weight",
                "attn_q_a_layernorm.weight",
                "attn_q_b_proj.weight",
                "attn_kv_a_mqa.weight",
                "attn_kv_a_norm.weight",
                "attn_kv_b.weight",
                "attn_kv_a_proj_with_mqa.weight",
                "attn_kv_a_layernorm.weight",
                "attn_kv_b_proj.weight",
                "self_attn.q_a_proj.weight",
                "self_attn.q_a_layernorm.weight",
                "self_attn.q_b_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight",
                "self_attn.kv_a_proj.weight",
                "self_attn.kv_a_layernorm.weight",
                "self_attn.kv_a_norm.weight",
                "self_attn.kv_b_proj.weight",
                "self_attention.q_a_proj.weight",
                "self_attention.q_a_layernorm.weight",
                "self_attention.q_a_norm.weight",
                "self_attention.q_b_proj.weight",
                "self_attention.kv_a_proj_with_mqa.weight",
                "self_attention.kv_a_proj.weight",
                "self_attention.kv_a_layernorm.weight",
                "self_attention.kv_a_norm.weight",
                "self_attention.kv_b_proj.weight",
                "attention.q_a_proj.weight",
                "attention.q_a_layernorm.weight",
                "attention.q_a_norm.weight",
                "attention.q_b_proj.weight",
                "attention.kv_a_proj_with_mqa.weight",
                "attention.kv_a_proj.weight",
                "attention.kv_a_layernorm.weight",
                "attention.kv_a_norm.weight",
                "attention.kv_b_proj.weight",
                "attn.wkv.weight",
            ] {
                let name = format!("{layer_prefix}.{suffix}");
                if gguf.tensor(&name).is_some() {
                    if !dense_decoder_present && !mla_decoder_present {
                        bail!(
                            "unsupported {family_label} GGUF tensor layout: tensor {name} uses incomplete MLA attention; CUDA {family_label} support requires either split attn_q/attn_k/attn_v decoder tensors or a complete MLA tensor set attn_q_a/attn_q_a_norm/attn_q_b/attn_kv_a_mqa/attn_kv_a_norm/attn_kv_b"
                        );
                    }
                    break;
                }
            }
        }
    }

    if !mla_metadata_keys.is_empty() && !dense_decoder_present && !mla_decoder_present {
        bail!(
            "unsupported {family_label} GGUF metadata {}: MLA attention metadata is present without split attn_q/attn_k/attn_v decoder tensors or a complete MLA tensor set; CUDA {family_label} support can load split decoder tensors and complete low-rank MLA tensor layouts",
            mla_metadata_keys.join(", ")
        );
    }

    Ok(())
}

fn decoder_dense_attention_tensors_present(gguf: &GgufFile, block_count: u32) -> bool {
    (0..block_count).all(|layer| {
        let prefix = format!("blk.{layer}");
        decoder_split_attention_tensors_present(gguf, &prefix)
            || qwen_dense_packed_qkv_weight_names(&prefix)
                .iter()
                .any(|name| gguf.tensor(name).is_some())
    })
}

fn qwen_mla_decoder_tensors_present(gguf: &GgufFile, block_count: u32) -> bool {
    (0..block_count).all(|layer| {
        let prefix = format!("blk.{layer}");
        qwen_mla_attention_tensors_present(gguf, &prefix)
    })
}

fn qwen_recurrent_ssm_decoder_tensors_present(gguf: &GgufFile, block_count: u32) -> bool {
    block_count != 0
        && (0..block_count).all(|layer| {
            let prefix = format!("blk.{layer}");
            let dense_attention = decoder_split_attention_tensors_present(gguf, &prefix)
                || qwen_dense_packed_qkv_weight_names(&prefix)
                    .iter()
                    .any(|name| gguf.tensor(name).is_some());
            let recurrent_ssm = qwen_ssm_layer_tensors_present(gguf, &prefix);
            let split_ffn = ["gate", "up", "down"].iter().all(|kind| {
                qwen_dense_ffn_weight_names(&prefix, kind)
                    .iter()
                    .any(|name| gguf.tensor(name).is_some())
            });
            let packed_ffn = qwen_dense_ffn_weight_names(&prefix, "down")
                .iter()
                .any(|name| gguf.tensor(name).is_some())
                && qwen_dense_packed_ffn_gate_up_weight_names(&prefix)
                    .into_iter()
                    .chain(qwen_dense_packed_ffn_up_gate_weight_names(&prefix))
                    .any(|name| gguf.tensor(&name).is_some());
            (dense_attention || recurrent_ssm) && (split_ffn || packed_ffn)
        })
}

fn qwen_any_recurrent_ssm_layer_tensors_present(gguf: &GgufFile, block_count: u32) -> bool {
    (0..block_count).any(|layer| {
        let prefix = format!("blk.{layer}");
        qwen_ssm_layer_tensors_present(gguf, &prefix)
    })
}

fn reject_unsupported_qwen_ssm_layout(
    gguf: &GgufFile,
    family: ModelFamily,
    prefix: &str,
) -> Result<()> {
    if !matches!(family, ModelFamily::Qwen2 | ModelFamily::Qwen3) {
        return Ok(());
    }

    let block_count = gguf
        .metadata_u32(&format!("{prefix}.block_count"))
        .unwrap_or(0);
    let dense_decoder_present = qwen_dense_decoder_tensors_present(gguf, block_count);
    let recurrent_ssm_decoder_present =
        qwen_recurrent_ssm_decoder_tensors_present(gguf, block_count);
    let mut ssm_metadata = Vec::new();
    for key in gguf.metadata.keys() {
        if key.starts_with(prefix)
            && let Some(feature) = qwen_ssm_metadata_feature(key)
        {
            ssm_metadata.push((key.clone(), feature));
        }
    }

    for tensor in &gguf.tensors {
        if let Some(feature) = qwen_ssm_tensor_feature(&tensor.name)
            && !dense_decoder_present
            && !recurrent_ssm_decoder_present
        {
            bail!(
                "unsupported Qwen GGUF tensor layout: tensor {} uses {feature}; CUDA Qwen support requires either dense attention/MLP decoder tensors or a complete recurrent SSM tensor set ssm_in or attn_qkv+attn_gate plus ssm_conv1d/ssm_dt/ssm_a/ssm_ba (or ssm_beta+ssm_alpha)/ssm_norm/ssm_out",
                tensor.name
            );
        }
    }

    if !ssm_metadata.is_empty() && !dense_decoder_present && !recurrent_ssm_decoder_present {
        let details = ssm_metadata
            .iter()
            .map(|(key, feature)| format!("{key} ({feature})"))
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "unsupported Qwen GGUF metadata {details}: recurrent/SSM metadata is present without dense decoder tensors or a complete recurrent SSM tensor set; CUDA Qwen support requires dense attn_q/attn_k/attn_v decoder tensors or complete Qwen3-Next SSM tensors"
        );
    }

    Ok(())
}

fn qwen_dense_decoder_tensors_present(gguf: &GgufFile, block_count: u32) -> bool {
    (0..block_count).all(|layer| {
        let prefix = format!("blk.{layer}");
        let dense_attention = decoder_split_attention_tensors_present(gguf, &prefix)
            || qwen_dense_packed_qkv_weight_names(&prefix)
                .iter()
                .any(|name| gguf.tensor(name).is_some());
        let split_ffn = ["gate", "up", "down"].iter().all(|kind| {
            qwen_dense_ffn_weight_names(&prefix, kind)
                .iter()
                .any(|name| gguf.tensor(name).is_some())
        });
        let packed_ffn = qwen_dense_ffn_weight_names(&prefix, "down")
            .iter()
            .any(|name| gguf.tensor(name).is_some())
            && qwen_dense_packed_ffn_gate_up_weight_names(&prefix)
                .into_iter()
                .chain(qwen_dense_packed_ffn_up_gate_weight_names(&prefix))
                .any(|name| gguf.tensor(&name).is_some());
        dense_attention && (split_ffn || packed_ffn)
    })
}

fn decoder_split_attention_tensors_present(gguf: &GgufFile, prefix: &str) -> bool {
    ["q", "k", "v"].iter().all(|suffix| {
        qwen_dense_attention_weight_names(prefix, suffix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
    })
}

pub fn qwen_dense_output_norm_weight_names() -> Vec<String> {
    [
        "output_norm.weight",
        "norm.weight",
        "model.norm.weight",
        "language_model.norm.weight",
        "language_model.model.norm.weight",
        "transformer.norm_f.weight",
        "model.transformer.norm_f.weight",
        "transformer.encoder.final_layernorm.weight",
        "transformer.final_layernorm.weight",
        "transformer.final_layer_norm.weight",
        "model.final_layernorm.weight",
        "model.final_layer_norm.weight",
        "model.transformer.final_layernorm.weight",
        "model.transformer.final_layer_norm.weight",
        "language_model.final_layernorm.weight",
        "language_model.final_layer_norm.weight",
        "language_model.model.final_layernorm.weight",
        "language_model.model.final_layer_norm.weight",
        "gpt_neox.final_layer_norm.weight",
        "final_layernorm.weight",
        "final_layer_norm.weight",
        "transformer.ln_f.weight",
        "model.transformer.ln_f.weight",
        "ln_f.weight",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

pub fn qwen_dense_token_embd_weight_names() -> Vec<String> {
    [
        "token_embd.weight",
        "model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "embed_tokens.weight",
        "model.transformer.embed_tokens.weight",
        "model.transformer.wte.weight",
        "model.tok_embeddings.weight",
        "tok_embeddings.weight",
        "transformer.wte.weight",
        "wte.weight",
        "gpt_neox.embed_in.weight",
        "model.embed_in.weight",
        "embed_in.weight",
        "transformer.embedding.word_embeddings.weight",
        "transformer.word_embeddings.weight",
        "model.word_embeddings.weight",
        "word_embeddings.weight",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

pub fn qwen_dense_output_weight_names() -> Vec<String> {
    [
        "output.weight",
        "lm_head.weight",
        "model.lm_head.weight",
        "transformer.lm_head.weight",
        "model.transformer.lm_head.weight",
        "language_model.lm_head.weight",
        "language_model.model.lm_head.weight",
        "transformer.output_layer.weight",
        "model.transformer.output_layer.weight",
        "gpt_neox.embed_out.weight",
        "model.output.weight",
        "language_model.output.weight",
        "language_model.model.output.weight",
        "output_layer.weight",
        "model.embed_out.weight",
        "model.transformer.embed_out.weight",
        "embed_out.weight",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

pub fn qwen_dense_output_bias_names() -> Vec<String> {
    weight_aliases_to_bias_names(qwen_dense_output_weight_names())
}

pub fn qwen_dense_attention_norm_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_norm.weight"),
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.input_layer_norm.weight"),
            format!("{prefix}.pre_attention_layernorm.weight"),
            format!("{prefix}.pre_attention_layer_norm.weight"),
            format!("{prefix}.self_attn_layer_norm.weight"),
            format!("{prefix}.attention_layernorm.weight"),
            format!("{prefix}.attention_norm.weight"),
            format!("{prefix}.ln_1.weight"),
            format!("{prefix}.ln1.weight"),
        ]
    })
}

pub fn qwen_dense_ffn_norm_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_norm.weight"),
            format!("{prefix}.attn_post_norm.weight"),
            format!("{prefix}.post_attention_norm.weight"),
            format!("{prefix}.post_attention_layernorm.weight"),
            format!("{prefix}.post_attention_layer_norm.weight"),
            format!("{prefix}.pre_feedforward_layernorm.weight"),
            format!("{prefix}.pre_feedforward_layer_norm.weight"),
            format!("{prefix}.post_feedforward_layernorm.weight"),
            format!("{prefix}.post_feedforward_layer_norm.weight"),
            format!("{prefix}.post_feedforward_norm.weight"),
            format!("{prefix}.post_feed_forward_layernorm.weight"),
            format!("{prefix}.post_feed_forward_layer_norm.weight"),
            format!("{prefix}.post_feed_forward_norm.weight"),
            format!("{prefix}.post_ffw_norm.weight"),
            format!("{prefix}.ffw_norm.weight"),
            format!("{prefix}.ffn_post_norm.weight"),
            format!("{prefix}.post_ffn_norm.weight"),
            format!("{prefix}.mlp_layer_norm.weight"),
            format!("{prefix}.ffn_layernorm.weight"),
            format!("{prefix}.ffn_layer_norm.weight"),
            format!("{prefix}.ln_2.weight"),
            format!("{prefix}.ln2.weight"),
        ]
    })
}

pub fn qwen_dense_attention_weight_names(prefix: &str, suffix: &str) -> Vec<String> {
    let Some((canonical, hf, llama)) = attention_projection_alias_parts(suffix) else {
        return layer_prefix_aliases(prefix, |prefix| {
            vec![format!("{prefix}.attn_{suffix}.weight")]
        });
    };
    layer_prefix_aliases(prefix, |prefix| {
        let mut names = vec![
            format!("{prefix}.{canonical}.weight"),
            format!("{prefix}.{hf}.weight"),
            format!("{prefix}.self_attn.{hf}.weight"),
            format!("{prefix}.self_attention.{hf}.weight"),
            format!("{prefix}.attention.{hf}.weight"),
            format!("{prefix}.attn.{hf}.weight"),
            format!("{prefix}.attention.{llama}.weight"),
            format!("{prefix}.self_attn.{llama}.weight"),
            format!("{prefix}.self_attention.{llama}.weight"),
            format!("{prefix}.attn.{llama}.weight"),
            format!("{prefix}.{llama}.weight"),
        ];
        match suffix {
            "q" => names.extend([
                format!("{prefix}.query.weight"),
                format!("{prefix}.self_attn.query.weight"),
                format!("{prefix}.self_attention.query.weight"),
                format!("{prefix}.attention.query.weight"),
                format!("{prefix}.attn.query.weight"),
                format!("{prefix}.Wq.weight"),
                format!("{prefix}.self_attn.Wq.weight"),
                format!("{prefix}.self_attention.Wq.weight"),
                format!("{prefix}.attention.Wq.weight"),
                format!("{prefix}.attn.Wq.weight"),
                format!("{prefix}.w_q.weight"),
                format!("{prefix}.self_attn.w_q.weight"),
                format!("{prefix}.self_attention.w_q.weight"),
                format!("{prefix}.attention.w_q.weight"),
                format!("{prefix}.attn.w_q.weight"),
            ]),
            "k" => names.extend([
                format!("{prefix}.key.weight"),
                format!("{prefix}.self_attn.key.weight"),
                format!("{prefix}.self_attention.key.weight"),
                format!("{prefix}.attention.key.weight"),
                format!("{prefix}.attn.key.weight"),
                format!("{prefix}.Wk.weight"),
                format!("{prefix}.self_attn.Wk.weight"),
                format!("{prefix}.self_attention.Wk.weight"),
                format!("{prefix}.attention.Wk.weight"),
                format!("{prefix}.attn.Wk.weight"),
                format!("{prefix}.w_k.weight"),
                format!("{prefix}.self_attn.w_k.weight"),
                format!("{prefix}.self_attention.w_k.weight"),
                format!("{prefix}.attention.w_k.weight"),
                format!("{prefix}.attn.w_k.weight"),
            ]),
            "v" => names.extend([
                format!("{prefix}.value.weight"),
                format!("{prefix}.self_attn.value.weight"),
                format!("{prefix}.self_attention.value.weight"),
                format!("{prefix}.attention.value.weight"),
                format!("{prefix}.attn.value.weight"),
                format!("{prefix}.Wv.weight"),
                format!("{prefix}.self_attn.Wv.weight"),
                format!("{prefix}.self_attention.Wv.weight"),
                format!("{prefix}.attention.Wv.weight"),
                format!("{prefix}.attn.Wv.weight"),
                format!("{prefix}.w_v.weight"),
                format!("{prefix}.self_attn.w_v.weight"),
                format!("{prefix}.self_attention.w_v.weight"),
                format!("{prefix}.attention.w_v.weight"),
                format!("{prefix}.attn.w_v.weight"),
            ]),
            _ => {}
        }
        if suffix == "output" {
            names.extend([
                format!("{prefix}.dense.weight"),
                format!("{prefix}.self_attn.dense.weight"),
                format!("{prefix}.self_attention.dense.weight"),
                format!("{prefix}.attention.dense.weight"),
                format!("{prefix}.attn.dense.weight"),
                format!("{prefix}.out_proj.weight"),
                format!("{prefix}.self_attn.out_proj.weight"),
                format!("{prefix}.self_attention.out_proj.weight"),
                format!("{prefix}.attention.out_proj.weight"),
                format!("{prefix}.attn.out_proj.weight"),
                format!("{prefix}.mixer.out_proj.weight"),
                format!("{prefix}.c_proj.weight"),
                format!("{prefix}.attn.c_proj.weight"),
                format!("{prefix}.self_attn.c_proj.weight"),
                format!("{prefix}.self_attention.c_proj.weight"),
                format!("{prefix}.Wo.weight"),
                format!("{prefix}.self_attn.Wo.weight"),
                format!("{prefix}.self_attention.Wo.weight"),
                format!("{prefix}.attention.Wo.weight"),
                format!("{prefix}.attn.Wo.weight"),
                format!("{prefix}.w_o.weight"),
                format!("{prefix}.self_attn.w_o.weight"),
                format!("{prefix}.self_attention.w_o.weight"),
                format!("{prefix}.attention.w_o.weight"),
                format!("{prefix}.attn.w_o.weight"),
                format!("{prefix}.out.weight"),
                format!("{prefix}.self_attn.out.weight"),
                format!("{prefix}.self_attention.out.weight"),
                format!("{prefix}.attention.out.weight"),
                format!("{prefix}.attn.out.weight"),
                format!("{prefix}.proj.weight"),
                format!("{prefix}.self_attn.proj.weight"),
                format!("{prefix}.self_attention.proj.weight"),
                format!("{prefix}.attention.proj.weight"),
                format!("{prefix}.attn.proj.weight"),
            ]);
        }
        names
    })
}

pub fn qwen_dense_attention_bias_names(prefix: &str, suffix: &str) -> Vec<String> {
    let Some((canonical, hf, llama)) = attention_projection_alias_parts(suffix) else {
        return layer_prefix_aliases(prefix, |prefix| {
            vec![format!("{prefix}.attn_{suffix}.bias")]
        });
    };
    layer_prefix_aliases(prefix, |prefix| {
        let mut names = vec![
            format!("{prefix}.{canonical}.bias"),
            format!("{prefix}.{hf}.bias"),
            format!("{prefix}.self_attn.{hf}.bias"),
            format!("{prefix}.self_attention.{hf}.bias"),
            format!("{prefix}.attention.{hf}.bias"),
            format!("{prefix}.attn.{hf}.bias"),
            format!("{prefix}.attention.{llama}.bias"),
            format!("{prefix}.self_attn.{llama}.bias"),
            format!("{prefix}.self_attention.{llama}.bias"),
            format!("{prefix}.attn.{llama}.bias"),
            format!("{prefix}.{llama}.bias"),
        ];
        match suffix {
            "q" => names.extend([
                format!("{prefix}.query.bias"),
                format!("{prefix}.self_attn.query.bias"),
                format!("{prefix}.self_attention.query.bias"),
                format!("{prefix}.attention.query.bias"),
                format!("{prefix}.attn.query.bias"),
                format!("{prefix}.Wq.bias"),
                format!("{prefix}.self_attn.Wq.bias"),
                format!("{prefix}.self_attention.Wq.bias"),
                format!("{prefix}.attention.Wq.bias"),
                format!("{prefix}.attn.Wq.bias"),
                format!("{prefix}.w_q.bias"),
                format!("{prefix}.self_attn.w_q.bias"),
                format!("{prefix}.self_attention.w_q.bias"),
                format!("{prefix}.attention.w_q.bias"),
                format!("{prefix}.attn.w_q.bias"),
            ]),
            "k" => names.extend([
                format!("{prefix}.key.bias"),
                format!("{prefix}.self_attn.key.bias"),
                format!("{prefix}.self_attention.key.bias"),
                format!("{prefix}.attention.key.bias"),
                format!("{prefix}.attn.key.bias"),
                format!("{prefix}.Wk.bias"),
                format!("{prefix}.self_attn.Wk.bias"),
                format!("{prefix}.self_attention.Wk.bias"),
                format!("{prefix}.attention.Wk.bias"),
                format!("{prefix}.attn.Wk.bias"),
                format!("{prefix}.w_k.bias"),
                format!("{prefix}.self_attn.w_k.bias"),
                format!("{prefix}.self_attention.w_k.bias"),
                format!("{prefix}.attention.w_k.bias"),
                format!("{prefix}.attn.w_k.bias"),
            ]),
            "v" => names.extend([
                format!("{prefix}.value.bias"),
                format!("{prefix}.self_attn.value.bias"),
                format!("{prefix}.self_attention.value.bias"),
                format!("{prefix}.attention.value.bias"),
                format!("{prefix}.attn.value.bias"),
                format!("{prefix}.Wv.bias"),
                format!("{prefix}.self_attn.Wv.bias"),
                format!("{prefix}.self_attention.Wv.bias"),
                format!("{prefix}.attention.Wv.bias"),
                format!("{prefix}.attn.Wv.bias"),
                format!("{prefix}.w_v.bias"),
                format!("{prefix}.self_attn.w_v.bias"),
                format!("{prefix}.self_attention.w_v.bias"),
                format!("{prefix}.attention.w_v.bias"),
                format!("{prefix}.attn.w_v.bias"),
            ]),
            _ => {}
        }
        if suffix == "output" {
            names.extend([
                format!("{prefix}.dense.bias"),
                format!("{prefix}.self_attn.dense.bias"),
                format!("{prefix}.self_attention.dense.bias"),
                format!("{prefix}.attention.dense.bias"),
                format!("{prefix}.attn.dense.bias"),
                format!("{prefix}.out_proj.bias"),
                format!("{prefix}.self_attn.out_proj.bias"),
                format!("{prefix}.self_attention.out_proj.bias"),
                format!("{prefix}.attention.out_proj.bias"),
                format!("{prefix}.attn.out_proj.bias"),
                format!("{prefix}.mixer.out_proj.bias"),
                format!("{prefix}.c_proj.bias"),
                format!("{prefix}.attn.c_proj.bias"),
                format!("{prefix}.self_attn.c_proj.bias"),
                format!("{prefix}.self_attention.c_proj.bias"),
                format!("{prefix}.Wo.bias"),
                format!("{prefix}.self_attn.Wo.bias"),
                format!("{prefix}.self_attention.Wo.bias"),
                format!("{prefix}.attention.Wo.bias"),
                format!("{prefix}.attn.Wo.bias"),
                format!("{prefix}.w_o.bias"),
                format!("{prefix}.self_attn.w_o.bias"),
                format!("{prefix}.self_attention.w_o.bias"),
                format!("{prefix}.attention.w_o.bias"),
                format!("{prefix}.attn.w_o.bias"),
                format!("{prefix}.out.bias"),
                format!("{prefix}.self_attn.out.bias"),
                format!("{prefix}.self_attention.out.bias"),
                format!("{prefix}.attention.out.bias"),
                format!("{prefix}.attn.out.bias"),
                format!("{prefix}.proj.bias"),
                format!("{prefix}.self_attn.proj.bias"),
                format!("{prefix}.self_attention.proj.bias"),
                format!("{prefix}.attention.proj.bias"),
                format!("{prefix}.attn.proj.bias"),
            ]);
        }
        names
    })
}

pub fn qwen_dense_gated_attention_q_weight_name(
    gguf: &GgufFile,
    prefix: &str,
    q_dim: u64,
    embed: u64,
) -> Option<String> {
    let gated_q_dim = q_dim.checked_mul(2)?;
    qwen_dense_attention_weight_names(prefix, "q")
        .into_iter()
        .find(|name| {
            gguf.tensor(name).is_some_and(|tensor| {
                tensor_dimensions_match_matrix(&tensor.info.dimensions, embed, gated_q_dim)
            })
        })
}

pub fn qwen_dense_gated_attention_q_bias_name(
    gguf: &GgufFile,
    prefix: &str,
    q_dim: u64,
) -> Option<String> {
    let gated_q_dim = q_dim.checked_mul(2)?;
    qwen_dense_attention_bias_names(prefix, "q")
        .into_iter()
        .find(|name| {
            gguf.tensor(name)
                .is_some_and(|tensor| tensor.info.dimensions == [gated_q_dim])
        })
}

pub fn qwen_dense_packed_qkv_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_qkv.weight"),
            format!("{prefix}.qkv.weight"),
            format!("{prefix}.qkv_proj.weight"),
            format!("{prefix}.self_attn.qkv.weight"),
            format!("{prefix}.self_attn.qkv_proj.weight"),
            format!("{prefix}.self_attention.qkv.weight"),
            format!("{prefix}.self_attention.qkv_proj.weight"),
            format!("{prefix}.attention.qkv.weight"),
            format!("{prefix}.attention.qkv_proj.weight"),
            format!("{prefix}.attn.qkv.weight"),
            format!("{prefix}.attn.qkv_proj.weight"),
            format!("{prefix}.query_key_value.weight"),
            format!("{prefix}.self_attn.query_key_value.weight"),
            format!("{prefix}.self_attention.query_key_value.weight"),
            format!("{prefix}.attention.query_key_value.weight"),
            format!("{prefix}.attn.query_key_value.weight"),
            format!("{prefix}.mixer.query_key_value.weight"),
            format!("{prefix}.Wqkv.weight"),
            format!("{prefix}.self_attn.Wqkv.weight"),
            format!("{prefix}.self_attention.Wqkv.weight"),
            format!("{prefix}.attention.Wqkv.weight"),
            format!("{prefix}.attn.Wqkv.weight"),
            format!("{prefix}.wqkv.weight"),
            format!("{prefix}.self_attn.wqkv.weight"),
            format!("{prefix}.self_attention.wqkv.weight"),
            format!("{prefix}.attention.wqkv.weight"),
            format!("{prefix}.attn.wqkv.weight"),
            format!("{prefix}.W_pack.weight"),
            format!("{prefix}.self_attn.W_pack.weight"),
            format!("{prefix}.self_attention.W_pack.weight"),
            format!("{prefix}.attention.W_pack.weight"),
            format!("{prefix}.attn.W_pack.weight"),
            format!("{prefix}.c_attn.weight"),
            format!("{prefix}.attn.c_attn.weight"),
            format!("{prefix}.attention.c_attn.weight"),
            format!("{prefix}.self_attn.c_attn.weight"),
            format!("{prefix}.self_attention.c_attn.weight"),
            format!("{prefix}.mixer.c_attn.weight"),
            format!("{prefix}.mixer.W_pack.weight"),
            format!("{prefix}.mixer.w_pack.weight"),
            format!("{prefix}.mixer.Wqkv.weight"),
            format!("{prefix}.mixer.wqkv.weight"),
        ]
    })
}

pub fn qwen_dense_packed_qkv_bias_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_qkv.bias"),
            format!("{prefix}.qkv.bias"),
            format!("{prefix}.qkv_proj.bias"),
            format!("{prefix}.self_attn.qkv.bias"),
            format!("{prefix}.self_attn.qkv_proj.bias"),
            format!("{prefix}.self_attention.qkv.bias"),
            format!("{prefix}.self_attention.qkv_proj.bias"),
            format!("{prefix}.attention.qkv.bias"),
            format!("{prefix}.attention.qkv_proj.bias"),
            format!("{prefix}.attn.qkv.bias"),
            format!("{prefix}.attn.qkv_proj.bias"),
            format!("{prefix}.query_key_value.bias"),
            format!("{prefix}.self_attn.query_key_value.bias"),
            format!("{prefix}.self_attention.query_key_value.bias"),
            format!("{prefix}.attention.query_key_value.bias"),
            format!("{prefix}.attn.query_key_value.bias"),
            format!("{prefix}.mixer.query_key_value.bias"),
            format!("{prefix}.Wqkv.bias"),
            format!("{prefix}.self_attn.Wqkv.bias"),
            format!("{prefix}.self_attention.Wqkv.bias"),
            format!("{prefix}.attention.Wqkv.bias"),
            format!("{prefix}.attn.Wqkv.bias"),
            format!("{prefix}.wqkv.bias"),
            format!("{prefix}.self_attn.wqkv.bias"),
            format!("{prefix}.self_attention.wqkv.bias"),
            format!("{prefix}.attention.wqkv.bias"),
            format!("{prefix}.attn.wqkv.bias"),
            format!("{prefix}.W_pack.bias"),
            format!("{prefix}.self_attn.W_pack.bias"),
            format!("{prefix}.self_attention.W_pack.bias"),
            format!("{prefix}.attention.W_pack.bias"),
            format!("{prefix}.attn.W_pack.bias"),
            format!("{prefix}.c_attn.bias"),
            format!("{prefix}.attn.c_attn.bias"),
            format!("{prefix}.attention.c_attn.bias"),
            format!("{prefix}.self_attn.c_attn.bias"),
            format!("{prefix}.self_attention.c_attn.bias"),
            format!("{prefix}.mixer.c_attn.bias"),
            format!("{prefix}.mixer.W_pack.bias"),
            format!("{prefix}.mixer.w_pack.bias"),
            format!("{prefix}.mixer.Wqkv.bias"),
            format!("{prefix}.mixer.wqkv.bias"),
        ]
    })
}

pub fn qwen_mla_q_a_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_q_a.weight"),
            format!("{prefix}.attn_q_a_proj.weight"),
            format!("{prefix}.self_attn.q_a_proj.weight"),
            format!("{prefix}.self_attention.q_a_proj.weight"),
            format!("{prefix}.attention.q_a_proj.weight"),
        ]
    })
}

pub fn qwen_mla_q_a_norm_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_q_a_norm.weight"),
            format!("{prefix}.attn_q_a_layernorm.weight"),
            format!("{prefix}.self_attn.q_a_layernorm.weight"),
            format!("{prefix}.self_attn.q_a_norm.weight"),
            format!("{prefix}.self_attention.q_a_layernorm.weight"),
            format!("{prefix}.self_attention.q_a_norm.weight"),
            format!("{prefix}.attention.q_a_layernorm.weight"),
            format!("{prefix}.attention.q_a_norm.weight"),
        ]
    })
}

pub fn qwen_mla_q_b_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_q_b.weight"),
            format!("{prefix}.attn_q_b_proj.weight"),
            format!("{prefix}.self_attn.q_b_proj.weight"),
            format!("{prefix}.self_attention.q_b_proj.weight"),
            format!("{prefix}.attention.q_b_proj.weight"),
        ]
    })
}

pub fn qwen_mla_kv_a_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_kv_a_mqa.weight"),
            format!("{prefix}.attn_kv_a_proj_with_mqa.weight"),
            format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
            format!("{prefix}.self_attn.kv_a_proj.weight"),
            format!("{prefix}.self_attention.kv_a_proj_with_mqa.weight"),
            format!("{prefix}.self_attention.kv_a_proj.weight"),
            format!("{prefix}.attention.kv_a_proj_with_mqa.weight"),
            format!("{prefix}.attention.kv_a_proj.weight"),
        ]
    })
}

pub fn qwen_mla_kv_a_norm_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_kv_a_norm.weight"),
            format!("{prefix}.attn_kv_a_layernorm.weight"),
            format!("{prefix}.self_attn.kv_a_layernorm.weight"),
            format!("{prefix}.self_attn.kv_a_norm.weight"),
            format!("{prefix}.self_attention.kv_a_layernorm.weight"),
            format!("{prefix}.self_attention.kv_a_norm.weight"),
            format!("{prefix}.attention.kv_a_layernorm.weight"),
            format!("{prefix}.attention.kv_a_norm.weight"),
        ]
    })
}

pub fn qwen_mla_kv_b_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_kv_b.weight"),
            format!("{prefix}.attn_kv_b_proj.weight"),
            format!("{prefix}.self_attn.kv_b_proj.weight"),
            format!("{prefix}.self_attention.kv_b_proj.weight"),
            format!("{prefix}.attention.kv_b_proj.weight"),
        ]
    })
}

/// DeepSeek-V3.2/GLM-5-class MLA ships the kv_b projection split into
/// per-head `attn_k_b` (stored transposed, for weight absorption) and
/// `attn_v_b` rank-3 tensors instead of one fused `attn_kv_b`.
pub fn qwen_mla_k_b_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_k_b.weight"),
            format!("{prefix}.attn_k_b_proj.weight"),
            format!("{prefix}.self_attn.k_b_proj.weight"),
            format!("{prefix}.self_attention.k_b_proj.weight"),
            format!("{prefix}.attention.k_b_proj.weight"),
        ]
    })
}

pub fn qwen_mla_v_b_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_v_b.weight"),
            format!("{prefix}.attn_v_b_proj.weight"),
            format!("{prefix}.self_attn.v_b_proj.weight"),
            format!("{prefix}.self_attention.v_b_proj.weight"),
            format!("{prefix}.attention.v_b_proj.weight"),
        ]
    })
}

fn qwen_mla_split_kv_b_present(gguf: &GgufFile, prefix: &str) -> bool {
    qwen_mla_k_b_weight_names(prefix)
        .iter()
        .any(|name| gguf.tensor(name).is_some())
        && qwen_mla_v_b_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
}

pub fn qwen_mla_attention_tensors_present(gguf: &GgufFile, prefix: &str) -> bool {
    let kv_latent_present = qwen_mla_kv_a_weight_names(prefix)
        .iter()
        .any(|name| gguf.tensor(name).is_some())
        && qwen_mla_kv_a_norm_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
        && (qwen_mla_kv_b_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
            || qwen_mla_split_kv_b_present(gguf, prefix));
    if !kv_latent_present {
        return false;
    }
    // Q-LoRA query decomposition (DeepSeek-V2/V3, GLM): q_a -> q_a_norm -> q_b.
    let q_lora_present = qwen_mla_q_a_weight_names(prefix)
        .iter()
        .any(|name| gguf.tensor(name).is_some())
        && qwen_mla_q_a_norm_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some())
        && qwen_mla_q_b_weight_names(prefix)
            .iter()
            .any(|name| gguf.tensor(name).is_some());
    // Full-Q MLA (DeepSeek-V2-Lite class): a single dense attn_q projection with
    // no Q-LoRA, still feeding the compressed KV latent above. The kv_latent
    // guard keeps plain dense attention (attn_q + attn_k + attn_v, no kv_a) out.
    let full_q_present = qwen_dense_attention_weight_names(prefix, "q")
        .iter()
        .any(|name| gguf.tensor(name).is_some());
    q_lora_present || full_q_present
}

pub fn qwen_ssm_in_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_in.weight"),
            format!("{prefix}.linear_attn.in_proj_qkvz.weight"),
            format!("{prefix}.gated_delta.in_proj_qkvz.weight"),
        ]
    })
}

pub fn qwen_ssm_qkv_weight_names(prefix: &str) -> Vec<String> {
    qwen_dense_packed_qkv_weight_names(prefix)
}

pub fn qwen_ssm_gate_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_gate.weight"),
            format!("{prefix}.wqkv_gate.weight"),
            format!("{prefix}.linear_attn.gate_proj.weight"),
            format!("{prefix}.linear_attn.in_proj_gate.weight"),
            format!("{prefix}.gated_delta.gate_proj.weight"),
        ]
    })
}

pub fn qwen_ssm_conv1d_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_conv1d.weight"),
            format!("{prefix}.conv1d.weight"),
            format!("{prefix}.linear_attn.conv1d.weight"),
            format!("{prefix}.gated_delta.conv1d.weight"),
        ]
    })
}

pub fn qwen_ssm_dt_bias_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_dt.bias"),
            format!("{prefix}.dt_bias"),
            format!("{prefix}.linear_attn.dt_bias"),
            format!("{prefix}.gated_delta.dt_bias"),
        ]
    })
}

pub fn qwen_ssm_a_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_a"),
            format!("{prefix}.ssm_a.weight"),
            format!("{prefix}.A_log"),
            format!("{prefix}.linear_attn.A_log"),
            format!("{prefix}.gated_delta.A_log"),
        ]
    })
}

pub fn qwen_ssm_ba_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_ba.weight"),
            format!("{prefix}.ssm_beta_alpha.weight"),
            format!("{prefix}.linear_attn.in_proj_ba.weight"),
            format!("{prefix}.gated_delta.in_proj_ba.weight"),
        ]
    })
}

/// Qwen3.5 splits the fused beta/alpha (`ssm_ba`) projection into separate
/// `ssm_beta` / `ssm_alpha` tensors of `time_step_rank` columns each.
pub fn qwen_ssm_beta_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_beta.weight"),
            format!("{prefix}.linear_attn.in_proj_beta.weight"),
            format!("{prefix}.linear_attn.beta_proj.weight"),
            format!("{prefix}.gated_delta.beta_proj.weight"),
        ]
    })
}

pub fn qwen_ssm_alpha_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_alpha.weight"),
            format!("{prefix}.linear_attn.in_proj_alpha.weight"),
            format!("{prefix}.linear_attn.alpha_proj.weight"),
            format!("{prefix}.gated_delta.alpha_proj.weight"),
        ]
    })
}

pub fn qwen_ssm_norm_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_norm.weight"),
            format!("{prefix}.linear_attn.norm.weight"),
            format!("{prefix}.gated_delta.norm.weight"),
        ]
    })
}

pub fn qwen_ssm_out_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ssm_out.weight"),
            format!("{prefix}.linear_attn.out_proj.weight"),
            format!("{prefix}.gated_delta.out_proj.weight"),
        ]
    })
}

pub fn qwen_ssm_layer_tensors_present(gguf: &GgufFile, prefix: &str) -> bool {
    qwen_ssm_layer_tensors_present_with(|name| gguf.tensor(name).is_some(), prefix)
}

pub fn qwen_dense_packed_ffn_gate_up_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_gate_up.weight"),
            format!("{prefix}.gate_up_proj.weight"),
            format!("{prefix}.mlp.gate_up_proj.weight"),
            format!("{prefix}.ffn.gate_up_proj.weight"),
            format!("{prefix}.feed_forward.gate_up_proj.weight"),
            format!("{prefix}.feed_forward.mlp.gate_up_proj.weight"),
            format!("{prefix}.gate_up.weight"),
            format!("{prefix}.mlp.gate_up.weight"),
            format!("{prefix}.ffn.gate_up.weight"),
            format!("{prefix}.feed_forward.gate_up.weight"),
            format!("{prefix}.feed_forward.mlp.gate_up.weight"),
            format!("{prefix}.fc1.weight"),
            format!("{prefix}.mlp.fc1.weight"),
            format!("{prefix}.ffn.fc1.weight"),
            format!("{prefix}.feed_forward.fc1.weight"),
            format!("{prefix}.feed_forward.mlp.fc1.weight"),
            format!("{prefix}.dense_h_to_4h.weight"),
            format!("{prefix}.mlp.dense_h_to_4h.weight"),
            format!("{prefix}.ffn.dense_h_to_4h.weight"),
            format!("{prefix}.feed_forward.dense_h_to_4h.weight"),
            format!("{prefix}.feed_forward.mlp.dense_h_to_4h.weight"),
            format!("{prefix}.c_fc.weight"),
            format!("{prefix}.mlp.c_fc.weight"),
            format!("{prefix}.ffn.c_fc.weight"),
            format!("{prefix}.feed_forward.c_fc.weight"),
            format!("{prefix}.feed_forward.mlp.c_fc.weight"),
            format!("{prefix}.w1w3.weight"),
            format!("{prefix}.mlp.w1w3.weight"),
            format!("{prefix}.ffn.w1w3.weight"),
            format!("{prefix}.feed_forward.w1w3.weight"),
            format!("{prefix}.feed_forward.mlp.w1w3.weight"),
        ]
    })
}

pub fn qwen_dense_packed_ffn_gate_up_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_dense_packed_ffn_gate_up_weight_names(prefix))
}

pub fn qwen_dense_packed_ffn_up_gate_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_up_gate.weight"),
            format!("{prefix}.up_gate_proj.weight"),
            format!("{prefix}.mlp.up_gate_proj.weight"),
            format!("{prefix}.ffn.up_gate_proj.weight"),
            format!("{prefix}.feed_forward.up_gate_proj.weight"),
            format!("{prefix}.feed_forward.mlp.up_gate_proj.weight"),
            format!("{prefix}.up_gate.weight"),
            format!("{prefix}.mlp.up_gate.weight"),
            format!("{prefix}.ffn.up_gate.weight"),
            format!("{prefix}.feed_forward.up_gate.weight"),
            format!("{prefix}.feed_forward.mlp.up_gate.weight"),
            format!("{prefix}.w3w1.weight"),
            format!("{prefix}.mlp.w3w1.weight"),
            format!("{prefix}.ffn.w3w1.weight"),
            format!("{prefix}.feed_forward.w3w1.weight"),
            format!("{prefix}.feed_forward.mlp.w3w1.weight"),
        ]
    })
}

pub fn qwen_dense_packed_ffn_up_gate_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_dense_packed_ffn_up_gate_weight_names(prefix))
}

pub fn qwen_phi_packed_qkv_weight_names(prefix: &str) -> Vec<String> {
    qwen_dense_packed_qkv_weight_names(prefix)
}

pub fn qwen_phi_packed_qkv_bias_names(prefix: &str) -> Vec<String> {
    qwen_dense_packed_qkv_bias_names(prefix)
}

pub fn qwen_phi_packed_ffn_gate_up_weight_names(prefix: &str) -> Vec<String> {
    qwen_dense_packed_ffn_gate_up_weight_names(prefix)
}

pub fn qwen_phi_packed_ffn_up_gate_weight_names(prefix: &str) -> Vec<String> {
    qwen_dense_packed_ffn_up_gate_weight_names(prefix)
}

pub fn qwen_dense_attention_head_norm_weight_names(prefix: &str, suffix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.attn_{suffix}_norm.weight"),
            format!("{prefix}.attn_{suffix}_layernorm.weight"),
            format!("{prefix}.attn_{suffix}_layer_norm.weight"),
            format!("{prefix}.{suffix}_norm.weight"),
            format!("{prefix}.{suffix}_layernorm.weight"),
            format!("{prefix}.{suffix}_layer_norm.weight"),
            format!("{prefix}.self_attn.{suffix}_norm.weight"),
            format!("{prefix}.self_attn.{suffix}_layernorm.weight"),
            format!("{prefix}.self_attn.{suffix}_layer_norm.weight"),
            format!("{prefix}.self_attention.{suffix}_norm.weight"),
            format!("{prefix}.self_attention.{suffix}_layernorm.weight"),
            format!("{prefix}.self_attention.{suffix}_layer_norm.weight"),
            format!("{prefix}.attention.{suffix}_norm.weight"),
            format!("{prefix}.attention.{suffix}_layernorm.weight"),
            format!("{prefix}.attention.{suffix}_layer_norm.weight"),
        ]
    })
}

pub fn qwen_dense_ffn_weight_names(prefix: &str, kind: &str) -> Vec<String> {
    let (canonical, hf, llama) = match kind {
        "gate" => ("ffn_gate", "gate_proj", "w1"),
        "up" => ("ffn_up", "up_proj", "w3"),
        "down" => ("ffn_down", "down_proj", "w2"),
        _ => {
            return layer_prefix_aliases(prefix, |prefix| {
                vec![format!("{prefix}.ffn_{kind}.weight")]
            });
        }
    };
    layer_prefix_aliases(prefix, |prefix| {
        let mut names = vec![
            format!("{prefix}.{canonical}.weight"),
            format!("{prefix}.{hf}.weight"),
            format!("{prefix}.mlp.{hf}.weight"),
            format!("{prefix}.ffn.{hf}.weight"),
            format!("{prefix}.feed_forward.{hf}.weight"),
            format!("{prefix}.feed_forward.mlp.{hf}.weight"),
            format!("{prefix}.feed_forward.{llama}.weight"),
            format!("{prefix}.mlp.{llama}.weight"),
            format!("{prefix}.ffn.{llama}.weight"),
            format!("{prefix}.feed_forward.mlp.{llama}.weight"),
            format!("{prefix}.{llama}.weight"),
        ];
        if kind == "down" {
            names.extend([
                format!("{prefix}.dense_4h_to_h.weight"),
                format!("{prefix}.mlp.dense_4h_to_h.weight"),
                format!("{prefix}.ffn.dense_4h_to_h.weight"),
                format!("{prefix}.feed_forward.dense_4h_to_h.weight"),
                format!("{prefix}.feed_forward.mlp.dense_4h_to_h.weight"),
                format!("{prefix}.fc2.weight"),
                format!("{prefix}.mlp.fc2.weight"),
                format!("{prefix}.ffn.fc2.weight"),
                format!("{prefix}.feed_forward.fc2.weight"),
                format!("{prefix}.feed_forward.mlp.fc2.weight"),
                format!("{prefix}.mlp.c_proj.weight"),
                format!("{prefix}.ffn.c_proj.weight"),
                format!("{prefix}.feed_forward.c_proj.weight"),
                format!("{prefix}.feed_forward.mlp.c_proj.weight"),
                format!("{prefix}.mlp.proj.weight"),
                format!("{prefix}.ffn.proj.weight"),
                format!("{prefix}.feed_forward.proj.weight"),
                format!("{prefix}.feed_forward.mlp.proj.weight"),
            ]);
        }
        names
    })
}

pub fn qwen_dense_ffn_bias_names(prefix: &str, kind: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_dense_ffn_weight_names(prefix, kind))
}

fn weight_aliases_to_bias_names(names: Vec<String>) -> Vec<String> {
    names
        .into_iter()
        .map(|name| {
            name.strip_suffix(".weight")
                .map(|prefix| format!("{prefix}.bias"))
                .unwrap_or_else(|| format!("{name}.bias"))
        })
        .collect()
}

pub fn qwen_moe_router_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_gate_inp.weight"),
            format!("{prefix}.router.weight"),
            format!("{prefix}.gate.weight"),
            format!("{prefix}.mlp.router.weight"),
            format!("{prefix}.mlp.gate.weight"),
            format!("{prefix}.mlp.moe.router.weight"),
            format!("{prefix}.mlp.moe.gate.weight"),
            format!("{prefix}.mlp.block_sparse_moe.router.weight"),
            format!("{prefix}.mlp.block_sparse_moe.gate.weight"),
            format!("{prefix}.moe.router.weight"),
            format!("{prefix}.moe.gate.weight"),
            format!("{prefix}.block_sparse_moe.router.weight"),
            format!("{prefix}.block_sparse_moe.gate.weight"),
            format!("{prefix}.feed_forward.router.weight"),
            format!("{prefix}.feed_forward.gate.weight"),
            format!("{prefix}.feed_forward.moe.router.weight"),
            format!("{prefix}.feed_forward.moe.gate.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.router.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.gate.weight"),
        ]
    })
}

pub fn qwen_moe_router_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_router_weight_names(prefix))
}

/// DeepSeek/GLM expert-selection correction bias (`e_score_correction_bias`):
/// added to gating scores for TOP-K SELECTION ONLY — the routed weight uses
/// the bias-free score. Distinct from the additive router bias above.
pub fn qwen_moe_selection_bias_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.exp_probs_b.bias"),
            format!("{prefix}.mlp.gate.e_score_correction_bias"),
            format!("{prefix}.mlp.gate.e_score_correction.bias"),
        ]
    })
}

pub fn qwen_moe_packed_expert_weight_names(prefix: &str, kind: &str) -> Vec<String> {
    let (canonical, hf, llama) = match kind {
        "gate" => ("ffn_gate_exps", "gate_proj", "w1"),
        "up" => ("ffn_up_exps", "up_proj", "w3"),
        "down" => ("ffn_down_exps", "down_proj", "w2"),
        _ => {
            return layer_prefix_aliases(prefix, |prefix| {
                vec![format!("{prefix}.ffn_{kind}_exps.weight")]
            });
        }
    };
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.{canonical}.weight"),
            format!("{prefix}.experts.{hf}.weight"),
            format!("{prefix}.mlp.experts.{hf}.weight"),
            format!("{prefix}.mlp.moe.experts.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{hf}.weight"),
            format!("{prefix}.moe.experts.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{hf}.weight"),
            format!("{prefix}.feed_forward.experts.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{hf}.weight"),
            format!("{prefix}.experts.{llama}.weight"),
            format!("{prefix}.mlp.experts.{llama}.weight"),
            format!("{prefix}.mlp.moe.experts.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{llama}.weight"),
            format!("{prefix}.moe.experts.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{llama}.weight"),
            format!("{prefix}.feed_forward.experts.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{llama}.weight"),
        ]
    })
}

pub fn qwen_moe_packed_expert_bias_names(prefix: &str, kind: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_packed_expert_weight_names(prefix, kind))
}

pub fn qwen_moe_packed_expert_gate_up_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_gate_up_exps.weight"),
            format!("{prefix}.experts.gate_up_proj.weight"),
            format!("{prefix}.mlp.experts.gate_up_proj.weight"),
            format!("{prefix}.mlp.moe.experts.gate_up_proj.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.gate_up_proj.weight"),
            format!("{prefix}.moe.experts.gate_up_proj.weight"),
            format!("{prefix}.block_sparse_moe.experts.gate_up_proj.weight"),
            format!("{prefix}.feed_forward.experts.gate_up_proj.weight"),
            format!("{prefix}.feed_forward.moe.experts.gate_up_proj.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.gate_up_proj.weight"),
            format!("{prefix}.experts.w1w3.weight"),
            format!("{prefix}.mlp.experts.w1w3.weight"),
            format!("{prefix}.mlp.moe.experts.w1w3.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.w1w3.weight"),
            format!("{prefix}.moe.experts.w1w3.weight"),
            format!("{prefix}.block_sparse_moe.experts.w1w3.weight"),
            format!("{prefix}.feed_forward.experts.w1w3.weight"),
            format!("{prefix}.feed_forward.moe.experts.w1w3.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.w1w3.weight"),
        ]
    })
}

pub fn qwen_moe_packed_expert_gate_up_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_packed_expert_gate_up_weight_names(prefix))
}

pub fn qwen_moe_packed_expert_up_gate_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_up_gate_exps.weight"),
            format!("{prefix}.experts.up_gate_proj.weight"),
            format!("{prefix}.mlp.experts.up_gate_proj.weight"),
            format!("{prefix}.mlp.moe.experts.up_gate_proj.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.up_gate_proj.weight"),
            format!("{prefix}.moe.experts.up_gate_proj.weight"),
            format!("{prefix}.block_sparse_moe.experts.up_gate_proj.weight"),
            format!("{prefix}.feed_forward.experts.up_gate_proj.weight"),
            format!("{prefix}.feed_forward.moe.experts.up_gate_proj.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.up_gate_proj.weight"),
            format!("{prefix}.experts.w3w1.weight"),
            format!("{prefix}.mlp.experts.w3w1.weight"),
            format!("{prefix}.mlp.moe.experts.w3w1.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.w3w1.weight"),
            format!("{prefix}.moe.experts.w3w1.weight"),
            format!("{prefix}.block_sparse_moe.experts.w3w1.weight"),
            format!("{prefix}.feed_forward.experts.w3w1.weight"),
            format!("{prefix}.feed_forward.moe.experts.w3w1.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.w3w1.weight"),
        ]
    })
}

pub fn qwen_moe_packed_expert_up_gate_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_packed_expert_up_gate_weight_names(prefix))
}

pub fn qwen_moe_per_expert_gate_up_weight_names(prefix: &str, expert: u64) -> Vec<String> {
    qwen_moe_per_expert_packed_gate_up_weight_names(
        prefix,
        expert,
        "ffn_gate_up",
        "gate_up_proj",
        "w1w3",
    )
}

pub fn qwen_moe_per_expert_gate_up_bias_names(prefix: &str, expert: u64) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_per_expert_gate_up_weight_names(prefix, expert))
}

pub fn qwen_moe_per_expert_up_gate_weight_names(prefix: &str, expert: u64) -> Vec<String> {
    qwen_moe_per_expert_packed_gate_up_weight_names(
        prefix,
        expert,
        "ffn_up_gate",
        "up_gate_proj",
        "w3w1",
    )
}

pub fn qwen_moe_per_expert_up_gate_bias_names(prefix: &str, expert: u64) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_per_expert_up_gate_weight_names(prefix, expert))
}

fn qwen_moe_per_expert_packed_gate_up_weight_names(
    prefix: &str,
    expert: u64,
    canonical: &str,
    hf: &str,
    llama: &str,
) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.{canonical}.{expert}.weight"),
            format!("{prefix}.{canonical}_exps.{expert}.weight"),
            format!("{prefix}.experts.{expert}.{hf}.weight"),
            format!("{prefix}.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.experts.{expert}.{llama}.weight"),
            format!("{prefix}.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{llama}.{expert}.weight"),
        ]
    })
}

pub fn qwen_moe_per_expert_weight_names(prefix: &str, kind: &str, expert: u64) -> Vec<String> {
    let (canonical, hf, llama) = match kind {
        "gate" => ("ffn_gate", "gate_proj", "w1"),
        "up" => ("ffn_up", "up_proj", "w3"),
        "down" => ("ffn_down", "down_proj", "w2"),
        _ => {
            return layer_prefix_aliases(prefix, |prefix| {
                vec![format!("{prefix}.ffn_{kind}.{expert}.weight")]
            });
        }
    };
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.{canonical}.{expert}.weight"),
            format!("{prefix}.experts.{expert}.{hf}.weight"),
            format!("{prefix}.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{expert}.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{hf}.{expert}.weight"),
            format!("{prefix}.experts.{expert}.{llama}.weight"),
            format!("{prefix}.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.experts.{llama}.{expert}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{expert}.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.experts.{llama}.{expert}.weight"),
        ]
    })
}

pub fn qwen_moe_per_expert_bias_names(prefix: &str, kind: &str, expert: u64) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_per_expert_weight_names(prefix, kind, expert))
}

pub fn qwen_moe_shared_expert_weight_names(prefix: &str, kind: &str) -> Vec<String> {
    let (canonical, hf, llama) = match kind {
        "gate" => ("ffn_gate_shexp", "gate_proj", "w1"),
        "up" => ("ffn_up_shexp", "up_proj", "w3"),
        "down" => ("ffn_down_shexp", "down_proj", "w2"),
        _ => {
            return layer_prefix_aliases(prefix, |prefix| {
                vec![format!("{prefix}.ffn_{kind}_shexp.weight")]
            });
        }
    };
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.{canonical}.weight"),
            format!("{prefix}.shared_expert.{hf}.weight"),
            format!("{prefix}.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.shared_expert.{llama}.weight"),
            format!("{prefix}.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_experts.{llama}.weight"),
            format!("{prefix}.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_experts.{llama}.weight"),
        ]
    })
}

pub fn qwen_moe_shared_expert_bias_names(prefix: &str, kind: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_shared_expert_weight_names(prefix, kind))
}

pub fn qwen_moe_shared_expert_gate_weight_names(prefix: &str) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.ffn_gate_inp_shexp.weight"),
            format!("{prefix}.shared_expert_gate.weight"),
            format!("{prefix}.shared_experts_gate.weight"),
            format!("{prefix}.mlp.shared_expert_gate.weight"),
            format!("{prefix}.mlp.shared_experts_gate.weight"),
            format!("{prefix}.mlp.moe.shared_expert_gate.weight"),
            format!("{prefix}.mlp.moe.shared_experts_gate.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_expert_gate.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_experts_gate.weight"),
            format!("{prefix}.moe.shared_expert_gate.weight"),
            format!("{prefix}.moe.shared_experts_gate.weight"),
            format!("{prefix}.block_sparse_moe.shared_expert_gate.weight"),
            format!("{prefix}.block_sparse_moe.shared_experts_gate.weight"),
            format!("{prefix}.feed_forward.shared_expert_gate.weight"),
            format!("{prefix}.feed_forward.shared_experts_gate.weight"),
            format!("{prefix}.feed_forward.moe.shared_expert_gate.weight"),
            format!("{prefix}.feed_forward.moe.shared_experts_gate.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_expert_gate.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_experts_gate.weight"),
        ]
    })
}

pub fn qwen_moe_shared_expert_gate_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_shared_expert_gate_weight_names(prefix))
}

pub fn qwen_moe_shared_expert_gate_up_weight_names(prefix: &str) -> Vec<String> {
    qwen_moe_shared_expert_packed_gate_up_weight_names(
        prefix,
        "ffn_gate_up_shexp",
        "gate_up_proj",
        "w1w3",
    )
}

pub fn qwen_moe_shared_expert_gate_up_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_shared_expert_gate_up_weight_names(prefix))
}

pub fn qwen_moe_shared_expert_up_gate_weight_names(prefix: &str) -> Vec<String> {
    qwen_moe_shared_expert_packed_gate_up_weight_names(
        prefix,
        "ffn_up_gate_shexp",
        "up_gate_proj",
        "w3w1",
    )
}

pub fn qwen_moe_shared_expert_up_gate_bias_names(prefix: &str) -> Vec<String> {
    weight_aliases_to_bias_names(qwen_moe_shared_expert_up_gate_weight_names(prefix))
}

fn qwen_moe_shared_expert_packed_gate_up_weight_names(
    prefix: &str,
    canonical: &str,
    hf: &str,
    llama: &str,
) -> Vec<String> {
    layer_prefix_aliases(prefix, |prefix| {
        vec![
            format!("{prefix}.{canonical}.weight"),
            format!("{prefix}.shared_expert.{hf}.weight"),
            format!("{prefix}.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.moe.shared_experts.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_expert.{hf}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_experts.{hf}.weight"),
            format!("{prefix}.shared_expert.{llama}.weight"),
            format!("{prefix}.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.mlp.block_sparse_moe.shared_experts.{llama}.weight"),
            format!("{prefix}.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.block_sparse_moe.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.moe.shared_experts.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_expert.{llama}.weight"),
            format!("{prefix}.feed_forward.block_sparse_moe.shared_experts.{llama}.weight"),
        ]
    })
}

fn layer_prefix_aliases<F>(prefix: &str, build: F) -> Vec<String>
where
    F: Fn(&str) -> Vec<String>,
{
    let mut names = Vec::new();
    for prefix in layer_prefix_variants(prefix) {
        for name in build(&prefix) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

fn layer_prefix_variants(prefix: &str) -> Vec<String> {
    let mut prefixes = vec![prefix.to_string()];
    let Some(layer) = prefix.strip_prefix("blk.") else {
        return prefixes;
    };
    if layer.is_empty() || layer.contains('.') {
        return prefixes;
    }
    for alias in [
        format!("model.layers.{layer}"),
        format!("language_model.model.layers.{layer}"),
        format!("language_model.layers.{layer}"),
        format!("layers.{layer}"),
        format!("model.decoder.layers.{layer}"),
        format!("language_model.model.decoder.layers.{layer}"),
        format!("language_model.decoder.layers.{layer}"),
        format!("decoder.layers.{layer}"),
        format!("encoder.layers.{layer}"),
        format!("transformer.encoder.layers.{layer}"),
        format!("transformer.decoder.layers.{layer}"),
        format!("transformer.layers.{layer}"),
        format!("transformer.blocks.{layer}"),
        format!("model.transformer.layers.{layer}"),
        format!("model.transformer.blocks.{layer}"),
        format!("transformer.h.{layer}"),
        format!("model.transformer.h.{layer}"),
        format!("h.{layer}"),
        format!("model.h.{layer}"),
        format!("gpt_neox.layers.{layer}"),
        format!("model.gpt_neox.layers.{layer}"),
    ] {
        if !prefixes.contains(&alias) {
            prefixes.push(alias);
        }
    }
    prefixes
}

fn attention_projection_alias_parts(
    suffix: &str,
) -> Option<(&'static str, &'static str, &'static str)> {
    match suffix {
        "q" => Some(("attn_q", "q_proj", "wq")),
        "k" => Some(("attn_k", "k_proj", "wk")),
        "v" => Some(("attn_v", "v_proj", "wv")),
        "output" => Some(("attn_output", "o_proj", "wo")),
        _ => None,
    }
}

fn qwen_ssm_metadata_feature(key: &str) -> Option<&'static str> {
    let lower = key.to_ascii_lowercase();
    if lower.contains("ssm") {
        Some("unsupported feature SSM")
    } else if lower.contains("mamba") {
        Some("unsupported feature Mamba/SSM")
    } else if lower.contains("delta") {
        Some("unsupported feature DeltaNet/recurrent decoder")
    } else if lower.contains("conv1d") || lower.contains("conv_1d") {
        Some("unsupported feature SSM convolution")
    } else if lower.contains("recurrent") {
        Some("unsupported feature recurrent decoder")
    } else {
        None
    }
}

fn qwen_ssm_tensor_feature(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("ssm") {
        Some("unsupported feature SSM")
    } else if lower.contains("mamba") {
        Some("unsupported feature Mamba/SSM")
    } else if lower.contains("delta") {
        Some("unsupported feature DeltaNet/recurrent decoder")
    } else if lower.contains("conv1d") || lower.contains("conv_1d") {
        Some("unsupported feature SSM convolution")
    } else if lower.contains("time_mix") || lower.contains("time-mix") {
        Some("unsupported feature time-mix recurrent decoder")
    } else if lower.contains("recurrent") {
        Some("unsupported feature recurrent decoder")
    } else {
        None
    }
}

fn qwen_attention_recurrent_layers(
    gguf: &GgufFile,
    prefix: &str,
    block_count: u32,
) -> Result<Option<Vec<bool>>> {
    let key = format!("{prefix}.attention.recurrent_layers");
    let Some(value) = gguf.metadata.get(&key) else {
        return Ok(None);
    };
    let MetadataValue::Array(values) = value else {
        bail!("GGUF metadata {key} must be an array of booleans or integers");
    };
    if values.len()
        != usize::try_from(block_count).context("qwen block_count does not fit usize")?
    {
        bail!(
            "GGUF metadata {key} must contain {block_count} entries, got {}",
            values.len()
        );
    }
    values
        .iter()
        .map(|value| match value {
            MetadataValue::Bool(value) => Ok(*value),
            MetadataValue::Uint8(value) => Ok(*value != 0),
            MetadataValue::Int8(value) => Ok(*value != 0),
            MetadataValue::Uint16(value) => Ok(*value != 0),
            MetadataValue::Int16(value) => Ok(*value != 0),
            MetadataValue::Uint32(value) => Ok(*value != 0),
            MetadataValue::Int32(value) => Ok(*value != 0),
            MetadataValue::Uint64(value) => Ok(*value != 0),
            MetadataValue::Int64(value) => Ok(*value != 0),
            _ => bail!("GGUF metadata {key} must be an array of booleans or integers"),
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// Optional per-layer u32 array (e.g. gemma4's `attention.head_count_kv` /
/// `feed_forward_length`, which are arrays when the value varies by layer).
fn qwen_metadata_u32_array(
    gguf: &GgufFile,
    key: &str,
    expected_len: u32,
) -> Result<Option<Vec<u32>>> {
    let Some(values) = qwen_metadata_u32_array_lenient(gguf, key)? else {
        return Ok(None);
    };
    if values.len()
        != usize::try_from(expected_len).context("qwen block_count does not fit usize")?
    {
        bail!(
            "GGUF metadata {key} must contain {expected_len} entries, got {}",
            values.len()
        );
    }
    Ok(Some(values))
}

/// Like `qwen_metadata_u32_array` but without a length requirement — for
/// per-layer arrays that may carry extra trailing entries (e.g. deepseek4's
/// `attention.compress_ratios` includes a slot for the stripped MTP layer).
fn qwen_metadata_u32_array_lenient(gguf: &GgufFile, key: &str) -> Result<Option<Vec<u32>>> {
    let Some(value) = gguf.metadata.get(key) else {
        return Ok(None);
    };
    let MetadataValue::Array(values) = value else {
        // Scalar form is handled by the metadata_u32 caller.
        return Ok(None);
    };
    values
        .iter()
        .map(|value| match value {
            MetadataValue::Uint8(value) => Ok((*value).into()),
            MetadataValue::Int8(value) => {
                u32::try_from(*value).map_err(|_| anyhow!("GGUF metadata {key} entry is negative"))
            }
            MetadataValue::Uint16(value) => Ok((*value).into()),
            MetadataValue::Int16(value) => {
                u32::try_from(*value).map_err(|_| anyhow!("GGUF metadata {key} entry is negative"))
            }
            MetadataValue::Uint32(value) => Ok(*value),
            MetadataValue::Int32(value) => {
                u32::try_from(*value).map_err(|_| anyhow!("GGUF metadata {key} entry is negative"))
            }
            MetadataValue::Uint64(value) => {
                u32::try_from(*value).map_err(|_| anyhow!("GGUF metadata {key} entry exceeds u32"))
            }
            MetadataValue::Int64(value) => u32::try_from(*value)
                .map_err(|_| anyhow!("GGUF metadata {key} entry out of u32 range")),
            _ => bail!("GGUF metadata {key} must be an array of integers"),
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// Optional per-layer bool array (e.g. gemma4's
/// `attention.sliding_window_pattern`).
fn qwen_metadata_bool_array(
    gguf: &GgufFile,
    key: &str,
    expected_len: u32,
) -> Result<Option<Vec<bool>>> {
    let Some(value) = gguf.metadata.get(key) else {
        return Ok(None);
    };
    let MetadataValue::Array(values) = value else {
        bail!("GGUF metadata {key} must be an array of booleans or integers");
    };
    if values.len()
        != usize::try_from(expected_len).context("qwen block_count does not fit usize")?
    {
        bail!(
            "GGUF metadata {key} must contain {expected_len} entries, got {}",
            values.len()
        );
    }
    values
        .iter()
        .map(|value| match value {
            MetadataValue::Bool(value) => Ok(*value),
            MetadataValue::Uint8(value) => Ok(*value != 0),
            MetadataValue::Int8(value) => Ok(*value != 0),
            MetadataValue::Uint16(value) => Ok(*value != 0),
            MetadataValue::Int16(value) => Ok(*value != 0),
            MetadataValue::Uint32(value) => Ok(*value != 0),
            MetadataValue::Int32(value) => Ok(*value != 0),
            MetadataValue::Uint64(value) => Ok(*value != 0),
            MetadataValue::Int64(value) => Ok(*value != 0),
            _ => bail!("GGUF metadata {key} must be an array of booleans or integers"),
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn qwen_ssm_layer_tensors_present_with<F>(has_tensor: F, prefix: &str) -> bool
where
    F: Fn(&str) -> bool,
{
    let legacy_in = qwen_ssm_in_weight_names(prefix)
        .iter()
        .any(|name| has_tensor(name));
    let optimized_in = qwen_ssm_qkv_weight_names(prefix)
        .iter()
        .any(|name| has_tensor(name))
        && qwen_ssm_gate_weight_names(prefix)
            .iter()
            .any(|name| has_tensor(name));
    (legacy_in || optimized_in)
        && qwen_ssm_conv1d_weight_names(prefix)
            .iter()
            .any(|name| has_tensor(name))
        && qwen_ssm_dt_bias_names(prefix)
            .iter()
            .any(|name| has_tensor(name))
        && qwen_ssm_a_names(prefix).iter().any(|name| has_tensor(name))
        && (qwen_ssm_ba_weight_names(prefix)
            .iter()
            .any(|name| has_tensor(name))
            || (qwen_ssm_beta_weight_names(prefix)
                .iter()
                .any(|name| has_tensor(name))
                && qwen_ssm_alpha_weight_names(prefix)
                    .iter()
                    .any(|name| has_tensor(name))))
        && qwen_ssm_norm_weight_names(prefix)
            .iter()
            .any(|name| has_tensor(name))
        && qwen_ssm_out_weight_names(prefix)
            .iter()
            .any(|name| has_tensor(name))
}

fn qwen_ssm_layer_tensors_present_in(tensors: &BTreeMap<&str, &TensorInfo>, prefix: &str) -> bool {
    qwen_ssm_layer_tensors_present_with(|name| tensors.contains_key(name), prefix)
}

#[derive(Clone, Copy)]
struct QwenSsmDims {
    conv_kernel: u64,
    time_step_rank: u64,
    value_dim: u64,
    conv_dim: u64,
    qkvz_dim: u64,
    ba_dim: u64,
    head_v_dim: u64,
}

fn qwen_ssm_dims(config: &QwenGgufConfig, prefix: &str) -> Result<Option<QwenSsmDims>> {
    if !config.recurrent_ssm_tensor_layout {
        return Ok(None);
    }
    let metadata_prefix = &config.architecture;
    let Some(conv_kernel) = config
        .ssm_conv_kernel
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        bail!("SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.conv_kernel");
    };
    let Some(inner_size) = config
        .ssm_inner_size
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        bail!("SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.inner_size");
    };
    let Some(state_size) = config
        .ssm_state_size
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        bail!("SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.state_size");
    };
    let Some(time_step_rank) = config
        .ssm_time_step_rank
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        bail!("SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.time_step_rank");
    };
    let Some(group_count) = config
        .ssm_group_count
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        bail!("SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.group_count");
    };
    if time_step_rank % group_count != 0 {
        bail!(
            "SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.time_step_rank={time_step_rank} to be divisible by {metadata_prefix}.ssm.group_count={group_count}"
        );
    }
    if inner_size % time_step_rank != 0 {
        bail!(
            "SSM tensor layout in {prefix} requires {metadata_prefix}.ssm.inner_size={inner_size} to be divisible by {metadata_prefix}.ssm.time_step_rank={time_step_rank}"
        );
    }
    let head_v_dim = inner_size / time_step_rank;
    let key_dim = state_size
        .checked_mul(group_count)
        .context("SSM key dimension overflows u64")?;
    let value_dim = head_v_dim
        .checked_mul(time_step_rank)
        .context("SSM value dimension overflows u64")?;
    let conv_dim = key_dim
        .checked_mul(2)
        .and_then(|value| value.checked_add(value_dim))
        .context("SSM convolution dimension overflows u64")?;
    let qkvz_dim = key_dim
        .checked_mul(2)
        .and_then(|value| value.checked_add(value_dim.checked_mul(2)?))
        .context("SSM qkvz dimension overflows u64")?;
    let ba_dim = time_step_rank
        .checked_mul(2)
        .context("SSM beta/alpha dimension overflows u64")?;
    Ok(Some(QwenSsmDims {
        conv_kernel,
        time_step_rank,
        value_dim,
        conv_dim,
        qkvz_dim,
        ba_dim,
        head_v_dim,
    }))
}

#[derive(Clone, Debug, Serialize)]
pub struct QwenTensorValidation {
    pub valid: bool,
    pub required_tensors: usize,
    pub optional_tensors_present: usize,
    pub tensor_count: usize,
    pub total_tensor_bytes: u64,
    pub errors: Vec<String>,
}

pub(crate) fn validate_qwen_tensors(
    gguf: &GgufFile,
    config: &QwenGgufConfig,
) -> QwenTensorValidation {
    let tensors = gguf
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let mut validator = QwenTensorValidator {
        tensors,
        errors: Vec::new(),
        required_tensors: 0,
        optional_tensors_present: 0,
    };

    let embed = u64::from(config.embedding_length);
    let vocab = config.vocab_size.map(u64::from);
    let block_count = config.block_count;
    let head_count = u64::from(config.attention_head_count);
    let kv_head_count = u64::from(config.attention_head_count_kv);
    if head_count == 0 {
        validator
            .errors
            .push("qwen attention.head_count must be greater than zero".to_string());
    }
    if kv_head_count == 0 {
        validator
            .errors
            .push("qwen attention.head_count_kv must be greater than zero".to_string());
    }
    if block_count == 0 {
        validator
            .errors
            .push("qwen block_count must be greater than zero".to_string());
    }
    let qk_head_dim = config.attention_key_head_dim().map(u64::from).unwrap_or(0);
    let v_head_dim = config
        .attention_value_head_dim()
        .map(u64::from)
        .unwrap_or(0);
    if qk_head_dim == 0 || v_head_dim == 0 {
        validator.errors.push(format!(
            "qwen attention key/value head dimensions could not be determined from embedding_length {embed}, attention.head_count {head_count}, attention.key_length {:?}, attention.value_length {:?}",
            config.attention_key_length, config.attention_value_length
        ));
    }
    let q_dim = qk_head_dim.saturating_mul(head_count);
    let k_dim = qk_head_dim.saturating_mul(kv_head_count);
    let v_dim = v_head_dim.saturating_mul(kv_head_count);
    let attention_output_dim = v_head_dim.saturating_mul(head_count);
    let ff = config.feed_forward_length.map(u64::from);
    let expert_count = config.expert_count.map(u64::from);
    let expert_ff = config.expert_feed_forward_length.map(u64::from);
    if config.expert_count.is_some() && config.expert_used_count.is_none() {
        validator
            .errors
            .push("qwen MoE metadata missing expert_used_count".to_string());
    }
    if let (Some(used), Some(total)) = (config.expert_used_count, config.expert_count)
        && (used == 0 || used > total)
    {
        validator.errors.push(format!(
            "qwen MoE expert_used_count {used} must be in 1..={total}"
        ));
    }
    // Group-limited routing (DeepSeek-V3 n_group > 1) is unimplemented in the
    // qwen MoE paths: running anyway would route experts silently wrong, so
    // reject at load. GLM-5.2-class checkpoints are n_group = 1 and pass.
    if let Some(groups) = config.expert_group_count
        && groups > 1
    {
        validator.errors.push(format!(
            "group-limited expert routing is unimplemented: {}.expert_group_count = {groups} (expert_group_used_count = {:?}); only expert_group_count <= 1 is supported",
            config.architecture, config.expert_group_used_count
        ));
    }
    if let (Some(used), Some(groups)) = (config.expert_group_used_count, config.expert_group_count)
        && groups > 0
        && used > groups
    {
        validator.errors.push(format!(
            "qwen MoE expert_group_used_count {used} must be <= expert_group_count {groups}"
        ));
    }

    validator.require_one_of(
        &qwen_dense_token_embd_weight_names(),
        embedding_matrix_rules(embed, vocab),
        DTypePolicy::Matrix,
    );
    validator.require_one_of(
        &qwen_dense_output_norm_weight_names(),
        vec![ShapeRule::exact([embed])],
        DTypePolicy::Any,
    );
    validator.optional_one_of(
        &qwen_dense_output_weight_names(),
        embedding_matrix_rules(embed, vocab),
        DTypePolicy::Matrix,
    );
    if let Some(vocab) = vocab {
        validator.optional_one_of(
            &qwen_dense_output_bias_names(),
            vec![ShapeRule::exact([vocab])],
            DTypePolicy::Any,
        );
    }

    for layer in 0..block_count {
        let prefix = format!("blk.{layer}");
        let uses_mla_attention = qwen_mla_attention_tensors_present_in(&validator.tensors, &prefix);
        let uses_recurrent_ssm = qwen_ssm_layer_tensors_present_in(&validator.tensors, &prefix);
        // Gemma-4 varies attention geometry per layer: sliding layers use the
        // `_swa` head dims with GQA, global layers the full dims with MQA and
        // no attn_v (V reuses K). Shadow the model-wide dims accordingly.
        let (qk_head_dim, _v_head_dim, q_dim, k_dim, v_dim, attention_output_dim, ff) =
            if config.is_gemma4() {
                let idx = layer as usize;
                let layer_qk = config.layer_key_head_dim(idx).map(u64::from).unwrap_or(0);
                let layer_v = config.layer_value_head_dim(idx).map(u64::from).unwrap_or(0);
                let layer_kv = u64::from(config.layer_head_count_kv(idx));
                (
                    layer_qk,
                    layer_v,
                    layer_qk.saturating_mul(head_count),
                    layer_qk.saturating_mul(layer_kv),
                    layer_v.saturating_mul(layer_kv),
                    layer_v.saturating_mul(head_count),
                    config.layer_feed_forward_length(idx).map(u64::from),
                )
            } else {
                (
                    qk_head_dim,
                    v_head_dim,
                    q_dim,
                    k_dim,
                    v_dim,
                    attention_output_dim,
                    ff,
                )
            };
        validator.require_one_of(
            &qwen_dense_attention_norm_weight_names(&prefix),
            vec![ShapeRule::exact([embed])],
            DTypePolicy::Any,
        );
        validator.require_one_of(
            &qwen_dense_ffn_norm_weight_names(&prefix),
            vec![ShapeRule::exact([embed])],
            DTypePolicy::Any,
        );
        if uses_recurrent_ssm {
            require_ssm_layer_tensors(&mut validator, config, &prefix, embed);
        } else if uses_mla_attention {
            require_mla_attention_tensors(&mut validator, config, &prefix, embed, head_count);
        } else if !qkv_split_tensors_present(&validator.tensors, &prefix)
            && let Some(packed_name) =
                dense_packed_qkv_name(&validator.tensors, &prefix, embed, q_dim, k_dim, v_dim)
        {
            validator.require(
                &packed_name,
                matrix_rules(embed, q_dim.saturating_add(k_dim).saturating_add(v_dim)),
                DTypePolicy::Matrix,
            );
        } else {
            let gated_q_name =
                dense_gated_attention_q_name(&validator.tensors, &prefix, embed, q_dim);
            if let Some(gated_q_name) = gated_q_name {
                validator.require(
                    &gated_q_name,
                    matrix_rules(embed, q_dim.saturating_mul(2)),
                    DTypePolicy::Matrix,
                );
            } else {
                validator.require_one_of(
                    &qwen_dense_attention_weight_names(&prefix, "q"),
                    matrix_rules(embed, q_dim),
                    DTypePolicy::Matrix,
                );
            }
            // Gemma-4: global layers ship no attn_v (V reuses K); the last
            // `shared_kv_layers` layers ship neither attn_k nor attn_v (they
            // reuse an earlier layer's KV).
            let gemma4_shared_kv = config.is_gemma4()
                && config.attention_shared_kv_layers.is_some_and(|shared| {
                    shared != 0 && layer >= block_count.saturating_sub(shared)
                });
            if gemma4_shared_kv {
                validator.optional_one_of(
                    &qwen_dense_attention_weight_names(&prefix, "k"),
                    matrix_rules(embed, k_dim),
                    DTypePolicy::Matrix,
                );
            } else {
                validator.require_one_of(
                    &qwen_dense_attention_weight_names(&prefix, "k"),
                    matrix_rules(embed, k_dim),
                    DTypePolicy::Matrix,
                );
            }
            if config.is_gemma4() {
                validator.optional_one_of(
                    &qwen_dense_attention_weight_names(&prefix, "v"),
                    matrix_rules(embed, v_dim),
                    DTypePolicy::Matrix,
                );
            } else {
                validator.require_one_of(
                    &qwen_dense_attention_weight_names(&prefix, "v"),
                    matrix_rules(embed, v_dim),
                    DTypePolicy::Matrix,
                );
            }
        }
        if !uses_recurrent_ssm {
            validator.require_one_of(
                &qwen_dense_attention_weight_names(&prefix, "output"),
                matrix_rules(attention_output_dim, embed),
                DTypePolicy::Matrix,
            );
        }
        if config.expert_count.is_some() && moe_router_tensor_present(&validator.tensors, &prefix) {
            let packed_expert_moe_complete =
                moe_packed_expert_tensors_complete(&validator.tensors, &prefix);
            let per_expert_packed_gate_up_moe_present =
                moe_any_per_expert_packed_gate_up_tensor_present(
                    &validator.tensors,
                    &prefix,
                    expert_count,
                );
            let per_expert_packed_gate_up_moe_complete = expert_count.is_some_and(|experts| {
                moe_per_expert_packed_gate_up_tensors_complete(&validator.tensors, &prefix, experts)
            });
            let use_per_expert_packed_gate_up_moe = !packed_expert_moe_complete
                && (per_expert_packed_gate_up_moe_present
                    || per_expert_packed_gate_up_moe_complete);
            let use_per_expert_moe = !packed_expert_moe_complete
                && !use_per_expert_packed_gate_up_moe
                && moe_any_per_expert_tensor_present(&validator.tensors, &prefix, expert_count);
            validator.require_one_of(
                &qwen_moe_router_weight_names(&prefix),
                expert_router_rules(embed, expert_count),
                DTypePolicy::Matrix,
            );
            if let Some(experts) = expert_count {
                validator.optional_one_of(
                    &qwen_moe_router_bias_names(&prefix),
                    vec![ShapeRule::exact([experts])],
                    DTypePolicy::Any,
                );
            }
            if use_per_expert_packed_gate_up_moe {
                require_per_expert_packed_gate_up_moe_tensors(
                    &mut validator,
                    &prefix,
                    embed,
                    expert_ff,
                    expert_count,
                );
            } else if use_per_expert_moe {
                require_per_expert_moe_tensors(
                    &mut validator,
                    &prefix,
                    embed,
                    expert_ff,
                    expert_count,
                );
            } else if let Some(packed_gate_up_name) = moe_packed_expert_gate_up_name(
                &validator.tensors,
                &prefix,
                embed,
                expert_ff,
                expert_count,
            ) {
                let Some(expert_ff) = expert_ff else {
                    validator.errors.push(
                        "model metadata missing expert_feed_forward_length for packed MoE gate/up layout"
                            .to_string(),
                    );
                    continue;
                };
                validator.require(
                    &packed_gate_up_name,
                    expert_matrix_rules(embed, Some(expert_ff.saturating_mul(2)), expert_count),
                    DTypePolicy::Matrix,
                );
                validator.require_one_of(
                    &qwen_moe_packed_expert_weight_names(&prefix, "down"),
                    expert_matrix_rules(embed, Some(expert_ff), expert_count),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &qwen_moe_packed_expert_gate_up_bias_names(&prefix)
                        .into_iter()
                        .chain(qwen_moe_packed_expert_up_gate_bias_names(&prefix))
                        .collect::<Vec<_>>(),
                    expert_bias_rules(expert_ff.saturating_mul(2), expert_count),
                    DTypePolicy::Any,
                );
                validator.optional_one_of(
                    &qwen_moe_packed_expert_bias_names(&prefix, "down"),
                    expert_bias_rules(embed, expert_count),
                    DTypePolicy::Any,
                );
            } else {
                validator.require_one_of(
                    &qwen_moe_packed_expert_weight_names(&prefix, "gate"),
                    expert_matrix_rules(embed, expert_ff, expert_count),
                    DTypePolicy::Matrix,
                );
                validator.require_one_of(
                    &qwen_moe_packed_expert_weight_names(&prefix, "up"),
                    expert_matrix_rules(embed, expert_ff, expert_count),
                    DTypePolicy::Matrix,
                );
                validator.require_one_of(
                    &qwen_moe_packed_expert_weight_names(&prefix, "down"),
                    expert_matrix_rules(embed, expert_ff, expert_count),
                    DTypePolicy::Matrix,
                );
                if let Some(expert_ff) = expert_ff {
                    validator.optional_one_of(
                        &qwen_moe_packed_expert_bias_names(&prefix, "gate"),
                        expert_bias_rules(expert_ff, expert_count),
                        DTypePolicy::Any,
                    );
                    validator.optional_one_of(
                        &qwen_moe_packed_expert_bias_names(&prefix, "up"),
                        expert_bias_rules(expert_ff, expert_count),
                        DTypePolicy::Any,
                    );
                    validator.optional_one_of(
                        &qwen_moe_packed_expert_bias_names(&prefix, "down"),
                        expert_bias_rules(embed, expert_count),
                        DTypePolicy::Any,
                    );
                }
            }
            let shared = [
                qwen_moe_shared_expert_weight_names(&prefix, "gate"),
                qwen_moe_shared_expert_weight_names(&prefix, "up"),
                qwen_moe_shared_expert_weight_names(&prefix, "down"),
            ];
            let shared_expert_gate = qwen_moe_shared_expert_gate_weight_names(&prefix);
            let shared_expert_gate_present = shared_expert_gate
                .iter()
                .any(|name| validator.tensors.contains_key(name.as_str()));
            let shared_split_present = shared
                .iter()
                .filter(|names| {
                    names
                        .iter()
                        .any(|name| validator.tensors.contains_key(name.as_str()))
                })
                .count();
            let shared_packed_gate_up = moe_shared_expert_packed_gate_up_weight_names(&prefix);
            let shared_packed_gate_up_present = shared_packed_gate_up
                .iter()
                .any(|name| validator.tensors.contains_key(name.as_str()));
            let shared_down_present = shared[2]
                .iter()
                .any(|name| validator.tensors.contains_key(name.as_str()));
            if shared_packed_gate_up_present {
                let Some(shared_ff) = ff.or(expert_ff) else {
                    validator.errors.push(
                        "model metadata missing feed_forward_length or expert_feed_forward_length for packed shared expert layout"
                            .to_string(),
                    );
                    continue;
                };
                if !shared_down_present {
                    let expected = [
                        shared_packed_gate_up.first().cloned().unwrap_or_default(),
                        shared[2].first().cloned().unwrap_or_default(),
                    ];
                    validator.errors.push(format!(
                        "layer {layer} has incomplete packed shared expert tensors; expected all of {}",
                        expected.join(", ")
                    ));
                }
                validator.optional_one_of(
                    &shared_packed_gate_up,
                    matrix_rules(embed, shared_ff.saturating_mul(2)),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &moe_shared_expert_packed_gate_up_bias_names(&prefix),
                    vec![ShapeRule::exact([shared_ff.saturating_mul(2)])],
                    DTypePolicy::Any,
                );
                validator.optional_one_of(
                    &shared[2],
                    feed_forward_matrix_rules(embed, Some(shared_ff)),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &qwen_moe_shared_expert_bias_names(&prefix, "down"),
                    vec![ShapeRule::exact([embed])],
                    DTypePolicy::Any,
                );
                validator.optional_one_of(
                    &shared_expert_gate,
                    matrix_rules(embed, 1),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &qwen_moe_shared_expert_gate_bias_names(&prefix),
                    vec![ShapeRule::exact([1])],
                    DTypePolicy::Any,
                );
            } else if shared_split_present != 0 && shared_split_present != shared.len() {
                let expected = shared
                    .iter()
                    .filter_map(|names| names.first())
                    .cloned()
                    .collect::<Vec<_>>();
                validator.errors.push(format!(
                    "layer {layer} has incomplete shared expert tensors; expected all of {}",
                    expected.join(", ")
                ));
            } else if shared_split_present == shared.len() {
                let shared_ff = ff.or(expert_ff);
                // GLM/DeepSeek shared experts are expert_ff wide, qwen2moe's
                // are n_ff wide — accept either.
                let mut shared_rules = feed_forward_matrix_rules(embed, shared_ff);
                if expert_ff.is_some() && expert_ff != shared_ff {
                    shared_rules.extend(feed_forward_matrix_rules(embed, expert_ff));
                }
                // DeepSeek-V2-Lite fuses `expert_shared_count` shared experts into
                // one MLP, widening its intermediate to expert_ff * shared_count.
                if let (Some(expert_ff), Some(shared_count)) =
                    (expert_ff, config.expert_shared_count)
                    && shared_count > 1
                {
                    shared_rules.extend(feed_forward_matrix_rules(
                        embed,
                        Some(expert_ff.saturating_mul(u64::from(shared_count))),
                    ));
                }
                validator.optional_one_of(&shared[0], shared_rules.clone(), DTypePolicy::Matrix);
                validator.optional_one_of(&shared[1], shared_rules.clone(), DTypePolicy::Matrix);
                validator.optional_one_of(&shared[2], shared_rules, DTypePolicy::Matrix);
                if let Some(shared_ff) = shared_ff {
                    validator.optional_one_of(
                        &qwen_moe_shared_expert_bias_names(&prefix, "gate"),
                        vec![ShapeRule::exact([shared_ff])],
                        DTypePolicy::Any,
                    );
                    validator.optional_one_of(
                        &qwen_moe_shared_expert_bias_names(&prefix, "up"),
                        vec![ShapeRule::exact([shared_ff])],
                        DTypePolicy::Any,
                    );
                    validator.optional_one_of(
                        &qwen_moe_shared_expert_bias_names(&prefix, "down"),
                        vec![ShapeRule::exact([embed])],
                        DTypePolicy::Any,
                    );
                }
                validator.optional_one_of(
                    &shared_expert_gate,
                    matrix_rules(embed, 1),
                    DTypePolicy::Matrix,
                );
                validator.optional_one_of(
                    &qwen_moe_shared_expert_gate_bias_names(&prefix),
                    vec![ShapeRule::exact([1])],
                    DTypePolicy::Any,
                );
            } else if shared_expert_gate_present {
                validator.errors.push(format!(
                    "layer {layer} has shared expert gate tensor without shared expert tensors; expected shared expert gate/up/down tensors or packed shared expert gate_up plus down"
                ));
            }
        } else if let Some(packed_name) =
            dense_packed_ffn_name(&validator.tensors, &prefix, embed, ff)
        {
            let Some(ff) = ff else {
                validator.errors.push(
                    "model metadata missing feed_forward_length for packed FFN layout".to_string(),
                );
                continue;
            };
            validator.require(
                &packed_name,
                matrix_rules(embed, ff.saturating_mul(2)),
                DTypePolicy::Matrix,
            );
            validator.optional_one_of(
                &qwen_dense_packed_ffn_gate_up_bias_names(&prefix)
                    .into_iter()
                    .chain(qwen_dense_packed_ffn_up_gate_bias_names(&prefix))
                    .collect::<Vec<_>>(),
                vec![ShapeRule::exact([ff.saturating_mul(2)])],
                DTypePolicy::Any,
            );
            validator.require_one_of(
                &qwen_dense_ffn_weight_names(&prefix, "down"),
                feed_forward_matrix_rules(embed, Some(ff)),
                DTypePolicy::Matrix,
            );
            validator.optional_one_of(
                &qwen_dense_ffn_bias_names(&prefix, "down"),
                vec![ShapeRule::exact([embed])],
                DTypePolicy::Any,
            );
        } else {
            validator.require_one_of(
                &qwen_dense_ffn_weight_names(&prefix, "gate"),
                feed_forward_matrix_rules(embed, ff),
                DTypePolicy::Matrix,
            );
            validator.require_one_of(
                &qwen_dense_ffn_weight_names(&prefix, "up"),
                feed_forward_matrix_rules(embed, ff),
                DTypePolicy::Matrix,
            );
            validator.require_one_of(
                &qwen_dense_ffn_weight_names(&prefix, "down"),
                feed_forward_matrix_rules(embed, ff),
                DTypePolicy::Matrix,
            );
            if let Some(ff) = ff {
                validator.optional_one_of(
                    &qwen_dense_ffn_bias_names(&prefix, "gate"),
                    vec![ShapeRule::exact([ff])],
                    DTypePolicy::Any,
                );
                validator.optional_one_of(
                    &qwen_dense_ffn_bias_names(&prefix, "up"),
                    vec![ShapeRule::exact([ff])],
                    DTypePolicy::Any,
                );
                validator.optional_one_of(
                    &qwen_dense_ffn_bias_names(&prefix, "down"),
                    vec![ShapeRule::exact([embed])],
                    DTypePolicy::Any,
                );
            }
        }

        if !uses_recurrent_ssm {
            validator.optional_one_of(
                &qwen_dense_attention_bias_names(&prefix, "output"),
                vec![ShapeRule::exact([embed])],
                DTypePolicy::Any,
            );
        }

        if !uses_mla_attention && !uses_recurrent_ssm {
            let q_bias_rules = if dense_gated_attention_q_name(
                &validator.tensors,
                &prefix,
                embed,
                q_dim,
            )
            .is_some()
            {
                vec![ShapeRule::exact([q_dim.saturating_mul(2)])]
            } else {
                vec![ShapeRule::exact([q_dim])]
            };
            validator.optional_one_of(
                &qwen_dense_attention_bias_names(&prefix, "q"),
                q_bias_rules,
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_dense_attention_bias_names(&prefix, "k"),
                vec![ShapeRule::exact([k_dim])],
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_dense_attention_bias_names(&prefix, "v"),
                vec![ShapeRule::exact([v_dim])],
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_dense_attention_head_norm_weight_names(&prefix, "q"),
                vec![ShapeRule::exact([qk_head_dim])],
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_dense_attention_head_norm_weight_names(&prefix, "k"),
                vec![ShapeRule::exact([qk_head_dim])],
                DTypePolicy::Any,
            );
            if qwen_dense_packed_qkv_bias_names(&prefix)
                .iter()
                .any(|name| validator.tensors.contains_key(name.as_str()))
            {
                validator.optional_one_of(
                    &qwen_dense_packed_qkv_bias_names(&prefix),
                    vec![ShapeRule::exact([q_dim
                        .saturating_add(k_dim)
                        .saturating_add(v_dim)])],
                    DTypePolicy::Any,
                );
            }
        }
    }

    let total_tensor_bytes = gguf
        .tensors
        .iter()
        .filter_map(|tensor| tensor.byte_len().ok())
        .sum();
    QwenTensorValidation {
        valid: validator.errors.is_empty(),
        required_tensors: validator.required_tensors,
        optional_tensors_present: validator.optional_tensors_present,
        tensor_count: gguf.tensors.len(),
        total_tensor_bytes,
        errors: validator.errors,
    }
}

struct QwenTensorValidator<'a> {
    tensors: BTreeMap<&'a str, &'a TensorInfo>,
    errors: Vec<String>,
    required_tensors: usize,
    optional_tensors_present: usize,
}

impl QwenTensorValidator<'_> {
    fn require(&mut self, name: &str, rules: Vec<ShapeRule>, dtype_policy: DTypePolicy) {
        self.required_tensors += 1;
        self.check(name, &rules, dtype_policy, true);
    }

    fn require_one_of(
        &mut self,
        names: &[String],
        rules: Vec<ShapeRule>,
        dtype_policy: DTypePolicy,
    ) {
        self.required_tensors += 1;
        self.check_one_of(names, &rules, dtype_policy, true);
    }

    fn optional_one_of(
        &mut self,
        names: &[String],
        rules: Vec<ShapeRule>,
        dtype_policy: DTypePolicy,
    ) {
        if names
            .iter()
            .any(|name| self.tensors.contains_key(name.as_str()))
        {
            self.optional_tensors_present += 1;
            self.check_one_of(names, &rules, dtype_policy, false);
        }
    }

    fn check_one_of(
        &mut self,
        names: &[String],
        rules: &[ShapeRule],
        dtype_policy: DTypePolicy,
        required: bool,
    ) {
        let Some(name) = names
            .iter()
            .find(|name| self.tensors.contains_key(name.as_str()))
        else {
            if required && let Some(primary) = names.first() {
                if names.len() > 1 {
                    self.errors.push(format!(
                        "missing required tensor {primary}; accepted aliases: {}",
                        names[1..].join(", ")
                    ));
                } else {
                    self.errors
                        .push(format!("missing required tensor {primary}"));
                }
            }
            return;
        };
        self.check(name, rules, dtype_policy, false);
    }

    fn check(
        &mut self,
        name: &str,
        rules: &[ShapeRule],
        dtype_policy: DTypePolicy,
        required: bool,
    ) {
        let Some(tensor) = self.tensors.get(name) else {
            if required {
                self.errors.push(format!("missing required tensor {name}"));
            }
            return;
        };

        if !rules.iter().any(|rule| rule.matches(&tensor.dimensions)) {
            self.errors.push(format!(
                "tensor {name} has shape {:?}; expected {}",
                tensor.dimensions,
                describe_shape_rules(rules)
            ));
        }

        if dtype_policy == DTypePolicy::Matrix
            && !tensor.dtype.is_quantized()
            && !matches!(
                tensor.dtype,
                GgufTensorType::F16 | GgufTensorType::BF16 | GgufTensorType::F32
            )
        {
            self.errors.push(format!(
                "matrix tensor {name} has dtype {}; CUDA accepts FP16/BF16/F32 or quantized matrix weights",
                tensor.dtype.label()
            ));
        }
    }
}

fn qkv_split_tensors_present(tensors: &BTreeMap<&str, &TensorInfo>, prefix: &str) -> bool {
    ["q", "k", "v"].iter().all(|suffix| {
        qwen_dense_attention_weight_names(prefix, suffix)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
    })
}

fn qwen_mla_attention_tensors_present_in(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
) -> bool {
    let kv_latent_present = qwen_mla_kv_a_weight_names(prefix)
        .iter()
        .any(|name| tensors.contains_key(name.as_str()))
        && qwen_mla_kv_a_norm_weight_names(prefix)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
        && (qwen_mla_kv_b_weight_names(prefix)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
            || (qwen_mla_k_b_weight_names(prefix)
                .iter()
                .any(|name| tensors.contains_key(name.as_str()))
                && qwen_mla_v_b_weight_names(prefix)
                    .iter()
                    .any(|name| tensors.contains_key(name.as_str()))));
    if !kv_latent_present {
        return false;
    }
    // Q-LoRA query decomposition (DeepSeek-V2/V3, GLM): q_a -> q_a_norm -> q_b.
    let q_lora_present = qwen_mla_q_a_weight_names(prefix)
        .iter()
        .any(|name| tensors.contains_key(name.as_str()))
        && qwen_mla_q_a_norm_weight_names(prefix)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
        && qwen_mla_q_b_weight_names(prefix)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()));
    // Full-Q MLA (DeepSeek-V2-Lite class): a single dense attn_q + KV latent.
    let full_q_present = qwen_dense_attention_weight_names(prefix, "q")
        .iter()
        .any(|name| tensors.contains_key(name.as_str()));
    q_lora_present || full_q_present
}

fn moe_router_tensor_present(tensors: &BTreeMap<&str, &TensorInfo>, prefix: &str) -> bool {
    qwen_moe_router_weight_names(prefix)
        .iter()
        .any(|name| tensors.contains_key(name.as_str()))
}

fn moe_packed_expert_tensors_complete(tensors: &BTreeMap<&str, &TensorInfo>, prefix: &str) -> bool {
    ["gate", "up", "down"].iter().all(|kind| {
        qwen_moe_packed_expert_weight_names(prefix, kind)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
    })
}

fn moe_any_per_expert_tensor_present(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    experts: Option<u64>,
) -> bool {
    let Some(experts) = experts else {
        return false;
    };
    (0..experts).any(|expert| {
        ["gate", "up", "down"].iter().any(|kind| {
            qwen_moe_per_expert_weight_names(prefix, kind, expert)
                .iter()
                .any(|name| tensors.contains_key(name.as_str()))
        })
    })
}

fn moe_per_expert_packed_gate_up_weight_names(prefix: &str, expert: u64) -> Vec<String> {
    let mut names = qwen_moe_per_expert_gate_up_weight_names(prefix, expert);
    for name in qwen_moe_per_expert_up_gate_weight_names(prefix, expert) {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn moe_per_expert_packed_gate_up_bias_names(prefix: &str, expert: u64) -> Vec<String> {
    let mut names = qwen_moe_per_expert_gate_up_bias_names(prefix, expert);
    for name in qwen_moe_per_expert_up_gate_bias_names(prefix, expert) {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn moe_shared_expert_packed_gate_up_weight_names(prefix: &str) -> Vec<String> {
    let mut names = qwen_moe_shared_expert_gate_up_weight_names(prefix);
    for name in qwen_moe_shared_expert_up_gate_weight_names(prefix) {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn moe_shared_expert_packed_gate_up_bias_names(prefix: &str) -> Vec<String> {
    let mut names = qwen_moe_shared_expert_gate_up_bias_names(prefix);
    for name in qwen_moe_shared_expert_up_gate_bias_names(prefix) {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

fn moe_any_per_expert_packed_gate_up_tensor_present(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    experts: Option<u64>,
) -> bool {
    let Some(experts) = experts else {
        return false;
    };
    (0..experts).any(|expert| {
        moe_per_expert_packed_gate_up_weight_names(prefix, expert)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
    })
}

fn moe_per_expert_packed_gate_up_tensors_complete(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    experts: u64,
) -> bool {
    (0..experts).all(|expert| {
        moe_per_expert_packed_gate_up_weight_names(prefix, expert)
            .iter()
            .any(|name| tensors.contains_key(name.as_str()))
            && qwen_moe_per_expert_weight_names(prefix, "down", expert)
                .iter()
                .any(|name| tensors.contains_key(name.as_str()))
    })
}

fn require_per_expert_packed_gate_up_moe_tensors(
    validator: &mut QwenTensorValidator<'_>,
    prefix: &str,
    embed: u64,
    expert_ff: Option<u64>,
    experts: Option<u64>,
) {
    let Some(experts) = experts else {
        validator
            .errors
            .push("qwen MoE metadata missing expert_count".to_string());
        return;
    };
    let Some(expert_ff) = expert_ff else {
        validator.errors.push(
            "model metadata missing expert_feed_forward_length for per-expert packed MoE gate/up layout"
                .to_string(),
        );
        return;
    };
    for expert in 0..experts {
        validator.require_one_of(
            &moe_per_expert_packed_gate_up_weight_names(prefix, expert),
            matrix_rules(embed, expert_ff.saturating_mul(2)),
            DTypePolicy::Matrix,
        );
        validator.optional_one_of(
            &moe_per_expert_packed_gate_up_bias_names(prefix, expert),
            vec![ShapeRule::exact([expert_ff.saturating_mul(2)])],
            DTypePolicy::Any,
        );
        validator.require_one_of(
            &qwen_moe_per_expert_weight_names(prefix, "down", expert),
            feed_forward_matrix_rules(embed, Some(expert_ff)),
            DTypePolicy::Matrix,
        );
        validator.optional_one_of(
            &qwen_moe_per_expert_bias_names(prefix, "down", expert),
            vec![ShapeRule::exact([embed])],
            DTypePolicy::Any,
        );
    }
}

fn require_per_expert_moe_tensors(
    validator: &mut QwenTensorValidator<'_>,
    prefix: &str,
    embed: u64,
    expert_ff: Option<u64>,
    experts: Option<u64>,
) {
    let Some(experts) = experts else {
        validator
            .errors
            .push("qwen MoE metadata missing expert_count".to_string());
        return;
    };
    for expert in 0..experts {
        validator.require_one_of(
            &qwen_moe_per_expert_weight_names(prefix, "gate", expert),
            feed_forward_matrix_rules(embed, expert_ff),
            DTypePolicy::Matrix,
        );
        validator.require_one_of(
            &qwen_moe_per_expert_weight_names(prefix, "up", expert),
            feed_forward_matrix_rules(embed, expert_ff),
            DTypePolicy::Matrix,
        );
        validator.require_one_of(
            &qwen_moe_per_expert_weight_names(prefix, "down", expert),
            feed_forward_matrix_rules(embed, expert_ff),
            DTypePolicy::Matrix,
        );
        if let Some(expert_ff) = expert_ff {
            validator.optional_one_of(
                &qwen_moe_per_expert_bias_names(prefix, "gate", expert),
                vec![ShapeRule::exact([expert_ff])],
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_moe_per_expert_bias_names(prefix, "up", expert),
                vec![ShapeRule::exact([expert_ff])],
                DTypePolicy::Any,
            );
            validator.optional_one_of(
                &qwen_moe_per_expert_bias_names(prefix, "down", expert),
                vec![ShapeRule::exact([embed])],
                DTypePolicy::Any,
            );
        }
    }
}

fn require_mla_attention_tensors(
    validator: &mut QwenTensorValidator<'_>,
    config: &QwenGgufConfig,
    prefix: &str,
    embed: u64,
    head_count: u64,
) {
    let metadata_prefix = &config.architecture;
    // Q-LoRA is optional: DeepSeek-V2-Lite-class MLA has no q_lora_rank and uses
    // a single full attn_q projection instead of the q_a -> q_a_norm -> q_b path.
    let q_lora = config
        .attention_q_lora_rank
        .map(u64::from)
        .filter(|value| *value != 0);
    let Some(kv_lora) = config
        .attention_kv_lora_rank
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        validator.errors.push(format!(
            "MLA tensor layout in {prefix} requires {metadata_prefix}.attention.kv_lora_rank"
        ));
        return;
    };
    let Some(qk_rope) = config
        .attention_qk_rope_head_dim
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        validator.errors.push(format!(
            "MLA tensor layout in {prefix} requires {metadata_prefix}.attention.qk_rope_head_dim"
        ));
        return;
    };
    let Some(qk_nope) = config.attention_qk_nope_head_dim.map(u64::from) else {
        validator.errors.push(format!(
            "MLA tensor layout in {prefix} requires {metadata_prefix}.attention.qk_nope_head_dim"
        ));
        return;
    };
    let Some(v_head_dim) = config
        .attention_v_head_dim
        .or(config.attention_value_length)
        .map(u64::from)
        .filter(|value| *value != 0)
    else {
        validator.errors.push(format!(
            "MLA tensor layout in {prefix} requires {metadata_prefix}.attention.v_head_dim or {metadata_prefix}.attention.value_length"
        ));
        return;
    };

    let qk_head_dim = qk_nope.saturating_add(qk_rope);
    if qk_head_dim == 0 || head_count == 0 {
        validator.errors.push(format!(
            "MLA tensor layout in {prefix} has invalid head dimensions qk_nope={qk_nope}, qk_rope={qk_rope}, heads={head_count}"
        ));
        return;
    }
    let q_dim = qk_head_dim.saturating_mul(head_count);
    let kv_a_dim = kv_lora.saturating_add(qk_rope);
    let kv_b_dim = head_count.saturating_mul(qk_nope.saturating_add(v_head_dim));

    match q_lora {
        Some(q_lora) => {
            validator.require_one_of(
                &qwen_mla_q_a_weight_names(prefix),
                matrix_rules(embed, q_lora),
                DTypePolicy::Matrix,
            );
            validator.require_one_of(
                &qwen_mla_q_a_norm_weight_names(prefix),
                vec![ShapeRule::exact([q_lora])],
                DTypePolicy::Any,
            );
            validator.require_one_of(
                &qwen_mla_q_b_weight_names(prefix),
                matrix_rules(q_lora, q_dim),
                DTypePolicy::Matrix,
            );
        }
        None => {
            // Full-Q MLA (DeepSeek-V2-Lite): one dense attn_q [embed -> q_dim].
            validator.require_one_of(
                &qwen_dense_attention_weight_names(prefix, "q"),
                matrix_rules(embed, q_dim),
                DTypePolicy::Matrix,
            );
        }
    }
    validator.require_one_of(
        &qwen_mla_kv_a_weight_names(prefix),
        matrix_rules(embed, kv_a_dim),
        DTypePolicy::Matrix,
    );
    validator.require_one_of(
        &qwen_mla_kv_a_norm_weight_names(prefix),
        vec![ShapeRule::exact([kv_lora])],
        DTypePolicy::Any,
    );
    let split_kv_b_present = qwen_mla_k_b_weight_names(prefix)
        .iter()
        .any(|name| validator.tensors.contains_key(name.as_str()))
        && qwen_mla_v_b_weight_names(prefix)
            .iter()
            .any(|name| validator.tensors.contains_key(name.as_str()));
    if split_kv_b_present {
        // glm-dsa/DeepSeek-V3.2 split: per-head rank-3 tensors, k_b stored
        // transposed (nope x kv_lora x heads) for weight absorption.
        validator.require_one_of(
            &qwen_mla_k_b_weight_names(prefix),
            vec![ShapeRule::exact([qk_nope, kv_lora, head_count])],
            DTypePolicy::Matrix,
        );
        validator.require_one_of(
            &qwen_mla_v_b_weight_names(prefix),
            vec![ShapeRule::exact([kv_lora, v_head_dim, head_count])],
            DTypePolicy::Matrix,
        );
    } else {
        validator.require_one_of(
            &qwen_mla_kv_b_weight_names(prefix),
            matrix_rules(kv_lora, kv_b_dim),
            DTypePolicy::Matrix,
        );
    }
}

fn require_ssm_layer_tensors(
    validator: &mut QwenTensorValidator<'_>,
    config: &QwenGgufConfig,
    prefix: &str,
    embed: u64,
) {
    let dims = match qwen_ssm_dims(config, prefix) {
        Ok(Some(dims)) => dims,
        Ok(None) => {
            validator.errors.push(format!(
                "SSM tensor layout in {prefix} requires recurrent SSM config metadata"
            ));
            return;
        }
        Err(err) => {
            validator.errors.push(err.to_string());
            return;
        }
    };
    let legacy_in_present = qwen_ssm_in_weight_names(prefix)
        .iter()
        .any(|name| validator.tensors.contains_key(name.as_str()));
    if legacy_in_present {
        validator.require_one_of(
            &qwen_ssm_in_weight_names(prefix),
            matrix_rules(embed, dims.qkvz_dim),
            DTypePolicy::Matrix,
        );
    } else {
        validator.require_one_of(
            &qwen_ssm_qkv_weight_names(prefix),
            matrix_rules(embed, dims.conv_dim),
            DTypePolicy::Matrix,
        );
        validator.require_one_of(
            &qwen_ssm_gate_weight_names(prefix),
            matrix_rules(embed, dims.value_dim),
            DTypePolicy::Matrix,
        );
    }
    validator.require_one_of(
        &qwen_ssm_conv1d_weight_names(prefix),
        matrix_rules(dims.conv_kernel, dims.conv_dim),
        DTypePolicy::Matrix,
    );
    validator.require_one_of(
        &qwen_ssm_dt_bias_names(prefix),
        vec![ShapeRule::exact([dims.time_step_rank])],
        DTypePolicy::Any,
    );
    validator.require_one_of(
        &qwen_ssm_a_names(prefix),
        vec![ShapeRule::exact([dims.time_step_rank])],
        DTypePolicy::Any,
    );
    let fused_ba_present = qwen_ssm_ba_weight_names(prefix)
        .iter()
        .any(|name| validator.tensors.contains_key(name.as_str()));
    if fused_ba_present {
        validator.require_one_of(
            &qwen_ssm_ba_weight_names(prefix),
            matrix_rules(embed, dims.ba_dim),
            DTypePolicy::Matrix,
        );
    } else {
        validator.require_one_of(
            &qwen_ssm_beta_weight_names(prefix),
            matrix_rules(embed, dims.time_step_rank),
            DTypePolicy::Matrix,
        );
        validator.require_one_of(
            &qwen_ssm_alpha_weight_names(prefix),
            matrix_rules(embed, dims.time_step_rank),
            DTypePolicy::Matrix,
        );
    }
    validator.require_one_of(
        &qwen_ssm_norm_weight_names(prefix),
        vec![ShapeRule::exact([dims.head_v_dim])],
        DTypePolicy::Any,
    );
    validator.require_one_of(
        &qwen_ssm_out_weight_names(prefix),
        matrix_rules(dims.value_dim, embed),
        DTypePolicy::Matrix,
    );
}

fn dense_packed_qkv_name(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    embed: u64,
    q_dim: u64,
    k_dim: u64,
    v_dim: u64,
) -> Option<String> {
    let qkv_dim = q_dim.checked_add(k_dim)?.checked_add(v_dim)?;
    qwen_dense_packed_qkv_weight_names(prefix)
        .into_iter()
        .find(|name| {
            tensor_matches_matrix_rules(tensors.get(name.as_str()).copied(), embed, qkv_dim)
        })
}

fn dense_gated_attention_q_name(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    embed: u64,
    q_dim: u64,
) -> Option<String> {
    let gated_q_dim = q_dim.checked_mul(2)?;
    qwen_dense_attention_weight_names(prefix, "q")
        .into_iter()
        .find(|name| {
            tensor_matches_matrix_rules(tensors.get(name.as_str()).copied(), embed, gated_q_dim)
        })
}

fn dense_packed_ffn_name(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    embed: u64,
    ff: Option<u64>,
) -> Option<String> {
    let ff = ff?;
    let packed_rows = ff.checked_mul(2)?;
    for name in qwen_dense_packed_ffn_gate_up_weight_names(prefix)
        .into_iter()
        .chain(qwen_dense_packed_ffn_up_gate_weight_names(prefix))
    {
        if tensor_matches_matrix_rules(tensors.get(name.as_str()).copied(), embed, packed_rows) {
            return Some(name);
        }
    }
    for prefix in layer_prefix_variants(prefix) {
        let gate_name = format!("{prefix}.ffn_gate.weight");
        let up_name = format!("{prefix}.ffn_up.weight");
        // Fused gate+up stored under `ffn_gate` (2x width) with no separate `ffn_up`.
        if !tensors.contains_key(up_name.as_str())
            && tensor_matches_matrix_rules(
                tensors.get(gate_name.as_str()).copied(),
                embed,
                packed_rows,
            )
        {
            return Some(gate_name);
        }
        // Fused gate+up stored under `ffn_up` (2x width) with no separate `ffn_gate` —
        // the llama.cpp layout for Phi-3 and similar SwiGLU models.
        if !tensors.contains_key(gate_name.as_str())
            && tensor_matches_matrix_rules(
                tensors.get(up_name.as_str()).copied(),
                embed,
                packed_rows,
            )
        {
            return Some(up_name);
        }
    }
    None
}

fn moe_packed_expert_gate_up_name(
    tensors: &BTreeMap<&str, &TensorInfo>,
    prefix: &str,
    embed: u64,
    expert_ff: Option<u64>,
    expert_count: Option<u64>,
) -> Option<String> {
    let expert_ff = expert_ff?;
    let packed_rows = expert_ff.checked_mul(2)?;
    qwen_moe_packed_expert_gate_up_weight_names(prefix)
        .into_iter()
        .chain(qwen_moe_packed_expert_up_gate_weight_names(prefix))
        .find(|name| {
            expert_matrix_rules(embed, Some(packed_rows), expert_count)
                .iter()
                .any(|rule| {
                    tensors
                        .get(name.as_str())
                        .is_some_and(|tensor| rule.matches(&tensor.dimensions))
                })
        })
}

fn tensor_matches_matrix_rules(tensor: Option<&TensorInfo>, left: u64, right: u64) -> bool {
    let Some(tensor) = tensor else {
        return false;
    };
    matrix_rules(left, right)
        .iter()
        .any(|rule| rule.matches(&tensor.dimensions))
}

fn tensor_dimensions_match_matrix(dims: &[u64], left: u64, right: u64) -> bool {
    matrix_rules(left, right)
        .iter()
        .any(|rule| rule.matches(dims))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DTypePolicy {
    Any,
    Matrix,
}

#[derive(Clone, Debug)]
enum ShapeRule {
    Exact(Vec<u64>),
    RankWithDim { rank: usize, dim: u64 },
    RankWithDims { rank: usize, dims: Vec<u64> },
}

impl ShapeRule {
    fn exact<const N: usize>(dims: [u64; N]) -> Self {
        Self::Exact(dims.to_vec())
    }

    fn matches(&self, dims: &[u64]) -> bool {
        match self {
            Self::Exact(expected) => dims == expected,
            Self::RankWithDim { rank, dim } => dims.len() == *rank && dims.contains(dim),
            Self::RankWithDims {
                rank,
                dims: required,
            } => dims.len() == *rank && required.iter().all(|dim| dims.contains(dim)),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Exact(dims) => format!("{dims:?}"),
            Self::RankWithDim { rank, dim } => {
                format!("rank {rank} with one dimension equal to {dim}")
            }
            Self::RankWithDims { rank, dims } => {
                format!("rank {rank} with dimensions containing {dims:?}")
            }
        }
    }
}

fn describe_shape_rules(rules: &[ShapeRule]) -> String {
    rules
        .iter()
        .map(ShapeRule::describe)
        .collect::<Vec<_>>()
        .join(" or ")
}

fn matrix_rules(left: u64, right: u64) -> Vec<ShapeRule> {
    if left == 0 || right == 0 {
        vec![ShapeRule::RankWithDim {
            rank: 2,
            dim: left.max(right),
        }]
    } else if left == right {
        vec![ShapeRule::exact([left, right])]
    } else {
        vec![
            ShapeRule::exact([left, right]),
            ShapeRule::exact([right, left]),
        ]
    }
}

fn embedding_matrix_rules(embed: u64, vocab: Option<u64>) -> Vec<ShapeRule> {
    match vocab {
        Some(vocab) => matrix_rules(embed, vocab),
        None => vec![ShapeRule::RankWithDim {
            rank: 2,
            dim: embed,
        }],
    }
}

fn feed_forward_matrix_rules(embed: u64, ff: Option<u64>) -> Vec<ShapeRule> {
    match ff {
        Some(ff) => matrix_rules(embed, ff),
        None => vec![ShapeRule::RankWithDim {
            rank: 2,
            dim: embed,
        }],
    }
}

fn expert_router_rules(embed: u64, experts: Option<u64>) -> Vec<ShapeRule> {
    match experts {
        Some(experts) => matrix_rules(embed, experts),
        None => vec![ShapeRule::RankWithDim {
            rank: 2,
            dim: embed,
        }],
    }
}

fn expert_matrix_rules(embed: u64, ff: Option<u64>, experts: Option<u64>) -> Vec<ShapeRule> {
    match (ff, experts) {
        (Some(ff), Some(experts)) => vec![
            ShapeRule::exact([embed, ff, experts]),
            ShapeRule::exact([ff, embed, experts]),
        ],
        (Some(ff), None) => vec![ShapeRule::RankWithDims {
            rank: 3,
            dims: vec![embed, ff],
        }],
        (None, Some(experts)) => vec![ShapeRule::RankWithDims {
            rank: 3,
            dims: vec![embed, experts],
        }],
        (None, None) => vec![ShapeRule::RankWithDim {
            rank: 3,
            dim: embed,
        }],
    }
}

fn expert_bias_rules(len: u64, experts: Option<u64>) -> Vec<ShapeRule> {
    match experts {
        Some(experts) if len != experts => vec![
            ShapeRule::exact([len, experts]),
            ShapeRule::exact([experts, len]),
        ],
        Some(experts) => vec![ShapeRule::exact([len, experts])],
        None => vec![ShapeRule::RankWithDim { rank: 2, dim: len }],
    }
}
