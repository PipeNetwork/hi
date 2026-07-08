use anyhow::Result;

use crate::backend::{GenerationEvent, GenerationOutput, GenerationRequest};
use crate::config::MlxModelConfig;
use crate::generate::TokenizerRuntime;
use crate::weights::WeightCatalog;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
pub struct NativeRuntime;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
impl NativeRuntime {
    pub fn load(
        _config: MlxModelConfig,
        _weights: WeightCatalog,
        _tokenizer: TokenizerRuntime,
    ) -> Result<Self> {
        anyhow::bail!("native MLX inference requires Apple Silicon macOS")
    }

    pub fn generate(&mut self, _request: GenerationRequest) -> Result<GenerationOutput> {
        anyhow::bail!("native MLX inference requires Apple Silicon macOS")
    }

    pub fn stream_generate<F>(
        &mut self,
        _request: GenerationRequest,
        _on_event: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(GenerationEvent) -> Result<()>,
    {
        anyhow::bail!("native MLX inference requires Apple Silicon macOS")
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub struct NativeRuntime {
    config: MlxModelConfig,
    tokenizer: TokenizerRuntime,
    model: Box<dyn CausalLm + Send>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
impl NativeRuntime {
    pub fn load(
        config: MlxModelConfig,
        weights: WeightCatalog,
        tokenizer: TokenizerRuntime,
    ) -> Result<Self> {
        let model = native::load_model(&config, &weights)?;
        Ok(Self {
            config,
            tokenizer,
            model,
        })
    }

    pub fn generate(&mut self, request: GenerationRequest) -> Result<GenerationOutput> {
        native::generate(&self.config, self.model.as_mut(), &self.tokenizer, request)
    }

    pub fn stream_generate<F>(
        &mut self,
        request: GenerationRequest,
        on_event: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(GenerationEvent) -> Result<()>,
    {
        native::stream_generate(
            &self.config,
            self.model.as_mut(),
            &self.tokenizer,
            request,
            on_event,
        )
    }

    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let config = crate::config::load_model_config(path)?;
        let weights = crate::weights::WeightCatalog::load(path)?;
        weights.validate_for_config(&config)?;
        let tokenizer = crate::generate::TokenizerRuntime::load(path)?;
        Self::load(config, weights, tokenizer)
    }

    /// Whether this model can be a speculative-decoding *target* (needs KV-cache rollback).
    pub fn supports_speculative(&self) -> bool {
        self.model.supports_rollback()
    }

    /// Whether this model has a built-in MTP head for self-speculative decoding (GLM-5.2).
    pub fn supports_mtp(&self) -> bool {
        self.model.supports_mtp()
    }

    /// Greedy self-speculative decoding via the model's own MTP head.
    pub fn mtp_generate<F>(&mut self, request: GenerationRequest, mut on_event: F) -> Result<GenerationOutput>
    where
        F: FnMut(GenerationEvent) -> Result<()>,
    {
        self.model
            .mtp_generate(&self.config, &self.tokenizer, &request, &mut on_event)
    }

    /// Greedy speculative decoding using `draft` as the proposal model. Output is identical to this
    /// (target) model's greedy decode.
    pub fn speculative_generate<F>(
        &mut self,
        draft: &mut NativeRuntime,
        request: GenerationRequest,
        k: usize,
        on_event: F,
    ) -> Result<(GenerationOutput, native::SpecStats)>
    where
        F: FnMut(GenerationEvent) -> Result<()>,
    {
        native::speculative_generate(
            &self.config,
            self.model.as_mut(),
            draft.model.as_mut(),
            &self.tokenizer,
            request,
            k,
            on_event,
        )
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub trait CausalLm {
    fn forward(&mut self, input_ids: &[u32]) -> Result<mlx_rs::Array>;
    fn reset_cache(&mut self);
    fn prepare_cache(&mut self, _capacity: i32) {}
    /// Roll the KV cache back to `to_offset` (drop everything after). Used by speculative decoding to
    /// discard rejected draft tokens. Default is a no-op; only models with a rollback-safe KV cache
    /// (not the SSM state models) override it, so `speculative_generate` checks `supports_rollback`.
    fn rollback_cache(&mut self, _to_offset: i32) {}
    fn supports_rollback(&self) -> bool {
        false
    }
    /// Whether this model has a multi-token-prediction head for self-speculative decoding.
    fn supports_mtp(&self) -> bool {
        false
    }
    /// Greedy self-speculative decoding via the model's own MTP head. Boxed callback keeps the trait
    /// object-safe. Only implemented by models with an MTP head (GLM-5.2); default errors.
    fn mtp_generate(
        &mut self,
        _config: &MlxModelConfig,
        _tokenizer: &TokenizerRuntime,
        _request: &GenerationRequest,
        _on_event: &mut dyn FnMut(GenerationEvent) -> Result<()>,
    ) -> Result<GenerationOutput> {
        Err(anyhow::anyhow!(
            "MTP self-speculation is not supported by this model"
        ))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod native {
    use std::collections::HashMap;
    use std::ffi::CString;

    use anyhow::{Result, anyhow, bail};
    use mlx_rs::fast::{
        ScaledDotProductAttentionMask, layer_norm, rms_norm, rope, scaled_dot_product_attention,
    };
    use mlx_rs::ops::indexing::{
        IndexOp, TryIndexMutOp, argmax_axis, put_along_axis, take_along_axis,
    };
    use mlx_rs::ops::{
        argpartition_axis, broadcast_to, concatenate_axis, conv1d, cos, dequantize, einsum, exp,
        identity, matmul, maximum, mean_axis, minimum, rsqrt, sigmoid, sin, softmax_axis,
        split_sections, stack_axis, sum_axis, tril, which, zeros_dtype,
    };
    use mlx_rs::nn::{silu, softplus};
    use mlx_rs::transforms::compile::{CallMut, Compile};
    use mlx_rs::{Array, Stream, transforms};

    use super::CausalLm;
    use crate::backend::{GenerationEvent, GenerationOutput, GenerationRequest};
    use crate::config::{MlxModelConfig, QuantizationSpec};
    use crate::generate::{LogitsProcessor, TokenizerRuntime, hit_stop};
    use crate::manifest::ModelFamily;
    use crate::weights::{WeightCatalog, mlx::load_arrays};

    pub fn load_model(
        config: &MlxModelConfig,
        weights: &WeightCatalog,
    ) -> Result<Box<dyn CausalLm + Send>> {
        let arrays = load_arrays(weights)?;
        if config.is_deepseek_v4() {
            return Ok(Box::new(DeepSeekV4Like::new(config.clone(), arrays)?));
        }
        match config.family {
            ModelFamily::Qwen2 | ModelFamily::Qwen3 | ModelFamily::Hy3 => {
                // Qwen3.5 gated-delta-net hybrid (linear-attn heads present) uses its own path.
                if config.linear_num_value_heads.is_some() {
                    Ok(Box::new(Qwen35Like::new(config.clone(), arrays)?))
                } else {
                    Ok(Box::new(QwenLike::new(config.clone(), arrays)?))
                }
            }
            ModelFamily::DeepSeek | ModelFamily::GlmFlash => {
                // Standard GQA GLM-4 uses q/k/v_proj (no MLA `kv_a_proj`); route it to Glm4Like.
                if config.family == ModelFamily::GlmFlash
                    && arrays.contains_key("model.layers.0.self_attn.q_proj.weight")
                    && !arrays.contains_key("model.layers.0.self_attn.kv_a_proj_with_mqa.weight")
                {
                    Ok(Box::new(Glm4Like::new(config.clone(), arrays)?))
                } else {
                    Ok(Box::new(MlaLike::new(config.clone(), arrays)?))
                }
            }
            ModelFamily::Llama
            | ModelFamily::Mistral
            | ModelFamily::Mixtral
            | ModelFamily::Gemma
            | ModelFamily::Phi => {
                bail!(
                    "{} MLX models are not supported by hi-mlx yet; use --backend cuda",
                    config.family.label()
                )
            }
        }
    }

    pub fn generate(
        config: &MlxModelConfig,
        model: &mut dyn CausalLm,
        tokenizer: &TokenizerRuntime,
        request: GenerationRequest,
    ) -> Result<GenerationOutput> {
        stream_generate(config, model, tokenizer, request, |_| Ok(()))
    }

    pub fn stream_generate<F>(
        config: &MlxModelConfig,
        model: &mut dyn CausalLm,
        tokenizer: &TokenizerRuntime,
        request: GenerationRequest,
        mut on_event: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(GenerationEvent) -> Result<()>,
    {
        let prompt_tokens = tokenizer.encode(&request.prompt)?;
        if prompt_tokens.is_empty() {
            bail!("prompt encoded to zero tokens");
        }
        model.reset_cache();
        let max_tokens = request.max_tokens.max(1);
        let cache_capacity = prompt_tokens
            .len()
            .saturating_add(max_tokens as usize)
            .min(i32::MAX as usize) as i32;
        model.prepare_cache(cache_capacity);

        let mut tokens = prompt_tokens.clone();
        let mut generated = Vec::new();
        let mut processor = LogitsProcessor::new(
            request.temperature,
            request.top_p,
            1.0,
            request.seed.unwrap_or(0x4849),
        );
        let mut decoded_text = String::new();
        let mut logits = prefill_logits(model, &prompt_tokens, prefill_chunk_size())?;
        for _ in 0..max_tokens {
            let next = if request.temperature <= f32::EPSILON {
                crate::generate::mlx::greedy_next_token(&logits)?
            } else {
                crate::generate::mlx::sample_next_token(&logits, &mut processor, &tokens)?
            };
            let Some(next) = next else {
                break;
            };
            tokens.push(next);
            generated.push(next);
            let current_text = tokenizer.decode(&generated)?;
            let delta = decoded_delta(&decoded_text, &current_text, tokenizer, next)?;
            decoded_text = current_text;
            on_event(GenerationEvent::TokenDelta {
                token_id: next,
                text: delta,
            })?;
            if hit_stop(&generated, &config.eos_token_ids) {
                break;
            }
            logits = model.forward(&[next])?;
        }
        let text = tokenizer.decode(&generated)?;
        let output = GenerationOutput {
            prompt_tokens: tokens.len().saturating_sub(generated.len()) as u64,
            completion_tokens: generated.len() as u64,
            text,
        };
        on_event(GenerationEvent::Finished {
            output: output.clone(),
        })?;
        Ok(output)
    }

    // Per-position greedy token (argmax over vocab) for a [1, seq, vocab] logits tensor. The argmax
    // runs on the GPU so only `seq` integers cross to the CPU, not the full seq×vocab logits.
    fn argmax_rows(logits: &Array) -> Result<Vec<u32>> {
        let shape = logits.shape();
        let seq = shape[shape.len() - 2];
        let vocab = shape[shape.len() - 1];
        let am = argmax_axis(&logits.reshape(&[seq, vocab])?, 1, false)?.as_type::<i32>()?;
        transforms::eval([&am])?;
        Ok(am.as_slice::<i32>().iter().map(|&x| x as u32).collect())
    }

    pub struct SpecStats {
        pub rounds: usize,
        pub proposed: usize,
        pub accepted: usize,
    }

    // Greedy speculative decoding: a small draft model proposes `k` tokens each round, the target
    // verifies them in a single forward (one weight read), accepts the longest matching prefix, and
    // appends the target's own correction/bonus token. Output is identical to the target's greedy
    // decode. Draft + target MUST share a tokenizer.
    pub fn speculative_generate<F>(
        config: &MlxModelConfig,
        target: &mut dyn CausalLm,
        draft: &mut dyn CausalLm,
        tokenizer: &TokenizerRuntime,
        request: GenerationRequest,
        k: usize,
        mut on_event: F,
    ) -> Result<(GenerationOutput, SpecStats)>
    where
        F: FnMut(GenerationEvent) -> Result<()>,
    {
        if !target.supports_rollback() {
            bail!(
                "speculative decoding needs a rollback-capable target (Qwen2/Qwen3 attention); \
                 this target model does not support KV-cache rollback"
            );
        }
        let k = k.max(1);
        let prompt_tokens = tokenizer.encode(&request.prompt)?;
        if prompt_tokens.is_empty() {
            bail!("prompt encoded to zero tokens");
        }
        let max_tokens = request.max_tokens.max(1) as usize;
        target.reset_cache();
        draft.reset_cache();
        let cap = (prompt_tokens.len() + max_tokens + k + 4).min(i32::MAX as usize) as i32;
        target.prepare_cache(cap);
        draft.prepare_cache(cap);

        // Prefill both models. Target uses the "anchor" trick: the last committed token is kept OUT
        // of the KV cache and prepended to each verify forward, so the correction token folds into the
        // next round's verify — one target weight-read per round instead of two.
        let logits_t = prefill_logits(target, &prompt_tokens, prefill_chunk_size())?;
        let logits_d = prefill_logits(draft, &prompt_tokens, prefill_chunk_size())?;
        let _ = &logits_t;
        let mut d_next = *argmax_rows(&logits_d)?.last().unwrap();
        let mut m = prompt_tokens.len() as i32; // committed length
        // Pull the last prompt token back out of the target cache to seed the anchor.
        target.rollback_cache(m - 1);
        let mut anchor = *prompt_tokens.last().unwrap();

        let mut generated: Vec<u32> = Vec::new();
        let mut decoded_text = String::new();
        let (mut rounds, mut proposed, mut accepted) = (0usize, 0usize, 0usize);
        let mut stop = false;

        while generated.len() < max_tokens && !stop {
            rounds += 1;
            // 1. Draft proposes k tokens greedily (draft cache: m -> m+k).
            let mut drafts: Vec<u32> = Vec::with_capacity(k);
            let mut d = d_next;
            for i in 0..k {
                drafts.push(d);
                let dl = draft.forward(&[d])?;
                if i + 1 < k {
                    d = *argmax_rows(&dl)?.last().unwrap();
                }
            }
            proposed += k;

            // 2. Target verifies [anchor, d_1..d_k] in ONE forward (cache: m-1 -> m+k).
            let mut vin = Vec::with_capacity(k + 1);
            vin.push(anchor);
            vin.extend_from_slice(&drafts);
            let tl = target.forward(&vin)?;
            let ta = argmax_rows(&tl)?; // ta[0]=target token at pos m, ta[j]=token at pos m+j

            // 3. Accept longest prefix: d_{i+1} accepted iff drafts[i] == ta[i].
            let mut n = 0usize;
            while n < k && drafts[n] == ta[n] {
                n += 1;
            }
            accepted += n;
            let correction = ta[n]; // target's token at the divergence (or the bonus if n==k)

            // 4. Commit accepted drafts + the correction/bonus token.
            let mut to_commit: Vec<u32> = drafts[..n].to_vec();
            to_commit.push(correction);
            for &tok in &to_commit {
                generated.push(tok);
                let current_text = tokenizer.decode(&generated)?;
                let delta = decoded_delta(&decoded_text, &current_text, tokenizer, tok)?;
                decoded_text = current_text;
                on_event(GenerationEvent::TokenDelta {
                    token_id: tok,
                    text: delta,
                })?;
                if generated.len() >= max_tokens || hit_stop(&generated, &config.eos_token_ids) {
                    stop = true;
                    break;
                }
            }
            if stop {
                break;
            }

            // 5. Target: keep [anchor, d_1..d_n] (cache -> m+n); the correction becomes the new anchor
            //    (processed for free in the next verify). Draft: keep d_1..d_n, then process correction.
            target.rollback_cache(m + n as i32);
            anchor = correction;
            draft.rollback_cache(m + n as i32);
            let nld = draft.forward(&[correction])?;
            d_next = *argmax_rows(&nld)?.last().unwrap();
            m += n as i32 + 1;
        }

        let text = tokenizer.decode(&generated)?;
        let output = GenerationOutput {
            prompt_tokens: prompt_tokens.len() as u64,
            completion_tokens: generated.len() as u64,
            text,
        };
        on_event(GenerationEvent::Finished {
            output: output.clone(),
        })?;
        Ok((output, SpecStats { rounds, proposed, accepted }))
    }

    fn prefill_logits(
        model: &mut dyn CausalLm,
        prompt_tokens: &[u32],
        chunk_size: usize,
    ) -> Result<Array> {
        let chunk_size = chunk_size.max(1);
        let mut logits = None;
        for chunk in prompt_tokens.chunks(chunk_size) {
            logits = Some(model.forward(chunk)?);
        }
        logits.ok_or_else(|| anyhow!("prompt encoded to zero tokens"))
    }

    fn prefill_chunk_size() -> usize {
        std::env::var("HI_MLX_PREFILL_CHUNK_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(2048)
    }

    fn decoded_delta(
        previous_text: &str,
        current_text: &str,
        tokenizer: &TokenizerRuntime,
        token: u32,
    ) -> Result<String> {
        if let Some(delta) = current_text.strip_prefix(previous_text) {
            Ok(delta.to_string())
        } else {
            tokenizer.decode(&[token])
        }
    }

    #[derive(Clone)]
    struct Cache {
        key: Option<Array>,
        value: Option<Array>,
        offset: i32,
        max_len: Option<i32>,
        capacity: Option<i32>,
        start: i32,
    }

    impl Cache {
        fn new() -> Self {
            Self::with_max_len(None)
        }

        fn with_max_len(max_len: Option<i32>) -> Self {
            Self {
                key: None,
                value: None,
                offset: 0,
                max_len,
                capacity: None,
                start: 0,
            }
        }

        fn reset(&mut self) {
            self.key = None;
            self.value = None;
            self.offset = 0;
            self.start = 0;
        }

        // Roll the write position back; the dense (fixed-capacity) buffer keeps its storage and the
        // stale positions past `to_offset` are overwritten by the next update.
        fn rollback(&mut self, to_offset: i32) {
            self.offset = to_offset.max(0);
        }

        fn prepare_capacity(&mut self, capacity: i32) {
            self.capacity = Some(capacity.max(1));
            self.reset();
        }

        fn update(&mut self, key: Array, value: Array) -> Result<(Array, Array)> {
            let (key, value, _) = self.update_with_start(key, value)?;
            Ok((key, value))
        }

        fn update_with_start(&mut self, key: Array, value: Array) -> Result<(Array, Array, i32)> {
            if self.max_len.is_some() {
                return self.update_ring(key, value);
            }
            if self.capacity.is_some() {
                return self.update_dense(key, value);
            }
            self.update_concat(key, value)
        }

        fn update_concat(&mut self, key: Array, value: Array) -> Result<(Array, Array, i32)> {
            let previous_offset = self.offset;
            let new_len = key.shape()[2];
            let out_key = match self.key.take() {
                Some(prev) => concatenate_axis(&[prev, key], 2)?,
                None => key,
            };
            let out_value = match self.value.take() {
                Some(prev) => concatenate_axis(&[prev, value], 2)?,
                None => value,
            };
            let total_len = previous_offset + new_len;
            let key_start = total_len - out_key.shape()[2];
            self.offset = total_len;
            self.start = key_start;

            let (stored_key, stored_value) = match self.max_len {
                Some(max_len) if out_key.shape()[2] > max_len => {
                    let trim_start = out_key.shape()[2] - max_len;
                    (
                        out_key.index((.., .., trim_start.., ..)),
                        out_value.index((.., .., trim_start.., ..)),
                    )
                }
                _ => (out_key.clone(), out_value.clone()),
            };
            self.key = Some(stored_key);
            self.value = Some(stored_value);
            Ok((out_key, out_value, key_start))
        }

        fn update_dense(&mut self, key: Array, value: Array) -> Result<(Array, Array, i32)> {
            let previous_offset = self.offset;
            let new_len = key.shape()[2];
            let total_len = previous_offset + new_len;
            let Some(capacity) = self.capacity else {
                return self.update_concat(key, value);
            };
            if total_len > capacity {
                self.capacity = None;
                let previous_key = self.materialized_key()?;
                let previous_value = self.materialized_value()?;
                self.key = previous_key;
                self.value = previous_value;
                return self.update_concat(key, value);
            }

            let mut key_buffer = self
                .key
                .take()
                .unwrap_or_else(|| dense_buffer_like(&key, capacity));
            let mut value_buffer = self
                .value
                .take()
                .unwrap_or_else(|| dense_buffer_like(&value, capacity));
            key_buffer.try_index_mut((.., .., previous_offset..total_len, ..), key)?;
            value_buffer.try_index_mut((.., .., previous_offset..total_len, ..), value)?;
            let out_key = key_buffer.index((.., .., ..total_len, ..));
            let out_value = value_buffer.index((.., .., ..total_len, ..));
            self.key = Some(key_buffer);
            self.value = Some(value_buffer);
            self.offset = total_len;
            self.start = 0;
            Ok((out_key, out_value, 0))
        }

        fn update_ring(&mut self, key: Array, value: Array) -> Result<(Array, Array, i32)> {
            let max_len = self.max_len.unwrap_or(1).max(1);
            let previous_offset = self.offset;
            let new_len = key.shape()[2];
            let total_len = previous_offset + new_len;

            let mut key_buffer = self
                .key
                .take()
                .unwrap_or_else(|| dense_buffer_like(&key, max_len));
            let mut value_buffer = self
                .value
                .take()
                .unwrap_or_else(|| dense_buffer_like(&value, max_len));

            if new_len >= max_len {
                let trim_start = new_len - max_len;
                key_buffer.try_index_mut(
                    (.., .., ..max_len, ..),
                    key.index((.., .., trim_start.., ..)),
                )?;
                value_buffer.try_index_mut(
                    (.., .., ..max_len, ..),
                    value.index((.., .., trim_start.., ..)),
                )?;
            } else {
                let write_start = previous_offset.rem_euclid(max_len);
                let first_len = (max_len - write_start).min(new_len);
                let first_end = write_start + first_len;
                key_buffer.try_index_mut(
                    (.., .., write_start..first_end, ..),
                    key.index((.., .., ..first_len, ..)),
                )?;
                value_buffer.try_index_mut(
                    (.., .., write_start..first_end, ..),
                    value.index((.., .., ..first_len, ..)),
                )?;
                let remaining = new_len - first_len;
                if remaining > 0 {
                    key_buffer.try_index_mut(
                        (.., .., ..remaining, ..),
                        key.index((.., .., first_len.., ..)),
                    )?;
                    value_buffer.try_index_mut(
                        (.., .., ..remaining, ..),
                        value.index((.., .., first_len.., ..)),
                    )?;
                }
            }

            self.key = Some(key_buffer);
            self.value = Some(value_buffer);
            self.offset = total_len;
            let stored_len = total_len.min(max_len);
            self.start = total_len - stored_len;
            let out_key = self
                .materialized_key()?
                .expect("ring cache key set after update");
            let out_value = self
                .materialized_value()?
                .expect("ring cache value set after update");
            Ok((out_key, out_value, self.start))
        }

        fn materialized_key(&self) -> Result<Option<Array>> {
            self.materialized(self.key.as_ref())
        }

        fn materialized_value(&self) -> Result<Option<Array>> {
            self.materialized(self.value.as_ref())
        }

        fn materialized(&self, buffer: Option<&Array>) -> Result<Option<Array>> {
            let Some(buffer) = buffer else {
                return Ok(None);
            };
            let Some(max_len) = self.max_len else {
                return Ok(Some(buffer.index((.., .., ..self.offset, ..))));
            };
            let stored_len = self.offset.min(max_len);
            if stored_len <= 0 {
                return Ok(None);
            }
            if stored_len < max_len {
                return Ok(Some(buffer.index((.., .., ..stored_len, ..))));
            }
            let start_pos = self.start.rem_euclid(max_len);
            if start_pos == 0 {
                Ok(Some(buffer.clone()))
            } else {
                Ok(Some(concatenate_axis(
                    &[
                        buffer.index((.., .., start_pos..max_len, ..)),
                        buffer.index((.., .., ..start_pos, ..)),
                    ],
                    2,
                )?))
            }
        }
    }

    fn dense_buffer_like(reference: &Array, capacity: i32) -> Array {
        let mut shape = reference.shape().to_vec();
        shape[2] = capacity;
        zeros_dtype(&shape, reference.dtype()).expect("valid dense KV cache shape")
    }

    #[derive(Clone)]
    struct KeyCache {
        key: Option<Array>,
        offset: i32,
        capacity: Option<i32>,
    }

    impl KeyCache {
        fn new() -> Self {
            Self {
                key: None,
                offset: 0,
                capacity: None,
            }
        }

        fn prepare_capacity(&mut self, capacity: i32) {
            self.capacity = Some(capacity.max(1));
            self.key = None;
            self.offset = 0;
        }

        fn rollback(&mut self, to_offset: i32) {
            self.offset = to_offset.max(0);
        }

        fn update(&mut self, key: Array) -> Result<Array> {
            if self.capacity.is_some() {
                return self.update_dense(key);
            }
            let out_key = match self.key.take() {
                Some(prev) => concatenate_axis(&[prev, key], 2)?,
                None => key,
            };
            self.offset = out_key.shape()[2];
            self.key = Some(out_key.clone());
            Ok(out_key)
        }

        fn update_dense(&mut self, key: Array) -> Result<Array> {
            let previous_offset = self.offset;
            let new_len = key.shape()[2];
            let total_len = previous_offset + new_len;
            let Some(capacity) = self.capacity else {
                return self.update(key);
            };
            if total_len > capacity {
                self.capacity = None;
                let previous = self
                    .key
                    .as_ref()
                    .map(|key| key.index((.., .., ..self.offset, ..)));
                self.key = previous;
                return self.update(key);
            }
            let mut buffer = self
                .key
                .take()
                .unwrap_or_else(|| dense_buffer_like(&key, capacity));
            buffer.try_index_mut((.., .., previous_offset..total_len, ..), key)?;
            let out_key = buffer.index((.., .., ..total_len, ..));
            self.offset = total_len;
            self.key = Some(buffer);
            Ok(out_key)
        }
    }

    #[derive(Clone)]
    enum Linear {
        Dense {
            weight: Array,
            bias: Option<Array>,
        },
        Quantized {
            weight: Array,
            scales: Array,
            biases: Option<Array>,
            bias: Option<Array>,
            group_size: i32,
            bits: i32,
            mode: String,
        },
    }

    impl Linear {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let weight = take(arrays, &format!("{prefix}.weight"))?;
            let bias = arrays.get(&format!("{prefix}.bias")).cloned();
            match arrays.get(&format!("{prefix}.scales")) {
                Some(scales) => {
                    let spec = quant_spec_for(config, prefix, &weight, Some(scales))?;
                    let biases = arrays.get(&format!("{prefix}.biases")).cloned();
                    require_biases_for_affine(prefix, &spec, biases.as_ref())?;
                    Ok(Self::Quantized {
                        weight,
                        scales: scales.clone(),
                        biases,
                        bias,
                        group_size: spec.group_size as i32,
                        bits: spec.bits as i32,
                        mode: spec.mode.as_str().to_string(),
                    })
                }
                _ => Ok(Self::Dense { weight, bias }),
            }
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let y = match self {
                Self::Dense { weight, bias } => {
                    let mut y = matmul(x, weight.t())?;
                    if let Some(bias) = bias {
                        y = y + bias;
                    }
                    y
                }
                Self::Quantized {
                    weight,
                    scales,
                    biases,
                    bias,
                    group_size,
                    bits,
                    mode,
                } => {
                    let mut y = quantized_matmul_mode(
                        x,
                        weight,
                        scales,
                        biases.as_ref(),
                        true,
                        *group_size,
                        *bits,
                        mode,
                    )?;
                    if let Some(bias) = bias {
                        y = y + bias;
                    }
                    y
                }
            };
            Ok(y)
        }
    }

    #[derive(Clone)]
    enum Embedding {
        Dense(Array),
        Quantized {
            weight: Array,
            scales: Array,
            biases: Option<Array>,
            group_size: i32,
            bits: i32,
            mode: String,
        },
    }

    impl Embedding {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let weight = take(arrays, &format!("{prefix}.weight"))?;
            match arrays.get(&format!("{prefix}.scales")) {
                Some(scales) => {
                    let spec = quant_spec_for(config, prefix, &weight, Some(scales))?;
                    let biases = arrays.get(&format!("{prefix}.biases")).cloned();
                    require_biases_for_affine(prefix, &spec, biases.as_ref())?;
                    Ok(Self::Quantized {
                        weight,
                        scales: scales.clone(),
                        biases,
                        group_size: spec.group_size as i32,
                        bits: spec.bits as i32,
                        mode: spec.mode.as_str().to_string(),
                    })
                }
                _ => Ok(Self::Dense(weight)),
            }
        }

        fn forward(&self, ids: &Array) -> Result<Array> {
            match self {
                Self::Dense(weight) => Ok(weight.index(ids)),
                Self::Quantized {
                    weight,
                    scales,
                    biases,
                    group_size,
                    bits,
                    mode,
                } => {
                    let shape = ids.shape().to_vec();
                    let flat = ids.flatten(None, None)?;
                    let w = weight.index(&flat);
                    let s = scales.index(&flat);
                    let b = biases.as_ref().map(|biases| biases.index(&flat));
                    let out = dequantize_mode(&w, &s, b.as_ref(), *group_size, *bits, mode)?;
                    let mut ret = shape;
                    ret.push(-1);
                    Ok(out.reshape(&ret)?)
                }
            }
        }

        fn as_linear(&self, x: &Array) -> Result<Array> {
            match self {
                Self::Dense(weight) => matmul(x, weight.t()).map_err(Into::into),
                Self::Quantized {
                    weight,
                    scales,
                    biases,
                    group_size,
                    bits,
                    mode,
                } => quantized_matmul_mode(
                    x,
                    weight,
                    scales,
                    biases.as_ref(),
                    true,
                    *group_size,
                    *bits,
                    mode,
                ),
            }
        }
    }

    #[derive(Clone)]
    struct RmsNorm {
        weight: Array,
        eps: f32,
    }

    impl RmsNorm {
        fn load(key: &str, arrays: &HashMap<String, Array>, eps: f32) -> Result<Self> {
            Ok(Self {
                weight: take(arrays, key)?,
                eps,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            Ok(rms_norm(x, &self.weight, self.eps)?)
        }
    }

    #[derive(Clone)]
    struct LayerNorm {
        weight: Array,
        bias: Option<Array>,
        eps: f32,
    }

    impl LayerNorm {
        fn load(prefix: &str, arrays: &HashMap<String, Array>, eps: f32) -> Result<Self> {
            Ok(Self {
                weight: take(arrays, &format!("{prefix}.weight"))?,
                bias: arrays.get(&format!("{prefix}.bias")).cloned(),
                eps,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            Ok(layer_norm(x, &self.weight, self.bias.as_ref(), self.eps)?)
        }
    }

    struct QwenAttention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        q_norm: Option<RmsNorm>,
        k_norm: Option<RmsNorm>,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rope_theta: f32,
        traditional_rope: bool,
        cache: Cache,
    }

    impl QwenAttention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let q_norm = RmsNorm::load(
                &format!("{prefix}.q_norm.weight"),
                arrays,
                config.rms_norm_eps,
            )
            .ok();
            let k_norm = RmsNorm::load(
                &format!("{prefix}.k_norm.weight"),
                arrays,
                config.rms_norm_eps,
            )
            .ok();
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: Linear::load(&format!("{prefix}.v_proj"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                q_norm,
                k_norm,
                n_heads: config.num_attention_heads as i32,
                n_kv_heads: config.num_key_value_heads as i32,
                head_dim: config.attention_head_dim() as i32,
                rope_theta: config.rope_theta,
                traditional_rope: config.family == ModelFamily::Qwen2,
                cache: Cache::new(),
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l) = (shape[0], shape[1]);
            let mut q = self
                .q_proj
                .forward(x)?
                .reshape(&[b, l, self.n_heads, self.head_dim])?;
            let mut k = self
                .k_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?;
            if let Some(norm) = &self.q_norm {
                q = norm.forward(&q)?;
            }
            if let Some(norm) = &self.k_norm {
                k = norm.forward(&k)?;
            }
            q = q.transpose_axes(&[0, 2, 1, 3])?;
            k = k.transpose_axes(&[0, 2, 1, 3])?;
            let v = self
                .v_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let offset = self.cache.offset;
            q = rope(
                q,
                self.head_dim,
                self.traditional_rope,
                self.rope_theta,
                1.0,
                offset,
                None,
            )?;
            k = rope(
                k,
                self.head_dim,
                self.traditional_rope,
                self.rope_theta,
                1.0,
                offset,
                None,
            )?;
            let (k, v) = self.cache.update(k, v)?;
            let scale = (self.head_dim as f32).powf(-0.5);
            let output = if l > 1 && offset == 0 {
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    ScaledDotProductAttentionMask::Causal,
                    None::<&Array>,
                )?
            } else if l > 1 {
                let mask = causal_attention_mask(l, k.shape()[2], offset);
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    ScaledDotProductAttentionMask::Array(&mask),
                    None::<&Array>,
                )?
            } else {
                scaled_dot_product_attention(&q, &k, &v, scale, None, None::<&Array>)?
            };
            let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                l,
                self.n_heads * self.head_dim,
            ])?;
            self.o_proj.forward(&output)
        }
    }

    struct Mlp {
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    }

    impl Mlp {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                gate_proj: Linear::load(&format!("{prefix}.gate_proj"), arrays, config)?,
                up_proj: Linear::load(&format!("{prefix}.up_proj"), arrays, config)?,
                down_proj: Linear::load(&format!("{prefix}.down_proj"), arrays, config)?,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let gate_pre = self.gate_proj.forward(x)?;
            let gate = sigmoid(&gate_pre)? * gate_pre;
            let up = self.up_proj.forward(x)?;
            self.down_proj.forward(&(gate * up))
        }
    }

    #[derive(Clone)]
    enum MultiLinear {
        Dense {
            weight: Array,
        },
        Quantized {
            weight: Array,
            scales: Array,
            biases: Option<Array>,
            group_size: i32,
            bits: i32,
            mode: String,
        },
    }

    impl MultiLinear {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let weight = take(arrays, &format!("{prefix}.weight"))?;
            match arrays.get(&format!("{prefix}.scales")) {
                Some(scales) => {
                    let spec = quant_spec_for(config, prefix, &weight, Some(scales))?;
                    let biases = arrays.get(&format!("{prefix}.biases")).cloned();
                    require_biases_for_affine(prefix, &spec, biases.as_ref())?;
                    Ok(Self::Quantized {
                        weight,
                        scales: scales.clone(),
                        biases,
                        group_size: spec.group_size as i32,
                        bits: spec.bits as i32,
                        mode: spec.mode.as_str().to_string(),
                    })
                }
                _ => Ok(Self::Dense { weight }),
            }
        }

        fn forward(&self, x: &Array, transpose: bool) -> Result<Array> {
            match self {
                Self::Dense { weight } => {
                    let rhs = if transpose {
                        weight.swap_axes(-1, -2)?
                    } else {
                        weight.clone()
                    };
                    matmul(x, &rhs).map_err(Into::into)
                }
                Self::Quantized {
                    weight,
                    scales,
                    biases,
                    group_size,
                    bits,
                    mode,
                } => quantized_matmul_mode(
                    x,
                    weight,
                    scales,
                    biases.as_ref(),
                    transpose,
                    *group_size,
                    *bits,
                    mode,
                ),
            }
        }
    }

    struct MlaIndexer {
        wq_b: Linear,
        wk: Linear,
        k_norm: LayerNorm,
        weights_proj: Linear,
        n_heads: i32,
        head_dim: i32,
        rope_head_dim: i32,
        index_topk: i32,
        rope_theta: f32,
        traditional_rope: bool,
        softmax_scale: f32,
        cache: KeyCache,
    }

    impl MlaIndexer {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Option<Self>> {
            if !arrays.contains_key(&format!("{prefix}.wq_b.weight")) {
                return Ok(None);
            }
            let head_dim = config
                .index_head_dim
                .ok_or_else(|| anyhow!("config.json missing index_head_dim for MLA indexer"))?
                as i32;
            let n_heads = config
                .index_n_heads
                .ok_or_else(|| anyhow!("config.json missing index_n_heads for MLA indexer"))?
                as i32;
            let rope_head_dim = config
                .qk_rope_head_dim
                .ok_or_else(|| anyhow!("config.json missing qk_rope_head_dim for MLA indexer"))?
                as i32;
            let index_topk = config
                .index_topk
                .ok_or_else(|| anyhow!("config.json missing index_topk for MLA indexer"))?
                as i32;
            Ok(Some(Self {
                wq_b: Linear::load(&format!("{prefix}.wq_b"), arrays, config)?,
                wk: Linear::load(&format!("{prefix}.wk"), arrays, config)?,
                k_norm: LayerNorm::load(&format!("{prefix}.k_norm"), arrays, 1e-5)?,
                weights_proj: Linear::load(&format!("{prefix}.weights_proj"), arrays, config)?,
                n_heads,
                head_dim,
                rope_head_dim,
                index_topk,
                rope_theta: config.rope_theta,
                traditional_rope: config.indexer_rope_interleave,
                softmax_scale: (head_dim as f32).powf(-0.5),
                cache: KeyCache::new(),
            }))
        }

        fn forward(
            &mut self,
            x: &Array,
            query_latent: &Array,
            mask: Option<&Array>,
        ) -> Result<Option<Array>> {
            let shape = x.shape();
            let (b, s) = (shape[0], shape[1]);
            let mut q = self
                .wq_b
                .forward(query_latent)?
                .reshape(&[b, s, self.n_heads, self.head_dim])?
                .swap_axes(1, 2)?;
            let mut k =
                self.k_norm
                    .forward(&self.wk.forward(x)?)?
                    .reshape(&[b, 1, s, self.head_dim])?;

            let offset = self.cache.offset;
            q = rope(
                q,
                self.rope_head_dim,
                self.traditional_rope,
                self.rope_theta,
                1.0,
                offset,
                None,
            )?;
            k = rope(
                k,
                self.rope_head_dim,
                self.traditional_rope,
                self.rope_theta,
                1.0,
                offset,
                None,
            )?;
            k = self.cache.update(k)?;
            if k.shape()[2] <= self.index_topk {
                return Ok(None);
            }

            let mut scores = matmul(&q, &k.swap_axes(-1, -2)?)?;
            scores = maximum(&scores, &Array::from_f32(0.0))?;
            let weights = self.weights_proj.forward(x)?
                * ((self.n_heads as f32).powf(-0.5) * self.softmax_scale);
            let weights = weights.swap_axes(-1, -2)?.expand_dims(-1)?;
            scores = scores * weights;
            scores = sum_axis(&scores, 1, Some(true))?;
            if let Some(mask) = mask {
                scores = apply_attention_mask(&scores, mask)?;
            }
            let partitioned = argpartition_axis(&scores, -self.index_topk, -1)?;
            Ok(Some(partitioned.index((.., .., .., (-self.index_topk)..))))
        }
    }

    struct MlaAttention {
        q_a_proj: Option<Linear>,
        q_a_layernorm: Option<RmsNorm>,
        q_b_proj: Option<Linear>,
        q_proj: Option<Linear>,
        kv_a_proj_with_mqa: Linear,
        kv_a_layernorm: RmsNorm,
        embed_q: MultiLinear,
        unembed_out: MultiLinear,
        o_proj: Linear,
        indexer: Option<MlaIndexer>,
        num_heads: i32,
        qk_nope_head_dim: i32,
        qk_rope_head_dim: i32,
        v_head_dim: i32,
        kv_lora_rank: i32,
        q_head_dim: i32,
        scale: f32,
        rope_theta: f32,
        cache: Cache,
    }

    impl MlaAttention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let q_lora_rank = config.q_lora_rank;
            let q_a_proj = q_lora_rank
                .map(|_| Linear::load(&format!("{prefix}.q_a_proj"), arrays, config))
                .transpose()?;
            let q_a_layernorm = q_lora_rank
                .map(|_| {
                    RmsNorm::load(
                        &format!("{prefix}.q_a_layernorm.weight"),
                        arrays,
                        config.rms_norm_eps,
                    )
                })
                .transpose()?;
            let q_b_proj = q_lora_rank
                .map(|_| Linear::load(&format!("{prefix}.q_b_proj"), arrays, config))
                .transpose()?;
            let q_proj = if q_lora_rank.is_none() {
                Some(Linear::load(&format!("{prefix}.q_proj"), arrays, config)?)
            } else {
                None
            };
            let qk_nope_head_dim = config
                .qk_nope_head_dim
                .ok_or_else(|| anyhow!("config.json missing qk_nope_head_dim for MLA model"))?
                as i32;
            let qk_rope_head_dim = config
                .qk_rope_head_dim
                .ok_or_else(|| anyhow!("config.json missing qk_rope_head_dim for MLA model"))?
                as i32;
            let v_head_dim = config.v_head_dim.unwrap_or(qk_nope_head_dim as u32) as i32;
            let kv_lora_rank = config
                .kv_lora_rank
                .ok_or_else(|| anyhow!("config.json missing kv_lora_rank for MLA model"))?
                as i32;
            let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
            Ok(Self {
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
                q_proj,
                kv_a_proj_with_mqa: Linear::load(
                    &format!("{prefix}.kv_a_proj_with_mqa"),
                    arrays,
                    config,
                )?,
                kv_a_layernorm: RmsNorm::load(
                    &format!("{prefix}.kv_a_layernorm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                embed_q: MultiLinear::load(&format!("{prefix}.embed_q"), arrays, config)?,
                unembed_out: MultiLinear::load(&format!("{prefix}.unembed_out"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                indexer: MlaIndexer::load(&format!("{prefix}.indexer"), arrays, config)?,
                num_heads: config.num_attention_heads as i32,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                kv_lora_rank,
                q_head_dim,
                scale: (q_head_dim as f32).powf(-0.5),
                rope_theta: config.rope_theta,
                cache: Cache::new(),
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l) = (shape[0], shape[1]);
            let (q, query_latent) = match (
                &self.q_proj,
                &self.q_a_proj,
                &self.q_a_layernorm,
                &self.q_b_proj,
            ) {
                (Some(q_proj), _, _, _) => (q_proj.forward(x)?, None),
                (None, Some(q_a), Some(q_norm), Some(q_b)) => {
                    let query_latent = q_norm.forward(&q_a.forward(x)?)?;
                    (q_b.forward(&query_latent)?, Some(query_latent))
                }
                _ => bail!("invalid MLA query projection state"),
            };
            let q = q
                .reshape(&[b, l, self.num_heads, self.q_head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let mut q_parts = split_sections(&q, &[self.qk_nope_head_dim], -1)?;
            let mut q_nope = q_parts.remove(0);
            let mut q_pe = q_parts.remove(0);

            let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
            let mut kv_parts = split_sections(&compressed_kv, &[self.kv_lora_rank], -1)?;
            let compressed_kv = kv_parts.remove(0);
            let mut k_pe = kv_parts
                .remove(0)
                .reshape(&[b, l, 1, self.qk_rope_head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let mut kv_latent = self
                .kv_a_layernorm
                .forward(&compressed_kv)?
                .expand_dims(1)?;

            let offset = self.cache.offset;
            q_pe = rope(
                q_pe,
                self.qk_rope_head_dim,
                true,
                self.rope_theta,
                1.0,
                offset,
                None,
            )?;
            k_pe = rope(
                k_pe,
                self.qk_rope_head_dim,
                true,
                self.rope_theta,
                1.0,
                offset,
                None,
            )?;
            let (cached_latent, cached_k_pe) = self.cache.update(kv_latent, k_pe)?;
            kv_latent = cached_latent;
            k_pe = cached_k_pe;

            let mut mask = if l > 1 {
                Some(causal_attention_mask(l, k_pe.shape()[2], offset))
            } else {
                None
            };
            if let (Some(indexer), Some(query_latent)) =
                (self.indexer.as_mut(), query_latent.as_ref())
            {
                if let Some(topk_indices) = indexer.forward(x, query_latent, mask.as_ref())? {
                    if l == 1 {
                        let idx = topk_indices.index((.., .., 0, ..)).expand_dims(-1)?;
                        let idx_latent =
                            broadcast_to(&idx, &[b, 1, idx.shape()[2], kv_latent.shape()[3]])?;
                        let idx_pe = broadcast_to(&idx, &[b, 1, idx.shape()[2], k_pe.shape()[3]])?;
                        kv_latent = take_along_axis(&kv_latent, &idx_latent, Some(2))?;
                        k_pe = take_along_axis(&k_pe, &idx_pe, Some(2))?;
                    } else {
                        let sparse_shape = [b, 1, l, kv_latent.shape()[2]];
                        let sparse = Array::zeros::<bool>(&sparse_shape)?;
                        let mut sparse =
                            put_along_axis(&sparse, &topk_indices, &Array::from_bool(true), -1)?;
                        if let Some(causal) = &mask {
                            sparse = sparse.logical_and(causal)?;
                        }
                        mask = Some(sparse);
                    }
                }
            }

            let mut pe_scores =
                matmul(&(q_pe * self.scale), &k_pe.swap_axes(-1, -2)?)?.as_type::<f32>()?;
            if let Some(mask) = &mask {
                pe_scores = apply_attention_mask(&pe_scores, mask)?;
            }
            let (k, v) = if l == 1 {
                q_nope = self.embed_q.forward(&q_nope, true)?;
                (kv_latent.clone(), kv_latent)
            } else {
                (
                    self.embed_q.forward(&kv_latent, false)?,
                    self.unembed_out.forward(&kv_latent, true)?,
                )
            };
            let q_nope = q_nope.as_type::<f32>()?;
            let k = k.as_type::<f32>()?;
            let v = v.as_type::<f32>()?;
            let mut output = scaled_dot_product_attention(
                &q_nope,
                &k,
                &v,
                self.scale,
                ScaledDotProductAttentionMask::Array(&pe_scores),
                None::<&Array>,
            )?;
            if l == 1 {
                output = self.unembed_out.forward(&output, true)?;
            }
            let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                l,
                self.num_heads * self.v_head_dim,
            ])?;
            self.o_proj.forward(&output)
        }

        fn reset_cache(&mut self) {
            self.cache.reset();
            if let Some(indexer) = &mut self.indexer {
                indexer.cache = KeyCache::new();
            }
        }
    }

    struct SwitchLinear {
        weight: Array,
        scales: Option<Array>,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: String,
    }

    impl SwitchLinear {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let weight = take(arrays, &format!("{prefix}.weight"))?;
            let scales = arrays.get(&format!("{prefix}.scales")).cloned();
            let spec = quant_spec_for(config, prefix, &weight, scales.as_ref())?;
            let biases = arrays.get(&format!("{prefix}.biases")).cloned();
            if scales.is_some() {
                require_biases_for_affine(prefix, &spec, biases.as_ref())?;
            }
            Ok(Self {
                weight,
                scales,
                biases,
                group_size: spec.group_size as i32,
                bits: spec.bits as i32,
                mode: spec.mode.as_str().to_string(),
            })
        }

        fn forward_expert(&self, x: &Array, expert: i32) -> Result<Array> {
            let weight = self.weight.index(expert);
            match &self.scales {
                Some(scales) => {
                    let expert_biases = self.biases.as_ref().map(|biases| biases.index(expert));
                    quantized_matmul_mode(
                        x,
                        &weight,
                        &scales.index(expert),
                        expert_biases.as_ref(),
                        true,
                        self.group_size,
                        self.bits,
                        &self.mode,
                    )
                }
                _ => matmul(x, &weight.t()).map_err(Into::into),
            }
        }

        /// Batched forward over all routed experts at once. `rhs_indices` selects the expert
        /// weight for each output position (see `gather_qmm_mode`).
        fn gather(&self, x: &Array, rhs_indices: &Array) -> Result<Array> {
            match &self.scales {
                Some(scales) => gather_qmm_mode(
                    x,
                    &self.weight,
                    scales,
                    self.biases.as_ref(),
                    rhs_indices,
                    true,
                    self.group_size,
                    self.bits,
                    &self.mode,
                ),
                None => bail!("hi-mlx batched MoE requires quantized expert weights"),
            }
        }
    }

    struct SwitchMlp {
        gate_proj: SwitchLinear,
        up_proj: SwitchLinear,
        down_proj: SwitchLinear,
    }

    impl SwitchMlp {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                gate_proj: SwitchLinear::load(&format!("{prefix}.gate_proj"), arrays, config)?,
                up_proj: SwitchLinear::load(&format!("{prefix}.up_proj"), arrays, config)?,
                down_proj: SwitchLinear::load(&format!("{prefix}.down_proj"), arrays, config)?,
            })
        }

        fn forward_expert(&self, x: &Array, expert: i32) -> Result<Array> {
            let gate_pre = self.gate_proj.forward_expert(x, expert)?;
            let gate = sigmoid(&gate_pre)? * gate_pre;
            let up = self.up_proj.forward_expert(x, expert)?;
            self.down_proj.forward_expert(&(gate * up), expert)
        }

        /// Batched SwiGLU over every routed expert at once. `x` is the expanded token tensor
        /// `[.., 1, 1, d]` and `inds` is `[.., top_k]`; returns `[.., top_k, 1, d]`.
        fn forward_batched(&self, x: &Array, inds: &Array) -> Result<Array> {
            let gate_pre = self.gate_proj.gather(x, inds)?;
            let gate = sigmoid(&gate_pre)? * gate_pre;
            let up = self.up_proj.gather(x, inds)?;
            self.down_proj.gather(&(gate * up), inds)
        }

        fn forward_expert_limited(&self, x: &Array, expert: i32, limit: f32) -> Result<Array> {
            let gate_pre = self.gate_proj.forward_expert(x, expert)?;
            let up_pre = self.up_proj.forward_expert(x, expert)?;
            let (gate_pre, up_pre) = if limit > 0.0 {
                let ceiling = Array::from_f32(limit);
                let floor = Array::from_f32(-limit);
                (
                    minimum(&gate_pre, &ceiling)?,
                    maximum(&minimum(&up_pre, &ceiling)?, &floor)?,
                )
            } else {
                (gate_pre, up_pre)
            };
            let gate = sigmoid(&gate_pre)? * gate_pre;
            self.down_proj.forward_expert(&(gate * up_pre), expert)
        }
    }

    struct MoEGate {
        weight: Array,
        correction_bias: Option<Array>,
        top_k: usize,
        n_group: usize,
        topk_group: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    }

    impl MoEGate {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                weight: take(arrays, &format!("{prefix}.weight"))?,
                correction_bias: arrays
                    .get(&format!("{prefix}.e_score_correction_bias"))
                    .cloned(),
                top_k: config.num_experts_per_tok.unwrap_or(1) as usize,
                n_group: config.n_group.max(1) as usize,
                topk_group: config.topk_group.max(1) as usize,
                norm_topk_prob: config.norm_topk_prob,
                routed_scaling_factor: config.routed_scaling_factor,
            })
        }

        fn route(&self, x: &Array) -> Result<Vec<Vec<(i32, f32)>>> {
            let logits = matmul(x, &self.weight.t())?;
            let scores = sigmoid(&logits)?.as_type::<f32>()?;
            transforms::eval([&scores])?;
            let shape = scores.shape();
            let (b, l, experts) = (shape[0], shape[1], shape[2]);
            if b != 1 {
                bail!("hi-mlx MLA/MoE generation currently supports batch size 1, got {b}");
            }
            let raw_scores = scores.as_slice::<f32>();
            let correction = match &self.correction_bias {
                Some(bias) => {
                    let bias = bias.as_type::<f32>()?;
                    transforms::eval([&bias])?;
                    Some(bias.as_slice::<f32>().to_vec())
                }
                None => None,
            };
            let experts = experts as usize;
            let mut routes = Vec::with_capacity(l as usize);
            for token in 0..l as usize {
                let start = token * experts;
                let raw = &raw_scores[start..start + experts];
                let mut adjusted = raw.to_vec();
                if let Some(correction) = &correction {
                    for (score, bias) in adjusted.iter_mut().zip(correction) {
                        *score += *bias;
                    }
                }
                self.mask_unselected_groups(&mut adjusted);
                let mut ranked = adjusted.iter().copied().enumerate().collect::<Vec<_>>();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                ranked.truncate(self.top_k.min(ranked.len()));
                let mut selected = ranked
                    .into_iter()
                    .map(|(idx, _)| (idx as i32, raw[idx]))
                    .collect::<Vec<_>>();
                if self.norm_topk_prob && selected.len() > 1 {
                    let denom = selected.iter().map(|(_, score)| *score).sum::<f32>();
                    if denom > f32::EPSILON {
                        for (_, score) in &mut selected {
                            *score /= denom;
                        }
                    }
                }
                for (_, score) in &mut selected {
                    *score *= self.routed_scaling_factor;
                }
                routes.push(selected);
            }
            Ok(routes)
        }

        fn mask_unselected_groups(&self, scores: &mut [f32]) {
            if self.n_group <= 1 || self.topk_group >= self.n_group {
                return;
            }
            let group_size = scores.len() / self.n_group;
            if group_size == 0 {
                return;
            }
            let mut groups = (0..self.n_group)
                .map(|group| {
                    let start = group * group_size;
                    let end = if group + 1 == self.n_group {
                        scores.len()
                    } else {
                        start + group_size
                    };
                    let mut top = scores[start..end].to_vec();
                    top.sort_by(|a, b| b.total_cmp(a));
                    let group_score = top.into_iter().take(2).sum::<f32>();
                    (group, group_score)
                })
                .collect::<Vec<_>>();
            groups.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let mut keep = vec![false; self.n_group];
            for (group, _) in groups.into_iter().take(self.topk_group) {
                keep[group] = true;
            }
            for (idx, score) in scores.iter_mut().enumerate() {
                if !keep[(idx / group_size).min(self.n_group - 1)] {
                    *score = f32::NEG_INFINITY;
                }
            }
        }
    }

    struct MoE {
        gate: MoEGate,
        switch_mlp: SwitchMlp,
        shared_experts: Option<Mlp>,
    }

    impl MoE {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                gate: MoEGate::load(&format!("{prefix}.gate"), arrays, config)?,
                switch_mlp: SwitchMlp::load(&format!("{prefix}.switch_mlp"), arrays, config)?,
                shared_experts: if config.n_shared_experts.is_some() {
                    Some(Mlp::load(
                        &format!("{prefix}.shared_experts"),
                        arrays,
                        config,
                    )?)
                } else {
                    None
                },
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l, d) = (shape[0], shape[1], shape[2]);
            if b != 1 {
                bail!("hi-mlx MLA/MoE generation currently supports batch size 1, got {b}");
            }
            let routes = self.gate.route(x)?;
            let mut outputs = Vec::with_capacity(l as usize);
            for token_idx in 0..l {
                let token = x.index((0, token_idx, ..)).reshape(&[1, 1, d])?;
                let mut acc = Array::zeros::<f32>(&[1, 1, d])?;
                for (expert, score) in &routes[token_idx as usize] {
                    acc = acc + self.switch_mlp.forward_expert(&token, *expert)? * *score;
                }
                outputs.push(acc);
            }
            let mut y = concatenate_axis(&outputs, 1)?;
            if let Some(shared) = &self.shared_experts {
                y = y + shared.forward(x)?;
            }
            Ok(y)
        }
    }

    enum MlaFfn {
        Dense(Mlp),
        Moe(MoE),
    }

    impl MlaFfn {
        fn load(
            layer_idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let prefix = format!("model.layers.{layer_idx}.mlp");
            if config.is_moe_layer(layer_idx) {
                Ok(Self::Moe(MoE::load(&prefix, arrays, config)?))
            } else {
                Ok(Self::Dense(Mlp::load(&prefix, arrays, config)?))
            }
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            match self {
                Self::Dense(mlp) => mlp.forward(x),
                Self::Moe(moe) => moe.forward(x),
            }
        }
    }

    struct MlaBlock {
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
        attention: MlaAttention,
        ffn: MlaFfn,
    }

    impl MlaBlock {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let prefix = format!("model.layers.{idx}");
            Ok(Self {
                input_layernorm: RmsNorm::load(
                    &format!("{prefix}.input_layernorm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                post_attention_layernorm: RmsNorm::load(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                attention: MlaAttention::load(&format!("{prefix}.self_attn"), arrays, config)?,
                ffn: MlaFfn::load(idx, arrays, config)?,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let r = self.attention.forward(&self.input_layernorm.forward(&x)?)?;
            let h = x + r;
            let r = self
                .ffn
                .forward(&self.post_attention_layernorm.forward(&h)?)?;
            Ok(h + r)
        }
    }

    // DeepSeek-V3-style multi-token-prediction head (GLM-5.2 layer 78). Given the trunk's pre-norm
    // hidden h_i and the embedding of the next token t_{i+1}, it predicts t_{i+2}:
    //   h' = eh_proj( concat[ hnorm(h_i), enorm(embed(t_{i+1})) ] );  then a full decoder block;
    //   logits = lm_head( shared_head.norm(block(h')) )   (the trunk lm_head is shared).
    // Used as the "draft" for self-speculative decoding; the trunk verifies the proposal.
    struct MtpHead {
        eh_proj: Linear,
        enorm: RmsNorm,
        hnorm: RmsNorm,
        block: MlaBlock,
        shared_norm: RmsNorm,
    }

    impl MtpHead {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let p = format!("model.layers.{idx}");
            Ok(Self {
                eh_proj: Linear::load(&format!("{p}.eh_proj"), arrays, config)?,
                enorm: RmsNorm::load(&format!("{p}.enorm.weight"), arrays, config.rms_norm_eps)?,
                hnorm: RmsNorm::load(&format!("{p}.hnorm.weight"), arrays, config.rms_norm_eps)?,
                block: MlaBlock::load(idx, arrays, config)?,
                shared_norm: RmsNorm::load(
                    &format!("{p}.shared_head.norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
            })
        }

        // h_main: [1,S,hidden] trunk hidden at positions i; next_tokens: [S] the token at i+1 for
        // each; returns logits [1,S,vocab] predicting the token at i+2. Advances the MTP KV cache.
        fn forward(
            &mut self,
            h_main: &Array,
            next_tokens: &[u32],
            embed: &Embedding,
            lm_head: &Linear,
        ) -> Result<Array> {
            let ids = Array::from_slice(next_tokens, &[1, next_tokens.len() as i32]);
            let e = embed.forward(&ids)?;
            // GLM-5.2 orders the eh_proj input as [enorm(embed); hnorm(hidden)] (reverse of the
            // DeepSeek-V3 paper order); HI_MTP_HFIRST switches it for models using the other order.
            let combined = if std::env::var_os("HI_MTP_HFIRST").is_some() {
                concatenate_axis(&[self.hnorm.forward(h_main)?, self.enorm.forward(&e)?], -1)?
            } else {
                concatenate_axis(&[self.enorm.forward(&e)?, self.hnorm.forward(h_main)?], -1)?
            };
            let h = self.eh_proj.forward(&combined)?;
            let h = self.block.forward(h)?;
            lm_head.forward(&self.shared_norm.forward(&h)?)
        }
    }

    struct MlaLike {
        embed_tokens: Embedding,
        layers: Vec<MlaBlock>,
        norm: RmsNorm,
        lm_head: Linear,
        // Optional multi-token-prediction head (DeepSeek-V3 style) for self-speculative decoding,
        // loaded from the extra `num_nextn_predict_layers` layer if present (e.g. GLM-5.2 layer 78).
        mtp: Option<MtpHead>,
    }

    impl MlaLike {
        fn new(config: MlxModelConfig, mut arrays: HashMap<String, Array>) -> Result<Self> {
            prepare_mla_weights(&config, &mut arrays)?;
            let layers = (0..config.num_hidden_layers)
                .map(|idx| MlaBlock::load(idx, &arrays, &config))
                .collect::<Result<Vec<_>>>()?;
            // The MTP head is the first "next-n" layer (index num_hidden_layers). Load it if present.
            let mtp = if config.num_nextn_predict_layers.unwrap_or(0) >= 1
                && arrays.contains_key(&format!(
                    "model.layers.{}.eh_proj.weight",
                    config.num_hidden_layers
                )) {
                Some(MtpHead::load(config.num_hidden_layers, &arrays, &config)?)
            } else {
                None
            };
            Ok(Self {
                embed_tokens: Embedding::load("model.embed_tokens", &arrays, &config)?,
                norm: RmsNorm::load("model.norm.weight", &arrays, config.rms_norm_eps)?,
                layers,
                lm_head: Linear::load("lm_head", &arrays, &config)?,
                mtp,
            })
        }

        // Run the trunk; return (logits, pre-final-norm hidden) for all positions. The hidden feeds
        // the MTP head; logits drive normal generation.
        fn forward_hidden(&mut self, input_ids: &[u32]) -> Result<(Array, Array)> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let mut h = self.embed_tokens.forward(&ids)?;
            for layer in &mut self.layers {
                h = layer.forward(h)?;
            }
            let logits = self.lm_head.forward(&self.norm.forward(&h)?)?;
            Ok((logits, h))
        }
    }

    impl CausalLm for MlaLike {
        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let (logits, _h) = self.forward_hidden(input_ids)?;
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn reset_cache(&mut self) {
            for layer in &mut self.layers {
                layer.attention.reset_cache();
            }
            if let Some(mtp) = &mut self.mtp {
                mtp.block.attention.reset_cache();
            }
        }

        fn prepare_cache(&mut self, capacity: i32) {
            for layer in &mut self.layers {
                layer.attention.cache.prepare_capacity(capacity);
                if let Some(indexer) = &mut layer.attention.indexer {
                    indexer.cache.prepare_capacity(capacity);
                }
            }
            if let Some(mtp) = &mut self.mtp {
                mtp.block.attention.cache.prepare_capacity(capacity);
                if let Some(indexer) = &mut mtp.block.attention.indexer {
                    indexer.cache.prepare_capacity(capacity);
                }
            }
        }

        fn rollback_cache(&mut self, to_offset: i32) {
            for layer in &mut self.layers {
                layer.attention.cache.rollback(to_offset);
                if let Some(indexer) = &mut layer.attention.indexer {
                    indexer.cache.rollback(to_offset);
                }
            }
            if let Some(mtp) = &mut self.mtp {
                mtp.block.attention.cache.rollback(to_offset);
                if let Some(indexer) = &mut mtp.block.attention.indexer {
                    indexer.cache.rollback(to_offset);
                }
            }
        }

        fn supports_rollback(&self) -> bool {
            true
        }

        fn supports_mtp(&self) -> bool {
            self.mtp.is_some()
        }

        fn mtp_generate(
            &mut self,
            config: &MlxModelConfig,
            tokenizer: &crate::generate::TokenizerRuntime,
            request: &GenerationRequest,
            on_event: &mut dyn FnMut(GenerationEvent) -> Result<()>,
        ) -> Result<GenerationOutput> {
            if self.mtp.is_none() {
                bail!("model has no MTP head");
            }
            let prompt_tokens = tokenizer.encode(&request.prompt)?;
            if prompt_tokens.is_empty() {
                bail!("prompt encoded to zero tokens");
            }
            let max_tokens = request.max_tokens.max(1) as usize;
            self.reset_cache();
            let cap = (prompt_tokens.len() + max_tokens + 4).min(i32::MAX as usize) as i32;
            self.prepare_cache(cap);

            // Prefill the trunk in one pass; keep all-position hidden for the MTP prefill.
            let (logits0, hidden0) = self.forward_hidden(&prompt_tokens)?;
            let p = prompt_tokens.len() as i32;
            let mut t0 = argmax_rows(&logits0.index((.., (p - 1)..p, ..)))?[0];
            let mut h_last = hidden0.index((.., (p - 1)..p, ..)); // trunk hidden at P-1

            // Prefill the MTP over positions 0..P-2 (h_i paired with prompt[i+1]) -> MTP cache = P-1.
            if p >= 2 {
                let h_slice = hidden0.index((.., 0..(p - 1), ..));
                let next: Vec<u32> = prompt_tokens[1..p as usize].to_vec();
                let mtp = self.mtp.as_mut().unwrap();
                let _ = mtp.forward(&h_slice, &next, &self.embed_tokens, &self.lm_head)?;
            }

            let mut m = p; // committed length; trunk cache = m, MTP cache = m-1
            let mut generated: Vec<u32> = Vec::new();
            let mut decoded_text = String::new();
            let (mut rounds, mut proposed, mut accepted) = (0usize, 0usize, 0usize);
            let mut stop = false;

            // commit helper: push token, emit delta, return true if generation should stop
            macro_rules! commit {
                ($tok:expr) => {{
                    let tok = $tok;
                    generated.push(tok);
                    let current = tokenizer.decode(&generated)?;
                    let delta = decoded_delta(&decoded_text, &current, tokenizer, tok)?;
                    decoded_text = current;
                    on_event(GenerationEvent::TokenDelta { token_id: tok, text: delta })?;
                    generated.len() >= max_tokens || hit_stop(&generated, &config.eos_token_ids)
                }};
            }

            while generated.len() < max_tokens && !stop {
                rounds += 1;
                // 1. MTP proposes t1 from (h_last, t0); MTP cache m-1 -> m.
                let t1 = {
                    let mtp = self.mtp.as_mut().unwrap();
                    let ml = mtp.forward(&h_last, &[t0], &self.embed_tokens, &self.lm_head)?;
                    argmax_rows(&ml)?[0]
                };
                proposed += 1;

                // 2. Trunk verifies [t0, t1]; trunk cache m -> m+2.
                let (tl, th) = self.forward_hidden(&[t0, t1])?;
                let ta = argmax_rows(&tl)?; // ta[0]=trunk token @ m+1, ta[1]=trunk token @ m+2
                let th0 = th.index((.., 0..1, ..)); // trunk hidden @ m
                let th1 = th.index((.., 1..2, ..)); // trunk hidden @ m+1

                if t1 == ta[0] {
                    // MTP correct: commit t0 and t1.
                    accepted += 1;
                    if commit!(t0) {
                        break;
                    }
                    if commit!(t1) {
                        break;
                    }
                    // MTP catch-up over position m (h_m paired with t1); MTP cache m -> m+1.
                    {
                        let mtp = self.mtp.as_mut().unwrap();
                        let _ = mtp.forward(&th0, &[t1], &self.embed_tokens, &self.lm_head)?;
                    }
                    t0 = ta[1];
                    h_last = th1;
                    m += 2;
                } else {
                    // MTP wrong: commit t0 and the trunk's correction c.
                    let c = ta[0];
                    if commit!(t0) {
                        break;
                    }
                    if commit!(c) {
                        break;
                    }
                    // MTP catch-up over position m (h_m paired with c); MTP cache m -> m+1.
                    {
                        let mtp = self.mtp.as_mut().unwrap();
                        let _ = mtp.forward(&th0, &[c], &self.embed_tokens, &self.lm_head)?;
                    }
                    // Trunk: drop the rejected t1, process c to get the next state.
                    for layer in &mut self.layers {
                        layer.attention.cache.rollback(m + 1);
                        if let Some(indexer) = &mut layer.attention.indexer {
                            indexer.cache.rollback(m + 1);
                        }
                    }
                    let (lc, hc) = self.forward_hidden(&[c])?;
                    t0 = argmax_rows(&lc)?[0];
                    h_last = hc;
                    m += 2;
                }
                stop = generated.len() >= max_tokens;
            }

            let text = tokenizer.decode(&generated)?;
            let output = GenerationOutput {
                prompt_tokens: prompt_tokens.len() as u64,
                completion_tokens: generated.len() as u64,
                text,
            };
            let rate = if proposed > 0 {
                accepted as f64 / proposed as f64 * 100.0
            } else {
                0.0
            };
            tracing::info!(
                "MTP self-speculation: {} tok over {rounds} rounds, MTP accept {rate:.0}% ({accepted}/{proposed})",
                generated.len()
            );
            on_event(GenerationEvent::Finished { output: output.clone() })?;
            Ok(output)
        }
    }

    enum V4GroupedLinear {
        Dense {
            weight: Array,
            bias: Option<Array>,
            groups: i32,
            rank: i32,
        },
        Quantized {
            weight: Array,
            scales: Array,
            biases: Option<Array>,
            bias: Option<Array>,
            group_size: i32,
            bits: i32,
            mode: String,
            groups: i32,
            rank: i32,
        },
    }

    impl V4GroupedLinear {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let groups = config.o_groups.unwrap_or(1) as i32;
            let rank = config
                .o_lora_rank
                .ok_or_else(|| anyhow!("config.json missing o_lora_rank for DeepSeek V4"))?
                as i32;
            let weight = take(arrays, &format!("{prefix}.weight"))?;
            let bias = arrays.get(&format!("{prefix}.bias")).cloned();
            match arrays.get(&format!("{prefix}.scales")) {
                Some(scales) => {
                    let spec = quant_spec_for(config, prefix, &weight, Some(scales))?;
                    let biases = arrays.get(&format!("{prefix}.biases")).cloned();
                    require_biases_for_affine(prefix, &spec, biases.as_ref())?;
                    Ok(Self::Quantized {
                        weight,
                        scales: scales.clone(),
                        biases,
                        bias,
                        group_size: spec.group_size as i32,
                        bits: spec.bits as i32,
                        mode: spec.mode.as_str().to_string(),
                        groups,
                        rank,
                    })
                }
                None => Ok(Self::Dense {
                    weight,
                    bias,
                    groups,
                    rank,
                }),
            }
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, s, features) = (shape[0], shape[1], shape[2]);
            let (groups, rank) = match self {
                Self::Dense { groups, rank, .. } | Self::Quantized { groups, rank, .. } => {
                    (*groups, *rank)
                }
            };
            let group_features = features / groups;
            let x = x.reshape(&[b, s, groups, group_features])?;
            let mut pieces = Vec::with_capacity(groups as usize);
            for group in 0..groups {
                let rows = group * rank..(group + 1) * rank;
                let xg = x.index((.., .., group, ..));
                let y = match self {
                    Self::Dense { weight, bias, .. } => {
                        let wg = weight.index((rows.clone(), ..));
                        let mut y = matmul(&xg, &wg.t())?;
                        if let Some(bias) = bias {
                            y = y + bias.index(rows.clone());
                        }
                        y
                    }
                    Self::Quantized {
                        weight,
                        scales,
                        biases,
                        bias,
                        group_size,
                        bits,
                        mode,
                        ..
                    } => {
                        let wg = weight.index((rows.clone(), ..));
                        let sg = scales.index((rows.clone(), ..));
                        let bg = biases
                            .as_ref()
                            .map(|biases| biases.index((rows.clone(), ..)));
                        let mut y = quantized_matmul_mode(
                            &xg,
                            &wg,
                            &sg,
                            bg.as_ref(),
                            true,
                            *group_size,
                            *bits,
                            mode,
                        )?;
                        if let Some(bias) = bias {
                            y = y + bias.index(rows.clone());
                        }
                        y
                    }
                };
                pieces.push(y);
            }
            concatenate_axis(&pieces, -1).map_err(Into::into)
        }
    }

    struct HyperConnection {
        func: Array,
        base: Array,
        scale: Array,
        hidden_size: i32,
        hc_mult: i32,
        eps: f32,
        sinkhorn_iters: i32,
        hc_eps: f32,
    }

    impl HyperConnection {
        fn load(
            prefixes: &[String],
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                func: take_any(arrays, prefixes, "fn")?,
                base: take_any(arrays, prefixes, "base")?,
                scale: take_any(arrays, prefixes, "scale")?,
                hidden_size: config.hidden_size as i32,
                hc_mult: config.hc_mult as i32,
                eps: config.rms_norm_eps,
                sinkhorn_iters: config.hc_sinkhorn_iters as i32,
                hc_eps: config.hc_eps,
            })
        }

        fn pre(&self, x: &Array) -> Result<(Array, Array, Array)> {
            let shape = x.shape();
            let (b, s) = (shape[0], shape[1]);
            let n = b * s;
            let xf = x
                .reshape(&[b, s, self.hc_mult * self.hidden_size])?
                .as_type::<f32>()?;
            let inv = rsqrt(&(mean_axis(&(xf.clone() * &xf), -1, Some(true))? + self.eps))?;
            let mixes = (matmul(&xf, &self.func.t())? * inv).reshape(&[n, -1])?;

            let hc = self.hc_mult;
            let pre_log = mixes.index((.., ..hc)) * self.scale.index(0) + self.base.index(..hc);
            let post_log = mixes.index((.., hc..(2 * hc))) * self.scale.index(1)
                + self.base.index(hc..(2 * hc));
            let comb_log = mixes.index((.., (2 * hc)..)).reshape(&[n, hc, hc])?
                * self.scale.index(2)
                + self.base.index((2 * hc)..).reshape(&[hc, hc])?;

            let pre = sigmoid(&pre_log)? + self.hc_eps;
            let post = sigmoid(&post_log)? * 2.0;
            let mut comb = softmax_axis(&comb_log, -1, Some(true))? + self.hc_eps;
            comb = comb.clone() / (sum_axis(&comb, 1, Some(true))? + self.hc_eps);
            for _ in 1..self.sinkhorn_iters {
                comb = comb.clone() / (sum_axis(&comb, 2, Some(true))? + self.hc_eps);
                comb = comb.clone() / (sum_axis(&comb, 1, Some(true))? + self.hc_eps);
            }

            let pre = pre.reshape(&[b, s, hc])?;
            let post = post.reshape(&[b, s, hc])?;
            let comb = comb.reshape(&[b, s, hc, hc])?;
            let y = sum_axis(
                &(pre.expand_dims(-1)? * x.as_type::<f32>()?),
                2,
                Some(false),
            )?;
            Ok((y, post, comb))
        }

        fn post(
            &self,
            f_out: &Array,
            residual: &Array,
            post: &Array,
            comb: &Array,
        ) -> Result<Array> {
            let term_new = post.expand_dims(-1)? * f_out.expand_dims(2)?.as_type::<f32>()?;
            let comb = comb.as_type::<f32>()?;
            let residual = residual.as_type::<f32>()?;
            let term_res = einsum("bsij,bsjd->bsid", [&comb, &residual])?;
            Ok(term_new + term_res)
        }
    }

    struct HyperHead {
        func: Array,
        base: Array,
        scale: Array,
        hidden_size: i32,
        hc_mult: i32,
        eps: f32,
        hc_eps: f32,
    }

    impl HyperHead {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                func: take(arrays, &format!("{prefix}.fn"))?,
                base: take(arrays, &format!("{prefix}.base"))?,
                scale: take(arrays, &format!("{prefix}.scale"))?,
                hidden_size: config.hidden_size as i32,
                hc_mult: config.hc_mult as i32,
                eps: config.rms_norm_eps,
                hc_eps: config.hc_eps,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, s) = (shape[0], shape[1]);
            let xf = x
                .reshape(&[b, s, self.hc_mult * self.hidden_size])?
                .as_type::<f32>()?;
            let inv = rsqrt(&(mean_axis(&(xf.clone() * &xf), -1, Some(true))? + self.eps))?;
            let mixes = matmul(&xf, &self.func.t())? * inv;
            let pre = sigmoid(&(mixes * self.scale.index(0) + &self.base))? + self.hc_eps;
            sum_axis(
                &(pre.expand_dims(-1)? * x.as_type::<f32>()?),
                2,
                Some(false),
            )
            .map_err(Into::into)
        }
    }

    struct V4Attention {
        wq_a: Linear,
        q_norm: RmsNorm,
        wq_b: Linear,
        wkv: Linear,
        kv_norm: RmsNorm,
        attn_sink: Option<Array>,
        wo_a: V4GroupedLinear,
        wo_b: Linear,
        cache: Cache,
        compressor: Option<V4Compressor>,
        indexer: Option<V4Indexer>,
        compressed_mask_cache: HashMap<(i32, i32, i32, i32), Array>,
        compress_ratio: i32,
        num_heads: i32,
        head_dim: i32,
        rope_head_dim: i32,
        nope_head_dim: i32,
        scale: f32,
        rope_theta: f32,
        eps: f32,
    }

    impl V4Attention {
        fn load(
            layer_idx: u32,
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            let rope_head_dim = config.qk_rope_head_dim.unwrap_or(0) as i32;
            let compress_ratio = config
                .compress_ratios
                .get(layer_idx as usize)
                .copied()
                .unwrap_or(0);
            let compressor = if compress_ratio > 0 {
                Some(V4Compressor::load(
                    &format!("{prefix}.compressor"),
                    arrays,
                    config,
                    compress_ratio as i32,
                    head_dim,
                )?)
            } else {
                None
            };
            let indexer = if compress_ratio == 4 {
                Some(V4Indexer::load(
                    &format!("{prefix}.indexer"),
                    arrays,
                    config,
                    compress_ratio as i32,
                )?)
            } else {
                None
            };
            Ok(Self {
                wq_a: Linear::load(&format!("{prefix}.wq_a"), arrays, config)?,
                q_norm: RmsNorm::load(
                    &format!("{prefix}.q_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                wq_b: Linear::load(&format!("{prefix}.wq_b"), arrays, config)?,
                wkv: Linear::load(&format!("{prefix}.wkv"), arrays, config)?,
                kv_norm: RmsNorm::load(
                    &format!("{prefix}.kv_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                attn_sink: arrays.get(&format!("{prefix}.attn_sink")).cloned(),
                wo_a: V4GroupedLinear::load(&format!("{prefix}.wo_a"), arrays, config)?,
                wo_b: Linear::load(&format!("{prefix}.wo_b"), arrays, config)?,
                cache: Cache::with_max_len(config.sliding_window.map(|window| window as i32)),
                compressor,
                indexer,
                compressed_mask_cache: HashMap::new(),
                compress_ratio: compress_ratio as i32,
                num_heads: config.num_attention_heads as i32,
                head_dim,
                rope_head_dim,
                nope_head_dim: head_dim - rope_head_dim,
                scale: (head_dim as f32).powf(-0.5),
                rope_theta: if compress_ratio == 0 {
                    config.rope_theta
                } else {
                    config.compress_rope_theta
                },
                eps: config.rms_norm_eps,
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, s) = (shape[0], shape[1]);
            let qr = self.q_norm.forward(&self.wq_a.forward(x)?)?;
            let mut q = self
                .wq_b
                .forward(&qr)?
                .reshape(&[b, s, self.num_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            q = q.clone() * rsqrt(&(mean_axis(&(q.clone() * &q), -1, Some(true))? + self.eps))?;

            let mut kv = self
                .kv_norm
                .forward(&self.wkv.forward(x)?)?
                .reshape(&[b, s, 1, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;

            let offset = self.cache.offset;
            if self.rope_head_dim > 0 {
                let mut q_parts = split_sections(&q, &[self.nope_head_dim], -1)?;
                let q_nope = q_parts.remove(0);
                let q_pe = v4_rope(
                    &q_parts.remove(0),
                    self.rope_head_dim,
                    self.rope_theta,
                    offset,
                    false,
                )?;
                q = concatenate_axis(&[q_nope, q_pe], -1)?;

                let mut k_parts = split_sections(&kv, &[self.nope_head_dim], -1)?;
                let k_nope = k_parts.remove(0);
                let k_pe = v4_rope(
                    &k_parts.remove(0),
                    self.rope_head_dim,
                    self.rope_theta,
                    offset,
                    false,
                )?;
                kv = concatenate_axis(&[k_nope, k_pe], -1)?;
            }

            let (k, v, key_start) = self.cache.update_with_start(kv.clone(), kv)?;
            let raw_mask = causal_attention_mask_with_key_start_and_window(
                s,
                k.shape()[2],
                offset,
                key_start,
                self.cache.max_len,
            );
            let (k, v, mask) = self.combined_kv_and_mask(x, &qr, offset, k, v, raw_mask)?;
            let mut out = match &mask {
                Some(mask) => scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    ScaledDotProductAttentionMask::Array(mask),
                    self.attn_sink.as_ref(),
                )?,
                None => scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    None,
                    self.attn_sink.as_ref(),
                )?,
            };

            if self.rope_head_dim > 0 {
                let mut out_parts = split_sections(&out, &[self.nope_head_dim], -1)?;
                let out_nope = out_parts.remove(0);
                let out_pe = v4_rope(
                    &out_parts.remove(0),
                    self.rope_head_dim,
                    self.rope_theta,
                    offset,
                    true,
                )?;
                out = concatenate_axis(&[out_nope, out_pe], -1)?;
            }
            let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                s,
                self.num_heads * self.head_dim,
            ])?;
            self.wo_b.forward(&self.wo_a.forward(&out)?)
        }

        fn combined_kv_and_mask(
            &mut self,
            x: &Array,
            query_latent: &Array,
            offset: i32,
            raw_k: Array,
            raw_v: Array,
            raw_mask: Array,
        ) -> Result<(Array, Array, Option<Array>)> {
            let Some(compressor) = self.compressor.as_mut() else {
                let mask = if raw_mask.shape()[3] > 1 || self.cache.max_len.is_some() {
                    Some(raw_mask)
                } else {
                    None
                };
                return Ok((raw_k, raw_v, mask));
            };
            let Some((mut compressed_k, mut compressed_v)) = compressor.update(x, offset)? else {
                return Ok((raw_k, raw_v, Some(raw_mask)));
            };

            let b = raw_k.shape()[0];
            let query_len = raw_mask.shape()[2];
            let mut compressed_mask =
                self.cached_compressed_attention_mask(query_len, compressed_k.shape()[2], offset);

            if let Some(indexer) = self.indexer.as_mut()
                && let Some(topk_indices) = indexer.forward(x, query_latent, offset)?
            {
                if query_len == 1 {
                    let idx = topk_indices.index((.., .., 0, ..)).expand_dims(-1)?;
                    let idx_k =
                        broadcast_to(&idx, &[b, 1, idx.shape()[2], compressed_k.shape()[3]])?;
                    let idx_v =
                        broadcast_to(&idx, &[b, 1, idx.shape()[2], compressed_v.shape()[3]])?;
                    compressed_k = take_along_axis(&compressed_k, &idx_k, Some(2))?;
                    compressed_v = take_along_axis(&compressed_v, &idx_v, Some(2))?;
                    compressed_mask = self.cached_compressed_attention_mask(
                        query_len,
                        compressed_k.shape()[2],
                        offset,
                    );
                } else {
                    let sparse_shape = [b, 1, query_len, compressed_k.shape()[2]];
                    let sparse = Array::zeros::<bool>(&sparse_shape)?;
                    let sparse =
                        put_along_axis(&sparse, &topk_indices, &Array::from_bool(true), -1)?;
                    compressed_mask = compressed_mask.logical_and(&sparse)?;
                }
            }

            let k = concatenate_axis(&[compressed_k, raw_k], 2)?;
            let v = concatenate_axis(&[compressed_v, raw_v], 2)?;
            let mask = concatenate_axis(&[compressed_mask, raw_mask], -1)?;
            Ok((k, v, Some(mask)))
        }

        fn cached_compressed_attention_mask(
            &mut self,
            query_len: i32,
            compressed_len: i32,
            offset: i32,
        ) -> Array {
            let key = (query_len, compressed_len, offset, self.compress_ratio);
            if let Some(mask) = self.compressed_mask_cache.get(&key) {
                return mask.clone();
            }
            if self.compressed_mask_cache.len() > 64 {
                self.compressed_mask_cache.clear();
            }
            let mask =
                compressed_attention_mask(query_len, compressed_len, offset, self.compress_ratio);
            self.compressed_mask_cache.insert(key, mask.clone());
            mask
        }

        fn reset_cache(&mut self) {
            self.cache.reset();
            self.compressed_mask_cache.clear();
            if let Some(compressor) = &mut self.compressor {
                compressor.reset();
            }
            if let Some(indexer) = &mut self.indexer {
                indexer.reset();
            }
        }

        fn prepare_cache(&mut self, capacity: i32) {
            self.cache.prepare_capacity(capacity);
            if let Some(compressor) = &mut self.compressor {
                compressor.prepare_capacity(capacity);
            }
            if let Some(indexer) = &mut self.indexer {
                indexer.prepare_capacity(capacity);
            }
        }
    }

    struct V4Compressor {
        wgate: Linear,
        wkv: Linear,
        norm: RmsNorm,
        ape: Array,
        cache: Cache,
        pending: Option<Array>,
        pending_start: i32,
        ratio: i32,
        head_dim: i32,
    }

    impl V4Compressor {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            ratio: i32,
            head_dim: i32,
        ) -> Result<Self> {
            Ok(Self {
                wgate: Linear::load(&format!("{prefix}.wgate"), arrays, config)?,
                wkv: Linear::load(&format!("{prefix}.wkv"), arrays, config)?,
                norm: RmsNorm::load(
                    &format!("{prefix}.norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                ape: take(arrays, &format!("{prefix}.ape"))?,
                cache: Cache::new(),
                pending: None,
                pending_start: 0,
                ratio,
                head_dim,
            })
        }

        fn update(&mut self, x: &Array, offset: i32) -> Result<Option<(Array, Array)>> {
            if self.pending.is_none() {
                self.pending_start = offset;
            }
            let combined = match self.pending.take() {
                Some(pending) => concatenate_axis(&[pending, x.clone()], 1)?,
                None => x.clone(),
            };
            let len = combined.shape()[1];
            let complete_len = (len / self.ratio) * self.ratio;
            if complete_len == 0 {
                self.pending = Some(combined);
                return self.cached();
            }

            let complete = combined.index((.., ..complete_len, ..));
            if complete_len < len {
                self.pending = Some(combined.index((.., complete_len.., ..)));
                self.pending_start += complete_len;
            } else {
                self.pending = None;
                self.pending_start = offset + x.shape()[1];
            }

            let (new_k, new_v) = self.compress_complete(&complete)?;
            self.cache.update(new_k, new_v)?;
            self.cached()
        }

        fn cached(&self) -> Result<Option<(Array, Array)>> {
            Ok(self
                .cache
                .key
                .as_ref()
                .zip(self.cache.value.as_ref())
                .map(|(key, value)| (key.clone(), value.clone())))
        }

        fn compress_complete(&self, x: &Array) -> Result<(Array, Array)> {
            let shape = x.shape();
            let (b, s) = (shape[0], shape[1]);
            let blocks = s / self.ratio;
            let out_dim = self.head_dim * 2;
            let gate = self
                .wgate
                .forward(x)?
                .reshape(&[b, blocks, self.ratio, out_dim])?
                + self.ape.reshape(&[1, 1, self.ratio, out_dim])?;
            let weights = softmax_axis(&gate, 2, Some(true))?;
            let kv = self
                .wkv
                .forward(x)?
                .reshape(&[b, blocks, self.ratio, out_dim])?;
            let compressed = sum_axis(&(weights * kv), 2, Some(false))?;
            let mut parts = split_sections(&compressed, &[self.head_dim], -1)?;
            let k = self.norm.forward(&parts.remove(0))?.expand_dims(1)?;
            let v = self.norm.forward(&parts.remove(0))?.expand_dims(1)?;
            Ok((k, v))
        }

        fn reset(&mut self) {
            self.cache.reset();
            self.pending = None;
            self.pending_start = 0;
        }

        fn prepare_capacity(&mut self, capacity: i32) {
            let compressed_capacity = (capacity + self.ratio - 1) / self.ratio;
            self.cache.prepare_capacity(compressed_capacity.max(1));
            self.pending = None;
            self.pending_start = 0;
        }
    }

    struct V4Indexer {
        compressor: V4Compressor,
        wq_b: Linear,
        weights_proj: Linear,
        n_heads: i32,
        head_dim: i32,
        index_topk: i32,
        ratio: i32,
        scale: f32,
    }

    impl V4Indexer {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            ratio: i32,
        ) -> Result<Self> {
            let head_dim = config.index_head_dim.ok_or_else(|| {
                anyhow!("config.json missing index_head_dim for DeepSeek V4 indexer")
            })? as i32;
            let n_heads = config.index_n_heads.ok_or_else(|| {
                anyhow!("config.json missing index_n_heads for DeepSeek V4 indexer")
            })? as i32;
            let index_topk = config
                .index_topk
                .ok_or_else(|| anyhow!("config.json missing index_topk for DeepSeek V4 indexer"))?
                as i32;
            Ok(Self {
                compressor: V4Compressor::load(
                    &format!("{prefix}.compressor"),
                    arrays,
                    config,
                    ratio,
                    head_dim,
                )?,
                wq_b: Linear::load(&format!("{prefix}.wq_b"), arrays, config)?,
                weights_proj: Linear::load(&format!("{prefix}.weights_proj"), arrays, config)?,
                n_heads,
                head_dim,
                index_topk,
                ratio,
                scale: (head_dim as f32).powf(-0.5),
            })
        }

        fn forward(
            &mut self,
            x: &Array,
            query_latent: &Array,
            offset: i32,
        ) -> Result<Option<Array>> {
            let Some((compressed_k, _)) = self.compressor.update(x, offset)? else {
                return Ok(None);
            };
            let compressed_len = compressed_k.shape()[2];
            if compressed_len <= self.index_topk {
                return Ok(None);
            }
            let shape = x.shape();
            let (b, s) = (shape[0], shape[1]);
            let q = self
                .wq_b
                .forward(query_latent)?
                .reshape(&[b, s, self.n_heads, self.head_dim])?
                .swap_axes(1, 2)?;
            let mut scores = matmul(&(q * self.scale), &compressed_k.swap_axes(-1, -2)?)?;
            scores = maximum(&scores, &Array::from_f32(0.0))?;
            let weights = self.weights_proj.forward(x)? * (self.n_heads as f32).powf(-0.5);
            let weights = weights.swap_axes(-1, -2)?.expand_dims(-1)?;
            scores = sum_axis(&(scores * weights), 1, Some(true))?;
            let mask = compressed_attention_mask(s, compressed_len, offset, self.ratio);
            scores = apply_attention_mask(&scores, &mask)?;
            let partitioned = argpartition_axis(&scores, -self.index_topk, -1)?;
            Ok(Some(partitioned.index((.., .., .., (-self.index_topk)..))))
        }

        fn reset(&mut self) {
            self.compressor.reset();
        }

        fn prepare_capacity(&mut self, capacity: i32) {
            self.compressor.prepare_capacity(capacity);
        }
    }

    struct V4MoEGate {
        weight: Array,
        correction_bias: Option<Array>,
        tid2eid: Option<Array>,
        hash: bool,
        top_k: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        scoring_func: String,
    }

    impl V4MoEGate {
        fn load(
            prefix: &str,
            layer_idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                weight: take(arrays, &format!("{prefix}.weight"))?,
                correction_bias: arrays
                    .get(&format!("{prefix}.e_score_correction_bias"))
                    .cloned(),
                tid2eid: arrays.get(&format!("{prefix}.tid2eid")).cloned(),
                hash: layer_idx < config.num_hash_layers,
                top_k: config.num_experts_per_tok.unwrap_or(1) as usize,
                norm_topk_prob: config.norm_topk_prob,
                routed_scaling_factor: config.routed_scaling_factor,
                scoring_func: config
                    .scoring_func
                    .clone()
                    .unwrap_or_else(|| "sqrtsoftplus".to_string()),
            })
        }

        fn route(&self, x: &Array, input_ids: &[u32]) -> Result<Vec<Vec<(i32, f32)>>> {
            let logits = matmul(x, &self.weight.t())?.as_type::<f32>()?;
            transforms::eval([&logits])?;
            let shape = logits.shape();
            let (b, s, experts) = (shape[0], shape[1], shape[2]);
            if b != 1 {
                bail!("hi-mlx DeepSeek V4 MoE generation currently supports batch size 1, got {b}");
            }
            let experts = experts as usize;
            let raw_logits = logits.as_slice::<f32>();
            let correction = match &self.correction_bias {
                Some(bias) => {
                    let bias = bias.as_type::<f32>()?;
                    transforms::eval([&bias])?;
                    Some(bias.as_slice::<f32>().to_vec())
                }
                None => None,
            };
            let tid2eid = match (&self.tid2eid, self.hash) {
                (Some(tid2eid), true) => {
                    let tid2eid = tid2eid.as_type::<i32>()?;
                    transforms::eval([&tid2eid])?;
                    Some(tid2eid.as_slice::<i32>().to_vec())
                }
                _ => None,
            };

            let mut routes = Vec::with_capacity(s as usize);
            for token in 0..s as usize {
                let start = token * experts;
                let scores = score_v4(&raw_logits[start..start + experts], &self.scoring_func);
                let selected = if self.hash {
                    let table = tid2eid
                        .as_ref()
                        .ok_or_else(|| anyhow!("DeepSeek V4 hash gate missing tid2eid tensor"))?;
                    let token_id = input_ids
                        .get(token)
                        .copied()
                        .unwrap_or_default()
                        .min((table.len() / self.top_k).saturating_sub(1) as u32)
                        as usize;
                    (0..self.top_k)
                        .map(|idx| table[token_id * self.top_k + idx] as usize)
                        .collect::<Vec<_>>()
                } else {
                    let mut adjusted = scores.clone();
                    if let Some(correction) = &correction {
                        for (score, bias) in adjusted.iter_mut().zip(correction) {
                            *score += *bias;
                        }
                    }
                    let mut ranked = adjusted.iter().copied().enumerate().collect::<Vec<_>>();
                    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    ranked
                        .into_iter()
                        .take(self.top_k.min(experts))
                        .map(|(idx, _)| idx)
                        .collect::<Vec<_>>()
                };
                let mut routed = selected
                    .into_iter()
                    .map(|idx| (idx as i32, scores[idx]))
                    .collect::<Vec<_>>();
                if self.scoring_func != "softmax" && self.norm_topk_prob && routed.len() > 1 {
                    let denom = routed.iter().map(|(_, score)| *score).sum::<f32>();
                    if denom > f32::EPSILON {
                        for (_, score) in &mut routed {
                            *score /= denom;
                        }
                    }
                }
                for (_, score) in &mut routed {
                    *score *= self.routed_scaling_factor;
                }
                routes.push(routed);
            }
            Ok(routes)
        }
    }

    struct V4MoE {
        gate: V4MoEGate,
        switch_mlp: SwitchMlp,
        shared_experts: Option<Mlp>,
        swiglu_limit: f32,
    }

    impl V4MoE {
        fn load(
            prefix: &str,
            layer_idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                gate: V4MoEGate::load(&format!("{prefix}.gate"), layer_idx, arrays, config)?,
                switch_mlp: SwitchMlp::load(&format!("{prefix}.switch_mlp"), arrays, config)?,
                shared_experts: if config.n_shared_experts.unwrap_or(0) > 0
                    && arrays.contains_key(&format!("{prefix}.shared_experts.gate_proj.weight"))
                {
                    Some(Mlp::load(
                        &format!("{prefix}.shared_experts"),
                        arrays,
                        config,
                    )?)
                } else {
                    None
                },
                swiglu_limit: config.swiglu_limit.unwrap_or(0.0),
            })
        }

        fn forward(&self, x: &Array, input_ids: &[u32]) -> Result<Array> {
            let shape = x.shape();
            let (b, s, d) = (shape[0], shape[1], shape[2]);
            if b != 1 {
                bail!("hi-mlx DeepSeek V4 MoE generation currently supports batch size 1, got {b}");
            }
            let routes = self.gate.route(x, input_ids)?;
            let mut outputs = Vec::with_capacity(s as usize);
            for token_idx in 0..s {
                let token = x.index((0, token_idx, ..)).reshape(&[1, 1, d])?;
                let mut acc = Array::zeros::<f32>(&[1, 1, d])?;
                for (expert, score) in &routes[token_idx as usize] {
                    acc = acc
                        + self.switch_mlp.forward_expert_limited(
                            &token,
                            *expert,
                            self.swiglu_limit,
                        )? * *score;
                }
                outputs.push(acc);
            }
            let mut y = concatenate_axis(&outputs, 1)?;
            if let Some(shared) = &self.shared_experts {
                y = y + shared.forward(x)?;
            }
            Ok(y)
        }
    }

    struct V4Block {
        attn_norm: RmsNorm,
        attention: V4Attention,
        hc_attn: HyperConnection,
        ffn_norm: RmsNorm,
        ffn: V4MoE,
        hc_ffn: HyperConnection,
    }

    impl V4Block {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let prefix = format!("model.layers.{idx}");
            Ok(Self {
                attn_norm: RmsNorm::load(
                    &format!("{prefix}.attn_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                attention: V4Attention::load(idx, &format!("{prefix}.attn"), arrays, config)?,
                hc_attn: HyperConnection::load(
                    &[format!("{prefix}.attn_hc"), format!("{prefix}.hc_attn")],
                    arrays,
                    config,
                )?,
                ffn_norm: RmsNorm::load(
                    &format!("{prefix}.ffn_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                ffn: V4MoE::load(&format!("{prefix}.ffn"), idx, arrays, config)?,
                hc_ffn: HyperConnection::load(
                    &[format!("{prefix}.ffn_hc"), format!("{prefix}.hc_ffn")],
                    arrays,
                    config,
                )?,
            })
        }

        fn forward(&mut self, h: Array, input_ids: &[u32]) -> Result<Array> {
            let residual = h.clone();
            let (y, post, comb) = self.hc_attn.pre(&h)?;
            let y = self.attention.forward(&self.attn_norm.forward(&y)?)?;
            let h = self.hc_attn.post(&y, &residual, &post, &comb)?;

            let residual = h.clone();
            let (y, post, comb) = self.hc_ffn.pre(&h)?;
            let y = self.ffn.forward(&self.ffn_norm.forward(&y)?, input_ids)?;
            self.hc_ffn.post(&y, &residual, &post, &comb)
        }
    }

    struct DeepSeekV4Like {
        embed_tokens: Embedding,
        layers: Vec<V4Block>,
        hc_head: HyperHead,
        norm: RmsNorm,
        lm_head: Linear,
        hc_mult: i32,
    }

    impl DeepSeekV4Like {
        fn new(config: MlxModelConfig, arrays: HashMap<String, Array>) -> Result<Self> {
            let layers = (0..config.num_hidden_layers)
                .map(|idx| V4Block::load(idx, &arrays, &config))
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                embed_tokens: Embedding::load("model.embed_tokens", &arrays, &config)?,
                layers,
                hc_head: HyperHead::load("model.hc_head", &arrays, &config)?,
                norm: RmsNorm::load("model.norm.weight", &arrays, config.rms_norm_eps)?,
                lm_head: Linear::load("lm_head", &arrays, &config)?,
                hc_mult: config.hc_mult as i32,
            })
        }
    }

    impl CausalLm for DeepSeekV4Like {
        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let h = self.embed_tokens.forward(&ids)?;
            let shape = h.shape();
            let mut h = broadcast_to(
                &h.expand_dims(2)?,
                &[shape[0], shape[1], self.hc_mult, shape[2]],
            )?;
            for layer in &mut self.layers {
                h = layer.forward(h, input_ids)?;
            }
            let h = self.norm.forward(&self.hc_head.forward(&h)?)?;
            let logits = self.lm_head.forward(&h)?;
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn reset_cache(&mut self) {
            for layer in &mut self.layers {
                layer.attention.reset_cache();
            }
        }

        fn prepare_cache(&mut self, capacity: i32) {
            for layer in &mut self.layers {
                layer.attention.prepare_cache(capacity);
            }
        }
    }

    struct QwenMoe {
        gate: Linear,
        switch_mlp: SwitchMlp,
        shared_expert: Option<Mlp>,
        shared_expert_gate: Option<Linear>,
        top_k: usize,
        norm_topk_prob: bool,
        // Hy3 (hy_v3) routing: sigmoid scores, expert-bias used only for top-k selection while
        // the routed weights use the bias-free sigmoid scores, then scaled by routed_scaling_factor.
        sigmoid_routing: bool,
        expert_bias: Option<Vec<f32>>,
        routed_scaling_factor: f32,
        // Read once at load (not per forward) — env lookups per layer/token tank throughput.
        compile_moe: bool,
    }

    impl QwenMoe {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let expert_bias = match arrays.get(&format!("{prefix}.gate.e_score_correction_bias")) {
                Some(b) => {
                    let b = b.as_type::<f32>()?;
                    transforms::eval([&b])?;
                    Some(b.as_slice::<f32>().to_vec())
                }
                None => None,
            };
            let compile_moe = std::env::var_os("HI_MLX_COMPILE_MOE").is_some();
            Ok(Self {
                gate: Linear::load(&format!("{prefix}.gate"), arrays, config)?,
                switch_mlp: SwitchMlp::load(&format!("{prefix}.switch_mlp"), arrays, config)?,
                shared_expert: if arrays
                    .contains_key(&format!("{prefix}.shared_expert.gate_proj.weight"))
                {
                    Some(Mlp::load(
                        &format!("{prefix}.shared_expert"),
                        arrays,
                        config,
                    )?)
                } else {
                    None
                },
                shared_expert_gate: if arrays
                    .contains_key(&format!("{prefix}.shared_expert_gate.weight"))
                {
                    Some(Linear::load(
                        &format!("{prefix}.shared_expert_gate"),
                        arrays,
                        config,
                    )?)
                } else {
                    None
                },
                top_k: config.num_experts_per_tok.unwrap_or(1) as usize,
                norm_topk_prob: config.norm_topk_prob,
                sigmoid_routing: config.family == ModelFamily::Hy3,
                expert_bias,
                routed_scaling_factor: config.routed_scaling_factor,
                compile_moe,
            })
        }

        /// Router: scores experts, selects top-k, and returns per-token `(expert, weight)` pairs.
        /// The selection is done on the CPU after a single readback of the small [experts] score
        /// vector — cheaper here than an on-device argpartition per layer, because hi-mlx runs
        /// eagerly (uncompiled), so a standalone argpartition kernel ×80 layers costs more than the
        /// readback. The expensive expert matmuls still run batched on the GPU (see `forward`).
        fn route(&self, x: &Array) -> Result<Vec<Vec<(i32, f32)>>> {
            let logits = self.gate.forward(x)?;
            // Hy3 scores experts with sigmoid; Qwen with softmax over the router logits.
            let scores = if self.sigmoid_routing {
                sigmoid(&logits.as_type::<f32>()?)?
            } else {
                softmax_axis(&logits, -1, Some(true))?.as_type::<f32>()?
            };
            transforms::eval([&scores])?;
            let shape = scores.shape();
            let (b, l, experts) = (shape[0], shape[1], shape[2]);
            if b != 1 {
                bail!("hi-mlx Qwen MoE generation currently supports batch size 1, got {b}");
            }
            let raw_scores = scores.as_slice::<f32>();
            let experts = experts as usize;
            let mut routes = Vec::with_capacity(l as usize);
            for token in 0..l as usize {
                let start = token * experts;
                let raw = &raw_scores[start..start + experts];
                // Rank by the selection score (Hy3 adds the expert bias); the routed weights below
                // still use the bias-free score.
                let mut ranked = (0..experts)
                    .map(|i| {
                        let sel = match &self.expert_bias {
                            Some(bias) => raw[i] + bias[i],
                            None => raw[i],
                        };
                        (i, sel)
                    })
                    .collect::<Vec<_>>();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                ranked.truncate(self.top_k.min(experts));
                let mut selected = ranked
                    .into_iter()
                    .map(|(idx, _)| (idx as i32, raw[idx]))
                    .collect::<Vec<_>>();
                if self.norm_topk_prob && selected.len() > 1 {
                    let denom = selected.iter().map(|(_, score)| *score).sum::<f32>();
                    if denom > f32::EPSILON {
                        for (_, score) in &mut selected {
                            *score /= denom;
                        }
                    }
                }
                if self.sigmoid_routing && self.routed_scaling_factor != 1.0 {
                    for (_, score) in &mut selected {
                        *score *= self.routed_scaling_factor;
                    }
                }
                routes.push(selected);
            }
            Ok(routes)
        }

        /// Eager fallback: CPU route + batched gather-matmul experts (used when the layer isn't the
        /// fully-quantized Hy3 shape the compiled path expects).
        fn forward_cpu(&self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l, d) = (shape[0], shape[1], shape[2]);
            if b != 1 {
                bail!("hi-mlx Qwen MoE generation currently supports batch size 1, got {b}");
            }
            let routes = self.route(x)?;
            let top_k = self.top_k as i32;
            // Batched gather-qmm needs quantized expert weights; fall back to the per-expert loop
            // for dense (unquantized) experts.
            let mut y = if self.switch_mlp.gate_proj.scales.is_some() {
                let mut idx_v = Vec::with_capacity(l as usize * self.top_k);
                let mut wts_v = Vec::with_capacity(l as usize * self.top_k);
                for token in &routes {
                    for (expert, weight) in token {
                        idx_v.push(*expert as u32);
                        wts_v.push(*weight);
                    }
                }
                let inds = Array::from_slice(&idx_v, &[l, top_k]);
                let weights = Array::from_slice(&wts_v, &[l, top_k, 1]);
                let xe = x.reshape(&[l, 1, 1, d])?;
                let expert_out = self
                    .switch_mlp
                    .forward_batched(&xe, &inds)?
                    .reshape(&[l, top_k, d])?
                    .as_type::<f32>()?;
                sum_axis(&(expert_out * weights), 1, Some(false))?.reshape(&[1, l, d])?
            } else {
                let mut outputs = Vec::with_capacity(l as usize);
                for token_idx in 0..l {
                    let token = x.index((0, token_idx, ..)).reshape(&[1, 1, d])?;
                    let mut acc = Array::zeros::<f32>(&[1, 1, d])?;
                    for (expert, score) in &routes[token_idx as usize] {
                        acc = acc + self.switch_mlp.forward_expert(&token, *expert)? * *score;
                    }
                    outputs.push(acc);
                }
                concatenate_axis(&outputs, 1)?
            };
            if let Some(shared) = &self.shared_expert {
                let shared_out = shared.forward(x)?.as_type::<f32>()?;
                y = match &self.shared_expert_gate {
                    Some(gate) => y + (sigmoid(&gate.forward(x)?)?.as_type::<f32>()? * shared_out),
                    None => y + shared_out,
                };
            }
            Ok(y)
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            if x.shape()[0] != 1 {
                bail!(
                    "hi-mlx Qwen MoE generation currently supports batch size 1, got {}",
                    x.shape()[0]
                );
            }
            // The compiled MoE (below) is numerically correct and proves MLX can fuse the router +
            // gather-qmm experts, but mlx_rs's `compile` re-traces on every call in this structure
            // (its TypeId cache doesn't hit when each layer passes different weight arrays), which is
            // slower than the eager batched path. Until the compiled closure is cached at load, the
            // batched path is the fast default; opt into the compiled path with HI_MLX_COMPILE_MOE=1.
            if !self.compile_moe {
                return self.forward_cpu(x);
            }
            // Only the fully-quantized Hy3 MoE shape (dense gate, quantized experts + always-on
            // quantized shared expert, expert bias, sigmoid routing) takes the compiled path.
            let compiled_ready = matches!(&self.gate, Linear::Dense { .. })
                && self.switch_mlp.gate_proj.scales.is_some()
                && self.expert_bias.is_some()
                && self.shared_expert.is_some()
                && self.shared_expert_gate.is_none()
                && self.sigmoid_routing;
            if !compiled_ready {
                return self.forward_cpu(x);
            }
            let Linear::Dense { weight: gate_w, .. } = &self.gate else {
                unreachable!()
            };
            let shared = self.shared_expert.as_ref().unwrap();
            let sl = |l: &SwitchLinear| -> (Array, Array, Array) {
                (
                    l.weight.clone(),
                    l.scales.clone().expect("quantized switch expert"),
                    l.biases.clone().expect("affine switch expert biases"),
                )
            };
            let ql = |l: &Linear| -> (Array, Array, Array) {
                match l {
                    Linear::Quantized {
                        weight, scales, biases, ..
                    } => (
                        weight.clone(),
                        scales.clone(),
                        biases.clone().expect("affine shared-expert biases"),
                    ),
                    _ => panic!("shared expert must be quantized"),
                }
            };
            let sw = &self.switch_mlp;
            let (sgw, sgs, sgb) = sl(&sw.gate_proj);
            let (suw, sus, sub) = sl(&sw.up_proj);
            let (sdw, sds, sdb) = sl(&sw.down_proj);
            let (hgw, hgs, hgb) = ql(&shared.gate_proj);
            let (huw, hus, hub) = ql(&shared.up_proj);
            let (hdw, hds, hdb) = ql(&shared.down_proj);
            let bias_vec = self.expert_bias.as_ref().unwrap();
            let expert_bias = Array::from_slice(bias_vec, &[bias_vec.len() as i32]);
            let inputs = vec![
                x.clone(),
                gate_w.clone(),
                expert_bias,
                sgw, sgs, sgb, suw, sus, sub, sdw, sds, sdb,
                hgw, hgs, hgb, huw, hus, hub, hdw, hds, hdb,
            ];
            let top_k = self.top_k as i32;
            let group_size = sw.gate_proj.group_size;
            let bits = sw.gate_proj.bits;
            let norm = self.norm_topk_prob;
            let scaling = self.routed_scaling_factor;
            // Reuse the cached compiled MoE (compiled once, kept alive), then materialize to chunk
            // the per-token graph the way the eager router's score readback used to.
            let y = run_moe_compiled(inputs.as_slice(), top_k, group_size, bits, norm, scaling)?;
            transforms::eval([&y])?;
            Ok(y)
        }
    }

    enum QwenFfn {
        Dense(Mlp),
        Moe(QwenMoe),
    }

    impl QwenFfn {
        fn load(
            layer_idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let prefix = format!("model.layers.{layer_idx}.mlp");
            if config.is_qwen_moe_layer(layer_idx) {
                Ok(Self::Moe(QwenMoe::load(&prefix, arrays, config)?))
            } else {
                Ok(Self::Dense(Mlp::load(&prefix, arrays, config)?))
            }
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            match self {
                Self::Dense(mlp) => mlp.forward(x),
                Self::Moe(moe) => moe.forward(x),
            }
        }
    }

    struct QwenBlock {
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
        attention: QwenAttention,
        ffn: QwenFfn,
    }

    impl QwenBlock {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let prefix = format!("model.layers.{idx}");
            Ok(Self {
                input_layernorm: RmsNorm::load(
                    &format!("{prefix}.input_layernorm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                post_attention_layernorm: RmsNorm::load(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                attention: QwenAttention::load(&format!("{prefix}.self_attn"), arrays, config)?,
                ffn: QwenFfn::load(idx, arrays, config)?,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let r = self.attention.forward(&self.input_layernorm.forward(&x)?)?;
            let h = x + r;
            let r = self
                .ffn
                .forward(&self.post_attention_layernorm.forward(&h)?)?;
            Ok(h + r)
        }
    }

    struct QwenLike {
        embed_tokens: Embedding,
        layers: Vec<QwenBlock>,
        norm: RmsNorm,
        lm_head: Option<Linear>,
    }

    impl QwenLike {
        fn new(config: MlxModelConfig, mut arrays: HashMap<String, Array>) -> Result<Self> {
            remap_hy3_moe_weights(&config, &mut arrays)?;
            prepare_qwen_moe_weights(&config, &mut arrays)?;
            let layers = (0..config.num_hidden_layers)
                .map(|idx| QwenBlock::load(idx, &arrays, &config))
                .collect::<Result<Vec<_>>>()?;
            let lm_head = if config.tie_word_embeddings {
                None
            } else {
                Some(Linear::load("lm_head", &arrays, &config)?)
            };
            Ok(Self {
                embed_tokens: Embedding::load("model.embed_tokens", &arrays, &config)?,
                norm: RmsNorm::load("model.norm.weight", &arrays, config.rms_norm_eps)?,
                layers,
                lm_head,
            })
        }
    }

    impl CausalLm for QwenLike {
        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let mut h = self.embed_tokens.forward(&ids)?;
            for layer in &mut self.layers {
                h = layer.forward(h)?;
            }
            h = self.norm.forward(&h)?;
            let logits = match &self.lm_head {
                Some(head) => head.forward(&h)?,
                None => self.embed_tokens.as_linear(&h)?,
            };
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn reset_cache(&mut self) {
            for layer in &mut self.layers {
                layer.attention.cache.reset();
            }
        }

        fn prepare_cache(&mut self, capacity: i32) {
            for layer in &mut self.layers {
                layer.attention.cache.prepare_capacity(capacity);
            }
        }

        fn rollback_cache(&mut self, to_offset: i32) {
            for layer in &mut self.layers {
                layer.attention.cache.rollback(to_offset);
            }
        }

        fn supports_rollback(&self) -> bool {
            true
        }
    }

    // Hy3 (hy_v3) stores its MoE router/shared-expert weights under different names than the
    // Qwen MoE loader expects. Rename them in place so the shared QwenFfn MoE path can load them.
    // The routed experts (`switch_mlp.*`) already match and are left untouched.
    fn remap_hy3_moe_weights(
        config: &MlxModelConfig,
        arrays: &mut HashMap<String, Array>,
    ) -> Result<()> {
        if config.family != ModelFamily::Hy3 {
            return Ok(());
        }
        for layer in 0..config.num_hidden_layers {
            let p = format!("model.layers.{layer}.mlp");
            let gp = format!("{p}.router.gate");
            // The router gate is stored quantized (often at a different bit width than the rest of
            // the model, e.g. 8-bit vs 4-bit). QwenFfn's gate does a plain dense matmul, so
            // dequantize it to a dense bf16 weight using the gate's own per-tensor quant spec.
            if let Some(weight) = arrays.remove(&format!("{gp}.weight")) {
                let scales = arrays.remove(&format!("{gp}.scales"));
                let biases = arrays.remove(&format!("{gp}.biases"));
                let dense = match (scales, config.quantization.standard_mlx_for(&gp)?) {
                    (Some(scales), Some((bits, group_size))) => dequantize_mode(
                        &weight,
                        &scales,
                        biases.as_ref(),
                        group_size as i32,
                        bits as i32,
                        "affine",
                    )?,
                    _ => weight,
                };
                transforms::eval([&dense])?;
                arrays.insert(format!("{p}.gate.weight"), dense);
            }
            if let Some(v) = arrays.remove(&format!("{p}.router.expert_bias")) {
                arrays.insert(format!("{p}.gate.e_score_correction_bias"), v);
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    if let Some(v) = arrays.remove(&format!("{p}.shared_mlp.{proj}.{suffix}")) {
                        arrays.insert(format!("{p}.shared_expert.{proj}.{suffix}"), v);
                    }
                }
            }
        }
        Ok(())
    }

    // ---------------------- Qwen3.5 (qwen3_5) gated-delta-net hybrid ----------------------
    // Hybrid: full-attention layers every `full_attention_interval` interleaved with gated-delta-net
    // (Mamba-style SSM) layers. Ported from mlx_lm's qwen3_5. The SSM runs in f32 for stability and
    // keeps its own conv + recurrent state (no KV cache).
    fn raw_array(arrays: &HashMap<String, Array>, key: &str) -> Result<Array> {
        arrays
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow!("hi-mlx Qwen3.5: missing tensor {key}"))
    }

    struct Qwen35Attention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        q_norm: Option<RmsNorm>,
        k_norm: Option<RmsNorm>,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rot_dims: i32,
        rope_theta: f32,
        cache: Cache,
    }

    impl Qwen35Attention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            let prf = config.partial_rotary_factor.unwrap_or(1.0);
            // Qwen3.5's checkpoint head counts don't match config (head_dim ≠ hidden/heads); derive
            // them from the projection output dims.
            let q_out = raw_array(arrays, &format!("{prefix}.q_proj.weight"))?.shape()[0];
            let k_out = raw_array(arrays, &format!("{prefix}.k_proj.weight"))?.shape()[0];
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: Linear::load(&format!("{prefix}.v_proj"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                q_norm: RmsNorm::load(&format!("{prefix}.q_norm.weight"), arrays, config.rms_norm_eps)
                    .ok(),
                k_norm: RmsNorm::load(&format!("{prefix}.k_norm.weight"), arrays, config.rms_norm_eps)
                    .ok(),
                // Gated attention: q_proj packs [queries; gate] → 2× the query width.
                n_heads: q_out / (2 * head_dim),
                n_kv_heads: k_out / head_dim,
                head_dim,
                rot_dims: ((head_dim as f32) * prf) as i32,
                rope_theta: config.rope_theta,
                cache: Cache::new(),
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l) = (shape[0], shape[1]);
            // Gated attention: q_proj → [queries | gate], each n_heads × head_dim.
            let qg = self
                .q_proj
                .forward(x)?
                .reshape(&[b, l, self.n_heads, 2 * self.head_dim])?;
            let mut qparts = split_sections(&qg, &[self.head_dim], -1)?;
            let gate = qparts.remove(1); // [b,l,n_heads,head_dim]
            let mut q = qparts.remove(0);
            let mut k = self
                .k_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?;
            if let Some(n) = &self.q_norm {
                q = n.forward(&q)?;
            }
            if let Some(n) = &self.k_norm {
                k = n.forward(&k)?;
            }
            q = q.transpose_axes(&[0, 2, 1, 3])?;
            k = k.transpose_axes(&[0, 2, 1, 3])?;
            let v = self
                .v_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let offset = self.cache.offset;
            q = rope(q, self.rot_dims, false, self.rope_theta, 1.0, offset, None)?;
            k = rope(k, self.rot_dims, false, self.rope_theta, 1.0, offset, None)?;
            let (k, v) = self.cache.update(k, v)?;
            let scale = (self.head_dim as f32).powf(-0.5);
            let output = if l > 1 && offset == 0 {
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    ScaledDotProductAttentionMask::Causal,
                    None::<&Array>,
                )?
            } else if l > 1 {
                let mask = causal_attention_mask(l, k.shape()[2], offset);
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    ScaledDotProductAttentionMask::Array(&mask),
                    None::<&Array>,
                )?
            } else {
                scaled_dot_product_attention(&q, &k, &v, scale, None, None::<&Array>)?
            };
            let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                l,
                self.n_heads * self.head_dim,
            ])?;
            // Output gate: out * sigmoid(gate).
            let gate = gate.reshape(&[b, l, self.n_heads * self.head_dim])?;
            let output = output * sigmoid(&gate)?;
            self.o_proj.forward(&output)
        }
    }

    struct GatedDeltaNet {
        in_proj_qkv: Linear,
        in_proj_z: Linear,
        in_proj_b: Linear,
        in_proj_a: Linear,
        conv1d_weight: Array,
        a_log: Array,
        dt_bias: Array,
        norm_weight: Array,
        qk_ones: Array,
        out_proj: Linear,
        num_v_heads: i32,
        num_k_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        key_dim: i32,
        value_dim: i32,
        conv_dim: i32,
        conv_kernel: i32,
        eps: f32,
        conv_state: Option<Array>,
        ssm_state: Option<Array>,
    }

    impl GatedDeltaNet {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let num_v_heads = config.linear_num_value_heads.unwrap_or(0) as i32;
            let num_k_heads = config.linear_num_key_heads.unwrap_or(0) as i32;
            let head_k_dim = config.linear_key_head_dim.unwrap_or(0) as i32;
            let head_v_dim = config.linear_value_head_dim.unwrap_or(0) as i32;
            let conv_kernel = config.linear_conv_kernel_dim.unwrap_or(4) as i32;
            let key_dim = num_k_heads * head_k_dim;
            let value_dim = num_v_heads * head_v_dim;
            let conv_dim = key_dim * 2 + value_dim;
            Ok(Self {
                in_proj_qkv: Linear::load(&format!("{prefix}.in_proj_qkv"), arrays, config)?,
                in_proj_z: Linear::load(&format!("{prefix}.in_proj_z"), arrays, config)?,
                in_proj_b: Linear::load(&format!("{prefix}.in_proj_b"), arrays, config)?,
                in_proj_a: Linear::load(&format!("{prefix}.in_proj_a"), arrays, config)?,
                conv1d_weight: raw_array(arrays, &format!("{prefix}.conv1d.weight"))?
                    .as_type::<f32>()?,
                a_log: raw_array(arrays, &format!("{prefix}.A_log"))?.as_type::<f32>()?,
                dt_bias: raw_array(arrays, &format!("{prefix}.dt_bias"))?.as_type::<f32>()?,
                norm_weight: raw_array(arrays, &format!("{prefix}.norm.weight"))?.as_type::<f32>()?,
                qk_ones: Array::ones::<f32>(&[head_k_dim])?,
                out_proj: Linear::load(&format!("{prefix}.out_proj"), arrays, config)?,
                num_v_heads,
                num_k_heads,
                head_k_dim,
                head_v_dim,
                key_dim,
                value_dim,
                conv_dim,
                conv_kernel,
                eps: config.rms_norm_eps,
                conv_state: None,
                ssm_state: None,
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let dtype = x.dtype();
            let shape = x.shape();
            let s = shape[1];
            let (hv, hk, dv) = (self.num_v_heads, self.head_k_dim, self.head_v_dim);
            let qkv = self.in_proj_qkv.forward(x)?.as_type::<f32>()?;
            let z = self
                .in_proj_z
                .forward(x)?
                .as_type::<f32>()?
                .reshape(&[1, s, hv, dv])?;
            let bb = self.in_proj_b.forward(x)?.as_type::<f32>()?;
            let aa = self.in_proj_a.forward(x)?.as_type::<f32>()?;

            // Causal depthwise conv1d over [conv_state | qkv]; carry the last kernel-1 frames.
            let keep = self.conv_kernel - 1;
            let conv_state = match self.conv_state.take() {
                Some(st) => st,
                None => Array::zeros::<f32>(&[1, keep, self.conv_dim])?,
            };
            let conv_in = concatenate_axis(&[&conv_state, &qkv], 1)?;
            let clen = conv_in.shape()[1];
            self.conv_state = Some(conv_in.index((.., (clen - keep)..clen, ..)));
            let conv_out = conv1d(&conv_in, &self.conv1d_weight, 1, 0, 1, self.conv_dim)?;
            let conv_out = silu(&conv_out)?;

            let mut parts = split_sections(&conv_out, &[self.key_dim, 2 * self.key_dim], -1)?;
            let v = parts.remove(2).reshape(&[1, s, hv, dv])?;
            let k = parts.remove(1).reshape(&[1, s, self.num_k_heads, hk])?;
            let q = parts.remove(0).reshape(&[1, s, self.num_k_heads, hk])?;

            // Weightless RMSNorm over head dim, with the mlx_lm scaling.
            let inv = (hk as f32).powf(-0.5);
            let q = rms_norm(&q, &self.qk_ones, 1e-6)? * (inv * inv);
            let k = rms_norm(&k, &self.qk_ones, 1e-6)? * inv;
            // GQA: repeat q,k heads up to num_v_heads.
            let rep = self.num_v_heads / self.num_k_heads;
            let q = broadcast_to(
                &q.reshape(&[1, s, self.num_k_heads, 1, hk])?,
                &[1, s, self.num_k_heads, rep, hk],
            )?
            .reshape(&[1, s, hv, hk])?;
            let k = broadcast_to(
                &k.reshape(&[1, s, self.num_k_heads, 1, hk])?,
                &[1, s, self.num_k_heads, rep, hk],
            )?
            .reshape(&[1, s, hv, hk])?;

            let beta = sigmoid(&bb)?;
            // g = exp(-exp(A_log) * softplus(a + dt_bias))
            let neg_a = exp(&self.a_log)? * -1.0;
            let g = exp(&(neg_a * softplus(&(aa + &self.dt_bias))?))?;

            // Decode (single token) uses the cheap recurrent step; prefill uses the chunk-parallel
            // scan (far fewer sequential ops). Both update self.ssm_state identically.
            let out = if s > 1 {
                self.scan_chunked(&q, &k, &v, &g, &beta, s)?
            } else {
                self.scan_recurrent(&q, &k, &v, &g, &beta, s)?
            };
            // Gated RMSNorm (Qwen3-Next style): norm the SSM output first, THEN gate by silu(z).
            let normed = rms_norm(&out, &self.norm_weight, self.eps)?;
            let gated = silu(&z)? * normed;
            let out = gated.reshape(&[1, s, self.value_dim])?.as_dtype(dtype)?;
            self.out_proj.forward(&out)
        }

        // Per-token recurrent step (used for decode, S==1). q,k: [1,S,Hv,Dk]; v: [1,S,Hv,Dv];
        // g,beta: [1,S,Hv]. Updates self.ssm_state; returns y [1,S,Hv,Dv].
        fn scan_recurrent(
            &mut self,
            q: &Array,
            k: &Array,
            v: &Array,
            g: &Array,
            beta: &Array,
            s: i32,
        ) -> Result<Array> {
            let (hv, hk, dv) = (self.num_v_heads, self.head_k_dim, self.head_v_dim);
            let mut state = match self.ssm_state.take() {
                Some(st) => st,
                None => Array::zeros::<f32>(&[1, hv, dv, hk])?,
            };
            // Fast path for decode (single token): the inputs are already one step, so skip the
            // per-token slicing / Vec / concatenate — fewer graph nodes per layer per token.
            if s == 1 {
                let qt = q.reshape(&[1, hv, 1, hk])?;
                let kt = k.reshape(&[1, hv, 1, hk])?;
                let vt = v.reshape(&[1, hv, dv])?;
                let gt = g.reshape(&[1, hv, 1, 1])?;
                let betat = beta.reshape(&[1, hv, 1])?;
                state = state * gt;
                let kv_mem = sum_axis(&(state.clone() * &kt), -1, false)?;
                let delta = (vt - kv_mem) * betat;
                state = state + (kt * delta.reshape(&[1, hv, dv, 1])?);
                let yt = sum_axis(&(state.clone() * qt), -1, false)?;
                self.ssm_state = Some(state);
                return Ok(yt.reshape(&[1, 1, hv, dv])?);
            }
            let mut ys: Vec<Array> = Vec::with_capacity(s as usize);
            for t in 0..s {
                let qt = q.index((.., t..(t + 1), .., ..)).reshape(&[1, hv, 1, hk])?;
                let kt = k.index((.., t..(t + 1), .., ..)).reshape(&[1, hv, 1, hk])?;
                let vt = v.index((.., t..(t + 1), .., ..)).reshape(&[1, hv, dv])?;
                let gt = g.index((.., t..(t + 1), ..)).reshape(&[1, hv, 1, 1])?;
                let betat = beta.index((.., t..(t + 1), ..)).reshape(&[1, hv, 1])?;
                state = state * gt;
                let kv_mem = sum_axis(&(state.clone() * &kt), -1, false)?;
                let delta = (vt - kv_mem) * betat;
                let delta_e = delta.reshape(&[1, hv, dv, 1])?;
                state = state + (kt.clone() * delta_e);
                let yt = sum_axis(&(state.clone() * qt), -1, false)?;
                ys.push(yt.reshape(&[1, 1, hv, dv])?);
            }
            self.ssm_state = Some(state);
            if ys.len() == 1 {
                Ok(ys.remove(0))
            } else {
                Ok(concatenate_axis(&ys.iter().collect::<Vec<_>>(), 1)?)
            }
        }

        // Chunk-parallel gated delta-rule scan (prefill). Precomputes the intra-chunk WY/UT quantities
        // batched over all chunks (with a Newton-Schulz unit-lower-triangular inverse), then a short
        // sequential scan over chunks. Mathematically identical to scan_recurrent (verified for C=1).
        fn scan_chunked(
            &mut self,
            q: &Array,
            k: &Array,
            v: &Array,
            g: &Array,
            beta: &Array,
            s: i32,
        ) -> Result<Array> {
            let (hv, hk, dv) = (self.num_v_heads, self.head_k_dim, self.head_v_dim);
            let cs: i32 = 64;
            let nc = (s + cs - 1) / cs;
            let sp = nc * cs;
            let pad = sp - s;
            // Pad the sequence to a multiple of the chunk size (g padded with 1 → no decay; beta with
            // 0 → padded steps contribute nothing; outputs sliced off at the end).
            let (q, k, v, g, beta) = if pad > 0 {
                let zq = Array::zeros::<f32>(&[1, pad, hv, hk])?;
                let zv = Array::zeros::<f32>(&[1, pad, hv, dv])?;
                let zb = Array::zeros::<f32>(&[1, pad, hv])?;
                let og = Array::ones::<f32>(&[1, pad, hv])?;
                (
                    concatenate_axis(&[q, &zq], 1)?,
                    concatenate_axis(&[k, &zq], 1)?,
                    concatenate_axis(&[v, &zv], 1)?,
                    concatenate_axis(&[g, &og], 1)?,
                    concatenate_axis(&[beta, &zb], 1)?,
                )
            } else {
                (q.clone(), k.clone(), v.clone(), g.clone(), beta.clone())
            };
            // [1,sp,Hv,D] -> [nc,Hv,cs,D]
            let q = q.reshape(&[nc, cs, hv, hk])?.transpose_axes(&[0, 2, 1, 3])?;
            let k = k.reshape(&[nc, cs, hv, hk])?.transpose_axes(&[0, 2, 1, 3])?;
            let v = v.reshape(&[nc, cs, hv, dv])?.transpose_axes(&[0, 2, 1, 3])?;
            let g = g.reshape(&[nc, cs, hv])?.transpose_axes(&[0, 2, 1])?;
            let beta = beta.reshape(&[nc, cs, hv])?.transpose_axes(&[0, 2, 1])?;

            let ltri = tril(Array::ones::<f32>(&[cs, cs])?, 0)?; // lower incl diag (for cumsum)
            let eye = identity::<f32>(cs)?;
            // Additive masks: 0 on the kept triangle, -1e9 elsewhere. Added to the (finite) log-decay
            // differences *before* exp, so masked-out entries become exp(-1e9)=0 with no inf·0 = NaN.
            let (mut pen_incl, mut pen_strict) =
                (vec![0f32; (cs * cs) as usize], vec![0f32; (cs * cs) as usize]);
            for t in 0..cs {
                for j in 0..cs {
                    let idx = (t * cs + j) as usize;
                    if t < j {
                        pen_incl[idx] = -1e9;
                    }
                    if t <= j {
                        pen_strict[idx] = -1e9;
                    }
                }
            }
            let pen_incl = Array::from_slice(&pen_incl, &[cs, cs]);
            let pen_strict = Array::from_slice(&pen_strict, &[cs, cs]);
            // Cumulative within-chunk log-decay lg_t = sum_{i<=t} log g_i. g can underflow to exactly
            // 0 when the per-step decay is extreme (e.g. Qwen3.5-MoE: neg_a*softplus ~ -1000), and
            // log(0) = -inf then makes the lg_t - lg_j differences below -inf-(-inf) = NaN. Clamp to a
            // tiny floor: where g underflows the decay is already complete, so exp(-69) ~ 0 is exact.
            let logg = maximum(&g, &Array::from_f32(1e-30))?
                .log()?
                .reshape(&[nc, hv, cs, 1])?;
            let lg = matmul(&ltri, &logg)?.reshape(&[nc, hv, cs])?;
            let gamma_e = exp(&lg)?.reshape(&[nc, hv, cs, 1])?; // gamma_t in [0,1]
            let lg_last = lg.index((.., .., (cs - 1)..cs)).reshape(&[nc, hv, 1])?;
            let gamma_last = exp(&lg_last)?.reshape(&[nc, hv, 1, 1])?;

            let kbar = k.clone() * gamma_e.clone(); // gamma_t k_t  (bounded, gamma<=1)
            let qbar = q.clone() * gamma_e.clone();
            let beta_e = beta.reshape(&[nc, hv, cs, 1])?;

            // Decay-ratio matrices D[t,j] = exp(lg_t - lg_j), masked (no k/gamma division).
            let diff = lg.reshape(&[nc, hv, cs, 1])? - lg.reshape(&[nc, hv, 1, cs])?;
            let d_incl = exp(&(diff.clone() + pen_incl))?; // lower incl diag, in (0,1]
            let d_strict = exp(&(diff + pen_strict))?; // strictly lower
            // A[t,j] = beta_t (k_t.k_j)(gamma_t/gamma_j), strictly lower-triangular.
            let kk = matmul(&k, &k.swap_axes(-1, -2)?)?;
            let a = beta_e.clone() * (kk * d_strict);
            // (I + A)^{-1} via Newton-Schulz (A strictly-lower nilpotent → exact in ceil(log2 cs) iters).
            let mmat = eye.clone() + a;
            let two_eye = eye.clone() * 2.0;
            let mut tinv = broadcast_to(&eye, &[nc, hv, cs, cs])?;
            let iters = (cs as f32).log2().ceil() as i32;
            for _ in 0..iters {
                let r = two_eye.clone() - matmul(&mmat, &tinv)?;
                tinv = matmul(&tinv, &r)?;
            }
            let w_all = matmul(&tinv, &(beta_e.clone() * v.clone()))?; // [nc,hv,cs,dv]
            let p_all = matmul(&tinv, &(beta_e.clone() * kbar.clone()))?; // [nc,hv,cs,hk]
            // intra attention (q_t.k_j)(gamma_t/gamma_j), lower incl diag.
            let qk_all = matmul(&q, &k.swap_axes(-1, -2)?)? * d_incl;
            // Kfinal_j = (gamma_C/gamma_j) k_j = k_j * exp(lg_last - lg_j).
            let d_last = exp(&(lg_last.clone() - lg.clone()))?.reshape(&[nc, hv, cs, 1])?;
            let kfinal_all = k.clone() * d_last;

            let mut state = match self.ssm_state.take() {
                Some(st) => st.reshape(&[hv, dv, hk])?,
                None => Array::zeros::<f32>(&[hv, dv, hk])?,
            };
            let mut ys: Vec<Array> = Vec::with_capacity(nc as usize);
            for c in 0..nc {
                let w_c = w_all.index((c..(c + 1), .., .., ..)).reshape(&[hv, cs, dv])?;
                let p_c = p_all.index((c..(c + 1), .., .., ..)).reshape(&[hv, cs, hk])?;
                let qk_c = qk_all.index((c..(c + 1), .., .., ..)).reshape(&[hv, cs, cs])?;
                let qbar_c = qbar.index((c..(c + 1), .., .., ..)).reshape(&[hv, cs, hk])?;
                let kfinal_c = kfinal_all
                    .index((c..(c + 1), .., .., ..))
                    .reshape(&[hv, cs, hk])?;
                let gl_c = gamma_last.index((c..(c + 1), .., .., ..)).reshape(&[hv, 1, 1])?;
                let state_t = state.swap_axes(-1, -2)?; // [hv,hk,dv]
                let u_c = w_c - matmul(&p_c, &state_t)?; // [hv,cs,dv]
                let y_c = matmul(&qbar_c, &state_t)? + matmul(&qk_c, &u_c)?;
                state = (gl_c * state.clone()) + matmul(&u_c.swap_axes(-1, -2)?, &kfinal_c)?;
                ys.push(y_c.swap_axes(0, 1)?.reshape(&[1, cs, hv, dv])?);
            }
            self.ssm_state = Some(state.reshape(&[1, hv, dv, hk])?);
            let out = concatenate_axis(&ys.iter().collect::<Vec<_>>(), 1)?; // [1,sp,hv,dv]
            Ok(out.index((.., 0..s, .., ..))) // unpad
        }

        fn reset(&mut self) {
            self.conv_state = None;
            self.ssm_state = None;
        }
    }

    enum Qwen35Mixer {
        Attn(Qwen35Attention),
        Linear(Box<GatedDeltaNet>),
    }

    struct Qwen35Layer {
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
        mixer: Qwen35Mixer,
        // Dense (qwen3_5) or shared-expert MoE (qwen3_5_moe) FFN, chosen per layer by QwenFfn::load.
        ffn: QwenFfn,
    }

    impl Qwen35Layer {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let p = format!("model.layers.{idx}");
            let interval = config.full_attention_interval.unwrap_or(4);
            let is_linear = (idx + 1) % interval != 0;
            let mixer = if is_linear {
                Qwen35Mixer::Linear(Box::new(GatedDeltaNet::load(
                    &format!("{p}.linear_attn"),
                    arrays,
                    config,
                )?))
            } else {
                Qwen35Mixer::Attn(Qwen35Attention::load(
                    &format!("{p}.self_attn"),
                    arrays,
                    config,
                )?)
            };
            Ok(Self {
                input_layernorm: RmsNorm::load(
                    &format!("{p}.input_layernorm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                post_attention_layernorm: RmsNorm::load(
                    &format!("{p}.post_attention_layernorm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                mixer,
                ffn: QwenFfn::load(idx, arrays, config)?,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let h = self.input_layernorm.forward(&x)?;
            let h = match &mut self.mixer {
                Qwen35Mixer::Attn(a) => a.forward(&h)?,
                Qwen35Mixer::Linear(l) => l.forward(&h)?,
            };
            let x = x + h;
            let h = self.post_attention_layernorm.forward(&x)?;
            let h = self.ffn.forward(&h)?;
            Ok(x + h)
        }
    }

    struct Qwen35Like {
        embed_tokens: Embedding,
        layers: Vec<Qwen35Layer>,
        norm: RmsNorm,
        lm_head: Option<Linear>,
    }

    impl Qwen35Like {
        fn new(config: MlxModelConfig, arrays: HashMap<String, Array>) -> Result<Self> {
            let layers = (0..config.num_hidden_layers)
                .map(|idx| Qwen35Layer::load(idx, &arrays, &config))
                .collect::<Result<Vec<_>>>()?;
            let lm_head = if config.tie_word_embeddings {
                None
            } else {
                Some(Linear::load("lm_head", &arrays, &config)?)
            };
            Ok(Self {
                embed_tokens: Embedding::load("model.embed_tokens", &arrays, &config)?,
                norm: RmsNorm::load("model.norm.weight", &arrays, config.rms_norm_eps)?,
                layers,
                lm_head,
            })
        }
    }

    impl CausalLm for Qwen35Like {
        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let mut h = self.embed_tokens.forward(&ids)?;
            for layer in &mut self.layers {
                h = layer.forward(h)?;
            }
            h = self.norm.forward(&h)?;
            let logits = match &self.lm_head {
                Some(head) => head.forward(&h)?,
                None => self.embed_tokens.as_linear(&h)?,
            };
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn reset_cache(&mut self) {
            for layer in &mut self.layers {
                match &mut layer.mixer {
                    Qwen35Mixer::Attn(a) => a.cache.reset(),
                    Qwen35Mixer::Linear(l) => l.reset(),
                }
            }
        }

        fn prepare_cache(&mut self, capacity: i32) {
            for layer in &mut self.layers {
                if let Qwen35Mixer::Attn(a) = &mut layer.mixer {
                    a.cache.prepare_capacity(capacity);
                }
            }
        }
    }

    // ---------------------- GLM-4 (glm4, GQA) ----------------------
    // Standard GQA GLM-4 (e.g. GLM-4-9B-0414): partial rotary, a fused `gate_up_proj` MLP, sandwich
    // norms (extra post_self_attn + post_mlp layernorms), and QKV biases. Distinct from the
    // MLA-based GLM-*-Flash variants, which stay on the MlaLike path.
    struct Glm4Attention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rot_dims: i32,
        rope_theta: f32,
        cache: Cache,
    }

    impl Glm4Attention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            let prf = config.partial_rotary_factor.unwrap_or(1.0);
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: Linear::load(&format!("{prefix}.v_proj"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                n_heads: config.num_attention_heads as i32,
                n_kv_heads: config.num_key_value_heads as i32,
                head_dim,
                rot_dims: ((head_dim as f32) * prf) as i32,
                rope_theta: config.rope_theta,
                cache: Cache::new(),
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l) = (shape[0], shape[1]);
            let mut q = self
                .q_proj
                .forward(x)?
                .reshape(&[b, l, self.n_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let mut k = self
                .k_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let v = self
                .v_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let offset = self.cache.offset;
            // Partial rotary: only the first `rot_dims` of each head are rotated.
            q = rope(q, self.rot_dims, false, self.rope_theta, 1.0, offset, None)?;
            k = rope(k, self.rot_dims, false, self.rope_theta, 1.0, offset, None)?;
            let (k, v) = self.cache.update(k, v)?;
            let scale = (self.head_dim as f32).powf(-0.5);
            let output = if l > 1 && offset == 0 {
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    ScaledDotProductAttentionMask::Causal,
                    None::<&Array>,
                )?
            } else if l > 1 {
                let mask = causal_attention_mask(l, k.shape()[2], offset);
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    ScaledDotProductAttentionMask::Array(&mask),
                    None::<&Array>,
                )?
            } else {
                scaled_dot_product_attention(&q, &k, &v, scale, None, None::<&Array>)?
            };
            let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                l,
                self.n_heads * self.head_dim,
            ])?;
            self.o_proj.forward(&output)
        }
    }

    struct Glm4Mlp {
        gate_up_proj: Linear,
        down_proj: Linear,
        intermediate: i32,
    }

    impl Glm4Mlp {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                gate_up_proj: Linear::load(&format!("{prefix}.gate_up_proj"), arrays, config)?,
                down_proj: Linear::load(&format!("{prefix}.down_proj"), arrays, config)?,
                intermediate: config.intermediate_size.unwrap_or(0) as i32,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            // Fused gate_up: first `intermediate` cols are the gate, the rest are up.
            let gu = self.gate_up_proj.forward(x)?;
            let mut parts = split_sections(&gu, &[self.intermediate], -1)?;
            let up = parts.remove(1);
            let gate = parts.remove(0);
            let hidden = (sigmoid(&gate)? * gate) * up;
            self.down_proj.forward(&hidden)
        }
    }

    struct Glm4Block {
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
        post_self_attn_layernorm: RmsNorm,
        post_mlp_layernorm: RmsNorm,
        attention: Glm4Attention,
        mlp: Glm4Mlp,
    }

    impl Glm4Block {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let p = format!("model.layers.{idx}");
            Ok(Self {
                input_layernorm: RmsNorm::load(
                    &format!("{p}.input_layernorm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                post_attention_layernorm: RmsNorm::load(
                    &format!("{p}.post_attention_layernorm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                post_self_attn_layernorm: RmsNorm::load(
                    &format!("{p}.post_self_attn_layernorm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                post_mlp_layernorm: RmsNorm::load(
                    &format!("{p}.post_mlp_layernorm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                attention: Glm4Attention::load(&format!("{p}.self_attn"), arrays, config)?,
                mlp: Glm4Mlp::load(&format!("{p}.mlp"), arrays, config)?,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            // GLM-4 sandwich norm: post-norm the attn and mlp sublayer outputs before the residual.
            let h = self.attention.forward(&self.input_layernorm.forward(&x)?)?;
            let h = self.post_self_attn_layernorm.forward(&h)?;
            let x = x + h;
            let h = self.mlp.forward(&self.post_attention_layernorm.forward(&x)?)?;
            let h = self.post_mlp_layernorm.forward(&h)?;
            Ok(x + h)
        }
    }

    struct Glm4Like {
        embed_tokens: Embedding,
        layers: Vec<Glm4Block>,
        norm: RmsNorm,
        lm_head: Option<Linear>,
    }

    impl Glm4Like {
        fn new(config: MlxModelConfig, arrays: HashMap<String, Array>) -> Result<Self> {
            let layers = (0..config.num_hidden_layers)
                .map(|idx| Glm4Block::load(idx, &arrays, &config))
                .collect::<Result<Vec<_>>>()?;
            let lm_head = if config.tie_word_embeddings {
                None
            } else {
                Some(Linear::load("lm_head", &arrays, &config)?)
            };
            Ok(Self {
                embed_tokens: Embedding::load("model.embed_tokens", &arrays, &config)?,
                norm: RmsNorm::load("model.norm.weight", &arrays, config.rms_norm_eps)?,
                layers,
                lm_head,
            })
        }
    }

    impl CausalLm for Glm4Like {
        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let mut h = self.embed_tokens.forward(&ids)?;
            for layer in &mut self.layers {
                h = layer.forward(h)?;
            }
            h = self.norm.forward(&h)?;
            let logits = match &self.lm_head {
                Some(head) => head.forward(&h)?,
                None => self.embed_tokens.as_linear(&h)?,
            };
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn reset_cache(&mut self) {
            for layer in &mut self.layers {
                layer.attention.cache.reset();
            }
        }

        fn prepare_cache(&mut self, capacity: i32) {
            for layer in &mut self.layers {
                layer.attention.cache.prepare_capacity(capacity);
            }
        }
    }

    fn prepare_qwen_moe_weights(
        config: &MlxModelConfig,
        arrays: &mut HashMap<String, Array>,
    ) -> Result<()> {
        let Some(num_experts) = config.n_routed_experts else {
            return Ok(());
        };
        for layer in 0..config.num_hidden_layers {
            if !config.is_qwen_moe_layer(layer) {
                continue;
            }
            let prefix = format!("model.layers.{layer}.mlp");
            if arrays.contains_key(&format!("{prefix}.switch_mlp.gate_proj.weight")) {
                continue;
            }
            if !arrays.contains_key(&format!("{prefix}.experts.0.gate_proj.weight")) {
                continue;
            }
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let first = format!("{prefix}.experts.0.{projection}.{suffix}");
                    if !arrays.contains_key(&first) {
                        continue;
                    }
                    let mut parts = Vec::with_capacity(num_experts as usize);
                    for expert in 0..num_experts {
                        parts.push(take(
                            arrays,
                            &format!("{prefix}.experts.{expert}.{projection}.{suffix}"),
                        )?);
                    }
                    let stacked = stack_axis(&parts, 0)?;
                    transforms::eval([&stacked])?;
                    drop(parts);
                    arrays.insert(
                        format!("{prefix}.switch_mlp.{projection}.{suffix}"),
                        stacked,
                    );
                    for expert in 0..num_experts {
                        arrays.remove(&format!("{prefix}.experts.{expert}.{projection}.{suffix}"));
                    }
                }
            }
        }
        Ok(())
    }

    fn prepare_mla_weights(
        config: &MlxModelConfig,
        arrays: &mut HashMap<String, Array>,
    ) -> Result<()> {
        let qk_nope = config
            .qk_nope_head_dim
            .ok_or_else(|| anyhow!("config.json missing qk_nope_head_dim for MLA model"))?
            as i32;
        let v_head = config.v_head_dim.unwrap_or(qk_nope as u32) as i32;
        let heads = config.num_attention_heads as i32;
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}.self_attn");
            if arrays.contains_key(&format!("{prefix}.embed_q.weight"))
                && arrays.contains_key(&format!("{prefix}.unembed_out.weight"))
            {
                continue;
            }
            let kv_b_key = format!("{prefix}.kv_b_proj.weight");
            if !arrays.contains_key(&kv_b_key) {
                continue;
            }
            let mut weight = take(arrays, &kv_b_key)?;
            if let (Some(scales), Some(biases)) = (
                arrays.get(&format!("{prefix}.kv_b_proj.scales")),
                arrays.get(&format!("{prefix}.kv_b_proj.biases")),
            ) {
                let dims = config
                    .kv_lora_rank
                    .ok_or_else(|| anyhow!("config.json missing kv_lora_rank for MLA model"))?
                    as i32;
                let bits = (weight.shape()[weight.shape().len() - 1] * 32) / dims;
                let group_size = dims / scales.shape()[scales.shape().len() - 1];
                weight = dequantize(&weight, scales, biases, group_size, bits)?;
            }
            let head_dim = qk_nope + v_head;
            let reshaped = weight.reshape(&[heads, head_dim, -1])?;
            let embed_q = reshaped.index((.., ..qk_nope, ..)).swap_axes(-1, -2)?;
            let unembed_out = reshaped.index((.., qk_nope.., ..));
            transforms::eval([&embed_q, &unembed_out])?;
            arrays.insert(format!("{prefix}.embed_q.weight"), embed_q);
            arrays.insert(format!("{prefix}.unembed_out.weight"), unembed_out);
            for suffix in ["weight", "scales", "biases", "bias"] {
                arrays.remove(&format!("{prefix}.kv_b_proj.{suffix}"));
            }
        }
        Ok(())
    }

    fn take(arrays: &HashMap<String, Array>, key: &str) -> Result<Array> {
        arrays
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow!("missing tensor {key}"))
    }

    fn take_any(
        arrays: &HashMap<String, Array>,
        prefixes: &[String],
        suffix: &str,
    ) -> Result<Array> {
        for prefix in prefixes {
            let key = format!("{prefix}.{suffix}");
            if let Some(array) = arrays.get(&key) {
                return Ok(array.clone());
            }
        }
        let looked = prefixes
            .iter()
            .map(|prefix| format!("{prefix}.{suffix}"))
            .collect::<Vec<_>>()
            .join(", ");
        Err(anyhow!("missing tensor; looked for {looked}"))
    }

    fn score_v4(logits: &[f32], scoring_func: &str) -> Vec<f32> {
        match scoring_func {
            "softmax" => {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut scores = logits
                    .iter()
                    .map(|value| (*value - max).exp())
                    .collect::<Vec<_>>();
                let denom = scores.iter().sum::<f32>();
                if denom > f32::EPSILON {
                    for score in &mut scores {
                        *score /= denom;
                    }
                }
                scores
            }
            "sigmoid" => logits
                .iter()
                .map(|value| 1.0 / (1.0 + (-*value).exp()))
                .collect(),
            _ => logits
                .iter()
                .map(|value| value.exp().ln_1p().sqrt())
                .collect(),
        }
    }

    fn quant_spec_for(
        config: &MlxModelConfig,
        prefix: &str,
        weight: &Array,
        scales: Option<&Array>,
    ) -> Result<QuantizationSpec> {
        let mut spec = config
            .quantization
            .mlx_quantization_for(prefix)?
            .unwrap_or(QuantizationSpec {
                bits: 4,
                group_size: 64,
                mode: crate::config::QuantizationMode::Affine,
            });
        // Dynamic/mixed-bit builds (e.g. GLM-5.2's MTP layer) omit per-tensor quant entries, so the
        // config default can be wrong. Infer the real bit width from the packing:
        //   in_packed = in*bits/32, n_groups = in/group_size  =>  bits = 32*in_packed/(n_groups*gs).
        if spec.mode.as_str() == "affine" {
            if let Some(scales) = scales {
                let gs = spec.group_size as i64;
                let in_packed = *weight.shape().last().unwrap_or(&0) as i64;
                let n_groups = *scales.shape().last().unwrap_or(&0) as i64;
                if gs > 0 && n_groups > 0 {
                    let bits = 32 * in_packed / (n_groups * gs);
                    if (2..=8).contains(&bits) {
                        spec.bits = bits as u32;
                    }
                }
            }
        }
        Ok(spec)
    }

    fn require_biases_for_affine(
        prefix: &str,
        spec: &QuantizationSpec,
        biases: Option<&Array>,
    ) -> Result<()> {
        if spec.mode.as_str() == "affine" && biases.is_none() {
            bail!("missing tensor {prefix}.biases for affine quantized weight");
        }
        Ok(())
    }

    fn optional_int(value: i32) -> mlx_sys::mlx_optional_int {
        mlx_sys::mlx_optional_int {
            value,
            has_value: true,
        }
    }

    fn optional_dtype_none() -> mlx_sys::mlx_optional_dtype {
        mlx_sys::mlx_optional_dtype {
            value: mlx_sys::mlx_dtype__MLX_FLOAT32,
            has_value: false,
        }
    }

    fn empty_array() -> mlx_sys::mlx_array {
        unsafe { mlx_sys::mlx_array_new() }
    }

    fn quantized_matmul_mode(
        x: &Array,
        weight: &Array,
        scales: &Array,
        biases: Option<&Array>,
        transpose: bool,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<Array> {
        let mode = CString::new(mode)?;
        let stream = Stream::default();
        let mut out = empty_array();
        let status = unsafe {
            mlx_sys::mlx_quantized_matmul(
                &mut out as *mut _,
                x.as_ptr(),
                weight.as_ptr(),
                scales.as_ptr(),
                biases.map(Array::as_ptr).unwrap_or_else(empty_array),
                transpose,
                optional_int(group_size),
                optional_int(bits),
                mode.as_ptr(),
                stream.as_ptr(),
            )
        };
        if status != 0 {
            unsafe { mlx_sys::mlx_array_free(out) };
            bail!("MLX quantized_matmul failed for {bits}-bit {mode:?} weights");
        }
        Ok(unsafe { Array::from_ptr(out) })
    }

    /// Batched gather + quantized matmul: for each output position i, computes
    /// `x[i] @ w[rhs_indices[i]].T`. Used to run all routed experts of a MoE layer in a few
    /// batched kernels instead of one quantized_matmul per (token, expert).
    fn gather_qmm_mode(
        x: &Array,
        weight: &Array,
        scales: &Array,
        biases: Option<&Array>,
        rhs_indices: &Array,
        transpose: bool,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<Array> {
        let mode = CString::new(mode)?;
        let stream = Stream::default();
        let mut out = empty_array();
        let status = unsafe {
            mlx_sys::mlx_gather_qmm(
                &mut out as *mut _,
                x.as_ptr(),
                weight.as_ptr(),
                scales.as_ptr(),
                biases.map(Array::as_ptr).unwrap_or_else(empty_array),
                empty_array(),   // lhs_indices: null → broadcast x's batch dims
                rhs_indices.as_ptr(),
                transpose,
                optional_int(group_size),
                optional_int(bits),
                mode.as_ptr(),
                false,           // sorted_indices
                stream.as_ptr(),
            )
        };
        if status != 0 {
            unsafe { mlx_sys::mlx_array_free(out) };
            bail!("MLX gather_qmm failed for {bits}-bit {mode:?} weights");
        }
        Ok(unsafe { Array::from_ptr(out) })
    }

    /// The full Hy3-style MoE forward as a single pure function of `[x, gate_w, expert_bias,
    /// switch(gate/up/down × w/s/b), shared(gate/up/down × w/s/b)]` (21 arrays). Written to be
    /// wrapped in `compile` so MLX fuses the router (sigmoid + argpartition + gather) and the
    /// expert/shared matmuls into a handful of kernels instead of ~hundreds of eager launches.
    #[allow(clippy::too_many_arguments)]
    fn moe_compiled(
        a: &[Array],
        top_k: i32,
        group_size: i32,
        bits: i32,
        norm: bool,
        scaling: f32,
    ) -> Result<Array> {
        let x = &a[0];
        let shape = x.shape();
        let (l, d) = (shape[1], shape[2]);
        // Router: dense gate, sigmoid scores, expert-bias for selection, bias-free weights.
        let logits = matmul(x, &a[1].t())?;
        let orig = sigmoid(&logits.as_type::<f32>()?)?;
        let sel = &orig + &a[2];
        let part = argpartition_axis(&sel, -top_k, -1)?;
        let inds = part.index((.., .., (-top_k)..));
        let mut w = take_along_axis(&orig, &inds, -1)?;
        if norm {
            let denom = sum_axis(&w, -1, Some(true))? + 1e-20;
            w = &w / &denom;
        }
        if scaling != 1.0 {
            w = w * scaling;
        }
        // Routed experts via batched gather-qmm SwiGLU.
        let inds_r = inds.reshape(&[l, top_k])?;
        let xe = x.reshape(&[l, 1, 1, d])?;
        let gp = gather_qmm_mode(&xe, &a[3], &a[4], Some(&a[5]), &inds_r, true, group_size, bits, "affine")?;
        let gp = sigmoid(&gp)? * gp;
        let up = gather_qmm_mode(&xe, &a[6], &a[7], Some(&a[8]), &inds_r, true, group_size, bits, "affine")?;
        let down = gather_qmm_mode(&(gp * up), &a[9], &a[10], Some(&a[11]), &inds_r, true, group_size, bits, "affine")?;
        let eo = down.reshape(&[l, top_k, d])?.as_type::<f32>()?;
        let wr = w.reshape(&[l, top_k, 1])?;
        let mut y = sum_axis(&(eo * wr), 1, Some(false))?.reshape(&[1, l, d])?;
        // Always-on shared expert (quantized SwiGLU MLP).
        let sg = quantized_matmul_mode(x, &a[12], &a[13], Some(&a[14]), true, group_size, bits, "affine")?;
        let sg = sigmoid(&sg)? * sg;
        let su = quantized_matmul_mode(x, &a[15], &a[16], Some(&a[17]), true, group_size, bits, "affine")?;
        let sd = quantized_matmul_mode(&(sg * su), &a[18], &a[19], Some(&a[20]), true, group_size, bits, "affine")?;
        y = y + sd.as_type::<f32>()?;
        Ok(y)
    }

    thread_local! {
        // Tracks whether the MLX compile-cache entry for the MoE closure has been warmed on this
        // thread, so we only leak one `Compiled` (see below) instead of one per call.
        static MOE_CACHE_WARM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Run the MoE through its compiled+fused graph. `Compiled::drop` calls
    /// `mlx_detail_compile_erase(id)`, which would evict the cached kernel every call and force a
    /// full re-trace; the MLX cache is keyed by the closure's TypeId, so we warm it once and
    /// `mem::forget` that first `Compiled` to keep the entry alive. Later calls build a fresh
    /// (same-TypeId) `Compiled` that hits the warm cache, and are dropped normally — except we also
    /// forget them so their `Drop` can't erase the shared entry.
    fn run_moe_compiled(
        inputs: &[Array],
        top_k: i32,
        group_size: i32,
        bits: i32,
        norm: bool,
        scaling: f32,
    ) -> Result<Array> {
        let f = move |a: &[Array]| -> Vec<Array> {
            vec![
                moe_compiled(a, top_k, group_size, bits, norm, scaling)
                    .expect("compiled MoE forward"),
            ]
        };
        let mut compiled = f.compile(false);
        let out = compiled
            .call_mut(inputs)
            .map_err(|e| anyhow!("compiled MoE: {e}"))?;
        std::mem::forget(compiled);
        MOE_CACHE_WARM.with(|w| w.set(true));
        Ok(out.into_iter().next().expect("compiled MoE output"))
    }

    fn dequantize_mode(
        weight: &Array,
        scales: &Array,
        biases: Option<&Array>,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<Array> {
        let mode = CString::new(mode)?;
        let stream = Stream::default();
        let mut out = empty_array();
        let status = unsafe {
            mlx_sys::mlx_dequantize(
                &mut out as *mut _,
                weight.as_ptr(),
                scales.as_ptr(),
                biases.map(Array::as_ptr).unwrap_or_else(empty_array),
                optional_int(group_size),
                optional_int(bits),
                mode.as_ptr(),
                empty_array(), // global_scale (null) — added in mlx-c 0.6.0
                optional_dtype_none(),
                stream.as_ptr(),
            )
        };
        if status != 0 {
            unsafe { mlx_sys::mlx_array_free(out) };
            bail!("MLX dequantize failed for {bits}-bit {mode:?} weights");
        }
        Ok(unsafe { Array::from_ptr(out) })
    }

    fn v4_rope(x: &Array, dims: i32, base: f32, offset: i32, inverse: bool) -> Result<Array> {
        if dims == 0 {
            return Ok(x.clone());
        }
        let shape = x.shape();
        if shape.len() != 4 {
            bail!("DeepSeek V4 RoPE expects a 4D tensor, got shape {shape:?}");
        }
        let (b, h, t) = (shape[0], shape[1], shape[2]);
        let half = dims / 2;
        let inv_freq = (0..half)
            .map(|idx| 1.0 / base.powf((2 * idx) as f32 / dims as f32))
            .collect::<Vec<_>>();
        let pos = (0..t).map(|idx| (offset + idx) as f32).collect::<Vec<_>>();
        let theta = Array::from_slice(&pos, &[t, 1]) * Array::from_slice(&inv_freq, &[1, half]);
        let theta = if inverse { theta * -1.0 } else { theta };
        let cos = cos(&theta)?.reshape(&[1, 1, t, half])?;
        let sin = sin(&theta)?.reshape(&[1, 1, t, half])?;
        let rot = x.reshape(&[b, h, t, half, 2])?;
        let x0 = rot.index((.., .., .., .., 0));
        let x1 = rot.index((.., .., .., .., 1));
        let y0 = x0.clone() * &cos - x1.clone() * &sin;
        let y1 = x0 * sin + x1 * cos;
        stack_axis(&[y0, y1], -1)?
            .reshape(&[b, h, t, dims])
            .map_err(Into::into)
    }

    fn causal_attention_mask(query_len: i32, key_len: i32, offset: i32) -> Array {
        causal_attention_mask_with_key_start_and_window(query_len, key_len, offset, 0, None)
    }

    fn causal_attention_mask_with_key_start_and_window(
        query_len: i32,
        key_len: i32,
        query_start: i32,
        key_start: i32,
        local_window: Option<i32>,
    ) -> Array {
        let mut mask = Vec::with_capacity((query_len * key_len) as usize);
        for query_idx in 0..query_len {
            let max_key = query_start + query_idx;
            let min_key = local_window
                .map(|window| max_key + 1 - window.max(1))
                .unwrap_or(i32::MIN);
            for key_idx in 0..key_len {
                let key_pos = key_start + key_idx;
                mask.push(key_pos <= max_key && key_pos >= min_key);
            }
        }
        Array::from_slice(&mask, &[1, 1, query_len, key_len])
    }

    fn compressed_attention_mask(
        query_len: i32,
        compressed_len: i32,
        query_start: i32,
        ratio: i32,
    ) -> Array {
        let mut mask = Vec::with_capacity((query_len * compressed_len) as usize);
        for query_idx in 0..query_len {
            let max_key = query_start + query_idx;
            for block_idx in 0..compressed_len {
                let block_end = (block_idx + 1) * ratio - 1;
                mask.push(block_end <= max_key);
            }
        }
        Array::from_slice(&mask, &[1, 1, query_len, compressed_len])
    }

    fn apply_attention_mask(scores: &Array, mask: &Array) -> Result<Array> {
        let masked = Array::from_f32(f32::NEG_INFINITY);
        which(mask, scores, &masked).map_err(Into::into)
    }
}
