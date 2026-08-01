use anyhow::Result;

use crate::backend::{GenerationEvent, GenerationOutput, GenerationRequest};
use crate::config::MlxModelConfig;
use crate::generate::TokenizerRuntime;
use crate::weights::WeightCatalog;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
pub struct NativeRuntime;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
pub struct SpecStats {
    pub rounds: usize,
    pub proposed: usize,
    pub accepted: usize,
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
impl NativeRuntime {
    pub fn from_path(_path: impl AsRef<std::path::Path>) -> Result<Self> {
        anyhow::bail!("native MLX inference requires Apple Silicon macOS")
    }

    pub fn supports_speculative(&self) -> bool {
        false
    }

    pub fn supports_mtp(&self) -> bool {
        false
    }

    pub fn load(
        _config: MlxModelConfig,
        _weights: WeightCatalog,
        _tokenizer: TokenizerRuntime,
        _stream_ctx: Option<&()>,
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

    pub fn mtp_generate<F>(
        &mut self,
        _request: GenerationRequest,
        _on_event: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(GenerationEvent) -> Result<()>,
    {
        anyhow::bail!("native MLX inference requires Apple Silicon macOS")
    }

    pub fn speculative_generate<F>(
        &mut self,
        _draft: &mut NativeRuntime,
        _request: GenerationRequest,
        _k: usize,
        _on_event: F,
    ) -> Result<(GenerationOutput, SpecStats)>
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
        stream_ctx: Option<&native::StreamContext>,
    ) -> Result<Self> {
        let model = native::load_model(&config, &weights, stream_ctx)?;
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
        Self::load(config, weights, tokenizer, None)
    }

    /// Whether this model can be a speculative-decoding *target* (needs KV-cache rollback).
    pub fn supports_speculative(&self) -> bool {
        self.model.supports_rollback()
    }

    /// Test-only accessors so diagnostics can drive the model directly rather than through the
    /// generation loop, which is what lets us compare raw logits.
    #[cfg(test)]
    pub fn model_for_test(&mut self) -> &mut dyn CausalLm {
        self.model.as_mut()
    }

    #[cfg(test)]
    pub fn tokenizer_for_test(&self) -> &TokenizerRuntime {
        &self.tokenizer
    }

    /// Whether this model family implements batched decode. Callers that batch must check this and
    /// fall back to serving requests one at a time when it is false.
    pub fn supports_batch(&self) -> bool {
        // Left-padded lockstep batching and ragged per-row-position batching are both driven
        // through stream_generate_batch, which dispatches on the family internally.
        self.model.supports_batch() || self.model.supports_ragged_batch()
    }

    /// Decode several requests in one set of forward passes. `on_event` is called with the index of
    /// the originating request so the caller can demux back to per-request streams. Returns one
    /// [`GenerationOutput`] per request, in the order given.
    pub fn stream_generate_batch<F>(
        &mut self,
        requests: &[GenerationRequest],
        on_event: F,
    ) -> Result<Vec<GenerationOutput>>
    where
        F: FnMut(usize, GenerationEvent) -> Result<()>,
    {
        native::stream_generate_batch(
            &self.config,
            self.model.as_mut(),
            &self.tokenizer,
            requests,
            on_event,
        )
    }

    /// Whether this model has a built-in MTP head for self-speculative decoding (GLM-5.2).
    pub fn supports_mtp(&self) -> bool {
        self.model.supports_mtp()
    }

    /// Greedy self-speculative decoding via the model's own MTP head.
    pub fn mtp_generate<F>(
        &mut self,
        request: GenerationRequest,
        mut on_event: F,
    ) -> Result<GenerationOutput>
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
/// Preprocessed media for a multimodal prefill. Flat data + shapes (no MLX types here so the trait
/// stays buildable off the native module); the model wraps them into arrays and scatters the tower
/// outputs into the embedding stream at the placeholder-token positions.
#[derive(Clone, Default)]
pub struct MediaFeatures {
    /// `(row-major data, shape)` for vision pixel values `[num_patches, 2, 40, 40, 3]`.
    pub pixel_values: Option<(Vec<f32>, Vec<i32>)>,
    /// `(row-major bins, shape)` for audio dMel ids `[num_frames, n_mel_bins]`.
    pub audio_ids: Option<(Vec<i32>, Vec<i32>)>,
    pub image_token_id: u32,
    pub audio_token_id: u32,
}

impl MediaFeatures {
    pub fn is_empty(&self) -> bool {
        self.pixel_values.is_none() && self.audio_ids.is_none()
    }
}

pub trait CausalLm {
    fn forward(&mut self, input_ids: &[u32]) -> Result<mlx_rs::Array>;
    fn reset_cache(&mut self);
    fn prepare_cache(&mut self, _capacity: i32) {}
    /// Whether this model can decode a batch of sequences in one forward. Families that have not
    /// been audited for batch>1 return false and the server keeps serving them one at a time.
    fn supports_batch(&self) -> bool {
        false
    }
    /// Batched forward. `input_ids` is `[B, L]` (row-major, one row per sequence) and the result is
    /// logits `[B, L, vocab]`.
    ///
    /// Sequences in a batch share one KV-cache write offset, so callers left-pad every prompt to a
    /// common length and pass `pad_lens[i]` = number of pad tokens at the front of row `i`. RoPE is
    /// relative, so a uniform per-row shift does not change the relative distances between that
    /// row's real tokens — the only correctness requirement is that the padded key positions are
    /// masked out of attention, which `stage_pad_lens` arranges.
    fn forward_batch(&mut self, _input_ids: &mlx_rs::Array) -> Result<mlx_rs::Array> {
        anyhow::bail!("batched decode is not implemented for this model family")
    }
    /// Ragged batched prefill: prefill each row independently at b=1 — compression blocks,
    /// sliding windows and RoPE positions anchored at 0 exactly as in single-sequence decode —
    /// then stack the per-row caches so subsequent `forward_batch` calls on `[b, 1]` ids decode
    /// all rows in lockstep at their own logical positions. Returns stacked last-position
    /// logits `[b, 1, vocab]`. `max_steps` sizes the decode-region buffers.
    fn prefill_batch_ragged(
        &mut self,
        _prompts: &[Vec<u32>],
        _max_steps: i32,
    ) -> Result<mlx_rs::Array> {
        anyhow::bail!("ragged batched prefill is not implemented for this model family")
    }
    /// Families that batch via ragged per-row prefill (per-row positions, no padding). Their
    /// prompts must go through [`Self::prefill_batch_ragged`]; left-padded `forward_batch`
    /// prefill would misalign their position-anchored compression blocks.
    fn supports_ragged_batch(&self) -> bool {
        false
    }
    /// Stage the per-row left-padding widths used to mask padded key positions in the next
    /// `forward_batch`. Follows the same stage-then-forward pattern as [`CausalLm::set_media`].
    /// `None` clears it (no padding).
    fn stage_pad_lens(&mut self, _pad_lens: Option<&[i32]>) {}
    /// Stage preprocessed media to be scattered into the next (prefill) forward. Default no-op;
    /// only multimodal models (Inkling) override it.
    fn set_media(&mut self, _media: MediaFeatures) {}
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
    use mlx_rs::nn::{gelu, gelu_approximate, silu, softplus};
    use mlx_rs::ops::indexing::{
        IndexOp, TryIndexMutOp, argmax_axis, put_along_axis, take_along_axis,
    };
    use mlx_rs::ops::{
        argpartition_axis, broadcast_to, concatenate_axis, conv1d, cos, dequantize, einsum, exp,
        identity, matmul, maximum, mean_axis, minimum, rsqrt, sigmoid, sin, softmax_axis,
        split_sections, stack_axis, sum_axis, tanh, tril, which, zeros_dtype,
    };
    use mlx_rs::transforms::compile::{CallMut, Compile};
    use mlx_rs::{Array, Stream, transforms};

    use super::{CausalLm, MediaFeatures};
    use crate::backend::{GenerationEvent, GenerationOutput, GenerationRequest};
    use crate::config::{MlxModelConfig, QuantizationSpec};
    use crate::generate::{LogitsProcessor, TokenizerRuntime, hit_stop};
    use crate::manifest::ModelFamily;
    use crate::weights::{WeightCatalog, mlx::load_arrays};

    pub fn load_model(
        config: &MlxModelConfig,
        weights: &WeightCatalog,
        stream_ctx: Option<&StreamContext>,
    ) -> Result<Box<dyn CausalLm + Send>> {
        // When streaming, collect the expert tensor names to skip during resident load.
        let skip_tensors: Option<std::collections::BTreeSet<String>> = stream_ctx.map(|ctx| {
            ctx.sources
                .values()
                .flat_map(|s| {
                    std::iter::once(s.weight_name.clone())
                        .chain(s.scales_name.iter().cloned())
                        .chain(s.biases_name.iter().cloned())
                })
                .collect()
        });
        let mut arrays = load_arrays(weights, skip_tensors.as_ref())?;
        if config.is_deepseek_v4() {
            remap_v4_bare_weights(&mut arrays);
            return Ok(Box::new(DeepSeekV4Like::new(
                config.clone(),
                arrays,
                stream_ctx,
            )?));
        }
        match config.family {
            ModelFamily::Inkling => Ok(Box::new(InklingLike::new(
                config.clone(),
                arrays,
                stream_ctx,
            )?)),
            ModelFamily::Laguna => Ok(Box::new(LagunaLike::new(
                config.clone(),
                arrays,
                stream_ctx,
            )?)),
            ModelFamily::Qwen2 if config.model_type == "nemotron" => {
                Ok(Box::new(NemotronLike::new(config.clone(), arrays)?))
            }
            ModelFamily::Qwen2 if config.model_type == "gpt_oss" => Ok(Box::new(GptOssLike::new(
                config.clone(),
                arrays,
                stream_ctx,
            )?)),
            ModelFamily::Qwen2 if config.model_type == "cohere2" => {
                Ok(Box::new(CohereLike::new(config.clone(), arrays)?))
            }
            ModelFamily::Qwen2 if config.model_type == "phimoe" => Ok(Box::new(PhiMoeLike::new(
                config.clone(),
                arrays,
                stream_ctx,
            )?)),
            ModelFamily::Qwen2 if config.model_type.starts_with("llama4") => Ok(Box::new(
                Llama4Like::new(config.clone(), arrays, stream_ctx)?,
            )),
            ModelFamily::Qwen2 | ModelFamily::Qwen3 | ModelFamily::Hy3 => {
                // Qwen3.5 gated-delta-net hybrid (linear-attn heads present) uses its own path.
                if config.linear_num_value_heads.is_some() {
                    Ok(Box::new(Qwen35Like::new(
                        config.clone(),
                        arrays,
                        stream_ctx,
                    )?))
                } else {
                    Ok(Box::new(QwenLike::new(config.clone(), arrays, stream_ctx)?))
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
                    Ok(Box::new(MlaLike::new(config.clone(), arrays, stream_ctx)?))
                }
            }
            ModelFamily::NemotronH => Ok(Box::new(NemotronHLike::new(
                config.clone(),
                arrays,
                stream_ctx,
            )?)),
            ModelFamily::MiniMax => Ok(Box::new(MiniMaxLike::new(
                config.clone(),
                arrays,
                stream_ctx,
            )?)),
            ModelFamily::LongCat => Ok(Box::new(LongCatLike::new(
                config.clone(),
                arrays,
                stream_ctx,
            )?)),
            ModelFamily::Gemma if config.model_type.starts_with("gemma") => {
                Ok(Box::new(Gemma4TextLike::new(config.clone(), arrays)?))
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

    // Inkling media placeholder tokens (soft-token slots) and content markers.
    const INKLING_IMAGE_TOKEN_ID: u32 = 200054;
    const INKLING_AUDIO_TOKEN_ID: u32 = 200053;

    /// Build Inkling input ids with media placeholders from the original chat messages, and collect
    /// the preprocessed features. Ported from the reference `InklingProcessor.apply`: a thinking-effort
    /// system header, then each message's parts in order — text tokenized inline, each image/audio part
    /// emitting `role + content-marker + [placeholder * count] + end_message` — then the generation
    /// prompt. Media parts are matched to `request.media_inputs` in collection order.
    fn build_inkling_multimodal(
        tokenizer: &TokenizerRuntime,
        request: &GenerationRequest,
    ) -> Result<(Vec<u32>, MediaFeatures)> {
        use hi_local_core::backend::{ImageSource, MultimodalInput};
        use serde_json::Value;
        let enc = |s: &str| tokenizer.encode(s);
        let mut ids: Vec<u32> = Vec::new();
        let mut pixel_data: Vec<f32> = Vec::new();
        let mut n_patches: usize = 0;
        let mut audio_data: Vec<i32> = Vec::new();
        let mut n_frames: usize = 0;
        let mut media = request.media_inputs.iter();

        ids.extend(enc(
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
        )?);
        for message in &request.messages {
            let role = match message.role.as_str() {
                "assistant" | "model" => "<|message_model|>",
                "system" | "developer" => "<|message_system|>",
                _ => "<|message_user|>",
            };
            // Normalize content to a list of parts.
            let parts: Vec<Value> = match &message.content {
                Some(Value::String(s)) => vec![serde_json::json!({"type": "text", "text": s})],
                Some(Value::Array(a)) => a.clone(),
                _ => Vec::new(),
            };
            for part in &parts {
                let is_text = part.as_str().is_some()
                    || part.get("type").and_then(Value::as_str) == Some("text")
                    || (part.get("type").is_none() && part.get("text").is_some());
                let is_image = part.get("type").and_then(Value::as_str) == Some("image_url")
                    || (part.get("type").is_none() && part.get("image_url").is_some());
                let is_audio = part.get("type").and_then(Value::as_str) == Some("input_audio")
                    || (part.get("type").is_none() && part.get("input_audio").is_some());
                if is_text {
                    let text = part
                        .as_str()
                        .or_else(|| part.get("text").and_then(Value::as_str))
                        .unwrap_or("");
                    ids.extend(enc(&format!(
                        "{role}<|content_text|>{text}<|end_message|>"
                    ))?);
                } else if is_image {
                    let Some(MultimodalInput::Image(img)) = media.next() else {
                        bail!("image content part without a matching decoded image");
                    };
                    let ImageSource::Data { bytes, .. } = &img.source else {
                        bail!("Inkling image input must be inline data (URL fetch not wired)");
                    };
                    let cap = std::env::var("HI_MLX_INKLING_MAX_LONG_EDGE")
                        .ok()
                        .and_then(|v| v.parse::<u32>().ok());
                    let (pv, n) = crate::inkling_media::preprocess_image(bytes, cap)?;
                    pixel_data.extend(pv);
                    n_patches += n;
                    ids.extend(enc(role)?);
                    ids.extend(enc("<|content_image|>")?);
                    ids.extend(std::iter::repeat_n(INKLING_IMAGE_TOKEN_ID, n));
                    ids.extend(enc("<|end_message|>")?);
                } else if is_audio {
                    let Some(MultimodalInput::Audio(au)) = media.next() else {
                        bail!("audio content part without a matching decoded audio");
                    };
                    let (aid, n) =
                        crate::inkling_media::preprocess_audio(&au.samples, au.sampling_rate)?;
                    audio_data.extend(aid);
                    n_frames += n;
                    ids.extend(enc(role)?);
                    ids.extend(enc("<|content_audio_input|>")?);
                    ids.extend(std::iter::repeat_n(INKLING_AUDIO_TOKEN_ID, n));
                    ids.extend(enc("<|audio_end|><|end_message|>")?);
                }
            }
        }
        ids.extend(enc("<|message_model|>")?);

        let media = MediaFeatures {
            pixel_values: (!pixel_data.is_empty())
                .then(|| (pixel_data, vec![n_patches as i32, 2, 40, 40, 3])),
            audio_ids: (!audio_data.is_empty()).then(|| (audio_data, vec![n_frames as i32, 80])),
            image_token_id: INKLING_IMAGE_TOKEN_ID,
            audio_token_id: INKLING_AUDIO_TOKEN_ID,
        };
        Ok((ids, media))
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
        // Inkling multimodal: build token ids with media placeholders from the original messages and
        // stage the preprocessed features for the prefill scatter. Every other case encodes the
        // already-rendered text prompt.
        let prompt_tokens =
            if config.family == ModelFamily::Inkling && !request.media_inputs.is_empty() {
                let (ids, media) = build_inkling_multimodal(tokenizer, &request)?;
                model.set_media(media);
                ids
            } else {
                tokenizer.encode(&request.prompt)?
            };
        if prompt_tokens.is_empty() {
            bail!("prompt encoded to zero tokens");
        }
        model.reset_cache();
        // A batched call that returned early (any `?`) leaves its per-row pad mask staged on every
        // layer, and this path shares the same model object. Decoding through a stale mask yields
        // fluent-looking garbage that persists until the process restarts, so clear it up front
        // rather than relying on the batch path's exit to have run.
        model.stage_pad_lens(None);
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
            // The stop token ends the turn and is not part of the reply. Most checkpoints make it a
            // special token that decodes to nothing, so appending it first was harmless; Laguna's
            // `</assistant>` is an ordinary added token, and emitting it leaked a turn marker into
            // the content and the stream. Stop before it is decoded either way.
            if hit_stop(&[next], &config.eos_token_ids) {
                break;
            }
            generated.push(next);
            let current_text = tokenizer.decode(&generated)?;
            let delta = decoded_delta(&decoded_text, &current_text, tokenizer, next)?;
            decoded_text = current_text;
            on_event(GenerationEvent::TokenDelta {
                token_id: next,
                text: delta,
            })?;
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

    /// Decode several requests in lockstep through one set of forward passes.
    ///
    /// Prompts are left-padded to a common length so every row shares the KV cache's single write
    /// offset; `stage_pad_lens` hides the pad positions from attention. RoPE is relative, so the
    /// uniform per-row position shift that left-padding introduces does not change the relative
    /// distances between a row's real tokens.
    ///
    /// Rows that hit their stop condition early keep stepping (their sampled token is discarded)
    /// until the whole batch drains. That wastes some compute on ragged batches but keeps the
    /// cache offset shared, which is what makes the batching legal at all. `on_event` receives the
    /// row index alongside each event so the caller can demux back to per-request streams.
    pub fn stream_generate_batch<F>(
        config: &MlxModelConfig,
        model: &mut dyn CausalLm,
        tokenizer: &TokenizerRuntime,
        requests: &[GenerationRequest],
        mut on_event: F,
    ) -> Result<Vec<GenerationOutput>>
    where
        F: FnMut(usize, GenerationEvent) -> Result<()>,
    {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if model.supports_ragged_batch() {
            return stream_generate_batch_ragged(config, model, tokenizer, requests, on_event);
        }
        if !model.supports_batch() {
            bail!("model family does not implement batched decode");
        }
        let b = requests.len();
        let prompts: Vec<Vec<u32>> = requests
            .iter()
            .map(|r| tokenizer.encode(&r.prompt))
            .collect::<Result<_>>()?;
        if prompts.iter().any(Vec::is_empty) {
            bail!("prompt encoded to zero tokens");
        }
        let width = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let pad_lens: Vec<i32> = prompts.iter().map(|p| (width - p.len()) as i32).collect();
        // Pad with token 0, NOT EOS. The padded positions are masked out of attention, so the
        // value should be irrelevant — but EOS is also what finished rows are fed to keep the
        // batch in lockstep, and reusing it conflates "this slot is padding" with "this row has
        // stopped". Token 0 is always a valid embedding index and carries no stop semantics.
        let pad_id = 0u32;
        let mut flat = Vec::with_capacity(b * width);
        for prompt in &prompts {
            flat.extend(std::iter::repeat_n(pad_id, width - prompt.len()));
            flat.extend_from_slice(prompt);
        }

        let max_tokens = requests
            .iter()
            .map(|r| r.max_tokens.max(1))
            .max()
            .unwrap_or(1);
        model.reset_cache();
        model.stage_pad_lens(None);
        model.prepare_cache(
            width.saturating_add(max_tokens as usize).min(i32::MAX as usize) as i32,
        );
        model.stage_pad_lens(Some(&pad_lens));

        // Debug: compute each row's single-sequence argmax FIRST, on the exact prompts this call
        // received, then run the real batched prefill and compare. Same process, same model, same
        // inputs — so a mismatch isolates the batch path itself rather than the test harness.
        let mut single_ref: Vec<Option<u32>> = Vec::new();
        let mut replay_mismatch = false;
        if std::env::var_os("HI_MLX_BATCH_DEBUG").is_some() {
            for prompt in &prompts {
                model.reset_cache();
                model.stage_pad_lens(None);
                model.prepare_cache(prompt.len() as i32 + 8);
                let lg = model.forward(prompt)?;
                single_ref.push(crate::generate::mlx::greedy_next_token(&last_row_logits(&lg, 0)?)?);
            }
            model.reset_cache();
            model.stage_pad_lens(None);
            model.prepare_cache(
                width.saturating_add(max_tokens as usize).min(i32::MAX as usize) as i32,
            );
            model.stage_pad_lens(Some(&pad_lens));
        }

        let ids = Array::from_slice(&flat, &[b as i32, width as i32]);
        let mut logits = model.forward_batch(&ids)?;
        let debug_on = std::env::var_os("HI_MLX_BATCH_DEBUG").is_some();
        if debug_on {
            eprintln!("[batch] b={b} width={width} pad_lens={pad_lens:?}");
            // Verify the id matrix agrees with pad_lens: row i must start with pad_lens[i] pad
            // tokens and then its real prompt. A disagreement here means the mask is hiding the
            // wrong positions, which is indistinguishable from a broken mask downstream.
            for i in 0..b {
                let row = &flat[i * width..(i + 1) * width];
                let lead = row.iter().take_while(|&&t| t == pad_id).count();
                let tail_ok = row[pad_lens[i] as usize..] == prompts[i][..];
                eprintln!(
                    "[batch]  ids row {i}: lead_pad={} expected={} tail_matches_prompt={} head={:?}",
                    lead, pad_lens[i], tail_ok, &row[..row.len().min(4)]
                );
            }
            let lshape = logits.shape().to_vec();
            eprintln!("[batch] prefill logits shape={lshape:?}");
            for (i, p) in prompts.iter().enumerate() {
                // argmax of this row's prefill logits, straight out of the same tensor the
                // sampler reads. If this disagrees with a single-sequence forward of the same
                // prompt, the fault is upstream of sampling.
                let rl = last_row_logits(&logits, i as i32)?;
                let am = crate::generate::mlx::greedy_next_token(&rl)?;
                let sr = single_ref.get(i).copied().flatten();
                eprintln!(
                    "[batch]  row {i}: tokens={} pad={} batched={:?} single={:?} {}",
                    p.len(), pad_lens[i], am, sr,
                    if am == sr { "MATCH" } else { "*** MISMATCH ***" }
                );
                replay_mismatch |= am != sr;
            }
        }

        if debug_on && replay_mismatch {
            // Serialize the exact inputs so a standalone test can replay them byte-for-byte.
            // Nine hypothesised differences between this call and hand-built reproductions all
            // tested clean, so stop guessing at the difference and capture it.
            let cap = width.saturating_add(max_tokens as usize).min(i32::MAX as usize) as i32;
            let dump = format!(
                "{width} {cap} {b}\n{}\n{}\n",
                pad_lens.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
                flat.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")
            );
            let path = std::env::var("HI_MLX_BATCH_DUMP")
                .unwrap_or_else(|_| "/tmp/batch_repro.txt".to_string());
            let _ = std::fs::write(&path, dump);
            eprintln!("[batch]  dumped failing inputs to {path}");

            // Identical inputs, identical staging, fresh cache. If this run matches the
            // single-sequence reference, the first forward was polluted by residual model state
            // (the debug block's own single-sequence passes) rather than by bad inputs or a bad
            // mask — which are byte-identical between the two calls.
            model.reset_cache();
            model.stage_pad_lens(None);
            model.prepare_cache(
                width.saturating_add(max_tokens as usize).min(i32::MAX as usize) as i32,
            );
            model.stage_pad_lens(Some(&pad_lens));
            let lg2 = model.forward_batch(&ids)?;
            for i in 0..b {
                let am2 = crate::generate::mlx::greedy_next_token(&last_row_logits(&lg2, i as i32)?)?;
                eprintln!("[batch]  REPLAY row {i}: pad={} argmax={:?}", pad_lens[i], am2);
            }
            logits = lg2;
        }

        let mut processors: Vec<LogitsProcessor> = requests
            .iter()
            .map(|r| {
                LogitsProcessor::new(
                    r.temperature,
                    r.top_p,
                    1.0,
                    r.seed.unwrap_or(0x4849),
                )
            })
            .collect();
        let mut tokens: Vec<Vec<u32>> = prompts.clone();
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut decoded: Vec<String> = vec![String::new(); b];
        let mut done: Vec<bool> = vec![false; b];
        // Rows whose Finished event has already gone out, so the drain below doesn't double-send.
        let mut emitted: Vec<bool> = vec![false; b];

        for _ in 0..max_tokens {
            let mut next_ids = Vec::with_capacity(b);
            for row in 0..b {
                // Last position's logits for this row: [1, 1, vocab].
                let row_logits = last_row_logits(&logits, row as i32)?;
                let next = if done[row] {
                    None
                } else if requests[row].temperature <= f32::EPSILON {
                    crate::generate::mlx::greedy_next_token(&row_logits)?
                } else {
                    crate::generate::mlx::sample_next_token(
                        &row_logits,
                        &mut processors[row],
                        &tokens[row],
                    )?
                };
                match next {
                    Some(next) if !done[row] => {
                        tokens[row].push(next);
                        if hit_stop(&[next], &config.eos_token_ids)
                            || generated[row].len() as u32 >= requests[row].max_tokens.max(1)
                        {
                            // Finish this row now rather than when the whole batch drains:
                            // otherwise every request in a batch waits for its slowest member,
                            // which turns a throughput win into a latency regression.
                            done[row] = true;
                            finish(row, &mut emitted, &prompts, &generated, tokenizer, &mut on_event)?;
                            next_ids.push(pad_id);
                        } else {
                            generated[row].push(next);
                            let current = tokenizer.decode(&generated[row])?;
                            let delta =
                                decoded_delta(&decoded[row], &current, tokenizer, next)?;
                            decoded[row] = current;
                            on_event(
                                row,
                                GenerationEvent::TokenDelta {
                                    token_id: next,
                                    text: delta,
                                },
                            )?;
                            if debug_on && generated[row].len() <= 12 {
                                // Per-row state at each early decode step. If a row's sampler is
                                // reading another row's context, or next_ids desynchronises from
                                // row order, it shows up as tokens[] / generated[] lengths that
                                // disagree with the row's own history.
                                eprintln!(
                                    "[batch]  step {:2} row {row}: tok={next:6} tokens_len={:3} gen_len={:3} last3={:?}",
                                    generated[row].len(), tokens[row].len(), generated[row].len(),
                                    &generated[row][generated[row].len().saturating_sub(3)..]
                                );
                            }
                            next_ids.push(next);
                        }
                    }
                    _ => {
                        if !done[row] {
                            done[row] = true;
                            finish(row, &mut emitted, &prompts, &generated, tokenizer, &mut on_event)?;
                        }
                        next_ids.push(pad_id);
                    }
                }
            }
            if done.iter().all(|&d| d) {
                break;
            }
            let step = Array::from_slice(&next_ids, &[b as i32, 1]);
            logits = model.forward_batch(&step)?;
        }
        model.stage_pad_lens(None);

        let mut outputs = Vec::with_capacity(b);
        for row in 0..b {
            if !emitted[row] {
                finish(row, &mut emitted, &prompts, &generated, tokenizer, &mut on_event)?;
            }
            outputs.push(GenerationOutput {
                prompt_tokens: prompts[row].len() as u64,
                completion_tokens: generated[row].len() as u64,
                text: tokenizer.decode(&generated[row])?,
            });
        }
        Ok(outputs)
    }


    /// Batched generation for families that prefill per-row and decode at per-row positions
    /// (no padding — see [`CausalLm::prefill_batch_ragged`]). The decode loop mirrors
    /// [`stream_generate_batch`]: per-row samplers, early per-row Finished events, finished
    /// rows fed a neutral token to keep the lockstep shape.
    pub fn stream_generate_batch_ragged<F>(
        config: &MlxModelConfig,
        model: &mut dyn CausalLm,
        tokenizer: &TokenizerRuntime,
        requests: &[GenerationRequest],
        mut on_event: F,
    ) -> Result<Vec<GenerationOutput>>
    where
        F: FnMut(usize, GenerationEvent) -> Result<()>,
    {
        let b = requests.len();
        let prompts: Vec<Vec<u32>> = requests
            .iter()
            .map(|r| tokenizer.encode(&r.prompt))
            .collect::<Result<_>>()?;
        if prompts.iter().any(Vec::is_empty) {
            bail!("prompt encoded to zero tokens");
        }
        let max_tokens = requests
            .iter()
            .map(|r| r.max_tokens.max(1))
            .max()
            .unwrap_or(1);
        let mut logits = model.prefill_batch_ragged(&prompts, max_tokens as i32 + 4)?;

        let pad_id = 0u32;
        let mut processors: Vec<LogitsProcessor> = requests
            .iter()
            .map(|r| LogitsProcessor::new(r.temperature, r.top_p, 1.0, r.seed.unwrap_or(0x4849)))
            .collect();
        let mut tokens: Vec<Vec<u32>> = prompts.clone();
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut decoded: Vec<String> = vec![String::new(); b];
        let mut done: Vec<bool> = vec![false; b];
        let mut emitted: Vec<bool> = vec![false; b];

        for _ in 0..max_tokens {
            let mut next_ids = Vec::with_capacity(b);
            for row in 0..b {
                let row_logits = last_row_logits(&logits, row as i32)?;
                let next = if done[row] {
                    None
                } else if requests[row].temperature <= f32::EPSILON {
                    crate::generate::mlx::greedy_next_token(&row_logits)?
                } else {
                    crate::generate::mlx::sample_next_token(
                        &row_logits,
                        &mut processors[row],
                        &tokens[row],
                    )?
                };
                match next {
                    Some(next) if !done[row] => {
                        tokens[row].push(next);
                        if hit_stop(&[next], &config.eos_token_ids)
                            || generated[row].len() as u32 >= requests[row].max_tokens.max(1)
                        {
                            done[row] = true;
                            finish(row, &mut emitted, &prompts, &generated, tokenizer, &mut on_event)?;
                            next_ids.push(pad_id);
                        } else {
                            generated[row].push(next);
                            let current = tokenizer.decode(&generated[row])?;
                            let delta = decoded_delta(&decoded[row], &current, tokenizer, next)?;
                            decoded[row] = current;
                            on_event(
                                row,
                                GenerationEvent::TokenDelta {
                                    token_id: next,
                                    text: delta,
                                },
                            )?;
                            next_ids.push(next);
                        }
                    }
                    _ => {
                        if !done[row] {
                            done[row] = true;
                            finish(row, &mut emitted, &prompts, &generated, tokenizer, &mut on_event)?;
                        }
                        next_ids.push(pad_id);
                    }
                }
            }
            if done.iter().all(|&d| d) {
                break;
            }
            let step = Array::from_slice(&next_ids, &[b as i32, 1]);
            logits = model.forward_batch(&step)?;
        }
        model.reset_cache();

        let mut outputs = Vec::with_capacity(b);
        for row in 0..b {
            if !emitted[row] {
                finish(row, &mut emitted, &prompts, &generated, tokenizer, &mut on_event)?;
            }
            outputs.push(GenerationOutput {
                prompt_tokens: prompts[row].len() as u64,
                completion_tokens: generated[row].len() as u64,
                text: tokenizer.decode(&generated[row])?,
            });
        }
        Ok(outputs)
    }

    // Emit a row's Finished event exactly once, as soon as that row stops generating.
    fn finish<F>(
        row: usize,
        emitted: &mut [bool],
        prompts: &[Vec<u32>],
        generated: &[Vec<u32>],
        tokenizer: &TokenizerRuntime,
        on_event: &mut F,
    ) -> Result<()>
    where
        F: FnMut(usize, GenerationEvent) -> Result<()>,
    {
        if emitted[row] {
            return Ok(());
        }
        emitted[row] = true;
        on_event(
            row,
            GenerationEvent::Finished {
                output: GenerationOutput {
                    prompt_tokens: prompts[row].len() as u64,
                    completion_tokens: generated[row].len() as u64,
                    text: tokenizer.decode(&generated[row])?,
                },
            },
        )
    }

    // Slice one row's final-position logits out of a [b, seq, vocab] tensor as [1, 1, vocab], so the
    // existing single-sequence samplers can be reused unchanged.
    fn last_row_logits(logits: &Array, row: i32) -> Result<Array> {
        let shape = logits.shape();
        let (seq, vocab) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        let row_slice = logits.index((row, seq - 1, ..));
        Ok(row_slice.reshape(&[1, 1, vocab])?)
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
        Ok((
            output,
            SpecStats {
                rounds,
                proposed,
                accepted,
            },
        ))
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

    #[cfg(test)]
    mod cache_batch_row_tests {
        use super::*;


        // Grouped gathered MoE dispatch must equal the per-token per-expert reference on
        // quantized experts (the real serving case — the ragged fixture uses dense experts and
        // exercises only the fallback).
        #[test]
        fn v4_moe_grouped_matches_reference() {
            let raw: serde_json::Value = serde_json::from_str(
                r#"{
                  "architectures": ["DeepseekV4ForCausalLM"],
                  "model_type": "deepseek_v4",
                  "hidden_size": 64,
                  "intermediate_size": 64,
                  "moe_intermediate_size": 32,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "head_dim": 64,
                  "qk_rope_head_dim": 2,
                  "q_lora_rank": 4,
                  "o_lora_rank": 4,
                  "o_groups": 1,
                  "n_routed_experts": 4,
                  "n_shared_experts": 0,
                  "num_experts_per_tok": 2,
                  "num_hash_layers": 0,
                  "scoring_func": "sqrtsoftplus",
                  "norm_topk_prob": true,
                  "routed_scaling_factor": 1.0,
                  "swiglu_limit": 10.0,
                  "hc_mult": 1,
                  "hc_sinkhorn_iters": 1,
                  "hc_eps": 1e-6,
                  "compress_ratios": [0],
                  "compress_rope_theta": 160000,
                  "vocab_size": 8,
                  "max_position_embeddings": 64,
                  "rms_norm_eps": 1e-6,
                  "rope_theta": 10000,
                  "tie_word_embeddings": false,
                  "eos_token_id": 7,
                  "quantization": {"group_size": 32, "bits": 8}
                }"#,
            )
            .unwrap();
            let config =
                crate::config::parse_model_config(std::path::Path::new("mem"), raw).unwrap();

            let mut phase = 0usize;
            let mut w = |shape: &[i32]| {
                let len = shape.iter().product::<i32>() as usize;
                let vals: Vec<f32> = (0..len)
                    .map(|i| (((phase + i) as f32) * 0.113).sin() * 0.4)
                    .collect();
                phase += len;
                Array::from_slice(&vals, shape)
            };
            let mut arrays = HashMap::new();
            arrays.insert("t.ffn.gate.weight".to_string(), w(&[4, 64]));
            for (name, shape) in [
                ("gate_proj", [4, 32, 64]),
                ("up_proj", [4, 32, 64]),
                ("down_proj", [4, 64, 32]),
            ] {
                let dense = w(&shape);
                let (wq, scales, biases) =
                    mlx_rs::ops::quantize(&dense, 32, 8).expect("quantize experts");
                arrays.insert(format!("t.ffn.switch_mlp.{name}.weight"), wq);
                arrays.insert(format!("t.ffn.switch_mlp.{name}.scales"), scales);
                arrays.insert(format!("t.ffn.switch_mlp.{name}.biases"), biases);
            }
            let moe = V4MoE::load("t.ffn", 0, &arrays, &config, None).unwrap();

            let x = {
                let vals: Vec<f32> = (0..(2 * 3 * 64) as usize)
                    .map(|i| ((i as f32) * 0.201).sin())
                    .collect();
                Array::from_slice(&vals, &[2, 3, 64])
            };
            let ids = vec![0u32; 6];
            let grouped = moe.forward(&x, &ids).unwrap();
            let routes = moe.gate.route(&x, &ids).unwrap();
            let reference = moe.forward_reference(&x, &routes).unwrap();
            let diff = grouped
                .subtract(&reference)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap();
            transforms::eval([&diff]).unwrap();
            let dv = diff.item::<f32>();
            println!("  grouped vs reference max diff = {dv:.6}");
            assert!(
                dv < 1e-2,
                "grouped MoE dispatch diverges from the per-expert reference (max diff {dv})"
            );
        }

        // Batched V4 decode scatters one row's completed compression block into that row's
        // compressed-cache lane: a leading row-range write (r..r+1, .., c..c+1, ..). Verify the
        // form at b>1 through kernel-materialized readbacks (as_slice on a view is untrusted).
        #[test]
        fn row_range_index_mut_writes_correctly() {
            let (b, h, cap, d) = (4i32, 2i32, 6i32, 3i32);
            let mut buf = zeros_dtype(&[b, h, cap, d], mlx_rs::Dtype::Float32).unwrap();
            for row in 0..b {
                let col = row % cap;
                let vals: Vec<f32> = (0..(h * d) as usize)
                    .map(|i| (1000 * (row + 1)) as f32 + 1.0 + i as f32)
                    .collect();
                let block = Array::from_slice(&vals, &[1, h, 1, d]);
                buf.try_index_mut((row..row + 1, .., col..col + 1, ..), &block)
                    .unwrap();
            }
            let mat = buf.add(Array::from_f32(0.0)).unwrap();
            transforms::eval([&mat]).unwrap();
            let flat = mat.as_slice::<f32>().to_vec();
            let at = |bb: i32, hh: i32, cc: i32, dd: i32| {
                (((bb * h + hh) * cap + cc) * d + dd) as usize
            };
            for row in 0..b {
                let col = row % cap;
                for hh in 0..h {
                    for dd in 0..d {
                        let expect = (1000 * (row + 1)) as f32 + 1.0 + (hh * d + dd) as f32;
                        assert_eq!(
                            flat[at(row, hh, col, dd)],
                            expect,
                            "row {row} head {hh} col {col} d {dd}"
                        );
                    }
                }
            }
            let nonzero = flat.iter().filter(|v| **v != 0.0).count();
            assert_eq!(nonzero as i32, b * h * d, "stray writes outside the target blocks");
        }

        // v4_rope_rows with per-row offsets must equal v4_rope run per row at that row's offset.
        #[test]
        fn v4_rope_rows_matches_per_row_single() {
            let (b, h, t, dims) = (3i32, 2i32, 5i32, 4i32);
            let n = (b * h * t * dims) as usize;
            let vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.23).sin()).collect();
            let x = Array::from_slice(&vals, &[b, h, t, dims]);
            let offsets = [7i32, 0, 129];
            for inverse in [false, true] {
                let batched = v4_rope_rows(&x, dims, 10000.0, &offsets, inverse).unwrap();
                for row in 0..b {
                    let xr = x
                        .index((row..row + 1, .., .., ..))
                        .add(Array::from_f32(0.0))
                        .unwrap();
                    let single = v4_rope(&xr, dims, 10000.0, offsets[row as usize], inverse).unwrap();
                    let diff = batched
                        .index((row..row + 1, .., .., ..))
                        .subtract(&single)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .max(None)
                        .unwrap();
                    transforms::eval([&diff]).unwrap();
                    let dv = diff.item::<f32>();
                    assert!(
                        dv < 1e-5,
                        "inverse={inverse} row {row} offset {} differs by {dv}",
                        offsets[row as usize]
                    );
                }
            }
        }


        // Row-distinct constants through the dense (preallocated) cache at b=4, checked against
        // the concat cache, which is trivially correct (pure concatenation). A disagreement means
        // the dense write path scrambles batch rows — the corruption shape seen in batched decode,
        // where prefill logits are right (computed before the cache is read back) and the first
        // decode step is wrong for every row but row 0.
        // RoPE at decode geometry (l=1) with b>1: every row must get the SAME rotation
        // (position = offset), so identical input rows must produce identical output rows.
        // If row r instead gets position offset + r, batched decode sees a growing positional
        // hole between prompt and generation for every row after the first — exactly the
        // observed "row 0 exact, rows 1+ think their token is one step further away".
        #[test]
        fn rope_l1_rows_get_same_position() {
            let (b, h, d) = (4i32, 2i32, 64i32);
            let one_row: Vec<f32> = (0..(h * d) as usize).map(|i| ((i as f32) * 0.17).sin()).collect();
            let mut vals = Vec::new();
            for _ in 0..b {
                vals.extend_from_slice(&one_row);
            }
            let x = Array::from_slice(&vals, &[b, h, 1, d]);
            let offset = 33;
            let out = rope(x, d, false, 1_000_000.0, 1.0, offset, None).unwrap();
            let out = out.add(Array::from_f32(0.0)).unwrap();
            transforms::eval([&out]).unwrap();
            let o = out.as_slice::<f32>().to_vec();
            let slab = (h * d) as usize;
            // reference: the same single row at b=1, same offset
            let x1 = Array::from_slice(&one_row, &[1, h, 1, d]);
            let r1 = rope(x1, d, false, 1_000_000.0, 1.0, offset, None).unwrap();
            let r1 = r1.add(Array::from_f32(0.0)).unwrap();
            transforms::eval([&r1]).unwrap();
            let refrow = r1.as_slice::<f32>().to_vec();
            // and rows at offsets 34..36, to identify the wrong position if rows differ
            for extra in 1..4 {
                let xr = Array::from_slice(&one_row, &[1, h, 1, d]);
                let rr = rope(xr, d, false, 1_000_000.0, 1.0, offset + extra, None).unwrap();
                let rr = rr.add(Array::from_f32(0.0)).unwrap();
                transforms::eval([&rr]).unwrap();
                let rv = rr.as_slice::<f32>().to_vec();
                let row = &o[extra as usize * slab..(extra as usize + 1) * slab];
                let d_ref: f32 = row.iter().zip(&refrow).map(|(a, c)| (a - c).abs()).fold(0.0, f32::max);
                let d_off: f32 = row.iter().zip(&rv).map(|(a, c)| (a - c).abs()).fold(0.0, f32::max);
                println!(
                    "  batched row {extra}: max|row - rope(offset={offset})| = {d_ref:.6}   max|row - rope(offset={})| = {d_off:.6}",
                    offset + extra
                );
            }
            let d0: f32 = o[..slab].iter().zip(&refrow).map(|(a, c)| (a - c).abs()).fold(0.0, f32::max);
            println!("  batched row 0 vs reference: {d0:.6}");
            let worst: f32 = (1..b as usize)
                .map(|r| {
                    o[r * slab..(r + 1) * slab]
                        .iter()
                        .zip(&o[..slab])
                        .map(|(a, c)| (a - c).abs())
                        .fold(0.0, f32::max)
                })
                .fold(0.0, f32::max);
            println!("  raw rope row skew (upstream bug, documented): {worst:.6}");
            // The fix: rope_rows must give every row the identical (correct) rotation.
            let xf = Array::from_slice(&vals, &[b, h, 1, d]);
            let fixed = rope_rows(xf, d, false, 1_000_000.0, 1.0, offset).unwrap();
            let fixed = fixed.add(Array::from_f32(0.0)).unwrap();
            transforms::eval([&fixed]).unwrap();
            let fo = fixed.as_slice::<f32>().to_vec();
            let worst_fixed: f32 = (0..b as usize)
                .map(|r| {
                    fo[r * slab..(r + 1) * slab]
                        .iter()
                        .zip(&refrow)
                        .map(|(a, c)| (a - c).abs())
                        .fold(0.0, f32::max)
                })
                .fold(0.0, f32::max);
            println!("  rope_rows worst row vs reference: {worst_fixed:.6}");
            assert!(
                worst_fixed < 1e-5,
                "rope_rows rows disagree with the b=1 reference (max diff {worst_fixed})"
            );
        }

        // The previous probe used toy geometry (f32, d=4, no GQA, no mask) and found nothing.
        // This one mirrors the exact decode call the 0.5B makes at step 1: bf16, head_dim 64,
        // GQA 14 q-heads over 2 kv-heads, kv as an axis-2 prefix view of a preallocated buffer,
        // and the real pad_attention_bias for pads [1,0,0,1]. Each factor is toggled so the
        // failing combination names itself.
        #[test]
        fn sdpa_decode_geometry_strided_vs_contiguous() {
            use mlx_rs::Dtype;
            let (b, hq, hkv, s, d, cap) = (4i32, 14i32, 2i32, 34i32, 64i32, 61i32);
            let n = (b * hkv * s * d) as usize;
            let kv_vals: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.317).sin() * 0.5).collect();
            let kfull32 = Array::from_slice(&kv_vals, &[b, hkv, s, d]);
            let qn = (b * hq * d) as usize;
            let q_vals: Vec<f32> = (0..qn).map(|i| ((i as f32) * 0.131).cos() * 0.5).collect();
            let q32 = Array::from_slice(&q_vals, &[b, hq, 1, d]);
            let scale = 0.125f32;
            let pads = [1i32, 0, 0, 1];

            for (label, dt) in [("f32", Dtype::Float32), ("bf16", Dtype::Bfloat16)] {
                let kfull = kfull32.as_dtype(dt).unwrap();
                let q = q32.as_dtype(dt).unwrap();
                let mut buf = zeros_dtype(&[b, hkv, cap, d], dt).unwrap();
                buf.try_index_mut((.., .., 0..s, ..), &kfull).unwrap();
                let kview = buf.index((.., .., ..s, ..));
                let kmat = kview.add(Array::from_f32(0.0)).unwrap().as_dtype(dt).unwrap();
                transforms::eval([&kmat]).unwrap();

                // no mask
                let ov = scaled_dot_product_attention(&q, &kview, &kview, scale, None, None::<&Array>).unwrap();
                let om = scaled_dot_product_attention(&q, &kmat, &kmat, scale, None, None::<&Array>).unwrap();
                let d0 = ov.subtract(&om).unwrap().abs().unwrap().max(None).unwrap();
                transforms::eval([&d0]).unwrap();

                // real decode bias: pad_attention_bias(pads, l=1, kv_len=s, offset=s-1, n_heads=hq)
                let bias = pad_attention_bias(&pads, 1, s, s - 1, hq).as_dtype(dt).unwrap();
                let ovb = scaled_dot_product_attention(
                    &q, &kview, &kview, scale,
                    ScaledDotProductAttentionMask::Array(&bias), None::<&Array>).unwrap();
                let omb = scaled_dot_product_attention(
                    &q, &kmat, &kmat, scale,
                    ScaledDotProductAttentionMask::Array(&bias), None::<&Array>).unwrap();
                let d1 = ovb.subtract(&omb).unwrap().abs().unwrap().max(None).unwrap();
                transforms::eval([&d1]).unwrap();

                // also: contiguous KV with mask, strided vs itself across mask presence is not
                // the question — but check view-vs-materialized OUTPUT PER ROW for the masked
                // case so a single bad row is visible.
                let dr = ovb.subtract(&omb).unwrap().abs().unwrap();
                let per_row = dr.reshape(&[b, hq * d]).unwrap().max_axis(1, false).unwrap();
                transforms::eval([&per_row]).unwrap();
                let pr: Vec<f32> = per_row.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
                println!(
                    "  [{label}] no-mask diff = {:.6}  masked diff = {:.6}  per-row(masked) = {pr:?}",
                    d0.as_dtype(Dtype::Float32).unwrap().item::<f32>(),
                    d1.as_dtype(Dtype::Float32).unwrap().item::<f32>(),
                );
            }
        }

        // SDPA at l=1 with the exact KV geometry batched decode produces: an axis-2 prefix view
        // of a preallocated [b, h, cap, d] buffer, at b>1. Compared against the same call with
        // the views materialized to contiguous. A nonzero diff convicts the l=1 fast kernel's
        // stride handling — which would explain row 0 clean / rows 1+ degenerate, b=1 clean,
        // and prefill (l>1, different kernel) clean.
        #[test]
        fn sdpa_l1_with_strided_kv_at_batch_gt_1() {
            let (b, h, s, d, cap) = (2i32, 2i32, 6i32, 4i32, 10i32);
            let kv_vals: Vec<f32> = (0..(b * h * s * d) as usize).map(|i| (i as f32) * 0.01).collect();
            let kfull = Array::from_slice(&kv_vals, &[b, h, s, d]);
            let mut buf = zeros_dtype(&[b, h, cap, d], kfull.dtype()).unwrap();
            buf.try_index_mut((.., .., 0..s, ..), &kfull).unwrap();
            let kview = buf.index((.., .., ..s, ..));
            let kmat = kview.add(Array::from_f32(0.0)).unwrap();
            transforms::eval([&kmat]).unwrap();

            let q_vals: Vec<f32> = (0..(b * h * d) as usize).map(|i| (i as f32) * 0.1 - 1.0).collect();
            let q = Array::from_slice(&q_vals, &[b, h, 1, d]);
            let scale = 0.5f32;

            let o_view =
                scaled_dot_product_attention(&q, &kview, &kview, scale, None, None::<&Array>)
                    .unwrap();
            let o_mat =
                scaled_dot_product_attention(&q, &kmat, &kmat, scale, None, None::<&Array>).unwrap();
            let diff = o_view.subtract(&o_mat).unwrap().abs().unwrap().max(None).unwrap();
            transforms::eval([&diff]).unwrap();
            let dv = diff.item::<f32>();
            println!("  sdpa(l=1, b={b}) strided-vs-contiguous max diff = {dv}");

            // Same comparison at b=1 (the geometry single-sequence decode uses, known good).
            let q1 = q.index((0..1, .., .., ..));
            let kv1 = buf.index((0..1, .., ..s, ..));
            let kv1m = kv1.add(Array::from_f32(0.0)).unwrap();
            let o1v =
                scaled_dot_product_attention(&q1, &kv1, &kv1, scale, None, None::<&Array>).unwrap();
            let o1m =
                scaled_dot_product_attention(&q1, &kv1m, &kv1m, scale, None, None::<&Array>).unwrap();
            let d1 = o1v.subtract(&o1m).unwrap().abs().unwrap().max(None).unwrap();
            transforms::eval([&d1]).unwrap();
            println!("  sdpa(l=1, b=1) strided-vs-contiguous max diff = {}", d1.item::<f32>());

            assert!(
                dv < 1e-4,
                "sdpa l=1 fast path mishandles strided kv at b>1 (max diff {dv})"
            );
        }

        #[test]
        fn dense_cache_preserves_batch_rows() {
            let (b, h, l, d) = (4i32, 2i32, 5i32, 3i32);
            let mut vals = Vec::new();
            for row in 0..b {
                for hh in 0..h {
                    for pos in 0..l {
                        for dd in 0..d {
                            vals.push((row * 1000 + hh * 100 + pos * 10 + dd) as f32);
                        }
                    }
                }
            }
            let k = Array::from_slice(&vals, &[b, h, l, d]);

            let mut dense = Cache::new();
            dense.prepare_capacity(l + 8);
            let (dk, _dv) = dense.update(k.clone(), k.clone()).unwrap();
            let mut concat = Cache::new();
            let (ck, _cv) = concat.update(k.clone(), k.clone()).unwrap();
            assert_eq!(dk.shape(), ck.shape(), "prefill readback shape");
            // as_slice on a strided view reads raw buffer memory, not the logical view —
            // materialize through a kernel first (see rope_rows / the batched-decode postmortem).
            let dk = dk.add(Array::from_f32(0.0)).unwrap();
            let ck = ck.add(Array::from_f32(0.0)).unwrap();
            transforms::eval([&dk, &ck]).unwrap();
            let (df, cf) = (dk.as_slice::<f32>().to_vec(), ck.as_slice::<f32>().to_vec());
            let bad = df.iter().zip(&cf).filter(|(a, c)| a != c).count();
            assert_eq!(bad, 0, "prefill: dense readback differs from concat in {bad} elements");

            // One decode step: a single new position per row.
            let mut step = Vec::new();
            for row in 0..b {
                for hh in 0..h {
                    for dd in 0..d {
                        step.push((90000 + row * 1000 + hh * 100 + dd) as f32);
                    }
                }
            }
            let s = Array::from_slice(&step, &[b, h, 1, d]);
            let (dk2, _) = dense.update(s.clone(), s.clone()).unwrap();
            let (ck2, _) = concat.update(s.clone(), s.clone()).unwrap();
            assert_eq!(dk2.shape(), ck2.shape(), "decode readback shape");
            let dk2 = dk2.add(Array::from_f32(0.0)).unwrap();
            let ck2 = ck2.add(Array::from_f32(0.0)).unwrap();
            transforms::eval([&dk2, &ck2]).unwrap();
            let (df2, cf2) = (dk2.as_slice::<f32>().to_vec(), ck2.as_slice::<f32>().to_vec());
            let mismatches: Vec<usize> = df2
                .iter()
                .zip(&cf2)
                .enumerate()
                .filter(|(_, (a, c))| a != c)
                .map(|(i, _)| i)
                .collect();
            assert!(
                mismatches.is_empty(),
                "decode: dense readback differs from concat at {} of {} elements \
                 (first at flat index {:?}: dense={} concat={})",
                mismatches.len(),
                df2.len(),
                mismatches.first(),
                mismatches.first().map(|&i| df2[i]).unwrap_or(f32::NAN),
                mismatches.first().map(|&i| cf2[i]).unwrap_or(f32::NAN),
            );
        }
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

        /// Logical input width. For a dense weight it is the trailing dim; for an affine-quantized
        /// weight the values are packed `32/bits` per u32, so the stored trailing dim is scaled up.
        fn in_features(&self) -> i32 {
            match self {
                Self::Dense { weight, .. } => weight.shape()[weight.shape().len() - 1],
                Self::Quantized { weight, bits, .. } => {
                    weight.shape()[weight.shape().len() - 1] * (32 / bits)
                }
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
        scale: f32,
        rope_theta: f32,
        traditional_rope: bool,
        use_rope: bool,
        // OLMo2 applies q/k RMSNorm to the full projection (dim = heads*head_dim) before reshape;
        // Qwen3 applies it per-head (dim = head_dim) after reshape.
        qk_norm_full: bool,
        cache: Cache,
        // Per-row left-padding widths for batched decode, staged by `QwenLike::stage_pad_lens`.
        // `None` (the single-sequence path) skips all pad-mask work entirely.
        pad_lens: Option<Vec<i32>>,
    }

    /// Additive attention bias `[b, 1, l, kv_len]` that hides left-padded key positions.
    ///
    /// Row `i` of a batch is padded with `pad_lens[i]` tokens at the front, so key positions
    /// `0..pad_lens[i]` hold garbage and must score -inf. When `l > 1` (prefill) the causal
    /// constraint is folded into the same array: query `qi` sits at absolute position
    /// `offset + qi` and may not see keys beyond it.
    /// RoPE that is safe for batched rows. The fast RoPE kernel mis-rotates rows when the
    /// leading batch dimension exceeds 1 with a short sequence axis: identical rows come back
    /// with differing rotations for the same offset (measured by
    ///  — row 0 exact, every later row garbled). Positions
    /// depend only on the sequence axis, so folding the batch into the head axis is exact and
    /// puts the call into the dim0 == 1 geometry the kernel computes correctly at every
    /// sequence length. b == 1 passes through untouched.
    fn rope_rows(
        x: Array,
        head_dim: i32,
        traditional: bool,
        theta: f32,
        scale: f32,
        offset: i32,
    ) -> Result<Array> {
        let shape = x.shape().to_vec();
        let (b, h, l, d) = (shape[0], shape[1], shape[2], shape[3]);
        if b <= 1 {
            return Ok(rope(x, head_dim, traditional, theta, scale, offset, None)?);
        }
        let folded = x.reshape(&[1, b * h, l, d])?;
        let out = rope(folded, head_dim, traditional, theta, scale, offset, None)?;
        Ok(out.reshape(&[b, h, l, d])?)
    }

    fn pad_attention_bias(pad_lens: &[i32], l: i32, kv_len: i32, offset: i32, n_heads: i32) -> Array {
        let b = pad_lens.len() as i32;
        let mut bias = vec![0.0f32; (b * l * kv_len) as usize];
        for (row, &pad) in pad_lens.iter().enumerate() {
            for qi in 0..l {
                // A query that IS a pad position has no legal key under the normal rules:
                // `ki < pad` hides the padding and the cache bound hides everything after it,
                // leaving an all--inf row whose softmax is NaN — which propagates through the
                // residual stream and corrupts the row's REAL positions too.
                //
                // Let such queries attend the padding instead (drop the `ki < pad` rule for
                // them) but STILL apply the cache bound. Skipping the row entirely, as an
                // earlier version did, let them read past the write head into uninitialised
                // cache — trading the NaN for different garbage.
                // Query qi sits at key position `key_base + qi`. Deriving key_base from the
                // tensor (kv_len - l) rather than from `offset` keeps the alignment correct even
                // when the cache hands back more keys than have been written — the two agree
                // whenever kv_len == offset + l, and only this form survives when it doesn't.
                let key_base = kv_len - l;
                let is_pad_query = key_base + qi < pad;
                let base = ((row as i32 * l + qi) * kv_len) as usize;
                for ki in 0..kv_len {
                    // `ki > offset + qi` is the causal bound during prefill, but it is ALSO the
                    // bound on what has actually been written to the cache. A dense cache is
                    // preallocated to its full capacity, so during decode (l == 1) positions
                    // past the write head hold uninitialised values — and without the causal
                    // term they were being attended. Apply the bound unconditionally.
                    let masked = (!is_pad_query && ki < pad) || ki > key_base + qi;
                    if masked {
                        bias[base + ki as usize] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        // Materialise across heads rather than relying on a size-1 broadcast. With a singleton
        // head dim a mixed-padding batch behaved as though ONE row's mask applied to all rows:
        // padded rows were correct while unpadded rows lost position 0 and degenerated. Building
        // the full [b, n_heads, l, kv_len] tensor removes the ambiguity.
        let per_row = (l * kv_len) as usize;
        let mut full = Vec::with_capacity(b as usize * n_heads as usize * per_row);
        for row in 0..b as usize {
            for _ in 0..n_heads {
                full.extend_from_slice(&bias[row * per_row..(row + 1) * per_row]);
            }
        }
        Array::from_slice(&full, &[b, n_heads, l, kv_len])
    }

    impl QwenAttention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            use_rope: bool,
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
                scale: config
                    .attention_multiplier
                    .unwrap_or((config.attention_head_dim() as f32).powf(-0.5)),
                rope_theta: config.rope_theta,
                // Qwen2 rotates half-split like every other arch on this path (granite, smollm3,
                // seed_oss, internlm, ernie4_5_moe); hardcoding interleaved here silently corrupted
                // them — positions near 0 are ~identity, so short prompts looked fine while longer
                // ones degenerated. Honor the checkpoint instead, defaulting off like mlx-lm.
                traditional_rope: config.rope_traditional,
                use_rope,
                qk_norm_full: arrays
                    .get(&format!("{prefix}.q_norm.weight"))
                    .map(|w| *w.shape().last().unwrap() > config.attention_head_dim() as i32)
                    .unwrap_or(false),
                cache: Cache::new(),
                pad_lens: None,
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l) = (shape[0], shape[1]);
            let mut q = self.q_proj.forward(x)?;
            let mut k = self.k_proj.forward(x)?;
            // OLMo2: normalize the full projection before splitting into heads.
            if self.qk_norm_full {
                if let Some(norm) = &self.q_norm {
                    q = norm.forward(&q)?;
                }
                if let Some(norm) = &self.k_norm {
                    k = norm.forward(&k)?;
                }
            }
            let mut q = q.reshape(&[b, l, self.n_heads, self.head_dim])?;
            let mut k = k.reshape(&[b, l, self.n_kv_heads, self.head_dim])?;
            // Qwen3 / EXAONE-4: per-head qk-norm after reshape.
            if !self.qk_norm_full {
                if let Some(norm) = &self.q_norm {
                    q = norm.forward(&q)?;
                }
                if let Some(norm) = &self.k_norm {
                    k = norm.forward(&k)?;
                }
            }
            q = q.transpose_axes(&[0, 2, 1, 3])?;
            k = k.transpose_axes(&[0, 2, 1, 3])?;
            let v = self
                .v_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let offset = self.cache.offset;
            if self.use_rope {
                q = rope_rows(
                    q,
                    self.head_dim,
                    self.traditional_rope,
                    self.rope_theta,
                    1.0,
                    offset,
                )?;
                k = rope_rows(
                    k,
                    self.head_dim,
                    self.traditional_rope,
                    self.rope_theta,
                    1.0,
                    offset,
                )?;
            }
            let (k, v) = self.cache.update(k, v)?;
            let scale = self.scale;
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
            // Batched rows are left-padded to a common length, so the leading key positions of a
            // short row hold pad tokens and must not be attended. When pad widths are staged,
            // recompute with an explicit bias that hides them (and, for prefill, folds in the
            // causal constraint). Single-sequence decoding never stages pad_lens, so the fast
            // paths above are untouched.
            let output = match self.pad_lens.as_ref() {
                // HI_MLX_FORCE_BIAS applies the explicit bias even with zero padding, so the
                // bias can be compared against the model's built-in causal mask on a batch that
                // needs no padding at all. If a fully-unpadded batch breaks under the bias, the
                // bias construction is wrong in general rather than mishandling padding.
                Some(pads)
                    if pads.iter().any(|&p| p > 0)
                        || std::env::var_os("HI_MLX_FORCE_BIAS").is_some() =>
                {
                    // Built in f32; SDPA requires the mask to promote to the query dtype (bf16),
                    // matching how the other additive masks in this file are cast.
                    let bias = pad_attention_bias(pads, l, k.shape()[2], offset, self.n_heads)
                        .as_dtype(q.dtype())?;
                    if std::env::var_os("HI_MLX_BATCH_DEBUG").is_some() {
                        // Print the actual tensors reaching SDPA rather than inferring them.
                        // q is [b, n_heads, l, head_dim]; the bias must line up on b/heads/l/kv.
                        eprintln!(
                            "[sdpa] q={:?} k={:?} bias={:?} pads={:?} l={l} offset={offset}",
                            q.shape(), k.shape(), bias.shape(), pads
                        );
                    }
                    scaled_dot_product_attention(
                        &q,
                        &k,
                        &v,
                        scale,
                        ScaledDotProductAttentionMask::Array(&bias),
                        None::<&Array>,
                    )?
                }
                _ => output,
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

    // deepseek_yarn rope for LongCat-2.0: returns (per-dim rope freqs, attention scale multiplier
    // mscale^2). Only active for LongCat with a rope_scaling of type deepseek_yarn; otherwise (None, 1).
    fn longcat_yarn_rope(config: &MlxModelConfig, dim: i32) -> Result<(Option<Array>, f32)> {
        use crate::manifest::ModelFamily;
        if config.family != ModelFamily::LongCat {
            return Ok((None, 1.0));
        }
        let Some(rs) = config.rope_scaling.as_ref() else {
            return Ok((None, 1.0));
        };
        let getf = |k: &str, d: f64| rs.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let factor = getf("factor", 1.0);
        if factor <= 1.0 {
            return Ok((None, 1.0));
        }
        let beta_fast = getf("beta_fast", 32.0);
        let beta_slow = getf("beta_slow", 1.0);
        let orig_max = getf("original_max_position_embeddings", 4096.0);
        let mscale_all_dim = getf("mscale_all_dim", 0.0);
        let base = config.rope_theta as f64;
        let half = (dim / 2) as usize;
        // Standard extrapolation freqs (theta per dim), and interpolated freqs (theta * factor).
        let theta: Vec<f64> = (0..half)
            .map(|i| base.powf(2.0 * i as f64 / dim as f64))
            .collect();
        // Correction range (in half-dim units) between beta_fast and beta_slow rotations.
        let find_dim = |num_rot: f64| {
            dim as f64 * (orig_max / (num_rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = find_dim(beta_fast).floor().max(0.0);
        let high = find_dim(beta_slow).ceil().min((half - 1) as f64);
        let denom = (high - low).max(1e-3);
        let freqs: Vec<f32> = (0..half)
            .map(|i| {
                let inv_extra = 1.0 / theta[i]; // extrapolation inv_freq
                let inv_inter = inv_extra / factor; // interpolation inv_freq
                let ramp = (((i as f64) - low) / denom).clamp(0.0, 1.0);
                let mask = 1.0 - ramp; // 1 at high freq (extrapolate), 0 at low freq (interpolate)
                let inv = inv_inter * (1.0 - mask) + inv_extra * mask;
                (1.0 / inv) as f32 // back to theta for mx rope `freqs`
            })
            .collect();
        let mscale = 0.1 * mscale_all_dim * factor.ln() + 1.0;
        Ok((
            Some(Array::from_slice(&freqs, &[half as i32])),
            (mscale * mscale) as f32,
        ))
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
        // LongCat-2.0: absorbed-MLA lora scaling + YARN rope freqs (None for other MLA archs).
        mla_scale_q: Option<f32>,
        mla_scale_kv: Option<f32>,
        rope_freqs: Option<Array>,
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
            let base_scale = (q_head_dim as f32).powf(-0.5);
            // deepseek_yarn rope: precompute per-dim freqs + attention mscale (LongCat-2.0 only).
            let (rope_freqs, mscale_sq) = longcat_yarn_rope(config, qk_rope_head_dim)?;
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
                scale: base_scale * mscale_sq,
                rope_theta: config.rope_theta,
                mla_scale_q: config.mla_scale_q_lora.then(|| {
                    (config.hidden_size as f32 / config.q_lora_rank.unwrap_or(1) as f32).sqrt()
                }),
                mla_scale_kv: config
                    .mla_scale_kv_lora
                    .then(|| (config.hidden_size as f32 / kv_lora_rank as f32).sqrt()),
                rope_freqs,
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
                    let mut query_latent = q_norm.forward(&q_a.forward(x)?)?;
                    if let Some(s) = self.mla_scale_q {
                        query_latent = query_latent * s;
                    }
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
            let mut kv_latent = self.kv_a_layernorm.forward(&compressed_kv)?;
            if let Some(s) = self.mla_scale_kv {
                kv_latent = kv_latent * s;
            }
            let mut kv_latent = kv_latent.expand_dims(1)?;

            let offset = self.cache.offset;
            let (rbase, rfreqs) = match &self.rope_freqs {
                Some(f) => (None, Some(f)),
                None => (Some(self.rope_theta), None),
            };
            q_pe = rope(
                q_pe,
                self.qk_rope_head_dim,
                true,
                rbase,
                1.0,
                offset,
                rfreqs,
            )?;
            k_pe = rope(
                k_pe,
                self.qk_rope_head_dim,
                true,
                rbase,
                1.0,
                offset,
                rfreqs,
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
        /// When `Some`, expert weights are fetched on demand from this pool
        /// instead of from the resident `weight`/`scales`/`biases` arrays. The
        /// resident arrays are empty placeholders in this case.
        stream: Option<StreamRef>,
    }

    /// The shared expert pool + metadata identifying which (layer, projection)
    /// this `SwitchLinear` fetches from.
    struct StreamRef {
        pool: std::sync::Arc<std::sync::Mutex<crate::expert_pool::ExpertPool>>,
        layer: u32,
        projection: &'static str,
        weight_name: String,
        scales_name: Option<String>,
        biases_name: Option<String>,
    }

    /// Context passed down the MoE load chain when expert streaming is enabled.
    /// Carries the shared pool and the stream plan's expert sources so each
    /// `SwitchLinear` can find its tensor names for the current layer.
    pub struct StreamContext {
        pool: std::sync::Arc<std::sync::Mutex<crate::expert_pool::ExpertPool>>,
        /// Expert sources from the plan, keyed by (layer, projection) for lookup.
        sources: std::collections::HashMap<(u32, &'static str), crate::expert_stream::ExpertSource>,
    }

    impl StreamContext {
        /// Build the context from the stream plan + a constructed pool.
        pub fn new(
            plan: &crate::expert_stream::ExpertStreamPlan,
            pool: crate::expert_pool::ExpertPool,
        ) -> Self {
            let sources = plan.sources.iter().map(|s| (s.key(), s.clone())).collect();
            StreamContext {
                pool: std::sync::Arc::new(std::sync::Mutex::new(pool)),
                sources,
            }
        }

        fn source(
            &self,
            layer: u32,
            projection: &'static str,
        ) -> Option<&crate::expert_stream::ExpertSource> {
            self.sources.get(&(layer, projection))
        }
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
                stream: None,
            })
        }

        /// Construct a streaming `SwitchLinear` that fetches expert slabs from
        /// the shared pool on demand. The quantization spec comes from the
        /// config (no resident weight to infer bits from).
        /// Dispatch between resident (`load`) and streaming (`load_streaming`)
        /// based on whether a `StreamContext` is provided and has a source for
        /// this (layer, projection).
        fn load_or_stream(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
            projection: &'static str,
        ) -> Result<Self> {
            // Extract the layer index from the prefix: "model.layers.{N}.mlp..."
            let layer = extract_layer_from_prefix(prefix);
            if let Some(ctx) = stream_ctx {
                if let Some(src) = ctx.source(layer, projection) {
                    return Self::load_streaming(
                        prefix,
                        config,
                        ctx.pool.clone(),
                        layer,
                        projection,
                        src.weight_name.clone(),
                        src.scales_name.clone(),
                        src.biases_name.clone(),
                    );
                }
            }
            // Resident path: the expert tensors must be in `arrays`.
            Self::load(prefix, arrays, config)
        }

        fn load_streaming(
            prefix: &str,
            config: &MlxModelConfig,
            pool: std::sync::Arc<std::sync::Mutex<crate::expert_pool::ExpertPool>>,
            layer: u32,
            projection: &'static str,
            weight_name: String,
            scales_name: Option<String>,
            biases_name: Option<String>,
        ) -> Result<Self> {
            let spec =
                config
                    .quantization
                    .mlx_quantization_for(prefix)?
                    .unwrap_or(QuantizationSpec {
                        bits: 4,
                        group_size: 64,
                        mode: crate::config::QuantizationMode::Affine,
                    });
            if scales_name.is_some() && biases_name.is_none() {
                require_biases_for_affine(prefix, &spec, None)?;
            }
            Ok(Self {
                // Placeholder weight — never used in the streaming path (the
                // pool supplies the real expert slabs). A 0-element f32 array
                // avoids allocating real storage.
                weight: Array::zeros::<f32>(&[]).unwrap_or_else(|_| unsafe {
                    Array::from_raw_data(std::ptr::null(), &[], mlx_rs::Dtype::Float32)
                }),
                scales: None,
                biases: None,
                group_size: spec.group_size as i32,
                bits: spec.bits as i32,
                mode: spec.mode.as_str().to_string(),
                stream: Some(StreamRef {
                    pool,
                    layer,
                    projection,
                    weight_name,
                    scales_name,
                    biases_name,
                }),
            })
        }

        fn forward_expert(&self, x: &Array, expert: i32) -> Result<Array> {
            // Streaming path: fetch expert slab from the pool on demand.
            if let Some(sref) = &self.stream {
                let mut pool = sref.pool.lock().unwrap();
                let weight = pool.weight_array(
                    sref.layer,
                    sref.projection,
                    expert as u32,
                    &sref.weight_name,
                )?;
                match &sref.scales_name {
                    Some(scales_name) => {
                        let scales = pool.scales_array(
                            sref.layer,
                            sref.projection,
                            expert as u32,
                            scales_name,
                        )?;
                        let biases = sref
                            .biases_name
                            .as_ref()
                            .map(|bn| {
                                pool.biases_array(sref.layer, sref.projection, expert as u32, bn)
                            })
                            .transpose()?;
                        // Trust the slab's own shapes over the config's bit width: the per-tensor
                        // quantization list can omit tensors, leaving them on a file-level default
                        // that does not match the data.
                        let bits =
                            derived_quant_bits(&weight, &scales, self.group_size, &self.mode)
                                .unwrap_or(self.bits);
                        // Name the tensor on failure: a bare "quantized_matmul failed for N-bit"
                        // does not say which of the streamed experts, shared experts or attention
                        // projections tripped, which matters on mixed-precision checkpoints where
                        // the bit width differs per tensor.
                        quantized_matmul_mode(
                            x,
                            &weight,
                            &scales,
                            biases.as_ref(),
                            true,
                            self.group_size,
                            bits,
                            &self.mode,
                        )
                        .map_err(|e| {
                            anyhow!("{e} (streamed expert {} #{expert})", sref.weight_name)
                        })
                    }
                    None => matmul(x, &weight.t()).map_err(Into::into),
                }
            } else {
                // Resident path (original).
                let weight = self.weight.index(expert);
                match &self.scales {
                    Some(scales) => {
                        let expert_biases = self.biases.as_ref().map(|biases| biases.index(expert));
                        let expert_scales = scales.index(expert);
                        let bits = derived_quant_bits(
                            &weight,
                            &expert_scales,
                            self.group_size,
                            &self.mode,
                        )
                        .unwrap_or(self.bits);
                        quantized_matmul_mode(
                            x,
                            &weight,
                            &expert_scales,
                            expert_biases.as_ref(),
                            true,
                            self.group_size,
                            bits,
                            &self.mode,
                        )
                    }
                    _ => matmul(x, &weight.t()).map_err(Into::into),
                }
            }
        }

        /// Batched forward over all routed experts at once. `rhs_indices` selects the expert
        /// weight for each output position (see `gather_qmm_mode`).
        ///
        /// Streaming path: when this `SwitchLinear` is backed by an on-demand pool, the resident
        /// `self.weight`/`self.scales` are 0-element placeholders, so `gather_qmm_mode` cannot run.
        /// Instead we decompose the batched gather into per-expert `forward_expert` calls (which
        /// fetch slabs from the pool) and scatter the results into the same `[.., top_k, 1, d]`
        /// layout the resident path produces. This is correct for any batch size / top_k and reuses
        /// the already-verified streaming forward path.
        /// Whether this projection can serve batched gathered dispatch (resident quantized
        /// weights or a streaming pool).
        fn supports_gather(&self) -> bool {
            self.scales.is_some() || self.stream.is_some()
        }

        /// Batched gathered dispatch through whichever backing store this projection has.
        fn gather_auto(&self, x: &Array, rhs_indices: &Array) -> Result<Array> {
            if self.stream.is_some() {
                self.gather_streaming(x, rhs_indices)
            } else {
                self.gather(x, rhs_indices)
            }
        }

        fn gather(&self, x: &Array, rhs_indices: &Array) -> Result<Array> {
            if self.stream.is_some() {
                return self.gather_streaming(x, rhs_indices);
            }
            match &self.scales {
                Some(scales) => gather_qmm_mode(
                    x,
                    &self.weight,
                    scales,
                    self.biases.as_ref(),
                    rhs_indices,
                    true,
                    self.group_size,
                    derived_quant_bits(&self.weight, scales, self.group_size, &self.mode)
                        .unwrap_or(self.bits),
                    &self.mode,
                ),
                None => bail!("hi-mlx batched MoE requires quantized expert weights"),
            }
        }

        /// Streaming decomposition of `gather`: for each batch position and each of its top_k
        /// routed experts, run `forward_expert` (pool-backed) and stack the per-expert outputs
        /// along the top_k axis to match the resident `gather_qmm_mode` output shape.
        fn gather_streaming(&self, x: &Array, rhs_indices: &Array) -> Result<Array> {
            let x_shape = x.shape();
            let idx_shape = rhs_indices.shape();
            // `x` is `[.., 1, 1, d]`; collapse the leading dims to a single batch axis.
            // `rhs_indices` is `[.., top_k]` with the same leading dims.
            let d = *x_shape.last().expect("gather x must have a trailing d dim");
            // The batch (token) count is the product of all leading dims of rhs_indices
            // except the final top_k axis.
            let top_k = *idx_shape
                .last()
                .expect("rhs_indices must have a trailing top_k dim");
            let batch: i32 = if idx_shape.len() <= 1 {
                1
            } else {
                idx_shape[..idx_shape.len() - 1].iter().product::<i32>()
            };
            // Routing indices are Int32 on most archs but Uint32 on some (MiniMax-M3), and
            // `as_slice` panics on a dtype mismatch rather than converting. Normalize first so the
            // streaming path does not abort mid-generation on an otherwise fine model.
            let rhs_indices_i32 = rhs_indices.as_type::<i32>()?;
            let idx_slice = rhs_indices_i32.as_slice::<i32>();
            // `x` arrives in one of two layouts. gate/up get `[.., 1, 1, d]` — one vector per token,
            // shared by all top_k experts — while down_proj is fed the gate*up product, which is
            // already `[.., top_k, 1, d]`, one vector per (token, expert). Assuming the first shape
            // made down_proj fail with "Cannot reshape array of size 344064 into shape (28,3072)",
            // off by exactly top_k. Flatten to rows and pick the row that matches the layout.
            let total: i32 = x_shape.iter().product();
            let rows = total / d;
            let per_expert_input = rows == batch.saturating_mul(top_k);
            let x_flat = x.reshape(&[rows, d])?;
            let mut per_token: Vec<Array> = Vec::with_capacity(batch as usize);
            for t in 0..batch {
                let mut per_expert: Vec<Array> = Vec::with_capacity(top_k as usize);
                for k in 0..top_k {
                    let row = if per_expert_input { t * top_k + k } else { t };
                    let token_x = x_flat.index((row, ..)).reshape(&[1, 1, d])?;
                    let expert = idx_slice[(t * top_k + k) as usize] as i32;
                    per_expert.push(self.forward_expert(&token_x, expert)?);
                }
                // Stack along axis 0 → `[top_k, 1, out]`.
                per_token.push(concatenate_axis(&per_expert, 0)?);
            }
            // Stack along axis 0 → `[batch, top_k, 1, out]`, then reshape to restore the
            // original leading dims (matching the resident `gather_qmm_mode` output layout).
            //
            // The trailing dim is the projection's *output* width, which is not the input width
            // `d`: gate/up are hidden->intermediate and down is intermediate->hidden. Reusing `d`
            // here only held for square projections and blew up everywhere else — MiniMax-M3
            // (6144 -> 3072) failed with "Cannot reshape array of size 294912 into shape
            // (24,4,1,6144)". Take it from the array we actually produced.
            let out = concatenate_axis(&per_token, 0)?;
            let out_dim = *out
                .shape()
                .last()
                .expect("expert output must have a trailing dim");
            let out_shape: Vec<i32> = idx_shape
                .iter()
                .copied()
                .chain([1, out_dim].iter().copied())
                .collect();
            out.reshape(&out_shape).map_err(Into::into)
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
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            Ok(Self {
                gate_proj: SwitchLinear::load_or_stream(
                    &format!("{prefix}.gate_proj"),
                    arrays,
                    config,
                    stream_ctx,
                    "gate_proj",
                )?,
                up_proj: SwitchLinear::load_or_stream(
                    &format!("{prefix}.up_proj"),
                    arrays,
                    config,
                    stream_ctx,
                    "up_proj",
                )?,
                down_proj: SwitchLinear::load_or_stream(
                    &format!("{prefix}.down_proj"),
                    arrays,
                    config,
                    stream_ctx,
                    "down_proj",
                )?,
            })
        }

        fn forward_expert(&self, x: &Array, expert: i32) -> Result<Array> {
            let gate_pre = self.gate_proj.forward_expert(x, expert)?;
            let gate = sigmoid(&gate_pre)? * gate_pre;
            let up = self.up_proj.forward_expert(x, expert)?;
            self.down_proj.forward_expert(&(gate * up), expert)
        }

        /// Prefetch all slabs for a set of experts in one batch. Called after
        /// the router selects the top-k experts, before the per-expert forward
        /// loop. This issues all reads (experts × 3 projections × weight/scales/
        /// biases) via a single `lio_listio` batch so the SSD services them
        /// concurrently. Subsequent `forward_expert` calls become cache hits.
        /// Part of the Inkling expert-prefetch path, not yet wired into serving.
        #[allow(dead_code)]
        fn prefetch_experts(&self, experts: &[i32]) -> Result<()> {
            self.prefetch_experts_impl(experts, true)
        }

        /// Async variant for cross-layer pipelining: submits the AIO batch
        /// with `LIO_NOWAIT` and returns immediately. The reads complete in
        /// the background while the previous layer's matmuls run. The next
        /// call to `prefetch_experts` (or `forward_expert`) will wait for
        /// these reads to finish.
        fn prefetch_experts_async(&self, experts: &[i32]) -> Result<()> {
            self.prefetch_experts_impl(experts, false)
        }

        fn prefetch_experts_impl(&self, experts: &[i32], wait: bool) -> Result<()> {
            // Collect all (layer, projection, expert, tensor_kind, tensor_name)
            // tuples across all 3 projections and all experts.
            let mut requests: Vec<(u32, &'static str, u32, &'static str, String)> = Vec::new();
            for proj in [&self.gate_proj, &self.up_proj, &self.down_proj] {
                if let Some(stream) = &proj.stream {
                    for &expert in experts {
                        // weight
                        requests.push((
                            stream.layer,
                            stream.projection,
                            expert as u32,
                            "weight",
                            stream.weight_name.clone(),
                        ));
                        // scales
                        if let Some(scales_name) = &stream.scales_name {
                            requests.push((
                                stream.layer,
                                stream.projection,
                                expert as u32,
                                "scales",
                                scales_name.clone(),
                            ));
                        }
                        // biases
                        if let Some(biases_name) = &stream.biases_name {
                            requests.push((
                                stream.layer,
                                stream.projection,
                                expert as u32,
                                "biases",
                                biases_name.clone(),
                            ));
                        }
                    }
                }
            }
            if requests.is_empty() {
                return Ok(());
            }
            // All projections share the same pool (it's an Arc<Mutex>).
            let pool = self.gate_proj.stream.as_ref().unwrap().pool.clone();
            let mut pool = pool.lock().unwrap();
            // Record expert selection for usage learning (pre-warm next time).
            let layer = self.gate_proj.stream.as_ref().unwrap().layer;
            pool.record_expert_usage(experts.iter().map(|&e| (layer, e as u32)));
            let result = if wait {
                pool.prefetch_batch(&requests)
            } else {
                pool.prefetch_batch_async(&requests)
            };
            // Persist usage history periodically (every call is fine — it's a
            // small JSON write, and the file is only created if usage is non-empty).
            pool.save_usage();
            result
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
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            Ok(Self {
                gate: MoEGate::load(&format!("{prefix}.gate"), arrays, config)?,
                switch_mlp: SwitchMlp::load(
                    &format!("{prefix}.switch_mlp"),
                    arrays,
                    config,
                    stream_ctx,
                )?,
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

            // ── Batch prefetch (async for cross-layer pipelining) ─────────
            // After routing, we know exactly which experts each token needs.
            // Submit all slab reads (experts × 3 projections × weight/scales/
            // biases) as one POSIX AIO batch with LIO_NOWAIT. The reads complete
            // in the background while we do the matmuls. The next layer's
            // prefetch_batch call (or get_array) will wait for these to finish.
            let all_experts: Vec<i32> = routes
                .iter()
                .flat_map(|token_routes| token_routes.iter().map(|(e, _)| *e))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            if !all_experts.is_empty() {
                self.switch_mlp.prefetch_experts_async(&all_experts)?;
            }

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
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let prefix = format!("model.layers.{layer_idx}.mlp");
            if config.is_moe_layer(layer_idx) {
                Ok(Self::Moe(MoE::load(&prefix, arrays, config, stream_ctx)?))
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
            stream_ctx: Option<&StreamContext>,
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
                ffn: MlaFfn::load(idx, arrays, config, stream_ctx)?,
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
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let p = format!("model.layers.{idx}");
            Ok(Self {
                eh_proj: Linear::load(&format!("{p}.eh_proj"), arrays, config)?,
                enorm: RmsNorm::load(&format!("{p}.enorm.weight"), arrays, config.rms_norm_eps)?,
                hnorm: RmsNorm::load(&format!("{p}.hnorm.weight"), arrays, config.rms_norm_eps)?,
                block: MlaBlock::load(idx, arrays, config, stream_ctx)?,
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
        fn new(
            config: MlxModelConfig,
            mut arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            prepare_mla_weights(&config, &mut arrays)?;
            let layers = (0..config.num_hidden_layers)
                .map(|idx| MlaBlock::load(idx, &arrays, &config, stream_ctx))
                .collect::<Result<Vec<_>>>()?;
            // The MTP head is the first "next-n" layer (index num_hidden_layers). Load it if present.
            let mtp = if config.num_nextn_predict_layers.unwrap_or(0) >= 1
                && arrays.contains_key(&format!(
                    "model.layers.{}.eh_proj.weight",
                    config.num_hidden_layers
                )) {
                Some(MtpHead::load(
                    config.num_hidden_layers,
                    &arrays,
                    &config,
                    stream_ctx,
                )?)
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
                    on_event(GenerationEvent::TokenDelta {
                        token_id: tok,
                        text: delta,
                    })?;
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
            on_event(GenerationEvent::Finished {
                output: output.clone(),
            })?;
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
        /// Stacked per-row state for batched decode; None during single-sequence serving.
        batch: Option<V4BatchState>,
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
                batch: None,
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

    /// One row's exact post-prefill attention state, captured from a b=1 prefill. Compression
    /// blocks, sliding-window contents and RoPE rotations are all anchored at position 0, so a
    /// per-row prefill is exact by construction — the batched machinery only preserves it.
    struct V4RowSnapshot {
        raw_k: Array,
        raw_v: Array,
        len: i32,
        comp_k: Option<Array>,
        comp_v: Option<Array>,
        comp_pending: Option<Array>,
        idx_comp_k: Option<Array>,
        idx_pending: Option<Array>,
    }

    /// Batched-decode state for one V4Attention layer.
    ///
    /// Every row's stored sliding window is stacked RIGHT-ALIGNED, so row `i` occupies raw
    /// columns `[w0 - stored_i, w0)` and decode steps append at the shared column `w0 + t`.
    /// Column `c` then has logical position `c + (len_i - w0)` for every row — one scalar shift
    /// per row — so query-key distances are column differences shared across rows, and the only
    /// truly per-row quantities are RoPE offsets, dead leading columns, and compressed-block
    /// visibility. Compressed blocks stack right-aligned at `cb0` the same way; a row's block
    /// `j` sits at column `cb0 - n0 + j` while `j < n0` and at `cb0 + (j - n0)` afterwards.
    struct V4BatchState {
        key: Array,
        value: Array,
        row_lens: Vec<i32>,
        start_cols: Vec<i32>,
        w0: i32,
        steps: i32,
        cap: i32,
        comp_key: Option<Array>,
        comp_value: Option<Array>,
        /// blocks stacked at prefill time (fixed) and total blocks now, per row.
        comp_counts0: Vec<i32>,
        comp_counts: Vec<i32>,
        cb0: i32,
        comp_cap: i32,
        pending: Vec<Option<Array>>,
        /// Sparse-indexer state. The indexer's compressor shares ratio and input stream with
        /// the main compressor, so their block spaces are identical: block j of row i lives at
        /// the same column in both stacked caches, and per-row top-k selection is expressed as
        /// extra -inf entries in the main compressed bias — the same softmax support set as
        /// single-mode's gather, hence mathematically identical attention.
        idx_comp_key: Option<Array>,
        idx_comp_counts0: Vec<i32>,
        idx_comp_counts: Vec<i32>,
        idx_cb0: i32,
        idx_comp_cap: i32,
        idx_pending: Vec<Option<Array>>,
    }

    impl V4Attention {
        /// Move the current single-sequence caches out as a row snapshot (and reset them),
        /// ready for [`Self::stack_rows`]. Call once per row, straight after its prefill.
        fn snapshot_row(&mut self, prompt_len: i32) -> Result<V4RowSnapshot> {
            let raw_k = self
                .cache
                .materialized_key()?
                .ok_or_else(|| anyhow!("V4 batch snapshot: empty raw cache"))?;
            let raw_v = self
                .cache
                .materialized_value()?
                .ok_or_else(|| anyhow!("V4 batch snapshot: empty raw cache"))?;
            let (comp_k, comp_v, comp_pending) = match self.compressor.as_ref() {
                Some(compressor) => {
                    let kv = compressor.cached()?;
                    (
                        kv.as_ref().map(|(k, _)| k.clone()),
                        kv.map(|(_, v)| v),
                        compressor.pending.clone(),
                    )
                }
                None => (None, None, None),
            };
            let (idx_comp_k, idx_pending) = match self.indexer.as_ref() {
                Some(indexer) => (
                    indexer.compressor.cached()?.map(|(k, _)| k),
                    indexer.compressor.pending.clone(),
                ),
                None => (None, None),
            };
            self.reset_cache();
            Ok(V4RowSnapshot {
                raw_k,
                raw_v,
                len: prompt_len,
                comp_k,
                comp_v,
                comp_pending,
                idx_comp_k,
                idx_pending,
            })
        }

        /// Stack per-row snapshots into the shared batched-decode buffers.
        fn stack_rows(&mut self, rows: Vec<V4RowSnapshot>, max_steps: i32) -> Result<()> {
            let b = rows.len() as i32;
            let d = self.head_dim;
            let dt = rows[0].raw_k.dtype();
            let stored: Vec<i32> = rows.iter().map(|r| r.raw_k.shape()[2]).collect();
            let w0 = stored.iter().copied().max().unwrap_or(0);
            let cap = w0 + max_steps.max(1);
            let mut key = zeros_dtype(&[b, 1, cap, d], dt)?;
            let mut value = zeros_dtype(&[b, 1, cap, d], dt)?;
            for (i, row) in rows.iter().enumerate() {
                let (i, s) = (i as i32, stored[i]);
                key.try_index_mut((i..i + 1, .., (w0 - s)..w0, ..), &row.raw_k)?;
                value.try_index_mut((i..i + 1, .., (w0 - s)..w0, ..), &row.raw_v)?;
            }

            let comp_counts0: Vec<i32> = rows
                .iter()
                .map(|r| r.comp_k.as_ref().map_or(0, |k| k.shape()[2]))
                .collect();
            let cb0 = comp_counts0.iter().copied().max().unwrap_or(0);
            let (comp_key, comp_value, comp_cap) = if self.compressor.is_some() {
                let ratio = self.compress_ratio.max(1);
                let comp_cap = cb0 + max_steps / ratio + 2;
                let mut ck = zeros_dtype(&[b, 1, comp_cap, d], dt)?;
                let mut cv = zeros_dtype(&[b, 1, comp_cap, d], dt)?;
                for (i, row) in rows.iter().enumerate() {
                    if let (Some(k), Some(v)) = (&row.comp_k, &row.comp_v) {
                        let (i, n) = (i as i32, k.shape()[2]);
                        ck.try_index_mut((i..i + 1, .., (cb0 - n)..cb0, ..), k)?;
                        cv.try_index_mut((i..i + 1, .., (cb0 - n)..cb0, ..), v)?;
                    }
                }
                (Some(ck), Some(cv), comp_cap)
            } else {
                (None, None, 0)
            };

            let idx_comp_counts0: Vec<i32> = rows
                .iter()
                .map(|r| r.idx_comp_k.as_ref().map_or(0, |k| k.shape()[2]))
                .collect();
            let idx_cb0 = idx_comp_counts0.iter().copied().max().unwrap_or(0);
            let (idx_comp_key, idx_comp_cap) = if let Some(indexer) = self.indexer.as_ref() {
                let ratio = self.compress_ratio.max(1);
                let idx_comp_cap = idx_cb0 + max_steps / ratio + 2;
                let idx_d = indexer.head_dim;
                let mut ck = zeros_dtype(&[b, 1, idx_comp_cap, idx_d], dt)?;
                for (i, row) in rows.iter().enumerate() {
                    if let Some(k) = &row.idx_comp_k {
                        let (i, n) = (i as i32, k.shape()[2]);
                        ck.try_index_mut((i..i + 1, .., (idx_cb0 - n)..idx_cb0, ..), k)?;
                    }
                }
                (Some(ck), idx_comp_cap)
            } else {
                (None, 0)
            };

            self.batch = Some(V4BatchState {
                key,
                value,
                row_lens: rows.iter().map(|r| r.len).collect(),
                start_cols: stored.iter().map(|s| w0 - s).collect(),
                w0,
                steps: 0,
                cap,
                comp_key,
                comp_value,
                comp_counts: comp_counts0.clone(),
                comp_counts0,
                cb0,
                comp_cap,
                pending: rows.iter().map(|r| r.comp_pending.clone()).collect(),
                idx_comp_key,
                idx_comp_counts0: idx_comp_counts0.clone(),
                idx_comp_counts: idx_comp_counts0,
                idx_cb0,
                idx_comp_cap,
                idx_pending: rows.iter().map(|r| r.idx_pending.clone()).collect(),
            });
            Ok(())
        }

        fn clear_batch(&mut self) {
            self.batch = None;
        }

        /// One batched decode step: `x` is `[b, 1, hidden]` — one new token per row, all rows
        /// at the same physical column but each at its own logical position.
        fn forward_batch_step(&mut self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let b = shape[0];
            // Scalars out first; tensor mutations re-borrow in short scopes below.
            let (t, w0, cap, cb0, comp_cap, row_lens, start_cols) = {
                let state = self
                    .batch
                    .as_ref()
                    .ok_or_else(|| anyhow!("forward_batch_step without stacked batch state"))?;
                if state.row_lens.len() as i32 != b {
                    bail!(
                        "batch width changed: stacked {} rows, got {b}",
                        state.row_lens.len()
                    );
                }
                (
                    state.steps,
                    state.w0,
                    state.cap,
                    state.cb0,
                    state.comp_cap,
                    state.row_lens.clone(),
                    state.start_cols.clone(),
                )
            };
            let offsets: Vec<i32> = row_lens.iter().map(|&l| l + t).collect();

            let qr = self.q_norm.forward(&self.wq_a.forward(x)?)?;
            let mut q = self
                .wq_b
                .forward(&qr)?
                .reshape(&[b, 1, self.num_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            q = q.clone() * rsqrt(&(mean_axis(&(q.clone() * &q), -1, Some(true))? + self.eps))?;
            let mut kv = self
                .kv_norm
                .forward(&self.wkv.forward(x)?)?
                .reshape(&[b, 1, 1, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            if self.rope_head_dim > 0 {
                let mut q_parts = split_sections(&q, &[self.nope_head_dim], -1)?;
                let q_nope = q_parts.remove(0);
                let q_pe = v4_rope_rows(
                    &q_parts.remove(0),
                    self.rope_head_dim,
                    self.rope_theta,
                    &offsets,
                    false,
                )?;
                q = concatenate_axis(&[q_nope, q_pe], -1)?;
                let mut k_parts = split_sections(&kv, &[self.nope_head_dim], -1)?;
                let k_nope = k_parts.remove(0);
                let k_pe = v4_rope_rows(
                    &k_parts.remove(0),
                    self.rope_head_dim,
                    self.rope_theta,
                    &offsets,
                    false,
                )?;
                kv = concatenate_axis(&[k_nope, k_pe], -1)?;
            }

            // Append this step's latent at the shared column w0 + t (k and v share the latent,
            // exactly as the single path passes the same tensor to update_with_start twice).
            let col = w0 + t;
            if col >= cap {
                bail!("V4 batch decode overran raw cache capacity ({cap})");
            }
            let (raw_k, raw_v) = {
                let state = self.batch.as_mut().unwrap();
                state.key.try_index_mut((.., .., col..col + 1, ..), &kv)?;
                state.value.try_index_mut((.., .., col..col + 1, ..), &kv)?;
                (
                    state.key.index((.., .., ..col + 1, ..)),
                    state.value.index((.., .., ..col + 1, ..)),
                )
            };
            let kv_len = col + 1;

            // Raw-side additive bias [b, 1, 1, kv_len]: dead leading columns per row, plus the
            // sliding window (column distance == logical distance, the shifts cancel).
            let window = self.cache.max_len;
            let mut raw_bias = vec![0.0f32; (b * kv_len) as usize];
            for i in 0..b {
                let base = (i * kv_len) as usize;
                let start = start_cols[i as usize];
                let min_col = match window {
                    Some(w) => (col - (w - 1)).max(start),
                    None => start,
                };
                for c in 0..min_col {
                    raw_bias[base + c as usize] = f32::NEG_INFINITY;
                }
            }
            let raw_bias = Array::from_slice(&raw_bias, &[b, 1, 1, kv_len]).as_dtype(q.dtype())?;

            // Compressed side: append the raw hidden token to each row's pending buffer and
            // emit that row's next block summary when `ratio` tokens have accumulated. The new
            // block lands at column cb0 + appended, because every row's stacked region ends at
            // cb0.
            if self.compressor.is_some() {
                let ratio = self.compress_ratio.max(1);
                for i in 0..b as usize {
                    let xi = x.index((i as i32..i as i32 + 1, .., ..));
                    let pending = {
                        let state = self.batch.as_mut().unwrap();
                        match state.pending[i].take() {
                            Some(p) => concatenate_axis(&[p, xi], 1)?,
                            None => xi,
                        }
                    };
                    if pending.shape()[1] >= ratio {
                        let block_in = pending.index((.., ..ratio, ..));
                        let (bk, bv) = self
                            .compressor
                            .as_ref()
                            .unwrap()
                            .compress_complete(&block_in)?;
                        let state = self.batch.as_mut().unwrap();
                        let appended = state.comp_counts[i] - state.comp_counts0[i];
                        let dest = cb0 + appended;
                        if dest >= comp_cap {
                            bail!("V4 batch decode overran compressed capacity ({comp_cap})");
                        }
                        let row = i as i32;
                        state
                            .comp_key
                            .as_mut()
                            .unwrap()
                            .try_index_mut((row..row + 1, .., dest..dest + 1, ..), &bk)?;
                        state
                            .comp_value
                            .as_mut()
                            .unwrap()
                            .try_index_mut((row..row + 1, .., dest..dest + 1, ..), &bv)?;
                        state.comp_counts[i] += 1;
                        let rest = pending.index((.., ratio.., ..));
                        state.pending[i] = if rest.shape()[1] > 0 { Some(rest) } else { None };
                    } else {
                        let state = self.batch.as_mut().unwrap();
                        state.pending[i] = Some(pending);
                    }
                }
            }

            // Indexer compressor: same per-row pending/emit dance as the main compressor,
            // writing into the indexer's stacked key lanes (only K is used for scores).
            if self.indexer.is_some() {
                let ratio = self.compress_ratio.max(1);
                let idx_cb0 = self.batch.as_ref().unwrap().idx_cb0;
                let idx_cap = self.batch.as_ref().unwrap().idx_comp_cap;
                for i in 0..b as usize {
                    let xi = x.index((i as i32..i as i32 + 1, .., ..));
                    let pending = {
                        let state = self.batch.as_mut().unwrap();
                        match state.idx_pending[i].take() {
                            Some(p) => concatenate_axis(&[p, xi], 1)?,
                            None => xi,
                        }
                    };
                    if pending.shape()[1] >= ratio {
                        let block_in = pending.index((.., ..ratio, ..));
                        let (bk, _bv) = self
                            .indexer
                            .as_ref()
                            .unwrap()
                            .compressor
                            .compress_complete(&block_in)?;
                        let state = self.batch.as_mut().unwrap();
                        let appended = state.idx_comp_counts[i] - state.idx_comp_counts0[i];
                        let dest = idx_cb0 + appended;
                        if dest >= idx_cap {
                            bail!("V4 batch decode overran indexer compressed capacity");
                        }
                        let row = i as i32;
                        state
                            .idx_comp_key
                            .as_mut()
                            .unwrap()
                            .try_index_mut((row..row + 1, .., dest..dest + 1, ..), &bk)?;
                        state.idx_comp_counts[i] += 1;
                        let rest = pending.index((.., ratio.., ..));
                        state.idx_pending[i] =
                            if rest.shape()[1] > 0 { Some(rest) } else { None };
                    } else {
                        let state = self.batch.as_mut().unwrap();
                        state.idx_pending[i] = Some(pending);
                    }
                }
            }

            // Assemble [compressed | raw] exactly like the single path's combined_kv_and_mask,
            // with per-row block visibility: block j visible iff (j+1)*ratio - 1 <= position.
            let (k, v, bias) = {
                let state = self.batch.as_ref().unwrap();
                if let Some(ck_full) = state.comp_key.as_ref() {
                    let ratio = self.compress_ratio.max(1);
                    let clen = state
                        .comp_counts
                        .iter()
                        .zip(&state.comp_counts0)
                        .map(|(&n, &n0)| cb0 + (n - n0))
                        .max()
                        .unwrap_or(cb0)
                        .max(1);
                    let ck = ck_full.index((.., .., ..clen, ..));
                    let cv = state
                        .comp_value
                        .as_ref()
                        .unwrap()
                        .index((.., .., ..clen, ..));
                    let mut cbias = vec![f32::NEG_INFINITY; (b * clen) as usize];
                    for i in 0..b as usize {
                        let (n0, total) = (state.comp_counts0[i], state.comp_counts[i]);
                        let base = i * clen as usize;
                        let pos = row_lens[i] + t;
                        for j in 0..total {
                            let cc = if j < n0 { cb0 - n0 + j } else { cb0 + (j - n0) };
                            if cc < clen && (j + 1) * ratio - 1 <= pos {
                                cbias[base + cc as usize] = 0.0;
                            }
                        }
                    }
                    // Sparse indexer: rows whose block count exceeds index_topk keep only
                    // their top-k blocks. Scores are computed with the single path's exact
                    // formula (relu(q·ckᵀ) head-summed with weights_proj), invalid columns
                    // pre-masked, top-k taken per row on CPU; unselected blocks turn -inf.
                    // Same softmax support set as single-mode's take_along_axis gather.
                    if let Some(indexer) = self.indexer.as_ref() {
                        let fires = state
                            .idx_comp_counts
                            .iter()
                            .any(|&n| n > indexer.index_topk);
                        if fires {
                            let ick_full = state
                                .idx_comp_key
                                .as_ref()
                                .ok_or_else(|| anyhow!("indexer fired without stacked keys"))?;
                            let ick = ick_full.index((.., .., ..clen, ..));
                            let iq = indexer
                                .wq_b
                                .forward(&qr)?
                                .reshape(&[b, 1, indexer.n_heads, indexer.head_dim])?
                                .swap_axes(1, 2)?;
                            let mut scores =
                                matmul(&(iq * indexer.scale), &ick.swap_axes(-1, -2)?)?;
                            scores = maximum(&scores, &Array::from_f32(0.0))?;
                            let iw = indexer.weights_proj.forward(x)?
                                * (indexer.n_heads as f32).powf(-0.5);
                            let iw = iw.swap_axes(-1, -2)?.expand_dims(-1)?;
                            let scores = sum_axis(&(scores * iw), 1, Some(true))?
                                .as_type::<f32>()?;
                            transforms::eval([&scores])?;
                            let flat = scores.as_slice::<f32>().to_vec();
                            for i in 0..b as usize {
                                let n = state.idx_comp_counts[i];
                                if n <= indexer.index_topk {
                                    continue;
                                }
                                let base = i * clen as usize;
                                // candidate columns = this row's visible blocks (bias 0.0)
                                let mut cand: Vec<(usize, f32)> = (0..clen as usize)
                                    .filter(|&cc| cbias[base + cc] == 0.0)
                                    .map(|cc| (cc, flat[base + cc]))
                                    .collect();
                                if cand.len() as i32 <= indexer.index_topk {
                                    continue;
                                }
                                cand.sort_by(|a, b| {
                                    b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0))
                                });
                                for &(cc, _) in cand.iter().skip(indexer.index_topk as usize) {
                                    cbias[base + cc] = f32::NEG_INFINITY;
                                }
                            }
                        }
                    }
                    let cbias =
                        Array::from_slice(&cbias, &[b, 1, 1, clen]).as_dtype(q.dtype())?;
                    (
                        concatenate_axis(&[ck, raw_k], 2)?,
                        concatenate_axis(&[cv, raw_v], 2)?,
                        concatenate_axis(&[cbias, raw_bias], -1)?,
                    )
                } else {
                    (raw_k, raw_v, raw_bias)
                }
            };

            let mut out = scaled_dot_product_attention(
                &q,
                &k,
                &v,
                self.scale,
                ScaledDotProductAttentionMask::Array(&bias),
                self.attn_sink.as_ref(),
            )?;
            if self.rope_head_dim > 0 {
                let mut out_parts = split_sections(&out, &[self.nope_head_dim], -1)?;
                let out_nope = out_parts.remove(0);
                let out_pe = v4_rope_rows(
                    &out_parts.remove(0),
                    self.rope_head_dim,
                    self.rope_theta,
                    &offsets,
                    true,
                )?;
                out = concatenate_axis(&[out_nope, out_pe], -1)?;
            }
            let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                1,
                self.num_heads * self.head_dim,
            ])?;
            self.batch.as_mut().unwrap().steps += 1;
            self.wo_b.forward(&self.wo_a.forward(&out)?)
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

            // Row-major over the whole batch: token t of row r sits at flat index r*s + t in
            // both the flattened logits and `input_ids` (batched callers pass ids row-major;
            // at b == 1 this is exactly the old single-sequence behaviour).
            let mut routes = Vec::with_capacity((b * s) as usize);
            for token in 0..(b * s) as usize {
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
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            Ok(Self {
                gate: V4MoEGate::load(&format!("{prefix}.gate"), layer_idx, arrays, config)?,
                switch_mlp: SwitchMlp::load(
                    &format!("{prefix}.switch_mlp"),
                    arrays,
                    config,
                    stream_ctx,
                )?,
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
            let routes = self.gate.route(x, input_ids)?;
            let k = routes.first().map_or(0, Vec::len);
            let uniform = k > 0 && routes.iter().all(|r| r.len() == k);
            let gatherable = self.switch_mlp.gate_proj.supports_gather()
                && self.switch_mlp.up_proj.supports_gather()
                && self.switch_mlp.down_proj.supports_gather();
            let mut y = if uniform && gatherable {
                // Grouped dispatch: one gathered matmul per projection for all b*s tokens and
                // their top-k experts, instead of b*s*k tiny per-expert matmul chains — this is
                // where batched MoE decode actually amortizes expert weight traffic. Same math
                // as forward_expert_limited (clamped SwiGLU), verified against the reference
                // path by `v4_moe_grouped_matches_reference`.
                let n = b * s;
                let mut inds = Vec::with_capacity((n as usize) * k);
                let mut wts = Vec::with_capacity((n as usize) * k);
                for route in &routes {
                    for &(expert, weight) in route {
                        inds.push(expert);
                        wts.push(weight);
                    }
                }
                let inds = Array::from_slice(&inds, &[n, k as i32]);
                let xe = x.reshape(&[n, 1, 1, d])?;
                let mut gate_pre = self.switch_mlp.gate_proj.gather_auto(&xe, &inds)?;
                let mut up_pre = self.switch_mlp.up_proj.gather_auto(&xe, &inds)?;
                if self.swiglu_limit > 0.0 {
                    let ceiling = Array::from_f32(self.swiglu_limit);
                    let floor = Array::from_f32(-self.swiglu_limit);
                    gate_pre = minimum(&gate_pre, &ceiling)?;
                    up_pre = maximum(&minimum(&up_pre, &ceiling)?, &floor)?;
                }
                let gate = sigmoid(&gate_pre)? * gate_pre;
                let down = self
                    .switch_mlp
                    .down_proj
                    .gather_auto(&(gate * up_pre), &inds)?;
                let eo = down.reshape(&[n, k as i32, d])?.as_type::<f32>()?;
                let w = Array::from_slice(&wts, &[n, k as i32, 1]);
                sum_axis(&(eo * w), 1, Some(false))?.reshape(&[b, s, d])?
            } else {
                self.forward_reference(x, &routes)?
            };
            if let Some(shared) = &self.shared_experts {
                y = y + shared.forward(x)?;
            }
            Ok(y)
        }

        /// Per-token per-expert loop: the numeric ground truth for
        /// `v4_moe_grouped_matches_reference`, and the fallback for dense (unquantized,
        /// non-streaming) experts or non-uniform routes.
        fn forward_reference(&self, x: &Array, routes: &[Vec<(i32, f32)>]) -> Result<Array> {
            let shape = x.shape();
            let (b, s, d) = (shape[0], shape[1], shape[2]);
            let mut rows = Vec::with_capacity(b as usize);
            for row in 0..b {
                let mut outputs = Vec::with_capacity(s as usize);
                for token_idx in 0..s {
                    let token = x.index((row, token_idx, ..)).reshape(&[1, 1, d])?;
                    let mut acc = Array::zeros::<f32>(&[1, 1, d])?;
                    for (expert, score) in &routes[(row * s + token_idx) as usize] {
                        acc = acc
                            + self.switch_mlp.forward_expert_limited(
                                &token,
                                *expert,
                                self.swiglu_limit,
                            )? * *score;
                    }
                    outputs.push(acc);
                }
                rows.push(concatenate_axis(&outputs, 1)?);
            }
            Ok(concatenate_axis(&rows, 0)?)
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
            stream_ctx: Option<&StreamContext>,
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
                ffn: V4MoE::load(&format!("{prefix}.ffn"), idx, arrays, config, stream_ctx)?,
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

        /// Batched decode step: identical residual structure, with the attention routed through
        /// the stacked per-row state. `input_ids` is the whole batch row-major (`b` ids at s=1),
        /// which is what the widened V4 MoE gate expects for hash-routed layers.
        fn forward_batch_step(&mut self, h: Array, input_ids: &[u32]) -> Result<Array> {
            let residual = h.clone();
            let (y, post, comb) = self.hc_attn.pre(&h)?;
            let y = self
                .attention
                .forward_batch_step(&self.attn_norm.forward(&y)?)?;
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
        fn new(
            config: MlxModelConfig,
            arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let layers = (0..config.num_hidden_layers)
                .map(|idx| V4Block::load(idx, &arrays, &config, stream_ctx))
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
        fn supports_ragged_batch(&self) -> bool {
            true
        }

        fn prefill_batch_ragged(&mut self, prompts: &[Vec<u32>], max_steps: i32) -> Result<Array> {
            if prompts.is_empty() {
                bail!("prefill_batch_ragged: empty batch");
            }
            let mut last = Vec::with_capacity(prompts.len());
            let mut snaps: Vec<Vec<V4RowSnapshot>> =
                (0..self.layers.len()).map(|_| Vec::new()).collect();
            for prompt in prompts {
                self.reset_cache();
                let lg = self.forward(prompt)?;
                let sh = lg.shape().to_vec();
                let (s, _vocab) = (sh[sh.len() - 2], sh[sh.len() - 1]);
                last.push(lg.index((.., (s - 1)..s, ..)));
                for (li, layer) in self.layers.iter_mut().enumerate() {
                    snaps[li].push(layer.attention.snapshot_row(prompt.len() as i32)?);
                }
            }
            for (li, layer) in self.layers.iter_mut().enumerate() {
                layer
                    .attention
                    .stack_rows(std::mem::take(&mut snaps[li]), max_steps)?;
            }
            let logits = concatenate_axis(&last, 0)?;
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn forward_batch(&mut self, input_ids: &Array) -> Result<Array> {
            let shape = input_ids.shape();
            let (b, l) = (shape[0], shape[1]);
            if l != 1 {
                bail!("V4 batched decode steps one token per row; prompts go through prefill_batch_ragged");
            }
            let ids32 = input_ids.as_type::<i32>()?;
            transforms::eval([&ids32])?;
            let flat: Vec<u32> = ids32.as_slice::<i32>().iter().map(|&v| v as u32).collect();
            let h = self.embed_tokens.forward(input_ids)?;
            let sh = h.shape().to_vec();
            let mut h = broadcast_to(&h.expand_dims(2)?, &[b, l, self.hc_mult, sh[2]])?;
            for (idx, layer) in self.layers.iter_mut().enumerate() {
                h = layer.forward_batch_step(h, &flat)?;
                if idx % 2 == 1 {
                    transforms::eval([&h])?;
                }
            }
            let h = self.norm.forward(&self.hc_head.forward(&h)?)?;
            let logits = self.lm_head.forward(&h)?;
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let h = self.embed_tokens.forward(&ids)?;
            let shape = h.shape();
            let mut h = broadcast_to(
                &h.expand_dims(2)?,
                &[shape[0], shape[1], self.hc_mult, shape[2]],
            )?;
            for (idx, layer) in self.layers.iter_mut().enumerate() {
                h = layer.forward(h, input_ids)?;
                // Flush the command buffer every few layers. 43 layers of per-token expert
                // dispatch build one enormous lazy graph, and evaluating it in a single Metal
                // command buffer trips the GPU watchdog on the 300GB checkpoint
                // (kIOGPUCommandBufferCallbackErrorTimeout) — especially on the first, cold
                // forward while weight pages fault in.
                if idx % 2 == 1 {
                    transforms::eval([&h])?;
                }
            }
            let h = self.norm.forward(&self.hc_head.forward(&h)?)?;
            let logits = self.lm_head.forward(&h)?;
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn reset_cache(&mut self) {
            for layer in &mut self.layers {
                layer.attention.reset_cache();
                layer.attention.clear_batch();
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
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            // Usually under the router (`{prefix}.gate.e_score_correction_bias`); Laguna hangs it
            // off the MoE block itself.
            let expert_bias = match arrays
                .get(&format!("{prefix}.gate.e_score_correction_bias"))
                .or_else(|| arrays.get(&format!("{prefix}.e_score_correction_bias")))
            {
                Some(b) => {
                    let b = b.as_type::<f32>()?;
                    transforms::eval([&b])?;
                    Some(b.as_slice::<f32>().to_vec())
                }
                None => None,
            };
            let compile_moe = std::env::var_os("HI_MLX_COMPILE_MOE").is_some();
            // Experts are usually stacked at `{prefix}.switch_mlp`, but dots.llm1 ships them stacked at
            // `{prefix}.experts` — load from there directly (renaming would break mixed-quant per-tensor
            // spec lookup, which is keyed by the original weight prefix).
            let experts_prefix =
                if arrays.contains_key(&format!("{prefix}.switch_mlp.gate_proj.weight")) {
                    format!("{prefix}.switch_mlp")
                } else {
                    format!("{prefix}.experts")
                };
            Ok(Self {
                gate: Linear::load(&format!("{prefix}.gate"), arrays, config)?,
                switch_mlp: SwitchMlp::load(&experts_prefix, arrays, config, stream_ctx)?,
                shared_expert: if arrays
                    .contains_key(&format!("{prefix}.shared_expert.gate_proj.weight"))
                {
                    Some(Mlp::load(
                        &format!("{prefix}.shared_expert"),
                        arrays,
                        config,
                    )?)
                } else if arrays.contains_key(&format!("{prefix}.shared_experts.gate_proj.weight"))
                {
                    // ERNIE-4.5 names its shared expert `shared_experts` (plural).
                    Some(Mlp::load(
                        &format!("{prefix}.shared_experts"),
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
                sigmoid_routing: config.family == ModelFamily::Hy3
                    || config.family == ModelFamily::Laguna
                    || config.model_type == "dots1",
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
            // Routing is per token and independent of which sequence a token belongs to, and
            // `scores` is row-major [b, l, experts], so b*l tokens flatten in order: row 0's
            // tokens first, then row 1's. Callers index `routes[row * l + token]`.
            let tokens = (b * l) as usize;
            let raw_scores = scores.as_slice::<f32>();
            let experts = experts as usize;
            let mut routes = Vec::with_capacity(tokens);
            for token in 0..tokens {
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
            // Experts are selected and applied per token, so a batch is just a longer token list.
            // `n` is that flattened length; the result is reshaped back to [b, l, d] at the end.
            let n = b * l;
            let routes = self.route(x)?;
            let top_k = self.top_k as i32;

            // ── Batch prefetch (async for cross-layer pipelining) ──────────
            // After routing, submit all needed expert slab reads as one AIO
            // batch with LIO_NOWAIT. Reads complete in the background during
            // the matmuls; the next layer waits for them.
            let all_experts: Vec<i32> = routes
                .iter()
                .flat_map(|token_routes| token_routes.iter().map(|(e, _)| *e))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            if !all_experts.is_empty() {
                self.switch_mlp.prefetch_experts_async(&all_experts)?;
            }

            // Batched gather-qmm needs quantized expert weights; fall back to the per-expert loop
            // for dense (unquantized) experts.
            let mut y = if self.switch_mlp.gate_proj.scales.is_some() {
                let mut idx_v = Vec::with_capacity(n as usize * self.top_k);
                let mut wts_v = Vec::with_capacity(n as usize * self.top_k);
                for token in &routes {
                    for (expert, weight) in token {
                        idx_v.push(*expert as u32);
                        wts_v.push(*weight);
                    }
                }
                let inds = Array::from_slice(&idx_v, &[n, top_k]);
                let weights = Array::from_slice(&wts_v, &[n, top_k, 1]);
                let xe = x.reshape(&[n, 1, 1, d])?;
                let expert_out = self
                    .switch_mlp
                    .forward_batched(&xe, &inds)?
                    .reshape(&[n, top_k, d])?
                    .as_type::<f32>()?;
                sum_axis(&(expert_out * weights), 1, Some(false))?.reshape(&[b, l, d])?
            } else {
                let mut outputs = Vec::with_capacity((b * l) as usize);
                for row in 0..b {
                    for token_idx in 0..l {
                        let token = x.index((row, token_idx, ..)).reshape(&[1, 1, d])?;
                        let mut acc = Array::zeros::<f32>(&[1, 1, d])?;
                        for (expert, score) in &routes[(row * l + token_idx) as usize] {
                            acc = acc + self.switch_mlp.forward_expert(&token, *expert)? * *score;
                        }
                        outputs.push(acc);
                    }
                }
                concatenate_axis(&outputs, 1)?.reshape(&[b, l, d])?
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
            // The compiled MoE closure below is written for a single sequence. The eager path is
            // batch-aware, and it is the default anyway (see the note below), so send batches there
            // rather than refusing them.
            if x.shape()[0] != 1 {
                return self.forward_cpu(x);
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
                        weight,
                        scales,
                        biases,
                        ..
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
                sgw,
                sgs,
                sgb,
                suw,
                sus,
                sub,
                sdw,
                sds,
                sdb,
                hgw,
                hgs,
                hgb,
                huw,
                hus,
                hub,
                hdw,
                hds,
                hdb,
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
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let prefix = format!("model.layers.{layer_idx}.mlp");
            if config.is_qwen_moe_layer(layer_idx) {
                Ok(Self::Moe(QwenMoe::load(
                    &prefix, arrays, config, stream_ctx,
                )?))
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
        // Pre-norm: norm1 = input_layernorm (on x), norm2 = post_attention_layernorm (on mlp input).
        // Post-norm (OLMo2/EXAONE-4): norm1 = post_attention_layernorm (on attn output), norm2 =
        // post_feedforward_layernorm (on mlp output). Detected by the presence of the latter.
        norm1: RmsNorm,
        norm2: RmsNorm,
        attention: QwenAttention,
        ffn: QwenFfn,
        residual_multiplier: f32,
        post_norm: bool,
    }

    impl QwenBlock {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let prefix = format!("model.layers.{idx}");
            let post_norm =
                arrays.contains_key(&format!("{prefix}.post_feedforward_layernorm.weight"));
            let (n1, n2) = if post_norm {
                ("post_attention_layernorm", "post_feedforward_layernorm")
            } else {
                ("input_layernorm", "post_attention_layernorm")
            };
            Ok(Self {
                norm1: RmsNorm::load(
                    &format!("{prefix}.{n1}.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                norm2: RmsNorm::load(
                    &format!("{prefix}.{n2}.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                attention: QwenAttention::load(
                    &format!("{prefix}.self_attn"),
                    arrays,
                    config,
                    // SmolLM3 NoPE: no_rope_layers[idx] == 0 means skip rope on this layer.
                    config
                        .no_rope_layers
                        .get(idx as usize)
                        .map(|&v| v != 0)
                        .unwrap_or(true),
                )?,
                ffn: QwenFfn::load(idx, arrays, config, stream_ctx)?,
                residual_multiplier: config.residual_multiplier,
                post_norm,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let m = self.residual_multiplier;
            let add = |a: Array, b: Array| if m != 1.0 { a + b * m } else { a + b };
            if self.post_norm {
                // norm applied to the sublayer output, then added to the residual.
                let r = self.norm1.forward(&self.attention.forward(&x)?)?;
                let h = add(x, r);
                let r = self.norm2.forward(&self.ffn.forward(&h)?)?;
                Ok(add(h, r))
            } else {
                let r = self.attention.forward(&self.norm1.forward(&x)?)?;
                let h = add(x, r);
                let r = self.ffn.forward(&self.norm2.forward(&h)?)?;
                Ok(add(h, r))
            }
        }
    }

    // ---------------------- GPT-OSS (gpt_oss) ----------------------
    // GQA with attention sinks (per-head learned logit in the softmax denominator) + sliding/full
    // hybrid attention + YARN rope; MoE with a biased router, top-k-then-softmax routing, and
    // SwiGLU-OAI experts (separate gate/up/down with bias). Pre-norm RMSNorm, `model.` prefix.
    struct GptOssAttention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        sinks: Array,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        rope_theta: f32,
        rope_freqs: Option<Array>,
        cache: Cache,
    }

    impl GptOssAttention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            rope_freqs: Option<Array>,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: Linear::load(&format!("{prefix}.v_proj"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                sinks: raw_array(arrays, &format!("{prefix}.sinks"))?.as_type::<f32>()?,
                n_heads: config.num_attention_heads as i32,
                n_kv_heads: config.num_key_value_heads as i32,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
                rope_theta: config.rope_theta,
                rope_freqs,
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
            let (rbase, rfreqs) = match &self.rope_freqs {
                Some(f) => (None, Some(f)),
                None => (Some(self.rope_theta), None),
            };
            q = rope(q, self.head_dim, false, rbase, 1.0, offset, rfreqs)?;
            k = rope(k, self.head_dim, false, rbase, 1.0, offset, rfreqs)?;
            let (k, v) = self.cache.update(k, v)?;
            // Attention sinks: an extra per-head logit in the softmax denominator (mlx sdpa `sinks`).
            let q = q.as_type::<f32>()?;
            let k = k.as_type::<f32>()?;
            let v = v.as_type::<f32>()?;
            let mask = if l > 1 {
                Some(causal_attention_mask(l, k.shape()[2], offset))
            } else {
                None
            };
            let output = scaled_dot_product_attention(
                &q,
                &k,
                &v,
                self.scale,
                match &mask {
                    Some(m) => ScaledDotProductAttentionMask::Array(m),
                    None => ScaledDotProductAttentionMask::Causal,
                },
                Some(&self.sinks),
            )?;
            let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                l,
                self.n_heads * self.head_dim,
            ])?;
            self.o_proj.forward(&output)
        }
    }

    struct GptOssMoe {
        router: Linear,
        switch_mlp: SwitchMlp,
        gate_bias: Array,
        up_bias: Array,
        down_bias: Array,
        top_k: usize,
        alpha: f32,
        limit: f32,
    }

    impl GptOssMoe {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let ep = format!("{prefix}.experts");
            let gb = raw_array(arrays, &format!("{ep}.gate_proj.bias"))?.as_type::<f32>()?;
            let ub = raw_array(arrays, &format!("{ep}.up_proj.bias"))?.as_type::<f32>()?;
            let db = raw_array(arrays, &format!("{ep}.down_proj.bias"))?.as_type::<f32>()?;
            transforms::eval([&gb, &ub, &db])?;
            Ok(Self {
                router: Linear::load(&format!("{prefix}.router"), arrays, config)?,
                switch_mlp: SwitchMlp::load(&ep, arrays, config, stream_ctx)?,
                gate_bias: gb,
                up_bias: ub,
                down_bias: db,
                top_k: config.num_experts_per_tok.unwrap_or(1) as usize,
                alpha: 1.702,
                limit: config.swiglu_limit.unwrap_or(7.0),
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l, d) = (shape[0], shape[1], shape[2]);
            if b != 1 {
                bail!("GPT-OSS MoE supports batch size 1, got {b}");
            }
            let logits = self.router.forward(x)?.as_type::<f32>()?;
            transforms::eval([&logits])?;
            let experts = *logits.shape().last().unwrap() as usize;
            let raw = logits.as_slice::<f32>();
            let mut outputs = Vec::with_capacity(l as usize);
            for token in 0..l as usize {
                let lg = &raw[token * experts..token * experts + experts];
                // top-k by logit, then softmax over just those k.
                let mut ranked = (0..experts).map(|e| (e, lg[e])).collect::<Vec<_>>();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                ranked.truncate(self.top_k);
                let maxl = ranked.iter().map(|&(_, v)| v).fold(f32::MIN, f32::max);
                let exps: Vec<f32> = ranked.iter().map(|&(_, v)| (v - maxl).exp()).collect();
                let denom: f32 = exps.iter().sum::<f32>() + 1e-20;
                let token_x = x.index((0, token as i32, ..)).reshape(&[1, 1, d])?;
                let mut acc = Array::zeros::<f32>(&[1, 1, d])?;
                for (k, &(expert, _)) in ranked.iter().enumerate() {
                    let w = exps[k] / denom;
                    let e = expert as i32;
                    let gate = self.switch_mlp.gate_proj.forward_expert(&token_x, e)?
                        + self.gate_bias.index(e);
                    let up = self.switch_mlp.up_proj.forward_expert(&token_x, e)?
                        + self.up_bias.index(e);
                    let act = swiglu_oai(&gate, &up, self.alpha, self.limit)?;
                    let out = self.switch_mlp.down_proj.forward_expert(&act, e)?
                        + self.down_bias.index(e);
                    acc = acc + out.as_type::<f32>()? * w;
                }
                outputs.push(acc);
            }
            Ok(concatenate_axis(&outputs, 1)?)
        }
    }

    struct GptOssBlock {
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
        attention: GptOssAttention,
        moe: GptOssMoe,
    }

    impl GptOssBlock {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            rope_freqs: Option<Array>,
            stream_ctx: Option<&StreamContext>,
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
                attention: GptOssAttention::load(
                    &format!("{p}.self_attn"),
                    arrays,
                    config,
                    rope_freqs,
                )?,
                moe: GptOssMoe::load(&format!("{p}.mlp"), arrays, config, stream_ctx)?,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let h = &x + self.attention.forward(&self.input_layernorm.forward(&x)?)?;
            let r = self
                .moe
                .forward(&self.post_attention_layernorm.forward(&h)?)?;
            Ok(h + r)
        }
    }

    struct GptOssLike {
        embed_tokens: Embedding,
        layers: Vec<GptOssBlock>,
        norm: RmsNorm,
        lm_head: Option<Linear>,
    }

    impl GptOssLike {
        fn new(
            config: MlxModelConfig,
            arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let (rope_freqs, _) = longcat_yarn_rope(&config, config.attention_head_dim() as i32)?;
            let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
            for idx in 0..config.num_hidden_layers {
                layers.push(GptOssBlock::load(
                    idx,
                    &arrays,
                    &config,
                    rope_freqs.clone(),
                    stream_ctx,
                )?);
            }
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

    impl CausalLm for GptOssLike {
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

    // ---------------------- Phi-3.5-MoE (phimoe) ----------------------
    // GQA with biased q/k/v/o + SuScaledRoPE (LongRoPE: per-dim long_factor freqs + an mscale on q/k) +
    // LayerNorm (with bias) + top-k-then-softmax MoE (no shared expert). Untied lm_head. Pre-norm.
    fn phi_surope_freqs(head_dim: i32, base: f32, long_factor: &[f32]) -> Array {
        let half = (head_dim / 2) as usize;
        let thetas: Vec<f32> = (0..half)
            .map(|i| {
                let lf = long_factor.get(i).copied().unwrap_or(1.0);
                lf * base.powf((2 * i) as f32 / head_dim as f32)
            })
            .collect();
        Array::from_slice(&thetas, &[half as i32])
    }

    struct PhiMoeAttention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        mscale: f32,
        rope_freqs: Array,
        cache: Cache,
    }

    impl PhiMoeAttention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            rope_freqs: &Array,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            let mscale = config
                .rope_scaling
                .as_ref()
                .and_then(|s| s.get("long_mscale"))
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: Linear::load(&format!("{prefix}.v_proj"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                n_heads: config.num_attention_heads as i32,
                n_kv_heads: config.num_key_value_heads as i32,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
                mscale,
                rope_freqs: rope_freqs.clone(),
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
            // SuScaledRoPE: scale q/k by mscale, then rope with the long-factor freqs (base unset).
            if self.mscale != 1.0 {
                q = q * self.mscale;
                k = k * self.mscale;
            }
            let offset = self.cache.offset;
            q = rope(
                q,
                self.head_dim,
                false,
                None::<f32>,
                1.0,
                offset,
                Some(&self.rope_freqs),
            )?;
            k = rope(
                k,
                self.head_dim,
                false,
                None::<f32>,
                1.0,
                offset,
                Some(&self.rope_freqs),
            )?;
            let (k, v) = self.cache.update(k, v)?;
            let output = if l > 1 && offset == 0 {
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    ScaledDotProductAttentionMask::Causal,
                    None::<&Array>,
                )?
            } else if l > 1 {
                let mask = causal_attention_mask(l, k.shape()[2], offset);
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    ScaledDotProductAttentionMask::Array(&mask),
                    None::<&Array>,
                )?
            } else {
                scaled_dot_product_attention(&q, &k, &v, self.scale, None, None::<&Array>)?
            };
            let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                l,
                self.n_heads * self.head_dim,
            ])?;
            self.o_proj.forward(&output)
        }
    }

    struct PhiMoeMoe {
        gate: Linear,
        switch_mlp: SwitchMlp,
        top_k: usize,
    }

    impl PhiMoeMoe {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            Ok(Self {
                gate: Linear::load(&format!("{prefix}.gate"), arrays, config)?,
                switch_mlp: SwitchMlp::load(
                    &format!("{prefix}.switch_mlp"),
                    arrays,
                    config,
                    stream_ctx,
                )?,
                top_k: config.num_experts_per_tok.unwrap_or(2) as usize,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l, d) = (shape[0], shape[1], shape[2]);
            if b != 1 {
                bail!("Phi-MoE supports batch size 1, got {b}");
            }
            let logits = self.gate.forward(x)?.as_type::<f32>()?;
            transforms::eval([&logits])?;
            let n_exp = *logits.shape().last().unwrap() as usize;
            let raw = logits.as_slice::<f32>();
            let mut outputs = Vec::with_capacity(l as usize);
            for token in 0..l as usize {
                let lg = &raw[token * n_exp..token * n_exp + n_exp];
                // top-k by gate logit, then softmax over just those k.
                let mut ranked = (0..n_exp).map(|e| (e, lg[e])).collect::<Vec<_>>();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                ranked.truncate(self.top_k);
                let maxl = ranked.iter().map(|&(_, v)| v).fold(f32::MIN, f32::max);
                let exps: Vec<f32> = ranked.iter().map(|&(_, v)| (v - maxl).exp()).collect();
                let denom: f32 = exps.iter().sum::<f32>() + 1e-20;
                let token_x = x.index((0, token as i32, ..)).reshape(&[1, 1, d])?;
                let mut acc = Array::zeros::<f32>(&[1, 1, d])?;
                for (k, &(expert, _)) in ranked.iter().enumerate() {
                    let w = exps[k] / denom;
                    acc = acc
                        + self
                            .switch_mlp
                            .forward_expert(&token_x, expert as i32)?
                            .as_type::<f32>()?
                            * w;
                }
                outputs.push(acc);
            }
            Ok(concatenate_axis(&outputs, 1)?)
        }
    }

    struct PhiMoeBlock {
        input_layernorm: LayerNorm,
        post_attention_layernorm: LayerNorm,
        attention: PhiMoeAttention,
        moe: PhiMoeMoe,
    }

    impl PhiMoeBlock {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            rope_freqs: &Array,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let p = format!("model.layers.{idx}");
            let eps = config.rms_norm_eps;
            Ok(Self {
                input_layernorm: LayerNorm::load(&format!("{p}.input_layernorm"), arrays, eps)?,
                post_attention_layernorm: LayerNorm::load(
                    &format!("{p}.post_attention_layernorm"),
                    arrays,
                    eps,
                )?,
                attention: PhiMoeAttention::load(
                    &format!("{p}.self_attn"),
                    arrays,
                    config,
                    rope_freqs,
                )?,
                moe: PhiMoeMoe::load(&format!("{p}.block_sparse_moe"), arrays, config, stream_ctx)?,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let h = &x + self.attention.forward(&self.input_layernorm.forward(&x)?)?;
            let r = self
                .moe
                .forward(&self.post_attention_layernorm.forward(&h)?)?;
            Ok(h + r)
        }
    }

    struct PhiMoeLike {
        embed_tokens: Embedding,
        layers: Vec<PhiMoeBlock>,
        norm: LayerNorm,
        lm_head: Linear,
    }

    impl PhiMoeLike {
        fn new(
            config: MlxModelConfig,
            arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            let long_factor: Vec<f32> = config
                .rope_scaling
                .as_ref()
                .and_then(|s| s.get("long_factor"))
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
                .unwrap_or_default();
            let rope_freqs = phi_surope_freqs(head_dim, config.rope_theta, &long_factor);
            let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
            for idx in 0..config.num_hidden_layers {
                layers.push(PhiMoeBlock::load(
                    idx,
                    &arrays,
                    &config,
                    &rope_freqs,
                    stream_ctx,
                )?);
            }
            Ok(Self {
                embed_tokens: Embedding::load("model.embed_tokens", &arrays, &config)?,
                norm: LayerNorm::load("model.norm", &arrays, config.rms_norm_eps)?,
                lm_head: Linear::load("lm_head", &arrays, &config)?,
                layers,
            })
        }
    }

    impl CausalLm for PhiMoeLike {
        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let mut h = self.embed_tokens.forward(&ids)?;
            for layer in &mut self.layers {
                h = layer.forward(h)?;
            }
            h = self.norm.forward(&h)?;
            let logits = self.lm_head.forward(&h)?;
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

    // ---------------------- Llama-4 (Scout/Maverick text) ----------------------
    // iRoPE (RoPE on 3 of every 4 layers, NoPE on the 4th) + weightless L2 qk-norm on the RoPE layers +
    // llama3 NTK-by-parts rope scaling + top-1 sigmoid MoE with an always-on shared expert. Pre-norm.
    // Llama-3 rope scaling: returns the per-dim theta values (base^(2i/dim)) after NTK-by-parts rescaling.
    fn llama3_rope_freqs(head_dim: i32, base: f32, scaling: &serde_json::Value) -> Array {
        let f = |k: &str, d: f32| -> f32 {
            scaling.get(k).and_then(|v| v.as_f64()).unwrap_or(d as f64) as f32
        };
        let (factor, low, high, orig_max) = (
            f("factor", 8.0),
            f("low_freq_factor", 1.0),
            f("high_freq_factor", 4.0),
            f("original_max_position_embeddings", 8192.0),
        );
        let low_wavelen = orig_max / low;
        let high_wavelen = orig_max / high;
        let half = (head_dim / 2) as usize;
        let mut thetas = Vec::with_capacity(half);
        for i in 0..half {
            let theta = base.powf((2 * i) as f32 / head_dim as f32);
            let inv_freq = 1.0 / theta;
            let wavelen = 2.0 * std::f32::consts::PI / inv_freq;
            let new_inv = if wavelen > low_wavelen {
                inv_freq / factor
            } else if wavelen < high_wavelen {
                inv_freq
            } else {
                let smooth = (orig_max / wavelen - low) / (high - low);
                (1.0 - smooth) * inv_freq / factor + smooth * inv_freq
            };
            thetas.push(1.0 / new_inv);
        }
        Array::from_slice(&thetas, &[half as i32])
    }

    struct Llama4Attention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        use_rope: bool,
        qk_norm: bool,
        rope_freqs: Array,
        qk_ones: Array,
        cache: Cache,
    }

    impl Llama4Attention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            layer_idx: u32,
            rope_freqs: &Array,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            // iRoPE: every 4th layer (idx 3, 7, ...) is NoPE.
            let use_rope = (layer_idx + 1) % 4 != 0;
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: Linear::load(&format!("{prefix}.v_proj"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                n_heads: config.num_attention_heads as i32,
                n_kv_heads: config.num_key_value_heads as i32,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
                use_rope,
                // qk-norm applies only on RoPE layers (config gate).
                qk_norm: use_rope
                    && config
                        .raw
                        .get("use_qk_norm")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                rope_freqs: rope_freqs.clone(),
                qk_ones: Array::ones::<f32>(&[head_dim])?,
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
            if self.use_rope {
                // Custom (llama3-scaled) freqs: base must be unset (MLX rejects base+freqs together).
                q = rope(
                    q,
                    self.head_dim,
                    false,
                    None::<f32>,
                    1.0,
                    offset,
                    Some(&self.rope_freqs),
                )?;
                k = rope(
                    k,
                    self.head_dim,
                    false,
                    None::<f32>,
                    1.0,
                    offset,
                    Some(&self.rope_freqs),
                )?;
                if self.qk_norm {
                    // Weightless L2 norm over head_dim (eps 1e-6), after RoPE.
                    q = rms_norm(&q, &self.qk_ones, 1e-6)?;
                    k = rms_norm(&k, &self.qk_ones, 1e-6)?;
                }
            }
            let (k, v) = self.cache.update(k, v)?;
            let output = if l > 1 && offset == 0 {
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    ScaledDotProductAttentionMask::Causal,
                    None::<&Array>,
                )?
            } else if l > 1 {
                let mask = causal_attention_mask(l, k.shape()[2], offset);
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    ScaledDotProductAttentionMask::Array(&mask),
                    None::<&Array>,
                )?
            } else {
                scaled_dot_product_attention(&q, &k, &v, self.scale, None, None::<&Array>)?
            };
            let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                l,
                self.n_heads * self.head_dim,
            ])?;
            self.o_proj.forward(&output)
        }
    }

    struct Llama4Moe {
        router: Linear,
        experts: SwitchMlp,
        shared_expert: Mlp,
    }

    impl Llama4Moe {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            Ok(Self {
                router: Linear::load(&format!("{prefix}.router"), arrays, config)?,
                experts: SwitchMlp::load(&format!("{prefix}.experts"), arrays, config, stream_ctx)?,
                shared_expert: Mlp::load(&format!("{prefix}.shared_expert"), arrays, config)?,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l, d) = (shape[0], shape[1], shape[2]);
            if b != 1 {
                bail!("Llama-4 MoE supports batch size 1, got {b}");
            }
            let logits = self.router.forward(x)?.as_type::<f32>()?;
            transforms::eval([&logits])?;
            let n_exp = *logits.shape().last().unwrap() as usize;
            let raw = logits.as_slice::<f32>();
            let mut outputs = Vec::with_capacity(l as usize);
            for token in 0..l as usize {
                let lg = &raw[token * n_exp..token * n_exp + n_exp];
                // Top-1 routing; the winning logit's sigmoid scales the expert *input*.
                let top1 = (0..n_exp).max_by(|&a, &b| lg[a].total_cmp(&lg[b])).unwrap();
                let score = 1.0 / (1.0 + (-lg[top1]).exp());
                let e = top1 as i32;
                let xt = x.index((0, token as i32, ..)).reshape(&[1, 1, d])?;
                let scaled = &xt * score;
                let gate = self.experts.gate_proj.forward_expert(&scaled, e)?;
                let up = self.experts.up_proj.forward_expert(&scaled, e)?;
                let act = silu(&gate)? * up;
                let expert_out = self.experts.down_proj.forward_expert(&act, e)?;
                let shared_out = self.shared_expert.forward(&xt)?;
                outputs.push(expert_out + shared_out);
            }
            Ok(concatenate_axis(&outputs, 1)?)
        }
    }

    struct Llama4Block {
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
        attention: Llama4Attention,
        moe: Llama4Moe,
    }

    impl Llama4Block {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            rope_freqs: &Array,
            stream_ctx: Option<&StreamContext>,
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
                attention: Llama4Attention::load(
                    &format!("{p}.self_attn"),
                    arrays,
                    config,
                    idx,
                    rope_freqs,
                )?,
                moe: Llama4Moe::load(&format!("{p}.feed_forward"), arrays, config, stream_ctx)?,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let h = &x + self.attention.forward(&self.input_layernorm.forward(&x)?)?;
            let r = self
                .moe
                .forward(&self.post_attention_layernorm.forward(&h)?)?;
            Ok(h + r)
        }
    }

    struct Llama4Like {
        embed_tokens: Embedding,
        layers: Vec<Llama4Block>,
        norm: RmsNorm,
        lm_head: Option<Linear>,
    }

    impl Llama4Like {
        fn new(
            config: MlxModelConfig,
            arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            let rope_freqs = match &config.rope_scaling {
                Some(s) => llama3_rope_freqs(head_dim, config.rope_theta, s),
                None => {
                    let half = (head_dim / 2) as usize;
                    let thetas: Vec<f32> = (0..half)
                        .map(|i| config.rope_theta.powf((2 * i) as f32 / head_dim as f32))
                        .collect();
                    Array::from_slice(&thetas, &[half as i32])
                }
            };
            let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
            for idx in 0..config.num_hidden_layers {
                layers.push(Llama4Block::load(
                    idx,
                    &arrays,
                    &config,
                    &rope_freqs,
                    stream_ctx,
                )?);
            }
            // Llama-4 leaves tie_word_embeddings unset (hi-mlx defaults it true) but ships a separate
            // lm_head — detect the weight instead of trusting the flag.
            let lm_head =
                if arrays.contains_key("lm_head.weight") || arrays.contains_key("lm_head.scales") {
                    Some(Linear::load("lm_head", &arrays, &config)?)
                } else {
                    None
                };
            Ok(Self {
                embed_tokens: Embedding::load("model.embed_tokens", &arrays, &config)?,
                norm: RmsNorm::load("model.norm.weight", &arrays, config.rms_norm_eps)?,
                layers,
                lm_head,
            })
        }
    }

    impl CausalLm for Llama4Like {
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

    // ---------------------- Cohere2 (Command-R 7B) ----------------------
    // LayerNorm (no bias) + parallel attention/MLP block (single input norm, both added to the residual)
    // + NoPE on full-attention layers (rope only on sliding layers) + logit_scale. Tied embeddings, no
    // embedding scale, `model.` prefix.
    struct CohereAttention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        rope_theta: f32,
        use_rope: bool,
        cache: Cache,
    }

    impl CohereAttention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            layer_idx: u32,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            let pattern = config
                .raw
                .get("sliding_window_pattern")
                .and_then(|v| v.as_u64())
                .unwrap_or(4)
                .max(1) as u32;
            // Sliding-window layers use RoPE; the periodic full-attention layers use NoPE.
            let use_rope = (layer_idx + 1) % pattern != 0;
            let cache = if use_rope {
                Cache::with_max_len(config.sliding_window.map(|w| w as i32))
            } else {
                Cache::new()
            };
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: Linear::load(&format!("{prefix}.v_proj"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                n_heads: config.num_attention_heads as i32,
                n_kv_heads: config.num_key_value_heads as i32,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
                rope_theta: config.rope_theta,
                use_rope,
                cache,
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
            if self.use_rope {
                q = rope(q, self.head_dim, true, self.rope_theta, 1.0, offset, None)?;
                k = rope(k, self.head_dim, true, self.rope_theta, 1.0, offset, None)?;
            }
            let (k, v) = self.cache.update(k, v)?;
            let output = if l > 1 && offset == 0 {
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    ScaledDotProductAttentionMask::Causal,
                    None::<&Array>,
                )?
            } else if l > 1 {
                let mask = causal_attention_mask(l, k.shape()[2], offset);
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    ScaledDotProductAttentionMask::Array(&mask),
                    None::<&Array>,
                )?
            } else {
                scaled_dot_product_attention(&q, &k, &v, self.scale, None, None::<&Array>)?
            };
            let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                l,
                self.n_heads * self.head_dim,
            ])?;
            self.o_proj.forward(&output)
        }
    }

    struct CohereBlock {
        input_layernorm: LayerNorm,
        attention: CohereAttention,
        mlp: Mlp,
    }

    impl CohereBlock {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let p = format!("model.layers.{idx}");
            let eps = cohere_norm_eps(config);
            Ok(Self {
                input_layernorm: LayerNorm::load(&format!("{p}.input_layernorm"), arrays, eps)?,
                attention: CohereAttention::load(&format!("{p}.self_attn"), arrays, config, idx)?,
                mlp: Mlp::load(&format!("{p}.mlp"), arrays, config)?,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            // Parallel: attention and MLP both read the single normed input, both add to the residual.
            let h = self.input_layernorm.forward(&x)?;
            let attn = self.attention.forward(&h)?;
            let ff = self.mlp.forward(&h)?;
            Ok(attn + ff + x)
        }
    }

    fn cohere_norm_eps(config: &MlxModelConfig) -> f32 {
        config
            .raw
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1e-5)
    }

    struct CohereLike {
        embed_tokens: Embedding,
        layers: Vec<CohereBlock>,
        norm: LayerNorm,
        logit_scale: f32,
    }

    impl CohereLike {
        fn new(config: MlxModelConfig, arrays: HashMap<String, Array>) -> Result<Self> {
            let layers = (0..config.num_hidden_layers)
                .map(|idx| CohereBlock::load(idx, &arrays, &config))
                .collect::<Result<Vec<_>>>()?;
            let logit_scale = config
                .raw
                .get("logit_scale")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(1.0);
            Ok(Self {
                embed_tokens: Embedding::load("model.embed_tokens", &arrays, &config)?,
                norm: LayerNorm::load("model.norm", &arrays, cohere_norm_eps(&config))?,
                layers,
                logit_scale,
            })
        }
    }

    impl CausalLm for CohereLike {
        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let mut h = self.embed_tokens.forward(&ids)?;
            for layer in &mut self.layers {
                h = layer.forward(h)?;
            }
            h = self.norm.forward(&h)?;
            let logits = self.embed_tokens.as_linear(&h)? * self.logit_scale;
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

    // ---------------------- Nemotron (Llama-based, non-H) ----------------------
    // Standard GQA (partial rotary) + LayerNorm1P (LayerNorm with weight+1) + squared-ReLU MLP
    // (down_proj(relu(up_proj(x))^2), no gate). Pre-norm, `model.` prefix.
    fn nemotron_ln1p(prefix: &str, arrays: &HashMap<String, Array>, eps: f32) -> Result<LayerNorm> {
        Ok(LayerNorm {
            weight: raw_array(arrays, &format!("{prefix}.weight"))? + 1.0f32,
            bias: arrays.get(&format!("{prefix}.bias")).cloned(),
            eps,
        })
    }

    struct NemoLmMlp {
        up_proj: Linear,
        down_proj: Linear,
    }

    impl NemoLmMlp {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                up_proj: Linear::load(&format!("{prefix}.up_proj"), arrays, config)?,
                down_proj: Linear::load(&format!("{prefix}.down_proj"), arrays, config)?,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let u = self.up_proj.forward(x)?;
            let r = maximum(&u, &Array::from_f32(0.0))?;
            self.down_proj.forward(&(&r * &r))
        }
    }

    struct NemoLmAttention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rot_dims: i32,
        scale: f32,
        rope_theta: f32,
        cache: Cache,
    }

    impl NemoLmAttention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            let rot_dims = (head_dim as f32 * config.partial_rotary_factor.unwrap_or(1.0)) as i32;
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: Linear::load(&format!("{prefix}.v_proj"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                n_heads: config.num_attention_heads as i32,
                n_kv_heads: config.num_key_value_heads as i32,
                head_dim,
                rot_dims,
                scale: (head_dim as f32).powf(-0.5),
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
            q = rope(q, self.rot_dims, false, self.rope_theta, 1.0, offset, None)?;
            k = rope(k, self.rot_dims, false, self.rope_theta, 1.0, offset, None)?;
            let (k, v) = self.cache.update(k, v)?;
            let output = if l > 1 && offset == 0 {
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    ScaledDotProductAttentionMask::Causal,
                    None::<&Array>,
                )?
            } else if l > 1 {
                let mask = causal_attention_mask(l, k.shape()[2], offset);
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    ScaledDotProductAttentionMask::Array(&mask),
                    None::<&Array>,
                )?
            } else {
                scaled_dot_product_attention(&q, &k, &v, self.scale, None, None::<&Array>)?
            };
            let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                l,
                self.n_heads * self.head_dim,
            ])?;
            self.o_proj.forward(&output)
        }
    }

    struct NemoLmBlock {
        input_layernorm: LayerNorm,
        post_attention_layernorm: LayerNorm,
        attention: NemoLmAttention,
        mlp: NemoLmMlp,
    }

    impl NemoLmBlock {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let p = format!("model.layers.{idx}");
            Ok(Self {
                input_layernorm: nemotron_ln1p(
                    &format!("{p}.input_layernorm"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                post_attention_layernorm: nemotron_ln1p(
                    &format!("{p}.post_attention_layernorm"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                attention: NemoLmAttention::load(&format!("{p}.self_attn"), arrays, config)?,
                mlp: NemoLmMlp::load(&format!("{p}.mlp"), arrays, config)?,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let h = &x + self.attention.forward(&self.input_layernorm.forward(&x)?)?;
            let r = self
                .mlp
                .forward(&self.post_attention_layernorm.forward(&h)?)?;
            Ok(h + r)
        }
    }

    struct NemotronLike {
        embed_tokens: Embedding,
        layers: Vec<NemoLmBlock>,
        norm: LayerNorm,
        lm_head: Option<Linear>,
    }

    impl NemotronLike {
        fn new(config: MlxModelConfig, arrays: HashMap<String, Array>) -> Result<Self> {
            let layers = (0..config.num_hidden_layers)
                .map(|idx| NemoLmBlock::load(idx, &arrays, &config))
                .collect::<Result<Vec<_>>>()?;
            let lm_head = if config.tie_word_embeddings {
                None
            } else {
                Some(Linear::load("lm_head", &arrays, &config)?)
            };
            Ok(Self {
                embed_tokens: Embedding::load("model.embed_tokens", &arrays, &config)?,
                norm: nemotron_ln1p("model.norm", &arrays, config.rms_norm_eps)?,
                layers,
                lm_head,
            })
        }
    }

    impl CausalLm for NemotronLike {
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

    struct QwenLike {
        embed_tokens: Embedding,
        layers: Vec<QwenBlock>,
        norm: RmsNorm,
        lm_head: Option<Linear>,
        embedding_multiplier: f32,
        logits_scaling: f32,
    }

    impl QwenLike {
        fn new(
            config: MlxModelConfig,
            mut arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            remap_hy3_moe_weights(&config, &mut arrays)?;
            prepare_qwen_moe_weights(&config, &mut arrays)?;
            let layers = (0..config.num_hidden_layers)
                .map(|idx| QwenBlock::load(idx, &arrays, &config, stream_ctx))
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
                embedding_multiplier: config.embedding_multiplier,
                logits_scaling: config.logits_scaling,
            })
        }
    }

    impl CausalLm for QwenLike {
        fn supports_batch(&self) -> bool {
            true
        }

        fn stage_pad_lens(&mut self, pad_lens: Option<&[i32]>) {
            for layer in &mut self.layers {
                layer.attention.pad_lens = pad_lens.map(<[i32]>::to_vec);
            }
        }

        /// Batched forward over `[B, L]` ids. Identical to [`CausalLm::forward`] except the input
        /// already carries a batch dimension: every op below is shape-generic, and the KV cache
        /// concatenates along the sequence axis, so `b > 1` needs no other change. Padded key
        /// positions are hidden by the bias staged via [`CausalLm::stage_pad_lens`].
        fn forward_batch(&mut self, input_ids: &Array) -> Result<Array> {
            let mut h = self.embed_tokens.forward(input_ids)?;
            if self.embedding_multiplier != 1.0 {
                h = h * self.embedding_multiplier;
            }
            for layer in &mut self.layers {
                h = layer.forward(h)?;
            }
            h = self.norm.forward(&h)?;
            let mut logits = match &self.lm_head {
                Some(head) => head.forward(&h)?,
                None => self.embed_tokens.as_linear(&h)?,
            };
            if self.logits_scaling != 1.0 {
                logits = logits / self.logits_scaling;
            }
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let mut h = self.embed_tokens.forward(&ids)?;
            if self.embedding_multiplier != 1.0 {
                h = h * self.embedding_multiplier;
            }
            for layer in &mut self.layers {
                h = layer.forward(h)?;
            }
            h = self.norm.forward(&h)?;
            let mut logits = match &self.lm_head {
                Some(head) => head.forward(&h)?,
                None => self.embed_tokens.as_linear(&h)?,
            };
            if self.logits_scaling != 1.0 {
                logits = logits / self.logits_scaling;
            }
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

    /// pipenetwork's V4-Flash MLX exports use a bare naming scheme (`embed.*`, `head.*`,
    /// `layers.N.attn.*`, `layers.N.ffn.experts.*`, `hc_head_*`) that their bundled
    /// `deepseek_v4_mlx` Python package resolves. Component names match `DeepSeekV4Like`'s
    /// loaders one-to-one; only the prefixes and a few spellings differ, so rename in place.
    /// Triggered only when the bare scheme is present, so HF-standard V4 checkpoints are
    /// untouched. The shared-expert w1/w2/w3 → gate/down/up mapping is pinned by tensor
    /// shapes: w1/w3 are [moe_inter, ·] (gate/up), w2 is [hidden, ·] (down).
    fn remap_v4_bare_weights(arrays: &mut HashMap<String, Array>) {
        if arrays.contains_key("model.embed_tokens.weight") || !arrays.contains_key("embed.weight")
        {
            return;
        }
        let keys: Vec<String> = arrays.keys().cloned().collect();
        for key in keys {
            let new = remap_v4_bare_key(&key);
            if new != key {
                if let Some(value) = arrays.remove(&key) {
                    arrays.insert(new, value);
                }
            }
        }
    }

    fn remap_v4_bare_key(key: &str) -> String {
        if let Some(rest) = key.strip_prefix("embed.") {
            return format!("model.embed_tokens.{rest}");
        }
        if let Some(rest) = key.strip_prefix("head.") {
            return format!("lm_head.{rest}");
        }
        if let Some(rest) = key.strip_prefix("hc_head_") {
            return format!("model.hc_head.{rest}");
        }
        if key == "norm.weight" {
            return "model.norm.weight".to_string();
        }
        if let Some(rest) = key.strip_prefix("layers.") {
            let mut k = rest.to_string();
            for hc in ["hc_attn", "hc_ffn"] {
                for suffix in ["base", "fn", "scale"] {
                    let flat = format!(".{hc}_{suffix}");
                    if k.ends_with(&flat) {
                        k = k.replace(&flat, &format!(".{hc}.{suffix}"));
                    }
                }
            }
            k = k.replace(".ffn.experts.", ".ffn.switch_mlp.");
            k = k.replace(".ffn.shared_experts.w1.", ".ffn.shared_experts.gate_proj.");
            k = k.replace(".ffn.shared_experts.w2.", ".ffn.shared_experts.down_proj.");
            k = k.replace(".ffn.shared_experts.w3.", ".ffn.shared_experts.up_proj.");
            if k.ends_with(".ffn.gate.bias") {
                k = k.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias");
            }
            return format!("model.layers.{k}");
        }
        key.to_string()
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

    /// Exact inverse of the batched unit-lower-triangular `I + a` (`a` strictly lower), by block
    /// doubling. For `X = [[L, 0], [C, R]]` the inverse is `[[L^-1, 0], [-R^-1 C L^-1, R^-1]]`, so
    /// once `t` inverts every diagonal block of width `w`, the update
    ///     `t <- t - t (a ∘ cross_w) t`
    /// inverts every block of width `2w`, where `cross_w` selects the `C` blocks at that scale.
    /// Exact after `ceil(log2 cs)` steps.
    ///
    /// This replaces a Newton-Schulz iteration that was mathematically exact here too (`a` is
    /// nilpotent) but unusable in f32: its partial sums accumulate high powers of `a`, which
    /// overshoot the answer by orders of magnitude before cancelling. When a chunk's k vectors are
    /// strongly correlated — routine in real layers — the overshoot reached ~1e13 against a true
    /// inverse of order 1, so every significant bit was lost. The resulting garbage fed the
    /// `u`/`state` recurrence in `scan_chunked`, which then diverged geometrically
    /// (1e22 → 1e27 → … → inf), and prompts past a few hundred tokens decoded as pure `!`
    /// (token 0). Block doubling keeps every intermediate at the scale of the final inverse.
    fn unit_lower_inverse(a: &Array, eye: &Array, shape: &[i32], cs: i32) -> Result<Array> {
        let mut t = broadcast_to(eye, shape)?;
        let mut width = 1;
        while width < cs {
            let span = width * 2;
            let mut cross = vec![0f32; (cs * cs) as usize];
            for row in 0..cs {
                for col in 0..cs {
                    // Same span-block, row in its lower half, column in its upper half.
                    if row / span == col / span && row % span >= width && col % span < width {
                        cross[(row * cs + col) as usize] = 1.0;
                    }
                }
            }
            let cross = Array::from_slice(&cross, &[cs, cs]);
            let coupling = matmul(&t, &(a.clone() * cross))?;
            t = t.clone() - matmul(&coupling, &t)?;
            width = span;
        }
        Ok(t)
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
                q_norm: RmsNorm::load(
                    &format!("{prefix}.q_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )
                .ok(),
                k_norm: RmsNorm::load(
                    &format!("{prefix}.k_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )
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
                norm_weight: raw_array(arrays, &format!("{prefix}.norm.weight"))?
                    .as_type::<f32>()?,
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
            let q = q
                .reshape(&[nc, cs, hv, hk])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = k
                .reshape(&[nc, cs, hv, hk])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let v = v
                .reshape(&[nc, cs, hv, dv])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let g = g.reshape(&[nc, cs, hv])?.transpose_axes(&[0, 2, 1])?;
            let beta = beta.reshape(&[nc, cs, hv])?.transpose_axes(&[0, 2, 1])?;

            let ltri = tril(Array::ones::<f32>(&[cs, cs])?, 0)?; // lower incl diag (for cumsum)
            let eye = identity::<f32>(cs)?;
            // Additive masks: 0 on the kept triangle, -1e9 elsewhere. Added to the (finite) log-decay
            // differences *before* exp, so masked-out entries become exp(-1e9)=0 with no inf·0 = NaN.
            let (mut pen_incl, mut pen_strict) = (
                vec![0f32; (cs * cs) as usize],
                vec![0f32; (cs * cs) as usize],
            );
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
            let tinv = unit_lower_inverse(&a, &eye, &[nc, hv, cs, cs], cs)?;
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
                let w_c = w_all
                    .index((c..(c + 1), .., .., ..))
                    .reshape(&[hv, cs, dv])?;
                let p_c = p_all
                    .index((c..(c + 1), .., .., ..))
                    .reshape(&[hv, cs, hk])?;
                let qk_c = qk_all
                    .index((c..(c + 1), .., .., ..))
                    .reshape(&[hv, cs, cs])?;
                let qbar_c = qbar
                    .index((c..(c + 1), .., .., ..))
                    .reshape(&[hv, cs, hk])?;
                let kfinal_c = kfinal_all
                    .index((c..(c + 1), .., .., ..))
                    .reshape(&[hv, cs, hk])?;
                let gl_c = gamma_last
                    .index((c..(c + 1), .., .., ..))
                    .reshape(&[hv, 1, 1])?;
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
            stream_ctx: Option<&StreamContext>,
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
                ffn: QwenFfn::load(idx, arrays, config, stream_ctx)?,
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
        fn new(
            config: MlxModelConfig,
            arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let layers = (0..config.num_hidden_layers)
                .map(|idx| Qwen35Layer::load(idx, &arrays, &config, stream_ctx))
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
            let h = self
                .mlp
                .forward(&self.post_attention_layernorm.forward(&x)?)?;
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

    // ---------------------- Nemotron-H (nemotron_h, Mamba2 hybrid) ----------------------
    // NVIDIA Nemotron-3 (Nano/Ultra) + TwoTower: a per-layer hybrid selected by
    // `hybrid_override_pattern` — 'M' = Mamba2 SSM, '*' = attention (GQA, NO RoPE; position comes
    // from the Mamba layers), '-' = dense ReLU^2 MLP, 'E' = MoE. Weights use the `backbone.` prefix.
    // The Mamba2 mixer runs the SSD recurrence per-token (correctness first, like the qwen3.5 scan).
    struct NemotronMamba2 {
        in_proj: Linear,
        conv1d_weight: Array,
        conv1d_bias: Option<Array>,
        a_log: Array,
        d: Array,
        dt_bias: Array,
        norm_weight: Array,
        norm_ones: Array,
        out_proj: Linear,
        num_heads: i32,
        head_dim: i32,
        n_groups: i32,
        state_size: i32,
        conv_dim: i32,
        conv_kernel: i32,
        intermediate: i32,
        group_size: i32,
        eps: f32,
        conv_state: Option<Array>,
        ssm_state: Option<Array>,
    }

    impl NemotronMamba2 {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let num_heads = config.mamba_num_heads.unwrap_or(0) as i32;
            let head_dim = config.mamba_head_dim.unwrap_or(0) as i32;
            let n_groups = config.mamba_n_groups.unwrap_or(1).max(1) as i32;
            let state_size = config.ssm_state_size.unwrap_or(128) as i32;
            let conv_kernel = config.mamba_conv_kernel.unwrap_or(4) as i32;
            let intermediate = num_heads * head_dim;
            let conv_dim = intermediate + 2 * n_groups * state_size;
            let group_size = (intermediate / n_groups).max(1);
            Ok(Self {
                in_proj: Linear::load(&format!("{prefix}.in_proj"), arrays, config)?,
                conv1d_weight: raw_array(arrays, &format!("{prefix}.conv1d.weight"))?
                    .as_type::<f32>()?,
                conv1d_bias: match arrays.get(&format!("{prefix}.conv1d.bias")) {
                    Some(b) => Some(b.as_type::<f32>()?),
                    None => None,
                },
                a_log: raw_array(arrays, &format!("{prefix}.A_log"))?.as_type::<f32>()?,
                d: raw_array(arrays, &format!("{prefix}.D"))?.as_type::<f32>()?,
                dt_bias: raw_array(arrays, &format!("{prefix}.dt_bias"))?.as_type::<f32>()?,
                norm_weight: raw_array(arrays, &format!("{prefix}.norm.weight"))?
                    .as_type::<f32>()?,
                norm_ones: Array::ones::<f32>(&[group_size])?,
                out_proj: Linear::load(&format!("{prefix}.out_proj"), arrays, config)?,
                num_heads,
                head_dim,
                n_groups,
                state_size,
                conv_dim,
                conv_kernel,
                intermediate,
                group_size,
                eps: config.rms_norm_eps,
                conv_state: None,
                ssm_state: None,
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let dtype = x.dtype();
            let s = x.shape()[1];
            let proj = self.in_proj.forward(x)?.as_type::<f32>()?;
            let parts = split_sections(
                &proj,
                &[self.intermediate, self.intermediate + self.conv_dim],
                -1,
            )?;
            let gate = &parts[0];
            let conv_in = &parts[1];
            let dt = &parts[2];
            // Causal depthwise conv over [conv_state | conv_in], carrying the last kernel-1 frames.
            let keep = self.conv_kernel - 1;
            let conv_state = match self.conv_state.take() {
                Some(st) => st,
                None => Array::zeros::<f32>(&[1, keep, self.conv_dim])?,
            };
            let cat = concatenate_axis(&[&conv_state, conv_in], 1)?;
            let clen = cat.shape()[1];
            self.conv_state = Some(cat.index((.., (clen - keep)..clen, ..)));
            let mut conv_out = conv1d(&cat, &self.conv1d_weight, 1, 0, 1, self.conv_dim)?;
            if let Some(bias) = &self.conv1d_bias {
                conv_out = conv_out + bias;
            }
            let conv_out = silu(&conv_out)?;
            let cparts = split_sections(
                &conv_out,
                &[
                    self.intermediate,
                    self.intermediate + self.n_groups * self.state_size,
                ],
                -1,
            )?;
            let y = self.scan(&cparts[0], &cparts[1], &cparts[2], dt, s)?;
            // MambaRMSNormGated: silu(gate) * y, then a group-wise (weightless) RMS norm * weight.
            let y = silu(gate)? * y;
            let ng = self.intermediate / self.group_size;
            let y = rms_norm(
                &y.reshape(&[1, s, ng, self.group_size])?,
                &self.norm_ones,
                self.eps,
            )?
            .reshape(&[1, s, self.intermediate])?;
            let y = y * &self.norm_weight;
            self.out_proj.forward(&y.as_dtype(dtype)?)
        }

        // SSD recurrence: state[h,dh,ds] = dA[h]*state + dt[h]*x[h,dh]*B[h,ds];
        //                 y[h,dh] = sum_ds(state*C[h,ds]) + D[h]*x[h,dh].
        fn scan(
            &mut self,
            x_ssm: &Array,
            bb: &Array,
            cc: &Array,
            dt: &Array,
            s: i32,
        ) -> Result<Array> {
            let (h, dh, g, ds) = (
                self.num_heads,
                self.head_dim,
                self.n_groups,
                self.state_size,
            );
            let x = x_ssm.reshape(&[1, s, h, dh])?;
            let bb = bb.reshape(&[1, s, g, ds])?;
            let cc = cc.reshape(&[1, s, g, ds])?;
            let dt = softplus(&(dt.reshape(&[1, s, h])? + &self.dt_bias))?;
            let dt = minimum(
                &maximum(&dt, &Array::from_f32(0.001))?,
                &Array::from_f32(100.0),
            )?;
            let a = exp(&self.a_log)? * -1.0; // [h]
            let per_group = h / g;
            let mut state = match self.ssm_state.take() {
                Some(st) => st,
                None => Array::zeros::<f32>(&[h, dh, ds])?,
            };
            let mut ys: Vec<Array> = Vec::with_capacity(s as usize);
            for t in 0..s {
                let dt_h = dt.index((0, t, ..)).reshape(&[h])?;
                let da = exp(&(&dt_h * &a))?.reshape(&[h, 1, 1])?;
                let dt_e = dt_h.reshape(&[h, 1, 1])?;
                let x_hd = x.index((0, t, .., ..)); // [h, dh]
                let x_e = x_hd.reshape(&[h, dh, 1])?;
                let b_t = broadcast_to(
                    &bb.index((0, t, .., ..)).reshape(&[g, 1, ds])?,
                    &[g, per_group, ds],
                )?
                .reshape(&[h, 1, ds])?;
                let c_t = broadcast_to(
                    &cc.index((0, t, .., ..)).reshape(&[g, 1, ds])?,
                    &[g, per_group, ds],
                )?
                .reshape(&[h, 1, ds])?;
                let dbx = (&dt_e * &x_e) * &b_t; // [h,dh,ds]
                state = &da * &state + dbx;
                let y_t =
                    sum_axis(&(&state * &c_t), -1, false)? + (self.d.reshape(&[h, 1])? * &x_hd);
                ys.push(y_t.reshape(&[1, 1, h * dh])?);
            }
            self.ssm_state = Some(state);
            if ys.len() == 1 {
                Ok(ys.remove(0))
            } else {
                Ok(concatenate_axis(&ys.iter().collect::<Vec<_>>(), 1)?)
            }
        }
    }

    struct NemotronAttention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        cache: Cache,
    }

    impl NemotronAttention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: Linear::load(&format!("{prefix}.v_proj"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                n_heads: config.num_attention_heads as i32,
                n_kv_heads: config.num_key_value_heads as i32,
                head_dim: config.attention_head_dim() as i32,
                cache: Cache::new(),
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l) = (shape[0], shape[1]);
            let q = self
                .q_proj
                .forward(x)?
                .reshape(&[b, l, self.n_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = self
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

    struct NemotronMlp {
        up_proj: Linear,
        down_proj: Linear,
    }

    impl NemotronMlp {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                up_proj: Linear::load(&format!("{prefix}.up_proj"), arrays, config)?,
                down_proj: Linear::load(&format!("{prefix}.down_proj"), arrays, config)?,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            // ReLU^2 activation (relu2).
            let u = self.up_proj.forward(x)?;
            let a = maximum(&u, &Array::from_f32(0.0))?;
            self.down_proj.forward(&(&a * &a))
        }
    }

    // Non-gated ReLU^2 switch experts (Nemotron uses SwitchMLP: fc1 -> relu^2 -> fc2, not SwiGLU).
    struct NemotronSwitchMlp {
        fc1: SwitchLinear,
        fc2: SwitchLinear,
    }

    impl NemotronSwitchMlp {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            Ok(Self {
                fc1: SwitchLinear::load_or_stream(
                    &format!("{prefix}.fc1"),
                    arrays,
                    config,
                    stream_ctx,
                    "gate_proj",
                )?,
                fc2: SwitchLinear::load_or_stream(
                    &format!("{prefix}.fc2"),
                    arrays,
                    config,
                    stream_ctx,
                    "up_proj",
                )?,
            })
        }

        fn forward_batched(&self, x: &Array, inds: &Array) -> Result<Array> {
            let h = self.fc1.gather(x, inds)?;
            let r = maximum(&h, &Array::from_f32(0.0))?;
            self.fc2.gather(&(&r * &r), inds)
        }
    }

    // Nemotron-H MoE 'E' block: DeepSeek-style sigmoid + e_score_correction_bias (noaux_tc) grouped
    // router, ReLU^2 experts, plus one always-on shared expert.
    struct NemotronHMoE {
        gate: Linear,
        expert_bias: Vec<f32>,
        switch_mlp: NemotronSwitchMlp,
        shared: NemotronMlp,
        top_k: usize,
        n_group: usize,
        topk_group: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    }

    impl NemotronHMoE {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let bias = raw_array(arrays, &format!("{prefix}.gate.e_score_correction_bias"))?
                .as_type::<f32>()?;
            transforms::eval([&bias])?;
            Ok(Self {
                gate: Linear::load(&format!("{prefix}.gate"), arrays, config)?,
                expert_bias: bias.as_slice::<f32>().to_vec(),
                switch_mlp: NemotronSwitchMlp::load(
                    &format!("{prefix}.switch_mlp"),
                    arrays,
                    config,
                    stream_ctx,
                )?,
                shared: NemotronMlp::load(&format!("{prefix}.shared_experts"), arrays, config)?,
                top_k: config.num_experts_per_tok.unwrap_or(1) as usize,
                n_group: config.n_group.max(1) as usize,
                topk_group: config.topk_group.max(1) as usize,
                norm_topk_prob: config.norm_topk_prob,
                routed_scaling_factor: config.routed_scaling_factor,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let d = x.shape()[2];
            let logits = self.gate.forward(x)?;
            let scores = sigmoid(&logits.as_type::<f32>()?)?;
            transforms::eval([&scores])?;
            let shape = scores.shape();
            let (l, n_experts) = (shape[1] as usize, shape[2] as usize);
            let raw = scores.as_slice::<f32>();
            let mut idx_v: Vec<u32> = Vec::with_capacity(l * self.top_k);
            let mut wts_v: Vec<f32> = Vec::with_capacity(l * self.top_k);
            for token in 0..l {
                let base = token * n_experts;
                let mut sel: Vec<f32> = (0..n_experts)
                    .map(|i| raw[base + i] + self.expert_bias[i])
                    .collect();
                // DeepSeek grouped selection: keep only the top `topk_group` groups (by sum of their
                // top-2 selection scores), masking the rest before the global top-k.
                if self.n_group > 1 {
                    let per = n_experts / self.n_group;
                    let mut gscore: Vec<(usize, f32)> = (0..self.n_group)
                        .map(|g| {
                            let mut vals: Vec<f32> = (0..per).map(|j| sel[g * per + j]).collect();
                            vals.sort_by(|a, b| b.total_cmp(a));
                            (g, vals[0] + vals.get(1).copied().unwrap_or(0.0))
                        })
                        .collect();
                    gscore.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let kept: Vec<usize> = gscore
                        .iter()
                        .take(self.topk_group)
                        .map(|(g, _)| *g)
                        .collect();
                    for g in 0..self.n_group {
                        if !kept.contains(&g) {
                            for j in 0..per {
                                sel[g * per + j] = f32::NEG_INFINITY;
                            }
                        }
                    }
                }
                let mut ranked: Vec<usize> = (0..n_experts).collect();
                ranked.sort_by(|&a, &b| sel[b].total_cmp(&sel[a]).then_with(|| a.cmp(&b)));
                ranked.truncate(self.top_k.min(n_experts));
                let mut w: Vec<f32> = ranked.iter().map(|&i| raw[base + i]).collect();
                if self.norm_topk_prob && w.len() > 1 {
                    let denom: f32 = w.iter().sum::<f32>() + 1e-20;
                    for x in &mut w {
                        *x /= denom;
                    }
                }
                for x in &mut w {
                    *x *= self.routed_scaling_factor;
                }
                for (k, &e) in ranked.iter().enumerate() {
                    idx_v.push(e as u32);
                    wts_v.push(w[k]);
                }
            }
            let top_k = self.top_k as i32;
            let inds = Array::from_slice(&idx_v, &[l as i32, top_k]);
            let weights = Array::from_slice(&wts_v, &[l as i32, top_k, 1]);
            let xe = x.reshape(&[l as i32, 1, 1, d])?;
            let expert_out = self
                .switch_mlp
                .forward_batched(&xe, &inds)?
                .reshape(&[l as i32, top_k, d])?
                .as_type::<f32>()?;
            let y = sum_axis(&(expert_out * weights), 1, false)?.reshape(&[1, l as i32, d])?;
            let y = y + self.shared.forward(x)?.as_type::<f32>()?;
            Ok(y)
        }
    }

    enum NemotronMixer {
        Mamba(Box<NemotronMamba2>),
        Attn(NemotronAttention),
        Mlp(NemotronMlp),
        Moe(Box<NemotronHMoE>),
    }

    struct NemotronBlock {
        norm: RmsNorm,
        mixer: NemotronMixer,
    }

    impl NemotronBlock {
        fn load(
            idx: u32,
            kind: char,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let p = format!("backbone.layers.{idx}");
            let norm = RmsNorm::load(&format!("{p}.norm.weight"), arrays, config.rms_norm_eps)?;
            let mixer = match kind {
                'M' => NemotronMixer::Mamba(Box::new(NemotronMamba2::load(
                    &format!("{p}.mixer"),
                    arrays,
                    config,
                )?)),
                '*' => NemotronMixer::Attn(NemotronAttention::load(
                    &format!("{p}.mixer"),
                    arrays,
                    config,
                )?),
                '-' => {
                    NemotronMixer::Mlp(NemotronMlp::load(&format!("{p}.mixer"), arrays, config)?)
                }
                'E' => NemotronMixer::Moe(Box::new(NemotronHMoE::load(
                    &format!("{p}.mixer"),
                    arrays,
                    config,
                    stream_ctx,
                )?)),
                other => bail!("nemotron_h block type '{other}' (layer {idx}) is not supported"),
            };
            Ok(Self { norm, mixer })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let h = self.norm.forward(&x)?;
            let h = match &mut self.mixer {
                NemotronMixer::Mamba(m) => m.forward(&h)?,
                NemotronMixer::Attn(a) => a.forward(&h)?,
                NemotronMixer::Mlp(m) => m.forward(&h)?,
                NemotronMixer::Moe(m) => m.forward(&h)?,
            };
            Ok(x + h)
        }
    }

    struct NemotronHLike {
        embed: Embedding,
        blocks: Vec<NemotronBlock>,
        norm_f: RmsNorm,
        lm_head: Linear,
    }

    impl NemotronHLike {
        fn new(
            config: MlxModelConfig,
            arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let pattern = config
                .hybrid_override_pattern
                .clone()
                .ok_or_else(|| anyhow::anyhow!("nemotron_h: missing hybrid_override_pattern"))?;
            let blocks = pattern
                .chars()
                .enumerate()
                .map(|(idx, kind)| {
                    NemotronBlock::load(idx as u32, kind, &arrays, &config, stream_ctx)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                embed: Embedding::load("backbone.embeddings", &arrays, &config)?,
                norm_f: RmsNorm::load("backbone.norm_f.weight", &arrays, config.rms_norm_eps)?,
                lm_head: Linear::load("lm_head", &arrays, &config)?,
                blocks,
            })
        }
    }

    impl CausalLm for NemotronHLike {
        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let mut h = self.embed.forward(&ids)?;
            for block in &mut self.blocks {
                h = block.forward(h)?;
            }
            h = self.norm_f.forward(&h)?;
            let logits = self.lm_head.forward(&h)?;
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn reset_cache(&mut self) {
            for block in &mut self.blocks {
                match &mut block.mixer {
                    NemotronMixer::Mamba(m) => {
                        m.conv_state = None;
                        m.ssm_state = None;
                    }
                    NemotronMixer::Attn(a) => a.cache.reset(),
                    NemotronMixer::Mlp(_) | NemotronMixer::Moe(_) => {}
                }
            }
        }

        fn prepare_cache(&mut self, capacity: i32) {
            for block in &mut self.blocks {
                if let NemotronMixer::Attn(a) = &mut block.mixer {
                    a.cache.prepare_capacity(capacity);
                }
            }
        }
    }

    fn sigmoid_f32(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    /// log(sigmoid(z)) = -softplus(-z), stable for large |z|.
    fn logsigmoid_f32(z: f32) -> f32 {
        if z >= 0.0 {
            -(1.0 + (-z).exp()).ln()
        } else {
            z - (1.0 + z.exp()).ln()
        }
    }

    /// In-place softmax over a small slice.
    fn softmax_inplace(v: &mut [f32]) {
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in v.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        if sum > 0.0 {
            for x in v.iter_mut() {
                *x /= sum;
            }
        }
    }

    // ---------------------- Inkling (thinkingmachines) text tower ----------------------
    // Ported from the checkpoint's own `inkling_mlx` package (text tower only; the vision/audio
    // towers under model.audio/model.visual are not loaded). Inkling has no RoPE — position enters
    // through a learned relative-logits bias — plus depthwise short convolutions on k/v and after
    // each sublayer, per-head q/k RMSNorm with a 1/head_dim scale (not 1/sqrt), a local/global
    // attention hybrid with distinct head geometries, log-scaling on the global layers, a
    // shared-expert-sink MoE, and muP logit scaling. Weights live under `model.llm.`.

    /// Depthwise causal short conv over `[B, L, C]` with a residual add, fp32, prefill only
    /// (no incremental conv-state cache — the matrix drives single-shot prompts). Weight is the
    /// MLX conv layout `[C, K, 1]`; left-pad by K-1 and keep the first L outputs for causality.
    struct InklingShortConv {
        weight: Array,
        channels: i32,
        kernel: i32,
        // Last (kernel-1) inputs carried across decode steps so a single new token still sees its
        // left context. None at prefill start = zero left-context. Mirrors the reference's conv cache.
        state: Option<Array>,
    }

    impl InklingShortConv {
        fn load(prefix: &str, arrays: &HashMap<String, Array>, channels: i32) -> Result<Self> {
            let weight = raw_array(arrays, &format!("{prefix}.weight"))?;
            let kernel = weight.shape()[1];
            Ok(Self {
                weight: weight.as_type::<f32>()?,
                channels,
                kernel,
                state: None,
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let dtype = x.dtype();
            let xf = x.as_type::<f32>()?;
            let (b, seq, c) = (xf.shape()[0], xf.shape()[1], xf.shape()[2]);
            let keep = self.kernel - 1;
            // Prepend the carried left context (zeros on the first call), run a "valid" conv (no
            // padding) over [left | x] to get exactly `seq` causal outputs, then keep the last
            // (kernel-1) inputs as the next step's context.
            let left = match self.state.take() {
                Some(state) => state,
                None => Array::zeros::<f32>(&[b, keep, c])?,
            };
            let x_in = concatenate_axis(&[&left, &xf], 1)?;
            let clen = x_in.shape()[1];
            self.state = Some(x_in.index((.., (clen - keep)..clen, ..)));
            let out = conv1d(&x_in, &self.weight, 1, 0, 1, self.channels)?;
            let out = out.index((.., 0..seq, ..)) + &xf; // residual on the new tokens
            out.as_dtype(dtype).map_err(Into::into)
        }

        fn reset(&mut self) {
            self.state = None;
        }
    }

    struct InklingAttention {
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wr: Linear,
        wo: Linear,
        k_sconv: InklingShortConv,
        v_sconv: InklingShortConv,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        rel_proj: Array, // [d_rel, rel_extent]
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        d_rel: i32,
        rel_extent: i32,
        scaling: f32,
        sliding_window: Option<i32>,
        log_scaling: Option<(f32, f32)>, // (alpha, n_floor) — global layers only
        // Full KV history (post-norm k/v). A full cache for every layer; the sliding window is
        // applied in the mask rather than by trimming the cache, which keeps kv positions absolute.
        cache: Cache,
    }

    impl InklingAttention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            is_sliding: bool,
        ) -> Result<Self> {
            let g = |k: &str, d: i64| {
                config
                    .raw
                    .get(k)
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(d)
            };
            let head_dim = if is_sliding {
                g("swa_head_dim", g("head_dim", 128))
            } else {
                g("head_dim", 128)
            } as i32;
            let n_heads = if is_sliding {
                g("swa_num_attention_heads", g("num_attention_heads", 64))
            } else {
                g("num_attention_heads", 64)
            } as i32;
            let n_kv_heads = if is_sliding {
                g("swa_num_key_value_heads", g("num_key_value_heads", 8))
            } else {
                g("num_key_value_heads", 8)
            } as i32;
            let d_rel = g("d_rel", 16) as i32;
            let sliding_window = g("sliding_window_size", 512) as i32;
            let rel_extent = if is_sliding {
                sliding_window
            } else {
                g("rel_extent", 1024) as i32
            };
            // Global layers get log-scaling above n_floor tokens; sliding layers never do.
            let log_scaling = (!is_sliding)
                .then(|| {
                    config
                        .raw
                        .get("log_scaling_n_floor")
                        .and_then(serde_json::Value::as_f64)
                        .map(|floor| {
                            let alpha = config
                                .raw
                                .get("log_scaling_alpha")
                                .and_then(serde_json::Value::as_f64)
                                .unwrap_or(0.1);
                            (alpha as f32, floor as f32)
                        })
                })
                .flatten();
            Ok(Self {
                wq: Linear::load(&format!("{prefix}.wq_du"), arrays, config)?,
                wk: Linear::load(&format!("{prefix}.wk_dv"), arrays, config)?,
                wv: Linear::load(&format!("{prefix}.wv_dv"), arrays, config)?,
                wr: Linear::load(&format!("{prefix}.wr_du"), arrays, config)?,
                wo: Linear::load(&format!("{prefix}.wo_ud"), arrays, config)?,
                k_sconv: InklingShortConv::load(
                    &format!("{prefix}.k_sconv"),
                    arrays,
                    n_kv_heads * head_dim,
                )?,
                v_sconv: InklingShortConv::load(
                    &format!("{prefix}.v_sconv"),
                    arrays,
                    n_kv_heads * head_dim,
                )?,
                q_norm: RmsNorm::load(
                    &format!("{prefix}.q_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                k_norm: RmsNorm::load(
                    &format!("{prefix}.k_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                rel_proj: raw_array(arrays, &format!("{prefix}.rel_logits_proj.proj"))?
                    .as_type::<f32>()?,
                n_heads,
                n_kv_heads,
                head_dim,
                d_rel,
                rel_extent,
                scaling: 1.0 / head_dim as f32, // q/k are per-head normed → 1/d, not 1/sqrt(d)
                sliding_window: is_sliding.then_some(sliding_window),
                log_scaling,
                cache: Cache::new(),
            })
        }

        /// Additive `[1, n_heads, Lq, kv_len]` mask combining the relative-position bias with the
        /// causal (and, for sliding layers, windowed) constraint. `rel` is `[1, Lq, n_heads, d_rel]`
        /// for the new query tokens; queries sit at absolute positions `offset..offset+Lq`, keys at
        /// `0..kv_len`. Bias per (query, key) is `rel[qi] · proj[:, dist]` for `dist = q_pos - kv_pos`
        /// when `0 <= dist < rel_extent`; future is -inf, sliding beyond the window is -inf, and (for
        /// global layers) causally-valid keys past rel_extent contribute with zero bias.
        fn build_mask(&self, rel: &Array, offset: i32, lq: i32, kv_len: i32) -> Result<Array> {
            let rel = rel
                .reshape(&[lq, self.n_heads, self.d_rel])?
                .transpose_axes(&[1, 0, 2])?;
            // rel_logits: [H, Lq, rel_extent] — one bias profile over distance per (head, query).
            let rel_logits = matmul(&rel, &self.rel_proj)?;
            transforms::eval([&rel_logits])?;
            let rl = rel_logits.as_slice::<f32>();
            let (h, ext) = (self.n_heads as usize, self.rel_extent as usize);
            let (lqs, kvs) = (lq as usize, kv_len as usize);
            let window = self.sliding_window.map(|w| w as usize);
            let neg_inf = f32::NEG_INFINITY;
            let mut mask = vec![0f32; h * lqs * kvs];
            for head in 0..h {
                for qi in 0..lqs {
                    let q_pos = offset as usize + qi;
                    let profile = &rl[(head * lqs + qi) * ext..(head * lqs + qi) * ext + ext];
                    for j in 0..kvs {
                        let out = (head * lqs + qi) * kvs + j;
                        if j > q_pos {
                            mask[out] = neg_inf; // future
                        } else {
                            let dist = q_pos - j;
                            if window.is_some_and(|w| dist >= w) {
                                mask[out] = neg_inf; // outside the sliding window
                            } else if dist >= ext {
                                mask[out] = 0.0; // global: causally valid, past the bias extent
                            } else {
                                mask[out] = profile[dist];
                            }
                        }
                    }
                }
            }
            Ok(Array::from_slice(&mask, &[1, h as i32, lq, kv_len]))
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l) = (shape[0], shape[1]);
            let q = self.wq.forward(x)?;
            // Short conv on k/v (post-projection, stateful across steps), then per-head norm.
            let k = self.k_sconv.forward(&self.wk.forward(x)?)?;
            let v = self.v_sconv.forward(&self.wv.forward(x)?)?;
            let rel = self.wr.forward(x)?.as_type::<f32>()?;

            let mut q = self
                .q_norm
                .forward(&q.reshape(&[b, l, self.n_heads, self.head_dim])?)?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = self
                .k_norm
                .forward(&k.reshape(&[b, l, self.n_kv_heads, self.head_dim])?)?
                .transpose_axes(&[0, 2, 1, 3])?;
            let v = v
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            // Append to the KV history; queries sit at `offset..offset+l`, keys span the full cache.
            let offset = self.cache.offset;
            let (k, v) = self.cache.update(k, v)?;
            let kv_len = k.shape()[2];

            let mut mask = self.build_mask(&rel, offset, l, kv_len)?;
            // Log-scaling on global layers, only for contexts beyond n_floor: scale q and the bias
            // by tau = 1 + alpha*ln(max((pos+1)/n_floor, 1)), per query position.
            if let Some((alpha, n_floor)) = self.log_scaling {
                if (offset + l) as f32 > n_floor {
                    let tau: Vec<f32> = (0..l)
                        .map(|qi| {
                            let eff = (offset + qi + 1) as f32 / n_floor;
                            1.0 + alpha * eff.max(1.0).ln()
                        })
                        .collect();
                    let tau_q = Array::from_slice(&tau, &[1, 1, l, 1]);
                    q = (q.as_type::<f32>()? * &tau_q).as_dtype(q.dtype())?;
                    mask = mask * &tau_q;
                }
            }

            // The mask is built in f32 (the additive bias + causal fill); SDPA needs it in the
            // query dtype (bf16), matching the reference's `mask.astype(q.dtype)`.
            let mask = mask.as_dtype(q.dtype())?;
            let out = scaled_dot_product_attention(
                &q,
                &k,
                &v,
                self.scaling,
                ScaledDotProductAttentionMask::Array(&mask),
                None::<&Array>,
            )?;
            let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                b,
                l,
                self.n_heads * self.head_dim,
            ])?;
            self.wo.forward(&out)
        }

        fn reset(&mut self) {
            self.cache.reset();
            self.k_sconv.reset();
            self.v_sconv.reset();
        }

        fn prepare(&mut self, capacity: i32) {
            self.cache.prepare_capacity(capacity);
        }
    }

    /// Inkling dense SwiGLU MLP with a learned scalar output gain (`global_scale`).
    struct InklingDenseMlp {
        mlp: Mlp,
        global_scale: Array,
    }

    impl InklingDenseMlp {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                mlp: Mlp::load(prefix, arrays, config)?,
                global_scale: raw_array(arrays, &format!("{prefix}.global_scale"))?,
            })
        }
        fn forward(&self, x: &Array) -> Result<Array> {
            Ok(self.mlp.forward(x)? * &self.global_scale)
        }
    }

    /// Inkling sparse MoE: sigmoid router with a correction bias, softmax over the selected routed
    /// logits *and* the shared-expert logits jointly (the shared experts form a routing "sink"),
    /// then routed + shared outputs summed. The stacked routed/shared experts reuse SwitchMlp.
    struct InklingMoe {
        // Router weight loaded raw (not via Linear): `mlp.gate.bias` is the 128-element correction
        // bias, applied to routed scores for selection, NOT a linear bias over the 130-wide logits.
        gate_weight: Array,
        gate_bias: Vec<f32>,
        experts: SwitchMlp,
        shared_experts: SwitchMlp,
        top_k: usize,
        n_shared: usize,
        route_scale: f32,
        gate_global_scale: f32,
    }

    impl InklingMoe {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let gi = |k: &str, d: i64| {
                config
                    .raw
                    .get(k)
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(d)
            };
            let gf = |k: &str, d: f64| {
                config
                    .raw
                    .get(k)
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(d)
            };
            let gate_bias = {
                let b = raw_array(arrays, &format!("{prefix}.gate.bias"))?.as_type::<f32>()?;
                transforms::eval([&b])?;
                b.as_slice::<f32>().to_vec()
            };
            let gate_global_scale = arrays
                .get(&format!("{prefix}.gate.global_scale"))
                .map(|g| -> Result<f32> {
                    let g = g.as_type::<f32>()?;
                    transforms::eval([&g])?;
                    Ok(g.as_slice::<f32>()[0])
                })
                .transpose()?
                .unwrap_or(1.0);
            Ok(Self {
                gate_weight: raw_array(arrays, &format!("{prefix}.gate.weight"))?,
                gate_bias,
                experts: SwitchMlp::load(&format!("{prefix}.experts"), arrays, config, stream_ctx)?,
                // Shared experts always stay resident: there are only two and they fire on every
                // token, and the stream source is keyed by (layer, projection) with no routed/shared
                // distinction — passing stream_ctx here would make them read the routed slabs.
                shared_experts: SwitchMlp::load(
                    &format!("{prefix}.shared_experts"),
                    arrays,
                    config,
                    None,
                )?,
                top_k: gi("num_experts_per_tok", 6) as usize,
                n_shared: gi("n_shared_experts", 2) as usize,
                route_scale: gf("route_scale", 1.0) as f32,
                gate_global_scale,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l, h) = (shape[0], shape[1], shape[2]);
            if b != 1 {
                bail!("hi-mlx Inkling MoE supports batch size 1, got {b}");
            }
            // Router logits over routed + shared experts, fp32 (top-k selection is precision-
            // sensitive; a flipped choice compounds over 64 MoE layers).
            let logits = matmul(x, &self.gate_weight.t())?.as_type::<f32>()?;
            transforms::eval([&logits])?;
            let all = logits.as_slice::<f32>();
            let n_routed = self.gate_bias.len();
            let n_total = all.len() / (l as usize);
            let scale = self.route_scale * self.gate_global_scale;

            let mut per_token: Vec<Array> = Vec::with_capacity(l as usize);
            for token in 0..l as usize {
                let row = &all[token * n_total..token * n_total + n_total];
                let routed_logits = &row[..n_routed];
                let shared_logits = &row[n_routed..];
                // Select top-k by sigmoid(score)+bias; combine with the softmax over the *selected*
                // routed logits and the shared logits together (logsigmoid → softmax).
                let mut ranked = (0..n_routed)
                    .map(|i| (i, sigmoid_f32(routed_logits[i]) + self.gate_bias[i]))
                    .collect::<Vec<_>>();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                ranked.truncate(self.top_k);

                // logsigmoid(z) = -softplus(-z); softmax over [selected routed logits, shared].
                let mut sel_logits: Vec<f32> = ranked
                    .iter()
                    .map(|(i, _)| logsigmoid_f32(routed_logits[*i]))
                    .chain(shared_logits.iter().map(|z| logsigmoid_f32(*z)))
                    .collect();
                softmax_inplace(&mut sel_logits);
                for w in &mut sel_logits {
                    *w *= scale;
                }

                let token_x = x.index((0, token as i32, ..)).reshape(&[1, 1, h])?;
                let mut acc = Array::zeros::<f32>(&[1, 1, h])?;
                for (slot, (expert, _)) in ranked.iter().enumerate() {
                    acc = acc
                        + self.experts.forward_expert(&token_x, *expert as i32)? * sel_logits[slot];
                }
                for s in 0..self.n_shared {
                    acc = acc
                        + self.shared_experts.forward_expert(&token_x, s as i32)?
                            * sel_logits[self.top_k + s];
                }
                per_token.push(acc.reshape(&[1, 1, h])?);
            }
            let out = concatenate_axis(&per_token.iter().collect::<Vec<_>>(), 1)?;
            out.reshape(&[b, l, h]).map_err(Into::into)
        }
    }

    enum InklingFfn {
        Dense(InklingDenseMlp),
        Moe(Box<InklingMoe>),
    }

    struct InklingBlock {
        attn: InklingAttention,
        attn_norm: RmsNorm,
        mlp_norm: RmsNorm,
        ffn: InklingFfn,
        attn_sconv: InklingShortConv,
        mlp_sconv: InklingShortConv,
    }

    impl InklingBlock {
        fn load(
            layer_idx: usize,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let prefix = format!("model.llm.layers.{layer_idx}");
            // local_layer_ids lists the sliding (local) layers; everything else is global.
            let local_ids: Vec<i64> = config
                .raw
                .get("local_layer_ids")
                .and_then(serde_json::Value::as_array)
                .map(|a| a.iter().filter_map(serde_json::Value::as_i64).collect())
                .unwrap_or_default();
            let is_sliding = local_ids.contains(&(layer_idx as i64));
            // Dense vs sparse: a layer is dense iff it has a plain mlp.gate_proj rather than
            // stacked experts (dense_mlp_idx marks it, but the tensors are authoritative).
            let dense = arrays.contains_key(&format!("{prefix}.mlp.gate_proj.weight"))
                && !arrays.contains_key(&format!("{prefix}.mlp.experts.gate_proj.weight"));
            let h = config.hidden_size as i32;
            let kernel = config
                .raw
                .get("sconv_kernel_size")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(4) as i32;
            let _ = kernel;
            Ok(Self {
                attn: InklingAttention::load(
                    &format!("{prefix}.attn"),
                    arrays,
                    config,
                    is_sliding,
                )?,
                attn_norm: RmsNorm::load(
                    &format!("{prefix}.attn_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                mlp_norm: RmsNorm::load(
                    &format!("{prefix}.mlp_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                ffn: if dense {
                    InklingFfn::Dense(InklingDenseMlp::load(
                        &format!("{prefix}.mlp"),
                        arrays,
                        config,
                    )?)
                } else {
                    InklingFfn::Moe(Box::new(InklingMoe::load(
                        &format!("{prefix}.mlp"),
                        arrays,
                        config,
                        stream_ctx,
                    )?))
                },
                attn_sconv: InklingShortConv::load(&format!("{prefix}.attn_sconv"), arrays, h)?,
                mlp_sconv: InklingShortConv::load(&format!("{prefix}.mlp_sconv"), arrays, h)?,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            // attn sublayer: pre-norm → attn → short conv → residual
            let h = self.attn.forward(&self.attn_norm.forward(&x)?)?;
            let h = self.attn_sconv.forward(&h)?;
            let x = x + h;
            // mlp sublayer: pre-norm → mlp → short conv → residual
            let normed = self.mlp_norm.forward(&x)?;
            let h = match &self.ffn {
                InklingFfn::Dense(m) => m.forward(&normed)?,
                InklingFfn::Moe(m) => m.forward(&normed)?,
            };
            let h = self.mlp_sconv.forward(&h)?;
            Ok(x + h)
        }

        fn reset(&mut self) {
            self.attn.reset();
            self.attn_sconv.reset();
            self.mlp_sconv.reset();
        }

        fn prepare(&mut self, capacity: i32) {
            self.attn.prepare(capacity);
        }
    }

    struct InklingLike {
        embed: Embedding,
        embed_norm: RmsNorm,
        layers: Vec<InklingBlock>,
        norm: RmsNorm,
        unembed: Linear,
        logits_mup: f32,
        unpadded_vocab: Option<i32>,
        // Loaded when the checkpoint carries them.
        vision_tower: Option<InklingVisionTower>,
        audio_tower: Option<InklingAudioTower>,
        // Media staged by set_media, scattered into the first (prefill) forward then cleared.
        pending_media: Option<MediaFeatures>,
    }

    impl InklingLike {
        fn new(
            config: MlxModelConfig,
            arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let layers = (0..config.num_hidden_layers as usize)
                .map(|idx| InklingBlock::load(idx, &arrays, &config, stream_ctx))
                .collect::<Result<Vec<_>>>()?;
            let logits_mup = config
                .raw
                .get("logits_mup_width_multiplier")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0) as f32;
            let unpadded_vocab = config
                .raw
                .get("unpadded_vocab_size")
                .and_then(serde_json::Value::as_i64)
                .map(|v| v as i32);
            // Load the media towers if their weights are present (they are small next to the text
            // tower). Absent them, this is a plain text model.
            let vision_tower = arrays
                .contains_key("model.visual.final_norm.weight")
                .then(|| InklingVisionTower::load(&arrays, &config))
                .transpose()?;
            let audio_tower = arrays
                .contains_key("model.audio.final_norm.weight")
                .then(|| InklingAudioTower::load(&arrays, &config))
                .transpose()?;
            let me = Self {
                embed: Embedding::load("model.llm.embed", &arrays, &config)?,
                embed_norm: RmsNorm::load(
                    "model.llm.embed_norm.weight",
                    &arrays,
                    config.rms_norm_eps,
                )?,
                norm: RmsNorm::load("model.llm.norm.weight", &arrays, config.rms_norm_eps)?,
                unembed: Linear::load("model.llm.unembed", &arrays, &config)?,
                logits_mup,
                unpadded_vocab,
                layers,
                vision_tower,
                audio_tower,
                pending_media: None,
            };
            Ok(me)
        }
    }

    /// Replace the rows of `embeds` (`[1, L, H]`) whose `input_ids` equal `token_id` with `features`
    /// (in sequence order). Placeholder positions are resolved on the host, matching the reference.
    fn scatter_features(
        embeds: &Array,
        input_ids: &[u32],
        token_id: u32,
        features: &Array,
    ) -> Result<Array> {
        let positions: Vec<i32> = input_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == token_id)
            .map(|(i, _)| i as i32)
            .collect();
        if positions.is_empty() {
            return Ok(embeds.clone());
        }
        let shape = embeds.shape().to_vec();
        let h = *shape.last().unwrap();
        let mut flat = embeds.reshape(&[-1, h])?;
        let idx = Array::from_slice(&positions, &[positions.len() as i32]);
        let feats = features.as_dtype(flat.dtype())?;
        flat = put_along_axis(
            &flat,
            &idx.reshape(&[positions.len() as i32, 1])?,
            &feats,
            0,
        )?;
        flat.reshape(&shape).map_err(Into::into)
    }

    impl CausalLm for InklingLike {
        fn set_media(&mut self, media: MediaFeatures) {
            self.pending_media = (!media.is_empty()).then_some(media);
        }

        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            // Incremental: each layer keeps a KV cache and four short-conv states, so only the new
            // tokens are processed and they still see the full history.
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let mut h = self.embed_norm.forward(&self.embed.forward(&ids)?)?;
            // On the prefill forward, encode staged media through the towers and scatter the results
            // into the token-embedding stream at the placeholder positions, then clear it.
            if let Some(media) = self.pending_media.take() {
                let px = media
                    .pixel_values
                    .as_ref()
                    .map(|(d, s)| Array::from_slice(d, s))
                    .and_then(|a| self.vision_tower.as_ref().map(|t| t.forward(&a)))
                    .transpose()?;
                let au = media
                    .audio_ids
                    .as_ref()
                    .map(|(d, s)| Array::from_slice(d, s))
                    .and_then(|a| self.audio_tower.as_ref().map(|t| t.forward(&a)))
                    .transpose()?;
                if let Some(feats) = px {
                    h = scatter_features(&h, input_ids, media.image_token_id, &feats)?;
                }
                if let Some(feats) = au {
                    h = scatter_features(&h, input_ids, media.audio_token_id, &feats)?;
                }
            }
            for layer in &mut self.layers {
                h = layer.forward(h)?;
            }
            h = self.norm.forward(&h)?;
            // muP logit scaling: divide hidden by the width multiplier before the unembed.
            if self.logits_mup != 1.0 {
                h = h / self.logits_mup;
            }
            let mut logits = self.unembed.forward(&h)?;
            // Trim the vocab padding so sampling never picks an unused id.
            if let Some(uv) = self.unpadded_vocab {
                if uv < logits.shape()[logits.shape().len() - 1] {
                    let l = logits.shape()[1];
                    logits = logits.index((.., 0..l, 0..uv));
                }
            }
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn reset_cache(&mut self) {
            for layer in &mut self.layers {
                layer.reset();
            }
        }

        fn prepare_cache(&mut self, capacity: i32) {
            for layer in &mut self.layers {
                layer.prepare(capacity);
            }
        }
    }

    // ---------------------- Inkling vision / audio towers ----------------------
    // The two non-text towers, ported from inkling_mlx/vision.py and audio.py. Both were checked
    // bit-exact against the reference. They embed media into the text hidden size so their outputs
    // can be scattered into the token-embedding stream at <|image|>/<|audio|> placeholder positions
    // (see model.py::_scatter_features). The serving path drives them from `set_media` + the prefill
    // `forward` media branch: `inkling_media` preprocesses image/audio, `build_inkling_multimodal`
    // interleaves the placeholder soft-tokens, and `scatter_features` splices the tower outputs in.

    /// Inkling audio tower: each frame is `n_mel_bins` discretized bins in `[0, mel_vocab_size)`;
    /// each bin indexes its own slice of a shared table (offset `bin * mel_vocab_size`) and the
    /// per-bin embeddings are summed, then normed.
    struct InklingAudioTower {
        encoder: Embedding,
        final_norm: RmsNorm,
        offsets: Array, // [n_mel_bins] = arange(n_mel_bins) * mel_vocab_size
    }

    impl InklingAudioTower {
        fn load(arrays: &HashMap<String, Array>, config: &MlxModelConfig) -> Result<Self> {
            let audio = config
                .raw
                .get("audio_config")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let ai = |k: &str, d: i64| {
                audio
                    .get(k)
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(d)
            };
            let n_mel_bins = ai("n_mel_bins", 80) as i32;
            let mel_vocab_size = ai("mel_vocab_size", 16) as i32;
            let offsets: Vec<i32> = (0..n_mel_bins).map(|b| b * mel_vocab_size).collect();
            Ok(Self {
                encoder: Embedding::load("model.audio.encoder", arrays, config)?,
                final_norm: RmsNorm::load(
                    "model.audio.final_norm.weight",
                    arrays,
                    config.rms_norm_eps,
                )?,
                offsets: Array::from_slice(&offsets, &[n_mel_bins]),
            })
        }

        /// `audio_ids`: `[.., n_mel_bins]` of bin indices. Returns `[.., text_hidden]`.
        fn forward(&self, audio_ids: &Array) -> Result<Array> {
            let ids = audio_ids + &self.offsets;
            let embeds = self.encoder.forward(&ids)?; // [.., n_mel_bins, hidden]
            let ndim = embeds.shape().len() as i32;
            let summed = sum_axis(&embeds, ndim - 2, Some(false))?; // sum over bins
            self.final_norm.forward(&summed)
        }
    }

    /// Inkling vision tower: an attention-free hierarchical MLP. Each layer folds a space/time block
    /// into the channel dim (`_fold_timespace_to_depth`) then projects (Linear, then RMSNorm+GELU on
    /// all but the last). The fold factors are derived from the loaded projection widths and the
    /// patch size: while the cumulative spatial fold is below `patch_size` a perfect-square shuffle is
    /// a spatial (hw) fold, after which the remaining folds are temporal — matching plan_out_scales.
    struct InklingVisionLayer {
        linear: Linear,
        norm: Option<RmsNorm>,
        t_fold: i32,
        hw_fold: i32,
    }

    struct InklingVisionTower {
        layers: Vec<InklingVisionLayer>,
        final_norm: RmsNorm,
    }

    impl InklingVisionTower {
        fn load(arrays: &HashMap<String, Array>, config: &MlxModelConfig) -> Result<Self> {
            let vision = config
                .raw
                .get("vision_config")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let vi = |k: &str, d: i64| {
                vision
                    .get(k)
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(d)
            };
            let n_layers = vi("n_layers", 4) as usize;
            let n_channels = vi("n_channels", 3) as i32;
            let patch_size = vi("patch_size", 40) as i32;
            let mut layers = Vec::with_capacity(n_layers);
            let mut prev_channels = n_channels;
            let mut cum_hw = 1i32;
            for i in 0..n_layers {
                let prefix = format!("model.visual.layers.linear_{i}");
                let linear = Linear::load(&prefix, arrays, config)?;
                let in_dim = linear.in_features();
                let out_dim = linear_out_features(&linear);
                let shuffle = (in_dim / prev_channels).max(1);
                // Perfect-square shuffle before the spatial budget is exhausted → spatial fold;
                // otherwise temporal.
                let root = (shuffle as f64).sqrt().round() as i32;
                let (t_fold, hw_fold) = if root * root == shuffle && cum_hw < patch_size {
                    cum_hw *= root;
                    (1, root)
                } else {
                    (shuffle, 1)
                };
                let norm = arrays
                    .get(&format!("model.visual.layers.norm_{i}.weight"))
                    .is_some()
                    .then(|| {
                        RmsNorm::load(
                            &format!("model.visual.layers.norm_{i}.weight"),
                            arrays,
                            config.rms_norm_eps,
                        )
                    })
                    .transpose()?;
                layers.push(InklingVisionLayer {
                    linear,
                    norm,
                    t_fold,
                    hw_fold,
                });
                prev_channels = out_dim;
            }
            Ok(Self {
                layers,
                final_norm: RmsNorm::load(
                    "model.visual.final_norm.weight",
                    arrays,
                    config.rms_norm_eps,
                )?,
            })
        }

        /// `pixel_values`: `[num_patches, T, H, W, C]`. Returns `[num_patches, text_hidden]`.
        fn forward(&self, pixel_values: &Array) -> Result<Array> {
            let num_patches = pixel_values.shape()[0];
            let mut x = pixel_values.clone();
            for layer in &self.layers {
                if layer.hw_fold > 1 || layer.t_fold > 1 {
                    x = fold_timespace_to_depth(&x, layer.t_fold, layer.hw_fold)?;
                }
                x = layer.linear.forward(&x)?;
                if let Some(norm) = &layer.norm {
                    x = gelu(&norm.forward(&x)?)?;
                }
            }
            let x = self.final_norm.forward(&x)?;
            x.reshape(&[num_patches, -1]).map_err(Into::into)
        }
    }

    fn linear_out_features(linear: &Linear) -> i32 {
        match linear {
            Linear::Dense { weight, .. } => weight.shape()[0],
            Linear::Quantized { weight, .. } => weight.shape()[0],
        }
    }

    /// `[B, T, H, W, C] -> [B, T/t, H/hw, W/hw, C*t*hw*hw]`, matching the reference fold.
    #[allow(dead_code)]
    fn fold_timespace_to_depth(x: &Array, t_fold: i32, hw_fold: i32) -> Result<Array> {
        let s = x.shape();
        let (b, t, h, w, c) = (s[0], s[1], s[2], s[3], s[4]);
        let (tn, hn, wn) = (t / t_fold, h / hw_fold, w / hw_fold);
        x.reshape(&[b, tn, t_fold, hn, hw_fold, wn, hw_fold, c])?
            .transpose_axes(&[0, 1, 3, 5, 2, 4, 6, 7])?
            .reshape(&[b, tn, hn, wn, t_fold * hw_fold * hw_fold * c])
            .map_err(Into::into)
    }

    // ---------------------- Laguna (poolside Laguna-S) ----------------------
    // Qwen3-MoE with five departures, all handled here:
    //   * the query-head count varies by layer type (48 on full-attention layers, 72 on sliding);
    //     KV heads stay constant, so q/o/g projections are sized per layer
    //   * softplus attention output gating, one gate per head, applied before o_proj
    //   * per-head q/k RMSNorm before rope
    //   * interleaved full / sliding-window(512) attention
    //   * two ropes: full layers use YaRN over half the head dims (partial_rotary 0.5, theta 5e5,
    //     factor 128); sliding layers use plain rope over all dims (theta 1e4)
    // The MoE itself is Qwen/Hy3-shaped — sigmoid router with an aux-loss-free correction bias plus
    // an always-on shared expert — so it reuses QwenMoe. Layer 0 is a dense MLP (mlp_only_layers).

    /// Rope parameters for one Laguna attention type, read from the nested `rope_parameters`.
    /// Returns `(rotated_dims, base, freqs, mscale)`. YaRN supplies per-dim `freqs` and leaves
    /// `base` unset (MLX rejects both together); mscale scales only the rotated dims, matching
    /// mlx-lm's YarnRoPE, so it cannot be folded into the attention scale under partial rotary.
    fn laguna_rope(
        config: &MlxModelConfig,
        head_dim: i32,
        is_sliding: bool,
    ) -> Result<(i32, Option<f32>, Option<Array>, f32)> {
        let key = if is_sliding {
            "sliding_attention"
        } else {
            "full_attention"
        };
        let cfg = config
            .raw
            .get("rope_parameters")
            .and_then(|rp| rp.get(key).or(Some(rp)))
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let getf = |k: &str, d: f64| cfg.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
        let theta = getf("rope_theta", config.rope_theta as f64);
        let partial = getf("partial_rotary_factor", 1.0);
        let dims = ((head_dim as f64 * partial) as i32).max(2) & !1; // even
        let rope_type = cfg
            .get("rope_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default");
        if rope_type != "yarn" {
            return Ok((dims, Some(theta as f32), None, 1.0));
        }
        let factor = getf("factor", 1.0);
        if factor <= 1.0 {
            return Ok((dims, Some(theta as f32), None, 1.0));
        }
        let beta_fast = getf("beta_fast", 32.0);
        let beta_slow = getf("beta_slow", 1.0);
        let orig_max = getf("original_max_position_embeddings", 4096.0);
        let half = (dims / 2) as usize;
        let find_dim = |rot: f64| {
            dims as f64 * (orig_max / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * theta.ln())
        };
        let low = find_dim(beta_fast).floor().max(0.0);
        let high = find_dim(beta_slow).ceil().min((half.max(1) - 1) as f64);
        let denom = (high - low).max(1e-3);
        let freqs: Vec<f32> = (0..half)
            .map(|i| {
                let extra = theta.powf(2.0 * i as f64 / dims as f64); // theta per dim
                let inter = extra * factor; // interpolated
                let ramp = (((i as f64) - low) / denom).clamp(0.0, 1.0);
                let mask = 1.0 - ramp;
                // mlx-lm: (inter*extra) / (inter*mask + extra*(1-mask))
                ((inter * extra) / (inter * mask + extra * (1.0 - mask))) as f32
            })
            .collect();
        // HF's attention_factor: 0.1*ln(factor)+1, applied to the rotated dims only.
        let mscale = 0.1 * factor.ln() + 1.0;
        Ok((
            dims,
            None,
            Some(Array::from_slice(&freqs, &[half as i32])),
            mscale as f32,
        ))
    }

    struct LagunaAttention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        g_proj: Option<Linear>,
        gate_per_head: bool,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        rot_dims: i32,
        rope_base: Option<f32>,
        rope_freqs: Option<Array>,
        /// Per-dim multiplier: `mscale` over the rotated dims, 1.0 over the pass-through tail.
        mscale_vec: Option<Array>,
        cache: Cache,
    }

    impl LagunaAttention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            layer_idx: usize,
            is_sliding: bool,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            let n_heads = config
                .num_attention_heads_per_layer
                .get(layer_idx)
                .copied()
                .unwrap_or(config.num_attention_heads) as i32;
            let (rot_dims, rope_base, rope_freqs, mscale) =
                laguna_rope(config, head_dim, is_sliding)?;
            let mscale_vec = (mscale != 1.0).then(|| {
                let mut v = vec![1.0f32; head_dim as usize];
                for slot in v.iter_mut().take(rot_dims as usize) {
                    *slot = mscale;
                }
                Array::from_slice(&v, &[head_dim])
            });
            let gating = config.attention_gating.as_deref();
            let gate_per_head = gating == Some("per-head") || gating == Some("per_head");
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: Linear::load(&format!("{prefix}.v_proj"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                g_proj: match gating {
                    Some(_) => Some(Linear::load(&format!("{prefix}.g_proj"), arrays, config)?),
                    None => None,
                },
                gate_per_head,
                q_norm: RmsNorm::load(
                    &format!("{prefix}.q_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                k_norm: RmsNorm::load(
                    &format!("{prefix}.k_norm.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?,
                n_heads,
                n_kv_heads: config.num_key_value_heads as i32,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
                rot_dims,
                rope_base,
                rope_freqs,
                mscale_vec,
                // Sliding layers keep only the window in the KV cache.
                cache: if is_sliding {
                    Cache::with_max_len(config.sliding_window.map(|w| w as i32))
                } else {
                    Cache::new()
                },
            })
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l) = (shape[0], shape[1]);
            let q = self
                .q_proj
                .forward(x)?
                .reshape(&[b, l, self.n_heads, self.head_dim])?;
            let k = self
                .k_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?;
            // Per-head RMSNorm over head_dim, before rope.
            let mut q = self.q_norm.forward(&q)?.transpose_axes(&[0, 2, 1, 3])?;
            let mut k = self.k_norm.forward(&k)?.transpose_axes(&[0, 2, 1, 3])?;
            let v = self
                .v_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            if let Some(m) = &self.mscale_vec {
                q = q * m;
                k = k * m;
            }
            let offset = self.cache.offset;
            q = rope(
                q,
                self.rot_dims,
                false,
                self.rope_base,
                1.0,
                offset,
                self.rope_freqs.as_ref(),
            )?;
            k = rope(
                k,
                self.rot_dims,
                false,
                self.rope_base,
                1.0,
                offset,
                self.rope_freqs.as_ref(),
            )?;
            let (k, v) = self.cache.update(k, v)?;
            let out = if l > 1 && offset == 0 {
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    ScaledDotProductAttentionMask::Causal,
                    None::<&Array>,
                )?
            } else if l > 1 {
                let mask = causal_attention_mask(l, k.shape()[2], offset);
                scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    self.scale,
                    ScaledDotProductAttentionMask::Array(&mask),
                    None::<&Array>,
                )?
            } else {
                scaled_dot_product_attention(&q, &k, &v, self.scale, None, None::<&Array>)?
            };
            let out = out.transpose_axes(&[0, 2, 1, 3])?;
            // Softplus gate before o_proj, computed in f32 like the reference.
            let out = match &self.g_proj {
                Some(g_proj) => {
                    let g = softplus(&g_proj.forward(x)?.as_type::<f32>()?)?;
                    if self.gate_per_head {
                        let g = g.reshape(&[b, l, self.n_heads, 1])?.as_type::<f32>()?;
                        (out.as_type::<f32>()? * g).as_dtype(x.dtype())?
                    } else {
                        let g = g.reshape(&[b, l, self.n_heads, self.head_dim])?;
                        (out.as_type::<f32>()? * g).as_dtype(x.dtype())?
                    }
                }
                None => out,
            };
            let out = out.reshape(&[b, l, self.n_heads * self.head_dim])?;
            self.o_proj.forward(&out)
        }
    }

    enum LagunaFfn {
        Dense(Mlp),
        Moe(Box<QwenMoe>),
    }

    struct LagunaBlock {
        attention: LagunaAttention,
        input_layernorm: RmsNorm,
        post_attention_layernorm: RmsNorm,
        ffn: LagunaFfn,
    }

    impl LagunaBlock {
        fn load(
            layer_idx: usize,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let prefix = format!("model.layers.{layer_idx}");
            let is_sliding = config
                .layer_types
                .get(layer_idx)
                .map(|t| t == "sliding_attention")
                .unwrap_or(false);
            // `mlp_only_layers` keeps layer 0 dense; every other layer is sparse when the router is
            // present (decoder_sparse_step is 1 for Laguna).
            let dense = config.mlp_only_layers.contains(&(layer_idx as u32))
                || config.n_routed_experts.unwrap_or(0) == 0
                || !arrays.contains_key(&format!("{prefix}.mlp.gate.weight"));
            Ok(Self {
                attention: LagunaAttention::load(
                    &format!("{prefix}.self_attn"),
                    arrays,
                    config,
                    layer_idx,
                    is_sliding,
                )?,
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
                ffn: if dense {
                    LagunaFfn::Dense(Mlp::load(&format!("{prefix}.mlp"), arrays, config)?)
                } else {
                    LagunaFfn::Moe(Box::new(QwenMoe::load(
                        &format!("{prefix}.mlp"),
                        arrays,
                        config,
                        stream_ctx,
                    )?))
                },
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let normed = self.input_layernorm.forward(&x)?;
            let h = x + self.attention.forward(&normed)?;
            let normed = self.post_attention_layernorm.forward(&h)?;
            let ffn_out = match &mut self.ffn {
                LagunaFfn::Dense(mlp) => mlp.forward(&normed)?,
                LagunaFfn::Moe(moe) => moe.forward(&normed)?,
            };
            Ok(h + ffn_out)
        }
    }

    struct LagunaLike {
        embed_tokens: Embedding,
        layers: Vec<LagunaBlock>,
        norm: RmsNorm,
        lm_head: Option<Linear>,
    }

    impl LagunaLike {
        fn new(
            config: MlxModelConfig,
            arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let layers = (0..config.num_hidden_layers as usize)
                .map(|idx| LagunaBlock::load(idx, &arrays, &config, stream_ctx))
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                embed_tokens: Embedding::load("model.embed_tokens", &arrays, &config)?,
                norm: RmsNorm::load("model.norm.weight", &arrays, config.rms_norm_eps)?,
                lm_head: if config.tie_word_embeddings {
                    None
                } else {
                    Some(Linear::load("lm_head", &arrays, &config)?)
                },
                layers,
            })
        }
    }

    impl CausalLm for LagunaLike {
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

    // ---------------------- Gemma-4 (gemma4_text) ----------------------
    // Per-layer sliding/full attention hybrid: full-attention layers use a wider head_dim, fewer
    // KV heads, k==v (no v_proj), and a proportional partial-rotary RoPE (theta 1e6, 25% rotated);
    // sliding layers use full-rotary RoPE (theta 1e4). Each block has q/k head-norms + a weightless
    // v-norm, four sandwich norms, a GeGLU MLP, and a learned per-layer scalar. Embeddings are scaled
    // by sqrt(hidden) and tied to the output; final logits are tanh-softcapped. The 31B/26B disable
    // KV-sharing and per-layer-input gating. NOTE: sliding layers use a plain causal mask here, so
    // outputs are exact only for contexts up to `sliding_window` (1024); longer contexts diverge.
    // Gemma norm weights: Gemma-2/3 store them as deviations (mlx_lm applies `rms_norm(x, 1 + weight)`),
    // while the pipenetwork Gemma-4 export folds the +1 into the stored weight. Add it back only for the
    // deviation convention (gemma2/gemma3) — for gemma4 the weight is already the effective scale.
    fn gemma_norm(
        key: &str,
        arrays: &HashMap<String, Array>,
        config: &MlxModelConfig,
    ) -> Result<RmsNorm> {
        let weight = raw_array(arrays, key)?;
        let weight = if config.model_type.starts_with("gemma4") {
            weight
        } else {
            weight + 1.0f32
        };
        Ok(RmsNorm {
            weight,
            eps: config.rms_norm_eps,
        })
    }

    struct Gemma4Attention {
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Option<Linear>,
        o_proj: Linear,
        // Gemma-3/4 have per-head qk-norm; Gemma-2 does not.
        q_norm: Option<RmsNorm>,
        k_norm: Option<RmsNorm>,
        v_ones: Array,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rope_theta: f32,
        rope_freqs: Option<Array>,
        eps: f32,
        // sdpa scale: Gemma-4 folds query_pre_attn scaling into q_norm (scale 1.0); Gemma-3 applies
        // query_pre_attn_scalar^-0.5 here.
        scale: f32,
        cache: Cache,
    }

    impl Gemma4Attention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            is_sliding: bool,
        ) -> Result<Self> {
            let n_heads = config.num_attention_heads as i32;
            let q_out = raw_array(arrays, &format!("{prefix}.q_proj.weight"))?.shape()[0];
            // Quantized weights are packed on the last axis; head_dim comes from q_out / n_heads.
            let head_dim = q_out / n_heads;
            let k_out = raw_array(arrays, &format!("{prefix}.k_proj.weight"))?.shape()[0];
            let n_kv_heads = (k_out / head_dim).max(1);
            let has_v = arrays.contains_key(&format!("{prefix}.v_proj.weight"))
                || arrays.contains_key(&format!("{prefix}.v_proj.scales"));
            // RoPE: sliding = full rotary (theta 1e4); full = proportional partial rotary (theta 1e6,
            // 25% of head_dim rotated; freqs = base^(2i/head_dim), inf for the unrotated tail).
            let (rope_theta, rope_freqs) = if is_sliding {
                // Sliding layers: full rotary at the local base (Gemma-3 rope_local_base_freq; 1e4).
                let local = config
                    .raw
                    .get("rope_local_base_freq")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(10_000.0) as f32;
                (local, None)
            } else if config.model_type.starts_with("gemma4") {
                // Gemma-4 full-attention layers: partial rotary (25% of head_dim rotated, theta 1e6).
                let base = 1_000_000.0f32;
                let half = head_dim / 2;
                let rot_half = (head_dim / 4) / 2; // partial_rotary_factor 0.25 -> rotated dims / 2
                let mut freqs = Vec::with_capacity(half as usize);
                for i in 0..rot_half {
                    freqs.push(base.powf((2 * i) as f32 / head_dim as f32));
                }
                for _ in rot_half..half {
                    freqs.push(f32::INFINITY);
                }
                (base, Some(Array::from_slice(&freqs, &[half])))
            } else {
                // Gemma-3 full-attention layers: full rotary at rope_theta (1e6).
                (config.rope_theta, None)
            };
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: if has_v {
                    Some(Linear::load(&format!("{prefix}.v_proj"), arrays, config)?)
                } else {
                    None
                },
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                q_norm: arrays
                    .contains_key(&format!("{prefix}.q_norm.weight"))
                    .then(|| gemma_norm(&format!("{prefix}.q_norm.weight"), arrays, config))
                    .transpose()?,
                k_norm: arrays
                    .contains_key(&format!("{prefix}.k_norm.weight"))
                    .then(|| gemma_norm(&format!("{prefix}.k_norm.weight"), arrays, config))
                    .transpose()?,
                v_ones: Array::ones::<f32>(&[head_dim])?,
                n_heads,
                n_kv_heads,
                head_dim,
                rope_theta,
                rope_freqs,
                eps: config.rms_norm_eps,
                scale: if config.model_type.starts_with("gemma4") {
                    1.0
                } else {
                    let s = config
                        .raw
                        .get("query_pre_attn_scalar")
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .unwrap_or(head_dim as f32);
                    s.powf(-0.5)
                },
                cache: Cache::new(),
            })
        }

        fn rope_apply(&self, x: &Array, offset: i32) -> Result<Array> {
            // MLX rope rejects base+freqs together: pass custom freqs (full layers, base ignored) or
            // the base theta (sliding layers).
            match &self.rope_freqs {
                Some(freqs) => Ok(rope(
                    x,
                    self.head_dim,
                    false,
                    None::<f32>,
                    1.0,
                    offset,
                    Some(freqs),
                )?),
                None => Ok(rope(
                    x,
                    self.head_dim,
                    false,
                    self.rope_theta,
                    1.0,
                    offset,
                    None::<&Array>,
                )?),
            }
        }

        fn forward(&mut self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l) = (shape[0], shape[1]);
            let offset = self.cache.offset;
            let q = self
                .q_proj
                .forward(x)?
                .reshape(&[b, l, self.n_heads, self.head_dim])?;
            let q = match &self.q_norm {
                Some(n) => n.forward(&q)?,
                None => q,
            };
            let q = q.transpose_axes(&[0, 2, 1, 3])?;
            let q = self.rope_apply(&q, offset)?;
            let k_raw = self
                .k_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?;
            let k = match &self.k_norm {
                Some(n) => n.forward(&k_raw)?,
                None => k_raw.clone(),
            };
            let k = k.transpose_axes(&[0, 2, 1, 3])?;
            let k = self.rope_apply(&k, offset)?;
            // Full-attention layers reuse the K projection as V (k==v), then apply a weightless v-norm.
            let v_raw = match &self.v_proj {
                Some(vp) => vp
                    .forward(x)?
                    .reshape(&[b, l, self.n_kv_heads, self.head_dim])?,
                None => k_raw,
            };
            let v = rms_norm(&v_raw, &self.v_ones, self.eps)?.transpose_axes(&[0, 2, 1, 3])?;
            let (k, v) = self.cache.update(k, v)?;
            let scale = self.scale;
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

    struct Gemma4Mlp {
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    }

    impl Gemma4Mlp {
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
            // GeGLU: down(gelu_approx(gate(x)) * up(x)).
            let gate = gelu_approximate(&self.gate_proj.forward(x)?)?;
            self.down_proj.forward(&(gate * self.up_proj.forward(x)?))
        }
    }

    struct Gemma4Block {
        input_ln: RmsNorm,
        post_attn_ln: RmsNorm,
        pre_ff_ln: RmsNorm,
        post_ff_ln: RmsNorm,
        attn: Gemma4Attention,
        mlp: Gemma4Mlp,
        // Gemma-4 per-layer output scalar; absent in Gemma-3.
        layer_scalar: Option<Array>,
    }

    impl Gemma4Block {
        fn load(
            idx: u32,
            is_sliding: bool,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let p = format!("model.layers.{idx}");
            let _eps = config.rms_norm_eps;
            Ok(Self {
                input_ln: gemma_norm(&format!("{p}.input_layernorm.weight"), arrays, config)?,
                post_attn_ln: gemma_norm(
                    &format!("{p}.post_attention_layernorm.weight"),
                    arrays,
                    config,
                )?,
                pre_ff_ln: gemma_norm(
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    arrays,
                    config,
                )?,
                post_ff_ln: gemma_norm(
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    arrays,
                    config,
                )?,
                attn: Gemma4Attention::load(&format!("{p}.self_attn"), arrays, config, is_sliding)?,
                mlp: Gemma4Mlp::load(&format!("{p}.mlp"), arrays, config)?,
                layer_scalar: arrays
                    .get(&format!("{p}.layer_scalar"))
                    .map(|a| a.as_type::<f32>())
                    .transpose()?,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let residual = x.clone();
            let h = self.input_ln.forward(&x)?;
            let h = self.attn.forward(&h)?;
            let h = self.post_attn_ln.forward(&h)?;
            let h = residual + h;
            let residual = h.clone();
            let ff = self.pre_ff_ln.forward(&h)?;
            let ff = self.mlp.forward(&ff)?;
            let ff = self.post_ff_ln.forward(&ff)?;
            let h = residual + ff;
            match &self.layer_scalar {
                Some(s) => Ok(h * s),
                None => Ok(h),
            }
        }
    }

    struct Gemma4TextLike {
        embed: Embedding,
        embed_scale: f32,
        blocks: Vec<Gemma4Block>,
        norm: RmsNorm,
        final_softcap: Option<f32>,
    }

    impl Gemma4TextLike {
        fn new(config: MlxModelConfig, arrays: HashMap<String, Array>) -> Result<Self> {
            // Gemma-3 gives an interval (`sliding_window_pattern`) rather than an explicit list: every
            // P-th layer is full attention, the rest sliding.
            let layer_types: Vec<String> = if config.layer_types.is_empty() {
                let pattern = config
                    .raw
                    .get("sliding_window_pattern")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(6)
                    .max(1) as u32;
                (0..config.num_hidden_layers)
                    .map(|i| {
                        if (i + 1) % pattern == 0 {
                            "full_attention".to_string()
                        } else {
                            "sliding_attention".to_string()
                        }
                    })
                    .collect()
            } else {
                config.layer_types.clone()
            };
            let blocks = layer_types
                .iter()
                .enumerate()
                .map(|(idx, kind)| {
                    Gemma4Block::load(idx as u32, kind == "sliding_attention", &arrays, &config)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                embed: Embedding::load("model.embed_tokens", &arrays, &config)?,
                embed_scale: (config.hidden_size as f32).sqrt(),
                norm: gemma_norm("model.norm.weight", &arrays, &config)?,
                final_softcap: config.final_logit_softcapping,
                blocks,
            })
        }
    }

    impl CausalLm for Gemma4TextLike {
        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let mut h = self.embed.forward(&ids)? * self.embed_scale;
            for block in &mut self.blocks {
                h = block.forward(h)?;
            }
            h = self.norm.forward(&h)?;
            let mut logits = self.embed.as_linear(&h)?;
            if let Some(cap) = self.final_softcap {
                let s = Array::from_f32(cap);
                logits = tanh(&(logits / &s))? * &s;
            }
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn reset_cache(&mut self) {
            for block in &mut self.blocks {
                block.attn.cache.reset();
            }
        }

        fn prepare_cache(&mut self, capacity: i32) {
            for block in &mut self.blocks {
                block.attn.cache.prepare_capacity(capacity);
            }
        }
    }

    // ---------------------- MiniMax-M3 (minimax_m3) ----------------------
    // GQA (partial rotary + per-head q/k norms) + a DeepSeek-style sigmoid/noaux MoE (`block_sparse_moe`)
    // above `first_k_dense_replace` dense layers, each MoE layer with a shared expert. FFNs use the
    // SwiGLU-OAI (GPT-OSS-style) clamped activation: clamp gate<=limit, up to +/-limit, then
    // (up+1)*gate*sigmoid(alpha*gate). Routing weights scaled by routed_scaling_factor. `model.` prefix.

    // MiniMax RMSNorms use the Gemma/T5 (1 + weight) convention (stored weights are deviations from 1).
    fn minimax_norm(key: &str, arrays: &HashMap<String, Array>, eps: f32) -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: raw_array(arrays, key)? + 1.0f32,
            eps,
        })
    }

    // SwiGLU-OAI activation (swiglu_alpha=1.702, swiglu_limit=7.0).
    fn swiglu_oai(gate: &Array, up: &Array, alpha: f32, limit: f32) -> Result<Array> {
        let hi = Array::from_f32(limit);
        let g = minimum(gate, &hi)?;
        let u = maximum(&minimum(up, &hi)?, &Array::from_f32(-limit))?;
        let glu = &g * sigmoid(&(&g * alpha))?;
        Ok((&u + 1.0f32) * glu)
    }

    struct MiniMaxMlp {
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
        alpha: f32,
        limit: f32,
    }

    impl MiniMaxMlp {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            Ok(Self {
                gate_proj: Linear::load(&format!("{prefix}.gate_proj"), arrays, config)?,
                up_proj: Linear::load(&format!("{prefix}.up_proj"), arrays, config)?,
                down_proj: Linear::load(&format!("{prefix}.down_proj"), arrays, config)?,
                alpha: 1.702,
                limit: config.swiglu_limit.unwrap_or(7.0),
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let g = self.gate_proj.forward(x)?;
            let u = self.up_proj.forward(x)?;
            self.down_proj
                .forward(&swiglu_oai(&g, &u, self.alpha, self.limit)?)
        }
    }

    struct MiniMaxAttention {
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

    impl MiniMaxAttention {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
        ) -> Result<Self> {
            let head_dim = config.attention_head_dim() as i32;
            // Per-head q/k RMSNorm over head_dim (when present), (1 + weight) convention.
            let load_norm = |name: &str| -> Result<Option<RmsNorm>> {
                if arrays.contains_key(&format!("{prefix}.{name}.weight")) {
                    Ok(Some(minimax_norm(
                        &format!("{prefix}.{name}.weight"),
                        arrays,
                        config.rms_norm_eps,
                    )?))
                } else {
                    Ok(None)
                }
            };
            Ok(Self {
                q_proj: Linear::load(&format!("{prefix}.q_proj"), arrays, config)?,
                k_proj: Linear::load(&format!("{prefix}.k_proj"), arrays, config)?,
                v_proj: Linear::load(&format!("{prefix}.v_proj"), arrays, config)?,
                o_proj: Linear::load(&format!("{prefix}.o_proj"), arrays, config)?,
                q_norm: load_norm("q_norm")?,
                k_norm: load_norm("k_norm")?,
                n_heads: config.num_attention_heads as i32,
                n_kv_heads: config.num_key_value_heads as i32,
                head_dim,
                rot_dims: config.rotary_dim.map(|d| d as i32).unwrap_or(head_dim),
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
                .reshape(&[b, l, self.n_heads, self.head_dim])?;
            if let Some(norm) = &self.q_norm {
                q = norm.forward(&q)?;
            }
            let mut q = q.transpose_axes(&[0, 2, 1, 3])?;
            let mut k = self
                .k_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?;
            if let Some(norm) = &self.k_norm {
                k = norm.forward(&k)?;
            }
            let mut k = k.transpose_axes(&[0, 2, 1, 3])?;
            let v = self
                .v_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let offset = self.cache.offset;
            q = rope(&q, self.rot_dims, false, self.rope_theta, 1.0, offset, None)?;
            k = rope(&k, self.rot_dims, false, self.rope_theta, 1.0, offset, None)?;
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

    struct MiniMaxMoE {
        gate: Linear,
        switch_mlp: SwitchMlp,
        shared: Option<MiniMaxMlp>,
        expert_bias: Vec<f32>,
        top_k: usize,
        alpha: f32,
        limit: f32,
        routed_scaling: f32,
    }

    impl MiniMaxMoE {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let bias = raw_array(arrays, &format!("{prefix}.e_score_correction_bias"))?
                .as_type::<f32>()?;
            transforms::eval([&bias])?;
            let shared =
                if arrays.contains_key(&format!("{prefix}.shared_experts.gate_proj.weight")) {
                    Some(MiniMaxMlp::load(
                        &format!("{prefix}.shared_experts"),
                        arrays,
                        config,
                    )?)
                } else {
                    None
                };
            Ok(Self {
                gate: Linear::load(&format!("{prefix}.gate"), arrays, config)?,
                switch_mlp: SwitchMlp::load(
                    &format!("{prefix}.switch_mlp"),
                    arrays,
                    config,
                    stream_ctx,
                )?,
                shared,
                expert_bias: bias.as_slice::<f32>().to_vec(),
                top_k: config.num_experts_per_tok.unwrap_or(1) as usize,
                alpha: 1.702,
                limit: config.swiglu_limit.unwrap_or(7.0),
                routed_scaling: config.routed_scaling_factor,
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let d = x.shape()[2];
            let scores = sigmoid(&self.gate.forward(x)?.as_type::<f32>()?)?;
            transforms::eval([&scores])?;
            let shape = scores.shape();
            let (l, n_experts) = (shape[1] as usize, shape[2] as usize);
            let raw = scores.as_slice::<f32>();
            let mut idx_v: Vec<u32> = Vec::with_capacity(l * self.top_k);
            let mut wts_v: Vec<f32> = Vec::with_capacity(l * self.top_k);
            for token in 0..l {
                let base = token * n_experts;
                // Rank by (score + bias); keep the bias-free scores; normalize over the top-k.
                let mut ranked: Vec<usize> = (0..n_experts).collect();
                ranked.sort_by(|&a, &b| {
                    (raw[base + b] + self.expert_bias[b])
                        .total_cmp(&(raw[base + a] + self.expert_bias[a]))
                        .then_with(|| a.cmp(&b))
                });
                ranked.truncate(self.top_k.min(n_experts));
                let mut w: Vec<f32> = ranked.iter().map(|&i| raw[base + i]).collect();
                let denom: f32 = w.iter().sum::<f32>() + 1e-20;
                for x in &mut w {
                    *x = *x / denom * self.routed_scaling;
                }
                for (k, &e) in ranked.iter().enumerate() {
                    idx_v.push(e as u32);
                    wts_v.push(w[k]);
                }
            }
            let top_k = self.top_k as i32;
            let inds = Array::from_slice(&idx_v, &[l as i32, top_k]);
            let weights = Array::from_slice(&wts_v, &[l as i32, top_k, 1]);
            let xe = x.reshape(&[l as i32, 1, 1, d])?;
            // Batched SwiGLU-OAI experts (clamped, alpha-scaled, (up+1)).
            let gate_pre = self.switch_mlp.gate_proj.gather(&xe, &inds)?;
            let up_pre = self.switch_mlp.up_proj.gather(&xe, &inds)?;
            let act = swiglu_oai(&gate_pre, &up_pre, self.alpha, self.limit)?;
            let expert_out = self
                .switch_mlp
                .down_proj
                .gather(&act, &inds)?
                .reshape(&[l as i32, top_k, d])?
                .as_type::<f32>()?;
            let mut y = sum_axis(&(expert_out * weights), 1, false)?.reshape(&[1, l as i32, d])?;
            if let Some(shared) = &self.shared {
                y = y + shared.forward(x)?.as_type::<f32>()?;
            }
            Ok(y)
        }
    }

    enum MiniMaxFfn {
        Dense(MiniMaxMlp),
        Moe(MiniMaxMoE),
    }

    struct MiniMaxLayer {
        input_ln: RmsNorm,
        post_attn_ln: RmsNorm,
        attn: MiniMaxAttention,
        ffn: MiniMaxFfn,
    }

    impl MiniMaxLayer {
        fn load(
            idx: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let p = format!("model.layers.{idx}");
            let eps = config.rms_norm_eps;
            // The first `first_k_dense_replace` layers are a dense MLP; the rest are MoE.
            let ffn = if arrays.contains_key(&format!("{p}.block_sparse_moe.gate.weight")) {
                MiniMaxFfn::Moe(MiniMaxMoE::load(
                    &format!("{p}.block_sparse_moe"),
                    arrays,
                    config,
                    stream_ctx,
                )?)
            } else {
                MiniMaxFfn::Dense(MiniMaxMlp::load(&format!("{p}.mlp"), arrays, config)?)
            };
            Ok(Self {
                input_ln: minimax_norm(&format!("{p}.input_layernorm.weight"), arrays, eps)?,
                post_attn_ln: minimax_norm(
                    &format!("{p}.post_attention_layernorm.weight"),
                    arrays,
                    eps,
                )?,
                attn: MiniMaxAttention::load(&format!("{p}.self_attn"), arrays, config)?,
                ffn,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let r = &x + self.attn.forward(&self.input_ln.forward(&x)?)?;
            let normed = self.post_attn_ln.forward(&r)?;
            let h = match &self.ffn {
                MiniMaxFfn::Dense(mlp) => mlp.forward(&normed)?,
                MiniMaxFfn::Moe(moe) => moe.forward(&normed)?,
            };
            Ok(r + h)
        }
    }

    struct MiniMaxLike {
        embed: Embedding,
        layers: Vec<MiniMaxLayer>,
        norm: RmsNorm,
        lm_head: Linear,
    }

    impl MiniMaxLike {
        fn new(
            config: MlxModelConfig,
            arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let layers = (0..config.num_hidden_layers)
                .map(|idx| MiniMaxLayer::load(idx, &arrays, &config, stream_ctx))
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                embed: Embedding::load("model.embed_tokens", &arrays, &config)?,
                norm: minimax_norm("model.norm.weight", &arrays, config.rms_norm_eps)?,
                lm_head: Linear::load("lm_head", &arrays, &config)?,
                layers,
            })
        }
    }

    impl CausalLm for MiniMaxLike {
        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let mut h = self.embed.forward(&ids)?;
            for layer in &mut self.layers {
                h = layer.forward(h)?;
            }
            h = self.norm.forward(&h)?;
            let logits = self.lm_head.forward(&h)?;
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn reset_cache(&mut self) {
            for layer in &mut self.layers {
                layer.attn.cache.reset();
            }
        }

        fn prepare_cache(&mut self, capacity: i32) {
            for layer in &mut self.layers {
                layer.attn.cache.prepare_capacity(capacity);
            }
        }
    }

    // ---------------------- LongCat-2.0 (longcat2) ----------------------
    // ScMoE decoder: each layer runs 2 absorbed-MLA attentions + 2 dense MLPs, plus one MoE computed on
    // the first sub-block's post-attn hidden and added as a shortcut. Input is an n-gram hashing
    // embedding; MoE routing is softmax + 128 identity "zero" experts; attention is the shared
    // MlaAttention (YARN + mla-lora scaling). Weights under `model.`, untied lm_head.

    struct NgramEmbedding {
        word_embeddings: Embedding,
        embedders: Vec<Embedding>,
        post_projs: Vec<Linear>,
        emb_vocab: Vec<i64>,
        vocab_mods: Vec<Vec<i64>>,
        k: usize,
        n: usize,
        norm: f32,
        eos: i64,
        context: Vec<u32>,
    }

    impl NgramEmbedding {
        fn load(config: &MlxModelConfig, arrays: &HashMap<String, Array>) -> Result<Self> {
            let k = config.oe_split_num.unwrap_or(1) as usize;
            let n = config.oe_neighbor_num.unwrap_or(1) as usize;
            let vocab = config.vocab_size as i64;
            let m = config
                .oe_vocab_size_ratio
                .ok_or_else(|| anyhow!("LongCat config missing oe_vocab_size_ratio"))?
                as f64
                * vocab as f64;
            let num = k * (n - 1);
            let p = "model.ngram_embeddings";
            let mut embedders = Vec::with_capacity(num);
            let mut post_projs = Vec::with_capacity(num);
            let mut emb_vocab = Vec::with_capacity(num);
            let mut vocab_mods = Vec::with_capacity(num);
            for index in 0..num {
                embedders.push(Embedding::load(
                    &format!("{p}.embedders.{index}"),
                    arrays,
                    config,
                )?);
                post_projs.push(Linear::load(
                    &format!("{p}.post_projs.{index}"),
                    arrays,
                    config,
                )?);
                let evd = (m + (index * 2 + 1) as f64) as i64;
                emb_vocab.push(evd);
            }
            // vocab_mods[(i,j)]: pm=1; repeat (i-1) times: pm = (pm*vocab) % evd; collect.
            for i in 2..=n {
                for j in 0..k {
                    let index = (i - 2) * k + j;
                    let evd = emb_vocab[index];
                    let mut pm: i64 = 1;
                    let mut mods = Vec::with_capacity(i - 1);
                    for _ in 0..(i - 1) {
                        pm = ((pm as i128 * vocab as i128) % evd as i128) as i64;
                        mods.push(pm);
                    }
                    vocab_mods.push(mods);
                }
            }
            Ok(Self {
                word_embeddings: Embedding::load(&format!("{p}.word_embeddings"), arrays, config)?,
                embedders,
                post_projs,
                emb_vocab,
                vocab_mods,
                k,
                n,
                norm: (1 + k * (n - 1)) as f32,
                eos: config.eos_token_ids.first().copied().unwrap_or(2) as i64,
                context: Vec::new(),
            })
        }

        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let offset = self.context.len();
            self.context.extend_from_slice(input_ids);
            let full = &self.context;
            let l = input_ids.len();
            // last EOS position strictly before each absolute index (for n-gram reach masking).
            let mut last_eos_before = vec![-1i64; full.len()];
            let mut last = -1i64;
            for (idx, &tok) in full.iter().enumerate() {
                last_eos_before[idx] = last;
                if tok as i64 == self.eos {
                    last = idx as i64;
                }
            }
            let id_arr = Array::from_slice(input_ids, &[1, l as i32]);
            let mut x = self.word_embeddings.forward(&id_arr)?;
            for i in 2..=self.n {
                for j in 0..self.k {
                    let index = (i - 2) * self.k + j;
                    let evd = self.emb_vocab[index];
                    let mods = &self.vocab_mods[index];
                    let mut new_ids = vec![0i32; l];
                    for (p, slot) in new_ids.iter_mut().enumerate() {
                        let abs = offset + p;
                        let reach = abs as i64 - last_eos_before[abs];
                        let mut ng = full[abs] as i128;
                        for t in 2..=i {
                            let back = t - 1;
                            // shift_right by (t-1), zeroed across an EOS within `back` positions.
                            let sh = if abs >= back && reach > back as i64 {
                                full[abs - back] as i128
                            } else {
                                0
                            };
                            ng += sh * mods[t - 2] as i128;
                        }
                        *slot = (ng.rem_euclid(evd as i128)) as i32;
                    }
                    let new_arr = Array::from_slice(&new_ids, &[1, l as i32]);
                    let emb = self.embedders[index].forward(&new_arr)?;
                    x = x + self.post_projs[index].forward(&emb)?;
                }
            }
            Ok(x / self.norm)
        }
    }

    struct LongCatMoe {
        router: Linear,
        e_score_bias: Vec<f32>,
        switch_mlp: SwitchMlp,
        n_routed: i32,
        top_k: usize,
        routed_scaling: f32,
        norm_topk: bool,
    }

    impl LongCatMoe {
        fn load(
            prefix: &str,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let bias = raw_array(arrays, &format!("{prefix}.router.e_score_correction_bias"))?
                .as_type::<f32>()?;
            transforms::eval([&bias])?;
            Ok(Self {
                router: Linear::load(&format!("{prefix}.router.classifier"), arrays, config)?,
                e_score_bias: bias.as_slice::<f32>().to_vec(),
                switch_mlp: SwitchMlp::load(
                    &format!("{prefix}.switch_mlp"),
                    arrays,
                    config,
                    stream_ctx,
                )?,
                n_routed: config.n_routed_experts.unwrap_or(0) as i32,
                top_k: config.num_experts_per_tok.unwrap_or(1) as usize,
                routed_scaling: config.routed_scaling_factor,
                // LongCat's norm_topk_prob defaults to false (the shared config default is true).
                norm_topk: config
                    .raw
                    .get("norm_topk_prob")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            })
        }

        fn forward(&self, x: &Array) -> Result<Array> {
            let shape = x.shape();
            let (b, l, d) = (shape[0], shape[1], shape[2]);
            if b != 1 {
                bail!("LongCat MoE supports batch size 1, got {b}");
            }
            let logits = self.router.forward(x)?.as_type::<f32>()?;
            transforms::eval([&logits])?;
            let experts = *logits.shape().last().unwrap() as usize;
            let raw_logits = logits.as_slice::<f32>();
            let mut outputs = Vec::with_capacity(l as usize);
            for token in 0..l as usize {
                let base = token * experts;
                let lg = &raw_logits[base..base + experts];
                // softmax over all experts (CPU, numerically stable).
                let maxl = lg.iter().cloned().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = lg.iter().map(|&v| (v - maxl).exp()).collect();
                let denom: f32 = exps.iter().sum::<f32>() + 1e-20;
                let s: Vec<f32> = exps.iter().map(|e| e / denom).collect();
                let s = &s[..];
                // top-k by (score + correction bias)
                let mut ranked = (0..experts)
                    .map(|e| (e, s[e] + self.e_score_bias.get(e).copied().unwrap_or(0.0)))
                    .collect::<Vec<_>>();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                ranked.truncate(self.top_k);
                // weights come from the (bias-free) softmax scores at the selected experts
                let mut weights = ranked.iter().map(|&(e, _)| s[e]).collect::<Vec<_>>();
                if self.norm_topk && weights.len() > 1 {
                    let denom = weights.iter().sum::<f32>() + 1e-20;
                    for w in &mut weights {
                        *w /= denom;
                    }
                }
                for w in &mut weights {
                    *w *= self.routed_scaling;
                }
                let token_x = x.index((0, token as i32, ..)).reshape(&[1, 1, d])?;
                let mut acc = Array::zeros::<f32>(&[1, 1, d])?;
                let mut identity_w = 0.0f32;
                for (&(expert, _), &w) in ranked.iter().zip(weights.iter()) {
                    if (expert as i32) < self.n_routed {
                        acc = acc + self.switch_mlp.forward_expert(&token_x, expert as i32)? * w;
                    } else {
                        // identity ("zero") expert: contributes the input scaled by its weight
                        identity_w += w;
                    }
                }
                if identity_w != 0.0 {
                    acc = acc + token_x.as_type::<f32>()? * identity_w;
                }
                outputs.push(acc);
            }
            Ok(concatenate_axis(&outputs, 1)?)
        }
    }

    struct LongCatLayer {
        attn: Vec<MlaAttention>,
        mlps: Vec<Mlp>,
        moe: LongCatMoe,
        input_ln: Vec<RmsNorm>,
        post_attn_ln: Vec<RmsNorm>,
    }

    impl LongCatLayer {
        fn load(
            layer: u32,
            arrays: &HashMap<String, Array>,
            config: &MlxModelConfig,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let p = format!("model.layers.{layer}");
            let mut attn = Vec::with_capacity(2);
            let mut mlps = Vec::with_capacity(2);
            let mut input_ln = Vec::with_capacity(2);
            let mut post_attn_ln = Vec::with_capacity(2);
            for i in 0..2 {
                attn.push(MlaAttention::load(
                    &format!("{p}.self_attn.{i}"),
                    arrays,
                    config,
                )?);
                mlps.push(Mlp::load(&format!("{p}.mlps.{i}"), arrays, config)?);
                input_ln.push(RmsNorm::load(
                    &format!("{p}.input_layernorm.{i}.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?);
                post_attn_ln.push(RmsNorm::load(
                    &format!("{p}.post_attention_layernorm.{i}.weight"),
                    arrays,
                    config.rms_norm_eps,
                )?);
            }
            Ok(Self {
                attn,
                mlps,
                moe: LongCatMoe::load(&format!("{p}.mlp"), arrays, config, stream_ctx)?,
                input_ln,
                post_attn_ln,
            })
        }

        fn forward(&mut self, x: Array) -> Result<Array> {
            let mut h = x;
            let mut shortcut: Option<Array> = None;
            for i in 0..2 {
                let residual = h.clone();
                let normed = self.input_ln[i].forward(&h)?;
                let a = self.attn[i].forward(&normed)?;
                h = residual + a;
                let residual = h.clone();
                let normed = self.post_attn_ln[i].forward(&h)?;
                if i == 0 {
                    // MoE runs on the first sub-block's post-attn hidden, added back as a shortcut.
                    shortcut = Some(self.moe.forward(&normed)?);
                }
                let m = self.mlps[i].forward(&normed)?;
                h = residual + m;
                if i == 1 {
                    h = h + shortcut.take().unwrap();
                }
            }
            Ok(h)
        }
    }

    struct LongCatLike {
        ngram: NgramEmbedding,
        layers: Vec<LongCatLayer>,
        norm: RmsNorm,
        lm_head: Linear,
    }

    impl LongCatLike {
        fn new(
            config: MlxModelConfig,
            arrays: HashMap<String, Array>,
            stream_ctx: Option<&StreamContext>,
        ) -> Result<Self> {
            let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
            for layer in 0..config.num_hidden_layers {
                layers.push(LongCatLayer::load(layer, &arrays, &config, stream_ctx)?);
            }
            Ok(Self {
                ngram: NgramEmbedding::load(&config, &arrays)?,
                layers,
                norm: RmsNorm::load("model.norm.weight", &arrays, config.rms_norm_eps)?,
                lm_head: Linear::load("lm_head", &arrays, &config)?,
            })
        }
    }

    impl CausalLm for LongCatLike {
        fn forward(&mut self, input_ids: &[u32]) -> Result<Array> {
            let mut h = self.ngram.forward(input_ids)?;
            for layer in &mut self.layers {
                h = layer.forward(h)?;
            }
            h = self.norm.forward(&h)?;
            let logits = self.lm_head.forward(&h)?;
            transforms::eval([&logits])?;
            Ok(logits)
        }

        fn reset_cache(&mut self) {
            self.ngram.context.clear();
            for layer in &mut self.layers {
                for a in &mut layer.attn {
                    a.cache.reset();
                }
            }
        }

        fn prepare_cache(&mut self, capacity: i32) {
            for layer in &mut self.layers {
                for a in &mut layer.attn {
                    a.cache.prepare_capacity(capacity);
                }
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
        let mut spec =
            config
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

    /// Extract the layer index from a prefix like `model.layers.{N}.mlp...`.
    /// Returns 0 if the pattern doesn't match (defensive — shouldn't happen for
    /// valid MoE prefixes).
    fn extract_layer_from_prefix(prefix: &str) -> u32 {
        // Inkling's text tower nests layers under `model.llm.layers.{N}`; every other arch uses
        // `model.layers.{N}`. Without the `.llm.` variant this returned 0 for every Inkling layer,
        // so streamed experts never matched their source and failed to load.
        prefix
            .strip_prefix("model.layers.")
            .or_else(|| prefix.strip_prefix("model.llm.layers."))
            .and_then(|rest| rest.split('.').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
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

    /// Recover the quantization bit width from a weight/scales pair.
    ///
    /// An affine-quantized weight packs `32 / bits` values per u32 word, and carries one scale per
    /// `group_size` values, so `weight_last * 32 / (scales_last * group_size)` is the width. The
    /// shapes are ground truth; the config's per-tensor list is not always complete — GLM-5.2-Alis
    /// omits layer 78's three expert projections, which then fall back to the file-level 4-bit
    /// default while the data is 3-bit, and `quantized_matmul` rejects them outright.
    ///
    /// Only meaningful for affine packing (mxfp4 and friends pack differently), and only trusted
    /// when the result is a width MLX actually supports.
    fn derived_quant_bits(
        weight: &Array,
        scales: &Array,
        group_size: i32,
        mode: &str,
    ) -> Option<i32> {
        if mode != "affine" || group_size <= 0 {
            return None;
        }
        let packed = *weight.shape().last()?;
        let groups = *scales.shape().last()?;
        let denom = groups.checked_mul(group_size)?;
        if denom <= 0 {
            return None;
        }
        let bits = packed.checked_mul(32)? / denom;
        matches!(bits, 2 | 3 | 4 | 5 | 6 | 8).then_some(bits)
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
                empty_array(), // lhs_indices: null → broadcast x's batch dims
                rhs_indices.as_ptr(),
                transpose,
                optional_int(group_size),
                optional_int(bits),
                mode.as_ptr(),
                false, // sorted_indices
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
        let gp = gather_qmm_mode(
            &xe,
            &a[3],
            &a[4],
            Some(&a[5]),
            &inds_r,
            true,
            group_size,
            bits,
            "affine",
        )?;
        let gp = sigmoid(&gp)? * gp;
        let up = gather_qmm_mode(
            &xe,
            &a[6],
            &a[7],
            Some(&a[8]),
            &inds_r,
            true,
            group_size,
            bits,
            "affine",
        )?;
        let down = gather_qmm_mode(
            &(gp * up),
            &a[9],
            &a[10],
            Some(&a[11]),
            &inds_r,
            true,
            group_size,
            bits,
            "affine",
        )?;
        let eo = down.reshape(&[l, top_k, d])?.as_type::<f32>()?;
        let wr = w.reshape(&[l, top_k, 1])?;
        let mut y = sum_axis(&(eo * wr), 1, Some(false))?.reshape(&[1, l, d])?;
        // Always-on shared expert (quantized SwiGLU MLP).
        let sg = quantized_matmul_mode(
            x,
            &a[12],
            &a[13],
            Some(&a[14]),
            true,
            group_size,
            bits,
            "affine",
        )?;
        let sg = sigmoid(&sg)? * sg;
        let su = quantized_matmul_mode(
            x,
            &a[15],
            &a[16],
            Some(&a[17]),
            true,
            group_size,
            bits,
            "affine",
        )?;
        let sd = quantized_matmul_mode(
            &(sg * su),
            &a[18],
            &a[19],
            Some(&a[20]),
            true,
            group_size,
            bits,
            "affine",
        )?;
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


    /// [`v4_rope`] with a per-row position offset: row `i` rotates at positions
    /// `offsets[i] + seq_index`. Batched decode places every row's step token in the same
    /// physical cache column while rows sit at different logical positions, so the rotation
    /// must vary per row. Same elementwise broadcast math as `v4_rope` — `cos`/`sin` are
    /// built `[b, 1, t, half]` and broadcast over heads — so there is no fast-kernel
    /// batch-dimension hazard (see `rope_rows` for the kernel bug that motivates caution).
    fn v4_rope_rows(
        x: &Array,
        dims: i32,
        base: f32,
        offsets: &[i32],
        inverse: bool,
    ) -> Result<Array> {
        if dims == 0 {
            return Ok(x.clone());
        }
        let shape = x.shape();
        if shape.len() != 4 {
            bail!("DeepSeek V4 RoPE expects a 4D tensor, got shape {shape:?}");
        }
        let (b, h, t) = (shape[0], shape[1], shape[2]);
        if offsets.len() != b as usize {
            bail!("v4_rope_rows: {} offsets for batch of {b}", offsets.len());
        }
        let half = dims / 2;
        let inv_freq = (0..half)
            .map(|idx| 1.0 / base.powf((2 * idx) as f32 / dims as f32))
            .collect::<Vec<_>>();
        let mut pos = Vec::with_capacity((b * t) as usize);
        for &off in offsets {
            for idx in 0..t {
                pos.push((off + idx) as f32);
            }
        }
        let theta = Array::from_slice(&pos, &[b, 1, t, 1])
            * Array::from_slice(&inv_freq, &[1, 1, 1, half]);
        let theta = if inverse { theta * -1.0 } else { theta };
        let cos = cos(&theta)?;
        let sin = sin(&theta)?;
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

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub use native::StreamContext;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx", test))]
mod batch_tests {
    use super::*;

    fn req(prompt: &str, max_tokens: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: prompt.to_string(),
            max_tokens,
            // Greedy: batching must be bit-identical to serial decoding, and sampling would
            // introduce RNG differences that mask a real divergence.
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            seed: Some(42),
            stop_sequences: Vec::new(),
            media_inputs: Vec::new(),
            messages: Vec::new(),
        }
    }

    /// A row's output must not depend on which requests share its batch.
    ///
    /// This is the property that actually validates left-padding. Row 0 is the shortest prompt, so
    /// it is padded by however much its longest batch-mate requires — pairing it with partners of
    /// two different lengths changes its padding width. If `pad_attention_bias` failed to hide the
    /// pad positions, row 0's output would change with its partner; because it is hidden, the row
    /// is independent and both runs agree.
    ///
    /// Note this deliberately does NOT compare against serial (batch-1) decoding. MLX dispatches
    /// different kernels for b=1 and b>1, and the resulting float differences flip greedy argmax
    /// wherever two logits are near-tied — verified by observing identical divergence with
    /// equal-length prompts, i.e. with no padding involved at all. Batch-invariance of that kind is
    /// not a property this (or any) batched implementation provides.
    ///
    /// Needs a real checkpoint, so it is opt-in:
    ///   HI_MLX_BATCH_TEST_MODEL=/path/to/mlx/model cargo test -p hi-mlx --features mlx -- --ignored
    #[test]
    #[ignore = "requires HI_MLX_BATCH_TEST_MODEL pointing at a local MLX checkpoint"]
    fn batched_row_output_is_independent_of_batch_mates() {
        let Some(path) = std::env::var_os("HI_MLX_BATCH_TEST_MODEL") else {
            panic!("set HI_MLX_BATCH_TEST_MODEL");
        };
        let mut runtime = NativeRuntime::from_path(&path).expect("load model");
        assert!(runtime.supports_batch(), "family lacks batched decode");

        let subject = req("fn add(a: i64, b: i64) -> i64 {", 24);
        let short_mate = req("fn mul(a: i64, b: i64) -> i64 {", 24);
        let long_mate = req(
            "// A deliberately longer prompt so the subject row is padded by a different width in \
             this batch than in the other one.\nfn gcd(a: u64, b: u64) -> u64 {",
            24,
        );

        let run = |rt: &mut NativeRuntime, mate: &GenerationRequest| -> String {
            let reqs = vec![subject.clone(), mate.clone()];
            let outs = rt
                .stream_generate_batch(&reqs, |_, _| Ok(()))
                .expect("batched generate");
            outs[0].text.clone()
        };

        let with_short = run(&mut runtime, &short_mate);
        let with_long = run(&mut runtime, &long_mate);
        assert_eq!(
            with_short, with_long,
            "row 0's output changed with its batch-mate, so padded key positions are leaking into \
             attention:\n  padded by {} : {with_short:?}\n  padded by {} : {with_long:?}",
            0, "more"
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx", test))]
mod batch_diag {
    use super::*;
    use mlx_rs::Array;
    use mlx_rs::transforms;
    use mlx_rs::ops::indexing::IndexOp;

    /// Diff batched vs single-sequence logits for the SAME prompt.
    ///
    /// Sampled text can't localise a masking bug: greedy decoding hides small logit errors and
    /// temperature sampling turns them into noise. Comparing the logit vectors directly says
    /// whether the padded batch reproduces the single-sequence forward, and at which stage.
    ///
    ///   HI_MLX_BATCH_TEST_MODEL=/path/to/model cargo test -p hi-mlx --features mlx \
    ///       batched_logits_match_single -- --ignored --nocapture
    #[test]
    #[ignore = "requires HI_MLX_BATCH_TEST_MODEL"]
    fn batched_logits_match_single() {
        let path = std::env::var_os("HI_MLX_BATCH_TEST_MODEL").expect("set HI_MLX_BATCH_TEST_MODEL");
        let mut rt = NativeRuntime::from_path(&path).expect("load");
        let tok = rt.tokenizer_for_test();

        let subject = "fn add(a: i64, b: i64) -> i64 {";
        let mate = "// a deliberately longer prompt so the subject row gets left-padded in the batch\nfn gcd(a: u64, b: u64) -> u64 {";
        let ids_s = tok.encode(subject).unwrap();
        let ids_m = tok.encode(mate).unwrap();

        // --- single-sequence prefill ---
        let single = {
            let m = rt.model_for_test();
            m.reset_cache();
            m.stage_pad_lens(None);
            m.prepare_cache(ids_s.len() as i32 + 8);
            let lg = m.forward(&ids_s).unwrap();
            last_row_vec(&lg, 0)
        };

        // --- batched prefill, subject left-padded to the mate's width ---
        let batched = {
            let m = rt.model_for_test();
            let width = ids_s.len().max(ids_m.len());
            let pads = [(width - ids_s.len()) as i32, (width - ids_m.len()) as i32];
            let pad_id = 0u32;
            let mut flat = Vec::new();
            for ids in [&ids_s, &ids_m] {
                flat.extend(std::iter::repeat_n(pad_id, width - ids.len()));
                flat.extend_from_slice(ids);
            }
            m.reset_cache();
            m.stage_pad_lens(None);
            m.prepare_cache(width as i32 + 8);
            m.stage_pad_lens(Some(&pads));
            let arr = Array::from_slice(&flat, &[2, width as i32]);
            let lg = m.forward_batch(&arr).unwrap();
            m.stage_pad_lens(None);
            last_row_vec(&lg, 0)
        };

        assert_eq!(single.len(), batched.len(), "vocab size mismatch");
        let (mut max_abs, mut argmax_s, mut argmax_b) = (0.0f32, 0usize, 0usize);
        for (i, (a, b)) in single.iter().zip(batched.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_abs { max_abs = d; }
            if *a > single[argmax_s] { argmax_s = i; }
            if *b > batched[argmax_b] { argmax_b = i; }
        }
        println!("  max |single - batched| = {max_abs:.4}");
        println!("  argmax single = {argmax_s}, batched = {argmax_b}");
        // bf16 round-off across different kernels is ~1e-2; anything far above that is a real bug.
        assert!(
            max_abs < 0.5,
            "batched prefill does not reproduce single-sequence logits (max diff {max_abs}) — \
             the padded row is not being masked correctly"
        );
    }

    /// Same comparison, but stepping through decode: feed identical tokens to both the
    /// single-sequence and the batched model and diff row 0's logits at every step. Prefill
    /// already matches, so the first step whose diff explodes is where the bug lives.
    #[test]
    #[ignore = "requires HI_MLX_BATCH_TEST_MODEL"]
    fn batched_decode_steps_match_single() {
        let path = std::env::var_os("HI_MLX_BATCH_TEST_MODEL").expect("set HI_MLX_BATCH_TEST_MODEL");
        let mut rt = NativeRuntime::from_path(&path).expect("load");
        let ids_s = rt.tokenizer_for_test().encode("fn add(a: i64, b: i64) -> i64 {").unwrap();
        let ids_m = rt.tokenizer_for_test().encode(
            "// a deliberately longer prompt so the subject row gets left-padded in the batch\nfn gcd(a: u64, b: u64) -> u64 {").unwrap();
        const STEPS: usize = 12;

        // Single-sequence: greedy decode, recording each step's logits and the token fed next.
        let (single_logits, fed) = {
            let m = rt.model_for_test();
            m.reset_cache();
            m.stage_pad_lens(None);
            m.prepare_cache(ids_s.len() as i32 + STEPS as i32 + 4);
            let mut lg = m.forward(&ids_s).unwrap();
            let (mut all, mut fed) = (Vec::new(), Vec::new());
            for _ in 0..STEPS {
                let v = last_row_vec(&lg, 0);
                let t = v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
                all.push(v);
                fed.push(t);
                lg = m.forward(&[t]).unwrap();
            }
            (all, fed)
        };

        // Batched: same prompt left-padded, fed the SAME tokens, so any divergence is the model's.
        let batched_logits = {
            let m = rt.model_for_test();
            let width = ids_s.len().max(ids_m.len());
            let pads = [(width - ids_s.len()) as i32, (width - ids_m.len()) as i32];
            let mut flat = Vec::new();
            for ids in [&ids_s, &ids_m] {
                flat.extend(std::iter::repeat_n(0u32, width - ids.len()));
                flat.extend_from_slice(ids);
            }
            m.reset_cache();
            m.stage_pad_lens(None);
            m.prepare_cache(width as i32 + STEPS as i32 + 4);
            m.stage_pad_lens(Some(&pads));
            let mut lg = m.forward_batch(&Array::from_slice(&flat, &[2, width as i32])).unwrap();
            let mut all = Vec::new();
            for step in 0..STEPS {
                all.push(last_row_vec(&lg, 0));
                let next = [fed[step], fed[step]];
                lg = m.forward_batch(&Array::from_slice(&next, &[2, 1])).unwrap();
            }
            m.stage_pad_lens(None);
            all
        };

        println!("  step   max|diff|   argmax_single  argmax_batched");
        let mut first_bad = None;
        for step in 0..STEPS {
            let (a, b) = (&single_logits[step], &batched_logits[step]);
            let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            let am = |v: &Vec<f32>| v.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0;
            println!("  {step:4}   {d:9.4}   {:>13}  {:>14}", am(a), am(b));
            if d > 0.5 && first_bad.is_none() { first_bad = Some(step); }
        }
        match first_bad {
            Some(s) => panic!("batched decode diverges from single-sequence at step {s}"),
            None => println!("  all {STEPS} decode steps match within bf16 tolerance"),
        }
    }


    /// Decode-step equality for EVERY row, not just row 0. The scheduler repro shows row 0
    /// matching its unbatched output exactly while rows 1..n degenerate — and the existing
    /// step test only ever diffed row 0, which is why it passes while production fails.
    /// Each row is fed ITS OWN single-mode greedy token at each step (the production shape),
    /// and its logits are diffed against its own single-mode reference.
    #[test]
    #[ignore = "requires HI_MLX_BATCH_TEST_MODEL"]
    fn batched_decode_all_rows_match_single() {
        let path = std::env::var_os("HI_MLX_BATCH_TEST_MODEL").expect("set HI_MLX_BATCH_TEST_MODEL");
        let mut rt = NativeRuntime::from_path(&path).expect("load");
        let markers = ["alpha", "bravo", "charlie", "delta"];
        let prompts: Vec<String> = markers
            .iter()
            .map(|m| {
                format!(
                    "Write a self-contained Rust module implementing a {m} data structure, \
                     with exactly 3 #[test] functions asserting concrete values. Output ONLY \
                     a ```rust code block."
                )
            })
            .collect();
        let ids: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| rt.tokenizer_for_test().encode(p).unwrap())
            .collect();
        let steps: usize = std::env::var("HI_MLX_TEST_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24);

        // Per-row single-mode greedy reference: logits at each step + the token fed next.
        let mut refs: Vec<(Vec<Vec<f32>>, Vec<u32>)> = Vec::new();
        for row_ids in &ids {
            let m = rt.model_for_test();
            m.reset_cache();
            m.stage_pad_lens(None);
            m.prepare_cache(row_ids.len() as i32 + steps as i32 + 4);
            let mut lg = m.forward(row_ids).unwrap();
            let (mut all, mut fed) = (Vec::new(), Vec::new());
            for _ in 0..steps {
                let v = last_row_vec(&lg, 0);
                let t = v
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32;
                all.push(v);
                fed.push(t);
                lg = m.forward(&[t]).unwrap();
            }
            refs.push((all, fed));
        }

        // Batched: left-pad, one prefill, then per-step feed row i its own reference token.
        // HI_MLX_TEST_CONCAT=1 runs the batched phase on a fresh runtime that never calls
        // prepare_cache, so the KV cache stays in concat mode — bisecting dense-cache
        // involvement out of the composite.
        let concat_mode = std::env::var_os("HI_MLX_TEST_CONCAT").is_some();
        let mut rt2;
        let m = if concat_mode {
            rt2 = NativeRuntime::from_path(&path).expect("load2");
            rt2.model_for_test()
        } else {
            rt.model_for_test()
        };
        let b = ids.len();
        let width = ids.iter().map(Vec::len).max().unwrap();
        let pads: Vec<i32> = ids.iter().map(|p| (width - p.len()) as i32).collect();
        let mut flat = Vec::new();
        for row_ids in &ids {
            flat.extend(std::iter::repeat_n(0u32, width - row_ids.len()));
            flat.extend_from_slice(row_ids);
        }
        m.reset_cache();
        m.stage_pad_lens(None);
        if !concat_mode {
            m.prepare_cache(width as i32 + steps as i32 + 4);
        }
        m.stage_pad_lens(Some(&pads));
        let mut lg = m
            .forward_batch(&Array::from_slice(&flat, &[b as i32, width as i32]))
            .unwrap();
        println!("  pads = {pads:?}  width = {width}  steps = {steps}");
        for (r, (_, fed)) in refs.iter().enumerate() {
            println!("  row {r} fed[0..8] = {:?}", &fed[..fed.len().min(8)]);
        }
        let mut first_bad: Option<(usize, usize)> = None;
        for step in 0..steps {
            let mut next = Vec::with_capacity(b);
            for row in 0..b {
                let v = last_row_vec(&lg, row as i32);
                let refv = &refs[row].0[step];
                let d = v
                    .iter()
                    .zip(refv)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                let am = |z: &[f32]| {
                    z.iter()
                        .enumerate()
                        .max_by(|p, q| p.1.total_cmp(q.1))
                        .unwrap()
                        .0
                };
                let (am_s, am_b) = (am(refv), am(&v));
                if step <= 3 && (am_b == am_s) {
                    println!("  step {step:3} row {row}: ok both argmax {am_s} (diff {d:.4})");
                }
                if d > 0.5 || am_b != am_s {
                    println!(
                        "  step {step:3} row {row}: max|diff|={d:9.4} argmax single={am_s:6} batched={am_b:6}{}",
                        if am_b != am_s { "  *** MISMATCH" } else { "" }
                    );
                    if first_bad.is_none() {
                        first_bad = Some((row, step));
                    }
                }
                next.push(refs[row].1[step]);
            }
            lg = m
                .forward_batch(&Array::from_slice(&next, &[b as i32, 1]))
                .unwrap();
        }
        m.stage_pad_lens(None);
        match first_bad {
            Some((row, step)) => {
                panic!("row {row} diverges from its single-sequence reference at step {step}")
            }
            None => println!("  all {b} rows match their references across {steps} steps"),
        }
    }


    /// All-rows decode equality for the ragged (per-row-position) V4 batching, on a tiny but
    /// DISCRIMINATING fixture: sin-filled weights (the zero fixtures produce constant logits and
    /// cannot catch positional or masking bugs), two layers — one plain sliding-window layer and
    /// one compressed layer with an indexer — an attention sink, ragged prompt lengths that
    /// straddle compression-block boundaries, and enough steps that the window slides and new
    /// blocks form mid-decode. Every row's logits must match its own single-sequence reference
    /// at every step.
    #[test]
    fn v4_ragged_batch_all_rows_match_single() {
        use std::collections::HashMap as Map;
        // Two configurations: top-k high enough never to fire (plain compressed attention),
        // and top-k 2 so per-row sparse selection engages mid-run for every row.
        for index_topk in [64i32, 2] {
        let dir = std::env::temp_dir().join(format!(
            "hi-mlx-v4-ragged-eq-{index_topk}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{
              "architectures": ["DeepseekV4ForCausalLM"],
              "model_type": "deepseek_v4",
              "hidden_size": 4,
              "intermediate_size": 8,
              "moe_intermediate_size": 4,
              "num_hidden_layers": 2,
              "num_attention_heads": 1,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "qk_rope_head_dim": 2,
              "q_lora_rank": 4,
              "index_head_dim": 2,
              "index_n_heads": 1,
              "index_topk": 64,
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
              "compress_ratios": [0, 4],
              "compress_rope_theta": 160000,
              "sliding_window": 6,
              "vocab_size": 16,
              "max_position_embeddings": 64,
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000,
              "tie_word_embeddings": false,
              "eos_token_id": 99
            }"#
            .replace(
                "\"index_topk\": 64",
                &format!("\"index_topk\": {index_topk}"),
            ),
        )
        .unwrap();
        {
            use tokenizers::Tokenizer;
            use tokenizers::models::wordlevel::WordLevel;
            let vocab: Map<String, u32> =
                (0..16u32).map(|i| (format!("t{i}"), i)).collect();
            let model = WordLevel::builder()
                .vocab(vocab)
                .unk_token("t0".to_string())
                .build()
                .unwrap();
            Tokenizer::new(model)
                .save(dir.join("tokenizer.json"), false)
                .unwrap();
        }
        {
            // Deterministic non-trivial weights: consecutive sin values with a running phase,
            // scaled small enough to keep the two-layer fixture numerically tame.
            let mut phase = 0usize;
            let mut w = |shape: &[i32]| {
                let len = shape.iter().product::<i32>() as usize;
                let vals: Vec<f32> = (0..len)
                    .map(|i| (((phase + i) as f32) * 0.37).sin() * 0.25)
                    .collect();
                phase += len;
                Array::from_slice(&vals, shape)
            };
            let ones = |len: usize| Array::from_slice(&vec![1.0f32; len], &[len as i32]);
            let mut arrays = Map::new();
            arrays.insert("model.embed_tokens.weight".to_string(), w(&[16, 4]));
            arrays.insert("lm_head.weight".to_string(), w(&[16, 4]));
            arrays.insert("model.hc_head.fn".to_string(), w(&[1, 4]));
            arrays.insert("model.hc_head.base".to_string(), w(&[1]));
            arrays.insert("model.hc_head.scale".to_string(), w(&[1]));
            arrays.insert("model.norm.weight".to_string(), ones(4));
            for layer in 0..2 {
                let prefix = format!("model.layers.{layer}");
                let attn = format!("{prefix}.attn");
                arrays.insert(format!("{prefix}.attn_norm.weight"), ones(4));
                arrays.insert(format!("{attn}.wq_a.weight"), w(&[4, 4]));
                arrays.insert(format!("{attn}.q_norm.weight"), ones(4));
                arrays.insert(format!("{attn}.wq_b.weight"), w(&[4, 4]));
                arrays.insert(format!("{attn}.wkv.weight"), w(&[4, 4]));
                arrays.insert(format!("{attn}.kv_norm.weight"), ones(4));
                arrays.insert(format!("{attn}.attn_sink"), w(&[1]));
                arrays.insert(format!("{attn}.wo_a.weight"), w(&[4, 4]));
                arrays.insert(format!("{attn}.wo_b.weight"), w(&[4, 4]));
                if layer == 1 {
                    arrays.insert(format!("{attn}.compressor.ape"), w(&[4, 8]));
                    arrays.insert(format!("{attn}.compressor.norm.weight"), ones(4));
                    arrays.insert(format!("{attn}.compressor.wgate.weight"), w(&[8, 4]));
                    arrays.insert(format!("{attn}.compressor.wkv.weight"), w(&[8, 4]));
                    arrays.insert(format!("{attn}.indexer.compressor.ape"), w(&[4, 4]));
                    arrays.insert(format!("{attn}.indexer.compressor.norm.weight"), ones(2));
                    arrays.insert(format!("{attn}.indexer.compressor.wgate.weight"), w(&[4, 4]));
                    arrays.insert(format!("{attn}.indexer.compressor.wkv.weight"), w(&[4, 4]));
                    arrays.insert(format!("{attn}.indexer.wq_b.weight"), w(&[2, 4]));
                    arrays.insert(format!("{attn}.indexer.weights_proj.weight"), w(&[1, 4]));
                }
                arrays.insert(format!("{prefix}.attn_hc.fn"), w(&[3, 4]));
                arrays.insert(format!("{prefix}.attn_hc.base"), w(&[3]));
                arrays.insert(format!("{prefix}.attn_hc.scale"), w(&[3]));
                arrays.insert(format!("{prefix}.ffn_norm.weight"), ones(4));
                arrays.insert(format!("{prefix}.ffn.gate.weight"), w(&[2, 4]));
                for name in ["gate_proj", "up_proj", "down_proj"] {
                    arrays.insert(format!("{prefix}.ffn.switch_mlp.{name}.weight"), w(&[2, 4, 4]));
                }
                arrays.insert(format!("{prefix}.ffn_hc.fn"), w(&[3, 4]));
                arrays.insert(format!("{prefix}.ffn_hc.base"), w(&[3]));
                arrays.insert(format!("{prefix}.ffn_hc.scale"), w(&[3]));
            }
            Array::save_safetensors(&arrays, None, dir.join("model.safetensors")).unwrap();
        }

        let mut rt = NativeRuntime::from_path(&dir).expect("load fixture");
        // Ragged lengths: 3 (shorter than the window), 7 (crosses one block boundary, longer
        // than the window), 5. Decode 12 steps so the window slides and blocks form mid-run.
        let prompts: Vec<Vec<u32>> = vec![
            vec![1, 5, 9],
            vec![2, 6, 10, 3, 7, 11, 4],
            vec![8, 12, 1, 13, 2],
        ];
        const STEPS: usize = 12;

        // Per-row single-sequence greedy references.
        let mut refs: Vec<(Vec<Vec<f32>>, Vec<u32>)> = Vec::new();
        for prompt in &prompts {
            let m = rt.model_for_test();
            m.reset_cache();
            let mut lg = m.forward(prompt).unwrap();
            let (mut all, mut fed) = (Vec::new(), Vec::new());
            for _ in 0..STEPS {
                let v = last_row_vec(&lg, 0);
                let t = v
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32;
                all.push(v);
                fed.push(t);
                lg = m.forward(&[t]).unwrap();
            }
            refs.push((all, fed));
        }

        // Ragged batched: per-row prefill + stacked decode, each row fed its own reference.
        let m = rt.model_for_test();
        m.reset_cache();
        let mut lg = m
            .prefill_batch_ragged(&prompts, STEPS as i32 + 4)
            .unwrap();
        let b = prompts.len();
        let mut first_bad: Option<(usize, usize, f32)> = None;
        for step in 0..STEPS {
            let mut next = Vec::with_capacity(b);
            for row in 0..b {
                let v = last_row_vec(&lg, row as i32);
                let refv = &refs[row].0[step];
                let d = v
                    .iter()
                    .zip(refv)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                if d > 1e-3 && first_bad.is_none() {
                    first_bad = Some((row, step, d));
                }
                next.push(refs[row].1[step]);
            }
            lg = m
                .forward_batch(&Array::from_slice(&next, &[b as i32, 1]))
                .unwrap();
        }
        m.reset_cache();
        let _ = std::fs::remove_dir_all(&dir);
        if let Some((row, step, d)) = first_bad {
            panic!("row {row} diverges from its single-sequence reference at step {step} (max diff {d})");
        }
        println!(
            "  all {b} ragged rows match their references across {STEPS} steps (index_topk {index_topk})"
        );
        }
    }

    /// Prefill logits for a padded row, swept across pad widths.
    ///
    /// The large-padding case already passes, and the failing generation path pads by exactly 1 —
    /// so sweep the width and find where `pad_attention_bias` stops reproducing the unpadded
    /// forward. A boundary error in the mask would show up at small widths and vanish at large
    /// ones, which is precisely the pattern the earlier diagnostics missed.
    #[test]
    #[ignore = "requires HI_MLX_BATCH_TEST_MODEL"]
    fn pad_width_sweep_logits_match_single() {
        let path = std::env::var_os("HI_MLX_BATCH_TEST_MODEL").expect("set HI_MLX_BATCH_TEST_MODEL");
        let mut rt = NativeRuntime::from_path(&path).expect("load");
        let ids = rt.tokenizer_for_test().encode("fn add(a: i64, b: i64) -> i64 {").unwrap();
        // Pre-encode distinct filler rows now, while the tokenizer is still borrowable.
        let filler_src: Vec<Vec<u32>> = (1..8)
            .map(|k| {
                let t = format!("// filler row {k} with distinct content\nfn f{k}(x: u64) -> u64 {{");
                rt.tokenizer_for_test().encode(&t).unwrap()
            })
            .collect();

        // reference: unpadded, single sequence
        let single = {
            let m = rt.model_for_test();
            m.reset_cache();
            m.stage_pad_lens(None);
            m.prepare_cache(ids.len() as i32 + 8);
            let lg = m.forward(&ids).unwrap();
            last_row_vec(&lg, 0)
        };
        let am = |v: &Vec<f32>| v.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0;
        println!("  pad slack   b   max|diff|   argmax(single={})  argmax(batched)", am(&single));

        let mut failures = Vec::new();
        // Sweep batch size too: the failing generation path runs b=4 while every passing
        // hand-built case so far used b=2.
        for (pad, slack, b) in [(1usize, 8i32, 2usize), (1, 40, 2), (1, 40, 3),
                                (1, 40, 4), (1, 40, 8), (3, 40, 4)] {
            let m = rt.model_for_test();
            let width = ids.len() + pad;
            // row 0 padded by `pad`; the rest are full-width filler.
            // Pad the SUBJECT row at index `subj`, deliberately NOT row 0: every earlier sweep
            // padded row 0 only, and row 0's slice starts at offset 0 under either a correct or
            // an incorrect per-row stride — so a row-indexing bug in the bias was invisible.
            // Pad TWO rows, interleaved, exactly as the failing generation path does
            // (pad_lens = [0, 1, 0, 1]). Every earlier sweep padded a single row, which is the
            // one structural difference left between the passing tests and the failing call.
            let subj = if b > 1 { 1usize } else { 0 };
            let second = if b > 3 { 3usize } else { subj };
            let mut pads = vec![0i32; b];
            pads[subj] = pad as i32;
            pads[second] = pad as i32;
            let mut flat = Vec::new();
            for k in 0..b {
                if k == subj || k == second {
                    flat.extend(std::iter::repeat_n(0u32, pad));
                    flat.extend_from_slice(&ids);
                } else {
                    let mut f = filler_src[k % filler_src.len()].clone();
                    f.resize(width, ids[0]);
                    flat.extend_from_slice(&f);
                    debug_assert_eq!(f.len(), width);
                }
            }
            m.reset_cache();
            m.stage_pad_lens(None);
            m.prepare_cache(width as i32 + slack);
            m.stage_pad_lens(Some(&pads));
            let lg = m.forward_batch(&Array::from_slice(&flat, &[b as i32, width as i32])).unwrap();
            m.stage_pad_lens(None);
            let got = last_row_vec(&lg, subj as i32);
            let d = single.iter().zip(&got).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            println!("  {pad:3} {slack:5} {b:3}   {d:9.4}   {:>13}  {:>15}", am(&single), am(&got));
            if d > 0.5 { failures.push((pad, slack, b, d)); }
        }
        assert!(failures.is_empty(), "pad widths diverging from single-sequence: {failures:?}");
    }

    /// Replay the exact inputs captured from a failing `stream_generate_batch` call.
    ///
    /// Every hand-built reproduction of that call has been correct, so this loads the literal
    /// arrays instead: same ids, same pad_lens, same cache capacity. If this fails, diff these
    /// inputs against the sweep's to find the discrepancy. If it passes, the arguments are
    /// identical and the divergence is in the model's state at call time, not its inputs.
    ///
    ///   HI_MLX_BATCH_DEBUG=1 ... each_client_receives_only_its_own_stream   # writes the dump
    ///   HI_MLX_BATCH_TEST_MODEL=... cargo test replay_captured_batch -- --ignored --nocapture
    #[test]
    #[ignore = "requires HI_MLX_BATCH_TEST_MODEL and a dump from a failing batch"]
    fn replay_captured_batch() {
        let path = std::env::var("HI_MLX_BATCH_DUMP")
            .unwrap_or_else(|_| "/tmp/batch_repro.txt".to_string());
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("no dump at {path}: {e} — run the failing test first"));
        let mut lines = text.lines();
        let hdr: Vec<i32> = lines.next().unwrap().split_whitespace()
            .map(|v| v.parse().unwrap()).collect();
        let (width, cap, b) = (hdr[0], hdr[1], hdr[2] as usize);
        let pad_lens: Vec<i32> = lines.next().unwrap().split(',')
            .map(|v| v.parse().unwrap()).collect();
        let flat: Vec<u32> = lines.next().unwrap().split(',')
            .map(|v| v.parse().unwrap()).collect();
        println!("  replaying: b={b} width={width} cap={cap} pad_lens={pad_lens:?}");

        let mpath = std::env::var_os("HI_MLX_BATCH_TEST_MODEL").expect("set HI_MLX_BATCH_TEST_MODEL");
        let mut rt = NativeRuntime::from_path(&mpath).expect("load");

        // single-sequence reference per row, from the row's unpadded ids
        let mut singles = Vec::new();
        for i in 0..b {
            let row = &flat[i * width as usize..(i + 1) * width as usize];
            let ids: Vec<u32> = row[pad_lens[i] as usize..].to_vec();
            let m = rt.model_for_test();
            m.reset_cache();
            m.stage_pad_lens(None);
            m.prepare_cache(ids.len() as i32 + 8);
            let lg = m.forward(&ids).unwrap();
            singles.push(crate::generate::mlx::greedy_next_token(&vec_to_logits(&last_row_vec(&lg, 0))).unwrap());
        }

        let m = rt.model_for_test();
        m.reset_cache();
        m.stage_pad_lens(None);
        m.prepare_cache(cap);
        m.stage_pad_lens(Some(&pad_lens));
        let lg = m.forward_batch(&Array::from_slice(&flat, &[b as i32, width])).unwrap();
        m.stage_pad_lens(None);

        // --- bisect: rerun the same rows in smaller sub-batches ---
        // If a row is correct alone or in a pair but wrong in the full batch, the trigger is
        // batch composition; if it is wrong even alone, the trigger is that row's own ids.
        for subset in [1usize, 2] {
            if subset >= b { continue; }
            let sub_flat: Vec<u32> = flat[..subset * width as usize].to_vec();
            let sub_pads: Vec<i32> = pad_lens[..subset].to_vec();
            let m = rt.model_for_test();
            m.reset_cache();
            m.stage_pad_lens(None);
            m.prepare_cache(cap);
            m.stage_pad_lens(Some(&sub_pads));
            let sl = m.forward_batch(&Array::from_slice(&sub_flat, &[subset as i32, width])).unwrap();
            m.stage_pad_lens(None);
            let got = crate::generate::mlx::greedy_next_token(
                &vec_to_logits(&last_row_vec(&sl, 0))).unwrap();
            println!("  [bisect] b={subset} row0 pad={} got={got:?} single={:?} {}",
                     sub_pads[0], singles[0],
                     if got == singles[0] { "MATCH" } else { "MISMATCH" });
        }

        let mut bad = Vec::new();
        for i in 0..b {
            let got = crate::generate::mlx::greedy_next_token(&vec_to_logits(&last_row_vec(&lg, i as i32))).unwrap();
            let ok = got == singles[i];
            println!("  row {i}: pad={} replay={got:?} single={:?} {}",
                     pad_lens[i], singles[i], if ok { "MATCH" } else { "*** MISMATCH ***" });
            if !ok { bad.push(i); }
        }
        assert!(bad.is_empty(), "replayed inputs still diverge on rows {bad:?}");
    }

    // Wrap a flat logits row back into the [1,1,vocab] shape the samplers expect.
    fn vec_to_logits(v: &[f32]) -> Array {
        Array::from_slice(v, &[1, 1, v.len() as i32])
    }

    fn last_row_vec(logits: &Array, row: i32) -> Vec<f32> {
        let shape = logits.shape();
        let (seq, vocab) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        let v = logits
            .index((row, seq - 1, ..))
            .reshape(&[vocab])
            .unwrap()
            .as_type::<f32>()
            .unwrap();
        transforms::eval([&v]).unwrap();
        v.as_slice::<f32>().to_vec()
    }
}
